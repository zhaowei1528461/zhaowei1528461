"""Independent stage 1 SUMO trajectory data, using NumPy and the stdlib only.

Vehicles use world meters, m/s, radians and the fields in VEHICLE_FIELDS.
NaN means unavailable; masks mean all eight fields are finite. Raw 31-D
observations remain separate and are never decoded into vehicle trajectories.
Recorded incoming actions are shifted to the interval [timestamp, next timestamp].
The original frames, including explicitly marked non-advancing terminal frames,
are retained for audit. Nothing in this module changes the training replay schema.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1
VEHICLE_FIELDS = ("x", "y", "vx", "vy", "yaw", "a", "length", "width")
EVENT_DEFAULTS = {
    "near_miss_ttc": 0.5, "low_ttc": 2.0,
    "low_progress_speed": 0.5, "low_progress_duration": 5.0,
}
EVENT_NAMES = ("collision", "near_miss", "low_ttc", "low_progress")
SOURCE_METADATA_FIELDS = ("prediction_source", "state_alignment", "position_reference",
                          "ttc_method", "ttc_parameters", "ttc_limitations", "event_verification")


@dataclass
class ScenarioEpisode:
    episode_id: str
    scenario_id: str
    parent_scenario_id: Optional[str]
    map_id: Optional[str]
    seed: Optional[int]
    source_type: Optional[str]
    source_path: Optional[str]
    source_cfg: Optional[str]
    coordinate_frame: str
    timestamps: np.ndarray
    dt: Optional[float]
    ego_state: np.ndarray
    ego_valid_mask: np.ndarray
    npc_ids: Tuple[str, ...]
    npc_states: np.ndarray
    npc_valid_mask: np.ndarray
    ego_actions: np.ndarray
    ego_action_mask: np.ndarray
    raw_states: np.ndarray
    raw_state_mask: np.ndarray
    events: Dict[str, np.ndarray]
    map_features: Any
    npc_actions: Any
    metadata: Dict[str, Any]
    raw_frames: List[Dict[str, Any]]

    @property
    def timestamp(self):
        return self.timestamps


@dataclass
class ScenarioWindow:
    scenario_id: str
    episode_id: str
    parent_scenario_id: Optional[str]
    map_id: Optional[str]
    seed: Optional[int]
    dt: float
    coordinate_frame: str
    timestamps: np.ndarray
    anchor_timestamp: float
    history_ego: np.ndarray
    future_ego: np.ndarray
    history_npcs: np.ndarray
    future_npcs: np.ndarray
    ego_mask: np.ndarray
    npc_mask: np.ndarray
    npc_ids: Tuple[str, ...]
    ego_actions: np.ndarray
    ego_action_mask: np.ndarray
    npc_actions: Any
    map_features: Any
    frenet_ego: Any
    frenet_npcs: Any
    event_labels: Tuple[str, ...]
    source_type: Optional[str]
    source_path: Optional[str]
    normalizer_id: Optional[str]
    validity_flags: Dict[str, Any]
    metadata: Dict[str, Any]


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_text(value):
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def _source_metadata(metadata):
    """Fill documented legacy collector semantics only with both source markers.

    Existing values, including explicit None, take precedence. This changes no
    stored state, position, TTC or recorded event, and claims no physical review.
    """
    result = copy.deepcopy(metadata or {})
    if (result.get("prediction_source") == "sumo_execution" and
            result.get("state_alignment") == "legacy_observation; terminal_state_may_precede_snapshot"):
        defaults = {"position_reference": "front_bumper_center",
                    "ttc_method": "legacy_longitudinal_gap_ttc",
                    "ttc_parameters": {"d_safe": 4.36, "lateral_gate": 2.0, "negative_clipped": True},
                    "ttc_limitations": "Only forward vehicles with closing longitudinal motion are considered; road geometry and future lateral motion are not constrained.",
                    "event_verification": "recorded_labels_only; physical_near_miss_not_verified"}
        for name, value in defaults.items():
            result.setdefault(name, value)
    return result


def _time_axis(frames):
    times = np.array([_number(frame.get("sim_time")) for frame in frames], dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("sim_time is missing or non-finite; real timestamps are required")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("sim_time must be strictly increasing; duplicate or decreasing timestamps")
    return times


def _vehicle_features(snapshot, missing):
    pos = snapshot.get("pos")
    if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 2:
        x, y = map(_number, pos)
    else:
        x, y = _number(snapshot.get("x")), _number(snapshot.get("y"))
    yaw = _number(snapshot.get("yaw"))  # Never reinterpret the SUMO degree angle.
    vx, vy = _number(snapshot.get("vx")), _number(snapshot.get("vy"))
    speed = _number(snapshot.get("v"))
    if not np.isfinite(vx) and np.isfinite(speed) and np.isfinite(yaw):
        vx = speed * math.cos(yaw)
    if not np.isfinite(vy) and np.isfinite(speed) and np.isfinite(yaw):
        vy = speed * math.sin(yaw)
    values = np.array([x, y, vx, vy, yaw, _number(snapshot.get("a")),
                       _number(snapshot.get("length")), _number(snapshot.get("width"))], dtype=np.float64)
    for index, name in enumerate(VEHICLE_FIELDS):
        if not np.isfinite(values[index]):
            missing[name] += 1
    valid = bool(np.isfinite(values).all() and values[6] > 0 and values[7] > 0)
    return values, valid


def _event_config(config=None):
    result = dict(EVENT_DEFAULTS)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("Unknown event thresholds: " + ", ".join(sorted(unknown)))
        result.update(config)
    for name, value in result.items():
        if not np.isfinite(_number(value)) or float(value) < 0:
            raise ValueError("Event threshold must be finite and nonnegative: " + name)
        result[name] = float(value)
    return result


def _derive_events(frames, timestamps, ego, ego_mask, config=None):
    cfg = _event_config(config)
    ttc = np.array([_number(frame.get("ttc")) for frame in frames])
    cost = np.array([_number(frame.get("cost")) for frame in frames])
    collision = np.array([bool(frame.get("collision", False)) or
                          frame.get("terminal_reason") == "collision" for frame in frames])
    near = np.isfinite(ttc) & (ttc >= 0) & (ttc <= cfg["near_miss_ttc"]) & ~collision
    low_ttc = np.isfinite(ttc) & (ttc >= 0) & (ttc <= cfg["low_ttc"])
    low_progress = np.zeros(len(frames), dtype=bool)
    speed = np.linalg.norm(ego[:, 2:4], axis=-1)
    start = None
    for i in range(len(frames)):
        if ego_mask[i] and speed[i] <= cfg["low_progress_speed"]:
            if start is None:
                start = i
            if timestamps[i] - timestamps[start] >= cfg["low_progress_duration"]:
                low_progress[start:i + 1] = True
        else:
            start = None
    return {"collision": collision, "near_miss": near, "low_ttc": low_ttc,
            "low_progress": low_progress, "ttc": ttc, "cost": cost}


def episode_from_record(record, ego_id="Auto", source_path=None, episode_id=None):
    """Adapt a recorded full replay, never a sweep summary or 31-D state alone."""
    if not isinstance(record, Mapping):
        raise ValueError("Episode record must be a mapping")
    trajectory = record.get("replay_trajectory") or {}
    raw_frames = trajectory.get("frames") or record.get("frames")
    if not isinstance(raw_frames, (list, tuple)) or not raw_frames:
        raise ValueError("Missing trajectory frames: summary/sweep records cannot form a ScenarioEpisode")
    if any(not isinstance(frame, Mapping) for frame in raw_frames):
        raise ValueError("Every trajectory frame must be a mapping")
    raw_frames = copy.deepcopy(list(raw_frames))
    metadata = _source_metadata(record.get("metadata"))
    frames, kept, collapsed = [], [], []
    for i, frame in enumerate(raw_frames):
        timestamp = _number(frame.get("sim_time"))
        if (frames and i == len(raw_frames) - 1 and frame.get("simulation_advanced") is False
                and frame.get("terminal_reason") and np.isfinite(timestamp)
                and timestamp == _number(frames[-1].get("sim_time"))):
            # Keep the preceding incoming action: this final action never ran.
            frames[-1] = dict(frames[-1], terminal_reason=frame["terminal_reason"])
            collapsed.append(i)
            continue
        frames.append(frame)
        kept.append(i)
    timestamps = _time_axis(frames)
    ntime = len(frames)
    npc_ids = tuple(sorted({str(vehicle["id"]) for frame in frames
                            for vehicle in (frame.get("vehicles") or [])
                            if isinstance(vehicle, Mapping) and vehicle.get("id") is not None
                            and str(vehicle["id"]) != str(ego_id)}))
    npc_indices = {vid: i for i, vid in enumerate(npc_ids)}
    ego = np.full((ntime, 8), np.nan)
    npcs = np.full((ntime, len(npc_ids), 8), np.nan)
    ego_mask = np.zeros(ntime, dtype=bool)
    npc_mask = np.zeros((ntime, len(npc_ids)), dtype=bool)
    raw_states = np.full((ntime, 31), np.nan)
    raw_state_mask = np.zeros(ntime, dtype=bool)
    missing = Counter()
    snapshots = 0
    for i, frame in enumerate(frames):
        seen = set()
        for snapshot in frame.get("vehicles") or []:
            if not isinstance(snapshot, Mapping) or snapshot.get("id") is None:
                raise ValueError("Vehicle snapshot needs an explicit id")
            vid = str(snapshot["id"])
            if vid in seen:
                raise ValueError("Duplicate vehicle id in frame: " + vid)
            seen.add(vid)
            snapshots += 1
            values, valid = _vehicle_features(snapshot, missing)
            if vid == str(ego_id):
                ego[i], ego_mask[i] = values, valid
            else:
                npcs[i, npc_indices[vid]], npc_mask[i, npc_indices[vid]] = values, valid
        try:
            state = np.asarray(frame.get("state", []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            state = np.array([])
        if state.shape == (31,):
            raw_states[i] = state
            raw_state_mask[i] = bool(np.isfinite(state).all())
    if snapshots == 0:
        raise ValueError("Missing vehicle snapshots in all trajectory frames; raw 31-D state cannot supply a vehicle trajectory")
    action_dim = None
    incoming = []
    for frame in frames:
        raw_action = frame.get("action")
        if raw_action is None:
            incoming.append(None)
            continue
        try:
            action = np.asarray(raw_action, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed ego action") from exc
        if not action.size:
            incoming.append(None)
            continue
        if action_dim is not None and action_dim != action.size:
            raise ValueError("Ego action dimensionality changes across frames")
        action_dim = action.size
        incoming.append(action)
    actions = np.full((ntime, action_dim or 1), np.nan)
    action_mask = np.zeros(ntime, dtype=bool)
    for i in range(ntime - 1):
        if incoming[i + 1] is not None:
            actions[i] = incoming[i + 1]
            action_mask[i] = bool(np.isfinite(actions[i]).all() and ego_mask[i] and ego_mask[i + 1])
    delta = np.diff(timestamps)
    dt = float(delta[0]) if delta.size and np.allclose(delta, delta[0], rtol=1e-6, atol=1e-8) else None
    source_cfg = record.get("source_cfg", trajectory.get("source_cfg"))
    identity = {"source_path": str(source_path) if source_path is not None else None,
                "source_cfg": source_cfg, "frames": raw_frames}
    stable = hashlib.sha256(_json_text(identity).encode("utf-8")).hexdigest()[:20]
    scenario_id = str(record.get("scenario_id") or record.get("parent_scenario_id") or
                      ("scenario-" + hashlib.sha256(str(source_cfg or source_path or stable).encode("utf-8")).hexdigest()[:20]))
    episode_id = str(episode_id or record.get("episode_id") or ("episode-" + stable))
    metadata.update({"schema_version": SCHEMA_VERSION, "vehicle_fields": list(VEHICLE_FIELDS),
                     "units": {"position": "m", "velocity": "m/s", "yaw": "rad", "a": "m/s^2"},
                     "ego_id": str(ego_id), "action_alignment": "interval_start",
                     "raw_action_alignment": "transition_ending_at_frame",
                     "missing_vehicle_fields": dict(missing), "vehicle_snapshot_count": snapshots,
                     "canonical_frame_indices": kept, "collapsed_terminal_frames": len(collapsed),
                     "collapsed_terminal_frame_indices": collapsed,
                     "event_config": _event_config(),
                     "event_methods": {"collision": "recorded collision flag or terminal_reason",
                                       "near_miss": "recorded nonnegative TTC threshold, excluding collision",
                                       "low_ttc": "recorded nonnegative TTC threshold",
                                       "low_progress": "observed speed below threshold for duration"}})
    metadata.setdefault("model_version", record.get("model_version"))
    metadata.setdefault("execution_valid", None)
    return ScenarioEpisode(episode_id, scenario_id, record.get("parent_scenario_id"),
                           record.get("map_id"), record.get("seed"), record.get("source_type"),
                           str(source_path) if source_path is not None else None, source_cfg,
                           "world", timestamps, dt, ego, ego_mask, npc_ids, npcs, npc_mask,
                           actions, action_mask, raw_states, raw_state_mask,
                           _derive_events(frames, timestamps, ego, ego_mask),
                           copy.deepcopy(record.get("map_features")), None, metadata, raw_frames)


def _safe_episode_id(value):
    # The same filename is safe on POSIX and Windows, including reserved devices.
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value)
            or value.endswith((".", " "))
            or value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *["COM" + str(i) for i in range(1, 10)], *["LPT" + str(i) for i in range(1, 10)]}):
        raise ValueError("episode_id must be a safe filename without paths or Windows device names")
    return value


def write_episode(episode, directory):
    """Save arrays and Unicode JSON in a single pickle-free compressed NPZ."""
    filename = _safe_episode_id(episode.episode_id) + ".npz"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    header = {name: getattr(episode, name) for name in
              ("episode_id", "scenario_id", "parent_scenario_id", "map_id", "seed", "source_type",
               "source_path", "source_cfg", "coordinate_frame", "dt", "npc_ids", "metadata", "map_features")}
    header["schema_version"] = SCHEMA_VERSION
    header["npc_actions"] = None  # No NPC control is exposed by the current SUMO snapshot.
    arrays = {name: getattr(episode, name) for name in
              ("timestamps", "ego_state", "ego_valid_mask", "npc_states", "npc_valid_mask",
               "ego_actions", "ego_action_mask", "raw_states", "raw_state_mask")}
    arrays.update({"event_" + name: values for name, values in episode.events.items()})
    arrays["metadata_json"] = np.asarray(_json_text(header))
    arrays["raw_frames_json"] = np.asarray(_json_text(episode.raw_frames))
    np.savez_compressed(str(path), **arrays)
    return path


def load_episode(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Scenario episode does not exist: " + str(path))
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            header = json.loads(str(archive["metadata_json"].item()))
            if header.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported ScenarioEpisode schema version")
            arrays = {name: archive[name].copy() for name in
                      ("timestamps", "ego_state", "ego_valid_mask", "npc_states", "npc_valid_mask",
                       "ego_actions", "ego_action_mask", "raw_states", "raw_state_mask")}
            events = {name[6:]: archive[name].copy() for name in archive.files if name.startswith("event_")}
            raw_frames = json.loads(str(archive["raw_frames_json"].item()))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Not a supported ScenarioEpisode NPZ: " + str(path)) from exc
    header["metadata"] = _source_metadata(header.get("metadata"))
    timestamps = arrays["timestamps"]
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all() or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0)):
        raise ValueError("ScenarioEpisode sim_time must be finite and strictly increasing")
    ntime, nnpc = len(timestamps), len(header["npc_ids"])
    required_shapes = {"ego_state": (ntime, 8), "ego_valid_mask": (ntime,),
                       "npc_states": (ntime, nnpc, 8), "npc_valid_mask": (ntime, nnpc),
                       "ego_action_mask": (ntime,), "raw_states": (ntime, 31), "raw_state_mask": (ntime,)}
    for name, shape in required_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError("Invalid ScenarioEpisode array shape: " + name)
    if arrays["ego_actions"].ndim != 2 or arrays["ego_actions"].shape[0] != ntime:
        raise ValueError("Invalid ScenarioEpisode ego_actions shape")
    return ScenarioEpisode(events=events, raw_frames=raw_frames, npc_actions=None,
                           **{name: header[name] for name in
                              ("episode_id", "scenario_id", "parent_scenario_id", "map_id", "seed", "source_type",
                               "source_path", "source_cfg", "coordinate_frame", "dt", "map_features", "metadata")},
                           npc_ids=tuple(header["npc_ids"]), **arrays)


def _unwrap_yaw(states, mask):
    output = states.copy()
    flattened = output.reshape((len(output), -1, 8))
    masks = mask.reshape((len(output), -1))
    for vehicle in range(flattened.shape[1]):
        indices = np.flatnonzero(masks[:, vehicle])
        if not indices.size:
            continue
        chunks = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        for chunk in chunks:
            flattened[chunk, vehicle, 4] = np.unwrap(flattened[chunk, vehicle, 4])
    return output


def _resample_states(source_times, values, mask, target_times, max_gap):
    values = _unwrap_yaw(values, mask)
    output = np.full((len(target_times),) + values.shape[1:], np.nan)
    valid = np.zeros((len(target_times),) + mask.shape[1:], dtype=bool)
    for i, timestamp in enumerate(target_times):
        right = int(np.searchsorted(source_times, timestamp, side="left"))
        exact = None
        if right < len(source_times) and abs(source_times[right] - timestamp) <= 1e-8:
            exact = right
        elif right and abs(source_times[right - 1] - timestamp) <= 1e-8:
            exact = right - 1
        if exact is not None:
            output[i], valid[i] = values[exact], mask[exact]
        elif right and right < len(source_times):
            left = right - 1
            gap = source_times[right] - source_times[left]
            if gap <= max_gap + 1e-8:
                valid[i] = mask[left] & mask[right]
                alpha = (timestamp - source_times[left]) / gap
                output[i] = values[left] + alpha * (values[right] - values[left])
        output[i] = np.where(np.expand_dims(valid[i], -1), output[i], np.nan)
    return output, valid


def to_ego_centric(states, anchor):
    """Rotate positions/velocities around one fixed world anchor [x,y,...,yaw]."""
    states = np.asarray(states, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if states.shape[-1] != 8 or anchor.shape != (8,):
        raise ValueError("Ego coordinate conversion requires eight-field states and one anchor")
    result = states.copy()
    if not np.isfinite(anchor[[0, 1, 4]]).all():
        return np.full_like(result, np.nan)
    c, s = math.cos(anchor[4]), math.sin(anchor[4])
    x, y = states[..., 0] - anchor[0], states[..., 1] - anchor[1]
    result[..., 0], result[..., 1] = c * x + s * y, -s * x + c * y
    vx, vy = states[..., 2], states[..., 3]
    result[..., 2], result[..., 3] = c * vx + s * vy, -s * vx + c * vy
    result[..., 4] = states[..., 4] - anchor[4]
    return result


def to_frenet(states, reference_line=None):
    """Project x/y onto supplied world polyline; return s,d only, or None/flags.

    Positive d is left of the nearest oriented segment. This geometric projection
    does not imply lane membership or road validity, and no line is inferred.
    """
    flags = {"frenet_available": False, "lane_membership_available": False}
    if reference_line is None:
        flags["frenet_reason"] = "missing_reference_line"
        return None, flags
    line = np.asarray(reference_line, dtype=float)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 or not np.isfinite(line).all():
        raise ValueError("reference_line must contain at least two finite world x/y points")
    delta = np.diff(line, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("reference_line contains duplicate consecutive points")
    states = np.asarray(states, dtype=float)
    if states.shape[-1] != 8:
        raise ValueError("Frenet input must have eight vehicle fields")
    output = np.full(states.shape[:-1] + (2,), np.nan)
    starts = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
    for index in np.ndindex(states.shape[:-1]):
        point = states[index][:2]
        if not np.isfinite(point).all():
            continue
        relative = point - line[:-1]
        alpha = np.clip(np.sum(relative * delta, axis=1) / lengths ** 2, 0, 1)
        projections = line[:-1] + alpha[:, None] * delta
        offset = point - projections
        segment = int(np.argmin(np.sum(offset ** 2, axis=1)))
        tangent = delta[segment] / lengths[segment]
        output[index] = [starts[segment] + alpha[segment] * lengths[segment],
                         tangent[0] * offset[segment, 1] - tangent[1] * offset[segment, 0]]
    flags["frenet_available"] = True
    flags["frenet_reason"] = "provided_reference_line"
    return output, flags


def make_windows(episode, history=11, future=30, dt=0.1, stride=1,
                 coordinate_frame="world", reference_line=None, event_config=None,
                 event_only=False, event_before=3.0, event_after=1.0, max_gap=None):
    """Resample real timestamps; history ends at anchor, future starts anchor+dt.

    No interpolation crosses a missing vehicle sample. Default maximum time gap
    is 1.5 times median source interval; callers may choose a stricter limit.
    A target action is valid only when its entire [t,t+dt] lies in one recorded
    control interval (or crosses intervals with identical recorded controls).
    """
    for name, value in (("history", history), ("future", future), ("stride", stride)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if name == "future" else 1):
            raise ValueError(name + " must be a positive integer (future may be zero)")
    if not np.isfinite(_number(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and positive")
    dt = float(dt)
    if coordinate_frame not in ("world", "ego-centric"):
        raise ValueError("Unknown coordinate_frame: " + str(coordinate_frame))
    if any(not np.isfinite(_number(v)) or float(v) < 0 for v in (event_before, event_after)):
        raise ValueError("Event window times must be finite and nonnegative")
    times = episode.timestamps
    if not len(times):
        return []
    if max_gap is None:
        max_gap = 1.5 * float(np.median(np.diff(times))) if len(times) > 1 else dt
    if not np.isfinite(_number(max_gap)) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    count = int(math.floor((times[-1] - times[0]) / dt + 1e-8)) + 1
    target = times[0] + np.arange(count, dtype=float) * dt
    total = history + future
    if count < total:
        return []
    ego, ego_mask = _resample_states(times, episode.ego_state, episode.ego_valid_mask, target, max_gap)
    npcs, npc_mask = _resample_states(times, episode.npc_states, episode.npc_valid_mask, target, max_gap)
    actions = np.full((count, episode.ego_actions.shape[1]), np.nan)
    action_mask = np.zeros(count, dtype=bool)
    for i, timestamp in enumerate(target[:-1]):
        left = int(np.searchsorted(times, timestamp + 1e-8, side="right")) - 1
        end = timestamp + dt
        last = int(np.searchsorted(times, end - 1e-8, side="right")) - 1
        if left < 0 or last >= len(times) - 1 or not ego_mask[i] or not ego_mask[i + 1]:
            continue
        intervals = slice(left, last + 1)
        if (np.all(episode.ego_action_mask[intervals]) and
                np.all(np.diff(times)[intervals] <= max_gap + 1e-8) and
                np.allclose(episode.ego_actions[intervals], episode.ego_actions[left], rtol=0, atol=1e-8)):
            actions[i], action_mask[i] = episode.ego_actions[left], True
    canonical = [episode.raw_frames[i] for i in episode.metadata.get("canonical_frame_indices", range(len(times)))]
    # A collapsed frame contributes only its terminal label, never its action.
    if episode.metadata.get("collapsed_terminal_frames"):
        canonical[-1] = dict(canonical[-1], terminal_reason=episode.raw_frames[-1].get("terminal_reason", ""))
    events = _derive_events(canonical, times, episode.ego_state, episode.ego_valid_mask, event_config)
    event_times = {name: times[events[name]] for name in EVENT_NAMES}
    all_events = np.concatenate(list(event_times.values()))
    windows = []
    for start in range(0, count - total + 1, stride):
        anchor_index = start + history - 1
        anchor_time = target[anchor_index]
        if event_only and not np.any((all_events >= anchor_time - event_after) &
                                     (all_events <= anchor_time + event_before)):
            continue
        span = slice(start, start + total)
        window_ego, window_npcs = ego[span].copy(), npcs[span].copy()
        emask, nmask = ego_mask[span].copy(), npc_mask[span].copy()
        labels = tuple(name for name in EVENT_NAMES if np.any((event_times[name] >= target[start] - 1e-8) &
                       (event_times[name] <= target[start + total - 1] + 1e-8)))
        frenet_ego, flags = to_frenet(window_ego, reference_line)
        frenet_npcs, _ = to_frenet(window_npcs, reference_line)
        flags["ego_anchor_available"] = bool(ego_mask[anchor_index])
        if coordinate_frame == "ego-centric":
            anchor = ego[anchor_index]
            window_ego, window_npcs = to_ego_centric(window_ego, anchor), to_ego_centric(window_npcs, anchor)
            if not ego_mask[anchor_index]:
                emask[:] = False
                nmask[:] = False
        windows.append(ScenarioWindow(episode.scenario_id, episode.episode_id, episode.parent_scenario_id,
                       episode.map_id, episode.seed, dt, coordinate_frame, target[span].copy(), float(anchor_time),
                       window_ego[:history], window_ego[history:], window_npcs[:history], window_npcs[history:],
                       emask, nmask, episode.npc_ids, actions[span].copy(), action_mask[span].copy(), None,
                       episode.map_features, frenet_ego, frenet_npcs, labels, episode.source_type,
                       episode.source_path, None, flags,
                       {"event_config": _event_config(event_config), "event_before": event_before,
                        "event_after": event_after, "max_interpolation_gap": float(max_gap),
                        "vehicle_fields": list(VEHICLE_FIELDS), "state_coordinate_frame": coordinate_frame,
                        "frenet_fields": ["s", "d"], "action_alignment": "interval_start",
                        **{name: copy.deepcopy(episode.metadata[name]) for name in SOURCE_METADATA_FIELDS
                           if name in episode.metadata}}))
    return windows


def split_episodes(episodes, ratios=(0.8, 0.1, 0.1), seed=0, group_by="scenario"):
    """Group whole scenario ancestry components, never split perturbed copies."""
    if len(ratios) != 3 or any(not np.isfinite(_number(r)) or r < 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("train/val/test ratios must be nonnegative and sum to one")
    if group_by not in ("scenario", "map"):
        raise ValueError("group_by must be scenario or map")
    parents = {}
    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            low, high = sorted((a, b))
            parents[high] = low
    for episode in episodes:
        find("scenario:" + episode.scenario_id)
        if episode.parent_scenario_id:
            union("scenario:" + episode.scenario_id, "scenario:" + str(episode.parent_scenario_id))
        if group_by == "map" and episode.map_id is not None:
            union("scenario:" + episode.scenario_id, "map:" + str(episode.map_id))
    grouped = {}
    for episode in episodes:
        grouped.setdefault(find("scenario:" + episode.scenario_id), []).append(episode)
    keys = sorted(grouped)
    np.random.RandomState(seed).shuffle(keys)
    desired = np.asarray(ratios) * len(keys)
    counts = np.floor(desired).astype(int)
    for index in np.argsort(-(desired - counts), kind="stable")[:len(keys) - sum(counts)]:
        counts[index] += 1
    result = {"train": [], "val": [], "test": []}
    offset = 0
    for name, count in zip(result, counts):
        for key in keys[offset:offset + count]:
            result[name].extend(sorted(grouped[key], key=lambda episode: episode.episode_id))
        offset += count
    return result


@dataclass
class ScenarioNormalizer:
    """Seven scalar statistics and circular yaw; inversion preserves direction.

    Yaw residuals lie in [-pi,pi]. Whole-turn counts are intentionally not encoded
    by normalization; original world yaw and unwrapped window yaw remain stored.
    """
    mean: np.ndarray
    std: np.ndarray
    count: int
    training_episode_ids: Tuple[str, ...]
    normalizer_id: str
    coordinate_frame: str = "world"

    def transform(self, states, mask=None):
        values = np.asarray(states, dtype=float)
        if values.shape[-1] != 8:
            raise ValueError("Normalizer requires eight vehicle fields")
        residual = values - self.mean
        residual[..., 4] = np.arctan2(np.sin(residual[..., 4]), np.cos(residual[..., 4]))
        result = residual / self.std
        if mask is not None:
            if np.shape(mask) != values.shape[:-1]:
                raise ValueError("Normalizer mask shape does not match states")
            result = np.where(np.asarray(mask)[..., None], result, np.nan)
        return result

    def inverse_transform(self, states, mask=None):
        values = np.asarray(states, dtype=float)
        if values.shape[-1] != 8:
            raise ValueError("Normalizer requires eight vehicle fields")
        result = values * self.std + self.mean
        result[..., 4] = np.arctan2(np.sin(result[..., 4]), np.cos(result[..., 4]))
        if mask is not None:
            if np.shape(mask) != values.shape[:-1]:
                raise ValueError("Normalizer mask shape does not match states")
            result = np.where(np.asarray(mask)[..., None], result, np.nan)
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_text({"schema_version": SCHEMA_VERSION,
                        "vehicle_fields": VEHICLE_FIELDS, "mean": self.mean, "std": self.std,
                        "count": self.count, "training_episode_ids": self.training_episode_ids,
                        "normalizer_id": self.normalizer_id, "coordinate_frame": self.coordinate_frame,
                        "yaw_mode": "circular_mean_wrapped_residual"}), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mean, std = np.asarray(payload["mean"], dtype=float), np.asarray(payload["std"], dtype=float)
        if (payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("yaw_mode") != "circular_mean_wrapped_residual"
                or mean.shape != (8,) or std.shape != (8,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
            raise ValueError("Invalid persisted ScenarioNormalizer")
        return cls(mean, std, int(payload["count"]), tuple(payload["training_episode_ids"]),
                   payload["normalizer_id"], payload["coordinate_frame"])


def fit_normalizer(splits):
    """Only the explicit train split may contribute statistics; pooled ego/NPC."""
    train = splits.get("train", [])
    if not train:
        raise ValueError("Cannot fit normalizer without train episodes")
    rows = []
    for episode in train:
        if episode.coordinate_frame != "world":
            raise ValueError("Training normalizer expects world coordinate episodes")
        rows.extend([episode.ego_state[episode.ego_valid_mask],
                     episode.npc_states[episode.npc_valid_mask]])
    values = np.concatenate(rows, axis=0)
    values = values[np.isfinite(values).all(axis=-1)]
    if not len(values):
        raise ValueError("No valid vehicle states in train split")
    mean, std = np.mean(values, axis=0), np.std(values, axis=0)
    mean[4] = math.atan2(float(np.mean(np.sin(values[:, 4]))), float(np.mean(np.cos(values[:, 4]))))
    yaw_residual = np.arctan2(np.sin(values[:, 4] - mean[4]), np.cos(values[:, 4] - mean[4]))
    std[4] = np.std(yaw_residual)
    std = np.where(std < 1e-8, 1.0, std)
    ids = tuple(sorted(episode.episode_id for episode in train))
    identity = _json_text({"mean": mean, "std": std, "count": len(values), "training_episode_ids": ids,
                          "yaw_mode": "circular_mean_wrapped_residual"})
    return ScenarioNormalizer(mean, std, len(values), ids, "normalizer-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20])


def pad_windows(windows):
    """Batch variable NPC counts; padded values are NaN and padded masks False."""
    if not windows:
        raise ValueError("Cannot pad an empty window batch")
    history, future = len(windows[0].history_ego), len(windows[0].future_ego)
    if any(len(window.history_ego) != history or len(window.future_ego) != future or
           not np.isclose(window.dt, windows[0].dt) or window.coordinate_frame != windows[0].coordinate_frame
           for window in windows):
        raise ValueError("Batch windows must share history/future, dt and coordinate_frame")
    nnpc = max(len(window.npc_ids) for window in windows)
    bsize = len(windows)
    output = {"history_ego": np.stack([w.history_ego for w in windows]),
              "future_ego": np.stack([w.future_ego for w in windows]),
              "history_npcs": np.full((bsize, history, nnpc, 8), np.nan),
              "future_npcs": np.full((bsize, future, nnpc, 8), np.nan),
              "npc_mask": np.zeros((bsize, history + future, nnpc), dtype=bool),
              "ego_mask": np.stack([w.ego_mask for w in windows]),
              "ego_actions": np.stack([w.ego_actions for w in windows]),
              "ego_action_mask": np.stack([w.ego_action_mask for w in windows]),
              "timestamps": np.stack([w.timestamps for w in windows]),
              "npc_ids": [w.npc_ids for w in windows], "scenario_ids": [w.scenario_id for w in windows]}
    for i, window in enumerate(windows):
        n = len(window.npc_ids)
        output["history_npcs"][i, :, :n] = window.history_npcs
        output["future_npcs"][i, :, :n] = window.future_npcs
        output["npc_mask"][i, :, :n] = window.npc_mask
    return output


def _input_paths(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Dataset input does not exist: " + str(path))
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in (".npz", ".jsonl", ".json"))
        if not files:
            raise ValueError("Dataset input directory contains no NPZ/JSONL/JSON files")
        return files
    if path.suffix.lower() not in (".npz", ".jsonl", ".json"):
        raise ValueError("Dataset input must be NPZ, JSONL or JSON")
    return [path]


def _iter_inputs(path):
    for input_path in _input_paths(path):
        if input_path.suffix.lower() == ".npz":
            yield str(input_path), input_path, None
        elif input_path.suffix.lower() == ".jsonl":
            with input_path.open(encoding="utf-8-sig") as handle:
                for number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            yield str(input_path) + ":" + str(number), json.loads(line), None
                        except json.JSONDecodeError as exc:
                            yield str(input_path) + ":" + str(number), None, str(exc)
        else:
            try:
                payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                yield str(input_path), None, str(exc)
                continue
            records = payload if isinstance(payload, list) else payload.get("records", [payload]) if isinstance(payload, dict) else [payload]
            for number, item in enumerate(records):
                yield str(input_path) + ":" + str(number), item, None


def _load_input(item, source, ego_id):
    return load_episode(item) if isinstance(item, Path) else episode_from_record(item, ego_id=ego_id, source_path=source)


def inspect_dataset(episodes, history=11, future=30, dt=0.1, split_seed=0):
    """Return JSON-ready diagnostics; rates refer to canonical observed samples."""
    report = {"episodes": len(episodes), "windows": 0, "event_counts": dict.fromkeys(EVENT_NAMES, 0),
              "missing_field_rates": {}, "mask_rates": {}, "time": {"strictly_increasing": True,
              "nonuniform_episodes": 0, "source_dt_values": []}, "shapes": [], "splits": {}}
    missing = Counter()
    snapshots = ego_total = ego_valid = npc_total = npc_valid = action_total = action_valid = raw_total = raw_valid = 0
    for episode in episodes:
        windows = make_windows(episode, history=history, future=future, dt=dt)
        report["windows"] += len(windows)
        for name in EVENT_NAMES:
            report["event_counts"][name] += int(np.count_nonzero(episode.events.get(name, [])))
        missing.update(episode.metadata.get("missing_vehicle_fields", {}))
        snapshots += episode.metadata.get("vehicle_snapshot_count", 0)
        ego_total += episode.ego_valid_mask.size
        ego_valid += int(episode.ego_valid_mask.sum())
        npc_total += episode.npc_valid_mask.size
        npc_valid += int(episode.npc_valid_mask.sum())
        action_total += episode.ego_action_mask.size
        action_valid += int(episode.ego_action_mask.sum())
        raw_total += episode.raw_state_mask.size
        raw_valid += int(episode.raw_state_mask.sum())
        if episode.dt is None and len(episode.timestamps) > 1:
            report["time"]["nonuniform_episodes"] += 1
        elif episode.dt is not None:
            report["time"]["source_dt_values"].append(episode.dt)
        report["shapes"].append({"episode_id": episode.episode_id, "ego_state": list(episode.ego_state.shape),
                                  "npc_states": list(episode.npc_states.shape), "raw_states": list(episode.raw_states.shape),
                                  "window_history_npcs": [history, len(episode.npc_ids), 8],
                                  "window_future_npcs": [future, len(episode.npc_ids), 8]})
    report["missing_field_rates"] = {name: missing[name] / snapshots if snapshots else None for name in VEHICLE_FIELDS}
    report["missing_field_rates"].update({"map_features": sum(e.map_features is None for e in episodes) / len(episodes) if episodes else None,
                                          "seed": sum(e.seed is None for e in episodes) / len(episodes) if episodes else None,
                                          "map_id": sum(e.map_id is None for e in episodes) / len(episodes) if episodes else None})
    report["mask_rates"] = {name: valid / total if total else None for name, valid, total in
                             (("ego_valid", ego_valid, ego_total), ("npc_valid", npc_valid, npc_total),
                              ("action_valid", action_valid, action_total), ("raw_state_valid", raw_valid, raw_total))}
    report["time"]["source_dt_values"] = sorted(set(round(value, 9) for value in report["time"]["source_dt_values"]))
    report["splits"] = {name: {"episodes": len(group), "scenario_ids": sorted({e.scenario_id for e in group}),
                               "episode_ids": [e.episode_id for e in group]}
                         for name, group in split_episodes(episodes, seed=split_seed).items()}
    report["event_config"] = _event_config()
    report["target_dt"] = dt
    return report


def prepare_dataset(episodes, directory, ratios=(0.8, 0.1, 0.1), seed=0,
                    group_by="scenario", input_sources=None):
    """Persist proposed ancestry-safe splits and a normalizer fit on train only.

    No trajectories are rewritten. Manifest input_source identifies the original
    NPZ or record line; normalizer training_episode_ids makes fitting auditable.
    """
    split = split_episodes(episodes, ratios=ratios, seed=seed, group_by=group_by)
    normalizer = fit_normalizer(split)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normalizer_path = normalizer.save(directory / "normalizer.json")
    input_sources = input_sources or {}
    manifest = {"artifact_type": "scenario_split_manifest", "schema_version": SCHEMA_VERSION,
                "seed": seed, "ratios": list(ratios), "group_by": group_by,
                "normalizer_id": normalizer.normalizer_id,
                "normalizer_path": str(normalizer_path), "splits": {}}
    for name, group in split.items():
        manifest["splits"][name] = [{"episode_id": episode.episode_id,
                                      "scenario_id": episode.scenario_id,
                                      "parent_scenario_id": episode.parent_scenario_id,
                                      "map_id": episode.map_id,
                                      "input_source": input_sources.get(episode.episode_id, episode.source_path)}
                                     for episode in group]
    manifest_path = directory / "split_manifest.json"
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "normalizer_path": str(normalizer_path),
            "normalizer_id": normalizer.normalizer_id,
            "splits": {name: len(group) for name, group in split.items()},
            "normalizer_training_states": normalizer.count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="Convert trajectory records to ScenarioEpisode NPZ")
    inspect = subparsers.add_parser("inspect_dataset", help="Inspect recorded episodes and proposed group splits")
    prepare = subparsers.add_parser("prepare_dataset", help="Persist scenario splits and train-only normalizer")
    for subparser in (convert, inspect, prepare):
        subparser.add_argument("--input", required=True)
        subparser.add_argument("--ego-id", default="Auto")
    convert.add_argument("--output", required=True)
    inspect.add_argument("--history", type=int, default=11)
    inspect.add_argument("--future", type=int, default=30)
    inspect.add_argument("--dt", type=float, default=0.1)
    inspect.add_argument("--split-seed", type=int, default=0)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split-seed", type=int, default=0)
    prepare.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    prepare.add_argument("--group-by", choices=("scenario", "map"), default="scenario")
    args = parser.parse_args(argv)
    episodes, errors, paths, input_sources = [], [], [], {}
    try:
        for source, item, error in _iter_inputs(args.input):
            try:
                if error:
                    raise ValueError(error)
                episode = _load_input(item, source, args.ego_id)
                if args.command == "convert":
                    paths.append(str(write_episode(episode, args.output)))
                episodes.append(episode)
                input_sources[episode.episode_id] = source
            except (ValueError, TypeError, KeyError, OSError) as exc:
                errors.append({"source": source, "error": str(exc)})
        if args.command == "convert":
            report = {"converted": len(paths), "rejected": len(errors), "output_paths": paths, "errors": errors}
        elif args.command == "prepare_dataset":
            report = prepare_dataset(episodes, args.output, ratios=args.ratios, seed=args.split_seed,
                                     group_by=args.group_by, input_sources=input_sources)
            report.update({"rejected": len(errors), "errors": errors})
        else:
            report = inspect_dataset(episodes, args.history, args.future, args.dt, args.split_seed)
            report.update({"rejected": len(errors), "errors": errors})
    except (ValueError, OSError) as exc:
        report = {"errors": [{"source": args.input, "error": str(exc)}], "rejected": 1}
        errors.append(report["errors"][0])
    print(_json_text(report))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
