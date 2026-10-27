"""Stage 2 footprint rules. No JAX, simulation, or training side effects.

TTC extrapolates constant world velocity and fixed heading until footprint
contact. It is not a simulator collision label or a calibrated probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

EPS = 1e-9
POSITION_REFERENCES = ('center', 'front_bumper_center')


@dataclass
class RiskConfig:
    ttc_horizon: float = 5.0
    near_ttc: float = 0.5
    near_distance: float = 2.0
    progress_speed: float = 0.5
    progress_duration: float = 5.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        'overlap': 0.4, 'ttc': 0.3, 'distance': 0.2, 'progress': 0.1})

    def __post_init__(self):
        for name in ('ttc_horizon', 'near_ttc', 'near_distance', 'progress_speed', 'progress_duration'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
            setattr(self, name, value)
        if not self.weights or set(self.weights) - {'overlap', 'ttc', 'distance', 'progress'}:
            raise ValueError('Unknown or empty risk weights')
        self.weights = {k: float(v) for k, v in self.weights.items()}
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError('Weights must be finite, nonnegative, and have positive sum')


@dataclass
class CandidateTrajectory:
    npc_states: np.ndarray  # [F,N,8], in the same frame and units as window.
    npc_mask: np.ndarray    # [F,N]
    npc_ids: Tuple[str, ...]
    dt: float
    coordinate_frame: str


@dataclass
class RiskReport:
    score: Optional[float]
    min_ttc: Optional[float]
    min_distance: Optional[float]
    predicted_collision_probability: Optional[float]
    progress_penalty: Optional[float]
    validity_flags: Dict[str, Any]
    uncertainty: Optional[float]
    model_version: Optional[str]
    prediction_source: str
    components: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _body(state, position_reference):
    state = np.asarray(state, dtype=float)
    if state.shape != (8,) or not np.isfinite(state).all() or np.any(state[6:8] <= 0):
        raise ValueError('Footprints need finite [x,y,vx,vy,yaw,a,length,width] and positive dimensions')
    if position_reference not in POSITION_REFERENCES:
        raise ValueError('Unknown vehicle position reference')
    forward = np.array([math.cos(state[4]), math.sin(state[4])])
    lateral = np.array([-forward[1], forward[0]])
    axes = np.array([forward, lateral])
    center = state[:2].copy()
    if position_reference == 'front_bumper_center':
        center -= 0.5 * state[6] * forward
    extent = state[6:8] / 2
    vertices = center + np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * extent @ axes
    return center, axes, extent, vertices


def _projection(body, axis):
    center, axes, extent, _ = body
    middle = float(center @ axis)
    radius = float(np.abs(axes @ axis) @ extent)
    return middle - radius, middle + radius


def _overlap(first, second):
    for axis in np.concatenate([first[1], second[1]]):
        lo1, hi1 = _projection(first, axis)
        lo2, hi2 = _projection(second, axis)
        if hi1 < lo2 - EPS or hi2 < lo1 - EPS:
            return False
    return True


def _vertex_edge_distance(vertices, polygon):
    starts = polygon
    edges = np.roll(polygon, -1, axis=0) - starts
    offsets = vertices[:, None, :] - starts[None, :, :]
    parameter = np.clip(np.sum(offsets * edges[None], axis=-1) / np.sum(edges * edges, axis=-1), 0., 1.)
    distances = np.linalg.norm(offsets - parameter[..., None] * edges[None], axis=-1)
    return float(distances.min())


def obb_distance(first_state, second_state, position_reference='center'):
    """Minimum Euclidean body clearance in meters; contact returns zero."""
    first, second = _body(first_state, position_reference), _body(second_state, position_reference)
    if _overlap(first, second):
        return 0.0
    return min(_vertex_edge_distance(first[3], second[3]), _vertex_edge_distance(second[3], first[3]))


def obb_ttc(first_state, second_state, position_reference='center', horizon=5.0):
    """Swept SAT intersection for translating rectangles; None means no contact within horizon."""
    if not math.isfinite(float(horizon)) or horizon <= 0:
        raise ValueError('TTC horizon must be finite and positive')
    first, second = _body(first_state, position_reference), _body(second_state, position_reference)
    relative_velocity = np.asarray(second_state)[2:4] - np.asarray(first_state)[2:4]
    entry, leave = 0.0, float(horizon)
    for axis in np.concatenate([first[1], second[1]]):
        lo1, hi1 = _projection(first, axis)
        lo2, hi2 = _projection(second, axis)
        speed = float(relative_velocity @ axis)
        if abs(speed) <= EPS:
            if hi1 < lo2 - EPS or hi2 < lo1 - EPS:
                return None
            continue
        bounds = ((lo1 - hi2) / speed, (hi1 - lo2) / speed)
        entry, leave = max(entry, min(bounds)), min(leave, max(bounds))
        if entry > leave + EPS:
            return None
    if leave < -EPS or entry > horizon + EPS:
        return None
    return float(max(0., min(entry, horizon)))


class RiskEvaluator:
    def __init__(self, config=None):
        self.config = config if isinstance(config, RiskConfig) else RiskConfig(**(config or {}))

    def evaluate_risk(self, window, candidate=None):
        ego = np.concatenate([window.history_ego, window.future_ego])
        npc = np.concatenate([window.history_npcs, window.future_npcs])
        h, f, n = len(window.history_ego), len(window.future_ego), len(window.npc_ids)
        times = np.asarray(window.timestamps, float)
        ego_mask, npc_mask = np.asarray(window.ego_mask, bool), np.asarray(window.npc_mask, bool)
        if (h < 1 or ego.shape != (h+f, 8) or npc.shape != (h+f, n, 8) or
                ego_mask.shape != (h+f,) or npc_mask.shape != (h+f, n) or times.shape != (h+f,)):
            raise ValueError('Window state/mask/timestamp shape mismatch')
        if (not math.isfinite(float(window.dt)) or window.dt <= 0 or not np.isfinite(times).all() or
                (len(times) > 1 and not np.allclose(np.diff(times), window.dt, atol=1e-7, rtol=1e-6))):
            raise ValueError('Window timestamps must match positive uniform dt')
        source = 'measured_trajectory_rule'
        if candidate is not None:
            if (tuple(candidate.npc_ids) != tuple(window.npc_ids) or
                    candidate.coordinate_frame != window.coordinate_frame or
                    not np.isclose(candidate.dt, window.dt) or
                    np.asarray(candidate.npc_states).shape != (f, n, 8) or
                    np.asarray(candidate.npc_mask).shape != (f, n)):
                raise ValueError('Candidate must match NPC IDs, dt, coordinate frame and future shapes')
            if np.any(npc_mask[h:] & ~np.asarray(candidate.npc_mask, bool)):
                raise ValueError('Candidate mask cannot hide NPCs observed in the original future')
            npc = npc.copy()
            npc_mask = npc_mask.copy()
            npc[h:], npc_mask[h:] = candidate.npc_states, candidate.npc_mask
            source = 'fixed_ego_candidate_rule'
        # Only the anchor and future are scored, not earlier history.
        ego, npc, ego_mask, npc_mask, times = (v[h-1:] for v in (ego, npc, ego_mask, npc_mask, times))
        finite_ego = np.isfinite(ego).all(-1) & (ego[:, 6:8] > 0).all(-1)
        finite_npc = np.isfinite(npc).all(-1) & (npc[:, :, 6:8] > 0).all(-1)
        contradiction = bool(np.any(ego_mask & ~finite_ego) or np.any(npc_mask & ~finite_npc))
        good_ego, good_npc = ego_mask & finite_ego, npc_mask & finite_npc
        each_frame_has_npc = bool(n == 0 or np.all(np.any(good_npc, axis=1)))
        internal_gap = False
        for j in range(n):
            present = np.flatnonzero(good_npc[:, j])
            if len(present) > 1 and not good_npc[present[0]:present[-1]+1, j].all():
                internal_gap = True
        reference = window.metadata.get('position_reference')
        flags = {'valid': False, 'position_reference': reference,
                 'position_reference_known': reference in POSITION_REFERENCES,
                 'coordinate_frame_supported': window.coordinate_frame in ('world', 'ego-centric'),
                 'states_in_physical_units': window.normalizer_id is None,
                 'mask_value_contradiction': contradiction, 'ego_complete': bool(good_ego.all()),
                 'npc_observation_each_frame': each_frame_has_npc,
                 'npc_internal_observation_gap': internal_gap,
                 'npc_mask_coverage_fraction': float(good_npc.mean()) if n else None,
                 'npc_absence_vs_capture_gap_verified': False,
                 'observed_pair_count': 0, 'no_npcs': n == 0, 'geometric_overlap': False,
                 'counterfactual_closed_loop': False, 'near_miss_rule': False,
                 'ttc_method': 'constant_velocity_fixed_heading_obb_contact',
                 'ttc_horizon_seconds': self.config.ttc_horizon,
                 'collision_probability_calibrated': False,
                 'progress_method': 'center_path_speed_proxy',
                 'distribution_calibration_available': False}
        report = RiskReport(None, None, None, None, None, flags, None, 'obb_rules_v1', source)
        if not all(flags[k] for k in ('position_reference_known', 'coordinate_frame_supported', 'states_in_physical_units')):
            return report
        distances, ttcs = [], []
        for i in range(len(ego)):
            if not good_ego[i]:
                continue
            for j in np.flatnonzero(good_npc[i]):
                distance = obb_distance(ego[i], npc[i, j], reference)
                distances.append(distance)
                ttc = obb_ttc(ego[i], npc[i, j], reference, self.config.ttc_horizon)
                if ttc is not None:
                    ttcs.append(ttc)
        flags['observed_pair_count'] = len(distances)
        flags['geometric_overlap'] = bool(distances and min(distances) <= EPS)
        flags['valid'] = bool(good_ego.all() and not contradiction and not internal_gap and
                              each_frame_has_npc and (n == 0 or distances))
        report.min_distance = min(distances) if distances else None
        report.min_ttc = min(ttcs) if ttcs else None
        flags['near_miss_rule'] = bool(not flags['geometric_overlap'] and (
            (report.min_distance is not None and report.min_distance <= self.config.near_distance) or
            (report.min_ttc is not None and report.min_ttc <= self.config.near_ttc)))
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.
        flags['progress_duration_seconds'] = duration
        flags['progress_duration_sufficient'] = duration + EPS >= self.config.progress_duration
        if good_ego.all() and flags['progress_duration_sufficient']:
            centers = np.array([_body(state, reference)[0] for state in ego])
            speed = float(np.linalg.norm(np.diff(centers, axis=0), axis=-1).sum() / duration)
            report.progress_penalty = float(np.clip(1 - speed / self.config.progress_speed, 0., 1.))
        report.components = {
            'overlap': float(flags['geometric_overlap']),
            'ttc': max(0., 1 - report.min_ttc/self.config.ttc_horizon) if report.min_ttc is not None else 0.,
            'distance': max(0., 1 - report.min_distance/self.config.near_distance) if report.min_distance is not None else 0.,
            'progress': report.progress_penalty}
        if flags['valid']:
            available = {k: v for k, v in report.components.items() if v is not None and self.config.weights.get(k, 0) > 0}
            denominator = sum(self.config.weights[k] for k in available)
            if denominator:
                report.score = float(sum(self.config.weights[k]*v for k, v in available.items())/denominator)
            else:
                flags['valid'] = False
                flags['no_weighted_components'] = True
        return report
