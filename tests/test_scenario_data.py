"""CPU-only behavioral tests for the stage 1 trajectory data boundary."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "embodied/envs/scenario_data.py"


def load_data_module():
    if not MODULE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("scenario_data_test_subject", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


data = load_data_module()


def vehicle(vid, x=0.0, y=0.0, yaw=0.0, speed=4.0):
    return {"id": vid, "pos": [x, y], "v": speed, "yaw": yaw,
            "a": -0.5, "length": 4.5, "width": 1.8,
            "road_id": "main", "lane_id": "main_0"}


def record(times=None, episode_id="episode-1", scenario_id="base-1", parent=None):
    times = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0] if times is None else times
    frames = []
    for i, timestamp in enumerate(times):
        vehicles = [vehicle("Auto", x=10.0 + 2 * i), vehicle("b", x=20.0 + i)]
        if i == 2:
            vehicles.pop()
        if i >= 4:
            vehicles.insert(0, vehicle("a", x=30.0 + i))
        frame = {"sim_time": timestamp, "state": [float(i)] * 31,
                 "vehicles": vehicles, "ttc": 0.4 if i == 3 else 8.0,
                 "cost": float(i == len(times) - 1),
                 "terminal_reason": "collision" if i == len(times) - 1 else ""}
        if i:
            frame["action"] = [float(i)]
        frames.append(frame)
    return {"episode_id": episode_id, "scenario_id": scenario_id,
            "parent_scenario_id": parent, "source_type": "sumo_original",
            "replay_trajectory": {"source_cfg": "maps/merge.sumocfg", "frames": frames}}


class ScenarioDataTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "The independent scenario_data module is missing")

    def test_vehicle_ids_survive_order_changes_and_missing_frames(self):
        episode = data.episode_from_record(record(), source_path="danger.jsonl:4")
        self.assertEqual(episode.npc_ids, ("a", "b"))
        np.testing.assert_equal(episode.npc_valid_mask[:, 1], [1, 1, 0, 1, 1, 1])
        np.testing.assert_equal(episode.npc_valid_mask[:, 0], [0, 0, 0, 0, 1, 1])
        self.assertEqual(episode.npc_states[4, 1, 0], 24.0)
        self.assertTrue(np.isnan(episode.npc_states[2, 1]).all())
        self.assertEqual(episode.source_path, "danger.jsonl:4")
        self.assertIsNone(episode.seed)
        self.assertIsNone(episode.map_id)
        self.assertEqual(episode.raw_states.shape, (6, 31))

    def test_known_sumo_provenance_backfill_adapts_and_loads_without_data_changes(self):
        item = record()
        baseline = data.episode_from_record(item)
        markers = {"prediction_source": "sumo_execution",
                   "state_alignment": "legacy_observation; terminal_state_may_precede_snapshot"}
        item["metadata"] = dict(markers)
        original = copy.deepcopy(item)
        adapted = data.episode_from_record(item)
        self.assertEqual(adapted.metadata.get("position_reference"), "front_bumper_center")
        self.assertEqual(adapted.metadata.get("ttc_method"), "legacy_longitudinal_gap_ttc")
        self.assertEqual(adapted.metadata.get("ttc_parameters"),
                         {"d_safe": 4.36, "lateral_gate": 2.0, "negative_clipped": True})
        limitations = adapted.metadata.get("ttc_limitations", "")
        for word in ("forward", "closing", "road", "lateral"):
            self.assertIn(word, limitations)
        self.assertEqual(adapted.metadata.get("event_verification"),
                         "recorded_labels_only; physical_near_miss_not_verified")
        self.assertEqual(item, original)
        np.testing.assert_equal(adapted.ego_state, baseline.ego_state)
        np.testing.assert_equal(adapted.npc_states, baseline.npc_states)
        np.testing.assert_equal(adapted.events["ttc"], [8, 8, 8, 0.4, 8, 8])
        self.assertEqual(adapted.raw_frames, original["replay_trajectory"]["frames"])
        with tempfile.TemporaryDirectory() as directory:
            path = data.write_episode(baseline, directory)
            # Reproduce a previously collected archive, without adapting it first.
            with np.load(str(path), allow_pickle=False) as archive:
                legacy_arrays = {name: archive[name].copy() for name in archive.files}
            header = json.loads(str(legacy_arrays["metadata_json"].item()))
            header["metadata"].update(markers)
            legacy_arrays["metadata_json"] = np.asarray(json.dumps(header))
            np.savez_compressed(str(path), **legacy_arrays)
            loaded = data.load_episode(path)
            with np.load(str(path), allow_pickle=False) as archive:
                for name in legacy_arrays:
                    np.testing.assert_equal(archive[name], legacy_arrays[name])
        for name in ("position_reference", "ttc_method", "ttc_parameters", "ttc_limitations", "event_verification"):
            self.assertEqual(loaded.metadata.get(name), adapted.metadata.get(name))
        for name in ("timestamps", "ego_state", "npc_states", "ego_actions", "raw_states", "ego_valid_mask", "npc_valid_mask"):
            np.testing.assert_equal(getattr(loaded, name), getattr(baseline, name))
        self.assertEqual(loaded.raw_frames, baseline.raw_frames)
        np.testing.assert_equal(loaded.events["ttc"], baseline.events["ttc"])
        window = data.make_windows(loaded, history=2, future=2)[0]
        for name in (*markers, "position_reference", "ttc_method", "ttc_parameters", "ttc_limitations", "event_verification"):
            self.assertEqual(window.metadata.get(name), loaded.metadata.get(name))
        self.assertNotIn("canonical_frame_indices", window.metadata)
        window.metadata["ttc_parameters"]["d_safe"] = 99
        self.assertEqual(loaded.metadata["ttc_parameters"]["d_safe"], 4.36)

    def test_explicit_source_metadata_takes_precedence_over_sumo_defaults(self):
        item = record()
        explicit = {"prediction_source": "sumo_execution",
                    "state_alignment": "legacy_observation; terminal_state_may_precede_snapshot",
                    "position_reference": "caller_reference", "ttc_method": "caller_ttc",
                    "ttc_parameters": {"d_safe": 7.0, "detail": {"caller": True}},
                    "ttc_limitations": "caller limitations", "event_verification": "caller verification"}
        item["metadata"] = copy.deepcopy(explicit)
        adapted = data.episode_from_record(item)
        with tempfile.TemporaryDirectory() as directory:
            loaded = data.load_episode(data.write_episode(adapted, directory))
        window = data.make_windows(loaded, history=2, future=2)[0]
        for episode in (adapted, loaded):
            for name, value in explicit.items():
                self.assertEqual(episode.metadata.get(name), value)
        for name, value in explicit.items():
            self.assertEqual(window.metadata.get(name), value)
        window.metadata["ttc_parameters"]["detail"]["caller"] = False
        self.assertTrue(loaded.metadata["ttc_parameters"]["detail"]["caller"])
        self.assertTrue(item["metadata"]["ttc_parameters"]["detail"]["caller"])

    def test_unknown_source_does_not_inherit_sumo_specific_assumptions(self):
        cases = ({}, {"prediction_source": "sumo_execution"},
                 {"state_alignment": "legacy_observation; terminal_state_may_precede_snapshot"},
                 {"prediction_source": "other_simulator",
                  "state_alignment": "legacy_observation; terminal_state_may_precede_snapshot"})
        for metadata in cases:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                item = record()  # A source_type label alone is insufficient evidence.
                item["metadata"] = dict(metadata)
                adapted = data.episode_from_record(item)
                loaded = data.load_episode(data.write_episode(adapted, directory))
                window = data.make_windows(loaded, history=2, future=2)[0]
                for container in (adapted.metadata, loaded.metadata, window.metadata):
                    for name in ("position_reference", "ttc_method", "ttc_parameters", "ttc_limitations", "event_verification"):
                        self.assertIsNone(container.get(name))
                for name, value in metadata.items():
                    self.assertEqual(window.metadata.get(name), value)

    def test_velocity_derivation_and_missing_fields_are_explicit(self):
        item = record()
        ego = item["replay_trajectory"]["frames"][0]["vehicles"][0]
        ego["yaw"] = math.pi / 2
        episode = data.episode_from_record(item)
        np.testing.assert_allclose(episode.ego_state[0, 2:4], [0, 4], atol=1e-6)
        del ego["length"]
        episode = data.episode_from_record(item)
        self.assertFalse(episode.ego_valid_mask[0])
        self.assertGreater(episode.metadata["missing_vehicle_fields"]["length"], 0)

    def test_actions_apply_to_following_interval_and_last_is_invalid(self):
        episode = data.episode_from_record(record())
        np.testing.assert_equal(episode.ego_actions[:-1, 0], [1, 2, 3, 4, 5])
        np.testing.assert_equal(episode.ego_action_mask, [1, 1, 1, 1, 1, 0])
        windows = data.make_windows(episode, history=2, future=2, dt=0.1)
        np.testing.assert_equal(windows[0].ego_actions[:, 0], [1, 1, 2, 2])
        np.testing.assert_allclose(windows[0].timestamps, [3.0, 3.1, 3.2, 3.3])

    def test_resampling_never_interpolates_over_a_missing_vehicle(self):
        episode = data.episode_from_record(record())
        windows = data.make_windows(episode, history=2, future=2, dt=0.1)
        self.assertEqual(len(windows), 8)
        window = windows[2]
        np.testing.assert_equal(window.npc_mask[:, 1], [1, 0, 0, 0])
        self.assertEqual(window.history_npcs.shape, (2, 2, 8))
        self.assertEqual(window.future_npcs.shape, (2, 2, 8))
        self.assertAlmostEqual(window.history_ego[1, 0], 13.0)

    def test_yaw_crossing_pi_uses_short_rotation(self):
        item = record([0.0, 0.2])
        item["replay_trajectory"]["frames"][0]["vehicles"][0]["yaw"] = math.pi - 0.1
        item["replay_trajectory"]["frames"][1]["vehicles"][0]["yaw"] = -math.pi + 0.1
        window = data.make_windows(data.episode_from_record(item), history=2, future=1)[0]
        self.assertAlmostEqual(abs(window.history_ego[1, 4]), math.pi, places=6)
        self.assertLess(abs(window.future_ego[0, 4] - window.history_ego[1, 4]), 0.11)

    def test_missing_ego_does_not_invent_anchor_or_action(self):
        item = record([0.0, 0.2, 0.4])
        item["replay_trajectory"]["frames"][1]["vehicles"].pop(0)
        episode = data.episode_from_record(item)
        np.testing.assert_equal(episode.ego_action_mask, [0, 0, 0])
        window = data.make_windows(episode, history=2, future=1, dt=0.2,
                                   coordinate_frame="ego-centric")[0]
        self.assertFalse(window.validity_flags["ego_anchor_available"])
        self.assertTrue(np.isnan(window.history_ego).all())

    def test_timestamp_errors_are_not_silently_repaired(self):
        for times in ([0.0, 0.0], [0.2, 0.1], [0.0, float("nan")]):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "sim_time"):
                data.episode_from_record(record(times))
        item = record()
        del item["replay_trajectory"]["frames"][0]["sim_time"]
        with self.assertRaisesRegex(ValueError, "sim_time"):
            data.episode_from_record(item)

    def test_terminal_only_duplicate_requires_explicit_no_simulation(self):
        item = record([0.0, 0.2])
        duplicate = copy.deepcopy(item["replay_trajectory"]["frames"][-1])
        duplicate.update(terminal_reason="goal", simulation_advanced=False, action=[99.0])
        item["replay_trajectory"]["frames"].append(duplicate)
        episode = data.episode_from_record(item)
        self.assertEqual(len(episode.timestamps), 2)
        self.assertEqual(len(episode.raw_frames), 3)
        self.assertEqual(episode.ego_actions[0, 0], 1.0)
        self.assertEqual(episode.metadata["collapsed_terminal_frames"], 1)
        self.assertEqual(episode.metadata["collapsed_terminal_frame_indices"], [2])

    def test_summary_record_is_rejected_without_fake_trajectory(self):
        with self.assertRaisesRegex(ValueError, "frames.*summary|summary.*frames"):
            data.episode_from_record({"scenario_id": "sweep-1", "min_ttc": 0.2})

    def test_fixed_ego_anchor_rotates_positions_and_velocities(self):
        states = np.array([[10, 22, 0, 4, 0, -1, 4.5, 1.8]], dtype=float)
        anchor = np.array([10, 20, 0, 0, math.pi / 2, 0, 4.5, 1.8])
        result = data.to_ego_centric(states, anchor)
        np.testing.assert_allclose(result[0, :4], [2, 0, 4, 0], atol=1e-6)
        self.assertAlmostEqual(result[0, 4], -math.pi / 2)
        np.testing.assert_equal(states[0, :2], [10, 22])

    def test_frenet_absence_and_reference_projection(self):
        states = np.array([[3, 2, 4, 0, 0, 0, 4.5, 1.8]], dtype=float)
        projected, flags = data.to_frenet(states)
        self.assertIsNone(projected)
        self.assertFalse(flags["frenet_available"])
        projected, flags = data.to_frenet(states, reference_line=[[0, 0], [10, 0]])
        np.testing.assert_allclose(projected, [[3, 2]])
        self.assertTrue(flags["frenet_available"])

    def test_roundtrip_is_pickle_free_and_preserves_raw_frames(self):
        episode = data.episode_from_record(record(), source_path="source.jsonl:7")
        with tempfile.TemporaryDirectory() as directory:
            path = data.write_episode(episode, directory)
            with np.load(str(path), allow_pickle=False) as archive:
                self.assertEqual(archive["npc_states"].shape, (6, 2, 8))
                self.assertNotEqual(archive["metadata_json"].dtype.kind, "O")
            restored = data.load_episode(path)
        np.testing.assert_equal(restored.ego_state, episode.ego_state)
        np.testing.assert_equal(restored.npc_valid_mask, episode.npc_valid_mask)
        self.assertEqual(restored.raw_frames, episode.raw_frames)
        self.assertEqual(restored.source_cfg, "maps/merge.sumocfg")
        self.assertEqual(restored.source_path, "source.jsonl:7")

    def test_episode_id_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for episode_id in ("../escape", "C:\\escape", "a/b", "..", "CON"):
                with self.subTest(episode_id=episode_id), self.assertRaises(ValueError):
                    data.write_episode(data.episode_from_record(record(episode_id=episode_id)), directory)

    def test_parent_chain_and_base_share_one_split(self):
        episodes = []
        for i in range(12):
            episodes.extend([
                data.episode_from_record(record(episode_id="base-" + str(i), scenario_id="s" + str(i))),
                data.episode_from_record(record(episode_id="child-" + str(i), scenario_id="c" + str(i), parent="s" + str(i))),
                data.episode_from_record(record(episode_id="grand-" + str(i), scenario_id="g" + str(i), parent="c" + str(i)))])
        split = data.split_episodes(episodes, seed=13)
        destinations = {ep.episode_id: name for name, group in split.items() for ep in group}
        for i in range(12):
            self.assertEqual(destinations["base-" + str(i)], destinations["child-" + str(i)])
            self.assertEqual(destinations["child-" + str(i)], destinations["grand-" + str(i)])
        self.assertTrue(split["train"] and split["val"] and split["test"])
        repeated = data.split_episodes(list(reversed(episodes)), seed=13)
        self.assertEqual({ep.episode_id: name for name, group in repeated.items() for ep in group}, destinations)

    def test_normalizer_fits_only_train_and_persists(self):
        train = data.episode_from_record(record([0.0, 0.2], episode_id="train"))
        held_out = data.episode_from_record(record([0.0, 0.2], episode_id="test"))
        held_out.ego_state[:, 0] = 100000.0
        held_out.npc_states[:, :, 0] = 100000.0
        normalizer = data.fit_normalizer({"train": [train], "val": [], "test": [held_out]})
        self.assertAlmostEqual(normalizer.mean[0], 15.75)
        self.assertEqual(normalizer.training_episode_ids, ("train",))
        transformed = normalizer.transform(train.npc_states, train.npc_valid_mask)
        self.assertTrue(np.isfinite(transformed[train.npc_valid_mask]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = normalizer.save(Path(directory) / "normalizer.json")
            restored = data.ScenarioNormalizer.load(path)
        np.testing.assert_equal(restored.mean, normalizer.mean)
        self.assertEqual(restored.normalizer_id, normalizer.normalizer_id)
        with self.assertRaisesRegex(ValueError, "train"):
            data.fit_normalizer({"train": [], "test": [held_out]})

    def test_normalizer_treats_pi_crossing_as_neighboring_directions(self):
        item = record([0.0, 0.2])
        for i, frame in enumerate(item["replay_trajectory"]["frames"]):
            for snapshot in frame["vehicles"]:
                snapshot["yaw"] = math.pi - 0.1 if i == 0 else -math.pi + 0.1
        episode = data.episode_from_record(item)
        normalizer = data.fit_normalizer({"train": [episode]})
        self.assertAlmostEqual(abs(normalizer.mean[4]), math.pi)
        self.assertAlmostEqual(normalizer.std[4], 0.1)
        normalized = normalizer.transform(episode.ego_state)
        np.testing.assert_allclose(normalized[:, 4], [-1, 1], atol=1e-6)
        restored = normalizer.inverse_transform(normalized)
        np.testing.assert_allclose(np.sin(restored[:, 4]), np.sin(episode.ego_state[:, 4]), atol=1e-6)

    def test_configurable_difficult_event_sampling_and_batch_padding(self):
        episode = data.episode_from_record(record())
        windows = data.make_windows(episode, history=2, future=2, dt=0.1,
                                   event_config={"near_miss_ttc": 0.5},
                                   event_only=True, event_before=0.1, event_after=0.1)
        self.assertTrue(windows)
        self.assertTrue(any("near_miss" in window.event_labels for window in windows))
        stricter = data.make_windows(episode, history=2, future=2, dt=0.1,
                                    event_config={"near_miss_ttc": 0.1})
        self.assertFalse(any("near_miss" in window.event_labels for window in stricter))
        smaller = record([0, 0.1, 0.2, 0.3])
        for frame in smaller["replay_trajectory"]["frames"]:
            frame["vehicles"] = frame["vehicles"][:1]
        second = data.make_windows(data.episode_from_record(smaller), history=2, future=2)[0]
        batch = data.pad_windows([windows[0], second])
        self.assertEqual(batch["history_npcs"].shape, (2, 2, 2, 8))
        self.assertFalse(batch["npc_mask"][1].any())

    def test_cli_convert_and_inspect_reports_bad_summary_and_valid_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "danger.jsonl"
            source.write_text(json.dumps(record()) + "\n" + json.dumps({"scenario_id": "summary"}) + "\n", encoding="utf-8")
            output = Path(directory) / "episodes"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            converted = subprocess.run([sys.executable, str(MODULE_PATH), "convert", "--input", str(source), "--output", str(output)], capture_output=True, text=True, env=env)
            self.assertEqual(converted.returncode, 1)
            report = json.loads(converted.stdout)
            self.assertEqual(report["converted"], 1)
            self.assertEqual(report["rejected"], 1)
            self.assertIn("frames", report["errors"][0]["error"])
            inspected = subprocess.run([sys.executable, str(MODULE_PATH), "inspect_dataset", "--input", str(output), "--history", "2", "--future", "2"], capture_output=True, text=True, env=env)
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            report = json.loads(inspected.stdout)
            self.assertEqual(report["episodes"], 1)
            self.assertEqual(report["windows"], 8)
            self.assertIn("collision", report["event_counts"])
            self.assertIn("missing_field_rates", report)
            self.assertIn("splits", report)

    def test_path_and_window_input_errors_are_clear(self):
        with self.assertRaises(FileNotFoundError):
            data.load_episode("nonexistent-stage1.npz")
        episode = data.episode_from_record(record())
        for options in ({"dt": 0}, {"history": 0}, {"future": -1}, {"stride": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                data.make_windows(episode, **options)

    def test_prepare_cli_persists_group_manifest_and_training_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            for i in range(12):
                data.write_episode(data.episode_from_record(record(episode_id="e" + str(i), scenario_id="s" + str(i))), source)
                data.write_episode(data.episode_from_record(record(episode_id="c" + str(i), scenario_id="c" + str(i), parent="s" + str(i))), source)
            output = Path(directory) / "prepared"
            result = subprocess.run([sys.executable, str(MODULE_PATH), "prepare_dataset", "--input", str(source), "--output", str(output), "--split-seed", "13"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
            normalizer = data.ScenarioNormalizer.load(report["normalizer_path"])
            self.assertEqual(set(normalizer.training_episode_ids), {item["episode_id"] for item in manifest["splits"]["train"]})
            self.assertTrue(manifest["splits"]["test"])

    def test_large_time_gap_produces_invalid_samples_and_controls(self):
        episode = data.episode_from_record(record([0.0, 0.2, 0.4, 1.4]))
        windows = data.make_windows(episode, history=2, future=2, dt=0.1)
        self.assertFalse(windows[5].ego_mask.any())
        self.assertFalse(windows[5].ego_action_mask.any())

    def test_raw_state_alone_is_rejected_without_any_vehicle_snapshots(self):
        item = record([0.0, 0.2])
        for frame in item["replay_trajectory"]["frames"]:
            frame["vehicles"] = []
        with self.assertRaisesRegex(ValueError, "vehicle.*snapshot|snapshot.*vehicle"):
            data.episode_from_record(item)

    def test_ego_only_episode_is_valid_without_npc_snapshots(self):
        item = record([0.0, 0.2])
        for frame in item["replay_trajectory"]["frames"]:
            frame["vehicles"] = [snapshot for snapshot in frame["vehicles"] if snapshot["id"] == "Auto"]
        episode = data.episode_from_record(item)
        self.assertTrue(episode.raw_state_mask.all())
        self.assertTrue(episode.ego_valid_mask.all())
        self.assertEqual(episode.npc_states.shape, (2, 0, 8))
        window = data.make_windows(episode, history=2, future=1)[0]
        self.assertEqual(window.history_npcs.shape, (2, 0, 8))
        self.assertTrue(window.ego_mask.all())

    def test_low_progress_uses_observed_duration_and_not_cost_as_collision(self):
        item = record([0.0, 1.0, 2.0, 3.0])
        for frame in item["replay_trajectory"]["frames"]:
            frame["vehicles"][0]["v"] = 0.0
            frame["terminal_reason"] = "timeout"
            frame["cost"] = 1.0
        episode = data.episode_from_record(item)
        self.assertFalse(episode.events["collision"].any())
        window = data.make_windows(episode, history=2, future=2, dt=1.0,
                                   event_config={"low_progress_duration": 2.0})[0]
        self.assertIn("low_progress", window.event_labels)


if __name__ == "__main__":
    unittest.main()
