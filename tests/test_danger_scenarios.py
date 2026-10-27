import json
import tempfile
import unittest
from pathlib import Path

from embodied.envs.danger_scenarios import StructuredDangerScenarioMiner


def make_replay_trajectory(source_cfg="/tmp/scenario_a.sumocfg"):
    """Mirror the vehicle snapshots and replay envelope emitted by SUMO."""
    return {
        "version": 1,
        "source_cfg": source_cfg,
        "critical_index": 0,
        "background_vehicle_frames": 1,
        "max_background_vehicles": 1,
        "frames": [{
            "step": 1,
            "sim_time": 0.1,
            "state": [0.4, 0.1, 0.2, -0.3, 0.0, 0.5] + [2.0] * 25,
            "cost": 1.0,
            "ttc": 0.5,
            "terminal_reason": "collision",
            "action": [0.2],
            "acc": 0.4,
            "vehicles": [{
                "id": "Auto",
                "type": "vehicle",
                "type_id": "car",
                "pos": [10.0, 2.0],
                "v": 4.0,
                "a": 0.4,
                "yaw": 0.0,
                "angle": 90.0,
                "road_id": "merge",
                "lane_id": "merge_0",
                "lane_index": 0,
                "lane_position": 10.0,
                "route_id": "ego_route",
                "route_edges": ["merge", "out"],
                "length": 4.5,
                "width": 1.8,
            }, {
                "id": "NPC_0",
                "type": "vehicle",
                "type_id": "car",
                "pos": [14.0, 2.0],
                "v": 2.0,
                "a": 0.0,
                "yaw": 0.0,
                "angle": 90.0,
                "road_id": "merge",
                "lane_id": "merge_0",
                "lane_index": 0,
                "lane_position": 14.0,
                "route_id": "npc_route",
                "route_edges": ["merge", "out"],
                "length": 4.5,
                "width": 1.8,
            }],
        }],
    }


def make_record(**overrides):
    record = {
        "event": "episode",
        "worker": 0,
        "episode": 1,
        "source_cfg": "/tmp/scenario_a.sumocfg",
        "length": 100,
        "total_cost": 1.0,
        "max_cost": 1.0,
        "min_ttc": 0.5,
        "terminal_reason": "collision",
        "event_type": "collision",
        "severity": 10.0,
        "replay_trajectory": make_replay_trajectory(
            overrides.get("source_cfg", "/tmp/scenario_a.sumocfg")),
        "critical_state": {
            "ego_speed_norm": 0.4,
            "closest_agent": {
                "slot": 0,
                "rel_x_norm": 0.1,
                "rel_y_norm": 0.2,
                "rel_vx_norm": -0.3,
                "rel_vy_norm": 0.0,
                "heading_norm": 0.5,
                "distance_norm": 0.22,
            },
        },
    }
    record.update(overrides)
    return record


def record_collision(miner, source_cfg="/tmp/scenario_a.sumocfg", **overrides):
    episode = {
        "source_cfg": source_cfg,
        "length": 100,
        "total_cost": 1.0,
        "max_cost": 1.0,
        "min_ttc": 0.5,
        "terminal_reason": "collision",
        "critical_state": [0.4, 0.1, 0.2, -0.3, 0.0, 0.5] + [2.0] * 25,
        "replay_trajectory": make_replay_trajectory(source_cfg),
    }
    episode.update(overrides)
    return miner.record_episode(**episode)


def write_events(directory, *records):
    path = Path(directory) / "danger_events_worker_0.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")
    return path


class StructuredDangerScenarioMinerTest(unittest.TestCase):

    def make_miner(self, directory, library_size=200, **overrides):
        config = {
            "enabled": True,
            "warmup_episodes": 0,
            "library_size": library_size,
        }
        config.update(overrides)
        return StructuredDangerScenarioMiner(
            config,
            output_dir=directory,
            worker_index=0)

    def test_duplicate_import_does_not_grow_library(self):
        with tempfile.TemporaryDirectory() as directory:
            record = make_record()
            write_events(directory, record, record)

            miner = self.make_miner(directory)
            stats = miner.set_active(True, record_events=False)

            self.assertEqual(len(miner.library), 1)
            self.assertEqual(len(miner.prototype_keys), 1)
            self.assertEqual(stats["external_prototypes_loaded"], 1.0)

    def test_different_prototypes_are_inserted(self):
        with tempfile.TemporaryDirectory() as directory:
            first = make_record(episode=1)
            second = make_record(
                episode=2,
                source_cfg="/tmp/scenario_b.sumocfg",
                critical_state={
                    "ego_speed_norm": 0.7,
                    "closest_agent": {"slot": 1, "distance_norm": 0.5},
                })
            write_events(directory, first, second)

            miner = self.make_miner(directory)
            miner.set_active(True, record_events=False)

            self.assertEqual(len(miner.library), 2)
            self.assertEqual(len(miner.prototype_keys), 2)

    def test_full_library_keeps_higher_severity_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            low = make_record(
                episode=1,
                event_type="low_progress",
                terminal_reason="max_time",
                severity=1.0)
            high = make_record(
                episode=2,
                source_cfg="/tmp/high.sumocfg",
                event_type="collision",
                terminal_reason="collision",
                severity=12.0)
            write_events(directory, low, high)

            miner = self.make_miner(directory, library_size=1)
            miner.set_active(True, record_events=False)

            self.assertEqual(len(miner.library), 1)
            self.assertEqual(miner.library[0]["source_cfg"], "/tmp/high.sumocfg")
            self.assertEqual(len(miner.prototype_keys), 1)

    def test_repeated_activation_does_not_reload_same_prototype(self):
        with tempfile.TemporaryDirectory() as directory:
            record = make_record()
            write_events(directory, record)

            miner = self.make_miner(directory)
            miner.set_active(True, record_events=False)
            miner.set_active(False)
            stats = miner.set_active(True, record_events=False)

            self.assertEqual(len(miner.library), 1)
            self.assertEqual(len(miner.prototype_keys), 1)
            self.assertEqual(stats["external_prototypes_loaded"], 1.0)

    def test_online_record_is_not_duplicated_when_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(directory)
            miner.set_active(True, record_events=True)
            mined = record_collision(miner)

            self.assertIsNotNone(mined)
            self.assertEqual(len(miner.library), 1)

            miner.set_active(False)
            stats = miner.set_active(True, record_events=False)

            self.assertEqual(len(miner.library), 1)
            self.assertEqual(len(miner.prototype_keys), 1)
            self.assertEqual(stats["external_prototypes_loaded"], 0.0)

    def test_legacy_activation_keeps_recording_and_online_mining(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(directory)
            miner.set_active(True)

            self.assertTrue(miner.recording)
            self.assertFalse(miner.collecting)
            self.assertTrue(miner.update_library)
            self.assertIsNotNone(record_collision(miner))
            self.assertTrue(miner.ready)
            self.assertEqual(miner.episodes_mined, 1)

            miner.set_active(False)
            self.assertFalse(miner.recording)
            self.assertFalse(miner.collecting)
            self.assertFalse(miner.ready)
            miner.set_active(True, False)
            self.assertFalse(miner.recording)
            self.assertTrue(miner.ready)

    def test_recording_with_library_frozen_writes_event_without_mining(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(directory)
            miner.set_active(True, record_events=True, update_library=False)

            mined = record_collision(miner)

            self.assertIsNone(mined)
            self.assertEqual(miner.library, [])
            self.assertEqual(miner.prototype_keys, set())
            self.assertEqual(miner.episodes_seen, 1)
            self.assertEqual(miner.episodes_mined, 0)
            event = json.loads(miner.output_path.read_text(encoding="utf-8"))
            self.assertEqual(event["event_type"], "collision")
            vehicles = event["replay_trajectory"]["frames"][0]["vehicles"]
            self.assertEqual(vehicles[1]["id"], "NPC_0")
            self.assertGreater(event["severity"], 10.0)

    def test_collection_can_be_active_with_legacy_event_recording_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(directory)
            miner.set_active(True, record_events=False, collect_episodes=True)

            self.assertTrue(miner.collecting)
            self.assertFalse(miner.recording)
            self.assertIsNone(record_collision(miner))
            self.assertFalse(miner.output_path.exists())
            self.assertEqual(miner.episodes_seen, 0)
            self.assertEqual(miner.library, [])

            miner.set_active(False, collect_episodes=True)
            self.assertFalse(miner.collecting)

    def test_activation_overrides_last_until_next_call_then_use_config(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(
                directory, collect_episodes=True, update_library=False)
            miner.set_active(True, record_events=False)
            self.assertTrue(miner.collecting)
            self.assertFalse(miner.update_library)

            miner.set_active(True, collect_episodes=False, update_library=True)
            self.assertFalse(miner.collecting)
            self.assertTrue(miner.update_library)
            miner.set_active(False)
            self.assertFalse(miner.collecting)
            miner.set_active(True)
            self.assertTrue(miner.collecting)
            self.assertFalse(miner.update_library)
            self.assertTrue(miner.recording)

    def test_disabled_miner_cannot_collect_even_with_explicit_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            miner = self.make_miner(directory, enabled=False, collect_episodes=True)
            miner.set_active(True, record_events=True, collect_episodes=True)

            self.assertFalse(miner.collecting)
            self.assertFalse(miner.recording)
            self.assertFalse(miner.ready)
            self.assertIsNone(record_collision(miner))
            self.assertFalse(miner.output_path.exists())

    def test_frozen_library_still_imports_sweep_and_reaches_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            write_events(directory, make_record())
            miner = self.make_miner(
                directory, warmup_episodes=20, update_library=False)
            stats = miner.set_active(True, record_events=True)
            before_library = json.loads(json.dumps(miner.library))
            before_keys = set(miner.prototype_keys)

            self.assertEqual(stats["external_prototypes_loaded"], 1.0)
            self.assertTrue(miner.ready)
            self.assertEqual(miner.episodes_seen, 20)
            self.assertIsNone(record_collision(miner))
            self.assertEqual(miner.library, before_library)
            self.assertEqual(miner.prototype_keys, before_keys)
            self.assertEqual(miner.episodes_mined, 0)

    def test_ego_only_trajectory_does_not_enter_online_library(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = make_replay_trajectory()
            replay["background_vehicle_frames"] = 0
            replay["max_background_vehicles"] = 0
            replay["frames"][0]["vehicles"] = replay["frames"][0]["vehicles"][:1]
            miner = self.make_miner(directory)
            miner.set_active(True)

            self.assertIsNone(record_collision(miner, replay_trajectory=replay))
            self.assertEqual(miner.library, [])
            self.assertEqual(miner.episodes_mined, 0)
            event = json.loads(miner.output_path.read_text(encoding="utf-8"))
            self.assertEqual(event["replay_skipped_reason"], "no_background_vehicle")

    def test_collection_and_library_switches_preserve_rng_and_variant_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_cfg = root / "scenario.sumocfg"
            source_cfg.write_text("<configuration/>\n", encoding="utf-8")
            legacy_dir = root / "legacy"
            collect_dir = root / "collect"
            legacy_dir.mkdir()
            collect_dir.mkdir()
            record = make_record(
                source_cfg=str(source_cfg),
                replay_trajectory=make_replay_trajectory(str(source_cfg)))
            write_events(legacy_dir, record)
            write_events(collect_dir, record)
            legacy = self.make_miner(legacy_dir, seed=123, generate_probability=1.0)
            collecting = self.make_miner(collect_dir, seed=123, generate_probability=1.0)
            initial_rng_state = legacy.rng.getstate()
            legacy.set_active(True, record_events=False)
            collecting.set_active(
                True, record_events=False, collect_episodes=True, update_library=False)

            self.assertEqual(legacy.rng.getstate(), initial_rng_state)
            self.assertEqual(collecting.rng.getstate(), initial_rng_state)
            self.assertTrue(collecting.collecting)
            self.assertTrue(legacy.ready)
            self.assertTrue(collecting.ready)
            for _ in range(10):
                legacy_variant = legacy.sample_variant(str(source_cfg))
                collecting_variant = collecting.sample_variant(str(source_cfg))
                self.assertIsNotNone(legacy_variant)
                self.assertEqual(legacy_variant, collecting_variant)
                self.assertEqual(legacy_variant["source_cfg"], str(source_cfg))
                self.assertEqual(legacy.rng.getstate(), collecting.rng.getstate())


if __name__ == "__main__":
    unittest.main()
