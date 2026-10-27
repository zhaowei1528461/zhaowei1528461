"""Exercise real collection methods without starting external SUMO."""
import ast
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from embodied.envs.danger_scenarios import StructuredDangerScenarioMiner


def load_collection_env(traci):
    path = Path(__file__).resolve().parents[1] / 'embodied/envs/sumo_merge.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'Sumoenv')
    names = {
        '_begin_episode_stats', '_capture_replay_frame',
        '_capture_initial_episode_frame', '_build_replay_trajectory',
        '_record_episode_step', '_finish_active_episode',
        '_write_collected_episode', 'set_danger_mining',
    }
    cls.body = [node for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    selected = ast.Module(body=[cls], type_ignores=[])
    namespace = {
        'np': np, 'math': math, 'traci': traci,
        '__name__': 'embodied.envs.sumo_merge', '__package__': 'embodied.envs',
    }
    exec(compile(ast.fix_missing_locations(selected), str(path), 'exec'), namespace)
    return namespace['Sumoenv']


class SumoEpisodeCollectionTest(unittest.TestCase):
    def make_env(self, directory, collect=True):
        clock = SimpleNamespace(time=0.0)
        traci = SimpleNamespace(
            simulation=SimpleNamespace(getTime=lambda: clock.time),
            vehicle=SimpleNamespace(getIDList=lambda: ['Auto', 'npc1']))
        env = load_collection_env(traci)()
        danger_dir = Path(directory) / 'danger_scenarios'
        env.danger_miner = StructuredDangerScenarioMiner(
            {'enabled': True, 'warmup_episodes': 0,
             'collect_episodes': collect, 'update_library': False},
            danger_dir, worker_index=0)
        env.danger_miner.set_active(True, record_events=False)
        env.path = str(Path(directory) / 'scene.sumocfg')
        env.worker_index = 0
        env.env_mode = 'train'
        env.episode = 7
        env.egoid = 'Auto'
        env._sumo_connection_lost = False
        env._test_traci = traci
        env._mark_sumo_connection_lost = lambda *args: setattr(
            env, '_sumo_connection_lost', True)
        env.ttc_compute = lambda state: (1.0, 0.1)

        def snapshot(veh_id):
            return {
                'id': veh_id, 'type': 'vehicle', 'type_id': 'car',
                'pos': [clock.time * 2 + (8 if veh_id == 'npc1' else 0), 0],
                'v': 2.0, 'a': 0.0, 'yaw': 0.0, 'angle': 90.0,
                'road_id': 'road', 'lane_id': 'road_0', 'lane_index': 0,
                'lane_position': clock.time * 2, 'route_id': 'route',
                'route_edges': ['road'], 'length': 4.5, 'width': 1.8,
            }

        env._get_vehicle_snapshot = snapshot
        return env, clock

    def run_episode(self, env, clock, terminal='collision', advances=True):
        env._begin_episode_stats({
            'mode': 'intent_attack', 'kind': 'intent_attack',
            'target_id': 'npc1', 'used_danger_mode': 'intent_attack'})
        state = np.zeros(31, np.float32)
        if hasattr(env, '_capture_initial_episode_frame'):
            env._capture_initial_episode_frame(state)
        clock.time = 0.2
        env._record_episode_step(state, 0.0, action=[0.25], acc=0.5)
        clock.time = 0.4 if advances else 0.2
        env._record_episode_step(
            state, 1.0, terminal_reason=terminal, action=[-0.5], acc=-1.0)
        env._finish_active_episode()

    def test_frozen_library_collects_final_execution_once(self):
        # Losing the collection branch while freeze is enabled loses the file.
        with tempfile.TemporaryDirectory() as directory:
            env, clock = self.make_env(directory)
            rng_before = env.danger_miner.rng.getstate()
            self.run_episode(env, clock)
            env._finish_active_episode()  # A subsequent reset also finalizes.
            files = list((Path(directory) / 'posttrain_episode_pool').glob('*.npz'))
            self.assertEqual(len(files), 1)
            self.assertEqual(env.danger_miner.library, [])
            self.assertEqual(env.danger_miner.rng.getstate(), rng_before)
            self.assertFalse(env.danger_miner.output_path.exists())
            from embodied.envs.scenario_data import load_episode
            episode = load_episode(files[0])
            np.testing.assert_allclose(episode.timestamps, [0.0, 0.2, 0.4])
            np.testing.assert_allclose(episode.ego_actions[:2, 0], [0.25, -0.5])
            self.assertEqual(episode.source_type, 'sumo_attack')
            self.assertEqual(episode.metadata['generated_variant']['target_id'], 'npc1')
            self.assertEqual(episode.metadata['event_type'], 'collision')
            self.assertIsNone(episode.seed)
            self.assertIsNone(episode.metadata['execution_valid'])

    def test_disabled_collection_writes_no_new_data(self):
        with tempfile.TemporaryDirectory() as directory:
            env, clock = self.make_env(directory, collect=False)
            self.run_episode(env, clock)
            self.assertFalse((Path(directory) / 'posttrain_episode_pool').exists())
            self.assertEqual(env.danger_miner.library, [])

    def test_disabled_collection_finalizes_legacy_log_once(self):
        with tempfile.TemporaryDirectory() as directory:
            env, clock = self.make_env(directory, collect=False)
            env.danger_miner.set_active(True, record_events=True)
            self.run_episode(env, clock)
            env._finish_active_episode()
            self.assertFalse(env._episode_active)
            self.assertEqual(env.danger_miner.episodes_seen, 1)
            self.assertEqual(len(env.danger_miner.output_path.read_text().splitlines()), 1)

    def test_write_failure_still_finalizes_legacy_log_once(self):
        with tempfile.TemporaryDirectory() as directory:
            env, clock = self.make_env(directory)
            env.danger_miner.set_active(True, record_events=True)
            with mock.patch('embodied.envs.scenario_data.write_episode',
                            side_effect=OSError('fixture write failure')):
                self.run_episode(env, clock)
                env._finish_active_episode()
            self.assertFalse(env._episode_active)
            self.assertEqual(env.danger_miner.episodes_seen, 1)
            self.assertEqual(len(env.danger_miner.output_path.read_text().splitlines()), 1)

    def test_terminal_capture_failure_preserves_known_execution_prefix(self):
        class FatalTraCIError(Exception):
            pass

        def fail():
            raise FatalTraCIError('fixture connection closed')

        for failing_api in ('vehicle_ids', 'time'):
            with self.subTest(failing_api=failing_api), tempfile.TemporaryDirectory() as directory:
                env, clock = self.make_env(directory)
                env._begin_episode_stats({'used_danger_mode': 'intent_attack'})
                state = np.zeros(31, np.float32)
                env._capture_initial_episode_frame(state)
                clock.time = 0.2
                env._record_episode_step(state, 0.0, action=[0.25])
                if failing_api == 'vehicle_ids':
                    env._test_traci.vehicle.getIDList = fail
                env._test_traci.simulation.getTime = fail
                env._record_episode_step(
                    state, 1.0, terminal_reason='collision', action=[-0.5])
                env._finish_active_episode()
                files = list((Path(directory) / 'posttrain_episode_pool').glob('*.npz'))
                self.assertEqual(len(files), 1)
                from embodied.envs.scenario_data import load_episode
                episode = load_episode(files[0])
                np.testing.assert_allclose(episode.timestamps, [0.0, 0.2])
                self.assertAlmostEqual(float(episode.ego_actions[0, 0]), 0.25)
                self.assertEqual(episode.metadata['event_type'], 'execution_error')
                self.assertEqual(episode.metadata['terminal_reason'], 'collision')
                self.assertTrue(episode.metadata['dropped_connection_terminal_frame'])
                self.assertFalse(episode.ego_action_mask[-1])

    def test_non_advanced_terminal_frame_preserves_applied_action(self):
        with tempfile.TemporaryDirectory() as directory:
            env, clock = self.make_env(directory)
            self.run_episode(env, clock, terminal='goal', advances=False)
            files = list((Path(directory) / 'posttrain_episode_pool').glob('*.npz'))
            self.assertEqual(len(files), 1)
            from embodied.envs.scenario_data import load_episode
            episode = load_episode(files[0])
            np.testing.assert_allclose(episode.timestamps, [0.0, 0.2])
            self.assertAlmostEqual(float(episode.ego_actions[0, 0]), 0.25)
            self.assertFalse(episode.ego_action_mask[-1])
            self.assertEqual(episode.metadata['collapsed_terminal_frames'], 1)


if __name__ == '__main__':
    unittest.main()
