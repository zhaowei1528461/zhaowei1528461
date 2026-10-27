"""Phase controls using real wrappers/miners without starting SUMO or training."""

import importlib
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import embodied
import yaml

from embodied.envs.danger_scenarios import StructuredDangerScenarioMiner


ROOT = Path(__file__).resolve().parents[1]


class _MinerBackend:
    """Replace only the SUMO constructor; retain the real mining behavior."""

    def __init__(self, danger_config, danger_output_dir, worker_index, **kwargs):
        self.danger_miner = StructuredDangerScenarioMiner(
            danger_config, danger_output_dir, worker_index=worker_index)

    def set_danger_mining(self, active, **kwargs):
        return self.danger_miner.set_active(active, **kwargs)


def make_wrapper(directory, danger, mode='train'):
    backend_module = types.ModuleType('embodied.envs.sumo_merge')
    backend_module.Sumoenv = _MinerBackend
    name = 'embodied.envs._danger_phase_sumo'
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'embodied' / 'envs' / 'sumo.py')
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
            'sys.modules', {'embodied.envs.sumo_merge': backend_module}):
        spec.loader.exec_module(module)
    return module.SumoEnv(
        danger=danger, env_mode=mode, danger_output_dir=directory)


def record_collision(miner):
    return miner.record_episode(
        source_cfg='phase_test.sumocfg', length=1,
        total_cost=1.0, max_cost=1.0, min_ttc=0.5,
        terminal_reason='collision', critical_state=[0.0] * 31,
        replay_trajectory={'frames': [{'vehicles': [
            {'id': 'Auto', 'x': 0.0, 'y': 0.0},
            {'id': 'npc_1', 'x': 10.0, 'y': 0.0},
        ]}]})


class DangerPhaseTest(unittest.TestCase):

    def danger_config(self, **overrides):
        config = {
            'enabled': True, 'warmup_episodes': 0,
            'collect_episodes': True, 'update_library': True,
        }
        config.update(overrides)
        return config

    def enable_posttraining(self, wrapper, freeze):
        module = importlib.import_module('embodied.run.train_eval')
        enable = getattr(module, '_enable_danger_posttraining', None)
        self.assertTrue(
            callable(enable), 'Post-training activation needs one shared helper')
        return enable(embodied.BatchEnv([wrapper], parallel=False), freeze)

    def test_eval_sweep_records_events_but_never_collects_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.danger_config()
            wrapper = make_wrapper(directory, config, mode='eval')
            miner = wrapper._env.danger_miner
            self.assertFalse(miner.config['collect_episodes'])
            self.assertTrue(config['collect_episodes'])
            wrapper.set_danger_mining(True, record_events=True)
            self.assertTrue(miner.recording)
            self.assertFalse(miner.collecting)
            record_collision(miner)
            self.assertEqual(len(miner.library), 1)
            self.assertTrue(miner.output_path.exists())

    def test_wrapper_collects_independently_of_event_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(
                directory, self.danger_config(collect_episodes=False))
            wrapper.set_danger_mining(
                True, record_events=False, collect_episodes=True,
                update_library=False)
            miner = wrapper._env.danger_miner
            self.assertTrue(miner.collecting)
            self.assertFalse(miner.recording)
            record_collision(miner)
            self.assertEqual(len(miner.library), 0)
            self.assertFalse(miner.output_path.exists())

    def test_old_wrapper_call_keeps_the_legacy_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(
                directory, self.danger_config(collect_episodes=False))
            wrapper.set_danger_mining(True, record_events=True)
            miner = wrapper._env.danger_miner
            self.assertTrue(miner.active)
            self.assertTrue(miner.recording)
            self.assertFalse(miner.collecting)
            record_collision(miner)
            self.assertEqual(len(miner.library), 1)

    def test_bootstrap_deactivation_stops_independent_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(directory, self.danger_config())
            wrapper.set_danger_mining(True, collect_episodes=True)
            self.assertTrue(wrapper._env.danger_miner.collecting)
            wrapper.set_danger_mining(False)
            self.assertFalse(wrapper._env.danger_miner.collecting)
            self.assertFalse(wrapper._env.danger_miner.recording)

    def test_frozen_posttraining_collects_without_mutating_the_old_library(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(directory, self.danger_config())
            self.enable_posttraining(wrapper, freeze=True)
            miner = wrapper._env.danger_miner
            self.assertTrue(miner.collecting)
            self.assertFalse(miner.recording)
            self.assertFalse(miner.update_library)
            record_collision(miner)
            self.assertEqual(len(miner.library), 0)
            self.assertFalse(miner.output_path.exists())

    def test_unfrozen_posttraining_respects_disabled_online_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(
                directory, self.danger_config(update_library=False))
            self.enable_posttraining(wrapper, freeze=False)
            miner = wrapper._env.danger_miner
            self.assertTrue(miner.collecting)
            self.assertTrue(miner.recording)
            self.assertFalse(miner.update_library)
            record_collision(miner)
            self.assertEqual(len(miner.library), 0)
            self.assertTrue(miner.output_path.exists())

    def test_unfrozen_legacy_posttraining_updates_the_library(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = make_wrapper(
                directory, self.danger_config(collect_episodes=False))
            self.enable_posttraining(wrapper, freeze=False)
            miner = wrapper._env.danger_miner
            self.assertFalse(miner.collecting)
            self.assertTrue(miner.recording)
            self.assertTrue(miner.update_library)
            record_collision(miner)
            self.assertEqual(len(miner.library), 1)


class DangerConfigurationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with (ROOT / 'configs.yaml').open(encoding='utf-8') as file:
            cls.configs = yaml.safe_load(file)

    def parse_configs(self, argv):
        parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
        config = embodied.Config(self.configs['defaults'])
        for name in parsed.configs:
            config = config.update(self.configs[name])
        return embodied.Flags(config).parse(other)

    def test_episode_overlay_changes_only_independent_collection(self):
        self.assertIn('sumo_long_tail_data', self.configs)
        legacy = self.parse_configs(
            ['--configs', 'osrp_sumo_lag_vector_danger'])
        collecting = self.parse_configs([
            '--configs', 'osrp_sumo_lag_vector_danger', 'sumo_long_tail_data'])
        self.assertFalse(legacy.env.sumo.danger.collect_episodes)
        self.assertTrue(collecting.env.sumo.danger.collect_episodes)
        self.assertTrue(collecting.env.sumo.danger.update_library)
        changed = {
            key for key, value in legacy.flat.items()
            if value != collecting.flat[key]
        }
        self.assertEqual(changed, {'env.sumo.danger.collect_episodes'})

    def test_new_flags_are_supported_in_default_and_danger_configs(self):
        for preset in ('defaults', 'osrp_sumo_lag_vector_danger'):
            with self.subTest(preset=preset):
                danger = self.configs[preset]['env']['sumo']['danger']
                self.assertIn('collect_episodes', danger)
                self.assertIn('update_library', danger)
                config = self.parse_configs([
                    '--configs', preset,
                    '--env.sumo.danger.collect_episodes', 'True',
                    '--env.sumo.danger.update_library', 'False'])
                self.assertTrue(config.env.sumo.danger.collect_episodes)
                self.assertFalse(config.env.sumo.danger.update_library)


if __name__ == '__main__':
    unittest.main()
