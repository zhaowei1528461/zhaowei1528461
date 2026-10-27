"""Metrics, honest candidates and complete train/checkpoint/evaluate workflows."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'embodied/envs'))
try:
    import torch
except ImportError:
    torch = None
try:
    metrics = importlib.import_module('trajectory_metrics')
    candidates = importlib.import_module('diffusion_candidate_adapter')
    training = importlib.import_module('trajectory_training') if torch is not None else None
except ImportError:
    metrics = candidates = training = None


def meta_fixture():
    anchor = np.array([100., 200., 2., 0., 0., 0., 4.5, 1.8])
    return {'episode_id': 'real-recorded-id', 'scenario_id': 'scene-1', 'parent_group_id': 'scenario:scene-1',
            'target_id': 'npc', 'source_path': '/recorded/episode.npz', 'source_sha256': 'a'*64,
            'window_index': 0, 'anchor_timestamp': 2., 'world_anchor': anchor,
            'target_anchor_world': anchor, 'target_anchor_local': np.array([0., 0., 2., 0., 0., 0., 4.5, 1.8]),
            'target_history_world': np.stack((anchor, anchor)), 'ego_history_world': np.stack((anchor, anchor)),
            'position_reference': 'front_bumper_center', 'split': 'val'}


class TrajectoryMetricsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(metrics, 'Trajectory evaluation and candidate adapters are missing')

    def test_metric_means_oracle_best_and_diversity_have_explicit_definitions(self):
        truth = np.zeros((1, 2, 2))
        predictions = np.array([[[[1., 0.], [1., 0.]], [[3., 0.], [3., 0.]]]])
        result = metrics.displacement_metrics(predictions, truth)
        self.assertEqual(result['mean_ADE_m'], 2.)
        self.assertEqual(result['mean_FDE_m'], 2.)
        self.assertEqual(result['minADE_m'], 1.)
        self.assertEqual(result['minFDE_m'], 1.)
        self.assertEqual(result['endpoint_diversity_m'], 2.)

    def test_constant_velocity_baseline_and_sampled_motion_use_anchor_and_dt(self):
        baseline = metrics.constant_velocity([meta_fixture()], future=3, dt=.2)
        np.testing.assert_allclose(baseline, [[[.4, 0.], [.8, 0.], [1.2, 0.]]])
        world = baseline[0]+np.array([100., 200.])
        states = metrics.states_from_xy(world, meta_fixture()['target_anchor_world'], .2)
        np.testing.assert_allclose(states[1:, 2:4], [[2., 0.]]*3, atol=1e-12)
        self.assertTrue(metrics.motion_valid(states, .2))
        world[0] += 100.
        self.assertFalse(metrics.motion_valid(metrics.states_from_xy(world, meta_fixture()['target_anchor_world'], .2), .2))

    def test_candidate_preserves_recorded_history_and_reference_without_simulator_claims(self):
        metadata = meta_fixture()
        world = np.array([[100.4, 200.], [100.8, 200.]])
        result = candidates.make_candidate(metadata, world, .2, {'normalizer_id': 'n', 'checkpoint_sha256': 'b'*64}, seed=1)
        self.assertFalse(result['simulation_verified'])
        self.assertFalse(result['generated_target_history'])
        self.assertEqual(result['position_reference'], 'front_bumper_center')
        self.assertEqual(result['source_type'], 'generated_candidate')
        np.testing.assert_array_equal(result['target_history']['states'], metadata['target_history_world'])
        self.assertEqual(len(result['target_trajectory']['states']), 3)
        self.assertIsNone(result['risk'])
        with tempfile.TemporaryDirectory() as directory:
            files = candidates.save_candidate(result, Path(directory)/'candidate')
            restored = json.loads(Path(files['json']).read_text())
            with np.load(files['npz'], allow_pickle=False) as archive:
                np.testing.assert_allclose(archive['target_states'], result['target_trajectory']['states'])
                self.assertFalse(json.loads(str(archive['metadata_json'].item()))['simulation_verified'])
            self.assertEqual(restored['candidate_id'], result['candidate_id'])


@unittest.skipIf(torch is None, 'PyTorch required for training/checkpoint workflows')
class TrajectoryTrainingTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, 'Standalone trajectory training pipeline is missing')

    def test_synthetic_cli_optimizes_saves_loads_and_evaluates_without_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            command = [sys.executable, str(ROOT/'embodied/envs/train_trajectory_diffusion.py'),
                '--synthetic-smoke', '--output', str(work/'train'), '--epochs', '1', '--max-steps', '2',
                '--batch-size', '4', '--base-dim', '8', '--dim-mults', '1', '2', '--condition-dim', '16',
                '--diffusion-steps', '8', '--sampling-steps', '4', '--samples', '2', '--train-limit', '16',
                '--val-limit', '4', '--torch-threads', '1']
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['optimization_steps'], 2)
            self.assertEqual(report['dataset_source'], 'synthetic_kinematics')
            self.assertTrue(report['checkpoint_roundtrip_verified'])
            self.assertEqual(report['validation']['evaluated_windows'], 4)
            self.assertIsNone(report['validation']['road_valid_rate'])
            checkpoint = work/'train/latest.pt'
            model, payload = training.load_checkpoint(checkpoint, 'cpu')
            self.assertEqual(payload['normalizer']['coordinate_frame'], 'ego-centric')
            with self.assertRaisesRegex(ValueError, 'fingerprint|dataset'):
                training.load_checkpoint(checkpoint, 'cpu', expected_fingerprint={'wrong': 'dataset'})
            evaluation = subprocess.run([sys.executable, str(ROOT/'embodied/envs/sample_trajectories.py'),
                '--checkpoint', str(checkpoint), '--synthetic-smoke', '--output', str(work/'samples'),
                '--split', 'val', '--limit', '3', '--samples', '2', '--sampling-steps', '4',
                '--export-limit', '1', '--torch-threads', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            self.assertEqual(evaluation.returncode, 0, evaluation.stdout+evaluation.stderr)
            sampled = json.loads(evaluation.stdout)
            self.assertEqual(sampled['evaluated_windows'], 3)
            self.assertEqual(len(sampled['exported_candidates']), 1)
            candidate = json.loads(Path(sampled['exported_candidates'][0]['json']).read_text())
            sampling = candidate['generation_metadata'].get('sampling_context')
            self.assertIsNotNone(sampling, 'Candidate lacks its random-number batch context')
            self.assertEqual(sampling['num_samples'], 2)
            self.assertEqual(sampling['batch_size'], 3)
            self.assertEqual(sampling['batch_row'], 0)

    def test_canonical_frozen_loader_can_train_and_validate_without_refitting_normalizer(self):
        from test_trajectory_dataset import prepared_fixture
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            normalizer_before = (work/'windows/normalizer.json').read_bytes()
            result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/train_trajectory_diffusion.py'),
                '--dataset-manifest', str(manifest), '--output', str(work/'train'), '--epochs', '1',
                '--max-steps', '1', '--train-limit', '4', '--val-limit', '2', '--batch-size', '2',
                '--base-dim', '8', '--dim-mults', '1', '2', '--condition-dim', '16',
                '--diffusion-steps', '8', '--sampling-steps', '4', '--samples', '2', '--torch-threads', '1'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['optimization_steps'], 1)
            self.assertEqual(report['validation']['evaluated_windows'], 2)
            self.assertTrue(report['validation']['limited_evaluation'])
            self.assertEqual(normalizer_before, (work/'windows/normalizer.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
