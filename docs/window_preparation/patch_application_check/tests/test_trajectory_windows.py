"""Frozen-split coverage and train-only local normalization behavior."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'embodied/envs'))
import scenario_data as data
import prepare_scenario_splits as split_tool

try:
    tool = importlib.import_module('prepare_trajectory_windows')
except ImportError:
    tool = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(i=0, count=7):
    frames = []
    for step in range(count):
        t = .2*step
        frames.append({'sim_time': t, 'state': [0.]*31, 'action': [0.],
                       'vehicles': [dict(id='Auto', pos=[100.+4*t, 0.], v=4., yaw=0., a=0., length=4.5, width=1.8),
                                    dict(id='npc', pos=[110.+2*t, 0.], v=2., yaw=0., a=0., length=4.5, width=1.8)]})
    return {'episode_id': 'e%d' % i, 'scenario_id': 's%d' % i, 'source_type': 'sumo_original',
            'metadata': {'prediction_source': 'sumo_execution', 'collection_complete': True,
                         'position_reference': 'front_bumper_center'},
            'replay_trajectory': {'frames': frames}}


def pool_fixture(work, alter=None):
    pool = work/'pool'
    for i in range(8):
        record = sample(i)
        if alter:
            alter(i, record)
        data.write_episode(data.episode_from_record(record), pool)
    report = split_tool.prepare_pool(pool, work/'split')
    assert report['exit_code'] == 0
    return work/'split/split_manifest.json'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class WindowIteratorTest(unittest.TestCase):
    def test_iterator_yields_independent_windows_with_original_timing_and_masks(self):
        self.assertTrue(hasattr(data, 'iter_windows'), 'Streaming window iterator is missing')
        episode = data.episode_from_record(sample())
        iterator = data.iter_windows(episode, history=3, future=2, dt=.2)
        self.assertIs(iter(iterator), iterator)
        first = next(iterator)
        np.testing.assert_allclose(first.timestamps, [0., .2, .4, .6, .8])
        self.assertAlmostEqual(first.anchor_timestamp, .4)
        self.assertTrue(first.ego_mask.all())
        self.assertTrue(first.npc_mask.all())
        first.history_ego[:] = -123.
        second = next(iterator)
        self.assertAlmostEqual(second.anchor_timestamp, .6)
        self.assertGreater(second.history_ego[0, 0], 100.)
        self.assertEqual(len(list(iterator)), 1)
        self.assertIsInstance(data.make_windows(episode, history=3, future=2, dt=.2), list)


class TrajectoryPreparationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(tool, 'Frozen-split trajectory preparation tool is missing')

    def prepare(self, manifest, output, **kwargs):
        return tool.prepare_windows(manifest, output, history=3, future=2, dt=.2, **kwargs)

    def test_frozen_membership_and_analytic_local_train_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            original = read_json(manifest)
            before = {str(p): sha(p) for p in (work/'pool').glob('*.npz')}
            before[str(manifest)] = sha(manifest)
            report = self.prepare(manifest, work/'windows')
            self.assertEqual(report['exit_code'], 0)
            self.assertEqual({k: v['eligible_windows'] for k, v in report['splits'].items()},
                             {'train': 18, 'val': 3, 'test': 3})
            self.assertEqual(report['splits']['train']['complete_target_trajectories'], 18)
            self.assertTrue(report['leakage_check']['passed'])
            lines = [json.loads(line) for line in (work/'windows/window_index.jsonl').read_text().splitlines()]
            self.assertEqual(len(lines), 24)
            expected = {r['episode_id']: name for name, rows in original['splits'].items() for r in rows}
            for line in lines:
                self.assertEqual(line['split'], expected[line['episode_id']])
                self.assertEqual(line['eligible_target_ids'], ['npc'])
            normalizer = data.ScenarioNormalizer.load(work/'windows/normalizer.json')
            self.assertEqual(normalizer.coordinate_frame, 'ego-centric')
            self.assertEqual(normalizer.count, 144)  # 18*(3 ego history + 5 target samples).
            self.assertAlmostEqual(normalizer.mean[0], 5.2)
            self.assertEqual(normalizer.std[1], 1.)
            self.assertEqual(set(normalizer.training_episode_ids), {'e0', 'e1', 'e2', 'e3', 'e6', 'e7'})
            for path, identity in before.items():
                self.assertEqual(sha(path), identity)
            dataset = read_json(work/'windows/window_dataset_manifest.json')
            self.assertEqual(dataset['source_manifest_sha256'], sha(manifest))
            self.assertFalse(dataset['tensors_materialized'])

    def test_extreme_held_out_states_do_not_affect_fitted_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            first = pool_fixture(work/'first')
            def alter(i, record):
                if i in (4, 5):
                    for frame in record['replay_trajectory']['frames']:
                        frame['vehicles'][1].update(pos=[-1e9, 1e9], length=1000., v=500.)
            second = pool_fixture(work/'second', alter=alter)
            self.prepare(first, work/'out1')
            self.prepare(second, work/'out2')
            a = data.ScenarioNormalizer.load(work/'out1/normalizer.json')
            b = data.ScenarioNormalizer.load(work/'out2/normalizer.json')
            np.testing.assert_array_equal(a.mean, b.mean)
            np.testing.assert_array_equal(a.std, b.std)
            self.assertEqual(a.count, b.count)

    def test_gap_coverage_is_counted_and_incomplete_targets_are_not_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                for step, frame in enumerate(record['replay_trajectory']['frames']):
                    if step != 2:
                        frame['vehicles'].append(dict(frame['vehicles'][1], id='history_gap'))
                    if step < 5:
                        frame['vehicles'].append(dict(frame['vehicles'][1], id='future_gap'))
            manifest = pool_fixture(work, alter=alter)
            report = self.prepare(manifest, work/'windows')
            rows = [json.loads(line) for line in (work/'windows/window_index.jsonl').read_text().splitlines()]
            episode_rows = [r for r in rows if r['episode_id'] == 'e0']
            self.assertEqual([r['eligible_target_ids'] for r in episode_rows],
                             [['future_gap', 'npc'], ['npc'], ['npc']])
            train = report['splits']['train']
            self.assertEqual(train['anchor_npc_samples'], 48)  # 6 episodes*(2+3+3).
            self.assertEqual(train['complete_target_trajectories'], 24)
            self.assertEqual(train['contiguous_history_frames']['1'], 6)
            self.assertEqual(train['contiguous_history_frames']['2'], 6)
            self.assertEqual(train['contiguous_future_frames']['0'], 6)
            self.assertEqual(train['contiguous_future_frames']['1'], 6)

    def test_invalid_ego_history_excludes_the_window_without_zero_filling(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                record['replay_trajectory']['frames'][2]['vehicles'][0].pop('a')
            manifest = pool_fixture(work, alter=alter)
            report = self.prepare(manifest, work/'windows')
            self.assertEqual(report['exit_code'], 2)
            self.assertEqual(report['splits']['train']['ego_history_incomplete_windows'], 18)
            self.assertEqual(report['splits']['train']['eligible_windows'], 0)
            self.assertFalse((work/'windows/normalizer.json').exists())

    def test_populated_but_short_split_is_reported_as_incomplete_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            report = tool.prepare_windows(manifest, work/'windows')  # Default 41 frames > 7.
            self.assertEqual(report['exit_code'], 2)
            self.assertFalse(report['normalizer']['fitted'])
            self.assertTrue(all(v['episodes_without_eligible_windows'] == v['episodes'] for v in report['splits'].values()))
            self.assertFalse((work/'windows/normalizer.json').exists())

    def test_forged_audit_does_not_allow_cross_split_parent_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            payload = read_json(manifest)
            payload['splits']['val'][0]['parent_scenario_id'] = payload['splits']['train'][0]['scenario_id']
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ancestry|overlap|leak'):
                self.prepare(manifest, work/'windows')

    def test_source_hash_change_prevents_normalizer_and_reports_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            with (work/'pool/e0.npz').open('ab') as stream:
                stream.write(b'changed after manifest')
            report = self.prepare(manifest, work/'windows')
            self.assertEqual(report['exit_code'], 1)
            self.assertEqual(len(report['rejected_episodes']), 1)
            self.assertIn('hash', report['rejected_episodes'][0]['error'].lower())
            self.assertFalse((work/'windows/normalizer.json').exists())

    def test_loaded_episode_identity_must_match_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            payload = read_json(manifest)
            payload['splits']['train'][0]['episode_id'] = 'wrong-episode'
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            report = self.prepare(manifest, work/'windows')
            self.assertEqual(report['exit_code'], 1)
            self.assertFalse((work/'windows/normalizer.json').exists())

    def test_relocated_pool_uses_relative_paths_and_verifies_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            relocated = work/'relocated'
            shutil.copytree(work/'pool', relocated)
            payload = read_json(manifest)
            payload['input_root'] = '/unavailable/old/root'
            for entries in payload['splits'].values():
                for entry in entries:
                    entry['input_source'] = '/unavailable/old/root/'+entry['relative_path']
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            report = self.prepare(manifest, work/'windows', input_root=relocated)
            self.assertEqual(report['exit_code'], 0)

    def test_relative_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            payload = read_json(manifest)
            payload['splits']['train'][0]['relative_path'] = '../outside.npz'
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'root|relative|escape'):
                self.prepare(manifest, work/'windows', input_root=work/'pool')

    def test_duplicate_episode_membership_is_rejected_even_with_passed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            payload = read_json(manifest)
            payload['splits']['val'].append(payload['splits']['train'][0])
            manifest.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate|Duplicate|overlap'):
                self.prepare(manifest, work/'windows')

    def test_previous_episode_is_released_before_next_load(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            refs = []
            original = tool.load_episode
            def tracked(path):
                if refs:
                    self.assertIsNone(refs[-1](), 'Full previous episode retained')
                episode = original(path)
                refs.append(weakref.ref(episode))
                return episode
            with mock.patch.object(tool, 'load_episode', side_effect=tracked):
                report = self.prepare(manifest, work/'windows')
            self.assertEqual(report['exit_code'], 0)
            self.assertIsNone(refs[-1]())

    def test_change_before_second_pass_cannot_produce_a_stale_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            changed = []
            def progress(phase, number, total, path, status):
                if phase == 'normalizer' and number == 0:
                    with (work/'pool/e0.npz').open('ab') as stream:
                        stream.write(b'changed between passes')
                    changed.append(True)
            report = self.prepare(manifest, work/'windows', progress=progress)
            self.assertTrue(changed)
            self.assertEqual(report['exit_code'], 1)
            self.assertFalse((work/'windows/normalizer.json').exists())

    def test_late_held_out_or_train_changes_prevent_publishing_success(self):
        for phase_end, episode_id in ((False, 'e5'), (True, 'e0')):
            with self.subTest(after_last_train_pass=phase_end), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                manifest = pool_fixture(work)
                changed = []
                def progress(phase, number, total, path, status):
                    trigger = number == total if phase_end else number == 0
                    if phase == 'normalizer' and trigger:
                        with (work/('pool/'+episode_id+'.npz')).open('ab') as stream:
                            stream.write(b'late source mutation')
                        changed.append(True)
                report = self.prepare(manifest, work/'windows', progress=progress)
                self.assertTrue(changed)
                self.assertEqual(report['exit_code'], 1)
                self.assertFalse((work/'windows/normalizer.json').exists())
                self.assertEqual(report['rejected_episodes'][0]['episode_id'], episode_id)
                self.assertEqual(report['rejected_episodes'][0]['phase'], 'source_check')

    def test_local_rotation_and_wrapped_yaw_statistics_are_finite_and_invertible(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                for j, frame in enumerate(record['replay_trajectory']['frames']):
                    frame['vehicles'][1]['yaw'] = math.pi-.1 if j % 2 else -math.pi+.1
            manifest = pool_fixture(work, alter=alter)
            self.prepare(manifest, work/'windows')
            normalizer = data.ScenarioNormalizer.load(work/'windows/normalizer.json')
            # Each window has 3 ego yaw=0 values and 5 NPC yaw values near +/-pi.
            angles = np.array([0.]*9 + [-math.pi+.1, math.pi-.1, -math.pi+.1, math.pi-.1, -math.pi+.1]
                              + [math.pi-.1, -math.pi+.1, math.pi-.1, -math.pi+.1, math.pi-.1]
                              + [-math.pi+.1, math.pi-.1, -math.pi+.1, math.pi-.1, -math.pi+.1])
            mean_yaw = math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
            residual = np.arctan2(np.sin(angles-mean_yaw), np.cos(angles-mean_yaw))
            self.assertAlmostEqual(normalizer.mean[4], mean_yaw)
            self.assertAlmostEqual(normalizer.std[4], residual.std())
            values = np.array([[3., -4., 2., 0., -math.pi+.1, 0., 4.5, 1.8]])
            restored = normalizer.inverse_transform(normalizer.transform(values))
            np.testing.assert_allclose(restored[:, [0, 1, 2, 3, 5, 6, 7]], values[:, [0, 1, 2, 3, 5, 6, 7]])
            np.testing.assert_allclose(np.sin(restored[:, 4]), np.sin(values[:, 4]))

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            output = work/'windows'
            output.mkdir()
            (output/'keep.txt').write_text('unchanged')
            with self.assertRaisesRegex(ValueError, 'new|empty'):
                self.prepare(manifest, output)
            self.assertEqual((output/'keep.txt').read_text(), 'unchanged')

    def test_cli_emits_progress_and_structured_report(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = pool_fixture(work)
            result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/prepare_trajectory_windows.py'),
                '--manifest', str(manifest), '--output', str(work/'windows'), '--history', '3',
                '--future', '2', '--dt', '.2', '--progress-every', '2'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('[windows', result.stderr)
            self.assertIn('[normalizer', result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['splits']['train']['eligible_windows'], 18)


if __name__ == '__main__':
    unittest.main()
