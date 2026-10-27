"""Frozen-index loading, history-only conditions and world coordinate behavior."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'embodied/envs'))
import prepare_trajectory_windows as prep
from test_trajectory_windows import pool_fixture, read_json

try:
    subject = importlib.import_module('trajectory_dataset')
except ImportError:
    subject = None


def prepared_fixture(work, alter=None):
    split = pool_fixture(work, alter=alter)
    prep.prepare_windows(split, work/'windows', history=3, future=2, dt=.2)
    return work/'windows/window_dataset_manifest.json'


class TrajectoryDatasetTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'Fixed-index trajectory dataset is missing')

    def test_shapes_masks_labels_and_world_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            dataset = subject.TrajectoryDataset(manifest, 'val', max_neighbors=3)
            self.assertEqual(len(dataset), 3)
            item = dataset[0]
            self.assertEqual(item['condition']['ego_history'].shape, (3, 8))
            self.assertEqual(item['condition']['target_history'].shape, (3, 8))
            self.assertEqual(item['condition']['neighbor_history'].shape, (3, 3, 8))
            self.assertFalse(item['condition']['neighbor_mask'].any())
            self.assertTrue(item['future_mask'].all())
            self.assertEqual(item['future_xy'].shape, (2, 2))
            for value in item['condition'].values():
                self.assertTrue(np.isfinite(value).all())
            np.testing.assert_allclose(dataset.xy_to_world(item['future_xy'], item['meta']),
                                       [[111.2, 0.], [111.6, 0.]], atol=2e-6)
            self.assertEqual(item['meta']['episode_id'], 'e5')
            self.assertEqual(item['meta']['target_id'], 'npc')
            self.assertEqual(item['meta']['position_reference'], 'front_bumper_center')

    def test_rotated_anchor_is_inverted_without_reusing_world_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                for step, frame in enumerate(record['replay_trajectory']['frames']):
                    frame['vehicles'][0].update(pos=[100., .8*step], yaw=math.pi/2)
                    frame['vehicles'][1].update(pos=[103., 4.+.4*step], yaw=math.pi/2)
            manifest = prepared_fixture(work, alter=alter)
            dataset = subject.TrajectoryDataset(manifest, 'val')
            item = dataset[0]
            np.testing.assert_allclose(dataset.xy_to_world(item['future_xy'], item['meta']),
                                       [[103., 5.2], [103., 5.6]], atol=2e-6)
            np.testing.assert_allclose(item['meta']['target_anchor_local'][:2], [3.2, -3.], atol=1e-12)

    def test_future_poisoning_changes_only_labels_not_conditions_or_target_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            first = subject.TrajectoryDataset(manifest, 'val')[0]
            original = subject._load_checked
            def poisoned(entry):
                episode, fingerprint = original(entry)
                episode.ego_state[3:, :4] += 10000.
                episode.npc_states[3:, :, :4] -= 10000.
                return episode, fingerprint
            with mock.patch.object(subject, '_load_checked', side_effect=poisoned):
                second = subject.TrajectoryDataset(manifest, 'val')[0]
            self.assertEqual(first['meta']['target_id'], second['meta']['target_id'])
            for name in first['condition']:
                np.testing.assert_array_equal(first['condition'][name], second['condition'][name])
            self.assertFalse(np.allclose(first['future_xy'], second['future_xy']))

    def test_training_target_sampling_is_seeded_and_held_out_is_history_based(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                for frame in record['replay_trajectory']['frames']:
                    other = dict(frame['vehicles'][1], id='farther', pos=[140., 0.])
                    frame['vehicles'].append(other)
            manifest = prepared_fixture(work, alter=alter)
            dataset = subject.TrajectoryDataset(manifest, 'train', seed=7)
            chosen = set()
            for epoch in range(20):
                dataset.set_epoch(epoch)
                a = dataset[0]['meta']['target_id']
                self.assertEqual(a, dataset[0]['meta']['target_id'])
                chosen.add(a)
            self.assertEqual(chosen, {'npc', 'farther'})
            held_out = subject.TrajectoryDataset(manifest, 'test', seed=99)
            self.assertEqual(held_out[0]['meta']['target_id'], 'npc')
            held_out.set_epoch(100)
            self.assertEqual(held_out[0]['meta']['target_id'], 'npc')

    def test_missing_neighbor_frames_keep_false_masks_and_finite_network_values(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def alter(i, record):
                for step, frame in enumerate(record['replay_trajectory']['frames']):
                    if step != 1:
                        frame['vehicles'].append(dict(frame['vehicles'][1], id='partial', pos=[125., 3.]))
            manifest = prepared_fixture(work, alter=alter)
            item = subject.TrajectoryDataset(manifest, 'val', max_neighbors=1)[0]
            np.testing.assert_array_equal(item['condition']['neighbor_mask'][:, 0], [True, False, True])
            np.testing.assert_array_equal(item['condition']['neighbor_history'][1, 0], np.zeros(8))
            self.assertTrue(np.isfinite(item['condition']['neighbor_history']).all())

    def test_index_corruption_is_rejected_before_using_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            with (work/'windows/window_index.jsonl').open('ab') as stream:
                stream.write(b'{}\n')
            with self.assertRaisesRegex(ValueError, 'index.*hash|hash.*index'):
                subject.TrajectoryDataset(manifest, 'train')

    def test_normalizer_parameter_changes_with_same_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            normalizer = work/'windows/normalizer.json'
            payload = read_json(normalizer)
            payload['mean'][0] += 100.
            normalizer.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'normalizer|Normalizer'):
                subject.TrajectoryDataset(manifest, 'train')

    def test_source_modification_is_caught_even_after_the_episode_was_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            dataset = subject.TrajectoryDataset(manifest, 'val')
            dataset[0]
            with (work/'pool/e5.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed|hash'):
                dataset[1]

    def test_index_rewrite_after_construction_cannot_inject_held_out_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            dataset = subject.TrajectoryDataset(manifest, 'train')
            index = work/'windows/window_index.jsonl'
            rows = index.read_bytes().splitlines(keepends=True)
            rows[0] = next(row for row in rows if json.loads(row)['split'] == 'val')
            index.write_bytes(b''.join(rows))
            with self.assertRaisesRegex(ValueError, 'index|split|changed'):
                dataset[0]

    def test_maps_changed_before_first_load_or_after_cache_use_are_rejected(self):
        for after_cache in (False, True):
            with self.subTest(after_cache=after_cache), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                maps = work/'maps'
                maps.mkdir()
                (maps/'scene.sumocfg').write_text('<configuration><input><net-file value="road.net.xml"/></input></configuration>')
                net = maps/'road.net.xml'
                net.write_text('<net><edge><lane width="3" speed="10" shape="0,0 100,0"/></edge></net>')
                def alter(i, record):
                    record['source_cfg'] = '/old/scene.sumocfg'
                manifest = prepared_fixture(work, alter=alter)
                dataset = subject.TrajectoryDataset(manifest, 'val', map_root=maps)
                if after_cache:
                    dataset[0]
                net.write_text('<net><edge><lane width="4" speed="10" shape="0,0 100,0"/></edge></net>')
                with self.assertRaisesRegex(ValueError, 'map|Map|changed|binding'):
                    dataset[1]

    def test_episode_cache_reuses_adjacent_rows_and_stays_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            dataset = subject.TrajectoryDataset(manifest, 'train', cache_size=2)
            dataset[0]
            dataset[1]
            self.assertEqual(dataset.cache_info()['loads'], 1)
            for i in range(0, len(dataset), 3):
                dataset[i]
                self.assertLessEqual(dataset.cache_info()['episodes'], 2)
            self.assertEqual(dataset.cache_info()['loads'], 6)
            sampler = subject.EpisodeSampler(dataset, seed=11)
            one = list(sampler)
            self.assertEqual(sorted(one), list(range(18)))
            self.assertEqual(one, list(sampler))

    def test_relocation_preserves_frozen_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            shutil.copytree(work/'pool', work/'moved_pool')
            shutil.copytree(work/'windows', work/'moved_windows')
            shutil.copytree(work/'split', work/'moved_split')
            dataset = subject.TrajectoryDataset(work/'moved_windows/window_dataset_manifest.json', 'val',
                        input_root=work/'moved_pool', split_manifest=work/'moved_split/split_manifest.json')
            self.assertEqual(dataset[0]['meta']['target_id'], 'npc')
            self.assertEqual(dataset.fingerprint['normalizer_id'], read_json(manifest)['normalizer']['normalizer_id'])

    def test_numpy_inspection_cli_reports_real_shapes_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            manifest = prepared_fixture(work)
            result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/trajectory_dataset.py'),
                     '--dataset-manifest', str(manifest), '--split', 'val', '--inspect', '2'],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['inspected_windows'], 2)
            self.assertEqual(report['label_shape'], [2, 2])
            self.assertTrue(report['finite_conditions'])

    def test_optional_sumo_map_uses_the_same_anchor_and_marks_missing_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            maps = work/'maps'
            maps.mkdir()
            (maps/'scene.sumocfg').write_text('<configuration><input><net-file value="road.net.xml"/></input></configuration>')
            (maps/'road.net.xml').write_text('<net><edge><lane id="a" width="3" speed="10" shape="100,-10 100,10"/></edge></net>')
            def alter(i, record):
                record['source_cfg'] = '/old/unavailable/scene.sumocfg'
            manifest = prepared_fixture(work, alter=alter)
            dataset = subject.TrajectoryDataset(manifest, 'val', map_root=maps)
            item = dataset[0]
            self.assertTrue(item['condition']['map_mask'].any())
            self.assertEqual(item['meta']['map_source']['identity_verified_against_recording'], False)
            missing = subject.TrajectoryDataset(manifest, 'val')[0]
            self.assertFalse(missing['condition']['map_mask'].any())

    def test_map_cache_is_bounded_and_map_content_is_part_of_dataset_identity(self):
        self.assertTrue(hasattr(subject.SumoMapStore, 'cache_info'), 'Bounded map cache is missing')
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            maps = work/'maps'
            maps.mkdir()
            net = maps/'road.net.xml'
            net.write_text('<net><edge><lane width="3" speed="10" shape="0,0 100,0"/></edge></net>')
            for i in range(6):
                (maps/('s%d.sumocfg' % i)).write_text('<configuration><input><net-file value="road.net.xml"/></input></configuration>')
            store = subject.SumoMapStore(maps, cache_size=2)
            for i in range(6):
                store.features('s%d.sumocfg' % i, np.zeros(8))
                self.assertLessEqual(store.cache_info()['maps'], 2)
            first = store.binding_fingerprint(['s0.sumocfg'])
            net.write_text('<net><edge><lane width="4" speed="10" shape="0,0 100,0"/></edge></net>')
            second = subject.SumoMapStore(maps).binding_fingerprint(['s0.sumocfg'])
            self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
