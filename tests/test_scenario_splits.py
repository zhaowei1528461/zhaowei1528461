"""Recorded-pool indexing, bounded loading and ancestry-safe split regressions."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'embodied/envs'))
import scenario_data as data
from test_scenario_data import record

try:
    splits = importlib.import_module('prepare_scenario_splits')
except ImportError:
    splits = None


def write_record(directory, episode_id, scenario_id, parent=None, source_type='sumo_original', complete=True):
    item = record(episode_id=episode_id, scenario_id=scenario_id, parent=parent)
    item['source_type'] = source_type
    item['metadata'] = {'prediction_source': 'sumo_execution' if source_type.startswith('sumo_') else 'synthetic_kinematics',
                        'collection_complete': complete}
    return data.write_episode(data.episode_from_record(item), directory)


def fixture_pool(directory, count=8):
    for i in range(count):
        write_record(directory, 'base-%d' % i, 's%d' % i)
        write_record(directory, 'copy-%d' % i, 'c%d' % i, parent='s%d' % i, source_type='sumo_copy')


class ScenarioSplitTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(splits, 'Streaming pool split tool is missing')

    def test_group_helper_merges_transitive_ancestry_and_absent_parents(self):
        self.assertTrue(hasattr(data, 'group_episodes'), 'Shared ancestry group helper is missing')
        entries = [SimpleNamespace(episode_id='one', scenario_id='child', parent_scenario_id='absent', map_id=None),
                   SimpleNamespace(episode_id='two', scenario_id='grand', parent_scenario_id='child', map_id=None),
                   SimpleNamespace(episode_id='three', scenario_id='other', parent_scenario_id=None, map_id=None)]
        groups = data.group_episodes(entries)
        self.assertEqual({frozenset(e.episode_id for e in g) for g in groups.values()},
                         {frozenset({'one', 'two'}), frozenset({'three'})})

    def test_inventory_counts_groups_and_persists_nonleaking_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, output = Path(directory)/'pool', Path(directory)/'prepared'
            fixture_pool(pool)
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in pool.glob('*.npz')}
            report = splits.prepare_pool(pool, output)
            self.assertEqual(report['exit_code'], 0)
            self.assertEqual(report['recorded_episode_count'], 16)
            self.assertEqual(report['independent_parent_group_count'], 8)
            self.assertEqual({k: v['episodes'] for k, v in report['splits'].items()}, {'train': 12, 'val': 2, 'test': 2})
            manifest = json.loads((output/'split_manifest.json').read_text())
            destinations = {row['episode_id']: name for name, rows in manifest['splits'].items() for row in rows}
            for i in range(8):
                self.assertEqual(destinations['base-%d' % i], destinations['copy-%d' % i])
            # Fixed-seed compatibility with the old splitter, including its stable ordering.
            self.assertEqual({k for k, v in destinations.items() if v == 'val'}, {'base-5', 'copy-5'})
            self.assertEqual({k for k, v in destinations.items() if v == 'test'}, {'base-4', 'copy-4'})
            self.assertTrue(report['leakage_check']['passed'])
            self.assertTrue(report['all_requested_splits_nonempty'])
            self.assertFalse(report['windows_materialized'])
            self.assertFalse((output/'normalizer.json').exists())
            self.assertEqual(len((output/'episode_index.jsonl').read_text().splitlines()), 16)
            self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in pool.glob('*.npz')})

    def test_previous_episode_is_released_before_the_next_load(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_pool(Path(directory)/'pool', count=3)
            refs = []
            original = splits.load_episode
            def tracked_load(path):
                if refs:
                    self.assertIsNone(refs[-1](), 'Previous full episode is retained in memory')
                episode = original(path)
                refs.append(weakref.ref(episode))
                return episode
            with mock.patch.object(splits, 'load_episode', side_effect=tracked_load):
                report = splits.prepare_pool(Path(directory)/'pool', Path(directory)/'out', ratios=(1., 0., 0.))
            self.assertEqual(report['recorded_episode_count'], 6)
            self.assertIsNone(refs[-1]())

    def test_identical_duplicate_files_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            fixture_pool(pool)
            (pool/'copies').mkdir()
            shutil.copyfile(pool/'base-0.npz', pool/'copies'/'duplicate.npz')
            report = splits.prepare_pool(pool, Path(directory)/'out')
            self.assertEqual(report['recorded_episode_count'], 16)
            self.assertEqual(report['duplicate_file_count'], 1)
            self.assertEqual(report['rejected_file_count'], 0)

    def test_conflicting_episode_id_excludes_both_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            fixture_pool(pool)
            altered = data.episode_from_record(record(episode_id='base-0', scenario_id='different'))
            data.write_episode(altered, pool/'conflicting')
            report = splits.prepare_pool(pool, Path(directory)/'out')
            self.assertEqual(report['exit_code'], 1)
            self.assertEqual(report['recorded_episode_count'], 15)
            self.assertEqual(report['rejected_file_count'], 2)
            manifest = json.loads((Path(directory)/'out/split_manifest.json').read_text())
            self.assertNotIn('base-0', [r['episode_id'] for rows in manifest['splits'].values() for r in rows])

    def test_synthetic_and_generated_candidates_are_excluded_from_real_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            write_record(pool, 'real', 'real')
            write_record(pool, 'synthetic', 'synthetic', source_type='synthetic')
            np.savez_compressed(pool/'candidate.npz', metadata_json=np.asarray(json.dumps({
                'schema_version': 1, 'format': 'long_tail_candidate_scene', 'simulation_verified': False})))
            report = splits.prepare_pool(pool, Path(directory)/'out', ratios=(1., 0., 0.))
            self.assertEqual(report['exit_code'], 0)
            self.assertEqual(report['recorded_episode_count'], 1)
            self.assertEqual(report['excluded_file_count'], 2)

    def test_corrupt_file_is_reported_and_prevents_success_status(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            write_record(pool, 'real', 'real')
            (pool/'broken.npz').write_bytes(b'not a zip archive')
            report = splits.prepare_pool(pool, Path(directory)/'out', ratios=(1., 0., 0.))
            self.assertEqual(report['exit_code'], 1)
            self.assertEqual(report['recorded_episode_count'], 1)
            self.assertEqual(report['rejected_file_count'], 1)

    def test_nonobject_metadata_is_rejected_without_aborting_the_pool(self):
        for canonical, nested in ((False, False), (True, False), (True, True)):
            with self.subTest(canonical=canonical, nested=nested), tempfile.TemporaryDirectory() as directory:
                pool, output = Path(directory)/'pool', Path(directory)/'out'
                write_record(pool, 'real', 'real')
                arrays = {}
                if canonical:
                    with np.load(pool/'real.npz', allow_pickle=False) as archive:
                        arrays = {name: archive[name] for name in archive.files}
                header = json.loads(str(arrays['metadata_json'].item())) if nested else []
                if nested:
                    header['metadata'] = []
                arrays['metadata_json'] = np.asarray(json.dumps(header))
                np.savez_compressed(pool/'malformed.npz', **arrays)
                report = splits.prepare_pool(pool, output, ratios=(1., 0., 0.))
                self.assertEqual(report['exit_code'], 1)
                self.assertEqual(report['recorded_episode_count'], 1)
                self.assertEqual(report['rejected_file_count'], 1)
                self.assertTrue((output/'dataset_report.json').is_file())

    def test_damaged_deflate_member_is_reported_without_aborting_the_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, output = Path(directory)/'pool', Path(directory)/'out'
            write_record(pool, 'real', 'real')
            target = pool/'damaged.npz'
            shutil.copyfile(pool/'real.npz', target)
            payload = bytearray(target.read_bytes())
            with zipfile.ZipFile(target) as archive:
                member = archive.getinfo('metadata_json.npy')
                self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)
            offset = member.header_offset
            start = offset + 30 + int.from_bytes(payload[offset+26:offset+28], 'little') + int.from_bytes(payload[offset+28:offset+30], 'little')
            payload[start] = (payload[start] & ~7) | 6  # Reserved DEFLATE block type.
            target.write_bytes(payload)
            report = splits.prepare_pool(pool, output, ratios=(1., 0., 0.))
            self.assertEqual(report['exit_code'], 1)
            self.assertEqual(report['recorded_episode_count'], 1)
            self.assertEqual(report['rejected_file_count'], 1)
            self.assertTrue((output/'dataset_report.json').is_file())

    def test_few_parent_groups_report_empty_splits_instead_of_splitting_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            fixture_pool(pool, count=1)
            report = splits.prepare_pool(pool, Path(directory)/'out')
            self.assertEqual(report['exit_code'], 2)
            self.assertEqual(report['independent_parent_group_count'], 1)
            self.assertEqual({k: v['episodes'] for k, v in report['splits'].items()}, {'train': 2, 'val': 0, 'test': 0})
            self.assertFalse(report['all_requested_splits_nonempty'])
            self.assertTrue(report['leakage_check']['passed'])

    def test_recorded_incomplete_prefix_is_counted_with_completeness_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = Path(directory)/'pool'
            write_record(pool, 'prefix', 'source', complete=False)
            report = splits.prepare_pool(pool, Path(directory)/'out', ratios=(1., 0., 0.))
            self.assertEqual(report['recorded_episode_count'], 1)
            self.assertEqual(report['collection_completeness_counts']['false'], 1)
            row = json.loads((Path(directory)/'out/episode_index.jsonl').read_text().splitlines()[0])
            self.assertFalse(row['collection_complete'])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, output = Path(directory)/'pool', Path(directory)/'out'
            fixture_pool(pool, count=1)
            output.mkdir()
            sentinel = output/'keep.txt'
            sentinel.write_text('original')
            with self.assertRaisesRegex(ValueError, 'empty|new'):
                splits.prepare_pool(pool, output)
            self.assertEqual(sentinel.read_text(), 'original')

    def test_cli_prints_progress_and_writes_final_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_pool(Path(directory)/'pool')
            result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/prepare_scenario_splits.py'),
                '--input', str(Path(directory)/'pool'), '--output', str(Path(directory)/'out'),
                '--progress-every', '4'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('[scan', result.stderr)
            self.assertIn('16/16', result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['independent_parent_group_count'], 8)
            self.assertTrue(Path(report['manifest_path']).is_file())


if __name__ == '__main__':
    unittest.main()
