"""Reproduce local compatibility and fixture checks; never infer server pool size."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'embodied/envs'))
sys.path.insert(0, str(ROOT/'tests'))
import scenario_data as current
from test_scenario_splits import write_record
import prepare_scenario_splits as preparation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = Path(__file__).parent/'source_before/embodied/envs/scenario_data.py'
    spec = importlib.util.spec_from_file_location('scenario_data_before_split_tool', baseline)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    comparisons = 0
    for count in (0, 1, 2, 3, 5, 8, 20, 80):
        rows = [SimpleNamespace(episode_id='e%03d' % i, scenario_id='s%d' % i,
                 parent_scenario_id=('s%d' % (i-1) if i % 3 else 'absent%d' % i),
                 map_id='map%d' % (i//9)) for i in range(count)]
        for group_by in ('scenario', 'map'):
            for ratios in ((.8, .1, .1), (.6, .2, .2), (1., 0., 0.), (0., .5, .5)):
                for seed in range(5):
                    for entries in (rows, rows[::-1]):
                        old = previous.split_episodes(entries, ratios=ratios, seed=seed, group_by=group_by)
                        new = current.split_episodes(entries, ratios=ratios, seed=seed, group_by=group_by)
                        assert {k: [e.episode_id for e in v] for k, v in old.items()} == {
                            k: [e.episode_id for e in v] for k, v in new.items()}
                        comparisons += 1
    report = {'legacy_split_exact_comparisons': comparisons,
              'server_full_pool_available': False}
    with tempfile.TemporaryDirectory(prefix='scenario_split_verify_') as directory:
        work = Path(directory)
        pool = work/'fixture_pool'
        for i in range(20):
            write_record(pool, 'base-%02d' % i, 'root-%02d' % i)
            write_record(pool, 'copy-%02d' % i, 'copy-%02d' % i,
                         parent='root-%02d' % i, source_type='sumo_copy')
            write_record(pool, 'grandchild-%02d' % i, 'grandchild-%02d' % i,
                         parent='copy-%02d' % i, source_type='sumo_attack')
        before = {p.name: digest(p) for p in pool.glob('*.npz')}
        fixture = preparation.prepare_pool(pool, work/'prepared')
        assert fixture['exit_code'] == 0
        assert fixture['recorded_episode_count'] == 60
        assert fixture['independent_parent_group_count'] == 20
        assert fixture['splits'] == {'train': {'episodes': 48, 'parent_groups': 16},
                                     'val': {'episodes': 6, 'parent_groups': 2},
                                     'test': {'episodes': 6, 'parent_groups': 2}}
        assert before == {p.name: digest(p) for p in pool.glob('*.npz')}
        manifest = json.loads((work/'prepared/split_manifest.json').read_text(encoding='utf-8'))
        destinations = {row['episode_id']: name for name, rows in manifest['splits'].items() for row in rows}
        for i in range(20):
            assert len({destinations[prefix+'-%02d' % i] for prefix in ('base', 'copy', 'grandchild')}) == 1
        report['synthetic_format_fixture'] = {
            'purpose': 'Generated unit-test records, not actual SUMO executions or server statistics',
            'episodes': 60, 'parent_groups': 20, 'splits': fixture['splits'],
            'leakage_check': fixture['leakage_check'], 'input_hashes_unchanged': True}
        sample = ROOT/'docs/stage3_uploaded_debug/sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0/posttrain_episode_pool/c4c745e14ccbd182d2815682.npz'
        source_hash = digest(sample)
        result = subprocess.run([sys.executable, str(ROOT/'embodied/envs/prepare_scenario_splits.py'),
                 '--input', str(sample), '--output', str(work/'real_prepared'), '--progress-every', '1'],
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', check=False)
        real = json.loads(result.stdout)
        assert result.returncode == real['exit_code'] == 2
        assert real['recorded_episode_count'] == real['independent_parent_group_count'] == 1
        assert real['splits'] == {'train': {'episodes': 1, 'parent_groups': 1},
                                 'val': {'episodes': 0, 'parent_groups': 0},
                                 'test': {'episodes': 0, 'parent_groups': 0}}
        assert real['leakage_check']['passed'] and not real['all_requested_splits_nonempty']
        assert source_hash == digest(sample)
        report['uploaded_real_episode'] = {
            'source_sha256': source_hash, 'input_hash_unchanged': True,
            'exit_code': result.returncode, 'episodes': 1, 'parent_groups': 1,
            'splits': real['splits'], 'leakage_check': real['leakage_check'],
            'limitation': 'Only one real uploaded episode is present locally; full-pool counts require the server run.'}
    output = Path(__file__).parent/'verification.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
