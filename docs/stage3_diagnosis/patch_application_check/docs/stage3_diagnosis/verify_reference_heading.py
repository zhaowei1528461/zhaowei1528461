"""Run the bounded real-data Stage 3 check with unchanged physical constraints."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'embodied/envs'))
from rssm_inference import file_sha256


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_exports(folder, summary, budget):
    counts = {}
    for method in ('random', 'cem'):
        result = summary['searches'][method]
        trials = [json.loads(line) for line in (folder/method/'trials.jsonl').read_text().splitlines()]
        assert len(trials) == result['evaluations'] == budget, 'Unequal or incomplete search budget'
        assert sum(int(t['valid']) for t in trials) == result['valid_count']
        scenes = sorted((folder/method).glob('top_*.json'))
        assert len(scenes) == len(result['top_candidate_ids'])
        for rank, path in enumerate(scenes):
            scene = read_json(path)
            assert scene['candidate_id'] == result['top_candidate_ids'][rank]
            assert scene['offline_constraints_passed'] and not scene['simulation_verified']
            assert scene['parameter_frame']['heading_source'] == 'target'
            assert scene['parameter_frame'] == summary['parameter_frame']
            assert scene['risk']['prediction_source'] == 'fixed_ego_candidate_rule'
            assert scene['risk']['score'] == result['top_risks'][rank]
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                assert arrays['npc_ids'].tolist() == scene['npcs']['ids']
                assert str(arrays['ego_id'].item()) == scene['ego']['id']
                np.testing.assert_array_equal(arrays['states'], np.asarray(scene['target_trajectory']['states']))
                np.testing.assert_array_equal(arrays['ego_state'], np.asarray(scene['ego']['states']))
                np.testing.assert_allclose(arrays['npc_states'], np.asarray(scene['npcs']['states'], dtype=float), equal_nan=True)
                np.testing.assert_array_equal(arrays['npc_valid_mask'], scene['npcs']['mask'])
                target_index = scene['npcs']['ids'].index(scene['target_id'])
                np.testing.assert_array_equal(arrays['states'], arrays['npc_states'][:, target_index])
                metadata = json.loads(str(arrays['metadata_json'].item()))
                assert metadata['parameter_frame'] == scene['parameter_frame']
        counts[method] = result['valid_count']
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--sumo-cfg', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--windows', type=int, nargs='+', default=[0, 100, 200, 300, 400, 500])
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeat-window', type=int, default=500)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    original_hash = file_sha256(args.input)
    windows = sorted(set(args.windows))
    if args.repeat_window not in windows:
        raise ValueError('--repeat-window must be included in --windows')
    report = {'source': str(args.input), 'source_sha256': original_hash,
              'reference_heading': 'target', 'seed': args.seed, 'budget_per_method': args.budget,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__},
              'windows': [], 'simulation_verified': False}
    summaries = {}
    for window, repeat in [(i, False) for i in windows] + [(args.repeat_window, True)]:
        folder = output/('w%d%s' % (window, '_repeat' if repeat else ''))
        command = [sys.executable, '-u', str(ROOT/'embodied/envs/run_scene_search.py'),
                   '--input', str(Path(args.input).resolve()), '--sumo-cfg', str(Path(args.sumo_cfg).resolve()),
                   '--reference-heading', 'target', '--window-index', str(window),
                   '--budget', str(args.budget), '--seed', str(args.seed), '--top-k', '5', '--output', str(folder)]
        print('Starting window=%d repeat=%s' % (window, repeat), flush=True)
        completed = subprocess.run(command)
        if completed.returncode not in (0, 2):
            raise RuntimeError('Search failed with exit code %d; inspect %s' % (completed.returncode, folder))
        summary = read_json(folder/'summary.json')
        counts = verify_exports(folder, summary, args.budget)
        if repeat:
            previous = summaries[window]
            for method in ('random', 'cem'):
                for key in ('valid_count', 'top_candidate_ids', 'top_risks', 'failure_counts'):
                    assert summary['searches'][method][key] == previous['searches'][method][key], 'Same-seed result changed'
                old = output/('w%d' % window)/method/'trials.jsonl'
                new = folder/method/'trials.jsonl'
                traces = []
                for path in (old, new):
                    trace = [json.loads(line) for line in path.read_text().splitlines()]
                    traces.append([{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in trace])
                assert traces[0] == traces[1], 'Same-seed trial parameters/objectives changed'
            report['same_seed_repeat_verified'] = True
        else:
            summaries[window] = summary
            risks = sum((s['top_risks'] for s in summary['searches'].values()), [])
            report['windows'].append({'window': window, 'target_id': summary['target_id'],
                                      'target_type': summary['source_map']['target_type'],
                                      'valid_counts': counts, 'max_rule_risk': max(risks) if risks else None,
                                      'exit_code': completed.returncode})
    assert original_hash == file_sha256(args.input), 'Source NPZ changed'
    report['source_unchanged'] = True
    report['export_roundtrips_verified'] = True
    report['all_requested_windows_have_candidates_for_both_methods'] = all(
        min(w['valid_counts'].values()) > 0 for w in report['windows'])
    (output/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['all_requested_windows_have_candidates_for_both_methods'] else 2


if __name__ == '__main__':
    sys.exit(main())
