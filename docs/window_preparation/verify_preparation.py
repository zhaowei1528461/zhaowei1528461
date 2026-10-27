"""Check the uploaded episode and independent format fixtures without training."""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'embodied/envs'))
sys.path.insert(0, str(ROOT/'tests'))
import scenario_data as data
import prepare_scenario_splits as splits
import prepare_trajectory_windows as tool
from test_scenario_data import record
from test_trajectory_windows import pool_fixture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-output', type=Path, default=HERE/'real_single_episode')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('scenario_data_before_windows',
                          HERE/'source_before/embodied/envs/scenario_data.py')
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    compared = 0
    for times in ([3., 3.2, 3.4, 3.6, 3.8, 4.], [0., .2, .7, .9, 1.1, 1.3]):
        for frame in ('world', 'ego-centric'):
            for dt in (.1, .2):
                for stride in (1, 2):
                    for event_only in (False, True):
                        kwargs = dict(history=3, future=2, dt=dt, stride=stride,
                                      coordinate_frame=frame, event_only=event_only)
                        old = previous.make_windows(previous.episode_from_record(record(times=times)), **kwargs)
                        new = data.make_windows(data.episode_from_record(record(times=times)), **kwargs)
                        assert len(old) == len(new)
                        for a, b in zip(old, new):
                            for key, value in a.__dict__.items():
                                if isinstance(value, np.ndarray):
                                    np.testing.assert_array_equal(value, getattr(b, key))
                                else:
                                    assert value == getattr(b, key), key
                            compared += 1
    evidence = {'exact_legacy_windows_compared': compared, 'server_full_pool_available': False}
    with tempfile.TemporaryDirectory(prefix='window_fixture_verify_') as directory:
        work = Path(directory)
        manifest = pool_fixture(work)
        before = {str(p): sha(p) for p in (work/'pool').glob('*.npz')}
        before[str(manifest)] = sha(manifest)
        result = tool.prepare_windows(manifest, work/'windows', history=3, future=2, dt=.2)
        assert result['exit_code'] == 0
        assert all(sha(Path(path)) == digest for path, digest in before.items())
        normalizer = data.ScenarioNormalizer.load(work/'windows/normalizer.json')
        assert abs(normalizer.mean[0]-5.2) < 1e-12
        assert normalizer.count == 144
        evidence['synthetic_format_fixture'] = {
            'purpose': 'Generated test records, not actual SUMO executions',
            'episodes': 8, 'independent_groups': 8, 'window_config': result['window_config'],
            'eligible_windows': {k: v['eligible_windows'] for k, v in result['splits'].items()},
            'train_normalizer_state_count': normalizer.count, 'analytic_local_x_mean': 5.2,
            'input_hashes_unchanged': True, 'leakage_check': result['leakage_check']}
    sample = ROOT/'docs/stage3_uploaded_debug/sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0/posttrain_episode_pool/c4c745e14ccbd182d2815682.npz'
    original_hash = sha(sample)
    output = args.real_output.resolve()
    # This train-only manifest is solely for local end-to-end checking. It is
    # never presented as an independent train/val/test validation dataset.
    splits.prepare_pool(sample, output/'split', ratios=(1., 0., 0.))
    manifest = output/'split/split_manifest.json'
    manifest_hash = sha(manifest)
    command = subprocess.run([sys.executable, '-u', str(ROOT/'embodied/envs/prepare_trajectory_windows.py'),
                             '--manifest', str(manifest), '--output', str(output/'windows')],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    assert command.returncode == 0, command.stdout+command.stderr
    print(command.stderr, end='')
    result = json.loads((output/'windows/window_report.json').read_text(encoding='utf-8'))
    assert result['exit_code'] == 0
    assert original_hash == sha(sample) and manifest_hash == sha(manifest)
    normalizer = data.ScenarioNormalizer.load(output/'windows/normalizer.json')
    assert normalizer.coordinate_frame == 'ego-centric'
    assert normalizer.training_episode_ids == ('c4c745e14ccbd182d2815682',)
    episode = data.load_episode(sample)
    rows, roundtrips = 0, 0
    for window in data.iter_windows(episode, history=11, future=30, dt=.2, coordinate_frame='ego-centric'):
        mask = window.npc_mask.all(axis=0)
        if window.ego_mask[:11].all() and mask.any():
            values = np.concatenate((window.history_ego,
                                     window.history_npcs[:, mask].reshape(-1, 8),
                                     window.future_npcs[:, mask].reshape(-1, 8)))
            assert np.isfinite(values).all()
            normalized = normalizer.transform(values)
            assert np.isfinite(normalized).all()
            restored = normalizer.inverse_transform(normalized)
            np.testing.assert_allclose(restored[:, [0, 1, 2, 3, 5, 6, 7]], values[:, [0, 1, 2, 3, 5, 6, 7]], atol=1e-10)
            np.testing.assert_allclose(np.sin(restored[:, 4]), np.sin(values[:, 4]), atol=1e-10)
            np.testing.assert_allclose(np.cos(restored[:, 4]), np.cos(values[:, 4]), atol=1e-10)
            rows += len(values)
            roundtrips += 1
    assert rows == normalizer.count
    evidence['uploaded_real_episode'] = {
        'source_sha256': original_hash, 'input_hash_unchanged': True,
        'frozen_manifest_unchanged': True, 'exit_code': result['exit_code'],
        'window_config': result['window_config'], 'train_coverage': result['splits']['train'],
        'normalizer_state_count': rows, 'windows_with_verified_normalization_roundtrip': roundtrips,
        'normalizer_id': normalizer.normalizer_id,
        'limitation': 'One real episode, train-only manifest for local end-to-end verification; no held-out generalization claim.'}
    (HERE/'verification.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
