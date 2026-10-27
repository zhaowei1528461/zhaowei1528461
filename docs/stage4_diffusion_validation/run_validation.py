"""Run actual CPU stage4 smoke workflows and preserve honest provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(name, script, arguments):
    result = subprocess.run([sys.executable, '-u', str(ROOT/'embodied/envs'/script), *arguments],
                 cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    (HERE/(name+'_stdout.json')).write_text(result.stdout, encoding='utf-8')
    (HERE/(name+'_stderr.log')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(name+' failed:\n'+result.stdout+'\n'+result.stderr)
    report = json.loads(result.stdout)
    assert report['exit_code'] == 0
    print(name+': passed', flush=True)
    return report


def main():
    global HERE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=HERE)
    args = parser.parse_args()
    HERE = args.output_root.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    original_paths = [ROOT/p for p in ('agent.py', 'nets.py', 'behaviors.py', 'ninjax.py', 'train.py',
                     'embodied/envs/scenario_data.py', 'embodied/envs/prepare_trajectory_windows.py')]
    original_paths += [ROOT/'diffscene-main'/p for p in ('models/temporal.py', 'models/diffusion.py',
                       'models/helpers.py', 'models/traffic_model.py', 'scripts/train_diffusion.py')]
    real = ROOT/'docs/window_preparation/real_single_episode_final/windows/window_dataset_manifest.json'
    original_paths += [real, real.parent/'normalizer.json', real.parent/'window_index.jsonl',
                        real.parent.parent/'split/split_manifest.json']
    original_paths.append(ROOT/'docs/stage3_uploaded_debug/sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0/posttrain_episode_pool/c4c745e14ccbd182d2815682.npz')
    before = {str(p.relative_to(ROOT)): sha(p) for p in original_paths}
    common = ['--base-dim', '16', '--dim-mults', '1', '2', '--condition-dim', '32',
              '--diffusion-steps', '20', '--sampling-steps', '10', '--batch-size', '4',
              '--samples', '4', '--torch-threads', '1', '--seed', '0', '--device', 'cpu', '--log-every', '20']
    synthetic = command('synthetic_train', 'train_trajectory_diffusion.py', [
        '--synthetic-smoke', '--output', str(HERE/'synthetic_train'), '--train-limit', '16', '--val-limit', '4',
        '--epochs', '16', '--max-steps', '64', *common])
    samples = command('synthetic_samples', 'sample_trajectories.py', [
        '--synthetic-smoke', '--checkpoint', synthetic['latest_checkpoint'], '--output', str(HERE/'synthetic_samples'),
        '--limit', '4', '--split', 'val', '--samples', '4', '--sampling-steps', '10', '--export-limit', '1', '--device', 'cpu'])
    inspection = command('real_inspect', 'trajectory_dataset.py', [
        '--dataset-manifest', str(real), '--split', 'train', '--inspect', '16'])
    recorded = command('real_train_only_smoke', 'train_trajectory_diffusion.py', [
        '--dataset-manifest', str(real), '--allow-train-only-smoke', '--output', str(HERE/'real_train_only_smoke'),
        '--train-limit', '16', '--epochs', '10', '--max-steps', '40', *common])
    actual_samples = command('real_train_samples', 'sample_trajectories.py', [
        '--dataset-manifest', str(real), '--checkpoint', recorded['latest_checkpoint'],
        '--output', str(HERE/'real_train_samples'), '--limit', '8', '--split', 'train',
        '--samples', '4', '--sampling-steps', '10', '--export-limit', '1', '--device', 'cpu'])
    assert synthetic['optimization_steps'] == 64 and synthetic['checkpoint_roundtrip_verified']
    assert recorded['optimization_steps'] == 40 and recorded['train_only_smoke']
    assert recorded['validation'] is None and recorded['checkpoint_roundtrip_verified']
    assert actual_samples['evaluation_split'] == 'train' and actual_samples['limited_evaluation']
    assert all(sha(ROOT/path) == digest for path, digest in before.items())
    for report in (samples, actual_samples):
        assert np.isfinite([report[k] for k in ('mean_ADE_m', 'mean_FDE_m', 'minADE_m', 'minFDE_m', 'noise_prediction_loss')]).all()
        assert report['road_valid_rate'] is None and not report['simulation_verified']
        candidate = json.loads(Path(report['exported_candidates'][0]['json']).read_text(encoding='utf-8'))
        assert candidate['generated_target_history'] is False and candidate['simulation_verified'] is False
        assert candidate['generation_metadata']['sampling_context']['num_samples'] == 4
        assert candidate['generation_metadata']['sampling_context']['batch_row'] == 0
    evidence = {'environment': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                                'numpy': np.__version__, 'device': 'cpu', 'cuda_available': torch.cuda.is_available()},
                'source_hashes_unchanged': before, 'synthetic_training': synthetic,
                'synthetic_validation_samples': samples, 'real_data_inspection': inspection,
                'real_train_only_smoke': recorded, 'real_train_only_samples': actual_samples,
                'server_full_pool_available': False,
                'limitations': ['The real local manifest contains one episode and train only.',
                                'No real independent validation/test or full 951-episode training was run locally.',
                                'Motion thresholds are provisional; missing maps are not counted as road-valid.',
                                'CPU smoke training is not a model-quality acceptance result or simulator validation.']}
    (HERE/'verification.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'synthetic_steps': 64, 'real_train_only_steps': 40,
                      'real_world_roundtrip_error_m': inspection['max_world_roundtrip_error_m'],
                      'original_hashes_unchanged': True, 'report': str(HERE/'verification.json')}, indent=2))


if __name__ == '__main__':
    main()
