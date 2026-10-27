"""Read-only replay of the uploaded Stage 3 trials against matching local maps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'embodied/envs'))
from scenario_data import load_episode, make_windows
from scene_parameters import SceneGenerator, SceneParameterSpec, DynamicsLimits
from scene_road import load_sumo_scene
from scene_search import SceneEvaluator
from rssm_inference import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=str(ROOT / 'docs/stage3_uploaded_debug'))
    parser.add_argument('--output', default=str(ROOT / 'docs/stage3_diagnosis/replay.json'))
    args = parser.parse_args()
    source_root = Path(args.input_root)
    source = next(source_root.rglob('posttrain_episode_pool/*.npz'))
    episode = load_episode(source)
    windows = make_windows(episode, history=11, future=30, dt=episode.dt)
    report = {
        'purpose': 'Replay original rejected candidates; no generator or constraints changed.',
        'input_sha256': file_sha256(source),
        'local_environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'episode': {'frames': len(episode.timestamps), 'npc_count': len(episode.npc_ids),
                    'native_dt': episode.dt, 'window_count': len(windows)},
        'windows': [],
    }
    for path in sorted(source_root.rglob('w*/summary.json'), key=lambda p: int(p.parent.name[1:])):
        original = json.loads(path.read_text(encoding='utf-8'))
        if report['input_sha256'] != original['source']['input_sha256']:
            raise ValueError('Source NPZ hash differs from original report')
        config = ROOT / 'embodied/generated_scenarios_merge_7' / Path(original['source_map']['sumo_cfg']).name
        road, limits, map_info = load_sumo_scene(config, original['target_id'], original['profile'])
        for field in ('cfg_sha256', 'net_sha256'):
            if map_info[field] != original['source_map'][field]:
                raise ValueError('Map hash differs: ' + field)
        if [r['sha256'] for r in map_info['routes']] != [r['sha256'] for r in original['source_map']['routes']]:
            raise ValueError('Route hash differs')
        window = windows[original['window_index']]
        index = window.npc_ids.index(original['target_id'])
        spec = SceneParameterSpec(original['parameter_bounds'], DynamicsLimits(**original['dynamics_limits']),
                                  tuple(original['behaviors']))
        generator = SceneGenerator(window, index, spec, original['profile']['acceleration_ramp'],
                                   original['profile']['validation_dt'])
        radius = spec.bounds['initial_gap'][1] + limits.max_speed * generator.horizon + 10.
        road = road.local(generator.origin, radius)
        evaluator = SceneEvaluator(limits, road, road_grid_spacing=original['profile']['road_grid_spacing'])
        delta = generator.npc[:2] - generator.origin
        heading_difference = np.arctan2(np.sin(generator.npc[4]-generator.ego[4]),
                                        np.cos(generator.npc[4]-generator.ego[4]))
        item = {'window': original['window_index'], 'target': original['target_id'],
                'target_type': map_info['target_type'], 'limits': original['dynamics_limits'],
                'anchor': window.anchor_timestamp,
                'heading_difference_deg': float(np.degrees(heading_difference)),
                'observed_longitudinal_gap': float(delta @ generator.rotation[:, 0]),
                'observed_lateral_gap': generator.initial_lateral,
                'observed_lateral_velocity': generator.lateral_speed,
                'observed_target_anchor_on_road': road.footprint_valid(generator.npc[None]),
                'source_environment': original['environment'], 'methods': {}}
        for method in ('random', 'cem'):
            trials = [json.loads(line) for line in (path.parent / method / 'trials.jsonl').read_text().splitlines()]
            failures = Counter()
            initial_offroad = valid = failure_mismatches = 0
            max_metric_difference = 0.
            fewest = None
            for trial in trials:
                candidate = generator.generate(trial['parameters'], trial['behavior'])
                evaluated = evaluator.evaluate(candidate)
                failures.update(evaluated.failures)
                valid += int(evaluated.valid)
                failure_mismatches += int(set(evaluated.failures) != set(trial['failures']))
                initial_offroad += int(not road.footprint_valid(candidate.states[:1]))
                for key in ('max_speed', 'max_abs_jerk', 'max_lateral_accel', 'max_yaw_rate'):
                    max_metric_difference = max(max_metric_difference,
                                                abs(evaluated.metrics[key]-trial['metrics'][key]))
                if fewest is None or len(evaluated.failures) < len(fewest['failures']):
                    velocity = candidate.dense_states[:, 2:4]
                    speed = np.linalg.norm(velocity, axis=-1)
                    along = np.sum(velocity*candidate.accelerations, axis=-1)/np.maximum(speed, 1e-9)
                    fewest = {'parameters': trial['parameters'], 'behavior': trial['behavior'],
                              'failures': evaluated.failures,
                              'min_actual_longitudinal_accel': float(along.min()),
                              'max_actual_longitudinal_accel': float(along.max()),
                              'metrics': evaluated.metrics}
            item['methods'][method] = {
                'evaluations': len(trials), 'valid_count': valid, 'failure_counts': dict(failures),
                'failure_set_mismatches_with_linux': failure_mismatches,
                'max_numeric_metric_difference_with_linux': max_metric_difference,
                'generated_initial_footprint_offroad_count': initial_offroad,
                'fewest_failures_trial': fewest,
            }
            print('window=%d method=%s valid=%d/%d initial_offroad=%d failure_mismatches=%d' %
                  (item['window'], method, valid, len(trials), initial_offroad, failure_mismatches), flush=True)
        report['windows'].append(item)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Report:', destination)


if __name__ == '__main__':
    main()
