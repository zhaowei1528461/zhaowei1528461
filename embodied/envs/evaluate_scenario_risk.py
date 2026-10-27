"""Standalone stage 2 CLI: native episode rules, optional original RSSM inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

try:
    from .scenario_data import load_episode, make_windows, _json_safe
    from .scenario_risk import RiskConfig, RiskEvaluator, obb_distance, obb_ttc, POSITION_REFERENCES
    from .world_model_adapter import WorldModelAdapter
    from .rssm_inference import NinjaxRSSMBackend, file_sha256
except ImportError:
    from scenario_data import load_episode, make_windows, _json_safe
    from scenario_risk import RiskConfig, RiskEvaluator, obb_distance, obb_ttc, POSITION_REFERENCES
    from world_model_adapter import WorldModelAdapter
    from rssm_inference import NinjaxRSSMBackend, file_sha256


def _json(value):
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def observed_frame_audit(episode, config):
    """Unique original samples, so overlapping windows do not inflate event counts."""
    reference = episode.metadata.get('position_reference')
    for i, stamp in enumerate(episode.timestamps):
        distances, ttcs = [], []
        ego = episode.ego_state[i]
        known = reference in POSITION_REFERENCES and episode.coordinate_frame == 'world'
        if known and episode.ego_valid_mask[i]:
            for j in np.flatnonzero(episode.npc_valid_mask[i]):
                npc = episode.npc_states[i, j]
                if not np.isfinite(npc).all() or not np.all(npc[6:8] > 0) or not np.all(ego[6:8] > 0):
                    continue
                distances.append((obb_distance(ego, npc, reference), episode.npc_ids[j]))
                ttc = obb_ttc(ego, npc, reference, config.ttc_horizon)
                if ttc is not None:
                    ttcs.append((ttc, episode.npc_ids[j]))
        nearest = min(distances) if distances else (None, None)
        first_contact = min(ttcs) if ttcs else (None, None)
        yield {'timestamp': float(stamp), 'position_reference_known': known,
               'min_distance': nearest[0], 'nearest_npc': nearest[1],
               'min_ttc': first_contact[0], 'ttc_npc': first_contact[1],
               'geometric_contact_or_overlap': bool(nearest[0] <= 1e-9) if nearest[0] is not None else None,
               'recorded_ttc': episode.events['ttc'][i], 'recorded_cost': episode.events['cost'][i],
               'recorded_collision': bool(episode.events['collision'][i]),
               'recorded_near_miss': bool(episode.events['near_miss'][i]),
               'prediction_source': 'measured_trajectory_rule'}


def _model_record(result):
    # Keep the JSON report small; decoded values are saved in separate pickle-free NPZ.
    omitted = ('latent', 'ground_truth', 'ground_truth_mask', 'free_rollout', 'teacher_forcing')
    record = {k: v for k, v in result.items() if k not in omitted}
    record['output_shapes'] = {}
    for mode in ('free_rollout', 'teacher_forcing'):
        record['output_shapes'][mode] = {k: list(np.asarray(v).shape)
            for k, v in (result.get(mode) or {}).items() if k != 'latent'}
    return record


def run(args):
    source, output = Path(args.input), Path(args.output)
    episode = load_episode(source)
    dt = float(args.dt) if args.dt is not None else episode.dt
    if dt is None:
        raise ValueError('Nonuniform episode requires explicit --dt for trajectory rules')
    config = RiskConfig(**json.loads(Path(args.risk_config).read_text(encoding='utf-8'))) if args.risk_config else RiskConfig()
    windows = make_windows(episode, history=args.history, future=args.future, dt=dt, stride=args.stride)
    if not windows:
        raise ValueError('No complete windows: reduce --history/--future or collect a longer episode')
    output.mkdir(parents=True, exist_ok=True)
    evaluator = RiskEvaluator(config)
    reports = [evaluator.evaluate_risk(window) for window in windows]
    with (output/'windows.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for index, (window, report) in enumerate(zip(windows, reports)):
            stream.write(_json({'window_index': index, 'episode_id': episode.episode_id,
                'scenario_id': episode.scenario_id, 'anchor_timestamp': window.anchor_timestamp,
                'dt': dt, 'seed': args.seed, 'event_labels': window.event_labels, 'risk': report.to_dict()})+'\n')
    frames = list(observed_frame_audit(episode, config))
    with (output/'observed_frames.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for frame in frames:
            stream.write(_json(frame)+'\n')
    model = {'requested': args.checkpoint is not None, 'checkpoint_path': args.checkpoint,
             'checkpoint_loaded': False, 'inference_verified': False, 'evaluated_window_count': 0,
             'error': None, 'reason': 'checkpoint_not_requested', 'model_version': None,
             'npc_plan_conditioning': False, 'world_vehicle_decoder': False,
             'predicted_collision_probability': None, 'uncertainty': None}
    if args.checkpoint:
        if args.max_model_windows <= 0:
            raise ValueError('--max-model-windows must be positive')
        model['reason'] = None
        try:
            backend = NinjaxRSSMBackend.from_checkpoint(args.checkpoint, args.model_config,
                                                        args.config_name, args.platform)
            model.update(checkpoint_loaded=True, model_version=backend.model_version,
                         config_source=backend.config_source, config_sha256=backend.config_sha256)
            adapter = WorldModelAdapter(episode, backend, args.seed, args.model_dt)
            chosen = np.unique(np.linspace(0, len(windows)-1, min(args.max_model_windows, len(windows))).astype(int))
            successes = 0
            with (output/'model_windows.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
                for index in chosen:
                    result = adapter.predict_future(windows[index])
                    record = {'window_index': int(index), 'anchor_timestamp': windows[index].anchor_timestamp,
                              **_model_record(result)}
                    stream.write(_json(record)+'\n')
                    if not result['supported']:
                        model['reason'] = result['reason']
                        continue
                    arrays = {'ground_truth': result['ground_truth'], 'ground_truth_mask': result['ground_truth_mask']}
                    for mode in ('free_rollout', 'teacher_forcing'):
                        arrays.update({mode+'_'+k: np.asarray(v) for k, v in (result.get(mode) or {}).items() if k != 'latent'})
                    arrays['metadata_json'] = np.asarray(_json(record))
                    np.savez_compressed(str(output/('prediction_%05d.npz' % index)), **arrays)
                    successes += 1
            model['evaluated_window_count'] = successes
            model['inference_verified'] = bool(len(chosen) and successes == len(chosen))
        except Exception as error:
            model['error'] = type(error).__name__ + ': ' + str(error)
            model['reason'] = 'model_loading_or_inference_failed'
    scores = [r.score for r in reports if r.score is not None]
    contacts = [item for item in frames if item['geometric_contact_or_overlap'] is True]
    summary = {'schema_version': 1, 'stage': 2, 'input_path': str(source), 'input_sha256': file_sha256(source),
               'episode_id': episode.episode_id, 'scenario_id': episode.scenario_id,
               'source_type': episode.source_type, 'simulation_seed': episode.seed,
               'frame_count': len(episode.timestamps), 'npc_count': len(episode.npc_ids),
               'native_dt': episode.dt, 'dt': dt, 'history': args.history, 'future': args.future,
               'stride': args.stride, 'seed': args.seed, 'rule_config': asdict(config),
               'position_reference': episode.metadata.get('position_reference'),
               'window_count': len(windows), 'valid_rule_window_count': sum(r.validity_flags['valid'] for r in reports),
               'risk_score_min': min(scores) if scores else None, 'risk_score_max': max(scores) if scores else None,
               'geometric_contact_or_overlap_frame_count': len(contacts),
               'contact_timestamps': [item['timestamp'] for item in contacts],
               'recorded_event_counts': {key: int(np.count_nonzero(episode.events[key]))
                                         for key in ('collision', 'near_miss', 'low_ttc', 'low_progress')},
               'legacy_ttc_zero_frame_count': int(np.count_nonzero(episode.events['ttc'] == 0)),
               'model': model, 'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                              'platform': platform.platform()},
               'limitations': ['Overlapping windows are not independent cases.',
                               'Geometric contacts are separate from simulator collision labels.',
                               'TTC extrapolates fixed heading and constant velocity within configured horizon.',
                               'No calibrated collision probability, epistemic uncertainty, or world-vehicle ADE/FDE.',
                               'Model dt defaults to episode native dt; verify it matches checkpoint training.']}
    (output/'summary.json').write_text(_json(summary)+'\n', encoding='utf-8')
    print('Rule windows: %d/%d valid; report: %s' % (summary['valid_rule_window_count'], len(windows), output/'summary.json'))
    if args.checkpoint:
        print('Checkpoint loaded: %s; RSSM inference verified: %s' % (model['checkpoint_loaded'], model['inference_verified']))
        if not model['inference_verified']:
            print(model['error'] or model['reason'], file=sys.stderr)
            return 2
    else:
        print('RSSM inference: unavailable (no checkpoint requested).')
    return 0 if summary['valid_rule_window_count'] == len(windows) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Stage 1 episode NPZ')
    parser.add_argument('--output', required=True, help='Directory for JSON/JSONL and optional model predictions')
    parser.add_argument('--dt', type=float, help='Rules grid; default: native episode dt')
    parser.add_argument('--history', type=int, default=11)
    parser.add_argument('--future', type=int, default=30)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--risk-config', help='JSON overrides of RiskConfig')
    parser.add_argument('--checkpoint', help='Original trusted checkpoint; absent means rules only')
    parser.add_argument('--model-config', help='Saved config.yaml; default: beside checkpoint')
    parser.add_argument('--config-name', help='Explicit fallback config name only if saved config does not exist')
    parser.add_argument('--model-dt', type=float, help='Original RSSM training timestep; default: episode native dt')
    parser.add_argument('--platform', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--max-model-windows', type=int, default=3)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
