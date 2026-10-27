"""Export single-NPC generated references; preserve measured history/provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .scenario_data import VEHICLE_FIELDS, _json_safe
    from .trajectory_metrics import states_from_xy
except ImportError:
    from scenario_data import VEHICLE_FIELDS, _json_safe
    from trajectory_metrics import states_from_xy


def make_candidate(metadata, future_world_xy, dt, generation, seed=0, sample_index=0):
    reference = metadata.get('position_reference')
    if reference not in ('center', 'front_bumper_center'):
        raise ValueError('Candidate export requires a known recorded position reference')
    states = states_from_xy(future_world_xy, metadata['target_anchor_world'], dt)
    identity = json.dumps({'episode_id': metadata['episode_id'], 'target_id': metadata['target_id'],
               'window_index': metadata['window_index'], 'seed': seed, 'sample_index': sample_index,
               'generation': generation}, sort_keys=True)
    candidate_id = hashlib.sha256(identity.encode()+states.astype('<f8').tobytes()).hexdigest()[:24]
    return {'schema_version': 1, 'format': 'long_tail_candidate_scene', 'candidate_kind': 'single_npc_reference',
            'candidate_id': candidate_id, 'generator': 'direct_trajectory_diffusion',
            'source_type': 'generated_candidate', 'prediction_source': 'diffusion_sample',
            'parent_scenario_id': metadata['scenario_id'], 'parent_group_id': metadata['parent_group_id'],
            'source_episode_id': metadata['episode_id'], 'source_path': metadata['source_path'],
            'source_sha256': metadata['source_sha256'], 'window_index': metadata['window_index'],
            'target_id': metadata['target_id'], 'seed': int(seed), 'sample_index': int(sample_index),
            'coordinate_frame': 'world', 'position_reference': reference, 'state_fields': list(VEHICLE_FIELDS),
            'units': {'position': 'm', 'velocity': 'm/s', 'time': 's', 'yaw': 'rad', 'a': 'm/s^2'},
            'dt': float(dt), 'anchor_timestamp': metadata['anchor_timestamp'],
            'target_history': {'states': metadata['target_history_world'].copy(), 'source': 'recorded'},
            'target_trajectory': {'states': states, 'relative_timestamps': np.arange(len(states))*dt,
                                  'mask': np.ones(len(states), dtype=bool)},
            'kinematics_method': 'preceding-interval position differences; low-speed yaw holds last heading',
            'generated_target_history': False, 'simulation_verified': False,
            'ego_response': 'not_simulated', 'other_vehicle_future': 'not_generated',
            'rssm_counterfactual_supported': False, 'risk': None, 'generation_metadata': dict(generation),
            'execution_requirements': ['Bind the simulator map and measured initial states.',
                                       'Validate NPC tracking and use a closed-loop ego controller.']}


def save_candidate(candidate, output_prefix):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    if json_path.exists() or npz_path.exists():
        raise ValueError('Candidate output already exists')
    payload = _json_safe(candidate)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(npz_path, target_states=candidate['target_trajectory']['states'],
                        target_history=candidate['target_history']['states'],
                        metadata_json=np.asarray(json.dumps(payload, ensure_ascii=False, allow_nan=False)))
    return {'json': str(json_path), 'npz': str(npz_path)}
