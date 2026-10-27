"""Meter-based displacement and explicitly provisional sampled kinematics."""
import numpy as np


PROVISIONAL_LIMITS = {'max_speed': 40., 'max_accel': 4., 'max_decel': 8.,
                      'max_jerk': 15., 'max_yaw_rate': 1., 'max_lateral_accel': 4.}


def displacement_metrics(predictions, truth):
    predictions, truth = np.asarray(predictions), np.asarray(truth)
    if predictions.ndim != 4 or predictions.shape[0] != len(truth) or predictions.shape[2:] != truth.shape[1:]:
        raise ValueError('Expected predictions[B,K,F,2] and truth[B,F,2]')
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite evaluation trajectory')
    distance = np.linalg.norm(predictions-truth[:, None], axis=-1)
    ade, fde = distance.mean(axis=-1), distance[..., -1]
    k = predictions.shape[1]
    diversity = [np.linalg.norm(predictions[:, a, -1]-predictions[:, b, -1], axis=-1)
                 for a in range(k) for b in range(a+1, k)]
    return {'mean_ADE_m': float(ade.mean()), 'mean_FDE_m': float(fde.mean()),
            'minADE_m': float(ade.min(axis=1).mean()), 'minFDE_m': float(fde.min(axis=1).mean()),
            'endpoint_diversity_m': float(np.stack(diversity).mean()) if diversity else 0.}


def constant_velocity(metadata, future, dt):
    times = np.arange(1, future+1)*dt
    return np.stack([m['target_anchor_local'][:2]+times[:, None]*m['target_anchor_local'][2:4]
                     for m in metadata])


def states_from_xy(future_xy, anchor, dt):
    future_xy, anchor = np.asarray(future_xy, dtype=float), np.asarray(anchor, dtype=float)
    if future_xy.ndim != 2 or future_xy.shape[1] != 2 or len(future_xy) < 1 or anchor.shape != (8,) or dt <= 0:
        raise ValueError('Invalid trajectory/anchor/dt')
    if not np.isfinite(future_xy).all() or not np.isfinite(anchor).all() or np.any(anchor[6:8] <= 0):
        raise ValueError('Finite positions and positive recorded dimensions required')
    states = np.tile(anchor, (len(future_xy)+1, 1))
    states[1:, :2] = future_xy
    velocity = np.diff(states[:, :2], axis=0)/dt
    states[1:, 2:4] = velocity
    speed = np.linalg.norm(velocity, axis=-1)
    yaw = float(anchor[4])
    for i, v in enumerate(velocity, 1):
        if speed[i-1] > 1e-3:
            yaw = float(np.arctan2(v[1], v[0]))
        states[i, 4] = yaw
    acceleration = np.diff(states[:, 2:4], axis=0)/dt
    tangent = np.stack((np.cos(states[1:, 4]), np.sin(states[1:, 4])), axis=-1)
    states[1:, 5] = np.sum(acceleration*tangent, axis=-1)
    return states


def motion_valid(states, dt, limits=None):
    limits = PROVISIONAL_LIMITS if limits is None else limits
    velocity = states[:, 2:4]
    acceleration = np.diff(velocity, axis=0)/dt
    anchor_acceleration = states[0, 5]*np.array([np.cos(states[0, 4]), np.sin(states[0, 4])])
    jerk = np.diff(np.vstack((anchor_acceleration, acceleration)), axis=0)/dt
    speed = np.linalg.norm(velocity[1:], axis=-1)
    tangent = np.stack((np.cos(states[1:, 4]), np.sin(states[1:, 4])), axis=-1)
    longitudinal = np.sum(acceleration*tangent, axis=-1)
    lateral = np.abs(velocity[1:, 0]*acceleration[:, 1]-velocity[1:, 1]*acceleration[:, 0])/np.maximum(speed, 1e-6)
    yaw_rate = np.arctan2(np.sin(np.diff(states[:, 4])), np.cos(np.diff(states[:, 4])))/dt
    return bool(np.isfinite(states).all() and np.max(speed) <= limits['max_speed']+1e-7
                and np.max(longitudinal) <= limits['max_accel']+1e-7
                and np.min(longitudinal) >= -limits['max_decel']-1e-7
                and np.max(np.linalg.norm(jerk, axis=-1)) <= limits['max_jerk']+1e-7
                and np.max(np.abs(yaw_rate)) <= limits['max_yaw_rate']+1e-7
                and np.max(lateral) <= limits['max_lateral_accel']+1e-7)
