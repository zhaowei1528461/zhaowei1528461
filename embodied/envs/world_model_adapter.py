"""Native SUMO observations -> optional RSSM backend; trajectory rules stay separate."""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np

try:
    from .scenario_risk import RiskEvaluator
except ImportError:  # Direct standalone script, without importing the training package.
    from scenario_risk import RiskEvaluator


def prediction_metrics(outputs, truth, truth_mask, recorded_cost=None, cost_threshold=.5):
    """Errors are in original normalized observation space, not meters/vehicle ADE."""
    truth, mask = np.asarray(truth, float), np.asarray(truth_mask, bool)
    if truth.ndim != 2 or truth.shape[1] != 31 or mask.shape != (len(truth),):
        raise ValueError('Expected truth[F,31] and truth_mask[F]')
    predicted = outputs.get('state') if outputs is not None else None
    errors = None
    if predicted is not None:
        predicted = np.asarray(predicted, float)
        if predicted.ndim != 2 or predicted.shape[1] != 31 or len(predicted) > len(truth):
            raise ValueError('Decoder state must be [F,31] in original observation units')
        length = len(predicted)
        usable = mask[:length] & np.isfinite(truth[:length]).all(-1) & np.isfinite(predicted).all(-1)
        errors = np.sqrt(np.mean((predicted - truth[:length])**2, axis=-1))
        errors[~usable] = np.nan
    steps = {}
    for step in (1, 5, 10, 30):
        available = errors is not None and len(errors) >= step and np.isfinite(errors[step-1])
        prefix = errors[:step] if errors is not None else np.array([])
        prefix = prefix[np.isfinite(prefix)]
        steps[str(step)] = {'state_rmse_at_step': float(errors[step-1]) if available else None,
                            'mean_state_rmse': float(prefix.mean()) if len(prefix) else None,
                            'valid_step_count': int(len(prefix)),
                            'unavailable_reason': None if available else 'no_decoder_or_insufficient_valid_ground_truth'}
    result = {'steps': steps, 'state_error_units': 'original_normalized_observation',
              'ADE': None, 'FDE': None,
              'trajectory_error_unavailable_reason': 'no_absolute_world_vehicle_decoder',
              'risk_recall': None, 'risk_positive_count': 0,
              'risk_recall_definition': 'head_cost >= threshold recalling recorded_cost >= threshold; not verified collision recall',
              'cost_threshold': float(cost_threshold), 'risk_unavailable_reason': 'no_cost_head_or_recorded_cost'}
    if outputs is not None and outputs.get('cost') is not None and recorded_cost is not None:
        costs, actual = np.asarray(outputs['cost'], float), np.asarray(recorded_cost, float)
        if costs.ndim != 1 or actual.shape != (len(truth),) or len(costs) > len(actual):
            raise ValueError('Cost predictions and labels must be scalar per step')
        usable = mask[:len(costs)] & np.isfinite(costs) & np.isfinite(actual[:len(costs)])
        positive = usable & (actual[:len(costs)] >= cost_threshold)
        count = int(positive.sum())
        result['risk_positive_count'] = count
        result['risk_unavailable_reason'] = None if count else 'no_positive_recorded_cost'
        if count:
            result['risk_recall'] = float(np.sum(positive & (costs >= cost_threshold)) / count)
    return result


class WorldModelAdapter:
    """Backend protocol: encode, rollout, teacher_force. Each receives explicit seed.

    ScenarioEpisode supplies raw 31D observations and interval-start actions.
    Windows alone cannot reconstruct original normalized nearest-vehicle slots.
    No interpolation or implicit inversion of the stage 1 normalizer is used.
    """
    def __init__(self, episode=None, backend=None, seed=0, model_dt=None, risk_config=None):
        self.episode, self.backend, self.seed = episode, backend, int(seed)
        self.model_dt = model_dt if model_dt is not None else (episode.dt if episode is not None else None)
        self.rules = RiskEvaluator(risk_config)

    def _unsupported(self, reason):
        return {'supported': False, 'reason': reason, 'model_version': getattr(self.backend, 'model_version', None),
                'prediction_source': 'rssm_unavailable', 'uncertainty': None,
                'capabilities': {'npc_plan_conditioning': False, 'world_vehicle_decoder': False,
                                 'calibrated_collision_probability': False}}

    @staticmethod
    def _finite_tree(value):
        if isinstance(value, Mapping):
            return bool(value) and all(WorldModelAdapter._finite_tree(v) for v in value.values())
        try:
            array = np.asarray(value)
            return bool(array.size and np.isfinite(array).all())
        except (TypeError, ValueError):
            return False

    @classmethod
    def _valid_prediction(cls, outputs, length):
        if not isinstance(outputs, Mapping) or not cls._finite_tree(outputs):
            return False
        for name, value in outputs.items():
            values = value.values() if name == 'latent' and isinstance(value, Mapping) else (value,)
            for item in values:
                array = np.asarray(item)
                if array.ndim < 1 or array.shape[0] != length:
                    return False
            if name == 'state' and np.asarray(value).shape != (length, 31):
                return False
            if name in ('cost', 'reward', 'cont', 'risk') and np.asarray(value).shape != (length,):
                return False
        return True

    def _indices(self, window):
        if self.episode is None:
            return None, 'source_episode_required_for_raw_observation'
        if window.episode_id != self.episode.episode_id or window.scenario_id != self.episode.scenario_id:
            return None, 'source_episode_identity_mismatch'
        if (self.model_dt is None or not math.isfinite(float(self.model_dt)) or self.model_dt <= 0 or
                not np.isclose(window.dt, self.model_dt, rtol=1e-6, atol=1e-7) or
                self.episode.dt is None or not np.isclose(self.episode.dt, self.model_dt)):
            return None, 'model_dt_mismatch'
        stamps = np.asarray(window.timestamps)
        h, f = len(window.history_ego), len(window.future_ego)
        if (h < 1 or stamps.shape != (h+f,) or not np.isfinite(stamps).all() or
                not np.allclose(np.diff(stamps), self.model_dt, atol=1e-7, rtol=1e-6)):
            return None, 'invalid_window_time_grid'
        indices = np.abs(stamps[:, None] - self.episode.timestamps[None]).argmin(axis=1)
        if (not np.allclose(stamps, self.episode.timestamps[indices], atol=1e-7, rtol=0) or
                not np.all(np.diff(indices) == 1)):
            return None, 'window_not_on_native_observation_grid'
        return indices, None

    @staticmethod
    def _observations(states):
        # Original Agent.preprocess needs is_terminal to derive cont. Encoder is state-only.
        length = len(states)
        return {'state': np.asarray(states, np.float32)[None],
                'is_first': np.zeros((1, length), bool),
                'is_terminal': np.zeros((1, length), bool)}

    def encode_history(self, window):
        if self.backend is None:
            return self._unsupported('no_rssm_backend')
        indices, reason = self._indices(window)
        if reason:
            return self._unsupported(reason)
        h = len(window.history_ego)
        selected = indices[:h]
        raw = self.episode.raw_states[selected]
        if (raw.shape != (h, 31) or not self.episode.raw_state_mask[selected].all() or not np.isfinite(raw).all()):
            return self._unsupported('missing_raw_history_observation')
        if (h > 1 and not self.episode.ego_action_mask[selected[:-1]].all()):
            return self._unsupported('missing_history_action')
        previous = np.zeros((1, h, 1), np.float32)
        if h > 1:
            previous[0, 1:] = self.episode.ego_actions[selected[:-1]]
        if not np.isfinite(previous).all():
            return self._unsupported('nonfinite_history_action')
        observations = self._observations(raw)
        observations['is_first'][0, 0] = True  # Explicit cold start at this window, not full-episode filtering.
        latent = self.backend.encode(observations, previous, seed=self.seed)
        if not isinstance(latent, Mapping) or not self._finite_tree(latent):
            return self._unsupported('invalid_history_posterior')
        return {'supported': True, 'latent': latent, 'input_shape': list(observations['state'].shape),
                'model_version': self.backend.model_version, 'prediction_source': 'rssm_history_posterior',
                'cold_start': True, 'native_dt': float(self.model_dt),
                'validity_flags': {'raw_state_abs_gt_5': bool(np.any(np.abs(raw) > 5.)),
                                   'range_hint_is_calibrated_ood': False}}

    def predict_future(self, window, ego_policy=None, npc_plan=None, horizon=None):
        if npc_plan is not None:
            return self._unsupported('npc_plan_not_supported')
        if ego_policy is not None:
            return self._unsupported('ego_policy_rollout_not_supported_use_recorded_actions')
        encoded = self.encode_history(window)
        if not encoded['supported']:
            return encoded
        f = len(window.future_ego)
        horizon = f if horizon is None else horizon
        if isinstance(horizon, bool) or int(horizon) != horizon or horizon <= 0 or horizon > f:
            raise ValueError('Horizon must be a positive integer within the window future')
        horizon = int(horizon)
        indices, _ = self._indices(window)
        h = len(window.history_ego)
        action_indices = indices[h-1:h-1+horizon]
        actions = self.episode.ego_actions[action_indices]
        if not self.episode.ego_action_mask[action_indices].all() or not np.isfinite(actions).all():
            return self._unsupported('missing_future_ego_action')
        future_indices = indices[h:h+horizon]
        truth = self.episode.raw_states[future_indices].copy()
        truth_mask = self.episode.raw_state_mask[future_indices].copy() & np.isfinite(truth).all(-1)
        alignment = self.episode.metadata.get('state_alignment', '') or ''
        stale = 'terminal_state_may_precede_snapshot' in alignment
        if stale:
            truth_mask[future_indices == len(self.episode.timestamps)-1] = False
        free = self.backend.rollout(encoded['latent'], actions[None], seed=self.seed+1)
        if not self._valid_prediction(free, horizon):
            return self._unsupported('invalid_free_rollout_output')
        # Teacher forcing cannot skip an invalid observation and still claim a posterior sequence.
        available = int(np.flatnonzero(~truth_mask)[0]) if (~truth_mask).any() else horizon
        teacher = None
        if available:
            teacher = self.backend.teacher_force(encoded['latent'], self._observations(truth[:available]),
                                                  actions[None, :available], seed=self.seed+2)
            if not self._valid_prediction(teacher, available):
                return self._unsupported('invalid_teacher_forcing_output')
        recorded = self.episode.events.get('cost')
        recorded = recorded[future_indices] if recorded is not None else None
        return {'supported': True, 'reason': None, 'model_version': self.backend.model_version,
                'prediction_source': 'rssm_recorded_ego_actions', 'cold_start': True,
                'native_dt': float(self.model_dt), 'seed': self.seed, 'horizon': horizon,
                'free_rollout': free, 'teacher_forcing': teacher,
                'ground_truth': truth, 'ground_truth_mask': truth_mask,
                'metrics': {'free_rollout': prediction_metrics(free, truth, truth_mask, recorded),
                            'teacher_forcing': prediction_metrics(teacher, truth, truth_mask, recorded)},
                'predicted_collision_probability': None, 'uncertainty': None,
                'capabilities': {'npc_plan_conditioning': False, 'world_vehicle_decoder': False,
                                 'normalized_observation_decoder': 'state' in free,
                                 'calibrated_collision_probability': False},
                'validity_flags': {**encoded['validity_flags'], 'teacher_forcing_steps': available,
                                   'terminal_observation_alignment_guard': stale,
                                   'actions_within_training_range': bool(np.all(np.abs(actions) <= 1.+1e-6)),
                                   'counterfactual_closed_loop': False}}

    def evaluate_risk(self, window, candidate=None):
        report = self.rules.evaluate_risk(window, candidate)
        report.validity_flags['rssm_backend_available'] = self.backend is not None
        report.validity_flags['rssm_predictions_used_for_geometry'] = False
        return report
