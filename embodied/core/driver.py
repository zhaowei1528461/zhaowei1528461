import collections

import numpy as np

from .basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._acts = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
            for k, v in self._env.act_space.items()}
        self._acts['reset'] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None
        self._cum_log_weight = np.zeros(len(self._env), np.float32)
        self._cum_log_weight_dual = np.zeros(len(self._env), np.float32)
        self._last_dual_driver_print = None
        self._episode_dual_mask = np.zeros(len(self._env), dtype=bool)

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(
            self, policy, steps=0, episodes=0,
            lag=0.0, lag_p=0.0, lag_i=0.0, lag_d=0.0,
            dual_enabled=0.0, dual_ratio=0.0, dual_blend=1.0,
            dual_beta=0.5):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(
                policy, step, episode, lag, lag_p, lag_i, lag_d,
                dual_enabled, dual_ratio, dual_blend, dual_beta)

    def _step(
            self, policy, step, episode, lag, lag_p, lag_i, lag_d,
            dual_enabled, dual_ratio, dual_blend, dual_beta):
        assert all(len(x) == len(self._env) for x in self._acts.values())

        acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
        obs = self._env.step(acts)

        obs['lagrange_penalty'] = lag * np.ones(len(self._env))
        obs['lagrange_p'] = lag_p * np.ones(len(self._env))
        obs['lagrange_i'] = lag_i * np.ones(len(self._env))
        obs['lagrange_d'] = lag_d * np.ones(len(self._env))

        ratio = float(np.clip(dual_ratio, 0.0, 1.0))
        enabled = bool(dual_enabled)
        dual_count = int(round(len(self._env) * ratio)) if enabled else 0
        dual_count = min(max(dual_count, 0), len(self._env))

        requested_dual_mask = np.zeros(len(self._env), dtype=bool)
        if dual_count > 0:
            requested_dual_mask[-dual_count:] = True
        new_episode = np.asarray(obs['is_first'], dtype=bool)
        self._episode_dual_mask[new_episode] = requested_dual_mask[new_episode]
        dual_mask = self._episode_dual_mask.copy()
        driver_state = (enabled, round(ratio, 6), dual_count)
        if driver_state != self._last_dual_driver_print:
            print(
                '[driver dual] '
                f'enabled={enabled} ratio={ratio:.3f} '
                f'envs={len(self._env)} dual_count={dual_count} '
                f'active_episode_dual={int(dual_mask.sum())}')
            self._last_dual_driver_print = driver_state

        obs['log_policy_dual_enabled'] = np.full(
            len(self._env), float(enabled), np.float32)
        obs['log_policy_dual_ratio'] = np.full(
            len(self._env), ratio, np.float32)
        obs['log_policy_dual_blend'] = np.full(
            len(self._env), float(dual_blend), np.float32)
        obs['log_policy_dual_mask'] = dual_mask.astype(np.float32)

        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs

        acts, self._state = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}

        instant_log_weight = np.asarray(
            acts.get('log_weight', np.zeros(len(self._env), np.float32)),
            dtype=np.float32)
        instant_log_weight_dual = np.asarray(
            acts.get('log_weight_dual', np.zeros(len(self._env), np.float32)),
            dtype=np.float32)

        # Store trajectory-level IS weights for the current state. The
        # instantaneous ratios from this action are folded into the next state.
        self._cum_log_weight[dual_mask] = 0.0
        self._cum_log_weight_dual[~dual_mask] = 0.0
        acts['log_weight'] = self._cum_log_weight.copy()
        acts['log_weight_dual'] = self._cum_log_weight_dual.copy()

        if obs['is_last'].any():
            mask = 1 - obs['is_last']
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}

        acts['reset'] = obs['is_last'].copy()
        self._acts = acts

        trns = {**obs, **acts}

        if obs['is_first'].any():
            for i, first in enumerate(obs['is_first']):
                if first:
                    self._eps[i].clear()

        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1

        if obs['is_last'].any():
            for i, done in enumerate(obs['is_last']):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            self._cum_log_weight[obs['is_last']] = 0.0
            self._cum_log_weight_dual[obs['is_last']] = 0.0

        active = ~obs['is_last'].astype(bool)
        beta = float(np.clip(dual_beta, 0.0, 1.0))
        main_active = active & (~dual_mask)
        dual_active = active & dual_mask
        self._cum_log_weight[main_active] = beta * (
            self._cum_log_weight[main_active] + instant_log_weight[main_active])
        self._cum_log_weight_dual[dual_active] = beta * (
            self._cum_log_weight_dual[dual_active] +
            instant_log_weight_dual[dual_active])

        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
