"""Optional original RSSM backend. Imports JAX only when explicitly instantiated.

Use on the original training server with its dependencies and saved config.
It reads the original checkpoint container, never creates missing parameters,
and never calls training or restores replay buffers.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import sys
import types

import numpy as np


def file_sha256(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_checkpoint(agent, filename):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(str(filename))
    import embodied
    checkpoint = embodied.Checkpoint(str(filename), log=False, parallel=False)
    checkpoint.agent = agent
    checkpoint.load(keys=['agent'])  # The original container can include step/replay, which are ignored.


class NinjaxRSSMBackend:
    @classmethod
    def from_checkpoint(cls, filename, config_path=None, config_name=None, platform='cpu'):
        filename = Path(filename)
        if not filename.is_file():
            raise FileNotFoundError(str(filename))
        config_path = Path(config_path) if config_path is not None else filename.parent / 'config.yaml'
        if not config_path.is_file() and config_name is None:
            raise FileNotFoundError('Saved model config required: ' + str(config_path))
        # The standalone script's sys.path is envs/, not repository root.
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import embodied
        import jax
        import jax.numpy as jnp
        import ruamel.yaml as yaml
        jax.config.update('jax_platform_name', platform)
        package = '_long_tail_original_rssm'
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(root)]
            sys.modules[package] = module
        original = importlib.import_module(package + '.agent')
        nj = importlib.import_module(package + '.ninjax')
        jaxutils = importlib.import_module(package + '.jaxutils')
        if config_path.is_file():
            with config_path.open(encoding='utf-8') as stream:
                config = embodied.Config(yaml.YAML(typ='safe').load(stream))
            config_source = str(config_path)
        else:
            config = embodied.Config(original.Agent.configs['defaults']).update(original.Agent.configs[config_name])
            config_source = 'explicit_config_name:' + config_name
        if config.encoder.mlp_keys != 'state' or config.encoder.cnn_keys != '$^':
            raise ValueError('This adapter supports the original state-only SUMO encoder, not arbitrary observation schemas')
        jaxutils.COMPUTE_DTYPE = getattr(jnp, config.jax.precision)
        obs_space = {'state': embodied.Space(np.float32, (31,)),
                     **{key: embodied.Space(np.float32) for key in ('reward', 'cost')},
                     **{key: embodied.Space(bool) for key in ('is_first', 'is_last', 'is_terminal', 'reset')}}
        act_space = {'action': embodied.Space(np.float32, (1,), -1., 1.), 'reset': embodied.Space(bool)}
        inner = original.Agent.inner(obs_space, act_space, embodied.Counter(), config, name='agent')
        backend = cls(inner, nj, jax, platform, file_sha256(filename))
        load_original_checkpoint(backend, filename)
        backend.config_source = config_source
        backend.config_sha256 = file_sha256(config_path) if config_path.is_file() else None
        backend.checkpoint_path = str(filename)
        return backend

    def __init__(self, agent, nj, jax, platform, checkpoint_sha256):
        self.agent, self.nj, self.jax = agent, nj, jax
        self.device = jax.devices(platform)[0]
        self.varibs = None
        self.model_version = 'checkpoint_sha256:' + checkpoint_sha256

    def save(self):
        return self.jax.device_get(self.varibs)

    def load(self, state):
        if not isinstance(state, dict) or not any(str(k).startswith('agent/wm/') for k in state):
            raise ValueError('Checkpoint agent entry does not contain original agent/wm parameters')
        self.varibs = self.jax.device_put(state, self.device)

    def _call(self, function, seed):
        output, _ = self.nj.pure(function)(self.varibs, self.jax.random.PRNGKey(int(seed)),
                                            create=False, modify=False)
        return self.jax.tree_util.tree_map(lambda value: np.asarray(self.jax.device_get(value)), output)

    def _device(self, value):
        return self.jax.device_put(value, self.device)

    def encode(self, observations, incoming_actions, seed):
        observations, actions = self._device(observations), self._device(incoming_actions)
        def forward():
            data = self.agent.preprocess(observations)
            posterior, _ = self.agent.wm.rssm.observe(self.agent.wm.encoder(data), actions, data['is_first'])
            return {k: v[:, -1] for k, v in posterior.items()}
        return self._call(forward, seed)

    def _decode(self, latent):
        wm = self.agent.wm
        result = {'latent': latent}
        if 'decoder' in wm.heads:
            result.update({k: v.mode() for k, v in wm.heads['decoder'](latent).items()})
        for name in ('reward', 'cost', 'cont', 'risk'):
            if name in wm.heads:
                result[name] = wm.heads[name](latent).mean()
        return result

    def rollout(self, latent, actions, seed):
        latent, actions = self._device(latent), self._device(actions)
        output = self._call(lambda: self._decode(self.agent.wm.rssm.imagine(actions, latent)), seed)
        return self.jax.tree_util.tree_map(lambda value: value[0], output)

    def teacher_force(self, latent, observations, incoming_actions, seed):
        latent, observations, actions = self._device(latent), self._device(observations), self._device(incoming_actions)
        def forward():
            data = self.agent.preprocess(observations)
            posterior, _ = self.agent.wm.rssm.observe(self.agent.wm.encoder(data), actions, data['is_first'], latent)
            return self._decode(posterior)
        return self.jax.tree_util.tree_map(lambda value: value[0], self._call(forward, seed))
