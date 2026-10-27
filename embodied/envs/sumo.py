import numpy as np
import embodied
# from .sumoenv import Sumoenv  # 你的原始 SUMO 环境类
from .sumo_merge import Sumoenv

from .sumo_merge import Sumoenv

class SumoEnv(embodied.Env):

    def __init__(self, config=None, **kwargs):
        print(
            "[SumoEnv init] "
            f"worker_index={kwargs.get('worker_index', 0)} "
            f"env_amount={kwargs.get('env_amount', 1)} "
            f"env_mode={kwargs.get('env_mode', 'train')}")
        danger_config = dict(kwargs.get('danger', {}))
        if 'danger_enabled' in kwargs:
            danger_config['enabled'] = bool(kwargs['danger_enabled'])
        if 'danger_generation_enabled' in kwargs:
            danger_config['generation_enabled'] = bool(
                kwargs['danger_generation_enabled'])
        if kwargs.get('env_mode', 'train') == 'eval':
            # Independent episodes belong to training workers. The sweep can
            # still record the existing danger events for the frozen library.
            danger_config['collect_episodes'] = False
        self._env = Sumoenv(
            worker_index=kwargs.get('worker_index', 0),
            env_amount=kwargs.get('env_amount', 1),
            env_mode=kwargs.get('env_mode', 'train'),
            danger_config=danger_config,
            danger_output_dir=kwargs.get('danger_output_dir', ''))
        self._last_acc = 0
        self._step = 0
        self._step_over1 = 0
        self._is_first = True

    @property
    def obs_space(self):
        return {
            'state': embodied.Space(np.float32, (31,)),
            'reward': embodied.Space(np.float32),
            'cost': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            'reset': embodied.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': embodied.Space(np.float32, (1,), -1.0, 1.0),
            'reset': embodied.Space(bool),
        }

    def step(self, action):
        if action.get('reset', False):
            return self._reset()
        else:

            state, reward, cost, done, info = self._env.step(
                action['action'][0])
            self._step += 1
            obs = {
                'state': state.astype(np.float32),
                'reward': np.float32(reward),
                'cost': np.float32(cost),
                'is_first': self._is_first,
                'is_last': done,
                'is_terminal': done,
                'reset': False,
            }
            self._is_first = False  # 只在第一次为 True
            return obs

    def _reset(self):

        state = self._env.reset()
        self._last_acc = 0
        self._step = 0
        self._step_over1 = 0
        self._is_first = True
        return {
            'state': state.astype(np.float32),
            'reward': np.float32(0),
            'cost': np.float32(0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'reset': True,
        }

    def render(self):
        return None
    
    def start(self,gui):
        try:
          self.close  
        except:
            pass
        self._env.start(gui)

    def set_danger_mining(
            self, active, record_events=None, collect_episodes=None,
            update_library=None):
        return self._env.set_danger_mining(
            active, record_events=record_events,
            collect_episodes=collect_episodes, update_library=update_library)

    def set_scenario_mode(self, mode):
        return self._env.set_scenario_mode(mode)

    def get_scenario_count(self):
        return self._env.get_scenario_count()

    def get_current_scenario(self):
        return self._env.get_current_scenario()

    def close(self):   
        self._env.close()
