# embodied/envs/onsite.py
import numpy as np
import embodied
from SafeDreamer.embodied.envs.paremeters import agent_par   # zhe ge mei chuan guo lai SafeDreamer/embodied/envs/paremeters.py
from .utils import Sumoenv  # 这里按你的实际路径；你发的代码里类名就是 Sumoenv

class SumoEnv(embodied.Env):
    """Adapter for onsite SUMO-like env to the embodied.Env interface."""

    def __init__(self, *args, **kwargs):
        # 吃掉上层多余参数（如 platform/task/mode 等）
        self._config = kwargs.get('config', None)
        # 你也可以把 kwargs 里的自定义超参传给 Sumoenv(...)
        self._env = Sumoenv(state_dim=agent_par['state_dim'],action_dim=agent_par['action_dim'],
                            cnn=agent_par['cnn'],safe_rule=agent_par['safe_rule'],cnn_state=agent_par['cnn_state'], 
                            use_spare_reward=agent_par['use_spare_reward'], sparse_episode=agent_par['change_episode'],
                            nb_ego_states=agent_par['nb_ego_states'],nb_states_per_vehicle=agent_par['nb_states_per_vehicle'],sensor_nb_vehicles=agent_par['sensor_nb_vehicles'])
        self._episode = 0
        self._is_first = True

        # 缓存当前观测向量（来自 Sumoenv.ingest() 之后的 _next_vec）
        self._cur_vec = None

    # --- spaces ---
    @property
    def obs_space(self):
        # 你的 Sumoenv.observation_space 里写的是 (56,), 但 get_features() 最后 concat 了 3 个 tracking_error
        # 所以这里按 59 给空间；dtype 和范围由上层模型处理
        return {
            'vector': embodied.Space(np.float32, (59,)),
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
        }

    @property
    def act_space(self):
        # 对齐 Sumoenv.action_space：Box(-1, 1, (2,))
        return {
            'action': embodied.Space(np.float32, (2,), -1.0, 1.0),
            'reset': embodied.Space(bool),
        }

    # BatchEnv 里会断言 len(env) == 0
    def __len__(self):
        return 0

    # --- main API ---
    def step(self, action):
        # 处理外部请求重置
        if action.get('reset', False):
            return self._reset()

        # 把 Sumoenv 中缓存的下一帧观测搬到当前帧
        vec = self._pop_next_vector()

        # 调用底层 step 计算 reward / done
        # 你的 Sumoenv.step(action, episode) 返回: reward, done, info
        # 注意：Sumoenv.step 并不会推进仿真，只基于 observation/test_info 计算奖励和终止标志
        reward, done, _info = self._env.step(action['action'], self._episode)

        obs = {
            'vector': vec.astype(np.float32),
            'reward': np.float32(reward),
            'is_first': bool(self._is_first),
            'is_last': bool(done),
            'is_terminal': bool(done),
        }
        self._is_first = False
        if done:
            self._episode += 1
        return obs

    def _reset(self):
        # 重置底层环境（你的 reset 接口需要 scenario/lane_info 时可从 config 注入）
        self._env.reset()  # 按需传 scenario_dict/lane_info

        # 清空本地状态
        self._is_first = True
        self._cur_vec = None

        # 返回“空”首帧（等外部仿真 first ingest 后，下一次 step 会输出真实观测）
        vec = self._safe_vector()
        return {
            'vector': vec.astype(np.float32),
            'reward': np.float32(0.0),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    # --- helpers ---
    def _pop_next_vector(self):
        """从 Sumoenv 的双缓冲中取出下一帧向量；若没有则返回上一次或全零。"""
        # 你的 Sumoenv 在 ingest(observation) 时会填充:
        # self._next_obs_raw, self._next_vec, self._has_next = True
        if getattr(self._env, '_has_next', False):
            self._cur_vec = np.array(self._env._next_vec, dtype=np.float32)
            # 清标志，防止重复读取
            self._env._cur_obs_raw = self._env._next_obs_raw
            self._env._cur_vec = self._env._next_vec
            self._env._next_obs_raw = None
            self._env._next_vec = None
            self._env._has_next = False

        if self._cur_vec is None:
            self._cur_vec = self._safe_vector()
        return self._cur_vec

    def _safe_vector(self):
        # 给一个安全的占位观测（59 维）
        return np.zeros((59,), dtype=np.float32)

    # 可选直通
    def start(self, gui=False):
        if hasattr(self._env, 'start'):
            self._env.start(gui)

    def render(self):
        return None

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()
