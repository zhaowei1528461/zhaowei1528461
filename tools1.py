# ========= 1) Space: 你自己的最小空间类型 =========
import numpy as np

class Space:
    def __init__(self, dtype, shape, low, high,discrete=None):
        # shape 支持标量(())
        if shape in (None, ()):
            self.shape = ()
        elif isinstance(shape, (list, tuple)):
            self.shape = tuple(shape)
        elif isinstance(shape, int):
            self.shape = (shape,)
        else:
            raise TypeError("shape must be None/(), int or tuple")
        self.dtype = dtype
        self.low = low
        self.high = high
        self.discrete = discrete
    def __repr__(self):
        return f"Space(dtype={getattr(self.dtype,'__name__',self.dtype)}, shape={self.shape}, low={self.low}, high={self.high},discrete={self.discrete}))"

# ========= 2) 将旧的 obs/act 空间转换成你需要的定义 =========
def _scalar_bound(x):
    # 低/高可能是数组或标量，这里统一成标量（动作裁剪用）
    try:
        return float(np.min(x))
    except Exception:
        return float(x)

def convert_spaces(old_obs_space: dict, old_act_space: Space):
    """
    old_obs_space: {'is_first': Space, 'is_terminal': Space, 'vector': Space}
    old_act_space: Space(shape=(2,), dtype=float32, low=-1.0, high=1.0)    # 举例
    ->
    new_obs_space: {
      'state','reward','cost','is_first','is_last','is_terminal','reset'
    }
    new_act_space: {
      'action': 与旧动作同 shape/dtype/范围,
      'reset':  标量 bool
    }
    """
    new_obs_space = {
        "state":       old_obs_space["vector"],  # 仅改名，不改形状/范围
        "reward":      Space(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
        "cost":        Space(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
        "is_first":    old_obs_space["is_first"],
        "is_last":     Space(dtype=bool, shape=(), low=False, high=True),
        "is_terminal": old_obs_space["is_terminal"],
        "reset":       Space(dtype=bool, shape=(), low=False, high=True),
    }

    # lo = -1
    # hi = 1
    new_act_space = {
        "action": Space(dtype=old_act_space.dtype, shape=old_act_space.shape, low=-1, high=1),
        "reset":  Space(dtype=bool, shape=(), low=False, high=True),
    }
    return new_obs_space, new_act_space

# ========= 3) 运行时映射函数（字段对应；不改维度） =========
def map_obs(env_obs: dict, reward, done, info, *, is_first=False, reset=False):
    """
    env_obs: {'is_first', 'is_terminal', 'vector'}
    return:  {'state','reward','cost','is_first','is_last','is_terminal','reset'}
    """
    # is_first 如果 env_obs 里没有，就用形参 is_first
    is_first_flag = env_obs.get("is_first", is_first)
    is_terminal = env_obs["is_terminal"]
    cost_val = (info.get("cost", 0.0) if isinstance(info, dict) else 0.0)

    return {
        "state":       env_obs["vector"],                                 # 不动形状/类型
        "reward":      np.asarray(reward, dtype=np.float32),              # 标量
        "cost":        np.asarray(cost_val, dtype=np.float32),            # 标量
        "is_first":    bool(is_first_flag),
        "is_last":     bool(done),
        "is_terminal": bool(np.asarray(is_terminal).item() if np.ndim(is_terminal)==0 else bool(is_terminal)),
        "reset":       bool(reset),
    }

def map_action(action_dict: dict, *, old_act_shape, old_low, old_high):
    """
    输入：{'action': 与旧 env 完全相同 shape（如 (2,)）, 'reset': bool}
    输出：旧 env 需要的动作（np.ndarray，shape 保持一致）
    """
    a = np.asarray(action_dict["action"], dtype=np.float32).reshape(old_act_shape)
    a = np.clip(a, _scalar_bound(old_low), _scalar_bound(old_high))
    return a