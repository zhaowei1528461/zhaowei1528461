import numpy as np

class RandomAgent:

  def __init__(self, act_space):
    self.act_space = act_space

  def policy(self, obs, state=None, mode='train'):
    # print('obs_policy',obs)

    batch_size = len(next(iter(obs.values())))
    # print('batch_size',batch_size)
    # print('act_space',self.act_space)



    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    # act['action'] = [0,0,0]
    # act['reset']
    act['log_entropy'] = np.zeros(batch_size)
    act['log_action_mean'] = np.zeros((batch_size, ) + self.act_space['action'].shape)
    act['log_action_std'] = np.zeros((batch_size, ) + self.act_space['action'].shape)
    act['log_plan_num_safe_traj'] = np.zeros(batch_size)
    act['log_plan_ret'] = np.zeros(batch_size)
    act['log_plan_cost'] = np.zeros(batch_size)
    act['log_lagrange_penalty'] = np.zeros(batch_size)
    act['log_behavior_logp'] = np.zeros(batch_size, np.float32)
    act['log_main_logp'] = np.zeros(batch_size, np.float32)
    act['log_dual_logp'] = np.zeros(batch_size, np.float32)
    act['log_use_dual'] = np.zeros(batch_size, np.float32)
    act['log_main_risk'] = np.zeros(batch_size, np.float32)
    act['log_risk_threshold'] = np.zeros(batch_size, np.float32)
    act['log_weight'] = np.zeros(batch_size, np.float32)
    act['log_weight_dual'] = np.zeros(batch_size, np.float32)
    act['log_behavior_policy_id'] = np.zeros(batch_size, np.float32)
    return act, state
