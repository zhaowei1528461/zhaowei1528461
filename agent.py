import embodied
import jax
import jax.numpy as jnp
import optax
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
from jax import debug

import jax
import jax.numpy as jnp

# 保持向后兼容
try:
    DeviceArray = jax.numpy.DeviceArray
except AttributeError:
    DeviceArray = jax.Array

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def finite_stats(name, x):
  x = x.astype(jnp.float32)
  finite = jnp.isfinite(x)
  safe_x = jnp.where(finite, x, 0.0)
  return {
      f'{name}_all_finite': finite.all().astype(jnp.float32),
      f'{name}_any_nan': jnp.isnan(x).any().astype(jnp.float32),
      f'{name}_any_inf': jnp.isinf(x).any().astype(jnp.float32),
      f'{name}_maxabs': jnp.abs(safe_x).max(),
  }


def mean_metric_values(values):
  total = values[0]
  for value in values[1:]:
    total = total + value
  return total / len(values)


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())
  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    elif config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.task_behavior.ac, self.wm, self.act_space, self.config, name='expl_behavior')
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def freeze_reference_policy(self):
    """Snapshot the bootstrap actor for post-training KL regularisation."""
    if not hasattr(self.task_behavior, 'freeze_reference_policy'):
      return {'reference_policy_frozen': 0.0}
    return self.task_behavior.freeze_reference_policy()

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    if self.config.expl_behavior in ['PIDPlanner']:
      expl_state['lagrange_penalty'] = obs['lagrange_penalty']
      task_state['lagrange_penalty'] = obs['lagrange_penalty']
      expl_state['lagrange_p'] = obs['lagrange_p']
      task_state['lagrange_p'] = obs['lagrange_p']
      expl_state['lagrange_i'] = obs['lagrange_i']
      task_state['lagrange_i'] = obs['lagrange_i']
      expl_state['lagrange_d'] = obs['lagrange_d']
      task_state['lagrange_d'] = obs['lagrange_d']
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    dual_enabled = obs.get('log_policy_dual_enabled', None)
    dual_ratio = obs.get('log_policy_dual_ratio', None)
    dual_blend = obs.get('log_policy_dual_blend', None)
    dual_mask = obs.get('log_policy_dual_mask', None)
    #self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(
        latent, task_state, mode=mode,
        dual_enabled=dual_enabled, dual_ratio=dual_ratio,
        dual_blend=dual_blend, dual_mask=dual_mask)
    # print('agent.py87', task_outs)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)#多策略的探索机制可以加在这里吗？
    if mode == 'eval':
      # print(90)
      if self.config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs = expl_outs
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])

      else:
        outs = task_outs
        action_out = outs['action']
        if hasattr(action_out, 'sample'):
          outs['action'] = action_out.sample(seed=nj.rng())
        else:
          outs['action'] = action_out

    elif mode == 'explore':
      # Preserve dual-mask action selection during the initial exploration
      # window when exploration is configured to reuse the task behavior.
      outs = task_outs if self.expl_behavior is self.task_behavior else expl_outs
      if self.config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
      else:
        # outs['log_entropy'] = outs['action'].entropy()
        # outs['action'] = outs['action'].sample(seed=nj.rng())
        action_out = outs['action']
        if hasattr(action_out, 'sample') and hasattr(action_out, 'entropy'):
            outs['log_entropy'] = action_out.entropy()
            outs['action'] = action_out.sample(seed=nj.rng())
        else:
            outs['log_entropy'] = jnp.zeros(action_out.shape[:1], dtype=jnp.float32)
            outs['action'] = action_out

        # print(105, outs['action'])
    elif mode == 'train':
      if self.config.expl_behavior in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
        outs = expl_outs
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
      else:
        outs = task_outs
        action_out = outs['action']
        if hasattr(action_out, 'sample') and hasattr(action_out, 'entropy'):
            outs['log_entropy'] = action_out.entropy()
            outs['action'] = action_out.sample(seed=nj.rng())
        else:
            outs['log_entropy'] = jnp.zeros(action_out.shape[:1], dtype=jnp.float32)
            outs['action'] = action_out
    if self.config.expl_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      outs['log_plan_action_mean'] = jnp.zeros(outs['action'].shape) 
      outs['log_plan_action_std'] = jnp.zeros(outs['action'].shape)
      outs['log_plan_num_safe_traj'] = jnp.zeros(outs['action'].shape[:1])
      outs['log_plan_ret'] = jnp.zeros(outs['action'].shape[:1])
      outs['log_plan_cost'] = jnp.zeros(outs['action'].shape[:1])
    if self.config.expl_behavior in ['PIDPlanner']:
      outs['log_lagrange_penalty'] = obs['lagrange_penalty'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_p'] = obs['lagrange_p'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_i'] = obs['lagrange_i'] * jnp.ones(outs['action'].shape[:1])
      outs['log_lagrange_d'] = obs['lagrange_d'] * jnp.ones(outs['action'].shape[:1])

    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    for key in (
        'log_behavior_logp',
        'log_main_logp',
        'log_dual_logp',
        'log_use_dual',
        'log_main_risk',
        'log_risk_threshold',
        'log_weight',
        'log_weight_dual',
        'log_behavior_policy_id',
    ):
      if key in data:
        value = data[key].astype(jnp.float32)
        metrics[f'batch_{key}_mean'] = value.mean()
        metrics[f'batch_{key}_maxabs'] = jnp.abs(value).max()
    if 'log_use_dual' in data:
      metrics['batch_log_use_dual_nonzero'] = (
          jnp.abs(data['log_use_dual'].astype(jnp.float32)) > 0).mean()
    # print('6666666666666666666666666666666666')
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)    #into  ac train
    metrics.update(mets)
    if self.config.expl_behavior != 'None' and self.config.expl_behavior not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    if 'model_loss_mean' in metrics:
      metrics['model_loss'] = metrics['model_loss_mean']
    policy_losses = [
        value for key, value in metrics.items()
        if key.endswith('actor_opt_loss')]
    if policy_losses:
      metrics['policy_loss'] = mean_metric_values(policy_losses)
    critic_losses = [
        value for key, value in metrics.items()
        if key.endswith('critic_opt_loss')]
    if critic_losses:
      metrics['critic_loss'] = mean_metric_values(critic_losses)
    outs = {}
    # print('8888888888888888888888888888888888')
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def report_eval(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report_eval(data))
    return report



  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    if self.config.use_cost:
      self.heads['cost'] = nets.MLP((), **config.cost_head, name='cost')
    if self.config.use_risk_model:  #加入风险头
      self.heads['risk'] = nets.MLP((), **config.risk_head, name='risk')
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    # print('lossstate',state)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      if key == 'cost':
        condition = jnp.greater_equal(data['cost'], 1.0)
        loss = -dist.log_prob(data['cost'].astype(jnp.float32))
        loss = jnp.where(condition, self.config.cost_weight * loss, loss)
      else:
        loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon, use_planner=False):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    if use_planner:
      keys += ['action_mean','action_std', 'lagrange_multiplier', 'penalty_multiplier']
      start = {k: v for k, v in start.items() if k in keys}
      start['action'] = policy(start,0)
      def step(prev, current_horizon): # add the current_horizon
        prev = prev.copy()
        action_mean = prev['action_mean']
        action_std = prev['action_std']
        state = self.rssm.img_step(prev, prev.pop('action'))
        return {**state, 'action_mean':action_mean, 'action_std':action_std, 'action': policy(prev,current_horizon+1)}
    else:
      start = {k: v for k, v in start.items() if k in keys}
      start['action'] = policy(start)
      def step(prev, _):
        prev = prev.copy()
        state = self.rssm.img_step(prev, prev.pop('action'))
        action = policy(state)
        # 2. 用 jax.device_get() 转成普通 NumPy 数组（强制执行计算）
        # debug.print("动作: {}", action)
        # 3. 打印具体数值（可以看形状、范围、前几个元素）
        # debug.print("动作 min={}, max={}", action.min(), action.max())
        return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    traj = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    #这个部分还要加入
    reward = self.heads['reward'](traj).mean()
    traj['reward'] = reward
    if self.config.use_cost:
      if self.config.use_cost_model and 'cost' in self.heads:
        cost = self.heads['cost'](traj).mean()
        
      else:
        cost = cost_from_state(self, traj)   #todo
      traj['cost'] = cost
      
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], data['action'][:6, :5],
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def report_eval(self, data_expand):
    state = self.initial(len(data_expand['is_first']))
    report = {}
    report.update(self.loss(data_expand, state)[-1][-1])
    context, _ = self.rssm.observe(
        self.encoder(data_expand)[:, :5], data_expand['action'][:, :5],
        data_expand['is_first'][:, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(data_expand['action'][:, 5:], start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = jaxutils.video_grid(model)
    for key in self.heads['decoder'].mlp_shapes.keys():
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      report[f'openl_{key}'] = model
      if 'openl_observation' in report.keys() and not self.config.use_cost:
        report[f'openl_cost'] = self.cost_from_recon(report['openl_observation'])
    return report

  def cost_from_recon(self, recon):
    # jax format
    hazards_size = 0.25
    batch_size = recon.shape[0] * recon.shape[1]
    hazard_obs = recon[:, :, 9:25].reshape(batch_size, -1, 2)
    hazards_dist = jnp.sqrt(jnp.sum(jnp.square(hazard_obs), axis=2)).reshape(
        batch_size,
        -1,
    )

    condition = jnp.less_equal(hazards_dist, hazards_size)
    cost = jnp.where(condition, 1.0, 0.0)
    cost = cost.sum(1)
    condition = jnp.greater_equal(cost, 1.0)
    cost = jnp.where(condition, 1.0, 0.0)

    cost = cost.reshape(recon.shape[0], recon.shape[1])
    print('error')
    return cost



  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    if 'cost' in data.keys():
      metrics['cost_max_data'] = jnp.abs(data['cost']).max()
    if 'cost' in dists.keys():
      metrics['cost_max_pred'] = jnp.abs(dists['cost'].mean()).max()
    if 'cost' in dists and 'cost' in data.keys() and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cost'], data['cost'], 0.1)
      metrics.update({f'cost_{k}': v for k, v in stats.items()})
    return metrics

class ImagSafeActorCritic(nj.Module):
  def __init__(self, critics, cost_critics, scales, cost_scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    cost_critics = {k: v for k, v in cost_critics.items() if scales[k]}

    for key, scale in scales.items():
      assert not scale or key in critics, key
    for key, cost_scale in cost_scales.items():
      assert not cost_scale or key in cost_critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.cost_critics = {k: v for k, v in cost_critics.items() if cost_scales[k]}
    #这个部分加入一个和cost类似的cr
    self.scales = scales
    self.cost_scales = cost_scales
    self.act_space = act_space
    self.config = config
    self.lagrange = jaxutils.Lagrange(self.config.lagrange_multiplier_init, self.config.penalty_multiplier_init, self.config.cost_limit, name=f'lagrange')
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(    
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.dual_actor = nets.MLP(    #这个部分多定义一个对偶策略
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.costnorms = {
        k: jaxutils.Moments(**config.costnorm, name=f'costnorm_{k}')
        for k in cost_critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)   #对这个部分的loss进行修改
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    for key, cost_critic in self.cost_critics.items():
      mets = cost_critic.train(traj, self.actor)
      metrics.update({f'{key}_cost_critic_{k}': v for k, v in mets.items()})
    return traj, metrics


  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    loss = loss.mean()
    # print('7777777777777777')
    if self.config.expl_behavior   not in ['CEMPlanner', 'CCEPlanner', 'PIDPlanner']:   # ???
      # print('8888888888888888888')
      cost_advs = []
      total = sum(self.cost_scales[k] for k in self.cost_critics)
      cost_rets = []
      for key, cost_critic in self.cost_critics.items():
        cost, cost_ret, base = cost_critic.score(traj, self.actor)
        cost_rets.append(cost_ret)
        offset, invscale = self.costnorms[key](cost_ret)
        normed_ret = (cost_ret - offset) / invscale
        normed_base = (base - offset) / invscale
        cost_advs.append((normed_ret - normed_base) * self.cost_scales[key] / total)
        metrics.update(jaxutils.tensorstats(cost, f'{key}_cost'))
        metrics.update(jaxutils.tensorstats(cost_ret, f'{key}_cost_raw'))
        metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_cost_normed'))
        metrics[f'{key}_cost_rate'] = (jnp.abs(ret) >= 0.5).mean()
      if self.config.pessimistic: 
        cost_ret_episode = jnp.stack(cost_rets).sum(0)
      else:
        cost_ret_episode = jnp.stack(cost_rets).mean(0)
      penalty, lagrange_multiplier, penalty_multiplier = self.lagrange(cost_ret_episode)
      metrics[f'lagrange_multiplier'] = lagrange_multiplier
      metrics[f'penalty_multiplier'] = penalty_multiplier
      metrics[f'penalty'] = penalty
      loss += penalty
    return loss, metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics

class ImagActorCritic(nj.Module):
  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}

    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}

    self.scales = scales
    self.act_space = act_space
    self.config = config

    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    # print('9999999999999999999999')
    def loss(start):
      policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics


  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
    ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics





class VFunction(nj.Module):  #貌似这个只估计state_value也不行必须估计action_value

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    assert len(rew) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]

class CostVFunction(nj.Module):


  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.cost_critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None, lag=1.0):
    rew = self.rewfn(traj)
    rew_repeat = rew # * self.config.env[self.config.task.split('_')[0]].repeat
    assert len(rew_repeat) == len(traj['action']) - 1, (
        'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [lag * value[-1]]
    interm = rew_repeat + lag * disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew_repeat, ret, value[:-1]
  


class DualImagSafeActorCritic(nj.Module):
  def __init__(self, act_space, config):
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete

    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)

    self.dual_actor = nets.MLP(
        name='dual_actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)

    self.use_reference_kl = bool(config.danger_posttrain.enabled)
    if self.use_reference_kl:
      # This network never receives gradients. Its parameters are copied from
      # `actor` exactly once at the bootstrap/post-training boundary.
      self.reference_actor = nets.MLP(
          name='reference_actor', dims='deter', shape=act_space.shape,
          **config.actor,
          dist=config.actor_dist_disc if disc else config.actor_dist_cont)
      self.reference_ready = nj.Variable(
          lambda: jnp.asarray(0.0, dtype=jnp.float32),
          name='reference_ready')
      self.reference_kl_lam = nj.Variable(
          lambda: jnp.asarray(
              config.danger_posttrain.reference_kl_lam_init,
              dtype=jnp.float32),
          name='reference_kl_lam')

    self.reward_q1 = RewardQFunction(config, name='reward_q1')
    self.reward_q2 = RewardQFunction(config, name='reward_q2')
    self.risk_q1 = RiskQFunction(config, name='risk_q1')
    self.risk_q2 = RiskQFunction(config, name='risk_q2')
    self.dual_risk_q1 = DualRiskQFunction(config, name='dual_risk_q1')
    self.dual_risk_q2 = DualRiskQFunction(config, name='dual_risk_q2')

    self.recover_q1 = RecoverQFunction(config, name='recover_q1')
    self.recover_q2 = RecoverQFunction(config, name='recover_q2')

    self.log_cg = nj.Variable(
        lambda: jnp.asarray(self.config.dual.init_log_cg, dtype=jnp.float32), name='log_cg')
    self.lam1 = nj.Variable(
        lambda: jnp.asarray(self.config.dual.init_lam1, dtype=jnp.float32), name='lam1')
    self.lam2 = nj.Variable(
        lambda: jnp.asarray(self.config.dual.init_lam2, dtype=jnp.float32), name='lam2')
    self.lam3 = nj.Variable(
        lambda: jnp.asarray(self.config.dual.init_lam3, dtype=jnp.float32), name='lam3')
    self.lam4 = nj.Variable(
        lambda: jnp.asarray(self.config.dual.init_lam4, dtype=jnp.float32), name='lam4')


    self.actor_opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)
    self.dual_actor_opt = jaxutils.Optimizer(name='dual_actor_opt', **config.dual_actor_opt)
    self.main_lam_opt = optax.adam(self.config.dual.lam_lr)
    self.dual_lam_opt = optax.adam(self.config.dual.lam_lr)
    if self.use_reference_kl:
      self.reference_kl_opt = optax.adam(
          self.config.danger_posttrain.reference_kl_lam_lr)

    self.rewnorm = jaxutils.Moments(**config.retnorm, name='dual_rewnorm')

  def freeze_reference_policy(self):
    """Copy the bootstrap main actor into the immutable KL reference actor."""
    if not self.use_reference_kl:
      return {'reference_policy_frozen': 0.0}
    actor_state = self.actor.getm()
    reference_state = {
        self.reference_actor.path + key[len(self.actor.path):]: sg(value)
        for key, value in actor_state.items()}
    self.reference_actor.putm(reference_state)
    # This method is intentionally called outside the regular nj.jit wrapper,
    # because it copies an already-initialised state tree. Derive the flag on
    # device instead of constructing a host scalar, which is disallowed by
    # the configured JAX host-to-device transfer guard.
    reference_ready = self.reference_ready.read()
    reference_frozen = (reference_ready == reference_ready).astype(reference_ready.dtype)
    self.reference_ready.write(reference_frozen)
    return {
        'reference_policy_frozen': reference_frozen,
        # This metric is host metadata, not an input to JAX computation.
        'reference_policy_parameter_tensors': float(len(reference_state)),
    }


  def update_cg(self, old_g, new_g, cost):

    def masked_mean(x, mask):
      mask = mask.astype(jnp.float32)
      return (x * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    pf = jnp.asarray(self.config.dual.pf, dtype=jnp.float32)

    # cg = exp(log_cg)，并限制范围
    min_cg = jnp.asarray(getattr(self.config.dual, 'min_cg', 0.005), dtype=jnp.float32)
    max_cg = jnp.asarray(getattr(self.config.dual, 'max_cg', 0.05), dtype=jnp.float32)
    min_log_cg = jnp.log(min_cg)
    max_log_cg = jnp.log(max_cg)
    cg = jnp.clip(jnp.exp(self.log_cg.read()), min_cg, max_cg)

    # violation 的判定，注意不要直接 cost > 0
    cost_prob = cost
    vio = cost_prob >= jnp.asarray(self.config.dual.cost_threshold, dtype=jnp.float32)

    # 区域划分基于 old_g
    fea = (old_g < pf) & (~vio)
    cri = fea & (old_g >= pf - cg)

    fea_ratio = fea.astype(jnp.float32).mean()

    term1 = masked_mean(
        jax.nn.leaky_relu((pf - new_g) - cg),
        vio,
    )

    fea_vio = cri & (new_g > pf)
    vio_ratio = fea_vio.astype(jnp.float32).mean()

    # 有可行样本，且没有边界破坏，就允许收缩 cg
    shrink_gate = ((fea_ratio > 0).astype(jnp.float32) *
                  (vio_ratio == 0).astype(jnp.float32))

    term2 = shrink_gate * jax.nn.leaky_relu(-cg)

    delta_cg = term1 + term2

    # 更新 log_cg
    new_log_cg = self.log_cg.read() + 0.98 * delta_cg
    new_log_cg = jnp.clip(new_log_cg, min_log_cg, max_log_cg)

    self.log_cg.write(new_log_cg)

    metrics = {
        'cg': cg,
        'delta_cg': delta_cg,
        'cg_term1': term1,
        'cg_term2': term2,
        'fea_ratio': fea_ratio,
        'cri_ratio': cri.astype(jnp.float32).mean(),
        'fea_vio_ratio': vio_ratio,
        'vio_ratio': vio.astype(jnp.float32).mean(),
        'new_cg': jnp.exp(new_log_cg),
        'min_cg': min_cg,
        'max_cg': max_cg,
    }
    return metrics

  def update_dual_lams(self, kl_dual_to_main, kl_main_to_dual):
    target_kl = jnp.asarray(self.config.dual.target_kl, dtype=jnp.float32)
    lam_lr = jnp.asarray(self.config.dual.lam_lr, dtype=jnp.float32)

    mean_kl_dual_to_main = kl_dual_to_main.mean()
    mean_kl_main_to_dual = kl_main_to_dual.mean()
    lam3 = jnp.maximum(self.lam3.read(), 0.0)
    lam4 = jnp.maximum(self.lam4.read(), 0.0)
    lam3_opt_state = self.get('lam3_opt_state', self.dual_lam_opt.init, lam3)
    lam4_opt_state = self.get('lam4_opt_state', self.dual_lam_opt.init, lam4)

    lam3_grads = target_kl - mean_kl_dual_to_main
    lam4_grads = target_kl - mean_kl_main_to_dual

    lam3_updates, lam3_opt_state = self.dual_lam_opt.update(lam3_grads, lam3_opt_state)
    lam4_updates, lam4_opt_state = self.dual_lam_opt.update(lam4_grads, lam4_opt_state)
    self.put('lam3_opt_state', lam3_opt_state)
    self.put('lam4_opt_state', lam4_opt_state)

    new_lam3 = jnp.maximum(optax.apply_updates(lam3, lam3_updates), 0.0)
    new_lam4 = jnp.maximum(optax.apply_updates(lam4, lam4_updates), 0.0)

    self.lam3.write(new_lam3)
    self.lam4.write(new_lam4)

    return {
        'target_kl': target_kl,
        'lam_lr': lam_lr,
        'new_lam3': new_lam3,
        'new_lam4': new_lam4,
    }

  def update_reference_kl_lam(self, mean_kl):
    """Dual-ascent update for KL(current policy || frozen bootstrap policy)."""
    if not self.use_reference_kl:
      return {}
    ready = self.reference_ready.read()
    target = jnp.asarray(
        self.config.danger_posttrain.reference_kl_target, dtype=jnp.float32)
    max_lam = jnp.asarray(
        self.config.danger_posttrain.reference_kl_lam_max, dtype=jnp.float32)
    lam = jnp.clip(self.reference_kl_lam.read(), 0.0, max_lam)
    opt_state = self.get('reference_kl_opt_state', self.reference_kl_opt.init, lam)
    # If KL exceeds its target, this negative gradient makes Optax increase
    # the multiplier, strengthening the constraint in subsequent updates.
    grads = ready * (target - mean_kl)
    updates, opt_state = self.reference_kl_opt.update(grads, opt_state)
    self.put('reference_kl_opt_state', opt_state)
    new_lam = jnp.clip(optax.apply_updates(lam, updates), 0.0, max_lam)
    self.reference_kl_lam.write(new_lam)
    return {
        'reference_kl_target': target,
        'reference_kl_lam': new_lam,
        'reference_kl_ready': ready,
    }

  def update_main_lams(self, fea_g_vio, inf_g_inc):
    lam1 = jnp.maximum(self.lam1.read(), 0.0)
    lam2 = jnp.maximum(self.lam2.read(), 0.0)
    lam1_opt_state = self.get('lam1_opt_state', self.main_lam_opt.init, lam1)
    lam2_opt_state = self.get('lam2_opt_state', self.main_lam_opt.init, lam2)

    lam1_grads = -fea_g_vio
    lam2_grads = -inf_g_inc

    lam1_updates, lam1_opt_state = self.main_lam_opt.update(lam1_grads, lam1_opt_state)
    lam2_updates, lam2_opt_state = self.main_lam_opt.update(lam2_grads, lam2_opt_state)
    self.put('lam1_opt_state', lam1_opt_state)
    self.put('lam2_opt_state', lam2_opt_state)

    new_lam1 = jnp.maximum(optax.apply_updates(lam1, lam1_updates), 0.0)
    new_lam2 = jnp.maximum(optax.apply_updates(lam2, lam2_updates), 0.0)

    self.lam1.write(new_lam1)
    self.lam2.write(new_lam2)

    return {
        'lam_lr': jnp.asarray(self.config.dual.lam_lr, dtype=jnp.float32),
        'fea_g_vio': fea_g_vio,
        'inf_g_inc': inf_g_inc,
        'new_lam1': new_lam1,
        'new_lam2': new_lam2,
      }

  def _policy_base_normal_log_prob(self, policy, action):
    base = getattr(policy, 'distribution', policy)
    params = getattr(base, 'parameters', {})
    loc = getattr(base, 'loc', None)
    scale = getattr(base, 'scale', None)
    if loc is None and isinstance(params, dict):
      loc = params.get('loc', None)
    if scale is None and isinstance(params, dict):
      scale = params.get('scale', None)
    if loc is None or scale is None:
      return policy.log_prob(action)
    loc = jnp.where(jnp.isfinite(loc), loc, 0.0).astype(jnp.float32)
    scale = jnp.where(jnp.isfinite(scale), scale, 1.0).astype(jnp.float32)
    scale = jnp.maximum(scale, jnp.asarray(1e-5, dtype=jnp.float32))
    action = action.astype(jnp.float32)
    log_prob = (
        -0.5 * jnp.square((action - loc) / scale)
        - jnp.log(scale)
        - 0.5 * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32)))
    dims = tuple(range(-len(self.act_space.shape), 0))
    return log_prob.sum(dims) if dims else log_prob

  def _policy_base_normal_params(self, policy):
    base = getattr(policy, 'distribution', policy)
    params = getattr(base, 'parameters', {})
    loc = getattr(base, 'loc', None)
    scale = getattr(base, 'scale', None)
    if loc is None and isinstance(params, dict):
      loc = params.get('loc', None)
    if scale is None and isinstance(params, dict):
      scale = params.get('scale', None)
    if loc is None or scale is None:
      return None, None
    loc = jnp.where(jnp.isfinite(loc), loc, 0.0).astype(jnp.float32)
    scale = jnp.where(jnp.isfinite(scale), scale, 1.0).astype(jnp.float32)
    return loc, jnp.maximum(scale, jnp.asarray(1e-5, dtype=jnp.float32))

  def _policy_kl_to_reference(self, policy, reference_policy):
    """Analytic base-normal KL, with a Monte-Carlo fallback for other dists."""
    loc, scale = self._policy_base_normal_params(policy)
    ref_loc, ref_scale = self._policy_base_normal_params(reference_policy)
    if loc is not None and ref_loc is not None:
      ref_loc, ref_scale = sg(ref_loc), sg(ref_scale)
      per_dim = (
          jnp.log(ref_scale / scale) +
          (jnp.square(scale) + jnp.square(loc - ref_loc)) /
          (2.0 * jnp.square(ref_scale)) - 0.5)
      dims = tuple(range(-len(self.act_space.shape), 0))
      return per_dim.sum(dims) if dims else per_dim
    action = policy.sample(seed=nj.rng())
    return jnp.maximum(
        policy.log_prob(action) - sg(reference_policy.log_prob(action)), 0.0)

  def _traj_policy_logps(self, traj):
    feat = {k: traj[k] for k in ('deter', 'stoch') if k in traj}
    action = traj['action']
    main_policy = self.actor(sg(feat))
    dual_policy = self.dual_actor(sg(feat))
    return (
        self._policy_base_normal_log_prob(main_policy, action),
        self._policy_base_normal_log_prob(dual_policy, action))

  def _traj_is_ratio(self, traj, numerator='main', cumulative=True):
    # Imagined trajectory-level IS approximation inside the world model.
    main_logp, dual_logp = self._traj_policy_logps(traj)
    if numerator == 'main':
      step_log_ratio = main_logp - dual_logp
    elif numerator == 'dual':
      step_log_ratio = dual_logp - main_logp
    else:
      raise ValueError(numerator)
    min_log = jnp.log(jnp.asarray(self.config.dual.is_min_weight, dtype=jnp.float32))
    max_log = jnp.log(jnp.asarray(self.config.dual.is_max_weight, dtype=jnp.float32))
    if cumulative:
      step_log_ratio = jnp.cumsum(step_log_ratio, axis=0)
    traj_log_ratio = jnp.clip(step_log_ratio, min_log, max_log)
    return jnp.exp(traj_log_ratio)

  def _apply_critic_is_weight(self, traj, critic_is_weight):
    updated = dict(traj)
    updated['critic_is_weight'] = critic_is_weight
    return updated

  def _apply_actor_is_weight(self, traj, is_weight, actor_mask=None):
    updated = dict(traj)
    if actor_mask is None:
      actor_mask = jnp.ones_like(traj['weight'])
    updated['is_weight'] = is_weight
    updated['actor_mask'] = actor_mask
    return updated

  def _weighted_mean(self, value, weight):
    weight = weight.astype(jnp.float32)
    total = jnp.maximum(weight.sum(), 1e-8)
    return (value * weight).sum() / total

  def _concat_trajs(self, left, right):
    keys = set(left.keys()) & set(right.keys())
    merged = {}
    for key in keys:
      merged[key] = jnp.concatenate([left[key], right[key]], axis=1)
    return merged


  def train_qs(self, main_traj, dual_traj, dual_enabled=1.0):
    reward = self._reward_target(main_traj)
    risk = self._risk_target(main_traj)
    dual_risk = self._dual_risk_target(dual_traj)
    recover = self._recover_target(main_traj)
    dual_enabled = jnp.asarray(dual_enabled, dtype=jnp.float32)
    cross_enabled = dual_enabled > 0.5
    dual_reward = self._reward_target(dual_traj)
    dual_risk_for_primal = self._risk_target(dual_traj)
    main_dual_risk = self._dual_risk_target(main_traj)
    r_pd = self._traj_is_ratio(dual_traj, numerator='main', cumulative=False)
    r_dp = self._traj_is_ratio(main_traj, numerator='dual', cumulative=False)
    cross_gate = (dual_enabled > 0.5).astype(jnp.float32)
    r_pd = r_pd * cross_gate
    r_dp = r_dp * cross_gate
    mets = {}
    primal_main_traj = self._apply_critic_is_weight(
        main_traj, jnp.ones_like(main_traj['weight']))
    primal_dual_traj = self._apply_critic_is_weight(dual_traj, r_pd)
    dual_dual_traj = self._apply_critic_is_weight(
        dual_traj, cross_gate * jnp.ones_like(dual_traj['weight']))
    dual_main_traj = self._apply_critic_is_weight(
        main_traj, (1.0 - cross_gate) * jnp.ones_like(main_traj['weight']) + r_dp)
    primal_critic_traj = self._concat_trajs(primal_main_traj, primal_dual_traj)
    dual_critic_traj = self._concat_trajs(dual_dual_traj, dual_main_traj)
    reward_target = jnp.concatenate([reward, dual_reward], axis=1)
    risk_target = jnp.concatenate([risk, dual_risk_for_primal], axis=1)
    dual_risk_target = jnp.concatenate([dual_risk, main_dual_risk], axis=1)
    mets.update({f'primal_from_dual_{k}': v for k, v in finite_stats('ratio', r_pd).items()})
    mets.update({f'dual_from_main_{k}': v for k, v in finite_stats('ratio', r_dp).items()})
    for key in ('action', 'reward', 'cont', 'weight', 'cost'):
      if key in main_traj:
        mets.update({f'main_imag_{k}': v for k, v in finite_stats(key, main_traj[key]).items()})
      if key in dual_traj:
        mets.update({f'dual_imag_{k}': v for k, v in finite_stats(key, dual_traj[key]).items()})
    if 'deter' in main_traj:
      mets.update({f'main_imag_{k}': v for k, v in finite_stats('deter', main_traj['deter']).items()})
    if 'stoch' in main_traj:
      mets.update({f'main_imag_{k}': v for k, v in finite_stats('stoch', main_traj['stoch']).items()})
    if 'deter' in dual_traj:
      mets.update({f'dual_imag_{k}': v for k, v in finite_stats('deter', dual_traj['deter']).items()})
    if 'stoch' in dual_traj:
      mets.update({f'dual_imag_{k}': v for k, v in finite_stats('stoch', dual_traj['stoch']).items()})
    mets.update({f'reward_target_{k}': v for k, v in finite_stats('target', reward).items()})
    mets.update({f'risk_target_{k}': v for k, v in finite_stats('target', risk).items()})
    mets.update({f'dual_risk_target_{k}': v for k, v in finite_stats('target', dual_risk).items()})
    mets.update({f'recover_target_{k}': v for k, v in finite_stats('target', recover).items()})
    m1 = self.reward_q1.train(primal_critic_traj, reward_target)
    mets.update({f'reward_q1_{k}': v for k, v in m1.items()})

    m2 = self.reward_q2.train(primal_critic_traj, reward_target)
    mets.update({f'reward_q2_{k}': v for k, v in m2.items()})

    g1 = self.risk_q1.train(primal_critic_traj, risk_target)
    mets.update({f'risk_q1_{k}': v for k, v in g1.items()})

    g2 = self.risk_q2.train(primal_critic_traj, risk_target)
    mets.update({f'risk_q2_{k}': v for k, v in g2.items()})

    dg1 = self.dual_risk_q1.train(dual_critic_traj, dual_risk_target)
    mets.update({f'dual_risk_q1_{k}': v for k, v in dg1.items()})

    dg2 = self.dual_risk_q2.train(dual_critic_traj, dual_risk_target)
    mets.update({f'dual_risk_q2_{k}': v for k, v in dg2.items()})

    r1 = self.recover_q1.train(main_traj, recover)
    mets.update({f'recover_q1_{k}': v for k, v in r1.items()})

    r2 = self.recover_q2.train(main_traj, recover)
    mets.update({f'recover_q2_{k}': v for k, v in r2.items()})
    
      #统一看的 q loss
    mets['reward_q_loss'] = 0.5 * (
        m1['critic_opt_loss'] + m2['critic_opt_loss']
    )
    mets['risk_q_loss'] = 0.5 * (
        g1['critic_opt_loss'] + g2['critic_opt_loss']
    )
    mets['dual_risk_q_loss'] = 0.5 * (
        dg1['critic_opt_loss'] + dg2['critic_opt_loss']
    )
    mets['q_loss'] = 0.25 * (
        m1['critic_opt_loss'] + m2['critic_opt_loss'] +
        g1['critic_opt_loss'] + g2['critic_opt_loss']
    )
    mets['recover_q_loss'] = 0.5 * (
    r1['critic_opt_loss'] + r2['critic_opt_loss']
    )
    return mets

  def _lambda_target(self, signal, disc, value):
    vals = [value[-1]]
    interm = signal + disc * value[1:] * (1.0 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    return sg(jnp.stack(list(reversed(vals))[:-1]))

  def _prob_lambda_target(self, signal, disc, value):
    return jnp.clip(self._lambda_target(signal, disc, value), 0.0, 1.0)

  def _reward_target(self, traj):
      q_value = jnp.minimum(
          self.reward_q1.target(traj),
          self.reward_q2.target(traj),
      )
      reward = traj['reward'][1:]
      disc = traj['cont'][1:] * 0.98
      return self._lambda_target(reward, disc, q_value)

  def _risk_target(self, traj):
    risk_value = jnp.maximum(
        self.risk_q1.target(traj),
        self.risk_q2.target(traj),
    )
    risk_value = jnp.clip(risk_value, 0.0, 1.0)
    cost = jnp.clip(traj['cost'][1:], 0.0, 1.0)
    disc = traj['cont'][1:] * (1.0 - cost) * self.config.dual.cost_gamma
    return self._prob_lambda_target(cost, disc, risk_value)

  def _dual_risk_target(self, traj):
    dual_risk_value = jnp.minimum(
        self.dual_risk_q1.target(traj),
        self.dual_risk_q2.target(traj),
    )
    dual_risk_value = jnp.clip(dual_risk_value, 0.0, 1.0)
    cost = jnp.clip(traj['cost'][1:], 0.0, 1.0)
    disc = traj['cont'][1:] * (1.0 - cost) * self.config.dual.cost_gamma
    return self._prob_lambda_target(cost, disc, dual_risk_value)
  
  def _recover_target(self, traj):
      recover_value = jnp.minimum(
          self.recover_q1.target(traj),
          self.recover_q2.target(traj),
      )
      recover_value = jnp.clip(recover_value, 0.0, 1.0)
      cost = jnp.clip(traj['cost'][1:], 0.0, 1.0)
      disc = traj['cont'][1:] * cost * 0.98
      return self._prob_lambda_target(1.0 - cost, disc, recover_value)


  def train(self, imagine, start, context, dual_enabled=None):
    dual_enabled = self.config.use_dual_policy if dual_enabled is None else dual_enabled

    def main_policy(s):
      return self.actor(sg(s)).sample(seed=nj.rng())

    def dual_policy(s):
      return self.dual_actor(sg(s)).sample(seed=nj.rng())

    main_traj = imagine(main_policy, start, self.config.imag_horizon)
    dual_traj = imagine(dual_policy, start, self.config.imag_horizon)
    dual_enabled_flag = jnp.asarray(dual_enabled, dtype=jnp.float32) #对偶策略是否开始环境交互
    cross_enabled = dual_enabled_flag > 0.5   #如果开始环境交互就允许对偶想象轨迹和主策略想象轨迹进行混合更新

    metrics = {}
    q_mets = self.train_qs(main_traj, dual_traj, dual_enabled=dual_enabled_flag)
    metrics.update(q_mets)

    r_pd = self._traj_is_ratio(dual_traj, numerator='main')
    r_dp = self._traj_is_ratio(main_traj, numerator='dual')
    cross_gate = (dual_enabled_flag > 0.5).astype(jnp.float32)
    r_pd = r_pd * cross_gate
    r_dp = r_dp * cross_gate
    main_actor_traj = self._concat_trajs(
        self._apply_actor_is_weight(
            main_traj, jnp.ones_like(main_traj['weight'])),
        self._apply_actor_is_weight(
            dual_traj, r_pd, cross_gate * jnp.ones_like(dual_traj['weight'])))
    dual_self_weight = cross_gate * jnp.ones_like(dual_traj['weight'])
    dual_main_weight = (1.0 - cross_gate) * jnp.ones_like(main_traj['weight']) + r_dp
    dual_actor_traj = self._concat_trajs(
        self._apply_actor_is_weight(
            dual_traj, jnp.ones_like(dual_traj['weight']), dual_self_weight),
        self._apply_actor_is_weight(main_traj, dual_main_weight))
    main_actor_cross_mean = r_pd.mean()
    dual_actor_cross_mean = r_dp.mean()
    metrics.update({
        'main_actor_is_weight_mean': main_actor_traj['is_weight'].mean(),
        'main_actor_world_weight_mean': main_actor_traj['weight'].mean(),
        'main_actor_mask_mean': main_actor_traj['actor_mask'].mean(),
        'dual_actor_is_weight_mean': dual_actor_traj['is_weight'].mean(),
        'dual_actor_world_weight_mean': dual_actor_traj['weight'].mean(),
        'dual_actor_mask_mean': dual_actor_traj['actor_mask'].mean(),
        'main_actor_is_weight_nonzero': (main_actor_traj['is_weight'] > 0).astype(jnp.float32).mean(),
        'main_actor_mask_nonzero': (main_actor_traj['actor_mask'] > 0).astype(jnp.float32).mean(),
        'dual_actor_is_weight_nonzero': (dual_actor_traj['is_weight'] > 0).astype(jnp.float32).mean(),
        'dual_actor_mask_nonzero': (dual_actor_traj['actor_mask'] > 0).astype(jnp.float32).mean(),
        'main_actor_cross_is_mean': main_actor_cross_mean,
        'dual_actor_cross_is_mean': dual_actor_cross_mean,
        'imagined_cross_is_enabled': cross_enabled.astype(jnp.float32),
        'dual_actor_uses_main_traj_only': (1.0 - dual_enabled_flag),
        'dual_actor_uses_dual_traj_cross_is': dual_enabled_flag,
    })
    metrics['dual_train/dual/IS_weight'] = main_actor_cross_mean
    metrics['dual_train/dual/IS_weight_dual'] = dual_actor_cross_mean
    metrics['dual_train/dual/IS_weight_main_mean'] = main_actor_traj['is_weight'].mean()
    metrics['dual_train/dual/IS_weight_dual_mean'] = dual_actor_traj['is_weight'].mean()

    actor_mets = self.train_main_actor(main_actor_traj)
    metrics.update(actor_mets)

    metrics['dual_train_enabled'] = dual_enabled_flag
    metrics['dual_actor_skipped'] = 1.0 - dual_enabled_flag
    dual_actor_mets = self.train_dual_actor_on_traj(
        dual_actor_traj, dual_enabled=dual_enabled_flag)
    metrics.update(dual_actor_mets)

    return main_traj, metrics

  def train_main_actor(self, traj):

      def loss_fn(traj):
        traj_in = {k: v[:-1] for k, v in traj.items()}
        policy = self.actor(sg(traj_in))
        action = policy.sample(seed=nj.rng())

        qa_traj = dict(traj_in)
        qa_traj['action'] = action

        q = jnp.minimum(
            self.reward_q1.predict(qa_traj),
            self.reward_q2.predict(qa_traj),
        )
        new_g = jnp.maximum(
            self.risk_q1.predict(qa_traj),
            self.risk_q2.predict(qa_traj),
        )
        old_g = jnp.maximum(
            self.risk_q1.predict(traj_in),
            self.risk_q2.predict(traj_in),
        )

        gr = jnp.minimum(
            self.recover_q1.predict(qa_traj),
            self.recover_q2.predict(qa_traj),
        )
        rew_offset, rew_invscale = self.rewnorm(q)
        norm_q = (q - rew_offset) / rew_invscale
        pf = jnp.asarray(self.config.dual.pf, dtype=jnp.float32)
        cg = jnp.clip(
            jnp.exp(self.log_cg.read()),
            jnp.asarray(self.config.dual.min_cg, dtype=jnp.float32),
            jnp.asarray(self.config.dual.max_cg, dtype=jnp.float32),
        )   #cg相当于是在约束边界0.1的右边再预留一些区域作为缓冲带
        cost = traj_in['cost']
        cost_prob = cost
        ent = policy.entropy()
        if self.use_reference_kl:
          reference_policy = self.reference_actor(sg(traj_in))
          reference_kl = self._policy_kl_to_reference(policy, reference_policy)
          reference_ready = self.reference_ready.read()
          reference_lam = jnp.clip(
              self.reference_kl_lam.read(), 0.0,
              jnp.asarray(
                  self.config.danger_posttrain.reference_kl_lam_max,
                  dtype=jnp.float32))
        else:
          reference_kl = jnp.zeros_like(q)
          reference_ready = jnp.asarray(0.0, dtype=jnp.float32)
          reference_lam = jnp.asarray(0.0, dtype=jnp.float32)

        vio = cost_prob >= jnp.asarray(self.config.dual.cost_threshold, dtype=jnp.float32)
        fea = (old_g < pf) & (~vio)   #即没有违反约束并且估计的风险在边界之内。
        cri = fea & (old_g >= (pf - cg))
        fea_inner = fea & (~cri)
        inf = (~fea) & (~vio)
        weight = traj_in.get('weight', jnp.ones_like(q))
        is_weight = traj_in.get('is_weight', jnp.ones_like(q))
        actor_mask = traj_in.get('actor_mask', jnp.ones_like(q))
        actor_weight = weight * actor_mask
        norm_weight = actor_weight / jnp.maximum(actor_weight.mean(), 1e-8)
 
        lam1 = jnp.maximum(self.lam1.read(), 0.0)
        lam2 = jnp.maximum(self.lam2.read(), 0.0)


        loss_vio = vio.astype(jnp.float32) * (-gr)
        loss_fea = fea_inner.astype(jnp.float32) * (-norm_q)
        loss_cri = cri.astype(jnp.float32) * ((-norm_q + lam1 * new_g) / (lam1 + 1.0))
        loss_inf = inf.astype(jnp.float32) * ((-norm_q + lam2 * new_g) / (lam2 + 1.0))
        loss = loss_fea + loss_cri + loss_inf + loss_vio
        loss = loss - self.config.actent * ent
        reference_kl_loss = reference_ready * reference_lam * reference_kl  #这个地方就是加入了 KL 惩罚项，只有当 reference_ready=1 时才会生效，reference_lam 是这个惩罚项的权重，reference_kl 是当前策略和 reference 策略之间的 KL 散度。
        loss = loss + reference_kl_loss
        loss = loss * sg(is_weight)
        loss = self._weighted_mean(loss, sg(norm_weight))

        metric_weight = jnp.maximum(actor_weight, 0.0)

        metrics = {
            'main_q': self._weighted_mean(q, metric_weight),
            'main_old_g': self._weighted_mean(old_g, metric_weight),
            'main_new_g': self._weighted_mean(new_g, metric_weight),
            'main_cg': cg,
            'fea_ratio': self._weighted_mean(fea.astype(jnp.float32), metric_weight),
            'fea_inner_ratio': self._weighted_mean(fea_inner.astype(jnp.float32), metric_weight),
            'cri_ratio': self._weighted_mean(cri.astype(jnp.float32), metric_weight),
            'inf_ratio': self._weighted_mean(inf.astype(jnp.float32), metric_weight),
            'vio_ratio': self._weighted_mean(vio.astype(jnp.float32), metric_weight),
            'pf_minus_cg': pf - cg,
            'reference_kl': self._weighted_mean(reference_kl, metric_weight),
            'reference_kl_loss': self._weighted_mean(reference_kl_loss, metric_weight),
            'reference_kl_lam': reference_lam,
            'reference_kl_ready': reference_ready,
        }
        metrics.update(finite_stats('main_action', action))
        metrics.update(finite_stats('main_q_tensor', q))
        metrics.update(finite_stats('main_old_g_tensor', old_g))
        metrics.update(finite_stats('main_new_g_tensor', new_g))
        metrics.update(finite_stats('main_entropy_tensor', ent))
        metrics.update(finite_stats('main_weight_tensor', weight))
        metrics.update(finite_stats('main_is_weight_tensor', is_weight))
        metrics.update(finite_stats('main_actor_mask_tensor', actor_mask))
        metrics.update(finite_stats('main_loss_tensor', loss))

        aux = {
            'metrics': metrics,
            'old_g': old_g,
            'new_g': new_g,
            'cost': cost,
            'fea_mask': fea,
            'cri_mask': cri,
        }
        return loss, aux

      mets, aux = self.actor_opt(self.actor, loss_fn, traj, has_aux=True)

      metrics = aux['metrics']
      metrics.update(mets)

      reference_mets = self.update_reference_kl_lam(metrics['reference_kl'])
      metrics.update(reference_mets)

      cg_mets = self.update_cg(
          aux['old_g'],
          aux['new_g'],
          aux['cost'],
      )
      metrics.update(cg_mets)
      fea_g_vio = self._weighted_mean(
          jax.nn.leaky_relu(aux['new_g'] - jnp.asarray(self.config.dual.pf, dtype=jnp.float32)),
          aux['cri_mask'].astype(jnp.float32))
      inf_g_inc = self._weighted_mean(
          jax.nn.leaky_relu(aux['new_g'] - aux['old_g']),
          (~aux['fea_mask']).astype(jnp.float32))
      lam_mets = self.update_main_lams(fea_g_vio, inf_g_inc)
      metrics.update({f'main_{k}': v for k, v in lam_mets.items()})
      metrics['dual_train/dual/lam1'] = jnp.maximum(self.lam1.read(), 0.0)
      metrics['dual_train/dual/lam2'] = jnp.maximum(self.lam2.read(), 0.0)

      return {f'main_actor_{k}': v for k, v in metrics.items()}

  def initial(self, batch_size):   
      return {}

  def policy(
      self, latent, state,
      dual_enabled=None, dual_ratio=None,
      dual_blend=None,
      dual_mask=None):

      dual_enabled = self.config.use_dual_policy if dual_enabled is None else dual_enabled

      dual_ratio = jnp.asarray(
          getattr(self.config.dual, 'dual_mix_ratio', 0.5)
          if dual_ratio is None else dual_ratio,
          dtype=jnp.float32)

      dual_blend = jnp.asarray(
          1.0 if dual_blend is None else dual_blend,
          dtype=jnp.float32)

      dual_ratio = jnp.clip(dual_ratio, 0.0, 1.0)
      dual_blend = jnp.clip(dual_blend, 0.0, 1.0)

      main_policy = self.actor(latent)
      dual_policy = self.dual_actor(latent)

      main_action = main_policy.sample(seed=nj.rng())
      dual_action = dual_policy.sample(seed=nj.rng())

      main_logp_main = main_policy.log_prob(main_action)
      dual_logp_dual = dual_policy.log_prob(dual_action)
      main_logp_dual = main_policy.log_prob(dual_action)
      dual_logp_main = dual_policy.log_prob(main_action)
      base_main_logp_main = self._policy_base_normal_log_prob(
          main_policy, main_action)
      base_dual_logp_dual = self._policy_base_normal_log_prob(
          dual_policy, dual_action)
      base_main_logp_dual = self._policy_base_normal_log_prob(
          main_policy, dual_action)
      base_dual_logp_main = self._policy_base_normal_log_prob(
          dual_policy, main_action)

      batch_size = main_action.shape[0]

      global_enable = jnp.asarray(dual_enabled, dtype=jnp.bool_)
      if global_enable.ndim > 0:
          global_enable = global_enable.astype(jnp.bool_)
      else:
          global_enable = jnp.full((batch_size,), global_enable, dtype=jnp.bool_)

      if dual_mask is None:
          dual_mask = (
              (jnp.arange(batch_size, dtype=jnp.float32) /
               jnp.maximum(batch_size, 1)) < dual_ratio)
      else:
          dual_mask = jnp.asarray(dual_mask, dtype=jnp.float32) > 0.5

      dual_mask = dual_mask & global_enable

      action = jnp.where(dual_mask[..., None], dual_action, main_action)

      behavior_logp = jnp.where(dual_mask, dual_logp_dual, main_logp_main)
      main_logp = jnp.where(dual_mask, main_logp_dual, main_logp_main)
      dual_logp = jnp.where(dual_mask, dual_logp_dual, dual_logp_main)

      log_weight = jnp.where(
          dual_mask,
          jnp.zeros_like(main_logp_main),
          jnp.clip(base_dual_logp_main - base_main_logp_main, -10.0, 10.0))

      log_weight_dual = jnp.where(
          dual_mask,
          jnp.clip(base_main_logp_dual - base_dual_logp_dual, -10.0, 10.0),
          jnp.zeros_like(main_logp_main))

      return {
          'action': action,
          'log_behavior_logp': behavior_logp,
          'log_main_logp': main_logp,
          'log_dual_logp': dual_logp,
          'log_main_risk': jnp.zeros_like(main_logp_main),
          'log_risk_threshold': jnp.zeros_like(main_logp_main),
          'log_use_dual': dual_mask.astype(jnp.float32),
          'log_dual_ratio': dual_ratio * jnp.ones_like(main_logp_main),
          'log_dual_blend': dual_blend * jnp.ones_like(main_logp_main),
          'log_weight': log_weight,
          'log_weight_dual': log_weight_dual,
          'log_behavior_policy_id': dual_mask.astype(jnp.float32),
      }, state


  def train_dual_actor_on_traj(self, dual_traj, dual_enabled=1.0):

      def loss_fn(dual_traj):
        traj_in = {k: v[:-1] for k, v in dual_traj.items()}
        policy = self.dual_actor(sg(traj_in))
        main_policy = self.actor(sg(traj_in))

        dual_action = policy.sample(seed=nj.rng())
        main_action = main_policy.sample(seed=nj.rng())

        dual_qa = dict(traj_in)
        dual_qa['action'] = dual_action

        dual_g = jnp.minimum(
            self.dual_risk_q1.predict(dual_qa),
            self.dual_risk_q2.predict(dual_qa),
        )

        dual_logp_dual = jnp.clip(
            self._policy_base_normal_log_prob(policy, dual_action), -20.0, 20.0)
        main_logp_dual = jnp.clip(
            self._policy_base_normal_log_prob(main_policy, dual_action), -20.0, 20.0)
        main_logp_main = jnp.clip(
            self._policy_base_normal_log_prob(main_policy, main_action), -20.0, 20.0)
        dual_logp_main = jnp.clip(
            self._policy_base_normal_log_prob(policy, main_action), -20.0, 20.0)

        kl_dual_to_main = jnp.clip(dual_logp_dual - main_logp_dual, 0.0, 10.0)
        kl_main_to_dual = jnp.clip(main_logp_main - dual_logp_main, 0.0, 10.0)

        lam3 = jnp.maximum(self.lam3.read(), 0.0)
        lam4 = jnp.maximum(self.lam4.read(), 0.0)
        if self.use_reference_kl:
          reference_policy = self.reference_actor(sg(traj_in))
          reference_kl = self._policy_kl_to_reference(policy, reference_policy)
          reference_ready = self.reference_ready.read()
          reference_lam = jnp.clip(
              self.reference_kl_lam.read(), 0.0,
              jnp.asarray(
                  self.config.danger_posttrain.reference_kl_lam_max,
                  dtype=jnp.float32))
        else:
          reference_kl = jnp.zeros_like(dual_g)
          reference_ready = jnp.asarray(0.0, dtype=jnp.float32)
          reference_lam = jnp.asarray(0.0, dtype=jnp.float32)

        ent = policy.entropy()
        weight = traj_in.get('weight', jnp.ones_like(dual_g))
        is_weight = traj_in.get('is_weight', jnp.ones_like(dual_g))
        actor_mask = traj_in.get('actor_mask', jnp.ones_like(dual_g))
        actor_weight = weight * actor_mask
        norm_weight = actor_weight / jnp.maximum(actor_weight.mean(), 1e-8)

        q_loss = -dual_g * sg(is_weight)
        kl_loss = lam3 * kl_dual_to_main + lam4 * kl_main_to_dual
        reference_kl_loss = reference_ready * reference_lam * reference_kl
        ent_loss = -self.config.actent * ent
        pre_loss = q_loss + kl_loss + reference_kl_loss
        loss = pre_loss + ent_loss
        loss = self._weighted_mean(loss, sg(norm_weight))

        metric_weight = jnp.maximum(actor_weight, 0.0)

        metrics = {
            'dual_g': self._weighted_mean(dual_g, metric_weight),
            'dual_entropy': self._weighted_mean(ent, metric_weight),
            'dual_kl_dual_to_main': self._weighted_mean(kl_dual_to_main, metric_weight),
            'dual_kl_main_to_dual': self._weighted_mean(kl_main_to_dual, metric_weight),
            'dual_q_loss': self._weighted_mean(q_loss, metric_weight),
            'dual_kl_loss': self._weighted_mean(kl_loss, metric_weight),
            'reference_kl': self._weighted_mean(reference_kl, metric_weight),
            'reference_kl_loss': self._weighted_mean(reference_kl_loss, metric_weight),
            'reference_kl_lam': reference_lam,
            'reference_kl_ready': reference_ready,
            'dual_lam3': lam3,
            'dual_lam4': lam4,
        }
        metrics.update(finite_stats('dual_action', dual_action))
        metrics.update(finite_stats('dual_main_action', main_action))
        metrics.update(finite_stats('dual_logp_dual_tensor', dual_logp_dual))
        metrics.update(finite_stats('main_logp_dual_tensor', main_logp_dual))
        metrics.update(finite_stats('main_logp_main_tensor', main_logp_main))
        metrics.update(finite_stats('dual_logp_main_tensor', dual_logp_main))
        metrics.update(finite_stats('dual_g_tensor', dual_g))
        metrics.update(finite_stats('dual_entropy_tensor', ent))
        metrics.update(finite_stats('dual_kl_dual_to_main_tensor', kl_dual_to_main))
        metrics.update(finite_stats('dual_kl_main_to_dual_tensor', kl_main_to_dual))
        metrics.update(finite_stats('dual_pre_loss_tensor', pre_loss))
        metrics.update(finite_stats('dual_weight_tensor', weight))
        metrics.update(finite_stats('dual_is_weight_tensor', is_weight))
        metrics.update(finite_stats('dual_actor_mask_tensor', actor_mask))
        metrics.update(finite_stats('dual_q_loss_tensor', q_loss))
        metrics.update(finite_stats('dual_kl_loss_tensor', kl_loss))
        metrics.update(finite_stats('dual_loss_tensor', loss))
        return loss, {'metrics': metrics}

      mets, aux = self.dual_actor_opt(self.dual_actor, loss_fn, dual_traj, has_aux=True)

      metrics = aux['metrics']
      metrics.update(mets)
      reference_mets = self.update_reference_kl_lam(metrics['reference_kl'])
      metrics.update({f'dual_{k}': v for k, v in reference_mets.items()})
      lam_mets = self.update_dual_lams(
          metrics['dual_kl_dual_to_main'],
          metrics['dual_kl_main_to_dual'],
      )
      metrics.update({f'dual_{k}': v for k, v in lam_mets.items()})
      metrics['dual_train_enabled'] = dual_enabled
      metrics['dual_train/dual/lam3'] = jnp.maximum(self.lam3.read(), 0.0)
      metrics['dual_train/dual/lam4'] = jnp.maximum(self.lam4.read(), 0.0)
      return {f'dual_actor_{k}': v for k, v in metrics.items()}


class QFunction(nj.Module):   #通用q_value的头

  def __init__(self, name, config):
    self.config = config
    self.net = nets.MLP((), name=f'{name}_net', dims=None, **self.config.qcritic)
    self.slow = nets.MLP((), name=f'{name}_slow', dims=None, **self.config.qcritic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name=f'{name}_opt', **self.config.critic_opt)

  def latent_feat(self,traj):
    feat = []
    if 'deter' in traj:
      feat.append(traj['deter'])
    if 'stoch' in traj:
      stoch = traj['stoch']
      feat.append(stoch.reshape((*stoch.shape[:-2], -1)))
    return jnp.concatenate(feat, -1)

  def q_input(self,traj):
    feat = self.latent_feat(traj)
    act = traj['action']
    x = jnp.concatenate([feat, act], -1)
    return {'tensor': x}



  def _dist(self, traj):
    inp = self.q_input(traj)
    return self.net(inp)

  def _slow_dist(self, traj):
    inp = self.q_input(traj)
    return self.slow(inp)

  def predict(self, traj):
    return self._dist(traj).mean()

  def target(self, traj):
    return self._slow_dist(traj).mean()

  def loss(self, traj, target):
    traj = {k: v[:-1] for k, v in traj.items()}
    dist = self._dist(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self._slow_dist(traj).mean()))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    weight = traj['weight']
    critic_is_weight = traj.get('critic_is_weight', jnp.ones_like(weight))
    critic_weight = weight * critic_is_weight
    norm_weight = critic_weight / jnp.maximum(critic_weight.mean(), 1e-8)
    loss = loss * sg(norm_weight)
    metrics = {'q_mean': dist.mean().mean()}
    metrics.update(finite_stats('q_input_action', traj['action']))
    metrics.update(finite_stats('q_input_weight', traj['weight']))
    metrics.update(finite_stats('q_input_critic_is_weight', critic_is_weight))
    metrics.update(finite_stats('q_input_critic_weight', critic_weight))
    metrics.update(finite_stats('q_target', target))
    metrics.update(finite_stats('q_pred', dist.mean()))
    metrics.update(finite_stats('q_reg', reg))
    metrics.update(finite_stats('q_loss_tensor', loss))
    if 'deter' in traj:
      metrics.update(finite_stats('q_input_deter', traj['deter']))
    if 'stoch' in traj:
      metrics.update(finite_stats('q_input_stoch', traj['stoch']))
    return loss.mean(), metrics

  def train(self, traj, target):
    mets, aux = self.opt(self.net, self.loss, traj, target, has_aux=True)
    self.updater()

    aux.update(mets)

    # 找 optimizer 里的 loss
    for k, v in mets.items():
      if 'loss' in k:
        aux['critic_opt_loss'] = v
        break

    return aux

class RewardQFunction(QFunction):  #奖励
  def __init__(self, config, name='reward_q'):
    super().__init__(name=name, config=config)
    self.opt = jaxutils.Optimizer(name=f'{name}_opt', **self.config.reward_critic_opt)
 
class RiskQFunction(QFunction):     # todo 代价风险
  def __init__(self, config, name='risk_q'):
    super().__init__(name=name, config=config)
    self.opt = jaxutils.Optimizer(name=f'{name}_opt', **self.config.risk_critic_opt)


class DualRiskQFunction(QFunction):
  def __init__(self, config, name='dual_risk_q'):
    super().__init__(name=name, config=config)
    self.opt = jaxutils.Optimizer(name=f'{name}_opt', **self.config.risk_critic_opt)

    
class RecoverQFunction(QFunction):
  def __init__(self, config, name='recover_q'):
    super().__init__(name=name, config=config)


def compute_risk_backup(traj, target_risk_q, config):
    # traj: [T+1, B, ...] 滚动的估计target_q 
    next_traj = {k: v[1:] for k, v in traj.items()}
    risk_next = jnp.clip(target_risk_q.target(next_traj), 0.0, 1.0)

    if 'cost' in traj:   #轨迹中加入cost的原因是为了在这个地方能够根据轨迹的cost来区分是可行区域还是不可行区域。
      cost = jnp.clip(traj['cost'][1:], 0.0, 1.0)
    else:
      raise ValueError('traj must contain cost for risk backup')

    if 'cont' in traj:
      done = jnp.clip(1.0 - traj['cont'][1:], 0.0, 1.0)
    else:
      done = jnp.zeros_like(cost)

    backup = cost + (1.0 - done) * (1.0 - cost) * config.dual.cost_gamma * risk_next
    return sg(jnp.clip(backup, 0.0, 1.0))
