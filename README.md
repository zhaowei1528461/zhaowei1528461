# SafeDreamer FPI

This repository is a SafeDreamer-style model-based reinforcement learning
project with a feasible dual policy iteration branch. The current default
configuration uses a dual Q actor-critic (`critic_type: dual_q`) and supports
SUMO, Safety Gym, Gym, DMC, and other environments through the local
`embodied` package.

## Main Entry Points

- `train.py`: command-line entry point. It loads `configs.yaml`, builds replay
  buffers, environments, loggers, the agent, and dispatches to the selected
  run script.
- `agent.py`: world model, standard imagined actor-critic variants, and the
  dual Q safe actor-critic implementation.
- `behaviors.py`: behavior wrapper that selects `VFunction`,
  `ImagSafeActorCritic`, or `DualImagSafeActorCritic` based on
  `config.critic_type`.
- `configs.yaml`: default and experiment-specific configuration.
- `embodied/`: local framework code for environments, replay buffers, drivers,
  logging, wrappers, and run loops.

The active debug defaults in `train.py` currently run:

```bash
python train.py --configs osrp_sumo_lag_vector --method osrp_sumo_lag_vector --run.script train_eval --task sumo_None --jax.logical_gpus 0
```

When no CLI arguments are passed, `USE_INTERNAL_DEBUG_ARGS = True` makes
`train.py` use those arguments automatically.

## Training Flow

1. `train.py` parses config, creates train/eval environments and replay
   buffers, then constructs `agent.Agent`.
2. `agent.Agent.train()` preprocesses replay batches, trains the world model,
   builds imagined start states, and calls `task_behavior.train()`.
3. `behaviors.Greedy` creates `DualImagSafeActorCritic` when
   `critic_type: dual_q`.
4. `DualImagSafeActorCritic.train()` imagines one rollout from the main actor
   and one rollout from the dual actor, trains Q critics, then trains the main
   actor and dual actor.
5. `embodied/run/train_eval.py` maintains the dual policy sampling gate. It
   enables real dual-policy data collection once the feasible-rate EMA exceeds
   `dual.enable_thresh`.

## Dual Q Components

`DualImagSafeActorCritic` owns:

- main actor: `actor`
- dual actor: `dual_actor`
- reward critics: `reward_q1`, `reward_q2`
- main risk critics: `risk_q1`, `risk_q2`
- dual risk critics: `dual_risk_q1`, `dual_risk_q2`
- recovery critics: `recover_q1`, `recover_q2`
- adaptive coefficients: `log_cg`, `lam1`, `lam2`, `lam3`, `lam4`

The Q critics use latent features plus action as input:

```text
q_input = concat(deter, flatten(stoch), action)
```

Their loss is a distributional negative log likelihood target loss plus slow
critic regularization.

## Importance Sampling Weights

There are two separate imagined-trajectory IS weight paths:

- critic path: `critic_is_weight`
- actor path: `is_weight` plus `actor_mask`

Both are computed from the current main and dual policies on imagined
trajectories, not from replay's stored `log_weight` or `log_weight_dual`.
Replay log weights are preserved and logged, but the current critic/actor
updates recompute ratios inside `DualImagSafeActorCritic`.

### Ratio Computation

`_traj_is_ratio(traj, numerator, cumulative)` computes:

```text
main_logp = log pi_main(action | latent)
dual_logp = log pi_dual(action | latent)

if numerator == "main":
  log_ratio = main_logp - dual_logp
if numerator == "dual":
  log_ratio = dual_logp - main_logp
```

When `cumulative=True`, it uses a cumulative sum over imagined time. When
`cumulative=False`, it uses per-step ratios. The log ratio is clipped by:

```text
log(dual.is_min_weight) <= log_ratio <= log(dual.is_max_weight)
```

The default clip range in `configs.yaml` is:

```yaml
dual:
  is_min_weight: 0.1
  is_max_weight: 5.0
```

### Critic Updates

Critic training uses per-step cross-policy IS ratios:

```python
r_pd = _traj_is_ratio(dual_traj, numerator='main', cumulative=False)
r_dp = _traj_is_ratio(main_traj, numerator='dual', cumulative=False)
```

where:

- `r_pd`: reweights dual-imagined samples as if they came from the main policy.
- `r_dp`: reweights main-imagined samples as if they came from the dual policy.

The gate is:

```python
cross_gate = (dual_enabled > 0.5).astype(jnp.float32)
```

For main/primal critics (`reward_q1`, `reward_q2`, `risk_q1`, `risk_q2`):

```text
main_traj weight: critic_is_weight = 1
dual_traj weight: critic_is_weight = r_pd * cross_gate
```

For dual risk critics (`dual_risk_q1`, `dual_risk_q2`):

```text
dual_traj weight: critic_is_weight = cross_gate
main_traj weight: critic_is_weight = (1 - cross_gate) + r_dp
```

Inside `QFunction.loss()`:

```python
critic_weight = traj['weight'] * critic_is_weight
norm_weight = critic_weight / max(mean(critic_weight), 1e-8)
loss = loss * stop_gradient(norm_weight)
return loss.mean()
```

So the IS weight changes the relative contribution of each time/batch element
to critic gradients, after multiplication by the world-model rollout
continuation weight `traj['weight']`. The normalization keeps the average
critic weight around 1, so IS changes sample emphasis more than global loss
scale.

The recovery critics are the exception: `recover_q1` and `recover_q2` are
trained only on `main_traj` and receive no `critic_is_weight`, so they fall back
to all-ones IS weights.

### Actor Updates

Actor training uses cumulative cross-policy IS ratios:

```python
r_pd = _traj_is_ratio(dual_traj, numerator='main', cumulative=True)
r_dp = _traj_is_ratio(main_traj, numerator='dual', cumulative=True)
```

For the main actor:

```text
main_traj: is_weight = 1, actor_mask = 1
dual_traj: is_weight = r_pd * cross_gate, actor_mask = cross_gate
```

Inside `train_main_actor()`:

```python
actor_weight = traj['weight'] * actor_mask
norm_weight = actor_weight / max(mean(actor_weight), 1e-8)
loss = region_loss - actent * entropy
loss = loss * stop_gradient(is_weight)
loss = weighted_mean(loss, stop_gradient(norm_weight))
```

Thus `is_weight` scales the main actor objective itself, while `actor_mask`
decides whether a trajectory branch participates in the weighted mean. The
world-model weight still contributes through `norm_weight`.

For the dual actor:

```text
dual_traj: is_weight = 1, actor_mask = cross_gate
main_traj: is_weight = (1 - cross_gate) + r_dp, actor_mask defaults to 1
```

Inside `train_dual_actor_on_traj()`:

```python
q_loss = -dual_g * stop_gradient(is_weight)
kl_loss = lam3 * KL(dual || main) + lam4 * KL(main || dual)
loss = q_loss + kl_loss - actent * entropy
loss = weighted_mean(loss, stop_gradient(norm_weight))
```

Only the dual risk term `-dual_g` is multiplied by IS. The KL and entropy terms
are not directly multiplied by `is_weight`; they are only affected by
`actor_mask` and `traj['weight']` through `norm_weight`.

## Important Metrics

Useful logged metrics for inspecting IS behavior include:

- `train/primal_from_dual_ratio_*`
- `train/dual_from_main_ratio_*`
- `train/*q_input_critic_is_weight_*`
- `train/main_actor_is_weight_mean`
- `train/dual_actor_is_weight_mean`
- `train/main_actor_mask_mean`
- `train/dual_actor_mask_mean`
- `train/dual_train/dual/IS_weight`
- `train/dual_train/dual/IS_weight_dual`

## Notes

- The repository does not currently include a `requirements.txt` or
  `pyproject.toml`; dependencies are inferred from imports and environment
  wrappers.
- The code expects JAX, Optax, TensorFlow Probability for JAX, NumPy,
  ruamel.yaml, and any environment-specific packages used by the selected task
  such as SUMO or Safety Gym.
- Several comments in the source appear mojibake-encoded in some shells. The
  executable code paths are still readable through symbol names and structure.

## 危险场景筛选与生成（结构化 SUMO 数据）

`osrp_sumo_lag_vector_danger` 是面向当前 31 维结构化状态训练的
World-Engine 风格配置。它不依赖图像或 3DGS，而是保留同样的闭环原则：
当前策略在哪些场景失败，就优先扩展和训练那些场景。

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0
```

## SUMO Danger Scenario Mode Commands

第二轮危险场景生成由 `env.sumo.danger.generation_mode` 控制，可选值：

- `copy`: 只使用危险数据复制，也就是 `scenario_copy`。
- `attack`: 只使用危险数据对抗，也就是 `intent_attack`。
- `replay`: 只使用危险数据回放。
- `mixed`: 三种混合，按 `replay_probability` 和 `intent_attack_probability` 采样。
- `off`: 不从危险库生成场景，reset 使用普通 SUMO 场景。

下面命令均使用当前的 `osrp_sumo_lag_vector_danger` 配置、`train_eval` 流程和 `sumo_None` 任务。

1. 只使用危险数据复制：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --env.sumo.danger.generation_mode copy
```

2. 只使用危险数据对抗：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --env.sumo.danger.generation_mode attack
```

3. 只使用危险数据回放：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --env.sumo.danger.generation_mode replay
```

4. 三种混合：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --env.sumo.danger.generation_mode mixed
```

5. 不使用场景评估，使用原始 `osrp_sumo_lag_vector_danger` 训练路径：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --danger_posttrain.enabled False
```

如果只想跳过全场景 sweep，但仍保留 bootstrap 后的 posttrain 框架，可使用：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0 --danger_posttrain.evaluate_all_scenarios False --env.sumo.danger.generation_mode off
```

训练期会执行下列闭环：

```text
前 5,000 个训练回合：普通 SUMO rollout + 常规 RL 更新
  -> 冻结 bootstrap actor，并保存 checkpoint
  -> 冻结策略遍历全部 .sumocfg 基础场景（不生成扰动）
  -> collision / near-miss(TTC、risk cost) / max-time 低进度筛选
  -> 进入危险场景闭环；actor 更新受 KL(当前策略 || bootstrap 策略) 约束
  -> SUMO rollout
  -> collision / near-miss(TTC、risk cost) / max-time 低进度筛选
  -> 使用 sweep 危险库（默认冻结，不再在线追加旧原型）
  -> Scenario Copy（重跑源 cfg）、Intent Attack 或背景车 Replay
  -> 按配置执行初始状态校验
  -> 高风险 replay 序列按比例过采样
```

危险原型与启用记录时的生成日志写入本次日志目录的
`danger_scenarios/danger_events_worker_*.jsonl`。默认 sweep 后冻结旧库和旧事件日志。
生成在 SUMO reset 时在线执行；初始状态校验、重试及是否回退由 danger 配置控制。
评估环境关闭生成，以便指标仍反映自然场景性能。

相关开关位于 `env.sumo.danger`，高风险 replay 的采样比例与 cost 阈值位于
`danger_replay`。默认的 `osrp_sumo_lag_vector` 保持原有行为；只有选择
`*_danger` 配置才会启用这套闭环。

阶段参数位于 `danger_posttrain`：当前危险配置的 `bootstrap_episodes: 5000`、
`reference_kl_target: 1.0`。KL 拉格朗日乘子会按目标自适应更新，并同时约束
主 actor 和 dual actor；冻结的参考 actor 会随 checkpoint 一起保存。

## SUMO 长尾数据采集（阶段 1）

在危险配置后追加 `sumo_long_tail_data`，后训练阶段会把实际执行的完整 episode
写入 `<本次日志目录>/posttrain_episode_pool/*.npz`。冻结旧危险库时仍可采集；
bootstrap 和评估不采集。未追加该配置时，新功能默认关闭。

```bash
python train.py --configs osrp_sumo_lag_vector_danger sumo_long_tail_data --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0
python -m embodied.envs.scenario_data inspect_dataset --input "<本次日志目录>/posttrain_episode_pool" --history 11 --future 30 --dt 0.1
python -m embodied.envs.scenario_data prepare_dataset --input "<本次日志目录>/posttrain_episode_pool" --output "<本次日志目录>/long_tail_dataset" --split-seed 0
```

独立模块提供多车 ID/mask 对齐、动作区间对齐、困难窗口、父场景分组划分和训练集归一化。
现有 31 维观测、1 维自车动作及 RSSM 输入保持原样；数据池尚未自动回灌训练。
数据契约、旧日志转换及验收方法见 [阶段 1 使用说明](docs/long_tail_stage1.md)。

## SUMO 规则风险和 RSSM 离线验证（阶段 2）

独立命令读取阶段 1 NPZ，按原生时间步输出 OBB 间距、TTC 和逐窗口风险；
提供 checkpoint 后可调用原 RSSM 做已记录自车动作下的 free rollout / teacher forcing 对照。

```bash
python embodied/envs/evaluate_scenario_risk.py --input docs/6c873565d1ba7e753c679303.npz --output docs/stage2_rule_validation --seed 0
```

原 31 维模型没有显式 NPC 轨迹条件输入或绝对世界多车 decoder。
服务器 checkpoint 命令、指标含义、实际样本结果与未验证项见 [阶段 2 使用说明](docs/long_tail_stage2.md)。

## SUMO 参数化长尾候选搜索（阶段 3）

新增独立 Random/CEM 搜索，共用预算、风险和约束；输出去重场景 JSON/NPZ，默认不接入训练。

```bash
python embodied/envs/run_scene_search.py --demo --output stage3_runs/demo_seed0 --budget 32 --seed 0 --top-k 5
```

真实 NPZ 需要显式传入对应 `--sumo-cfg`。候选采用固定实测自车响应进行离线评分，
尚未在 SUMO/CARLA 验证执行；接口、真实数据结果和运行命令见 [阶段 3 使用说明](docs/long_tail_stage3.md)。
合流道路上目标车与自车朝向差异较大时，可显式加 `--reference-heading target`；
原 `ego` 参考方向仍为默认，物理/道路约束保持不变，详见 [真实样本修复与复验](docs/stage3_diagnosis/README.md)。
