# SUMO 长尾数据阶段 1：采集、转换和验收

本批在现有 SUMO copy/attack/replay 流程上增加独立轨迹数据管道，落地任务书阶段 1。
现有真实 SUMO 闭环运行结论沿用用户的日志验证。本批验收覆盖新增数据边界，
不把 CPU 或合成测试当作新增采集器已在真实 SUMO 跑通的证据。

## 启用采集

```bash
python train.py --configs osrp_sumo_lag_vector_danger sumo_long_tail_data --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0
```

沿用已有训练启动和 checkpoint 恢复方式，只追加 `sumo_long_tail_data`。
`train.py` 仍按原有规则给日志目录追加时间；本批没有增加跨日志目录迁移危险库或恢复 phase 的功能。
记录本次实际日志目录，下文用 `<RUN_DIR>` 表示。

数据写入 `<RUN_DIR>/posttrain_episode_pool/<episode_id>.npz`，日志显示
`[episode collection] ... path=...`。只在后训练阶段采集，bootstrap、sweep 和评估关闭新采集。
正常结束、碰撞、超时和连接故障的 episode 都进入采集流程；不足两个有效时间点时跳过并报告。
episode 结束时写盘，因此采集文件不会在 episode 尚未结束时出现。

新增开关位于 `env.sumo.danger`：

| 开关 | 默认值 | 作用 |
| --- | --- | --- |
| `collect_episodes` | `False` | 写独立的实际执行轨迹；overlay 设为 `True` |
| `update_library` | `True` | 控制在线原型插入旧危险库 |
| `freeze_danger_library_after_sweep`（在 `danger_posttrain`） | `True` | 后训练停用旧事件记录和在线入库，新采集仍可开启 |

`set_danger_mining(active, record_events=None, collect_episodes=None, update_library=None)`
分别控制阶段激活、旧日志、新采集和在线入库。省略新增参数时读取配置。
旧库外部导入、原有生成采样和随机数路径保持原样；新 NPZ 池不会被旧 JSONL loader 导入。
关闭新采集可移除 overlay，或追加 `--env.sumo.danger.collect_episodes False`。

## 原数据和新数据的边界

调用链是：

```text
train_eval 后训练激活
  -> SUMO reset：已有生成器执行 copy/attack/replay
  -> 捕获 reset 初始帧及每次实际执行后的车辆快照
  -> episode 结束：旧矿工使用原有危险窗口；新采集保存完整 episode
  -> ScenarioEpisode NPZ
  -> make_windows / pad_windows
  -> 场景分组划分 + 训练集归一化
```

采集复用原有每步快照，并额外保留初始帧；新 episode 不受旧 `max_replay_steps` 窗口长度限制。
旧训练仍接收 `state[31]` 和自车 `action[1]`，RSSM、actor 和训练 replay 的接口没有改变。
原始 `state` 独立存储，不用它反推多车位置；末帧旧 state 可能先于车辆快照，元数据明确记录这一限制。

实际变更文件：

| 文件 | 变更 |
| --- | --- |
| `embodied/envs/scenario_data.py` | 新增数据契约、转换、窗口、坐标、划分、归一化与 CLI |
| `embodied/envs/sumo_merge.py` | 接入完整 episode 采集和错误处理 |
| `embodied/envs/danger_scenarios.py`、`embodied/envs/sumo.py` | 分离采集/记录/在线入库，并透传开关 |
| `embodied/run/train_eval.py`、`configs.yaml` | 后训练激活、评估隔离和默认关闭的 overlay |
| `tests/test_danger_scenarios.py` | 修正背景车 fixture 并补兼容回归 |
| `tests/test_scenario_data.py`、`tests/test_danger_phase.py`、`tests/test_sumo_episode_collection.py` | 新增数据契约、阶段开关及 SUMO 采集边界测试 |

`source_type` 标记 `sumo_original/sumo_copy/sumo_attack/sumo_replay`，
`prediction_source=sumo_execution`。保存最终实际使用的变体、目标车辆、源 cfg 和原帧。
`seed/map_id/model_version` 在不能核实时为 `None`；危险采样 seed 单独记录，不能当作 SUMO 的 seed。
生成变体有独立 `scenario_id`，同一源 cfg 使用共同 `parent_scenario_id`。
在线父 ID 当前按 cfg 文件名建立，重名源 cfg 会被保守地放进同一分组。

车辆快照或时间读取失败时，只保存失败前可确认的轨迹前缀：
`metadata.event_type=execution_error`、`collection_complete=False`，保留原终止原因和丢弃帧。
写盘或格式失败显示 `[episode collection failed]`，episode 不会被重复写入旧日志。
数据池没有宣称轨迹已满足动力学、道路或复现约束；`execution_valid=None`，后续仍需验收。

## Episode 和 Window 契约

车辆特征固定为 `F=8`：`[x, y, vx, vy, yaw, a, length, width]`。
世界坐标单位为米，速度 m/s，yaw 为弧度，加速度 m/s²。
已知本仓库 SUMO 采集器的 x/y 参考点为前保险杠中心；后续使用车体 OBB 时需要另行转换几何中心。
参考点定义见 [SUMO 官方文档](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)。
速度分量优先使用实测 `vx/vy`，否则由实测速度和 yaw 推导；不把 SUMO 的角度制 `angle` 当弧度。
各 NPC 按 ID 跨帧对齐，缺失用 `NaN` 和 `False` mask；NPC 数量可以为零。
车辆有效 mask 要求八字段有限且长宽为正。

| 字段 | Episode shape | Window / batch shape |
| --- | --- | --- |
| `timestamps` | `[T]` | `[H+K]` / `[B,H+K]` |
| `ego_state` | `[T,8]` | `history_ego[H,8]`、`future_ego[K,8]` |
| `npc_states` | `[T,N,8]` | `history_npcs[H,N,8]`、`future_npcs[K,N,8]` |
| `npc_valid_mask` | `[T,N]` | `npc_mask[H+K,N]` / `[B,H+K,Nmax]` |
| `ego_actions` | `[T,A]`，当前 `A=1` | `[H+K,A]` / `[B,H+K,A]` |
| `raw_states` | `[T,31]` | 保留在 Episode；不作为车辆轨迹特征 |

`pad_windows` 将不同 N 补齐为 Nmax，NPC 历史和未来分别为
`[B,H,Nmax,8]`、`[B,K,Nmax,8]`，padding mask 为 False。
`npc_actions` 当前为 None；`map_features` 仅保留实际提供的数据。
Episode 保存每帧 `events`，Window 保存 `event_labels`，同时保存标签定义和阈值。

默认 `H=11`、`K=30`、`dt=0.1` 可配置。历史最后一帧是 anchor，未来第一帧为 anchor+dt，
总窗口覆盖 4 秒。原始时间轴必须来自 `sim_time`，严格递增；原始不等频 episode 的 `dt=None`。
重采样不跨缺失车辆样本，默认最大插值间隔为原始时间间隔中位数的 1.5 倍。
yaw 按每段连续有效轨迹 unwrap，再插值，避免跨 ±π 跳变。

原始帧的 action 表示到达该帧所施加的动作；转换后 `ego_actions[t]` 对应 `[t,t+1]`。
末帧没有可确认的下一步动作，其 mask 为 False。重采样动作使用区间控制，避免对控制量线性插值。
到达终点时若最后一帧明确标记 `simulation_advanced=False` 且时间重复，保留终止结果，
合并该帧并排除未执行的最后动作；原帧仍可审计。其他重复/倒序时间明确拒绝。

保留世界坐标；`make_windows(..., coordinate_frame='ego-centric')` 以历史末帧自车位置和朝向为固定参考系。
`to_frenet` 只对明确提供的世界坐标参考折线投影出 `[s,d]`；没有参考线时返回 None 和缺失标记。
它不推断车道归属，也不把世界 vx/vy 冒充 Frenet 速度。

## 检查、划分和旧日志转换

```bash
python -m embodied.envs.scenario_data inspect_dataset --input "<RUN_DIR>/posttrain_episode_pool" --history 11 --future 30 --dt 0.1 --split-seed 0
python -m embodied.envs.scenario_data prepare_dataset --input "<RUN_DIR>/posttrain_episode_pool" --output "<RUN_DIR>/long_tail_dataset" --ratios .8 .1 .1 --split-seed 0 --group-by scenario
```

`inspect_dataset` 输出 episode/window 数、事件计数、字段缺失率、mask 有效率、shape、
原始 dt 和按父场景分组的拟议划分。事件计数是帧计数；重复重叠窗口不是独立场景数量。
不足 4 秒的完整 episode 可以保存，但默认窗口数量为零；可调整历史/未来长度。

`prepare_dataset` 输出 `split_manifest.json` 和 `normalizer.json`，不复制原轨迹或改变在线训练。
划分先合并同场景及祖先关联，再按固定 seed 分组；同一源场景的扰动不会跨 train/val/test。
默认按组数分配 0.8/0.1/0.1，episode 数量比例不一定相同。
`--group-by map` 可额外合并明确具有相同 map_id 的场景；未知 map 不凭空分组为同一地图。
归一化只拟合 train 的有效世界坐标车辆，保存训练 episode IDs。
七个标量字段使用均值方差，yaw 使用圆周均值和 wrapped residual 标准差；
该归一化器用于世界坐标，不应直接套用到 ego-centric 数据。

旧危险 JSONL 中若包含 `replay_trajectory.frames[].vehicles`，可离线转换：

```bash
python -m embodied.envs.scenario_data convert --input "<RUN_DIR>/danger_scenarios/danger_events_worker_0.jsonl" --output "<RUN_DIR>/converted_episode_pool" --ego-id Auto
```

没有 frames 的 sweep 汇总、variant 日志和整条完全没有车辆快照的记录明确拒绝。
零星缺车继续使用 mask。CLI 报告每条 rejected 的源位置及原因，并返回非零退出码；
混合输入中有效记录仍可转换。旧危险窗口本身可能已被截断，离线转换不会补造完整 episode。

困难窗口示例：

```python
from embodied.envs.scenario_data import load_episode, make_windows, pad_windows

episode = load_episode('posttrain_episode_pool/<episode_id>.npz')
windows = make_windows(
    episode, history=11, future=30, dt=0.1,
    event_only=True, event_before=3.0, event_after=1.0,
    event_config={'near_miss_ttc': 0.5, 'low_ttc': 2.0,
                  'low_progress_speed': 0.5, 'low_progress_duration': 5.0})
if windows:
    batch = pad_windows(windows)
```

collision 来自实测碰撞标记或终止原因；near-miss 来自记录的非负 TTC 阈值且排除 collision；
low-TTC 单独标记；low-progress 依据实测低速持续时间。
这套窗口标签是显式定义的观测基线，不是新风险模型。
在线 episode 摘要沿用旧矿工的事件优先级和阈值，独立保存于 metadata，冻结旧日志时仍能标记碰撞。

## 本批验证和下一阶段依赖

```bash
python -m unittest discover -s tests -v
```

2026-10-10 最终验证：

- 使用 `D:/anaconda3/python.exe`（Python 3.8.3）运行上述 unittest discovery，**52 项通过**，1.768 秒。
- 独立 CLI 合成样本：20 个 episode、420 个默认窗口；train/val/test 为 16/2/2，父场景无交叉，归一化训练 IDs 与 train 完全一致。
- 真实可见的 500 条 sweep 汇总转换为 0 个 episode、500 条明确拒绝，原因均为缺少轨迹帧。
- 修改及新增 Python 文件语法编译通过；配置组合由测试使用真实 Flags 解析验证。
- 独立审查发现的重复结束、终止快照故障及无车辆快照问题已补回归并修复，冻结日志时的碰撞标签也完成 RED→GREEN 验证。

实现时的 SUMO 边界测试使用真实环境方法并 mock 外部 TraCI，本地没有启动 GPU 训练或真实 SUMO。
随后已验收用户提供的实际采集 NPZ：123 帧、11 个 NPC、205 个默认窗口，时轴、ID/mask 和动作对齐通过。
详情见 [真实样本格式验收报告](long_tail_stage1_real_sample_validation.md)。
整批采集的数量、耗时、磁盘增长、多场景划分和物理事件验证仍待进一步运行验收。

真实运行新增采集后，应先检查 pool 的 episode/window 数、时间、缺失率和事件分布，
再固定场景划分供后续实验使用。当前可见的 500 条 `bootstrap_sweep_results.jsonl`
只有汇总，没有车辆帧，无法代替轨迹数据。

下一批依赖本数据池的实际样本与质量检查，建立任务书阶段 2 的规则风险/RSSM 推理适配器，
随后才做阶段 3 的 Random/CEM 场景搜索。当前 RSSM 没有显式 NPC 计划条件输入，
不能直接宣称可对指定 NPC 轨迹作反事实闭环预测。
Diffusion、风险引导、可复现长尾执行验收和 verified-tail 回灌仍属于后续阶段。
