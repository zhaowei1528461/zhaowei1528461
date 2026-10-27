# SUMO 长尾数据阶段 2：规则风险与 RSSM 验证

本批新增独立规则评分器、模型适配器和离线验证命令。它们只读取阶段 1 episode，
没有修改原 RSSM、训练、replay 或 checkpoint 写入流程。未调用这些模块时不会运行新功能。
目前本机完成规则和 CPU 接口验证，真实 checkpoint 位于用户的另一台服务器，尚未验证其模型前向。

## 先在本机验证规则

在项目根目录、带 NumPy 的 Python 环境运行：

```bash
python embodied/envs/evaluate_scenario_risk.py --input docs/6c873565d1ba7e753c679303.npz --output docs/stage2_rule_validation --seed 0
```

当前 Windows 环境可将 `python` 换成 `D:/anaconda3/python.exe`。
默认使用输入的原生 dt；这份数据为 0.2 秒，history=11、future=30，未来跨度 6 秒。
阶段 1 曾以 dt=0.1 形成 205 个插值窗口，本命令原生网格形成 83 个窗口；这是采样参数差异。

输出文件：

- `summary.json`：输入 SHA256、场景 ID、sim seed、配置、环境、规则窗口数和模型验证状态。
- `windows.jsonl`：逐窗口风险，包含 score、min_ttc、min_distance、progress_penalty、validity_flags、来源。
- `observed_frames.jsonl`：每个原始时间点的几何规则与旧 TTC/cost/collision 标签并排保存；不重复计数重叠窗口。

规则只验证已给出的轨迹及数据边界，不能由此证明整套训练或新生成器已经在 SUMO 跑通。
已有 SUMO 闭环的日志结论仍按用户之前的验证处理。

## 在 checkpoint 所在服务器验证 RSSM

先把本次新模块、阶段 1 数据模块及上述 NPZ 同步到服务器项目，在原训练环境中执行：

```bash
python embodied/envs/evaluate_scenario_risk.py \
  --input docs/6c873565d1ba7e753c679303.npz \
  --output sumo_fpi_log/20261010-160631_osrp_sumo_lag_vector_danger_sumo_None_0/stage2_validation \
  --checkpoint sumo_fpi_log/20261010-160631_osrp_sumo_lag_vector_danger_sumo_None_0/checkpoint.ckpt \
  --model-dt 0.2 --seed 0 --max-model-windows 3 --platform cpu
```

默认读取 checkpoint 同目录的 `config.yaml`，不用猜模型尺寸或训练配置。
若配置存放在其他位置，追加 `--model-config <实际config.yaml路径>`。
只有保存的配置确实丢失时，才可显式使用 `--config-name osrp_sumo_lag_vector_danger`；
这无法还原训练时的 CLI 覆盖，配置不匹配应处理为失败。
`--model-dt` 必须等于原训练环境时间步；0.2 是本份数据的采样间隔，仍须核对训练设置。
CPU 太慢时，可在原有可用 GPU 环境把 `--platform cpu` 改为 `--platform gpu`。

检查命令退出码和输出：

1. `summary.json` 中 `model.checkpoint_loaded=true` 表示原 checkpoint 的 agent 条目已读入。
2. `model.inference_verified=true` 表示抽取窗口的模型前向全部返回完整、有限的输出；仅加载成功不算前向通过。
3. `model_windows.jsonl` 分别记录 free rollout 和 teacher forcing 的 1/5/10/30 步观测误差与不可用原因。
4. `prediction_*.npz` 保存解码 state/reward/cost/cont、原 raw_states 真值及 mask。数组无 pickle。

退出码 0 表示当前命令请求的检查通过；规则输入不满足条件返回 1；请求的 checkpoint
加载/推理失败或适配条件不支持返回 2。失败仍会保留已完成的规则报告及模型错误原因。
默认抽取起、中、末 3 个窗口，可增大 `--max-model-windows` 做更广验证。
固定输入、checkpoint 和 seed 可重跑对照；不保证不同 JAX 版本/设备之间逐位相同。

## 实际模型能力与接口

原调用链为 `Agent.preprocess → wm.encoder → rssm.observe/img_step/imagine → 原 decoder/reward/cost/cont`。
原 `state` 是 31 维归一化 observation，自车 `action` 是 1 维；其中邻车槽位不是固定车辆 ID。
模型没有显式 NPC plan 输入或绝对世界多车轨迹 decoder。

| 接口 | 输入及结果 |
| --- | --- |
| `WorldModelAdapter(episode, backend=None, seed=0, model_dt=None)` | 同一源 episode 提供 raw_states 和原生动作；backend 未提供时明确不可推理 |
| `encode_history(window)` | 原生 `[1,H,31]` 观测和 `[1,H,1]` incoming actions，返回 posterior latent；在窗口首帧明确冷启动 |
| `predict_future(window, ego_policy=None, npc_plan=None, horizon=None)` | 已记录 `[1,F,1]` 自车动作驱动 prior free rollout；未来观测用于独立 teacher forcing 对照 |
| `evaluate_risk(window, candidate=None)` | 规则 `RiskReport`；基于实测轨迹或固定实测自车响应的 NPC 候选，不使用 latent 当位置 |

`npc_plan`、自定义 `ego_policy` 会返回 `supported=False`。
插值时间网格、缺失历史 raw_state、缺失动作、错误源 episode、非有限/截断模型输出也明确拒绝。
保留原 `embodied.Checkpoint.load(keys=['agent'])` 容器读取方式；忽略 step/replay 恢复，
Ninjax 前向设置 `create=False, modify=False`，禁止为缺失 checkpoint 参数随机补权重。

teacher forcing 会输入未来观测，不能当作自由预测性能；free rollout 只输入已记录的自车动作，
不是会根据新 NPC 轨迹重新响应的规控闭环。带旧终帧错位标记时，最后 raw_state 不做误差真值。

模型误差以原归一化 31 维 observation 的 RMSE 表示，不能称米制 ADE/FDE。
ADE/FDE 为 None；有记录 cost 正样本时，risk recall 表示 cost head 对记录 cost 的阈值召回，
不代表已校准碰撞概率或真实碰撞召回。当前样本 cost 全为 0，不能计算正样本召回率。
`predicted_collision_probability` 和可靠 `uncertainty` 都为 None。
raw_state 绝对值大于 5 的标志仅是范围提醒，没有训练分布校准，不能当作已验证 OOD 检测。

## 规则定义和缺失数据

车辆字段为 `[x,y,vx,vy,yaw,a,length,width]`，单位米、秒、弧度。
窗口需要注明 `position_reference`。SUMO 返回前保险杠中心，规则按每辆车的半车长沿 yaw
后移得到车身中心，才计算 OBB。[SUMO TraCI 位置定义](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)
未知位置参考、归一化车辆轨迹或未知坐标系不评分。

TTC 是固定朝向、匀速平移 OBB 在默认 5 秒内首次接触的时间。
远离、平行不同车道或相对静止且不重叠返回 None；已有接触返回 0。
它可能与未来实际转弯轨迹不同，不是实际碰撞时刻。
min_distance 是车体边界间距，接触/重叠返回 0。
近失规则要求未发生几何接触，同时间距不超过 2 米或外推 TTC 不超过 0.5 秒。

score 对 overlap、TTC、distance、progress 的 0–1 项加权，默认权重 0.4/0.3/0.2/0.1。
progress 是车身中心路程的平均速度代理，低于 0.5 m/s 时提高惩罚，至少需要 5 秒。
缺少足够时长时 progress=None，剩余项重算权重；没有道路目标进度指标。
阈值和权重可通过 `--risk-config <JSON文件>` 覆盖。

```json
{"ttc_horizon": 5.0, "near_distance": 2.0, "near_ttc": 0.5,
 "progress_duration": 5.0, "progress_speed": 0.5,
 "weights": {"overlap": 0.4, "ttc": 0.3, "distance": 0.2, "progress": 0.1}}
```

无 NPC 的 N=0 窗口与 NPC 全部缺失分开处理。N>0 时，如果任一评分时刻没有可用 NPC，
窗口不能用于低风险排序；`valid=false, score=None`。mask 声称有效但状态为 NaN/无效尺寸也不能评分。
同一 NPC 在两次有效观测之间出现内部缺口时也不能评分，避免被另一辆始终可见的 NPC 掩盖。
各 ID 的首尾缺席可能是入场/驶离或采集缺口，报告覆盖率及未知原因；
可评分仅针对有效观测到的车对，本批不能由 mask 单独确认首尾缺席原因或场景完整性。
候选 `CandidateTrajectory` 为 `[F,N,8]`、`[F,N]` mask、同一 NPC IDs/dt/坐标系，
不能关闭原未来中有效 NPC 的 mask 来降低风险；不能称反事实闭环。

## 本份真实数据的结果和验收边界

已运行规则命令，报告在 [stage2_rule_validation/summary.json](stage2_rule_validation/summary.json)。
输入 123 帧、11 NPC、dt=0.2，83/83 个窗口通过规则输入检查。
原文件 SHA256 保持 `4029e30a9ef159fc122c67d86d3383c3372d553317e90508e7e53f57448d2394`。

记录 collision=0、near_miss=17、low_ttc=18，旧 TTC 为 0 的帧数为 16。
新完整 OBB 检测到 42 帧接触/重叠，时间 15.2–23.4 秒，最近车辆为 `obs_004`。
已核对该段原快照的坐标、yaw、车辆长宽与前保险杠转换。
旧 `check_collision(..., xishu=0.7)` 使用缩小尺寸及固定 2.18 米偏移，
旧 cost 还包含阈值处理，检测口径与新规则不同；仅凭 NPZ 不能确认这些帧的 SUMO 实际碰撞事件。
因此原 collision 标签不变，新几何结果单独保留，不自动回写为训练碰撞标签。

CPU 验证覆盖规则边界、动作时间对齐、原生网格、无 decoder、无模型回退、旧 checkpoint
容器只选 agent、CLI 重跑一致和失败退出。测试里的 CPUSequenceBackend 是接口测试替身，
不计作真实 RSSM 的推理证据。真实 JAX/checkpoint、SUMO 场景重演、概率校准尚未验证。

```bash
python -m unittest discover -s tests
```

本轮最终执行上述全量命令：87 项通过（2026-10-10，Python 3.8.3 / NumPy 1.24.4，4.357 秒）。
其中阶段 2 新增 32 项测试，阶段 1 及原有回归 55 项。
另用用户提供的服务器相对 checkpoint 路径在本机验证缺失文件分支：退出码 2，
规则仍为 83/83，`checkpoint_loaded=false`、`inference_verified=false`，明确记录 FileNotFoundError。

后续先验收服务器真实模型报告，再按任务书阶段 3 做参数搜索基线。当前没有实现 Random/CEM、
Diffusion 或自动训练回灌。几何风险与旧标签存在口径差异，应在回灌前明确选用的风险定义。
