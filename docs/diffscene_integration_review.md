# DiffScene 代码审查与阶段四接入方案

日期：2026-10-26。用户已选择：**复用 DiffScene 的时序去噪网络，直接生成单个 NPC 的 30 帧未来轨迹**。本文保留实现前的源码审查和接入设计；后续已完成的实现、CPU 验证与服务器命令见 [阶段四使用说明](long_tail_stage4.md)。短训仅验证运行链路，生成质量尚未通过验收。

## 1. 目前真实存在的条件

已有 canonical episode、固定父场景划分、窗口索引和 train-only 自车局部坐标归一化器。依据用户提供的服务器运行报告：

| 集合 | episode | 合格窗口 | 完整目标轨迹 |
| --- | ---: | ---: | ---: |
| train | 779 | 428346 | 3469519 |
| val | 98 | 57393 | 456045 |
| test | 74 | 40127 | 309545 |

服务器窗口产物目录为 `stage4_windows_20261026_160548`，窗口配置 H=11、F=30、dt=0.2、stride=1，即 2 秒历史（含 anchor）和 6 秒未来。归一化器为 `normalizer-c2fff174d1731e3650fc`，拟合来自 779 个 train episode。原划分清单 SHA256 为 `52add82e1a413bd04d753843874bb3bb8f78d47535de996430137ac7e1ab2b92`。这些全池数字来自服务器报告，不是本地独立重跑结果。

本地上传的 `diffscene-main` 含 46 个 Python 文件，语法解析通过；没有 `datasets/` 自定义数据包，也没有 `.pth/.pt/.ckpt` 权重。当前本地 Python 3.8.3 环境没有 torch、einops、torch_geometric、configargparse 或 nuscenes，因此本次验证限于源码、依赖检查和 NumPy 公式核对。检查证据、关键源文件 SHA256 记录在 `docs/diffscene_review/static_checks.json`。

## 2. 上传代码实际在做什么

当前调用链是：

```text
NuScenesDataset + NuScenesMapEnv
  -> scene_graph: past / future / visibility / semantic / vehicle attributes
  -> TrafficModel.encode_map / encode_past / encode_future
  -> posterior_mu (含未来信息的训练目标), prior_mu (历史/地图条件)
  -> DiffusionLatent.loss(posterior_mu, prior_mu)
  -> 采样 latent
  -> TrafficModel.decoder
  -> 多车轨迹
```

关键源码位置：

- `scripts/train_diffusion.py:155` 附近的 `run_one_epoch`，训练目标是 `post_mu`，条件是 `prior_mu`；`main` 明确要求预训练的 TrafficModel 权重。
- `models/traffic_model.py:46` 硬编码 nuScenes 的 `dt=0.5`；状态为 `(x,y,hx,hy,s,hdot)` 六维，车长/车宽单独两维，和当前八维 canonical 状态不同。
- `models/diffusion.py:242` 将 `[N,latent_dim]` 重复成 `[N,sample_size,latent_dim]` 再加噪；默认 `sample_size=20`。这个轴是重复 latent 的样本轴，不是当前数据的 30 个未来时间点。
- `models/temporal.py:60` 的 TemporalUnet 接收 `[B,H,D]`，通过 ResidualTemporalBlock 的 context MLP 注入条件，是直接轨迹扩散可复用的主要结构。
- `models/helpers.py:31` 的 Downsample1d 和 Upsample1d 实际都使用 stride=1，当前上传版本不改变时序长度。不能只凭类名推断它要求长度为 8 的倍数，也不应无依据把 30 帧改成 32 帧。

因此，接入不等于把旧训练脚本的数据路径改成 SUMO NPZ。

## 3. 路线选择

| 路线 | 工作及代价 | 本次决定 |
| --- | --- | --- |
| 保留 CVAE → latent diffusion → decoder | 补齐 datasets/map/graph 接口，适配六维状态、属性和 dt；先训练并验证 SUMO TrafficModel，再冻结它训练 latent diffusion。NuScenes 权重即使另行提供，也不能未经迁移验证便视为 SUMO 模型 | 暂不采用 |
| 复用时序网络，直接对 NPC 未来 x/y 做扩散 | 读取已经验收的固定窗口；新增历史/邻车/地图条件编码器；替换 latent 的重复样本逻辑，统一扩散日程与采样接口 | **用户已选择**，符合任务书阶段四 |

第一版保留实测历史，只生成一个指定 NPC 的未来；其他车辆作为历史条件。离线评估需要参考背景未来时，可明确保留录制计划，但不能把这些未来当作生成模型的条件或声称背景已经作出闭环响应。

## 4. 目标数据流和张量契约

```text
window_dataset_manifest.json + window_index.jsonl
  -> 校验固定划分、来源 hash、配置和 normalizer_id
  -> 按 episode 缓存原始/重采样数据，按索引取当前窗口
  -> ego anchor 局部坐标 + 已拟合 normalizer
  -> 历史/邻车/可用地图条件编码器
  -> TemporalUnet 风格去噪网络 + DDPM 噪声预测训练
  -> DDPM / DDIM 无风险引导采样
  -> 反归一化 + 同一 anchor 变换回世界坐标
  -> 运动学派生、有效性指标和候选导出
```

建议的第一版接口：

| 数据 | 形状 | 说明 |
| --- | --- | --- |
| `ego_history` | `[B,11,8]` | 只用历史，含 anchor |
| `target_history` | `[B,11,8]` | 指定完整目标 NPC 的历史 |
| `neighbor_history` | `[B,11,N,8]` | 可变 NPC 数，用 padding 和 mask；不包含重复的目标 |
| `neighbor_mask` | `[B,11,N]` | 原始可见性和有效性，不将缺失误当真实零状态 |
| `map_features / map_mask` | 由 SUMO 几何适配器显式定义 | 当前地图缺失时保留缺失标记，不伪造 nuScenes 图层 |
| `future_target_xy` | `[B,30,2]` | 训练标签，来自指定目标的未来；不送入条件编码器 |
| `condition` | `[B,C]` | 历史/邻车/有效地图的编码；C 由新配置明确指定 |
| 模型采样输出 | `[B,K,30,2]` | K 是独立采样数量，和 30 帧时间轴分开 |

原始八维数据约定保持 `(x,y,vx,vy,yaw,a,length,width)`。条件编码器可从 yaw 派生 sin/cos，避免角度分支跳跃，但不能将导入框架的六维状态直接当成这八维状态。已有 normalizer 的 x/y 均值和标准差可用于 x/y 标签；PyTorch 端应以 buffer 实现对应变换，并校验 ID/坐标系，不把 NumPy normalizer 误当原 TrafficModel 的状态/属性 normalizer。

目标从当前行的 `eligible_target_ids` 中选择。建议 train 采用受 seed/epoch 控制的目标采样，val 固定一个仅依据历史/anchor 的确定性选择规则；训练单位因此是“窗口+目标”，不等于 1.47 亿拟合状态。第一版如何处理全部窗口/全部目标必须写进配置和报告，不能把完整目标轨迹数量直接当作每轮训练步数。

## 5. 数据适配的关键要求

1. 沿用已固定的 779/98/74 episode 成员关系；不在 DataLoader 内重新随机划分重叠窗口。
2. 读取 `window_dataset_manifest.json`，检查 `preparation_exit_code=0`、schema、H/F/dt、索引 hash 和 normalizer 身份。原始数据路径迁移沿用相对路径和显式 input-root。
3. 验证每个来源 NPZ 的 hash，再按当前窗口配置和 `start_grid_index` 取样；`window_index` 不是原始帧号。
4. 使用有上限的每 worker episode 缓存/episode 分组采样。不要为几十万个窗口逐次完整解压同一个 NPZ，也不要每次 `__getitem__` 都调用 `make_windows` 重建整个 episode 的全部窗口。
5. 缺失帧保留 mask。送入神经网络前可以把被 mask 的位置数值置零以避免 NaN 传播，但零必须同时受 mask 控制，不能改写原始数据或计入有效 loss。
6. 模型条件不读取 `future_ego`、邻车录制未来、目标未来编码或未来事件标签。后验 future encoder 属于原 latent 路线，不能移植进直接轨迹模型的条件分支。
7. 地图优先接已有 `scene_road.py` 的 SUMO lane/junction 几何，围绕同一 ego anchor 转换。它没有 nuScenes 的图层语义，也不自动证明原录制地图身份；缺失映射应显式报告。场景类型没有可靠标记时使用 unknown/mask。

## 6. 复用前需要修正的源码问题

以下来自静态源码检查；除明确注明的 NumPy 公式核对外，没有在 PyTorch 中执行证明。

### 扩散时间步索引不一致

`VarianceSchedule` 在 `betas` 头部补零，形成索引 0..T；`loss` 和采样循环却用 0..T-1。同时 `q_posterior` 将部分系数 `[1:]` 截断、另一些缓冲不截断，再用相同 t 取值，前向加噪和反向后验对应的步不一致。

NumPy 核对：对于 beta=[0,0.1,0.2,0.3,0.4]，`q_sample(t=2)` 使用 alpha_bar=0.72；同一 t 的正确后验系数 c1=0.677630927，当前截断数组的下标 2 却取到第 3 步的 0.513222664。新实现应统一 0..T-1 的训练/缓冲/反向采样约定，加入端点和公式测试。

`sigmas_inflex` 的命名或在 DDPM 更新中混合方差，不等于实现 DDIM 的跳步更新。阶段四要求的 DDIM 20/50 步需要独立实现并验证。

### 采样形状是 latent 专用

`p_losses` 默认复制 20 份 latent；`p_sample_loop` 用 `[shape[0],sample_size,shape[1]]` 重建形状，忽略直接轨迹需要的时间/坐标轴；`forward` 的 `len(cond[0])` 也不能作为一般 `[B,C]` 条件的 batch 大小。需要以 `[B,30,2]` 加噪，采样 K 份时只展开 batch，再还原 `[B,K,30,2]`。

### 训练/设备/接口问题

- `utils/torch.py:12` 默认选择 `cuda:1`，应替换成独立入口的可配置 device。
- 原训练脚本没有显式冻结预训练 TrafficModel；优化器只管理扩散网络参数。若走 latent 路线，需要 eval/requires_grad/no_grad 的一致边界；直接轨迹路线无需引入该模型。
- 梯度累积只在 batch 数达到 4 的倍数时 step，尾批梯度可能跨 epoch 残留；新训练循环需要处理尾批并正确归一化。
- Logger 保存 tensor loss、构造时写默认 CSV；新入口应写到独立输出目录，记录 detached 标量和来源配置，避免实例化模型就覆盖日志。
- `test_diffusion_without_sample.py` 按两个返回值解包 `p_sample_loop`，当前采样函数只返回一个 tensor；不能当作已验收测试复用。
- 不直接照搬包含 nuScenes、PyG、LLM/RAG 等依赖的完整 environment.yml。直接轨迹路线优先只使用所需 PyTorch/NumPy 依赖；实际服务器 torch/CUDA 能力仍需检测。

## 7. 接回现有阶段二/三的边界

预测 x/y 使用当前数据的 `position_reference`。本次真实样本标记为前保险杠中心；反归一化和世界坐标恢复后不能直接改标签成车身中心。根据真实 anchor、dt 和明确的低速规则派生速度/yaw，再按真实车长处理参考点转换；派生方法和任何插值/平滑都要记录，不能视为天然满足物理约束。

可以复用 `scenario_risk.CandidateTrajectory` 和规则评估接口，但需要新的扩散候选适配/导出层。`run_scene_search.export_candidate` 当前硬编码 `generated_target_history=True`，要求参数搜索专用的 `parameter_frame/generation_settings`，并写死 world/center。直接轨迹扩散保留实测历史，不应该照搬这个导出函数或伪造六个 CEM 参数。

第一版目标是普通生成，导出的 `simulation_verified=false`。先报告采样轨迹的运动/道路有效率；RSSM 的显式 NPC 反事实条件能力限制仍然存在。LLM/RAG guidance、风险梯度、自动回灌和闭环执行属于之后阶段，不是这次接入的先决依赖。

## 8. 拟新增文件与实施顺序

以下路径是**接入设计，尚未创建这些训练文件**；现有 RSSM、Dreamer 入口和上传参考代码保留。

| 拟新增文件 | 责任 |
| --- | --- |
| `embodied/envs/trajectory_dataset.py` | 固定索引、来源验证、缓存、目标选择、局部坐标/normalizer 适配 |
| `embodied/envs/trajectory_denoiser.py` | 从上传 TemporalUnet/ResidualTemporalBlock 改造的轻量时序去噪器与 mask-aware 条件编码器 |
| `embodied/envs/trajectory_diffusion.py` | 一致的扩散日程、噪声预测 loss、DDPM/DDIM，无风险引导 |
| `embodied/envs/train_trajectory_diffusion.py` | 独立训练/验证入口、seed、优化、可选 EMA、checkpoint 元数据 |
| `embodied/envs/sample_trajectories.py` | 采样、反归一化/世界变换、评测与可视化产物 |
| `embodied/envs/diffusion_candidate_adapter.py` | 实测历史保留、参考点转换、运动学派生和明确来源的候选输出 |
| `tests/test_trajectory_diffusion*.py` | 数据/mask/无未来条件、公式/shape、反向/保存加载/采样测试 |
| `docs/long_tail_stage4.md` 与配置 JSON | 依赖、参数、服务器命令、验证证据和限制 |

建议按三个可验收步骤实施：

1. **数据适配**：从 train 读取小 batch，核对 11/30/0.2、目标/邻车 mask、原 normalizer_id、世界坐标往返。此时不训练模型，不重跑全池划分/归一化。
2. **最小直接轨迹扩散**：复用时序结构，修正扩散索引与形状；10–20 个 synthetic 小样本完成前向、反向、优化、checkpoint 往返及无引导采样；再用少量真实 train 窗口验证能学习标签。PyTorch 前后向未通过前不开始完整训练。
3. **完整训练与离线验证**：仅 train 优化、val 选择 checkpoint；最终 test 保留到配置确定后。报告标准与 best-of-K ADE/FDE、多样性、采样时间、连续性/动力学/道路有效率及地图缺失率；比较仅用历史信息的恒速基线。最后接候选导出，保留真实来源和 `simulation_verified=false`。

checkpoint 必须保存模型配置、schema、dt/H/F、状态/位置参考约定、normalizer 参数与 ID、固定划分及窗口索引 hash、seed 和优化状态。所有新入口独立启用，关闭时现有 Dreamer/RSSM 调用链不变。
