# 阶段 2：风险评分与 RSSM 推理适配 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** 使用真实 NPZ 产生可追溯的规则风险报告，并提供原 checkpoint 的只读 RSSM 推理入口。

**Architecture:** 新模块消费阶段 1 ScenarioWindow 和对应 ScenarioEpisode。规则评分仅处理米/秒/弧度车辆轨迹；RSSM 保留原 31 维 observation、原动作与原模型模块，不把解码 observation 解释成绝对多车位置。

**Tech Stack:** Python >=3.8，NumPy；模型入口延迟导入训练环境已有 JAX/Optax/TFP/ruamel。

**Spec:** 数据闭环任务书.md 阶段 2，以及下面的实现约定。

## 实现约定 / Global Constraints

- 原训练、配置、网络、checkpoint 写入方式不变。全部功能是独立命令或适配器，默认不会运行。
- SUMO front_bumper_center 转换成车身中心后计算 OBB；未知位置参考拒绝几何评分。坐标系必须为 world 或固定锚点 ego-centric。
- TTC 是匀速、固定朝向 OBB 接触的外推时间，限定 horizon；平行不同车道、远离、相对静止不返回伪 TTC。不是实际撞击标签。
- score 为归一化 TTC、车体间距、几何重叠、低进度项的加权均值；阈值/权重可配置。低进度为车身中心路程速度代理，要求至少 5 秒，没有道路终点进度语义。
- candidate 只能替换同一窗口 NPC 轨迹，固定实测自车响应；必须核对 dt/坐标系/ID/mask。不能称反事实闭环。
- RSSM 输入原始 raw_states(T,31)，只接受原 episode 原生均匀时间网格和已知动作，不插值归一化邻车槽位。窗口冷启动 posterior；有 terminal 对齐风险的最后 raw_state 不做误差真值。
- RSSM 显式 npc_plan 或 ego_policy 暂不支持。支持实测动作的 free prior rollout 和 future observation teacher forcing；预测仅输出 latent/原 decoder/reward/cost/cont。
- 碰撞概率及可靠 uncertainty 始终 None；原 cost/risk 头未经碰撞概率校准。ADE/FDE 不适用。已知记录 cost 在有正样本时可报告阈值 risk recall，但不叫真实碰撞召回。
- checkpoint 通过原 embodied.Checkpoint.load(keys=['agent']) 读取；禁用新参数创建，配置不兼容必须报错，禁止随机补权重。读取同目录 config.yaml，原配置名只能作为显式备选。
- 本机 checkpoint 不存在且缺 JAX；不得宣称真实 RSSM 验证通过。提供服务器命令、失败退出码及状态报告。

## Review Focus

1. 前保险杠坐标及旋转车辆导致中心错误：测试 unequal length 和旋转不变性。
2. 侧向相邻、交叉、低速 TTC 错误：手算 footprint 冲突与无冲突案例。
3. 缺失 NPC/NaN 被当安全：测试 mask、维度和无覆盖，输出不可评分标志。
4. 窗口插值或动作错一帧：测试原生 dt、incoming action 历史、interval-start future actions。
5. 真实模型不可用被 fake 成成功：CLI 缺失 checkpoint 非零退出，模型版本/源/不能评价指标明确记录。

## Task 1: 规则风险

**Files:** embodied/envs/scenario_risk.py；tests/test_scenario_risk.py。

**Interfaces:** RiskConfig、RiskReport、CandidateTrajectory；RiskEvaluator.evaluate_risk(window,candidate=None)。

- [x] 写测试：OBB 间距 6 m、相向碰撞 TTC 3 s、侧邻无 TTC、重叠 TTC 0、缺失 mask/未知参考、固定自车候选、短窗口低进度不可用、阈值权重合法。
- [x] 运行 unittest，确认缺失模块导致失败。
- [x] 实现 SAT 平移交叠区间和边段距离；标注 measured_trajectory_rule / fixed_ego_candidate_rule。
- [x] 运行规则测试，确认解析值、旋转不变性、JSON 无 NaN。

## Task 2: 模型边界及原 checkpoint 接口

**Files:** embodied/envs/world_model_adapter.py；embodied/envs/rssm_inference.py；tests/test_world_model_adapter.py。

**Interfaces:** WorldModelAdapter(episode,backend=None,seed=0,model_dt=None)、encode_history、predict_future、evaluate_risk；NinjaxRSSMBackend.from_checkpoint；prediction_metrics。

- [x] 写测试：无模型、NPC 条件拒绝、原网格匹配、动作帧对齐、无 decoder、误差 1/5/10/30 与不适用指标、终帧对齐、真实 checkpoint 格式选择 agent 读入（CPU 替身仅验证格式边界）。
- [x] 确认 RED，然后新增只读 backend；复用 Agent.inner、原 preprocess、rssm.observe/imagine 和 decoder/head。
- [x] 运行适配器 CPU 测试；真实 JAX/checkpoint 测试留服务器执行并明确状态。

## Task 3: 可执行验证报告

**Files:** embodied/envs/evaluate_scenario_risk.py；tests/test_stage2_cli.py；docs/long_tail_stage2.md；README.md 链接。

**Interfaces:** 独立脚本 --input --output [--checkpoint --model-config] [--dt --history --future --stride --seed --max-model-windows]。

- [x] CLI 测试：真实过程写逐窗口 JSONL+summary，缺失 checkpoint 返回非零且保留规则报告，重复 seed 规则结果一致。
- [x] 实现命令与 provenance（输入 SHA256、配置、原生 dt、模型路径、运行环境及缺失能力）。
- [x] 用用户真实 NPZ 跑规则报告；查看风险与旧 TTC/碰撞标签差异。
- [x] 全部 unittest discover；保存真实结果及服务器命令。

## Ledger

- Ruling: 已有用户“然后下一步”授权当前阶段，继续内联执行；不再请求同一范围授权。用户提供 checkpoint 位于其他服务器，当前无服务器连接能力。
- Ruling: 当前目录没有 Git，保留独立新增文件和原训练路径；不初始化 Git、不提交、不创建 worktree。
- Pre-flight: Task 1 → Task 2 传递 RiskReport；Task 1/2 → Task 3 传递评估字典和 JSON 可序列化数据。统一使用 None 表示不可用，不能输出 Infinity/NaN。
- Task 1 Ruling: unequal-length 保险杠测试的手算期望从 5 改为 4 米；车身区间 [-4,0]、[4,10]，原期望遗漏两车半长之和，几何实现无需修改。
- Task 1: 14 规则测试已通过；初次缺失模块 RED，覆盖率及 candidate mask 漏洞另有 RED→GREEN 回归。最终新增逐 ID 内部缺口回归后复验。
- Task 2: complete for CPU interface — 14 测试通过；非有限 posterior/截断或空预测必须 supported=False。真实 checkpoint 位于服务器，本机不能验证 JAX 前向。
- Task 3: CLI 3 测试通过；真实数据原生 dt=0.2 产生 83 窗口，首次规则检查全部有效。更新覆盖率限制后重新运行最终验证。
- Ruling: 模型可用性与质量分开；没有校准头不报告碰撞概率，缺少 cost 正样本不报告 recall；禁止 NaN 输出被 CLI 计为推理成功。
- Ruling: 复核发现仅 anchor 有 NPC 的窗口会被低风险排序，改为每个评分时刻至少有一个可用 NPC（N=0 单独处理），报告 mask 覆盖率和缺失原因未验证；候选不能隐藏原未来有效 NPC。部分 NPC 离场/缺口原因需后续来源证据，不能猜。
- Ruling: 多 NPC 场景仍可能用其他车辆遮盖单 ID 内部缺口；新增检测逐 ID 首次至末次观测之间的缺口，有缺口不评分。首尾缺席保留为覆盖率限制，不要求 union 中所有 ID 全程存在。
- Review: 独立 reviewer 已返回缺失 NPC 评分问题，并核对原模型调用、动作索引与 checkpoint 容器未发现新增确定缺陷；reviewer 最终因模型额度中断，不记录为整批独立复核通过。已返回的问题按 RED→GREEN 修复，继续本地完整验收。
- Final verification 2026-10-10: 全量 `D:/anaconda3/python.exe -m unittest discover -s tests` 87/87，4.357 秒。规则 15、适配器 14、CLI 3、原有 55。真实规则命令退出 0、83/83；服务器相对路径本机缺失分支退出 2、模型状态为 false。真实 RSSM 验收仍待服务器执行。
- Task 1/2/3 local deliverables: complete。新增文件及各自职责已在本计划和阶段 2 说明中列出；README 增加阶段 2 入口。阶段 3 未开始。
- Session continuity: 用户要求暂停后，本轮收到“继续修改”恢复，沿用此前新增文件，没有重做阶段 1。下次优先查看本 ledger、docs/long_tail_stage2.md 和服务器 summary/model_windows 输出，勿把 CPUSequenceBackend 测试当作真实模型。
