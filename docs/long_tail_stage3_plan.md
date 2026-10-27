# Stage 3 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Continue inline, with TDD and a final independent review.

**Goal:** 在现有 SUMO 数据上建立相同预算的 Random/CEM 参数搜索基线，输出可审查的候选轨迹和场景 JSON。

**Architecture:** 车辆中心坐标的确定性纵向平滑加速度、横向五次多项式生成一辆关键 NPC；其他车辆及自车沿实测路径。道路和动力学约束先过滤，阶段 2 规则风险排序。原 RSSM 不支持 NPC 条件，明确不用于候选反事实评分。

**Tech Stack:** Python >=3.8、NumPy、stdlib XML/JSON。无新增安装、无训练修改、无 Diffusion。

**Spec:** 数据闭环任务书.md 阶段 3；本文以下约定。

## Global Constraints / Design

- SceneParameterSpec 包括 initial_gap、relative_speed、trigger_time、longitudinal_accel、target_offset、lane_change_duration；标签 cut_in / keep_lane。场景起点允许改变 initial conditions，不把它拼接成原实测 NPC 的连续历史；明确生成相容的合成历史。
- 生成/评分统一中心位置，原前保险杠快照先转换。未来 ego/background 为固定观测响应，不宣称闭环结果。
- 纵向加速度由五次 smoothstep 连续过渡，横向位置用一般五次多项式，起末 p/v/a 连续。检查速度/加速度/jerk/yaw-rate、初始 ego 重叠、目标 NPC 与其他 NPC 冲突；未来 ego 接触保留为风险事件。
- SUMO cfg 显式传入，解析 net-file/route-files，提取车型 maxSpeed/accel/decel；横向/jerk/yaw-rate 等无原车型配置的约束放独立 JSON profile 并标明来源。缺配置不猜真实地图。
- RoadEnvelope 使用车道中心线 buffer、junction polygon 的采样可行驶区检测；车体格点间距和时间子采样显式输出，不冒充连续精确 roadmesh 或路线连通性验证。拒绝道路边界外的候选。
- Synthetic demo 使用显式双车道矩形，仅证明算法可跑。真实输入必须提供对应 SUMO cfg，导出地图来源/hash；不能默默将实测 ego 路径当可行驶区。
- 两种搜索共用 SceneEvaluator、参数上下限、约束、总 evaluate 次数和 seed。CEM 混合 categorical 标签和 bounded Gaussian 连续参数，精英比例/轮数/平滑/方差下限可配。失败候选计入预算，不隐藏重试。
- 逐候选 objective、风险、失败类型、重复惩罚，Top-K 仅选离线约束通过且按轨迹量化去重的候选；CEM 不保证单次优于 Random。耗时单列，不纳入复现哈希。
- 导出 JSON 含初态、时间戳、所有 NPC/self 计划、坐标和单位、模型能力、源场景、参数、hash、seed、离线约束、simulation_verified=false。是后续执行器的数据契约，非原生 CARLA/SUMO 配置，不能直接视为已跑通。
- 不修改原训练、RSSM、阶段 1/2 数据格式。当前无服务器 checkpoint 验收结果也可继续规则基线，模型验证待服务器执行。

## Review Focus

1. 新 initial_gap 与旧 history 冲突：合成 target history 和 anchor/future p/v/a 连续测试。
2. 五次轨迹触发/end、低速和负速度：手算端点/导数测试；无法停住的强制动按无效计入预算。
3. OBB 中心/地图坐标混淆：旋转场景、前保险杠转换、离路和背景碰撞测试。
4. CEM 预算余数/全无效/离散标签塌缩：精确计数、同 seed、方差下限和显式空 Top-K。
5. 合成地图/真实候选被写成真实验证：CLI 输出来源、hash、simulation_verified=false 和不支持的 RSSM 条件。

## Tasks

### Task 1: 运动学与道路
Files: `embodied/envs/scene_road.py`, `scene_parameters.py`, `docs/stage3_search_config.json`, `tests/test_scene_generation.py`。
Interfaces: RoadEnvelope.contains/footprint_valid；DynamicsLimits；SceneParameterSpec.from_window；CandidateScenario；SceneGenerator.generate(parameters,behavior)。
- [x] RED：五次端点、触发连续、间距/速度、原数据不变、SUMO cfg 相对路径及车型、离路、合成历史。
- [x] 实现 NumPy 轨迹、显式道路 envelope 和生成验证数据。
- [x] GREEN：运动学及 XML tests。

### Task 2: 公用评估和预算搜索
Files: `embodied/envs/scene_search.py`, `tests/test_scene_search.py`。
Interfaces: SceneEvaluator.evaluate(candidate)；RandomSearch/CEMSearch.run(generator,evaluate,budget,seed,top_k)；SearchResult。
- [x] RED：未来 ego 风险允许、背景碰撞/越界/动力学不允许、同预算 seed、CEM 余数和全无效、去重。
- [x] 实现共用风险函数、objective/Top-K、局部 RNG、CEM categorical+continuous updates。
- [x] GREEN：无修改全局 RNG，预算不因失败被扩充。

### Task 3: CLI / 真实输入 / 文档
Files: `embodied/envs/run_scene_search.py`, `tests/test_scene_search_cli.py`, `docs/long_tail_stage3.md`, README.md。
- [x] RED：synthetic demo 输出、same budget/config、JSON roundtrip、缺地图拒绝、全无效非零状态。
- [x] 实现 --demo 或 --input --sumo-cfg，window/target/profile/search 配置，逐候选 JSONL、summary、Top-K 场景 JSON/NPZ。
- [x] 运行 synthetic 和用户实际 NPZ（显式本地 scenario_057 cfg），记录实际有效率/风险/失败，不能编造 Top-K 数量。
- [x] 全量 unittest、独立复核、更新 ledger 和服务器命令。

## Ledger

- 2026-10-20 用户明确要求第三阶段修改；范围依据任务书，继续内联实施，不重复询问同范围授权。
- 当前无 Git，不初始化或复制完整数据目录；新模块独立、README 只补入口。
- 找到本地对应 `embodied/generated_scenarios_merge_7/scenario_057_easy_veh011.sumocfg`、net 和 routes。源 NPZ 的 map_features 虽为空，但命令可显式绑定此 cfg，来源验证范围需注明。
- Pre-flight: Task 1 给 Task 2 的候选统一中心坐标、含原 window 条件和 dense validation samples；Task 2 给 CLI 的结果可 JSON 序列化，None 表示不可用。
- Task 1/2/3 complete: 仅新增独立模块/测试/说明和 README 链接，无改动原训练、阶段 1/2 模块及真实 NPZ。
- Review fixes (RED→GREEN): 短 ramp 可能漏过 jerk93.75峰值→加入中点/终点、横向导数实根；路网全局最高限速和错误相邻匹配→包含中心的同向车道，交叠取低；仅 junction 内允许最近线段 fallback。NPZ 保存 npc_ids/ego_id。
- Self-review fixes (RED→GREEN): exact 原始帧不受下一缺失帧 yaw NaN*0 污染；candidate ID 包含生成轨迹；不吞评估函数 ValueError；生成失败遵守配置 invalid_penalty。
- Final tests: 115 项通过，8.512秒，阶段3新增28项。synthetic budget32 seed0 Random20/CEM26有效，各Top5，退出0。
- Final actual validation: window70/anchor16.2s/obs_004/budget256/seed0，Random0有效，CEM3有效且均risk0.9，退出2（至少一方法无候选），仍导出完整报告与CEM3场景。最终仅以 docs/stage3_demo_verified 和 docs/stage3_real_verified 为验收输出。
- Investigation: 初始window0/budget32及window50/budget64均无有效候选，window70/budget64曾CEM3/Random0，window82/budget64全无效；这些是定位过程，不是额外补进最终相同预算的候选。
- Remaining limits: 固定局部 heading 参数化不保证所有弯路窗口可行；map历史hash未知；采样道路/OBB非精确连续验证；无RSSM NPC条件能力，无SUMO/CARLA执行证明。未开始阶段4。
- Final artifacts: 已验证 JSON/NPZ actor ID、shape、逐方法精确预算、原 NPZ SHA256 不变；早期5次项目内诊断 summary 汇总至 docs/stage3_search_investigation.json，删除仅本轮创建的中间报告目录，保留两个 verified 目录及最终 Top-K。
