# 阶段 3：Random / CEM 场景参数搜索

本阶段新增独立低成本生成基线：参数化关键 NPC → 确定性轨迹 → 道路/动力学/OBB 约束 →
阶段 2 规则风险 → 去重 Top-K。保持原训练入口、RSSM、数据采集和阶段 1/2 默认行为。
所有输出都是离线候选，`simulation_verified=false`，尚未在 SUMO/CARLA 中验证执行。

## 运行和验收

在带 NumPy 的环境、项目根目录运行。当前 Windows 可把 python 换为 `D:/anaconda3/python.exe`。
输出目录须为空或不存在，以免新结果混入旧候选。

```bash
python embodied/envs/run_scene_search.py --demo --output stage3_runs/demo_seed0 --budget 32 --seed 0 --top-k 5
```

真实数据需要显式绑定对应 SUMO cfg。本仓库存在该样本同名的 cfg、路网和 routes：

```bash
python embodied/envs/run_scene_search.py \
  --input docs/6c873565d1ba7e753c679303.npz \
  --sumo-cfg embodied/generated_scenarios_merge_7/scenario_057_easy_veh011.sumocfg \
  --window-index 70 --budget 256 --seed 0 --top-k 5 \
  --output stage3_runs/real_window70_seed0
```

`--window-index 70` 对应当前 H=11、F=30、原生 dt=0.2 下的 anchor=16.2 秒，
自动选择完整可观测且邻近的 NPC；本样本为 obs_004。可用 `--target-id` 显式指定其他完整轨迹。
没有地图时拒绝真实输入，不会把实测 ego 轨迹外推成“已知道路”。
来源 episode 未记录 map hash，因此显式 cfg 绑定及当前坐标匹配不能证明与录制时的路网逐字节相同；
输出 `source_map_identity_verified_against_recorded_hash=false`，同时保存当前 cfg/net/routes hash。

退出码：0 为两种方法均导出至少一个候选；2 为搜索完成，但至少一种方法没有可导出候选；
1 为输入/配置错误或评估异常。真实样本本次返回 2：Random 无候选，CEM 有 3 个，报告完整保留。

## 参数和运动学

`SceneParameterSpec` 包括：initial_gap、relative_speed、trigger_time、longitudinal_accel、
target_offset、lane_change_duration；离散标签为 cut_in / keep_lane。
初始间距为 ego anchor 前方的车身中心纵向距离，相对速度沿该 anchor heading 定义。
target_offset 对 cut_in 表示 ego 参考线附近的终点横向偏移；对 keep_lane 表示原横向位置的小偏移。
这是同向 merge/cut-in 的局部参数化，不是通用任意路网驾驶规划器。

配置来源分开保存：

- 从 routes 的目标 vehicle/vType 读取 maxSpeed、accel、decel，并读取 passenger 车道 geometry/speed。
- 横向加速度、jerk、yaw-rate、加速度 ramp、采样分辨率和惩罚在
  [stage3_search_config.json](stage3_search_config.json)；它们是显式搜索约束，不能当作原车型已标定参数。
- 间距/速度/时间参数范围按车型限制和窗口时长构建，保留 target_offset ±0.3 米的局部搜索。
  笛卡尔积内不保证每组参数都可行，失败也消耗预算并计入失败类型。

纵向加速度按五次 smoothstep 从 0 平滑过渡至指定加速度，并解析积分得到位置/速度。
横向位置用一般五次多项式匹配起末位置、速度、加速度。触发前和结束后分段保持 p/v/a 连续。
检查包含常规时间网格、纵向 ramp 中点/终点及横向多项式各阶导数的实根，避免短 ramp 漏过 jerk 峰值。
组合 yaw-rate、道路和 OBB 仍为采样检查，不声明连续时间精确证明。

改变 initial_gap / relative_speed 会改变场景初态，因此生成对应 target 合成历史，标记
`generated_target_history=true`。没有把修改后的轨迹直接伪装成原 NPC 实测历史的连续延伸。
ego 与其他 NPC 使用原窗口观测路径，目标 NPC 之外的背景不会重新响应。

所有生成和几何检查使用车身中心位置。原 SUMO 前保险杠坐标先按每车真实半长和 yaw 转换；
导出同时声明 `position_reference=center`，执行器不能直接把它当成 TraCI 前保险杠坐标。
速度/位置/角度单位为 m/s、m、rad，不能直接当成 CARLA 的坐标和度制旋转。

## 约束与风险

候选拒绝项包括反向运动、车型速度/纵向加速度超限、横向加速度/jerk/yaw-rate 超限、
初始 ego 重叠、目标 NPC 与背景 NPC 新增重叠，以及离开采样可行驶范围。
未来 NPC-ego 几何接触保留为风险事件，不作为动力学无效的通用过滤条件。

道路检测使用 passenger 车道中心线 buffer 和 junction polygon；对完整车身格点做检测，
默认空间间距 0.5 米、时间网格不大于 0.1 秒。它不是精确 roadmesh，也未验证 route/lane 连通性。
限速优先采用包含车辆中心且同向的车道，重叠时取较低限制；仅 junction polygon 内无包含车道时
退到最近同向线段，避免用无关快车道覆盖慢车道。整车型 maxSpeed 约束同时保留。

通过上述约束后调用同一个阶段 2 `RiskEvaluator.evaluate_risk(context,candidate)`。
风险项和权重沿用阶段 2。源 NPC mask 不能用来隐藏已知车辆；缺失数据规则保持不变。
有效率指本模块的离线约束通过率，不是真实仿真执行成功率。

原 RSSM 不支持显式 NPC plan 或绝对多车世界 decoder，因此候选不使用其反事实 cost 排序。
报告明确记录 `npc_plan_not_supported`，不把原窗口一次 model rollout 套给所有新候选。
模型原 checkpoint 的阶段 2 服务器验证仍需单独完成。

## 搜索公平性与复现

RandomSearch 与 CEMSearch 使用完全相同的 generator/spec、SceneEvaluator、约束和总预算。
`--budget` 是每种方法的生成/评估尝试数；拒绝项、生成失败和重复样本均消耗预算，不额外补采样。
CEM 支持 categorical 标签与各标签的连续 Gaussian 分布，样本裁剪在相同参数边界内。
精英比例、轮数、平滑及方差下限可通过参数调整：

```bash
python embodied/envs/run_scene_search.py --demo --output stage3_runs/demo_cem_config \
  --budget 64 --cem-iterations 4 --elite-fraction 0.25 --smoothing 0.7 --min-std-fraction 0.05
```

有效候选用于 CEM 更新；整轮无有效候选时保持分布，继续消耗剩余已分配预算。
两方法各自保留同一定义的轨迹量化去重 archive，默认 0.5 米分辨率。
重复候选的 objective 有独立惩罚；Top-K 对有效轨迹去重后按原风险排序，不把无效场景塞满 K。
该量化去重不能保证捕获所有近似重复事件；报告保留指纹和重复计数。
固定输入、参数、环境与 seed 可复现参数、objective、候选 ID；耗时单列，不要求重复执行一致。
单次 CEM 结果不能作为统计意义上优于 Random 的结论。

## 输出及接口

`summary.json` 包含输入/map/profile hash、窗口/target、参数边界、动力学限制、两方法预算、
有效率、Top-K 风险、耗时、失败类型及不支持能力。
每种方法目录中有逐候选 `trials.jsonl`，以及 `top_001.json/.npz` 等。
JSON 包含所有 actor 的初态/轨迹/mask、中心坐标语义、生成设置、参数、风险和完整搜索配置。
NPZ 不使用 pickle，显式保存 npc_ids、ego_id、states、npc_states、mask 和 metadata_json。

```python
generator = SceneGenerator(window, target_index, spec)
evaluate = SceneEvaluator(limits, road).evaluate
random_result = RandomSearch().run(generator, evaluate, budget=64, seed=0, top_k=5)
cem_result = CEMSearch(iterations=4).run(generator, evaluate, budget=64, seed=0, top_k=5)
```

场景 JSON 是后续 SUMO/CARLA 执行器可读的数据描述，不是原生 sumocfg 或 CARLA ScenarioRunner 配置。
执行器仍需初始化场景、绑定路网、将中心点转为模拟器坐标参考并提供闭环 ego controller。
不能仅凭 JSON 文件认定真实碰撞、真实 near-miss 或训练闭环改善。

## 本轮真实验证结果（2026-10-20）

最终合成报告：[stage3_demo_verified/summary.json](stage3_demo_verified/summary.json)。
最终真实报告：[stage3_real_verified/summary.json](stage3_real_verified/summary.json)。
定位期间的早期结果汇总在 [stage3_search_investigation.json](stage3_search_investigation.json)，
其中数据属于最终复核修正前的诊断运行，不能替代上述最终验收报告。

| 输入 | 方法 | 总预算 | 离线有效 | 去重 Top-K | 最高规则风险 | 搜索耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 显式双车道 demo | Random | 32 | 20 | 5 | 0.14775 | 0.676 s |
| 显式双车道 demo | CEM | 32 | 26 | 5 | 0.20166 | 0.928 s |
| 实际 NPZ，窗口70 | Random | 256 | 0 | 0 | 不可用 | 10.899 s |
| 实际 NPZ，窗口70 | CEM | 256 | 3 | 3 | 0.9 | 6.565 s |

真实 CEM 的有效率为 1.171875%，且 3 个候选的规则分均为 0.9；这是风险项饱和，并不代表严重程度完全。
失败类型会重叠计数。真实 Random 主要包含反向 172、纵向加速度超限 176、离路 65、
yaw-rate 超限 142；CEM 也存在大量约束拒绝。没有调整车型加速度或地图边界以制造通过结果。
早期窗口0和50未找到有效候选，说明固定 anchor heading 的局部生成受转弯道路、
大横向间距和 SlowCar1 的低加速度限制影响；不保证任意窗口能产生候选。
进一步改善应考虑道路参考线参数化/更合适的条件与边界，而不能把约束失败当成长尾成功。

最终全量命令 `D:/anaconda3/python.exe -m unittest discover -s tests`：115 项通过，8.512 秒；
其中本阶段新增 28 项，原有阶段 1/2 回归 87 项。原 NPZ 不改写。
独立复核提出的短 ramp jerk、逐车道限速和 NPZ ID 问题均已补回归修复；额外覆盖缺失帧插值、
轨迹 ID、错误不被吞掉、全无效退出码和精确预算。

下一步优先将这 3 个 CEM 候选送入 SUMO 执行验证，检查位置/速度跟踪误差、初始化、
实际 ego 响应和真实事件，再决定数据回灌。当前没有自动回灌，也没有开始阶段 4 Diffusion。
