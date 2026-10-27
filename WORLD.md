# World Engine 代码与论文解读 README

> 关注重点：危险场景的识别、危险场景的扩展，以及代码与论文中对应的实现关系。

## 1. 项目概览

World Engine 是一个面向自动驾驶后训练（post-training）的世界模型框架。它的核心目标不是继续被动收集更多普通驾驶数据，而是主动发现自动驾驶模型容易失败的长尾危险场景，并将这些场景重建、扩展、仿真，再用于强化学习式后训练，从而提升模型在安全关键场景下的表现。

论文中的整体闭环可以概括为：

```text
真实驾驶日志
    ↓
预训练自动驾驶模型
    ↓
失败/危险场景挖掘
    ↓
3DGS 场景重建
    ↓
行为世界模型生成多样化交通交互
    ↓
闭环仿真 rollout
    ↓
强化学习后训练
    ↓
安全性评估与部署
```

代码仓库主要由两个部分组成：

```text
WorldEngine-main/
├── projects/
│   ├── AlgEngine/   # 算法训练、评估、危险样本挖掘、RLFT 配置
│   └── SimEngine/   # 闭环仿真、场景管理、渲染、agent/metric/reward 管理
├── docs/            # 安装、数据组织、配置和使用说明
└── scripts/         # 闭环测试脚本
```

---

## 2. 论文中的危险场景识别逻辑

### 2.1 为什么要识别危险场景

论文指出，自动驾驶系统真正的安全边界往往由少量长尾事件决定，例如：

- 突然横穿的行人
- 激进 cut-in
- 复杂路口博弈
- 车辆侧向进入
- 停车车辆遮挡
- 接近碰撞的 near-miss 场景
- 出道路、违规驶入不可行驶区域

普通日志中这类事件非常稀疏，单纯扩大预训练数据会出现边际收益递减。因此 World Engine 先训练一个 base driving agent，再用它去真实日志中“暴露自己的失败点”。

### 2.2 危险场景如何被识别

论文中的识别方法是：

1. 先用大规模真实日志通过 imitation learning 训练 base agent；
2. 将每个真实日志场景输入 base agent；
3. 得到 base agent 的规划轨迹；
4. 在轻量仿真器中让 ego 轨迹与真实 logged traffic、HD map 进行非反应式 rollout；
5. 如果出现碰撞或驶出道路，则标记为 safety-critical / long-tail event。

也就是说，危险场景不是人工写规则枚举出来的，而是由当前模型的失败行为自动挖掘出来的。

可以理解为：

```text
危险场景 = 当前模型能力边界附近的失败样本
```

这种做法有两个优点：

- 场景来自真实日志，因此具备真实交通上下文；
- 场景由 base agent 失败定义，因此与后训练目标高度相关。

### 2.3 论文中的危险类型

论文和实验中主要关注以下危险或困难情况：

| 类型 | 含义 |
|---|---|
| collision | ego 与动态/静态障碍物发生碰撞 |
| off-road / drivable-area violation | ego 离开合法可行驶区域 |
| low ego progress | ego 虽然未碰撞，但进度明显不足，可能过度保守或卡死 |
| cut-in | 相邻车辆突然切入 ego 车道 |
| side-entry | 侧向车辆进入 ego 路径 |
| pull-out | 前方或侧方车辆突然驶出 |
| parked vehicle | 停车车辆造成遮挡或绕行风险 |
| intersection interaction | 路口多车/行人交互风险 |

---

## 3. 代码中的危险场景识别实现

危险场景挖掘主要位于：

```text
projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py
```

该脚本根据模型评估结果和 PDM/PDMS 相关指标采样 rare case。核心函数包括：

```python
sample_collision()
sample_off_road()
sample_low_ego_progress()
```

对应关系如下：

| 代码函数 | 识别目标 | 典型指标 |
|---|---|---|
| `sample_collision()` | 碰撞危险场景 | `no_at_fault_collisions == 0` |
| `sample_off_road()` | 出道路危险场景 | `drivable_area_compliance == 0` |
| `sample_low_ego_progress()` | 低进度困难场景 | `ego_progress` 位于最低分位 |

输出通常是 YAML split 文件，例如：

```text
projects/AlgEngine/configs/navsim_splits/navtrain_split/
├── navtrain_50pct_collision.yaml
├── navtrain_50pct_off_road.yaml
└── navtrain_50pct_ep_1pct.yaml
```

这些 YAML 文件后续会被训练配置引用，用作后训练数据来源。

---

## 4. 论文中的危险场景扩展逻辑

危险场景扩展是 World Engine 的关键。它不是只复用原始失败日志，而是把每个真实危险场景变成一个可交互、可变化的训练环境。

整体扩展分为两层：

```text
真实危险日志
    ↓
3DGS 场景重建：得到可渲染、可交互的 photorealistic world  （这个我通过结构化数据实现）
    ↓
行为世界模型：生成周围交通参与者的多样化未来轨迹
    ↓
闭环 rollout：ego 与周围 agent 真实交互
    ↓
生成更多危险变体
```

### 4.1 3DGS 场景重建

论文中的 Simulation Engine 使用 3D Gaussian Splatting 重建驾驶场景。它将场景拆成：

| 部分 | 内容 | 作用 |
|---|---|---|
| static background | 道路、建筑、植被、路面结构 | 保持真实环境外观 |
| dynamic objects | 车辆、行人、骑行者等 | 可独立移动、删除、改变轨迹 |

这种场景图式分解让系统可以在保持真实视觉质量的同时，改变动态交通参与者的位置和行为。

核心能力包括：

- 从新的 ego pose 渲染相机观测；
- 移动或重排交通参与者；
- 支持闭环仿真中 ego 轨迹偏离原始日志；
- 为强化学习生成大量 rollout。

### 4.2 行为世界模型扩展

Simulation Engine 解决“看起来真实”的问题，Behaviour World Model 解决“交通行为合理且多样”的问题。

论文中行为世界模型是基于 diffusion / DiT 的多智能体轨迹生成模型。输入包括：

- 地图拓扑；
- 历史 agent 状态；
- ego 当前动作或规划；
- 可选 goal / waypoint 约束。

输出是周围交通参与者的未来轨迹。

它支持两类控制方式：

| 控制方式 | 作用 |
|---|---|
| goal conditioning | 指定某个 agent 的目标点，例如让车切入 ego 前方 |
| optimization guidance | 在采样过程中引导生成更接近危险边界的行为，例如接近碰撞阈值但仍保持物理合理 |

---

## 5. 补充材料中的两种危险场景扩展策略

论文补充材料给出了两个具体策略：Scenario Copy 和 Intent Attack。

### 5.1 Scenario Copy

Scenario Copy 的目标是复制原有危险场景的交互结构，同时引入轻微扰动，避免完全重复。

流程：

1. 在原始危险场景中找到 ego 附近的相关 agents；
2. 对每个 agent 的原始终点加入随机扰动；
3. 扰动范围通常在 1 米以内；
4. 用行为世界模型根据扰动后的目标点生成完整交互轨迹。

公式形式：

```text
G_new = G_original + δ, ||δ|| < 1m
```

作用：

- 保留原始危险模式；
- 生成同一危险模式下的多样变体；
- 适合扩展 near-miss、cut-in、路口交互等真实长尾场景。

### 5.2 Intent Attack

Intent Attack 的目标是主动制造更强的安全压力。

流程：

1. 从 ego 最近的 K 个周围车辆中随机选一个 adversarial agent；
2. 将该 agent 的目标点强行设置到 ego 目标点附近；
3. 让行为世界模型生成完整场景；
4. 得到更容易与 ego 发生路径冲突的高风险交互。

公式形式：

```text
G_adversary = G_ego + ε, |ε| ≤ Δ
```

其中 Δ 通常为 1–2 米。

作用：

- 主动生成 cut-in、抢道、交叉冲突等危险行为；
- 增加训练集中高价值、低频的安全关键样本；
- 用于 stress-test ego planner 的反应能力。

### 5.3 生成后的有效性检查

生成的危险变体不会直接进入训练集，而是要经过后处理和验证：

| 检查项 | 目的 |
|---|---|
| LQR tracker 优化 | 保证轨迹可执行 |
| initial collision check | 排除初始状态已经不可避免碰撞的无效场景 |
| drivable area check | 保证车辆仍在合法道路区域 |
| physical plausibility check | 保证行为符合基本运动学约束 |

只有通过验证的场景才会加入后训练数据。

---

## 6. 代码中的危险场景扩展和后训练配置

World Engine 的后训练配置主要位于：

```text
projects/AlgEngine/configs/worldengine/
```

重点配置包括：

```text
e2e_vadv2_50pct_rlft_rare_log.py
e2e_vadv2_50pct_rlft_rare_syn_replay.py
e2e_vadv2_50pct_rlft_rare_rollout.py
e2e_vadv2_50pct_rlft_rare_rollout_bwm.py
```

它们对应不同后训练数据源：

| 配置文件 | 含义 |
|---|---|
| `rlft_common_log.py` | 使用普通日志后训练 |
| `rlft_rare_log.py` | 使用真实 rare log 后训练 |
| `rlft_rare_syn_replay.py` | 使用合成 replay 数据后训练 |
| `rlft_rare_rollout.py` | 使用闭环 rollout 后训练 |
| `rlft_rare_rollout_bwm.py` | 使用 Behaviour World Model 增强后的完整 World Engine 后训练 |

其中最完整的是：

```text
projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_rollout_bwm.py
```

它代表论文中的完整 World Engine pipeline：真实危险场景 + 闭环 rollout + 行为世界模型增强。

---

## 7. SimEngine 代码结构说明

SimEngine 负责闭环仿真、场景状态推进、渲染、metric 和 reward 管理。

关键目录如下：

```text
projects/SimEngine/worldengine/
├── engine/          # 仿真引擎主逻辑
├── envs/            # 环境构建
├── manager/         # agent、data、metric、reward、render、scenario 管理器
├── render/          # 渲染模块，包含 MTGS/3DGS 渲染
├── scenario/        # 场景描述与状态解析
├── runner/          # 分布式 rollout / testing 执行入口
├── components/      # agent、map、parameter space 等组件
└── utils/           # 几何、并行、配置等工具
```

重点文件：

| 文件 | 作用 |
|---|---|
| `worldengine/engine/base_engine.py` | 仿真引擎基础逻辑 |
| `worldengine/manager/scenario_manager.py` | 场景加载与状态管理 |
| `worldengine/manager/agent_manager.py` | ego 和周围 agent 管理 |
| `worldengine/manager/render_manager.py` | 渲染管理 |
| `worldengine/manager/metric_manager.py` | 仿真指标统计 |
| `worldengine/manager/dense_reward_manager.py` | dense reward 计算 |
| `worldengine/render/mtgs/mtgs.py` | 3DGS/MTGS 渲染相关实现 |
| `worldengine/runner/run_simulation.py` | rollout / simulation 执行入口 |

SimEngine 的功能可以概括为：

```text
读取场景 → 初始化 ego/agent/map → 接收 planner 动作
→ 推进仿真状态 → 渲染新观测 → 计算 reward/metric
→ 保存 rollout 结果
```

---

## 8. AlgEngine 代码结构说明

AlgEngine 负责模型训练、评估、危险样本采样和后训练配置。

关键目录如下：

```text
projects/AlgEngine/
├── closed_loop/         # 闭环测试对接逻辑
├── configs/             # 训练、评估、后训练配置
├── mmdet3d_plugin/      # 模型、数据集、loss、head、planner 等
└── scripts/             # 训练、测试、rare case sampling 脚本
```

重点文件：

| 文件 | 作用 |
|---|---|
| `scripts/rare_case_sampling_by_pdms.py` | 根据 PDMS 指标挖掘危险场景 |
| `scripts/train.py` | 训练入口 |
| `scripts/test.py` | 测试入口 |
| `closed_loop/sim_test.py` | 闭环仿真测试 |
| `closed_loop/monitor.py` | 闭环监控 |
| `closed_loop/post_processor.py` | 仿真输出后处理 |
| `configs/worldengine/*.py` | World Engine 训练与后训练配置 |
| `mmdet3d_plugin/navformer/dense_heads/traj_scoring_head_RL.py` | RL 风格轨迹评分/规划头 |
| `mmdet3d_plugin/losses/planning_loss.py` | 规划损失 |
| `mmdet3d_plugin/datasets/eval_utils/metric_utils.py` | 指标计算相关工具 |

AlgEngine 的危险场景处理链路可以理解为：

```text
模型评估结果
    ↓
PDMS / collision / DAC / ego_progress 指标
    ↓
rare_case_sampling_by_pdms.py
    ↓
生成 rare split yaml
    ↓
WorldEngine RLFT 配置读取 rare split
    ↓
后训练
```

---

## 9. 强化学习后训练逻辑

论文将后训练表述为带行为正则的强化学习问题：

```text
最大化：
累计 reward - KL(当前策略 || 预训练策略)
```

这样做的目的是：

- 用 reward 引导模型在危险场景中更安全；
- 用 KL 约束避免模型偏离预训练策略太远；
- 防止在提升长尾安全性的同时破坏普通场景能力。

### 9.1 Reward 组成

论文中的 reward 包括硬约束和软目标。

硬约束：

| 约束 | 含义 |
|---|---|
| no-at-fault collision | 不发生 ego 责任碰撞 |
| drivable-area compliance | 不离开合法可行驶区域 |

软目标：

| 目标 | 含义 |
|---|---|
| ego progress | 沿 route 前进 |
| time-to-collision | 保持足够 TTC 安全裕度 |
| comfort | 加速度、横向加速度、转向变化平顺 |

可概括为：

```text
安全优先，其次进度，最后舒适性
```

### 9.2 Hard Experience Mining

并不是所有 rollout 帧都同等重要。World Engine 会优先保留：

- 即将碰撞的帧；
- 已经碰撞或出界的帧；
- 明显偏离人类驾驶的帧；
- 复杂博弈和恢复动作帧。

这些 hard frames 对后训练贡献更大。

---

## 10. 论文实验结论

论文中的主要实验结论：

1. World Engine 后训练比单纯扩大预训练数据更高效；
2. 只用 common logs 后训练对 rare closed-loop 帮助有限，甚至可能退化；
3. 只用 rare logs 可提升 open-loop rare PDMS，但 closed-loop 提升有限；
4. rare rollout 明显优于静态 replay；
5. 加入 Behaviour World Model 后，closed-loop rare success rate 和 PDMS* 最好；
6. 在生产级 ADS 仿真和真实道路测试中也能降低碰撞和危险事件。

论文表格中的关键结果：

| 方法 | Rare closed-loop SR | Rare closed-loop PDMS* |
|---|---:|---:|
| base model | 73.66 | 60.98 |
| rare synthetic replay | 87.20 | 63.22 |
| rare rollout w/o Behaviour WM | 77.96 | 67.33 |
| full World Engine | 88.89 | 70.12 |

这说明：

```text
危险场景扩展 + 交互式闭环 rollout + 行为世界模型
```

比单纯 replay 或普通日志后训练更有效。

---

## 11. 一句话总结

World Engine 的核心不是“多收集数据”，而是：

```text
用 base model 找出自己会失败的真实危险场景，
用 3DGS 和行为世界模型把这些场景扩展成大量可交互危险变体，
再用带安全 reward 和行为正则的 RL 后训练修正模型。
```

从代码角度看，重点关注：

```text
危险场景识别：
projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py

后训练配置：
projects/AlgEngine/configs/worldengine/

闭环仿真：
projects/SimEngine/worldengine/

渲染重建：
projects/SimEngine/worldengine/render/

reward / metric：
projects/SimEngine/worldengine/manager/
projects/AlgEngine/mmdet3d_plugin/datasets/eval_utils/
```

---

## 12. 建议阅读顺序

如果想快速理解代码和论文的对应关系，建议按以下顺序阅读：

1. 论文第 2 节：World Engine Overview
2. 论文第 2.1 节：Pre-training and Long-tail Event Discovery
3. 论文第 2.2 节：Simulation Engine
4. 论文第 2.3 节：Behaviour World Model
5. 论文 Supplementary B.1.4：Scenario Copy / Intent Attack
6. `projects/AlgEngine/scripts/rare_case_sampling_by_pdms.py`
7. `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_rollout_bwm.py`
8. `projects/SimEngine/worldengine/runner/run_simulation.py`
9. `projects/SimEngine/worldengine/manager/metric_manager.py`
10. `projects/SimEngine/worldengine/manager/dense_reward_manager.py`

这样可以从论文概念逐步对应到代码实现。

---

## 13. 本仓库中针对 31 维结构化 SUMO 状态的落地

本仓库的状态不是 3DGS 图像，而是：

```text
[ego speed,
 6 × (relative x, relative y, relative vx, relative vy, relative heading)]
```

因此不能直接复刻 World Engine 的 3DGS 渲染和 diffusion 行为世界模型；但可以保留其最关键的**失败驱动闭环**。实现位于：

```text
embodied/envs/danger_scenarios.py       # 危险原型挖掘与变体采样
embodied/envs/sumo_merge.py             # SUMO 中执行变体、有效性检查
embodied/replay/replays.py               # Danger replay，高风险序列过采样
configs.yaml                             # osrp_sumo_lag_vector_danger 配置
```

对应关系如下：

| World Engine 概念 | 本项目结构化版本 |
|---|---|
| failure / rare-case discovery | collision、TTC < 阈值、risk cost 超阈值、`max_time` 低进度 |
| rare scenario library | 保存源 `.sumocfg`、关键状态、严重度的有界原型库 |
| Scenario Copy | 重放源 `.sumocfg`，对最近相关车施加小速度扰动 |
| Intent Attack | 对同路段最近车施加速度压力，并尝试切入 ego 所在车道 |
| validity filtering | ego 存在、无初始碰撞、最小初始间距检查；失败则回退普通场景 |
| hard experience mining | replay 以 `danger_replay.sample_ratio` 提高含高 cost 序列的采样率 |

训练时使用：

```bash
python train.py --configs osrp_sumo_lag_vector_danger --method osrp_sumo_lag_vector_danger --run.script train_eval --task sumo_None --jax.logical_gpus 0
```

每个 SUMO worker 会将原型和生成轨迹写到：

```text
<logdir>/danger_scenarios/danger_events_worker_*.jsonl
```

该文件记录场景来源、危险类别、严重度和变体参数，因此可以在后续做离线统计、复现与筛选。`eval` 环境默认关闭生成，避免把合成压力场景混入自然场景评估。

### 两阶段训练与 bootstrap 策略 KL 约束

`osrp_sumo_lag_vector_danger` 采用如下时序，而不是从第一个回合就挖掘危险样本：

```text
第 1 ~ 10,000 个训练回合
  普通场景训练；危险矿工和场景生成均关闭
        ↓
第 10,000 回合结束
  冻结当前主 actor 为 π_bootstrap，并保存 bootstrap checkpoint
        ↓
全场景评估
  π_bootstrap 逐个闭环评估 cfg 目录下全部 .sumocfg；只筛选，不生成扰动
        ↓
危险后训练
  将评估得到的失败原型导入训练 worker，开启 Scenario Copy / Intent Attack
  并以 KL(π_current || π_bootstrap) ≤ target 的自适应拉格朗日约束更新 actor
```

具体配置位于 `danger_posttrain`：

```yaml
bootstrap_episodes: 10000
reference_kl_target: 0.05
reference_kl_lam_init: 1.0
reference_kl_lam_lr: 1e-4
```

参考策略网络是 bootstrap 时主 actor 的参数快照，会和 checkpoint 一同保存；主 actor 与 dual actor 的损失都会加入该 KL 项，因此危险后训练不会无约束地偏离前 10,000 回合已经获得的常规驾驶能力。
