# 任务一：真实 NPZ 样本的统一格式验收

输入文件：`6c873565d1ba7e753c679303.npz`。
原文件 SHA-256：`4029e30a9ef159fc122c67d86d3383c3372d553317e90508e7e53f57448d2394`。
该文件是用户提供的实际采集数据；本轮没有启动新的 SUMO 或模型训练。

## 实际数据与统一接口

源场景为 `scenario_057_easy_veh011.sumocfg`，`source_type=sumo_original`，
`prediction_source=sumo_execution`，终止原因为 `goal`，采集标记完整。
规范时间轴从 0.2 到 24.6 秒，覆盖 24.4 秒，原始 dt 为 0.2 秒。

| 磁盘字段 | 统一 Episode 接口 | 此样本 shape |
| --- | --- | --- |
| `timestamps` | `episode.timestamps` | `[123]` |
| `ego_state`、`ego_valid_mask` | 同名字段 | `[123,8]`、`[123]` |
| `npc_states`、`npc_valid_mask` | 同名字段，NPC ID 存在 `npc_ids` | `[123,11,8]`、`[123,11]` |
| `ego_actions`、`ego_action_mask` | 同名字段 | `[123,1]`、`[123]` |
| `event_*` | `episode.events['ttc'/'collision'/...]` | 各 `[123]` |
| `metadata_json` | 身份字段及 `episode.metadata` | JSON 标量 |
| `raw_frames_json` | `episode.raw_frames` | 124 个原始帧 |
| `raw_states`、`raw_state_mask` | 保留的旧 31 维观测 | `[123,31]`、`[123]` |

车辆八字段为 `[x,y,vx,vy,yaw,a,length,width]`，单位分别为米、m/s、弧度、m/s²及米。
这里的 x/y 来自 SUMO `getPosition`，参考点是**前保险杠中心**，不直接表示车体中心。
位置语义依据 [SUMO 官方车辆变量文档](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)。
本批补充参考点元信息，不对原始 x/y 做平移。

## 实际检查结果

- 自车 123 帧全部有效，旧 31 维 state 全部有效。
- 11 个 NPC 按 ID 对齐，mask 与逐帧原始车辆列表完全一致；每辆车只有一个连续存在区间。
- NPC mask 有效率约 73.02%，缺失来自进入/离开场景；缺失值为 NaN，有效值均有限，未发现内部缺帧。
- 122 个有效自车动作全部对应下一原始帧的 incoming action，逐项误差为零；末点 action mask 为 False。
- 原始第 124 帧是在 24.6 秒未推进仿真的 `goal` 终止帧，已正确折叠，保留原帧用于审计。
  最后已执行动作是 `0.2350003719`，未执行的 `0.7744574547` 没有覆盖它。
- yaw 与原始 SUMO angle 的方向一致；转换重算差异仅为浮点舍入，未发现角度跳变或 ID 对齐异常。

默认 `history=11/future=30/dt=0.1` 得到 **205 个重叠窗口**，全部 ego mask 有效。
`event_only=True` 得到 **111 个困难窗口**。
这两个数量都是从同一个 episode 切出的片段数量，不代表独立场景数量。

两个窗口组成 batch 后：

```text
history_ego:  [2,11,8]
future_ego:   [2,30,8]
history_npcs: [2,11,11,8]
future_npcs:  [2,30,11,8]
npc_mask:     [2,41,11]
ego_actions:  [2,41,1]
```

ego-centric anchor 转换已检查，历史末点的位置和相对 yaw 为零。
没有地图参考线，Frenet 明确为 None；没有编造地图几何或 NPC 控制量。

## 事件标签的实际含义

| 标签 | 标记帧数 | 来源 |
| --- | --- | --- |
| collision | 0 | 原始碰撞标记/终止原因 |
| near_miss | 17 | 旧 TTC 非负且不超过 0.5 秒的阈值标签 |
| low_ttc | 18 | 旧 TTC 非负且不超过 2.0 秒的阈值标签 |
| low_progress | 0 | 低速持续时间标签 |

16 个 TTC=0 帧有实际邻车，并非缺失 NPC 填零。
旧 `ttc_compute` 使用固定 4.36 米纵向距离阈值，将负结果截成零，
只筛选当前 `x>0、abs(y)<2` 的前方邻车，没有道路冲突关系或横向未来轨迹检查。
例如 16.6 秒的纵向距离约 3.582 米、闭合速度约 0.034 m/s，因小于固定距离阈值得到 TTC=0。
这个零值不能直接解释为碰撞时刻。

因此保留已有标签，并用 `ttc_method/ttc_parameters/ttc_limitations/event_verification`
明确记录其计算来源和限制，不把它们标记为已验证的物理近失。
episode 摘要的 near-miss 阈值为 2.5 秒，规范帧标签为 0.5 秒，两者分别保留。

## 验收范围

该样本已通过任务一的真实输入读取、统一格式、窗口生成、mask 和动作对齐验收。
临时目录中的真实样本归一化保存/加载及逆变换已验证，标量最大误差约 `2.22e-16`，yaw 方向误差为零。

只有一个父场景，默认划分为 train=1、val=0、test=0；不能用本样本验证真实多场景划分或最终测试表现。
样本是 `sumo_original`，不证明 copy/attack/replay 采集覆盖。
seed、地图、模型版本和物理执行有效性保持未知，不能由本文件证明场景可重演。

机器可读的检查结果见 [long_tail_stage1_real_sample_validation.json](long_tail_stage1_real_sample_validation.json)。

规范副本见 [standardized_episode_pool/6c873565d1ba7e753c679303.npz](standardized_episode_pool/6c873565d1ba7e753c679303.npz)。
它补齐来源元信息并记录输入文件路径；逐数组及原始帧比较均与原文件一致，原文件哈希没有改变。
读入兼容只为两个来源标记都符合已知采集器的记录补缺省说明，已有显式说明优先，未知来源不推断。
这些来源说明也传递到 Window，防止下游把阈值标签或位置参考点当作另一种语义。

补充元信息后的全量验证：`D:/anaconda3/python.exe -m unittest discover -s tests -v`，
**55 项通过**，2.141 秒；其中新增三项来源元信息回归先 RED 再 GREEN。

```bash
python -m embodied.envs.scenario_data inspect_dataset --input docs/6c873565d1ba7e753c679303.npz --history 11 --future 30 --dt 0.1
```
