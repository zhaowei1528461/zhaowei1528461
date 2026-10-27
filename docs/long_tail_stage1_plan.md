# SUMO 长尾数据阶段 1（2026-10-10）

用户已用运行日志验证现有 SUMO 流程，并批准在现有实现上增加独立轨迹采集和 Episode/Window 适配。本轮只实现这一批，不训练 Diffusion、不改 RSSM/actor/replay 的输入契约。

## 稳定接口与决定

- `StructuredDangerScenarioMiner.set_active(active, record_events=None, collect_episodes=None, update_library=None)`：旧调用兼容；`collecting` 控制独立执行数据，`update_library` 只决定在线原型是否插入旧库。旧库导入、生成采样和随机数保持原样。
- 新采集默认关闭；新增配置 overlay `sumo_long_tail_data` 显式启用。冻结原库时仍可采集，数据存入 `<logdir>/posttrain_episode_pool/`，不被旧库的 JSONL loader 导入。
- `scenario_data.episode_from_record(record, ego_id='Auto', source_path=None)`、`write_episode(episode, directory)`、`load_episode(path)`、`make_windows(episode, history=11, future=30, dt=0.1)` 是采集与离线转换的边界。
- 规范车辆字段为 `x,y,vx,vy,yaw,a,length,width`，世界坐标米、速度 m/s、yaw 弧度；车辆 ID 跨帧对齐，缺失用 mask。时间轴来自 `sim_time`，日志中的 action 是到达当前帧所施加的控制，规范动作对齐到后续区间。
- 保留原始 31 维 state、source_cfg、原始日志引用；无法确认的 seed、模型版本或地图几何写 None，不伪造。
- 为正常和失败 episode 保存执行数据；是否物理有效/可复现属于后续验收，当前数据池不命名为 verified_tail_pool。
- 用户批准的设计已足以执行；在无 Git 元数据的当前目录直接修改，变更前旧文件保存在系统临时目录 `sumo_stage1_20261010_baseline`，不创建新仓库。

## 实施顺序与责任

- [x] 数据模块与测试：Episode/Window、保存加载、按 ID/mask 重采样和角度处理、坐标转换、父场景划分、仅训练集的标准化、inspect/convert/prepare CLI。
- [x] 矿工开关与测试：采集/记录/入库分离，修正旧在线入库测试的背景车 fixture，测试冻结库记录与 RNG 不变。
- [x] 训练接入与配置：wrapper 透传；首次及恢复后训练同一激活逻辑；评估不启用新数据采集；overlay 关闭后旧行为一致。
- [x] 环境采集：复用一次快照、初始帧、完整 episode、最终变体元信息；仅新采集启用时写独立池；旧矿工依旧使用原危险窗口。
- [x] 集成验证：冻结原库但新池有数据、缺失 NPC/角度/时间与动作对齐、无 SUMO 的真实方法加 TraCI 边界 mock、完整 unittest discovery。
- [x] 文档与审查：中文数据契约和命令，独立代码审查，记录真实验证与未验证项。

## 验收范围

CPU 测试、离线转换、配置组合和 mock SUMO 验证本批新增行为。现有真实 SUMO运行结论沿用用户验证；本机不重新启动 GPU 训练。新轨迹记录在真实 SUMO 中的规模/延迟待用户运行启用命令验收。500 条 sweep 汇总用作基线索引，缺车辆帧时转换应清楚报错。

最终记录：52 项 unittest 通过；独立 CLI 合成 20 episode 生成 420 窗口且父场景划分无交叉；500 条真实 sweep 汇总因无 frames 明确拒绝。使用说明见 [long_tail_stage1.md](long_tail_stage1.md)。本批未实现阶段 2–8，不自动开始下一阶段。

真实输入补充：用户提供 123 帧/11 NPC 的 NPZ，已形成 205 默认窗口并验证时轴、ID/mask、动作对齐。
任务一补齐已知采集器的位置参考点和 TTC 标签来源，输出保留原数组与原帧的规范副本；新全量测试 55 项通过。
验收证据见 [真实样本报告](long_tail_stage1_real_sample_validation.md)，单样本不代表整批划分或物理近失已验证。
