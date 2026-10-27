# 固定划分下的轨迹窗口检查和 train 归一化

服务器已有 `stage4_dataset_20261026_151923/split_manifest.json`。本工具沿用其成员关系，不重新随机划分窗口。它扫描原始 NPZ，保存合格窗口的轻量索引，并只用 train 拟合自车 anchor 局部坐标的归一化器。无需 SUMO、JAX、GPU 或现有 Dreamer checkpoint。

## 安装和运行

已同步完整新代码时直接运行。使用本次补丁包时，在项目根目录执行：

```bash
unzip -o trajectory_window_preparation.zip
patch --dry-run -p1 < trajectory_window_preparation.patch &&
patch -p1 < trajectory_window_preparation.patch
python -m unittest discover -s tests
```

补丁基于上次已安装的统计/划分工具，只修改 `scenario_data.py` 的切窗接口，并新增 CLI、测试和本文档。不改写原始 NPZ、固定划分、训练入口或 checkpoint。

服务器命令：

```bash
conda activate dreamersafe
cd /home/ubuntu/zw/dreamer_huifang_10_26_1

RUN="sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0"
SPLIT="$RUN/stage4_dataset_20261026_151923"
OUT="$RUN/stage4_windows_$(date +%Y%m%d_%H%M%S)"

python -u embodied/envs/prepare_trajectory_windows.py \
  --manifest "$SPLIT/split_manifest.json" \
  --output "$OUT" \
  --history 11 --future 30 --dt 0.2 --stride 1 \
  --progress-every 25
```

`[windows N/951]` 是覆盖检查和第一遍统计；`[normalizer N/M]` 是仅对有合格窗口的 train episode 再读一遍，以拟合圆周 yaw 的残差方差。`M` 不包含 val/test。最后 `[source_check N/951]` 再核对所有来源文件的 SHA256，防止文件在前一遍处理后变化。每遍打印开始、第一个、每 25 个和最后一个文件，以及被拒绝文件。每个文件需解压和检查，整个池需要一定时间；进度持续刷新，不会把全池窗口攒到内存中。请使用停止更新的采集池或稳定快照；逐文件复核不能替代文件系统的全目录原子快照。

如采集池整体迁移到其他目录，可以额外传入 `--input-root /实际/posttrain_episode_pool`。工具使用固定清单中的相对路径和 SHA256；不要修改清单来重新分配成员。输出目录必须新建或为空。

## 窗口和目标规则

- `dt=0.2`：沿用本次采集的 5 Hz 时间尺度。`0.19999999999999996` 等原生 dt 只是浮点差异。
- `history=11`：11 帧历史，包含当前 anchor，从 anchor 前 2 秒到 anchor。
- `future=30`：anchor 后的 30 帧，即未来 6 秒；第一帧在 anchor+0.2 秒。
- `stride=1`：相邻窗口移动一帧。窗口有重叠，全部继承其 episode 的固定集合。
- 自车历史的 8 个字段必须全部有效；自车未来完整性单独统计，不作为输入特征或归一化拟合样本。
- NPC 的历史和未来共 41 帧必须同时有有效 mask 和有限的 8 维状态，才是该窗口的完整监督目标。一个窗口可以包含多个完整目标；至少一个完整目标才写入窗口索引。
- 覆盖统计以 anchor 时刻可见的 NPC 为分母，并统计从 anchor 向后连续可用的历史帧数，以及从 anchor+dt 起向前连续可用的未来帧数。不会跨缺车位置补零。

沿用原有重采样规则：源采样缺失时不跨缺车间隔插值；默认不跨超过原生中位采样间隔 1.5 倍的时间间隙。若明确改成更小 `dt`，新增帧属于插值，不等于新增真实采样。工具不会改变原始采样频率。

## 坐标和归一化

每个窗口用自车 **anchor 时刻**的位置和 yaw 建立固定局部坐标，旋转位置、速度和 yaw，保留既有八字段 `(x,y,vx,vy,yaw,a,length,width)`。不会用未来自车位姿改变参考系。位置参考点沿用原始记录；本次真实数据标记为 `front_bumper_center`，这里没有擅自转换为车身中心。来源语义保存在每个 episode 的 `source_provenance` 中。

拟合总体为：每个合格 train 窗口的自车历史，加上该窗口全部完整目标 NPC 的历史和未来。未来 NPC 是训练标签，可用于 train 的统计；val/test 的状态和未来自车均不参与拟合。重叠窗口中的重复状态会重复贡献，这是窗口加权的统计，不是独立真实帧数量。

使用流式稳定总体均值/方差。yaw 用圆周均值，并用包裹到 `[-pi,pi]` 的角度残差计算标准差；接近零的标准差置为 1。归一化器保存为现有 `ScenarioNormalizer` 格式，明确标记 `coordinate_frame=ego-centric`。它不能直接应用于世界坐标轨迹，也不会覆盖旧的世界坐标归一化器。

## 输出与验收

| 文件 | 内容 |
| --- | --- |
| `window_report.json` | 每个集合及 episode 的窗口、目标覆盖率计数、连续历史/未来长度直方图、拒绝原因和来源信息 |
| `window_index.jsonl` | 每行一个合格窗口：集合、源路径/哈希、episode、父场景组、窗口编号、anchor、完整目标车 ID 列表 |
| `window_dataset_manifest.json` | 原固定清单的哈希、窗口配置、索引哈希、局部坐标约定、归一化器信息和状态 |
| `normalizer.json` | 仅在准备成功时生成；包含拟合 episode ID、状态计数、均值和标准差 |

`windows_scanned` 是所有可切出的时间窗口数，`eligible_windows` 是自车历史和目标覆盖均合格的窗口数，`complete_target_trajectories` 是合格窗口内的完整目标车轨迹数。后者可以大于窗口数，也不代表独立车辆数或独立场景数。`complete_history_targets` / `complete_future_targets` 分别统计单独满足历史/未来要求的 anchor NPC；完整监督需要同时满足两者。

后续可计算：

```text
窗口保留率 = eligible_windows / windows_scanned
完整目标覆盖率 = complete_target_trajectories / anchor_npc_samples
```

完整目标覆盖率的分子只计自车历史合格的窗口，分母包含所有扫描窗口的 anchor NPC；自车历史缺失时该比率也会下降。完整来源记录不一定产生完整目标窗口。

退出码为 0 时要求：原清单与真实成员的祖先泄漏检查通过，所有来源文件哈希/身份检查通过，所有比例非零的集合都有合格窗口，且 train 归一化器成功拟合。退出码 2 表示输入干净但合格窗口或训练样本不足；退出码 1 表示输入、来源哈希、身份或清单错误。逐 episode 错误会保留有效部分的诊断索引/报告，但不生成可用归一化器。只有 `preparation_exit_code=0` 的数据清单可进入后续训练准备。

后续 loader 必须按索引和固定配置重建窗口、验证源哈希，并只选该行的完整目标 ID。`window_index` 是按本次配置的迭代编号；`start_grid_index` 是重采样后的起始网格位置。不要把编号直接当作原始 NPZ 帧号。当前没有存储模型张量，也没有把目标车的未来或未来自车信息作为条件输入。

这些检查证明记录中的覆盖、身份和分组一致性，不证明轨迹符合物理约束或已经通过 SUMO 实际执行验证。未来可见性用于离线监督筛选，不能作为部署时目标选择的额外已知信息。

## 本地验证

新增 18 项测试覆盖 train-only 拟合、缺失轨迹、固定划分、防伪造泄漏标记、源哈希变化、目录迁移、内存释放、两遍统计间及最后拟合遍之后修改文件、局部坐标和圆周 yaw、CLI 进度及输出保护。还逐字段对照原实现的 136 个窗口，结果一致。完整测试和打包结果见 `trajectory_window_preparation_validation.json`（补丁包附带）。

本地实际上传样本只有一个真实 episode：593 帧、58 个全程 ID。默认配置得到 553 个时间窗口，其中 552 个窗口合格，共 5337 条完整目标轨迹；归一化总体 224889 个窗口加权状态。552 个合格窗口的归一化/反归一化均通过检查，原 NPZ 和固定清单哈希不变。为验证完整流程，该单样本使用明确标记的 train-only 清单；它没有独立验证集/测试集，不能据此声称模型泛化或服务器全池数据质量通过。

服务器 951 个 episode 的具体合格窗口数和分组规模，必须以服务器实际运行报告为准。
