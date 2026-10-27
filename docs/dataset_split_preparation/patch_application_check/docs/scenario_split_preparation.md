# 真实 episode 统计与父场景划分

本工具用于阶段四之前的数据准备：逐个检查 `posttrain_episode_pool` 中的 NPZ，统计记录的 SUMO episode 和父场景组，保存固定的 train/val/test 索引。默认比例为 80% / 10% / 10%，seed 为 0。运行时持续输出扫描进度，不生成窗口、归一化器或训练模型。

## 在服务器运行

同步本次代码后，在项目根目录和 `dreamersafe` 环境中执行：

```bash
conda activate dreamersafe
cd /home/ubuntu/zw/dreamer_huifang_10_26

RUN="sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0"
OUT="$RUN/stage4_dataset_$(date +%Y%m%d_%H%M%S)"

python -u embodied/envs/prepare_scenario_splits.py \
  --input "$RUN/posttrain_episode_pool" \
  --output "$OUT" \
  --ratios 0.8 0.1 0.1 \
  --split-seed 0 \
  --progress-every 25

python -m json.tool "$OUT/dataset_report.json"
```

不要把输出重定向到原来的 `stage3_dataset_report.json`。新工具自己写文件，终端保留进度和最终报告。输入应使用停止更新的采集目录或稳定快照；只纳入扫描开始时找到的文件，发现某文件在读取期间变化会拒绝该文件。每次使用新的输出目录，已有非空目录会被拒绝，防止覆盖固定划分。输入 NPZ 不会被修改。

## 输出和判定

| 文件 | 内容 |
| --- | --- |
| `dataset_report.json` | 去重后的 episode 数、父场景组数、各集合规模、来源类型、完整性、拒绝和排除原因、泄漏检查 |
| `episode_index.jsonl` | 每个被接受 episode 的路径、SHA256、原生 dt、帧数、父场景组和集合名称 |
| `split_manifest.json` | 固定 seed/比例和 train/val/test 完整成员清单；后续切窗必须遵守此清单 |

优先查看以下字段：

- `recorded_episode_count`：通过本工具检查、去除相同 ID 的字节相同副本，并排除冲突 ID 后的 episode 数。
- `independent_parent_group_count`：`scenario_id` 与 `parent_scenario_id` 连接成的完整祖先连通组数。缺少父 episode 时仍按记录的父 ID 合组，孙代扰动也归入同组。
- `splits`：每个集合的 episode 数和父场景组数。比例按父场景组分配，因此 episode 比例不一定正好为 80/10/10。
- `leakage_check.passed`：集合间 episode ID、祖先标识和父场景组交集都必须为 0。
- `rejected_file_count` / `excluded_file_count` / `duplicate_file_count`：分别列出损坏或冲突文件、不属于真实执行的数据，以及字节相同的重复 episode 文件。
- `collection_completeness_counts`：完整、未完整及未知的采集数量。已执行的未完整前缀仍统计，并保留标记，避免静默删掉失败样本。

退出码：

| 码 | 含义 |
| --- | --- |
| `0` | 无拒绝文件，所有要求的集合都非空，祖先泄漏检查通过 |
| `1` | 有损坏/冲突文件、没有可接受 episode，或配置/输出错误。逐文件错误会保存有效部分的清单和报告；先处理拒绝原因，再生成新的划分 |
| `2` | 文件检查通过，但父场景组太少，按现有分配规则有集合为空。报告仍保存；补充独立父场景后重新划分，不拆开同源 episode 填空 |

这里的“独立”仅指记录中的祖先关系分离，不证明地图、路况或语义独立。工具依赖记录的来源和父场景标识；错误或缺失的祖先标签仍需要修正。`sumo_original`、`sumo_copy`、`sumo_attack`、`sumo_replay` 可被纳入；显式非 SUMO 执行来源、合成轨迹和阶段三生成的候选轨迹被排除。旧数据没有 `prediction_source` 时依据 SUMO `source_type` 纳入，并计入 `source_type_only_provenance_count` 供检查。

## 后续使用约束

同一 episode 的所有窗口必须沿用清单中的集合；不要再次随机分配窗口。新增数据后应生成新版清单，并保留旧版以便复现实验。后续归一化器只用 train 拟合，val/test 不参与拟合。

本次检查覆盖 canonical NPZ 的 schema、数组形状、时间戳和原生 dt 一致性；没有逐窗口检查目标车覆盖率，也没有验证候选的 SUMO 实际回放。非均匀时间戳数据可保留 `dt=null`，后续切窗阶段再决定采样方案。

## 本地验证依据

```bash
python -m unittest discover -s tests
```

本次在 Python 3.8.3 / NumPy 1.24.4 下通过 137 项测试，其中新增 13 项覆盖祖先链、内存释放、去重/冲突、损坏归档、来源排除、空集合和命令行进度。640 组原划分函数对照结果一致。

生成的格式测试池包含 60 个测试 episode / 20 个父场景组，得到 train 48/16、val 6/2、test 6/2（episode/组），交集为 0；这些是测试夹具，不是服务器采集统计。

本地实际上传样本只有 `c4c745e14ccbd182d2815682.npz`，检查得到 1 个 episode / 1 个父场景组、train 1、val/test 0，退出码 2，源 SHA256 保持 `0652a3fb0e59e6860a8c5bdb5aedb08b729d192317776764fa729a6eedc905cf`。服务器全池的规模以运行上述命令后的报告为准。

已有完整代码同步时直接运行。使用补丁包时，在项目根目录解压并应用：

```bash
unzip -o scenario_split_preparation.zip
patch --dry-run -p1 < scenario_split_preparation.patch &&
patch -p1 < scenario_split_preparation.patch
python -m unittest discover -s tests
```

补丁包括 `scenario_data.py` 的祖先分组函数提取、新 CLI、新测试和本文档。旧数据准备入口与训练入口的默认行为保持原样。无需重新训练现有 Dreamer 模型或重建阶段三 episode。
