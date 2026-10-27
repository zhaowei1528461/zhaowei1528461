# 阶段三真实数据排查与修复（2026-10-26）

本次处理 `stage3_debug_bundle.tar.gz` 中的真实 episode `c4c745e14ccbd182d2815682`。
原始 NPZ 为 593 帧、58 个 NPC、原生 dt=0.2，形成 553 个 H=11/F=30 窗口。
只修复阶段三独立场景生成器，不修改 RSSM、训练、采集、风险权重或车辆/道路阈值。

## 原因与证据

上传 NPZ 的 SHA256 为 `0652a3fb0e59e6860a8c5bdb5aedb08b729d192317776764fa729a6eedc905cf`。
当前 cfg/net/routes 哈希与服务器原搜索报告逐一匹配。该匹配确认本次复现使用同一批文件，
不补造录制时未记录的地图哈希；`source_map_identity_verified_against_recorded_hash` 仍为 false。

原模式沿自车 anchor 朝向建立参数轴。本批所选 NPC 与自车朝向相差 4.8–41.5 度，
`keep_lane` 也被迫将横向速度压回自车参考方向。独立采样速度/加速度还会产生反向运动等无效组合。
上传的 6×2×256=3072 次尝试在本地重放仍全部无效，详见 [原始尝试复现](replay.json)。
所有原实测目标车的 anchor 车身均通过同一采样道路包络检查，不能把失败归咎于 NPZ 位置无效。

例如窗口 0 中参数加速度为 0.07265 m/s² 的一次 `keep_lane` 尝试，实际切向加速度峰值约
0.31333 m/s²，超过 SlowCar1 的 0.2 m/s²。窗口 400 中动力学通过后还有离路及背景车重叠问题。

Linux 原报告使用 Python 3.10.9/NumPy 1.23.5；本地验证为 Python 3.8.3/NumPy 1.24.4。
3071/3072 次原尝试的失败项集合完全匹配，1 次存在数值边界差异。跨平台不承诺逐字节候选 ID 一致。

## 最小修复

- 新增 `--reference-heading target`，参数轴改用目标 NPC 的 anchor 朝向。
- 参数范围中的自车投影速度与生成器使用同一参考方向；`keep_lane` 不再人为转向自车朝向。
- 原 `--reference-heading ego` 仍为默认，默认物理轨迹保持兼容。旧命令不加新选项不会切换模式。
- summary、候选 JSON、NPZ metadata 和搜索 provenance 保存 `parameter_frame`：世界坐标原点、轴角度、来源。
- 失败指标增加实际切向加速度最小值/最大值和最小前向速度。
- 生成器明确拒绝非 world 输入，防止直接调用时将 ego-centric 坐标错误声明为 world。

这仍是局部固定方向近似，不是曲线路网规划器。`cut_in` 的终点参考线穿过自车 anchor、
平行于选定参数轴，不自动代表真实车道中心线。道路、车速、加速度、jerk、yaw-rate、
初始重叠和 NPC-NPC 碰撞过滤均保持原有标准；没有隐藏补采样。

## 正式离线验证结果

同一真实 NPZ、同一 cfg/net/routes、每种方法每窗口预算 256、seed=0、Top-K=5：

| 窗口 | 目标 NPC | 原 Random/CEM 有效 | 修复模式 Random 有效 | 修复模式 CEM 有效 | 最高规则风险 |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | obs_003 | 0 / 0 | 22 | 114 | 0 |
| 100 | obs_002 | 0 / 0 | 5 | 44 | 0.04852 |
| 200 | obs_026 | 0 / 0 | 11 | 98 | 0.03374 |
| 300 | obs_042 | 0 / 0 | 11 | 98 | 0.05019 |
| 400 | obs_064 | 0 / 0 | 2 | 17 | 0.02721 |
| 500 | obs_019 | 0 / 0 | 12 | 121 | 0.9 |

这是同一 episode 的六个窗口，不是六个独立场景，也不能据此宣称 CEM 的统计优势。
窗口 500 完整同 seed 重跑，参数、objective、失败项、有效数量、候选 ID 和规则风险一致；耗时排除比较。
所有导出 JSON/NPZ 的 actor IDs、状态、mask、目标轨迹和元数据一致，源 NPZ SHA256 未变。
汇总结果见 [verification.json](verified_target_heading/verification.json)。本次本地完整逐窗口产物保留在同目录；
补丁包附带汇总报告，服务器运行复验工具后会在指定输出目录生成全部 summary/trials/Top-K。
最终源码下又重新生成并检查全部 57 个去重导出场景，均通过约束且与导出状态/风险一致，
见 [最终源码复核](final_checks.json)。窗口 500 原实测窗口规则分约为 0.3842，生成 Top-1 为 0.9，
该候选与固定实测自车在未来采样时刻发生 OBB 接触，见 [轨迹示意图](window500_candidate.png)。

全量测试：`D:/anaconda3/python.exe -m unittest discover -s tests`，最终 **124 项通过，8.830 秒**。
新增 9 项覆盖参考方向、投影速度边界、刚体旋转、原数据不改写、默认兼容、错误模式、
实际加速度报告、CLI/JSON/NPZ 契约及非 world 输入拒绝。测试先出现预期失败，再实现修复。
独立审查还对 200 个默认模式候选与原源码比较，states/dense states/a/jerk/IDs 均完全一致；
对非 world 输入的元数据边界问题已补失败回归并修复。

## 服务器安装和复验

将 `stage3_reference_heading_fix.patch` 放到服务器项目根目录。先检查补丁能否应用，再执行：

```bash
patch --dry-run -p1 < stage3_reference_heading_fix.patch
patch -p1 < stage3_reference_heading_fix.patch
python -m unittest discover -s tests
```

修复包包含补丁、验证报告与本说明；补丁修改三个生产模块、更新两份原有文档，并新增测试和复验工具。
本地已在原源码副本完成 dry-run 与正式应用，7 个目标文件与验证后的源码完全一致。

单窗口复验：

```bash
RUN="sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0"
NPZ="$RUN/posttrain_episode_pool/c4c745e14ccbd182d2815682.npz"
CFG="embodied/generated_scenarios_merge_7/scenario_324_medium_veh065.sumocfg"

python embodied/envs/run_scene_search.py \
  --input "$NPZ" --sumo-cfg "$CFG" \
  --reference-heading target --window-index 500 \
  --budget 256 --seed 0 --top-k 5 \
  --output "$RUN/stage3_target_w500_$(date +%Y%m%d_%H%M%S)"
```

CFG 应绑定与本次原报告哈希匹配的文件；原服务器录制路径位于另一个仓库目录，
也可显式传入原来已使用的绝对路径。不要仅凭文件同名就更换不同的地图/车型配置。

六窗口复验与同 seed 重跑：

```bash
python docs/stage3_diagnosis/verify_reference_heading.py \
  --input "$NPZ" --sumo-cfg "$CFG" \
  --output "$RUN/stage3_target_verified_$(date +%Y%m%d_%H%M%S)"
```

该工具逐窗口输出进度，自动检查公平预算、导出契约、窗口 500 同 seed 重跑及源文件哈希，
最终写 `verification.json`。不同 Python/NumPy/平台不保证精确复现本地的全部计数；
应以服务器实际报告为准，并在同一环境内验证 seed 重跑一致性。

## 验收边界和阶段四依赖

当前已验证生成/约束/规则排序/导出链路，所有候选仍为 `simulation_verified=false`。
0.9 是固定实测自车响应下的规则风险分，不是碰撞概率或 SUMO 真实碰撞结果。
当前没有执行 SUMO/CARLA 闭环，也没有开始阶段四 Diffusion。

进入阶段四前仍需对完整真实 episode 池做质量检查，按 scenario/parent 谱系划分数据，
确认 train/val/test 有独立来源。此上传包只有一个 episode，不能验证整池划分或训练充分性。
阶段四监督应来自真实采集轨迹，不能把阶段三候选冒充实测监督；ego-centric 的标准化器必须只用训练集重拟合。
