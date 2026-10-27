# 阶段四：单 NPC 直接轨迹扩散

本版已实现固定窗口加载、无引导扩散训练、保存/加载、离线评估和候选导出。输入沿用阶段四已经冻结的划分、窗口索引和归一化器；不需要重新采集或重新训练 Dreamer。

实现完成不代表生成质量达标。本地只有一个真实 episode；40 次真实样本更新后的运动有效率为 0，不能作为可执行危险场景使用。服务器全量 train/val 尚未在本地运行。

## 服务器直接运行

以下命令在项目根目录执行；目录名按你当前实际位置修改。已有完整代码同步时，跳过补丁安装。

```bash
conda activate dreamersafe
cd /home/ubuntu/zw/dreamer_huifang_10_21_2

# 仅使用补丁包时执行；dry-run 成功后再应用。
unzip -o stage4_direct_diffusion.zip
patch --dry-run --batch --forward -p1 < stage4_direct_diffusion.patch &&
patch --batch --forward -p1 < stage4_direct_diffusion.patch

python -c "import sys, numpy, torch; print(sys.version); print('numpy', numpy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

依赖为 NumPy 和 PyTorch，绘图另需可选的 Matplotlib。最终实际验证的运行环境为 Python 3.8.20、NumPy 1.24.4、PyTorch 2.3.0、CPU。新入口不依赖 NuScenes、PyG、TrafficModel 权重或 Dreamer 的 JAX 网络。先确认当前环境能够导入 torch；不要为了本功能覆盖原有 Dreamer 依赖。

### 1. 检查现有数据加载器

沿用已经通过验收的 951 个 episode 和原始 train/val/test 划分，不重建索引、不重新拟合归一化器。

```bash
RUN="sumo_fpi_log/20261021-105954_osrp_sumo_lag_vector_danger_sumo_None_0"
WINDOWS="$RUN/stage4_windows_20261026_160548"
SPLIT="$RUN/stage4_dataset_20261026_151923/split_manifest.json"
POOL="$RUN/posttrain_episode_pool"
MANIFEST="$WINDOWS/window_dataset_manifest.json"

# 数组参数保留在同一个 Bash 会话中，后续命令继续使用。
DATA=(--dataset-manifest "$MANIFEST" --split-manifest "$SPLIT" --input-root "$POOL")

python -u embodied/envs/trajectory_dataset.py "${DATA[@]}" \
  --split train --inspect 16

python -m unittest discover -s tests
```

数据检查应返回 `exit_code: 0`、`label_shape: [30, 2]`、`finite_conditions: true`，世界坐标往返误差应远小于毫米。启动时会校验并扫描整个索引；`--inspect 16` 只限制取样检查的数量。不会一次性加载全部 NPZ 或物化数百万条轨迹张量。

清单内保留了旧机器绝对路径，所以命令显式传入当前 `--split-manifest` 和 `--input-root`。若窗口目录移动，索引和归一化器优先从 `window_dataset_manifest.json` 的同目录寻找。

可选地图绑定：若 `source_cfg` 旧路径失效，且对应 sumocfg 和 net 已搬到新目录，设置一次：

```bash
# 仅在实际拥有对应地图目录时添加；net-file 仍按 sumocfg 内的相对路径解析。
# DATA+=(--map-root /absolute/path/to/generated_scenarios_merge_7)
```

该目录按 sumocfg 文件名匹配；请在训练和采样时使用相同地图集合。地图缺失时以 mask 和 missing-map 表示，`road_valid_rate` 为 `null`。现有文件的哈希绑定用于防止训练过程中静默更换地图；这不等于证明当前地图与录制时文件完全相同。

### 2. 先用合成小样本检查训练链路

这一项仅检查 PyTorch 前向、反向、checkpoint 和采样，不衡量真实数据效果。

```bash
SMOKE="$RUN/stage4_synthetic_$(date +%Y%m%d_%H%M%S)"
python -u embodied/envs/train_trajectory_diffusion.py \
  --synthetic-smoke --output "$SMOKE" \
  --epochs 4 --max-steps 16 --batch-size 4 \
  --base-dim 16 --dim-mults 1 2 --condition-dim 32 \
  --diffusion-steps 20 --sampling-steps 10 --samples 4 \
  --device auto --seed 0 --log-every 4
```

成功报告包含 `optimization_steps: 16`、`checkpoint_roundtrip_verified: true`、`latest_checkpoint` 和 `best_checkpoint`。输出目录必须是新目录或空目录，重跑时换一个名称。

### 3. 用真实 train/val 跑小规模检查

先检查现有真实 train/val 能够完成训练和验证；不使用 `--allow-train-only-smoke`。

```bash
PROBE="$RUN/stage4_real_probe_$(date +%Y%m%d_%H%M%S)"
python -u embodied/envs/train_trajectory_diffusion.py "${DATA[@]}" \
  --output "$PROBE" --train-limit 128 --val-limit 32 \
  --epochs 2 --max-steps 32 --batch-size 8 \
  --base-dim 16 --dim-mults 1 2 --condition-dim 32 \
  --diffusion-steps 100 --sampling-steps 20 --samples 4 \
  --device auto --seed 0 --log-every 8

EVAL="$PROBE/val_samples"
python -u embodied/envs/sample_trajectories.py "${DATA[@]}" \
  --checkpoint "$PROBE/best.pt" --output "$EVAL" \
  --split val --limit 32 --batch-size 8 \
  --samples 5 --sampling-steps 20 --export-limit 3 --device auto
```

128/32 是索引中各 split 的前缀窗口，用于快速检查，可能来自很少的父场景，不能作为总体质量估计。训练目标会按 seed、epoch、窗口身份从完整目标列表中选取；val/test 固定选择 anchor 时距自车最近的合格目标，同距离按 ID 排序。

### 4. 再扩大训练并独立评估

小规模真实检查成功后，再运行以下训练配置。20 个 epoch 是起始配置，需要根据真实验证曲线决定训练长度，不保证收敛。

```bash
TRAIN="$RUN/stage4_diffusion_train_$(date +%Y%m%d_%H%M%S)"
python -u embodied/envs/train_trajectory_diffusion.py "${DATA[@]}" \
  --output "$TRAIN" --train-limit 0 --val-limit 1024 \
  --epochs 20 --batch-size 64 --num-workers 2 \
  --diffusion-steps 100 --sampling-steps 20 --samples 4 \
  --device auto --seed 0 --log-every 100

# 用全部 val 窗口评估已选定 checkpoint；此步骤可能耗时较长。
python -u embodied/envs/sample_trajectories.py "${DATA[@]}" \
  --checkpoint "$TRAIN/best.pt" --output "$TRAIN/val_full" \
  --split val --limit 0 --batch-size 64 --num-workers 2 \
  --samples 5 --sampling-steps 20 --export-limit 3 --device auto
```

`--train-limit 0` 使用全部 train 窗口；`--val-limit 1024` 仍是前缀子集，仅用于每个 epoch 的快速监控。需要全量验证参与 checkpoint 选择时，从训练开始改成 `--val-limit 0`。默认值为 512，也属于子集。`best.pt` 按所配置 val 集上的 `minADE_m` 选择，`latest.pt` 保存最近一次更新；checkpoint 含优化器状态，但当前 CLI 尚未实现断点续训。

固定配置和 checkpoint 后，再把独立评估命令改为 `--split test --limit 0` 并使用新输出目录。不要反复查看 test 来调整训练配置。

## 数据和模型契约

新增模块均位于 `embodied/envs/`，没有改动原有 Dreamer/RSSM、回放池、训练默认入口或上传的 DiffScene 文件。

| 文件 | 作用 |
| --- | --- |
| `trajectory_dataset.py` | 固定清单校验、窗口读取、目标选择、历史条件、坐标逆变换、episode 局部采样 |
| `trajectory_maps.py` | 有限容量地图缓存、静态车道特征、缺失标记和文件绑定检查 |
| `trajectory_denoiser.py` | 改编 DiffScene 的时序残差去噪结构，加入历史/地图/类型条件 |
| `trajectory_diffusion.py` | 一致的 0 到 T−1 扩散日程、噪声 MSE、DDPM 和 DDIM |
| `trajectory_training.py` | 训练、验证、checkpoint、计时和可选绘图 |
| `train_trajectory_diffusion.py` | 独立训练入口 |
| `sample_trajectories.py` | 离线采样、指标和候选导出入口 |
| `trajectory_metrics.py` | ADE/FDE、恒速基线、多样性和采样运动检查 |
| `diffusion_candidate_adapter.py` | 保留来源与实测历史的 JSON/NPZ 候选导出 |

现有配置为 H=11、F=30、dt=0.2。模型输入的车辆状态字段沿用 canonical 八维；条件为 ego/target/邻车的历史、对应 mask、可用静态地图和已知场景类型。邻车最多 8 辆，只按 anchor 历史距离排序。无效值置零并保留 mask；没有地图或邻车时编码仍然有限。

未来目标 x/y 是训练标签 `[B,30,2]`，不会送入条件编码器。未来自车和其他 NPC 不进入条件或目标优先级计算。已有完整未来 mask 仅用于数据合格性筛选。采样输出 `[B,K,30,2]`，先按冻结的 train-only 参数反归一化，再还原世界坐标。

没有照搬原版 TrafficModel 的 latent 轴：原版重复的 20 个 latent 样本不等于 30 个未来帧。本版复用时序残差、时间嵌入和条件注入结构；显式区分 batch、候选数和时间轴，并修正原版日程索引问题。不引入 LLM、RSSM、规则风险或其他引导项。

索引保留字节偏移，episode 缓存默认每个进程最多 2 个，地图缓存最多 4 个。读取时检查固定 split 成员、源 NPZ 身份和哈希；索引或地图在启动校验后发生常规文件变更会报错。归一化器 ID 根据完整参数、拟合范围和 train 源文件身份重算验证，不允许更换参数后沿用原 ID。训练和独立采样还会核对 checkpoint 中的数据集指纹。

## 指标与产物如何看

训练输出 `config.json`、`metrics.jsonl`、`training_report.json`、`best.pt`、`latest.pt`。采样输出 `evaluation.json`，以及数量受 `--export-limit` 限制的 `candidate_*.json/.npz`；安装 Matplotlib 时同时输出 PNG。

| 指标 | 定义 |
| --- | --- |
| `mean_ADE_m` / `mean_FDE_m` | 全部 K 条采样的平均误差，单位米 |
| `minADE_m` / `minFDE_m` | 对每个窗口分别用真实未来挑选最优 K 条中的误差；两项可以来自不同候选，是 oracle 指标 |
| `constant_velocity_ADE_m` / `constant_velocity_FDE_m` | 只使用目标 anchor 位置和速度的恒速预测基线 |
| `endpoint_diversity_m` | K 条采样末端两两距离的平均值；大并不等于有效 |
| `motion_valid_rate` | 采样点差分得到的运动量满足暂定阈值的比例 |
| `road_valid_rate` | 有地图且已知位置参考时，采样车身 footprint 位于车道缓冲区的比例；无法检查时为 null |
| `seconds_per_window` | 包含该窗口 K 条候选的采样耗时，CUDA 会同步计时；不含数据加载和绘图 |

运动阈值暂定为速度 40 m/s、加速 4 m/s²、减速 8 m/s²、jerk 15 m/s³、偏航角速度 1 rad/s、横向加速度 4 m/s²。这些阈值尚未按 SlowCar1 等车型标定，不是 SUMO 车辆动力学验收。道路检查是离散采样检查，也不是连续碰撞或仿真验证。

采样默认 DDIM，可用 `--method ddpm --sampling-steps T` 运行完整 DDPM；T 要与 checkpoint 中的 `diffusion_steps` 相同。记录 seed、checkpoint 哈希和实际 batch 大小/行号/K/窗口顺序，可在相同软件、设备和 batch 设置下复现本次采样；更改 batch 会改变随机数消费顺序。CPU 已验证保存前后固定种子逐元素相等，GPU 尚未验证。

候选保留原始目标历史，`target_trajectory` 含实测 anchor 加 30 个生成点。世界位置继续使用录制时的 `center` 或 `front_bumper_center`，不会混用。速度由前一时间间隔的位置差分得到，低速时保持前一个朝向；这些是轨迹推导量。候选明确记录 `source_type: generated_candidate`、`simulation_verified: false`、`risk: null`，没有生成自车响应或其他车辆未来，不会自动写入 replay 或注入 SUMO。执行场景前还需独立的跟踪控制和闭环仿真验收。

## 已获得的本地证据与边界

完整测试通过 **182 项**（43.277 秒）。涵盖未来条件隔离、mask、目标可复现、坐标往返、源文件/索引/归一化器检查、地图变更、解析扩散系数、前向反向、DDPM/DDIM、保存加载、真实格式 train/val fixture、评估和候选来源。测试时使用现有 omnisafe 环境，并仅在测试进程通过 PYTHONPATH 补充 base 环境已有的 imageio 等旧项目依赖，没有安装或覆盖依赖。

本地工作流使用 16 条合成训练样本和上传的一个真实 episode。真实窗口清单只含 train，因此显式使用 `--allow-train-only-smoke`，没有把它冒充 val/test。实际完成了合成 64 次更新、真实 40 次更新、两种 checkpoint 往返和独立采样导出。原始代码、DiffScene 参考代码和本地真实源数据哈希均保持不变。

下表是这次短训的真实结果，表明生成质量尚未通过；没有通过裁剪、投影或挑选未来最优候选来掩盖发散采样。

| 检查 | 更新次数 | 评估窗口 | 平均 ADE | minADE | 恒速 ADE | 运动有效率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 合成 val | 64 | 4 | 约 4158 m | 约 3272 m | 约 0.067 m | 0 |
| 真实单 episode 的 train 子集 | 40 | 8 | 约 5610 m | 约 4970 m | 约 2.246 m | 0 |

训练噪声损失下降只能说明优化链路有效，不能证明轨迹生成可用。下一步应先运行上述服务器真实 train/val 检查，再扩大训练，观察误差、恒速基线、物理比例和轨迹图。只有真实独立验证结果达标后，才进入候选执行与场景质量验收。当前不提供训练好的可用生成模型。

完整本地证据位于 `docs/stage4_diffusion_validation/final_verified/verification.json`；补丁包中的 `stage4_validation.json` 是同一报告。服务器 951 个 episode 的全量训练、真实 val/test 质量、GPU 训练和 SUMO 闭环执行未在本地验证。
