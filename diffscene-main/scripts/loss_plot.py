import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import MaxNLocator

# 1. 读取 CSV 文件
df = pd.read_csv('logs/diffusion_train.csv')


# 2. 提取 loss 列中的数值
def extract_loss(tensor_str):
    match = re.search(r'tensor\(([\d.]+),', tensor_str)
    if match:
        return float(match.group(1))
    else:
        return None


df['loss_value'] = df['loss'].apply(extract_loss)

# 3. 检查并清理数据
print(f"缺失值数量: {df['loss_value'].isnull().sum()}")
df = df.dropna(subset=['loss_value'])

# 4. 按epoch计算统计信息
epoch_stats = df.groupby('epoch').agg({
    'loss_value': ['mean', 'min', 'max', 'std', 'count'],
    'timestep': 'first'
}).reset_index()

# 扁平化列名
epoch_stats.columns = ['epoch', 'loss_mean', 'loss_min', 'loss_max', 'loss_std', 'loss_count', 'first_timestep']

print("\n=== 按Epoch统计的详细信息 ===")
print(epoch_stats.to_string())

# 5. 创建一个图形，包含多个子图
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Training Loss Analysis by Epoch', fontsize=16, fontweight='bold')

# 子图1: 按epoch平均的loss曲线（主要图表）
ax1 = plt.subplot(2, 2, 1)
epochs = epoch_stats['epoch']
loss_means = epoch_stats['loss_mean']
loss_stds = epoch_stats['loss_std']

# 绘制平均loss曲线
line = ax1.plot(epochs, loss_means,
                marker='o',
                markersize=8,
                linewidth=2.5,
                color='darkblue',
                label='Mean Loss per Epoch',
                zorder=3)

# 添加误差条（标准差）
ax1.fill_between(epochs,
                 loss_means - loss_stds,
                 loss_means + loss_stds,
                 alpha=0.2,
                 color='steelblue',
                 label='±1 Std Dev')

# 添加数据点标签（显示具体数值）
for i, (epoch, mean_loss) in enumerate(zip(epochs, loss_means)):
    if i % 5 == 0 or i == len(epochs) - 1:  # 每隔5个epoch显示一次，以及最后一个epoch
        ax1.annotate(f'{mean_loss:.3f}',
                     xy=(epoch, mean_loss),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9,
                     color='darkblue',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='lightgray'))

ax1.set_title('Mean Loss per Epoch with Standard Deviation', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Mean Loss', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 确保x轴显示整数

# 子图2: Loss范围（最小值和最大值）
ax2 = plt.subplot(2, 2, 2)
ax2.fill_between(epochs,
                 epoch_stats['loss_min'],
                 epoch_stats['loss_max'],
                 alpha=0.3,
                 color='lightcoral',
                 label='Loss Range')
ax2.plot(epochs, epoch_stats['loss_min'],
         marker='v',
         markersize=5,
         linewidth=1.5,
         color='darkred',
         label='Min Loss',
         alpha=0.7)
ax2.plot(epochs, epoch_stats['loss_max'],
         marker='^',
         markersize=5,
         linewidth=1.5,
         color='firebrick',
         label='Max Loss',
         alpha=0.7)
ax2.plot(epochs, loss_means,
         marker='o',
         markersize=6,
         linewidth=2,
         color='darkblue',
         label='Mean Loss',
         alpha=0.8)

ax2.set_title('Loss Range per Epoch (Min/Max/Mean)', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss Value', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# 子图3: 每个epoch的训练步数
ax3 = plt.subplot(2, 2, 3)
bars = ax3.bar(epochs, epoch_stats['loss_count'],
               color='teal',
               alpha=0.7,
               edgecolor='darkgreen',
               linewidth=1)

# 在每个柱状图上添加数值
for bar, count in zip(bars, epoch_stats['loss_count']):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
             f'{int(count)}',
             ha='center', va='bottom',
             fontsize=9)

ax3.set_title('Training Steps per Epoch', fontsize=14)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Number of Steps', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.6, axis='y')
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

# 子图4: Loss下降趋势分析
ax4 = plt.subplot(2, 2, 4)

# 原始数据散点图（按颜色区分epoch）
cmap = plt.cm.get_cmap('viridis')
norm = plt.Normalize(epochs.min(), epochs.max())
scatter = ax4.scatter(df['timestep'], df['loss_value'],
                      c=df['epoch'],
                      cmap=cmap,
                      norm=norm,
                      alpha=0.5,
                      s=10,
                      edgecolors='none')

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Epoch', fontsize=12)

# 添加epoch平均线的趋势
ax4.plot(epoch_stats['first_timestep'], loss_means,
         marker='o',
         markersize=8,
         linewidth=3,
         color='darkorange',
         label='Epoch Mean Trend',
         zorder=4)

ax4.set_title('All Loss Values with Epoch Colors and Mean Trend', fontsize=14)
ax4.set_xlabel('Timestep', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 6. 创建一个专门展示epoch平均loss的简洁图表
plt.figure(figsize=(14, 7))

# 绘制主曲线
plt.plot(epochs, loss_means,
         marker='o',
         markersize=10,
         linewidth=3,
         color='darkblue',
         label='Mean Loss per Epoch')

# 添加误差阴影区域
plt.fill_between(epochs,
                 loss_means - loss_stds,
                 loss_means + loss_stds,
                 alpha=0.2,
                 color='steelblue',
                 label='Standard Deviation')

# 突出显示关键点
# 标记最低loss的epoch
min_loss_idx = loss_means.idxmin()
min_epoch = epochs.iloc[min_loss_idx]
min_loss = loss_means.iloc[min_loss_idx]
plt.scatter(min_epoch, min_loss,
            s=200,
            color='red',
            marker='*',
            edgecolors='black',
            linewidth=2,
            zorder=5,
            label=f'Min Loss: {min_loss:.4f} at Epoch {min_epoch}')

# 标记loss下降明显的点
loss_diff = np.diff(loss_means)
significant_drops = np.where(loss_diff < -0.01)[0]  # loss下降超过0.01
for idx in significant_drops:
    epoch1 = epochs.iloc[idx]
    epoch2 = epochs.iloc[idx + 1]
    loss1 = loss_means.iloc[idx]
    loss2 = loss_means.iloc[idx + 1]
    plt.annotate(f'↓{loss1 - loss2:.3f}',
                 xy=(epoch2, loss2),
                 xytext=(5, -15),
                 textcoords='offset points',
                 ha='left',
                 fontsize=10,
                 fontweight='bold',
                 color='darkgreen',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

# 添加数据点标签（选择性地显示）
for i, (epoch, mean_loss) in enumerate(zip(epochs, loss_means)):
    if i == 0 or i == len(epochs) - 1 or epoch % 10 == 0 or i == min_loss_idx:
        plt.annotate(f'{mean_loss:.4f}',
                     xy=(epoch, mean_loss),
                     xytext=(0, 15),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9,
                     color='darkblue',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='lightgray'))

plt.title('Training Progress: Mean Loss per Epoch', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.xlim(epochs.min() - 0.5, epochs.max() + 0.5)

# 添加次要刻度线
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', alpha=0.3)

# 添加背景色区分训练阶段
if len(epochs) > 10:
    # 将训练分为早期、中期、后期（如果有足够数据）
    early_end = min(5, len(epochs) // 3)
    mid_end = min(10, len(epochs) * 2 // 3)

    plt.axvspan(epochs.min(), epochs.iloc[early_end],
                alpha=0.05, color='green', label='Early Training')
    plt.axvspan(epochs.iloc[early_end], epochs.iloc[mid_end],
                alpha=0.05, color='yellow', label='Mid Training')
    plt.axvspan(epochs.iloc[mid_end], epochs.max(),
                alpha=0.05, color='red', label='Late Training')

plt.tight_layout()

# 7. 保存图表
plt.savefig('epoch_mean_loss.png', dpi=300, bbox_inches='tight')
plt.savefig('epoch_mean_loss.pdf', bbox_inches='tight')

# 8. 显示训练总结
print("\n" + "=" * 60)
print("训练总结报告")
print("=" * 60)
print(f"总训练轮数: {len(epoch_stats)}")
print(f"总训练步数: {df['timestep'].max()}")
print(f"平均每轮步数: {epoch_stats['loss_count'].mean():.1f}")

print(f"\nLoss统计:")
print(f"  起始Loss: {loss_means.iloc[0]:.4f}")
print(f"  最终Loss: {loss_means.iloc[-1]:.4f}")
print(f"  最佳Loss: {min_loss:.4f} (Epoch {min_epoch})")
print(f"  总下降: {loss_means.iloc[0] - loss_means.iloc[-1]:.4f}")
print(f"  相对改进: {(1 - loss_means.iloc[-1] / loss_means.iloc[0]) * 100:.1f}%")

print(f"\n训练稳定性:")
print(f"  平均标准差: {loss_stds.mean():.4f}")
print(f"  最大epoch内波动: {(epoch_stats['loss_max'] - epoch_stats['loss_min']).max():.4f}")

# 计算收敛性
if len(epochs) >= 10:
    last_5_avg = loss_means.iloc[-5:].mean()
    last_10_avg = loss_means.iloc[-10:].mean() if len(epochs) >= 10 else last_5_avg
    print(f"\n收敛性分析:")
    print(f"  最后5轮平均Loss: {last_5_avg:.4f}")
    print(f"  最后10轮平均Loss: {last_10_avg:.4f}")

    if abs(last_5_avg - last_10_avg) < 0.001:
        print("  ✓ Loss似乎已收敛")
    else:
        print(f"  ⚠ Loss仍在变化 (差异: {abs(last_5_avg - last_10_avg):.4f})")

print("=" * 60)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import re

# 1. 读取 CSV 文件
df = pd.read_csv('logs/diffusion_train.csv')

# 2. 提取 loss 列中的数值（去掉 tensor(...) 包装）
def extract_loss(tensor_str):
    # 使用正则表达式提取括号内的数字
    match = re.search(r'tensor\(([\d.]+),', tensor_str)
    if match:
        return float(match.group(1))
    else:
        return None

df['loss_value'] = df['loss'].apply(extract_loss)

# 3. 检查是否有空值
print(f"缺失值数量: {df['loss_value'].isnull().sum()}")

# 4. 绘制 loss 随 timestep 变化的曲线
plt.figure(figsize=(14, 7))
plt.plot(df['timestep'], df['loss_value'], label='Training Loss', linewidth=1, alpha=0.7)

# 5. 可选：标记 epoch 变化点
epoch_changes = df[df['epoch'].diff() > 0].index.tolist()
if epoch_changes:
    for change in epoch_changes:
        plt.axvline(x=df.loc[change, 'timestep'], color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Epoch Change' if change == epoch_changes[0] else "")

# 6. 设置图表样式
plt.title('Training Loss over Timesteps', fontsize=16)
plt.xlabel('Timestep', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# 7. 显示图表
plt.show()

# 8. 可选：保存图表为图片
plt.savefig('diffusion_loss_plot.png', dpi=300)
print("图表已保存为 diffusion_loss_plot.png")