import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

# ===================== 1. 路径配置 =====================
save_root = "experiment_results"  # 结果根目录
save_dir = os.path.join(save_root, "results")  # 拼接子目录路径
os.makedirs(save_dir, exist_ok=True)  # 创建目录

# ===================== 2. 数据 =====================
models = ['L-GAM(10簇)', 'FPR-ADMM', 'IDL-Greedy', 'XGBoost']
train_r2 = [0.7868, 0.7853, 0.7558, 0.8211]
val_r2 = [0.7810, 0.7817, 0.7557, 0.7966]
test_r2 = [0.7718, 0.7203, 0.7508, 0.7821]
generalization = test_r2
time_cost = [259.0, 4.0, 5.3, 12.0]

# ===================== 3. 绘图设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# 背景色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']
sizes = [100, 100, 100, 100]

# ===================== 4. 绘制散点图和自适应阴影 =====================
y_min = min(time_cost) * 0.8  # y轴下限自适应
y_max = max(time_cost) * 1.2  # y轴上限自适应
ax.set_yscale('log')
ax.set_ylim(max(y_min, 1), y_max)  # 对数刻度下保证最小值 >= 1
ax.set_xlim(0.70, 0.82)

for i in range(len(models)):
    # x范围：验证集与测试集差异
    x_left = min(val_r2[i], test_r2[i])
    x_right = max(val_r2[i], test_r2[i])
    rect_width = x_right - x_left

    # y范围：自适应，避免太小被裁掉
    y_bottom = max(time_cost[i] * 0.95, 1)
    y_top = max(time_cost[i] * 1.05, y_bottom * 1.01)  # 高度至少略大于底
    rect_height = y_top - y_bottom

    # 绘制阴影矩形
    rect = Rectangle((x_left, y_bottom), rect_width, rect_height,
                     color=colors[i], alpha=0.2, linewidth=0, zorder=2)
    ax.add_patch(rect)

    # 中心点虚线
    ax.axvline(x=test_r2[i], color=colors[i], alpha=0.5, linestyle='--',
               linewidth=2, zorder=2)

    # 绘制散点，保证y >= 1
    ax.scatter(generalization[i], max(time_cost[i], 1),
               color=colors[i], marker=markers[i], s=sizes[i],
               label=models[i], alpha=0.9, edgecolors='black', linewidth=1.5,
               zorder=5)

    # 算法名称标注
    ax.annotate(models[i],
                xy=(generalization[i], max(time_cost[i], 1)),
                xytext=(-20, -20), textcoords='offset points',
                fontsize=13, fontweight='bold')

# 图例说明
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.2,
          label='阴影区域表示验证集与测试集\n拟合优度差异')
]
first_legend = ax.legend(handles=legend_elements, loc='upper right',
                         fontsize=10, title='图例说明', title_fontsize=11, framealpha=0.9)
ax.add_artist(first_legend)

# 坐标轴标签
ax.set_xlabel('泛化能力', fontsize=12, fontweight='bold')
ax.set_ylabel('时间代价（对数刻度）', fontsize=12, fontweight='bold')

# 网格
ax.grid(True, alpha=0.3, linestyle='--', which='both')

# 保存和显示
save_path = os.path.join(save_dir, 'generalization_vs_time_with_shading.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
