import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from sklearn.metrics import mean_absolute_error
import os

# === 数据加载部分（保持不变）===
work_dir_local = 'C:/Users/19807/Desktop/E-F-TRAIN/F-14/局域势函数训练结果'
work_dir_group = 'C:/Users/19807/Desktop/E-F-TRAIN/F-14/集团势函数训练结果'

E_local_true = np.load(os.path.join(work_dir_local, "E_train_true.npy"))
F_local_true = np.load(os.path.join(work_dir_local, "F_train_true.npy"))
E_local_pred = np.load(os.path.join(work_dir_local, "best_E_train_pred.npy"))
F_local_pred = np.load(os.path.join(work_dir_local, "best_F_train_pred.npy"))

E_group_true = np.load(os.path.join(work_dir_group, "E_train_true.npy"))
F_group_true = np.load(os.path.join(work_dir_group, "F_train_true.npy"))
E_group_pred = np.load(os.path.join(work_dir_group, "best_E_train_pred.npy"))
F_group_pred = np.load(os.path.join(work_dir_group, "best_F_train_pred.npy"))

F_local_true = F_local_true.reshape(-1)
F_local_pred = F_local_pred.reshape(-1)
F_group_true = F_group_true.reshape(-1)
F_group_pred = F_group_pred.reshape(-1)

mae_E_local_meV = mean_absolute_error(E_local_true, E_local_pred) * 1000
mae_F_local_meV_per_A = mean_absolute_error(F_local_true, F_local_pred) * 1000
mae_E_group_meV = mean_absolute_error(E_group_true, E_group_pred) * 1000
mae_F_group_meV_per_A = mean_absolute_error(F_group_true, F_group_pred) * 1000

dist_E_local = np.abs(E_local_pred - E_local_true)
dist_E_group = np.abs(E_group_pred - E_group_true)
dist_F_local = np.abs(F_local_pred - F_local_true)
dist_F_group = np.abs(F_group_pred - F_group_true)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

def format_ax(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6, labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Arial')

def format_ax_with_minor_ticks(ax):
    """为力图添加副刻度的格式化函数"""
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6, labelsize=16)
    # 主刻度长度6，副刻度长度4
    ax.tick_params(axis='both', which='major', width=2, length=6)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Arial')


# ===================== 图1：能量预测 =====================
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 坐标范围设置
x_min = -135
x_max = -75
margin = 0.5
x_min = x_min - margin
x_max = x_max + margin

diag_line = np.linspace(x_min, x_max, 100)
valid_ticks = np.arange(-130, -75, 10)
tick_labels = [f'{int(t)}' for t in valid_ticks]

# --- 左图 ---
sc1 = ax1.scatter(E_local_true, E_local_pred, c=dist_E_local,
                  cmap='coolwarm', s=25, vmin=0, vmax=1.0)
ax1.plot(diag_line, diag_line, 'k--', lw=2, alpha=0.7)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(x_min, x_max)
ax1.set_xticks(valid_ticks)
ax1.set_yticks(valid_ticks)
ax1.set_xticklabels(tick_labels, fontsize=16, fontweight='bold')
ax1.set_yticklabels(tick_labels, fontsize=16, fontweight='bold') # 左图显示数字
ax1.set_xlabel('DFT Energy (×10³ meV)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Predict Energy (×10³ meV)', fontsize=16, fontweight='bold')
ax1.set_title('Local Potential', fontsize=16, fontweight='bold', pad=10)
ax1.text(0.97, 0.20, f'MAE: {mae_E_local_meV:.2f} meV',
         transform=ax1.transAxes, ha='right', va='center',
         fontsize=14, fontweight='bold')
format_ax(ax1)

# --- 右图 ---
sc2 = ax2.scatter(E_group_true, E_group_pred, c=dist_E_group,
                  cmap='coolwarm', s=25, vmin=0, vmax=1.0)
ax2.plot(diag_line, diag_line, 'k--', lw=2, alpha=0.7)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(x_min, x_max)
ax2.set_xticks(valid_ticks)
ax2.set_xticklabels(tick_labels, fontsize=16, fontweight='bold')

# 【修改点】设置刻度位置，但标签设为空列表
ax2.set_yticks(valid_ticks) 
ax2.set_yticklabels([]) # 这里设为空，不显示数字，但刻度线还在

# 不设置 ylabel，因为左图已经有了
ax2.set_xlabel('DFT Energy (×10³ meV)', fontsize=16, fontweight='bold')
ax2.set_title('Polyhedral-Cluster Potential', fontsize=16, fontweight='bold', pad=10)
ax2.text(0.97, 0.20, f'MAE: {mae_E_group_meV:.2f} meV',
         transform=ax2.transAxes, ha='right', va='center',
         fontsize=14, fontweight='bold')
format_ax(ax2)

# === 颜色条 ===
bbox1 = ax1.get_position()
bbox2 = ax2.get_position()
y_min = min(bbox1.y0, bbox2.y0)
y_max = min(bbox1.y1, bbox2.y1)
cbar_width = 0.015
cbar_height = y_max - y_min

cbar_left = 0.90
cbar_bottom = y_min
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
norm = Normalize(vmin=0, vmax=1.0)
cbar = ColorbarBase(cbar_ax, cmap='coolwarm', norm=norm, orientation='vertical')
cbar.set_ticks(np.arange(0, 1.1, 0.1))
cbar.set_ticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)], fontsize=16, fontweight='bold')
cbar.ax.tick_params(width=2, length=6, labelsize=16)
for spine in cbar_ax.spines.values():
    spine.set_linewidth(2)
cbar.set_label('Absolute Error (×10³ meV)', fontsize=16, fontweight='bold', rotation=270, labelpad=17)

plt.subplots_adjust(left=0.08, right=0.88, wspace=0.08)
plt.savefig('energy_comparison_separated.png', dpi=300, bbox_inches='tight')


# ===================== 图2：力的预测 =====================
fig2 = plt.figure(figsize=(9, 4))

ax3 = fig2.add_subplot(1, 2, 1)
ax4 = fig2.add_subplot(1, 2, 2)

# 设置固定的坐标轴范围
f_min_display = -11.5
f_max_display = 11.5
diag_line_f = np.linspace(f_min_display, f_max_display, 100)

# 主刻度：间隔5；副刻度：间隔2.5
major_ticks = np.arange(-10, 11, 5)
minor_ticks = np.arange(-10, 11, 2.5)

# --- 左图：局部势函数的力预测 ---
sc3 = ax3.scatter(F_local_true, F_local_pred, c=dist_F_local,
                  cmap='coolwarm', s=10, alpha=0.6, 
                  vmin=0, vmax=1.0)
ax3.plot(diag_line_f, diag_line_f, 'k--', lw=2, alpha=0.7)
ax3.set_xlim(f_min_display, f_max_display)
ax3.set_ylim(f_min_display, f_max_display)

ax3.set_xticks(major_ticks)
ax3.set_xticks(minor_ticks, minor=True)
ax3.set_yticks(major_ticks)
ax3.set_yticks(minor_ticks, minor=True)

ax3.set_xticklabels([f'{int(t)}' if t % 10 == 0 else f'{t:.0f}' for t in major_ticks], 
                    fontsize=16, fontweight='bold')
ax3.set_yticklabels([f'{int(t)}' if t % 10 == 0 else f'{t:.0f}' for t in major_ticks], 
                    fontsize=16, fontweight='bold') # 左图显示数字

ax3.set_xlabel('DFT Force (×10³ meV/Å)', fontsize=16, fontweight='bold')
ax3.set_ylabel('Predict Force (×10³ meV/Å)', fontsize=16, fontweight='bold')
ax3.set_title('Local Potential', fontsize=16, fontweight='bold', pad=10)
ax3.text(0.97, 0.20, f'MAE: {mae_F_local_meV_per_A:.2f} meV/Å',
         transform=ax3.transAxes, ha='right', va='center',
         fontsize=14, fontweight='bold')
format_ax_with_minor_ticks(ax3)

# --- 右图：集团势函数的力预测 ---
sc4 = ax4.scatter(F_group_true, F_group_pred, c=dist_F_group,
                  cmap='coolwarm', s=10, alpha=0.6,
                  vmin=0, vmax=1.0)
ax4.plot(diag_line_f, diag_line_f, 'k--', lw=2, alpha=0.7)
ax4.set_xlim(f_min_display, f_max_display)
ax4.set_ylim(f_min_display, f_max_display)

ax4.set_xticks(major_ticks)
ax4.set_xticks(minor_ticks, minor=True)
ax4.set_yticks(major_ticks)
ax4.set_yticks(minor_ticks, minor=True)

ax4.set_xticklabels([f'{int(t)}' if t % 10 == 0 else f'{t:.0f}' for t in major_ticks], 
                    fontsize=16, fontweight='bold')

# 【修改点】设置刻度位置，但标签设为空列表
ax4.set_yticklabels([]) # 这里设为空，不显示数字，但刻度线还在

# 不设置 ylabel
ax4.set_xlabel('DFT Force (×10³ meV/Å)', fontsize=16, fontweight='bold')
ax4.set_title('Polyhedral-Cluster Potential', fontsize=16, fontweight='bold', pad=10)
ax4.text(0.97, 0.20, f'MAE: {mae_F_group_meV_per_A:.2f} meV/Å',
         transform=ax4.transAxes, ha='right', va='center',
         fontsize=14, fontweight='bold')
format_ax_with_minor_ticks(ax4)

# === 颜色条 ===
bbox3 = ax3.get_position()
bbox4 = ax4.get_position()
y_min_f = min(bbox3.y0, bbox4.y0)
y_max_f = min(bbox3.y1, bbox4.y1)
cbar_width_f = 0.015
cbar_height_f = y_max_f - y_min_f

cbar_left_f = 0.90
cbar_bottom_f = y_min_f
cbar_ax_f = fig2.add_axes([cbar_left_f, cbar_bottom_f, cbar_width_f, cbar_height_f])

norm_f = Normalize(vmin=0, vmax=1.0)
cbar_f = ColorbarBase(cbar_ax_f, cmap='coolwarm', norm=norm_f, orientation='vertical')

cbar_f.set_ticks(np.arange(0, 1.1, 0.1))
cbar_f.set_ticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.1)], fontsize=16, fontweight='bold')
cbar_f.ax.tick_params(width=2, length=6, labelsize=16)
for spine in cbar_ax_f.spines.values():
    spine.set_linewidth(2)
cbar_f.set_label('Absolute Error (×10³ meV/Å)', fontsize=16, fontweight='bold', rotation=270, labelpad=17)

plt.subplots_adjust(left=0.08, right=0.88, wspace=0.08)
plt.savefig('force_comparison_separated.png', dpi=300, bbox_inches='tight')

# 显示两个图
plt.show()