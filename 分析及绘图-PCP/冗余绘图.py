import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. 全局样式设置
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

FONT_WEIGHT = 'bold'
LINE_WIDTH = 3.0 
DATA_LABEL_SIZE = 18
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20

# 颜色定义
BLUE_GRAD = [(0/255, 97/255, 161/255), (0/255, 157/255, 221/255), (237/255, 249/255, 254/255)]
RED_GRAD = [(197/255, 8/255, 4/255), (252/255, 120/255, 98/255), (237/255, 249/255, 254/255)]
COLOR_TEXT_BLUE = '#0b5299'
COLOR_TEXT_RED = '#96170b'

# ==========================================
# 2. 渐变填充函数
# ==========================================
def draw_custom_gradient_bar(ax, xpos, width, height, color_list):
    if height <= 0: return
    n_rows = 100 
    grad = np.linspace(0, 1, n_rows).reshape(-1, 1)
    x_edges = [xpos - width/2, xpos + width/2]
    y_edges = np.linspace(0, height, n_rows + 1)
    cmap = LinearSegmentedColormap.from_list('custom_smooth', color_list)
    ax.pcolormesh(x_edges, y_edges, np.flipud(grad), cmap=cmap, shading='flat', zorder=1)

# ==========================================
# 3. 数据准备
# ==========================================
labels = ['Local\ndescriptor', 'Polyhedron-cluster\ndescriptor']
desc_nums = [952, 250]
redundancy = [45.6, 18.0]

x = np.array([0, 1.5]) 
width = 0.45 

# ==========================================
# 4. 创建画布
# ==========================================
fig, ax1 = plt.subplots(figsize=(7, 4)) 
ax2 = ax1.twinx()

ax1.set_ylim(0, 1200)
ax2.set_ylim(0, 100)
scale_factor = 1200 / 100 

# ==========================================
# 5. 绘图执行
# ==========================================
for i in range(len(x)):
    # 蓝色柱子
    blue_x = x[i] - width/2 - 0.01
    h_blue = desc_nums[i]
    draw_custom_gradient_bar(ax1, blue_x, width, h_blue, BLUE_GRAD)
    ax1.text(blue_x, h_blue + 10, f'{int(h_blue)}', 
             ha='center', va='bottom', color=COLOR_TEXT_BLUE, 
             fontname='Arial', fontsize=DATA_LABEL_SIZE, weight=FONT_WEIGHT, zorder=5)

    # 红色柱子
    red_x = x[i] + width/2 + 0.01
    mapped_h_red = redundancy[i] * scale_factor
    draw_custom_gradient_bar(ax1, red_x, width, mapped_h_red, RED_GRAD)
    ax2.text(red_x, redundancy[i] + 1.0, f'{redundancy[i]}%', 
             ha='center', va='bottom', color=COLOR_TEXT_RED, 
             fontname='Arial', fontsize=DATA_LABEL_SIZE, weight=FONT_WEIGHT, zorder=5)

# ==========================================
# 6. 坐标轴精修
# ==========================================

# --- 左轴 (蓝色) ---
ax1.set_ylabel('Descriptor dimensionality', color=COLOR_TEXT_BLUE, 
               fontname='Arial', fontsize=AXIS_LABEL_SIZE, weight=FONT_WEIGHT, 
               labelpad=5)
ax1.set_yticks([0, 300, 600, 900, 1200])
ax1.set_yticklabels([0, 300, 600, 900, 1200], fontname='Arial', fontsize=TICK_LABEL_SIZE, 
                    fontweight=FONT_WEIGHT, color=COLOR_TEXT_BLUE)
ax1.tick_params(axis='y', colors=COLOR_TEXT_BLUE, width=LINE_WIDTH, length=10)

# --- 右轴 (红色) ---
ax2.set_ylabel('Descriptor redundancy (%)', color=COLOR_TEXT_RED, 
               fontname='Arial', fontsize=AXIS_LABEL_SIZE, weight=FONT_WEIGHT, 
               rotation=270, labelpad=20)
ax2.set_yticks([0, 20, 40, 60, 80, 100])
ax2.set_yticklabels([0, 20, 40, 60, 80, 100], fontname='Arial', fontsize=TICK_LABEL_SIZE, 
                    fontweight=FONT_WEIGHT, color=COLOR_TEXT_RED)
ax2.tick_params(axis='y', colors=COLOR_TEXT_RED, width=LINE_WIDTH, length=10)

# --- X 轴 (修改处：增加 pad 值以拉大距离) ---
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontname='Arial', fontsize=AXIS_LABEL_SIZE, weight=FONT_WEIGHT, color='black')
# 关键修改：将 pad 设置为 10 (默认通常是 4)，这样标签会向下移动，远离下框线
ax1.tick_params(axis='x', length=0, pad=10) 

# ==========================================
# 7. 边框设置
# ==========================================

# 1. 先统一设置所有边框的线宽和层级
for ax in [ax1, ax2]:
    for spine_name, spine in ax.spines.items():
        spine.set_linewidth(LINE_WIDTH)
        spine.set_zorder(10)

# 2. 显式控制左右脊柱的可见性和颜色
ax1.spines['left'].set_visible(True)
ax1.spines['left'].set_color(COLOR_TEXT_BLUE)

ax2.spines['left'].set_visible(False)

ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color(COLOR_TEXT_RED)

ax1.spines['right'].set_visible(False)

# 上下边框保持黑色
ax1.spines['top'].set_color('black')
ax1.spines['bottom'].set_color('black')

ax1.set_xlim(x[0]-0.8, x[1]+0.8)
ax1.patch.set_facecolor('white')
ax2.patch.set_visible(False)

plt.tight_layout()
plt.show()