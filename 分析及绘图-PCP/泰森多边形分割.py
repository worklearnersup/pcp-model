#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整整合版：计算 Voronoi 距离 + 叠加抖动散点箱线图
背景优化：从上到下由淡紫渐变为极淡紫 (底部接近白色)
修改点：Y轴刻度标签字体已单独加大，箱子边框已去除
"""

import sys
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.core import Structure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
import warnings

# 基础配置
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# -------------------------------------------------
# 1. 路径设置
# -------------------------------------------------
poscar_path_1 = r"C:/Users/19807/Desktop/E-F-TRAIN/F-14/POSCAR"
poscar_path_2 = r"C:/Users/19807/Desktop/E-F-TRAIN/F-14-00/POSCAR"

label_1 = "Rocksalt structure"
label_2 = "Distorted structure"

# -------------------------------------------------
# 2. 核心计算函数
# -------------------------------------------------
def get_supercell_coords(structure, min_sc=(-3, -3, -3), max_sc=(5, 5, 5)):
    lattice = structure.lattice.matrix
    a, b, c = lattice[0], lattice[1], lattice[2]
    
    try:
        orig_coords = np.array([site.coords for site in structure if site.specie.symbol == 'O'])
    except AttributeError:
        orig_coords = np.array([site.coords for site in structure if str(site.specie) == 'O'])
        
    if len(orig_coords) == 0:
        return np.array([]), np.array([]), [], np.array([])

    coords, orig_map, sc_image = [], [], []
    for i in range(min_sc[0], max_sc[0]):
        for j in range(min_sc[1], max_sc[1]):
            for k in range(min_sc[2], max_sc[2]):
                shift = i * a + j * b + k * c
                for idx, coord in enumerate(orig_coords):
                    coords.append(coord + shift)
                    orig_map.append(idx)
                    sc_image.append((i, j, k))
    return orig_coords, np.array(coords), orig_map, np.array(sc_image)

def compute_two_types_distances(poscar_path, label):
    print(f"正在处理: {label}...")
    try:
        structure = Structure.from_file(poscar_path)
    except Exception as e:
        print(f"读取文件失败 ({poscar_path}): {e}")
        return np.array([]), np.array([]), label

    orig, expanded, orig_map, sc_image = get_supercell_coords(structure)
    
    if len(expanded) == 0:
        print(f"警告: {label} 中未找到 O 原子或超胞构建失败。")
        return np.array([]), np.array([]), label

    vor = Voronoi(expanded)
    neighbors = defaultdict(set)
    for p1, p2 in vor.ridge_points:
        neighbors[p1].add(p2)
        neighbors[p2].add(p1)

    center_indices = []
    for idx_exp, (idx_orig, img) in enumerate(zip(orig_map, sc_image)):
        if img[0] == 0 and img[1] == 0 and img[2] == 0:
            center_indices.append(idx_exp)

    face_dists, atomic_dists = [], []
    for idx_exp in center_indices:
        center_coord = expanded[idx_exp]
        for ridge, (p1, p2) in zip(vor.ridge_vertices, vor.ridge_points):
            if -1 in ridge: continue
            if idx_exp in (p1, p2):
                face_center = vor.vertices[ridge].mean(axis=0)
                face_dists.append(np.linalg.norm(face_center - center_coord))
        for nei in neighbors[idx_exp]:
            atomic_dists.append(np.linalg.norm(expanded[nei] - center_coord))

    return np.array(face_dists), np.array(atomic_dists), label

# -------------------------------------------------
# 3. 执行计算
# -------------------------------------------------
face_1, atomic_1, l1 = compute_two_types_distances(poscar_path_1, label_1)
face_2, atomic_2, l2 = compute_two_types_distances(poscar_path_2, label_2)

# -------------------------------------------------
# 4. 绘图
# -------------------------------------------------
if len(face_1) == 0 or len(face_2) == 0:
    print("错误：未获取到有效数据，请检查 POSCAR 路径。")
    # 测试用随机数据 (如需测试请取消下方注释)
    """
    face_1 = np.random.normal(2.5, 0.5, 50)
    face_2 = np.random.normal(2.8, 0.6, 50)
    atomic_1 = np.random.normal(3.0, 0.4, 50)
    atomic_2 = np.random.normal(3.2, 0.5, 50)
    """
    sys.exit()

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2.5 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titleweight'] = 'bold' 

fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

# --- 背景渐变 ---
top_color = "#D1C4E9"   
bottom_color = "#FDFBFF" 
colors_gradient = [top_color, bottom_color]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_fade_purple", colors_gradient)

y_min, y_max = 0, 6
x_min, x_max = 0, 5
gradient_data = np.linspace(1.0, 0.0, 100).reshape(-1, 1)

ax.imshow(gradient_data, extent=[x_min, x_max, y_min, y_max], 
          origin='lower', cmap=cmap, aspect='auto', zorder=0, alpha=0.45)

# --- 颜色与位置配置 ---
colors_deep = {"rocks": "#E64B35", "dist": "#0072B2"} 
colors_light = {"rocks": "#FADBD8", "dist": "#ADD8E6"} 

box_width, within_group_spacing, between_group_spacing = 0.4, 0.6, 1.5
face_center = 1.0
positions = [face_center - within_group_spacing/2, face_center + within_group_spacing/2,
             face_center + between_group_spacing - within_group_spacing/2,
             face_center + between_group_spacing + within_group_spacing/2]

data_groups = [face_1, face_2, atomic_1, atomic_2]
group_keys = ['rocks', 'dist', 'rocks', 'dist']

for pos, data, key in zip(positions, data_groups, group_keys):
    bp = ax.boxplot(data, positions=[pos], widths=box_width, patch_artist=True, showfliers=False,
                medianprops=dict(color=colors_deep[key], linewidth=2.5),  
                whiskerprops=dict(color=colors_deep[key], linewidth=2.5),  
                capprops=dict(color=colors_deep[key], linewidth=2.5),  
                # 【修改点】将 color 替换为 edgecolor='none'，去掉箱子边框
                boxprops=dict(facecolor=colors_light[key], edgecolor='none', linewidth=2.5),  
                zorder=2) 
    
    jitter = np.random.normal(0, 0.04, size=len(data))
    ax.scatter(pos + jitter, data, color=colors_deep[key], s=30, alpha=0.85, edgecolors='none', zorder=3)

# --- 标签与刻度设置 ---
ax.set_xticks([face_center, face_center + between_group_spacing])
ax.set_xticklabels(["Distance to Voronoi face", "O–O atomic distance"], 
                   fontsize=14, fontweight='bold', fontfamily='Arial')
ax.set_ylabel("Distance (Å)", fontsize=13.5, fontweight='bold', fontfamily='Arial')

ax.set_ylim(y_min, y_max)
ax.set_xlim(0, face_center + between_group_spacing + 1.2)

for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    spine.set_zorder(4)

# 设置刻度线样式
ax.tick_params(axis='both', width=2.5, length=6)

# 【关键修改】分别设置 X 轴和 Y 轴字体大小
# 1. 设置 X 轴刻度标签 (保持 14)
for label in ax.get_xticklabels():
    label.set_fontfamily('Arial')
    label.set_fontweight('bold')
    label.set_fontsize(14)

# 2. 设置 Y 轴刻度标签 (加大到 18)
for label in ax.get_yticklabels():
    label.set_fontfamily('Arial')
    label.set_fontweight('bold')
    label.set_fontsize(18)  # <--- 这里改大了

plt.tight_layout()

output_filename = "final_sci_plot_fade_purple.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n📊 绘图已成功保存为：{output_filename}")
print("   (Y轴刻度字体已加大，箱子边框已去除)")
plt.show()