import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon
from ase.io import read
from tqdm import tqdm
import warnings
import sys
import os

# 确保输出编码正确
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# 设置字体：优先使用 Arial，数学部分使用 dejavusans（无衬线）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei'],
    'font.size': 10,
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'axes.linewidth': 2.5,
    'axes.unicode_minus': False,
})

def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """在曲线下方填充渐变色 (从底部透明到顶部半透明)"""
    ax = ax or plt.gca()
    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()
    
    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha
    
    # 创建垂直渐变数组
    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, np.newaxis]
    
    xmin, xmax = x.min(), x.max()
    ymin, ymax = 0, y.max()
    
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder - 1)
    
    xy = np.vstack([np.column_stack([x, y]), [[x[-1], 0], [x[0], 0]]])
    clip_path = Polygon(xy, closed=True, facecolor='none', edgecolor='none')
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    
    return line, im

def compute_rdf_specie(atoms, center_indices, neighbor_indices, rc, Ng, exclude_self=True):
    """计算单帧的径向分布函数 (RDF)"""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    L_times_pbc = np.diag(cell)
    invL = 1.0 / L_times_pbc
    
    N_center = len(center_indices)
    N_neighbor = len(neighbor_indices)
    dr = rc / Ng
    bins = np.linspace(dr / 2, rc, Ng)
    
    g = np.zeros(Ng)
    
    for center_idx in center_indices:
        distVec = positions[neighbor_indices] - positions[center_idx]
        distVec = distVec - np.round(distVec * invL) * L_times_pbc
        distSca = np.sqrt(np.sum(distVec**2, axis=1))
        
        mask = (distSca < rc) & (distSca > 1e-3)
        if exclude_self:
            mask = mask & (neighbor_indices != center_idx)
        distSca = distSca[mask]
        
        if len(distSca) > 0:
            hist, _ = np.histogram(distSca, bins=np.concatenate([[0], bins]))
            g += hist
    
    volume = atoms.get_volume()
    density = N_neighbor / volume
    r = bins
    ideal_gas = 4 * np.pi * r**2 * density * dr
    g_r = g / (ideal_gas * N_center)
    
    return r, g_r

def compute_rdf_from_xdatcar(xdatcar_path, center_element, neighbor_element, rc=8.0, Ng=300, start_frame=0, end_frame=None):
    """从 XDATCAR 文件计算平均 RDF"""
    try:
        traj = read(xdatcar_path, index=":", format="vasp-xdatcar")
    except Exception as e:
        raise FileNotFoundError(f"无法读取文件 {xdatcar_path}: {e}")
        
    if end_frame is None:
        end_frame = len(traj)
    traj = traj[start_frame:end_frame]
    
    if len(traj) == 0:
        raise ValueError("选定的帧范围为空")

    symbols = traj[0].get_chemical_symbols()
    center_indices = [i for i, s in enumerate(symbols) if s == center_element]
    neighbor_indices = [i for i, s in enumerate(symbols) if s == neighbor_element]
    
    if not center_indices or not neighbor_indices:
        raise ValueError(f"元素 {center_element} 或 {neighbor_element} 不存在于结构中")
    
    all_g_r = []
    for atoms in tqdm(traj, desc=f"Processing {center_element}-{neighbor_element}", ncols=70):
        r, g_r = compute_rdf_specie(atoms, center_indices, neighbor_indices, rc, Ng)
        all_g_r.append(g_r)
    
    avg_g_r = np.mean(all_g_r, axis=0)
    return r, avg_g_r

def plot_rdf_results(xdatcar_path, element_pairs, rc=8.0, Ng=300, figsize=(4, 3),
                    x_tick_step=2, y_tick_step=5, tick_fontsize=8, 
                    axis_label_fontsize=10, axis_label_weight='bold'):
    """绘制 RDF 曲线，添加从上到下的淡蓝色渐变背景，并调整 X 轴标签位置"""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # ================= 背景：从上到下的淡蓝色渐变 =================
    top_color = "#BBDEFB"   
    bottom_color = "#F0F8FF" 
    colors_gradient = [top_color, bottom_color]
    cmap_bg = mcolors.LinearSegmentedColormap.from_list("custom_fade_blue", colors_gradient)
    
    y_min, y_max = 0, 15
    x_min, x_max = 0, rc
    
    # 生成渐变数据 (1.0->0.0 对应 bottom->top 颜色映射)
    gradient_data = np.linspace(1.0, 0.0, 100).reshape(-1, 1)
    
    # 绘制背景 (zorder=0)
    ax.imshow(gradient_data, extent=[x_min, x_max, y_min, y_max], 
              origin='lower', cmap=cmap_bg, aspect='auto', zorder=0, alpha=0.6)
    # =============================================================

    for center_element, neighbor_element, color in element_pairs:
        try:
            r, avg_g_r = compute_rdf_from_xdatcar(
                xdatcar_path, center_element, neighbor_element, rc, Ng)
            
            # 绘制曲线和填充 (zorder=2)
            line, im = gradient_fill(r, avg_g_r, fill_color=color, ax=ax, 
                          lw=2.0, color=color, alpha=0.6, zorder=2)
            
        except Exception as e:
            print(f"Error processing {center_element}-{neighbor_element}: {str(e)}")
    
    # Y轴标签
    ylabel_text = r'RDF of rocksalt-like structure $\boldsymbol{g(r)}$'
    ax.set_ylabel(
        ylabel_text,
        fontsize=axis_label_fontsize,
        fontweight=axis_label_weight,
        fontfamily='Arial'
    )
    
    # ================= 【修改点】X轴标签上移 =================
    # labelpad: 控制标签与刻度线的距离。
    # 默认值约为 4.0。设置为负数 (如 -3 到 -6) 可以让标签向上移动，靠近刻度。
    # 这里设置为 -4，你可以根据视觉效果微调 (例如 -2, -5, -6)
    ax.set_xlabel(
        'Distance (Å)', 
        fontsize=axis_label_fontsize, 
        fontweight=axis_label_weight, 
        fontfamily='Arial',
        labelpad=-4  # <--- 关键修改：负值使标签上移
    )
    # =========================================================
    
    ax.set_xlim(0, rc)
    ax.set_ylim(0, 15)
    
    # 刻度设置
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_tick_step))
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_tick_step))
    ax.tick_params(axis='both', which='major', width=2.5, length=6)
    
    # 刻度标签：Arial + 加粗
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(tick_fontsize)
    
    # 确保边框线在最上层
    for spine in ax.spines.values():
        spine.set_zorder(3)
        spine.set_linewidth(2.5)
    
    # 使用 tight_layout 时，labelpad 为负数可能会导致标签被切掉。
    # 如果发生这种情况，可以稍微增加 rect 参数的下边距，或者手动调整 savefig 的 bbox_inches
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # rect=[left, bottom, right, top]，稍微留出底部空间防止切断
    
    output_name = 'RDF_from_XDATCAR_blue_gradient_adjusted.png'
    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    print(f"\n📊 绘图已保存为: {output_name}")
    print("   (X轴标签已上移，背景为淡蓝渐变)")
    plt.show()

if __name__ == "__main__":
    # 配置参数
    XDATCAR_PATH = r"C:/Users/19807/Desktop/E-F-TRAIN/F-14/XDATCAR"  
    
    ELEMENT_PAIRS = [
        ('O', 'Li', '#00AA00'),
        ('O', 'Mn', '#6B5B95'), 
        ('O', 'Ti', '#2198D2'),
        ('O', 'O', '#FF6F61')
    ]
    
    if not os.path.exists(XDATCAR_PATH):
        print(f"❌ 错误：找不到文件 '{XDATCAR_PATH}'")
        print("   请检查代码中的 XDATCAR_PATH 路径是否正确。")
    else:
        plot_rdf_results(
            XDATCAR_PATH,
            ELEMENT_PAIRS,
            rc=8.0,
            Ng=300,
            x_tick_step=2,
            y_tick_step=5,
            tick_fontsize=15,
            axis_label_fontsize=15,
            axis_label_weight='bold'
        )