import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 设置全局字体为 Times New Roman，字号为 18
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 22,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

# 数据准备
volume_compression = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])  # %
static_energy = np.array([
    -125.485, -125.522, -125.541, -125.543, -125.530,
    -125.501, -125.458, -125.402, -125.333, -125.251, -125.159
])  # eV

model_energy = np.array([
    -144.3375, 144.9400, 144.6836, 144.4959, 144.3570, 144.2895, 144.2795, 144.4039, 144.5598, 144.7521, 145.0580
])  # eV

# 插值：使用三次样条插值，生成更平滑曲线
x_new = np.linspace(-5, 5, 100)  # 更细的横坐标点
f_static = interp1d(volume_compression, static_energy, kind='cubic')
f_model = interp1d(volume_compression, model_energy, kind='cubic')

y_static_smooth = f_static(x_new)
y_model_smooth = f_model(x_new)

# 第一张图: 静态计算能量
plt.figure(figsize=(10, 6))
plt.plot(x_new, y_static_smooth, color='blue', linewidth=2)
plt.scatter(volume_compression, static_energy, color='blue', s=50, zorder=5)
plt.xlabel("Volume compression percentage (%)")
plt.ylabel("Energy (eV)")
plt.title("EOS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 第二张图: 模型预测能量
plt.figure(figsize=(10, 6))
plt.plot(x_new, y_model_smooth,  color='red', linewidth=2)
plt.scatter(volume_compression, model_energy, color='red', s=50, zorder=5)
plt.xlabel("Volume compression percentage (%)")
plt.ylabel("Energy (eV)")
plt.title("EOS")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()