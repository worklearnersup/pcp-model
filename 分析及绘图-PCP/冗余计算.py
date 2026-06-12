import os
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#参数调节半径0-3.5-5埃，n=2-6，l=4-7
def generate_soap_descriptors(
    WorkDir, 
    FileName, 
    output_file, 
    r_cut=3.5, 
    n_max=3, 
    l_max=4, 
    sigma=0.5,
    start_structure=1, 
    end_structure=4999,
    step=2
):
    """生成SOAP描述符并确保所有输出文件严格对应"""
    try:
        # 读取轨迹文件
        outcar = read(os.path.join(WorkDir, FileName), index=':')
        total_samples = len(outcar)
        print(f"总结构数: {total_samples}")

        # 验证索引范围
        if (start_structure < 0 or end_structure >= total_samples or 
            start_structure > end_structure):
            raise ValueError(
                f"无效的结构索引范围: start={start_structure}, "
                f"end={end_structure}, 总结构数={total_samples}"
            )

        # 选择结构并确保步长有效
        traj_indices = list(range(start_structure, min(end_structure, total_samples-1) + 1))[::step]
        if not traj_indices:
            raise ValueError("没有选择任何结构，请检查参数")
            
        traj = [outcar[i] for i in traj_indices]
        print(f"已选择{len(traj)}个结构进行处理(步长={step})")

        # 初始化SOAP生成器
        soap_original = SOAP(
            species=["Li", "Mn", "Ti", "O"],
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma
        )
        
        soap_oxygen = SOAP(
            species=["O"],
            periodic=True,
            r_cut=7,    # 氧-氧相互作用使用更大的截断半径，5埃时，n=3，l=5
            n_max=8,
            l_max=10,
            sigma=0.5
        )

        # 初始化存储列表
        data = {
            'positions': [],
            'energies': [],
            'descriptors': [],
            'derivatives': [],
            'forces': []
        }

        # 处理每个结构
        for idx, atoms in enumerate(traj):
            oxygen_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'O']
            if not oxygen_indices:
                print(f"警告: 结构{traj_indices[idx]}没有氧原子，已跳过")
                continue

            # 总是选择第一个氧原子作为中心
            center = oxygen_indices[0]
            
            # 收集数据
            data['positions'].append(atoms.get_positions().astype(np.float32))
            data['energies'].append(atoms.get_potential_energy())
            data['forces'].append(atoms.get_forces().astype(np.float32))
            
            # 计算描述符和导数
            deriv_original, desc_original = soap_original.derivatives(
                atoms, centers=[center], method="numerical", return_descriptor=True
            )
            deriv_oxygen, desc_oxygen = soap_oxygen.derivatives(
                atoms, centers=[center], method="numerical", return_descriptor=True
            )
            
            # 合并描述符
            data['descriptors'].append(
                np.concatenate([desc_original, desc_oxygen], axis=1).astype(np.float32)
            )
            data['derivatives'].append(
                np.concatenate([deriv_original, deriv_oxygen], axis=-1).astype(np.float32)
            )

        # 检查是否有有效数据
        if not data['positions']:
            raise RuntimeError("没有生成任何有效数据 - 所有结构均无氧原子或发生错误")

        # 转换为numpy数组并验证形状一致性
        n_structures = len(data['positions'])
        print(f"\n最终将保存{n_structures}个结构的数据")
        
        # 验证所有数组长度一致
        assert all(len(v) == n_structures for v in data.values()), "数据长度不一致!"
        
        # 保存文件
        np.save(os.path.join(WorkDir, output_file + "_r.npy"), np.array(data['positions']))
        np.save(os.path.join(WorkDir, output_file + "_E.npy"), np.array(data['energies'], dtype=np.float32))
        np.save(os.path.join(WorkDir, output_file + "_D.npy"), np.vstack(data['descriptors']))
        np.save(os.path.join(WorkDir, output_file + "_dD_dr.npy"), np.array(data['derivatives']).squeeze(axis=1))
        np.save(os.path.join(WorkDir, output_file + "_F.npy"), np.array(data['forces']))
        
        # 打印详细信息
        print("\n成功生成并保存了以下文件:")
        print(f"1. 原子坐标: {output_file}_r.npy - 形状: {np.array(data['positions']).shape}")
        print(f"2. 系统能量: {output_file}_E.npy - 形状: {np.array(data['energies']).shape}")
        print(f"3. SOAP描述符: {output_file}_D.npy - 形状: {np.vstack(data['descriptors']).shape}")
        print(f"4. 描述符导数: {output_file}_dD_dr.npy - 形状: {np.array(data['derivatives']).squeeze(axis=1).shape}")
        print(f"5. 原子受力: {output_file}_F.npy - 形状: {np.array(data['forces']).shape}")
        
        # 验证对应关系
        print("\n验证对应关系:")
        for i in range(min(3, n_structures)):  # 检查前3个结构
            print(f"结构{i}:")
            print(f"  原子数: {len(data['positions'][i])}")
            print(f"  受力形状: {data['forces'][i].shape}")
            print(f"  描述符形状: {data['descriptors'][i].shape}")
            print(f"  导数形状: {data['derivatives'][i].squeeze().shape}")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        raise  # 重新抛出异常以便调试


def analyze_feature_redundancy(descriptor_file, threshold=0.9, plot_heatmap=False):
    """
    分析描述符特征之间的皮尔逊相关性，并计算冗余度
    
    参数:
        descriptor_file: _D.npy 文件路径
        threshold: 冗余判断阈值 (绝对值)
        plot_heatmap: 是否绘制相关性热图
        
    返回:
        redundancy_percentage: 高相关特征对的比例 (%)
        highly_correlated_pairs: 高相关特征对列表
    """
    # 加载描述符数据
    D = np.load(descriptor_file)
    n_samples, n_features = D.shape
    print(f"\n分析特征冗余度: {n_features} 个特征，{n_samples} 个样本")

    if n_features > 5000:
        print("⚠️ 特征维度太高，建议先降维或采样部分特征进行估算！")
    
    # 初始化变量
    correlated_pairs = []

    # 计算每一对特征之间的皮尔逊相关系数
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr, _ = pearsonr(D[:, i], D[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            if abs(corr) >= threshold:
                correlated_pairs.append((i, j, corr))

    # 计算冗余比例
    total_pairs = n_features * (n_features - 1) // 2
    redundancy_percentage = (len(correlated_pairs) / total_pairs) * 100

    print(f"\n总特征对数: {total_pairs}")
    print(f"高相关特征对数 ({threshold:.2f}): {len(correlated_pairs)}")
    print(f"特征冗余度: {redundancy_percentage:.2f}%")

    # 绘制热图
    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        plt.imshow(np.abs(corr_matrix), cmap='viridis', interpolation='none')
        plt.colorbar(label='|Pearson Correlation|')
        plt.title('Feature Correlation Heatmap')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        plt.show()

    return redundancy_percentage, correlated_pairs


if __name__ == "__main__":
    work_dir = 'C:/Users/pc/Desktop/E-F-TRAIN/F-14-00'
    out_file = 'SOAP_Descriptors'

    # Step 1: 生成描述符
    generate_soap_descriptors(
        WorkDir=work_dir,
        FileName='OUTCAR',
        output_file=out_file,
        start_structure=1,
        end_structure=4999,
        step=2
    )

    # Step 2: 分析冗余（使用完整路径）
    d_path = os.path.join(work_dir, f'{out_file}_D.npy')

    redundancy, pairs = analyze_feature_redundancy(
        d_path,
        threshold=0.9,
        plot_heatmap=True
    )