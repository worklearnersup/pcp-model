import os
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP
from itertools import product

def get_soap_feature_labels(species, n_max, l_max):
    """生成SOAP特征标签，包含原子对和(n,l)信息"""
    labels = []
    # 生成原子对组合
    for s1, s2 in product(species, repeat=2):
        # 生成(n,l)组合
        for n in range(n_max):
            for l in range(l_max + 1):
                labels.append(f"{s1}-{s2}_n{n}_l{l}")
    return labels

def generate_soap_descriptors_with_labels(
    WorkDir, 
    FileName, 
    output_file, 
    r_cut=3.5, 
    n_max=2, 
    l_max=4, 
    sigma=0.5,
    start_structure=1, 
    end_structure=4999,
    step=2
):
    """生成SOAP描述符并保存特征标签文件"""
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
            r_cut=7,    # 氧-氧相互作用使用更大的截断半径
            n_max=4,
            l_max=6,
            sigma=0.5
        )

        # 生成特征标签
        feature_labels_original = get_soap_feature_labels(
            species=["Li", "Mn", "Ti", "O"],
            n_max=n_max,
            l_max=l_max
        )
        
        feature_labels_oxygen = get_soap_feature_labels(
            species=["O"],
            n_max=4,
            l_max=6
        )
        
        # 合并特征标签
        combined_labels = feature_labels_original + feature_labels_oxygen
        
        # 保存特征标签到文件
        label_file_path = os.path.join(WorkDir, output_file + "_features.txt")
        with open(label_file_path, 'w') as f:
            f.write("SOAP特征标签说明:\n")
            f.write("格式: 中心原子-邻居原子_n[n值]_l[l值]\n\n")
            f.write("主SOAP描述符特征:\n")
            for i, label in enumerate(feature_labels_original):
                f.write(f"特征 {i}: {label}\n")
            
            f.write("\n氧SOAP描述符特征:\n")
            for i, label in enumerate(feature_labels_oxygen, start=len(feature_labels_original)):
                f.write(f"特征 {i}: {label}\n")
        
        print(f"特征标签已保存到: {label_file_path}")

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
        print(f"6. 特征标签: {output_file}_features.txt")

    except Exception as e:
        print(f"\n错误发生: {str(e)}")
        raise

if __name__ == "__main__":
    generate_soap_descriptors_with_labels(
        WorkDir='C:/Users/pc/Desktop/E-F-TRAIN/F-14-00',
        FileName='OUTCAR',
        output_file='SOAP_Descriptors',
        start_structure=1,
        end_structure=4999,
        step=2
    )