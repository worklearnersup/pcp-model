import os
import numpy as np
from ase.io import read
from dscribe.descriptors import SOAP

def generate_pcp_soap(
    WorkDir, FileName, output_file,
    short_species=["Li", "Mn", "Ti", "O"],
    short_r_cut=3, short_n_max=2, short_l_max=4, short_sigma=0.5,
    medium_species=["O"],
    medium_r_cut=6, medium_n_max=4, medium_l_max=6, medium_sigma=0.5,
    start_structure=0, end_structure=None, step=2
):
    # 1. 读取轨迹
    file_path = os.path.join(WorkDir, FileName)
    traj = read(file_path, index=':')
    traj = traj[start_structure : end_structure : step]
    print(f"处理结构数量: {len(traj)}")

    # 2. 初始化 SOAP
    soap_short = SOAP(species=short_species, periodic=True, r_cut=short_r_cut, 
                      n_max=short_n_max, l_max=short_l_max, sigma=short_sigma)
    soap_medium = SOAP(species=medium_species, periodic=True, r_cut=medium_r_cut, 
                       n_max=medium_n_max, l_max=medium_l_max, sigma=medium_sigma)

    # 【关键步骤】必须先对 SOAP 进行 fit，或者执行一次计算，名称空间才会被填充
    # 这里我们使用第一个结构来完成 fit 并获取名称
    soap_short.create(traj[0])
    soap_medium.create(traj[0])
    
    # 获取特征名称并打印
    print("\n--- 维度与特征组成分析 ---")
    short_names = soap_short.get_feature_names()
    print(f"短程维度: {len(short_names)}")
    print("前 10 个短程特征名称:")
    print(short_names[:10]) 
    
    # 3. 计算数据
    descriptors_list, derivatives_list, energies_list, forces_list = [], [], [], []
    
    for atoms in traj:
        oxygen_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "O"]
        if not oxygen_indices: continue
        center = [oxygen_indices[0]]

        deriv_short, desc_short = soap_short.derivatives(atoms, centers=center, method="numerical", return_descriptor=True)
        deriv_medium, desc_medium = soap_medium.derivatives(atoms, centers=center, method="numerical", return_descriptor=True)

        descriptors_list.append(np.concatenate([desc_short, desc_medium], axis=1))
        derivatives_list.append(np.concatenate([deriv_short, deriv_medium], axis=-1))
        energies_list.append(atoms.get_potential_energy())
        forces_list.append(atoms.get_forces())

    # 4. 保存
    prefix = os.path.join(WorkDir, output_file)
    np.save(prefix + "_D.npy", np.vstack(descriptors_list).astype(np.float32))
    np.save(prefix + "_dD_dr.npy", np.array(derivatives_list).squeeze(axis=1).astype(np.float32))
    np.save(prefix + "_E.npy", np.array(energies_list, dtype=np.float32))
    np.save(prefix + "_F.npy", np.array(forces_list, dtype=np.float32))
    
    print("\n[完成] 数据已保存。")

if __name__ == "__main__":
    generate_pcp_soap(
        WorkDir="C:/Users/19807/Desktop/E-F-TRAIN/F-14-00",
        FileName="OUTCAR",
        output_file="PCP_Descriptors"
    )