import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 工作目录配置
WorkDir = 'C:/Users/19807/Desktop/E-F-TRAIN/F-14-00/集团势函数训练结果'

# ------------------------------
# 数据加载与验证
# ------------------------------
def load_and_validate_data():
    data_dict = {
        "r": np.load(os.path.join(WorkDir, "SOAP_Descriptors_r.npy")),
        "D": np.load(os.path.join(WorkDir, "SOAP_Descriptors_D.npy")),
        "F": np.load(os.path.join(WorkDir, "SOAP_Descriptors_F.npy")),
        "E": np.load(os.path.join(WorkDir, "SOAP_Descriptors_E.npy")).reshape(-1, 1),
        "dD_dr": np.load(os.path.join(WorkDir, "SOAP_Descriptors_dD_dr.npy"))
    }

    if len(data_dict["dD_dr"].shape) == 5 and data_dict["dD_dr"].shape[1] == 1:
        data_dict["dD_dr"] = data_dict["dD_dr"][:, 0, :, :, :]
    print("处理后的dD_dr_numpy shape:", data_dict["dD_dr"].shape)
    return data_dict

# ------------------------------
# 数据预处理
# ------------------------------
def preprocess_data(data_dict, test_size=0.2):
    total_samples = len(data_dict["r"])
    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42)

    train_data = {k: v[train_idx] for k, v in data_dict.items()}
    val_data = {k: v[val_idx] for k, v in data_dict.items()}
    
    scaler = StandardScaler().fit(train_data["D"])
    train_data["D"] = scaler.transform(train_data["D"])
    val_data["D"] = scaler.transform(val_data["D"])
    data_dict["D"] = scaler.transform(data_dict["D"])
    
    # 在preprocess_data函数中添加对E和F的标准化
    scaler_F = StandardScaler().fit(train_data["F"].reshape(-1, 3))
    train_data["F"] = scaler_F.transform(train_data["F"].reshape(-1, 3)).reshape(-1, train_data["F"].shape[1], 3)
    val_data["F"] = scaler_F.transform(val_data["F"].reshape(-1, 3)).reshape(-1, val_data["F"].shape[1], 3)

    scale_tensor = scaler.scale_[None, None, :]
    train_data["dD_dr"] = train_data["dD_dr"] / scale_tensor
    val_data["dD_dr"] = val_data["dD_dr"] / scale_tensor
    data_dict["dD_dr"] = data_dict["dD_dr"] / scale_tensor

    return train_data, val_data, data_dict, scaler, train_idx, val_idx

# ------------------------------
# 模型定义
# ------------------------------
class EnergyForceMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512]*3, num_atoms=20):
        super().__init__()
        self.num_atoms = num_atoms
        
        # 共享的特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.001)
        )
        
        # 能量预测分支
        self.energy_branch = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.001),
            nn.Linear(hidden_sizes[1], 1)
        )
        
        # 力预测分支
        self.force_branch = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.001),
            nn.Linear(hidden_sizes[1], num_atoms * 3)  # 输出 num_atoms * 3 个力分量
        )

    def forward(self, x):
        shared_features = self.shared_layers(x)
        energy = self.energy_branch(shared_features)
        force = self.force_branch(shared_features).view(-1, self.num_atoms, 3)  # 重塑为 [batch_size, num_atoms, 3]
        return energy, force

# ------------------------------
# 主训练流程
# ------------------------------
def main():
    data_dict = load_and_validate_data()
    train_data, val_data, full_data, scaler, train_idx, val_idx = preprocess_data(data_dict)
    
    # 保存真实数据
    np.save(os.path.join(WorkDir, "E_train_true.npy"), train_data["E"])
    np.save(os.path.join(WorkDir, "F_train_true.npy"), train_data["F"])
    np.save(os.path.join(WorkDir, "E_val_true.npy"), val_data["E"])
    np.save(os.path.join(WorkDir, "F_val_true.npy"), val_data["F"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练数据
    D_train = torch.tensor(train_data["D"], dtype=torch.float32, requires_grad=True, device=device)
    E_train = torch.tensor(train_data["E"], dtype=torch.float32, device=device)
    F_train = torch.tensor(train_data["F"], dtype=torch.float32, device=device)
    
    # 验证数据
    D_val = torch.tensor(val_data["D"], dtype=torch.float32, requires_grad=True, device=device)
    E_val = torch.tensor(val_data["E"], dtype=torch.float32, device=device)
    F_val = torch.tensor(val_data["F"], dtype=torch.float32, device=device)

    model = EnergyForceMLP(input_size=train_data["D"].shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    var_e = torch.var(E_train).item()
    var_f = torch.var(F_train.view(-1)).item()

    print("\n开始训练...")
    
    # 初始化最佳训练集性能指标
    best_train_metric = float('inf')  # 使用 ΔE 和 ΔF 的加权和作为指标
    best_epoch = -1

    # 收敛曲线数据
    train_metrics = []
    val_metrics = []

    # 新增：保存每次最佳模型记录
    best_model_records = []  # 用于记录每次保存最佳模型的信息

    for epoch in range(10001):
        # ======== 训练步骤 ========
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        pred_e, pred_f = model(D_train)
        
        # 计算能量损失
        energy_loss = F.mse_loss(pred_e, E_train) / var_e
        
        # 计算力损失
        force_loss = F.mse_loss(pred_f, F_train) / var_f
        
        # 总损失（可以调整权重）
        loss = energy_loss + force_loss
        loss.backward()
       
        # 梯度裁剪
        if epoch > 3000:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        # ======== 验证步骤 ========
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                # 训练集预测
                pred_e_train, pred_f_train = model(D_train)
                
                # 验证集预测
                pred_e_val, pred_f_val = model(D_val)

            # 打印监控信息
            with torch.no_grad():
                train_e_diff = (pred_e_train - E_train).abs().mean().item()
                train_f_diff = (pred_f_train - F_train).abs().mean().item()
                val_e_diff = (pred_e_val - E_val).abs().mean().item()
                val_f_diff = (pred_f_val - F_val).abs().mean().item()
                
                # 综合训练集性能指标（可以根据需求调整权重）
                train_metric = train_e_diff + train_f_diff
                val_metric = val_e_diff + val_f_diff

                # 保存收敛曲线数据
                train_metrics.append(train_metric)
                val_metrics.append(val_metric)

                print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.3e} | "
                      f"Energy Loss: {energy_loss.item():.3e} | Force Loss: {force_loss.item():.3e} | "
                      f"Train ΔE: {train_e_diff:.3f} | Train ΔF: {train_f_diff:.3f} | "
                      f"Val ΔE: {val_e_diff:.3f} | Val ΔF: {val_f_diff:.3f} | "
                      f"Train Metric: {train_metric:.3f} | Val Metric: {val_metric:.3f}")

                # 如果训练集性能更优，保存模型和预测结果
                if train_metric < best_train_metric:
                    best_train_metric = train_metric
                    best_epoch = epoch

                    # 保存最佳模型
                    torch.save(model.state_dict(), os.path.join(WorkDir, "best_model.pth"))

                    # 保存最佳训练集预测结果
                    np.save(os.path.join(WorkDir, "best_E_train_pred.npy"), pred_e_train.detach().cpu().numpy())
                    np.save(os.path.join(WorkDir, "best_F_train_pred.npy"), pred_f_train.detach().cpu().numpy())

                    # 新增：记录最佳模型信息
                    best_model_records.append({
                        "epoch": epoch,
                        "train_delta_e": train_e_diff,
                        "train_delta_f": train_f_diff,
                        "train_metric": train_metric
                    })

                    print(f"  -> 保存最佳模型和训练集预测结果 (Epoch {best_epoch}, Train Metric: {best_train_metric:.3f})")

            # 保存验证集预测结果
            np.save(os.path.join(WorkDir, "E_val_pred.npy"), pred_e_val.detach().cpu().numpy())
            np.save(os.path.join(WorkDir, "F_val_pred.npy"), pred_f_val.detach().cpu().numpy())

        # 全数据集预测
        if epoch % 5000 == 0 or epoch == 5000:
            with torch.no_grad():
                D_full = torch.tensor(full_data["D"], dtype=torch.float32, device=device)
                E_full_pred, F_full_pred = model(D_full)  # 获取能量和力预测
                np.save(os.path.join(WorkDir, f"E_full_pred_{epoch}.npy"), E_full_pred.cpu().numpy())
                np.save(os.path.join(WorkDir, f"F_full_pred_{epoch}.npy"), F_full_pred.cpu().numpy())  # 保存力预测

    # 保存收敛曲线数据
    np.save(os.path.join(WorkDir, "train_metrics.npy"), np.array(train_metrics))
    np.save(os.path.join(WorkDir, "val_metrics.npy"), np.array(val_metrics))

    # 新增：保存最佳模型记录
    np.save(os.path.join(WorkDir, "best_model_records.npy"), best_model_records)

if __name__ == "__main__":
    main()  