import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import cv2
import os
import sys
import seaborn as sns
import warnings
from datetime import datetime

# --- Global Configuration / 全局配置 ---

plt.rcParams.update({
    'font.size': 14,  # Default font size / 全局默认字体大小
    'axes.titlesize': 18,  # Title size / 图表标题大小
    'axes.labelsize': 16,  # Axis label size / 坐标轴标签大小
    'xtick.labelsize': 12,  # X-tick size / X轴刻度大小
    'ytick.labelsize': 12,  # Y-tick size / Y轴刻度大小
    'legend.fontsize': 12,  # Legend font size / 图例字体大小
    'figure.dpi': 100  # Figure resolution / 图片分辨率
})

# Set computation device / 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device} (使用的设备)")

# Ignore specific warnings / 忽略特定的警告信息
warnings.filterwarnings("ignore", "Plan failed with a cudnnException", category=UserWarning)


# --- 1. Data Processing & Dataset / 数据处理与数据集定义 ---

class MicroplasticDataset(Dataset):
    """
    Custom Dataset for Microplastic SERS data.
    自定义微塑料 SERS 数据集类。
    """

    def __init__(self, X, Y):
        """
        Args:
            X (np.ndarray): Input features (spectra). Shape: (N, L, 1) or (N, L).
            Y (np.ndarray): Labels.
        """
        # Transpose data to match Conv1d input requirement: (N, L, 1) -> (N, 1, L)
        # 转换数据维度以适配 Conv1d 输入: (N, L, 1) -> (N, 1, L)
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_preprocess_data(filepath):
    """
    Loads data from CSV, applies Savitzky-Golay baseline correction, and normalizes (0-1).
    加载 CSV 数据，应用 S-G 基线校正，并进行 0-1 归一化处理。

    Args:
        filepath (str): Path to the CSV data file.

    Returns:
        tuple: (X_processed, Y_raw, wavenumbers)
    """
    try:
        data_raw = pd.read_csv(filepath, header=None)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        sys.exit(1)

    # Parse wavenumbers, labels, and features
    # 解析波数、标签和特征数据
    wavenumbers = data_raw.iloc[0, 3:].astype(float).values
    Y_raw = data_raw.iloc[1:, 0:3].values.astype(int)
    X_raw = data_raw.iloc[1:, 3:].values.astype(float)
    X = X_raw.copy()

    # --- Savitzky-Golay Filter Configuration / S-G 滤波参数配置 ---
    WINDOW_LENGTH = 101
    POLY_ORDER = 3
    N_FEATURES = X.shape[1]

    # Adjust window size if features are fewer than default window
    # 如果特征数少于默认窗口，动态调整窗口大小
    if N_FEATURES < WINDOW_LENGTH:
        WINDOW_LENGTH = N_FEATURES - 1 if N_FEATURES % 2 == 0 else N_FEATURES - 2
    if WINDOW_LENGTH < 1: WINDOW_LENGTH = 1
    if WINDOW_LENGTH % 2 == 0: WINDOW_LENGTH += 1
    if WINDOW_LENGTH <= POLY_ORDER: POLY_ORDER = WINDOW_LENGTH - 1

    # Apply S-G Baseline Correction
    # 执行 S-G 基线校正
    if WINDOW_LENGTH > POLY_ORDER:
        print(f"Starting S-G baseline correction (Window={WINDOW_LENGTH}, Poly={POLY_ORDER})...")
        for i in range(X.shape[0]):
            baseline = savgol_filter(X[i, :], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, axis=0)
            X[i, :] = X[i, :] - baseline

    # Min-Max Normalization (0-1)
    # 最小-最大归一化 (0-1)
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)  # Add epsilon to avoid div by zero

    # Add channel dimension: (N, L) -> (N, L, 1)
    # 增加通道维度
    X = np.expand_dims(X, axis=-1)

    print(f"Data loaded and preprocessed. Samples: {X.shape[0]}, Features: {X.shape[1]}")
    return X, Y_raw, wavenumbers


def binary_to_categorical(Y_binary):
    """
    Converts binary labels [PS, PVC, PMMA] to a single categorical index (0-7).
    将二元标签转换为单一的类别索引。

    Mapping: [PS, PVC, PMMA] -> 4*PS + 2*PVC + 1*PMMA
    """
    categorical_labels = Y_binary[:, 0] * 4 + Y_binary[:, 1] * 2 + Y_binary[:, 2] * 1
    return categorical_labels


# --- 2. Model Architecture / 模型核心组件 ---

class GroupNorm(nn.GroupNorm):
    """
    Wrapper for Group Normalization with safe default groups.
    Group Normalization 的封装，包含安全默认分组设置。
    """

    def __init__(self, num_channels, num_groups=8, eps=1e-5, affine=True):
        if num_channels < num_groups:
            num_groups = max(1, num_channels // 2)
        super(GroupNorm, self).__init__(num_groups, num_channels, eps=eps, affine=affine)


class SmoothBCELoss(nn.Module):
    """
    BCE Loss with Label Smoothing.
    带有标签平滑的二元交叉熵损失函数。
    """

    def __init__(self, epsilon=0.1):
        super(SmoothBCELoss, self).__init__()
        self.epsilon = epsilon
        self.base_criterion = nn.BCELoss()

    def forward(self, pred, target):
        smooth_target = target * (1.0 - self.epsilon) + 0.5 * self.epsilon
        return self.base_criterion(pred, smooth_target)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) style Channel Attention Module.
    压缩-激励（SE）风格的通道注意力模块。
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class BasicBlock(nn.Module):
    """
    ResNet Basic Block with 1D Convolutions and optional Attention.
    基于一维卷积的 ResNet 基础块，可选配注意力机制。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5, use_attention=True, dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        # First convolution block
        # 第一层卷积块
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.gn1 = GroupNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)

        # Second convolution block
        # 第二层卷积块
        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.gn2 = GroupNorm(out_channels * self.expansion)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

        # Shortcut connection handling
        # 捷径连接处理 (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                GroupNorm(out_channels * self.expansion))

        self.attention = ChannelAttention(out_channels * self.expansion) if use_attention else None

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.dropout2(out)

        if self.attention is not None: out = self.attention(out)

        out += identity
        out = self.relu(out)
        return out


class AttentionResNet(nn.Module):
    """
    Multi-Branch ResNet with Attention for Multi-Label Classification.
    用于多标签分类的带注意力机制的多分支 ResNet。
    """

    def __init__(self, input_length, num_classes=3, base_channels=64, layers=[3, 3],
                 fc_dropout_rate=0.0, block_dropout_rate=0.0):
        super(AttentionResNet, self).__init__()
        self.in_channels = base_channels
        self.block_dropout_rate = block_dropout_rate

        # Initial Feature Extraction
        # 初始特征提取
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = GroupNorm(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        # ResNet 层级
        self.layer1 = self._make_layer(BasicBlock, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Calculate flattened size for Fully Connected layers
        # 计算全连接层的输入维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._forward_features(dummy_input)
            flat_size = x.numel() // x.shape[0]

        # Multi-branch Output Heads (One for each class)
        # 多分支输出头 (每个类别一个分支)
        self.branches = nn.ModuleList()
        for i in range(num_classes):
            branch = nn.Sequential(
                nn.Linear(flat_size, 64),
                nn.ReLU(),
                nn.Dropout(fc_dropout_rate),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.branches.append(branch)

        # Hooks for Grad-CAM
        # 用于 Grad-CAM 的钩子
        self.activations = None
        self.gradients = None

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, dropout_rate=self.block_dropout_rate))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, dropout_rate=self.block_dropout_rate))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        """Helper to forward pass until flattening. / 辅助函数：前向传播直到展平层。"""
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def save_gradient(self, grad):
        """Hook to capture gradients for Grad-CAM. / 捕捉梯度用于 Grad-CAM。"""
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Register hooks for Grad-CAM if gradients are required
        # 如果需要梯度，注册 Hook 以捕捉激活值和梯度
        if x.requires_grad:
            self.activations = x.clone()
            x.register_hook(self.save_gradient)
        else:
            self.activations = None
            self.gradients = None

        x = self.avgpool(x)
        x = self.flatten(x)

        # Compute output for each branch
        # 计算每个分支的输出
        outputs = [branch(x) for branch in self.branches]
        return outputs


MultiBranchCNN = AttentionResNet


# --- 3. Evaluation Functions / 评估函数 ---

def evaluate_model_on_real_data(model, data_loader):
    """
    Evaluates model on real data.
    在真实数据上评估模型。

    Returns:
        avg_loss, accuracies, overall_accuracy, Y_true, Y_prob, Y_pred
    """
    model.eval()
    criterion = nn.BCELoss()

    Y_true_all = []
    Y_prob_all = [[], [], []]
    Y_pred_all = [[], [], []]
    total_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)

            # Sum loss across all branches
            # 计算所有分支的损失总和
            loss = sum([criterion(outputs[i].squeeze(1), Y_batch[:, i]) for i in range(3)])
            total_loss += loss.item() * X_batch.size(0)

            Y_true_all.append(Y_batch.cpu().numpy())

            # Collect predictions
            # 收集预测结果
            for i in range(3):
                prob = outputs[i].squeeze(1).cpu().numpy()
                predicted = (outputs[i].squeeze(1) > 0.5).float().cpu().numpy()
                Y_prob_all[i].extend(prob)
                Y_pred_all[i].extend(predicted)

    Y_true_all = np.concatenate(Y_true_all, axis=0)
    Y_prob_all = np.array(Y_prob_all).T
    Y_pred_all_binary = np.array(Y_pred_all).T

    # Calculate metrics
    # 计算指标
    accuracies = [np.sum(Y_pred_all_binary[:, i] == Y_true_all[:, i]) / Y_true_all.shape[0] for i in range(3)]
    Y_pred_match_Y_true = np.all(Y_pred_all_binary == Y_true_all, axis=1)
    overall_accuracy = np.sum(Y_pred_match_Y_true) / Y_true_all.shape[0]
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss, accuracies, overall_accuracy, Y_true_all, Y_prob_all, Y_pred_all_binary


# --- 4. Plotting & Export Functions / 绘图与导出函数 ---

def plot_roc_curves(Y_true, Y_prob, save_dir):
    """
    Plots multi-branch ROC curves and saves data to CSV.
    绘制多分支 ROC 曲线并将数据保存为 CSV。
    """




MP_NAMES = ['PS', 'PVC', 'PMMA']
plt.figure(figsize=(10, 8))

roc_data = {'FPR': [], 'TPR': [], 'AUC': [], 'Branch': []}

# Plot per-branch ROC
# 绘制每个分支的 ROC
for i in range(Y_true.shape[1]):
    fpr, tpr, _ = roc_curve(Y_true[:, i], Y_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'ROC curve of {MP_NAMES[i]} (AUC = {roc_auc:.2f})')

    # Collect data for CSV
    # 收集数据以保存 CSV
    roc_data['FPR'].extend(fpr)
    roc_data['TPR'].extend(tpr)
    roc_data['AUC'].extend([roc_auc] * len(fpr))
    roc_data['Branch'].extend([MP_NAMES[i]] * len(fpr))

    # Plot Micro-average ROC
    # 绘制微平均 ROC
    fpr_micro, tpr_micro, _ = roc_curve(Y_true.ravel(), Y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})', linestyle=':', linewidth=4)

    roc_data['FPR'].extend(fpr_micro)
    roc_data['TPR'].extend(tpr_micro)
    roc_data['AUC'].extend([roc_auc_micro] * len(fpr_micro))
    roc_data['Branch'].extend(['Micro-average'] * len(fpr_micro))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Real Data Evaluation: Multi-Branch ROC')
    plt.legend(loc="lower right")

    roc_path = os.path.join(save_dir, 'real_data_roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"✅ ROC curve saved to {roc_path}")

    # Save to CSV
    # 保存至 CSV
    roc_df = pd.DataFrame(roc_data)
    roc_csv_path = os.path.join(save_dir, 'real_data_roc_curve_data.csv')
    roc_df.to_csv(roc_csv_path, index=False)
    print(f"✅ ROC data saved to CSV: {roc_csv_path}")


def plot_component_detection_matrix(Y_true_binary, Y_prob_all, save_dir):
    """
    Plots a 3x7 matrix of average predicted probabilities for each component combination.
    绘制 3x7 矩阵，展示每个组件组合的平均预测概率。
    """

    TARGET_CATEGORIES = {
        'PS': 4, 'PVC': 2, 'PMMA': 1, 'PS/PVC': 6,
        'PS/PMMA': 5, 'PVC/PMMA': 3, 'PS/PVC/PMMA': 7,
    }
    MP_NAMES = ['PS', 'PVC', 'PMMA']

    category_labels_ordered = [
        'PS', 'PVC', 'PMMA', 'PS/PVC',
        'PS/PMMA', 'PVC/PMMA', 'PS/PVC/PMMA'
    ]
    category_indices_ordered = [TARGET_CATEGORIES[label] for label in category_labels_ordered]

    probability_matrix = np.zeros((3, 7))
    Y_true_cat = binary_to_categorical(Y_true_binary)

    # Compute average probability for each category
    # 计算每个类别的平均概率
    for col_idx, cat_index in enumerate(category_indices_ordered):
        indices = np.where(Y_true_cat == cat_index)[0]

        if len(indices) == 0:
            continue

        Y_prob_sub = Y_prob_all[indices]

        for row_idx in range(3):
            avg_prob = np.mean(Y_prob_sub[:, row_idx])
            probability_matrix[row_idx, col_idx] = avg_prob

    # Plot Heatmap
    # 绘制热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(probability_matrix, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=category_labels_ordered, yticklabels=MP_NAMES,
                cbar_kws={'label': 'Average Predicted Probability'}, vmin=0.0, vmax=1.0)

    plt.title('Real Data Evaluation: Component Average Predicted Probability Matrix (3x7)')
    plt.xlabel('True Sample Category')
    plt.ylabel('Component Prediction Branch')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'real_data_component_average_probability_matrix_3x7.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ 3x7 Probability matrix saved to {cm_path}")

    # Save to CSV
    # 保存数据至 CSV
    prob_matrix_df = pd.DataFrame(probability_matrix,
                                  index=MP_NAMES,
                                  columns=category_labels_ordered)
    prob_csv_path = os.path.join(save_dir, 'real_data_3x7_probability_matrix_data.csv')
    prob_matrix_df.to_csv(prob_csv_path)
    print(f"✅ 3x7 Matrix data saved to CSV: {prob_csv_path}")


def generate_grad_cam(model, input_tensor, target_branch_index=0):
    """
    Generates Grad-CAM heatmap for a specific branch.
    为特定分支生成 Grad-CAM 热力图。
    """
    model.eval()
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad_(True)

    outputs = model(input_tensor)
    target_output = outputs[target_branch_index]
    model.zero_grad()

    # Backpropagation to get gradients
    # 反向传播以获取梯度
    target_output.backward(torch.ones_like(target_output), retain_graph=True)

    if model.gradients is None or model.activations is None:
        return np.zeros(input_tensor.shape[2]), target_output.cpu().item()

    gradients = model.gradients.data
    activations = model.activations.data

    # Pooling gradients to get weights
    # 对梯度进行池化以获得权重
    pooled_gradients = torch.mean(gradients, dim=2)
    weights = pooled_gradients.squeeze(0)

    activations = activations.squeeze(0)
    heatmap_channels = weights.unsqueeze(1) * activations
    heatmap = torch.sum(heatmap_channels, dim=0)
    heatmap = F.relu(heatmap)

    heatmap_np = heatmap.cpu().numpy()
    # Normalize to [0, 1]
    # 归一化至 [0, 1]
    if np.max(heatmap_np) > 0:
        heatmap_np = heatmap_np / np.max(heatmap_np)
    else:
        heatmap_np = np.zeros_like(heatmap_np)

    return heatmap_np, target_output.cpu().item()


def find_sample_indices_for_7_classes(Y_true_binary):
    """
    Finds one representative sample index for each of the 7 class combinations.
    为 7 种类别组合中的每一种找到一个代表性样本索引。
    """
    Y_true_cat = binary_to_categorical(Y_true_binary)
    target_indices = np.arange(1, 8)  # 1 to 7 (exclude None=0)
    sample_indices = {}

    for cat_idx in target_indices:
        indices = np.where(Y_true_cat == cat_idx)[0]
        if len(indices) > 0:
            sample_indices[cat_idx] = indices[0]

    return sample_indices


def plot_and_save_grad_cam_7_classes(model, test_dataset, wavenumbers, Y_true_binary, save_dir):
    """
    Generates and saves Grad-CAM plots and CSV data for representative samples.
    生成并保存代表性样本的 Grad-CAM 图表和 CSV 数据。
    """

    sample_indices = find_sample_indices_for_7_classes(Y_true_binary)

    CLASS_LABELS = [
        "None", "PMMA", "PVC", "PVC/PMMA",
        "PS", "PS/PMMA", "PS/PVC", "PS/PVC/PMMA"
    ]
    MP_NAMES = ['PS', 'PVC', 'PMMA']

    print(f"\n--- Generating Grad-CAM for {len(sample_indices)} representative samples ---")

    for cat_idx, data_idx_in_test_set in sample_indices.items():
        class_name = CLASS_LABELS[cat_idx]

        # Get sample data
        # 获取样本数据
        X_test_sample_raw, Y_test_sample = test_dataset[data_idx_in_test_set]
        test_input_tensor = X_test_sample_raw.unsqueeze(0).to(device)

        spectrum_data = X_test_sample_raw.squeeze().cpu().numpy()

        # Prepare CSV data dict
        # 准备 CSV 数据字典
        cam_data = {'Wavenumber': wavenumbers, 'Spectrum': spectrum_data}

        plt.figure(figsize=(12, 6))
        plt.title(f'Grad-CAM: Sample from Class {class_name}')

        plt.xlabel('Raman Shift ($cm^{-1}$)')
        plt.ylabel('Normalized Intensity')
        plt.plot(wavenumbers, spectrum_data, label='Normalized SERS Spectrum', color='black', alpha=0.8)

        for i in range(3):
            heatmap_np, prediction_prob = generate_grad_cam(model, test_input_tensor, target_branch_index=i)

            # Resize heatmap to match spectrum length
            # 调整热力图大小以匹配光谱长度
            heatmap_resized = cv2.resize(heatmap_np.reshape(-1, 1),
                                         (1, wavenumbers.shape[0]),
                                         interpolation=cv2.INTER_LINEAR).squeeze()

            # Visualize CAM scaled by spectrum intensity
            # 将 CAM 数值化（缩放到谱线幅度范围以便绘图）
            heatmap_y_pos = np.max(spectrum_data)
            cam_visualization = heatmap_resized * heatmap_y_pos

            plt.fill_between(wavenumbers, 0, cam_visualization,
                             alpha=0.3,
                             label=f'{MP_NAMES[i]} Focus (Pred: {prediction_prob:.2f})')

            # Save raw CAM data
            # 保存原始 CAM 数据
            cam_data[f'CAM_{MP_NAMES[i]}'] = heatmap_resized

        cam_path = os.path.join(save_dir, f'real_data_grad_cam_class_{class_name.replace("/", "_")}.png')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(cam_path)
        plt.close()

        # Save to CSV
        # 保存 Grad-CAM 数据至 CSV
        cam_df = pd.DataFrame(cam_data)
        cam_csv_path = os.path.join(save_dir, f'real_data_grad_cam_data_{class_name.replace("/", "_")}.csv')
        cam_df.to_csv(cam_csv_path, index=False)
        print(f"  - ✅ Grad-CAM data for {class_name} saved to CSV: {cam_csv_path}")


def run_evaluation_pipeline(model_path, data_filepath, input_length, output_save_dir):
    """
    Executes the full evaluation pipeline.
    执行完整的评估流程。

    Steps:
    1. Load Data / 加载数据
    2. Load Model / 加载模型
    3. Evaluate / 评估
    4. Plot & Save / 绘图与保存
    """
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        sys.exit(1)

    # Configuration (Must match training config)
    # 配置 (必须与训练配置一致)
    BASE_CHANNELS = 64
    LAYERS = [3, 3]
    BLOCK_DROPOUT = 0.0
    FC_DROPOUT = 0.0

    print(f"\n--- Starting Evaluation on '{data_filepath}' ---")
    print(f"Model Config: AttentionResNet (Base Channels={BASE_CHANNELS}, Layers={LAYERS})")

    # 1. Load and preprocess
    # 1. 加载和预处理
    X_real_np, Y_real_np, W = load_and_preprocess_data(data_filepath)

    # 2. Initialize and load model
    # 2. 初始化并加载模型
    model = MultiBranchCNN(
        input_length,
        num_classes=3,
        base_channels=BASE_CHANNELS,
        layers=LAYERS,
        block_dropout_rate=BLOCK_DROPOUT,
        fc_dropout_rate=FC_DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    print(f"✅ Model weights loaded from {model_path}")

    # 3. Create DataLoader
    # 3. 创建数据加载器
    real_dataset = MicroplasticDataset(X_real_np, Y_real_np)
    real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)

    # 4. Run evaluation
    # 4. 运行评估
    avg_loss, accs, overall_acc, Y_true_binary, Y_prob_all, Y_pred_binary = evaluate_model_on_real_data(
        model, real_loader
    )

    # 5. Plotting and Export
    # 5. 绘图和 CSV 导出
    plot_roc_curves(Y_true_binary, Y_prob_all, output_save_dir)
    plot_component_detection_matrix(Y_true_binary, Y_prob_all, output_save_dir)
    plot_and_save_grad_cam_7_classes(model, real_dataset, W, Y_true_binary, output_save_dir)

    # 6. Summary
    # 6. 结果总结
    acc_ps, acc_pvc, acc_pmma = accs
    final_component_avg_acc = np.mean(accs)

    print(f"\n--- Evaluation Summary ---")
    print(f"Total Samples: {Y_true_binary.shape[0]}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"PS Accuracy: {acc_ps * 100:.2f}%")
    print(f"PVC Accuracy: {acc_pvc * 100:.2f}%")
    print(f"PMMA Accuracy: {acc_pmma * 100:.2f}%")
    print(f"Avg Branch Accuracy: {final_component_avg_acc * 100:.2f}%")
    print(f"==================================================")
    print(f"✅ Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"All plots and CSV files saved to: {output_save_dir}")


# --- 5. Main Execution Block / 主执行模块 ---

if __name__ == '__main__':
    # ----------------------------------------------------
    # 🚨 Configuration Section / 配置区域 🚨
    # Please modify these paths according to your environment.
    # 请根据您的实际情况修改以下路径。
    # ----------------------------------------------------

    # 1. Path to trained model weights
    # 1. 训练好的模型权重文件路径
    MODEL_WEIGHTS_PATH = 'model_output_kfold_cv_xxx/best_model_fold_x.pth'

    # 2. Path to real test data (CSV)
    # 2. 真实数据集文件路径
    REAL_DATA_FILEPATH = 'test_data_compiled.csv'

    # 3. Model input length (Must match training)
    # 3. 模型输入长度 (必须与训练时一致，如 1015)
    MODEL_INPUT_LENGTH = 1015

    # 4. Output directory for results
    # 4. 结果保存目录
    SAVE_DIR = f'real_data_evaluation_results__kfold_cv3'
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ----------------------------------------------------

    run_evaluation_pipeline(
        MODEL_WEIGHTS_PATH,
        REAL_DATA_FILEPATH,
        MODEL_INPUT_LENGTH,
        SAVE_DIR
    )