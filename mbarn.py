import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import cv2
import os
import warnings
import sys
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime

# --- Configuration & Styling / 配置与样式 ---
plt.rcParams.update({
    'font.size': 18,  # Global font size / 全局字体大小
    'axes.titlesize': 22,  # Title size / 标题大小
    'axes.labelsize': 20,  # Axis label size / 轴标签大小
    'xtick.labelsize': 16,  # X-axis tick size / X轴刻度大小
    'ytick.labelsize': 16,  # Y-axis tick size / Y轴刻度大小
    'legend.fontsize': 16,  # Legend font size / 图例字体大小
    'axes.labelweight': 'bold',  # Bold labels / 标签加粗
    'figure.dpi': 300  # High DPI for publication / 高分辨率
})

# Set computation device / 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {device} / 使用设备: {device}")

# Suppress non-fatal cuDNN warnings / 抑制非致命的 cuDNN 警告
warnings.filterwarnings("ignore", "Plan failed with a cudnnException", category=UserWarning)


# --- 1. Data Processing & Dataset / 数据处理与数据集定义 ---

class MicroplasticDataset(Dataset):
    """
    Custom Dataset for Microplastic SERS data.
    自定义微塑料 SERS 数据集。
    """

    def __init__(self, X, Y):
        # Convert dimensions / 转换维度:
        # X: (N, L, 1) -> (N, 1, L) for Conv1d compatibility
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_preprocess_data(filepath):
    """
    Load and preprocess data: Savitzky-Golay baseline correction and Min-Max normalization.
    加载并预处理数据：S-G 基线校正和 0-1 归一化。
    """
    try:
        data_raw = pd.read_csv(filepath, header=None)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath} / 错误：找不到文件路径")
        sys.exit(1)

    # Data loading logic (Assuming header at row 0, data starts at row 1)
    # 数据加载逻辑（假设第0行为表头，第1行开始为数据）
    wavenumbers = data_raw.iloc[0, 3:].astype(float).values
    Y_raw = data_raw.iloc[1:, 0:3].values.astype(int)
    X_raw = data_raw.iloc[1:, 3:].values.astype(float)
    X = X_raw.copy()

    # Baseline correction (Savitzky-Golay) / 基线校正 (S-G)
    WINDOW_LENGTH = 101
    POLY_ORDER = 3
    N_FEATURES = X.shape[1]

    # Adjust window length dynamically / 动态调整窗口长度
    if N_FEATURES < WINDOW_LENGTH:
        WINDOW_LENGTH = N_FEATURES - 1 if N_FEATURES % 2 == 0 else N_FEATURES - 2
    if WINDOW_LENGTH < 1: WINDOW_LENGTH = 1
    if WINDOW_LENGTH % 2 == 0: WINDOW_LENGTH += 1
    if WINDOW_LENGTH <= POLY_ORDER: POLY_ORDER = WINDOW_LENGTH - 1

    if WINDOW_LENGTH > POLY_ORDER:
        print(
            f"Starting S-G baseline correction for {X.shape[0]} samples (Window={WINDOW_LENGTH}, Poly={POLY_ORDER})...")
        print(f"开始对 {X.shape[0]} 个样本进行 S-G 基线校正...")
        for i in range(X.shape[0]):
            baseline = savgol_filter(X[i, :], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, axis=0)
            X[i, :] = X[i, :] - baseline

    # Normalization (Min-Max 0-1) / 归一化
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    X = np.expand_dims(X, axis=-1)  # Shape: (N, L, 1)

    print(f"Data processing complete. Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"数据加载和预处理完成。样本总数: {X.shape[0]}, 特征点数: {X.shape[1]}")
    return X, Y_raw, wavenumbers


# --- 2. Core Components / 核心组件 (GroupNorm, Loss) ---

class GroupNorm(nn.GroupNorm):
    """
    Wrapper for GroupNorm with dynamic group calculation.
    具有动态组计算功能的 GroupNorm 封装。
    """

    def __init__(self, num_channels, num_groups=8, eps=1e-5, affine=True):
        if num_channels < num_groups:
            num_groups = max(1, num_channels // 2)
        super(GroupNorm, self).__init__(num_groups, num_channels, eps=eps, affine=affine)


class SmoothBCELoss(nn.Module):
    """
    Binary Cross Entropy Loss with Label Smoothing.
    带有标签平滑的二元交叉熵损失函数。
    """

    def __init__(self, epsilon=0.1):
        super(SmoothBCELoss, self).__init__()
        self.epsilon = epsilon
        self.base_criterion = nn.BCELoss()

    def forward(self, pred, target):
        # target * (1 - eps) + 0.5 * eps
        smooth_target = target * (1.0 - self.epsilon) + 0.5 * self.epsilon
        return self.base_criterion(pred, smooth_target)


# --- 3. Model Definition / 模型定义 (Attention ResNet) ---

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
    Residual Block with GroupNorm and Optional Attention.
    包含 GroupNorm 和可选注意力机制的残差块。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5, use_attention=True, dropout_rate=0.0):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.gn1 = GroupNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout1d(p=dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2, bias=False)
        self.gn2 = GroupNorm(out_channels * self.expansion)
        self.dropout2 = nn.Dropout1d(p=dropout_rate)

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

        if self.attention is not None:
            out = self.attention(out)

        out += identity
        out = self.relu(out)
        return out


class AttentionResNet(nn.Module):
    """
    Main Model Architecture: 1D ResNet with Multi-Branch Output.
    主模型架构：带有注意力机制和多分支输出的 1D ResNet。
    """

    def __init__(self, input_length, num_classes=3, base_channels=64, layers=[3, 3],
                 fc_dropout_rate=0.0, block_dropout_rate=0.0):
        super(AttentionResNet, self).__init__()
        self.in_channels = base_channels
        self.block_dropout_rate = block_dropout_rate

        # Initial layers / 初始层
        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = GroupNorm(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers / 残差层
        self.layer1 = self._make_layer(BasicBlock, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, layers[1], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # Calculate flattened size dynamically / 动态计算扁平化后的尺寸
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._forward_features(dummy_input)
            flat_size = x.numel() // x.shape[0]

        print(f"DEBUG: Feature size after main body (flat_size): {flat_size}")

        # Multi-branch output heads (one for each class) / 多分支输出头（每个类别一个）
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

        # For Grad-CAM hooks / 用于 Grad-CAM 的钩子
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
        self.gradients = grad

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Register hooks for Grad-CAM if gradients are required
        # 如果需要计算梯度，则注册 Grad-CAM 钩子
        if x.requires_grad:
            self.activations = x.clone()
            x.register_hook(self.save_gradient)
        else:
            self.activations = None
            self.gradients = None

        x = self.avgpool(x)
        x = self.flatten(x)

        outputs = [branch(x) for branch in self.branches]
        return outputs


MultiBranchCNN = AttentionResNet


# --- 4. Training & Evaluation / 训练与评估 ---

def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model.
    评估模型，返回损失、准确率、F1分数等指标。
    """
    model.eval()
    Y_true_all = []
    Y_prob_all = [[], [], []]
    Y_pred_all = [[], [], []]
    total_loss = 0

    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)

            # Calculate multi-branch loss / 计算多分支损失
            loss = sum([criterion(outputs[i].squeeze(1), Y_batch[:, i]) for i in range(3)])

            total_loss += loss.item() * X_batch.size(0)
            Y_true_all.append(Y_batch.cpu().numpy())

            for i in range(3):
                prob = outputs[i].squeeze(1).cpu().numpy()
                predicted = (outputs[i].squeeze(1) > 0.5).float().cpu().numpy()
                Y_prob_all[i].extend(prob)
                Y_pred_all[i].extend(predicted)

    Y_true_all = np.concatenate(Y_true_all, axis=0)
    Y_prob_all = np.array(Y_prob_all).T
    Y_pred_all_binary = np.array(Y_pred_all).T

    # Per-branch accuracy / 单分支准确率
    accuracies = [np.sum(Y_pred_all_binary[:, i] == Y_true_all[:, i]) / Y_true_all.shape[0] for i in range(3)]

    # F1 Score (handle division by zero) / F1 分数 (处理分母为零)
    f1_scores = [f1_score(Y_true_all[:, i], Y_pred_all_binary[:, i], zero_division=0) for i in range(3)]
    avg_f1 = np.mean(f1_scores)

    # Overall accuracy (all branches correct) / 总体准确率 (所有标签全对)
    Y_pred_match_Y_true = np.all(Y_pred_all_binary == Y_true_all, axis=1)
    overall_accuracy = np.sum(Y_pred_match_Y_true) / Y_true_all.shape[0]
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss, accuracies, overall_accuracy, f1_scores, avg_f1, Y_true_all, Y_prob_all, Y_pred_all_binary


def train_model(model, fold_index, train_loader, val_loader, epochs=150, save_dir='.', label_smoothing_epsilon=0.35):
    """
    Train a single fold model and print performance report.
    训练单折模型，并在结束时打印性能报告。
    """
    train_criterion = SmoothBCELoss(epsilon=label_smoothing_epsilon)
    eval_criterion = SmoothBCELoss(epsilon=label_smoothing_epsilon)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_acc_ps': [], 'val_acc_pvc': [], 'val_acc_pmma': [],
               'overall_acc': []}
    best_overall_acc = 0.0
    best_model_path = os.path.join(save_dir, f'best_model_fold_{fold_index}.pth')

    print(f"\n--- Fold {fold_index}: Start Training / 开始训练 ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = sum([train_criterion(outputs[i].squeeze(1), Y_batch[:, i]) for i in range(3)])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        scheduler.step()

        # Evaluate / 评估
        val_loss, val_accs, overall_acc, _, _, _, _, _ = evaluate_model(model, val_loader, eval_criterion)

        history['train_loss'].append(total_train_loss / len(train_loader.dataset))
        history['val_loss'].append(val_loss)
        history['val_acc_ps'].append(val_accs[0])
        history['val_acc_pvc'].append(val_accs[1])
        history['val_acc_pmma'].append(val_accs[2])
        history['overall_acc'].append(overall_acc)

        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 50 == 0 or epoch == epochs:
            print(
                f"Fold {fold_index} Epoch {epoch}/{epochs}: Loss {val_loss:.4f} | Overall Acc: {overall_acc * 100:.2f}%")

    # Final report for this fold / 本折的最终报告
    print(f"\n--- Fold {fold_index} Training Complete, Best Performance Report / 训练完成，最佳性能报告 ---")
    model.load_state_dict(torch.load(best_model_path))

    # Use standard BCELoss for final evaluation metric
    # 使用标准 BCELoss 进行最终评估
    f_loss, f_accs, f_overall, f_f1, f_f1_avg, _, _, _ = evaluate_model(model, val_loader, nn.BCELoss())

    MP_NAMES = ['PS', 'PVC', 'PMMA']
    for i in range(3):
        print(f"  > {MP_NAMES[i]:<4} Accuracy: {f_accs[i] * 100:>6.2f}% | F1-Score: {f_f1[i]:.4f}")
    print(f"  > Mean F1: {f_f1_avg:.4f}")
    print(f"  > ⭐ Overall Accuracy (All Correct): {f_overall * 100:.2f}%")
    print("-" * 40)

    return history, best_model_path, best_overall_acc


# --- 5. Data Export / 数据导出 (CSV) ---

def save_history_to_csv(all_histories, save_dir, file_name='kfold_training_metrics.csv'):
    """
    Save training history of all folds to CSV.
    将所有折叠的训练历史记录保存到 CSV 文件。
    """
    all_dfs = []

    # Map internal keys to display names / 映射内部键名到显示名称
    metric_map = {
        'train_loss': 'Train Loss',
        'val_loss': 'Validation Loss',
        'val_acc_ps': 'PS Acc',
        'val_acc_pvc': 'PVC Acc',
        'val_acc_pmma': 'PMMA Acc',
        'overall_acc': 'Overall Acc'
    }

    for fold, history in all_histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        df_data = {'Epoch': list(epochs)}
        for key, display_name in metric_map.items():
            column_name = f'Fold_{fold} - {display_name}'
            df_data[column_name] = history.get(key, [])

        df = pd.DataFrame(df_data)
        all_dfs.append(df)

    # Merge all DataFrames / 合并所有 DataFrame
    final_df = all_dfs[0]
    for i in range(1, len(all_dfs)):
        final_df = pd.merge(final_df, all_dfs[i].drop(columns=['Epoch']),
                            left_index=True, right_index=True)

    csv_path = os.path.join(save_dir, file_name)
    final_df.to_csv(csv_path, index=False)
    print(f"\n✅ All training metrics saved to CSV / 所有训练指标已保存至: {csv_path}")


# --- 6. Visualization & Grad-CAM / 绘图与 Grad-CAM ---

def plot_history(history, save_dir, fold_index):
    """
    Plot and save Loss and Accuracy curves.
    绘制并保存 Loss 和 Accuracy 曲线。
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold_index} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCELoss Sum)')
    plt.legend()
    loss_path = os.path.join(save_dir, f'best_fold_{fold_index}_loss_curve.png')
    plt.savefig(loss_path)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [a * 100 for a in history['val_acc_ps']], label='PS Accuracy')
    plt.plot(epochs, [a * 100 for a in history['val_acc_pvc']], label='PVC Accuracy')
    plt.plot(epochs, [a * 100 for a in history['val_acc_pmma']], label='PMMA Accuracy')
    plt.plot(epochs, [a * 100 for a in history['overall_acc']], label='Overall Accuracy (All Labels Correct)',
             linestyle='--', color='red')
    plt.title(f'Fold {fold_index} Validation Accuracy per Branch and Overall Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    acc_path = os.path.join(save_dir, f'best_fold_{fold_index}_accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()


def plot_roc_curves(Y_true, Y_prob, save_dir, fold_index):
    """
    Calculate and plot ROC curves, save data to CSV.
    计算、绘制 ROC 曲线，并保存数据至 CSV。
    """
    MP_NAMES = ['PS', 'PVC', 'PMMA']
    plt.figure(figsize=(10, 8))

    roc_data = {'FPR': [], 'TPR': [], 'AUC': [], 'Branch': []}

    for i in range(Y_true.shape[1]):
        fpr, tpr, _ = roc_curve(Y_true[:, i], Y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'ROC curve of {MP_NAMES[i]} (AUC = {roc_auc:.2f})')

        roc_data['FPR'].extend(fpr)
        roc_data['TPR'].extend(tpr)
        roc_data['AUC'].extend([roc_auc] * len(fpr))
        roc_data['Branch'].extend([MP_NAMES[i]] * len(fpr))

    # Micro-average ROC
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
    plt.title(f'Fold {fold_index} Multi-Branch Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    roc_path = os.path.join(save_dir, f'best_fold_{fold_index}_multi_label_roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC Curve saved to {roc_path}")

    roc_df = pd.DataFrame(roc_data)
    roc_csv_path = os.path.join(save_dir, f'best_fold_{fold_index}_roc_curve_data.csv')
    roc_df.to_csv(roc_csv_path, index=False)
    print(f"✅ ROC Data saved to CSV / ROC 数据已保存: {roc_csv_path}")


def binary_to_categorical(Y_binary):
    """
    Convert (N, 3) binary labels to (N,) categorical indices (0-7).
    将 (N, 3) 二元标签 [PS, PVC, PMMA] 转换为 (N,) 类别索引 (0-7)。
    """
    categorical_labels = Y_binary[:, 0] * 4 + Y_binary[:, 1] * 2 + Y_binary[:, 2] * 1
    return categorical_labels


def plot_component_detection_matrix(Y_true_binary, Y_prob_all, save_dir, fold_index):
    """
    Plot average prediction probability matrix for 3x7 components.
    绘制 3x7 组件的平均预测概率矩阵。
    """
    TARGET_CATEGORIES = {
        'PS': 4, 'PVC': 2, 'PMMA': 1, 'PS/PVC': 6,
        'PS/PMMA': 5, 'PVC/PMMA': 3, 'PS/PVC/PMMA': 7,
    }
    MP_NAMES = ['PS', 'PVC', 'PMMA']
    category_labels_ordered = ['PS', 'PVC', 'PMMA', 'PS/PVC', 'PS/PMMA', 'PVC/PMMA', 'PS/PVC/PMMA']
    category_indices_ordered = [TARGET_CATEGORIES[label] for label in category_labels_ordered]

    probability_matrix = np.zeros((3, 7))
    Y_true_cat = binary_to_categorical(Y_true_binary)

    print(f"\n--- Fold {fold_index}: Calculating Component Probability Matrix (3x7) / 计算 3x7 概率矩阵 ---")

    for col_idx, cat_index in enumerate(category_indices_ordered):
        indices = np.where(Y_true_cat == cat_index)[0]

        if len(indices) == 0:
            continue

        Y_prob_sub = Y_prob_all[indices]
        for row_idx in range(3):
            avg_prob = np.mean(Y_prob_sub[:, row_idx])
            probability_matrix[row_idx, col_idx] = avg_prob

    # Plot Heatmap / 绘制热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(probability_matrix, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=category_labels_ordered, yticklabels=MP_NAMES,
                cbar_kws={'label': 'Average Predicted Probability'}, vmin=0.0, vmax=1.0)

    plt.title(f'Fold {fold_index} Component Average Predicted Probability Matrix (3x7)')
    plt.xlabel('True Sample Category')
    plt.ylabel('Component Prediction Branch')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, f'best_fold_{fold_index}_component_average_probability_matrix_3x7.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Probability Matrix saved to {cm_path}")

    # Save CSV
    prob_matrix_df = pd.DataFrame(probability_matrix, index=MP_NAMES, columns=category_labels_ordered)
    prob_csv_path = os.path.join(save_dir, f'best_fold_{fold_index}_3x7_probability_matrix_data.csv')
    prob_matrix_df.to_csv(prob_csv_path)
    print(f"✅ Matrix Data saved to CSV / 矩阵数据已保存: {prob_csv_path}")


def generate_grad_cam(model, input_tensor, target_branch_index=0):
    """
    Generate Grad-CAM heatmap.
    生成 Grad-CAM 热力图。
    """
    model.eval()
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad_(True)

    outputs = model(input_tensor)
    target_output = outputs[target_branch_index]
    model.zero_grad()
    target_output.backward(torch.ones_like(target_output), retain_graph=True)

    if model.gradients is None or model.activations is None:
        return np.zeros(input_tensor.shape[2]), target_output.cpu().item()

    gradients = model.gradients.data
    activations = model.activations.data

    # Pooling gradients / 梯度池化
    pooled_gradients = torch.mean(gradients, dim=2)
    weights = pooled_gradients.squeeze(0)

    activations = activations.squeeze(0)
    heatmap_channels = weights.unsqueeze(1) * activations
    heatmap = torch.sum(heatmap_channels, dim=0)
    heatmap = F.relu(heatmap)

    heatmap_np = heatmap.cpu().numpy()

    # Denoising Logic / 去噪逻辑
    # 1. Power-CAM enhancement (Squared) / 指数增强 (平方)
    heatmap_np = np.power(heatmap_np, 2)

    if np.max(heatmap_np) > 0:
        heatmap_np = heatmap_np / np.max(heatmap_np)

    # 2. Hard threshold filtering / 硬阈值过滤
    heatmap_np[heatmap_np < 0.2] = 0

    return heatmap_np, target_output.cpu().item()


def find_sample_indices_for_7_classes(Y_true_binary):
    """
    Find one representative sample index for each of the 7 classes.
    找到 7 个类别的代表性样本索引。
    """
    Y_true_cat = binary_to_categorical(Y_true_binary)
    target_indices = np.arange(1, 8)  # 1 to 7 (Exclude None=0)
    sample_indices = {}

    for cat_idx in target_indices:
        indices = np.where(Y_true_cat == cat_idx)[0]
        if len(indices) > 0:
            sample_indices[cat_idx] = indices[0]

    return sample_indices


def plot_and_save_grad_cam_7_classes(model, test_dataset, wavenumbers, Y_true_binary, save_dir, fold_index):
    """
    Generate and save Grad-CAM plots and data for 7 representative samples.
    生成并保存 7 个代表性分类样本的 Grad-CAM 图表及数据。
    """
    sample_indices = find_sample_indices_for_7_classes(Y_true_binary)

    CLASS_LABELS = [
        "None", "PMMA", "PVC", "PVC/PMMA",
        "PS", "PS/PMMA", "PS/PVC", "PS/PVC/PMMA"
    ]
    MP_NAMES = ['PS', 'PVC', 'PMMA']

    print(f"\n--- Fold {fold_index}: Generating Grad-CAM for {len(sample_indices)} samples / 正在生成样本热力图 ---")

    for cat_idx, data_idx_in_test_set in sample_indices.items():
        class_name = CLASS_LABELS[cat_idx]
        X_test_sample_raw, Y_test_sample = test_dataset[data_idx_in_test_set]
        test_input_tensor = X_test_sample_raw.unsqueeze(0).to(device)
        spectrum_data = X_test_sample_raw.squeeze().cpu().numpy()

        cam_data = {'Wavenumber': wavenumbers, 'Spectrum': spectrum_data}

        plt.figure(figsize=(12, 6))
        plt.title(f'Fold {fold_index} Grad-CAM: Sample from Class {class_name}')
        plt.xlabel('Raman Shift ($cm^{-1}$)')
        plt.ylabel('Normalized Intensity')

        # Plot original spectrum / 绘制原始光谱
        plt.plot(wavenumbers, spectrum_data, label='Normalized SERS Spectrum', color='black', alpha=0.8)

        for i in range(3):
            heatmap_np, prediction_prob = generate_grad_cam(model, test_input_tensor, target_branch_index=i)

            # Resize heatmap to match spectrum length / 将热力图调整至光谱长度
            heatmap_resized = cv2.resize(heatmap_np.reshape(-1, 1),
                                         (1, wavenumbers.shape[0]),
                                         interpolation=cv2.INTER_LINEAR).squeeze()

            # Visualization scaling / 可视化缩放
            heatmap_y_pos = np.max(spectrum_data)
            cam_visualization = heatmap_resized * heatmap_y_pos

            plt.fill_between(wavenumbers, 0, cam_visualization,
                             alpha=0.3,
                             label=f'{MP_NAMES[i]} Focus (Pred: {prediction_prob:.2f})')

            cam_data[f'CAM_{MP_NAMES[i]}'] = heatmap_resized

        cam_path = os.path.join(save_dir, f'best_fold_{fold_index}_grad_cam_class_{class_name.replace("/", "_")}.png')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(cam_path)
        plt.close()

        # Save CAM data to CSV / 保存 CAM 数据
        cam_df = pd.DataFrame(cam_data)
        cam_csv_path = os.path.join(save_dir,
                                    f'best_fold_{fold_index}_grad_cam_data_{class_name.replace("/", "_")}.csv')
        cam_df.to_csv(cam_csv_path, index=False)
        print(f"  - ✅ Grad-CAM Data saved / 数据已保存: {cam_csv_path}")


# --- 7. Main Execution / 主执行模块 (K-Fold CV) ---

if __name__ == '__main__':
    # Setup directories / 设置目录
    SAVE_DIR = f'model_output_kfold_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(SAVE_DIR, exist_ok=True)

    FILE_PATH = 'batch_spectra.csv'
    N_SPLITS = 5  # 5-Fold / 五折
    BATCH_SIZE = 64
    EPOCHS = 150
    RANDOM_SEED = 42

    # Model Parameters / 模型参数
    BASE_CHANNELS = 64
    LAYERS = [3, 3]
    FC_DROPOUT = 0.0
    BLOCK_DROPOUT = 0.0
    LABEL_SMOOTHING_EPSILON = 0.1

    print(f"\n--- 5-Fold Cross-Validation Setup / 五折交叉验证设置 ---")
    print(f"Splits: {N_SPLITS}, Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Loss: Label Smoothing (Epsilon: {LABEL_SMOOTHING_EPSILON})")

    # A. Load Data / 加载数据
    X_all_np, Y_all_np, W = load_and_preprocess_data(FILE_PATH)
    INPUT_LENGTH = X_all_np.shape[1]

    # K-Fold Split / 划分 K-Fold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    all_histories = {}
    best_overall_acc_cv = -1.0
    best_fold_index = -1
    best_fold_model_path = None
    best_fold_val_indices = None

    # Iterate over folds / 遍历所有折叠
    for fold, (train_indices, val_indices) in enumerate(kfold.split(X_all_np)):
        fold_index = fold + 1

        # 1. Split Data / 数据划分
        X_train_fold, X_val_fold = X_all_np[train_indices], X_all_np[val_indices]
        Y_train_fold, Y_val_fold = Y_all_np[train_indices], Y_all_np[val_indices]

        # 2. Dataset & DataLoader
        train_dataset = MicroplasticDataset(X_train_fold, Y_train_fold)
        val_dataset = MicroplasticDataset(X_val_fold, Y_val_fold)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 3. Create Model (Fresh instance per fold) / 创建模型 (每折需重新实例化)
        model = MultiBranchCNN(
            INPUT_LENGTH,
            num_classes=3,
            base_channels=BASE_CHANNELS,
            layers=LAYERS,
            fc_dropout_rate=FC_DROPOUT,
            block_dropout_rate=BLOCK_DROPOUT
        ).to(device)

        # 4. Train / 训练
        training_history, current_best_model_path, current_best_acc = train_model(
            model,
            fold_index,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            save_dir=SAVE_DIR,
            label_smoothing_epsilon=LABEL_SMOOTHING_EPSILON
        )

        # 5. Record History / 记录历史
        all_histories[fold_index] = training_history

        # 6. Update Best Fold / 更新最佳折叠
        if current_best_acc > best_overall_acc_cv:
            best_overall_acc_cv = current_best_acc
            best_fold_index = fold_index
            best_fold_model_path = current_best_model_path
            best_fold_val_indices = val_indices

        print(f"Fold {fold_index} Complete. Best Overall Acc: {current_best_acc * 100:.2f}%")

    print(f"\n==================================================")
    print(f"✅ 5-Fold CV Complete. Best Fold: {best_fold_index}, Acc: {best_overall_acc_cv * 100:.2f}%")
    print(f"==================================================")

    # --- 8. Export All Folds History / 导出所有历史数据 ---
    save_history_to_csv(all_histories, SAVE_DIR)

    # --- 9. Final Evaluation of Best Fold / 最佳折叠的最终评估 ---

    # Load Best Model / 加载最佳模型
    best_model = MultiBranchCNN(
        INPUT_LENGTH,
        num_classes=3,
        base_channels=BASE_CHANNELS,
        layers=LAYERS,
        fc_dropout_rate=FC_DROPOUT,
        block_dropout_rate=BLOCK_DROPOUT
    ).to(device)
    best_model.load_state_dict(torch.load(best_fold_model_path))

    # Prepare Best Validation Data / 准备最佳验证集
    X_best_val, Y_best_val = X_all_np[best_fold_val_indices], Y_all_np[best_fold_val_indices]
    best_val_dataset = MicroplasticDataset(X_best_val, Y_best_val)
    best_val_loader = DataLoader(best_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation / 评估
    test_loss_bce, test_accs_bce, overall_accuracy_bce, f1_list, f1_avg, Y_true_binary, Y_prob_all, Y_pred_binary = evaluate_model(
        best_model, best_val_loader, nn.BCELoss()
    )

    print(f"\n--- Best Fold (Fold {best_fold_index}) Final Performance (BCELoss) ---")
    print(f"Final Test Loss: {test_loss_bce:.4f}")

    # A. Plot History / 绘制历史
    plot_history(all_histories[best_fold_index], SAVE_DIR, best_fold_index)

    # B. Plot ROC / 绘制 ROC
    plot_roc_curves(Y_true_binary, Y_prob_all, SAVE_DIR, best_fold_index)

    # C. Plot Matrix / 绘制矩阵
    plot_component_detection_matrix(Y_true_binary, Y_prob_all, SAVE_DIR, best_fold_index)

    # D. Grad-CAM / 生成热力图
    plot_and_save_grad_cam_7_classes(best_model, best_val_dataset, W, Y_true_binary, SAVE_DIR, best_fold_index)

    print(f"\n--- Final Model Summary (Best Fold {best_fold_index}) ---")
    print(f"PS   Acc: {test_accs_bce[0] * 100:.2f}%, F1: {f1_list[0]:.4f}")
    print(f"PVC  Acc: {test_accs_bce[1] * 100:.2f}%, F1: {f1_list[1]:.4f}")
    print(f"PMMA Acc: {test_accs_bce[2] * 100:.2f}%, F1: {f1_list[2]:.4f}")
    print(f"==================================================")
    print(f"Mean F1: {f1_avg:.4f}")
    print(f"✅ Overall Accuracy: {overall_accuracy_bce * 100:.2f}%")

    print("\n--- All Tasks Completed / 所有任务已完成 ---")
    print(f"Results saved to directory: {SAVE_DIR}")