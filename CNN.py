import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import warnings
import sys

# --- Global Plotting Configuration / 全局绘图配置 ---
plt.rcParams.update({
    'font.size': 14,  # Default font size / 全局默认字体大小
    'axes.titlesize': 18,  # Title size / 图表标题大小
    'axes.labelsize': 16,  # Axis labels size / 坐标轴标签大小
    'xtick.labelsize': 12,  # X-axis tick size / X轴刻度文字大小
    'ytick.labelsize': 12,  # Y-axis tick size / Y轴刻度文字大小
    'legend.fontsize': 12,  # Legend font size / 图例字体大小
    'figure.dpi': 100  # Figure resolution / 分辨率
})

# Set computation device / 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device} (使用的设备)")

# Ignore specific CUDNN warnings / 忽略特定的 CUDNN 警告
warnings.filterwarnings("ignore", "Plan failed with a cudnnException", category=UserWarning)


# --- 1. Data Processing & Dataset Definition / 数据处理与数据集定义 ---

class MicroplasticDataset(Dataset):
    """
    Custom Dataset for Microplastic SERS data (Single-label 7 classes).
    自定义微塑料 SERS 数据集 (单标签 7 分类)。
    """

    def __init__(self, X, Y_cat):
        """
        Args:
            X: Input features. Shape (N, L, 1) -> (N, C_in=1, L) for Conv1d.
            Y_cat: Categorical labels (0-6).
        """
        # Convert dimensions for Conv1d: (N, L, 1) -> (N, 1, L)
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        # Labels must be Long type for CrossEntropyLoss
        self.Y = torch.from_numpy(Y_cat).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_preprocess_data(filepath):
    """
    Load and preprocess data from CSV.
    加载并预处理 CSV 数据。
    Format: [Label_Categorical_Index (1-7), Wavenumber_1, ...]
    """
    try:
        data_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File not found at: {filepath}")
        sys.exit(1)

    Y_categorical_1_7 = data_raw.iloc[:, 0].values
    X_raw = data_raw.iloc[:, 1:].values.astype(float)
    X = X_raw.copy()
    wavenumbers = data_raw.columns[1:].astype(float).values

    # Validate label range / 验证标签范围 (1-7)
    if np.min(Y_categorical_1_7) < 1 or np.max(Y_categorical_1_7) > 7:
        print("❌ Error: Label range in CSV is invalid!")
        print(f"Detected {np.min(Y_categorical_1_7)} to {np.max(Y_categorical_1_7)}. Must be 1-7.")
        sys.exit(1)

    # Convert 1-7 to 0-6 index for PyTorch / 将 1-7 转换为 PyTorch 所需的 0-6 索引
    Y_categorical_0_6 = Y_categorical_1_7 - 1

    # --- S-G Baseline Correction & Normalization / S-G 基线校正与归一化 ---
    WINDOW_LENGTH = 101
    POLY_ORDER = 3
    N_FEATURES = X.shape[1]

    # Adjust window size dynamically / 动态调整窗口大小
    if N_FEATURES < WINDOW_LENGTH:
        WINDOW_LENGTH = N_FEATURES - 1 if N_FEATURES % 2 == 0 else N_FEATURES - 2
    if WINDOW_LENGTH < 3: WINDOW_LENGTH = 3
    if WINDOW_LENGTH % 2 == 0: WINDOW_LENGTH += 1
    if WINDOW_LENGTH <= POLY_ORDER: POLY_ORDER = WINDOW_LENGTH - 1

    if WINDOW_LENGTH > POLY_ORDER:
        print(f"Applying S-G baseline correction (Window={WINDOW_LENGTH}, Poly={POLY_ORDER})...")
        for i in range(X.shape[0]):
            baseline = savgol_filter(X[i, :], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, axis=0)
            X[i, :] = X[i, :] - baseline

    # Min-Max Normalization (0-1) / 最小-最大归一化
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    # Add channel dimension / 增加通道维度: (N, L) -> (N, L, 1)
    X = np.expand_dims(X, axis=-1)

    print(f"Preprocessing complete. Samples: {X.shape[0]}, Features: {X.shape[1]}")
    return X, Y_categorical_0_6.astype(int), wavenumbers.astype(float)


# --- 2. PyTorch Model Definition: Simple CNN / PyTorch 模型定义：简单 CNN ---


[Image of 1D convolutional neural network architecture]


class SimpleCNNModel(nn.Module):
    """
    Simple CNN Architecture for 1D Spectral Data.
    适用于一维光谱数据的简单 CNN 架构。
    """

    def __init__(self, input_length, num_classes=7):
        super(SimpleCNNModel, self).__init__()
        # Conv layers / 卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Calculate flattened size / 计算展平后的特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._forward_features(dummy_input)
            flat_size = x.numel() // x.shape[0]

        if not hasattr(self, 'flat_size_printed'):
            print(f"DEBUG: Calculated flat_size: {flat_size}")
            self.flat_size_printed = True

        # Classifier head / 分类器
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        # Hooks for Grad-CAM / 用于 Grad-CAM 的变量
        self.activations = None
        self.gradients = None

    def save_gradient(self, grad):
        """Gradient hook / 梯度钩子"""
        self.gradients = grad

    def _forward_features(self, x):
        """Feature extraction / 特征提取阶段"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Layer for Grad-CAM / 在最后一层卷积层挂载钩子
        x = self.conv2(x)
        x = F.relu(x)

        # Capture high-res features before final pooling / 在最后池化前捕获高分辨率特征
        if x.requires_grad:
            self.activations = x.clone()
            x.register_hook(self.save_gradient)

        x = self.pool2(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        outputs = self.classifier(x)
        return outputs


# --- 3. Training & Evaluation Functions / 训练与评估函数 ---

def evaluate_model(model, data_loader, criterion):
    """Evaluate model on provided data loader / 在指定数据加载器上评估模型"""
    model.eval()
    Y_true_all, Y_pred_all = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in data_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            Y_true_all.append(Y_batch.cpu().numpy())
            Y_pred_all.append(predicted.cpu().numpy())

    Y_true_all = np.concatenate(Y_true_all, axis=0)
    Y_pred_all = np.concatenate(Y_pred_all, axis=0)
    overall_accuracy = np.sum(Y_pred_all == Y_true_all) / Y_true_all.shape[0]
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, overall_accuracy, Y_true_all, Y_pred_all, outputs


def train_model(model, train_loader, test_loader, epochs=100, save_dir='.'):
    """Training loop for 7-class classification / 7 分类训练循环"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {'train_loss': [], 'test_loss': [], 'overall_acc': []}
    best_overall_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')

    print("\n--- Starting Training (7-Class) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * X_batch.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)
        test_loss, overall_acc, _, _, _ = evaluate_model(model, test_loader, criterion)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['overall_acc'].append(overall_acc)

        # Save best model / 保存最佳模型
        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs}: Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
                  f"Overall Acc: {overall_acc * 100:.2f}%")
    return history, best_model_path


# --- 4. Grad-CAM Functions / Grad-CAM 函数 ---

def generate_grad_cam(model, input_tensor, target_class_index=0):
    """Generates Grad-CAM heatmap for 1D CNN / 生成 1D CNN 的 Grad-CAM 热力图"""
    model.eval()
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad_(True)
    outputs = model(input_tensor)

    # Target logit for the specific class / 针对特定类别的 Logit 输出
    target_output_logit = outputs[0, target_class_index]
    model.zero_grad()
    target_output_logit.backward(retain_graph=True)

    if model.gradients is None or model.activations is None:
        return np.zeros(input_tensor.shape[2]), target_output_logit.cpu().item()

    gradients = model.gradients.data
    activations = model.activations.data

    # Global average pooling of gradients / 梯度的全局平均池化
    pooled_gradients = torch.mean(gradients, dim=2)
    weights = pooled_gradients.squeeze(0)
    activations = activations.squeeze(0)

    # Weighted sum of activations / 激活图的加权和
    heatmap_channels = weights.unsqueeze(1) * activations
    heatmap = torch.sum(heatmap_channels, dim=0)
    heatmap = F.relu(heatmap)

    # Normalization / 归一化
    heatmap_np = heatmap.cpu().numpy()
    if np.max(heatmap_np) > 0:
        heatmap_np = heatmap_np / np.max(heatmap_np)
    return heatmap_np, target_output_logit.cpu().item()


# --- 5. Saving & Visualization / 结果保存与可视化 ---

CATEGORY_NAMES = {
    0: 'PMMA', 1: 'PVC', 2: 'PVC/PMMA',
    3: 'PS', 4: 'PS/PMMA', 5: 'PS/PVC', 6: 'Triple'
}
CATEGORY_LABELS = [CATEGORY_NAMES[i] for i in range(7)]


def plot_history(history, save_dir):
    """Plot Loss and Accuracy curves / 绘制 Loss 与 Accuracy 曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss Curve / 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['test_loss'], label='Testing Loss')
    plt.title('Training and Testing Loss (Cross Entropy)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. Accuracy Curve / 绘制 Accuracy 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [a * 100 for a in history['overall_acc']], label='Accuracy', color='red')
    plt.title('Testing Overall Accuracy (7 Classes)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()


def plot_confusion_matrix_7x7(Y_true, Y_pred, save_dir):
    """Generate normalized confusion matrix / 生成归一化混淆矩阵"""
    cm = confusion_matrix(Y_true, Y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CATEGORY_LABELS, yticklabels=CATEGORY_LABELS,
                cbar_kws={'label': 'Normalized Accuracy'})

    plt.title('Normalized Confusion Matrix (7 Classes)')
    plt.xlabel('Predicted Category')
    plt.ylabel('True Category')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_7x7.png'))
    plt.close()


def find_sample_indices_for_7_classes(Y_true):
    """Find first representative sample index for each class / 寻找各类别第一个代表性样本的索引"""
    sample_indices = {}
    for cat_idx in range(7):
        indices = np.where(Y_true == cat_idx)[0]
        if len(indices) > 0:
            sample_indices[cat_idx] = indices[0]
    return sample_indices


def plot_and_save_grad_cam_7_classes(model, test_dataset, wavenumbers, Y_true, save_dir):
    """Plot Grad-CAM for representative samples / 生成并保存代表性样本的 Grad-CAM 图"""
    sample_indices = find_sample_indices_for_7_classes(Y_true)
    print(f"\n--- Generating Grad-CAM for {len(sample_indices)} samples ---")

    for cat_idx, data_idx in sample_indices.items():
        class_name = CATEGORY_NAMES.get(cat_idx, "Unknown")
        X_test_sample, _ = test_dataset[data_idx]
        test_input_tensor = X_test_sample.unsqueeze(0).to(device)

        heatmap_np, _ = generate_grad_cam(model, test_input_tensor, target_class_index=cat_idx)

        plt.figure(figsize=(12, 6))
        plt.title(f'Grad-CAM: Class "{class_name}"')
        plt.xlabel('Raman Shift ($cm^{-1}$)')
        plt.ylabel('Normalized Intensity')

        # Original spectrum / 原始光谱
        spectrum_data = X_test_sample.squeeze().cpu().numpy()
        plt.plot(wavenumbers, spectrum_data, label='SERS Spectrum', color='black', alpha=0.8)

        # Resize heatmap to match spectrum / 调整热力图大小以匹配光谱
        heatmap_resized = cv2.resize(heatmap_np.reshape(-1, 1),
                                     (1, wavenumbers.shape[0]),
                                     interpolation=cv2.INTER_LINEAR).squeeze()

        plt.fill_between(wavenumbers, 0, heatmap_resized, alpha=0.4, color='orange', label='Grad-CAM Focus')

        save_path = os.path.join(save_dir, f'grad_cam_class_{class_name.replace("/", "_")}.png')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(0, 1.05)
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved Grad-CAM for {class_name} at: {save_path}")


# --- 6. Main Execution Block / 主执行区块 ---

if __name__ == '__main__':
    SAVE_DIR = 'model_output_simple_cnn'
    os.makedirs(SAVE_DIR, exist_ok=True)
    FILE_PATH = 'batch_spectra_with_single_label_index.csv'

    # A. Data Loading / 数据加载与准备
    X, Y_cat, W = load_and_preprocess_data(FILE_PATH)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42)

    # Dataloaders / 数据加载器
    train_loader = DataLoader(MicroplasticDataset(X_train, Y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(MicroplasticDataset(X_test, Y_test), batch_size=64, shuffle=False)

    # B. Model Initialization & Training / 模型初始化与训练
    INPUT_LENGTH = X_train.shape[1]
    model = SimpleCNNModel(INPUT_LENGTH, num_classes=7).to(device)
    training_history, best_model_path = train_model(model, train_loader, test_loader, epochs=50, save_dir=SAVE_DIR)

    # C. Visualizing History / 训练历史可视化
    plot_history(training_history, SAVE_DIR)

    # D. Evaluate Best Model / 加载最佳模型进行最终评估
    best_model = SimpleCNNModel(INPUT_LENGTH, num_classes=7).to(device)
    try:
        best_model.load_state_dict(torch.load(best_model_path))
    except FileNotFoundError:
        print(f"❌ Error: Could not find model file: {best_model_path}")
        sys.exit(1)

    # Final evaluation / 最终评估
    criterion_final = nn.CrossEntropyLoss()
    _, overall_acc, Y_true_final, Y_pred_final, _ = evaluate_model(best_model, test_loader, criterion_final)

    # E. Confusion Matrix / 混淆矩阵
    plot_confusion_matrix_7x7(Y_true_final, Y_pred_final, SAVE_DIR)

    # F. Grad-CAM Generation / Grad-CAM 图表生成
    plot_and_save_grad_cam_7_classes(best_model, MicroplasticDataset(X_test, Y_test), W, Y_true_final, SAVE_DIR)

    # G. Summary / 最终性能总结
    print(f"\n--- Final Model Performance (Simple CNN, 7 Classes) ---")
    print(f"==================================================")
    print(f"✅ Overall Accuracy: {overall_acc * 100:.2f}%")
    print(f"\n--- Task Complete. Results saved in: {SAVE_DIR} ---")