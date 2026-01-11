import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Global Plotting Configuration / 全局绘图配置 ---
plt.rcParams.update({
    'font.size': 14,           # Default font size / 全局默认字体大小
    'axes.titlesize': 18,      # Chart title size / 图表标题大小
    'axes.labelsize': 16,      # Axis label size (X, Y) / 坐标轴标签 (X, Y轴) 大小
    'xtick.labelsize': 12,     # X-axis tick label size / X轴刻度文字大小
    'ytick.labelsize': 12,     # Y-axis tick label size / Y轴刻度文字大小
    'legend.fontsize': 12,     # Legend font size / 图例字体大小
    'figure.dpi': 100          # Resolution for clarity / 提高分辨率，让图片更清晰
})

# Device configuration / 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use / 使用的设备: {device}")

# Category names (Index 0-6 corresponds to label 1-7) 
# 7 类别名称 (索引 0-6 对应标签 1-7 的顺序)
CATEGORY_NAMES = {
    0: 'PMMA', 1: 'PVC', 2: 'PVC/PMMA',
    3: 'PS', 4: 'PS/PMMA', 5: 'PS/PVC', 6: 'Triple'
}
CATEGORY_LABELS = [CATEGORY_NAMES[i] for i in range(7)]


# --- 1. Data Processing & Dataset Definition / 数据处理与 Dataset 定义 ---

class MicroplasticDataset(Dataset):
    """Custom SERS dataset for microplastics (Single-label 7-class classification)"""
    """自定义微塑料 SERS 数据集 (单标签 7 分类)"""

    def __init__(self, X, Y_cat):
        # Convert X from (N, L, 1) to (N, 1, L) to match Conv1d input format
        # 转换 X: (N, L, 1) -> (N, 1, L) 以符合 CNN 1D 的输入格式
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)
        # Y_cat must be Long type (0-6)
        # Y_cat 必须是 0-6 的 Long 类型
        self.Y = torch.from_numpy(Y_cat).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_and_preprocess_data(filepath):
    """Load and preprocess spectral data from CSV with single-label indices."""
    """加载、预处理数据，从包含单一类别索引的 CSV 读取。"""
    try:
        # Expecting CSV with header; first column as label (1-7)
        # 假设 CSV 包含标头，且第一列为类别标签 (1-7)
        data_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File path not found / 错误：找不到文件路径: {filepath}")
        sys.exit(1)

    # Data extraction / 数据提取
    # Extract wavenumbers from header starting from column 2
    # Wavenumbers (从标头提取，从第二列开始)
    wavenumbers = data_raw.columns[1:].astype(float).values

    # Extract Y labels (1-7) from the first column
    # Y_raw: 第一列 (类别标签 1-7)
    Y_categorical_1_7 = data_raw.iloc[:, 0].values.astype(int)

    # Extract X spectral data from remaining columns
    # X_raw: 剩余所有列 (光谱数据)
    X_raw = data_raw.iloc[:, 1:].values.astype(float)
    X = X_raw.copy()

    # Convert labels 1-7 to 0-6 for PyTorch requirements
    # 关键步骤：标签转换 (1-7 到 PyTorch 要求的 0-6)
    Y_categorical_0_6 = Y_categorical_1_7 - 1

    # Range check for labels (0-6)
    # 检查转换后的范围 (确保在 0-6 内)
    if np.max(Y_categorical_0_6) > 6 or np.min(Y_categorical_0_6) < 0:
        print(
            f"❌ Critical Error: Label range ({np.min(Y_categorical_0_6)} to {np.max(Y_categorical_0_6)}) is invalid."
            f"Check if '{filepath}' contains Label 0."
            f"❌ 严重错误：转换后的标签范围超出 0-6 范围。检查文件是否包含标签 0。"
        )
        sys.exit(1)

    # Savitzky-Golay (S-G) Baseline Correction / S-G 基线校正
    WINDOW_LENGTH = 101
    POLY_ORDER = 3
    N_FEATURES = X.shape[1]

    # Validate S-G parameters / 确保 S-G 参数有效
    if N_FEATURES < WINDOW_LENGTH:
        WINDOW_LENGTH = N_FEATURES - 1 if N_FEATURES % 2 == 0 else N_FEATURES - 2
    if WINDOW_LENGTH < 3: WINDOW_LENGTH = 3
    if WINDOW_LENGTH % 2 == 0: WINDOW_LENGTH += 1
    if WINDOW_LENGTH <= POLY_ORDER: POLY_ORDER = WINDOW_LENGTH - 1

    if WINDOW_LENGTH > POLY_ORDER:
        print(f"Starting S-G baseline correction (Window={WINDOW_LENGTH}, Poly={POLY_ORDER})...")
        for i in range(X.shape[0]):
            baseline = savgol_filter(X[i, :], window_length=WINDOW_LENGTH, polyorder=POLY_ORDER, axis=0)
            X[i, :] = X[i, :] - baseline

    # Normalization / 归一化
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension / 添加 channel 维度 (N, L, 1)

    print(f"Data loading complete. Samples: {X.shape[0]}, Features: {X.shape[1]}")
    return X, Y_categorical_0_6, wavenumbers


# --- 2. Model Definition: Simple CNN / PyTorch 模型定义：Simple CNN 模型 ---



class SimpleCNNModel(nn.Module):
    """Basic 1D CNN architecture for spectral classification"""
    """基础 1D CNN 架构，用于光谱分类"""
    def __init__(self, input_length, num_classes=7):
        super(SimpleCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Dynamic calculation of flattened features / 动态计算展平后的特征数
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_length)
            x = self._forward_features(dummy_input)
            flat_size = x.numel() // x.shape[0]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.activations = None
        self.gradients = None

    def save_gradient(self, grad):
        """Gradient hook (Optional) / 梯度钩子"""
        pass

    def _forward_features(self, x):
        """Feature extraction block / 特征提取模块"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

    def forward(self, x):
        """Forward pass / 前向传播"""
        x = self._forward_features(x)
        x = self.flatten(x)
        outputs = self.classifier(x)
        return outputs


# --- 3. Evaluation & Plotting / 评估和绘图函数 ---

def evaluate_model_on_data(model, data_loader):
    """Evaluates the model, returns loss, accuracy, true labels, and predictions."""
    """评估模型，返回损失、准确率、真实标签和预测标签"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    Y_true_all = []
    Y_pred_all = []
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

    return avg_loss, overall_accuracy, Y_true_all, Y_pred_all


def plot_confusion_matrix_7x7(Y_true, Y_pred, save_dir):
    """Plot and save a 7x7 normalized confusion matrix."""
    """绘制 7x7 混淆矩阵并保存"""
    cm = confusion_matrix(Y_true, Y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized[np.isnan(cm_normalized)] = 0

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

    cm_path = os.path.join(save_dir, 'evaluation_confusion_matrix_7x7_simple_cnn.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Confusion matrix saved to / 混淆矩阵图已保存至: {cm_path}")


def run_evaluation_pipeline(model_path, data_filepath, input_length, output_save_dir):
    """Full evaluation pipeline / 执行完整的评估流程"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found / 错误：找不到模型权重文件: {model_path}")
        sys.exit(1)

    print(f"\n--- Starting Evaluation / 开始评估: {data_filepath} ---")

    # 1. Load and preprocess data / 加载和预处理数据
    X_real_np, Y_real_np, _ = load_and_preprocess_data(data_filepath)

    # 2. Initialize model and load weights / 初始化模型并加载权重
    model = SimpleCNNModel(input_length, num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Weights loaded from / 模型权重已加载: {model_path}")

    # 3. Create DataLoader / 创建 DataLoader
    real_dataset = MicroplasticDataset(X_real_np, Y_real_np)
    real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)

    # 4. Run evaluation / 运行评估
    avg_loss, overall_acc, Y_true_final, Y_pred_final = evaluate_model_on_data(
        model, real_loader
    )

    # 5. Output results / 结果输出
    print(f"\n--- Evaluation Summary / 模型评估总结 ---")
    print(f"Total samples / 样本总数: {Y_true_final.shape[0]}")
    print(f"Avg Loss / 平均损失: {avg_loss:.4f}")
    print(f"==================================================")
    print(f"✅ Overall Accuracy (7 Classes) / 总体准确率: {overall_acc * 100:.2f}%")

    # 6. Plot results / 绘制结果图
    plot_confusion_matrix_7x7(Y_true_final, Y_pred_final, output_save_dir)


# --- 4. Main Execution Block / 主执行区块 ---

if __name__ == '__main__':
    # 1. Path to trained Simple CNN model weights / 训练好的模型权重路径
    MODEL_WEIGHTS_PATH = 'model_output_simple_cnn/best_model.pth'

    # 2. Path to evaluation dataset / 评估数据集路径
    EVAL_DATA_FILEPATH = 'test_data_compiled_with_single_label_index.csv'

    # 3. Model input length / 模型输入长度
    MODEL_INPUT_LENGTH = 1015

    # 4. Result save directory / 结果保存目录
    SAVE_DIR = 'evaluation_results_simple_cnn_final'
    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        run_evaluation_pipeline(
            MODEL_WEIGHTS_PATH,
            EVAL_DATA_FILEPATH,
            MODEL_INPUT_LENGTH,
            SAVE_DIR
        )
    except Exception as e:
        print(f"\nError occurred / 发生错误: {e}")
        print("Please check file paths and CSV structure [Label (1-7), Spectral Data...].")
        print("请检查文件路径、模型输入长度及 CSV 结构。")