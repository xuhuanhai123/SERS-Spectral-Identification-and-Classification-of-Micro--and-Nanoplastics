import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import seaborn as sns
import os

# ==========================================
# 1. 数据预处理与“预清洗”逻辑 (Label 1-7)
# ==========================================
def load_and_clean_data(csv_path):
    print("正在加载并预清洗数据 (筛选 7 类混合物)...")
    # 读取原始 3 列二进制标签数据
    data = pd.read_csv(csv_path, header=None, skiprows=1)
    Y_bin = data.iloc[:, 0:3].values.astype(int)
    X_raw = data.iloc[:, 3:].values.astype(float)

    # 计算类别索引 (1-7): PS=4, PVC=2, PMMA=1
    # 组合举例: PS+PVC = 6, PS+PVC+PMMA = 7
    Y_idx = (Y_bin[:, 0] * 4 + Y_bin[:, 1] * 2 + Y_bin[:, 2] * 1)

    # 【预清洗】剔除全 0 样本 (无塑料背景)
    mask = Y_idx > 0
    X_clean = X_raw[mask]
    Y_clean = Y_idx[mask] - 1  # 转换为 0-6 索引以符合 PyTorch 要求
    Y_bin_clean = Y_bin[mask]

    # 执行 S-G 基线校正 (W=101, P=3)
    print(f"正在执行 S-G 校正与归一化 (剩余样本量: {len(X_clean)})...")
    for i in range(X_clean.shape[0]):
        baseline = savgol_filter(X_clean[i, :], 101, 3)
        X_clean[i, :] = X_clean[i, :] - baseline

    # Min-Max 归一化
    X_max = X_clean.max(axis=1, keepdims=True)
    X_min = X_clean.min(axis=1, keepdims=True)
    X_clean = (X_clean - X_min) / (X_max - X_min + 1e-8)

    return X_clean, Y_clean, Y_bin_clean



def run_comparison(csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, Y_bin = load_and_clean_data(csv_path)

    X_train, X_test, y_train, y_test, y_bin_tr, y_bin_te = train_test_split(
        X, Y, Y_bin, test_size=0.2, random_state=42
    )

    results = []

    # 1. PCA-LDA
    print("评估 PCA-LDA...")
    pca = PCA(n_components=10)
    lda = LDA().fit(pca.fit_transform(X_train), y_train)
    y_pred = lda.predict(pca.transform(X_test))
    results.append(['PCA-LDA', accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')])

    # 2. SVM
    print("评估 SVM...")
    svm = SVC(kernel='rbf', C=0.2).fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results.append(['SVM', accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')])


    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1-Score'])
    print("\n" + "=" * 45 + "\n7分类预清洗任务对比结果：\n" + "=" * 45)
    print(df)

if __name__ == "__main__":
    run_comparison('batch_spectra.csv')