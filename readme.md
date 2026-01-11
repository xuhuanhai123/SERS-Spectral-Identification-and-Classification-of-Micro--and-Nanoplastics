# MBARN: 基于深度学习的微纳米塑料 SERS 光谱智能识别系统

本项目基于 **MBARN (Multi-Branch Attention Residual Network)** 模型，实现了复杂环境水体中微纳米塑料（PS, PVC, PMMA）及其混合体系的自动化识别。项目涵盖了从原始数据增强、标签转换、模型训练到可解释性分析的完整全流程。



## 📁 文件夹结构说明

* **`Pure water/`**: 存放实验室超纯水环境下采集的原始 SERS 光谱数据（训练集来源）。
* **`Rain/`**: 存放模拟降雨等真实复杂环境下的测试光谱数据（用于验证模型泛化能力）。

## 📄 文件功能指南

### 1. 数据预处理与增强 (Data Preparation)
* **`batch processing-data augmentation.py`**: 核心增强脚本。通过模拟基线漂移、高斯噪声、波数偏移等策略扩充样本量，生成 `batch_spectra.csv`。
* **`batch processing - test data.py`**: 批量处理 `Rain` 文件夹下的原始测试数据，生成整合后的 `test_data_compiled.csv`。

### 2. 标签转换与格式化 (Label Engineering)
* **`batch processing - conversion of binary labels to one-hot labels.py`**: 将 [PS, PVC, PMMA] 多标签二元向量转换为 1-7 的单类别索引。
    * 生成训练集：`batch_spectra_with_single_label_index.csv`
    * 生成测试集：`test_data_compiled_with_single_label_index..csv`

### 3. 模型构建、训练和初步性能评估 (Model Implementation)
* **`mbarn.py`**: 阐明论文核心模型架构。集成 **通道注意力机制 (Channel Attention)**，实现对微弱指纹信号的自动强化。并生成一系列相关图表来评估 MBARN 模型的各类性能指标。
* **`CNN.py`**: 基准对照组（Baseline）的标准卷积神经网络实现。

### 4. 鲁棒性验证与对比实验 (Evaluation & Comparison)
* **`mbarn_evaluate.py` / `CNN_evaluate.py`**: 分别对模型在测试集（如 Rain 场景）上的准确率、F1 分数及混淆矩阵进行定量评估。
* **`model_compare.py`**: 综合对比经典模型（如 SVM, PCA-LDA）的预测性能。

### 5. 消融实验——注意力决策分析 (Interpretability)
* **`Interpretability in Mixed Environments.py`**: 利用 **Grad-CAM** 算法生成对于混合塑料识别时不同消融模型各自的预测分支在注意力分配上的 Grad-CAM 热力图。验证 MBARN 模型是否相对于其他消融模型更精准聚焦于塑料的化学键指纹峰（如 PMMA 的 812 $cm^{-1}$ 处）。



## 🛠️ 环境要求

请参考 `requirements.txt` 安装相关依赖。核心环境如下：
* **Python 3.9+**
* **PyTorch 2.3.0+cu118** (建议使用 GPU 加速)
* **Scipy**: 用于 Savitzky-Golay (S-G) 平滑等信号处理
* **Scikit-learn**: 用于性能评价指标计算
* **SHAP / Matplotlib**: 用于模型解释性可视化

## 🚀 核心工作流 (Pipeline)

1.  **数据生成**: 运行 `batch processing-data augmentation.py` 处理 `Pure water` 原始数据。
2.  **标签准备**: 运行标签转换脚本，将二元标签映射为单列类别索引。
3.  **模型训练和性能评估**: 运行 `mbarn.py` 进行网络参数训练与权重保存，导出训练收敛特性图、ROC曲线图、预测混淆矩阵图和 Grad-CAM 热力图，进行初步模型性能评测和物理意义分析。
4.  **鲁棒性验证**: 运行 `mbarn_evaluate.py` 评估模型在 `Rain` 复杂环境下的鲁棒性。
5.  **解释性实验**: 运行 `Interpretability in Mixed Environments.py` 进行注意力决策分析。

---

**备注**: 本项目代码实现严格遵循论文《基于超滑金膜平台的微纳米塑料 SERS 智能识别研究》中的实验设计。