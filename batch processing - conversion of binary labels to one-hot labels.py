import pandas as pd
import numpy as np
import os


# 1. Define transformation function / 定义转换函数
def binary_to_categorical_index(Y_binary):
    """
    Convert (N, 3) binary labels [PS, PVC, PMMA] to (N,) categorical indices (1-7).
    Mapping logic: PS*4 + PVC*2 + PMMA*1

    将 (N, 3) 二元标签 [PS, PVC, PMMA] 转换为 (N,) 类别索引 (1-7)。
    映射逻辑: PS*4 + PVC*2 + PMMA*1
    """
    # Ensure input is integer type / 确保输入是整数类型
    Y_binary = Y_binary.astype(int)

    # Perform weighted summation to get categorical indices (1-7)
    # 0: None, 1: PMMA, 2: PVC, 3: PVC/PMMA, 4: PS, 5: PS/PMMA, 6: PS/PVC, 7: Triple
    # 执行加权求和，得到类别索引 (1-7)
    categorical_labels = Y_binary[:, 0] * 4 + Y_binary[:, 1] * 2 + Y_binary[:, 2] * 1
    return categorical_labels.astype(int)


# 2. File Path Settings / 设置文件路径
FILE_PATH = 'test_data_compiled.csv'
OUTPUT_FILE_PATH = 'test_data_compiled_with_single_label_index.csv'
LABEL_COLUMNS = ['Label_PS', 'Label_PVC', 'Label_PMMA']  # Original label columns / 原始数据中的标签列名

try:
    # 3. Load data / 加载数据
    data_raw = pd.read_csv(FILE_PATH)

    # Extract binary labels (first 3 columns)
    # 提取二元标签 (假设前三列为 Label_PS, Label_PVC, Label_PMMA)
    Y_binary_raw = data_raw.iloc[:, 0:3].values

    # 4. Perform transformation / 执行转换
    Y_categorical_index = binary_to_categorical_index(Y_binary_raw)

    # 5. Create new DataFrame with single label and spectral data
    # 5. 创建新的 DataFrame，仅包含新标签和光谱数据

    # a. Remove old binary label columns (assume spectra start from 4th column)
    # 移除旧的二元标签列 (假设光谱数据从第4列开始)
    X_data = data_raw.iloc[:, 3:]

    # b. Create new label series / 创建新的标签 Series
    Y_new = pd.Series(Y_categorical_index, name='Label_Categorical_Index')

    # c. Concatenate categorical label with spectral data
    # 合并新的单一标签和光谱数据
    data_output = pd.concat([Y_new, X_data], axis=1)

    # 6. Save new CSV file / 保存新的 CSV 文件
    data_output.to_csv(OUTPUT_FILE_PATH, index=False)

    print(f"✅ Label transformation completed / 标签转换完成。")
    print(f"File saved to / 新文件已保存至: {OUTPUT_FILE_PATH}")
    print(f"Structure: Single categorical column followed by spectral data.")
    print(f"新文件的结构: 只有一列类别索引和光谱数据。")
    print("-" * 50)

    # Print header preview (first 5 and last column)
    # 打印前 5 列和最后一列数据，确认输出格式
    print("Preview of first row / 新文件的第一行内容:")
    print(data_output.iloc[:, [0, 1, 2, 3, 4, -1]].head(1).to_markdown(index=False))
    print("-" * 50)

except FileNotFoundError:
    print(f"❌ Error: File {FILE_PATH} not found / 错误：找不到文件。")
except Exception as e:
    print(f"An error occurred / 发生错误: {e}")