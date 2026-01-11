import os
import numpy as np

# -------------------------- Configuration / 核心配置 --------------------------
# Mapping of filenames to multi-label binary vectors
# 类别名称到多标签二元向量的映射
CLASS_TO_LABEL = {
    "PS": [1, 0, 0],
    "PVC": [0, 1, 0],
    "PMMA": [0, 0, 1],
    "PS+PVC": [1, 1, 0],
    "PS+PMMA": [1, 0, 1],
    "PVC+PMMA": [0, 1, 1],
    "PS+PVC+PMMA": [1, 1, 1]
}

# Please modify the test data folder name here
# 请在此处修改您的测试数据文件夹名称
TEST_DATA_DIR = "Rain"

# Filename for the output compiled CSV
# 输出 CSV 文件名
OUTPUT_CSV = "test_data_compiled.csv"


# -------------------------- Data Loading Functions / 光谱读取函数 --------------------------
def read_single_spectrum(file_path):
    """
    Reads a single spectrum file and returns (wavenumber, intensity).
    读取单条光谱文件，返回(波数, 强度)。

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        tuple: (raman_shifts, intensity) if successful, (None, None) otherwise.
    """
    try:
        # Load data using numpy, assuming space or tab delimiter
        # 使用 numpy 加载数据，默认识别空格或制表符分隔
        data = np.loadtxt(file_path, delimiter=None)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("File must contain exactly two columns (Wavenumber + Intensity)")
        return data[:, 0], data[:, 1]  # Return wavenumber and intensity / 返回波数与强度
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return None, None


# -------------------------- Batch Processing / 批量处理 --------------------------
def compile_test_data():
    """
    Compiles all spectral files in the target directory into a single CSV file.
    将目标文件夹中的所有光谱文件整合并导出为单个 CSV 文件。
    """
    # Check if the directory exists
    # 检查光谱文件夹是否存在
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Error: Directory '{TEST_DATA_DIR}' not found. Please check 'TEST_DATA_DIR'.")
        return

    all_lines = []
    header_written = False
    reference_wavenum_len = None

    # Iterate through all spectral files
    # 遍历所有光谱文件
    for filename in os.listdir(TEST_DATA_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(TEST_DATA_DIR, filename)
        print(f"Processing: {filename}")

        # 1. Identify classification category from filename
        # 根据文件名识别类别
        class_name = None
        if "PS+PVC+PMMA" in filename or "PS+PMMA+PVC" in filename:
            class_name = "PS+PVC+PMMA"
        elif "PS+PVC" in filename:
            class_name = "PS+PVC"
        elif "PS+PMMA" in filename:
            class_name = "PS+PMMA"
        elif "PVC+PMMA" in filename or "PMMA+PVC" in filename:
            class_name = "PVC+PMMA"
        elif "PS" in filename:
            class_name = "PS"
        elif "PVC" in filename:
            class_name = "PVC"
        elif "PMMA" in filename:
            class_name = "PMMA"
        else:
            print(f"Skipping unidentified file: {filename}")
            continue

        # 2. Read spectral data
        # 读取光谱数据
        raman_shifts, intensity = read_single_spectrum(file_path)

        if intensity is None:
            continue

        label = CLASS_TO_LABEL[class_name]

        # 3. Check for dimensional consistency
        # 检查光谱维度（长度）一致性
        current_len = len(intensity)
        if reference_wavenum_len is None:
            reference_wavenum_len = current_len
        elif current_len != reference_wavenum_len:
            print(f"Warning: {filename} dimensions mismatch ({current_len} vs {reference_wavenum_len}). Skipping.")
            continue

        # 4. Generate header (only once)
        # 生成表头（仅限首次执行）
        if not header_written:
            # Use wavenumbers as header column names
            # 使用读取到的波数作为表头列名
            header = ["Label_PS", "Label_PVC", "Label_PMMA"] + raman_shifts.tolist()
            all_lines.append(header)
            header_written = True

        # 5. Append data row (labels + intensity)
        # 添加数据行（标签 + 强度）
        data_line = list(label) + intensity.tolist()
        all_lines.append(data_line)

    # 6. Save data to CSV file
    # 保存整合后的 CSV 文件
    if len(all_lines) <= 1:
        print("No valid spectral data found. Save cancelled.")
        return

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        for i, line in enumerate(all_lines):
            formatted = []
            for j, val in enumerate(line):
                if i == 0:
                    formatted.append(f"{val}")  # Header / 表头
                elif j < 3:
                    formatted.append(f"{int(val)}")  # Integer Labels / 标签列（整数）
                else:
                    formatted.append(f"{val:.6f}")  # Float Intensity / 强度数据（保留6位小数）
            f.write(",".join(formatted) + "\n")

    print(f"Success! {len(all_lines) - 1} spectra compiled and saved to {OUTPUT_CSV}")


# -------------------------- Main Execution / 程序入口 --------------------------
if __name__ == "__main__":
    compile_test_data()