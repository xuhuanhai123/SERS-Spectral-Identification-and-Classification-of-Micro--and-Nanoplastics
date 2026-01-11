import os
import numpy as np

# Import scipy.signal for Savitzky-Golay smoothing
# 导入 scipy.signal 用于 Savitzky-Golay 平滑
try:
    from scipy.signal import savgol_filter
except ImportError:
    print("Warning: SciPy library not installed. Smoothing will be skipped. Run 'pip install scipy'.")
    print("警告：未安装 SciPy 库。平滑功能将跳过。请运行 'pip install scipy' 安装。")
    savgol_filter = None

# -------------------------- Configuration / 核心配置 --------------------------
# Mapping of class names to multi-label vectors
# 类别名称到多标签向量的映射
CLASS_TO_LABEL = {
    "PS": [1, 0, 0],
    "PVC": [0, 1, 0],
    "PMMA": [0, 0, 1],
    "PS+PVC": [1, 1, 0],
    "PS+PMMA": [1, 0, 1],
    "PVC+PMMA": [0, 1, 1],
    "PS+PVC+PMMA": [1, 1, 1]
}

DATA_DIR = "Pure water"  # Directory containing spectra files / 光谱文件夹名称
OUTPUT_CSV = "batch_spectra.csv"  # Output CSV filename / 输出CSV文件名
EXPAND_NUM = 10  # Number of augmented samples per original / 每个原始光谱扩展的数量
MAX_SHIFT_PIXELS = 10  # Maximum pixels for wavenumber shifting / 波数平移点数


# -------------------------- Data Augmentation / 数据增强函数 --------------------------

def add_gaussian_noise(spectrum, noise_level=0.20):
    """
    Combines additive and multiplicative Gaussian noise to simulate instrument and photon noise.
    组合加性高斯噪声和乘性高斯噪声，模拟真实仪器和光子噪声。
    """
    # 1. Multiplicative noise (simulates signal-dependent noise/photon statistics)
    # 乘性噪声 (模拟信噪相关噪声)
    multiplicative_factor = np.random.uniform(0.01, 0.05)  # 1% to 5% intensity fluctuation
    multiplicative_noise = np.random.normal(loc=1.0, scale=multiplicative_factor, size=spectrum.shape)

    # 2. Additive noise (simulates instrument background noise)
    # 加性噪声 (模拟仪器本底噪声)
    additive_noise_std = noise_level * spectrum.std()
    additive_noise = np.random.normal(loc=0, scale=additive_noise_std, size=spectrum.shape)

    # Total Noise: (Spectrum * Multiplicative) + Additive
    return (spectrum * multiplicative_noise) + additive_noise


def add_baseline_drift(spectrum, drift_strength=0.30):
    """
    Simulates baseline drift using polynomials, S/U curves, and asymmetric edge lifts.
    模拟基线漂移：使用多项式、S/U 形曲线以及强制不对称抬升。
    """
    seq_len = len(spectrum)
    x = np.arange(seq_len) / (seq_len - 1)  # Normalized x-coordinates [0, 1]

    drift_range = spectrum.max() - spectrum.min()
    baseline = np.zeros_like(x)

    # --- 1. Random polynomial drift (Order 0-5) / 随机多项式漂移 ---
    poly_order = np.random.randint(0, 6)
    coeffs = np.random.uniform(-drift_strength, drift_strength, size=poly_order + 1)
    poly_baseline = np.polyval(coeffs, x) * drift_range * 0.7
    baseline += poly_baseline

    # --- 2. S-shape or U-shape curve (50% probability) / S形或U形曲线 ---
    if np.random.random() < 0.5:
        strength = np.random.uniform(0.1, 0.3) * drift_range
        if np.random.choice([True, False]):
            shape_drift = strength * (x - 0.5) ** 2  # U-shape
        else:
            shape_drift = strength * (x - 0.5) ** 3  # S-shape
        baseline += shape_drift

    # --- 3. Asymmetric edge lifting (40% probability) / 不对称边缘抬升 ---
    if np.random.random() < 0.4:
        edge_strength = np.random.uniform(0.15, 0.4) * drift_range
        if np.random.choice([True, False]):
            edge_drift = edge_strength * (1 - x ** 2)  # Lift low wavenumber side
        else:
            edge_drift = edge_strength * x ** 2  # Lift high wavenumber side

        if np.random.random() < 0.2:
            edge_drift *= -1  # Allow strong decline
        baseline += edge_drift

    return spectrum + baseline


def random_scaling(spectrum, scale_range=(0.4, 1.6)):
    """
    Randomly scales the intensity of the spectrum.
    随机缩放光谱强度。
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return spectrum * scale


def random_wavenumber_shift(spectrum, max_shift_pixels=MAX_SHIFT_PIXELS):
    """
    Randomly shifts the spectrum along the x-axis to simulate wavenumber drift.
    随机平移光谱强度，模拟波数漂移。
    """
    shift = np.random.randint(-max_shift_pixels, max_shift_pixels + 1)
    if shift == 0:
        return spectrum

    shifted = np.roll(spectrum, shift)
    fill_value = np.min(spectrum)

    if shift > 0:
        shifted[:shift] = fill_value  # Fill the beginning with minimum value
    elif shift < 0:
        shifted[shift:] = fill_value  # Fill the end with minimum value

    return shifted


def apply_savgol_filter(spectrum, window_length=11, polyorder=2):
    """
    Applies Savitzky-Golay smoothing to simulate post-processing.
    应用 Savitzky-Golay 平滑，模拟后处理过程。
    """
    if savgol_filter is None:
        return spectrum

    # Ensure window_length is odd and suitable for spectrum length
    if window_length % 2 == 0:
        window_length += 1
    if len(spectrum) < window_length:
        window_length = len(spectrum) - (len(spectrum) % 2 == 0)

    if window_length <= polyorder:
        return spectrum

    return savgol_filter(spectrum, window_length, polyorder)


# -------------------------- Spectral IO / 光谱读取与预处理 --------------------------

def read_single_spectrum(file_path):
    """
    Reads a spectrum file and returns (wavenumber, intensity).
    读取光谱文件，返回（波数, 强度）。
    """
    try:
        data = np.loadtxt(file_path, delimiter=None)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("File must contain two columns (Wavenumber + Intensity)")
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Read failed {file_path}: {e}")
        return None, None


def process_and_expand(raman_shifts, intensity, label):
    """
    Applies augmentation to a single spectrum and generates expanded samples.
    处理单条光谱并生成增强样本，返回数据列表。
    """
    if raman_shifts is None or intensity is None:
        return [], None

    csv_lines = []
    for i in range(EXPAND_NUM):
        enhanced = intensity.copy()

        if i == 0:
            pass  # Keep original spectrum for the first sample
        else:
            # 1. Random Scaling (70% probability)
            if np.random.random() < 0.7:
                enhanced = random_scaling(enhanced)

            # 2. Wavenumber shift (60% probability)
            if np.random.random() < 0.6:
                enhanced = random_wavenumber_shift(enhanced)

            # 3. Baseline drift (95% probability)
            if np.random.random() < 0.95:
                enhanced = add_baseline_drift(enhanced)

            # 4. Random noise (80% probability)
            if np.random.random() < 0.8:
                enhanced = add_gaussian_noise(enhanced)

            # 5. Mild smoothing (70% probability)
            if np.random.random() < 0.7:
                enhanced = apply_savgol_filter(enhanced)

        line = list(label) + enhanced.tolist()
        csv_lines.append(line)

    return csv_lines, raman_shifts


# -------------------------- Batch Processing / 批量处理与导出 --------------------------

def batch_to_csv():
    """
    Main function: Batch processes files in DATA_DIR and saves to OUTPUT_CSV.
    主函数：批量处理文件夹中的光谱并导出为 CSV 文件。
    """
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    all_lines = []
    header_written = False
    reference_wavenum_len = None

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        print(f"Processing: {filename}")

        # Label identification logic based on filename
        # 基于文件名的标签识别逻辑
        class_name = None
        if "PS+PVC+PMMA" in filename or "PMMA+PVC+PS" in filename:
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

        raman_shifts, intensity = read_single_spectrum(file_path)
        label = CLASS_TO_LABEL[class_name]
        csv_lines, wavenums = process_and_expand(raman_shifts, intensity, label)

        if not csv_lines or wavenums is None:
            continue

        # Check dimension consistency / 检查维度一致性
        current_len = len(wavenums)
        if reference_wavenum_len is None:
            reference_wavenum_len = current_len
        elif current_len != reference_wavenum_len:
            print(f"Warning: Dim mismatch in {filename}, skipping.")
            continue

        # Write header for the first valid file
        if not header_written:
            header = ["Label_PS", "Label_PVC", "Label_PMMA"] + wavenums.tolist()
            all_lines.append(header)
            header_written = True

        all_lines.extend(csv_lines)

    if len(all_lines) <= 1:
        print("No valid data generated.")
        return

    # Write data to CSV file / 写入 CSV 文件
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        for i, line in enumerate(all_lines):
            formatted = []
            for j, val in enumerate(line):
                if i == 0:  # Header
                    formatted.append(f"{val}")
                elif j < 3:  # Int Labels
                    formatted.append(f"{int(val)}")
                else:  # Float intensity
                    formatted.append(f"{val:.6f}")
            f.write(",".join(formatted) + "\n")

    print(f"Batch processing complete! Generated {len(all_lines) - 1} spectra saved to {OUTPUT_CSV}")


# -------------------------- Execution / 执行 --------------------------
if __name__ == "__main__":
    batch_to_csv()