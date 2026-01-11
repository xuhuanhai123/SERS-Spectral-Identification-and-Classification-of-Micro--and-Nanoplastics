import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import warnings
import random


# ==========================================
# 0. 环境配置
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = 'Averaged_Interpretability_Final'
os.makedirs(RESULT_DIR, exist_ok=True)
DATA_PATH = 'batch_spectra.csv'


# ==========================================
# 1. 模型架构定义
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_c, max(in_c // 4, 1), 1), nn.ReLU(),
            nn.Conv1d(max(in_c // 4, 1), in_c, 1), nn.Sigmoid()
        )

    def forward(self, x): return x * self.fc(x)


class AblationBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_res=True, use_attn=True):
        super().__init__()
        self.use_res, self.use_attn = use_res, use_attn
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 5, stride, 2, bias=False),
            nn.GroupNorm(8, out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, 5, 1, 2, bias=False),
            nn.GroupNorm(8, out_c)
        )
        self.attn = ChannelAttention(out_c) if use_attn else nn.Identity()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, stride, bias=False),
            nn.GroupNorm(8, out_c)
        ) if use_res and (stride != 1 or in_c != out_c) else nn.Identity()

    def forward(self, x):
        out = self.attn(self.conv(x))
        if self.use_res:
            out += self.shortcut(x) if isinstance(self.shortcut, nn.Sequential) else x
        return F.relu(out)


class MBARN_Ablation(nn.Module):
    def __init__(self, use_res, use_attn):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 64, 7, 2, 3), nn.GroupNorm(8, 64), nn.ReLU())
        self.layer = AblationBlock(64, 128, 2, use_res, use_attn)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(3)])

    def forward(self, x):
        x = self.layer(self.stem(x))
        features = x
        x = self.pool(x).flatten(1)
        return [torch.sigmoid(h(x)) for h in self.heads], features


# ==========================================
# 2. Grad-CAM 解释器 (加入高斯平滑优化)
# ==========================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.features = None

    def save_gradient(self, grad): self.gradients = grad

    def __call__(self, x, label_idx):
        target_layer = self.model.layer.conv[3]
        handler = target_layer.register_backward_hook(lambda m, i, o: self.save_gradient(o[0]))
        out, features = self.model(x)
        self.features = features
        self.model.zero_grad()
        out[label_idx].backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        cam = torch.sum(weights * self.features, dim=1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)

        # 物理特性平滑：使用高斯滤波让曲线更平滑，符合拉曼光谱包络形态
        cam = gaussian_filter1d(cam, sigma=2.0)

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handler.remove()
        return np.interp(np.linspace(0, 1, x.shape[2]), np.linspace(0, 1, len(cam)), cam)


# ==========================================
# 3. 多样本平均实验逻辑
# ==========================================
def run_averaged_interpretability():
    # A. 数据加载
    data_raw = pd.read_csv(DATA_PATH, header=None, low_memory=False)
    W = data_raw.iloc[0, 3:].astype(float).values
    Y_all = data_raw.iloc[1:, 0:3].values.astype(float)
    X_all = data_raw.iloc[1:, 3:].values.astype(float)

    # B. 筛选 PVC+PMMA 混合样本集合
    target_indices = np.where((Y_all[:, 1] == 1) & (Y_all[:, 2] == 1) & (Y_all[:, 0] == 0))[0]
    if len(target_indices) < 3:  # 如果纯混合样太少，则放宽条件
        target_indices = np.where((Y_all[:, 1] == 1) & (Y_all[:, 2] == 1))[0]

    num_to_avg = min(20, len(target_indices))
    selected_indices = target_indices[:num_to_avg]
    print(f"🚀 正在对 {num_to_avg} 个混合样本进行平均解释性分析...")

    # C. 实验配置
    configs = [
        {"name": "Baseline (CNN)", "res": False, "attn": False},
        {"name": "CNN + Residual", "res": True, "attn": False},
        {"name": "MBARN (Full)", "res": True, "attn": True}
    ]
    comp_names = ['PS', 'PVC', 'PMMA']
    physical_peaks = {
        'PVC': [637, 695],  # C-Cl stretching
        'PMMA': [812]  # C-H bending, C=O stretching
    }

    # 初始化存储
    accumulated_results = {cfg['name']: {1: [], 2: []} for cfg in configs}
    accumulated_spectra = []

    # D. 循环计算每个样本的解释热力图
    for idx in selected_indices:
        # 单样本预处理
        x_raw = X_all[idx]
        x_proc = x_raw - savgol_filter(x_raw, 51, 3)
        x_proc = (x_proc - x_proc.min()) / (x_proc.max() - x_proc.min() + 1e-8)
        accumulated_spectra.append(x_proc)

        input_tensor = torch.from_numpy(x_proc).float().unsqueeze(0).unsqueeze(0).to(device)

        for cfg in configs:
            model = MBARN_Ablation(cfg['res'], cfg['attn']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # 针对性拟合该样本
            for _ in range(50):
                model.train()
                out, _ = model(input_tensor)
                loss = sum(F.binary_cross_entropy(out[i].view(-1),
                                                  torch.tensor([Y_all[idx, i]], device=device).float()) for i in
                           range(3))
                loss.backward();
                optimizer.step();
                optimizer.zero_grad()

            model.eval()
            gcam = GradCAM(model)
            accumulated_results[cfg['name']][1].append(gcam(input_tensor, label_idx=1))  # PVC
            accumulated_results[cfg['name']][2].append(gcam(input_tensor, label_idx=2))  # PMMA

    # E. 结果绘图
    avg_spectrum = np.mean(accumulated_spectra, axis=0)

    # 增加 figsize 确保大字号下布局不拥挤
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    colors = {'Baseline (CNN)': '#4C72B0', 'CNN + Residual': '#E1812C', 'MBARN (Full)': '#55A868'}

    for b_idx, ax in enumerate(axes):
        t_idx = b_idx + 1  # 对应 PVC=1, PMMA=2
        t_name = comp_names[t_idx]

        # 1. 绘制平均光谱背景 (稍微加粗)
        ax.plot(W, avg_spectrum, color='black', alpha=0.15, label='Mean Mixed Spectrum', linewidth=1.5)

        # 2. 标注物理参考线
        for peak in physical_peaks[t_name]:
            ax.axvline(x=peak, color='red', linestyle='--', alpha=0.5, linewidth=2,
                       label='Characteristic Peak' if peak == physical_peaks[t_name][0] else "")

        # 3. 绘制各模型平均 Attention
        for name in accumulated_results:
            mean_cam = np.mean(accumulated_results[name][t_idx], axis=0)
            # 加粗曲线以配合大字号
            ax.plot(W, mean_cam, label=f'{name} Focus', color=colors[name], linewidth=3.0)
            if "Full" in name:
                ax.fill_between(W, 0, mean_cam, color=colors[name], alpha=0.15)

        # --- 字体与刻度调优 ---
        # 设置子图标题 (加粗, 字号18)
        ax.set_title(f"Targeting {t_name} Fingerprints (Averaged over {num_to_avg} samples)",
                     fontsize=20, fontweight='bold', pad=15)

        # 设置纵轴标签 (字号16)
        ax.set_ylabel("Attention Score", fontsize=18, labelpad=10)

        # 设置刻度数字大小 (字号15)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # 设置图例 (字号14, 设置背景框提高可读性)
        ax.legend(loc='upper right', fontsize=13, frameon=True, shadow=True, facecolor='white')

        # 网格线稍微明显一点
        ax.grid(True, alpha=0.4, linestyle=':')

    # 设置横轴标签 (字号18)
    plt.xlabel("Raman Shift ($cm^{-1}$)", fontsize=18, labelpad=10)

    # 调整整体布局，防止标签裁剪
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存高分辨率图片
    save_path = os.path.join(RESULT_DIR, 'PVC_PMMA_Averaged_Interpretation_LargeFont.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图像已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_averaged_interpretability()