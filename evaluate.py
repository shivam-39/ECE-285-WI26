# evaluate.py — Evaluation Metrics & Iterative Stability Analysis

import os
import math
import lpips
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import config as cfg
from corruption import corrupt_batch, corrupt_single


# Optional LPIPS
# ---------------------------------------------------------------------------
# try:
#     import lpips
#     _LPIPS_NET = None # lazy init
#     LPIPS_AVAILABLE = True
# except ImportError:
#     LPIPS_AVAILABLE = False
#     print("[evaluate] Warning: 'lpips' package not found. LPIPS metric disabled.")
#     print("Install with: pip install lpips")
#
#
# def _get_lpips_net(device):
#     global _LPIPS_NET
#     if _LPIPS_NET is None:
#         _LPIPS_NET = lpips.LPIPS(net="alex").to(device)
#     return _LPIPS_NET



# Metric functions (inputs: tensors in [-1, 1], shape [B, C, H, W])
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    # Move to [0, 1]
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """Structural Similarity Index (higher is better). Mean over batch & channels."""
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target.clamp(-1, 1) + 1) / 2

    B, C, H, W = pred.shape
    # Using simple windowed average approximation
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x = F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_x2
    sig_y = F.avg_pool2d(target * target, window_size, 1, pad) - mu_y2
    sig_xy = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    ssim_map = numerator / denominator.clamp(min=1e-8)
    return ssim_map.mean().item()


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: str) -> float:
    """LPIPS perceptual distance (lower is better)."""
    # if not LPIPS_AVAILABLE:
    #     return float("nan")
    # net = _get_lpips_net(device)
    net = lpips.LPIPS(net="alex").to(device)
    with torch.no_grad():
        dist = net(pred.clamp(-1, 1), target.clamp(-1, 1))
    return dist.mean().item()


# Full evaluation on a DataLoader
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(G, val_loader, device: str) -> Dict[str, float]:
    """Evaluate generator on the validation set. Returns mean PSNR, SSIM, LPIPS over all batches."""
    G.eval()
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0

    for x_real in tqdm(val_loader, desc=" <=> Evaluating <=> ", leave=False):
        x_real = x_real.to(device, non_blocking=True)
        y, c, _ = corrupt_batch(x_real)
        y = y.to(device); c = c.to(device)

        x_hat = G(y, c)

        psnr_sum += psnr(x_hat, x_real)
        ssim_sum += ssim(x_hat, x_real)
        lpips_sum += compute_lpips(x_hat, x_real, device)
        n += 1

    return {
        "psnr":  psnr_sum / max(n, 1),
        "ssim":  ssim_sum / max(n, 1),
        "lpips": lpips_sum / max(n, 1),
    }



# Iterative Reconstruction Stability Analysis
# ---------------------------------------------------------------------------
@torch.no_grad()
def stability_analysis(
    G,
    x_clean: torch.Tensor, # [B, C, H, W] clean images
    num_iterations: int = cfg.STABILITY_ITERATIONS,
    corruption_idx: int = 0, # which corruption to use
    device: str = cfg.DEVICE,
    save_dir: str = cfg.RESULTS_DIR,
) -> Dict[str, List[float]]:
    """
    Iterative Reconstruction Stability Analysis. 
    Records PSNR, SSIM, LPIPS vs the original x_clean at each iteration. 
    Also saves a visualisation of the error trajectory.
    """
    G.eval()
    x = x_clean.to(device)

    history_psnr = []
    history_ssim = []
    history_lpips = []

    # k=0 baseline: clean vs clean
    history_psnr.append(psnr(x, x))
    history_ssim.append(ssim(x, x))
    history_lpips.append(0.0)

    x_hat = x.clone()
    from corruption import _CORRUPTION_FNS, one_hot_vector
    corrupt_fn = _CORRUPTION_FNS[corruption_idx]
    c = torch.stack([one_hot_vector(corruption_idx)] * x.shape[0]).to(device)

    for k in range(1, num_iterations + 1):
        # Corrupt the current reconstruction
        y_list = [corrupt_fn(x_hat[i].to(device)) for i in range(x_hat.shape[0])]
        y = torch.stack(y_list).to(device)

        # Reconstruct
        x_hat = G(y, c)

        history_psnr.append(psnr(x_hat, x))
        history_ssim.append(ssim(x_hat, x))
        history_lpips.append(compute_lpips(x_hat, x, device))

        print(f"[Stability] Iter {k:2d}/{num_iterations} | "
              f"PSNR={history_psnr[-1]:.2f}  "
              f"SSIM={history_ssim[-1]:.4f}  "
              f"LPIPS={history_lpips[-1]:.4f}")

    # Plot error trajectories
    _plot_stability(history_psnr, history_ssim, history_lpips, save_dir=save_dir)

    return {
        "psnr": history_psnr,
        "ssim": history_ssim,
        "lpips": history_lpips,
    }


def _plot_stability(psnr_hist, ssim_hist, lpips_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    iters = list(range(len(psnr_hist)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, psnr_hist, marker="o", color="steelblue")
    axes[0].set_title("PSNR vs Iteration"); axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("PSNR (dB)")

    axes[1].plot(iters, ssim_hist, marker="o", color="darkorange")
    axes[1].set_title("SSIM vs Iteration"); axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("SSIM")

    axes[2].plot(iters, lpips_hist, marker="o", color="seagreen")
    axes[2].set_title("LPIPS vs Iteration"); axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("LPIPS (↓)")

    plt.suptitle("Iterative Reconstruction Stability Analysis", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "stability_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" Done: Stability plot saved -> {path}")
