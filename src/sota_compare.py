# sota_compare.py — SOTA Baseline Comparison Using LaMa
#
# LaMa: "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
# Suvorov et al., WACV 2022  |  https://github.com/advimman/lama
#
# Installation (run once):
#   pip install simple-lama-inpainting
#
# This script:
#   1. Loads your existing val_loader (same split, same images as your model)
#   2. Applies only the MASKING corruption (LaMa's native task)
#   3. Runs LaMa inference image-by-image
#   4. Computes PSNR / SSIM / LPIPS — identical metric functions to evaluate.py
#   5. Runs Iterative Reconstruction Stability Analysis for LaMa
#   6. Saves a side-by-side visual grid: [corrupted | LaMa | ground truth]
#   7. Prints a final comparison table: Your Model vs LaMa
#
# Usage:
#   # Evaluate LaMa only:
#   python sota_compare.py
#
#   # Also print comparison against your model's saved metrics:
#   python sota_compare.py --your-model-psnr 28.5 --your-model-ssim 0.88 --your-model-lpips 0.12
#
#   # Limit to N batches (fast smoke-test):
#   python sota_compare.py --max-batches 5

import argparse
import os
import math
import numpy as np
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import src.config as cfg
from dataset import build_dataloaders
from evaluate import psnr, ssim, compute_lpips   # reuse your metric functions
from src.corruption import corrupt_batch              # to extract masks for LaMa


# ---------------------------------------------------------------------------
# LaMa wrapper
# ---------------------------------------------------------------------------
class LamaInpainter:
    """
    Thin wrapper around the `simple-lama-inpainting` pip package.
    Handles tensor <-> PIL conversion and device placement.

    Install:  pip install simple-lama-inpainting
    The package auto-downloads the pretrained weights on first use (~200 MB).
    """

    def __init__(self, device: str = "cpu"):
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError:
            raise ImportError(
                "\n[Error] LaMa package not found.\n"
                "Install it with:  pip install simple-lama-inpainting\n"
                "Then re-run this script."
            )
        self.model  = SimpleLama()          # loads weights automatically
        self.device = device
        print("[LaMa] Model loaded successfully.")

    @torch.no_grad()
    def inpaint_batch(
        self,
        x_real: torch.Tensor,   # [B, 3, H, W]  clean images in [-1, 1]
        masks: torch.Tensor,    # [B, 1, H, W]  binary masks  (1 = missing region)
    ) -> torch.Tensor:
        """
        Run LaMa on a batch.  Returns reconstructed tensor in [-1, 1].
        LaMa operates on PIL images so we loop per-image — standard practice.
        """
        B = x_real.shape[0]
        results = []

        for i in range(B):
            # Convert clean image tensor [-1,1] → PIL RGB
            img_np = _tensor_to_uint8(x_real[i])          # (H, W, 3) uint8
            pil_img = Image.fromarray(img_np)

            # Convert mask tensor [0,1] → PIL L  (255 = hole)
            mask_np = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
            pil_mask = Image.fromarray(mask_np, mode="L")

            # LaMa inference
            pil_out = self.model(pil_img, pil_mask)        # returns PIL image

            # Back to tensor in [-1, 1]
            out_np = np.array(pil_out).astype(np.float32) / 127.5 - 1.0
            out_t  = torch.from_numpy(out_np).permute(2, 0, 1)  # (3, H, W)
            results.append(out_t)

        return torch.stack(results).to(x_real.device)


# ---------------------------------------------------------------------------
# Mask extraction helper
# ---------------------------------------------------------------------------
def extract_mask_from_corruption(
    x_real: torch.Tensor,
    y_corrupted: torch.Tensor,
) -> torch.Tensor:
    """
    Derive a binary mask from the difference between a clean image and its
    masked/corrupted version.  Pixels that differ significantly are treated
    as the 'missing' region (mask = 1).

    Works well for masking corruption. For blur/noise/LR the mask will
    cover the full image — LaMa will still run but it is not its native task.
    Threshold is tuned for the [-1,1] range.
    """
    diff  = (x_real - y_corrupted).abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
    mask  = (diff > 0.05).float()
    return mask


# ---------------------------------------------------------------------------
# Iterative Stability Analysis for LaMa
# ---------------------------------------------------------------------------
@torch.no_grad()
def lama_stability_analysis(
    lama: LamaInpainter,
    x_clean: torch.Tensor,              # [B, 3, H, W]
    num_iterations: int = cfg.STABILITY_ITERATIONS,
    save_dir: str = cfg.RESULTS_DIR,
    device: str = cfg.DEVICE,
) -> Dict[str, List[float]]:
    """
    Mirrors evaluate.stability_analysis() but for LaMa.
    Uses MASKING corruption (LaMa's native task) for the iterative loop.
    """
    from src.corruption import _CORRUPTION_FNS, one_hot_vector
    # Corruption index 0 is assumed to be masking — adjust if your ordering differs
    MASK_CORRUPTION_IDX = 0
    corrupt_fn = _CORRUPTION_FNS[MASK_CORRUPTION_IDX]

    x = x_clean.to(device)
    history_psnr  = [psnr(x, x)]           # k=0: perfect
    history_ssim  = [ssim(x, x)]
    history_lpips = [0.0]

    x_hat = x.clone()

    for k in range(1, num_iterations + 1):
        # Corrupt current reconstruction
        y_list = [corrupt_fn(x_hat[i]) for i in range(x_hat.shape[0])]
        y = torch.stack(y_list).to(device)

        # Extract binary mask and run LaMa
        masks = extract_mask_from_corruption(x_hat, y)
        x_hat = lama.inpaint_batch(x_hat, masks)

        p = psnr(x_hat, x)
        s = ssim(x_hat, x)
        l = compute_lpips(x_hat, x, device)

        history_psnr.append(p)
        history_ssim.append(s)
        history_lpips.append(l)

        print(f"[LaMa Stability] Iter {k:2d}/{num_iterations} | "
              f"PSNR={p:.2f}  SSIM={s:.4f}  LPIPS={l:.4f}")

    _plot_lama_stability(history_psnr, history_ssim, history_lpips, save_dir)

    return {"psnr": history_psnr, "ssim": history_ssim, "lpips": history_lpips}


def _plot_lama_stability(psnr_hist, ssim_hist, lpips_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    iters = list(range(len(psnr_hist)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(iters, psnr_hist,  marker="o", color="steelblue")
    axes[0].set_title("LaMa — PSNR vs Iteration")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("PSNR (dB)")

    axes[1].plot(iters, ssim_hist,  marker="o", color="darkorange")
    axes[1].set_title("LaMa — SSIM vs Iteration")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("SSIM")

    axes[2].plot(iters, lpips_hist, marker="o", color="seagreen")
    axes[2].set_title("LaMa — LPIPS vs Iteration")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("LPIPS (↓)")

    plt.suptitle("LaMa Iterative Reconstruction Stability Analysis", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "lama_stability_analysis.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[LaMa] Stability plot saved → {path}")


# ---------------------------------------------------------------------------
# Visual comparison grid
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_comparison_grid(
    x_real: torch.Tensor,       # [B, 3, H, W]  ground truth
    y_corrupted: torch.Tensor,  # [B, 3, H, W]  corrupted input
    x_lama: torch.Tensor,       # [B, 3, H, W]  LaMa output
    save_path: str,
    n_images: int = 8,
):
    """
    Save a clearly labeled 3-row grid using matplotlib:
      Row 1 — Corrupted Input      (what LaMa receives)
      Row 2 — LaMa Reconstruction  (LaMa output)
      Row 3 — Ground Truth         (original clean image)

    Each row has a bold row label on the left, and each column is
    numbered (Sample 1, Sample 2, …) along the top.
    All tensors in [-1, 1].
    """
    B = min(x_real.shape[0], n_images)

    def to_hwc(t):
        """Single CHW tensor [-1,1] → HWC float32 in [0,1]."""
        return ((t.clamp(-1, 1) + 1) / 2).cpu().float().permute(1, 2, 0).numpy()

    row_data = [
        ("Corrupted Input",      y_corrupted),
        ("LaMa Reconstruction",  x_lama),
        ("Ground Truth",         x_real),
    ]

    n_rows = len(row_data)
    # Extra left column for row labels
    fig, axes = plt.subplots(
        n_rows, B,
        figsize=(B * 2.2, n_rows * 2.4),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )

    # Ensure axes is always 2-D
    if B == 1:
        axes = axes[:, None]

    ROW_COLORS = ["#d9534f", "#5b9bd5", "#5cb85c"]   # red / blue / green

    for r, (label, tensor) in enumerate(row_data):
        for c in range(B):
            ax = axes[r, c]
            ax.imshow(to_hwc(tensor[c]), interpolation="lanczos")
            ax.set_xticks([]); ax.set_yticks([])

            # Coloured border to visually group each row
            for spine in ax.spines.values():
                spine.set_edgecolor(ROW_COLORS[r])
                spine.set_linewidth(2.5)

            # Column header on the top row only
            if r == 0:
                ax.set_title(f"Sample {c + 1}", fontsize=8, fontweight="bold", pad=4)

        # Row label on the leftmost cell
        axes[r, 0].set_ylabel(
            label,
            fontsize=9,
            fontweight="bold",
            rotation=90,
            labelpad=6,
            color=ROW_COLORS[r],
        )

    # Overall title
    fig.suptitle(
        "LaMa Inpainting — Comparison Grid\n"
        "Red = Corrupted Input  |  Blue = LaMa Reconstruction  |  Green = Ground Truth",
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[LaMa] Labeled comparison grid saved → {save_path}")


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------
def print_comparison_table(
    lama_metrics: Dict[str, float],
    your_metrics: Optional[Dict[str, float]] = None,
):
    SEP = "+" + "-"*22 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    HDR = f"| {'Metric':<20} | {'LaMa (SOTA)':>10} | {'Your Model':>10} | {'Δ (Yours−LaMa)':>10} |"

    print("\n" + SEP)
    print(HDR)
    print(SEP)

    metrics_info = [
        ("PSNR (dB) ↑",  "psnr",  True),
        ("SSIM ↑",        "ssim",  True),
        ("LPIPS ↓",       "lpips", False),
    ]

    for label, key, higher_better in metrics_info:
        lama_val = lama_metrics.get(key, float("nan"))
        if your_metrics is not None:
            your_val = your_metrics.get(key, float("nan"))
            delta    = your_val - lama_val
            # Positive delta means "your model is better" for PSNR/SSIM
            # Negative delta means "your model is better" for LPIPS
            if higher_better:
                symbol = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            else:
                symbol = "▲" if delta < 0 else ("▼" if delta > 0 else "=")
            row = (f"| {label:<20} | {lama_val:>10.4f} | {your_val:>10.4f} | "
                   f"{delta:>+9.4f}{symbol} |")
        else:
            row = f"| {label:<20} | {lama_val:>10.4f} | {'N/A':>10} | {'N/A':>10} |"
        print(row)

    print(SEP)
    if your_metrics is None:
        print("  (Pass --your-model-psnr / --ssim / --lpips to fill the 'Your Model' column)")
    print()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a single CHW tensor in [-1, 1] to HWC uint8 numpy array."""
    t = (t.clamp(-1, 1) + 1) / 2        # [0, 1]
    t = t.cpu().float().permute(1, 2, 0) # HWC
    return (t.numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_lama(
    lama: LamaInpainter,
    val_loader,
    device: str,
    max_batches: Optional[int] = None,
    save_grid: bool = True,
) -> Dict[str, float]:
    """
    Run LaMa on the validation split and compute average PSNR/SSIM/LPIPS.
    Only the MASKING corruption is used, since that is LaMa's native task.
    """
    psnr_sum = ssim_sum = lpips_sum = 0.0
    n = 0
    saved_grid = False

    for batch_idx, x_real in enumerate(tqdm(val_loader, desc="[LaMa] Evaluating")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_real = x_real.to(device)

        # Use corrupt_batch to get a masked version + one-hot vector
        # We keep only the corrupted image here; the mask is re-derived below
        y, c, corrupt_fn = corrupt_batch(x_real)
        y = y.to(device)

        # Build binary mask for LaMa from the difference
        masks = extract_mask_from_corruption(x_real, y)

        # LaMa inference
        x_lama = lama.inpaint_batch(x_real, masks)

        psnr_sum  += psnr(x_lama, x_real)
        ssim_sum  += ssim(x_lama, x_real)
        lpips_sum += compute_lpips(x_lama, x_real, device)
        n += 1

        # Save a visual grid from the first batch
        if save_grid and not saved_grid:
            grid_path = os.path.join(cfg.RESULTS_DIR, "lama_comparison_grid.png")
            save_comparison_grid(x_real, y, x_lama, save_path=grid_path)
            saved_grid = True

    return {
        "psnr":  psnr_sum  / max(n, 1),
        "ssim":  ssim_sum  / max(n, 1),
        "lpips": lpips_sum / max(n, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="SOTA Comparison: LaMa vs Your Corruption-Aware GAN"
    )
    p.add_argument("--data-root",   type=str,   default=cfg.DATA_ROOT)
    p.add_argument("--img-size",    type=int,   default=cfg.IMG_SIZE)
    p.add_argument("--batch-size",  type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--max-batches", type=int,   default=None,
                   help="Cap evaluation at N batches (quick smoke-test).")
    p.add_argument("--skip-stability", action="store_true",
                   help="Skip iterative stability analysis (faster).")
    p.add_argument("--no-grid",     action="store_true",
                   help="Skip saving the visual comparison grid.")

    # Optional: pass in your model's pre-computed metrics for the table
    p.add_argument("--your-model-psnr",  type=float, default=None)
    p.add_argument("--your-model-ssim",  type=float, default=None)
    p.add_argument("--your-model-lpips", type=float, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    cfg.DATA_ROOT  = args.data_root
    cfg.IMG_SIZE   = args.img_size
    cfg.BATCH_SIZE = args.batch_size

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Warning] CUDA not available — LaMa will run on CPU (slower).")

    cfg.make_dir()

    # ── Data ────────────────────────────────────────────────────────────────
    _, val_loader = build_dataloaders(
        root_dir   = cfg.DATA_ROOT,
        img_size   = cfg.IMG_SIZE,
        batch_size = cfg.BATCH_SIZE,
    )

    # ── LaMa ────────────────────────────────────────────────────────────────
    lama = LamaInpainter(device=device)

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("\n[LaMa] Running evaluation on validation set...")
    lama_metrics = evaluate_lama(
        lama,
        val_loader,
        device      = device,
        max_batches = args.max_batches,
        save_grid   = not args.no_grid,
    )

    print(f"\n[LaMa Evaluation Results]")
    print(f"  PSNR : {lama_metrics['psnr']:.2f} dB")
    print(f"  SSIM : {lama_metrics['ssim']:.4f}")
    print(f"  LPIPS: {lama_metrics['lpips']:.4f}")

    # ── Stability Analysis ───────────────────────────────────────────────────
    # if not args.skip_stability:
    #     print("\n[LaMa] Running Iterative Reconstruction Stability Analysis...")
    #     batch = next(iter(val_loader))[:8].to(device)
    #     lama_stability_analysis(lama, batch, device=device)

    # ── Comparison table ────────────────────────────────────────────────────
    # your_metrics = None
    # if all(v is not None for v in [
    #     args.your_model_psnr, args.your_model_ssim, args.your_model_lpips
    # ]):
    #     your_metrics = {
    #         "psnr":  args.your_model_psnr,
    #         "ssim":  args.your_model_ssim,
    #         "lpips": args.your_model_lpips,
    #     }

    # print_comparison_table(lama_metrics, your_metrics)


if __name__ == "__main__":
    main()