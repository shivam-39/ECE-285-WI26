# visualize.py — Visualization Utilities

import os
import math

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.config as cfg
from src.corruption import corrupt_batch, CORRUPTION_NAMES


# Loss Curves
# ---------------------------------------------------------------------------s
def plot_loss_curves(history: dict, save_dir: str = cfg.PLOT_DIR):
    """
    Plot training & validation loss curves from the history dict.
    history = {
        "train": [ {"g_total": _, "g_adv": _, "g_l1": _, "g_stab": _, "d": _}, _],
        "val":   [ {same keys}, _],
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    train_h = history.get("train", [])
    val_h   = history.get("val",   [])
    if not train_h:
        return

    epochs = list(range(1, len(train_h) + 1))
    keys   = ["g_total", "g_adv", "g_l1", "g_stab", "d"]
    labels = {
        "g_total": "Generator Total",
        "g_adv":   "Generator Adversarial",
        "g_l1":    "Generator L1",
        "g_stab":  "Generator Stability",
        "d":       "Discriminator",
    }
    colors = ["steelblue", "darkorange", "seagreen", "mediumpurple", "crimson"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for ax, key, color in zip(axes, keys, colors):
        t_vals = [d[key] for d in train_h]
        ax.plot(epochs, t_vals, label="Train", color=color, linewidth=1.8)
        if val_h:
            v_vals = [d[key] for d in val_h]
            ax.plot(epochs, v_vals, label="Val", color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax.set_title(labels[key], fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Hide unused subplot
    axes[-1].set_visible(False)

    plt.suptitle("Training Loss Curves — Corruption-Aware GAN", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" OK - Loss curves saved -> {path}")



# Sample Image Grid
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_sample_grid(
    G,
    x_real: torch.Tensor,   # [B, C, H, W] clean images (on device)
    epoch: int,
    device: str,
    save_dir: str = cfg.RESULTS_DIR,
    nrow: int = 4,
):
    """
    For each corruption type, save a 3-row grid:
      Row 1: original clean images
      Row 2: corrupted images
      Row 3: generator reconstructions
    """
    os.makedirs(save_dir, exist_ok=True)
    G.eval()

    n = min(nrow, x_real.shape[0])
    x_clean = x_real[:n]

    # One figure per corruption type
    for c_idx, c_name in enumerate(CORRUPTION_NAMES):
        from src.corruption import corrupt_single, one_hot_vector

        y_list, c_list = [], []
        for i in range(n):
            y_i, c_i = corrupt_single(x_clean[i].to(device), c_idx)
            y_list.append(y_i)
            c_list.append(c_i)
        y = torch.stack(y_list).to(device)
        c = torch.stack(c_list).to(device)

        x_hat = G(y, c)

        # Denorm: [-1,1] → [0,1]
        def dn(t):
            return ((t.clamp(-1, 1) + 1) / 2).to(device)

        grid = torch.cat([dn(x_clean), dn(y), dn(x_hat)], dim=0)  # [3n, C, H, W]
        img_grid = vutils.make_grid(grid, nrow=n, padding=4, normalize=False, pad_value=0.5)

        fig, ax = plt.subplots(figsize=(n * 2.5, 8))
        ax.imshow(img_grid.permute(1, 2, 0).detach().cpu().numpy())
        ax.axis("off")
        tmp_epoch = epoch
        if tmp_epoch == cfg.NUM_EPOCHS:
            tmp_epoch = 100
        ax.set_title(
            f"Epoch {tmp_epoch}  |  Corruption: {c_name.upper()}\n"
            f"Top: Original   Middle: Corrupted   Bottom: Reconstructed",
            fontsize=10,
        )
        path = os.path.join(save_dir, f"epoch_{epoch:04d}_{c_name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()

    print(f" OK - Sample grids saved -> {save_dir}/epoch_{epoch:04d}_*.png")


# Hyperparameter Summary Table (printed to stdout + saved as PNG)
# ---------------------------------------------------------------------------
def print_hyperparameter_table(save_dir: str = cfg.PLOT_DIR): # type: ignore
    """Print a formatted table of all key hyperparameters and save as PNG."""
    rows = [
        ("IMAGE RESOLUTION",        f"{cfg.IMG_SIZE} × {cfg.IMG_SIZE}"),
        ("Channels",                str(cfg.IMG_CHANNELS)),
        ("Dataset limit",           str(cfg.DATASET_LIMIT)),
        ("Train split",             f"{cfg.TRAIN_SPLIT:.0%}"),
        ("", ""),
        ("CORRUPTION",              ""),
        ("Types",                   ", ".join(CORRUPTION_NAMES)),
        ("Sampling probs",          str(cfg.CORRUPTION_PROBS)),
        ("Mask ratio range",        f"[{cfg.MASK_MIN_RATIO}, {cfg.MASK_MAX_RATIO}]"),
        ("Blur σ range",            f"[{cfg.BLUR_SIGMA_MIN}, {cfg.BLUR_SIGMA_MAX}]"),
        ("Low-res scale range",     f"[{cfg.LR_SCALE_MIN}, {cfg.LR_SCALE_MAX}]"),
        ("Noise σ range",           f"[{cfg.NOISE_SIGMA_MIN}, {cfg.NOISE_SIGMA_MAX}]"),
        ("", ""),
        ("MODEL",                   ""),
        ("Gen base filters",        str(cfg.GEN_BASE_FILTERS)),
        ("Gen down-sample stages",  str(cfg.GEN_NUM_DOWNS)),
        ("Disc base filters",       str(cfg.DISC_BASE_FILTERS)),
        ("Disc PatchGAN layers",    str(cfg.DISC_NUM_LAYERS)),
        ("Conditioning dim",        str(cfg.CONDITIONING_DIM)),
        ("", ""),
        ("TRAINING",                ""),
        ("Epochs",                  str(cfg.NUM_EPOCHS)),
        ("Batch size",              str(cfg.BATCH_SIZE)),
        ("LR Generator",            str(cfg.LEARNING_RATE_G)),
        ("LR Discriminator",        str(cfg.LEARNING_RATE_D)),
        ("Adam β₁ / β₂",            f"{cfg.BETA1} / {cfg.BETA2}"),
        ("λ GAN / L1 / Stab",       f"{cfg.LAMBDA_GAN} / {cfg.LAMBDA_L1} / {cfg.LAMBDA_STAB}"),
        ("D steps per G step",      str(cfg.D_STEPS_PER_G)),
        ("AMP (mixed precision)",   str(cfg.USE_AMP)),
        ("Device",                  cfg.DEVICE),
    ]

    # Terminal print
    print("\n" + "="*55)
    print(f"  {'HYPERPARAMETER':<30}  {'VALUE':<20}")
    print("="*55)
    for k, v in rows:
        if k == "":
            print("-"*55)
        elif v == "":
            print(f"  ── {k}")
        else:
            print(f"  {k:<30}  {v}")
    print("="*55 + "\n")

    # Save as image
    os.makedirs(save_dir, exist_ok=True)
    display_rows = [(k, v) for k, v in rows if k != "" or v != ""]
    fig, ax = plt.subplots(figsize=(9, len(display_rows) * 0.35 + 1))
    ax.axis("off")
    table_data = [[k, v] for k, v in display_rows]
    col_labels  = ["Hyperparameter", "Value"]
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.45],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    plt.title("Hyperparameter Settings — Corruption-Aware GAN", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "hyperparameters.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" OK - Hyperparameter table saved -> {path}")
