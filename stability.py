import os
import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import config as cfg
from corruption import _CORRUPTION_FNS, CORRUPTION_NAMES, one_hot_vector
from evaluate   import psnr, ssim, compute_lpips


# Output sub-directory
# ---------------------------------------------------------------------------
STABILITY_DIR = os.path.join(cfg.RESULTS_DIR, "stability")

# Plot colour palette — consistent throughout
_METRIC_COLORS     = {"psnr": "steelblue", "ssim": "darkorange", "lpips": "seagreen"}
_CORRUPTION_COLORS = ["royalblue", "tomato", "mediumseagreen", "mediumpurple"]


# Small shared helpers
# ---------------------------------------------------------------------------
def _dn_np(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


def _iter_border_color(k: int, k_max: int) -> str:
    ratio = k / max(k_max, 1)
    r = int(ratio * 210)
    g = int((1.0 - ratio) * 170 + 30)
    b = 30
    return f"#{r:02x}{g:02x}{b:02x}"


# Core iterative loop (single corruption type)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_single_corruption(
    G,
    x_clean: torch.Tensor,   # [B, C, H, W] clean images, already on device
    corruption_idx: int,
    num_iterations: int,
    device: str,
    save_dir: str,
) -> Tuple[Dict[str, List[float]], List[np.ndarray]]:

    G.eval()
    c_name     = CORRUPTION_NAMES[corruption_idx]
    corrupt_fn = _CORRUPTION_FNS[corruption_idx]

    # Fixed one-hot conditioning vector for this corruption type
    c_vec = torch.stack([one_hot_vector(corruption_idx)] * x_clean.shape[0]).to(device)

    # k = 0  baseline — clean image compared to itself
    history: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
    history["psnr"].append(psnr(x_clean, x_clean))   # -> inf
    history["ssim"].append(ssim(x_clean, x_clean))
    history["lpips"].append(0.0)

    # Collect the first image in the batch as a numpy frame for strip visualisation
    frames: List[np.ndarray] = [_dn_np(x_clean[0])]

    _save_iter_grid(
        x_clean, x_clean, x_clean,
        iteration=0, c_name=c_name,
        metrics=(history["psnr"][0], history["ssim"][0], history["lpips"][0]),
        save_dir=save_dir,
    )

    x_hat = x_clean.clone()

    iter_bar = tqdm(
        range(1, num_iterations + 1),
        desc=f"  [{c_name:7s}]",
        leave=False,
        unit="iter",
    )
    for k in iter_bar:
        # Corrupt the current reconstruction
        y = torch.stack([corrupt_fn(x_hat[i]) for i in range(x_hat.shape[0])]).to(device)

        # Reconstruct
        x_hat = G(y, c_vec)

        # Metrics vs the ORIGINAL clean image
        p = psnr(x_hat, x_clean)
        s = ssim(x_hat, x_clean)
        l = compute_lpips(x_hat, x_clean, device)

        history["psnr"].append(p)
        history["ssim"].append(s)
        history["lpips"].append(l)

        # Store frame for strip visualisation
        frames.append(_dn_np(x_hat[0]))

        iter_bar.set_postfix({"PSNR": f"{p:.2f}", "SSIM": f"{s:.4f}", "LPIPS": f"{l:.4f}"})

        _save_iter_grid(
            x_clean, y, x_hat,
            iteration=k, c_name=c_name,
            metrics=(p, s, l),
            save_dir=save_dir,
        )

    return history, frames


# Per-iteration image grid
# ---------------------------------------------------------------------------
def _save_iter_grid(
    x_clean: torch.Tensor,
    y: torch.Tensor,
    x_hat: torch.Tensor,
    iteration: int,
    c_name: str,
    metrics: tuple,           # (psnr_val, ssim_val, lpips_val)
    save_dir: str,
    nrow: int = 4,
):
    os.makedirs(save_dir, exist_ok=True)
    n = min(nrow, x_clean.shape[0])

    def dn(t: torch.Tensor) -> torch.Tensor:
        return ((t[:n].clamp(-1, 1) + 1) / 2).cpu()

    grid     = torch.cat([dn(x_clean), dn(y), dn(x_hat)], dim=0)
    img_grid = vutils.make_grid(grid, nrow=n, padding=4, normalize=False, pad_value=0.5)

    p, s, l  = metrics
    psnr_str = "inf (baseline)" if math.isinf(p) else f"{p:.2f} dB"

    fig, ax = plt.subplots(figsize=(n * 2.6, 8))
    ax.imshow(img_grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title(
        f"Corruption: {c_name.upper()}   |   Iteration k = {iteration}\n"
        f"PSNR: {psnr_str}    SSIM: {s:.4f}    LPIPS: {l:.4f}\n"
        "[ Top: Original    Middle: Corrupted    Bottom: Reconstructed ]",
        fontsize=10, pad=8,
    )
    path = os.path.join(save_dir, f"stability_{c_name}_iter_{iteration:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


# n=5 special output — per-corruption degradation strip
# ---------------------------------------------------------------------------
def _plot_degradation_strip(
    c_name: str,
    frames: List[np.ndarray],     # [k=0, k=1, ..., k=5]  HWC arrays in [0, 1]
    history: Dict[str, List[float]],
    save_dir: str,
):

    os.makedirs(save_dir, exist_ok=True)
    N = len(frames)   # 6  (k=0 .. k=5)

    # Build GridSpec: alternating [image, arrow, image, arrow, ..., image]
    # width_ratios example for N=6: [10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10]
    col_widths = []
    for i in range(N):
        col_widths.append(10)
        if i < N - 1:
            col_widths.append(1)  # arrow column

    fig = plt.figure(figsize=(N * 2.8, 4.0))
    gs  = gridspec.GridSpec(
        1, len(col_widths),
        figure=fig,
        width_ratios=col_widths,
        wspace=0.04,
    )

    for i, frame in enumerate(frames):
        img_col = i * 2          # image axes column index in GridSpec
        ax_img  = fig.add_subplot(gs[0, img_col])
        ax_img.imshow(frame, vmin=0, vmax=1)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        # Colour-coded border: green -> red as quality drops
        border_color = _iter_border_color(i, N - 1)
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)

        # Label beneath each panel
        if i == 0:
            label_text = "k = 0\n(Original)"
        else:
            p_val = history["psnr"][i]
            s_val = history["ssim"][i]
            l_val = history["lpips"][i]
            psnr_str = f"{p_val:.1f} dB" if not math.isinf(p_val) else "inf"
            label_text = f"k = {i}\nPSNR: {psnr_str}\nSSIM: {s_val:.3f}\nLPIPS: {l_val:.3f}"

        ax_img.set_xlabel(label_text, fontsize=7.5, labelpad=5, ha="center")

        # Arrow between panels (into the next narrow GridSpec column)
        if i < N - 1:
            arr_col = i * 2 + 1
            ax_arr  = fig.add_subplot(gs[0, arr_col])
            ax_arr.axis("off")
            ax_arr.text(
                0.5, 0.5, "→",
                ha="center", va="center",
                fontsize=22, fontweight="bold",
                color="#444444",
                transform=ax_arr.transAxes,
            )

    fig.suptitle(
        f"Reconstruction Degradation over Iterations — {c_name.upper()} Corruption\n"
        "Border colour: green (k=0, highest quality)  →  red (k=5, most degraded)",
        fontsize=11, fontweight="bold", y=1.04,
    )

    path = os.path.join(save_dir, f"stability_{c_name}_degradation_strip.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f" OK - Degradation strip saved -> {path}")


# n=5 special output — all corruptions in one combined grid
# ---------------------------------------------------------------------------
def _plot_all_corruptions_degradation(
    all_frames: Dict[str, List[np.ndarray]],    # {c_name -> [k=0..5]}
    all_histories: Dict[str, Dict[str, List[float]]],
    save_dir: str,
):

    os.makedirs(save_dir, exist_ok=True)

    corruption_names = list(all_frames.keys())
    n_rows = len(corruption_names)   # 4
    n_cols = 6                       # k = 0 .. 5

    # GridSpec column structure per row:
    # [label | img | arr | img | arr | img | arr | img | arr | img | arr | img]
    #   1.5      10   1    10   1    10   1    10   1    10   1    10
    label_w  = 1.5
    img_w    = 10
    arrow_w  = 1
    col_widths = [label_w] + [img_w, arrow_w] * (n_cols - 1) + [img_w]
    # Total cols = 1 + (n_cols-1)*2 + 1 = 1 + 10 + 1 = 12
    n_gs_cols = len(col_widths)   # 12

    row_h    = 1.8   # inches per row — kept short to favour landscape aspect ratio
    fig = plt.figure(figsize=(n_gs_cols * 1.35, n_rows * row_h + 0.8))
    gs  = gridspec.GridSpec(
        n_rows, n_gs_cols,
        figure=fig,
        width_ratios=col_widths,
        hspace=0.15,
        wspace=0.03,
    )

    for row, c_name in enumerate(corruption_names):
        frames  = all_frames[c_name]
        history = all_histories[c_name]

        # Row label on the far left
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")
        ax_label.text(
            0.95, 0.5, c_name.upper(),
            ha="right", va="center",
            fontsize=11, fontweight="bold",
            color=_CORRUPTION_COLORS[row % len(_CORRUPTION_COLORS)],
            transform=ax_label.transAxes,
            rotation=0,
        )

        for i, frame in enumerate(frames):
            img_gs_col = 1 + i * 2    # GridSpec column index for this image

            ax_img = fig.add_subplot(gs[row, img_gs_col])
            ax_img.imshow(frame, vmin=0, vmax=1)
            ax_img.set_xticks([]); ax_img.set_yticks([])

            # Colour-coded border
            border_color = _iter_border_color(i, n_cols - 1)
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.0)

            # Column header — only on the very first row
            if row == 0:
                ax_img.set_title(f"k = {i}", fontsize=9, fontweight="bold", pad=4)

            # Metric label — only on the last row (keeps the grid uncluttered)
            if row == n_rows - 1 and i > 0:
                p_val = history["psnr"][i]
                s_val = history["ssim"][i]
                psnr_str = f"{p_val:.1f}dB" if not math.isinf(p_val) else "inf"
                ax_img.set_xlabel(
                    f"PSNR {psnr_str}  SSIM {s_val:.3f}",
                    fontsize=6.0, labelpad=2,
                )

            # Arrow column (to the right of this image, except after the last image)
            if i < n_cols - 1:
                arr_gs_col = img_gs_col + 1
                ax_arr = fig.add_subplot(gs[row, arr_gs_col])
                ax_arr.axis("off")
                ax_arr.text(
                    0.5, 0.5, "→",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="#444444",
                    transform=ax_arr.transAxes,
                )

    fig.suptitle(
        "Reconstruction Degradation — All Corruption Types  (k = 0 → 5)\n"
        "Left labels: corruption type   |   Border: green = high quality, red = most degraded",
        fontsize=12, fontweight="bold", y=1.01,
    )

    path = os.path.join(save_dir, "stability_all_corruptions_degradation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f" OK - Combined degradation grid saved -> {path}")


# n=5 special output — k=0 vs k=5 for all corruptions
# ---------------------------------------------------------------------------
def _plot_k0_vs_k5(
    all_frames: Dict[str, List[np.ndarray]],     # {c_name -> [k=0..5]}
    all_histories: Dict[str, Dict[str, List[float]]],
    save_dir: str,
):

    os.makedirs(save_dir, exist_ok=True)

    corruption_names = list(all_frames.keys())
    n_rows = len(corruption_names)

    # GridSpec: [label | k=0 img | arrow | k=5 img]
    #            1.5       10       1.2      10
    col_widths = [1.5, 10, 1.2, 10]
    n_gs_cols  = len(col_widths)

    row_h = 2.2   # taller than the 6-column strip — only 2 images per row
    fig = plt.figure(figsize=(n_gs_cols * 2.6, n_rows * row_h + 0.9))
    gs  = gridspec.GridSpec(
        n_rows, n_gs_cols,
        figure=fig,
        width_ratios=col_widths,
        hspace=0.18,
        wspace=0.04,
    )

    for row, c_name in enumerate(corruption_names):
        frames  = all_frames[c_name]
        history = all_histories[c_name]

        # ── Row label ──────────────────────────────────────────────────────
        ax_label = fig.add_subplot(gs[row, 0])
        ax_label.axis("off")
        ax_label.text(
            0.95, 0.5, c_name.upper(),
            ha="right", va="center",
            fontsize=12, fontweight="bold",
            color=_CORRUPTION_COLORS[row % len(_CORRUPTION_COLORS)],
            transform=ax_label.transAxes,
        )

        # ── k=0 image (green border) ───────────────────────────────────────
        ax0 = fig.add_subplot(gs[row, 1])
        ax0.imshow(frames[0], vmin=0, vmax=1)
        ax0.set_xticks([]); ax0.set_yticks([])
        for spine in ax0.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(_iter_border_color(0, 5))
            spine.set_linewidth(3.5)
        if row == 0:
            ax0.set_title("k = 0  (Original)", fontsize=10, fontweight="bold", pad=5)
        # Metric annotation: k=0 is always perfect — just label it clearly
        ax0.set_xlabel("Clean input", fontsize=8, labelpad=3)

        # ── Arrow ──────────────────────────────────────────────────────────
        ax_arr = fig.add_subplot(gs[row, 2])
        ax_arr.axis("off")
        ax_arr.text(
            0.5, 0.5, "→",
            ha="center", va="center",
            fontsize=28, fontweight="bold",
            color="#444444",
            transform=ax_arr.transAxes,
        )

        # ── k=5 image (red border) ─────────────────────────────────────────
        ax5 = fig.add_subplot(gs[row, 3])
        ax5.imshow(frames[5], vmin=0, vmax=1)
        ax5.set_xticks([]); ax5.set_yticks([])
        for spine in ax5.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(_iter_border_color(5, 5))
            spine.set_linewidth(3.5)
        if row == 0:
            ax5.set_title("k = 5  (After 5 Iterations)", fontsize=10, fontweight="bold", pad=5)
        # Metrics at k=5
        p5 = history["psnr"][5]
        s5 = history["ssim"][5]
        l5 = history["lpips"][5]
        psnr_str = f"{p5:.2f} dB" if not math.isinf(p5) else "inf"
        ax5.set_xlabel(
            f"PSNR: {psnr_str}   SSIM: {s5:.3f}   LPIPS: {l5:.3f}",
            fontsize=7.5, labelpad=3,
        )

    fig.suptitle(
        "Before vs After — Original (k=0) and Final Reconstruction (k=5)  |  All Corruption Types\n"
        "Green border = clean original   |   Red border = reconstructed after 5 iterations",
        fontsize=11, fontweight="bold", y=1.02,
    )

    path = os.path.join(save_dir, "stability_k0_vs_k5.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f" OK - k=0 vs k=5 comparison saved -> {path}")


# Per-corruption metric curves
# ---------------------------------------------------------------------------
def _plot_corruption_metrics(
    history: Dict[str, List[float]],
    c_name: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    iters = list(range(len(history["psnr"])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PSNR — skip k=0 (inf) for the line; mark it with a dashed reference line
    ax = axes[0]
    finite_iters = [i for i, v in zip(iters, history["psnr"]) if not math.isinf(v)]
    finite_vals  = [v for v in history["psnr"] if not math.isinf(v)]
    ax.plot(finite_iters, finite_vals, marker="o", color=_METRIC_COLORS["psnr"], linewidth=2)
    if finite_iters and finite_iters[0] > 0:
        ax.axvline(x=0, color="grey", linestyle=":", alpha=0.6, label="k=0: inf (baseline)")
        ax.legend(fontsize=8)
    ax.set_title(f"PSNR  [{c_name.upper()}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration k"); ax.set_ylabel("PSNR (dB, higher is better)"); ax.grid(alpha=0.3)

    # SSIM — start from k=1
    ax = axes[1]
    ax.plot(iters[1:], history["ssim"][1:], marker="o", color=_METRIC_COLORS["ssim"], linewidth=2)
    ax.set_title(f"SSIM  [{c_name.upper()}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration k"); ax.set_ylabel("SSIM (higher is better)"); ax.grid(alpha=0.3)

    # LPIPS — start from k=1
    ax = axes[2]
    ax.plot(iters[1:], history["lpips"][1:], marker="o", color=_METRIC_COLORS["lpips"], linewidth=2)
    ax.set_title(f"LPIPS  [{c_name.upper()}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration k"); ax.set_ylabel("LPIPS (lower is better)"); ax.grid(alpha=0.3)

    plt.suptitle(
        f"Iterative Reconstruction Stability — {c_name.upper()} Corruption",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f"stability_{c_name}_metrics.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" OK - Metric curves saved -> {path}")


# All-corruption summary plot
# ---------------------------------------------------------------------------
def _plot_summary(
    all_histories: Dict[str, Dict[str, List[float]]],
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    metric_keys   = ["psnr",          "ssim",      "lpips"]
    metric_labels = ["PSNR (dB, up)", "SSIM (up)", "LPIPS (down)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[col]
        for c_idx, (c_name, history) in enumerate(all_histories.items()):
            # Skip k=0 baseline so infinite PSNR does not break the y-axis
            xs = list(range(1, len(history[mkey])))
            ys = history[mkey][1:]
            if not xs:
                continue
            ax.plot(
                xs, ys,
                marker="o",
                color=_CORRUPTION_COLORS[c_idx % len(_CORRUPTION_COLORS)],
                label=c_name.upper(),
                linewidth=2,
                markersize=5,
            )
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration k"); ax.set_ylabel(mlabel)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        "Iterative Reconstruction Stability — All Corruption Types",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "stability_all_corruptions_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" OK - Summary plot saved -> {path}")


# Heatmap: metric value at each (corruption x iteration)
# ---------------------------------------------------------------------------
def _plot_metric_heatmap(
    all_histories: Dict[str, Dict[str, List[float]]],
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    c_names   = list(all_histories.keys())
    num_iters = max(len(h["psnr"]) for h in all_histories.values()) - 1  # exclude k=0

    metrics       = ["psnr",          "ssim",      "lpips"]
    metric_labels = ["PSNR (dB, up)", "SSIM (up)", "LPIPS (down)"]
    cmaps         = ["Blues",          "Oranges",   "Greens"]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(c_names) * 0.9 + 1.5)))

    for ax, mkey, mlabel, cmap in zip(axes, metrics, metric_labels, cmaps):
        matrix = np.zeros((len(c_names), num_iters))
        for r, c_name in enumerate(c_names):
            for col, v in enumerate(all_histories[c_name][mkey][1: num_iters + 1]):
                matrix[r, col] = v

        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(num_iters))
        ax.set_xticklabels([f"k={k+1}" for k in range(num_iters)], fontsize=8)
        ax.set_yticks(range(len(c_names)))
        ax.set_yticklabels([n.upper() for n in c_names], fontsize=9)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration")

        for r in range(len(c_names)):
            for col in range(num_iters):
                ax.text(col, r, f"{matrix[r, col]:.2f}",
                        ha="center", va="center", fontsize=7.5, color="black")

    plt.suptitle(
        "Stability Heatmap — Metric x Corruption Type x Iteration",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "stability_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f" OK - Heatmap saved -> {path}")


# Drift report
# ---------------------------------------------------------------------------
def _write_drift_report(
    all_histories: Dict[str, Dict[str, List[float]]],
    save_dir: str,
):

    os.makedirs(save_dir, exist_ok=True)
    sep   = "=" * 75
    lines = [
        sep,
        "  ITERATIVE RECONSTRUCTION STABILITY — DRIFT REPORT",
        sep,
        f"  {'Corruption':<12}  {'Metric':<8}  {'k=1':>8}  {'k=N':>8}  {'Delta':>9}  Verdict",
        "-" * 75,
    ]

    for c_name, history in all_histories.items():
        for mkey in ["psnr", "ssim", "lpips"]:
            vals = history[mkey]
            if len(vals) < 2:
                continue
            v_first = vals[1]           # k=1 (first real reconstruction)
            v_last  = vals[-1]          # k=N (final iteration)
            delta   = v_last - v_first
            thresh  = 1.0 if mkey == "psnr" else 0.05
            # PSNR & SSIM: higher is better -> negative delta = drift
            # LPIPS: lower is better       -> positive delta = drift
            drifting = (delta < -thresh) if mkey != "lpips" else (delta > thresh)
            verdict  = "DRIFTING  !" if drifting else "STABLE   OK"
            lines.append(
                f"  {c_name.upper():<12}  {mkey.upper():<8}  "
                f"{v_first:>8.4f}  {v_last:>8.4f}  {delta:>+9.4f}  {verdict}"
            )
        lines.append("-" * 75)

    lines += [sep, "  Threshold: PSNR > 1 dB drop | SSIM / LPIPS > 0.05 change", sep]
    report = "\n".join(lines)

    print("\n" + report)
    path = os.path.join(save_dir, "stability_drift_report.txt")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f" OK - Drift report saved -> {path}")


# Console metric table
# ---------------------------------------------------------------------------
def _print_metric_table(c_name: str, history: Dict[str, List[float]]):
    print(f"\n  {'Iter':>4}  {'PSNR (dB)':>12}  {'SSIM':>10}  {'LPIPS':>10}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*10}")
    for k, (p, s, l) in enumerate(zip(history["psnr"], history["ssim"], history["lpips"])):
        psnr_str = "    inf(base)" if math.isinf(p) else f"{p:>12.4f}"
        suffix   = "  <- baseline" if k == 0 else ""
        print(f"  {k:>4}  {psnr_str}  {s:>10.4f}  {l:>10.4f}{suffix}")
    print()


# Public API
# ---------------------------------------------------------------------------
def run_stability_analysis(
    G,
    x_clean: torch.Tensor,                          # [B, C, H, W] clean images on device
    num_iterations: int = cfg.STABILITY_ITERATIONS,
    corruption_indices: Optional[List[int]] = None,  # None = all corruption types
    device: str = cfg.DEVICE,
    save_dir: str = STABILITY_DIR,
) -> Dict[str, Dict[str, List[float]]]:

    if corruption_indices is None:
        corruption_indices = list(range(cfg.NUM_CORRUPTION_TYPES))

    os.makedirs(save_dir, exist_ok=True)
    x_clean = x_clean.to(device)

    print("\n" + "=" * 60)
    print(f"  Iterative Reconstruction Stability Analysis")
    print(f"  Iterations  : {num_iterations}")
    print(f"  Images      : {x_clean.shape[0]}")
    print(f"  Corruptions : {[CORRUPTION_NAMES[i] for i in corruption_indices]}")
    print(f"  Save dir    : {save_dir}")
    print("=" * 60)

    all_histories: Dict[str, Dict[str, List[float]]] = {}
    all_frames:    Dict[str, List[np.ndarray]]        = {}  # only used when num_iterations == 5

    for c_idx in corruption_indices:
        c_name = CORRUPTION_NAMES[c_idx]
        print(f"\n[Stability] -- Corruption: {c_name.upper()} -----------------")

        history, frames = _run_single_corruption(
            G              = G,
            x_clean        = x_clean,
            corruption_idx = c_idx,
            num_iterations = num_iterations,
            device         = device,
            save_dir       = save_dir,
        )
        all_histories[c_name] = history
        all_frames[c_name]    = frames

        _print_metric_table(c_name, history)
        _plot_corruption_metrics(history, c_name, save_dir)

        # Per-corruption degradation strip (only when num_iterations == 5)
        if num_iterations == 5:
            _plot_degradation_strip(c_name, frames, history, save_dir)

    # Cross-corruption outputs (only meaningful when more than one corruption was run)
    if len(all_histories) > 1:
        _plot_summary(all_histories, save_dir)
        _plot_metric_heatmap(all_histories, save_dir)

        # Combined degradation grid (only when num_iterations == 5)
        if num_iterations == 5:
            _plot_all_corruptions_degradation(all_frames, all_histories, save_dir)
            _plot_k0_vs_k5(all_frames, all_histories, save_dir)

    _write_drift_report(all_histories, save_dir)

    print(f"\n OK - All stability outputs saved under: {save_dir}\n")
    return all_histories