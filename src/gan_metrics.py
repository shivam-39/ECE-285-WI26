# gan_metrics.py — For Inception Score and FID

import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.config as cfg
from src.corruption import corrupt_batch, one_hot_vector, _CORRUPTION_FNS, CORRUPTION_NAMES


# InceptionV3 wrapper
# ---------------------------------------------------------------------------
class InceptionFeatureExtractor(nn.Module):
    """
    Wrapper around torchvision InceptionV3 that returns:
        softmax_probs  [B, 1000] — class probabilities (for IS)
        pool3_features [B, 2048] — avgpool activations (for FID)
    Always runs in float32 and eval mode — metric values are sensitive to precision and require a frozen backbone.
    """
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.net = inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
        except TypeError:
            # older torchvision without the Weights enum
            from torchvision.models import inception_v3
            self.net = inception_v3(pretrained=True, transform_input=False)

        self.net.eval()
        self.net.aux_logits = False   # disable auxiliary head

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x : [B, 3, H, W]  float32, values in [-1, 1]
        Returns (probs [B,1000], feats [B,2048])
        """
        x = x.float().clamp(-1.0, 1.0)

        # Rescale [-1,1] -> [0,1] then apply ImageNet normalisation
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406],
                             device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225],
                             device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        # InceptionV3 requires 299×299
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299),
                              mode="bilinear", align_corners=False)

        # Capture pool3 features via forward hook
        pool_out = []
        handle   = self.net.avgpool.register_forward_hook(
            lambda m, i, o: pool_out.append(o.flatten(1))
        )
        logits = self.net(x)
        handle.remove()

        probs = F.softmax(logits, dim=1)    # [B, 1000]
        feats = pool_out[0]    # [B, 2048]
        return probs, feats


# Singleton — load InceptionV3 only once per process
_INCEPTION: Optional[InceptionFeatureExtractor] = None


def _get_inception(device: str) -> InceptionFeatureExtractor:
    global _INCEPTION
    if _INCEPTION is None:
        print("[gan_metrics] Loading pretrained InceptionV3 "
              "(downloads ~100 MB on first run) ...")
        _INCEPTION = InceptionFeatureExtractor()
    return _INCEPTION.to(device)



# Feature collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_gen_features(G, loader, device: str, n_samples: int, corruption_idx: int):
    """
    Generate n_samples reconstructions through G and return their
    Inception (probs [N,1000], feats [N,2048]).
    corruption_idx: -1 random mix (same distribution as training), 0-3  fixed corruption type
    """
    inception = _get_inception(device)
    G.eval()

    all_probs, all_feats, collected = [], [], 0
    pbar = tqdm(total=n_samples, desc="    generating", unit="img", leave=False)

    for x_real in loader:
        if collected >= n_samples:
            break
        x_real = x_real.to(device, non_blocking=True)

        if corruption_idx == -1:
            y, c, _ = corrupt_batch(x_real)
            c = c.to(device)    # corrupt_batch returns c on CPU
        else:
            # Use per-image corruption fn
            B_ = x_real.shape[0]
            y  = torch.stack([_CORRUPTION_FNS[corruption_idx](x_real[i]) for i in range(B_)], dim=0)
            c  = F.one_hot(
                torch.full((B_,), corruption_idx, device=device, dtype=torch.long),
                num_classes=cfg.NUM_CORRUPTION_TYPES,
            ).float()

        x_hat = G(y, c).float().clamp(-1.0, 1.0)
        probs, feats = inception(x_hat)

        all_probs.append(probs.cpu())
        all_feats.append(feats.cpu())
        collected += x_real.shape[0]
        pbar.update(x_real.shape[0])

    pbar.close()
    return (torch.cat(all_probs)[:n_samples],
            torch.cat(all_feats)[:n_samples])


@torch.no_grad()
def _collect_real_features(loader, device: str, n_samples: int):
    """Collect Inception pool3 features for real validation images."""
    inception = _get_inception(device)
    all_feats, collected = [], 0

    for x_real in tqdm(loader, desc="    real feats", unit="batch", leave=False):
        if collected >= n_samples:
            break
        x_real = x_real.to(device, non_blocking=True)
        _, feats = inception(x_real)
        all_feats.append(feats.cpu())
        collected += x_real.shape[0]

    return torch.cat(all_feats)[:n_samples]   # [N, 2048]


# IS computation
# ---------------------------------------------------------------------------
def _compute_is(probs: torch.Tensor, n_splits: int = 10):
    """
    Compute Inception Score (IS) over n_splits. Splits provide a mean, std estimate. Standard n_splits = 10.
    Returns (is_mean, is_std).
    """
    N          = probs.shape[0]
    split_size = N // n_splits
    scores     = []

    for i in range(n_splits):
        p_yx = probs[i * split_size : (i + 1) * split_size]    # [K, 1000]
        p_y  = p_yx.mean(dim=0, keepdim=True)                  # [1, 1000]
        kl   = (p_yx * (torch.log(p_yx + 1e-10) - torch.log(p_y  + 1e-10))).sum(dim=1).mean()
        scores.append(math.exp(kl.item()))

    return float(np.mean(scores)), float(np.std(scores))



# FID computation
# ---------------------------------------------------------------------------
def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Symmetric matrix square root via scipy (preferred) or numpy eigen-decomp.
    scipy.linalg.sqrtm is more numerically stable for near-singular matrices.
    """
    try:
        from scipy.linalg import sqrtm
        result = sqrtm(A)
        if np.iscomplexobj(result):
            imag_max = np.abs(result.imag).max()
            if imag_max > 1e-3:
                print(f"[gan_metrics] Warning: matrix sqrt has imaginary "
                      f"component {imag_max:.2e} — FID may be slightly off.")
            result = result.real
        return result
    except ImportError:
        # Fallback: A = V * D * V^T  ->  A^1/2 = V * D^1/2 * V^T
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 0.0)
        return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _compute_fid(real_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    """
    Computes FID
    real_feats : [N_r, 2048]
    gen_feats  : [N_g, 2048]
    """
    r = real_feats.numpy().astype(np.float64)
    g = gen_feats.numpy().astype(np.float64)

    mu_r, mu_g = r.mean(0), g.mean(0)
    cov_r = np.cov(r, rowvar=False)
    cov_g = np.cov(g, rowvar=False)

    diff_sq    = np.sum((mu_r - mu_g) ** 2)
    cov_sqrt   = _matrix_sqrt(cov_r @ cov_g)
    trace_term = np.trace(cov_r + cov_g - 2.0 * cov_sqrt)

    return float(np.real(diff_sq + trace_term))


# Output saving
# ---------------------------------------------------------------------------
def _save_results(results: dict, save_dir: str = cfg.RESULTS_DIR):
    """
    Print results table to stdout; save bar chart PNG and plain-text file.
    results = {"label": {"is_mean":_, "is_std":_, "fid":_}, _}
    """
    os.makedirs(save_dir, exist_ok=True)

    # Terminal table
    print("\n" + "=" * 54)
    print(f"  {'Corruption':<12}  {'IS (↑ better)':<22}  FID (↓ better)")
    print("=" * 54)
    for name, m in results.items():
        is_str = f"{m['is_mean']:.3f} ± {m['is_std']:.3f}"
        print(f"  {name:<12}  {is_str:<22}  {m['fid']:.2f}")
    print("=" * 54)

    # Bar chart
    names    = list(results.keys())
    is_means = [results[n]["is_mean"] for n in names]
    is_stds  = [results[n]["is_std"]  for n in names]
    fids     = [results[n]["fid"]     for n in names]
    x        = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(names) * 2.5), 5))

    ax1.bar(x, is_means, 0.5, yerr=is_stds, color="steelblue", capsize=5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_title("Inception Score  (↑ better)", fontweight="bold")
    ax1.set_ylabel("IS"); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, fids, 0.5, color="crimson")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_title("Fréchet Inception Distance  (↓ better)", fontweight="bold")
    ax2.set_ylabel("FID"); ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("GAN Metrics — IS & FID", fontsize=13, fontweight="bold")
    plt.tight_layout()

    chart_path = os.path.join(save_dir, "gan_metrics.png")
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  OK -> Chart -> {chart_path}")

    # Plain-text summary
    txt_path = os.path.join(save_dir, "gan_metrics.txt")
    with open(txt_path, "w") as f:
        f.write("GAN Metrics — Inception Score & FID\n")
        f.write("=" * 54 + "\n")
        f.write(f"{'Corruption':<12}  {'IS mean':<12}  {'IS std':<12}  FID\n")
        f.write("-" * 54 + "\n")
        for name, m in results.items():
            f.write(f"{name:<12}  {m['is_mean']:<12.4f}  "
                    f"{m['is_std']:<12.4f}  {m['fid']:.4f}\n")
    print(f"  OK -> Text  -> {txt_path}")


# Public API — called from main.py
# ---------------------------------------------------------------------------
def evaluate_gan_metrics(
    G,
    val_loader,
    device:         str,
    n_samples:      int = cfg.METRIC_SAMPLE_N,
    corruption_idx: int = -1,
    n_splits:       int = 10,
    label:          str = "random",
) -> dict:
    """
    Compute IS and FID for one corruption mode.
    
    Parameters:
    G              : trained Generator  (eval mode will be set internally)
    val_loader     : validation DataLoader  (source of real images for FID)
    device         : 'cuda' or 'cpu'
    n_samples      : images to generate  (≥2048 recommended for FID)
    corruption_idx : -1 = random mix,  0-3 = fixed type
    n_splits       : IS split count  (10 is standard)
    label          : name used in the output table

    Returns dict with keys  is_mean, is_std, fid.
    """
    if n_samples < 2048:
        print(f"[gan_metrics] Warning: n_samples={n_samples} is below the "
              f"recommended minimum of 2048 for a reliable FID estimate.")

    c_display = label if label != "random" else "random (training mix)"
    print(f"\n[GAN Metrics]  corruption={c_display}  n_samples={n_samples}")

    print("  [1/3] Generating images + Inception features ...")
    gen_probs, gen_feats = _collect_gen_features(
        G, val_loader, device, n_samples, corruption_idx)

    print("  [2/3] Real image Inception features ...")
    real_feats = _collect_real_features(val_loader, device, n_samples)

    print("  [3/3] Computing IS and FID ...")
    is_mean, is_std = _compute_is(gen_probs, n_splits)
    fid             = _compute_fid(real_feats, gen_feats)

    result = {"is_mean": is_mean, "is_std": is_std, "fid": fid}
    _save_results({label: result})
    return result


def evaluate_all_corruptions(
    G,
    val_loader,
    device:    str,
    n_samples: int = cfg.METRIC_SAMPLE_N,
    n_splits:  int = 10,
) -> dict:
    """
    Compute IS and FID for each of the 4 corruption types plus the random mix.
    Real features are extracted once and reused for all FID computations.

    Returns dict keyed by corruption name.
    """
    print(f"\n[GAN Metrics]  all corruption modes  n_samples={n_samples}")
    print("  Extracting real image features (done once for all FID calls) ...")
    real_feats = _collect_real_features(val_loader, device, n_samples)

    all_results = {}
    modes = [("random", -1)] + list(enumerate(CORRUPTION_NAMES))

    for entry in modes:
        if isinstance(entry, tuple) and isinstance(entry[0], str):
            name, c_idx = entry          # ("random", -1)
        else:
            c_idx, name = entry          # (0, "mask") etc.

        print(f"\n  ── {name} ──")
        gen_probs, gen_feats = _collect_gen_features(
            G, val_loader, device, n_samples, c_idx)
        is_mean, is_std = _compute_is(gen_probs, n_splits)
        fid             = _compute_fid(real_feats, gen_feats)

        all_results[name] = {"is_mean": is_mean, "is_std": is_std, "fid": fid}
        print(f"    IS={is_mean:.3f} ± {is_std:.3f}   FID={fid:.2f}")

    _save_results(all_results)
    return all_results



