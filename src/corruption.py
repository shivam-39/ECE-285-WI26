# corruption.py — Corruption Operators

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import src.config as cfg


# Corruption index constants (matches CORRUPTION_PROBS order in config)
# ---------------------------------------------------------------------------
CORRUPTION_MASK =   0
CORRUPTION_BLUR =   1
CORRUPTION_LOWRES = 2
CORRUPTION_NOISE =  3
CORRUPTION_NAMES = ["mask", "blur", "lowres", "noise"]


# Individual corruption functions  (operate on CHW tensors in [-1, 1])
# ---------------------------------------------------------------------------

def apply_mask(x: torch.Tensor) -> torch.Tensor:
    """Random rectangular binary-mask corruption."""
    _, H, W = x.shape
    mask = torch.ones_like(x) # 1 = keep, 0 = remove
    num_holes = random.randint(*cfg.MASK_NUM_HOLES)
    for _ in range(num_holes):
        ratio = random.uniform(cfg.MASK_MIN_RATIO, cfg.MASK_MAX_RATIO)
        hole_h = int(H * ratio ** 0.5)
        hole_w = int(W * ratio ** 0.5)
        y0 = random.randint(0, H - hole_h)
        x0 = random.randint(0, W - hole_w)
        mask[:, y0:y0 + hole_h, x0:x0 + hole_w] = 0.0
    # Fill masked regions with zeros (= gray in [-1,1] normalisation)
    corrupted = x * mask
    return corrupted


def apply_blur(x: torch.Tensor) -> torch.Tensor:
    """Gaussian blur corruption."""
    sigma = random.uniform(cfg.BLUR_SIGMA_MIN, cfg.BLUR_SIGMA_MAX)
    k = cfg.BLUR_KERNEL_SIZE
    # Build 2-D Gaussian kernel
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    gauss = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0) # [k, k]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0) # [1, 1, k, k]
    kernel_2d = kernel_2d.repeat(x.shape[0], 1, 1, 1) # [C, 1, k, k]
    x_4d = x.unsqueeze(0) # [1, C, H, W]
    padding = k // 2
    blurred = F.conv2d(x_4d, kernel_2d, padding=padding, groups=x.shape[0])
    return blurred.squeeze(0).clamp(-1.0, 1.0)


def apply_lowres(x: torch.Tensor) -> torch.Tensor:
    """Low-resolution (downscale -> upscale) corruption."""
    _, H, W = x.shape
    scale = random.randint(cfg.LR_SCALE_MIN, cfg.LR_SCALE_MAX)
    small_h, small_w = max(1, H // scale), max(1, W // scale)
    x_4d = x.unsqueeze(0)
    down = F.interpolate(x_4d, size=(small_h, small_w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
    return up.squeeze(0).clamp(-1.0, 1.0)


def apply_noise(x: torch.Tensor) -> torch.Tensor:
    """Additive Gaussian noise corruption."""
    sigma = random.uniform(cfg.NOISE_SIGMA_MIN, cfg.NOISE_SIGMA_MAX)
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(-1.0, 1.0)


# Dispatcher
# ---------------------------------------------------------------------------
_CORRUPTION_FNS = [apply_mask, apply_blur, apply_lowres, apply_noise]


def sample_corruption_index() -> int:
    """Sample a corruption type index according to configured probabilities."""
    return random.choices(range(len(cfg.CORRUPTION_PROBS)), weights=cfg.CORRUPTION_PROBS, k=1)[0]


def one_hot_vector(idx: int, num_classes: int = cfg.NUM_CORRUPTION_TYPES) -> torch.Tensor:
    """Return a 1-D one-hot tensor of length num_classes."""
    v = torch.zeros(num_classes)
    v[idx] = 1.0
    return v


def corrupt_batch(x_batch: torch.Tensor):
    """Apply a randomly-sampled corruption to each image in a batch.
    Parameters:
        x_batch: Tensor [B, C, H, W] clean images in [-1, 1]
    Returns:
        y_batch: Tensor [B, C, H, W], corrupted images
        c_batch: Tensor [B, K], one-hot corruption-type indicators
        idx_list: list[int], corruption type per sample
    """
    B = x_batch.shape[0]
    y_list, c_list, idx_list = [], [], []
    for i in range(B):
        idx = sample_corruption_index()
        y = _CORRUPTION_FNS[idx](x_batch[i])
        y_list.append(y)
        c_list.append(one_hot_vector(idx))
        idx_list.append(idx)
    y_batch = torch.stack(y_list, dim=0)
    c_batch = torch.stack(c_list, dim=0)
    return y_batch, c_batch, idx_list


def corrupt_single(x: torch.Tensor, corruption_idx: int):
    """Apply a specific corruption to a single CHW image. Returns (corrupted_image, one_hot_vector)"""
    y = _CORRUPTION_FNS[corruption_idx](x)
    c = one_hot_vector(corruption_idx)
    return y, c
