# losses.py — Loss Functions

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg



# Adversarial Loss (standard GAN with label smoothing)
# ---------------------------------------------------------------------------
class GANLoss(nn.Module):
    """GAN loss compatible with the PatchGAN discriminator output (a logit map rather than a single scalar).
    Supports two modes:
      bce - Binary Cross-Entropy (standard GAN)
      lsgan - Least-Squares GAN (more stable gradients)
    """

    def __init__(self, mode: str = "bce", smooth: float = 0.1):
        """
        mode: 'bce' or 'lsgan'
        smooth: one-sided label smoothing for real targets (0 = disabled)
        """
        super().__init__()
        self.mode   = mode
        self.smooth = smooth

    def _target(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if is_real:
            val = 1.0 - self.smooth # e.g. 0.9 with smooth=0.1
        else:
            val = 0.0
        return torch.full_like(pred, val)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = self._target(pred, is_real)
        if self.mode == "bce":
            return F.binary_cross_entropy_with_logits(pred, target)
        elif self.mode == "lsgan":
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown GAN loss mode: {self.mode}")



# Stability Loss
# ---------------------------------------------------------------------------
def stability_loss(
    G,
    y: torch.Tensor,
    c: torch.Tensor,
    x_hat: torch.Tensor,
    corrupt_fn,
    device: str = cfg.DEVICE,
) -> torch.Tensor:
    """
    We re-corrupt the single-pass output x_hat, then reconstruct again. 
    The corruption type is kept the same (same c) for determining comparison.
    Parameters:
        G: Generator
        y: [B, C, H, W] corrupted input
        c: [B, K] one-hot corruption indicator
        x_hat: [B, C, H, W] = G(y, c)  (already computed, avoids a forward pass)
        corrupt_fn: callable — takes a [B,C,H,W] batch and the c tensor, returns (corrupted_batch, c_batch), we use the same c here.
        device: torch device string
    """
    with torch.no_grad():
        # Re-corrupt x_hat using the same corruption type as the original pass
        y2 = _re_corrupt(x_hat.detach(), c, corrupt_fn, device)

    # Second generator pass  (gradients flow through G here)
    x_hat2 = G(y2, c)
    loss = F.l1_loss(x_hat2, x_hat.detach())
    return loss


def _re_corrupt(
    x_batch: torch.Tensor,
    c_batch: torch.Tensor,
    corrupt_fn,
    device: str,
) -> torch.Tensor:
    """
    Apply the same corruption type (from c_batch) to each image in x_batch.
    This is called inside stability_loss to avoid gradient tracking.
    """
    from corruption import _CORRUPTION_FNS
    B = x_batch.shape[0]
    y_list = []
    for i in range(B):
        c_idx = int(c_batch[i].argmax().item())
        y_i = _CORRUPTION_FNS[c_idx](x_batch[i].to(device))
        y_list.append(y_i)
    return torch.stack(y_list).to(device)



# Total Loss Bundle
# ---------------------------------------------------------------------------
class TotalLoss:
    """Wraps all loss components. Returns a dict of scalar tensors for logging."""

    def __init__(self, device: str = cfg.DEVICE):
        self.adv_loss = GANLoss(mode="bce").to(device)
        self.device   = device

    def discriminator_loss(
        self,
        D,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Standard GAN discriminator objective:"""
        pred_real = D(x_real)
        pred_fake = D(x_fake.detach()) # detach: don't update G
        loss_real = self.adv_loss(pred_real, is_real=True)
        loss_fake = self.adv_loss(pred_fake, is_real=False)
        return (loss_real + loss_fake) * 0.5

    def generator_loss(
        self,
        D,
        G,
        x_real: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        x_hat: torch.Tensor,
        corrupt_fn,
    ) -> dict:
        """Total generator objective"""

        # Adversarial 
        pred_fake = D(x_hat)
        l_adv = self.adv_loss(pred_fake, is_real=True)

        # Pixel-level L1 reconstruction
        l_l1 = F.l1_loss(x_hat, x_real)

        # Stability regularisation
        l_stab = stability_loss(G, y, c, x_hat, corrupt_fn, device=self.device)

        l_total = (
            cfg.LAMBDA_GAN * l_adv
            + cfg.LAMBDA_L1 * l_l1
            + cfg.LAMBDA_STAB * l_stab
        )

        return {
            "total": l_total,
            "adv": l_adv,
            "l1": l_l1,
            "stab": l_stab,
        }
