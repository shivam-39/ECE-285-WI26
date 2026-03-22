# train.py — Training Loop

import os
import csv
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# PyTorch 2.2+ preferred path
try:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast
    
    def autocast(enabled=True): # wrap to keep call-site uniform
        return _autocast("cuda", enabled=enabled)

# PyTorch 1.x / 2.0 path
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
    except ImportError:
        # Absolute fallback: AMP disabled, use no-op stubs
        import contextlib

        cfg.USE_AMP = False
        @contextlib.contextmanager
        def autocast(enabled=False):
            yield

        class GradScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass


import src.config as cfg
from src.corruption import corrupt_batch
from src.losses import TotalLoss
from src.visualize import save_sample_grid, plot_loss_curves


# Optimiser factory
# ---------------------------------------------------------------------------
def build_optimizers(G, D):
    opt_G = torch.optim.Adam(
        G.parameters(),
        lr=cfg.LEARNING_RATE_G,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    opt_D = torch.optim.Adam(
        D.parameters(),
        lr=cfg.LEARNING_RATE_D,
        betas=(cfg.BETA1, cfg.BETA2),
    )
    return opt_G, opt_D


# Single epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    epoch: int,
    G, D,
    train_loader,
    opt_G, opt_D,
    loss_fn: TotalLoss,
    scaler_G: GradScaler,
    scaler_D: GradScaler,
    device: str,
) -> dict:
    """Run one full epoch; returns dict of mean losses."""
    G.train(); D.train()

    sum_losses = {k: 0.0 for k in ["g_total", "g_adv", "g_l1", "g_stab", "d"]}
    n_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.NUM_EPOCHS}]", leave=True, unit="batch")

    for x_real in pbar:
        x_real = x_real.to(device, non_blocking=True)

        # Apply corruptions on-the-fly 
        y, c, _ = corrupt_batch(x_real)
        y = y.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        d_loss = torch.Tensor()

        # Step 1 — Update Discriminator  (D_STEPS_PER_G times)
        for _ in range(cfg.D_STEPS_PER_G):
            opt_D.zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    x_fake = G(y, c)
                d_loss = loss_fn.discriminator_loss(D, x_real, x_fake)

            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

        # Step 2 — Update Generator
        opt_G.zero_grad(set_to_none=True)
        with autocast():
            x_fake = G(y, c)
            g_losses = loss_fn.generator_loss(D, G, x_real, y, c, x_fake, corrupt_batch)

        scaler_G.scale(g_losses["total"]).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        # Accumulate losses
        sum_losses["g_total"] += g_losses["total"].item()
        sum_losses["g_adv"]   += g_losses["adv"].item()
        sum_losses["g_l1"]    += g_losses["l1"].item()
        sum_losses["g_stab"]  += g_losses["stab"].item()
        sum_losses["d"]       += d_loss.item()

        pbar.set_postfix({
            "G": f"{g_losses['total'].item():.4f}",
            "D": f"{d_loss.item():.4f}",
            "L1": f"{g_losses['l1'].item():.4f}",
        })

    mean_losses = {k: v / n_batches for k, v in sum_losses.items()}
    return mean_losses


# Validation pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(G, D, val_loader, loss_fn: TotalLoss, device: str) -> dict:
    G.eval(); D.eval()
    sum_losses = {k: 0.0 for k in ["g_total", "g_adv", "g_l1", "g_stab", "d"]}
    n_batches = len(val_loader)

    for x_real in tqdm(val_loader, desc="  ↳ Validation", leave=False, unit="batch"):
        x_real = x_real.to(device, non_blocking=True)
        y, c, _ = corrupt_batch(x_real)
        y = y.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)

        x_fake = G(y, c)
        d_loss = loss_fn.discriminator_loss(D, x_real, x_fake)
        g_losses = loss_fn.generator_loss(D, G, x_real, y, c, x_fake, corrupt_batch)

        sum_losses["g_total"] += g_losses["total"].item()
        sum_losses["g_adv"]   += g_losses["adv"].item()
        sum_losses["g_l1"]    += g_losses["l1"].item()
        sum_losses["g_stab"]  += g_losses["stab"].item()
        sum_losses["d"]       += d_loss.item()

    return {k: v / n_batches for k, v in sum_losses.items()}


# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(epoch, G, D, opt_G, opt_D, history, filename):
    torch.save({
        "epoch":   epoch,
        "G_state": G.state_dict(),
        "D_state": D.state_dict(),
        "opt_G":   opt_G.state_dict(),
        "opt_D":   opt_D.state_dict(),
        "history": history,
    }, filename)
    print(f"OK - Checkpoint saved -> {filename}")


def load_checkpoint(path, G, D, opt_G, opt_D, device):
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    print(f"OK - Checkpoint loaded from epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt.get("history", {})


# CSV logger
# ---------------------------------------------------------------------------
class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = None

    def write(self, row: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# Main training function
# ---------------------------------------------------------------------------
def train(G, D, train_loader, val_loader, device: str, resume: str):
    """
    Full training loop.
    Parameters:
        G, D: Generator, Discriminator
        train_loader: training DataLoader
        val_loader: validation DataLoader
        device: 'cuda' or 'cpu'
        resume: optional path to a checkpoint .pt file to resume from
    """
    opt_G, opt_D = build_optimizers(G, D)
    loss_fn      = TotalLoss(device=device)
    scaler_G     = GradScaler(enabled=cfg.USE_AMP)
    scaler_D     = GradScaler(enabled=cfg.USE_AMP)

    start_epoch = 1
    history = {"train": [], "val": []}      # list of dicts per epoch

    if resume and os.path.isfile(resume):
        start_epoch, history = load_checkpoint(resume, G, D, opt_G, opt_D, device)
        start_epoch += 1

    csv_log = CSVLogger(os.path.join(cfg.LOG_DIR, "training_log.csv")) # type: ignore

    # get a fixed eval batch for visualisation
    eval_batch = next(iter(val_loader))[:cfg.NUM_EVAL_IMAGES].to(device)

    print("\n" + "="*60)
    print(f"  Starting training for {cfg.NUM_EPOCHS} epochs")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {cfg.BATCH_SIZE}")
    print(f"  Image size   : {cfg.IMG_SIZE}×{cfg.IMG_SIZE}")
    print(f"  LR (G / D)   : {cfg.LEARNING_RATE_G} / {cfg.LEARNING_RATE_D}")
    print(f"  Loss weights : GAN={cfg.LAMBDA_GAN}  L1={cfg.LAMBDA_L1}  Stab={cfg.LAMBDA_STAB}")
    print("="*60 + "\n")

    t0 = time.time()

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        # Training
        train_losses = train_one_epoch(epoch, G, D, train_loader, opt_G, opt_D, loss_fn, scaler_G, scaler_D, device)

        # Validation
        val_losses = validate(G, D, val_loader, loss_fn, device)

        # Logging
        history["train"].append(train_losses)
        history["val"].append(val_losses)

        log_row = {"epoch": epoch}
        log_row.update({f"train_{k}": v for k, v in train_losses.items()})
        log_row.update({f"val_{k}":   v for k, v in val_losses.items()})
        csv_log.write(log_row)

        elapsed = (time.time() - t0) / 60
        print(
            f"\n  Epoch {epoch:3d}/{cfg.NUM_EPOCHS} | "
            f"G_total={train_losses['g_total']:.4f}  "
            f"D={train_losses['d']:.4f}  "
            f"L1={train_losses['g_l1']:.4f}  "
            f"Stab={train_losses['g_stab']:.4f}  "
            f"[val G={val_losses['g_total']:.4f}]  "
            f"Elapsed: {elapsed:.1f} min\n"
        )

        # Sample images
        if epoch % cfg.EVAL_INTERVAL == 0 or epoch == 1:
            save_sample_grid(G, eval_batch, epoch, device)
            plot_loss_curves(history)

        # Checkpoint
        if epoch % cfg.CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"ckpt_epoch_{epoch:04d}.pt")
            save_checkpoint(epoch, G, D, opt_G, opt_D, history, ckpt_path)

    # Final artefacts
    save_checkpoint(cfg.NUM_EPOCHS, G, D, opt_G, opt_D, history, os.path.join(cfg.CHECKPOINT_DIR, "final.pt"))
    plot_loss_curves(history)
    csv_log.close()
    print(f"\nOK - Training complete in {(time.time()-t0)/60:.1f} minutes.")
