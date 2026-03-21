# main.py — Entry Point

import argparse
import random
import os

import numpy as np
import torch

import src.config as cfg
from dataset   import build_dataloaders
from models    import build_models, print_model_summary
from train     import train, load_checkpoint
from evaluate  import evaluate, stability_analysis
from visualize import print_hyperparameter_table, save_sample_grid
from stability import run_stability_analysis
from gan_metrics import evaluate_gan_metrics, evaluate_all_corruptions



# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # Determining ops
    torch.backends.cudnn.benchmark = True # faster on fixed-size inputs



# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Corruption-Aware GAN for Image Inpainting (ECE-285)")

    p.add_argument("--resume", 
                   type=str, 
                   default=None, 
                   help="Path to checkpoint .pt file to resume training.")
    
    p.add_argument("--eval-only", 
                   action="store_true", 
                   help="Skip training; run evaluation on the val set.")
    
    p.add_argument("--stability-only", 
                   action="store_true", 
                   help="Skip training; run stability analysis only.")
    
    p.add_argument("--checkpoint", 
                   type=str, 
                   default=None, 
                   help="Checkpoint to load for eval/stability-only modes.")
    
    p.add_argument("--data-root", 
                   type=str, 
                   default=cfg.DATA_ROOT, 
                   help=f"Path to image dataset (default: {cfg.DATA_ROOT})")
    
    p.add_argument("--img-size", 
                   type=int, 
                   default=cfg.IMG_SIZE, 
                   help=f"Image resolution (default: {cfg.IMG_SIZE})")
    
    p.add_argument("--batch-size", 
                   type=int, 
                   default=cfg.BATCH_SIZE, 
                   help=f"Batch size (default: {cfg.BATCH_SIZE})")
    
    p.add_argument("--epochs", 
                   type=int, 
                   default=cfg.NUM_EPOCHS, 
                   help=f"Number of training epochs (default: {cfg.NUM_EPOCHS})")
    
    p.add_argument("--gan-metrics",   
                   action="store_true",
                   help="Compute Inception Score + FID using a trained checkpoint.")
    
    p.add_argument("--metrics-n",     
                   type=int, 
                   default=5000,
                   help="Number of images for IS/FID evaluation (default: 5000).")
    
    p.add_argument("--metrics-corruption", 
                   type=str, 
                   default="random",
                   choices=["random", "mask", "blur", "lowres", "noise", "all"],
                   help="Corruption type to use for IS/FID: "
                        "random (training mix) | mask | blur | lowres | noise | "
                        "all (runs each type separately).")

    return p.parse_args()



# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Override config from CLI args
    cfg.DATA_ROOT   = args.data_root
    cfg.IMG_SIZE    = args.img_size
    cfg.BATCH_SIZE  = args.batch_size
    cfg.NUM_EPOCHS  = args.epochs

    set_seed(cfg.SEED)

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[Warning] CUDA not available — running on CPU (training will be slow).")
        cfg.DEVICE = "cpu"
        cfg.USE_AMP = False
    else:
        print("[OK] CUDA available — running on GPU (training will be fast).")

    # Check and create dirs if required.
    cfg.make_dir() 
    # Print hyperparameter table
    print_hyperparameter_table()
    
    # Data
    train_loader, val_loader = build_dataloaders(
        root_dir = cfg.DATA_ROOT,
        img_size = cfg.IMG_SIZE,
        batch_size = cfg.BATCH_SIZE,
    )

    # Models
    G, D = build_models(device)
    print_model_summary(G, "Generator")
    print_model_summary(D, "Discriminator")

    # Load checkpoint if needed
    ckpt_path = args.checkpoint or args.resume
    if ckpt_path and os.path.isfile(ckpt_path):
        from train import build_optimizers
        opt_G, opt_D = build_optimizers(G, D)
        load_checkpoint(ckpt_path, G, D, opt_G, opt_D, device)

    # Eval only 
    if args.eval_only:
        metrics = evaluate(G, val_loader, device)
        print(f"\n[Evaluation Results]")
        print(f" PSNR: {metrics['psnr']:.2f} dB")
        print(f" SSIM: {metrics['ssim']:.4f}")
        print(f" LPIPS: {metrics['lpips']:.4f}")

        # Also save sample images
        batch = next(iter(val_loader))[:cfg.NUM_EVAL_IMAGES].to(device)
        save_sample_grid(G, batch, epoch=0, device=device)
        return

    # Stability only 
    if args.stability_only:
        batch = next(iter(val_loader))[:8].to(device)
        for c_idx in range(cfg.NUM_CORRUPTION_TYPES):
            print(f"\n[Stability Analysis — Corruption: {c_idx}]")
            stability_analysis(G, batch, corruption_idx=c_idx, device=device)
        run_stability_analysis(G, batch, device=device)
        return
    
    # GAN metrics (IS + FID) 
    if args.gan_metrics:
        from corruption import CORRUPTION_NAMES
        if not (args.checkpoint or args.resume):
            print("[Error] --gan-metrics requires --checkpoint <path>")
            return
 
        c_arg = args.metrics_corruption
        if c_arg == "all":
            evaluate_all_corruptions(G, val_loader, device,
                                     n_samples=args.metrics_n)
        else:
            c_idx = -1 if c_arg == "random" else CORRUPTION_NAMES.index(c_arg)
            evaluate_gan_metrics(G, val_loader, device,
                                 n_samples=args.metrics_n,
                                 corruption_idx=c_idx)
        return

    # Full training
    train(G, D, train_loader, val_loader, device=device, resume=args.resume)

    # Final evaluation
    print("\n[Final Evaluation on Validation Set]")
    metrics = evaluate(G, val_loader, device)
    print(f" PSNR: {metrics['psnr']:.2f} dB")
    print(f" SSIM: {metrics['ssim']:.4f}")
    print(f" LPIPS: {metrics['lpips']:.4f}")

    # Stability analysis 
    print("\n[Running Iterative Reconstruction Stability Analysis]")
    batch = next(iter(val_loader))[:8].to(device)
    for c_idx in range(cfg.NUM_CORRUPTION_TYPES):
        stability_analysis(G, batch, corruption_idx=c_idx, device=device)
    run_stability_analysis(G, batch, device=device)


if __name__ == "__main__":
    main()
