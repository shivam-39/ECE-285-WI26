# Corruption-Aware & Recon-Stable GAN 

Stable and Robust Image Inpainting Using Corruption-Aware GANs and Iterative Reconstruction Stability Analysis.

#### **The project was done as part of the course CSE-291 at UCSD.

---

## Project Structure

```
cars_dcgan/src
config.py        -> ALL hyperparameters
corruption.py    -> Mask / Blur / Low-Res / Noise operators
dataset.py       -> Image DataLoader
models.py        -> Generator (U-Net) + Discriminator (PatchGAN)
losses.py        -> GAN / L1 / Stability loss functions
train.py         -> Training loop (tqdm, AMP, checkpointing, CSV log)
evaluate.py      -> PSNR / SSIM / LPIPS + Stability Analysis
visualize.py     -> Loss curves, sample grids, hyperparameter table
stability.py     -> Full IRSA pipeline (per-corruption drift plots and metrics)
sota_compare.py  -> Comparison against state-of-the-art inpainting model
main.py          -> Entry point (CLI)
requirements.txt -> package dependencies
```

---

## Setup

### 1. Create a virtual environment
```bash
# Recommended - conda & python 3.11
conda create --name cars_gan python=3.11
conda activate cars_gan

# If using UV
uv venv .vcars_gan && source .vcars_gan/bin/activate

# If using venv
python -m venv .vcars_gan && source .vcars_gan/bin/activate

# If on cloud-based env or dsmlp
conda create --name cars_gan python=3.11
source /opt/conda/etc/profile.d/conda.sh
conda activate cars_gan

```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Intel Image Classification dataset
```bash
# Extract and place all images under data directory (sub-folders are fine)
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
```

---

## Usage
```bash
# Train from scratch
python main.py

# Resume training from a checkpoint
python main.py --resume file_path/ckpt_epoch_0050.pt

# Override hyperparameters via CLI
python main.py --img-size 128 --batch-size 16 --epochs 50

# Evaluate a trained model (no training)
python main.py --eval-only --checkpoint file_path/final.pt

# Run iterative stability analysis only
python main.py --stability-only --checkpoint file_path/final.pt

# Evaluate metrics for model performance for all corruption types
python main.py --gan-metrics --checkpoint file_path/final.pt

# Evaluate metrics for model performance for specific corruption_type from [mask, blur, lowres, noise]
python main.py --gan-metrics --checkpoint file_path/final.pt --metrics-corruption corruption_type

# Run Sota Comaprison with trained model - Evaluate LaMa only
python src/sota_compare.py

# Run Sota Comaprison and print comparison against test-model's saved metrics
python src/sota_compare.py --test-model-psnr 28.5 --test-model-ssim 0.88 --test-model-lpips 0.12

# Run Sota Comaprison - Limit to N batches (fast smoke-test):
python src/sota_compare.py --max-batches 5
```

---

<!-- ## Hyperparameter Tuning

**All tunable values live in `config.py`** — do not hard-coded elsewhere.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `IMG_SIZE` | 256 | Reduce to 128 for faster iteration |
| `BATCH_SIZE` | 8 | Safe for 1× GPU w/ 16 GB VRAM at 256² |
| `NUM_EPOCHS` | 100 | |
| `LEARNING_RATE_G / D` | 2e-4 / 1e-4 | DCGAN defaults |
| `LAMBDA_GAN / L1 / STAB` | 1 / 10 / 2 | Loss weights |
| `GEN_BASE_FILTERS` | 64 | Double for capacity |
| `CORRUPTION_PROBS` | [0.25×4] | Adjust sampling distribution |

--- -->

## Outputs

At  each run a folder with current timestamp is created and all below sub-folder are inside that timestamp folder.

<table class="demo">
<thead>
    <tr>
        <th>Path</th>
        <th>Contents</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td><code>plots/hyperparameters.png</code></td>
        <td>Hyperparameter table</td>
    </tr>
    <tr>
        <td><code>plots/loss_curves.png</code></td>
        <td>Training/validation loss curves</td>
    </tr>
    <tr>
        <td><code>results/epoch_XXXX_<type>.png</code></td>
        <td>Sample image grids per corruption</td>
    </tr>
    <tr>
        <td><code>results/stability_analysis.png</code></td>
        <td>PSNR/SSIM/LPIPS over iterations</td>
    </tr>
    <tr>
        <td><code>checkpoints/ckpt_epoch_XXXX.pt</code></td>
        <td>Model checkpoints</td>
    </tr>
    <tr>
        <td><code>logs/training_log.csv</code></td>
        <td>Per-epoch loss values</td>
    </tr>
    <tr>
        <td><code>results/stability_full.png</code></td>
        <td>Full IRSA drift plots per corruption type from stability.py</td>
    </tr>
    <tr>
        <td><code>results/sota_comparison.png</code></td>
        <td>Side-by-side metric comparison against SOTA model</td>
    </tr>
    </tbody>
</table>


## Architecture Summary

**Generator** - Corruption-Aware U-Net (DCGAN)
- Input: corrupted image + spatially-broadcast corruption-type vector c (&isin;R<sup>4</sup>)
- Encoder: 6× strided Conv2d + LeakyReLU + BatchNorm
- Bottleneck: Conv2d + ReLU
- Decoder: 6× ConvTranspose2d + BatchNorm + ReLU (dropout in first 3 layers)
- Skip connections link every encoder layer to its symmetric decoder layer
- Output: Tanh -> [-1, 1]

**Discriminator** — PatchGAN (DCGAN)
- Operates on local image patches (not global scalar output)
- 3× strided Conv2d layers + LeakyReLU + BatchNorm
- Output: spatial logit map [B, 1, H', W']

**Loss**
```
L_total = λ_GAN × L_GAN + λ_L1 × L_L1 + λ_stab × L_stab
```

---

## Initial Run

- GPU: 1× with ≥ 8 GB VRAM (16 GB for comfortable batch_size=8 at 256 x 256)
- RAM: 16 GB
- Suggested quick test: `--img-size 128 --batch-size 16 --epochs 10`
