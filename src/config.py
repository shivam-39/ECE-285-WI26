# config.py — Hyperparameter & Constants File

import os
from datetime import datetime


# Paths
# ---------------------------------------------------------------------------
DATA_ROOT           = "./../data" # Base directory for the dataset
RESULTS_BASE_DIR    = "./../" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CHECKPOINT_DIR      = RESULTS_BASE_DIR + "/checkpoints" # Where model weights are saved
LOG_DIR             = RESULTS_BASE_DIR + "/logs" # CSV logs
RESULTS_DIR         = RESULTS_BASE_DIR + "/results" # Generated image samples
PLOT_DIR            = RESULTS_BASE_DIR + "/plots" # Loss curve plots


# Derived — do not edit
def make_dir():
    for _d in [RESULTS_BASE_DIR, DATA_ROOT, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, PLOT_DIR]:
        os.makedirs(_d, exist_ok=True)


# Data / Preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE        = 256 # Resolution (H = W). We use 256, can be reduced to 128 for faster iteration.
IMG_CHANNELS    = 3 # RGB
NUM_WORKERS     = 4 # Dataloader workers
TRAIN_SPLIT     = 0.9 # Fraction used for training
DATASET_LIMIT   = None # Max images loaded from disk (None = all)


# Corruption Sampling
# ---------------------------------------------------------------------------
# Probability for each corruption type: [mask, blur, lowres, noise], must sum to 1.0
CORRUPTION_PROBS = [0.25, 0.25, 0.25, 0.25]

# Mask
MASK_MIN_RATIO = 0.10 # Min fraction of image to mask
MASK_MAX_RATIO = 0.40 # Max fraction of image to mask
MASK_NUM_HOLES = (1, 4) # (min, max) number of rectangular holes

# Blur
BLUR_SIGMA_MIN   = 1.0
BLUR_SIGMA_MAX   = 5.0
BLUR_KERNEL_SIZE = 21 # Must be odd

# Low-Resolution
LR_SCALE_MIN = 2 # Downscale factor range min
LR_SCALE_MAX = 8 # Downscale factor range max

# Additive Gaussian Noise
# In [-1, 1] pixel range
NOISE_SIGMA_MIN = 0.02
NOISE_SIGMA_MAX = 0.15


# Model Architecture
# ---------------------------------------------------------------------------
NUM_CORRUPTION_TYPES = 4 # K in our report
CONDITIONING_DIM     = NUM_CORRUPTION_TYPES # c is an element of R^K

# Generator (U-Net)
GEN_BASE_FILTERS = 64 # Channels in first encoder layer, doubles each level
GEN_NUM_DOWNS    = 6 # Number of down-sampling stages (encoder depth). At IMG_SIZE=256 -> bottleneck is 256/2^6 = 4×4

# Discriminator (PatchDCGAN)
DISC_BASE_FILTERS = 64
DISC_NUM_LAYERS   = 5 # no of layers in PatchGAN


# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS      = 100
BATCH_SIZE      = 8 # 8 for 1xGPU with 16 GB RAM at 256x256 img
LEARNING_RATE_G = 1e-4 # Generator LR (Adam)
LEARNING_RATE_D = 5e-5 # Discriminator LR (Adam)
BETA1           = 0.5 # Adam beta_1
BETA2           = 0.999 # Adam beta_2

# Loss weights
LAMBDA_GAN  = 8.0 # GAN Loss weight
LAMBDA_L1   = 4.0 # Pixel reconstruction weight
LAMBDA_STAB = 1.0 # Stability regularisation weight

# Discriminator update frequency
D_STEPS_PER_G = 1 # Train D this many times per G update

# Evaluation & Logging
EVAL_INTERVAL        = 20 # Evaluate & save sample images every N epochs
CHECKPOINT_INTERVAL  = 20 # Save checkpoint every N epochs
NUM_EVAL_IMAGES      = 16 # Images shown in the sample grid
STABILITY_ITERATIONS = 5 # Number of iterative reconstruction cycles


# Misc
# ---------------------------------------------------------------------------
SEED    = 42
USE_AMP = True # Automatic Mixed Precision (speeds up on modern GPUs)
DEVICE  = "cuda" # "cuda" or "cpu"










