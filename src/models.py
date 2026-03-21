# models.py — Generator (U-Net) & Discriminator (PatchGAN)

import torch
import torch.nn as nn
import src.config as cfg



# Shared building blocks
# ---------------------------------------------------------------------------
def conv_block(in_ch, out_ch, stride=2, use_bn=True, act="leaky"):
    """Encoder block: Conv -> BN -> LeakyReLU (DCGAN discriminator)."""
    layers = [nn.Conv2d(
        in_ch, 
        out_ch, 
        kernel_size=4, 
        stride=stride, 
        padding=1, 
        bias=not use_bn
    )]

    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch)) # type: ignore
    if act == "leaky":
        layers.append(nn.LeakyReLU(0.2, inplace=True)) # type: ignore
    elif act == "relu":
        layers.append(nn.ReLU(inplace=True)) # type: ignore
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, dropout=False, use_bn=True):
    """Decoder block: ConvTranspose2d -> BN -> (Dropout) -> ReLU (DCGAN generator)."""
    layers = [nn.ConvTranspose2d(
        in_ch, 
        out_ch, 
        kernel_size=4,
        stride=2, 
        padding=1, 
        bias=not use_bn
    )]
    
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)



# Generator — U-Net
# ---------------------------------------------------------------------------
class CorruptionAwareGenerator(nn.Module):
    """
    U-Net generator conditioned on a corruption-type one-hot vector.
    The conditioning vector is spatially broadcast and concatenated with the
    corrupted image at the input stage.
    Input channels : IMG_CHANNELS + NUM_CORRUPTION_TYPES
    Output channels: IMG_CHANNELS  (tanh activation → [-1, 1])
    """

    def __init__(
        self,
        img_channels: int = cfg.IMG_CHANNELS,
        num_corruptions: int = cfg.NUM_CORRUPTION_TYPES,
        base_filters: int = cfg.GEN_BASE_FILTERS,
        num_downs: int = cfg.GEN_NUM_DOWNS,
    ):
        super().__init__()
        self.num_corruptions = num_corruptions
        in_ch = img_channels + num_corruptions  # conditioned input

        f = base_filters    # channel multiplier
        # Encoder (with skip connections). 
        # Block 0: no BN on first encoder layer DCGAN
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_ch, f, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = conv_block(f,    f*2,  act="leaky")
        self.enc2 = conv_block(f*2,  f*4,  act="leaky")
        self.enc3 = conv_block(f*4,  f*8,  act="leaky")
        self.enc4 = conv_block(f*8,  f*8,  act="leaky")
        self.enc5 = conv_block(f*8,  f*8,  act="leaky")

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f*8, f*8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder (with skip connections from encoder). 
        # Decoder input channels = bottleneck out + skip from corresponding encoder
        self.dec5 = deconv_block(f*8,        f*8,  dropout=True)
        self.dec4 = deconv_block(f*8 + f*8,  f*8,  dropout=True)
        self.dec3 = deconv_block(f*8 + f*8,  f*8,  dropout=True)
        self.dec2 = deconv_block(f*8 + f*8,  f*4)
        self.dec1 = deconv_block(f*4 + f*4,  f*2)
        self.dec0 = deconv_block(f*2 + f*2,  f)

        # Output layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(f + f, img_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """DCGAN weight initialisation: N(0, 0.02) for conv/bn."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        y : [B, C, H, W] — corrupted image in [-1, 1]
        c : [B, K] — one-hot corruption type indicator
        """
        B, _, H, W = y.shape
        # Broadcast c spatially and concatenate
        c_spatial = c.view(B, self.num_corruptions, 1, 1).expand(B, -1, H, W)
        x = torch.cat([y, c_spatial], dim=1)    # [B, C+K, H, W]

        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Bottleneck
        b  = self.bottleneck(e5)

        # Decoder with skip connections
        d5 = self.dec5(b)
        d4 = self.dec4(torch.cat([d5, e5], dim=1))
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        d1 = self.dec1(torch.cat([d2, e2], dim=1))
        d0 = self.dec0(torch.cat([d1, e1], dim=1))

        out = self.out_conv(torch.cat([d0, e0], dim=1))
        return out


# Discriminator — PatchGAN (DCGAN)
# ---------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that evaluates local NxN patches.
    Produces a spatial map of real/fake scores rather than a single scalar.
    Input: (real or fake image)  [B, C, H, W]
    """

    def __init__(
        self,
        img_channels: int  = cfg.IMG_CHANNELS,
        base_filters: int  = cfg.DISC_BASE_FILTERS,
        num_layers: int = cfg.DISC_NUM_LAYERS,
    ):
        super().__init__()
        f = base_filters
        layers = []

        # Layer 0 — no BN (DCGAN convention)
        layers += [
            nn.Conv2d(img_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Intermediate layers — channels double (capped at f*8)
        in_f, out_f = f, f * 2
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_f  = out_f
            out_f = min(out_f * 2, f * 8)

        # Final conv — stride 1, one output channel (logit)
        layers += [
            nn.Conv2d(in_f, min(in_f * 2, f * 8), kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(min(in_f * 2, f * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(min(in_f * 2, f * 8), 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns spatial logit map [B, 1, H', W'] — no sigmoid applied here."""
        return self.model(x)


# Model / summary helper
# ---------------------------------------------------------------------------
def build_models(device: str = cfg.DEVICE):
    """Instantiate and move generator + discriminator to device."""
    G = CorruptionAwareGenerator().to(device)
    D = PatchDiscriminator().to(device)
    return G, D


def print_model_summary(model: nn.Module, name: str = "Model"):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"[{name}] Trainable parameters: {num_params:,}")
