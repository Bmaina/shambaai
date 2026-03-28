"""
ShambaAI GeoSR — Enhanced Deep Super-Resolution for Sentinel-2 imagery.

Architecture: EDSR (Enhanced Deep Residual Networks for Single Image SR)
- Batch Normalisation REMOVED to preserve absolute spectral reflectance values.
  Standard BN normalises feature maps, destroying the radiometric accuracy
  needed for NDVI / Red-Edge stress index computation.
- Spectral Fidelity Loss: Pixel L1 + Spectral Angle Mapper (SAM) + Edge loss.
- Scale factor: 4x  (10m Sentinel-2 → 2.5m synthetic resolution)

Reference: Lim et al. 2017 "Enhanced Deep Residual Networks for Single Image
Super-Resolution" — adapted for multispectral remote sensing.

Why not GAN?
  GANs hallucinate visually plausible but spectrally inaccurate textures.
  For NDVI and Red-Edge indices, spectral fidelity beats visual sharpness.
  EDSR wins on scientific accuracy, not perceptual quality.

Training strategy (Wald's Protocol):
  1. Take real 10m Sentinel-2 bands.
  2. Downsample 4x → 40m (simulates coarser input).
  3. Train model to reconstruct the 10m original from 40m input.
  4. At inference, feed real 10m → output 2.5m synthetic resolution.
  This avoids needing expensive PlanetScope training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    EDSR residual block: Conv → ReLU → Conv, NO BatchNorm.
    Residual scaling (0.1) stabilises deep network training without BN.
    """
    def __init__(self, n_feats: int = 64, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class UpsampleBlock(nn.Module):
    """Pixel-shuffle upsampling. Sharper and cheaper than bilinear."""
    def __init__(self, scale: int, n_feats: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class EDSR(nn.Module):
    """
    GeoSR EDSR backbone for multispectral super-resolution.

    Args:
        n_bands:     Number of input spectral bands (default 4: B4, B8, B5, B11)
        scale:       Upscaling factor (4 → 10m to 2.5m)
        n_feats:     Feature map depth (64 = lightweight, 256 = full EDSR)
        n_resblocks: Number of residual blocks (16 = baseline, 32 = EDSR+)
    """
    def __init__(
        self,
        n_bands: int = 4,
        scale: int = 4,
        n_feats: int = 64,
        n_resblocks: int = 16,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale

        # Head: map input bands to feature space
        self.head = nn.Conv2d(n_bands, n_feats, 3, padding=1)

        # Body: stack of residual blocks (NO BatchNorm)
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),  # post-residual conv
        )

        # Upsampling
        self.upsample = UpsampleBlock(scale, n_feats)

        # Tail: project back to band space
        self.tail = nn.Conv2d(n_feats, n_bands, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bicubic residual: pre-upscale then add network correction
        x_up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        res  = self.body(feat)
        res  = self.upsample(feat + res)  # skip from head
        out  = self.tail(res)
        return x_up + out  # global residual learning


# ── Loss Functions ────────────────────────────────────────────────────────────

class SpectralAngleMapperLoss(nn.Module):
    """
    SAM Loss: measures angular distance between spectral vectors.
    Ensures the AI preserves the spectral SIGNATURE (shape) of each pixel,
    not just the intensity. Critical for NDVI/Red-Edge accuracy.
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W) — normalise along channel dim
        pred_n   = F.normalize(pred,   dim=1, eps=1e-8)
        target_n = F.normalize(target, dim=1, eps=1e-8)
        cos_sim  = (pred_n * target_n).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        sam      = torch.acos(cos_sim)  # angle in radians
        return sam.mean()


class EdgePreservingLoss(nn.Module):
    """
    Sobel edge loss. Penalises blurring of field boundaries.
    Fall Armyworm infestations START at field edges — preserving
    boundary sharpness is agronomically critical, not just cosmetic.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        # Register as buffers (move to device automatically, not trained)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Sobel per band, then average
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.view(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(target))


class SpectralFidelityLoss(nn.Module):
    """
    Combined loss = α·L1 + β·SAM + γ·Edge
    Default weights tuned for East African smallholder crop imagery.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.sam   = SpectralAngleMapperLoss()
        self.edge  = EdgePreservingLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1   = F.l1_loss(pred, target)
        sam  = self.sam(pred, target)
        edge = self.edge(pred, target)
        return self.alpha * l1 + self.beta * sam + self.gamma * edge


# ── Wald's Protocol Data Augmentation ────────────────────────────────────────

def walds_protocol_pair(
    hr_patch: torch.Tensor,
    scale: int = 4,
    noise_std: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (LR, HR) training pairs from high-resolution Sentinel-2 data.

    Wald's Protocol:
      1. Start with real 10m Sentinel-2 patch (HR ground truth).
      2. Blur with Gaussian kernel (simulate PSF of 40m sensor).
      3. Downsample 4x → 40m (LR input).
      4. Add slight sensor noise.
    Train model to reconstruct (2) from (3) — at inference, feed real 10m
    and receive synthetic 2.5m output.

    For heterogeneous African smallholder plots (intercropping, fences):
      The model learns spectral + spatial texture differences between:
        - Crops (repetitive linear texture)
        - Fences (linear non-vegetative signature)
        - Bushes (chaotic fractal texture)
      This enables implicit spectral unmixing at upscaling time.

    Args:
        hr_patch:  (C, H, W) normalised Sentinel-2 float32 tensor
        scale:     downsampling factor (default 4)
        noise_std: sensor noise standard deviation
    Returns:
        lr_patch: (C, H//scale, W//scale) — model input
        hr_patch: (C, H, W)               — training target
    """
    _, H, W = hr_patch.shape
    # Gaussian blur before downsampling (anti-aliasing, PSF simulation)
    pad   = 2
    sigma = scale / (2 * 3.14159) ** 0.5
    k     = _gaussian_kernel(5, sigma).to(hr_patch.device)
    C     = hr_patch.shape[0]
    k4d   = k.view(1, 1, 5, 5).expand(C, 1, 5, 5)
    blurred = F.conv2d(hr_patch.unsqueeze(0), k4d, padding=pad, groups=C).squeeze(0)

    # Downsample
    lr = F.interpolate(blurred.unsqueeze(0), scale_factor=1 / scale,
                       mode="bilinear", align_corners=False).squeeze(0)

    # Sensor noise
    lr = lr + torch.randn_like(lr) * noise_std
    return lr.clamp(0, 1), hr_patch


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


# ── Inference Helper ──────────────────────────────────────────────────────────

@torch.inference_mode()
def super_resolve(
    model: EDSR,
    lr_tensor: torch.Tensor,
    device: str = "cpu",
    tile_size: int = 64,
    overlap: int = 8,
) -> torch.Tensor:
    """
    Tiled super-resolution inference for large satellite patches.
    Tiles with overlap to avoid boundary artefacts.

    Args:
        model:     trained EDSR instance
        lr_tensor: (C, H, W) float32 low-res input
        tile_size: LR tile size in pixels
        overlap:   tile overlap in pixels
    Returns:
        (C, H*scale, W*scale) super-resolved tensor
    """
    model.eval().to(device)
    lr = lr_tensor.to(device)
    C, H, W = lr.shape
    scale    = model.scale
    step     = tile_size - overlap
    out_H, out_W = H * scale, W * scale
    output   = torch.zeros(C, out_H, out_W, device=device)
    count    = torch.zeros(C, out_H, out_W, device=device)

    for y in range(0, H, step):
        for x in range(0, W, step):
            y1, x1 = min(y, H - tile_size), min(x, W - tile_size)
            y2, x2 = y1 + tile_size, x1 + tile_size
            tile = lr[:, y1:y2, x1:x2].unsqueeze(0)
            sr   = model(tile).squeeze(0)
            oy1, ox1 = y1 * scale, x1 * scale
            oy2, ox2 = y2 * scale, x2 * scale
            output[:, oy1:oy2, ox1:ox2] += sr
            count[:, oy1:oy2, ox1:ox2]  += 1

    return (output / count.clamp(min=1)).clamp(0, 1)
