"""
train.py  –  train the 2× per-channel upscaler (tinygrad port)
"""

import random
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_state_dict, safe_save, safe_load

from model import (UpscaleNet, SigmaBranch, FilterNet, gaussian_blur,
                   manual_downscale2x, upscale_edi_2x, upsample2x, box_blur5x5_np)


# ── loss helpers ──────────────────────────────────────────────────────────────

#from nloss import NLossLoss
#loss_fn = NLossLoss("nloss.safetensors")

def loss_l1(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()

def loss_mse(pred: Tensor, target: Tensor) -> Tensor:
    #print(target.shape) # (64, 1, 128, 128)
    return ((pred - target) ** 2).mean()

# NOTE: not actually lE loss. misnamed!
#def loss_lE(pred: Tensor, target: Tensor, z: Tensor) -> Tensor:
#    residual = pred - target
#    abs_residual = residual.abs()
#    noisy_target = target + abs_residual * z
#    return (pred - noisy_target).abs().mean()

# NOTE: not actually lE loss. misnamed! this is only the non-lE part of lE loss.
# Proper lE loss requires some data from the network itself.
#def loss_lE(mu: Tensor, hr: Tensor, z: Tensor) -> Tensor:
#    sigma_gt = (hr - mu).abs()
#    z = z * (sigma_gt > sigma_gt.mean())
#    loss_main = (mu - hr + sigma_gt*z.detach()).abs().mean()
#    return loss_main

def loss_lE(mu: Tensor, hr: Tensor, sigma: Tensor = None, beta: float = 0.01) -> Tensor:
    """
    ℓE loss from "Revisiting ℓ1 Loss in Super-Resolution" (arXiv:2201.10084).

    Eq. (10): Ez[ |ŷ - (μ + |ŷ - μ| * z)| ]
      = Ez[ |(ŷ - μ) - |ŷ - μ| * z| ]
      Backprop flows through |ŷ - μ| (NOT detached) — that's the key property
      that keeps gradient direction correct (Lemma III.1).

    Eq. (13): β * | |ŷ - μ|.detach() - σ |₁  (auxiliary σ branch loss)
      Paper §IV-C: pixels where |ŷ - μ| < mean are masked to zero before
      this auxiliary loss so σ focuses on hard/high-freq samples.
      
    Paper's choice of beta is 0.01, implicitly PSNR-optimized. You should
      probably use a higher value.
    
    The depixelation loss addition is also not in the paper.
    """
    # Algorithm 1 (paper §V) — transcribed exactly:
    #   sigma_gt = abs(hr - mu)
    #   z = randn(hr.shape)
    #   z = z * (sigma_gt > mean(sigma_gt))   # zero easy samples
    #   loss_main = mean(abs(mu + sigma_gt * z.detach() - hr))
    #   loss_aux  = mean(abs(sigma_pre - sigma_gt.detach()))
    #   return loss_main + beta * loss_aux
    beta = 0.2
    sigma_gt = (hr - mu).abs()
    z = Tensor.randn(*mu.shape)
    z = z * (sigma_gt > sigma_gt.mean()).detach()
    main_loss = (mu + sigma_gt * z.detach() - hr).abs().mean()

    if sigma is not None:
        aux_loss = (sigma - sigma_gt.detach()).abs().mean()
        main_loss = main_loss + beta * aux_loss

    if LE_DEPIX_WEIGHT > 0.0:
        main_loss = main_loss + loss_depix(mu, hr) * LE_DEPIX_WEIGHT

    return main_loss


def ntlfb():
    # =================================================================
    # 2. FILTER BANK DEFINITION
    # =================================================================
    # Define High-Frequency Extractors (Sobel-style)
    filter_bank = [
        # DC offset intentionally left out
        [[ 1.0,  0.0, -1.0], # 1, 0
         [ 1.0,  0.0, -1.0],
         [ 1.0,  0.0, -1.0]],
         
        [[ 0.5, -1.0,  0.5], # 2, 0
         [ 0.5, -1.0,  0.5],
         [ 0.5, -1.0,  0.5]],
         
         
        [[ 1.0,  1.0,  1.0], # 0, 1
         [ 0.0,  0.0,  0.0],
         [-1.0, -1.0, -1.0]],
         
        [[ 1.0,  0.0, -1.0], # 1, 1
         [ 0.0,  0.0,  0.0],
         [-1.0,  0.0,  1.0]],
         
        [[ 0.5, -1.0,  0.5], # 2, 1
         [ 0.0,  0.0,  0.0],
         [-0.5,  1.0, -0.5]],
         
         
        [[ 0.5,  0.5,  0.5], # 0, 2
         [-1.0, -1.0, -1.0],
         [ 0.5,  0.5,  0.5]],
         
        [[ 0.5,  0.0, -0.5], # 1, 2
         [-1.0,  0.0,  1.0],
         [ 0.5,  0.0, -0.5]],
         
        [[ 0.5, -1.0,  0.5], # 2, 2
         [-1.0,  2.0, -1.0],
         [ 0.5, -1.0,  0.5]],
         
        # not DCT kernels, but helpful
        
        # diagonal gradients
        [[ 0.0,  0.0,  2.0],
         [ 0.0,  2.0, -3.0],
         [ 2.0, -3.0,  0.0]],
        [[ 2.0,  0.0,  0.0],
         [-3.0,  2.0,  0.0],
         [ 0.0, -3.0,  2.0]],
         
        [[ 0.0,  1.0,  0.0],
         [ 1.0,  0.0, -1.0],
         [ 0.0, -1.0,  1.0]],
        [[ 0.0,  1.0,  0.0],
         [-1.0,  0.0,  1.0],
         [ 0.0, -1.0,  0.0]],
         
        [[ 1.0,  0.0, -1.0],
         [ 0.0,  1.0,  0.0],
         [-1.0,  0.0,  1.0]],
        [[-1.0,  0.0,  1.0],
         [ 0.0,  1.0,  0.0],
         [ 1.0,  0.0, -1.0]],
        
        # small edges
        [[ 1.0, -1.0,  0.0],
         [ 1.0, -1.0,  0.0],
         [ 0.0,  0.0,  0.0]],
        [[ 1.0,  1.0,  0.0],
         [-1.0, -1.0,  0.0],
         [ 0.0,  0.0,  0.0]],
        #
        ## grid noise 
        #[[ 1.0, -1.0,  1.0],
        # [-1.0,  1.0, -1.0],
        # [ 1.0, -1.0,  1.0]],
        #[[ 1.0,  0.0,  1.0],
        # [ 0.0,  0.0,  0.0],
        # [ 1.0,  0.0,  1.0]],
        #[[ 0.5, -1.0,  0.5],
        # [ 0.0,  0.0,  0.0],
        # [ 0.5, -1.0,  0.5]],
        #[[ 0.5,  0.0,  0.5],
        # [-1.0,  0.0, -1.0],
        # [ 0.5,  0.0,  0.5]],
        #[[ 0.0,  0.0,  0.0],
        # [ 0.0,  1.0,  0.0],
        # [ 0.0,  0.0,  0.0]],
    ]
    
    # =================================================================
    # CENTERIZE & SELF-NORMALIZE KERNELS & KERNEL COUNT
    # =================================================================
    normalized_bank = []
    for f in filter_bank:
        # 1. Centerize (Zero-mean) the kernel to remove DC bias
        mean_val = sum(val for row in f for val in row) / 9.0
        centered_f = [[val - mean_val for val in row] for row in f]
        
        # 2. L1 Normalize AND scale down by the number of filters in the bank
        abs_sum = sum(abs(val) for row in centered_f for val in row)
        norm_factor = abs_sum if abs_sum > 0.0 else 1.0
        normalized_bank.append([[val / norm_factor for val in row] for row in centered_f])
    filter_bank = normalized_bank
    return filter_bank

ntlfb_filter_bank = ntlfb()

def neighborhood_dct_texture_loss(
    pred: Tensor,
    target: Tensor,
    w_l1: float = 1.0,
    w_l2: float = 0.0,
    w_tex: float = 0.7,
    w_grad: float = 0.0,
    w_l1_lowpass: float = 0.1,
    w_depix: float = 0.2,
    mask_kernel: int = 3
) -> Tensor:
    # Require at least leakage amounts of DC reconstruction
    if w_l1_lowpass < 0.1: w_l1_lowpass = 0.1
    
    # 1. Baseline Pixel Loss
    loss_pixel = (pred - target).abs().mean()
    loss_pixel_l2 = ((pred - target) ** 2).mean()
    
    # =================================================================
    # MANUAL CONVOLUTION HELPERS
    # =================================================================
    
    def apply_3x3_filter(t: Tensor, w: list[list[float]]) -> Tensor:
        """Applies an arbitrary 3x3 filter kernel via manual slicing."""
        # Initialize accumulator with zeros matching the output shape
        out = t[:, :, 1:-1, 1:-1] * 0.0
        
        if w[0][0] != 0.0: out = out + t[:, :, :-2, :-2]  * w[0][0]
        if w[0][1] != 0.0: out = out + t[:, :, :-2, 1:-1] * w[0][1]
        if w[0][2] != 0.0: out = out + t[:, :, :-2, 2:]   * w[0][2]
        
        if w[1][0] != 0.0: out = out + t[:, :, 1:-1, :-2]  * w[1][0]
        if w[1][1] != 0.0: out = out + t[:, :, 1:-1, 1:-1] * w[1][1]
        if w[1][2] != 0.0: out = out + t[:, :, 1:-1, 2:]   * w[1][2]
        
        if w[2][0] != 0.0: out = out + t[:, :, 2:, :-2]  * w[2][0]
        if w[2][1] != 0.0: out = out + t[:, :, 2:, 1:-1] * w[2][1]
        if w[2][2] != 0.0: out = out + t[:, :, 2:, 2:]   * w[2][2]
        
        return out

    def apply_triangle_blur(t: Tensor) -> Tensor:
        """Separable [0.25, 0.5, 0.25] 3x3 lowpass blur."""
        out = t[:, :, 1:-1, 1:-1] * 0.0
        
        out = out + t[:, :, :-2, :-2]   * (1.0/16.0)
        out = out + t[:, :, :-2, 1:-1]  * (2.0/16.0)
        out = out + t[:, :, :-2, 2:]    * (1.0/16.0)
        
        out = out + t[:, :, 1:-1, :-2]  * (2.0/16.0)
        out = out + t[:, :, 1:-1, 1:-1] * (4.0/16.0)
        out = out + t[:, :, 1:-1, 2:]   * (2.0/16.0)
        
        out = out + t[:, :, 2:, :-2]    * (1.0/16.0)
        out = out + t[:, :, 2:, 1:-1]   * (2.0/16.0)
        out = out + t[:, :, 2:, 2:]     * (1.0/16.0)
        
        return out

    # =================================================================
    # 3. NEIGHBORHOOD TEXTURE LOSS (Per Filter)
    # =================================================================
    total_neighborhood_loss = 0.0
    
    p = mask_kernel // 2
    pool_pad = ((0, 0), (0, 0), (p, p), (p, p))

    if w_tex > 0.0:
        use_block_dct = True  # False = sliding DCT wavelet, True = block DCT
        pred_pad = pred
        target_pad = target
        for f in ntlfb_filter_bank:
            if use_block_dct:
                B, C, H, W = pred_pad.shape
                H3, W3 = (H // 3) * 3, (W // 3) * 3
                p_trim = pred_pad  [:, :, :H3, :W3]
                t_trim = target_pad[:, :, :H3, :W3]
                p_blocks = p_trim.reshape(B, C, H3//3, 3, W3//3, 3).permute(0,1,3,5,2,4).reshape(B, C*9, H3//3, W3//3)
                t_blocks = t_trim.reshape(B, C, H3//3, 3, W3//3, 3).permute(0,1,3,5,2,4).reshape(B, C*9, H3//3, W3//3)
                flat_w = Tensor([f[r][c] for r in range(3) for c in range(3)] * C).reshape(1, C*9, 1, 1)
                pred_mag   = (p_blocks * flat_w).reshape(B, C, 9, H3//3, W3//3).sum(axis=2).abs()
                target_mag = (t_blocks * flat_w).reshape(B, C, 9, H3//3, W3//3).sum(axis=2).abs()
                # We need all pixels to be influenced by all their neighbors, even those at their block edges, and
                #  even if it causes overshoot (influence by something 5 pixels away).
                # So we need to locally avgpool even though we're doing a block xform.
                pred_mag   = pred_mag  .avg_pool2d(kernel_size=(mask_kernel, mask_kernel), stride=1)
                target_mag = target_mag.avg_pool2d(kernel_size=(mask_kernel, mask_kernel), stride=1)
            else:
                # Extract high-frequency features for this specific filter shape
                pred_feat   = apply_3x3_filter(pred_pad, f)
                target_feat = apply_3x3_filter(target_pad, f)

                pred_mag   = pred_feat  .abs().avg_pool2d(kernel_size=(mask_kernel, mask_kernel), stride=1)
                target_mag = target_feat.abs().avg_pool2d(kernel_size=(mask_kernel, mask_kernel), stride=1)
            
            diffused_SE = (pred_mag - target_mag).abs() * Tensor.rand(pred_mag.shape) * 2.0
            per_sample_rand = Tensor.rand(diffused_SE.shape[0]).reshape(-1, 1, 1, 1) * 2.0
            total_neighborhood_loss = total_neighborhood_loss + (diffused_SE * per_sample_rand).mean()
    
    # =================================================================
    # 4. LOWPASSED L1 LOSS
    # =================================================================
    if w_l1_lowpass > 0.0:
        pred_blur = apply_triangle_blur(pred)
        target_blur = apply_triangle_blur(target)
        loss_lowpass = (pred_blur - target_blur).abs().mean()
    else:
        loss_lowpass = 0.0
    
    # =================================================================
    # 5. PIXELATION DETECTOR LOSS
    # =================================================================
    # Penalize false sharp edges at inter-source-pixel boundaries.
    # In a 2x upscaler, columns 2k and 2k+1 come from the same source pixel,
    # so x=1,3,5,... -> x=2,4,6,... are inter-block boundaries.
    # Pixelation = pred is sharper there than target is.
    if w_depix > 0.0:
        W = min(pred.shape[3], target.shape[3])
        H = min(pred.shape[2], target.shape[2])

        # horizontal inter-block: x=1->x=2, x=3->x=4
        # downsample rows by 2 (2x1 box) first so single-row diagonal edges cancel out
        eh = (H // 2) * 2
        p_h = (pred  [:, :, :eh, :W:1][:, :, 0::2, :] + pred  [:, :, :eh, :W:1][:, :, 1::2, :]) * 0.5
        t_h = (target[:, :, :eh, :W:1][:, :, 0::2, :] + target[:, :, :eh, :W:1][:, :, 1::2, :]) * 0.5
        n = (W - 1) // 2
        pred_inter_h   = (p_h[:, :, :, 1::2][:, :, :, :n] - p_h[:, :, :, 2::2][:, :, :, :n]).abs()
        target_inter_h = (t_h[:, :, :, 1::2][:, :, :, :n] - t_h[:, :, :, 2::2][:, :, :, :n]).abs()
        loss_depix_h = ((pred_inter_h - target_inter_h).relu()).mean()

        # vertical inter-block: y=1->y=2, y=3->y=4
        # downsample cols by 2 (1x2 box) first so single-col diagonal edges cancel out
        ew = (W // 2) * 2
        p_v = (pred  [:, :, :H:1, :ew][:, :, :, 0::2] + pred  [:, :, :H:1, :ew][:, :, :, 1::2]) * 0.5
        t_v = (target[:, :, :H:1, :ew][:, :, :, 0::2] + target[:, :, :H:1, :ew][:, :, :, 1::2]) * 0.5
        m = (H - 1) // 2
        pred_inter_v   = (p_v[:, :, 1::2, :][:, :, :m, :] - p_v[:, :, 2::2, :][:, :, :m, :]).abs()
        target_inter_v = (t_v[:, :, 1::2, :][:, :, :m, :] - t_v[:, :, 2::2, :][:, :, :m, :]).abs()
        loss_depix_v = ((pred_inter_v - target_inter_v).relu()).mean()

        loss_depix = (loss_depix_h + loss_depix_v) * 0.5
    else:
        loss_depix = 0.0


    # 6. Combine and Return
    return (loss_pixel * w_l1) + (loss_pixel_l2 * w_l2) \
            + (total_neighborhood_loss * w_tex) + (loss_lowpass * w_l1_lowpass) \
            + (loss_depix * w_depix)

loss_hd = neighborhood_dct_texture_loss

# Weight for depixelation penalty applied to ℓE loss (0.0 = disabled)
LE_DEPIX_WEIGHT: float = 0.1

def loss_depix(pred: Tensor, target: Tensor) -> Tensor:
    """Penalize false sharp edges at inter-source-pixel boundaries (2× upscaler)."""
    W = min(pred.shape[3], target.shape[3])
    H = min(pred.shape[2], target.shape[2])

    eh = (H // 2) * 2
    p_h = (pred  [:, :, :eh, :W][:, :, 0::2, :] + pred  [:, :, :eh, :W][:, :, 1::2, :]) * 0.5
    t_h = (target[:, :, :eh, :W][:, :, 0::2, :] + target[:, :, :eh, :W][:, :, 1::2, :]) * 0.5
    n = (W - 1) // 2
    loss_h = ((p_h[:, :, :, 1::2][:, :, :, :n] - p_h[:, :, :, 2::2][:, :, :, :n]).abs()
            - (t_h[:, :, :, 1::2][:, :, :, :n] - t_h[:, :, :, 2::2][:, :, :, :n]).abs()).relu().mean()

    ew = (W // 2) * 2
    p_v = (pred  [:, :, :H, :ew][:, :, :, 0::2] + pred  [:, :, :H, :ew][:, :, :, 1::2]) * 0.5
    t_v = (target[:, :, :H, :ew][:, :, :, 0::2] + target[:, :, :H, :ew][:, :, :, 1::2]) * 0.5
    m = (H - 1) // 2
    loss_v = ((p_v[:, :, 1::2, :][:, :, :m, :] - p_v[:, :, 2::2, :][:, :, :m, :]).abs()
            - (t_v[:, :, 1::2, :][:, :, :m, :] - t_v[:, :, 2::2, :][:, :, :m, :]).abs()).relu().mean()

    return (loss_h + loss_v) * 0.5

def loss_adv_filter_upscaler(pred: Tensor, target: Tensor, filter_net: "FilterNet") -> Tensor:
    """Generator-side loss: penalize pred by the adversarial filter score."""
    diff = pred - target
    return filter_net(diff).mean()


def loss_adv_filter_discriminator(pred: Tensor, target: Tensor, filter_net: "FilterNet") -> Tensor:
    """Discriminator-side loss: filter_net should score high on real errors.
    Uses detached pred so gradients don't flow back to the upscaler."""
    diff = pred.detach() - target
    return -filter_net(diff).mean()


# ── sinc downscale ────────────────────────────────────────────────────────────

# Number of lobes for Lanczos sinc downscale used during LR generation.
# 0 = always use box filter; >0 = 50% chance to use sinc instead of box.
SINC_DOWNSCALE_LOBES: int = 3


def _sinc_downscale2x(t: np.ndarray, lobes: int) -> np.ndarray:
    """2× Lanczos-windowed sinc downscale. t: (C, H, W) float32 in [0,1]."""
    C, H, W = t.shape
    out_h, out_w = H // 2, W // 2

    # Phase-matched to box filter: output pixel o centred at input 2*o + 0.5.
    # 4*lobes taps; tap j is at input offset j, so distance from centre = j - (half - 0.5).
    half = lobes * 2  # = 2*lobes
    n_taps = 4 * lobes
    j = np.arange(n_taps, dtype=np.float64)
    d = j - (half - 0.5)
    kern = np.sinc(d * 0.5) * np.sinc(d / (2.0 * lobes))
    kern = (kern / kern.sum()).astype(np.float32)  # (n_taps,)

    # Pad left by (half-1) so output pixel 0's first tap lands at padded index 0.
    # Output pixel o's taps are at padded indices [2*o, 2*o + n_taps).
    tw = np.pad(t, ((0,0),(0,0),(half-1, half)), mode='edge')  # (C, H, W + n_taps - 1)

    # W pass: gather (C, H, out_w, n_taps) then sum → (C, H, out_w)
    col_idx = 2 * np.arange(out_w)[:, None] + np.arange(n_taps)[None, :]  # (out_w, n_taps)
    tmp = tw[:, :, col_idx]          # (C, H, out_w, n_taps)
    tmp = (tmp * kern).sum(axis=3)   # (C, H, out_w)

    # H pass: pad tmp along H, same gather pattern → (C, out_h, out_w)
    tmp_pad = np.pad(tmp, ((0,0),(half-1, half),(0,0)), mode='edge')  # (C, H+n_taps-1, out_w)
    row_idx = 2 * np.arange(out_h)[:, None] + np.arange(n_taps)[None, :]  # (out_h, n_taps)
    tmp2 = tmp_pad[:, row_idx, :]    # (C, out_h, n_taps, out_w)
    out = (tmp2 * kern[None, None, :, None]).sum(axis=2)  # (C, out_h, out_w)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ── augmentation helpers ──────────────────────────────────────────────────────

def _orient(t: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    """Apply one of 8 isometric orientations to a (C, H, W) numpy array."""
    if flip:
        t = t[:, :, ::-1].copy()
    for _ in range(rot % 4):
        # rot90 CCW: transpose then flip rows
        t = np.transpose(t, (0, 2, 1))[:, ::-1, :].copy()
    return t


# ── dataset ───────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}


def _load_as_float(path: Path) -> np.ndarray:
    """Load image as float32 (C, H, W) numpy array in [0, 1]."""
    img = Image.open(path)
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode not in ("L", "RGB", "RGBA", "LA"):
        img = img.convert("RGB")

    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    H, W, C = arr.shape
    H -= H % 2
    W -= W % 2
    arr = arr[:H, :W]
    return arr.transpose(2, 0, 1).copy()  # (C, H, W)


class UpscaleDataset:
    def __init__(
        self,
        folder: str,
        patch_size: int = 48,
        patches_per_pair: int = 200,
        n_aug: int = 8,
        brightness_range: float = 0.15,
        precompute_factor: int = 1,
        pad: int = 0,
        baseline_fn=None,
        is_3ch: bool = False,
    ):
        self.patch_size       = patch_size
        self.patches_per_pair = patches_per_pair
        self.precompute_factor = precompute_factor
        self.pad              = pad
        self.baseline_fn = baseline_fn if baseline_fn is not None else upscale_edi_2x
        self.is_3ch = is_3ch

        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._load(Path(folder), n_aug, brightness_range)

        if not self.pairs:
            raise RuntimeError("No valid images found.")
        if len(self.pairs) < self.precompute_factor:
            raise RuntimeError(f"Not enough pairs ({len(self.pairs)}) for precompute_factor={self.precompute_factor}.")

    def _load(self, folder: Path, n_aug: int, brightness_range: float):
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not paths:
            raise FileNotFoundError(f"No images in {folder}")
        for p in tqdm(paths, desc="Preprocessing images"):
            try:
                hr = _load_as_float(p)
                if self.is_3ch:
                    if hr.shape[0] < 3:
                        continue  # skip grayscale images for 3-channel models
                    hr = hr[:3]
                self._add_image(hr, n_aug, brightness_range)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
        random.shuffle(self.pairs)

    def _add_image(self, hr: np.ndarray, n_aug: int, brightness_range: float):
        if SINC_DOWNSCALE_LOBES > 0 and random.random() < 0.5:
            lr = _sinc_downscale2x(hr, SINC_DOWNSCALE_LOBES)
        else:
            lr = manual_downscale2x(hr)
        if isinstance(lr, Tensor):
            lr = lr.numpy()
        baseline_out = self.baseline_fn(Tensor(lr))
        if isinstance(baseline_out, Tensor):
            baseline_out = baseline_out.numpy()

        C, lH, lW = lr.shape
        ph  = self.patch_size
        pad = self.pad

        total_patches = n_aug * self.precompute_factor * self.patches_per_pair
        for _ in range(total_patches):
            # crop a patch from the full-resolution arrays
            if lH > ph + 2 * pad and lW > ph + 2 * pad:
                ly = random.randint(pad, lH - ph - pad - 1)
                lx = random.randint(pad, lW - ph - pad - 1)
                lr_p  = lr          [:, ly - pad : ly + ph + pad, lx - pad : lx + ph + pad]
                base_p = baseline_out[:, ly * 2   : (ly + ph) * 2, lx * 2   : (lx + ph) * 2]
                hr_p  = hr          [:, ly * 2   : (ly + ph) * 2, lx * 2   : (lx + ph) * 2]
            elif lH > ph and lW > ph:
                ly = random.randint(0, lH - ph - 1)
                lx = random.randint(0, lW - ph - 1)
                lr_p  = lr          [:, ly : ly + ph,       lx : lx + ph      ]
                base_p = baseline_out[:, ly * 2 : (ly+ph)*2, lx * 2 : (lx+ph)*2]
                hr_p  = hr          [:, ly * 2 : (ly+ph)*2, lx * 2 : (lx+ph)*2]
            else:
                lr_p   = lr
                base_p = baseline_out
                hr_p   = hr

            rot    = random.randint(0, 3)
            flip   = random.random() > 0.5
            brightness = random.uniform(0.5, 1.0) if random.random() > 0.5 else 1.0

            a_lr   = _orient(lr_p,   rot, flip)
            a_base = _orient(base_p, rot, flip)
            a_hr   = _orient(hr_p,   rot, flip)

            if brightness != 1.0:
                a_lr   = a_lr   * brightness
                a_base = a_base * brightness
                a_hr   = a_hr   * brightness

            residual = a_hr - a_base

            if not self.is_3ch:
                c = random.randrange(C)
                a_lr      = a_lr[c:c+1]
                a_base    = a_base[c:c+1]
                residual  = residual[c:c+1]
                a_hr      = a_hr[c:c+1]

            self.pairs.append((a_lr.astype(np.float16), residual.astype(np.float16), a_base.astype(np.float16), a_hr.astype(np.float16)))

    def __len__(self) -> int:
        return len(self.pairs)

    def get_patch(self, patch_idx: int):
        lr, residual, baseline, hr = self.pairs[patch_idx % len(self.pairs)]
        return lr, residual, baseline, hr


# ── saving ────────────────────────────────────────────────────────────────────


def trysave(model, out_path, sigma_branch=None, filter_net=None):
    import time
    from safetensors.numpy import save_file as sf_save_numpy
    model._add_config_tensor()

    tensors_np = {k: v.numpy() for k, v in get_state_dict(model).items()}
    if sigma_branch is not None:
        for k, v in get_state_dict(sigma_branch).items():
            tensors_np[f"sigma_branch.{k}"] = v.numpy()
    if filter_net is not None:
        for k, v in get_state_dict(filter_net).items():
            tensors_np[f"filter_net.{k}"] = v.numpy()
    # Because of transient OS-side file locking (e.g. windows defender passive scans)
    #   we need to try saving multiple times.
    for attempt in range(3):
        try:
            sf_save_numpy(tensors_np, str(out_path))
            return
        except Exception as e:
            print(e)
            if attempt < 2:
                time.sleep(0.02)

    model._remove_config_tensor()
    print("failed to save!")


# ── validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_dataset, device_str) -> float:
    Tensor.training = False
    total = 0.0
    n = 0
    for lr_patch, res_patch, _baseline, _hr in val_dataset.pairs:
        lr_in  = Tensor(lr_patch.astype(np.float32)[np.newaxis])
        res_in = Tensor(res_patch.astype(np.float32)[np.newaxis])
        pred = model(lr_in)
        pl = (loss_l1(pred, res_in) + loss_mse(pred, res_in)).numpy().item()
        total += pl
        n += 1

    return total / n if n > 0 else float("inf")


# ── gradient clipping ─────────────────────────────────────────────────────────

def clip_grad_norm_(params, max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_sq = sum(g.pow(2).sum() for g in grads)
    total_norm = total_sq.sqrt().numpy().item()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale


# ── cosine LR ─────────────────────────────────────────────────────────────────

def cosine_lr(epoch: int, total_epochs: int, lr_max: float, lr_min: float) -> float:
    ret = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))
    i = min(float(epoch + 1.0) * 0.35, 1.0)
    return ret * i


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import model as model_module
    model_module.LEAKY_SLOPE = args.leaky_slope

    net = UpscaleNet(False)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        import shutil
        import tempfile
        import os

        fd, temp_path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        shutil.copy(args.resume, temp_path)

        state = safe_load(temp_path)
        net.load_weights(state)

        try:
            os.remove(temp_path)
        except OSError:
            pass

        print(f"Resumed weights from {args.resume}")

    filter_net = None
    if args.adv_filter_loss:
        filter_net = FilterNet()
        if args.resume and Path(args.resume).exists():
            import shutil, tempfile, os
            fd, temp_path = tempfile.mkstemp(suffix=".safetensors")
            os.close(fd)
            shutil.copy(args.resume, temp_path)
            state = safe_load(temp_path)
            fn_state = {k[len("filter_net."):]: v
                        for k, v in state.items() if k.startswith("filter_net.")}
            if fn_state:
                from tinygrad.nn.state import load_state_dict as tg_load_state_dict
                tg_load_state_dict(filter_net, fn_state)
                print("Resumed filter_net weights")
            else:
                print("No filter_net weights in checkpoint — starting fresh")
            try:
                os.remove(temp_path)
            except OSError:
                pass

    sigma_branch = None
    if args.le_loss:
        # Determine pre-final feature channel count from the final layer's input channels.
        # final layer weight shape: (out_c, in_c_per_group, k, k)
        final_w = net._final_layer.weight
        final_in_c = final_w.shape[1] * net._final_layer.groups
        out_c = 3 if net.is_3ch else 1  # channels after pixel-shuffle
        sigma_branch = SigmaBranch(final_in_c, out_c, is_3ch=net.is_3ch)

        if args.resume and Path(args.resume).exists():
            import shutil, tempfile, os
            fd, temp_path = tempfile.mkstemp(suffix=".safetensors")
            os.close(fd)
            shutil.copy(args.resume, temp_path)
            state = safe_load(temp_path)
            sigma_state = {k[len("sigma_branch."):]: v
                           for k, v in state.items() if k.startswith("sigma_branch.")}
            if sigma_state:
                from tinygrad.nn.state import load_state_dict as tg_load_state_dict
                tg_load_state_dict(sigma_branch, sigma_state)
                print("Resumed sigma branch weights")
            else:
                print("No sigma branch weights in checkpoint — starting sigma branch fresh")
            try:
                os.remove(temp_path)
            except OSError:
                pass

    net_params    = sum(p.numpy().size for p in get_state_dict(net).values())
    sigma_params  = sum(p.numpy().size for p in get_state_dict(sigma_branch).values()) if sigma_branch is not None else 0
    filter_params = sum(p.numpy().size for p in get_state_dict(filter_net).values()) if filter_net is not None else 0
    extra = ""
    if sigma_branch is not None:
        extra += f" + {sigma_params:,} (sigma)"
    if filter_net is not None:
        extra += f" + {filter_params:,} (filter_net)"
    if extra:
        print(f"Parameters: {net_params:,} (net){extra} = {net_params+sigma_params+filter_params:,}")
    else:
        print(f"Parameters: {net_params:,}")

    params = list(get_state_dict(net).values())
    if sigma_branch is not None:
        params = params + list(get_state_dict(sigma_branch).values())
    # filter_net has its own optimizer — not added to main params

    pad = sum((conv._k - 1) // 2 * conv._dilation
              for conv in [*net.layers, net._final_layer]
              if conv._k > 1)
    print(f"Receptive field pad: {pad} LR px")
    if pad > 0:
        net.set_padding_enabled(False)

    trysave(net, out_path, sigma_branch, filter_net)

    if net.is_raw:
        def baseline_fn(t):
            arr = t.numpy() if isinstance(t, Tensor) else t
            C, H, W = arr.shape
            return np.zeros((C, H * 2, W * 2), dtype=np.float32)
    elif net.is_blurbilinear:
        def baseline_fn(t):
            arr = t.numpy() if isinstance(t, Tensor) else t
            blurred = box_blur5x5_np(arr[np.newaxis], is_wrapping=False)[0]
            return upsample2x(blurred, is_wrapping=False)
    elif net.is_bilinear:
        baseline_fn = upsample2x
    else:
        baseline_fn = upscale_edi_2x
    dataset = UpscaleDataset(
        args.data,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
        n_aug=args.n_aug,
        brightness_range=args.brightness_range,
        precompute_factor=args.precompute_factor,
        pad=pad,
        baseline_fn=baseline_fn,
        is_3ch=net.is_3ch,
    )

    print(f"Dataset: {len(dataset.pairs)} precomputed pairs")

    val_data_folder = args.val_data if args.val_data else args.data
    val_dataset = UpscaleDataset(
        val_data_folder,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
        n_aug=args.n_aug,
        brightness_range=args.brightness_range,
        precompute_factor=1,
        pad=pad,
        baseline_fn=baseline_fn,
        is_3ch=net.is_3ch,
    )
    print(f"Validation set: {len(val_dataset.pairs)} pairs")

    opt = AdamW(params, lr=args.lr, b1=0.9, b2=0.999, weight_decay=0.0)
    best_loss = float("inf")

    use_basic_loss = not args.fancy_loss
    use_le_loss = args.le_loss
    use_adv_filter = args.adv_filter_loss

    filter_params = list(get_state_dict(filter_net).values()) if filter_net is not None else []
    
    opt_filter = AdamW(filter_params, lr=1e-4, b1=0.5, b2=0.9, weight_decay=1e-4) if filter_net is not None else None

    def _optstep():
        clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        for p in params:
            p.assign(p.clamp(-3.99, 3.99))

    def _filter_optstep():
        opt_filter.step()
        #filter_net.clamp_weights(0.05)

    if use_adv_filter:
        @TinyJit
        def train_step(lr_batch: Tensor, res_batch: Tensor, base_batch: Tensor, hr_batch: Tensor, z: Tensor) -> Tensor:
            pred = net(lr_batch)
            pred_full = pred + base_batch

            # Generator step: softplus(-D(pred)) large when pred scores negative, near-zero when pred fools critic
            # filter_net grads from gen_loss are discarded by opt_filter.zero_grad() before disc step
            dc = hr_batch.mean(axis=(2, 3), keepdim=True)
            opt.zero_grad()
            d_pred = filter_net(pred_full)
            gen_loss = 0.7 * (pred - res_batch).abs().mean() + 0.1 * (-d_pred).softplus().mean() + 0.2 * loss_depix(pred_full, hr_batch)
            gen_loss.backward()
            _optstep()

            # Discriminator step: non-saturating logistic (StyleGAN2), pushing pred negative and hr positive
            opt_filter.zero_grad()
            d_pred_detached = filter_net((pred_full).detach())
            d_hr = filter_net(hr_batch)
            disc_loss = d_pred_detached.softplus().mean() + (-d_hr).softplus().mean()
            disc_loss.backward()
            _filter_optstep()

            return Tensor.stack(gen_loss, (-d_pred).softplus().mean(), (-d_hr).softplus().mean()).realize()
    elif use_le_loss:
        @TinyJit
        def train_step(lr_batch: Tensor, res_batch: Tensor, base_batch: Tensor, hr_batch: Tensor, z: Tensor) -> Tensor:
            opt.zero_grad()
            pred, feats = net.forward_with_features(lr_batch)
            sigma = sigma_branch(feats)
            loss = loss_lE(pred, res_batch, sigma=sigma, beta=args.le_beta)
            loss.backward()
            _optstep()
            return loss.realize()
    elif use_basic_loss:
        @TinyJit
        def train_step(lr_batch: Tensor, res_batch: Tensor, base_batch: Tensor, hr_batch: Tensor, z: Tensor) -> Tensor:
            opt.zero_grad()
            pred = net(lr_batch)
            loss = loss_l1(pred, res_batch)
            loss.backward()
            _optstep()
            return loss.realize()
    else:
        @TinyJit
        def train_step(lr_batch: Tensor, res_batch: Tensor, base_batch: Tensor, hr_batch: Tensor, z: Tensor) -> Tensor:
            opt.zero_grad()
            pred = net(lr_batch)
            loss = loss_hd(pred, res_batch)
            loss.backward()
            _optstep()
            return loss.realize()

    chunk_size = max(1, len(dataset.pairs) // args.precompute_factor)

    for epoch in range(start_epoch, args.epochs + 1):
        new_lr = cosine_lr(epoch - 1, args.epochs, args.lr, args.lr * 0.1)
        opt.lr = Tensor([new_lr])

        Tensor.training = True

        chunk_idx = (epoch - 1) % args.precompute_factor
        start_p = chunk_idx * chunk_size
        end_p   = min(start_p + chunk_size, len(dataset.pairs))

        indices = list(range(start_p, end_p))
        random.shuffle(indices)

        running_loss = 0.0
        running_d_pred = 0.0
        running_d_hr = 0.0
        steps = 0

        pbar = tqdm(range(0, len(indices) - args.batch_size + 1, args.batch_size),
                    desc=f"Epoch {epoch:4d}/{args.epochs}", leave=False)

        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + args.batch_size]

            lr_list, res_list, base_list, hr_list = [], [], [], []
            for pair_i in batch_idx:
                lr_p, res_p, base_p, hr_p = dataset.get_patch(pair_i)
                lr_list.append(lr_p)
                res_list.append(res_p)
                base_list.append(base_p)
                hr_list.append(hr_p)

            lr_batch   = Tensor(np.stack(lr_list,   axis=0).astype(np.float32))
            res_batch  = Tensor(np.stack(res_list,  axis=0).astype(np.float32))
            base_batch = Tensor(np.stack(base_list, axis=0).astype(np.float32))
            hr_batch   = Tensor(np.stack(hr_list,   axis=0).astype(np.float32))
            z_batch    = Tensor(np.random.randn(*res_batch.shape).astype(np.float32))

            out = train_step(lr_batch, res_batch, base_batch, hr_batch, z_batch)
            if use_adv_filter:
                vals = out.numpy()
                running_loss   += vals[0]
                running_d_pred += vals[1]
                running_d_hr   += vals[2]
            else:
                running_loss += out.numpy().item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)

        filter_suffix = ""
        if use_adv_filter:
            avg_d_pred = running_d_pred / max(steps, 1)
            avg_d_hr   = running_d_hr   / max(steps, 1)
            filter_suffix = f" | d_pred {avg_d_pred:.4f}  d_hr {avg_d_hr:.4f}"

        if args.strict_validation:
            val_loss = run_validation(net, val_dataset, "cpu")
            print(f"Epoch {epoch:4d} | train {avg_loss:.6f} | val {val_loss:.6f} | lr {new_lr:.2e}{filter_suffix}")
            if val_loss < best_loss:
                best_loss = val_loss
                trysave(net, out_path, sigma_branch, filter_net)
                print(f"  saved {out_path}  (best val loss {best_loss:.6f})")
        else:
            print(f"Epoch {epoch:4d} | loss {avg_loss:.6f} | lr {new_lr:.2e}{filter_suffix}")
            trysave(net, out_path, sigma_branch, filter_net)

    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    p = argparse.ArgumentParser(description="Train 2× per-channel image upscaler")
    p.add_argument("data",           help="Folder of high-resolution training images")
    p.add_argument("--out",          default="upscaler.safetensors")
    p.add_argument("--resume",       default=None)
    p.add_argument("--epochs",            type=int,   default=700)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--patch-size",        type=int,   default=64)
    p.add_argument("--n-aug",             type=int,   default=4)
    p.add_argument("--precompute-factor", type=int,   default=10)
    p.add_argument("--patches-per-pair",  type=int,   default=20)
    p.add_argument("--brightness-range",  type=float, default=0.15)
    p.add_argument("--lr",                type=float, default=2e-3)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--leaky-slope",       type=float, default=0.003)
    p.add_argument("--fancy-loss",        action="store_true")
    p.add_argument("--le-loss",           action="store_true", help="Use ℓE loss (arXiv:2201.10084)")
    p.add_argument("--le-beta",           type=float, default=0.01, help="Penalty factor β for σ auxiliary loss")
    p.add_argument("--adv-filter-loss",   action="store_true", help="Use adversarial filter network loss")
    p.add_argument("--strict-validation", action="store_true")
    p.add_argument("--val-data",          default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
