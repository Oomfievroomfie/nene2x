"""
model.py  –  2× per-channel image upscaler (shared between train / infer)

Architecture (per channel):
  Conv2d(1→128, 9×9, reflect-pad)  + LeakyReLU(0.1)
  Conv2d(128→64, 1×1)              + LeakyReLU(0.1)
  Conv2d(64→16,  1×1)              + LeakyReLU(0.1)
  Conv2d(16→4,   1×1)
  PixelShuffle(2)   →  (B, 1, 2H, 2W)

The network is run on one channel at a time.
Outputs are the RESIDUAL above bilinear interpolation, not raw pixel values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The structure of the network is configurable.
# Each comma-separated token is "KxK_C": a Conv2d with a K×K kernel producing C output channels.
# The final token must produce exactly 4 channels (consumed by PixelShuffle×2).
# Example: "3x3_12,1x1_48,1x1_24,1x1_4"

#gConfig = "3x3_8,3x3_4" # better than FSR, hits 35_5db psnr / 0_9888 ssim
#gConfig = "3x3_4,3x3_8,1x1_4"
#gConfig = "3x3_8,3x3_4,1x1_4" # a little smaller than cdnnx "veryfast"
#gConfig = "3x3_8,3x3_4,3x3_4" # 520 params, roughly the same as cdnnx "veryfast", but has a hard time training up
#gConfig = "3x3_12,3x3_16,1x1_16,3x3_4"
#gConfig = "3x3_16,1x1_12,3x3_16,1x1_12,3x3_12,1x1_4" # 3600 params. gets really good (>37.5db psnr) really fast.
#gConfig = "3x3_16,1x1_12,3x3_12,1x1_12,3x3_4" # 2264 params. performs really well. 37.7+ psnr.
#gConfig = "3x3_64,1x1_32,3x3_64,1x1_32,3x3_16,1x1_4" # 27,988 params. annoyingly not good enough. 38ish psnr at 150 epochs.
#gConfig = "5x5_48,1x1_32,3x3_48,1x1_32,3x3_32,1x1_16,1x1_8,1x1_4" # 28,204 params. tends to get stuck during the first 40 epochs.
#gConfig = "3x3_48,1x1_32,3x3_48,1x1_32,3x3_32,1x1_16,1x1_8,1x1_4" # ... passes 38.4 psnr, then regresses to ~32 psnr lol.
#gConfig = "3x3_48,1x1_32,3x3_48,1x1_32,3x3_32,1x1_4" # easy 38.4db psnr at 20 epochs. 38.8. etc.
#gConfig = "3x3_48,1x1_32,3x3_48,1x1_32,3x3_48,1x1_32,3x3_48,1x1_32,1x1_4" # 39.4+ psnr
#gConfig = "3x3_64,1x1_32,3x3_64,1x1_32,3x3_64,1x1_32,1x1_4" # ... 39.5+
#gConfig = "3x3_48,1x1_48,3x3_48,1x1_48,3x3_48,1x1_48,1x1_16,1x1_4" # dead early
#gConfig = "3x3_48,1x1_48,3x3_48,1x1_48,3x3_48,1x1_48,1x1_4" # 37db early. 38.5. 38.8. 39.3 at 45. stuck at about 39.5? i think?
#gConfig = "5x5_48,1x1_48,3x3_48,1x1_48,3x3_48,1x1_48,1x1_4" # 38.7 at 45. 39.5 stuck maybe?
#gConfig = "3x3_64,3x3_48,1x1_48,3x3_48,1x1_32,3x3_32,1x1_4" # stuck at 39.5 too
#gConfig = "3x3_12,3x3_24,1x1_24,3x3_48,1x1_48,3x3_64,1x1_64,3x3_32,1x1_4" # 39.5 at 50. 39.7+ at 60. stuck at 39.7
#gConfig = "3x3_16,3x3_32,1x1_32,3x3_48,1x1_48,3x3_64,1x1_64,3x3_32,1x1_32,1x1_4" # also 39.6ish
#gConfig = "3x3_24,3x3_24,1x1_24,3x3_48,1x1_48,3x3_64,1x1_64,3x3_32,3x3_32,1x1_4" # ...
#gConfig = "3x3_24,1x1_24,3x3_48,1x1_48,3x3_96,1x1_96,3x3_96,1x1_96,1x1_4" # ... stuck at 39.8
#gConfig = "5x5_32,1x1_32,5x5_64,1x1_64,5x5_128,1x1_128,3x3_48,1x1_48,1x1_4" # ... trains too slow to test reliably
#gConfig = "3x3_32,1x1_32,5x5_64,1x1_64,3x3_96,1x1_64,3x3_48,1x1_32,3x3_16,1x1_12,1x1_4" # ... trains too slow to test reliably
#gConfig = "3x3_32,3x3_96,1x1_96,3x3_96,1x1_32,3x3_32,1x1_4" # ... stuck at 39.7

#gConfig = "3x3_64,3x3_64,1x1_128,3x3_128,1x1_64,3x3_64,3x3_32,1x1_4" # recommended by gemini fast. hit 40db psnr
#gConfig = "3x3_32,3x3_64,1x1_64,3x3_128,1x1_96,3x3_64,3x3_48,1x1_4" # let's try this... also hit it
#gConfig = "3x3_32,3x3_64,3x3_128,1x1_96,3x3_64,1x1_48,3x3_32,1x1_24,1x1_4" # let's try this...

#gConfig = "3x3_64,3x3_128,1x1_192,3x3_192,1x1_192,3x3_192,1x1_128,3x3_128,3x3_64,3x3_4" # recommended by gemini thinking
#gConfig = "3x3_32,3x3_64,3x3_128,3x3_192,1x1_192,3x3_128,3x3_64,3x3_4" # 

# ... reorganized training data

#gConfig = "3x3_32,3x3_64,1x1_64,3x3_128,1x1_96,3x3_64,3x3_48,1x1_4"
gConfig = "3x3_32,3x3_64,1x1_96,3x3_96,1x1_32,3x3_32,1x1_4" # ...


# Intentionally extremely barely-leaky slope so that inference can treat the layer as ReLU instead of LeakyReLU.
# If this causes checkerboard/scanline artifacts in a given model, run a fine-tuning run with train.py --resume
#  and a low learning rate (e.g. 0.0005) and --leaky-slope 0.00001 Doing so will
#  fine-tune with true ReLU and get rid of the artifacts.
#LEAKY_SLOPE = 0.0002
LEAKY_SLOPE = 0.00005


def _parse_config(cfg: str) -> list:
    """Parse a config string into a list of (kernel_size, out_channels) tuples.

    Example: "3x3.12,1x1.48,1x1.4"  →  [(3, 12), (1, 48), (1, 4)]
    """
    specs = []
    for token in cfg.split(","):
        token = token.strip()
        kpart, c_str = token.split("_")
        k = int(kpart.split("x")[0])   # KxK assumed square
        specs.append((k, int(c_str)))
    assert specs[-1][1] == 4, (
        f"Final layer must output exactly 4 channels for PixelShuffle×2, got {specs[-1][1]}"
    )
    return specs


def _config_to_str(specs: list) -> str:
    """Inverse of _parse_config."""
    return ",".join(f"{k}x{k}.{c}" for k, c in specs)


class UpscaleNet(nn.Module):
    """
    Input : (B, 1, H, W)  – single channel LR, float in [0, 1]
    Output: (B, 1, 2H, 2W) – residual above bilinear-interpolated input
    """

    def __init__(self, is_wrapping=False):
        super().__init__()
        self.is_wrapping = is_wrapping
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        self._build_layers(_parse_config(gConfig))
        self._init_weights()

    @staticmethod
    def _config_from_state(state: dict) -> list:
        """Recover the layer spec list entirely from weight tensor shapes."""
        keys = sorted(
            k for k in state if k.startswith("layers.") and k.endswith(".weight")
        )
        assert "zfinal.weight" in state, "No 'zfinal.weight' found in state dict"
        specs = [(state[k].shape[2], state[k].shape[0]) for k in keys]
        w = state["zfinal.weight"]
        specs.append((w.shape[2], w.shape[0]))
        return specs

    def _build_layers(self, specs: list):
        """(Re-)construct all learnable layers from a list of (kernel_size, out_channels)."""
        in_c = 1
        convs = []
        for k, out_c in specs:
            pad = (k - 1) // 2
            # Only use a non-zero padding_mode when there is actual padding.
            # When padding=0 (e.g. 1x1 layers), leave padding_mode at the default
            # 'zeros' so PyTorch takes the fast cudnn path instead of F.pad + conv.
            pm = ("circular" if self.is_wrapping else "replicate") if pad > 0 else "zeros"
            convs.append(nn.Conv2d(in_c, out_c, kernel_size=k, padding=pad, padding_mode=pm))
            in_c = out_c
        # Keep the final layer separate so forward() never needs a [:-1] slice.
        self.layers = nn.ModuleList(convs[:-1])
        self.zfinal  = convs[-1]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Infer architecture from *state_dict* shapes, rebuild layers, then load."""
        self._build_layers(self._config_from_state(state_dict))
        self._init_weights()
        return super().load_state_dict(state_dict, strict=strict)

    def _init_weights(self):
        def center_weights_(m):
            w = m.weight.data
            w_flat = w.view(w.shape[0], -1)
            w_flat.sub_(w_flat.mean(dim=1, keepdim=True))
            m.weight.data = w_flat.view_as(w)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_uniform_(
                nn.init.kaiming_normal_(
                    #m.weight, a=LEAKY_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
                    m.weight, a=LEAKY_SLOPE, mode="fan_out", nonlinearity="leaky_relu"
                )
                center_weights_(m)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_padding_enabled(self, enabled: bool):
        """Toggle replicate/circular padding on spatial layers.
        Disable during training when patches are pre-padded with real data."""
        for conv in [*self.layers, self.zfinal]:
            if conv.kernel_size[0] > 1:
                conv.padding_mode = ("circular" if self.is_wrapping else "replicate") if enabled else "zeros"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  →  (B, 1, 2H, 2W) residual"""
        for conv in self.layers:
            x = self.act(conv(x))
        x = self.zfinal(x)               # no activation on final layer
        return F.pixel_shuffle(x, 2)    # (B, 4, H, W) → (B, 1, 2H, 2W)

    @torch.no_grad()
    def upscale_channel(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Upscale one image channel.
        lr : (H, W) float32 tensor in [0, 1]
        returns : (2H, 2W) float32 tensor clamped to [0, 1]
        """
        x        = lr.unsqueeze(0).unsqueeze(0)          # (1, 1, H, W)
        base     = upscale_edi_2x(x, self.is_wrapping)  # (1, 1, 2H, 2W)
        residual = self(x)
        return (base + residual).squeeze(0).squeeze(0).clamp(0.0, 1.0)


# ── shared image utilities (used by both train.py and infer.py) ───────────────

def make_gaussian_kernel(size: int, sigma: float,
                         channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    k2d = g.outer(g)
    return k2d.view(1, 1, size, size).expand(channels, 1, size, size).contiguous()


def gaussian_blur(t: torch.Tensor, size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur on a (C, H, W) CPU float tensor."""
    C = t.shape[0]
    kernel = make_gaussian_kernel(size, sigma, C, t.device)
    return F.conv2d(t.unsqueeze(0), kernel,
                    padding=size // 2, groups=C).squeeze(0)


def manual_downscale2x(t: torch.Tensor) -> torch.Tensor:
    """
    2× area-average downscale.  No PIL, no extra blur – just averages each 2×2 block.
    t: (C, H, W) where H and W are even.
    """
    C, H, W = t.shape
    return t.reshape(C, H // 2, 2, W // 2, 2).mean(dim=(2, 4))

import numpy as np

def upsample2x(t: torch.Tensor, is_wrapping: bool = False) -> torch.Tensor:
    batched = t.dim() == 4
    if not batched:
        t = t.unsqueeze(0)
    if is_wrapping:
        t_pad = F.pad(t, (1, 1, 1, 1), mode="circular")
    else:
        t_pad = F.pad(t, (1, 1, 1, 1), mode="replicate")

    is_directml = False
    try:
        import torch_directml
        is_directml = True
    except:
        pass
    
    if is_directml:
        # We can't use bilinear mode because the correct corner-centering mode doesn't work
        #  properly on directML devices. Instead we do nearest neighbor than blur. For 2x this
        #  is identical to properly centered bilinear.
        # A shader implementation should just use normal bilinear filtering.
        out_pad = F.interpolate(t_pad, scale_factor=2, mode="nearest")
        
        # We use a 5x5 kernel to implicitly crop off the unused 1->2 pixels of padding on each side.
        kernel = torch.zeros((t.shape[1], 1, 5, 5), device=t.device, dtype=t.dtype)
        
        kernel[:, :, 1, 1] = 1.0/16.0
        kernel[:, :, 2, 1] = 1.0/8.0
        kernel[:, :, 3, 1] = 1.0/16.0
        
        kernel[:, :, 1, 2] = 1.0/8.0
        kernel[:, :, 2, 2] = 1.0/4.0
        kernel[:, :, 3, 2] = 1.0/8.0
        
        kernel[:, :, 1, 3] = 1.0/16.0
        kernel[:, :, 2, 3] = 1.0/8.0
        kernel[:, :, 3, 3] = 1.0/16.0
        
        out = F.conv2d(out_pad, kernel, groups=t.shape[1])
    else:
        # Upstream pytorch. Bilinear with align_corners=True is safe.
        out_pad = F.interpolate(t_pad, scale_factor=2, mode="bilinear", align_corners=False)
        out = out_pad[:,:,2:-2,2:-2]
        
    return out if batched else out.squeeze(0)


def upscale_edi_2x(t: torch.Tensor, is_wrapping: bool = False) -> torch.Tensor:
    """
    Upscales a tensor (C, W, H) or (G, D, W, H) by 2x spatially.
    Follows OpenGL pixel-center logic and supports clamp vs wrap.
    """
    
    device = t.device
    original_dim = t.dim()

    if original_dim == 3:
        t = t.unsqueeze(0)

    g, d, w, h = t.shape
    ow, oh = w * 2, h * 2

    # Pad upfront so all 2x2 neighborhood accesses stay in-bounds.
    # idx_x/idx_y reach -1 at the low end and w/h at the high end,
    # so a 1-pixel border is exactly enough.
    pad_mode = "circular" if is_wrapping else "replicate"
    t_pad = F.pad(t, (1, 1, 1, 1), mode=pad_mode)  # (g, d, w+2, h+2)

    # 1. Coordinate Generation (OpenGL Pixel Center Logic)
    x_coords = (torch.arange(ow, device=device) + 0.5) / 2.0
    y_coords = (torch.arange(oh, device=device) + 0.5) / 2.0
    uvx_x, uvx_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # 2. Quad Sampling Indices
    # +1 offset accounts for the 1-pixel pad, so raw idx -1 maps to index 0.
    idx_x = torch.floor(uvx_x - 0.5).long() + 1
    idx_y = torch.floor(uvx_y - 0.5).long() + 1

    # Sample 2x2 neighbourhood — no clamping or wrapping needed here.
    s_a = t_pad[:, :, idx_x,     idx_y    ]  # Top-Left
    s_b = t_pad[:, :, idx_x + 1, idx_y    ]  # Top-Right
    s_c = t_pad[:, :, idx_x,     idx_y + 1]  # Bottom-Left
    s_d = t_pad[:, :, idx_x + 1, idx_y + 1]  # Bottom-Right

    # 3. Local Fractional Coordinates
    tx = (uvx_x - (idx_x - 1 + 0.5)).view(1, 1, ow, oh)  # undo the +1 offset
    ty = (uvx_y - (idx_y - 1 + 0.5)).view(1, 1, ow, oh)

    # 4. Edge Direction Detection
    diff_q = torch.abs(s_d - s_a)
    diff_r = torch.abs(s_c - s_b)
    # dd: 0.0 = edge along A-D diagonal, 1.0 = edge along B-C diagonal
    dd = torch.clamp((diff_r - diff_q) * 8.0, -1.0, 1.0) * 0.5 + 0.5

    # 5. Interpolation iBS (Barycentric Subset)
    mask_bs = (tx + ty < 1.0)
    iBS_1 = s_c * ty + s_b * tx + s_a * (1.0 - (tx + ty))
    iBS_2 = s_c * (1.0-tx) + s_b * (1.0-ty) + s_d * (1.0 - ((1.0-tx) + (1.0-ty)))
    iBS = torch.where(mask_bs, iBS_1, iBS_2)

    # 6. Interpolation iFS (Flipped Subset)
    mask_fs = (tx > ty)
    iFS_1 = s_a * (1.0-tx) + s_d * ty       + s_b * (1.0 - ((1.0-tx) + ty))
    iFS_2 = s_a * (1.0-ty) + s_d * tx       + s_c * (1.0 - (tx + (1.0-ty)))
    iFS = torch.where(mask_fs, iFS_1, iFS_2)

    # 7. Final Mixing
    bilinear = (s_a * (1-tx)*(1-ty)
              + s_b *    tx *(1-ty)
              + s_c * (1-tx)*   ty
              + s_d *    tx *   ty)

    edi_ver = torch.where(torch.round(dd) > 0.5, iFS, iBS)
    mix_factor = torch.abs(dd - 0.5) * 2.0
    res = (1.0 - mix_factor) * bilinear + mix_factor * edi_ver
    
    # 8. high frequency layer
    kc = 3          # kernel half-size
    kg = 2          # tap offset
    ks = 0.5        # strength
    
    bilinear = F.pad(bilinear, (kc, kc, kc, kc), mode=pad_mode)
    kernel = torch.zeros((d, 1, kc*2+1, kc*2+1), device=device, dtype=t.dtype)
    kernel[:, :, kc,    kc-kg] = -ks / 6.0
    kernel[:, :, kc-kg, kc   ] = -ks / 6.0
    kernel[:, :, kc,    kc   ] = -ks / 3.0 + ks
    kernel[:, :, kc+kg, kc   ] = -ks / 6.0
    kernel[:, :, kc,    kc+kg] = -ks / 6.0
    bilinear = F.conv2d(bilinear, kernel, groups=d)
    
    res += bilinear

    return res.squeeze(0) if original_dim == 3 else res
