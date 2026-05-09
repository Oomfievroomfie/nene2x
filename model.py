"""
model.py  –  2× per-channel image upscaler (tinygrad port)
"""

import math
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict as tg_load_state_dict

gConfig = "b,3x3_12,3x3_12,3x3_4"
LEAKY_SLOPE = 0.00005


# ── helpers ───────────────────────────────────────────────────────────────────

def pixel_shuffle(x: Tensor, factor: int = 2) -> Tensor:
    B, C, H, W = x.shape
    C_out = C // (factor * factor)
    x = x.reshape(B, C_out, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C_out, H * factor, W * factor)


def rot90_tensor(x: Tensor, k: int) -> Tensor:
    """Rotate 90° CCW k times in the last two dims."""
    for _ in range(k % 4):
        x = x.transpose(-2, -1).flip(-2)
    return x


# ── config parsing ────────────────────────────────────────────────────────────

def _parse_config(cfg: str) -> tuple:
    is_bilinear = cfg.startswith("b,")
    if is_bilinear:
        cfg = cfg[2:]
    specs = []
    for token in cfg.split(","):
        token = token.strip()
        flags = set()
        while token[-1] in ("d", "n"):
            flags.add(token[-1])
            token = token[:-1]
        kpart, c_str = token.split("_")
        k = int(kpart.split("x")[0])
        specs.append((k, int(c_str), "d" in flags, "n" in flags))
    assert specs[-1][1] == 4, f"Final layer must output 4 channels, got {specs[-1][1]}"
    return specs, is_bilinear


def _config_to_str(specs: list, is_bilinear: bool = False) -> str:
    def _token(k, c, dw, nb):
        return f"{k}x{k}_{c}{'d' if dw else ''}{'n' if nb else ''}"
    body = ",".join(_token(*s) for s in specs)
    return ("b," + body) if is_bilinear else body


# ── Conv2d wrapper with manual padding ────────────────────────────────────────

class Conv2dManual:
    """Conv2d with manual replicate/circular padding. weight/bias exposed directly."""

    def __init__(self, in_c: int, out_c: int, k: int,
                 groups: int = 1, bias: bool = True, is_wrapping: bool = False):
        self._k = k
        self._pad_size = (k - 1) // 2
        self.padding_enabled = True
        self.is_wrapping = is_wrapping
        self.groups = groups

        fan_out = out_c * k * k // groups
        std = math.sqrt(2.0 / ((1.0 + LEAKY_SLOPE ** 2) * fan_out))
        self.weight = Tensor.randn(out_c, in_c // groups, k, k) * std
        self.bias = Tensor.zeros(out_c) if bias else None
        self._center_weights()

    def _center_weights(self):
        w = self.weight
        out_c = w.shape[0]
        w_flat = w.reshape(out_c, -1)
        mean = w_flat.mean(axis=1, keepdim=True)
        self.weight = (w_flat - mean).reshape(w.shape)

    @property
    def kernel_size(self):
        return (self._k, self._k)

    def __call__(self, x: Tensor) -> Tensor:
        if self.padding_enabled and self._pad_size > 0:
            mode = "circular" if self.is_wrapping else "replicate"
            p = self._pad_size
            x = x.pad(((0, 0), (0, 0), (p, p), (p, p)), mode=mode)
        return x.conv2d(self.weight, self.bias, groups=self.groups)


# ── model ─────────────────────────────────────────────────────────────────────

class UpscaleNet:
    def __init__(self, is_wrapping: bool = False):
        self.padding_enabled = True
        self.is_wrapping = is_wrapping
        specs, is_bilinear = _parse_config(gConfig)
        self.is_bilinear = is_bilinear
        self._build_layers(specs)

    @staticmethod
    def _config_from_state(state: dict) -> tuple:
        is_bilinear = "zfinalb.weight" in state
        final_key = "zfinalb.weight" if is_bilinear else "zfinal.weight"
        final_pfx = "zfinalb"        if is_bilinear else "zfinal"
        assert final_key in state, f"No '{final_key}' in state dict"

        keys = sorted(k for k in state if k.startswith("layers.") and k.endswith(".weight"))
        specs = []
        for i, k in enumerate(keys):
            w = state[k]
            pfx = k[:-len(".weight")]
            is_depthwise = (i > 0) and (w.shape[1] == 1)
            no_bias = (pfx + ".bias") not in state
            specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))

        w = state[final_key]
        is_depthwise = (len(keys) > 0) and (w.shape[1] == 1)
        no_bias = (final_pfx + ".bias") not in state
        specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))
        return specs, is_bilinear

    def _build_layers(self, specs: list):
        in_c = 1
        convs = []
        for k, out_c, is_depthwise, no_bias in specs:
            groups = in_c if is_depthwise else 1
            convs.append(Conv2dManual(in_c, out_c, k, groups=groups,
                                      bias=not no_bias, is_wrapping=self.is_wrapping))
            in_c = out_c
        self.layers = convs[:-1]
        final_attr = "zfinalb" if self.is_bilinear else "zfinal"
        setattr(self, final_attr, convs[-1])
        for stale in ("zfinal", "zfinalb"):
            if stale != final_attr and hasattr(self, stale):
                delattr(self, stale)

    def load_weights(self, state: dict, strict: bool = True):
        specs, is_bilinear = self._config_from_state(state)
        self.is_bilinear = is_bilinear
        self._build_layers(specs)
        tg_load_state_dict(self, state, strict=strict)

    @property
    def _final_layer(self) -> Conv2dManual:
        return self.zfinalb if self.is_bilinear else self.zfinal

    def set_padding_enabled(self, enabled: bool):
        self.padding_enabled = enabled
        for conv in [*self.layers, self._final_layer]:
            if conv._k > 1:
                conv.padding_enabled = enabled

    def __call__(self, x: Tensor) -> Tensor:
        for conv in self.layers:
            x = conv(x).leaky_relu(LEAKY_SLOPE)
        x = self._final_layer(x)
        return pixel_shuffle(x, 2)

    @TinyJit
    def _jit_call(self, x: Tensor) -> Tensor:
        return self.__call__(x)

    def upscale_channel(self, lr: Tensor) -> Tensor:
        x = lr.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x_np = x.numpy()
        if self.is_bilinear:
            base_np = upsample2x_np(x_np, self.is_wrapping)
        else:
            base_np = upscale_edi_2x_np(x_np, self.is_wrapping)
        base = Tensor(base_np.astype(np.float32))

        residual = self._jit_call(x)

        if not self.padding_enabled:
            _, _, rH, rW = residual.shape
            _, _, bH, bW = base.shape
            dH, dW = (bH - rH) // 2, (bW - rW) // 2
            base = base[:, :, dH:dH+rH, dW:dW+rW]

        return (base + residual).squeeze(0).squeeze(0).clamp(0.0, 1.0)


# ── numpy utility functions (used by dataset preprocessing and inference) ─────


def _gaussian_blur_np(t: np.ndarray, size: int = 3, sigma: float = 1.0) -> np.ndarray:
    coords = np.arange(size, dtype=np.float32) - np.float32(size // 2)
    g = np.exp(np.float32(-0.5) * (coords / np.float32(sigma)) ** 2)
    g = (g / g.sum()).astype(np.float32)
    k2d = np.outer(g, g).astype(np.float32)
    C, H, W = t.shape
    pad = size // 2
    out = np.zeros_like(t, dtype=np.float32)
    for c in range(C):
        row_padded = np.pad(t[c], pad, mode='reflect')
        conv = np.zeros((H, W), dtype=np.float32)
        for ky in range(size):
            for kx in range(size):
                conv += row_padded[ky:ky+H, kx:kx+W] * k2d[ky, kx]
        out[c] = conv
    return out


def gaussian_blur(t, size: int = 3, sigma: float = 1.0):
    """Gaussian blur on a (C, H, W) numpy float32 array or Tensor."""
    if isinstance(t, Tensor):
        return Tensor(_gaussian_blur_np(t.numpy(), size, sigma))
    return _gaussian_blur_np(t, size, sigma)


def manual_downscale2x(t):
    """2× area-average downscale. t: (C, H, W) numpy array."""
    if isinstance(t, Tensor):
        arr = t.numpy()
        result = manual_downscale2x(arr)
        return Tensor(result)
    C, H, W = t.shape
    return t.reshape(C, H // 2, 2, W // 2, 2).mean(axis=(2, 4)).astype(np.float32)


def upsample2x_np(t: np.ndarray, is_wrapping: bool = False) -> np.ndarray:
    """
    2x bilinear upsample. t: (B, C, H, W) or (C, H, W) numpy float32.
    Returns same number of dims.
    """
    batched = t.ndim == 4
    if not batched:
        t = t[np.newaxis]
    B, C, H, W = t.shape
    pad_mode = "wrap" if is_wrapping else "edge"
    # Pad by 1
    t_pad = np.pad(t, ((0,0),(0,0),(1,1),(1,1)), mode=pad_mode)
    # 2x bilinear via pixel-center aligned interpolation
    # For 2x: each output pixel at position (i,j) comes from input at (i+0.5)/2 - 0.5
    # Simple: upsample then crop (matching the torch implementation)
    H_pad, W_pad = H + 2, W + 2
    out_H, out_W = H_pad * 2, W_pad * 2
    # Bilinear interpolate t_pad to (B, C, out_H, out_W)
    # Using numpy manual bilinear
    ys = (np.arange(out_H, dtype=np.float32) + 0.5) / 2.0 - 0.5
    xs = (np.arange(out_W, dtype=np.float32) + 0.5) / 2.0 - 0.5
    ys = np.clip(ys, 0, H_pad - 1)
    xs = np.clip(xs, 0, W_pad - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, H_pad - 1)
    x1 = np.minimum(x0 + 1, W_pad - 1)
    ty = (ys - y0).astype(np.float32)
    tx = (xs - x0).astype(np.float32)
    # Gather
    a = t_pad[:, :, y0[:, np.newaxis], x0[np.newaxis, :]]
    b = t_pad[:, :, y0[:, np.newaxis], x1[np.newaxis, :]]
    c = t_pad[:, :, y1[:, np.newaxis], x0[np.newaxis, :]]
    d = t_pad[:, :, y1[:, np.newaxis], x1[np.newaxis, :]]
    ty = ty[:, np.newaxis]
    tx = tx[np.newaxis, :]
    out_pad = (a * (1-ty)*(1-tx) + b * (1-ty)*tx
             + c *    ty *(1-tx) + d *    ty *tx)
    out = out_pad[:, :, 2:-2, 2:-2]
    return (out if batched else out[0]).astype(np.float32)


def upsample2x(t, is_wrapping: bool = False):
    """Wrapper that accepts numpy or Tensor."""
    if isinstance(t, Tensor):
        return Tensor(upsample2x_np(t.numpy().astype(np.float32), is_wrapping))
    return upsample2x_np(t.astype(np.float32), is_wrapping)


def upscale_edi_2x_np(t: np.ndarray, is_wrapping: bool = False) -> np.ndarray:
    """
    EDI 2x upscale. t: (B or not, C, H, W) numpy float32.
    Returns same number of dims.
    """
    original_dim = t.ndim
    if original_dim == 3:
        t = t[np.newaxis]

    g, d, w, h = t.shape
    ow, oh = w * 2, h * 2

    pad_mode = "wrap" if is_wrapping else "edge"
    t_pad = np.pad(t, ((0,0),(0,0),(1,1),(1,1)), mode=pad_mode)

    x_coords = (np.arange(ow, dtype=np.float32) + 0.5) / 2.0
    y_coords = (np.arange(oh, dtype=np.float32) + 0.5) / 2.0
    uvx_x, uvx_y = np.meshgrid(x_coords, y_coords, indexing='ij')

    idx_x = np.floor(uvx_x - 0.5).astype(np.int32) + 1
    idx_y = np.floor(uvx_y - 0.5).astype(np.int32) + 1

    s_a = t_pad[:, :, idx_x,     idx_y    ]
    s_b = t_pad[:, :, idx_x + 1, idx_y    ]
    s_c = t_pad[:, :, idx_x,     idx_y + 1]
    s_d = t_pad[:, :, idx_x + 1, idx_y + 1]

    tx = (uvx_x - (idx_x - 1 + 0.5))[np.newaxis, np.newaxis, :, :]
    ty = (uvx_y - (idx_y - 1 + 0.5))[np.newaxis, np.newaxis, :, :]

    diff_q = np.abs(s_d - s_a)
    diff_r = np.abs(s_c - s_b)
    dd = np.clip((diff_r - diff_q) * np.float32(8.0), np.float32(-1.0), np.float32(1.0)) * np.float32(0.5) + np.float32(0.5)

    mask_bs = (tx + ty < np.float32(1.0))
    iBS_1 = s_c * ty + s_b * tx + s_a * (np.float32(1.0) - (tx + ty))
    iBS_2 = s_c * (np.float32(1.0)-tx) + s_b * (np.float32(1.0)-ty) + s_d * (np.float32(1.0) - ((np.float32(1.0)-tx) + (np.float32(1.0)-ty)))
    iBS = np.where(mask_bs, iBS_1, iBS_2)

    mask_fs = (tx > ty)
    iFS_1 = s_a * (np.float32(1.0)-tx) + s_d * ty       + s_b * (np.float32(1.0) - ((np.float32(1.0)-tx) + ty))
    iFS_2 = s_a * (np.float32(1.0)-ty) + s_d * tx       + s_c * (np.float32(1.0) - (tx + (np.float32(1.0)-ty)))
    iFS = np.where(mask_fs, iFS_1, iFS_2)

    bilinear = (s_a * (np.float32(1.0)-tx)*(np.float32(1.0)-ty)
              + s_b *    tx *(np.float32(1.0)-ty)
              + s_c * (np.float32(1.0)-tx)*   ty
              + s_d *    tx *   ty)

    edi_ver = np.where(np.round(dd) > np.float32(0.5), iFS, iBS)
    mix_factor = np.abs(dd - np.float32(0.5)) * np.float32(2.0)
    res = (np.float32(1.0) - mix_factor) * bilinear + mix_factor * edi_ver

    # High frequency layer
    kc = 3
    kg = 2
    ks = 0.5
    bilinear_pad = np.pad(bilinear, ((0,0),(0,0),(kc,kc),(kc,kc)), mode=pad_mode)

    def _tap(dy, dx):
        H_out, W_out = bilinear.shape[2], bilinear.shape[3]
        y0, x0 = kc + dy, kc + dx
        return bilinear_pad[:, :, y0:y0+H_out, x0:x0+W_out]

    hf = (_tap(0,    0) * np.float32(-ks/3.0 + ks)
        + _tap(0,  -kg) * np.float32(-ks/6.0)
        + _tap(-kg,  0) * np.float32(-ks/6.0)
        + _tap(0,  +kg) * np.float32(-ks/6.0)
        + _tap(+kg,  0) * np.float32(-ks/6.0))
    res += hf

    out = res[0] if original_dim == 3 else res
    return out.astype(np.float32)


def upscale_edi_2x(t, is_wrapping: bool = False):
    """Wrapper that accepts numpy or Tensor."""
    if isinstance(t, Tensor):
        return Tensor(upscale_edi_2x_np(t.numpy().astype(np.float32), is_wrapping))
    return upscale_edi_2x_np(t.astype(np.float32), is_wrapping)
