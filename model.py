"""
model.py  –  2× per-channel image upscaler (tinygrad port)
"""

import math
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import get_state_dict, load_state_dict as tg_load_state_dict

"""
The network is run on one channel at a time.
Outputs are the RESIDUAL above EDI or bilinear interpolation, not raw pixel values.

Config string syntax
────────────────────
Each comma-separated token is "KxK_C[d][n][q]":
  KxK   – square kernel size
  C     – number of output channels
  d     – (optional suffix) grouped convolution: if out_c >= in_c, depthwise
           (groups=in_channels); if out_c < in_c, reverse-depthwise
           (groups=out_channels, each output gets its own input subset).
           e.g. "3x3_8d" (expand) or "3x3_20d" after 40 channels (reduce)
  n     – (optional suffix) no bias parameters
  q     – (optional suffix) dilated convolution (dilation = 2)

Global prefix "b," (before the first token) marks a *bilinear-base* model.
  • The base upscale uses plain bilinear filtering (upsample2x) instead of EDI.
  • Detected automatically on load via "zzzgConfig_" tensor.
  Example: "b,3x3_12,1x1_12,3x3_24,1x1_4"

Global prefix "g," marks a *blurred-bilinear-base* model.
  • 5x5 box blur (respects wrapping) applied to LR before bilinear upscale.
  Example: "g,3x3_12,1x1_12,3x3_24,1x1_4"

Global prefix "x," marks a *raw* model..
  • Network predicts raw colors, not residuals from a base upscale.
  • Unstable and very hard to train.
  • Might be useful if your loss function is GAN-like.
  Example: "x,3x3_12,1x1_4"

Global prefix "r," marks a *3-channel RGB* model.
  • The network takes all 3 RGB channels simultaneously instead of one at a time.
  • Can be combined with other prefixes, e.g. "r,g,3x3_16,3x3_32,1x1_12"
  Example: "r,g,3x3_16,3x3_32,3x3_64,1x1_12"
"""

#gConfig = "3x3_8,1x1_4,3x3_10,3x3_6,1x1_4n"
#gConfig = "g,3x3_32,3x3_48,3x3_48,1x1_4"
#gConfig = "g,3x3_24,3x3_48,1x1_64,3x3_320d,1x1_32,3x3_24,1x1_4"
#gConfig = "g,3x3_8,3x3_32d,1x1_12,3x3_12,3x3_4"
#gConfig = "g,3x3_12,3x3_32,1x1_24,3x3_96d,1x1_12,3x3_12,1x1_4"
#gConfig = "g,3x3_12,3x3_24,3x3_32,1x1_24,3x3_24,1x1_16,1x1_4"
#gConfig = "g,3x3_12,3x3_24,3x3_48,1x1_64,3x3_320d,1x1_32,3x3_24,1x1_4"
#gConfig = "3x3_32,3x3_96,3x3_192,1x1_128,3x3_96,3x3_64,1x1_4"
#gConfig = "3x3_32,3x3_64,1x1_64,3x3_128,1x1_96,3x3_64,3x3_48,1x1_4"
#gConfig = "b,3x3_4,3x3_11,1x1_11d,1x1_4n"
#gConfig = "b,3x3_4,3x3_9,1x1_9,1x1_4"
#gConfig = "3x3_4,3x3_9,1x1_9,1x1_4"
#gConfig = "b,3x3_12,1x1_8,1x1_4"
#gConfig = "r,b,3x3_12,3x3_24,3x3_48,1x1_24,1x1_12"
#gConfig = "g,3x3_12,3x3_16q,3x3_16,3x3_16,1x1_4"
#gConfig = "g,3x3_12,3x3_12q,3x3_32,1x1_24,3x3_96d,1x1_12,3x3_12,1x1_4"
#gConfig = "g,3x3_12,3x3_12q,3x3_24,3x3_32,1x1_24,3x3_24,1x1_16,1x1_4"
#gConfig = "g,3x3_12,3x3_12q,3x3_24,3x3_32,1x1_24,3x3_24,1x1_16,1x1_4"
#gConfig = "g,3x3_24,3x3_32q,3x3_48,1x1_64,3x3_320d,1x1_24,3x3_24,1x1_4"
#gConfig = "g,3x3_24,3x3_48,3x3_72,3x3_72,1x1_4"

#gConfig = "g,3x3_12,3x3_24qd,3x3_24,3x3_24qd,3x3_32,3x3_48,1x1_4"

#gConfig = "g,3x3_12,3x3_24,3x3_24,3x3_32,3x3_72,1x1_4"
#gConfig = "g,3x3_12,3x3_24,3x3_24,3x3_32,3x3_72,1x1_24,1x1_4"
#gConfig = "g,3x3_24,3x3_24qd,3x3_32,3x3_32,3x3_64,3x3_96,1x1_24,1x1_4"
#gConfig = "g,3x3_24,3x3_24,3x3_32,3x3_32,3x3_64,3x3_96,1x1_24,1x1_4"
#gConfig = "g,3x3_16,3x3_24,3x3_32q,3x3_32,3x3_32,1x1_4"
#gConfig = "r,3x3_20,3x3_20,3x3_32,5x5_12"
#gConfig = "g,3x3_32,3x3_32q,3x3_48,3x3_64,1x1_64,3x3_64,3x3_128,1x1_96,3x3_48,1x1_4"
#gConfig = "r,g,3x3_32,3x3_32n,3x3_32n,3x3_32n,3x3_32n,3x3_32n,1x1_12n"

#gConfig = "g,3x3_24,3x3_32q,3x3_32,1x1_48,3x3_72,1x1_64,3x3_48,3x3_32,1x1_4"
#gConfig = "3x3_12,3x3_12q,3x3_32,1x1_24,3x3_96d,1x1_12,3x3_12,1x1_4"
#gConfig = "g,3x3_12,3x3_20q,3x3_20,3x3_20d,3x3_32,1x1_4"
#gConfig = "3x3_12,3x3_20q,3x3_20,3x3_20d,3x3_32,1x1_4"
#gConfig = "g,3x3_12,3x3_20q,3x3_24,3x3_28,3x3_56,1x1_4"
#gConfig = "g,3x3_16,3x3_24q,3x3_96d,1x1_24,3x3_28,3x3_52,3x3_4"
#gConfig = "g,3x3_16,3x3_24q,3x3_96d,1x1_24,3x3_32,3x3_56,1x1_4"
#gConfig = "g,3x3_16,3x3_48q,3x3_96d,1x1_38,3x3_38,3x3_56,3x3_96,1x1_4"
#gConfig = "g,3x3_32,3x3_48q,3x3_96d,3x3_48d,3x3_56,3x3_96,1x1_4"
#gConfig = "g,3x3_32,3x3_32q,3x3_40,3x3_48,3x3_480d,3x3_96d,3x3_40,1x1_4"
#gConfig = "g,3x3_24,3x3_24,3x3_48,3x3_64,3x3_96,3x3_16d,1x1_4"
gConfig = "g,3x3_12,3x3_20q,3x3_20,3x3_20d,3x3_320d,3x3_64d,1x1_4"
#gConfig = "g,3x3_24,3x3_24,3x3_48,3x3_64,3x3_96,1x1_4"
#gConfig = "g,3x3_32,3x3_32q,3x3_48,3x3_96,3x3_40,1x1_4"

# 9 25 49 81 121 169 225

LEAKY_SLOPE = 0.00004


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
    is_3ch = cfg.startswith("r,")
    if is_3ch:
        cfg = cfg[2:]
    is_raw = cfg.startswith("x,")
    if is_raw:
        cfg = cfg[2:]
    is_bilinear = cfg.startswith("b,")
    if is_bilinear:
        cfg = cfg[2:]
        is_raw = False
    is_blurbilinear = cfg.startswith("g,")
    if is_blurbilinear:
        cfg = cfg[2:]
        is_raw = False
        is_bilinear = False
    specs = []
    for token in cfg.split(","):
        token = token.strip()
        flags = set()
        while token[-1] in ("d", "n", "q"):
            flags.add(token[-1])
            token = token[:-1]
        kpart, c_str = token.split("_")
        k = int(kpart.split("x")[0])
        specs.append((k, int(c_str), "d" in flags, "n" in flags, "q" in flags))
    expected_out = 12 if is_3ch else 4
    assert specs[-1][1] == expected_out, f"Final layer must output {expected_out} channels, got {specs[-1][1]}"
    return specs, is_bilinear, is_raw, is_blurbilinear, is_3ch


def _config_to_str(specs: list, is_bilinear: bool = False, is_blurbilinear: bool = False, is_raw: bool = False, is_3ch: bool = False) -> str:
    def _token(k, c, dw, nb, dil=False):
        return f"{k}x{k}_{c}{'d' if dw else ''}{'n' if nb else ''}{'q' if dil else ''}"
    body = ",".join(_token(*s) for s in specs)
    if is_blurbilinear:
        body = "g," + body
    elif is_bilinear:
        body = "b," + body
    elif is_raw:
        body = "x," + body
    if is_3ch:
        body = "r," + body
    return body


# ── Conv2d wrapper with manual padding ────────────────────────────────────────

class Conv2dManual:
    """Conv2d with manual replicate/circular padding. weight/bias exposed directly."""

    def __init__(self, in_c: int, out_c: int, k: int,
                 groups: int = 1, bias: bool = True, is_wrapping: bool = False, dilation: int = 1, stride: int = 1):
        self._k = k
        self._stride = stride
        self._dilation = dilation
        self._pad_size = (k - 1) // 2 * dilation
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
        return x.conv2d(self.weight, self.bias, groups=self.groups, dilation=self._dilation, stride=self._stride)


# ── model ─────────────────────────────────────────────────────────────────────

class UpscaleNet:
    def __init__(self, is_wrapping: bool = False):
        self.padding_enabled = True
        self.is_wrapping = is_wrapping
        specs, is_bilinear, is_raw, is_blurbilinear, is_3ch = _parse_config(gConfig)
        self.is_bilinear = is_bilinear
        self.is_raw = is_raw
        self.is_blurbilinear = is_blurbilinear
        self.is_3ch = is_3ch
        self._build_layers(specs)

    @staticmethod
    def _config_from_state(state: dict) -> tuple:
        is_raw = "zfinalx.weight" in state
        is_bilinear = "zfinalb.weight" in state
        is_blurbilinear = "zfinalg.weight" in state
        final_key = "zfinal.weight"
        final_pfx = "zfinal"
        if is_bilinear:
            final_key = "zfinalb.weight"
            final_pfx = "zfinalb"
        if is_blurbilinear:
            final_key = "zfinalg.weight"
            final_pfx = "zfinalg"
            is_bilinear = False
        if is_raw:
            final_key = "zfinalx.weight"
            final_pfx = "zfinalx"
            is_bilinear = False
            is_blurbilinear = False
        assert final_key in state, f"No '{final_key}' in state dict"

        keys = sorted(k for k in state if k.startswith("layers.") and k.endswith(".weight"))
        specs = []
        first_in_c = None
        for i, k in enumerate(keys):
            w = state[k]
            pfx = k[:-len(".weight")]
            if i == 0:
                first_in_c = w.shape[1]
                is_depthwise = False
                prev_out_c = w.shape[0]
            else:
                groups = prev_out_c // w.shape[1]
                is_depthwise = (groups > 1)
                prev_out_c = w.shape[0]
            no_bias = (pfx + ".bias") not in state
            specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))

        w = state[final_key]
        if first_in_c is None:
            first_in_c = w.shape[1]
            prev_out_c = 0
        groups = prev_out_c // w.shape[1] if prev_out_c else 1
        is_depthwise = (len(keys) > 0) and (groups > 1)
        no_bias = (final_pfx + ".bias") not in state
        specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))
        is_3ch = (first_in_c == 3)
        return specs, is_bilinear, is_raw, is_blurbilinear, is_3ch

    def _build_layers(self, specs: list):
        in_c = 3 if self.is_3ch else 1
        convs = []
        for k, out_c, is_depthwise, no_bias, *rest in specs:
            is_dilated = rest[0] if rest else False
            if is_depthwise:
                groups = in_c if out_c >= in_c else out_c
            else:
                groups = 1
            dilation = 2 if is_dilated else 1
            convs.append(Conv2dManual(in_c, out_c, k, groups=groups,
                                      bias=not no_bias, is_wrapping=self.is_wrapping,
                                      dilation=dilation))
            in_c = out_c
        self.layers = convs[:-1]
        final_attr = "zfinal"
        if self.is_bilinear:
            final_attr = "zfinalb"
        if self.is_blurbilinear:
            final_attr = "zfinalg"
        if self.is_raw:
            final_attr = "zfinalx"
        setattr(self, final_attr, convs[-1])
        for stale in ("zfinal", "zfinalb", "zfinalg", "zfinalx"):
            if stale != final_attr and hasattr(self, stale):
                delattr(self, stale)
        
        cfg_str = _config_to_str(specs, self.is_bilinear, self.is_blurbilinear, self.is_raw, self.is_3ch)
        for stale in [k for k in self.__dict__ if k.startswith("zzzgConfig_")]:
            delattr(self, stale)
        self.zzzgConfig = cfg_str

    def _add_config_tensor(self):
        if hasattr(self, 'zzzgConfig') and isinstance(self.zzzgConfig, str):
            attr_name = f"zzzgConfig_{self.zzzgConfig}"
            if not hasattr(self, attr_name):
                setattr(self, attr_name, Tensor([0.0]))

    def _remove_config_tensor(self):
        for k in list(self.__dict__):
            if k.startswith("zzzgConfig_"):
                delattr(self, k)

    def load_weights(self, state: dict, strict: bool = True):
        cfg_key = next((k for k in state if k.startswith("zzzgConfig_")), None)
        if cfg_key is not None:
            cfg_str = cfg_key[len("zzzgConfig_"):]
            specs, is_bilinear, is_raw, is_blurbilinear, is_3ch = _parse_config(cfg_str)
        else:
            specs, is_bilinear, is_raw, is_blurbilinear, is_3ch = self._config_from_state(state)
        self.is_bilinear = is_bilinear
        self.is_raw = is_raw
        self.is_blurbilinear = is_blurbilinear
        self.is_3ch = is_3ch
        self._build_layers(specs)
        tg_load_state_dict(self, state, strict=strict)

    @property
    def _final_layer(self) -> Conv2dManual:
        if self.is_bilinear:
            return self.zfinalb
        if self.is_blurbilinear:
            return self.zfinalg
        if self.is_raw:
            return self.zfinalx
        return self.zfinal

    def set_padding_enabled(self, enabled: bool):
        self.padding_enabled = enabled
        for conv in [*self.layers, self._final_layer]:
            if conv._k > 1:
                conv.padding_enabled = enabled

    def __call__(self, x: Tensor, return_features: bool = False) -> Tensor:
        for conv in self.layers:
            x = conv(x).leaky_relu(LEAKY_SLOPE)
        if return_features:
            return x
        x = self._final_layer(x)
        return pixel_shuffle(x, 2)

    def forward_with_features(self, x: Tensor) -> tuple:
        """Returns (pred, pre_final_features) for use with sigma branch."""
        feats = self.__call__(x, return_features=True)
        pred = pixel_shuffle(self._final_layer(feats), 2)
        return pred, feats

    @TinyJit
    def _jit_call(self, x: Tensor) -> Tensor:
        return self.__call__(x)

    @TinyJit
    def _jit_call_with_features(self, x: Tensor) -> tuple:
        return self.forward_with_features(x)

    def upscale_channel(self, lr: Tensor) -> Tensor:
        """lr: (H,W) for 1-channel models, or (3,H,W) for 3-channel models."""
        if self.is_3ch:
            x = lr.unsqueeze(0)  # (1,3,H,W)
        else:
            x = lr.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        x_np = x.numpy()
        if self.is_blurbilinear:
            blurred = box_blur5x5_np(x_np, self.is_wrapping)
            base_np = upsample2x_np(blurred, self.is_wrapping)
        elif self.is_bilinear or self.is_raw:
            base_np = upsample2x_np(x_np, self.is_wrapping)
        else:
            base_np = upscale_edi_2x_np(x_np, self.is_wrapping)
        base = Tensor(base_np.astype(np.float32))

        if self.is_raw:
            base *= 0.0

        residual = self._jit_call(x)

        if not self.padding_enabled:
            _, _, rH, rW = residual.shape
            _, _, bH, bW = base.shape
            dH, dW = (bH - rH) // 2, (bW - rW) // 2
            base = base[:, :, dH:dH+rH, dW:dW+rW]

        if self.is_3ch:
            # returns (3, 2H, 2W)
            return (base + residual).squeeze(0).clamp(0.0, 1.0)
        return (base + residual).squeeze(0).squeeze(0).clamp(0.0, 1.0)


# ── Adversarial filter network ────────────────────────────────────────────────

class FilterNet:
    """
    Learnable version of the fixed filter-bank texture loss.
    Like the fixed filter bank: applies learned 3x3 filters to both pred and target,
      takes abs of each response to get texture magnitudes, then compares them.

    __call__(pred, target) -> scalar loss (pred_mag - target_mag).abs().mean()
    """

    def _center_and_limit_weights(self, f):
        f.weight = f.weight.clamp(-1.0, 1.0)
        #f.weight = f.weight - f.weight.mean(axis=[1,2,3], keepdim=True)
        #f.weight = f.weight / f.weight.square().sum(axis=[1,2,3], keepdim=True).sqrt().clamp(1e-8, float('inf'))
        pass

    def __init__(self):
        self.filters  = Conv2dManual(1, 8, 3, bias=False, stride=3)
        self.filters2 = Conv2dManual(8, 16, 3, bias=False, stride=2)
        self._center_and_limit_weights(self.filters )
        self._center_and_limit_weights(self.filters2)

    def normalize_filters(self):
        self._center_and_limit_weights(self.filters )
        self._center_and_limit_weights(self.filters2)
        pass

    def features(self, x: Tensor) -> Tensor:
        x = self.filters(x).abs()
        x = self.filters2(x).abs()
        return x

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_f   = self.features(pred)  .avg_pool2d(kernel_size=(3, 3), stride=2)
        target_f = self.features(target).avg_pool2d(kernel_size=(3, 3), stride=2)
        diff = (pred_f - target_f).abs()
        diff = diff * Tensor.rand(diff.shape) * 2.0
        per_sample = Tensor.rand(diff.shape[0]).reshape(-1, 1, 1, 1) * 2.0
        
        #norm = (self.filters.weight.abs().sum() + self.filters2.weight.abs().sum()) / 16.0
        norm = 1.0
        return (diff * per_sample / norm).sum(axis=1).mean()


class FilterNet2:
    """
        
    """

    def _limit_weights(self, f):
        f.weight = f.weight.clamp(-1.0, 1.0)

    def _center_weights(self, f):
        f.weight = f.weight - f.weight.mean(axis=[1,2,3], keepdim=True)

    def __init__(self):
        self.filters  = Conv2dManual(1, 8, 3, bias=False, stride=2)
        self.filters2 = Conv2dManual(8, 24, 3, bias=False)
        self.filters3 = Conv2dManual(24, 8, 3, bias=False, stride=2)
        self._limit_weights(self.filters)
        self._center_weights(self.filters)
        self._limit_weights(self.filters2)
        self._limit_weights(self.filters3)

    def normalize_filters(self):
        self._limit_weights(self.filters)
        self._center_weights(self.filters)
        self._limit_weights(self.filters2)
        self._limit_weights(self.filters3)

    def features(self, x: Tensor) -> Tensor:
        x = self.filters(x).abs()
        x = self.filters2(x).abs()
        x = self.filters3(x).abs()
        return x

    def __call__(self, pred: Tensor) -> Tensor:
        pred_f = self.features(pred).avg_pool2d(kernel_size=(3, 3), stride=2)
        pred_f = pred_f * Tensor.rand(pred_f.shape) * 2.0
        per_sample = Tensor.rand(pred_f.shape[0]).reshape(-1, 1, 1, 1) * 2.0
        
        norm = (self.filters.weight.abs().sum() + self.filters2.weight.abs().sum()) / 16.0
        
        return (pred_f * per_sample / norm).sum(axis=1).mean()


# ── Sigma branch for ℓE loss ──────────────────────────────────────────────────

class SigmaBranch:
    """
    Lightweight branch that predicts per-pixel σ from pre-final backbone features.
    σ is used only during training (ℓE loss); discarded at inference.
    Output shape matches the pixel-shuffled prediction: (B, C_out//4, 2H, 2W).

    The paper (arXiv:2201.10084 Fig. 3) uses: 4× (Conv 3×3 + PReLU) with 160 channels,
    followed by a pixel-shuffle upsampler (Conv 3×3 + PixelShuffle). That's designed for
    EDSR-scale nets (1.4M–40M params). Here we use a single 3×3 + 1×1 head with
    mid = clamp(in_c // 8, 8, 48) to keep the branch small relative to tiny nets.
    """
    def __init__(self, in_c: int, out_c: int, is_3ch: bool = False):
        mid = max(8, min(in_c // 8, 48))
        self.conv1 = Conv2dManual(in_c, mid, 3)
        self.prelu_slope = Tensor.full((mid,), 0.25)  # learnable PReLU slope per channel
        self.head  = Conv2dManual(mid, out_c * 4, 1)  # out_c*4 → pixel_shuffle → out_c

    def __call__(self, feats: Tensor) -> Tensor:
        x = self.conv1(feats)
        x = x.maximum(0) + self.prelu_slope.reshape(1, -1, 1, 1) * x.minimum(0)
        x = self.head(x)
        return pixel_shuffle(x, 2).sigmoid()


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


def box_blur5x5_np(t: np.ndarray, is_wrapping: bool = False) -> np.ndarray:
    """Separable triangle blur (kernel [0.25, 0.5, 0.25]) on (B, C, H, W) or (C, H, W) float32."""
    batched = t.ndim == 4
    if not batched:
        t = t[np.newaxis]
    pad_mode = "wrap" if is_wrapping else "edge"
    # H pass
    t_pad = np.pad(t, ((0,0),(0,0),(1,1),(0,0)), mode=pad_mode)
    t = t_pad[:, :, :-2, :] * 0.25 + t_pad[:, :, 1:-1, :] * 0.5 + t_pad[:, :, 2:, :] * 0.25
    # W pass
    t_pad = np.pad(t, ((0,0),(0,0),(0,0),(1,1)), mode=pad_mode)
    t = t_pad[:, :, :, :-2] * 0.25 + t_pad[:, :, :, 1:-1] * 0.5 + t_pad[:, :, :, 2:] * 0.25
    return (t if batched else t[0]).astype(np.float32)


def upsample2x_np(t: np.ndarray, is_wrapping: bool = False) -> np.ndarray:
    """
    2x bilinear upsample. t: (B, C, H, W) or (C, H, W) numpy float32.
    Pixel-center aligned: output pixel i samples input at (i+0.5)/2 - 0.5,
    giving alternating weights 0.75/0.25 and 0.25/0.75 between neighbors.
    Implemented as two separable passes (H then W).
    """
    batched = t.ndim == 4
    if not batched:
        t = t[np.newaxis]
    B, C, H, W = t.shape
    pad_mode = "wrap" if is_wrapping else "edge"

    # H pass: (B, C, H, W) -> (B, C, 2H, W)
    t_pad = np.pad(t, ((0,0),(0,0),(1,1),(0,0)), mode=pad_mode)  # (B, C, H+2, W)
    out_h = np.empty((B, C, H * 2, W), dtype=np.float32)
    out_h[:, :, 0::2, :] = t_pad[:, :, 1:-1, :] * 0.75 + t_pad[:, :, :-2,  :] * 0.25
    out_h[:, :, 1::2, :] = t_pad[:, :, 1:-1, :] * 0.75 + t_pad[:, :, 2:,   :] * 0.25

    # W pass: (B, C, 2H, W) -> (B, C, 2H, 2W)
    t_pad = np.pad(out_h, ((0,0),(0,0),(0,0),(1,1)), mode=pad_mode)  # (B, C, 2H, W+2)
    out = np.empty((B, C, H * 2, W * 2), dtype=np.float32)
    out[:, :, :, 0::2] = t_pad[:, :, :, 1:-1] * 0.75 + t_pad[:, :, :, :-2 ] * 0.25
    out[:, :, :, 1::2] = t_pad[:, :, :, 1:-1] * 0.75 + t_pad[:, :, :, 2:  ] * 0.25

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
