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

Config string syntax
────────────────────
Each comma-separated token is "KxK_C[d]":
  K×K   – square kernel size
  C     – number of output channels
  d     – (optional suffix) depthwise convolution: each channel is convolved
           independently (groups=in_channels).  e.g. "3x3_8d"
  n     – (optional suffix) no bias parameters

Global prefix "b," (before the first token) marks a *bilinear-base* model.
  • The base upscale uses plain bilinear filtering (upsample2x) instead of EDI.
  • Detected automatically on load: the final weight key is "zfinalb" instead
    of "zfinal".
  Example: "b,3x3_12,1x1_12,3x3_24,1x1_4"
"""

import torch
import tinygrad.frontend.torch
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

#gConfig = "3x3_32,3x3_64,1x1_64,3x3_128,1x1_96,3x3_64,3x3_48,1x1_4" # beat waifu2x at some point on marona. 192k params
#gConfig = "3x3_24,3x3_48,1x1_64,3x3_72,1x1_32,3x3_24,1x1_4" # 64k params. 39.9 psnr
#gConfig = "3x3_24,3x3_32,1x1_24,3x3_64,1x1_24,3x3_16,1x1_4" # 27k params. 39.7 psnr
#gConfig = "3x3_24,3x3_32,1x1_24,3x3_16,1x1_4" # 11k params. 38.6 psnr. scrapping
#gConfig = "3x3_24,3x3_32,1x1_24,3x3_24,1x1_16,1x1_4" # 13.6k params. 39.45 psnr
#gConfig = "3x3_12,3x3_16,3x3_24,1x1_16,1x1_4" # 5.8k params. 38 psnr plateau at 150?
#gConfig = "3x3_16,3x3_16,1x1_16,3x3_24,1x1_16,1x1_4" # 6.7k params. same.
#gConfig = "3x3_16,3x3_24,1x1_16,3x3_24,1x1_16,1x1_4" # 8k params. 38.3?
#gConfig = "3x3_24,3x3_24,1x1_16,3x3_24,1x1_16,1x1_4" # 9.7k params. stuck at 38 too lol
#gConfig = "3x3_16,3x3_24,1x1_24,3x3_24,1x1_16,1x1_4" # 9.9k params. 39 psnr.
#gConfig = "3x3_12,3x3_24,3x3_24,1x1_12,1x1_4" # 8.2k params. 38.7 psnr.
#gConfig = "3x3_16,3x3_24,3x3_24,1x1_4" # 8.9k. oh. oops.
#gConfig = "3x3_12,1x1_12,3x3_16,3x3_24,1x1_4" # 5.6k params. 38.3+
#gConfig = "3x3_16,1x1_12,3x3_16,1x1_12,3x3_24,1x1_4" # 5k params. 38.3db psnr
#gConfig = "3x3_12,1x1_12,3x3_24,1x1_4" # 3k params, 37.5ish psnr
#gConfig = "3x3_16,1x1_10,3x3_20,1x1_16,1x1_4" # 2554 params.  37.3ish psnr
#gConfig = "3x3_12,3x3_12,3x3_8,3x3_4" # 2592 params. architecturally equivalent to cdnnx "fast". 37.67ish psnr.
#gConfig = "3x3_12,1x1_4,3x3_12,1x1_4,3x3_8,1x1_4" # 1000 params. 36.75db psnr.
#gConfig = "3x3_12,1x1_8,3x3_12,3x3_4" # 1536 params. good! made a copy and everything.
#gConfig = "3x3_12,1x1_8,3x3_8,1x1_4,3x3_4" # 992 params. garbage, unfortunately.
#gConfig = "3x3_12,1x1_8,3x3_8,3x3_4" # 1100 params. 37db psnr
#gConfig = "3x3_8,1x1_4,3x3_12,1x1_8,3x3_4" # 956 params. 36.8
#gConfig = "3x3_10,3x3_6,1x1_6,3x3_4" # 908 params. gets stuck.
#gConfig = "3x3_8,3x3_8,3x3_4" # 956 params (yes). 36.6.
#gConfig = "3x3_10,1x1_8,3x3_8,1x1_4,3x3_4" # 956 params (yes). bad too
#gConfig = "3x3_10,1x1_8,3x3_8,3x3_4" # 1064 params. 37+db psnr ---- 
#gConfig = "3x3_4,3x3_8,1x1_8,3x3_8,1x1_4,1x1_4" # 1048 params. garbage.
#gConfig = "3x3_4,3x3_8,1x1_8,3x3_8,1x1_4" # 1028 params. 36.7, bad.
#gConfig = "3x3_4,3x3_12,1x1_4" # 536 params. 36.3db psnr.
#gConfig = "3x3_8,1x1_6,3x3_4" # 354 params. 36.1db psnr.
#gConfig = "3x3_8,1x1_4,3x3_12,1x1_4" # 612 params. 36.6+.
#gConfig = "3x3_6,3x3_4" # 280 params. 35.5
#gConfig = "3x3_4,1x1_4,3x3_4,1x1_4" # 228 params. 34.4, garbage.
#gConfig = "3x3_8,1x1_4,3x3_4" # 264 params. 34.8db psnr.
#gConfig = "3x3_8,1x1_4" # 116 params. 34.35 psnr
#gConfig = "3x3_8,3x3_8dn,1x1_4" # 188 params. 34.75 psnr
#gConfig = "3x3_8,1x1_6,3x3_12d,1x1_4n" # 306 params. easily 35.9+ db psnr
#gConfig = "3x3_8,1x1_4,3x3_8d,1x1_4n" # 228 params. 35.1db psnr
#gConfig = "1x3_4,3x1_8d,1x1_4n" # 120 params. 33.8db psnr
#gConfig = "3x3_6,3x3_6d,1x1_4n" # 144 params. 34.8db psnr
#gConfig = "3x3_10,1x1_8,3x3_16d,1x1_8,3x3_4n" # 772 params. 36.8 psnr
#gConfig = "3x3_8,3x3_24d,1x1_8,3x3_4n" # 808 params. 37+ psnr

#gConfig = "b,3x3_6,1x1_6,1x1_4" # 130 params. unfortunately, too small.
#gConfig = "3x3_10,1x1_6,1x1_4n" # 180 params. 35.1db psnr
#gConfig = "3x3_24,3x3_96d,1x1_24,3x3_96d,1x1_32,1x1_4" # 6000 params. 
#gConfig = "3x3_24,3x3_32,1x1_24,3x3_24,1x1_16,1x1_4" # ... OLD. 13.6k ...
#gConfig = "3x3_12,3x3_72d,1x1_24,3x3_24,1x1_24,1x1_4" # 8500 params. 38.9db psnr. worse than 3x3_16,3x3_24,1x1_24,3x3_24,1x1_16,1x1_4
#gConfig = "3x3_12,3x3_32,1x1_24,3x3_96d,1x1_24,1x1_4" # 7788 params. 38.94db psnr.
#gConfig = "3x3_12,3x3_32,1x1_24,3x3_120d,1x1_24,1x1_4" # 8604 params. stuck at 38

gConfig = "3x3_24,3x3_48,1x1_64,3x3_320d,1x1_32,3x3_24,1x1_4" # 34k params. goal is 39.7~39.9 psnr.


# Intentionally extremely barely-leaky slope so that inference can treat the layer as ReLU instead of LeakyReLU.
# If this causes checkerboard/scanline artifacts in a given model, run a fine-tuning run with train.py --resume
#  and a low learning rate (e.g. 0.0005) and --leaky-slope 0.00001 Doing so will
#  fine-tune with true ReLU and get rid of the artifacts.
#LEAKY_SLOPE = 0.0002
LEAKY_SLOPE = 0.00005


def _parse_config(cfg: str) -> tuple:
    """Parse a config string into (specs, is_bilinear).

    specs       – list of (kernel_size, out_channels, is_depthwise, no_bias) 4-tuples
    is_bilinear – True when the string starts with the "b," prefix

    Suffix flags (append after the channel count in any order):
      d – depthwise: convolution stays within each channel (groups=in_channels)
      n – no bias:   bias term is omitted from the Conv2d

    Examples:
      "3x3_12,1x1_4"      →  ([(3,12,False,False),(1,4,False,False)], False)
      "b,3x3_12,1x1_4"    →  ([(3,12,False,False),(1,4,False,False)], True)
      "3x3_8d,1x1_4"      →  ([(3,8,True,False),(1,4,False,False)],  False)
      "3x3_8n,1x1_4"      →  ([(3,8,False,True),(1,4,False,False)],  False)
      "3x3_8dn,1x1_4"     →  ([(3,8,True,True),(1,4,False,False)],   False)
    """
    is_bilinear = cfg.startswith("b,")
    if is_bilinear:
        cfg = cfg[2:]
    specs = []
    for token in cfg.split(","):
        token = token.strip()
        # Strip suffix flags in any order; collect them as a set.
        flags = set()
        while token[-1] in ("d", "n"):
            flags.add(token[-1])
            token = token[:-1]
        kpart, c_str = token.split("_")
        k = int(kpart.split("x")[0])   # KxK assumed square
        specs.append((k, int(c_str), "d" in flags, "n" in flags))
    assert specs[-1][1] == 4, (
        f"Final layer must output exactly 4 channels for PixelShuffle×2, got {specs[-1][1]}"
    )
    return specs, is_bilinear


def _config_to_str(specs: list, is_bilinear: bool = False) -> str:
    """Inverse of _parse_config."""
    def _token(k, c, dw, nb):
        return f"{k}x{k}_{c}{'d' if dw else ''}{'n' if nb else ''}"
    body = ",".join(_token(*s) for s in specs)
    return ("b," + body) if is_bilinear else body


class UpscaleNet(nn.Module):
    """
    Input : (B, 1, H, W)  – single channel LR, float in [0, 1]
    Output: (B, 1, 2H, 2W) – residual above bilinear-interpolated input
    """

    def __init__(self, is_wrapping=False):
        super().__init__()
        self.padding_enabled = True
        self.is_wrapping = is_wrapping
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        specs, is_bilinear = _parse_config(gConfig)
        self.is_bilinear = is_bilinear
        self._build_layers(specs)
        self._init_weights()

    @staticmethod
    def _config_from_state(state: dict) -> tuple:
        """Recover the layer spec list and is_bilinear flag from weight tensor shapes.

        Returns (specs, is_bilinear) where specs is a list of
        (kernel_size, out_channels, is_depthwise, no_bias) 4-tuples.

        Bilinear detection: the final weight key is "zfinalb.weight" instead of
        "zfinal.weight".

        Depthwise detection: a Conv2d with groups=in_channels stores weights of
        shape (out_c, 1, k, k).  For the very first layer in_c is always 1, so
        we cannot distinguish – it is assumed non-depthwise.

        No-bias detection: the corresponding ".bias" key is absent from state.
        """
        is_bilinear = "zfinalb.weight" in state
        final_key = "zfinalb.weight" if is_bilinear else "zfinal.weight"
        final_pfx = "zfinalb"        if is_bilinear else "zfinal"
        assert final_key in state, f"No '{final_key}' found in state dict"

        keys = sorted(
            k for k in state if k.startswith("layers.") and k.endswith(".weight")
        )
        specs = []
        for i, k in enumerate(keys):
            w = state[k]
            pfx = k[:-len(".weight")]          # e.g. "layers.0"
            # shape[1]==1 on a non-first layer means depthwise (groups=in_c).
            # The first layer always has shape[1]==1 regardless, so skip it.
            is_depthwise = (i > 0) and (w.shape[1] == 1)
            no_bias = (pfx + ".bias") not in state
            specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))

        w = state[final_key]
        # zfinal is depthwise only when there are preceding layers to compare against.
        is_depthwise = (len(keys) > 0) and (w.shape[1] == 1)
        no_bias = (final_pfx + ".bias") not in state
        specs.append((w.shape[2], w.shape[0], is_depthwise, no_bias))
        return specs, is_bilinear

    def _build_layers(self, specs: list):
        """(Re-)construct all learnable layers from a list of
        (kernel_size, out_channels, is_depthwise, no_bias) 4-tuples."""
        in_c = 1
        convs = []
        for k, out_c, is_depthwise, no_bias in specs:
            pad = (k - 1) // 2
            # Only use a non-zero padding_mode when there is actual padding.
            # When padding=0 (e.g. 1x1 layers), leave padding_mode at the default
            # 'zeros' so PyTorch takes the fast cudnn path instead of F.pad + conv.
            pm = ("circular" if self.is_wrapping else "replicate") if pad > 0 else "zeros"
            groups = in_c if is_depthwise else 1
            convs.append(nn.Conv2d(in_c, out_c, kernel_size=k, padding=pad,
                                   padding_mode=pm, groups=groups, bias=not no_bias))
            in_c = out_c
        # Keep the final layer separate so forward() never needs a [:-1] slice.
        # The name encodes the base-upscale type so it can be detected on load.
        self.layers = nn.ModuleList(convs[:-1])
        final_attr = "zfinalb" if self.is_bilinear else "zfinal"
        setattr(self, final_attr, convs[-1])
        # Remove the stale attribute when switching between bilinear and EDI modes.
        for stale in ("zfinal", "zfinalb"):
            if stale != final_attr and hasattr(self, stale):
                delattr(self, stale)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Infer architecture from *state_dict* shapes, rebuild layers, then load."""
        specs, is_bilinear = self._config_from_state(state_dict)
        self.is_bilinear = is_bilinear
        self._build_layers(specs)
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

    @property
    def _final_layer(self) -> nn.Conv2d:
        """The final Conv2d, regardless of whether the model is bilinear or EDI."""
        return self.zfinalb if self.is_bilinear else self.zfinal

    def set_padding_enabled(self, enabled: bool):
        """Toggle replicate/circular padding on spatial layers.
        Disable during training when patches are pre-padded with real data."""
        self.padding_enabled = enabled
        for conv in [*self.layers, self._final_layer]:
            if conv.kernel_size[0] > 1:
                if enabled:
                    pad = (conv.kernel_size[0] - 1) // 2
                    conv.padding = (pad, pad)
                    conv.padding_mode = "circular" if self.is_wrapping else "replicate"
                else:
                    conv.padding = (0, 0)
                    conv.padding_mode = "zeros"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  →  (B, 1, 2H, 2W) residual"""
        for conv in self.layers:
            x = self.act(conv(x))
        x = self._final_layer(x)         # no activation on final layer
        return F.pixel_shuffle(x, 2)    # (B, 4, H, W) → (B, 1, 2H, 2W)

    @torch.no_grad()
    def upscale_channel(self, lr: torch.Tensor) -> torch.Tensor:
        x        = lr.unsqueeze(0).unsqueeze(0)
        if self.is_bilinear:
            base = upsample2x(x, self.is_wrapping)
        else:
            base = upscale_edi_2x(x, self.is_wrapping)
        residual = self(x)
        if not self.padding_enabled:
            _, _, rH, rW = residual.shape
            _, _, bH, bW = base.shape
            dH, dW = (bH - rH) // 2, (bW - rW) // 2
            base = base[:, :, dH:dH+rH, dW:dW+rW]
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
