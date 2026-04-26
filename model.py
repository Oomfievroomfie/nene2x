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

# k: kernel size (single dimension)
# c: number of kernels
# ms: size of middle layers
# nm: number of middle layers
gConfig = {
    #"k": 9, "c": 128, "ms": 64, "nm": 2 # 90KB
    #"k": 9, "c": 48, "ms": 48, "nm": 3 # ~40KB, bad
    #"k": 9, "c": 64, "ms": 48, "nm": 3 # ~52KB, good <- priority -----
    
    #"k": 7, "c": 64, "ms": 32, "nm": 3 # ~30KB, good
    #"k": 7, "c": 32, "ms": 48, "nm": 3 # ~32KB, good
    #"k": 7, "c": 64, "ms": 48, "nm": 2 # ~35KB, good
    #"k": 7, "c": 32, "ms": 32, "nm": 3 # ~20KB, good <- priority -----
    
    #"k": 5, "c": 32, "ms": 24, "nm": 2 # ~9KB, good
    #"k": 5, "c": 24, "ms": 24, "nm": 2 # ~9KB, good
    #"k": 5, "c": 16, "ms": 24, "nm": 3 # ~9KB, good
    #"k": 5, "c": 24, "ms": 28, "nm": 3 # ~13KB, good <- priority -----
    
    #"k": 5, "c": 16, "ms": 20, "nm": 3 # ~7KB, good <- priority -----
    
    #"k": 5, "c": 12, "ms": 14, "nm": 2 # 3.56KB, bad (checkerboards
    #"k": 5, "c": 12, "ms": 16, "nm": 2 # ~4KB, ...
    #"k": 5, "c": 12, "ms": 16, "nm": 2 # 3.9KB, decent
    
    #"k": 3, "c": 16, "ms": 24, "nm": 3 # ~8KB, good
    #"k": 3, "c": 20, "ms": 20, "nm": 2 # ~5KB, OKish
    #"k": 3, "c": 16, "ms": 16, "nm": 1 # ~2.5KB, ???
    #"k": 3, "c": 16, "ms": 8, "nm": 2 # ~2.1KB, bad
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # ~5KB, OKish
    #"k": 3, "c": 16, "ms": 12, "nm": 2 # ~2.8KB, barely OKish
    #"k": 3, "c": 8, "ms": 16, "nm": 2 # ~2.77KB, OKish somehow
    #"k": 3, "c": 12, "ms": 14, "nm": 2 # ~???KB, barely OKish
    #"k": 3, "c": 12, "ms": 16, "nm": 2 # ~3.2KB, good
    #"k": 5, "c": 10, "ms": 8, "nm": 2 # 2.35KB, bad
    #"k": 5, "c": 16, "ms": 16, "nm": 2 # bad
    #"k": 5, "c": 12, "ms": 12, "nm": 2 # 3.2KB, good <- priority ----- (this is where things start to get dicey for realtime)
    
    #"k": 3, "c": 32, "ms": 32, "nm": 3 # really good actually!
    #"k": 3, "c": 32, "ms": 10, "nm": 3 # disappointing
    #"k": 3, "c": 32, "ms": 16, "nm": 3 # disappointing
    #"k": 3, "c": 32, "ms": 24, "nm": 3 # plausible, cutting early to test 20
    #"k": 3, "c": 32, "ms": 20, "nm": 3 # works but took until epoch 200ish to eliminate all basic artifacts
    
    #"k": 3, "c": 32, "ms": 24, "nm": 2 # plausible at epoch 100
    #"k": 3, "c": 24, "ms": 24, "nm": 2 # implausible
    #"k": 3, "c": 28, "ms": 24, "nm": 2 # blocky gradients at epoch 240 still
    
    #"k": 3, "c": 30, "ms": 22, "nm": 2 # bad
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # artifacty at 200
    #"k": 3, "c": 16, "ms": 24, "nm": 2 # right on the edge of "barely good enough"
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # likewise, but smaller so lol
    #"k": 3, "c": 24, "ms": 20, "nm": 2 # works well, but just barely
    #"k": 3, "c": 20, "ms": 20, "nm": 2 # works well, but just barely
    #"k": 3, "c": 16, "ms": 20, "nm": 3 # low artifact density
    
    #"k": 3, "c": 16, "ms": 20, "nm": 1
    
    #"k": 3, "c": 12, "ms": 10, "nm": 2 # 2.1KBish
    #"k": 3, "c": 10, "ms": 12, "nm": 2 # 2.28KBish
    #"k": 3, "c": 10, "ms": 10, "nm": 2 # 1.99KBish
    
    #"k": 3, "c": 12, "ms": 12, "nm": 3 # 3.21KB, good (yes)
    
    #"k": 5, "c": 12, "ms": 8, "nm": 2 # ?
    #"k": 5, "c": 12, "ms": 8, "nm": 1 # ? failed to train lol
    #"k": 5, "c": 16, "ms": 16, "nm": 1 # ? 3.4KB, looks good.
    #"k": 5, "c": 12, "ms": 16, "nm": 1 # ? 2.7KB, worse than below.
    #"k": 3, "c": 12, "ms": 10, "nm": 2 # ...., looks good. consider it.
    #"k": 3, "c": 12, "ms": 6, "nm": 2 # 1.6KB -- failure to converge
    
    #"k": 3, "c": 10, "ms": 12, "nm": 1 # ~1.5KB, bad
    #"k": 3, "c": 12, "ms": 6, "nm": 2 # ~1.6KB, bad
    #"k": 3, "c": 12, "ms": 8, "nm": 2 # ~1.9KB, good <- priority ----- (this is where realtime becomes viable)
    #"k": 5, "c": 10, "ms": 6, "nm": 2 # ~2.1KB, bad
    #"k": 5, "c": 10, "ms": 12, "nm": 1 # ~2.2KB, ...ehhh, worse than 3_12_8_2 so far
    
    
    #"k": 3, "c": 16, "ms": 8, "nm": 0 # ~1.1KB, bad
    #"k": 3, "c": 12, "ms": 8, "nm": 1 # ~1.4KB, ehhhh
    #"k": 3, "c": 8, "ms": 8, "nm": 2 # ~1.57KB, ehhhh
    #"k": 3, "c": 10, "ms": 10, "nm": 1 # 1.42KB, best of this lot, but...
    #"k": 3, "c": 8, "ms": 12, "nm": 1 # 1.35KB, bad
    #"k": 3, "c": 12, "ms": 12, "nm": 1 # 1.71KB... no, cancel
    #"k": 3, "c": 12, "ms": 10, "nm": 1 # 1.52KB... good enough <- priority ----- (cheapest version that gives reasonable results so far)
    
    #"k": 3, "c": 12, "ms": 0, "nm": 0 # ehhh. it did SOME stuff, but not much.
    #"k": 3, "c": 10, "ms": 6, "nm": 1 # actually works lol <- priority ------
    
    #"k": 3, "c": 13, "ms": 14, "nm": 2 #
    #"k": 3, "c": 9, "ms": 16, "nm": 2 #
    #"k": 3, "c": 14, "ms": 14, "nm": 2
    #"k": 3, "c": 9, "ms": 14, "nm": 1 # bad
    #"k": 3, "c": 9, "ms": 9, "nm": 2 # bad
    #"k": 3, "c": 14, "ms": 10, "nm": 2 # OKish but....
    #"k": 3, "c": 14, "ms": 14, "nm": 1 # very initialization rng sensitive
    #"k": 3, "c": 12, "ms": 16, "nm": 1 # 
    #"k": 3, "c": 9, "ms": 14, "nm": 1 # OK for the size
    #"k": 3, "c": 9, "ms": 9, "nm": 1 #OKish but...
    #"k": 3, "c": 11, "ms": 11, "nm": 1 <--- picked.
    
    
    # WITH NEW: NEW MEANS THAT THERE IS AN ADDITIONAL 4-OUTPUT 3x3 PRE-PASS CONVOLUTION
    
    
    #"new": True, "k": 3, "c": 8, "ms": 12, "nm": 2 # ?
    #"new": True, "k": 3, "c": 11, "ms": 11, "nm": 2 # OK. 759 params. can i get smaller?
    #"new": True, "k": 3, "c": 9, "ms": 11, "nm": 2 # ehhh. (663 params)
    #"new": True, "k": 3, "c": 11, "ms": 9, "nm": 2 # (685) how do?
    #"new": True, "k": 3, "c": 14, "ms": 14, "nm": 2 # good! but 1040 parameters.
    #"new": True, "k": 3, "c": 13, "ms": 11, "nm": 2 # artifacty. 850ish params.
    #"new": True, "k": 3, "c": 12, "ms": 12, "nm": 2 # 848 params. sharp but artifacty.
    #"new": True, "k": 3, "c": 12, "ms": 14, "nm": 2 # already half-decent at 80 epochs! so is 14 outputs the key?
    #"new": True, "k": 3, "c": 8, "ms": 13, "nm": 2 # stuck with jaggies
    #"new": True, "k": 3, "c": 8, "ms": 14, "nm": 2 # already good at 80 epochs lmao
    #"new": True, "k": 3, "c": 8, "ms": 14, "nm": 2 # good at 222 epochs, but... iffy. (732 parameters)
    #"new": True, "k": 3, "c": 9, "ms": 13, "nm": 2 # sus at 80 epochs. 14 in mid layers is the necessity.
    
    #"new": True, "k": 3, "c": 9, "ms": 14, "nm": 2 # good, somehow! but vulnerable to bad initialization. 783 params. see upscaler_n_3x3_9_14_2.safetensors
    
    #"k": 3, "c": 9, "ms": 14, "nm": 2 # 500 params. artifacty at 160 in ways that look like it won't train out.
    #"k": 3, "c": 10, "ms": 14, "nm": 2 # 526 params. artifacty.
    #"k": 3, "c": 12, "ms": 14, "nm": 2 # 526 params. artifacty.
    
    
    
    #"new": True, "k": 3, "c": 16, "ms": 16, "nm": 2 # ew
    #"new": True, "k": 3, "c": 12, "ms": 18, "nm": 2 # yay (very good at 200 epochs)
    #"new": True, "k": 3, "c": 16, "ms": 20, "nm": 2 #
    #"new": True, "k": 3, "c": 14, "ms": 18, "nm": 3 #
    #"new": True, "k": 3, "c": 20, "ms": 22, "nm": 2
    #"new": True, "k": 3, "c": 18, "ms": 16, "nm": 3 # blocky
    #"new": True, "k": 3, "c": 18, "ms": 18, "nm": 3 # blocky
    #"new": True, "k": 3, "c": 20, "ms": 22, "nm": 2 # badly blocky at 450 epoche
    #"new": True, "k": 3, "c": 16, "ms": 18, "nm": 3 # at 250 epochs, not blocky.
    
    #"new": True, "k": 5, "c": 14, "ms": 10, "nm": 2 # too slow
    #"new": True, "k": 5, "c": 11, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 11, "ms": 16, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 12, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 14, "ms": 10, "nm": 2 # OFF A CLIFF.
    #"new": True, "k": 5, "c": 13, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 32, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 36, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 40, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 48, "nm": 2 # not too slow.
    #"new": True, "k": 5, "c": 13, "ms": 64, "nm": 2 # WAY TOO SLOW
    
    #"new": True, "k": 3, "c": 24, "ms": 24, "nm": 3
    
    #"new": True, "k": 5, "c": 12, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 10, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 24, "nm": 3 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 32, "nm": 2 # not too slow
    
    #"new": True, "k": 3, "c": 32, "ms": 32, "nm": 3 # good, BUT, over-wrought, lmao
    #"new": True, "k": 3, "c": 24, "ms": 28, "nm": 3 #
    #"new": True, "k": 3, "c": 22, "ms": 24, "nm": 3 # wheee (priority ------)
    
    
    # OOPS I IMPLEMENTED "NEW" WRONG LOL AND GROUPS WERE BEING BROADCAST AS CHANNELS. TIME TO REDO IT ALL LOL
    
    
    
    #"new": True, "k": 3, "c": 24, "ms": 12, "nm": 2 # ...
    #"new": True, "k": 3, "c": 32, "ms": 18, "nm": 1 # good!!! see upscaler - Copy (27)
    #"new": True, "k": 3, "c": 28, "ms": 16, "nm": 1 # dead almost every time. when NOT dead, tends to develop noise later on.
    #"new": True, "k": 3, "c": 24, "ms": 18, "nm": 1 # need good initialization RNG, barely works.
    #"new": True, "k": 3, "c": 28, "ms": 18, "nm": 1 # works more than "barely" but needs good rng
    
    #"new": True, "k": 3, "c": 32, "ms": 12, "nm": 1 # actually works but sometimes has screwy rng. same with 32 -> 14
    #"new": True, "k": 3, "c": 32, "ms": 16, "nm": 1 # actually works and managed to train properly
    
    #"new": True, "k": 3, "c": 32, "ms": 22, "nm": 2 # -------
    
    #"new": True, "k": 5, "c": 32, "ms": 26, "nm": 3 # -------
    
    
    
    "k": 3, "c": 16, "ms": 16, "nm": 2
}

# Intentionally extremely barely-leaky slope so that inference can treat the layer as ReLU instead of LeakyReLU.
# If this causes checkerboard/scanline artifacts in a given model, run a fine-tuning run with train.py --resume
#  and a low learning rate; doing so will #  fine-tune with true ReLU and get rid of the artifacts.
LEAKY_SLOPE = 0.001
KERNEL_SIZE = gConfig["k"]

class UpscaleNet(nn.Module):
    """
    Input : (B, 1, H, W)  – single channel LR, float in [0, 1]
    Output: (B, 1, 2H, 2W) – residual above bilinear-interpolated input
    """

    def __init__(self, is_wrapping=False):
        super().__init__()
        self.is_wrapping = is_wrapping
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        self._build_layers("new" in gConfig, gConfig["c"], gConfig["ms"], gConfig["nm"], gConfig["k"])
        self._init_weights()

    @staticmethod
    def _config_from_state(state: dict) -> dict:
        """Recover k / c / ms / nm entirely from safetensors weight shapes."""
        conv_w = state["conv.weight"]          # (c, 1, k, k)
        c = conv_w.shape[0]
        k = conv_w.shape[2]

        mid_keys = sorted(
            key for key in state if key.startswith("mids.") and key.endswith(".weight")
        )
        nm = len(mid_keys)
        if nm > 0:
            ms = state[mid_keys[0]].shape[0]  # output channels of mids.0
        else:
            ms = c  # unused when nm == 0, but keep it sane

        ret = {"k": k, "c": c, "ms": ms, "nm": nm}
        
        if "bconv.weight" in state:
            ret["new"] = True
        
        return ret

    def _build_layers(self, new: bool, c: int, ms: int, nm: int, k: int):
        """(Re-)construct all learnable layers from explicit dimensions."""
        self.conv_output_size = c
        self.mid_size         = ms
        self.num_mids         = nm
        self.mids             = nn.ModuleList()
        
        _ichannels = 1
        _igroups = 1
        
        # WITH NEW: NEW MEANS THAT THERE IS AN ADDITIONAL 4-OUTPUT 3x3 PRE-PASS CONVOLUTION
        if new:
            _ichannels = 4
            _igroups = 4
            self.bconv = nn.Conv2d(
                1, _ichannels, kernel_size=3, padding=1,
                padding_mode="circular" if self.is_wrapping else "replicate",
            )
        else:
            try:
                del self.bconv
            except:
                pass
        
        self.conv = nn.Conv2d(
            _ichannels, c, kernel_size=k, padding=(k - 1) // 2, groups=_igroups,
            padding_mode="circular" if self.is_wrapping else "replicate",
        )
        if nm == 0:
            self.zfinal = nn.Conv2d(c, 4, kernel_size=1)
        else:
            self.mids.append(nn.Conv2d(c, ms, kernel_size=1))
            for _ in range(1, nm):
                self.mids.append(nn.Conv2d(ms, ms, kernel_size=1))
            self.zfinal = nn.Conv2d(ms, 4, kernel_size=1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Infer architecture from *state_dict* shapes, rebuild layers, then load."""
        cfg = self._config_from_state(state_dict)
        
        gConfig = cfg
        # always use relu during inference or resumption
        self.act = nn.ReLU(inplace=True)
        self._build_layers("new" in cfg, cfg["c"], cfg["ms"], cfg["nm"], cfg["k"])
        self._init_weights()
        
        #return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=LEAKY_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  →  (B, 1, 2H, 2W) residual"""
        
        # NEW
        try:
            x = self.act(self.bconv(x))
        except:
            pass
        
        x = self.act(self.conv(x))   # KERNEL_SIZE x KERNEL_SIZE feature extraction
        for i in range(0, self.num_mids):
            x = self.act(self.mids[i](x))
        x = self.zfinal(x)            # no activation on output
        return F.pixel_shuffle(x, 2) # (B,4,H,W) → (B,1,2H,2W)

    @torch.no_grad()
    def upscale_channel(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Upscale one image channel.
        lr : (H, W) float32 tensor in [0, 1]
        returns : (2H, 2W) float32 tensor clamped to [0, 1]
        """
        x        = lr.unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
        base     = upsample2x(x, self.is_wrapping)  # (1,1,2H,2W)
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
        out_pad = F.interpolate(t_pad, scale_factor=2, mode="bilinear", align_corners=True)
        out = out_pad[:,:,2:-2,2:-2]
        
    return out if batched else out.squeeze(0)
