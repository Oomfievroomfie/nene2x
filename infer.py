"""
infer.py  –  run the trained 2× upscaler on an image or a folder

Usage
-----
# single image
python infer.py  upscaler.safetensors  photo.png  -o photo_2x.png

# whole folder  (outputs written to <input_folder>/upscaled/ by default)
python infer.py  upscaler.safetensors  /path/to/images/

# whole folder, custom output directory
python infer.py  upscaler.safetensors  /path/to/images/  -o /path/to/out/

The model processes each colour channel independently (R, G, B, A …)
and sums the network's residual prediction with the bilinear baseline.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file

from model import UpscaleNet

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device, do_wrap=True) -> UpscaleNet:
    model = UpscaleNet(do_wrap)
    state = load_file(weights_path)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


# ── image I/O helpers ─────────────────────────────────────────────────────────

def _normalise_mode(img: Image.Image) -> Image.Image:
    """Convert exotic PIL modes to L / RGB / RGBA."""
    if img.mode in ("L", "RGB", "RGBA"):
        return img
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode == "LA":
        return img                       # handled below as 2-channel
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.asarray(img).astype(np.float32)
        arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "L")
    # fallback
    return img.convert("RGB")


def _to_float(img: Image.Image) -> tuple[torch.Tensor, str]:
    """
    PIL image  →  (C, H, W) float32 in [0, 1],  original mode string.
    """
    img  = _normalise_mode(img)
    mode = img.mode
    arr  = np.asarray(img)

    if arr.dtype == np.uint16:
        data = arr.astype(np.float32) / 65535.0
    else:
        data = arr.astype(np.float32) / 255.0

    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    return torch.from_numpy(data.copy()).permute(2, 0, 1), mode


def _to_pil(t: torch.Tensor, mode: str) -> Image.Image:
    """(C, H, W) float32  →  PIL image in the original mode."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], "L")
    return Image.fromarray(arr, mode)


# ── gridline removal ──────────────────────────────────────────────────────────
# Direct NumPy translation of gridline_remover.omwfx (author: Wareya).
#
# Each pass detects pixels that sit on an isolated spike along one axis by
# comparing two measures:
#   A  – second-difference (spike indicator)  |a*0.5 - b + q - c + d*0.5|
#   B  – first-difference  (slope indicator)  |a*-0.5 + q + d*-0.5|
# When A > B the pixel looks more like a thin line than a smooth gradient, so
# it is blended toward (q*0.5 + b*0.25 + c*0.25).  The blend weight is
# normalised by the local contrast range so the effect is scale-invariant.
# Pass X runs horizontally; pass Y runs vertically (with a 2× stronger blend).

def _gridline_pass(arr: np.ndarray, axis: int, y_pass: bool) -> np.ndarray:
    """
    One directional pass of gridline removal.

    arr   : float32 (H, W, C) in [0, 1]
    axis  : 1 = horizontal (X pass),  0 = vertical (Y pass)
    y_pass: if True use the 2× blend multiplier from the Y pass
    """
    # Replicate-pad 2 pixels on each side along the chosen axis so that
    # edge pixels don't wrap or go out of bounds.
    pw = [(0, 0)] * arr.ndim
    pw[axis] = (2, 2)
    p = np.pad(arr, pw, mode="edge")   # shape along axis is N+4

    # Slice out each of the five tap positions.
    # Along the padded axis the original data sits at indices [2 : N+2].
    #   q -> center (original pixel)
    #   a -> +2 ahead,  b -> +1 ahead
    #   c -> -1 behind, d -> -2 behind
    sl = [slice(None)] * arr.ndim
    def _tap(offset):
        s = list(sl)
        n = arr.shape[axis]
        lo = 2 + offset
        hi = lo + n         # hi = 2 + offset + n  →  may use None for the end
        s[axis] = slice(lo, hi if hi < p.shape[axis] else None)
        return p[tuple(s)]

    q = _tap( 0)
    a = _tap(+2)
    b = _tap(+1)
    c = _tap(-1)
    d = _tap(-2)

    A = np.abs(a * 0.5  - b + q - c + d * 0.5)   # second difference
    B = np.abs(a * -0.5 + q      + d * -0.5)      # first  difference

    lo = np.minimum(q, np.minimum(a, np.minimum(b, np.minimum(c, d))))
    hi = np.maximum(q, np.maximum(a, np.maximum(b, np.maximum(c, d))))
    mag = hi - lo + 0.01                           # local contrast range

    blend = q * 0.5 + b * 0.25 + c * 0.25         # neighbour-averaged value

    scale = 2.0 if y_pass else 1.0
    t = np.clip((A - B) / mag * scale, 0.0, 1.0)  # per-channel blend weight

    return q * (1.0 - t) + blend * t


def remove_gridlines(img: Image.Image) -> Image.Image:
    """
    Apply the two-pass (X then Y) gridline-removal filter to a PIL image.
    Operates in float32 to avoid rounding accumulation; preserves the
    original PIL mode (L / RGB / RGBA / etc.).
    """
    mode = img.mode
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:                   # grayscale: add channel dim
        arr = arr[:, :, np.newaxis]

    arr = _gridline_pass(arr, axis=1, y_pass=False)   # horizontal
    arr = _gridline_pass(arr, axis=0, y_pass=True)    # vertical

    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode)
    return Image.fromarray(arr, mode)


# ── YCgCo colour-space helpers ────────────────────────────────────────────────
# Operates on (C, H, W) float32 tensors in [0, 1].
# For RGB the three channels map to Y / Cg / Co.
# For RGBA the alpha channel is passed through unchanged.
# Grayscale (C == 1) is left untouched.

def _rgb_to_ycgco(t: torch.Tensor) -> torch.Tensor:
    """(C, H, W) RGB[A] → YCgCo[A].  Chroma is offset by 0.5 to stay in [0, 1]."""
    if t.shape[0] < 3:
        return t
    r, g, b = t[0], t[1], t[2]
    co  =  0.5  * r - 0.5  * b                  # [-0.5, 0.5]
    tmp = -0.25 * r + 0.5  * g - 0.25 * b
    y   =  0.25 * r + 0.5  * g + 0.25 * b
    cg  = tmp                                    # [-0.5, 0.5]
    out = torch.stack([y, cg + 0.5, co + 0.5], dim=0)  # shift chroma → [0, 1]
    if t.shape[0] == 4:                          # preserve alpha
        out = torch.cat([out, t[3:4]], dim=0)
    return out


def _ycgco_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """(C, H, W) YCgCo[A] → RGB[A].  Assumes chroma channels are offset by 0.5."""
    if t.shape[0] < 3:
        return t
    y       = t[0]
    cg      = t[1] - 0.5                        # undo the [0, 1] offset
    co      = t[2] - 0.5
    tmp     = y - cg
    r       = (tmp + co).clamp(0, 1)
    g       = (y   + cg).clamp(0, 1)
    b       = (tmp - co).clamp(0, 1)
    out = torch.stack([r, g, b], dim=0)
    if t.shape[0] == 4:
        out = torch.cat([out, t[3:4]], dim=0)
    return out


# ── bilinear upsample helper ─────────────────────────────────────────────────

def _bilinear_upsample_2x(ch: torch.Tensor) -> torch.Tensor:
    """
    2× upscale a single (H, W) channel with corner-aligned bilinear
    interpolation (align_corners=True).

    Corner-aligned means pixel (0,0) in the LR grid maps exactly to pixel (0,0)
    in the HR grid, and pixel (H-1, W-1) maps to (2H-2, 2W-2).  This is the
    correct convention for chroma upsampling alongside a luma channel that was
    upscaled with a learned network whose output is also corner-pinned.
    """
    return F.interpolate(
        ch[None, None],          # (1, 1, H, W)
        scale_factor=2,
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).squeeze(0)      # (2H, 2W)


# ── per-image upscaling ───────────────────────────────────────────────────────

timesum = 0

@torch.no_grad()
def upscale_image(model: UpscaleNet,
                  img: Image.Image,
                  device: torch.device,
                  tile_size: int | None = None,
                  wrap: bool = True,
                  ycgco: bool = False,
                  yonly: bool = False) -> Image.Image:
    global timesum
    import time
    """
    Upscale one PIL image 2×.

    tile_size  –  if set, process the image in overlapping tiles of this LR
                  size (useful for very large images that don't fit in VRAM).
                  Set to e.g. 256 for 8 GB GPU.
    ycgco      –  if True, convert RGB(A) to YCgCo before upscaling and back
                  afterwards.  The network sees decorrelated luma/chroma channels
                  which can improve sharpness.  Grayscale images are unaffected.
    yonly      –  if True, apply the network only to the Y (luma) channel in
                  YCgCo space and use corner-aligned bilinear interpolation for
                  the two chroma channels (and alpha if present).  Implies
                  --ycgco automatically.  Fastest mode for colour images.
    """
    tensor, mode = _to_float(img)              # (C, H, W)
    tensor = tensor.to(device)
    # --yonly implies YCgCo decomposition
    if yonly:
        ycgco = True
    if ycgco:
        tensor = _rgb_to_ycgco(tensor)
    C, H, W = tensor.shape

    start = time.perf_counter()
    if tile_size is None:
        # ── whole-image path (fastest) ────────────────────────────────────────
        channels = []
        for c in range(C):
            if yonly and c != 0:
                channels.append(_bilinear_upsample_2x(tensor[c]))  # (2H, 2W)
            else:
                channels.append(model.upscale_channel(tensor[c]))  # (2H, 2W)
        result = torch.stack(channels, dim=0)                   # (C, 2H, 2W)
    else:
        # ── tiled path (memory-safe for large images) ─────────────────────────
        #pad    = ((KERNEL_SIZE-1)//2)           # LR pixels of overlap on each edge
        pad    = (model.conv.size()[3] - 1) // 2
        stride = tile_size   # LR pixels per tile (without overlap)
        out_H, out_W = H * 2, W * 2

        result = torch.zeros(C, out_H, out_W, device=device)

        y_starts = range(0, H, stride)
        x_starts = range(0, W, stride)

        # THIS INTENTIONALLY DOES NOT FADE. FADING IS INCORRECT.
        # We only need 4 or 5 pixels of support on each side. Fading would be stupid.

        # Pad the entire tensor once to handle edge tiles easily
        # +1 because bilinear baseline needs 1 more pixel than the network's pad context
        P = pad + 1
        pad_mode = "circular" if wrap else "replicate"
        tensor_pad = F.pad(tensor.unsqueeze(0), (P, P, P, P), mode=pad_mode).squeeze(0)

        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(y0 + stride, H)
                x1 = min(x0 + stride, W)
                
                # Fetch tile with 5-pixel padding (for bilinear)
                tile_lr5 = tensor_pad[:, y0 : y1 + 2*P, x0 : x1 + 2*P].unsqueeze(0)
                
                # Fetch tile with 4-pixel padding (for network)
                tile_lr = tensor_pad[:, y0 + 1 : y1 + 2*P - 1, x0 + 1 : x1 + 2*P - 1]

                # Bilinear baseline for this padded tile
                base_pad = F.interpolate(
                    tile_lr5, scale_factor=2,
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                
                # Crop off the 1 LR pixel (2 HR pixels) of extra bilinear context
                tile_base = base_pad[:, 2:-2, 2:-2]

                # Network residual – only luma (c=0) when --yonly, else all channels
                net_chans = [0] if yonly else list(range(C))
                tile_res_channels = []
                for c in net_chans:
                    ch = tile_lr[c:c+1].unsqueeze(0)           # (1,1,th,tw)
                    res = model(ch).squeeze(0)                 # (1,2th,2tw)
                    tile_res_channels.append(res)
                tile_res = torch.cat(tile_res_channels, dim=0) # (len(net_chans),2th,2tw)

                tile_hr = (tile_base[net_chans] + tile_res).clamp(0, 1)

                # Crop off the 4-pixel LR safety padding (= 8 HR pixels), keeping core
                valid_hr = tile_hr[:, pad*2:-pad*2, pad*2:-pad*2]

                # Paste into result (only for the network-processed channels)
                for i, c in enumerate(net_chans):
                    result[c, y0*2:y1*2, x0*2:x1*2] = valid_hr[i]

        # --yonly: bilinear-upsample chroma (channels 1, 2) and alpha (3) in one
        # shot on the full image – cheaper than tiling and avoids tile boundaries.
        if yonly:
            for c in range(1, C):
                result[c] = _bilinear_upsample_2x(tensor[c])

    end = time.perf_counter()
    timesum += end-start
    
    if ycgco:
        result = _ycgco_to_rgb(result)
    return _to_pil(result, mode)


# ── main entry ────────────────────────────────────────────────────────────────

def process(args: argparse.Namespace):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
    except:
        pass
    
    print(f"Device : {device}")
    model = load_model(args.model, device, not args.no_wrap)
    print(f"Weights: {args.model}")

    inp = Path(args.input)

    # ── resolve input paths and output directory ──────────────────────────────
    if inp.is_file():
        if inp.suffix.lower() not in _IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {inp.suffix}")
        paths   = [inp]
        out_dir = Path(args.output).parent if args.output else inp.parent
        single_out = Path(args.output) if args.output else None
    elif inp.is_dir():
        paths   = sorted(p for p in inp.iterdir()
                         if p.suffix.lower() in _IMAGE_EXTS)
        out_dir = Path(args.output) if args.output else inp / "upscaled"
        single_out = None
        if not paths:
            raise FileNotFoundError(f"No images found in {inp}")
    else:
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── process each image ────────────────────────────────────────────────────
    for path in tqdm(paths, desc="Upscaling"):
        try:
            img = Image.open(path)

            result = upscale_image(model, img, device,
                                   tile_size=args.tile_size,
                                   wrap=not args.no_wrap,
                                   ycgco=args.ycgco,
                                   yonly=args.yonly)

            if args.gridline_removal:
                result = remove_gridlines(result)

            # Decide output path
            if single_out:
                # honour the exact filename the user asked for
                dest = single_out if single_out.suffix else \
                       single_out.with_suffix(path.suffix)
            else:
                # preserve extension; fall back to .png for formats that PIL
                # can't losslessly round-trip
                ext = path.suffix.lower()
                if ext in {".jpg", ".jpeg"}:
                    ext = ".png"   # avoid re-compression loss
                dest = out_dir / (path.stem + "_2x" + ext)

            result.save(dest)
            print(f"  {path.name}  {img.size} → {result.size}  →  {dest.name}")

        except Exception as e:
            print(f"  ERROR {path.name}: {e}")

    print(f"Elapsed for all inference: {timesum:.6f}s")
    
    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="2× upscale image(s) with a trained safetensors model"
    )
    p.add_argument("model",  help="Path to .safetensors weights file")
    p.add_argument("input",  help="Input image file or folder")
    p.add_argument("--output", "-o", default=None,
                   help="Output file (single input) or folder (directory input). "
                        "Default: <input>_2x.png  /  <input_dir>/upscaled/")
    p.add_argument("--tile-size", type=int, default=None,
                   help="Process in LR tiles of this size (e.g. 256) to reduce "
                        "VRAM usage on large images.  Default: whole image at once.")
    p.add_argument("--no-wrap", action="store_true")
    p.add_argument("--ycgco", action="store_true",
                   help="Convert RGB(A) to YCgCo before upscaling and back afterwards. "
                        "Decorrelates luma from chroma so the network processes each "
                        "independently.  Grayscale images are unaffected.")
    p.add_argument("--yonly", action="store_true",
                   help="Apply the network only to the Y (luma) channel in YCgCo space "
                        "and use corner-aligned bilinear interpolation for the chroma "
                        "channels.  Faster than --ycgco and avoids any chroma ringing "
                        "introduced by the network.  Implies --ycgco automatically.")
    p.add_argument("--gridline-removal", action="store_true",
                   help="Run a two-pass (horizontal then vertical) gridline-removal "
                        "filter on the upscaled image before saving.  Each pass detects "
                        "isolated single-pixel spikes along its axis and blends them "
                        "toward their nearest neighbours.  Useful for suppressing "
                        "upscaling artefacts that form regular grid patterns.")
    p.add_argument("--leaky-slope",      type=float,   default=0.003)
    args = p.parse_args()
    import model
    model.LEAKY_SLOPE = args.leaky_slope
    process(args)


if __name__ == "__main__":
    main()
