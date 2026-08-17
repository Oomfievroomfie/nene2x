"""
infer.py  –  run the trained 2× upscaler on an image or a folder (tinygrad port)

Usage
-----
# single image
python infer.py  upscaler.safetensors  photo.png  -o photo_2x.png

# whole folder
python infer.py  upscaler.safetensors  /path/to/images/
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import ddslop

from tinygrad import Tensor
from tinygrad.nn.state import safe_load

from model import UpscaleNet

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(weights_path: str, do_wrap: bool = True) -> UpscaleNet:
    model = UpscaleNet(do_wrap)
    state = safe_load(weights_path)
    model.load_weights(state)

    pad = sum((conv._k - 1) // 2 * conv._dilation
              for conv in [*model.layers, model._final_layer]
              if conv._k > 1)
    model.infer_pad = pad
    model.set_padding_enabled(False)

    from tinygrad.nn.state import get_state_dict
    total_params = sum(p.numpy().size for p in get_state_dict(model).values())
    print(f"Parameters: {total_params:,}")
    return model


# ── image I/O helpers ─────────────────────────────────────────────────────────

def _normalise_mode(img: Image.Image) -> Image.Image:
    if img.mode in ("L", "RGB", "RGBA"):
        return img
    if img.mode == "P":
        return img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode == "LA":
        return img
    if img.mode in ("I", "I;16", "I;16B"):
        arr = np.asarray(img).astype(np.float32)
        arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "L")
    return img.convert("RGB")


def _to_float(img: Image.Image) -> tuple:
    img  = _normalise_mode(img)
    mode = img.mode
    arr  = np.asarray(img)
    if arr.dtype == np.uint16:
        data = arr.astype(np.float32) / 65535.0
    else:
        data = arr.astype(np.float32) / 255.0
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    return data.transpose(2, 0, 1).copy(), mode  # (C, H, W)


def _to_pil(arr: np.ndarray, mode: str) -> Image.Image:
    """(C, H, W) float32  →  PIL image."""
    out = (arr.transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
    if out.shape[2] == 1:
        return Image.fromarray(out[:, :, 0], "L")
    return Image.fromarray(out, mode)


# ── gridline removal ──────────────────────────────────────────────────────────

def _gridline_pass(arr: np.ndarray, axis: int, y_pass: bool) -> np.ndarray:
    pw = [(0, 0)] * arr.ndim
    pw[axis] = (2, 2)
    p = np.pad(arr, pw, mode="edge")

    sl = [slice(None)] * arr.ndim
    def _tap(offset):
        s = list(sl)
        n = arr.shape[axis]
        lo = 2 + offset
        hi = lo + n
        s[axis] = slice(lo, hi if hi < p.shape[axis] else None)
        return p[tuple(s)]

    q = _tap( 0); a = _tap(+2); b = _tap(+1); c = _tap(-1); d = _tap(-2)

    A = np.abs(a * 0.5  - b + q - c + d * 0.5)
    B = np.abs(a * -0.5 + q      + d * -0.5)

    lo  = np.minimum(q, np.minimum(a, np.minimum(b, np.minimum(c, d))))
    hi  = np.maximum(q, np.maximum(a, np.maximum(b, np.maximum(c, d))))
    mag = hi - lo + 0.01
    blend = q * 0.5 + b * 0.25 + c * 0.25
    scale = 2.0 if y_pass else 1.0
    t = np.clip((A - B) / mag * scale, 0.0, 1.0)
    return q * (1.0 - t) + blend * t


def remove_gridlines(img: Image.Image) -> Image.Image:
    mode = img.mode
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    arr = _gridline_pass(arr, axis=1, y_pass=False)
    arr = _gridline_pass(arr, axis=0, y_pass=True)
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        return Image.fromarray(arr[:, :, 0], mode)
    return Image.fromarray(arr, mode)


# ── YCgCo colour-space helpers ────────────────────────────────────────────────

def _rgb_to_ycgco(t: np.ndarray) -> np.ndarray:
    """(C, H, W) RGB[A] → YCgCo[A]. Operates on numpy float32."""
    if t.shape[0] < 3:
        return t
    r, g, b = t[0], t[1], t[2]
    co  =  0.5  * r - 0.5  * b
    tmp = -0.25 * r + 0.5  * g - 0.25 * b
    y   =  0.25 * r + 0.5  * g + 0.25 * b
    cg  = tmp
    out = np.stack([y, cg + 0.5, co + 0.5], axis=0)
    if t.shape[0] == 4:
        out = np.concatenate([out, t[3:4]], axis=0)
    return out


def _ycgco_to_rgb(t: np.ndarray) -> np.ndarray:
    """(C, H, W) YCgCo[A] → RGB[A]."""
    if t.shape[0] < 3:
        return t
    y  = t[0]
    cg = t[1] - 0.5
    co = t[2] - 0.5
    tmp = y - cg
    r = np.clip(tmp + co, 0, 1)
    g = np.clip(y   + cg, 0, 1)
    b = np.clip(tmp - co, 0, 1)
    out = np.stack([r, g, b], axis=0)
    if t.shape[0] == 4:
        out = np.concatenate([out, t[3:4]], axis=0)
    return out


# ── baseline (no-NN) upscale dispatch ──────────────────────────────────────

BASELINE_MODES = ("bilinear", "blurbilinear", "jinc", "edi")


def _baseline_upscale_np(t: np.ndarray, mode: str, is_wrapping: bool) -> np.ndarray:
    """2× upscale (B, C, H, W) or (C, H, W) float32 with no neural network."""
    import model as model_module
    if mode == "bilinear":
        return model_module.upsample2x_np(t, is_wrapping)
    elif mode == "blurbilinear":
        blurred = model_module.box_blur5x5_np(t, is_wrapping)
        return model_module.upsample2x_np(blurred, is_wrapping)
    elif mode == "jinc":
        return model_module.jinc_upsample2x_np(t, is_wrapping)
    elif mode == "edi":
        return model_module.upscale_edi_2x_np(t, is_wrapping)
    raise ValueError(f"Unknown baseline mode: {mode}")


# ── bilinear upsample helper ─────────────────────────────────────────────────

def _bilinear_upsample_2x(ch: np.ndarray) -> np.ndarray:
    """2× upscale a (H, W) channel with corner-aligned bilinear interpolation.

    Corner-aligned = half-pixel offset: the 2× output grid extends beyond
    [0, H-1], so sampling near the boundary repeats edge pixels rather than
    interpolating to nothing.

    For a 2→1000 case this puts ~250 output positions past each end of the
    source range — those get edge-clamped instead of interpolating inward.
    """
    H, W = ch.shape
    H2, W2 = H * 2, W * 2
    # half-pixel offset: (i + 0.5) * H/H2 - 0.5  →  linspace(-0.25, H-0.75, H2)
    ys = np.linspace(-0.25, H - 0.75, H2, dtype=np.float32)
    xs = np.linspace(-0.25, W - 0.75, W2, dtype=np.float32)
    # edge-clamp so out-of-range positions sample the boundary pixel
    ys = ys.clip(0, H - 1)
    xs = xs.clip(0, W - 1)
    y0 = np.floor(ys).astype(np.int32).clip(0, H - 2 if H > 1 else 0)
    x0 = np.floor(xs).astype(np.int32).clip(0, W - 2 if W > 1 else 0)
    y1 = y0 + 1
    x1 = x0 + 1
    ty = (ys - y0).astype(np.float32)[:, np.newaxis]
    tx = (xs - x0).astype(np.float32)[np.newaxis, :]
    a = ch[y0[:, np.newaxis], x0[np.newaxis, :]]
    b = ch[y0[:, np.newaxis], x1[np.newaxis, :]]
    c = ch[y1[:, np.newaxis], x0[np.newaxis, :]]
    d = ch[y1[:, np.newaxis], x1[np.newaxis, :]]
    return a * (1-ty)*(1-tx) + b * (1-ty)*tx + c * ty*(1-tx) + d * ty*tx


# ── per-image upscaling ───────────────────────────────────────────────────────

timesum = 0

def upscale_image(model: UpscaleNet,
                  img: Image.Image,
                  tile_size: int | None = None,
                  wrap: bool = True,
                  ycgco: bool = False,
                  yonly: bool = False,
                  baseline: str | None = None) -> Image.Image:
    global timesum
    import time

    noise_scale = Tensor([1.0])
    tensor, mode = _to_float(img)  # numpy (C, H, W)

    if yonly:
        ycgco = True
    if ycgco and not model.is_3ch:
        tensor = _rgb_to_ycgco(tensor)
    C, H, W = tensor.shape

    pad      = model.infer_pad
    pad_mode = "wrap" if wrap else "edge"

    start = time.perf_counter()

    if baseline is not None:
        # ── fast-path: no neural network, just the baseline upscaler ──
        base_np = _baseline_upscale_np(tensor[np.newaxis], baseline, wrap)[0]
        if C == 4:
            alpha_up = _bilinear_upsample_2x(tensor[3])
            base_np = np.concatenate([base_np, alpha_up[np.newaxis]], axis=0)
        result = base_np
    elif model.is_3ch:
        # pass all 3 RGB planes together; alpha (if present) bilinear-upscaled separately
        rgb3 = tensor[:3]
        if tile_size is None or tile_size <= 1:
            if pad > 0:
                rgb_in = np.pad(rgb3, ((0,0),(pad,pad),(pad,pad)), mode=pad_mode)
            else:
                rgb_in = rgb3
            ch_t = Tensor(rgb_in.astype(np.float32))
            result = model.upscale_channel(ch_t, noise_scale).numpy()  # (3, 2H, 2W)
        else:
            stride       = tile_size
            out_H, out_W = H * 2, W * 2
            off          = 1
            result       = np.zeros((3, out_H + off, out_W + off), dtype=np.float32)

            ph = pad + stride
            total_h = H + ph + ph
            total_w = W + ph + ph
            if pad_mode == "wrap" and (ph >= H or ph >= W or stride >= W):
                ys = (np.arange(total_h) - ph) % H
                xs = (np.arange(total_w) - ph) % W
                tensor_pad = rgb3[:, ys][:, :, xs]
            else:
                tensor_pad = np.pad(rgb3, ((0,0),(ph,ph),(ph,ph)), mode=pad_mode)

            for y0 in range(0, H, stride):
                y1 = min(y0 + stride, H)
                for x0 in range(0, W, stride):
                    x1 = min(x0 + stride, W)
                    tile = tensor_pad[:, y0+stride : y1+stride+2*pad,
                                         x0+stride : x1+stride+2*pad]
                    out_h, out_w = (y1 - y0) * 2, (x1 - x0) * 2
                    ch_t = Tensor(tile)
                    result[:, y0*2+off : y1*2+off, x0*2+off : x1*2+off] = \
                        model.upscale_channel(ch_t, noise_scale).numpy()[:, :out_h, :out_w]

            result = result[:, off:off+out_H, off:off+out_W]

        if C == 4:
            alpha_up = _bilinear_upsample_2x(tensor[3])
            result = np.concatenate([result, alpha_up[np.newaxis]], axis=0)
    else:
        if tile_size is None or tile_size <= 1:
            # whole-image path
            if pad > 0:
                tensor_in = np.pad(tensor, ((0,0),(pad,pad),(pad,pad)), mode=pad_mode)
            else:
                tensor_in = tensor
            channels = []
            for c in range(C):
                if yonly and c != 0:
                    channels.append(_bilinear_upsample_2x(tensor[c]))
                else:
                    ch_t = Tensor(tensor_in[c].astype(np.float32))
                    channels.append(model.upscale_channel(ch_t, noise_scale).numpy())
            result = np.stack(channels, axis=0)  # (C, 2H, 2W)
        else:
            stride       = tile_size
            out_H, out_W = H * 2, W * 2
            off          = 1
            result       = np.zeros((C, out_H + off, out_W + off), dtype=np.float32)

            ph = pad + stride
            total_h = H + ph + pad
            total_w = W + ph + pad
            if pad_mode == "wrap" and (ph >= H or pad >= W or stride >= W):
                ys = (np.arange(total_h) - ph) % H
                xs = (np.arange(total_w) - ph) % W
                tensor_pad = tensor[:, ys][:, :, xs]
            else:
                tensor_pad = np.pad(tensor, ((0,0),(ph,pad),(ph,pad)), mode=pad_mode)

            full_tile_h = stride + 2 * pad
            full_tile_w = stride + 2 * pad
            net_chans = [0] if yonly else list(range(C))
            for y0 in range(0, H, stride):
                y1 = min(y0 + stride, H)
                for x0 in range(0, W, stride):
                    x1 = min(x0 + stride, W)
                    tile = tensor_pad[:, y0+stride : y1+stride+2*pad,
                                         x0+stride : x1+stride+2*pad]
                    th, tw = tile.shape[1], tile.shape[2]
                    if th < full_tile_h or tw < full_tile_w:
                        tile = np.pad(tile,
                                      ((0,0),(0, full_tile_h-th),(0, full_tile_w-tw)),
                                      mode="edge")
                    out_h, out_w = (y1 - y0) * 2, (x1 - x0) * 2
                    for c in net_chans:
                        ch_t = Tensor(tile[c].astype(np.float32))
                        result[c, y0*2+off : y1*2+off, x0*2+off : x1*2+off] = \
                            model.upscale_channel(ch_t, noise_scale).numpy()[:out_h, :out_w]

            result = result[:, off:off+out_H, off:off+out_W]

            if yonly:
                for c in range(1, C):
                    result[c] = _bilinear_upsample_2x(tensor[c])

    end = time.perf_counter()
    timesum += end - start

    if ycgco and not model.is_3ch:
        result = _ycgco_to_rgb(result)
    return _to_pil(result, mode)


# ── main entry ────────────────────────────────────────────────────────────────

def process(args: argparse.Namespace):
    import model as model_module
    model_module.LEAKY_SLOPE = args.leaky_slope

    print(f"Loading model: {args.model}")
    net = load_model(args.model, args.wrap)
    print(f"Weights: {args.model}")

    if args.baseline:
        print(f"Baseline mode: {args.baseline} (neural network bypassed)")

    inp = Path(args.input)

    if inp.is_file():
        if inp.suffix.lower() not in _IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {inp.suffix}")
        paths      = [inp]
        out_dir    = Path(args.output).parent if args.output else inp.parent
        single_out = Path(args.output) if args.output else None
    elif inp.is_dir():
        paths      = sorted(p for p in inp.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        out_dir    = Path(args.output) if args.output else inp / "upscaled"
        single_out = None
        if not paths:
            raise FileNotFoundError(f"No images found in {inp}")
    else:
        raise FileNotFoundError(f"Input not found: {inp}")

    out_dir.mkdir(parents=True, exist_ok=True)

    Tensor.training = False
    Tensor.manual_seed(args.seed)
    np.random.seed(args.seed)

    for path in tqdm(paths, desc="Upscaling"):
        try:
            img = Image.open(path)
            result = upscale_image(net, img,
                                   tile_size=args.tile_size,
                                   wrap=args.wrap,
                                   ycgco=args.ycgco,
                                   yonly=args.yonly,
                                   baseline=args.baseline)

            if args.gridline_removal:
                result = remove_gridlines(result)

            ext = ".png"
            if args.extension:
                ext = args.extension
            if single_out:
                dest = single_out if single_out.suffix else single_out.with_suffix(ext)
            elif inp.is_file():
                dest = out_dir / (path.stem + args.suffix + ext)
            else:
                dest = out_dir / (path.stem + ext)

            if ext == ".dds":
                pixel_format = "DXT5" if len(result.getbands()) == 4 else "DXT1"
                ddslop.save_dds(result, dest, pixel_format=pixel_format, mipmaps=False)
            else:
                result.save(dest)
            print(f"  {path.name}  {img.size} → {result.size}  →  {dest.name}")

        except Exception as e:
            import traceback
            print(f"  ERROR {path.name}: {e}")
            traceback.print_exc()

    print(f"Elapsed for all inference: {timesum:.6f}s")
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="2× upscale image(s) with a trained safetensors model")
    p.add_argument("model",  help="Path to .safetensors weights file")
    p.add_argument("input",  help="Input image file or folder")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--extension", "-e", default=None)
    p.add_argument("--suffix", "-s", default="_2x",
                   help="Suffix appended to the stem when upscaling a single file (default: _2x)")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--wrap", action="store_true")
    p.add_argument("--ycgco", action="store_true")
    p.add_argument("--yonly", action="store_true")
    p.add_argument("--gridline-removal", action="store_true")
    p.add_argument("--baseline", choices=BASELINE_MODES, default=None,
                   help="Debug: use the classical upscaler directly with no neural network")
    p.add_argument("--leaky-slope", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    process(args)


if __name__ == "__main__":
    main()
