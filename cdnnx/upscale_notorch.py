#!/usr/bin/env python3
"""
upscale.py -- run a CuNNy safetensors model on an image file.

Usage:
    python upscale.py <image> [-m MODEL] [-o OUTPUT]

Examples:
    python upscale.py photo.png
        -> saves photo_2x.png next to photo.png using the bundled 4x32 model

    python upscale.py photo.png -m pretrained/fast-NVL.safetensors
        -> same but with the faster, lighter model

    python upscale.py photo.png -o /tmp/out.png
        -> write to an explicit path

Notes:
    rgb=True  models process all three channels together natively.
    rgb=False models are luma-only; this script runs them once per RGB
    channel and recombines, so you always get a colour output.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(path: str) -> dict:
    """Return a dict of {name: np.ndarray} tensors + decoded JSON metadata."""
    m = dict(load_file(path))
    with safe_open(path, framework="np") as f:
        raw_meta = f.metadata()
    if raw_meta:
        for k, v in raw_meta.items():
            m[k] = json.loads(v)
    return m


# ---------------------------------------------------------------------------
# YCgCo colour-space helpers  (only used for rgb=False models)
# Operate on (3, H, W) float32 numpy arrays in [0, 1].
# ---------------------------------------------------------------------------

def _rgb_to_ycgco(t: np.ndarray) -> np.ndarray:
    """(3, H, W) RGB → YCgCo.  Chroma is offset by 0.5 to stay in [0, 1]."""
    r, g, b = t[0], t[1], t[2]
    co  =  0.5  * r - 0.5  * b
    tmp = -0.25 * r + 0.5  * g - 0.25 * b
    y   =  0.25 * r + 0.5  * g + 0.25 * b
    cg  = tmp
    return np.stack([y, cg + 0.5, co + 0.5], axis=0)


def _ycgco_to_rgb(t: np.ndarray) -> np.ndarray:
    """(3, H, W) YCgCo → RGB.  Assumes chroma channels are offset by 0.5."""
    y  = t[0]
    cg = t[1] - 0.5
    co = t[2] - 0.5
    tmp = y - cg
    r = np.clip(tmp + co, 0.0, 1.0)
    g = np.clip(y   + cg, 0.0, 1.0)
    b = np.clip(tmp - co, 0.0, 1.0)
    return np.stack([r, g, b], axis=0)


# ---------------------------------------------------------------------------
# Pure-numpy inference helpers
# ---------------------------------------------------------------------------

def conv2d_same(x, w, b):
    """
    2-D convolution with padding='same', no stride.

    x : (C_in,  H, W)  float32
    w : (C_out, C_in, kH, kW)  float32
    b : (C_out,)  float32  or  None
    -> (C_out, H, W)
    """
    from scipy.signal import correlate2d

    C_out, C_in, kH, kW = w.shape
    H, W = x.shape[1], x.shape[2]
    out = np.zeros((C_out, H, W), dtype=np.float32)

    for oc in range(C_out):
        acc = np.zeros((H, W), dtype=np.float32)
        for ic in range(C_in):
            acc += correlate2d(x[ic], w[oc, ic], mode='same')
        out[oc] = acc

    if b is not None:
        out += b[:, None, None]
    return out


def pixel_shuffle(x, r):
    """
    PyTorch-compatible pixel_shuffle.
    x : (C*r*r, H, W)  ->  (C, H*r, W*r)
    """
    C_r2, H, W = x.shape
    C = C_r2 // (r * r)
    x = x.reshape(C, r, r, H, W)
    x = x.transpose(0, 3, 1, 4, 2)   # (C, H, r, W, r)
    return x.reshape(C, H * r, W * r)


def bilinear_2x(x):
    """2x bilinear upscale.  x : (C, H, W) -> (C, 2H, 2W)"""
    C, H, W = x.shape
    imgs = [Image.fromarray(
                (np.clip(x[c], 0, 1) * 255).astype(np.uint8)
            ).resize((W * 2, H * 2), Image.BILINEAR)
            for c in range(C)]
    return np.stack([np.array(i).astype(np.float32) / 255.0 for i in imgs])


def act(x, quant):
    """Activation: clamped leaky-relu (quant) or plain relu."""
    if quant:
        a = 0.01
        return np.clip(x, 0.0, 1.0) + a * np.maximum(x - 1.0, 0.0)
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_model(m, img_np):
    """
    Run one forward pass.

    img_np : (C_in, H, W) float32 in [0, 1]
             C_in=1 for luma models, C_in=3 for rgb models
    returns : (C_in, 2H, 2W) float32 in [0, 1]
    """
    quant = m['quant']

    conv_keys = sorted(
        [k for k in m if k.startswith('conv.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[1])
    )

    x = img_np.copy()
    x = act(conv2d_same(x, m['cin.weight'], m['cin.bias']), quant)
    for ck in conv_keys:
        x = act(conv2d_same(x, m[ck], None), quant)
    x = conv2d_same(x, m['cout.weight'], m['cout.bias'])
    x = pixel_shuffle(x, 2)
    x = x + bilinear_2x(img_np)
    return np.clip(x, 0.0, 1.0)


def upscale(m, tensor, ycgco=False, yonly=False):
    """
    Upscale a (3, H, W) RGB tensor with any CuNNy model.

    rgb=True  models: feed all 3 channels together natively.
              ycgco / yonly are ignored for rgb=True models.
    rgb=False models: run each channel independently and recombine.
      ycgco  – convert to YCgCo before running each channel through the
               network, then convert back.  Decorrelates luma from chroma.
      yonly  – run the network only on the Y (luma) channel; upsample the
               two chroma channels with bilinear interpolation.  Faster and
               avoids network-induced chroma ringing.  Implies ycgco.
    """
    if m['rgb']:
        return run_model(m, tensor)

    # ycgco / yonly only apply to the per-channel (rgb=False) path
    if yonly:
        ycgco = True
    if ycgco:
        tensor = _rgb_to_ycgco(tensor)

    if yonly:
        y_up  = run_model(m, tensor[0:1])
        cg_up = bilinear_2x(tensor[1:2])
        co_up = bilinear_2x(tensor[2:3])
        result = np.concatenate([y_up, cg_up, co_up], axis=0)
    else:
        channels = [run_model(m, tensor[c:c+1]) for c in range(3)]
        result = np.concatenate(channels, axis=0)   # (3, 2H, 2W)

    if ycgco:
        result = _ycgco_to_rgb(result)
    return result


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def image_to_tensor(path):
    """Load image -> (3, H, W) float32 in [0, 1], always RGB."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
    return arr.transpose(2, 0, 1)                     # (3, H, W)


def tensor_to_image(t):
    """(3, H, W) float32 [0,1] -> RGB PIL Image."""
    arr = (np.clip(t, 0.0, 1.0).transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_model_path():
    here = Path(__file__).parent
    for name in ('4x32-NVL', '4x24-NVL', '4x16-NVL', '4x12-NVL',
                 '3x12-NVL', 'fast-NVL', 'faster-NVL', 'veryfast-NVL'):
        p = here / 'pretrained' / f'{name}.safetensors'
        if p.exists():
            return p
    candidates = list((here / 'pretrained').glob('*.safetensors'))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        'No model found. Pass -m path/to/model.safetensors explicitly.')


def main():
    parser = argparse.ArgumentParser(
        description='Upscale an image 2x with a CuNNy neural network.')
    parser.add_argument('image', type=Path,
                        help='Input image file (PNG, JPEG, ...)')
    parser.add_argument('-m', '--model', type=Path, default=None,
                        help='Path to .safetensors model '
                             '(default: best available in pretrained/)')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output path (default: <stem>_2xc.png next to input)')
    parser.add_argument('--ycgco', action='store_true',
                        help='(rgb=False models only) Convert RGB to YCgCo before '
                             'upscaling and back afterwards. Decorrelates luma from '
                             'chroma so each channel is processed independently.')
    parser.add_argument('--yonly', action='store_true',
                        help='(rgb=False models only) Run the network on the Y (luma) '
                             'channel only; upsample chroma with bilinear interpolation. '
                             'Faster and avoids chroma ringing. Implies --ycgco.')
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f'Error: image not found: {args.image}')

    model_path = args.model or default_model_path()
    if not model_path.exists():
        sys.exit(f'Error: model not found: {model_path}')

    out_path = args.output or args.image.parent / (args.image.stem + '_2xc.png')

    print(f'Model : {model_path.name}')
    print(f'Input : {args.image}')
    print(f'Output: {out_path}')

    print('Loading model...')
    m = load_model(str(model_path))

    print('Loading image... ', end='', flush=True)
    tensor = image_to_tensor(args.image)
    _, H, W = tensor.shape
    print(f'{W}x{H}  ->  {W*2}x{H*2}')

    print('Running inference...')
    
    import time
    timesum = 0.0
    start = time.perf_counter()
    result = upscale(m, tensor, ycgco=args.ycgco, yonly=args.yonly)
    end = time.perf_counter()
    timesum += end-start
    print(f"Elapsed for all inference: {timesum:.6f}s")
    
    tensor_to_image(result).save(out_path)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
