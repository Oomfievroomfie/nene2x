#!/usr/bin/env python3
"""
upscale.py -- run a CuNNy safetensors model on an image file.

Usage:
    python upscale.py <image> [-m MODEL] [-o OUTPUT] [--cpu]

Examples:
    python upscale.py photo.png
    python upscale.py photo.png -m pretrained/fast-NVL.safetensors
    python upscale.py photo.png -o /tmp/out.png
    python upscale.py photo.png --cpu

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
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.numpy import load_file
from safetensors import safe_open


# ---------------------------------------------------------------------------
# YCgCo colour-space helpers  (only used for rgb=False models)
# Operate on (1, 3, H, W) float32 tensors in [0, 1].
# ---------------------------------------------------------------------------

def _rgb_to_ycgco(t: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) RGB → YCgCo.  Chroma is offset by 0.5 to stay in [0, 1]."""
    r, g, b = t[:, 0], t[:, 1], t[:, 2]
    co  =  0.5  * r - 0.5  * b
    tmp = -0.25 * r + 0.5  * g - 0.25 * b
    y   =  0.25 * r + 0.5  * g + 0.25 * b
    cg  = tmp
    return torch.stack([y, cg + 0.5, co + 0.5], dim=1)


def _ycgco_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """(1, 3, H, W) YCgCo → RGB.  Assumes chroma channels are offset by 0.5."""
    y  = t[:, 0]
    cg = t[:, 1] - 0.5
    co = t[:, 2] - 0.5
    tmp = y - cg
    r = (tmp + co).clamp(0, 1)
    g = (y   + cg).clamp(0, 1)
    b = (tmp - co).clamp(0, 1)
    return torch.stack([r, g, b], dim=1)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(path: str) -> dict:
    """Return tensors as np.ndarray plus decoded JSON metadata."""
    m = dict(load_file(path))
    with safe_open(path, framework="np") as f:
        raw_meta = f.metadata()
    if raw_meta:
        for k, v in raw_meta.items():
            m[k] = json.loads(v)
    return m


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(m: dict, img: torch.Tensor) -> torch.Tensor:
    """
    One forward pass.

    img : (1, C_in, H, W) float32  (C_in=1 luma, C_in=3 rgb)
    returns : (1, C_in, 2H, 2W) float32, clamped to [0, 1]
    """
    quant = m['quant']

    def w(key):
        return torch.from_numpy(m[key]).to(img.device)

    def act(x):
        if quant:
            return torch.clamp(x, 0., 1.) + 0.01 * F.relu(x - 1.)
        return F.relu(x)

    conv_keys = sorted(
        [k for k in m if k.startswith('conv.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[1])
    )

    x = act(F.conv2d(img, w('cin.weight'), w('cin.bias'), padding=1))
    for ck in conv_keys:
        x = act(F.conv2d(x, w(ck), padding=1))
    x = F.conv2d(x, w('cout.weight'), w('cout.bias'), padding=1)
    x = F.pixel_shuffle(x, 2)
    x = x + F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
    return x.clamp(0., 1.)


def upscale(m: dict, tensor: torch.Tensor,
            ycgco: bool = False, yonly: bool = False) -> torch.Tensor:
    """
    Upscale a (1, 3, H, W) RGB tensor with any CuNNy model.

    rgb=True  models: feed all 3 channels together natively.
              ycgco / yonly are ignored for rgb=True models.
    rgb=False models: run each channel independently and recombine.
      ycgco  – convert to YCgCo before running each channel through the
               network, then convert back.  Decorrelates luma from chroma.
      yonly  – run the network only on the Y (luma) channel; upsample the
               two chroma channels with corner-aligned bilinear interpolation.
               Faster and avoids network-induced chroma ringing.
               Implies ycgco automatically.
    """
    if m['rgb']:
        return run_model(m, tensor)

    # ycgco / yonly only apply to the per-channel (rgb=False) path
    if yonly:
        ycgco = True
    if ycgco:
        tensor = _rgb_to_ycgco(tensor)

    if yonly:
        y_up  = run_model(m, tensor[:, 0:1])
        cg_up = F.interpolate(tensor[:, 1:2], scale_factor=2,
                               mode='bilinear', align_corners=True)
        co_up = F.interpolate(tensor[:, 2:3], scale_factor=2,
                               mode='bilinear', align_corners=True)
        result = torch.cat([y_up, cg_up, co_up], dim=1)
    else:
        channels = [run_model(m, tensor[:, c:c+1]) for c in range(3)]
        result = torch.cat(channels, dim=1)

    if ycgco:
        result = _ycgco_to_rgb(result)
    return result


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def image_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image -> (1, 3, H, W) float32 in [0, 1] on device."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0      # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t.to(device)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) float32 [0,1] -> RGB PIL Image."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_model_path() -> Path:
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
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    parser.add_argument('--ycgco', action='store_true',
                        help='(rgb=False models only) Convert RGB to YCgCo before '
                             'upscaling and back afterwards. Decorrelates luma from '
                             'chroma so each channel is processed independently.')
    parser.add_argument('--yonly', action='store_true',
                        help='(rgb=False models only) Run the network on the Y (luma) '
                             'channel only; upsample chroma with corner-aligned bilinear '
                             'interpolation. Faster and avoids chroma ringing. '
                             'Implies --ycgco automatically.')
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f'Error: image not found: {args.image}')

    model_path = args.model or default_model_path()
    if not model_path.exists():
        sys.exit(f'Error: model not found: {model_path}')

    out_path = args.output or args.image.parent / (args.image.stem + '_2xc.png')

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    print(f'Model  : {model_path.name}')
    print(f'Device : {device}')
    print(f'Input  : {args.image}')
    print(f'Output : {out_path}')

    print('Loading model...')
    m = load_model(str(model_path))

    print('Loading image... ', end='', flush=True)
    tensor = image_to_tensor(args.image, device)
    _, _, H, W = tensor.shape
    print(f'{W}x{H}  ->  {W*2}x{H*2}')

    print('Running inference...')
    with torch.no_grad():
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
