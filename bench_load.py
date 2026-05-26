import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from model import (UpscaleNet, upscale_edi_2x, upscale_edi_2x_np,
                   upsample2x, upsample2x_np, manual_downscale2x, box_blur5x5_np)
from train import _sinc_downscale2x, SINC_DOWNSCALE_LOBES
from tinygrad import Tensor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

if not paths:
    print("No images found.")
    sys.exit(1)

net = UpscaleNet(False)
if net.is_raw:
    def baseline_fn(lr):
        C, H, W = lr.shape
        return np.zeros((C, H * 2, W * 2), dtype=np.float32)
    baseline_name = "zeros (raw)"
elif net.is_blurbilinear:
    def baseline_fn(lr):
        blurred = box_blur5x5_np(lr[np.newaxis], is_wrapping=False)[0]
        return upsample2x_np(blurred, is_wrapping=False)
    baseline_name = "box-blur + bilinear"
elif net.is_bilinear:
    def baseline_fn(lr):
        return upsample2x_np(lr, is_wrapping=False)
    baseline_name = "bilinear"
else:
    def baseline_fn(lr):
        return upscale_edi_2x_np(lr, is_wrapping=False)
    baseline_name = "EDI"

print(f"baseline: {baseline_name}")

def load_one(p):
    img = Image.open(p)
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

n = len(paths)
t_load = t_down = t_sinc = t_base = 0.0

for p in paths:
    t0 = time.perf_counter()
    hr = load_one(p)
    t1 = time.perf_counter()
    lr = manual_downscale2x(hr)
    if isinstance(lr, Tensor):
        lr = lr.numpy()
    t2 = time.perf_counter()
    if SINC_DOWNSCALE_LOBES > 0:
        _sinc_downscale2x(hr, SINC_DOWNSCALE_LOBES)
    t3 = time.perf_counter()
    baseline_fn(lr)
    t4 = time.perf_counter()
    t_load += t1 - t0
    t_down += t2 - t1
    t_sinc += t3 - t2
    t_base += t4 - t3

total = t_load + t_down + t_sinc + t_base
print(f"{n} images  total {total:.3f}s  ({total/n*1000:.1f} ms/image)")
print(f"  load:      {t_load:.3f}s  ({t_load/n*1000:.1f} ms/image)")
print(f"  downscale: {t_down:.3f}s  ({t_down/n*1000:.1f} ms/image)")
print(f"  sinc:      {t_sinc:.3f}s  ({t_sinc/n*1000:.1f} ms/image)" + (" (disabled)" if SINC_DOWNSCALE_LOBES == 0 else ""))
print(f"  baseline:  {t_base:.3f}s  ({t_base/n*1000:.1f} ms/image)")
