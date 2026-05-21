"""
train.py  –  train the 2× per-channel upscaler (tinygrad port)
"""

import random
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_state_dict, safe_save, safe_load

from model import (UpscaleNet, gaussian_blur,
                   manual_downscale2x, upscale_edi_2x, upsample2x, box_blur5x5_np)


# ── loss helpers ──────────────────────────────────────────────────────────────

def loss_l1(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()

def loss_mse(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()

def loss_lE(pred: Tensor, target: Tensor) -> Tensor:
    residual = pred - target
    abs_residual = residual.abs()
    z = Tensor.randn(*pred.shape)
    noisy_target = target + abs_residual * z
    return (pred - noisy_target).abs().mean()


# ── augmentation helpers ──────────────────────────────────────────────────────

def _orient(t: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    """Apply one of 8 isometric orientations to a (C, H, W) numpy array."""
    if flip:
        t = t[:, :, ::-1].copy()
    for _ in range(rot % 4):
        # rot90 CCW: transpose then flip rows
        t = np.transpose(t, (0, 2, 1))[:, ::-1, :].copy()
    return t


# ── dataset ───────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}


def _load_as_float(path: Path) -> np.ndarray:
    """Load image as float32 (C, H, W) numpy array in [0, 1]."""
    img = Image.open(path)
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


class UpscaleDataset:
    def __init__(
        self,
        folder: str,
        patch_size: int = 48,
        patches_per_pair: int = 200,
        n_aug: int = 8,
        brightness_range: float = 0.15,
        precompute_factor: int = 1,
        pad: int = 0,
        baseline_fn=None,
    ):
        self.patch_size       = patch_size
        self.patches_per_pair = patches_per_pair
        self.precompute_factor = precompute_factor
        self.pad              = pad
        self.baseline_fn = baseline_fn if baseline_fn is not None else upscale_edi_2x

        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._load(Path(folder), n_aug, brightness_range)

        if not self.pairs:
            raise RuntimeError("No valid images found.")
        if len(self.pairs) < self.precompute_factor:
            raise RuntimeError(f"Not enough pairs ({len(self.pairs)}) for precompute_factor={self.precompute_factor}.")

    def _load(self, folder: Path, n_aug: int, brightness_range: float):
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not paths:
            raise FileNotFoundError(f"No images in {folder}")
        for p in tqdm(paths, desc="Preprocessing images"):
            try:
                hr = _load_as_float(p)
                self._add_image(hr, n_aug, brightness_range)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
        random.shuffle(self.pairs)

    def _add_image(self, hr: np.ndarray, n_aug: int, brightness_range: float):
        blur_variants = [(hr, hr)]
        precomputed = {}
        for _, hr_for_lr in blur_variants:
            key = id(hr_for_lr)
            if key not in precomputed:
                lr = manual_downscale2x(hr_for_lr)
                # baseline_fn accepts numpy, returns numpy
                baseline_fn = self.baseline_fn
                if isinstance(lr, Tensor):
                    lr = lr.numpy()
                baseline_in = Tensor(lr)
                baseline_out = baseline_fn(baseline_in)
                if isinstance(baseline_out, Tensor):
                    baseline_out = baseline_out.numpy()
                precomputed[key] = (lr, baseline_out)

        total_aug = n_aug * self.precompute_factor
        for _ in range(total_aug):
            hr_out, hr_for_lr = random.choice(blur_variants)
            rot    = random.randint(0, 3)
            flip   = random.random() > 0.5
            brightness = random.uniform(0.5, 1.0) if random.random() > 0.5 else 1.0

            lr, baseline = precomputed[id(hr_for_lr)]

            a_out      = _orient(hr_out,   rot, flip)
            a_lr       = _orient(lr,       rot, flip)
            a_baseline = _orient(baseline, rot, flip)

            if brightness != 1.0:
                a_out      = a_out      * brightness
                a_lr       = a_lr       * brightness
                a_baseline = a_baseline * brightness

            residual = a_out - a_baseline
            self.pairs.append((a_lr.astype(np.float32), residual.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def get_patch(self, pair_idx: int):
        lr, residual = self.pairs[pair_idx % len(self.pairs)]
        C, lH, lW = lr.shape
        ph  = self.patch_size
        pad = self.pad

        if lH > ph + 2 * pad and lW > ph + 2 * pad:
            ly = random.randint(pad, lH - ph - pad - 1)
            lx = random.randint(pad, lW - ph - pad - 1)
            lr       = lr      [:, ly - pad : ly + ph + pad, lx - pad : lx + ph + pad]
            residual = residual[:, ly * 2   : (ly + ph) * 2, lx * 2   : (lx + ph) * 2]
        elif lH > ph and lW > ph:
            ly = random.randint(0, lH - ph - 1)
            lx = random.randint(0, lW - ph - 1)
            lr       = lr      [:, ly : ly + ph,       lx : lx + ph      ]
            residual = residual[:, ly * 2 : (ly+ph)*2, lx * 2 : (lx+ph)*2]

        c = random.randrange(C)
        return lr[c:c+1], residual[c:c+1]


# ── saving ────────────────────────────────────────────────────────────────────


def trysave(model, out_path):
    import time
    from safetensors.numpy import save_file as sf_save_numpy
    tensors_np = {k: v.numpy() for k, v in get_state_dict(model).items()}
    for attempt in range(3):
        try:
            sf_save_numpy(tensors_np, str(out_path))
            return
        except Exception as e:
            print(e)
            if attempt < 2:
                time.sleep(0.01)
    print("failed to save!")


# ── validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_dataset, device_str) -> float:
    ph  = val_dataset.patch_size
    pad = val_dataset.pad

    Tensor.training = False
    total = 0.0
    n = 0
    for lr, residual in val_dataset.pairs:
        C, lH, lW = lr.shape
        if lH > ph + 2 * pad and lW > ph + 2 * pad:
            ly = (lH - ph) // 2
            lx = (lW - ph) // 2
            lr_patch  = lr      [:, ly - pad : ly + ph + pad, lx - pad : lx + ph + pad]
            res_patch = residual[:, ly * 2   : (ly + ph) * 2, lx * 2   : (lx + ph) * 2]
        elif lH > ph and lW > ph:
            ly = (lH - ph) // 2
            lx = (lW - ph) // 2
            lr_patch  = lr      [:, ly : ly + ph,       lx : lx + ph      ]
            res_patch = residual[:, ly * 2 : (ly+ph)*2, lx * 2 : (lx+ph)*2]
        else:
            lr_patch  = lr
            res_patch = residual

        lr_in  = Tensor(lr_patch[:, np.newaxis])    # (C, 1, H, W)
        res_in = Tensor(res_patch[:, np.newaxis])

        pred = model(lr_in)
        pl = (loss_l1(pred, res_in) + loss_mse(pred, res_in)).numpy().item()
        total += pl
        n += 1

    return total / n if n > 0 else float("inf")


# ── gradient clipping ─────────────────────────────────────────────────────────

def clip_grad_norm_(params, max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_sq = sum(g.pow(2).sum() for g in grads)
    total_norm = total_sq.sqrt().numpy().item()
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale


# ── cosine LR ─────────────────────────────────────────────────────────────────

def cosine_lr(epoch: int, total_epochs: int, lr_max: float, lr_min: float) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))


# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import model as model_module
    model_module.LEAKY_SLOPE = args.leaky_slope

    net = UpscaleNet(False)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        import shutil
        import tempfile
        import os
        
        fd, temp_path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        shutil.copy(args.resume, temp_path)
        
        state = safe_load(temp_path)
        net.load_weights(state)
        
        try:
            os.remove(temp_path)
        except OSError:
            pass
            
        print(f"Resumed weights from {args.resume}")
    
    params = list(get_state_dict(net).values())
    total_params = sum(p.numpy().size for p in params)
    print(f"Parameters: {total_params:,}")

    pad = sum((conv._k - 1) // 2
              for conv in [*net.layers, net._final_layer]
              if conv._k > 1)
    print(f"Receptive field pad: {pad} LR px")
    if pad > 0:
        net.set_padding_enabled(False)

    trysave(net, out_path)

    if net.is_raw:
        def baseline_fn(t):
            arr = t.numpy() if isinstance(t, Tensor) else t
            C, H, W = arr.shape
            return np.zeros((C, H * 2, W * 2), dtype=np.float32)
    elif net.is_blurbilinear:
        def baseline_fn(t):
            arr = t.numpy() if isinstance(t, Tensor) else t
            blurred = box_blur5x5_np(arr[np.newaxis], is_wrapping=False)[0]
            return upsample2x(blurred, is_wrapping=False)
    elif net.is_bilinear:
        baseline_fn = upsample2x
    else:
        baseline_fn = upscale_edi_2x
    dataset = UpscaleDataset(
        args.data,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
        n_aug=args.n_aug,
        brightness_range=args.brightness_range,
        precompute_factor=args.precompute_factor,
        pad=pad,
        baseline_fn=baseline_fn,
    )

    print(f"Dataset: {len(dataset.pairs)} precomputed pairs")

    val_data_folder = args.val_data if args.val_data else args.data
    val_dataset = UpscaleDataset(
        val_data_folder,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
        n_aug=args.n_aug,
        brightness_range=args.brightness_range,
        precompute_factor=1,
        pad=pad,
        baseline_fn=baseline_fn,
    )
    print(f"Validation set: {len(val_dataset.pairs)} pairs")

    opt = AdamW(params, lr=args.lr, b1=0.9, b2=0.999, weight_decay=0.0)
    best_loss = float("inf")

    use_basic_loss = args.basic_loss
    @TinyJit
    def train_step(lr_batch: Tensor, res_batch: Tensor) -> Tensor:
        opt.zero_grad()
        pred = net(lr_batch)
        if use_basic_loss:
            loss = loss_l1(pred, res_batch) + loss_mse(pred, res_batch)
        else:
            loss = loss_lE(pred, res_batch)
        loss.backward()
        clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        for p in params:
            p.assign(p.clamp(-3.99, 3.99))
        return loss.realize()

    chunk_size = max(1, len(dataset.pairs) // args.precompute_factor)

    for epoch in range(start_epoch, args.epochs + 1):
        new_lr = cosine_lr(epoch - 1, args.epochs, args.lr, args.lr * 0.1)
        opt.lr = Tensor([new_lr])

        Tensor.training = True

        chunk_idx = (epoch - 1) % args.precompute_factor
        start_p = chunk_idx * chunk_size
        end_p   = min(start_p + chunk_size, len(dataset.pairs))
        active_pair_indices = list(range(start_p, end_p))

        indices = active_pair_indices * args.patches_per_pair
        random.shuffle(indices)

        running_loss = 0.0
        steps = 0

        pbar = tqdm(range(0, len(indices) - args.batch_size + 1, args.batch_size),
                    desc=f"Epoch {epoch:4d}/{args.epochs}", leave=False)

        for batch_start in pbar:
            batch_idx = indices[batch_start : batch_start + args.batch_size]

            lr_list, res_list = [], []
            for pair_i in batch_idx:
                lr_p, res_p = dataset.get_patch(pair_i)
                lr_list.append(lr_p)
                res_list.append(res_p)

            lr_batch  = Tensor(np.stack(lr_list,  axis=0).astype(np.float32))
            res_batch = Tensor(np.stack(res_list, axis=0).astype(np.float32))

            loss = train_step(lr_batch, res_batch)
            running_loss += loss.numpy().item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)

        if args.strict_validation:
            val_loss = run_validation(net, val_dataset, "cpu")
            print(f"Epoch {epoch:4d} | train {avg_loss:.6f} | val {val_loss:.6f} | lr {new_lr:.2e}")
            if val_loss < best_loss:
                best_loss = val_loss
                trysave(net, out_path)
                print(f"  saved {out_path}  (best val loss {best_loss:.6f})")
        else:
            print(f"Epoch {epoch:4d} | loss {avg_loss:.6f} | lr {new_lr:.2e}")
            trysave(net, out_path)

    print("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    p = argparse.ArgumentParser(description="Train 2× per-channel image upscaler")
    p.add_argument("data",           help="Folder of high-resolution training images")
    p.add_argument("--out",          default="upscaler.safetensors")
    p.add_argument("--resume",       default=None)
    p.add_argument("--epochs",            type=int,   default=700)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--patch-size",        type=int,   default=64)
    p.add_argument("--n-aug",             type=int,   default=4)
    p.add_argument("--precompute-factor", type=int,   default=10)
    p.add_argument("--patches-per-pair",  type=int,   default=20)
    p.add_argument("--brightness-range",  type=float, default=0.15)
    p.add_argument("--lr",                type=float, default=2e-3)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--leaky-slope",       type=float, default=0.003)
    p.add_argument("--basic-loss",        action="store_true")
    p.add_argument("--strict-validation", action="store_true")
    p.add_argument("--val-data",          default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
