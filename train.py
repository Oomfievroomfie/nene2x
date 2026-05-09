"""
train.py  –  train the 2× per-channel upscaler

Usage
-----
python train.py  /path/to/hires/images  --out upscaler.safetensors  [options]

The script expects a folder of "ground-truth" high-resolution images.
It manufactures paired (LR, HR) training data automatically.

Key design decisions
--------------------
• Per-channel training: each (LR patch, residual patch) is a single scalar channel,
  so the same weights generalise across R / G / B / A.

• Three blur variants per image (augments the training distribution):
    1. Original HR  → downscale → LR,          target = original HR
    2. Blurred HR   → downscale → LR,          target = blurred HR
    3. Original HR as target, but  blur(HR) → downscale → LR
       (simulates a slightly soft acquisition pipeline)

• Combinatorial-but-not-exhaustive augmentation: each training sample
  independently draws a random orientation (8 options) and optionally a random
  global brightness offset.  We don't enumerate all 3 × 8 × N combinations.

• Fully-convolutional training on random patches: expressing the FC layers as
  1×1 convolutions means the entire image can be processed in one pass, and
  random patch crops let us build large batches from small images while also
  providing implicit regularisation via varied spatial contexts.

• Mixed-precision (AMP) + gradient clipping + cosine-annealing LR.
"""

import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from model import (UpscaleNet, gaussian_blur,
                   manual_downscale2x, upscale_edi_2x, upsample2x)


# ── loss helpers ──────────────────────────────────────────────────────────────

def loss_gradient(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on first-order spatial gradients (x and y).

    This directly penalises errors in edge structure rather than just pixel
    values. It is the key fix that breaks the 'predict near-zero residual'
    degeneracy that L1/MSE alone falls into: both of those losses push
    toward the conditional mean of the residual, which across diverse patches
    is very close to zero.  The gradient loss forces the network to get the
    *direction* of high-frequency content right.
    """
    # Compute difference tensor once so grad of (pred-target) = grad(pred)-grad(target)
    d = pred - target
    loss_x = d[:, :, :, 1:] - d[:, :, :, :-1]
    loss_y = d[:, :, 1:, :] - d[:, :, :-1, :]
    return loss_x.abs().mean() + loss_y.abs().mean()

_z_buffer = None

def loss_lE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Implements the ℓ_E loss from "Revisiting ℓ1 Loss in Super-Resolution" (He & Cheng, 2022).
    
    Args:
        pred: Network output (μ), shape (B, C, H, W)
        target: Ground truth (ŷ), same shape
    
    Returns:
        Scalar loss tensor.
    """
    global _z_buffer
    # Residual between prediction and target
    residual = pred - target  # shape (B, C, H, W)
    
    # Absolute residual – used as data‑adaptive noise scale
    abs_residual = torch.abs(residual)  # |ŷ - μ|
    
    # Sample Gaussian noise of the same shape
    z = torch.randn_like(pred)  # z ~ N(0,1)
    
    # Construct noisy target: ŷ + |ŷ-μ| * z
    noisy_target = target + abs_residual * z
    
    # L1 loss between prediction and noisy target
    loss = F.l1_loss(pred, noisy_target, reduction='mean')
    
    return loss

# ── augmentation helpers ──────────────────────────────────────────────────────

def _orient(t: torch.Tensor, rot: int, flip: bool) -> torch.Tensor:
    """Apply one of 8 isometric orientations to a (C, H, W) tensor."""
    if flip:
        t = t.flip(-1)
    if rot:
        t = t.rot90(rot, dims=[-2, -1])
    return t


# ── dataset ───────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".dds"}


def _load_as_float(path: Path) -> torch.Tensor:
    """
    Load an image as a float32 (C, H, W) tensor in [0, 1].
    Handles L / RGB / RGBA / P modes.  Crops to even spatial dimensions.
    """
    img = Image.open(path)

    # Normalise palette / exotic modes
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode not in ("L", "RGB", "RGBA", "LA"):
        img = img.convert("RGB")

    arr = np.asarray(img)

    # Handle 16-bit images
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 65535.0)
    else:
        arr = arr.astype(np.float32) / 255.0

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    H, W, C = arr.shape
    H -= H % 2          # crop to even dimensions – must be the very first step
    W -= W % 2
    arr = arr[:H, :W]

    return torch.from_numpy(arr.copy()).permute(2, 0, 1)  # (C, H, W)


class UpscaleDataset(Dataset):
    """
    All augmentation is done upfront at load time and stored as precomputed
    (lr, residual) full-image tensor pairs.  __getitem__ does nothing except
    pull a random spatial patch from a random stored pair — essentially free.

    Stored pairs come from:
      • 3 blur variants per image
      • n_aug * precompute_factor combined (orientation × brightness) augmentations per image
    Total = n_images × n_aug × precompute_factor  pairs in CPU RAM.

    The "sparse patch sampling" approach is still the key training optimisation:
    each __getitem__ returns a different random crop, so the network sees many
    distinct spatial contexts per pair without redundant full-image processing.
    """

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
        self.patch_size        = patch_size
        self.patches_per_pair  = patches_per_pair
        self.precompute_factor = precompute_factor
        self.pad               = pad
        # baseline_fn(lr_tensor) -> 2x upscaled tensor used as the residual base.
        # Defaults to EDI (upscale_edi_2x); pass upsample2x for bilinear models.
        self.baseline_fn = baseline_fn if baseline_fn is not None else upscale_edi_2x

        # Precomputed storage: list of (lr, residual), both (C, H/2, W/2) / (C, H, W)
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._load(Path(folder), n_aug, brightness_range)

        if not self.pairs:
            raise RuntimeError("No valid images found – check your data folder.")
        if len(self.pairs) < self.precompute_factor:
            raise RuntimeError(f"Not enough pairs ({len(self.pairs)}) for precompute_factor={self.precompute_factor}. Increase data or decrease precompute_factor.")

    def _load(self, folder: Path, n_aug: int, brightness_range: float):
        paths = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in _IMAGE_EXTS)
        if not paths:
            raise FileNotFoundError(f"No images in {folder}")

        for p in tqdm(paths, desc="Preprocessing images"):
            try:
                hr = _load_as_float(p)
                self._add_image(hr, n_aug, brightness_range, self.baseline_fn)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
        
        random.shuffle(self.pairs) 

    def _add_image(self, hr: torch.Tensor, n_aug: int, brightness_range: float, baseline_fn=None):
        if baseline_fn is None:
            baseline_fn = self.baseline_fn
        hr_blur = gaussian_blur(hr)

        # The three blur variants are options, not an outer loop.
        # Each augmented sample randomly picks one.
        # EDIT: dummied out. made the upscale quality worse instead of better.
        blur_variants = [
            (hr, hr),  # variant 1: clean LR → sharp target
        ]

        # Downscale and upscale once per image per variant, not per augmentation.
        # Rotation/flip commute with both ops; brightness offset is applied afterward.
        precomputed = {
            id(hr_for_lr): (manual_downscale2x(hr_for_lr), baseline_fn(manual_downscale2x(hr_for_lr)))
            for _, hr_for_lr in blur_variants
        }

        total_aug = n_aug * self.precompute_factor
        for _ in range(total_aug):
            hr_out, hr_for_lr = random.choice(blur_variants)
            rot    = random.randint(0, 3)
            flip   = random.random() > 0.5
            #offset = (random.uniform(-brightness_range, brightness_range)
            #          if random.random() > 0.5 else 0.0)
            
            brightness = (random.uniform(0.5, 1.0)
                      if random.random() > 0.5 else 1.0)

            lr, baseline = precomputed[id(hr_for_lr)]

            a_out     = _orient(hr_out,  rot, flip)
            a_lr      = _orient(lr,      rot, flip)
            a_baseline= _orient(baseline,rot, flip)

            #if offset != 0.0:
            #    a_out      = (a_out      + offset).clamp(0.0, 1.0)
            #    a_lr       = (a_lr       + offset).clamp(0.0, 1.0)
            #    a_baseline = (a_baseline + offset).clamp(0.0, 1.0)
            if brightness != 1.0:
                a_out      = a_out      * brightness
                a_lr       = a_lr       * brightness
                a_baseline = a_baseline * brightness

            residual = a_out - a_baseline
            self.pairs.append((a_lr, residual))

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    def __getitem__(self, idx: int):
        # idx maps perfectly to a pair and allows chunked samplers to pick precise pairs
        lr, residual = self.pairs[idx % len(self.pairs)]

        C, lH, lW = lr.shape
        ph  = self.patch_size
        pad = self.pad

        # Random spatial patch (the only work done at training time).
        # When pad > 0, the LR patch is (ph + 2*pad) on each side so the network
        # sees only real context pixels and needs no padding at all.  The residual
        # target stays ph-sized; the training loop crops the prediction to match.
        if lH > ph + 2 * pad and lW > ph + 2 * pad:
            ly = random.randint(pad, lH - ph - pad - 1)
            lx = random.randint(pad, lW - ph - pad - 1)
            lr       = lr      [:, ly - pad : ly + ph + pad, lx - pad : lx + ph + pad]
            residual = residual[:, ly * 2   : (ly + ph) * 2, lx * 2   : (lx + ph) * 2]
        elif lH > ph and lW > ph:
            ly = random.randint(0, lH - ph - 1)
            lx = random.randint(0, lW - ph - 1)
            lr       = lr      [:, ly : ly + ph,        lx : lx + ph      ]
            residual = residual[:, ly * 2 : (ly+ph)*2,  lx * 2 : (lx+ph)*2]

        # Random single channel
        c = random.randrange(C)
        return lr[c : c + 1], residual[c : c + 1]   # (1, ph+2*pad, pw+2*pad), (1, 2ph, 2pw)


class ChunkedRandomSampler(Sampler):
    """
    Yields indices for 1/X of the precomputed pairs per epoch.
    Ensures that each epoch sees a disjoint subset of pairs, rotating every X epochs.
    """
    def __init__(self, total_pairs: int, patches_per_pair: int, precompute_factor: int):
        self.total_pairs = total_pairs
        self.patches_per_pair = patches_per_pair
        self.precompute_factor = precompute_factor
        self.chunk_size = total_pairs // precompute_factor
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        chunk_idx = self.epoch % self.precompute_factor
        start_pair = chunk_idx * self.chunk_size
        
        if chunk_idx == self.precompute_factor - 1:
            end_pair = self.total_pairs
        else:
            end_pair = start_pair + self.chunk_size

        # Generate indices ensuring exactly `patches_per_pair` access per active pair
        p = torch.arange(start_pair, end_pair)
        k = torch.arange(self.patches_per_pair) * self.total_pairs
        indices = (p.unsqueeze(0) + k.unsqueeze(1)).view(-1)
        indices = indices[torch.randperm(len(indices))].tolist()
        return iter(indices)

    def __len__(self):
        chunk_idx = self.epoch % self.precompute_factor
        if chunk_idx == self.precompute_factor - 1:
            return (self.total_pairs - chunk_idx * self.chunk_size) * self.patches_per_pair
        return self.chunk_size * self.patches_per_pair


def trysave(model, out_path):
        try:
            save_file(model.state_dict(), str(out_path))
        except:
            import time
            time.sleep(0.005)
            try:
                save_file(model.state_dict(), str(out_path))
            except:
                print("failed to save!")
                pass


def run_validation(model, val_dataset, device) -> float:
    loss_l1  = nn.L1Loss()
    loss_mse = nn.MSELoss()

    ph  = val_dataset.patch_size
    pad = val_dataset.pad

    model.eval()
    total = torch.tensor(0.0, device=device)  # accumulate on GPU, no per-step sync
    n = 0
    with torch.no_grad():
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

            # Stack all channels into a single batch — one forward pass per pair
            lr_in  = lr_patch.unsqueeze(1).to(device)   # (C, 1, ph+2*pad, pw+2*pad)
            res_in = res_patch.unsqueeze(1).to(device)  # (C, 1, 2ph, 2pw)

            pred = model(lr_in)
            # padding consumed by valid convs — output is already (C, 1, 2ph, 2pw)

            #pixel_loss    = loss_mse(pred, res_in)
            pixel_loss    = loss_l1(pred, res_in) + loss_mse(pred, res_in)
            total += pixel_loss   # stays on GPU
            n += 1

    return (total / n).item() if n > 0 else float("inf")  # single GPU→CPU sync

# ── training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    # Device selection
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
        
    
    print(f"Device: {device}")

    # Mixed precision (CUDA only – MPS AMP still maturing)
    use_amp = device.type == "cuda"
    try:
        scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    except:
        scaler  = torch.cuda.amp.GradScaler("cuda", enabled=use_amp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Model
    model = UpscaleNet(False)

    # Optionally resume from checkpoint
    start_epoch = 1
    best_loss   = float("inf")
    if args.resume and Path(args.resume).exists():
        
        #state = load_file(args.resume)
        #model.load_state_dict(state)
        
        import safetensors
        with open(args.resume, 'rb') as f:
            buffer = f.read()
            state = safetensors.torch.load(buffer)
            model.load_state_dict(state)
        
        print(f"Resumed weights from {args.resume}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    model = model.to(device)

    # Total LR-space receptive field padding — used to pre-pad training patches
    # so the network never has to pad during the forward pass.
    pad = sum((conv.kernel_size[0] - 1) // 2
              for conv in [*model.layers, model._final_layer]
              if conv.kernel_size[0] > 1)
    print(f"Receptive field pad: {pad} LR px")
    if pad > 0:
        model.set_padding_enabled(False)
    
    try:
        save_file(model.state_dict(), str(out_path))
    except:
        pass
    
    
    # Dataset / loader
    baseline_fn = upsample2x if model.is_bilinear else upscale_edi_2x
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

    sampler = ChunkedRandomSampler(
        total_pairs=len(dataset.pairs),
        patches_per_pair=args.patches_per_pair,
        precompute_factor=args.precompute_factor,
    )

    print(
        f"Dataset: {len(dataset.pairs)} precomputed pairs "
        f"({args.n_aug * args.precompute_factor} augmentations/image, blur variant sampled randomly)"
    )
    print(
        f"Epochs process {len(sampler):,} patch samples "
        + (f"(using 1/{args.precompute_factor} of precomputed pairs per epoch)" if args.precompute_factor > 1 else "(all pairs per epoch)")
    )

    from torch.utils.data.dataloader import default_collate
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type != "cpu"),
        persistent_workers=False,
        drop_last=True,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    )

    # Validation dataset — created once, reused every epoch.
    # precompute_factor=1 means all pairs are used each time, which naturally
    # produces ~the same number of patches as one training epoch (the training
    # epoch also sees n_images × n_aug pairs, just drawn from a larger pool).
    # Fresh random augmentation samples are generated independently of the
    # training dataset's samples.
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

    # Optimiser + schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=args.lr,
        foreach=False,
        #foreach=True,
    )
    
    #optimizer = torch.optim.SGD(
    #    model.parameters(),
    #    lr=1e-3,
    #    momentum=0.9,
    #    weight_decay=1e-4,
    #)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # Exponential cosine annealing with 10-epoch fade-in:
    #import math
    #from torch.optim.lr_scheduler import LambdaLR
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 
    #    math.exp(math.log((args.lr*0.033) / args.lr) * (1 - 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))))# * min(1.0, (epoch + 0.25) * 0.1)
    #)

    loss_l1 = nn.L1Loss()   # L1 → sharper results than MSE
    loss_mse = nn.MSELoss()

    torch.set_num_threads(1)
    
    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:4d}/{args.epochs}", leave=False)
        for lr_patch, res_patch in pbar:
            lr_patch  = lr_patch.to(device,  non_blocking=True)
            res_patch = res_patch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            #with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Not supported by old versions of torch-directml. dummy it out and don't use amp.
            if True:
                pred = model(lr_patch)              # (B, 1, 2*ph, 2*pw)  padding consumed by valid convs
                # Pixel loss: L1 for sharp gradients, MSE to penalise large outliers.
                # Gradient loss: L1 on spatial first-differences. This is the critical
                # term — it prevents the network collapsing to a near-zero residual
                # (the conditional mean under L1/MSE is ≈ 0 for diverse patches).
                absolute_loss = loss_l1(pred, res_patch)
                proportional_loss = loss_mse(pred, res_patch)
                pixel_loss = absolute_loss + proportional_loss
                
                if not args.basic_loss:
                    lE_loss = loss_lE(pred, res_patch)
                    loss = lE_loss
                
                if args.basic_loss:
                    loss = pixel_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            for param in model.parameters():
                torch.clamp(param, min=-3.99, max=3.99)
            
            #with torch.no_grad():
            #    for param in model.parameters():
            #        param.clamp_(-3.99, 3.99)
            
            lossitem = loss
            running_loss += lossitem
            #pbar.set_postfix(loss=f"{lossitem:.5f}")

        scheduler.step()

        avg_loss = running_loss.item() / len(loader)
        lr_now   = scheduler.get_last_lr()[0]

        if args.strict_validation:
            val_loss = run_validation(model, val_dataset, device)
            print(f"Epoch {epoch:4d} | train {avg_loss:.6f} | val {val_loss:.6f} | lr {lr_now:.2e}")
            if val_loss < best_loss:
                best_loss = val_loss
                trysave(model, out_path)
                print(f"  ✓ saved  {out_path}  (best val loss {best_loss:.6f})")
        else:
            print(f"Epoch {epoch:4d} | loss {avg_loss:.6f} | lr {lr_now:.2e}")
            trysave(model, out_path)

    print("Training complete.")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "4" # Limits OpenMP threads (e.g., to 4)
    os.environ["MKL_NUM_THREADS"] = "4" # Limits Intel MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = "4" # Limits NumExpr threads
    
    p = argparse.ArgumentParser(
        description="Train 2× per-channel image upscaler"
    )
    p.add_argument("data",
                   help="Folder of high-resolution training images")
    p.add_argument("--out", default="upscaler.safetensors",
                   help="Output .safetensors path  (default: upscaler.safetensors)")
    p.add_argument("--resume", default=None,
                   help="Load existing .safetensors weights and continue training")
    p.add_argument("--epochs",           type=int,   default=700)
    p.add_argument("--batch-size",       type=int,   default=64)
    
    p.add_argument("--patch-size",        type=int,   default=64,
                   help="LR patch height/width during training  (default: 64)")
    
    p.add_argument("--n-aug",              type=int,   default=4,
                   help="Augmented copies per blur variant per image  (default: 4)")
    
    p.add_argument("--precompute-factor",  type=int,   default=10,
                   help="Precompute X times as many pairs, use 1/X per epoch (default: 10)")
    p.add_argument("--patches-per-pair",   type=int,   default=20,
                   help="Random patch samples per stored pair per epoch  (default: 20)")
    p.add_argument("--brightness-range",  type=float, default=0.15,
                   help="Max ±brightness shift for augmentation  (default: 0.15)")
    p.add_argument("--lr",                type=float, default=2e-3)
    p.add_argument("--num-workers",       type=int,   default=4)
    p.add_argument("--leaky-slope",       type=float,   default=0.003)
    p.add_argument("--basic-loss",        action="store_true")
    p.add_argument("--strict-validation", action="store_true")
    p.add_argument("--val-data", default=None,
                   help="Folder of validation images (default: same as data folder)")
    args = p.parse_args()
    import model
    model.LEAKY_SLOPE = args.leaky_slope
    train(args)


if __name__ == "__main__":
    main()