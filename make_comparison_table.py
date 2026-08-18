#!/usr/bin/env python3
"""Generate a markdown table comparing every image under example_outputs/
against its original, using comparer.py's metric functions.

For each scene (e.g. "alice", "photo") the *_orig.png file is the ground
truth. Every other image in the directory that belongs to that scene is
compared against it, EXCEPT the "original half" file (<scene>_half.png)
and the originals themselves (comparing an image to itself is meaningless).

Output is three markdown tables per scene (3 columns each): pico / mini,
medium / silly / photo sillyish3, and nn / bilinear / fsr 1.x, with the
original only in the first table. The nene2x model columns show their
parameter counts, e.g. "pico (260 params)". Every image has its PSNR /
SSIM / LPIPS underneath it in the same cell (GitHub-compatible <br>).
Image references are relative markdown images, so the table renders
anywhere it is committed alongside example_outputs/.

Reuses comparer.py's internals: comparer.psnr_func, comparer.ssim_func and
comparer.load_lpips(). Image loading/normalization and the LPIPS tensor
conversion mirror comparer.run_comparison (PSNR on RGB float arrays,
SSIM on grayscale, both with data_range=255; LPIPS tensors in [-1, 1]).
"""

import argparse
import os
import re
import sys

import numpy as np
from PIL import Image
from safetensors import safe_open

import comparer  # reuse comparer.py's metric functions

ORIG_SUFFIX = "_orig.png"
HALF_SUFFIX = "_half.png"

# Output ordering, row-major across the grid: (label, regex matched
# against the filename). "silly" needs the leading "_" so
# half_phsillyish3_2x isn't mistaken for it.
TABLE_1 = [
    ("pico",       r"pico"),
    ("mini",       r"mini"),
    ("medium",     r"med"),
    ("silly",      r"_silly"),
]
TABLE_2 = [
    ("photo sillyish3", r"phsilly"),
    ("nn",              r"nn|nearest"),
    ("bilinear",        r"bilinear"),
    ("fsr 1.x",         r"fsr"),
]

COLS = 3  # grid columns per scene row

# safetensors model file for each nene2x output column, for parameter counts.
MODEL_FILES = {
    "pico":            "pretrained/nene2x_pico_b,3x3_12,1x1_8,1x1_4.safetensors",
    "mini":            "pretrained/nene2x_mini_3x3_12,1x1_8,3x3_12,3x3_4.safetensors",
    "medium":          "pretrained/nene2x_medium_g,3x3_12,3x3_12q,3x3_32,1x1_24,3x3_96d,1x1_12,3x3_12,1x1_4.safetensors",
    "silly":           "pretrained/nene2x_silly_g,3x3_24,3x3_32q,3x3_32,1x1_48,3x3_72,1x1_64,3x3_48,3x3_32,1x1_4.safetensors",
    "photo sillyish3": "pretrained_photo/nene2x_photo_sillyish3_g,3x3_24,3x3_24,3x3_48,3x3_64,3x3_96,1x1_4.safetensors",
}

_param_counts = {}


def count_params(path):
    """Number of parameters in a safetensors model file (cached)."""
    if path not in _param_counts:
        total = 0
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                total += f.get_tensor(key).size
        _param_counts[path] = total
    return _param_counts[path]


def find_scene(filename, scenes):
    """Return the scene whose prefix matches filename (longest match), or None."""
    matches = [s for s in scenes if filename.startswith(s + "_")]
    return max(matches, key=len) if matches else None


def find_originals(dirpath):
    """Map scene name -> path of its *_orig.png ground truth."""
    originals = {}
    for name in sorted(os.listdir(dirpath)):
        if name.endswith(ORIG_SUFFIX):
            scene = name[: -len(ORIG_SUFFIX)]
            originals[scene] = os.path.join(dirpath, name)
    return originals


def collect_rows(dirpath, originals):
    """Return (scene, filename, path, original_path) for every image to compare."""
    rows = []
    for name in sorted(os.listdir(dirpath)):
        path = os.path.join(dirpath, name)
        if not os.path.isfile(path):
            continue
        scene = find_scene(name, originals)
        if scene is None:
            print(f"warning: no matching original for {name}, skipping", file=sys.stderr)
            continue
        # The originals themselves: comparing an image to itself is meaningless.
        if name == scene + ORIG_SUFFIX:
            continue
        # The "original half" file: half-resolution input to the upscalers.
        # Note: <scene>_half_*.png outputs are full size and ARE compared.
        if name == scene + HALF_SUFFIX:
            continue
        rows.append((scene, name, path, originals[scene]))
    rows.sort(key=lambda r: (r[0], kind_sort_key(r[1])[:2], r[1]))
    return rows


def make_lpips_loss():
    torch, lpips_lib = comparer.load_lpips()
    loss_fn = lpips_lib.LPIPS(net="alex")

    def to_lpips_tensor(arr):
        # Same conversion as comparer.run_comparison: [-1, 1], shape (N, C, H, W).
        t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    return loss_fn, to_lpips_tensor


def compute_metrics(gt_path, est_path, loss_fn, to_lpips_tensor):
    """Mirror comparer.run_comparison's metric computations; return (psnr, ssim, lpips)."""
    img_gt = Image.open(gt_path)
    img_est = Image.open(est_path)
    if img_gt.size != img_est.size:
        raise ValueError(
            f"dimension mismatch: {est_path} {img_est.size} vs ground truth "
            f"{gt_path} {img_gt.size}"
        )
    gt_arr = np.array(img_gt.convert("RGB")).astype(float)
    est_arr = np.array(img_est.convert("RGB")).astype(float)
    gt_gray = np.array(img_gt.convert("L")).astype(float)
    est_gray = np.array(img_est.convert("L")).astype(float)

    psnr = comparer.psnr_func(gt_arr, est_arr, data_range=255)
    ssim = comparer.ssim_func(gt_gray, est_gray, data_range=255)
    lpips = loss_fn(to_lpips_tensor(gt_arr), to_lpips_tensor(est_arr)).item()
    return psnr, ssim, lpips


def fmt_psnr(value):
    return "inf" if np.isinf(value) else f"{value:.2f}"


def kind_sort_key(name):
    """Return (table index, column position, column label) for a filename.
    Unknown files sort at the end of table 2, alphabetically."""
    for t, kinds in enumerate((TABLE_1, TABLE_2)):
        for i, (label, pattern) in enumerate(kinds):
            if re.search(pattern, name):
                return t, i, label
    return 1, len(TABLE_2), os.path.splitext(name)[0]


def header_label(name):
    """Column header for a filename; nene2x model columns get param counts."""
    label = kind_sort_key(name)[2]
    path = MODEL_FILES.get(label)
    if path:
        try:
            return f"{label} ({count_params(path):,} params)"
        except OSError as e:
            print(f"warning: {path}: {e}, no param count", file=sys.stderr)
    return label


def render_table(rows, md_dir, loss_fn, to_lpips_tensor):
    """Render one markdown table per scene: a single strip with the original
    followed by every output image, each with its metrics underneath."""
    computed = []
    for scene, name, path, orig_path in rows:
        try:
            psnr, ssim, lpips = compute_metrics(orig_path, path, loss_fn, to_lpips_tensor)
        except Exception as e:
            print(f"warning: {path}: {e}, skipping", file=sys.stderr)
            continue
        computed.append((scene, name, path, orig_path, psnr, ssim, lpips))

    by_scene = {}
    scene_order = []
    for scene, name, path, orig_path, psnr, ssim, lpips in computed:
        if scene not in by_scene:
            by_scene[scene] = []
            scene_order.append(scene)
        by_scene[scene].append((name, path, orig_path, psnr, ssim, lpips))

    lines = ["<!-- PSNR / SSIM: higher is better. LPIPS: lower is better. -->", ""]
    for scene in scene_order:
        entries = by_scene[scene]
        orig_path = entries[0][2]
        rel_orig = os.path.relpath(orig_path, md_dir).replace(os.sep, "/")

        lines.append(f"## {scene}")
        lines.append("")

        # Dummy text of the same length as the metrics line, so the
        # original cell lines up with the output cells.
        _n, _p, _o, psnr0, ssim0, lpips0 = entries[0]
        metrics_line = f"PSNR {fmt_psnr(psnr0)} / SSIM {ssim0:.4f} / LPIPS {lpips0:.4f}"
        dummy = re.sub(r"[0-9.]", "-", metrics_line)

        labels = ["Original"]
        cells = [f"![{os.path.basename(orig_path)}]({rel_orig})<br>{dummy}"]
        for name, path, _o, psnr, ssim, lpips in entries:
            labels.append(header_label(name))
            rel = os.path.relpath(path, md_dir).replace(os.sep, "/")
            cells.append(
                f"![{name}]({rel})<br>"
                f"PSNR {fmt_psnr(psnr)} / SSIM {ssim:.4f} / LPIPS {lpips:.4f}"
            )

        # One table per row of the grid, COLS columns each, with its own header.
        for i in range(0, len(cells), COLS):
            heads = labels[i:i + COLS]
            cols = cells[i:i + COLS]
            if len(cols) < COLS:  # pad a ragged last table
                heads += [""] * (COLS - len(heads))
                cols += [""] * (COLS - len(cols))
            lines.append("| " + " | ".join(heads) + " |")
            lines.append("| --- | --- | --- |")
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare every image under a directory against its *_orig.png "
            "ground truth (skipping the <scene>_half.png originals) and emit "
            "a markdown table with PSNR, SSIM and LPIPS."
        )
    )
    parser.add_argument("--dir", default="example_outputs",
                        help="directory containing originals and outputs (default: example_outputs)")
    parser.add_argument("--output", "-o", default=None,
                        help="write the table to this file instead of stdout")
    args = parser.parse_args()

    dirpath = args.dir
    originals = find_originals(dirpath)
    if not originals:
        print(f"error: no *_orig.png ground truth files found in {dirpath}", file=sys.stderr)
        sys.exit(1)

    loss_fn, to_lpips_tensor = make_lpips_loss()
    rows = collect_rows(dirpath, originals)
    md_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else os.getcwd()
    table = render_table(rows, md_dir, loss_fn, to_lpips_tensor)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"wrote {len(rows)} rows to {args.output}")
    else:
        sys.stdout.write(table)


if __name__ == "__main__":
    main()
