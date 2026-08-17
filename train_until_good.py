import subprocess
import sys
import re

DATA = sys.argv[1] if len(sys.argv) > 1 else "train"

attempt = 0
while True:
    attempt += 1
    print(f"\n=== Attempt {attempt} ===")

    r = subprocess.run(["uv", "run", "python", "train.py", DATA, "--epochs", "50", "--out", "upscaler.safetensors", "--seed", str(attempt+42)])
    if r.returncode != 0:
        print(f"train.py failed (exit {r.returncode}), retrying...")
        continue

    r = subprocess.run(["uv", "run", "python", "infer.py", "upscaler.safetensors", "test/marona_half.png", "--leaky-slope", "0.0"])
    if r.returncode != 0:
        print(f"infer.py failed (exit {r.returncode}), retrying...")
        continue

    r = subprocess.run(
        ["uv", "run", "python", "comparer.py", "test/marona_orig.png", "test/marona_half_2x.png"],
        capture_output=True, text=True
    )
    print(r.stdout)
    if r.returncode != 0:
        print(f"comparer.py failed (exit {r.returncode}), retrying...")
        continue

    m = re.search(r"PSNR:\s+([\d.]+)", r.stdout)
    if not m:
        print("Could not parse PSNR from comparer output, retrying...")
        continue

    psnr = float(m.group(1))
    print(f"PSNR: {psnr}")
    
    target = 36.4
    #target = 34.3

    if psnr >= target:
        print(f"PSNR {psnr} >= {target} — success!")
        sys.exit(0)
    else:
        print(f"PSNR {psnr} < {target}, retrying...")
