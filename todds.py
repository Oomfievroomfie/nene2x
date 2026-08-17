import argparse
import os
import sys
from PIL import Image
import ddslop

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tiff', '.tif', '.webp'}


def has_partial_alpha(img: Image.Image) -> bool:
    """Return True if the image has any pixel with alpha < 255."""
    if img.mode not in ("RGBA", "LA"):
        return False
    alpha = img.getchannel("A")
    return alpha.getextrema()[0] < 255


def resolve_dxt0(img: Image.Image) -> str:
    """DXT0 pseudo-format: DXT1 if all alpha is fully opaque, otherwise DXT5."""
    return "DXT5" if has_partial_alpha(img) else "DXT1"


def convert_single(input_path: str, output_path: str, fmt: str, mipmaps: bool, mips_linear: bool):
    """Convert one image file to DDS."""
    with Image.open(input_path) as img:
        print(f"  Processing: {input_path}")
        print(f"  Dimensions: {img.size[0]}x{img.size[1]}")

        final_format = resolve_dxt0(img) if fmt == "DXT0" else fmt
        if fmt == "DXT0":
            print(f"  DXT0 → {final_format}")

        ddslop.save_dds(
            image=img,
            dest=output_path,
            pixel_format=final_format,
            mipmaps=mipmaps,
            mipmaps_linear=mips_linear,
            pca=3,
        )
        print(f"  Saved: {output_path} [{final_format}]")


def process_folder(input_dir: str, fmt: str, mipmaps: bool, mips_linear: bool):
    """Batch-convert all images in input_dir, writing to input_dir/dxt/."""
    output_dir = os.path.join(input_dir, "dxt")
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)

    if not files:
        print(f"No image files found in '{input_dir}'.")
        return

    print(f"Found {len(files)} image(s) in '{input_dir}':")
    ok = 0
    for fname in files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".dds")
        try:
            convert_single(in_path, out_path, fmt, mipmaps, mips_linear)
            ok += 1
        except Exception as e:
            print(f"  Failed: {fname}: {e}")

    print(f"\nDone: {ok}/{len(files)} images → '{output_dir}/'")


def run_conversion():
    parser = argparse.ArgumentParser(
        description="Convert images to DDS using ddslop."
    )
    parser.add_argument("input", help="Path to a source image or a folder of images")
    parser.add_argument("-o", "--output", help="Output DDS path (ignored when input is a folder)")
    parser.add_argument(
        "-f", "--format",
        choices=["DXT0", "DXT1", "DXT5", "BC7", "BC7lite", "BC7nano", "BC7zero"],
        default="DXT0",
        help="DDS compression format (default: DXT0, auto-picks DXT1/DXT5 based on alpha)",
    )
    parser.add_argument("--no-mips", action="store_false", dest="mipmaps", help="Disable mipmap generation")
    parser.add_argument("--mips-linear", action="store_true", dest="mips_linear", help="Generate mipmaps in linear RGB")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: '{args.input}' does not exist.")
        sys.exit(1)

    # Folder mode
    if os.path.isdir(args.input):
        process_folder(args.input, args.format, args.mipmaps, args.mips_linear)
        return

    # Single-file mode
    output = args.output or os.path.splitext(args.input)[0] + ".dds"
    try:
        convert_single(args.input, output, args.format, args.mipmaps, args.mips_linear)
    except Exception as e:
        print(f"Failed to convert image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_conversion()
