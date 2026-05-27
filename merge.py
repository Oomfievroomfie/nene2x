import numpy as np
from PIL import Image
import argparse

def separable_blur(img, radius):
    # box blur approximation via two passes
    size = int(radius * 2 + 1)
    if size < 1:
        return img
    kernel = np.ones(size) / size

    def blur_axis(arr, k):
        pad = len(k) // 2
        out = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            for row in range(arr.shape[0]):
                out[row, :, c] = np.convolve(arr[row, :, c], k, mode='same')
        return out

    def blur_axis_v(arr, k):
        pad = len(k) // 2
        out = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            for col in range(arr.shape[1]):
                out[:, col, c] = np.convolve(arr[:, col, c], k, mode='same')
        return out

    return blur_axis_v(blur_axis(img, kernel), kernel)

def merge(low_path, high_path, radius, output_path):
    low_img = Image.open(low_path).convert('RGB')
    high_img = Image.open(high_path).convert('RGB')

    if low_img.size != high_img.size:
        raise ValueError(f"Image sizes differ: {low_img.size} vs {high_img.size}")

    low = np.array(low_img, dtype=np.float32)
    high = np.array(high_img, dtype=np.float32)

    low_blurred = separable_blur(low, radius)
    high_blurred = separable_blur(high, radius)
    high_pass = high - high_blurred

    result = low_blurred + high_pass
    result = np.clip(np.round(result), 0, 255).astype(np.uint8)

    Image.fromarray(result).save(output_path)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge two images: lowpass one, highpass the other')
    parser.add_argument('low', help='Image to take lowpass from')
    parser.add_argument('high', help='Image to take highpass from')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--radius', type=float, default=4.0, help='Blur radius for frequency cutoff (default: 4.0)')
    args = parser.parse_args()

    merge(args.low, args.high, args.radius, args.output)
