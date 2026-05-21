# Entirely vibecoded implementation of FSR1.x
# Higher quality than AMD's official FidelityFX CLI tool for some reason.
# AMD's tool doesn't support adjusting sharpness in FSR+Scale mode.

# official CLI
# $ uv run python comparer.py test/marona_orig.png test/marona_half_2xfsrcli.png --lpips
# (468, 864)
# -0.03143969424412091
# Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
# Loading model from: C:\Users\wareya\dev\nene2x\.venv\Lib\site-packages\lpips\weights\v0.1\alex.pth
# --------------------------------------------------
# Metrics for: test/marona_half_2xfsrcli.png
# PSNR:          34.10  (Better: Higher)
# DFT PSNR:      37.01  (Better: Higher)
# SSIM:         0.9867  (Better: Higher)
# NMI:          1.5275  (Better: Higher)
# MAE:          0.0077  (Better: Lower)
# VOI:          5.1640  (Better: Lower)
# B-VOI:        1.1449  (Better: Lower)
# LPIPS (Alex): 0.0186  (Better: Lower)
# AED:          0.0282  (Better: Lower)
# --------------------------------------------------
# 
# # this one
# $ uv run python comparer.py test/marona_orig.png test/marona_half_2xfsr.png --lpips
# (468, 864)
# -0.03143969424412091
# Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
# Loading model from: C:\Users\wareya\dev\nene2x\.venv\Lib\site-packages\lpips\weights\v0.1\alex.pth
# --------------------------------------------------
# Metrics for: test/marona_half_2xfsr.png
# PSNR:          34.51  (Better: Higher)
# DFT PSNR:      37.49  (Better: Higher)
# SSIM:         0.9875  (Better: Higher)
# NMI:          1.5243  (Better: Higher)
# MAE:          0.0074  (Better: Lower)
# VOI:          5.2381  (Better: Lower)
# B-VOI:        1.1351  (Better: Lower)
# LPIPS (Alex): 0.0134  (Better: Lower)
# AED:          0.0318  (Better: Lower)
# --------------------------------------------------


import argparse
import os
import sys
from PIL import Image
import moderngl

def process_image(input_filepath, sharpness):
    file_root, file_ext = os.path.splitext(input_filepath)
    output_filepath = f"{file_root}_2xfsr{file_ext}"

    try:
        source_image = Image.open(input_filepath).convert("RGBA")
    except Exception as e:
        print(f"Could not load the image: {e}")
        return

    width, height = source_image.size
    new_width, new_height = width * 2, height * 2

    context = moderngl.create_standalone_context(require=430)

    # Texture 0: Original input
    input_texture = context.texture((width, height), 4, source_image.tobytes())
    input_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    # Texture 1: Intermediate upscaled output from EASU (32-bit float to preserve precision/negative lobes)
    easu_texture = context.texture((new_width, new_height), 4, dtype='f4')
    
    # Texture 2: Final sharpened output from RCAS
    output_texture = context.texture((new_width, new_height), 4)

    # ---------------------------------------------------------
    # PASS 1: EASU (Edge Adaptive Spatial Upsampling)
    # ---------------------------------------------------------
    fsr1_easu_source = """
    #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
    
    layout(rgba8, binding = 0) uniform readonly image2D img_in;
    layout(rgba32f, binding = 1) uniform writeonly image2D img_out;

    // Evaluates a 5-tap cross to determine local edge direction and length
    void FsrEasuSet(float lC, float lL, float lR, float lT, float lB, float w, inout vec2 dir, inout float len) {
        float lenX = max(abs(lR - lC), abs(lC - lL));
        float dirX = lR - lL;
        lenX = lenX > 0.0 ? clamp(abs(dirX) / lenX, 0.0, 1.0) : 0.0;
        lenX *= lenX;

        float lenY = max(abs(lB - lC), abs(lC - lT));
        float dirY = lB - lT;
        lenY = lenY > 0.0 ? clamp(abs(dirY) / lenY, 0.0, 1.0) : 0.0;
        lenY *= lenY;

        dir += vec2(dirX, dirY) * w;
        len += (lenX + lenY) * w;
    }

    void FsrEasuTap(
        inout vec4 color_accum, inout float weight_accum, vec2 offset,
        vec2 dir, vec2 length_scale, float lobe_strength, float clipping_point, vec4 tap_color
    ) {
        vec2 v;
        // Project spatial offset along and across the gradient
        v.x = (offset.x * dir.x) + (offset.y * dir.y);
        v.y = (offset.x * -dir.y) + (offset.y * dir.x);
        v *= length_scale;

        float d2 = v.x * v.x + v.y * v.y;
        d2 = min(d2, clipping_point);

        // Standard FSR1 Lanczos polynomial window
        float wB = (2.0 / 5.0) * d2 - 1.0;
        float wA = lobe_strength * d2 - 1.0;
        wB *= wB;
        wA *= wA;
        wB = (25.0 / 16.0) * wB - (9.0 / 16.0);
        float weight = wB * wA; 

        color_accum += tap_color * weight;
        weight_accum += weight;
    }

    void main() {
        ivec2 out_pos = ivec2(gl_GlobalInvocationID.xy);
        ivec2 out_dim = imageSize(img_out);
        if (out_pos.x >= out_dim.x || out_pos.y >= out_dim.y) return;

        vec2 in_dim = vec2(imageSize(img_in));
        vec2 uv = (vec2(out_pos) + 0.5) / vec2(out_dim);
        vec2 in_pos = uv * in_dim - 0.5;

        ivec2 base_pos = ivec2(floor(in_pos));
        vec2 fract_pos = fract(in_pos);
        ivec2 max_dim = ivec2(in_dim) - 1;

        // FSR1 12-tap sampling footprint bounds
        const ivec2 tap_offsets[12] = ivec2[](
            ivec2(0, -1), ivec2(1, -1),
            ivec2(-1, 0), ivec2(0, 0), ivec2(1, 0), ivec2(2, 0),
            ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1), ivec2(2, 1),
            ivec2(0, 2), ivec2(1, 2)
        );

        vec4 colors[12];
        float lumas[12];
        vec3 luma_w = vec3(0.299, 0.587, 0.114);

        for (int i = 0; i < 12; i++) {
            ivec2 p = clamp(base_pos + tap_offsets[i], ivec2(0), max_dim);
            colors[i] = imageLoad(img_in, p);
            lumas[i] = dot(colors[i].rgb, luma_w);
        }

        float wx = fract_pos.x;
        float wy = fract_pos.y;
        float w0 = (1.0 - wx) * (1.0 - wy);
        float w1 = wx * (1.0 - wy);
        float w2 = (1.0 - wx) * wy;
        float w3 = wx * wy;

        vec2 dir = vec2(0.0);
        float len = 0.0;

        FsrEasuSet(lumas[3], lumas[2], lumas[4], lumas[0], lumas[7], w0, dir, len);
        FsrEasuSet(lumas[4], lumas[3], lumas[5], lumas[1], lumas[8], w1, dir, len);
        FsrEasuSet(lumas[7], lumas[6], lumas[8], lumas[3], lumas[10], w2, dir, len);
        FsrEasuSet(lumas[8], lumas[7], lumas[9], lumas[4], lumas[11], w3, dir, len);

        vec2 edge_dir;
        vec2 length_scale;
        
        // MISSING MATH RESTORED: Remap raw len [0, 2] smoothly to [0, 1] curve
        len = len * 0.5;
        len *= len;

        float dir2 = dir.x * dir.x + dir.y * dir.y;
        if (dir2 < 0.0001) {
            edge_dir = vec2(1.0, 0.0);
            length_scale = vec2(1.0, 1.0);
        } else {
            edge_dir = dir * inversesqrt(dir2);
            float stretch = 1.0 / max(abs(edge_dir.x), abs(edge_dir.y));
            
            // x: squash window across the edge to retain crispness
            // y: stretch window along the edge to smooth stair-stepping. (Capped to max 2x stretch now)
            length_scale = vec2(
                1.0 + (stretch - 1.0) * len,
                1.0 - 0.5 * len
            );
        }

        // MISSING MATH RESTORED: Dynamic clipping boundaries
        // Based on the 'edge' strength, window shifts from isotropic to deeply negative lobed
        float lobe = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
        float clip = 1.0 / lobe; // Guarantees window tapers to exactly 0.0 weight at the edge boundary

        vec4 color_accum = vec4(0.0);
        float weight_accum = 0.0;

        for (int i = 0; i < 12; i++) {
            vec2 offset = vec2(tap_offsets[i]) - fract_pos;
            FsrEasuTap(color_accum, weight_accum, offset, edge_dir, length_scale, lobe, clip, colors[i]);
        }

        vec4 final_color;
        if (abs(weight_accum) > 0.000001) {
            final_color = color_accum / weight_accum;
        } else {
            final_color = colors[3]; 
        }

        imageStore(img_out, out_pos, final_color);
    }
    """

    # ---------------------------------------------------------
    # PASS 2: RCAS (Robust Contrast Adaptive Sharpening)
    # ---------------------------------------------------------
    fsr1_rcas_source = """
    #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
    
    layout(rgba32f, binding = 0) uniform readonly image2D img_in;
    layout(rgba8, binding = 1) uniform writeonly image2D img_out;
    
    uniform float sharpness;
    
    void main() {
        ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
        ivec2 dim = imageSize(img_out);
        if (pos.x >= dim.x || pos.y >= dim.y) return;
        
        vec4 original_pixel = imageLoad(img_in, pos);
        
        if (sharpness <= 0.0) {
            imageStore(img_out, pos, vec4(clamp(original_pixel.rgb, 0.0, 1.0), original_pixel.a));
            return;
        }

        ivec2 max_dim = dim - 1;
        
        vec3 c = original_pixel.rgb;
        vec3 n = imageLoad(img_in, clamp(pos + ivec2(0, -1), ivec2(0), max_dim)).rgb;
        vec3 s = imageLoad(img_in, clamp(pos + ivec2(0, 1), ivec2(0), max_dim)).rgb;
        vec3 w = imageLoad(img_in, clamp(pos + ivec2(-1, 0), ivec2(0), max_dim)).rgb;
        vec3 e = imageLoad(img_in, clamp(pos + ivec2(1, 0), ivec2(0), max_dim)).rgb;

        vec3 min_c = min(min(n, s), min(w, e));
        vec3 max_c = max(max(n, s), max(w, e));
        
        vec3 limitMin = min(min_c, c);
        vec3 limitMax = max(max_c, c);
        
        vec3 limit = min(c - limitMin, limitMax - c);
        vec3 peakC = 1.0 / (max_c - min_c + 0.0001);
        
        float max_lobe = sharpness * -0.25;
        vec3 w3 = max_lobe * (limit * peakC);
        
        vec3 final_color = (c + (n + s + w + e) * w3) / (1.0 + 4.0 * w3);
        
        imageStore(img_out, pos, vec4(clamp(final_color, 0.0, 1.0), original_pixel.a));
    }
    """

    easu_program = context.compute_shader(fsr1_easu_source)
    rcas_program = context.compute_shader(fsr1_rcas_source)

    groups_x = (new_width + 15) // 16
    groups_y = (new_height + 15) // 16

    # Run Pass 1
    input_texture.bind_to_image(0, read=True, write=False)
    easu_texture.bind_to_image(1, read=False, write=True)
    easu_program.run(groups_x, groups_y, 1)

    # Run Pass 2
    easu_texture.bind_to_image(0, read=True, write=False) 
    output_texture.bind_to_image(1, read=False, write=True) 
    rcas_program["sharpness"].value = sharpness
    rcas_program.run(groups_x, groups_y, 1)

    output_data = output_texture.read()
    result_image = Image.frombytes("RGBA", (new_width, new_height), output_data)
    result_image.save(output_filepath)
    print(f"Upscaling complete! Saved to: {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Upscale an image 2x using FSR1 (Float Pipeline) via GLSL Compute Shaders.")
    parser.add_argument("image", help="Path to the source image file")
    parser.add_argument("-s", "--sharpness", type=float, default=0.5, 
                        help="RCAS Sharpening intensity from 0.0 (none) to 1.0 (max). Default: 0.5")
    options = parser.parse_args()

    if not os.path.isfile(options.image):
        print(f"Error: File not found: {options.image}")
        sys.exit(1)

    clamped_sharpness = max(0.0, min(1.0, options.sharpness))
    process_image(options.image, clamped_sharpness)

if __name__ == "__main__":
    main()