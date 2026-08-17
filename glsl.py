import argparse
from pathlib import Path
from safetensors.torch import load_file


def _chunked_array_str(tensor, float_fmt="{:.5f}", threshold=0.0):
    """Flattens a tensor and formats it into chunked string for GLSL.
    Values with absolute value < threshold are replaced with 0.0.
    """
    vals = tensor.flatten().tolist()
    if threshold > 0:
        vals = [0.0 if abs(v) < threshold else v for v in vals]
    chunks = [
        ", ".join(float_fmt.format(v) for v in vals[i:i+16])
        for i in range(0, len(vals), 16)
    ]
    return ",\n        ".join(chunks), len(vals)


def generate_glsl_shader(weights_path: str, activation: str, threshold: float) -> str:
    state = load_file(weights_path)

    # Discover intermediate layers from state dict (simple models only:
    # layers.0, layers.1, ... with no depthwise/dilation/noise flags).
    layer_w = []
    layer_b = []
    n = 0
    while f"layers.{n}.weight" in state:
        layer_w.append(state[f"layers.{n}.weight"])
        bk = f"layers.{n}.bias"
        layer_b.append(state[bk] if bk in state else None)
        n += 1

    # Detect final output layer key (matching model.py _config_from_state).
    if "zfinalb.weight" in state:
        final_key, final_bias_key = "zfinalb.weight", "zfinalb.bias"
    elif "zfinalg.weight" in state:
        final_key, final_bias_key = "zfinalg.weight", "zfinalg.bias"
    elif "zfinalj.weight" in state:
        final_key, final_bias_key = "zfinalj.weight", "zfinalj.bias"
    elif "zfinalx.weight" in state:
        final_key, final_bias_key = "zfinalx.weight", "zfinalx.bias"
    else:
        final_key, final_bias_key = "zfinal.weight", "zfinal.bias"

    final_w = state[final_key]
    final_b = state.get(final_bias_key, None)

    # First layer is the spatial convolution – read kernel size and out channels
    k = layer_w[0].shape[2]   # kernel size (square, e.g. 3 for 3x3)
    c0 = layer_w[0].shape[0]  # output channels of first layer

    # Apply activation helper (shared by all hidden layers)
    def apply_activation(var_name):
        if activation == "relu":
            return f"        {var_name} = max(0.0, {var_name});"
        elif activation == "leaky_relu":
            return f"        {var_name} = max(0.01 * {var_name}, {var_name});"
        elif activation == "gelu":
            return f"        {var_name} = 0.5 * {var_name} * (1.0 + tanh(sqrt(2.0/3.14159265) * ({var_name} + 0.044715 * {var_name} * {var_name} * {var_name})));"
        return ""

    # Helper to emit a const float array into the shader
    def add_array(shader, name, tensor):
        val_str, size = _chunked_array_str(tensor, threshold=threshold)
        shader.append(f"    const float {name}[{size}] = float[{size}](\n        {val_str}\n    );")

    shader = []
    shader.append("vec4 getResiduals(sampler2D tex, vec4 which_channel, vec2 uv, vec2 tex_size_pixels) {")
    shader.append("    vec2 inv_res = 1.0 / tex_size_pixels;")
    shader.append(f"    int half_k = {k // 2};")
    shader.append("")

    # 1. Bake weights into const arrays
    add_array(shader, "conv_w", layer_w[0])
    if layer_b[0] is not None:
        add_array(shader, "conv_b", layer_b[0])
    for i in range(1, n):
        add_array(shader, f"mid_{i}_w", layer_w[i])
        if layer_b[i] is not None:
            add_array(shader, f"mid_{i}_b", layer_b[i])
    add_array(shader, "zfinal_w", final_w)
    if final_b is not None:
        add_array(shader, "zfinal_b", final_b)
    shader.append("")

    # 2. Extract pixel neighbourhood (single input channel)
    shader.append(f"    float neigh[{k*k}];")
    shader.append("    int idx = 0;")
    shader.append("    for (int dy = -half_k; dy <= half_k; dy++) {")
    shader.append("        for (int dx = -half_k; dx <= half_k; dx++) {")
    shader.append("            vec2 offset = vec2(float(dx), float(dy)) * inv_res;")
    shader.append("            neigh[idx++] = dot(texture2DLod(tex, uv + offset, 0.0), which_channel);")
    shader.append("        }")
    shader.append("    }")
    shader.append("")

    # 3. Layer 0: spatial convolution (in_c=1, so spatial-only inner loop)
    shader.append(f"    float h0[{c0}];")
    shader.append("    int w_idx = 0;")
    shader.append(f"    for (int i=0; i<{c0}; i++) {{")
    if layer_b[0] is not None:
        shader.append("        h0[i] = conv_b[i];")
    else:
        shader.append("        h0[i] = 0.0;")
    shader.append(f"        for (int j=0; j<{k*k}; j++) {{")
    shader.append("            h0[i] += neigh[j] * conv_w[w_idx++];")
    shader.append("        }")
    act_str = apply_activation("h0[i]")
    if act_str:
        shader.append(act_str)
    shader.append("    }")
    shader.append("")

    # 4. Middle layers (1x1 pointwise convolutions)
    prev_h = "h0"
    prev_size = c0
    for mi in range(1, n):
        curr_c = layer_w[mi].shape[0]
        curr_h = f"h{mi}"
        has_bias = layer_b[mi] is not None
        shader.append(f"    float {curr_h}[{curr_c}];")
        shader.append("    w_idx = 0;")
        shader.append(f"    for (int i=0; i<{curr_c}; i++) {{")
        if has_bias:
            shader.append(f"        {curr_h}[i] = mid_{mi}_b[i];")
        else:
            shader.append(f"        {curr_h}[i] = 0.0;")
        shader.append(f"        for (int j=0; j<{prev_size}; j++) {{")
        shader.append(f"            {curr_h}[i] += {prev_h}[j] * mid_{mi}_w[w_idx++];")
        shader.append("        }")
        act_str = apply_activation(f"{curr_h}[i]")
        if act_str:
            shader.append(act_str)
        shader.append("    }")
        shader.append("")
        prev_h = curr_h
        prev_size = curr_c

    # 5. Final layer (map to 4 output channels, no activation)
    if final_b is not None:
        shader.append("    vec4 out_val = vec4(zfinal_b[0], zfinal_b[1], zfinal_b[2], zfinal_b[3]);")
    else:
        shader.append("    vec4 out_val = vec4(0.0, 0.0, 0.0, 0.0);")
    shader.append("    w_idx = 0;")
    shader.append("    for (int i=0; i<4; i++) {")
    shader.append(f"        for (int j=0; j<{prev_size}; j++) {{")
    shader.append(f"            out_val[i] += {prev_h}[j] * zfinal_w[w_idx++];")
    shader.append("        }")
    shader.append("    }")
    shader.append("")
    shader.append("    return out_val;")
    shader.append("}")

    return "\n".join(shader)


def main():
    p = argparse.ArgumentParser(description="Convert a safetensors upscaler to a GLSL shader.")
    p.add_argument("model", help="Path to .safetensors weights file")
    p.add_argument("--output", "-o", default=None, help="Output GLSL file. Defaults to stdout.")
    p.add_argument("--activation", choices=["relu", "leaky_relu", "gelu", "none"], default="relu",
                   help="Activation function used in the hidden layers (default: relu)")
    p.add_argument("--weight-threshold", type=float, default=0.0,
                   help="Zero out weights/biases with absolute value less than this threshold (default: 0.0)")
    args = p.parse_args()

    glsl_code = generate_glsl_shader(args.model, args.activation, args.weight_threshold)

    if args.output:
        Path(args.output).write_text(glsl_code)
        print(f"GLSL shader written to {args.output}")
    else:
        print(glsl_code)


if __name__ == "__main__":
    main()
