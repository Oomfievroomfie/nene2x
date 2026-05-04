import argparse
from pathlib import Path
from safetensors.torch import load_file

def _chunked_array_str(tensor, float_fmt="{:.5f}", threshold=0.0):
    """Flattens a tensor and formats it into chunked string for GLSL.
    Values with absolute value < threshold are replaced with 0.0.
    """
    vals = tensor.flatten().tolist()
    # Apply threshold: zero out small values
    if threshold > 0:
        vals = [0.0 if abs(v) < threshold else v for v in vals]
    chunks = [
        ", ".join(float_fmt.format(v) for v in vals[i:i+16]) 
        for i in range(0, len(vals), 16)
    ]
    return ",\n        ".join(chunks), len(vals)

def generate_glsl_shader(weights_path: str, activation: str, threshold: float) -> str:
    state = load_file(weights_path)

    has_bconv = "bconv.weight" in state

    # Infer dimensions from the shapes
    conv_w = state["conv.weight"]
    c, conv_in_ch, k, _ = conv_w.shape  # conv_in_ch == 4 for new, 1 for old

    mids_w = []
    mids_b = []
    nm = 0
    while f"mids.{nm}.weight" in state:
        mids_w.append(state[f"mids.{nm}.weight"])
        mids_b.append(state[f"mids.{nm}.bias"])
        nm += 1

    ms = mids_w[0].shape[0] if nm > 0 else 0

    zfinal_w = state["zfinal.weight"]
    zfinal_b = state["zfinal.bias"]

    # Apply activation helper (shared by both functions)
    def apply_activation(var_name):
        if activation == "relu":
            return f"        {var_name} = max(0.0, {var_name});"
        elif activation == "leaky_relu":
            return f"        {var_name} = max(0.01 * {var_name}, {var_name});"
        elif activation == "gelu":
            return f"        {var_name} = 0.5 * {var_name} * (1.0 + tanh(sqrt(2.0/3.14159265) * ({var_name} + 0.044715 * {var_name} * {var_name} * {var_name})));"
        return ""

    # Helper to emit a const float array into a shader list
    def add_array(shader, name, tensor):
        val_str, size = _chunked_array_str(tensor, threshold=threshold)
        shader.append(f"    const float {name}[{size}] = float[{size}](\n        {val_str}\n    );")

    parts = []  # will hold the one or two function strings

    # -------------------------------------------------------------------------
    # NEW networks only: Pass-1 function  getBconvOutput
    #   Conv2d(1 -> 4, 3x3) + activation
    #   Returns vec4 written to an RGBA render target; one pass per channel.
    # -------------------------------------------------------------------------
    if has_bconv:
        bconv_w = state["bconv.weight"]   # shape (4, 1, 3, 3)
        bconv_b = state["bconv.bias"]     # shape (4,)
        bk = bconv_w.shape[2]            # always 3, but read from weights to be safe

        fn = []
        fn.append("vec4 getBconvOutput(sampler2D tex, vec4 which_channel, vec2 uv, vec2 tex_size_pixels) {")
        fn.append("    vec2 inv_res = 1.0 / tex_size_pixels;")
        fn.append(f"    int half_k = {bk // 2};")
        fn.append("")

        add_array(fn, "bconv_w", bconv_w)
        add_array(fn, "bconv_b", bconv_b)
        fn.append("")

        # Sample 3x3 neighbourhood of the single input channel
        fn.append(f"    float neigh[{bk*bk}];")
        fn.append("    int idx = 0;")
        fn.append("    for (int dy = -half_k; dy <= half_k; dy++) {")
        fn.append("        for (int dx = -half_k; dx <= half_k; dx++) {")
        fn.append("            vec2 offset = vec2(float(dx), float(dy)) * inv_res;")
        fn.append("            neigh[idx++] = dot(texture2DLod(tex, uv + offset, 0.0), which_channel);")
        fn.append("        }")
        fn.append("    }")
        fn.append("")

        # bconv: 1 input channel x bk*bk spatial -> 4 output channels
        fn.append("    vec4 out_val;")
        fn.append("    int w_idx = 0;")
        fn.append("    for (int i = 0; i < 4; i++) {")
        fn.append("        float v = bconv_b[i];")
        fn.append(f"        for (int j = 0; j < {bk*bk}; j++) {{")
        fn.append("            v += neigh[j] * bconv_w[w_idx++];")
        fn.append("        }")
        act_str = apply_activation("v")
        if act_str:
            fn.append("    " + act_str.lstrip())  # de-indent one level to match 'v' scope
        fn.append("        out_val[i] = v;")
        fn.append("    }")
        fn.append("")
        fn.append("    return out_val;")
        fn.append("}")

        parts.append("\n".join(fn))

    # -------------------------------------------------------------------------
    # Pass-2 / only function: getResiduals
    #
    #   OLD network: getResiduals(sampler2D tex, vec4 which_channel, vec2 uv, vec2 tex_size_pixels)
    #     - samples single float per neighbourhood position
    #     - conv inner loop: k*k inputs
    #
    #   NEW network: getResiduals(sampler2D bconv_tex, vec2 uv, vec2 tex_size_pixels)
    #     - samples vec4 per neighbourhood position from the bconv render target
    #     - conv inner loop: in_ch (0..3) x k*k spatial inputs, matching
    #       PyTorch weight layout (out, in, kH, kW)
    # -------------------------------------------------------------------------
    shader = []

    if has_bconv:
        shader.append("vec4 getResiduals(sampler2D bconv_tex, vec2 uv, vec2 tex_size_pixels) {")
    else:
        shader.append("vec4 getResiduals(sampler2D tex, vec4 which_channel, vec2 uv, vec2 tex_size_pixels) {")

    shader.append("    vec2 inv_res = 1.0 / tex_size_pixels;")
    shader.append(f"    int half_k = {k // 2};")
    shader.append("")

    # 1. Bake weights into const arrays
    add_array(shader, "conv_w", conv_w)
    add_array(shader, "conv_b", state["conv.bias"])

    for i in range(nm):
        add_array(shader, f"mid_{i}_w", mids_w[i])
        add_array(shader, f"mid_{i}_b", mids_b[i])

    add_array(shader, "zfinal_w", zfinal_w)
    add_array(shader, "zfinal_b", zfinal_b)
    shader.append("")

    # 2. Extract pixel neighbourhood
    if has_bconv:
        # Each sample from the bconv render target is a vec4
        shader.append(f"    vec4 neigh[{k*k}];")
        shader.append("    int idx = 0;")
        shader.append("    for (int dy = -half_k; dy <= half_k; dy++) {")
        shader.append("        for (int dx = -half_k; dx <= half_k; dx++) {")
        shader.append("            vec2 offset = vec2(float(dx), float(dy)) * inv_res;")
        shader.append("            neigh[idx++] = texture2DLod(bconv_tex, uv + offset, 0.0);")
        shader.append("        }")
        shader.append("    }")
    else:
        shader.append(f"    float neigh[{k*k}];")
        shader.append("    int idx = 0;")
        shader.append("    for (int dy = -half_k; dy <= half_k; dy++) {")
        shader.append("        for (int dx = -half_k; dx <= half_k; dx++) {")
        shader.append("            vec2 offset = vec2(float(dx), float(dy)) * inv_res;")
        shader.append("            neigh[idx++] = dot(texture2DLod(tex, uv + offset, 0.0), which_channel);")
        shader.append("        }")
        shader.append("    }")
    shader.append("")

    # 3. Layer 1 (Spatial Convolution)
    shader.append(f"    float h0[{c}];")
    shader.append("    int w_idx = 0;")
    shader.append(f"    for (int i=0; i<{c}; i++) {{")
    shader.append("        h0[i] = conv_b[i];")

    if has_bconv:
        # Weight layout: (out=c, in=4, kH, kW) -> iterate in_ch then spatial
        shader.append(f"        for (int in_ch=0; in_ch<{conv_in_ch}; in_ch++) {{")
        shader.append(f"            for (int j=0; j<{k*k}; j++) {{")
        shader.append("                h0[i] += neigh[j][in_ch] * conv_w[w_idx++];")
        shader.append("            }")
        shader.append("        }")
    else:
        shader.append(f"        for (int j=0; j<{k*k}; j++) {{")
        shader.append("            h0[i] += neigh[j] * conv_w[w_idx++];")
        shader.append("        }")

    act_str = apply_activation("h0[i]")
    if act_str: shader.append(act_str)
    shader.append("    }")
    shader.append("")

    # 4. Middle Layers (1x1 Convolutions) -- identical for both network types
    prev_h = "h0"
    prev_size = c
    for i in range(nm):
        curr_h = f"h{i+1}"
        shader.append(f"    float {curr_h}[{ms}];")
        shader.append("    w_idx = 0;")
        shader.append(f"    for (int i=0; i<{ms}; i++) {{")
        shader.append(f"        {curr_h}[i] = mid_{i}_b[i];")
        shader.append(f"        for (int j=0; j<{prev_size}; j++) {{")
        shader.append(f"            {curr_h}[i] += {prev_h}[j] * mid_{i}_w[w_idx++];")
        shader.append("        }")
        act_str = apply_activation(f"{curr_h}[i]")
        if act_str: shader.append(act_str)
        shader.append("    }")
        shader.append("")
        prev_h = curr_h
        prev_size = ms

    # 5. Final Layer (Output mapping) -- identical for both network types
    shader.append("    vec4 out_val = vec4(zfinal_b[0], zfinal_b[1], zfinal_b[2], zfinal_b[3]);")
    shader.append("    w_idx = 0;")
    shader.append("    for (int i=0; i<4; i++) {")
    shader.append(f"        for (int j=0; j<{prev_size}; j++) {{")
    shader.append(f"            out_val[i] += {prev_h}[j] * zfinal_w[w_idx++];")
    shader.append("        }")
    shader.append("    }")
    shader.append("")
    shader.append("    return out_val;")
    shader.append("}")

    parts.append("\n".join(shader))

    return "\n\n".join(parts)

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
