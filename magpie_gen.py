#!/usr/bin/env python3
"""magpie_gen.py — generate a MagpieFX shader for Magpie from a nene2x model.

Converts a nene2x safetensors upscaler into a multi-pass MagpieFX (DirectX 11
compute) HLSL effect that upscales a window 2x in real time.

Supported nene2x features (everything the model format supports that is
deterministic and shader-representable):

  * base upscalers: EDI (default), `b,` bilinear, `g,` blurred-bilinear,
    `j,` jinc, `x,` raw (no base)
  * `r,` 3-channel RGB models (12-channel pixel-shuffle output)
  * layer flags: `d` grouped/depthwise convs (both directions), `n` no bias,
    `q` dilated conv (dilation 2), arbitrary square kernel sizes
  * the `w` (activation noise) flag is *not* representable deterministically
    in a shader — it is skipped with a warning (no shipped model uses it)

Color handling: 1-channel models run on the luma of the YCgCo color space
(like `infer.py --ycgco --yonly`), with Cg/Co/alpha bilinearly upscaled.
3-channel (`r,`) models run on full RGB.

The shader is multi-pass: every non-1x1 convolution is its own pass, computing
up to 12 channels per pass (3 float4 textures). Weights and biases are inlined
in the pass bodies as min16float matrix ops (mul(v, min16float4x4(...))), like
the reference effects; nothing is baked into COMMON except the base-upscaler
helpers and the leaky-slope constant.

Usage:
  python magpie_gen.py pretrained/nene2x_medium_....safetensors -o nene2x_medium.hlsl

This generator was written from scratch for this project; it does not use any
code from other projects' shader generators.
"""

import argparse
import sys

import numpy as np
from safetensors import safe_open

import model as M  # for the jinc kernel builder (project's own function)

BASE_EDI, BASE_BILINEAR, BASE_BLUR, BASE_JINC, BASE_RAW = range(5)
BASE_NAMES = {BASE_EDI: "EDI", BASE_BILINEAR: "bilinear",
              BASE_BLUR: "blurred-bilinear", BASE_JINC: "jinc", BASE_RAW: "raw"}

CH = "rgba"


# ---------------------------------------------------------------------------
# model parsing (mirrors model.py, standalone — no tinygrad)
# ---------------------------------------------------------------------------

def parse_cfg_string(cfg):
    """'r,g,3x3_12,...' -> (specs, base_mode, is_3ch)."""
    is_3ch = cfg.startswith("r,")
    if is_3ch:
        cfg = cfg[2:]
    base = BASE_EDI
    if cfg.startswith("x,"):
        cfg = cfg[2:]
        base = BASE_RAW
    if cfg.startswith("b,"):
        cfg = cfg[2:]
        base = BASE_BILINEAR
    if cfg.startswith("g,"):
        cfg = cfg[2:]
        base = BASE_BLUR
    if cfg.startswith("j,"):
        cfg = cfg[2:]
        base = BASE_JINC
    specs = []
    for tok in cfg.split(","):
        tok = tok.strip()
        flags = set()
        while tok and tok[-1] in "dnqw":
            flags.add(tok[-1])
            tok = tok[:-1]
        kpart, c_str = tok.split("_")
        k = int(kpart.split("x")[0])
        specs.append((k, int(c_str), "d" in flags, "n" in flags,
                      "q" in flags, "w" in flags))
    expected = 12 if is_3ch else 4
    assert specs[-1][1] == expected, \
        f"final layer must output {expected} channels, got {specs[-1][1]}"
    return specs, base, is_3ch


def infer_from_state(state):
    """Infer specs/base/3ch from the state dict when no zzzgConfig_ key exists
    (mirrors model.UpscaleNet._config_from_state)."""
    for pfx, base in (("zfinalx", BASE_RAW), ("zfinalb", BASE_BILINEAR),
                      ("zfinalg", BASE_BLUR), ("zfinalj", BASE_JINC)):
        if f"{pfx}.weight" in state:
            final_key, final_pfx, base = f"{pfx}.weight", pfx, base
            break
    else:
        final_key, final_pfx, base = "zfinal.weight", "zfinal", BASE_EDI
    assert final_key in state, f"no {final_key} in state dict"

    keys = sorted(k for k in state if k.startswith("layers.") and k.endswith(".weight"))
    specs = []
    prev_out = 0
    first_in = None
    for i, k in enumerate(keys):
        w = state[k]
        pfx = k[:-len(".weight")]
        in_c = w.shape[1]
        if i == 0:
            first_in = in_c
            depthwise = False
        else:
            depthwise = in_c < prev_out and prev_out % in_c == 0
        prev_out = w.shape[0]
        no_bias = (pfx + ".bias") not in state
        specs.append((w.shape[2], w.shape[0], depthwise, no_bias, False, False))
    w = state[final_key]
    if first_in is None:
        first_in = w.shape[1]
    depthwise = len(keys) > 0 and w.shape[1] < prev_out and prev_out % w.shape[1] == 0
    no_bias = (final_pfx + ".bias") not in state
    specs.append((w.shape[2], w.shape[0], depthwise, no_bias, False, False))
    return specs, base, first_in == 3


def load_state(path):
    state = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def parse_model(state):
    cfg_key = next((k for k in state if k.startswith("zzzgConfig_")), None)
    if cfg_key is not None:
        return parse_cfg_string(cfg_key[len("zzzgConfig_"):])
    return infer_from_state(state)


# ---------------------------------------------------------------------------
# weight extraction: layers = [(k, w, b, dilation, noise, in_c, groups)]
# ---------------------------------------------------------------------------

def extract_layers(state, specs, is_3ch):
    in_c = 3 if is_3ch else 1
    layers = []
    for i, (k, out_c, depthwise, no_bias, dilated, noise) in enumerate(specs[:-1]):
        w = state[f"layers.{i}.weight"]
        b = state.get(f"layers.{i}.bias")
        g = (in_c if out_c >= in_c else out_c) if depthwise else 1
        assert w.shape == (out_c, in_c // g, k, k), \
            f"layer {i} weight shape {w.shape} != {(out_c, in_c // g, k, k)}"
        layers.append((k, w, b, 2 if dilated else 1, noise, in_c, g))
        in_c = out_c
    final_pfx = "zfinal"
    for pfx in ("zfinalx", "zfinalb", "zfinalg", "zfinalj"):
        if f"{pfx}.weight" in state:
            final_pfx = pfx
            break
    fw = state[f"{final_pfx}.weight"]
    fb = state.get(f"{final_pfx}.bias")
    k, out_c, depthwise, no_bias, dilated, noise = specs[-1]
    g = (in_c if out_c >= in_c else out_c) if depthwise else 1
    assert fw.shape == (out_c, in_c // g, k, k), \
        f"final weight shape {fw.shape} != {(out_c, in_c // g, k, k)}"
    return layers, (k, fw, fb, 2 if dilated else 1, noise, in_c, g)


# ---------------------------------------------------------------------------
# HLSL emission helpers
# ---------------------------------------------------------------------------

def fmt_float(v):
    return format(float(v), ".9g")


def fmt_w(v):
    """Compact weight literal (5 significant digits — matches min16float
    precision, same style as the reference effects' weight tables)."""
    v = float(v)
    return "0" if v == 0.0 else format(v, ".4e")


def bake_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    vals = ", ".join(fmt_float(v) for v in flat)
    return f"static const float {name}[{len(flat)}] = {{ {vals} }};"


def emit_pass(passno, pname, ins, outnames, wh, body):
    """body is a list of lines (already indented)."""
    out = []
    out.append(f"//!PASS {passno}")
    out.append(f"//!IN {', '.join(ins)}")
    out.append(f"//!OUT {', '.join(outnames)}")
    out.append("//!BLOCK_SIZE 8")
    out.append("//!NUM_THREADS 64")
    out.append(f"void Pass{passno}(uint2 blockStart, uint3 threadId) {{")
    out.append("    uint2 p = blockStart + Rmp8x8(threadId.x);")
    # The guard must bound THIS pass's OUT texture. GetInputSize()/GetOutputSize()
    # return the EFFECT's input/output sizes (W and 2W), not the per-pass texture
    # sizes — using them let every 1x pass write out of bounds (garbage residuals
    # + static). GetDimensions gives the pass texture's real size.
    out.append(f"    uint ow, oh; {outnames[0]}.GetDimensions(ow, oh);")
    out.append("    if (p.x >= ow || p.y >= oh) return;")
    out.extend(body)
    out.append("}")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Generate a MagpieFX (Magpie) 2x upscaling shader from a "
                    "nene2x safetensors model.")
    ap.add_argument("model", help="path to .safetensors model")
    ap.add_argument("--output", "-o", default=None, help="output .hlsl file (default: stdout)")
    ap.add_argument("--leaky-slope", type=float, default=0.0,
                    help="leaky ReLU slope (0 = pure ReLU, the default, matching "
                         "infer.py's default; the training default is 0.003)")
    args = ap.parse_args()

    state = load_state(args.model)
    specs, base, is_3ch = parse_model(state)
    layers, final = extract_layers(state, specs, is_3ch)

    noise_flags = [l[4] for l in layers] + [final[4]]
    if any(noise_flags):
        print("warning: model uses 'w' (activation noise) layers — noise is "
              "stochastic and cannot be represented in a shader; ignoring it "
              "(deterministic output)", file=sys.stderr)

    total_params = sum(int(np.asarray(state[k]).size) for k in state
                       if not k.startswith("filter_net_adv")
                       and not k.startswith("zzzgConfig"))

    shader = generate_shader(specs, layers, final, base, is_3ch, args.leaky_slope)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(shader)
        print(f"wrote {args.output} ({total_params:,} params, "
              f"{shader.count('//!PASS')} passes, "
              f"{len(shader.splitlines()):,} lines)", file=sys.stderr)
    else:
        sys.stdout.write(shader)


def generate_shader(specs, layers, final, base, is_3ch, slope):
    LRW, LRH = "INPUT_WIDTH", "INPUT_HEIGHT"
    S2 = (f"{LRW} * 2", f"{LRH} * 2")
    cfg = specs_to_cfg(specs, base, is_3ch)

    # The reference (model.py/infer.py) pads the input once by PAD (edge) and
    # runs valid, padding-free convs, so its feature maps shrink per layer.
    # The shader mirrors that exactly: PassPad writes the edge-replicated
    # input on a padded grid, each conv pass runs valid (non-centered taps,
    # unclamped reads — every read is in range because each layer's texture is
    # sized to its remaining pad), and the final map lands exactly INPUT-sized.
    PAD = sum((k - 1) // 2 * dil for (k, w, b, dil, n, in_c, g) in layers if k > 1) \
        + ((final[0] - 1) // 2 * final[3] if final[0] > 1 else 0)
    NPW, NPH = f"{LRW} + {2 * PAD}", f"{LRH} + {2 * PAD}"

    # ------------------------------------------------------------------
    # pass / texture plan
    # ------------------------------------------------------------------
    tex_defs = []          # (name, width_expr, height_expr)
    passes = []            # (name, [ins], out, wh)

    def add_tex(name, wh):
        tex_defs.append((name, wh[0], wh[1]))
        return name

    if not is_3ch:
        tex_ycgco = add_tex("texYcgCo", (LRW, LRH))   # unpadded, for chroma/alpha
        tex_ypad = add_tex("texYPad", (NPW, NPH))     # padded luma, network input
        tex_inpad = None
    else:
        tex_ycgco = None
        tex_ypad = None
        tex_inpad = add_tex("texInPad", (NPW, NPH))   # padded RGB, network input
    tex_blur = add_tex("texBlurY", (NPW, NPH)) if base == BASE_BLUR else None

    if not is_3ch:
        passes.append(("PassYcgCo", ["INPUT"], [tex_ycgco], (LRW, LRH),
                       ycgco_body(tex_ycgco)))
    passes.append(("PassPad", ["INPUT"],
                   [tex_ypad if not is_3ch else tex_inpad], (NPW, NPH),
                   pad_body(tex_ypad if not is_3ch else tex_inpad, is_3ch, PAD)))
    if tex_blur:
        passes.append(("PassBlur", [tex_ypad if not is_3ch else tex_inpad],
                       [tex_blur], (NPW, NPH),
                       blur_body(tex_blur, is_3ch, PAD)))

    prev_reads = [tex_inpad if is_3ch else tex_ypad]
    for li, (k, w, b, dilation, noise, in_c, g) in enumerate(layers):
        out_c = w.shape[0]
        # all network maps live on the full padded grid; reads are clamped, so
        # every texel is always written and no read can go out of bounds
        names = [add_tex(f"texL{li}_{j}", (NPW, NPH))
                 for j in range((out_c + 3) // 4)]
        for o0 in range(0, out_c, 12):
            outs = names[o0 // 4:(o0 + 12) // 4]
            body = conv_pass_body(k, w, b, dilation, in_c, g, o0,
                                  prev_reads, hidden=True, out_names=outs, slope=slope)
            passes.append((f"PassL{li}_{o0 // 12}",
                           list(dict.fromkeys(prev_reads)), outs, (NPW, NPH), body))
        prev_reads = [names[c // 4] for c in range(out_c)]

    fw = final[1]
    fout_c = fw.shape[0]
    resid = [add_tex("texResid" if fout_c == 4 else f"texResid{j}", (NPW, NPH))
             for j in range((fout_c + 3) // 4)]
    for o0 in range(0, fout_c, 12):
        outs = resid[o0 // 4:(o0 + 12) // 4]
        body = conv_pass_body(final[0], fw, final[2], final[3],
                              final[5], final[6], o0, prev_reads,
                              hidden=False, out_names=outs, slope=slope)
        passes.append((f"PassFinal{o0 // 12}", list(dict.fromkeys(prev_reads)),
                       outs, (NPW, NPH), body))

    tex_base = add_tex("texBase", S2)
    base_ins = list(dict.fromkeys([tex_ycgco] if not is_3ch else ["INPUT"]))
    if tex_ypad:
        base_ins.append(tex_ypad)
    if tex_inpad:
        base_ins.append(tex_inpad)
    if tex_blur:
        base_ins.append(tex_blur)
    passes.append(("PassBase", base_ins, [tex_base], S2,
                   base_body(base, is_3ch, tex_ycgco, tex_ypad, tex_inpad, tex_blur, PAD)))
    passes.append(("PassCombine", [tex_base] + resid, ["OUTPUT"], S2,
                   combine_body(is_3ch, resid)))

    # ------------------------------------------------------------------
    # emit
    # ------------------------------------------------------------------
    L = []
    A = L.append
    A("//!MAGPIE EFFECT")
    A("//!VERSION 4")
    A("// MagpieFX shader generated by magpie_gen.py (nene2x project's own code).")
    A(f"// config: {cfg}")
    A(f"// base upscaler: {BASE_NAMES[base]}   channels: {'RGB' if is_3ch else 'YCgCo luma'}   "
      f"activation: {'ReLU' if slope == 0.0 else 'leaky ' + fmt_float(slope)}")
    A("//!TEXTURE")
    A("Texture2D INPUT;")
    A("//!TEXTURE")
    A(f"//!WIDTH {S2[0]}")
    A(f"//!HEIGHT {S2[1]}")
    A("Texture2D OUTPUT;")
    for name, w, h in tex_defs:
        A("//!TEXTURE")
        A(f"//!WIDTH {w}")
        A(f"//!HEIGHT {h}")
        # Half-precision intermediates (the standard MagpieFX choice): fp16
        # storage halves texture memory/bandwidth; loads convert to fp32.
        A("//!FORMAT R16G16B16A16_FLOAT")
        A(f"Texture2D {name};")
    A("")
    A("//!COMMON")
    if slope != 0.0:
        A(f"static const float SLOPE = {fmt_float(slope)};")
    if base == BASE_JINC:
        for nm, dx, dy in (("jk00", -0.25, -0.25), ("jk10", 0.25, -0.25),
                           ("jk01", -0.25, 0.25), ("jk11", 0.25, 0.25)):
            A(bake_array(nm, M._build_jinc_kernel_offset(4, dx, dy)))
    A("")
    A(common_fns(base, is_3ch))
    A("")

    passno = 0
    for pname, ins, outnames, wh, body in passes:
        passno += 1
        L.extend(emit_pass(passno, pname, ins, outnames, wh, body))
    return "\n".join(L)


def specs_to_cfg(specs, base, is_3ch):
    def tok(k, c, dw, nb, dil, noise):
        return f"{k}x{k}_{c}{'d' if dw else ''}{'n' if nb else ''}" \
               f"{'q' if dil else ''}{'w' if noise else ''}"
    body = ",".join(tok(*s) for s in specs)
    if base == BASE_JINC:
        body = "j," + body
    elif base == BASE_BLUR:
        body = "g," + body
    elif base == BASE_BILINEAR:
        body = "b," + body
    elif base == BASE_RAW:
        body = "x," + body
    if is_3ch:
        body = "r," + body
    return body


# ---------------------------------------------------------------------------
# pass body emitters
# ---------------------------------------------------------------------------

def ycgco_body(tex_ycgco):
    return [
        "    float4 c = INPUT.Load(int3(int2(p), 0));",
        "    float y = 0.25*c.r + 0.5*c.g + 0.25*c.b;",
        "    float cg = -0.25*c.r + 0.5*c.g - 0.25*c.b + 0.5;",
        "    float co = 0.5*c.r - 0.5*c.b + 0.5;",
        f"    {tex_ycgco}[p] = float4(y, cg, co, c.a);",
    ]


def conv_pass_body(k, w, b, dilation, in_c, g, o0, prev_reads, hidden, out_names, slope=0.0):
    """Emit one conv pass computing up to 12 output channels (3 float4
    textures). All network maps live on the full padded grid; taps are
    non-centered (output s reads input s + ky*dilation, matching the
    reference's valid convs) with CLAMPED reads, so every texel is written and
    no read can go out of bounds. For every output pixel (p < W) the clamped
    reads never fire and the values match infer.py exactly. Weights are
    inlined as min16float matrix ops."""
    out_c = w.shape[0]
    per = in_c // g
    body = [f"    uint cw, ch; {out_names[0]}.GetDimensions(cw, ch);",
            "    int2 SZ = int2((int)cw, (int)ch);"]

    def wt(ky, kx, o, ic):
        # outputs past out_c don't exist (partial last group): zero weight
        return fmt_w(w[o, ic, ky, kx]) if o < out_c else "0"

    for og, out_tex in enumerate(out_names):
        oa = o0 + 4 * og
        # bias for the (possibly partial) group of 4; missing channels -> 0
        vals = []
        for j in range(4):
            if b is not None and oa + j < out_c:
                vals.append(fmt_float(b[oa + j]))
            else:
                vals.append("0.0")
        body.append(f"    min16float4 r{og} = min16float4({', '.join(vals)});")

    def m4(ky, kx, oa, in_base, span):
        """min16float4x4 constructor args: matrix row i = input in_base+i,
        col j = output oa+j (mul treats the sample as a row vector). The
        constructor lists rows in order, so arg (4*i+j) = w[oa+j][in_base+i]."""
        vals = []
        for i in range(4):
            for j in range(4):
                ic = in_base + i
                vals.append(wt(ky, kx, oa + j, ic) if ic < span else "0")
        return "min16float4x4(" + ", ".join(vals) + ")"

    def tap_lines(ky, kx, tap, ind):
        ox = kx * dilation
        oy = ky * dilation
        lines = []
        if g == 1:
            ntex = (in_c + 3) // 4
            for ti in range(ntex):
                lines.append(f"{ind}min16float4 pv{tap}_{ti} = (min16float4)"
                             f"LoadClamp({prev_reads[4 * ti]}, int2(p) + int2({ox}, {oy}), SZ);")
            for og, _tex in enumerate(out_names):
                oa = o0 + 4 * og
                for ti in range(ntex):
                    lines.append(f"{ind}r{og} += mul(pv{tap}_{ti}, {m4(ky, kx, oa, 4 * ti, in_c)});")
        else:
            # grouped/depthwise: each output channel accumulates its own
            # group's slice (out_c//g can be < 4, e.g. 3x3_20d -> 1: every
            # output reads a different input channel). Load each distinct
            # (texture, component) once per tap and reuse across outputs.
            g_per_out = out_c // g
            for og, _tex in enumerate(out_names):
                oa = o0 + 4 * og
                users = {}
                for j in range(4):
                    o = oa + j
                    if o >= out_c:
                        break
                    grp = o // g_per_out
                    for ic in range(per):
                        pc = grp * per + ic
                        users.setdefault((prev_reads[pc], pc % 4), []).append((o, ic))
                load_vars = {}
                for li2, ((t, comp), _u) in enumerate(users.items()):
                    vn = f"pv{tap}_{og}_{li2}"
                    load_vars[(t, comp)] = vn
                    lines.append(f"{ind}min16float4 {vn} = (min16float4)"
                                 f"LoadClamp({t}, int2(p) + int2({ox}, {oy}), SZ);")
                for j in range(4):
                    o = oa + j
                    if o >= out_c:
                        break
                    grp = o // g_per_out
                    terms = []
                    for ic in range(per):
                        pc = grp * per + ic
                        vn = load_vars[(prev_reads[pc], pc % 4)]
                        terms.append(f"{wt(ky, kx, o, ic)} * {vn}.{CH[pc % 4]}")
                    lines.append(f"{ind}r{og}.{CH[j]} += {' + '.join(terms)};")
        return lines

    tap = 0
    for ky in range(k):
        for kx in range(k):
            body.extend(tap_lines(ky, kx, tap, "    "))
            tap += 1
    if hidden:
        if slope == 0.0:
            for og in range(len(out_names)):
                body.append(f"    r{og} = max(r{og}, 0.0);")
        else:
            for og in range(len(out_names)):
                body.append(f"    r{og} = max(r{og}, SLOPE * r{og});")
    for og, out_tex in enumerate(out_names):
        body.append(f"    {out_tex}[p] = r{og};")
    return body


def pad_body(tex_out, is_3ch, pad):
    """The reference's single edge pad of the input tensor: writes the
    edge-replicated input (YCgCo-converted for 1ch) at every padded position.
    This is the ONLY edge duplication in the whole pipeline — exactly where
    model.py/infer.py pad."""
    body = [
        f"    float4 c = LoadClamp(INPUT, int2(p) - int2({pad}, {pad}), "
        "int2((int)GetInputSize().x, (int)GetInputSize().y));",
    ]
    if not is_3ch:
        body += [
            "    float y = 0.25*c.r + 0.5*c.g + 0.25*c.b;",
            "    float cg = -0.25*c.r + 0.5*c.g - 0.25*c.b + 0.5;",
            "    float co = 0.5*c.r - 0.5*c.b + 0.5;",
            f"    {tex_out}[p] = float4(y, cg, co, c.a);",
        ]
    else:
        body.append(f"    {tex_out}[p] = c;")
    return body


def blur_body(tex_blur, is_3ch, pad):
    """Pre-blur pass over the full padded grid: the reference's box_blur5x5
    runs on the padded input, so the blur (with its own internal edge padding,
    i.e. clamped reads) is computed at every padded position — including the
    virtual edge positions the base's outer taps need."""
    src = "INPUT" if is_3ch else "texYPad"
    body = [f"    uint bw, bh; {tex_blur}.GetDimensions(bw, bh);"]
    body.append("    int2 SZ = int2((int)bw, (int)bh);")
    for c in range(3 if is_3ch else 1):
        body.append(f"    float b{c} = Blur({src}, SZ, int2(p), {c});")
    if is_3ch:
        body.append(f"    {tex_blur}[p] = float4(b0, b1, b2, 0.0);")
    else:
        body.append(f"    {tex_blur}[p] = float4(b0, 0.0, 0.0, 0.0);")
    return body


def base_body(base, is_3ch, tex_ycgco, tex_ypad, tex_inpad, tex_blur, pad):
    # The 2x pass is written in terms of GetOutputSize() (the 2x intermediate's
    # own size). The luma base reads the PADDED textures at padded 2x position
    # pp = p + 2*PAD (the reference computes the base on the padded input and
    # crops by 2*PAD); the chroma/alpha use infer.py's extrapolating bilinear
    # on the unpadded texYcgCo / INPUT at p.
    d = 2 * pad
    body = [
        f"    int2 PSZ = int2((int)GetOutputSize().x / 2 + {d}, (int)GetOutputSize().y / 2 + {d});",
        f"    uint2 pp = p + uint2({d}, {d});",
    ]
    if not is_3ch:
        body.append(f"    float4 ycg = BilinearExtrap({tex_ycgco}, "
                     "int2((int)GetOutputSize().x / 2, (int)GetOutputSize().y / 2), p);")
        if base == BASE_BILINEAR:
            body.append(f"    float bY = BilinearBase({tex_ypad}, PSZ, 0, pp);")
        elif base == BASE_BLUR:
            # PassBlur precomputed the blur at every padded position, so the
            # g, base is plain bilinear taps over the pre-blurred texture.
            body.append(f"    float bY = BilinearBase({tex_blur}, PSZ, 0, pp);")
        elif base == BASE_JINC:
            body.append(f"    float bY = Jinc_Base({tex_ypad}, PSZ, 0, pp);")
        elif base == BASE_EDI:
            body.append(f"    float bY = EDI_Base({tex_ypad}, PSZ, 0, pp);")
        else:
            body.append("    float bY = 0.0;")
        body.append("    texBase[p] = float4(bY, ycg.g, ycg.b, ycg.a);")
    else:
        src = tex_inpad
        if base == BASE_BILINEAR:
            for c in range(3):
                body.append(f"    float b{c} = BilinearBase({src}, PSZ, {c}, pp);")
        elif base == BASE_BLUR:
            for c in range(3):
                body.append(f"    float b{c} = BilinearBase({tex_blur}, PSZ, {c}, pp);")
        elif base == BASE_JINC:
            for c in range(3):
                body.append(f"    float b{c} = Jinc_Base({src}, PSZ, {c}, pp);")
        elif base == BASE_EDI:
            for c in range(3):
                body.append(f"    float b{c} = EDI_Base({src}, PSZ, {c}, pp);")
        else:
            for c in range(3):
                body.append(f"    float b{c} = 0.0;")
        body.append(f"    float4 a4 = BilinearExtrap(INPUT, "
                     "int2((int)GetOutputSize().x / 2, (int)GetOutputSize().y / 2), p);")
        body.append("    texBase[p] = float4(b0, b1, b2, a4.a);")
    return body


def combine_body(is_3ch, resid):
    body = []
    if not is_3ch:
        body += [
            "    uint2 lr = p >> 1;",
            "    int c = 2 * (int)(p.y & 1u) + (int)(p.x & 1u);",
            f"    float4 r4 = {resid[0]}.Load(int3(int2(lr), 0));",
            "    float rv = c == 0 ? r4.r : (c == 1 ? r4.g : (c == 2 ? r4.b : r4.a));",
            "    float4 base4 = texBase.Load(int3(int2(p), 0));",
            "    float y = clamp(base4.r + rv, 0.0, 1.0);",
            "    float cg = base4.g - 0.5;",
            "    float co = base4.b - 0.5;",
            "    float tmp = y - cg;",
            "    float r = clamp(tmp + co, 0.0, 1.0);",
            "    float g = clamp(y + cg, 0.0, 1.0);",
            "    float b = clamp(tmp - co, 0.0, 1.0);",
            "    OUTPUT[p] = float4(r, g, b, base4.a);",
        ]
    else:
        body += [
            "    uint2 lr = p >> 1;",
            "    int c = 2 * (int)(p.y & 1u) + (int)(p.x & 1u);",
        ]
        for co in range(3):
            body.append(
                f"    float r{co} = {resid[co]}.Load(int3(int2(lr), 0))[c];")
        body += [
            "    float4 base4 = texBase.Load(int3(int2(p), 0));",
            "    float r = clamp(base4.r + r0, 0.0, 1.0);",
            "    float g = clamp(base4.g + r1, 0.0, 1.0);",
            "    float b = clamp(base4.b + r2, 0.0, 1.0);",
            "    OUTPUT[p] = float4(r, g, b, base4.a);",
        ]
    return body


# shared helper functions emitted into COMMON (only the pieces the model's
# base mode needs — unused functions must not reference undefined symbols like
# the jinc kernels, or every pass fails to compile)
FNS_LOADCLAMP = """\
float4 LoadClamp(Texture2D<float4> tex, int2 pos, int2 size) {
    pos = clamp(pos, int2(0, 0), size - 1);
    return tex.Load(int3(pos, 0));
}
"""
FNS_BILINEAR = """\
float BilinearBase(Texture2D<float4> tex, int2 size, int comp, uint2 p) {
    int x = (int)p.x, y = (int)p.y;
    int x0 = (p.x & 1u) == 0u ? x / 2 - 1 : x / 2;
    int x1 = (p.x & 1u) == 0u ? x / 2 : x / 2 + 1;
    int y0 = (p.y & 1u) == 0u ? y / 2 - 1 : y / 2;
    int y1 = (p.y & 1u) == 0u ? y / 2 : y / 2 + 1;
    float wx0 = (p.x & 1u) == 0u ? 0.25 : 0.75;
    float wx1 = (p.x & 1u) == 0u ? 0.75 : 0.25;
    float wy0 = (p.y & 1u) == 0u ? 0.25 : 0.75;
    float wy1 = (p.y & 1u) == 0u ? 0.75 : 0.25;
    int2 c0 = clamp(int2(x0, y0), int2(0, 0), size - 1);
    int2 c1 = clamp(int2(x1, y0), int2(0, 0), size - 1);
    int2 c2 = clamp(int2(x0, y1), int2(0, 0), size - 1);
    int2 c3 = clamp(int2(x1, y1), int2(0, 0), size - 1);
    return wy0*wx0*LoadClamp(tex, c0, size)[comp] + wy0*wx1*LoadClamp(tex, c1, size)[comp]
         + wy1*wx0*LoadClamp(tex, c2, size)[comp] + wy1*wx1*LoadClamp(tex, c3, size)[comp];
}
"""
FNS_BLUR = """\
float Blur1(Texture2D<float4> tex, int2 size, int2 pos, int comp) {
    return 0.25*LoadClamp(tex, pos + int2(0, -1), size)[comp]
         + 0.5 *LoadClamp(tex, pos, size)[comp]
         + 0.25*LoadClamp(tex, pos + int2(0, 1), size)[comp];
}
float Blur(Texture2D<float4> tex, int2 size, int2 pos, int comp) {
    return 0.25*Blur1(tex, size, pos + int2(-1, 0), comp)
         + 0.5 *Blur1(tex, size, pos, comp)
         + 0.25*Blur1(tex, size, pos + int2(1, 0), comp);
}
"""
FNS_BILINEAREXTRAP = """\
float4 BilinearExtrap(Texture2D<float4> tex, int2 size, uint2 p) {
    // Chroma/alpha upsample: infer.py's bil_up interpolates with tap weights
    // that go outside [0,1] at the border (linear extrapolation), so border
    // pixels do NOT replicate the edge texel — clamping them would make the
    // edge look clamped one texel early.
    float2 q = (float2(p) - 0.5) * 0.5;
    int2 f = (int2)floor(q);
    int2 i0 = clamp(f, int2(0, 0), size - 2);
    int2 i1 = i0 + 1;
    float2 t = q - (float2)i0;
    float4 a = tex.Load(int3(i0, 0));
    float4 b = tex.Load(int3(int2(i1.x, i0.y), 0));
    float4 c = tex.Load(int3(int2(i0.x, i1.y), 0));
    float4 d = tex.Load(int3(i1, 0));
    return a*(1.0-t.x)*(1.0-t.y) + b*t.x*(1.0-t.y) + c*(1.0-t.x)*t.y + d*t.x*t.y;
}
"""
FNS_EDI = """\
float EDI_Bilinear(Texture2D<float4> tex, int2 size, int2 pos, int comp) {
    // pos is in 2x OUTPUT space (up to 2*size-1), so clamp to the 2x bounds —
    // clamping to size-1 here force-clamped reads to the LR texture edge for
    // every pixel past the top-left quadrant (the "clamped copies" bug).
    pos = clamp(pos, int2(0, 0), size * 2 - 1);
    float ux = ((float)pos.y + 0.5) * 0.5;   // row
    float uy = ((float)pos.x + 0.5) * 0.5;   // col
    int ix = (int)floor(ux - 0.5) + 1;       // row index
    int iy = (int)floor(uy - 0.5) + 1;       // col index
    float s_a = LoadClamp(tex, int2(iy - 1, ix - 1), size)[comp];
    float s_b = LoadClamp(tex, int2(iy - 1, ix),     size)[comp];
    float s_c = LoadClamp(tex, int2(iy,     ix - 1), size)[comp];
    float s_d = LoadClamp(tex, int2(iy,     ix),     size)[comp];
    float tx = ux - (float)ix + 0.5;         // row weight
    float ty = uy - (float)iy + 0.5;         // col weight
    return s_a*(1.0-tx)*(1.0-ty) + s_b*tx*(1.0-ty) + s_c*(1.0-tx)*ty + s_d*tx*ty;
}
float EDI_Base(Texture2D<float4> tex, int2 size, int comp, uint2 p) {
    float ux = ((float)p.y + 0.5) * 0.5;   // row
    float uy = ((float)p.x + 0.5) * 0.5;   // col
    int ix = (int)floor(ux - 0.5) + 1;     // row index
    int iy = (int)floor(uy - 0.5) + 1;     // col index
    float s_a = LoadClamp(tex, int2(iy - 1, ix - 1), size)[comp];
    float s_b = LoadClamp(tex, int2(iy - 1, ix),     size)[comp];
    float s_c = LoadClamp(tex, int2(iy,     ix - 1), size)[comp];
    float s_d = LoadClamp(tex, int2(iy,     ix),     size)[comp];
    float tx = ux - (float)ix + 0.5;       // row weight
    float ty = uy - (float)iy + 0.5;       // col weight
    float dd = clamp((abs(s_c - s_b) - abs(s_d - s_a)) * 8.0, -1.0, 1.0) * 0.5 + 0.5;
    float iBS = (tx + ty < 1.0)
        ? s_c*ty + s_b*tx + s_a*(1.0 - (tx + ty))
        : s_c*(1.0-tx) + s_b*(1.0-ty) + s_d*(1.0 - ((1.0-tx) + (1.0-ty)));
    float iFS = (tx > ty)
        ? s_a*(1.0-tx) + s_d*ty + s_b*(1.0 - ((1.0-tx) + ty))
        : s_a*(1.0-ty) + s_d*tx + s_c*(1.0 - (tx + (1.0-ty)));
    float bil = EDI_Bilinear(tex, size, int2(p), comp);
    float ev = (round(dd) > 0.5) ? iFS : iBS;
    float mix = abs(dd - 0.5) * 2.0;
    float resv = (1.0 - mix) * bil + mix * ev;
    float hf = 0.5*(2.0/3.0)*bil - 0.5/6.0*(EDI_Bilinear(tex, size, int2(p) + int2(0, -2), comp)
        + EDI_Bilinear(tex, size, int2(p) + int2(-2, 0), comp)
        + EDI_Bilinear(tex, size, int2(p) + int2(0, 2), comp)
        + EDI_Bilinear(tex, size, int2(p) + int2(2, 0), comp));
    return resv + hf;
}
"""
FNS_JINC = """\
float Jinc_Base(Texture2D<float4> tex, int2 size, int comp, uint2 p) {
    int i = (int)(p.x >> 1), j = (int)(p.y >> 1);
    float v = 0.0;
    uint2 ph = p & 1u;
    if (ph.y == 0u && ph.x == 0u) {
        [unroll] for (int a = 0; a < 8; a++) [unroll] for (int b = 0; b < 8; b++)
            v += jk11[a][b] * LoadClamp(tex, int2(i + 4 - b, j + 4 - a), size)[comp];
    } else if (ph.y == 1u && ph.x == 0u) {
        [unroll] for (int a = 0; a < 8; a++) [unroll] for (int b = 0; b < 8; b++)
            v += jk10[a][b] * LoadClamp(tex, int2(i + 4 - b, j + 4 - a), size)[comp];
    } else if (ph.y == 0u && ph.x == 1u) {
        [unroll] for (int a = 0; a < 8; a++) [unroll] for (int b = 0; b < 8; b++)
            v += jk01[a][b] * LoadClamp(tex, int2(i + 4 - b, j + 4 - a), size)[comp];
    } else {
        [unroll] for (int a = 0; a < 8; a++) [unroll] for (int b = 0; b < 8; b++)
            v += jk00[a][b] * LoadClamp(tex, int2(i + 4 - b, j + 4 - a), size)[comp];
    }
    return v;
}
"""


def common_fns(base, is_3ch):
    """Only the helper functions this model's base mode actually needs.
    BilinearExtrap (chroma/alpha) and LoadClamp are always needed."""
    parts = [FNS_LOADCLAMP, FNS_BILINEAREXTRAP]
    if base == BASE_EDI:
        parts.append(FNS_EDI)
    elif base == BASE_BILINEAR:
        parts.append(FNS_BILINEAR)
    elif base == BASE_BLUR:
        # PassBlur needs Blur/Blur1; the g, base is BilinearBase over the
        # pre-blurred padded texture
        parts.append(FNS_BLUR)
        parts.append(FNS_BILINEAR)
    elif base == BASE_JINC:
        parts.append(FNS_JINC)
    return "\n".join(parts)


if __name__ == "__main__":
    main()
