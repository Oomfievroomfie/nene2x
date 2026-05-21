"""
gen.py — CuNNy shader generator (safetensors edition)

Replaces the original pickle-based loader with safetensors + JSON metadata.
Everything else (shader generation via mpv.py / magpie.py) is unchanged.

Usage is identical to the original:
    python gen.py <impl>  -m <model.safetensors> [...]

Where <impl> is one of:  mpv  magpie
"""

import argparse
import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import load_file
from safetensors import safe_open
from common import *


# ---------------------------------------------------------------------------
# safetensors loader
# ---------------------------------------------------------------------------

def load_model(path: str) -> dict:
    """Load a .safetensors CuNNy model.

    Returns a dict whose structure is identical to what the old pickle loader
    produced:
      - numpy arrays for every weight tensor (keyed by their original names)
      - scalar / list metadata decoded from the safetensors metadata header
    """
    m: dict = dict(load_file(path))   # {name: np.ndarray, ...}

    # The safetensors format stores metadata as {str: str}.
    # We JSON-encoded every non-array value at save time, so decode it back.
    with safe_open(path, framework="np") as f:
        raw_meta = f.metadata()       # {str: str} | None

    if raw_meta:
        for k, v in raw_meta.items():
            m[k] = json.loads(v)

    return m


# ---------------------------------------------------------------------------
# dynamic impl loader (unchanged from original)
# ---------------------------------------------------------------------------

def load_or_reload(name):
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def run(model, impl, extra, name=None):
    stem = Path(model).stem
    m = load_model(model)
    args = SimpleNamespace(**{
        'rgb':     m['rgb'],
        'quant':   m['quant'],
        'quant_8': m['quant-8'],
        'size':    m.get('size', 0),
        'stem':    name if name else stem,
        'name':    name if name else stem[:stem.rfind('-')],
        'extra':   extra,
    })
    init()
    return load_or_reload(impl).main(m, args, False)


def help(impl):
    return load_or_reload(impl).main(None, None, True)


# ---------------------------------------------------------------------------
# CLI (mirrors the original exactly)
# ---------------------------------------------------------------------------

if not sys.platform == 'emscripten':
    parser = argparse.ArgumentParser()
    parser.add_argument('impl', type=str,
                        help='shader backend to use, e.g. mpv or magpie')
    parser.add_argument('-m', '--model', action='append', type=str,
                        required=True,
                        help='path to a .safetensors model file '
                             '(may be repeated)')
    args, extra = parser.parse_known_args()
    extra = []
    for model in args.model:
        fp, shader = run(model, args.impl, extra)
        with open(f'test/{fp}', 'w') as f:
            f.write(shader)
