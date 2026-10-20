"""Shared helpers for the residuelisp harnesses: our code only.

Instrumentation of their package is limited to:
  * silencing their debug output: the module-level print in syntax.parser,
    language.encoding and language.interpreter becomes a no-op, and the
    interpreter's debug f-strings, which call decode() on every evaluate
    step, get a stub decode while a program runs (the real decode reads the
    final result). decode is only used inside those log messages;
  * resetting RHC's class-level state (z_ms, codebook) per run, so a seed
    fixes every vector;
  * probe noise, as in holis_common: before every CleanupMemory.recall and
    every nearest-key scan of AssociativeMemory.deref, add complex Gaussian
    noise of norm sigma * |probe|.
"""

import json
import math
import os
import platform
import sys
import time

import numpy as np

import upstream

R = upstream.load_residuelisp()
I, E, V, S = R.interpreter, R.encoding, R.vsa, R.syntax


def _quiet(*a, **k):
    return None


for _m in (sys.modules["syntax.parser"], E, I):
    _m.print = _quiet
REAL_DECODE = I.decode

_ORIG_RECALL = E.CleanupMemory.recall
_ORIG_DEREF = E.AssociativeMemory.deref
STATE = {"sigma": 0.0, "rng": np.random.default_rng(0), "steps": 0,
         "max_steps": None}


class Budget(Exception):
    """A run made more cleanups than its budget: under noise their
    tuple_to_list can loop forever (it stops only when car raises)."""


def _noisy(x):
    s = STATE["sigma"]
    if s <= 0:
        return x
    d = np.asarray(x.data)  # FHRR, or RHC for a fresh integer: both complex
    n = (STATE["rng"].normal(size=d.size) + 1j * STATE["rng"].normal(size=d.size)) / math.sqrt(2)
    n *= s * np.linalg.norm(d) / math.sqrt(d.size)
    return V.FHRR(d + n)


def _tick():
    STATE["steps"] += 1
    if STATE["max_steps"] is not None and STATE["steps"] > STATE["max_steps"]:
        raise Budget()


def _recall(self, x):
    _tick()
    return _ORIG_RECALL(self, _noisy(x))


def _deref(self, ptr):
    # under noise the probe is never an exact key, so deref always scans
    _tick()
    return _ORIG_DEREF(self, _noisy(ptr))


E.CleanupMemory.recall = _recall
E.AssociativeMemory.deref = _deref


def set_noise(sigma, seed):
    STATE["sigma"] = float(sigma)
    STATE["rng"] = np.random.default_rng(20_000 + seed)


def reset(seed):
    np.random.seed(seed)
    V.RHC.z_ms = []
    V.RHC.codebook = {}


def encode_env(dim, rhc=True):
    scheme = (E.IntegerEncodingScheme.RHCIntegers if rhc
              else E.IntegerEncodingScheme.ListIntegers)
    return E.EncodingEnvironment(vsa=V.FHRR, dim=dim, integer_encoding_scheme=scheme)


def parse(src):
    return S.parse(S.lex(src))


def evaluate(src, dim=1000, seed=0, rhc=True, sigma=0.0, noisy_decode=True,
             max_steps=None):
    """Their interpret(), split so noise can be switched on for evaluation.
    Returns (decoded result, encoding env, eval env, raw result vector)."""
    reset(seed)
    set_noise(0.0, seed)
    enc = encode_env(dim, rhc)
    rep = E.encode(parse(src), enc)
    ev = I.EvalEnvironment(E.AssociativeMemory(enc.vsa, enc.dim), None)
    set_noise(sigma, seed)
    STATE["steps"], STATE["max_steps"] = 0, max_steps
    I.decode = lambda *a, **k: "<log>"
    try:
        res = I.evaluate(rep, enc, ev)
        I.decode = REAL_DECODE          # decode recurses through the module name
        if not noisy_decode:
            set_noise(0.0, seed)
        out = REAL_DECODE(res, enc, ev)
    finally:
        I.decode = REAL_DECODE
        STATE["max_steps"] = None
        set_noise(0.0, seed)
    return out, enc, ev, res


def attempt(fn):
    try:
        return {"value": fn(), "error": None}
    except Budget:
        return {"value": None, "error": "budget"}
    except RecursionError:
        return {"value": None, "error": "RecursionError"}
    except Exception as e:  # noqa: BLE001
        return {"value": None, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def show(x):
    return x if isinstance(x, str) else json.dumps(x)


def meta(script, settings):
    return {"script": script,
            "upstream": upstream.residuelisp_provenance(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "date": time.strftime("%Y-%m-%d"),
            "settings": settings}


def write(name, obj):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path, file=sys.stderr)
