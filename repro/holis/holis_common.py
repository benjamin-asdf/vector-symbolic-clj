"""Shared helpers for the holis harnesses: my code only.

Instrumentation of their notebook is limited to:
  * probe noise: before every cleanup (SimpleCleanUp.recall, used for the
    item memory `mem` and the function memory `func`, and SimpleAssoc.recall,
    the pointer memory), add Gaussian noise of norm sigma * |probe|, as in my
    paper (scripts/paper_f2.py): p + sigma*|p|/sqrt(D) * N(0, I);
  * a budget on cons cells, so a run that noise sends into runaway
    evaluation stops and counts as a failure;
  * a counter of how often SimpleAssoc.recall falls back to returning the
    probe itself because no key passed its threshold theta = 0.2.
"""

import json
import math
import os
import platform
import sys
import time

import numpy as np

import upstream

H = upstream.load_holis()

_ORIG_MEM_RECALL = H.SimpleCleanUp.recall
_ORIG_ASSOC_RECALL = H.SimpleAssoc.recall

STATE = {"sigma": 0.0, "rng": np.random.default_rng(0), "fallbacks": 0,
         "assoc_noise": True}


def _noisy(hrr):
    s = STATE["sigma"]
    if s <= 0:
        return hrr
    v = hrr.v
    n = STATE["rng"].normal(size=v.size) * (s * np.linalg.norm(v) / math.sqrt(v.size))
    return H.HRR(data=v + n)


def _mem_recall(self, hrr):
    return _ORIG_MEM_RECALL(self, _noisy(hrr))


def _assoc_recall(self, probe):
    p = _noisy(probe) if STATE["assoc_noise"] else probe
    out = _ORIG_ASSOC_RECALL(self, p)
    if out is p and p not in self.A:
        STATE["fallbacks"] += 1
    return out


H.SimpleCleanUp.recall = _mem_recall
H.SimpleAssoc.recall = _assoc_recall


class Budget(Exception):
    pass


def set_noise(sigma, seed):
    STATE["sigma"] = float(sigma)
    STATE["rng"] = np.random.default_rng(10_000 + seed)
    STATE["fallbacks"] = 0


def new_env(dim, seed, theta=None, max_cells=None):
    """A fresh holis environment. Their code draws every vector from numpy's
    global RNG, so the seed fixes all of them. theta=None keeps their 0.2."""
    np.random.seed(seed)
    env = H.vsa_env(dim)
    H.global_env = env  # read() / lispexpr look this up in some paths
    if theta is not None:
        env["assoc"].theta = theta
    if max_cells is not None:
        assoc = env["assoc"]
        orig = assoc.memorize
        count = {"n": 0}

        def capped(probe, trace):
            count["n"] += 1
            if count["n"] > max_cells:
                raise Budget()
            return orig(probe, trace)

        assoc.memorize = capped
        env["_cells"] = count
    return env


def inner(env, name):
    """One of vsa_env's inner functions (car, cdr, ccons, atomic, ...), taken
    from the closure of an exposed one, so I call their code, not a copy."""
    seen, todo = set(), [env["CAR"], env["evall"], env["holo"]]
    while todo:
        f = todo.pop()
        if id(f) in seen or not callable(f) or not hasattr(f, "__code__"):
            continue
        seen.add(id(f))
        if f.__name__ == name:
            return f
        for cell in f.__closure__ or ():
            try:
                todo.append(cell.cell_contents)
            except ValueError:
                pass
    raise KeyError(name)


def run(env, src):
    """Evaluate every top-level form of src with their evaluator, read back
    each result with their reader, and return the list of readings."""
    return [env["read"](env["evall"](env["holo"](s))) for s in H.parse(src)]


def attempt(fn):
    """Run fn(); turn the ways a corrupted run can end into a label."""
    try:
        return {"value": fn(), "error": None}
    except Budget:
        return {"value": None, "error": "budget"}
    except RecursionError:
        return {"value": None, "error": "RecursionError"}
    except Exception as e:  # noqa: BLE001  (their code raises many kinds)
        return {"value": None, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def show(x):
    return json.dumps(x) if not isinstance(x, str) else x


def meta(script, settings):
    return {"script": script,
            "upstream": upstream.holis_provenance(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "date": time.strftime("%Y-%m-%d"),
            "settings": settings}


def write(name, obj):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("wrote", path, file=sys.stderr)
