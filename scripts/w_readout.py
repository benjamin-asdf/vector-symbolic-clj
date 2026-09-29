"""W3: world-readout accuracy vs number of worlds vs D, on the W1 substrate.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/w_readout.py [--quick]

Uses resources/vsc/hdc.py as the interpreter does (Proj memory, ProjCleanup
with exact cell fields, the `readout` behind `worlds`), without the JVM.
Four readouts of a superposition of k worlds with weights ~ U(0.5, 1.5)
(spread 3 : 1), normalised:

  pointer   x = Σ w p over k list pointers, read directly
  first     (first x): one cell read through the memory, then read
  int       x = Σ w B^n over k integers in [-1000, 1000], read directly
  int-env   an integer superposition stored in an environment cell and read
            back (what `let` does), then read

A trial is exact when the readout finds exactly the true worlds; TV is the
total variation distance of the weights. Writes out/w-readout.csv and
figures/w-readout.png.
"""

import csv
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "resources", "vsc"))
import hdc  # noqa: E402

OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")

ATOMS, LISTS = 1000, 1500
KS_PTR = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
KS_INT = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]
DIMS = [1024, 2048, 4096, 8192]
TRIALS = 20


class World:
    """A memory laid out like the interpreter's: atoms, and 2-element lists
    as pointer cells whose traces are stored unnormalised (gains)."""

    def __init__(self, dim, seed):
        self.s = hdc.Space(dim, seed)
        self.mem = hdc.make_memory(self.s, "proj")
        self.L, self.R = self.s.unitary(), self.s.unitary()
        self.nums = hdc.Numbers(self.s)
        self.C = hdc.make_cleanup(self.s, self.mem, self.nums)
        self.C.set_roles(self.L, self.R)
        self.C.set_int_range(-1024, 1024)
        self.empty = self.mem.get(self.mem.add(self.s.unitary(), 10))
        self.atoms = [self.mem.add(self.s.unitary(), 1) for _ in range(ATOMS)]
        self.sym = self.mem.get(self.atoms[0])
        rng = np.random.default_rng(seed)
        self.lists = []
        for _ in range(LISTS):
            a, b = rng.choice(self.atoms[1:], 2, replace=False)
            p = self.cell(self.mem.get(b), self.empty)
            p = self.cell(self.mem.get(a), p)
            self.lists.append((self.mem.nearest(p)[0], int(a)))
        self.rng = rng

    def cell(self, a, b):
        tr = self.s.bind(self.L, a) + self.s.bind(self.R, b)
        return self.mem.get(self.mem.intern(self.s.unitary(), tr, 10))

    def weights(self, k):
        w = self.rng.uniform(0.5, 1.5, k)
        return w / w.sum()


def score(found, truth):
    """(exact, TV) of a readout {key: w} against the true {key: w}."""
    keys = set(found) | set(truth)
    tv = 0.5 * sum(abs(found.get(q, 0.0) - truth.get(q, 0.0)) for q in keys)
    return set(found) == set(truth), tv


def readout(W, v):
    ws, _ = hdc.readout(W.C, v, -1024, 1024, 1.0, 256)
    return ws


def as_dist(ws, num=False):
    out = {}
    for label, i, w in ws:
        key = ("n", i) if label == hdc.Cleanup.NUM else ("m", i)
        out[key] = out.get(key, 0.0) + w
    return out


def trial_ptr(W, k):
    idx = W.rng.choice(len(W.lists), k, replace=False)
    w = W.weights(k)
    x = sum(wi * W.mem.get(W.lists[j][0]) for wi, j in zip(w, idx))
    truth_p, truth_f = {}, {}
    for wi, j in zip(w, idx):
        truth_p[("m", W.lists[j][0])] = truth_p.get(("m", W.lists[j][0]), 0) + wi
        truth_f[("m", W.lists[j][1])] = truth_f.get(("m", W.lists[j][1]), 0) + wi
    r1 = score(as_dist(readout(W, x)), truth_p)
    r2 = score(as_dist(readout(W, W.C.part(W.L, x))), truth_f)
    return r1, r2


def trial_int(W, k):
    ns = W.rng.choice(np.arange(-1000, 1001), k, replace=False)
    w = W.weights(k)
    x = sum(wi * W.nums.vec(int(n)) for wi, n in zip(w, ns))
    truth = {("n", int(n)): wi for n, wi in zip(ns, w)}
    r1 = score(as_dist(readout(W, x)), truth)
    env = W.cell(W.sym, x / np.linalg.norm(x))  # unit fields, as vsc.worlds stores them
    r2 = score(as_dist(readout(W, W.C.part(W.R, env))), truth)
    return r1, r2


def run(dims, ks_ptr, ks_int, trials):
    rows = []
    for d in dims:
        W = World(d, seed=d)
        t0 = time.time()
        for k in ks_ptr:
            for t in range(trials):
                (e1, tv1), (e2, tv2) = trial_ptr(W, k)
                rows.append(dict(dim=d, readout="pointer", k=k, trial=t, exact=int(e1), tv=tv1))
                rows.append(dict(dim=d, readout="first", k=k, trial=t, exact=int(e2), tv=tv2))
        for k in ks_int:
            for t in range(trials):
                (e1, tv1), (e2, tv2) = trial_int(W, k)
                rows.append(dict(dim=d, readout="int", k=k, trial=t, exact=int(e1), tv=tv1))
                rows.append(dict(dim=d, readout="int-env", k=k, trial=t, exact=int(e2), tv=tv2))
        print(f"D={d}: {time.time() - t0:.0f} s, M has {W.mem.n} rows", flush=True)
    return rows


def summarise(rows):
    agg = {}
    for r in rows:
        key = (r["readout"], r["dim"], r["k"])
        a = agg.setdefault(key, [0, 0, 0.0])
        a[0] += r["exact"]
        a[1] += 1
        a[2] += r["tv"]
    return {k: (e / n, tv / n) for k, (e, n, tv) in agg.items()}


def capacity_table(summ, dims):
    """Largest k with P(exact) ≥ 0.9 and with mean TV < 0.1, per readout, D."""
    out = {}
    for ro in ("pointer", "first", "int", "int-env"):
        for d in dims:
            ks = sorted(k for (r, dd, k) in summ if r == ro and dd == d)
            ex = [k for k in ks if summ[(ro, d, k)][0] >= 0.9]
            tv = [k for k in ks if summ[(ro, d, k)][1] < 0.1]
            # the largest k below which every point passes
            def lim(ok):
                best = 0
                for k in ks:
                    if k in ok:
                        best = k
                    else:
                        break
                return best
            out[(ro, d)] = (lim(ex), lim(tv))
    return out


def plot(summ, dims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blues = ["#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex="col")
    titles = {"pointer": "pointers, read directly", "first": "(first x): one cell read",
              "int": "integers, read directly", "int-env": "integers through a let cell"}
    for c, ro in enumerate(("pointer", "first", "int", "int-env")):
        for d, col in zip(dims, blues):
            ks = sorted(k for (r, dd, k) in summ if r == ro and dd == d)
            ex = [summ[(ro, d, k)][0] for k in ks]
            tv = [summ[(ro, d, k)][1] for k in ks]
            axes[0, c].plot(ks, ex, "-o", color=col, label=f"D={d}")
            axes[1, c].plot(ks, tv, "-o", color=col)
            cap = d // 100 if ro in ("pointer", "first") else d // 400
            axes[0, c].axvline(cap, color=col, lw=0.8, ls=":")
        axes[0, c].set_title(titles[ro], loc="left", fontsize=10)
        axes[1, c].set_xlabel("worlds k")
        axes[0, c].set_ylim(-0.03, 1.03)
        axes[1, c].set_ylim(0, 0.6)
        for ax in axes[:, c]:
            ax.set_xscale("log", base=2)
            ax.grid(True, color="#e1e0d9", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel("P(exact worlds)")
    axes[1, 0].set_ylabel("mean TV of the weights")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("W3  world readout vs number of worlds and D  (dotted: the capacity `worlds` "
                 "enforces, D/100 and D/400)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "w-readout.png"), dpi=140)


def main():
    quick = "--quick" in sys.argv
    dims = [1024, 2048] if quick else DIMS
    trials = 4 if quick else TRIALS
    rows = run(dims, KS_PTR, KS_INT, trials)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "w-readout.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summ = summarise(rows)
    for (ro, d), (e, t) in sorted(capacity_table(summ, dims).items()):
        print(f"{ro:8s} D={d:5d}  exact>=0.9 up to k={e:3d}   TV<0.1 up to k={t:3d}")
    plot(summ, dims)


if __name__ == "__main__":
    main()
