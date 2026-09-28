"""Q3: both-branch `if` on a superposed test, and the paper's ⊕.

Paper (Tomkins-Flanagan & Kelly 2024, eq. 1, checked against arXiv 2510.17889):
    a ⊕ b = [E‖a‖ > θ↑] a + [E‖a‖ < θ↓] b + [θ↓ ≤ E‖a‖ ≤ θ↑] ν(a + b)
used as  cond(r) = sim(car(car r), T)·cdr(car r) ⊕ cond(cdr r),
so a = α·then with α = sim(test, T), and b = the (lazily evaluated) else.

(a) then-branch share vs the true probability π that the test holds, for
    soft-if (weights T·t, F·t), ⊕, and softmax(β·[α, β']) selection.
(b) recursion with worlds that stop at different depths:
    (defn len [xs] (if (empty? xs) 0 (inc (len (rest xs)))))
    (defn dbl [n] (if (zero? n) 0 (inc (inc (dbl (dec n))))))
    on superposed arguments, with and without splitting the argument between
    the branches.
"""

import json

import numpy as np

import common  # noqa: F401
from common import (OUT, FIG, C, INK2, HeteroMemory, as_dist, bind, nu, style,
                    tv, unbind, unit_rows, unitary, worlds)
import hdc

D = 2048


# -- (a) branch weights -----------------------------------------------------

def branch_share(pi, rule, theta=(0.1, 0.9), beta=None, D=D, rng=None, noise=0.0):
    """Share of the then-branch in the returned superposition, given a test
    vector t = π T + (1-π) F (what a lifted predicate returns)."""
    rng = rng or np.random.default_rng(0)
    T, F = unit_rows(rng, 2, D).astype(np.float64)
    t = pi * T + (1 - pi) * F
    if noise:
        t = t + noise * rng.standard_normal(D) / np.sqrt(D)
    tn = nu(t)
    alpha, alphaF = float(tn @ T), float(tn @ F)
    if rule == "soft":
        # coefficients of t on {T, F} (least squares, so a small T·F
        # correlation does not bias them); this is what `worlds` reads
        c, *_ = np.linalg.lstsq(np.stack([T, F]).T, tn, rcond=None)
        a, b = max(c[0], 0), max(c[1], 0)
    elif rule == "oplus":
        lo, hi = theta
        if alpha > hi:
            a, b = alpha, 0.0
        elif alpha < lo:
            a, b = 0.0, 1.0
        else:
            a, b = alpha, 1.0   # ν(α·then + else), then ⊥ else
    elif rule == "softmax":
        z = beta * np.array([alpha, alphaF])
        e = np.exp(z - z.max())
        a, b = e / e.sum()
    return a / (a + b) if a + b > 0 else 0.0


def part_a():
    pis = np.linspace(0, 1, 201)
    out = {}
    rules = [("soft-if (coefficients of t on T, F)", "soft", {}),
             ("paper ⊕, θ = (0.1, 0.9)", "oplus", {"theta": (0.1, 0.9)}),
             ("paper ⊕, θ = (0.3, 0.7)", "oplus", {"theta": (0.3, 0.7)}),
             ("softmax β = 4", "softmax", {"beta": 4}),
             ("softmax β = 16", "softmax", {"beta": 16}),
             ("softmax β = 64", "softmax", {"beta": 64})]
    for name, r, kw in rules:
        out[name] = [branch_share(p, r, **kw) for p in pis]
    # mean |share - π| over π
    err = {name: float(np.mean(np.abs(np.array(v) - pis))) for name, v in out.items()}
    return pis, out, err


# -- (b) recursion with diverging depths -----------------------------------

class ListWorld:
    """A small pointer memory with lists, the threshold-linear backend, and
    a cleanup back onto the pointer codebook between steps."""

    def __init__(self, rng, n_cells=2000):
        self.rng = rng
        self.sp = hdc.Space(D, 7)
        self.nums = hdc.Numbers(self.sp)
        self.L, self.R = unitary(rng, D), unitary(rng, D)
        self.atoms = unit_rows(rng, 200, D).astype(np.float64)
        self.E = unit_rows(rng, 1, D)[0].astype(np.float64)   # the () atom
        self.K, self.V = [], []
        for _ in range(n_cells // 6):   # distractor lists load the memory
            self.mk_list(int(rng.integers(1, 8)))
        self.freeze()
        self.gain = np.sqrt(2)

    def cell(self, a, rest):
        p = unit_rows(self.rng, 1, D)[0].astype(np.float64)
        self.K.append(p)
        self.V.append(nu(bind(self.L, a) + bind(self.R, rest)))
        return p

    def mk_list(self, n):
        v = self.E
        for _ in range(n):
            v = self.cell(self.atoms[self.rng.integers(len(self.atoms))], v)
        return v

    def freeze(self):
        self.mem = HeteroMemory(np.array(self.K, np.float32), np.array(self.V, np.float32))
        self.CB = np.vstack([self.mem.K.astype(np.float64), self.E[None]])

    def rest(self, x):
        t, _ = self.mem.read(x, "thresh")
        # a cell trace is ν(L⊗a + R⊗r): each field carries 1/√2 of it. Without
        # the gain every step shrinks every world by √2, and worlds that exit
        # deeper come back under-weighted by 2^(-depth/2).
        q = self.gain * unbind(self.R, t)
        S, c = worlds(q, self.CB, kmax=32)
        return c @ self.CB[S] if len(S) else np.zeros(D)


def len_super(W, x, split, depth=0, cap=12, log=None):
    """(len x) with a both-branch if. `split`: the else-branch gets only the
    worlds where (empty? x) is false; otherwise the whole x."""
    if np.linalg.norm(x) < 1e-6 or depth > cap:
        if depth > cap and log is not None:
            log.append("cap")
        return np.zeros(D)
    pe = float(W.E @ x)                         # (empty? x), linear in x
    then_ = pe * W.nums.vec(0)
    arg = x - pe * W.E if split else x
    if log is not None:
        log.append((depth, round(pe, 3), round(float(np.linalg.norm(arg)), 3)))
    if np.linalg.norm(arg) < 1e-3:              # prune: no world left in else
        return then_
    return then_ + bind(W.nums.vec(1), len_super(W, W.rest(arg), split, depth + 1, cap, log))


def dbl_super(nums, x, split, ns, CB, depth=0, cap=12, log=None):
    """(dbl n) with a both-branch if on (zero? n). The predicate is read off
    the world readout (the residue encoding makes the plain dot B^0·x leak:
    n shares p's band with 0 whenever p | n)."""
    if np.linalg.norm(x) < 1e-6 or depth > cap:
        if depth > cap and log is not None:
            log.append("cap")
        return np.zeros(D)
    S, c = worlds(x, CB, kmax=16)
    wd = dict(zip(ns[S].tolist(), c.tolist()))
    p0 = wd.get(0, 0.0)
    then_ = p0 * nums.vec(0)
    arg = x - p0 * nums.vec(0) if split else x
    if log is not None:
        log.append((depth, {k: round(v, 3) for k, v in wd.items()}))
    if np.linalg.norm(arg) < 1e-3:
        return then_
    inner = dbl_super(nums, bind(nums.vec(-1), arg), split, ns, CB, depth + 1, cap, log)
    return then_ + bind(nums.vec(2), inner)


def readout_ints(v, nums, R=(-30, 30)):
    ns = np.arange(R[0], R[1] + 1)
    CB = np.stack([nums.vec(int(n)) for n in ns])
    S, c = worlds(v, CB, kmax=16)
    return {int(ns[i]): round(x, 3) for i, x in as_dist(S, c).items()}


def part_b(rng):
    W = ListWorld(rng)
    lists = [W.mk_list(n) for n in (2, 3, 5)]
    W.freeze()
    w = np.array([0.5, 0.3, 0.2])
    x = sum(wi * li for wi, li in zip(w, lists))
    out = {}
    for gain in (1.0, np.sqrt(2)):
        W.gain = gain
        for split in (True, False):
            log = []
            r = len_super(W, x, split, log=log)
            out[f"len gain={gain:.3f} split={split}"] = dict(worlds=readout_ints(r, W.nums), trace=log)
    ns = np.arange(-30, 31)
    CB = np.stack([W.nums.vec(int(n)) for n in ns])
    xn = 0.5 * W.nums.vec(2) + 0.3 * W.nums.vec(3) + 0.2 * W.nums.vec(5)
    for split in (True, False):
        log = []
        r = dbl_super(W.nums, xn, split, ns, CB, log=log)
        out[f"dbl split={split}"] = dict(worlds=readout_ints(r, W.nums), trace=[str(t) for t in log])
    return out


def plot_a(pis, shares):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    styles = ["-", "-", "--", ":", ":", ":"]
    cols = [C[0], C[1], C[1], C[2], C[3], C[6]]
    for (name, v), ls, c in zip(shares.items(), styles, cols):
        ax.plot(pis, v, ls, lw=1.8, color=c, label=name)
    style(ax, "Q3  then-branch share of the result vs P(test)", "π = weight of worlds where the test holds",
          "then-branch share")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q3-if-oplus.png", dpi=140)


if __name__ == "__main__":
    pis, shares, err = part_a()
    print("mean |share - π|:", {k: round(v, 3) for k, v in err.items()})
    plot_a(pis, shares)
    b = part_b(np.random.default_rng(5))
    for k, v in b.items():
        print(k, v["worlds"])
        for t in v["trace"][:14]:
            print("   ", t)
    json.dump(dict(err=err, recursion=b), open(f"{OUT}/q3.json", "w"), indent=1, default=str)
