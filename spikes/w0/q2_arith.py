"""Q2: arithmetic lifting under residue integers (hdc.Numbers).

(a) exactness: (+ (amb 1 2) 10) = B^10 ⊗ ν(B^1 + B^2) vs ν(B^11 + B^12)
(b) world readout from a superposed integer: OMP over the integer codebook
    [-R, R]; residue chimeras appear as false worlds
(c) chains of T random ±c ops with per-op noise σ: weight survival
(d) sharing: (let [x (amb 1 2)] (+ x x)). Superposition gives run-time choice
    {2:¼, 3:½, 4:¼}; call-time choice is {2:½, 4:½}. A direct-sum ("spectral
    worlds") encoding, one disjoint frequency band per world, gives the latter.
"""

import json

import numpy as np

import common  # noqa: F401  (sets thread env, sys.path)
from common import OUT, FIG, C, INK2, as_dist, cos, nu, style, tv, worlds
import hdc

D = 2048


def setup(D=D, seed=0):
    sp = hdc.Space(D, seed)
    nums = hdc.Numbers(sp)
    return sp, nums


def int_codebook(nums, R):
    ns = np.arange(-R, R + 1)
    return ns, np.stack([nums.vec(int(n)) for n in ns]).astype(np.float64)


def superpose_ints(nums, vals, w):
    return sum(x * nums.vec(int(v)) for v, x in zip(vals, w))


def readout(v, ns, CB):
    S, c = worlds(v, CB, kmax=64)
    return {int(ns[i]): x for i, x in as_dist(S, c).items()}


def part_a(sp, nums):
    v = nu(nums.vec(1) + nums.vec(2))
    u = sp.bind(nums.vec(10), v)
    target = nu(nums.vec(11) + nums.vec(12))
    w = 0.3 * nums.vec(1) + 0.7 * nums.vec(2)
    uw = sp.bind(nums.vec(10), w)
    A = np.stack([nums.vec(11), nums.vec(12)])
    c, *_ = np.linalg.lstsq(A.T, uw, rcond=None)
    # what the interpreter's single-number readout makes of it
    single = nums.read(u)
    # per-band residues seen in the superposition: two per band, so chimeras
    return dict(cos_equal=1 - cos(u, target), max_abs_diff=float(np.max(np.abs(u - target))),
                weights_after=c.tolist(), single_read=single)


def chimeras(nums, a, b, R):
    """Integers in [-R, R] whose residue on every modulus matches a or b."""
    out = []
    for n in range(-R, R + 1):
        if n in (a, b):
            continue
        if all(n % p in (a % p, b % p) for p in nums.moduli):
            out.append(n)
    return out


def part_b(nums, rng, R=1000, trials=30):
    ns, CB = int_codebook(nums, R)
    res = {}
    for k in [1, 2, 3, 4, 6, 8, 12, 16]:
        ok = 0
        tvs = []
        for _ in range(trials):
            vals = rng.choice(ns, k, replace=False)
            w = rng.uniform(0.5, 1.5, k)
            w /= w.sum()
            got = readout(superpose_ints(nums, vals, w), ns, CB)
            want = {int(v): float(x) for v, x in zip(vals, w)}
            t = tv(got, want)
            tvs.append(t)
            ok += set(got) == set(want) and t < 0.1
        res[k] = dict(p_exact=ok / trials, tv=float(np.mean(tvs)))
    return res


def part_c(sp, nums, rng, trials=20):
    ns, CB = int_codebook(nums, 400)
    res = {}
    for sigma in [0.0, 0.05, 0.1, 0.2, 0.4]:
        for T in [1, 10, 30, 100, 300]:
            oks, tvs = 0, []
            for _ in range(trials):
                vals = np.array([3, 17, -40])
                w = np.array([0.5, 0.3, 0.2])
                v = superpose_ints(nums, vals, w)
                shift = 0
                for _ in range(T):
                    c = int(rng.integers(-5, 6))
                    v = sp.bind(nums.vec(c), v)
                    shift += c
                    if sigma > 0:
                        v = v + sigma * np.linalg.norm(v) * rng.standard_normal(D) / np.sqrt(D)
                if np.any(np.abs(vals + shift) > 400):
                    v = sp.bind(nums.vec(-shift), v)   # keep in codebook range
                    s2 = 0
                else:
                    s2 = shift
                got = readout(v, ns, CB)
                want = {int(x + s2): float(y) for x, y in zip(vals, w)}
                t = tv(got, want)
                tvs.append(t)
                oks += set(got) == set(want) and t < 0.1
            res[f"{sigma}/{T}"] = dict(sigma=sigma, T=T, p_exact=oks / trials, tv=float(np.mean(tvs)))
    return res


# -- (d) sharing: run-time vs call-time choice ------------------------------

def part_d(sp, nums, rng):
    ns, CB = int_codebook(nums, 50)
    x = 0.5 * nums.vec(1) + 0.5 * nums.vec(2)
    xx = sp.bind(x, x)                       # (+ x x) on a plain superposition
    runtime = readout(xx, ns, CB)
    # correlated pair: y = x + 10 computed from x, then (+ x y) (= 2x+10)
    y = sp.bind(nums.vec(10), x)
    xy = readout(sp.bind(x, y), ns, CB)

    # direct-sum worlds: world i owns a disjoint random set of Fourier bins
    k = 2
    bins = np.arange(D // 2 + 1)
    perm = rng.permutation(bins)
    masks = [np.zeros(D // 2 + 1, bool) for _ in range(k)]
    for i, part in enumerate(np.array_split(perm, k)):
        masks[i][part] = True

    def band(v, i):
        X = np.fft.rfft(v)
        return np.fft.irfft(np.where(masks[i], X, 0), n=D)

    def ds(vals, w):
        return sum(np.sqrt(k) * x * band(nums.vec(int(v)), i)
                   for i, (v, x) in enumerate(zip(vals, w)))

    def ds_worlds(v):
        out = {}
        for i in range(k):
            vi = band(v, i)
            Bi = np.stack([band(c, i) for c in CB])
            S, c = worlds(vi, Bi, kmax=4)
            for j, cc in zip(S, c):
                out[(i, int(ns[j]))] = float(cc)
        tot = sum(max(c, 0) for c in out.values())
        return {key: c / tot for key, c in out.items() if c > 0}

    X = ds([1, 2], [0.5, 0.5])
    # binding is per-bin multiplication, so worlds never mix; the √k scale
    # is applied again after a product to keep band amplitudes comparable
    XX = sp.bind(X, X) / np.sqrt(k)
    Y = sp.bind(nums.vec(10), X)
    XY = sp.bind(X, Y) / np.sqrt(k)
    return dict(plain_x_plus_x=runtime, plain_x_plus_y=xy,
                direct_sum_x_plus_x=ds_worlds(XX), direct_sum_x_plus_y=ds_worlds(XY))


def plot_c(res_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, s in enumerate([0.0, 0.05, 0.1, 0.2, 0.4]):
        rs = sorted([r for r in res_c.values() if r["sigma"] == s], key=lambda r: r["T"])
        ax.plot([r["T"] for r in rs], [r["tv"] for r in rs], "-o", ms=3, lw=1.6,
                color=C[i], label=f"op noise σ = {s}")
    ax.set_xscale("log")
    style(ax, "Q2  weights (.5,.3,.2) of 3 integer worlds after T ops, D = 2048",
          "T (chained ±c ops)", "TV(recovered, true)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q2-arith-chain.png", dpi=140)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    sp, nums = setup()
    out = {}
    out["a"] = part_a(sp, nums)
    print("a", out["a"])
    out["a_chimeras_11_12_in_1000"] = chimeras(nums, 11, 12, 100000)[:20]
    print("chimeras of {11,12} within ±100000:", out["a_chimeras_11_12_in_1000"])
    out["b"] = part_b(nums, rng)
    print("b", out["b"])
    out["c"] = part_c(sp, nums, rng)
    for r in out["c"].values():
        print("c", r)
    out["d"] = {k: {str(a): round(b, 3) for a, b in v.items()} for k, v in part_d(sp, nums, rng).items()}
    print("d", out["d"])
    json.dump(out, open(f"{OUT}/q2.json", "w"), indent=1, default=str)
    plot_c(out["c"])
