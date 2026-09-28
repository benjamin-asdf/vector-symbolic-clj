"""Q4: the beta collapse curve.

N stored patterns X (unit rows). A probe with k worlds, p = Σ wᵢ xᵢ + noise,
is iterated through a separable Hopfield update p ← Xᵀ f(X p):

  softmax(β)   f(s) = softmax(β s/|p|)·|p|     modern Hopfield / attention
  thresh       f(s) = s·[|s|/|p| > z/√D]        threshold-linear
  linear       f(s) = s                          β -> 0 tangent (mean removed)

After 1, 3 and 10 steps: world coefficients c = least squares of p on the
k world patterns; effective #worlds = participation ratio of c; weight error
TV(c, w); noise fraction = 1 - |proj|²/|p|² (distractor + noise energy).

Theory (derived in the doc): the k-world symmetric state has gain β/√k under
iteration (unstable when β > √k), and one step suppresses N distractors only
when β ≳ √k·ln(N/k).
"""

import json
import sys

import numpy as np

from common import (OUT, FIG, C, HeteroMemory, participation, unit_rows,
                    style, tv)

D = 2048
BETAS = np.round(np.logspace(0, 3, 28), 3)
STEPS = [1, 3, 10]


def run_case(rng, X, k, weights, sigma, mode, beta=None):
    N = X.shape[0]
    mem = HeteroMemory(X, X)
    rows = rng.choice(N, k, replace=False)
    w = np.asarray(weights, float)
    w = w / w.sum()
    Xw = X[rows].astype(np.float64)
    p = w @ Xw
    p = p + sigma * np.linalg.norm(p) * rng.standard_normal(D) / np.sqrt(D)
    out = {}
    for t in range(1, max(STEPS) + 1):
        p, _ = mem.read(p, mode, beta=beta)
        n = np.linalg.norm(p)
        if n == 0:
            break
        p = p / n
        if t in STEPS:
            c, *_ = np.linalg.lstsq(Xw.T, p, rcond=None)
            proj = Xw.T @ c
            cp = np.clip(c, 0, None)
            dist = {i: x / cp.sum() for i, x in enumerate(cp)} if cp.sum() > 0 else {}
            out[t] = dict(pr=participation(cp),
                          tv=tv(dist, {i: x for i, x in enumerate(w)}),
                          noise=float(1 - (proj @ proj) / (p @ p)))
    return out


def main(only=None):
    rng = np.random.default_rng(4)
    res = [] if only is None else [r for r in json.load(open(f"{OUT}/q4.json")) if r["mode"] != only]
    for N in [100, 1000, 10000]:
        X = unit_rows(rng, N, D)
        for k in [2, 4, 8]:
            for wkind in ["equal", "unequal"]:
                weights = np.ones(k) if wkind == "equal" else np.linspace(1.0, 0.4, k)
                for sigma in [0.0, 1.0]:
                    cases = [("linear", None), ("thresh", None), ("proj", None)] + \
                            [("softmax", b) for b in BETAS]
                    if only is not None:
                        cases = [c for c in cases if c[0] == only]
                    for mode, beta in cases:
                        for s in range(8):
                            o = run_case(rng, X, k, weights, sigma, mode, beta)
                            for t, r in o.items():
                                res.append(dict(N=N, k=k, w=wkind, sigma=sigma,
                                                mode=mode, beta=beta, seed=s,
                                                step=t, **r))
            print("N", N, "k", k, flush=True)
    json.dump(res, open(f"{OUT}/q4.json", "w"))
    return res


def agg(res, **sel):
    rs = [r for r in res if all(r[a] == b for a, b in sel.items())]
    return {m: float(np.mean([r[m] for r in rs])) for m in ("pr", "tv", "noise")} if rs else None


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    for col, N in enumerate([100, 1000, 10000]):
        k = 4
        for j, t in enumerate(STEPS):
            pr = [agg(res, N=N, k=k, w="equal", sigma=1.0, mode="softmax",
                      beta=b, step=t)["pr"] for b in BETAS]
            nz = [agg(res, N=N, k=k, w="equal", sigma=1.0, mode="softmax",
                      beta=b, step=t)["noise"] for b in BETAS]
            axs[0, col].plot(BETAS, pr, "-", lw=1.8, color=C[j], label=f"softmax, {t} step{'s' if t > 1 else ''}")
            axs[1, col].plot(BETAS, nz, "-", lw=1.8, color=C[j], label=f"softmax, {t} step{'s' if t > 1 else ''}")
        th = agg(res, N=N, k=k, w="equal", sigma=1.0, mode="thresh", step=10)
        axs[0, col].axhline(th["pr"], color=C[2], lw=1.2, ls="--", label="threshold-linear, 10 steps")
        axs[1, col].axhline(th["noise"], color=C[2], lw=1.2, ls="--", label="threshold-linear, 10 steps")
        pj = agg(res, N=N, k=k, w="equal", sigma=1.0, mode="proj", step=10)
        if pj:
            axs[0, col].axhline(pj["pr"], color=C[6], lw=1.2, ls="-.", label="threshold + projection, 10 steps")
            axs[1, col].axhline(pj["noise"], color=C[6], lw=1.2, ls="-.", label="threshold + projection, 10 steps")
        b_stab = np.sqrt(k)
        b_sup = np.sqrt(k) * np.log(N / k)
        for ax in axs[:, col]:
            ax.axvline(b_stab, color=INK2_, lw=0.8, ls=":")
            ax.axvline(b_sup, color=INK2_, lw=0.8, ls="-.")
            ax.set_xscale("log")
        axs[0, col].text(b_stab * 1.05, 0.15, "β=√k", fontsize=7, color=INK2_)
        axs[0, col].text(b_sup * 1.05, 0.45, "√k·ln(N/k)", fontsize=7, color=INK2_)
        style(axs[0, col], f"N = {N} stored patterns",
              None, "effective #worlds (PR)" if col == 0 else None)
        style(axs[1, col], None, "β", "noise + distractor energy" if col == 0 else None)
        axs[0, col].set_ylim(0, 4.3)
        axs[1, col].set_ylim(0, 1.02)
    axs[1, 2].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Q4  β collapse curve: k = 4 equal worlds + probe noise (σ = 1) iterated through softmax(β Xp)ᵀX",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q4-beta-collapse.png", dpi=140)

    # weight distortion, unequal weights, one step
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for j, N in enumerate([100, 1000, 10000]):
        tvs = [agg(res, N=N, k=4, w="unequal", sigma=0.0, mode="softmax",
                   beta=b, step=1)["tv"] for b in BETAS]
        ax.plot(BETAS, tvs, "-", lw=1.8, color=C[j], label=f"softmax, N={N}")
    th = agg(res, N=1000, k=4, w="unequal", sigma=0.0, mode="thresh", step=1)["tv"]
    ax.axhline(th, color=C[2], ls="--", lw=1.2, label="threshold-linear")
    ax.set_xscale("log")
    style(ax, "Q4  weight error after ONE step, weights ∝ (1, .8, .6, .4)", "β",
          "TV(recovered, true weights)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q4-weight-distortion.png", dpi=140)


INK2_ = "#52514e"


def table(res):
    lines = []
    for N in [100, 1000, 10000]:
        for k in [2, 4, 8]:
            for w in ["equal", "unequal"]:
                win = []
                for b in BETAS:
                    a = agg(res, N=N, k=k, w=w, sigma=1.0, mode="softmax", beta=b, step=10)
                    if a["pr"] >= 0.8 * k and a["noise"] < 0.1:
                        win.append(float(b))
                a1 = [agg(res, N=N, k=k, w=w, sigma=1.0, mode="softmax", beta=b, step=1)
                      for b in BETAS]
                win1 = [float(b) for b, a in zip(BETAS, a1) if a["pr"] >= 0.8 * k and a["noise"] < 0.1]
                th = agg(res, N=N, k=k, w=w, sigma=1.0, mode="thresh", step=10)
                pj = agg(res, N=N, k=k, w=w, sigma=1.0, mode="proj", step=10)
                li = agg(res, N=N, k=k, w=w, sigma=1.0, mode="linear", step=10)
                lines.append(dict(N=N, k=k, w=w, window_step1=[min(win1), max(win1)] if win1 else None,
                                  window_step10=[min(win), max(win)] if win else None,
                                  thresh10=th, proj10=pj, linear10=li))
    return lines


if __name__ == "__main__":
    if "--plot" in sys.argv:
        res = json.load(open(f"{OUT}/q4.json"))
    elif "--only" in sys.argv:
        res = main(sys.argv[sys.argv.index("--only") + 1])
    else:
        res = main()
    for line in table(res):
        print(line)
    json.dump(table(res), open(f"{OUT}/q4_summary.json", "w"), indent=1)
    plot(res)
