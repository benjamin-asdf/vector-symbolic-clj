"""Q1: linearity through the pointer layer.

A memory of N list cells: pointer K[j] -> trace V[j] = ν(L⊗A[first_j] + R⊗K[next_j]).
Probe p = Σ wᵢ K[jᵢ] (k worlds). first(p) = worlds(L ⊘ deref(p)) over atoms A,
rest(p) = worlds(R ⊘ deref(p)) over pointers K. Compare memory backends.

Metrics per trial
  deref_cos  cos(deref(p), Σ wᵢ V[jᵢ])        raw fidelity, before any cleanup
  first_tv   TV(recovered first-worlds, ideal)  weight recovery error
  first_ok   support exactly right and TV < 0.1
  chain_ok   first(restᵈ(p)) right for d = 1..4, raw vs cleaned between steps
"""

import json
import sys
import time

import numpy as np

from common import (OUT, FIG, C, INK2, HeteroMemory, as_dist, bind, cos, tv,
                    unbind, unit_rows, unitary, worlds, style)

MODES = [("codebook", {}), ("linear", {}), ("thresh", {}),
         ("softmax", {"beta": 10.0}), ("softmax", {"beta": 30.0}),
         ("softmax", {"beta": 100.0})]


def mode_name(m, kw):
    return m if not kw else f"{m}(β={kw['beta']:g})"


def build(rng, D, N, n_atoms):
    A = unit_rows(rng, n_atoms, D)
    K = unit_rows(rng, N, D)
    L, R = unitary(rng, D), unitary(rng, D)
    first = rng.integers(0, n_atoms, N)
    nxt = rng.integers(0, N, N)
    FL, FR = np.fft.rfft(L), np.fft.rfft(R)
    V = np.empty((N, D), dtype=np.float32)
    for s in range(0, N, 2048):
        e = slice(s, min(N, s + 2048))
        t = (np.fft.irfft(FL * np.fft.rfft(A[first[e]], axis=1), n=D, axis=1)
             + np.fft.irfft(FR * np.fft.rfft(K[nxt[e]], axis=1), n=D, axis=1))
        V[e] = t / np.linalg.norm(t, axis=1, keepdims=True)
    return dict(A=A, K=K, V=V, L=L, R=R, first=first, nxt=nxt,
                mem=HeteroMemory(K, V))


def ideal(rows, w, table):
    d = {}
    for j, x in zip(rows, w):
        d[int(table[j])] = d.get(int(table[j]), 0.0) + float(x)
    return d


def trial(rng, m, k, modes, depth=4):
    N = m["K"].shape[0]
    rows = rng.choice(N, k, replace=False)
    w = rng.uniform(0.5, 1.5, k)
    w /= w.sum()
    p = w @ m["K"][rows].astype(np.float64)
    ideal_trace = w @ m["V"][rows].astype(np.float64)
    want_first = ideal(rows, w, m["first"])
    out = {}
    for mode, kw in modes:
        name = mode_name(mode, kw)
        t, _ = m["mem"].read(p, mode, **kw)
        r = {"deref_cos": cos(t, ideal_trace)}
        S, c = worlds(unbind(m["L"], t), m["A"], kmax=2 * k + 8)
        got = as_dist(S, c)
        r["first_tv"] = tv(got, want_first)
        r["first_ok"] = bool(set(got) == set(want_first) and r["first_tv"] < 0.1)
        # chains: first(rest^d(p)), raw (no cleanup between steps) vs cleaned
        for clean in (False, True):
            cur_rows, cur_w, q = rows, w, p
            oks = []
            for dd in range(1, depth + 1):
                # ground truth of the next level
                nd = ideal(cur_rows, cur_w, m["nxt"])
                cur_rows = np.array(list(nd.keys()))
                cur_w = np.array(list(nd.values()))
                t, _ = m["mem"].read(q, mode, **kw)
                q = unbind(m["R"], t)
                if clean:
                    S, c = worlds(q, m["K"], kmax=2 * k + 8)
                    q = (c @ m["K"][S].astype(np.float64)) if len(S) else q
                t2, _ = m["mem"].read(q, mode, **kw)
                S, c = worlds(unbind(m["L"], t2), m["A"], kmax=2 * k + 8)
                got = as_dist(S, c)
                wf = ideal(cur_rows, cur_w, m["first"])
                oks.append(bool(set(got) == set(wf) and tv(got, wf) < 0.1))
            r["chain_clean" if clean else "chain_raw"] = oks
        out[name] = r
    return out


def main(quick=False):
    rng = np.random.default_rng(1)
    Ds = [1024, 2048, 4096]
    Ns = [1000, 4000, 16000]
    ks = [1, 2, 4, 8, 16, 32, 64]
    trials = 4 if quick else 10
    n_atoms = 2000
    res = []
    for D in Ds:
        for N in Ns:
            t0 = time.time()
            m = build(rng, D, N, n_atoms)
            for k in ks:
                modes = MODES if k <= 32 else [x for x in MODES if x[0] != "codebook"]
                for _ in range(trials):
                    for name, r in trial(rng, m, k, modes).items():
                        res.append(dict(D=D, N=N, k=k, mode=name, **r))
            print(f"D={D} N={N} {time.time() - t0:.0f}s", flush=True)
            del m
    with open(f"{OUT}/q1.json", "w") as f:
        json.dump(res, f)
    return res


def summarize(res):
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["D"], r["N"], r["mode"], r["k"])].append(r)
    rows = []
    for (D, N, mode, k), rs in sorted(agg.items()):
        rows.append(dict(
            D=D, N=N, mode=mode, k=k,
            deref_cos=np.mean([r["deref_cos"] for r in rs]),
            first_tv=np.mean([r["first_tv"] for r in rs]),
            first_ok=np.mean([r["first_ok"] for r in rs]),
            chain_raw=np.mean([r["chain_raw"] for r in rs], axis=0).tolist(),
            chain_clean=np.mean([r["chain_clean"] for r in rs], axis=0).tolist()))
    return rows


def capacity(rows, D, N, mode, key="first_ok", level=0.9):
    """Largest k with success ≥ level (all smaller k also passing)."""
    best = 0
    for r in sorted([r for r in rows if r["D"] == D and r["N"] == N
                     and r["mode"] == mode], key=lambda r: r["k"]):
        v = r[key] if not isinstance(r[key], list) else r[key][-1]
        if v >= level:
            best = r["k"]
        else:
            break
    return best


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = ["codebook", "linear", "thresh", "softmax(β=10)",
             "softmax(β=30)", "softmax(β=100)"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
    for col, N in enumerate([1000, 4000, 16000]):
        for i, mode in enumerate(modes):
            rs = sorted([r for r in rows if r["D"] == 2048 and r["N"] == N
                         and r["mode"] == mode], key=lambda r: r["k"])
            ks = [r["k"] for r in rs]
            axs[0, col].plot(ks, [r["deref_cos"] for r in rs], "-o", ms=3, lw=1.6,
                             color=C[i], label=mode)
            axs[1, col].plot(ks, [r["first_tv"] for r in rs], "-o", ms=3, lw=1.6,
                             color=C[i], label=mode)
        style(axs[0, col], f"D = 2048, N = {N} cells", None,
              "cos(deref p, Σ wᵢ traceᵢ)" if col == 0 else None)
        style(axs[1, col], None, "k worlds",
              "TV(first-worlds, ideal)" if col == 0 else None)
        axs[0, col].set_xscale("log", base=2)
        axs[0, col].set_ylim(0, 1.02)
        axs[1, col].set_ylim(0, 1.0)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q1  first/rest of a superposition of k lists through the pointer memory",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q1-linearity.png", dpi=140)

    # capacity vs D: largest k whose mean weight error stays under 0.1
    fig, ax = plt.subplots(figsize=(6, 3.8))
    Ds = [1024, 2048, 4096]
    for i, mode in enumerate(["linear", "thresh", "softmax(β=30)", "codebook"]):
        for j, N in enumerate([1000, 16000]):
            ys = [k_at_tv(rows, D, N, mode) for D in Ds]
            ax.plot(Ds, ys, ["-o", "--s"][j], ms=4, lw=1.6,
                    color=C[[1, 2, 4, 0][i]], label=f"{mode}, N={N}")
    ax.plot(Ds, [D / 100 for D in Ds], ":", color=INK2, lw=1, label="D/100")
    style(ax, "Q1  world capacity: largest k with TV(first-worlds) < 0.1", "D",
          "k (interpolated)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q1-capacity.png", dpi=140)


def k_at_tv(rows, D, N, mode, level=0.1):
    rs = sorted([r for r in rows if r["D"] == D and r["N"] == N and r["mode"] == mode],
                key=lambda r: r["k"])
    prev = None
    for r in rs:
        if r["first_tv"] >= level:
            if prev is None:
                return 1.0
            # log-linear interpolation between prev and r
            a, b = prev["first_tv"], r["first_tv"]
            f = (level - a) / (b - a)
            return float(2 ** (np.log2(prev["k"]) + f * (np.log2(r["k"]) - np.log2(prev["k"]))))
        prev = r
    return float(rs[-1]["k"])

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if "--plot" in sys.argv:
        res = json.load(open(f"{OUT}/q1.json"))
    else:
        res = main(quick)
    rows = summarize(res)
    for D in [1024, 2048, 4096]:
        for N in [1000, 4000, 16000]:
            caps = {m: capacity(rows, D, N, m) for m in
                    ["codebook", "linear", "thresh", "softmax(β=10)",
                     "softmax(β=30)", "softmax(β=100)"]}
            ccap = {m: capacity(rows, D, N, m, "chain_clean") for m in ["linear", "thresh"]}
            rcap = {m: capacity(rows, D, N, m, "chain_raw") for m in ["linear", "thresh"]}
            ktv = {m: round(k_at_tv(rows, D, N, m), 1) for m in ["codebook", "linear", "thresh", "softmax(β=30)"]}
            print(f"D={D:5d} N={N:6d} cap(first)={caps} k@TV<.1={ktv} chain4 clean={ccap} raw={rcap}")
    with open(f"{OUT}/q1_summary.json", "w") as f:
        json.dump(rows, f, indent=1)
    plot(rows)
