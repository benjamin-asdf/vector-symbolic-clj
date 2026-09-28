"""Q5: level-synchronous BFS with one vector per level.

Random directed graph, n nodes, out-degree 3. Node atoms X (unit rows).
Adjacency memory: key X[u] -> value A[u] = Σ_{v ∈ N(u)} X[v]  (unnormalised,
so every neighbour carries amplitude 1).

Level step (one heteroassociative read for the whole frontier):
    y      = mem(f)                         f = Σ_{u ∈ frontier} X[u]
    next   = {v : X[v]·y > ½·unit}           unit = amplitude a frontier row got
    new    = next minus visited
    f'     = Σ_{v ∈ new} X[v]               re-binarised ("snapped") frontier
visited is either a host bitset ("host visited") or itself one superposed
vector Vis = Σ X[v] tested by X[v]·Vis > ½ ("vector visited").
Variant "no snap": y is fed back raw (path counts, A^t x), detection only for scoring.

Correctness: every node's BFS distance (or unreachable) exactly right.
"""

import json
import sys
import time

import numpy as np

from common import (OUT, FIG, C, INK2, HeteroMemory, style, unit_rows)

DEG = 3


def graph(rng, n):
    nbrs = [rng.choice(np.delete(np.arange(n), u), DEG, replace=False) for u in range(n)]
    return nbrs


def bfs_true(nbrs, s):
    n = len(nbrs)
    dist = np.full(n, -1)
    dist[s] = 0
    fr = [s]
    t = 0
    peak = 1
    while fr:
        t += 1
        nx = []
        for u in fr:
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = t
                    nx.append(v)
        peak = max(peak, len(nx))
        fr = nx
    return dist, peak


def bfs_vec(X, mem, s, mode, beta=None, visited="host", snap=True, max_levels=64):
    n, D = X.shape
    dist = np.full(n, -1)
    dist[s] = 0
    f = X[s].astype(np.float64)
    vis = f.copy()
    idx = np.array([s])
    for t in range(1, max_levels):
        y, a = mem.read(f, mode, beta=beta)
        ny = np.linalg.norm(y)
        if ny < 1e-9:
            break
        sc = X @ y.astype(np.float32)
        # a neighbour of one frontier node carries the amplitude that node got
        # from the memory (1 for linear/thresh, |f|/|F| for softmax); detect
        # at half of it, which balances misses against false positives
        unit = float(np.mean(a[idx])) if snap else 1.0
        nxt = sc > 0.5 * unit
        if visited == "host":
            new = nxt & (dist < 0)
        else:
            new = nxt & ((X @ vis.astype(np.float32)) < 0.5)
            new &= dist < 0  # a node is only scored once; it cannot "unvisit"
        idx = np.nonzero(new)[0]
        if len(idx) == 0:
            break
        dist[idx] = t
        fnew = X[idx].astype(np.float64).sum(0)
        vis += fnew
        f = fnew if snap else y
    return dist


def main():
    rng = np.random.default_rng(9)
    configs = [("linear", None, "host", True), ("thresh", None, "host", True),
               ("softmax", 10.0, "host", True), ("softmax", 30.0, "host", True),
               ("softmax", 100.0, "host", True), ("thresh", None, "vector", True),
               ("thresh", None, "host", False)]
    res = []
    for D in [512, 1024, 2048, 4096, 8192]:
        for n in [30, 100, 300, 1000, 3000]:
            t0 = time.time()
            for trial in range(16 if n <= 1000 else 6):
                nbrs = graph(rng, n)
                X = unit_rows(rng, n, D)
                A = np.zeros((n, D), np.float32)
                for u in range(n):
                    A[u] = X[nbrs[u]].sum(0)
                mem = HeteroMemory(X, A)
                s = int(rng.integers(n))
                truth, peak = bfs_true(nbrs, s)
                for mode, beta, vis, snap in configs:
                    d = bfs_vec(X, mem, s, mode, beta, vis, snap)
                    name = mode if beta is None else f"softmax(β={beta:g})"
                    if vis == "vector":
                        name += ", vector visited"
                    if not snap:
                        name += ", no snap"
                    res.append(dict(D=D, n=n, mode=name, exact=bool(np.all(d == truth)),
                                    node_acc=float(np.mean(d == truth)), peak=peak,
                                    levels=int(truth.max())))
            print(f"D={D} n={n} {time.time() - t0:.1f}s", flush=True)
    json.dump(res, open(f"{OUT}/q5.json", "w"))
    return res


def summary(res):
    import collections
    g = collections.defaultdict(list)
    for r in res:
        g[(r["mode"], r["D"], r["n"])].append(r)
    return {k: dict(exact=np.mean([r["exact"] for r in v]),
                    node_acc=np.mean([r["node_acc"] for r in v]),
                    peak=np.mean([r["peak"] for r in v])) for k, v in g.items()}


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sm = summary(res)
    modes = ["thresh", "thresh, vector visited", "linear", "softmax(β=10)",
             "softmax(β=30)", "softmax(β=100)", "thresh, no snap"]
    ls = ["-", "--", "-", "-", "-", "-", ":"]
    cols = [C[2], C[6], C[1], C[0], C[3], C[4], C[7]]
    Ds = [512, 1024, 2048, 4096, 8192]
    ns = [30, 100, 300, 1000, 3000]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    for col, n in enumerate([30, 100, 300]):
        for i, m in enumerate(modes):
            axs[0, col].plot(Ds, [sm[(m, D, n)]["exact"] for D in Ds], ls[i], marker="o",
                             ms=3, lw=1.6, color=cols[i], label=m)
            axs[1, col].plot(Ds, [sm[(m, D, n)]["node_acc"] for D in Ds], ls[i], marker="o",
                             ms=3, lw=1.6, color=cols[i], label=m)
        peak = np.mean([sm[(modes[0], D, n)]["peak"] for D in Ds])
        dstar = 4 * DEG * peak * (np.sqrt(2 * np.log(n)) + 1) ** 2
        for ax in axs[:, col]:
            ax.axvline(dstar, color=INK2, lw=0.8, ls="-.")
            ax.set_xscale("log", base=2)
        axs[0, col].text(dstar * 1.05, 0.05, "predicted D*", fontsize=7, color=INK2)
        style(axs[0, col], f"n = {n} nodes (peak frontier ≈ {peak:.0f})", None,
              "P(all BFS distances exact)" if col == 0 else None)
        style(axs[1, col], None, "D", "fraction of nodes at the right distance" if col == 0 else None)
    axs[0, 0].set_ylim(-0.03, 1.03)
    axs[1, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Q5  BFS with one superposed frontier vector per level (random digraph, out-degree 3)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/w0-q5-bfs.png", dpi=140)

if __name__ == "__main__":
    res = json.load(open(f"{OUT}/q5.json")) if "--plot" in sys.argv else main()
    sm = summary(res)
    for (m, D, n), v in sorted(sm.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(f"{m:28s} D={D:5d} n={n:5d} exact={v['exact']:.2f} node_acc={v['node_acc']:.3f} peak={v['peak']:.0f}")
    plot(res)
