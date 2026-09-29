"""Figures for the W3 runner CSVs (specs/w-*.edn).

    .venv/bin/python scripts/plot_worlds.py [w-lift] [w-bfs] [w-demos]

Reads out/<exp>.csv, writes figures/<exp>.png and prints the table the
write-up quotes. Proportions carry Wilson 95% intervals over the seeds.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG = ROOT / "out", ROOT / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUES = ["#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 4.5,
})


def read(exp):
    with open(OUT / f"{exp}.csv") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fnum(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def group(rows, keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def ok(r):
    return r["correct"] == "true"


def plot_lift():
    rows = read("w-lift")
    lifts = ["dbl", "len", "nth3", "inc20", "let3"]
    titles = {"dbl": "(dbl n): split if, 5 levels", "len": "(len xs): split if + rest",
              "nth3": "first of rest³", "inc20": "20 incs (binding)", "let3": "3 nested let lookups"}
    dims = sorted({int(r["dim"]) for r in rows})
    fig, axes = plt.subplots(1, len(lifts), figsize=(14, 3.2), sharey=True)
    g = group(rows, ["lift", "dim", "op-noise"])
    print("w-lift: mean TV (P exact) by op noise")
    for ax, lf in zip(axes, lifts):
        for d, col in zip(dims, BLUES[1:] if len(dims) == 3 else BLUES):
            xs = sorted({float(r["op-noise"]) for r in rows})
            tv = []
            for x in xs:
                rs = [r for r in rows if r["lift"] == lf and int(r["dim"]) == d and float(r["op-noise"]) == x]
                tvs = [fnum(r.get("m.tv"), 1.0) if not r.get("error") else 1.0 for r in rs]
                tv.append(sum(tvs) / max(1, len(tvs)))
                if d == 2048:
                    k = sum(ok(r) for r in rs)
                    print(f"  {lf:6s} D={d} σ={x:.2f}: TV {tv[-1]:.3f}  exact {k}/{len(rs)}")
            ax.plot(xs, tv, "-o", color=col, label=f"D={d}")
        ax.set_title(titles[lf], loc="left")
        ax.set_xlabel("op noise σ")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("mean TV of the worlds (1 = failed)")
    axes[0].legend(fontsize=8)
    fig.suptitle("W3  linear lifting under op noise", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "w-lift.png", dpi=140)


def plot_bfs():
    rows = read("w-bfs")
    dims = sorted({int(r["dim"]) for r in rows})
    ns = sorted({int(r["n"]) for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    print("w-bfs: P(all levels exact), node accuracy, peak frontier")
    for d, col in zip(dims, BLUES):
        ps, lo, hi, acc = [], [], [], []
        for n in ns:
            rs = [r for r in rows if int(r["dim"]) == d and int(r["n"]) == n]
            k = sum(ok(r) for r in rs)
            p, l, h = wilson(k, len(rs))
            ps.append(p); lo.append(p - l); hi.append(h - p)
            a = [fnum(r.get("m.node-acc"), 0.0) if not r.get("error") else 0.0 for r in rs]
            acc.append(sum(a) / max(1, len(a)))
            pk = [fnum(r.get("m.peak-frontier")) for r in rs if r.get("m.peak-frontier")]
            errs = sorted({r["error"][:60] for r in rs if r.get("error")})
            print(f"  D={d:5d} n={n:4d}: exact {k}/{len(rs)}  node-acc {acc[-1]:.3f}"
                  f"  peak {max(pk) if pk else '-'}  {errs[:1]}")
        a1.errorbar(ns, ps, yerr=[lo, hi], fmt="-o", color=col, capsize=2, label=f"D={d}")
        a2.plot(ns, acc, "-o", color=col, label=f"D={d}")
    for ax, t, yl in ((a1, "P(every level exact)", "P(exact)"), (a2, "nodes at the right level", "fraction")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("graph size n (out-degree 2)")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left")
        ax.set_ylim(-0.03, 1.03)
    a1.legend(fontsize=8)
    fig.suptitle("W3  level-synchronous BFS, one vector per level", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "w-bfs.png", dpi=140)


def plot_demos():
    rows = read("w-demos")
    dims = sorted({int(r["dim"]) for r in rows})
    demos = ["dice4", "dice6", "sprinkler", "pmap", "bfs12"]
    labels = {"dice4": "2d4 sum (7 int worlds)", "dice6": "2d6 sum (11 int worlds)",
              "sprinkler": "P(rain | wet)", "pmap": "for-worlds map", "bfs12": "BFS, 12 nodes"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    print("w-demos: exact / mean TV by D")
    for dm, col in zip(demos, CAT):
        ps, tvs = [], []
        for d in dims:
            rs = [r for r in rows if r["demo"] == dm and int(r["dim"]) == d]
            k = sum(ok(r) for r in rs)
            ps.append(k / max(1, len(rs)))
            t = [fnum(r.get("m.tv"), 1.0) if not r.get("error") else 1.0 for r in rs]
            tvs.append(sum(t) / max(1, len(t)))
            errs = sorted({r["error"][:70] for r in rs if r.get("error")})
            print(f"  {dm:9s} D={d:5d}: exact {k}/{len(rs)}  TV {tvs[-1]:.3f}  {errs[:1]}")
        a1.plot(dims, ps, "-o", color=col, label=labels[dm])
        if dm != "bfs12":
            a2.plot(dims, tvs, "-o", color=col, label=labels[dm])
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel("D")
    a1.set_ylabel("P(correct)")
    a1.set_ylim(-0.03, 1.03)
    a2.set_ylabel("mean TV to exact enumeration (1 = threw)")
    a2.set_ylim(-0.02, 1.02)
    a1.legend(fontsize=8)
    fig.suptitle("W3  the W2 demos vs D", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "w-demos.png", dpi=140)


if __name__ == "__main__":
    which = sys.argv[1:] or ["w-lift", "w-bfs", "w-demos"]
    FIG.mkdir(exist_ok=True)
    for w in which:
        {"w-lift": plot_lift, "w-bfs": plot_bfs, "w-demos": plot_demos}[w]()
