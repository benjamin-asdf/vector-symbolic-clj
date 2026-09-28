"""Figures for the experiment CSVs written by vsc.experiments.

    .venv/bin/python scripts/plot.py e1 [e2 ...] [--figures]

Reads out/<exp>.csv, writes out/<exp>*.png (with --figures, also copies
them to figures/). Proportions carry Wilson 95% intervals over all trials
pooled across seeds. Theory curves are drawn dashed where there is a closed
form (E1, E3).
"""

import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = ROOT / "figures"

# reference palette (dataviz skill): surface, ink, hairline grid
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# ordinal blue ramp, light to dark (steps 250 … 700)
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 4.5,
})


def ramp(n):
    """n ordinal steps of the blue ramp, evenly spread, light to dark."""
    picks = {1: [2], 2: [1, 4], 3: [0, 2, 5], 4: [0, 2, 3, 5], 5: [0, 1, 2, 3, 5]}
    idx = picks.get(n) or np.linspace(0, len(BLUES) - 1, n).round().astype(int)
    return [BLUES[i] for i in idx]


# ---------------------------------------------------------------------------
# data

def read(exp):
    path = OUT / f"{exp}.csv"
    with open(path) as f:
        header = [l[2:].strip() for l in f if l.startswith("#")]
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return rows, header


def num(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def wilson(k, n, z=1.96):
    """(p, lo, hi): Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def pooled(rows, key, x, flip=False):
    """{series: [(x, p, lo, hi, n_runs)]}: successes/trials pooled over seeds.
    flip: report the failure proportion instead."""
    acc = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k, n = num(r["successes"], 0), num(r["trials"], 0)
        a = acc[(key(r), x(r))]
        a[0] += k
        a[1] += n
        a[2] += 1
    out = defaultdict(list)
    for (s, xv), (k, n, runs) in sorted(acc.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if flip:
            k = n - k
        p, lo, hi = wilson(k, n)
        out[s].append((xv, p, lo, hi, runs))
    return out


def draw(ax, pts, color, label, marker="o", ls="-"):
    xs = np.array([p[0] for p in pts])
    p = np.array([p[1] for p in pts])
    lo = np.array([p[2] for p in pts])
    hi = np.array([p[3] for p in pts])
    ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(xs, p, ls, color=color, marker=marker, label=label, markeredgecolor=SURFACE,
            markeredgewidth=0.8)


def finish(fig, name, note=None):
    if note:
        fig.text(0.01, -0.02, note, fontsize=7, color=INK2, ha="left", va="top")
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def blas_note(header):
    return "  ·  ".join(h for h in header if h.startswith(("numpy", "blas")))[:160]


# ---------------------------------------------------------------------------
# theory

def _phi_cdf(x):
    """Standard normal CDF, vectorised (Abramowitz–Stegun 7.1.26 erf)."""
    x = np.asarray(x, dtype=float)
    t = np.abs(x) / np.sqrt(2)
    k = 1.0 / (1.0 + 0.3275911 * t)
    y = 1.0 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k
               + 0.254829592) * k * np.exp(-t * t)
    return 0.5 * (1.0 + np.sign(x) * y)


def p_cleanup(signal, noise_sd, distractor_sd, n_distractors):
    """P(signal + ε₀ > max of n εᵢ), ε₀ ~ N(0, noise_sd²), εᵢ ~ N(0, distractor_sd²),
    by quadrature over ε₀."""
    z = np.linspace(-8, 8, 4001)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    c = np.clip(_phi_cdf((signal + noise_sd * z) / distractor_sd), 1e-300, 1.0)
    return float(np.sum(w * np.exp(n_distractors * np.log(c))))


def e1_theory(sigma, dim, load):
    """Probe = unit atom + noise of norm σ: the atom scores 1 + ε₀, ε₀ ~ N(0, σ²/D);
    each other atom scores εᵢ ~ N(0, (1 + σ²)/D)."""
    return p_cleanup(1.0, sigma / math.sqrt(dim), math.sqrt((1 + sigma ** 2) / dim), load - 1)


# ---------------------------------------------------------------------------
# E1

def plot_e1():
    rows, header = read("e1")
    loads = sorted({int(num(r["load"])) for r in rows})
    dims = sorted({int(num(r["dim"])) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    fig, axes = plt.subplots(1, len(loads), figsize=(3.1 * len(loads), 3.2), sharey=True)
    for ax, L in zip(np.atleast_1d(axes), loads):
        sub = [r for r in rows if int(num(r["load"])) == L]
        data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r["probe-noise"]))
        for d in dims:
            if d not in data:
                continue
            draw(ax, data[d], colors[d], f"D={d}")
            xs = np.linspace(0, max(p[0] for p in data[d]), 120)
            ax.plot(xs, [e1_theory(s, d, L) for s in xs], "--", color=colors[d], linewidth=1.2)
        ax.set_title(f"|M| = {L:,}")
        ax.set_xlabel("probe noise σ")
    np.atleast_1d(axes)[0].set_ylabel("P(nearest = stored atom)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle("E1  primitive cleanup: measured (solid, Wilson 95%, 20 seeds × 20 probes)"
                 " vs theory (dashed); σ = noise norm / signal norm",
                 x=0.01, y=1.03, ha="left")
    finish(fig, "e1", blas_note(header))

    # threshold σ₅₀ vs √D, measured (interpolated) and theory
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    lcol = dict(zip(loads, CAT))
    report = []
    for L in loads:
        sub = [r for r in rows if int(num(r["load"])) == L]
        data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r["probe-noise"]))
        xs_m, ys_m, xs_t, ys_t = [], [], [], []
        for d in dims:
            if d not in data:
                continue
            s50 = crossing([p[0] for p in data[d]], [p[1] for p in data[d]], 0.5)
            grid = np.linspace(0, 80, 801)
            t50 = crossing(grid, [e1_theory(s, d, L) for s in grid], 0.5)
            xs_m.append(math.sqrt(d)); ys_m.append(s50)
            xs_t.append(math.sqrt(d)); ys_t.append(t50)
            report.append((L, d, s50, t50))
        ax.plot(xs_m, ys_m, "o-", color=lcol[L], label=f"|M|={L:,}")
        ax.plot(xs_t, ys_t, "--", color=lcol[L], linewidth=1.2)
    ax.set_xlabel("√D")
    ax.set_ylabel("σ₅₀ (noise at P = 0.5)")
    ax.legend(fontsize=8)
    ax.set_title("E1  cleanup threshold vs √D (dashed: theory)", loc="left")
    finish(fig, "e1-threshold", blas_note(header))
    print("E1 σ50 (load, D, measured, theory):")
    for L, d, s, t in report:
        print(f"  {L:>7} {d:>5}  {s:6.2f}  {t:6.2f}")


def crossing(xs, ys, level):
    """First x where ys falls through `level` (linear interpolation)."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), list(zip(xs, ys))[1:]):
        if y0 >= level > y1:
            return x0 + (y0 - level) * (x1 - x0) / (y0 - y1)
    return float("nan")


# ---------------------------------------------------------------------------
# E2 / E4: small multiples per task

TASK_TITLES = {
    "list4": "list, 4 atoms", "list16": "list, 16", "list64": "list, 64",
    "map3": "map, 3 kw→kw", "map8": "map, 8 kw→kw", "intmap8": "map, 8 kw→int",
    "set3": "set, 3", "set8": "set, 8", "nested3": "vector of 3 maps",
    "arith": "(+ 17 25)", "program": "(fact 4)",
}


def small_multiples(exp, xkey, xlabel, title, xscale=None):
    rows, header = read(exp)
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    dims = sorted({int(num(r["dim"])) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    ncol = 3 if len(tasks) > 4 else len(tasks)
    nrow = math.ceil(len(tasks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), sharex=True, sharey=True,
                             squeeze=False)
    summary = {}
    for ax, t in zip(axes.flat, tasks):
        sub = [r for r in rows if r["task"] == t]
        data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r[xkey]))
        for d in dims:
            if d in data:
                draw(ax, data[d], colors[d], f"D={d}")
                summary[(t, d)] = data[d]
        ax.set_title(TASK_TITLES.get(t, t))
        if xscale == "symlog":
            ax.set_xscale("symlog", linthresh=0.01)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(correct)")
    axes.flat[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(title, x=0.01, y=0.99, ha="left")
    finish(fig, exp, blas_note(header))
    return summary


def print_x50(summary, label):
    print(f"{label}: x at P = 0.5 (task, D)")
    for (t, d), pts in sorted(summary.items()):
        print(f"  {t:>9} {d:>5}  {crossing([p[0] for p in pts], [p[1] for p in pts], 0.5):6.3f}")


def plot_e2():
    s = small_multiples("e2", "probe-noise", "probe noise σ",
                        "E2  structure round trip under probe noise (Wilson 95%, 20 seeds)")
    print_x50(s, "E2 σ50")


def plot_e2op():
    s = small_multiples("e2op", "op-noise", "op noise σ (per bind/unbind/bundle)",
                        "E2  structure round trip under op noise (Wilson 95%, 20 seeds)")
    print_x50(s, "E2op σ50")


def plot_e4():
    s = small_multiples("e4", "lesion", "fraction of dimensions lesioned",
                        "E4  lesion after storage (Wilson 95%, 20 seeds)", xscale="symlog")
    print_x50(s, "E4 f50")


# ---------------------------------------------------------------------------
# E3: capacity

def plot_e3():
    rows, header = read("e3")
    dims = sorted({int(num(r["dim"])) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6), sharex=True, sharey=True)
    for i, (struct, parts) in enumerate([("map", 3), ("set", 2)]):
        for j, (probe, flip, ylabel) in enumerate([("get", False, "P(get correct)"),
                                                   ("absent", True, "P(false positive)")]):
            ax = axes[i, j]
            sub = [r for r in rows if r["task"] == f"{struct}-{probe}"]
            data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r["n"]), flip=flip)
            for d in dims:
                if d not in data:
                    continue
                draw(ax, data[d], colors[d], f"D={d}")
                nstar = d / (parts * 4.5 ** 2)
                ax.axvline(nstar, color=colors[d], linestyle=":", linewidth=1)
                if probe == "get":
                    ns = np.array(sorted({p[0] for p in data[d]}))
                    msize = {n: np.median([num(r["m-size"]) for r in sub
                                           if int(num(r["dim"])) == d and num(r["n"]) == n])
                             for n in ns}
                    th = [theory_get(struct, parts, n, d, msize[n]) for n in ns]
                    ax.plot(ns, th, "--", color=colors[d], linewidth=1.2)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{struct}: {ylabel}")
            if i == 1:
                ax.set_xlabel("entries n")
            if j == 0:
                ax.set_ylabel("proportion")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("E3  capacity, no noise (dotted: n* where 1/√(parts·n) = 4.5/√D;"
                 " dashed: theory)", x=0.01, y=1.03, ha="left")
    finish(fig, "e3", blas_note(header))
    for struct in ("map", "set"):
        sub = [r for r in rows if r["task"] == f"{struct}-get"]
        data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r["n"]))
        print(f"E3 {struct}-get: n at P = 0.9 / 0.5 per D")
        for d in dims:
            pts = data.get(d, [])
            print(f"  D={d:>5}  n90={crossing([p[0] for p in pts], [p[1] for p in pts], 0.9):6.1f}"
                  f"  n50={crossing([p[0] for p in pts], [p[1] for p in pts], 0.5):6.1f}")


def theory_get(struct, parts, n, d, msize):
    """Map get: the value's share 1/√(3n) must win the cleanup over |M| rows
    with chance similarities ~ N(0, 1/D). Set membership, the O(1) path:
    the share 1/√(2n) (+ noise) must clear max(share/2, 4.5/√D); the
    ambiguous band, decided by explaining away, is not modelled."""
    share = 1 / math.sqrt(parts * n)
    sd = 1 / math.sqrt(d)
    if struct == "map":
        return p_cleanup(share, sd, sd, max(1, msize - 1))
    thr = max(0.5 * share, 4.5 * sd)
    return float(1 - _phi_cdf((thr - share) / sd))


# ---------------------------------------------------------------------------
# E5: interference over a run

def plot_e5():
    rows, header = read("e5")
    dims = sorted({int(num(r["dim"])) for r in rows})
    sigmas = sorted({num(r["probe-noise"]) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(3.4 * (len(sigmas) + 1), 3.2))
    print("E5 (σ, D, n, median M, P(correct), P(diverged), median first divergence / cleanups)")
    hazard = []
    for ax, s in zip(axes, sigmas):
        for d in dims:
            sub = [r for r in rows if num(r["probe-noise"]) == s and int(num(r["dim"])) == d]
            ns = sorted({int(num(r["n"])) for r in sub})
            pts, div = [], []
            for n in ns:
                rs = [r for r in sub if int(num(r["n"])) == n]
                m = float(np.median([num(r["m-size"]) for r in rs]))
                k = sum(num(r["successes"], 0) for r in rs)
                pts.append((n, *wilson(k, len(rs)), len(rs)))
                dv = [num(r["first-divergence"]) for r in rs]
                nd = sum(1 for x in dv if x >= 0)
                refc = float(np.median([num(r["ref-cleanups"]) for r in rs]))
                fd = np.median([x for x in dv if x >= 0]) if nd else float("nan")
                div.append((n, nd / len(rs)))
                p = k / len(rs)
                if 0 < p < 1:
                    hazard.append((s, d, n, -math.log(p) / refc * 1000))
                print(f"  {s:4.2f} {d:>5} {n:>3} {m:7.0f}  {p:5.2f}  {nd / len(rs):5.2f}"
                      f"  {fd:7.0f} / {refc:7.0f}")
            draw(ax, pts, colors[d], f"D={d}")
            ax.plot([p[0] for p in div], [1 - p[1] for p in div], ":", color=colors[d],
                    linewidth=1.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"probe noise σ = {s:g}")
        ax.set_xlabel("n (recursion depth of (cd n ()))")
    axes[0].set_ylabel("P(correct)  (dotted: P(no divergence))")
    axes[0].legend(fontsize=8, loc="center right")
    ax = axes[-1]
    for s, mk in zip(sigmas, "os^"):
        for d in dims:
            pts = [(n, h) for (s2, d2, n, h) in hazard if s2 == s and d2 == d]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=colors[d],
                    label=f"D={d}, σ={s:g}")
    ax.set_title("−ln P(correct) per 1000 cleanups (0 < P < 1 only)")
    ax.set_xlabel("n")
    ax.legend(fontsize=7)
    fig.suptitle("E5  interference over a run (Wilson 95%, 20 seeds); M holds ≈ 150 + 5n traces",
                 x=0.01, y=1.03, ha="left")
    finish(fig, "e5", blas_note(header))


def plot_e5b():
    rows, header = read("e5b")
    dims = sorted({int(num(r["dim"])) for r in rows})
    sigmas = sorted({num(r["probe-noise"]) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    fig, axes = plt.subplots(1, len(sigmas), figsize=(3.6 * len(sigmas), 3.2), sharey=True)
    print("E5b (σ, D, load, P(correct))")
    for ax, s in zip(np.atleast_1d(axes), sigmas):
        sub = [r for r in rows if num(r["probe-noise"]) == s]
        data = pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r["load"]) + 1)
        for d in dims:
            if d in data:
                draw(ax, data[d], colors[d], f"D={d}")
                for p in data[d]:
                    print(f"  {s:4.1f} {d:>5} {p[0] - 1:>6.0f}  {p[1]:5.2f}")
        ax.set_xscale("log")
        ax.set_title(f"probe noise σ = {s:g}")
        ax.set_xlabel("extra atoms preloaded into M (+1)")
    np.atleast_1d(axes)[0].set_ylabel("P((cd 10 ()) correct)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("E5b  interference from |M| alone (Wilson 95%, 20 seeds)", x=0.01, y=1.03, ha="left")
    finish(fig, "e5b", blas_note(header))


# ---------------------------------------------------------------------------
# E6: cost

def plot_e6():
    rows, header = read("e6")
    dims = sorted({int(num(r["dim"])) for r in rows})
    colors = dict(zip(dims, ramp(len(dims))))
    ops = ["nearest", "clean", "deref", "bind"]
    fig, axes = plt.subplots(1, len(ops), figsize=(3.2 * len(ops), 3.2), sharey=True)
    print("E6 µs/op (backend, D, |M|): " + " ".join(ops))
    table = defaultdict(list)
    for r in rows:
        table[(r["memory"], int(num(r["dim"])), int(num(r["load"])))].append(r)
    for ax, op in zip(axes, ops):
        for mem, ls in (("codebook", "-"), ("linear", "--")):
            for d in dims:
                keys = sorted(k for k in table if k[0] == mem and k[1] == d)
                xs = [k[2] for k in keys]
                ys = [np.median([num(r[f"m.{op}"]) for r in table[k]]) for k in keys]
                ax.plot(xs, ys, ls, marker="o", color=colors[d],
                        label=f"D={d}" if mem == "codebook" else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(op)
        ax.set_xlabel("|M| (rows)")
    for k in sorted(table):
        print(f"  {k[0]:>8} {k[1]:>5} {k[2]:>6}  " +
              " ".join(f"{np.median([num(r[f'm.{op}']) for r in table[k]]):9.1f}" for op in ops))
    axes[0].set_ylabel("µs per call (median of 3)")
    axes[0].legend(fontsize=8)
    fig.suptitle("E6  cost per substrate op, in Python (solid: codebook, dashed: linear)",
                 x=0.01, y=1.03, ha="left")
    finish(fig, "e6", blas_note(header))

    path = OUT / "e6task.csv"
    if not path.exists():
        return
    rows, header = read("e6task")
    loads = sorted({int(num(r["load"])) for r in rows})
    lcol = dict(zip(loads, CAT))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    print("E6task (backend, D, load): correct/runs, wall ms, ops, rows scanned, M size")
    for mem, ls in (("codebook", "-"), ("linear", "--")):
        for L in loads:
            sub = [r for r in rows if r["memory"] == mem and int(num(r["load"])) == L]
            ds = sorted({int(num(r["dim"])) for r in sub})
            wall, opsn, rowsn = [], [], []
            for d in ds:
                rs = [r for r in sub if int(num(r["dim"])) == d]
                wall.append(np.median([num(r["wall-ms"]) for r in rs]))
                opsn.append(np.median([num(r["ops-total"]) for r in rs]))
                rowsn.append(np.median([num(r["ops.rows"]) for r in rs]))
                ok = sum(r["correct"] == "true" for r in rs)
                print(f"  {mem:>8} {d:>5} {L:>6}  {ok}/{len(rs)}  {wall[-1]:8.0f}  {opsn[-1]:8.0f}"
                      f"  {rowsn[-1]:10.0f}  {np.median([num(r['m-size']) for r in rs]):6.0f}"
                      f"  {rs[0]['error'][:50]}")
            lab = f"{mem}, +{L:,} atoms"
            axes[0].plot(ds, wall, ls, marker="o", color=lcol[L], label=lab)
            axes[1].plot(ds, opsn, ls, marker="o", color=lcol[L], label=lab)
            axes[2].plot(ds, rowsn, ls, marker="o", color=lcol[L], label=lab)
    for ax, t in zip(axes, ["wall ms per task", "substrate ops per task", "memory rows scanned per task"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(t)
        ax.set_xlabel("D")
    axes[0].legend(fontsize=7)
    fig.suptitle("E6  task-level cost of (fact 4) (median of 5 seeds; solid codebook, dashed linear)",
                 x=0.01, y=1.03, ha="left")
    finish(fig, "e6task", blas_note(header))


PLOTS = {"e1": plot_e1, "e2": plot_e2, "e2op": plot_e2op, "e3": plot_e3, "e4": plot_e4,
         "e5": plot_e5, "e5b": plot_e5b, "e6": plot_e6}


def main(argv):
    keep = "--figures" in argv
    exps = [a for a in argv if not a.startswith("--")] or list(PLOTS)
    for e in exps:
        before = set(OUT.glob("*.png"))
        PLOTS[e]()
        if keep:
            FIG.mkdir(exist_ok=True)
            for p in sorted(set(OUT.glob(f"{e}*.png")) | (set(OUT.glob("*.png")) - before)):
                if p.stem == e or p.stem.startswith(e + "-") or p.stem == e + "task":
                    shutil.copy(p, FIG / p.name)


if __name__ == "__main__":
    main(sys.argv[1:])
