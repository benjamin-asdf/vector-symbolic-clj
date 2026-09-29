"""P2 figures: the P1 curves after calibration (step 0) and the backend
comparison (codebook vs modern Hopfield vs linear).

    .venv/bin/python scripts/plot_p2.py [e2c e2opc e4c p2-calibration p2-e1 ...] [--figures]

Reads out/<exp>.csv, writes out/<name>.png (with --figures, also copies
them to figures/). Helpers, palette and Wilson intervals come from plot.py.
"""

import shutil
import sys
from collections import defaultdict

import numpy as np

import plot
from plot import (CAT, FIG, INK, INK2, OUT, SURFACE, TASK_TITLES, blas_note, crossing, draw,
                  finish, num, plt, pooled, ramp, read, small_multiples, print_x50)


# ---------------------------------------------------------------------------
# step 0: the P1 curves after calibration

def plot_e2c():
    s = small_multiples("e2c", "probe-noise", "probe noise σ",
                        "E2 after calibration: round trip under probe noise (Wilson 95%, 20 seeds)")
    print_x50(s, "E2c σ50")


def plot_e2opc():
    s = small_multiples("e2opc", "op-noise", "op noise σ (per bind/unbind/bundle)",
                        "E2 after calibration: round trip under op noise (Wilson 95%, 20 seeds)")
    print_x50(s, "E2opc σ50")


def plot_e4c():
    s = small_multiples("e4c", "lesion", "fraction of dimensions lesioned",
                        "E4 after the dense 0 and calibration (Wilson 95%, 20 seeds)", xscale="symlog")
    print_x50(s, "E4c f50")


def x50(pts):
    """x where P falls through 0.5; inf if it never does in range, 0 if it
    starts below."""
    if pts[0][1] < 0.5:
        return 0.0
    v = crossing([p[0] for p in pts], [p[1] for p in pts], 0.5)
    if v != v and pts[-1][1] >= 0.5:
        return float("inf")
    return v


def x50_table(exp, xkey):
    rows, _ = read(exp)
    data = defaultdict(dict)
    for t in dict.fromkeys(r["task"] for r in rows):
        sub = [r for r in rows if r["task"] == t]
        for d, pts in pooled(sub, lambda r: int(num(r["dim"])), lambda r: num(r[xkey])).items():
            data[t][d] = (x50(pts), max(p[0] for p in pts))
    return data


def plot_calibration():
    """σ50 (f50) vs D, before (P1) and after calibration, per structure."""
    panels = [("e2", "e2c", "probe-noise", "E2  probe noise: σ₅₀"),
              ("e2op", "e2opc", "op-noise", "E2  op noise: σ₅₀"),
              ("e4", "e4c", "lesion", "E4  lesion: f₅₀")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    print("calibration: x50 before -> after (exp, task, D); inf = never below 0.5 in range")
    for ax, (old, new, xkey, title) in zip(axes, panels):
        try:
            before, after = x50_table(old, xkey), x50_table(new, xkey)
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        tasks = [t for t in after if t in before]
        for c, t in zip(CAT + CAT, tasks):
            dims = sorted(after[t])
            top = max(after[t][d][1] for d in dims)
            ys_a = [after[t][d][0] for d in dims]
            ys_b = [before[t].get(d, (float("nan"),))[0] for d in dims]
            ax.plot(dims, [min(y, top) for y in ys_a], "o-", color=c, label=TASK_TITLES.get(t, t))
            ax.plot(dims, [min(y, top) for y in ys_b], "o--", color=c, markerfacecolor=SURFACE,
                    linewidth=1.2)
            for d, a, b in zip(dims, ys_a, ys_b):
                print(f"  {new:>6} {t:>9} {d:>5}  {b:6.3f} -> {a:6.3f}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("D")
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Step 0: fixed thresholds (dashed, hollow) vs calibrated decisions (solid); "
                 "a point at the top edge never fell below 0.5 in range", x=0.01, y=1.03, ha="left")
    finish(fig, "p2-calibration")


# ---------------------------------------------------------------------------
# backends compared

# fixed order: a backend keeps its colour and marker in every figure
BACKENDS = ["codebook", "linear", "mhn-snap-b4-i3", "mhn-snap-b16-i3", "mhn-snap-b16-i10",
            "mhn-soft-b16-i1", "mhn-soft-b64-i1", "mhn-soft-b64-i3"]
MARKERS = "osD^v<>p"
BETA_RAMP = dict(zip([1, 4, 16, 64], ramp(4)))
ITER_STYLE = {1: "-", 3: "--", 10: ":"}


def backend_style(label):
    if label in BACKENDS:
        k = BACKENDS.index(label)
        return dict(color=CAT[k], marker=MARKERS[k])
    return dict(color=INK2, marker="x")


def by_backend(exp, xkey, xlabel, title, xscale=None):
    rows, header = read(exp)
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    dims = sorted({int(num(r["dim"])) for r in rows})
    fig, axes = plt.subplots(len(tasks), len(dims), figsize=(3.3 * len(dims), 2.5 * len(tasks)),
                             sharex=True, sharey=True, squeeze=False)
    backends = [b for b in BACKENDS if any(r["backend"] == b for r in rows)]
    print(f"{exp}: x50 (task, D, backend), P at the first x")
    for i, t in enumerate(tasks):
        for j, d in enumerate(dims):
            ax = axes[i, j]
            sub = [r for r in rows if r["task"] == t and int(num(r["dim"])) == d]
            data = pooled(sub, lambda r: r["backend"], lambda r: num(r[xkey]))
            for b in backends:
                if b in data:
                    st = backend_style(b)
                    draw(ax, data[b], st["color"], b, marker=st["marker"])
                    print(f"  {t:>9} {d:>5} {b:>18}  x50 {x50(data[b]):7.3f}  P0 {data[b][0][1]:5.2f}")
            ax.set_title(f"{TASK_TITLES.get(t, t)}, D={d}")
            if xscale == "symlog":
                ax.set_xscale("symlog", linthresh=0.01)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(correct)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(title, x=0.01, y=1.0, ha="left")
    finish(fig, exp, blas_note(header))


def plot_p2_e1():
    rows, header = read("p2-e1")
    dims = sorted({int(num(r["dim"])) for r in rows})
    loads = sorted({int(num(r["load"])) for r in rows})
    groups = [("snap", 3), ("snap", 10), ("soft", 1), ("soft", 3), ("soft", 10)]
    cols = [(d, L) for d in dims for L in loads]
    fig, axes = plt.subplots(len(groups), len(cols), figsize=(2.7 * len(cols), 2.2 * len(groups)),
                             sharey=True, squeeze=False)
    s50, cos0 = {}, {}
    try:
        crows, _ = read("p2-e1c")
        for r in crows:
            r["backend"] = "classical"
    except FileNotFoundError:
        crows = []
    for j, (d, L) in enumerate(cols):
        sub = [r for r in rows + crows if int(num(r["dim"])) == d and int(num(r["load"])) == L]
        data = pooled(sub, lambda r: r["backend"], lambda r: num(r["probe-noise"]))
        for lab, pts in data.items():
            s50[(d, L, lab)] = x50(pts)
            cs = [num(r.get("m.cos")) for r in sub if r["backend"] == lab and num(r["probe-noise"]) == 0]
            cos0[(d, L, lab)] = float(np.mean(cs)) if cs else float("nan")
        for i, (mode, it) in enumerate(groups):
            ax = axes[i, j]
            for ref, col, ls in (("codebook", INK, "-"), ("linear", INK2, ":"),
                                 ("classical", CAT[7], "-.")):
                if ref in data:
                    ax.plot([p[0] for p in data[ref]], [p[1] for p in data[ref]], ls, color=col,
                            linewidth=1.3, label=ref if ref != "classical" else "classical Hopfield")
            for beta in [1, 4, 16, 64]:
                lab = f"mhn-{mode}-b{beta}-i{it}"
                if lab in data:
                    draw(ax, data[lab], BETA_RAMP[beta], f"mhn β={beta}")
            if i == 0:
                ax.set_title(f"D={d}, |M|={L:,}")
            if j == 0:
                ax.set_ylabel(f"{mode}, {it} step{'s' if it > 1 else ''}\nP(correct)")
            if i == len(groups) - 1:
                ax.set_xlabel("probe noise σ")
    h, l = axes[2, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle("P2 E1  primitive cleanup per backend: correct = the readout's nearest row is the "
                 "probed atom (Wilson 95%, 20 seeds × 20 probes); black: codebook, grey dotted: linear",
                 x=0.01, y=1.01, ha="left")
    finish(fig, "p2-e1", blas_note(header))
    print("P2 E1 σ50 and cos(readout, atom) at σ=0 (D, load, backend)")
    for k in sorted(s50):
        print(f"  {k[0]:>5} {k[1]:>6} {k[2]:>18}  σ50 {s50[k]:6.2f}  cos0 {cos0[k]:5.3f}")
    fig, axes = plt.subplots(len(loads), len(dims), figsize=(3.0 * len(dims), 2.7 * len(loads)),
                             sharey=True, sharex=True, squeeze=False)
    for ax, (d, L) in zip(axes.T.flat, cols):
        ref = s50.get((d, L, "codebook"))
        for mode, col in (("snap", CAT[0]), ("soft", CAT[1])):
            for it in (1, 3, 10):
                ys = [s50.get((d, L, f"mhn-{mode}-b{b}-i{it}"), float("nan")) / ref
                      for b in [1, 4, 16, 64]]
                if all(y != y for y in ys):
                    continue
                ax.plot([1, 4, 16, 64], ys, ITER_STYLE[it], marker="s" if mode == "snap" else "o",
                        color=col, label=f"{mode}, {it} step{'s' if it > 1 else ''}")
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.axhline(s50.get((d, L, "linear"), float("nan")) / ref, color=INK2, linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"D={d}, |M|={L:,}")
    for ax in axes[-1]:
        ax.set_xlabel("β")
    for ax in axes[:, 0]:
        ax.set_ylabel("σ₅₀ / σ₅₀(codebook)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle("P2 E1  noise tolerance relative to the codebook (black = 1, dotted = linear); "
                 "snap with 1 step is the codebook for every β", x=0.01, y=1.03, ha="left")
    finish(fig, "p2-e1-summary", blas_note(header))


def plot_p2_e2():
    by_backend("p2-e2", "probe-noise", "probe noise σ",
               "P2 E2  structure round trip under probe noise, per backend (Wilson 95%, 20 seeds)")


def plot_p2_e4():
    by_backend("p2-e4", "lesion", "fraction of dimensions lesioned",
               "P2 E4  lesion per backend (Wilson 95%, 20 seeds)", xscale="symlog")


def plot_p2_e3():
    rows, header = read("p2-e3")
    dims = sorted({int(num(r["dim"])) for r in rows})
    backends = [b for b in BACKENDS if any(r["backend"] == b for r in rows)]
    structs = [s for s in ("map", "set") if any(r["task"] == f"{s}-get" for r in rows)]
    fig, axes = plt.subplots(len(structs), len(dims) + 1,
                             figsize=(3.1 * (len(dims) + 1), 2.8 * len(structs)), squeeze=False)
    print("P2 E3 n90 / n50 (struct, D, backend); 0 = below the level at the smallest n")
    for i, st in enumerate(structs):
        n90 = defaultdict(dict)
        for j, d in enumerate(dims):
            ax = axes[i, j]
            sub = [r for r in rows if r["task"] == f"{st}-get" and int(num(r["dim"])) == d]
            data = pooled(sub, lambda r: r["backend"], lambda r: num(r["n"]))
            for b in backends:
                if b not in data:
                    continue
                sty = backend_style(b)
                draw(ax, data[b], sty["color"], b, marker=sty["marker"])
                xs, ps = [p[0] for p in data[b]], [p[1] for p in data[b]]
                a = 0.0 if ps[0] < 0.9 else crossing(xs, ps, 0.9)
                if a != a:
                    a = float(max(xs))              # never below 0.9 in range
                c = 0.0 if ps[0] < 0.5 else crossing(xs, ps, 0.5)
                n90[b][d] = a
                print(f"  {st} {d:>5} {b:>18}  n90 {a:6.1f}  n50 {c:6.1f}")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{st}: P(get correct), D={d}")
            ax.set_xlabel("entries n")
        ax = axes[i, -1]
        for b in backends:
            sty = backend_style(b)
            ds = sorted(n90[b])
            ax.plot(ds, [n90[b][d] for d in ds], "-", color=sty["color"], marker=sty["marker"], label=b)
        ax.set_xscale("log", base=2)
        ax.set_title(f"{st}: n90 vs D")
        ax.set_xlabel("D")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle("P2 E3  capacity per backend, no noise (Wilson 95%, 10 seeds); n90 = 0: below 0.9 "
                 "already at the smallest n", x=0.01, y=1.02, ha="left")
    finish(fig, "p2-e3", blas_note(header))


def plot_p2_tasks():
    rows, header = read("p2-tasks")
    benches = list(dict.fromkeys(r["bench"] for r in rows))
    fig, axes = plt.subplots(1, len(benches), figsize=(2.9 * len(benches), 3.0), sharey=True,
                             squeeze=False)
    backends = [b for b in BACKENDS if any(r["backend"] == b for r in rows)]
    print("P2 tasks: P(correct) (bench, backend, σ)")
    for ax, t in zip(axes.flat, benches):
        sub = [r for r in rows if r["bench"] == t]
        data = pooled(sub, lambda r: r["backend"], lambda r: num(r["probe-noise"]))
        for b in backends:
            if b in data:
                st = backend_style(b)
                draw(ax, data[b], st["color"], b, marker=st["marker"])
                print(f"  {t:>18} {b:>18}  " + "  ".join(f"σ={p[0]:g}: {p[1]:.2f}" for p in data[b]))
        ax.set_title(t)
        ax.set_xlabel("probe noise σ")
    axes.flat[0].set_ylabel("P(exact result)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    d = sorted({int(num(r["dim"])) for r in rows})
    fig.suptitle(f"P2  bench task accuracy per backend, D={d}, knobs on from boot (Wilson 95%)",
                 x=0.01, y=1.03, ha="left")
    finish(fig, "p2-tasks", blas_note(header))


def plot_p2_collapse():
    rows, header = read("p2-collapse")
    Ns = sorted({int(num(r["load"])) for r in rows})
    ks = sorted({int(num(r["k"])) for r in rows})
    wts = list(dict.fromkeys(r["weights"] for r in rows))
    its = sorted({int(num(r["iters"])) for r in rows})
    icol = dict(zip(its, ramp(len(its))))
    for w in wts:
        combos = [(N, k) for N in Ns for k in ks]
        fig, axes = plt.subplots(2, len(combos), figsize=(2.3 * len(combos), 4.8), sharex=True,
                                 squeeze=False)
        print(f"P2 collapse ({w} weights): β with P(PR ≥ 0.8k and noise < 0.1) ≥ 0.5 (N, k, iters)")
        for j, (N, k) in enumerate(combos):
            for it in its:
                sub = [r for r in rows if int(num(r["load"])) == N and int(num(r["k"])) == k
                       and int(num(r["iters"])) == it and r["weights"] == w]
                betas = sorted({num(r["beta"]) for r in sub})
                col = lambda key, b: np.mean([num(r[key]) for r in sub if num(r["beta"]) == b])
                pr = [col("m.pr", b) for b in betas]
                nz = [col("m.noise", b) for b in betas]
                ok = [col("successes", b) for b in betas]
                axes[0, j].plot(betas, pr, "o-", color=icol[it], label=f"{it} step{'s' if it > 1 else ''}")
                axes[1, j].plot(betas, nz, "o-", color=icol[it])
                win = [b for b, o in zip(betas, ok) if o >= 0.5]
                print(f"  N={N:>6} k={k} iters={it:>2}: " +
                      (f"β ∈ [{min(win):g}, {max(win):g}] ({len(win)} of {len(betas)} grid points)"
                       if win else "empty") +
                      "; PR/noise at β = " +
                      ", ".join(f"{b:g}: {p:.2f}/{n:.2f}" for b, p, n in zip(betas, pr, nz)
                                if b in (4.0, 16.0, 64.0)))
            axes[0, j].axhline(k, color=INK2, linewidth=0.8, linestyle=":")
            axes[0, j].axhline(0.8 * k, color=INK2, linewidth=0.8, linestyle="--")
            axes[1, j].axhline(0.1, color=INK2, linewidth=0.8, linestyle="--")
            axes[0, j].set_title(f"N={N:,}, k={k}")
            axes[1, j].set_xscale("log")
            axes[1, j].set_xlabel("β")
            axes[0, j].set_ylim(0, max(ks) + 0.5)
            axes[1, j].set_ylim(-0.03, 1.03)
        axes[0, 0].set_ylabel("items surviving (PR)")
        axes[1, 0].set_ylabel("noise + distractor energy")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"P2  β as a collapse knob: k-item probe, {w} weights (+ noise of equal norm) "
                     "through the soft mhn backend (mean over seeds); dashed: 'keeps 0.8k' / 'cleans'",
                     x=0.01, y=1.02, ha="left")
        finish(fig, f"p2-collapse-{w}", blas_note(header))


PLOTS = {"e2c": plot_e2c, "e2opc": plot_e2opc, "e4c": plot_e4c, "p2-calibration": plot_calibration,
         "p2-e1": plot_p2_e1, "p2-e2": plot_p2_e2, "p2-e3": plot_p2_e3, "p2-e4": plot_p2_e4,
         "p2-tasks": plot_p2_tasks, "p2-collapse": plot_p2_collapse}


def main(argv):
    keep = "--figures" in argv
    exps = [a for a in argv if not a.startswith("--")] or list(PLOTS)
    for e in exps:
        before = {p: p.stat().st_mtime for p in OUT.glob("*.png")}
        PLOTS[e]()
        if keep:
            FIG.mkdir(exist_ok=True)
            for p in OUT.glob("*.png"):
                fresh = p not in before or p.stat().st_mtime > before[p]
                if fresh and (p.stem == e or p.stem.startswith(e + "-")):
                    shutil.copy(p, FIG / p.name)


if __name__ == "__main__":
    main(sys.argv[1:])
