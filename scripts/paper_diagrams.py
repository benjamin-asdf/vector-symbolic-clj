"""F1 (architecture, paper §3) and F8 (the CEK machine, paper §7.2).

    .venv/bin/python scripts/paper_diagrams.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HOST, VEC, SUB = "#fdebd0", "#d6eaf8", "#e8f6e8"


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="#333333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)


def arrow(ax, a, b, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 lw=0.9, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03, text,
                ha="center", fontsize=7.5, style="italic")


def f1():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4), ax.axis("off")
    box(ax, 0.1, 1.9, 1.5, 0.8, "reader\n(host)\ntext → vectors", HOST)
    box(ax, 8.4, 1.9, 1.5, 0.8, "printer\n(host)\nvectors → text", HOST)
    ax.add_patch(FancyBboxPatch((2.0, 0.95), 6.0, 2.85,
                                boxstyle="round,pad=0.02", fc="none",
                                ec="#2471a3", lw=1.2, ls="--"))
    ax.text(5.0, 3.62, "vector machine: every value is one HRR vector",
            ha="center", fontsize=9, color="#2471a3", weight="bold")
    box(ax, 2.2, 2.35, 2.2, 1.0,
        "evaluator\nveval (§3) or\nCEK machine (§7.2)", VEC)
    box(ax, 4.7, 2.35, 3.1, 1.0,
        "item memory M\natoms · pointers p ↦ traces\n(hash-consed; label = kind)", VEC)
    box(ax, 2.2, 1.1, 2.2, 0.95, "globals F\nprobe L⊗sym", VEC)
    box(ax, 4.7, 1.1, 3.1, 0.95,
        "integer readout\nresidue bands → CRT\nn = Bⁿ⊗Z", VEC)
    box(ax, 2.0, 0.1, 6.0, 0.6,
        "substrate: Python/numpy — ⊗ ⊘ ν cos, cleanup (table · Hopfield · proj)",
        SUB)
    arrow(ax, (1.6, 2.3), (2.2, 2.8), "vectors")
    arrow(ax, (7.8, 2.8), (8.4, 2.3), "vectors")
    arrow(ax, (4.4, 2.85), (4.7, 2.85))
    arrow(ax, (3.3, 2.35), (3.3, 2.05))
    arrow(ax, (6.25, 2.35), (6.25, 2.05))
    ax.text(0.85, 3.3, "host touches\nplain data only here", ha="center",
            fontsize=7.5, color="#a04000")
    ax.text(9.15, 3.3, "…and here", ha="center", fontsize=7.5, color="#a04000")
    fig.tight_layout()
    fig.savefig("figures/paper-f1-architecture.png", dpi=150)


def f8():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.6), ax.axis("off")
    box(ax, 0.2, 3.3, 4.2, 1.0,
        "state  s ↦ ν(C⊗control + E⊗env\n+ K⊗continuation + M⊗mode)\n"
        "continuation = list of 9 frame types in W", VEC)
    box(ax, 5.2, 3.3, 4.6, 1.0,
        "fetch (12 instructions) → key(s) =\nν(FM⊗mode + FK⊗kind(C) + FH⊗kind⊗class(car C)\n"
        "+ FI⊗kind⊗class(C) + FF⊗type(top frame))", VEC, fs=8)
    box(ax, 5.2, 1.75, 4.6, 1.05,
        "rule memory R: 35 rules, nearest key wins\nsum keys: j/√(5k) ⇒ most specific rule\n"
        "short keys act as defaults", VEC)
    box(ax, 0.2, 1.75, 4.2, 1.05,
        "microprogram = list of instruction vectors\n9 ops: FIELD CLEAN BIND UNBIND REC\n"
        "MOV JMPEQ PRIM DEF · 13 registers", VEC)
    box(ax, 0.2, 0.2, 9.6, 1.0,
        "host loop, 47 lines, no Lisp:  loop { KEY ← fetch(s);  r ← R.nearest(KEY);  "
        "s ← run(r, s);  collect W } until HALT", HOST, bold=True)
    arrow(ax, (4.4, 3.8), (5.2, 3.8), "shape")
    arrow(ax, (7.5, 3.3), (7.5, 2.8), "nearest")
    arrow(ax, (5.2, 2.27), (4.4, 2.27), "run")
    arrow(ax, (2.3, 2.8), (2.3, 3.3), "new state")
    fig.tight_layout()
    fig.savefig("figures/paper-f8-cek-machine.png", dpi=150)


f1()
f8()
print("wrote figures/paper-f1-architecture.png, figures/paper-f8-cek-machine.png")
