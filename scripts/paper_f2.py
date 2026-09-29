"""F2: why superposed cells fail and pointer cells don't (paper §4.1).

Rebuilds the original encoding, a cell as nu(L*a + R*b + phi), next to the
pointer encoding (cell identity is a fresh unitary pointer, the trace sits
behind it), and replays the environment chains that recursion creates:
env(x, y) = ((b . y) . ((a . x) . ())) for many x, y, as in (* a b)
recursing on b. For every env it probes cdr, which should return the
chain ((a . x) . ()), and records whether cleanup finds it or a sibling
((a . x') . ()), plus the margin between the two.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/paper_f2.py
"""

import json
import os
import sys

import numpy as np

D = int(os.environ.get("D", 2048))
rng = np.random.default_rng(7)


def unitary():
    spec = np.exp(1j * rng.uniform(-np.pi, np.pi, D // 2 + 1))
    spec[0] = rng.choice([-1.0, 1.0])
    spec[-1] = rng.choice([-1.0, 1.0])
    return np.fft.irfft(spec, n=D)


def bind(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=D)


def unbind(a, c):
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(c), n=D)


def nu(v):
    return v / np.linalg.norm(v)


L, R, PHI = unitary(), unitary(), unitary()
EMPTY, A, B = unitary(), unitary(), unitary()


def run(encoding, n_x, n_y, n_atoms=500, sigma=0.0):
    """Probe cdr of every env, with relative probe noise sigma; return
    (P(correct), margins, sibling sims)."""
    items = [unitary() for _ in range(n_atoms)]  # a vocabulary, as in M
    nums = [unitary() for _ in range(max(n_x, n_y))]
    memory, traces = [], {}

    def store(v):
        memory.append(v)
        return v

    def cell(a, b):
        t = nu(bind(L, a) + bind(R, b) + (PHI if encoding == "superposed" else 0))
        if encoding == "superposed":
            return store(t)
        p = store(unitary())
        traces[len(memory) - 1] = t
        return p

    for v in items + [EMPTY, A, B] + nums:
        store(v)
    rests = {x: cell(cell(A, nums[x]), EMPTY) for x in range(n_x)}
    envs = [(x, cell(cell(B, nums[y]), rests[x]))
            for x in range(n_x) for y in range(n_y)]
    M = np.array(memory)
    keys = list(range(len(memory)))
    rest_idx = {x: next(i for i in keys if memory[i] is rests[x]) for x in rests}

    correct, margins = 0, []
    for x, env in envs:
        trace = traces[next(i for i in keys if memory[i] is env)] \
            if encoding == "pointer" else env
        probe = nu(unbind(R, trace))
        probe = nu(probe + sigma / np.sqrt(D) * rng.normal(size=D))
        s = M @ probe
        true = s[rest_idx[x]]
        sib = max(s[rest_idx[x2]] for x2 in rests if x2 != x) if n_x > 1 else -1
        margins.append(float(true - sib))
        correct += int(np.argmax(s) == rest_idx[x])
    sib_sims = [float(nu(rests[0]) @ nu(rests[x])) for x in range(1, n_x)]
    return correct / len(envs), margins, sib_sims


SIGMAS = (0, 0.25, 0.5, 1, 2, 4, 8, 16)
out = {"D": D, "curve": {}, "noise": {}, "margins": {}, "sibling_sim": {}}
for enc in ("superposed", "pointer"):
    out["curve"][enc] = []
    for n_x in (2, 8, 32, 64):
        p, _, _ = run(enc, n_x, 8)
        out["curve"][enc].append([n_x, p])
        print(enc, "siblings", n_x, "P(cdr correct)", round(p, 3), flush=True)
    out["noise"][enc] = []
    for sg in SIGMAS:
        p, _, _ = run(enc, 32, 8, sigma=sg)
        out["noise"][enc].append([sg, p])
        print(enc, "sigma", sg, "P(cdr correct)", round(p, 3), flush=True)
    _, m, s = run(enc, 32, 8)
    out["margins"][enc] = m
    out["sibling_sim"][enc] = s
    print(enc, "margin mean", round(float(np.mean(m)), 3),
          "min", round(float(np.min(m)), 3),
          "sibling cos mean", round(float(np.mean(s)), 3), flush=True)

os.makedirs("out", exist_ok=True)
json.dump(out, open("out/paper-f2.json", "w"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
for enc, c in (("superposed", "#c0392b"), ("pointer", "#2471a3")):
    a1.hist(out["margins"][enc], bins=40, alpha=0.7, color=c, label=enc)
    xs, ps = zip(*out["noise"][enc])
    a2.plot([max(x, 0.125) for x in xs], ps, "o-", color=c, label=enc)
a1.axvline(0, color="k", lw=0.8)
a1.set_xlabel("cos(probe, true rest) − max cos(probe, sibling rest)")
a1.set_ylabel("count")
a1.set_title(f"margin of cdr cleanup, 32 sibling chains, D={D}")
a1.legend(frameon=False)
a2.set_xscale("log", base=2)
a2.set_xlabel("probe noise σ (σ = 0 plotted at 1/8)")
a2.set_ylabel("P(cdr cleans up to the right cell)")
a2.set_ylim(-0.03, 1.03)
a2.set_title(f"cdr under noise, 32 sibling chains, D={D}")
a2.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/paper-f2-cell-confusion.png", dpi=150)
print("wrote figures/paper-f2-cell-confusion.png", file=sys.stderr)
