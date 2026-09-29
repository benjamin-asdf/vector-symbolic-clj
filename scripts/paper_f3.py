"""F3: residue-integer chimeras (paper §4.3).

x = nu(B^11 + B^12). Every integer m whose residue mod each prime p equals
either 11 mod p or 12 mod p matches x on every band: 2^6 = 64 such m in
[0, P), two of them true members, 62 chimeras. Left panel: cos(x, B^m) for
the true members, the chimeras, and random integers. Right panel: the
per-band match of x with one chimera.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/paper_f3.py
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "resources/vsc")
import hdc  # noqa: E402

space = hdc.Space(2048, 42)
nums = hdc.Numbers(space)
P, moduli = nums.P, nums.moduli


def crt(residues):
    return nums._crt(list(residues))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


x = nums.vec(11) + nums.vec(12)
combos = [crt(r) for r in itertools.product(*[(11 % p, 12 % p) for p in moduli])]
true = [11, 12]
chimeras = sorted(set(combos) - set(true))
rng = np.random.default_rng(0)
randoms = [int(m) for m in rng.integers(-nums.half, nums.half, 200)]

s_true = [cos(x, nums.vec(m)) for m in true]
s_chim = [cos(x, nums.vec(m)) for m in chimeras]
s_rand = [cos(x, nums.vec(m)) for m in randoms]
print("members", [round(v, 3) for v in s_true])
print("chimeras", len(chimeras), "cos mean", round(np.mean(s_chim), 3),
      "min", round(min(s_chim), 3), "max", round(max(s_chim), 3))
print("random cos mean", round(np.mean(s_rand), 3), "max", round(max(s_rand), 3))
print("example chimeras", chimeras[:6])

# per-band agreement of x with one chimera: share of the band's energy
X = np.fft.rfft(x) * nums.zconj
ex = chimeras[len(chimeras) // 2]
C = np.fft.rfft(nums.vec(ex)) * nums.zconj
band_cos = [float((np.conj(C[b]) @ X[b]).real / np.linalg.norm(C[b]) / np.linalg.norm(X[b]))
            for b in nums.bands]
which = ["11" if ex % p == 11 % p else "12" for p in moduli]
print("chimera", ex, "matches", dict(zip(moduli, which)))

os.makedirs("out", exist_ok=True)
json.dump({"members": s_true, "chimeras": s_chim, "random": s_rand,
           "example": ex, "band_cos": band_cos, "which": which},
          open("out/paper-f3.json", "w"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
jit = np.random.default_rng(1)
for row, (vals, c, name) in enumerate(((s_rand, "#999999", "200 random integers"),
                                       (s_chim, "#c0392b", f"{len(chimeras)} chimeras"),
                                       (s_true, "#2471a3", "members 11, 12"))):
    a1.scatter(vals, row + jit.uniform(-0.18, 0.18, len(vals)), s=10, color=c)
a1.set_yticks([0, 1, 2], ["200 random\nintegers", f"{len(chimeras)}\nchimeras",
                          "members\n11, 12"])
a1.set_xlabel("cos(ν(B¹¹ + B¹²), Bᵐ)")
a1.set_title("every chimera scores exactly like a member")
a2.bar(range(len(moduli)), band_cos,
       color=["#2471a3" if w == "11" else "#27ae60" for w in which])
a2.set_xticks(range(len(moduli)), [f"p={p}\n≡{w}" for p, w in zip(moduli, which)])
a2.set_ylabel("per-band cos with x")
a2.set_title(f"chimera {ex}: each band matches 11 or 12")
fig.tight_layout()
fig.savefig("figures/paper-f3-chimeras.png", dpi=150)
print("wrote figures/paper-f3-chimeras.png", file=sys.stderr)
