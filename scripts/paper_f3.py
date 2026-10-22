"""Residue-integer chimeras of the old banded encoding (history page, §2).

x = nu(B^11 + B^12). In the banded encoding (one band of Fourier bins per
prime, my first version, rebuilt below as `Banded`), every integer m whose
residue mod each prime p equals either 11 mod p or 12 mod p matches x on
every band: 2^6 = 64 such m in [0, P), two of them true members, 62
chimeras. Left panel: cos(x, B^m) for the true members, the chimeras and
random integers, banded, and the same chimeras in the product code the
interpreter now uses (`hdc.Numbers`). Right panel: the per-band match of x
with one chimera, banded.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/paper_f3.py
"""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "resources/vsc")
import hdc  # noqa: E402


class Banded:
    """The banded residue encoding of hdc.Numbers before the product code:
    the bins split at random into one band per prime p, on which every
    coefficient of B is a p-th root of unity (n = B^n ⊗ Z, Z as there)."""

    def __init__(self, space, moduli=(5, 7, 11, 13, 17, 19)):
        d = space.dim
        self.dim, self.moduli = d, list(moduli)
        self.P = int(np.prod(self.moduli))
        self.half = (self.P - 1) // 2
        rng = np.random.default_rng([space.seed, 5])
        bins = np.arange(1, d // 2 if d % 2 == 0 else d // 2 + 1)
        rng.shuffle(bins)
        self.bands = np.array_split(bins, len(self.moduli))
        self.expo = np.zeros(d // 2 + 1)
        for band, p in zip(self.bands, self.moduli):
            self.expo[band] = rng.integers(1, p, size=band.size) / p
        self.zspec = hdc.Numbers(space).zspec
        self.zconj = np.conj(self.zspec)

    def vec(self, n):
        spec = np.exp(2j * np.pi * ((self.expo * int(n)) % 1.0)) * self.zspec
        return np.fft.irfft(spec, n=self.dim)

    def crt(self, residues):
        n = 0
        for r, p in zip(residues, self.moduli):
            q = self.P // p
            n += r * q * pow(q, -1, p)
        n %= self.P
        return n - self.P if n > self.half else n


space = hdc.Space(2048, 42)
nums = Banded(space)
prod = hdc.Numbers(space)
P, moduli = nums.P, nums.moduli


def crt(residues):
    return nums.crt(list(residues))


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
xp = prod.vec(11) + prod.vec(12)
p_true = [cos(xp, prod.vec(m)) for m in true]
p_chim = [cos(xp, prod.vec(m)) for m in chimeras]
p_rand = [cos(xp, prod.vec(m)) for m in randoms]
print("product code: members", [round(v, 3) for v in p_true], "chimeras cos mean",
      round(np.mean(p_chim), 3), "max", round(max(p_chim), 3),
      "random mean", round(np.mean(p_rand), 3), "max", round(max(p_rand), 3))
print("product code: sim(0, 5005)", round(cos(prod.vec(0), prod.vec(5005)), 3),
      "banded:", round(cos(nums.vec(0), nums.vec(5005)), 3))

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
           "example": ex, "band_cos": band_cos, "which": which,
           "product": {"members": p_true, "chimeras": p_chim, "random": p_rand}},
          open("out/paper-f3.json", "w"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
jit = np.random.default_rng(1)
for row, (vals, c, name) in enumerate(((p_chim, "#7f8c8d", "chimeras, product code"),
                                       (s_rand, "#999999", "200 random integers"),
                                       (s_chim, "#c0392b", f"{len(chimeras)} chimeras"),
                                       (s_true, "#2471a3", "members 11, 12"))):
    a1.scatter(vals, row + jit.uniform(-0.18, 0.18, len(vals)), s=10, color=c)
a1.set_yticks([0, 1, 2, 3], [f"{len(chimeras)} fakes,\nproduct code", "200 random\nintegers",
                             f"{len(chimeras)}\nfakes", "members\n11, 12"])
a1.set_xlabel("similarity to the set {11, 12}")
a1.set_title("banded: every fake scores like a member")
a2.bar(range(len(moduli)), band_cos,
       color=["#2471a3" if w == "11" else "#27ae60" for w in which])
a2.set_xticks(range(len(moduli)), [f"{p} h\nlike {w}" for p, w in zip(moduli, which)])
a2.set_ylabel("match with the set, per clock")
a2.set_title(f"fake {ex}: every clock matches 11 or 12")
fig.tight_layout()
fig.savefig("figures/paper-f3-chimeras.png", dpi=150)
print("wrote figures/paper-f3-chimeras.png", file=sys.stderr)
