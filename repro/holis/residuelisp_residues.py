"""4.3 residue chimeras and 4.4 zero, against residuelisp's own RHC code.

Their integer code is Kymn et al.'s product encoding: zeta(x) is the
elementwise product over moduli m of z_m**x, each z_m a vector of random
m-th roots of unity. Every coordinate therefore mixes all moduli. Our
encoding gave each modulus its own band of coordinates, and the chimeras
and the zero problem came from those bands. This script measures, with
their RHC.encode, decode_rhc (the interpreter's decoder) and
resonator_decoding (in their library; the interpreter does not call it):

  * cos(zeta(11) + zeta(12), zeta(x)) for the members, for every chimera
    (an integer whose residue mod each modulus matches one of the members),
    and for all other integers in 0..1154;
  * what decode_rhc and resonator_decoding read from that superposition,
    and from a single integer as control;
  * cos(zeta(0), zeta(n)) for n divisible by some of the moduli;
  * (+ 17 25) through their algebra with coordinate 0 zeroed (our 4.4
    lesion: in our time-domain HRR, B^0 was a delta at index 0).

Superposed integers never arise inside a residuelisp program: integers are
built only from literals and + / -, and cells hold them behind pointers.
So everything here is at the level of their library.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python residuelisp_residues.py
"""

import math
from itertools import product

import numpy as np

import residuelisp_common as C

V, I = C.V, C.I
MODULI = V.DEFAULT_MODULI
M = math.prod(MODULI)
DIMS = (1000, 4096)
SEEDS = range(5)


def cos(x, y):
    return float(np.real(np.vdot(y, x)) / (np.linalg.norm(x) * np.linalg.norm(y)))


def chimeras(a, b):
    res = []
    for choice in product(*[(a % m, b % m) for m in MODULI]):
        x = V.crt(list(choice), MODULI)
        if x not in (a, b):
            res.append(x)
    return sorted(set(res))


def resonate(data, iters=200):
    num = V.RHC(data, z_m=V.RHC.z_ms)       # their bases, not fresh ones
    hist, books = V.resonator_decoding(num, max_iters=iters)
    last = hist[-1]
    # a resonator recovers each factor up to a phase (factors can swap signs
    # in pairs), so read each residue by magnitude
    residues = [int(np.argmax(np.abs(np.conj(books[m]).T @ last[m]))) for m in MODULI]
    return residues, V.crt(residues, MODULI)


A, B = 11, 12
CH = chimeras(A, B)
out = {"meta": C.meta("residuelisp_residues.py", {
    "dims": DIMS, "seeds": list(SEEDS), "moduli": MODULI, "range": M,
    "superposition": f"zeta({A}) + zeta({B})", "resonator_iters": 200}),
    "chimeras_of_11_12": CH, "superposition": [], "zero": [], "lesion": []}

for D in DIMS:
    for seed in SEEDS:
        C.reset(seed)
        enc = C.encode_env(D)
        Z = {x: V.RHC.encode(D, x).data for x in range(M)}
        sup = Z[A] + Z[B]
        sims = {x: cos(sup, Z[x]) for x in range(M)}
        others = [abs(sims[x]) for x in range(M) if x not in CH and x not in (A, B)]
        dec = I.decode_rhc(V.RHC(sup, z_m=V.RHC.z_ms), enc)
        dec1 = I.decode_rhc(V.RHC(Z[A], z_m=V.RHC.z_ms), enc)
        res_sup, x_sup = resonate(sup)
        res_one, x_one = resonate(Z[A])
        row = {"D": D, "seed": seed,
               "cos_member_11": sims[A], "cos_member_12": sims[B],
               "cos_chimera_mean": float(np.mean([sims[x] for x in CH])),
               "cos_chimera_max": float(np.max([sims[x] for x in CH])),
               "abs_cos_other_mean": float(np.mean(others)),
               "abs_cos_other_max": float(np.max(others)),
               "decode_rhc(sup)": dec, "decode_rhc(11)": dec1,
               "resonator(sup)": x_sup, "resonator(sup)_residues": res_sup,
               "resonator(sup)_is_chimera": x_sup in CH,
               "resonator(11)": x_one}
        out["superposition"].append(row)
        print(row, flush=True)

        z0 = Z[0]
        zero = {"D": D, "seed": seed, "max|zeta(0)-1|": float(np.max(np.abs(z0 - 1)))}
        for n in (1, 3, 15, 105, 385, 1155 - 1, 1155):
            zero[f"cos(0,{n})"] = cos(z0, V.RHC.encode(D, n).data)
        out["zero"].append(zero)

        x, y = Z[17].copy(), Z[25].copy()
        x[0] = 0
        y[0] = 0
        s = x * y
        best = max(range(200), key=lambda k: cos(s, Z[k]))
        out["lesion"].append({"D": D, "seed": seed, "argmax_0_199(lesioned 17*25)": best,
                              "cos_to_42": cos(s, Z[42])})

C.write("residuelisp_residues.json", out)
