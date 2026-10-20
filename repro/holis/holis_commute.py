"""4.2 Binding commutes, in holis.

A cell is lhs*a + rhs*b + phi with circular convolution, which commutes.
The roles lhs, rhs and the tag phi are also ordinary Lisp atoms: vsa_env puts
them in the lexicon under the names 'lhs', 'rhs', 'φ', so a program that
mentions the symbol lhs gets the role vector itself. Then the cell (x . lhs)
contains lhs*x + rhs*lhs, and unbinding lhs finds x and rhs at equal
strength, because rhs*lhs = lhs*rhs. The paired cells (x . y) / (y . x),
which have no role symbols, are the control.

Each program goes through their parser, evaluator and reader.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_commute.py
"""

from collections import Counter

import holis_common as C

PROGRAMS = [
    ("(CAR (x . lhs))", "x"),
    ("(CDR (rhs . x))", "x"),
    ("(CAR (x . rhs))", "x"),
    ("(CDR (lhs . x))", "x"),
    ("(CAR (x . y))", "x"),
    ("(CAR (y . x))", "y"),
    ("(CDR (x . y))", "y"),
    ("(CDR (y . x))", "x"),
]
DIMS = (512, 2048, 8192)
SEEDS = range(40)

C.set_noise(0, 0)
rows = []
for D in DIMS:
    for src, want in PROGRAMS:
        got = Counter()
        for seed in SEEDS:
            env = C.new_env(D, seed)
            r = C.attempt(lambda: C.run(env, src))
            v = r["value"]
            got[C.show(v[0] if v else r["error"])] += 1
        rows.append({"program": src, "D": D, "expected": want,
                     "correct": got.get(want, 0), "runs": len(SEEDS),
                     "results": dict(got.most_common())})
        print(D, src, dict(got), flush=True)

C.write("holis_commute.json", {
    "meta": C.meta("holis_commute.py", {"dims": DIMS, "seeds": len(SEEDS), "sigma": 0}),
    "rows": rows})
