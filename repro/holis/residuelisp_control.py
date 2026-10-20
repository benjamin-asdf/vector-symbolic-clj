"""Control: residuelisp works noise-free on the programs its design covers.

D = 1000 (the dimension of their main.py), FHRR with RHC integers, 5 seeds.
A space before every ')' that follows an integer works around a lexer
issue (see residuelisp_errata.py).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python residuelisp_control.py
"""

import residuelisp_common as C

PROGRAMS = [
    ("(+ 1 2 )", "3"),
    ("(+ 20 22 )", "42"),
    ("(- 7 3 )", "4"),
    ("(car (cons 1 2 ) )", "1"),
    ("(cdr (cons a b))", "b"),
    ("((lambda (x) ((lambda (y) (cons x (cons y nil))) n)) m)", ["m", "n"]),
    ("(eq? a a)", "#t"),
    ("(eq? a b)", "#f"),
    ("(eq? 3 3 )", "#t"),
    ("(int? 3 )", "#t"),
    ("(int? a)", "#f"),
    ("(and #t #t)", "#t"),
    ("(and #f #t)", "#f"),
    ("(if #t yes no)", "yes"),
    ("(if #f yes no)", "no"),
]
SEEDS = range(5)
D = 1000

rows = []
for src, want in PROGRAMS:
    ok, seen = 0, {}
    for seed in SEEDS:
        r = C.attempt(lambda: C.evaluate(src, dim=D, seed=seed)[0])
        v = C.show(r["value"] if r["error"] is None else r["error"])
        seen[v] = seen.get(v, 0) + 1
        ok += int(r["value"] == want)
    rows.append({"program": src, "expected": C.show(want), "correct": ok,
                 "runs": len(SEEDS), "results": seen})
    print(src, ok, "/", len(SEEDS), seen, flush=True)

C.write("residuelisp_control.json", {
    "meta": C.meta("residuelisp_control.py", {"D": D, "vsa": "FHRR",
                                              "integers": "RHCIntegers, moduli [3, 5, 7, 11]",
                                              "seeds": list(SEEDS), "sigma": 0}),
    "rows": rows})
