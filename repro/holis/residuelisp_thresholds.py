"""Trap 3 (paper §4.3): fixed thresholds, checked in residuelisp.

residuelisp keeps holis's fixed numbers (AssociativeMemory theta = 0.2,
is_approx_eq floor = 0.2), but compares them with an unnormalised FHRR
similarity, Re(x^H y)/D. A noisy probe keeps similarity ~1 to its own key,
whatever sigma is; what grows is the spread of everything else,
~sqrt((1 + sigma^2)/(2D)). So a fixed 0.2 is crossed by chance at a sigma
that grows like sqrt(D), and I expect the noise cliff to move with D,
unlike holis.

Program (no integers, so runs are fast):
  ((lambda (x) ((lambda (y) (cons x (cons y nil))) n)) m)  ->  (m n)
Probe noise on every cleanup and every deref scan, during evaluation and
read-back, 10 seeds per (D, sigma).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python residuelisp_thresholds.py
"""

import residuelisp_common as C

PROGRAM = "((lambda (x) ((lambda (y) (cons x (cons y nil))) n)) m)"
DIMS = (500, 2000, 8000)
SIGMAS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SEEDS = range(10)
MAX_STEPS = 5000   # cleanups + derefs per run; noise-free the program needs far fewer

rows = []
for D in DIMS:
    for s in SIGMAS:
        ok, errs, wrong = 0, {}, {}
        for seed in SEEDS:
            r = C.attempt(lambda: C.evaluate(PROGRAM, dim=D, seed=seed, sigma=s, max_steps=MAX_STEPS)[0])
            if r["value"] == ["m", "n"]:
                ok += 1
            elif r["error"]:
                errs[r["error"]] = errs.get(r["error"], 0) + 1
            else:
                k = C.show(r["value"])[:60]
                wrong[k] = wrong.get(k, 0) + 1
        rows.append({"D": D, "sigma": s, "correct": ok, "runs": len(SEEDS),
                     "errors": errs, "wrong_values": wrong})
        print(D, s, ok, "/", len(SEEDS), errs, wrong, flush=True)

C.write("residuelisp_thresholds.json", {
    "meta": C.meta("residuelisp_thresholds.py", {
        "program": PROGRAM, "max_steps": MAX_STEPS, "dims": DIMS, "sigmas": SIGMAS, "seeds": len(SEEDS),
        "noise": "p + sigma*|p|/sqrt(D)*CN(0,I) before every CleanupMemory.recall and AssociativeMemory.deref"}),
    "rows": rows})
