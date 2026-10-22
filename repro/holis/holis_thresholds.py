"""Trap 3 (paper §4.3): fixed thresholds cap robustness, in holis.

Their pointer memory SimpleAssoc.recall dereferences a pointer by exact key,
and otherwise by the nearest key, but only if its cosine to the probe exceeds
theta = 0.2; else it returns the probe itself. A noisy probe of norm
sigma*|p| has cosine 1/sqrt(1 + sigma^2) to its key, below 0.2 for
sigma > sqrt(24) = 4.90, whatever D is.

Control: the same code with theta = 5/sqrt(D), a threshold relative to the
chance level of cosines between unrelated vectors (sd 1/sqrt(D)); this is
the fix my paper prescribes. One attribute set on their object; everything
else, including their item memory, is theirs. (Pure argmax, theta = -1, is
not a usable control: their atomic() relies on the threshold to tell an
atom, which matches no pointer, from a cell; with theta = -1 every atom
dereferences to some cell and even noise-free reads fail.)

Three levels, probe noise on every cleanup:
  lookup   one CAR of a stored 4-list (their car(): assoc then mem)
  read     a 4-list (a b c d) read back with their read()
  program  ((LAMBDA (x) (CONS x (b c))) a)  ->  (a b c)

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_thresholds.py
"""

import math

import numpy as np

import holis_common as C

DIMS = (512, 2048, 8192)
Z = 5.0
MODES = {"theirs (theta=0.2)": lambda D: None,
         "chance floor (theta=5/sqrt(D))": lambda D: Z / math.sqrt(D)}
SIG_LOOKUP = (0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 16)
SIG_READ = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12)
SIG_PROG = (0, 2, 4, 5, 6, 8, 10)
LOOKUP_LISTS, LOOKUP_REPEATS = 25, 8      # 200 lookups per cell
READ_SEEDS, PROG_SEEDS = 20, 10
PROGRAM = "((LAMBDA (x) (CONS x (b c))) a)"
MAX_CELLS = 300   # noise-free the program builds 28 cells

out = {"meta": C.meta("holis_thresholds.py", {
    "dims": DIMS, "modes": list(MODES), "sigma_lookup": SIG_LOOKUP,
    "sigma_read": SIG_READ, "sigma_program": SIG_PROG,
    "lookup": f"{LOOKUP_LISTS} lists x {LOOKUP_REPEATS} repeats per (D, sigma, mode), env seed 0",
    "read_seeds": READ_SEEDS, "program_seeds": PROG_SEEDS, "program": PROGRAM,
    "noise": "p + sigma*|p|/sqrt(D)*N(0,I) before every SimpleCleanUp.recall and SimpleAssoc.recall",
    "max_cells_program": MAX_CELLS,
    "why_small": "their SimpleCleanUp grows by 100 zero rows per stored item past 100 rows (erratum E4); larger memories exceed 8 GB at D = 8192"}),
    "theory": {"sigma_where_cos_hits_0.2": math.sqrt(24),
               "sigma_where_cos_hits_5/sqrt(D)": {str(D): math.sqrt(D / Z**2 - 1) for D in DIMS}},
    "lookup": [], "read": [], "program": []}

# lookup: one environment per (D, mode), 50 stored 4-lists of fresh atoms
for D in DIMS:
    for mode, theta in MODES.items():
        env = C.new_env(D, 0, theta=theta(D))
        car = C.inner(env, "car")
        lists = []
        for i in range(LOOKUP_LISTS):
            atoms = [f"w{i}_{k}" for k in range(4)]
            lists.append((env["holo"](atoms), atoms[0]))
        for s in SIG_LOOKUP:
            C.set_noise(s, int(s * 100) + D)
            ok, fb, n = 0, 0, 0
            for ptr, first in lists:
                for _ in range(LOOKUP_REPEATS):
                    before = C.STATE["fallbacks"]
                    got = C.H.closest(car(ptr), env["lex"])
                    fb += int(C.STATE["fallbacks"] > before)
                    ok += int(got == first)
                    n += 1
            out["lookup"].append({"D": D, "mode": mode, "sigma": s, "correct": ok,
                                  "trials": n, "theta_fallbacks": fb})
            print("lookup", D, mode, s, ok, "/", n, "fallbacks", fb, flush=True)

# read: a fresh environment per seed; the list is built noise-free, read noisy
for D in DIMS:
    for mode, theta in MODES.items():
        for s in SIG_READ:
            ok, errs = 0, {}
            for seed in range(READ_SEEDS):
                env = C.new_env(D, seed, theta=theta(D))
                C.set_noise(0, seed)
                v = env["holo"](["a", "b", "c", "d"])
                C.set_noise(s, seed)
                r = C.attempt(lambda: env["read"](v))
                ok += int(r["value"] == ["a", "b", "c", "d"])
                if r["error"]:
                    errs[r["error"]] = errs.get(r["error"], 0) + 1
            out["read"].append({"D": D, "mode": mode, "sigma": s, "correct": ok,
                                "runs": READ_SEEDS, "errors": errs})
            print("read", D, mode, s, ok, "/", READ_SEEDS, errs, flush=True)

# program: parse noise-free, evaluate and read back under noise
for D in DIMS:
    for mode, theta in MODES.items():
        for s in SIG_PROG:
            ok, errs, wrong = 0, {}, {}
            for seed in range(PROG_SEEDS):
                env = C.new_env(D, seed, theta=theta(D), max_cells=MAX_CELLS)
                C.set_noise(0, seed)
                expr = env["holo"](C.H.parse(PROGRAM)[0])
                C.set_noise(s, seed)
                r = C.attempt(lambda: env["read"](env["evall"](expr)))
                if r["value"] == ["a", "b", "c"]:
                    ok += 1
                elif r["error"]:
                    errs[r["error"]] = errs.get(r["error"], 0) + 1
                else:
                    k = C.show(r["value"])[:60]
                    wrong[k] = wrong.get(k, 0) + 1
            out["program"].append({"D": D, "mode": mode, "sigma": s, "correct": ok,
                                   "runs": PROG_SEEDS, "errors": errs,
                                   "wrong_values": wrong})
            print("program", D, mode, s, ok, "/", PROG_SEEDS, errs, wrong, flush=True)

C.write("holis_thresholds.json", out)
