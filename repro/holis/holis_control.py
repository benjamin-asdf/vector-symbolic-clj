"""Control: their interpreter, unmodified apart from loading, works noise-free.

Small programs over 5 seeds at D = 512 (their default) and D = 2048, and the
authors' own fizzbuzz helpers (their DEFINEs, verbatim from the notebook)
at D = 512, one seed, because each call takes minutes.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_control.py
"""

import time

import holis_common as C

SMALL = [
    ("(CAR (a b c))", ["a"]),
    ("(CDR (a b c))", [["b", "c"]]),
    ("(CONS a (b c))", [["a", "b", "c"]]),
    ("(EQ a a)", ["T"]),
    ("(EQ a b)", ["F"]),
    ("(ATOM a)", ["T"]),
    ("(ATOM (a b))", ["F"]),
    ("(COND ((EQ a a) . yes) (T . no))", ["yes"]),
    ("(COND ((EQ a b) . yes) (T . no))", ["no"]),
    ("((LAMBDA (x) (CONS x (b c))) a)", [["a", "b", "c"]]),
]


def num(n):
    s = "NIL"
    for _ in range(n):
        s = f"({s})"
    return s


# Peano numerals as in their notebook: 0 = NIL, n+1 = (n)
FIZZ = [
    (f"(countdown {num(3)} {num(1)})", "countdown 3 1", [[["NIL"]]]),   # 2
    (f"(divisible {num(3)} {num(3)})", "divisible 3 3", ["T"]),
    (f"(divisible {num(4)} {num(3)})", "divisible 4 3", ["F"]),
    ("(and T F)", "and T F", ["F"]),
    (f"(take {num(2)} (a b c))", "take 2 (a b c)", [["a", "b"]]),
]

SEEDS = range(5)
rows = []
C.set_noise(0.0, 0)
for D in (512, 2048):
    for src, want in SMALL:
        ok = 0
        for seed in SEEDS:
            env = C.new_env(D, seed)
            r = C.attempt(lambda: C.run(env, src))
            ok += int(r["value"] == want)
        rows.append({"program": src, "D": D, "expected": C.show(want),
                     "correct": ok, "runs": len(SEEDS)})
        print(D, src, ok, "/", len(SEEDS), flush=True)

fizz = []
env = C.new_env(512, 0)
t = time.time()
defs = C.run(env, C.H.lisp_fizzbuzz_program)
fizz.append({"program": "their fizzbuzz DEFINEs", "result": C.show(defs),
             "expected": C.show(["NIL"] * 9), "correct": defs == ["NIL"] * 9,
             "seconds": round(time.time() - t, 1)})
for src, label, want in FIZZ:
    t = time.time()
    r = C.attempt(lambda: C.run(env, src))
    fizz.append({"program": label, "source": src, "result": C.show(r["value"] or r["error"]),
                 "expected": C.show(want), "correct": r["value"] == want,
                 "seconds": round(time.time() - t, 1)})
    print(fizz[-1], flush=True)

C.write("holis_control.json", {
    "meta": C.meta("holis_control.py", {"D": [512, 2048], "seeds": list(SEEDS),
                                        "fizzbuzz_D": 512, "fizzbuzz_seed": 0,
                                        "sigma": 0}),
    "small": rows, "fizzbuzz": fizz})
