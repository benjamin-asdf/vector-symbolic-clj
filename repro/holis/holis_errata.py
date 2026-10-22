"""Errata in holis that my list did not cover, all noise-free.

E1  CAR of an atom.  Their evaluator decides "is this head a lambda
    expression?" by (car(head) == LAMBDA) > 0.2, and "is it a defined
    function?" by car(func.recall(lhs*head)) == head, also when head is an
    atom or the empty function memory returns a zero row. An atom has no CAR:
    car() then cleans up a meaningless probe and returns an arbitrary stored
    item. When that item happens to be LAMBDA (or a matching pointer) the
    evaluator takes a wrong branch.
    Measured: ((LAMBDA (x) (CONS x (b c))) a) over 100 seeds at D = 512 and
    2048. Control: the same code with car() answering NIL for anything that
    is not a stored pointer (exact key in their pointer memory), patched into
    their closures; nothing else changes.
E2  CAR and CDR do not evaluate their argument: (CAR (CDR (a b c))) -> CDR.
E3  LABEL is dispatched as a basic function but raises (the paper says LABEL
    is omitted; the notebook still lists it).
E4  SimpleCleanUp.memorize never raises its capacity m, so past 100 items
    each stored item appends another 100 zero rows: memory and lookup time
    grow quadratically with the number of items.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_errata.py
"""

from collections import Counter

import holis_common as C

H = C.H
PROGRAM = "((LAMBDA (x) (CONS x (b c))) a)"
SEEDS = range(100)


def closures(env):
    seen, todo, out = set(), [env[k] for k in env if callable(env[k])], []
    while todo:
        f = todo.pop()
        if id(f) in seen or not hasattr(f, "__code__"):
            continue
        seen.add(id(f))
        out.append(f)
        for cell in f.__closure__ or ():
            try:
                todo.append(cell.cell_contents)
            except ValueError:
                pass
    return out


def guard_car(env):
    """Make their car() return NIL for a non-pointer. Rebinds the free
    variable `car` in every closure of vsa_env that uses it."""
    orig = C.inner(env, "car")
    assoc, nil = env["assoc"], env["NIL"]

    def car(c, ind=""):
        return orig(c) if c in assoc.A else nil

    for f in closures(env):
        for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            if name == "car":
                cell.cell_contents = car


C.set_noise(0, 0)
out = {"meta": C.meta("holis_errata.py", {"program": PROGRAM, "seeds": len(SEEDS),
                                          "dims": [512, 2048], "sigma": 0}),
       "car_of_atom": [], "car_of_atom_examples": [], "car_of_a_spread": {},
       "other": []}

for D in (512, 2048):
    for variant in ("theirs", "car(atom) = NIL"):
        wrong, examples, spread = 0, [], Counter()
        for seed in SEEDS:
            env = C.new_env(D, seed)
            if variant != "theirs":
                guard_car(env)
            r = C.attempt(lambda: C.run(env, PROGRAM))
            if r["value"] != [["a", "b", "c"]]:
                wrong += 1
                if len(examples) < 4:
                    examples.append({"seed": seed,
                                     "result": C.show(r["value"] or r["error"])})
            if variant == "theirs":
                spread[str(H.closest(C.inner(env, "car")(env["lex"]["a"]), env["lex"]))] += 1
        out["car_of_atom"].append({"D": D, "variant": variant, "wrong": wrong,
                                   "runs": len(SEEDS)})
        for e in examples:
            out["car_of_atom_examples"].append({"D": D, "variant": variant, **e})
        if variant == "theirs":
            out["car_of_a_spread"][str(D)] = dict(spread.most_common())
        print(D, variant, wrong, "/", len(SEEDS), examples[:2], flush=True)

for src, label in [("(CAR (CDR (a b c)))", "Lisp 1.5 gives b"),
                   ("(LABEL f (LAMBDA (x) x))", "LABEL")]:
    res = Counter()
    for seed in range(5):
        env = C.new_env(512, seed)
        r = C.attempt(lambda: C.run(env, src))
        res[C.show(r["value"] if r["error"] is None else r["error"])] += 1
    out["other"].append({"program": src, "note": label, "D": 512, "runs": 5,
                         "results": dict(res)})
    print(src, dict(res), flush=True)

# E4: SimpleCleanUp.memorize grows the matrix by k = 100 rows whenever
# i >= m, but never raises m, so past 100 items every item adds 100 rows.
growth = []
for n_items in (100, 200, 400, 800):
    mem = H.SimpleCleanUp(512)
    for _ in range(n_items):
        mem.memorize(H.HRR(512))
    growth.append({"items stored": n_items, "matrix rows": int(mem.M.shape[0]),
                   "MB at D=512": round(mem.M.nbytes / 1e6, 1)})
    print(growth[-1], flush=True)
out["memory_growth"] = growth
env = C.new_env(512, 0)
C.run(env, H.lisp_fizzbuzz_program)
out["memory_after_fizzbuzz_defines"] = {"items stored": env["mem"].i,
                                        "matrix rows": int(env["mem"].M.shape[0])}
print(out["memory_after_fizzbuzz_defines"], flush=True)

C.write("holis_errata.json", out)
