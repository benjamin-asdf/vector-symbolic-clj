"""Test the suggested fixes for holis's errata 4.2, E2, E3 and E4, noise-free.

Each fix is a runtime patch of a fresh environment (or of one class method);
the notebook stays untouched and none of its code is copied. For each fix
we run
  (a) the erratum's minimal programs, with and without the fix;
  (b) the noise-free small control programs of holis_control.py with the
      fix applied (regression), and their fizzbuzz helpers (D = 512, seeds
      0-4, their DEFINEs and then each helper in one environment, each
      call stopped after CALL_LIMIT seconds). The fizzbuzz runs all use the
      E4 fix, which changes no result and turns minutes into seconds.

E1's fix (car of a non-pointer answers NIL, tested in holis_errata.py) is
rerun here on the fizzbuzz helpers, because their DEFINE relies on car of
a non-pointer: the function memory holds cell bundles, not pointers.

The fixes, as suggested in the notebook's "Errata for the authors":
  4.2  keep the roles lhs, rhs, φ out of the lexicon and the item memory.
       vsa_env draws them as before; we then delete them from env['lex']
       and their rows from mem. The state is the one the suggested
       `reserved` change gives: the same random draws, the roles only in
       env[...].
  E2   CAR, CDR and CONS evaluate their arguments with ev_until_done.
       Variants: as suggested; plus QUOTE returning its argument (the
       notebook's first suggestion for quoted lists); plus reading their
       (QUOTE (x)) wrapper as x (the fix that works).
  E3   remove LABEL from the basic functions; or keep it with an `ind`
       parameter and a one-argument memorize (as suggested; recursion still
       fails); or keep it as a DEFINE that returns the lambda (works).
  E4   SimpleCleanUp.memorize raises its capacity m when it grows.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_fixes.py
"""

import ast
import contextlib
import os
import signal
import time
from collections import Counter

import numpy as np

import holis_common as C

H = C.H
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(5)
DIMS = (512, 2048)
COMMUTE_SEEDS = range(40)          # as holis_commute.py
COMMUTE_DIMS = (512, 2048, 8192)
CALL_LIMIT = 60                    # seconds per fizzbuzz call (with E4 each takes < 3 s)
FIZZ_SEEDS = range(5)
MAX_CELLS = 300                    # budget per small program (their memory grows as cells²)


def read_list(script, name):
    """A literal list assigned in one of our scripts, read without running it."""
    tree = ast.parse(open(os.path.join(HERE, script)).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


SMALL = read_list("holis_control.py", "SMALL")
COMMUTE = read_list("holis_commute.py", "PROGRAMS")


def num(n):
    s = "NIL"
    for _ in range(n):
        s = f"({s})"
    return s


# their fizzbuzz helpers, as in holis_control.py
FIZZ = [
    (f"(countdown {num(3)} {num(1)})", "countdown 3 1", [[["NIL"]]]),
    (f"(divisible {num(3)} {num(3)})", "divisible 3 3", ["T"]),
    (f"(divisible {num(4)} {num(3)})", "divisible 4 3", ["F"]),
    ("(and T F)", "and T F", ["F"]),
    (f"(take {num(2)} (a b c))", "take 2 (a b c)", [["a", "b"]]),
]


# ---------------------------------------------------------------- helpers
def closures(env):
    """Every function reachable from env's callables through closures."""
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


def find(env, name):
    for f in closures(env):
        if f.__name__ == name:
            return f
    raise KeyError(name)


def rebind(env, name, fn):
    """Rebind the free variable `name` in every closure of vsa_env."""
    for f in closures(env):
        for n, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            if n == name:
                cell.cell_contents = fn


# ---------------------------------------------------------------- the fixes
def fix_e1(env):
    """The tested E1 control (holis_errata.py): car(non-pointer) = NIL."""
    orig, assoc, nil = C.inner(env, "car"), env["assoc"], env["NIL"]

    def car(c, ind=""):
        return orig(c) if c in assoc.A else nil
    rebind(env, "car", car)


def fix_e1_cells(env):
    """E1 variant: car answers NIL unless c is a stored pointer or a stored
    cell bundle (what their function memory holds)."""
    orig, assoc, nil = C.inner(env, "car"), env["assoc"], env["NIL"]

    def car(c, ind=""):
        return orig(c) if (c in assoc.A or c in assoc.A.reverse) else nil
    rebind(env, "car", car)


def fix_roles(env):
    """4.2: the roles stay in env[...] but leave env['lex'] and mem."""
    lex, mem = env["lex"], env["mem"]
    roles = [env[r] for r in ("lhs", "rhs", "φ")]
    for r in ("lhs", "rhs", "φ"):
        h = lex.data.pop(r)
        lex.reverse.pop(h, None)
    keep = [i for i in range(mem.i)
            if not any(np.array_equal(mem.M[i], h.v) for h in roles)]
    rows = mem.M.shape[0]
    mem.M = np.concatenate([mem.M[keep], np.zeros((rows - len(keep), mem.n))], axis=0)
    mem.i = len(keep)


def fix_e2(env, quote=None):
    """E2: CAR, CDR, CONS evaluate their argument(s) first.
    quote=None: as suggested. quote="arg": also QUOTE returns its argument
    (the notebook's suggestion). quote="unwrap": QUOTE stays theirs, and a
    value of the form (QUOTE (x)) that ev_until_done returns is read as x."""
    car, cdr = C.inner(env, "car"), C.inner(env, "cdr")
    ccons, evall, Q = find(env, "ccons"), env["evall"], env["lex"]["QUOTE"]

    def val(x, ind=""):
        v = evall(x, ind)
        if quote == "unwrap" and (car(v) == Q) > 0.2:   # their floor
            v = car(car(cdr(v)))                       # their QUOTE wraps the tail
        return v

    def caar(c, ind=""):
        return car(val(car(c), ind))

    def cdar(c, ind=""):
        return cdr(val(car(c), ind))

    def cons(ab, ind=""):
        return ccons(val(car(ab), ind), val(car(cdr(ab)), ind))
    env.update({"CAR": caar, "CDR": cdar, "CONS": cons})
    if quote == "arg":
        env["QUOTE"] = lambda c, ind="": car(c)


def fix_e3_remove(env):
    env["basic"].remove("LABEL")      # the same list object as basic_fs
    del env["LABEL"]


def fix_e3_keep(env):
    car, cdr, mem, ll = C.inner(env, "car"), C.inner(env, "cdr"), env["mem"], find(env, "ll")

    def label(lbl_exp, ind=""):
        lbl, exp = car(lbl_exp), car(cdr(lbl_exp))
        mem.memorize(lbl)
        return ll(lbl, exp, exp)
    env["LABEL"] = label


def fix_e3_define(env):
    """E3 kept, working: LABEL enters the name in the function memory, as
    their DEFINE does, and returns the lambda. Unlike Lisp 1.5 the name is
    then global."""
    car, cdr, define = C.inner(env, "car"), C.inner(env, "cdr"), find(env, "define")

    def label(lbl_exp, ind=""):
        define(lbl_exp, ind)
        return car(cdr(lbl_exp))
    env["LABEL"] = label


_ORIG_MEMORIZE = H.SimpleCleanUp.memorize


def memorize_e4(self, hrr):
    """E4: their memorize, plus raising m when the matrix grows."""
    if self.i >= self.m:
        self.M = np.concatenate([self.M, np.zeros((self.k, self.n))], axis=0)
        self.m += self.k
    self.M[self.i, :] = hrr.v
    self.i = self.i + 1
    return hrr


@contextlib.contextmanager
def class_e4(on):
    H.SimpleCleanUp.memorize = memorize_e4 if on else _ORIG_MEMORIZE
    try:
        yield
    finally:
        H.SimpleCleanUp.memorize = _ORIG_MEMORIZE


# variant name -> list of env patches; "E4" in the name switches the class patch
VARIANTS = {
    "theirs": [],
    "E1": [fix_e1],
    "E1 + E4": [fix_e1],
    "E1 admitting cells": [fix_e1_cells],
    "E1 admitting cells + E4": [fix_e1_cells],
    "4.2": [fix_roles],
    "E2": [fix_e2],
    "E2 + QUOTE returns its argument": [lambda e: fix_e2(e, quote="arg")],
    "E2 + unwrap QUOTE": [lambda e: fix_e2(e, quote="unwrap")],
    "E3 removed": [fix_e3_remove],
    "E3 kept": [fix_e3_keep],
    "E3 as DEFINE": [fix_e3_define],
    "E4": [],
    "E2 + unwrap QUOTE + E4": [lambda e: fix_e2(e, quote="unwrap")],
    "all but E1 (4.2, E2 + unwrap QUOTE, E3 removed, E4)":
        [fix_roles, lambda e: fix_e2(e, quote="unwrap"), fix_e3_remove],
}


def env_for(variant, D, seed, max_cells=MAX_CELLS):
    env = C.new_env(D, seed, max_cells=max_cells)
    for f in VARIANTS[variant]:
        f(env)
    return env


def uses_e4(variant):
    return "E4" in variant


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _alarm)


def tally(src, want, variant, dims=DIMS, seeds=SEEDS, first=False):
    """Run src under variant; count readings equal to want. With first=True
    compare only the first reading (as holis_commute.py does)."""
    rows = []
    with class_e4(uses_e4(variant)):
        for D in dims:
            got, ok = Counter(), 0
            for seed in seeds:
                env = env_for(variant, D, seed)
                r = C.attempt(lambda: C.run(env, src))
                v = r["value"]
                if first and v:
                    v = v[0]
                got[C.show(v if r["error"] is None else r["error"])] += 1
                ok += int(r["error"] is None and v == want)
            rows.append({"program": src, "variant": variant, "D": D,
                         "expected": C.show(want), "correct": ok, "runs": len(seeds),
                         "results": dict(got.most_common())})
            print(rows[-1], flush=True)
    return rows


def regression(variant):
    rows = []
    for src, want in SMALL:
        rows += tally(src, want, variant)
    return {"variant": variant, "programs": len(SMALL), "dims": list(DIMS),
            "correct": sum(r["correct"] for r in rows),
            "runs": sum(r["runs"] for r in rows), "rows": rows}


def fizzbuzz(variant, seed):
    """Their DEFINEs, then each helper, in one env (D = 512)."""
    out = []

    def call(src):
        t = time.time()
        signal.alarm(CALL_LIMIT)
        try:
            r = C.attempt(lambda: C.run(env, src))
        except Timeout:
            r = {"value": None, "error": "Timeout: "}
        signal.alarm(0)
        if r["error"] == "Timeout: ":
            r["error"] = f"stopped after {CALL_LIMIT} s"
        return r, round(time.time() - t, 1)

    with class_e4(uses_e4(variant)):
        env = env_for(variant, 512, seed, max_cells=None)
        r, secs = call(H.lisp_fizzbuzz_program)
        out.append({"variant": variant, "seed": seed, "program": "their fizzbuzz DEFINEs",
                    "result": C.show(r["value"] or r["error"]),
                    "expected": C.show(["NIL"] * 9),
                    "correct": r["value"] == ["NIL"] * 9, "seconds": secs})
        print(out[-1], flush=True)
        for src, label, want in FIZZ:
            r, secs = call(src)
            out.append({"variant": variant, "seed": seed, "program": label, "source": src,
                        "result": C.show(r["value"] or r["error"]),
                        "expected": C.show(want), "correct": r["value"] == want,
                        "seconds": secs})
            print(out[-1], flush=True)
    return out


E2_PROGRAMS = [
    ("(CAR (CDR (a b c)))", ["b"]),
    ("(CDR (CDR (a b c)))", [["c"]]),
    ("(CONS (CAR (a b)) (c d))", [["a", "c", "d"]]),
    ("(CAR (CDR (QUOTE (a b c))))", ["b"]),
    ("(CAR (QUOTE (CDR a)))", ["CDR"]),
    ("(CAR (QUOTE (a b)))", ["a"]),
    ("(CONS (QUOTE a) (QUOTE (b c)))", [["a", "b", "c"]]),
]
E3_PROGRAMS = [
    ("(LABEL f (LAMBDA (x) x))", [["LAMBDA", ["x"], "x"]]),
    ("((LABEL f (LAMBDA (x) x)) a)", ["a"]),
    ("((LABEL f (LAMBDA (x) (COND ((EQ x NIL) . done) (T . (f (CAR x)))))) ((NIL)))",
     ["done"]),
]


def main():
    C.set_noise(0, 0)
    out = {"meta": C.meta("holis_fixes.py", {
        "seeds": len(SEEDS), "dims": list(DIMS), "sigma": 0,
        "commute_seeds": len(COMMUTE_SEEDS), "commute_dims": list(COMMUTE_DIMS),
        "fizzbuzz": {"D": 512, "seeds": list(FIZZ_SEEDS), "call_limit_s": CALL_LIMIT},
        "max_cells": MAX_CELLS, "variants": list(VARIANTS)}),
        "roles": [], "e2": [], "e3": [], "e4_growth": [], "regression": [],
        "fizzbuzz": []}

    # 4.2: the commute programs, as holis_commute.py runs them
    for variant in ("theirs", "4.2"):
        for src, want in COMMUTE:
            out["roles"] += tally(src, want, variant, COMMUTE_DIMS, COMMUTE_SEEDS, first=True)

    for variant in ("theirs", "E2", "E2 + QUOTE returns its argument", "E2 + unwrap QUOTE"):
        for src, want in E2_PROGRAMS:
            out["e2"] += tally(src, want, variant)

    for variant in ("theirs", "E3 removed", "E3 kept", "E3 as DEFINE"):
        for src, want in E3_PROGRAMS:
            out["e3"] += tally(src, want, variant, dims=(512,))

    for on in (False, True):
        with class_e4(on):
            for n_items in (100, 200, 400, 800):
                mem = H.SimpleCleanUp(512)
                t = time.time()
                for _ in range(n_items):
                    mem.memorize(H.HRR(512))
                out["e4_growth"].append({"variant": "E4" if on else "theirs",
                                         "items stored": n_items,
                                         "matrix rows": int(mem.M.shape[0]),
                                         "MB at D=512": round(mem.M.nbytes / 1e6, 1)})
                print(out["e4_growth"][-1], flush=True)

    for variant in ("theirs", "E1", "E1 admitting cells", "4.2", "E2", "E2 + unwrap QUOTE",
                    "E3 removed", "E3 kept", "E3 as DEFINE", "E4",
                    "all but E1 (4.2, E2 + unwrap QUOTE, E3 removed, E4)"):
        reg = regression(variant)
        out["regression"].append(reg)
        print("regression", variant, reg["correct"], "/", reg["runs"], flush=True)
        C.write("holis_fixes.json", out)      # keep partial results

    for variant in ("E4", "E1 + E4", "E1 admitting cells + E4", "E2 + unwrap QUOTE + E4"):
        for seed in FIZZ_SEEDS:
            out["fizzbuzz"] += fizzbuzz(variant, seed)
        C.write("holis_fixes.json", out)

    C.write("holis_fixes.json", out)


if __name__ == "__main__":
    main()
