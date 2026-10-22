"""Test the suggested fixes for residuelisp's errata R1-R5, noise-free.

Each fix is a runtime patch of one function in their modules (their files
stay untouched). For each fix I run
  (a) the erratum's minimal programs, with and without the fix;
  (b) the noise-free control programs of residuelisp_control.py with the
      fix applied (regression);
and, for R1, (c) their arithmetic test pattern: evaluate (+ 5 5 ), then
compare the result with several integers, through their equals (theirs and
fixed) and through the decoded value.

The fixes, as suggested in the notebook's "Errata for the authors":
  R1  equals: strip the shared __int tag from two integers before the atom
      comparison.
  R2  _and_: the second clause tests rhs instead of lhs.
  R3  if_: evaluate the condition before testing it.
  R4  decode_rhc: search 0..1154 (the product of the moduli), not 0..199.
  R5  _digit: peek at the next character instead of consuming it.

D = 1000, 5 seeds, as in residuelisp_errata.py.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python residuelisp_fixes.py
"""

import ast
import contextlib
import os
import sys
from copy import deepcopy

import numpy as np

import residuelisp_common as C

I, E, V, S = C.I, C.E, C.V, C.S
P = sys.modules["syntax.parser"]
D, SEEDS = 1000, range(5)
MAX_STEPS = 5000   # cleanups + derefs per run, as in residuelisp_thresholds.py
HERE = os.path.dirname(os.path.abspath(__file__))


def control_programs():
    """PROGRAMS from residuelisp_control.py, read without running it."""
    tree = ast.parse(open(os.path.join(HERE, "residuelisp_control.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "PROGRAMS":
            return ast.literal_eval(node.value)
    raise KeyError("PROGRAMS")


CONTROL = control_programs()

# ---------------------------------------------------------------- the fixes
_ORIG = {"equals": I.equals, "_and_": I._and_, "if_": I.if_,
         "decode_rhc": I.decode_rhc, "_digit": P._digit}


def equals_r1(lhs, rhs, enc_env, eval_env):
    """R1: both atomic and both integers -> compare without the tag, then
    hand over to their equals (which recurses through I.equals)."""
    t = enc_env.codebook["#t"]
    if (I.is_approx_eq(I.check_atomic(lhs, enc_env), t, enc_env)
            and I.is_approx_eq(I.check_atomic(rhs, enc_env), t, enc_env)
            and I.is_true(I.check_int(lhs, enc_env), enc_env)
            and I.is_true(I.check_int(rhs, enc_env), enc_env)):
        tag = enc_env.codebook["__int"]
        lhs, rhs = lhs - tag, rhs - tag
    return _ORIG["equals"](lhs, rhs, enc_env, eval_env)


def and_r2(lhs, rhs, enc_env):
    """R2: their _and_ with the second clause on rhs."""
    return not (I.is_nil(lhs, enc_env) or I.is_false(lhs, enc_env)) and not (
        I.is_nil(rhs, enc_env) or I.is_false(rhs, enc_env))


def if_r3(rand, enc_env, eval_env):
    """R3: their if_ with the condition evaluated."""
    condition = I.evaluate(I.car(rand, enc_env, eval_env), enc_env, eval_env)
    branches = I.cdr(rand, enc_env, eval_env)
    if I.is_nil(branches, enc_env):
        raise I.InterpreterError("ERROR: Conditional expression requires "
                                 "`CONSEQUENT` and `ALTERNATE`")
    consequent = I.car(branches, enc_env, eval_env)
    alternate = I.car(I.cdr(branches, enc_env, eval_env), enc_env, eval_env)
    if I.is_nil(condition, enc_env) or I.is_false(condition, enc_env):
        return I.evaluate(alternate, enc_env, eval_env)
    return I.evaluate(consequent, enc_env, eval_env)


def decode_rhc_r4(value, enc_env, floor=0.2):
    """R4: their decode_rhc with the search over 0..prod(moduli)-1. The codes
    are built from the encoder's roots RHC.z_ms, as RHC.encode builds them.
    Calling RHC.encode 1155 times gives the same answers, but each call
    constructs an RHC object whose __init__ recomputes the roots: about 20 s
    per decode at D = 1000, against well under 1 s here."""
    codebook = deepcopy(V.RHC.codebook)
    for i in range(int(np.prod(V.RHC.moduli))):
        if i not in codebook:
            codebook[i] = np.prod([z_m ** i for z_m in V.RHC.z_ms], axis=0)
    keys, codes = zip(*codebook.items())
    sims = [V.RHC.similarity(value.data, code) for code in codes]
    return str(keys[int(np.argmax(sims))])


def digit_r5(d, conts):
    """R5: their _digit, peeking before it consumes."""
    ds = [d]
    while (curr := P._peek(conts)) is not None and curr.isdigit():
        ds.append(P._next(conts))
    return P.Token("".join(ds), P.TokenKind.Int)


FIXES = {
    "R1": [(I, "equals", equals_r1)],
    "R2": [(I, "_and_", and_r2)],
    "R3": [(I, "if_", if_r3)],
    "R4": [(I, "decode_rhc", decode_rhc_r4)],
    "R5": [(P, "_digit", digit_r5)],
}
FIXES["all"] = [p for k in ("R1", "R2", "R3", "R4", "R5") for p in FIXES[k]]


@contextlib.contextmanager
def patched(name):
    """Apply fix `name` (None = theirs) and undo it afterwards."""
    patches = FIXES.get(name, [])
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    for m, a, f in patches:
        setattr(m, a, f)
    try:
        yield
    finally:
        for m, a, f in saved:
            setattr(m, a, f)


# ---------------------------------------------------------------- running
def tally(src, want, fix):
    """Run src over SEEDS with fix applied. `want` is a value, a set of
    acceptable values, or None (no expectation)."""
    ok_set = want if isinstance(want, (set, frozenset)) else {C.show(want)}
    seen, ok = {}, 0
    with patched(fix):
        for seed in SEEDS:
            r = C.attempt(lambda: C.evaluate(src, dim=D, seed=seed, max_steps=MAX_STEPS)[0])
            v = C.show(r["value"] if r["error"] is None else r["error"])
            seen[v] = seen.get(v, 0) + 1
            ok += int(r["error"] is None and v in ok_set)
    return {"program": src, "expected": " or ".join(sorted(ok_set)), "fix": fix or "theirs",
            "correct": ok, "runs": len(SEEDS), "results": seen}


# erratum -> (program, expected) ; expected may be a set of acceptable values
ERRATA = {
    "R1": [("(eq? 3 5 )", "#f"), ("(eq? 3 1000 )", "#f"), ("(eq? 3 3 )", "#t"),
           ("(eq? (+ 1 2 ) 3 )", "#t"), ("(eq? (+ 1 2 ) 4 )", "#f"),
           ("(eq? a 3 )", "#f")],
    "R2": [("(and #t #f)", "#f"), ("(and #t nil)", {"#f", "nil"}),
           ("(and #t #t)", "#t"), ("(and #f #t)", "#f"),
           ("(eq? (cons a (cons b nil)) (cons a (cons c nil)))", "#f"),
           ("(eq? (cons a (cons b nil)) (cons a (cons b nil)))", "#t")],
    "R3": [("(if (eq? a b) yes no)", "no"), ("(if (eq? a a) yes no)", "yes"),
           ("(if (and #f #f) yes no)", "no"), ("(if (and #t #t) yes no)", "yes"),
           ("(if #f yes no)", "no"), ("(if #t yes no)", "yes")],
    "R4": [("(+ 150 100 )", "250"), ("(+ 199 1 )", "200"),
           ("(- 3 5 )", "1153"), ("(+ 1000 154 )", "1154"),
           ("(+ 100 99 )", "199")],
    "R5": [("(+ (+ 1 2) 3)", "6"), ("(+ 1 2)", "3"), ("(car (cons 1 2))", "1"),
           ("(+ (+ 1 2 ) 3 )", "6")],
}
# programs that need two fixes together; run only under "all"
COMBINED = [("(if (eq? 3 5) yes no)", "no"), ("(if (eq? (+ 2 3) 5) yes no)", "yes"),
            ("(eq? (cons 1 (cons 2 nil)) (cons 1 (cons 3 nil)))", "#f"),
            ("(eq? (cons 1 (cons 2 nil)) (cons 1 (cons 2 nil)))", "#t"),
            ("(+ (+ 150 100) 1)", "251")]


def oracle(seed, fix):
    """Their test pattern (test_rhc_add): evaluate, encode the expected
    integer in the same env, compare with equals; and the decoded value."""
    with patched(fix):
        C.reset(seed)
        enc = C.encode_env(D)
        ev = I.EvalEnvironment(E.AssociativeMemory(enc.vsa, enc.dim), None)
        I.decode = lambda *a, **k: "<log>"
        C.STATE["steps"], C.STATE["max_steps"] = 0, MAX_STEPS
        try:
            value = I.evaluate(E.encode(C.parse("(+ 5 5 )"), enc), enc, ev)
            eq_res = {}
            for n in (10, 7, 0, 999):
                want = E.encode(C.parse(str(n)), enc)
                eq_res[str(n)] = I.closest(I.equals(value, want, enc, ev), enc)
        finally:
            I.decode = C.REAL_DECODE
            C.STATE["max_steps"] = None
        decoded = C.REAL_DECODE(value, enc, ev)
    return {"seed": seed, "fix": fix or "theirs", "program": "(+ 5 5 )",
            "equals(result, encode(n))": eq_res,
            "decode(result)": decoded,
            "decode(result) == n": {str(n): decoded == str(n) for n in (10, 7, 0, 999)}}


def their_tests(fix):
    """Their test_rhc_add / test_rhc_sub sources, verbatim, with the right
    and a wrong expected value, judged by equals and by decode."""
    rows = []
    for src, right, wrong in [("(+ 1 2)", "3", "4"), ("(- 5 2)", "3", "4")]:
        for seed in SEEDS:
            with patched(fix):
                C.reset(seed)
                enc = C.encode_env(D)
                ev = I.EvalEnvironment(E.AssociativeMemory(enc.vsa, enc.dim), None)
                I.decode = lambda *a, **k: "<log>"
                C.STATE["steps"], C.STATE["max_steps"] = 0, MAX_STEPS
                try:
                    r = C.attempt(lambda: I.evaluate(E.encode(C.parse(src), enc), enc, ev))
                    res = {}
                    if r["error"] is None:
                        for n in (right, wrong):
                            want = E.encode(C.parse(n), enc)
                            res["equals " + n] = I.is_true(
                                I.equals(r["value"], want, enc, ev), enc)
                finally:
                    I.decode = C.REAL_DECODE
                    C.STATE["max_steps"] = None
                if r["error"] is None:
                    res["decode"] = C.REAL_DECODE(r["value"], enc, ev)
                else:
                    res["error"] = r["error"]
            rows.append({"source": src, "seed": seed, "fix": fix or "theirs",
                         "right": right, "wrong": wrong, **res})
    return rows


def main():
    out = {"meta": C.meta("residuelisp_fixes.py",
                          {"D": D, "seeds": list(SEEDS), "sigma": 0, "max_steps": MAX_STEPS,
                           "fixes": list(FIXES)}),
           "errata": [], "regression": [], "regression_unspaced": [],
           "test_oracle": [], "their_tests": [], "lexer": []}

    for name, progs in ERRATA.items():
        for src, want in progs:
            for fix in (None, name):
                row = {"erratum": name, **tally(src, want, fix)}
                out["errata"].append(row)
                print(row, flush=True)
    for src, want in [p for ps in ERRATA.values() for p in ps] + COMBINED:
        for fix in ((None, "all") if (src, want) in COMBINED else ("all",)):
            row = {"erratum": "combined" if (src, want) in COMBINED else "all fixes",
                   **tally(src, want, fix)}
            out["errata"].append(row)
            print(row, flush=True)

    for fix in (None, "R1", "R2", "R3", "R4", "R5", "all"):
        ok = 0
        rows = []
        for src, want in CONTROL:
            r = tally(src, want, fix)
            rows.append(r)
            ok += int(r["correct"] == r["runs"])
        out["regression"].append({"fix": fix or "theirs", "programs": len(CONTROL),
                                  "all_seeds_correct": ok,
                                  "correct": sum(r["correct"] for r in rows),
                                  "runs": sum(r["runs"] for r in rows),
                                  "rows": rows})
        print("regression", fix, ok, "/", len(CONTROL), flush=True)

    # R5: the control programs without the space before ')'
    for fix in (None, "R5"):
        rows = [tally(src.replace(" )", ")"), want, fix) for src, want in CONTROL]
        out["regression_unspaced"].append({
            "fix": fix or "theirs", "programs": len(CONTROL),
            "all_seeds_correct": sum(int(r["correct"] == r["runs"]) for r in rows),
            "correct": sum(r["correct"] for r in rows),
            "runs": sum(r["runs"] for r in rows), "rows": rows})
        print("unspaced", fix, out["regression_unspaced"][-1]["all_seeds_correct"], flush=True)

    for fix in (None, "R1"):
        for seed in SEEDS:
            out["test_oracle"].append(oracle(seed, fix))
            print(out["test_oracle"][-1], flush=True)
    for fix in (None, "R1", "all"):
        out["their_tests"] += their_tests(fix)
    for r in out["their_tests"]:
        print(r, flush=True)

    for fix in (None, "R5"):
        with patched(fix):
            for src in ["(g 1)", "(+ (+ 1 2) 3)"]:
                out["lexer"].append({"fix": fix or "theirs", "source": src,
                                     "tokens": " ".join(t.cont for t in S.lex(src))})
                print(out["lexer"][-1], flush=True)

    C.write("residuelisp_fixes.json", out)


if __name__ == "__main__":
    main()
