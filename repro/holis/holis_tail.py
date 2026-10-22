"""E5 (holis): a list whose head is no function has its tail evaluated as a call.

For a list whose head is no function, `ev` returns
ccons(ev(head), ev(tail)) (evaluating the head too), and `ev_until_done` returns
ccons(head, ev_until_done(cdr(e))). Both evaluate the tail as one
expression, so an element that is a function name or a lambda is applied
to the elements after it: (a CAR (b c)) gives the pair (a . b), and
(g (LAMBDA (x) x)) gives (g LAMBDA (x) x). Noise-free. The paper says to
leave such a list unevaluated; their programs rely on the tail evaluation.

The alternative tested here evaluates the tail element by element. As a runtime patch, without
copying their code: `ev` and `ev_until_done` are rebound, in every closure
of vsa_env, to wrappers that call theirs, except where theirs is about to
evaluate such a tail (recognised from the calling frame); that tail is
evaluated one element at a time instead.

I run the programs with their code and with the alternative (D = 512,
2048, seeds 0-4), the E3 program with LABEL removed, the noise-free control
programs (regression), and their fizzbuzz helpers (with E4, as in
holis_fixes.py), with the alternative alone and with E2. Their `take` builds its list
with (CONS (CAR l) (take ...)); CONS does not evaluate its arguments (E2),
so the recursive call runs only because the tail is evaluated as a call.

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_tail.py
"""

import sys

import holis_common as C
import holis_fixes as F

FLOOR = 0.2                     # their floor


def elementwise(env):
    """The alternative to E5: the tail of a list that is not a call is
    evaluated per element (the head still by their ev)."""
    ev, evud = F.find(env, "ev"), F.find(env, "ev_until_done")
    car, cdr = C.inner(env, "car"), C.inner(env, "cdr")
    ccons, atomic, T = F.find(env, "ccons"), F.find(env, "atomic"), env["T"]

    def each(t, f, ind):
        if (atomic(t) == T) > FLOOR:
            return t
        return ccons(f(car(t), ind), each(cdr(t), f, ind))

    def ev_fixed(e, ind=""):
        up = sys._getframe(1)
        # their ev, last line: ccons(ev(head), ev(tail))
        if up.f_code is ev.__code__ and e is up.f_locals.get("tail"):
            return each(e, ev_fixed, ind)
        return ev(e, ind)

    def ev_until_done_fixed(e, ind=""):
        up = sys._getframe(1)
        # their ev_until_done calls itself on e (keep going) or on cdr(e)
        if up.f_code is evud.__code__ and e is not up.f_locals.get("e"):
            return each(e, ev_until_done_fixed, ind)
        return evud(e, ind)
    F.rebind(env, "ev", ev_fixed)
    F.rebind(env, "ev_until_done", ev_until_done_fixed)


F.VARIANTS.update({
    "elementwise": [elementwise],
    "elementwise + E4": [elementwise],
    "E3 removed + elementwise": [F.fix_e3_remove, elementwise],
    "E2 + unwrap QUOTE + elementwise + E4": [lambda e: F.fix_e2(e, quote="unwrap"), elementwise],
})

PROGRAMS = [
    ("(a CAR (b c))", [["a", "CAR", ["b", "c"]]]),
    ("(g (LAMBDA (x) x))", [["g", ["LAMBDA", ["x"], "x"]]]),
    ("(a (LAMBDA (x) x) b)", [["a", ["LAMBDA", ["x"], "x"], "b"]]),
    ("(a (CAR (b c)) d)", [["a", "b", "d"]]),
    ("(a b c)", [["a", "b", "c"]]),
]
LABEL = ("(LABEL f (LAMBDA (x) x))", [["LABEL", "f", ["LAMBDA", ["x"], "x"]]])


def main():
    C.set_noise(0, 0)
    out = {"meta": C.meta("holis_tail.py", {
        "seeds": len(F.SEEDS), "dims": list(F.DIMS), "sigma": 0,
        "fizzbuzz": {"D": 512, "seeds": list(F.FIZZ_SEEDS), "call_limit_s": F.CALL_LIMIT},
        "max_cells": F.MAX_CELLS}),
        "programs": [], "label": [], "regression": [], "fizzbuzz": []}
    for variant in ("theirs", "elementwise"):
        for src, want in PROGRAMS:
            out["programs"] += F.tally(src, want, variant)
    for variant in ("E3 removed", "E3 removed + elementwise"):
        out["label"] += F.tally(*LABEL, variant, dims=(512,))
    for variant in ("theirs", "elementwise"):
        reg = F.regression(variant)
        out["regression"].append(reg)
        print("regression", variant, reg["correct"], "/", reg["runs"], flush=True)
        C.write("holis_tail.json", out)
    for variant in ("E4", "elementwise + E4", "E2 + unwrap QUOTE + elementwise + E4"):
        for seed in F.FIZZ_SEEDS:
            out["fizzbuzz"] += F.fizzbuzz(variant, seed)
        C.write("holis_tail.json", out)
    C.write("holis_tail.json", out)


if __name__ == "__main__":
    main()
