"""Errata in residuelisp, all noise-free, D = 1000, 5 seeds.

R1  Every integer carries the same tag: int(x) = zeta(x) + __int (their
    encode_rhc_integer, eq. 11 of the paper). eq? on atoms is
    |sim(x, y)| > 0.2 with their unnormalised FHRR similarity, so any two
    integers are eq?: the tag alone contributes sim(__int, __int) = 1.
    This is my trap 4 (a shared field makes unrelated values similar).
    Their test oracle for arithmetic, equals(result, encode(expected)),
    inherits it: I evaluate (+ 5 5) and compare it with 7.
R2  _and_ tests its left argument twice and never the right one:
    (and #t #f) -> #t. Structural eq? on lists combines car and cdr with
    _and_, so it compares only the cars.
R3  if does not evaluate its condition: the condition slot is the
    unevaluated form, a pointer, which is neither nil nor #f, so any
    compound condition takes the consequent.
R4  decode_rhc searches the integers 0..199 (plus literals seen), while the
    moduli [3, 5, 7, 11] represent 0..1154; results outside the search come
    back as some other number. Subtraction below zero wraps mod 1155.
R5  The lexer drops a ')' that directly follows an integer, so
    (+ (+ 1 2) 3) parses as (+ (+ 1 2 3)).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python residuelisp_errata.py
"""

import numpy as np

import residuelisp_common as C

I, E = C.I, C.E
D, SEEDS = 1000, range(5)


def tally(src, want=None, extra=None):
    seen, ok = {}, 0
    for seed in SEEDS:
        r = C.attempt(lambda: C.evaluate(src, dim=D, seed=seed)[0])
        v = C.show(r["value"] if r["error"] is None else r["error"])
        seen[v] = seen.get(v, 0) + 1
        ok += int(want is not None and r["value"] == want)
    row = {"program": src, "expected": C.show(want) if want is not None else None,
           "correct": ok, "runs": len(SEEDS), "results": seen}
    if extra:
        row.update(extra)
    print(row, flush=True)
    return row


out = {"meta": C.meta("residuelisp_errata.py", {"D": D, "seeds": list(SEEDS), "sigma": 0}),
       "programs": [], "tag_similarity": [], "test_oracle": [], "lexer": []}

# R1: shared tag
for src, want, tag in [("(eq? 3 5 )", "#f", "R1"), ("(eq? 3 1000 )", "#f", "R1"),
                       ("(eq? 3 3 )", "#t", "R1 control"),
                       ("(and #t #f)", "#f", "R2"), ("(and #t nil)", "nil or #f", "R2"),
                       ("(eq? (cons a (cons b nil)) (cons a (cons c nil)))", "#f", "R2"),
                       ("(eq? (cons a (cons b nil)) (cons a (cons b nil)))", "#t", "R2 control"),
                       ("(if (eq? a b) yes no)", "no", "R3"),
                       ("(if (and #f #f) yes no)", "no", "R3"),
                       ("(if #f yes no)", "no", "R3 control"),
                       ("(+ 150 100 )", "250", "R4"),
                       ("(+ 199 1 )", "200", "R4"),
                       ("(- 3 5 )", "-2 (1153 mod 1155)", "R4"),
                       ("(+ 100 99 )", "199", "R4 control")]:
    out["programs"].append(tally(src, want if not want.startswith(("-2", "nil or")) else None,
                                 {"erratum": tag, "expected": want}))

for seed in SEEDS:
    C.reset(seed)
    enc = C.encode_env(D)
    vs = {n: E.encode(C.parse(str(n)), enc) for n in (3, 5, 1000)}
    a = enc.codebook["a"] if "a" in enc.codebook else E.encode(C.parse("a"), enc)
    b = E.encode(C.parse("b"), enc)

    def sim(x, y):
        return float(enc.vsa.similarity(x.data, y.data))

    def cos(x, y):
        return float(np.real(np.vdot(x.data, y.data)) /
                     (np.linalg.norm(x.data) * np.linalg.norm(y.data)))
    out["tag_similarity"].append({
        "seed": seed,
        "sim(3,5)": sim(vs[3], vs[5]), "sim(3,1000)": sim(vs[3], vs[1000]),
        "sim(3,3)": sim(vs[3], vs[3]), "sim(a,b)": sim(a, b),
        "cos(3,5)": cos(vs[3], vs[5]), "cos(3,1000)": cos(vs[3], vs[1000])})

    # their test pattern (src/tests/test_interpret.py, test_interpret_rhc_add):
    # evaluate, encode the expected integer in the same env, compare with equals
    ev = I.EvalEnvironment(E.AssociativeMemory(enc.vsa, enc.dim), None)
    I.decode = lambda *a, **k: "<log>"
    try:
        value = I.evaluate(E.encode(C.parse("(+ 5 5 )"), enc), enc, ev)
        res = {}
        for n in (10, 7, 0, 999):
            want = E.encode(C.parse(str(n)), enc)
            r = I.equals(value, want, enc, ev)
            res[str(n)] = I.closest(r, enc)
    finally:
        I.decode = C.REAL_DECODE
    out["test_oracle"].append({"seed": seed, "program": "(+ 5 5 )",
                               "equals(result, encode(n))": res})
    print(out["tag_similarity"][-1], out["test_oracle"][-1], flush=True)

for src in ["(g 1)", "(g 1 )", "(+ (+ 1 2) 3)", "(+ (+ 1 2 ) 3 )"]:
    toks = [t.cont for t in C.S.lex(src)]
    row = {"source": src, "tokens": " ".join(toks)}
    if src.startswith("(+"):
        r = C.attempt(lambda: C.evaluate(src, dim=D, seed=0)[0])
        row["result_seed0"] = C.show(r["value"] if r["error"] is None else r["error"])
    out["lexer"].append(row)
    print(row, flush=True)

C.write("residuelisp_errata.json", out)
