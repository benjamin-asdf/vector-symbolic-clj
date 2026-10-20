"""4.1 Superposed cells, in holis: the paper's cell versus the notebook's.

The paper (eq. 2) defines cons(a, b) = nu(L*a + R*b + phi): the cell is its
own content. The notebook does not do that: ccons() stores the bundle behind
a fresh random pointer in SimpleAssoc, "because tuples turn out to degrade
rather quickly". So 4.1 does not apply to their code as written. To see the
paper's cell inside their code we add a "paper mode", two instance
attributes on their SimpleAssoc object:

  memorize(probe, trace) -> (nu(trace), nu(trace))   the cell is its bundle
  recall(probe)          -> probe                    no pointer to follow

Everything else (car, cdr, read, the evaluator, the item memory) is theirs.
In paper mode there is no pointer memory, so probe noise only reaches the
item-memory cleanups; in pointer mode it reaches both, and the pointer
memory's fixed theta = 0.2 caps it (see holis_thresholds.py). The read test
therefore also runs "pointer, theta=5/sqrt(D)": their pointer cells with a
chance-relative threshold, which isolates the cell encoding from that cap.
(theta = -1, pure argmax, breaks their atom test even noise-free.)

Measured, D = 2048:
  sibling   cosine between Peano numerals n and n+1 (their numerals:
            0 = NIL, n+1 = (n)), 5 seeds
  read      numerals 1..8 built, then numeral 4 read back with their read(),
            20 seeds per (mode, sigma)
  program   three small programs, noise-free, 10 seeds per mode, and the
            cosine between a form and its first evaluation step, which
            their ev_until_done compares with 0.2 to decide "evaluates to
            itself". (Their countdown takes ~210 s per run at this D in
            pointer mode, too slow for a sweep.)

    OPENBLAS_NUM_THREADS=1 .venv/bin/python holis_cells.py
"""

import numpy as np

import holis_common as C

H = C.H
D = 2048
SIG_READ = (0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16)
READ_SEEDS, PROG_SEEDS = 20, 10


def num(n):
    s = "NIL"
    for _ in range(n):
        s = f"({s})"
    return s


def paper(env):
    a = env["assoc"]
    a.memorize = lambda probe, trace: (trace.unit(), trace.unit())
    a.recall = lambda probe: probe


def make(mode, seed, **kw):
    if mode == "pointer, theta=5/sqrt(D)":
        kw["theta"] = 5.0 / D ** 0.5
    if mode == "pointer, theta=-1":
        kw["theta"] = -1.0
    env = C.new_env(D, seed, **kw)
    if mode == "paper":
        paper(env)
    return env


out = {"meta": C.meta("holis_cells.py", {
    "D": D, "sigma_read": SIG_READ, "read_seeds": READ_SEEDS,
    "program_seeds": PROG_SEEDS, "program_sigma": 0, "max_cells_program": 20000}),
    "sibling": [], "read": [], "program": []}

C.set_noise(0, 0)
for mode in ("pointer", "paper"):
    for seed in range(5):
        env = make(mode, seed)
        v = [env["holo"](H.parse(num(k))[0]) for k in range(1, 8)]
        out["sibling"].append({
            "mode": mode, "seed": seed,
            "cos(n,n+1)_mean": float(np.mean([v[i] == v[i + 1] for i in range(6)])),
            "cos(n,n+2)_mean": float(np.mean([v[i] == v[i + 2] for i in range(5)]))})
        print(out["sibling"][-1], flush=True)

target = H.parse(num(4))[0]
want = C.run(make("pointer", 0), num(4))  # their reader's rendering of 4
for mode in ("pointer", "pointer, theta=5/sqrt(D)", "pointer, theta=-1", "paper"):
    for s in SIG_READ:
        ok = 0
        for seed in range(READ_SEEDS):
            env = make(mode, seed)
            C.set_noise(0, seed)
            vs = [env["holo"](H.parse(num(k))[0]) for k in range(1, 9)]
            C.set_noise(s, seed)
            r = C.attempt(lambda: env["read"](vs[3]))
            ok += int(r["value"] == want[0])
        out["read"].append({"mode": mode, "sigma": s, "correct": ok, "runs": READ_SEEDS})
        print("read", mode, s, ok, "/", READ_SEEDS, flush=True)

PROGRAMS = [("(CAR (a b c))", ["a"]),
            ("(COND ((EQ a b) . yes) (T . no))", ["no"]),
            ("((LAMBDA (x) (CONS x (b c))) a)", [["a", "b", "c"]])]
for mode in ("pointer", "paper"):
    for src, want in PROGRAMS:
        ok, seen, sims = 0, {}, []
        for seed in range(PROG_SEEDS):
            env = make(mode, seed, max_cells=20000)
            C.set_noise(0, seed)
            r = C.attempt(lambda: C.run(env, src))
            v = C.show(r["value"] if r["error"] is None else r["error"])
            seen[v] = seen.get(v, 0) + 1
            ok += int(r["value"] == want)
            # their ev_until_done stops when the new expression e is similar
            # to the old one ee: (ee == e) > 0.2. Measure that cosine for the
            # first evaluation step of the form.
            env2 = make(mode, seed, max_cells=20000)
            form = env2["holo"](H.parse(src)[0])
            step = C.attempt(lambda: env2["ev"](form))
            if step["value"] is not None:
                sims.append(float(form == step["value"]))
        out["program"].append({"mode": mode, "program": src, "expected": C.show(want),
                               "correct": ok, "runs": PROG_SEEDS, "results": seen,
                               "cos(form, ev(form))_mean": float(np.mean(sims)) if sims else None})
        print("program", out["program"][-1], flush=True)

C.write("holis_cells.json", out)
