# Plan

Four workstreams on top of the working interpreter:

- **R**: robustness and capacity curves, codebook vs. Hopfield cleanup
- **T**: task-level curves
- **S**: a self-hosted interpreter (no host `veval` loop)
- **W**: superposition ("many worlds") programming

Plus the notes for the write-up (`docs/WRITEUP-NOTES.md`).

Order and dependencies:

```
P0 harness ──┬── P1 codebook curves ── P2 Hopfield curves ──┐
             └── P3 task suite ──────────────────────────────┤
P4 self-hosting: S1 metacircular → S2 vector CEK machine → S3 tower
P5 superposition: W0 design spike → W1 core → W2 demos → W3 curves
                  (needs P2's soft memory; is strongest on top of S2)
P6 write-up
```

---

## P0: Experiment infrastructure (1–2 days)

Goal: one reproducible command per figure.

1. [x] **Pluggable cleanup memory.** Split `hdc.py`'s `Memory` into an interface
   (`nearest`, `recall`, `intern`, `deref`, `peel`) with interchangeable
   backends. The interpreter must not care which backend it runs on. Backends:
   - `codebook`: the current lookup table, hardmax(M p) M
   - `mhn`: a modern Hopfield network, softmax(β M p) M (P2)
   - `linear`: β = 0, plain M^T M p; the limiting case, needed for W
2. [x] **Noise knobs**, all in the substrate so that no path can bypass them:
   - `op-noise σ`: Gaussian noise added to the output of every
     bind/unbind/bundle (a noisy neural substrate)
   - `probe-noise σ`: noise on every cleanup probe
   - `memory-damage σ`: noise added once to the stored rows of M and V
     (synaptic damage)
   - `lesion f`: zero a fixed random fraction f of dimensions in every vector,
     stored ones included (cell death: the paper's "degrades gracefully"
     claim, tested directly)
   - `degrade` (exists): one-shot noise on an input value
3. [~] **Instrumentation.** Count operations per kind, record the top-1/top-2
   similarity margin of every cleanup, and record the first failing cleanup
   (wrong index) when ground truth is known. Ground truth comes from running
   the same machine with noise off and the same seed.
4. [x] **Runner.** `vsc.experiments` takes an EDN experiment spec (grid of
   dim × backend × noise × seed × task) and writes one CSV row per run.
   Execution goes through a process pool. The memory cap follows CLAUDE.md:
   each worker is a JVM at `-Xmx2g` plus Python ≈ 3 GB, so run 5 workers,
   after checking `free -g`.
5. [x] **Plots.** `scripts/plot.py` (matplotlib, in `.venv`) turns CSVs into
   figures with Wilson 95% intervals over at least 20 seeds. Theory curves are
   overlaid where there is a closed form.

Done when: `clojure -M:jvm:exp specs/e1.edn` → `out/e1.csv` →
`python scripts/plot.py e1` → `out/e1.png`, reproducible from a seed.

---

## P1: Robustness curves, codebook cleanup (2 days)

| exp | question | x-axis | y-axis | series |
|---|---|---|---|---|
| [x] E1 | primitive cleanup | probe noise σ | P(correct) | D ∈ {512…8192}, load \|M\| ∈ {1e2…1e5} |
| [x] E2 | structure retrieval | noise σ | P(exact round-trip) | list length, map n, set n |
| [x] E3 | capacity | n (entries) | P(get correct), P(false +) | D; theory line for 1/√(parts·n) vs 4.5/√D |
| [x] E4 | lesion | % dims zeroed | P(correct) | D, structure type |
| [x] E5 | interference over a run | #traces in M (program length) | first-failure rate | D |
| [x] E6 | cost | D, \|M\| | wall time/op, ops/task | backend |

Theory for E1 (overlay): the probe is signal s plus noise.
P(correct) = P(s + ε₀ > maxᵢ εᵢ) with εᵢ ~ N(0, σ_eff²/D) over |M| − 1
distractors. This is a numeric integral, and it gives the "predicted vs.
measured" figure.

Expected result, to check rather than assume: a sharp threshold in σ that
moves right as √D grows, and a log|M| dependence on load.

---

## P2: Hopfield cleanup (2 days)

The paper's own proposal: the modern Hopfield network (Ramsauer et al. 2020),
M(p) = softmax(β M^T p) M, iterated until the output stops changing.

- **Autoassociative cleanup:** iterate p ← softmax(β Mᵀp) M. The output is a
  blend. Two readout modes:
  - `snap`: argmax row. This equals the codebook as β → ∞.
  - `soft`: return the blend itself, and let errors or superpositions travel
    downstream. `soft` is what W needs.
- **Heteroassociative pointer memory:** V^T softmax(β K p). That is exactly an
  attention head with keys K (pointers) and values V (traces). Write-up point:
  every `first`/`rest` is one attention lookup.
- **Storage:** patterns are stored as rows (dense associative memory form), so
  no gradient training is needed. This avoids the runtime-encoding cost the
  paper worries about. Grossberg-style gated training is out of scope; it
  stays in the discussion only.
- **Sweeps:** β ∈ {1, 4, 16, 64, ∞} × iterations {1, 3, 10}, across E1–E4.
- **Classical Hopfield** (Hebbian, sign activation, ~0.14 N capacity) is an
  optional baseline. It needs bipolar (MAP) vectors, so only E1 is run, as
  the "why not the old one" point.

Key figures:
- accuracy vs. noise for codebook vs. MHN(β)
- capacity vs. D for all three backends
- **β as a collapse knob**, a preview of W: at low β a superposed probe keeps
  several stored items; at high β it collapses to one.

Risk: at low β the blends compound through a recursive program and the run
drifts. That is a result, not a bug. Measure how many steps a soft run
survives.

---

## P3: Task-level curves (1–2 days, parallel with P1)

The benchmark suite (`bench/*.clj`) is small enough for about 20 seeds × the
full grid:

| task | stresses |
|---|---|
| `(fact 5)`, `(fib 8)` | recursion depth, env chains |
| `map`/`filter`/`reduce` over `(range 10)` | list walking, closures |
| assoc/update loop on a map (n = 5…20) | map rebuild, capacity |
| graph reachability (8–20 nodes) | maps of sets, membership |
| insertion sort (Peano compare via `dec`) | long runs, many traces |
| metacircular λ-eval (existing example) | program as data, nested closures |
| tree insert/lookup (BST as nested maps) | deep nesting |

Metric: exact output match. Secondary metrics: steps until the first wrong
cleanup, and which operation kind failed first. Plot one small-multiples
figure of task accuracy vs. noise, per backend and per D.

Compute budget: about 8 tasks × 10 noise levels × 3 D × 3 backends × 20 seeds
≈ 14k runs, at ~1–3 s each ≈ 8 h serial, ≈ 2 h on 5 workers.

---

## P4: Self-hosting (1–2 weeks)

Aim: the evaluator lives in vector memory as data, and the host only runs a
tiny, language-agnostic loop. Three stages, each useful on its own.

### S1: Metacircular evaluator for the whole dialect (2–3 days)

- `resources/vsc/eval.clj`, written in the dialect: `quote if do def fn let
  cond and or`, closures as tagged lists, env as an a-list or VSA map,
  primitives called through the host `apply`.
- Test differentially: every existing test, run through `(vsc-eval form)`,
  must equal the host `veval` result.
- Measure the tower's overhead (ops per step) and its robustness compared
  with one level, using P3 tasks under noise. Does an interpreted level
  amplify noise?

### S2: A vector CEK machine; no host `veval` (1 week, the core piece)

- **State:** one pointer, s ↦ ν(C⊗control + E⊗env + K⊗continuation). The
  continuation is explicit data (a list of frames), so there is no host stack
  and tail calls come for free. That removes the current `-Xss512m` hack and
  lifts the missing-`recur` limitation.
- **Transition rules** live in a rule memory R, keyed by the shape of the
  state: kind(control) ⊗ head-symbol ⊗ frame-type. Each rule's value is a
  short microprogram in a fixed register language (UNBIND, BIND, BUNDLE,
  CLEANUP, INTERN, PUSH, POP, PRIM k). Rules are data vectors and can be
  printed, inspected, or rewritten by programs.
- **The host loop is the only host code:**
  `while s ≠ HALT: rule = R(key(s)); s = run-micro(rule, s)`, about 50 lines
  that know nothing about Lisp. This is the honest meaning of "fully
  vector-symbolic": the ISA is fixed, like a CPU's, and the language exists
  only as vectors in memory. It is ACT-R-like: production rules selected by
  similarity.
- Rule dispatch through cleanup means rules can also be selected **softly**
  (with an MHN), which is the hook for W.
- Differential tests against host `veval` over the whole suite.

### S3: The tower (2 days)

- Run S1's evaluator *on* S2's machine: a Lisp interpreter written in the
  dialect, executed by a vector machine whose rules are vectors.
- Measure the slowdown and robustness per level.
- Stretch: an S1 program that rewrites S2's rule memory, i.e. a program
  changing its own interpreter.

---

## P5: Superposition programming, "many worlds" (1–2 weeks)

The idea: a value may be a weighted superposition of values (worlds). Every
linear operation acts on all worlds at once, and nonlinear steps (cleanup,
branching) are where worlds interact or collapse.

### W0: Design spike (1–2 days). Resolve this first.

**The key tension.** Our pointer layer broke linearity. `cons(a+b, rest)`
gets a fresh random pointer for the superposed trace, not the sum
ptr(a,rest) + ptr(b,rest). Options:

1. **Linear traces** (the paper's original eq. 2, no pointers): linearity
   holds (L⊗(a+b) = L⊗a + L⊗b), but you pay crosstalk, which is exactly the
   failure we fixed with pointers.
2. **Superpositions of pointers:** a superposed value is Σ wᵢ ptrᵢ, and
   structural operations are applied per world, via linearity of the
   *heteroassociative* memory. With a linear memory (β = 0),
   deref(Σ wᵢ pᵢ) = Σ wᵢ traceᵢ, so `first` of a superposition of lists is
   the superposition of their `first`s, with no enumeration. Pointers stay
   robust, and superposition survives wherever the memory is linear.
3. A hybrid: linear memory inside "superposed regions", snapping memory
   elsewhere.

Hypothesis to test: option 2 works, and **β is the collapse knob**. β = 0
keeps all worlds (linear), β → ∞ is measurement (hardmax), and intermediate β
is partial decoherence. That makes P2 and P5 one story.

### W1: Core language (3–4 days)

- `(amb a b c)` and `(superpose {a 0.5, b 0.5})` build Σ wᵢ vᵢ.
- `(worlds x)` returns `{value weight}`, found by explaining away. The
  weights are amplitudes, and the capacity bound says how many worlds can be
  read back.
- `(collapse x)` gives the argmax world; `(sample x)` samples with p ∝ weight.
- Arithmetic lifts exactly: `(+ (amb 1 2) 10)` = B¹⁰⊗(B¹+B²) = `(amb 11 12)`,
  because binding distributes over addition.
- **`if` on a superposed condition evaluates both branches** and weights them
  by sim(test, true) and sim(test, false). This is the paper's ⊕ operator
  (eq. 1) with its middle case, ν(a+b), promoted from "undefined" to the
  semantics.
- `(assert p x)`, or amplitude filtering: reweight each world by the
  truthiness of p in that world, then renormalise. This is classical
  post-selection, not Grover.
- Recursion with superposed arguments: worlds that terminate at different
  depths need the S2 machine, where the whole machine state is superposed
  (below).

### W2: Demos (3 days)

1. **Level-synchronous BFS in one vector per level.** The frontier is a
   superposed set, and next = adjacency(frontier) through a linear
   heteroassociative memory, one lookup per level for all nodes. Compare with
   the P3 reachability task.
2. **Parallel map over worlds:** `(map f (amb xs ys))`.
3. **Constraint search in a product space.** (Σa)⊗(Σb)⊗(Σc) represents
   |A|·|B|·|C| combinations in one vector. Find a satisfying one with a
   resonator network (Frady, Kent et al. 2020). Candidates: graph colouring on
   small graphs, or 4-queens.
4. **Probabilistic reading:** weights as probabilities, making this a small
   probabilistic language. Compare `(worlds ...)` against exact enumeration
   (total-variation distance).
5. **Superposed machine states (on S2):** step Σ wᵢ sᵢ. Rule dispatch through
   a soft memory runs several programs "at once" until they diverge. Measure
   how long coherent joint execution lasts.

### W3: Curves (2 days)

- world-readout accuracy vs. #worlds vs. D (the superposition capacity law)
- exactness of linear lifting vs. op noise
- BFS correctness vs. graph size, D and β
- the β collapse curve: effective #worlds surviving k steps, vs. β

Honest bounds to state up front: the number of worlds that can be read back
is O(D) (the same capacity law as maps). The exponential representational
power exists only in factored (product) form, and extracting an answer then
needs a resonator search, which is iterative and can fail.

---

## P6: Write-up

See `docs/WRITEUP-NOTES.md`. Draft once P1–P3 are in. S and W are either
sections of one paper or a second one (decide after W0).

---

## Open questions

- Is the S2 microcode ISA small enough to count as "not the interpreter"?
  Target: ≤ 10 instructions, ≤ 50 host lines.
- Soft (MHN) rule dispatch in S2: does it degrade gracefully or chaotically?
- W option 2 needs a *linear* heteroassociative memory, whose capacity is
  O(D), not exponential. Is there a middle β that keeps a few worlds and
  still cleans up?
- The GC for M (reachability from the root pointers) is needed for long P3
  runs. Implement it in P0 if E5 shows interference.
