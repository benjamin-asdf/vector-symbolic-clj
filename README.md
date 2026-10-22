# vector-symbolic-clj

A Clojure dialect in which **every value is a single high-dimensional vector**:
lists, vectors, maps, sets, keywords, integers, closures, and the environment
the evaluator runs in.

This is Tomkins-Flanagan & Kelly,
[*Hey Pentti, We Did It!: A Fully Vector-Symbolic Lisp*](https://arxiv.org/abs/2510.17889)
(MathPsych/ICCM 2024), redone for Clojure. They built Lisp 1.5 on holographic reduced
representations (HRR). This project builds a Clojure subset on HRR, with
Python/numpy as the hyperdimensional-computing substrate (via libpython-clj).
Fittingly, the paper opens with Carin Meier's Clojure/conj 2023 talk
introducing VSAs to Clojure programmers.

**Read the notebook:**
[A Vector-Symbolic Clojure](https://benjamin-schwerdtner.de/vector-symbolic-clojure/),
a Clay notebook whose results are computed live by this code. The paper
draft is in [`paper/draft.md`](paper/draft.md).

```
$ clojure -M:jvm:run examples/demo.clj
=> (:name {:name "Ada", :lang :clojure})
"Ada"
=> (= {:a 1, :b 2} {:b 2, :a 1})
true
=> (fib 10)
55
=> (first (degrade (quote (a b c)) 100))     ;; noise as large as the signal
a
```

## Running

```sh
uv venv .venv && uv pip install --python .venv/bin/python numpy   # substrate
clojure -M:jvm:run                          # line REPL
clojure -M:jvm:run examples/demo.clj        # run a file
clojure -M:jvm:test                         # full suite, excluding ^:slow tests (~5 min)
clojure -M:jvm:test-s1                      # S1 differential suite, tagged ^:slow (~25 min)
clojure -M:jvm:bench                        # P3 task suite: timing table (bench/README.md)
clojure -M:jvm:clay render_notebook.clj     # the Clay notebook → _notebook/ (needs Quarto)
```

The `.venv` matters. A distro numpy built against reference BLAS makes every
cleanup about 40× slower. `vsc.hdc` picks up `.venv/bin/python` automatically,
or the interpreter named in `$VSC_PYTHON`.

## Layout

| file | role |
|---|---|
| `resources/vsc/hdc.py` | substrate: HRR algebra, cleanup memories, integer readout |
| `src/vsc/hdc.clj` | thin libpython-clj bridge; vectors stay opaque Python objects |
| `src/vsc/core.clj` | encoding, evaluator, primitives, reader/printer boundary |
| `src/vsc/machine.clj` | S2: a vector CEK machine, rules and microcode as vectors, no host `veval` (`docs/results-S2.md`) |
| `resources/vsc/vsc_machine.py` | the machine's working memory, code/rule/operand memories and datapath |
| `src/vsc/worlds.clj` | W: superposition ("many worlds") programming on the `:proj` memory (`docs/results-W.md`) |
| `examples/worlds/` | W2 demos: BFS one vector per level, parallel map, dice, a Bayes net |
| `resources/vsc/prelude.clj` | `map`, `filter`, `reduce`, `*`, … written *in the dialect* |
| `resources/vsc/eval.clj` | S1: an evaluator for the whole dialect, written in the dialect (`docs/results-S1.md`) |
| `examples/metacircular.clj` | a λ-calculus interpreter written in the dialect |
| `bench/` | P3 task suite, expected values derived in real Clojure (`bench/README.md`) |
| `src/vsc/bench.clj` | suite runner: timing, trace growth, substrate call counts |

## Encoding

L, R, K, C, V, N are unitary role vectors, ⊗ is circular convolution, ⊘
its inverse, ν normalisation, Π a fixed random permutation of the
dimensions, and M the cleanup memory.

| value | vector |
|---|---|
| symbol, keyword, string, `nil`, `true` | random unitary atom |
| `(a . b)` | pointer *p*, with M: *p* ↦ ν(L⊗a + R⊗b) |
| `first` / `rest` | M(L ⊘ M(p)) / M(R ⊘ M(p)), the paper's eqs. 3–4 |
| `[x₀ … xₙ₋₁]` | v = ν(Σᵢ Πⁱ(V⊗xᵢ)), stored in M as itself, M: v ↦ N⊗n; `nth` is M(V ⊘ Π⁻ⁱ v) |
| `{k v …}` | *p* ↦ ν(ν(Σ k⊗Π(v)) + ν(K⊗Σ k) + C⊗n), Π the random permutation |
| `#{x …}` | *p* ↦ ν(ν(Σ x) + C⊗n) |
| `(fn …)` closure | *p* ↦ ν(L⊗form + R⊗env) |
| integer n | Bⁿ, so `+` is ⊗, `-` is ⊘, `inc` is ⊗B |
| environment | a Lisp 1.5 association list of cells |
| globals | memory F, probed with L⊗sym (paper eq. 11) |

The evaluator (`veval`) only inspects expressions through bind, unbind,
bundle, similarity and cleanup. Special forms are recognised by a similarity
lookup of the head symbol. The kind of a value is whatever memory recognises
it. The host touches plain data in only two places: the reader, which encodes
text as vectors, and the printer, which decodes vectors back to text. The
control flow of `veval` is host Clojure, just as it is Python in the authors'
implementation.

**Clojure-specific parts the paper doesn't cover**
- **Maps are native VSA records.** `get` is one unbind plus a cleanup.
  Equality ignores order for free, because superposition commutes:
  `(= {:a 1 :b 2} {:b 2 :a 1})`.
- **Maps and sets enumerate by explaining away.** Clean up the residual,
  subtract the item found, repeat. Each trace stores its size, so `count` is
  O(1) and exactly n items are peeled.
- **Keywords, maps and sets are callable.** `(:k m)`, `(m :k)`, `(#{1 2} 2)`.
- **Closures and `let` replace substitution-based λ.** The paper uses
  substitution with relabelling; lexical environments are the Clojure way.
- **The substrate is exposed in the language:** `bind`, `unbind`, `bundle`,
  `cleanup`, `degrade`, `similarity`.

## Where the paper's scheme needed changes

1. **Plain superposed tuples break.** A cell that is only ν(L⊗a + R⊗b) is
   ~0.9 similar to any cell sharing most of its parts. Recursive calls create
   many such environment cells (`((b . 3) . ((a . 5)))` next to
   `((b . 2) . ((a . 5)))`), and cleanup returned the neighbour: `(fact 5)`
   failed. The authors' own notebook hits the same wall ("tuples turn out to
   degrade rather quickly") and switches to a heteroassociative memory. So
   does this project: every composite is a random pointer that M maps to its
   trace. Equal traces are hash-consed onto one pointer, so structural
   equality is still just vector similarity.
2. **Integers use a residue number system, in the product code.** This
   follows the authors' follow-up, [*…We Did (More of) It!*](https://arxiv.org/abs/2511.08767),
   and Kymn et al. (2025). B's phase at every Fourier bin is a random
   multiple of 2π/P, P = 5·7·11·13·17·19, so every bin mixes all six clocks
   and two integers are unrelated unless they are equal: a superposition of
   integers matches only its members. Readout: every integer the machine
   builds or reads (literals, counts, printed values, results of `inc`,
   `dec`, `+`, `-`) is kept in a small memory, which the plain cleanup
   reads; the superposition cleanup, the printer and the worlds readout
   also search [−1024, 1024]. Arithmetic
   whose operands are not remembered, or whose result fails its check,
   reads the result exactly against all of ±808,307 by one FFT of length P
   (about 60–85 ms).
3. **Integers are boxed inside superpositions.** Integer (and vector) map
   keys, map values and set members are boxed behind hash-consed pointers,
   which are near-orthogonal.
4. **Map entries are directed.** Binding commutes, so the entry 1⊗2 = 2⊗1
   answers the probe 2 with 1 at full strength: `(get {1 2 2 3} 2)`
   returned 1. An entry is k⊗Π(v), with Π the fixed random permutation; a
   cyclic shift would not do, since it commutes with circular convolution.
   Π is on the value because keys are unitary and unbind exactly. The P3
   `map-update` task found this; the first fix boxed every value behind its
   own pointer instead.
5. **Lists and vectors make opposite trade-offs, and the program picks.**
   Pointers (change 1) give up what makes HDC attractive for structured
   data: similar structures as similar vectors. So only lists are pointer
   chains (robust, unbounded, opaque). A vector is a positional
   superposition, stored as itself:
   - vectors that share elements at the same positions are similar;
   - `nth` and `count` are one readout each;
   - a vector holds about D/48 elements (43 at D = 2048);
   - a vector tolerates less noise than a list, and less the longer it is
     (σ₅₀ 5.0 for 4 elements and 2.0 for 16, against 5–6 for any list, at
     D = 2048);
   - a superposition of vectors keeps only each position's marginal.

   Use lists for recursion, code and tuples of correlated worlds, and
   vectors for records and indexed data. The decision, its reasons and its
   measurements are in `docs/DECISION-sequences.md`.

## Superposition programming ("many worlds")

A value can be a weighted superposition of ordinary values, still one
vector. `vsc.worlds` runs the dialect on a threshold + least-squares memory
(`:proj`) that keeps every world at its weight through memory reads:

```clojure
(require '[vsc.worlds :as w])
(w/init! {:dim 4096})
(w/run '(+ (amb 1 2) 10))                        ;; => #worlds {11 0.5, 12 0.5}
(w/run '(let [x (amb 1 2)] (+ x x)))             ;; => #worlds {2 0.25, 3 0.5, 4 0.25}
(w/run '(for-worlds [x (amb 1 2)] (+ x x)))      ;; => #worlds {2 0.5, 4 0.5}
```

Weights are integers going in (relative) and per mille coming out, because
the dialect has no floats.

| primitive | what it does |
|---|---|
| `(amb a b …)` | the uniform superposition of its arguments |
| `(superpose {v w, …})` | Σ (w/Σw)·v; equal worlds merge and their weights add |
| `(worlds x)` | `{world weight‰}`; throws past capacity or when the worlds found leave most of x unexplained |
| `(weight x v)` | the weight of world v in x, in ‰; 0 if absent |
| `(collapse x)` | the heaviest world (a measurement) |
| `(collapse x β)` | a partial collapse, weights ∝ exp(β·wᵢ); distorts the weights |
| `(sample x)` | one world, drawn with probability ∝ weight (seeded) |
| `(assert p x)` | post-selection: keep the worlds where `(p v)` holds, renormalise; throws if none survive |
| `(for-worlds [v x …] body)` | special form: enumerate each binding in turn, so later expressions see the *same* world (call-time choice); returns Σ w·body |
| `(support x)` | the uniform superposition over x's worlds (the set of them) |
| `(without x y)`, `(union x y)` | set difference and union of uniform superpositions |
| `(relation spec)`, `(follow r x)` | a stored relation (key ↦ the set of its values), and one lookup that finds the successors of every world of x at once |

Destructors (`first`, `rest`, `get`), variable lookup, `let`/`fn` and
`+`/`inc` lift linearly: one memory read for all worlds. Other primitives
and constructors enumerate the worlds. `if` on a superposed test runs both
branches π : 1−π and splits the tested variable between them, so recursion
whose depth differs between worlds terminates.
Choice is **run-time** (each occurrence is its own draw, as the second line
shows), because a single vector carries no world identity. `for-worlds` gives
call-time choice by explicit enumeration. Build joint worlds as lists, not
vectors: a vector is linear in its elements, so [1 1] + [2 2] = [1 2] + [2 1],
and a superposition of vectors keeps only each position's marginal.

Capacity is about D/100 worlds per memory step and 5 integer worlds at any D.
Beyond that, `worlds` throws. It is not faster than enumeration. The API,
the semantics, the demos, the curves and the limits are in
`docs/results-W.md`.

## Experiments

The substrate is instrumented for robustness experiments (docs/PLAN.md, P0).
Everything acts inside `hdc.py`, so no interpreter path bypasses it:

| option to `init!` / `set-knobs!` | effect |
|---|---|
| `:memory :codebook` / `:linear` / `:mhn` | cleanup backend: hardmax table, β = 0 linear readout, modern Hopfield (P2, a slot) |
| `:op-noise σ` | v + σ‖v‖/√D·z on every bind/unbind/bundle output |
| `:probe-noise σ` | the same on every cleanup probe |
| `:lesion f` | a fixed fraction f of dimensions is zero in every vector, stored rows included |
| `:memory-damage σ` | noise of norm σ on every stored row, applied by `(vsc/damage!)` |
| `:count-ops? :log-margins?` | per-kind op counters, top1 − top2 margin of every cleanup |

With every knob off the stored rows are bit-identical to the plain substrate.

```sh
clojure -M:jvm:exp specs/e1.edn            # grid → out/e1.csv (2 JVM workers)
.venv/bin/python scripts/plot.py e1         # → out/e1.png, e1-threshold.png
```

Results of P1 (codebook robustness curves): `docs/results-P1.md`, figures in
`figures/`.

## Limits

- **Capacity grows linearly with D.** Measured, and enforced with an error
  rather than silently returning garbage:

  | D | map entries | set members | vector elements |
  |---|---|---|---|
  | 2048 (default) | 21 | 32 | 43 |
  | 4096 | 43 | 64 | 86 |

  These are single-lookup margins. A workload that reads a map hundreds of
  times fails sooner: the `map-update` bench task is exact up to 12 entries at
  D=2048, but not at 16 or 20 (`bench/README.md`).
  Change it with `(vsc/init! {:dim 4096})`. Lists have no capacity limit,
  because every cell is its own pointer: a 200-element list round-trips.
- **Speed.** `(fib 10)` takes about 9 s. The evaluator re-reads the program
  from vector space on every step, and each cleanup is a scan over M.
- **M only grows.** Every environment cell a call creates is stored, and
  nothing is collected.
- **Not supported yet:** macros, `loop`/`recur` (and no TCO), floats,
  `<`/`>`, string functions, destructuring.
