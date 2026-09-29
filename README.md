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
[A Vector-Symbolic Clojure](https://faster-than-light-memes.xyz/vector-symbolic-clojure/),
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

L, R, K, C are unitary role vectors, ⊗ is circular convolution, ⊘ its
inverse, ν normalisation, and M the cleanup memory.

| value | vector |
|---|---|
| symbol, keyword, string, `nil`, `true` | random unitary atom |
| `(a . b)` | pointer *p*, with M: *p* ↦ ν(L⊗a + R⊗b) |
| `first` / `rest` | M(L ⊘ M(p)) / M(R ⊘ M(p)), the paper's eqs. 3–4 |
| `[a b]` | the same, labelled as a vector cell |
| `{k v …}` | *p* ↦ ν(ν(Σ k⊗⟨v⟩) + ν(K⊗Σ k) + C⊗n), ⟨v⟩ a pointer to v |
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
2. **Integers use a residue number system.** This follows the authors'
   follow-up, [*…We Did (More of) It!*](https://arxiv.org/abs/2511.08767).
   The Fourier bins are split into one band per prime (5, 7, 11, 13, 17, 19).
   On p's band, B's coefficients are p-th roots of unity. Cleanup is
   algorithmic: find each residue by matching its band against p candidates,
   then apply the CRT. That is 72 candidate checks instead of a codebook scan.
   It still works inside a superposition, where reading a single phase fails,
   and it covers ±808,307.
3. **Integers are boxed inside superpositions.** Under the residue encoding, n
   and n+p agree on p's band, so explaining away over a set of integers
   latched onto "chimera" numbers. Map keys, map values and set members are
   therefore boxed behind hash-consed pointers, which are near-orthogonal.
4. **Map values get their own pointer.** Binding commutes, so if a value is
   also a key, the entry k′⊗k answers the probe k with k′ at full strength:
   `(get {1 2 2 3} 2)` returned 1. Each value is stored as ⟨v⟩, a
   hash-consed pointer to v that no key ever equals. The P3 `map-update`
   task found this.

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

Primitives: `amb`, `superpose`, `worlds`, `weight`, `collapse`, `sample`,
`assert` (post-selection), `support`/`without`/`union`, `relation`/`follow`,
and the special form `for-worlds`. Destructors, `let`/`fn` and `+`/`inc` lift
linearly. Other primitives and constructors enumerate the worlds. `if` on a
superposed test runs both branches π : 1−π and splits the tested variable
between them, so recursion whose depth differs between worlds terminates.
Choice is **run-time** (each occurrence is its own draw, as the second line
shows), because a single vector carries no world identity. `for-worlds` gives
call-time choice by explicit enumeration.

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

  | D | map entries | set members |
  |---|---|---|
  | 2048 (default) | 21 | 32 |
  | 4096 | 43 | 64 |

  These are single-lookup margins. A workload that reads a map hundreds of
  times fails sooner: the `map-update` bench task is exact up to 12 entries at
  D=2048, but not at 16 or 20 (`bench/README.md`).
  Change it with `(vsc/init! {:dim 4096})`. Lists have no capacity limit,
  because every cell is its own pointer: a 200-element vector round-trips.
- **Speed.** `(fib 10)` takes about 9 s. The evaluator re-reads the program
  from vector space on every step, and each cleanup is a scan over M.
- **M only grows.** Every environment cell a call creates is stored, and
  nothing is collected.
- **Not supported yet:** macros, `loop`/`recur` (and no TCO), floats,
  `<`/`>`, string functions, destructuring. `rest` of a vector returns a
  vector tail.
