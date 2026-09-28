# vector-symbolic-clj

A Clojure dialect in which **every value is a single high-dimensional vector**:
lists, vectors, maps, sets, keywords, integers, closures, and the environment
the evaluator runs in.

This is Tomkins-Flanagan & Kelly,
[*Hey Pentti, We Did It!: A Fully Vector-Symbolic Lisp*](https://arxiv.org/abs/2510.17889)
(ICCM 2024), redone for Clojure. They built Lisp 1.5 on holographic reduced
representations (HRR). This project builds a Clojure subset on HRR, with
Python/numpy as the hyperdimensional-computing substrate (via libpython-clj).
Fittingly, the paper opens with Carin Meier's Clojure/conj 2023 talk
introducing VSAs to Clojure programmers.

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
clojure -M:jvm:test                         # 123 assertions incl. the bench suite, ~80 s
clojure -M:jvm:bench                        # P3 task suite: timing table (bench/README.md)
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
| `resources/vsc/prelude.clj` | `map`, `filter`, `reduce`, `*`, … written *in the dialect* |
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
