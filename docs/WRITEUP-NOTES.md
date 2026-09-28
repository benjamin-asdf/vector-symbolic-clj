# Write-up notes

Working notes, not prose. Claims marked **[verify]** are not yet backed by
our own measurements or checked citations.

## Title candidates

- *Hey Pentti, Records Too!: A Vector-Symbolic Clojure*
- *Many Worlds in One Vector: Superposition Programming on a Vector-Symbolic Lisp*
  (if W is its own paper)
- *What Breaks When You Run Lisp on Hypervectors*

## One-paragraph pitch

Tomkins-Flanagan & Kelly showed that Lisp 1.5 can be expressed in a VSA. We
build a working Clojure dialect in which every value, including maps, sets,
closures and the environment, is one HRR vector, with the evaluator touching
values only through bind, unbind, bundle, similarity and cleanup. Running
real programs exposed failure modes that the equations hide:
- superposed tuples get confused under recursion
- residue-encoded integers form "chimeras" inside superpositions
- capacity limits are hard

We fix each one, measure robustness and capacity with a lookup-table and a
modern-Hopfield cleanup, and [S] move the interpreter itself into vector
memory. [W] Finally, we show that the same machinery supports superposition
programming, with the Hopfield β acting as a collapse knob.

## Contributions (the list as it stands)

1. A tested vector-symbolic Clojure: lists, vectors, **maps and sets as native
   VSA records**, closures, 103 differential assertions against real Clojure.
2. **Failure analysis of the paper's encoding:**
   - superposed cells are ~0.9 similar when they share parts, so cleanup
     confuses them; `(fact 5)` failed
   - fix: a heteroassociative pointer memory with hash-consing, which keeps
     structural equality as similarity
3. **Residue integers × superposition:** chimera numbers (n and n+p agree on
   p's band). Fix: boxing integers inside superpositions.
4. **Explicit capacity law with enforcement:** maps 21, sets 32 at D = 2048;
   linear in D. Adaptive membership (O(1) for clear hits and misses, exact
   explaining away in the ambiguous band).
5. [P1–P3] Robustness and capacity curves at the primitive, structure and
   task levels, codebook vs. MHN, with theory overlays.
6. [S] A self-hosted interpreter: a CEK machine whose state and rules are
   vectors, driven by a Lisp-agnostic host loop of about 50 lines.
7. [W] Superposition programming: `amb`/`worlds`/`collapse`, both-branch
   `if` as the paper's ⊕ operator, and BFS in one vector per level.

## Outline

1. **Introduction:** Kanerva's "HDC-Lisp", the paper, why Clojure (records
   and sets are native VSA objects; Meier's Clojure/conj talk is the paper's
   own opening).
2. **Background:** HRR, cleanup memories, CCC framing (brief; cite the
   paper).
3. **The encoding:** a table of values → vectors; the pointer layer;
   residue integers; maps and sets with a size field.
4. **What broke:** the three failure stories, each with a minimal
   reproduction and the fix. This is likely the most useful section for VSA
   practitioners.
5. **Robustness and capacity:** E1–E6 and the task curves; codebook vs. MHN.
6. **Self-hosting** [S].
7. **Superposition programming** [W].
8. **Discussion:** cleanup is attention; pointers vs. linearity; limits.
9. **Limitations and future work.**

## Figures (planned)

- F1: architecture: reader → vector machine (M, F, numbers) → printer; what
  the host touches.
- F2: similarity histogram of env cells under the linear encoding vs. with
  pointers (the `fact` failure, visualised).
- F3: residue bands and a chimera example.
- F4: E1 accuracy vs. noise, per D, with the theory overlay.
- F5: capacity vs. D (measured and predicted) for maps and sets.
- F6: task accuracy small multiples, codebook vs. MHN(β).
- F7: lesion curve (the graceful-degradation claim).
- F8 [S]: CEK state layout and the host loop, the whole thing on one page.
- F9 [W]: the β collapse curve; BFS frontier as one vector.

## Anecdotes worth keeping (they carry the argument)

- `(fact 5)` → "unable to resolve symbol: a". The cause was cleanup
  returning a sibling environment cell. The authors' notebook says "tuples
  turn out to degrade rather quickly" and quietly switches to pointers; the
  paper doesn't mention it.
- `(reduce + 0 [1 2 3 4])` returned **15**: silent wrong answers, not
  crashes. That is the argument for capacity enforcement.
- Single-phase integer readout: exact on clean vectors, useless inside a
  superposition. That led to residue bands.
- `(zero? 5005)` would have been true: 5005 = 5·7·11·13 shares four of six
  bands with 0. Caught in review, and now a test.
- Engineering footnote: distro numpy on reference BLAS made cleanup 40×
  slower, and OpenBLAS thread pools burned 20 cores on tiny matrix-vector
  products. Worth one sentence on reproducibility.

## Discussion points

- **Cleanup is attention.** A heteroassociative MHN lookup is
  softmax(β K p)ᵀ V. Every `first`/`rest` is one attention head call. That
  connects to transformer interpretability, where the paper cites Tamkin et
  al. on monosemantic features. [verify the framing against Ramsauer et al.,
  "Hopfield Networks is All You Need"]
- **Pointers vs. linearity** is the central design tension. Pointers buy
  robustness and lose superposition; linear traces keep superposition and
  suffer crosstalk. β interpolates between the two regimes. (This is the
  bridge to W.)
- **"Fully vector-symbolic" needs a definition.** Data only (the paper, and
  us so far) vs. data + control state (S2) vs. data + control + rules (S2
  with the rule memory). We should state which one we claim at each stage.
- **Cartesian closedness is necessary, not sufficient.** Turing-completeness
  holds in principle, but practical computation hinges on memory capacity
  and noise, which is the paper's own conclusion. We quantify it.
- **Biological reading:** the lesion curves test graceful degradation
  directly; the paper asserts it.

## Related work to read and cite [verify all]

- Tomkins-Flanagan & Kelly 2024, *Hey Pentti, We Did It!* (arXiv 2510.17889)
- Tomkins-Flanagan & Kelly, *…We Did (More of) It!: Residue Arithmetic*
  (arXiv 2511.08767)
- Tomkins-Flanagan, Hanley & Kelly, *…We Did It Again!: Differentiable
  vector-symbolic types* (arXiv 2510.16533)
- Kanerva 2014 (the "HDC-Lisp" discussion); Plate 1995 (HRR)
- Kleyko et al. 2022/2023 VSA surveys; Heddes et al. 2023 (torchhd)
- Ramsauer et al. 2020, modern Hopfield networks; Krotov & Hopfield, dense
  associative memory
- Frady, Kent, Olshausen & Sommer 2020, resonator networks (for W product
  spaces)
- Residue hyperdimensional computing: Kymn et al. [verify authors and year]
- Furlong & Eliasmith, fractional power encoding and probability in VSAs
  (for W's probabilistic reading) [verify]
- Smolensky tensor product representations; Legendre, Miyata & Smolensky
  1990
- ACT-R in VSAs: Kelly, Arora, West & Reitter 2020 (for S2's rule memory)
- Carin Meier, Clojure/conj 2023, and any Clojure VSA libraries (search
  before claiming "first vector-symbolic Clojure")

## Claims to be careful with

- Don't claim a speed advantage: we are about 10⁴–10⁵× slower than plain
  Clojure.
- Don't say "exponential worlds" without "factored form + resonator search".
  Readable superpositions are O(D).
- MHN exponential capacity applies to retrieval with well-separated patterns
  and a tuned β. Our runtime-stored, correlated traces may not get it.
  Measure; don't quote.
- The host still parses and prints, and in S2 it runs the microcode loop.
  Say exactly where the boundary is.

## Reproducibility checklist

- seeds in every spec; `init!` takes `:seed`
- `pyproject.toml` pins numpy; record the numpy and BLAS versions in each
  CSV header
- one command per figure (P0)
- tag the commit used for each figure
