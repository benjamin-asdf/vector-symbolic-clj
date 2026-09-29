# Hey Pentti, Records Too! A Vector-Symbolic Clojure, and What Breaks When You Run It

*Draft v0.2, 2026-09-29. Numbers are from `docs/results-*.md` and
`out/paper-f*.json`; figure paths are relative to the repo root.*

## Abstract

Tomkins-Flanagan and Kelly (2024) expressed the elementary functions of
Lisp 1.5 in a vector-symbolic architecture (VSA). We build a working
language on the idea: a Clojure dialect in which every value is one
holographic reduced representation (HRR) vector. That covers lists, maps,
sets, integers, closures and the environment. It passes 371 differential
assertions against real Clojure. Running programs exposed five failure
modes that the algebra hides, and we fix each one.

- Cleanup follows its textbook model, and capacity is linear in the
  dimension D.
- Stored structures survive 60–80% of dimensions lesioned; computation
  survives about 4%.
- A modern Hopfield network, proposed as the natural cleanup memory, never
  beats a lookup table.
- We move the interpreter into vector memory: a CEK machine whose state,
  continuation and 35 transition rules are vectors, driven by 47 lines of
  Lisp-free host code, with proper tail calls.

**Superposition programming.** A value may be a weighted sum of *worlds*,
and the program computes on all of them at once. Destructors act on every
world in one memory read, integer arithmetic lifts exactly, and function
bodies run once for all worlds. Probabilistic inference with such vectors is
known; what is new is a general-purpose language whose values may be
superpositions. It works only with a memory outside the softmax family
(thresholded projection). The native semantics are Hussmann's run-time
choice, and capacity is about D/100 worlds.

## 1. Introduction

Kanerva (2014) sketched a hypervector Lisp. Tomkins-Flanagan and Kelly
(2024; TK24) gave VSA definitions of CONS, CAR, CDR, EQ, ATOM, LAMBDA and
COND, and argued Turing-completeness via Cartesian closure. An existence
proof leaves open what happens when programs run. This paper is about that
gap.

**Why Clojure.** Its core types, persistent maps and sets, are what VSAs
represent best. A map is a record, Σ kᵢ⊗vᵢ, read with one unbind and one
cleanup, and its equality ignores key order because superposition
commutes. Lisp 1.5 has only cons cells, which is the structure VSAs handle
worst. Meier (2023) introduced VSAs to Clojure programmers, and her
`vsa-clj` library implements VSA operations in Clojure. To our knowledge,
no prior work besides TK24 and its follow-ups (Hanley et al., 2025;
Tomkins-Flanagan et al., 2025) runs a Lisp *evaluator*
inside a VSA, and none does so for Clojure.

**Contributions.**
1. A working vector-symbolic Clojure (§3).
2. Five failure modes of the straightforward encoding, each with cause and
   fix (§4).
3. Robustness and capacity measured at three levels (primitive cleanup,
   stored structures, whole programs) under three kinds of noise (§5).
4. A negative result: Hopfield cleanup never wins, and β is not a collapse
   knob (§6).
5. The interpreter in vector memory: a metacircular tower and a vector CEK
   machine (§7).
6. **Superposition programming**: semantics, a working memory design,
   demos, capacity and limits (§8).

The substrate is Python/numpy, reached through libpython-clj. Clojure only
holds opaque references to vectors.

## 2. Background

**VSAs** are surveyed by Kleyko et al. (2022, 2023). **HRR** (Plate, 1995):
- Space: ℝᴰ.
- Binding ⊗ is circular convolution; unbinding ⊘ uses the involution and is
  exact for unitary vectors.
- Bundling is a sum followed by normalisation ν; similarity is the cosine.
- Random vectors have cosines of order N(0, 1/D).

**Cleanup** maps a noisy vector to the nearest stored item. The simplest
form is a lookup table, M(p) = hardmax(M p)ᵀM. TK24 stress that the cleanup
memory is part of the architecture, and suggest a modern Hopfield network
(Ramsauer et al., 2020).

**The Lisp VSA of TK24**:
- a cell is ν(L⊗a + R⊗b + φ);
- CAR and CDR unbind a role, then clean up;
- EQ is similarity;
- conditionals use a saturating operator ⊕ (TK24, eq. 1).

Their reference notebook, unlike the paper, stores tuples behind pointers,
because "tuples turn out to degrade rather quickly". §4.1 quantifies why.

## 3. A vector-symbolic Clojure

| value | vector |
|---|---|
| symbol, keyword, string, `nil`, booleans | random unitary atom, stored in M with its kind as a label |
| `(a . b)`, `[a b]` | pointer p, with M: p ↦ ν(L⊗a + R⊗b) |
| `first` / `rest` | M(L ⊘ M(p)) / M(R ⊘ M(p)) |
| `{k v …}` | p ↦ ν(ν(Σ k⊗⟨v⟩) + ν(K⊗Σ k) + C⊗n) |
| `#{x …}` | p ↦ ν(ν(Σ x) + C⊗n) |
| closure | p ↦ ν(L⊗form + R⊗env) |
| integer n | Bⁿ⊗Z (residue number system, §4.3) |
| environment / globals | an association list of cells / a memory F probed with L⊗sym |

*Table 1. The encoding. L, R, K, C are unitary roles, and ⟨v⟩ is a
hash-consed pointer to v.*

- **Pointers.** Every composite is a random pointer that M associates with
  its trace. Equal traces are hash-consed onto one pointer, so structural
  equality is one similarity test.
- **Arithmetic** is binding: `+` is ⊗, and `inc` binds with B.
- **Kinds.** A value's kind is the label of the memory row it cleans up to.
- **Maps and sets** store their size (C⊗n), so `count` is O(1), and
  enumeration by explaining away peels exactly n items.
- **The boundary** (`figures/paper-f1-architecture.png`). The host reads
  text into vectors and prints vectors back. The evaluator between them
  touches values only through VSA operations and cleanup.
- **Tests.**
  - 46 tests with 371 assertions compare each form's result against real
    Clojure; a slow set adds 85 assertions.
  - A 12-program benchmark suite takes its expected values from
    independent real-Clojure programs, and all 12 are exact at zero noise.

## 4. What breaks

**4.1 Superposed cells are fragile** (`figures/paper-f2-cell-confusion.png`).
- **Symptom.** In our first implementation, which used TK24's cells, `(fact
  5)` failed with "unable to resolve symbol: a", and `(reduce + 0 [1 2 3
  4])` returned 15.
- **Cause.** Recursion creates sibling chains such as ((a . 5) . ()) and
  ((a . 4) . ()). As superpositions they have cosine 0.89 to each other. A
  `cdr` probe's margin over the best sibling is then 0.05, against 0.66 for
  pointer cells. Noise-free, a single lookup still succeeds, even with 64
  siblings. But the thin margin halves the noise tolerance: 50% failure at
  probe noise σ ≈ 4 against ≈ 10 for pointer cells, over thousands of
  chained lookups.
- **Caveat.** We did not isolate which lookup failed in the first version
  before replacing it.
- **Fix.** Pointers with hash-consing.
- **Lesson.** Similarity of content is not identity.

**4.2 Binding commutes.**
- **Symptom.** `(get {1 2 2 3} 2)` returned 1.
- **Cause.** The entry 1 ↦ 2 contributes 1⊗2 = 2⊗1, so probing the key 2
  recovers 1 at full strength.
- **Fix.** Store each value as its own pointer ⟨v⟩, which no key equals.

**4.3 Residue integers form chimeras** (`figures/paper-f3-chimeras.png`).
- **The encoding.** Scanning the whole integer range on every cleanup was
  the dominant cost of our first encoding. Following Kymn et al. (2025) and
  Hanley et al. (2025), we split the Fourier bins into one band per prime p ∈ {5, 7, 11, 13, 17,
  19}. On p's band B's coefficients are p-th roots of unity, so Bⁿ there
  depends only on n mod p. Cleanup becomes algorithmic: match each band
  against p candidates, combine by the CRT, confirm by similarity. That is
  72 checks instead of a scan, over a range of ±808,307.
- **The failure.** In a superposition, every integer whose residues each
  match *some* member matches on every band. For ν(B¹¹ + B¹²) there are 62
  such chimeras, and each has cosine 0.665 to the superposition, exactly
  the value of the members 11 and 12. Random integers score 0.00 on average.
- **Fix.** Integers inside maps and sets are boxed behind pointers.

**4.4 Zero.**
- sim(B⁰, Bⁿ) = (number of moduli dividing n)/6. So a threshold-based
  `zero?` would report 5005 = 5·7·11·13 as zero. We caught this in review.
- B⁰ is also a delta at index 0, so lesioning that one dimension destroyed
  all arithmetic, which we observed at f = 0.005.
- **Fix.** n = Bⁿ⊗Z with a random unitary Z. `(+ 17 25)` is then correct in
  300 of 300 runs up to 10% lesion.

**4.5 Fixed thresholds cap robustness.**
- **Symptom.** Every stored structure failed at σ ≈ 1.5–1.7 for every D,
  while the underlying cleanup survives σ ≈ 14.
- **Cause.** The interpreter recognised a kind at cos > 0.5 and a global at
  cos > 0.4. A clean vector's cosine to its noisy probe is 1/√(1 + σ²),
  which crosses 0.5 at σ = √3, whatever D is. The global lookup's
  0.707/√(1 + σ²) crosses 0.4 at σ ≈ 1.46. Both match the measurements.
- **Fix.** Decide only by argmax, or relative to the chance floor z/√D.
  Equality cleans up both sides and compares identities.
- **Lesson.** Robustness belongs to the decisions, not to the vectors.
- **Prior practice.** Fixed thresholds are common: Nengo's associative
  memory defaults to 0.3 for every dimension, and TK24 set θ↑, θ↓ "a little
  under 1 and a little over 0". That chance similarity scales as 1/√D is
  standard (e.g. Thomas et al., 2021). To our knowledge, the consequence for
  decisions has not been stated in the systems we surveyed: a fixed cutoff
  caps noise tolerance at a D-independent level.

## 5. Robustness and capacity

**Protocol.**
- σ is relative: a unit vector gets noise of norm σ.
- Probe noise hits every cleanup probe; operation noise hits every bind,
  unbind and bundle; a lesion zeroes a fraction f of dimensions everywhere.
- All knobs act inside the substrate. With every knob off, memory is
  bit-identical to the uninstrumented system.
- Wilson 95% intervals, over typically 20 seeds.

**Primitive cleanup matches theory** (`figures/e1.png`).
- The prediction P(1 + ε₀ > maxᵢ εᵢ) is within 0.007 of the measurements on
  average.
- σ₅₀ scales with √D: 8.9 → 36.9 from D = 512 to 8192 at |M| = 10².
- Load matters little: 1000× more atoms lowers σ₅₀ from 17.8 to 10.2 at
  D = 2048.

**Structures** (`figures/e2c.png`). After the fix of §4.5, the cliffs move
with √D:

| σ₅₀, probe noise | D = 1024 | 2048 | 4096 |
|---|---|---|---|
| 4-list | 1.60 → 4.40 | 1.69 → 6.20 | 1.70 → 8.56 |
| 8-set | 1.54 → 2.56 | 1.68 → 3.84 | 1.70 → 5.00 |
| 8-map | 1.26 → 1.09 | 1.58 → 1.83 | 1.70 → 3.04 |

*Table 2. Before → after calibration.*

Maps gain least. A map value is recalled at share 1/√(3n), and a lost value
returns *silently* as another stored atom. Overall the interpreter stays
2–8× below its own cleanup, because a round trip chains dozens of cleanups
and the weakest decides.

**Capacity is linear in D** (`figures/e3.png`).
- n₉₀ ≈ D/64 map entries and D/36 set members. The enforced caps give
  P ≥ 0.98.
- Overload forgets rather than hallucinates: false membership stays at or
  below 0.06% up to 3× capacity.
- Programs have less headroom than single lookups: a map-update loop makes
  about 2n² lookups and fails at n = 20 (D = 2048) even without noise.

**Lesions: storage degrades gracefully, computation does not**
(`figures/e4c.png`).
- Lists survive f₅₀ = 0.60–0.80.
- `(fact 4)` fails at f₅₀ ≈ 0.035 at every D, because each bind re-loses
  the dead dimensions: a chain of k binds keeps (1 − f)ᵏ of the signal
  until the next cleanup.

**Run length, not memory size, drives failure** (`figures/e5.png`,
`figures/e5b.png`).
- Near the cliff, P(correct) decays geometrically with the number of
  cleanups.
- 10⁴ extra stored atoms changed nothing (320/320 correct).
- A growing memory costs time, not correctness: a scan of 10⁴ rows at
  D = 8192 takes 10.9 ms, bandwidth-bound.

## 6. Hopfield cleanup does not help

We implemented the modern Hopfield network (MHN) in dense-associative form.
Autoassociative cleanup iterates p ← Mᵀ softmax(β M p̂); pointer dereference
is Vᵀ softmax(β K p̂), which is one attention head. It returns either the
winning row (snap) or the blend (soft). We compared it with the lookup
table and with a linear (β = 0) memory on every experiment of §5 and on the
benchmark programs (`figures/p2-e1.png`, `figures/p2-tasks.png`).

| σ₅₀, D = 2048, \|M\| = 10³ | |
|---|---|
| lookup table | 14.1 |
| MHN soft, β = 16, 1 step | 11.8 |
| MHN snap, β = 16, 10 steps | 10.8 |
| MHN snap, β = 4, 10 steps | 0 |
| classical Hopfield | 3.9 |

*Table 3. The 50% noise level for each cleanup memory.*

- **β ≥ 64 is the table**, down to which seeds fail, and 1.5–3× slower.
- **Everything else is worse.** Iteration lowers the tolerance and stops it
  growing with √D (19.9 → 11.6 at D = 4096). Small β falls into the mean of
  all rows. One soft step at β = 16 fails every benchmark task at σ = 2,
  where the table passes three of five.
- **Why.** For a probe that is one stored row plus isotropic noise, argmax
  is the maximum-likelihood decision, and iterating cannot add information.
  Lesions do not help either, because stored patterns are damaged exactly
  like the cue.
- **β is not a collapse knob.** We hoped small β would keep all k items of
  a superposed probe and large β would collapse to one. One step has a
  window for equal weights only (N = 1000, k = 4: β ∈ [12, 32], left edge
  ≈ √k·ln(N/k)). With 3 or 10 steps no β works in any of 54
  configurations (`figures/p2-collapse-equal.png`).

## 7. The interpreter in vector memory

**7.1 A tower.** A metacircular evaluator for the whole dialect, written in
the dialect and run by the host evaluator.
- **Design.** Closures are tagged lists. Environments are chains of small
  VSA maps, each searched by one `contains?`, which is 16% fewer cleanups
  than an association list.
- **Cost.** One level costs a constant 61–65 host steps per interpreted
  step, and a third level (a λ-calculus interpreter) 64× again, so k levels
  cost about 65^(k−1). Wall time grows 66–130×, because each level grows
  memory 16–27× faster.
- **No tail calls.** Every nested evaluation recurses on the host stack.
  On 256 KiB the host evaluator reaches depth 416, and the tower reaches 9.

**7.2 A vector CEK machine** (`figures/paper-f8-cek-machine.png`).
- **State.** The whole state is one pointer,
  s ↦ ν(C⊗control + E⊗env + K⊗continuation + M⊗mode). The continuation is
  a list of frame records.
- **Rules.** 35 transition rules live in a rule memory. Each rule's value
  is a microprogram, a list of instruction vectors over 9 instructions and
  13 registers.
- **Keys are sums, not products.** A key bundles role-bound features (mode,
  kind, head class, identity class, top frame). A rule's key names only the
  features it tests. A fully matching k-feature rule scores √(k/5), so
  nearest-neighbour selection picks the *most specific* matching rule, and
  short keys act as defaults. The minimum winning margin over 2,380
  selections is 0.096, more than 4σ.
- **The host loop**: fetch the key, select the nearest rule, run it,
  collect working memory. That is 47 lines with no Lisp in them.
- **Host knowledge left**, listed explicitly: argument marshalling for host
  primitives walks cons cells; two memory labels; the four empty-collection
  atoms; one `apply` helper.
- **Results.**
  - Every test form gives the host evaluator's result.
  - Tail calls run in constant space: `(countdown 10000)` completes on a
    256 KiB stack, where the host evaluator overflows at 1000.
  - `(fact 5)` takes 2.04 s against the host's 2.12 s, with a decoded-
    instruction cache; decoding every instruction from vectors is 12×
    slower.
  - The depth limit is now the growth of M, not the stack.
- **"Fully vector-symbolic", made precise.** TK24 and §3 are
  vector-symbolic in data. The machine is vector-symbolic in data, control
  state and rules, and a program could rewrite its own interpreter. The
  host still parses, prints, runs primitives and a fixed instruction set,
  like a CPU.

## 8. Superposition programming

A weighted sum of values is a vector too. §8 asks whether a program can
compute on all of its worlds at once.

**Memory.** Pointers break linearity: `cons` of a superposed element gets a
fresh pointer, not the sum of the pointers of the individual cells. So
superposition has to survive dereferencing: M(Σ wᵢpᵢ) should be
Σ wᵢ·traceᵢ.
- **A linear memory** also returns every other stored trace. Its fidelity
  is 1/√(1 + N/D): 0.34 at N/D = 8.
- **The table and the softmax** collapse from k = 2 worlds.
- **Thresholded projection (*proj*) works.** It keeps the rows above chance
  (τ = (√(2 ln N) + margin)/√D) and solves least squares on them. Weights
  come back within total variation 0.01–0.06, independent of memory load,
  for about D/100 worlds. On a single world it equals the table, so
  ordinary programs run unchanged.

**Semantics.**
- **Building and reading.** `amb` and `superpose` build Σ wᵢvᵢ, and the
  weights read directly as probabilities (cf. Furlong & Eliasmith, 2022).
  `worlds` reads them back and throws past capacity. `collapse`, `sample` and `assert` (post-selection)
  complete the set.
- **Destructors lift linearly.** `first`, `rest`, environment and global
  lookup, and `get` with a superposed key are each one memory read for
  every world. Integer arithmetic lifts exactly, because ⊗ distributes over
  +: `(+ (amb 1 2) 10)` equals `(amb 11 12)` to 10⁻¹⁵.
- **Function bodies run once** for all worlds. Constructors, and primitives
  with no linear form, enumerate.
- **`if` mixes both branches π : 1−π**, where π is the truthy weight of the
  test. TK24's ⊕ gets the outer cases right (prune a branch no world needs,
  which is lazy evaluation). Its middle case weights the branches α : 1 with
  α = cos(test, T), and errs by 0.09–0.16 in the branch share. Weighting by
  the test's coefficients errs by 0.000 (`figures/w0-q3-if-oplus.png`).
- **Recursion.** Each world of the tested variable goes to its own branch,
  or a world that reached its base case recurses forever. Every field read
  is multiplied by its known gain (√2 per cell), or deeper worlds lose
  weight as 2^(−d/2). With both, `(dbl (superpose {2 5, 3 3, 5 2}))` gives
  {4: .5, 6: .3, 10: .2}.
- **Run-time choice.** A vector carries no world labels, so two occurrences
  of a variable are independent draws. `(let [x (amb 1 2)] (+ x x))` gives
  {2: ¼, 3: ½, 4: ¼}. This is *run-time choice* in the sense of Hussmann
  (1993), as opposed to the *call-time choice* of functional logic
  languages such as Curry, where a variable keeps one value. Call-time
  choice would need a tag on every world (√k signal cost, k fixed in
  advance), so it is explicit, via `for-worlds`.

| demo | result | correct at |
|---|---|---|
| sprinkler Bayesian network | P(rain \| wet) = 0.3625 (exact 0.36255) | D ≥ 1024 |
| 2d4 dice sum, one bind | 7 worlds exact to 10⁻¹⁵ | D ≥ 4096 |
| BFS, one vector per level | levels equal the host BFS | 20 / 40 / 80 nodes at D = 1024 / 2048 / 4096 |
| parallel map | total variation ≤ 0.0003 | D ≥ 1024 |

*Table 4. Superposition demos, 5 of 5 seeds per point
(`figures/w-demos.png`, `figures/w-bfs.png`).*

The Bayesian query is a check of the semantics, not a contribution.
Representing distributions as superposed vectors and conditioning on them
is established (Furlong & Eliasmith, 2022, 2024; Dewulf et al., 2023), and
this query enumerates its worlds with `for-worlds`, so it is no faster than
ordinary code. The contribution is the language: superpositions as ordinary
values, with destructors, lookup and function bodies lifted over all worlds.

- **Failures throw** a capacity error instead of returning a wrong
  distribution. The measured exceptions: wide-range integers at D = 8192 (3
  of 20 silently wrong), one BFS run (1 of 67 nodes misplaced), and the
  smallest world under operation noise.
- **Capacity.** One memory step reads back 8, 24 and 32 worlds, and at
  least 64, at D = 1024, 2048, 4096 and 8192. Integer worlds stay at about
  5 for every D, because residue integers are correlated
  (`figures/w-readout.png`).
- **Noise.** The weight error grows linearly with operation noise.
  Programs that clean up every few steps hold to σ = 0.3; a chain of 20
  binds with no cleanup fails from σ = 0.2 (`figures/w-lift.png`).
- **No speedup.** A superposed step still scans memory, and the sprinkler
  query takes 1.4–13 s. What superposition buys is a programming model, not
  speed. Exponentially many worlds exist only in factored form, which needs
  a resonator search to read out (Frady et al., 2020; Kent et al., 2020); we
  did not build one.

## 9. Discussion

- **Cleanup is attention**, and a symbolic interpreter is the wrong regime
  for it. Its probes are one item plus noise, and every decision needs a
  single winner, so argmax is optimal and softness only leaks. The one
  place that wants blends, superposition, needs a separation function
  outside the softmax family (in the terms of Millidge et al., 2022).
- **Pointers versus linearity** is the central tension. Pointers buy
  identity and lose superposition; a linear memory keeps superposition and
  crosstalks with everything stored. *proj* is linear on the detected
  worlds and snaps everywhere else.
- **Cartesian closure is necessary, not sufficient.** We quantify TK24's
  "great demands of memory":
  - a record holds about D/64 entries;
  - reliability decays with the number of cleanups;
  - memory growth, not stack depth, bounds recursion once control is in
    vectors.
- **Graceful degradation, qualified.** Storage survives the loss of 60–80%
  of dimensions, computation about 4%. A brain-like VSA computer needs
  cleanup interleaved with computation at a fine grain.

## 10. Limitations

- **Speed:** seconds per small program, against microseconds natively.
- **Scale:** records hold about D/64 entries, superpositions about D/100
  worlds, integer superpositions about 5.
- **Language coverage:** no macros, `loop`/`recur`, floats, ordering
  comparisons or strings. The CEK machine does not detect arity errors.
- **No garbage collection** for M.
- **Statistics:** some grids are coarse (4–5 seeds for tasks and demos).

## 11. Conclusion

A Clojure made of vectors runs: records, sets, closures, and the
interpreter's own control state. The findings are where it bends:
- identity versus similarity;
- symmetric binding;
- correlated number codes;
- thresholds;
- signal leaking between cleanups.

Hopfield networks fix none of these. Superposition programming works within
a narrow capacity, with run-time choice as its native semantics and a
non-softmax memory to keep the worlds apart.

## References

- Dewulf, P., De Baets, B., & Stock, M. (2023). The hyperdimensional
  transform for distributional modelling, regression and classification.
  arXiv:2311.08150.
- Frady, E. P., Kent, S. J., Olshausen, B. A., & Sommer, F. T. (2020).
  Resonator networks, 1: An efficient solution for factoring
  high-dimensional, distributed representations of data structures. *Neural
  Computation*, 32(12), 2311–2331. doi:10.1162/neco_a_01331
- Furlong, P. M., & Eliasmith, C. (2022). Fractional binding in vector
  symbolic architectures as quasi-probability statements. *Proc. CogSci
  2022*, 259–266.
- Furlong, P. M., & Eliasmith, C. (2024). Modelling neural probabilistic
  computation using vector symbolic architectures. *Cognitive
  Neurodynamics*, 18(6). doi:10.1007/s11571-023-10031-7
- Hussmann, H. (1993). *Nondeterminism in Algebraic Specifications and
  Algebraic Programs*. Birkhäuser.
- Hanley, C., Tomkins-Flanagan, E., & Kelly, M. A. (2025). Hey Pentti, we
  did (more of) it!: A vector-symbolic Lisp with residue arithmetic. *IJCNN
  2025*. arXiv:2511.08767.
- Kanerva, P. (2014). Computing with 10,000-bit words. *52nd Allerton
  Conference on Communication, Control, and Computing*, 304–310.
- Kent, S. J., Frady, E. P., Sommer, F. T., & Olshausen, B. A. (2020).
  Resonator networks, 2: Factorization performance and capacity compared to
  optimization-based methods. *Neural Computation*, 32(12), 2332–2388.
  doi:10.1162/neco_a_01329
- Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2022, 2023). A
  survey on hyperdimensional computing aka vector symbolic architectures,
  Parts I and II. *ACM Computing Surveys*, 55(6), Art. 130; 55(9), Art. 175.
- Kymn, C. J., Kleyko, D., Frady, E. P., Bybee, C., Kanerva, P., Sommer,
  F. T., & Olshausen, B. A. (2025). Computing with residue numbers in
  high-dimensional representation. *Neural Computation*, 37(1), 1–37.
  doi:10.1162/neco_a_01723
- Meier, C. (2023). Vector symbolic architectures in Clojure. Talk,
  Clojure/Conj 2023. https://youtu.be/j7ygjfbBJD0. Library:
  https://github.com/gigasquid/vsa-clj
- Millidge, B., Salvatori, T., Song, Y., Lukasiewicz, T., & Bogacz, R.
  (2022). Universal Hopfield networks: A general framework for single-shot
  associative memory models. *ICML 2022*, PMLR 162, 15561–15583.
- Nengo associative memory, `nengo/networks/assoc_mem.py` (default
  `threshold=0.3`). https://github.com/nengo/nengo
- Plate, T. A. (1995). Holographic reduced representations. *IEEE
  Transactions on Neural Networks*, 6(3), 623–641.
- Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., et al.
  (2021). Hopfield networks is all you need. *ICLR 2021*. arXiv:2008.02217.
- Thomas, A., Dasgupta, S., & Rosing, T. (2021). A theoretical perspective
  on hyperdimensional computing. arXiv:2010.07426.
- Tomkins-Flanagan, E., & Kelly, M. A. (2024). Hey Pentti, we did it!: A
  fully vector-symbolic Lisp. Abstract, MathPsych/ICCM 2024.
  arXiv:2510.17889.
- Tomkins-Flanagan, E., Hanley, C., & Kelly, M. A. (2025). Hey Pentti, we
  did it again!: Differentiable vector-symbolic types that prove polynomial
  termination. Abstract, MathPsych/ICCM 2025. arXiv:2510.16533.

## Appendix: Reproducibility

- Substrate: numpy 2.5.3 wheels (scipy-openblas, one BLAS thread). A distro
  numpy on reference BLAS made cleanup about 40× slower.
- Figures:
  - §5–§8: `clojure -M:jvm:exp specs/<name>.edn`, then
    `.venv/bin/python scripts/plot*.py <name>`.
  - F1, F8: `scripts/paper_diagrams.py`.
  - F2: `scripts/paper_f2.py`.
  - F3: `scripts/paper_f3.py`.
- Tests: `clojure -M:jvm:test` (about 5 min), `clojure -M:jvm:test-s1`
  (about 25 min), `clojure -M:jvm:bench`.
