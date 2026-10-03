# Plan: shift-encoded integers, and pointers only where needed

Two design changes, both prompted by the question of what the alternatives
to our integer and cell encodings would have been.

## 1. Integers by cyclic shift

### Decision

n = shiftₙ(Z): the anchor Z (a fixed random unitary vector), cyclically
shifted by n places. Equivalently Bⁿ⊗Z with B = δ₁, the delta at index 1.
This replaces the residue number system.

Why this is the right B:
- **Binding is plus.** The cyclic shift commutes with circular convolution:
  shift_a(Z) ⊗ shift_b(Z) = shift_{a+b}(Z⊗Z). Unbinding Z once gives
  a + b. `inc`/`dec` are a roll by ±1, exact and O(D). A random permutation
  would not commute with binding, so it would not give this.
- **Integers are exactly orthogonal.** sim(n, m) is Z's autocorrelation at
  lag n − m, which is about N(0, 1/D) for n ≠ m. So there are no chimeras,
  zero is not special, and 5005 is not near 0.
- **Decoding is one FFT.** corr(v, Z) = irfft(rfft(v)·conj(rfft(Z))) gives
  the similarity of v to every integer at once, and the argmax is n. It is
  a full codebook scan at the cost of an FFT, with no stored codebook.
- **Superposed integers read out as peaks**, one per world, with the peak
  height as the weight. The residue design's limit of about 5 integer
  worlds disappears; capacity is that of any orthogonal codebook.

**The price: range.** Shifts have period D, so integers live mod D, in
[−D/2, D/2): ±1024 at D = 2048. Overflow wraps silently and cannot be
detected from the vector. Options, to decide after measuring:
- (a) accept it: the benchmark tasks use small integers, and D = 4096 gives
  ±2048;
- (b) the reader and arithmetic primitives check the range on the host and
  refuse, as `num-vec` does today;
- (c) large integers as numerals in the dialect: lists of base-D digits,
  with carry written in the dialect.

Periodicity is what the residue system avoided, by combining several
periods; that is exactly where its chimeras came from. Range versus
orthogonality is the real trade-off, and we choose orthogonality.

### Work

1. **Substrate** (`hdc.py`): a `ShiftNumbers` class with `vec(n)` (a roll of
   Z), `step` (δ₁), `offset` (Z), `read(v)` → [n, sim] by the FFT argmax,
   and `peaks(v, k)` for superpositions. Same interface as `Numbers`, so the
   core changes only at construction.
2. **Core**: switch `init!` to `ShiftNumbers`; keep `+` = a⊗b⊘Z and `-`,
   `inc`, `dec` as they are (they already take Z into account).
3. **Remove what only the residue design needed**:
   - `Numbers` (residue bands, CRT, `_band_scores`, `int_dots`,
     `more_ints`, `int_worlds`), replaced by FFT peaks;
   - the boxing of integers inside maps and sets (`slot`/`unslot` for
     `:num`). Keep the boxing of map *values* ⟨v⟩: it fixes the commuting
     key/value bug (§4.2), which is unrelated;
   - `int-print-range`, and the integer special cases in `worlds`
     capacity (D/400);
   - `scripts/paper_f3.py` and `figures/paper-f3-chimeras.png`.
   Keep `zero?` as equality (still correct) and the dense-zero test (Z is
   the anchor, so it holds by construction).
4. **Tests.** All suites must pass. New tests: orthogonality of 0…100
   (max |sim| < 5/√D), `+`/`-`/`inc` exact over the range, wrap-around
   behaviour as decided above, superposed integer readout of 20 worlds at
   D = 2048, and lesioning index 0 for `(+ 17 25)`.
5. **Measure against the residue baseline** with the existing specs:
   E2/E4 (structures, lesion), the 12 bench tasks, and `w-readout` for
   integer worlds. Expected: integer-world capacity rises from about 5 to
   about D/100, and maps holding integers lose no robustness. `(fact 4)`
   under lesion should stay at about 0.035, because its decay happens
   between cleanups (unchanged).
6. **Texts.** Following "findings update the ideas":
   - notebook: drop failure modes 3 (chimeras) and 4 (the 5005 trap); keep
     the dense-zero point only if it is still true in the new design (it
     is not: Z is dense from the start);
   - paper: replace §4.3–4.4 with one sentence where it would be tempting:
     "A residue number system makes cleanup algorithmic, but its integers
     correlate, which yields chimeras in superpositions; the cyclic shift
     gives exactly orthogonal integers and FFT decoding, at the price of a
     range of ±D/2";
   - rename "seven failure modes" to five (4.6 and 4.7 came from §3 of this
     plan); update the abstract, the README
     and `docs/results-*.md` (keep the measured residue numbers in the
     results docs as history, marked as the old design).

## 2. Fractional power encoding: an optional real-number type

The same scheme extends to non-integers. A fractional shift x (phases
2πkx/D in the Fourier domain) is bandlimited interpolation, and
sim(x, y) is a Dirichlet (≈ sinc) kernel of x − y: integers stay
orthogonal, and reals within about 1 of each other are similar. Addition
is still binding.

- A separate kind `:real` with its own anchor Z_r, so that integers stay
  exact and orthogonal.
- **Bandwidth parameter** b ≤ D/2: use only the lowest b frequencies. A
  smaller b gives a wider kernel (smoother similarity, better for `<`-like
  comparisons and noisy magnitudes) and fewer distinguishable values.
- **Decoding**: the correlation peak with sub-sample interpolation
  (parabolic or sinc); precision is set by noise and b.
- Opens `<`, `>` by similarity (a smooth kernel), and probability density
  readouts (Furlong & Eliasmith's FPE work).
- Optional and later; not needed by any current task.

## 3. Cells and sequences: two options

Today every cell is a hash-consed pointer p ↦ ν(L⊗a + R⊗b). That fixed
§4.1, but it gave up what makes HDC attractive for structured data: similar
structures are similar vectors, and constructors are linear, so they lift
over superposed worlds without enumeration.

### Why pointers won §4.1

A cleanup's margin is 1 minus the similarity of the target to the nearest
wrong entry in the codebook it is cleaned against. Pointers make that
codebook random, so the margin is about 1. TK24's traces lost margin for
two reasons:

1. **ν at every level.** Element i of a nested cell has amplitude about
   2^(−i/2), so siblings differ only far down (cosine 0.89 in §4.1).
2. **Intermediate sub-traces are cleaned up against stored sub-traces.**
   A look-alike only hurts when a *sub-tree* competes with *other
   sub-trees*. A leaf cleaned up against random atoms keeps its full margin,
   however similar the trees are.

Both options below remove these causes in different places.

### Rejected: pointer on collision (old rule A)

Allocating a pointer when a new trace comes too close to a stored one
makes the encoding depend on what happens to be stored already. Equality
survives if the rule runs at `intern` time, but whether `cons` is linear,
and whether two values are similar, would depend on run history. That
makes the superposition semantics unpredictable.

### Option 1: transparent trees, with an explicit pointer form

**Encoding.** No ν. A tree is a flat superposition of its leaves under
their paths, Σⱼ pathⱼ ⊗ leafⱼ:

    cons(x, l) = L⊗Π(x) + R⊗l        ;; Π a fixed random permutation

- **Π is necessary.** Binding commutes, so without it the path R^i L R^j L
  equals R^(i+j) L², and different shapes alias: the y in [[x y] z] would
  sit where the y in [w [y]] sits. Π is linear, so superposition is kept.
- **R must not be δ₁**, the step of the shift integers: Rⁱ⊗L⊗(the integer
  n) would equal L⊗shift^(i+n)(Z), so 4 at position 0 would equal 3 at
  position 1. Use a random unitary R.

**Operations.**
- `nth i` = cleanup(Π⁻¹(L ⊘ R⁻ⁱ⊗v)): one unbind and one cleanup against
  atoms, O(1), no chain.
- `rest` = R⁻¹⊗(v − L⊗Π(car v)): exact, since car is cleaned up and R is
  unitary.
- **Exact field recovery.** When every other field of a record can be
  cleaned up, the remaining field comes out exactly by subtraction. Example:
  an environment cell (sym, tree): clean sym, subtract it, and the tree is
  left without noise.
- **Cleanup by reconstruction.** A noisy tree is cleaned by decoding every
  leaf against the atoms and re-encoding it exactly. An element that does
  not clean up to an atom is a subtree, and the decoder descends;
  terminators give the shape. No trace is stored in M, so look-alikes never
  compete, and no hash-consing is needed: equal content is the same vector.
  Cost: O(leaves) per cleanup instead of one table lookup.

**What it keeps.** Similarity at every depth: [1 [2 3]] and [1 [2 4]] share
two of three (path, leaf) pairs. `cons` is linear in both arguments, so
superposed data needs no enumeration and, for data, no *proj* memory.

**The pointer form.** `(box x)` (reader tag `#ptr`) marks a subtree as
opaque: a hash-consed pointer as today. It is placed
- **explicitly** by the program, where identity matters more than
  similarity;
- **always** by the interpreter, for environments, closures and CEK frames,
  where look-alikes are the norm (the `(fact 5)` failure);
- **automatically, bottom-up by size:** a subtree with more than B leaves
  is boxed. This depends only on content, so equal values stay equal.

Small things are transparent, big things are opaque.

**Costs.**
- Capacity: about D/64 leaves per transparent tree (the map limit, since
  both are sums of products of near-random unitary vectors), roughly 32 at
  D = 2048 and 128 at D = 8192. Walking down a list improves it: each
  `rest` removes a term.
- Similarity stops at a box. A pointer carrying a content sketch,
  p = ν(random + γ·sketch), would trade back part of the margin. Leave it
  out at first.
- Under per-operation noise, an unboxed spine has no stored attractor:
  noise grows as √k over k operations until the next reconstruction or box.

### Option 2: lists are pointers, vectors are permutation sequences

Split by type. The programmer picks the trade-off per value, as when
choosing between a list and a vector in Clojure, and the documentation
states the choice: lists are robust and opaque, vectors are similar and
linear but bounded.

- **Lists** (`cons`, `'(…)`, `seq`) stay hash-consed pointer cells, as
  today. They are robust, have no length limit, are opaque, and are not
  linear. Recursion, accumulators and the interpreter's own structures use
  them.
- **Vectors** (`[…]`) are positional superpositions:

      [x₀ … xₙ₋₁] = Σᵢ Πⁱ(V⊗xᵢ) + C⊗n

  - `nth i` = cleanup(V ⊘ Π⁻ⁱ(v)), O(1), like Clojure's vectors.
  - `conj` adds Πⁿ(V⊗x) and steps the count: linear.
  - `count` is the C field, a shift integer, so a superposition of vectors
    of different lengths has a superposed count.
  - V is needed for nesting: Πⁱ(Πʲ(y)) = Π^(i+j)(y) would alias a nested
    element with a flat one, but V⊗ does not commute with Π.
  - Similar vectors are similar; equal vectors are the same vector without
    hash-consing; a vector holds about D/64 elements.
  - Nested vectors stay transparent under the same capacity budget; a
    nested list is a pointer and is opaque.
  - `rest`/`seq` of a vector return a list, as in Clojure.

**Costs.** Similarity and linearity exist only for vectors (and maps and
sets, which are already superpositions). A program that wants both
recursion over a sequence and similarity has to convert. The capacity
limit is a hard error, as for maps today.

### Comparison

| | Option 1 | Option 2 |
|---|---|---|
| similarity of nested data | at every depth, up to a box | vectors only |
| linear constructors | all data | vectors |
| interpreter structures | boxed by the interpreter | pointer lists |
| capacity limit | B leaves, then auto-box | D/64 per vector, hard error |
| new machinery | Π, reconstruction cleanup, `box` | Π, vector ops, count field |
| change to today's code | large | small: lists unchanged |

### Measurements (both options)

- Readout margin of an element or leaf against size and depth.
- Noise at 50% failure compared with pointer chains (the §4.1 measure).
- Whether similarity tracks shared structure.
- `(fact 5)`, `(cd 100)`, E2/E4 and the 12 bench tasks against
  all-pointers.
- Superposition: a superposed element through `cons`/`conj` without
  enumeration, read out with `worlds`.

### Decision

**Option 2, adopted (October 2026).** The implementation, the reasons, the
consequences and the measurements are in `docs/DECISION-sequences.md`. It
differs from the sketch above in four places, each found by running it:
- the count is not in the vector's content but the trace its row points to:
  as a term of the content (C⊗n, then ½N⊗n) it made vectors of one length
  similar to each other and to maps and sets, which broke superposition
  readouts;
- the capacity is calibrated to about D/48, not derived;
- integer elements are boxed, as in maps and sets;
- a superposition of vectors keeps only each position's marginal, so joint
  worlds are built as lists.

## Order

1. Shift integers (§1), with tests and measurements. Independent of the
   rest.
2. Text clean-up for the residue decision (§1.6).
3. Cells and sequences (§3): decided, option 2 (`docs/DECISION-sequences.md`).
4. FPE reals (§2), optional.
