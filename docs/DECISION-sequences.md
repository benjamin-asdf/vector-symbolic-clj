# Decision: lists are pointers, vectors are positional superpositions

Status: adopted (October 2026). This decides §3 of
`docs/PLAN-numbers-and-cells.md` for option 2.

## The decision

The dialect has two sequence types, and they make opposite trade-offs. The
program picks one per value, the way a Clojure program picks between a list
and a vector.

| | list `(a b c)` | vector `[a b c]` |
|---|---|---|
| encoding | chain of hash-consed pointers, p ↦ ν(L⊗a + R⊗rest) | v = ν(Σᵢ Πⁱ(V⊗xᵢ)), stored in M as itself, with M: v ↦ N⊗n |
| similar contents | unrelated vectors | similar vectors |
| element i | i `cdr` steps, then `car` | one readout, M(V ⊘ Π⁻ⁱ(v)) |
| `count` | walks the chain | one memory read, M(N ⊘ M(v)) |
| length | unlimited | about D/48 (43 at D = 2048, 86 at 4096) |
| equal contents | equal by hash-consing | equal by construction (M only dedupes rows) |
| noise tolerance | nearly flat in length (σ₅₀ 6.3 / 5.6 / 5.3 for n = 4 / 16 / 32, D = 2048) | falls with length (σ₅₀ 5.0 / 2.0 / 1.2) |
| read out of a field | cleaned up against random pointers | cleaned up against stored look-alikes too (cost measured below) |
| superposition of several | keeps the joint worlds | keeps only each position's marginal |
| inside a map or set | as is | boxed behind a pointer, like integers |
| integer elements | as is | boxed behind a pointer (hash-consed per value) |

Use a **list** for recursion, accumulators, code, and tuples of correlated
worlds. Use a **vector** for records whose similarity matters, for indexed
access, and for short data.

## Why

Making similar structures similar vectors is a large part of what HDC
promises for structured data. The pointer cells that fixed §4.1 of the paper
give that up everywhere. The trade-off itself cannot be avoided:

- A cleanup's margin is 1 minus the similarity of the target to the nearest
  wrong entry in the set it is cleaned against.
- An encoding that makes look-alikes similar makes exactly that similarity
  high.
- So similarity and robustness are the same quantity seen from two sides.

Option 1 of the plan, transparent trees everywhere with an explicit pointer
form, would have spent this trade-off for the program by rule. Option 2 lets
the program spend it, and keeps the interpreter's own structures
(environments, closures, code, the CEK machine's frames) on the robust side,
because those are lists.

## How vectors are encoded, and why each part is there

- **Positions are permutation powers Πⁱ.** Π is a fixed random permutation
  of the dimensions, drawn from its own stream (`Space.permute`).
  - It is not binding with a role R, and it is not a cyclic shift. Both
    commute with binding. Once integers are shifts (plan §1), Rⁱ⊗L⊗n would
    equal L⊗shiftⁱ⁺ⁿ(Z), so 4 at position 0 would equal 3 at position 1.
- **V binds each element.** A nested vector's terms would otherwise alias
  its parent's: Πⁱ(Πʲ(y)) = Πⁱ⁺ʲ(y). Π does not commute with V⊗, so
  Πⁱ(V⊗Πʲ(V⊗y)) is a different vector.
- **A vector is a row of M itself, labelled :vec, not a pointer.** Its kind
  is then recognised like any other value's, cleanup works, and a
  superposition of a few vectors reads back through the item memory.
- **The count is not part of the content.** The row of v points to N⊗n,
  and `count` reads it through the memory, M(N ⊘ M(v)). Two earlier
  versions put the count into the vector, and both failed:
  - as C⊗n, the role of map and set sizes, every vector of length n
    correlated with every map and set of size n, and peeling a set found
    stored vectors as its members;
  - as ½N⊗n, any two vectors of the same length still had similarity 0.2.
    The *proj* memory keeps every row above chance, so it read unrelated
    vectors into superpositions as spurious worlds: a nested `[:y :z]`
    decoded with `[acc x]` and `[i acc]` at small weights, and a `relation`
    built from vectors lost a key once enough vectors were stored.

  Now unrelated vectors are orthogonal, similarity comes from shared
  elements alone (3 of 4 shared: 0.74; none: 0.02), and each element's
  share of the trace is 1/√n.
- **Capacity is calibrated: 1.5 parts in `capacity`, about D/48.** Every
  element is one cleanup, with no explaining away to help. Keyword and
  integer elements read back exactly up to 48 elements at D = 2048 (5
  seeds) and start failing at 56 (3 of 280 elements wrong, 13 of 320 at
  64). The cap, 43 at D = 2048, sits a little below that.
- **Integer elements are boxed**, as in maps and sets. In the ½N⊗n version
  the residue readout of an integer inside a vector failed from 16
  elements at D = 2048 (5 of 80 elements wrong), where the codebook was
  still exact. A box is hash-consed per value, so vectors that share
  integers at the same positions stay similar. Boxed, integers read back as
  well as keywords. Nested vectors are not boxed, so similarity reaches
  into them.
- **The exact-copy thresholds follow the capacity.** Two vectors of n
  elements that differ in one have similarity 1 − 1/n. Above n = 200
  (D ≳ 9600) that passes θ_eq = 0.995, so `init!` raises θ_eq to
  1 − 0.25/capacity and the row dedupe to 1 − 0.125/capacity.
- **N and V come from their own random streams** (`Space.aux_unitary`), so
  adding them shifts no atom. Programs without vectors draw the same atoms
  as before.

## Consequences

1. **Superposed vectors keep only marginals.** A vector is linear in its
   elements, so [1 1] + [2 2] = [1 2] + [2 1] exactly. A superposition of
   vectors over the same positions cannot be decomposed back into its joint
   worlds. The dice and sprinkler demos built their joint worlds as vectors
   and read back wrong distributions (TV 0.44–0.50 and 0.43–0.82 over 3
   seeds at D = 4096, `scripts/failure_modes.clj`). They now build
   them as lists. A few linearly independent vectors, like
   `(amb [1 2] [10 20])`, still read back exactly.
2. **`first` of a superposed vector lifts linearly**: element 0 is one
   unbind, so `vsc.worlds` reads it without enumerating. `rest` of a vector
   is a new list in each world, so it enumerates.
3. **Vectors are less noise tolerant than lists, and more so the longer
   they are.** Each element is read at share 1/√n, against 1/√2
   for a cell field, so σ₅₀ falls with n (E2v below). A 4-vector is
   nearly as tolerant as a list, a 16-vector about as fragile as an
   8-entry map. The look-alike cost, which motivated pointers in the first
   place, is small by comparison (E2l below).
4. **Indexing reads the index.** `nth`/`get` clean up the index once to an
   integer and apply Π⁻ⁱ. A permutation power is not a vector the machine
   could bind with, so here the host chooses an operation from a cleaned-up
   number, as the CEK machine's `PRIM` does.
5. **Capacity is a hard error**, as for maps and sets: `(vec (range 50))`
   fails at D = 2048, and works at D = 4096. A long sequence is a list.
6. **The CEK machine walks a vector as `(seq v)`.** Its microcode read
   vector literals, `let` bindings and `fn` parameters as cell chains. Each
   site now converts once with a `PRIM seq`, as the map and set literals
   already did. The ISA is unchanged.
7. **`rest` of a vector returns a list**, as in Clojure. Before, it returned
   a vector tail.
8. **On the linear memory, small programs got more reliable.** Over 20
   seeds, `(mul 3 n)` succeeds 19/17/13/9 times for n = 2..5, against
   19/13/8/2 with pointer-cell vectors. The test now asserts a rate over 10
   seeds instead of one draw.

## Measurements

All at probe noise σ on every cleanup (a unit probe gets noise of norm σ),
12 seeds, Wilson 95% intervals in the figures. σ₅₀ is the noise at which
half the round trips fail.

**E2v: round trip of a stored sequence** (`specs/e2v.edn`,
`figures/e2v.png`). The structure is stored noise-free as a global, then
read back and decoded under noise.

| σ₅₀ | D = 1024 | 2048 | 4096 |
|---|---|---|---|
| list, 4 atoms | 4.44 | 6.27 | 8.34 |
| list, 16 | 4.00 | 5.60 | 8.00 |
| list, 32 | 3.60 | 5.33 | 7.66 |
| vector, 4 atoms | 3.00 | 4.96 | 7.20 |
| vector, 16 | 1.16 | 2.04 | 2.93 |
| vector, 32 | – (over capacity) | 1.16 | 1.71 |
| vector of 3 maps | 1.87 | 2.84 | 4.27 |
| *vector of 3 maps, pointer cells (P1, 20 seeds)* | *1.98* | *2.96* | *4.36* |

- A list's tolerance hardly depends on its length: it is a chain of cleanups
  at share 1/√2 each, and the weakest decides.
- A vector's tolerance falls with its length, as its element share
  1/√n predicts.
- Near its capacity a vector survives little noise (σ₅₀ = 1.16 for 32
  elements at D = 2048). The capacity is a noise-free limit, as it is for
  maps.
- The vector of 3 maps matches the pointer-cell baseline at every D: its
  weakest part is still the maps.
- In the ½N⊗n version (count in the content, share 1/√(1.25 n), cap D/64)
  the same grid gave 2.97 / 1.71 / 0.69 for 4 / 16 / 32 elements at
  D = 2048, and 3.20 for the vector of maps at 4096.

**E2l: the price of similarity** (`specs/e2l.edn`, `figures/e2l.png`).
k sequences of 8 symbols that differ only in the last one are stored in one
list, and the first is read back out of its field. With k = 16 a vector is
cleaned up against 15 stored look-alikes at similarity 7/8; with k = 1
there is nothing to confuse it with.

| σ₅₀ | D = 1024 | 2048 | 4096 |
|---|---|---|---|
| list, alone | 4.27 | 5.60 | 7.52 |
| list, 15 look-alikes | 3.80 | 5.44 | 7.68 |
| vector, alone | 2.51 | 3.09 | 4.27 |
| vector, 15 look-alikes | 2.40 | 2.93 | 4.16 |

- Look-alikes cost a vector of 8 about 3–5% of its σ₅₀, the same order
  as for a list (whose look-alikes are unrelated pointers, so its change
  is noise). Its weakest step is reading an element (share 0.35), not
  telling it from its look-alikes.
- The look-alike margin hurts less than it looks, because only noise along
  the difference between the two vectors can swap them, and that
  difference has norm √(2(1 − s)). Per unit of noise the margin is
  √((1 − s)/2) times the probe's share, not 1 − s. For n = 8 (s = 7/8)
  that is 0.25 × 0.71 ≈ 0.18, against 0.35 for an element; for n = 32
  (s = 31/32) it is 0.09 against 0.18. That is exactly half an element's
  share at every n (an earlier version of this note said look-alikes
  matter only near capacity, which does not follow). So the margin alone
  does not explain the small measured cost. A likely reason, not
  isolated: telling the vector from its look-alikes is one decision
  against 15 rows, while each of the n element reads competes with every
  stored atom, and the weakest of those already sets σ₅₀.
- This is not §4.1 of the paper again. There, look-alikes were cleaned up
  at every step of a recursion and the margins of thousands of reads
  compounded. Here a vector is cleaned up once per read out of a field, and
  the interpreter's own structures are lists.

Not measured: operation noise and lesions (E2op, E4) for vectors, and the
P3 benchmark under noise. All 12 benchmark tasks remain exact at zero
noise.

## What stays open

- **Shift integers (plan §1)** would let integer elements go unboxed: a
  shift integer is read by one FFT against an orthogonal codebook, so the
  readout that failed here should not fail. That would save one pointer
  per integer element.
- **Cleanup by reconstruction** (plan option 1): clean a noisy vector by
  decoding each element against the atoms and re-encoding it, instead of
  snapping to the nearest stored vector. That would remove consequence 3,
  at O(n) cleanups per read.
- **Algebraic `conj`**: today `conj` onto a vector rebuilds it. Adding
  Πⁿ(V⊗x) and stepping the count would be linear.
