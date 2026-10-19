# Issue findings: where each failure mode of paper §4 lived in the code

*Handover note, 2026-10-01, updated 2026-10-08. Traces the failure modes
of `paper/draft.md` §4 back to the code that had them, and records which
ones git can prove. `scripts/failure_modes.clj` rebuilds each broken
variant next to the current code (with `with-redefs`) and runs both.*

## Summary

| § | failure mode | broken code in git? | fixed in | evidence |
|---|---|---|---|---|
| 4.1 | superposed cells are fragile | **no**, fixed before `bf2a8c7` | before the first commit | reconstruction `scripts/paper_f2.py` |
| 4.2 | binding commutes | yes | `6ab342c` | `6ab342c^:src/vsc/core.clj:217` |
| 4.3 | residue integers form chimeras | **no**, fixed before `bf2a8c7` | before the first commit | reconstruction `scripts/paper_f3.py` |
| 4.4 | zero | partly (lesion bug yes, threshold `zero?` never existed) | `434b4f8` | `434b4f8^:resources/vsc/hdc.py:316`, `core.clj:284,546–547` |
| 4.5 | fixed thresholds cap robustness | yes | `434b4f8` | `434b4f8^:src/vsc/core.clj:46–47` and six call sites |
| (new) | readout bias in integer recognition | yes | `434b4f8` | `434b4f8^:resources/vsc/hdc.py:639` |
| 4.6 | a shared field (vector count in the trace) | **no**, fixed before `1b7a7ba` | `1b7a7ba` | `scripts/failure_modes.clj` |
| 4.7 | a sum of tuples forgets the joint | **no**, fixed before `1b7a7ba` | `1b7a7ba` | `scripts/failure_modes.clj` |

Line numbers of broken code refer to the parent revision named in each
section (`git show <rev>^:<path>`). "Current" line numbers are as of
`1b7a7ba`.

**Reproduced end to end** (`scripts/failure_modes.clj`, 2026-10-08):

| § | rebuilt broken | current code |
|---|---|---|
| 4.1 | "unable to resolve symbol: a" in 3/3 seeds at D = 2048, σ = 2; `(reduce …)` gives 10, not 15 | 3/3 correct |
| 4.2 | `(get {1 2 2 3} 2)` → 1 | → 3 |
| 4.3 | **not reproducible**: unboxed, 13102 is still no member of `#{11 12}` | — |
| 4.4 | sim(0, 5005) = 0.646 | `(zero? 5005)` false; dimension 0 dead, `(+ 17 25)` = 42 |
| 4.5 | kind check at 0.5: list round trip 0/10 from σ = 2 | 10/10 to σ = 3, 9/10 at σ = 5 |
| 4.6 | C⊗n: similarity 0.50, a set reads a stored vector; ½N⊗n: similarity 0.22, spurious worlds | similarity 0.02, clean |
| 4.7 | joints as vectors: dice TV 0.44–0.50, sprinkler 0.43–0.82 | as lists: 0.000 |

## 4.2 Binding commutes

- **Fixed in** `6ab342c` (P3 task suite). Found by the `map-update` bench task.
- **Broken code**, `6ab342c^:src/vsc/core.clj:217`, in `mk-map`:
  ```clojure
  slots (mapv (fn [[k v]] [(slot k) (slot v)]) entries)]
  ```
  Keys and values went through the same `slot`. An integer value was the same
  boxed pointer as that integer used as a key, and k′⊗k = k⊗k′, so
  `(get {1 2 2 3} 2)` returned 1.
- **Fix:** `value-slot` = `(pointer :vbox x)` (current `core.clj:249`),
  with the new label `16 :vbox` and `unslot` accepting `#{:box :vbox}`.
- **Superseded** in `b3b1c87`: an entry is the directed bind k⊗Π(v), one
  substrate op (`h/bind-dir`, core.clj `entry`/`entry-value`); `value-slot`
  and `:vbox` are gone. `scripts/failure_modes.clj` now rebuilds the bug by
  swapping `entry` back to the plain bind.

## 4.4 Zero

- **Fixed in** `434b4f8` ("Calibrated decisions and a dense integer 0").
  Found by P1, E4 (lesion at f = 0.005).
- **Broken code** in `434b4f8^`:
  - `resources/vsc/hdc.py:316`, `Numbers.vec`: n = Bⁿ with no offset, so
    B⁰ is the identity of binding, a delta at index 0.
  - `src/vsc/core.clj:284`: `(defn- zero-vec [] (num-vec 0))`, so 0 is that
    delta.
  - `src/vsc/core.clj:546`: `['+ (fn [xs] (reduce bind (zero-vec) xs))]`.
    Every sum starts from the delta, so lesioning dimension 0 kills all
    arithmetic.
  - `src/vsc/core.clj:547`: `-` via `unbind` / `h/inverse`. It can pass
    through intermediates equal to B⁰.
  - `src/vsc/core.clj:630`: `:base (h/num-vec nums 1)`.
- **Fix:** n = Bⁿ⊗Z with a random unitary Z from its own stream
  (`Numbers.zspec`, `step`, `offset`). `v-add` / `v-sub` are ordered so that
  no intermediate is B⁰.
- **Correction for the paper:** the threshold-based `zero?` (5005 read as
  zero) was never in the code. Already in `bf2a8c7`, `core.clj:287` says
  `;; not theta-atom: n shares a band with 0 for every modulus dividing it`,
  and `v-zero?` goes through `eq?`. The draft's "caught in review" is
  accurate; keep it from being read as an observed failure.

## 4.5 Fixed thresholds cap robustness

- **Fixed in** `434b4f8`. Found by P1, E2 (cliffs at σ ≈ 1.5–1.7 for every D).
- **Broken code**, `434b4f8^:src/vsc/core.clj`:

  | line | code | role |
  |---|---|---|
  | 46–47 | `theta-atom 0.5`, `theta-def 0.4` | the constants |
  | 107 | `kind`: `(if (> s theta-atom) ...` | kind check, cliff at σ = √3 |
  | 133–134 | `vnil?`, `truthy?`: `(> (sim v ...) theta-atom)` | nil / truthiness |
  | 343, 351 | `global-slot`, `global-value`: `(> (sim (unbind (role :L) slot) sym) theta-def)` | global lookup, cliff at σ ≈ 1.46 |
  | 381 | `special-form`: `(when (> s theta-atom) ...)` | special-form dispatch |
  | 447 | `apply-fn` `:prim`: `(when (< s theta-atom) (fail ...))` | primitive recall |

- **Fix:** `eq?` decides exact copies and chance-level pairs by one
  similarity and everything in between by cleaning up both sides and
  comparing identities. The kind check and primitive recall compare with the
  chance floor `(chance z-recognise)` = 4.5/√D.
- **Correction (2026-10-08):** this note said `theta-atom` survived only for
  `vsc.machine`'s own datapath. Wrong when written: `prim-name` used it too
  (added by the superposition hooks, `cd708f5`, after the fix), and
  `vsc.worlds` calls it for every primitive of a superposed call. Now fixed:
  `prim-name` compares with `(chance z-recognise)`. `vsc.machine`'s `prim`
  still uses 0.5.

## New: readout bias in integer recognition (not in the paper)

- **Fixed in** `434b4f8`.
- **Broken code**, `434b4f8^:resources/vsc/hdc.py:639`, in
  `Cleanup._recognize`: `if sn > s:`. The residue readout's similarity was
  compared raw with the table's best match. The readout maximises over each
  band's candidates, so even on a random probe it reaches ≈ (4.0 ± 0.5)/√D.
  Once noise pulled an atom's similarity down to about 5/√D, the integer
  readout won and a symbol came back as a number.
- **Fix:** compare excesses over each side's chance level:
  `sn - NUM_NULL/√D > s - √(2 ln n)/√D`, with `NUM_NULL = 4.0` (measured,
  D = 1024…4096).
- **Paper:** this is the same lesson as §4.5 (decide relative to the chance
  floor, not by raw similarity). It is worth one sentence there.

## 4.1 Superposed cells: not in git

- The version with TK24 cells (the value *is* ν(L⊗a + R⊗b)) was never
  committed. Already in `bf2a8c7`, `cell` (current `core.clj:181`) goes
  through `pointer`. Hash-consing is `Memory.intern` in `hdc.py` (current
  line 575).
- **Evidence:**
  - `scripts/paper_f2.py` rebuilds the old encoding next to the pointer
    encoding and replays the environment chains of recursion. This is a
    reconstruction, not the original code.
  - The design note in the `vsc.core` namespace docstring (current
    `core.clj:28–32`).
  - Regression tests for the original symptoms:
    `test/vsc/core_test.clj:106` (`(fact 5)`) and `:116`
    (`(reduce + 0 [1 2 3 4])`).
- The draft already carries the caveat ("we did not isolate which lookup
  failed").

## 4.3 Chimeras: not in git

- Integer boxing (`slot`, current `core.clj:241`) is already in `bf2a8c7`
  (`git log -S"box"` starts there). `slot` now boxes vectors as well as
  integers in maps and sets, and `vec-slot` (`core.clj:385`) boxes integer
  elements of vectors.
- **Evidence:** `scripts/paper_f3.py` shows the 62 chimeras of ν(B¹¹ + B¹²)
  on the real `hdc.Numbers`. It reproduces the mechanism, not a failing
  program run.
- **Not reproducible end to end** on the current code: with `slot` replaced
  by identity, `(contains? #{11 12} 13102)` is still false, because the
  readout over integers that failed has been replaced.
- The paper now says that its evidence is a reconstruction.

## 4.6 and 4.7: not in git

- Both arose while vectors became positional superpositions and were fixed
  before `1b7a7ba`. 4.6: the count was first C⊗n, then ½N⊗n, in the
  vector's trace; it is now the trace the vector's row points to
  (`store_valued` in `hdc.py`). 4.7: the dice and sprinkler demos built
  their joint worlds as vectors; they now build lists.
- **Evidence:** `scripts/failure_modes.clj` rebuilds both broken variants.
  The `[5 6]` member that the paper's first draft quoted depends on which
  vectors happen to be stored; the rebuild pulls in a prelude vector.

## Open tasks

1. **Paper:** done 2026-10-08 (fix commits for 4.2, 4.4, 4.5; 4.3 marked
   as a reconstruction; the readout-bias sentence in 4.5).
2. **Reproduce end to end:** done with `scripts/failure_modes.clj` for 4.1,
   4.2, 4.4–4.7. 4.1 now reproduces its message under noise; the §4.1
   caveat (which lookup failed) still stands. 4.3 does not reproduce.
3. **Optional:** one git tag per failure mode with a single reproduction
   command in the README ("check out, run, watch it break"), so readers
   such as the TK24 authors can verify each claim.

## Commands used

```sh
git show 6ab342c -- src/vsc/core.clj
git show 434b4f8 -- src/vsc/core.clj resources/vsc/hdc.py
git show 434b4f8^:src/vsc/core.clj | rg -n "theta-atom|theta-def|zero-vec|reduce bind|h/inverse"
git show 6ab342c^:src/vsc/core.clj | rg -n "\(slot v\)"
git log -S"box" --oneline -- src resources
```
