# W results: superposition ("many worlds") programming

Code: `src/vsc/worlds.clj` (the API, the hooks, the dialect primitives),
the `Proj` memory, `ProjCleanup` and `readout` in `resources/vsc/hdc.py`,
the hook points in `src/vsc/core.clj` (marked `[W1 hook]`), demos in
`examples/worlds/`, tests in `test/vsc/worlds_test.clj` and
`test/vsc/worlds_demos_test.clj`. Design and the W0 measurements behind it:
`docs/W0-superposition-design.md`.

Curves (W3), reproducible:

| figure | source | run |
|---|---|---|
| `figures/w-readout.png` | `scripts/w_readout.py` (Python only) | `OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/w_readout.py` (`--quick`: D ≤ 2048, 4 trials) |
| `figures/w-lift.png` | `specs/w-lift.edn` | `clojure -M:jvm:exp specs/w-lift.edn`, about 15 min |
| `figures/w-bfs.png` | `specs/w-bfs.edn` | about 35 min |
| `figures/w-demos.png` | `specs/w-demos.edn` | about 7 min |

The runner CSVs go to `out/`, and `.venv/bin/python scripts/plot_worlds.py`
turns them into the figures and prints the tables quoted below. The task
generators are in `src/vsc/experiments/worlds.clj`. Every grid point has 5
seeds (readout: 20 trials).

## 1. What it is

A value can be a weighted superposition of ordinary values (its worlds):
x = Σ wᵢ vᵢ with wᵢ ≥ 0 and Σ wᵢ = 1, and it is still **one vector**. The
weights are the linear coefficients themselves, not squared amplitudes, so
they read directly as probabilities. There is no interference: weights
never go negative.

`(vsc.worlds/init! opts)` builds a machine with the `:proj` memory, the
hooks, the primitives and `for-worlds`. On single-world programs `:proj` is
the codebook, bit for bit (a test fingerprints both), so everything that
ran before runs the same.

```clojure
(require '[vsc.worlds :as w])
(w/init! {:dim 4096})
(w/run '(+ (amb 1 2) 10))                      ;; => #worlds {11 0.5, 12 0.5}
(w/run-worlds '(first (amb '(a b) '(c d))))    ;; => {a 0.5, c 0.5}
```

### The memory that makes it work

`Proj` (W0 §6): find the support S = {j : |kⱼ·p|/|p| > τ} with
τ = (√(2 ln N) + margin)/√D, then solve least squares on S, clip the
coefficients at 0 and normalise them. Every trace records its norm before it
was normalised (the field gain), so `deref` returns each world at its true
weight at any depth. With |S| ≤ 1 it returns exactly what the codebook
returns. `readout` (behind `worlds`) is orthogonal matching pursuit over M
and the integer candidates [−1024, 1024]. It stops at chance level.

## 2. The API, as implemented

Host side (`vsc.worlds`): `init!`, `run`, `run-string`, `eval-form`,
`run-worlds` (a form's worlds as `{value weight}`), `worlds`, `weight`,
`tv-distance`, `capacity`, `readout`. Superpositions print as
`#worlds {value weight}`.

Dialect side. The dialect has no floats, so weights inside it are integers:
relative weights going in, per mille coming out.

| form | semantics |
|---|---|
| `(superpose {v w, …})` | Σ (w/Σw)·v. Integer relative weights. Equal worlds merge, and their weights add. |
| `(amb a b …)` | uniform `superpose`. `(amb)` throws. |
| `(worlds x)` | a map {world weight‰}. **Throws** "superposition capacity exceeded" beyond `capacity` (D/100 value worlds, D/400 integer worlds), and "unreadable superposition" when the worlds found leave more than half of x's energy unexplained. |
| `(weight x v)` | weight of world v in x, in ‰; 0 if absent |
| `(collapse x)` | the heaviest world (a measurement; deterministic) |
| `(collapse x β)` | partial collapse: weights ∝ exp(β·(wᵢ − w_max)/‖w‖₂). This distorts the weights (W0 §4), which is why it is called collapse and not a soft superposition. |
| `(sample x)` | one world, drawn with p ∝ w (seeded host RNG, `:seed`) |
| `(assert p x)` | post-selection: {v ↦ w·[p v]}/Z, one call of p per world. Throws when no world survives. This is classical conditioning, not Grover search. |
| `(for-worlds [v x, u y …] body …)` | **call-time choice**: enumerates each binding in turn (later ones see one world of the earlier ones) and returns Σ w·body |
| `(support x)` | the uniform superposition over x's worlds (set semantics) |
| `(without x y)`, `(union x y)` | set difference and set union for uniform superpositions, from coefficients and norms only. y is never read out, so it may hold more worlds than `worlds` could read. |
| `(relation spec)`, `(follow r x)` | a heteroassociative relation in M, one row per key (R⊗k ↦ the uniform superposition of its values), and one lookup that finds the successors of every world of x together |

How the rest of the language treats superpositions:

- **Destructors lift linearly.** `first` and `rest` (when every world is a
  non-empty cell of one kind), environment lookup, global lookup,
  `inc`/`dec`/`+`/`-` (binding distributes over sums), and `get` with a
  superposed key. These are one memory read regardless of k.
- **Closures run once.** A `fn` applied to a superposed argument evaluates
  its body once, in an environment that holds the superposition. A
  superposed *function* applies each of its worlds.
- **Everything else enumerates.** Any other primitive with a superposed
  argument computes Σ wᵢ f(xᵢ) over the joint worlds of its arguments (the
  product distribution, capped at capacity²).
- **Constructors enumerate.** `(cons (amb 1 2) ())` = `(amb (1) (2))`: the
  worlds are proper values, and equal worlds merge. The lists, vectors,
  maps and sets built by `mk-*` all go through the `:construct` hook.
- **Superposition-aware `if`** (and `cond`/`and`/`or`, which share one
  `branch` hook). π = the weight of the truthy worlds of the test value.
  If π > 1 − ε only `then` runs, and if π < ε only `else` runs (ε = 0.02,
  so pruning is lazy evaluation). Otherwise both branches run and are mixed
  π : 1−π. If the test depends on exactly **one** superposed variable, that
  variable is **split**: each world goes to the branch its own test value
  takes. This is what makes recursion whose depth differs between worlds
  terminate: `(dbl (superpose {2 5 3 3 5 2}))` gives {4 .5, 6 .3, 10 .2}. A
  test on several superposed variables enumerates their joint worlds.
  `and`/`or` return the test value as the branch sees it (its truthy or
  falsy worlds only).

## 3. Run-time vs call-time choice: the decision

**Default: run-time choice.** Every occurrence of a superposed value chooses
independently:

```clojure
(let [x (amb 1 2)] (+ x x))            ;; => #worlds {2 0.25, 3 0.5, 4 0.25}
(let [x (amb 1 2)] (= x x))            ;; => #worlds {true 0.5, false 0.5}
(for-worlds [x (amb 1 2)] (+ x x))     ;; => #worlds {2 0.5, 4 0.5}   call-time
```

This is not a policy the code picks at each site. It follows from the
representation, and the code only makes it consistent:

1. A superposition is one vector with no world labels. x ⊛ x for
   x = Σ wᵢvᵢ is Σᵢⱼ wᵢwⱼ vᵢ⊛vⱼ, the product distribution. Nothing in the
   vector says that the two occurrences are the same draw. So `+` (native:
   binding) and the enumerating primitives (`product` over their
   arguments' worlds) both give the product.
2. Call-time choice by default would need world identity on every value,
   which means per-world tags, or the direct-sum encoding of W0 §2 (each
   world in its own band of Fourier bins). That costs √k of SNR per world
   and fixes k in advance, and it gives up what makes superposition worth
   having: a function body evaluated **once** for all worlds (the parallel
   map below runs its recursion once, not twice).
3. So call-time choice is **explicit**: `for-worlds` enumerates, and it is
   exact (the enumeration is on the host; each body sees one world).
4. The split `if` restores the correlation you need most, and only that
   one: within a branch, the tested variable holds only the worlds that
   took that branch. Correlation with *other* variables is not tracked. The
   sprinkler demo's `rain-given-wet-naive` shows the failure: written with
   `let` instead of `for-worlds`, P(rain | wet) comes out as the prior 0.2
   (a test asserts this), not 0.363.

Rule of thumb: `let`/`fn` are right for independent draws (dice: every
`(die)` is its own draw, and `+` convolves them in one bind). Use
`for-worlds` whenever a later expression must see the *same* world as an
earlier one (joint distributions, conditioning).

## 4. Demos (W2), and how they depend on D (`figures/w-demos.png`)

| demo | file | result | D where it is right, 5/5 seeds |
|---|---|---|---|
| BFS, one vector per level, 12 nodes | `bfs.clj` | levels [{n0} {n1 n5} {n2 n6 n9} {n3 n10}], equal to the host BFS | 1024–8192 |
| parallel map | `parallel-map.clj` | `(map inc (amb [1 2 3] [10 20 30]))` runs the recursion once: 8 worlds of ⅛ (run-time choice), every per-position marginal exact; `for-worlds` gives {(2 3 4) ½, (11 21 31) ½}, TV ≤ 0.0003 | 1024–8192 |
| dice, 2d4 | `dice.clj` | the sum's 7 worlds exact to 1e-15 in **one** bind; conditioning with `for-worlds` + `assert` | 4096, 8192 (at 2048, 7 > 5 = capacity: throws) |
| dice, 2d6 | (spec only) | 11 worlds exact to 1e-15 | 8192 (throws below) |
| sprinkler network | `sprinkler.clj` | P(rain \| wet) = 0.3625 (exact 0.36255), joint within TV 0.02 | 1024–8192, TV 0.001 at D=1024, < 0.0003 above |

Where a demo fails in this grid, it **throws** (capacity exceeded). It never
returns a wrong distribution. Cost: the sprinkler query scans 2.9M rows and
takes 1.4 s at D=1024 (12.7 s at D=8192), and BFS-12 scans 3.7M rows.

## 5. Curves (W3)

### 5.1 World readout vs number of worlds vs D (`figures/w-readout.png`)

Superpositions of k worlds with weights drawn from U(0.5, 1.5) and then
normalised, 20 trials per point, in a memory laid out like the
interpreter's (1000 atoms, 1500 two-element lists). "Exact" means the
readout finds exactly the true worlds. The table gives the largest k up to
which every point passes:

| readout | D=1024 | 2048 | 4096 | 8192 | enforced capacity |
|---|---|---|---|---|---|
| pointers, read directly: P(exact) ≥ 0.9 | ≥ 64 | ≥ 64 | ≥ 64 | ≥ 64 | D/100 = 10, 20, 40, 81 |
| `(first x)`, one cell read: P(exact) ≥ 0.9 | 8 | 24 | 32 | ≥ 64 | |
| `(first x)`: mean TV < 0.1 | 12 | 24 | 48 | ≥ 64 | |
| integers, direct: P(exact) ≥ 0.9 | 5 | 5 | 5 | 5 | D/400 = 2, 5, 10, 20 |
| integers through a `let` cell | 5 | 5 | 5 | 5 | |

- Stored pointers read back perfectly well past capacity, with TV 0 up to
  k = 64 at every D: pursuit over near-orthogonal stored rows is easy. The
  binding case is a **memory step**. `(first x)` carries crosstalk from the
  other field and scales linearly with D, at about D/85 to D/128 worlds. This
  is the regime the D/100 capacity is set for, and it holds at every D
  except the 12-world point at D=1024 (P = 0.75).
- **Integer readout does not scale with D.** It is exact up to 5 worlds and
  degrades from 6 at every D from 1024 to 8192. Residue integers are
  correlated with each other, and the candidate range has 2049 of them
  (W0 §2). So D/400 is too strict at D=1024 (it allows 2, and 5 work) and
  too loose at D ≥ 4096 (it allows 10 and 20). Past 5 worlds, `worlds`
  almost always refuses through its residual check rather than lying. A
  separate check (scratch, 20 trials per point) found no silent wrong
  answers at D=2048 for k ≤ 12, and 3 of 20 at D=8192, k=6, for integers
  spread over [−1000, 1000]. Consecutive integers do better at large D
  (at D=8192, 19/20 exact at k=6, 15/20 at k=8), which is why 2d6's 11
  sums read exactly there.

### 5.2 Exactness of linear lifting vs op noise (`figures/w-lift.png`)

Five programs with three worlds (weights .5/.3/.2). Op noise σ is on every
bind/unbind/bundle after setup. Exact means the same worlds and TV < 0.05.
Cells give exact/5, with the mean TV of the exact runs in parentheses:

| program | D | σ=0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| `dbl`: split `if`, 5 levels | 1024 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| | 2048 | 5/5 (0.000) | 5/5 (0.003) | 5/5 (0.005) | 5/5 (0.011) | 5/5 (0.019) | 0/5 |
| | 4096 | 5/5 (0.000) | 5/5 (0.002) | 5/5 (0.003) | 5/5 (0.007) | 5/5 (0.011) | 0/5 |
| `len`: split `if` + `rest` | 1024 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| | 2048 | 5/5 (0.000) | 5/5 (0.005) | 5/5 (0.010) | 5/5 (0.021) | 4/5 (0.032) | 0/5 |
| | 4096 | 5/5 (0.000) | 5/5 (0.004) | 5/5 (0.008) | 5/5 (0.016) | 5/5 (0.026) | 0/5 |
| `first` of `rest`³ | 1024 | 5/5 (0.002) | 5/5 (0.008) | 5/5 (0.015) | 5/5 (0.031) | 3/5 (0.049) | 0/5 |
| | 2048 | 5/5 (0.001) | 5/5 (0.004) | 5/5 (0.009) | 5/5 (0.019) | 5/5 (0.030) | 2/5 (0.051) |
| | 4096 | 5/5 (0.000) | 5/5 (0.004) | 5/5 (0.008) | 5/5 (0.016) | 5/5 (0.024) | 3/5 (0.044) |
| 20 `inc`s | 1024 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| | 2048 | 5/5 (0.000) | 5/5 (0.002) | 5/5 (0.004) | 0/5 | 0/5 | 0/5 |
| | 4096 | 5/5 (0.000) | 5/5 (0.001) | 5/5 (0.003) | 0/5 | 0/5 | 0/5 |
| 3 nested `let`s | 1024 | 5/5 (0.000) | 5/5 (0.006) | 5/5 (0.012) | 5/5 (0.023) | 4/5 (0.035) | 1/5 (0.133) |
| | 2048 | 5/5 (0.000) | 5/5 (0.003) | 5/5 (0.005) | 5/5 (0.011) | 5/5 (0.017) | 4/5 (0.031) |
| | 4096 | 5/5 (0.000) | 5/5 (0.001) | 5/5 (0.003) | 5/5 (0.005) | 5/5 (0.008) | 5/5 (0.017) |

- **Noise-free, lifting is exact.** Integer programs are exact to about
  1e-15, and pointer programs have TV ≤ 0.002. The integer programs fail at
  D=1024 only because the integer capacity there is 2 (D/400, three worlds
  asked). They throw, although §5.1 shows that 3 would read fine.
- The weight error grows **linearly in σ** (TV ≈ 0.06–0.1·σ at D=2048 for every
  program), and it roughly halves when D doubles.
- The cliff depends on the **program's depth, not on its kind**. Five
  recursion levels and three cell reads hold to σ = 0.3. Twenty bindings
  with no cleanup in between fail at σ = 0.2, as the (1+σ²)^(−T) decay of
  W0 §2 predicts, while three `let` lookups hold to σ = 0.5.
- **How it fails.** 120 of the 122 errored runs throw ("unreadable
  superposition": most of the energy is unexplained; or a capacity check),
  and 2 hit the budget (noisy recursion that never reaches its base case at
  D=1024). The silent failures are (a) TV just over 0.05 with the right
  worlds (σ ≥ 0.3), (b) the **smallest world (0.2) lost** at σ = 0.5,
  D=1024 (`nth3`, `let3`), which is W0's "smallest worlds go first", and
  (c) 20 `inc`s at σ = 0.5, which decode to an unrecognised vector
  (`#hdv ?`), not to a wrong integer.

### 5.3 BFS vs graph size and D (`figures/w-bfs.png`)

Random digraphs with n nodes and out-degree 2, level-synchronous BFS from
`:n0` using `follow`/`support`/`without`/`union` (the program is
`bfs-program` in `src/vsc/experiments/worlds.clj`, the same as
`examples/worlds/bfs.clj`). Exact means every level is right:

| n (reachable, peak frontier) | D=1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| 10 (9, 2–3) | 5/5 | 5/5 | 5/5 | 5/5 |
| 20 (18, 4–6) | 4/5 | 5/5 | 5/5 | 5/5 |
| 40 (34, ≤ 11) | 0/5 | 5/5 | 5/5 | 5/5 |
| 80 (61–67, 10–15) | 0/5 | 0/5 | 5/5 | 4/5 |
| 160 | 0/5 | 0/5 | 0/5 | 0/5 (time budget) |

- The limiting quantity is not the frontier but the raw successor set
  `(follow graph frontier)` before `without` removes the visited nodes,
  about 2× the frontier. Every failure at D ≤ 4096 is a **throw** of
  "capacity exceeded: 22 value worlds, capacity 20" and the like. So the
  rule is D ≳ 100·(peak successor set), consistent with W0's
  D ≳ 4·d·|F|·(√(2 ln n)+1)².
- One silent error: D=8192, n=80, seed 2 put 1 of 67 nodes at the wrong
  level (node accuracy 0.985). It still fails after the §7 readout fix,
  and the same graph is exact at D=4096. The cause is not isolated. The
  likely suspect is `without`/`union`: they decide membership from a
  coefficient against a threshold derived from a norm (1/(2n), with n
  estimated from ‖y‖²), with no readout, so they have no refusal path.
- n = 160 at D=8192 hit the 60 s per-run budget (the whole run takes
  107 s). This is cost, not correctness: a level costs one lookup, but that
  lookup scans all of M (20M rows scanned for n = 80).

## 6. Limits, honestly

1. **Capacity is small and linear in D.** About D/100 worlds per memory
   step, 5 integer worlds at any D, and joint enumeration capped at
   capacity². The exponential "many worlds" claim would need factored
   representations and a resonator search, which were not built.
2. **Run-time choice is the default semantics** (§3). It is correct for
   independent draws. It is silently *different* from what a Clojure
   programmer expects for `(let [x (amb 1 2)] (= x x))`. `for-worlds` is
   the fix, and the programmer has to know when to use it.
3. **No speedup over enumeration.** A superposed step costs a scan of M
   (O(N·D)), plus the readout (O(k·N·D)) wherever a primitive enumerates.
   Function bodies run once for all worlds, but the sprinkler query takes
   1.4–13 s where host enumeration takes microseconds.
4. **Refusal is the policy, and it mostly holds.** `worlds` throws past
   capacity or on an unexplained residual, and every demo failure in the
   grids is a throw. The exceptions are measured: wide-range integers at
   D=8192, k=6 (3/20 silent), one BFS run at D=8192 (the norm-based set
   operations), and under op noise the smallest world can vanish without a
   throw (§5.2).
5. **Op noise.** Lifting survives σ ≤ 0.3 at D ≥ 2048 for recursion, cell
   reads and `let` chains. Long binding chains (20 `inc`s) fail at
   σ ≥ 0.2, since the signal decays as (1+σ²)^(−T) and nothing cleans up
   between binds.
6. **Weights inside the dialect are integers** (‰ out, relative weights
   in), because the dialect has no floats.
7. **Correlations are tracked only through the split `if`,** and only for
   a single superposed variable per test. Several superposed variables in
   one test force enumeration. Correlation with variables not in the test
   is lost (the naive sprinkler).
8. **Superposed machine states (W2.5) were not attempted.** W0 §6 risk 2
   applies: stepping Σ wᵢsᵢ through a binary primitive mixes worlds unless
   the states are direct-sum encoded.

## 7. Changes made while producing the curves

- Least-squares round-off left phantom worlds of weight ~1e-9 (a candidate
  the pursuit tried and the solve zeroed). They made 2 of 5 noise-free
  `inc20` runs at D=2048 fail the "same worlds" check. `readout` now drops
  worlds below 1e-6 and renormalises. This is a numerical floor, not the
  ε = 0.02 branch pruning, which would wrongly drop real small worlds like
  the sprinkler's 0.0198 joint world. `init!`'s docstring had claimed that
  worlds below ε were dropped. It now says that only branches are.
  The w-lift grid was rerun after this fix. The demo and BFS grids ran
  before it: their failures are throws, which the floor cannot change, and
  the one silent BFS failure was rerun after it, with the same result.
  `scripts/w_readout.py` scores the raw `hdc.readout`, without the floor,
  so its "exact" fractions are, if anything, a lower bound.
- Native runner tasks had no budget. One noisy recursion (dbl, D=1024,
  σ=0.5) never reached its base case and hung the w-lift grid. The worlds
  task generators now set their own budget (2M ops, 60k rows, 60 s).

## 8. Core hook points (for merging)

All are marked `[W1 hook]` in `src/vsc/core.clj`. Without `vsc.worlds`
every hook is nil and the plain code runs:

- `set-hooks!` and `hook`: `{:apply :branch :construct :decode :global :special}`, reset by every `init!`
- `cell`: under the `:construct` hook, the fields are ν-normalised and the trace is stored unnormalised, so `:proj` records the gain
- `mk-list` / `mk-vec` / `mk-map` / `mk-set` go through `construct` (split into `mk-map-plain` and `mk-set-plain`)
- `decode` → `:decode` hook, with `decode-plain`. `num-value` reads superposed integers against `int-print-range` (±1024), so the plain printer shows `#worlds` instead of residue chimeras
- `global-value` → the `:global` hook (the global's value read without L-field crosstalk)
- `special-form` knows the labels added by `add-special!` (`:extra-specials`), and `eval-special` falls through to the `:special` hook
- `branch`: one hook for `if`, `cond`, `and`, `or` (these were rewritten on top of it)
- `apply-fn` → the `:apply` hook, with `apply-plain`
- `init!`: the default `:memory` comes from `$VSC_MEMORY` (default `:codebook`), and `:proj` is documented
- new public fns: `add-primitive!`, `prim-name`, `add-special!`

`hdc.py` gained `Proj`, `ProjCleanup`, `readout`, `int_worlds`,
`int_dots`, `more_ints`, `lincomb`, `coef`, `associate` and `follow`, and `hdc.clj` has the bridges. The `mhn` backend
and the core thresholds are untouched.
