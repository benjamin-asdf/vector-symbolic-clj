# P3 task suite

Small programs in the dialect, used to measure task accuracy under noise
(see `docs/PLAN.md`, P3). One file per task. The decoded value of the last
form is the result, and the metric is an exact match with `expected.edn`.

```sh
clojure -M:jvm:bench                 # every task: ok/FAIL, timing table
clojure -M:jvm:bench fib bst         # just these
clojure -M:jvm:bench --dim 4096      # at another dimension
clojure -J-Xmx256m -M bench/reference/gen.clj   # regenerate expected.edn (no Python)
```

`test/vsc/bench_test.clj` runs every task at noise 0 as part of
`clojure -M:jvm:test`.

## Tasks

Measured at D=2048, seed 42, one fresh machine per task. **ms** is the wall
time of the program alone (not of `init!`). It was measured while other jobs
loaded the machine, so unloaded runs are about 1.5× faster. **calls** is the
number of calls into the Python substrate, which does not depend on load.
**traces** is the growth of M during the run.

| task | size | expected | stresses | ms | calls | traces |
|---|---|---|---|---|---|---|
| `fact` | n = 5 | `120` | recursion depth, env chains; `*` is repeated addition | 2309 | 11 856 | 113 |
| `fib` | n = 8 | `21` | tree recursion, many short-lived env cells | 3697 | 20 462 | 50 |
| `map-filter-reduce` | (range 10) | `[(0 2 … 18) (0 2 4 6 8) 25]` | list walking, closures and a set as functions | 3708 | 17 856 | 250 |
| `map-update-5` | n = 5 | `{0 3, 1 5, … 4 11}` | map rebuild on every assoc/update | 1943 | 9 449 | 184 |
| `map-update-10` | n = 10 | `{0 3, … 9 21}` | the same | 3679 | 19 794 | 289 |
| `map-update-12` | n = 12 | `{0 3, … 11 25}` | the same, largest n exact at D=2048 | 5453 | 24 653 | 331 |
| `graph-reach-8` | 8 nodes | `#{0 1 2 3 4}` | maps of sets, membership, set enumeration | 1582 | 8 194 | 176 |
| `graph-reach-12` | 12 nodes | `#{0 1 2 3 5 6 9 10}` | the same | 3112 | 13 918 | 255 |
| `graph-reach-20` | 20 nodes | `#{0 … 16}` | the same; the graph map holds 20 entries | 7124 | 28 793 | 450 |
| `insertion-sort` | 7 values in 0..6 | `(0 1 2 3 4 5 6)` | long runs, many traces | 4826 | 23 040 | 250 |
| `metacircular` | k = 2 | `[7 3]` | program as data, nested closures, VSA-map envs | 6983 | 30 578 | 339 |
| `bst` | 5 keys, depth 4 | `[(0 1 2 3 4) 4 (true false)]` | deep nesting: BST as nested maps | 7888 | 28 769 | 389 |

About 98% of the wall time is spent inside the substrate, and `part`
(car/cdr: two scans over M) takes most of it. The cost of a task is roughly
the number of interpreted steps times |M|.

## Where the sizes live

Each task's parameters are `(def …)` forms at the top of its file: `n`, `k`,
`xs`, `ks`/`probes`, or the `graph` literal. The variants
(`map-update-*`, `graph-reach-*`) are separate files that differ only in those
forms. To change a size, edit the `def` and rerun `gen.clj`. The reference
programs read their parameters from the bench files, so the two cannot drift
apart.

## Expected values

`expected.edn` maps `task -> {:file :expected :notes}`. It is generated, not
written by hand. `bench/reference/<name>.clj` evaluates to a real-Clojure
function of the bench file's `def` parameters. The reference programs are
written in idiomatic host Clojure and deliberately use other algorithms: `<`
and `sort`, BFS with `loop`/`recur`, a vector-node tree, and host lambdas for
the λ-programs. A shared mistake is therefore unlikely.

For the experiment runner: `(vsc.bench/tasks)` returns that map, and
`(vsc.bench/run-task 'fib {:dim 4096 :seed 7})` runs one task on a fresh
machine. It returns `:value :ok? :ms :traces :calls :calls-by-fn
:substrate-ms`.

## Notes per task

- **lt? is linear in the values.** There is no `<`, so `insertion-sort` and
  `bst` compare by decrementing both sides at once:
  `(lt? a b)` takes min(a, b) + 1 recursive calls. The values are kept
  small for that reason.
- **map-update: the practical capacity is below the enforced one.** The
  enforced limit (21 entries at D=2048) is a single-lookup margin. One
  `get` on an n-entry map has signal 1/√(3n) ± ~0.025 (0.129 at n = 20),
  while the best of ~1000 random rows in M reaches ~0.07–0.08. So roughly
  one lookup in 70 fails at n = 20, and map-update does ~2n² lookups.
  Measured at noise 0, D=2048:

  | n | seeds exact | runtime |
  |---|---|---|
  | 10 | 1/1 | 3.4 s |
  | 12 | 4/4 (seeds 42, 7, 1, 3) | ~4 s |
  | 16 | 2/4 (fails on 42 and 1) | ~6 s |
  | 20 | 0/3 (1–2 wrong entries) | ~9 s |
  | 20 at D=4096 | 1/1 (an earlier variant with `*` values) | — |

  The suite therefore stops at n = 12. `graph-reach-20` does hold a
  20-entry map, but it reads each entry only once or twice.
- **map-update also exercises a fixed bug.** Values collide with keys
  (2 is both). Before the fix, `(get {1 2 2 3} 2)` returned 1 (see the
  root README, "Where the paper's scheme needed changes").
- **metacircular** is `examples/metacircular.clj` with the recursive sum
  parameterised by `k`. The result is a vector, so both the curried and the
  self-applied program are checked.
