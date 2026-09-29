# S1 results: a metacircular evaluator for the dialect

S1 is `resources/vsc/eval.clj`, an evaluator for the whole dialect written in
the dialect. It is loaded at the host level, and `(vsc-eval form env)` then
evaluates `form` one level up. The host `veval` runs the evaluator program,
and the evaluator runs the user program. Each of the evaluator's closures,
environments and global tables is a vector the host builds and inspects
through VSA operations. Two levels, or three with the λ-calculus example on
top.

## What it implements

- **Special forms:** `quote if do def defn fn let cond and or`, with the
  host's semantics, including the edge cases: `(if nil 1)` → nil, `(and)` →
  true, `(or nil false)` → false (the last value), `cond` → nil when nothing
  matches.
- **Closures** are tagged lists `(:vsc/closure name params body env)`. `&`
  varargs bind the rest of the arguments as a list, exactly as the host does
  (`()` when there are none). A named fn, `(fn self [..] ..)`, and every
  `defn` bind their own name to the closure on entry. `defn` closes over the
  empty env, like the host, so globals resolve dynamically and mutual
  recursion (`even?*`/`odd?*`) works.
- **Environments** are lists of **frames**, and a frame is a VSA map
  `{sym val …}`. A call pushes one frame with all its params, built by a
  single `hash-map` call. A `let` pushes an empty frame and binds into it one
  name at a time, which keeps `let` sequential. When a frame reaches
  `vsc-frame-size` (10) names, a new frame is opened. Lookup walks the frames
  innermost first, with one `contains?` per frame.
- **Globals** use the same representation: the frame list `vsc-globals`
  belongs to the level. `def`/`defn` inside `vsc-eval` bind there and never
  touch the host's F memory. The test checks this: after `(s1 '(def x 1))`,
  the host can't resolve `x`. When a name is in neither the env nor the
  globals, the host's globals are consulted through `(eval sym)`. That is how
  primitives are reached, and it is the only way host state leaks into the
  level.
- **Application:** S1 closures are applied by S1 itself. Every other value
  (primitives, keywords, maps, sets, and the host's own closures) is handed
  to the host `apply`. Three primitives are intercepted, because they must
  see this level's closures and globals: `apply` (spreads its arguments,
  then applies at S1), `eval` (evaluates at S1 in the empty env) and `fn?`
  (true for S1 closures).
- **The prelude is loaded through `vsc-eval`**, so `map`, `filter`,
  `reduce`, `comp`, `partial` and the others are S1 closures in S1's global
  frames, and they shadow the host's versions. The test asserts
  `(first map)` = `:vsc/closure` at S1.

No core change was needed. `symbol?`, `list?`, `contains?` and `=` on
symbols, `fn?`, `hash-map`, `eval` and `apply` were enough. `hdc.py`,
`hdc.clj` and `core.clj` are untouched.

### Why frames of VSA maps rather than an a-list

Both were implemented and measured. An a-list `((sym val) …)` has no
capacity limit, and extending it is a single `cons`. However, at S1 every
lookup is an *interpreted* walk. Each binding it passes costs a
dialect-level `cond` with `empty?`, `first`, `first` and `=`, roughly six
host eval steps. Every reference to a primitive walks *all* enclosing
bindings plus every global before it falls through to the host. A frame is
searched by one `contains?`, which is one unbind and one cleanup inside the
host primitive, with no interpreted steps at all.

The capacity limit of a single map (~21 entries at D=2048) applies only per
frame, and the chain lifts it. The remaining limits are that a single call
can bind at most ~21 params, and membership tests carry the host maps'
~4.5σ false-positive floor (`noise-floor` in core.clj). On `(fact 2)` the
a-list version and the first frame version cost the same (43.1k vs 42.9k
cleanups-by-part). Building each call frame with one `hash-map` instead of a
chain of `assoc`s, and resolving through frames directly, then cut `(* 3 2)`
from 21.7k to 18.3k parts (−16%). The frame design is also the VSA-native
one: an environment is a record, not a linked list walked by the
interpreter.

## Differential tests

`test/vsc/metacircular_test.clj`. Each form is evaluated by the host and by
`(vsc-eval (quote form) ())`, and the decoded results must be equal. Each
deftest starts from a fresh machine: `init!`, eval.clj, the prelude through
S1, and the programs defined at both levels. It starts fresh because M only
grows, and cleanup cost grows with it.

| deftest | forms | runs in |
|---|---|---|
| `s1-smoke` | 8 forms + closure check: data, map literal, set call, let, or, varargs, `(fact 2)`, `(map inc [1])` | default suite (~1.5 min) |
| `s1-lisp-and-clojure-data` | 33 forms: Lisp 1.5 elementary functions, vectors, nested maps, sets, keyword/map/set calls, assoc/dissoc/contains?, order-free `=`, count/nth/conj/seq/next, predicates, arithmetic; keys by explaining away | `:slow` |
| `s1-special-forms` | 22 forms: all special forms incl. edge cases, `&` varargs, curried closures, named recursive fn, `eval` of built code, `fn?`; `def` staying inside the level, redefinition | `:slow` |
| `s1-programs` | `(fact 4)`, `(fib 5)`, mutual recursion `[(even?* 3) (odd?* 3)]` | `:slow` |
| `s1-prelude` | `map`, `filter`+`range`, `reduce`, `reverse`, `concat`, `(map :k …)`, `comp`, `partial`, `update`, `apply`, `*`, all defined *inside* S1 | `:slow` |
| `s1-three-levels` | `examples/metacircular.clj`'s `m-eval`/`m-apply` defined in S1, running the curried λ program: host → S1 → λ-interpreter | `:slow` |

```
clojure -M:jvm:test      # core tests + s1-smoke; the :slow set is excluded
Ran 9 tests containing 112 assertions.
0 failures, 0 errors.
real	1m22.657s

clojure -M:jvm:test-s1   # the whole S1 namespace
Ran 6 tests containing 85 assertions.
0 failures, 0 errors.
real	22m25.460s
```

The `:test` alias now passes `-e :slow`, and the new `:test-s1` alias runs
`vsc.metacircular-test` in full. `VSC_S1_TIMING=1` prints the S1 wall time
and |M| per form.

Left out as host-only: the out-of-range integer error (it is raised by the
reader when the form is encoded, before any evaluator runs) and the
`degrade` robustness tests (the noise is drawn from the substrate's RNG,
whose state differs between runs). The 400000-scale arithmetic and the
`(= 0 5005)` family are covered by `(zero? 5005)`, and the rest of that
group was left out for time only.

The recursion example of `metacircular.clj` (4+3+2+1 by self-application)
was also left out for time. The curried example alone takes 250–460 s at S1.

## Slowdown per level

`bench/s1_overhead.clj`: the same form at the host and at S1, each on a
fresh machine with the same |M| at the start (~1500 traces). `veval` counts
host evaluation steps. `part` counts M(role ⊘ M(p)) cleanups (every
first/rest), and `cleanup+recognize` counts the remaining cleanup-memory
probes. "new traces" is the growth of M.

| form | level | wall s | veval | part | cleanup+recognize | new traces |
|---|---|---|---|---|---|---|
| `(fact 2)` | host | 0.74 | 68 | 461 | 338 | 21 |
| | s1 | 57.7 | 4372 | 36158 | 26153 | 420 |
| `(fact 3)` | host | 1.19 | 111 | 770 | 556 | 29 |
| | s1 | 78.3 | 7173 | 59294 | 42908 | 582 |
| `(fact 4)` | host | 1.51 | 206 | 1471 | 1042 | 44 |
| | s1 | 162.5 | 13338 | 110146 | 79771 | 943 |
| `(fact 5)` | host | 5.81 | 535 | 3936 | 2734 | 92 |
| | s1 | 716.0 | 34641 | 285720 | 207120 | 2226 |
| `(fib 3)` | host | 0.60 | 76 | 457 | 373 | 11 |
| | s1 | 66.4 | 4901 | 40076 | 29060 | 297 |
| `(fib 4)` | host | 1.50 | 139 | 842 | 683 | 13 |
| | s1 | 114.4 | 8996 | 73543 | 53336 | 348 |
| `(fib 5)` | host | 2.05 | 236 | 1433 | 1160 | 16 |
| | s1 | 171.7 | 15304 | 125075 | 90724 | 400 |
| `(fib 6)` | host | 4.36 | 396 | 2409 | 1947 | 18 |
| | s1 | 411.0 | 25707 | 210074 | 152388 | 452 |
| `(countdown 8)` | host | 0.89 | 80 | 503 | 388 | 22 |
| | s1 | 84.3 | 5092 | 42032 | 30425 | 405 |
| λ curried, `m-eval` | host (2 levels) | 3.58 | 243 | 1840 | 1390 | 70 |
| | s1 (3 levels) | 462.0 | 15609 | 127975 | 92844 | 1127 |

S1 / host ratios:

| form | wall | veval | part | new traces |
|---|---|---|---|---|
| fact 2 / 3 / 4 / 5 | 78 / 66 / 108 / 123× | 64 / 65 / 65 / 65× | 78 / 77 / 75 / 73× | 20 / 20 / 21 / 24× |
| fib 3 / 4 / 5 / 6 | 111 / 76 / 84 / 94× | 64 / 65 / 65 / 65× | 88 / 87 / 87 / 87× | 27 / 27 / 25 / 25× |
| countdown 1 / 2 / 4 / 8 | 48 / 85 / 76 / 95× | 61 / 62 / 63 / 64× | 88 / 86 / 84 / 84× | 22 / 21 / 20 / 18× |
| λ curried (3 vs 2 levels) | 129× | 64× | 70× | 16× |

Readings:

- **One level costs a constant ~65 host steps per interpreted step.** The
  marginal ratio from `(fact 4)` to `(fact 5)` is
  (34641 − 13338) / (535 − 206) = 64.7, and every form lands at 61–65×. It
  does not depend on the program, and it is the same for the third level
  (λ-interpreter under S1 vs under the host: 64×). A tower of k interpreted
  levels therefore costs ~65^(k−1) steps.
- **Cleanups grow a bit faster, 73–88×.** A host step at the S1 level
  touches more structure: it re-reads the evaluator's code cells, and every
  global reference in the evaluator walks the host env of the evaluator
  function it occurs in, three cleanups per binding.
- **Wall time is 66–130×.** Per cleanup it is the same at both levels
  (~1.5–2.5 ms at |M| ≈ 2000, D = 2048). The spread comes from |M| growing
  during a run: a cleanup is a scan over M, and S1 grows M 16–27× faster
  (every argument list, frame and closure record is a new trace). This is
  why the longest runs (`(fact 5)`, λ) show the largest wall-time ratios:
  the tower amplifies the cost of the missing GC.
- Absolute costs: at S1 one closure call costs ~7–10 s. `(fact 5)` takes
  12 min, and the whole differential suite 22–28 min.

Where S1's own steps go, for `(* 3 2)` (3 calls; parts per evaluator
function): `vsc-eval` 27%, variable lookup (`vsc-lookup` +
`vsc-find-global`) 29%, `vsc-evlis` 13%, `vsc-bind-params` 11%,
`vsc-apply` 6%. A primitive reference is the most expensive thing an S1
program does. It is resolved by probing every env frame and every global
frame (misses), then the host's F via `(eval sym)`. Resolving such
references once, at `fn` time, would remove most of that, at the cost of
late binding for redefined primitives.

## Recursion depth and stack

`bench/s1_stack.clj` runs `(countdown n)` in a thread with a fixed small
stack. It gallops up from n = 1 and then bisects for the largest n that
completes.

| stack | host depth | S1 depth | host / S1 |
|---|---|---|---|
| 256 KiB | 416 | 9 | 46.2 |
| 1 MiB (partial) | 1024 ≤ d < 1280 | not reached | |

At 256 KiB a host-level dialect call costs ~630 bytes of JVM stack, and an
S1-level call costs ~28 KiB, which is 46× more. That is below the 65× step
ratio, because many of the host steps an S1 call makes return before the
next one starts. They cost time but no depth. With the default `-Xss512m`,
that extrapolates to roughly 18k nested S1 calls (vs. ~800k at the host),
if the stack need stays linear. The 1 MiB row suggests it is slightly
sublinear at the host (1024–1279 rather than 4 × 416 = 1664). A fixed
per-thread overhead would explain that too.

The sweep over 256 KiB and 1 MiB was stopped after 14 hours. Every probe
grows M, nothing collects it, and so each probe is slower than the last.
The host bisection at 1 MiB had reached 1024 ≤ d < 1280, and S1 at 1 MiB
had not started. The script now runs only the 256 KiB point. Another point
needs a fresh `init!` per probe and a time limit per probe.

The sampled JVM stack depth in `bench/s1_overhead.clj` gives the host
slope: 83, 118, 188, 328 frames for countdown 1, 2, 4, 8, i.e. **35 JVM
frames per dialect-level call** at the host. The S1 samples saturate at
1005, because the JVM truncates `getStackTrace` at
`MaxJavaStackTraceDepth` = 1024 frames. S1 already needs more than that for
`(countdown 1)`, so the S1 column there is only a lower bound, and the
thread-stack bisection above is the real measurement.

The stack need per S1 call is large for a structural reason. One S1 call to
a user fn nests `vsc-eval` → `vsc-apply` → `vsc-apply-closure` →
`vsc-eval-body` → `vsc-eval` (the `if`) → `vsc-eval-special` → `vsc-eval`
(the tail call) → `vsc-apply` …. That is about 7–10 dialect-level calls,
each costing ~35 JVM frames at the host, plus the host's own recursion
through the dialect `cond`/`if`/`let` inside each of them.

## Tail calls: what S1 can and cannot do

S1 cannot give its programs proper tail calls, or any recursion depth
independent of the host stack:

- `vsc-eval` is a dialect function evaluated by the host `veval`, and the
  host evaluator recurses on the JVM stack for every nested evaluation,
  tail positions included. A tail call in the user program is a
  `vsc-eval` → `vsc-apply` → `vsc-eval-body` → `vsc-eval` chain in the
  evaluator, and every link of that chain is a host call that stays on the
  JVM stack until the callee returns.
- Rewriting S1 as a trampoline or in CPS does not help. The dialect has no
  loop construct (`loop`/`recur` are not supported), so the trampoline's
  driving loop would itself be a recursive dialect function, and that is
  host recursion again. S1 can only *move* the recursion from one of its
  functions to another. It cannot remove it, because iteration does not
  exist at the level it runs on.
- What S1 *can* do: keep its own state as explicit data, which it already
  does (env frames, globals, closures). And it could make the continuation
  explicit data as well (a CEK-style S1), so that the *only* recursion left
  is one host loop. That loop is exactly what the dialect can't express and
  what S2 provides: a host `while` over a machine state whose continuation
  is a list of frames in vector memory, so recursion depth is bounded by
  memory, not by `-Xss`.

So S1 inherits the host's stack limit and divides it by ~46 (measured
above). It also keeps the `-Xss512m` requirement. Both
motivate S2.

## Semantics that could not be reproduced exactly

1. **Closures are visible as data.** An S1 closure is a list, so `list?`,
   `coll?`, `seq`, `count`, `first` and `=` see
   `(:vsc/closure name params body env)`, and the printer shows it as that
   list instead of `#fn (fn …)`. `fn?` is intercepted and agrees with the
   host. A distinct kind would need the host to hand out closure-labelled
   cells for arbitrary data (a new primitive), which S1 did not need for any
   test. The differential tests never return a closure.
2. **Host closures leak in through the global fallback.** Any host global
   that S1 does not redefine resolves to the host's value. A host *closure*
   reached that way (for instance the host's `vsc-eval` itself) runs at the
   host level, and it cannot call back an S1 closure: passing one to it
   fails with "not a function", because to the host an S1 closure is a list.
   Only `apply` and `eval`, the primitives that take code, are intercepted.
   Loading the prelude through S1 keeps all higher-order library functions
   at S1.
3. **Error messages differ.** The dialect has no `throw`, so arity errors
   are raised by applying the message string, which makes the host fail
   with `not a function: "too few arguments for params"`. Unresolved symbols
   still produce the host's own `unable to resolve symbol: x`, because
   resolution ends in `(eval sym)`.
4. **Capacity.** A single call binds at most one map's worth of params
   (~21 at D=2048; the host fails loudly, `map of n exceeds the capacity`).
   The host evaluator has no per-call limit.
5. **Duplicate param names** (`(fn [x x] x)`): the first binding wins at
   S1 (keys are consed in reverse, and `hash-map` lets later keys win), the
   last one at the host.
6. **Membership is probabilistic.** Frame and global lookups use the same
   `contains?` as host maps. A hit needs the probe share to clear the
   ~4.5σ noise floor, and ambiguous cases fall back to exact explaining
   away. No misresolution was observed in the test and benchmark runs, but
   it is a bounded-error mechanism, not an exact one.
7. **Only one mutable cell per level.** `vsc-globals` is a host global,
   rebound with the host `def`. An S1 running inside S1 would get its
   `vsc-globals` as an ordinary global of the outer S1, so the levels stay
   separate. This was not run, because of the cost: ~65² ≈ 4000× per step.

## Files

- `resources/vsc/eval.clj`: the evaluator (dialect)
- `test/vsc/metacircular_test.clj`: the differential tests
- `bench/s1_overhead.clj`, `bench/s1_stack.clj`: the measurements above
- `deps.edn`: `:test` excludes `:slow`; `:test-s1` runs the full S1 suite
