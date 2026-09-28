# S2 results: a vector CEK machine, no host `veval`

Code: `src/vsc/machine.clj` (machine, microcode, host loop),
`resources/vsc/vsc_machine.py` (the machine's memories and datapath),
`test/vsc/machine_test.clj` (differential and deep-recursion tests),
`src/vsc/machine_report.clj` (every number below:
`clojure -M:jvm -m vsc.machine-report out`).

The machine reuses the core's data layer unchanged: cells, the hash-consed
item memory M, the integer readout, maps and sets, the global memory F and
the primitives. No line of `core.clj`/`hdc.py`/`hdc.clj` was edited;
`vsc.machine` reaches a few private core functions through their vars.

## 1. State layout

The whole machine state is one pointer s in a working memory W:

```
W: s ↦ ν(C⊗control + E⊗env + K⊗kont + M⊗mode)
```

| mode | control | E field |
|---|---|---|
| EVAL | an expression | the environment (the core's a-list) |
| RET | a value, returned to the top frame | (unused) |
| APPLY | a function value | its argument list |
| HALT / ERROR | the result / the offending value | |

The continuation is a linked list of frames in W. A frame is a record
ν(T⊗type + A⊗a + B⊗b + E⊗env + N⊗next):

| frame | a | b | pushed by |
|---|---|---|---|
| `halt-fr` | | | boot |
| `if-fr` | (then else) | | `if` |
| `do-fr` | remaining body forms | | a body with >1 form left |
| `def-fr` | the symbol | | `def` |
| `let-fr` | remaining bindings | body | `let` |
| `cond-fr` | (then t₂ x₂ …) | | `cond` |
| `and-fr` / `or-fr` | remaining forms | | `and` / `or` |
| `args-fr` | values so far (reversed list in W) | expressions left | a call, `[…]`, `{…}`, `#{…}` |

A body's last form, a taken branch, a `cond` consequent, and the last form
of `and`/`or` are evaluated without pushing a frame. So **every tail call
runs in constant continuation space**, and the host stack never grows
(there is no host recursion).

W is a small heteroassociative memory whose rows can be freed. States,
frames and argument lists live there and are collected by mark-and-sweep
from the current state whenever W has doubled. Everything the program
builds (environments, closures, lists) goes to M exactly as with the host
evaluator: `traces+` below is identical for both.

**Deviation from the plan, justified.** The plan's key was a product
kind ⊗ head ⊗ mode ⊗ frame-type. A product cannot express "don't care",
and most rules care about only two of the four. The key here is a bundle of
role-bound features:

```
key(s) = ν( FM⊗mode + FK⊗kind(C) + FH⊗kind(C)⊗class(car C)
          + FI⊗kind(C)⊗class(C) + FF⊗type(top frame) )
```

A rule's key names only the features it tests, for example
ν(FM⊗EVAL + FK⊗k-list + FH⊗k-list⊗`if`). The state key always has all five
features, so a rule with k features of which j match scores j/√(5k). A
fully matching rule therefore scores 0.447, 0.632 or 0.775 for k = 1, 2, 3,
and **the most specific matching rule wins by nearest-neighbour alone**.
Less specific keys act as defaults: `{:mode EVAL}` is "self-evaluating",
`{:mode APPLY}` is "not a function". The fifth feature, FI (the class of C
itself), exists because `apply` and `eval` are primitives that must
re-enter evaluation, so the machine intercepts them by identity.
Binding FH/FI with kind(C) stops, for example, an integer whose garbage
`car` happens to clean up to `if` from matching the `if` rule.

## 2. The ISA (9 instructions)

Operands are vectors in an operand memory: 13 registers
`S C E K M T A B N X Y Z KEY`, opcodes, spaces, modes, frame types, kind
atoms, data constants, and branch targets. A register's name vector
doubles as the role under which the register is stored in a record, so
`REC S W C C E E K K M M` reads "pack C, E, K, M into a new state".

| instruction | effect |
|---|---|
| `FIELD d x r s` | d ← clean_s(r ⊘ deref(x)): read a field of a record or cell |
| `CLEAN d x s` | d ← clean_s(x); s ∈ VAL, CONST, KIND, CLASS, F |
| `BIND d x y` | d ← x ⊗ y |
| `UNBIND d x y` | d ← x ⊘ y |
| `REC d s r₁ x₁ …` | d ← store_s(ν(Σ rᵢ⊗xᵢ)); s ∈ W (fresh pointer), LIST/FN (hash-consed pointer in M), NONE (the bare trace) |
| `MOV d x` | d ← x |
| `JMPEQ x y l` | if sim(x, y) > 0.6, jump to l |
| `PRIM d f x` | d ← host primitive f applied to argument list x |
| `DEF x y` | F ↞ (x ↦ y) (the core's `define!`) |

Assembler macros: `JMP l` = `JMPEQ NIL NIL l`; `PACK` = the state `REC`.
Cleanup spaces: VAL is W, then M and the integer readout; CONST is the
operand memory; KIND maps a value to its kind atom (the label of the memory
row it cleans up to, with the empty collections as their own kinds); CLASS
is the special forms plus the `apply`/`eval` primitive atoms, else `none`;
F is the global memory.

A microprogram is a list of instruction vectors: code cell
p ↦ ν(IL⊗instr + IR⊗next), instr ↦ ν(OP⊗opcode + Σᵢ Pᵢ⊗operandᵢ), 12
operand positions. Decoding one instruction takes three cleanups in the
code memory and one batched unbind of the 13 field roles, cleaned up
against the operand memory.

## 3. The rule table

35 rules plus 8 shared routines (`:ret`, `:eval`, `:error`, `:ret-nil`,
`:body`, `:let-next`, `:seq-next`, `:cond-next`), plus the 12-instruction
`fetch` program that computes the key and the 4-instruction `boot`. The
table is built from Clojure data at init; after that only the vectors are
used. The full table, disassembled back out of R and the code memory, is
in `out/s2-rules.txt` (`(machine/print-rule-table)`).

| mode | rule (key features beyond the mode) | instrs |
|---|---|---|
| EVAL | self-evaluating (none: the default) | 1 |
| | unknown-vector (kind k-unknown) | 1 |
| | symbol (k-sym): walk the a-list, then global memory F | 18 |
| | call (k-list): push args-fr, evaluate the head | 4 |
| | vector-literal (k-vec), map-literal (k-map), set-literal (k-set): evaluate the elements as the arguments of `vector` / `conj {}` / `conj #{}` | 3 / 6 / 6 |
| | quote, if, do, def, defn, fn, let, cond, and, or (k-list, head) | 3 5 2 6 7 2 4 2 7 5 |
| RET | bad-frame (none: the default) | 1 |
| | halt, if-branch, do-next, def-store, let-bind, cond-test, and-next, or-next, arg (frame type) | 2 12 4 5 10 9 7 8 24 |
| APPLY | not-a-function (none: the default) | 1 |
| | primitive (k-prim) | 2 |
| | apply, eval (k-prim, ident `apply`/`eval`) | 5 / 3 |
| | keyword (k-kw), map (k-map), set (k-set): via `get` | 6 / 3 / 3 |
| | closure (k-fn): named self-binding, params, `&` rest, body | 31 |

`arg` (return to an args frame) is the longest rule after `closure`: cons
the value onto the reversed list; if expressions remain, push the next
args frame, else reverse the list in a micro-loop and switch to APPLY.

A trace of `((fn [x] (if (zero? x) :zero (inc x))) 1)`, decoded from the
state vectors (`(machine/print-trace form)`, `out/s2-trace.txt`):

```
  0 EVAL   ((fn [x] (if (zero? x) :zero (inc x))) 1) call             halt
  1 EVAL   (fn [x] (if (zero? x) :zero (inc x)))    fn               args halt
  2 RET    #fn (fn [x] (if (zero? x) :zero (inc x))) arg              args halt
  3 EVAL   1                                        self-evaluating  args halt
  4 RET    1                                        arg              args halt
  5 APPLY  #fn (fn [x] (if (zero? x) :zero (inc x))) closure          halt
  6 EVAL   (if (zero? x) :zero (inc x))             if               halt
  7 EVAL   (zero? x)                                call             if halt
  8 EVAL   zero?                                    symbol           args if halt
  9 RET    #prim zero?                              arg              args if halt
 10 EVAL   x                                        symbol           args if halt
 11 RET    1                                        arg              args if halt
 12 APPLY  #prim zero?                              primitive        if halt
 13 RET    false                                    if-branch        if halt
 14 EVAL   (inc x)                                  call             halt
 15 EVAL   inc                                      symbol           args halt
 16 RET    #prim inc                                arg              args halt
 17 EVAL   x                                        symbol           args halt
 18 RET    1                                        arg              args halt
 19 APPLY  #prim inc                                primitive        halt
 20 RET    2                                        halt             halt
=> 2
```

Columns: step, mode, control, rule selected, frame types on the
continuation (top first). At step 14 the `if` frame is already gone: the
branch runs in tail position.

## 4. What host code remains

**The loop and the micro-interpreter** (`machine.clj`, section "the host
loop"): 47 lines of code. This is `exec!`, a `case` over the 9 opcodes, 12
lines; `run-micro!`, fetch/execute until the end of the list, 6 lines; and
`run-state!`, 20 lines:

```
loop: run fetch → KEY;  (i, sim, pc) = R.nearest(KEY);  run pc;
      collect W;  stop if sim(M, HALT) > θ
```

It contains no Lisp. It sees opcodes, registers and spaces, never kinds,
special forms or frames.

**The datapath glue** (section "the datapath"): 30 lines. There is one
substrate call per instruction, the decode call (with the instruction
cache), and the `PRIM` calling convention: find f in the core's primitive
memory P, marshal the argument list into a host seq, call the host
function.

**Python datapath** (`vsc_machine.py`, `Work`, ~350 lines): the memories and
the operations behind the instructions (field read, cleanup, record
building, instruction decode, rule selection, W collection). It is
substrate, like `hdc.py`.

Where host code still knows something about Lisp:
1. `PRIM` walks a cons-cell list to marshal arguments, and knows the
   list/vector labels (10, 11) to stop.
2. `REC … LIST|FN` passes the core's labels 10/14.
3. KIND knows which M rows are the four empty-collection atoms.
4. Primitives are host functions, as in the plan. One machine-level
   primitive, `%spread` (the spreading step of `apply`), is host code.

Everything else is in R and the code memory as vectors: the special forms,
evaluation order, environments and lookup, closure application, `&` rest
arguments, named-fn self reference, tail position, and which values are
functions.

**Two optimisations**, both switchable and neither changing a result:
- **Instruction cache** (`(machine/icache! false)` or
  `(init! {:icache? false})` turns it off). Each code cell is fetched and
  decoded from vectors once; afterwards the host reuses the decoded
  fields. The code memory is immutable after init. Without it, every
  instruction is decoded by cleanup: correct, and 12× slower.
- **Memos in the datapath** (`Work`). Exact vectors are recognised through a
  byte-hash memo. Role spectra are cached. A field read first cleans up
  against the constituents the record was built from. That candidate is
  the argmax a full scan of M would return, because every other stored
  item has similarity ~N(0, 1/√D) against a signal ≥ 1/√12. The W
  collector marks through recorded constituents; the test suite checks
  mid-run that this marks exactly the set that marking by unbind + cleanup
  finds (`gc_agrees`).

## 5. Performance: machine vs. host `veval`

D = 2048, fresh machine per row, fact/fib and the prelude loaded,
`out/s2-bench.edn`. "Steps": host = `veval` calls, machine = rule firings.
"M scans": full scans of the item memory (host: recognize, cleanup, 2 per
`part`, nearest/intern/deref on M; machine: the same, counted in the
datapath and in primitives). `traces+`: traces added to M by the run.

| run | wall | steps | instrs | instrs/step | M scans | M scans/step | traces+ |
|---|---|---|---|---|---|---|---|
| host `(fact 5)` | 2.12 s | 535 | | | 10 866 | 20.3 | 92 |
| machine `(fact 5)` | 2.04 s | 1 143 | 33 038 | 28.9 | 706 | 0.62 | 92 |
| machine `(fact 5)`, no icache | 25.5 s | 1 143 | 33 038 | 28.9 | 706 | 0.62 | 92 |
| host `(fib 8)` | 3.36 s | 1 070 | | | 18 570 | 17.4 | 22 |
| machine `(fib 8)` | 3.64 s | 2 380 | 64 078 | 26.9 | 1 033 | 0.43 | 22 |

Reading it:
- **About 2.1 machine steps per host `veval` call.** A CEK machine splits
  each evaluation into an EVAL step and a RET step, and argument
  evaluation into one args-frame step per argument.
- **About 28 instructions per step**, of which 12 are `fetch`, the key
  computation. At about 60 µs per instruction (caches on), the machine
  matches the host's wall time.
- **The machine touches M 15–18× less often**: the memos serve almost every
  field read from the record's own constituents. Its cost is FFTs
  (39 138 for `(fact 5)`, ~34 per step) and the operand/code memories,
  not M.
- **Trace growth in M is identical** (92, 22): the machine builds exactly
  the environment cells and closures the host builds, and they
  hash-cons to the same pointers. The machine's own records (2 387 for
  `(fact 5)`, 4 879 for `(fib 8)`) live and die in W; every one of them
  was collected.
- The first versions were 18× slower than the host. The time went into
  JVM↔Python calls and into scanning oversized caches; profiling with
  cProfile drove the memo design above.

## 6. Tail calls and deep recursion

All of these run the machine on a thread with a **256 KB** stack (a quarter
of the JVM default), in a JVM started without `-Xss512m`:

| program | host `veval` (256 KB stack) | machine (256 KB stack) |
|---|---|---|
| `(countdown 1000)`, tail-recursive | StackOverflowError | `:done` |
| `(ping 501)`, mutual tail recursion | | `:pong` |
| `(sum-to 300)`, non-tail | StackOverflowError | 45150 |
| `(countdown 10000)` (opt-in, `VSC_DEEP=1`) | | `:done`, 514 s (51 ms/call); M grew to 20 374 traces |

The host needs about 3 KB of JVM stack per recursion level and overflows a
1 MB default stack between depth 200 and 400. That is why the repo runs
with `-Xss512m`. The machine's depth is bounded by W's size, not by the
host stack. For a tail call the continuation does not grow at all: W's live
set stays at a handful of records, and the collector frees the rest.
Non-tail recursion (`sum-to`) grows the continuation in W, one `args-fr`
per pending `+`, and W grows with it.

`machine_test.clj` run on its own without `-Xss512m`:
`Ran 12 tests containing 119 assertions. 0 failures, 0 errors.` (147 s).
The host side of each comparison runs on a thread with a 512 MB stack
created inside the test, so no JVM flag is needed.

**Why not 100 000.** The machine's cost per call grows only mildly:
33 ms/call at depth 50, 51 ms/call at depth 10 000. The growth comes from
the hash-consing scans of the growing M. M itself is the problem. Every call binds `n` in a fresh environment
cell, so M grows by 2 traces per call for the host and the machine alike.
100 000 calls would add ~200 000 rows × 2 048 × 4 B × 2 (keys and traces)
≈ 3.3 GB, doubled during growth, and every hash-consing intern scans all of
M. This is the "M only grows" limit from the README, and it needs the GC for
M from the plan's open questions, not a better machine.

## 7. Differential coverage

`machine_test.clj` runs **every form of `core_test.clj`**, comparing
`machine/run` against `vsc.core/run` on the same core:
- Lisp 1.5 functions, Clojure data, arithmetic, special forms
- the `fact`/`fib`/`even?*` programs, the prelude functions
- redefinition, `eval` of built code, the metacircular interpreter
- the robustness cases with `degrade`

It also covers `(do)`, `(and)`, `(or)`, `(let [] 7)`, `(cond)`,
`(if false 1)`, two error cases (an unbound symbol, applying a vector), and
the deep-recursion tests above. The prelude itself is loaded by the
machine. 119 assertions, all passing.

Semantic notes found on the way:
- Arity errors are not detected by the machine: missing arguments bind to
  whatever the empty list's `car` cleans up to. The host reports them.
  This is one more rule to add (a `JMPEQ` on the empty list in the bind
  loop).
- Error messages are coarser: "machine error at <value>".

## 8. Soft dispatch (not implemented)

Rule selection is argmax over key similarities. Over all 2 380 selections
of `(fib 8)` (`out/s2-margins.edn`):
- the winning key's similarity is ≥ 0.448, the best loser's ≤ 0.617
- the winner−runner-up margin is min 0.096, mean 0.153

Measured in the substrate's noise unit 1/√D ≈ 0.022, the minimum margin is
over 4σ, so hard dispatch tolerates key noise of σ ≈ 0.02 per key without
error. The margin comes from the feature count, not from D: the structural
runner-up of a fully matched k-feature rule is its own (k−1)-feature
default.

If selection were a softmax blend, w_r ∝ exp(β·sim(key, r)), the winner's
weight over those 2 380 selections would be:

| β | winner weight, min | winner weight, mean |
|---|---|---|
| 1 | 0.038 | 0.045 |
| 10 | 0.21 | 0.51 |
| 30 | 0.74 | 0.94 |
| 100 | 0.9998 | 0.99997 |

What a blend would *mean*:
- **Blending code pointers does nothing useful.** Instruction fetch cleans
  up the blended pointer to one code cell, so the blend collapses to
  argmax at the first fetch, unless two rules nearly tie. Then the machine
  would jump into a mixture of two programs, which is chaos rather than
  superposition.
- **The meaningful version runs every rule with weight w_r** on the same
  state and bundles the successor states: s' = Σ w_r · run(r, s). Every
  instruction then has to act on superposed registers. That is the plan's W2
  demo 5, "superposed machine states", and it needs linear (β = 0) memories for
  deref and cleanup, or the first `FIELD` collapses the blend.
- **Leakage goes systematically to the defaults.** A state's runner-up is
  the rule's own generalisation (`if` → `call` → self-evaluating). At
  moderate β, a soft machine would partly treat `(if a b c)` as a function
  call and partly as a constant. Soft dispatch would need a
  specificity-sharpened key, for example a product of per-feature matches,
  before β can be lowered.
- β ≈ 100 is indistinguishable from argmax here, so an MHN rule memory at
  that β would run this suite unchanged. The interesting regime is
  β ≈ 10–30, where the mean winner still carries 50–94% of the weight
  but the worst case falls to 21–74%.

## 9. Open problems

- **M growth.** Unchanged from the host, and the real limit on deep
  recursion. It needs reachability GC for M, with roots in F, W and the
  operand memory.
- **Fetch dominates.** Twelve of ~28 instructions per step recompute the
  key. A key could be carried in the state and updated incrementally by
  each rule, but that moves shape knowledge into every rule.
- **The icache is where the 12× goes.** Honest VSA decoding of every
  instruction costs about 0.7 ms at D = 2048. Merging the code and operand
  memories into one attention step would be the neural-substrate answer.
- **The PRIM calling convention knows cons cells.** Moving argument
  marshalling into microcode is possible (a W-list → host-seq primitive is
  still needed at the boundary).
- **Soft dispatch** needs specificity-aware keys (§8) before superposed
  machine states (W) can build on it.
- **S3** (the tower): run S1's evaluator on this machine; rules are data, so
  an S1 program could in principle rewrite R.
