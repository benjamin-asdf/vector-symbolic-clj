# Reproducing my failure modes in the original authors' code

Harnesses that run the failure modes of my paper (`paper/draft.md` §4; the page numbers them as the overview's traps 1–5)
against the original code of the two "Hey Pentti" papers. The write-up is
the notebook `notebooks/holis_reproduction.clj`, which reads the JSON files
in this directory.

| code | paper | pinned commit | license |
|---|---|---|---|
| [eilene-ftf/holis](https://github.com/eilene-ftf/holis), `holis.ipynb` | Tomkins-Flanagan & Kelly, "Hey Pentti, We Did It!" (ICCM 2024) | `db208d782863febfb50ec46b25a3700369f5cfea` | AGPL-3.0 |
| [hanleyc01/residuelisp](https://github.com/hanleyc01/residuelisp) | Hanley, Tomkins-Flanagan & Kelly, "Hey Pentti, We Did (More of) It!" (IJCNN 2025, arXiv:2511.08767) | `09864c74f1a35680a0008e59d7ab2571269a986f` | MIT |

None of their code is vendored here. The holis fixes wrap its functions
rather than re-typing them; the residuelisp fixes (MIT) re-type only the
few short functions they replace, each attributed in its docstring.
`upstream.py` fetches it at run time
from the pinned commit (from a local clone if `HOLIS_CLONE` /
`RESIDUELISP_CLONE` is set, else from GitHub) into `.upstream/`
(gitignored). holis is a notebook: I exec its code cells into a module.
residuelisp is a package: I put its `src/` on `sys.path`.

## Setup

residuelisp needs Python ≥ 3.13; holis needs only numpy.

```sh
cd repro/holis
python3 -m venv .venv            # Python 3.14.7 was used
.venv/bin/pip install numpy more-itertools   # numpy 2.5.4, more-itertools 11.2.1
```

## Run

Each script writes one JSON file next to itself. Run them one at a time.
Times are on one core of the machine I used.

```sh
cd repro/holis
export OPENBLAS_NUM_THREADS=1
ulimit -v 8000000
.venv/bin/python holis_control.py          # noise-free control      ~25 min
.venv/bin/python holis_errata.py           # E1-E4, noise-free       ~1 min
.venv/bin/python holis_commute.py          # trap 2                  ~1 min
.venv/bin/python holis_thresholds.py       # trap 3                  ~30 min
.venv/bin/python holis_cells.py            # trap 1, paper mode      ~3 min
.venv/bin/python residuelisp_control.py    # noise-free control      ~1 min
.venv/bin/python residuelisp_errata.py     # R1-R5, noise-free       ~1 min
.venv/bin/python residuelisp_residues.py   # integer sums, zero      ~15 min
.venv/bin/python residuelisp_thresholds.py # trap 3                  ~10 min
.venv/bin/python holis_fixes.py            # fixes trap 2, E1-E4     ~12 min
.venv/bin/python holis_tail.py             # E5 and the alternative  ~20 min
.venv/bin/python residuelisp_fixes.py      # fixes R1-R5             ~10 min
```

## What the instrumentation changes

- **Probe noise**, as in my paper: before every cleanup, the probe p
  becomes p + σ·|p|/√D · N(0, I) (complex N for residuelisp's FHRR).
  holis: `SimpleCleanUp.recall` (item and function memory) and
  `SimpleAssoc.recall` (pointer memory). residuelisp:
  `CleanupMemory.recall` and `AssociativeMemory.deref`.
- **Controls** are one attribute or one function each: holis's pointer
  threshold set to 5/√D instead of 0.2 (`holis_thresholds.py`; −1, plain
  argmax, breaks their atom test, see `holis_cells.py`), holis's inner `car`
  answering NIL for non-pointers (`holis_errata.py`), holis's cell stored
  as its own bundle, the paper's eq. 2 (`holis_cells.py`).
- **Fixes**: `holis_fixes.py` and `residuelisp_fixes.py` test the fixes
  suggested in the notebook's "Errata for the authors". Each fix is a
  runtime patch: for residuelisp, one module function replaced by my
  version; for holis, a fresh environment changed after `vsa_env` builds
  it (closure variables, `env` entries, lexicon and item memory) or, for
  E4, one class method wrapped. No upstream file changes and no holis
  code is copied. Each script runs the erratum's programs with and without the fix,
  then the noise-free control programs with the fix.
- **Budget**: a holis run that builds more than a fixed number of cells,
  or a residuelisp run that makes more than a fixed number of cleanups, is
  stopped and counted as a failure.
- **Logging**: residuelisp prints debug output on every step and calls its
  `decode` inside those messages; I silence the prints and stub `decode`
  while a program runs.
- **Spacing**: residuelisp's lexer drops a `)` right after an integer, so
  my residuelisp programs put a space there (erratum R5).

## Outputs

| file | what |
|---|---|
| `holis_control.json` | noise-free control, small programs and their fizzbuzz helpers |
| `holis_errata.json` | E1 CAR of an atom (with control), E2 CAR does not evaluate, E3 LABEL, E4 memory growth |
| `holis_commute.json` | trap 2 via role symbols, with (x . y) / (y . x) controls |
| `holis_thresholds.json` | trap 3 lookups, reads, a program; θ = 0.2 vs 5/√D |
| `holis_cells.json` | trap 1 paper mode vs pointer cells |
| `residuelisp_control.json` | noise-free control |
| `residuelisp_errata.json` | R1 integer tag (trap 4), R2 `and`, R3 `if`, R4 decoder range, R5 lexer |
| `residuelisp_residues.json` | integer superpositions and zero against their RHC code (not on the page) |
| `residuelisp_thresholds.json` | trap 3 noise sweep |
| `holis_fixes.json` | fixes for trap 2, E2, E3, E4 (and E1 rechecked): erratum programs before and after, control programs with each fix, fizzbuzz helpers over 5 seeds with E1, E2 and E4 |
| `holis_tail.json` | E5 list tails (a question, not an erratum): programs with their code and with element-wise tails, LABEL with E3, control programs, fizzbuzz helpers with elementwise tails alone and with E2 |
| `residuelisp_fixes.json` | fixes for R1-R5: erratum programs before and after, control programs with each fix, R1 test pattern, R5 lexer |

Every JSON file records the upstream commit, Python and numpy versions,
and the settings it ran with.

-----

*Generated by Claude on Benjamin Schwerdtner's behalf.*
