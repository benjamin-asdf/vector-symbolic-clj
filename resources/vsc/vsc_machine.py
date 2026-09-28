"""Substrate helpers for the vector CEK machine (vsc.machine, stage S2).

Kept apart from hdc.py on purpose, so that the pluggable-memory refactor of
hdc.py merges trivially: this module touches the long-term memory M only
through its public methods (nearest, deref, get, intern) and the Cleanup's
recognize, never through its arrays.

Everything here is HRR algebra plus nearest-neighbour cleanup. It adds

  * Table: a small heteroassociative cleanup memory whose rows can be freed.
    The machine owns five: the working memory W (machine states,
    continuation frames, argument lists), the code memory, the rule memory R,
    the operand memory (registers, opcodes, constants, branch targets) and
    the class memory (special forms, machine-level primitives).
  * Work: W, a mark-and-sweep collector for it, and fused calls (field read,
    instruction decode, rule selection, list walk) that keep the JVM <->
    Python chatter down.

Work also keeps three memos that change a result's cost, never the result:

  * exact: recognition of an exact vector, keyed by the hash of its bytes.
    Values are immutable and M only grows, and a vector that matched a
    stored row with similarity > 0.9 keeps that row as its nearest
    neighbour as M grows, because new rows are random.
  * parts: for a pointer, the spectrum of its trace plus the constituents it
    was built from (known when the machine built the record, or learned by
    one full cleanup). A field read ("role (/) trace", a noisy probe) is
    cleaned up against those few constituents first. The true constituent
    has similarity ~1/sqrt(k) for a k-field record, every other stored item
    ~N(0, 1/sqrt(D)), so the constituent that clears the acceptance
    threshold is the argmax a full scan of M would return.
  * role spectra: rfft of every role vector, computed once.
"""

from collections import OrderedDict

import numpy as np

F32 = np.float32
WCELL = -2   # recognize() label for a pointer into W
NUM = 4      # the Cleanup's label for integers
END = -7     # code-memory label of the end-of-program marker
INSTR = -8   # code-memory label of an instruction vector

ACCEPT = 0.25       # a constituent/W hit: signal >= 1/sqrt(12) ~ 0.29, noise < 0.1
NUM_ACCEPT = 0.3    # an integer readout of a noisy probe


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(F32)


def _h(v):
    return hash(v.tobytes())


class Table:
    """Rows key_i -> value_i with labels; freed rows are zeroed and skipped."""

    def __init__(self, dim, cap=64):
        self.dim = int(dim)
        self.K = np.zeros((cap, self.dim), F32)
        self.V = np.zeros((cap, self.dim), F32)
        self.labels = np.full(cap, -1, np.int32)
        self.live = np.zeros(cap, bool)
        self.n = 0
        self.free = []

    def _grow(self):
        self.K = np.concatenate([self.K, np.zeros_like(self.K)])
        self.V = np.concatenate([self.V, np.zeros_like(self.V)])
        self.labels = np.concatenate([self.labels, np.full_like(self.labels, -1)])
        self.live = np.concatenate([self.live, np.zeros_like(self.live)])

    def add(self, key, value=None, label=0):
        if self.free:
            i = self.free.pop()
        else:
            if self.n >= self.K.shape[0]:
                self._grow()
            i = self.n
            self.n += 1
        self.K[i] = unit(key)
        self.V[i] = 0.0 if value is None else unit(value)
        self.labels[i] = int(label)
        self.live[i] = True
        return i

    def drop(self, i):
        self.K[i] = 0.0
        self.V[i] = 0.0
        self.live[i] = False
        self.labels[i] = -1
        self.free.append(int(i))

    def nearest(self, v):
        if self.n == 0:
            return -1, 0.0
        a = self.K[: self.n] @ unit(v)
        a[~self.live[: self.n]] = -np.inf
        i = int(np.argmax(a))
        return i, float(a[i])

    def key(self, i):
        return self.K[i].astype(np.float64)

    def count(self):
        return int(self.live[: self.n].sum())


class Parts:
    """What the machine knows about one pointer: the spectrum of its trace
    and the constituents it was built from."""

    __slots__ = ("spec", "cands", "rows", "zero")

    def __init__(self, spec, cands=()):
        self.spec = spec
        self.zero = not np.any(spec)
        self.cands = list(cands)
        self.rows = None

    def add(self, v):
        self.cands.append(v)
        self.rows = None

    def match(self, u):
        if not self.cands:
            return -1, 0.0
        if self.rows is None:
            self.rows = np.stack([unit(c) for c in self.cands])
        a = self.rows @ unit(u)
        j = int(np.argmax(a))
        return j, float(a[j])


class Work:
    """The vector machine's working memory and datapath helpers.

    `cleanup` is the core's hdc.Cleanup (item memory M + integer readout);
    `F` is the core's global memory. Spaces for field reads and cleanups:
    0 VAL (W, then M and the integers), 1 CONST (the operand memory),
    2 KIND (a kind atom), 3 CLASS (special forms and machine-level
    primitives, else the `none` atom), 4 F (a global binding slot)."""

    def __init__(self, cleanup, F, parts_cap=16384, heap=64):
        self.C = cleanup
        self.sp = cleanup.space
        self.mem = cleanup.mem
        self.nums = cleanup.nums
        self.F = F
        d = self.sp.dim
        self.d = d
        self.W = Table(d, heap)
        self.wh = {}                  # hash(W key) -> row
        self.wparts = {}              # W row -> Parts
        self.parts = OrderedDict()    # hash(M pointer) -> Parts, LRU
        self.parts_cap = int(parts_cap)
        self.exact = {}               # hash(value) -> (label, index, sim)
        self.rspec = {}               # hash(role) -> conj(rfft(role))
        self.opm = Table(d, 256)
        self.cls = Table(d, 32)
        self.code = Table(d, 1024)
        self.R = Table(d, 64)
        self.zero = np.zeros(d)
        self.zc = None
        self.gc_at = 64
        self.icache = None
        self.entries = {}
        self.stats = {}
        self.reset_stats()

    # -- setup ----------------------------------------------------------------

    def set_roles(self, L, R, IL, IR, fields, gc_roles):
        """L, R: the core's cell roles. IL, IR: code-cell roles. fields: the
        instruction field roles OP, P1..Pn. gc_roles: every role a W record
        may use (the collector unbinds all of them)."""
        spec = lambda r: np.conj(np.fft.rfft(r))
        self.L, self.Rr = L, R
        self.ILc, self.IRc = spec(IL), spec(IR)
        self.fieldc = np.stack([spec(r) for r in fields])
        self.gcc = np.stack([spec(r) for r in gc_roles])

    def set_kinds(self, labels, atoms, empty_idx, empty_atoms, unknown, wcell, none):
        self.kinds = {int(l): a for l, a in zip(labels, atoms)}
        self.empties = {int(i): a for i, a in zip(empty_idx, empty_atoms)}
        self.unknown, self.wcell, self.none = unknown, wcell, none
        self.list_labels = set()

    def set_list_labels(self, labels):
        self.list_labels = set(int(l) for l in labels)

    def reset_stats(self):
        for k in ("scan", "hit", "miss", "fft", "decode", "alloc", "gc", "freed"):
            self.stats[k] = 0

    def get_stats(self):
        return [[k, int(v)] for k, v in self.stats.items()]

    # -- algebra ------------------------------------------------------------

    def _role(self, r):
        """conj(rfft(r)), cached: unbinding r from X is irfft(_role(r) * X)."""
        h = _h(r)
        s = self.rspec.get(h)
        if s is None:
            s = np.conj(np.fft.rfft(r))
            self.rspec[h] = s
        return s

    def _fft(self, x):
        self.stats["fft"] += 1
        return np.fft.rfft(x)

    def _ifft(self, X):
        self.stats["fft"] += 1
        return np.fft.irfft(X, n=self.d)

    def _record_spec(self, vals):
        """Spectrum and trace of nu(sum_i role_i (x) x_i), vals = r1, x1, r2, x2, ..."""
        X = np.zeros(self.d // 2 + 1, complex)
        for r, x in zip(vals[0::2], vals[1::2]):
            X = X + np.conj(self._role(r)) * self._fft(x)
        t = self._ifft(X)
        n = np.linalg.norm(t)
        if n > 0:
            t, X = t / n, X / n
        return X, t

    def record(self, *vals):
        """The trace nu(sum_i role_i (x) x_i), not stored anywhere."""
        return self._record_spec(vals)[1]

    def bind(self, a, b):
        return self._ifft(self._fft(a) * self._fft(b))

    def unbind(self, a, c):
        return self._ifft(self._role(a) * self._fft(c))

    # -- memory ---------------------------------------------------------------

    def alloc(self, *vals):
        """A fresh pointer p in W with W: p -> nu(sum role_i (x) x_i)."""
        self.stats["alloc"] += 1
        X, t = self._record_spec(vals)
        i = self.W.add(self.sp.unitary(), t, 0)
        p = self.W.key(i)
        self.wh[_h(p)] = i
        self.wparts[i] = Parts(X, vals[1::2])
        return p

    def intern(self, label, *vals):
        """The core's hash-consed pointer in M for nu(sum role_i (x) x_i), as
        vsc.core's `pointer` makes it."""
        self.stats["scan"] += 1
        X, t = self._record_spec(vals)
        i = self.mem.intern(self.sp.unitary(), t, int(label))
        p = self.mem.get(i)
        h = _h(p)
        self.exact[h] = (int(label), i, 1.0)
        if h not in self.parts:
            self._remember(h, Parts(X, vals[1::2]))
        return p

    def _remember(self, h, parts):
        self.parts[h] = parts
        if len(self.parts) > self.parts_cap:
            self.parts.popitem(last=False)

    def _parts(self, p):
        """Parts of pointer p: W first, then the memo, then M. A probe that is
        no pointer (an atom, an integer) has an all-zero trace."""
        h = _h(p)
        i = self.wh.get(h)
        if i is not None:
            return self.wparts[i]
        e = self.parts.get(h)
        if e is not None:
            self.parts.move_to_end(h)
            self.stats["hit"] += 1
            return e
        self.stats["miss"] += 1
        i, s = self.W.nearest(p)
        if s > 0.9:
            return self.wparts[i]
        if self.nums.read(p)[1] > 0.9:
            e = Parts(np.zeros(self.d // 2 + 1, complex))
        else:
            self.stats["scan"] += 1
            i, s = self.mem.nearest(p)
            if s < 0.9:
                e = Parts(np.zeros(self.d // 2 + 1, complex))
            else:
                self.stats["scan"] += 1
                e = Parts(self._fft(self.mem.deref(p)))
        self._remember(h, e)
        return e

    def deref(self, p):
        return self._ifft(self._parts(p).spec)

    def recognize(self, v):
        """(label, index, similarity, cleaned vector) of probe v."""
        h = _h(v)
        i = self.wh.get(h)
        if i is not None:
            return WCELL, i, 1.0, v
        e = self.exact.get(h)
        if e is not None:
            return e[0], e[1], e[2], v
        out = self._recognize(v)
        if out[0] != WCELL and out[2] > 0.999:
            self.exact[h] = out[:3]
        return out

    def _recognize(self, v):
        """The full cleanup: W, then the integer readout, then a scan of M."""
        i, s = self.W.nearest(v)
        if s > ACCEPT:
            return WCELL, i, s, self.W.key(i)
        n, sn = self.nums.read(v)
        if sn > NUM_ACCEPT:
            return NUM, n, sn, self.nums.vec(n)
        self.stats["scan"] += 1
        label, i, s = self.C.recognize(v)
        if label == NUM:
            return label, i, s, self.nums.vec(i)
        return label, i, s, self.mem.get(i)

    def kind(self, v):
        label, i, s, _ = self.recognize(v)
        if label == WCELL:
            return self.wcell
        if s <= 0.5:
            return self.unknown
        if label != NUM and i in self.empties:
            return self.empties[i]
        return self.kinds.get(label, self.unknown)

    def clean(self, v, space):
        space = int(space)
        if space == 0:
            return self.recognize(v)[3]
        if space == 1:
            return self.opm.key(self.opm.nearest(v)[0])
        if space == 2:
            return self.kind(v)
        if space == 3:
            i, s = self.cls.nearest(v)
            return self.cls.key(i) if s > 0.5 else self.none
        if space == 4:
            self.stats["scan"] += 1
            return self.F.get(self.F.nearest(v)[0])
        raise ValueError(space)

    def field(self, p, role, space):
        """clean_space(role (/) deref(p)): one field of a record or cell."""
        e = self._parts(p)
        space = int(space)
        if e.zero:
            # no pointer: the core's car of an atom is the cleanup of zero
            if self.zc is None:
                self.zc = self.clean(self.zero, 0)
            return self.zc if space == 0 else self.clean(self.zero, space)
        u = self._ifft(self._role(role) * e.spec)
        if space in (0, 1):
            j, s = e.match(u)
            if s > ACCEPT:
                return e.cands[j]
            r = self.clean(u, space)
            if np.linalg.norm(u) > 0:
                e.add(r)
            return r
        return self.clean(u, space)

    def items_of(self, p, limit=1000000):
        """Elements of the list/vector cell chain p (cells in W or M)."""
        out = []
        for _ in range(limit):
            label, i, s, _ = self.recognize(p)
            if not (label == WCELL or (label in self.list_labels and s > 0.5
                                       and i not in self.empties)):
                break
            out.append(self.field(p, self.L, 0))
            p = self.field(p, self.Rr, 0)
        return out

    # -- W collector ----------------------------------------------------------

    def gc(self, root, force=False):
        """Mark the W records reachable from root through any role, drop the
        rest. Runs when W has doubled since the last collection."""
        if not force and self.W.count() < self.gc_at:
            return 0
        self.stats["gc"] += 1
        marked = set()
        stack = []
        if root is not None:
            i, s = self.W.nearest(root)
            if s > 0.9:
                stack.append(i)
        while stack:
            i = stack.pop()
            if i in marked:
                continue
            marked.add(i)
            U = np.fft.irfft(self.gcc * self.wparts[i].spec[None, :], n=self.d).astype(F32)
            S = U @ self.W.K[: self.W.n].T
            S[:, ~self.W.live[: self.W.n]] = -np.inf
            for j, sj in zip(np.argmax(S, axis=1), np.max(S, axis=1)):
                if sj > ACCEPT and int(j) not in marked:
                    stack.append(int(j))
        dropped = 0
        for i in range(self.W.n):
            if self.W.live[i] and i not in marked:
                self.wh.pop(_h(self.W.key(i)), None)
                self.wparts.pop(i, None)
                self.W.drop(i)
                dropped += 1
        self.stats["freed"] += dropped
        self.gc_at = max(64, 2 * len(marked))
        return dropped

    def w_count(self):
        return self.W.count()

    # -- code, operands, rules -------------------------------------------------

    def asm_instr(self, fields):
        """The instruction vector nu(OP (x) op + sum_i P_i (x) a_i) as a pointer in
        the code memory, shared by equal instructions. `fields` holds one
        vector per field role (None for an absent operand)."""
        acc = np.zeros(self.d)
        for c, x in zip(self.fieldc, fields):
            if x is not None:
                acc = acc + np.fft.irfft(np.conj(c) * np.fft.rfft(x), n=self.d)
        t = unit(acc)
        live = self.code.labels[: self.code.n] == INSTR
        if live.any():
            a = self.code.V[: self.code.n] @ t
            a[~live] = -np.inf
            j = int(np.argmax(a))
            if a[j] > 0.999:
                return self.code.key(j)
        return self.code.key(self.code.add(self.sp.unitary(), t, INSTR))

    def asm_cell(self, key, instr, nxt):
        """Code cell key -> nu(IL (x) instr + IR (x) next)."""
        t = np.fft.irfft(np.conj(self.ILc) * np.fft.rfft(instr), n=self.d) + \
            np.fft.irfft(np.conj(self.IRc) * np.fft.rfft(nxt), n=self.d)
        return self.code.key(self.code.add(key, t, 0))

    def asm_end(self):
        return self.code.key(self.code.add(self.sp.unitary(), None, END))

    def decode(self, pc):
        """Fetch and decode the instruction at code pointer pc: the operand
        memory indices of its fields (-1 where a field is absent) and the next
        code pointer (None at the end of a program)."""
        self.stats["decode"] += 1
        i, _ = self.code.nearest(pc)
        X = np.fft.rfft(self.code.V[i])
        j, _ = self.code.nearest(np.fft.irfft(self.ILc * X, n=self.d))
        k, _ = self.code.nearest(np.fft.irfft(self.IRc * X, n=self.d))
        T = np.fft.rfft(self.code.V[j])
        U = np.fft.irfft(self.fieldc * T[None, :], n=self.d).astype(F32)
        S = U @ self.opm.K[: self.opm.n].T
        idx, val = np.argmax(S, axis=1), np.max(S, axis=1)
        ops = [int(a) if s > 0.15 else -1 for a, s in zip(idx, val)]
        nxt = None if self.code.labels[k] == END else self.code.key(k)
        return ops, nxt

    def select(self, key):
        """Rule selection: (rule index, similarity, entry code pointer) of the
        rule key nearest to `key`."""
        i, s = self.R.nearest(key)
        if i not in self.entries:
            self.entries[i] = self.R.V[i].astype(np.float64)
        return i, s, self.entries[i]

    def rule_sims(self, key):
        """Similarity of `key` to every rule key (for the soft-dispatch note)."""
        return (self.R.K[: self.R.n] @ unit(key)).tolist()

    def probe(self, table, v):
        """[index, similarity] of v's nearest row in one of the small tables."""
        i, s = getattr(self, table).nearest(v)
        return [i, s]
