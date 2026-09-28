"""HDC substrate: holographic reduced representations (Plate, 1995) and
cleanup memories, as used by Tomkins-Flanagan & Kelly (2024).

Everything numeric lives here. The Clojure side only ever holds opaque
references to numpy vectors and asks this module to combine, compare and
clean them up.

The substrate is also where experiments act. Every vector-producing
operation goes through a `Space`, which owns the noise knobs (op noise,
probe noise, memory damage, lesion) and the instrumentation (per-kind op
counters, the cleanup margin log, an op budget). With every knob off the
arithmetic is exactly that of the plain substrate: knobs draw from their own
random stream and are skipped entirely at zero, so results stay
byte-identical.

Cleanup memories are pluggable backends behind one interface (`Memory`):

  codebook   hardmax(M p) M: the lookup table (default)
  linear     β = 0: Mᵀ M p and Vᵀ K p, the linear readout, never snapped
  mhn        softmax(β K p): modern Hopfield network (P2, a slot for now)
"""

import gc
import os
import time
import weakref

# Matrix-vector products here are small; BLAS thread pools only burn cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402


class BudgetExceeded(RuntimeError):
    """A run used more substrate ops, memory rows or time than allowed."""


class Space:
    """An HRR vector space R^dim with circular convolution as the product,
    plus the machine's noise knobs and instrumentation."""

    def __init__(self, dim=4096, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        # weak: a memory refers to its space, so strong refs both ways would
        # make a cycle and keep every discarded machine's arrays alive until
        # Python's cyclic collector happens to run
        self._memories = []
        # instrumentation
        self.counting = False
        self.counts = {}
        self.logging = False
        self.log = []
        self.max_ops = None
        self.max_rows = None
        self.deadline = None
        self.ops = 0
        self.configure()

    # -- knobs --------------------------------------------------------------

    def configure(self, op_noise=0.0, probe_noise=0.0, lesion=0.0, noise_seed=None):
        """Set the noise knobs. Noise draws from its own stream, seeded from
        the space's seed (or `noise_seed`), never from the atom stream.

        op_noise σ     every bind/unbind/bundle output v gets v + σ|v|/√D·z
        probe_noise σ  every cleanup probe v gets v + σ|v|/√D·z, so a unit
                       probe carries noise of expected norm σ
        lesion f       a fixed random fraction f of dimensions is dead: zero
                       in every vector produced, stored or compared. Rows
                       already stored are lesioned when this is set."""
        self.op_noise = float(op_noise)
        self.probe_noise = float(probe_noise)
        self.lesion = float(lesion)
        base = [self.seed, 1] if noise_seed is None else [int(noise_seed), 1]
        self.noise_rng = np.random.default_rng(base)
        self.mask = None
        if self.lesion > 0:
            dead = np.random.default_rng([base[0], 2]).permutation(self.dim)
            self.mask = np.ones(self.dim)
            self.mask[dead[: int(round(self.lesion * self.dim))]] = 0.0
            for mem in self.memories():
                mem._lesion_rows()

    def knobs(self):
        return {"op_noise": self.op_noise, "probe_noise": self.probe_noise,
                "lesion": self.lesion}

    def _noise(self, v, sigma):
        return v + (sigma * np.linalg.norm(v) / np.sqrt(self.dim)) * \
            self.noise_rng.standard_normal(self.dim)

    def _made(self, kind, v):
        """Every produced vector passes here: count it, lesion it."""
        self.tick(kind)
        if self.mask is not None:
            v = v * self.mask
        return v

    def _op(self, kind, v):
        """Output of bind/unbind/bundle: op noise, then lesion."""
        self.tick(kind)
        if self.op_noise > 0:
            v = self._noise(v, self.op_noise)
        if self.mask is not None:
            v = v * self.mask
        return v

    def _in(self, v):
        """Inputs lose their dead dimensions too (a host-held copy made
        before the lesion must not smuggle them back in)."""
        return v * self.mask if self.mask is not None else v

    def perturb(self, v):
        """A cleanup probe as the memory sees it: probe noise, lesion."""
        if self.probe_noise > 0:
            v = self._noise(v, self.probe_noise)
        if self.mask is not None:
            v = v * self.mask
        return v

    def damage(self, sigma):
        """Synaptic damage: add noise of norm ~σ to every stored row of every
        memory of this space, once."""
        for mem in self.memories():
            mem.damage(sigma)

    def memories(self):
        """The live memories over this space."""
        live = [r() for r in self._memories]
        self._memories = [weakref.ref(m) for m in live if m is not None]
        return [m for m in live if m is not None]

    # -- instrumentation ----------------------------------------------------

    def tick(self, kind, k=1):
        if self.counting:
            self.counts[kind] = self.counts.get(kind, 0) + k
        if self.max_ops is not None:
            self.ops += 1
            if self.ops > self.max_ops:
                raise BudgetExceeded("op budget of %d exceeded" % self.max_ops)
            if self.deadline is not None and self.ops % 256 == 0 \
                    and time.monotonic() > self.deadline:
                raise BudgetExceeded("time budget exceeded")

    def instrument(self, counting=None, logging=None):
        if counting is not None:
            self.counting = bool(counting)
        if logging is not None:
            self.logging = bool(logging)

    def reset_instruments(self):
        self.counts = {}
        self.log = []
        self.ops = 0

    def budget(self, max_ops=None, max_rows=None, seconds=None):
        """Abort the run (BudgetExceeded) past this many ops, rows in any
        memory, or seconds from now. None lifts a limit."""
        self.max_ops = None if max_ops is None else int(max_ops)
        self.max_rows = None if max_rows is None else int(max_rows)
        self.deadline = None if seconds is None else time.monotonic() + float(seconds)
        if self.max_ops is None and self.deadline is not None:
            self.max_ops = 1 << 62
        self.ops = 0

    def get_counts(self):
        return dict(self.counts)

    def record(self, kind, a, i):
        """Log one cleanup: [kind, winner, top-1 similarity, top1 − top2].
        A best match below the chance floor 4.5/√D is a miss (a lookup that
        is meant to fail, such as a symbol that is not a special form): it
        is logged with winner -1 and no margin."""
        if not self.logging:
            return
        s1 = float(a[i]) if i >= 0 else 0.0
        if s1 < 4.5 / np.sqrt(self.dim):
            self.log.append((kind, -1, s1, float("nan")))
            return
        if a.shape[0] >= 2:
            s2 = float(np.partition(a, -2)[-2])
        else:
            s2 = 0.0
        self.log.append((kind, int(i), s1, s1 - s2))

    def relabel(self, kind, winner, s1):
        """Replace the last log entry: the winner came from a readout, not
        from the table, so it has no runner-up margin."""
        if self.logging and self.log:
            self.log[-1] = (kind, int(winner), float(s1), float("nan"))

    def get_log(self):
        return [list(e) for e in self.log]

    def log_indices(self):
        """[kind, winner] of every logged cleanup, for ground-truth diffs."""
        return [[e[0], e[1]] for e in self.log]

    def margin_stats(self):
        if not self.log:
            return {"n": 0}
        m = np.array([e[3] for e in self.log if e[3] == e[3]])
        s = np.array([e[2] for e in self.log])
        if m.size == 0:
            return {"n": len(self.log)}
        return {"n": len(self.log), "min": float(m.min()),
                "p05": float(np.percentile(m, 5)), "median": float(np.median(m)),
                "s1_min": float(s.min()), "s1_median": float(np.median(s))}

    # -- atoms --------------------------------------------------------------

    def unitary(self):
        """A random unitary vector: every Fourier coefficient has modulus 1,
        so binding with it is exactly invertible and powers stay unit length."""
        d = self.dim
        spec = np.exp(1j * self.rng.uniform(-np.pi, np.pi, d // 2 + 1))
        spec[0] = self.rng.choice([-1.0, 1.0])
        if d % 2 == 0:
            spec[-1] = self.rng.choice([-1.0, 1.0])
        return self._made("unitary", np.fft.irfft(spec, n=d))

    def unitaries(self, k):
        """k random unitary vectors as rows (one batch, one rng draw per row
        exactly as k calls of `unitary` would make)."""
        return np.stack([self.unitary() for _ in range(int(k))])

    def identity(self):
        """The unit of binding (a delta at index 0)."""
        v = np.zeros(self.dim)
        v[0] = 1.0
        return self._made("identity", v)

    # -- algebra ------------------------------------------------------------

    def bind(self, a, b):
        a, b = self._in(a), self._in(b)
        return self._op("bind", np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=self.dim))

    def unbind(self, a, c):
        """a ⊘ c: bind c with the involution (approximate inverse) of a.
        Exact when a is unitary."""
        a, c = self._in(a), self._in(c)
        return self._op("unbind",
                        np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(c), n=self.dim))

    def inverse(self, a):
        return self._made("inverse", np.fft.irfft(np.conj(np.fft.rfft(self._in(a))), n=self.dim))

    def power(self, a, k):
        """Integer power under binding: a ⊗ a ⊗ ... (k times)."""
        return self._made("power", np.fft.irfft(np.fft.rfft(self._in(a)) ** int(k), n=self.dim))

    def bundle(self, *vs):
        if self.mask is not None:
            vs = [self._in(v) for v in vs]
        return self._op("bundle", np.sum(vs, axis=0))

    def normalize(self, v):
        self.tick("normalize")
        v = self._in(v)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def sim(self, a, b):
        self.tick("sim")
        a, b = self._in(a), self._in(b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))

    def degrade(self, v, amount):
        """Add Gaussian noise with norm `amount` * |v|."""
        noise = self.normalize(self.rng.normal(size=self.dim))
        return self._made("degrade", v + float(amount) * np.linalg.norm(v) * noise)


class Numbers:
    """Integers as powers of one unitary base: n = B^n, so n + m = B^n ⊗ B^m.

    B is a residue-number-system base (cf. Tomkins-Flanagan & Kelly, "Hey
    Pentti, We Did (More of) It!"): the Fourier bins are split at random into
    one band per prime modulus p, and on the band of p every coefficient of B
    is a p-th root of unity. B^n on that band therefore depends only on
    n mod p. Cleanup is algorithmic: find each residue by matching its band
    against p candidates, combine them with the Chinese remainder theorem,
    and confirm with the similarity of v to B^n. It tolerates superposition
    noise, because each band still carries a coherent share of the signal."""

    def __init__(self, space, moduli=(5, 7, 11, 13, 17, 19)):
        self.space = space
        self.moduli = [int(p) for p in moduli]
        self.P = int(np.prod(self.moduli))
        self.half = (self.P - 1) // 2
        d = space.dim
        bins = np.arange(1, d // 2 if d % 2 == 0 else d // 2 + 1)
        space.rng.shuffle(bins)
        self.bands = np.array_split(bins, len(self.moduli))
        self.expo = np.zeros(d // 2 + 1)  # B's phase at bin k = 2π expo[k]
        self.cands = []
        for band, p in zip(self.bands, self.moduli):
            a = space.rng.integers(1, p, size=band.size)
            self.expo[band] = a / p
            # candidate spectra for residues 0..p-1, conjugated for matching
            r = np.arange(p)[:, None]
            self.cands.append(np.exp(-2j * np.pi * r * a[None, :] / p))
        self.spec = np.exp(2j * np.pi * self.expo)

    def base(self):
        return self.vec(1)

    def vec(self, n):
        spec = np.exp(2j * np.pi * ((self.expo * int(n)) % 1.0))
        return self.space._made("num_vec", np.fft.irfft(spec, n=self.space.dim))

    def _crt(self, residues):
        n = 0
        for r, p in zip(residues, self.moduli):
            q = self.P // p
            n += r * q * pow(q, -1, p)
        n %= self.P
        return n - self.P if n > self.half else n

    def read(self, v):
        """[n, similarity of v to B^n]."""
        return self._read(self.space.perturb(v))

    def _read(self, v):
        self.space.tick("num_read")
        X = np.fft.rfft(v)
        residues = [int(np.argmax((c @ X[band]).real))
                    for band, c in zip(self.bands, self.cands)]
        n = self._crt(residues)
        return [n, self.space.sim(v, self.vec(n))]


def peel(memories, v, theta, limit=256):
    """Explaining away over several memories at once: repeatedly clean up the
    residual of a superposition and subtract what was found. Returns
    [memory-index, row] pairs for constituents whose similarity to the
    original exceeds theta. Backend-agnostic: it only uses `scores`, the
    similarities of a probe to every stored key."""
    space = memories[0].space if memories else None
    n = np.linalg.norm(v)
    if n == 0:
        return []
    r = (v / n).astype(np.float32)
    out = []
    if space is not None:
        space.tick("peel")
    for _ in range(int(limit)):
        probe = r
        if space is not None and (space.probe_noise > 0 or space.mask is not None):
            probe = space.perturb(r.astype(np.float64)).astype(np.float32)
        best = (-np.inf, -1, -1)
        best_a = None
        for j, mem in enumerate(memories):
            if mem.n == 0:
                continue
            a = mem.scores(probe)
            i = int(np.argmax(a))
            if a[i] > best[0]:
                best = (float(a[i]), j, i)
                best_a = a
        s, j, i = best
        if best_a is not None:
            space.record("peel", best_a, i)
        if s < theta:
            break
        out.append([j, i])
        r = r - s * memories[j].M[i]
    return out


class Memory:
    """Cleanup memory, optionally heteroassociative: row i of M (a key, or
    "pointer") is associated with row i of V (the trace it points to).

    Rows are stored unit length. Each row carries an integer label, which the
    host uses as the kind of the stored item (symbol, number, list, ...).

    This class is the interface and the `codebook` backend, M(p) =
    hardmax(M p) M. Public methods take care of probe noise, lesion,
    counting and the margin log; a backend overrides only the hooks:

      scores(u)          similarity of unit probe u to every key, (n,)
      _select(u, kind)   -> (index, similarity, scores): the winner
      _clean(u, a, i)    autoassociative readout for probe u
      _deref(u, a, i)    heteroassociative readout (a trace) for pointer u

    where a = scores(u) and i = argmax(a) as computed by `_select`."""

    backend = "codebook"

    def __init__(self, space, capacity=4096, name=None):
        self.space = space
        self.dim = space.dim
        self.name = name
        self.M = np.zeros((int(capacity), self.dim), dtype=np.float32)
        self.V = np.zeros((int(capacity), self.dim), dtype=np.float32)
        self.labels = np.full(int(capacity), -1, dtype=np.int32)
        self.n = 0
        space._memories.append(weakref.ref(self))

    def _grow(self):
        self.M = np.concatenate([self.M, np.zeros_like(self.M)])
        self.V = np.concatenate([self.V, np.zeros_like(self.V)])
        self.labels = np.concatenate([self.labels, np.full_like(self.labels, -1)])

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    def _store(self, v):
        """A row as stored: lesioned, unit length, float32."""
        m = self.space.mask
        return self._unit(v * m if m is not None else v)

    def _probe(self, v):
        """A probe as the memory sees it: noisy, lesioned, unit, float32."""
        return self._unit(self.space.perturb(v))

    # -- backend hooks (codebook) ---------------------------------------------

    def scores(self, u):
        self.space.tick("rows", self.n)
        return self.M[: self.n] @ u

    def _select(self, u, kind):
        if self.n == 0:
            return -1, 0.0, None
        a = self.scores(u)
        i = int(np.argmax(a))
        self.space.record(kind, a, i)
        return i, float(a[i]), a

    def _clean(self, u, a, i):
        return self.M[i].astype(np.float64)

    def _deref(self, u, a, i):
        return self.V[i].astype(np.float64)

    # -- interface ------------------------------------------------------------

    def size(self):
        return self.n

    def label(self, i):
        return int(self.labels[int(i)])

    def nearest(self, v):
        """[index, similarity] of the best-matching key, or [-1, 0.0]."""
        self.space.tick("nearest")
        i, s, _ = self._select(self._probe(v), "nearest")
        return [i, s]

    def recall(self, v):
        """[index, label, similarity] of the cleaned-up key."""
        self.space.tick("recall")
        i, s, _ = self._select(self._probe(v), "recall")
        return [i, int(self.labels[i]) if i >= 0 else -1, s]

    def clean(self, v):
        """M(v): the backend's autoassociative readout of probe v."""
        self.space.tick("clean")
        u = self._probe(v)
        i, _, a = self._select(u, "clean")
        return self._clean(u, a, i)

    def deref(self, p):
        """The trace associated with pointer p (for the codebook: the trace of
        the key nearest to p)."""
        self.space.tick("deref")
        u = self._probe(p)
        i, _, a = self._select(u, "deref")
        return self._deref(u, a, i)

    def _exact_nearest(self, v):
        """Storage-time lookup (hash-consing): no probe noise, not logged."""
        if self.n == 0:
            return [-1, 0.0]
        a = self.scores(self._unit(v))
        i = int(np.argmax(a))
        return [i, float(a[i])]

    def add(self, v, label=0, dedupe=None):
        """Append key v; with `dedupe`, return the index of an existing key
        more similar than it instead of storing a copy."""
        self.space.tick("add")
        if dedupe is not None and self.n > 0:
            i, s = self._exact_nearest(v)
            if s > dedupe:
                return i
        if self.n >= self.M.shape[0]:
            if self.space.max_rows is not None and self.n >= self.space.max_rows:
                raise BudgetExceeded("memory budget of %d rows exceeded" % self.space.max_rows)
            self._grow()
        if self.space.max_rows is not None and self.n >= self.space.max_rows:
            raise BudgetExceeded("memory budget of %d rows exceeded" % self.space.max_rows)
        self.M[self.n] = self._store(v)
        self.labels[self.n] = int(label)
        self.n += 1
        return self.n - 1

    def add_random(self, k, label=0):
        """Store k fresh random unitary atoms (a bulk `add` of `unitary`s)."""
        for _ in range(int(k)):
            self.add(self.space.unitary(), label)
        return self.n

    def intern(self, key, trace, label, dedupe=0.999):
        """Associate a fresh pointer `key` with `trace`, unless an equal trace
        of the same label is already stored (hash-consing): then return that
        trace's index, so equal structures share one pointer."""
        self.space.tick("intern")
        t = self._store(trace)
        if self.n > 0:
            self.space.tick("rows", self.n)
            a = self.V[: self.n] @ t
            a[self.labels[: self.n] != int(label)] = -np.inf
            i = int(np.argmax(a))
            if a[i] > dedupe:
                return i
        i = self.add(key, label)
        self.V[i] = t
        return i

    def put(self, i, v):
        self.space.tick("put")
        self.M[int(i)] = self._store(v)

    def get(self, i):
        return self.M[int(i)].astype(np.float64)

    def digest(self):
        """sha1 of the stored rows, traces and labels: a reproducibility check."""
        import hashlib
        h = hashlib.sha1()
        for X in (self.M, self.V, self.labels):
            h.update(np.ascontiguousarray(X[: self.n]).tobytes())
        return h.hexdigest()

    # -- damage -------------------------------------------------------------

    def damage(self, sigma):
        """Add noise of expected norm σ to every stored key and trace row,
        then renormalise. Empty trace rows (keys without a trace) stay empty."""
        sigma = float(sigma)
        if sigma <= 0 or self.n == 0:
            return
        rng, d = self.space.noise_rng, self.dim
        for X in (self.M, self.V):
            rows = X[: self.n].astype(np.float64)
            live = np.linalg.norm(rows, axis=1) > 0
            rows[live] += (sigma / np.sqrt(d)) * rng.standard_normal((int(live.sum()), d))
            if self.space.mask is not None:
                rows *= self.space.mask
            nrm = np.linalg.norm(rows, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            X[: self.n] = (rows / nrm).astype(np.float32)

    def _lesion_rows(self):
        m = self.space.mask
        for X in (self.M, self.V):
            rows = X[: self.n] * m.astype(np.float32)
            nrm = np.linalg.norm(rows, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            X[: self.n] = rows / nrm


class Linear(Memory):
    """β = 0, the linear limit: M(p) = Mᵀ M p and deref(p) = Vᵀ K p, returned
    as is, never snapped to a row. Superpositions pass through linearly
    (deref(Σ wᵢ pᵢ) = Σ wᵢ traceᵢ), at the price of crosstalk from every
    stored row. `nearest`/`recall` still report the argmax row."""

    backend = "linear"

    def _clean(self, u, a, i):
        if a is None:
            return np.zeros(self.dim)
        return (self.M[: self.n].T @ a).astype(np.float64)

    def _deref(self, u, a, i):
        if a is None:
            return np.zeros(self.dim)
        return (self.V[: self.n].T @ a).astype(np.float64)


class Hopfield(Memory):
    """Modern Hopfield network (Ramsauer et al. 2020): M(p) = softmax(β M p) M
    iterated, deref(p) = Vᵀ softmax(β K p). A slot for P2, not implemented.

    Intended options: beta (float, ∞ = codebook), iters (int), mode "snap"
    (return the argmax row after iterating) or "soft" (return the blend).
    Override `_select` (the winner after iterating, for nearest/recall and
    snap), `_clean` and `_deref`; keep the Memory constructor signature and
    take the options as keywords."""

    backend = "mhn"

    def __init__(self, space, capacity=4096, name=None, beta=16.0, iters=1, mode="snap"):
        raise NotImplementedError("the mhn backend is P2 and not implemented yet")


BACKENDS = {"codebook": Memory, "linear": Linear, "mhn": Hopfield}


def make_memory(space, backend="codebook", name=None, **opts):
    """A fresh cleanup memory of the named backend over `space`."""
    if backend not in BACKENDS:
        raise ValueError("unknown memory backend %r (have %s)" % (backend, sorted(BACKENDS)))
    return BACKENDS[backend](space, name=name, **opts)


class Cleanup:
    """The combined cleanup of a machine: the item memory M (atoms and
    pointers) plus the algorithmic readout of integers. Fusing the steps
    into one call keeps the host/substrate chatter down."""

    NUM = 4

    def __init__(self, space, mem, nums, theta=0.9):
        self.space, self.mem, self.nums, self.theta = space, mem, nums, theta

    def _recognize(self, v):
        """v is already perturbed; returns ([label, index, sim], u, scores)."""
        mem = self.mem
        u = mem._unit(v)
        i, s, a = mem._select(u, "recognize")
        if s > self.theta:
            return [int(mem.labels[i]), i, s], u, a
        n, sn = self.nums._read(v)
        if sn > s:
            self.space.relabel("num", n, sn)
            return [self.NUM, n, sn], u, a
        return [int(mem.labels[i]) if i >= 0 else -1, i, s], u, a

    def recognize(self, v):
        """[label, index, similarity]; index is n itself for integers."""
        self.space.tick("recognize")
        return self._recognize(self.space.perturb(v))[0]

    def cleanup(self, v):
        self.space.tick("cleanup")
        (label, i, s), u, a = self._recognize(self.space.perturb(v))
        return self.nums.vec(i) if label == self.NUM else self.mem._clean(u, a, i)

    def part(self, role, p):
        """M(role ⊘ M(p)): one field of the trace pointer p refers to."""
        return self.cleanup(self.space.unbind(role, self.mem.deref(p)))


def bench(space, mem, k=50):
    """Seconds per call of the substrate ops, measured inside Python (no
    host bridge), on memory `mem` as it is. Fills the trace rows V with a
    permutation of the keys so that `deref` has something to return."""
    rng = np.random.default_rng(0)
    if mem.n:
        mem.V[: mem.n] = mem.M[rng.permutation(mem.n)]
    a, b = space.unitary(), space.unitary()
    parts = [mem.get(i) for i in range(min(8, mem.n))]
    sup = np.sum(parts, axis=0) if parts else a
    out = {}

    def t(name, f):
        t0 = time.perf_counter()
        for _ in range(int(k)):
            f()
        out[name] = (time.perf_counter() - t0) / int(k)

    t("bind", lambda: space.bind(a, b))
    t("unbind", lambda: space.unbind(a, b))
    t("bundle", lambda: space.bundle(a, b, a))
    t("sim", lambda: space.sim(a, b))
    t("nearest", lambda: mem.nearest(a))
    t("clean", lambda: mem.clean(a))
    t("deref", lambda: mem.deref(a))
    t("peel8", lambda: peel([mem], sup, -1.0, len(parts)))
    return out


def collect():
    """Run Python's cyclic garbage collector; the number of objects freed."""
    return gc.collect()


def versions():
    """numpy version and the BLAS it was built against."""
    info = {"numpy": np.__version__, "blas": "unknown"}
    try:
        cfg = np.show_config(mode="dicts")
        b = cfg["Build Dependencies"]["blas"]
        info["blas"] = "%s %s (%s)" % (b.get("name"), b.get("version"),
                                        b.get("openblas configuration", "").strip())
    except Exception:  # noqa: BLE001
        pass
    return info
