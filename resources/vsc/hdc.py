"""HDC substrate: holographic reduced representations (Plate, 1995) and
lookup-table cleanup memories, as used by Tomkins-Flanagan & Kelly (2024).

Everything numeric lives here. The Clojure side only ever holds opaque
references to numpy vectors and asks this module to combine, compare and
clean them up.
"""

import os

# Matrix-vector products here are small; BLAS thread pools only burn cores.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402


class Space:
    """An HRR vector space R^dim with circular convolution as the product."""

    def __init__(self, dim=4096, seed=0):
        self.dim = int(dim)
        self.rng = np.random.default_rng(int(seed))

    # -- atoms --------------------------------------------------------------

    def unitary(self):
        """A random unitary vector: every Fourier coefficient has modulus 1,
        so binding with it is exactly invertible and powers stay unit length."""
        d = self.dim
        spec = np.exp(1j * self.rng.uniform(-np.pi, np.pi, d // 2 + 1))
        spec[0] = self.rng.choice([-1.0, 1.0])
        if d % 2 == 0:
            spec[-1] = self.rng.choice([-1.0, 1.0])
        return np.fft.irfft(spec, n=d)

    def identity(self):
        """The unit of binding (a delta at index 0)."""
        v = np.zeros(self.dim)
        v[0] = 1.0
        return v

    # -- algebra ------------------------------------------------------------

    def bind(self, a, b):
        return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=self.dim)

    def unbind(self, a, c):
        """a ⊘ c: bind c with the involution (approximate inverse) of a.
        Exact when a is unitary."""
        return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(c), n=self.dim)

    def inverse(self, a):
        return np.fft.irfft(np.conj(np.fft.rfft(a)), n=self.dim)

    def power(self, a, k):
        """Integer power under binding: a ⊗ a ⊗ ... (k times)."""
        return np.fft.irfft(np.fft.rfft(a) ** int(k), n=self.dim)

    def bundle(self, *vs):
        return np.sum(vs, axis=0)

    def normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def sim(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))

    def degrade(self, v, amount):
        """Add Gaussian noise with norm `amount` * |v|."""
        noise = self.normalize(self.rng.normal(size=self.dim))
        return v + float(amount) * np.linalg.norm(v) * noise


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
        return np.fft.irfft(spec, n=self.space.dim)

    def _crt(self, residues):
        n = 0
        for r, p in zip(residues, self.moduli):
            q = self.P // p
            n += r * q * pow(q, -1, p)
        n %= self.P
        return n - self.P if n > self.half else n

    def read(self, v):
        """[n, similarity of v to B^n]."""
        X = np.fft.rfft(v)
        residues = [int(np.argmax((c @ X[band]).real))
                    for band, c in zip(self.bands, self.cands)]
        n = self._crt(residues)
        return [n, self.space.sim(v, self.vec(n))]


def peel(memories, v, theta, limit=256):
    """Explaining away over several memories at once: repeatedly clean up the
    residual of a superposition and subtract what was found. Returns
    [memory-index, row] pairs for constituents whose similarity to the
    original exceeds theta."""
    n = np.linalg.norm(v)
    if n == 0:
        return []
    r = (v / n).astype(np.float32)
    out = []
    for _ in range(int(limit)):
        best = (-np.inf, -1, -1)
        for j, mem in enumerate(memories):
            if mem.n == 0:
                continue
            a = mem.M[: mem.n] @ r
            i = int(np.argmax(a))
            if a[i] > best[0]:
                best = (float(a[i]), j, i)
        s, j, i = best
        if s < theta:
            break
        out.append([j, i])
        r = r - s * memories[j].M[i]
    return out


class Memory:
    """Lookup-table cleanup memory M(p) = hardmax(M p) M, optionally
    heteroassociative: row i of M (a key, or "pointer") is associated with
    row i of V (the trace it points to).

    Rows are stored unit length. Each row carries an integer label, which the
    host uses as the kind of the stored item (symbol, number, list, ...)."""

    def __init__(self, dim, capacity=4096):
        self.dim = int(dim)
        self.M = np.zeros((int(capacity), self.dim), dtype=np.float32)
        self.V = np.zeros((int(capacity), self.dim), dtype=np.float32)
        self.labels = np.full(int(capacity), -1, dtype=np.int32)
        self.n = 0

    def _grow(self):
        self.M = np.concatenate([self.M, np.zeros_like(self.M)])
        self.V = np.concatenate([self.V, np.zeros_like(self.V)])
        self.labels = np.concatenate([self.labels, np.full_like(self.labels, -1)])

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    def size(self):
        return self.n

    def nearest(self, v):
        """[index, similarity] of the best-matching key, or [-1, 0.0]."""
        if self.n == 0:
            return [-1, 0.0]
        a = self.M[: self.n] @ self._unit(v)
        i = int(np.argmax(a))
        return [i, float(a[i])]

    def add(self, v, label=0, dedupe=None):
        """Append key v; with `dedupe`, return the index of an existing key
        more similar than it instead of storing a copy."""
        if dedupe is not None and self.n > 0:
            i, s = self.nearest(v)
            if s > dedupe:
                return i
        if self.n >= self.M.shape[0]:
            self._grow()
        self.M[self.n] = self._unit(v)
        self.labels[self.n] = int(label)
        self.n += 1
        return self.n - 1

    def intern(self, key, trace, label, dedupe=0.999):
        """Associate a fresh pointer `key` with `trace`, unless an equal trace
        of the same label is already stored (hash-consing): then return that
        trace's index, so equal structures share one pointer."""
        t = self._unit(trace)
        if self.n > 0:
            a = self.V[: self.n] @ t
            a[self.labels[: self.n] != int(label)] = -np.inf
            i = int(np.argmax(a))
            if a[i] > dedupe:
                return i
        i = self.add(key, label)
        self.V[i] = t
        return i

    def put(self, i, v):
        self.M[int(i)] = self._unit(v)

    def get(self, i):
        return self.M[int(i)].astype(np.float64)

    def deref(self, p):
        """The trace associated with the key nearest to pointer p."""
        i, _ = self.nearest(p)
        return self.V[i].astype(np.float64)

    def recall(self, v):
        """[index, label, similarity] of the cleaned-up key."""
        i, s = self.nearest(v)
        return [i, int(self.labels[i]) if i >= 0 else -1, s]


class Cleanup:
    """The combined cleanup of a machine: the item memory M (atoms and
    pointers) plus the algorithmic readout of integers. Fusing the steps
    into one call keeps the host/substrate chatter down."""

    NUM = 4

    def __init__(self, space, mem, nums, theta=0.9):
        self.space, self.mem, self.nums, self.theta = space, mem, nums, theta

    def recognize(self, v):
        """[label, index, similarity]; index is n itself for integers."""
        i, s = self.mem.nearest(v)
        if s > self.theta:
            return [int(self.mem.labels[i]), i, s]
        n, sn = self.nums.read(v)
        if sn > s:
            return [self.NUM, n, sn]
        return [int(self.mem.labels[i]) if i >= 0 else -1, i, s]

    def cleanup(self, v):
        label, i, s = self.recognize(v)
        return self.nums.vec(i) if label == self.NUM else self.mem.get(i)

    def part(self, role, p):
        """M(role ⊘ M(p)): one field of the trace pointer p refers to."""
        return self.cleanup(self.space.unbind(role, self.mem.deref(p)))
