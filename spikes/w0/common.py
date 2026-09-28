"""W0 spike: shared prototype backends for superposition ("many worlds").

Nothing here is production code. It sits on top of resources/vsc/hdc.py
(Space, Numbers) and adds, locally, the memory backends the spike compares:

  codebook   hardmax(K p) V          -- the interpreter's current memory
  linear     Vᵀ (K p)                -- beta -> 0 tangent of attention
  softmax    Vᵀ softmax(beta K p̂)   -- modern Hopfield / attention head
  thresh     Vᵀ [K p̂ > tau](K p)    -- threshold-linear separation

and a world readout (orthogonal matching pursuit + least squares), which is
what `(worlds x)` would do.
"""

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "resources", "vsc"))
import hdc  # noqa: E402

FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(HERE, "out")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# reference categorical palette (dataviz skill), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
     "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)


# -- vectors ---------------------------------------------------------------

def unit_rows(rng, n, d, dtype=np.float32):
    """n random unit vectors (Gaussian, normalised): atoms and pointers."""
    x = rng.standard_normal((n, d)).astype(dtype)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def unitary(rng, d):
    spec = np.exp(1j * rng.uniform(-np.pi, np.pi, d // 2 + 1))
    spec[0] = 1.0
    if d % 2 == 0:
        spec[-1] = 1.0
    return np.fft.irfft(spec, n=d)


def bind(a, b):
    d = a.shape[-1]
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=d)


def unbind(a, c):
    d = c.shape[-1]
    return np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(c), n=d)


def nu(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def z_floor(n):
    """Detection z-score for a max over n Gaussian distractors, with margin."""
    return np.sqrt(2 * np.log(max(n, 2))) + 1.0


# -- heteroassociative memories: p -> trace --------------------------------

class HeteroMemory:
    """Rows K (keys/pointers) -> V (traces). `read(p, mode)` returns the
    (unnormalised) trace blend and the row coefficients it used."""

    def __init__(self, K, V):
        self.K, self.V = K, V

    def coeffs(self, p, mode, beta=None, tau_z=None):
        s = self.K @ p.astype(self.K.dtype)
        n = np.linalg.norm(p)
        if mode == "codebook":
            a = np.zeros_like(s)
            i = int(np.argmax(s))
            a[i] = n  # keep the probe's scale
            return a
        if mode == "linear":
            return s
        if mode == "softmax":
            z = beta * s / n
            z = z - z.max()
            e = np.exp(z)
            return n * e / e.sum()
        if mode == "thresh":
            d = self.K.shape[1]
            tau = (tau_z if tau_z is not None else z_floor(len(s))) / np.sqrt(d)
            return np.where(np.abs(s) / n > tau, s, 0.0)
        if mode == "proj":
            # threshold for the support, then least squares on it: the
            # orthogonal projector onto the detected worlds (idempotent)
            d = self.K.shape[1]
            tau = (tau_z if tau_z is not None else z_floor(len(s))) / np.sqrt(d)
            S = np.nonzero(np.abs(s) / n > tau)[0]
            a = np.zeros(len(s))
            if len(S):
                a[S], *_ = np.linalg.lstsq(self.K[S].T.astype(np.float64), p, rcond=None)
            return a
        raise ValueError(mode)

    def read(self, p, mode, **kw):
        a = self.coeffs(p, mode, **kw)
        nz = np.nonzero(a)[0]
        if len(nz) < len(a) // 4:
            return (a[nz] @ self.V[nz]).astype(np.float64), a
        return (a @ self.V).astype(np.float64), a


# -- world readout: which codebook items, with which weights ---------------

def worlds(x, C, kmax=256, z=None, idx=None):
    """Orthogonal matching pursuit of x over codebook rows C.
    Returns (support indices, least-squares coefficients). Stops when the
    residual's best match is indistinguishable from chance (z/√D).
    `idx` restricts the search to a candidate subset (optional)."""
    d = C.shape[1]
    z = z_floor(C.shape[0]) if z is None else z
    x = x.astype(np.float64)
    r = x.copy()
    S, w = [], np.zeros(0)
    Cf = C if idx is None else C[idx]
    for _ in range(kmax):
        rn = np.linalg.norm(r)
        if rn < 1e-9 * np.linalg.norm(x):
            break
        s = Cf @ r.astype(Cf.dtype)
        i = int(np.argmax(np.abs(s)))
        if abs(s[i]) / rn < z / np.sqrt(d) or i in S:
            break
        S.append(i)
        A = Cf[S].astype(np.float64)
        w, *_ = np.linalg.lstsq(A.T, x, rcond=None)
        r = x - A.T @ w
    S = np.array(S, dtype=int)
    if idx is not None and len(S):
        S = np.asarray(idx)[S]
    return S, w


def as_dist(S, w):
    """{index: probability} from signed coefficients (negatives dropped)."""
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    t = w.sum()
    return {int(i): float(x / t) for i, x in zip(S, w) if x > 0} if t > 0 else {}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def participation(p):
    """Participation ratio 1/Σp² of a probability vector: the effective
    number of surviving worlds (k for k equal worlds, 1 for one)."""
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    t = p.sum()
    if t <= 0:
        return 0.0
    p = p / t
    return float(1.0 / np.sum(p * p))
