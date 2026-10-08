"""P2 experiment kernels that run straight on the substrate (no interpreter):
primitive cleanup per backend (E1) and the β collapse curve.

Both go through the backends' public methods (`clean`, `nearest`), so they
measure the implementation in hdc.py, not a re-derivation of it. Called from
vsc.experiments.tasks.
"""

import numpy as np

import hdc

_primed = {}


def _rows(dim, load, seed):
    """A codebook of `load` random unitary atoms, cached for the last
    (dim, load, seed) only."""
    key = (int(dim), int(load), int(seed))
    if key not in _primed:
        _primed.clear()
        hdc.collect()
        space = hdc.Space(int(dim), int(seed))
        mem = hdc.make_memory(space, "codebook", capacity=int(load))
        mem.add_random(int(load), 1)
        _primed[key] = (space, mem)
    return _primed[key]


def _like(space, base, backend, opts):
    """A memory of `backend` holding the same rows as `base` (shared arrays)."""
    mem = hdc.make_memory(space, backend, capacity=1, **opts)
    mem.M, mem.V, mem.labels, mem.n = base.M, base.V, base.labels, base.n
    return mem


def probe_trials(dim, load, seed, sigma, trials, backend, opts):
    """Probe a memory of `load` atoms with one of them under probe noise σ.
    The readout y = clean(probe) is correct if its nearest stored row, taken
    noise-free, is the probed atom; `cos` is the mean similarity of y to it.
    For a snapping backend y is a stored row, so this is `nearest`."""
    space, base = _rows(dim, load, seed)
    space.configure(probe_noise=float(sigma))
    mem = _like(space, base, str(backend), dict(opts))
    rng = np.random.default_rng([int(seed), int(round(float(sigma) * 1000)), 7])
    hits, cos = 0, 0.0
    for _ in range(int(trials)):
        i = int(rng.integers(base.n))
        x = base.M[i].astype(np.float64)
        y = mem.clean(x)
        a = base.M[: base.n] @ hdc.Memory._unit(y)
        hits += int(np.argmax(a)) == i
        ny = np.linalg.norm(y)
        cos += float(x @ y / ny) if ny > 0 else 0.0
    space.configure()
    return {"successes": hits, "trials": int(trials), "cos": cos / int(trials)}


_hard = {}


def _hard_rows(dim, load, seed, cluster, csim, damage):
    """A codebook for the conditions where iterating could matter.

    cluster > 1: rows come in groups of `cluster` look-alikes with pairwise
    similarity `csim` (row = √csim·c + √(1−csim)·r, c shared by the group).
    damage > 0: every stored row then gets noise of norm `damage` once and
    is renormalised (as Memory.damage), so the probe's clean atom is no
    longer stored. Returns (space, memory of the stored rows, clean rows)."""
    space = hdc.Space(int(dim), int(seed))
    rng = np.random.default_rng([int(seed), int(cluster), int(round(float(csim) * 1000)), 19])
    d, n, g = int(dim), int(load), max(1, int(cluster))

    def unit(X):
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    R = unit(rng.standard_normal((n, d)))
    if g > 1:
        C = unit(rng.standard_normal(((n + g - 1) // g, d)))
        R = unit(np.sqrt(csim) * C[np.arange(n) // g] + np.sqrt(1 - csim) * R)
    S = R
    if damage > 0:
        S = unit(R + (float(damage) / np.sqrt(d)) * rng.standard_normal((n, d)))
    mem = hdc.make_memory(space, "codebook", capacity=n)
    mem.M[:n] = S.astype(np.float32)
    mem.labels[:n] = 1
    mem.n = n
    return space, mem, R


def probe_trials_hard(dim, load, seed, sigma, trials, backend, opts,
                      cluster=1, csim=0.0, damage=0.0):
    """probe_trials on look-alike rows and/or damaged rows. The probe is the
    clean atom plus probe noise; correct if the readout's nearest *stored*
    row is the probed one's (identity is what the interpreter needs). For
    isotropic noise and equal-norm rows the argmax is still the
    maximum-likelihood decision here, so the table remains optimal; what
    this measures is whether iteration or blending lose more when rows are
    correlated or damaged."""
    key = (int(dim), int(load), int(seed), int(cluster), float(csim), float(damage))
    if key not in _hard:
        _hard.clear()
        hdc.collect()
        _hard[key] = _hard_rows(dim, load, seed, cluster, csim, damage)
    space, base, R = _hard[key]
    space.configure(probe_noise=float(sigma))
    mem = _like(space, base, str(backend), dict(opts))
    rng = np.random.default_rng([int(seed), int(round(float(sigma) * 1000)), 23])
    hits = 0
    for _ in range(int(trials)):
        i = int(rng.integers(base.n))
        y = mem.clean(R[i])
        a = base.M[: base.n] @ hdc.Memory._unit(y)
        hits += int(np.argmax(a)) == i
    space.configure()
    return {"successes": hits, "trials": int(trials)}


_classical = {}


def classical_trials(dim, load, seed, sigma, trials, steps=10):
    """E1 baseline: a classical Hopfield network (Hopfield 1982). `load`
    random bipolar (±1) patterns, Hebbian weights W = XᵀX/D with zero
    diagonal, synchronous sign updates for `steps` steps. The probe is a
    pattern plus Gaussian noise of norm σ·|x| (the same relative noise as
    the other backends' probe noise). Correct if the final state's nearest
    pattern is the probed one; `exact` is the rate of perfect recall.
    Capacity is ≈ 0.14·D patterns, so it needs no HRR vectors at all but
    also cannot hold much."""
    key = (int(dim), int(load), int(seed))
    if key not in _classical:
        _classical.clear()
        rng = np.random.default_rng([int(seed), 13])
        X = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(int(load), int(dim)))
        W = X.T @ X / np.float32(dim)
        np.fill_diagonal(W, 0.0)
        _classical[key] = (X, W)
    X, W = _classical[key]
    d = X.shape[1]
    rng = np.random.default_rng([int(seed), int(round(float(sigma) * 1000)), 17])
    hits, exact = 0, 0
    for _ in range(int(trials)):
        i = int(rng.integers(X.shape[0]))
        s = X[i] + np.float32(sigma) * rng.standard_normal(d).astype(np.float32)
        s = np.where(s >= 0, 1.0, -1.0).astype(np.float32)
        for _ in range(int(steps)):
            t = np.where(W @ s >= 0, 1.0, -1.0).astype(np.float32)
            if np.array_equal(t, s):
                break
            s = t
        hits += int(np.argmax(X @ s)) == i
        exact += bool(np.array_equal(s, X[i]))
    return {"successes": hits, "trials": int(trials), "exact": exact / int(trials)}


def collapse(dim, n_items, k, seed, sigma, beta, iters, weights="equal"):
    """The β collapse curve. N atoms in a soft modern Hopfield memory; the
    probe is a superposition of k of them (equal weights, or 1 … 0.4) plus
    noise of norm σ·|signal|. After `iters` softmax steps, fit the output on
    the k items by least squares:

      pr     participation ratio (Σc)²/Σc² of the clipped coefficients, the
             effective number of items that survive
      noise  1 − |projection|²/|y|²: energy outside the k items' span
      top    share of the largest coefficient
      tv     total variation between the coefficient shares and the weights

    This is W0's q4 measurement, run on the hdc.py backend."""
    space, base = _rows(dim, n_items, seed)
    rng = np.random.default_rng([int(seed), int(k), int(round(float(sigma) * 100)), 11])
    rows = rng.choice(base.n, int(k), replace=False)
    w = np.ones(int(k)) if weights == "equal" else np.linspace(1.0, 0.4, int(k))
    w = w / w.sum()
    X = base.M[rows].astype(np.float64)
    p = w @ X
    p = p + float(sigma) * np.linalg.norm(p) * rng.standard_normal(base.dim) / np.sqrt(base.dim)
    mem = _like(space, base, "mhn", {"beta": float(beta), "iters": int(iters), "mode": "soft"})
    y = mem.clean(p)
    c, *_ = np.linalg.lstsq(X.T, y, rcond=None)
    proj = X.T @ c
    cp = np.clip(c, 0, None)
    tot = cp.sum()
    pr = float(tot ** 2 / (cp @ cp)) if tot > 0 else 0.0
    share = cp / tot if tot > 0 else np.zeros_like(cp)
    return {"pr": pr,
            "noise": float(1 - (proj @ proj) / (y @ y)) if y @ y > 0 else 1.0,
            "top": float(share.max()),
            "tv": float(0.5 * np.abs(share - w).sum())}
