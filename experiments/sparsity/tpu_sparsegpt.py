"""
tpu_sparsegpt.py
----------------
The SparseGPT solver (Frantar & Alistarh, 2023) in JAX, for Cloud TPU.

Model-agnostic: ``sparsegpt_solve(W, H)`` returns the pruned weight matrix
given its input Hessian. It mirrors ``sparsegpt.py`` (the untouched CPU
reference) step by step: dead columns, 1% mean-diagonal damping, Cholesky,
Hinv = Linv^T Linv, upper Cholesky of Hinv, then the 128-column lazy block
sweep in which, for N:M, the (M - N) lowest w^2 / d^2 scores of each group of
M columns are pruned on the weights *as updated by the preceding columns*,
followed by rank-1 OBS updates and the lazy trailing update. Float32 with
HIGHEST matmul precision throughout. Given the same H and W it reproduces the
CPU reference to float32 rounding (100% of masks; tests/test_tpu_sparsegpt_solver.py).

``tpu_gemma4.calibrate_layers_tpu`` is the Gemma-4 calibrator built on it.

Attribution Notice:
    Adapted from SparseGPT (https://github.com/IST-DASLab/sparsegpt,
    Apache-2.0) via this package's ``sparsegpt.py``. Modifications include:
      - JAX port: jit-compiled Cholesky / triangular solves and a
        ``lax.fori_loop`` column sweep so the solve runs on Cloud TPU.
      - N:M semantics: N weights kept per group of M (as in ``sparsegpt.py``).
      - Rank-based group top-k with lowest-index tie-break; a precomputed
        (U, dead, ok) may be shared by matrices with a common input.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from jax import lax

F32 = jnp.float32
BF16 = jnp.bfloat16
_HI = lax.Precision.HIGHEST

# Compiling the first pruned layer of each attention type takes ~1.5-2.5 min
# (a few dozen kernels, one per matrix shape). A persistent cache makes later
# runs start at full speed. Override with TPU_SPARSEGPT_JAX_CACHE ("" = off).
_JAX_CACHE = os.environ.get(
    "TPU_SPARSEGPT_JAX_CACHE",
    "/mnt/pd/jax_cache/tpu_sparsegpt" if os.path.isdir("/mnt/pd") else "")
if _JAX_CACHE and not jax.config.jax_compilation_cache_dir:
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

# --------------------------------------------------------------------------
# torch <-> numpy/JAX conversion (bfloat16 goes through an int16 view)
# --------------------------------------------------------------------------


def _to_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().to("cpu").contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _to_torch(a: np.ndarray) -> torch.Tensor:
    a = np.ascontiguousarray(a)
    if a.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(a.copy())


def _gram(x: jnp.ndarray) -> jnp.ndarray:
    """sum_t x_t x_t^T in float32 for a (tokens, dim) activation block."""
    return lax.dot_general(x,
                           x, (((0, ), (0, )), ((), ())),
                           precision=_HI if x.dtype == F32 else None,
                           preferred_element_type=F32)


# --------------------------------------------------------------------------
# OBS solver (one weight matrix)
# --------------------------------------------------------------------------

_TRI_BASE = 2048


def _lower_tri_inv(L: jnp.ndarray) -> jnp.ndarray:
    """Inverse of a lower-triangular matrix by recursive 2x2 blocking.

    inv([[A, 0], [B, C]]) = [[inv(A), 0], [-inv(C) B inv(A), inv(C)]].
    XLA's TPU triangular_solve against a 21504-wide identity needs ~150 GB of
    temporaries; this form is matmul-bound and needs a few copies of L.
    """
    n = L.shape[0]
    if n <= _TRI_BASE:
        return lax.linalg.triangular_solve(L,
                                           jnp.eye(n, dtype=L.dtype),
                                           left_side=True,
                                           lower=True)
    h = ((n // 2 + 127) // 128) * 128
    Ai = _lower_tri_inv(L[:h, :h])
    Ci = _lower_tri_inv(L[h:, h:])
    Bi = -jnp.matmul(
        jnp.matmul(Ci, L[h:, :h], precision=_HI), Ai, precision=_HI)
    top = jnp.concatenate([Ai, jnp.zeros((h, n - h), L.dtype)], axis=1)
    bot = jnp.concatenate([Bi, Ci], axis=1)
    return jnp.concatenate([top, bot], axis=0)


@functools.partial(jax.jit, static_argnames=("percdamp", ))
def prepare_hinv(H: jnp.ndarray, percdamp: float):
    """Damped upper Cholesky factor of H^-1, as in SparseGPT.fasterprune.

    Returns (U, dead, ok): U upper triangular with H^-1 = U^T U, the mask of
    dead input columns (zero Hessian diagonal), and whether U is finite (a
    failed Cholesky yields NaNs in JAX instead of raising as torch does).
    """
    n = H.shape[0]
    eye = jnp.eye(n, dtype=F32)
    dead = jnp.diagonal(H) == 0
    H = H + eye * dead.astype(F32)  # H[dead, dead] = 1
    damp = percdamp * jnp.mean(jnp.diagonal(H))
    H = H + eye * damp
    L = jnp.linalg.cholesky(H)
    Linv = _lower_tri_inv(L)
    Hinv = jnp.matmul(Linv.T, Linv, precision=_HI)
    U = jnp.triu(jnp.linalg.cholesky(Hinv).T)
    return U, dead, jnp.all(jnp.isfinite(U))


def _group_prune_mask(tmp: jnp.ndarray, n_prune: int) -> jnp.ndarray:
    """True for the n_prune smallest entries of each row (ties: lower index)."""
    k = tmp.shape[1]
    a = tmp[:, None, :]  # candidate j
    b = tmp[:, :, None]  # element k
    before = jnp.arange(k)[None, :] < jnp.arange(k)[:, None]  # j < k
    rank = jnp.sum((a < b) | ((a == b) & before[None]), axis=-1)
    return rank < n_prune


def _block_solve(W1, Hinv1, prunen, prunem, sparsity):
    """One lazy block of SparseGPT: returns (Q1, Err1, losses)."""
    rows, count = W1.shape
    d = jnp.diagonal(Hinv1)
    if prunen == 0:
        tmp = W1**2 / (d[None, :])**2
        thresh = jnp.sort(tmp.ravel())[int(tmp.size * sparsity)]
        fixed_mask = tmp <= thresh
        group = 4 if count % 4 == 0 else 1
    else:
        if count % prunem:
            raise ValueError(
                f"block of {count} columns is not a multiple of prunem={prunem}"
            )
        fixed_mask = None
        group = prunem

    def body(g, carry):
        W1, Q1, Err1, losses = carry
        c = g * group
        if prunen == 0:
            pmask = lax.dynamic_slice(fixed_mask, (0, c), (rows, group))
        else:
            wg = lax.dynamic_slice(W1, (0, c), (rows, group))
            dg = lax.dynamic_slice(d, (c, ), (group, ))
            pmask = _group_prune_mask(wg**2 / (dg[None, :])**2,
                                      prunem - prunen)
        qs, errs = [], []
        for k in range(group):
            col = c + k
            w = lax.dynamic_slice(W1, (0, col), (rows, 1))[:, 0]
            dk = lax.dynamic_index_in_dim(d, col, keepdims=False)
            q = jnp.where(pmask[:, k], 0.0, w)
            err = (w - q) / dk
            losses = losses + (w - q)**2 / dk**2 / 2
            hrow = lax.dynamic_slice(Hinv1, (col, 0), (1, count))
            # Hinv1 is upper triangular, so columns < col receive exact zeros.
            W1 = W1 - err[:, None] * hrow
            qs.append(q)
            errs.append(err)
        Q1 = lax.dynamic_update_slice(Q1, jnp.stack(qs, 1), (0, c))
        Err1 = lax.dynamic_update_slice(Err1, jnp.stack(errs, 1), (0, c))
        return W1, Q1, Err1, losses

    zeros = jnp.zeros_like(W1)
    _, Q1, Err1, losses = lax.fori_loop(
        0, count // group, body, (W1, zeros, zeros, jnp.zeros((rows, ), F32)))
    return Q1, Err1, losses


@functools.partial(jax.jit,
                   static_argnames=("prunen", "prunem", "blocksize",
                                    "sparsity"))
def obs_prune(W, U, dead, prunen: int, prunem: int, blocksize: int,
              sparsity: float):
    """SparseGPT column sweep. W: (rows, cols) float32; U from prepare_hinv.

    Returns (pruned W in float32, summed OBS loss).
    """
    rows, cols = W.shape
    W = jnp.where(dead[None, :], 0.0, W)
    col_ids = jnp.arange(cols)

    def step(W, i1, count):
        W1 = lax.dynamic_slice(W, (0, i1), (rows, count))
        Hinv1 = lax.dynamic_slice(U, (i1, i1), (count, count))
        Q1, Err1, l1 = _block_solve(W1, Hinv1, prunen, prunem, sparsity)
        W = lax.dynamic_update_slice(W, Q1, (0, i1))
        # W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]; masked rows leave W[:, :i2]
        # (already final) untouched because they add exact zeros.
        Hrow = lax.dynamic_slice(U, (i1, 0), (count, cols))
        Hrow = jnp.where(col_ids[None, :] >= i1 + count, Hrow, 0.0)
        W = W - jnp.matmul(Err1, Hrow, precision=_HI)
        return W, jnp.sum(l1)

    nfull = cols // blocksize

    def full_block(b, carry):
        W, loss = carry
        W, block_loss = step(W, b * blocksize, blocksize)
        return W, loss + block_loss

    W, loss = lax.fori_loop(0, nfull, full_block, (W, jnp.zeros((), F32)))
    tail = cols - nfull * blocksize
    if tail:
        W, block_loss = step(W, nfull * blocksize, tail)
        loss = loss + block_loss
    return W, loss


def sparsegpt_solve(W: jnp.ndarray,
                    H: jnp.ndarray,
                    sparsity: float = 0.0,
                    prunen: int = 0,
                    prunem: int = 0,
                    blocksize: int = 128,
                    percdamp: float = 0.01,
                    hinv=None,
                    check: bool = True):
    """Prune one float32 weight matrix given its Hessian. Returns (W, loss).

    ``hinv`` may carry a precomputed (U, dead, ok) for matrices that share an
    input (q/k/v, gate/up): their Hessians are identical, so is their inverse.
    ``check=False`` skips the (blocking) Cholesky-success check; the caller
    must then check ``hinv[2]`` itself.
    """
    if hinv is None:
        hinv = prepare_hinv(H, float(percdamp))
    U, dead, ok = hinv
    if check:
        _check_ok(ok)
    return obs_prune(W, U, dead, int(prunen), int(prunem), int(blocksize),
                     float(sparsity))


def _check_ok(ok) -> None:
    if not bool(ok):
        raise RuntimeError(
            "SparseGPT: Cholesky of the damped Hessian failed (non-finite "
            "factor); increase percdamp")


def check_nm(prunen: int, prunem: int, blocksize: int) -> None:
    if prunen:
        if not 0 < prunen < prunem:
            raise ValueError(
                f"need 0 < prunen < prunem, got {prunen}:{prunem}")
        if blocksize % prunem:
            raise ValueError(
                f"blocksize {blocksize} must be a multiple of prunem {prunem}")
