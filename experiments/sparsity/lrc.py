"""
lrc.py
------
Low-Rank Compensated (LRC) N:M structured weight pruning on top of SparseGPT.

Mathematical Formulation
------------------------
Minimizing ``||W - S - U @ V||_F`` in weight space is the wrong objective; the
right objective is the error in the layer output under the distribution of
input activations ``x`` that the layer actually sees:

    E_x ||(W - S - U @ V) x||^2 = tr((W - S - U @ V) H (W - S - U @ V)^T)
                                = ||(W - S - U @ V) L||_F^2,

where ``H = E[x x^T] + damp * I = L @ L^T`` is the Cholesky-factored empirical
second-moment matrix (``L`` lower-triangular, ``damp = percdamp * mean(diag H)``).

Thus LRC solves the joint problem in the ``L``-metric:

    minimize    ||(W - S - U @ V) @ L||_F^2
    over        S in S_{N:M},  U in R^{d_out x k},  V in R^{k x d_in}.

Exact Rank-k Correction for Fixed S (Eckart-Young in the L-metric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For fixed ``S``, substitute ``N = (U @ V) @ L`` on the whitened residual
``R_tilde = (W - S) @ L``. Since ``L`` is invertible (guaranteed by the ridge
damping), ``U @ V -> N`` is a bijection on rank-<=k matrices, reducing the
subproblem to ``min_{rank(N) <= k} ||R_tilde - N||_F^2``. By the
Eckart-Young-Mirsky theorem, the exact minimizer is the truncated SVD
``N* = SVD_k(R_tilde)`` (with residual distortion ``sum_{j > k} sigma_j^2``),
and un-whitening gives:

    (U @ V)* = SVD_k((W - S) @ L) @ L^-1.

Joint Alternating Optimization of S and U @ V (T Iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pruning first and fitting the residual second is suboptimal: the pruning step
should know that a rank-k correction is coming and leave behind a
low-rank-friendly residual. Starting from ``U = 0, V = 0``, each iteration
``t = 0 .. T - 1`` alternates two non-increasing moves on the same objective
``||(W - S - U @ V) @ L||_F^2``:

1. **S-step (fix U @ V, prune W - U @ V with SparseGPT in the L-metric):**
   Form ``W_eff = W - U @ V`` and run SparseGPT blockwise OBS pruning using
   the upper Cholesky factor of ``H^-1`` to obtain the N:M sparse matrix ``S``.

2. **UV-step (fix S, exact rank-k projection via Gram eigendecomposition):**
   Form the whitened residual ``R_tilde = (W - S) @ L``. Instead of a full
   rectangular SVD of ``R_tilde``, form the smaller symmetric PSD Gram matrix
   on ``min(d_out, d_in)`` and extract its top-k eigenvectors via
   ``torch.linalg.eigh`` (projecting directly onto the top-k singular subspace
   without dividing by singular values):
     - Tall/square (``d_out >= d_in``): ``G_in = R_tilde^T @ R_tilde``
       (``d_in x d_in``) has top-k eigenvectors ``V_k``. Setting
       ``U = R_tilde @ V_k`` and ``Vh_k = V_k^T`` gives
       ``U @ Vh_k = R_tilde @ (V_k @ V_k^T) = SVD_k(R_tilde)``.
     - Wide (``d_out < d_in``): ``G_out = R_tilde @ R_tilde^T``
       (``d_out x d_out``) has top-k eigenvectors ``U_k``. Setting ``U = U_k``
       and ``Vh_k = U_k^T @ R_tilde`` gives
       ``U @ Vh_k = (U_k @ U_k^T) @ R_tilde = SVD_k(R_tilde)``.
   Un-whiten the right factor by solving the lower-triangular system
   ``V @ L = Vh_k`` (i.e. ``V = Vh_k @ L^-1``) via ``solve_triangular``.

After ``T`` iterations, ``W_final = S + U @ V`` is written back to
``gpt.layer.weight.data`` so downstream layers calibrate against the
LRC-compensated activations and checkpoint serialization writes the resulting
tensor directly.
"""

import time
from typing import NamedTuple

import torch

from .sparsegpt import SparseGPT


class LrcResult(NamedTuple):
    """Outputs and diagnostic whitened Frobenius errors from ``lrc_prune``."""
    U: torch.Tensor
    V: torch.Tensor
    S: torch.Tensor
    error_sparsegpt: float
    error_lrc: float


def lrc_prune(
    gpt: SparseGPT,
    sparsity: float = 0.5,
    prunen: int = 2,
    prunem: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    T: int = 3,
    k: int = 128,
) -> LrcResult:
    """Prunes ``gpt.layer`` via T-step LRC alternating minimization.

    Args:
        gpt: A :class:`SparseGPT` instance whose ``H`` has been accumulated via
            ``add_batch`` across calibration sequences.
        sparsity: Unstructured target sparsity (used only when ``prunen == 0``).
        prunen: Number of weights *kept* per block of ``prunem`` columns.
        prunem: Block width for N:M structured sparsity (e.g., 4 for 2:4).
        blocksize: Column block size for SparseGPT OBS updates.
        percdamp: Relative diagonal dampening factor for ``H``.
        T: Number of alternating minimization iterations (S-step + UV-step).
        k: Target rank of the low-rank compensation factors ``U`` and ``V``.

    Returns:
        An :class:`LrcResult` containing ``(U, V, S, error_sparsegpt, error_lrc)``.
        Also updates ``gpt.layer.weight.data`` in place to ``S + U @ V``.
    """
    W_orig = gpt._clone_weight_matrix()
    tick = time.time()

    # Factorize dampened Hessian once:
    #   H + damp * I = L @ L^T  (L lower-triangular)
    #   Hinv = upper-Cholesky factor of (H + damp * I)^-1 for OBS updates
    L, Hinv = gpt._factorize_hessian(W_orig, percdamp=percdamp)

    k_eff = min(k, gpt.rows, gpt.columns)
    U = torch.zeros((gpt.rows, k_eff), device=gpt.dev, dtype=W_orig.dtype)
    V = torch.zeros((k_eff, gpt.columns), device=gpt.dev, dtype=W_orig.dtype)
    S = W_orig

    error_sparsegpt = 0.0
    error_lrc = 0.0

    for t in range(T):
        # Step 1 (S-step): Fix U @ V and prune W_eff = W_orig - U @ V with
        # SparseGPT blockwise OBS solver in the L-metric -> N:M sparse S.
        W_eff = (W_orig - U.matmul(V)).clone()
        S, _ = gpt._obs_prune_blocks(
            W_eff,
            Hinv,
            sparsity=sparsity,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
        )

        # Whitened residual in L-metric: R_tilde = (W_orig - S) @ L.
        # At iteration t = 0 (where U @ V = 0), ||R_tilde||_F is the uncompensated
        # SparseGPT whitened reconstruction error.
        R_tilde = (W_orig - S).matmul(L)
        if t == 0:
            error_sparsegpt = torch.norm(R_tilde, p="fro").item()

        # Step 2 (UV-step): Fix S and compute SVD_k(R_tilde) @ L^-1 via
        # Gram-matrix eigendecomposition on min(d_out, d_in), projecting onto
        # the top-k singular subspace without dividing by singular values.
        if k_eff > 0:
            if gpt.rows >= gpt.columns:
                # Tall or square (d_out >= d_in): G_in = R_tilde^T @ R_tilde [d_in, d_in]
                G = R_tilde.t().matmul(R_tilde)
                eig_vals, eig_vecs = torch.linalg.eigh(G)
                idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                V_k = eig_vecs[:, idx]
                U = R_tilde.matmul(V_k)
                Vh_k = V_k.t()
            else:
                # Wide (d_out < d_in): G_out = R_tilde @ R_tilde^T [d_out, d_out]
                G = R_tilde.matmul(R_tilde.t())
                eig_vals, eig_vecs = torch.linalg.eigh(G)
                idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                U = eig_vecs[:, idx]
                Vh_k = U.t().matmul(R_tilde)

            # Un-whiten right factor: solve V @ L = Vh_k <=> V = Vh_k @ L^-1
            V = torch.linalg.solve_triangular(L, Vh_k, upper=False, left=False)

            R_lrc = (W_orig - S - U.matmul(V)).matmul(L)
            error_lrc = torch.norm(R_lrc, p="fro").item()
        else:
            error_lrc = error_sparsegpt

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("time %.2f" % (time.time() - tick))
    print("error SparseGPT: %.4f | LRC (T=%d, k=%d): %.4f" %
          (error_sparsegpt, T, k, error_lrc))

    # Materialize W_final = S + U @ V back into layer weight tensor
    gpt._write_weight_matrix(S + U.matmul(V))

    return LrcResult(
        U=U,
        V=V,
        S=S,
        error_sparsegpt=error_sparsegpt,
        error_lrc=error_lrc,
    )


def make_lrc_pruner(
    sparsity: float = 0.5,
    prunen: int = 2,
    prunem: int = 4,
    blocksize: int = 128,
    percdamp: float = 0.01,
    T: int = 3,
    k: int = 128,
):
    """Build a per-layer matrix pruner callback prune_fn(gpt) for gemma4_sequential."""

    def _prune(gpt: SparseGPT) -> LrcResult:
        return lrc_prune(
            gpt,
            sparsity=sparsity if sparsity > 0 else 0.5,
            prunen=prunen,
            prunem=prunem,
            blocksize=blocksize,
            percdamp=percdamp,
            T=T,
            k=k,
        )

    return _prune
