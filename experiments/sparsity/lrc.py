"""
lrc.py
------
Low-Rank Compensated (LRC) structured weight pruning on top of SparseGPT.

What LRC Is
~~~~~~~~~~~
Standard 2:4 (or N:M) structured sparsity constrains every contiguous group of
M columns in a weight matrix ``W`` (shape ``[d_out, d_in]``) to retain at most
N non-zero weights. Even after SparseGPT's Optimal Brain Surgeon (OBS)
second-order column updates, projections whose residual error is concentrated in
a small number of dominant singular directions suffer noticeable activation
reconstruction loss.

Low-Rank Compensation (LRC) decomposes the dense weight matrix as the sum of an
N:M structured-sparse matrix ``S`` and a low-rank adapter ``U @ V`` of rank
``k << min(d_out, d_in)``:

    W ≈ S + U @ V,   where S ∈ S_{N:M},  U ∈ R^{d_out x k},  V ∈ R^{k x d_in}.

Whitened Objective
~~~~~~~~~~~~~~~~~~
Given calibration inputs ``X``, SparseGPT accumulates the empirical input
Hessian ``H = (2 / nsamples) * X @ X^T + damp * I`` and computes its lower
Cholesky factor ``H = L @ L^T``. Because ``||A @ X||_F^2`` is proportional to
``tr(A @ H @ A^T) = ||A @ L||_F^2``, minimizing the layer-output reconstruction
error is equivalent to solving the whitened Frobenius problem:

    min_{S ∈ S_{N:M}, rank(U @ V) <= k}  ||(W - S - U @ V) @ L||_F^2.

Alternating Minimization (T Iterations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starting from ``U_0 = 0, V_0 = 0``, each iteration ``t = 0 .. T - 1`` alternates
between two exact/one-shot subproblems in the ``L``-whitened metric:

1. **S-step (SparseGPT OBS with fixed U @ V):**
   Form the residual target ``W_eff = W - U @ V`` and run SparseGPT's blockwise
   OBS solver using the upper Cholesky factor of ``H^-1`` to obtain the N:M
   sparse matrix ``S``.

2. **UV-step (Rank-k projection in whitened space):**
   Fix ``S`` and form the whitened residual ``R_tilde = (W - S) @ L``. By the
   Eckart-Young-Mirsky theorem, the optimal rank-k approximation to ``R_tilde``
   in Frobenius norm is its truncated SVD ``U_tilde_k @ Sigma_k @ Vh_k``.
   Instead of a full rectangular SVD, we compute the symmetric eigendecomposition
   (``torch.linalg.eigh``) of the Gram matrix on whichever dimension of
   ``R_tilde`` is smaller (``min(d_out, d_in)``):
     - Tall/square (``d_out >= d_in``): ``G = R_tilde^T @ R_tilde`` (``d_in x d_in``).
       Top-k eigenvectors form ``Vh_k``, and ``U = R_tilde @ Vh_k^T``.
     - Wide (``d_out < d_in``): ``G = R_tilde @ R_tilde^T`` (``d_out x d_out``).
       Top-k eigenpairs give ``(Sigma_k^2, U_tilde_k)``, ``U = U_tilde_k @ Sigma_k``,
       and ``Vh_k = Sigma_k^-1 @ U_tilde_k^T @ R_tilde``.
   Finally, we un-whiten the right factor by solving the lower-triangular system
   ``V @ L = Vh_k`` (i.e., ``V = Vh_k @ L^-1``) via ``solve_triangular``.

After ``T`` iterations, the compensated weight matrix ``W_final = S + U @ V`` is
written back to ``gpt.layer.weight.data`` so that downstream layers calibrate
against the LRC-compensated activations and checkpoint serialization can write
the resulting tensor directly.
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
        # SparseGPT's blockwise OBS solver in the H-metric -> N:M sparse S.
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

        # Step 2 (UV-step): Fix S and compute the best rank-k approximation to
        # R_tilde via Gram-matrix eigendecomposition on the smaller dimension,
        # then un-whiten V by solving V @ L = Vh_k.
        if k_eff > 0:
            if gpt.rows >= gpt.columns:
                # Tall or square matrix (d_out >= d_in):
                # Gram matrix G = R_tilde^T @ R_tilde has shape [d_in, d_in].
                G = R_tilde.t().matmul(R_tilde)
                eig_vals, eig_vecs = torch.linalg.eigh(G)
                idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                Vh_k = eig_vecs[:, idx].t()
                U = R_tilde.matmul(eig_vecs[:, idx])
                # Solve lower-triangular system V @ L = Vh_k <=> V = Vh_k @ L^-1
                V = torch.linalg.solve_triangular(L,
                                                  Vh_k,
                                                  upper=False,
                                                  left=False)
            else:
                # Wide matrix (d_out < d_in):
                # Gram matrix G = R_tilde @ R_tilde^T has shape [d_out, d_out].
                G = R_tilde.matmul(R_tilde.t())
                eig_vals, eig_vecs = torch.linalg.eigh(G)
                idx = torch.argsort(eig_vals, descending=True)[:k_eff]
                Sigma_k = torch.sqrt(torch.clamp(eig_vals[idx], min=0.0))
                U_k = eig_vecs[:, idx]
                inv_sigma = torch.where(Sigma_k > 1e-7, 1.0 / Sigma_k,
                                        torch.zeros_like(Sigma_k))
                Vh_k = inv_sigma.unsqueeze(1) * (U_k.t().matmul(R_tilde))
                U = U_k * Sigma_k.unsqueeze(0)
                # Solve lower-triangular system V @ L = Vh_k <=> V = Vh_k @ L^-1
                V = torch.linalg.solve_triangular(L,
                                                  Vh_k,
                                                  upper=False,
                                                  left=False)

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
