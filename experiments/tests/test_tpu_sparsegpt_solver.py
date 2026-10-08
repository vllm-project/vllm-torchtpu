"""JAX SparseGPT solver vs the torch CPU reference on the same (W, H).

``sparsity.tpu_sparsegpt.sparsegpt_solve`` must reproduce
``sparsity.sparsegpt.SparseGPT.fasterprune`` to float32 rounding when both
see the same weight matrix and the same Hessian: identical N:M masks and
weights that differ only by accumulated float32 round-off.

The JAX solver is backend-agnostic, so the equivalence tests run on the JAX
CPU backend by default (``JAX_PLATFORMS`` is forced to ``cpu`` unless set
explicitly) and need no TPU. ``test_solver_matches_reference_on_tpu`` is the
same comparison on a TPU backend; it is marked ``hardware_smoke`` and skips
unless the process was started with ``JAX_PLATFORMS=tpu``::

    JAX_PLATFORMS=tpu pytest -m hardware_smoke \
        tests/test_tpu_sparsegpt_solver.py
"""
import contextlib
import io
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SPARSEGPT_JAX_CACHE", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sparsity import tpu_sparsegpt  # noqa: E402
from sparsity.sparsegpt import SparseGPT  # noqa: E402

check_nm = tpu_sparsegpt.check_nm
prepare_hinv = tpu_sparsegpt.prepare_hinv
sparsegpt_solve = tpu_sparsegpt.sparsegpt_solve

# (rows, cols, prunen, prunem, n_dead_columns). 320 = 2 * 128 + 64 exercises
# the tail block that is shorter than ``blocksize``.
CASES = [
    pytest.param(256, 512, 2, 4, 0, id="2of4-256x512"),
    pytest.param(256, 512, 1, 4, 0, id="1of4-256x512"),
    pytest.param(256, 512, 2, 4, 8, id="2of4-256x512-dead8"),
    pytest.param(128, 320, 2, 4, 0, id="2of4-128x320-tail64"),
    pytest.param(128, 320, 1, 4, 4, id="1of4-128x320-tail64-dead4"),
]

# float32 weights ~ N(0, 1/cols) (|w| < 0.2); the two implementations differ
# only by float32 summation order in the OBS updates.
ATOL_W = 1e-4
RTOL_LOSS = 1e-3


def _reference(W: torch.Tensor, x: torch.Tensor, prunen: int, prunem: int):
    """torch SparseGPT on a float32 nn.Linear. Returns (pruned W, loss, H)."""
    rows, cols = W.shape
    layer = nn.Linear(cols, rows, bias=False)
    layer.weight.data.copy_(W)
    g = SparseGPT(layer)
    g.add_batch(x, None)
    H = g.H.clone()  # fasterprune consumes self.H in place
    # The reference prints "error <sum of Losses>" instead of returning it.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g.fasterprune(0.0, prunen=prunen, prunem=prunem)
    loss = float(buf.getvalue().split("error")[-1].split()[0])
    return layer.weight.data.clone(), loss, H


def _run_case(rows, cols, prunen, prunem, n_dead, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(rows, cols) / cols**0.5
    x = torch.randn(16, 64, cols)
    dead = torch.zeros(cols, dtype=torch.bool)
    if n_dead:
        dead_idx = torch.randperm(cols)[:n_dead]
        x[..., dead_idx] = 0.0
        dead[dead_idx] = True
    ref, ref_loss, H = _reference(W, x, prunen, prunem)
    Wp, loss = sparsegpt_solve(jnp.asarray(W.numpy()),
                               jnp.asarray(H.numpy()),
                               prunen=prunen,
                               prunem=prunem)
    got = torch.from_numpy(np.asarray(Wp))
    return W, ref, got, ref_loss, float(loss), dead


def _assert_equivalent(rows, cols, prunen, prunem, W, ref, got, dead):
    assert got.shape == ref.shape
    assert torch.isfinite(got).all()
    ref_mask, got_mask = ref != 0, got != 0
    mask_agree = (ref_mask == got_mask).float().mean().item()
    assert mask_agree == 1.0, f"mask agreement {mask_agree:.6f} < 1"
    kept = got_mask.view(rows, -1, prunem).sum(-1)
    assert bool((kept <= prunen).all()), "N:M pattern violated"
    assert bool((got[:, dead] == 0).all()), "dead columns must be zeroed"
    max_abs = (got - ref).abs().max().item()
    assert max_abs <= ATOL_W, f"max |jax - torch| = {max_abs:.3e}"
    assert (got != W).any()  # guards against a no-op solver


@pytest.mark.parametrize("rows,cols,prunen,prunem,n_dead", CASES)
def test_solver_matches_reference_on_cpu(rows, cols, prunen, prunem, n_dead):
    assert jax.default_backend() == "cpu"
    W, ref, got, ref_loss, loss, dead = _run_case(rows, cols, prunen, prunem,
                                                  n_dead)
    _assert_equivalent(rows, cols, prunen, prunem, W, ref, got, dead)
    assert ref_loss > 0.0
    assert abs(loss - ref_loss) <= RTOL_LOSS * ref_loss


def test_prepare_hinv_dead_columns_and_damping():
    n = 64
    torch.manual_seed(2)
    A = torch.randn(256, n)
    A[:, 3] = 0.0
    A[:, 17] = 0.0
    H = (A.t() @ A).numpy()
    U, dead, ok = prepare_hinv(jnp.asarray(H), 0.01)
    assert bool(ok)
    assert np.array_equal(np.asarray(dead), np.isin(np.arange(n), [3, 17]))
    U = np.asarray(U)
    assert np.isfinite(U).all()
    assert np.allclose(U, np.triu(U))
    # H^-1 = U^T U for the damped Hessian with H[dead, dead] = 1.
    Hd = H.copy()
    Hd[3, 3] = Hd[17, 17] = 1.0
    Hd = Hd + np.eye(n, dtype=np.float32) * (0.01 * np.diag(Hd).mean())
    np.testing.assert_allclose(U.T @ U @ Hd, np.eye(n), atol=1e-3)


def test_prepare_hinv_reports_failed_cholesky():
    H = -np.eye(16, dtype=np.float32)  # negative definite
    _, _, ok = prepare_hinv(jnp.asarray(H), 0.01)
    assert not bool(ok)
    with pytest.raises(RuntimeError, match="Cholesky"):
        sparsegpt_solve(jnp.zeros((8, 16), jnp.float32),
                        jnp.asarray(H),
                        prunen=2,
                        prunem=4)


def test_tail_block_must_be_multiple_of_m():
    # 128 + 6: the trailing block of 6 columns cannot hold 2:4 groups.
    torch.manual_seed(3)
    A = torch.randn(512, 134)
    H = jnp.asarray((A.t() @ A).numpy())
    W = jnp.asarray(torch.randn(8, 134).numpy())
    with pytest.raises(ValueError, match="not a multiple of prunem"):
        sparsegpt_solve(W, H, prunen=2, prunem=4)


def test_check_nm_rejects_bad_ratios():
    check_nm(0, 0, 128)
    check_nm(2, 4, 128)
    with pytest.raises(ValueError, match="0 < prunen < prunem"):
        check_nm(4, 4, 128)
    with pytest.raises(ValueError, match="multiple of prunem"):
        check_nm(2, 6, 128)


def test_torch_numpy_roundtrip_bfloat16():
    t = torch.randn(4, 8).to(torch.bfloat16)
    back = tpu_sparsegpt._to_torch(tpu_sparsegpt._to_np(t))
    assert back.dtype == torch.bfloat16
    assert torch.equal(back, t)


@pytest.mark.hardware_smoke
@pytest.mark.parametrize("rows,cols,prunen,prunem,n_dead", CASES)
def test_solver_matches_reference_on_tpu(rows, cols, prunen, prunem, n_dead):
    if jax.default_backend() != "tpu":
        pytest.skip("start pytest with JAX_PLATFORMS=tpu on a TPU VM")
    W, ref, got, ref_loss, loss, dead = _run_case(rows, cols, prunen, prunem,
                                                  n_dead)
    _assert_equivalent(rows, cols, prunen, prunem, W, ref, got, dead)
    assert abs(loss - ref_loss) <= RTOL_LOSS * ref_loss
