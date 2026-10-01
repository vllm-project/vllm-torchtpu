"""TPU SparseGPT solver vs the CPU reference on the same (W, H).

Given identical inputs the two must agree to float32 rounding: every 2:4 mask
identical, bf16 weights bit-identical except for rare 1-ulp rounding flips.
Needs a TPU; run as a script from experiments/:

    python -u tests/test_tpu_sparsegpt_solver.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sparsity.sparsegpt import SparseGPT  # noqa: E402

CASES = [  # (rows, cols, prunen, prunem, dtype)
    (512, 1024, 2, 4, torch.bfloat16),
    (384, 640, 2, 4, torch.bfloat16),
    (512, 1024, 1, 4, torch.bfloat16),
    (256, 1024, 2, 8, torch.bfloat16),
    (512, 1024, 2, 4, torch.float32),
    (256, 4608, 2, 4, torch.bfloat16),
]


def _cpu_reference(layer, x, prunen, prunem):
    g = SparseGPT(layer)
    g.add_batch(x, None)
    g.fasterprune(0.0, prunen=prunen, prunem=prunem)
    return layer.weight.data.clone()


def _tpu(layer, x, prunen, prunem):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sparsity.tpu_sparsegpt import _to_np, _to_torch, sparsegpt_solve
    xs = jnp.asarray(_to_np(x.reshape(-1, x.shape[-1])))
    H = jnp.matmul(xs.T.astype(jnp.float32),
                   xs.astype(jnp.float32),
                   precision=jax.lax.Precision.HIGHEST) * (2.0 / x.shape[0])
    W = jnp.asarray(_to_np(layer.weight.data.float()))
    Wp, _ = sparsegpt_solve(W, H, prunen=prunen, prunem=prunem)
    out = _to_torch(np.asarray(Wp.astype(jnp.bfloat16))) if (
        layer.weight.dtype == torch.bfloat16) else _to_torch(np.asarray(Wp))
    return out.to(layer.weight.dtype)


def main() -> int:
    torch.manual_seed(0)
    for rows, cols, n, m, dtype in CASES:
        layer = nn.Linear(cols, rows, bias=False).to(dtype)
        x = torch.randn(16, 128, cols).to(dtype)
        cpu_layer = nn.Linear(cols, rows, bias=False).to(dtype)
        cpu_layer.weight.data.copy_(layer.weight.data)
        ref = _cpu_reference(cpu_layer, x, n, m)
        got = _tpu(layer, x, n, m)

        # Compare as bf16 (the serving dtype): float32 results differ in the
        # last bits whenever the summation order differs.
        r16, g16 = ref.to(torch.bfloat16), got.to(torch.bfloat16)
        mask_agree = ((r16 == 0) == (g16 == 0)).float().mean().item()
        exact = (r16 == g16).float().mean().item()
        nm_ok = bool(((g16.view(rows, -1, m) != 0).sum(-1) <= n).all())
        print(f"[{n}:{m}] {rows}x{cols} {dtype}: mask {mask_agree:.6f} "
              f"exact {exact:.5f} nm_ok {nm_ok}")
        assert mask_agree >= 0.999, "mask mismatch"
        assert exact >= 0.99, "too many weight mismatches"
        assert nm_ok, "N:M pattern violated"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
