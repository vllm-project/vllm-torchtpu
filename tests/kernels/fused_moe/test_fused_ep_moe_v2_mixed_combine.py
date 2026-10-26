# SPDX-License-Identifier: Apache-2.0
"""TPU_DSV4_EP_MIXED_COMBINE_KERNEL: the mixed-bucket combine as one Pallas
call (mixed_combine_kernel.build_mixed_combine) must leave the layer's output
bit-exact against _combine_arrivals + the row_routes select it replaces, on
both mixed paths (the mixed plan kernel's and the XLA sharded plan's), and must
not engage at decode buckets.

The kernel's own tests (tests/kernels/fused_moe/mixed_plan/test_accuracy.py)
check it against a float64 oracle with a fault control at the served shapes.
"""

import jax
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2, layer
from vllm_torchtpu.kernels.fused_moe.v2 import mixed_combine_kernel

from .test_fused_ep_moe_v2 import (  # noqa: F401
    CAPACITY,
    _make_inputs,
    _make_weights,
    _mesh,
    _require_eight_devices,
)

pytestmark = pytest.mark.multichip

# hidden 1024: 8 lane blocks, so the default strided relayout (LB % 8 == 0) is what runs.
SHAPE = dict(hidden=1024, inter=256, e_total=384, topk=6)


def _run(mesh, tokens, seed, monkeypatch, plan_kernel, combine_kernel, calls):
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh, seed, hidden=SHAPE["hidden"], inter=SHAPE["inter"],
        e_total=SHAPE["e_total"], weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh, seed + 1, tokens=tokens, hidden=SHAPE["hidden"], e_total=SHAPE["e_total"]
    )
    # The switches are patched, not set through the environment: envs may cache.
    monkeypatch.setattr(layer, "_mixed_plan_kernel_enabled", lambda: plan_kernel)
    monkeypatch.setattr(layer, "_mixed_combine_kernel_enabled", lambda: combine_kernel)
    jax.clear_caches()
    out = fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, topk=SHAPE["topk"], renormalize=True,
        mesh=mesh, capacity=CAPACITY, weight_format=WeightFormat.FP8, sharded_plan=True,
    )
    return np.asarray(out)


@pytest.mark.parametrize("plan_kernel", [True, False], ids=["plan_kernel", "xla_plan"])
@pytest.mark.parametrize("tokens", [256, 1024, 2048], ids=["t32", "t128", "t256"])
def test_mixed_combine_kernel_is_bit_exact_in_the_layer(tokens, plan_kernel, monkeypatch):
    mesh = _mesh()
    calls = []
    real = mixed_combine_kernel.build_mixed_combine

    def spy(*a, **k):
        calls.append(a[0].shape)
        return real(*a, **k)

    monkeypatch.setattr(layer, "build_mixed_combine", spy)
    want = _run(mesh, tokens, 40, monkeypatch, plan_kernel, False, calls)
    assert not calls
    got = _run(mesh, tokens, 40, monkeypatch, plan_kernel, True, calls)
    assert calls, "the mixed combine kernel did not engage"
    assert np.abs(want).max() > 0
    np.testing.assert_array_equal(got, want)


def test_mixed_combine_kernel_does_not_engage_at_decode_buckets(monkeypatch):
    mesh = _mesh()
    calls = []
    monkeypatch.setattr(layer, "build_mixed_combine", lambda *a, **k: calls.append(1))
    _run(mesh, 64, 42, monkeypatch, True, True, calls)
    assert not calls
