# SPDX-License-Identifier: Apache-2.0
"""TPU_DSV4_EP_MIXED_PLAN_KERNEL: the mixed-bucket routing plan as one Pallas
call (mixed_plan_kernel.build_mixed_plan) must leave the layer's output
bit-exact against the XLA sharded plan it replaces.

The kernel's own oracle test (tests/kernels/fused_moe/mixed_plan/) checks
every operand against a first-principles numpy re-derivation with a fault
control; this one checks the layer wiring: the operands reach the fused EP
kernel and the combine in the right order, and the switch engages only at
mixed buckets. The switch is patched rather than set through the
environment: vllm_torchtpu.envs may cache a value once read in-process.
"""

import jax
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2, layer
from vllm_torchtpu.kernels.fused_moe.v2 import mixed_plan_kernel

from .test_fused_ep_moe_v2 import (  # noqa: F401
    CAPACITY,
    _make_inputs,
    _make_weights,
    _mesh,
    _require_eight_devices,
)

pytestmark = pytest.mark.multichip

# DeepSeek-V4-Pro's routing geometry (384 experts, top-6) on a small FFN.
SHAPE = dict(hidden=512, inter=256, e_total=384, topk=6)


@pytest.mark.parametrize("oneshot", [False, True], ids=["xla_exchange", "oneshot_exchange"])
@pytest.mark.parametrize("tokens", [256, 1024, 2048], ids=["t32", "t128", "t256"])
def test_mixed_plan_kernel_is_bit_exact_in_the_layer(tokens, oneshot, monkeypatch):
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh, 30, hidden=SHAPE["hidden"], inter=SHAPE["inter"],
        e_total=SHAPE["e_total"], weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh, 31, tokens=tokens, hidden=SHAPE["hidden"], e_total=SHAPE["e_total"]
    )
    common = dict(
        topk=SHAPE["topk"], renormalize=True, mesh=mesh, capacity=CAPACITY,
        weight_format=WeightFormat.FP8, sharded_plan=True,
    )
    calls = []
    real = mixed_plan_kernel.build_mixed_plan

    def spy(*a, **k):
        calls.append(k["t_local"])
        return real(*a, **k)

    monkeypatch.setattr(layer, "build_mixed_plan", spy)
    monkeypatch.setattr(layer, "_mixed_plan_kernel_enabled", lambda: False)
    jax.clear_caches()
    want = np.asarray(fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common))
    assert not calls
    monkeypatch.setattr(layer, "_mixed_plan_kernel_enabled", lambda: True)
    monkeypatch.setattr(layer, "_mixed_exchange_oneshot_enabled", lambda: oneshot)
    jax.clear_caches()
    got = np.asarray(fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common))
    assert calls, "the mixed plan kernel did not engage"
    assert np.abs(want).max() > 0
    np.testing.assert_array_equal(got, want)


def test_mixed_plan_kernel_does_not_engage_at_decode_buckets(monkeypatch):
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh, 32, hidden=SHAPE["hidden"], inter=SHAPE["inter"],
        e_total=SHAPE["e_total"], weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(mesh, 33, tokens=64, hidden=SHAPE["hidden"], e_total=SHAPE["e_total"])
    calls = []
    monkeypatch.setattr(layer, "build_mixed_plan", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(layer, "_mixed_plan_kernel_enabled", lambda: True)
    jax.clear_caches()
    fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, topk=SHAPE["topk"], renormalize=True,
        mesh=mesh, capacity=CAPACITY, weight_format=WeightFormat.FP8, sharded_plan=True,
    )
    assert not calls
