# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE prefill kernel (routed + shared expert,
combine and all-reduce) against the decode kernel tests' decomposed layer, with
the experts TP-sliced or EP-sharded, whole or streamed in intermediate blocks,
and the shared expert gated (Qwen3.5) or ungated with a routed scale
(GLM-5.2)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.experimental.test_fused_moe_decode_kernel import (
    GLM_SCALE,
    E,
    _inputs,
    _max_rel,
    _reference,
    setup,  # noqa: F401  (the module fixture)
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill,
    plan_blocks,
)

pytestmark = pytest.mark.multichip

F8 = jnp.float8_e4m3fn


def _prefill(rank, x, *rest, ep=False, glm=False, i_block=None):
    *weights, shared_gate_weight, tw, ti = rest
    return fused_moe_prefill(
        x,
        *weights,
        None if glm else shared_gate_weight,
        tw,
        ti,
        axis_name="d",
        expert_start=rank.reshape(()) * E if ep else None,
        routed_scale=GLM_SCALE if glm else 1.0,
        i_block=i_block,
    )


# "blocked" forces blocked mode (the one GLM-5.2's EP experts take) with ib
# 128: two blocks per expert, and the sparse combine.
_PATHS = {"kernel": _prefill, "blocked": functools.partial(_prefill, i_block=128)}


# 256: one combine chunk on the Pallas combine. 1000: a chunk the Pallas
# combines do not tile (their XLA form). 4096: two chunks, two all-reduces.
@pytest.mark.parametrize("m", [256, 1000, 4096])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("path", ["kernel", "blocked"])
def test_matches_decomposed_layer(setup, m, ep, path):  # noqa: F811
    if path == "blocked" and m == 4096:
        pytest.skip("two-chunk case on the whole-expert path only")
    run, ids = _inputs(setup, m, ep, padded=True, seed=m)
    got = run(functools.partial(_PATHS[path], ep=ep))
    ref = run(functools.partial(_reference, ep=ep))
    assert all(np.array_equal(got[0], got[r]) for r in range(len(got))), (
        "ranks disagree"
    )
    assert np.isfinite(got).all()
    # FP8 activations, BF16 slab rows and a BF16 all-reduce against f32 math:
    # a few percent per row.
    assert _max_rel(got[0], ref[0]) < 0.08
    # Positive control: a reference missing one routed expert must fail the bound.
    # In EP mode the local id of expert ids[0, 0] is only that on its owner rank.
    broken = run(functools.partial(_reference, skip_expert=int(ids[0, 0]) % E, ep=ep))
    assert _max_rel(got[0], broken[0]) > 0.08


@pytest.mark.parametrize("path", ["kernel", "blocked"])
def test_glm_ungated_shared_expert_routed_scale(setup, path):  # noqa: F811
    """GLM-5.2 under EP: no shared-expert gate and routed_scaling_factor 2.5 on
    the routed experts only, against the kernel-off composition. Must-fail
    control: the scale applied to the shared expert too."""
    run, _ = _inputs(setup, 256, ep=True, padded=True, seed=200)
    got = run(functools.partial(_PATHS[path], ep=True, glm=True))
    ref = run(functools.partial(_reference, ep=True, glm=True))
    assert np.isfinite(got).all()
    assert _max_rel(got[0], ref[0]) < 0.08
    wrong = run(functools.partial(_reference, ep=True, glm=True, scale_shared=True))
    assert _max_rel(got[0], wrong[0]) > 0.08


def test_plan_blocks_only_oversized_experts():
    """Whole experts for Qwen3.5-397B (TP8 i 128, EP8 i 1024) and GLM-5.2 at
    TP16 (i 128) at any token count; GLM-5.2's EP16 experts (i 2048) stream
    512-wide blocks through 3 slots below 8192 tokens, 1024-wide through 2
    from 8192 (taller tiles)."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    for tokens in (512, 16384):
        assert plan_blocks(tokens, 4096, 128, 128, F8) is None
        assert plan_blocks(tokens, 4096, 1024, 128, F8) is None
        assert plan_blocks(tokens, 6144, 128, 128, F8) is None
    assert plan_blocks(1024, 6144, 2048, 128, F8) == (512, 3, 512)
    assert plan_blocks(8192, 6144, 2048, 128, F8) == (1024, 2, 256)
