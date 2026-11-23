# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused router kernel (``TPU_MOE_ROUTER_FUSED_EPILOGUE``) vs the unfused path.

Two references are used:

- ``_unfused_xla``: the unfused TPU path's operations written in jax and run
  by XLA:CPU. The bias add, the existing top-k kernel (interpreted), the
  gather, the row sum, the clamp, the divide and the scale. The fused kernel
  must equal it bit for bit.
- ``moe_routing.select_experts`` with the flag off, which off TPU runs torch
  ops. torch:CPU and XLA:CPU can add the 8 selected weights in different
  orders, so the renormalized weights may differ from it by one float32
  rounding of the sum. Indices must still be equal.

The kernel runs with ``interpret=True``, where its row sum is also an XLA:CPU
reduction. These tests do not check Mosaic lowering, and they do not check
that Mosaic and XLA:TPU add the row sum in the same order.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.router_topk import select, select_fused
from vllm_torchtpu.layers.adapter import moe_routing

# GLM-5.2 router: 256 routed experts, 8 per token, sigmoid scores with a
# correction bias, renormalized, routed scale 2.5.
EXPERTS = 256
TOPK = 8
SCALE = 2.5


def _layer(bias, renormalize, scale):
    return SimpleNamespace(
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        renormalize=renormalize,
        scoring_func="sigmoid",
        custom_routing_function=None,
        e_score_correction_bias=bias,
        routed_scaling_factor=scale,
        moe_config=SimpleNamespace(experts_per_token=TOPK),
    )


def _unfused(monkeypatch, logits, bias, renormalize, scale):
    monkeypatch.setattr(moe_routing, "_ROUTER_TOPK", "rowmax")
    monkeypatch.setattr(moe_routing, "_ROUTER_FUSED_EPILOGUE", False)
    monkeypatch.setattr(moe_routing.envs, "TPU_MOE_TOPK_WEIGHTS_FLOAT32", True)
    hidden = torch.zeros(logits.shape[0], 8, dtype=torch.bfloat16)
    return moe_routing.select_experts(
        hidden_states=hidden,
        router_logits=logits,
        topk=TOPK,
        renormalize=renormalize,
        scoring_fn="sigmoid",
        layer=_layer(bias, renormalize, scale),
    )


def _unfused_xla(scores, bias, renormalize, scale):
    """The unfused TPU router path after the scoring function, in jax ops."""
    _, ids = select(scores + bias[None, :], TOPK, interpret=True)
    w = jnp.take_along_axis(scores, ids, axis=-1)
    if renormalize:
        w = w / jnp.maximum(jnp.sum(w, axis=-1, keepdims=True), 1e-20)
    if scale != 1.0:
        w = w * scale
    return np.asarray(w), np.asarray(ids)


def _inputs(rows, seed):
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(rows, EXPERTS, generator=gen) * 2.0).to(torch.bfloat16)
    # Bias of the same order as the scores, so it changes the selection.
    bias = torch.rand(EXPERTS, generator=gen) * 0.5
    return logits, bias


@pytest.mark.parametrize("rows", [16, 32, 100, 1000])
@pytest.mark.parametrize("renormalize,scale", [(True, SCALE), (False, 1.0)])
def test_fused_kernel_equals_unfused_path(monkeypatch, rows, renormalize, scale):
    logits, bias = _inputs(rows, rows)
    ref_w, ref_i = _unfused(monkeypatch, logits, bias, renormalize, scale)

    scores = logits.float().sigmoid()
    got_w, got_i = select_fused(
        jnp.asarray(scores.numpy()),
        jnp.asarray(bias.numpy()),
        TOPK,
        renormalize,
        scale,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(got_i), ref_i.numpy())
    # At most one float32 rounding of the row sum (different add order).
    np.testing.assert_allclose(np.asarray(got_w), ref_w.numpy(), rtol=3e-7, atol=0)

    xla_w, xla_i = _unfused_xla(
        jnp.asarray(scores.numpy()), jnp.asarray(bias.numpy()), renormalize, scale
    )
    np.testing.assert_array_equal(np.asarray(got_i), xla_i)
    np.testing.assert_array_equal(np.asarray(got_w), xla_w)


def test_bias_changes_the_selection_but_not_the_weight_source(monkeypatch):
    """Positive control: the bias must move experts, and the returned weight
    must be the unbiased score of each selected expert."""
    logits, bias = _inputs(16, 7)
    scores = logits.float().sigmoid()
    _, plain_i = torch.topk(scores, TOPK, dim=-1)
    got_w, got_i = select_fused(
        jnp.asarray(scores.numpy()),
        jnp.asarray(bias.numpy()),
        TOPK,
        False,
        1.0,
        interpret=True,
    )
    got_i = torch.from_numpy(np.asarray(got_i)).long()
    assert not torch.equal(got_i, plain_i)
    np.testing.assert_array_equal(np.asarray(got_w), scores.gather(-1, got_i).numpy())


def test_select_experts_flag_on_equals_flag_off_off_tpu(monkeypatch):
    """The gate reaches the fused branch and keeps dtypes and values (torch
    fallback off TPU)."""
    logits, bias = _inputs(16, 3)
    ref_w, ref_i = _unfused(monkeypatch, logits, bias, True, SCALE)
    monkeypatch.setattr(moe_routing, "_ROUTER_FUSED_EPILOGUE", True)
    called = []
    real = moe_routing.rowmax_select_fused
    monkeypatch.setattr(
        moe_routing,
        "rowmax_select_fused",
        lambda *a, **k: called.append(1) or real(*a, **k),
    )
    got_w, got_i = moe_routing.select_experts(
        hidden_states=torch.zeros(16, 8, dtype=torch.bfloat16),
        router_logits=logits,
        topk=TOPK,
        renormalize=True,
        scoring_fn="sigmoid",
        layer=_layer(bias, True, SCALE),
    )
    assert called
    assert got_w.dtype == ref_w.dtype == torch.float32
    assert got_i.dtype == torch.int32
    assert torch.equal(got_i, ref_i)
    assert torch.equal(got_w, ref_w)


def test_fused_epilogue_env_defaults_to_false(monkeypatch):
    import vllm_torchtpu.envs as envs

    monkeypatch.delenv("TPU_MOE_ROUTER_FUSED_EPILOGUE", raising=False)
    assert envs.environment_variables["TPU_MOE_ROUTER_FUSED_EPILOGUE"]() is False
