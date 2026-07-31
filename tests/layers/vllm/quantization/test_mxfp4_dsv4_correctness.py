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
"""Numerical-correctness tests for VllmMxfp4MoEMethod against real,
checkpoint-format MXFP4-quantized weights, exercising both DeepSeek-V4's
grouped-topk routing and its hash-based routing.

Unlike tests/layers/vllm/quantization/test_mxfp4_tpu.py (which pre-populates
VLLM_FP4_CACHE_DIR with random uint8 "weights" and only asserts the output has
the right shape/dtype and no NaN/Inf), these tests:
  1. Construct real MXFP4-quantized checkpoint-format weights (block-32,
     e2m1 packed + e8m0 scale) from known float ground truth, matching the
     exact tensor layout `process_weights_after_loading` expects to read from
     `layer.w1_temp` / `w3_temp` / `w2_temp` / `w13_weight_scale` /
     `w2_weight_scale` (i.e. what the real weight loader would have written).
  2. Leave `layer.layer_idx` unset so `process_weights_after_loading` cannot
     hit its on-disk cache and is forced through the real
     dequantize-then-requantize-then-pack code path.
  3. Compare `apply_monolithic`'s output against a plain-torch MoE reference
     computed from the *dequantized* checkpoint weights (not the
     pre-quantization float weights -- see `_load_real_quantized_weights`'s
     docstring for why), isolating for correctness of the actual
     routing/GMM/activation computation rather than conflating it with
     inherent MXFP4 quantization noise.

This closes the gap flagged in docs/dsv4_architecture_and_rationale.md
Sec 3.12: the only other MoE-numerics test (test_dsv4_moe_routing.py) uses
unquantized bf16 weights and uniform-random routing, never exercising the
real checkpoint's MXFP4-quantized experts or DSV4's hash-based routing (used
by 3 of 43 layers).
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig, RoutingMethodType)

from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod


def t2j(t: torch.Tensor):
    return jnp.asarray(t.detach().cpu().numpy())


def j2t(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


MXFP4_CHECKPOINT_BLOCK_SIZE = 32


def quantize_to_mxfp4(
        weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor's last axis to checkpoint-format MXFP4.

    Ported verbatim (same algorithm) from the trusted sibling reference
    (~/workspace/tpu-inference, tests/layers/vllm/test_mxfp4.py's
    `quantize_to_mxfp4`) per this branch's standing methodology of diffing
    against tpu-inference rather than inventing a fresh quantization scheme
    to test against. Produces the same on-disk layout the real weight loader
    writes: e2m1 values packed 2-per-byte, block-32 e8m0 (power-of-2) scales.
    """
    jweight = t2j(weight)
    e2m1_finfo = jnp.finfo(jnp.float4_e2m1fn)
    dtype_min = float(e2m1_finfo.min)
    dtype_max = float(e2m1_finfo.max)

    weight_shape = jweight.shape
    weight_block = jweight.reshape(weight_shape[:-1] +
                                   (-1, MXFP4_CHECKPOINT_BLOCK_SIZE))
    abs_max = jnp.max(jnp.abs(weight_block), axis=-1, keepdims=True)
    scale = abs_max / dtype_max

    weight_q = jnp.clip(weight_block / scale, dtype_min, dtype_max)
    weight_q = weight_q.astype(jnp.float4_e2m1fn).reshape(weight_shape[:-1] +
                                                          (-1, 2))
    weight_packed = jax.lax.bitcast_convert_type(weight_q, jnp.uint8)

    e8m0_finfo = jnp.finfo(jnp.float8_e8m0fnu)
    _, scale_exp = jnp.frexp(scale.squeeze(axis=-1))
    scale_exp -= 1
    scale_exp = (scale_exp - e8m0_finfo.minexp).astype(jnp.uint8)

    return j2t(weight_packed), j2t(scale_exp)


def ref_moe_grouped_topk(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
) -> torch.Tensor:
    """Plain-torch grouped-topk MoE forward, matching moe_routing.tpu_grouped_topk
    (sigmoid scoring, no correction bias) and gmm_v2.silu_and_mul_with_clamp.
    """
    num_tokens, num_experts = router_logits.shape
    scores = router_logits.float().sigmoid()
    experts_per_group = num_experts // num_expert_group
    group_scores = scores.view(num_tokens, num_expert_group,
                               experts_per_group).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(num_tokens, num_expert_group,
                                                  experts_per_group).reshape(
                                                      num_tokens, num_experts))
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return _dense_expert_compute(x, w1, w3, w2, topk_weights, topk_ids)


def ref_moe_hash(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    hash_indices_table: torch.Tensor,
    input_ids: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Plain-torch hash-routed MoE forward, matching
    moe_routing.select_experts's hash-MoE branch (softmax scoring, weights
    gathered from the unbiased scores at the hash-forced expert id, then
    renormalized -- with top_k=1 this collapses every weight to exactly 1.0,
    which is real select_experts behavior, not a quirk of this reference).
    """
    scores = router_logits.float().softmax(dim=-1)
    topk_ids = hash_indices_table[input_ids.long()].long()
    if topk_ids.ndim == 1:
        topk_ids = topk_ids[:, None]
    topk_weights = scores.gather(-1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(
        dim=-1, keepdim=True).clamp(min=1e-20)
    return _dense_expert_compute(x, w1, w3, w2, topk_weights, topk_ids)


def _dense_expert_compute(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    out = torch.zeros_like(x, dtype=torch.float32)
    xf = x.float()
    for t in range(num_tokens):
        acc = torch.zeros(x.shape[1], dtype=torch.float32)
        for slot in range(topk):
            e = int(topk_ids[t, slot])
            gate = xf[t] @ w1[e].float().T
            up = xf[t] @ w3[e].float().T
            gate = gate.clamp(max=10.0)
            up = up.clamp(min=-10.0, max=10.0)
            hidden = F.silu(gate) * up
            down = hidden @ w2[e].float().T
            acc = acc + topk_weights[t, slot].float() * down
        out[t] = acc
    return out.to(x.dtype)


@contextlib.contextmanager
def _mxfp4_context(device):
    """`VllmMxfp4MoEMethod` (a CustomOp) and its `get_tensor_model_parallel_*`
    calls both need a live vLLM config / TP-group stand-in. Construction,
    weight loading, and the forward pass all need to happen inside this same
    context (matching the pattern test_mxfp4_tpu.py uses).
    """
    dummy_vllm_config = SimpleNamespace(
        model_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        lora_config=None,
        device_config=SimpleNamespace(device=device),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            enable_expert_parallel=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
            expert_placement_strategy="linear",
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            static_all_moe_layers=[],
            custom_ops=["all"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
            max_cudagraph_capture_size=0,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, ),
    )
    with set_current_vllm_config(dummy_vllm_config), \
         patch("vllm_torchtpu.layers.vllm.quantization.mxfp4.get_tensor_model_parallel_world_size", return_value=1):
        yield


def _build_moe_layer(device,
                     num_experts,
                     top_k,
                     hidden_size,
                     intermediate_size,
                     use_grouped_topk,
                     num_expert_group,
                     topk_group,
                     hash_indices_table=None):
    moe_parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
        pcp_size=1,
        pcp_rank=0,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=moe_parallel_config,
        in_dtype=torch.bfloat16,
    )

    kwargs = dict(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=torch.bfloat16,
        renormalize=True,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        pcp_size=1,
        activation="silu",
    )
    if use_grouped_topk:
        kwargs.update(use_grouped_topk=True,
                      num_expert_group=num_expert_group,
                      topk_group=topk_group,
                      scoring_func="sigmoid")
    else:
        kwargs.update(scoring_func="softmax",
                      hash_indices_table=hash_indices_table)
    layer = FusedMoE(**kwargs)
    method = VllmMxfp4MoEMethod(moe_config)
    # Normally set by Mxfp4MoEMethod.create_weights, which we bypass (like
    # test_mxfp4_tpu.py) since we populate the packed/scale tensors directly.
    method.intermediate_size = intermediate_size
    method.hidden_size = hidden_size

    layer._tpu_activation_str = "silu"
    layer.renormalize = True
    layer.device = device
    layer.quant_method = method
    return layer, method


def _load_real_quantized_weights(layer, method, num_experts, hidden_size,
                                 intermediate_size, generator, device):
    """Populate layer.w1_temp/w3_temp/w2_temp + scale params with real,
    checkpoint-format MXFP4-quantized weights (as the real weight loader
    would), and return the *dequantized* (post-checkpoint-quantization)
    float ground truth to build a reference against.

    The reference is built from the dequantized weights, not the
    pre-quantization float weights: checkpoint-format MXFP4 (block-32, an
    8-value-per-sign e2m1 grid) is inherently lossy for arbitrary values --
    confirmed separately (scratch/debug_full_reference_check.py) that
    requantizing already-dequantized values for the kernel's own internal
    format (process_weights_after_loading's second stage) is a further
    exact no-op. So the *only* meaningful, non-inherent source of error left
    to check is whether apply_monolithic's routing + GMM matmul correctly
    reproduces what the checkpoint's own dequantized weights represent --
    comparing against the pre-quantization float would conflate that with
    expected, unavoidable first-stage quantization noise instead.
    """
    # process_weights_after_loading reads layer.w13_weight.device as the
    # target device for its output (normally set by create_weights, which we
    # bypass, matching test_mxfp4_tpu.py's technique) -- a bare placeholder
    # is enough, it gets deleted immediately once read.
    layer.w13_weight = torch.nn.Parameter(torch.zeros(1, device=device),
                                          requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.zeros(1, device=device),
                                         requires_grad=False)
    w1_true = torch.randn(
        num_experts, intermediate_size, hidden_size, generator=generator) / 4
    w3_true = torch.randn(
        num_experts, intermediate_size, hidden_size, generator=generator) / 4
    w2_true = torch.randn(
        num_experts, hidden_size, intermediate_size, generator=generator) / 4

    w1_packed, w1_scale = quantize_to_mxfp4(w1_true)
    w3_packed, w3_scale = quantize_to_mxfp4(w3_true)
    w2_packed, w2_scale = quantize_to_mxfp4(w2_true)

    layer.w1_temp = w1_packed
    layer.w3_temp = w3_packed
    layer.w2_temp = w2_packed
    layer.w13_weight_scale = torch.cat([w1_scale, w3_scale], dim=1)
    layer.w2_weight_scale = w2_scale
    # Ensure the on-disk FP4 cache can't short-circuit process_weights: no
    # layer_idx means process_weights_after_loading can never build a cache
    # key, so the real dequantize/requantize path always runs.
    assert getattr(layer, "layer_idx", None) is None

    from vllm_torchtpu.layers.common.quantization import \
        dequantize_mxfp4_packed
    w1_ref = dequantize_mxfp4_packed(w1_packed,
                                     w1_scale,
                                     axis=2,
                                     out_dtype=torch.bfloat16).float()
    w3_ref = dequantize_mxfp4_packed(w3_packed,
                                     w3_scale,
                                     axis=2,
                                     out_dtype=torch.bfloat16).float()
    w2_ref = dequantize_mxfp4_packed(w2_packed,
                                     w2_scale,
                                     axis=2,
                                     out_dtype=torch.bfloat16).float()

    return w1_ref, w3_ref, w2_ref


class TestMxfp4DSV4Correctness:

    def test_grouped_topk_routing_matches_reference(self, device):
        torch.manual_seed(0)
        generator = torch.Generator().manual_seed(0)
        num_experts, top_k = 8, 2
        num_expert_group, topk_group = 4, 2
        hidden_size, intermediate_size = 128, 128
        num_tokens = 8

        with _mxfp4_context(device):
            layer, method = _build_moe_layer(
                device,
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                use_grouped_topk=True,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
            w1_true, w3_true, w2_true = _load_real_quantized_weights(
                layer, method, num_experts, hidden_size, intermediate_size,
                generator, device)

            method.process_weights_after_loading(layer)

            x = torch.randn(num_tokens,
                            hidden_size,
                            generator=generator,
                            dtype=torch.bfloat16) / 4
            router_logits = torch.randn(num_tokens,
                                        num_experts,
                                        generator=generator,
                                        dtype=torch.bfloat16)

            x_dev = x.to(device)
            router_logits_dev = router_logits.to(device)
            actual = method.apply_monolithic(layer, x_dev, router_logits_dev)
            actual = actual.to("cpu")

        expected = ref_moe_grouped_topk(x,
                                        router_logits,
                                        w1_true,
                                        w3_true,
                                        w2_true,
                                        top_k,
                                        num_expert_group,
                                        topk_group,
                                        renormalize=True)

        assert not torch.isnan(actual).any()
        torch.testing.assert_close(actual.float(),
                                   expected.float(),
                                   atol=6e-2,
                                   rtol=6e-2)

    def test_hash_routing_matches_reference(self, device):
        torch.manual_seed(1)
        generator = torch.Generator().manual_seed(1)
        num_experts, top_k = 8, 1
        hidden_size, intermediate_size = 128, 128
        num_tokens = 8
        vocab_size = 32

        # [vocab_size, top_k]: select_experts infers per-token k from this
        # trailing dim (topk_ids.shape[-1]) after indexing by input_ids, so a
        # 1-D table would be silently misread as [num_tokens] instead.
        hash_indices_table = torch.randint(0,
                                           num_experts, (vocab_size, top_k),
                                           generator=generator,
                                           dtype=torch.int32)

        with _mxfp4_context(device):
            layer, method = _build_moe_layer(
                device,
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                use_grouped_topk=False,
                num_expert_group=0,
                topk_group=0,
                # Real deployments load this onto the accelerator alongside
                # the rest of the model's weights/buffers.
                hash_indices_table=hash_indices_table.to(device),
            )
            w1_true, w3_true, w2_true = _load_real_quantized_weights(
                layer, method, num_experts, hidden_size, intermediate_size,
                generator, device)

            method.process_weights_after_loading(layer)

            x = torch.randn(num_tokens,
                            hidden_size,
                            generator=generator,
                            dtype=torch.bfloat16) / 4
            router_logits = torch.randn(num_tokens,
                                        num_experts,
                                        generator=generator,
                                        dtype=torch.bfloat16)
            input_ids = torch.randint(0,
                                      vocab_size, (num_tokens, ),
                                      generator=generator)

            x_dev = x.to(device)
            router_logits_dev = router_logits.to(device)
            input_ids_dev = input_ids.to(device)
            actual = method.apply_monolithic(layer,
                                             x_dev,
                                             router_logits_dev,
                                             input_ids=input_ids_dev)
            actual = actual.to("cpu")

        expected = ref_moe_hash(x, router_logits, hash_indices_table,
                                input_ids, w1_true, w3_true, w2_true)

        assert not torch.isnan(actual).any()
        torch.testing.assert_close(actual.float(),
                                   expected.float(),
                                   atol=6e-2,
                                   rtol=6e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
