# Copyright 2025 Google LLC
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
"""Torch bridge for fused MoE based on fused_moe_func in fused_moe_gmm."""

import enum
import functools
from typing import Any, Callable, Optional

import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.common.fused_moe_gmm import (
    fused_moe_func, quantize_to_native_fp4_kmajor, requant_unpack_kmajor,
    unpack_fp4_to_e2m1)

_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str, Optional[int], Any],
                              Callable] = {}
_load_kmajor_fp4_op = None
_requant_kmajor_fp4_ops: dict[int, Callable] = {}
_quantize_native_fp4_kmajor_ops: dict[int, Callable] = {}


def _resolve_activation_name(activation: Any) -> str:
    if isinstance(activation, enum.Enum):
        return activation.value
    return activation.value if hasattr(activation,
                                       "value") else str(activation)


def get_fused_moe_activation(activation, moe_config) -> str:
    """Encode vLLM's activation config for the Pallas GMM kernel."""
    activation = _resolve_activation_name(activation)
    if activation != "situ":
        return activation

    beta = moe_config.activation_situ_beta
    linear_beta = moe_config.activation_situ_linear_beta
    assert beta is not None
    return f"situ:{beta}:{'none' if linear_beta is None else linear_beta}"


def load_kmajor_fp4(w_u8: torch.Tensor) -> torch.Tensor:
    """One-time load transform: packed uint8 ``[..., N, K//2]`` -> native
    fp4 ``torch.float4_e2m1fn_x2`` ``[..., K, N]`` (K-major).

    Unpacks (bitcast uint8 -> float4_e2m1fn) and transposes the contracting axis
    to gmm_v2's K-major layout ONCE at load. The result is stored as a native-fp4
    Parameter, so the forward hands it straight to gmm_v2 with no per-forward
    unpack or transpose (which keeps the 480B HBM footprint bounded). Requires
    torch_tpu's native ``torch.float4_e2m1fn_x2`` dtype
    (google-pytorch/torch_tpu#1560).
    """
    global _load_kmajor_fp4_op
    if _load_kmajor_fp4_op is None:
        _load_kmajor_fp4_op = pallas.jax_op("pallas::nvfp4_load_kmajor",
                                            unpack_fp4_to_e2m1)
    return _load_kmajor_fp4_op(w_u8)


def requant_load_kmajor_fp4(w_u8: torch.Tensor, scale_f: torch.Tensor,
                            block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """W4A8 requant + K-major load in one JAX op (matches tpu-inference). Takes
    the packed uint8 weight ``[..., N, K//2]`` + its fused fp32 block-16 scale
    ``[..., N, K//16]``; returns native-fp4 ``torch.float4_e2m1fn_x2`` ``[...,
    K, N]`` requantized to ``block`` plus the fp32 kernel scale ``[...,
    K//block, 1, N]``. Doing the dequant+requant in JAX (vs torch) avoids
    materializing the dequantized weight on the host and rounds to fp4 with the
    same cast the kernel reads."""
    op = _requant_kmajor_fp4_ops.get(block)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_requant_kmajor_b{block}",
            functools.partial(requant_unpack_kmajor, block=block))
        _requant_kmajor_fp4_ops[block] = op
    return op(w_u8, scale_f)


def quantize_native_fp4_kmajor(
        w: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an already-dequantized, K-major float weight to native FP4."""
    op = _quantize_native_fp4_kmajor_ops.get(block)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_quantize_native_kmajor_b{block}",
            functools.partial(quantize_to_native_fp4_kmajor, block=block))
        _quantize_native_fp4_kmajor_ops[block] = op
    return op(w)


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
):
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    # TODO: once the pod's vLLM includes --moe-backend (added upstream in
    # vLLM PR #33807, 2026-03-01; absent in the 0.19.0 we run), register the
    # SparseCore vs plain variants through that selector + the MoE oracle
    # instead of this env flag -- the MoE analog of how #191 moved batched
    # RPA onto --attention-backend CUSTOM. Note MoEBackend has no CUSTOM slot
    # yet, so that also needs an upstream OOT-backend hook.
    use_sparse_core = envs.USE_MOE_SPARSE_CORE

    # Pass experts_start as a runtime tensor rather than static closure arg so every EP rank compiles an identical program.
    # See fa8faaf5 for background on rank-uniform sharding in compiled Pallas graphs.
    wrapped_fn = functools.partial(
        fused_moe_func,
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=envs.ONEHOT_MOE_PERMUTE_THRESHOLD,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS)

    fused_moe_kernel_impl = pallas.jax_op(op_name, wrapped_fn)

    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (topk, activation, use_ep, rhs_quant_dtype)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
):
    cache_key = (topk, activation, use_ep, rhs_quant_dtype)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
    )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: Optional[torch.Tensor],
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP) -- see
    ``moe_routing.get_experts_start_buffer``. It must be a 0-d int32 tensor
    (a persistent buffer registered once per layer), not a Python int: every
    EP rank owns a different value, and passing it as real tensor data (vs.
    binding it as a Python int into the compiled closure) keeps the compiled
    program identical across ranks. The kernel remaps global ids to local ids
    with an elementwise subtract and masks non-local experts. When present,
    the sparse-core parity path dispatches ragged gather/gather-reduce.
    """
    use_ep = experts_start is not None
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    if experts_start is None:
        # The compiled program for use_ep=False never reads this operand
        # (see the `if use_ep:` guard in fused_moe_func), but the custom op
        # still needs a concrete tensor to call with a fixed arity.
        experts_start = torch.zeros((),
                                    dtype=torch.int32,
                                    device=hidden_states.device)
    return fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        experts_start,
    )
