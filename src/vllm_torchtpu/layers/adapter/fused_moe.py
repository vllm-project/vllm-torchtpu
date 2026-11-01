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
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.compilation import backends as vllm_compilation_backends
from vllm.distributed.parallel_state import get_dp_group, get_ep_group

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.tpu_communicator import combine_splits
from vllm_torchtpu.kernels.megablox.quantized_rows import (
    ROWS_PER_WORD,
    quantized_row_width,
)
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    dispatches_quantized_rows,
    fused_moe_func,
    quantize_dispatch_rows,
    quantize_to_native_fp4_kmajor,
    quantizes_dispatch_rows,
    requant_unpack_kmajor,
    resolve_onehot_permute_threshold,
    unpack_fp4_to_e2m1,
)

_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str, bool, Any, bool, bool], Callable] = {}
_load_kmajor_fp4_op = None
_requant_kmajor_fp4_ops: dict[int, Callable] = {}
_quantize_native_fp4_kmajor_ops: dict[tuple[int, bool], Callable] = {}

# Registered at import: the EP dispatch quantizes inside the compiled forward,
# where Dynamo cannot trace a registration.
_quantize_dispatch_rows_op = pallas.jax_op(
    "pallas::quantize_dispatch_rows", quantize_dispatch_rows
)
_quantize_dispatch_rows_op.register_fake(
    lambda x: x.new_empty(
        (x.shape[0] // ROWS_PER_WORD, quantized_row_width(x.shape[1])),
        dtype=torch.int32,
    )
)


def _resolve_activation_name(activation: Any) -> str:
    if isinstance(activation, enum.Enum):
        return activation.value
    return activation.value if hasattr(activation, "value") else str(activation)


def get_fused_moe_activation(activation, moe_config) -> str:
    """Encode vLLM's activation config for the Pallas GMM kernel."""
    activation = _resolve_activation_name(activation)
    if activation != "situ":
        return activation

    beta = moe_config.activation_situ_beta
    linear_beta = moe_config.activation_situ_linear_beta
    assert beta is not None
    return f"situ:{beta}:{'none' if linear_beta is None else linear_beta}"


class TpuMoEActivationMixin:
    """Holds the Pallas GMM kernel's activation string for one MoE layer.

    Written by process_weights_after_loading, read by apply_monolithic.
    vLLM builds one quant method per MoE layer, so the method is the right
    owner. Defines no __init__, so it does not disturb the upstream
    __init__s these methods deliberately bypass; mix it in ahead of the
    upstream class.
    """

    _tpu_activation_str: str | None = None

    def _resolve_tpu_activation(self, layer) -> str:
        """Encode this layer's activation for the kernel.

        Overridden where a method needs a different encoding -- the string
        is not uniformly recomputable from layer.activation.
        """
        return get_fused_moe_activation(layer.activation, layer.moe_config)

    def _set_tpu_activation(self, layer) -> str:
        """Resolve and store; returns it, since callers also use it locally."""
        self._tpu_activation_str = self._resolve_tpu_activation(layer)
        return self._tpu_activation_str


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
        _load_kmajor_fp4_op = pallas.jax_op(
            "pallas::nvfp4_load_kmajor", unpack_fp4_to_e2m1
        )
    return _load_kmajor_fp4_op(w_u8)


def requant_load_kmajor_fp4(
    w_u8: torch.Tensor, scale_f: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
            functools.partial(requant_unpack_kmajor, block=block),
        )
        _requant_kmajor_fp4_ops[block] = op
    return op(w_u8, scale_f)


def quantize_native_fp4_kmajor(
    w: torch.Tensor, block: int, pack: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an already-dequantized, K-major float weight to native FP4.

    ``pack=False`` hands the weight back as ``torch.float4_e2m1fn_x2`` rather
    than packed uint8 -- the form the fused EP MoE kernel takes. The two are
    distinct traced programs with distinct op names, so the cache is keyed on
    both: one qualname serving two closures would silently dispatch to
    whichever registered first.
    """
    key = (block, pack)
    op = _quantize_native_fp4_kmajor_ops.get(key)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_quantize_native_kmajor_b{block}"
            f"{'' if pack else '_unpacked'}",
            functools.partial(quantize_to_native_fp4_kmajor, block=block, pack=pack),
        )
        _quantize_native_fp4_kmajor_ops[key] = op
    return op(w)


def quantizes_moe_dispatch_rows(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    *,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
) -> bool:
    """fused_moe_gmm.quantizes_dispatch_rows for these EP experts' weights.

    Call it at weight load: it asks the TPU, which a traced forward cannot.
    """

    def shape(t: torch.Tensor | None):
        if t is None:
            return None
        dtype = jnp.dtype(str(t.dtype).removeprefix("torch."))
        return jax.ShapeDtypeStruct(tuple(t.shape), dtype)

    return quantizes_dispatch_rows(
        *map(shape, (w1, w2, w1_scale, w2_scale, w1_bias, w2_bias)),
        topk=topk,
        activation=activation,
        rhs_quant_dtype=rhs_quant_dtype,
    )


def dispatches_quantized(
    num_tokens: int,
    topk: int,
    is_sequence_parallel: bool,
    onehot_moe_permute_threshold: int,
) -> bool:
    """Whether an EP dispatch of num_tokens bf16 rows per rank sends them as
    quantize_moe_dispatch_rows words, to experts quantizes_moe_dispatch_rows
    admitted; fused_moe_gmm computes the same output from the words.

    The group is the one TpuDeviceCommunicator.dispatch gathers over, and
    the threshold the one the fused MoE op was built with
    (resolve_onehot_permute_threshold). Only the backbone compares its token
    count: the MTP drafter has one graph for every bucket (see
    combine_splits). Integer arithmetic only: the count is dynamic in the
    trace.
    """
    group = get_ep_group() if is_sequence_parallel else get_dp_group()
    return vllm_compilation_backends.model_tag == "backbone" and (
        dispatches_quantized_rows(
            num_tokens, group.world_size, topk, onehot_moe_permute_threshold
        )
    )


def quantize_moe_dispatch_rows(hidden_states: torch.Tensor) -> torch.Tensor:
    """hidden_states' rows quantized as the experts read them, as int32 words
    (``ROWS_PER_WORD`` rows each), in about half the bytes."""
    return _quantize_dispatch_rows_op(hidden_states)


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
    skip_padded_tokens: bool,
    rhs_quant_dtype=None,
    combine_in_pieces: bool = False,
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

    # Pass experts_start as a runtime tensor rather than static closure arg so every EP
    # rank compiles an identical program.
    # See fa8faaf5 for background on rank-uniform sharding in compiled Pallas graphs.
    wrapped_fn = functools.partial(
        fused_moe_func,
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        combine_in_pieces=combine_in_pieces,
    )

    fused_moe_kernel_impl = pallas.jax_op(op_name, wrapped_fn)

    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, w1, w2, *args, **kwargs):
        if hidden_states.dtype == torch.int32:
            # quantize_moe_dispatch_rows words: one bf16 row per token out.
            topk_ids = args[5]
            return hidden_states.new_empty(
                (topk_ids.shape[0], w2.shape[-1]), dtype=torch.bfloat16
            )
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (
        topk,
        activation,
        use_ep,
        rhs_quant_dtype,
        skip_padded_tokens,
        combine_in_pieces,
    )
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
    combine_in_pieces: bool = False,
):
    if skip_padded_tokens is None:
        skip_padded_tokens = envs.TPU_MOE_SKIP_PADDED_TOKENS
    cache_key = (
        topk,
        activation,
        use_ep,
        rhs_quant_dtype,
        skip_padded_tokens,
        combine_in_pieces,
    )
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        combine_in_pieces=combine_in_pieces,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing.

    An EP layer gets both unpermute variants: which one a forward takes is
    decided while it is traced (see ``fused_moe_gmm``).
    """
    for combine_in_pieces in (False, True) if use_ep else (False,):
        _get_fused_moe_custom_op(
            topk=topk,
            activation=activation,
            use_ep=use_ep,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
            combine_in_pieces=combine_in_pieces,
        )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
    combine_in_pieces: bool | None = None,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``hidden_states`` may also be quantize_moe_dispatch_rows words of the
    gathered rows, where dispatches_quantized holds; the output is the bf16
    one the rows give.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP) -- see
    ``moe_routing.get_experts_start_buffer``. It must be a 0-d int32 tensor
    (a persistent buffer registered once per layer), not a Python int: every
    EP rank owns a different value, and passing it as real tensor data (vs.
    binding it as a Python int into the compiled closure) keeps the compiled
    program identical across ranks. The kernel remaps global ids to local ids
    with an elementwise subtract and masks non-local experts. When present,
    the sparse-core parity path dispatches ragged gather/gather-reduce.

    ``combine_in_pieces`` says the output goes, unchanged, to vLLM's EP combine
    (``TpuDeviceCommunicator.combine``), so its rows are laid out piece-major
    whenever that combine splits (``ep_combine_layout``). By default it does
    for an EP layer whenever the combine splits, which relies on the routed
    output reaching that combine (from vLLM's MoERunner ``_maybe_combine`` or
    from a method that dispatches its routes) with nothing token-wise in
    between; a caller that consumes the output itself passes False and gets
    token order.
    """
    use_ep = experts_start is not None
    if combine_in_pieces is None:
        combine_in_pieces = use_ep and combine_splits()
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
        combine_in_pieces=combine_in_pieces,
    )
    if experts_start is None:
        # The compiled program for use_ep=False never reads this operand
        # (see the `if use_ep:` guard in fused_moe_func), but the custom op
        # still needs a concrete tensor to call with a fixed arity.
        experts_start = torch.zeros((), dtype=torch.int32, device=hidden_states.device)
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
