# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu.gdn_pool_layout import (
    QWEN_GDN_ARCHITECTURES, PooledGDNStateLayout,
    derive_pooled_gdn_state_layout, pooled_gdn_state_dtypes,
    pooled_gdn_state_itemsize, unified_kv_layout_enabled_for_architecture)
from vllm_torchtpu.kernels.gdn.head_geometry import derive_gdn_head_geometry
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def unified_kv_layout_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether the deployment uses the attention-shaped unified KV pool.

    An explicit TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL wins in either direction.
    Unset selects the pool for the GDN architectures that read their recurrent
    state out of the attention pages; everything else keeps the per-layer KV
    caches, which is the only layout an attention-only model has and the only
    one the other hybrid families implement.
    """
    return unified_kv_layout_enabled_for_architecture(
        vllm_config.model_config.architecture)


_TPU_CACHE_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_ds_mla": torch.uint8,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

_SLOT_ALIGNMENT_BYTES = 16


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _align_block_to_backend(block_size: int, supported) -> int:
    """Smallest manager block >= block_size the backend can serve.

    A fixed int entry means the kernel runs that exact block size, so the
    manager block must be a multiple of it (vLLM's kernel-block machinery
    splits manager blocks into kernel blocks); a MultipleOf(b) entry accepts
    any multiple of b directly."""
    from vllm.v1.attention.backend import MultipleOf
    candidates = []
    for entry in supported:
        base = entry.base if isinstance(entry, MultipleOf) else int(entry)
        candidates.append(_round_up_to_multiple(block_size, base))
    return min(candidates) if candidates else block_size


def _resolve_tpu_cache_dtype(vllm_config: VllmConfig) -> torch.dtype:
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    cache_dtype = cache_config.cache_dtype
    if cache_dtype == "auto":
        cache_dtype = model_config.dtype
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype

    dtype = _TPU_CACHE_DTYPE_TO_TORCH_DTYPE.get(cache_dtype.lower().strip())
    if dtype is None:
        raise ValueError(f"Unsupported TPU KV cache dtype: {cache_dtype}")
    return dtype


def _tpu_attention_slot_size_bytes(vllm_config: VllmConfig, backend_cls,
                                   block_size: int) -> int:
    model_config = vllm_config.model_config
    cache_dtype = _resolve_tpu_cache_dtype(vllm_config)
    page_size = backend_cls.get_kv_cache_page_size_bytes(
        block_size,
        model_config.get_num_kv_heads(vllm_config.parallel_config),
        model_config.get_head_size(),
        cache_dtype,
    )
    if isinstance(page_size, torch.Tensor):
        page_size = int(page_size.item())
    return page_size


def _tpu_attention_page_size_bytes(vllm_config: VllmConfig,
                                   backend_cls) -> int:
    return _tpu_attention_slot_size_bytes(vllm_config, backend_cls, 1)


def _pool_row_tokens(vllm_config: VllmConfig, backend_cls) -> int:
    """Tokens per physical pool row (the pool's `shape[1]` unit).

    GQA/RPA pools keep tokens on their own axis, so a row is one token. The
    MLA v2 cache packs the token axis by the KV dtype packing (bf16: 2,
    fp8: 4), so a pool row holds that many tokens and a 1-token page already
    occupies a whole row. Detected by probing page linearity: with packing p
    the page stays flat until p tokens and doubles by 2p.
    """
    p = 1
    while (_tpu_attention_slot_size_bytes(vllm_config, backend_cls, 2 * p)
           != 2 * _tpu_attention_slot_size_bytes(vllm_config, backend_cls, p)):
        p *= 2
        if p > 128:
            raise ValueError(
                "TPU FA page is not token-linear: page size does not settle "
                "into a linear regime within 256 tokens")
    return p


def _localize_gdn_state_shapes_for_pcp(
    vllm_config: VllmConfig,
    tp_local_shapes: tuple[tuple[int, ...], ...],
    pcp_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Apply GDN's Q/K replication and V sharding to TP-local shapes."""
    if len(tp_local_shapes) != 2:
        raise ValueError(
            "TPU unified hybrid KV pool requires two GDN state regions "
            f"(conv, SSM), got shapes={tp_local_shapes}")

    model_config = vllm_config.model_config
    hf_config = model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    tp_size = int(parallel_config.tensor_parallel_size)
    num_kq_heads = int(hf_config.linear_num_key_heads)
    num_v_heads = int(hf_config.linear_num_value_heads)
    tp_geometry = derive_gdn_head_geometry(num_kq_heads, num_v_heads, tp_size)
    tp_local_kq_heads = tp_geometry.local_num_kq_heads
    tp_local_v_heads = tp_geometry.local_num_v_heads
    d_k = int(hf_config.linear_key_head_dim)
    d_v = int(hf_config.linear_value_head_dim)
    geometry = derive_gdn_head_geometry(tp_local_kq_heads, tp_local_v_heads,
                                        pcp_size)

    conv_shape, recurrent_shape = tp_local_shapes
    expected_conv_dim = (2 * tp_local_kq_heads * d_k + tp_local_v_heads * d_v)
    if not conv_shape or conv_shape[-1] != expected_conv_dim:
        raise ValueError(
            "GDN state conv width does not match its TP-local heads: "
            f"shape={conv_shape}, expected_width={expected_conv_dim}.")
    if not recurrent_shape or recurrent_shape[0] != tp_local_v_heads:
        raise ValueError("GDN recurrent state does not match its TP-local V "
                         f"heads: shape={recurrent_shape}, "
                         f"expected_heads={tp_local_v_heads}.")

    return geometry.local_state_shapes(tp_local_shapes, d_k, d_v)


def _hybrid_mamba_state_layout(
    vllm_config: VllmConfig,
    fa_physical_bytes_per_token: int,
) -> PooledGDNStateLayout | None:
    model_config = vllm_config.model_config
    if not model_config.is_hybrid:
        return None

    from vllm.model_executor.models import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )
    tp_local_shapes = tuple(
        model_cls.get_mamba_state_shape_from_config(vllm_config))
    parallel_config = vllm_config.parallel_config
    if model_config.architecture in QWEN_GDN_ARCHITECTURES:
        # mamba_utils imports current_platform, which loads this module while
        # selecting TpuPlatform. Defer until platform registration is complete.
        from vllm.model_executor.layers.mamba.mamba_utils import \
            is_conv_state_dim_first

        # Model-level sizing precedes layer construction. Use the same
        # whole-head TP geometry as the TPU GDN layer, including replicas.
        hf_config = model_config.hf_text_config
        geometry = derive_gdn_head_geometry(
            hf_config.linear_num_key_heads, hf_config.linear_num_value_heads,
            parallel_config.tensor_parallel_size)
        tp_local_shapes = geometry.local_state_shapes(
            tp_local_shapes,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            conv_dim_axis=0 if is_conv_state_dim_first() else -1)
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size > 1:
        if model_config.architecture in QWEN_GDN_ARCHITECTURES:
            shapes = _localize_gdn_state_shapes_for_pcp(
                vllm_config, tp_local_shapes, pcp_size)
        else:
            # Other hybrid families retain their existing effective-TP state
            # sharding until they define a distinct PCP group layout.
            orig_tp = parallel_config.tensor_parallel_size
            try:
                parallel_config.tensor_parallel_size = orig_tp * pcp_size
                shapes = tuple(
                    model_cls.get_mamba_state_shape_from_config(vllm_config))
            finally:
                parallel_config.tensor_parallel_size = orig_tp
    else:
        shapes = tp_local_shapes

    if len(shapes) != 2:
        raise ValueError(
            "TPU unified hybrid KV pool requires two GDN state regions "
            f"(conv, SSM), got shapes={shapes}")

    # Byte widths come from the pool's own layout helper. Conv remains fixed
    # BF16, while SSM follows the model class's configured dtype. Every other
    # reader of pool bytes (the kernel state plan and the Raiden manifest) is
    # sourced from the same helper, so slot sizing cannot use a different
    # dtype source without contradicting the physical layout.
    #
    # Shapes still come from the model class, which is a different kind of
    # fact: kernel taps and head counts are architecture, not layout. They do
    # carry the spec-decode conv widening, so under spec decode this is wider
    # than `pooled_gdn_conv_state_bytes()`, which drops that widening on
    # purpose (the pool rolls back via per-slot checkpoints instead). Keep it:
    # `VllmGatedDeltaNetAttention.get_state_shape()` still declares the widened
    # conv to vLLM, and the padded page has to cover what vLLM accounts for.
    # Narrowing here means narrowing there in the same change.
    dtypes = tuple(model_cls.get_mamba_state_dtype_from_config(vllm_config))
    if len(dtypes) != 2:
        raise ValueError(
            "TPU unified hybrid KV pool requires two GDN state dtypes "
            f"(conv, SSM), got dtypes={dtypes}")
    conv_dtype, ssm_dtype = pooled_gdn_state_dtypes(dtypes)
    conv_itemsize = pooled_gdn_state_itemsize(conv_dtype)
    ssm_itemsize = pooled_gdn_state_itemsize(ssm_dtype)
    conv_bytes = math.prod(shapes[0]) * conv_itemsize
    ssm_bytes = math.prod(shapes[1]) * ssm_itemsize

    if (pcp_size > 1
            and model_config.architecture not in QWEN_GDN_ARCHITECTURES):
        # Same item sizes on both sides of the ratio, so the check compares
        # sharding rather than dtype bookkeeping.
        full_conv_bytes = math.prod(tp_local_shapes[0]) * conv_itemsize
        full_ssm_bytes = math.prod(tp_local_shapes[1]) * ssm_itemsize
        if (conv_bytes + ssm_bytes) * pcp_size != (full_conv_bytes +
                                                   full_ssm_bytes):
            raise ValueError(
                "PCP-local Mamba state size must be exactly 1/pcp_size of "
                "the TP-local state: "
                f"architecture={model_config.architecture!r}, "
                f"pcp_size={pcp_size}, full_page_size_bytes="
                f"{full_conv_bytes + full_ssm_bytes}, local_page_size_bytes="
                f"{conv_bytes + ssm_bytes}")

    return derive_pooled_gdn_state_layout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=fa_physical_bytes_per_token,
    )


def _tpu_attention_raw_payload_bytes_per_token(vllm_config: VllmConfig) -> int:
    model_config = vllm_config.model_config
    dtype_size = torch.empty(
        (), dtype=_resolve_tpu_cache_dtype(vllm_config)).element_size()
    return (model_config.get_num_kv_heads(vllm_config.parallel_config) * 2 *
            model_config.get_head_size() * dtype_size)


def _derive_tpu_block_slot_config(
    vllm_config: VllmConfig,
    backend_cls,
    *,
    input_block_size: int,
) -> dict[str, int | str | Sequence[object] | None]:
    supported = backend_cls.get_supported_kernel_block_sizes()
    # The pool row (the pool tensor's shape[1] unit) is the quantum the GDN
    # state regions are measured in. GQA/RPA pools hold one token per row;
    # the MLA pool packs the token axis, holding `fa_pool_row_tokens` tokens
    # per row, so the manager block must cover the state regions times that
    # packing — and stay a multiple of it.
    fa_pool_row_tokens = _pool_row_tokens(vllm_config, backend_cls)
    fa_physical_bytes_per_pool_row = _tpu_attention_page_size_bytes(
        vllm_config, backend_cls)
    fa_raw_payload_bytes_per_token = (
        _tpu_attention_raw_payload_bytes_per_token(vllm_config))
    fa_layout_padding_bytes_per_pool_row = (
        fa_physical_bytes_per_pool_row -
        fa_pool_row_tokens * fa_raw_payload_bytes_per_token)

    mamba_layout = _hybrid_mamba_state_layout(vllm_config,
                                              fa_physical_bytes_per_pool_row)
    mamba_raw_state_bytes = (None if mamba_layout is None else
                             mamba_layout.conv_bytes + mamba_layout.ssm_bytes)
    mamba_required_state_bytes = (None if mamba_layout is None else
                                  mamba_layout.required_bytes)
    mamba_fit_block_size: int | None = None
    user_block_size_floor: int | None = None
    backend_min_page_size = backend_cls.get_min_page_size(vllm_config)
    final_block_size = input_block_size
    block_size_source = "input_block_size"
    if mamba_layout is not None:
        mamba_fit_block_size = (mamba_layout.required_tokens *
                                fa_pool_row_tokens)
        if vllm_config.cache_config.user_specified_block_size:
            user_block_size_floor = input_block_size
        block_size_floor = max(
            mamba_fit_block_size,
            user_block_size_floor or 0,
            backend_min_page_size,
        )
        final_block_size = _align_block_to_backend(block_size_floor, supported)
        final_block_size = _round_up_to_multiple(final_block_size,
                                                 fa_pool_row_tokens)
        if (user_block_size_floor is not None
                and block_size_floor == user_block_size_floor):
            block_size_source = "user_block_size_floor"
        elif block_size_floor == backend_min_page_size:
            block_size_source = "backend_min_page_size"
        else:
            block_size_source = "mamba_state_fit"

    fa_physical_slot_bytes = _tpu_attention_slot_size_bytes(
        vllm_config, backend_cls, final_block_size)
    final_block_slot_bytes = fa_physical_slot_bytes
    assert final_block_slot_bytes % _SLOT_ALIGNMENT_BYTES == 0, (
        "TPU cache slot must preserve the existing byte-alignment contract",
        final_block_slot_bytes,
        _SLOT_ALIGNMENT_BYTES,
    )
    if mamba_layout is not None:
        expected_fa_slot_bytes = ((final_block_size // fa_pool_row_tokens) *
                                  fa_physical_bytes_per_pool_row)
        if fa_physical_slot_bytes != expected_fa_slot_bytes:
            raise ValueError("TPU unified hybrid FA page is not token-linear: "
                             f"block_size={final_block_size}, "
                             f"pool_row_tokens={fa_pool_row_tokens}, "
                             f"fa_physical_bytes_per_pool_row="
                             f"{fa_physical_bytes_per_pool_row}, "
                             f"expected_page_bytes={expected_fa_slot_bytes}, "
                             f"actual_page_bytes={fa_physical_slot_bytes}")
        if fa_physical_slot_bytes < mamba_layout.required_bytes:
            raise ValueError(
                "TPU unified hybrid FA page cannot contain pooled GDN state: "
                f"fa_page_bytes={fa_physical_slot_bytes}, "
                f"mamba_required_state_bytes="
                f"{mamba_layout.required_bytes}")

    fa_raw_payload_slot_bytes = (final_block_size *
                                 fa_raw_payload_bytes_per_token)
    fa_layout_padding_slot_bytes = (final_block_slot_bytes -
                                    fa_raw_payload_slot_bytes)
    mamba_slot_padding_bytes: int | None = None
    mamba_layout_padding_bytes: int | None = None
    mamba_slot_tail_padding_bytes: int | None = None
    if mamba_layout is not None:
        mamba_slot_padding_bytes = (final_block_slot_bytes -
                                    mamba_raw_state_bytes)
        mamba_layout_padding_bytes = (mamba_layout.required_bytes -
                                      mamba_raw_state_bytes)
        mamba_slot_tail_padding_bytes = (final_block_slot_bytes -
                                         mamba_layout.required_bytes)

    return {
        "backend": backend_cls.get_name(),
        "block_size_source": block_size_source,
        "input_block_size": input_block_size,
        "backend_supported_kernel_block_sizes": supported,
        "backend_min_page_size": backend_min_page_size,
        "user_block_size_floor": user_block_size_floor,
        "mamba_raw_state_bytes": mamba_raw_state_bytes,
        "mamba_required_state_bytes": mamba_required_state_bytes,
        "fa_raw_payload_bytes_per_token": fa_raw_payload_bytes_per_token,
        "fa_pool_row_tokens": fa_pool_row_tokens,
        "fa_physical_bytes_per_pool_row": fa_physical_bytes_per_pool_row,
        "fa_layout_padding_bytes_per_pool_row":
        fa_layout_padding_bytes_per_pool_row,
        "mamba_fit_block_size": mamba_fit_block_size,
        "final_block_size": final_block_size,
        "fa_raw_payload_slot_bytes": fa_raw_payload_slot_bytes,
        "fa_layout_padding_slot_bytes": fa_layout_padding_slot_bytes,
        "fa_physical_slot_bytes": fa_physical_slot_bytes,
        "final_block_slot_bytes": final_block_slot_bytes,
        "fa_slot_tail_padding_bytes": 0,
        "mamba_layout_padding_bytes": mamba_layout_padding_bytes,
        "mamba_slot_tail_padding_bytes": mamba_slot_tail_padding_bytes,
        "mamba_slot_padding_bytes": mamba_slot_padding_bytes,
    }


def _optional_log_value(value: int | None) -> int | str:
    return "n/a" if value is None else value


def _log_tpu_block_size_derivation(derivation: dict[str, object]) -> None:
    logger.info(
        "TPU block_size derivation path: backend=%s -> "
        "backend_supported_kernel_block_sizes=%s -> input_block_size=%s -> "
        "mamba_raw_state_bytes=%s -> mamba_required_state_bytes=%s -> "
        "fa_pool_row(tokens=%s, raw_payload_per_token=%s + "
        "layout_padding=%s -> physical_bytes=%s) -> mamba_fit_block_size=%s -> "
        "user_block_size_floor=%s -> backend_min_page_size=%s -> "
        "final_block_size=%s (source=%s).",
        derivation["backend"],
        derivation["backend_supported_kernel_block_sizes"],
        derivation["input_block_size"],
        _optional_log_value(derivation["mamba_raw_state_bytes"]),
        _optional_log_value(derivation["mamba_required_state_bytes"]),
        derivation["fa_pool_row_tokens"],
        derivation["fa_raw_payload_bytes_per_token"],
        derivation["fa_layout_padding_bytes_per_pool_row"],
        derivation["fa_physical_bytes_per_pool_row"],
        _optional_log_value(derivation["mamba_fit_block_size"]),
        _optional_log_value(derivation["user_block_size_floor"]),
        derivation["backend_min_page_size"],
        derivation["final_block_size"],
        derivation["block_size_source"],
    )
    logger.info(
        "TPU block_slot derivation path: final_block_size=%s -> "
        "fa_raw_payload_slot_bytes=%s -> "
        "fa_layout_padding_slot_bytes=%s -> fa_physical_slot_bytes=%s -> "
        "cache_slot_bytes=%s -> mamba_layout_padding_bytes=%s -> "
        "mamba_slot_tail_padding_bytes=%s -> mamba_slot_padding_bytes=%s -> "
        "fa_slot_tail_padding_bytes=%s.",
        derivation["final_block_size"],
        derivation["fa_raw_payload_slot_bytes"],
        derivation["fa_layout_padding_slot_bytes"],
        derivation["fa_physical_slot_bytes"],
        derivation["final_block_slot_bytes"],
        _optional_log_value(derivation["mamba_layout_padding_bytes"]),
        _optional_log_value(derivation["mamba_slot_tail_padding_bytes"]),
        _optional_log_value(derivation["mamba_slot_padding_bytes"]),
        derivation["fa_slot_tail_padding_bytes"],
    )


def update_tpu_block_size_and_slot_config(vllm_config: VllmConfig,
                                          backend_cls) -> None:
    cache_config = vllm_config.cache_config
    input_block_size = cache_config.block_size
    if not input_block_size:
        return

    derivation = _derive_tpu_block_slot_config(
        vllm_config,
        backend_cls,
        input_block_size=input_block_size,
    )
    _log_tpu_block_size_derivation(derivation)

    final_block_size = int(derivation["final_block_size"])
    min_page_size = backend_cls.get_min_page_size(vllm_config)
    if final_block_size < min_page_size:
        raise ValueError(
            "TPU FA block_size lowering would violate min_page_size: "
            f"{input_block_size} -> {final_block_size}, "
            f"min_page_size={min_page_size}, "
            f"supported_kernel_block_sizes="
            f"{derivation['backend_supported_kernel_block_sizes']}.")

    if final_block_size != input_block_size:
        logger.info(
            "Adjusting TPU FA block_size for %s: %s -> %s, "
            "block_size_source=%s, supported_kernel_block_sizes=%s.",
            backend_cls.get_name(),
            input_block_size,
            final_block_size,
            derivation["block_size_source"],
            derivation["backend_supported_kernel_block_sizes"],
        )
        cache_config.block_size = final_block_size

    if (vllm_config.model_config.is_hybrid
            and derivation["mamba_raw_state_bytes"] is not None):
        logger.info(
            "Aligning hybrid Mamba KV cache to TPU block slot: "
            "mamba_block_size %s -> %s, mamba_page_size_padded %s -> %s.",
            cache_config.mamba_block_size,
            final_block_size if cache_config.mamba_cache_mode == "align" else
            cache_config.mamba_block_size,
            cache_config.mamba_page_size_padded,
            derivation["final_block_slot_bytes"],
        )
        if cache_config.mamba_cache_mode == "align":
            # State follows the block table; mamba blocks are pool blocks.
            cache_config.mamba_block_size = final_block_size
        cache_config.mamba_page_size_padded = int(
            derivation["final_block_slot_bytes"])
