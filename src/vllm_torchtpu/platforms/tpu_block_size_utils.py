# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


def unified_kv_layout_enabled(vllm_config: "VllmConfig") -> bool:
    """Whether the deployment uses the attention-shaped unified KV pool."""
    del vllm_config
    from vllm_torchtpu import envs as tpu_envs
    return tpu_envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL


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


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ceil_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


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


def _hybrid_mamba_page_size_bytes(vllm_config: VllmConfig) -> int | None:
    model_config = vllm_config.model_config
    if not model_config.is_hybrid:
        return None

    from vllm.model_executor.models import ModelRegistry
    from vllm.v1.kv_cache_interface import MambaSpec

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )
    return MambaSpec(
        # This helper is only used for the unified pool. Its pooled GDN op
        # reads and writes the full TP-local state from one attention-shaped
        # page, even when attention prefill is context-parallel.
        shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=-1,
    ).page_size_bytes


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
    fa_physical_bytes_per_token = _tpu_attention_page_size_bytes(
        vllm_config, backend_cls)
    fa_raw_payload_bytes_per_token = (
        _tpu_attention_raw_payload_bytes_per_token(vllm_config))
    fa_layout_padding_bytes_per_token = (fa_physical_bytes_per_token -
                                         fa_raw_payload_bytes_per_token)

    mamba_raw_state_bytes = _hybrid_mamba_page_size_bytes(vllm_config)
    mamba_fit_block_size: int | None = None
    final_block_size = input_block_size
    block_size_source = "input_block_size"
    if mamba_raw_state_bytes is not None:
        mamba_fit_block_size = _round_up_to_multiple(
            _ceil_div(mamba_raw_state_bytes, fa_physical_bytes_per_token),
            16,
        )
        user_specified = vllm_config.cache_config.user_specified_block_size
        if user_specified and input_block_size < mamba_fit_block_size:
            raise ValueError(
                "Explicit TPU block size is below Mamba fit floor for the "
                "unified KV pool: "
                f"block_size={input_block_size}, "
                f"mamba_fit_block_size={mamba_fit_block_size}")
        if user_specified:
            # An explicit block size that already contains the mamba slot
            # is honored: the fit size is a floor, not a mandate.
            # Disaggregated deployments may also provide an explicit shared
            # block size so the P/D geometries nest. The production pooled
            # path requires a size at or above the Mamba fit floor.
            final_block_size = _align_block_to_backend(input_block_size,
                                                       supported)
            block_size_source = "user_block_size"
        else:
            # Fit-size block (non-pow2 pages are RPA-supported): the
            # attention page contains the whole mamba slot with <1%
            # padding, so mamba state addresses whole token rows of an
            # ordinary attention page. Backends with a fixed kernel block
            # (batched RPA: 256) get the fit size rounded up to a
            # splittable multiple; the extra padding only exists inside
            # state blocks, a small fraction of the pool.
            final_block_size = _align_block_to_backend(mamba_fit_block_size,
                                                       supported)
            block_size_source = "mamba_state_fit"
            if vllm_config.kv_transfer_config is not None:
                # Disaggregated P/D: the KV connector requires the prefill
                # and decode block sizes to nest (one a multiple of the
                # other), which the per-TP fit sizes do not guarantee.
                # Rounding up to a power of two restores that: power-of-two
                # sizes always nest, and each role derives one independently
                # (decode's lower TP yields the larger, block-containing
                # size). The extra padding lives only in state blocks.
                final_block_size = _ceil_power_of_two(final_block_size)
                block_size_source = "mamba_state_fit_pow2"

    fa_physical_slot_bytes = _tpu_attention_slot_size_bytes(
        vllm_config, backend_cls, final_block_size)
    slot_base_bytes = fa_physical_slot_bytes
    if mamba_raw_state_bytes is not None:
        slot_base_bytes = max(mamba_raw_state_bytes, fa_physical_slot_bytes)
    final_block_slot_bytes = _round_up_to_multiple(slot_base_bytes,
                                                   _SLOT_ALIGNMENT_BYTES)

    fa_raw_payload_slot_bytes = (final_block_size *
                                 fa_raw_payload_bytes_per_token)
    fa_layout_padding_slot_bytes = (final_block_size *
                                    fa_layout_padding_bytes_per_token)
    fa_slot_tail_padding_bytes = (final_block_slot_bytes -
                                  fa_physical_slot_bytes)
    mamba_slot_padding_bytes: int | None = None
    if mamba_raw_state_bytes is not None:
        mamba_slot_padding_bytes = (final_block_slot_bytes -
                                    mamba_raw_state_bytes)

    return {
        "backend": backend_cls.get_name(),
        "block_size_source": block_size_source,
        "input_block_size": input_block_size,
        "backend_supported_kernel_block_sizes": supported,
        "mamba_raw_state_bytes": mamba_raw_state_bytes,
        "fa_raw_payload_bytes_per_token": fa_raw_payload_bytes_per_token,
        "fa_physical_bytes_per_token": fa_physical_bytes_per_token,
        "fa_layout_padding_bytes_per_token": fa_layout_padding_bytes_per_token,
        "mamba_fit_block_size": mamba_fit_block_size,
        "final_block_size": final_block_size,
        "fa_raw_payload_slot_bytes": fa_raw_payload_slot_bytes,
        "fa_layout_padding_slot_bytes": fa_layout_padding_slot_bytes,
        "fa_physical_slot_bytes": fa_physical_slot_bytes,
        "slot_base_bytes": slot_base_bytes,
        "slot_alignment_bytes": _SLOT_ALIGNMENT_BYTES,
        "final_block_slot_bytes": final_block_slot_bytes,
        "fa_slot_tail_padding_bytes": fa_slot_tail_padding_bytes,
        "mamba_slot_padding_bytes": mamba_slot_padding_bytes,
    }


def _optional_log_value(value: int | None) -> int | str:
    return "n/a" if value is None else value


def _log_tpu_block_size_derivation(derivation: dict[str, object]) -> None:
    mamba_fit_block_size = derivation["mamba_fit_block_size"]
    mamba_raw_state_bytes = derivation["mamba_raw_state_bytes"]
    if mamba_raw_state_bytes is None:
        mamba_fit_formula = "n/a"
    else:
        mamba_fit_formula = (
            "ceil(%s / %s) rounded_to_16 = %s" %
            (mamba_raw_state_bytes, derivation["fa_physical_bytes_per_token"],
             mamba_fit_block_size))

    logger.info(
        "TPU block_size derivation path: backend=%s -> "
        "backend_supported_kernel_block_sizes=%s -> input_block_size=%s -> "
        "mamba_raw_state_bytes=%s -> "
        "fa_bytes_per_token(raw_payload=%s + layout_padding=%s -> "
        "physical=%s) -> mamba_fit_block_size=%s -> "
        "final_block_size=%s "
        "(source=%s).",
        derivation["backend"],
        derivation["backend_supported_kernel_block_sizes"],
        derivation["input_block_size"],
        _optional_log_value(derivation["mamba_raw_state_bytes"]),
        derivation["fa_raw_payload_bytes_per_token"],
        derivation["fa_layout_padding_bytes_per_token"],
        derivation["fa_physical_bytes_per_token"],
        mamba_fit_formula,
        derivation["final_block_size"],
        derivation["block_size_source"],
    )
    logger.info(
        "TPU block_slot derivation path: final_block_size=%s -> "
        "fa_raw_payload_slot_bytes=%s -> "
        "fa_layout_padding_slot_bytes=%s -> fa_physical_slot_bytes=%s -> "
        "slot_base_bytes=max(mamba_raw_state_bytes=%s, "
        "fa_physical_slot_bytes=%s) = %s -> "
        "slot_alignment_bytes=%s -> "
        "final_block_slot_bytes=round_up(%s, %s) = %s -> "
        "mamba_slot_padding_bytes=%s -> fa_slot_tail_padding_bytes=%s -> "
        "block_size_backend_validation=deferred.",
        derivation["final_block_size"],
        derivation["fa_raw_payload_slot_bytes"],
        derivation["fa_layout_padding_slot_bytes"],
        derivation["fa_physical_slot_bytes"],
        _optional_log_value(derivation["mamba_raw_state_bytes"]),
        derivation["fa_physical_slot_bytes"],
        derivation["slot_base_bytes"],
        derivation["slot_alignment_bytes"],
        derivation["slot_base_bytes"],
        derivation["slot_alignment_bytes"],
        derivation["final_block_slot_bytes"],
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
