# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import functools
import inspect
from typing import Any, ClassVar

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import (VllmConfig, get_current_vllm_config,
                         get_current_vllm_config_or_none)
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType,
                                       MLAAttentionImpl)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group as _get_dcp_group
from vllm_torchtpu.distributed.dcp import get_or_create_dcp_mesh
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.batched_rpa import \
    configs as batched_rpa_configs
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    PCP_STREAMING_RPA_INPUT_PARTITION_SPECS, get_pcp_streaming_mesh,
    invoke_pcp_streaming_op, make_pcp_streaming_rpa_kernel,
    pcp_streaming_jax_op)
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.v2 import kernel as mla_v2_kernel
from vllm_torchtpu.layers.adapter.cp_attention import \
    build_dcp_kernels as _build_dcp_kernels
from vllm_torchtpu.layers.adapter.cp_attention import \
    forward_with_dcp as _forward_with_dcp
from vllm_torchtpu.layers.adapter.cp_mla_attention import (
    all_gather_heads, merge_lse_partials_scatter_heads)
from vllm_torchtpu.layers.core.attention_interface import (
    attention, attention_bundled, mla_attention, ragged_paged_attention,
    ragged_paged_attention_batched)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.layers.core.sequence_layout import \
    is_pcp_streaming_attention_metadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.tpu_info import get_chip_version
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# Temporary: selects the batched_rpa_longctx fork over mainline batched_rpa
if envs.USE_BATCHED_RPA_LONGCTX:
    import vllm_torchtpu.kernels.experimental.batched_rpa_longctx.wrapper as rpa_batched_wrapper
else:
    import vllm_torchtpu.kernels.experimental.batched_rpa.wrapper as rpa_batched_wrapper

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128

_SPARSE_MLA_DCP_INPUT_PARTITION_SPECS = (
    P(DCP_AXIS_NAME),  # kv_cache_nope, this rank's position shard
    P(DCP_AXIS_NAME),  # kv_cache_rope
    P(),  # ql_nope
    P(),  # q_pe
    P(),  # kv_c_normed -- no all-gather: every rank already has it
    P(),  # k_pe
    # This rank's slice of every token's global top-k, already local indices.
    # Sharded, so the global leading dim is `dcp_size * num_tokens`.
    P(DCP_AXIS_NAME),  # local_topk_indices
    P(),  # seq_lens
    P(),  # block_tables (virtual page ordinals)
    P(),  # query_start_loc
    P(),  # request_distribution
)
_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS = (
    P(DCP_AXIS_NAME),  # updated nope cache
    P(DCP_AXIS_NAME),  # updated rope cache
    P(DCP_AXIS_NAME),  # partial attention output over this rank's KV positions
    # Matching log-sum-exps, `-inf` where this rank owns nothing.
    P(DCP_AXIS_NAME),  # partial lse
)

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
TPU_STR_DTYPE_TO_TORCH_DTYPE = {
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


def _is_fp8_kv_cache_dtype(cache_dtype_str: str) -> bool:
    return cache_dtype_str.lower().strip() in frozenset(
        ("fp8", "fp8_e4m3", "fp8_e5m2"))


def _resolve_kv_cache_dtype(cache_dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    normalized = cache_dtype.lower().strip()
    if normalized == "auto":
        raise ValueError(
            "cache_dtype='auto' must be resolved to a concrete torch.dtype "
            "before calling get_kv_cache_shape.")
    dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(normalized)
    if dtype is None:
        raise ValueError(f"Unsupported KV cache dtype string: {cache_dtype}")
    return dtype


def get_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Return number of dtype values packed into a packing_bits lane."""
    bits = torch.empty((), dtype=dtype).element_size() * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"The bit width must divide {packing_bits}, but got {bits} for "
            f"dtype={dtype}.")
    return packing_bits // bits


# HND is SEQ_ALONG_LANE: head_dim, not KV heads, fills the 32-bit words, so an
# fp8 page is genuinely half a bf16 one (b/510425663).
KV_LAYOUT_BY_VLLM_LAYOUT: dict[
    VllmKVCacheLayout, batched_rpa_configs.KVLayout] = {
        VllmKVCacheLayout.LBNHC:
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        VllmKVCacheLayout.BLNHC:
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        VllmKVCacheLayout.LBHNC: batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
        VllmKVCacheLayout.BLHNC: batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
    }


def get_tpu_min_page_size(vllm_config: VllmConfig) -> int:
    max_num_page_per_req = (1024 * 1024 // 2 //
                            vllm_config.scheduler_config.max_num_seqs // 4)
    min_page_size = cdiv(vllm_config.model_config.max_model_len,
                         max_num_page_per_req)
    min_page_size = 1 << (min_page_size - 1).bit_length()
    return min_page_size


def _pallas_rpa_kernel_impl(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    skip_kv_update: bool,
    rpa_func,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    shard: bool = True,
    kv_block_cap: int | None = None,
    use_causal_mask: bool = True,
    kv_layout: batched_rpa_configs.KVLayout | None = None,
) -> tuple[jax.Array, jax.Array]:
    metadata = AttentionMetadata(
        input_positions=
        None,  # NOTE: vLLM applies RoPE before attention, so input_positions is not consumed here.
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_kv_cache, outputs = attention(
        kv_cache,
        query,
        key,
        value,
        metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=sinks,
        attention_chunk_size=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=rpa_func,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        shard=shard,
        kv_block_cap=kv_block_cap,
        use_causal_mask=use_causal_mask,
        kv_layout=kv_layout,
    )
    return new_kv_cache, outputs


def _pallas_rpa_kernel_default(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Default Pallas RPA kernel entry — used by `PallasAttentionBackendImpl`.

    Full signature (not `*args, **kwargs`) is required so `pallas.jax_op` can
    introspect argument types via `_verify_signature`.
    """
    return _pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=ragged_paged_attention,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )


# KV-fetch block cap (tokens) for the tp=1 eagle3 draft's local RPA kernel.
# Capping bkv shrinks the dominant KV scratch tile; the flash kernel loops over more KV chunks.
# 1024 was validated on Llama-3.1-8B (TP=2) and Qwen3-Coder-480b(TP=8). Models with
# many KV heads or longer sequences may need a smaller value to avoid VMEM OOM.
_DRAFT_KV_BLOCK_CAP = 1024


def _pallas_rpa_kernel_local(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Local (non-shard_map) RPA entry for the tp=1 eagle3 draft.

    Identical to `_pallas_rpa_kernel_default` but invokes the kernel WITHOUT
    shard_map (`shard=False`), so the output lands on the worker's own device
    instead of a global chip-0 partition. Also caps the KV-fetch block via
    `kv_block_cap` for speculative decoding draft model.
    """
    return _pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=ragged_paged_attention,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        shard=False,
        kv_block_cap=_DRAFT_KV_BLOCK_CAP,
        use_causal_mask=use_causal_mask,
    )


def _pallas_rpa_kernel_default_bundled(
    kv_cache_bundle: jax.Array,
    layer_idx: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    use_causal_mask: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Default bundled Pallas RPA kernel entry for block-major execution.

    Invoked when `VllmModelWrapperContext.kv_cache_bundle` is set. Matches the
    standard kernel entry point interface, taking the full KV cache bundle as the
    first argument and a dynamic int32 scalar layer_idx as the second argument.
    """
    if sinks is not None:
        # Attention sinks are unsupported in the bundled block-major kernel path.
        raise NotImplementedError(
            "VLLM_TPU_BLOCK_MAJOR_KV=1: attention sinks are not supported "
            "by the bundled RPA path")
    metadata = AttentionMetadata(
        input_positions=None,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_bundle, outputs = attention_bundled(
        kv_cache_bundle,
        layer_idx,
        query,
        key,
        value,
        metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_chunk_size=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
    )
    return new_bundle, outputs


def _pallas_rpa_kernel_batched(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    sinks: jax.Array | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    *,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    use_causal_mask: bool = True,
    kv_layout: batched_rpa_configs.KVLayout,
    decode_query_size: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Batched-RPA Pallas kernel entry — used by `PallasBatchedRPAAttentionBackendImpl`."""
    return _pallas_rpa_kernel_impl(
        kv_cache,
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        sinks,
        q_scale,
        k_scale,
        v_scale,
        mesh=mesh,
        sliding_window=sliding_window,
        skip_kv_update=skip_kv_update,
        rpa_func=(functools.partial(ragged_paged_attention_batched,
                                    decode_query_size=decode_query_size) if
                  decode_query_size > 1 else ragged_paged_attention_batched),
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        use_causal_mask=use_causal_mask,
        kv_layout=kv_layout,
    )


# =========================================================================================
# VLLM Attention Backend
# =========================================================================================


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class PallasAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                            TPU_HEAD_SIZE_ALIGNMENT)
        # Two different RPA kernels have different KV cache layouts:
        # - hd64 (head_dim=64): K/V packed along head_dim
        # - v3 (head_dim!=64): K/V packed along heads
        # The Pallas kernels expect a 5D KV cache: [L, S, Kx2 / kv_packing, kv_packing, H]
        # where Kx2 = num_kv_heads for hd64 and Kx2 = num_kv_heads * 2 for v3.
        use_hd64 = (head_size == 64)
        # vLLM's OffloadingConnectorWorker.register_kv_caches probes this
        # method without a concrete dtype to discover the num_blocks logical
        # dimension (it only reads test_shape.index(num_blocks); see
        # vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py).
        # num_blocks is at dim 0 in our layout regardless of dtype, so return
        # a placeholder shape that satisfies the probe without resolving the
        # dtype.
        if (isinstance(cache_dtype_str, str)
                and cache_dtype_str.lower().strip() == "auto"):
            num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
            return (num_blocks, block_size, num_kv_heads_x2, 1,
                    padded_head_size)
        kv_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        kv_packing = get_dtype_packing(kv_dtype)

        num_kv_heads_x2 = num_kv_heads if use_hd64 else num_kv_heads * 2
        num_kv_heads_x2 = cdiv(num_kv_heads_x2, kv_packing) * kv_packing
        return (
            num_blocks,
            block_size,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            padded_head_size,
        )

    @classmethod
    def get_kv_cache_page_size_bytes(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        """Return one Pallas KV-cache page size including layout padding.

        Sized through `cls` so a subclass reports what it actually allocates.
        """
        dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        shape = cls.get_kv_cache_shape(
            1,
            block_size,
            num_kv_heads,
            head_size,
            dtype,
        )
        num_elements = functools.reduce(lambda x, y: x * y, shape, 1)
        return num_elements * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the PallasMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return get_tpu_min_page_size(vllm_config)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            return (VllmKVCacheLayout.BLNHC, )
        return (VllmKVCacheLayout.LBNHC, )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # TPU kernels consume packed records, including head/lane padding.
        # Publish their byte geometry so vLLM computes native L/B strides
        # from the same page size as the TPU allocation.
        page_bytes = cls.get_kv_cache_page_size_bytes(spec.block_size,
                                                      spec.num_kv_heads,
                                                      spec.head_size,
                                                      spec.dtype)
        return dataclasses.replace(spec,
                                   num_head_slots=1,
                                   state_content_bytes=page_bytes //
                                   spec.block_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False, ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO: This is a temporary fix for vmem OOM.
        # For long model length, we use 16 page-size to avoid too much
        # VMEM spill. A more robust solution should be implemented to
        # handle VREG spills.
        if vllm_config.model_config.max_model_len > 8192:
            return 16
        page_size = next_power_of_2(
            vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


@register_backend(AttentionBackendEnum.CUSTOM)
class PallasBatchedRPAAttentionBackend(PallasAttentionBackend):
    """Pallas attention backend wired to the batched-RPA kernel.

    Registered under `AttentionBackendEnum.CUSTOM` by
    `TpuPlatform.pre_register_and_update`; user opts in via
    `--attention-backend CUSTOM`. The Impl subclass sets `use_batched_rpa=True`,
    which plumbs through `_pallas_rpa_kernel` → `attention()` →
    `sharded_ragged_paged_attention()` to pick the batched kernel function.
    Any multiple of 128 is a valid page size for this kernel;
    `get_preferred_block_size` is overridden so the auto-selected default (no
    `--block-size` given) stays at 256. Under `VLLM_KV_CACHE_LAYOUT=HND`, the
    non-PCP kernel accepts only page size 128; PCP streaming retains support
    for every multiple of 128 listed below.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["PallasBatchedRPAAttentionBackendImpl"]:
        return PallasBatchedRPAAttentionBackendImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC, VllmKVCacheLayout.LBHNC)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        is_auto = (isinstance(cache_dtype_str, str)
                   and cache_dtype_str.lower().strip() == "auto")
        # Resolve the layout last: the dtype-less probe and hd64 callers
        # do not need a current vLLM config.
        if (head_size == 64
                or is_auto or KV_LAYOUT_BY_VLLM_LAYOUT[get_current_vllm_config(
                ).cache_config.get_resolved_kv_cache_layout()]
                is not batched_rpa_configs.KVLayout.SEQ_ALONG_LANE):
            return PallasAttentionBackend.get_kv_cache_shape(
                num_blocks, block_size, num_kv_heads, head_size,
                cache_dtype_str)
        torch_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        # Resolve the chip here rather than in the kernel: the scheduler
        # process calls this and owns no chip, so the wrapper must not query a
        # live device.
        return rpa_batched_wrapper.get_kv_cache_shape(
            total_num_pages=num_blocks,
            page_size=block_size,
            actual_num_kv_heads=num_kv_heads,
            actual_head_dim=head_size,
            kv_dtype=pallas.pallas.TORCH_TO_JAX_DTYPE_MAP[torch_dtype],
            kv_layout=KV_LAYOUT_BY_VLLM_LAYOUT[get_current_vllm_config(
            ).cache_config.get_resolved_kv_cache_layout()],
            chip_version=get_chip_version(),
        )

    @staticmethod
    def get_supported_kernel_block_sizes():
        # Needs a current vLLM config: the layout falls through to the KV
        # connector when `VLLM_KV_CACHE_LAYOUT` is unset.
        if envs.USE_BATCHED_RPA_LONGCTX:
            return [128, 256, 512, 1024, 2048, 4096]
        # SEQ_ALONG_LANE maps a page onto one 128-lane tile; `RpaConfigs`
        # rejects any other page size in non-PCP mode. PCP streaming supports
        # each 128-aligned page size listed below.
        if get_current_vllm_config().cache_config.get_resolved_kv_cache_layout(
        ) is VllmKVCacheLayout.LBHNC:
            parallel_config = get_current_vllm_config().parallel_config
            if parallel_config.prefill_context_parallel_size > 1:
                return [128, 256, 512, 1024, 2048, 4096]
            return [128]
        return [256]


class PallasAttentionBackendImpl(AttentionImpl):
    _kernel_instance_counter = 0
    # Registry of shared custom ops keyed to avoid registering duplicate Pallas
    # kernels for layers with identical configs.
    # Mapping of (sliding_window, mesh, q_scale, k_scale, v_scale) -> custom op
    _kernel_registry: dict = {}

    # Each Impl subclass points at its own RPA kernel entry function. Wrapped
    # with `staticmethod` so attribute access through `self` doesn't bind it
    # as a method. Subclasses override this to select a different kernel.
    _kernel_entry: ClassVar = staticmethod(_pallas_rpa_kernel_default)
    _kernel_op_prefix: ClassVar[str] = "pallas::rpa_kernel"
    decode_query_size: int = 1

    # Bundled (block-major) RPA kernel entry: accepts the full KV cache bundle
    # and a dynamic scalar layer_idx. Registered under a dedicated prefix to avoid
    # collisions with layer-major ops in the shared registry.
    _kernel_entry_bundled: ClassVar = staticmethod(
        _pallas_rpa_kernel_default_bundled)
    _kernel_op_prefix_bundled: ClassVar[str] = "pallas::rpa_kernel_bundled"

    # Class-level registry of bundled kernel instances. Layers sharing identical
    # configurations reuse a single custom op, emitting one shared OpOverload node
    # across the FX graph rather than duplicating JAX tracing and MLIR compilation.
    _bundled_kernel_registry: ClassVar[dict] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.use_causal_mask = kwargs.get("use_causal_mask", True)

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        self.kv_cache_quantized_dtype = None
        # Only set kv_cache_quantized_dtype for fp8 KV cache
        if _is_fp8_kv_cache_dtype(kv_cache_dtype):
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                kv_cache_dtype.lower().strip()]

        # Store sinks for attention sink optimization
        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")
        # Cache the resolved config value during construction: compiled
        # forwards have no active vLLM config and must not look it up.
        self.kv_layout = KV_LAYOUT_BY_VLLM_LAYOUT[get_current_vllm_config(
        ).cache_config.get_resolved_kv_cache_layout()]
        self._pool_is_seq_along_lane = (
            isinstance(self, PallasBatchedRPAAttentionBackendImpl)
            and self.kv_layout is batched_rpa_configs.KVLayout.SEQ_ALONG_LANE)
        self.rpa_kernel = None
        # Populated by initialize_kernel() before torch.compile traces the
        # model. Forward can then reuse the custom op without resolving a PCP
        # mesh from inside Dynamo's fullgraph capture.
        self._kernel_config_cache: dict = {}
        # Set by initialize_kernel() when PCP routes this layer to the
        # streaming kernel.
        self._pcp_streaming = False
        # DCP (Decode Context Parallelism) state. Populated by initialize_kernel()
        # when decode_context_parallel_size > 1.
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.rpa_dcp_cache_kernel = None  # CACHE_ONLY pass, returns (out, lse)
        self.rpa_dcp_new_kernel = None  # NEW_TOKENS_ONLY pass, returns (out, lse)
        # Block-major bundled kernel and pre-allocated layer index. Initialized by
        # setup_bundled() after bundle allocation and prior to torch.compile tracing.
        # During profile_run, ctx.kv_cache_bundle is None and forward falls back to
        # standard execution.
        self.rpa_kernel_bundled = None
        self._bundle_layer_idx_tensor: torch.Tensor | None = None

    @classmethod
    def _allocate_kernel_instance_id(cls) -> int:
        kernel_instance_id = cls._kernel_instance_counter
        cls._kernel_instance_counter += 1
        return kernel_instance_id

    def _build_rpa_kernel(
        self,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        skip_kv_update: bool = False,
        use_pcp_streaming: bool = False,
        cp_kv_cache_interleave_size: int = 0,
    ):
        ctx = get_vllm_model_wrapper_context()
        vllm_config = ctx.vllm_config
        max_model_len = (vllm_config.model_config.max_model_len
                         if use_pcp_streaming and vllm_config is not None else
                         None)
        pcp_kv_layout = (self.kv_layout if use_pcp_streaming else
                         batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE)
        config_key = (self._kernel_op_prefix, self.sliding_window, q_scale,
                      k_scale, v_scale, use_pcp_streaming,
                      cp_kv_cache_interleave_size, max_model_len,
                      pcp_kv_layout, self.decode_query_size)
        existing = self._kernel_config_cache.get(config_key)
        if existing is not None:
            return existing

        # Reuse an existing custom op if one with the same config already exists.
        # `_kernel_op_prefix` is included so subclasses (e.g. the batched RPA
        # variant) don't collide with the base impl in the shared registry.
        # `skip_kv_update` is part of the key so KV-sharing (read-only) layers
        # get their own kernel variant and never share an op with KV-owning
        # layers of an otherwise-identical config.
        mesh, op_mesh, input_partition_specs = self._select_kernel_mesh(
            ctx.mesh, use_pcp_streaming)
        registry_key = (self._kernel_op_prefix,
                        self.sliding_window, self.scale, self.logits_soft_cap,
                        id(mesh), q_scale, k_scale, v_scale, skip_kv_update,
                        use_pcp_streaming, cp_kv_cache_interleave_size,
                        max_model_len, self.use_causal_mask, pcp_kv_layout,
                        self.decode_query_size)
        existing = self._kernel_registry.get(registry_key)
        if existing is not None:
            self._kernel_config_cache[config_key] = existing
            return existing

        kernel_instance_id = self._allocate_kernel_instance_id()
        op_name = f"{self._kernel_op_prefix}_{kernel_instance_id}"

        if use_pcp_streaming:
            pcp_make_kwargs = dict(
                mesh=mesh,
                sliding_window=self.sliding_window,
                sm_scale=self.scale,
                soft_cap=self.logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                skip_kv_update=skip_kv_update,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                kv_layout=pcp_kv_layout,
            )
            wrapped_fn = make_pcp_streaming_rpa_kernel(**pcp_make_kwargs)
        else:
            entry_params = inspect.signature(self._kernel_entry).parameters
            entry_kwargs = {}
            if "kv_layout" in entry_params:
                entry_kwargs["kv_layout"] = self.kv_layout
            if "decode_query_size" in entry_params:
                entry_kwargs["decode_query_size"] = self.decode_query_size
            wrapped_fn = functools.partial(
                self._kernel_entry,
                mesh=mesh,
                sliding_window=self.sliding_window,
                sm_scale=self.scale,
                soft_cap=self.logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                skip_kv_update=skip_kv_update,
                use_causal_mask=self.use_causal_mask,
                **entry_kwargs,
            )

        # Register as a custom op to mark it as an op boundary in Dynamo.
        # This prevents torch.compile from tracing into the Pallas kernel internals.
        #
        # Kernel-iteration mode runs the op eagerly between compiled pieces
        # (splitting_ops); donating kv_cache there kills the buffer that
        # rpa_kernel_impl's copy_ writes back into. Inside a compiled full
        # graph XLA rewires the alias, so donation is only safe when the op
        # is compiled.
        rpa_donate_argnums = (None if envs.TPU_KERNEL_ITER_MODE else (0, ))
        if use_pcp_streaming:
            rpa_kernel_op = pcp_streaming_jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=rpa_donate_argnums,
                mesh=op_mesh,
                input_partition_specs=input_partition_specs)
            if envs.TPU_KERNEL_ITER_MODE:
                from vllm_torchtpu.compilation import kernel_reload

                def _rebuild_pcp_callable(name=op_name,
                                          make_kwargs=pcp_make_kwargs,
                                          op_mesh=op_mesh,
                                          specs=input_partition_specs,
                                          donate=rpa_donate_argnums):
                    import importlib
                    adapter = importlib.import_module(
                        "vllm_torchtpu.kernels.experimental."
                        "pcp_streaming_rpa.vllm_adapter")
                    fn = adapter.make_pcp_streaming_rpa_kernel(**make_kwargs)
                    return adapter.build_pcp_streaming_callable(
                        name,
                        fn,
                        donate_argnums=donate,
                        mesh=op_mesh,
                        input_partition_specs=specs)

                kernel_reload.register_builder(op_name, _rebuild_pcp_callable)
        else:
            rpa_kernel_op = pallas.jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=rpa_donate_argnums,
                mesh=op_mesh,
                input_partition_specs=(input_partition_specs))

        # We must overwrite the default fake implementation as vLLM uses dynamic
        # dimensions for the query.
        def _fake_rpa_op(kv_cache: torch.Tensor, query: torch.Tensor, *args,
                         **kwargs):
            return torch.empty_like(kv_cache), torch.empty_like(query)

        rpa_kernel_op.register_fake(_fake_rpa_op)

        def rpa_kernel_impl(kv_cache, *args, **kwargs):
            if use_pcp_streaming:
                new_kv_cache, output = invoke_pcp_streaming_op(
                    rpa_kernel_op, kv_cache, args, kwargs)
            else:
                new_kv_cache, output = rpa_kernel_op(kv_cache, *args, **kwargs)
            if new_kv_cache.shape != kv_cache.shape:
                raise RuntimeError(
                    "RPA kernel returned an incompatible KV cache shape: "
                    f"expected {tuple(kv_cache.shape)}, got "
                    f"{tuple(new_kv_cache.shape)}.")
            # Plain donation + copy_ writeback: XLA lifts the copy_ input
            # mutation and aliases the op output onto the donated pool, so this
            # compiles to an in-place pool update.
            kv_cache.copy_(new_kv_cache)
            return output

        self._kernel_registry[registry_key] = rpa_kernel_impl
        self._kernel_config_cache[config_key] = rpa_kernel_impl
        return rpa_kernel_impl

    def _run_dcp_forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return _forward_with_dcp(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            sliding_window=self.sliding_window,
            sm_scale=self.scale,
            logits_soft_cap=self.logits_soft_cap,
            kv_cache_quantized_dtype=self.kv_cache_quantized_dtype,
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
        )

    def _validate_pcp_streaming_support(self, skip_kv_update: bool) -> None:
        unsupported_features = []
        if self.kv_layout is batched_rpa_configs.KVLayout.SEQ_ALONG_LANE:
            if not isinstance(self, PallasBatchedRPAAttentionBackendImpl):
                unsupported_features.append(
                    "VLLM_KV_CACHE_LAYOUT=HND with a non-CUSTOM backend")
            if self.kv_cache_quantized_dtype is None:
                unsupported_features.append(
                    "VLLM_KV_CACHE_LAYOUT=HND without an FP8 KV cache")
        if self.sinks is not None:
            unsupported_features.append("attention sinks")
        if self.logits_soft_cap is not None:
            unsupported_features.append("logits soft cap")
        if skip_kv_update:
            unsupported_features.append("skip_kv_update")
        if unsupported_features:
            raise NotImplementedError(
                "PCP streaming attention does not support: "
                f"{', '.join(unsupported_features)}.")

    @staticmethod
    def _select_kernel_mesh(default_mesh, use_pcp_streaming: bool):
        if not use_pcp_streaming:
            return default_mesh, None, None

        pcp_mesh = get_pcp_streaming_mesh()
        return pcp_mesh, pcp_mesh, PCP_STREAMING_RPA_INPUT_PARTITION_SPECS

    def _build_bundled_kernel(
        self,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
    ):
        """Constructs and caches the bundled RPA custom op via `pallas.jax_op`.

        Uses in-place buffer donation on the full bundle (arg 0) with a dynamic
        `layer_idx` scalar. Layers sharing identical kernel configurations reuse
        the cached op via `_bundled_kernel_registry`.
        """
        # Fail closed when the layer-major kernel is overridden (e.g. subclass
        # ClassVar like batched RPA, or per-instance rebinds like single-device
        # local kernels) without a corresponding bundled counterpart, preventing
        # silent fallback to default kernels. Read via `self` to detect instance overrides.
        if (self._kernel_entry is not PallasAttentionBackendImpl._kernel_entry
                and self._kernel_entry_bundled
                is PallasAttentionBackendImpl._kernel_entry_bundled):
            raise NotImplementedError(
                "VLLM_TPU_BLOCK_MAJOR_KV=1: "
                f"{type(self).__name__} overrides the layer-major RPA "
                "kernel but has no bundled variant; refusing to fall "
                "back to the default bundled kernel.")
        ctx = get_vllm_model_wrapper_context()
        mesh = ctx.mesh
        registry_key = (self._kernel_op_prefix_bundled, self.sliding_window,
                        self.scale, self.logits_soft_cap, id(mesh), q_scale,
                        k_scale, v_scale, self.use_causal_mask)
        existing = self._bundled_kernel_registry.get(registry_key)
        if existing is not None:
            return existing

        kernel_instance_id = self._allocate_kernel_instance_id()
        op_name = f"{self._kernel_op_prefix_bundled}_{kernel_instance_id}"
        wrapped_fn = functools.partial(
            self._kernel_entry_bundled,
            mesh=mesh,
            sliding_window=self.sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=self.scale,
            soft_cap=self.logits_soft_cap,
            use_causal_mask=self.use_causal_mask,
        )

        # Eager kernel iteration mode runs without JAX tracing; omit donation
        # to prevent prematurely releasing the buffer needed for copy_ writeback.
        bundled_donate_argnums = (None if envs.TPU_KERNEL_ITER_MODE else (0, ))
        rpa_kernel_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=bundled_donate_argnums,
        )

        def _fake_rpa_op(kv_cache_bundle: torch.Tensor,
                         layer_idx: torch.Tensor, query: torch.Tensor, *args,
                         **kwargs):
            return (torch.empty_like(kv_cache_bundle), torch.empty_like(query))

        rpa_kernel_op.register_fake(_fake_rpa_op)

        def rpa_kernel_bundled_impl(kv_cache_bundle, *args, **kwargs):
            new_bundle, output = rpa_kernel_op(kv_cache_bundle, *args,
                                               **kwargs)
            if new_bundle.shape != kv_cache_bundle.shape:
                raise RuntimeError(
                    "Bundled RPA kernel returned an incompatible bundle "
                    f"shape: expected {tuple(kv_cache_bundle.shape)}, got "
                    f"{tuple(new_bundle.shape)}.")
            # In-place writeback: XLA aliases the donated input buffer to the op
            # output, compiling the copy_ operation into an in-place HBM update.
            kv_cache_bundle.copy_(new_bundle)
            return output

        self._bundled_kernel_registry[registry_key] = rpa_kernel_bundled_impl
        return rpa_kernel_bundled_impl

    def setup_bundled(self, layer_idx: int, bundle_device: torch.device,
                      layer: "AttentionLayer") -> None:
        """Initializes the bundled kernel and pre-allocates the layer index tensor.

        Called by the TPU runner immediately after bundle allocation and BEFORE
        `torch.compile` traces `capture_model`. Mesh, sliding window, and scaling
        parameters are bound into the compiled op.
        """
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value != 0.0 and v_scale_value != 0.0:
                k_scale = k_scale_value
                v_scale = v_scale_value
        # Pre-allocate layer index tensor on device to avoid per-step Host-to-Device copies.
        self._bundle_layer_idx_tensor = torch.tensor(int(layer_idx),
                                                     dtype=torch.int32,
                                                     device=bundle_device)
        self.rpa_kernel_bundled = self._build_bundled_kernel(
            q_scale, k_scale, v_scale)

    def initialize_kernel(self, layer: AttentionLayer) -> None:
        """Pre-build the RPA kernel before torch.compile traces the model.

        Must be called after model weights are loaded (so scales are known)
        and after the mesh is available via the model wrapper context, but
        before forward().
        """
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value != 0.0 and v_scale_value != 0.0:
                k_scale = k_scale_value
                v_scale = v_scale_value
        # KV-sharing (cross-layer) layers reuse the target layer's K/V already
        # written into the shared cache, so their kernel must attend without
        # writing. See `PallasAttentionBackendImpl.__init__` and the runner's
        # cache aliasing for how the shared cache tensor is set up.
        skip_kv_update = self.kv_sharing_target_layer_name is not None
        ctx = get_vllm_model_wrapper_context()
        vllm_config = ctx.vllm_config
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)
        pcp_configured = (parallel_config is not None and
                          parallel_config.prefill_context_parallel_size > 1)
        if pcp_configured:
            self._validate_pcp_streaming_support(skip_kv_update)
            self.decode_query_size = 1
            self._pcp_streaming = True
            self.rpa_kernel = self._build_rpa_kernel(
                q_scale,
                k_scale,
                v_scale,
                skip_kv_update=skip_kv_update,
                use_pcp_streaming=True,
                cp_kv_cache_interleave_size=parallel_config.
                cp_kv_cache_interleave_size,
            )
            return

        dcp_configured = (parallel_config is not None and getattr(
            parallel_config, 'decode_context_parallel_size', 1) > 1)
        if dcp_configured:
            # TODO(kwang3939): forward decode_query_size to the DCP ops
            self.decode_query_size = 1
            dcp_group = _get_dcp_group()
            if dcp_group is not None:
                self.dcp_world_size = int(dcp_group.world_size)
                self.dcp_rank = int(dcp_group.rank_in_group)
                self.rpa_dcp_cache_kernel, self.rpa_dcp_new_kernel = (
                    _build_dcp_kernels(sliding_window=self.sliding_window,
                                       sm_scale=self.scale,
                                       logits_soft_cap=self.logits_soft_cap,
                                       q_scale=q_scale,
                                       k_scale=k_scale,
                                       v_scale=v_scale,
                                       cp_group_size=self.dcp_world_size,
                                       cp_rank=self.dcp_rank))
            return

        self.rpa_kernel = self._build_rpa_kernel(q_scale,
                                                 k_scale,
                                                 v_scale,
                                                 skip_kv_update=skip_kv_update)

    def runs_batched_rpa_schedule(self) -> bool:
        """Whether forward runs the batched RPA kernel, which keeps a whole
        step's (query block, KV block) schedule in SMEM. The PCP streaming
        kernel, the DCP kernels, the long-context fork, the head-dim-64
        kernel and the bundled kernel keep no such table."""
        # TODO(#1015): remove, with the runner's schedule check, once the
        # batched RPA kernel enforces its own schedule bound.
        other_kernel = (
            self._pcp_streaming  # PCP: the streaming kernel
            or self.dcp_world_size > 1  # DCP: the long-context kernels
            or self.rpa_kernel_bundled is not None  # block-major KV: bundled
            or envs.USE_BATCHED_RPA_LONGCTX  # the long-context fork
            or self.head_size == 64  # the head-dim-64 kernel (see use_hd64)
        )
        return (self._kernel_entry is _pallas_rpa_kernel_batched
                and not other_kernel)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """Process sinks after model loading - convert to float32 as required by RPA kernel."""
        if self.sinks is not None:
            # RPA v3 kernel requires sinks to be float32
            self.sinks = torch.nn.Parameter(self.sinks.to(torch.float32),
                                            requires_grad=False)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads_x2 // kv_packing,
                 kv_packing, padded_head_size] (preferred)
                or legacy 4D
                [num_blocks, block_size, num_kv_heads_x2, padded_head_size]
                or, under SEQ_ALONG_LANE,
                [num_blocks, num_kv_heads_x2, padded_head_size // kv_packing,
                 kv_packing, block_size]
            attn_metadata: Metadata for attention.
            output: buffer written in place, shape
                = [num_tokens, num_heads, head_size_v]
        Returns:
            shape = [num_tokens, num_heads, head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for PallasAttentionBackendImpl")

        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            if output is None:
                output = torch.ones_like(query)
            return output

        query_dim = query.dim()
        if query_dim == 3:
            q_len = query.shape[0]
            q_compute_dim = query.shape[1] * query.shape[2]
        else:
            q_len, q_compute_dim = query.shape

        if key.dim() == 3:
            k_len = key.shape[0]
            k_compute_dim = key.shape[1] * key.shape[2]
        else:
            k_len, k_compute_dim = key.shape

        assert q_compute_dim == self.head_size * self.num_heads
        assert k_compute_dim == self.head_size * self.num_kv_heads

        query = query.view(q_len, self.num_heads, self.head_size)
        key = key.view(k_len, self.num_kv_heads, self.head_size)
        value = value.view(k_len, self.num_kv_heads, self.head_size)
        assert key.shape == value.shape

        # A layer whose head_size is narrower than the page it writes into —
        # a DFlash draft layer sharing the target's unified pool, whose page
        # geometry comes from whichever attention spec defined the pool (see
        # `kv_cache_materializer._pool_attention_geometry`). Pad q/k/v out to
        # the page width so the kernel's slot writes line up, and slice the
        # result back below. Zeros contribute nothing to the QK dot product,
        # `self.scale` comes from the real head_size and is passed separately,
        # and the padded V columns are dropped — so the output is unchanged.
        #
        # Only the head dim is reconciled. A layer whose `num_kv_heads`
        # disagrees with the pool's would write the wrong slots and padding
        # cannot fix that, so the two must already agree.
        # SEQ_ALONG_LANE ends in page_size; its head_dim words are dims 2-3.
        if self._pool_is_seq_along_lane:
            pool_head_dim = kv_cache.shape[2] * kv_cache.shape[3]
        else:
            pool_head_dim = kv_cache.shape[-1]
        if pool_head_dim > self.head_size:
            pad_size = pool_head_dim - self.head_size
            query = torch.nn.functional.pad(query, (0, pad_size))
            key = torch.nn.functional.pad(key, (0, pad_size))
            value = torch.nn.functional.pad(value, (0, pad_size))
        if self.kv_cache_quantized_dtype:
            k_scale_value = layer._k_scale_float
            v_scale_value = layer._v_scale_float
            if k_scale_value == 0.0 or v_scale_value == 0.0:
                raise ValueError(
                    "k_scale_float and v_scale_float must be non-zero")
            key, value = quantize_kv(self.kv_cache_quantized_dtype, key, value,
                                     k_scale_value, v_scale_value)

        sink = self.sinks
        ctx = get_vllm_model_wrapper_context()
        vllm_config = ctx.vllm_config
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)

        if self.dcp_world_size > 1:
            if sink is not None or ctx.kv_cache_bundle is not None:
                raise NotImplementedError(
                    "DCP attention does not support attention sinks or "
                    "the bundled KV cache path.")
            outputs = self._run_dcp_forward(layer, query, key, value, kv_cache,
                                            attn_metadata)
            if outputs.shape[-1] > self.head_size:
                outputs = outputs[..., :self.head_size]
            if query_dim == 2:
                outputs = outputs.reshape(q_len,
                                          self.num_heads * self.head_size)
            if output is not None:
                output.copy_(outputs)
            return outputs

        use_pcp_streaming = is_pcp_streaming_attention_metadata(attn_metadata)
        pcp_configured = (parallel_config is not None and
                          parallel_config.prefill_context_parallel_size > 1)
        if pcp_configured and not use_pcp_streaming:
            raise RuntimeError(
                "PCP is configured, but attention metadata does not use "
                "the PCP streaming sequence layout.")
        skip_kv_update = self.kv_sharing_target_layer_name is not None
        if use_pcp_streaming:
            self._validate_pcp_streaming_support(skip_kv_update)
        cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
            if use_pcp_streaming else 0)
        rpa_kernel = self._build_rpa_kernel(
            None,
            layer._k_scale_float if self.kv_cache_quantized_dtype else None,
            layer._v_scale_float if self.kv_cache_quantized_dtype else None,
            skip_kv_update=skip_kv_update,
            use_pcp_streaming=use_pcp_streaming,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

        # TODO (geyuhao) the support of this API is pending discussion.
        # This line will only influence performance, not functionality
        # # Mark kv_cache avaliable for donation
        # pallas.set_buffer_donor_(kv_cache, True)

        # Block-major execution path: when a KV cache bundle is mounted on the context,
        # route through the bundled RPA kernel. The per-layer `kv_cache` argument
        # passed by vLLM (a strided view) is bypassed in favor of direct, in-place
        # updates to the full bundle. During profile_run, `ctx.kv_cache_bundle` is None;
        # Dynamo guards on this check trigger retracing once the bundle is initialized.
        if ctx.kv_cache_bundle is not None:
            if use_pcp_streaming or skip_kv_update:
                raise NotImplementedError(
                    "VLLM_TPU_BLOCK_MAJOR_KV=1: the bundled RPA path does "
                    "not support PCP streaming or KV sharing")
            outputs = self.rpa_kernel_bundled(
                ctx.kv_cache_bundle,
                self._bundle_layer_idx_tensor,
                query,
                key,
                value,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                sink,
            )
        else:
            # Call the operator
            outputs = rpa_kernel(
                kv_cache,
                query,
                key,
                value,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
                sink,
            )

        # Drop the padding added above so callers see the layer's own width.
        if outputs.shape[-1] > self.head_size:
            outputs = outputs[..., :self.head_size]
        # TODO (geyuhao) ideally we don't want this
        if not torch.compiler.is_compiling():
            synchronize_tensors(ctx.kv_cache_bundle if ctx.kv_cache_bundle
                                is not None else kv_cache,
                                wait=False)

        if query_dim == 2:
            outputs = outputs.reshape(q_len, self.num_heads * self.head_size)

        if output is not None:
            output.copy_(outputs)
        return outputs


class PallasBatchedRPAAttentionBackendImpl(PallasAttentionBackendImpl):
    """Impl variant that dispatches to the batched RPA Pallas kernel."""
    _kernel_entry = staticmethod(_pallas_rpa_kernel_batched)
    _kernel_op_prefix = "pallas::rpa_kernel_batched"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        vllm_config = get_current_vllm_config_or_none()
        spec = vllm_config.speculative_config if vllm_config else None
        # Run K + 1 verify tokens in the DECODE stage.
        # head_dim 64 is rerouted to the hd64 kernel.
        if (spec is not None and envs.USE_BATCHED_RPA_LONGCTX
                and self.head_size != 64):
            self.decode_query_size = spec.num_speculative_tokens + 1


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class PallasMLAttentionBackend(AttentionBackend):
    """TPU attention backend utilizing customized Pallas kernels for DeepSeek MLA.
    """
    supported_kv_cache_dtypes = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_ds_mla",
    ]

    # DeepSeek-V4 MLA packed-cache constants (64-dim RoPE tail, 64-wide UE8M0 blocks).
    _DS_MLA_ROPE_HEAD_DIM = 64
    _DS_MLA_QUANT_BLOCK = 64

    @staticmethod
    def _ds_mla_packed_width(nope_dim: int, rope_head_dim: int,
                             quant_block: int) -> int:
        """Bytes per token in the packed sparse (head_dim=512) KV cache."""
        # nope fp8 (1B) + rope bf16 (2B) + UE8M0 block scale (1B)
        return nope_dim + rope_head_dim * 2 + (nope_dim // quant_block)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def is_mla() -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["PallasMLAttentionBackendImpl"]:
        return PallasMLAttentionBackendImpl

    @staticmethod
    def _is_ds_mla_packed_cache(cache_dtype_str: str | torch.dtype) -> bool:
        # Recognizes both raw "fp8_ds_mla" string and resolved torch.uint8 dtype.
        if isinstance(cache_dtype_str,
                      str) and cache_dtype_str.lower().strip() == "fp8_ds_mla":
            return True
        return isinstance(cache_dtype_str,
                          torch.dtype) and cache_dtype_str == torch.uint8

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
        *,
        head_size_is_packed_width: bool = False,
    ) -> tuple[int, ...]:
        """Shape of one MLA KV cache tensor.

        Set `head_size_is_packed_width` when `head_size` is already the packed
        byte width; callers passing `kv_lora_rank` leave it False.
        """
        if (isinstance(cache_dtype_str, str)
                and cache_dtype_str.lower().strip() == "auto"):
            return (num_blocks, block_size, 1, cdiv(head_size, 128) * 128)
        if PallasMLAttentionBackend._is_ds_mla_packed_cache(cache_dtype_str):
            # Packed layout is [nope fp8 | rope bf16 | UE8M0 scales] padded to 128-aligned minor dim.
            if head_size_is_packed_width:
                packed_width = head_size
            else:
                rope_head_dim = PallasMLAttentionBackend._DS_MLA_ROPE_HEAD_DIM
                quant_block = PallasMLAttentionBackend._DS_MLA_QUANT_BLOCK
                nope_dim = head_size - rope_head_dim
                packed_width = (PallasMLAttentionBackend._ds_mla_packed_width(
                    nope_dim, rope_head_dim, quant_block))
            kv_packing = get_dtype_packing(torch.uint8)
            return mla_v2_kernel.get_kv_cache_shape(
                total_num_pages=num_blocks,
                page_size=block_size,
                kv_dim=packed_width,
                kv_dtype=None,
                kv_packing=kv_packing,
            )
        kv_dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        kv_packing = get_dtype_packing(kv_dtype)
        return mla_v2_kernel.get_kv_cache_shape(
            total_num_pages=num_blocks,
            page_size=block_size,
            kv_dim=head_size,
            kv_dtype=None,
            kv_packing=kv_packing,
        )

    @staticmethod
    def get_sparse_kv_cache_specs(
        num_blocks: int,
        block_size: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[SparseMLAKVCacheSpec, SparseMLAKVCacheSpec]:
        """The (nope, rope) specs for one sparse-MLA layer."""
        rope_dim = PallasMLAttentionBackend._DS_MLA_ROPE_HEAD_DIM
        kv_packing = get_dtype_packing(
            _resolve_kv_cache_dtype(cache_dtype_str))
        return (
            SparseMLAKVCacheSpec.create(
                KVCacheType.NOPE,
                KVCacheLayout(envs.TPU_SPARSE_MLA_NOPE_LAYOUT),
                num_blocks,
                block_size,
                head_size - rope_dim,
                kv_packing,
            ),
            SparseMLAKVCacheSpec.create(
                KVCacheType.ROPE,
                KVCacheLayout(envs.TPU_SPARSE_MLA_ROPE_LAYOUT),
                num_blocks,
                block_size,
                rope_dim,
                kv_packing,
            ),
        )

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        shape = PallasMLAttentionBackend.get_kv_cache_shape(
            1,
            block_size,
            num_kv_heads,
            head_size,
            dtype,
        )
        num_elements = functools.reduce(lambda x, y: x * y, shape, 1)
        return num_elements * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return 1024

    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return get_tpu_min_page_size(vllm_config)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            return (VllmKVCacheLayout.BLNHC, )
        return (VllmKVCacheLayout.LBNHC, )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # TPU kernels consume packed records, including head/lane padding.
        # Publish their byte geometry so vLLM computes native L/B strides
        # from the same page size as the TPU allocation.
        page_bytes = cls.get_kv_cache_page_size_bytes(spec.block_size,
                                                      spec.num_kv_heads,
                                                      spec.head_size,
                                                      spec.dtype)
        return dataclasses.replace(spec,
                                   num_head_slots=1,
                                   state_content_bytes=page_bytes //
                                   spec.block_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False, ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4)
        return (0, 1, 2, 3)


class VllmTPUDeepseekV32IndexerBackend(DeepseekV32IndexerBackend):
    """Indexer backend for TPU.

    The TPU materializer retains the native 3-D indexer K cache shape
    `(num_blocks, block_size, head_size)` consumed by streamindex. vLLM 0.29
    no longer owns this backend shape hook.

    The page size hooks below delegate to `PallasMLAttentionBackend`. They are
    not the indexer's own page size: `cache_config.block_size` is global across
    KV cache groups, and `update_block_size_for_backend` consults whichever
    backend `Platform._find_non_ssm_backend` returns first. An indexer cache
    registers before its `mla_attn` sibling, so on a sparse-MLA model these are
    the hooks that decide the block size for the MLA latent cache too --
    hardcoding a second copy of the constant would let the two drift, and the
    effective value would silently depend on layer registration order.
    """

    @staticmethod
    def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str | torch.dtype = "auto") -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC, )

    @staticmethod
    def get_name() -> str:
        return "TPU_STREAMINDEX_INDEXER"

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        return PallasMLAttentionBackend.get_page_size(vllm_config)

    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        return PallasMLAttentionBackend.get_min_page_size(vllm_config)


class PallasMLAttentionBackendImpl(MLAAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        v_head_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        parallel_config = getattr(get_current_vllm_config_or_none(),
                                  "parallel_config", None)
        dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        self.dcp_size = dcp_size if isinstance(dcp_size,
                                               int) and dcp_size > 1 else 1
        self.dcp_interleave_size = getattr(parallel_config,
                                           "cp_kv_cache_interleave_size", 1)
        self._dcp_mesh = None
        if self.dcp_size > 1:
            # The indexer refuses pcp>1 and dcp>1 together (see
            # `VllmTPUSparseAttnIndexer.__init__`)
            pcp_size = getattr(parallel_config,
                               "prefill_context_parallel_size", 1)
            if isinstance(pcp_size, int) and pcp_size > 1:
                raise NotImplementedError(
                    f"pcp_size={pcp_size} and dcp_size={self.dcp_size} are "
                    "both >1; sparse MLA has one KV-position interleave, "
                    "not two.")
            if self.dcp_interleave_size % kv_cache_utils.WORD_BYTES:
                raise ValueError(
                    "DCP sparse MLA needs --cp-kv-cache-interleave-size to be "
                    f"a multiple of {kv_cache_utils.WORD_BYTES} "
                    f"(kv_cache_utils.WORD_BYTES), got "
                    f"{self.dcp_interleave_size}. It is also the best value for "
                    "top-k load balance across the shards -- coarser cycles "
                    "let a contiguous run of winners land on fewer ranks.")
            self._dcp_mesh = get_or_create_dcp_mesh()

    def _get_kv_scales(
            self,
            layer: Any) -> tuple[float | None, float | None, float | None]:
        """Harvest scalar quantization scales from heterogeneous layer attributes.

        Extracting deterministic scalar float values guarantees purely static execution inside
        Pallas FX graphs without tracing dynamic tensor-shape overhead right when consuming
        heterogeneous checkpoint FP8 scales.
        """
        q_scale = getattr(layer, "_q_scale_float", None)
        if q_scale is None and hasattr(layer, "_q_scale"):
            q_scale = (layer._q_scale.item()
                       if isinstance(layer._q_scale, torch.Tensor)
                       and layer._q_scale.ndim == 0 else getattr(
                           layer._q_scale, "tolist", lambda: layer._q_scale)())

        k_scale = getattr(layer, "_k_scale_float", None)
        if k_scale is None and hasattr(layer, "_k_scale"):
            k_scale = (layer._k_scale.item()
                       if isinstance(layer._k_scale, torch.Tensor)
                       and layer._k_scale.ndim == 0 else getattr(
                           layer._k_scale, "tolist", lambda: layer._k_scale)())

        v_scale = getattr(layer, "_v_scale_float", None)
        if v_scale is None and hasattr(layer, "_v_scale"):
            v_scale = (layer._v_scale.item()
                       if isinstance(layer._v_scale, torch.Tensor)
                       and layer._v_scale.ndim == 0 else getattr(
                           layer._v_scale, "tolist", lambda: layer._v_scale)())
        if v_scale is None:
            v_scale = k_scale

        return q_scale, k_scale, v_scale

    def _build_mla_op(
        self,
        layer: Any,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ):
        vllm_context = get_vllm_model_wrapper_context()

        def mla_attention_core_tpu(
            kv_cache: jax.Array,
            q_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            metadata = AttentionMetadata(
                input_positions=None,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )
            return mla_attention(
                q_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache,
                metadata,
                vllm_context.mesh,
                layer.num_heads,
                layer.qk_nope_head_dim,
                sm_scale=layer.scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        op_name = f"pallas::mla_attention_{layer.layer_name.replace('.', '_')}"
        mla_jax_op = pallas.jax_op(op_name,
                                   mla_attention_core_tpu,
                                   donate_argnums=(0, ))

        def _fake_mla(kv_cache, q_nope, q_pe, kv_c_normed, k_pe, *args,
                      **kwargs):
            num_tokens = q_nope.size(0)
            out_shape = (num_tokens, layer.num_heads, layer.kv_lora_rank)
            return torch.empty_like(kv_cache), torch.empty(
                out_shape, dtype=q_nope.dtype, device=q_nope.device)

        mla_jax_op.register_fake(_fake_mla)

        def mla_impl(kv_cache: torch.Tensor, q_nope: torch.Tensor,
                     q_pe: torch.Tensor, kv_c_normed: torch.Tensor,
                     k_pe: torch.Tensor, seq_lens: torch.Tensor,
                     block_tables: torch.Tensor, query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor) -> torch.Tensor:
            new_kv, outputs = mla_jax_op(kv_cache, q_nope, q_pe, kv_c_normed,
                                         k_pe, seq_lens, block_tables,
                                         query_start_loc, request_distribution)

            kv_cache.copy_(new_kv)
            return outputs

        return mla_impl

    def _build_sparse_mla_op(self, layer: Any):
        from vllm_torchtpu.layers.core.attention_interface import \
            sparse_mla_attention

        vllm_context = get_vllm_model_wrapper_context()

        # Per-tensor fp8 dequant scale for the packed nope cache -- the
        # same checkpoint k_scale the dense path uses.
        k_scale = None
        if layer.kv_cache_quantized_dtype is not None:
            _, k_scale, _ = self._get_kv_scales(layer)

        def sparse_mla_attention_core_tpu(
            kv_cache_nope: jax.Array,
            kv_cache_rope: jax.Array,
            ql_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            topk_indices: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            nope_spec, rope_spec = layer.mla_kv_spec
            return sparse_mla_attention(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache_nope,
                kv_cache_rope,
                topk_indices,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                vllm_context.mesh,
                nope_spec,
                rope_spec,
                sm_scale=layer.scale,
                k_scale=k_scale,
            )

        op_name = ("pallas::sparse_mla_attention_"
                   f"{layer.layer_name.replace('.', '_')}")
        sparse_mla_jax_op = pallas.jax_op(op_name,
                                          sparse_mla_attention_core_tpu,
                                          donate_argnums=(0, 1))

        def _fake_sparse_mla(kv_cache_nope, kv_cache_rope, ql_nope, q_pe,
                             kv_c_normed, k_pe, topk_indices, *args, **kwargs):
            num_tokens = ql_nope.size(0)
            out_shape = (num_tokens, layer.num_heads, layer.kv_lora_rank)
            return (torch.empty_like(kv_cache_nope),
                    torch.empty_like(kv_cache_rope),
                    torch.empty(out_shape,
                                dtype=ql_nope.dtype,
                                device=ql_nope.device))

        sparse_mla_jax_op.register_fake(_fake_sparse_mla)

        def sparse_mla_impl(
                kv_cache: tuple[torch.Tensor, torch.Tensor],
                ql_nope: torch.Tensor, q_pe: torch.Tensor,
                kv_c_normed: torch.Tensor, k_pe: torch.Tensor,
                topk_indices: torch.Tensor, seq_lens: torch.Tensor,
                block_tables: torch.Tensor, query_start_loc: torch.Tensor,
                request_distribution: torch.Tensor) -> torch.Tensor:
            nope_cache, rope_cache = kv_cache
            new_nope, new_rope, outputs = sparse_mla_jax_op(
                nope_cache, rope_cache, ql_nope, q_pe, kv_c_normed, k_pe,
                topk_indices, seq_lens, block_tables, query_start_loc,
                request_distribution)
            nope_cache.copy_(new_nope)
            rope_cache.copy_(new_rope)
            return outputs

        return sparse_mla_impl

    def _build_sparse_mla_dcp_op(self, layer: Any, dcp_size: int,
                                 interleave_size: int):
        """DCP variant of `_build_sparse_mla_op`: shard the KV cache only.

        In DCP the queries are replicated. `kv_c_normed`/`k_pe` in particular
        need no all-gather -- they are computed from a replicated residual
        stream, so every rank already holds every row and only its write mask
        (owner rank) differs.

        The op returns unmerged partials. Combining them is one log-sum-exp
        reduction over the DCP group, is done by the torch
        caller (`cp_mla_attention.merge_lse_partials_scatter_heads`), which
        also narrows the head axis back down.
        """
        from vllm_torchtpu.layers.core.attention_interface import \
            sparse_mla_attention_dcp

        mesh = self._dcp_mesh

        k_scale = None
        if layer.kv_cache_quantized_dtype is not None:
            _, k_scale, _ = self._get_kv_scales(layer)

        def sparse_mla_attention_core_tpu_dcp(
            kv_cache_nope: jax.Array,
            kv_cache_rope: jax.Array,
            ql_nope: jax.Array,
            q_pe: jax.Array,
            kv_c_normed: jax.Array,
            k_pe: jax.Array,
            local_topk_indices: jax.Array,
            seq_lens: jax.Array,
            block_tables: jax.Array,
            query_start_loc: jax.Array,
            request_distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            nope_spec, rope_spec = layer.mla_kv_spec
            return sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c_normed,
                k_pe,
                kv_cache_nope,
                kv_cache_rope,
                local_topk_indices,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                mesh,
                nope_spec,
                rope_spec,
                sm_scale=layer.scale,
                k_scale=k_scale,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
            )

        op_name = ("pallas::sparse_mla_attention_dcp_"
                   f"{layer.layer_name.replace('.', '_')}")
        sparse_mla_jax_op = pcp_streaming_jax_op(
            op_name,
            sparse_mla_attention_core_tpu_dcp,
            donate_argnums=(0, 1),
            mesh=mesh,
            input_partition_specs=_SPARSE_MLA_DCP_INPUT_PARTITION_SPECS,
            output_partition_specs=_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS,
        )

        # We must overwrite the default fake implementation as vLLM uses
        # dynamic dimensions for the query.
        def _fake_sparse_mla_dcp(kv_cache_nope, kv_cache_rope, ql_nope, *rest,
                                 **kwargs):
            del rest, kwargs
            return (
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                torch.empty_like(ql_nope),
                # LSE: one f32 per (token, head).
                ql_nope.new_empty(ql_nope.shape[:2], dtype=torch.float32),
            )

        sparse_mla_jax_op.register_fake(_fake_sparse_mla_dcp)

        def sparse_mla_dcp_impl(
            kv_cache: tuple[torch.Tensor, torch.Tensor],
            ql_nope: torch.Tensor,
            q_pe: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            local_topk_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nope_cache, rope_cache = kv_cache
            new_nope, new_rope, outputs, lse = sparse_mla_jax_op(
                nope_cache, rope_cache, ql_nope, q_pe, kv_c_normed, k_pe,
                local_topk_indices, seq_lens, block_tables, query_start_loc,
                request_distribution)
            nope_cache.copy_(new_nope)
            rope_cache.copy_(new_rope)
            return outputs, lse

        return sparse_mla_dcp_impl

    def forward(
        self,
        layer: Any,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Executes complete TPU multi-head latent attention evaluation."""
        assert isinstance(
            q, tuple) and len(q) == 2, "q must be a tuple of (q_nope, q_pe)"
        q_nope, q_pe = q
        input_dtype = q_nope.dtype

        # For determine_available_memory when cache memory buffer is empty right during probe
        # (before binding, `layer.kv_cache` is the layer's default empty
        # tensor; after binding, sparse layers hold a (nope, rope) pair).
        if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
            if output is None:
                # Preserve symbolic token dimensions during the memory probe.
                template = q_nope.flatten(1)[:, :1].expand(
                    -1, layer.num_heads * layer.v_head_dim)
                return torch.ones_like(template)
            output.fill_(1)
            return output

        # Evaluate projection matrices directly across input precision (`bfloat16`/`float16`/`fp8`)
        # without dynamic `.to(torch.float32)` casting right before `torch.bmm`.
        q_nope_t = q_nope.transpose(0, 1)
        w_uk_t = layer.W_UK_T.to(
            q_nope_t.dtype
        ) if layer.W_UK_T.dtype != q_nope_t.dtype else layer.W_UK_T
        ql_nope = torch.bmm(q_nope_t, w_uk_t)
        if hasattr(layer, "W_UK_T_scale"):
            ql_nope = ql_nope * layer.W_UK_T_scale
        ql_nope = ql_nope.transpose(0, 1).to(input_dtype)

        q_scale, k_scale, v_scale = self._get_kv_scales(layer)

        if layer.kv_cache_quantized_dtype is not None:
            kv_c_normed, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                         kv_c_normed,
                                         value=None,
                                         k_scale=k_scale)
            k_pe, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                  k_pe,
                                  value=None,
                                  k_scale=k_scale)

        ql_nope_flat = ql_nope.view(-1, layer.num_heads, layer.kv_lora_rank)
        q_pe_flat = q_pe.view(-1, layer.num_heads, layer.qk_rope_head_dim)
        kv_c_normed_flat = kv_c_normed.view(-1, layer.kv_lora_rank)
        k_pe_flat = k_pe.view(-1, layer.qk_rope_head_dim)

        topk_indices = kwargs.get("topk_indices")
        if topk_indices is not None:
            assert (isinstance(kv_cache, (tuple, list))
                    and len(kv_cache) == 2), (
                        "sparse MLA layers use a native (nope, rope) split "
                        f"cache; got {type(kv_cache)}")
            if self.dcp_size > 1:
                assert getattr(layer, "sparse_mla_dcp_op", None) is not None, (
                    "DCP sparse MLA op was never built; "
                    "`process_weights_after_loading` must run first.")
                # DCP re-spends `dcp_size` of TP's ways on positions instead of
                # heads, so attention runs at `tp // dcp` head sharding: every
                # rank attends with the whole DCP group's heads over its own
                # KV shard, and the head axis is scattered back after the
                # merge.
                ql_nope_dcp = all_gather_heads(ql_nope_flat)
                q_pe_dcp = all_gather_heads(q_pe_flat)
                # `topk_indices` here is not global positions: the DCP indexer
                # already resolved each token's global top-k into this rank's
                # own local cache indices, `-1` where it owns nothing.
                partial, lse = layer.sparse_mla_dcp_op(
                    kv_cache,
                    ql_nope_dcp,
                    q_pe_dcp,
                    kv_c_normed_flat,
                    k_pe_flat,
                    topk_indices,
                    attn_metadata.seq_lens,
                    attn_metadata.block_tables,
                    attn_metadata.query_start_loc,
                    attn_metadata.request_distribution,
                )
                # The scatter undoes the head all-gather above.
                outputs = merge_lse_partials_scatter_heads(partial, lse)
            else:
                if (not hasattr(layer, "sparse_mla_op")
                        or layer.sparse_mla_op is None):
                    layer.sparse_mla_op = self._build_sparse_mla_op(layer)

                outputs = layer.sparse_mla_op(
                    kv_cache,
                    ql_nope_flat,
                    q_pe_flat,
                    kv_c_normed_flat,
                    k_pe_flat,
                    topk_indices,
                    attn_metadata.seq_lens,
                    attn_metadata.block_tables,
                    attn_metadata.query_start_loc,
                    attn_metadata.request_distribution,
                )
        else:
            if not hasattr(layer, "mla_op") or layer.mla_op is None:
                layer.mla_op = self._build_mla_op(layer,
                                                  q_scale=q_scale,
                                                  k_scale=k_scale,
                                                  v_scale=v_scale)

            outputs = layer.mla_op(
                kv_cache,
                ql_nope_flat,
                q_pe_flat,
                kv_c_normed_flat,
                k_pe_flat,
                attn_metadata.seq_lens,
                attn_metadata.block_tables,
                attn_metadata.query_start_loc,
                attn_metadata.request_distribution,
            )

        outputs_t = outputs.reshape(-1, layer.num_heads,
                                    layer.kv_lora_rank).transpose(0, 1)
        w_uv = layer.W_UV.to(
            outputs_t.dtype
        ) if layer.W_UV.dtype != outputs_t.dtype else layer.W_UV
        out_proj = torch.bmm(outputs_t, w_uv)
        if hasattr(layer, "W_UV_scale"):
            out_proj = out_proj * layer.W_UV_scale
        outputs = out_proj.transpose(0, 1).to(input_dtype).reshape(
            -1, layer.num_heads * layer.v_head_dim)

        if output is not None and outputs is not output:
            output.copy_(outputs)

        return outputs

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Structural no-op. TPU Pallas MLA unifies projection and evaluation inside `self.forward`."""
        pass

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Structural no-op. TPU Pallas MLA unifies projection and evaluation inside `self.forward`."""
        pass

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Structural no-op. KV cache updates occur in-place during ragged paged attention kernel run."""
        pass
