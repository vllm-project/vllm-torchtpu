# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import ClassVar

import jax
import torch
from torch_tpu._internal import pallas, sync
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType)

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    PCP_STREAMING_RPA_INPUT_PARTITION_SPECS, get_pcp_streaming_mesh,
    invoke_pcp_streaming_op, make_pcp_streaming_rpa_kernel,
    pcp_streaming_jax_op)
from vllm_torchtpu.layers.common.attention_interface import (
    attention, ragged_paged_attention, ragged_paged_attention_batched)
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.common.quantization import (is_floating_dtype,
                                                      quantize_kv)
from vllm_torchtpu.layers.common.sequence_layout import \
    is_pcp_streaming_attention_metadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128

# Note: TPU can fp8 as storage dtype but doesn't support converting from uint8
# from to fp32 directly. That's why it has a dtype mapping different from GPU
TPU_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
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
    )


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
        rpa_func=ragged_paged_attention_batched,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
    )


# =========================================================================================
# VLLM Attention Backend
# =========================================================================================


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
        if not is_floating_dtype(kv_dtype):
            raise NotImplementedError(
                f"Integer KV cache dtype is not supported yet: {kv_dtype}")
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

    @staticmethod
    def get_kv_cache_page_size_bytes(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> int:
        """Return one Pallas KV-cache page size including layout padding."""
        dtype = _resolve_kv_cache_dtype(cache_dtype_str)
        shape = PallasAttentionBackend.get_kv_cache_shape(
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
        max_num_page_per_req = (1024 * 1024 // 2 //
                                vllm_config.scheduler_config.max_num_seqs // 4)
        min_page_size = cdiv(vllm_config.model_config.max_model_len,
                             max_num_page_per_req)
        min_page_size = 1 << (min_page_size - 1).bit_length()
        return min_page_size

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


class PallasBatchedRPAAttentionBackend(PallasAttentionBackend):
    """Pallas attention backend wired to the batched-RPA kernel.

    Registered under `AttentionBackendEnum.CUSTOM` by
    `TpuPlatform.pre_register_and_update`; user opts in via
    `--attention-backend CUSTOM`. The Impl subclass sets `use_batched_rpa=True`,
    which plumbs through `_pallas_rpa_kernel` → `attention()` →
    `sharded_ragged_paged_attention()` to pick the batched kernel function.
    Declares `get_supported_kernel_block_sizes() -> [256]` so vLLM's standard
    `get_preferred_block_size` flow picks the right block size automatically.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["PallasBatchedRPAAttentionBackendImpl"]:
        return PallasBatchedRPAAttentionBackendImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
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
        self.rpa_kernel = None
        # Populated by initialize_kernel() before torch.compile traces the
        # model. Forward can then reuse the custom op without resolving a PCP
        # mesh from inside Dynamo's fullgraph capture.
        self._kernel_config_cache: dict = {}

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
        vllm_config = getattr(ctx, "vllm_config", None)
        max_model_len = (getattr(vllm_config.model_config, "max_model_len",
                                 None) if use_pcp_streaming
                         and vllm_config is not None else None)
        config_key = (self._kernel_op_prefix, self.sliding_window, q_scale,
                      k_scale, v_scale, use_pcp_streaming,
                      cp_kv_cache_interleave_size, max_model_len)
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
        registry_key = (self._kernel_op_prefix, self.sliding_window,
                        self.scale, self.logits_soft_cap, id(mesh), q_scale,
                        k_scale, v_scale, skip_kv_update, use_pcp_streaming,
                        cp_kv_cache_interleave_size, max_model_len)
        existing = self._kernel_registry.get(registry_key)
        if existing is not None:
            self._kernel_config_cache[config_key] = existing
            return existing

        kernel_instance_id = self._allocate_kernel_instance_id()
        op_name = f"{self._kernel_op_prefix}_{kernel_instance_id}"

        if use_pcp_streaming:
            wrapped_fn = make_pcp_streaming_rpa_kernel(
                mesh=mesh,
                sliding_window=self.sliding_window,
                sm_scale=self.scale,
                soft_cap=self.logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                skip_kv_update=skip_kv_update,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                max_model_len=max_model_len,
            )
        else:
            # Prepare wrapper function with static arguments.
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
            )

        # Register as a custom op to mark it as an op boundary in Dynamo.
        # This prevents torch.compile from tracing into the Pallas kernel internals.
        if use_pcp_streaming:
            rpa_kernel_op = pcp_streaming_jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=(0, ),
                mesh=op_mesh,
                input_partition_specs=input_partition_specs)
        else:
            rpa_kernel_op = pallas.jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=(0, ),
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
            kv_cache.copy_(new_kv_cache)
            return output

        self._kernel_registry[registry_key] = rpa_kernel_impl
        self._kernel_config_cache[config_key] = rpa_kernel_impl
        return rpa_kernel_impl

    def _validate_pcp_streaming_support(self, skip_kv_update: bool) -> None:
        unsupported_features = []
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
        vllm_config = getattr(ctx, "vllm_config", None)
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)
        pcp_configured = (parallel_config is not None and getattr(
            parallel_config, "prefill_context_parallel_size", 1) > 1)
        if pcp_configured:
            self._validate_pcp_streaming_support(skip_kv_update)
            self.rpa_kernel = self._build_rpa_kernel(
                q_scale,
                k_scale,
                v_scale,
                skip_kv_update=skip_kv_update,
                use_pcp_streaming=True,
                cp_kv_cache_interleave_size=getattr(
                    parallel_config, "cp_kv_cache_interleave_size", 0),
            )
            return

        self.rpa_kernel = self._build_rpa_kernel(q_scale,
                                                 k_scale,
                                                 v_scale,
                                                 skip_kv_update=skip_kv_update)

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
        vllm_config = getattr(ctx, "vllm_config", None)
        parallel_config = (None if vllm_config is None else
                           vllm_config.parallel_config)
        use_pcp_streaming = is_pcp_streaming_attention_metadata(attn_metadata)
        pcp_configured = (parallel_config is not None and getattr(
            parallel_config, "prefill_context_parallel_size", 1) > 1)
        if pcp_configured and not use_pcp_streaming:
            raise RuntimeError(
                "PCP is configured, but attention metadata does not use "
                "the PCP streaming sequence layout.")
        skip_kv_update = self.kv_sharing_target_layer_name is not None
        if use_pcp_streaming:
            self._validate_pcp_streaming_support(skip_kv_update)
        cp_kv_cache_interleave_size = (getattr(
            parallel_config, "cp_kv_cache_interleave_size", 0)
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

        # TODO (geyuhao) ideally we don't want this
        if not torch.compiler.is_compiling():
            sync.synchronize(kv_cache)

        if query_dim == 2:
            outputs = outputs.reshape(q_len, self.num_heads * self.head_size)

        if output is not None:
            output.copy_(outputs)
        return outputs


class PallasBatchedRPAAttentionBackendImpl(PallasAttentionBackendImpl):
    """Impl variant that dispatches to the batched RPA Pallas kernel."""
    _kernel_entry = staticmethod(_pallas_rpa_kernel_batched)
    _kernel_op_prefix = "pallas::rpa_kernel_batched"
