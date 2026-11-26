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
# Provenance: model-owned copy of
# src/vllm_torchtpu/layers/adapter/custom_ops/gdn_attention_op.py
#   (reference at 33ee547ba947, Qwen3.8 changes through
#   77eff5e0908cb1a06ad640223adbc3d87ffeeaa3).
#   Owned by models/qwen4_exp from this copy on.

import dataclasses
import functools
import itertools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from einops import rearrange
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_torchtpu.block_major_pool import (
    BlockMajorPoolBindable,
    BlockMajorPoolLayout,
    flat_manager_ids,
    kernel_view,
    unified_block_major_enabled,
)
from vllm_torchtpu.distributed.pcp import (
    get_or_create_pcp_mesh,
    get_pcp_rank,
    get_pcp_world_size,
)
from vllm_torchtpu.gdn_pool_layout import (
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.kernels.gdn.head_geometry import (
    GdnHeadGeometry,
    derive_gdn_head_geometry,
)
from vllm_torchtpu.layers.adapter.gdn_linear import (
    GdnColumnParallelLinear,
    GdnInterleavedColumnParallelLinear,
)
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.core.sequence_layout import (
    is_pcp_streaming_attention_metadata,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import fresh_buffer

from ..config import mtp_gdn_two_slot_enabled
from ..gdn_attention import (
    gdn_step_metadata,
    run_jax_gdn_attention,
    run_jax_gdn_attention_pcp_tp_prefill,
    run_jax_gdn_attention_pooled,
    run_jax_gdn_attention_pooled_pcp_prefill_projection,
    run_jax_gdn_multi_layer_replay_pooled,
)
from ..kernels.gdn_v3.input_layout import (
    Conv1DInput,
)
from ..metadata_access import (
    mamba_ckpt_of,
    mamba_state_of,
)
from ..runner.two_slot import INVALID_TOKEN_ID
from .rows_update_op import update_rows_ as _update_verify_rows_

# Set by the model plan through `configure` before the layers are built (the
# defaults are the small-batch / non-c48 choices).
_OPTIONS = {
    # One gdn_step_metadata op per step and cache group (plain unified pool).
    "GDN_HOIST_METADATA": False,
    # The pooled kernel reads a load-time float32 [kernel, 1, dim] conv weight.
    "GDN_CONV_WEIGHT_KMAJOR": False,
}


def configure(**options) -> None:
    """Set `_OPTIONS` entries; unknown keys raise."""
    unknown = set(options) - set(_OPTIONS)
    if unknown:
        raise KeyError(f"unknown gdn_attention_op options: {sorted(unknown)}")
    _OPTIONS.update(options)


_GDN_PCP_AXIS_NAME = "pcp"
_GDN_TP_AXIS_NAME = "tp"
_GDN_PCP_TP_STATE_AXIS = (_GDN_TP_AXIS_NAME, _GDN_PCP_AXIS_NAME)

if TYPE_CHECKING:

    class _GdnAttentionMetadata(Protocol):
        """The per-layer AttentionMetadata fields that GDN reads.

        The runner fills them with torch tensors, while AttentionMetadata
        annotates them as jax.Array.
        """

        block_tables: torch.Tensor
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        request_distribution: torch.Tensor
        mamba_state_indices: torch.Tensor | None


def _to_jax_ssm_state_dtype(dtype: torch.dtype) -> jnp.dtype:
    if dtype == torch.float32:
        return jnp.dtype(jnp.float32)
    if dtype == torch.bfloat16:
        return jnp.dtype(jnp.bfloat16)
    raise ValueError(
        f"Unsupported mamba_ssm_cache_dtype for TPU GDN: "
        f"{dtype}; expected float32 or bfloat16"
    )


def gdn_attention_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    slot_read_offsets: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Speculative decoding: vLLM's MambaSpec widens the conv state by
    # `num_spec` columns per slot. The extra columns are unused on TPU
    # (rollback keeps one full checkpoint per group slot instead of a
    # rolling window); only the first kernel_size - 1 columns hold data, so
    # slice them out for the kernel and write the result back into the same
    # (donated) buffer afterwards.
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, : kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    (new_conv_state, new_recurrent_state), output = run_jax_gdn_attention(
        mixed_qkv,
        b,
        a,
        conv_state_in,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        slot_read_offsets,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        mesh=mesh,
        num_spec_tokens=num_spec_tokens,
    )
    if state_len > kernel_size - 1:
        # Write the kernel result into the first kernel_size - 1 columns of the
        # donated conv_state in place and leave the unused spec tail untouched.
        # A concat here would materialize a fresh full-width array that no
        # longer derives from the donated buffer, defeating the conv-cache
        # input_output_alias (a full-width alloc + copy per GDN layer per step);
        # the dynamic-update-slice keeps the write inside the donated buffer.
        new_conv_state = conv_state.at[:, : kernel_size - 1, :].set(new_conv_state)

    return new_conv_state, new_recurrent_state, output


def gdn_attention_pooled_core_tpu(
    mixed_qkv: jax.Array | Conv1DInput,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    slot_read_offsets: jax.Array | None = None,
    ckpt_indices: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    recurrent_state_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype = jnp.bfloat16,
    num_spec_tokens: int = 0,
    conv_weight_kmajor: bool = False,
    single_checkpoint: bool = False,
    batched_only: bool = False,
    commit_lens: jax.Array | None = None,
    read_offset_indices: jax.Array | None = None,
    step_metadata: tuple[jax.Array, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    return run_jax_gdn_attention_pooled(
        mixed_qkv,
        b,
        a,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        slot_read_offsets,
        ckpt_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        recurrent_state_dtype=recurrent_state_dtype,
        conv_state_dtype=conv_state_dtype,
        mesh=mesh,
        num_spec_tokens=num_spec_tokens,
        conv_weight_kmajor=conv_weight_kmajor,
        single_checkpoint=single_checkpoint,
        batched_only=batched_only,
        commit_lens=commit_lens,
        read_offset_indices=read_offset_indices,
        step_metadata=step_metadata,
    )


def gdn_attention_core_tpu_pcp_prefill(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, : kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    (new_conv_state, new_recurrent_state), output = (
        run_jax_gdn_attention_pcp_tp_prefill(
            mixed_qkv,
            b,
            a,
            conv_state_in,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            mesh=mesh,
        )
    )
    if state_len > kernel_size - 1:
        new_conv_state = conv_state.at[:, : kernel_size - 1, :].set(new_conv_state)
    return new_conv_state, new_recurrent_state, output


def _get_pcp_size(vllm_config: VllmConfig) -> int:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    return pcp_size if pcp_size > 1 else 1


def _localize_gdn_mamba_spec_for_pcp(
    spec: MambaSpec,
    pcp_size: int,
    *,
    num_kq_heads: int,
    num_v_heads: int,
    d_k: int,
    d_v: int,
) -> MambaSpec:
    if pcp_size <= 1 or len(spec.shapes) != 2:
        return spec
    if is_conv_state_dim_first():
        raise NotImplementedError(
            "TPU GDN PCP state sharding requires VLLM_SSM_CONV_STATE_LAYOUT=SD."
        )

    conv_shape = tuple(spec.shapes[0])
    recurrent_shape = tuple(spec.shapes[1])
    if not conv_shape or not recurrent_shape:
        return spec

    geometry = derive_gdn_head_geometry(num_kq_heads, num_v_heads, pcp_size)
    expected_conv_dim = 2 * num_kq_heads * d_k + num_v_heads * d_v
    if conv_shape[-1] != expected_conv_dim:
        raise ValueError(
            "GDN MambaSpec conv width does not match its TP-local heads: "
            f"shape={conv_shape}, expected_width={expected_conv_dim}."
        )
    if recurrent_shape[0] != num_v_heads:
        raise ValueError(
            "GDN MambaSpec recurrent heads do not match its "
            f"TP-local V heads: shape={recurrent_shape}, "
            f"expected_heads={num_v_heads}."
        )

    unpadded_page_size = dataclasses.replace(
        spec, page_size_padded=None
    ).page_size_bytes
    page_size_padded = spec.page_size_padded
    if page_size_padded == unpadded_page_size:
        page_size_padded = None

    return dataclasses.replace(
        spec,
        shapes=geometry.local_state_shapes(spec.shapes, d_k, d_v),
        page_size_padded=page_size_padded,
    )


# GDN custom ops shared by every layer with the same kernel config. A
# JaxCallable caches its jax.export per instance, so one op per layer re-exported
# (and re-lowered) the identical kernel for every layer at every compile bucket.
_shared_gdn_ops: dict[tuple, Any] = {}
_shared_gdn_op_ids = itertools.count()


def _get_shared_gdn_op(kind: str, key: tuple, build: Callable[[str], Any]) -> Any:
    """Return the ``qwen4_exp::<kind>_<n>`` op for ``(kind, key)``, building it once.

    ``kind`` and ``key`` together must cover every value that the op's kernel
    and fake capture. ``build`` gets the ``<kind>_<n>`` part of the name and
    registers the op under ``qwen4_exp::``.
    """
    op = _shared_gdn_ops.get((kind, key))
    if op is None:
        op = build(f"{kind}_{next(_shared_gdn_op_ids)}")
        _shared_gdn_ops[(kind, key)] = op
    return op


def _gdn_step_metadata_jax(
    slot_read_offsets: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    distribution: jax.Array,
    ckpt_indices: jax.Array,
) -> tuple[jax.Array, ...]:
    return gdn_step_metadata(
        slot_read_offsets,
        state_indices,
        query_start_loc,
        seq_lens,
        distribution,
        ckpt_indices,
    )


def _gdn_step_metadata_op() -> Any:
    return _get_shared_gdn_op(
        "gdn_step_metadata",
        (),
        lambda name: pallas.jax_op(f"qwen4_exp::{name}", _gdn_step_metadata_jax),
    )


class GdnStepMetadataCache:
    """Per-forward memo of the pooled GDN step metadata (VLLM_TPU_GDN_HOIST_METADATA).

    The model creates one per forward and hands it to every GDN layer. Layers
    of one KV-cache group share one AttentionMetadata object, so the first of
    them computes `gdn_step_metadata` and the rest reuse it. Keyed on object
    identity (AttentionMetadata is an unhashable dataclass); a forward sees a
    handful of groups, so the scan is short.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[Any, tuple[torch.Tensor, ...]]] = []

    def get(
        self, attn_metadata: Any, build: Callable[[], tuple[torch.Tensor, ...]]
    ) -> tuple[torch.Tensor, ...]:
        for key, value in self._entries:
            if key is attn_metadata:
                return value
        value = build()
        self._entries.append((attn_metadata, value))
        return value


# Not registered with register_oot: the shared op serves every other model's
# QwenGatedDeltaNetAttention; this copy is built only by Qwen3.8-Flash, which
# constructs its subclass (gdn.Qwen4ExpGatedDeltaNetAttention) directly.
class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention, BlockMajorPoolBindable):
    # Whether pooled ops consume the merged block-major unified pool, along
    # with `(pool_index, layout)` once bound by `set_block_major_pool()`.
    # Class-level defaults support test/mock instances constructed without `__init__`.
    _bm_kernel: bool = False
    _pooled_conv_state_dtype: torch.dtype = torch.bfloat16
    _bm_region: tuple[int, BlockMajorPoolLayout] | None = None
    _two_slot_mtp: bool = False
    _verify_win_cap: int = 0
    _verify_qkv_dim: int = 0
    _verify_local_n_v: int = 0
    _verify_act_dtype: torch.dtype = torch.bfloat16
    _shared_verify_flat_info: tuple[Any, ...] | None = None
    _shared_verify_flat_tensors: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None
    _shared_verify_sp_flat_tensors: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None
    _shared_verify_backbone_ref: Any = None
    # Type-only annotations for non-persistent buffers registered in __init__
    # (no class defaults so nn.Module.register_buffer is never shadowed).
    _verify_qkv_buf: torch.Tensor | None
    _verify_b_buf: torch.Tensor | None
    _verify_a_buf: torch.Tensor | None

    # Narrowed to the layer that _create_conv1d installs.
    # pyrefly: ignore[bad-override-mutable-attribute]
    conv1d: GdnColumnParallelLinear

    def __init__(
        self,
        *args,
        pooled_conv_state_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        _to_jax_ssm_state_dtype(pooled_conv_state_dtype)
        self._pooled_conv_state_dtype = pooled_conv_state_dtype
        super().__init__(*args, **kwargs)
        self._create_conv1d()
        # `_require_pcp_projection_parameters` hands this weight straight to
        # `fused_qkvz_projection_pcp_gdn`, which validates it as [n_out, n_in].
        if hasattr(self, "in_proj_qkvz"):
            setattr(self.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, True)
        # Bound by the runner during KV-cache initialization; None until then
        # (warmup/profiling runs check this).
        # vLLM's MambaBase types kv_cache as tuple[Tensor, ...].
        # pyrefly: ignore[bad-assignment]
        self.kv_cache = None
        # The wrapper context types vllm_config as object.
        vllm_config = cast(
            VllmConfig | None, get_vllm_model_wrapper_context().vllm_config
        )
        self._bm_kernel = vllm_config is not None and unified_block_major_enabled(
            vllm_config
        )
        self.gdn_op = self._build_gdn_op()
        self.gdn_pooled_op = self._build_pooled_gdn_op(block_major=self._bm_kernel)
        self._two_slot_mtp = vllm_config is not None and mtp_gdn_two_slot_enabled(
            vllm_config
        )
        self.gdn_pooled_single_ckpt_op = (
            self._build_pooled_gdn_op(
                block_major=self._bm_kernel,
                single_checkpoint=True,
            )
            if self._two_slot_mtp
            else None
        )
        self.gdn_pooled_replay_op = (
            self._build_pooled_gdn_op(
                block_major=self._bm_kernel,
                single_checkpoint=True,
                batched_only=True,
            )
            if self._two_slot_mtp
            else None
        )
        self._init_verify_input_buffers(vllm_config)
        # Block-major state ids are per layer (pool_index), so only the
        # plain unified pool can share one step's metadata across layers.
        self.gdn_pooled_hoisted_op = (
            self._build_pooled_gdn_op(hoisted=True)
            if _OPTIONS["GDN_HOIST_METADATA"] and not self._bm_kernel
            else None
        )
        # Built here, not on first use: registering a custom op while dynamo
        # traces the forward is an error.
        self.gdn_step_metadata_op = (
            _gdn_step_metadata_op() if self.gdn_pooled_hoisted_op is not None else None
        )
        pcp_enabled = self._pcp_streaming_enabled()
        self._pcp_streaming_configured = pcp_enabled
        self.gdn_pcp_op = (
            self._build_gdn_op(pcp_streaming=True) if pcp_enabled else None
        )
        self.gdn_pooled_pcp_op = (
            self._build_pooled_pcp_gdn_op(block_major=self._bm_kernel)
            if pcp_enabled
            else None
        )

    def set_block_major_pool(
        self, pool_index: int, layout: BlockMajorPoolLayout
    ) -> None:
        """Binds this layer to its region index within the merged block-major pool."""
        assert self._bm_kernel, "pooled ops were built for per-region pools"
        self._bm_region = (pool_index, layout)

    def _bm_state_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if not self._bm_kernel:
            return ids
        assert self._bm_region is not None
        pool_index, layout = self._bm_region
        return flat_manager_ids(ids, pool_index, layout.num_pools)

    def _bm_ckpt_ids(self, ckpt_ids: torch.Tensor) -> torch.Tensor:
        """Map manager checkpoint block IDs to physical block-major pool IDs.

        Null sentinel entries (``0``) remain ``0`` so inactive checkpoint
        columns never address layer-0+pool_index physical blocks.
        """
        if not self._bm_kernel:
            return ckpt_ids
        assert self._bm_region is not None
        pool_index, layout = self._bm_region
        if layout.num_pools <= 1:
            return ckpt_ids
        mapped = flat_manager_ids(ckpt_ids, pool_index, layout.num_pools)
        return torch.where(ckpt_ids > 0, mapped, torch.zeros_like(mapped))

    def _init_verify_input_buffers(self, vllm_config: VllmConfig | None) -> None:
        """Allocate non-persistent verify staging buffers for 2-slot MTP replay."""
        if not self._two_slot_mtp or vllm_config is None:
            self._verify_win_cap = 0
            self.register_buffer("_verify_qkv_buf", None, persistent=False)
            self.register_buffer("_verify_b_buf", None, persistent=False)
            self.register_buffer("_verify_a_buf", None, persistent=False)
            return
        sched = vllm_config.scheduler_config
        max_reqs = int(sched.max_num_seqs)
        max_batched = int(
            sched.max_num_batched_tokens or max_reqs * (self.num_spec + 1)
        )
        win_cap = max(1, min(max_batched, max_reqs * (self.num_spec + 1)))
        qkv_dim = self.tp_head_geometry.local_conv_dim(self.head_k_dim, self.head_v_dim)
        local_n_v = self.num_v_heads // self.tp_size
        act_dtype = self.model_config.dtype
        assert isinstance(act_dtype, torch.dtype)
        dev = self.A_log.device
        self._verify_win_cap = win_cap
        self._verify_qkv_dim = qkv_dim
        self._verify_local_n_v = local_n_v
        self._verify_act_dtype = act_dtype
        self.register_buffer(
            "_verify_qkv_buf",
            torch.zeros((win_cap, qkv_dim), dtype=act_dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_verify_b_buf",
            torch.zeros((win_cap, local_n_v), dtype=act_dtype, device=dev),
            persistent=False,
        )
        self.register_buffer(
            "_verify_a_buf",
            torch.zeros((win_cap, local_n_v), dtype=act_dtype, device=dev),
            persistent=False,
        )

    def bind_shared_verify_flat(
        self,
        qkv_flat: torch.Tensor,
        b_flat: torch.Tensor,
        a_flat: torch.Tensor,
        ord_idx: int,
        *,
        win_stride: int | None = None,
        sp_flat_tensors: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        backbone_ref: Any = None,
    ) -> None:
        """Bind this layer onto shared 2D verify staging buffers.

        Non-SP decode buckets (`16..1024`) use major-axis `[L * win_stride, dim]`
        (`dim=0`) to avoid per-layer cross-lane `dim=-1` shifts. Sequence-parallel
        buckets (`2048..8192`) optionally preserve `[win_cap, L * dim]` (`dim=-1`)
        so SP HLO buffer lifetimes remain unchanged alongside `_qsa_sel_buf`.
        """
        object.__setattr__(
            self, "_shared_verify_flat_tensors", (qkv_flat, b_flat, a_flat)
        )
        object.__setattr__(self, "_shared_verify_sp_flat_tensors", sp_flat_tensors)
        object.__setattr__(self, "_shared_verify_backbone_ref", backbone_ref)
        self._shared_verify_flat_info = (
            ord_idx,
            self._verify_qkv_dim,
            self._verify_local_n_v,
            win_stride,
            self._verify_win_cap,
        )
        self._verify_qkv_buf = None
        self._verify_b_buf = None
        self._verify_a_buf = None

    def get_active_verify_flat_tensors(
        self, *, force_sp: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return the active shared verify flat buffer tuple (SP or non-SP)."""
        return _active_verify_flat_tensors(self, force_sp=force_sp)

    def get_verify_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return staged verify ``(qkv, b, a)`` buffers for rejection replay."""
        flat = _active_verify_flat_tensors(self)
        info = self._shared_verify_flat_info
        if flat is not None and info is not None:
            qkv_f, b_f, a_f = flat
            ord_idx, qd, vd, w_stride, w_cap = info
            if qkv_f.shape[-1] == qd and w_stride is not None:
                r0 = ord_idx * w_stride
                r1 = r0 + min(w_stride, w_cap)
                return (
                    qkv_f[r0:r1].contiguous(),
                    b_f[r0:r1].contiguous(),
                    a_f[r0:r1].contiguous(),
                )
            return (
                qkv_f[:, ord_idx * qd : (ord_idx + 1) * qd].contiguous(),
                b_f[:, ord_idx * vd : (ord_idx + 1) * vd].contiguous(),
                a_f[:, ord_idx * vd : (ord_idx + 1) * vd].contiguous(),
            )
        qkv_buf = self._verify_qkv_buf
        b_buf = self._verify_b_buf
        a_buf = self._verify_a_buf
        if qkv_buf is None or b_buf is None or a_buf is None:
            return None
        return (qkv_buf, b_buf, a_buf)

    def snapshot_verify_inputs(
        self, max_rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Clone fixed-shape ``(win_cap, ...)`` staged verify buffers across chunks.

        Keeping the full static ``win_cap`` row count matches single-chunk
        replay so multi-chunk steps never trigger XLA/Pallas recompilations;
        rows past ``max_rows`` lie outside ``query_start_loc`` for the chunk's
        windowed sequences and carry ``commit_lens == 0``.
        """
        staged = self.get_verify_inputs()
        if staged is None:
            return None
        qkv_b, b_b, a_b = staged
        rows = max(0, min(int(max_rows), int(qkv_b.shape[0])))
        if rows == 0:
            return None
        return (
            qkv_b.clone(),
            b_b.clone(),
            a_b.clone(),
        )

    @torch.no_grad()
    def replay_two_slot_state(
        self,
        *,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_read_offsets: torch.Tensor,
        ckpt_indices: torch.Tensor,
        commit_lens: torch.Tensor,
        staged_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Re-advance GDN state from `c_read` to `commit_lens` into `1 - c_read`."""
        kv_cache = self.kv_cache
        if kv_cache is None or len(kv_cache) != 1 or kv_cache[0].numel() == 0:
            return
        if not self._two_slot_mtp:
            raise RuntimeError(
                f"replay_two_slot_state called on non-two-slot GDN layer {self.prefix}"
            )
        replay_op = self.gdn_pooled_replay_op
        if replay_op is None:
            raise RuntimeError(
                f"Two-slot GDN layer {self.prefix} has no gdn_pooled_replay_op"
            )
        inputs = (
            staged_inputs if staged_inputs is not None else self.get_verify_inputs()
        )
        if inputs is None:
            raise RuntimeError(
                f"Two-slot GDN layer {self.prefix} has no staged verify inputs "
                "for replay"
            )
        qkv_in, b_in, a_in = inputs
        win_rows = int(qkv_in.shape[0])
        qsl_clamped = torch.clamp(query_start_loc.to(torch.int32), 0, win_rows)
        raw_state = state_indices.to(torch.int32)
        phys_state = self._bm_state_ids(raw_state)
        phys_ckpt = self._bm_ckpt_ids(ckpt_indices.to(torch.int32))
        (recurrent_state,) = kv_cache
        replay_op(
            qkv_in,
            b_in,
            a_in,
            recurrent_state,
            self._pooled_conv_weight(),
            self.conv1d.bias,
            self.A_log,
            self.dt_bias,
            phys_state,
            qsl_clamped,
            distribution.to(torch.int32),
            seq_lens.to(torch.int32),
            slot_read_offsets,
            phys_ckpt,
            commit_lens.to(torch.int32),
            raw_state if self._bm_kernel else None,
        )

    @property
    def tp_head_geometry(self) -> GdnHeadGeometry:
        return derive_gdn_head_geometry(
            self.num_k_heads, self.num_v_heads, self.tp_size
        )

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        if self.gqa_interleaved_layout:
            group_v = self.num_v_heads // self.num_k_heads * self.head_v_dim
            return GdnInterleavedColumnParallelLinear(
                input_size=hidden_size,
                geometry=self.tp_head_geometry,
                group_sizes=[self.head_k_dim, self.head_k_dim, group_v, group_v],
                replicated_segments=[True, True, False, False],
                quant_config=quant_config,
                prefix=prefix,
            )
        return GdnColumnParallelLinear(
            input_size=hidden_size,
            geometry=self.tp_head_geometry,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
            include_z=True,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        if self.gqa_interleaved_layout:
            group_v = num_v_heads // self.num_k_heads
            return GdnInterleavedColumnParallelLinear(
                input_size=hidden_size,
                geometry=self.tp_head_geometry,
                group_sizes=[group_v, group_v],
                replicated_segments=[False, False],
                quant_config=quant_config,
                prefix=prefix,
            )
        return super().create_ba_proj(hidden_size, num_v_heads, quant_config, prefix)

    def _create_conv1d(self) -> None:
        # Upstream constructs conv1d directly and installs its Mamba loader
        # after create_qkvz_proj. Replace it before checkpoint loading, using
        # the same Q/K/V partitioning as the input projection.
        self.conv1d = GdnColumnParallelLinear(
            input_size=self.conv_kernel_size,
            geometry=self.tp_head_geometry,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
            include_z=False,
            params_dtype=self.conv1d.weight.dtype,
            prefix=f"{self.prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        assert isinstance(spec, MambaSpec)
        return _localize_gdn_mamba_spec_for_pcp(
            spec,
            _get_pcp_size(vllm_config),
            num_kq_heads=self.tp_head_geometry.local_num_kq_heads,
            num_v_heads=self.tp_head_geometry.local_num_v_heads,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        conv_state_dtype, temporal_state_dtype = super().get_state_dtype()
        if unified_kv_layout_enabled_for_architecture(self.model_config.architecture):
            # Declare exactly what the pool physically stores. vLLM sizes the
            # mamba page from the dtype declared HERE and asserts the padded
            # page covers it, while the block-slot derivation budgets the page
            # from the same explicit conv state dtype. Declaring anything
            # wider is not conservative, it is a contradiction: the two
            # disagreed once and every hybrid GDN engine failed to start on
            # that assert.
            conv_state_dtype = self._pooled_conv_state_dtype
        else:
            # TODO: Support bf16 conv state.
            # Per-layer cache: unlike the pool, the declared dtype IS the
            # storage dtype (vLLM strides the raw tensor with it), and this
            # kernel path still reads fp32 conv state. Nothing pads this
            # layout, so the wider declaration costs only its own bytes.
            conv_state_dtype = torch.float32
        return conv_state_dtype, temporal_state_dtype

    def _pooled_conv_weight(self) -> torch.Tensor:
        """The conv weight in the layout the pooled GDN op was built for."""
        if _OPTIONS["GDN_CONV_WEIGHT_KMAJOR"]:
            w = self.conv_weight_kmajor
            assert isinstance(w, torch.Tensor)
            return w
        return self.conv1d.weight

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Back the conv weight with a buffer of its own 3-D shape.

        The upstream layer rebinds `conv1d.weight` to an `unsqueeze(1)` view
        of the 2-D ColumnParallelLinear buffer and the sharded loader fills it
        through row slices, so the device buffer behind the Parameter stays
        2-D. torch_tpu hands a compiled executable the buffer behind each
        argument and re-materializes a shape-mismatched argument as a
        standalone `tt_jit_as_strided` program before every forward. One
        `fresh_buffer` copy at load time removes that per-step program.
        """
        del act_dtype
        w = self.conv1d.weight
        self.conv1d.weight.data = fresh_buffer(w)
        if _OPTIONS["GDN_CONV_WEIGHT_KMAJOR"]:
            # The pooled kernel's float32 [kernel_size, 1, dim] weight, so the
            # swap and cast run once here instead of in every forward. The
            # other GDN paths keep reading `conv1d.weight`.
            self.register_buffer(
                "conv_weight_kmajor",
                fresh_buffer(w.detach().permute(2, 1, 0).to(torch.float32)),
                persistent=False,
            )

    # pyrefly: ignore[bad-override]
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        geometry = self.tp_head_geometry
        conv_state_shape, temporal_state_shape = geometry.local_state_shapes(
            super().get_state_shape(),
            self.head_k_dim,
            self.head_v_dim,
            conv_dim_axis=0 if is_conv_state_dim_first() else -1,
        )
        conv_state_shape = conv_state_shape[:-1] + (1, conv_state_shape[-1])
        return conv_state_shape, temporal_state_shape

    @staticmethod
    def _pcp_streaming_enabled() -> bool:
        vllm_context = get_vllm_model_wrapper_context()
        if vllm_context.vllm_config is None:
            return False
        parallel_config = cast(VllmConfig, vllm_context.vllm_config).parallel_config
        return parallel_config.prefill_context_parallel_size > 1

    def _build_gdn_op(self, *, pcp_streaming: bool = False):
        local_num_v_heads = self.num_v_heads // self.tp_size
        local_num_kq_heads = self.tp_head_geometry.local_num_kq_heads
        has_conv_bias = self.conv1d.bias is not None
        vllm_context = get_vllm_model_wrapper_context()
        # The non-PCP op uses num_spec_tokens for verify/rollback. PCP is a
        # prefill-only path admitted at the platform boundary; its widened
        # conv-state tail is preserved by gdn_attention_core_tpu_pcp_prefill.
        num_spec_tokens = self.num_spec

        def _fake_gdn(mixed_qkv, _b, _a, conv_state, recurrent_state, *args, **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            return (
                torch.empty_like(conv_state),
                torch.empty_like(recurrent_state),
                torch.empty(out_shape, dtype=mixed_qkv.dtype, device=mixed_qkv.device),
            )

        if pcp_streaming:
            parallel_config = cast(VllmConfig, vllm_context.vllm_config).parallel_config
            interleave_size = parallel_config.cp_kv_cache_interleave_size
            if not isinstance(interleave_size, int) or interleave_size <= 0:
                raise ValueError(
                    "GDN PCP streaming requires cp_kv_cache_interleave_size > 0."
                )
            pcp_mesh = get_or_create_pcp_mesh(
                axis_name=_GDN_PCP_AXIS_NAME,
                tp_axis_name=_GDN_TP_AXIS_NAME,
            )
            pcp_size = get_pcp_world_size()
            if pcp_size != pcp_mesh.shape[_GDN_PCP_AXIS_NAME]:
                raise ValueError(
                    f"PCP world size {pcp_size} does not match mesh shape "
                    f"{pcp_mesh.shape[_GDN_PCP_AXIS_NAME]}."
                )
            if self.tp_size != pcp_mesh.shape[_GDN_TP_AXIS_NAME]:
                raise ValueError(
                    f"TP size {self.tp_size} does not match mesh shape "
                    f"{pcp_mesh.shape[_GDN_TP_AXIS_NAME]}."
                )
            derive_gdn_head_geometry(local_num_kq_heads, local_num_v_heads, pcp_size)
            op_name = f"qwen4_exp::gdn_attention_pcp_{self.prefix.replace('.', '_')}"
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu_pcp_prefill,
                mesh=pcp_mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
            )
            input_partition_specs = (
                PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME),  # mixed_qkv
                PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME),  # b
                PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME),  # a
                PartitionSpec(None, None, _GDN_PCP_TP_STATE_AXIS),  # conv_state
                PartitionSpec(
                    None, _GDN_PCP_TP_STATE_AXIS, None, None
                ),  # recurrent_state
                PartitionSpec(_GDN_TP_AXIS_NAME),  # conv_weight
                (
                    PartitionSpec(_GDN_TP_AXIS_NAME) if has_conv_bias else None
                ),  # conv_bias
                PartitionSpec(_GDN_TP_AXIS_NAME),  # A_log
                PartitionSpec(_GDN_TP_AXIS_NAME),  # dt_bias
                PartitionSpec(),  # state_indices
                PartitionSpec(),  # query_start_loc
                PartitionSpec(),  # request_distribution
                PartitionSpec(),  # seq_lens
            )
            output_partition_specs = (
                PartitionSpec(None, None, _GDN_PCP_TP_STATE_AXIS),  # new_conv_state
                PartitionSpec(
                    None, _GDN_PCP_TP_STATE_AXIS, None, None
                ),  # new_recurrent_state
                PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME, None),  # output
            )
            # Torchtpu native jax_op. The pcp_streaming wrapper adds only
            # kernel_reload's dispatcher, which GDN does not use.
            # pyrefly: ignore[no-matching-overload]
            gdn_jax_op = pallas.jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=(3, 4),
                mesh=pcp_mesh,
                input_partition_specs=input_partition_specs,
                output_partition_specs=output_partition_specs,
            )
        else:
            op_name = f"qwen4_exp::gdn_attention_{self.prefix.replace('.', '_')}"
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu,
                mesh=vllm_context.mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                num_spec_tokens=num_spec_tokens,
            )
            gdn_jax_op = pallas.jax_op(op_name, wrapped_fn, donate_argnums=(3, 4))

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            conv_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
            slot_read_offsets: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # The PCP prefill op keeps its original 13-arg signature. Its
            # speculative state tail is not a rollback operand.
            extra_args = () if pcp_streaming else (slot_read_offsets,)
            new_conv, new_rec, outputs = gdn_jax_op(
                mixed_qkv,
                b,
                a,
                conv_state,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
                *extra_args,
            )

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def _prepare_pooled_gdn_input(
        self,
        mixed_qkv: jax.Array,
        seq_lens: jax.Array,
        mesh: jax.sharding.Mesh,
    ) -> jax.Array | Conv1DInput:
        """Keep the ordinary projection input unchanged."""
        return mixed_qkv

    def _build_pooled_gdn_op(
        self,
        *,
        block_major: bool = False,
        hoisted: bool = False,
        single_checkpoint: bool = False,
        batched_only: bool = False,
    ):
        """The pooled GDN op. ``hoisted``: the variant taking the step's
        `gdn_step_metadata` outputs (VLLM_TPU_GDN_HOIST_METADATA)."""
        vllm_context = get_vllm_model_wrapper_context()
        local_num_v_heads = self.num_v_heads // self.tp_size
        vllm_config = cast(VllmConfig, vllm_context.vllm_config)
        mesh = vllm_context.mesh
        n_kq = self.tp_head_geometry.local_num_kq_heads
        d_k = self.head_k_dim
        d_v = self.head_v_dim
        kernel_size = self.conv_kernel_size
        recurrent_state_dtype = _to_jax_ssm_state_dtype(self.get_state_dtype()[1])
        conv_state_dtype = _to_jax_ssm_state_dtype(self._pooled_conv_state_dtype)
        # Speculative decoding: verify windows run in the GDN kernel's SPEC
        # mode with one state checkpoint per window position kept inside
        # the request's state block; rejected drafts are rolled back by
        # checkpoint selection (see TPUModelRunner.mamba_slot_read_offsets).
        num_spec_tokens = self.num_spec
        conv_weight_kmajor = _OPTIONS["GDN_CONV_WEIGHT_KMAJOR"]

        # Written out rather than functools.partial so pool_block_tokens is
        # read per call instead of at build time; pallas.jax_op requires a
        # fully annotated signature, so the operands are spelled out.
        def run(
            mixed_qkv,
            b,
            a,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            slot_read_offsets,
            ckpt_indices,
            commit_lens=None,
            read_offset_indices=None,
            step_metadata=None,
        ):
            # Fold the `(num_blocks, rows_per_block, *page)` block-major pool so
            # pool adapters index flat kernel blocks along dim 0; both reshapes
            # compile to zero-copy bitcasts on the donated buffer.
            pool = kernel_view(recurrent_state) if block_major else recurrent_state
            prepared_qkv = (
                mixed_qkv
                if batched_only
                else self._prepare_pooled_gdn_input(mixed_qkv, seq_lens, mesh)
            )
            result = gdn_attention_pooled_core_tpu(
                prepared_qkv,
                b,
                a,
                pool,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                slot_read_offsets,
                ckpt_indices,
                mesh=mesh,
                n_kq=n_kq,
                n_v=local_num_v_heads,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                # Manager block size: the pool may be born at a smaller
                # kernel granularity for backends with a fixed kernel
                # block. Read lazily at first call: the executor finalizes
                # cache_config.block_size only AFTER load_model, and hybrid
                # models construct GDN layers before the first
                # full-attention layer would ever see the adjusted value.
                pool_block_tokens=vllm_config.cache_config.block_size,
                recurrent_state_dtype=recurrent_state_dtype,
                conv_state_dtype=conv_state_dtype,
                num_spec_tokens=num_spec_tokens,
                conv_weight_kmajor=conv_weight_kmajor,
                single_checkpoint=single_checkpoint,
                batched_only=batched_only,
                commit_lens=commit_lens,
                read_offset_indices=read_offset_indices,
                step_metadata=step_metadata,
            )
            if not block_major:
                return result
            new_pool, output = result
            return new_pool.reshape(recurrent_state.shape), output

        def wrapped_fn(
            mixed_qkv: jax.Array,
            b: jax.Array,
            a: jax.Array,
            recurrent_state: jax.Array,
            conv_weight: jax.Array,
            conv_bias: jax.Array | None,
            A_log: jax.Array,
            dt_bias: jax.Array,
            state_indices: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            slot_read_offsets: jax.Array | None = None,
            ckpt_indices: jax.Array | None = None,
            commit_lens: jax.Array | None = None,
            read_offset_indices: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array]:
            return run(
                mixed_qkv,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                slot_read_offsets,
                ckpt_indices,
                commit_lens=commit_lens,
                read_offset_indices=read_offset_indices,
            )

        # VLLM_TPU_GDN_HOIST_METADATA: the step's `gdn_step_metadata` outputs
        # replace the slot read-offset buffer and the checkpoint ids.
        def wrapped_hoisted_fn(
            mixed_qkv: jax.Array,
            b: jax.Array,
            a: jax.Array,
            recurrent_state: jax.Array,
            conv_weight: jax.Array,
            conv_bias: jax.Array | None,
            A_log: jax.Array,
            dt_bias: jax.Array,
            state_indices: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            read_offsets: jax.Array,
            ckpt_indices: jax.Array,
            ps_query_start_loc: jax.Array,
            ps_seq_lens: jax.Array,
            ps_state_indices: jax.Array,
            ps_read_offsets: jax.Array,
            ps_ckpt_indices: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return run(
                mixed_qkv,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                None,
                None,
                step_metadata=(
                    read_offsets,
                    ckpt_indices,
                    ps_query_start_loc,
                    ps_seq_lens,
                    ps_state_indices,
                    ps_read_offsets,
                    ps_ckpt_indices,
                ),
            )

        def _fake_gdn(mixed_qkv, _b, _a, recurrent_state, *args, **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, d_v)
            out = torch.empty(out_shape, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
            return torch.empty_like(recurrent_state), out

        def build_op(op_name: str):
            # The recurrent state (arg 3) is the attention-shaped pool: the ssm
            # and conv byte-regions are read/written through it by the pool
            # adapters, so it is the only donated input. inplace-donation
            # aliases its program-level output to the input buffer so the
            # ~pool-sized update is not double-counted at compile.
            op = pallas.jax_op(
                f"qwen4_exp::{op_name}",
                wrapped_hoisted_fn if hoisted else wrapped_fn,
                donate_argnums=(3,),
            )
            op.register_fake(_fake_gdn)
            return op

        if hoisted:
            assert not block_major and not single_checkpoint and not batched_only
            kind = "gdn_attention_pooled_hoisted"
        elif batched_only:
            assert single_checkpoint
            kind = (
                "gdn_attention_pooled_replay_bm"
                if block_major
                else "gdn_attention_pooled_replay"
            )
        elif single_checkpoint:
            kind = (
                "gdn_attention_pooled_s1_bm"
                if block_major
                else "gdn_attention_pooled_s1"
            )
        else:
            kind = "gdn_attention_pooled_bm" if block_major else "gdn_attention_pooled"
        gdn_jax_op = _get_shared_gdn_op(
            kind,
            (
                type(self)._prepare_pooled_gdn_input,
                id(mesh),
                id(vllm_config),
                n_kq,
                local_num_v_heads,
                d_k,
                d_v,
                kernel_size,
                recurrent_state_dtype,
                conv_state_dtype,
                num_spec_tokens,
                conv_weight_kmajor,
                single_checkpoint,
                batched_only,
            ),
            build_op,
        )

        def gdn_impl(
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
            slot_read_offsets: torch.Tensor | None = None,
            ckpt_indices: torch.Tensor | None = None,
            *extra: torch.Tensor | None,
        ) -> torch.Tensor:
            # Hoisted: `slot_read_offsets` / `ckpt_indices` are the step's
            # read offsets / int32 checkpoint ids, followed by the 5 rolled
            # per-sequence inputs (see gdn_step_metadata).
            # Non-hoisted: optional `(commit_lens, read_offset_indices)`.
            if hoisted:
                assert len(extra) == 5
            else:
                assert len(extra) <= 2
            new_rec, outputs = gdn_jax_op(
                mixed_qkv,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
                slot_read_offsets,
                ckpt_indices,
                *extra,
            )

            # Plain donation + copy_ writeback (aliased in-place by XLA).
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def _build_pooled_pcp_gdn_op(self, *, block_major: bool = False):
        vllm_context = get_vllm_model_wrapper_context()
        local_num_v_heads = self.num_v_heads // self.tp_size
        local_num_kq_heads = self.tp_head_geometry.local_num_kq_heads
        has_conv_bias = self.conv1d.bias is not None
        parallel_config = cast(VllmConfig, vllm_context.vllm_config).parallel_config
        interleave_size = parallel_config.cp_kv_cache_interleave_size
        if not isinstance(interleave_size, int) or interleave_size <= 0:
            raise ValueError(
                "GDN pooled PCP prefill requires cp_kv_cache_interleave_size > 0."
            )
        pcp_mesh = get_or_create_pcp_mesh(
            axis_name=_GDN_PCP_AXIS_NAME,
            tp_axis_name=_GDN_TP_AXIS_NAME,
        )
        pcp_size = get_pcp_world_size()
        if pcp_size != pcp_mesh.shape[_GDN_PCP_AXIS_NAME]:
            raise ValueError(
                f"PCP world size {pcp_size} does not match mesh shape "
                f"{pcp_mesh.shape[_GDN_PCP_AXIS_NAME]}."
            )
        if self.tp_size != pcp_mesh.shape[_GDN_TP_AXIS_NAME]:
            raise ValueError(
                f"TP size {self.tp_size} does not match mesh shape "
                f"{pcp_mesh.shape[_GDN_TP_AXIS_NAME]}."
            )
        derive_gdn_head_geometry(local_num_kq_heads, local_num_v_heads, pcp_size)

        vllm_config = cast(VllmConfig, vllm_context.vllm_config)
        d_k = self.head_k_dim
        d_v = self.head_v_dim
        kernel_size = self.conv_kernel_size
        recurrent_state_dtype = _to_jax_ssm_state_dtype(self.get_state_dtype()[1])

        def wrapped_fn(
            hidden_states: jax.Array,
            qkvz_weight: jax.Array,
            b: jax.Array,
            a: jax.Array,
            recurrent_state: jax.Array,
            conv_weight: jax.Array,
            conv_bias: jax.Array | None,
            A_log: jax.Array,
            dt_bias: jax.Array,
            state_indices: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            qkvz_weight_scale: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            pool = kernel_view(recurrent_state) if block_major else recurrent_state
            result = run_jax_gdn_attention_pooled_pcp_prefill_projection(
                hidden_states,
                qkvz_weight,
                qkvz_weight_scale,
                b,
                a,
                pool,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=pcp_mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                pool_block_tokens=vllm_config.cache_config.block_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                recurrent_state_dtype=recurrent_state_dtype,
            )
            if not block_major:
                return result
            new_pool, output, z = result
            return new_pool.reshape(recurrent_state.shape), output, z

        op_name = (
            "qwen4_exp::gdn_attention_pooled_pcp_fused_"
            f"{self.prefix.replace('.', '_')}"
            f"{'_bm' if block_major else ''}"
        )
        input_partition_specs = (
            PartitionSpec(_GDN_PCP_AXIS_NAME, None),  # hidden states
            PartitionSpec(_GDN_TP_AXIS_NAME),  # QKVZ weight
            PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME),  # b
            PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME),  # a
            PartitionSpec(_GDN_PCP_TP_STATE_AXIS),  # recurrent_state pool (rank-local)
            PartitionSpec(_GDN_TP_AXIS_NAME),  # conv_weight
            (PartitionSpec(_GDN_TP_AXIS_NAME) if has_conv_bias else None),  # conv_bias
            PartitionSpec(_GDN_TP_AXIS_NAME),  # A_log
            PartitionSpec(_GDN_TP_AXIS_NAME),  # dt_bias
            PartitionSpec(),  # state_indices
            PartitionSpec(),  # query_start_loc
            PartitionSpec(),  # request_distribution
            PartitionSpec(),  # seq_lens
            PartitionSpec(_GDN_TP_AXIS_NAME),  # optional QKVZ weight scale
        )
        output_partition_specs = (
            PartitionSpec(_GDN_PCP_TP_STATE_AXIS),  # new pool
            PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME, None),  # output
            PartitionSpec(_GDN_PCP_AXIS_NAME, _GDN_TP_AXIS_NAME, None),  # z
        )
        # Torchtpu native jax_op. The pcp_streaming wrapper adds only
        # kernel_reload's dispatcher, which GDN does not use.
        # pyrefly: ignore[no-matching-overload]
        gdn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            # Keep optional operands after the pool so filtering None at the
            # TorchTPU boundary cannot change the donated tensor's index.
            donate_argnums=(4,),
            mesh=pcp_mesh,
            input_partition_specs=input_partition_specs,
            output_partition_specs=output_partition_specs,
        )

        def _fake_gdn(hidden_states, _weight, _b, _a, recurrent_state, *args, **kwargs):
            num_tokens = hidden_states.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            output = torch.empty(
                out_shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            return torch.empty_like(recurrent_state), output, torch.empty_like(output)

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(
            hidden_states: torch.Tensor,
            qkvz_weight: torch.Tensor,
            qkvz_weight_scale: torch.Tensor | None,
            b: torch.Tensor,
            a: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            new_rec, outputs, z = gdn_jax_op(
                hidden_states,
                qkvz_weight,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                request_distribution,
                seq_lens,
                qkvz_weight_scale,
            )
            # Match the non-PCP pooled path: donation aliases the Pallas
            # result to the input pool, while copy_ exposes the mutation to
            # the surrounding compiled graph without rebinding its storage.
            recurrent_state.copy_(new_rec)
            return outputs, z

        return gdn_impl

    def _require_pcp_projection_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return BF16/FP8 QKVZ parameters in canonical runtime layout."""
        if hasattr(self, "in_proj_qkv") or not hasattr(self, "in_proj_qkvz"):
            raise RuntimeError(
                "GDN pooled PCP prefill requires a combined QKVZ projection."
            )

        projection = self.in_proj_qkvz
        weight = getattr(projection, "weight", None)
        weight_scale = getattr(projection, "weight_scale", None)
        bias = getattr(projection, "bias", None)
        if weight is None or weight.ndim != 2 or bias is not None:
            raise RuntimeError(
                "GDN pooled PCP prefill requires bias-free rank-2 QKVZ weights."
            )
        if weight.dtype == torch.bfloat16:
            if weight_scale is not None:
                raise RuntimeError("BF16 QKVZ weights must not have a scale.")
        elif weight.dtype == torch.float8_e4m3fn:
            if weight_scale is None or weight_scale.dtype != torch.float32:
                raise RuntimeError("FP8 QKVZ weights require an FP32 scale.")
            from ..kernels.gdn_v3.projection_scale import (
                projection_scale_layout,
            )

            try:
                projection_scale_layout(weight.shape, weight_scale.shape)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        else:
            raise RuntimeError(
                "GDN pooled PCP prefill requires BF16 or float8_e4m3fn QKVZ weights."
            )
        return weight, weight_scale

    def _has_active_pooled_pcp_state(self, kv_cache) -> bool:
        """Whether a PCP-configured worker owns an allocated unified pool."""
        return (
            kv_cache is not None
            and len(kv_cache) == 1
            and kv_cache[0].numel() > 0
            and self._pcp_streaming_configured
        )

    def _apply_output_projection(
        self, core_attn_out: torch.Tensor, z: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        local_num_v_heads = self.num_v_heads // self.tp_size
        core_attn_out = core_attn_out.view(
            num_tokens, local_num_v_heads, self.head_v_dim
        )
        core_attn_out = self.norm(core_attn_out, z)
        # pyrefly: ignore[bad-argument-type]
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output, _ = self.out_proj(core_attn_out)
        return output

    def _forward_fused_pooled_pcp(
        self, hidden_states: torch.Tensor, state_pool: torch.Tensor, attn_metadata
    ) -> torch.Tensor:
        """Run the mandatory fused path for an allocated pooled PCP call."""
        if not is_pcp_streaming_attention_metadata(attn_metadata):
            raise RuntimeError(
                "PCP is configured with a unified pool, but attention "
                "metadata does not use the PCP streaming sequence layout."
            )

        gdn_pooled_pcp_op = self.gdn_pooled_pcp_op
        assert gdn_pooled_pcp_op is not None
        qkvz_weight, qkvz_weight_scale = self._require_pcp_projection_parameters()
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        state_indices = mamba_state_of(attn_metadata)
        assert state_indices is not None
        state_indices = self._bm_state_ids(state_indices.to(torch.int32))
        core_attn_out, z = gdn_pooled_pcp_op(
            hidden_states,
            qkvz_weight,
            qkvz_weight_scale,
            b,
            a,
            state_pool,
            self.conv1d.weight,
            self.conv1d.bias,
            self.A_log,
            self.dt_bias,
            state_indices,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
            attn_metadata.seq_lens,
        )
        num_tokens = hidden_states.size(0)
        if core_attn_out.shape[0] != num_tokens:
            raise RuntimeError("GDN op returned an incompatible output shape.")
        return self._apply_output_projection(core_attn_out, z, num_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        gdn_step_metadata_cache: GdnStepMetadataCache | None = None,
        verify_collector: (
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        kv_cache = self.kv_cache
        if self._has_active_pooled_pcp_state(kv_cache):
            assert kv_cache is not None
            attn_metadata = cast(
                "dict[str, _GdnAttentionMetadata]", get_forward_context().attn_metadata
            )[self.prefix]
            return self._forward_fused_pooled_pcp(
                hidden_states,
                kv_cache[0],
                attn_metadata,
            )

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            # pyrefly: ignore[not-callable]
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            # pyrefly: ignore[not-callable]
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            # Every checkpoint loader produces [q, k, v, z] and [b, a].
            qkv_size = self.tp_head_geometry.local_conv_dim(
                self.head_k_dim, self.head_v_dim
            )
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        local_num_v_heads = self.num_v_heads // self.tp_size

        # During warmup or memory profiling, the kv_cache might not be allocated yet
        if kv_cache is None or kv_cache[0].numel() == 0:
            core_attn_out = torch.zeros(
                (num_tokens, local_num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            recurrent_state = None
        else:
            attn_metadata = cast(
                "dict[str, _GdnAttentionMetadata]", get_forward_context().attn_metadata
            )[self.prefix]

            if len(kv_cache) == 1:
                # Unified block pool: the attention-shaped pool carries the
                # ssm and conv byte-regions; state ids come from the pool
                # metadata builder.
                (recurrent_state,) = kv_cache
                md_state = mamba_state_of(attn_metadata)
                assert md_state is not None
                raw_state_indices = md_state.to(torch.int32)
                state_indices = self._bm_state_ids(raw_state_indices)
                # Speculative decoding: the GDN kernel's windowed segment
                # covers both 1-token decodes and speculative verify
                # windows (the batch is ordered [decode][verify][prefill]);
                # ragged paged attention keeps `request_distribution` with
                # its 1-token decode front segment.
                request_distribution = attn_metadata.request_distribution
                slot_read_offsets = getattr(
                    attn_metadata, "mamba_slot_read_offsets", None
                )
                mamba_request_distribution = getattr(
                    attn_metadata, "mamba_request_distribution", None
                )
                if mamba_request_distribution is not None:
                    request_distribution = mamba_request_distribution
                if self.num_spec > 0:
                    assert slot_read_offsets is not None, (
                        "Speculative decoding with GDN layers requires "
                        "mamba_slot_read_offsets in the attention metadata."
                    )
                ckpt_indices = mamba_ckpt_of(attn_metadata)
                use_two_slot = (
                    self._two_slot_mtp
                    and self.gdn_pooled_single_ckpt_op is not None
                    and ckpt_indices is not None
                    and ckpt_indices.ndim == 2
                    and ckpt_indices.shape[1] in (1, 2)
                )
                if use_two_slot and self._verify_win_cap > 0:
                    n_save = min(num_tokens, self._verify_win_cap)
                    if n_save > 0:
                        qkv_save = mixed_qkv[:n_save]
                        b_save = b[:n_save]
                        a_save = a[:n_save]
                        if verify_collector is not None:
                            verify_collector.append((qkv_save, b_save, a_save))
                        else:
                            qkv_buf = self._verify_qkv_buf
                            b_buf = self._verify_b_buf
                            a_buf = self._verify_a_buf
                            if (
                                qkv_buf is not None
                                and b_buf is not None
                                and a_buf is not None
                            ):
                                if qkv_buf.device.type == "tpu":
                                    _update_verify_rows_(qkv_buf, qkv_save)
                                    _update_verify_rows_(b_buf, b_save)
                                    _update_verify_rows_(a_buf, a_save)
                                else:
                                    qkv_buf[:n_save].copy_(qkv_save.to(qkv_buf.dtype))
                                    b_buf[:n_save].copy_(b_save.to(b_buf.dtype))
                                    a_buf[:n_save].copy_(a_save.to(a_buf.dtype))
                if use_two_slot and ckpt_indices is not None:
                    ckpt_indices = self._bm_ckpt_ids(ckpt_indices.to(torch.int32))
                step_metadata: tuple[torch.Tensor | None, ...] = ()
                pooled_op = (
                    self.gdn_pooled_single_ckpt_op
                    if use_two_slot
                    else self.gdn_pooled_op
                )
                assert pooled_op is not None
                if (
                    not use_two_slot
                    and gdn_step_metadata_cache is not None
                    and self.gdn_pooled_hoisted_op is not None
                    and self.gdn_step_metadata_op is not None
                    and slot_read_offsets is not None
                    and ckpt_indices is not None
                ):
                    # VLLM_TPU_GDN_HOIST_METADATA: the read-offset gather, the
                    # int32 casts and the PER_SEQ roll once per step and group.
                    group_inputs = (
                        slot_read_offsets,
                        state_indices,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        request_distribution,
                        ckpt_indices,
                    )
                    slot_read_offsets, ckpt_indices, *rest = (
                        gdn_step_metadata_cache.get(
                            attn_metadata,
                            lambda: tuple(self.gdn_step_metadata_op(*group_inputs)),
                        )
                    )
                    step_metadata = tuple(rest)
                    pooled_op = self.gdn_pooled_hoisted_op
                elif use_two_slot and self._bm_kernel and slot_read_offsets is not None:
                    # Under merged block-major pools, `slot_read_offsets` is
                    # indexed by unscaled manager block ID (`raw_state_indices`),
                    # while `state_indices` / `ckpt_indices` hold physical pool
                    # block IDs.
                    step_metadata = (None, raw_state_indices)
                core_attn_out = pooled_op(
                    mixed_qkv,
                    b,
                    a,
                    recurrent_state,
                    self._pooled_conv_weight(),
                    self.conv1d.bias,
                    self.A_log,
                    self.dt_bias,
                    state_indices,
                    attn_metadata.query_start_loc,
                    request_distribution,
                    attn_metadata.seq_lens,
                    slot_read_offsets,
                    ckpt_indices,
                    *step_metadata,
                )
                if core_attn_out.shape[0] != num_tokens:
                    raise RuntimeError("GDN op returned an incompatible output shape.")
            else:
                conv_state, recurrent_state = kv_cache
                # Recurrent-state slot id per persistent-batch position.
                # Compact-mamba: the mamba pool has only `_mamba_num_blocks` slots
                # (< attention `num_blocks`), so the slot id is carried explicitly
                # in `mamba_state_indices` (in [0, _mamba_num_blocks)) rather than
                # derived from the attention `block_tables[:, 0]`. Fall back to
                # `block_tables[:, 0]` only when compact sizing was skipped and
                # mamba shares the attention block pool (uniform layout).
                if mamba_state_of(attn_metadata) is not None:
                    state_indices = mamba_state_of(attn_metadata).to(torch.int32)
                else:
                    max_reqs = attn_metadata.seq_lens.shape[0]
                    max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
                    block_tables_2d = torch.reshape(
                        attn_metadata.block_tables,
                        (max_reqs, max_blocks_per_req),
                    )
                    state_indices = block_tables_2d[:, 0].to(torch.int32)

                # Speculative decoding: the GDN kernel's windowed segment
                # covers both 1-token decodes and speculative verify windows
                # (the batch is ordered [decode][verify][prefill/mixed]);
                # ragged paged attention keeps using `request_distribution`
                # with its 1-token decode front segment.
                request_distribution = attn_metadata.request_distribution
                slot_read_offsets = getattr(
                    attn_metadata, "mamba_slot_read_offsets", None
                )
                mamba_request_distribution = getattr(
                    attn_metadata, "mamba_request_distribution", None
                )
                if mamba_request_distribution is not None:
                    request_distribution = mamba_request_distribution
                use_pcp_streaming = is_pcp_streaming_attention_metadata(attn_metadata)
                if not use_pcp_streaming and self.num_spec > 0:
                    assert slot_read_offsets is not None, (
                        "Speculative decoding with GDN layers requires "
                        "mamba_slot_read_offsets in the attention metadata."
                    )

                # Execute the TorchTPU custom op
                if not use_pcp_streaming:
                    core_attn_out = self.gdn_op(
                        mixed_qkv,
                        b,
                        a,
                        conv_state,
                        recurrent_state,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        self.A_log,
                        self.dt_bias,
                        state_indices,
                        attn_metadata.query_start_loc,
                        request_distribution,
                        attn_metadata.seq_lens,
                        slot_read_offsets,
                    )
                else:
                    gdn_pcp_op = self.gdn_pcp_op
                    if gdn_pcp_op is None:
                        raise RuntimeError(
                            "GDN PCP prefill op was not initialized during model "
                            "loading."
                        )
                    core_attn_out = gdn_pcp_op(
                        mixed_qkv,
                        b,
                        a,
                        conv_state,
                        recurrent_state,
                        self.conv1d.weight,
                        self.conv1d.bias,
                        self.A_log,
                        self.dt_bias,
                        state_indices,
                        attn_metadata.query_start_loc,
                        attn_metadata.request_distribution,
                        attn_metadata.seq_lens,
                    )
                if core_attn_out.shape[0] != num_tokens:
                    if not use_pcp_streaming:
                        raise RuntimeError(
                            "GDN op returned an incompatible output shape."
                        )
                    start = get_pcp_rank() * num_tokens
                    core_attn_out = core_attn_out[start : start + num_tokens]

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._apply_output_projection(core_attn_out, z, num_tokens)


def _active_verify_flat_tensors(
    layer: Any, *, force_sp: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Select the SP (`dim=-1`) or non-SP (`dim=0`) shared verify flat tensors."""
    bref = layer._shared_verify_backbone_ref
    sp_flat = layer._shared_verify_sp_flat_tensors
    use_sp = (
        force_sp
        if force_sp is not None
        else (bref is not None and bool(bref._last_verify_flat_from_sp))
    )
    if use_sp and sp_flat is not None:
        return sp_flat
    return layer._shared_verify_flat_tensors


def _compute_flip_commit_lens_and_valid(
    query_start_loc: torch.Tensor,
    next_tokens: torch.Tensor,
    win_and_reqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive `(qsl_i32, wr_i32, num_valid, cl_st)` for fused replay+flip."""
    qsl_i32 = query_start_loc.to(torch.int32)
    q_lens = qsl_i32[1:] - qsl_i32[:-1]
    max_seqs = q_lens.shape[0]
    raw_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
    if raw_valid.shape[0] >= max_seqs:
        num_valid = raw_valid[:max_seqs]
    else:
        num_valid = torch.nn.functional.pad(
            raw_valid, (0, max_seqs - raw_valid.shape[0])
        )
    row_idx = torch.arange(max_seqs, device=q_lens.device, dtype=torch.int32)
    wr_i32 = win_and_reqs.to(torch.int32)
    is_win = row_idx < wr_i32[0]
    rejected = is_win & (num_valid < q_lens) & (num_valid > 0)
    cl_st = torch.where(rejected, num_valid, torch.zeros_like(q_lens)).unsqueeze(0)
    return qsl_i32, wr_i32, num_valid, cl_st


def _torch_to_jax_cpu(t: torch.Tensor) -> jax.Array:
    if t.dtype == torch.bfloat16:
        arr = t.detach().cpu().view(torch.uint16).numpy()
        return jnp.asarray(arr).view(jnp.bfloat16)
    return jnp.asarray(t.detach().cpu().numpy())


def _copy_jax_to_torch_cpu(dst: torch.Tensor, src: jax.Array) -> None:
    if dst.dtype == torch.bfloat16:
        raw = np.asarray(jax.lax.bitcast_convert_type(src, jnp.uint16)).copy()
        dst.copy_(torch.from_numpy(raw).view(torch.bfloat16))
    else:
        dst.copy_(torch.from_numpy(np.asarray(src).copy()))


def compute_fused_flip_and_ple_offsets_jax(
    slot_read_offsets: jax.Array,
    raw_state_indices_stack: jax.Array,
    query_start_loc: jax.Array,
    win_and_reqs: jax.Array,
    ple_state_stack: jax.Array | None = None,
    ple_offsets_1d: jax.Array | None = None,
) -> jax.Array:
    """Pure JAX fused read-offset scatter covering GDN ping-pong + PLE groups.

    Flips active 2-slot ping-pong GDN slots (`0 <-> 1`), resets non-windowed
    active GDN rows to `0`, and optionally scatters `num_valid - 1` offsets
    for all standard multi-slot PLE Mamba groups in a single buffer update.
    """
    max_seqs = int(query_start_loc.shape[0]) - 1
    row_idx = jnp.arange(max_seqs, dtype=jnp.int32)
    q_lens = query_start_loc[1:].astype(jnp.int32) - query_start_loc[:-1].astype(
        jnp.int32
    )
    num_win = win_and_reqs[0].astype(jnp.int32)
    num_reqs = win_and_reqs[1].astype(jnp.int32)
    active = (row_idx < num_reqs) & (q_lens > 0)
    is_win_row = row_idx < num_win
    n_off = int(slot_read_offsets.shape[0])
    s_i32 = raw_state_indices_stack.astype(jnp.int32)
    valid_pp = active[None, :] & (s_i32 > 0) & (s_i32 < n_off)
    targets = jnp.where(valid_pp, s_i32, 0)
    old_off = jnp.clip(slot_read_offsets[targets], 0, 1)
    flipped = (1 - old_off).astype(slot_read_offsets.dtype)
    new_vals = jnp.where(
        valid_pp & is_win_row[None, :],
        flipped,
        jnp.zeros_like(flipped),
    )
    if ple_state_stack is not None and ple_offsets_1d is not None:
        p_i32 = ple_state_stack.astype(jnp.int32)
        ple_active = row_idx < num_reqs
        valid_ple = ple_active[None, :] & (p_i32 > 0) & (p_i32 < n_off)
        ple_targets = jnp.where(valid_ple, p_i32, 0)
        ple_vals = jnp.where(
            valid_ple,
            jnp.broadcast_to(
                ple_offsets_1d.astype(slot_read_offsets.dtype)[None, :],
                p_i32.shape,
            ),
            jnp.zeros(p_i32.shape, dtype=slot_read_offsets.dtype),
        )
        all_targets = jnp.concatenate(
            [targets.reshape(-1), ple_targets.reshape(-1)], axis=0
        )
        all_vals = jnp.concatenate([new_vals.reshape(-1), ple_vals.reshape(-1)], axis=0)
        return slot_read_offsets.at[all_targets].set(all_vals)
    return slot_read_offsets.at[targets.reshape(-1)].set(new_vals.reshape(-1))


class MultiLayerPooledReplayRunner:
    """Single-graph multi-layer 2-slot ping-pong GDN replay + offset updater.

    Combines `L` homogeneous pooled GDN layers (arbitrary `L`, e.g. 36 or 69)
    into one donated JAX custom op under the `qwen4_exp::` namespace:
      - Invokes `run_multi_layer_batched_replay_pooled` (one Pallas kernel
        launch with 1D `jnp.nonzero` compaction) gated by `jax.lax.cond` so an
        all-accepted step performs zero replay kernel work.
      - `replay_and_flip(...)` optionally updates both ping-pong GDN and
        standard multi-slot PLE read offsets in the same donated graph.
    """

    def __init__(
        self,
        layers: Sequence[VllmGatedDeltaNetAttention],
    ) -> None:
        self.layers = tuple(layers)
        self.num_layers = len(self.layers)
        layer0 = self.layers[0]
        vllm_context = get_vllm_model_wrapper_context()
        vllm_config = cast(VllmConfig, vllm_context.vllm_config)
        mesh = vllm_context.mesh
        local_num_v_heads = layer0.num_v_heads // layer0.tp_size
        n_kq = layer0.tp_head_geometry.local_num_kq_heads
        d_k = layer0.head_k_dim
        d_v = layer0.head_v_dim
        kernel_size = layer0.conv_kernel_size
        qkv_dim = layer0.tp_head_geometry.local_conv_dim(d_k, d_v)
        num_spec_tokens = layer0.num_spec
        recurrent_state_dtype = _to_jax_ssm_state_dtype(layer0.get_state_dtype()[1])
        conv_state_dtype = _to_jax_ssm_state_dtype(layer0._pooled_conv_state_dtype)
        conv_weight_kmajor = _OPTIONS["GDN_CONV_WEIGHT_KMAJOR"]
        block_major = bool(layer0._bm_kernel)
        has_conv_bias = layer0.conv1d.bias is not None
        pool_indices: tuple[int, ...] = (
            tuple(int(cast(Any, layer._bm_region)[0]) for layer in self.layers)
            if block_major
            else ()
        )
        num_pools = int(cast(Any, layer0._bm_region)[1].num_pools) if block_major else 1
        num_layers = self.num_layers
        self._has_conv_bias = has_conv_bias

        def _replay_pool_fn(
            rec_state: jax.Array,
            mixed_qkv_in: jax.Array,
            b_in: jax.Array,
            a_in: jax.Array,
            conv_weight_stack: jax.Array,
            conv_bias_stack: jax.Array | None,
            A_log_stack: jax.Array,
            dt_bias_stack: jax.Array,
            raw_state_indices_stack: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            slot_read_offsets: jax.Array,
            raw_ckpt_indices_stack: jax.Array,
            commit_lens_stack: jax.Array,
            *,
            flat_layout: bool,
        ) -> jax.Array:
            cur_pool = kernel_view(rec_state) if block_major else rec_state
            if flat_layout:
                if mixed_qkv_in.shape[-1] == qkv_dim:
                    # Non-SP dim=0 layout: [L * win_stride, dim].
                    win_rows = mixed_qkv_in.shape[0] // num_layers
                    qkv_3d = mixed_qkv_in.reshape(num_layers, win_rows, qkv_dim)
                    b_3d = b_in.reshape(num_layers, win_rows, local_num_v_heads)
                    a_3d = a_in.reshape(num_layers, win_rows, local_num_v_heads)
                else:
                    # SP dim=-1 layout: [win_cap, L * dim].
                    win_rows = mixed_qkv_in.shape[0]
                    qkv_3d = jnp.swapaxes(
                        mixed_qkv_in.reshape(win_rows, num_layers, qkv_dim), 0, 1
                    )
                    b_3d = jnp.swapaxes(
                        b_in.reshape(win_rows, num_layers, local_num_v_heads), 0, 1
                    )
                    a_3d = jnp.swapaxes(
                        a_in.reshape(win_rows, num_layers, local_num_v_heads), 0, 1
                    )
            else:
                win_rows = mixed_qkv_in.shape[1]
                qkv_3d = mixed_qkv_in
                b_3d = b_in
                a_3d = a_in
            qsl_clamped = jnp.minimum(query_start_loc.astype(jnp.int32), win_rows)
            block_tokens = vllm_config.cache_config.block_size
            s_rows = raw_state_indices_stack.shape[0]
            c_rows = raw_ckpt_indices_stack.shape[0]
            max_seqs = seq_lens.shape[0]

            raw_s_2d = (
                jnp.broadcast_to(raw_state_indices_stack, (num_layers, max_seqs))
                if s_rows == 1
                else raw_state_indices_stack
            ).astype(jnp.int32)
            raw_c_3d = (
                jnp.broadcast_to(
                    raw_ckpt_indices_stack,
                    (num_layers, max_seqs, raw_ckpt_indices_stack.shape[-1]),
                )
                if c_rows == 1
                else raw_ckpt_indices_stack
            ).astype(jnp.int32)
            if block_major and num_pools > 1:
                pidx_arr = jnp.asarray(pool_indices, dtype=jnp.int32)[:, None]
                s_ids_2d = raw_s_2d * num_pools + pidx_arr
                c_ids_3d = jnp.where(
                    raw_c_3d > 0,
                    raw_c_3d * num_pools + pidx_arr[:, :, None],
                    0,
                )
                ro_idx_2d = raw_s_2d
            else:
                s_ids_2d = raw_s_2d
                c_ids_3d = raw_c_3d
                ro_idx_2d = s_ids_2d
            ro_max = max(0, int(slot_read_offsets.shape[0]) - 1)
            ro_2d = slot_read_offsets[jnp.clip(ro_idx_2d, 0, ro_max)]
            # Keep [1, max_seqs] commit_lens_stack when shared across layers
            # so the kernel selects the 1D `jnp.nonzero` compaction fast path.
            cur_pool = run_jax_gdn_multi_layer_replay_pooled(
                qkv_3d,
                b_3d,
                a_3d,
                cur_pool,
                conv_weight_stack,
                conv_bias_stack,
                A_log_stack,
                dt_bias_stack,
                s_ids_2d,
                ro_2d,
                c_ids_3d,
                commit_lens_stack.astype(jnp.int32),
                qsl_clamped,
                distribution,
                seq_lens,
                mesh=mesh,
                n_kq=n_kq,
                n_v=local_num_v_heads,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                pool_block_tokens=block_tokens,
                recurrent_state_dtype=recurrent_state_dtype,
                conv_state_dtype=conv_state_dtype,
                num_spec_tokens=num_spec_tokens,
                conv_weight_kmajor=conv_weight_kmajor,
            )
            if block_major:
                cur_pool = cur_pool.reshape(rec_state.shape)
            return cur_pool

        def _dispatch(
            rec_state: jax.Array,
            mixed_qkv_in: jax.Array,
            b_in: jax.Array,
            a_in: jax.Array,
            conv_weight_stack: jax.Array,
            conv_bias_stack: jax.Array | None,
            A_log_stack: jax.Array,
            dt_bias_stack: jax.Array,
            raw_state_indices_stack: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            slot_read_offsets: jax.Array,
            raw_ckpt_indices_stack: jax.Array,
            commit_lens_stack: jax.Array,
            *,
            flat_layout: bool,
            win_and_reqs: jax.Array | None = None,
            ple_state_stack: jax.Array | None = None,
            ple_offsets_1d: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array]:
            req_width = commit_lens_stack.shape[1]
            valid_mask = (
                jnp.arange(req_width, dtype=jnp.int32)
                < distribution[0].astype(jnp.int32)
            )[None, :]
            has_work = jnp.any(valid_mask & (commit_lens_stack > 0))
            out_state = jax.lax.cond(
                has_work,
                lambda s: _replay_pool_fn(
                    s,
                    mixed_qkv_in,
                    b_in,
                    a_in,
                    conv_weight_stack,
                    conv_bias_stack,
                    A_log_stack,
                    dt_bias_stack,
                    raw_state_indices_stack,
                    query_start_loc,
                    distribution,
                    seq_lens,
                    slot_read_offsets,
                    raw_ckpt_indices_stack,
                    commit_lens_stack,
                    flat_layout=flat_layout,
                ),
                lambda s: s,
                rec_state,
            )
            if win_and_reqs is not None:
                new_ro = compute_fused_flip_and_ple_offsets_jax(
                    slot_read_offsets,
                    raw_state_indices_stack,
                    query_start_loc,
                    win_and_reqs,
                    ple_state_stack=ple_state_stack,
                    ple_offsets_1d=ple_offsets_1d,
                )
                return out_state, new_ro
            return out_state, raw_state_indices_stack[0, 0]

        def _make_wrapped(
            flat_layout: bool, flip: bool, with_bias: bool, with_ple: bool = False
        ):
            if not flip and not with_bias:

                def fn(
                    rec: jax.Array,
                    qkv: jax.Array,
                    b: jax.Array,
                    a: jax.Array,
                    cw: jax.Array,
                    al: jax.Array,
                    dtb: jax.Array,
                    s: jax.Array,
                    qsl: jax.Array,
                    dist: jax.Array,
                    sl: jax.Array,
                    ro: jax.Array,
                    c: jax.Array,
                    cl: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _dispatch(
                        rec,
                        qkv,
                        b,
                        a,
                        cw,
                        None,
                        al,
                        dtb,
                        s,
                        qsl,
                        dist,
                        sl,
                        ro,
                        c,
                        cl,
                        flat_layout=flat_layout,
                    )

                return fn
            if not flip and with_bias:

                def fn(
                    rec: jax.Array,
                    qkv: jax.Array,
                    b: jax.Array,
                    a: jax.Array,
                    cw: jax.Array,
                    cb: jax.Array,
                    al: jax.Array,
                    dtb: jax.Array,
                    s: jax.Array,
                    qsl: jax.Array,
                    dist: jax.Array,
                    sl: jax.Array,
                    ro: jax.Array,
                    c: jax.Array,
                    cl: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _dispatch(
                        rec,
                        qkv,
                        b,
                        a,
                        cw,
                        cb,
                        al,
                        dtb,
                        s,
                        qsl,
                        dist,
                        sl,
                        ro,
                        c,
                        cl,
                        flat_layout=flat_layout,
                    )

                return fn
            if flip and not with_bias and not with_ple:

                def fn(
                    rec: jax.Array,
                    qkv: jax.Array,
                    b: jax.Array,
                    a: jax.Array,
                    cw: jax.Array,
                    al: jax.Array,
                    dtb: jax.Array,
                    s: jax.Array,
                    qsl: jax.Array,
                    dist: jax.Array,
                    sl: jax.Array,
                    ro: jax.Array,
                    c: jax.Array,
                    cl: jax.Array,
                    wr: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _dispatch(
                        rec,
                        qkv,
                        b,
                        a,
                        cw,
                        None,
                        al,
                        dtb,
                        s,
                        qsl,
                        dist,
                        sl,
                        ro,
                        c,
                        cl,
                        flat_layout=flat_layout,
                        win_and_reqs=wr,
                    )

                return fn
            if flip and with_bias and not with_ple:

                def fn(
                    rec: jax.Array,
                    qkv: jax.Array,
                    b: jax.Array,
                    a: jax.Array,
                    cw: jax.Array,
                    cb: jax.Array,
                    al: jax.Array,
                    dtb: jax.Array,
                    s: jax.Array,
                    qsl: jax.Array,
                    dist: jax.Array,
                    sl: jax.Array,
                    ro: jax.Array,
                    c: jax.Array,
                    cl: jax.Array,
                    wr: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _dispatch(
                        rec,
                        qkv,
                        b,
                        a,
                        cw,
                        cb,
                        al,
                        dtb,
                        s,
                        qsl,
                        dist,
                        sl,
                        ro,
                        c,
                        cl,
                        flat_layout=flat_layout,
                        win_and_reqs=wr,
                    )

                return fn
            if flip and not with_bias and with_ple:

                def fn(
                    rec: jax.Array,
                    qkv: jax.Array,
                    b: jax.Array,
                    a: jax.Array,
                    cw: jax.Array,
                    al: jax.Array,
                    dtb: jax.Array,
                    s: jax.Array,
                    qsl: jax.Array,
                    dist: jax.Array,
                    sl: jax.Array,
                    ro: jax.Array,
                    c: jax.Array,
                    cl: jax.Array,
                    wr: jax.Array,
                    ps: jax.Array,
                    po: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _dispatch(
                        rec,
                        qkv,
                        b,
                        a,
                        cw,
                        None,
                        al,
                        dtb,
                        s,
                        qsl,
                        dist,
                        sl,
                        ro,
                        c,
                        cl,
                        flat_layout=flat_layout,
                        win_and_reqs=wr,
                        ple_state_stack=ps,
                        ple_offsets_1d=po,
                    )

                return fn

            def fn(
                rec: jax.Array,
                qkv: jax.Array,
                b: jax.Array,
                a: jax.Array,
                cw: jax.Array,
                cb: jax.Array,
                al: jax.Array,
                dtb: jax.Array,
                s: jax.Array,
                qsl: jax.Array,
                dist: jax.Array,
                sl: jax.Array,
                ro: jax.Array,
                c: jax.Array,
                cl: jax.Array,
                wr: jax.Array,
                ps: jax.Array,
                po: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return _dispatch(
                    rec,
                    qkv,
                    b,
                    a,
                    cw,
                    cb,
                    al,
                    dtb,
                    s,
                    qsl,
                    dist,
                    sl,
                    ro,
                    c,
                    cl,
                    flat_layout=flat_layout,
                    win_and_reqs=wr,
                    ple_state_stack=ps,
                    ple_offsets_1d=po,
                )

            return fn

        def _fake_replay_only(rec: torch.Tensor, *args, **kwargs):
            del args, kwargs
            return torch.empty_like(rec), torch.empty(
                (), dtype=torch.int32, device=rec.device
            )

        def _fake_replay_and_flip(rec: torch.Tensor, *args, **kwargs):
            del kwargs
            ro_idx = (12 if has_conv_bias else 11) - 1
            ro = args[ro_idx]
            return torch.empty_like(rec), torch.empty_like(ro)

        def build_ops(op_name: str):
            fn_std = _make_wrapped(False, False, has_conv_bias)
            op_std = pallas.jax_op(
                f"qwen4_exp::{op_name}_std", fn_std, donate_argnums=(0,)
            )
            op_std.register_fake(_fake_replay_only)

            fn_flat = _make_wrapped(True, False, has_conv_bias)
            op_flat = pallas.jax_op(
                f"qwen4_exp::{op_name}_flat", fn_flat, donate_argnums=(0,)
            )
            op_flat.register_fake(_fake_replay_only)

            ro_pos = 12 if has_conv_bias else 11
            fn_flip = _make_wrapped(True, True, has_conv_bias, with_ple=False)
            op_flip = pallas.jax_op(
                f"qwen4_exp::{op_name}_flip", fn_flip, donate_argnums=(0, ro_pos)
            )
            op_flip.register_fake(_fake_replay_and_flip)

            fn_flip_ple = _make_wrapped(True, True, has_conv_bias, with_ple=True)
            op_flip_ple = pallas.jax_op(
                f"qwen4_exp::{op_name}_flip_ple",
                fn_flip_ple,
                donate_argnums=(0, ro_pos),
            )
            op_flip_ple.register_fake(_fake_replay_and_flip)

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def _compiled_std(
                recurrent_state: torch.Tensor,
                qkv_3d: torch.Tensor,
                b_3d: torch.Tensor,
                a_3d: torch.Tensor,
                *rest: torch.Tensor,
            ) -> None:
                new_rec, _ = op_std(recurrent_state, qkv_3d, b_3d, a_3d, *rest)
                recurrent_state.copy_(new_rec)

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def _compiled_flat(
                recurrent_state: torch.Tensor,
                qkv_flat: torch.Tensor,
                b_flat: torch.Tensor,
                a_flat: torch.Tensor,
                *rest: torch.Tensor,
            ) -> None:
                new_rec, _ = op_flat(recurrent_state, qkv_flat, b_flat, a_flat, *rest)
                recurrent_state.copy_(new_rec)

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def _compiled_flip(
                recurrent_state: torch.Tensor,
                qkv_flat: torch.Tensor,
                b_flat: torch.Tensor,
                a_flat: torch.Tensor,
                *rest: torch.Tensor,
            ) -> None:
                w_cnt = 4 if has_conv_bias else 3
                weights = rest[:w_cnt]
                s_st, c_st, qsl, next_tokens, win_and_reqs, dist, slens, ro = rest[
                    w_cnt:
                ]
                qsl_i32, wr_i32, _, cl_st = _compute_flip_commit_lens_and_valid(
                    qsl, next_tokens, win_and_reqs
                )
                new_rec, new_ro = op_flip(
                    recurrent_state,
                    qkv_flat,
                    b_flat,
                    a_flat,
                    *weights,
                    s_st.to(torch.int32),
                    qsl_i32,
                    dist.to(torch.int32),
                    slens.to(torch.int32),
                    ro,
                    c_st.to(torch.int32),
                    cl_st,
                    wr_i32,
                )
                recurrent_state.copy_(new_rec)
                ro.copy_(new_ro)

            @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
            def _compiled_flip_with_ple(
                recurrent_state: torch.Tensor,
                qkv_flat: torch.Tensor,
                b_flat: torch.Tensor,
                a_flat: torch.Tensor,
                *rest: torch.Tensor,
            ) -> None:
                w_cnt = 4 if has_conv_bias else 3
                weights = rest[:w_cnt]
                (
                    s_st,
                    c_st,
                    qsl,
                    next_tokens,
                    win_and_reqs,
                    dist,
                    slens,
                    ro,
                    ple_s_st,
                ) = rest[w_cnt:]
                qsl_i32, wr_i32, num_valid, cl_st = _compute_flip_commit_lens_and_valid(
                    qsl, next_tokens, win_and_reqs
                )
                ple_off_1d = (num_valid - 1).clamp(min=0)
                new_rec, new_ro = op_flip_ple(
                    recurrent_state,
                    qkv_flat,
                    b_flat,
                    a_flat,
                    *weights,
                    s_st.to(torch.int32),
                    qsl_i32,
                    dist.to(torch.int32),
                    slens.to(torch.int32),
                    ro,
                    c_st.to(torch.int32),
                    cl_st,
                    wr_i32,
                    ple_s_st.to(torch.int32),
                    ple_off_1d,
                )
                recurrent_state.copy_(new_rec)
                ro.copy_(new_ro)

            return (
                fn_std,
                fn_flat,
                fn_flip,
                fn_flip_ple,
                _compiled_std,
                _compiled_flat,
                _compiled_flip,
                _compiled_flip_with_ple,
            )

        kind = (
            f"gdn_multi_layer_pooled_replay_{num_layers}l"
            f"{'_bm' if block_major else ''}"
            f"{'_bias' if has_conv_bias else ''}"
        )
        (
            self._jax_fn_std,
            self._jax_fn_flat,
            self._jax_fn_flip,
            self._jax_fn_flip_ple,
            self._compiled_std,
            self._compiled_flat,
            self._compiled_flip,
            self._compiled_flip_with_ple,
        ) = _get_shared_gdn_op(
            kind,
            (
                id(mesh),
                id(vllm_config),
                num_layers,
                n_kq,
                local_num_v_heads,
                d_k,
                d_v,
                kernel_size,
                recurrent_state_dtype,
                conv_state_dtype,
                num_spec_tokens,
                conv_weight_kmajor,
                block_major,
                num_pools,
                pool_indices,
                has_conv_bias,
            ),
            build_ops,
        )
        self._weight_key: tuple[int, ...] | None = None
        self._cached_weight_stacks: tuple[torch.Tensor, ...] | None = None

    def _weight_stacks(self) -> tuple[torch.Tensor, ...]:
        # Stacked model weights are immutable during TPU serving; skip 3*L
        # `.data_ptr()` PJRT calls once cached on TPU device.
        if (
            self._cached_weight_stacks is not None
            and self.layers[0].A_log.device.type == "tpu"
        ):
            return self._cached_weight_stacks
        key_parts: list[int] = []
        cws: list[torch.Tensor] = []
        cbs: list[torch.Tensor] = []
        alogs: list[torch.Tensor] = []
        dtbs: list[torch.Tensor] = []
        for layer in self.layers:
            cw = layer._pooled_conv_weight()
            cws.append(cw)
            key_parts.append(int(cw.data_ptr()))
            key_parts.append(int(cw._version))
            if self._has_conv_bias:
                cb = layer.conv1d.bias
                assert cb is not None
                cbs.append(cb)
                key_parts.append(int(cb.data_ptr()))
                key_parts.append(int(cb._version))
            alogs.append(layer.A_log)
            dtbs.append(layer.dt_bias)
            key_parts.append(int(layer.A_log.data_ptr()))
            key_parts.append(int(layer.A_log._version))
            key_parts.append(int(layer.dt_bias.data_ptr()))
            key_parts.append(int(layer.dt_bias._version))
        key = tuple(key_parts)
        if self._cached_weight_stacks is not None and self._weight_key == key:
            return self._cached_weight_stacks
        if self._has_conv_bias:
            stacks: tuple[torch.Tensor, ...] = (
                torch.stack(cws, dim=0).contiguous(),
                torch.stack(cbs, dim=0).contiguous(),
                torch.stack(alogs, dim=0).contiguous(),
                torch.stack(dtbs, dim=0).contiguous(),
            )
        else:
            stacks = (
                torch.stack(cws, dim=0).contiguous(),
                torch.stack(alogs, dim=0).contiguous(),
                torch.stack(dtbs, dim=0).contiguous(),
            )
        self._weight_key = key
        self._cached_weight_stacks = stacks
        return stacks

    def can_replay_and_flip_fast(self) -> bool:
        """True when all layers share one backbone-owned verify flat buffer."""
        if self.layers[0]._shared_verify_flat_tensors is None:
            return False
        return all(
            layer._shared_verify_flat_tensors is not None for layer in self.layers
        )

    @torch.no_grad()
    def replay_and_flip(
        self,
        *,
        state_stack: torch.Tensor,
        ckpt_stack: torch.Tensor,
        query_start_loc: torch.Tensor,
        next_tokens: torch.Tensor,
        win_and_reqs: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_read_offsets: torch.Tensor,
        ple_state_stack: torch.Tensor | None = None,
        force_sp: bool | None = None,
    ) -> None:
        kv0 = self.layers[0].kv_cache
        if kv0 is None or len(kv0) != 1 or kv0[0].numel() == 0:
            return
        pool = kv0[0]
        flat0 = _active_verify_flat_tensors(self.layers[0], force_sp=force_sp)
        assert flat0 is not None
        qkv_flat, b_flat, a_flat = flat0
        weight_stacks = self._weight_stacks()
        if pool.device.type == "tpu":
            if ple_state_stack is not None:
                self._compiled_flip_with_ple(
                    pool,
                    qkv_flat,
                    b_flat,
                    a_flat,
                    *weight_stacks,
                    state_stack,
                    ckpt_stack,
                    query_start_loc,
                    next_tokens,
                    win_and_reqs,
                    distribution,
                    seq_lens,
                    slot_read_offsets,
                    ple_state_stack,
                )
            else:
                self._compiled_flip(
                    pool,
                    qkv_flat,
                    b_flat,
                    a_flat,
                    *weight_stacks,
                    state_stack,
                    ckpt_stack,
                    query_start_loc,
                    next_tokens,
                    win_and_reqs,
                    distribution,
                    seq_lens,
                    slot_read_offsets,
                )
            return

        # CPU fallback for unit tests (bit-exact with compiled TPU graph).
        qsl_i32, wr_i32, num_valid, cl_st = _compute_flip_commit_lens_and_valid(
            query_start_loc, next_tokens, win_and_reqs
        )
        j_weights = [_torch_to_jax_cpu(w) for w in weight_stacks]
        if ple_state_stack is not None:
            ple_off_1d = (num_valid - 1).clamp(min=0)
            new_rec, new_ro = self._jax_fn_flip_ple(
                _torch_to_jax_cpu(pool),
                _torch_to_jax_cpu(qkv_flat),
                _torch_to_jax_cpu(b_flat),
                _torch_to_jax_cpu(a_flat),
                *j_weights,
                _torch_to_jax_cpu(state_stack.to(torch.int32)),
                _torch_to_jax_cpu(qsl_i32),
                _torch_to_jax_cpu(distribution.to(torch.int32)),
                _torch_to_jax_cpu(seq_lens.to(torch.int32)),
                _torch_to_jax_cpu(slot_read_offsets),
                _torch_to_jax_cpu(ckpt_stack.to(torch.int32)),
                _torch_to_jax_cpu(cl_st),
                _torch_to_jax_cpu(wr_i32),
                _torch_to_jax_cpu(ple_state_stack.to(torch.int32)),
                _torch_to_jax_cpu(ple_off_1d),
            )
        else:
            new_rec, new_ro = self._jax_fn_flip(
                _torch_to_jax_cpu(pool),
                _torch_to_jax_cpu(qkv_flat),
                _torch_to_jax_cpu(b_flat),
                _torch_to_jax_cpu(a_flat),
                *j_weights,
                _torch_to_jax_cpu(state_stack.to(torch.int32)),
                _torch_to_jax_cpu(qsl_i32),
                _torch_to_jax_cpu(distribution.to(torch.int32)),
                _torch_to_jax_cpu(seq_lens.to(torch.int32)),
                _torch_to_jax_cpu(slot_read_offsets),
                _torch_to_jax_cpu(ckpt_stack.to(torch.int32)),
                _torch_to_jax_cpu(cl_st),
                _torch_to_jax_cpu(wr_i32),
            )
        _copy_jax_to_torch_cpu(pool, new_rec)
        _copy_jax_to_torch_cpu(slot_read_offsets, new_ro)

    @torch.no_grad()
    def __call__(
        self,
        *,
        raw_state_indices_stack: torch.Tensor,
        raw_ckpt_indices_stack: torch.Tensor,
        commit_lens_stack: torch.Tensor,
        query_start_loc: torch.Tensor,
        distribution: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_read_offsets: torch.Tensor,
        staged_inputs: (
            dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
        ) = None,
        force_sp: bool | None = None,
    ) -> None:
        kv0 = self.layers[0].kv_cache
        if kv0 is None or len(kv0) != 1 or kv0[0].numel() == 0:
            return
        pool = kv0[0]
        weight_stacks = self._weight_stacks()
        flat0 = _active_verify_flat_tensors(self.layers[0], force_sp=force_sp)
        if staged_inputs is None and flat0 is not None:
            qkv_flat, b_flat, a_flat = flat0
            if pool.device.type == "tpu":
                self._compiled_flat(
                    pool,
                    qkv_flat,
                    b_flat,
                    a_flat,
                    *weight_stacks,
                    raw_state_indices_stack.to(torch.int32),
                    query_start_loc.to(torch.int32),
                    distribution.to(torch.int32),
                    seq_lens.to(torch.int32),
                    slot_read_offsets,
                    raw_ckpt_indices_stack.to(torch.int32),
                    commit_lens_stack.to(torch.int32),
                )
                return
            new_rec, _ = self._jax_fn_flat(
                _torch_to_jax_cpu(pool),
                _torch_to_jax_cpu(qkv_flat),
                _torch_to_jax_cpu(b_flat),
                _torch_to_jax_cpu(a_flat),
                *[_torch_to_jax_cpu(w) for w in weight_stacks],
                _torch_to_jax_cpu(raw_state_indices_stack.to(torch.int32)),
                _torch_to_jax_cpu(query_start_loc.to(torch.int32)),
                _torch_to_jax_cpu(distribution.to(torch.int32)),
                _torch_to_jax_cpu(seq_lens.to(torch.int32)),
                _torch_to_jax_cpu(slot_read_offsets),
                _torch_to_jax_cpu(raw_ckpt_indices_stack.to(torch.int32)),
                _torch_to_jax_cpu(commit_lens_stack.to(torch.int32)),
            )
            _copy_jax_to_torch_cpu(pool, new_rec)
            return

        # Per-layer staged verify tensors stacked along dim=0: [L, win_rows, dim].
        qkvs, bs, as_ = [], [], []
        for layer in self.layers:
            item = (
                staged_inputs.get(layer.prefix)
                if staged_inputs is not None
                else layer.get_verify_inputs()
            )
            if item is None:
                return
            qkvs.append(item[0])
            bs.append(item[1])
            as_.append(item[2])
        qkv_3d = torch.stack(qkvs, dim=0).contiguous()
        b_3d = torch.stack(bs, dim=0).contiguous()
        a_3d = torch.stack(as_, dim=0).contiguous()
        if pool.device.type == "tpu":
            self._compiled_std(
                pool,
                qkv_3d,
                b_3d,
                a_3d,
                *weight_stacks,
                raw_state_indices_stack.to(torch.int32),
                query_start_loc.to(torch.int32),
                distribution.to(torch.int32),
                seq_lens.to(torch.int32),
                slot_read_offsets,
                raw_ckpt_indices_stack.to(torch.int32),
                commit_lens_stack.to(torch.int32),
            )
            return
        new_rec, _ = self._jax_fn_std(
            _torch_to_jax_cpu(pool),
            _torch_to_jax_cpu(qkv_3d),
            _torch_to_jax_cpu(b_3d),
            _torch_to_jax_cpu(a_3d),
            *[_torch_to_jax_cpu(w) for w in weight_stacks],
            _torch_to_jax_cpu(raw_state_indices_stack.to(torch.int32)),
            _torch_to_jax_cpu(query_start_loc.to(torch.int32)),
            _torch_to_jax_cpu(distribution.to(torch.int32)),
            _torch_to_jax_cpu(seq_lens.to(torch.int32)),
            _torch_to_jax_cpu(slot_read_offsets),
            _torch_to_jax_cpu(raw_ckpt_indices_stack.to(torch.int32)),
            _torch_to_jax_cpu(commit_lens_stack.to(torch.int32)),
        )
        _copy_jax_to_torch_cpu(pool, new_rec)


def build_multi_layer_pooled_replay_runner(
    layers: Sequence[Any],
) -> MultiLayerPooledReplayRunner | None:
    """Build a single-launch multi-layer replay runner when `layers` share one pool."""
    if not layers:
        return None
    gdn_layers = tuple(layers)
    l0 = gdn_layers[0]
    if not isinstance(l0, VllmGatedDeltaNetAttention):
        return None
    kv0 = l0.kv_cache
    if kv0 is None or len(kv0) != 1 or kv0[0].numel() == 0:
        return None
    pool0 = kv0[0]
    block_major = bool(l0._bm_kernel)
    has_bias = l0.conv1d.bias is not None
    bm_num_pools = (
        int(cast(Any, l0._bm_region)[1].num_pools)
        if (block_major and l0._bm_region is not None)
        else 0
    )
    for layer in gdn_layers:
        if not isinstance(layer, VllmGatedDeltaNetAttention):
            return None
        if not layer._two_slot_mtp:
            return None
        kv = layer.kv_cache
        if kv is None or len(kv) != 1 or kv[0] is not pool0:
            return None
        if (layer.conv1d.bias is not None) != has_bias:
            return None
        if (
            layer.num_v_heads != l0.num_v_heads
            or layer.num_k_heads != l0.num_k_heads
            or layer.head_k_dim != l0.head_k_dim
            or layer.head_v_dim != l0.head_v_dim
            or layer.conv_kernel_size != l0.conv_kernel_size
            or layer.num_spec != l0.num_spec
            or bool(layer._bm_kernel) != block_major
        ):
            return None
        if block_major and (
            layer._bm_region is None
            or int(cast(Any, layer._bm_region)[1].num_pools) != bm_num_pools
        ):
            return None
    return MultiLayerPooledReplayRunner(gdn_layers)
