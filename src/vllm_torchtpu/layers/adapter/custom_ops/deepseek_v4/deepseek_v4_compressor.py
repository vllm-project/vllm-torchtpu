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
"""TPU DeepSeek-V4 KV/score compressor, on the compress-and-store kernels."""

import functools
from typing import TYPE_CHECKING, cast

import jax
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import CompressorStateCache, DeepseekCompressor
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowMLASpec

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config as compressor_config,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compressor_v1 import (
    CompressorInputs,
    compressor_forward,
    compressors_forward,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)
from vllm_torchtpu.utils import align_to, synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

logger = init_logger(__name__)

_compressor_op_cache = {}

# The mesh has no batch axis today, so every batch-axis spec resolves to None.
# TODO(patemotter): name it here once the DP-attention PR adds the axis.
BATCH_AXIS = None


def _run_compressor(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    state_block_tables: jax.Array,
    query_start_loc: jax.Array,
    k_block_tables: jax.Array,
    request_distribution: jax.Array,
    cache: jax.Array,
    rope_cache: jax.Array | None,
    state_cache: jax.Array | None,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Shared body of the three op variants, for CSA compressor, HCA compressor,
    and CSA indexer compressor respectively."""
    return compressor_forward(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        block_table=state_block_tables,
        query_start_loc=query_start_loc,
        kv_block_table=k_block_tables,
        cache=cache,
        rope_cache=rope_cache,
        state_cache=state_cache,
        distribution=request_distribution,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )


# `pallas.jax_op` inspects the signature to split tensor from static args, so
# each cache arity needs its own module-level function rather than varargs.
def _compressor_jax_csa(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    state_block_tables: jax.Array,
    query_start_loc: jax.Array,
    k_block_tables: jax.Array,
    request_distribution: jax.Array,
    cache: jax.Array,
    rope_cache: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array]:
    """CSA: NoPE and RoPE in two arrays, state shares the NoPE buffer."""
    if cache.shape[0] == 0:
        return cache, rope_cache
    new_cache, new_rope_cache, _ = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        state_block_tables=state_block_tables,
        query_start_loc=query_start_loc,
        k_block_tables=k_block_tables,
        request_distribution=request_distribution,
        cache=cache,
        rope_cache=rope_cache,
        state_cache=None,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )
    # The CSA kernel always returns a rope cache.
    assert new_rope_cache is not None
    return new_cache, new_rope_cache


def _compressor_jax_hca(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    state_block_tables: jax.Array,
    query_start_loc: jax.Array,
    k_block_tables: jax.Array,
    request_distribution: jax.Array,
    cache: jax.Array,
    state_cache: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> tuple[jax.Array, jax.Array]:
    """HCA: no RoPE array, and the state lives on a CSA NoPE array."""
    if cache.shape[0] == 0 or state_cache.shape[0] == 0:
        return cache, state_cache
    new_cache, _, new_state_cache = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        state_block_tables=state_block_tables,
        query_start_loc=query_start_loc,
        k_block_tables=k_block_tables,
        request_distribution=request_distribution,
        cache=cache,
        rope_cache=None,
        state_cache=state_cache,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )
    # The HCA kernel always returns a state cache.
    assert new_state_cache is not None
    return new_cache, new_state_cache


def _compressor_jax_indexer(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    state_block_tables: jax.Array,
    query_start_loc: jax.Array,
    k_block_tables: jax.Array,
    request_distribution: jax.Array,
    cache: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
) -> jax.Array:
    """Lightning indexer: one array holds the records and the state."""
    if cache.shape[0] == 0:
        return cache
    new_cache, _, _ = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        state_block_tables=state_block_tables,
        query_start_loc=query_start_loc,
        k_block_tables=k_block_tables,
        request_distribution=request_distribution,
        cache=cache,
        rope_cache=None,
        state_cache=None,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )
    return new_cache


def _compressor_jax_csa_with_indexer(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    state_block_tables: jax.Array,
    query_start_loc: jax.Array,
    k_block_tables: jax.Array,
    request_distribution: jax.Array,
    cache: jax.Array,
    rope_cache: jax.Array,
    indexer_wkv_wgate: jax.Array,
    indexer_ape: jax.Array,
    indexer_norm_weight: jax.Array,
    indexer_cos_sin_cache: jax.Array,
    indexer_state_block_tables: jax.Array,
    indexer_query_start_loc: jax.Array,
    indexer_k_block_tables: jax.Array,
    indexer_request_distribution: jax.Array,
    indexer_cache: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    indexer_state_block_size: int,
    indexer_head_dim: int,
    indexer_compress_ratio: int,
    indexer_overlap: bool,
    indexer_rms_eps: float,
    indexer_quant_block: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A CSA compressor and its layer's indexer compressor, which compress
    the same hidden states: `_compressor_jax_csa` and
    `_compressor_jax_indexer` in one op, whose states one kernel projects."""
    if cache.shape[0] == 0 and indexer_cache.shape[0] == 0:
        return cache, rope_cache, indexer_cache
    if cache.shape[0] == 0 or indexer_cache.shape[0] == 0:
        raise ValueError(
            "DeepSeek-V4 CSA and indexer compressors traced with one cache "
            f"empty: CSA {cache.shape}, indexer {indexer_cache.shape}"
        )
    csa = CompressorInputs(
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        block_table=state_block_tables,
        query_start_loc=query_start_loc,
        kv_block_table=k_block_tables,
        cache=cache,
        rope_cache=rope_cache,
        distribution=request_distribution,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
    )
    indexer = CompressorInputs(
        wkv_wgate=indexer_wkv_wgate,
        ape=indexer_ape,
        norm_weight=indexer_norm_weight,
        cos_sin_cache=indexer_cos_sin_cache,
        block_table=indexer_state_block_tables,
        query_start_loc=indexer_query_start_loc,
        kv_block_table=indexer_k_block_tables,
        cache=indexer_cache,
        rope_cache=None,
        distribution=indexer_request_distribution,
        state_block_size=indexer_state_block_size,
        head_dim=indexer_head_dim,
        compress_ratio=indexer_compress_ratio,
        overlap=indexer_overlap,
        rms_eps=indexer_rms_eps,
        quant_block=indexer_quant_block,
    )
    # The indexer's weight is the smaller, so it goes first (see
    # `compressors_forward`).
    (new_indexer_cache, _, _), (new_cache, new_rope_cache, _) = compressors_forward(
        hidden_states, positions, (indexer, csa)
    )
    assert new_rope_cache is not None
    return new_cache, new_rope_cache, new_indexer_cache


class VllmCompressorStateCache(CompressorStateCache):
    """Compressor state cache, paged to fit the array that hosts it."""

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__(
            state_dim,
            dtype,
            compress_ratio,
            prefix,
        )
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff
        self.compress_ratio = compress_ratio
        # `block_size` deliberately keeps the base class's value here.
        # It cannot be derived yet: `cache_config.block_size` is still vLLM's
        # small default at construction time -- the platform's DSv4 override
        # is not visible on any config reachable from a submodule constructor
        # -- and at that size every mode's state page floors to zero rows.
        # `get_kv_cache_spec` recomputes it from the finalized config, which
        # runs before `_build_compressor_op` bakes it into the kernel.

    def _derive_block_size(self, kv_cache_block_size: int) -> int:
        """State tokens per page of the array that hosts this state.

        `block_size` is the granularity of the block table vLLM builds and
        that the compressor kernel indexes with, so it must be the number of
        token states that physically fit in one page of the *hosting* array.
        CSA's and the indexer's state caches overlay their own compressed-KV
        array; HCA's overlays a *CSA NoPE* array, so it is the hosting array's
        page that sets the count, not this mode's own.

        Not a closed form: HCA writes two rows per record, the indexer array
        is 256 lanes wide, HCA's state is hosted on a *CSA* page, and CSA and
        the indexer are floored to a shared value so vLLM groups them
        together. Ask the kernel's own layout model instead of guessing.
        """
        try:
            if kv_cache_block_size // self.compress_ratio <= 0:
                raise ValueError(
                    f"a page of {kv_cache_block_size} tokens holds no "
                    f"compressed row at compress_ratio {self.compress_ratio}"
                )
            mode = compressor_config.select_mode(
                self.head_dim, self.compress_ratio == 4
            )
            block_size = compressor_config.state_block_size(
                mode,
                kv_cache_block_size,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            )
            if block_size <= 0:
                raise ValueError(
                    f"{mode.value} state rows do not fit in a page derived "
                    f"from a {kv_cache_block_size}-token KV block"
                )
        except (ValueError, AssertionError, ZeroDivisionError) as exc:
            raise ValueError(
                f"DeepSeek-V4 compressor state cache {self.prefix!r} cannot "
                f"be paged for cache block size {kv_cache_block_size} "
                f"(head_dim {self.head_dim}, compress_ratio "
                f"{self.compress_ratio}): {exc}"
            ) from exc
        return block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # `_build_compressor_op` bakes `block_size` in as the kernel's
        # `state_block_size`, so spec and kernel must not disagree.
        self.block_size = self._derive_block_size(vllm_config.cache_config.block_size)
        # uint8 is deliberate: the kernel writes raw f32 state bytes, and
        # declaring the real dtype would make the byte budget disagree with
        # the packed layout the host array actually uses. The physical overlay
        # onto that host is done by the runner.
        bytes_per_row = self.state_dim * 4
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(bytes_per_row, 128),
            dtype=torch.uint8,
            sliding_window=self.sliding_window,
            alignment=None,
        )


class VllmDeepseekCompressor(DeepseekCompressor):
    """TPU compressor: projects, saves state, compresses and stores."""

    def __init__(self, *args, **kwargs) -> None:
        orig_state_cache = dsv4_compressor.CompressorStateCache
        self._wkv_wgate_transposed = False
        dsv4_compressor.CompressorStateCache = VllmCompressorStateCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_compressor.CompressorStateCache = orig_state_cache
        # The MXFP4 record layout the base class selects for the indexer has no
        # TPU kernel: ours always emits the FP8/UE8M0 layout, so honoring the
        # flag would write records the indexer reads as MXFP4.
        if self.use_fp4_cache:
            raise NotImplementedError(
                "DeepSeek-V4 on TPU does not support "
                "`attention_config.use_fp4_indexer_cache`; the compressor "
                "kernel only emits the FP8/UE8M0 cache layout."
            )
        self.num_layers = (
            get_current_vllm_config().model_config.hf_config.num_hidden_layers
        )

    def transpose_wkv_wgate(self) -> None:
        """Stores ``fused_wkv_wgate.weight`` transposed, once, at load time.

        vLLM lays linear weights out as [out_features, in_features], but the
        compress-and-store kernel consumes them as
        [hidden_size, 2 * coff * head_dim]. Transposing in the forward pass
        makes XLA materialize the transposed copy on every step; the weight is
        only ever read by that kernel, so it can just be stored that way.
        """
        if self._wkv_wgate_transposed:
            return
        w_t = cast(torch.Tensor, self.fused_wkv_wgate.weight).data.t().contiguous()
        # torch_tpu can keep `w_t` as a view chain on the old parameter and
        # re-materialize it at the `pallas.jax_op` boundary every step
        # (`tt_jit_as_strided`). A fresh buffer + copy_ breaks the chain; same
        # pattern as `_fresh()` in mla_attention_op.py and the flip in fp8.py.
        weight = torch.empty(w_t.shape, dtype=w_t.dtype, device=w_t.device).copy_(w_t)
        self.fused_wkv_wgate.weight = torch.nn.Parameter(weight, requires_grad=False)
        self._wkv_wgate_transposed = True
        if weight.device.type == "tpu":
            synchronize_tensors([weight])

    # head_dim == 512 with overlap: CSA, which splits NoPE and RoPE.
    # head_dim == 512 without:      HCA, whose state lives on a CSA array.
    # head_dim == 128:              lightning indexer, one array for both.
    @property
    def _has_rope_cache(self) -> bool:
        return self.head_dim == 512 and self.overlap

    @property
    def _separate_state(self) -> bool:
        return not self.overlap

    @property
    def compressor_op(self):
        """Built on first use, since `state_block_size` is baked in statically.

        Until `get_kv_cache_spec` runs during KV cache init, that attribute
        holds the base class's CUDA-derived value rather than what this host
        page can take, and the op would trace against the wrong geometry.
        """
        op = self.__dict__.get("_compressor_op_instance")
        if op is None:
            op = self._build_compressor_op()
            object.__setattr__(self, "_compressor_op_instance", op)
        return op

    def _build_compressor_op(self):
        mesh = get_pallas_op_mesh()

        assert self.head_dim in (512, 128), self.head_dim
        if self._has_rope_cache:
            variant, jax_fn = "csa", _compressor_jax_csa
        elif self._separate_state:
            variant, jax_fn = "hca", _compressor_jax_hca
        else:
            variant, jax_fn = "indexer", _compressor_jax_indexer

        wrapped_fn = functools.partial(
            jax_fn,
            state_block_size=self.state_cache.block_size,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
        )

        op_name = (
            f"pallas::deepseek_v4_compressor_{variant}"
            f"_{self.head_dim}_{self.compress_ratio}"
            f"_{self.state_cache.block_size}"
        )

        global _compressor_op_cache
        if op_name in _compressor_op_cache:
            return _compressor_op_cache[op_name]

        attn_data_axis = None
        batch_axis = BATCH_AXIS
        input_partition_specs = (
            P(attn_data_axis, None),  # hidden_states
            P(),  # wkv_wgate
            P(),  # ape
            P(),  # norm_weight
            P(),  # cos_sin_cache
            P(attn_data_axis),  # positions
            P(batch_axis),  # state_block_tables
            P(),  # query_start_loc
            P(batch_axis),  # k_block_tables
            P(),  # request_distribution
            P(),  # cache
        )
        donate_argnums = (len(input_partition_specs) - 1,)  # cache
        if variant != "indexer":
            input_partition_specs += (P(),)  # rope_cache / state_cache
            donate_argnums += (len(input_partition_specs) - 1,)

        compressor_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            donate_argnums=donate_argnums,
            input_partition_specs=input_partition_specs,
        )

        if variant == "indexer":

            def _fake_compressor(
                hidden_states,
                wkv_wgate,
                ape,
                norm_weight,
                cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                *args,
                **kwargs,
            ):
                return torch.empty_like(cache)
        else:

            def _fake_compressor(
                hidden_states,
                wkv_wgate,
                ape,
                norm_weight,
                cos_sin_cache,
                positions,
                state_block_tables,
                query_start_loc,
                k_block_tables,
                request_distribution,
                cache,
                second_cache,
                *args,
                **kwargs,
            ):
                return torch.empty_like(cache), torch.empty_like(second_cache)

        compressor_jax_op.register_fake(_fake_compressor)

        _compressor_op_cache[op_name] = compressor_jax_op
        return compressor_jax_op

    @staticmethod
    def _as_kernel_cache_view(cache: torch.Tensor | None) -> torch.Tensor | None:
        """The kernels require uint8; vLLM allocates as `kv_cache_dtype`.

        Every candidate dtype is one byte wide, so `.view` is a lossless
        bitcast that shares storage -- assigning would convert instead, and
        the writes would not land in the real cache. The int32 CSA arrays (see
        `csa_cache_layout`) are already what the kernels take.
        """
        if cache is None or cache.numel() == 0:
            return None
        if cache.dtype in (torch.uint8, torch.int32):
            return cache
        return cache.view(torch.uint8)

    def _kernel_operands(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> (
        tuple[tuple[torch.Tensor, ...], torch.Tensor | None, torch.Tensor | None] | None
    ):
        """The op's operands up to its cache, then CSA's RoPE array and HCA's
        state array (each None for the other modes); None when there is
        nothing to compress."""
        attn_ctx = get_forward_context().attn_metadata
        if not isinstance(attn_ctx, dict):
            # As in the base class: without per-layer metadata there is
            # nothing to compress.
            return None

        # Each cache must use its own layer's metadata: they are separate
        # groups with separate block tables, so borrowing another layer's
        # would scatter rows into that layer's pages.
        state_prefix = self.state_cache.prefix
        _k_cache_obj = getattr(self, "k_cache", None)
        k_cache_prefix = getattr(
            _k_cache_obj, "prefix", getattr(_k_cache_obj, "custom_prefix", None)
        ) or getattr(self, "k_cache_prefix", None)
        for _name, _key in (("state_cache", state_prefix), ("k_cache", k_cache_prefix)):
            if _key not in attn_ctx:
                raise KeyError(
                    f"DeepSeek-V4 compressor {_name} prefix {_key!r} has no "
                    "attention metadata; using another layer's would corrupt "
                    f"its pages. Known: {sorted(attn_ctx)}"
                )
        assert k_cache_prefix is not None
        state_metadata = cast("AttentionMetadata | None", attn_ctx[state_prefix])
        k_cache_metadata = cast("AttentionMetadata | None", attn_ctx[k_cache_prefix])
        if state_metadata is None or k_cache_metadata is None:
            return None

        # For CSA the k_cache layer binds a `(nope, rope)` pair.
        entry = getattr(self.k_cache, "kv_cache", None)
        main, rope = entry if isinstance(entry, tuple) else (entry, None)
        cache = self._as_kernel_cache_view(main)
        if cache is None:
            return None

        state_block_tables = state_metadata.block_tables
        k_block_tables = k_cache_metadata.block_tables

        rope_cache = None
        state_cache = None
        if self._has_rope_cache:
            rope_cache = self._as_kernel_cache_view(rope)
            if rope_cache is None:
                raise RuntimeError(
                    f"DeepSeek-V4 CSA compressor {state_prefix!r} has no "
                    "companion RoPE KV array; the kernel writes the RoPE "
                    "channels there, and without it they would be dropped. "
                    "Expected the runner to have bound a (nope, rope) pair "
                    f"onto layer {k_cache_prefix!r}."
                )
        if self._separate_state:
            state_cache = self._as_kernel_cache_view(
                getattr(self.state_cache, "kv_cache", None)
            )
            if state_cache is None:
                # `cache` is already real by here, so this is not the
                # profiling pass: skipping the compressor would silently
                # leave every later token's state unwritten.
                raise RuntimeError(
                    f"DeepSeek-V4 HCA compressor {state_prefix!r} has a live "
                    "compressed-KV array but no state array; the kernel needs "
                    "a separate buffer for the f32 state, an HCA page being "
                    "far too small to host it. Expected the runner to have "
                    "overlaid this state cache onto a CSA NoPE array."
                )

        assert self._wkv_wgate_transposed, (
            "fused_wkv_wgate must be transposed at load time; the model's "
            "load_weights is expected to call transpose_wkv_wgate(). It has "
            "quant_config=None, so it never reaches a TPU linear method and "
            "the canonical (k, n) flip does not apply to it."
        )
        operands = (
            hidden_states,
            self.fused_wkv_wgate.weight,
            self.ape,
            self.norm.weight,
            rotary_emb.cos_sin_cache,
            positions,
            state_block_tables,
            state_metadata.query_start_loc,
            k_block_tables,
            state_metadata.request_distribution,
            cache,
        )
        return operands, rope_cache, state_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        kernel_operands = self._kernel_operands(hidden_states, positions, rotary_emb)
        if kernel_operands is None:
            return
        operands, rope_cache, state_cache = kernel_operands
        cache = operands[-1]
        if rope_cache is not None:
            new_cache, new_rope_cache = self.compressor_op(*operands, rope_cache)
            cache.copy_(new_cache)
            rope_cache.copy_(new_rope_cache)
        elif state_cache is not None:
            new_cache, new_state_cache = self.compressor_op(*operands, state_cache)
            cache.copy_(new_cache)
            state_cache.copy_(new_state_cache)
        else:
            cache.copy_(self.compressor_op(*operands))

    def forward_with_indexer(
        self,
        indexer_compressor: "VllmDeepseekCompressor",
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        indexer_rotary_emb: nn.Module,
    ) -> None:
        """This CSA compressor's `forward` and its layer's indexer
        compressor's, in one op: both project the same hidden states, which
        one kernel then reads once (`_compressor_jax_csa_with_indexer`)."""
        assert self._has_rope_cache, "only a CSA compressor pairs with an indexer"
        assert not (
            indexer_compressor._has_rope_cache or indexer_compressor._separate_state
        ), "the paired compressor must be a lightning indexer's"
        csa = self._kernel_operands(hidden_states, positions, rotary_emb)
        indexer = indexer_compressor._kernel_operands(
            hidden_states, positions, indexer_rotary_emb
        )
        if csa is None and indexer is None:
            return
        if csa is None or indexer is None:
            raise RuntimeError(
                f"DeepSeek-V4 CSA compressor {self.state_cache.prefix!r} and "
                f"indexer compressor {indexer_compressor.state_cache.prefix!r} "
                "must both have live caches or neither; one would be skipped."
            )
        operands, rope_cache, _ = csa
        indexer_operands, _, _ = indexer
        assert rope_cache is not None
        cache, indexer_cache = operands[-1], indexer_operands[-1]
        # The indexer's operands but the hidden states and positions it shares.
        new_cache, new_rope_cache, new_indexer_cache = self.paired_op(
            indexer_compressor
        )(
            *operands,
            rope_cache,
            *indexer_operands[1:5],
            *indexer_operands[6:],
        )
        cache.copy_(new_cache)
        rope_cache.copy_(new_rope_cache)
        indexer_cache.copy_(new_indexer_cache)

    def paired_op(self, indexer_compressor: "VllmDeepseekCompressor"):
        """The `forward_with_indexer` op, kept on this compressor since a CSA
        compressor always pairs with its own layer's indexer. Constructing it
        builds PartitionSpecs, which Dynamo cannot trace, so the runner
        prebuilds it through `build_paired_op`."""
        op = self.__dict__.get("_paired_op_instance")
        if op is None:
            op = self.build_paired_op(indexer_compressor)
        return op

    def build_paired_op(self, indexer_compressor: "VllmDeepseekCompressor"):
        """(Re)builds `paired_op`, which bakes in both state block sizes: the
        runner calls it again once they are final, as it does for
        `compressor_op`."""
        op = self._build_paired_op(indexer_compressor)
        object.__setattr__(self, "_paired_op_instance", op)
        return op

    def _build_paired_op(self, indexer_compressor: "VllmDeepseekCompressor"):
        mesh = get_pallas_op_mesh()
        idx = indexer_compressor
        wrapped_fn = functools.partial(
            _compressor_jax_csa_with_indexer,
            state_block_size=self.state_cache.block_size,
            head_dim=self.head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            rms_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            indexer_state_block_size=idx.state_cache.block_size,
            indexer_head_dim=idx.head_dim,
            indexer_compress_ratio=idx.compress_ratio,
            indexer_overlap=idx.overlap,
            indexer_rms_eps=idx.rms_norm_eps,
            indexer_quant_block=idx._quant_block,
        )
        op_name = (
            f"pallas::deepseek_v4_compressor_csa_with_indexer"
            f"_{self.head_dim}_{self.compress_ratio}_{self.state_cache.block_size}"
            f"_{idx.head_dim}_{idx.compress_ratio}_{idx.state_cache.block_size}"
        )

        global _compressor_op_cache
        if op_name in _compressor_op_cache:
            return _compressor_op_cache[op_name]

        attn_data_axis = None
        batch_axis = BATCH_AXIS
        per_compressor_specs = (
            P(),  # wkv_wgate
            P(),  # ape
            P(),  # norm_weight
            P(),  # cos_sin_cache
        )
        tables_specs = (
            P(batch_axis),  # state_block_tables
            P(),  # query_start_loc
            P(batch_axis),  # k_block_tables
            P(),  # request_distribution
        )
        input_partition_specs = (
            P(attn_data_axis, None),  # hidden_states
            *per_compressor_specs,
            P(attn_data_axis),  # positions
            *tables_specs,
            P(),  # cache
            P(),  # rope_cache
            *per_compressor_specs,  # the indexer's
            *tables_specs,  # the indexer's
            P(),  # indexer cache
        )
        num_csa_operands = 12
        donate_argnums = (
            num_csa_operands - 2,  # cache
            num_csa_operands - 1,  # rope_cache
            len(input_partition_specs) - 1,  # indexer cache
        )
        paired_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            donate_argnums=donate_argnums,
            input_partition_specs=input_partition_specs,
        )

        def _fake_paired(*args, **kwargs):
            cache, rope_cache, indexer_cache = (
                args[num_csa_operands - 2],
                args[num_csa_operands - 1],
                args[len(input_partition_specs) - 1],
            )
            return (
                torch.empty_like(cache),
                torch.empty_like(rope_cache),
                torch.empty_like(indexer_cache),
            )

        paired_jax_op.register_fake(_fake_paired)

        _compressor_op_cache[op_name] = paired_jax_op
        return paired_jax_op
