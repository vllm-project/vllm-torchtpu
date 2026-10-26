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
import re
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import CompressorStateCache, DeepseekCompressor
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_torchtpu import envs
from vllm_torchtpu.core.hca_state_checkpoint import state_spec_cls
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config as compressor_config,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compressor_v1 import (
    compressor_forward,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compressor_v1 import (
    pair_proj as compressor_v1_pair_proj,
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
# Every compressor built in this process, so the op prebuild can pair a CSA
# layer's KV compressor with its indexer's (TPU_DSV4_COMPRESSOR_PAIR).
_COMPRESSORS: list = []

# The mesh has no batch axis today, so every batch-axis spec resolves to None.
# TODO(patemotter): name it here once the DP-attention PR adds the axis.
BATCH_AXIS = None


def _run_compressor(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    wkv_scale: jax.Array,
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
    seq_along_lane: bool = False,
    skip_proj: bool = False,
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
        seq_along_lane=seq_along_lane,
        skip_proj=skip_proj,
        proj_runs_min_tokens=envs.TPU_DSV4_COMPRESSOR_PROJ_RUNS_MIN_TOKENS,
        proj_runs_tile_n=envs.TPU_DSV4_COMPRESSOR_PROJ_RUNS_TILE_N,
        proj_runs_indexer=envs.TPU_DSV4_COMPRESSOR_PROJ_RUNS_INDEXER,
        prefill_block=envs.TPU_DSV4_COMPRESSOR_PREFILL_BLOCK,
        prefill_block_min_tokens=envs.TPU_DSV4_COMPRESSOR_PREFILL_BLOCK_MIN_TOKENS,
        proj_stream_max_tokens=envs.TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS,
        proj_stream_cfg=envs.TPU_DSV4_COMPRESSOR_PROJ_STREAM_CFG,
        proj_stream_min_width=envs.TPU_DSV4_COMPRESSOR_PROJ_STREAM_MIN_WIDTH,
        # bf16 weights carry a placeholder scale that is never read.
        wkv_scale=wkv_scale if wkv_wgate.dtype == jnp.float8_e4m3fn else None,
    )


# `pallas.jax_op` inspects the signature to split tensor from static args, so
# each cache arity needs its own module-level function rather than varargs.
def _compressor_jax_csa(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    wkv_scale: jax.Array,
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
    skip_proj: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """CSA: NoPE and RoPE in two arrays, state shares the NoPE buffer."""
    if cache.shape[0] == 0:
        return cache, rope_cache
    new_cache, new_rope_cache, _ = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        wkv_scale=wkv_scale,
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
        skip_proj=skip_proj,
    )
    # The CSA kernel always returns a rope cache.
    assert new_rope_cache is not None
    return new_cache, new_rope_cache


def _compressor_jax_hca(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    wkv_scale: jax.Array,
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
    skip_proj: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """HCA: no RoPE array, and the state lives on a CSA NoPE array."""
    if cache.shape[0] == 0 or state_cache.shape[0] == 0:
        return cache, state_cache
    new_cache, _, new_state_cache = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        wkv_scale=wkv_scale,
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
        skip_proj=skip_proj,
    )
    # The HCA kernel always returns a state cache.
    assert new_state_cache is not None
    return new_cache, new_state_cache


def _compressor_jax_indexer(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    wkv_scale: jax.Array,
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
    skip_proj: bool = False,
) -> jax.Array:
    """Lightning indexer: one array holds the SEQ_ALONG_LANE records and state."""
    if cache.shape[0] == 0:
        return cache
    new_cache, _, _ = _run_compressor(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        wkv_scale=wkv_scale,
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
        seq_along_lane=True,
        skip_proj=skip_proj,
    )
    return new_cache


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
        is as many lanes wide as its page holds tokens, HCA's state is hosted
        on a *CSA* page, and CSA and the indexer are floored to a shared value
        so vLLM groups them together. Ask the kernel's own layout model instead
        of guessing.
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
                indexer_seq_along_lane=True,
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
        spec_cls = state_spec_cls(
            self.compress_ratio,
            self.sliding_window,
            vllm_config.cache_config.block_size,
            envs.TPU_KV_DROP_HCA_STATE_CHECKPOINT,
        )
        return spec_cls(
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
        _COMPRESSORS.append(self)

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
        n = w_t.shape[1]
        if envs.TPU_DSV4_COMPRESSOR_FP8:
            # Per-output-channel e4m3: scale = amax_k |w| / 448 (0 -> 1).
            w_f = w_t.float()
            scale = w_f.abs().amax(dim=0) / 448.0
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            w_t = (w_f / scale[None, :]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        else:
            scale = torch.ones((n,), dtype=torch.float32, device=w_t.device)
        weight = torch.empty(w_t.shape, dtype=w_t.dtype, device=w_t.device).copy_(w_t)
        scale = torch.empty((n,), dtype=torch.float32, device=w_t.device).copy_(scale)
        self.fused_wkv_wgate.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("_wkv_scale", scale, persistent=False)
        self._wkv_wgate_transposed = True
        if weight.device.type == "tpu":
            synchronize_tensors([weight, scale])

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
            if envs.TPU_DSV4_COMPRESSOR_PAIR and not self._separate_state:
                # The runner's op prebuild reaches this before compile: build
                # the TPU_DSV4_COMPRESSOR_PAIR ops now, since building a
                # jax_op inside the compiled graph is not traceable.
                _ = self.compressor_op_skip_proj
                partner = _pair_partner(self)
                if partner is not None:
                    a, b = (self, partner) if self.head_dim == 512 else (partner, self)
                    _get_pair_op([_pair_statics(a), _pair_statics(b)])
        return op

    @property
    def compressor_op_skip_proj(self):
        """`compressor_op` without the projection (its state saved by
        `pair_proj_forward`)."""
        op = self.__dict__.get("_compressor_op_skip_instance")
        if op is None:
            op = self._build_compressor_op(skip_proj=True)
            object.__setattr__(self, "_compressor_op_skip_instance", op)
        return op

    def _build_compressor_op(self, skip_proj: bool = False):
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
            skip_proj=skip_proj,
        )

        op_name = (
            f"pallas::deepseek_v4_compressor_{variant}"
            f"_{self.head_dim}_{self.compress_ratio}"
            f"_{self.state_cache.block_size}"
            f"{'_noproj' if skip_proj else ''}"
        )

        global _compressor_op_cache
        if op_name in _compressor_op_cache:
            return _compressor_op_cache[op_name]

        attn_data_axis = None
        batch_axis = BATCH_AXIS
        input_partition_specs = (
            P(attn_data_axis, None),  # hidden_states
            P(),  # wkv_wgate
            P(),  # wkv_scale
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
                wkv_scale,
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
                wkv_scale,
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

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ):
        """The op operands, or None when there is nothing to compress."""
        attn_ctx = get_forward_context().attn_metadata
        if not isinstance(attn_ctx, dict):
            # As in the base class: without per-layer metadata there is
            # nothing to compress.
            return

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
            return

        # For CSA the k_cache layer binds a `(nope, rope)` pair.
        entry = getattr(self.k_cache, "kv_cache", None)
        main, rope = entry if isinstance(entry, tuple) else (entry, None)
        cache = self._as_kernel_cache_view(main)
        if cache is None:
            return

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
            self._wkv_scale,
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
        return operands, cache, rope_cache, state_cache, state_metadata, k_block_tables

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        skip_proj: bool = False,
    ) -> None:
        # skip_proj: `pair_proj_forward` already saved this step's state.
        skip = skip_proj
        prep = self._prepare(hidden_states, positions, rotary_emb)
        if prep is None:
            return
        operands, cache, rope_cache, state_cache, _, _ = prep
        op = self.compressor_op_skip_proj if skip else self.compressor_op
        if rope_cache is not None:
            new_cache, new_rope_cache = op(*operands, rope_cache)
            cache.copy_(new_cache)
            rope_cache.copy_(new_rope_cache)
        elif state_cache is not None:
            new_cache, new_state_cache = op(*operands, state_cache)
            cache.copy_(new_cache)
            state_cache.copy_(new_state_cache)
        else:
            cache.copy_(op(*operands))


_pair_op_cache: dict = {}


def _pair_proj_jax(
    hidden_states: jax.Array,
    positions: jax.Array,
    cos_sin_a: jax.Array,
    cos_sin_b: jax.Array,
    w_a: jax.Array,
    ws_a: jax.Array,
    ape_a: jax.Array,
    st_a: jax.Array,
    qsl_a: jax.Array,
    kvt_a: jax.Array,
    dist_a: jax.Array,
    host_a: jax.Array,
    w_b: jax.Array,
    ws_b: jax.Array,
    ape_b: jax.Array,
    st_b: jax.Array,
    qsl_b: jax.Array,
    kvt_b: jax.Array,
    dist_b: jax.Array,
    host_b: jax.Array,
    *,
    statics_a: tuple,
    statics_b: tuple,
) -> tuple[jax.Array, jax.Array]:
    statics_a, statics_b = dict(statics_a), dict(statics_b)

    def grp(w, ws, ape, st, qsl, kvt, dist, host, cos_sin, statics):
        separate = statics["separate_state"]
        return dict(
            wkv_wgate=w,
            wkv_scale=ws if w.dtype == jnp.float8_e4m3fn else None,
            ape=ape,
            block_table=st,
            query_start_loc=qsl,
            kv_block_table=kvt,
            # derive_metadata reads only the host's geometry; the KV array is
            # not needed when the state is separate (HCA, not paired).
            cache=host,
            state_cache=host if separate else None,
            distribution=dist,
            cos_sin_cache=cos_sin,
            **{k: v for k, v in statics.items() if k != "separate_state"},
        )
    return compressor_v1_pair_proj(
        hidden_states, positions,
        [grp(w_a, ws_a, ape_a, st_a, qsl_a, kvt_a, dist_a, host_a, cos_sin_a, statics_a),
         grp(w_b, ws_b, ape_b, st_b, qsl_b, kvt_b, dist_b, host_b, cos_sin_b, statics_b)],
        stream_cfg=envs.TPU_DSV4_COMPRESSOR_PROJ_STREAM_CFG,
    )


def _layer_of(comp) -> tuple[str, bool] | None:
    m = re.search(r"layers\.(\d+)\.", comp.state_cache.prefix or "")
    if m is None:
        return None
    return m.group(1), "indexer" in comp.state_cache.prefix


def _pair_partner(comp):
    """The other compressor of the same CSA layer (KV <-> indexer), if any."""
    me = _layer_of(comp)
    if me is None:
        return None
    for other in _COMPRESSORS:
        if other is comp or other._separate_state:
            continue
        them = _layer_of(other)
        if them is not None and them[0] == me[0] and them[1] != me[1]:
            return other
    return None


def _pair_statics(comp) -> dict:
    return dict(
        compress_ratio=comp.compress_ratio,
        state_block_size=comp.state_cache.block_size,
        head_dim=comp.head_dim,
        overlap=comp.overlap,
        seq_along_lane=comp.head_dim == 128,
        separate_state=False,
    )


def _get_pair_op(statics):
    key = tuple(tuple(sorted(st.items())) for st in statics)  # == pair_proj_forward's
    op = _pair_op_cache.get(key)
    if op is None:
        name = "pallas::deepseek_v4_compressor_pair_proj_" + "_".join(
            f"{st['head_dim']}x{st['compress_ratio']}x{st['state_block_size']}"
            for st in statics
        )
        grp_specs = (P(), P(), P(), P(BATCH_AXIS), P(), P(BATCH_AXIS), P(), P())
        op = pallas.jax_op(
            name,
            functools.partial(
                _pair_proj_jax,
                statics_a=tuple(sorted(statics[0].items())),
                statics_b=tuple(sorted(statics[1].items())),
            ),
            mesh=get_pallas_op_mesh(),
            donate_argnums=(11, 19),
            input_partition_specs=(P(None, None), P(None), P(), P())
            + grp_specs
            + grp_specs,
        )

        def _fake(*a, **k):
            return torch.empty_like(a[11]), torch.empty_like(a[19])

        op.register_fake(_fake)
        _pair_op_cache[key] = op
    return op


def pair_proj_forward(
    comp_a: "VllmDeepseekCompressor",
    comp_b: "VllmDeepseekCompressor",
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    rotary_a: nn.Module,
    rotary_b: nn.Module,
) -> bool:
    """Saves the state of two compressors of the same tokens (a CSA layer's KV
    compressor and its indexer's) with one weight-stream call. Returns True
    when it did -- the caller then runs both compressors with skip_proj=True --
    and False when this step does not qualify (nothing written). The decision
    depends only on statics and the token count, so it is trace-stable.
    """
    preps = []
    for comp, rot in ((comp_a, rotary_a), (comp_b, rotary_b)):
        if comp._separate_state:
            return False
        prep = comp._prepare(hidden_states, positions, rot)
        if prep is None:
            return False
        preps.append((comp, prep))
    # Look up only: the ops are prebuilt with the compressors' own (see
    # `compressor_op`); building one here would happen inside the compiled graph.
    key = tuple(
        tuple(sorted(_pair_statics(c).items())) for c in (comp_a, comp_b)
    )
    op = _pair_op_cache.get(key)
    if op is None or any(
        c.__dict__.get("_compressor_op_skip_instance") is None for c in (comp_a, comp_b)
    ):
        return False
    n = hidden_states.shape[0]
    if not (0 < n <= envs.TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS):
        return False
    args = [hidden_states, positions]
    cos = []
    for comp, (operands, cache, _, _, _, _) in preps:
        (_, w, ws, ape, _, cos_sin, _, st, qsl, kvt, dist, _) = operands
        cos.append(cos_sin)
        args += [w, ws, ape, st, qsl, kvt, dist, cache]
    args = args[:2] + cos + args[2:]
    new_a, new_b = op(*args)
    preps[0][1][1].copy_(new_a)
    preps[1][1][1].copy_(new_b)
    return True
