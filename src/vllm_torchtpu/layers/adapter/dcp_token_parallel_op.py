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
"""Token-parallel DCP prefill custom ops for Sparse MLA and DSA indexer top-k."""

import dataclasses
import functools
import inspect
import weakref
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import (
    dcp_lane_operand,
    dcp_rank_operand,
    get_or_create_dcp_mesh,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    kv_cache_append,
    streamindex_topk,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    kv_cache_utils as idx_kv_cache_utils,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel, page_gather
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    update_sparse_mla_kv_cache_dcp,
    update_sparse_mla_kv_cache_dcp_sharded,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    DCP_TP_ZIGZAG_BANDS,
    MLA_MIXED_LIVE_WIDTHS,
    STREAMIDX_DEFAULT_MIXED_KV_PAGES,
    bucket_gate_edge,
    get_shared_jax_op,
    glm_dsa_v7x_active,
    is_tpu_v7x,
)

logger = init_logger(__name__)

cp_mla_attention: Any
try:
    from vllm_torchtpu.layers.adapter import cp_mla_attention
except ModuleNotFoundError:
    cp_mla_attention = None

vllm_compilation_backends: Any
try:
    from vllm.compilation import backends as vllm_compilation_backends
except ImportError:
    vllm_compilation_backends = None

MIN_TOKENS = 1024
MAX_SUPPORTED_SEQS = 256
EXTRA_PAGES = 4
SPLIT_ROWS_ALIGN = 16
ATTEND_CHUNK_TOKENS = 256


def attend_chunk_kwargs(env_chunk_kwargs: dict[str, int]) -> dict[str, int]:
    """Return ``env_chunk_kwargs`` when set, else 256-row prefill chunks."""
    if env_chunk_kwargs:
        return env_chunk_kwargs
    return {"gather_and_attention_chunk_size": ATTEND_CHUNK_TOKENS}


def serves_sequence_row_oproj(o_proj: Any) -> bool:
    """True when ``o_proj`` is bias-free per-channel FP8 in canonical ``(k, n)``."""
    weight = getattr(o_proj, "weight", None)
    scale = getattr(o_proj, "weight_scale", None)
    return (
        weight is not None
        and scale is not None
        and getattr(o_proj, "bias", None) is None
        and bool(getattr(o_proj, WEIGHT_FLIPPED_ATTR, False))
        and weight.dtype == torch.float8_e4m3fn
        and scale.dim() == 1
        and scale.shape[0] == weight.shape[1]
    )


GATHER_CAPS: tuple[int, ...] = (
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    44,
    64,
    96,
    128,
    192,
    260,
    384,
    512,
    768,
)


@dataclasses.dataclass
class _TokenParallelState:
    enabled: bool = False
    dcp_size: int = 1
    over_capacity: bool = False


_state = _TokenParallelState()

_JAX_RETURN_TYPES: dict[int, Any] = {
    1: jax.Array,
    2: tuple[jax.Array, jax.Array],
    3: tuple[jax.Array, jax.Array, jax.Array],
    4: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
}


def make_typed_jax_wrapper(
    fn: Callable[..., Any], arg_names: tuple[str, ...], num_outputs: int
) -> Callable[..., Any]:
    """Attach an explicit `jax.Array` `inspect.Signature` for `pcp_streaming_jax_op`."""
    ret_type = _JAX_RETURN_TYPES[num_outputs]

    def wrapped(*args: jax.Array) -> Any:
        return fn(*args)

    params = [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
        )
        for name in arg_names
    ]
    wrapped_any: Any = wrapped
    wrapped_any.__signature__ = inspect.Signature(params, return_annotation=ret_type)
    wrapped_any.__annotations__ = {
        **{name: jax.Array for name in arg_names},
        "return": ret_type,
    }
    return wrapped


def supported_for_config(vllm_config: Any | None) -> bool:
    """True strictly when token-parallel DCP prefill is safe for ``vllm_config``."""
    if vllm_config is None:
        return False
    model_cfg = getattr(vllm_config, "model_config", None)
    if not glm_dsa_v7x_active(model_cfg):
        return False
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    if parallel_cfg is None:
        return False
    dcp = int(getattr(parallel_cfg, "decode_context_parallel_size", 1))
    il = int(getattr(parallel_cfg, "cp_kv_cache_interleave_size", 1))
    if dcp <= 1 or il < 1:
        return False
    if getattr(vllm_config, "lora_config", None) is not None:
        return False
    if getattr(vllm_config, "speculative_config", None) is not None:
        return False
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        return False
    cache_cfg = getattr(vllm_config, "cache_config", None)
    if cache_cfg is None:
        return False
    block_size = int(getattr(cache_cfg, "block_size", 0))
    if block_size <= 0 or block_size % (dcp * il) != 0:
        return False
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    sizes = getattr(comp_cfg, "compile_sizes", None) if comp_cfg is not None else None
    if sizes:
        hi = [int(s) for s in sizes if int(s) >= MIN_TOKENS]
        if not hi or any(s % dcp != 0 for s in hi):
            return False
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    if sched_cfg is None:
        return False
    max_num_seqs = int(getattr(sched_cfg, "max_num_seqs", 0))
    return 0 < max_num_seqs <= MAX_SUPPORTED_SEQS


def configure_for(vllm_config: Any | None) -> bool:
    """Evaluate and cache whether token-parallel DCP prefill is enabled."""
    _state.over_capacity = False
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    dcp = int(getattr(parallel_cfg, "decode_context_parallel_size", 1) or 1)
    _state.dcp_size = dcp
    _state.enabled = supported_for_config(vllm_config)
    model_cfg = getattr(vllm_config, "model_config", None)
    if (
        not _state.enabled
        and vllm_config is not None
        and glm_dsa_v7x_active(model_cfg)
        and dcp > 1
    ):
        logger.warning_once(
            "Skipping token-parallel DCP prefill for current configuration "
            "(dcp=%d); falling back to standard DCP MLA.",
            dcp,
        )
    return _state.enabled


def _is_backbone_tag() -> bool:
    if vllm_compilation_backends is None:
        return True
    return getattr(vllm_compilation_backends, "model_tag", "backbone") == "backbone"


def applies(num_tokens: int, vllm_config: Any | None = None) -> bool:
    """True when backbone prefill bucket ``num_tokens`` routes token-parallel."""
    if vllm_config is not None:
        enabled = supported_for_config(vllm_config)
        dcp = int(vllm_config.parallel_config.decode_context_parallel_size)
    else:
        enabled = _state.enabled
        dcp = _state.dcp_size
    if not enabled or dcp <= 1:
        return False
    if not torch.compiler.is_compiling() and (
        _state.over_capacity or num_tokens < MIN_TOKENS or num_tokens % dcp != 0
    ):
        return False
    return (
        glm_dsa_v7x_active()
        and _is_backbone_tag()
        and num_tokens > bucket_gate_edge(MIN_TOKENS)
    )


def layout_for(vllm_config: Any) -> dcp_token_parallel.TokenParallelLayout:
    """Build the ``TokenParallelLayout`` geometry from ``vllm_config``."""
    parallel = vllm_config.parallel_config
    dcp = int(parallel.decode_context_parallel_size)
    page_size = int(vllm_config.cache_config.block_size)
    sched = vllm_config.scheduler_config
    max_num_seqs = int(sched.max_num_seqs)
    max_batched = int(getattr(sched, "max_num_batched_tokens", 0) or 0)
    vps = page_size * dcp
    max_model_len = int(vllm_config.model_config.max_model_len)
    token_budget = max(max_model_len, max_batched)
    max_pages = -(-token_budget // vps) + max_num_seqs + EXTRA_PAGES
    req_zigzag = DCP_TP_ZIGZAG_BANDS
    tp = int(getattr(parallel, "tensor_parallel_size", dcp))
    zigzag_bands = (
        req_zigzag
        if (
            glm_dsa_v7x_active(vllm_config.model_config)
            and req_zigzag > 1
            and dcp == tp
            and dcp >= 2
        )
        else 1
    )
    return dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=int(parallel.cp_kv_cache_interleave_size),
        page_size=page_size,
        split_rows=-(-max_num_seqs // SPLIT_ROWS_ALIGN) * SPLIT_ROWS_ALIGN,
        max_pages=max_pages,
        gather_caps=tuple(c for c in GATHER_CAPS if c < max_pages) + (max_pages,),
        zigzag_bands=zigzag_bands,
    )


def check_step_capacity(
    vllm_config: Any,
    padded_num_tokens: int,
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    num_reqs: int,
) -> bool:
    """Check whether scheduled prefill requests fit ``layout.max_pages``."""
    _state.over_capacity = False
    if not supported_for_config(vllm_config):
        return True
    dcp = int(vllm_config.parallel_config.decode_context_parallel_size)
    if padded_num_tokens < MIN_TOKENS or padded_num_tokens % dcp != 0:
        return True
    layout = layout_for(vllm_config)
    q_start = query_start_loc[:num_reqs]
    q_end = query_start_loc[1 : num_reqs + 1]
    listed = (q_end > layout.split_rows) & (q_end > q_start)
    vps = layout.virtual_page_size
    need = int(np.sum((seq_lens[:num_reqs][listed] + vps - 1) // vps))
    if need > layout.max_pages:
        _state.over_capacity = True
        logger.warning_once(
            "Token-parallel DCP prefill step needs %d gather pages (buffer holds %d); "
            "falling back to standard DCP MLA where uncompiled.",
            need,
            layout.max_pages,
        )
        return False
    return True


def _dcp_rows_metadata_jax(
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    qsl = jnp.minimum(query_start_loc, split_rows)
    cut = (query_start_loc[1:] - query_start_loc[:-1]) - (qsl[1:] - qsl[:-1])
    req = jnp.arange(seq_lens.shape[0], dtype=distribution.dtype)
    n = jnp.sum((query_start_loc[:-1] < split_rows) & (req < distribution[2])).astype(
        distribution.dtype
    )
    dist = jnp.stack(
        [jnp.minimum(distribution[0], n), jnp.minimum(distribution[1], n), n]
    )
    return seq_lens - cut, qsl, dist


def _sharded_rank_ids() -> torch.Tensor | None:
    if not glm_dsa_v7x_active():
        return None
    return dcp_rank_operand()


def make_attend(
    *,
    sm_scale: float,
    k_scale: float,
    cache_layout: str,
    lkv_dim: int,
    packed_rope: bool | None = None,
    split_q: bool = False,
    mixed_body_levers: tuple[str, ...] = (),
) -> Callable[..., jax.Array]:
    """Build the unsharded-cache ``attend`` callback over gathered KV pages."""
    from vllm_torchtpu.layers.core.attention_interface import sparse_mla_chunk_kwargs

    packed = {"packed_rope": True} if packed_rope else {}

    def attend(q, g_nope, g_rope, topk, num_rows, q_pe=None):
        split_kw = {"q_pe": q_pe} if split_q and q_pe is not None else {}
        out = sparse_mla_kernel.sparse_ragged_paged_attention(
            q,
            g_nope,
            g_rope,
            topk,
            None,
            jnp.stack([jnp.int32(0), num_rows.astype(jnp.int32)]),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=sm_scale,
            k_scale=k_scale,
            cache_layout=cache_layout,
            mixed_body_levers=mixed_body_levers,
            **attend_chunk_kwargs(sparse_mla_chunk_kwargs()),
            **packed,
            **split_kw,
        )
        return out if out.shape[-1] == lkv_dim else out[..., :lkv_dim]

    chunk_kw = attend_chunk_kwargs(sparse_mla_chunk_kwargs())
    chunk_rows = int(
        chunk_kw.get(
            "gather_and_attention_chunk_size",
            sparse_mla_kernel.DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
        )
    )

    def attend_hm(q, g_nope, g_rope, topk, num_rows, q_pe=None):
        split_kw = {"q_pe": q_pe} if split_q and q_pe is not None else {}
        chunks = sparse_mla_kernel.sparse_ragged_paged_attention(
            q,
            g_nope,
            g_rope,
            topk,
            None,
            jnp.stack([jnp.int32(0), num_rows.astype(jnp.int32)]),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=sm_scale,
            k_scale=k_scale,
            cache_layout=cache_layout,
            mixed_body_levers=mixed_body_levers,
            head_major_out_chunks=True,
            **attend_chunk_kwargs(sparse_mla_chunk_kwargs()),
            **packed,
            **split_kw,
        )
        assert isinstance(chunks, tuple)
        return tuple(c if c.shape[-1] == lkv_dim else c[..., :lkv_dim] for c in chunks)

    attend.head_major_chunks = attend_hm  # type: ignore[attr-defined]
    attend.head_major_chunk_rows = chunk_rows  # type: ignore[attr-defined]

    return attend


_TP_STREAMIDX_MIN_BKVP = 8
_TP_STREAMIDX_WIDE_BKVP = 16
_TP_STREAMIDX_WIDE_BKVP_KV_TOKENS = 49152
_TP_STREAMIDX_WIDE_KV_TOKENS = 16384
_TP_STREAMIDX_WIDE_CHUNK_TOKENS = 128


def _token_parallel_streamidx_knobs(
    num_rows: int,
    natural_shape: tuple[int, ...],
    kv_layout: KVLayout,
    mixed_bkvp: int | None,
    chunk_tokens: int | None,
) -> tuple[int | None, int | None]:
    num_pages = natural_shape[0]
    page_tokens = (
        natural_shape[-1]
        if kv_layout == KVLayout.SEQ_ALONG_LANE
        else natural_shape[1] * natural_shape[2]
    )
    kv_tokens = num_pages * page_tokens
    if mixed_bkvp is None:
        tp_mixed_bkvp = None
    elif (
        kv_tokens >= _TP_STREAMIDX_WIDE_BKVP_KV_TOKENS
        and num_pages >= _TP_STREAMIDX_WIDE_BKVP
        and num_pages % _TP_STREAMIDX_WIDE_BKVP == 0
    ):
        tp_mixed_bkvp = max(mixed_bkvp, _TP_STREAMIDX_WIDE_BKVP)
    elif num_pages >= _TP_STREAMIDX_MIN_BKVP:
        tp_mixed_bkvp = max(mixed_bkvp, _TP_STREAMIDX_MIN_BKVP)
    else:
        tp_mixed_bkvp = mixed_bkvp
    tp_chunk_tokens = (
        _TP_STREAMIDX_WIDE_CHUNK_TOKENS
        if (
            chunk_tokens == 256
            and kv_tokens > _TP_STREAMIDX_WIDE_KV_TOKENS
            and num_rows >= 256
            and num_rows % _TP_STREAMIDX_WIDE_CHUNK_TOKENS == 0
        )
        else chunk_tokens
    )
    return tp_mixed_bkvp, tp_chunk_tokens


def lead_streamidx_block_sizes(
    split_rows: int,
    bkvp: tuple[int, int, int],
    bq: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], bool]:
    if split_rows >= 16 and split_rows % 16 == 0 and bq[2] < 16:
        return bkvp, (bq[0], bq[1], 16), True
    return bkvp, bq, False


def make_select_topk(
    topk: int,
    kv_layout: KVLayout,
    *,
    chunk_tokens: int | None,
    small_bucket: tuple[int | None, int | None, int],
    skip_empty_decode: bool,
    skip_empty_decode_chunked: bool,
    mixed_bkvp: int | None,
    decode_pack: bool,
    shared_meta: bool,
    coop_v2: bool | None = None,
    double_slice_budget: bool | None = None,
) -> dcp_token_parallel.SelectTopkFn:
    """Build the unsharded ``streamindex_topk`` callback over gathered pages."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

    glm_v7x = glm_dsa_v7x_active()
    eff_coop_v2 = bool(glm_v7x) if coop_v2 is None else coop_v2
    eff_double_budget = (
        bool(glm_v7x) if double_slice_budget is None else double_slice_budget
    )

    def select(q_r, w_r, natural, rr, table):
        tp_mixed_bkvp, tp_chunk_tokens = _token_parallel_streamidx_knobs(
            q_r.shape[0], natural.shape, kv_layout, mixed_bkvp, chunk_tokens
        )
        bkvp, bq = mla_op._streamidx_block_sizes(
            q_r.shape[0], *small_bucket, tp_mixed_bkvp, decode_pack
        )
        page_tokens = (
            natural.shape[-1]
            if kv_layout == KVLayout.SEQ_ALONG_LANE
            else natural.shape[1] * natural.shape[2]
        )
        kv_tokens = natural.shape[0] * page_tokens
        enable_early_exit = mla_op.STREAMIDX_ENABLE_EARLY_EXIT and kv_tokens <= 2 * topk
        u32_sal = (
            kv_layout == KVLayout.SEQ_ALONG_LANE
            and natural.ndim == 3
            and natural.dtype == jnp.uint32
        )
        zero = jnp.int32(0)
        return streamindex_topk(
            q_r,
            w_r,
            natural,
            rr.seq_lens,
            table,
            rr.cu_q_lens,
            jnp.stack([zero, zero, rr.num_seqs]),
            k=topk,
            compression_ratio=1,
            num_kv_pages_per_block=bkvp,
            num_queries_per_block=bq,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            enable_early_exit=enable_early_exit,
            kv_layout=kv_layout,
            chunk_tokens=tp_chunk_tokens,
            decode_pack=decode_pack,
            decode_page_major=True if u32_sal else None,
            shared_meta=shared_meta,
            decode_as_mixed=True,
            coop_v2=eff_coop_v2,
            double_slice_budget=eff_double_budget,
        )

    return select


def _tie_rank_to_partial(
    partial: jax.Array, rank_ids: jax.Array, mesh: Any, sharded_spec: Any
) -> jax.Array:
    def _tie_local(part_local: jax.Array, rank_local: jax.Array) -> jax.Array:
        bits = jax.lax.bitcast_convert_type(
            part_local[0, 0, 0].astype(jnp.bfloat16), jnp.uint16
        ).astype(jnp.int32)
        zero = (jax.lax.optimization_barrier(bits) == jnp.int32(-1)).astype(
            rank_local.dtype
        )
        return rank_local + zero

    return jax.shard_map(
        _tie_local,
        mesh=mesh,
        in_specs=(sharded_spec, sharded_spec),
        out_specs=sharded_spec,
        check_vma=False,
    )(partial, rank_ids)


def build_attention_op(
    layer: Any,
    impl: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    q_replicated: bool,
    *,
    local_oproj: bool = True,
) -> Any:
    """Build or reuse the single-op token-parallel Sparse MLA custom op."""
    from vllm.distributed import get_tensor_model_parallel_world_size

    from vllm_torchtpu.layers.core.attention_interface import (
        _mixed_body_levers,
        sparse_mla_attention_dcp,
    )

    mesh = impl._dcp_mesh
    assert mesh is not None
    dcp, il = impl.dcp_size, impl.dcp_interleave_size
    if (dcp, il) != (layout.dcp_size, layout.interleave_size):
        raise ValueError(f"backend DCP ({dcp}, {il}) does not match {layout}")
    k_scale = None
    if layer.kv_cache_quantized_dtype is not None:
        _, k_scale, _ = impl._get_kv_scales(layer)
    dequant = float(k_scale) if k_scale is not None else 1.0
    sm_scale = float(layer.scale)

    rank_tensor = _sharded_rank_ids()
    has_rank = rank_tensor is not None
    uv_in_tp = bool(
        layout.dcp_size == get_tensor_model_parallel_world_size()
        and not hasattr(layer, "W_UV_scale")
    )
    mixed_levers = _mixed_body_levers(enabled=glm_dsa_v7x_active())
    num_heads = int(layer.num_heads)
    out_dim = int(layer.v_head_dim if uv_in_tp else layer.kv_lora_rank)
    chunk_tokens = envs.TPU_SPARSE_MLA_CHUNK_TOKENS
    split_q_inputs = bool(q_replicated)
    target_layer = layer
    del layer, impl

    layers: list[weakref.ref[Any]] = []
    sharded, rep = P(DCP_AXIS_NAME), P()

    def _run(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q: jax.Array,
        q_pe: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        topk_in: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        rank_ids: jax.Array | None,
        w_uv: jax.Array | None,
        w_o: jax.Array | None = None,
        s_o: jax.Array | None = None,
        q_tail: jax.Array | None = None,
        *,
        kv_sharded: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        live = [
            sharer
            for ref in layers
            if (sharer := ref()) is not None and hasattr(sharer, "mla_kv_spec")
        ]
        nope_spec, rope_spec = live[0].mla_kv_spec
        for other in live[1:]:
            if other.mla_kv_spec != (nope_spec, rope_spec):
                raise NotImplementedError(
                    f"{getattr(other, 'layer_name', other)} shares a sparse MLA "
                    f"TP op with {getattr(live[0], 'layer_name', live[0])} but "
                    "recorded different KV cache specs "
                    f"({other.mla_kv_spec} != {(nope_spec, rope_spec)})."
                )
        if nope_spec.page_size != layout.page_size:
            raise ValueError(
                f"sparse MLA cache page {nope_spec.page_size} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        if nope_spec.layout is not rope_spec.layout:
            raise ValueError("sparse MLA needs matching NoPE and RoPE layouts")

        glm_v7x_kv = glm_dsa_v7x_active()

        def write_kv(nope, rope, kv_c, k_pe_in, sl, bt, qsl, rank):
            return update_sparse_mla_kv_cache_dcp(
                nope,
                rope,
                kv_c,
                k_pe_in,
                sl,
                bt,
                qsl,
                rank,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp,
                interleave_size=il,
                compact_rows=bool(glm_v7x_kv),
            )

        attend = make_attend(
            sm_scale=sm_scale,
            k_scale=dequant,
            cache_layout=nope_spec.layout.value,
            packed_rope=rope_spec.packed_rope,
            lkv_dim=kv_c_normed.shape[-1],
            split_q=True,
            mixed_body_levers=mixed_levers,
        )
        read_pages = (
            functools.partial(page_gather.gather_pages, row_width=128),
            functools.partial(
                page_gather.gather_pages, row_width=rope_spec.token_bytes // 4
            ),
        )
        s = layout.split_rows
        num_tokens = kv_c_normed.shape[0] * dcp if kv_sharded else kv_c_normed.shape[0]
        rows = layout.rows_per_rank(num_tokens)

        def _write_once(nope, rope, kv_c, k_pe_in, sl, bt, qsl, *r_arg):
            rank = (
                r_arg[0][0]
                if r_arg
                else dcp_token_parallel.rank_as_data(DCP_AXIS_NAME, dcp)
            )
            if kv_sharded:
                return update_sparse_mla_kv_cache_dcp_sharded(
                    nope,
                    rope,
                    kv_c,
                    k_pe_in,
                    sl,
                    bt,
                    qsl,
                    rank,
                    axis_name=DCP_AXIS_NAME,
                    nope_spec=nope_spec,
                    rope_spec=rope_spec,
                    dcp_size=dcp,
                    interleave_size=il,
                    zigzag_bands=layout.effective_zigzag_bands(kv_c.shape[0] * dcp),
                )
            return write_kv(nope, rope, kv_c, k_pe_in, sl, bt, qsl, rank)

        r_extra = () if rank_ids is None else (rank_ids,)
        kv_spec = sharded if kv_sharded else rep
        kv_cache_nope, kv_cache_rope = jax.shard_map(
            _write_once,
            mesh=mesh,
            in_specs=(sharded, sharded, kv_spec, kv_spec, rep, rep, rep)
            + (sharded,) * len(r_extra),
            out_specs=(sharded, sharded),
            check_vma=False,
        )(
            kv_cache_nope,
            kv_cache_rope,
            kv_c_normed,
            k_pe,
            seq_lens,
            block_tables,
            query_start_loc,
            *r_extra,
        )

        seq_s, qsl_s, dist_s = _dcp_rows_metadata_jax(
            seq_lens, query_start_loc, request_distribution, s
        )
        if q_replicated:
            if q_tail is not None:
                q_dcp = jax.shard_map(
                    lambda x: x,
                    mesh=mesh,
                    in_specs=sharded,
                    out_specs=rep,
                    check_vma=False,
                )(q)
                q_tp = q_tail
            else:
                q_dcp = jax.shard_map(
                    lambda x: x[:s],
                    mesh=mesh,
                    in_specs=sharded,
                    out_specs=rep,
                    check_vma=False,
                )(q)
                q_tp = jax.shard_map(
                    lambda x: x[s:],
                    mesh=mesh,
                    in_specs=sharded,
                    out_specs=sharded,
                    check_vma=False,
                )(q)
            pe_dcp = jax.shard_map(
                lambda x: x[:s],
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(q_pe)
            pe_tp = jax.shard_map(
                lambda x: x[s:],
                mesh=mesh,
                in_specs=sharded,
                out_specs=sharded,
                check_vma=False,
            )(q_pe)
        else:
            q_dcp = jax.shard_map(
                lambda x: jax.lax.all_gather(x[:s], DCP_AXIS_NAME, axis=1, tiled=True),
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(q)
            pe_dcp = jax.shard_map(
                lambda x: jax.lax.all_gather(x[:s], DCP_AXIS_NAME, axis=1, tiled=True),
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(q_pe)
            q_tp, pe_tp = q, q_pe

        head_topk = jax.shard_map(
            lambda x: x[:s],
            mesh=mesh,
            in_specs=sharded,
            out_specs=sharded,
            check_vma=False,
        )(topk_in)
        tail_topk = jax.shard_map(
            lambda x: x[s : s + rows],
            mesh=mesh,
            in_specs=sharded,
            out_specs=sharded,
            check_vma=False,
        )(topk_in)

        kv_c_lead = (
            jnp.zeros((s, kv_c_normed.shape[-1]), kv_c_normed.dtype)
            if kv_sharded
            else kv_c_normed[:s]
        )
        k_pe_lead = (
            jnp.zeros((s, k_pe.shape[-1]), k_pe.dtype) if kv_sharded else k_pe[:s]
        )
        lead_live_widths = (128, *MLA_MIXED_LIVE_WIDTHS) if glm_v7x_kv else None
        lead_batch_size = 8 if s <= 64 and s % 8 == 0 else None
        _, _, partial, lse = sparse_mla_attention_dcp(
            q_dcp,
            pe_dcp,
            kv_c_lead,
            k_pe_lead,
            kv_cache_nope,
            kv_cache_rope,
            head_topk,
            seq_s,
            block_tables,
            qsl_s,
            dist_s,
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=sm_scale,
            k_scale=dequant,
            dcp_size=dcp,
            interleave_size=il,
            dcp_rank_ids=rank_ids,
            merge_pieces=False,
            skip_kv_write=True,
            mixed_live_widths_override=lead_live_widths,
            attention_kernel_batch_size=lead_batch_size,
        )
        if w_o is not None:
            assert s_o is not None and w_uv is not None and rank_ids is not None
            _, _, projected = dcp_token_parallel.sparse_mla_attention_with_lead_oproj(
                layout,
                mesh,
                DCP_AXIS_NAME,
                kv_cache_nope,
                kv_cache_rope,
                q_tp,
                None,
                None,
                tail_topk,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                lead_partial=partial,
                lead_lse=lse,
                write_kv=None,
                attend=attend,
                w_uv=w_uv,
                w_o=w_o,
                w_o_scale=s_o,
                rank_ids=rank_ids,
                q_own_rows=q_replicated,
                read_pages=read_pages,
                q_pe=pe_tp,
                w_uv_replicated=True,
                num_tokens=num_tokens,
                flat_topk_cap_strided=True,
                epilogue_3d=True,
            )
            return kv_cache_nope, kv_cache_rope, projected

        _, _, tail_out = dcp_token_parallel.sparse_mla_attention(
            layout,
            mesh,
            DCP_AXIS_NAME,
            kv_cache_nope,
            kv_cache_rope,
            q_tp,
            None,
            None,
            tail_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            write_kv=None,
            attend=attend,
            rank_ids=rank_ids,
            q_own_rows=q_replicated,
            read_pages=read_pages,
            w_uv=w_uv,
            q_pe=pe_tp,
            token_rows=False,
            w_uv_replicated=False,
            num_tokens=num_tokens,
            flat_topk_cap_strided=True,
        )

        def _merge_and_join(part, lse_part, tail_local, *w_uv_arg):
            lses = jax.lax.all_gather(lse_part, DCP_AXIS_NAME, axis=0, tiled=False)
            m = jax.nn.logsumexp(lses, axis=0)
            safe_m = jnp.where(jnp.isfinite(m), m, 0.0)
            w = jnp.where(jnp.isfinite(lse_part), jnp.exp(lse_part - safe_m), 0.0)
            weighted = (part.astype(jnp.float32) * w[..., None]).astype(part.dtype)
            head_local = jax.lax.psum_scatter(
                weighted, DCP_AXIS_NAME, scatter_dimension=1, tiled=True
            )
            if w_uv_arg:
                head_local = (
                    jnp.einsum(
                        "hrl,hlv->hrv",
                        jnp.swapaxes(head_local, 0, 1),
                        w_uv_arg[0].astype(head_local.dtype),
                    )
                    .swapaxes(0, 1)
                    .astype(tail_local.dtype)
                )
            return jnp.concatenate(
                [head_local.astype(tail_local.dtype), tail_local[: num_tokens - s]],
                axis=0,
            )

        uv_extra = () if w_uv is None else (w_uv,)
        full_out = jax.shard_map(
            _merge_and_join,
            mesh=mesh,
            in_specs=(sharded, sharded, sharded) + (sharded,) * len(uv_extra),
            out_specs=sharded,
            check_vma=False,
        )(partial, lse, tail_out, *uv_extra)
        return kv_cache_nope, kv_cache_rope, full_out

    def _dispatch(*args: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        if split_q_inputs:
            n, r, qh, qt = args[:4]
            rest = args[4:]
            base_tail, extra = rest[:8], rest[8:]
            rk = extra[0] if has_rank else None
            uv = extra[int(has_rank)] if uv_in_tp else None
            return cast(Any, _run)(n, r, qh, *base_tail, rk, uv, q_tail=qt)
        base, extra = args[:11], args[11:]
        rk = extra[0] if has_rank else None
        uv = extra[int(has_rank)] if uv_in_tp else None
        return cast(Any, _run)(*base, rk, uv)

    q_arg_names = ("qh", "qt", "qp") if split_q_inputs else ("qn", "qp")
    arg_names = (
        ("n", "r", *q_arg_names, "kc", "kp", "tk", "sl", "bt", "qsl", "rd")
        + (("rk",) if has_rank else ())
        + (("uv",) if uv_in_tp else ())
    )
    in_specs: tuple[P, ...] = (
        (sharded, sharded)
        + (sharded,) * len(q_arg_names)
        + (
            rep,
            rep,
            sharded,
            rep,
            rep,
            rep,
            rep,
        )
    )
    if has_rank:
        in_specs += (sharded,)
    if uv_in_tp:
        in_specs += (sharded,)

    fn = make_typed_jax_wrapper(_dispatch, arg_names, 3)
    kv_c_rest_idx = 2 if split_q_inputs else 1

    def _fake(kv_cache_nope, kv_cache_rope, q, *rest, **kwargs):
        del kwargs
        kv_c_normed = rest[kv_c_rest_idx]
        return (
            torch.empty_like(kv_cache_nope),
            torch.empty_like(kv_cache_rope),
            q.new_empty((kv_c_normed.shape[0], num_heads, out_dim)),
        )

    def _build_op(op_name: str):
        built = pcp_streaming_jax_op(
            op_name,
            fn,
            donate_argnums=(0, 1),
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=(sharded, sharded, sharded),
        )
        built.register_fake(_fake)
        return built, layers

    op, sharers = get_shared_jax_op(
        "sparse_mla_tp",
        (
            id(mesh),
            layout,
            bool(q_replicated),
            sm_scale,
            dequant,
            uv_in_tp,
            mixed_levers,
            chunk_tokens,
            has_rank,
            num_heads,
            out_dim,
        ),
        _build_op,
    )
    if getattr(target_layer, "use_sparse", True) and not any(
        ref() is target_layer for ref in sharers
    ):
        sharers.append(weakref.ref(target_layer))

    oproj_op = None
    oproj_sharded_kv_op = None
    if local_oproj and has_rank and uv_in_tp:
        oproj_arg_names = (
            "n",
            "r",
            *q_arg_names,
            "kc",
            "kp",
            "tk",
            "sl",
            "bt",
            "qsl",
            "rd",
            "rk",
            "uv",
            "wo",
            "so",
        )

        def _make_oproj_wrapper(kv_sharded: bool):
            def _oproj_dispatch(
                *args: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                if split_q_inputs:
                    n, r, qh, qt = args[:4]
                    rest = args[4:]
                    return cast(Any, _run)(
                        n, r, qh, *rest, q_tail=qt, kv_sharded=kv_sharded
                    )
                return cast(Any, _run)(*args, kv_sharded=kv_sharded)

            return make_typed_jax_wrapper(_oproj_dispatch, oproj_arg_names, 3)

        def _fake_oproj(kv_cache_nope, kv_cache_rope, q, *rest, **kwargs):
            del kwargs
            kv_c_normed, w_o = rest[kv_c_rest_idx], rest[-2]
            return (
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                q.new_empty((kv_c_normed.shape[0] // layout.dcp_size, w_o.shape[1])),
            )

        def _fake_oproj_sharded_kv(kv_cache_nope, kv_cache_rope, q, *rest, **kwargs):
            del kwargs
            kv_c_normed, w_o = rest[kv_c_rest_idx], rest[-2]
            return (
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                q.new_empty((kv_c_normed.shape[0], w_o.shape[1])),
            )

        oproj_in_specs = in_specs[:-1] + (rep, sharded, sharded)

        def _build_oproj_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _make_oproj_wrapper(kv_sharded=False),
                donate_argnums=(0, 1),
                mesh=mesh,
                input_partition_specs=oproj_in_specs,
                output_partition_specs=(sharded, sharded, sharded),
            )
            built.register_fake(_fake_oproj)
            return built, layers

        oproj_key = (
            id(mesh),
            layout,
            bool(q_replicated),
            sm_scale,
            dequant,
            mixed_levers,
            chunk_tokens,
            num_heads,
            out_dim,
        )
        oproj_op, _ = get_shared_jax_op(
            "sparse_mla_tp_oproj",
            oproj_key,
            _build_oproj_op,
        )

        kv_c_spec_idx = 3 + kv_c_rest_idx
        sharded_kv_in_specs = (
            oproj_in_specs[:kv_c_spec_idx]
            + (sharded, sharded)
            + oproj_in_specs[kv_c_spec_idx + 2 :]
        )

        def _build_oproj_sharded_kv_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _make_oproj_wrapper(kv_sharded=True),
                donate_argnums=(0, 1),
                mesh=mesh,
                input_partition_specs=sharded_kv_in_specs,
                output_partition_specs=(sharded, sharded, sharded),
            )
            built.register_fake(_fake_oproj_sharded_kv)
            return built, layers

        oproj_sharded_kv_op, _ = get_shared_jax_op(
            "sparse_mla_tp_oproj_sharded_kv",
            oproj_key,
            _build_oproj_sharded_kv_op,
        )

    qc_head_shift_op = None
    gather_owned_k_op = None
    indexer_owned_rows_op = None
    if q_replicated and oproj_sharded_kv_op is not None and rank_tensor is not None:
        split_rows = layout.split_rows

        def _qc_head_shift(q_c_local: jax.Array, rk: jax.Array) -> jax.Array:
            # q_c_local is the global [T, C] array here, outside shard_map.
            zigzag_local = layout.effective_zigzag_bands(q_c_local.shape[0]) > 1
            return jax.shard_map(
                lambda qc, r: dcp_token_parallel.gather_qc_head_rows(
                    qc,
                    r,
                    dcp_size=dcp,
                    split_rows=split_rows,
                    axis_name=DCP_AXIS_NAME,
                    zigzag_local=zigzag_local,
                ),
                mesh=mesh,
                in_specs=(sharded, sharded),
                out_specs=sharded,
                check_vma=False,
            )(q_c_local, rk)

        def _fake_qc_head_shift(
            q_c_local: torch.Tensor, rk: torch.Tensor
        ) -> torch.Tensor:
            del rk
            return q_c_local.new_empty(
                (q_c_local.shape[0] + split_rows, q_c_local.shape[1])
            )

        def _build_qc_head_shift_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _qc_head_shift,
                donate_argnums=(),
                mesh=mesh,
                input_partition_specs=(sharded, sharded),
                output_partition_specs=(sharded,),
            )
            built.register_fake(_fake_qc_head_shift)
            return built, layers

        qc_head_shift_op, _ = get_shared_jax_op(
            "dcp_qc_head_shift",
            (id(mesh), dcp, split_rows, layout.zigzag_bands),
            _build_qc_head_shift_op,
        )

        def _gather_owned_k(k_local: jax.Array, raw_own: jax.Array) -> jax.Array:
            return jax.shard_map(
                lambda kl, ro: jnp.take(
                    lax.all_gather(kl, DCP_AXIS_NAME, axis=0, tiled=True),
                    ro,
                    axis=0,
                ),
                mesh=mesh,
                in_specs=(sharded, sharded),
                out_specs=sharded,
                check_vma=False,
            )(k_local, raw_own)

        def _fake_gather_owned_k(
            k_local: torch.Tensor, raw_own: torch.Tensor
        ) -> torch.Tensor:
            return k_local.new_empty((raw_own.shape[0], k_local.shape[1]))

        def _build_gather_owned_k_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _gather_owned_k,
                donate_argnums=(),
                mesh=mesh,
                input_partition_specs=(sharded, sharded),
                output_partition_specs=(sharded,),
            )
            built.register_fake(_fake_gather_owned_k)
            return built, layers

        gather_owned_k_op, _ = get_shared_jax_op(
            "dcp_gather_owned_k",
            (id(mesh), dcp),
            _build_gather_owned_k_op,
        )

        def _indexer_owned_rows(
            seq_lens: jax.Array,
            page_indices: jax.Array,
            cu_q_lens: jax.Array,
            tok_slice: jax.Array,
            rk: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            num_tokens = int(tok_slice.shape[0])
            zb = layout.effective_zigzag_bands(num_tokens)
            live_pages = next(
                (
                    int(kc.kv_cache.shape[0])
                    for ref in layers
                    if (ly := ref()) is not None
                    and (kc := getattr(ly, "indexer_k_cache", None)) is not None
                    and kc.kv_cache.numel() > 0
                ),
                0,
            )
            eff_pages = live_pages if live_pages > 0 else (2**29 // max(dcp, 1))

            def _per_rank(
                sl: jax.Array,
                pi: jax.Array,
                cql: jax.Array,
                ts: jax.Array,
                r_arr: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                r_live, _ = lax.optimization_barrier((r_arr, ts))
                page, slot = idx_kv_cache_utils.get_dcp_page_and_slot(
                    num_tokens,
                    sl,
                    pi,
                    cql,
                    r_live[0],
                    page_size=layout.page_size,
                    num_pages=eff_pages,
                    dcp_size=dcp,
                    interleave_size=layout.interleave_size,
                )
                dst_rows = idx_kv_cache_utils.get_dst_rows(page, slot, layout.page_size)
                own_idx, _ = kv_cache_append.compact_owned_row_indices(
                    dst_rows,
                    num_seqs=sl.shape[0],
                    dcp_size=dcp,
                    interleave_size=layout.interleave_size,
                )
                raw_own = kv_cache_append.zigzag_to_rank_major_idx(
                    own_idx,
                    num_tokens=num_tokens,
                    dcp_size=dcp,
                    zigzag_bands=zb,
                )
                return own_idx, raw_own

            return jax.shard_map(
                _per_rank,
                mesh=mesh,
                in_specs=(rep, rep, rep, sharded, sharded),
                out_specs=(sharded, sharded),
                check_vma=False,
            )(seq_lens, page_indices, cu_q_lens, tok_slice, rk)

        def _fake_indexer_owned_rows(
            seq_lens: torch.Tensor,
            page_indices: torch.Tensor,
            cu_q_lens: torch.Tensor,
            tok_slice: torch.Tensor,
            rk: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del page_indices, cu_q_lens, rk
            cap = kv_cache_append.owned_rows_bound_from_local(
                tok_slice.shape[0],
                num_seqs=seq_lens.shape[0],
                interleave_size=layout.interleave_size,
            )
            return (
                seq_lens.new_empty((cap,), dtype=torch.int32),
                seq_lens.new_empty((cap,), dtype=torch.int32),
            )

        def _build_indexer_owned_rows_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _indexer_owned_rows,
                donate_argnums=(),
                mesh=mesh,
                input_partition_specs=(rep, rep, rep, sharded, sharded),
                output_partition_specs=(sharded, sharded),
            )
            built.register_fake(_fake_indexer_owned_rows)
            return built, layers

        indexer_owned_rows_op, _ = get_shared_jax_op(
            "dcp_indexer_owned_rows",
            (
                id(mesh),
                dcp,
                layout.page_size,
                layout.interleave_size,
                layout.zigzag_bands,
            ),
            _build_indexer_owned_rows_op,
        )

    def run(
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        q_pe: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        topk_in: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
        w_uv: torch.Tensor | None = None,
        w_o: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        kv_sharded: bool = False,
    ) -> torch.Tensor:
        nope_cache, rope_cache = kv_cache
        if split_q_inputs:
            if isinstance(q, tuple):
                q_head, q_tail = q
            else:
                q_head, q_tail = q[: layout.split_rows], q[layout.split_rows :]
            q_args: tuple[Any, ...] = (q_head, q_tail, q_pe)
        else:
            assert isinstance(q, torch.Tensor)
            q_args = (q, q_pe)
        if w_o is not None:
            selected_oproj_op = oproj_sharded_kv_op if kv_sharded else oproj_op
            if selected_oproj_op is None or w_uv is None or rank_tensor is None:
                raise RuntimeError(
                    "sequence-row o_proj requested but its op was not built"
                )
            new_nope, new_rope, out = selected_oproj_op(
                nope_cache,
                rope_cache,
                *q_args,
                kv_c_normed,
                k_pe,
                topk_in,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                rank_tensor,
                w_uv,
                *w_o,
            )
            nope_cache.copy_(new_nope)
            rope_cache.copy_(new_rope)
            return out
        if kv_sharded:
            raise ValueError("kv_sharded requires sequence-row o_proj (w_o)")
        new_nope, new_rope, out = op(
            nope_cache,
            rope_cache,
            *q_args,
            kv_c_normed,
            k_pe,
            topk_in,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            *(() if rank_tensor is None else (rank_tensor,)),
            *(() if w_uv is None else (w_uv,)),
        )
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)
        return out

    run.layout = layout  # type: ignore[attr-defined]
    run.q_replicated = q_replicated  # type: ignore[attr-defined]
    run.uv_in_tp = uv_in_tp  # type: ignore[attr-defined]
    run.split_q = True  # type: ignore[attr-defined]
    run.split_q_inputs = split_q_inputs  # type: ignore[attr-defined]
    run.serves_oproj = oproj_op is not None  # type: ignore[attr-defined]
    run.serves_sharded_kv = oproj_sharded_kv_op is not None  # type: ignore[attr-defined]
    run.qc_head_shift_op = qc_head_shift_op  # type: ignore[attr-defined]
    run.gather_owned_k_op = gather_owned_k_op  # type: ignore[attr-defined]
    run.indexer_owned_rows_op = indexer_owned_rows_op  # type: ignore[attr-defined]
    run.rank = dcp_rank_operand() if q_replicated else None  # type: ignore[attr-defined]
    return run


def q_rows(tp_op: Any, num_tokens: int) -> torch.Tensor:
    """Return row indices ``[split_rows + rows_per_rank]`` projected on this rank."""
    layout = tp_op.layout
    rows = layout.rows_per_rank(num_tokens)
    rank = tp_op.rank
    nb = layout.effective_zigzag_bands(num_tokens)
    if nb > 1:
        r = rank.reshape(())
        s = layout.split_rows
        dcp = layout.dcp_size
        band = rows // nb
        offs = torch.arange(band, dtype=rank.dtype, device=rank.device)
        bands = [
            ((k * dcp + r if k % 2 == 0 else (k + 1) * dcp - 1 - r) * band + offs)
            for k in range(nb)
        ]
        own = torch.cat(bands, dim=0)
        pad = own[-1:].expand(s)
        r0_tail = torch.cat([own[s:], pad], dim=0)
        tail = torch.where(r == 0, r0_tail, own)
        head = torch.arange(s, dtype=rank.dtype, device=rank.device)
        return torch.cat([head, tail], dim=0)
    ar = torch.arange(layout.split_rows + rows, dtype=rank.dtype, device=rank.device)
    own = ar + torch.where(ar >= layout.split_rows, rank * rows, 0)
    return torch.clamp(own, max=num_tokens - 1)


def attention(
    layer: Any,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    ql_nope: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    topk_rows: torch.Tensor,
    attn_metadata: Any,
    q_replicated: bool,
    w_o: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_sharded: bool = False,
) -> torch.Tensor:
    """Run token-parallel Sparse MLA attention for a prefill bucket."""
    tp_op = layer.sparse_mla_tp_op
    if tp_op.q_replicated != q_replicated:
        raise ValueError(
            f"token-parallel op built for q_replicated={tp_op.q_replicated}, "
            f"called with {q_replicated}"
        )
    layout = tp_op.layout
    s = layout.split_rows
    num_tokens = (
        kv_c_normed.shape[0] * layout.dcp_size if kv_sharded else kv_c_normed.shape[0]
    )
    rows = layout.rows_per_rank(num_tokens)
    if isinstance(ql_nope, tuple):
        q_head, q_tail = ql_nope
        if q_replicated and (q_head.shape[0] != s or q_tail.shape[0] != rows):
            raise ValueError(
                f"split q has ({q_head.shape[0]}, {q_tail.shape[0]}) rows, "
                f"not ({s}, {rows})"
            )
        q_dtype = q_head.dtype
    else:
        if q_replicated and ql_nope.shape[0] != s + rows:
            raise ValueError(
                f"q has {ql_nope.shape[0]} rows, not the {s + rows} q_rows"
            )
        q_dtype = ql_nope.dtype
    if kv_sharded and w_o is None:
        raise ValueError("kv_sharded requires sequence-row o_proj (w_o)")
    md = attn_metadata
    w_uv_arg = None
    if getattr(tp_op, "uv_in_tp", False):
        w_uv_src = layer.W_UV if w_o is None else layer.W_UV_tp_all
        w_uv_arg = w_uv_src.to(q_dtype) if w_uv_src.dtype != q_dtype else w_uv_src
    out = tp_op(
        kv_cache,
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        topk_rows,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        w_uv_arg,
        w_o,
        kv_sharded=kv_sharded,
    )
    if cp_mla_attention is not None:
        return cp_mla_attention._retain_rank(out)
    return out


def build_indexer_op(
    indexer: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    q_own_rows: bool = False,
) -> Any:
    """Build or reuse the single-op token-parallel DSA indexer custom op."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

    dcp, il = layout.dcp_size, layout.interleave_size
    s = layout.split_rows
    mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
    topk = int(indexer.topk_tokens)
    glm_v7x = glm_dsa_v7x_active()
    chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
    hist_merge = bool(glm_v7x and dcp <= 16 and 16 % dcp == 0)
    a2a_merge = bool(envs.TPU_DCP_INDEXER_A2A_MERGE or glm_v7x)
    a2a_min_tokens = int(envs.TPU_DCP_INDEXER_A2A_MIN_TOKENS)
    small_bq = envs.TPU_STREAMIDX_SMALL_BUCKET_BQ
    if small_bq is None and glm_v7x:
        small_bq = 4
    small_bkvp = envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP
    if small_bkvp is None and glm_v7x:
        small_bkvp = 32
    small_bucket = (small_bq, small_bkvp, int(envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS))
    kv_cache_mode = (
        envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
        if envs.is_indexer_kv_cache_mode_explicit()
        else None
    )
    skip_empty_decode = bool(glm_v7x)
    skip_empty_decode_chunked = bool(glm_v7x)
    mixed_bkvp = STREAMIDX_DEFAULT_MIXED_KV_PAGES if glm_v7x else None
    decode_pack = bool(glm_v7x)
    shared_meta = bool(getattr(indexer, "_shared_meta", glm_v7x))
    k_write_compact = bool(glm_v7x)

    rank_tensor = _sharded_rank_ids()
    has_rank = rank_tensor is not None
    small_ag = bool(glm_v7x and is_tpu_v7x() and has_rank)
    lane_tensor = dcp_lane_operand() if small_ag else None
    has_lane = lane_tensor is not None
    del indexer
    sharded, rep = P(DCP_AXIS_NAME), P()

    def _run(
        q_bytes: jax.Array,
        weights: jax.Array,
        cache_kv: jax.Array,
        k_packed: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        rank_ids: jax.Array | None,
        lane_ids: jax.Array | None,
        *,
        k_sharded: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        k_for_mode = k_packed[: k_packed.shape[0] // dcp] if k_sharded else k_packed
        writer, kv_layout = mla_op._resolve_indexer_kv_cache_mode(
            cache_kv, k_for_mode, mode=kv_cache_mode
        )
        if kv_layout != KVLayout.SEQ_ALONG_LANE:
            raise NotImplementedError(
                "Token-parallel DCP indexer requires SEQ_ALONG_LANE index-K "
                f"cache, got {kv_layout}"
            )
        if cache_kv.shape[-1] != layout.page_size:
            raise ValueError(
                f"index-K page {cache_kv.shape[-1]} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        idx_scatter_kw: dict[str, Any] = {}
        if k_sharded:
            step_tok = (
                int(q_bytes.shape[0] - dcp * layout.split_rows)
                if q_own_rows
                else int(k_packed.shape[0])
            )
            idx_scatter_kw["step_tokens"] = step_tok
            zb = layout.effective_zigzag_bands(step_tok)
            if zb > 1:
                idx_scatter_kw["zigzag_bands"] = zb
        cache_kv = mla_op._cp_scatter_indexer_k(
            cache_kv,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            mesh=mesh,
            cp_size=dcp,
            interleave_size=il,
            axis_name=DCP_AXIS_NAME,
            writer=writer,
            kv_layout=kv_layout,
            rank_ids=rank_ids,
            compact=k_write_compact,
            k_sharded=k_sharded,
            **idx_scatter_kw,
        )
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
        select = make_select_topk(
            topk,
            kv_layout,
            chunk_tokens=chunk_tokens,
            small_bucket=small_bucket,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            mixed_bkvp=mixed_bkvp,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
            coop_v2=bool(glm_v7x),
            double_slice_budget=bool(glm_v7x),
        )
        if q_own_rows:
            q_dcp = jax.shard_map(
                lambda x: x[:s],
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(q)
            w_dcp = jax.shard_map(
                lambda x: x[:s],
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(weights)
            q_tp = jax.shard_map(
                lambda x: x[s:],
                mesh=mesh,
                in_specs=sharded,
                out_specs=sharded,
                check_vma=False,
            )(q)
            w_tp = jax.shard_map(
                lambda x: x[s:],
                mesh=mesh,
                in_specs=sharded,
                out_specs=sharded,
                check_vma=False,
            )(weights)
        else:
            q_dcp, w_dcp = q[:s], weights[:s]
            q_tp, w_tp = q, weights

        seq_s, qsl_s, dist_s = _dcp_rows_metadata_jax(
            seq_lens, cu_q_lens, distribution, s
        )
        bkvp_dcp, bq_dcp = mla_op._streamidx_block_sizes(
            s, *small_bucket, mixed_bkvp, decode_pack
        )
        bkvp_dcp, bq_dcp, lead_decode_as_mixed = lead_streamidx_block_sizes(
            s, bkvp_dcp, bq_dcp
        )
        large = mla_op._dcp_indexer_a2a_for_bucket(s, True, a2a_min_tokens)
        head = streamindex_topk_dcp(
            q_dcp,
            w_dcp,
            cache_kv,
            seq_s,
            page_indices,
            qsl_s,
            dist_s,
            mesh=mesh,
            k=topk,
            compression_ratio=1,
            dcp_size=dcp,
            interleave_size=il,
            num_kv_pages_per_block=bkvp_dcp,
            num_queries_per_block=bq_dcp,
            decode_req_batch_size=2 if glm_v7x else 4,
            kv_layout=kv_layout,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            chunk_tokens=chunk_tokens,
            a2a_merge=large and a2a_merge and not hist_merge,
            hist_merge=large and hist_merge,
            dcp_rank_ids=rank_ids,
            dcp_lane_ids=lane_ids,
            flat_merge=bool(glm_v7x),
            small_all_gather=small_ag,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
            decode_as_mixed=lead_decode_as_mixed,
            coop_v2=bool(glm_v7x),
            double_slice_budget=bool(glm_v7x),
            coop_merge=bool(glm_v7x),
        )
        tail = dcp_token_parallel.indexer_topk(
            layout,
            mesh,
            DCP_AXIS_NAME,
            cache_kv,
            q_tp,
            w_tp,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            select_topk=select,
            rank_ids=rank_ids,
            q_own_rows=q_own_rows,
            flat_topk_cap_strided=True,
            index_k_packed_u32=(kv_layout == KVLayout.SEQ_ALONG_LANE),
        )
        full = jax.shard_map(
            lambda h, t: jnp.concatenate([h, t], axis=0),
            mesh=mesh,
            in_specs=(sharded, sharded),
            out_specs=sharded,
            check_vma=False,
        )(head, tail)
        return cache_kv, full

    idx_arg_names = (
        ("qb", "w", "kv", "kp", "sl", "pi", "cql", "dist")
        + (("rk",) if has_rank else ())
        + (("ln",) if has_lane else ())
    )

    def _make_idx_wrapper(k_sharded: bool):
        def _idx_dispatch(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
            base, extra = args[:8], args[8:]
            rk = extra[0] if has_rank else None
            ln = extra[int(has_rank)] if has_lane else None
            return cast(Any, _run)(*base, rk, ln, k_sharded=k_sharded)

        return make_typed_jax_wrapper(_idx_dispatch, idx_arg_names, 2)

    in_specs: tuple[P, ...] = (
        sharded if q_own_rows else rep,
        sharded if q_own_rows else rep,
        sharded,
        rep,
        rep,
        rep,
        rep,
        rep,
    )
    if has_rank:
        in_specs += (sharded,)
    if has_lane:
        in_specs += (sharded,)

    def _fake(q_bytes, weights, cache_kv, *rest, **kwargs):
        del weights, rest, kwargs
        out_rows = (
            q_bytes.shape[0]
            if q_own_rows
            else layout.split_rows + q_bytes.shape[0] // dcp
        )
        return (
            torch.empty_like(cache_kv),
            q_bytes.new_empty((out_rows, topk), dtype=torch.int32),
        )

    def _build_idx_op(op_name: str):
        built = pcp_streaming_jax_op(
            op_name,
            _make_idx_wrapper(k_sharded=False),
            donate_argnums=(2,),
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=(sharded, sharded),
        )
        built.register_fake(_fake)
        return built

    op_key = (
        id(mesh),
        layout,
        bool(q_own_rows),
        int(topk),
        chunk_tokens,
        bool(hist_merge),
        bool(a2a_merge),
        int(a2a_min_tokens),
        small_bucket,
        kv_cache_mode,
        bool(skip_empty_decode),
        bool(skip_empty_decode_chunked),
        mixed_bkvp,
        bool(decode_pack),
        shared_meta,
        bool(k_write_compact),
        has_rank,
        small_ag,
        has_lane,
    )
    op = get_shared_jax_op("streamidx_topk_tp", op_key, _build_idx_op)

    sharded_k_op = None
    if q_own_rows and has_rank:
        sharded_k_in_specs = in_specs[:3] + (sharded,) + in_specs[4:]

        def _build_sharded_k_op(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _make_idx_wrapper(k_sharded=True),
                donate_argnums=(2,),
                mesh=mesh,
                input_partition_specs=sharded_k_in_specs,
                output_partition_specs=(sharded, sharded),
            )
            built.register_fake(_fake)
            return built

        sharded_k_op = get_shared_jax_op(
            "streamidx_topk_tp_sharded_k", op_key, _build_sharded_k_op
        )

    def run(*args: torch.Tensor, k_sharded: bool = False) -> torch.Tensor:
        extra: tuple[torch.Tensor, ...] = ()
        if rank_tensor is not None:
            extra = (*extra, rank_tensor)
        if lane_tensor is not None:
            extra = (*extra, lane_tensor)
        selected_op = sharded_k_op if k_sharded else op
        if selected_op is None:
            raise ValueError("k_sharded requires q_own_rows and rank_ids")
        new_cache, flat = selected_op(*args, *extra)
        args[2].copy_(new_cache)
        return flat

    run.layout = layout  # type: ignore[attr-defined]
    run.q_own_rows = q_own_rows  # type: ignore[attr-defined]
    run.serves_sharded_k = sharded_k_op is not None  # type: ignore[attr-defined]
    return run


def indexer_rows(
    indexer: Any,
    q_bytes: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    k_packed: torch.Tensor,
    seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    distribution: torch.Tensor,
    *,
    k_sharded: bool = False,
) -> torch.Tensor:
    """Run token-parallel DSA indexer top-k for a prefill bucket."""
    extra = {"k_sharded": True} if k_sharded else {}
    return indexer.topk_op_tp(
        q_bytes,
        weights,
        kv_cache,
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **extra,
    )
