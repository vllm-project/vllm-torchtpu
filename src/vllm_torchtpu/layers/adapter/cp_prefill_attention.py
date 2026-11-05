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
"""Zigzag context-parallel prefill attention of one prompt over the TP group.

A context-parallel forward holds, on each tensor-parallel rank, the rows of two
zigzag pieces of one prompt with every query and KV head
(`kernels.experimental.batched_rpa.cp_prefill`). `cp_prefill_attention` gives
those rows their attention over the whole prompt and leaves the layer's
tensor-parallel KV cache as decode steps read it:

* full attention: the ranks all-gather their fp8 K/V rows, attend through the
  zigzag page table and cache their own KV head(s) of every prompt position;
* sliding window: the ranks all-gather their pieces' tails and the pages the
  next decode's window reaches; each attends within [halo | piece], the halos
  being the tails of the pieces before its own, and caches its heads of the
  window.

vllm-torchtpu compiles functional collectives into the program and so needs
every rank to run the same program: the rank reaches the ops as a device
tensor, never as a constant (rank-specialized programs computed wrong results
or hung in an 8-rank check). Each layer runs exactly one collective, on 2-D
int8 rows (or, with TPU_CP_PAGED_GATHER, int8 rows of the prologue's
[rows, slots, packing, d] layout), whose result feeds the one op that produces
the layer's output.

The step must hold exactly one request, the whole prompt, at batch index 0;
the caller checks that, and keeps prefix caching off (sliding-window layers
cache only the decode window). The batched-RPA work runs in torch custom ops that
`prepare_cp_prefill_attention` builds before tracing, as
`PallasAttentionBackendImpl.initialize_kernel` does for the regular kernel.
"""

import dataclasses
import functools
from typing import Any

import torch
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import cp_prefill
from vllm_torchtpu.layers.adapter.attention import (
    PallasBatchedRPAAttentionBackendImpl,
    native_kv_cache_view,
)
from vllm_torchtpu.layers.core.quantization import quantize_kv
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LayerOps:
    full_attention: Any = None
    sliding_exchange: Any = None
    sliding_attention: Any = None
    # From the layer's qkv projection rows (LayerConfig.prologue_eps).
    prologue: Any = None


_OPS: dict[cp_prefill.LayerConfig, LayerOps] = {}


def _register(name: str, fn, fake, donate: bool, **bound):
    # As in PallasAttentionBackendImpl._build_rpa_kernel: donating the cache
    # is only safe once the op is compiled into the graph.
    donate_argnums = (0,) if donate and not envs.TPU_KERNEL_ITER_MODE else None
    op = pallas.jax_op(
        f"cp_prefill::{name}_{len(_OPS)}",
        functools.partial(fn, **bound),
        donate_argnums=donate_argnums,
    )
    op.register_fake(fake)
    return op


def _register_layer(
    name: str,
    fn,
    fn_fp8_rows,
    cfg: cp_prefill.LayerConfig,
    cache_view,
    num_rows,
):
    """A layer op: (kv_cache, [rows, heads, d] output), or with
    `cfg.fp8_out_rows` (kv_cache, [rows, heads * d] fp8 rows, [rows, 1] f32
    scales). `num_rows(args)` reads the row count off the op's operands."""
    heads, d = cfg.num_heads, cfg.head_dim

    def fake(kv_cache, *args):
        rows = num_rows(args)
        if cfg.fp8_out_rows:
            fp8 = kv_cache.new_empty((rows, heads * d), dtype=torch.float8_e4m3fn)
            scales = kv_cache.new_empty((rows, 1), dtype=torch.float32)
            return torch.empty_like(kv_cache), fp8, scales
        # The output takes the query's dtype (args[0]: query or q_hbm).
        out = kv_cache.new_empty((rows, heads, d), dtype=args[0].dtype)
        return torch.empty_like(kv_cache), out

    return _register(
        name,
        fn_fp8_rows if cfg.fp8_out_rows else fn,
        fake,
        donate=True,
        cfg=cfg,
        cache_view=cache_view,
    )


def build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    """The torch ops `cp_layer` runs for layers like `cfg`, built once."""
    if cfg not in _OPS:
        _OPS[cfg] = _build_ops(cfg)
    return _OPS[cfg]


def _build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    local_kv_heads = cfg.local_kv_heads
    cache_view = functools.partial(
        native_kv_cache_view,
        num_kv_heads=local_kv_heads,
        head_dim=cfg.head_dim,
        kv_layout=cfg.kv_layout,
    )
    if cfg.prologue_eps is not None:
        return _build_ops_from_qkv(cfg, cache_view)
    if cfg.sliding_window is None:
        return LayerOps(
            full_attention=_register_layer(
                "full_attention",
                cp_prefill.full_attention_layer,
                cp_prefill.full_attention_layer_fp8_rows,
                cfg,
                cache_view,
                lambda args: args[0].shape[0],  # query
            )
        )

    def fake_exchange(keys, values, seq_lens, rank, page_size):
        # Two piece tails of halo pages, then halo + 1 window pages.
        halo_pages = cp_prefill.num_halo_pages(cfg.sliding_window, page_size)
        rows = (3 * halo_pages + 1) * page_size
        width = 2 * cfg.num_kv_heads * cfg.head_dim
        return keys.new_empty((rows, width), dtype=torch.int8)

    return LayerOps(
        sliding_exchange=_register(
            "sliding_exchange",
            cp_prefill.sliding_exchange_rows,
            fake_exchange,
            donate=False,
            cfg=cfg,
        ),
        sliding_attention=_register_layer(
            "sliding_attention",
            cp_prefill.sliding_attention_layer,
            cp_prefill.sliding_attention_layer_fp8_rows,
            cfg,
            cache_view,
            lambda args: args[0].shape[0],  # query
        ),
    )


def _build_ops_from_qkv(cfg: cp_prefill.LayerConfig, cache_view) -> LayerOps:
    """The ops of `cp_prefill_attention_from_qkv`: the prologue kernel turns the
    qkv rows into the attention kernel's query and new K/V layouts, and the
    layer ops take those."""
    d, heads, kv_heads = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    # prepare_inputs pads 2 * kv_heads K/V heads up to whole fp8 words of 4.
    group, slots = heads // kv_heads, -(-2 * kv_heads // 4)

    def fake_prologue(qkv, *args):
        rows = qkv.shape[0]
        q_hbm = qkv.new_empty((kv_heads, rows, group // 2, 2, d), dtype=torch.bfloat16)
        new_kv = qkv.new_empty((rows, slots, 4, d), dtype=torch.float8_e4m3fn)
        if not cfg.prologue_fp8_queries:
            return q_hbm, new_kv
        # wrapper.quantize_queries' layout: four heads per fp8 word.
        q_fp8 = qkv.new_empty(
            (kv_heads, rows, group // 4, 4, d), dtype=torch.float8_e4m3fn
        )
        return q_hbm, new_kv, q_fp8

    ops: dict[str, Any] = dict(
        prologue=_register(
            "prologue", cp_prefill.qkv_prologue, fake_prologue, donate=False, cfg=cfg
        )
    )
    if cfg.sliding_window is None:
        if cfg.prologue_fp8_queries:
            # (kv_cache, q_hbm, q_fp8, kv_rows, seq_lens, block_tables, rank)
            fns = (
                cp_prefill.full_attention_layer_from_q8,
                cp_prefill.full_attention_layer_from_q8_fp8_rows,
            )
        else:
            fns = (
                cp_prefill.full_attention_layer_from_q,
                cp_prefill.full_attention_layer_from_q_fp8_rows,
            )
        ops["full_attention"] = _register_layer(
            "full_attention_from_q",
            *fns,
            cfg,
            cache_view,
            lambda args: args[0].shape[1],  # q_hbm
        )
        return LayerOps(**ops)

    def fake_exchange(x, *args):
        # Two piece tails of halo pages, then halo + 1 window pages.
        halo_pages = cp_prefill.num_halo_pages(cfg.sliding_window, args[-1])
        rows = (3 * halo_pages + 1) * args[-1]
        row = (slots, 4, d) if cfg.paged_gather else (slots * 4 * d,)
        return x.new_empty((rows, *row), dtype=torch.int8)

    if cfg.exchange_from_qkv:
        # (qkv, cos, sin, q norm weight, k norm weight, seq_lens, rank, page size)
        ops["sliding_exchange"] = _register(
            "sliding_exchange_from_qkv",
            cp_prefill.sliding_exchange_rows_from_qkv,
            fake_exchange,
            donate=False,
            cfg=cfg,
        )
    else:
        # (new_kv, seq_lens, rank, page size)
        ops["sliding_exchange"] = _register(
            "sliding_exchange_from_kv",
            cp_prefill.sliding_exchange_rows_from_kv,
            fake_exchange,
            donate=False,
            cfg=cfg,
        )
    ops["sliding_attention"] = _register_layer(
        "sliding_attention_from_q",
        cp_prefill.sliding_attention_layer_from_q,
        cp_prefill.sliding_attention_layer_from_q_fp8_rows,
        cfg,
        cache_view,
        lambda args: args[0].shape[1],  # q_hbm
    )
    return LayerOps(**ops)


def _layer_config(
    attn: Any,
    num_heads: int,
    num_kv_heads: int,
    world: int,
    prologue_eps: float | None = None,
) -> cp_prefill.LayerConfig:
    impl = attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "Context-parallel prefill attention needs --attention-backend CUSTOM, "
            f"got {type(impl).__name__}."
        )
    if impl.rpa_kernel is None and impl.rpa_dcp_cache_kernel is None:
        # initialize_kernel decides DCP and PCP, which the checks below read.
        raise RuntimeError(
            f"{attn.layer_name}: prepare_cp_prefill_attention needs the backend's "
            "kernels, which initialize_kernel builds; this layer has none."
        )
    if impl.kv_cache_quantized_dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "Context-parallel prefill attention needs an fp8_e4m3 KV cache."
        )
    unsupported = [
        what
        for what, present in (
            ("attention sinks", impl.sinks is not None),
            ("KV sharing", impl.kv_sharing_target_layer_name is not None),
            ("DCP", impl.dcp_world_size > 1),
            ("PCP streaming", impl._pcp_streaming),
            ("the block-major KV pool", impl._bm_kernel),
        )
        if present
    ]
    if unsupported:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support "
            f"{', '.join(unsupported)}."
        )
    if 2 * num_kv_heads % 4:
        raise NotImplementedError(
            f"{num_kv_heads} KV heads do not fill whole fp8 K/V words; the packed "
            "pages would need padding slots."
        )
    if impl.num_heads * world != num_heads:
        raise ValueError(
            f"{num_heads} query heads are not {world} ranks x {impl.num_heads}."
        )
    if impl.fp8_compute and (num_heads // num_kv_heads) % 4:
        raise ValueError(
            f"TPU_FP8_ATTENTION: {num_heads} query heads per {num_kv_heads} KV heads "
            "are not whole fp8 packing groups."
        )
    if prologue_eps is not None and (
        (num_heads // num_kv_heads) % 2 or impl.head_size % 128
    ):
        raise NotImplementedError(
            f"The qkv prologue needs an even number of query heads per KV head and "
            f"a lane-aligned head size, got {num_heads // num_kv_heads} and "
            f"{impl.head_size}."
        )
    sliding = impl.sliding_window is not None
    if envs.TPU_RPA_PROLOGUE_FP8_QUERIES and not envs.TPU_RPA_PROLOGUE_DMA_STORES:
        raise ValueError(
            "TPU_RPA_PROLOGUE_FP8_QUERIES needs TPU_RPA_PROLOGUE_DMA_STORES"
        )
    dma_stores = envs.TPU_RPA_PROLOGUE_DMA_STORES and prologue_eps is not None
    # The layer runs every head here, so its own head ratio can admit fp8 QK/PV
    # where the tensor-parallel shard's does not.
    fp8_compute = impl.fp8_compute or (
        envs.TPU_FP8_ATTENTION
        and envs.TPU_CP_FP8_ATTENTION
        and impl._fp8_compute_supported(num_heads, num_kv_heads)
    )
    cfg = cp_prefill.LayerConfig(
        world=world,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=impl.head_size,
        sliding_window=impl.sliding_window,
        sm_scale=impl.scale,
        soft_cap=impl.logits_soft_cap,
        k_scale=attn._k_scale_float,
        v_scale=attn._v_scale_float,
        fp8_compute=fp8_compute,
        kv_layout=impl.kv_layout,
        sliding_prefill_max_bkv=(
            envs.TPU_RPA_SLIDING_PREFILL_MAX_BKV if sliding else None
        ),
        fence_output=envs.TPU_RPA_FENCE_OUTPUT,
        unmasked_interior=envs.TPU_RPA_UNMASKED_INTERIOR,
        head_page_writes=envs.TPU_CP_KV_WRITE_KERNEL and not sliding,
        shared_kv_rotary_dims=impl.shared_kv_rotary_dims,
        prologue_eps=prologue_eps,
        fp8_out_rows=envs.TPU_CP_FUSED_EPILOGUE,
        exchange_from_qkv=(
            envs.TPU_CP_EARLY_HALO and sliding and prologue_eps is not None
        ),
        paged_gather=envs.TPU_CP_PAGED_GATHER and prologue_eps is not None,
        # The tensor-parallel prologue's switches, for the same kernel options.
        prologue_dma_stores=dma_stores,
        prologue_fp8_queries=(
            dma_stores
            and envs.TPU_RPA_PROLOGUE_FP8_QUERIES
            and fp8_compute
            and not sliding
        ),
    )
    if cfg.local_kv_heads != impl.num_kv_heads:
        raise ValueError(
            f"{num_kv_heads} KV heads on {world} ranks give {cfg.local_kv_heads} per "
            f"rank; the layer's cache holds {impl.num_kv_heads}."
        )
    return cfg


def prepare_cp_prefill_attention(
    attn: Any,
    *,
    num_heads: int,
    num_kv_heads: int,
    world: int,
    prologue_eps: float | None = None,
) -> None:
    """Builds the ops `cp_prefill_attention` runs for this layer.

    Call it outside any compiled region, after the KV scales are loaded and the
    backend's `initialize_kernel`.

    Args:
        attn: The layer's vLLM `Attention` module (tensor-parallel heads).
        num_heads: Query heads of the layer (all ranks').
        num_kv_heads: KV heads of the layer (all ranks').
        world: Tensor-parallel world size.
        prologue_eps: The q/k/v norm epsilon, to run the layer from its qkv
            rows (`cp_prefill_attention_from_qkv`) instead.
    """
    if get_vllm_model_wrapper_context().kv_cache_bundle is not None:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support the bundled KV cache."
        )
    cfg = _layer_config(attn, num_heads, num_kv_heads, world, prologue_eps)
    if cfg not in _OPS:
        stores = ""
        if cfg.prologue_dma_stores:
            stores = " with DMA stores" + (
                " and FP8 queries" if cfg.prologue_fp8_queries else ""
            )
        logger.info_once(
            "Context-parallel prefill attention on %d ranks: %d query / %d KV heads, "
            "sliding window %s, %s QK/PV, KV head pages written by %s, %s%s.",
            world,
            num_heads,
            num_kv_heads,
            cfg.sliding_window,
            "fp8" if cfg.fp8_compute else "bf16",
            "the head-page kernel" if cfg.head_page_writes else "XLA slices",
            "from qkv rows" if prologue_eps is not None else "from q, k and v",
            stores,
        )
    attn._cp_prefill_config = cfg
    attn._cp_prefill_ops = build_ops(cfg)


def _placeholder_output(
    like: torch.Tensor, cfg: cp_prefill.LayerConfig
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """What the layer returns before KV caches exist, in its output's form."""
    rows = like.shape[0]
    if cfg.fp8_out_rows:
        width = cfg.num_heads * cfg.head_dim
        fp8 = like.new_zeros((rows, width), dtype=torch.float8_e4m3fn)
        return fp8, like.new_ones((rows, 1), dtype=torch.float32)
    return like.new_ones(like.shape)


def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _int8_rows(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """fp8 [n, kv, d] K/V as int8 `cp_prefill.pack_kv_rows` rows (2-D)."""
    kv = torch.cat([key.view(torch.int8), value.view(torch.int8)], dim=-1)
    return kv.reshape(key.shape[0], -1)


def cp_prefill_attention(
    attn: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attention of this rank's zigzag rows over the whole prompt.

    Args:
        attn: The layer's vLLM `Attention` module, prepared with
            `prepare_cp_prefill_attention`.
        query: [rows, num_heads * head_dim] of this rank's two pieces (first,
            then second), after the q norm and rotary embedding.
        key: [rows, num_kv_heads * head_dim], after the k norm and rotary.
        value: [rows, num_kv_heads * head_dim], after the v norm.
        plan: The step's zigzag plan; its page size is the KV cache's.
        rank: [1] int32 device tensor holding this tensor-parallel rank, a
            program input (not a constant), so every rank runs one program.

    Returns:
        [rows, num_heads * head_dim] attention output, or with
        TPU_CP_FUSED_EPILOGUE its per-token fp8 rows and [rows, 1] f32 scales
        (the o projection's quantized input).
    """
    cfg: cp_prefill.LayerConfig | None = getattr(attn, "_cp_prefill_config", None)
    if cfg is None or cfg.world != plan.world:
        raise RuntimeError(
            f"{attn.layer_name}: call prepare_cp_prefill_attention for a world of "
            f"{plan.world} before tracing."
        )
    if not isinstance(rank, torch.Tensor):
        raise TypeError(
            "Pass the rank as a device tensor: a constant rank specializes each "
            "rank's program, which compiled collectives do not support."
        )
    rows, d = query.shape[0], cfg.head_dim
    if rows != 2 * plan.piece_tokens:
        raise ValueError(f"{rows} rows are not the two pieces of {plan}")
    attn_metadata, layer, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        # The memory-profiling run, before caches exist; the backend does the same.
        return _placeholder_output(query, cfg)
    key, value = quantize_kv(
        layer.impl.kv_cache_quantized_dtype,
        key.view(rows, cfg.num_kv_heads, d),
        value.view(rows, cfg.num_kv_heads, d),
        cfg.k_scale,
        cfg.v_scale,
    )
    out = cp_layer(
        attn._cp_prefill_ops,
        cfg,
        plan,
        kv_cache,
        query.view(rows, cfg.num_heads, d),
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        rank,
        get_tp_group().device_group,
    )
    if cfg.fp8_out_rows:
        return out
    return out.reshape(rows, cfg.num_heads * d)


def cp_prefill_attention_from_qkv(
    attn: Any,
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """`cp_prefill_attention` from the layer's qkv projection rows.

    One Pallas pass (`prologue.qkv_to_rpa_inputs`) runs the per-head q/k
    (weighted) and v (unweighted) RMSNorms, the neox rotary and the FP8 K/V
    quantization, and writes the query and new K/V in the attention kernel's
    layouts, which the layer then takes as they are.

    Args:
        attn: The layer's vLLM `Attention` module, prepared with
            `prepare_cp_prefill_attention(..., prologue_eps=...)`.
        qkv: [rows, (num_heads + 2 * num_kv_heads) * head_dim] bf16 of this
            rank's two pieces.
        cos, sin: [rows, rotary_dim // 2] rotary cache rows of the rows'
            positions.
        q_norm_weight, k_norm_weight: [head_dim] norm weights.
        plan, rank: As for `cp_prefill_attention`.

    Returns:
        As `cp_prefill_attention`.
    """
    cfg: cp_prefill.LayerConfig | None = getattr(attn, "_cp_prefill_config", None)
    if cfg is None or cfg.world != plan.world or cfg.prologue_eps is None:
        raise RuntimeError(
            f"{attn.layer_name}: call prepare_cp_prefill_attention with "
            f"prologue_eps for a world of {plan.world} before tracing."
        )
    if not isinstance(rank, torch.Tensor):
        raise TypeError("Pass the rank as a device tensor (see cp_prefill_attention).")
    rows = qkv.shape[0]
    if rows != 2 * plan.piece_tokens:
        raise ValueError(f"{rows} rows are not the two pieces of {plan}")
    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        # The memory-profiling run, before caches exist.
        return _placeholder_output(
            qkv.new_empty(rows, cfg.num_heads * cfg.head_dim), cfg
        )
    ops: LayerOps = attn._cp_prefill_ops
    group = get_tp_group().device_group
    seq_lens, block_tables = attn_metadata.seq_lens, attn_metadata.block_tables
    weights = (q_norm_weight, k_norm_weight)
    if cfg.exchange_from_qkv:
        # The exchanged rows' own prologue: the all-gather overlaps the rest's.
        exchanged = ops.sliding_exchange(
            qkv, cos, sin, *weights, seq_lens, rank, plan.page_size
        )
        gathered = _waited(funcol.all_gather_single(exchanged, 0, group))
    # With cfg.prologue_fp8_queries also the FP8 queries, which the
    # full-attention op takes after q_hbm.
    q_hbm, new_kv, *q_fp8 = ops.prologue(qkv, cos, sin, *weights)
    if cfg.sliding_window is None:
        # [rows, slots, 4, d] is `pack_kv_rows`' order: as int8 rows of that
        # layout (cfg.paged_gather) or 2-D.
        kv_rows = new_kv.view(torch.int8)
        if not cfg.paged_gather:
            kv_rows = kv_rows.reshape(rows, -1)
        gathered = _waited(funcol.all_gather_single(kv_rows, 0, group))
        new_kv_cache, *out = ops.full_attention(
            kv_cache, q_hbm, *q_fp8, gathered, seq_lens, block_tables, rank
        )
    else:
        if not cfg.exchange_from_qkv:
            exchanged = ops.sliding_exchange(new_kv, seq_lens, rank, plan.page_size)
            gathered = _waited(funcol.all_gather_single(exchanged, 0, group))
        new_kv_cache, *out = ops.sliding_attention(
            kv_cache, q_hbm, new_kv, gathered, seq_lens, block_tables, rank
        )
    # The ops donate the cache; the copy_ folds into that alias when compiled.
    kv_cache.copy_(new_kv_cache)
    if cfg.fp8_out_rows:
        return tuple(out)
    return out[0].reshape(rows, cfg.num_heads * cfg.head_dim)


def cp_layer(
    ops: LayerOps,
    cfg: cp_prefill.LayerConfig,
    plan: cp_prefill.ZigzagPlan,
    kv_cache: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    rank: torch.Tensor,
    group: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """`cp_prefill_attention` past the forward context: [rows, heads, d] query,
    fp8 [rows, kv_heads, d] K/V, [1] int32 rank; writes `kv_cache` in place.
    Returns the [rows, heads, d] output, or with `cfg.fp8_out_rows` the o
    projection's fp8 rows and scales."""
    if cfg.sliding_window is None:
        gathered = _waited(funcol.all_gather_single(_int8_rows(key, value), 0, group))
        new_kv_cache, *out = ops.full_attention(
            kv_cache, query, gathered, seq_lens, block_tables, rank
        )
    else:
        rows = ops.sliding_exchange(key, value, seq_lens, rank, plan.page_size)
        gathered = _waited(funcol.all_gather_single(rows, 0, group))
        new_kv_cache, *out = ops.sliding_attention(
            kv_cache, query, key, value, gathered, seq_lens, block_tables, rank
        )
    # The ops donate the cache; the copy_ folds into that alias when compiled.
    kv_cache.copy_(new_kv_cache)
    return tuple(out) if cfg.fp8_out_rows else out[0]
