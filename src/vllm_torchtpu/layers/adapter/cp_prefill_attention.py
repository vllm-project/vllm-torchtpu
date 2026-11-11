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
"""Zigzag context-parallel prefill attention of one prompt over the TP group."""

import dataclasses
import functools
import inspect
from typing import Any

import jax
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
from vllm_torchtpu.tpu_info import v7x_model_fastpaths_enabled

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class LayerOps:
    full_attention: Any = None
    sliding_exchange: Any = None
    sliding_attention: Any = None
    prologue: Any = None


_OPS: dict[cp_prefill.LayerConfig, LayerOps] = {}


_TWO_ARRAYS = tuple[jax.Array, jax.Array]
_THREE_ARRAYS = tuple[jax.Array, jax.Array, jax.Array]


def _make_sig(
    arg_specs: tuple[tuple[str, Any], ...],
    return_annotation: Any = inspect.Signature.empty,
) -> inspect.Signature:
    params = [
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann)
        for n, ann in arg_specs
    ]
    return inspect.Signature(params, return_annotation=return_annotation)


def _register(
    name: str,
    fn: Any,
    fake: Any,
    donate: bool,
    arg_specs: tuple[tuple[str, Any], ...],
    return_annotation: Any,
    **bound: Any,
):
    donate_argnums = (0,) if donate and not envs.TPU_KERNEL_ITER_MODE else None
    bound_fn = functools.partial(fn, **bound)

    def wrapped(*args: Any):
        return bound_fn(*args)

    def wrapped_fake(*args: Any):
        return fake(*args)

    wrapped.__signature__ = _make_sig(arg_specs, return_annotation)  # type: ignore[attr-defined]
    wrapped_fake.__signature__ = _make_sig(tuple((n, Any) for n, _ in arg_specs))  # type: ignore[attr-defined]
    op = pallas.jax_op(
        f"cp_prefill::{name}_{len(_OPS)}", wrapped, donate_argnums=donate_argnums
    )
    op.register_fake(wrapped_fake)
    return op


def _register_layer(
    name: str,
    fn: Any,
    cfg: cp_prefill.LayerConfig,
    cache_view: Any,
    num_rows: Any,
    arg_names: tuple[str, ...],
    **bound: Any,
):
    heads, d = cfg.num_heads, cfg.head_dim

    def fake(kv_cache, *args):
        rows = num_rows(args)
        if cfg.fp8_out_rows:
            fp8 = kv_cache.new_empty((rows, heads * d), dtype=torch.float8_e4m3fn)
            scales = kv_cache.new_empty((rows, 1), dtype=torch.float32)
            return torch.empty_like(kv_cache), fp8, scales
        out = kv_cache.new_empty((rows, heads, d), dtype=args[0].dtype)
        return torch.empty_like(kv_cache), out

    return _register(
        name,
        fn,
        fake,
        donate=True,
        arg_specs=tuple((n, jax.Array) for n in arg_names),
        return_annotation=_THREE_ARRAYS if cfg.fp8_out_rows else _TWO_ARRAYS,
        cfg=cfg,
        cache_view=cache_view,
        **bound,
    )


def build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    """Return cached custom ops for ``cfg``."""
    if cfg not in _OPS:
        _OPS[cfg] = _build_ops(cfg)
    return _OPS[cfg]


def _build_ops(cfg: cp_prefill.LayerConfig) -> LayerOps:
    d, heads, kv_heads = cfg.head_dim, cfg.num_heads, cfg.num_kv_heads
    group, slots = heads // kv_heads, -(-2 * kv_heads // 4)
    from_q = cfg.prologue_eps is not None
    cache_view = functools.partial(
        native_kv_cache_view,
        num_kv_heads=cfg.local_kv_heads,
        head_dim=d,
        kv_layout=cfg.kv_layout,
    )
    row_count = (
        (lambda args: args[0].shape[1]) if from_q else (lambda args: args[0].shape[0])
    )
    prologue_op = None
    if from_q:

        def fake_prologue(qkv, *args):
            rows = qkv.shape[0]
            q_hbm = qkv.new_empty(
                (kv_heads, rows, group // 2, 2, d), dtype=torch.bfloat16
            )
            new_kv = qkv.new_empty((rows, slots, 4, d), dtype=torch.float8_e4m3fn)
            if not cfg.prologue_fp8_queries:
                return q_hbm, new_kv
            q_fp8 = qkv.new_empty(
                (kv_heads, rows, group // 4, 4, d), dtype=torch.float8_e4m3fn
            )
            return q_hbm, new_kv, q_fp8

        prologue_op = _register(
            "prologue",
            cp_prefill.qkv_prologue,
            fake_prologue,
            donate=False,
            arg_specs=tuple(
                (n, jax.Array)
                for n in ("qkv", "cos", "sin", "q_norm_weight", "k_norm_weight")
            ),
            return_annotation=(
                _THREE_ARRAYS if cfg.prologue_fp8_queries else _TWO_ARRAYS
            ),
            cfg=cfg,
        )

    if cfg.sliding_window is None or cfg.cache_all_pages:
        full_args = (
            (
                "kv_cache",
                "q_hbm",
                "q_fp8",
                "kv_rows",
                "seq_lens",
                "block_tables",
                "rank",
            )
            if from_q and cfg.prologue_fp8_queries
            else (
                ("kv_cache", "q_hbm", "kv_rows", "seq_lens", "block_tables", "rank")
                if from_q
                else (
                    "kv_cache",
                    "query",
                    "kv_rows",
                    "seq_lens",
                    "block_tables",
                    "rank",
                )
            )
        )
        return LayerOps(
            prologue=prologue_op,
            full_attention=_register_layer(
                "full_attention_from_q" if from_q else "full_attention",
                cp_prefill.cp_full_attention,
                cfg,
                cache_view,
                row_count,
                full_args,
                from_q=from_q,
                from_q8=cfg.prologue_fp8_queries,
            ),
        )

    source = "qkv" if cfg.exchange_from_qkv else ("prepared" if from_q else "kv")
    ex_tensors = {
        "qkv": (
            "qkv",
            "cos",
            "sin",
            "q_norm_weight",
            "k_norm_weight",
            "seq_lens",
            "rank",
        ),
        "prepared": ("new_kv", "seq_lens", "rank"),
        "kv": ("keys", "values", "seq_lens", "rank"),
    }[source]

    def fake_exchange(x, *args):
        page_size = int(args[-1])
        halo_pages, win_pages = cp_prefill._sliding_exchange_pages(cfg, page_size)
        rows = (2 * halo_pages + win_pages) * page_size
        if from_q:
            row = (slots, 4, d) if cfg.paged_gather else (slots * 4 * d,)
        else:
            row = (2 * kv_heads * d,)
        return x.new_empty((rows, *row), dtype=torch.int8)

    slide_args = (
        ("kv_cache", "q_hbm", "new_kv", "gathered", "seq_lens", "block_tables", "rank")
        if from_q
        else (
            "kv_cache",
            "query",
            "keys",
            "values",
            "gathered",
            "seq_lens",
            "block_tables",
            "rank",
        )
    )
    return LayerOps(
        prologue=prologue_op,
        sliding_exchange=_register(
            f"sliding_exchange_{source}",
            cp_prefill.cp_sliding_exchange_rows,
            fake_exchange,
            donate=False,
            arg_specs=(*((n, jax.Array) for n in ex_tensors), ("page_size", int)),
            return_annotation=jax.Array,
            cfg=cfg,
            source=source,
        ),
        sliding_attention=_register_layer(
            "sliding_attention_from_q" if from_q else "sliding_attention",
            cp_prefill.cp_sliding_attention,
            cfg,
            cache_view,
            row_count,
            slide_args,
            from_q=from_q,
        ),
    )


def _supports_head_page_writes(
    impl: PallasBatchedRPAAttentionBackendImpl, num_kv_heads: int, world: int
) -> bool:
    local_heads = cp_prefill.tp_kv_heads(num_kv_heads, world, 0)[1]
    if impl.kv_layout == cp_prefill.configs.KVLayout.SEQ_ALONG_LANE:
        return local_heads == 1
    return local_heads == 1 or local_heads % 2 == 0


def _layer_config(
    attn: Any,
    num_heads: int,
    num_kv_heads: int,
    world: int,
    prologue_eps: float | None = None,
    fp8_out_rows: bool = False,
) -> cp_prefill.LayerConfig:
    impl = attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "Context-parallel prefill attention needs --attention-backend CUSTOM, "
            f"got {type(impl).__name__}."
        )
    if impl.rpa_kernel is None and impl.rpa_dcp_cache_kernel is None:
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
            f"fp8_compute requires whole 4-way query packing groups, got "
            f"{num_heads} query heads per {num_kv_heads} KV heads."
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
    cache_all_pages = sliding and impl._prefix_caching
    dma_stores = prologue_eps is not None
    v7x_fastpaths = v7x_model_fastpaths_enabled()
    fp8_compute = impl.fp8_compute or (
        v7x_fastpaths and impl._fp8_compute_supported(num_heads, num_kv_heads)
    )
    head_page_writes = (not sliding) and _supports_head_page_writes(
        impl, num_kv_heads, world
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
        fence_output=v7x_fastpaths,
        unmasked_interior=v7x_fastpaths,
        head_page_writes=head_page_writes,
        shared_kv_rotary_dims=impl.shared_kv_rotary_dims,
        prologue_eps=prologue_eps,
        fp8_out_rows=fp8_out_rows,
        exchange_from_qkv=sliding and not cache_all_pages and prologue_eps is not None,
        paged_gather=prologue_eps is not None,
        prologue_dma_stores=dma_stores,
        prologue_fp8_queries=dma_stores and fp8_compute and not sliding,
        cache_all_pages=cache_all_pages,
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
    fp8_out_rows: bool = False,
) -> None:
    """Build and attach context-parallel custom ops onto ``attn``."""
    if get_vllm_model_wrapper_context().kv_cache_bundle is not None:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support the bundled KV cache."
        )
    cfg = _layer_config(
        attn, num_heads, num_kv_heads, world, prologue_eps, fp8_out_rows
    )
    if cfg not in _OPS:
        logger.info_once(
            "Context-parallel prefill attention on %d ranks: %d query / %d KV heads, "
            "sliding window %s, %s QK/PV, prologue=%s.",
            world,
            num_heads,
            num_kv_heads,
            cfg.sliding_window,
            "fp8" if cfg.fp8_compute else "bf16",
            prologue_eps is not None,
        )
    attn._cp_prefill_config = cfg
    attn._cp_prefill_ops = build_ops(cfg)


def _placeholder_output(
    like: torch.Tensor, cfg: cp_prefill.LayerConfig
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    rows = like.shape[0]
    if cfg.fp8_out_rows:
        width = cfg.num_heads * cfg.head_dim
        fp8 = like.new_zeros((rows, width), dtype=torch.float8_e4m3fn)
        return fp8, like.new_ones((rows, 1), dtype=torch.float32)
    return like.new_ones(like.shape)


def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _int8_rows(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    kv = torch.cat([key.view(torch.int8), value.view(torch.int8)], dim=-1)
    return kv.reshape(key.shape[0], -1)


def _validate_cp_call(
    attn: Any,
    rows: int,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
    *,
    need_prologue: bool,
) -> cp_prefill.LayerConfig:
    cfg: cp_prefill.LayerConfig | None = getattr(attn, "_cp_prefill_config", None)
    if (
        cfg is None
        or cfg.world != plan.world
        or (need_prologue and cfg.prologue_eps is None)
    ):
        raise RuntimeError(
            f"{attn.layer_name}: call prepare_cp_prefill_attention for world "
            f"{plan.world} before tracing."
        )
    if not isinstance(rank, torch.Tensor):
        raise TypeError("Pass rank as a [1] int32 device tensor, not a Python scalar.")
    if rows != 2 * plan.piece_tokens:
        raise ValueError(f"{rows} rows are not the two pieces of {plan}")
    return cfg


def cp_prefill_attention(
    attn: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attention of this rank's zigzag rows over the whole prompt from ``q, k, v``."""
    rows = query.shape[0]
    cfg = _validate_cp_call(attn, rows, plan, rank, need_prologue=False)
    d = cfg.head_dim
    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        return _placeholder_output(query, cfg)
    q_key, q_val = quantize_kv(
        torch.float8_e4m3fn,
        key.view(rows, cfg.num_kv_heads, d),
        value.view(rows, cfg.num_kv_heads, d),
        cfg.k_scale,
        cfg.v_scale,
    )
    assert q_val is not None
    out = cp_layer(
        attn._cp_prefill_ops,
        cfg,
        plan,
        kv_cache,
        query.view(rows, cfg.num_heads, d),
        q_key,
        q_val,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        rank,
        get_tp_group().device_group,
    )
    if cfg.fp8_out_rows or isinstance(out, tuple):
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
    """Attention of this rank's zigzag rows from fused ``qkv`` projection rows."""
    rows = qkv.shape[0]
    cfg = _validate_cp_call(attn, rows, plan, rank, need_prologue=True)
    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        return _placeholder_output(
            qkv.new_empty(rows, cfg.num_heads * cfg.head_dim), cfg
        )
    ops: LayerOps = attn._cp_prefill_ops
    group = get_tp_group().device_group
    seq_lens, block_tables = attn_metadata.seq_lens, attn_metadata.block_tables
    weights = (q_norm_weight, k_norm_weight)
    gathered: torch.Tensor | None = None
    if cfg.exchange_from_qkv:
        exchanged = ops.sliding_exchange(
            qkv, cos, sin, *weights, seq_lens, rank, plan.page_size
        )
        gathered = _waited(funcol.all_gather_single(exchanged, 0, group))
    q_hbm, new_kv, *q_fp8 = ops.prologue(qkv, cos, sin, *weights)
    if cfg.sliding_window is None or cfg.cache_all_pages:
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
        assert gathered is not None
        new_kv_cache, *out = ops.sliding_attention(
            kv_cache, q_hbm, new_kv, gathered, seq_lens, block_tables, rank
        )
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
    if cfg.sliding_window is None or cfg.cache_all_pages:
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
    kv_cache.copy_(new_kv_cache)
    return tuple(out) if cfg.fp8_out_rows else out[0]
