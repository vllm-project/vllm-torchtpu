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
or hung in an 8-rank check). Each layer runs exactly one collective, on a 2-D
tensor, whose result feeds the one op that produces the layer's output.

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
    if cfg.sliding_window is None:

        def fake_full(kv_cache, query, *args):
            return torch.empty_like(kv_cache), torch.empty_like(query)

        return LayerOps(
            full_attention=_register(
                "full_attention",
                cp_prefill.full_attention_layer,
                fake_full,
                donate=True,
                cfg=cfg,
                cache_view=cache_view,
            )
        )

    def fake_exchange(keys, values, seq_lens, rank, page_size):
        # Two piece tails of halo pages, then halo + 1 window pages.
        halo_pages = cp_prefill.num_halo_pages(cfg.sliding_window, page_size)
        rows = (3 * halo_pages + 1) * page_size
        width = 2 * cfg.num_kv_heads * cfg.head_dim
        return keys.new_empty((rows, width), dtype=torch.int8)

    def fake_sliding(kv_cache, query, *args):
        return torch.empty_like(kv_cache), torch.empty_like(query)

    return LayerOps(
        sliding_exchange=_register(
            "sliding_exchange",
            cp_prefill.sliding_exchange_rows,
            fake_exchange,
            donate=False,
            cfg=cfg,
        ),
        sliding_attention=_register(
            "sliding_attention",
            cp_prefill.sliding_attention_layer,
            fake_sliding,
            donate=True,
            cfg=cfg,
            cache_view=cache_view,
        ),
    )


def _layer_config(
    attn: Any, num_heads: int, num_kv_heads: int, world: int
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
    if impl.sliding_window is not None and world < 3:
        raise ValueError(
            f"Sliding-window layers need 3 or more ranks to exchange halos, got {world}"
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
    sliding = impl.sliding_window is not None
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
        head_page_writes=envs.TPU_CP_KV_WRITE_KERNEL and not sliding,
    )
    if cfg.local_kv_heads != impl.num_kv_heads:
        raise ValueError(
            f"{num_kv_heads} KV heads on {world} ranks give {cfg.local_kv_heads} per "
            f"rank; the layer's cache holds {impl.num_kv_heads}."
        )
    return cfg


def prepare_cp_prefill_attention(
    attn: Any, *, num_heads: int, num_kv_heads: int, world: int
) -> None:
    """Builds the ops `cp_prefill_attention` runs for this layer.

    Call it outside any compiled region, after the KV scales are loaded and the
    backend's `initialize_kernel`.

    Args:
        attn: The layer's vLLM `Attention` module (tensor-parallel heads).
        num_heads: Query heads of the layer (all ranks').
        num_kv_heads: KV heads of the layer (all ranks').
        world: Tensor-parallel world size.
    """
    if get_vllm_model_wrapper_context().kv_cache_bundle is not None:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support the bundled KV cache."
        )
    cfg = _layer_config(attn, num_heads, num_kv_heads, world)
    if cfg not in _OPS:
        logger.info_once(
            "Context-parallel prefill attention on %d ranks: %d query / %d KV heads, "
            "sliding window %s, %s QK/PV, KV head pages written by %s.",
            world,
            num_heads,
            num_kv_heads,
            cfg.sliding_window,
            "fp8" if cfg.fp8_compute else "bf16",
            "the head-page kernel" if cfg.head_page_writes else "XLA slices",
        )
    attn._cp_prefill_config = cfg
    attn._cp_prefill_ops = build_ops(cfg)


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
) -> torch.Tensor:
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
        [rows, num_heads * head_dim] attention output.
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
        return torch.ones_like(query)
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
    return out.reshape(rows, cfg.num_heads * d)


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
) -> torch.Tensor:
    """`cp_prefill_attention` past the forward context: [rows, heads, d] query,
    fp8 [rows, kv_heads, d] K/V, [1] int32 rank; writes `kv_cache` in place."""
    if cfg.sliding_window is None:
        gathered = _waited(funcol.all_gather_single(_int8_rows(key, value), 0, group))
        new_kv_cache, out = ops.full_attention(
            kv_cache, query, gathered, seq_lens, block_tables, rank
        )
    else:
        rows = ops.sliding_exchange(key, value, seq_lens, rank, plan.page_size)
        gathered = _waited(funcol.all_gather_single(rows, 0, group))
        new_kv_cache, out = ops.sliding_attention(
            kv_cache, query, key, value, gathered, seq_lens, block_tables, rank
        )
    # The ops donate the cache; the copy_ folds into that alias when compiled.
    kv_cache.copy_(new_kv_cache)
    return out
