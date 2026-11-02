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

* full attention: the ranks all-gather their fp8 K/V pages, attend through the
  zigzag page table and cache their own KV head(s) of every prompt position;
* sliding window: each rank receives the pages before its two pieces from the
  ranks holding them (collective permutes, in a JAX op over the TP mesh) and
  attends within [halo | piece]; the pages the next decode's window reaches go
  to every rank's heads in an all-to-all.

The step must hold exactly one request, the whole prompt, at batch index 0;
the caller checks that, and keeps prefix caching off (sliding-window layers
cache only the decode window). The batched-RPA work runs in torch custom ops that
`prepare_cp_prefill_attention` builds before tracing, as
`PallasAttentionBackendImpl.initialize_kernel` does for the regular kernel.
"""

import dataclasses
import functools
from typing import Any

import jax
import torch
import torch.distributed as dist
from jax.sharding import PartitionSpec as P
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from torch_tpu._internal.pallas import get_pallas_mesh
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
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


# The mesh axis of the halo exchange: the TP ranks in rank order.
_AXIS = "cp"


@dataclasses.dataclass(frozen=True)
class _LayerOps:
    full_attention: Any = None
    halo_exchange: Any = None
    sliding_attention: Any = None
    window_write: Any = None


@functools.cache
def _mesh(world: int) -> jax.sharding.Mesh:
    # Pallas meshes span the torch.distributed world; here that is the TP group.
    if dist.get_world_size() != world:
        raise NotImplementedError(
            "Context-parallel prefill needs the tensor-parallel group to be the "
            f"whole world ({world} of {dist.get_world_size()} ranks)."
        )
    return get_pallas_mesh((_AXIS,))


def _halo_exchange_fn(mesh: jax.sharding.Mesh, sliding_window: int):
    # A closure, not a partial: sharded_jax_op checks every parameter's type.
    def halo_exchange(keys: jax.Array, values: jax.Array, page_size: int) -> jax.Array:
        halo_pages = cp_prefill.num_halo_pages(sliding_window, page_size)

        def exchange(k, v):
            return cp_prefill.exchange_halos(
                k, v, halo_pages, page_size, _AXIS, mesh.size
            )

        return jax.shard_map(
            exchange,
            mesh=mesh,
            in_specs=(P(_AXIS), P(_AXIS)),
            out_specs=P(_AXIS),
            check_vma=False,
        )(keys, values)

    return halo_exchange


_OPS: dict[cp_prefill.LayerConfig, _LayerOps] = {}


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


def _build_ops(cfg: cp_prefill.LayerConfig) -> _LayerOps:
    _, local_kv_heads = cfg.tp_kv_heads
    cache_view = functools.partial(
        native_kv_cache_view,
        num_kv_heads=local_kv_heads,
        head_dim=cfg.head_dim,
        kv_layout=cfg.kv_layout,
    )
    if cfg.sliding_window is None:

        def fake_full(kv_cache, query, *args):
            return torch.empty_like(kv_cache), torch.empty_like(query)

        return _LayerOps(
            full_attention=_register(
                "full_attention",
                cp_prefill.full_attention_layer,
                fake_full,
                donate=True,
                cfg=cfg,
                cache_view=cache_view,
            )
        )

    def fake_sliding(query, keys, values, halos, seq_lens):
        page_size = halos.shape[1]
        num_window = cp_prefill.num_halo_pages(cfg.sliding_window, page_size) + 1
        send = (cfg.world, num_window, page_size, 2 * local_kv_heads, cfg.head_dim)
        return (
            torch.empty_like(query),
            query.new_empty(send, dtype=torch.int8),
            query.new_empty((num_window,), dtype=torch.int32),
        )

    def fake_window(kv_cache, *args):
        return torch.empty_like(kv_cache)

    def fake_halos(keys, values, page_size):
        halo_pages = cp_prefill.num_halo_pages(cfg.sliding_window, page_size)
        page = (page_size, 2 * cfg.num_kv_heads // 4, 4, cfg.head_dim)
        return keys.new_empty((2 * halo_pages, *page))

    mesh = _mesh(cfg.world)
    halo_exchange = sharded_jax_op(
        f"cp_prefill::halo_exchange_{len(_OPS)}",
        _halo_exchange_fn(mesh, cfg.sliding_window),
        mesh=mesh,
        input_partition_specs=(P(_AXIS), P(_AXIS), P()),
        output_partition_specs=P(_AXIS),
    )
    halo_exchange.register_fake(fake_halos)
    return _LayerOps(
        halo_exchange=halo_exchange,
        sliding_attention=_register(
            "sliding_attention",
            cp_prefill.sliding_attention_layer,
            fake_sliding,
            donate=False,
            cfg=cfg,
        ),
        window_write=_register(
            "window_write",
            cp_prefill.window_write_layer,
            fake_window,
            donate=True,
            cfg=cfg,
            cache_view=cache_view,
        ),
    )


def _layer_config(
    attn: Any, num_heads: int, num_kv_heads: int, rank: int, world: int
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
    _, local_kv_heads = cp_prefill.tp_kv_heads(num_kv_heads, world, rank)
    if local_kv_heads != impl.num_kv_heads:
        raise ValueError(
            f"{num_kv_heads} KV heads on {world} ranks give {local_kv_heads} per "
            f"rank; the layer's cache holds {impl.num_kv_heads}."
        )
    if impl.fp8_compute and (num_heads // num_kv_heads) % 4:
        raise ValueError(
            f"TPU_FP8_ATTENTION: {num_heads} query heads per {num_kv_heads} KV heads "
            "are not whole fp8 packing groups."
        )
    sliding = impl.sliding_window is not None
    return cp_prefill.LayerConfig(
        rank=rank,
        world=world,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=impl.head_size,
        sliding_window=impl.sliding_window,
        sm_scale=impl.scale,
        soft_cap=impl.logits_soft_cap,
        k_scale=attn._k_scale_float,
        v_scale=attn._v_scale_float,
        fp8_compute=impl.fp8_compute,
        kv_layout=impl.kv_layout,
        sliding_prefill_max_bkv=(
            envs.TPU_RPA_SLIDING_PREFILL_MAX_BKV if sliding else None
        ),
    )


def prepare_cp_prefill_attention(
    attn: Any, *, num_heads: int, num_kv_heads: int, rank: int, world: int
) -> None:
    """Builds the ops `cp_prefill_attention` runs for this layer.

    Call it outside any compiled region, after the KV scales are loaded and the
    backend's `initialize_kernel`.

    Args:
        attn: The layer's vLLM `Attention` module (tensor-parallel heads).
        num_heads: Query heads of the layer (all ranks').
        num_kv_heads: KV heads of the layer (all ranks').
        rank: This tensor-parallel rank.
        world: Tensor-parallel world size.
    """
    if get_vllm_model_wrapper_context().kv_cache_bundle is not None:
        raise NotImplementedError(
            "Context-parallel prefill attention does not support the bundled KV cache."
        )
    cfg = _layer_config(attn, num_heads, num_kv_heads, rank, world)
    if cfg not in _OPS:
        _OPS[cfg] = _build_ops(cfg)
        logger.info_once(
            "Context-parallel prefill attention on %d ranks: %d query / %d KV heads, "
            "sliding window %s, %s QK/PV.",
            world,
            num_heads,
            num_kv_heads,
            cfg.sliding_window,
            "fp8" if cfg.fp8_compute else "bf16",
        )
    attn._cp_prefill_config = cfg


def _waited(t: torch.Tensor) -> torch.Tensor:
    return funcol.wait_tensor(t) if isinstance(t, funcol.AsyncCollectiveTensor) else t


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block d of x's leading axis goes to rank d; block s of the result came
    from rank s."""
    return _waited(funcol.all_to_all_single(x.contiguous(), None, None, group))


def cp_prefill_attention(
    attn: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: cp_prefill.ZigzagPlan,
    rank: int,
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
        rank: This tensor-parallel rank.

    Returns:
        [rows, num_heads * head_dim] attention output.
    """
    cfg: cp_prefill.LayerConfig | None = getattr(attn, "_cp_prefill_config", None)
    if cfg is None or cfg.rank != rank or cfg.world != plan.world:
        raise RuntimeError(
            f"{attn.layer_name}: call prepare_cp_prefill_attention for rank {rank} "
            f"of {plan.world} before tracing."
        )
    rows, d = query.shape[0], cfg.head_dim
    if rows != 2 * plan.piece_tokens:
        raise ValueError(f"{rows} rows are not the two pieces of {plan}")
    ops = _OPS[cfg]
    attn_metadata, layer, kv_cache, _ = get_attention_context(attn.layer_name)
    if kv_cache.numel() == 0:
        # The memory-profiling run, before caches exist; the backend does the same.
        return torch.ones_like(query)
    group = get_tp_group().device_group
    key, value = quantize_kv(
        layer.impl.kv_cache_quantized_dtype,
        key.view(rows, cfg.num_kv_heads, d),
        value.view(rows, cfg.num_kv_heads, d),
        cfg.k_scale,
        cfg.v_scale,
    )
    query = query.view(rows, cfg.num_heads, d)
    seq_lens = attn_metadata.seq_lens
    block_tables = attn_metadata.block_tables
    if cfg.sliding_window is None:
        # fp8 crosses the collective as raw bytes, in the layout of
        # cp_prefill.pack_kv_pages: per token K_0 V_0 K_1 V_1 ... in words of 4.
        kv = torch.cat([key.view(torch.int8), value.view(torch.int8)], dim=-1)
        pages = kv.reshape(
            rows // plan.page_size, plan.page_size, 2 * cfg.num_kv_heads // 4, 4, d
        )
        gathered = _waited(funcol.all_gather_single(pages, 0, group))
        new_kv_cache, out = ops.full_attention(
            kv_cache, query, gathered, seq_lens, block_tables
        )
    else:
        halos = ops.halo_exchange(key, value, plan.page_size)
        out, send, owners = ops.sliding_attention(query, key, value, halos, seq_lens)
        received = _all_to_all(send, group)
        new_kv_cache = ops.window_write(
            kv_cache, received, owners, seq_lens, block_tables
        )
    # The ops donate the cache; the copy_ folds into that alias when compiled.
    kv_cache.copy_(new_kv_cache)
    return out.reshape(rows, cfg.num_heads * d)
