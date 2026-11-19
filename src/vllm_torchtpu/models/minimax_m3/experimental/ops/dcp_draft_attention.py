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
"""Torch entry points of the DCP-sharded sliding-window draft attention.

A drafter whose KV cache is DCP-sharded like the target's (instead of whole
on every rank) writes its context K/V with `write_with_dcp` and attends with
`forward_with_dcp`: an all-gather of the queries, the Pallas paged
sliding-window flash attention over the rank's owned pages, an LSE merge
across the group, and the query block's own attention merged by LSE. The
JAX / Pallas kernels are in `experimental/kernels/dcp_draft_attention.py`.
"""

import functools
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.models.minimax_m3.experimental.kernels.dcp_draft_attention import (
    attend_block,
    attend_cached_flash,
    merge,
    write_kv,
)

# ---------------------------------------------------------------------------
# Torch entry points (one jax_op per static configuration)
#
# Build them eagerly with `build_ops` (from module setup): `pallas.jax_op`
# inspects the function signature, which Dynamo cannot trace inside a
# compiled forward.
# ---------------------------------------------------------------------------


def _name_float(value: float | None) -> str:
    """A float rendered exactly and safely inside a torch op name: its hex
    form (two scales never share a name; ``1e-05`` keeps no ``-``)."""
    if value is None:
        return "none"
    hexed = float(value).hex()
    return hexed.replace(".", "_").replace("-", "m").replace("+", "p")


def build_ops(
    *,
    dcp: int,
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None,
    v_scale: float | None,
) -> dict[str, Any]:
    """The write, cached-attention and block ops of one draft layer."""
    # `attend_block` applies no window inside the query block.
    assert max_query_len <= sliding_window, (max_query_len, sliding_window)
    return dict(
        write=_write_op(dcp, k_scale, v_scale),
        attend=_attend_op(
            kv_heads, max_query_len, sliding_window, sm_scale, dcp, k_scale, v_scale
        ),
        block=_block_op(max_query_len, sm_scale, causal),
    )


@functools.cache
def _write_op(dcp: int, k_scale: float | None, v_scale: float | None):
    def _write(
        kv_cache: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> jax.Array:
        return write_kv(
            kv_cache,
            key,
            value,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp=dcp,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )

    name = (
        f"pallas::minimax_m3_dcp_draft_write_d{dcp}"
        f"_k{_name_float(k_scale)}_v{_name_float(v_scale)}"
    )
    op = pallas.jax_op(name, _write, donate_argnums=(0,))
    op.register_fake(lambda kv_cache, *args: torch.empty_like(kv_cache))
    return op


@functools.cache
def _attend_op(
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    k_scale: float | None,
    v_scale: float | None,
):
    def _attend(
        q_all: jax.Array,
        kv_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return attend_cached_flash(
            q_all,
            kv_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            kv_heads=kv_heads,
            max_query_len=max_query_len,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            dcp=dcp,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )

    name = (
        f"pallas::minimax_m3_dcp_draft_flash_g{kv_heads}_q{max_query_len}"
        f"_w{sliding_window}_s{_name_float(sm_scale)}_d{dcp}"
        f"_k{_name_float(k_scale)}_v{_name_float(v_scale)}"
    )
    op = pallas.jax_op(name, _attend)
    op.register_fake(
        lambda q, *args: (
            torch.empty_like(q),
            torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        )
    )
    return op


@functools.cache
def _block_op(max_query_len: int, sm_scale: float, causal: bool):
    def _block(
        q: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        query_start_loc: jax.Array,
        ctx_out: jax.Array,
        ctx_lse: jax.Array,
    ) -> jax.Array:
        out, lse = attend_block(
            q,
            key,
            value,
            seq_lens,
            query_start_loc,
            max_query_len=max_query_len,
            sm_scale=sm_scale,
            causal=causal,
        )
        return merge(ctx_out.astype(out.dtype), ctx_lse, out, lse)[0]

    name = (
        "pallas::minimax_m3_dcp_draft_block"
        f"_q{max_query_len}_s{_name_float(sm_scale)}_c{int(causal)}"
    )
    op = pallas.jax_op(name, _block)
    op.register_fake(lambda q, *args: torch.empty_like(q))
    return op


def write_with_dcp(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    dcp: int,
    kv_head_replicas: int,
    ops: dict[str, Any],
) -> None:
    """Write the drafter's context K/V (own KV head per rank) into the
    DCP-sharded cache: each rank keeps the tokens of its blocks."""
    group = get_dcp_group()
    assert group is not None, "DCP group is not initialized"
    if dcp > kv_head_replicas:
        gathered = group.all_gather(torch.cat([key, value], dim=1).contiguous(), dim=1)
        num_tokens, _, head_dim = gathered.shape
        per_rank = gathered.view(num_tokens, dcp, 2, head_dim)[:, ::kv_head_replicas]
        key, value = per_rank[:, :, 0].contiguous(), per_rank[:, :, 1].contiguous()
    new_cache = ops["write"](
        kv_cache,
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    kv_cache.copy_(new_cache)


def _round_trip(x: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    """``x`` as the cache stores it and attention reads it back: scaled,
    clipped to an fp8 dtype's finite range (as `write_kv`'s quantization),
    cast to ``dtype``, widened and rescaled."""
    x = x.float() / scale
    if dtype.is_floating_point and dtype.itemsize == 1:
        limit = torch.finfo(dtype).max
        x = x.clamp(-limit, limit)
    return x.to(dtype).float() * scale


def forward_with_dcp(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    dcp: int,
    k_scale: float | None,
    v_scale: float | None,
    ops: dict[str, Any],
) -> torch.Tensor:
    """The drafter's attention of its query block (``query [T, H, D]``, own
    heads; ``key``/``value [T, 1, D]``, own KV head) over the window of a
    DCP-sharded cache plus the block itself. The block is not written.
    ``k_scale``/``v_scale`` are the layer's KV scales when the cache is
    quantized (both set), else None."""

    group = get_dcp_group()
    assert group is not None, "DCP group is not initialized"
    # Every rank attends all the group's query heads over its own share of the
    # cached window; the block's K/V stay per rank (own KV head), so only the
    # queries cross the group. ``[T, dcp * H, D]``, rank-major.
    q_all = group.all_gather(query.contiguous(), dim=1)
    ctx_out, ctx_lse = ops["attend"](
        q_all,
        kv_cache,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    ctx_out, ctx_lse = cp_lse_ag_out_rs(ctx_out, ctx_lse, group, return_lse=True)
    # The block's K/V as the cache would hold them (quantized), own KV head.
    if kv_cache.dtype != key.dtype and k_scale is not None and v_scale is not None:
        key = _round_trip(key, kv_cache.dtype, k_scale)
        value = _round_trip(value, kv_cache.dtype, v_scale)
    return ops["block"](
        query,
        key.to(query.dtype),
        value.to(query.dtype),
        attn_metadata.seq_lens,
        attn_metadata.query_start_loc,
        ctx_out,
        ctx_lse,
    )
