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
"""The torch entry points (`write_with_dcp`, `forward_with_dcp`) over
simulated DCP ranks vs dense sliding-window attention.

Each rank runs in its own thread against a group whose `all_gather` /
`reduce_scatter` meet at a barrier, so the rank-major query all-gather, the
KV-head selection of the context write, the LSE merge back to each rank's
heads and the block's fp8 round trip all run as on TPU. The ops are the
JAX functions the jax_ops wrap (the jax_ops need a TPU device).
"""

import threading
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.models.minimax_m3.experimental.test_dcp_draft_attention import (
    HEAD_DIM,
    PAGE,
    _cache,
    _dense_reference,
)
from vllm_torchtpu.models.minimax_m3.experimental.kernels import (
    dcp_draft_attention as cda,
)
from vllm_torchtpu.models.minimax_m3.experimental.ops import (
    dcp_draft_attention as ops_mod,
)

WINDOW = 512
MAX_QUERY_LEN = 7
HEADS_PER_RANK = 4
SM_SCALE = HEAD_DIM**-0.5

_JNP = {
    torch.bfloat16: (torch.int16, jnp.bfloat16),
    torch.float8_e4m3fn: (torch.uint8, jnp.float8_e4m3fn),
}


def _to_jax(t: torch.Tensor):
    if t.dtype in _JNP:
        bits, dtype = _JNP[t.dtype]
        return jnp.asarray(t.view(bits).numpy()).view(dtype)
    return jnp.asarray(t.numpy())


def _to_torch(a) -> torch.Tensor:
    for t_dtype, (bits, dtype) in _JNP.items():
        if a.dtype == dtype:
            raw = np.asarray(a.view(jnp.int16 if bits == torch.int16 else jnp.uint8))
            return torch.from_numpy(raw.copy()).view(t_dtype)
    return torch.from_numpy(np.asarray(a).copy())


class _Group:
    """A DCP group of `world_size` thread-ranks meeting at a barrier."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._slots: list = [None] * world_size
        self._local = threading.local()

    def bind(self, rank: int) -> None:
        self._local.rank = rank

    @property
    def rank_in_group(self) -> int:
        return self._local.rank

    def _exchange(self, tensor):
        self._slots[self.rank_in_group] = tensor
        self._barrier.wait()
        parts = list(self._slots)
        self._barrier.wait()
        return parts

    def all_gather(self, tensor, dim=0):
        return torch.cat(self._exchange(tensor), dim=dim)

    def reduce_scatter(self, tensor, dim=-1):
        total = torch.stack(self._exchange(tensor)).sum(0)
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, self.rank_in_group * width, width)


def _ops(dcp, kv_heads, k_scale, v_scale, causal):
    def write(kv_cache, key, value, seq_lens, tables, qsl, rank):
        out = cda.write_kv(
            *map(_to_jax, (kv_cache, key, value, seq_lens, tables, qsl)),
            dcp=dcp,
            rank=jnp.int32(int(rank[0])),
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return _to_torch(out)

    def attend(q_all, kv_cache, seq_lens, tables, qsl, rank):
        out, lse = cda.attend_cached_flash(
            *map(_to_jax, (q_all, kv_cache, seq_lens, tables, qsl)),
            kv_heads=kv_heads,
            max_query_len=MAX_QUERY_LEN,
            sliding_window=WINDOW,
            sm_scale=SM_SCALE,
            dcp=dcp,
            rank=jnp.int32(int(rank[0])),
            k_scale=k_scale,
            v_scale=v_scale,
            interpret=True,
        )
        return _to_torch(out), _to_torch(lse)

    def block(q, key, value, seq_lens, qsl, ctx_out, ctx_lse):
        out, lse = cda.attend_block(
            *map(_to_jax, (q, key, value, seq_lens, qsl)),
            max_query_len=MAX_QUERY_LEN,
            sm_scale=SM_SCALE,
            causal=causal,
        )
        ctx = _to_jax(ctx_out).astype(out.dtype)
        return _to_torch(cda.merge(ctx, _to_jax(ctx_lse), out, lse)[0])

    return dict(write=write, attend=attend, block=block)


def _run_ranks(dcp, fn):
    group = _Group(dcp)
    results: list = [None] * dcp
    errors: list = []

    def run(rank):
        group.bind(rank)
        try:
            results[rank] = fn(rank)
        except BaseException as e:  # surface in the main thread
            errors.append(e)
            group._barrier.abort()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ops_mod, "get_dcp_group", lambda: group)
        threads = [threading.Thread(target=run, args=(r,)) for r in range(dcp)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if errors:
        raise errors[0]
    return results


@pytest.mark.parametrize(
    ("dcp", "replicas", "cache_dtype", "causal"),
    [
        (2, 1, torch.bfloat16, False),
        (2, 2, torch.bfloat16, True),
        (2, 1, torch.float8_e4m3fn, False),
    ],
)
def test_write_and_forward_with_dcp_match_dense(dcp, replicas, cache_dtype, causal):
    rng = np.random.default_rng(dcp * 10 + replicas)
    kv_heads = max(1, dcp // replicas)  # KV heads every rank's cache holds
    total_kv_heads = max(1, dcp // replicas)
    scale = None if cache_dtype == torch.bfloat16 else 0.5
    ctx = np.array([300, 37, 0, 600], np.int32)
    q_len = np.array([7, 3, 7, 7], np.int32)
    num_seqs = len(ctx)
    pages_per_seq = -(-int((ctx + q_len).max()) // PAGE) + 1
    logical = -(-pages_per_seq // dcp) + 1
    num_pages = num_seqs * logical + 1
    tables = torch.from_numpy(
        rng.permutation(num_pages)[: num_seqs * logical].astype(np.int32)
    )
    cu_ctx = torch.from_numpy(np.concatenate([[0], np.cumsum(ctx)]).astype(np.int32))
    cu_q = torch.from_numpy(np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32))

    def randn(*shape):
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    # Global K/V per KV head; rank r holds KV head r // replicas.
    ctx_k, ctx_v = (
        randn(int(ctx.sum()), total_kv_heads, HEAD_DIM),
        randn(int(ctx.sum()), total_kv_heads, HEAD_DIM),
    )
    num_q = int(q_len.sum())
    blk_k, blk_v = (
        randn(num_q, total_kv_heads, HEAD_DIM),
        randn(num_q, total_kv_heads, HEAD_DIM),
    )
    query = randn(num_q, dcp * HEADS_PER_RANK, HEAD_DIM).to(torch.bfloat16)
    ops = _ops(dcp, kv_heads, scale, scale, causal)
    cache_shape = _cache(num_pages, kv_heads, jnp.dtype(_JNP[cache_dtype][1])).shape

    def rank_step(rank):
        head = rank // replicas
        own = slice(head, head + 1)
        cache = torch.zeros(cache_shape, dtype=cache_dtype)
        dcp_rank = torch.tensor([rank], dtype=torch.int32)
        ops_mod.write_with_dcp(
            key=ctx_k[:, own].to(torch.bfloat16),
            value=ctx_v[:, own].to(torch.bfloat16),
            kv_cache=cache,
            attn_metadata=SimpleNamespace(
                seq_lens=torch.from_numpy(ctx),
                block_tables=tables,
                query_start_loc=cu_ctx,
            ),
            dcp_rank=dcp_rank,
            dcp=dcp,
            kv_head_replicas=replicas,
            ops=ops,
        )
        heads = slice(rank * HEADS_PER_RANK, (rank + 1) * HEADS_PER_RANK)
        return ops_mod.forward_with_dcp(
            query=query[:, heads].contiguous(),
            key=blk_k[:, own].to(torch.bfloat16),
            value=blk_v[:, own].to(torch.bfloat16),
            kv_cache=cache,
            attn_metadata=SimpleNamespace(
                seq_lens=torch.from_numpy(ctx + q_len),
                block_tables=tables,
                query_start_loc=cu_q,
            ),
            dcp_rank=dcp_rank,
            dcp=dcp,
            k_scale=scale,
            v_scale=scale,
            ops=ops,
        )

    outputs = _run_ranks(dcp, rank_step)

    def stored(x):  # the bytes the cache holds, read back
        x = x.to(torch.bfloat16).float()
        if scale is None:
            return x.numpy()
        limit = torch.finfo(cache_dtype).max
        q = (x / scale).clamp(-limit, limit).to(cache_dtype).float() * scale
        return q.numpy()

    keys, values = [], []
    for s in range(num_seqs):
        c0, c1, q0, q1 = cu_ctx[s], cu_ctx[s + 1], cu_q[s], cu_q[s + 1]
        keys.append(np.concatenate([stored(ctx_k[c0:c1]), stored(blk_k[q0:q1])]))
        values.append(np.concatenate([stored(ctx_v[c0:c1]), stored(blk_v[q0:q1])]))
    ref = _dense_reference(
        query.float().numpy(),
        keys,
        values,
        ctx + q_len,
        cu_q.numpy(),
        WINDOW,
        SM_SCALE,
        causal,
    )
    for rank, out in enumerate(outputs):
        heads = slice(rank * HEADS_PER_RANK, (rank + 1) * HEADS_PER_RANK)
        np.testing.assert_allclose(
            out.float().numpy(), ref[:, heads], rtol=3e-2, atol=3e-2
        )


def test_op_names_are_exact_and_registrable():
    """Scales like 1e-05 make valid op names, and scales that differ past
    the 8th decimal build distinct ops."""
    assert ops_mod._name_float(1e-05) == "0x1_4f8b588e368f1pm17"
    assert ops_mod._name_float(None) == "none"
    kw = dict(dcp=2, kv_heads=1, max_query_len=7, sliding_window=WINDOW, causal=False)
    a = ops_mod.build_ops(sm_scale=0.088388347, k_scale=1e-05, v_scale=1e-05, **kw)
    b = ops_mod.build_ops(sm_scale=0.0883883471, k_scale=1e-05, v_scale=1e-05, **kw)
    assert a["attend"] is not b["attend"] and a["block"] is not b["block"]


def test_build_ops_rejects_query_block_wider_than_window():
    with pytest.raises(AssertionError):
        ops_mod.build_ops(
            dcp=2,
            kv_heads=1,
            max_query_len=8,
            sliding_window=4,
            sm_scale=SM_SCALE,
            causal=False,
            k_scale=None,
            v_scale=None,
        )
