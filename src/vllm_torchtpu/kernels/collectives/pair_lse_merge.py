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
"""LSE merge of two ranks' partial attentions in one exchange.

`dcp_comm.cp_lse_ag_out_rs` over a group of two ranks all-gathers the LSEs,
weights each partial and reduce-scatters the heads: three collectives in a row
(an all-gather, a reduce-scatter and its async done). Here each rank sends its
whole partial (bf16 ``[T, 2H, D]``) and LSE (f32 ``[T, 2H, 1]``) to its peer in
one round of remote copies and merges its own H heads locally, with the same
f32 math: ``m = logsumexp(lse_0, lse_1)``, ``sum_i where(w_i > 0, out_i * w_i,
0)`` for ``w_i = exp(lse_i - m)``, the two terms added in either order (exact).

The kernel runs under shard_map over a 1-D mesh in which every rank has one
peer (pairs disjoint). `exchange_xla` is the same merge on a collective-permute
for shapes the kernel does not take.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
COLLECTIVE_ID = 0
_MESH = pl.DeviceIdType.MESH
# Partial bytes the kernel takes: draft decode blocks (64 rows at 16 heads of
# 128). VMEM holds ~24x this: the partials, the f32 LSEs whose 1-wide lane
# dim pads to 128 (twice a bf16 partial each) and the f32 temporaries.
MAX_BYTES = 256 << 10


def _merge(out_me, lse_me, out_peer, lse_peer, block, heads):
    """This rank's ``heads`` heads (block ``block``) of the merged partials,
    f32 ``[T, 2H, D]`` / ``[T, 2H, 1]`` in; f32 ``[T, H, D]``, ``[T, H, 1]``."""
    m = jnp.maximum(lse_me, lse_peer)
    m = jnp.where(jnp.abs(m) == jnp.inf, 0.0, m)  # torch.logsumexp's guard
    lse = jnp.log(jnp.exp(lse_me - m) + jnp.exp(lse_peer - m)) + m

    def weighted(out, part):
        w = jnp.exp(part - lse)
        return jnp.where(w > 0, out * w, 0.0)

    out = weighted(out_me, lse_me) + weighted(out_peer, lse_peer)
    pick = jnp.asarray(block, jnp.int32) == 0
    return (
        jnp.where(pick, out[:, :heads], out[:, heads:]),
        jnp.where(pick, lse[:, :heads], lse[:, heads:]),
    )


def supported(out_shape, dtype) -> bool:
    t, two_h, d = out_shape
    return (
        two_h % 16 == 0
        and d % 128 == 0
        and t * two_h * d * jnp.dtype(dtype).itemsize <= MAX_BYTES
    )


@functools.cache
def _call(shape: tuple[int, int, int], dtype, interpret: Any = False):
    t, two_h, d = shape
    heads = two_h // 2

    def kernel(meta_ref, out_ref, lse_ref, o_ref, ol_ref, r_out, r_lse, send, recv):
        peer, block = meta_ref[0], meta_ref[1]
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier, inc=1, device_id=(peer,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, 1)
        copies = [
            pltpu.make_async_remote_copy(
                src,
                dst,
                send.at[k],
                recv.at[k],
                device_id=(peer,),
                device_id_type=_MESH,
            )
            for k, (src, dst) in enumerate(((out_ref, r_out), (lse_ref, r_lse)))
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait_recv()
        out, lse = _merge(
            out_ref[...].astype(F32),
            lse_ref[...],
            r_out[...].astype(F32),
            r_lse[...],
            block,
            heads,
        )
        o_ref[...] = out.astype(o_ref.dtype)
        ol_ref[...] = lse
        for copy in copies:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((t, heads, d), dtype),
            jax.ShapeDtypeStruct((t, heads, 1), F32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem],
            out_specs=[vmem, vmem],
            scratch_shapes=[
                pltpu.VMEM(shape, dtype),
                pltpu.VMEM((t, two_h, 1), F32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=32 * MAX_BYTES + (4 << 20),
        ),
        interpret=interpret,
        name=f"pair_lse_merge-t_{t}-h_{two_h}-d_{d}",
    )


def _block_partial(q, k, v, *, q_block, valid_rows, sm_scale, causal):
    """Attention of each request's query rows over its own block's K/V (the
    rows are requests' blocks of ``q_block`` back to back, ``valid_rows`` of
    them real): f32 ``q [T, H, D]``, ``k``/``v [T, D]`` in; out f32 ``[T, H,
    D]`` and lse ``[T, H, 1]`` (``-inf`` with a zero out where a row sees no
    key), the math of cp_draft_attention.attend_block. Every head at once:
    the query rows as ``[T * H, D]``."""
    t, heads, d = q.shape
    rows = t * heads
    tok = lax.broadcasted_iota(jnp.int32, (rows, t), 0) // heads
    col = lax.broadcasted_iota(jnp.int32, (rows, t), 1)
    visible = (tok // q_block == col // q_block) & (tok < valid_rows)
    visible &= col < valid_rows
    if causal:
        visible &= col <= tok
    s = lax.dot_general(
        q.reshape(rows, d), k, (((1,), (1,)), ((), ())), preferred_element_type=F32
    )
    s = jnp.where(visible, s * sm_scale, -jnp.inf)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(m == -jnp.inf, 0.0, m)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)  # noqa: E741
    l_safe = jnp.where(l > 0, l, 1.0)
    out = jnp.dot(p, v, preferred_element_type=F32) / l_safe
    lse = jnp.where(l > 0, m_safe + jnp.log(l_safe), -jnp.inf)
    return out.reshape(t, heads, d), lse.reshape(t, heads, 1)


def _merge_two(out_a, lse_a, out_b, lse_b):
    """cp_draft_attention.merge (f32): ``[T, H, D]`` / ``[T, H, 1]``."""
    m = jnp.maximum(lse_a, lse_b)
    m_safe = jnp.where(jnp.abs(m) == jnp.inf, 0.0, m)
    wa = jnp.exp(lse_a - m_safe)
    wb = jnp.exp(lse_b - m_safe)
    norm = wa + wb
    norm_safe = jnp.where(norm > 0, norm, 1.0)
    return (wa * out_a + wb * out_b) / norm_safe


def _round_trip(x, scale, cache_dtype, dtype):
    """x as the draft KV cache holds it (quantized with ``scale`` to
    ``cache_dtype``, when given), then in the query dtype ``dtype``; f32."""
    if scale is not None:
        x = (x / scale).astype(cache_dtype).astype(F32) * scale
    return x.astype(dtype).astype(F32)


def _rows_to_column(row, rows: int):
    """A lane-major ``[1, rows]`` row as a ``[rows, 1]`` column."""
    eye = lax.broadcasted_iota(jnp.int32, (rows, rows), 0) == lax.broadcasted_iota(
        jnp.int32, (rows, rows), 1
    )
    return jnp.sum(
        jnp.where(eye, jnp.broadcast_to(row, (rows, rows)), 0.0), axis=1, keepdims=True
    )


@functools.cache
def _call_block(
    shape: tuple[int, int, int],
    dtype,
    num_reqs: int,
    q_block: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None,
    v_scale: float | None,
    cache_dtype,
    interpret: Any = False,
):
    t, two_h, d = shape
    heads = two_h // 2
    rows = q_block * two_h
    valid_rows = num_reqs * q_block

    def kernel(
        meta_ref,
        out_ref,
        lse_ref,
        q_ref,
        k_ref,
        v_ref,
        o_ref,
        lse_me,
        r_out,
        r_lse,
        send,
        recv,
    ):
        peer, block = meta_ref[0], meta_ref[1]
        # The attention kernel's per-request lane-major LSEs as [T, 2H, 1].
        for r in range(num_reqs):
            column = _rows_to_column(lse_ref[r], rows)
            lse_me[pl.ds(r * q_block, q_block)] = column.reshape(q_block, two_h, 1)
        if valid_rows < t:
            lse_me[pl.ds(valid_rows, t - valid_rows)] = jnp.full(
                (t - valid_rows, two_h, 1), -jnp.inf, F32
            )
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier, inc=1, device_id=(peer,), device_id_type=_MESH)
        pl.semaphore_wait(barrier, 1)
        copies = [
            pltpu.make_async_remote_copy(
                src,
                dst,
                send.at[k],
                recv.at[k],
                device_id=(peer,),
                device_id_type=_MESH,
            )
            for k, (src, dst) in enumerate(((out_ref, r_out), (lse_me, r_lse)))
        ]
        for copy in copies:
            copy.start()
        # The block's own attention runs while the partials cross.
        key = _round_trip(k_ref[...].astype(F32), k_scale, cache_dtype, dtype)
        value = _round_trip(v_ref[...].astype(F32), v_scale, cache_dtype, dtype)
        out_b, lse_b = _block_partial(
            q_ref[...].astype(F32),
            key,
            value,
            q_block=q_block,
            valid_rows=valid_rows,
            sm_scale=sm_scale,
            causal=causal,
        )
        for copy in copies:
            copy.wait_recv()
        ctx, ctx_lse = _merge(
            out_ref[...].astype(F32),
            lse_me[...],
            r_out[...].astype(F32),
            r_lse[...],
            block,
            heads,
        )
        # Rounded where the unfused path hands bf16 tensors between ops.
        ctx = ctx.astype(dtype).astype(F32)
        out_b = out_b.astype(dtype).astype(F32)
        merged = _merge_two(ctx, ctx_lse, out_b, lse_b)
        o_ref[...] = merged.astype(o_ref.dtype)
        for copy in copies:
            copy.wait_send()

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((t, heads, d), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem] * 5,
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((t, two_h, 1), F32),
                pltpu.VMEM(shape, dtype),
                pltpu.VMEM((t, two_h, 1), F32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            vmem_limit_bytes=48 * MAX_BYTES + (4 << 20),
        ),
        interpret=interpret,
        name=f"pair_lse_merge_block-t_{t}-h_{two_h}-d_{d}",
    )


def exchange_block(
    out: jax.Array,
    lse: jax.Array,
    meta: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    q_block: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None = None,
    v_scale: float | None = None,
    cache_dtype=None,
    interpret: Any = False,
) -> jax.Array:
    """`exchange`, then this rank's heads merged with the attention of the
    draft block's own K/V (cp_draft_attention.attend_block + merge), in one
    kernel; call under shard_map. ``out [T, 2H, D]`` and ``lse [R, 1, Q *
    2H]`` as the draft attention kernel leaves them
    (cp_draft_attention.attend_pages ``native_lse``: request r's rows
    ``r * Q`` on, padding rows past ``R * Q``); ``q [T, H, D]`` this rank's
    heads; ``k`` / ``v [T, 1, D]`` the block's new K/V, quantized like the
    cache (``k_scale``, ``v_scale``, ``cache_dtype``) in the kernel. Returns
    ``[T, H, D]`` in out's dtype. ``interpret`` as in `exchange`."""
    t, two_h, d = out.shape
    num_reqs = lse.shape[0]
    assert lse.shape[1:] == (1, q_block * two_h), (lse.shape, q_block, two_h)
    call = _call_block(
        out.shape,
        out.dtype,
        num_reqs,
        q_block,
        sm_scale,
        causal,
        k_scale,
        v_scale,
        cache_dtype,
        interpret,
    )
    return call(
        meta.reshape(2).astype(jnp.int32),
        out,
        lse.astype(F32),
        q,
        k.reshape(t, d),
        v.reshape(t, d),
    )


def exchange(out: jax.Array, lse: jax.Array, meta: jax.Array, interpret: Any = False):
    """Kernel merge; call under shard_map. out ``[T, 2H, D]``, lse f32
    ``[T, 2H]``, meta int32 ``[2]`` = (peer's mesh index, this rank's block
    0 / 1). Returns ``([T, H, D]`` in out's dtype, f32 ``[T, H])``.
    ``interpret``: `pl.pallas_call`'s (tests: ``pltpu.InterpretParams()``)."""
    t, two_h, _ = out.shape
    o, ol = _call(out.shape, out.dtype, interpret)(
        meta.reshape(2).astype(jnp.int32), out, lse.astype(F32).reshape(t, two_h, 1)
    )
    return o, ol.reshape(t, two_h // 2)


def exchange_xla(out, lse, block, axis: str, perm):
    """`exchange` on a collective-permute over `axis` with the static pairs
    ``perm`` [(src, dst), ...]; block (traced int32) as in `exchange`."""
    t, two_h, _ = out.shape
    lse = lse.astype(F32).reshape(t, two_h, 1)
    out_peer = lax.ppermute(out, axis, perm)
    lse_peer = lax.ppermute(lse, axis, perm)
    o, ol = _merge(
        out.astype(F32), lse, out_peer.astype(F32), lse_peer, block, two_h // 2
    )
    return o.astype(out.dtype), ol.reshape(t, two_h // 2)


def reference(outs, lses, block):
    """cp_lse_ag_out_rs's math for the rank of group index ``block``, from
    both ranks' ``outs [2][T, 2H, D]`` and ``lses [2][T, 2H]``."""
    lses = jnp.stack([x.astype(F32) for x in lses])
    merged = jax.nn.logsumexp(lses, axis=0)
    terms = []
    for out, lse in zip(outs, lses):
        w = jnp.exp(lse - merged)[..., None]
        terms.append(jnp.where(w > 0, out.astype(F32) * w, 0.0))
    heads = outs[0].shape[1] // 2
    total = terms[0] + terms[1]
    sl = slice(block * heads, (block + 1) * heads)
    return total[:, sl].astype(outs[0].dtype), merged[:, sl]
