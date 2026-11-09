# Copyright 2026 The SGLang Authors.
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
#
# Adapted from sglang-torchtpu sglang_tpu/multimodal_gen/diffusion_ring_attention.py
# (Apache-2.0).
"""TPU Pallas diffusion ring attention with ICI DMA hops and hybrid fixed-m."""

import functools
import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

LOG2_E = 1.44269504
FIXED_M_RECENTER = 88.0
FIXED_M_SAFE_BOUND = 213.0
FIXED_M_RING_SAFE_BOUND = FIXED_M_SAFE_BOUND / 2.0

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
SV_DIM_NUMBERS = (((0,), (0,)), ((), ()))

_COLLECTIVE_ID = 8
VMEM_LIMIT_BYTES = 64 * 2**20
_LANE = 128

_DEFAULT_BLOCK_Q = 512
_DEFAULT_BLOCK_KV = 2048
_UNTUNED_VMEM_BUDGET = 56 * 2**20
_V7X_VMEM_CAP_BYTES = int(63.94 * 2**20)
_HYBRID_RAGGED_SCORE_LIMIT = 8_700_000

# Tuned (block_q, block_kv, block_kv_compute, block_q_sub) tiles on TPU v7x
# keyed by (ring_size, shard_len).
_TUNED_BLOCKS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (8, 4950): (5120, 2560, 2560, 2560),
    (8, 9450): (9600, 2560, 2560, 3200),
    (8, 13950): (14080, 3584, 3584, 2816),
    (8, 18450): (18816, 2688, 2688, 2688),
    (8, 22950): (23040, 2560, 2560, 2560),
    (4, 9900): (9984, 2560, 2560, 3328),
    (4, 18900): (19200, 3328, 3328, 3200),
    (4, 27900): (28160, 2816, 2816, 2816),
    (4, 36900): (18560, 8192, 2048, 3712),
    (4, 45900): (23040, 2560, 2560, 2560),
    (8, 2145): (2304, 2048, 2048, 1152),
    (8, 4095): (4096, 4096, 4096, 2048),
    (8, 6045): (6144, 6144, 2048, 3072),
    (8, 7995): (8192, 2048, 2048, 4096),
    (8, 9945): (9984, 2048, 2048, 3328),
    (4, 4290): (4352, 4352, 4352, 2176),
    (4, 8190): (8192, 8192, 4096, 2048),
    (4, 12090): (12160, 2432, 2432, 2432),
    (4, 15990): (16000, 2816, 2816, 3200),
    (4, 19890): (19968, 5632, 5632, 1536),
}

# Pure online-softmax tiles when use_fixed_m is False.
_TUNED_ONLINE_BLOCKS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (4, 8190): (8448, 4096, 4096, 2816),
    (4, 12090): (12544, 6144, 6144, 1792),
    (4, 15990): (16128, 6144, 2048, 5376),
    (4, 19890): (19968, 3328, 3328, 3328),
    (4, 9900): (9984, 3584, 3584, 3328),
    (4, 18900): (19200, 2816, 2816, 3840),
    (4, 36900): (37632, 1024, 1024, 5376),
    (4, 45900): (23040, 3072, 3072, 2304),
}


def auto_block_q_sub(
    block_q: int, lane: int = _LANE, max_q_sub: int | None = None
) -> int:
    """Return the largest lane-multiple divisor of block_q within ceiling."""
    ceiling = max(lane, (block_q // 4) // lane * lane)
    if max_q_sub is not None:
        ceiling = min(ceiling, max_q_sub)
    q_sub = lane
    d = lane
    while d <= block_q:
        if block_q % d == 0:
            if d <= ceiling:
                q_sub = d
            else:
                break
        d += lane
    return q_sub


_SINGLE_BUFFER_Q_MIN = 4608


def is_single_buffered_q(block_q: int, shard_len: int) -> bool:
    """Return True when Q/Out use pl.Buffered(1) on whole-shard tiles > 4608."""
    return block_q >= shard_len and block_q > _SINGLE_BUFFER_Q_MIN


def estimate_vmem_bytes(
    block_q: int,
    block_q_sub: int,
    block_kv: int,
    block_kv_compute: int | None = None,
    *,
    single_buffered_q: bool = True,
    head_dim: int = 128,
) -> int:
    """Estimate scoped-VMEM usage in bytes on TPU v7x."""
    if block_kv_compute is None:
        block_kv_compute = block_kv
    scale = head_dim / 128.0
    ml_bytes_per_row = 2.0 * NUM_SUBLANES * 4.0
    per_row_128 = 1128.8 if single_buffered_q else 1661.9
    per_row = ml_bytes_per_row + (per_row_128 - ml_bytes_per_row) * scale
    est = (
        per_row * block_q
        + 4.28 * block_q_sub * block_kv_compute
        + 1274.0 * scale * block_kv
        - 5.99 * 2**20 * scale
    )
    return max(0, int(est + 0.999999))


def _untuned_q_sub(
    block_q: int,
    block_kv: int,
    head_dim: int,
    *,
    single_buffered_q: bool = True,
) -> int:
    """Choose block_q_sub so the score tile fits within the VMEM budget."""
    min_vmem = estimate_vmem_bytes(
        block_q,
        _LANE,
        block_kv,
        block_kv,
        single_buffered_q=single_buffered_q,
        head_dim=head_dim,
    )
    if min_vmem > _UNTUNED_VMEM_BUDGET:
        raise ValueError(
            f"ring attention block_q={block_q} exceeds the "
            f"{_UNTUNED_VMEM_BUDGET // 2**20} MiB VMEM budget"
        )
    ceiling = max(_LANE, (block_q // 4) // _LANE * _LANE)
    q_sub = _LANE
    for d in range(_LANE, ceiling + 1, _LANE):
        if block_q % d == 0:
            if (
                estimate_vmem_bytes(
                    block_q,
                    d,
                    block_kv,
                    block_kv,
                    single_buffered_q=single_buffered_q,
                    head_dim=head_dim,
                )
                <= _UNTUNED_VMEM_BUDGET
            ):
                q_sub = d
            else:
                break
    return q_sub


def select_untuned_blocks(
    shard_len: int,
    *,
    use_fixed_m: bool | None = None,
    head_dim: int = 128,
) -> tuple[int, int, int, int]:
    """Select (block_q, block_kv, block_kv_compute, block_q_sub) for a shard."""
    fixed_m = use_fixed_m is not False
    r0_full = max(_LANE, -(-shard_len // _LANE) * _LANE)
    max_bkv = 4096 if shard_len >= 7500 else 6144
    bkv_ladder = tuple(
        b
        for b in (
            1024,
            1280,
            1536,
            1792,
            2048,
            2304,
            2560,
            2816,
            3072,
            3328,
            3584,
            3840,
            4096,
            4608,
            5120,
            5632,
            6144,
        )
        if b <= max_bkv
    )
    bkv_aligned = {
        (((shard_len + n - 1) // n + _LANE - 1) // _LANE) * _LANE for n in range(1, 20)
    }
    bkv_cands = sorted(
        {b for b in bkv_ladder if b <= r0_full}
        | {b for b in bkv_aligned if 1024 <= b <= min(r0_full, max_bkv)}
    )
    if not bkv_cands:
        bkv_cands = [min(r0_full, _DEFAULT_BLOCK_KV)]

    if shard_len <= 34_000:
        pass_counts = [1, max(2, -(-shard_len // 23040))]
    else:
        pass_counts = [max(2, -(-shard_len // 23040))]
    vmem_budget = int(0.94 * _V7X_VMEM_CAP_BYTES)

    for n_passes in pass_counts:
        target_q = -(-shard_len // n_passes)
        r0 = -(-target_q // _LANE) * _LANE
        max_pad = 0.035 if target_q >= 4096 else 0.06
        r_max = max(r0, -(-int(target_q * (1.0 + max_pad)) // _LANE) * _LANE)
        q_blocks = range(r0, r_max + 1, _LANE)

        best_score = float("inf")
        best_tuple: tuple[int, int, int, int] | None = None

        for bq in q_blocks:
            single_buf = is_single_buffered_q(bq, shard_len)
            qs_cap = (3840 if target_q >= 7500 else 4096) if fixed_m else 3840
            max_qs = min(qs_cap, max(_LANE, (bq // 2) // _LANE * _LANE))
            min_qs = min(1536, max(_LANE, (bq // 8) // _LANE * _LANE))
            q_subs = [
                d
                for d in range(_LANE, max_qs + 1, _LANE)
                if bq % d == 0 and d >= min_qs
            ]
            if not q_subs:
                continue
            for qs in q_subs:
                n_q = bq // qs
                for bkv in bkv_cands:
                    bkvc_list = [bkv]
                    # Only split bkv when one KV block covers the shard (IMEM limit).
                    if bkv >= shard_len and bkv >= 4096:
                        for c in (2048, 2560, 3072):
                            if c < bkv and bkv % c == 0:
                                bkvc_list.append(c)
                    for bkvc in bkvc_list:
                        if (
                            estimate_vmem_bytes(
                                bq,
                                qs,
                                bkv,
                                bkvc,
                                single_buffered_q=single_buf,
                                head_dim=head_dim,
                            )
                            > vmem_budget
                        ):
                            continue
                        if (
                            fixed_m
                            and (shard_len % bkv != 0)
                            and (qs * bkvc > _HYBRID_RAGGED_SCORE_LIMIT)
                        ):
                            continue
                        n_kv = -(-shard_len // bkv)
                        kv_pad_ratio = (n_kv * bkv) / shard_len
                        q_pad_ratio = (n_passes * bq) / shard_len
                        n_compute = n_kv * (bkv // bkvc)
                        score = (
                            q_pad_ratio * kv_pad_ratio
                            + 0.0060 * n_kv
                            + 0.0025 * n_compute
                            + 0.0022 * n_q
                        )
                        if score < best_score:
                            best_score = score
                            best_tuple = (bq, bkv, bkvc, qs)

        if best_tuple is not None:
            return best_tuple

    bkv = min(r0_full, _DEFAULT_BLOCK_KV)
    qs = _untuned_q_sub(
        r0_full,
        bkv,
        head_dim,
        single_buffered_q=is_single_buffered_q(r0_full, shard_len),
    )
    return (r0_full, bkv, bkv, qs)


def _pad_seq(tensor: Any, multiple: int, axis: int) -> Any:
    import jax.numpy as jnp

    pad = -tensor.shape[axis] % multiple
    if not pad:
        return tensor
    widths = [(0, 0)] * tensor.ndim
    widths[axis] = (0, pad)
    return jnp.pad(tensor, widths)


def max_row_norm_per_head(x: Any, *, seq_axis: int = 2) -> Any:
    """Max L2 row norm per head over batch and sequence."""
    import jax.numpy as jnp

    return jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1)).max(axis=(0, seq_axis))


def ring_fixed_m_operands(qn_max: Any, mk_h: Any) -> Any:
    """Build the (2, heads) scalar-prefetch array for hybrid fixed-m ring attention."""
    import jax.numpy as jnp

    bound = qn_max * mk_h
    # Keep row 0 constant so scalar prefetch stays off Mosaic's critical path.
    m = jnp.full_like(bound, float(math.floor(FIXED_M_RING_SAFE_BOUND)))
    return jnp.stack([m, (bound <= FIXED_M_RING_SAFE_BOUND).astype(jnp.float32)])


def fixed_m_update(qk: Any, v: Any, m: Any) -> tuple[Any, Any]:
    """Compute one KV sub-block's softmax sum and output at a pinned m."""
    import jax.numpy as jnp
    from jax import lax

    s_curr = jnp.exp2(qk - m)
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True).astype(jnp.float32)
    o_curr = lax.dot_general(
        v, s_cast, SV_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    return l_curr, o_curr


def online_update(qk: Any, v: Any, m_prev: Any) -> tuple[Any, Any, Any, Any]:
    """Compute one KV sub-block's online-softmax terms."""
    import jax.numpy as jnp
    from jax import lax

    m_curr = qk.max(axis=0)[None, :]
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v, s_cast, SV_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    return m_next, alpha, l_curr, o_curr


class Blocks(NamedTuple):
    """Resolved tiling for one ring attention launch."""

    block_q: int
    block_kv: int
    block_kv_compute: int
    block_kv_compute_in: int
    block_q_sub: int | None

    @classmethod
    def for_shard(
        cls,
        shard_len: int,
        ring_size: int,
        pinned: Mapping[str, Any] | None = None,
        head_dim: int = 128,
        *,
        use_fixed_m: bool | None = None,
    ) -> "Blocks":
        pinned = {k: v for k, v in (pinned or {}).items() if v is not None}
        key = (ring_size, shard_len)
        tuned = (
            (_TUNED_ONLINE_BLOCKS.get(key) or _TUNED_BLOCKS.get(key))
            if use_fixed_m is False
            else _TUNED_BLOCKS.get(key)
        )
        block_q_sub: int | None
        if tuned is not None:
            block_q, block_kv, tuned_bkvc, block_q_sub = tuned
        elif not pinned.keys() & {
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_q_sub",
        }:
            block_q, block_kv, tuned_bkvc, block_q_sub = select_untuned_blocks(
                shard_len, use_fixed_m=use_fixed_m, head_dim=head_dim
            )
        else:
            block_q = -(-shard_len // _DEFAULT_BLOCK_Q) * _DEFAULT_BLOCK_Q
            block_kv = _DEFAULT_BLOCK_KV
            tuned_bkvc = block_kv
            block_q_sub = None
        block_q = int(pinned.get("block_q", block_q))
        block_kv = int(pinned.get("block_kv", block_kv))
        default_bkvc = tuned_bkvc if "block_kv" not in pinned else block_kv
        block_kv_compute = int(pinned.get("block_kv_compute", default_bkvc))
        raw_q_sub = pinned.get("block_q_sub", block_q_sub)
        resolved_q_sub: int | None = int(raw_q_sub) if raw_q_sub is not None else None
        if "block_q_sub" not in pinned and (
            resolved_q_sub is None or block_q % resolved_q_sub != 0
        ):
            resolved_q_sub = _untuned_q_sub(
                block_q,
                block_kv_compute,
                head_dim,
                single_buffered_q=is_single_buffered_q(block_q, shard_len),
            )
        return cls(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_kv_compute_in=int(
                pinned.get("block_kv_compute_in", block_kv_compute)
            ),
            block_q_sub=resolved_q_sub,
        )

    def apply(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        ring_axis: str,
        ring_size: int,
        axis_names: tuple[str, ...],
        use_fixed_m: bool = False,
        mk: Any = None,
        interpret: Any = False,
    ) -> Any:
        shard_len = q.shape[2]
        q = _pad_seq(q, self.block_q, axis=2)
        q_sub = self.block_q_sub or auto_block_q_sub(self.block_q)
        if self.block_q % q_sub:
            raise ValueError(
                f"block_q ({self.block_q}) must be a multiple of block_q_sub ({q_sub})"
            )
        return ring_attention_forward(
            q,
            _pad_seq(k, self.block_kv, axis=2),
            _pad_seq(v, self.block_kv, axis=2),
            block_q=self.block_q,
            block_kv=self.block_kv,
            block_kv_compute=self.block_kv_compute,
            block_kv_compute_in=self.block_kv_compute_in,
            q_sub_block=q_sub,
            q_seq_len=shard_len,
            kv_seq_len=shard_len,
            ring_axis=ring_axis,
            ring_size=ring_size,
            axis_names=axis_names,
            use_fixed_m=use_fixed_m,
            mk=mk,
            interpret=interpret,
        )


def _neighbor(
    axis_names: tuple[str, ...], ring_axis: str, offset: int
) -> tuple[Any, ...]:
    from jax import lax

    ring_size = lax.axis_size(ring_axis)
    idx = lax.axis_index(ring_axis)
    nxt = lax.rem(idx + offset + ring_size, ring_size)
    return tuple(nxt if a == ring_axis else lax.axis_index(a) for a in axis_names)


def _ring_attention_kernel(
    mk_ref: Any,
    q_ref: Any,
    k_hbm: Any,
    v_hbm: Any,
    o_ref: Any,
    k_buf: Any,
    v_buf: Any,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    o_scratch_ref: Any,
    local_sem: Any,
    send_sem: Any,
    recv_sem: Any,
    credit_sem: Any,
    *,
    mask_value: float,
    grid_width: int,
    ring_size: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
    kv_seq_len: int,
    q_heads_per_kv_head: int,
    num_kv_heads: int,
    use_fixed_m: bool,
    q_sub: int,
    axis_names: tuple[str, ...],
    ring_axis: str,
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    float32 = jnp.float32
    head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        raise NotImplementedError(
            f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}"
        )

    b, h, i, j, r = (pl.program_id(n) for n in range(5))
    num_b, num_h, num_i = (pl.num_programs(n) for n in range(3))

    hk = b * num_kv_heads + h // q_heads_per_kv_head
    q_idx = (b * num_h + h) * num_i + i

    t = (q_idx * grid_width + j) * ring_size + r
    slot = lax.rem(t, 2)
    nslot = 1 - slot

    is_first_hop = (j == 0) & (r == 0)
    is_last_hop = (j == grid_width - 1) & (r == ring_size - 1)
    is_first_step = t == 0
    is_last_step = (q_idx == num_b * num_h * num_i - 1) & is_last_hop

    upstream = _neighbor(axis_names, ring_axis, -1)
    downstream = _neighbor(axis_names, ring_axis, +1)

    def _local_load(dst_slot: Any, kv_head: Any, block: Any) -> tuple[Any, Any]:
        src = (pl.ds(kv_head, 1), pl.ds(block * bkv, bkv))
        dst = pl.ds(dst_slot, 1)
        return (
            pltpu.make_async_copy(k_hbm.at[src], k_buf.at[dst], local_sem.at[0]),
            pltpu.make_async_copy(v_hbm.at[src], v_buf.at[dst], local_sem.at[1]),
        )

    def _remote_push(src_slot: Any, dst_slot: Any) -> tuple[Any, Any]:
        src, dst = pl.ds(src_slot, 1), pl.ds(dst_slot, 1)
        return (
            pltpu.make_async_remote_copy(
                k_buf.at[src],
                k_buf.at[dst],
                send_sem.at[0],
                recv_sem.at[0],
                device_id=downstream,
            ),
            pltpu.make_async_remote_copy(
                v_buf.at[src],
                v_buf.at[dst],
                send_sem.at[1],
                recv_sem.at[1],
                device_id=downstream,
            ),
        )

    @pl.when(is_first_step)
    def _barrier() -> None:
        sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(sem, 1, device_id=upstream)
        pl.semaphore_signal(sem, 1, device_id=downstream)
        pl.semaphore_wait(sem, 2)

    @pl.when(is_first_step)
    def _prime() -> None:
        for dma in _local_load(0, hk, 0):
            dma.start()
        for dma in _local_load(0, hk, 0):
            dma.wait()

    @pl.when(jnp.logical_not(is_first_step))
    def _await_current() -> None:
        @pl.when(r == 0)
        def _await_local() -> None:
            for dma in _local_load(slot, hk, j):
                dma.wait()

        @pl.when(r > 0)
        def _await_remote() -> None:
            for dma in _remote_push(nslot, slot):
                dma.wait_recv()

    @pl.when(r > 0)
    def _retire_send() -> None:
        for dma in _remote_push(nslot, slot):
            dma.wait_send()

    # Release nslot in hop t's prologue so Mosaic cannot hoist it early.
    @pl.when((r < ring_size - 1) & (ring_size > 1))
    def _release() -> None:
        pl.semaphore_signal(credit_sem, 1, device_id=upstream)

    @pl.when(jnp.logical_not(is_last_step))
    def _prefetch_next() -> None:
        @pl.when(r < ring_size - 1)
        def _push() -> None:
            pl.semaphore_wait(credit_sem, 1)
            for dma in _remote_push(slot, nslot):
                dma.start()

        @pl.when(r == ring_size - 1)
        def _reload_own() -> None:
            at_q_end = j == grid_width - 1
            next_q_idx = q_idx + 1
            next_bh = lax.div(next_q_idx, num_i)
            next_b = lax.div(next_bh, num_h)
            next_h = lax.rem(next_bh, num_h)
            next_hk = next_b * num_kv_heads + lax.div(next_h, q_heads_per_kv_head)
            load_hk = jnp.where(at_q_end, next_hk, hk)
            load_j = jnp.where(at_q_end, 0, j + 1)
            for dma in _local_load(nslot, load_hk, load_j):
                dma.start()

    is_fixed: Any = mk_ref[1, h] > 0.5 if use_fixed_m else None

    def _m_fixed() -> Any:
        return mk_ref[0, h] - FIXED_M_RECENTER

    n_q = q_ref.shape[0] // q_sub
    two_body_last = n_q > 1 or grid_width == 1

    def _for_each_q_chunk(fn: Any) -> None:
        if n_q > 1:

            def _qbody(c: Any, carry: Any) -> Any:
                fn(pl.multiple_of(c * q_sub, _LANE))
                return carry

            lax.fori_loop(0, n_q, _qbody, None)
        else:
            for qs in range(0, q_ref.shape[0], q_sub):
                fn(qs)

    def _init_fixed_chunk(qs: Any) -> None:
        qsl = pl.ds(qs, q_sub)
        o_scratch_ref[:, qsl] = jnp.zeros(
            (o_scratch_ref.shape[0], q_sub), o_scratch_ref.dtype
        )
        l_scratch_ref[:, qsl] = jnp.zeros(
            (l_scratch_ref.shape[0], q_sub), l_scratch_ref.dtype
        )
        if not two_body_last:
            m_scratch_ref[:, qsl] = jnp.full(
                (m_scratch_ref.shape[0], q_sub), _m_fixed(), m_scratch_ref.dtype
            )

    def _init_online_chunk(qs: Any) -> None:
        qsl = pl.ds(qs, q_sub)
        o_scratch_ref[:, qsl] = jnp.zeros(
            (o_scratch_ref.shape[0], q_sub), o_scratch_ref.dtype
        )
        l_scratch_ref[:, qsl] = jnp.zeros(
            (l_scratch_ref.shape[0], q_sub), l_scratch_ref.dtype
        )
        m_scratch_ref[:, qsl] = jnp.full(
            (m_scratch_ref.shape[0], q_sub), mask_value, m_scratch_ref.dtype
        )

    @pl.when(is_first_hop)
    def _init() -> None:
        if use_fixed_m:
            assert is_fixed is not None

            @pl.when(is_fixed)
            def _init_fixed() -> None:
                _for_each_q_chunk(_init_fixed_chunk)

            @pl.when(jnp.logical_not(is_fixed))
            def _init_online() -> None:
                _for_each_q_chunk(_init_online_chunk)

        else:
            _for_each_q_chunk(_init_online_chunk)

    def _online_inner(
        qk: Any, v_chunk: Any, m_prev: Any, l_prev: Any, o_prev: Any
    ) -> tuple[Any, Any, Any]:
        step = bkv_compute_in
        for c in range(0, qk.shape[0], step):
            m_next, alpha, l_curr, o_curr = online_update(
                qk[c : c + step], v_chunk[c : c + step], m_prev
            )
            o_prev = alpha[0:1, ...] * o_prev + o_curr
            m_prev, l_prev = m_next, l_curr + alpha * l_prev
        return m_prev, l_prev, o_prev

    def _fixed_inner(
        qk: Any, v_chunk: Any, m_fix: Any, l_prev: Any, o_prev: Any
    ) -> tuple[Any, Any]:
        step = bkv_compute_in
        for c in range(0, qk.shape[0], step):
            l_curr, o_curr = fixed_m_update(
                qk[c : c + step], v_chunk[c : c + step], m_fix
            )
            l_prev, o_prev = l_prev + l_curr, o_prev + o_curr
        return l_prev, o_prev

    def _step(offset: Any, length: int, fixed: bool) -> None:
        sl = pl.ds(offset, length)
        k_blk = k_buf[slot, sl, :]
        v_chunk = v_buf[slot, sl, :]

        def _one_q_chunk(qs: Any) -> None:
            qsl = pl.ds(qs, q_sub)
            qk = lax.dot_general(
                k_blk, q_ref[qsl, :], NT_DIM_NUMBERS, preferred_element_type=float32
            )
            if fixed:
                l_prev, o_prev = _fixed_inner(
                    qk,
                    v_chunk,
                    _m_fixed(),
                    l_scratch_ref[:, qsl],
                    o_scratch_ref[:, qsl],
                )
                l_scratch_ref[:, qsl] = l_prev
            else:
                m_prev, l_prev, o_prev = _online_inner(
                    qk,
                    v_chunk,
                    m_scratch_ref[:, qsl],
                    l_scratch_ref[:, qsl],
                    o_scratch_ref[:, qsl],
                )
                m_scratch_ref[:, qsl] = m_prev
                l_scratch_ref[:, qsl] = l_prev
            o_scratch_ref[:, qsl] = o_prev

        _for_each_q_chunk(_one_q_chunk)

    def compute_body_fixed(kv_compute_index: Any, _: Any) -> None:
        _step(kv_compute_index * bkv_compute, bkv_compute, True)

    def compute_body_online(kv_compute_index: Any, _: Any) -> None:
        _step(kv_compute_index * bkv_compute, bkv_compute, False)

    def _run_full(fixed: bool) -> None:
        body = compute_body_fixed if fixed else compute_body_online
        lax.fori_loop(0, bkv // bkv_compute, body, None)

    def _run_tail(fixed: bool) -> None:
        body = compute_body_fixed if fixed else compute_body_online
        if kv_seq_len % bkv == 0:
            lax.fori_loop(0, bkv // bkv_compute, body, None)
            return
        remain = kv_seq_len % bkv
        iter_num = (remain + bkv_compute - 1) // bkv_compute
        if remain % bkv_compute == 0:
            lax.fori_loop(0, iter_num, body, None)
        else:
            if iter_num > 1:
                lax.fori_loop(0, iter_num - 1, body, None)
            _step((iter_num - 1) * bkv_compute, remain % bkv_compute, fixed)

    if not use_fixed_m:
        if grid_width > 1:

            @pl.when(j != grid_width - 1)
            def _body() -> None:
                _run_full(False)

        @pl.when(j == grid_width - 1)
        def _last_body() -> None:
            _run_tail(False)

    else:
        assert is_fixed is not None
        if grid_width > 1:

            @pl.when((j != grid_width - 1) & is_fixed)
            def _body_fixed() -> None:
                _run_full(True)

            @pl.when((j != grid_width - 1) & jnp.logical_not(is_fixed))
            def _body_online() -> None:
                _run_full(False)

        if two_body_last:

            @pl.when((j == grid_width - 1) & is_fixed)
            def _last_body_fixed() -> None:
                _run_tail(True)

            @pl.when((j == grid_width - 1) & jnp.logical_not(is_fixed))
            def _last_body_online() -> None:
                _run_tail(False)

        else:

            @pl.when(j == grid_width - 1)
            def _last_body() -> None:
                _run_tail(False)

    @pl.when(is_last_hop)
    def _write_out() -> None:
        def _write_out_chunk(qs: Any) -> None:
            qsl = pl.ds(qs, q_sub)
            l_val = l_scratch_ref[:, qsl]
            l_inv = jnp.tile(1.0 / l_val, (head_dim_v_repeats, 1))
            o_ref[:, qsl] = (o_scratch_ref[:, qsl] * l_inv).astype(o_ref.dtype)

        _for_each_q_chunk(_write_out_chunk)


def ring_attention_forward(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_kv_compute_in: int,
    q_sub_block: int,
    q_seq_len: int,
    kv_seq_len: int,
    ring_axis: str,
    ring_size: int,
    axis_names: tuple[str, ...],
    use_fixed_m: bool = False,
    mk: Any = None,
    vmem_limit_bytes: int | None = VMEM_LIMIT_BYTES,
    mask_value: float = DEFAULT_MASK_VALUE,
    interpret: Any = False,
) -> Any:
    """Launch single-call Pallas ring attention over [B, H, S, D] shards."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    batch, num_q_heads, _, head_dim_qk = q.shape
    num_kv_heads = k.shape[1]
    kv_pad = k.shape[2]
    head_dim_v = v.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    bq, bkv = block_q, block_kv
    q_sub = q_sub_block
    if bq % q_sub != 0 or q_sub % _LANE != 0:
        raise ValueError(
            f"q_sub_block ({q_sub}) must be a multiple of {_LANE} "
            f"dividing block_q ({bq})"
        )
    if bkv % block_kv_compute != 0:
        raise ValueError(
            f"block_kv_compute ({block_kv_compute}) must divide block_kv ({bkv})"
        )
    if not use_fixed_m:
        mk = jnp.zeros((2, 1), jnp.float32)
    elif mk is None:
        raise ValueError("use_fixed_m requires mk")

    grid_width = (kv_seq_len + bkv - 1) // bkv
    grid_height = (q_seq_len + bq - 1) // bq
    grid = (batch, num_q_heads, grid_height, grid_width, ring_size)

    resident: dict[str, Any] = (
        {"pipeline_mode": pl.Buffered(1)} if is_single_buffered_q(bq, q_seq_len) else {}
    )

    def q_index_map(b: Any, h: Any, i: Any, j: Any, r: Any, *_: Any) -> tuple[Any, ...]:
        del j, r
        return (b, h, i, 0)

    def out_index_map(
        b: Any, h: Any, i: Any, j: Any, r: Any, *_: Any
    ) -> tuple[Any, ...]:
        del j, r
        return (b, h, 0, i)

    in_specs = [
        pl.BlockSpec((None, None, bq, head_dim_qk), q_index_map, **resident),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    out_specs = pl.BlockSpec((None, None, head_dim_v, bq), out_index_map, **resident)
    out_shape = jax.ShapeDtypeStruct(
        (batch, num_q_heads, head_dim_v, q_seq_len), q.dtype
    )
    scratch_shapes = [
        pltpu.VMEM((2, bkv, head_dim_qk), k.dtype),
        pltpu.VMEM((2, bkv, head_dim_v), v.dtype),
        pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
        pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
        pltpu.VMEM((head_dim_v, bq), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.REGULAR,
    ]

    out = pl.pallas_call(
        functools.partial(
            _ring_attention_kernel,
            mask_value=mask_value,
            grid_width=grid_width,
            ring_size=ring_size,
            bkv=bkv,
            bkv_compute=block_kv_compute,
            bkv_compute_in=block_kv_compute_in,
            head_dim_v=head_dim_v,
            kv_seq_len=kv_seq_len,
            q_heads_per_kv_head=q_heads_per_kv_head,
            num_kv_heads=num_kv_heads,
            use_fixed_m=use_fixed_m,
            q_sub=q_sub,
            axis_names=tuple(axis_names),
            ring_axis=ring_axis,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * 5,
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
            collective_id=_COLLECTIVE_ID,
            has_side_effects=False,
        ),
        out_shape=out_shape,
        interpret=interpret,
        name="diffusion_ring_attention",
    )(
        mk,
        q,
        k.reshape(batch * num_kv_heads, kv_pad, head_dim_qk),
        v.reshape(batch * num_kv_heads, kv_pad, head_dim_v),
    )
    assert isinstance(out, jax.Array)
    return jnp.swapaxes(out, 2, 3)


def ring_attention_sharded(
    q: Any,
    k: Any,
    v: Any,
    *,
    ring_axis: str,
    axis_names: tuple[str, ...],
    scale: float,
    use_fixed_m: bool = True,
    pinned: Mapping[str, Any] | None = None,
    interpret: Any = False,
) -> Any:
    """Run shard_map body on [1, ..., B, S_shard, H, D] local shards."""
    import jax
    import jax.numpy as jnp

    lead = len(axis_names)
    lead_axes = tuple(range(lead))
    q_s = jnp.squeeze(q, axis=lead_axes)
    k_s = jnp.squeeze(k, axis=lead_axes)
    v_s = jnp.squeeze(v, axis=lead_axes)

    ring_size = jax.lax.axis_size(ring_axis)
    shard_len = q_s.shape[1]
    head_dim = q_s.shape[-1]
    q_s = (q_s * (scale * LOG2_E)).astype(q_s.dtype)

    mk = None
    if use_fixed_m:
        # Prevent XLA from duplicating Q/K producers into the norm reductions.
        q_s, k_s = jax.lax.optimization_barrier((q_s, k_s))
        qn_max = max_row_norm_per_head(q_s, seq_axis=1)
        mk_h = max_row_norm_per_head(k_s, seq_axis=1)
        if q_s.shape[2] != k_s.shape[2]:
            mk_h = jnp.repeat(mk_h, q_s.shape[2] // k_s.shape[2], axis=0)
        qn_max, mk_h = jax.lax.pmax((qn_max, mk_h), ring_axis)
        mk = ring_fixed_m_operands(qn_max, mk_h)

    blocks = Blocks.for_shard(
        shard_len,
        ring_size,
        pinned,
        head_dim=head_dim,
        use_fixed_m=use_fixed_m,
    )
    out = blocks.apply(
        jnp.swapaxes(q_s, 1, 2),
        jnp.swapaxes(k_s, 1, 2),
        jnp.swapaxes(v_s, 1, 2),
        ring_axis=ring_axis,
        ring_size=ring_size,
        axis_names=axis_names,
        use_fixed_m=use_fixed_m,
        mk=mk,
        interpret=interpret,
    )
    return jnp.expand_dims(jnp.swapaxes(out, 1, 2), axis=lead_axes)
