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
"""GDN verify-window tile, one window position and one value head (with
paired heads, two per MXU dot) at a time.

`wrapper.inner_kernel` computes a whole tile as large arrays: the conv on
the compact [chunk, 1, dim_size] layout (one 128-lane chunk of one token
per vector register) and the recurrence on the [num_v_heads, kq_head_dim,
v_head_dim] FP32 state of every window position. Neither fits in
registers, and the recurrence's MXU weight pushes (two 128 x 128 states per
head and position) leave most of the VPU idle. This tile instead:

* keeps each token's activations dense, [dim_size / 128, 128] (one head
  per row), so the conv touches 8x fewer registers;
* decodes and encodes the pooled conv state one tap (one dense
  [dim_size / 128, 128] block) at a time instead of row by row;
* loops over window positions without unrolling: position `pos` runs the
  conv of token `pos`, writes conv checkpoint `pos`, and updates the
  recurrent state one value head at a time through a VMEM scratch, storing
  each head's checkpoint as it is produced. The heads of a position are
  independent, which hides each head's MXU round trip, and the conv and
  checkpoint work fills the slots the weight pushes leave free.

With paired heads off, every element sees the same operations in the
same order as in `inner_kernel` (the conv's tap sum, SiLU, per-row masks,
L2 norms and gates, per-head dots, which `recurrent_gdn` batches over
heads), so outputs and every checkpoint byte are bitwise equal.

With two-slot checkpoints (`GDNConfig.two_slot_ckpts`) a sequence keeps
only the window's start state and the state after its last real token. A
partially accepted window leaves its accepted tokens pending; the next
window first recomputes them from the start state (`replay`: the same
per-token arithmetic, no outputs, no checkpoints) and writes the result
back to the read tile in the state's storage dtype, which is exactly the
state a checkpoint store and reload would have given the window. The
tokens to recompute come from the start tile's conv block, where each
window stores its first `window_size - 1` tokens' conv inputs and b / a
rows past the conv state. When a mamba block boundary falls strictly
inside the window, the state after the window position that completes the
block goes to a third tile, the bytes the one-block-per-position kernel
writes for that checkpoint, which the write-back stores in the completed
block (`memory_ref.two_slot_boundary`).

With paired heads (`GDNConfig.paired_heads`) and an even number of value
heads per key head, every dot of k or q with the states, in the window
loop, the replay and the boundary recompute alike, takes two value heads
of one key head: their states side by side form one
[kq_head_dim, 2 * v_head_dim] MXU weight, which halves the weight pushes
that otherwise set the loop's pace. Each output lane still comes from the
same BF16-rounded state column and the same row; that the MXU sums it to
the same bits is hardware behaviour, which
tests/kernels/test_gdn_v3_paired_dot.py checks. With the pushes halved the
loop is bound by vector work, so it then selects no whole states in the
window loop either: a fresh sequence's read tile, which copy_in leaves
unread, is zero filled, and past a two-slot window's last real token each
update is an exact no-op (decay exactly 1, added term k_t * -0.0 with
k = +0), which keeps the state bit for bit if the VPU keeps x * 1.0 and
x + -0.0 exact (also checked by that test). The two-slot start tile goes
through the has_initial_state select in both forms, so its bytes are
re-packed alike.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, memory_ref, vmem_ldst


def _lanes() -> int:
    return pltpu.get_tpu_info().num_lanes


def _head_groups(cfg: config.GDNConfig, kq_head: int) -> list[tuple[int, ...]]:
    """The value heads of key head `kq_head` in order, one group per MXU
    dot: two heads each with `cfg.paired_heads` and an even number of value
    heads per key head, else one."""
    heads = range(kq_head * cfg.v_per_kq_head, (kq_head + 1) * cfg.v_per_kq_head)
    size = 2 if cfg.paired_heads and cfg.v_per_kq_head % 2 == 0 else 1
    return [tuple(heads[i : i + size]) for i in range(0, len(heads), size)]


def _stored_rows(cfg: config.GDNConfig) -> tuple[int, ...] | None:
    """Logical conv row held by each stored row of one conv tap (None for
    the identity), or () when the pooled row order is not the same
    permutation in every tap or moves value rows."""
    assert cfg.state_plan is not None
    region = cfg.state_plan.conv
    rows = cfg.dim_size // _lanes()
    if region.rows_perm is None:
        return None
    perm = region.rows_perm
    tap = perm[:rows]
    for j in range(cfg.prev_kernel_size):
        if tuple(p - j * rows for p in perm[j * rows : (j + 1) * rows]) != tap:
            return ()
    qk_rows = 2 * cfg.num_kq_heads * cfg.kq_head_dim // _lanes()
    if sorted(tap) != list(range(rows)) or tuple(tap[qk_rows:]) != tuple(
        range(qk_rows, rows)
    ):
        return ()
    stored = [0] * rows
    for logical, s in enumerate(tap):
        stored[s] = logical
    return None if tap == tuple(range(rows)) else tuple(stored)


def supported(cfg: config.GDNConfig, block_shape: tuple[int, ...], block_dtype) -> bool:
    """Whether this tile implements `cfg` exactly as inner_kernel does, for
    pooled source blocks of `block_shape` / `block_dtype`."""
    lanes = _lanes()
    plan = cfg.state_plan
    if (
        cfg.mode != config.GDNMode.BATCHED
        or cfg.window_size <= 1
        or cfg.is_kda
        or plan is None
        or not plan.whole_block_dma
        or cfg.kq_head_dim != lanes
        or cfg.v_head_dim != lanes
        or cfg.dim_size % lanes
    ):
        return False
    conv = plan.conv
    rows = cfg.dim_size // lanes
    if (
        conv.view_dtype != jnp.bfloat16
        or conv.lane_split != 1
        or conv.nblocks != 1
        or conv.rows_used != cfg.prev_kernel_size * rows
        or rows % 8
        or _stored_rows(cfg) == ()
    ):
        return False
    if cfg.two_slot_ckpts and _window_rows_base(cfg) + _window_rows(
        cfg
    ) > _conv_view_rows(block_shape, block_dtype):
        return False
    return plan.recurrent.view_dtype == jnp.uint32 and vmem_ldst.supports_head_states(
        block_shape, block_dtype, cfg
    )


def _conv_view_rows(block_shape: tuple[int, ...], block_dtype) -> int:
    """Typed BF16 rows of one conv tile (one whole source block)."""
    nbytes = math.prod(block_shape) * jnp.dtype(block_dtype).itemsize
    return nbytes // (_lanes() * jnp.dtype(jnp.bfloat16).itemsize)


def _window_rows_base(cfg: config.GDNConfig) -> int:
    """First typed row past a conv checkpoint's state and zero padding:
    where two-slot checkpoints keep a window's tokens."""
    used = cfg.prev_kernel_size * cfg.dim_size // _lanes()
    return -(-used // 16) * 16


def _window_rows(cfg: config.GDNConfig) -> int:
    """Typed rows (whole packed BF16 registers) of the tokens a two-slot
    start tile keeps: the conv inputs of the window's first
    `window_size - 1` tokens, then their b and a rows."""
    rows = (cfg.window_size - 1) * (cfg.dim_size // _lanes() + 2)
    return -(-rows // 16) * 16


def _conv_view(slot_ref: jax.Ref) -> jax.Ref:
    """One checkpoint's conv region tile as typed [rows, lanes] BF16."""
    return typed_ldst.typed_view(slot_ref.at[0], jnp.bfloat16, 1)


def _to_stored(block: jax.Array, cfg: config.GDNConfig) -> jax.Array:
    """[dim_size / 128, lanes] block from logical to the pool's stored row
    order; only the rows that move are touched."""
    order = _stored_rows(cfg)
    if order is None:
        return block
    head = len(order)
    while head and order[head - 1] == head - 1:
        head -= 1
    moved = jnp.concat([block[order[i] : order[i] + 1] for i in range(head)], axis=0)
    return jnp.concat([moved, block[head:]], axis=0)


def _load_conv_taps(slot_ref: jax.Ref, cfg: config.GDNConfig) -> jax.Array:
    """The conv state as FP32 [prev_kernel_size * dim_size / 128, lanes] in
    the pool's stored row order: per element, the values
    vmem_ldst._load_conv_state decodes."""
    used = cfg.prev_kernel_size * cfg.dim_size // _lanes()
    return _conv_view(slot_ref)[pl.ds(0, used)].astype(jnp.float32)


def _store_conv_ckpt(slot_ref: jax.Ref, cfg: config.GDNConfig, taps) -> None:
    """Stores one conv checkpoint given as FP32 taps in the stored row
    order: the bytes vmem_ldst.store_state_region writes, zero padding
    included."""
    view = _conv_view(slot_ref)
    lanes = _lanes()
    used = cfg.prev_kernel_size * cfg.dim_size // lanes
    # Whole packed BF16 registers: the data rows, then the zero padding.
    head = min(view.shape[0], -(-used // 16) * 16)
    taps = list(taps)
    if head > used:
        taps.append(jnp.zeros((head - used, lanes), jnp.float32))
    view[pl.ds(0, head)] = jnp.concat(taps, axis=0).astype(jnp.bfloat16)
    if head < view.shape[0]:
        view[pl.ds(head, view.shape[0] - head)] = jnp.zeros(
            (view.shape[0] - head, view.shape[1]), jnp.bfloat16
        )


def _store_conv_start(
    slot_ref: jax.Ref, cfg: config.GDNConfig, taps, tokens, gates
) -> None:
    """Stores a two-slot start tile's conv block: the start state's taps
    as `_store_conv_ckpt` does, then the conv inputs (stored row order) of
    each window token that may have to be recomputed, then their b and a
    rows, then zeros. Every value came from BF16, so the BF16 store is
    exact."""
    view = _conv_view(slot_ref)
    lanes = _lanes()
    base = _window_rows_base(cfg)
    used = cfg.prev_kernel_size * cfg.dim_size // lanes
    blocks = list(taps)
    if base > used:
        blocks.append(jnp.zeros((base - used, lanes), jnp.float32))
    blocks.extend(tokens)
    for gate in gates:
        blocks.extend(gate)
    data = jnp.concat(blocks, axis=0)
    end = -(-data.shape[0] // 16) * 16
    if end > data.shape[0]:
        data = jnp.concat(
            [data, jnp.zeros((end - data.shape[0], lanes), jnp.float32)], axis=0
        )
    view[pl.ds(0, end)] = data.astype(jnp.bfloat16)
    if end < view.shape[0]:
        view[pl.ds(end, view.shape[0] - end)] = jnp.zeros(
            (view.shape[0] - end, view.shape[1]), jnp.bfloat16
        )


def run(
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size] FP32 or packed tokens
    b_slot_ref: jax.Ref | None,  # [seq, chunk, 1, lanes]; None when packed
    a_slot_ref: jax.Ref | None,  # [seq, chunk, 1, lanes]; None when packed
    conv_in_ref: jax.Ref,  # [seq, 1, nblocks, *block]
    rec_in_ref: jax.Ref,  # [seq, 1, nblocks, *block]
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v_head_dim]
    conv_out_ref: jax.Ref,  # [seq, window (two-slot: 3), nblocks, *block]
    rec_out_ref: jax.Ref,  # [seq, window (two-slot: 3), nblocks, *block]
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    state_ref: jax.Ref,  # [seq, num_v_heads, kq_head_dim, v_head_dim] FP32
    lhs_ref: jax.Ref,  # [seq, chunk + prev_kernel_size, dim_size / 128, 128] FP32
    conv_w_ref: jax.Ref,  # [kernel_size + 1, dim_size / 128, 128] FP32
    # Two-slot checkpoints only: the previous window's conv inputs after the
    # start taps [seq, prev_kernel_size + window - 1, dim_size / 128, 128] and
    # its b / a rows [seq, window - 1, 2, 128], FP32.
    rep_ref: jax.Ref | None = None,
    rep_gate_ref: jax.Ref | None = None,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array,
) -> None:
    lanes = _lanes()
    rows = cfg.dim_size // lanes
    nv, nkq = cfg.num_v_heads, cfg.num_kq_heads
    ks, prev = cfg.kernel_size, cfg.prev_kernel_size
    compute = cfg.dtypes.compute
    q_scale = cfg.kq_head_dim**-0.5
    seqs = range(cfg.seq_tile_size)

    real_sizes = [metadata_ref.p_id_to_r_size[p_id, idx] for idx in seqs]
    has_initial_state = [
        metadata_ref.s_idx_has_initial_state[metadata_ref.p_id_to_s_idx[p_id, idx]]
        for idx in seqs
    ]
    two = cfg.two_slot_ckpts
    # Paired heads select no whole states in the window loop: a fresh
    # sequence's read tile is zero filled, and past a two-slot window's last
    # real token each update is an exact no-op (decay 1, added term -0.0).
    state_selects = not cfg.paired_heads
    num_replay = cfg.window_size - 1
    replays: list[jax.Array] = []
    # Two-slot: window position whose state a block boundary inside the
    # window needs, or -1, per sequence.
    snap_pos: list[jax.Array] = []
    if two:
        assert rep_ref is not None and rep_gate_ref is not None
        replays = [
            memory_ref.two_slot_read(
                metadata_ref, metadata_ref.p_id_to_s_idx[p_id, idx], real_sizes[idx]
            )[1]
            for idx in seqs
        ]
        snap_pos = [
            memory_ref.two_slot_boundary(
                metadata_ref, metadata_ref.p_id_to_s_idx[p_id, idx], real_sizes[idx]
            )[0]
            for idx in seqs
        ]
    if not state_selects:
        # A live sequence without an initial state starts from zeros: its
        # read tile, which copy_in leaves unread, is zero filled.
        for idx in seqs:
            fresh = jnp.logical_and(
                metadata_ref.p_id_is_first_tile[p_id, idx],
                jnp.logical_not(has_initial_state[idx]),
            )
            pl.when(fresh)(
                lambda idx=idx: vmem_ldst.zero_state_tile(rec_in_ref, cfg, idx)
            )

    # Conv inputs per sequence, dense and in the pool's stored row order: the
    # previous taps (zero for a fresh sequence), then the window's tokens.
    # The conv is per row, so with its weights in the same order it yields
    # the activations in that order and every checkpoint is a plain copy of
    # three input rows.
    for idx in seqs:
        prev_taps = jnp.where(
            has_initial_state[idx], _load_conv_taps(conv_in_ref.at[idx, 0], cfg), 0
        )
        # Two-slot: the start taps head the previous window's tokens, which
        # the start tile stored past its conv state; the window's own taps
        # are copied in once the replay has advanced past those tokens.
        tap_ref = rep_ref if two else lhs_ref
        for tap in range(prev):
            tap_ref[idx, tap] = prev_taps[tap * rows : (tap + 1) * rows]
        if two:
            stored = _conv_view(conv_in_ref.at[idx, 0])[
                pl.ds(_window_rows_base(cfg), _window_rows(cfg))
            ].astype(jnp.float32)
            gate0 = num_replay * rows
            for j in range(num_replay):
                rep_ref[idx, prev + j] = stored[j * rows : (j + 1) * rows]
                rep_gate_ref[idx, j] = stored[gate0 + 2 * j : gate0 + 2 * j + 2]
        for t in range(cfg.chunk_size):
            if cfg.packed_tokens:
                token = qkv_slot_ref[idx, t, pl.ds(0, rows)]
            else:
                token = qkv_slot_ref.at[idx].reshape(-1, lanes)[pl.ds(t * rows, rows)]
            lhs_ref[idx, prev + t] = _to_stored(token, cfg)
    for k in range(ks):
        weight = weights_ref.conv.weight.at[k].reshape(-1, lanes)[...]
        conv_w_ref[k] = _to_stored(weight.astype(jnp.float32), cfg)
    if weights_ref.conv.bias is not None:
        bias = weights_ref.conv.bias.reshape(-1, lanes)[...]
        conv_w_ref[ks] = _to_stored(bias.astype(jnp.float32), cfg)
    order = _stored_rows(cfg)
    stored_row = list(range(rows)) if order is None else [0] * rows
    if order is not None:
        for s_row, logical in enumerate(order):
            stored_row[logical] = s_row

    padding_size = cfg.aligned_num_v_heads - nv
    a_log = jnp.pad(weights_ref.gdn.a_log[...], (0, padding_size)).reshape(1, -1)
    neg_a = -jnp.exp(a_log.astype(compute))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], (0, padding_size)).reshape(1, -1)
    dt_bias = dt_bias.astype(compute)
    batched_dims = (((2,), (1,)), ((0,), (0,)))

    def gate_rows(idx, t):
        # Token t's b and a rows, [1, lanes] each.
        if cfg.packed_tokens:
            return (
                qkv_slot_ref[idx, t, pl.ds(rows, 1)],
                qkv_slot_ref[idx, t, pl.ds(rows + 1, 1)],
            )
        return b_slot_ref[idx, t], a_slot_ref[idx, t]

    def head_scalar(row, head):
        # One head lane of a gate row as [1, 1], picked like
        # compute_gdn.recurrent_gdn's fused_transpose_broadcast does.
        picked = compute_gdn.fused_transpose_broadcast(
            row[:, :nv].reshape(1, 1, nv), src_dim=2, dst_dim=1
        )
        return picked[0, head : head + 1]

    def head_dot(row, mats):
        # A [1, lanes] row times the group's [kq_head_dim, v_head_dim]
        # matrices side by side: one MXU weight for the whole group.
        rhs = mats[0] if len(mats) == 1 else jnp.concat(mats, axis=1)
        return jax.lax.dot(
            row[None],
            rhs[None],
            dimension_numbers=batched_dims,
            preferred_element_type=jnp.float32,
        )[0].astype(compute)

    def head_part(prod, j, n):
        # Head j's lanes of a group product over n heads.
        return prod if n == 1 else prod[:, j * lanes : (j + 1) * lanes]

    def recompute(j, idx, taps_ref, gates, start_ref, out_ref, out_tile):
        # Token j recomputed with the arithmetic `position` gave it, state
        # only: conv inputs from `taps_ref`, b / a rows `gates`, the state
        # before token 0 from `start_ref`; each token's state is stored in
        # tile `out_tile` of `out_ref` in the state's storage dtype.
        act = jnp.zeros((rows, lanes), jnp.float32)
        for k in range(ks):
            act += taps_ref[idx, j + k] * conv_w_ref[k]
        if weights_ref.conv.bias is not None:
            act += conv_w_ref[ks]
        act = jax.nn.silu(act)
        b_row, a_row = gates
        beta = jax.nn.sigmoid(b_row.astype(compute))
        gating = jnp.exp(neg_a * jax.nn.softplus(a_row.astype(compute) + dt_bias))
        for kq_head in range(nkq):
            r = stored_row[nkq + kq_head]
            k = compute_gdn.l2_norm(act[r : r + 1].astype(compute))
            k_t = compute_gdn.fused_transpose_broadcast(k, src_dim=1, dst_dim=0)
            for heads in _head_groups(cfg, kq_head):
                vs, decayed = [], []
                for head in heads:
                    vs.append(act[2 * nkq + head : 2 * nkq + head + 1].astype(compute))
                    if j == 0:
                        state = vmem_ldst.load_state_head(start_ref, cfg, idx, head)
                    else:
                        state = state_ref[idx, head]
                    decayed.append(state * head_scalar(gating, head))
                v_updated = head_dot(k, decayed)
                for h, head in enumerate(heads):
                    v_new = head_scalar(beta, head) * (
                        vs[h] - head_part(v_updated, h, len(heads))
                    )
                    state = decayed[h] + k_t * v_new
                    state_ref[idx, head] = state
                    vmem_ldst.store_state_head(out_ref, cfg, idx, out_tile, head, state)

    def replay(j, idx):
        # Token j of the previous window, recomputed from the start state.
        # The window starts from the read tile, so the state is left there
        # as a checkpoint store would.
        gates = rep_gate_ref[idx, j]
        recompute(j, idx, rep_ref, (gates[0:1], gates[1:2]), rec_in_ref, rec_in_ref, 0)

    def snapshot(idx):
        # The state at a block boundary inside the window, after position
        # `snap_pos`: the window's first tokens recomputed from the start
        # tile, which gives the bytes checkpoint `snap_pos` holds.
        for j in range(cfg.chunk_size - 1):
            pl.when(j <= snap_pos[idx])(
                lambda j=j: recompute(
                    j, idx, lhs_ref, gate_rows(idx, j), rec_out_ref, rec_out_ref, 2
                )
            )
        vmem_ldst.zero_state_padding(rec_out_ref, cfg, idx, 2)
        # Inputs ending at that position, as conv checkpoint `snap_pos`
        # holds them.
        _store_conv_ckpt(
            conv_out_ref.at[idx, 2],
            cfg,
            [lhs_ref[idx, snap_pos[idx] + 1 + j] for j in range(prev)],
        )

    def position(pos, first):
        for idx in seqs:
            real_size = real_sizes[idx]
            valid = pos < real_size

            # compute_conv1d.causal_conv1d's tap sum, then SiLU (stored order).
            act = jnp.zeros((rows, lanes), jnp.float32)
            for k in range(ks):
                act += lhs_ref[idx, pos + k] * conv_w_ref[k]
            if weights_ref.conv.bias is not None:
                act += conv_w_ref[ks]
            act = jax.nn.silu(act)

            # Conv checkpoint `pos`: the inputs ending at the window's
            # position `pos`, or at the last real token if that is earlier.
            if not two:
                start = jnp.where(
                    jnp.logical_and(real_size >= 1, real_size <= pos),
                    real_size,
                    pos + 1,
                )
                _store_conv_ckpt(
                    conv_out_ref.at[idx, pos],
                    cfg,
                    [lhs_ref[idx, start + j] for j in range(prev)],
                )

            b_row, a_row = gate_rows(idx, pos)
            beta = jnp.where(valid, jax.nn.sigmoid(b_row.astype(compute)), 0)
            gating = neg_a * jax.nn.softplus(a_row.astype(compute) + dt_bias)
            if two and not state_selects:
                # Past the last real token the decay is exactly 1.
                gating = jnp.where(valid, jnp.exp(gating), 1.0)
            else:
                gating = jnp.exp(jnp.where(valid, gating, 0))

            out_rows = []
            for kq_head in range(nkq):
                r = stored_row[kq_head]
                q = compute_gdn.l2_norm(
                    jnp.where(valid, act[r : r + 1].astype(compute), 0)
                )
                q = q * q_scale
                r = stored_row[nkq + kq_head]
                k = compute_gdn.l2_norm(
                    jnp.where(valid, act[r : r + 1].astype(compute), 0)
                )
                k_t = compute_gdn.fused_transpose_broadcast(k, src_dim=1, dst_dim=0)
                for heads in _head_groups(cfg, kq_head):
                    vs, states, decayed = [], [], []
                    for head in heads:
                        v = act[2 * nkq + head : 2 * nkq + head + 1]
                        vs.append(jnp.where(valid, v.astype(compute), 0))
                        if first:
                            state = vmem_ldst.load_state_head(
                                rec_in_ref, cfg, idx, head
                            )
                            if state_selects:
                                state = jnp.where(has_initial_state[idx], state, 0)
                            if two:
                                # The start tile, through the select in both
                                # forms: without it Mosaic folds the BF16 ->
                                # FP32 -> BF16 round trip into a raw copy.
                                start_state = (
                                    state
                                    if state_selects
                                    else jnp.where(has_initial_state[idx], state, 0)
                                )
                                vmem_ldst.store_state_head(
                                    rec_out_ref, cfg, idx, 0, head, start_state
                                )
                        else:
                            state = state_ref[idx, head]
                        states.append(state)
                        decayed.append(state * head_scalar(gating, head))
                    v_updated = head_dot(k, decayed)
                    updates = []
                    for h, head in enumerate(heads):
                        v_new = head_scalar(beta, head) * (
                            vs[h] - head_part(v_updated, h, len(heads))
                        )
                        if two and not state_selects:
                            # Past the last real token k is +0, so the added
                            # term k_t * -0.0 is -0.0 and the state is kept bit
                            # for bit (x * 1 + -0.0 == x, -0.0 included).
                            v_new = jnp.where(valid, v_new, -0.0)
                        updates.append(decayed[h] + k_t * v_new)
                    outs = head_dot(q, updates)
                    for h, head in enumerate(heads):
                        out_rows.append(head_part(outs, h, len(heads)))
                        updated = updates[h]
                        if two:
                            if state_selects:
                                # Past the last real token the state stays put, so the
                                # window's last position holds the last state.
                                updated = jnp.where(valid, updated, states[h])
                            state_ref[idx, head] = updated
                            if pos == cfg.chunk_size - 1:
                                vmem_ldst.store_state_head(
                                    rec_out_ref, cfg, idx, 1, head, updated
                                )
                        else:
                            state_ref[idx, head] = updated
                            vmem_ldst.store_state_head(
                                rec_out_ref, cfg, idx, pos, head, updated
                            )
            out_slot_ref[idx, pos] = jnp.concat(out_rows, axis=0).astype(
                out_slot_ref.dtype
            )

    if two:
        # Only sequences with tokens left to recompute run a replay
        # position; the heads of one sequence keep the MXU busy.
        for j in range(num_replay):
            for idx in seqs:
                pl.when(j < replays[idx])(lambda j=j, idx=idx: replay(j, idx))
        for idx in seqs:
            for tap in range(prev):
                lhs_ref[idx, tap] = rep_ref[idx, replays[idx] + tap]
    for pos in range(cfg.chunk_size):
        position(pos, first=pos == 0)
    if two:
        for idx in seqs:
            # Inputs ending at the last real token, as conv checkpoint
            # `real_size - 1` holds them.
            last = jnp.maximum(real_sizes[idx], 1)
            _store_conv_ckpt(
                conv_out_ref.at[idx, 1],
                cfg,
                [lhs_ref[idx, last + j] for j in range(prev)],
            )
            _store_conv_start(
                conv_out_ref.at[idx, 0],
                cfg,
                [lhs_ref[idx, j] for j in range(prev)],
                [lhs_ref[idx, prev + t] for t in range(num_replay)],
                [gate_rows(idx, t) for t in range(num_replay)],
            )
    for idx in seqs:
        # Two-slot snapshot tiles are padded where they are written.
        for t in range(2 if two else cfg.num_ckpt_tiles):
            vmem_ldst.zero_state_padding(rec_out_ref, cfg, idx, t)
    if two:
        # Off the position loop: only sequences with a boundary inside the
        # window recompute, and the loop keeps its single block.
        for idx in seqs:
            pl.when(snap_pos[idx] >= 0)(lambda idx=idx: snapshot(idx))
