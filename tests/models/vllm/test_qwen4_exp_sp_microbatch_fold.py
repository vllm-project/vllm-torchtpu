# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SP_MICROBATCH_FOLD_ALIGN: micro-batched sequence-parallel halves
under VLLM_TPU_MAMBA_TAIL_FOLD give the whole step's GDN outputs, internal
prefill checkpoints and final states, bit for bit.

The GDN layer runs its real ``forward`` (half metadata, A's rows in front of
B's call, the op variant and its operands) against a stand-in for the pooled
op. The stand-in builds the kernel's PER_SEQ metadata with the real
``metadata.compute_per_seq_metadata`` and runs it tile by tile with the
kernel's own tile math (``compute_conv1d.causal_conv1d`` then
``compute_gdn.chunked_gdn``): each tile reads its rows padded to a whole tile
with the rows after them, takes the pool state on a first tile with history
and the carry otherwise, and a last tile writes its state to its checkpoint
block or override. CPU interpret mode cannot run the Pallas kernel with
multi-row tiles, so this is the strongest CPU check; the kernel on TPU must
confirm it.
"""

from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.gdn.v3 import compute_conv1d, compute_gdn, metadata
from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp

N_KQ, N_V, DK, DV, KSIZE = 1, 2, 16, 16, 4
DIM = 2 * N_KQ * DK + N_V * DV
CHUNK = 8  # the kernel's PER_SEQ tile
SPEC = 2  # checkpoints per request slot (read offsets 0 / 1)
CONV = (KSIZE - 1) * DIM
STATE = CONV + N_V * DK * DV
NUM_BLOCKS = 64


@pytest.fixture(autouse=True)
def cpu_default():
    with torch.device("cpu"):
        yield


def _cfg():
    return NS(
        seq_tile_size=1,
        chunk_size=CHUNK,
        sub_chunk_size=CHUNK,
        chunk_pair=False,
        use_recurrent=False,
        window_size=1,
        kernel_size=KSIZE,
        prev_kernel_size=KSIZE - 1,
        dim_size=DIM,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=DK,
        v_head_dim=DV,
        v_per_kq_head=N_V // N_KQ,
        triangular_block_size=CHUNK // 2,
        dtypes=NS(compute=jnp.float32),
    )


@jax.jit
def _tile(x, b, a, real, conv_prev, rec_prev, weight, a_log, dt_bias):
    """One PER_SEQ tile of the fused kernel: conv1d over the carry and the
    tile's rows, SiLU, then chunked GDN. Returns (out, conv carry, state)."""
    cfg = _cfg()
    lhs = jnp.concatenate([conv_prev, x])[None, :, None, :]
    conv_out, conv_state = compute_conv1d.causal_conv1d(
        real.reshape(1), lhs, weight, None, cfg
    )
    act = jax.nn.silu(conv_out[0, :, 0, :])

    def heads(cols, n, d):
        return act[:, cols].reshape(CHUNK, n, d).transpose(1, 0, 2)[None]

    kq = N_KQ * DK
    q = heads(slice(0, kq), N_KQ, DK)
    k = heads(slice(kq, 2 * kq), N_KQ, DK)
    v = heads(slice(2 * kq, DIM), N_V, DV)
    out, state = compute_gdn.chunked_gdn(
        real.reshape(1),
        q,
        k,
        v,
        b[None, None],
        a[None, None],
        rec_prev[None],
        a_log,
        dt_bias,
        cfg,
    )
    return out[0], conv_state[0, -1, :, 0, :], state[0, 0]


def _rows(x, base, n):
    """``n`` rows of ``x`` from ``base``, zero past its end (a tile's DMA
    window: rows past the tile's own are the buffer's next rows)."""
    part = x[base : base + n]
    return jnp.concatenate([part, jnp.zeros((n - part.shape[0], *x.shape[1:]))])


def _unpack(pool, block):
    s = pool[block]
    return s[:CONV].reshape(KSIZE - 1, DIM), s[CONV:].reshape(N_V, DK, DV)


def _pack(conv, rec):
    return jnp.concatenate([conv.reshape(-1), rec.reshape(-1)])


def emulate(x, b, a, pool, qsl, dist, seq_lens, ckpt_ids, read, weights, extra):
    """The pooled op's state contract on the kernel's tile math. ``extra`` is
    ``(offsets, blocks[, state_rows])`` (``fused_conv1d_gdn``'s prefill_ckpt)."""
    out = jnp.zeros((x.shape[0], N_V, DV), jnp.float32)
    qsl = np.asarray(qsl)
    zero_conv = jnp.zeros((KSIZE - 1, DIM), jnp.float32)
    zero_rec = jnp.zeros((N_V, DK, DV), jnp.float32)

    def initial(s, seq_len, q_len):
        if seq_len - q_len > 0:
            return _unpack(pool, int(ckpt_ids[s, read[s]]))
        return zero_conv, zero_rec

    # Windowed segment (decode / verify): one tile per sequence. Identical in
    # both arms, so a plain final-state write stands in for its checkpoints.
    for s in range(int(dist[0])):
        lo, hi = int(qsl[s]), int(qsl[s + 1])
        if hi == lo:
            continue
        conv, rec = initial(s, int(seq_lens[s]), hi - lo)
        o, conv, rec = _tile(
            _rows(x, lo, CHUNK),
            _rows(b, lo, CHUNK),
            _rows(a, lo, CHUNK),
            jnp.int32(hi - lo),
            conv,
            rec,
            *weights,
        )
        out = out.at[lo:hi].set(o[: hi - lo])
        pool = pool.at[int(ckpt_ids[s, 0])].set(_pack(conv, rec))

    state_rows = extra[2] if len(extra) == 3 else None
    md = metadata.compute_per_seq_metadata(
        cfg=NS(
            batch_size=x.shape[0], chunk_size=CHUNK, tile_size=CHUNK, seq_tile_size=1
        ),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        query_start_loc=jnp.asarray(qsl, jnp.int32),
        state_indices=jnp.asarray(ckpt_ids[:, 0], jnp.int32),
        read_offsets=jnp.asarray(read, jnp.int32),
        start_seq=jnp.int32(int(dist[0])),
        end_seq=jnp.int32(int(dist[-1])),
        ckpt_indices=jnp.asarray(ckpt_ids, jnp.int32),
        prefill_ckpt=(jnp.asarray(extra[0]), jnp.asarray(extra[1])),
        state_rows=None if state_rows is None else jnp.asarray(state_rows),
    )
    data = {
        k: np.asarray(getattr(md, k).data)
        for k in (
            "p_id_to_s_idx",
            "p_id_to_r_base",
            "p_id_to_r_size",
            "p_id_is_first_tile",
            "p_id_is_last_tile",
            "p_id_to_state_override",
        )
    }
    has_initial = np.asarray(md.s_idx_has_initial_state)
    rolled_ckpt = np.asarray(md.s_idx_to_ckpt_indices)
    rolled_read = np.asarray(md.s_idx_to_read_offset)
    conv, rec = zero_conv, zero_rec
    for p in range(int(md.num_tiles)):
        s = int(data["p_id_to_s_idx"][p])
        base, size = int(data["p_id_to_r_base"][p]), int(data["p_id_to_r_size"][p])
        if data["p_id_is_first_tile"][p]:
            if has_initial[s]:
                conv, rec = _unpack(pool, int(rolled_ckpt[s, rolled_read[s]]))
            else:
                conv, rec = zero_conv, zero_rec
        o, conv, rec = _tile(
            _rows(x, base, CHUNK),
            _rows(b, base, CHUNK),
            _rows(a, base, CHUNK),
            jnp.int32(size),
            conv,
            rec,
            *weights,
        )
        out = out.at[base : base + size].set(o[:size])
        if data["p_id_is_last_tile"][p]:
            override = int(data["p_id_to_state_override"][p])
            block = override if override > 0 else int(rolled_ckpt[s, 0])
            pool = pool.at[block].set(_pack(conv, rec))
    return out, pool


def _j(t):
    return jnp.asarray(t.numpy())


class PooledOpStandIn:
    """The pooled GDN op (``gdn_pooled_ckpt_op`` / ``..._rows_op``) on the
    emulator; records the operands it was called with."""

    def __init__(self, num_extra):
        self.num_extra = num_extra
        self.calls = []

    def __call__(
        self,
        x,
        b,
        a,
        pool,
        weight,
        bias,
        a_log,
        dt_bias,
        state_ids,
        qsl,
        dist,
        seq_lens,
        slot_read,
        ckpt_indices,
        *extra,
    ):
        assert len(extra) == self.num_extra and bias is None
        read = slot_read[state_ids.long()].numpy()
        self.calls.append(
            NS(rows=x.shape[0], qsl=qsl.clone(), extra=[e.clone() for e in extra])
        )
        out, new_pool = emulate(
            _j(x),
            _j(b),
            _j(a),
            _j(pool),
            qsl.numpy(),
            dist.numpy(),
            seq_lens.numpy(),
            ckpt_indices.numpy(),
            read,
            (_j(weight), _j(a_log), _j(dt_bias)),
            tuple(e.numpy() for e in extra),
        )
        pool.copy_(torch.from_numpy(np.asarray(new_pool)))
        return torch.from_numpy(np.asarray(out))


def _fake_layer(pool, ckpt_ids, md):
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    rng = np.random.default_rng(3)
    weight = torch.from_numpy(rng.normal(size=(KSIZE, 1, DIM)).astype(np.float32) * 0.3)
    a_log = torch.from_numpy(rng.normal(size=(N_V,)).astype(np.float32) * 0.5)
    dt_bias = torch.from_numpy(rng.normal(size=(N_V,)).astype(np.float32) - 1)

    def cols(lo, hi):
        return lambda h: (h[:, lo:hi], None)

    layer = NS(
        prefix="layer",
        kv_cache=(pool,),
        num_spec=SPEC - 1,
        num_v_heads=N_V,
        tp_size=1,
        head_v_dim=DV,
        in_proj_qkv=cols(0, DIM),
        in_proj_ba=cols(DIM, DIM + 2 * N_V),
        in_proj_z=cols(DIM + 2 * N_V, DIM + 2 * N_V + N_V * DV),
        conv1d=NS(bias=None),
        A_log=a_log,
        dt_bias=dt_bias,
        gdn_pooled_op=None,
        gdn_pooled_ckpt_op=PooledOpStandIn(2),
        gdn_pooled_ckpt_rows_op=PooledOpStandIn(3),
        md=md,
    )
    layer._has_active_pooled_pcp_state = lambda kv: False
    layer._bm_state_ids = lambda ids: ids
    layer._pooled_conv_weight = lambda: weight
    layer._apply_output_projection = lambda core, z, n: core.reshape(n, -1)
    layer.forward = lambda h, cache=None, sp_half=None: (
        VllmGatedDeltaNetAttention.forward(layer, h, cache, sp_half)
    )
    return layer


TOKENS, HALF, GRID = 128, 64, 16  # GRID: a multiple of CHUNK
FOLD_UNIT = 32  # checkpoint offsets: whole fold units


def _step(lens, context, offsets, read, decodes=2):
    """A step: ``decodes`` one-token decodes, then prefills of ``lens`` with
    ``context`` rows of history, checkpoint ``offsets`` (0: none) and read
    offsets ``read``; the rest of the bucket is padding."""
    all_lens = [1] * decodes + list(lens)
    starts = np.concatenate([[0], np.cumsum(all_lens)]).astype(np.int32)
    assert starts[-1] <= TOKENS
    n = len(all_lens)
    ctx = np.asarray([5] * decodes + list(context), np.int64)
    seq_lens = (ctx + np.asarray(all_lens)).astype(np.int32)
    offs = np.asarray([0] * decodes + list(offsets), np.int32)
    assert all(o % FOLD_UNIT == 0 and o < n_rows for o, n_rows in zip(offsets, lens))
    reads = np.asarray([1] * decodes + list(read), np.int32)
    ckpt_ids = np.stack([2 + SPEC * np.arange(n) + t for t in range(SPEC)], axis=1)
    fold_blocks = 2 + SPEC * n + np.arange(n)  # one checkpoint block each
    width = 2
    table = np.zeros((n, width), np.int32)
    table[:, 1] = fold_blocks
    slot_read = np.zeros(NUM_BLOCKS, np.int32)
    slot_read[ckpt_ids[:, 0]] = reads
    md = NS(
        query_start_loc=torch.from_numpy(starts),
        seq_lens=torch.from_numpy(seq_lens),
        request_distribution=torch.tensor([decodes, n, n], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_state_indices=torch.from_numpy(ckpt_ids[:, 0].astype(np.int32)),
        mamba_ckpt_indices=torch.from_numpy(ckpt_ids.astype(np.int32)),
        mamba_slot_read_offsets=torch.from_numpy(slot_read),
        mamba_prefill_ckpt=torch.from_numpy(
            np.stack([offs, np.ones(n, np.int32)]).astype(np.int32)
        ),
        block_tables=torch.from_numpy(table.reshape(-1)),
    )
    return md, ckpt_ids, fold_blocks


def _inputs(seed):
    rng = np.random.default_rng(seed)
    width = DIM + 2 * N_V + N_V * DV
    # bf16-representable values, as the projections produce.
    h = torch.from_numpy(rng.normal(size=(TOKENS, width)).astype(np.float32))
    pool = torch.from_numpy(rng.normal(size=(NUM_BLOCKS, STATE)).astype(np.float32))
    return h.to(torch.bfloat16).float(), pool * 0.5


def _context(monkeypatch, md):
    from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op

    monkeypatch.setattr(
        gdn_attention_op,
        "get_forward_context",
        lambda: NS(attn_metadata={"layer": md}),
    )


def _whole(monkeypatch, md, ckpt_ids, h, pool):
    _context(monkeypatch, md)
    layer = _fake_layer(pool.clone(), ckpt_ids, md)
    out = layer.forward(h)
    return out, layer.kv_cache[0]


def _halves(monkeypatch, md, ckpt_ids, h, pool, grid=GRID):
    _context(monkeypatch, md)
    layer = _fake_layer(pool.clone(), ckpt_ids, md)
    plan = sp.MicroBatchPlan(TOKENS, 2, grid)
    outs = [
        layer.forward(h[g * HALF : (g + 1) * HALF], sp_half=(plan, g)) for g in range(2)
    ]
    return torch.cat(outs), layer.kv_cache[0], layer, plan


# (prefill lens, context, checkpoint offsets, read offsets); 2 decodes first.
CASES = {
    # The crossing request [23, 120) has its checkpoint at 87, in half B;
    # B restarts it at 55 (= 64 - (64 - 23) % 16) and recomputes 55..63.
    "ckpt_in_b": ([21, 97, 6], [3, 7, 0], [0, 64, 0], [0, 1, 0]),
    # Checkpoint at 35 in half A; the crossing request [3, 100) restarts at 51.
    "ckpt_in_a": ([1, 97, 20], [0, 9, 4], [0, 32, 0], [0, 1, 1]),
    # Checkpoint exactly at T/2 = 64, which is on the request's grid: B steps
    # back one grid to 48, so the checkpoint tile runs in B.
    "ckpt_at_half": ([30, 78, 14], [2, 0, 3], [0, 32, 0], [1, 0, 0]),
    # Checkpoint exactly at the restart row (23 + 32 = 55): B restarts at 39.
    "ckpt_at_restart": ([21, 90, 10], [3, 6, 0], [0, 32, 0], [0, 1, 0]),
    # A request ends exactly at T/2; the next one starts there.
    "boundary_at_half": ([62, 64], [4, 8], [32, 32], [1, 0]),
    # The crossing request starts 4 rows before T/2: A writes no state for it
    # and B reads its initial state from its own read offset.
    "restart_at_start": ([40, 18, 68], [1, 3, 11], [32, 0, 32], [0, 0, 1]),
    # The same, without history: B starts it fresh.
    "restart_at_start_fresh": ([40, 18, 68], [1, 3, 0], [32, 0, 32], [0, 0, 0]),
    # Several prefills with checkpoints on both sides and in the crossing one.
    "multiple_prefills": ([38, 70, 18], [2, 5, 7], [32, 32, 0], [1, 1, 0]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_fold_halves_match_the_whole_step(monkeypatch, case):
    lens, context, offsets, read = CASES[case]
    md, ckpt_ids, fold_blocks = _step(lens, context, offsets, read)
    h, pool = _inputs(seed=sorted(CASES).index(case))
    whole, whole_pool = _whole(monkeypatch, md, ckpt_ids, h, pool)
    halves, halves_pool, layer, plan = _halves(monkeypatch, md, ckpt_ids, h, pool)

    np.testing.assert_array_equal(halves.numpy(), whole.numpy())
    np.testing.assert_array_equal(halves_pool.numpy(), whole_pool.numpy())
    # Not vacuous: every checkpoint was written and the prefills ran.
    for o, block in zip([0, 0] + list(offsets), fold_blocks):
        changed = not np.array_equal(whole_pool[block].numpy(), pool[block].numpy())
        assert changed == (o > 0)
    # Both halves took the state-rows op; B with A's last rows in front.
    calls = layer.gdn_pooled_ckpt_rows_op.calls
    assert [c.rows for c in calls] == [HALF, HALF + plan.prefix_rows]
    assert not layer.gdn_pooled_ckpt_op.calls


def test_crossing_request_restarts_on_its_grid():
    md, _, _ = _step(*CASES["ckpt_in_b"])
    starts = md.query_start_loc
    ckpt = md.mamba_prefill_ckpt[0]
    a, b = sp.gdn_fold_halves(
        starts, md.seq_lens, md.request_distribution, ckpt, HALF, GRID
    )
    prefix = sp.fold_prefix_rows(GRID)
    assert prefix == 32 and b.prefix_rows == prefix and a.prefix_rows == 0
    # Requests: 2 decodes, [2, 23), crossing [23, 120), [120, 126).
    assert a.query_start_loc.tolist() == [0, 1, 2, 23, 64, 64]
    # A writes the crossing request's state after 32 rows (at 55), not at 64.
    assert a.state_rows.tolist() == [1, 1, 21, 32, 0]
    assert a.ckpt_offsets.tolist() == [0, 0, 0, 0, 0]  # 87 is past 55
    # B's frame is rows [32, 128): the crossing request from 55 (frame 23).
    assert b.query_start_loc.tolist() == [23, 23, 23, 23, 88, 94]
    assert b.ckpt_offsets.tolist() == [0, 0, 0, 87 - 55, 0]
    assert b.resume_from_carry.tolist() == [False, False, False, True, False]
    assert b.seq_lens.tolist() == md.seq_lens.tolist()
    assert b.distribution.tolist() == [0, 5, 5]


def test_without_alignment_the_checkpoint_misses(monkeypatch):
    """Positive control: halves as without VLLM_TPU_SP_MICROBATCH_FOLD_ALIGN
    (the gate forced open, checkpoint offsets merely re-based to each half)
    lose the crossing request's checkpoint: its tiles in half B start at
    T/2, off its grid, so no tile ends at the checkpoint."""
    md, ckpt_ids, fold_blocks = _step(*CASES["ckpt_in_b"])
    h, pool = _inputs(seed=0)
    whole, whole_pool = _whole(monkeypatch, md, ckpt_ids, h, pool)

    # The real layer refuses fold checkpoints in unaligned halves.
    _context(monkeypatch, md)
    layer = _fake_layer(pool.clone(), ckpt_ids, md)
    with pytest.raises(NotImplementedError, match="FOLD_ALIGN"):
        layer.forward(h[:HALF], sp_half=(sp.MicroBatchPlan(TOKENS, 2, 0), 0))

    # What it would compute: gdn_half metadata, offsets re-based to the half.
    starts, seq_lens = md.query_start_loc, md.seq_lens
    offsets = md.mamba_prefill_ckpt[0]
    blocks = torch.from_numpy(fold_blocks.astype(np.int32))
    state = pool.clone()
    outs = []
    for g in range(2):
        half = sp.gdn_half(starts, seq_lens, md.request_distribution, g, HALF)
        before = sp.rows_before(starts, g * HALF)
        rows = half.query_start_loc[1:] - half.query_start_loc[:-1]
        rel = offsets - before
        rel = torch.where((rel > 0) & (rel <= rows), rel, torch.zeros_like(rel))
        read = md.mamba_slot_read_offsets[md.mamba_state_indices.long()]
        if half.resume_from_carry is not None:
            read = torch.where(half.resume_from_carry, torch.zeros_like(read), read)
        layer = _fake_layer(state, ckpt_ids, md)
        x = h[g * HALF : (g + 1) * HALF]
        out, new_state = emulate(
            _j(x[:, :DIM]),
            _j(x[:, DIM : DIM + N_V]),
            _j(x[:, DIM + N_V : DIM + 2 * N_V]),
            _j(state),
            half.query_start_loc.numpy(),
            half.distribution.numpy(),
            half.seq_lens.numpy(),
            ckpt_ids,
            read.numpy(),
            (_j(layer._pooled_conv_weight()), _j(layer.A_log), _j(layer.dt_bias)),
            (rel.numpy(), blocks.numpy()),
        )
        state = torch.from_numpy(np.asarray(new_state))
        outs.append(torch.from_numpy(np.asarray(out)).reshape(HALF, -1))
    crossing_block = fold_blocks[3]
    # The whole step wrote the checkpoint; the unaligned halves never did.
    assert not np.array_equal(
        whole_pool[crossing_block].numpy(), pool[crossing_block].numpy()
    )
    np.testing.assert_array_equal(
        state[crossing_block].numpy(), pool[crossing_block].numpy()
    )
    assert not np.array_equal(torch.cat(outs).numpy(), whole.numpy())


@pytest.mark.parametrize("grid", [CHUNK, GRID, 2 * GRID])
def test_any_grid_the_tile_divides(monkeypatch, grid):
    """The restart grid may be any multiple of the kernel's tile that divides
    the checkpoint unit (the layer uses the largest prefill tile)."""
    md, ckpt_ids, _ = _step(*CASES["multiple_prefills"])
    h, pool = _inputs(seed=5)
    whole, whole_pool = _whole(monkeypatch, md, ckpt_ids, h, pool)
    halves, halves_pool, _, _ = _halves(monkeypatch, md, ckpt_ids, h, pool, grid)
    np.testing.assert_array_equal(halves.numpy(), whole.numpy())
    np.testing.assert_array_equal(halves_pool.numpy(), whole_pool.numpy())


def test_per_seq_metadata_state_rows():
    """``state_rows`` moves the final-state write to the tile ending there
    (none for 0) and leaves every other field alone."""
    lens = (1, 40, 30)
    qsl = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)

    def build(state_rows=None):
        return metadata.compute_per_seq_metadata(
            cfg=NS(
                batch_size=sum(lens), chunk_size=CHUNK, tile_size=CHUNK, seq_tile_size=1
            ),
            seq_lens=jnp.asarray([9, 50, 30], jnp.int32),
            query_start_loc=qsl,
            state_indices=jnp.asarray([1, 2, 3], jnp.int32),
            read_offsets=jnp.zeros((3,), jnp.int32),
            start_seq=jnp.int32(1),
            end_seq=jnp.int32(3),
            state_rows=state_rows,
        )

    base = build()
    n = int(base.num_tiles)
    assert n == 5 + 4
    last = np.asarray(base.p_id_is_last_tile.data)[:n]
    assert last.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 1]
    # Sequence 1 (rolled 0) writes after 16 rows; sequence 2 never.
    cut = build(jnp.asarray([1, 16, 0], jnp.int32))
    assert np.asarray(cut.p_id_is_last_tile.data)[:n].tolist() == [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # A row count at or past the query keeps the last tile.
    same = build(jnp.asarray([1, 40, 99], jnp.int32))
    np.testing.assert_array_equal(np.asarray(same.p_id_is_last_tile.data)[:n], last)
    for name in (
        "p_id_to_s_idx",
        "p_id_to_r_base",
        "p_id_to_r_size",
        "p_id_is_first_tile",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(cut, name).data)[:n],
            np.asarray(getattr(base, name).data)[:n],
        )


_SC_AG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather"


@pytest.mark.parametrize(
    "tail_fold,align,expected,grid",
    [
        (True, False, False, 0),
        (True, True, True, 128),
        (False, True, True, 0),  # alignment only matters under the fold
        (False, False, True, 0),
    ],
)
def test_fold_align_lifts_the_tail_fold_gate(
    monkeypatch, tail_fold, align, expected, grid
):
    sp.reset_microbatch()
    try:
        monkeypatch.setenv("LIBTPU_INIT_ARGS", f"{_SC_AG}=true")
        monkeypatch.setattr(
            sp.envs, "VLLM_TPU_SP_MICROBATCH_OVERLAP", True, raising=False
        )
        monkeypatch.setattr(
            sp.envs, "VLLM_TPU_SP_MICROBATCH_OVERLAP_FORCE_OFF", False, raising=False
        )
        monkeypatch.setattr(
            sp.envs, "VLLM_TPU_MAMBA_TAIL_FOLD", tail_fold, raising=False
        )
        monkeypatch.setattr(
            sp.envs, "VLLM_TPU_SP_MICROBATCH_FOLD_ALIGN", align, raising=False
        )
        monkeypatch.setattr(sp.envs, "TPU_GDN_CHUNK_PAIR", False, raising=False)
        assert sp.microbatch_enabled() is expected
        assert sp.fold_align_grid() == grid
    finally:
        sp.reset_microbatch()
