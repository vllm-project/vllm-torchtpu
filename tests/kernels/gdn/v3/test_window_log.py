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
"""GDN v3 CheckpointMode.LOG against SELECT on the same inputs.

The plan checks run anywhere. `test_log_matches_select_tpu` drives the pooled
kernel in both modes, each on its own copy of a pool at the served Qwen3.8
Flash decode geometry (f8 pool of 512-row kernel blocks, FP32 state, 2 key
and 6 value heads, K = 3), through a random schedule of verify windows,
shorter windows, one-token steps, PER_SEQ chunks, fresh, landed and seeded
starts, and checks bit for bit on every step:

- the kernel outputs;
- checkpoint block 0's state and conv rows after the step;
- the state and conv state the next step loads: SELECT's checkpoint
  [read_offset] against LOG's checkpoint 0 with the first read_offset logged
  updates applied (decoded and folded by a small kernel built from the same
  vmem_ldst functions the LOG kernel loads with).

It also folds on the host in numpy (IEEE f32, one rounding per multiply and
per add) and reports how often that equals SELECT's checkpoint, which shows
whether the kernels' state update contracts into a fused multiply-add.
"""

import collections
import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import compute_conv1d, compute_gdn, config, vmem_ldst, wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

N_KQ, N_V, D, K, DIM, KS = 2, 6, 128, 3, 1280, 4
W = K + 1
ROWS, PAYLOAD, LANES = 512, (1, 4), 256
F32 = jnp.dtype(jnp.float32)

SELECT_PLAN = config.StateSourcePlan(
    stride=1,
    conv=config.StateRegion(0, 1, 384, 16, F32, 1, 15),
    recurrent=config.StateRegion(0, 1, 0, 384, F32, 2, 768),
)
LOG_PLAN = dataclasses.replace(
    SELECT_PLAN,
    log=config.StateRegion(0, 1, 400, 14, F32, 2, 27),
    log_conv=config.StateRegion(0, 1, 414, 15, F32, 1, 15),
)
STATE_ROWS = 400  # recurrent + conv rows of a block
LOG_END = 429


def _validate(plan, **overrides):
    kwargs = dict(
        log_positions=K,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D,
        v_head_dim=D,
        dim_size=DIM,
        block_rows=ROWS,
        raw_lanes=LANES,
        raw_row_bytes=1024,
    )
    kwargs.update(overrides)
    config.validate_log_plan(plan, **kwargs)


class TestLogPlan:
    def test_served_plan_is_valid(self):
        _validate(LOG_PLAN)
        assert config.log_layout_rows(K, N_KQ, N_V) == (3, 9, 27)

    def test_rejects_overlap_with_conv(self):
        bad = dataclasses.replace(
            LOG_PLAN,
            log=dataclasses.replace(LOG_PLAN.log, row0=390),
            log_conv=dataclasses.replace(LOG_PLAN.log_conv, row0=404),
        )
        with pytest.raises(ValueError, match="overlap"):
            _validate(bad)

    def test_rejects_wrong_rows_used(self):
        bad = dataclasses.replace(
            LOG_PLAN, log=dataclasses.replace(LOG_PLAN.log, rows_used=26)
        )
        with pytest.raises(ValueError, match="rows_used"):
            _validate(bad)

    def test_rejects_gap_between_regions(self):
        bad = dataclasses.replace(
            LOG_PLAN, log_conv=dataclasses.replace(LOG_PLAN.log_conv, row0=415)
        )
        with pytest.raises(ValueError, match="follow"):
            _validate(bad)

    def test_rejects_block_overrun(self):
        with pytest.raises(ValueError, match="overrun"):
            _validate(LOG_PLAN, block_rows=420)

    def test_select_metadata_unchanged(self):
        # The new fields stay out of SELECT kernels' metadata and name.
        cfg = _probe_cfg(SELECT_PLAN, config.CheckpointMode.SELECT, 0)
        meta = cfg.get_metadata()
        assert not any("log" in key for key in meta), meta
        assert "checkpoint_mode" not in meta
        assert not cfg.get_kernel_name().endswith("_log")


def _probe_cfg(plan, mode=config.CheckpointMode.LOG, log_positions=K):
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=F32,
            act_out=jnp.dtype(jnp.bfloat16),
            compute=F32,
            recurrent_state=F32,
            conv_state=F32,
        ),
        batch_size=1,
        dim_size=DIM,
        kernel_size=KS,
        tile_size=1,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D,
        v_head_dim=D,
        state_plan=plan,
        window_size=W,
        checkpoint_mode=mode,
        log_positions=log_positions,
    )


class TestPaddedSlots:
    """Padded slots of a LOG-mode batch move no state or log rows: the log
    refs read on `first tile and has_initial_state` and write on `last tile
    and r_size > 0`, the predicates of the state refs."""

    def test_padded_slots_schedule_nothing(self):
        from vllm_torchtpu.kernels.gdn.v3 import metadata

        num_slots, batched, per_seq = 32, 5, 3
        qlens = np.zeros(num_slots, np.int32)
        qlens[:batched] = [4, 3, 1, 4, 2]
        qlens[batched : batched + per_seq] = [9, 17, 33]
        qsl = jnp.asarray(np.concatenate([[0], np.cumsum(qlens)]), jnp.int32)
        seq_lens = jnp.asarray(qlens + 100 * (qlens > 0), jnp.int32)
        state_indices = jnp.asarray(np.arange(num_slots) * W, jnp.int32)
        offsets = jnp.asarray(np.arange(num_slots) % W, jnp.int32)
        ckpt = jnp.asarray(np.arange(num_slots * W).reshape(num_slots, W), jnp.int32)
        cfg = dataclasses.replace(_probe_cfg(LOG_PLAN), tile_size=8, batch_size=128)
        meta = metadata.compute_batched_seq_metadata(
            cfg, seq_lens, qsl, state_indices, offsets, jnp.int32(batched), ckpt
        )
        first = np.asarray(meta.p_id_is_first_tile.data)
        last = np.asarray(meta.p_id_is_last_tile.data)
        r_size = np.asarray(meta.p_id_to_r_size.data)
        has_init = np.asarray(meta.s_idx_has_initial_state)
        s_idx = np.asarray(meta.p_id_to_s_idx.data)
        reads = first & has_init[s_idx]
        writes = last & (np.minimum(r_size, cfg.state_out_positions) > 0)
        assert int(meta.num_tiles) == 1
        assert not reads[batched:].any() and not writes[batched:].any()
        assert reads[:batched].all() and writes[:batched].all()

        per_seq_cfg = dataclasses.replace(
            cfg, mode=config.GDNMode.PER_SEQ, tile_size=16, window_size=1
        )
        meta = metadata.compute_per_seq_metadata(
            per_seq_cfg,
            seq_lens,
            qsl,
            state_indices,
            offsets,
            jnp.int32(batched),
            jnp.int32(batched + per_seq),
            ckpt,
        )
        num_tiles = int(meta.num_tiles)
        assert num_tiles == 1 + 2 + 3
        s_idx = np.asarray(meta.p_id_to_s_idx.data)[:num_tiles]
        assert set(s_idx.tolist()) == {0, 1, 2}  # rolled: slots 5, 6, 7


class TestFoldMatchesWindow:
    """The fold of the logged operands reproduces SELECT's checkpoints, on the
    kernel's compute functions run eagerly (one rounding per op), for every
    window size and offset."""

    def test_fold_reproduces_every_checkpoint(self):
        rng = np.random.default_rng(7)
        seqs, aligned = 3, D
        select_cfg = _probe_cfg(SELECT_PLAN, config.CheckpointMode.SELECT, 0)
        log_cfg = _probe_cfg(LOG_PLAN)
        select_cfg = dataclasses.replace(select_cfg, tile_size=seqs)
        log_cfg = dataclasses.replace(log_cfg, tile_size=seqs)

        def arr(*shape, scale=1.0):
            return jnp.asarray(rng.normal(0, scale, shape), jnp.float32)

        a_log = jnp.pad(jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32), (0, aligned - N_V))
        dt_bias = jnp.pad(arr(N_V), (0, aligned - N_V))
        state = arr(seqs, N_V, D, D, scale=0.1)
        conv = arr(seqs, KS - 1, 1, DIM)
        qkv = arr(seqs, W, 1, DIM)
        conv_w = arr(KS, 1, DIM, scale=0.5)
        b = arr(seqs, W, 1, aligned)
        a = arr(seqs, W, 1, aligned)
        for r in range(1, W + 1):
            real = jnp.full((seqs,), r, jnp.int32)
            lhs = jnp.concat([conv, qkv], axis=1)
            outs = {}
            for name, cfg in (("select", select_cfg), ("log", log_cfg)):
                conv_out, conv_ckpts = compute_conv1d.causal_conv1d(real, lhs, conv_w, None, cfg)
                act = jax.nn.silu(conv_out)
                q = vmem_ldst.load_compact_heads(act, N_KQ, D)
                k = vmem_ldst.load_compact_heads(act, N_KQ, D, N_KQ * D)
                v = vmem_ldst.load_compact_heads(act, N_V, D, 2 * N_KQ * D)
                outs[name] = (
                    conv_ckpts,
                    compute_gdn.recurrent_gdn(
                        real, q, k, v, jnp.expand_dims(b, 1), jnp.expand_dims(a, 1),
                        state, a_log, dt_bias, cfg,
                    ),
                )
            select_conv, (select_out, select_states) = outs["select"]
            log_conv, (log_out, log_state0, (gates, keys, updates)) = outs["log"]
            np.testing.assert_array_equal(np.asarray(select_out), np.asarray(log_out))
            for idx in range(seqs):
                for off in range(r):
                    rec, cv = vmem_ldst.fold_window_log(
                        log_state0[idx, 0],
                        log_conv[idx, 0],
                        gates[idx].reshape(K, 1, 1, D),
                        keys[idx],
                        updates[idx],
                        qkv[idx, 1:],
                        jnp.int32(off),
                        log_cfg,
                    )
                    np.testing.assert_array_equal(
                        np.asarray(rec).view(np.uint32),
                        np.asarray(select_states[idx, off]).view(np.uint32),
                        err_msg=f"recurrent r={r} offset={off}",
                    )
                    np.testing.assert_array_equal(
                        np.asarray(cv).view(np.uint32),
                        np.asarray(select_conv[idx, off]).view(np.uint32),
                        err_msg=f"conv r={r} offset={off}",
                    )


# --------------------------------------------------------------------------
# TPU part.
# --------------------------------------------------------------------------

NUM_SEQS = 32
NUM_STEPS = int(os.environ.get("F7_GATE1_STEPS", "64"))
TOKENS = 512
NUM_BLOCKS = 1 + 4 * W * NUM_SEQS
PROBES = NUM_SEQS * W  # one row per (sequence, offset 0..K)


@functools.partial(jax.jit, static_argnames=("plan",), donate_argnums=(3,))
def _run(qkv, b, a, pool, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck, *, plan):
    return wrapper.fused_conv1d_gdn(
        Conv1DInput(qkv, jnp.bfloat16),
        b,
        a,
        None,
        None,
        conv_w,
        None,
        a_log,
        dt_bias,
        qsl,
        si,
        dist,
        sl,
        ro,
        ck,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KS,
        state_source=pool,
        state_plan=plan,
        num_spec_tokens=K,
        conv_weight_kmajor=True,
    )


def _probe_kernel(
    blocks_ref,
    offsets_ref,
    pool_ref,
    rec_out,
    conv_out,
    gates_out,
    keys_out,
    updates_out,
    conv_in_out,
    rec_tile,
    conv_tile,
    log_tile,
    log_conv_tile,
    sem,
    *,
    cfg,
):
    """Decodes one block's state and log and applies `offset` logged updates,
    through the LOG kernel's own load functions."""
    i = pl.program_id(0)
    block = blocks_ref[i]
    plan = cfg.state_plan
    for region, tile in (
        (plan.recurrent, rec_tile),
        (plan.conv, conv_tile),
        (plan.log, log_tile),
        (plan.log_conv, log_conv_tile),
    ):
        copy = pltpu.make_async_copy(
            pool_ref.at[
                pl.ds(block * plan.stride + region.kb0, region.nblocks),
                pl.ds(region.row0, region.nrows),
            ],
            tile.at[0, 0],
            sem,
        )
        copy.start()
        copy.wait()
    rec = vmem_ldst._load_recurrent_state(rec_tile, cfg, 0)
    conv = vmem_ldst._load_conv_state(conv_tile, cfg, 0)
    log = vmem_ldst.load_window_log(log_tile.at[0, 0], log_conv_tile.at[0, 0], cfg)
    rec, conv = vmem_ldst.fold_window_log(rec, conv, *log, offsets_ref[i], cfg)
    rec_out[0] = rec
    conv_out[0] = conv
    gates_out[0] = log[0]
    keys_out[0] = log[1]
    updates_out[0] = log[2]
    conv_in_out[0] = log[3]


@jax.jit
def _probe(pool, blocks, offsets):
    cfg = _probe_cfg(LOG_PLAN)
    n = blocks.shape[0]
    plan = cfg.state_plan

    def tile(region):
        return pltpu.VMEM((1, 1, region.nblocks, region.nrows, *PAYLOAD, LANES), pool.dtype)

    def out(*shape):
        return pl.BlockSpec((1, *shape), lambda i, *_: (i,) + (0,) * len(shape))

    shapes = [
        ((N_V, D, D), jnp.float32),
        ((KS - 1, 1, DIM), jnp.float32),
        ((K, 1, 1, D), jnp.float32),
        ((K, N_KQ, 1, D), jnp.float32),
        ((K, N_V, 1, D), jnp.float32),
        ((K, 1, DIM), jnp.float32),
    ]
    return pl.pallas_call(
        functools.partial(_probe_kernel, cfg=cfg),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(n,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=[out(*s) for s, _ in shapes],
            scratch_shapes=[
                tile(plan.recurrent),
                tile(plan.conv),
                tile(plan.log),
                tile(plan.log_conv),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=[jax.ShapeDtypeStruct((n, *s), dt) for s, dt in shapes],
    )(blocks, offsets, pool)


POOL_DTYPE = jnp.dtype(jnp.float8_e4m3fn)  # the served pool's; 1 KiB rows


def _u8(pool):
    return jax.lax.bitcast_convert_type(pool, jnp.uint8)


def _f8(raw):
    """The pool from its bytes (a trailing byte axis for wider pool dtypes)."""
    return jax.lax.bitcast_convert_type(raw, POOL_DTYPE)


# Copies and byte comparisons run as Pallas DMAs and Mosaic bitcasts: XLA
# element ops on the f8 pool (a dynamic-update-slice with its bounds select,
# or a uint8 bitcast view) are not guaranteed to move raw bytes unchanged.
def _hbm():
    return pl.BlockSpec(memory_space=pltpu.HBM)


def _smem():
    return pl.BlockSpec(memory_space=pltpu.SMEM)


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("r0", "r1"))
def _copy_rows_many(pool, dst, src, *, r0, r1):
    """pool[dst[i], r0:r1] = pool[src[i], r0:r1] for every i, in order."""

    def kernel(dst_ref, src_ref, _, pool_ref, sem):
        for i in range(dst.shape[0]):
            copy = pltpu.make_async_copy(
                pool_ref.at[pl.ds(src_ref[i], 1), pl.ds(r0, r1 - r0)],
                pool_ref.at[pl.ds(dst_ref[i], 1), pl.ds(r0, r1 - r0)],
                sem,
            )
            copy.start()
            copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        in_specs=[_smem(), _smem(), _hbm()],
        out_specs=_hbm(),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        input_output_aliases={2: 0},
    )(dst, src, pool)


@functools.partial(jax.jit, donate_argnums=0, static_argnames=("r0", "r1"))
def _copy_rows_from_many(dst_pool, src_pool, dst, src, *, r0, r1):
    """dst_pool[dst[i], r0:r1] = src_pool[src[i], r0:r1] for every i."""

    def kernel(dst_ref, src_ref, _, src_pool_ref, pool_ref, sem):
        for i in range(dst.shape[0]):
            copy = pltpu.make_async_copy(
                src_pool_ref.at[pl.ds(src_ref[i], 1), pl.ds(r0, r1 - r0)],
                pool_ref.at[pl.ds(dst_ref[i], 1), pl.ds(r0, r1 - r0)],
                sem,
            )
            copy.start()
            copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(dst_pool.shape, dst_pool.dtype),
        in_specs=[_smem(), _smem(), _hbm(), _hbm()],
        out_specs=_hbm(),
        scratch_shapes=[pltpu.SemaphoreType.DMA],
        input_output_aliases={2: 0},
    )(dst, src, dst_pool, src_pool)


def _copy_rows(pool, dst, src, *, r0, r1):
    one = lambda x: jnp.asarray([x], jnp.int32)
    return _copy_rows_many(pool, one(dst), one(src), r0=r0, r1=r1)


def _copy_rows_from(dst_pool, src_pool, dst, src, *, r0, r1):
    one = lambda x: jnp.asarray([x], jnp.int32)
    return _copy_rows_from_many(dst_pool, src_pool, one(dst), one(src), r0=r0, r1=r1)


@functools.partial(jax.jit, static_argnames=("r0", "r1"))
def _row_mask_equal(pool_a, blocks_a, pool_b, blocks_b, *, r0, r1):
    """[n, r1 - r0, 1] int32: 1 where row r of pool_a block blocks_a[i] has
    the bytes of row r of pool_b block blocks_b[i]."""
    rows = r1 - r0

    def kernel(ba_ref, bb_ref, a_ref, b_ref, out_ref, ta, tb, sem):
        i = pl.program_id(0)
        for src, idx, tile in ((a_ref, ba_ref, ta), (b_ref, bb_ref, tb)):
            copy = pltpu.make_async_copy(
                src.at[pl.ds(idx[i], 1), pl.ds(r0, rows)], tile, sem
            )
            copy.start()
            copy.wait()

        def words(tile):
            ref = tile.at[0]
            if jnp.dtype(ref.dtype).itemsize != 4:
                ref = ref.bitcast(jnp.int32)
            return ref.reshape(rows, -1)[...]

        diff = (words(ta) != words(tb)).astype(jnp.int32)
        out_ref[0] = 1 - jnp.max(diff, axis=-1, keepdims=True)

    n = blocks_a.shape[0]
    tile = pltpu.VMEM((1, rows, *pool_a.shape[2:]), pool_a.dtype)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(n,),
            in_specs=[_hbm(), _hbm()],
            out_specs=pl.BlockSpec((1, rows, 1), lambda i, *_: (i, 0, 0)),
            scratch_shapes=[tile, tile, pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct((n, rows, 1), jnp.int32),
    )(blocks_a, blocks_b, pool_a, pool_b)


def _rows_equal(pool_a, blocks_a, pool_b, blocks_b, *, rows):
    eq = _row_mask_equal(pool_a, blocks_a, pool_b, blocks_b, r0=0, r1=rows)
    return jnp.all(eq[..., 0] == 1, axis=1)


def _rows_unchanged(before, after):
    """[blocks, rows] mask of source rows whose bytes did not change."""
    blocks = jnp.arange(before.shape[0], dtype=jnp.int32)
    return _row_mask_equal(before, blocks, after, blocks, r0=0, r1=ROWS)[..., 0] == 1


def _pool_copy(pool):
    blocks = jnp.arange(pool.shape[0], dtype=jnp.int32)
    return _copy_rows_from_many(jnp.zeros_like(pool), pool, blocks, blocks, r0=0, r1=ROWS)


def _seed_blocks(pool_log, pool_select, dst, src):
    """LOG pool with block `dst[i]`'s state rows replaced by SELECT's block `src[i]`."""
    return _copy_rows_from_many(_pool_copy(pool_log), pool_select, dst, src, r0=0, r1=STATE_ROWS)


@jax.jit
def _bits_equal(a, b):
    a = jax.lax.bitcast_convert_type(a, jnp.uint32)
    b = jax.lax.bitcast_convert_type(b, jnp.uint32)
    return jnp.all(a == b, axis=tuple(range(1, a.ndim)))


def _numpy_fold(rec, conv, gates, keys, updates, conv_in, offset):
    """Host fold: one IEEE f32 rounding per multiply and per add."""
    rec = rec.astype(np.float32).copy()
    for j in range(offset):
        g = gates[j, 0, 0, :N_V].astype(np.float32)[:, None, None]
        # +0.0 maps -0.0 to +0.0, as the kernels' lane-to-sublane transpose does.
        k_t = np.repeat(keys[j, :, 0, :] + np.float32(0.0), N_V // N_KQ, axis=0)
        state_updated = rec * g
        state_new = k_t[:, :, None] * updates[j]
        rec = state_updated + state_new
    rows = np.concatenate([conv, conv_in], axis=0)
    return rec, rows[offset : offset + KS - 1]


@dataclasses.dataclass
class _Seq:
    group: list[int]
    seq_len: int = 0
    ro_select: int = 0
    ro_log: int = 0
    last_r: int = 0  # real size of its last BATCHED window, 0 otherwise
    seeded: bool = False  # seeded and not run since


def test_log_matches_select_tpu(monkeypatch):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    monkeypatch.setenv("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")
    rng = np.random.default_rng(int(os.environ.get("F7_GATE1_SEED", "20261111")))
    # Diagnostics: preloaded-state check, a second run of each kernel, a LOG run
    # resumed from SELECT's state at offset 0, per-row output differences; a
    # failing sequence restarts fresh and the run goes on.
    diag = os.environ.get("F7_GATE1_DIAG") == "1"
    restart = set()

    raw = rng.integers(0, 256, size=NUM_BLOCKS * ROWS * 1024, dtype=np.uint8)
    byte_axis = (POOL_DTYPE.itemsize,) if POOL_DTYPE.itemsize > 1 else ()
    raw = raw.reshape(NUM_BLOCKS, ROWS, *PAYLOAD, LANES, *byte_axis)
    pool_select = _f8(jnp.asarray(raw))
    pool_log = _pool_copy(pool_select)
    del raw
    conv_w = jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32)
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32)
    dt_bias = jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16)

    free = list(range(1, NUM_BLOCKS))
    rng.shuffle(free)

    def new_group():
        return [free.pop() for _ in range(W)]

    seqs = [_Seq(group=new_group()) for _ in range(NUM_SEQS)]
    counts = dict.fromkeys(
        [
            "batched_full",
            "batched_short",
            "one_token",
            "per_seq",
            "fresh_batched",
            "fresh_per_seq",
            "landed",
            "seeded",
            "fold_load_batched",
            "fold_load_per_seq",
            "one_token_then_verify",
            "per_seq_then_batched",
            "batched_then_per_seq",
            "skipped",
            "probes",
            "numpy_fold_checked",
            "numpy_fold_equal",
        ],
        0,
    )
    failures = []
    prev_kind = ["fresh"] * NUM_SEQS

    for step in range(NUM_STEPS):
        # Starts before the step: fresh, landed (a state written into block 0
        # from outside, a previous owner's log left in place, offset 0) and
        # seeded (a block crossing: SELECT copies checkpoint[offset] into the
        # new checkpoint 0 and resets the offset; LOG copies block 0 whole,
        # log included, and keeps the offset).
        fresh = [step == 0] * NUM_SEQS
        step_failures = len(failures)
        for s, seq in enumerate(seqs):
            if step == 0:
                continue
            u = rng.random()
            if s in restart:
                u = 0.0
            if u < 0.04:
                free.extend(seq.group)
                seq.group, seq.seq_len, seq.ro_select, seq.ro_log = new_group(), 0, 0, 0
                seq.last_r, seq.seeded = 0, False
                fresh[s] = True
            elif u < 0.08 and seq.seq_len > 0:
                donors = [d for d in range(NUM_SEQS) if d != s and seqs[d].seq_len > 0]
                d0 = seqs[donors[rng.integers(len(donors))]].group[0]
                old = seq.group
                seq.group = new_group()
                b0 = seq.group[0]
                pool_log = _copy_rows(pool_log, b0, d0, r0=0, r1=STATE_ROWS)
                pool_select = _copy_rows_from(pool_select, pool_log, b0, d0, r0=0, r1=STATE_ROWS)
                stale = seqs[rng.integers(NUM_SEQS)].group[0]
                pool_log = _copy_rows(pool_log, b0, stale, r0=STATE_ROWS, r1=LOG_END)
                free.extend(old)
                seq.seq_len += int(rng.integers(1, 1000))
                seq.ro_select = seq.ro_log = seq.last_r = 0
                seq.seeded = False
                counts["landed"] += 1
            elif u < 0.14 and seq.seq_len > 0:
                old = seq.group
                seq.group = new_group()
                pool_select = _copy_rows(
                    pool_select, seq.group[0], old[seq.ro_select], r0=0, r1=STATE_ROWS
                )
                pool_log = _copy_rows(pool_log, seq.group[0], old[0], r0=0, r1=ROWS)
                free.extend(old)
                seq.ro_select = seq.last_r = 0
                seq.seeded = True
                counts["seeded"] += 1

        # Schedule: BATCHED windows first, PER_SEQ chunks after them.
        batched, per_seq = [], []
        for s, seq in enumerate(seqs):
            if not fresh[s] and rng.random() < 0.1:
                counts["skipped"] += 1
                continue
            u = rng.random()
            if u < 0.55:
                r = W
                batched.append((s, r, "verify"))
            elif u < 0.70:
                r = int(rng.integers(2, W))
                batched.append((s, r, "verify"))
            elif u < 0.85:
                batched.append((s, 1, "one_token"))
            elif len(per_seq) < 6:
                per_seq.append((s, int(rng.integers(W + 1, 34)), "per_seq"))
            else:
                batched.append((s, 1, "one_token"))
        order = batched + per_seq

        qlens = np.zeros(NUM_SEQS, np.int32)
        si = np.zeros(NUM_SEQS, np.int32)
        ck = np.zeros((NUM_SEQS, W), np.int32)
        sl = np.zeros(NUM_SEQS, np.int32)
        ro_sel = np.zeros(NUM_SEQS, np.int32)
        ro_log = np.zeros(NUM_SEQS, np.int32)
        for slot, (s, n, kind) in enumerate(order):
            seq = seqs[s]
            qlens[slot] = n
            si[slot] = seq.group[0]
            ck[slot] = seq.group
            sl[slot] = seq.seq_len + n
            ro_sel[slot] = seq.ro_select
            ro_log[slot] = seq.ro_log
            if fresh[s]:
                counts["fresh_per_seq" if kind == "per_seq" else "fresh_batched"] += 1
            elif seq.ro_log > 0:
                counts["fold_load_per_seq" if kind == "per_seq" else "fold_load_batched"] += 1
            if prev_kind[s] == "one_token" and kind == "verify":
                counts["one_token_then_verify"] += 1
            if prev_kind[s] == "per_seq" and kind != "per_seq":
                counts["per_seq_then_batched"] += 1
            if prev_kind[s] in ("verify", "one_token") and kind == "per_seq":
                counts["batched_then_per_seq"] += 1
            if kind == "verify":
                counts["batched_full" if n == W else "batched_short"] += 1
            else:
                counts[kind] += 1
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        assert qsl[-1] <= TOKENS
        dist = np.array([len(batched), len(order), len(order)], np.int32)

        qkv = jnp.asarray(rng.normal(0, 1, (TOKENS, 1, DIM)), jnp.float32)
        b = jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32)
        a = jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32)
        common = (jnp.asarray(qsl), jnp.asarray(si), jnp.asarray(dist), jnp.asarray(sl))

        restart = set()
        before_select, before_log = _pool_copy(pool_select), _pool_copy(pool_log)
        if diag:
            # The state each scheduled sequence is about to load, both modes.
            slots = range(len(order))
            pre_sel = _probe(
                pool_select,
                jnp.asarray([ck[i, ro_sel[i]] for i in slots] + [0] * (PROBES - len(order)), jnp.int32),
                jnp.zeros((PROBES,), jnp.int32),
            )
            pre_log = _probe(
                pool_log,
                jnp.asarray([si[i] for i in slots] + [0] * (PROBES - len(order)), jnp.int32),
                jnp.asarray(list(ro_log[: len(order)]) + [0] * (PROBES - len(order)), jnp.int32),
            )
            pre_rec = np.asarray(_bits_equal(pre_sel[0], pre_log[0]))[: len(order)]
            pre_conv = np.asarray(_bits_equal(pre_sel[1], pre_log[1]))[: len(order)]
            with_state = np.asarray([sl[i] > qlens[i] for i in slots], bool)
            for i in np.nonzero(~(pre_rec & pre_conv) & with_state)[0]:
                print(f"DIAG step {step}: preloaded state differs for sequence {order[i][0]} slot {i} "
                      f"(rec {bool(pre_rec[i])}, conv {bool(pre_conv[i])}, ro_sel {ro_sel[i]}, ro_log {ro_log[i]})")
            del pre_sel, pre_log
            alt_pool = _seed_blocks(
                before_log,
                before_select,
                jnp.asarray([si[i] for i in slots], jnp.int32),
                jnp.asarray([ck[i, ro_sel[i]] for i in slots], jnp.int32),
            )
            _, out_alt = _run(
                qkv, b, a, alt_pool, conv_w, a_log, dt_bias, *common,
                jnp.zeros((NUM_SEQS,), jnp.int32), jnp.asarray(ck), plan=LOG_PLAN,
            )
            twice_sel_pool, twice_sel = _run(
                qkv, b, a, _pool_copy(before_select), conv_w, a_log, dt_bias, *common,
                jnp.asarray(ro_sel), jnp.asarray(ck), plan=SELECT_PLAN,
            )
            twice_log_pool, twice_log = _run(
                qkv, b, a, _pool_copy(before_log), conv_w, a_log, dt_bias, *common,
                jnp.asarray(ro_log), jnp.asarray(ck), plan=LOG_PLAN,
            )
        pool_select, out_select = _run(
            qkv, b, a, pool_select, conv_w, a_log, dt_bias, *common,
            jnp.asarray(ro_sel), jnp.asarray(ck), plan=SELECT_PLAN,
        )
        pool_log, out_log = _run(
            qkv, b, a, pool_log, conv_w, a_log, dt_bias, *common,
            jnp.asarray(ro_log), jnp.asarray(ck), plan=LOG_PLAN,
        )

        # 1. Outputs, every row.
        out_eq = np.asarray(
            jax.lax.bitcast_convert_type(out_select, jnp.uint16)
            == jax.lax.bitcast_convert_type(out_log, jnp.uint16)
        )
        bad_rows = np.nonzero(~out_eq.reshape(TOKENS, -1).all(axis=1))[0]

        def row_owner(row):
            slot = int(np.searchsorted(qsl, row, side="right") - 1)
            if slot >= len(order):
                return f"padding row {row}"
            return f"row {row} seq {order[slot][0]} slot {slot} pos {row - qsl[slot]} kind {order[slot][2]}"

        if diag:
            def rows_differ(x, y):
                eq = np.asarray(
                    jax.lax.bitcast_convert_type(x, jnp.uint16)
                    == jax.lax.bitcast_convert_type(y, jnp.uint16)
                ).reshape(TOKENS, -1)
                return np.nonzero(~eq.all(axis=1))[0], eq

            for row in bad_rows:
                xs = np.asarray(out_select[row]).astype(np.float32)
                xl = np.asarray(out_log[row]).astype(np.float32)
                print(f"DIAG step {step}: output {row_owner(row)}: {(~out_eq.reshape(TOKENS, -1)[row]).sum()} "
                      f"elements differ, max |d| {np.abs(xs - xl).max():.3g}")
            for name, x, y in (
                ("select run twice", out_select, twice_sel),
                ("log run twice", out_log, twice_log),
                ("log resumed from select state vs select", out_alt, out_select),
            ):
                rows, _ = rows_differ(x, y)
                print(f"DIAG step {step}: {name}: {rows.size} rows differ"
                      + (f", first {row_owner(int(rows[0]))}" if rows.size else ""))
            for name, x, y in (
                ("select pool run twice", twice_sel_pool, pool_select),
                ("log pool run twice", twice_log_pool, pool_log),
            ):
                changed = ~np.asarray(_rows_unchanged(x, y))
                print(f"DIAG step {step}: {name}: {int(changed.sum())} rows differ")
            del twice_sel_pool, twice_log_pool, alt_pool, out_alt, twice_sel, twice_log
        if bad_rows.size:
            row = int(bad_rows[0])
            slot = int(np.searchsorted(qsl, row, side="right") - 1)
            failures.append(
                f"step {step}: outputs differ in {bad_rows.size} rows, first row {row} "
                f"(slot {slot}, position {row - qsl[min(slot, NUM_SEQS)]})"
            )

        # Offsets after the step.
        for s, n, kind in order:
            seq = seqs[s]
            seq.seq_len += n
            if kind == "verify":
                accepted = int(rng.integers(1, n + 1))
                seq.ro_select = seq.ro_log = accepted - 1
                seq.last_r = n
            else:
                seq.ro_select = seq.ro_log = 0
                seq.last_r = 1 if kind == "one_token" else 0
            seq.seeded = False
            prev_kind[s] = kind

        # Padded slots and rows, skipped sequences and free blocks: no row
        # outside what a scheduled sequence owns changes. SELECT writes
        # checkpoints 0..r-1 (checkpoint 0 for PER_SEQ), LOG block 0's state
        # rows and, BATCHED only, its log rows.
        allowed_select = np.zeros((NUM_BLOCKS, ROWS), bool)
        allowed_log = np.zeros((NUM_BLOCKS, ROWS), bool)
        for s_, n_, kind_ in order:
            group = seqs[s_].group
            for j in range(1 if kind_ == "per_seq" else min(n_, W)):
                allowed_select[group[j], :STATE_ROWS] = True
            allowed_log[group[0], : (STATE_ROWS if kind_ == "per_seq" else LOG_END)] = True
        for name, before, after, allowed in (
            ("select", before_select, pool_select, allowed_select),
            ("log", before_log, pool_log, allowed_log),
        ):
            changed = ~np.asarray(_rows_unchanged(before, after))
            stray = np.argwhere(changed & ~allowed)
            if stray.size:
                failures.append(
                    f"step {step}: {name} kernel wrote {len(stray)} rows it does not own, "
                    f"first (block, row) {tuple(int(x) for x in stray[0])}"
                )
            counts[f"rows_written_{name}"] = counts.get(f"rows_written_{name}", 0) + int(changed.sum())
        del before_select, before_log

        # 2. Checkpoint block 0 bytes (state and conv rows) of every
        # sequence that ran.
        ran = [s for s, _, _ in order]
        blocks0 = jnp.asarray([seqs[s].group[0] for s in ran], jnp.int32)
        eq0 = np.asarray(_rows_equal(pool_select, blocks0, pool_log, blocks0, rows=STATE_ROWS))
        for idx in np.nonzero(~eq0)[0]:
            failures.append(f"step {step}: block 0 rows differ for sequence {ran[idx]}")

        # 3. The state the next step loads, for every sequence and for every
        # offset its last window can name (offset 0..r-1).
        sel_blocks, log_blocks, log_offs, owners = [], [], [], []
        for s, seq in enumerate(seqs):
            if seq.seeded:
                # Seeded and not run since: SELECT's checkpoint 0 holds the
                # copy of its checkpoint[offset], LOG's block 0 the old block 0.
                pairs = [(seq.group[0], seq.ro_log)]
            else:
                pairs = [(seq.group[j], j) for j in range(max(seq.last_r, seq.ro_log + 1))]
            for sel_block, off in pairs:
                sel_blocks.append(sel_block)
                log_blocks.append(seq.group[0])
                log_offs.append(off)
                owners.append((s, off, off == seq.ro_log))
        pad = PROBES - len(owners)
        sel_blocks += [0] * pad
        log_blocks += [0] * pad
        log_offs += [0] * pad
        zeros = jnp.zeros((PROBES,), jnp.int32)
        sel_probe = _probe(pool_select, jnp.asarray(sel_blocks, jnp.int32), zeros)
        log_probe = _probe(pool_log, jnp.asarray(log_blocks, jnp.int32), jnp.asarray(log_offs, jnp.int32))
        rec_eq = np.asarray(_bits_equal(sel_probe[0], log_probe[0]))
        conv_eq = np.asarray(_bits_equal(sel_probe[1], log_probe[1]))
        counts["probes"] += len(owners)
        for idx, (s, off, drawn) in enumerate(owners):
            if diag and not rec_eq[idx]:
                xs = np.asarray(sel_probe[0][idx])
                xl = np.asarray(log_probe[0][idx])
                diff = xs.view(np.uint32) != xl.view(np.uint32)
                where = np.argwhere(diff)
                print(f"DIAG step {step}: seq {s} offset {off}: {diff.sum()} state elements differ; "
                      f"per head {diff.sum(axis=(1, 2)).tolist()}; rows with a diff "
                      f"{np.unique(where[:, 1]).size}, cols {np.unique(where[:, 2]).size}; first "
                      f"{where[:4].tolist()}; max |d| {np.abs(xs - xl)[diff].max():.3g}, "
                      f"max |x| {np.abs(xs).max():.3g}")
            if not (rec_eq[idx] and conv_eq[idx]):
                failures.append(
                    f"step {step}: sequence {s} offset {off} (drawn {drawn}): "
                    f"recurrent equal {bool(rec_eq[idx])}, conv equal {bool(conv_eq[idx])}"
                )

        # 4. Host fold on every 4th step (and on any mismatch): does the
        # kernels' update equal the unfused IEEE two-statement form?
        if step % 4 == 3 or failures:
            base = _probe(pool_log, jnp.asarray(log_blocks, jnp.int32), zeros)
            base = [np.asarray(x) for x in base]
            sel_rec, sel_conv = np.asarray(sel_probe[0]), np.asarray(sel_probe[1])
            for idx, (s, off, drawn) in enumerate(owners):
                if off == 0:
                    continue
                rec, conv = _numpy_fold(*(x[idx] for x in base), off)
                counts["numpy_fold_checked"] += 1
                if np.array_equal(rec.view(np.uint32), sel_rec[idx].view(np.uint32)) and (
                    np.array_equal(conv.view(np.uint32), sel_conv[idx].view(np.uint32))
                ):
                    counts["numpy_fold_equal"] += 1
                elif counts["numpy_fold_checked"] - counts["numpy_fold_equal"] <= 3:
                    diff = np.nonzero(rec.view(np.uint32) != sel_rec[idx].view(np.uint32))
                    print(
                        f"NUMPY_FOLD_DIFF step {step} seq {s} offset {off}: "
                        f"{diff[0].size} recurrent elements differ, first at {[int(d[0]) for d in diff] if diff[0].size else None}"
                    )
        if diag:
            # Restart every sequence named in this step's failures.
            for line in failures[step_failures:]:
                for word in ("sequence ", "seq "):
                    if word in line:
                        tail = line.split(word, 1)[1].split()[0].strip(":,()")
                        if tail.isdigit():
                            restart.add(int(tail))
            if failures[step_failures:]:
                print(f"DIAG step {step}: {len(failures) - step_failures} failures, restarting {sorted(restart)}")
        elif failures:
            break

    print("GATE1_COUNTS", " ".join(f"{k}={v}" for k, v in counts.items()))
    print("GATE1_STEPS", step + 1, "FAILURES", len(failures))
    for line in failures[:20]:
        print("GATE1_FAIL", line)
    assert not failures, failures[:5]


# Probe P1 (lead 10.168): prefill-sized PER_SEQ chunks at the decode bench's
# s128 shapes (2048 tokens a step, 128 sequence slots), where gate 1 drew
# decode-sized chunks only.
P1_SLOTS = 128
P1_TOKENS = 2048
P1_DECODERS = int(os.environ.get("F7_P1_DECODERS", "32"))
P1_PREFILLS = int(os.environ.get("F7_P1_PREFILLS", "3"))
P1_CHUNKS = int(os.environ.get("F7_P1_CHUNKS", "5"))
P1_TAIL_STEPS = int(os.environ.get("F7_P1_TAIL_STEPS", "6"))


def test_log_matches_select_prefill_chunks_tpu(monkeypatch):
    """PER_SEQ LOG against PER_SEQ SELECT on prefill-sized chunks, as the
    decode bench's fill runs them: each step holds the decoding sequences'
    BATCHED windows and one prompt's PER_SEQ chunk filling the rest of the
    2048 tokens. A prompt spans several chunks, its state carried through the
    pool across a block crossing before each chunk (SELECT copies checkpoint
    [offset] into the new checkpoint 0, LOG copies block 0 whole); after its
    last chunk it decodes in BATCHED windows. Checked bit for bit on every
    step: the outputs, block 0's state and conv rows of every sequence that
    ran, the state every sequence loads next at each offset its last window
    names, and that no kernel writes a row it does not own."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    monkeypatch.setenv("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")
    rng = np.random.default_rng(int(os.environ.get("F7_P1_SEED", "20261113")))
    num_seqs = P1_DECODERS + P1_PREFILLS
    num_blocks = 1 + W * (num_seqs + 4)
    raw = rng.integers(0, 256, size=num_blocks * ROWS * 1024, dtype=np.uint8)
    byte_axis = (POOL_DTYPE.itemsize,) if POOL_DTYPE.itemsize > 1 else ()
    raw = raw.reshape(num_blocks, ROWS, *PAYLOAD, LANES, *byte_axis)
    pool_select = _f8(jnp.asarray(raw))
    pool_log = _pool_copy(pool_select)
    del raw
    conv_w = jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32)
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32)
    dt_bias = jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16)
    free = list(range(1, num_blocks))
    rng.shuffle(free)

    def new_group():
        return [free.pop() for _ in range(W)]

    seqs = [_Seq(group=new_group()) for _ in range(num_seqs)]
    # Sequences 0..P1_DECODERS-1 decode from step 0; the rest are prompts,
    # prefilled one at a time in order.
    prompt_len = {
        s: P1_CHUNKS * P1_TOKENS + int(rng.integers(1, P1_TOKENS))
        for s in range(P1_DECODERS, num_seqs)
    }
    decoding = list(range(P1_DECODERS))
    queue = list(range(P1_DECODERS, num_seqs))
    started, handed_over = set(), set()
    counts = collections.Counter()
    chunk_lens = []
    failures = []
    step = 0
    tail = 0
    while queue or tail < P1_TAIL_STEPS:
        if not queue:
            tail += 1
        # Block crossings before the step: the prompt's next chunk starts in a
        # new block group; a decoding sequence crosses now and then.
        crossing = [queue[0]] if queue and queue[0] in started else []
        crossing += [s for s in decoding if seqs[s].seq_len > 0 and rng.random() < 0.1]
        for s in crossing:
            seq = seqs[s]
            old = seq.group
            seq.group = new_group()
            pool_select = _copy_rows(
                pool_select, seq.group[0], old[seq.ro_select], r0=0, r1=STATE_ROWS
            )
            pool_log = _copy_rows(pool_log, seq.group[0], old[0], r0=0, r1=ROWS)
            free.extend(old)
            seq.ro_select = seq.last_r = 0
            seq.seeded = True
            counts["crossings_prefill" if s in queue else "crossings_decode"] += 1

        batched = []
        for s in decoding:
            u = rng.random()
            if u < 0.7 or seqs[s].seq_len == 0:
                batched.append((s, W, "verify"))
            elif u < 0.85:
                batched.append((s, int(rng.integers(2, W)), "verify"))
            else:
                batched.append((s, 1, "one_token"))
        per_seq = []
        if queue:
            s = queue[0]
            room = P1_TOKENS - sum(n for _, n, _ in batched)
            n = min(prompt_len[s] - seqs[s].seq_len, room)
            per_seq.append((s, n, "per_seq"))
            chunk_lens.append(n)
            started.add(s)
        order = batched + per_seq
        assert len(order) <= P1_SLOTS

        qlens = np.zeros(P1_SLOTS, np.int32)
        si = np.zeros(P1_SLOTS, np.int32)
        ck = np.zeros((P1_SLOTS, W), np.int32)
        sl = np.zeros(P1_SLOTS, np.int32)
        ro_sel = np.zeros(P1_SLOTS, np.int32)
        ro_log = np.zeros(P1_SLOTS, np.int32)
        for slot, (s, n, kind) in enumerate(order):
            seq = seqs[s]
            qlens[slot], si[slot], ck[slot] = n, seq.group[0], seq.group
            sl[slot] = seq.seq_len + n
            ro_sel[slot], ro_log[slot] = seq.ro_select, seq.ro_log
            if kind == "per_seq":
                counts["chunks_with_state" if seq.seq_len else "chunks_fresh"] += 1
            elif seq.ro_log > 0:
                counts["fold_loads"] += 1
            if kind != "per_seq" and s >= P1_DECODERS and s not in handed_over:
                handed_over.add(s)
                counts["handovers_to_batched"] += 1
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        assert qsl[-1] <= P1_TOKENS
        dist = np.array([len(batched), len(order), len(order)], np.int32)
        qkv = jnp.asarray(rng.normal(0, 1, (P1_TOKENS, 1, DIM)), jnp.float32)
        b = jnp.asarray(rng.normal(0, 1, (P1_TOKENS, N_V)), jnp.float32)
        a = jnp.asarray(rng.normal(0, 1, (P1_TOKENS, N_V)), jnp.float32)
        common = (jnp.asarray(qsl), jnp.asarray(si), jnp.asarray(dist), jnp.asarray(sl))

        before_select, before_log = _pool_copy(pool_select), _pool_copy(pool_log)
        pool_select, out_select = _run(
            qkv, b, a, pool_select, conv_w, a_log, dt_bias, *common,
            jnp.asarray(ro_sel), jnp.asarray(ck), plan=SELECT_PLAN,
        )
        pool_log, out_log = _run(
            qkv, b, a, pool_log, conv_w, a_log, dt_bias, *common,
            jnp.asarray(ro_log), jnp.asarray(ck), plan=LOG_PLAN,
        )

        # 1. Outputs, every row.
        out_eq = np.asarray(
            jax.lax.bitcast_convert_type(out_select, jnp.uint16)
            == jax.lax.bitcast_convert_type(out_log, jnp.uint16)
        ).reshape(P1_TOKENS, -1).all(axis=1)
        bad = np.nonzero(~out_eq)[0]
        if bad.size:
            slots = np.searchsorted(qsl, bad, side="right") - 1
            kinds = sorted({order[i][2] if i < len(order) else "padding" for i in slots})
            failures.append(f"step {step}: outputs differ in {bad.size} rows ({', '.join(kinds)}), first row {int(bad[0])}")
        counts["output_rows_checked"] += int(qsl[len(order)])

        # Offsets after the step.
        for s, n, kind in order:
            seq = seqs[s]
            seq.seq_len += n
            if kind == "verify":
                seq.ro_select = seq.ro_log = int(rng.integers(1, n + 1)) - 1
                seq.last_r = n
            else:
                seq.ro_select = seq.ro_log = 0
                seq.last_r = 1 if kind == "one_token" else 0
            seq.seeded = False
        if queue and seqs[queue[0]].seq_len == prompt_len[queue[0]]:
            decoding.append(queue.pop(0))
            counts["prompts_done"] += 1

        # 2. No row outside what a scheduled sequence owns changes.
        allowed_select = np.zeros((num_blocks, ROWS), bool)
        allowed_log = np.zeros((num_blocks, ROWS), bool)
        for s_, n_, kind_ in order:
            group = seqs[s_].group
            for j in range(1 if kind_ == "per_seq" else min(n_, W)):
                allowed_select[group[j], :STATE_ROWS] = True
            allowed_log[group[0], : (STATE_ROWS if kind_ == "per_seq" else LOG_END)] = True
        for name, before, after, allowed in (
            ("select", before_select, pool_select, allowed_select),
            ("log", before_log, pool_log, allowed_log),
        ):
            stray = np.argwhere(~np.asarray(_rows_unchanged(before, after)) & ~allowed)
            if stray.size:
                failures.append(f"step {step}: {name} kernel wrote {len(stray)} rows it does not own")
        del before_select, before_log

        # 3. Block 0's state and conv rows of every sequence that ran.
        ran = [s for s, _, _ in order]
        blocks0 = jnp.asarray([seqs[s].group[0] for s in ran], jnp.int32)
        eq0 = np.asarray(_rows_equal(pool_select, blocks0, pool_log, blocks0, rows=STATE_ROWS))
        for idx in np.nonzero(~eq0)[0]:
            failures.append(f"step {step}: block 0 rows differ for sequence {ran[idx]} ({order[idx][2]})")

        # 4. The state each sequence loads next, at every offset its last
        # window names.
        sel_blocks, log_blocks, log_offs, owners = [], [], [], []
        for s, seq in enumerate(seqs):
            if seq.seq_len == 0:
                continue
            for j in range(max(seq.last_r, seq.ro_log + 1)):
                sel_blocks.append(seq.group[j])
                log_blocks.append(seq.group[0])
                log_offs.append(j)
                owners.append((s, j))
        zeros = jnp.zeros((len(owners),), jnp.int32)
        sel_probe = _probe(pool_select, jnp.asarray(sel_blocks, jnp.int32), zeros)
        log_probe = _probe(pool_log, jnp.asarray(log_blocks, jnp.int32), jnp.asarray(log_offs, jnp.int32))
        rec_eq = np.asarray(_bits_equal(sel_probe[0], log_probe[0]))
        conv_eq = np.asarray(_bits_equal(sel_probe[1], log_probe[1]))
        counts["probes"] += len(owners)
        for idx, (s, off) in enumerate(owners):
            if not (rec_eq[idx] and conv_eq[idx]):
                failures.append(
                    f"step {step}: sequence {s} offset {off}: recurrent equal "
                    f"{bool(rec_eq[idx])}, conv equal {bool(conv_eq[idx])}"
                )
        step += 1
        if failures:
            break

    print("P1_COUNTS", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("P1_CHUNK_LENS", f"n={len(chunk_lens)} min={min(chunk_lens)} max={max(chunk_lens)}")
    print("P1_STEPS", step, "FAILURES", len(failures))
    for line in failures[:20]:
        print("P1_FAIL", line)
    assert counts.get("prompts_done", 0) == P1_PREFILLS or failures
    assert not failures, failures[:5]
