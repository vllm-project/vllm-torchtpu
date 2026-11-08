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
"""GDN v3 batched tiles at a served unified-pool geometry.

One TP4 core of Qwen3.8 (4 kq / 32 v heads, head dim 128) on the fp8
SEQ_ALONG_LANE pool with a bf16 recurrent state: one checkpoint is 16 pool
pages (1 MiB), so a 4-token window needs 4.3 MiB of VMEM per sequence and
several sequences fit in one tile. The served call (multi-sequence tiles,
separate state read and checkpoint buffers) must reproduce the
one-sequence, in-out-buffer kernel bit for bit, including partial windows,
padded and fresh slots, prefill chunks after the windowed segment, and
rollback across steps.
"""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, vmem_ldst, wrapper
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V, D, KERNEL_SIZE = 4, 32, 128, 4
DIM = N_KQ * D * 2 + N_V * D
PAGE = (2, 64, 4, 128)  # fp8 SEQ_ALONG_LANE pool page (1 KV head, head dim 256)
PAGE_BYTES = int(np.prod(PAGE))
SPLIT = 17  # 2176-token manager blocks of 128-token pages
SEQS = 12


@pytest.fixture(autouse=True)
def _seq_along_lane_pool(monkeypatch):
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")


def _pool(seed: int, num_blocks: int) -> jax.Array:
    """fp8 pool whose bytes read back as small finite bf16 values."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 0x3B, size=(num_blocks * SPLIT,) + PAGE, dtype=np.uint8)
    raw |= (rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7).astype(np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)


def _weights(seed: int) -> dict:
    keys = iter(jax.random.split(jax.random.key(seed), 3))
    return dict(
        conv_weight=(0.5 * jax.random.normal(next(keys), (DIM, 1, KERNEL_SIZE))).astype(
            jnp.bfloat16
        ),
        conv_bias=None,
        a_log=jax.random.normal(next(keys), (N_V,)),
        dt_bias=jax.random.normal(next(keys), (N_V,)),
    )


def _step(
    seed: int, num_spec: int, lens, dist, fresh=(), offsets=None, num_tokens=None
) -> dict:
    """One call's activations and metadata; `lens` are tokens per sequence."""
    window = num_spec + 1
    seqs = len(lens)
    lens = np.asarray(lens)
    live = np.arange(seqs) < dist[-1]
    qsl = np.zeros(seqs + 1, np.int32)
    qsl[1:] = np.cumsum(np.where(live, lens, 0))
    num_tokens = int(qsl[-1]) if num_tokens is None else num_tokens
    history = np.where(np.isin(np.arange(seqs), fresh), 0, 1000)
    keys = iter(jax.random.split(jax.random.key(seed), 3))
    step = dict(
        qkv=jax.random.normal(next(keys), (num_tokens, DIM)).astype(jnp.bfloat16),
        b=jax.random.normal(next(keys), (num_tokens, N_V)).astype(jnp.bfloat16),
        a=jax.random.normal(next(keys), (num_tokens, N_V)).astype(jnp.bfloat16),
        query_start_loc=jnp.asarray(qsl),
        distribution=jnp.asarray(dist, jnp.int32),
        seq_lens=jnp.asarray(np.where(live, history + lens, 0), jnp.int32),
        state_indices=jnp.asarray(1 + np.arange(seqs) * window, jnp.int32),
        num_spec_tokens=num_spec,
    )
    if num_spec:
        # Checkpoint blocks of a sequence deliberately not in window order.
        ckpts = 1 + np.arange(seqs)[:, None] * window + np.arange(window)[::-1]
        step["ckpt_indices"] = jnp.asarray(ckpts, jnp.int32)
        step["read_offsets"] = jnp.asarray(
            np.zeros(seqs) if offsets is None else offsets, jnp.int32
        )
    return step


def _pooled_call(pool, step, weights, decode_tile_size=None):
    plan = gdn_attention._build_v3_pool_state_plan(
        pool,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        pool_block_tokens=SPLIT * 128,
        qk_pair_layout=True,
        recurrent_state_dtype=jnp.bfloat16,
    )
    tile = {} if decode_tile_size is None else dict(decode_tile_size=decode_tile_size)
    return wrapper.fused_conv1d_gdn(
        conv_state=None,
        recurrent_state=None,
        state_source=pool,
        state_plan=plan,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        **step,
        **weights,
        **tile,
    )


def _dense_call(states, step, weights, decode_tile_size=None):
    step = dict(step)
    if "ckpt_indices" in step:
        # Dense spec decoding addresses checkpoints as base slot + position.
        del step["ckpt_indices"]
    tile = {} if decode_tile_size is None else dict(decode_tile_size=decode_tile_size)
    states, out = wrapper.fused_conv1d_gdn(
        conv_state=states[0],
        recurrent_state=states[1],
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        **step,
        **weights,
        **tile,
    )
    return states, out


def _reference(monkeypatch, call, *args):
    """The kernel before the change: one sequence per tile, one in-out
    buffer per state."""
    with monkeypatch.context() as m:
        m.setattr(config.GDNConfig, "SPLIT_BATCHED_STATE_IO", False)
        jax.clear_caches()
        result = call(*args, 1)
    jax.clear_caches()
    return result


def _assert_same(got, want, what):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        g, w = np.asarray(g), np.asarray(w)
        np.testing.assert_array_equal(g.view(np.uint8), w.view(np.uint8), err_msg=what)


def _tile(recurrent_ckpt_bytes, conv_ckpt_bytes, window=4):
    return wrapper.windowed_decode_tile_size(
        4,
        window=window,
        recurrent_ckpt_bytes=recurrent_ckpt_bytes,
        conv_ckpt_bytes=conv_ckpt_bytes,
        token_bytes=DIM * 4 + 2 * 128 * 4 + N_V * D * 2,
    )


def test_windowed_tile_follows_tile_bytes(monkeypatch):
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: types.SimpleNamespace(vmem_capacity_bytes=64 << 20),
    )
    conv_f32 = (KERNEL_SIZE - 1) * DIM * 4
    # Pooled bf16 state: 16 pages per checkpoint plus one conv page, four
    # checkpoints and one read state per sequence.
    assert _tile(16 * PAGE_BYTES, PAGE_BYTES) == 2
    # Dense states: the tile holds the recurrent state in its own dtype.
    assert _tile(N_V * D * D * 2, conv_f32) == 2
    assert _tile(N_V * D * D * 4, conv_f32) == 1
    monkeypatch.setattr(config.GDNConfig, "SPLIT_BATCHED_STATE_IO", False)
    assert _tile(16 * PAGE_BYTES, PAGE_BYTES) == 3


W = 4  # 3-token drafts
SCHEDULES = {
    # Full windows, then mixed acceptance rolls each sequence back to a
    # different checkpoint, then full windows again.
    "verify_full": (
        3,
        [
            dict(lens=[W] * SEQS, dist=(SEQS,) * 3),
            dict(lens=[W] * SEQS, dist=(SEQS,) * 3, offsets=np.arange(SEQS) % W),
            dict(lens=[W] * SEQS, dist=(SEQS,) * 3, offsets=np.arange(SEQS) * 3 % W),
        ],
    ),
    # 1-4 token windows, a fresh slot, and padded slots past the live ones.
    "verify_partial": (
        3,
        [
            dict(
                lens=[4, 1, 3, 2, 4, 4, 1, 2, 3, 4, 4, 4],
                dist=(9, 9, 9),
                fresh=(4,),
                num_tokens=48,
            ),
            dict(
                lens=[2, 4, 1, 4, 3, 1, 4, 2, 4, 4, 4, 4],
                dist=(10, 10, 10),
                offsets=[3, 0, 2, 1, 2, 0, 0, 1, 3, 0, 0, 0],
                num_tokens=48,
            ),
        ],
    ),
    # Verify windows followed by prefill chunks in the PER_SEQ segment.
    "verify_prefill": (
        3,
        [
            dict(
                lens=[4, 2, 4, 1, 3, 4, 37, 70],
                dist=(6, 8, 8),
                fresh=(7,),
                num_tokens=128,
            ),
            dict(
                lens=[4, 4, 1, 4, 2, 4, 4, 12],
                dist=(7, 8, 8),
                offsets=[1, 3, 0, 0, 2, 0, 0, 0],
                num_tokens=128,
            ),
        ],
    ),
    # One-token decode: full batch, then padded slots and a fresh one.
    "decode": (
        0,
        [
            dict(lens=[1] * SEQS, dist=(SEQS,) * 3),
            dict(lens=[1] * SEQS, dist=(10, 10, 10), fresh=(3,), num_tokens=SEQS),
            dict(lens=[1] * 6 + [9, 33], dist=(6, 8, 8), num_tokens=48),
        ],
    ),
}


@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
def test_pooled_batched_tiles_match_reference(schedule, monkeypatch):
    # The served verify call (no explicit tile) groups several sequences.
    assert _tile(16 * PAGE_BYTES, PAGE_BYTES) > 1
    num_spec, steps = SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(1)
    pool = _pool(2, 1 + seqs * (num_spec + 1) + 1)
    ref_pool = jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = _step(10 + i, num_spec, **spec)
        pool, out = _pooled_call(pool, step, weights)
        ref_pool, ref_out = _reference(
            monkeypatch, _pooled_call, ref_pool, step, weights
        )
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")


@pytest.mark.parametrize("state_dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("schedule", ["verify_partial", "decode"])
def test_dense_batched_tiles_match_reference(schedule, state_dtype, monkeypatch):
    num_spec, steps = SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    slots = 1 + seqs * (num_spec + 1)
    keys = iter(jax.random.split(jax.random.key(3), 2))
    states = (
        (0.5 * jax.random.normal(next(keys), (slots, KERNEL_SIZE - 1, DIM))).astype(
            jnp.bfloat16
        ),
        (0.1 * jax.random.normal(next(keys), (slots, N_V, D, D))).astype(state_dtype),
    )
    ref_states = jax.tree.map(lambda x: jnp.array(x, copy=True), states)
    weights = _weights(4)
    for i, spec in enumerate(steps):
        step = _step(20 + i, num_spec, **spec)
        states, out = _dense_call(states, step, weights)
        ref_states, ref_out = _reference(
            monkeypatch, _dense_call, ref_states, step, weights
        )
        _assert_same((out, states), (ref_out, ref_states), f"{schedule} step {i}")


def _inner_kernel_reference(monkeypatch, call, *args):
    """The served tile computed by wrapper.inner_kernel instead of
    window_tile."""
    with monkeypatch.context() as m:
        m.setattr(config.GDNConfig, "PER_HEAD_WINDOWS", False)
        jax.clear_caches()
        result = call(*args)
    jax.clear_caches()
    return result


WINDOW_SCHEDULES = dict(
    SCHEDULES,
    # The 4-host operating point: 24 sequences per engine, full windows,
    # then every rollback offset.
    verify_served=(
        3,
        [
            dict(lens=[W] * 24, dist=(24,) * 3),
            dict(lens=[W] * 24, dist=(24,) * 3, offsets=np.arange(24) % W),
        ],
    ),
    # An odd live count leaves a padded slot in the last tile; a lone live
    # sequence pads the only tile. Like the served request arrays, the
    # per-sequence arrays cover whole tiles.
    verify_odd=(
        3,
        [dict(lens=[W] * 24, dist=(23,) * 3, offsets=np.arange(24) % W, num_tokens=96)],
    ),
    verify_one=(
        3,
        [dict(lens=[3, 1, 1, 1], dist=(1, 1, 1), offsets=[2, 0, 0, 0], num_tokens=4)],
    ),
)


@pytest.mark.parametrize("schedule", sorted(WINDOW_SCHEDULES))
def test_window_tile_matches_inner_kernel(schedule, monkeypatch):
    num_spec, steps = WINDOW_SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(5)
    pool = _pool(6, 1 + seqs * (num_spec + 1) + 1)
    ref_pool = jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = _step(30 + i, num_spec, **spec)
        pool, out = _pooled_call(pool, step, weights)
        ref_pool, ref_out = _inner_kernel_reference(
            monkeypatch, _pooled_call, ref_pool, step, weights
        )
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")


def _block_bytes(pool, block, pages=SPLIT, half_last=False):
    """Raw bytes of manager block `block` (its first `pages` pages; with
    `half_last`, only the first half of the last one: the conv state)."""
    raw = np.asarray(pool).view(np.uint8)[block * SPLIT : block * SPLIT + pages]
    raw = raw.reshape(pages, -1)
    if half_last:
        return np.concatenate([raw[:-1].reshape(-1), raw[-1, : raw.shape[1] // 2]])
    return raw.reshape(-1)


TWO_SLOT_SCHEDULES = {
    # The 4-host operating point: full windows, every acceptance count.
    "served": dict(seqs=24, lens=lambda t, s: W, live=24),
    # 1-4 token windows (one-token windows commit), padded tail slots.
    "mixed": dict(seqs=12, lens=lambda t, s: 1 + (3 * s + t) % W, live=10),
    # Verify windows followed by prefill chunks of resuming sequences.
    "prefill": dict(seqs=8, lens=lambda t, s: (37, 70)[s - 6] if s >= 6 else W, live=8),
}


# Options of the two-slot calls; "inner_ref" computes the one-block-per-
# position reference with wrapper.inner_kernel instead of window_tile.
TWO_SLOT_KNOBS = {
    "plain": {},
    "inner_ref": {},
    "packed": dict(packed_tokens=True),
    "single": dict(single_launch=True),
    "packed_single": dict(packed_tokens=True, single_launch=True),
}


@pytest.mark.parametrize("boundaries", [False, True])
@pytest.mark.parametrize("knobs", sorted(TWO_SLOT_KNOBS))
@pytest.mark.parametrize("schedule", sorted(TWO_SLOT_SCHEDULES))
def test_two_slot_checkpoints_match_window_checkpoints(
    schedule, knobs, boundaries, monkeypatch
):
    """Two state blocks per sequence plus recomputation of accepted tokens
    give the same outputs, and the same committed state bytes, as one block
    per window position, over steps covering every acceptance count; with
    `knobs` on the two-slot calls only. With `boundaries`, windows also
    complete mamba blocks at every window position: a block completed
    strictly inside the window receives the bytes of the one-block-per-
    position checkpoint at that position, and a window whose last token
    completes the pair's block 0 leaves its last state there."""
    spec = TWO_SLOT_SCHEDULES[schedule]
    seqs, live = spec["seqs"], spec["live"]
    windowed = 6 if schedule == "prefill" else live
    ckpt4 = 1 + np.arange(seqs)[:, None] * W + np.arange(W)[None]
    # Per sequence: the pair (deliberately not in order), then a block a
    # boundary inside the window completes.
    blocks2 = 1 + np.arange(seqs)[:, None] * 3 + np.array([1, 0, 2])[None]
    pool4 = _pool(7, 1 + seqs * W + 1)
    raw2 = np.asarray(_pool(8, 1 + seqs * 3 + 1)).view(np.uint8).copy()
    raw4 = np.asarray(pool4).view(np.uint8)
    for s in range(seqs):
        dst, src = blocks2[s, 0] * SPLIT, ckpt4[s, 0] * SPLIT
        raw2[dst : dst + SPLIT] = raw4[src : src + SPLIT]
    pool2 = jax.lax.bitcast_convert_type(jnp.asarray(raw2), jnp.float8_e4m3fn)
    weights = _weights(9)
    offsets = np.zeros(seqs, np.int32)
    codes = np.zeros(seqs, np.int32)
    pending = {}  # sequence -> committed 4-slot block bytes awaiting a replay
    for t in range(6):
        lens = np.array([spec["lens"](t, s) for s in range(seqs)])
        # Window position whose state a mamba block boundary needs: none,
        # every position strictly inside the window, then the last one.
        bpos = np.full(seqs, -1)
        if boundaries:
            for s in range(windowed):
                bpos[s] = (s + 2 * t) % (lens[s] + 1) - 1
        inside = (bpos >= 0) & (bpos < lens - 1)
        ckpt2 = np.stack(
            [blocks2[:, 0], blocks2[:, 1], np.where(inside, blocks2[:, 2], 0), bpos],
            axis=1,
        )
        dist = (windowed, live, live)
        step = _step(40 + t, 3, lens, dist, num_tokens=int(lens.sum()))
        step4 = dict(step, ckpt_indices=jnp.asarray(ckpt4, jnp.int32))
        step4["read_offsets"] = jnp.asarray(offsets)
        step2 = dict(step, ckpt_indices=jnp.asarray(ckpt2, jnp.int32))
        step2["read_offsets"] = jnp.asarray(codes)
        if knobs == "inner_ref":
            pool4, out4 = _inner_kernel_reference(
                monkeypatch, _pooled_call, pool4, step4, weights
            )
        else:
            pool4, out4 = _pooled_call(pool4, step4, weights)
        step2 = dict(step2, two_slot_ckpts=True, **TWO_SLOT_KNOBS[knobs])
        pool2, out2 = _pooled_call(pool2, step2, weights)
        _assert_same(out2, out4, f"{schedule} step {t} outputs")
        drafts = np.where(np.arange(seqs) < windowed, lens - 1, 0)
        # Each window's start block: block 1 when its last token completes
        # block 0, else the one its code names.
        ends = (drafts > 0) & (bpos == drafts)
        start = np.where(ends, 1, codes & 1)
        for s, want in pending.items():
            if lens[s] == 1:
                # A one-token window commits to block 0 (checked below).
                continue
            # The start state a replay rebuilt is the committed checkpoint.
            got = _block_bytes(pool2, blocks2[s, start[s]], half_last=True)
            np.testing.assert_array_equal(got, want, err_msg=f"{schedule} {t} s{s}")
        pending = {}
        for s in range(live):
            accepted = (s + t) % (drafts[s] + 1)
            if drafts[s] and inside[s]:
                # The snapshot, whatever the acceptance.
                np.testing.assert_array_equal(
                    _block_bytes(pool2, blocks2[s, 2]),
                    _block_bytes(pool4, ckpt4[s, bpos[s]]),
                    err_msg=f"{schedule} step {t} s{s} boundary state",
                )
            if ends[s]:
                np.testing.assert_array_equal(
                    _block_bytes(pool2, blocks2[s, 0]),
                    _block_bytes(pool4, ckpt4[s, drafts[s]]),
                    err_msg=f"{schedule} step {t} s{s} boundary last state",
                )
            if drafts[s] == 0:
                committed, nxt = ckpt4[s, 0], 0
                last = blocks2[s, 0]
            elif accepted == drafts[s]:
                committed, nxt = ckpt4[s, accepted], 1 - start[s]
                last = blocks2[s, nxt]
            else:
                committed, nxt = ckpt4[s, accepted], start[s] + 2 * (accepted + 1)
                last = None
            if last is None:
                pending[s] = _block_bytes(pool4, committed, half_last=True)
            else:
                np.testing.assert_array_equal(
                    _block_bytes(pool2, last),
                    _block_bytes(pool4, committed),
                    err_msg=f"{schedule} step {t} s{s} last state",
                )
            offsets[s] = accepted if drafts[s] else 0
            codes[s] = nxt


@pytest.mark.parametrize("schedule", sorted(WINDOW_SCHEDULES))
def test_packed_tokens_match_per_token_tiles(schedule):
    # Packed activations change only the operand layout: verify windows,
    # prefill chunks after them and one-token decodes keep every bit.
    num_spec, steps = WINDOW_SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(7)
    pool = _pool(8, 1 + seqs * (num_spec + 1) + 1)
    ref_pool = jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = _step(40 + i, num_spec, **spec)
        pool, out = _pooled_call(pool, dict(step, packed_tokens=True), weights)
        ref_pool, ref_out = _pooled_call(ref_pool, step, weights)
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("schedule", sorted(WINDOW_SCHEDULES))
def test_single_launch_matches_two_launches(schedule, packed):
    # One kernel running both segments keeps every bit of the two calls.
    num_spec, steps = WINDOW_SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(9)
    pool = _pool(10, 1 + seqs * (num_spec + 1) + 1)
    ref_pool = jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = dict(_step(50 + i, num_spec, **spec), packed_tokens=packed)
        pool, out = _pooled_call(pool, dict(step, single_launch=True), weights)
        ref_pool, ref_out = _pooled_call(ref_pool, step, weights)
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")


# Paired value heads (the path gdn_paired_heads): the switch on against off, from
# the same inputs, must keep every output and pool bit.

PAIRED_KNOBS = {
    "plain": {},
    "packed_single": dict(packed_tokens=True, single_launch=True),
}


@pytest.mark.parametrize("knobs", sorted(PAIRED_KNOBS))
@pytest.mark.parametrize("schedule", sorted(WINDOW_SCHEDULES))
def test_paired_heads_match_single_heads(schedule, knobs):
    """One block per window position: 1-4 token windows, fresh and padded
    slots, prefill chunks after the windows, one-token decodes."""
    num_spec, steps = WINDOW_SCHEDULES[schedule]
    seqs = len(steps[0]["lens"])
    weights = _weights(11)
    pool = _pool(12, 1 + seqs * (num_spec + 1) + 1)
    ref_pool = jnp.array(pool, copy=True)
    for i, spec in enumerate(steps):
        step = dict(_step(60 + i, num_spec, **spec), **PAIRED_KNOBS[knobs])
        pool, out = _pooled_call(pool, dict(step, paired_heads=True), weights)
        ref_pool, ref_out = _pooled_call(ref_pool, step, weights)
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {i}")


PAIRED_TWO_SLOT = {
    "served": dict(TWO_SLOT_SCHEDULES["served"], fresh=lambda t: ()),
    # Slots 3 and 7 start fresh at steps 0 and 3: a new request in a reused
    # slot, whose blocks still hold the previous request's states.
    "mixed": dict(
        TWO_SLOT_SCHEDULES["mixed"], fresh=lambda t: (3, 7) if t in (0, 3) else ()
    ),
    "prefill": dict(
        TWO_SLOT_SCHEDULES["prefill"], fresh=lambda t: (2,) if t == 1 else ()
    ),
}


def _two_slot_paired_steps(pool, schedule, knobs, boundaries, weights, seed, edit=None):
    """Six two-slot steps of `schedule` from `pool`, with paired heads on one
    copy and off on the other; every step's outputs and pool bytes must
    match. Windows complete mamba blocks at every position when
    `boundaries`; the acceptance pattern covers every count, so later steps
    recompute pending tokens. `edit(step)` may change a step's inputs."""
    spec = PAIRED_TWO_SLOT[schedule]
    seqs, live = spec["seqs"], spec["live"]
    windowed = 6 if schedule == "prefill" else live
    blocks2 = 1 + np.arange(seqs)[:, None] * 3 + np.array([1, 0, 2])[None]
    ref_pool = jnp.array(pool, copy=True)
    codes = np.zeros(seqs, np.int32)
    for t in range(6):
        lens = np.array([spec["lens"](t, s) for s in range(seqs)])
        fresh = spec["fresh"](t)
        codes[list(fresh)] = 0
        bpos = np.full(seqs, -1)
        if boundaries:
            for s in range(windowed):
                bpos[s] = (s + 2 * t) % (lens[s] + 1) - 1
        inside = (bpos >= 0) & (bpos < lens - 1)
        ckpt2 = np.stack(
            [blocks2[:, 0], blocks2[:, 1], np.where(inside, blocks2[:, 2], 0), bpos],
            axis=1,
        )
        step = _step(
            seed + t, 3, lens, (windowed, live, live), fresh, num_tokens=int(lens.sum())
        )
        step.update(
            ckpt_indices=jnp.asarray(ckpt2, jnp.int32),
            read_offsets=jnp.asarray(codes),
            two_slot_ckpts=True,
            **PAIRED_KNOBS[knobs],
        )
        if edit is not None:
            step = edit(step)
        pool, out = _pooled_call(pool, dict(step, paired_heads=True), weights)
        ref_pool, ref_out = _pooled_call(ref_pool, step, weights)
        _assert_same((out, pool), (ref_out, ref_pool), f"{schedule} step {t}")
        # The next read codes, as the scheduler sets them.
        drafts = np.where(np.arange(seqs) < windowed, lens - 1, 0)
        ends = (drafts > 0) & (bpos == drafts)
        start = np.where(ends, 1, codes & 1)
        for s in range(live):
            accepted = (s + t) % (drafts[s] + 1)
            if drafts[s] == 0:
                codes[s] = 0
            elif accepted == drafts[s]:
                codes[s] = 1 - start[s]
            else:
                codes[s] = start[s] + 2 * (accepted + 1)


@pytest.mark.parametrize("boundaries", [False, True])
@pytest.mark.parametrize("knobs", sorted(PAIRED_KNOBS))
@pytest.mark.parametrize("schedule", sorted(PAIRED_TWO_SLOT))
def test_paired_heads_two_slot_match_single_heads(schedule, knobs, boundaries):
    """Two-slot checkpoints: replays of every pending count, 1-4 token
    windows (one-token windows commit to block 0), fresh slots, boundary
    snapshots inside the window and windows ending on block 0, padded
    slots, prefill chunks after the windows."""
    seqs = PAIRED_TWO_SLOT[schedule]["seqs"]
    pool = _pool(13, 1 + seqs * 3 + 1)
    _two_slot_paired_steps(pool, schedule, knobs, boundaries, _weights(14), 70)


def _state_writer_cfg(pool) -> config.GDNConfig:
    plan = gdn_attention._build_v3_pool_state_plan(
        pool,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        pool_block_tokens=SPLIT * 128,
        qk_pair_layout=True,
        recurrent_state_dtype=jnp.bfloat16,
    )
    f32 = jnp.dtype(jnp.float32)
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=f32,
            act_out=jnp.dtype(jnp.bfloat16),
            compute=f32,
            recurrent_state=f32,
            conv_state=f32,
        ),
        batch_size=1,
        dim_size=DIM,
        kernel_size=KERNEL_SIZE,
        tile_size=1,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D,
        v_head_dim=D,
        state_plan=plan,
    )


def _state_pages(pool, blocks) -> np.ndarray:
    """Pool pages of the recurrent region of manager blocks `blocks`."""
    plan = _state_writer_cfg(pool).state_plan
    first = np.asarray(blocks)[:, None] * plan.stride + plan.recurrent.kb0
    return (first + np.arange(plan.recurrent.nblocks)[None]).reshape(-1)


def _encode_states(pool, states: jax.Array) -> jax.Array:
    """[n, N_V, D, D] BF16 head states -> the recurrent pool pages of n
    manager blocks, with the bytes vmem_ldst.store_state_head stores."""
    cfg = _state_writer_cfg(pool)
    region = cfg.state_plan.recurrent

    def kernel(state_ref, page_ref):
        for head in range(N_V):
            view = vmem_ldst._head_view(page_ref, region, cfg, head)
            bits = pltpu.bitcast(state_ref[0, head], jnp.uint32)
            view[...] = bits.reshape(view.shape)

    n = states.shape[0]
    return pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((1, N_V, D, D), lambda i: (i, 0, 0, 0))],
        out_specs=pl.BlockSpec((region.nblocks,) + PAGE, lambda i: (i, 0, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n * region.nblocks,) + PAGE, pool.dtype),
    )(states)


def _decode_states(pool, pages: jax.Array) -> jax.Array:
    """Inverse of `_encode_states`: the BF16 values
    vmem_ldst.load_state_head reads (before its FP32 cast)."""
    cfg = _state_writer_cfg(pool)
    region = cfg.state_plan.recurrent

    def kernel(page_ref, state_ref):
        for head in range(N_V):
            view = vmem_ldst._head_view(page_ref, region, cfg, head)
            state_ref[0, head] = pltpu.bitcast(view[...], jnp.bfloat16).reshape(D, D)

    n = pages.shape[0] // region.nblocks
    return pl.pallas_call(
        kernel,
        grid=(n,),
        in_specs=[pl.BlockSpec((region.nblocks,) + PAGE, lambda i: (i, 0, 0, 0, 0))],
        out_specs=pl.BlockSpec((1, N_V, D, D), lambda i: (i, 0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n, N_V, D, D), jnp.bfloat16),
    )(pages)


def _with_states(pool, blocks, bits: np.ndarray) -> jax.Array:
    """`pool` whose manager blocks `blocks` hold the BF16 head states with
    bit patterns `bits` [len(blocks), N_V, D, D] (uint16); checks that the
    pool reads them back bit for bit."""
    states = jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)
    pages = _state_pages(pool, blocks)
    # Pages move as bytes: no float conversion touches them.
    raw = jax.lax.bitcast_convert_type(pool, jnp.uint8)
    encoded = _encode_states(pool, states)
    raw = raw.at[pages].set(jax.lax.bitcast_convert_type(encoded, jnp.uint8))
    pool = jax.lax.bitcast_convert_type(raw, pool.dtype)
    back = _decode_states(pool, jax.lax.bitcast_convert_type(raw[pages], pool.dtype))
    np.testing.assert_array_equal(np.asarray(back).view(np.uint16), bits)
    return pool


def _special_bits(rng, n: int) -> np.ndarray:
    """[n, N_V, D, D] BF16 bit patterns, one kind per block in turn: random
    normal values; signed zeros (whole heads of -0.0 and +0.0, and a mix);
    subnormals (whole heads, and a mix); large finite values (whole heads of
    +-max, and a mix); random finite bit patterns of every exponent."""
    shape = (N_V, D, D)
    normal = np.asarray(
        jnp.asarray(0.1 * rng.standard_normal((n,) + shape), jnp.float32).astype(
            jnp.bfloat16
        )
    ).view(np.uint16)
    out = normal.copy()
    for b in range(n):
        sign = (rng.integers(0, 2, shape) << 15).astype(np.uint16)
        pick = rng.random(shape)
        kind = b % 5
        if kind == 1:
            out[b, :8] = 0x8000
            out[b, 8:16] = 0
            mix = np.where(pick < 0.5, 0x8000, np.where(pick < 0.75, 0, out[b]))
            out[b, 16:] = mix[16:]
        elif kind == 2:
            sub = sign | rng.integers(1, 0x80, shape).astype(np.uint16)
            out[b, :16] = sub[:16]
            out[b, 16:] = np.where(pick < 0.1, sub, out[b])[16:]
        elif kind == 3:
            big = sign | rng.integers(0x7E00, 0x7F80, shape).astype(np.uint16)
            out[b, :8] = (sign | 0x7F7F)[:8]
            out[b, 8:] = np.where(pick < 0.02, big, out[b])[8:]
        elif kind == 4:
            exponent = rng.integers(0, 255, shape).astype(np.uint16) << 7
            out[b] = sign | exponent | rng.integers(0, 0x80, shape).astype(np.uint16)
    return out


def _nonfinite_bits(rng, n: int) -> np.ndarray:
    choice = np.array([0x7FC0, 0xFFC0, 0x7F80, 0xFF80, 0x7F81, 0x7FFF], np.uint16)
    return choice[rng.integers(0, len(choice), (n, N_V, D, D))]


def _nonfinite_padding_rows(step):
    """Non-finite qkv, b and a rows past the last live token, which the
    kernel must never read: copy_in moves only each slot's live rows."""
    end = int(np.asarray(step["query_start_loc"])[-1])
    step = dict(step)
    for name, value in (("qkv", jnp.nan), ("b", jnp.inf), ("a", -jnp.inf)):
        step[name] = step[name].at[end:].set(value)
    return step


@pytest.mark.parametrize("knobs", sorted(PAIRED_KNOBS))
def test_paired_heads_special_states(knobs):
    """States holding -0.0, subnormals, large finite values and random finite
    BF16 values in the live slots, non-finite values in the blocks of padded
    and unused slots: two-slot (1-4 token windows, fresh slots, boundaries)
    and one block per window position. Activation rows past the last live
    token are non-finite too, which only checks that they are never read."""
    rng = np.random.default_rng(15)
    # Two-slot, 12 slots, 10 live: slots 10 and 11 are padded.
    seqs, live = PAIRED_TWO_SLOT["mixed"]["seqs"], PAIRED_TWO_SLOT["mixed"]["live"]
    pool = _pool(16, 1 + seqs * 3 + 1)
    used = np.arange(1, 1 + live * 3)
    idle = np.concatenate([[0], np.arange(1 + live * 3, 1 + seqs * 3 + 1)])
    pool = _with_states(pool, used, _special_bits(rng, len(used)))
    pool = _with_states(pool, idle, _nonfinite_bits(rng, len(idle)))
    _two_slot_paired_steps(
        pool, "mixed", knobs, True, _weights(17), 80, edit=_nonfinite_padding_rows
    )
    # One block per window position: verify_partial's slots 10 and 11 stay
    # padded (slot 9 is padded in the first step only).
    num_spec, steps = WINDOW_SCHEDULES["verify_partial"]
    seqs, live = len(steps[0]["lens"]), max(spec["dist"][0] for spec in steps)
    pool = _pool(18, 1 + seqs * (num_spec + 1) + 1)
    used = np.arange(1, 1 + live * (num_spec + 1))
    idle = np.concatenate(
        [[0], np.arange(1 + live * (num_spec + 1), pool.shape[0] // SPLIT)]
    )
    pool = _with_states(pool, used, _special_bits(rng, len(used)))
    pool = _with_states(pool, idle, _nonfinite_bits(rng, len(idle)))
    ref_pool = jnp.array(pool, copy=True)
    weights = _weights(19)
    for i, spec in enumerate(steps):
        step = dict(_step(90 + i, num_spec, **spec), **PAIRED_KNOBS[knobs])
        step = _nonfinite_padding_rows(step)
        pool, out = _pooled_call(pool, dict(step, paired_heads=True), weights)
        ref_pool, ref_out = _pooled_call(ref_pool, step, weights)
        _assert_same((out, pool), (ref_out, ref_pool), f"verify_partial step {i}")
