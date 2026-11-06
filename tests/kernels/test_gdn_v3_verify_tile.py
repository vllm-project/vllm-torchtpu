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
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper
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
