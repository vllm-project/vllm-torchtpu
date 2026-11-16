# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_HISTORY_ROW_KERNEL against XLA's history gathers, byte for byte, on TPU.

`pooled_short_conv_speculative` with the flag off and on, each jitted with the
pool donated as the runner does: the whole pool after the call and the conv
output must be byte-identical. Served per-rank geometry (Qwen3.8-Flash, TP8):
PLE history 9 x 10240 bf16 in an fp8 pool of 512-token blocks, MTP K=3,
max_num_seqs 128. The 1024-token layout takes the request-major checkpoint
path (more tokens than rows x window), the others the token-major one.

The crossing test runs two steps. Between them a block copy and a read-offset
reset do what the runner's seed copy does when a request's state column
advances: the checkpoint group slides by one block, the resumed checkpoint is
copied into the new checkpoint 0, and the next step reads checkpoint 0 while
writing all four, on pages the previous step wrote.

The HLO test compiles the flag-on call and checks that no XLA gather fusion
reads a [R, 9, 10240] history any more.
"""

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import pool as ple_pool

WINDOW = 4
REQ_ROWS = 128
POOL_BLOCK = (512, 1, 4, 256)
POOL_DTYPE = jnp.float8_e4m3fn
PLE_C, PLE_S, PLE_DIL = 10240, 9, 3
FLAG = "VLLM_TPU_PLE_HISTORY_ROW_KERNEL"

# name -> (token bucket, row segments (count, length, windowed, initial, state)).
# Rows with tokens lead, as the runner lays them out; the rest are idle.
LAYOUTS = {
    "decode24_t96": (96, [(24, 4, True, True, True)]),
    "decode48_t192": (192, [(48, 4, True, True, True)]),
    "decode64_t256": (256, [(64, 4, True, True, True)]),
    "decode96_t384": (384, [(96, 4, True, True, True)]),
    "decode80_t320": (320, [(80, 4, True, True, True)]),
    "decode128_t512": (512, [(128, 4, True, True, True)]),
    # request-major checkpoints: prefill rows and a few decode windows
    "prefill_t1024": (
        1024,
        [
            (8, 4, True, True, True),
            (1, 600, False, True, True),
            (1, 300, False, False, True),
            (1, 5, False, True, True),
            (1, 3, False, True, False),
        ],
    ),
    # a padded token tail, requests with 1 to 3 rows, a windowed row without
    # a state slot and one without initial state
    "short_t96": (
        96,
        [
            (12, 4, True, True, True),
            (3, 3, True, True, True),
            (4, 2, True, True, True),
            (5, 1, True, True, True),
            (1, 4, True, True, False),
            (2, 4, True, False, True),
        ],
    ),
    # decode windows, then prefill rows longer and shorter than the state
    "mixed_t512": (
        512,
        [
            (100, 4, True, True, True),
            (1, 60, False, True, True),
            (1, 40, False, False, True),
            (1, 5, False, True, True),
            (1, 7, False, False, True),
        ],
    ),
}


def _random_pool(seed, blocks):
    bits = jax.random.bits(jax.random.key(seed), (blocks,) + POOL_BLOCK, jnp.uint8)
    return jax.lax.bitcast_convert_type(bits, POOL_DTYPE)


def _pool_bytes(pool):
    return np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8))


def _rows(segments):
    out = []
    for count, length, windowed, initial, has_state in segments:
        out += [(length, windowed, initial, has_state)] * count
    assert len(out) <= REQ_ROWS
    return out + [(0, False, False, False)] * (REQ_ROWS - len(out))


def _inputs(name, seed):
    tokens, segments = LAYOUTS[name]
    rows = _rows(segments)
    rng = np.random.default_rng(seed)
    lengths = np.array([r[0] for r in rows], np.int32)
    assert lengths.sum() <= tokens
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    windowed = np.array([r[1] for r in rows])
    initial = np.array([r[2] for r in rows])
    has_state = np.array([r[3] for r in rows])
    # One spare block per row past the checkpoint groups, for the crossing.
    blocks = 1 + (WINDOW + 1) * REQ_ROWS + 8
    perm = rng.permutation(np.arange(1, blocks))[: (WINDOW + 1) * REQ_ROWS]
    groups = perm.reshape(REQ_ROWS, WINDOW + 1).astype(np.int32)
    ckpt = groups[:, :WINDOW].copy()
    ckpt[~has_state] = 0
    ids = np.where(has_state, ckpt[:, 0], 0).astype(np.int32)
    offsets = rng.integers(0, WINDOW, REQ_ROWS)
    read_ids = ckpt[np.arange(REQ_ROWS), offsets].astype(np.int32)
    pool = _random_pool(seed, blocks)
    # Finite histories in every block a row reads; random bytes elsewhere.
    histories = jax.random.normal(
        jax.random.key(seed + 1), (REQ_ROWS, PLE_S * PLE_C), jnp.bfloat16
    )
    pool = jax.jit(ple_pool.write_region)(pool, histories, jnp.asarray(read_ids))
    x = jax.random.normal(jax.random.key(seed + 2), (tokens, PLE_C), jnp.bfloat16)
    weight = jax.random.normal(jax.random.key(seed + 3), (PLE_C, 4), jnp.bfloat16)
    meta = dict(groups=groups, has_state=has_state, offsets=offsets)
    args = [
        x,
        weight,
        pool,
        jnp.asarray(starts),
        jnp.asarray(ids),
        jnp.asarray(initial),
        jnp.asarray(read_ids),
        jnp.asarray(ckpt),
        jnp.asarray(windowed),
    ]
    return args, meta


def _step(flag):
    """A fresh jit per flag value: the flag is read while tracing."""
    os.environ[FLAG] = "1" if flag else "0"

    def fn(x, weight, pool, *rest):
        return ple_pool.pooled_short_conv_speculative(
            x, weight, pool, *rest, dilation=PLE_DIL, state_len=PLE_S
        )

    return jax.jit(fn, donate_argnums=(2,))


def _run(args, flag):
    args = list(args)
    args[2] = jnp.copy(args[2])
    pool, out = _step(flag)(*args)
    return _pool_bytes(pool), np.asarray(out).view(np.uint16)


def _assert_same(new, old, what):
    new_pool, new_out = new
    old_pool, old_out = old
    diff = np.nonzero((new_pool != old_pool).reshape(new_pool.shape[0], -1).any(1))[0]
    assert diff.size == 0, f"{what}: pool blocks differ: {diff[:16].tolist()}"
    np.testing.assert_array_equal(new_out, old_out, err_msg=f"{what}: conv output")


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_row_kernel_pool_bytes(name):
    args, _ = _inputs(name, seed=len(name))
    before = _pool_bytes(args[2])
    old = _run(args, False)
    new = _run(args, True)
    written = (old[0] != before).reshape(before.shape[0], -1).any(1).sum()
    print(f"PLEROWS exact layout={name} blocks_written={int(written)}", flush=True)
    assert written > 0
    _assert_same(new, old, name)


def _seed_crossing(args, meta, rng):
    """The runner's seed copy and offset reset for a third of the rows.

    Crossing rows: the group slides to blocks 1..4 of the row's five, the
    checkpoint the row resumes from is copied into the new checkpoint 0 (a
    whole-block copy, as the seed copy does), and the row reads checkpoint 0.
    Other rows resume from one of the checkpoints the first step wrote.
    """
    groups, has_state = meta["groups"], meta["has_state"]
    # A row resumes from a checkpoint the previous step wrote: the read
    # offset is below the accepted count, so below the row's written columns.
    starts = np.asarray(args[3])
    written = np.where(
        np.asarray(args[8]), np.minimum(starts[1:] - starts[:-1], WINDOW), 1
    )
    accepted = (rng.random(REQ_ROWS) * np.maximum(written, 1)).astype(np.int64)
    cross = has_state & (np.arange(REQ_ROWS) % 3 == 0)
    ckpt = np.where(cross[:, None], groups[:, 1:], groups[:, :WINDOW]).astype(np.int32)
    ckpt[~has_state] = 0
    src = groups[np.arange(REQ_ROWS), accepted][cross]
    dst = groups[cross, 1]
    read_ids = ckpt[np.arange(REQ_ROWS), np.where(cross, 0, accepted)].astype(np.int32)
    ids = np.where(has_state, ckpt[:, 0], 0).astype(np.int32)
    out = list(args)
    out[4], out[6], out[7] = jnp.asarray(ids), jnp.asarray(read_ids), jnp.asarray(ckpt)
    return (jnp.asarray(src), jnp.asarray(dst)), out


def _seed_then_step(flag):
    os.environ[FLAG] = "1" if flag else "0"

    def fn(src, dst, x, weight, pool, *rest):
        pool = pool.at[dst].set(pool[src])
        return ple_pool.pooled_short_conv_speculative(
            x, weight, pool, *rest, dilation=PLE_DIL, state_len=PLE_S
        )

    return jax.jit(fn, donate_argnums=(4,))


@pytest.mark.parametrize("name", ["decode48_t192", "short_t96"])
def test_row_kernel_across_a_seed_crossing(name):
    args, meta = _inputs(name, seed=100 + len(name))
    results = {}
    for flag in (False, True):
        step1 = list(args)
        step1[2] = jnp.copy(args[2])
        pool, out1 = _step(flag)(*step1)
        rng = np.random.default_rng(7)
        (src, dst), step2 = _seed_crossing(args, meta, rng)
        step2[0] = jax.random.normal(jax.random.key(9), step2[0].shape, jnp.bfloat16)
        step2[2] = pool
        pool, out2 = _seed_then_step(flag)(src, dst, *step2)
        results[flag] = (
            _pool_bytes(pool),
            np.concatenate(
                [np.asarray(out1).view(np.uint16), np.asarray(out2).view(np.uint16)]
            ),
        )
    print(f"PLEROWS crossing layout={name} checked", flush=True)
    _assert_same(results[True], results[False], f"{name} crossing")


_HIST_PARAM = re.compile(r"^\s+\S+ = bf16\[\d+,9,10240\]\S* parameter\(")


def _history_gather_fusions(hlo):
    """Fused computations that take a [R, 9, 10240] parameter and gather."""
    found, block = 0, []
    for line in hlo.split("\n"):
        if line.endswith("{") and not line.startswith(" "):
            block = []
        block.append(line)
        if line == "}":
            if any(_HIST_PARAM.match(b) for b in block) and any(" gather(" in b for b in block):
                found += 1
            block = []
    return found


@pytest.mark.parametrize("name", ["decode80_t320", "prefill_t1024"])
def test_row_kernel_leaves_no_xla_history_gather(name):
    args, _ = _inputs(name, seed=3)
    counts = {}
    for flag in (False, True):
        hlo = _step(flag).lower(*args).compile().as_text()
        counts[flag] = _history_gather_fusions(hlo)
    print(f"PLEROWS hlo layout={name} xla_history_gathers off={counts[False]} on={counts[True]}", flush=True)
    assert counts[False] > 0, "the check found no history gather with the flag off"
    assert counts[True] == 0
