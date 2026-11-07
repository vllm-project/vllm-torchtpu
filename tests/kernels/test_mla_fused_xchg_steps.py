# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_MLA_FUSED_XCHG_MAX_TOKENS, no TPU needed: the decode MLA kernel runs
the DCP exchange and merge itself over its 8-token grid steps.

On the served (dcp_lane 2, dcp 8) mesh at 32 heads per rank, top-k 1024
capped at 512 (some tokens overflow into the fix-up), the decode row widths
and body levers ("step8"), the kernel with `send_exchange` (merge mode) on a
16- or 32-token bucket returns, bit for bit on every rank, what the
send-layout kernel then `small_all_to_all` with `merge_live` and `live_rows`
return (TPU_DCP_MLA_SEND_SMALL_A2A_MERGE + _LIVE_ROWS): with every token
real, some padding tokens, and none. Positive control: every step's partial
rows pushed into the next rank's slot (the lse rows not) must fail.

Pallas interpret mode with DMAs run on wait, so a merge that read a slot
before its copy was waited would read stale rows. Runs on 16 devices; CPU
virtual devices are sufficient, e.g.
XLA_FLAGS=--xla_force_host_platform_device_count=16."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_to_all
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

pytestmark = pytest.mark.multichip

LANES_N, W, HEADS, V, TOPK, CAP = 2, 8, 32, 512, 1024, 512
L = HEADS // W
XS = P(("dcp_lane", "dcp"))


@pytest.fixture
def mesh():
    if jax.device_count() < LANES_N * W:
        pytest.skip("requires 16 devices; CPU virtual devices are sufficient")
    devices = np.asarray(jax.devices()[: LANES_N * W]).reshape(LANES_N, W)
    return Mesh(devices, ("dcp_lane", "dcp"))


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(
        pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    monkeypatch.setattr(pltpu, "with_memory_space_constraint", lambda x, _: x)


def _fp8_bytes(rng, shape):
    """Finite fp8 e4m3 bytes (no NaN pattern), random sign."""
    sign = rng.integers(0, 2, size=shape, dtype=np.uint8) << 7
    return rng.integers(0, 0x60, size=shape, dtype=np.uint8) | sign


def _problem(tokens, real):
    """Each rank's pre-gathered decode inputs, stacked over the 16 ranks: q
    replicated over a lane's DCP ranks, KV and kv_lens per rank."""
    rng = np.random.default_rng(1000 + 31 * tokens + real)
    n = LANES_N * W
    q = np.repeat(rng.standard_normal((LANES_N, tokens, HEADS, V)), W, 0)
    q_pe = np.repeat(rng.standard_normal((LANES_N, tokens, HEADS, 256)), W, 0)
    ang = rng.integers(0, 4096, size=(LANES_N, tokens, 1)) / 10000 ** (
        np.arange(32) / 32
    )
    cos, sin = np.cos(ang), np.sin(ang)
    tab = np.stack(
        [np.concatenate([cos] * 4, -1), np.concatenate([-sin, sin, -sin, sin], -1)], 2
    )
    windows = TOPK // CAP
    nope = _fp8_bytes(rng, (n, windows * tokens, CAP, 512)).view(np.uint32)
    rope = _fp8_bytes(rng, (n, windows * tokens, CAP, 2, 64))
    rope[..., 1, :] = 0  # 64 rope bytes and 64 pad bytes per key
    rope = rope.reshape(n, windows * tokens, CAP // 4, 512).view(np.uint32)
    kv_lens = rng.integers(0, TOPK + 1, size=(n, tokens)).astype(np.int32)
    kv_lens[:, ::3] = rng.integers(0, 300, size=kv_lens[:, ::3].shape)
    kv_lens[3, ::5] = 0  # rank 3 owns none of these tokens' keys: lse -inf
    kv_lens[:, real:] = 0  # padding tokens
    return [
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(q_pe, jnp.bfloat16),
        jnp.asarray(np.repeat(tab, W, 0), jnp.float32),
        jnp.asarray(nope),
        jnp.asarray(rope),
        jnp.asarray(kv_lens),
        jnp.asarray(np.tile(np.arange(W, dtype=np.int32), LANES_N)),
        jnp.asarray(np.repeat(np.arange(LANES_N, dtype=np.int32), W)),
    ]


def _attend(x, tokens, real, *, exchange=False, fault=None):
    q, q_pe, tab, nope, rope, kv_lens, rank, lane = (a[0] for a in x)
    extra = {}
    if exchange:
        extra = dict(
            send_exchange=("dcp", "dcp_lane", 16 << 20, False, True),
            send_peer=jnp.stack([rank, lane, jnp.int32(real)]),
        )
    return sparse_mla.sparse_attention_chunk(
        q,
        nope,
        rope,
        kv_lens,
        None,
        None,
        start_pos=0,
        end_pos=tokens,
        token_end=jnp.int32(real),
        sm_scale=1 / 24,
        k_scale=0.1,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        small_chunk_tokens=32,
        small_chunk_batch_size=4,
        topk_cap=CAP,
        topk_override=TOPK,
        is_dcp_mode=True,
        q_pe=q_pe,
        lse_lane_dense=True,
        lse_mask_empty=True,
        row_widths=(128, 256, 384),
        row_levers=sparse_mla.DECODE_BODY_DEFAULT,
        q_rope_table=tab,
        lse_fill=True,
        prologue_fault=fault,
        send_world=W,
        pre_gathered=(nope, rope, None),
        **extra,
    )


def _separate(tokens, real):
    """The send-layout kernel, then the small all-to-all's live merge."""

    def body(*x):
        _, send = _attend(x, tokens, real)
        live = jnp.full((1,), real * L, jnp.int32)
        merged = all_to_all.small_all_to_all(
            send.reshape(-1, V),
            num_cores=W,
            axis="dcp",
            row0=(sparse_mla.SEND_PAD_ROWS - 1) // 8 * 8,
            rank=x[6],
            lane_axis="dcp_lane",
            lane=x[7],
            hbm_operand=True,
            merge_live=live,
            merge_row=sparse_mla.SEND_PAD_ROWS,
            live_rows=live,
            live_row0=sparse_mla.SEND_PAD_ROWS,
        )
        return jax.lax.bitcast_convert_type(merged, jnp.int16)[None]

    return body


def _fused(tokens, real, fault=None):
    def body(*x):
        return _attend(x, tokens, real, exchange=True, fault=fault)[1][None]

    return body


def _run(mesh, body, args):
    args = [jax.device_put(a, NamedSharding(mesh, XS)) for a in args]
    fn = jax.shard_map(
        body, mesh=mesh, in_specs=(XS,) * len(args), out_specs=XS, check_vma=False
    )
    return np.asarray(jax.jit(fn)(*args))


@pytest.mark.parametrize(
    "tokens,real", [(16, 16), (16, 11), (16, 0), (32, 32), (32, 19)]
)
def test_fused_merge_equals_kernel_then_small_all_to_all(mesh, interpret, tokens, real):
    args = _problem(tokens, real)
    want = _run(mesh, _separate(tokens, real), args)
    got = _run(mesh, _fused(tokens, real), args)
    assert want.shape == (LANES_N * W, tokens * L, V)
    np.testing.assert_array_equal(got, want)
    if real:
        assert (want[:, : real * L] != 0).any()
        assert not want[:, real * L :].any()


def test_misrouted_partials_fail(mesh, interpret):
    args = _problem(16, 16)
    want = _run(mesh, _separate(16, 16), args)
    bad = _run(mesh, _fused(16, 16, fault="send_peer_misorder"), args)
    assert (bad != want).mean() > 0.5
