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
"""The DCP indexer score all-to-all kernel on the 8 cores of one host, bitwise
against ``indexer_a2a.reference`` and ``lax.all_to_all(..., tiled=True)`` on
both the direct-HBM (``tokens_per_core == 1``) and peer-VMEM staging
(``tokens_per_core > 1``, up to the ``T = 224`` cap) paths, with the core's
index read from the axis or handed in as data, on a one-axis mesh and on a
``(lane, axis)`` mesh like the DCP one."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    all_gather as ag,
)
from vllm_torchtpu.kernels.experimental.small_collectives import (
    indexer_a2a,
)

from .oracle import AXIS, N, custom_call_operands_in_vmem

pytestmark = pytest.mark.multichip

LANE = "lane"
K = indexer_a2a.SCORE_A2A_K


def _scores(n: int, tokens_per_core: int, seed: int) -> np.ndarray:
    """Distinct int32 words per (core, token, slot), including the -inf bit
    pattern the DCP indexer carries for unfilled local slots."""
    rng = np.random.default_rng(seed)
    t = n * tokens_per_core
    x = rng.integers(-(2**31), 2**31 - 1, size=(n, t, K), dtype=np.int64).astype(
        np.int32
    )
    x[:, :, ::11] = np.int32(np.array(-np.inf, np.float32).view(np.int32))
    return x


def _run(mesh, body, per_core, ranks, lanes=None):
    """``body(x, rank, lane)`` per core over every mesh axis; returns [cores, out]."""
    axes = mesh.axis_names
    spec = P(axes)
    sharding = NamedSharding(mesh, spec)
    n = per_core.shape[0]
    x = jax.device_put(jnp.asarray(per_core.reshape(-1)), sharding)
    r = jax.device_put(jnp.asarray(ranks, jnp.int32), sharding)
    lane = jax.device_put(
        jnp.asarray(lanes if lanes is not None else np.zeros(n), jnp.int32), sharding
    )
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    out = np.asarray(fn(x, r, lane))
    return out.reshape(n, -1)


# tokens_per_core = 1 takes the direct-HBM path; 4 and 28 (T = 224, the largest
# admitted bucket at N = 8) stage through peer VMEM scratch.
@pytest.mark.parametrize("tokens_per_core", [1, 4, 28])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_reference_and_xla_all_to_all(mesh, tokens_per_core, rank_operand):
    per_core = _scores(N, tokens_per_core, seed=tokens_per_core + int(rank_operand))
    t = N * tokens_per_core

    def body(x, r, _):
        return indexer_a2a.small_indexer_score_all_to_all(
            x,
            num_cores=N,
            tokens_per_core=tokens_per_core,
            k=K,
            axis=AXIS,
            rank=r if rank_operand else None,
        )

    got = _run(mesh, body, per_core, range(N)).reshape(N, tokens_per_core, N * K)
    want = indexer_a2a.reference(per_core)
    np.testing.assert_array_equal(got, want)

    xla = _run(
        mesh,
        lambda x, r, _: lax.all_to_all(
            x.reshape(t, K), AXIS, split_axis=0, concat_axis=1, tiled=True
        ).reshape(-1),
        per_core,
        range(N),
    ).reshape(N, tokens_per_core, N * K)
    np.testing.assert_array_equal(got, xla)


@pytest.mark.parametrize("tokens_per_core", [1, 4])
def test_float_bits_pass_through(mesh, tokens_per_core):
    """f32 blocks (NaN payloads, -0.0, -inf) come back bit for bit."""
    bits = _scores(N, tokens_per_core, seed=17 + tokens_per_core)
    bits[:, 0, 0] = np.int32(-2147483648)  # -0.0f
    bits[:, 0, 1] = np.int32(0x7FC01234)  # quiet NaN with payload
    per_core = bits.view(np.float32)
    got = _run(
        mesh,
        lambda x, r, _: indexer_a2a.small_indexer_score_all_to_all(
            x,
            num_cores=N,
            tokens_per_core=tokens_per_core,
            k=K,
            axis=AXIS,
            rank=r,
        ),
        per_core,
        range(N),
    )
    want = indexer_a2a.reference(bits).reshape(N, -1)
    np.testing.assert_array_equal(got.view(np.int32), want)


@pytest.mark.parametrize("tokens_per_core", [1, 4])
def test_repeated_calls_and_stage3_all_gather_in_one_program(mesh, tokens_per_core):
    """Back-to-back calls share the barrier semaphore and chain cleanly into
    Stage 3's ``small_all_gather``."""
    per_core = _scores(N, tokens_per_core, seed=31 + tokens_per_core)

    def body(x, r, _):
        a = indexer_a2a.small_indexer_score_all_to_all(
            x, num_cores=N, tokens_per_core=tokens_per_core, k=K, axis=AXIS, rank=r
        )
        b = indexer_a2a.small_indexer_score_all_to_all(
            x + 1, num_cores=N, tokens_per_core=tokens_per_core, k=K, axis=AXIS, rank=r
        )
        # Take the first K words per token as stand-in Stage-2 slots and gather.
        slots = (
            a.reshape(tokens_per_core, N * K)[:, :K]
            ^ b.reshape(tokens_per_core, N * K)[:, -K:]
        ).reshape(-1)
        gathered = ag.small_all_gather(slots, num_cores=N, axis=AXIS, rank=r)
        return jnp.concatenate([a, b, gathered])

    got = _run(mesh, body, per_core, range(N))
    want_a = indexer_a2a.reference(per_core)
    want_b = indexer_a2a.reference(per_core + np.int32(1))
    slots_ref = want_a[:, :, :K] ^ want_b[:, :, -K:]
    want_ag = np.broadcast_to(
        slots_ref.reshape(N * tokens_per_core * K), (N, N * tokens_per_core * K)
    )
    span = tokens_per_core * N * K
    np.testing.assert_array_equal(got[:, :span], want_a.reshape(N, span))
    np.testing.assert_array_equal(got[:, span : 2 * span], want_b.reshape(N, span))
    np.testing.assert_array_equal(got[:, 2 * span :], want_ag)


@pytest.mark.parametrize("tokens_per_core", [1, 4])
@pytest.mark.parametrize("lane_data", [False, True])
def test_lane_mesh_exchanges_within_each_lane(tokens_per_core, lane_data):
    """(2 lanes, 4 cores): each lane is its own group, as DCP's lanes are."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    lanes, n = 2, N // 2
    lane_mesh = Mesh(np.array(jax.devices()[:N]).reshape(lanes, n), (LANE, AXIS))
    per_core = np.concatenate(
        [
            _scores(n, tokens_per_core, seed=40 + ln + tokens_per_core)
            for ln in range(lanes)
        ],
        axis=0,
    )
    ranks = np.tile(np.arange(n), lanes)
    lane_ids = np.repeat(np.arange(lanes), n)

    def body(x, r, lane):
        return indexer_a2a.small_indexer_score_all_to_all(
            x,
            num_cores=n,
            tokens_per_core=tokens_per_core,
            k=K,
            axis=AXIS,
            rank=r,
            lane_axis=LANE,
            lane=lane if lane_data else None,
        )

    got = _run(lane_mesh, body, per_core, ranks, lane_ids).reshape(
        lanes, n, tokens_per_core, n * K
    )
    for ln in range(lanes):
        want = indexer_a2a.reference(per_core[ln * n : (ln + 1) * n])
        np.testing.assert_array_equal(got[ln], want)


@pytest.mark.parametrize("tokens_per_core", [1, 4])
def test_a_wrong_rank_order_fails(mesh, tokens_per_core):
    """Positive control: cores 0 and 1 told each other's index write into each
    other's sender slots on peer cores, and the comparison has to see it."""
    per_core = _scores(N, tokens_per_core, seed=55 + tokens_per_core)
    got = _run(
        mesh,
        lambda x, r, _: indexer_a2a.small_indexer_score_all_to_all(
            x, num_cores=N, tokens_per_core=tokens_per_core, k=K, axis=AXIS, rank=r
        ),
        per_core,
        [1, 0, *range(2, N)],
    )
    want = indexer_a2a.reference(per_core).reshape(N, -1)
    assert (got[2] != want[2]).any()


@pytest.mark.parametrize("tokens_per_core", [1, 4])
def test_hbm_operand_is_bitwise_and_not_staged_in_vmem(mesh, tokens_per_core):
    """``hbm_operand=True`` (TPU_SMALL_COLL_HBM_OPERAND): bitwise exact when
    the score buffer comes out of a preceding op, and the compiled call reads
    it from HBM without a copy-start/-done into VMEM in front of it."""
    per_core = _scores(N, tokens_per_core, seed=71 + tokens_per_core)
    length = N * tokens_per_core * K

    def body(x, r, _):
        return indexer_a2a.small_indexer_score_all_to_all(
            x ^ jnp.int32(0x5A5A),
            num_cores=N,
            tokens_per_core=tokens_per_core,
            k=K,
            axis=AXIS,
            rank=r,
            hbm_operand=True,
        )

    got = _run(mesh, body, per_core, range(N)).reshape(N, tokens_per_core, N * K)
    want = indexer_a2a.reference(per_core ^ np.int32(0x5A5A))
    np.testing.assert_array_equal(got, want)

    spec = P(mesh.axis_names)
    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False
        )
    )
    args = (
        jax.ShapeDtypeStruct((N * length,), jnp.int32),
        jax.ShapeDtypeStruct((N,), jnp.int32),
        jax.ShapeDtypeStruct((N,), jnp.int32),
    )
    lowered = fn.lower(*args)
    assert "input_memory_space_colors" in lowered.as_text()
    staged = custom_call_operands_in_vmem(
        lowered.compile().as_text(), "small_idx_score_a2a"
    )
    assert staged and not any(staged.values()), staged


def test_streamindex_topk_dcp_a2a_small_coll_matches_xla():
    """End-to-end ``streamindex_topk_dcp`` on 8 TPU cores with SparseCore
    (``T = 64, k = 2048``): ``a2a_small_coll=True`` matches ``False`` bit for
    bit."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    try:
        if pltpu.get_tpu_info().sparse_core is None:
            pytest.skip("needs SparseCore")
    except Exception:
        pytest.skip("needs SparseCore")

    t, kv_len, page_size = 64, 32768, 1024
    h_idx, d_idx, width = 32, 128, 256
    dcp_mesh = Mesh(np.asarray(jax.devices()[:N]).reshape(1, N), (LANE, DCP_AXIS_NAME))
    rep = NamedSharding(dcp_mesh, P())
    sharded = NamedSharding(dcp_mesh, P(DCP_AXIS_NAME))

    vals = jax.random.normal(jax.random.key(7), (kv_len, d_idx), jnp.float32)
    fp8 = lax.bitcast_convert_type(vals.astype(jnp.float8_e4m3fn), jnp.uint8)
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    rec = jnp.pad(rec, ((0, 0), (0, width - rec.shape[-1])))
    vpages = kv_len // (page_size * N)
    shards = jnp.concatenate(
        [rec[r::N].reshape(vpages, page_size // 4, 4, width) for r in range(N)], 0
    )
    cache_kv = jax.device_put(shards, sharded)

    kq, kw = jax.random.split(jax.random.key(101))
    q = jax.device_put(
        (jax.random.normal(kq, (t, h_idx, d_idx), jnp.float32) * 0.5).astype(
            jnp.float8_e4m3fn
        ),
        rep,
    )
    weights = jax.device_put(
        jax.random.normal(kw, (t, h_idx), jnp.float32).astype(jnp.bfloat16), rep
    )
    seq_lens = jax.device_put(jnp.array([kv_len], jnp.int32), rep)
    page_indices = jax.device_put(jnp.arange(vpages, dtype=jnp.int32), rep)
    cu_q_lens = jax.device_put(jnp.array([0, t], jnp.int32), rep)
    dist = jax.device_put(jnp.array([0, 0, 1], jnp.int32), rep)
    rank_ids = jax.device_put(jnp.arange(N, dtype=jnp.int32), sharded)
    lane_ids = jax.device_put(jnp.zeros((N,), dtype=jnp.int32), sharded)

    def run(small_coll: bool) -> np.ndarray:
        out = jax.jit(
            lambda q_, w_, kv_, sl_, pi_, cq_, d_, r_, ln_: streamindex_topk_dcp(
                q_,
                w_,
                kv_,
                sl_,
                pi_,
                cq_,
                d_,
                mesh=dcp_mesh,
                k=K,
                compression_ratio=1,
                dcp_size=N,
                interleave_size=1,
                num_kv_pages_per_block=2,
                num_queries_per_block=64,
                a2a_merge=True,
                a2a_small_coll=small_coll,
                dcp_rank_ids=r_,
                dcp_lane_ids=ln_,
            )
        )(
            q,
            weights,
            cache_kv,
            seq_lens,
            page_indices,
            cu_q_lens,
            dist,
            rank_ids,
            lane_ids,
        )
        return np.asarray(out)

    np.testing.assert_array_equal(run(True), run(False))
