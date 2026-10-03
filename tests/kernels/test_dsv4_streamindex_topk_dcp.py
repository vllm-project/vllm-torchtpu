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
"""Tests for DCP (decode context parallel) StreamIndex Top-K"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_local_to_global,
    cp_rank_as_data,
    dcp_sc_compact,
    metadata,
    streamindex_topk,
    streamindex_topk_dcp,
)

P = jax.sharding.PartitionSpec


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_candidate_all_gather_is_rank_major(dcp_size):
    """The merge resolves a winner as rank `slot // k`, position `slot % k`.

    Both readings assume this layout, and nothing else in the merge would
    notice it changing: every rank would resolve the wrong candidate and still
    return a full, plausible, silently wrong list. Asserting the full
    rank-major arange pins order and chunk width together.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")

    rows, k = 4, 8
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME,))

    def _local(_):
        rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
        # Rank r writes r*k + j, so a rank-major gather reads back as arange.
        mine = rank * k + jnp.arange(k, dtype=jnp.int32)
        return jax.lax.all_gather(
            jnp.broadcast_to(mine, (rows, k)), DCP_AXIS_NAME, axis=1, tiled=True
        )

    gathered = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(DCP_AXIS_NAME),
            check_vma=False,
        )
    )(jnp.zeros((rows, 1), jnp.int32))

    got = np.asarray(gathered).reshape(dcp_size, rows, dcp_size * k)
    want = np.broadcast_to(
        np.arange(dcp_size * k, dtype=np.int32), (rows, dcp_size * k)
    )
    for r in range(dcp_size):
        np.testing.assert_array_equal(
            got[r], want, err_msg=f"rank {r} saw a non-rank-major candidate axis"
        )


def _require_sparsecore():
    """Probe in a test body: a chip touch at collection breaks spawned workers."""
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:  # no TPU at all
        pass
    pytest.skip("needs SparseCore")


def _xla_pack_reference(vals):
    """The scatter the kernel replaced, unreachable in `src`, kept as an oracle."""
    num_rows, width = vals.shape
    keep = vals >= 0

    # Exclusive prefix count of kept entries gives the destination slot, < width.
    slot = jnp.cumsum(keep, axis=1, dtype=jnp.int32) - keep
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

    # Column `width` is a dump for the dropped entries, sliced off at the end.
    out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
    out = out.at[rows, jnp.where(keep, slot, width)].set(jnp.where(keep, vals, -1))
    return out[:, :width]


def _sparse_fixture(num_rows=16, width=128, keep_rate=0.5, seed=20260906):
    """Non-negative values with `-1` holes; row 0 is full and row 1 is empty."""
    rng = np.random.default_rng(seed)
    vals = rng.integers(0, 1 << 20, size=(num_rows, width)).astype(np.int32)
    vals[rng.random((num_rows, width)) > keep_rate] = -1
    vals[0] = rng.integers(0, 1 << 20, size=width).astype(np.int32)
    if num_rows > 1:
        vals[1] = -1
    return vals


@pytest.mark.cpu_test
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_build_rank_payload_places_owned_slice_and_neg_one_elsewhere(dcp_size):
    """`build_rank_payload` places `local_idxs` at `[r*w:(r+1)*w]` and `-1` elsewhere."""
    num_rows, width = 6, 16
    rng = np.random.default_rng(20261002 + dcp_size)
    local_idxs = rng.integers(0, 4096, size=(num_rows, width), dtype=np.int32)

    for r in range(dcp_size):
        payload = np.asarray(
            dcp_sc_compact.build_rank_payload(
                jnp.asarray(local_idxs), dcp_size, jnp.int32(r)
            )
        )
        assert payload.shape == (num_rows, dcp_size * width)
        for peer in range(dcp_size):
            col_slice = payload[:, peer * width : (peer + 1) * width]
            if peer == r:
                np.testing.assert_array_equal(col_slice, local_idxs)
            else:
                assert np.all(col_slice == -1)


@pytest.mark.parametrize("num_rows,width", [(16, 128), (1, 16), (33, 256)])
@pytest.mark.parametrize("keep_rate", [0.1, 0.5, 0.9])
def test_sparsecore_pack_matches_xla(num_rows, width, keep_rate):
    """Elementwise, not just as a set: both packs must emit in ascending order."""
    _require_sparsecore()
    vals = _sparse_fixture(num_rows, width, keep_rate)

    want = np.asarray(_xla_pack_reference(jnp.asarray(vals)))
    got = np.asarray(dcp_sc_compact.pack_nonnegative(jnp.asarray(vals)))

    np.testing.assert_array_equal(got, want)


def _simulate_merge(num_rows, k, span, dcp_size, interleave_c, seed=20260929):
    """Stages 1-3 of `streamindex_topk_dcp` in numpy, as `resolve` receives them.

    Returns each rank's rank-local candidate list, the merged flat slot columns,
    and the true global top-k the ranks must reassemble between them.
    """
    rng = np.random.default_rng(seed)
    # Distinct scores, so the top-k is unambiguous and the partition is exact.
    scores = rng.permutation(num_rows * span).reshape(num_rows, span).astype(np.float32)
    owner = (np.arange(span) // interleave_c) % dcp_size

    local_idxs = np.full((dcp_size, num_rows, k), -1, np.int32)
    local_scores = np.full((dcp_size, num_rows, k), -np.inf, np.float32)
    for r in range(dcp_size):
        owned = np.flatnonzero(owner == r)
        for t in range(num_rows):
            best = owned[np.argsort(-scores[t, owned])][:k]
            local_idxs[r, t, : len(best)] = np.asarray(
                metadata.cp_global_to_local(jnp.asarray(best), dcp_size, interleave_c)
            )
            local_scores[r, t, : len(best)] = scores[t, best]

    # Stage 2 is rank-major, so rank r's k candidates land at columns [r*k, r*k+k).
    gathered = np.concatenate(list(local_scores), axis=1)
    slots = np.full((num_rows, k), -1, np.int32)
    for t in range(num_rows):
        order = np.argsort(-gathered[t])[:k]
        order = order[np.isfinite(gathered[t, order])]
        slots[t, : len(order)] = order

    want = [set(np.argsort(-scores[t])[:k].tolist()) for t in range(num_rows)]
    return local_idxs, slots, want


@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4), (8, 64)])
def test_resolve_partitions_the_global_topk(dcp_size, interleave_c):
    """The ranks' packed outputs reassemble the global top-k, each entry once."""
    _require_sparsecore()
    num_rows, k, span = 8, 32, 4096
    local_idxs, slots, want = _simulate_merge(num_rows, k, span, dcp_size, interleave_c)

    lists = np.stack(
        [
            np.asarray(
                dcp_sc_compact.resolve_owned_winners(
                    jnp.asarray(local_idxs[r]), jnp.asarray(slots), k, jnp.int32(r)
                )
            )
            for r in range(dcp_size)
        ]
    )
    assert lists.shape == (dcp_size, num_rows, k)

    for t in range(num_rows):
        got = []
        for r in range(dcp_size):
            row = lists[r, t]
            for local in row[row >= 0].tolist():
                g = int(cp_local_to_global(local, r, dcp_size, interleave_c))
                # A rank must only ever keep positions it actually owns.
                assert (g // interleave_c) % dcp_size == r
                got.append(g)
        # Multiplicity, not just set equality: the LSE merge double-counts a
        # position delivered to two ranks.
        assert len(got) == len(set(got))
        assert set(got) == want[t]


# ===================================================================== #
# TPU7x HLO Lowering Verification (2-stage vs single_stage_score_ag).
# ===================================================================== #


@pytest.mark.cpu_test
def test_dcp_tpu7x_lowering_zero_2d_gather_and_single_stage_score_ag(monkeypatch):
    """Verify TPU7x HLO lowering of `dcp_sc_compact` and `streamindex_topk_dcp`:
    1. `dcp_sc_compact` primitives contain 0 `stablehlo.gather` ops.
    2. `streamindex_topk_dcp` contains 0 2D score/index `stablehlo.gather` ops in
       both `single_stage_score_ag=False` and `single_stage_score_ag=True`.
    3. `single_stage_score_ag=True` (explicit or via `TPU_STREAMIDX_DCP_SINGLE_TOPK_MAX_KV`)
       all-gathers the full `(B, L_local)` score tensor and invokes SparseCore top-k
       once instead of twice.
    """
    ad = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=2, platform="tpu"
    )
    mesh1 = jax.sharding.AbstractMesh((1,), ("x",), abstract_device=ad)
    with jax.sharding.use_abstract_mesh(mesh1):
        x_spec = jax.ShapeDtypeStruct((8, 2048), jnp.int32)
        rank_spec = jax.ShapeDtypeStruct((), jnp.int32)
        hlo_pack = (
            jax.jit(dcp_sc_compact.pack_nonnegative)
            .trace(x_spec)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        assert "stablehlo.gather" not in hlo_pack

        hlo_resolve = (
            jax.jit(
                lambda l, s, r: dcp_sc_compact.resolve_owned_winners(
                    l, s, 2048, r
                )
            )
            .trace(x_spec, x_spec, rank_spec)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        assert "stablehlo.gather" not in hlo_resolve

        hlo_payload = (
            jax.jit(lambda l, r: dcp_sc_compact.build_rank_payload(l, 2, r))
            .trace(x_spec, rank_spec)
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )
        assert "stablehlo.gather" not in hlo_payload

    dcp_size = 2
    mesh_dcp = jax.sharding.AbstractMesh(
        (dcp_size,), (DCP_AXIS_NAME,), abstract_device=ad
    )
    q_len, h_idx, d_idx, k = 128, 32, 128, 2048
    kv_len, page_size, width = 32768, 1024, 256
    vpages = kv_len // (page_size * dcp_size)

    q_spec = jax.ShapeDtypeStruct((q_len, h_idx, d_idx), jnp.float8_e4m3fn)
    w_spec = jax.ShapeDtypeStruct((q_len, h_idx), jnp.bfloat16)
    kc_spec = jax.ShapeDtypeStruct(
        (dcp_size * vpages, page_size // 4, 4, width),
        jnp.uint8,
        sharding=jax.sharding.NamedSharding(mesh_dcp, P(DCP_AXIS_NAME)),
    )
    kl_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    pt_spec = jax.ShapeDtypeStruct((vpages,), jnp.int32)
    qsl_spec = jax.ShapeDtypeStruct((2,), jnp.int32)
    dist_spec = jax.ShapeDtypeStruct((3,), jnp.int32)

    def _lower_dcp(**extra_kw) -> str:
        with jax.sharding.use_abstract_mesh(mesh_dcp):
            fn = jax.jit(
                lambda q, w, kc, kl, pt, qsl, dist: streamindex_topk_dcp(
                    q,
                    w,
                    kc,
                    kl,
                    pt,
                    qsl,
                    dist,
                    mesh=mesh_dcp,
                    k=k,
                    compression_ratio=1,
                    dcp_size=dcp_size,
                    interleave_size=1,
                    num_kv_pages_per_block=2,
                    num_queries_per_block=128,
                    **extra_kw,
                )
            )
            return (
                fn.trace(
                    q_spec, w_spec, kc_spec, kl_spec, pt_spec, qsl_spec, dist_spec
                )
                .lower(lowering_platforms=("tpu",))
                .as_text()
            )

    hlo_two_stage = _lower_dcp(single_stage_score_ag=False)
    hlo_single_stage = _lower_dcp(single_stage_score_ag=True)

    # Any 2D score/index gather from `take_along_axis` would index a 2D tensor
    # `tensor<128x...xi32>` or `tensor<128x...xf32>`. Only 1D metadata gathers
    # on `query_start_loc` (`tensor<2xi32>`) and `k_lens` (`tensor<1xi32>`) are allowed.
    for mode_name, hlo_text in [
        ("two_stage", hlo_two_stage),
        ("single_stage", hlo_single_stage),
    ]:
        gather_lines = [
            line for line in hlo_text.splitlines() if "stablehlo.gather" in line
        ]
        for line in gather_lines:
            assert re.search(r"\(tensor<\d+xi32>,", line), (
                f"Unexpected non-1D gather in {mode_name}: {line}"
            )

    # 2-stage all-gathers `(128, 2048) -> (128, 4096)`; 1-stage all-gathers
    # `(128, 16384) -> (128, 32768)`.
    assert "(tensor<128x2048xi32>) -> tensor<128x4096xi32>" in hlo_two_stage
    assert "(tensor<128x16384xi32>) -> tensor<128x32768xi32>" in hlo_single_stage

    # Verify env threshold `TPU_STREAMIDX_DCP_SINGLE_TOPK_MAX_KV`.
    monkeypatch.setenv("TPU_STREAMIDX_DCP_SINGLE_TOPK_MAX_KV", "32768")
    streamindex_topk_dcp._clear_cache()
    hlo_env_enabled = _lower_dcp()
    assert "(tensor<128x16384xi32>) -> tensor<128x32768xi32>" in hlo_env_enabled

    monkeypatch.setenv("TPU_STREAMIDX_DCP_SINGLE_TOPK_MAX_KV", "16384")
    streamindex_topk_dcp._clear_cache()
    hlo_env_disabled = _lower_dcp()
    assert "(tensor<128x2048xi32>) -> tensor<128x4096xi32>" in hlo_env_disabled


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX), jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(
        vals.astype(jnp.float8_e4m3fn), jnp.uint8
    ).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


def _need_devices(dcp_size):
    n = len(jax.devices())
    if n < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {n}")


def _dcp_case(dcp_size, seed):
    """Records plus the positional args and kwargs of one sharded DCP call."""
    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(seed))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    weights = jax.random.normal(kw, (Q_LEN, H_IDX), jnp.float32).astype(jnp.bfloat16)

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate(
        [
            rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
            for r in range(dcp_size)
        ],
        0,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,))
    shards = jax.device_put(shards, jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    args = (
        q,
        weights,
        shards,
        jnp.array([KV_LEN], jnp.int32),
        jnp.arange(vpages, dtype=jnp.int32),
        jnp.array([0, Q_LEN], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    kwargs = dict(
        mesh=mesh,
        k=K,
        compression_ratio=1,
        dcp_size=dcp_size,
        interleave_size=1,
        num_kv_pages_per_block=2,
        num_queries_per_block=128,
    )
    return rec, args, kwargs


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("single_stage_score_ag", [False, True])
def test_dcp_reassembles_the_unsharded_global_topk(
    dcp_size, single_stage_score_ag
):
    """Union the per-rank lists back together; it must be the global top-k"""
    _need_devices(dcp_size)
    rec, args, kwargs = _dcp_case(dcp_size, 0)
    q, weights = args[0], args[1]

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(
            q,
            weights,
            rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
            jnp.array([KV_LEN], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32),
            jnp.array([0, Q_LEN], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=128,
        )
    )

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(
        streamindex_topk_dcp(
            *args, **kwargs, single_stage_score_ag=single_stage_score_ag
        )
    ).reshape(dcp_size, Q_LEN, K)
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
        got = set()
        for r in range(dcp_size):
            row = local_topk[r, t]
            for local in row[row >= 0].tolist():
                got.add(int(cp_local_to_global(local, r, dcp_size, 1)))
        want = set(reference[t][reference[t] >= 0].tolist())
        assert got == want, (
            f"token {t}: {len(want - got)} missing, {len(got - want)} spurious"
        )


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("chunk_tokens", [32, 64, 128])
def test_dcp_chunking_does_not_change_the_result(chunk_tokens, dcp_size):
    _need_devices(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 11)

    hlo = streamindex_topk_dcp.lower(
        *args, **kwargs, chunk_tokens=chunk_tokens
    ).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the DCP path ran a single unchunked"
        " pass, so this comparison would pass trivially"
    )

    unchunked = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    chunked = np.asarray(
        streamindex_topk_dcp(*args, **kwargs, chunk_tokens=chunk_tokens)
    )
    np.testing.assert_array_equal(chunked, unchunked)
