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

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    mla_flash_deep_prefix as _mla_dp,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.mla_a2a import (
    mla_local_normed_a2a_attention,
)

can_fuse_mla_flash_a2a_deep_prefix = _mla_dp.can_fuse_mla_flash_a2a_deep_prefix
mla_flash_prefill_packed_a2a_deep_prefix = (
    _mla_dp.mla_flash_prefill_packed_a2a_deep_prefix
)
mla_flash_prefill_packed_deep_prefix = _mla_dp.mla_flash_prefill_packed_deep_prefix
select_mla_flash_tile_rows = _mla_dp.select_mla_flash_tile_rows


def _numpy_fp64_packed_mla_oracle(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    length: int,
    segments: np.ndarray,
    *,
    count: int,
    arena: tuple[np.ndarray, np.ndarray] | None = None,
    layer: int = 0,
) -> np.ndarray:
    heads, tokens, _ = query.shape
    out = np.zeros((heads, tokens, 128), dtype=np.float64)
    offsets = segments[:count]
    lengths = segments[count : 2 * count]
    bases = segments[3 * count : 4 * count] if arena is not None else None
    cached_sizes = segments[4 * count : 5 * count] if arena is not None else None

    q64 = query.astype(np.float64)
    k64 = key.astype(np.float64)
    v64 = value.astype(np.float64)
    ak64 = arena[0][layer].astype(np.float64) if arena is not None else None
    av64 = arena[1][layer].astype(np.float64) if arena is not None else None

    for s in range(count):
        off = int(offsets[s])
        seq_end = min(int(off + lengths[s]), int(length))
        if seq_end <= off:
            continue
        for r in range(off, seq_end):
            for h in range(heads):
                k_parts = []
                v_parts = []
                if (
                    arena is not None
                    and bases is not None
                    and cached_sizes is not None
                    and int(cached_sizes[s]) > 0
                ):
                    b_start = int(bases[s])
                    b_end = b_start + int(cached_sizes[s])
                    assert ak64 is not None and av64 is not None
                    k_parts.append(ak64[h, b_start:b_end, :])
                    v_parts.append(av64[h, :, b_start:b_end].T)
                k_parts.append(k64[h, off : r + 1, :])
                v_parts.append(v64[h, off : r + 1, :])
                all_k = np.concatenate(k_parts, axis=0)
                all_v = np.concatenate(v_parts, axis=0)
                scores = all_k @ q64[h, r, :]
                max_s = np.max(scores)
                probs = np.exp2(scores - max_s)
                out[h, r, :] = (probs @ all_v) / np.sum(probs)
    return out.astype(np.float32)


def _make_ring_tables(token_ranks: tuple[int, ...]) -> np.ndarray:
    num_ranks = len(token_ranks)
    tables = np.zeros((3, 3, num_ranks), dtype=np.int32)
    for fam in range(3):
        tables[fam, 0] = np.arange(num_ranks, dtype=np.int32)
        tables[fam, 1] = np.arange(num_ranks, dtype=np.int32)
        tables[fam, 2] = np.asarray(token_ranks, dtype=np.int32)
    return tables


def _unstripe_owner_attended(
    owner_rows: np.ndarray, *, num_ranks: int, stripe: int
) -> np.ndarray:
    _, rows_per_rank, _ = owner_rows.shape
    blocks = rows_per_rank // stripe
    tokens = num_ranks * rows_per_rank
    window_384 = (
        owner_rows.reshape(num_ranks, blocks, stripe, 384)
        .transpose(1, 0, 2, 3)
        .reshape(tokens, 3, 128)
    )
    return window_384.transpose(1, 0, 2)


def test_should_select_expected_tile_rows_across_window_and_prefix_regimes() -> None:
    assert select_mla_flash_tile_rows(512, tile_rows=128) == (512, 512, 512)
    assert select_mla_flash_tile_rows(16384, tile_rows=128) == (2048, 1024, 1024)
    dummy_arena = (
        jnp.zeros((1, 3, 4096, 256), jnp.float8_e4m3fn),
        jnp.zeros((1, 3, 128, 4096), jnp.float8_e4m3fn),
    )
    assert select_mla_flash_tile_rows(16384, tile_rows=128, arena=dummy_arena) == (
        2048,
        1024,
        2048,
    )
    assert select_mla_flash_tile_rows(
        4096,
        tile_rows=128,
        a2a_block_rows=2048,
        mla_flash_qtile=512,
        mla_flash_ktile=1024,
    ) == (2048, 1024, 1024)
    assert can_fuse_mla_flash_a2a_deep_prefix(
        16384, 32, 64, tile_rows=128, arena=dummy_arena
    )
    assert can_fuse_mla_flash_a2a_deep_prefix(16384, 32, 64, tile_rows=128, arena=None)
    assert not can_fuse_mla_flash_a2a_deep_prefix(16384, 32, 0)
    assert not can_fuse_mla_flash_a2a_deep_prefix(16384, 32, 12)
    assert not can_fuse_mla_flash_a2a_deep_prefix(1000, 4, 64)


def test_should_reject_invalid_operand_shapes_and_contracts() -> None:
    q = jnp.zeros((3, 256, 256), jnp.bfloat16)
    k = jnp.zeros((3, 256, 256), jnp.bfloat16)
    v = jnp.zeros((3, 256, 128), jnp.bfloat16)
    segs3 = jnp.asarray([0, 256, 0], jnp.int32)
    segs5 = jnp.asarray([0, 256, 0, 0, 256], jnp.int32)
    tables = jnp.asarray(_make_ring_tables((0,)), jnp.int32)

    with pytest.raises(ValueError, match=r"query must be \[H,T,256\]"):
        mla_flash_prefill_packed_deep_prefix(
            jnp.zeros((256, 256), jnp.bfloat16), k, v, 256, segs3, count=1
        )
    with pytest.raises(ValueError, match=r"query must be \[H,T,256\]"):
        mla_flash_prefill_packed_deep_prefix(
            jnp.zeros((3, 256, 128), jnp.bfloat16), k, v, 256, segs3, count=1
        )
    with pytest.raises(ValueError, match="3 heads"):
        mla_flash_prefill_packed_deep_prefix(
            jnp.zeros((4, 256, 256), jnp.bfloat16), k, v, 256, segs3, count=1
        )
    with pytest.raises(ValueError, match="window's rows"):
        mla_flash_prefill_packed_deep_prefix(
            q, jnp.zeros((3, 128, 256), jnp.bfloat16), v, 256, segs3, count=1
        )
    with pytest.raises(ValueError, match="segments must be"):
        mla_flash_prefill_packed_deep_prefix(q, k, v, 256, segs5, count=1)

    bad_arena = (
        jnp.zeros((1, 3, 300, 256), jnp.float8_e4m3fn),
        jnp.zeros((1, 3, 128, 300), jnp.float8_e4m3fn),
    )
    with pytest.raises(ValueError, match="arena must be"):
        mla_flash_prefill_packed_deep_prefix(
            q,
            k,
            v,
            256,
            segs5,
            count=1,
            arena=bad_arena,
            mla_flash_ktile=256,
        )

    with pytest.raises(ValueError, match=r"query must be \[H,T,256\]"):
        mla_flash_prefill_packed_a2a_deep_prefix(
            tables,
            jnp.zeros((256, 256), jnp.bfloat16),
            k,
            v,
            256,
            segs3,
            count=1,
            num_ranks=1,
            stripe=256,
        )
    with pytest.raises(ValueError, match="tables"):
        mla_flash_prefill_packed_a2a_deep_prefix(
            jnp.zeros((3, 3, 2), jnp.int32),
            q,
            k,
            v,
            256,
            segs3,
            count=1,
            num_ranks=1,
            stripe=256,
        )
    with pytest.raises(ValueError, match="incompatible with num_ranks"):
        mla_flash_prefill_packed_a2a_deep_prefix(
            tables, q, k, v, 256, segs3, count=1, num_ranks=1, stripe=12
        )
    with pytest.raises(ValueError, match="incompatible with tokens"):
        mla_flash_prefill_packed_a2a_deep_prefix(
            jnp.asarray(_make_ring_tables((0, 1, 2)), jnp.int32),
            jnp.zeros((3, 768, 256), jnp.bfloat16),
            jnp.zeros((3, 768, 256), jnp.bfloat16),
            jnp.zeros((3, 768, 128), jnp.bfloat16),
            768,
            segs3,
            count=1,
            num_ranks=3,
            stripe=256,
            tile=256,
            mla_flash_qtile=1024,
            mla_flash_ktile=2048,
            mla_flash_prefix_tiles=False,
        )


@pytest.mark.parametrize("arena_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
def test_should_match_oracle_and_tile_ref_on_unaligned_prefix_with_nan_tails(
    arena_dtype: jnp.dtype,
) -> None:
    rng = np.random.default_rng(701)
    tokens = 256
    cap = 640
    scale = float(pm.SCALE_LOG2)

    q_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * scale).astype(
        np.float32
    )
    k_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    v_np = (rng.standard_normal((3, tokens, 128), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    v_np[:, 96:128, :] = np.nan
    v_np[:, 224:, :] = np.nan

    ak_np = (rng.standard_normal((2, 3, cap, 256), dtype=np.float32) * 0.6).astype(
        np.float32
    )
    av_np = (rng.standard_normal((2, 3, 128, cap), dtype=np.float32) * 0.6).astype(
        np.float32
    )
    av_np[1, :, :, :20] = np.nan
    av_np[1, :, :, 430:448] = np.nan
    av_np[1, :, :, 608:] = np.nan
    ak_np[0, :, :, :] = np.nan
    av_np[0, :, :, :] = np.nan

    segments = np.asarray([0, 128, 96, 96, 0, 0, 20, 448, 410, 160], dtype=np.int32)
    length = 224

    q_j = jnp.asarray(q_np, jnp.bfloat16)
    k_j = jnp.asarray(k_np, jnp.bfloat16)
    v_j = jnp.asarray(v_np, jnp.bfloat16)
    ak_j = jnp.asarray(ak_np, arena_dtype)
    av_j = jnp.asarray(av_np, arena_dtype)

    got = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_j,
            k_j,
            v_j,
            length,
            jnp.asarray(segments, jnp.int32),
            count=2,
            tile=128,
            arena=(ak_j, av_j),
            layer=1,
            interpret=True,
        ),
        dtype=np.float32,
    )
    tile_ref = np.asarray(
        pm.mla_flash_prefill_packed(
            q_j,
            k_j,
            v_j,
            length,
            jnp.asarray(segments, jnp.int32),
            count=2,
            tile=128,
            arena=(ak_j, av_j),
            layer=1,
            interpret=True,
        ),
        dtype=np.float32,
    )
    oracle = _numpy_fp64_packed_mla_oracle(
        np.asarray(q_j, np.float32),
        np.asarray(k_j, np.float32),
        np.where(np.isnan(v_np), 0.0, np.asarray(v_j, np.float32)),
        length,
        segments,
        count=2,
        arena=(
            np.where(np.isnan(ak_np), 0.0, np.asarray(ak_j.astype(jnp.float32))),
            np.where(np.isnan(av_np), 0.0, np.asarray(av_j.astype(jnp.float32))),
        ),
        layer=1,
    )

    assert not np.isnan(got).any()
    assert np.max(np.abs(got[:, 96:128, :])) == 0.0
    assert np.max(np.abs(got[:, 224:, :])) == 0.0
    assert np.max(np.abs(got - tile_ref)) == 0.0
    rel_l2 = float(np.linalg.norm(got - oracle) / np.linalg.norm(oracle))
    tol = 0.035 if arena_dtype == jnp.float8_e4m3fn else 0.004
    assert rel_l2 < tol

    av_pert = av_np.copy()
    av_pert[1, :, :, 150:250] += 3.5
    got_pert = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_j,
            k_j,
            v_j,
            length,
            jnp.asarray(segments, jnp.int32),
            count=2,
            tile=128,
            arena=(ak_j, jnp.asarray(av_pert, arena_dtype)),
            layer=1,
            interpret=True,
        ),
        dtype=np.float32,
    )
    assert float(np.linalg.norm(got_pert - got) / np.linalg.norm(got)) > 0.08


def test_should_match_tile_ref_on_multi_spike_interior_reruns_and_bf16_jump() -> None:
    rng = np.random.default_rng(702)
    tokens = 256
    cap = 1024
    q_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * 0.18).astype(
        np.float32
    )
    k_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * 0.18).astype(
        np.float32
    )
    v_np = (rng.standard_normal((3, tokens, 128), dtype=np.float32) * 0.25).astype(
        np.float32
    )
    ak_np = (rng.standard_normal((1, 3, cap, 256), dtype=np.float32) * 0.18).astype(
        np.float32
    )
    av_np = (rng.standard_normal((1, 3, 128, cap), dtype=np.float32) * 0.25).astype(
        np.float32
    )
    for idx, blk in enumerate((2, 6)):
        ch = 16 + idx
        q_np[:, idx * 17, ch] = 3.2
        ak_np[0, :, blk * 128 + 5, ch] = 3.2

    segs = jnp.asarray([0, tokens, 0, 0, cap], jnp.int32)
    q_j = jnp.asarray(q_np, jnp.bfloat16)
    k_j = jnp.asarray(k_np, jnp.bfloat16)
    v_j = jnp.asarray(v_np, jnp.bfloat16)
    ak_fp8 = jnp.asarray(ak_np, jnp.float8_e4m3fn)
    av_fp8 = jnp.asarray(av_np, jnp.float8_e4m3fn)

    got_fp8 = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_j,
            k_j,
            v_j,
            tokens,
            segs,
            count=1,
            tile=128,
            arena=(ak_fp8, av_fp8),
            interpret=True,
        ),
        dtype=np.float32,
    )
    ref_fp8 = np.asarray(
        pm.mla_flash_prefill_packed(
            q_j,
            k_j,
            v_j,
            tokens,
            segs,
            count=1,
            tile=128,
            arena=(ak_fp8, av_fp8),
            interpret=True,
        ),
        dtype=np.float32,
    )
    oracle_fp8 = _numpy_fp64_packed_mla_oracle(
        np.asarray(q_j.astype(jnp.float8_e4m3fn).astype(jnp.float32)),
        np.asarray(k_j, np.float32),
        np.asarray(v_j, np.float32),
        tokens,
        np.asarray(segs),
        count=1,
        arena=(
            np.asarray(ak_fp8.astype(jnp.float32)),
            np.asarray(av_fp8.astype(jnp.float32)),
        ),
    )
    assert np.max(np.abs(got_fp8 - ref_fp8)) == 0.0
    assert (
        float(np.linalg.norm(got_fp8 - oracle_fp8) / np.linalg.norm(oracle_fp8)) < 0.035
    )

    tokens_bf = 384
    q_bf_np = (rng.standard_normal((3, tokens_bf, 256), dtype=np.float32) * 0.1).astype(
        np.float32
    )
    k_bf_np = (rng.standard_normal((3, tokens_bf, 256), dtype=np.float32) * 0.1).astype(
        np.float32
    )
    v_bf_np = (
        rng.standard_normal((3, tokens_bf, 128), dtype=np.float32) * 0.25
    ).astype(np.float32)
    ak_bf_np = ak_np.copy()
    ak_bf_np[0, 1, 256:384, :64] += 4.0
    q_bf_np[1, :, :64] = 3.0
    k_bf_np[1, 150, :64] = 5.0
    segs_bf = jnp.asarray([0, tokens_bf, 0, 0, cap], jnp.int32)
    q_bf_j = jnp.asarray(q_bf_np, jnp.bfloat16)
    k_bf_j = jnp.asarray(k_bf_np, jnp.bfloat16)
    v_bf_j = jnp.asarray(v_bf_np, jnp.bfloat16)
    ak_bf = jnp.asarray(ak_bf_np, jnp.bfloat16)
    av_bf = jnp.asarray(av_np, jnp.bfloat16)
    got_bf = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_bf_j,
            k_bf_j,
            v_bf_j,
            tokens_bf,
            segs_bf,
            count=1,
            tile=128,
            arena=(ak_bf, av_bf),
            mla_flash_qtile=128,
            mla_flash_ktile=128,
            mla_flash_prefix_tiles=False,
            interpret=True,
        ),
        dtype=np.float32,
    )
    oracle_bf = _numpy_fp64_packed_mla_oracle(
        np.asarray(q_bf_j, np.float32),
        np.asarray(k_bf_j, np.float32),
        np.asarray(v_bf_j, np.float32),
        tokens_bf,
        np.asarray(segs_bf),
        count=1,
        arena=(np.asarray(ak_bf, np.float32), np.asarray(av_bf, np.float32)),
    )
    assert not np.isnan(got_bf).any()
    assert float(np.linalg.norm(got_bf - oracle_bf) / np.linalg.norm(oracle_bf)) < 0.005


@pytest.mark.parametrize("with_size0_prefix", [False, True])
def test_should_match_tile_ref_on_causal_diagonal_pair_and_fallback(
    with_size0_prefix: bool,
) -> None:
    rng = np.random.default_rng(705 + int(with_size0_prefix))
    tokens = 2048
    scale = float(pm.SCALE_LOG2)
    q_j = jnp.asarray(
        rng.standard_normal((3, tokens, 256), np.float32) * scale, jnp.bfloat16
    )
    k_j = jnp.asarray(
        rng.standard_normal((3, tokens, 256), np.float32) * 0.4, jnp.bfloat16
    )
    v_j = jnp.asarray(
        rng.standard_normal((3, tokens, 128), np.float32) * 0.4, jnp.bfloat16
    )
    arena = (
        (
            jnp.zeros((1, 3, 2048, 256), jnp.float8_e4m3fn),
            jnp.zeros((1, 3, 128, 2048), jnp.float8_e4m3fn),
        )
        if with_size0_prefix
        else None
    )

    segs_full = (
        jnp.asarray([0, tokens, 0, 0, 0], jnp.int32)
        if with_size0_prefix
        else jnp.asarray([0, tokens, 0], jnp.int32)
    )
    got_full = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_j,
            k_j,
            v_j,
            tokens,
            segs_full,
            count=1,
            tile=512,
            arena=arena,
            mla_flash_qtile=1024,
            mla_flash_ktile=512,
            interpret=True,
        ),
        dtype=np.float32,
    )
    ref_full = np.asarray(
        pm.mla_flash_prefill_packed(
            q_j,
            k_j,
            v_j,
            tokens,
            segs_full,
            count=1,
            tile=512,
            arena=arena,
            interpret=True,
        ),
        dtype=np.float32,
    )
    oracle_full = _numpy_fp64_packed_mla_oracle(
        np.asarray(q_j, np.float32),
        np.asarray(k_j, np.float32),
        np.asarray(v_j, np.float32),
        tokens,
        np.asarray(segs_full),
        count=1,
    )
    assert np.max(np.abs(got_full - ref_full)) <= 5e-4
    assert (
        float(np.linalg.norm(got_full - oracle_full) / np.linalg.norm(oracle_full))
        < 0.004
    )

    segs_packed = (
        jnp.asarray([0, 700, 700, 1100, 0, 0, 0, 0, 0, 0], jnp.int32)
        if with_size0_prefix
        else jnp.asarray([0, 700, 700, 1100, 0, 0], jnp.int32)
    )
    got_packed = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            q_j,
            k_j,
            v_j,
            1800,
            segs_packed,
            count=2,
            tile=512,
            arena=arena,
            mla_flash_qtile=1024,
            mla_flash_ktile=512,
            interpret=True,
        ),
        dtype=np.float32,
    )
    ref_packed = np.asarray(
        pm.mla_flash_prefill_packed(
            q_j,
            k_j,
            v_j,
            1800,
            segs_packed,
            count=2,
            tile=512,
            arena=arena,
            interpret=True,
        ),
        dtype=np.float32,
    )
    oracle_packed = _numpy_fp64_packed_mla_oracle(
        np.asarray(q_j, np.float32),
        np.asarray(k_j, np.float32),
        np.asarray(v_j, np.float32),
        1800,
        np.asarray(segs_packed),
        count=2,
    )
    assert np.max(np.abs(got_packed[:, 1800:, :])) == 0.0
    assert np.max(np.abs(got_packed - ref_packed)) <= 5e-4
    assert (
        float(
            np.linalg.norm(got_packed[:, :1800, :] - oracle_packed[:, :1800, :])
            / np.linalg.norm(oracle_packed[:, :1800, :])
        )
        < 0.004
    )


@pytest.mark.parametrize("prefixed", [False, True])
@pytest.mark.parametrize("stripe", [32, 64])
def test_should_route_attended_slices_across_4_devices_in_fused_a2a(
    prefixed: bool, stripe: int
) -> None:
    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("Requires >= 4 CPU devices (JAX_NUM_CPU_DEVICES=4)")
    mesh = Mesh(np.array(devs[:4]), ("k3_sp",))
    num_ranks = 4
    tokens = 512
    cap = 768
    token_ranks = (2, 0, 3, 1)
    tables = _make_ring_tables(token_ranks)

    rng = np.random.default_rng(703 + stripe + int(prefixed))
    scale = float(pm.SCALE_LOG2)
    q_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * scale).astype(
        np.float32
    )
    k_np = (rng.standard_normal((3, tokens, 256), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    v_np = (rng.standard_normal((3, tokens, 128), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    ak_np = (rng.standard_normal((1, 3, cap, 256), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    av_np = (rng.standard_normal((1, 3, 128, cap), dtype=np.float32) * 0.5).astype(
        np.float32
    )
    segs_np = (
        np.asarray([0, tokens, 0, 0, cap], dtype=np.int32)
        if prefixed
        else np.asarray([0, tokens, 0], dtype=np.int32)
    )

    rep = jax.NamedSharding(mesh, P())
    tbl_j = jax.device_put(jnp.asarray(tables, jnp.int32), rep)
    q_j = jax.device_put(jnp.asarray(q_np, jnp.bfloat16), rep)
    k_j = jax.device_put(jnp.asarray(k_np, jnp.bfloat16), rep)
    v_j = jax.device_put(jnp.asarray(v_np, jnp.bfloat16), rep)
    seg_j = jax.device_put(jnp.asarray(segs_np, jnp.int32), rep)
    ak_j = jax.device_put(jnp.asarray(ak_np, jnp.float8_e4m3fn), rep)
    av_j = jax.device_put(jnp.asarray(av_np, jnp.float8_e4m3fn), rep)
    ip = pltpu.InterpretParams(detect_races=True)

    @jax.jit
    def run_4dev(tbl, q_in, k_in, v_in, seg_in, ak_in, av_in):
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(), P(), P(), P(), P(), P(), P()),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
        def _sm(t_, q_, k_, v_, s_, ak_, av_):
            rk = jax.lax.axis_index("k3_sp")
            arena_arg = (ak_, av_) if prefixed else None
            o = mla_flash_prefill_packed_a2a_deep_prefix(
                t_,
                q_,
                k_,
                v_,
                tokens,
                s_,
                count=1,
                num_ranks=num_ranks,
                stripe=stripe,
                tile=128,
                arena=arena_arg,
                sp_axis="k3_sp",
                mla_flash_qtile=256,
                mla_flash_ktile=256,
                interpret=ip,
            )
            return o[rk][None]

        return _sm(tbl, q_in, k_in, v_in, seg_in, ak_in, av_in)

    sharded_out = np.asarray(
        run_4dev(tbl_j, q_j, k_j, v_j, seg_j, ak_j, av_j), dtype=np.float32
    )
    owner_ordered = np.empty_like(sharded_out)
    for mesh_rank, owner in enumerate(token_ranks):
        owner_ordered[owner] = sharded_out[mesh_rank]
    ref_single = np.asarray(
        mla_flash_prefill_packed_deep_prefix(
            jnp.asarray(q_np, jnp.bfloat16),
            jnp.asarray(k_np, jnp.bfloat16),
            jnp.asarray(v_np, jnp.bfloat16),
            tokens,
            jnp.asarray(segs_np, jnp.int32),
            count=1,
            tile=128,
            arena=(
                jnp.asarray(ak_np, jnp.float8_e4m3fn),
                jnp.asarray(av_np, jnp.float8_e4m3fn),
            )
            if prefixed
            else None,
            mla_flash_qtile=256,
            mla_flash_ktile=256,
            interpret=True,
        ),
        dtype=np.float32,
    )
    unstriped = _unstripe_owner_attended(
        owner_ordered, num_ranks=num_ranks, stripe=stripe
    )
    assert np.max(np.abs(unstriped - ref_single)) == 0.0


@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("use_pallas_a2a", [True, False])
def test_should_compose_through_mla_local_normed_a2a_attention(
    with_prefix: bool, use_pallas_a2a: bool
) -> None:
    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("Requires >= 4 CPU devices (JAX_NUM_CPU_DEVICES=4)")
    mesh = Mesh(np.array(devs[:4]), ("k3_sp",))
    num_ranks = 4
    rows = 64
    tokens = num_ranks * rows
    width = 256
    q_rank = 128
    cap = 4096
    token_ranks = (1, 3, 0, 2)
    tables = _make_ring_tables(token_ranks)

    rng = np.random.default_rng(710 + int(use_pallas_a2a) + 2 * int(with_prefix))
    lh_np = (rng.standard_normal((num_ranks, rows, width), np.float32) * 0.2).astype(
        np.float32
    )
    px_np = (rng.standard_normal((num_ranks, rows, width), np.float32) * 0.2).astype(
        np.float32
    )
    mqa_np = (rng.standard_normal((width, q_rank), np.float32) * 0.1).astype(np.float32)
    mqb_np = (
        rng.standard_normal((num_ranks, q_rank, 3 * 256), np.float32) * 0.1
    ).astype(np.float32)
    mka_np = (rng.standard_normal((width, 640), np.float32) * 0.1).astype(np.float32)
    mkb_np = (rng.standard_normal((num_ranks, 512, 3 * 256), np.float32) * 0.1).astype(
        np.float32
    )
    mqn_np = np.ones((q_rank,), dtype=np.float32)
    mkn_np = np.ones((512,), dtype=np.float32)
    mg_np = (rng.standard_normal((1, width, num_ranks * 384), np.float32) * 0.1).astype(
        np.float32
    )
    mo_np = (rng.standard_normal((1, num_ranks * 384, width), np.float32) * 0.1).astype(
        np.float32
    )
    ak_np = (rng.standard_normal((num_ranks, 1, 3, cap, 256), np.float32) * 0.3).astype(
        np.float32
    )
    av_np = (rng.standard_normal((num_ranks, 1, 3, 128, cap), np.float32) * 0.3).astype(
        np.float32
    )
    segs_np = (
        np.asarray([0, tokens, 0, 0, cap], dtype=np.int32)
        if with_prefix
        else np.asarray([0, tokens, 0], dtype=np.int32)
    )

    shd = jax.NamedSharding(mesh, P("k3_sp"))
    rep = jax.NamedSharding(mesh, P())
    ip = pltpu.InterpretParams(detect_races=True)

    @jax.jit
    def run_layer(lh, px, mqa, mqb, mka, mkb, mqn, mkn, mg, mo, tbl, seg, ak, av):
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(
                P("k3_sp"),
                P("k3_sp"),
                P(),
                P("k3_sp"),
                P(),
                P("k3_sp"),
                P(),
                P(),
                P(),
                P(),
                P(),
                P(),
                P("k3_sp"),
                P("k3_sp"),
            ),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )
        def _sm(
            lh_, px_, mqa_, mqb_, mka_, mkb_, mqn_, mkn_, mg_, mo_, tbl_, seg_, ak_, av_
        ):
            after, lat = mla_local_normed_a2a_attention(
                lh_[0],
                px_[0],
                query_down_projection=mqa_,
                query_up_projection=mqb_[0],
                key_value_down_projection=mka_,
                key_value_up_projection=mkb_[0],
                query_normalization=mqn_,
                key_value_normalization=mkn_,
                m_gate_family=mg_,
                m_o_family=mo_,
                slot=0,
                length=224,
                segment_desc=seg_,
                num_ranks=num_ranks,
                sp_axis="k3_sp",
                segments=1,
                flash_tile=128,
                token_ranks=token_ranks,
                tables=tbl_,
                arena_override=(ak_[0], av_[0]) if with_prefix else None,
                use_pallas_a2a=use_pallas_a2a,
                use_pallas_owner=False,
                interpret=ip,
            )
            return after[None], lat[None]

        return _sm(lh, px, mqa, mqb, mka, mkb, mqn, mkn, mg, mo, tbl, seg, ak, av)

    after_out, lat_out = run_layer(
        jax.device_put(jnp.asarray(lh_np, jnp.bfloat16), shd),
        jax.device_put(jnp.asarray(px_np, jnp.bfloat16), shd),
        jax.device_put(jnp.asarray(mqa_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(mqb_np, jnp.bfloat16), shd),
        jax.device_put(jnp.asarray(mka_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(mkb_np, jnp.bfloat16), shd),
        jax.device_put(jnp.asarray(mqn_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(mkn_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(mg_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(mo_np, jnp.bfloat16), rep),
        jax.device_put(jnp.asarray(tables, jnp.int32), rep),
        jax.device_put(jnp.asarray(segs_np, jnp.int32), rep),
        jax.device_put(jnp.asarray(ak_np, jnp.float8_e4m3fn), shd),
        jax.device_put(jnp.asarray(av_np, jnp.float8_e4m3fn), shd),
    )

    after_np = np.asarray(after_out, dtype=np.float32)
    assert after_np.shape == (num_ranks, rows, width)
    assert not np.isnan(after_np).any()
    assert float(np.linalg.norm(after_np)) > 0.0
    assert lat_out.shape == (num_ranks, tokens, 576)
