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
"""Behavioral specs for 3D head-major shift+FP8-quant+pair-split epilogue."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel as tp
from vllm_torchtpu.kernels.mla import shift_quant_split_3d as sq3d

AXIS = "dcp"
SENTINEL = 999.0


def _interpret_flag():
    if jax.default_backend() == "tpu":
        return False
    return pltpu.InterpretParams(detect_races=True)


def _independent_oracle(
    lead_hrv: np.ndarray,
    carried_hrv: np.ndarray,
    own_hrv: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent numpy/JAX reference matching dcp_token_parallel.quantize_rows."""
    num_heads, num_rows, head_dim = own_hrv.shape
    split_rows = lead_hrv.shape[1]
    lead = lead_hrv if rank == 0 else carried_hrv
    shifted_hrv = np.concatenate([lead, own_hrv[:, : num_rows - split_rows, :]], axis=1)
    shifted_2d = jnp.asarray(
        np.swapaxes(shifted_hrv, 0, 1).reshape(num_rows, num_heads * head_dim),
        dtype=jnp.bfloat16,
    )
    x_q, scale = tp.quantize_rows(shifted_2d)
    half = (num_heads * head_dim) // 2
    if rank % 2 == 0:
        mine_q, peer_q = x_q[:, :half], x_q[:, half:]
    else:
        mine_q, peer_q = x_q[:, half:], x_q[:, :half]
    return np.asarray(mine_q), np.asarray(peer_q), np.asarray(scale)


def test_should_select_largest_fitting_row_block_and_reject_invalid_shapes():
    assert sq3d.select_row_block(64, 1024, 64, 256) == 64
    assert sq3d.select_row_block(64, 128, 64, 256) == 64
    assert sq3d.select_row_block(64, 96, 32, 256) == 32
    assert sq3d.select_row_block(64, 64, 64, 256) == 32
    # Tight VMEM budget steps 64 -> 32 -> None cleanly.
    assert (
        sq3d.select_row_block(64, 1024, 64, 256, vmem_limit_bytes=7 * 1024 * 1024) == 32
    )
    assert (
        sq3d.select_row_block(64, 1024, 64, 256, vmem_limit_bytes=4 * 1024 * 1024)
        is None
    )
    # Invalid bounds / alignment rejected.
    assert sq3d.select_row_block(64, 64, 0, 256) is None
    assert sq3d.select_row_block(64, 32, 32, 256) is None
    assert sq3d.select_row_block(64, 64, 96, 256) is None
    assert sq3d.select_row_block(64, 100, 32, 256) is None

    assert sq3d.supports_shift_quant_split_3d(
        64, 1024, 64, 256, jnp.bfloat16, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d(
        64, 1024, 64, 256, jnp.float32, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d(
        3, 128, 64, 256, jnp.bfloat16, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d(
        4, 128, 64, 64, jnp.bfloat16, allow_non_tpu=True
    )
    if jax.default_backend() != "tpu":
        assert not sq3d.supports_shift_quant_split_3d(64, 1024, 64, 256, jnp.bfloat16)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("num_heads", "num_rows", "split_rows", "head_dim"),
    [
        (4, 128, 64, 128),  # BM=64 path
        (4, 96, 32, 128),  # BM=32 path
        (4, 64, 64, 128),  # split_rows == num_rows boundary (BM=32, 2 blocks)
    ],
)
def test_should_match_quantize_rows_bit_exact_across_ranks_and_blocks(
    rank: int,
    num_heads: int,
    num_rows: int,
    split_rows: int,
    head_dim: int,
):
    rng = np.random.default_rng(100 + rank * 31 + num_rows)
    # Asymmetric per-head scales catch any mine/peer half swap or head permutation.
    head_scales = np.linspace(0.5, 4.0, num_heads, dtype=np.float32)[:, None, None]
    lead_f32 = rng.standard_normal((num_heads, split_rows, head_dim)).astype(np.float32)
    carried_f32 = rng.standard_normal((num_heads, split_rows, head_dim)).astype(
        np.float32
    )
    own_f32 = rng.standard_normal((num_heads, num_rows, head_dim)).astype(np.float32)
    lead_f32 *= head_scales
    carried_f32 *= head_scales * 1.7
    own_f32 *= head_scales * 0.8

    # Anti-cancellation sentinels on ignored buffers and shifted-out tail rows.
    if rank == 0:
        carried_f32[...] = SENTINEL
    else:
        lead_f32[...] = SENTINEL
    own_f32[:, num_rows - split_rows :, :] = -SENTINEL

    # Exercise exact zero rows (scale -> 1.0) and large values (> FP8_MAX).
    lead_f32[:, 0, :] = 0.0
    carried_f32[:, 0, :] = 0.0
    if num_rows > split_rows:
        own_f32[:, 0, :] = 0.0
        own_f32[0, 1, 0] = 600.0
        own_f32[num_heads - 1, 1, head_dim - 1] = -750.0

    lead_bf16 = jnp.asarray(lead_f32, dtype=jnp.bfloat16)
    carried_bf16 = jnp.asarray(carried_f32, dtype=jnp.bfloat16)
    own_bf16 = jnp.asarray(own_f32, dtype=jnp.bfloat16)
    rank_arr = jnp.asarray([rank], dtype=jnp.int32)

    want_mine, want_peer, want_scale = _independent_oracle(
        np.asarray(lead_bf16.astype(jnp.float32)),
        np.asarray(carried_bf16.astype(jnp.float32)),
        np.asarray(own_bf16.astype(jnp.float32)),
        rank,
    )
    ref_mine, ref_peer, ref_scale = sq3d.shift_quant_split_3d_ref(
        lead_bf16, carried_bf16, own_bf16, rank_arr
    )
    got_mine, got_peer, got_scale = sq3d.shift_quant_split_3d_fp8(
        lead_bf16,
        carried_bf16,
        own_bf16,
        rank_arr,
        interpret=_interpret_flag(),
    )

    for mine, peer, scale in (
        (np.asarray(ref_mine), np.asarray(ref_peer), np.asarray(ref_scale)),
        (np.asarray(got_mine), np.asarray(got_peer), np.asarray(got_scale)),
    ):
        np.testing.assert_array_equal(mine.view(np.uint8), want_mine.view(np.uint8))
        np.testing.assert_array_equal(peer.view(np.uint8), want_peer.view(np.uint8))
        np.testing.assert_array_equal(scale.view(np.uint32), want_scale.view(np.uint32))


def test_should_reject_invalid_inputs_in_shift_quant_split_3d_fp8():
    good_lead = jnp.zeros((4, 64, 128), jnp.bfloat16)
    good_own = jnp.zeros((4, 128, 128), jnp.bfloat16)
    rk = jnp.asarray([0], jnp.int32)

    with pytest.raises(ValueError, match="3D"):
        sq3d.shift_quant_split_3d_fp8(
            jnp.zeros((64, 512), jnp.bfloat16), good_lead, good_own, rk
        )
    with pytest.raises(ValueError, match="match carried_hrv"):
        sq3d.shift_quant_split_3d_fp8(
            good_lead, jnp.zeros((4, 32, 128), jnp.bfloat16), good_own, rk
        )
    with pytest.raises(ValueError, match="incompatible"):
        sq3d.shift_quant_split_3d_fp8(
            jnp.zeros((2, 64, 128), jnp.bfloat16),
            jnp.zeros((2, 64, 128), jnp.bfloat16),
            good_own,
            rk,
        )
    with pytest.raises(ValueError, match="bfloat16"):
        sq3d.shift_quant_split_3d_fp8(
            good_lead.astype(jnp.float32),
            good_lead.astype(jnp.float32),
            good_own.astype(jnp.float32),
            rk,
        )
    with pytest.raises(ValueError, match="even"):
        sq3d.shift_quant_split_3d_fp8(
            jnp.zeros((3, 64, 128), jnp.bfloat16),
            jnp.zeros((3, 64, 128), jnp.bfloat16),
            jnp.zeros((3, 128, 128), jnp.bfloat16),
            rk,
        )
    with pytest.raises(ValueError, match="128"):
        sq3d.shift_quant_split_3d_fp8(
            jnp.zeros((4, 64, 64), jnp.bfloat16),
            jnp.zeros((4, 64, 64), jnp.bfloat16),
            jnp.zeros((4, 128, 64), jnp.bfloat16),
            rk,
        )
    with pytest.raises(ValueError, match="Unsupported"):
        sq3d.shift_quant_split_3d_fp8(
            jnp.zeros((4, 16, 128), jnp.bfloat16),
            jnp.zeros((4, 16, 128), jnp.bfloat16),
            jnp.zeros((4, 128, 128), jnp.bfloat16),
            rk,
        )


def test_should_match_serial_oproj_bit_exact_with_3d_epilogue_on_4_devices():
    dcp = 4
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    mesh = Mesh(np.array(jax.devices()[:dcp]), (AXIS,))
    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())

    num_heads, head_dim, lkv, hidden = 4, 128, 16, 32
    split_rows, rows = 32, 64
    num_tokens = dcp * rows
    half_cols = (num_heads * head_dim) // 2
    layout = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=4,
        split_rows=split_rows,
        max_pages=2,
    )

    rng = np.random.default_rng(707)
    nope_cache = jax.device_put(jnp.zeros((dcp * 2, 4, lkv), jnp.bfloat16), shd)
    rope_cache = jax.device_put(jnp.zeros((dcp * 2, 4, 8), jnp.bfloat16), shd)
    q = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_tokens, num_heads, lkv)) * 0.25, jnp.bfloat16
        ),
        shd,
    )
    flat_topk = jax.device_put(jnp.zeros((num_tokens, 4), jnp.int32), shd)
    seq_lens = jax.device_put(jnp.asarray([4], jnp.int32), rep)
    block_tables = jax.device_put(jnp.asarray([0, 1], jnp.int32), rep)
    qsl = jax.device_put(jnp.asarray([0, num_tokens], jnp.int32), rep)
    dist = jax.device_put(jnp.asarray([0, 0, 1], jnp.int32), rep)
    lead_part = jax.device_put(
        jnp.asarray(
            rng.standard_normal((dcp * split_rows, num_heads, lkv)) * 0.25,
            jnp.bfloat16,
        ),
        shd,
    )
    lead_lse = jax.device_put(
        jnp.asarray(
            rng.standard_normal((dcp * split_rows, num_heads)) * 0.5, jnp.float32
        ),
        shd,
    )
    w_uv = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_heads, lkv, head_dim)) * 0.1, jnp.bfloat16
        ),
        rep,
    )
    w_o_full = (
        rng.standard_normal((num_heads * head_dim, hidden)).astype(np.float32) * 0.5
    )
    w_o_fp8 = jnp.asarray(w_o_full, jnp.float8_e4m3fn)
    w_o_halves = jnp.concatenate(
        [
            w_o_fp8[:half_cols] if r % 2 == 0 else w_o_fp8[half_cols:]
            for r in range(dcp)
        ],
        axis=0,
    )
    w_o_shd = jax.device_put(w_o_halves, shd)
    s_o_vec = jnp.asarray(rng.uniform(0.01, 0.05, size=(hidden,)), jnp.float32)
    s_o_shd = jax.device_put(jnp.tile(s_o_vec, (dcp,)), shd)
    rank_ids = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    def _attend(q_tok, g_nope, g_rope, flat, n_rows):
        del g_nope, g_rope, flat, n_rows
        return q_tok

    _, _, own_v, lead_v = tp.sparse_mla_attention_with_lead(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        None,
        None,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        lead_partial=lead_part,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=True,
    )
    want = tp.project_sequence_rows_from_v(
        layout, mesh, AXIS, lead_v, own_v, rank_ids, w_o_shd, s_o_shd
    )

    for epilogue_3d in (False, True):
        _, _, got = tp.sparse_mla_attention_with_lead_oproj(
            layout,
            mesh,
            AXIS,
            nope_cache,
            rope_cache,
            q,
            None,
            None,
            flat_topk,
            seq_lens,
            block_tables,
            qsl,
            dist,
            lead_partial=lead_part,
            lead_lse=lead_lse,
            write_kv=None,
            attend=_attend,
            w_uv=w_uv,
            w_o=w_o_shd,
            w_o_scale=s_o_shd,
            rank_ids=rank_ids,
            q_own_rows=True,
            w_uv_replicated=True,
            num_tokens=num_tokens,
            flat_topk_cap_strided=True,
            epilogue_3d=epilogue_3d,
        )
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
        )

    # Also verify the 3D head-major shift+quant+dot flow end-to-end across ranks
    # on CPU (using shift_quant_split_3d_ref directly in a 4-device shard_map).
    def _hrv_ref_fn(own_rhv, lead_rhv, rk_a, w_loc, s_loc):
        own_hrv = jnp.swapaxes(own_rhv, 0, 1)
        lead_hrv = jnp.swapaxes(lead_rhv, 0, 1)
        perm_shift = [(i, i + 1) for i in range(dcp - 1)]
        pairs = [(i, i ^ 1) for i in range(dcp)]
        carried_hrv = jax.lax.ppermute(
            own_hrv[:, rows - split_rows :, :], AXIS, perm_shift
        )
        w_peer = jax.lax.ppermute(w_loc, AXIS, pairs)
        mine, peer, x_sc = sq3d.shift_quant_split_3d_ref(
            lead_hrv, carried_hrv, own_hrv, rk_a
        )
        dims = (((1,), (0,)), ((), ()))
        y = jax.lax.dot_general(
            mine, w_loc, dims, preferred_element_type=jnp.float32
        ) + jax.lax.dot_general(peer, w_peer, dims, preferred_element_type=jnp.float32)
        return (y * x_sc * s_loc.astype(jnp.float32)[None, :]).astype(own_hrv.dtype)

    got_3d_ref = jax.shard_map(
        _hrv_ref_fn,
        mesh=mesh,
        in_specs=(P(AXIS), P(), P(AXIS), P(AXIS), P(AXIS)),
        out_specs=P(AXIS),
        check_vma=False,
    )(own_v, lead_v, rank_ids, w_o_shd, s_o_shd)
    np.testing.assert_array_equal(
        np.asarray(got_3d_ref).view(np.uint16), np.asarray(want).view(np.uint16)
    )


def test_should_fallback_cleanly_for_bf16_pair_halves_and_reject_partial_oproj_args():
    dcp = 4
    if len(jax.devices()) < dcp:
        pytest.skip(f"needs {dcp} devices")
    mesh = Mesh(np.array(jax.devices()[:dcp]), (AXIS,))
    shd = NamedSharding(mesh, P(AXIS))
    rep = NamedSharding(mesh, P())

    num_heads, head_dim, lkv, hidden = 4, 128, 16, 32
    split_rows, rows = 32, 64
    num_tokens = dcp * rows
    half_cols = (num_heads * head_dim) // 2
    layout = tp.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=1,
        page_size=4,
        split_rows=split_rows,
        max_pages=2,
    )

    rng = np.random.default_rng(808)
    nope_cache = jax.device_put(jnp.zeros((dcp * 2, 4, lkv), jnp.bfloat16), shd)
    rope_cache = jax.device_put(jnp.zeros((dcp * 2, 4, 8), jnp.bfloat16), shd)
    q = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_tokens, num_heads, lkv)) * 0.25, jnp.bfloat16
        ),
        shd,
    )
    flat_topk = jax.device_put(jnp.zeros((num_tokens, 4), jnp.int32), shd)
    seq_lens = jax.device_put(jnp.asarray([4], jnp.int32), rep)
    block_tables = jax.device_put(jnp.asarray([0, 1], jnp.int32), rep)
    qsl = jax.device_put(jnp.asarray([0, num_tokens], jnp.int32), rep)
    dist = jax.device_put(jnp.asarray([0, 0, 1], jnp.int32), rep)
    lead_part = jax.device_put(
        jnp.asarray(
            rng.standard_normal((dcp * split_rows, num_heads, lkv)) * 0.25,
            jnp.bfloat16,
        ),
        shd,
    )
    lead_lse = jax.device_put(
        jnp.asarray(
            rng.standard_normal((dcp * split_rows, num_heads)) * 0.5, jnp.float32
        ),
        shd,
    )
    w_uv = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_heads, lkv, head_dim)) * 0.1, jnp.bfloat16
        ),
        rep,
    )
    w_o_fp8 = jnp.asarray(
        rng.standard_normal((num_heads * head_dim, hidden)) * 0.5, jnp.float8_e4m3fn
    )
    w_o_halves = jnp.concatenate(
        [
            w_o_fp8[:half_cols] if r % 2 == 0 else w_o_fp8[half_cols:]
            for r in range(dcp)
        ],
        axis=0,
    )
    w_o_shd = jax.device_put(w_o_halves, shd)
    # 2D per-shard scale exercises the non-FP8-pair-halves fallback path cleanly.
    s_o_2d = jax.device_put(
        jnp.asarray(
            rng.uniform(0.01, 0.05, size=(dcp * (dcp // 2), hidden)), jnp.float32
        ),
        shd,
    )
    rank_ids = jax.device_put(jnp.arange(dcp, dtype=jnp.int32), shd)

    def _attend(q_tok, g_nope, g_rope, flat, n_rows):
        del g_nope, g_rope, flat, n_rows
        return q_tok

    _, _, own_v, lead_v = tp.sparse_mla_attention_with_lead(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        None,
        None,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        lead_partial=lead_part,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_attend,
        w_uv=w_uv,
        rank_ids=rank_ids,
        q_own_rows=True,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=True,
    )
    want = tp.project_sequence_rows_from_v(
        layout, mesh, AXIS, lead_v, own_v, rank_ids, w_o_shd, s_o_2d
    )
    _, _, got = tp.sparse_mla_attention_with_lead_oproj(
        layout,
        mesh,
        AXIS,
        nope_cache,
        rope_cache,
        q,
        None,
        None,
        flat_topk,
        seq_lens,
        block_tables,
        qsl,
        dist,
        lead_partial=lead_part,
        lead_lse=lead_lse,
        write_kv=None,
        attend=_attend,
        w_uv=w_uv,
        w_o=w_o_shd,
        w_o_scale=s_o_2d,
        rank_ids=rank_ids,
        q_own_rows=True,
        w_uv_replicated=True,
        num_tokens=num_tokens,
        flat_topk_cap_strided=True,
        epilogue_3d=True,
    )
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )

    with pytest.raises(ValueError, match="together"):
        tp.sparse_mla_attention_with_lead_oproj(
            layout,
            mesh,
            AXIS,
            nope_cache,
            rope_cache,
            q,
            None,
            None,
            flat_topk,
            seq_lens,
            block_tables,
            qsl,
            dist,
            lead_partial=lead_part,
            lead_lse=lead_lse,
            write_kv=None,
            attend=_attend,
            w_uv=w_uv,
            w_o=w_o_shd,
            w_o_scale=None,  # type: ignore[arg-type] -- intentional guard test
            num_tokens=num_tokens,
        )


def test_should_validate_supported_shapes_for_remote_shift_quant_split_3d():
    assert sq3d.supports_shift_quant_split_3d_remote(
        64, 1024, 64, 256, jnp.bfloat16, 16, allow_non_tpu=True
    )
    assert sq3d.supports_shift_quant_split_3d_remote(
        4, 192, 64, 128, jnp.bfloat16, 2, allow_non_tpu=True
    )
    assert sq3d.supports_shift_quant_split_3d_remote(
        4, 96, 32, 128, jnp.bfloat16, 4, allow_non_tpu=True
    )
    # Reject dcp_size<2, <3 row blocks, unsupported split_rows, tight VMEM.
    assert not sq3d.supports_shift_quant_split_3d_remote(
        64, 1024, 64, 256, jnp.bfloat16, 1, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        4, 128, 64, 128, jnp.bfloat16, 4, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        4, 288, 96, 128, jnp.bfloat16, 4, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        64,
        1024,
        64,
        256,
        jnp.bfloat16,
        16,
        vmem_limit_bytes=7 * 1024 * 1024,
        allow_non_tpu=True,
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        4, 192, 64, 128, jnp.float32, 4, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        3, 192, 64, 128, jnp.bfloat16, 4, allow_non_tpu=True
    )
    assert not sq3d.supports_shift_quant_split_3d_remote(
        4, 192, 64, 64, jnp.bfloat16, 4, allow_non_tpu=True
    )
    if jax.default_backend() != "tpu":
        assert not sq3d.supports_shift_quant_split_3d_remote(
            64, 1024, 64, 256, jnp.bfloat16, 16
        )


def test_should_reject_invalid_inputs_in_shift_quant_split_3d_fp8_remote():
    good_lead = jnp.zeros((4, 64, 128), jnp.bfloat16)
    good_own = jnp.zeros((4, 192, 128), jnp.bfloat16)
    rk = jnp.asarray([0], jnp.int32)

    with pytest.raises(ValueError, match="3D"):
        sq3d.shift_quant_split_3d_fp8_remote(
            jnp.zeros((64, 512), jnp.bfloat16), good_own, rk, dcp_size=4, axis_name=AXIS
        )
    with pytest.raises(ValueError, match="incompatible"):
        sq3d.shift_quant_split_3d_fp8_remote(
            jnp.zeros((2, 64, 128), jnp.bfloat16),
            good_own,
            rk,
            dcp_size=4,
            axis_name=AXIS,
        )
    with pytest.raises(ValueError, match="bfloat16"):
        sq3d.shift_quant_split_3d_fp8_remote(
            good_lead.astype(jnp.float32), good_own, rk, dcp_size=4, axis_name=AXIS
        )
    with pytest.raises(ValueError, match="dcp_size"):
        sq3d.shift_quant_split_3d_fp8_remote(
            good_lead, good_own, rk, dcp_size=1, axis_name=AXIS
        )
    with pytest.raises(ValueError, match="Unsupported remote"):
        sq3d.shift_quant_split_3d_fp8_remote(
            good_lead,
            jnp.zeros((4, 128, 128), jnp.bfloat16),
            rk,
            dcp_size=4,
            axis_name=AXIS,
        )


@pytest.mark.parametrize(
    ("num_rows", "split_rows"),
    [
        (192, 64),  # minimal n_body == 2 with BM=64
        (256, 64),  # n_body == 3 with BM=64 (exercises drain of b-2)
        (96, 32),  # minimal n_body == 2 with BM=32
    ],
)
def test_should_match_ref_bit_exact_with_remote_push_and_race_detection(
    num_rows: int, split_rows: int
):
    if jax.device_count() < 4:
        pytest.skip("requires 4 JAX devices for multi-rank remote shift_quant test")
    dcp_size = 4
    num_heads = 4
    head_dim = 128
    mesh = Mesh(np.array(jax.devices()[:dcp_size]), (AXIS,))
    sh_dcp = NamedSharding(mesh, P(AXIS))
    sh_rep = NamedSharding(mesh, P())

    rng = np.random.default_rng(700 + num_rows * 13 + split_rows)
    head_scales = np.linspace(0.5, 4.0, num_heads, dtype=np.float32)[:, None, None]
    # Run two distinct consecutive batches in one shard_map to test barrier reuse.
    owns, leads = [], []
    for batch_idx in range(2):
        own_f32 = (
            rng.standard_normal((dcp_size, num_heads, num_rows, head_dim)).astype(
                np.float32
            )
            * head_scales[None, ...]
            * (0.7 + 0.5 * batch_idx)
        )
        lead_f32 = (
            rng.standard_normal((num_heads, split_rows, head_dim)).astype(np.float32)
            * head_scales
        )
        # Anti-cancellation sentinel on rank (dcp_size - 1) shifted-out tail rows.
        own_f32[dcp_size - 1, :, num_rows - split_rows :, :] = -SENTINEL
        # Zero rows (scale -> 1.0) and large values (> FP8_MAX clipping).
        lead_f32[:, 0, :] = 0.0
        own_f32[:, :, 0, :] = 0.0
        own_f32[1, 0, num_rows - split_rows + 1, 0] = 650.0
        own_f32[2, num_heads - 1, 1, head_dim - 1] = -780.0
        owns.append(
            jax.device_put(
                jnp.asarray(
                    own_f32.reshape(dcp_size * num_heads, num_rows, head_dim),
                    jnp.bfloat16,
                ),
                sh_dcp,
            )
        )
        leads.append(jax.device_put(jnp.asarray(lead_f32, jnp.bfloat16), sh_rep))
    rank_ids = jax.device_put(jnp.arange(dcp_size, dtype=jnp.int32), sh_dcp)
    perm_shift = [(i, i + 1) for i in range(dcp_size - 1)]

    def _ref(l0, o0, l1, o1, rk):
        c0 = jax.lax.ppermute(o0[:, num_rows - split_rows :, :], AXIS, perm_shift)
        c1 = jax.lax.ppermute(o1[:, num_rows - split_rows :, :], AXIS, perm_shift)
        return (
            *sq3d.shift_quant_split_3d_ref(l0, c0, o0, rk),
            *sq3d.shift_quant_split_3d_ref(l1, c1, o1, rk),
        )

    def _got(l0, o0, l1, o1, rk):
        kw = dict(dcp_size=dcp_size, axis_name=AXIS, interpret=_interpret_flag())
        return (
            *sq3d.shift_quant_split_3d_fp8_remote(l0, o0, rk, **kw),
            *sq3d.shift_quant_split_3d_fp8_remote(l1, o1, rk, **kw),
        )

    ref_out = jax.shard_map(
        _ref,
        mesh=mesh,
        in_specs=(P(), P(AXIS), P(), P(AXIS), P(AXIS)),
        out_specs=(P(AXIS),) * 6,
        check_vma=False,
    )(leads[0], owns[0], leads[1], owns[1], rank_ids)
    got_out = jax.shard_map(
        _got,
        mesh=mesh,
        in_specs=(P(), P(AXIS), P(), P(AXIS), P(AXIS)),
        out_specs=(P(AXIS),) * 6,
        check_vma=False,
    )(leads[0], owns[0], leads[1], owns[1], rank_ids)

    for idx in (0, 1, 3, 4):
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(got_out[idx], jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(ref_out[idx], jnp.uint8)),
        )
    for idx in (2, 5):
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(got_out[idx], jnp.uint32)),
            np.asarray(jax.lax.bitcast_convert_type(ref_out[idx], jnp.uint32)),
        )
