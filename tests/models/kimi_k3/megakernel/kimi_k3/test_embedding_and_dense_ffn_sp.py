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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    embedding_and_dense_ffn_sp as _emb_sp,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.helpers import (
    expert_activation,
)

embed_deconflicted_local_rows = _emb_sp.embed_deconflicted_local_rows
half_ring_streams = _emb_sp.half_ring_streams
pipelined_dense_ffn_sp_pallas = _emb_sp.pipelined_dense_ffn_sp_pallas
rank_half_reduce_bf16_fast_pallas = _emb_sp.rank_half_reduce_bf16_fast_pallas
select_dense_ffn_compute_tiles = _emb_sp.select_dense_ffn_compute_tiles
select_half_reduce_chunk_rows = _emb_sp.select_half_reduce_chunk_rows
sp_token_index_for_ranks = _emb_sp.sp_token_index_for_ranks


def _synthetic_ring_tables(
    num_ranks: int, *, token_ranks: tuple[int, ...] | None = None
) -> np.ndarray:
    owners = (
        np.arange(num_ranks, dtype=np.int32)
        if token_ranks is None
        else np.asarray(token_ranks, dtype=np.int32)
    )
    base = np.arange(num_ranks, dtype=np.int32)
    rings = (
        base,
        np.roll(base, 1),
        base[::-1],
    )
    tables = np.zeros((3, 3, num_ranks), dtype=np.int32)
    for family, order in enumerate(rings):
        tables[family, 1, :] = order
        for pos, rk in enumerate(order):
            tables[family, 0, int(rk)] = pos
        tables[family, 2, :] = owners
    return tables


def _reference_embedding_sp(
    tokens_by_rank: np.ndarray,
    tables_by_rank: np.ndarray,
    *,
    length: int,
    vocab_per_rank: int,
) -> np.ndarray:
    num_ranks, local_tokens = tokens_by_rank.shape
    stripe = min(64, local_tokens)
    flat_tokens = tokens_by_rank.reshape(-1)
    full_vocab = tables_by_rank.reshape(num_ranks * vocab_per_rank, -1)
    out = np.zeros((num_ranks, local_tokens, full_vocab.shape[1]), dtype=np.float32)
    for r in range(num_ranks):
        for loc in range(local_tokens):
            win_pos = (loc // stripe) * num_ranks * stripe + r * stripe + (loc % stripe)
            tok = int(flat_tokens[r * local_tokens + loc])
            if win_pos < length and 0 <= tok < full_vocab.shape[0]:
                out[r, loc] = full_vocab[tok]
    return np.asarray(jnp.asarray(out, jnp.bfloat16).astype(jnp.float32))


def _reference_dense_ffn_sp(
    hidden_by_rank: np.ndarray,
    prefix_by_rank: np.ndarray,
    gate_up_by_rank: np.ndarray,
    down_by_rank: np.ndarray,
    *,
    length: int,
    token_ranks: tuple[int, ...],
) -> np.ndarray:
    num_ranks, rows, width = hidden_by_rank.shape
    stripe = min(64, rows)
    blocks = rows // stripe
    out = np.zeros((num_ranks, rows, width), dtype=np.float32)
    for blk in range(blocks):
        owner_ordered = np.zeros((num_ranks * stripe, width), dtype=np.float32)
        for rk in range(num_ranks):
            owner = token_ranks[rk]
            owner_ordered[owner * stripe : (owner + 1) * stripe] = hidden_by_rank[
                rk, blk * stripe : (blk + 1) * stripe
            ]
        x_bf16 = jnp.asarray(owner_ordered, jnp.bfloat16)
        partial_sum_f32 = np.zeros((num_ranks * stripe, width), dtype=np.float32)
        for rk in range(num_ranks):
            gu_w = jnp.asarray(gate_up_by_rank[rk], jnp.bfloat16)
            dn_w = jnp.asarray(down_by_rank[rk], jnp.bfloat16)
            gu = jnp.dot(x_bf16, gu_w, preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            gate, up = jnp.split(gu.astype(jnp.float32), 2, axis=1)
            mid = expert_activation(gate, up).astype(jnp.bfloat16)
            part_bf16 = jnp.dot(mid, dn_w, preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            partial_sum_f32 += np.asarray(part_bf16.astype(jnp.float32))
        for rk in range(num_ranks):
            owner = token_ranks[rk]
            red_slice = partial_sum_f32[owner * stripe : (owner + 1) * stripe]
            px_slice = prefix_by_rank[rk, blk * stripe : (blk + 1) * stripe]
            for s in range(stripe):
                win_tok = blk * num_ranks * stripe + owner * stripe + s
                row_f32 = px_slice[s] + (red_slice[s] if win_tok < length else 0.0)
                out[rk, blk * stripe + s] = np.asarray(
                    jnp.asarray(row_f32, jnp.bfloat16).astype(jnp.float32)
                )
    return out


@pytest.mark.parametrize(
    ("rows", "expected_chunk"),
    [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (192, 192),
        (256, 256),
        (320, 160),
        (384, 192),
        (448, 224),
        (512, 256),
    ],
)
def test_should_select_exact_divisor_chunk_across_all_ten_serving_windows(
    rows: int, expected_chunk: int
):
    chunk = select_half_reduce_chunk_rows(rows)
    assert chunk == expected_chunk
    assert rows % chunk == 0
    assert chunk % 16 == 0


def test_should_reject_invalid_shapes_and_unaligned_parameters():
    with pytest.raises(ValueError, match="width >= 384"):
        half_ring_streams(256)
    with pytest.raises(ValueError, match="rows >= 16"):
        select_half_reduce_chunk_rows(8)
    with pytest.raises(ValueError, match="No valid 16-aligned chunk"):
        select_half_reduce_chunk_rows(16, chunk_cap=8)
    with pytest.raises(ValueError, match="Cannot partition"):
        select_dense_ffn_compute_tiles(16, max_tiles=0)

    tbl4 = _synthetic_ring_tables(4)
    x_f32 = jnp.zeros((64, 512), jnp.float32)
    x_bf16 = jnp.zeros((64, 512), jnp.bfloat16)
    with pytest.raises(ValueError, match="even num_ranks >= 4"):
        rank_half_reduce_bf16_fast_pallas(x_bf16, tbl4, num_ranks=2)
    with pytest.raises(ValueError, match="BF16 partial_local"):
        rank_half_reduce_bf16_fast_pallas(x_f32, tbl4, num_ranks=4)
    with pytest.raises(ValueError, match="divisible by num_ranks"):
        rank_half_reduce_bf16_fast_pallas(
            jnp.zeros((66, 512), jnp.bfloat16), tbl4, num_ranks=4
        )

    px = jnp.zeros((16, 512), jnp.bfloat16)
    gu = jnp.zeros((512, 256), jnp.bfloat16)
    dn = jnp.zeros((128, 512), jnp.bfloat16)
    ln = jnp.asarray([32], jnp.int32)
    with pytest.raises(ValueError, match="even num_ranks >= 4"):
        pipelined_dense_ffn_sp_pallas(px, px, gu, dn, ln, tbl4, num_ranks=2)
    with pytest.raises(ValueError, match="do not match"):
        pipelined_dense_ffn_sp_pallas(
            px, jnp.zeros((16, 256), jnp.bfloat16), gu, dn, ln, tbl4, num_ranks=4
        )
    with pytest.raises(ValueError, match="16-aligned rows"):
        pipelined_dense_ffn_sp_pallas(
            jnp.zeros((8, 512), jnp.bfloat16),
            jnp.zeros((8, 512), jnp.bfloat16),
            gu,
            dn,
            ln,
            tbl4,
            num_ranks=4,
        )


@pytest.mark.parametrize("local_tokens", [16, 32, 128])
def test_should_match_clamped_gather_bitwise_when_deconflicting_embedding_rows(
    local_tokens: int,
):
    num_ranks = 4
    vocab_per_rank = 64
    width = 128
    seq_len = num_ranks * local_tokens
    length_val = seq_len - 20
    rng = np.random.default_rng(4830 + local_tokens)
    tables = rng.standard_normal((num_ranks, vocab_per_rank, width)).astype(np.float32)
    win_tokens = rng.integers(
        0, num_ranks * vocab_per_rank, size=seq_len, dtype=np.int32
    )
    win_tokens[3:9] = -1
    win_tokens[12:20] = 0
    win_tokens[length_val:] = 0
    tables[0, 0, :] = 999.0

    stripe = min(64, local_tokens)
    order = (
        np.arange(seq_len, dtype=np.int32)
        .reshape(-1, num_ranks, stripe)
        .transpose(1, 0, 2)
        .reshape(num_ranks, local_tokens)
    )
    tokens_by_rank = win_tokens[order]
    flat_token_ids = jnp.asarray(tokens_by_rank.reshape(seq_len), jnp.int32)
    length_arr = jnp.asarray([length_val], jnp.int32)

    reduced = np.zeros((num_ranks, local_tokens, width), dtype=np.float32)
    for rk in range(num_ranks):
        part = embed_deconflicted_local_rows(
            flat_token_ids,
            jnp.asarray(tables[rk], jnp.bfloat16),
            length_arr,
            rank=jnp.asarray(rk, jnp.int32),
            num_ranks=num_ranks,
            local_tokens=local_tokens,
            vocab_per_rank=vocab_per_rank,
            reduce_dtype=jnp.bfloat16,
        )
        reduced += np.asarray(part.astype(jnp.float32)).reshape(
            num_ranks, local_tokens, width
        )
    got = np.asarray(jnp.asarray(reduced, jnp.bfloat16).astype(jnp.float32))
    ref = _reference_embedding_sp(
        tokens_by_rank, tables, length=length_val, vocab_per_rank=vocab_per_rank
    )
    np.testing.assert_array_equal(got, ref)
    assert sp_token_index_for_ranks(1, 0, stripe, num_ranks=num_ranks) == stripe

    bad_ref = _reference_embedding_sp(
        tokens_by_rank, tables, length=seq_len, vocab_per_rank=vocab_per_rank
    )
    assert float(np.max(np.abs(got - bad_ref))) > 500.0


@pytest.mark.parametrize("rows", [16, 64])
def test_should_reduce_scatter_bf16_partials_accurately_with_permuted_owners_on_4_ranks(
    rows: int,
):
    num_ranks = 4
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip("requires >= 4 host JAX devices")
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    width = 384
    token_ranks = (2, 0, 3, 1)
    tables = _synthetic_ring_tables(num_ranks, token_ranks=token_ranks)
    rng = np.random.default_rng(5100 + rows)
    partials_np = (
        rng.standard_normal((num_ranks, num_ranks * rows, width)).astype(np.float32)
        * 0.25
    )
    partials_bf16 = jnp.asarray(partials_np, jnp.bfloat16)

    run_rs = jax.jit(
        jax.shard_map(
            lambda p: rank_half_reduce_bf16_fast_pallas(
                p[0],
                tables,
                num_ranks=num_ranks,
                axis_name="k3_sp",
                collective_id=61 if rows == 16 else 62,
                chunk_cap=32,
                interpret=True,
            )[None],
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )
    got = np.asarray(run_rs(partials_bf16).astype(jnp.float32))

    partials_q = np.asarray(partials_bf16.astype(jnp.float32))
    expected = np.zeros((num_ranks, rows, width), dtype=np.float32)
    for rk in range(num_ranks):
        owner = token_ranks[rk]
        summed = np.sum(partials_q[:, owner * rows : (owner + 1) * rows, :], axis=0)
        expected[rk] = np.asarray(jnp.asarray(summed, jnp.bfloat16).astype(jnp.float32))

    rel_l2 = float(np.linalg.norm(got - expected) / np.linalg.norm(expected))
    assert rel_l2 < 3.5e-3

    onehot_np = np.zeros((num_ranks, num_ranks * rows, width), dtype=np.float32)
    for r_row in range(num_ranks * rows):
        active_rk = (r_row * 3 + 1) % num_ranks
        onehot_np[active_rk, r_row] = partials_np[active_rk, r_row]
    onehot_bf16 = jnp.asarray(onehot_np, jnp.bfloat16)
    run_rs_onehot = jax.jit(
        jax.shard_map(
            lambda p: rank_half_reduce_bf16_fast_pallas(
                p[0],
                tables,
                num_ranks=num_ranks,
                axis_name="k3_sp",
                collective_id=71 if rows == 16 else 72,
                chunk_cap=32,
                interpret=True,
            )[None],
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )
    got_onehot = np.asarray(run_rs_onehot(onehot_bf16).astype(jnp.float32))
    onehot_q = np.asarray(onehot_bf16.astype(jnp.float32))
    exp_onehot = np.stack(
        [
            np.sum(
                onehot_q[:, token_ranks[rk] * rows : (token_ranks[rk] + 1) * rows, :],
                axis=0,
            )
            for rk in range(num_ranks)
        ],
        axis=0,
    )
    np.testing.assert_array_equal(got_onehot, exp_onehot)

    zeroed = partials_q.copy()
    zeroed[0] = 0.0
    bad_expected = np.zeros_like(expected)
    for rk in range(num_ranks):
        owner = token_ranks[rk]
        bad_expected[rk] = np.sum(
            zeroed[:, owner * rows : (owner + 1) * rows, :], axis=0
        )
    assert float(np.linalg.norm(got - bad_expected) / np.linalg.norm(expected)) > 0.15


def test_should_exercise_four_slot_credit_wraparound_when_ring_hops_exceed_four():
    num_ranks = 10
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip("requires >= 10 host JAX devices to exercise >= 5 half-ring hops")
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    rows, width = 16, 384
    tables = _synthetic_ring_tables(num_ranks)
    rng = np.random.default_rng(6100)
    partials_bf16 = jnp.asarray(
        rng.standard_normal((num_ranks, num_ranks * rows, width)) * 0.2, jnp.bfloat16
    )

    run_rs10 = jax.jit(
        jax.shard_map(
            lambda p: rank_half_reduce_bf16_fast_pallas(
                p[0],
                tables,
                num_ranks=num_ranks,
                axis_name="k3_sp",
                collective_id=63,
                interpret=True,
            )[None],
            mesh=mesh,
            in_specs=(P("k3_sp"),),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )
    got = np.asarray(run_rs10(partials_bf16).astype(jnp.float32))
    partials_q = np.asarray(partials_bf16.astype(jnp.float32))
    expected = np.stack(
        [
            np.sum(partials_q[:, r * rows : (r + 1) * rows, :], axis=0)
            for r in range(num_ranks)
        ],
        axis=0,
    )
    rel_l2 = float(np.linalg.norm(got - expected) / np.linalg.norm(expected))
    assert rel_l2 < 4e-3


@pytest.mark.parametrize(("rows", "coll_id"), [(16, 64), (128, 65)])
def test_should_match_dense_ffn_oracle_and_preserve_invalid_tail_across_blocks(
    rows: int, coll_id: int
):
    num_ranks = 4
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip("requires >= 4 host JAX devices")
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    width, mid_w = 512, 128
    token_ranks = (1, 3, 0, 2)
    tables = _synthetic_ring_tables(num_ranks, token_ranks=token_ranks)
    total_tokens = num_ranks * rows
    length_val = total_tokens - 20
    stripe = min(64, rows)
    rng = np.random.default_rng(7100 + rows)

    hidden_np = rng.standard_normal((num_ranks, rows, width)).astype(np.float32) * 0.3
    prefix_np = rng.standard_normal((num_ranks, rows, width)).astype(np.float32) * 0.5
    for rk in range(num_ranks):
        owner = token_ranks[rk]
        for loc in range(rows):
            win_tok = (
                (loc // stripe) * num_ranks * stripe + owner * stripe + (loc % stripe)
            )
            if win_tok >= length_val:
                hidden_np[rk, loc] = 50.0
    gu_np = rng.standard_normal((num_ranks, width, 2 * mid_w)).astype(np.float32) * 0.08
    dn_np = rng.standard_normal((num_ranks, mid_w, width)).astype(np.float32) * 0.08

    hidden_bf16 = jnp.asarray(hidden_np, jnp.bfloat16)
    prefix_bf16 = jnp.asarray(prefix_np, jnp.bfloat16)
    gu_bf16 = jnp.asarray(gu_np, jnp.bfloat16)
    dn_bf16 = jnp.asarray(dn_np, jnp.bfloat16)
    length_arr = jnp.asarray([length_val], jnp.int32)

    run_ffn = jax.jit(
        jax.shard_map(
            lambda x, px, gu, dn, ln: pipelined_dense_ffn_sp_pallas(
                x[0],
                px[0],
                gu[0],
                dn[0],
                ln,
                tables,
                num_ranks=num_ranks,
                axis_name="k3_sp",
                collective_id=coll_id,
                interpret=True,
            )[None],
            mesh=mesh,
            in_specs=(
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P(),
            ),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )
    got = np.asarray(
        run_ffn(hidden_bf16, prefix_bf16, gu_bf16, dn_bf16, length_arr).astype(
            jnp.float32
        )
    )
    ref = _reference_dense_ffn_sp(
        np.asarray(hidden_bf16.astype(jnp.float32)),
        np.asarray(prefix_bf16.astype(jnp.float32)),
        np.asarray(gu_bf16.astype(jnp.float32)),
        np.asarray(dn_bf16.astype(jnp.float32)),
        length=length_val,
        token_ranks=token_ranks,
    )
    rel_l2 = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert rel_l2 < 2.5e-3

    px_q = np.asarray(prefix_bf16.astype(jnp.float32))
    for rk in range(num_ranks):
        owner = token_ranks[rk]
        for loc in range(rows):
            win_tok = (
                (loc // stripe) * num_ranks * stripe + owner * stripe + (loc % stripe)
            )
            if win_tok >= length_val:
                np.testing.assert_array_equal(got[rk, loc], px_q[rk, loc])

    unmasked_ref = _reference_dense_ffn_sp(
        np.asarray(hidden_bf16.astype(jnp.float32)),
        np.asarray(prefix_bf16.astype(jnp.float32)),
        np.asarray(gu_bf16.astype(jnp.float32)),
        np.asarray(dn_bf16.astype(jnp.float32)),
        length=total_tokens,
        token_ranks=token_ranks,
    )
    assert float(np.max(np.abs(got - unmasked_ref))) > 1.0


def test_should_exercise_credit_wraparound_in_pipelined_dense_ffn_on_10_ranks():
    num_ranks = 10
    devs = jax.devices()
    if len(devs) < num_ranks:
        pytest.skip("requires >= 10 host JAX devices")
    mesh = Mesh(np.asarray(devs[:num_ranks]), ("k3_sp",))
    rows, width, mid_w = 16, 512, 128
    tables = _synthetic_ring_tables(num_ranks)
    rng = np.random.default_rng(8100)
    hidden_bf16 = jnp.asarray(
        rng.standard_normal((num_ranks, rows, width)) * 0.25, jnp.bfloat16
    )
    prefix_bf16 = jnp.asarray(
        rng.standard_normal((num_ranks, rows, width)) * 0.25, jnp.bfloat16
    )
    gu_bf16 = jnp.asarray(
        rng.standard_normal((num_ranks, width, 2 * mid_w)) * 0.08, jnp.bfloat16
    )
    dn_bf16 = jnp.asarray(
        rng.standard_normal((num_ranks, mid_w, width)) * 0.08, jnp.bfloat16
    )
    length_arr = jnp.asarray([num_ranks * rows - 8], jnp.int32)

    run_ffn10 = jax.jit(
        jax.shard_map(
            lambda x, px, gu, dn, ln: pipelined_dense_ffn_sp_pallas(
                x[0],
                px[0],
                gu[0],
                dn[0],
                ln,
                tables,
                num_ranks=num_ranks,
                axis_name="k3_sp",
                collective_id=66,
                interpret=True,
            )[None],
            mesh=mesh,
            in_specs=(
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P("k3_sp"),
                P(),
            ),
            out_specs=P("k3_sp"),
            check_vma=False,
        )
    )
    got = np.asarray(
        run_ffn10(hidden_bf16, prefix_bf16, gu_bf16, dn_bf16, length_arr).astype(
            jnp.float32
        )
    )
    ref = _reference_dense_ffn_sp(
        np.asarray(hidden_bf16.astype(jnp.float32)),
        np.asarray(prefix_bf16.astype(jnp.float32)),
        np.asarray(gu_bf16.astype(jnp.float32)),
        np.asarray(dn_bf16.astype(jnp.float32)),
        length=num_ranks * rows - 8,
        token_ranks=tuple(range(num_ranks)),
    )
    assert float(np.linalg.norm(got - ref) / np.linalg.norm(ref)) < 3e-3


def test_should_execute_make_embedding_sp_and_make_dense_ffn_sp_on_32_host_devices():
    devs = jax.devices()
    if len(devs) < 32:
        pytest.skip("requires 32 host JAX devices")
    mesh = Mesh(np.asarray(devs[:32]), ("sp",))
    seq_len = 512
    local_tokens = seq_len // 32
    vocab_per_rank = 64
    rng = np.random.default_rng(9100)

    emb_fn = pm.make_embedding_sp(mesh, seq_len=seq_len, vocab_per_rank=vocab_per_rank)
    tokens = rng.integers(
        -1, 32 * vocab_per_rank, size=(32, local_tokens), dtype=np.int32
    )
    tokens[:, -2:] = 0
    table = rng.standard_normal((32, vocab_per_rank, 7168)).astype(np.float32) * 0.2
    table[0, 0, :] = 777.0
    length_arr = jnp.asarray([seq_len - 32], jnp.int32)
    got_emb = np.asarray(
        emb_fn(
            jnp.asarray(tokens, jnp.int32),
            jnp.asarray(table, jnp.bfloat16),
            length_arr,
        ).astype(jnp.float32)
    )
    ref_emb = _reference_embedding_sp(
        tokens, table, length=seq_len - 32, vocab_per_rank=vocab_per_rank
    )
    np.testing.assert_array_equal(got_emb, ref_emb)

    dense_fn = pm.make_dense_ffn_sp(mesh, seq_len=seq_len)
    x_bf16 = jnp.asarray(
        rng.standard_normal((32, local_tokens, 7168)) * 0.1, jnp.bfloat16
    )
    px_bf16 = jnp.asarray(
        rng.standard_normal((32, local_tokens, 7168)) * 0.1, jnp.bfloat16
    )
    gu_bf16 = jnp.asarray(rng.standard_normal((32, 7168, 2304)) * 0.02, jnp.bfloat16)
    dn_bf16 = jnp.asarray(rng.standard_normal((32, 1152, 7168)) * 0.02, jnp.bfloat16)
    got_dense = np.asarray(
        dense_fn(x_bf16, px_bf16, gu_bf16, dn_bf16, length_arr).astype(jnp.float32)
    )
    ref_dense = _reference_dense_ffn_sp(
        np.asarray(x_bf16.astype(jnp.float32)),
        np.asarray(px_bf16.astype(jnp.float32)),
        np.asarray(gu_bf16.astype(jnp.float32)),
        np.asarray(dn_bf16.astype(jnp.float32)),
        length=seq_len - 32,
        token_ranks=tuple(range(32)),
    )
    rel_l2 = float(np.linalg.norm(got_dense - ref_dense) / np.linalg.norm(ref_dense))
    assert rel_l2 < 3e-3
