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
"""CPU contract tests for the zigzag context-parallel prefill plan.

The attention values need a device; these check the plan's page arithmetic
against the K/V each position must see, the serving-cache writes against the
pages `wrapper.prepare_inputs` lays out, and that the kernel lowers for tpu7x.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, cp_prefill, wrapper

FP8 = jnp.float8_e4m3fn
PAGE = 256
TPU7X = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
LAYOUTS = [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE]


def _kv(num_tokens, kv_heads, head_dim, seed):
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((num_tokens, kv_heads, head_dim), np.float32)
    v = rng.standard_normal((num_tokens, kv_heads, head_dim), np.float32)
    return jnp.asarray(k, FP8), jnp.asarray(v, FP8)


def _rank_rows(x, plan, rank):
    return jnp.concatenate([x[s:e] for s, e in plan.rank_slices(rank)])


def _bytes(x):
    return np.asarray(x).view(np.uint8)


def _serving_pages(k, v, layout):
    """The pages a TP rank's kernel writes for these K/V heads (prepare_inputs)."""
    n, kv_heads, head_dim = k.shape
    q = jnp.zeros((n, kv_heads, head_dim), jnp.bfloat16)
    with use_abstract_mesh(TPU7X):
        _, new_kv = wrapper.prepare_inputs(
            q, k, v, q.dtype, k.dtype, kv_layout=layout, page_size=PAGE
        )
    if layout == configs.KVLayout.SEQ_ALONG_LANE:
        # [2 kv, d / p, p, n] -> [n / PAGE, 2 kv, d / p, p, PAGE]
        *head, _ = new_kv.shape
        return new_kv.reshape(*head, n // PAGE, PAGE).transpose(3, 0, 1, 2, 4)
    return new_kv.reshape(n // PAGE, PAGE, *new_kv.shape[1:])


def test_zigzag_plan_covers_the_prompt_and_balances_causal_pairs():
    plan = cp_prefill.make_zigzag_plan(81920, world=8, page_size=PAGE)
    assert plan.piece_tokens == 5120
    covered = sorted(r for rank in range(8) for r in plan.rank_slices(rank))
    assert covered == [(i * 5120, (i + 1) * 5120) for i in range(16)]
    # Causal (query, key) pairs per rank: sum over its rows of (position + 1).
    pairs = [
        sum((s + e + 1) * (e - s) // 2 for s, e in plan.rank_slices(rank))
        for rank in range(8)
    ]
    assert max(pairs) == min(pairs)


def test_zigzag_plan_rejects_unaligned_token_counts():
    with pytest.raises(ValueError, match="multiple of"):
        cp_prefill.make_zigzag_plan(80128, world=8, page_size=PAGE)


def test_gathered_page_table_finds_every_position():
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    k, v = _kv(plan.num_tokens, 4, 128, seed=0)
    gathered = jnp.concatenate(
        [
            cp_prefill.pack_kv_pages(
                _rank_rows(k, plan, r), _rank_rows(v, plan, r), PAGE
            )
            for r in range(plan.world)
        ]
    )
    in_order = cp_prefill.pack_kv_pages(k, v, PAGE)
    table = plan.gathered_page_table()
    np.testing.assert_array_equal(_bytes(gathered[table]), _bytes(in_order))
    position_pages = plan.gathered_position_pages()
    np.testing.assert_array_equal(table[position_pages], np.arange(plan.num_pages))


def test_traced_rank_gives_the_python_rank_metadata():
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)

    @jax.jit
    def traced(rank):
        return plan.sub_request_metadata(rank), plan.sliding_page_tables(rank, 4)

    for rank in range(plan.world):
        expected = (plan.sub_request_metadata(rank), plan.sliding_page_tables(rank, 4))
        for got, want in zip(jax.tree.leaves(traced(rank)), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("traced", [False, True])
def test_halo_exchange_gives_each_piece_its_window(traced):
    halo = 4
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    pp, ht = plan.piece_pages, halo * PAGE
    k, v = _kv(plan.num_tokens, 16, 128, seed=1)
    in_order = cp_prefill.pack_kv_pages(k, v, PAGE)
    rows = [(_rank_rows(k, plan, r), _rank_rows(v, plan, r)) for r in range(plan.world)]
    # Every rank's tail rows, as sliding_exchange_rows sends them.
    tails = [cp_prefill.pack_piece_tails(kr, vr, halo, PAGE) for kr, vr in rows]
    local_pages = jax.jit(cp_prefill.sliding_local_pages, static_argnums=3)
    for rank in range(plan.world):
        # As sliding_attention_layer reads its halos from the gathered rows.
        first_src, second_src = cp_prefill.halo_neighbours(rank, plan.world)
        halos = cp_prefill.kv_pages_from_rows(
            jnp.concatenate([tails[first_src][:ht], tails[second_src][ht : 2 * ht]]),
            128,
            PAGE,
        )
        args = (halos, *rows[rank], plan)
        local = (
            local_pages(*args, jnp.int32(rank)) if traced else local_pages(*args, rank)
        )
        tables = np.asarray(plan.sliding_page_tables(rank, halo))
        for sub, p in enumerate(plan.rank_pieces(rank)):
            # The halo pages the window reads before the piece...
            seen = np.arange(max(p * pp - halo, 0), p * pp)
            np.testing.assert_array_equal(
                _bytes(local[tables[sub][seen]]), _bytes(in_order[seen])
            )
            # ...and the piece's own pages, which the kernel only writes.
            assert (tables[sub][p * pp : (p + 1) * pp] == 2 * halo).all()


@pytest.mark.parametrize("halo", [0, 3])
def test_sliding_page_tables_refuse_halos_outside_one_piece(halo):
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    with pytest.raises(ValueError, match="halo"):
        plan.sliding_page_tables(0, halo)


def test_sliding_exchange_refuses_pieces_shorter_than_the_halo():
    # 8 ranks x 2 pieces of 2 pages: a 1024-token window needs 4 pages of halo.
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    rows = jnp.zeros((2 * plan.piece_tokens, 4, 128), jnp.float8_e4m3fn)
    cfg = _layer_config(4, 1024, configs.KVLayout.HEAD_ALONG_SUBLANE)
    with pytest.raises(ValueError, match="halo"):
        cp_prefill.sliding_exchange_rows(
            rows,
            rows,
            jnp.array([plan.num_pages * PAGE], jnp.int32),
            jnp.array([0], jnp.int32),
            PAGE,
            cfg=cfg,
        )


@pytest.mark.parametrize("fence", [False, True])
def test_layer_config_passes_the_output_fence(fence):
    cfg = dataclasses.replace(
        _layer_config(16, 1024, configs.KVLayout.HEAD_ALONG_SUBLANE),
        fence_output=fence,
    )
    kwargs = cfg.rpa_kwargs()
    assert kwargs.get("fence_output", False) is fence
    assert kwargs["sliding_window"] == 1024


def test_sub_request_metadata_places_queries_at_absolute_positions():
    plan = cp_prefill.make_zigzag_plan(81920, world=8, page_size=PAGE)
    cu_q_lens, kv_lens, distribution = plan.sub_request_metadata(2)
    np.testing.assert_array_equal(cu_q_lens, [0, 5120, 10240])
    np.testing.assert_array_equal(kv_lens, [3 * 5120, 14 * 5120])
    np.testing.assert_array_equal(distribution, [0, 0, 2])


@pytest.mark.parametrize(
    "num_kv_heads,expected",
    [(4, [(r // 2, 1) for r in range(8)]), (16, [(2 * r, 2) for r in range(8)])],
)
def test_tp_kv_heads_follow_tensor_parallel_sharding(num_kv_heads, expected):
    assert [cp_prefill.tp_kv_heads(num_kv_heads, 8, r) for r in range(8)] == expected


def _block_table(num_cache_pages, num_pages, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.permutation(num_cache_pages)[:num_pages], jnp.int32)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_full_attention_write_caches_each_rank_heads_up_to_the_prompt(layout):
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    num_kv_heads, head_dim, num_cache_pages = 4, 128, 48
    seq_len = plan.num_tokens - PAGE - 44  # the last page is all padding
    k, v = _kv(plan.num_tokens, num_kv_heads, head_dim, seed=2)
    gathered = jnp.concatenate(
        [
            cp_prefill.pack_kv_pages(
                _rank_rows(k, plan, r), _rank_rows(v, plan, r), PAGE
            )
            for r in range(plan.world)
        ]
    )
    block_table = _block_table(num_cache_pages, plan.num_pages, seed=3)
    src, dst, count = cp_prefill.full_attention_page_writes(
        plan, block_table, jnp.int32(seq_len)
    )
    written = np.asarray(block_table[: -(-seq_len // PAGE)])
    for rank in range(plan.world):
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        heads = slice(first, first + num_heads)
        expected = _serving_pages(k[:, heads], v[:, heads], layout)
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        pages = cp_prefill.cache_pages_from_slots(
            cp_prefill.kv_head_slots(gathered, first, num_heads), layout
        )
        with pltpu.force_tpu_interpret_mode():
            cache = cp_prefill.write_kv_pages(cache, pages, src, dst, count)
        np.testing.assert_array_equal(
            _bytes(cache[written]), _bytes(expected[: len(written)])
        )
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


@pytest.mark.parametrize(
    "num_kv_heads,layout",
    [
        (2, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (4, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (16, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (2, configs.KVLayout.SEQ_ALONG_LANE),
        (4, configs.KVLayout.SEQ_ALONG_LANE),
    ],
)
def test_head_page_writes_cache_what_slot_slices_cache(num_kv_heads, layout):
    """One head (the high or the low half of a slot word) or two per rank."""
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    num_cache_pages, seq_len = 48, plan.num_tokens - PAGE - 44
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=10)
    gathered = jnp.concatenate(
        [
            cp_prefill.pack_kv_pages(
                _rank_rows(k, plan, r), _rank_rows(v, plan, r), PAGE
            )
            for r in range(plan.world)
        ]
    )
    block_table = _block_table(num_cache_pages, plan.num_pages, seed=11)
    writes = cp_prefill.full_attention_page_writes(
        plan, block_table, jnp.int32(seq_len)
    )
    for rank in range(plan.world):
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        mine = cp_prefill.cache_pages_from_slots(
            cp_prefill.kv_head_slots(gathered, first, num_heads), layout
        )
        shape = (num_cache_pages, *mine.shape[1:])
        with pltpu.force_tpu_interpret_mode():
            want = cp_prefill.write_kv_pages(jnp.full(shape, 7, FP8), mine, *writes)
            got = cp_prefill.write_kv_head_pages(
                jnp.full(shape, 7, FP8),
                gathered,
                jnp.int32(first),
                *writes,
                num_heads=num_heads,
                kv_layout=layout,
            )
        np.testing.assert_array_equal(_bytes(got), _bytes(want))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "seq_len",
    [
        16 * 8 * PAGE,  # the window inside the last piece
        9 * 8 * PAGE + 100,  # the window across two pieces held by two ranks
        700,  # shorter than the window
    ],
)
def test_sliding_window_write_caches_the_decode_window(seq_len, layout):
    window, num_kv_heads, head_dim, num_cache_pages = 1024, 16, 128, 160
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    k, v = _kv(plan.num_tokens, num_kv_heads, head_dim, seed=4)
    position_pages, count = cp_prefill.window_position_pages(
        jnp.int32(seq_len), window, PAGE
    )
    held = jnp.stack(
        [
            cp_prefill.window_rows_held(
                _rank_rows(k, plan, r), _rank_rows(v, plan, r), plan, r, position_pages
            )
            for r in range(plan.world)
        ]
    ).reshape(plan.world, position_pages.shape[0], PAGE, -1)
    block_table = _block_table(num_cache_pages, plan.num_pages, seed=5)
    src, dst, count = cp_prefill.window_page_writes(block_table, position_pages, count)
    first_page = max(seq_len - window + 1, 0) // PAGE
    needed = np.arange(first_page, -(-seq_len // PAGE))
    np.testing.assert_array_equal(np.asarray(position_pages)[: int(count)], needed)
    for rank in range(plan.world):
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        heads = slice(first, first + num_heads)
        expected = _serving_pages(k[:, heads], v[:, heads], layout)
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        owners = cp_prefill.window_page_owners(plan, position_pages)
        slots = cp_prefill.window_slots_from_rows(
            held, owners, first, num_heads, head_dim
        )
        pages = cp_prefill.cache_pages_from_slots(slots, layout)
        with pltpu.force_tpu_interpret_mode():
            cache = cp_prefill.write_kv_pages(cache, pages, src, dst, count)
        written = np.asarray(block_table[needed])
        np.testing.assert_array_equal(_bytes(cache[written]), _bytes(expected[needed]))
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


def _layer_config(num_kv_heads, window, layout):
    return cp_prefill.LayerConfig(
        world=8,
        num_heads=2 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        sliding_window=window,
        sm_scale=1.0,
        soft_cap=None,
        k_scale=1.0,
        v_scale=1.0,
        fp8_compute=False,
        kv_layout=layout,
    )


def _identity_view(cache):
    return cache, 1, lambda native: native


def _int8(x):
    return jax.lax.bitcast_convert_type(x, jnp.int8)


def test_layer_config_plan_is_the_step_plan():
    plan = cp_prefill.make_zigzag_plan(81920, world=8, page_size=PAGE)
    cfg = _layer_config(4, None, configs.KVLayout.HEAD_ALONG_SUBLANE)
    assert cfg.plan(2 * plan.piece_tokens, PAGE) == plan


@pytest.mark.parametrize("layout", LAYOUTS)
def test_full_attention_layer_caches_rank_heads(monkeypatch, layout):
    monkeypatch.setattr(
        cp_prefill, "cp_prefill_attention", lambda q, *args, **kwargs: q
    )
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    num_kv_heads, num_cache_pages, seq_len = 4, 48, 16 * 2 * PAGE - 300
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=6)
    gathered = jnp.concatenate(
        [
            cp_prefill.pack_kv_rows(_rank_rows(k, plan, r), _rank_rows(v, plan, r))
            for r in range(plan.world)
        ]
    )
    block_table = _block_table(num_cache_pages, plan.num_pages + 4, seed=7)
    # Two request slots; the step's request is the first.
    block_tables = jnp.concatenate([block_table, jnp.zeros_like(block_table)])
    seq_lens = jnp.array([seq_len, 0], jnp.int32)
    written = np.asarray(block_table[: -(-seq_len // PAGE)])
    query = jnp.zeros((2 * plan.piece_tokens, 8, 128), jnp.bfloat16)
    cfg = _layer_config(num_kv_heads, None, layout)
    for rank in (0, 5):
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        expected = _serving_pages(
            k[:, first : first + num_heads], v[:, first : first + num_heads], layout
        )
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            cache, _ = cp_prefill.full_attention_layer(
                cache,
                query,
                _int8(gathered),
                seq_lens,
                block_tables,
                jnp.array([rank], jnp.int32),
                cfg=cfg,
                cache_view=_identity_view,
            )
        np.testing.assert_array_equal(
            _bytes(cache[written]), _bytes(expected[: len(written)])
        )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_full_attention_layer_head_page_writes_cache_rank_heads(monkeypatch, layout):
    def attention(q, pages, *args, return_kv_pages=False, **kwargs):
        return (q, pages) if return_kv_pages else q

    monkeypatch.setattr(cp_prefill, "cp_prefill_attention", attention)
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    num_kv_heads, num_cache_pages, seq_len = 2, 48, 16 * 2 * PAGE - 300
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=12)
    gathered = jnp.concatenate(
        [
            cp_prefill.pack_kv_rows(_rank_rows(k, plan, r), _rank_rows(v, plan, r))
            for r in range(plan.world)
        ]
    )
    block_table = _block_table(num_cache_pages, plan.num_pages + 4, seed=13)
    block_tables = jnp.concatenate([block_table, jnp.zeros_like(block_table)])
    seq_lens = jnp.array([seq_len, 0], jnp.int32)
    written = np.asarray(block_table[: -(-seq_len // PAGE)])
    query = jnp.zeros((2 * plan.piece_tokens, 8, 128), jnp.bfloat16)
    cfg = dataclasses.replace(
        _layer_config(num_kv_heads, None, layout), head_page_writes=True
    )
    for rank in (0, 6):  # KV heads 0 and 1: the low and the high slot half
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        expected = _serving_pages(
            k[:, first : first + num_heads], v[:, first : first + num_heads], layout
        )
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            cache, _ = cp_prefill.full_attention_layer(
                cache,
                query,
                _int8(gathered),
                seq_lens,
                block_tables,
                jnp.array([rank], jnp.int32),
                cfg=cfg,
                cache_view=_identity_view,
            )
        np.testing.assert_array_equal(
            _bytes(cache[written]), _bytes(expected[: len(written)])
        )
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


def test_head_page_writes_refuse_two_heads_of_a_seq_along_lane_cache():
    with pytest.raises(NotImplementedError, match="SEQ_ALONG_LANE"):
        dataclasses.replace(
            _layer_config(16, None, configs.KVLayout.SEQ_ALONG_LANE),
            head_page_writes=True,
        )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sliding_layers_cache_the_decode_window(monkeypatch, layout):
    monkeypatch.setattr(
        cp_prefill, "cp_prefill_attention", lambda q, *args, **kwargs: q
    )
    window, num_kv_heads, num_cache_pages = 1024, 16, 160
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    seq_len = 9 * 8 * PAGE + 100  # the window across two ranks' pieces
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=8)
    in_order = cp_prefill.pack_kv_pages(k, v, PAGE)
    block_table = _block_table(num_cache_pages, plan.num_pages, seed=9)
    seq_lens = jnp.array([seq_len], jnp.int32)
    query = jnp.zeros((2 * plan.piece_tokens, 32, 128), jnp.bfloat16)
    rows = [(_rank_rows(k, plan, r), _rank_rows(v, plan, r)) for r in range(plan.world)]
    cfg = _layer_config(num_kv_heads, window, layout)
    # The all-gather: every rank's row, in rank order.
    gathered = jnp.concatenate(
        [
            cp_prefill.sliding_exchange_rows(
                *rows[r], seq_lens, jnp.array([r], jnp.int32), PAGE, cfg=cfg
            )
            for r in range(plan.world)
        ]
    )
    position_pages, count = cp_prefill.window_position_pages(
        jnp.int32(seq_len), window, PAGE
    )
    needed = np.asarray(position_pages)[: int(count)]
    halo = cp_prefill.num_halo_pages(window, PAGE)
    by_rank = gathered.reshape(plan.world, -1, 2 * num_kv_heads * 128)
    ht, pp = halo * PAGE, plan.piece_pages
    for rank in range(plan.world):
        # Its halos: the tails of the pieces before its two (where they exist),
        # the first-piece tail of the rank below and the second-piece tail of
        # the rank above.
        for slot, src, piece in (
            (0, (rank - 1) % 8, plan.rank_pieces(rank)[0]),
            (1, (rank + 1) % 8, plan.rank_pieces(rank)[1]),
        ):
            if piece == 0 or (rank == 7 and src == 0):
                continue
            np.testing.assert_array_equal(
                _bytes(by_rank[src, slot * ht : (slot + 1) * ht]).reshape(-1),
                _bytes(in_order[piece * pp - halo : piece * pp]).reshape(-1),
            )
        first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, plan.world, rank)
        expected = _serving_pages(
            k[:, first : first + num_heads], v[:, first : first + num_heads], layout
        )
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            cache, _ = cp_prefill.sliding_attention_layer(
                cache,
                query,
                *rows[rank],
                gathered,
                seq_lens,
                block_table,
                jnp.array([rank], jnp.int32),
                cfg=cfg,
                cache_view=_identity_view,
            )
        written = np.asarray(block_table[needed])
        np.testing.assert_array_equal(_bytes(cache[written]), _bytes(expected[needed]))
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


@pytest.mark.parametrize(
    "heads,head_dim,window,fp8_compute",
    [
        ((32, 4), 512, None, True),
        # Gemma4-26B full attention: every rank's 16 query / 2 KV heads.
        ((16, 2), 512, None, True),
        ((32, 16), 256, 1024, False),
    ],
)
def test_cp_prefill_attention_lowers_for_tpu7x(heads, head_dim, window, fp8_compute):
    plan = cp_prefill.make_zigzag_plan(16 * 4 * PAGE, world=8, page_size=PAGE)
    num_q_heads, num_kv_heads = heads
    rows = 2 * plan.piece_tokens
    slots = 2 * num_kv_heads // 4
    if window is None:
        num_pages, tables = plan.num_pages, plan.gathered_page_table()
    else:
        num_pages = 2 * 4 + 1
        tables = plan.sliding_page_tables(1, 4)
    with use_abstract_mesh(TPU7X):
        q = jax.ShapeDtypeStruct((rows, num_q_heads, head_dim), jnp.bfloat16)
        pages = jax.ShapeDtypeStruct((num_pages, PAGE, slots, 4, head_dim), FP8)
        kv = jax.ShapeDtypeStruct((rows, num_kv_heads, head_dim), FP8)

        def call(q, pages, k, v):
            new_kv = {} if window is None else dict(keys=k, values=v)
            return cp_prefill.cp_prefill_attention(
                q,
                pages,
                tables,
                plan,
                1,
                num_kv_heads=num_kv_heads,
                sliding_window=window,
                fp8_compute=fp8_compute,
                **new_kv,
            )

        lowered = (
            jax.jit(call).trace(q, pages, kv, kv).lower(lowering_platforms=("tpu",))
        )
    assert "RPAm-snh-p256" in lowered.as_text()


@pytest.mark.parametrize(
    "num_kv_heads,head_dim,layout",
    [
        (2, 512, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (16, 256, configs.KVLayout.HEAD_ALONG_SUBLANE),
        (2, 512, configs.KVLayout.SEQ_ALONG_LANE),
    ],
)
def test_head_page_writes_lower_for_tpu7x(num_kv_heads, head_dim, layout):
    """The one-head (slot word halves, or the transposed SEQ_ALONG_LANE page)
    and the even-head (word rows) kernels."""
    num_pages, num_cache_pages = 64, 128
    first, num_heads = cp_prefill.tp_kv_heads(num_kv_heads, 8, 3)
    with use_abstract_mesh(TPU7X):
        slots = 2 * num_kv_heads // 4
        pages = jax.ShapeDtypeStruct((num_pages, PAGE, slots, 4, head_dim), FP8)
        head_rows = max(num_heads // 2, 1)
        page = (
            (2, head_dim // 4, 4, PAGE)
            if layout == configs.KVLayout.SEQ_ALONG_LANE
            else (PAGE, head_rows, 4, head_dim)
        )
        cache = jax.ShapeDtypeStruct((num_cache_pages, *page), FP8)
        index = jax.ShapeDtypeStruct((num_pages,), jnp.int32)

        def call(cache, pages, src, dst):
            return cp_prefill.write_kv_head_pages(
                cache,
                pages,
                jnp.int32(first),
                src,
                dst,
                jnp.int32(40),
                num_heads=num_heads,
                kv_layout=layout,
            )

        lowered = jax.jit(call).trace(cache, pages, index, index)
        text = lowered.lower(lowering_platforms=("tpu",)).as_text()
    assert "tpu_custom_call" in text


def test_head_page_writes_refuse_an_odd_head_count_above_one():
    with pytest.raises(NotImplementedError, match="even number"):
        dataclasses.replace(
            _layer_config(24, None, configs.KVLayout.HEAD_ALONG_SUBLANE),
            head_page_writes=True,
        )


def _new_kv(k, v):
    """K/V as `wrapper.prepare_inputs` lays new K/V out: [n, slots, 4, d]."""
    q = jnp.zeros((k.shape[0], k.shape[1], k.shape[2]), jnp.bfloat16)
    with use_abstract_mesh(TPU7X):
        _, new_kv = wrapper.prepare_inputs(
            q, k, v, q.dtype, k.dtype, kv_layout=LAYOUTS[0], page_size=PAGE
        )
    return new_kv


def test_new_kv_rows_are_pack_kv_rows():
    k, v = _kv(64, 4, 128, seed=20)
    np.testing.assert_array_equal(
        _bytes(_new_kv(k, v).reshape(64, -1)), _bytes(cp_prefill.pack_kv_rows(k, v))
    )


@pytest.mark.parametrize("seq_len", [16 * 8 * PAGE, 9 * 8 * PAGE + 100])
def test_sliding_exchange_rows_from_kv_match(seq_len):
    window, num_kv_heads = 1024, 16
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=21)
    cfg = _layer_config(num_kv_heads, window, LAYOUTS[0])
    seq_lens = jnp.array([seq_len], jnp.int32)
    for rank in (0, 3, 7):
        kr, vr = _rank_rows(k, plan, rank), _rank_rows(v, plan, rank)
        args = (seq_lens, jnp.array([rank], jnp.int32), PAGE)
        want = cp_prefill.sliding_exchange_rows(kr, vr, *args, cfg=cfg)
        got = cp_prefill.sliding_exchange_rows_from_kv(_new_kv(kr, vr), *args, cfg=cfg)
        np.testing.assert_array_equal(_bytes(got), _bytes(want))


def _attention_stub(calls, pages_seen=None):
    """cp_prefill_attention that records how it was called (and, given
    `pages_seen`, the pages it attends over) and returns its query (or the
    prepared one's shape of zeros) and the pages."""

    def attention(q, pages, *args, return_kv_pages=False, **kwargs):
        calls.append(kwargs.get("prepared_inputs"))
        if pages_seen is not None:
            pages_seen.append(pages)
        return (q, pages) if return_kv_pages else q

    return attention


@pytest.mark.parametrize("head_page_writes", [False, True])
def test_full_attention_layer_from_q_caches_like_from_q_k_v(
    monkeypatch, head_page_writes
):
    calls = []
    monkeypatch.setattr(cp_prefill, "cp_prefill_attention", _attention_stub(calls))
    layout = configs.KVLayout.HEAD_ALONG_SUBLANE
    plan = cp_prefill.make_zigzag_plan(16 * 2 * PAGE, world=8, page_size=PAGE)
    num_kv_heads, num_cache_pages, seq_len = 2, 48, 16 * 2 * PAGE - 300
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=22)
    gathered = _int8(
        jnp.concatenate(
            [_new_kv(_rank_rows(k, plan, r), _rank_rows(v, plan, r)) for r in range(8)]
        ).reshape(plan.num_tokens, -1)
    )
    block_table = _block_table(num_cache_pages, plan.num_pages + 4, seed=23)
    block_tables = jnp.concatenate([block_table, jnp.zeros_like(block_table)])
    seq_lens = jnp.array([seq_len, 0], jnp.int32)
    rows = 2 * plan.piece_tokens
    # _layer_config: two query heads per KV head.
    q_hbm = jnp.ones((num_kv_heads, rows, 1, 2, 128), jnp.bfloat16)
    cfg = dataclasses.replace(
        _layer_config(num_kv_heads, None, layout), head_page_writes=head_page_writes
    )
    for rank in (0, 6):
        args = (seq_lens, block_tables, jnp.array([rank], jnp.int32))
        cache = jnp.full((num_cache_pages, PAGE, 1, 4, 128), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            want, _ = cp_prefill.full_attention_layer(
                cache,
                jnp.zeros((rows, 4, 128), jnp.bfloat16),
                gathered,
                *args,
                cfg=cfg,
                cache_view=_identity_view,
            )
            got, out = cp_prefill.full_attention_layer_from_q(
                jnp.full_like(cache, 7),
                q_hbm,
                gathered,
                *args,
                cfg=cfg,
                cache_view=_identity_view,
            )
        np.testing.assert_array_equal(_bytes(got), _bytes(want))
        assert out.shape == (rows, 4, 128)
        prepared_q, prepared_kv = calls[-1]
        assert prepared_q is q_hbm and prepared_kv.shape == (rows, 1, 4, 128)


def test_sliding_attention_layer_from_q_caches_like_from_q_k_v(monkeypatch):
    calls, pages_seen = [], []
    monkeypatch.setattr(
        cp_prefill, "cp_prefill_attention", _attention_stub(calls, pages_seen)
    )
    window, num_kv_heads, num_cache_pages = 1024, 16, 160
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    seq_len = 9 * 8 * PAGE + 100
    k, v = _kv(plan.num_tokens, num_kv_heads, 128, seed=24)
    block_table = _block_table(num_cache_pages, plan.num_pages, seed=25)
    seq_lens = jnp.array([seq_len], jnp.int32)
    rows = [(_rank_rows(k, plan, r), _rank_rows(v, plan, r)) for r in range(plan.world)]
    cfg = _layer_config(num_kv_heads, window, LAYOUTS[0])
    gathered = jnp.concatenate(
        [
            cp_prefill.sliding_exchange_rows(
                *rows[r], seq_lens, jnp.array([r], jnp.int32), PAGE, cfg=cfg
            )
            for r in range(plan.world)
        ]
    )
    num_rows = 2 * plan.piece_tokens
    q_hbm = jnp.ones((num_kv_heads, num_rows, 1, 2, 128), jnp.bfloat16)
    for rank in (0, 7):
        args = (gathered, seq_lens, block_table, jnp.array([rank], jnp.int32))
        cache = jnp.full((num_cache_pages, PAGE, 8, 4, 128), 7, FP8)
        new_kv = _new_kv(*rows[rank])
        with pltpu.force_tpu_interpret_mode():
            want, _ = cp_prefill.sliding_attention_layer(
                cache,
                jnp.zeros((num_rows, 32, 128), jnp.bfloat16),
                *rows[rank],
                *args,
                cfg=cfg,
                cache_view=_identity_view,
            )
            local_want = calls[-1]
            got, out = cp_prefill.sliding_attention_layer_from_q(
                jnp.full_like(cache, 7),
                q_hbm,
                new_kv,
                *args,
                cfg=cfg,
                cache_view=_identity_view,
            )
        np.testing.assert_array_equal(_bytes(got), _bytes(want))
        # The same [halo | halo | scratch] pages: on the last rank the second
        # halo is the rank's own first-piece tail, here sliced from new_kv.
        np.testing.assert_array_equal(_bytes(pages_seen[-1]), _bytes(pages_seen[-2]))
        assert out.shape == (num_rows, 32, 128) and local_want is None
        prepared_q, prepared_kv = calls[-1]
        assert prepared_q is q_hbm and prepared_kv is new_kv


@pytest.mark.parametrize(
    "heads,head_dim,window",
    # Gemma4-26B full attention (fp8 QK/PV) and sliding window.
    [((16, 2), 512, None), ((16, 8), 256, 1024)],
)
def test_layers_from_q_lower_for_tpu7x(heads, head_dim, window):
    plan = cp_prefill.make_zigzag_plan(16 * 8 * PAGE, world=8, page_size=PAGE)
    num_q_heads, num_kv_heads = heads
    rows, slots = 2 * plan.piece_tokens, 2 * num_kv_heads // 4
    cfg = dataclasses.replace(
        _layer_config(num_kv_heads, window, configs.KVLayout.HEAD_ALONG_SUBLANE),
        num_heads=num_q_heads,
        head_dim=head_dim,
        fp8_compute=window is None,
        head_page_writes=window is None,
        prologue_eps=1e-6,
    )
    with use_abstract_mesh(TPU7X):
        q_hbm = jax.ShapeDtypeStruct(
            (num_kv_heads, rows, num_q_heads // num_kv_heads // 2, 2, head_dim),
            jnp.bfloat16,
        )
        cache = jax.ShapeDtypeStruct((64, PAGE, 1, 4, head_dim), FP8)
        meta = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((plan.num_pages,), jnp.int32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        width = 2 * num_kv_heads * head_dim
        kw = dict(cfg=cfg, cache_view=_identity_view)
        if window is None:
            gathered = jax.ShapeDtypeStruct((plan.num_tokens, width), jnp.int8)
            fn = functools.partial(cp_prefill.full_attention_layer_from_q, **kw)
            lowered = jax.jit(fn).trace(cache, q_hbm, gathered, *meta)
        else:
            new_kv = jax.ShapeDtypeStruct((rows, slots, 4, head_dim), FP8)
            exchange_rows = (3 * cp_prefill.num_halo_pages(window, PAGE) + 1) * PAGE
            gathered = jax.ShapeDtypeStruct((8 * exchange_rows, width), jnp.int8)
            fn = functools.partial(cp_prefill.sliding_attention_layer_from_q, **kw)
            lowered = jax.jit(fn).trace(cache, q_hbm, new_kv, gathered, *meta)
        text = lowered.lower(lowering_platforms=("tpu",)).as_text()
    assert "RPAm-snh-p256" in text

