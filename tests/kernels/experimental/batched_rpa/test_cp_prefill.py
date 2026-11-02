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
    # The tails each rank sends, as pack_piece_tails packs them.
    piece = plan.piece_tokens
    tails = [
        [
            cp_prefill.pack_kv_pages(kr[end - ht : end], vr[end - ht : end], PAGE)
            for end in (piece, 2 * piece)
        ]
        for kr, vr in rows
    ]
    up, down = cp_prefill.halo_permutations(plan.world)
    received = [[None, None] for _ in range(plan.world)]
    for slot, pairs in enumerate((up, down)):
        for src, dst in pairs:
            received[dst][slot] = tails[src][slot]
    local_pages = jax.jit(cp_prefill.sliding_local_pages, static_argnums=3)
    for rank in range(plan.world):
        halos = jnp.concatenate(received[rank])
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


@pytest.mark.parametrize("fence", [False, True])
def test_layer_config_passes_the_output_fence(fence):
    cfg = dataclasses.replace(
        _layer_config(0, 16, 1024, configs.KVLayout.HEAD_ALONG_SUBLANE),
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
    sent = [
        cp_prefill.window_slots_for_ranks(
            _rank_rows(k, plan, r),
            _rank_rows(v, plan, r),
            plan,
            r,
            position_pages,
            num_kv_heads,
        )
        for r in range(plan.world)
    ]
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
        # The all-to-all: rank `rank` receives slice `rank` of what every rank sent.
        received = jnp.stack([x[rank] for x in sent])
        owners = cp_prefill.window_page_owners(plan, position_pages)
        slots = cp_prefill.window_slots_from_ranks(received, owners)
        pages = cp_prefill.cache_pages_from_slots(slots, layout)
        with pltpu.force_tpu_interpret_mode():
            cache = cp_prefill.write_kv_pages(cache, pages, src, dst, count)
        written = np.asarray(block_table[needed])
        np.testing.assert_array_equal(_bytes(cache[written]), _bytes(expected[needed]))
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


def _layer_config(rank, num_kv_heads, window, layout):
    return cp_prefill.LayerConfig(
        rank=rank,
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
    cfg = _layer_config(0, 4, None, configs.KVLayout.HEAD_ALONG_SUBLANE)
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
    for rank in (0, 5):
        cfg = _layer_config(rank, num_kv_heads, None, layout)
        first, num_heads = cfg.tp_kv_heads
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
                cfg=cfg,
                cache_view=_identity_view,
            )
        np.testing.assert_array_equal(
            _bytes(cache[written]), _bytes(expected[: len(written)])
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
    sent = [
        cp_prefill.sliding_exchange_blocks(
            *rows[r], seq_lens, PAGE, cfg=_layer_config(r, num_kv_heads, window, layout)
        )
        for r in range(plan.world)
    ]
    position_pages, count = cp_prefill.window_position_pages(
        jnp.int32(seq_len), window, PAGE
    )
    needed = np.asarray(position_pages)[: int(count)]
    halo = cp_prefill.num_halo_pages(window, PAGE)
    halo_bytes = halo * PAGE * 2 * num_kv_heads * 128
    pp = plan.piece_pages
    for rank in range(plan.world):
        cfg = _layer_config(rank, num_kv_heads, window, layout)
        # The all-to-all: rank `rank` receives block `rank` of what every rank sent.
        received = jnp.stack([x[rank] for x in sent])
        # Its halos: the tails of the pieces before its two (where they exist).
        for src, piece in (
            ((rank - 1) % 8, plan.rank_pieces(rank)[0]),
            ((rank + 1) % 8, plan.rank_pieces(rank)[1]),
        ):
            if piece == 0 or (rank == 7 and src == 0):
                continue
            np.testing.assert_array_equal(
                _bytes(received[src, :halo_bytes]),
                _bytes(in_order[piece * pp - halo : piece * pp]).reshape(-1),
            )
        first, num_heads = cfg.tp_kv_heads
        expected = _serving_pages(
            k[:, first : first + num_heads], v[:, first : first + num_heads], layout
        )
        cache = jnp.full((num_cache_pages, *expected.shape[1:]), 7, FP8)
        with pltpu.force_tpu_interpret_mode():
            cache, _ = cp_prefill.sliding_attention_layer(
                cache,
                query,
                *rows[rank],
                received,
                seq_lens,
                block_table,
                cfg=cfg,
                cache_view=_identity_view,
            )
        written = np.asarray(block_table[needed])
        np.testing.assert_array_equal(_bytes(cache[written]), _bytes(expected[needed]))
        untouched = np.setdiff1d(np.arange(num_cache_pages), written)
        assert (np.asarray(cache[untouched]).astype(np.float32) == 7).all()


@pytest.mark.parametrize(
    "heads,head_dim,window,fp8_compute",
    [((32, 4), 512, None, True), ((32, 16), 256, 1024, False)],
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
