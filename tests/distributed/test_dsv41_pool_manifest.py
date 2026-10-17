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
"""DeepSeek-V4.1 (v2 kernels) pool manifest: the byte contract with the kernels.

The Raiden transfer copies the byte prefix of each pool page that the manifest
declares live. That is only correct when the manifest's row order is the order
the kernels actually write, which for the v2 kernels differs from the upstream
view shape in two places:

* CSA2 SWA pages are plane-major (`write_swa_word_pool`): token t of a
  P-token page is word row t (plane 0) and row P + t (plane 1).
* The compressor state is a ring (`CircularBufferSpec`): one block per
  request, slot = position % capacity, never paged by tokens.

These tests derive token -> byte offsets from the kernel formulas and check the
manifest against them, so a layout drift fails here rather than as a wrong
answer on a live P/D pair.
"""

import types

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer import tpu_connector
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import lower_dsv4_row_spans

PAGE_BYTES = 32768
NUM_BLOCKS = 4
ROWS = NUM_BLOCKS * PAGE_BYTES // 512
ATTN = "model.layers.2.attn"
SWA = f"{ATTN}.swa_cache"
STATE = f"{ATTN}.compressor.state_cache"
IDX = f"{ATTN}.indexer.k_cache"


def _spec(block_size, tokens_per_state=1):
    return types.SimpleNamespace(block_size=block_size, tokens_per_state=tokens_per_state)


def _group(name, spec):
    return types.SimpleNamespace(layer_names=[name], kv_cache_spec=spec)


def _materialization(*, csa2=True):
    main = torch.zeros((ROWS, 128), dtype=torch.int32)
    scale = torch.zeros((ROWS,), dtype=torch.float32)
    # (shape, dtype, block stride bytes, byte offset in the shared pool).
    layouts = {
        SWA: ((NUM_BLOCKS, 32, 512), torch.bfloat16, PAGE_BYTES, 0),
        ATTN: ((NUM_BLOCKS, 64, 512), torch.uint8, PAGE_BYTES, 0),
        STATE: ((NUM_BLOCKS, 8, 1024), torch.float32, PAGE_BYTES, 0),
        IDX: ((NUM_BLOCKS, 128, 256), torch.uint8, PAGE_BYTES, 0),
    }
    named = {name: torch.empty(0) for name in layouts}
    materialized = types.SimpleNamespace(
        native_cache_layouts=layouts,
        native_raw_cache={name: main for name in layouts},
        rope_pools={},
        scale_pools={ATTN: scale},
        csa2_layers=frozenset({ATTN}) if csa2 else frozenset(),
    )
    groups = [
        _group(SWA, _spec(32)),
        _group(ATTN, _spec(128, tokens_per_state=2)),
        _group(STATE, _spec(8)),
        _group(IDX, _spec(256, tokens_per_state=2)),
    ]
    return named, groups, [main, scale], materialized


def _build(csa2=True):
    named, groups, raw, materialized = _materialization(csa2=csa2)
    return rpm.build_dsv4_pool_manifest(
        named_kv_caches=named,
        kv_cache_groups=groups,
        raw_tensors=raw,
        dsv41_materialized_kv=materialized,
    )


def _pool(manifest, tag):
    (pool,) = [pool for pool in manifest.pools if pool.tag == tag]
    return pool


def _live_byte_ranges(pool, block, num_tokens, page_tokens):
    """Byte ranges of the shared pool a transfer of ``num_tokens`` ships."""
    registration = lower_dsv4_row_spans(
        tag=pool.tag,
        num_tokens=num_tokens,
        transfer_rank=0,
        parallelism=1,
        page_tokens=page_tokens,
        live_bytes_per_block=pool.live_bytes_per_block,
        row_bytes=pool.regions[0].unit_bytes,
        block_ids=[block],
    )
    return [
        (
            pool.base_offset_bytes + block * pool.block_stride_bytes + span.src_offset_bytes,
            pool.base_offset_bytes
            + block * pool.block_stride_bytes
            + span.src_offset_bytes
            + span.size_bytes,
        )
        for span in registration.spans
    ]


def _covered(ranges, start, size):
    return any(lo <= start and start + size <= hi for lo, hi in ranges)


def swa_token_bytes(block, token, page_tokens=32):
    """Where `write_swa_word_pool` puts token ``token`` of page ``block``.

    physical_row = (rec // P) * 2P + rec % P + plane * P, 512 bytes per row.
    """
    rec = block * page_tokens + token
    rows = [
        (rec // page_tokens) * 2 * page_tokens + rec % page_tokens + plane * page_tokens
        for plane in (0, 1)
    ]
    return [(row * 512, 512) for row in rows]


def test_swa_pages_are_two_plane_pools_under_csa2():
    manifest = _build(csa2=True)
    tags = sorted(pool.tag for pool in manifest.pools)
    assert "dsv4.swa.g0" not in tags
    assert "dsv4.swa.p0.g0" in tags and "dsv4.swa.p1.g0" in tags
    p0, p1 = _pool(manifest, "dsv4.swa.p0.g0"), _pool(manifest, "dsv4.swa.p1.g0")
    assert (p0.base_offset_bytes, p1.base_offset_bytes) == (0, 32 * 512)
    assert p0.block_stride_bytes == p1.block_stride_bytes == PAGE_BYTES
    assert p0.live_bytes_per_block == p1.live_bytes_per_block == 32 * 512

    # A partial page: every byte the kernel wrote for the live tokens must be
    # inside the transferred ranges, and no byte of a dead token may be.
    block, live = 3, 5
    ranges = _live_byte_ranges(p0, block, live, 32) + _live_byte_ranges(p1, block, live, 32)
    for token in range(32):
        for start, size in swa_token_bytes(block, token):
            assert _covered(ranges, start, size) == (token < live), (token, start)


def test_swa_without_csa2_keeps_the_view_rows():
    manifest = _build(csa2=False)
    swa = _pool(manifest, "dsv4.swa.g0")
    assert swa.regions[0].unit_bytes == 1024 and swa.regions[0].num_units == 32


def test_compressor_ring_pool_is_one_whole_block():
    manifest = _build()
    ring = _pool(manifest, "dsv4.state.csa.g2")
    assert ring.live_bytes_per_block == PAGE_BYTES
    assert ring.regions[0].unit_bytes == 4096 and ring.regions[0].num_units == 8
    # The connector lowers a slot group with num_tokens == page_tokens: the
    # whole block travels regardless of the request length.
    block = 2
    ranges = _live_byte_ranges(ring, block, 8, 8)
    assert ranges == [(block * PAGE_BYTES, (block + 1) * PAGE_BYTES)]


def test_csa2_records_and_scales_share_one_block_table():
    manifest = _build()
    nope = _pool(manifest, "dsv4.csa.nope.r2.g1")
    scale = _pool(manifest, "dsv4.csa.rope.scale.r2.g1")
    assert nope.storage_index == 0 and scale.storage_index == 1
    assert nope.regions[0].unit_bytes == 512 and nope.regions[0].num_units == 64
    assert scale.regions[0].unit_bytes == 4 and scale.regions[0].num_units == 64
    assert scale.block_stride_bytes == PAGE_BYTES // 128
    # 13 tokens at ratio 2 are 7 records: rows 0..6 of both pools.
    assert _live_byte_ranges(nope, 1, 13, 128) == [(PAGE_BYTES, PAGE_BYTES + 7 * 512)]
    assert _live_byte_ranges(scale, 1, 13, 128) == [(256, 256 + 7 * 4)]


def _build_with(materialized_overrides):
    named, groups, raw, materialized = _materialization()
    for key, value in materialized_overrides.items():
        setattr(materialized, key, value)
    return rpm.build_dsv4_pool_manifest(
        named_kv_caches=named, kv_cache_groups=groups, raw_tensors=raw,
        dsv41_materialized_kv=materialized)


def test_consumer_layer_swa_keeps_token_rows():
    """KV-sharing consumer layers have `_csa2=False`: their SWA cache is
    written token-major even while the source layers write plane-major."""
    manifest = _build_with({"csa2_layers": frozenset({"model.layers.9.attn"})})
    swa = _pool(manifest, "dsv4.swa.g0")
    assert swa.regions[0].unit_bytes == 1024 and swa.regions[0].num_units == 32


def test_exact_swa_word_flag_wins_over_owner_heuristic():
    planes = _build_with({"csa2_layers": frozenset(), "swa_word_layers": frozenset({SWA})})
    assert {p.tag for p in planes.pools} >= {"dsv4.swa.p0.g0", "dsv4.swa.p1.g0"}
    rows = _build_with({"csa2_layers": frozenset({ATTN}),
                        "swa_word_layers": frozenset({"model.layers.9.attn.swa_cache"})})
    assert _pool(rows, "dsv4.swa.g0").regions[0].unit_bytes == 1024


def test_ring_groups_are_slot_groups_in_the_transfer_plan():
    from vllm.v1.kv_cache_interface import CircularBufferSpec

    ring_spec = CircularBufferSpec.__new__(CircularBufferSpec)
    object.__setattr__(ring_spec, "block_size", 8)
    groups = [
        _group(SWA, types.SimpleNamespace(block_size=32, sliding_window=128)),
        _group(ATTN, types.SimpleNamespace(block_size=128)),
        _group(STATE, ring_spec),
    ]
    plan = tpu_connector._stage3_transfer_groups(groups, multi_group=True)
    assert [group.slot for group in plan] == [False, False, True]
    assert plan[2].window_tokens is None and plan[2].page_tokens == 8
    assert plan[0].window_tokens == 128


@pytest.mark.parametrize("ids", [[], [0]])
def test_slot_block_rejects_missing_or_null_blocks(ids):
    with pytest.raises(ValueError):
        tpu_connector._stage3_slot_block(ids, group_label="g2")


def test_slot_block_keeps_the_first_block_only():
    assert tpu_connector._stage3_slot_block([17, 18], group_label="g2") == [17]
