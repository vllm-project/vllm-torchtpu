# SPDX-License-Identifier: Apache-2.0
"""bench prefill's all-rank pool digest (tools/megakernel/pool_digest.py):
the same written blocks give the same digest whatever device holds which
context-parallel rank and block, and one changed written byte changes it."""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test

_PATH = pathlib.Path(__file__).resolve().parents[3] / "tools/megakernel/pool_digest.py"
_SPEC = importlib.util.spec_from_file_location("pool_digest", _PATH)
pd = importlib.util.module_from_spec(_SPEC)
sys.modules["pool_digest"] = pd
_SPEC.loader.exec_module(pd)

DCP, PAGE, ROWS, ROW_BYTES, STATE_ROWS = 4, 4, 6, 8, 3
MLA_LAYERS = [(0, 3), (1, 7), (2, 11)]  # (pool, model layer)
KDA_LAYERS = [(p, layer, 1 + layer % 3, PAGE) for p, layer in ((0, 0), (0, 1), (0, 2))]
KDA_LAYERS += [(1, 4, 1, PAGE), (1, 5, 2, PAGE), (1, 6, 3, PAGE)]
SPANS = [(8, 30), (0, 13)]  # request 0 continues an 8-token prefix
BLOCKS = 40


def _content(*key):
    """What the kernel writes for a unit: fixed by its logical key."""
    seed = abs(hash(key)) % 2**32
    return np.random.default_rng(seed).integers(0, 256, (ROWS, ROW_BYTES), np.uint8)


def _device(cp_rank, rng):
    """One device's pools after the step, and its block tables: garbage
    everywhere (NaN-like bytes included), then the regions it owns."""
    pools = rng.integers(0, 256, (3, BLOCKS, ROWS, ROW_BYTES), np.uint8)
    pools[:, :, 0, :2] = 0xFF
    free = list(rng.permutation(np.arange(1, BLOCKS)))
    tables = []
    for r, (start, end) in enumerate(SPANS):
        mla = np.array([free.pop() for _ in range(-(-end // (PAGE * DCP)))])
        kda = [np.zeros(-(-end // PAGE), np.int64) for _ in range(3)]
        for row in kda:
            row[(end - 1) // PAGE] = free.pop()
        tables.append((mla, *kda))
        for k in range(start // PAGE, -(-end // PAGE)):
            if k % DCP == cp_rank:
                for pool, layer in MLA_LAYERS:
                    pools[pool, mla[k // DCP]] = _content(pool, layer, r, k)
        for pool, layer, group, _ in KDA_LAYERS:
            block = tables[r][group][(end - 1) // PAGE]
            pools[pool, block, :STATE_ROWS] = _content(pool, layer, r, cp_rank)[
                :STATE_ROWS
            ]
    return pools, tables


def _units(pools, tables, cp_rank):
    held = pd.regions(
        tables,
        SPANS,
        mla=MLA_LAYERS,
        kda=KDA_LAYERS,
        mla_group=0,
        page=PAGE,
        dcp=DCP,
        cp_rank=cp_rank,
        page_bytes=ROWS * ROW_BYTES,
        state_bytes=STATE_ROWS * ROW_BYTES,
    )
    return pd.units(
        held, lambda pool, blocks: pools[pool][blocks].reshape(len(blocks), -1)
    )


def _digest(devices, order):
    """The digest of devices holding context-parallel ranks ``order``."""
    gathered = [_units(*devices[d], order[d]) for d in range(DCP)]
    expected = pd.expected_keys(
        SPANS, mla=MLA_LAYERS, kda=KDA_LAYERS, page=PAGE, dcp=DCP
    )
    return pd.combine(gathered, expected, pools=3)


def _devices(order, seed):
    rng = np.random.default_rng(seed)
    return [_device(order[d], rng) for d in range(DCP)]


def test_digest_ignores_which_device_holds_what():
    order = [0, 1, 2, 3]
    first = _digest(_devices(order, seed=1), order)
    # Another placement: devices hold other ranks, blocks and garbage.
    moved = [2, 0, 3, 1]
    second = _digest(_devices(moved, seed=2), moved)
    assert first == second
    devices = _devices(order, seed=3)
    gathered = [_units(*devices[d], order[d]) for d in range(DCP)]
    expected = pd.expected_keys(
        SPANS, mla=MLA_LAYERS, kda=KDA_LAYERS, page=PAGE, dcp=DCP
    )
    assert pd.combine(gathered[::-1], expected, pools=3) == first


def test_digest_follows_every_written_byte_and_no_other():
    order = [0, 1, 2, 3]
    devices = _devices(order, seed=4)
    base = _digest(devices, order)
    pools, tables = devices[2]
    # A byte the step wrote: request 1's KDA layer 5 state on rank 2.
    block = tables[1][2][(SPANS[1][1] - 1) // PAGE]
    pools[1, block, STATE_ROWS - 1, 3] ^= 1
    changed = _digest(devices, order)
    assert changed["all"] != base["all"]
    assert changed["pools"][1] != base["pools"][1]
    assert changed["pools"][0] == base["pools"][0]
    pools[1, block, STATE_ROWS - 1, 3] ^= 1
    # Bytes it did not write: a state block's spare rows, a free block.
    pools[1, block, STATE_ROWS, 0] ^= 1
    pools[0, 0] ^= 0x55
    assert _digest(devices, order) == base
    # Request 0's last page (7, on rank 3) holds tokens 28 and 29 of 4: its
    # first two entries (an entry is a quarter of the page here).
    pools, tables = devices[3]
    page = pools[2, tables[0][0][7 // DCP]].reshape(PAGE, -1)
    page[2:] ^= 1
    assert _digest(devices, order) == base
    page[1, 0] ^= 1
    assert _digest(devices, order)["pools"][2] != base["pools"][2]


def test_digest_refuses_missing_or_doubled_units():
    order = [0, 1, 2, 3]
    devices = _devices(order, seed=5)
    gathered = [_units(*devices[d], order[d]) for d in range(DCP)]
    expected = pd.expected_keys(
        SPANS, mla=MLA_LAYERS, kda=KDA_LAYERS, page=PAGE, dcp=DCP
    )
    with pytest.raises(ValueError, match="missing"):
        pd.combine(gathered[:3], expected, pools=3)
    with pytest.raises(ValueError, match="distinct"):
        pd.combine(gathered + gathered[:1], expected, pools=3)


def test_word_sums_take_nan_bit_patterns_as_integers():
    order = [0, 1, 2, 3]
    result = _digest(_devices(order, seed=6), order)
    assert all(isinstance(s, int) and 0 <= s < 2**64 for _, s in result["pools"])
    nan = np.full((1, 16), 0xFF, np.uint8)  # NaN in FP32, BF16 and E4M3FN
    ((_, _, words),) = pd.units([((0, 0, 0, 0, 0), 0, 0, 16)], lambda p, b: nan)
    assert words == 4 * 0xFFFFFFFF


def test_a_column_past_the_table_is_refused():
    with pytest.raises(ValueError, match="past its table"):
        pd.regions(
            [(np.array([1]), np.array([2]), np.array([3]), np.array([4]))],
            [(0, 2 * PAGE)],
            mla=MLA_LAYERS,
            kda=KDA_LAYERS,
            mla_group=0,
            page=PAGE,
            dcp=DCP,
            cp_rank=0,
            page_bytes=ROWS * ROW_BYTES,
            state_bytes=STATE_ROWS * ROW_BYTES,
        )


def test_layers_follow_the_cache_groups():
    tensors = [object() for _ in range(2)]
    caches = {
        "model.layers.3.self_attn.mla_attn.attn": tensors[0],
        "model.layers.7.self_attn.mla_attn.attn": tensors[1],
        "model.layers.0.self_attn": tensors[0],
        "model.layers.4.self_attn": tensors[1],
    }
    groups = [
        types.SimpleNamespace(
            layer_names=tuple(n for n in caches if "mla" in n), block_size=16
        ),
        types.SimpleNamespace(layer_names=("model.layers.0.self_attn",), block_size=4),
        types.SimpleNamespace(layer_names=("model.layers.4.self_attn",), block_size=4),
    ]
    mla, kda, mla_group = pd.layers_of(tensors, caches, groups)
    assert mla == [(0, 3), (1, 7)] and mla_group == 0
    assert kda == [(0, 0, 1, 4), (1, 4, 2, 4)]
