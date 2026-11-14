# SPDX-License-Identifier: Apache-2.0
"""An all-rank digest of the unified-pool blocks a prefill step wrote, in an
order that does not depend on which device holds what (``bench prefill``).

A step writes, for each of its requests, the MLA pages of its new tokens
(global page ``k`` on the rank whose context-parallel rank is ``k % dcp``, at
the block of its MLA table's entry ``k // dcp``; a page's token ``t`` is its
``t``-th entry, so a partly filled last page counts its tokens only) and each
KDA layer's state rows at the column holding its last token (every rank its
own heads: the rank of context-parallel rank ``q`` holds head shard ``q``).
Each rank hashes the regions it holds (:func:`units`), each under a key made
of logical positions
only -- pool, kind, model layer, request, page or head shard; the keys of all
ranks sort into one canonical order, and a pool's digest hashes its units'
hashes in that order (:func:`combine`). Each pool also gets the sum of its
regions' 32-bit words, an integer checksum that NaN bit patterns cannot
poison.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence

import numpy as np

MLA, KDA = 0, 1  # the kind field of a unit's key


def layers_of(tensors: Sequence, caches: dict, groups: Sequence):
    """``(mla, kda, mla_group)``: the MLA layers as ``(pool, model layer)``,
    the KDA layers as ``(pool, model layer, group, block size)`` and the KV
    cache group of the MLA layers; ``pool`` is the index of a layer's tensor
    in ``tensors`` (``caches``: layer name -> tensor), ``group`` its index in
    ``groups``. MLA layers are named ``...mla_attn...``."""
    pool_of = {id(tensor): p for p, tensor in enumerate(tensors)}
    mla, kda, mla_groups = [], [], set()
    for g, group in enumerate(groups):
        for name in group.layer_names:
            layer = int(re.search(r"layers\.(\d+)\.", name)[1])
            pool = pool_of[id(caches[name])]
            if ".mla_attn." in name:
                mla.append((pool, layer))
                mla_groups.add(g)
            else:
                kda.append((pool, layer, g, int(group.block_size)))
    if len(mla_groups) != 1:
        raise ValueError(f"MLA layers in KV cache groups {sorted(mla_groups)}")
    return sorted(mla), sorted(kda), mla_groups.pop()


def regions(
    tables: Sequence[Sequence[np.ndarray]],
    spans: Sequence[tuple[int, int]],
    *,
    mla: Sequence[tuple[int, int]],
    kda: Sequence[tuple[int, int, int, int]],
    mla_group: int,
    page: int,
    dcp: int,
    cp_rank: int,
    page_bytes: int,
    state_bytes: int,
) -> list[tuple[tuple[int, ...], int, int, int]]:
    """The regions this rank holds of what a step wrote: ``(key, pool, block,
    bytes)`` (the block's leading ``bytes``) for request ``r``'s tokens
    ``spans[r]`` (``tables[r]``: its block-table rows by KV cache group);
    ``page_bytes`` is an MLA page's size, ``state_bytes`` a KDA state's."""
    found = []
    for r, (table, (start, end)) in enumerate(zip(tables, spans, strict=True)):
        for k in range(start // page, -(-end // page)):
            if k % dcp == cp_rank:
                block = int(table[mla_group][k // dcp])
                written = (min(end, (k + 1) * page) - k * page) * (page_bytes // page)
                for pool, layer in mla:
                    found.append(((pool, MLA, layer, r, k), pool, block, written))
        for pool, layer, group, size in kda:
            column = (end - 1) // size
            if column >= len(table[group]):
                raise ValueError(f"request {r}'s KDA column {column} is past its table")
            block = int(table[group][column])
            found.append(((pool, KDA, layer, r, cp_rank), pool, block, state_bytes))
    return found


def units(
    held: Sequence[tuple[tuple[int, ...], int, int, int]],
    read: Callable[[int, list[int]], np.ndarray],
) -> list[tuple[tuple[int, ...], bytes, int]]:
    """``(key, sha256, word sum)`` of each region in ``held``; ``read(pool,
    blocks)`` returns those blocks of a pool as uint8 ``[n, block bytes]``."""
    by_pool: dict[int, list[int]] = {}
    for _, pool, block, _ in held:
        by_pool.setdefault(pool, [])
        if block not in by_pool[pool]:
            by_pool[pool].append(block)
    data = {pool: (blocks, read(pool, blocks)) for pool, blocks in by_pool.items()}
    out = []
    for key, pool, block, size in held:
        blocks, raw = data[pool]
        region = np.ascontiguousarray(raw[blocks.index(block)].reshape(-1)[:size])
        words = int(region.view(np.uint32).sum(dtype=np.uint64))
        out.append((key, hashlib.sha256(region.tobytes()).digest(), words))
    return out


def expected_keys(
    spans: Sequence[tuple[int, int]],
    *,
    mla: Sequence[tuple[int, int]],
    kda: Sequence[tuple[int, int, int, int]],
    page: int,
    dcp: int,
) -> set[tuple[int, ...]]:
    """Every unit key a step's writes produce over all ranks."""
    keys = set()
    for r, (start, end) in enumerate(spans):
        for k in range(start // page, -(-end // page)):
            keys.update((pool, MLA, layer, r, k) for pool, layer in mla)
        for pool, layer, _, _ in kda:
            keys.update((pool, KDA, layer, r, q) for q in range(dcp))
    return keys


def combine(
    gathered: Sequence[Sequence[tuple[tuple[int, ...], bytes, int]]],
    expected: set[tuple[int, ...]],
    pools: int,
) -> dict:
    """Every rank's units (in any rank order) as ``{"all": sha256 hex,
    "pools": [[sha256 hex, word sum mod 2**64], ...]}``; raises unless the
    units are exactly ``expected``, each once."""
    found = [unit for rank_units in gathered for unit in rank_units]
    keys = [key for key, _, _ in found]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        missing, extra = sorted(expected - set(keys)), sorted(set(keys) - expected)
        raise ValueError(
            f"pool digest: {len(keys)} units, {len(set(keys))} distinct, "
            f"{len(expected)} expected (missing {missing[:3]}, unexpected "
            f"{extra[:3]})"
        )
    digests, sums = [hashlib.sha256() for _ in range(pools)], [0] * pools
    for key, digest, words in sorted(found):
        digests[key[0]].update(digest)
        sums[key[0]] = (sums[key[0]] + words) % 2**64
    hexes = [digest.hexdigest() for digest in digests]
    return {
        "all": hashlib.sha256("".join(hexes).encode()).hexdigest(),
        "pools": [[h, s] for h, s in zip(hexes, sums, strict=True)],
    }
