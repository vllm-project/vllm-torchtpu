# SPDX-License-Identifier: Apache-2.0
"""Independent scalar/NumPy contracts for the kernel-only region replay."""

import ml_dtypes
import numpy as np


def dispatch(counts, seg_start, rank):
    ep, experts = counts.shape
    local = experts // ep
    me = int(rank[0])
    raw = counts.T.copy()
    aligned = ((raw + 1) // 2) * 2
    starts = np.cumsum(aligned, axis=1, dtype=np.int32) - aligned
    total = aligned.sum(axis=1, dtype=np.int32)
    visit = np.stack(
        [
            np.lexsort((np.arange(local), -total[d * local : (d + 1) * local]))
            for d in range(ep)
        ]
    )
    base = np.cumsum(total.reshape(ep, local), axis=1, dtype=np.int32) - total.reshape(
        ep, local
    )
    seg = np.zeros((3, experts), np.int32)
    remote = np.zeros(local, np.int32)
    for v in range(local):
        for d in range(ep):
            e = int(visit[d, v])
            ge = d * local + e
            i = v * ep + d
            n = int(counts[me, ge])
            seg[0, i] = (d << 24) | (e << 16) | n
            seg[1, i] = seg_start[ge] if total[ge] else 0
            seg[2, i] = base[d, e] + starts[ge, me] if n else 0
            remote[v] += n if d != me else 0
    recv = counts[:, me * local : (me + 1) * local].sum(axis=0, dtype=np.int32)
    meta = np.concatenate(
        (np.zeros(local + 1, np.int32), recv, [remote[2:].sum(), counts[me].sum()])
    ).astype(np.int32)
    sc = np.array(
        [recv[visit[me, 0]], recv[visit[me, 1]], remote[0], remote[1]], np.int32
    )
    return (
        raw,
        aligned,
        starts,
        total,
        seg.reshape(-1),
        meta,
        visit[me].astype(np.int32),
        sc,
    )


def local(pairs, counts, q):
    tokens = q.shape[0]
    pairs = pairs.reshape(tokens, -1)
    order = [t for e in range(counts.size) for t in range(tokens) if e in pairs[t]]
    order += [tokens - 1] * (pairs.size - len(order))
    return q[order], np.cumsum(counts, dtype=np.int32) - counts


def combine(arrivals, scales, pos, mirror, weights, valid):
    values = arrivals.reshape(arrivals.shape[0], -1).astype(np.float32)
    coefficients = scales.reshape(-1)[np.minimum(mirror, scales.size - 1)] * weights
    out = np.zeros((pos.shape[0], values.shape[1]), np.float32)
    for k in range(pos.shape[1]):
        term = (
            values[np.minimum(pos[:, k], values.shape[0] - 1)]
            * coefficients[:, k, None]
        )
        out = term if k == 0 else out + term
    out = out.astype(ml_dtypes.bfloat16)
    out[~valid] = 0
    return out


def swa(low, high, table, starts, positions, minimum, values):
    tokens, requests = positions.size, table.shape[0]
    columns = (tokens + 128 + 63) // 64 + 1
    metadata = np.zeros((requests, columns + 2), np.int32)
    bounds = np.zeros((requests, 2), np.int32)
    records = np.full(tokens, -1, np.int32)
    banks = [low.copy(), high.copy()]
    capacity = low.shape[0] * 64
    for request in range(requests):
        a, b = map(int, starts[request : request + 2])
        count = b - a
        first = int(positions[min(a, tokens - 1)])
        left, right = (max(first - 127, 0), first + count) if count else (0, 0)
        bounds[request] = left, right
        metadata[request, :2] = left // 64 * 64, minimum[request]
        for j in range(columns):
            page = left // 64 + j
            if count and page < table.shape[1] and page * 64 < right:
                metadata[request, j + 2] = table[request, page]
        for t in range(a, b):
            page = min(int(positions[t]) // 64, table.shape[1] - 1)
            record = int(table[request, page]) * 64 + int(positions[t]) % 64
            records[t] = record
            if 64 <= record < 2 * capacity:
                bank, row = divmod(record, capacity)
                banks[bank].reshape(capacity, 512)[row] = values[t]
    return *banks, metadata, bounds, records


def swa_edge_fixture(original):
    low, high, table, starts, positions, minimum, values = original
    table = table.copy()
    first = positions.reshape(48, 6)[:, 0]
    lengths = np.array([0 if i % 7 == 0 else 1 + i % 6 for i in range(48)], np.int32)
    starts = np.concatenate(([0], np.cumsum(lengths, dtype=np.int32))).astype(np.int32)
    positions = np.zeros_like(positions)
    for r, n in enumerate(lengths):
        positions[starts[r] : starts[r + 1]] = first[r] + np.arange(n)
    # Null page, out-of-range bank, and valid first row of the second bank.
    for r, page in [(1, 0), (2, 2051), (3, 1024)]:
        table[r, first[r] // 64] = page
    return low, high, table, starts, positions, minimum, values
