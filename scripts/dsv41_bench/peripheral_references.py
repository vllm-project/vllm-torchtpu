# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy contracts for peripheral kernel replays."""

import ml_dtypes
import numpy as np


def check_quantization(xs, actual):
    """Exact candidate/baseline is separate from compiler rounding at ties.

    Validate scales against real division and differing FP8 codes only when
    the real quotient lies within FP32 rounding error of their midpoint.
    This does not allow an arbitrary one-FP8-ULP error.
    """
    x = xs[0].astype(np.float64)
    scale = np.max(np.abs(x), axis=-1, keepdims=True) / 448
    scale = np.where(scale == 0, 1, scale)
    q, got_scale = (np.asarray(a) for a in actual)
    np.testing.assert_allclose(got_scale, scale, rtol=2e-7, atol=0)
    ideal = x / scale
    rounded = ideal.astype(ml_dtypes.float8_e4m3fn)
    differing = q.view(np.uint8) != rounded.view(np.uint8)
    a, b = q.astype(np.float64), rounded.astype(np.float64)
    if np.any(differing):
        code_distance = np.abs(
            q.view(np.uint8).astype(np.int16) - rounded.view(np.uint8).astype(np.int16)
        )
        assert np.all(code_distance[differing] == 1)
        distance = np.abs(ideal - (a + b) / 2)
        allowance = 4 * np.finfo(np.float32).eps * np.maximum(1, np.abs(ideal))
        assert np.all(distance[differing] <= allowance[differing])
    return {
        "rounding_boundary_codes": int(np.count_nonzero(differing)),
        "scale_max_relative_error": float(np.max(np.abs(got_scale / scale - 1))),
        "contract": "nearest FP8; FP32 arithmetic allowed at adjacent-code midpoints",
    }


def reference(name, xs, *, page_size=256):
    if name.startswith("quant"):
        x = xs[0].astype(np.float32)
        scale = np.max(np.abs(x), axis=-1, keepdims=True) / np.float32(448)
        scale = np.where(scale == 0, np.float32(1), scale)
        return np.clip(x / scale, -448, 448).astype(ml_dtypes.float8_e4m3fn), scale
    if name == "compact":
        records = xs[0]
        valid = (records >= 0) & (records < 32768)
        pairs = np.zeros((records.size, 2), np.int32)
        indices = np.flatnonzero(valid)
        pairs[: indices.size, 0] = indices
        pairs[: indices.size, 1] = records[indices]
        offsets = [np.count_nonzero(valid[:i]) for i in range(0, records.size + 1, 32)]
        return pairs.reshape(-1), np.asarray(offsets, np.int32)
    if name == "rope":
        table, positions = xs
        return table[np.clip(positions, 0, table.shape[0] - 1)]
    if name == "main_records":
        selected, table, starts = xs
        output = np.full(selected.shape, -1, np.int32)
        counts = np.zeros(selected.shape[0], np.int32)
        for i, row in enumerate(selected):
            if i >= starts[-1]:
                continue
            request = np.searchsorted(starts[1:], i, side="right")
            valid = row >= 0
            logical = row[valid]
            output[i, valid] = (
                table[request, np.clip(logical // page_size, 0, table.shape[1] - 1)]
                * page_size
                + logical % page_size
            )
            counts[i] = np.count_nonzero(valid)
        return output, counts
    if name == "cache_unpack":
        low, high, records = xs
        cache = np.concatenate((low, high))
        ids = np.where(
            records >= 0, records, np.arange(records.size) % (cache.shape[0] // 2)
        )
        words = cache[ids[None, :] * 2 + np.arange(2)[:, None]].view(np.uint32)
        lo = (words & 65535).astype(np.uint16).view(ml_dtypes.bfloat16)
        hi = (words >> 16).astype(np.uint16).view(ml_dtypes.bfloat16)
        return np.stack((lo, hi), axis=1).reshape(4, records.size, 128)
    if name == "mhc_pre":
        flat, packed, scale, base, previous = xs
        t, width = flat.shape
        m, d, mixes = previous.shape[1], width // previous.shape[1], 24
        weight = packed.astype(np.float64)
        weight = (
            weight[:, :mixes]
            + weight[:, mixes : 2 * mixes]
            + weight[:, 2 * mixes : 3 * mixes]
        )
        x = flat.astype(np.float64)
        norm = (x @ weight) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)

        def sigmoid(a):
            return 1 / (1 + np.exp(-a))

        pre = sigmoid(norm[:, :m] * scale[0] + base[:m]) + 1e-6
        post = sigmoid(norm[:, m : 2 * m] * scale[1] + base[m : 2 * m]) * 2
        logits = norm[:, 2 * m :].reshape(t, m, m) * scale[2] + base[2 * m :].reshape(
            1, m, m
        )
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        comb = exp / exp.sum(axis=-1, keepdims=True) + 1e-6
        comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
        for _ in range(19):
            comb /= comb.sum(axis=-1, keepdims=True) + 1e-6
            comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
        f = flat.astype(np.float32).reshape(t, m, d)
        terms = f * previous[:, :, None]
        value = (terms[:, 0] + terms[:, 2]) + (terms[:, 1] + terms[:, 3])
        return (
            post[..., None].astype(np.float32),
            comb.astype(np.float32),
            value.astype(flat.dtype),
            pre.astype(np.float32),
        )
    if name == "engram":
        hidden, kv, qw, kw, mask = xs
        t, m, d = hidden.shape
        h = hidden.astype(np.float64)
        keys = kv[:, : m * d].astype(np.float64).reshape(t, m, d)
        values = kv[:, m * d :].astype(np.float64)
        dot = np.sum(h * qw * kw * keys, axis=-1)
        dot /= np.sqrt(np.mean(h * h, axis=-1) + 1e-6)
        dot /= np.sqrt(np.mean(keys * keys, axis=-1) + 1e-6) * np.sqrt(d)
        gate_input = np.copysign(np.sqrt(np.maximum(np.abs(dot), 1e-6)), dot)
        gate = mask[:, None] / (1 + np.exp(-gate_input))
        return (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)
    if name == "router":
        scores, logits, indices, weights, scales = xs
        valid = np.any(np.isfinite(scores), axis=1) & np.any(
            np.isfinite(logits), axis=1
        )
        weights = weights / np.where(
            valid[:, None], np.maximum(weights.sum(axis=1, keepdims=True), 1e-20), 1
        )
        weights = np.where(valid[:, None], weights, 0)
        indices = np.where(valid[:, None], indices, 384).astype(np.int32)
        blocks = indices.reshape(27, 64)
        hist = np.stack(
            [np.bincount(row, minlength=385)[:384] for row in blocks]
        ).astype(np.int32)
        packed = np.zeros(2560, np.int32)
        packed[:1728] = indices.reshape(-1)
        packed[1792:2080] = scales.reshape(-1).view(np.int32)
        packed[2176:2560] = hist.sum(axis=0)
        return weights, packed, blocks, hist, valid[:, None]
    raise KeyError(name)
