# SPDX-License-Identifier: Apache-2.0
"""Compile proof and three-way byte-exact preparation checks before writer A/B."""

import argparse
import functools
import json
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("vllm_torchtpu")
package.__path__ = [str(ROOT / "src/vllm_torchtpu")]
sys.modules.setdefault("vllm_torchtpu", package)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import cache  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.prepare import (  # noqa: E402
    prepare_sorted_fp8,
    supports_sorted_fp8,
)


def native(keys, columns, order, scales=None):
    if scales is None:
        q, s, _ = cache.quantize_index_rows(keys)
    else:
        s = scales
        q = (keys.astype(jnp.float32) / s[:, None]).astype(jnp.float8_e4m3fn)
    bits = (
        jax.lax.bitcast_convert_type(q, jnp.uint8)
        .astype(jnp.uint32)
        .reshape(288, 32, 4)
    )
    words = (
        bits[:, :, 0] | bits[:, :, 1] << 8 | bits[:, :, 2] << 16 | bits[:, :, 3] << 24
    )
    values = jnp.concatenate(
        (words.astype(jnp.int32), jax.lax.bitcast_convert_type(s, jnp.int32)[:, None]),
        axis=1,
    )
    return jnp.pad(columns[order], (0, 96)), jnp.pad(values[order].T, ((0, 0), (0, 96)))


def reference(keys, columns, order, scales=None):
    values = keys.astype(np.float32)
    exponent = np.ceil(
        np.log2(np.maximum(np.abs(values).max(1), np.float32(1e-4)) / np.float32(448))
    ).astype(np.int32)
    if scales is None:
        scales = np.exp2(exponent).astype(np.float32)
    bits = (
        (values / scales[:, None])
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
        .reshape(288, 32, 4)
        .astype(np.uint32)
    )
    words = (
        bits[:, :, 0] | bits[:, :, 1] << 8 | bits[:, :, 2] << 16 | bits[:, :, 3] << 24
    )
    packed = np.c_[words.view(np.int32), scales.view(np.int32)]
    return np.pad(columns[order], (0, 96)), np.pad(packed[order].T, ((0, 0), (0, 96)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(137)
    columns = rng.integers(0, 256, 288, dtype=np.int32)
    order = rng.permutation(288).astype(np.int32)

    def candidate(keys, columns, order):
        _, scales, _ = cache.quantize_index_rows(keys)
        return prepare_sorted_fp8(keys, columns, order, scales)

    functions = {"native": jax.jit(native), "candidate": jax.jit(candidate)}
    results = []
    for kind in (
        "normal",
        "zero",
        "tiny",
        "large",
        "powers",
        "scale_edges",
        "ties",
        "subnormal_edges",
    ):
        keys = rng.normal(size=(288, 128))
        if kind == "zero":
            keys[:] = 0
        elif kind == "tiny":
            keys *= 1e-9
        elif kind == "large":
            keys *= 1e9
        elif kind == "powers":
            keys = np.broadcast_to(
                np.exp2(np.arange(288) % 40 - 20)[:, None], (288, 128)
            ).copy()
        elif kind == "scale_edges":
            maximum = np.array([446, 448, 450])[np.arange(288) % 3] * np.exp2(
                np.arange(288) % 41 - 20
            )
            keys = rng.uniform(-0.99, 0.99, (288, 128)) * maximum[:, None]
            keys[:, 0] = maximum
        elif kind == "ties":
            values = (
                np.arange(256, dtype=np.uint8)
                .view(ml_dtypes.float8_e4m3fn)
                .astype(np.float32)
            )
            values = np.unique(values[np.isfinite(values)])
            midpoints = (values[:-1] + values[1:]) / 2
            keys = midpoints[np.arange(288 * 128).reshape(288, 128) % midpoints.size]
            keys[:, 0] = 448
            keys *= np.exp2(np.arange(288) % 41 - 20)[:, None]
        elif kind == "subnormal_edges":
            keys = np.zeros((288, 128), np.float32)
            keys[:, 0] = 448
            keys[:, 1:7] = np.array([1, 2, 3, -1, -2, -3]) * (2.0**-9)
            keys *= np.exp2(np.arange(288) % 41 - 20)[:, None]
        keys = keys.astype(ml_dtypes.bfloat16)
        expected = reference(keys, columns, order)
        outputs = {}
        for name, fn in functions.items():
            actual = jax.block_until_ready(
                fn(jnp.asarray(keys), jnp.asarray(columns), jnp.asarray(order))
            )
            outputs[name] = tuple(np.asarray(a) for a in actual)
            if not args.diagnose:
                for a, e in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(
                        np.asarray(a), e, err_msg=f"{kind}/{name}"
                    )
        mismatches = {
            name: [
                int(np.count_nonzero(a != e))
                for a, e in zip(value, expected, strict=True)
            ]
            for name, value in outputs.items()
        }
        result = {
            "case": kind,
            "three_way_byte_exact": not any(sum(x) for x in mismatches.values()),
            "mismatches_to_reference": mismatches,
            "native_candidate_mismatches": [
                int(np.count_nonzero(a != b))
                for a, b in zip(outputs["native"], outputs["candidate"], strict=True)
            ],
        }
        # Full composition keeps original encoded semantics, including known
        # scale-producer boundary rounding. New work is checked independently
        # at its explicit (keys, columns, order, scales) contract below.
        assert result["native_candidate_mismatches"] == [0, 0], result
        fixture_scales = np.empty(288, np.float32)
        fixture_scales[order] = expected[1][32, :288].copy().view(np.float32)
        for factor in (1, 2):
            provided = fixture_scales * factor
            expected_cut = reference(keys, columns, order, provided)
            for fn in (jax.jit(native), jax.jit(prepare_sorted_fp8)):
                actual_cut = jax.block_until_ready(
                    fn(
                        jnp.asarray(keys),
                        jnp.asarray(columns),
                        jnp.asarray(order),
                        jnp.asarray(provided),
                    )
                )
                for a, e in zip(actual_cut, expected_cut, strict=True):
                    np.testing.assert_array_equal(np.asarray(a), e)
        result["explicit_scale_cut_three_way_byte_exact"] = True
        result["explicit_scale_fixtures"] = 2
        if not result["three_way_byte_exact"]:

            def decode(words):
                payload = words[:32, :288].T.astype(np.uint32)
                bits = ((payload[..., None] >> (np.arange(4) * 8)) & 255).astype(
                    np.uint8
                )
                values = (
                    bits.reshape(288, 128)
                    .view(ml_dtypes.float8_e4m3fn)
                    .astype(np.float32)
                )
                return values * words[32, :288].copy().view(np.float32)[:, None]

            decoded = {name: decode(value[1]) for name, value in outputs.items()}
            wanted = decode(expected[1])
            result["decoded_reference_mismatches"] = {
                name: int(np.count_nonzero(value != wanted))
                for name, value in decoded.items()
            }
            result["decoded_native_candidate_mismatches"] = int(
                np.count_nonzero(decoded["native"] != decoded["candidate"])
            )
            result["decoded_reference_max_abs"] = {
                name: float(np.max(np.abs(value - wanted)))
                for name, value in decoded.items()
            }
            result["scale_reference_mismatches"] = {
                name: int(np.count_nonzero(value[1][32] != expected[1][32]))
                for name, value in outputs.items()
            }
            np.savez(
                args.out / f"{kind}-mismatch.npz",
                keys=keys.astype(np.float32),
                order=order,
                columns=columns,
                expected=expected[1],
                native=outputs["native"][1],
                candidate=outputs["candidate"][1],
            )
        results.append(result)
        (args.out / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
        print(results[-1], flush=True)

    for tokens, dtype, page_size, num_banks, offset in (
        (256, jnp.bfloat16, 256, 2, 0),
        (288, jnp.float32, 256, 2, 0),
        (288, jnp.bfloat16, 128, 2, 0),
        (288, jnp.bfloat16, 256, 1, 0),
        (288, jnp.bfloat16, 256, 2, 16),
    ):
        stride = page_size // 2
        banks = tuple(
            jnp.full((stride * 4, 128), 17, jnp.int32) for _ in range(num_banks)
        )
        layout = cache.PackedFP8Layout(
            4 * num_banks - bool(offset), stride, offset, page_size
        )
        records = rng.choice(layout.pages * page_size, tokens, replace=False).astype(
            np.int32
        )
        records[0], records[-1] = -1, layout.pages * page_size
        keys = jnp.asarray(rng.normal(size=(tokens, 128)), dtype=dtype)
        slots = jnp.asarray(records)
        supported = supports_sorted_fp8(banks, slots, keys, page_size)
        assert supported == bool(offset)
        native_fn = functools.partial(cache._write_fp8_records_native, layout=layout)
        candidate_fn = functools.partial(cache.write_fp8_records, layout=layout)
        # The guard must produce precisely the original operations for fallback.
        if not supported:
            assert str(jax.make_jaxpr(native_fn)(banks, slots, keys)) == str(
                jax.make_jaxpr(candidate_fn)(banks, slots, keys)
            )
        original = jax.block_until_ready(jax.jit(native_fn)(banks, slots, keys))
        actual = jax.block_until_ready(jax.jit(candidate_fn)(banks, slots, keys))
        v = np.asarray(keys, np.float32)
        e = np.ceil(
            np.log2(np.maximum(np.abs(v).max(1), np.float32(1e-4)) / np.float32(448))
        ).astype(np.int32)
        scale = np.exp2(e).astype(np.float32)
        bits = (
            (v / scale[:, None])
            .astype(ml_dtypes.float8_e4m3fn)
            .view(np.uint8)
            .reshape(tokens, 32, 4)
            .astype(np.uint32)
        )
        words = (
            bits[:, :, 0]
            | bits[:, :, 1] << 8
            | bits[:, :, 2] << 16
            | bits[:, :, 3] << 24
        )
        packed = np.c_[words.view(np.int32), scale.view(np.int32)]
        expected = np.full((stride * 4 * num_banks, 128), 17, np.int32)
        for i, record in enumerate(records):
            if record < 0 or record >= layout.pages * page_size:
                continue
            addresses = (
                ((record // page_size) * stride + offset) * 128
                + np.arange(33) * page_size
                + record % page_size
            )
            expected.reshape(-1)[addresses] = packed[i]
        np.testing.assert_array_equal(np.concatenate(actual), expected)
        np.testing.assert_array_equal(np.concatenate(original), expected)
        results.append(
            {
                "writer_shape": [tokens, str(dtype), page_size, num_banks, offset],
                "fused": supported,
                "three_way_byte_exact": True,
                "same_jaxpr": not supported,
            }
        )
        (args.out / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
        print(results[-1], flush=True)


if __name__ == "__main__":
    main()
