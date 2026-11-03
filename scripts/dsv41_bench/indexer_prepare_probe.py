# SPDX-License-Identifier: Apache-2.0
"""Compile proof and three-way byte-exact preparation checks before writer A/B."""

import argparse
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
from indexer_prepare import cache, prepare  # noqa: E402


def native(keys, columns, order):
    q, s, _ = cache.quantize_index_rows(keys)
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


def reference(keys, columns, order):
    values = keys.astype(np.float32)
    exponent = np.ceil(
        np.log2(np.maximum(np.abs(values).max(1), np.float32(1e-4)) / np.float32(448))
    ).astype(np.int32)
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(137)
    columns = rng.integers(0, 256, 288, dtype=np.int32)
    order = rng.permutation(288).astype(np.int32)
    functions = {"native": jax.jit(native), "candidate": jax.jit(prepare)}
    results = []
    for kind in ("normal", "zero", "tiny", "large", "powers"):
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
        keys = keys.astype(ml_dtypes.bfloat16)
        expected = reference(keys, columns, order)
        for name, fn in functions.items():
            actual = jax.block_until_ready(
                fn(jnp.asarray(keys), jnp.asarray(columns), jnp.asarray(order))
            )
            for a, e in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(
                    np.asarray(a), e, err_msg=f"{kind}/{name}"
                )
        results.append({"case": kind, "three_way_byte_exact": True})
        (args.out / "probe.json").write_text(json.dumps(results, indent=2) + "\n")
        print(results[-1], flush=True)


if __name__ == "__main__":
    main()
