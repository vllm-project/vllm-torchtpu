# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eight-device JAX integration smoke, before independent full-model A/B servers.

Exercises the production model bridge, not a compiled40-layer model. No timing
or throughput claims. AB_SMOKE_OUT names the JSON receipt (or output directory).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as attention
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import gather_cache_values
from vllm_torchtpu.models.vllm.deepseek_v41.v2.candidate_cache import (
    prepare_candidate_cache_jax,
    select_candidate_cache_jax,
)


def integrated_case(lo, hi, native_lo, native_hi, q, weights, keys):
    """One replica's actual writer, readers, scorer and full selection paths."""
    slots = jnp.array([0, 1, 127, 128, 8192, 8193, 12286, -1], jnp.int32)
    original = attention.write_index_keys((lo, hi), slots, keys)
    native = attention.write_index_keys(
        (native_lo, native_hi), slots, keys, halfplane=True
    )
    records = jnp.stack((jnp.arange(4096), jnp.arange(8192, 12288))).astype(jnp.int32)
    requests = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 6)
    lengths = jnp.tile(jnp.array([4096, 4095, 257, 0, 128, 512], jnp.int32), 2)
    candidates = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32), (12, 128))
    checks = []
    for split in (False, True):
        control = gather_cache_values(original, slots[:-1], 128, split=split)
        actual = gather_cache_values(
            native, slots[:-1], 128, split=split, halfplane=True
        )
        checks.append(jnp.all(control.view(jnp.uint16) == actual.view(jnp.uint16)))
    for page_records in (None, 128):
        control = attention.paged_index_scores_jax(
            q, original, records, weights, lengths, requests, page_records
        )
        actual = attention.paged_index_scores_jax(
            q, native, records, weights, lengths, requests, page_records, halfplane=True
        )
        checks.append(jnp.all(control.view(jnp.uint32) == actual.view(jnp.uint32)))
    for candidate_filter in (False, True):
        options = dict(
            topk=512,
            block_size=8,
            candidate_write=False,
            candidate_filter=candidate_filter,
            unordered=True,
            page_records=128,
        )
        control = attention.paged_indexer_jax(
            q, original, records, weights, lengths, candidates, requests, **options
        )[0]
        actual = attention.paged_indexer_jax(
            q,
            native,
            records,
            weights,
            lengths,
            candidates,
            requests,
            halfplane=True,
            **options,
        )[0]
        checks.append(jnp.all(control == actual))
    publications = []
    for pool, halfplane in ((original, False), (native, True)):
        publications.append(
            attention.paged_indexer_jax(
                q,
                pool,
                records,
                weights,
                lengths,
                candidates,
                requests,
                topk=512,
                block_size=8,
                candidate_write=True,
                candidate_filter=False,
                unordered=True,
                page_records=128,
                halfplane=halfplane,
            )
        )
    checks.append(
        jnp.all(publications[0][0] == publications[1][0])
        & jnp.all(publications[0][1] == publications[1][1])
    )
    for packed in (False, True):
        prepared = []
        for pool, halfplane in ((original, False), (native, True)):
            prepared.append(
                prepare_candidate_cache_jax(
                    pool,
                    records,
                    candidates,
                    requests,
                    lengths,
                    8,
                    4096,
                    packed=packed,
                    halfplane=halfplane,
                )
            )
        checks.append(prepared[0][-1][0] & prepared[1][-1][0])
        outputs = []
        for pool, halfplane, data in (
            (original, False, prepared[0]),
            (native, True, prepared[1]),
        ):
            candidate_keys, ids, membership, _, usable = data
            outputs.append(
                select_candidate_cache_jax(
                    q,
                    pool,
                    records,
                    weights,
                    lengths,
                    candidates,
                    requests,
                    lengths.max().reshape(1),
                    candidate_keys,
                    ids,
                    membership,
                    usable,
                    8,
                    128,
                    1,
                    halfplane=halfplane,
                )
            )
        checks.append(jnp.all(outputs[0] == outputs[1]))
    fallback_outputs = []
    for pool, halfplane, data in (
        (original, False, prepared[0]),
        (native, True, prepared[1]),
    ):
        candidate_keys, ids, membership, _, _ = data
        fallback_outputs.append(
            select_candidate_cache_jax(
                q,
                pool,
                records,
                weights,
                lengths,
                candidates,
                requests,
                lengths.max().reshape(1),
                candidate_keys,
                ids,
                membership,
                jnp.zeros((1,), jnp.bool_),
                8,
                128,
                1,
                halfplane=halfplane,
            )
        )
    checks.append(jnp.all(fallback_outputs[0] == fallback_outputs[1]))
    # Idempotence and untouched physical rows make accidental sibling/full-pool
    # replacement observable, even if a selected-ID comparison missed it.
    again = attention.write_index_keys(native, slots, keys, halfplane=True)
    checks.append(jnp.all(again[0] == native[0]) & jnp.all(again[1] == native[1]))
    checks.append(jnp.all(native[0][100] == native_lo[100]))
    checks.append(jnp.all(native[1][100] == native_hi[100]))
    return jnp.stack(checks)


def fixture(seed):
    rng = np.random.default_rng(seed)
    values = rng.integers(-4, 5, (16384, 128)).astype(np.float32)
    bits = (values.view(np.uint32) >> 16).astype(np.uint32)
    ordinary = (bits[:, ::2] | (bits[:, 1::2] << 16)).view(np.int32).reshape(-1, 128)
    split = np.concatenate((bits[:, ::2], bits[:, 1::2]), axis=-1).reshape(-1, 2, 128)
    native = (split[:, 0] | (split[:, 1] << 16)).view(np.int32)
    q = rng.integers(-2, 3, (12, 32, 128)).astype(np.float32)
    weights = rng.integers(-2, 3, (12, 32)).astype(np.float32)
    if seed == 0:
        q.fill(0)
        q[:, 0, 0] = 1
        q[:, 1, 1] = -1
        weights.fill(0)
        weights[:, 0] = 1
        weights[:, 1] = -0.5
    keys = rng.integers(-4, 5, (8, 128)).astype(np.float32)
    return (
        jnp.asarray(ordinary[:4096]),
        jnp.asarray(ordinary[4096:]),
        jnp.asarray(native[:4096]),
        jnp.asarray(native[4096:]),
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(weights),
        jnp.asarray(keys, jnp.bfloat16),
    )


def main():
    devices = jax.local_devices()
    if len(devices) != 8 or any(d.platform != "tpu" for d in devices):
        raise RuntimeError("Integration smoke requires exactly8 local TPU devices")
    run = jax.pmap(integrated_case, devices=devices)
    cases = []
    for seed in (0, 73):
        inputs = [
            jax.device_put_sharded([value] * 8, devices) for value in fixture(seed)
        ]
        checks = np.asarray(run(*inputs).block_until_ready())
        if not checks.all():
            raise AssertionError(
                f"Bridge mismatch seed={seed}: {np.argwhere(~checks).tolist()}"
            )
        cases.append(
            {
                "seed": seed,
                "logical_devices": 8,
                "checks_per_device": checks.shape[1],
                "passed": True,
            }
        )
    receipt = {
        "status": "passed",
        "logical_devices": 8,
        "checks": sum(c["checks_per_device"] * 8 for c in cases),
        "cases": cases,
        "scope": "Integrated JAX bridge; no full-model compile or performance claim",
        "layouts": ["ordinary", "halfplane"],
    }
    destination = Path(os.environ["AB_SMOKE_OUT"])
    if destination.suffix != ".json":
        destination = destination / "smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
