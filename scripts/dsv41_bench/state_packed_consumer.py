# SPDX-License-Identifier: Apache-2.0
"""Bounded actual-consumer verification, not Torch dispatch/performance evidence.

CPU: python state_packed_consumer.py --cpu-test
TPU7: python state_packed_consumer.py --out /tmp/state-packed-consumer
No timing claims. Original adversarial metadata-only cases remain unchanged.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import state_packed_shapes as metadata

RECORDS = 1024
BANK_RECORDS = RECORDS // 2


def fixture(tokens, kind):
    """Adapt original case distributions to bounded, unique ring writes."""
    table, starts, kv, positions, minimum = metadata.inputs(tokens, kind)
    # Unique request pages straddle both banks; final page tests capacity edge.
    pages = np.arange(8, 56, dtype=np.int32)
    pages[-1] = RECORDS // 16 - 1
    if kind == "bounds":
        pages[:3] = [-1, 0, 1]
    table = np.repeat(pages[:, None], 8, axis=1)
    for r in range(48):
        begin, end = starts[r : r + 2]
        # Contiguous positions eliminate original forced-position write aliases.
        # An absent ring read is safe: request-first position is even (invalid).
        positions[begin:end] = 2 * r + int(pages[r] >= 0) + np.arange(end - begin)
    positions[starts[-1] :] = 0
    return table, starts, kv, positions, minimum


def reference_metadata(arrays):
    """Independent loop oracle; uses source inputs, never candidate metadata."""
    addresses, _ = metadata.oracle(arrays)
    prev, valid, write, read = addresses
    tables = []
    for vector in (write, read):
        pairs = np.zeros((len(vector), 2), np.int32)
        indices = np.flatnonzero((vector >= 0) & (vector < RECORDS))
        pairs[: len(indices)] = np.stack((indices, vector[indices]), axis=1)
        tables.extend((pairs.ravel(), np.array([0, len(indices)], np.int32)))
    assert np.all((write == -1) | ((write >= 0) & (write < RECORDS)))
    assert np.all((read == -1) | ((read >= 0) & (read < RECORDS)))
    live = write[write >= 0]
    assert len(np.unique(live)) == len(live), "Consumer requires unique writes"
    assert np.all((valid == 0) | (prev != 0) | (read >= 0))
    return addresses, (prev[:, None], valid[:, None], *tables)


def cpu_test():
    ns = metadata.candidate_namespace(cpu=True)
    ns["total_records"] = RECORDS
    cases = []
    for tokens in metadata.TOKENS:
        assert metadata.consumer_config(tokens).token_block == tokens
        for kind in metadata.CASES:
            arrays = fixture(tokens, kind)
            addresses, expected = reference_metadata(arrays)
            actual = ns["_compressor_state_addresses"](*arrays)
            metadata.check(actual, addresses)
            packed = [actual[0][:, None], actual[1][:, None]]
            for vector in actual[2:]:
                pairs, offsets = ns["compact"](vector)
                packed.extend((pairs.ravel(), offsets))
            metadata.check(packed, expected)
            cases.append({"tokens": tokens, "case": kind, "metadata_exact": True})
    return cases


def load_consumer():
    """Reuse loader, without its obsolete private prefetch-replacement adapter."""
    from attention_consumer_compile import load_module

    base = metadata.COMPRESS.parent
    rope = load_module(base / "rope_gather.py", "bounded_rope")
    tuning = load_module(metadata.TUNING, "bounded_tuning")
    tree = ast.parse(metadata.COMPRESS.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("vllm_torchtpu.")
        )
    ]
    ns = {
        "rope_gather": rope,
        "_LANE": rope._LANE,
        "_lane_gather": rope._lane_gather,
        "COMPRESS_ROPE_TUNING": tuning.COMPRESS_ROPE_TUNING,
        "CompressConfig": tuning.CompressConfig,
    }
    exec(compile(tree, str(metadata.COMPRESS), "exec"), ns)
    return ns["fused_compress_rope"]


def verify(args, result):
    import jax
    import jax.numpy as jnp
    from attention_consumer import dense_reference

    if jax.devices()[0].platform != "tpu":
        raise RuntimeError("Actual consumer execution requires TPU; no CPU substitute")
    result["devices"] = [str(d) + ":" + d.device_kind for d in jax.devices()]
    result["jax_version"] = jax.__version__
    consumer = load_consumer()
    packed = metadata.candidate_namespace()["compressor_state_packed"]
    dense = dense_reference()
    rng = np.random.default_rng(20261104)
    for tokens in metadata.TOKENS:
        config = metadata.consumer_config(tokens)
        assert config.token_block == tokens

        def consume(
            x, weight, gamma, rope, low, high, prev, valid, *tables, config=config
        ):
            latent, words, scales, banks = consumer(
                x,
                weight,
                gamma,
                rope,
                prev,
                valid,
                (low, high),
                config=config,
                ring_tables=tuple(tables),
            )
            return latent, words, scales, *banks

        def candidate(x, weight, gamma, rope, low, high, *arrays):
            return consume(
                x,
                weight,
                gamma,
                rope,
                low,
                high,
                *packed(*arrays, total_records=RECORDS),
            )

        baseline = jax.jit(consume, donate_argnums=(4, 5))
        candidate = jax.jit(candidate, donate_argnums=(4, 5))
        x = jnp.asarray(rng.normal(0, 0.5, (tokens, 5120)), jnp.bfloat16)
        weight = jnp.asarray(rng.normal(0, 0.02, (5120, 1024)), jnp.bfloat16)
        gamma = jnp.asarray(rng.normal(size=512), jnp.bfloat16)
        angles = rng.uniform(-3, 3, (tokens, 32)).astype(np.float32)
        rope = jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), axis=1))
        # Nonconstant records expose wrong bank/record selection and read-after-write.
        initial = rng.normal(0, 0.1, (RECORDS, 1024)).astype(np.float32)
        owners = initial.view(np.int32).reshape(2, BANK_RECORDS * 8, 128)
        for kind in metadata.CASES:
            arrays = fixture(tokens, kind)
            addresses, host_metadata = reference_metadata(arrays)
            metadata.check(
                packed(*[jnp.asarray(a) for a in arrays], total_records=RECORDS),
                host_metadata,
            )
            prev, valid, write, read = addresses
            ring_prev = initial[np.maximum(read, 0)].copy()
            ring_prev[read < 0] = 0
            projection, ref_latent, ref_words, ref_scales = jax.device_get(
                dense(
                    x,
                    weight,
                    gamma,
                    prev,
                    ring_prev,
                    valid,
                    rope[:, :32],
                    rope[:, 32:],
                )
            )
            expected_owner = initial.copy()
            indices = np.flatnonzero(write >= 0)
            expected_owner[write[indices]] = projection[indices]

            # Separate fresh allocations: donated owners may not be reused.
            def arguments(
                tail, x=x, weight=weight, gamma=gamma, rope=rope, owners=owners
            ):
                return (
                    x,
                    weight,
                    gamma,
                    rope,
                    *[jax.device_put(a.copy()) for a in owners],
                    *[jnp.asarray(a) for a in tail],
                )

            ref_args = arguments(host_metadata)
            candidate_args = arguments(arrays)
            if kind == metadata.CASES[0]:
                for name, fn, values in (
                    ("host_metadata", baseline, ref_args),
                    ("packed", candidate, candidate_args),
                ):
                    lowered = fn.lower(*values)
                    compiled = lowered.compile()
                    stem = args.out / f"t{tokens}-{name}"
                    stem.with_suffix(".stablehlo.txt").write_text(lowered.as_text())
                    stem.with_suffix(".hlo.txt").write_text(compiled.as_text())
            baseline_output = jax.device_get(baseline(*ref_args))
            candidate_output = jax.device_get(candidate(*candidate_args))
            # Same actual consumer should be bit-identical, including full owners.
            metadata.check(candidate_output, baseline_output)
            latent, words, scales, low, high = candidate_output
            np.testing.assert_allclose(
                latent.astype(np.float32),
                ref_latent.astype(np.float32),
                rtol=0.02,
                atol=0.02,
            )
            np.testing.assert_allclose(scales[:, 0], ref_scales, rtol=0.02, atol=2e-5)

            # Packed words are integers, not numerically comparable. Decode FP8.
            def unpack(value, tokens=tokens):
                import ml_dtypes

                planes = np.stack(
                    [
                        (value.astype(np.uint32) >> shift) & 255
                        for shift in (0, 8, 16, 24)
                    ],
                    axis=1,
                )
                return (
                    planes.astype(np.uint8)
                    .view(ml_dtypes.float8_e4m3fn)
                    .astype(np.float32)
                    .reshape(tokens, 512)
                )

            np.testing.assert_allclose(
                unpack(words) * scales,
                unpack(ref_words) * ref_scales[:, None],
                rtol=0.15,
                atol=0.08,
            )
            got_owner = (
                np.concatenate((low, high)).view(np.float32).reshape(RECORDS, 1024)
            )
            np.testing.assert_allclose(got_owner, expected_owner, rtol=2e-4, atol=2e-5)
            untouched = np.ones(RECORDS, bool)
            untouched[write[indices]] = False
            np.testing.assert_array_equal(got_owner[untouched], initial[untouched])
            result["consumer_cases"].append(
                {
                    "tokens": tokens,
                    "case": kind,
                    "actual_consumer_exact_ab": True,
                    "dense_reference_pass": True,
                    "full_owners_pass": True,
                    "write_count": len(indices),
                    "read_count": int(np.sum(read >= 0)),
                    "max_latent_error": float(
                        np.max(
                            np.abs(
                                latent.astype(np.float32)
                                - ref_latent.astype(np.float32)
                            )
                        )
                    ),
                }
            )
            (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "scope": "Actual JAX consumer: host-oracle vs packed metadata; no timing",
        "torch_dispatcher_verified": False,
        "speedup_claimed": False,
        "target_experiment": "exp-0rfzvkskez",
        "geometry": {
            "tokens": metadata.TOKENS,
            "batch": 48,
            "hidden": 5120,
            "ratio": 2,
            "records": RECORDS,
            "banks": 2,
            "whole_tile": True,
        },
        "fixture_changes": [
            "All 35 are adapted bounded consumer cases, not original 35",
            "Unique request pages; bounds retains -1,0,1,last-page",
            "Contiguous request positions replace aliasing forced positions",
            "Padded positions zero; original starts/kv/replay minima retained",
            "Original out-of-capacity and huge positions: metadata-only",
        ],
        "cpu_cases": cpu_test(),
        "consumer_cases": [],
        "source_sha256": {},
    }
    sources = [
        Path(__file__),
        Path(metadata.__file__),
        metadata.PREP,
        metadata.COMPRESS,
        metadata.TUNING,
        metadata.COMPRESS.with_name("compress_reference.py"),
        metadata.COMPRESS.with_name("reference.py"),
        metadata.COMPRESS.with_name("rope_gather.py"),
        Path(__file__).with_name("attention_consumer.py"),
        Path(__file__).with_name("attention_consumer_compile.py"),
    ]
    result["source_sha256"] = {
        str(p.relative_to(metadata.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    if not args.cpu_test:
        if args.out is None:
            parser.error("--out required for TPU execution")
        args.out.mkdir(parents=True, exist_ok=True)
        verify(args, result)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
