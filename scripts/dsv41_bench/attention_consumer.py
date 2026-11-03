# SPDX-License-Identifier: Apache-2.0
"""Torch metadata + actual compressor/state updates, same-geometry A/B."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from attention_consumer_compile import consumer_functions, load_module
from attention_state import ROOT, native_function, packed_function, reference

ROWS = (33423360, 28652288)
INITIAL = np.float32(0.125)
WORD = int(np.asarray(INITIAL).view(np.int32))


def encode_reference_cpu(latent, rope, valid):
    """Independent FP32 RoPE/encoding with the consumer's rounding boundaries.

    Keep the two products in FP32 before the BF16 rounding. Constant division
    and variable division use FP32 reciprocals, as in the Pallas arithmetic.
    No Pallas, custom-op, or shared encoder helper is called by this reference.
    """
    tail = latent[:, -64:]
    cosine, sine = rope[:, :32], rope[:, 32:]
    even = tail[:, ::2] * cosine - tail[:, 1::2] * sine
    odd = tail[:, 1::2] * cosine + tail[:, ::2] * sine
    rotated = latent.copy()
    rotated[:, -64:] = (
        np.stack((even, odd), axis=-1)
        .reshape(-1, 64)
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    rotated[valid[:, 0] == 0] = 0
    scale = np.maximum(abs(rotated).max(1, keepdims=True), np.float32(1e-4))
    scale = scale * np.float32(1 / 448)
    q = (rotated * np.reciprocal(scale)).astype(ml_dtypes.float8_e4m3fn)
    planes = q.view(np.uint8).reshape(-1, 4, 128).astype(np.uint32)
    words = planes[:, 0] | planes[:, 1] << 8 | planes[:, 2] << 16 | planes[:, 3] << 24
    return words.view(np.int32), scale


def dense_reference():
    base = ROOT / "src/vllm_torchtpu/kernels/attention_preprocessing"
    rotate = load_module(base / "reference.py", "consumer_dense_rotate").rotate
    path = base / "compress_reference.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    ns = {"rotate": rotate}
    exec(compile(tree, str(path), "exec"), ns)
    return jax.jit(ns["compress"])


def fixture(kind):
    lengths = np.full(48, 6, np.int32)
    if kind == "padding":
        lengths[31:] = 0
    elif kind == "long_request":
        lengths[:] = 0
        lengths[:4] = [70, 0, 31, 186]
    elif kind == "empty":
        lengths[:] = 0
    starts = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
    # Unique pages on both physical banks, including their adjacent pages.
    blocks = np.r_[np.arange(261096, 261120), np.arange(261120, 261144)]
    table = np.repeat(blocks.astype(np.int32)[:, None], 8, axis=1)
    positions = np.zeros(288, np.int32)
    for r, length in enumerate(lengths):
        begin = starts[r]
        positions[begin : begin + length] = 98303 + np.arange(length)
    minimum = np.zeros(48, np.int32)
    if kind == "replay":
        minimum[::3] = 98304
    return table, starts, positions, minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--integration", choices=("bridge", "caller"))
    parser.add_argument("--int64-positions", action="store_true")
    parser.add_argument("--state-capacity", type=int, choices=(8, 16), default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas, sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    ns, config, source = consumer_functions()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    specs = (jax.sharding.PartitionSpec(),) * 10
    packed = packed_function(16, 16, 0)

    def original_compress(
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rope: jax.Array,
        prev: jax.Array,
        valid: jax.Array,
        low: jax.Array,
        high: jax.Array,
        write: jax.Array,
        read: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        latent, words, scales, banks = ns["fused_compress_rope"](
            x, weight, gamma, rope, prev, valid, (low, high), write, read, config=config
        )
        return latent, words, scales, *banks

    def candidate_compress(
        table: jax.Array,
        starts: jax.Array,
        positions: jax.Array,
        minimum: jax.Array,
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rope: jax.Array,
        low: jax.Array,
        high: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        prev, valid, *tables = packed(table, starts, starts, positions, minimum)
        shape_only = jnp.empty((288,), jnp.int32)
        latent, words, scales, banks = ns["compress_with_prepacked_tables"](
            x,
            weight,
            gamma,
            rope,
            prev,
            valid,
            (low, high),
            shape_only,
            shape_only,
            config=config,
            prefetch_override=tables,
        )
        return latent, words, scales, *banks, valid[:, 0] != 0

    original_op = pallas.jax_op(
        "pallas::dsv41_consumer_native_c16_s16_o0_v13",
        original_compress,
        mesh=mesh,
        input_partition_specs=specs,
        donate_argnums=(6, 7),
    )
    candidate_op = pallas.jax_op(
        "pallas::dsv41_consumer_packed_c16_s16_o0_v17",
        candidate_compress,
        mesh=mesh,
        input_partition_specs=specs,
        donate_argnums=(8, 9),
    )

    def fake_outputs(x, low, high):
        return (
            x.new_empty((288, 512)),
            x.new_empty((288, 128), dtype=torch.int32),
            x.new_empty((288, 1), dtype=torch.float32),
            torch.empty_like(low),
            torch.empty_like(high),
        )

    @original_op.register_fake
    def fake_original(x, weight, gamma, rope, prev, valid, low, high, write, read):
        return fake_outputs(x, low, high)

    @candidate_op.register_fake
    def fake_candidate(table, starts, positions, minimum, x, weight, gamma, rope, a, b):
        return *fake_outputs(x, a, b), x.new_empty((288,), dtype=torch.bool)

    native = native_function(16, 16, 0)

    def original(table, starts, positions, minimum, x, weight, gamma, rope, a, b):
        prev, valid, write, read = native(table, starts, starts, positions, minimum)
        latent, words, scales, low, high = original_op(
            x, weight, gamma, rope, prev, valid, a, b, write, read
        )
        a.copy_(low)
        b.copy_(high)
        return latent, words, scales, valid[:, 0] != 0

    def candidate(table, starts, positions, minimum, x, weight, gamma, rope, a, b):
        latent, words, scales, low, high, valid = candidate_op(
            table, starts, positions, minimum, x, weight, gamma, rope, a, b
        )
        a.copy_(low)
        b.copy_(high)
        return latent, words, scales, valid

    if args.integration:
        from attention_integration import (
            bridge_candidate,
            caller_functions,
            production_ops,
        )

        ops = production_ops(mesh)
        if args.integration == "caller":
            original, candidate = caller_functions(ops, args.state_capacity)
        else:
            candidate = bridge_candidate(ops)

    functions = {
        "native": torch.compile(original, backend="tpu", fullgraph=True, dynamic=False),
        "packed": torch.compile(
            candidate, backend="tpu", fullgraph=True, dynamic=False
        ),
    }
    rng = np.random.default_rng(20261104)
    host = (
        torch.tensor(rng.normal(size=(288, 5120)) * 0.5, dtype=torch.bfloat16),
        torch.tensor(rng.normal(size=(5120, 1024)) * 0.02, dtype=torch.bfloat16),
        torch.tensor(rng.normal(size=(512,)), dtype=torch.bfloat16),
    )
    angles = rng.uniform(-3, 3, size=(288, 32)).astype(np.float32)
    rope = np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
    compute = tuple(x.to("tpu") for x in host) + (torch.from_numpy(rope).to("tpu"),)
    result = {
        "scope": (
            "Torch native metadata or packed + unchanged compressor + state copy_; "
            "external valid mask for publish/index is also live"
        ),
        "integration": args.integration,
        "int64_positions": args.int64_positions,
        "geometry": {
            "capacity": args.state_capacity,
            "stride": args.state_capacity,
            "offset": 0,
            "bank_rows": ROWS,
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                source,
                Path(__file__),
                Path(__file__).with_name("attention_state.py"),
                Path(__file__).with_name("attention_consumer_compile.py"),
            )
        },
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
        },
        "device_memory_before": jax.devices()[0].memory_stats(),
        "correctness": [],
        "independent_reference": (
            "dense JAX projection/pool_norm + independent NumPy FP32 RoPE/encoding; "
            "dense JAX encoding mismatch also recorded"
        ),
    }

    if args.integration:
        sources = (
            "scripts/dsv41_bench/attention_integration.py",
            "src/vllm_torchtpu/kernels/experimental/decode_preparation.py",
            "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/preprocess_ops.py",
            "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py",
        )
        result["source_sha256"].update(
            {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in sources
            }
        )

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    banks = tuple(
        torch.full((rows, 128), WORD, dtype=torch.int32, device="tpu") for rows in ROWS
    )
    sync.synchronize(tensors=list(banks), wait=True)
    result["device_memory_after_allocation"] = jax.devices()[0].memory_stats()
    save()
    print("allocated real banks", result["device_memory_after_allocation"], flush=True)
    ref = dense_reference()
    ref_compute = tuple(jnp.asarray(x.float().numpy(), jnp.bfloat16) for x in host)

    def reset():
        for b in banks:
            b.fill_(WORD)
        sync.synchronize(tensors=list(banks), wait=True)

    def inputs(kind):
        arrays = fixture(kind)
        tensors = [torch.from_numpy(a).to("tpu") for a in arrays]
        if args.int64_positions:
            tensors[2] = tensors[2].to(torch.int64)
        return arrays, tuple(tensors) + compute + banks

    cases = (
        ("decode",)
        if args.diagnostic
        else ("decode", "padding", "long_request", "replay", "empty")
    )
    for kind in cases:
        arrays, values = inputs(kind)
        table, starts, pos, minimum = arrays
        prev, valid, write, read = reference(
            (table, starts, starts, pos, minimum),
            args.state_capacity,
            args.state_capacity,
            0,
        )
        initial_previous = np.zeros((288, 1024), np.float32)
        initial_previous[read >= 0] = INITIAL
        score, latent, words, scales = ref(
            *ref_compute,
            jnp.asarray(prev[:, 0]),
            jnp.asarray(initial_previous),
            jnp.asarray(valid[:, 0]),
            jnp.asarray(rope[:, :32]),
            jnp.asarray(rope[:, 32:]),
        )
        dense_expected = [
            np.asarray(latent, np.float32),
            np.asarray(words).copy(),
            np.asarray(scales[:, None]),
            valid[:, 0] != 0,
        ]
        dense_expected[1][valid[:, 0] == 0] = 0
        cpu_words, cpu_scales = encode_reference_cpu(dense_expected[0], rope, valid)
        expected = [dense_expected[0], cpu_words, cpu_scales, dense_expected[3]]
        if args.integration == "caller":
            slots = np.where(np.arange(288) < starts[-1], (pos // 2) % 64, -1)
            main_records = np.where(valid[:, 0] != 0, slots, -1)
            expected += [main_records, slots]
            dense_expected += [main_records, slots]
        score = np.asarray(score)
        rows = np.flatnonzero(write >= 0)
        assert np.unique(write[rows]).size == rows.size
        checks = []
        baseline_outputs = None
        baseline_states = []
        for name, fn in functions.items():
            reset()
            outputs = fn(*values)
            sync.synchronize(tensors=list(outputs) + list(banks), wait=True)
            actual = [
                outputs[0].float().cpu().numpy(),
                outputs[1].cpu().numpy(),
                outputs[2].cpu().numpy(),
                outputs[3].cpu().numpy(),
            ]
            actual += [x.cpu().numpy() for x in outputs[4:]]
            mismatches = []
            for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                mismatches.append(int(np.count_nonzero(a != b)))
                if not args.diagnostic:
                    np.testing.assert_array_equal(
                        a, b, err_msg=f"{kind}/{name}/output{i}"
                    )
                if baseline_outputs is not None:
                    np.testing.assert_array_equal(
                        a,
                        baseline_outputs[i],
                        err_msg=f"{kind}/native-packed/output{i}",
                    )
            if baseline_outputs is None:
                baseline_outputs = actual
            if args.diagnostic:
                np.savez(
                    args.out / f"{name}-diagnostic.npz",
                    **{f"actual_{i}": a for i, a in enumerate(actual)},
                    **{f"reference_{i}": b for i, b in enumerate(expected)},
                    rope=rope,
                    valid=valid,
                    score=score,
                )
            changed = []
            state_mismatches = []
            cut = 0
            for bank_id, (bank, bank_rows) in enumerate(zip(banks, ROWS, strict=True)):
                records = bank_rows // 8
                select = rows[(write[rows] >= cut) & (write[rows] < cut + records)]
                ids = torch.from_numpy((write[select] - cut).astype(np.int64)).to("tpu")
                got = bank.reshape(-1, 1024).index_select(0, ids).cpu().numpy()
                want = score[select].view(np.int32)
                state_mismatches.append(int(np.count_nonzero(got != want)))
                if args.diagnostic:
                    np.savez(
                        args.out / f"{name}-bank{bank_id}.npz",
                        actual=got,
                        reference=want,
                    )
                if not args.diagnostic:
                    np.testing.assert_array_equal(
                        got, want, err_msg=f"{kind}/{name}/state"
                    )
                elif name == "packed":
                    np.testing.assert_array_equal(got, baseline_states[bank_id])
                else:
                    baseline_states.append(got)
                total = int(torch.count_nonzero(bank != WORD).cpu())
                assert total == int(np.count_nonzero(want != WORD)), (kind, name, total)
                changed.append(total)
                cut += records
            checks.append(
                {
                    "path": name,
                    "outputs_reference_mismatches": mismatches,
                    "dense_jax_reference_mismatches": [
                        int(np.count_nonzero(a != b))
                        for a, b in zip(actual, dense_expected, strict=True)
                    ],
                    "strict_reference": not args.diagnostic,
                    "state_reference_mismatches": state_mismatches,
                    "changed_words": changed,
                }
            )
            print(kind, checks[-1], flush=True)
        result["correctness"].append({"case": kind, "paths": checks})
        save()

    if args.diagnostic:
        result["diagnostic_complete"] = True
        save()
        return

    arrays, values = inputs("decode")
    table, starts, pos, minimum = arrays
    _, _, write, read = reference(
        (table, starts, starts, pos, minimum),
        args.state_capacity,
        args.state_capacity,
        0,
    )
    assert not np.intersect1d(write[write >= 0], read[read >= 0]).size
    reset()

    def run(fn):
        outputs = fn(*values)
        sync.synchronize(tensors=list(outputs) + list(banks), wait=True)

    for _ in range(20):
        for fn in functions.values():
            run(fn)
    samples = {name: [] for name in functions}
    for i in range(args.samples):
        order = list(functions.items())
        for name, fn in order if i % 2 == 0 else order[::-1]:
            before = time.perf_counter_ns()
            run(fn)
            samples[name].append((time.perf_counter_ns() - before) / 1000)
    result["synchronous_us"] = {
        name: {
            "median": float(np.median(xs)),
            "p10": float(np.percentile(xs, 10)),
            "p90": float(np.percentile(xs, 90)),
            "samples": xs,
        }
        for name, xs in samples.items()
    }
    save()
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for i in range(16):
                for name, fn in functions.items():
                    with jax.profiler.TraceAnnotation(name, step_num=i):
                        run(fn)
    result["passed"] = True
    save()
    print({k: v["median"] for k, v in result["synchronous_us"].items()}, flush=True)


if __name__ == "__main__":
    main()
