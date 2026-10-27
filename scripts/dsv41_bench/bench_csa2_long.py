# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare 8-Q decode CSA2 against prefill on identical inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import types
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from jax.experimental.pallas import tpu as pltpu


def load_kernels(root):
    # Keep this standalone benchmark independent of Torch initialization.
    for name in (
        "vllm_torchtpu",
        "vllm_torchtpu.kernels",
        "vllm_torchtpu.kernels.deepseek_v41",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / "src" / name.replace(".", "/"))]
            sys.modules[name] = package
    from vllm_torchtpu.kernels.deepseek_v41.csa2.decode import csa2_decode
    from vllm_torchtpu.kernels.deepseek_v41.csa2.prefill import csa2_prefill

    return csa2_prefill, csa2_decode


def make_inputs(batch=48, num_tokens=128, heads=64, varied_counts=False):
    rng = np.random.default_rng(83)
    total = batch * num_tokens

    def random(shape):
        return (rng.standard_normal(shape, dtype=np.float32) * 0.2).astype(
            ml_dtypes.bfloat16
        )

    words = np.empty((total, 512, 128), np.int32)
    for start in range(0, total, 32):
        stop = min(start + 32, total)
        fp8 = (
            rng.standard_normal((stop - start, 512, 4, 128), dtype=np.float32) * 4
        ).astype(ml_dtypes.float8_e4m3fn)
        bits = fp8.view(np.uint8).astype(np.uint32)
        words[start:stop] = np.bitwise_or.reduce(
            bits << (8 * np.arange(4, dtype=np.uint32))[None, None, :, None], axis=2
        ).view(np.int32)
    # Exercise early context, every alignment, and independent page tables.
    first = 2048 + (np.arange(batch, dtype=np.int32) * 31) % 64
    if varied_counts:
        first[: min(batch, 3)] = np.array([0, 126, 191], np.int32)[: min(batch, 3)]
    left = np.maximum(0, first - 127)
    base = left // 64 * 64
    columns = (127 + num_tokens + 63 + 63) // 64
    ids = rng.permutation(np.arange(1, batch * columns + 1, dtype=np.int32))
    swa = random((1 + batch * columns, 64, 512))
    swa[0] = 0
    angles = rng.standard_normal((total, 64), dtype=np.float32)
    rope = np.stack(
        (
            np.repeat(np.cos(angles), 2, axis=1),
            np.repeat(np.sin(angles), 2, axis=1)
            * np.tile(np.array([1, -1], np.float32), 64),
        ),
        axis=1,
    )
    counts = (
        np.resize(np.array([0, 1, 17, 127, 511, 512], np.int32), total)
        if varied_counts
        else np.full(total, 512, np.int32)
    )
    minimum = np.maximum(left, first - 80) if varied_counts else left
    return dict(
        q=random((total, heads, 512)),
        main_kv=words,
        main_kv_scale=rng.uniform(0.01, 0.025, (total, 512)).astype(np.float32),
        swa_kv=swa,
        positions=first,
        sink=np.linspace(-1, 1, heads, dtype=np.float32),
        rope_cache=rope,
        swa_page=np.column_stack((base, minimum, ids.reshape(batch, columns))).astype(
            np.int32
        ),
        main_lengths=counts,
    )


def reference(inputs, num_tokens, token_ids):
    result = []
    for token in token_ids:
        seq, offset = divmod(token, num_tokens)
        words = inputs["main_kv"][token].view(np.uint32)
        bits = np.stack([(words >> (8 * g)).astype(np.uint8) for g in range(4)], 1)
        main = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(512, 512)
        main *= inputs["main_kv_scale"][token, :, None]
        main = main[: inputs["main_lengths"][token]]
        table = inputs["swa_page"][seq]
        swa = inputs["swa_kv"][table[2:]].astype(np.float32).reshape(-1, 512)
        position = int(inputs["positions"][seq]) + offset
        keys = table[0] + np.arange(swa.shape[0])
        visible = (keys >= max(0, position - 127, table[1])) & (keys <= position)
        values = np.concatenate((main, swa[visible]))
        logits = inputs["q"][token].astype(np.float32) @ values.T / np.sqrt(512)
        maximum = np.maximum(np.max(logits, axis=1), inputs["sink"])
        weights = np.exp(logits - maximum[:, None])
        weights /= (weights.sum(axis=1) + np.exp(inputs["sink"] - maximum))[:, None]
        expected = weights @ values
        tail = expected[:, 384:].copy()
        expected[:, 384:] = (
            tail * inputs["rope_cache"][token, 0]
            + tail[:, np.arange(128) ^ 1] * inputs["rope_cache"][token, 1]
        )
        result.append(expected)
    return np.stack(result)


def prefill_call(kernel, inputs, num_tokens):
    batch = inputs["positions"].shape[0]
    return kernel(
        inputs["q"],
        inputs["main_kv"],
        inputs["main_kv_scale"],
        inputs["swa_kv"],
        jax.numpy.arange(batch + 1, dtype=jax.numpy.int32) * num_tokens,
        (inputs["positions"][:, None] + jax.numpy.arange(num_tokens)).reshape(-1),
        inputs["sink"],
        inputs["rope_cache"],
        swa_metadata=inputs["swa_page"],
        main_lengths=inputs["main_lengths"],
        bq=16,
    )


def device_events(directory):
    from jax.profiler import ProfileData

    events = []
    for path in directory.rglob("*.xplane.pb"):
        for plane in ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            for line in plane.lines:
                for event in line.events:
                    module = line.name == "XLA Modules" and "jit_run" in event.name
                    operation = event.name.partition(" = ")[0]
                    kernel = (
                        line.name == "XLA Ops"
                        and " custom-call(" in event.name
                        and ("csa2_decode" in operation or "csa2_prefill" in operation)
                    )
                    if module or kernel:
                        events.append(
                            dict(
                                kind="module" if module else "kernel",
                                name=event.name,
                                duration_us=event.duration_ns / 1000,
                            )
                        )
    return events


def compile_variant(call, inputs):
    @jax.jit
    def run(data):
        return call(data)

    return run.lower(inputs).compile()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--heads", type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefill, decode = load_kernels(root)
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    assert versions == dict(jax="0.11.2", jaxlib="0.11.2", libtpu="0.0.50")
    device = jax.devices("tpu")[0]
    runtime = dict(
        versions=versions,
        device=str(device),
        device_kind=device.device_kind,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "src/vllm_torchtpu/kernels/deepseek_v41/csa2").glob("*.py")
        },
    )
    (args.out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    host = make_inputs(args.batch, args.num_tokens, args.heads)
    selected = sorted(
        {
            seq * args.num_tokens + offset
            for seq in sorted({0, min(1, args.batch - 1), args.batch - 1})
            for offset in (0, 7, 8, 15, 16, 31, 32, 63, 64, args.num_tokens - 1)
            if offset < args.num_tokens
        }
    )
    expected = reference(host, args.num_tokens, selected)
    with jax.default_device(device):
        inputs = jax.device_put(host, device)
        jax.block_until_ready(inputs)
        peak = pltpu.get_tpu_info().mem_bw_bytes_per_second
        variants = (
            ("prefill", lambda data: prefill_call(prefill, data, args.num_tokens)),
            (
                "tile8",
                lambda data: decode(**data, num_tokens=args.num_tokens),
            ),
        )
        compiled_variants = []
        for name, call in variants:
            print(json.dumps(dict(stage="compile", variant=name)), flush=True)

            start = time.perf_counter()
            compiled = compile_variant(call, inputs)
            compile_seconds = time.perf_counter() - start
            actual = compiled(inputs).block_until_ready()
            output = np.asarray(actual).astype(np.float32)
            assert np.isfinite(output).all()
            got = output[selected]
            np.testing.assert_allclose(got, expected, atol=3e-3, rtol=3e-2)
            correctness = dict(
                checked_tokens=len(selected),
                relative_l2=float(
                    np.linalg.norm(got - expected) / np.linalg.norm(expected)
                ),
                max_absolute_error=float(np.max(np.abs(got - expected))),
            )
            del output, actual
            compiled_variants.append((name, compiled, compile_seconds, correctness))
            print(
                json.dumps(
                    dict(stage="verified", variant=name, correctness=correctness)
                ),
                flush=True,
            )
        # Alternate order across rounds to check stable improvements on one device.
        for repeat in range(2):
            for name, compiled, compile_seconds, correctness in (
                compiled_variants if repeat == 0 else reversed(compiled_variants)
            ):
                for _ in range(5):
                    compiled(inputs).block_until_ready()
                host_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    for _ in range(20):
                        result = compiled(inputs)
                    result.block_until_ready()
                    host_times.append((time.perf_counter() - start) * 1e6 / 20)
                case = args.out / f"{name}-repeat{repeat}"
                events = []
                for profile_round in range(3):
                    directory = case / "xprof" / str(profile_round)
                    with jax.profiler.trace(str(directory), create_perfetto_link=False):
                        for _ in range(10):
                            result = compiled(inputs)
                        result.block_until_ready()
                    events.extend(device_events(directory))
                case.mkdir(parents=True, exist_ok=True)
                (case / "events.json").write_text(json.dumps(events, indent=2))
                times = {
                    kind: np.array(
                        [e["duration_us"] for e in events if e["kind"] == kind]
                    )
                    for kind in ("module", "kernel")
                }
                assert len(times["module"]) == 30, (
                    name,
                    events[:10],
                    len(times["module"]),
                )
                assert len(times["kernel"]) == 30, (name, len(times["kernel"]))
                # Baseline uses its general ragged tile plan; report only common
                # logical input/output bytes, not inferred physical DMA traffic.
                logical = sum(
                    host[k].nbytes
                    for k in (
                        "q",
                        "main_kv",
                        "main_kv_scale",
                        "rope_cache",
                        "positions",
                        "swa_page",
                        "main_lengths",
                        "sink",
                    )
                )
                logical += (
                    host["q"].nbytes + args.batch * (127 + args.num_tokens) * 512 * 2
                )
                median = float(np.median(times["module"]))
                metric = dict(
                    variant=name,
                    repeat=repeat,
                    batch_size=args.batch,
                    num_tokens=args.num_tokens,
                    tokens=args.batch * args.num_tokens,
                    query_tile=8 if name == "tile8" else None,
                    heads=args.heads,
                    device_median_us=median,
                    kernel_median_us=float(np.median(times["kernel"])),
                    device_p10_us=float(np.percentile(times["module"], 10)),
                    device_p90_us=float(np.percentile(times["module"], 90)),
                    device_event_count=30,
                    host_continuous_mean_us=float(np.mean(host_times)),
                    tokens_per_second=args.batch * args.num_tokens * 1e6 / median,
                    nominal_io_bytes=logical,
                    nominal_mbu_percent=100 * logical / (median * 1e-6 * peak),
                    compile_seconds=compile_seconds,
                    correctness=correctness,
                    versions=versions,
                )
                with (args.out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
