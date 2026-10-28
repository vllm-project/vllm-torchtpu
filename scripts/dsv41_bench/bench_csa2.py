# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the shared CSA2 kernel on real paged attention inputs."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import jax
import ml_dtypes
import numpy as np


def load_kernel(root):
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
    from vllm_torchtpu.kernels.deepseek_v41.csa2.csa2 import csa2

    return csa2


def make_inputs(batch=48, num_tokens=128, heads=64, varied_counts=False, page_size=64):
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
    first = 2048 + (np.arange(batch, dtype=np.int32) * 31) % page_size
    if varied_counts:
        first[: min(batch, 3)] = np.array([0, 126, 191], np.int32)[: min(batch, 3)]
    left = np.maximum(0, first - 127)
    base = left // page_size * page_size
    columns = (127 + num_tokens + 2 * (page_size - 1)) // page_size
    ids = rng.permutation(np.arange(1, batch * columns + 1, dtype=np.int32))
    swa = random((1 + batch * columns, page_size, 512))
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
        starts=np.arange(batch + 1, dtype=np.int32) * num_tokens,
        positions=(first[:, None] + np.arange(num_tokens, dtype=np.int32)).reshape(-1),
        sink=np.linspace(-1, 1, heads, dtype=np.float32),
        rope_cache=rope,
        swa_metadata=np.column_stack(
            (base, minimum, ids.reshape(batch, columns))
        ).astype(np.int32),
        main_lengths=counts,
    )


def reference(inputs, token_ids):
    result = []
    for token in token_ids:
        seq = np.searchsorted(inputs["starts"], token, side="right") - 1
        words = inputs["main_kv"][token].view(np.uint32)
        bits = np.stack([(words >> (8 * g)).astype(np.uint8) for g in range(4)], 1)
        main = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(512, 512)
        main *= inputs["main_kv_scale"][token, :, None]
        main = main[: inputs["main_lengths"][token]]
        table = inputs["swa_metadata"][seq]
        swa = inputs["swa_kv"][table[2:]].astype(np.float32).reshape(-1, 512)
        position = int(inputs["positions"][token])
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
                        and "csa2" in operation
                    )
                    if module or kernel:
                        events.append(
                            dict(
                                kind="module" if module else "kernel",
                                name=event.name[:600],
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
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--num-tokens", type=int, default=6)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--page-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--bq", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    kernel = load_kernel(root)
    variants = [("candidate", kernel)]
    if args.baseline:
        spec = importlib.util.spec_from_file_location("baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        variants.insert(0, ("baseline", module.csa2))
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    device = jax.devices("tpu")[0]
    sources = list((root / "src/vllm_torchtpu/kernels/deepseek_v41/csa2").glob("*.py"))
    sources += [Path(__file__)] + ([args.baseline] if args.baseline else [])
    (args.out / "runtime.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                device=str(device),
                device_kind=device.device_kind,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
            ),
            indent=2,
        )
    )
    host = make_inputs(
        args.batch, args.num_tokens, args.heads, page_size=args.page_size
    )
    selected = sorted(
        {
            seq * args.num_tokens + offset
            for seq in {0, min(1, args.batch - 1), args.batch - 1}
            for offset in {0, 1, 5, 7, 8, 15, 16, 63, 64, args.num_tokens - 1}
            if offset < args.num_tokens
        }
    )
    expected = reference(host, selected)
    baseline_output = None
    with jax.default_device(device):
        inputs = jax.device_put(host, device)
        jax.block_until_ready(inputs)
        for name, call in variants:
            print(json.dumps(dict(action="compile", variant=name)), flush=True)
            start = time.perf_counter()
            compiled = compile_variant(
                lambda data, call=call: call(**data, bq=args.bq), inputs
            )
            compile_seconds = time.perf_counter() - start
            case = args.out / name
            case.mkdir(parents=True, exist_ok=True)
            (case / "optimized.hlo").write_text(compiled.as_text())
            output = np.asarray(compiled(inputs).block_until_ready()).astype(np.float32)
            assert np.isfinite(output).all()
            np.testing.assert_allclose(output[selected], expected, atol=3e-3, rtol=3e-2)
            correctness = dict(
                checked_tokens=len(selected),
                max_absolute_error=float(np.max(np.abs(output[selected] - expected))),
                relative_l2=float(
                    np.linalg.norm(output[selected] - expected)
                    / np.linalg.norm(expected)
                ),
            )
            if name == "baseline":
                baseline_output = output
            elif baseline_output is not None:
                np.testing.assert_allclose(
                    output, baseline_output, atol=3e-3, rtol=3e-2
                )
                correctness["all_outputs_bitwise_equal"] = bool(
                    np.array_equal(output, baseline_output)
                )
                correctness["baseline_max_absolute_error"] = float(
                    np.max(np.abs(output - baseline_output))
                )
            for _ in range(20):
                compiled(inputs).block_until_ready()
            with jax.profiler.trace(str(case / "xprof"), create_perfetto_link=False):
                for _ in range(30):
                    result = compiled(inputs)
                result.block_until_ready()
            events = device_events(case)
            (case / "events.json").write_text(json.dumps(events, indent=2))
            times = {
                kind: [e["duration_us"] for e in events if e["kind"] == kind]
                for kind in ("module", "kernel")
            }
            assert all(len(t) == 30 for t in times.values()), times
            metric = dict(
                variant=name,
                batch_size=args.batch,
                num_tokens=args.num_tokens,
                heads=args.heads,
                page_size=args.page_size,
                query_tile=args.bq,
                device_median_us=float(np.median(times["module"])),
                kernel_median_us=float(np.median(times["kernel"])),
                device_p10_us=float(np.percentile(times["module"], 10)),
                device_p90_us=float(np.percentile(times["module"], 90)),
                device_event_count=30,
                compile_seconds=compile_seconds,
                correctness=correctness,
                versions=versions,
            )
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
