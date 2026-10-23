# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CSA2 decode timing from TPU device events and host dispatch."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from jax.experimental.pallas import tpu as pltpu


def make_inputs(batch: int, heads: int, page_size: int) -> dict:
    rng = np.random.default_rng(83)
    tokens = batch * 6

    def random(shape):
        return (rng.standard_normal(shape) * 0.2).astype(ml_dtypes.bfloat16)

    fp8 = (rng.standard_normal((tokens, 512, 4, 128)) * 4).astype(
        ml_dtypes.float8_e4m3fn
    )
    bits = fp8.view(np.uint8).astype(np.uint32)
    words = np.bitwise_or.reduce(
        bits << (8 * np.arange(4, dtype=np.uint32))[None, None, :, None], axis=2
    ).view(np.int32)
    first = 2048 + np.arange(batch, dtype=np.int32) % 64
    base = (first - 127) // page_size * page_size
    columns = (133 + page_size - 1 + page_size - 1) // page_size
    page_ids = rng.permutation(np.arange(1, batch * columns + 1, dtype=np.int32))
    pages = random((1 + batch * columns, page_size, 512))
    pages[0] = 0
    angles = rng.standard_normal((tokens, 64)).astype(np.float32)
    rope = np.stack(
        (
            np.repeat(np.cos(angles), 2, axis=1),
            np.repeat(np.sin(angles), 2, axis=1)
            * np.tile(np.array([1, -1], np.float32), 64),
        ),
        axis=1,
    )
    return dict(
        q=random((tokens, heads, 512)),
        main_kv=words,
        main_kv_scale=rng.uniform(0.01, 0.025, (tokens, 512)).astype(np.float32),
        swa_kv=pages,
        positions=first,
        sink=np.linspace(-1, 1, heads, dtype=np.float32),
        rope_cache=rope,
        swa_page=np.column_stack(
            (base, first - 127, page_ids.reshape(batch, columns))
        ).astype(np.int32),
        main_lengths=np.full(tokens, 512, np.int32),
    )


def verify(inputs: dict, actual: np.ndarray) -> dict:
    errors, reference_values = [], []
    for token in sorted({0, 5, actual.shape[0] - 6, actual.shape[0] - 1}):
        seq, offset = divmod(token, 6)
        words = inputs["main_kv"][token].view(np.uint32)
        bits = np.stack([(words >> (8 * g)).astype(np.uint8) for g in range(4)], 1)
        main = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32).reshape(512, 512)
        main *= inputs["main_kv_scale"][token, :, None]
        page = inputs["swa_page"][seq]
        swa = inputs["swa_kv"][page[2:]].astype(np.float32).reshape(-1, 512)
        position = int(inputs["positions"][seq]) + offset
        keys = page[0] + np.arange(swa.shape[0])
        visible = (keys >= max(position - 127, page[1])) & (keys <= position)
        values = np.concatenate((main, swa[visible]))
        logits = inputs["q"][token].astype(np.float32) @ values.T / np.sqrt(512)
        maximum = np.maximum(logits.max(axis=1), inputs["sink"])
        weights = np.exp(logits - maximum[:, None])
        weights /= (weights.sum(axis=1) + np.exp(inputs["sink"] - maximum))[:, None]
        expected = weights @ values
        tail = expected[:, 384:].copy()
        expected[:, 384:] = (
            tail * inputs["rope_cache"][token, 0]
            + tail[:, np.arange(128) ^ 1] * inputs["rope_cache"][token, 1]
        )
        got = actual[token].astype(np.float32)
        np.testing.assert_allclose(got, expected, atol=3e-3, rtol=3e-2)
        errors.append((got - expected).ravel())
        reference_values.append(expected.ravel())
    error = np.concatenate(errors)
    return dict(
        checked_tokens=len(errors),
        relative_l2=float(
            np.linalg.norm(error) / np.linalg.norm(np.concatenate(reference_values))
        ),
        max_absolute_error=float(np.max(np.abs(error))),
    )


def device_events(directory: Path) -> list[dict]:
    from jax.profiler import ProfileData

    events = []
    for path in directory.rglob("*.xplane.pb"):
        for plane in ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                for event in line.events:
                    if "csa2_decode" in event.name:
                        events.append(
                            dict(
                                plane=plane.name,
                                name=event.name,
                                duration_us=float(event.duration_ns) / 1000,
                            )
                        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 48])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--profile-rounds", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "src/vllm_torchtpu/kernels/deepseek_v41/csa2_decode.py"
    spec = importlib.util.spec_from_file_location("decode_benchmark_kernel", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    assert versions == dict(jax="0.11.2", jaxlib="0.11.2", libtpu="0.0.50")
    device = jax.devices("tpu")[0]
    peak_hbm = pltpu.get_tpu_info().mem_bw_bytes_per_second
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runtime.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                device=str(device),
                device_kind=device.device_kind,
                devices_used=1,
                devices_visible=len(jax.devices()),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                nominal_hbm_bytes_per_second=peak_hbm,
                bandwidth_basis="Per TensorCore nominal peak, not measured HBM traffic",
            ),
            indent=2,
        )
    )
    with jax.default_device(device):
        for batch in args.batches:
            host = make_inputs(batch, args.heads, args.page_size)
            inputs = jax.device_put(host, device)
            jax.block_until_ready(inputs)
            start = time.perf_counter()
            compiled = module.csa2_decode.lower(**inputs).compile()
            compile_seconds = time.perf_counter() - start
            actual = compiled(**inputs)
            actual.block_until_ready()
            output = np.asarray(actual)
            assert np.isfinite(output.astype(np.float32)).all()
            correctness = verify(host, output)
            for _ in range(20):
                compiled(**inputs).block_until_ready()
            host_us = []
            for _ in range(5):
                start = time.perf_counter()
                outputs = [compiled(**inputs) for _ in range(100)]
                jax.block_until_ready(outputs)
                host_us.append((time.perf_counter() - start) * 1e6 / 100)
                del outputs
            case = args.out / f"b{batch}"
            case.mkdir(exist_ok=True)
            events = []
            for run in range(args.profile_rounds):
                directory = case / "xprof" / str(run)
                with jax.profiler.trace(str(directory), create_perfetto_link=False):
                    outputs = [compiled(**inputs) for _ in range(args.profile_steps)]
                    jax.block_until_ready(outputs)
                events.extend(device_events(directory))
                del outputs
            (case / "device-events.json").write_text(json.dumps(events, indent=2))
            assert len(events) == args.profile_rounds * args.profile_steps, len(events)
            times = np.array([e["duration_us"] for e in events])
            tokens = batch * 6
            # Count one logical read per Q/main/scale/RoPE/metadata, 256 SWA
            # rows per sequence, one shared sink read, and one output write.
            byte_counts = dict(
                q=host["q"].nbytes,
                main=host["main_kv"].nbytes,
                scales=host["main_kv_scale"].nbytes,
                rope=host["rope_cache"].nbytes,
                metadata=sum(
                    host[k].nbytes for k in ("positions", "swa_page", "main_lengths")
                ),
                swa=batch * 256 * 512 * 2,
                sink=host["sink"].nbytes,
                output=output.nbytes,
            )
            logical_bytes = sum(byte_counts.values())
            median = float(np.median(times))
            metric = dict(
                variant="csa2_decode",
                batch_size=batch,
                tokens=tokens,
                num_tokens=6,
                num_seqs=2 if batch % 2 == 0 else 1,
                heads=args.heads,
                page_size=args.page_size,
                device_median_us=median,
                latency_ms=median / 1000,
                device_p10_us=float(np.percentile(times, 10)),
                device_p90_us=float(np.percentile(times, 90)),
                device_event_count=len(events),
                host_continuous_mean_us=float(np.mean(host_us)),
                host_continuous_rounds_us=host_us,
                compile_seconds=compile_seconds,
                nominal_io_bytes=logical_bytes,
                nominal_io_breakdown=byte_counts,
                effective_bandwidth_tb_s=logical_bytes / median / 1e6,
                nominal_mbu_percent=100 * logical_bytes / (median * 1e-6 * peak_hbm),
                correctness=correctness,
                versions=versions,
            )
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
            print(json.dumps(metric), flush=True)


if __name__ == "__main__":
    main()
