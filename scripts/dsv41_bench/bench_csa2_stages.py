# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Matched CSA2 attention A/B for encoder prefill, decoder prefill and decode.

Encoder/decoder identify DSV41 layer stages, independently of the mask. These
isolated causal cases exclude SparseCore gather, KV publication and projection.
Use the serving harness separately to establish actual framework routing and
end-to-end effects. Baseline can be a generic CSA2 file or a directory
containing decode.py and prefill.py.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import jax
import numpy as np
from bench_csa2_long import compile_variant, load_kernels, make_inputs, prefill_call
from bench_csa2_long import reference as reference_output


def events_from_trace(directory: Path) -> list[dict]:
    events = []
    for path in directory.rglob("*.xplane.pb"):
        for plane in jax.profiler.ProfileData.from_file(str(path)).planes:
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
                                name=event.name,
                                plane=plane.name,
                                duration_us=event.duration_ns / 1000,
                            )
                        )
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--stage",
        choices=("prefill_encoder", "prefill_decoder", "decode"),
        action="append",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    prefill, decode = load_kernels(root)

    def load_baseline(name):
        path = args.baseline / f"{name}.py" if args.baseline.is_dir() else args.baseline
        spec = importlib.util.spec_from_file_location(f"csa2_baseline_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            getattr(module, f"csa2_{name}") if args.baseline.is_dir() else module.csa2
        )

    baseline_prefill = load_baseline("prefill")
    baseline_decode = load_baseline("decode")
    versions = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    assert versions == dict(jax="0.11.2", jaxlib="0.11.2", libtpu="0.0.50")
    args.out.mkdir(parents=True, exist_ok=True)
    device = jax.devices("tpu")[0]
    sources = list((root / "src/vllm_torchtpu/kernels/deepseek_v41/csa2").glob("*.py"))
    sources += (
        list(args.baseline.glob("*.py")) if args.baseline.is_dir() else [args.baseline]
    )
    sources.append(Path(__file__))
    (args.out / "runtime.json").write_text(
        json.dumps(
            dict(
                versions=versions,
                device=str(device),
                device_kind=device.device_kind,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                },
                timing_basis="Single device; kernel and complete XLA module events",
                layer_stage_basis="Model encoder/decoder layers, not mask semantics",
            ),
            indent=2,
        )
    )
    cases = [
        ("prefill_encoder", 1, 2048),
        ("prefill_decoder", 48, 128),
        ("decode", 48, 6),
    ]
    for stage, batch, num_tokens in cases:
        if args.stage and stage not in args.stage:
            continue
        host = make_inputs(batch, num_tokens, 64)
        selected = sorted(
            {
                seq * num_tokens + offset
                for seq in {0, min(1, batch - 1), batch - 1}
                for offset in {
                    0,
                    min(7, num_tokens - 1),
                    min(8, num_tokens - 1),
                    min(15, num_tokens - 1),
                    min(16, num_tokens - 1),
                    min(63, num_tokens - 1),
                    min(64, num_tokens - 1),
                    num_tokens - 1,
                }
            }
        )
        expected = reference_output(host, num_tokens, selected)
        with jax.default_device(device):
            inputs = jax.device_put(host, device)
            jax.block_until_ready(inputs)

            def baseline_call(data, num_tokens=num_tokens, stage=stage):
                if args.baseline.is_dir() and stage != "prefill_encoder":
                    return baseline_decode(**data, num_tokens=num_tokens)
                return prefill_call(baseline_prefill, data, num_tokens)

            def candidate_call(data, num_tokens=num_tokens, stage=stage):
                if stage == "prefill_encoder":
                    return prefill_call(prefill, data, num_tokens)
                return decode(**data, num_tokens=num_tokens)

            variants = []
            for variant, call in (
                ("baseline", baseline_call),
                ("candidate", candidate_call),
            ):
                print(
                    json.dumps(
                        dict(
                            stage=stage,
                            tokens=num_tokens,
                            variant=variant,
                            action="compile",
                        )
                    ),
                    flush=True,
                )
                start = time.perf_counter()
                compiled = compile_variant(call, inputs)
                compile_seconds = time.perf_counter() - start
                output = np.asarray(compiled(inputs).block_until_ready()).astype(
                    np.float32
                )
                assert np.isfinite(output).all()
                actual = output[selected]
                np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-2)
                correctness = dict(
                    checked_tokens=len(selected),
                    max_absolute_error=float(np.max(np.abs(actual - expected))),
                    relative_l2=float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    ),
                )
                variants.append((variant, compiled, compile_seconds, correctness))
                del output
            for repeat in range(args.rounds):
                for variant, compiled, compile_seconds, correctness in (
                    variants if repeat == 0 else reversed(variants)
                ):
                    for _ in range(20):
                        compiled(inputs).block_until_ready()
                    host_times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        for _ in range(30):
                            result = compiled(inputs)
                        result.block_until_ready()
                        host_times.append((time.perf_counter() - start) * 1e6 / 30)
                    directory = (
                        args.out / f"{stage}-b{batch}-q{num_tokens}-{variant}-{repeat}"
                    )
                    with jax.profiler.trace(
                        str(directory / "xprof"), create_perfetto_link=False
                    ):
                        for _ in range(30):
                            result = compiled(inputs)
                        result.block_until_ready()
                    events = events_from_trace(directory)
                    (directory / "events.json").write_text(json.dumps(events, indent=2))
                    samples = {
                        kind: [e["duration_us"] for e in events if e["kind"] == kind]
                        for kind in ("module", "kernel")
                    }
                    assert all(len(values) == 30 for values in samples.values()), (
                        samples
                    )
                    metric = dict(
                        variant=variant,
                        stage=stage,
                        repeat=repeat,
                        batch_size=batch,
                        num_tokens=num_tokens,
                        tokens=batch * num_tokens,
                        heads=64,
                        page_size=64,
                        main_rows=512,
                        latency_ms=float(np.median(samples["module"])) / 1000,
                        kernel_median_us=float(np.median(samples["kernel"])),
                        module_median_us=float(np.median(samples["module"])),
                        module_p10_us=float(np.percentile(samples["module"], 10)),
                        module_p90_us=float(np.percentile(samples["module"], 90)),
                        host_queued_mean_us=float(np.mean(host_times)),
                        device_event_count=30,
                        compile_seconds=compile_seconds,
                        correctness=correctness,
                    )
                    with (args.out / "metrics.jsonl").open("a") as file:
                        file.write(json.dumps(metric) + "\n")
                    print(json.dumps(metric), flush=True)
            del variants, inputs
            jax.clear_caches()


if __name__ == "__main__":
    main()
