import argparse
import gzip
import hashlib
import json
import os
import runpy
import shutil
import statistics
import time
from datetime import datetime
from pathlib import Path

# Set compiler and tracing flags before importing JAX (profiling guide §5.1).
artifact_dir = Path(os.environ.get("ARTIFACT_LOCAL_DIR", "/tmp/pallas-profile"))
llo_dir = artifact_dir / "compiler" / "llo"
hlo_dir = artifact_dir / "compiler" / "hlo"
llo_dir.mkdir(parents=True, exist_ok=True)
hlo_dir.mkdir(parents=True, exist_ok=True)

os.environ["LIBTPU_INIT_ARGS"] = " ".join(
    [
        os.environ.get("LIBTPU_INIT_ARGS", ""),
        "--xla_enable_custom_call_region_trace=true",
        "--xla_xprof_register_llo_debug_info=true",
        f"--xla_mosaic_dump_to={llo_dir}",
    ]
).strip()
os.environ["XLA_FLAGS"] = " ".join(
    [
        os.environ.get("XLA_FLAGS", ""),
        f"--xla_dump_to={hlo_dir}",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_as_proto",
    ]
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# Load the pure JAX kernel without running package startup hooks.
repo_root = Path(__file__).resolve().parents[2]
kernel_path = repo_root / "src/vllm_torchtpu/kernels/fused_moe/v2/router_ops.py"
parser = argparse.ArgumentParser(description="TPU router device-time benchmark")
parser.add_argument("--rows", type=int, default=256)
parser.add_argument("--block-rows", type=int, default=256)
parser.add_argument("--kernel-file", type=Path, default=kernel_path)
args = parser.parse_args()
kernel_path = args.kernel_file.resolve()
kernel_sha256 = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
pallas_select = runpy.run_path(str(kernel_path))["pallas_select"]
print(f"Kernel: {kernel_path} (sha256={kernel_sha256})", flush=True)
rows, experts, topk = args.rows, 384, 6
block_rows = args.block_rows
if rows <= 0 or block_rows <= 0 or rows % block_rows:
    parser.error("--rows must be positive and divisible by --block-rows")
print(
    f"rows={rows}, experts={experts}, topk={topk}, block_rows={block_rows}", flush=True
)
scores = jax.random.uniform(jax.random.key(0), (rows, experts), dtype=jnp.float32)

# Pallas selector
select = jax.jit(
    lambda x: pallas_select(x, topk=topk, block_rows=block_rows, interpret=False)
)

# 基线：同形状的 XLA top_k
baseline = jax.jit(lambda x: jax.lax.top_k(x, topk))

actual_weights, actual_indices = jax.device_get(select(scores))
expected_weights, expected_indices = jax.device_get(baseline(scores))
np.testing.assert_array_equal(actual_indices, expected_indices)
np.testing.assert_array_equal(actual_weights, expected_weights)
print("Correctness: weights and expert IDs match XLA top_k exactly.", flush=True)


def device_samples(profile_dir: Path, iters: int) -> tuple[list[float], dict[str, int]]:
    """Read complete TPU XLA-module events; Chrome trace durations are in us."""
    per_device = {}
    for trace_file in sorted(profile_dir.glob("*.trace.json.gz")):
        with gzip.open(trace_file, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        devices = {
            event["pid"]: event.get("args", {}).get("name", "")
            for event in events
            if event.get("ph") == "M"
            and event.get("name") == "process_name"
            and "/device:TPU" in event.get("args", {}).get("name", "")
        }
        module_lanes = {
            (event["pid"], event["tid"])
            for event in events
            if event.get("ph") == "M"
            and event.get("name") == "thread_name"
            and event.get("args", {}).get("name") == "XLA Modules"
        }
        for pid, device in devices.items():
            modules = sorted(
                (
                    event
                    for event in events
                    if event.get("ph") == "X"
                    and event.get("pid") == pid
                    and (pid, event.get("tid")) in module_lanes
                ),
                key=lambda event: event["ts"],
            )
            if not modules:
                continue
            if len(modules) != iters:
                raise RuntimeError(
                    f"{trace_file}: {device} has {len(modules)} module events; "
                    f"expected {iters}. Refusing incomplete device timing."
                )
            per_device[f"{trace_file.name}:{device}"] = [
                float(event["dur"]) for event in modules
            ]
    if not per_device:
        raise RuntimeError(f"No TPU XLA Modules events found in {profile_dir}")
    # Calls are synchronized before the next iteration, so device order aligns.
    samples = [max(values[i] for values in per_device.values()) for i in range(iters)]
    return samples, {name: len(values) for name, values in per_device.items()}


def wait_for_profile_slot(logdir: Path) -> None:
    # Avoid same-second profiler sessions overwriting each other in this logdir.
    sessions = logdir / "plugins" / "profile"
    while (sessions / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")).exists():
        time.sleep(0.05)


def benchmark(fn, x, *, logdir: Path, name: str, warmup=10, iters=50):
    for _ in range(warmup):
        result = fn(x)
    jax.block_until_ready(result)  # 等待编译和预热执行完成

    logdir.mkdir(parents=True, exist_ok=True)
    wait_for_profile_slot(logdir)
    before = set((logdir / "plugins" / "profile").glob("*"))
    with jax.profiler.trace(str(logdir)):
        for step in range(iters):
            with jax.profiler.StepTraceAnnotation(name, step_num=step):
                result = fn(x)
                jax.block_until_ready(result)  # JAX 异步执行，计时前必须同步
    sessions = set((logdir / "plugins" / "profile").glob("*")) - before
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one new profile session under {logdir}")
    profile_dir = sessions.pop()
    samples, devices = device_samples(profile_dir, iters)
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "stdev_us": statistics.pstdev(samples),
        "samples_us": samples,
        "device_event_counts": devices,
        "profile_dir": str(profile_dir),
        "scope": "TPU XLA Modules; per-call maximum across active devices",
    }


logdir = artifact_dir / "xprof"
measurements = {}
for name, fn in (("pallas_select", select), ("xla_top_k", baseline)):
    print(f"Starting {name} device timing (warmup, then 50 samples)...", flush=True)
    measurements[name] = {
        "rows": rows,
        "experts": experts,
        "topk": topk,
        "block_rows": block_rows,
        "kernel_file": str(kernel_path),
        "kernel_sha256": kernel_sha256,
        "correctness": "exact_match_with_xla_top_k",
        **benchmark(fn, scores, name=name, logdir=logdir),
    }
    median_us = measurements[name]["median_us"]
    print(f"{name} device median: {median_us:.3f} us", flush=True)
report_path = artifact_dir / "device_times.json"
report_path.write_text(json.dumps(measurements, indent=2))
shutil.copy2(
    report_path, artifact_dir / f"device_times_{kernel_path.stem}_tokens_{rows}.json"
)
if kernel_path.stem == "router_ops":
    shutil.copy2(report_path, artifact_dir / f"device_times_tokens_{rows}.json")
print(f"Device timing report: {report_path}", flush=True)


# benchmark() has already compiled and warmed both functions outside the trace.
opts = jax.profiler.ProfileOptions()
opts.advanced_configuration = {
    "tpu_enable_periodic_counter_sampling": True,
    "tpu_tc_perf_counter_sampling_options": (
        "interval_us:1 scaling:0 counter_size_bits:1 "
        "indices:1 indices:3 indices:4 indices:10 indices:11 "
        "indices:31 indices:32 indices:33 indices:34 indices:35 "
        "indices:37 indices:38 indices:56 indices:57 indices:58 "
        "indices:73 indices:74 indices:75 indices:105"
    ),
    "num_tensor_cores_to_trace_per_device": 1,
}
logdir = artifact_dir / "xprof"
print(f"Starting XProf counter capture; output: {logdir}", flush=True)
wait_for_profile_slot(logdir)
with jax.profiler.trace(str(logdir), profiler_options=opts):
    for name, fn in (("pallas_select", select), ("xla_top_k", baseline)):
        for step in range(50):
            with jax.profiler.StepTraceAnnotation(name, step_num=step):
                jax.block_until_ready(fn(scores))

print("XProf capture and export completed.", flush=True)

# Keep the standard session tree and flat host files for Falcon /view.
profile_dir = max((logdir / "plugins" / "profile").iterdir(), key=lambda p: p.name)
xplanes = list(profile_dir.glob("*.xplane.pb"))
if not xplanes:
    raise RuntimeError(f"XProf produced no xplane files in {profile_dir}")
for profile_file in [*xplanes, *profile_dir.glob("*.trace.json.gz")]:
    shutil.copy2(profile_file, artifact_dir / profile_file.name)
print(f"LLO: {llo_dir}")
print(f"HLO: {hlo_dir}")
print(f"XProf: {logdir}")
