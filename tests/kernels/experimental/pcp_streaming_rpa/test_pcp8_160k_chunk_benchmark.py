# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-397B PCP8 long-chunk RPA layout benchmark.

This opt-in TPU benchmark reproduces one full-attention layer for a single
request with 128K cached tokens followed by a 32K prefill chunk.  PCP8 assigns
4K query rows and 16K history rows to every rank.

The serving geometry is intentionally copied from
``tpu_benchmark_daily/scripts/start_prefill_server.sh --config pcp8`` and its
latest startup log:

* DP1 / PCP8 / TP1;
* 32 Q heads, 2 KV heads and head dimension 256 (Qwen3.5-397B);
* FP8 KV cache using HEAD_ALONG_SUBLANE by default, with an opt-in
  SEQ_ALONG_LANE comparison;
* 32,768 max batched tokens and 64 request metadata slots;
* manager block size 4,352, split into 256-token CUSTOM kernel pages;
* 256-token PCP KV-cache interleave; and
* the production PCP kernel defaults: q_block_size=q_compute_size=512.

The manager block size is recorded in the result but is not directly visible
to RPA: vLLM expands manager blocks into 256-token kernel pages before building
attention metadata.

Run with:

  TPU_RUN_PCP8_160K_BENCHMARK=1 pytest -s \
    tests/kernels/experimental/pcp_streaming_rpa/\
test_pcp8_160k_chunk_benchmark.py

Set ``TPU_PCP8_160K_RESULT_DIR`` to retain per-rank samples and ``summary.json``
outside pytest's temporary directory.

Set ``TPU_PCP8_160K_PROFILE_DIR`` to capture three already-compiled attention
steps from all eight ranks into one merged XProf/TensorBoard run.

Set TPU_PCP8_160K_KV_LAYOUT=SEQ_ALONG_LANE to profile the new layout; the
default remains HEAD_ALONG_SUBLANE.
"""

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.nightly]

WORLD_SIZE = 8
MAX_NUM_BATCHED_TOKENS = 32 * 1024
MAX_NUM_SEQS = 64
MAX_MODEL_LEN = 256 * 1024
HISTORY_TOKENS = 128 * 1024
CHUNK_TOKENS = 32 * 1024
TOTAL_KV_TOKENS = HISTORY_TOKENS + CHUNK_TOKENS
LOCAL_QUERY_TOKENS = CHUNK_TOKENS // WORLD_SIZE
LOCAL_HISTORY_TOKENS = HISTORY_TOKENS // WORLD_SIZE

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 256
KV_PACKING = 4
PACKED_KV_GROUPS = math.ceil(2 * NUM_KV_HEADS / KV_PACKING)

MANAGER_BLOCK_SIZE = 4352
KERNEL_PAGE_SIZE = 256
PCP_INTERLEAVE_SIZE = 256
Q_BLOCK_SIZE = 512
Q_COMPUTE_SIZE = 512

MAX_LOCAL_PAGES_PER_SEQ = MAX_MODEL_LEN // (WORLD_SIZE * KERNEL_PAGE_SIZE)
ACTIVE_LOCAL_PAGES = TOTAL_KV_TOKENS // (WORLD_SIZE * KERNEL_PAGE_SIZE)
WARMUP_STEPS = 2
BENCHMARK_STEPS = 10
PROFILE_STEPS = 3
LAUNCH_TIMEOUT_SECONDS = 900
KV_LAYOUT_ENV = "TPU_PCP8_160K_KV_LAYOUT"


def _selected_kv_layout_name() -> str:
    layout = os.getenv(KV_LAYOUT_ENV, "HEAD_ALONG_SUBLANE").strip().upper()
    if layout not in ("HEAD_ALONG_SUBLANE", "SEQ_ALONG_LANE"):
        raise ValueError(f"Unsupported {KV_LAYOUT_ENV}={layout!r}.")
    return layout


def _require_benchmark() -> None:
    if os.getenv("TPU_RUN_PCP8_160K_BENCHMARK") != "1":
        pytest.skip("set TPU_RUN_PCP8_160K_BENCHMARK=1 to run the PCP8 benchmark")


def _worker_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
            prepare_tpu_environment,
        )
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")

    prepared_keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    original_values = {key: os.environ.get(key) for key in prepared_keys}
    try:
        for key in prepared_keys:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    # Match the TorchTPU execution controls used by the PCP service launcher.
    env.setdefault("PJRT_DEVICE", "TPU")
    env.setdefault("VLLM_TARGET_DEVICE", "tpu")
    env.setdefault("VLLM_PLUGINS", "torchtpu")
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("TPU_PARALLEL_PRECOMPILE", "1")
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", "disabled")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    if env.get("TPU_PCP8_160K_PROFILE_DIR"):
        env["TPU_PCP8_PROFILE_SESSION_KEY"] = (
            f"pcp8_160k_{os.getpid()}_{time.time_ns()}"
        )
    return env


def _run_worker_group(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[4],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        raise AssertionError(
            f"PCP8 benchmark workers timed out. Output tail:\n{output[-12000:]}"
        ) from exc
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _gather_global_device_ids(torch, dist, tpu_distributed) -> tuple[int, ...]:
    local_id = torch.tensor(
        [int(tpu_distributed.global_device_id())],
        dtype=torch.int32,
        device="tpu",
    )
    gathered = [torch.empty_like(local_id) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, local_id)
    return tuple(int(tensor.cpu().item()) for tensor in gathered)


def _build_mesh(jax, np, global_device_ids: tuple[int, ...], axis_name: str):
    devices_by_id = {int(device.id): device for device in jax.devices()}
    missing = [
        device_id for device_id in global_device_ids if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(
            f"JAX does not expose worker TPU ids {missing}; "
            f"available={sorted(devices_by_id)}"
        )
    devices = np.asarray([devices_by_id[device_id] for device_id in global_device_ids])
    return jax.sharding.Mesh(devices, axis_names=(axis_name,))


def _make_inputs(torch):
    kv_layout = _selected_kv_layout_name()
    query = torch.zeros(
        (LOCAL_QUERY_TOKENS, NUM_Q_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="tpu",
    )
    key = torch.zeros(
        (LOCAL_QUERY_TOKENS, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="tpu",
    )
    value = torch.ones_like(key)
    if kv_layout == "SEQ_ALONG_LANE":
        kv_cache_shape = (
            MAX_LOCAL_PAGES_PER_SEQ,
            2 * NUM_KV_HEADS,
            HEAD_DIM // KV_PACKING,
            KV_PACKING,
            KERNEL_PAGE_SIZE,
        )
    else:
        kv_cache_shape = (
            MAX_LOCAL_PAGES_PER_SEQ,
            KERNEL_PAGE_SIZE,
            PACKED_KV_GROUPS,
            KV_PACKING,
            HEAD_DIM,
        )
    kv_cache = torch.zeros(kv_cache_shape, dtype=torch.float8_e4m3fn, device="tpu")

    # Production pads request metadata to max_num_seqs=64 even though only the
    # first request is live in this benchmark.
    seq_lens = torch.zeros((MAX_NUM_SEQS,), dtype=torch.int32, device="tpu")
    seq_lens[0] = TOTAL_KV_TOKENS
    query_start_loc = torch.full(
        (MAX_NUM_SEQS + 1,), CHUNK_TOKENS, dtype=torch.int32, device="tpu"
    )
    query_start_loc[0] = 0
    request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device="tpu")

    block_tables = torch.zeros(
        (MAX_NUM_SEQS, MAX_LOCAL_PAGES_PER_SEQ),
        dtype=torch.int32,
        device="tpu",
    )
    block_tables[0] = torch.arange(
        MAX_LOCAL_PAGES_PER_SEQ, dtype=torch.int32, device="tpu"
    )
    return kv_cache, (
        query,
        key,
        value,
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
    )


def _expected_local_output(torch, rank: int):
    # Query ownership is rank-major in 256-token chunks. Q=K=0 makes all
    # attention logits equal, history V is zero and current V is one, so row i
    # is exactly (new tokens visible so far) / (all tokens visible so far).
    offsets = []
    cycle = WORLD_SIZE * PCP_INTERLEAVE_SIZE
    chunk_start = rank * PCP_INTERLEAVE_SIZE
    while chunk_start < CHUNK_TOKENS:
        chunk_end = min(chunk_start + PCP_INTERLEAVE_SIZE, CHUNK_TOKENS)
        offsets.extend(range(chunk_start, chunk_end))
        chunk_start += cycle
    assert len(offsets) == LOCAL_QUERY_TOKENS
    offsets_t = torch.tensor(offsets, dtype=torch.float32)
    return (offsets_t + 1.0) / (HISTORY_TOKENS + offsets_t + 1.0)


def _profile_attention(
    torch, dist, cpu_group, sync, attention, kv_cache, args, rank: int
):
    configured = os.getenv("TPU_PCP8_160K_PROFILE_DIR", "").strip()
    if not configured:
        return kv_cache, None, []

    from torch_tpu._internal.profiler import TpuProfilerConfig

    from vllm_torchtpu import profiler_trace

    profile_dir = str(Path(configured).resolve())
    session_key = os.environ["TPU_PCP8_PROFILE_SESSION_KEY"]
    capture_dir = profiler_trace.rank_capture_dir(profile_dir, rank)
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    canonical_ts = profiler_trace.resolve_canonical_dst_ts(
        profile_dir, rank, session_key=session_key
    )
    Path(capture_dir).mkdir(parents=True, exist_ok=True)

    handler = torch.profiler.tensorboard_trace_handler(
        dir_name=capture_dir, use_gzip=True
    )
    config = TpuProfilerConfig(
        run_dir=capture_dir,
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=1,
        experimental_options={
            "tpu_trace_mode": "TRACE_COMPUTE",
            "tpu_num_sparse_cores_to_trace": 1,
            "tpu_num_sparse_core_tiles_to_trace": 1,
        },
    )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=handler,
        experimental_config=config,
    )

    profile_samples_ms = []
    output = None
    dist.barrier(group=cpu_group)
    profiler.__enter__()
    try:
        # Start only after every rank has armed its profiler. Warmup and
        # compilation completed before this helper was called.
        dist.barrier(group=cpu_group)
        for step in range(PROFILE_STEPS):
            start = time.perf_counter()
            with torch.profiler.record_function(f"pcp8_160k_attention_step_{step}"):
                kv_cache, output = attention(kv_cache, *args)
                sync.synchronize([kv_cache, output], wait=True)
            profile_samples_ms.append((time.perf_counter() - start) * 1e3)
    finally:
        profiler.__exit__(None, None, None)

    # All captures must be complete before ranks concurrently merge their
    # uniquely prefixed files into the shared TensorBoard run.
    dist.barrier(group=cpu_group)
    profiler_trace.merge_rank_capture(
        capture_dir,
        profile_dir,
        canonical_ts,
        rank,
        world_size=WORLD_SIZE,
    )
    dist.barrier(group=cpu_group)
    if rank == 0:
        profiler_trace.clear_canonical_ts_marker(profile_dir, session_key)
    return kv_cache, output, profile_samples_ms


def _run_worker(result_dir: Path) -> None:
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist
    from torch_tpu._internal import sync
    from torch_tpu._internal.distributed import tpu_distributed

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    initialized = False
    result: dict[str, object]
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        cpu_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="gloo")

        # Match vLLM startup: initialize TorchTPU before querying JAX topology.
        torch.empty((1,), device="tpu").cpu()
        import jax

        from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
            PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
            make_pcp_streaming_rpa_kernel,
            pcp_streaming_jax_op,
        )
        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
            PCP_AXIS_NAME,
        )

        global_device_ids = _gather_global_device_ids(torch, dist, tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids, PCP_AXIS_NAME)
        kv_cache, args = _make_inputs(torch)
        kv_layout = KVLayout[_selected_kv_layout_name()]
        sync.synchronize([kv_cache, *args], wait=True)

        entry = make_pcp_streaming_rpa_kernel(
            q_scale=None,
            k_scale=1.0,
            v_scale=1.0,
            mesh=mesh,
            sliding_window=None,
            sm_scale=1.0 / math.sqrt(HEAD_DIM),
            soft_cap=None,
            skip_kv_update=False,
            cp_kv_cache_interleave_size=PCP_INTERLEAVE_SIZE,
            q_block_size=Q_BLOCK_SIZE,
            q_compute_size=Q_COMPUTE_SIZE,
            kv_layout=kv_layout,
        )
        attention = pcp_streaming_jax_op(
            f"pcp8_160k_benchmark::qwen35_397b_h128k_q32k_{kv_layout.name.lower()}",
            entry,
            donate_argnums=(0,),
            mesh=mesh,
            input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
        )

        output = None
        for _ in range(WARMUP_STEPS):
            kv_cache, output = attention(kv_cache, *args)
            sync.synchronize([kv_cache, output], wait=True)

        kv_cache, profiled_output, profile_samples_ms = _profile_attention(
            torch, dist, cpu_group, sync, attention, kv_cache, args, rank
        )
        if profiled_output is not None:
            output = profiled_output

        samples_ms = []
        for _ in range(BENCHMARK_STEPS):
            dist.barrier(group=cpu_group)
            start = time.perf_counter()
            kv_cache, output = attention(kv_cache, *args)
            sync.synchronize([kv_cache, output], wait=True)
            samples_ms.append((time.perf_counter() - start) * 1e3)

        assert output is not None
        actual = output[:, 0, 0].cpu().float()
        expected = _expected_local_output(torch, rank)
        output_max_abs = float((actual - expected).abs().max().item())
        if output_max_abs > 0.01:
            raise AssertionError(f"rank={rank} output_max_abs={output_max_abs}")

        result = {
            "rank": rank,
            "global_device_id": int(tpu_distributed.global_device_id()),
            "mesh_device_ids": global_device_ids,
            "samples_ms": samples_ms,
            "median_ms": statistics.median(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "output_max_abs": output_max_abs,
            "profile_samples_ms": profile_samples_ms,
        }
    except BaseException as exc:  # noqa: BLE001 - report worker failures
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001 - best-effort worker cleanup
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _resolve_result_dir(tmp_path: Path) -> Path:
    configured = os.getenv("TPU_PCP8_160K_RESULT_DIR", "").strip()
    layout = _selected_kv_layout_name().lower()
    result_dir = (
        Path(configured).resolve()
        if configured
        else (tmp_path / f"pcp8_qwen35_397b_h128k_q32k_{layout}")
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _extract_profiled_kernel_summary() -> dict[str, object]:
    configured = os.getenv("TPU_PCP8_160K_PROFILE_DIR", "").strip()
    if not configured:
        return {}

    import jax

    prefixes = {
        "current": "%pcp_streaming_attention_current_state_page_groups_multi_head.",
        "history": "%pcp_streaming_attention_history_output_page_groups_multi_head.",
    }
    core_step_samples = []
    for xplane_path in sorted(Path(configured).rglob("*.xplane.pb")):
        profile = jax.profiler.ProfileData.from_file(str(xplane_path))
        for plane in profile.planes:
            if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
                continue
            events = {key: [] for key in prefixes}
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                for event in line.events:
                    for key, prefix in prefixes.items():
                        if event.name.startswith(prefix):
                            events[key].append(event)
            if not events["current"] and not events["history"]:
                continue
            for values in events.values():
                values.sort(key=lambda event: event.start_ns)
            assert len(events["current"]) == PROFILE_STEPS
            assert len(events["history"]) == PROFILE_STEPS
            for step, (current, history) in enumerate(
                zip(events["current"], events["history"])
            ):
                current_ms = current.duration_ns / 1e6
                history_ms = history.duration_ns / 1e6
                core_step_samples.append(
                    {
                        "xplane": xplane_path.name,
                        "device_plane": plane.name,
                        "step": step,
                        "current_state_ms": current_ms,
                        "history_output_ms": history_ms,
                        "kernel_sum_ms": current_ms + history_ms,
                    }
                )

    expected_samples = WORLD_SIZE * PROFILE_STEPS
    assert len(core_step_samples) == expected_samples, (
        f"expected {expected_samples} profiled core-steps, found "
        f"{len(core_step_samples)}"
    )
    current_samples = [
        float(sample["current_state_ms"]) for sample in core_step_samples
    ]
    history_samples = [
        float(sample["history_output_ms"]) for sample in core_step_samples
    ]
    sum_samples = [float(sample["kernel_sum_ms"]) for sample in core_step_samples]
    critical_step_samples = [
        max(
            float(sample["kernel_sum_ms"])
            for sample in core_step_samples
            if sample["step"] == step
        )
        for step in range(PROFILE_STEPS)
    ]
    return {
        "profiled_kernel_current_state_core_step_median_ms": statistics.median(
            current_samples
        ),
        "profiled_kernel_history_output_core_step_median_ms": statistics.median(
            history_samples
        ),
        "profiled_kernel_sum_core_step_median_ms": statistics.median(sum_samples),
        "profiled_kernel_critical_step_samples_ms": critical_step_samples,
        "profiled_kernel_critical_step_median_ms": statistics.median(
            critical_step_samples
        ),
        "profiled_kernel_core_step_samples": core_step_samples,
    }


def test_pcp8_qwen35_397b_history128k_chunk32k_layout_benchmark(tmp_path, capsys):
    _require_benchmark()
    result_dir = _resolve_result_dir(tmp_path)
    completed = _run_worker_group(result_dir)

    results = []
    for rank in range(WORLD_SIZE):
        result_path = result_dir / f"rank_{rank}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            results.append({"rank": rank, "error": "missing worker result"})
    errors = [result for result in results if "error" in result]
    assert completed.returncode == 0 and not errors, (
        f"PCP8 benchmark failure: returncode={completed.returncode}, "
        f"results={errors}\nworker output tail:\n{completed.stdout[-12000:]}"
    )

    # A distributed step completes when its slowest rank completes. Aggregate
    # sample-by-sample rank maxima instead of averaging independently timed
    # ranks.
    step_samples_ms = [
        max(float(result["samples_ms"][sample_idx]) for result in results)
        for sample_idx in range(BENCHMARK_STEPS)
    ]
    median_ms = statistics.median(step_samples_ms)
    ordered = sorted(step_samples_ms)
    p90_ms = ordered[math.ceil(0.9 * len(ordered)) - 1]
    summary = {
        "model": "Qwen3.5-397B-A17B-FP8",
        "layout": _selected_kv_layout_name(),
        "world_size": WORLD_SIZE,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": WORLD_SIZE,
        "tensor_parallel_size": 1,
        "num_q_heads": NUM_Q_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "kv_cache_dtype": "float8_e4m3fn",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "manager_block_size": MANAGER_BLOCK_SIZE,
        "kernel_page_size": KERNEL_PAGE_SIZE,
        "pcp_interleave_size": PCP_INTERLEAVE_SIZE,
        "q_block_size": Q_BLOCK_SIZE,
        "q_compute_size": Q_COMPUTE_SIZE,
        "history_tokens_global": HISTORY_TOKENS,
        "chunk_tokens_global": CHUNK_TOKENS,
        "history_tokens_per_rank": LOCAL_HISTORY_TOKENS,
        "query_tokens_per_rank": LOCAL_QUERY_TOKENS,
        "active_local_pages": ACTIVE_LOCAL_PAGES,
        "allocated_local_pages": MAX_LOCAL_PAGES_PER_SEQ,
        "warmup_steps": WARMUP_STEPS,
        "benchmark_steps": BENCHMARK_STEPS,
        "profile_steps": PROFILE_STEPS,
        "profile_dir": os.getenv("TPU_PCP8_160K_PROFILE_DIR") or None,
        "profile_step_samples_ms_by_rank": {
            str(result["rank"]): result["profile_samples_ms"] for result in results
        },
        "step_samples_ms": step_samples_ms,
        # These wall-time fields include host dispatch and scheduling. The
        # profiled device-kernel fields below are the primary performance
        # metric when profiling is enabled.
        "median_attention_call_wall_ms": median_ms,
        "min_attention_call_wall_ms": min(step_samples_ms),
        "p90_attention_call_wall_ms": p90_ms,
        "max_output_abs_error": max(
            float(result["output_max_abs"]) for result in results
        ),
    }
    summary.update(_extract_profiled_kernel_summary())
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with capsys.disabled():
        kernel_ms = summary.get("profiled_kernel_critical_step_median_ms")
        kernel_text = (
            f", device_kernel={float(kernel_ms):.3f} ms/layer"
            if kernel_ms is not None
            else ""
        )
        print(
            f"\nPCP8 Qwen3.5-397B {summary['layout']} RPA benchmark: "
            f"history=128K, chunk=32K, local_q=4K{kernel_text}, "
            f"call_wall_median={median_ms:.3f} ms, "
            f"call_wall_p90={p90_ms:.3f} ms, "
            f"samples={step_samples_ms}, result_dir={result_dir}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        _run_worker(args.result_dir)
        return
    raise ValueError("expected --worker")


if __name__ == "__main__":
    _main()
