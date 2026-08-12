# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed custom-op checks for unified PCP packed KV layouts."""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.multichip

WORLD_SIZE = 8
PAGE_SIZE = 128
LOCAL_TOKENS = 128
GLOBAL_TOKENS = WORLD_SIZE * LOCAL_TOKENS
HEAD_DIM = 128
LAUNCH_TIMEOUT_SECONDS = 420

CASES = (
    {
        "name": "bf16_split_interleave",
        "kv_dtype": "bfloat16",
        "num_kv_heads": 1,
        "q_per_kv": 1,
        "k_scale": None,
        "v_scale": None,
        "interleave_size": 32,
    },
    {
        "name": "bf16_single",
        "kv_dtype": "bfloat16",
        "num_kv_heads": 1,
        "q_per_kv": 1,
        "k_scale": None,
        "v_scale": None,
    },
    {
        "name": "fp8_mqa",
        "kv_dtype": "float8_e4m3fn",
        "num_kv_heads": 1,
        "q_per_kv": 2,
        "k_scale": 1.0,
        "v_scale": 1.0,
    },
    {
        "name": "fp8_mha",
        "kv_dtype": "float8_e4m3fn",
        "num_kv_heads": 2,
        "q_per_kv": 1,
        "k_scale": 1.0,
        "v_scale": 1.0,
    },
    {
        "name": "fp8_gqa_odd_kv_heads",
        "kv_dtype": "float8_e4m3fn",
        "num_kv_heads": 3,
        "q_per_kv": 2,
        "k_scale": 1.0,
        "v_scale": 1.0,
    },
)


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
        from torch_tpu._internal.distributed.launchers.singlehost_wrapper import \
            prepare_tpu_environment
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

    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                   "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
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
        raise AssertionError("Unified PCP workers timed out. Output tail:\n"
                             f"{output[-12000:]}") from exc
    return subprocess.CompletedProcess(process.args, process.returncode,
                                       output, "")


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
        device_id for device_id in global_device_ids
        if device_id not in devices_by_id
    ]
    if missing:
        raise RuntimeError(f"JAX does not expose worker TPU ids {missing}; "
                           f"available={sorted(devices_by_id)}")
    devices = np.asarray(
        [devices_by_id[device_id] for device_id in global_device_ids])
    return jax.sharding.Mesh(devices, axis_names=(axis_name, ))


def _torch_dtype(torch, name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"Unsupported dtype name: {name}")


def _kv_cache_shape(num_kv_heads: int, kv_packing: int) -> tuple[int, ...]:
    packed_kv_groups = math.ceil((2 * num_kv_heads) / kv_packing)
    return (1, PAGE_SIZE, packed_kv_groups, kv_packing, HEAD_DIM)


def _make_case_inputs(torch, case: dict[str, object]):
    num_kv_heads = int(case["num_kv_heads"])
    q_per_kv = int(case["q_per_kv"])
    num_q_heads = num_kv_heads * q_per_kv
    kv_dtype = _torch_dtype(torch, str(case["kv_dtype"]))
    kv_packing = 4 if case["kv_dtype"] == "float8_e4m3fn" else 2

    q = torch.zeros((LOCAL_TOKENS, num_q_heads, HEAD_DIM),
                    dtype=torch.bfloat16,
                    device="tpu")
    k = torch.zeros((LOCAL_TOKENS, num_kv_heads, HEAD_DIM),
                    dtype=torch.bfloat16,
                    device="tpu")
    v = torch.full((LOCAL_TOKENS, num_kv_heads, HEAD_DIM),
                   2.0,
                   dtype=torch.bfloat16,
                   device="tpu")
    kv_cache = torch.zeros(_kv_cache_shape(num_kv_heads, kv_packing),
                           dtype=kv_dtype,
                           device="tpu")
    seq_lens = torch.tensor([GLOBAL_TOKENS], dtype=torch.int32, device="tpu")
    block_tables = torch.tensor([0], dtype=torch.int32, device="tpu")
    query_start_loc = torch.tensor([0, GLOBAL_TOKENS],
                                   dtype=torch.int32,
                                   device="tpu")
    distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device="tpu")
    return kv_cache, (q, k, v, seq_lens, block_tables, query_start_loc,
                      distribution)


def _run_case(torch, sync, mesh, case: dict[str, object]) -> dict[str, object]:
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
        PCP_STREAMING_RPA_INPUT_PARTITION_SPECS, make_pcp_streaming_rpa_kernel,
        pcp_streaming_jax_op)

    kv_cache, args = _make_case_inputs(torch, case)
    sync.synchronize([kv_cache, *args], wait=True)
    entry = make_pcp_streaming_rpa_kernel(
        q_scale=None,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        mesh=mesh,
        sliding_window=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=int(case.get("interleave_size",
                                                 PAGE_SIZE)),
        q_block_size=LOCAL_TOKENS,
        q_compute_size=64,
    )
    attention = pcp_streaming_jax_op(
        f"pcp_unified_head_matrix::{case['name']}",
        entry,
        donate_argnums=(0, ),
        mesh=mesh,
        input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
    )
    new_kv_cache, output = attention(kv_cache, *args)
    kv_cache.copy_(new_kv_cache)
    sync.synchronize([kv_cache, output], wait=True)
    output_cpu = output.cpu().float()
    max_abs = float((output_cpu - 2.0).abs().max().item())
    if max_abs > 0.03:
        raise AssertionError(f"{case['name']} output max_abs={max_abs}")
    return {
        "name": case["name"],
        "output_max_abs": max_abs,
        "output_shape": tuple(output.shape),
        "kv_cache_shape": tuple(kv_cache.shape),
    }


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
        torch.empty((1, ), device="tpu").cpu()
        import jax

        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import \
            PCP_AXIS_NAME

        global_device_ids = _gather_global_device_ids(torch, dist,
                                                      tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids, PCP_AXIS_NAME)
        case_results = [_run_case(torch, sync, mesh, case) for case in CASES]
        result = {"rank": rank, "cases": case_results}
    except BaseException as exc:
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


@pytest.mark.nightly
def test_pcp_attention_executes_unified_bf16_fp8_single_multi_head_matrix(
        tmp_path):
    result_dir = tmp_path / "pcp_unified_head_matrix"
    result_dir.mkdir()
    completed = _run_worker_group(result_dir)
    combined_output = completed.stdout

    results = []
    for rank in range(WORLD_SIZE):
        result_path = result_dir / f"rank_{rank}.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            results.append({"rank": rank, "error": "missing worker result"})
    errors = [result for result in results if "error" in result]
    assert completed.returncode == 0 and not errors, (
        f"Unified PCP matrix failure: returncode={completed.returncode}, "
        f"results={errors}\nworker output tail:\n{combined_output[-12000:]}")

    summary = {
        case["name"]:
        max(result["cases"][idx]["output_max_abs"] for result in results)
        for idx, case in enumerate(CASES)
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    print("Unified PCP head/dtype matrix summary:",
          json.dumps(summary, sort_keys=True))


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
