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
"""Regression test for PCP attention barriers across compiled executions.

The production failure happened when a fixed Pallas collective id was reused by
separately compiled prefill and decode graphs. An uneven PCP schedule allowed a
fast rank to enter the next graph while another rank was still finishing the
previous barrier generation, which could corrupt ring state or halt the TPU.

This test drives the public PCP attention adapter from eight TorchTPU workers.
It runs 28 independent layer caches through a 432-token prefill followed by a
one-token decode, then validates every cache and the final attention output.
"""

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
PAGE_SIZE = 256
LOCAL_PADDED_TOKENS = 2048
Q_BLOCK_SIZE = 512
PREFILL_TOKENS = 432
MAX_MODEL_LEN = 16 * 1024
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 28
LAUNCH_TIMEOUT_SECONDS = 240


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
        # A previous test or launcher may leave rendezvous values for a
        # different TPU slice. This worker group needs a fresh local launch.
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
        raise AssertionError(
            "PCP workers timed out and were terminated. Output tail:\n"
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


def _make_inputs(torch, sync, rank: int):

    def make_qkv(valid_tokens: int, value: float):
        q = torch.zeros(
            (LOCAL_PADDED_TOKENS, NUM_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="tpu",
        )
        k = torch.zeros(
            (LOCAL_PADDED_TOKENS, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="tpu",
        )
        v = torch.zeros_like(k)
        if valid_tokens:
            v[:valid_tokens].fill_(value)
        return q, k, v

    prefill_valid_tokens = (PAGE_SIZE if rank == 0 else PREFILL_TOKENS -
                            PAGE_SIZE if rank == 1 else 0)
    decode_valid_tokens = 1 if rank == 1 else 0
    prefill_q, prefill_k, prefill_v = make_qkv(prefill_valid_tokens, 1.0)
    decode_q, decode_k, decode_v = make_qkv(decode_valid_tokens, 9.0)

    local_cache_blocks = MAX_MODEL_LEN // (WORLD_SIZE * PAGE_SIZE)
    cache_shape = (
        local_cache_blocks,
        PAGE_SIZE,
        NUM_KV_HEADS,
        2,
        HEAD_DIM,
    )
    caches = tuple(
        torch.zeros(cache_shape, dtype=torch.bfloat16, device="tpu")
        for _ in range(NUM_LAYERS))
    block_tables = torch.arange(local_cache_blocks,
                                dtype=torch.int32,
                                device="tpu")
    prefill_seq_lens = torch.tensor([PREFILL_TOKENS],
                                    dtype=torch.int32,
                                    device="tpu")
    decode_seq_lens = torch.tensor([PREFILL_TOKENS + 1],
                                   dtype=torch.int32,
                                   device="tpu")
    prefill_query_start_loc = torch.tensor([0, PREFILL_TOKENS],
                                           dtype=torch.int32,
                                           device="tpu")
    decode_query_start_loc = torch.tensor([0, 1],
                                          dtype=torch.int32,
                                          device="tpu")
    prefill_distribution = torch.tensor([0, 0, 1],
                                        dtype=torch.int32,
                                        device="tpu")
    decode_distribution = torch.tensor([1, 1, 1],
                                       dtype=torch.int32,
                                       device="tpu")
    sync.synchronize([
        *caches,
        prefill_q,
        prefill_k,
        prefill_v,
        decode_q,
        decode_k,
        decode_v,
        block_tables,
        prefill_seq_lens,
        decode_seq_lens,
        prefill_query_start_loc,
        decode_query_start_loc,
        prefill_distribution,
        decode_distribution,
    ],
                     wait=True)
    prefill_args = (prefill_q, prefill_k, prefill_v, prefill_seq_lens,
                    block_tables, prefill_query_start_loc,
                    prefill_distribution)
    decode_args = (decode_q, decode_k, decode_v, decode_seq_lens, block_tables,
                   decode_query_start_loc, decode_distribution)
    return caches, prefill_args, decode_args


def _build_compiled_steps(torch, mesh):
    from torch_tpu._internal import compile as tpu_compile

    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
        PCP_STREAMING_RPA_INPUT_PARTITION_SPECS, make_pcp_streaming_rpa_kernel,
        pcp_streaming_jax_op)

    entry = make_pcp_streaming_rpa_kernel(
        q_scale=None,
        k_scale=None,
        v_scale=None,
        mesh=mesh,
        sliding_window=None,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
        soft_cap=None,
        skip_kv_update=False,
        cp_kv_cache_interleave_size=PAGE_SIZE,
        max_model_len=MAX_MODEL_LEN,
        q_block_size=Q_BLOCK_SIZE,
        q_compute_size=128,
    )
    attention = pcp_streaming_jax_op(
        "pcp_barrier_regression::streaming_attention",
        entry,
        donate_argnums=(0, ),
        mesh=mesh,
        input_partition_specs=PCP_STREAMING_RPA_INPUT_PARTITION_SPECS,
    )

    def run_layers(caches, q, k, v, seq_lens, block_tables, query_start_loc,
                   distribution):
        output = q
        for layer_cache in caches:
            new_cache, output = attention(
                layer_cache,
                output,
                k,
                v,
                seq_lens,
                block_tables,
                query_start_loc,
                distribution,
            )
            layer_cache.copy_(new_cache)
        return output

    def prefill(caches, q, k, v, seq_lens, block_tables, query_start_loc,
                distribution):
        return run_layers(caches, q, k, v, seq_lens, block_tables,
                          query_start_loc, distribution)

    def decode(caches, q, k, v, seq_lens, block_tables, query_start_loc,
               distribution):
        return run_layers(caches, q, k, v, seq_lens, block_tables,
                          query_start_loc, distribution)

    backend = tpu_compile.TpuBackend()
    return (
        torch.compile(prefill, fullgraph=True, dynamic=False, backend=backend),
        torch.compile(decode, fullgraph=True, dynamic=False, backend=backend),
    )


def _validate_outputs(torch, caches, output,
                      rank: int) -> dict[str, float | int]:
    expected_cache = torch.zeros_like(caches[0], device="cpu").float()
    if rank == 0:
        expected_cache[0, :PAGE_SIZE, :, 1, :].fill_(1.0)
    elif rank == 1:
        prefill_on_rank = PREFILL_TOKENS - PAGE_SIZE
        expected_cache[0, :prefill_on_rank, :, 1, :].fill_(1.0)
        expected_cache[0, prefill_on_rank, :, 1, :].fill_(9.0)

    cache_max_abs = 0.0
    for cache in caches:
        cache_max_abs = max(
            cache_max_abs,
            float((cache.cpu().float() - expected_cache).abs().max().item()),
        )
    output_cpu = output.cpu().float()
    output_max_abs = 0.0
    if rank == 1:
        expected_output = (PREFILL_TOKENS + 9.0) / (PREFILL_TOKENS + 1)
        output_max_abs = float(
            (output_cpu[0] - expected_output).abs().max().item())
    if cache_max_abs != 0.0 or output_max_abs > 0.01:
        raise AssertionError(f"rank={rank} cache_max_abs={cache_max_abs} "
                             f"output_max_abs={output_max_abs}")
    return {
        "decode_owner": int(rank == 1),
        "cache_max_abs": cache_max_abs,
        "output_max_abs": output_max_abs,
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
        cpu_group = dist.new_group(ranks=list(range(WORLD_SIZE)),
                                   backend="gloo")

        # Match vLLM startup: initialize TorchTPU before querying JAX topology.
        torch.empty((1, ), device="tpu").cpu()
        import jax

        from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import \
            PCP_AXIS_NAME

        global_device_ids = _gather_global_device_ids(torch, dist,
                                                      tpu_distributed)
        mesh = _build_mesh(jax, np, global_device_ids, PCP_AXIS_NAME)
        caches, prefill_args, decode_args = _make_inputs(torch, sync, rank)
        prefill, decode = _build_compiled_steps(torch, mesh)

        dist.barrier(group=cpu_group)
        hidden = prefill(caches, *prefill_args)
        sync.synchronize([*caches, hidden], wait=True)
        dist.barrier(group=cpu_group)
        output = decode(caches, *decode_args)
        sync.synchronize([*caches, output], wait=True)
        metrics = _validate_outputs(torch, caches, output, rank)
        dist.barrier(group=cpu_group)
        result = {
            "rank": rank,
            "global_device_id": int(tpu_distributed.global_device_id()),
            "mesh_device_ids": global_device_ids,
            "num_layers": NUM_LAYERS,
            **metrics,
        }
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


def test_consecutive_attention_layers_tolerate_rank_schedule_skew(tmp_path):
    """Uneven prefill/decode schedules must not corrupt the next PCP call."""
    result_dir = tmp_path / "pcp_entry_barrier"
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
        f"PCP worker failure: returncode={completed.returncode}, "
        f"results={errors}\nworker output tail:\n{combined_output[-12000:]}")
    assert all(result["cache_max_abs"] == 0.0 for result in results)
    assert all(result["output_max_abs"] <= 0.01 for result in results)


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
