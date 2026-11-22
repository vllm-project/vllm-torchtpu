# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TPU_SAMPLER_VOCAB_PARALLEL_TARGET on real devices: the shard verify,
compiled with its TP collectives, at the served verify shape.

Eight ranks on one host (4 chips x 2 cores) form a TP group and each runs
the compiled `vocab_parallel_spec_verify` on its 19,360-column shard of
bf16[160, 154880] logits (40 requests with MTP k=3, 48 padded), for
synthetic and real acceptance. Checks:

  1. It lowers and runs: the softmax-statistics all-gather, the radix
     all-reduces, the draft-probability all-reduce and the pair all-gather
     inside one compiled program.
  2. Every rank returns the same [48, 4] output, the property the ranks'
     shared decode state needs.
  3. Against the single-device rejection sampler on CPU fed the same
     uniforms: each request keeps the same number of tokens (the accept
     decisions), and at most 3 of 40 requests differ in a token, which
     device and CPU arithmetic can flip where two race values nearly tie.
  4. TPU_SAMPLER_VP_UNIFORM_SLICE (the shard's uniform columns by a dynamic
     slice of the last dimension) returns the same output, bit for bit.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
VOCAB = 154880
CASE = [3] * 40
PADDED_NUM_REQS = 48
LAUNCH_TIMEOUT_SECONDS = 1800


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
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
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
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    env.pop("LIBTPU_INIT_ARGS", None)
    return env


def _inputs():
    """The chunk, built the same way on every rank (and for the reference)."""
    from tests.runner import test_vocab_parallel_target as cpu_tests

    cpu_tests.TOKEN_PADDINGS = [16, 32, 64, 128, 160, 192]
    logits, md, temps, top_p = cpu_tests._chunk(
        CASE, 7, VOCAB, padded_num_reqs=PADDED_NUM_REQS
    )
    uniforms = cpu_tests._uniforms(md, VOCAB, 7)
    return cpu_tests, logits, md, temps, top_p, uniforms


def _measure(rank: int, group) -> dict[str, object]:
    import torch

    from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
    from vllm_torchtpu.layers.adapter.sample.vocab_parallel_verify import (
        vocab_parallel_spec_verify,
    )
    from vllm_torchtpu.runner.tpu_runner import SAMPLING_EPS

    cpu_tests, logits, md, temps, top_p, uniforms = _inputs()
    accept_u, recover_u, bonus_u = uniforms
    width = VOCAB // WORLD_SIZE
    dev = torch.device("tpu")
    shard = logits[:, rank * width : (rank + 1) * width].contiguous().to(dev)
    args = [
        shard,
        md.target_logits_indices.to(dev),
        md.bonus_logits_indices.to(dev),
        md.draft_token_ids.to(dev),
        md.draft_lengths.to(dev),
        md.segment_ids.to(dev),
        md.group_indices.to(dev),
        temps.to(dev),
        top_p.to(dev),
        accept_u.to(dev),
        recover_u.to(dev),
        bonus_u.to(dev),
        torch.tensor(rank * width, dtype=torch.int32).to(dev),
    ]
    rates = torch.tensor(
        rs.unconditional_to_conditional_rates(cpu_tests.RATES), dtype=torch.float32
    ).to(dev)
    compiled = torch.compile(
        vocab_parallel_spec_verify, backend="tpu", fullgraph=True, dynamic=False
    )
    out: dict[str, object] = {"rank": rank}
    for synthetic in (True, False):
        for uniform_slice in (False, True):
            tail = (
                width,
                WORLD_SIZE,
                3,
                rates if synthetic else None,
                synthetic,
                3,
                SAMPLING_EPS,
                0,
                uniform_slice,
            )
            t0 = time.perf_counter()
            tokens = compiled(*args, *tail).cpu()
            first_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(20):
                res = compiled(*args, *tail)
            res.cpu()
            mode = ("synthetic" if synthetic else "real") + (
                "_slice" if uniform_slice else ""
            )
            out[mode] = {
                "tokens": tokens.tolist(),
                "first_call_s": round(first_s, 2),
                "wall_us_per_call": round((time.perf_counter() - t0) / 20 * 1e6, 1),
            }
    return out


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed import parallel_state
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        world_group = get_world_group()
        group = init_model_parallel_group(
            [list(range(WORLD_SIZE))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="tp",
        )
        # tensor_model_parallel_all_reduce / all_gather read the TP group.
        parallel_state._TP = group
        result = _measure(rank, group)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        result_path.write_text(json.dumps(result), encoding="utf-8")
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _run_worker_group_once(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[2],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def test_shard_verify_runs_compiled_on_eight_ranks(tmp_path):
    import torch

    completed = _run_worker_group_once(tmp_path)
    if "PjRtClient is not initialized" in (completed.stdout or ""):
        time.sleep(60)
        completed = _run_worker_group_once(tmp_path)
    results = []
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    assert completed.returncode == 0 and not errors, (
        f"worker failure: returncode={completed.returncode}, errors={errors}\n"
        f"output tail:\n{completed.stdout[-12000:]}"
    )
    cpu_tests, logits, md, temps, top_p, uniforms = _inputs()
    n = len(CASE)
    for mode in ("synthetic", "real"):
        got = [torch.tensor(r[mode]["tokens"]) for r in results]
        for g in got[1:]:
            assert torch.equal(g, got[0]), mode
        ref = cpu_tests._reference(
            logits, md, temps, top_p, uniforms, mode == "synthetic"
        )
        filled_got = (got[0][:n] != -1).sum(dim=1)
        filled_ref = (ref[:n] != -1).sum(dim=1)
        differ = int((got[0][:n] != ref[:n]).any(dim=1).sum())
        print(
            f"VPT_MULTICHIP {mode}: requests differing {differ}/{n}, "
            f"wall us/call {[r[mode]['wall_us_per_call'] for r in results]}, "
            f"first call s {results[0][mode]['first_call_s']}",
            flush=True,
        )
        assert torch.equal(filled_got, filled_ref), mode
        assert differ <= 3, (mode, differ)
        sliced = [torch.tensor(r[mode + "_slice"]["tokens"]) for r in results]
        print(
            f"VPT_MULTICHIP {mode}_slice: wall us/call "
            f"{[r[mode + '_slice']['wall_us_per_call'] for r in results]}",
            flush=True,
        )
        for g in sliced:
            assert torch.equal(g, got[0]), mode + "_slice"


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        raise ValueError("expected --worker")
    _run_worker(args.result_dir)


if __name__ == "__main__":
    _main()
