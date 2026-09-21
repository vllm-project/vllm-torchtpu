# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Intra-chip latent projections on real devices.

The unit tests pin the arithmetic; this pins the three things only hardware can
answer, and each of them is a hard stop for the design:

  1. A 2-rank subgroup of the TP group can be created at all, and it pairs the
     cores that actually share a chip.
  2. ``all_gather`` over that subgroup lowers and runs -- vLLM routes TPU
     collectives through ``torch.ops.vllm.all_gather`` keyed on the group name,
     a path only ever exercised on the world/TP groups before.
  3. Both survive ``torch.compile``. The projections sit inside the compiled
     decode graph, so a collective that only works eagerly is no use.

Run on one host with 4 chips x 2 cores.
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
# The real K3 latent-projection shapes, scaled down only in the token count.
HIDDEN = 7168
LATENT = 3584
NUM_TOKENS = 2
LAUNCH_TIMEOUT_SECONDS = 600


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

    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    # Drop env leaked by an in-process engine run earlier in the pytest
    # session (e.g. tests/entrypoints/llm/test_generate_tp.py, which
    # collection orders before this file). env_override.py assembles
    # LIBTPU_INIT_ARGS at import; one of its additions,
    # --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes,
    # miscompiles this test's compiled GEMM+all-gather into garbage
    # (scripts/debug/ag_offload_flag_repro.py isolates it). The workers
    # re-import vllm_torchtpu themselves, but only after the device is
    # touched, so the leaked value is what their client init sees.
    env.pop("LIBTPU_INIT_ARGS", None)
    env["TPU_LATENT_PROJ_INTRA_CHIP_TP"] = "1"
    return env


def _measure(rank: int) -> dict[str, object]:
    """Form the chip group, then check a sharded projection against a full one."""
    import torch

    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group

    group = get_intra_chip_group()
    if group is None:
        raise AssertionError("no intra-chip group was formed")

    topology = get_chip_topology()
    chip_index, _ = topology.chip_of(rank)
    peers = sorted(group.ranks)

    # (1) The group must be this rank's chip, not an arbitrary pair.
    expected = sorted(
        r for r in range(WORLD_SIZE) if topology.chip_of(r)[0] == chip_index
    )
    if peers != expected:
        raise AssertionError(
            f"group {peers} is not chip {chip_index}'s cores {expected}"
        )

    shard = LATENT // group.world_size
    torch.manual_seed(1234)  # same weight on every rank
    weight = torch.randn(LATENT, HIDDEN, dtype=torch.bfloat16)
    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)

    x_tpu = x.to("tpu")
    weight_tpu = weight.to("tpu")
    # The reference is the *replicated device* projection, not a CPU one. A CPU
    # bf16 matmul accumulates differently from the MXU, and over a 7168-long
    # contraction of unit normals the outputs land near 85, where one bf16 ULP
    # is 0.5 -- so comparing against CPU reports half-unit "errors" that say
    # nothing about the sharding. Device-vs-device isolates the one variable.
    reference = torch.nn.functional.linear(x_tpu, weight_tpu).cpu().float()

    local_weight = weight_tpu.narrow(0, group.rank_in_group * shard, shard)

    def sharded(inp):
        partial = torch.nn.functional.linear(inp, local_weight)
        return group.all_gather(partial, dim=-1)

    # (2) Eager.
    eager = sharded(x_tpu).cpu().float()

    # (3) Compiled -- the case that actually matters, since these layers live
    # inside the compiled decode graph.
    from torch_tpu._internal import compile as tpu_compile

    compiled_fn = torch.compile(
        sharded, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )
    compiled = compiled_fn(x_tpu).cpu().float()

    eager_max = float((eager - reference).abs().max().item())
    compiled_max = float((compiled - reference).abs().max().item())
    scale = float(reference.abs().max().item())

    # Compiled must be bit-exact. Column-parallel splits the output, so every
    # element is still accumulated over the whole contraction in one go -- the
    # sharded graph should reduce to the same arithmetic, and it does. This is
    # the path that matters: the projections live inside the compiled decode
    # graph.
    #
    # Eager is allowed about one bf16 ULP. Dispatching op-by-op, the narrow
    # [1792, 7168] GEMM does not always land on the same kernel as the
    # full-width one, and a differently tiled accumulation rounds differently
    # in the last place. That is kernel selection, not a mis-shard: a wrong
    # shard or a swapped gather is off by whole tensor halves, not 1e-3.
    if compiled_max != 0.0 or eager_max > 4e-3 * scale:
        raise AssertionError(
            f"rank={rank} eager_max={eager_max} compiled_max={compiled_max} "
            f"scale={scale}"
        )

    return {
        "rank": rank,
        "chip": chip_index,
        "group_ranks": peers,
        "rank_in_group": int(group.rank_in_group),
        "eager_max": eager_max,
        "compiled_max": compiled_max,
    }


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object]
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        # Match vLLM startup: touch the device before querying JAX topology.
        torch.empty((1,), device="tpu").cpu()

        from vllm.distributed.parallel_state import init_distributed_environment

        # Only the world group is set up, not the TP group: the intra-chip
        # group is derived from the chip topology and the world size, and
        # `initialize_model_parallel` would drag in a full VllmConfig, which
        # the TPU platform hook cannot build without a real model.
        init_distributed_environment(
            world_size=WORLD_SIZE,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result = _measure(rank)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
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


def _run_worker_group_once(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[3],
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
        output = exc.output or ""
        process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _run_worker_group(result_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the workers, retrying through the device-teardown race.

    In the full CI session an in-process engine test runs before this file,
    and its workers release the TPUs asynchronously. Spawned too soon after,
    this test's workers fail init_process_group with "PjRtClient is not
    initialized" (CI build 3112). That is a transient resource race, not a
    code defect, so wait for the devices to drain and try again.
    """
    completed = _run_worker_group_once(result_dir)
    attempts = 1
    while "PjRtClient is not initialized" in (completed.stdout or "") and attempts < 3:
        attempts += 1
        time.sleep(60)
        completed = _run_worker_group_once(result_dir)
    return completed


def test_intra_chip_gather_matches_the_replicated_projection(tmp_path):
    result_dir = tmp_path / "intra_chip"
    result_dir.mkdir()
    completed = _run_worker_group(result_dir)

    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
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

    # The worker already enforces the per-rank bounds; restate the one that
    # carries the claim so a reader of this test sees it.
    assert all(r["compiled_max"] == 0.0 for r in results), (
        "the compiled sharded projection must be bit-exact vs the replicated "
        f"one: {[(r['rank'], r['compiled_max']) for r in results]}"
    )
    # Four chips, two cores each, and the two cores of a chip must have taken
    # different shards -- if both took shard 0 the gather would still return a
    # full-width tensor, holding the same half twice.
    by_chip: dict[int, list[int]] = {}
    for r in results:
        by_chip.setdefault(r["chip"], []).append(r["rank_in_group"])
    assert len(by_chip) == 4
    assert all(sorted(v) == [0, 1] for v in by_chip.values()), by_chip


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
