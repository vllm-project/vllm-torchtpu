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
"""The MTP drafter's sized DCP merge (`TPU_MTP_DCP_MERGE_SIZED`) over real
collectives, beside the reduce-scatter merge the drafter runs today.

On one host (8 ranks), eager and compiled (the test backend compiles static
shapes only; the drafter's dynamic-bucket trace is exercised by the twin/mirror runs):

  1. The op lowers and runs; both branches (packed all-to-all at <= the cap,
     lse all-gather + reduce-scatter above) are within one bf16 ulp of the f64
     merge, including -inf rows, a rank owning nothing and NaN partials of
     weight 0.
  2. Reported: the share of bf16 outputs that differ bit for bit from today's
     reduce-scatter merge (the all-to-all sums the ranks in another order).
  3. Positive control: the packed lse rows rotated by one rank fail check 1.
  4. `--bench-tokens`: per-merge device time, today vs the sized op.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cp_mla_merge_multichip as base  # noqa: E402

WORLD_SIZE = base.WORLD_SIZE
HEADS = base.HEADS
VDIM = base.VDIM
CAP = 64


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _check(rank, world, group, cp_mla):
    import torch

    report = {}
    local = HEADS // world

    def today(o, lse):
        return cp_mla.merge_lse_partials_scatter_heads(o, lse, is_backbone=False)

    for tokens in (16, 8, 128):
        outputs, lses = base._global_partials(world, tokens)
        poisoned = outputs.clone()
        poisoned[~torch.isfinite(lses)] = float("nan")
        ref = base._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
        o_dev, p_dev = outputs[rank].to("tpu"), poisoned[rank].to("tpu")
        l_dev = lses[rank].to("tpu")
        got = {}
        for arm, flag in (("today", False), ("sized", True)):
            cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = flag
            for how in ("eager", "compiled"):
                key = f"{arm}/{how}@{tokens}"
                try:
                    fn = today if how == "eager" else _compile(today)
                    y = fn(o_dev, l_dev).cpu()
                    yp = fn(p_dev, l_dev).cpu()
                    got[(arm, how)] = y
                    c = {"shape": list(y.shape), **base._error_stats(y, ref)}
                    c["nan_input_finite"] = bool(torch.isfinite(yp.float()).all())
                    c["nan_input_bitwise"] = bool(
                        torch.equal(yp.view(torch.int16), y.view(torch.int16))
                    )
                    if ("today", "compiled") in got and arm == "sized":
                        same = y.view(torch.int16) == got[("today", "compiled")].view(
                            torch.int16
                        )
                        c["frac_bitwise_vs_today_compiled"] = float(same.float().mean())
                    report[key] = c
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:600]}
    # Positive control (new shape, so the op re-traces with the fault).
    cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = True
    real = cp_mla._unpack_words

    def rotated(recv, tokens, local_, dtype):
        import jax.numpy as jnp

        o_recv, l_recv = real(recv, tokens, local_, dtype)
        return o_recv, jnp.roll(l_recv, 1, axis=0)

    cp_mla._unpack_words = rotated
    try:
        outputs, lses = base._global_partials(world, 24)
        ref = base._reference(outputs, lses)[:, rank * local : (rank + 1) * local]
        y = _compile(today)(outputs[rank].to("tpu"), lses[rank].to("tpu")).cpu()
        report["control_rotated_lse/compiled@24"] = base._error_stats(y, ref)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report["control_rotated_lse/compiled@24"] = {"error": repr(exc)[:600]}
    finally:
        cp_mla._unpack_words = real
    return report


def _bench(rank, cp_mla, tokens_list, reps=10):
    import torch

    def chain(k):
        def run(o, lse):
            for _ in range(k):
                y = cp_mla.merge_lse_partials_scatter_heads(o, lse, is_backbone=False)
                s = y.reshape(-1)[:1].float() * 0.0
                o = o.slice_scatter(o[:1] + s.to(o.dtype), dim=0, start=0, end=1)
                lse = lse.slice_scatter(lse[:1] + s, dim=0, start=0, end=1)
            return lse[:1, :1]

        return run

    out = {}
    for tokens in tokens_list:
        torch.manual_seed(rank)
        o = torch.randn(tokens, HEADS, VDIM).to(torch.bfloat16).to("tpu")
        lse = (torch.randn(tokens, HEADS) * 3).to("tpu")
        for arm, flag in (("today", False), ("sized", True)):
            cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = flag
            times = {}
            for k in (1, 1 + reps):
                fn = _compile(chain(k))
                fn(o, lse).cpu()
                fn(o, lse).cpu()
                samples = []
                for _ in range(7):
                    t0 = time.perf_counter()
                    fn(o, lse).cpu()
                    samples.append(time.perf_counter() - t0)
                samples.sort()
                times[k] = samples[len(samples) // 2]
            out[f"{arm}@{tokens}"] = {
                "per_merge_us": (times[1 + reps] - times[1]) / reps * 1e6
            }
            if rank == 0:
                print(f"BENCH T={tokens} {arm}: {out[f'{arm}@{tokens}']}", flush=True)
    return out


def _run_worker(result_dir: Path, bench_tokens: list[int]) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "world": world}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        from vllm_torchtpu.distributed import dcp
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="mtp_dcp_merge_sized_test",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp.get_dcp_group = lambda: group
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A = False
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS = CAP
        cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = True
        cp_mla.prebuild_sized_merge(world)
        result["check"] = _check(rank, world, group, cp_mla)
        if bench_tokens:
            result["bench"] = _bench(rank, cp_mla, bench_tokens)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2))
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
    if "error" in result:
        raise SystemExit(1)


def assert_results(results):
    errors = [r for r in results if "error" in r]
    assert not errors, errors
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            if key.startswith("control"):
                assert not c["within_1ulp"], (
                    "positive control not caught",
                    r["rank"],
                    key,
                    c,
                )
                continue
            assert c["finite"] and c["nan_input_finite"], (r["rank"], key, c)
            if key.startswith("sized"):
                assert c["within_1ulp"], (r["rank"], key, c)
                assert c["nan_input_bitwise"], (r["rank"], key, c)


def test_sized_merge_on_real_collectives(tmp_path):
    result_dir = tmp_path / "sized"
    result_dir.mkdir()
    extra = []
    if os.environ.get("DCP_MERGE_BENCH_TOKENS"):
        extra = ["--bench-tokens", os.environ["DCP_MERGE_BENCH_TOKENS"]]
    import subprocess

    cmd = base._worker_command(result_dir, extra)
    cmd[cmd.index(str(Path(base.__file__).resolve()))] = str(Path(__file__).resolve())
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=base._prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=base.LAUNCH_TIMEOUT_SECONDS,
    )
    results = base._results(result_dir, WORLD_SIZE)
    for line in (completed.stdout or "").splitlines():
        if line.startswith("BENCH "):
            print(line)
    print(json.dumps(results[0], indent=1))
    if any("error" in r for r in results):
        print(completed.stdout[-12000:])
    assert_results(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--bench-tokens", default="")
    a = p.parse_args()
    a.result_dir.mkdir(parents=True, exist_ok=True)
    _run_worker(a.result_dir, [int(t) for t in a.bench_tokens.split(",") if t])
