# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The DCP merge of real prologue / kv-in-gather / rank-operand partials over
real collectives, pack on vs off (composition check, compiled half).

Reads `T4_COMB_PARTIALS`, the npz `tests/kernels/test_t4_combined_composition.py`
writes (`T4_COMB_DUMP`): per step, every rank's partial output and lse from the
all-flags-off and the all-flags-on attention runs. On 8 ranks, with the s23
merge env (a2a merge for <= 64 tokens, chunk8 + collapsed lse gather above),
each rank merges the off partials with TPU_DCP_LSE_MERGE_A2A_PACK off and the
on partials with PACK + REAL_ROWS on (query_start_loc [0, real]), eager and
compiled. On the real rows eager must be bit for bit and compiled within one
bf16 ulp (the pack's known compiled tolerance); the count is reported. With
real rows the padding rows of the on merge are zero. Positive control: the on
partials with every rank's lse
rolled by one token must fail the same check.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_cp_mla_merge_multichip as mm  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
S23_MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": True,
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": 64,
    "TPU_DCP_MERGE_CHUNKED": 8,
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 1024,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 2048,
}


def _ulp(a, b):
    import torch

    def line(x):
        x = x.view(torch.int16).to(torch.int32)
        return torch.where(x < 0, -(x & 0x7FFF), x)

    return (line(a) - line(b)).abs()


def _partials(z, step, tag, rank, dcp, roll=False):
    import numpy as np
    import torch

    out = torch.from_numpy(z[f"s{step}_{tag}_out"].view(np.int16).copy())
    lse = torch.from_numpy(z[f"s{step}_{tag}_lse"].view(np.float32).copy())
    n = out.shape[0] // dcp
    o = out.view(torch.bfloat16).reshape(dcp, n, *out.shape[1:])[rank]
    lse = lse.reshape(dcp, n, -1)[rank]
    if roll:
        lse = torch.roll(lse, 1, 0)
    return o.contiguous(), lse.contiguous()


def _check(rank, cp_mla, path):
    import numpy as np
    import torch

    z = np.load(path)
    dcp = int(z["dcp"])
    assert dcp == WORLD_SIZE, dcp
    for k, v in S23_MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    merge = cp_mla.merge_lse_partials_scatter_heads
    report = {}
    for step in range(int(z["steps"])):
        for how in ("eager", "compiled"):
            key = f"s{step}/{how}"
            try:
                res = {}
                real = int(z[f"s{step}_real"])
                qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
                for tag, on, roll in (
                    ("off", False, False),
                    ("on", True, False),
                    ("on", True, True),
                ):
                    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_PACK = on
                    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS = on

                    def fn0(o, lse, qsl=qsl):
                        return merge(o, lse, query_start_loc=qsl)

                    fn = fn0 if how == "eager" else mm._compile(fn0)
                    o, lse = _partials(z, step, tag, rank, dcp, roll)
                    res[(tag, roll)] = fn(o.to("tpu"), lse.to("tpu")).cpu()
                want, got = res[("off", False)], res[("on", False)]
                u = _ulp(got[:real], want[:real])
                c = _ulp(res[("on", True)][:real], want[:real])
                a2a = want.shape[0] <= S23_MERGE_ENV["TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS"]
                report[key] = {
                    "tokens": int(want.shape[0]),
                    "real": real,
                    "elements": int(u.numel()),
                    "n_diff": int((u > 0).sum()),
                    "max_ulp": int(u.max()) if u.numel() else 0,
                    # Real rows only apply on the a2a (decode) merge.
                    "pad_zero": bool((got[real:] == 0).all()) if a2a else None,
                    "control_n_diff": int((c > 0).sum()),
                    "control_max_ulp": int(c.max()) if c.numel() else 0,
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:400]}
    return report


def _run_worker(result_dir: Path, path: str) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank}
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
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="t4c_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        # The real-rows exchange is a JAX op on the DCP mesh, which reads the
        # group from `distributed.dcp`; register it before any trace.
        from vllm_torchtpu.distributed import dcp as dcp_mod

        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        result["check"] = _check(rank, cp_mla, path)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_merge_of_composed_partials_on_real_collectives(tmp_path):
    path = os.environ.get("T4_COMB_PARTIALS")
    if not path or not os.path.exists(path):
        pytest.skip("T4_COMB_PARTIALS (npz from test_t4_combined_composition) unset")
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--result-dir",
        str(tmp_path),
        "--partials",
        path,
    ]
    done = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=mm._prepare_worker_env(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=mm.LAUNCH_TIMEOUT_SECONDS,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-8000:],
    )
    total = {}
    for r in results:
        for key, c in r["check"].items():
            assert "error" not in c, (r["rank"], key, c)
            t = total.setdefault(
                key, dict(elements=0, n_diff=0, max_ulp=0, ctl=0, pad_zero=True)
            )
            if c["pad_zero"] is not None:
                t["pad_zero"] = t["pad_zero"] and c["pad_zero"]
            t["elements"] += c["elements"]
            t["n_diff"] += c["n_diff"]
            t["max_ulp"] = max(t["max_ulp"], c["max_ulp"])
            t["ctl"] += c["control_n_diff"]
    for key, t in sorted(total.items()):
        print(f"MERGE {key}: {t}")
        if key.endswith("eager"):
            assert t["n_diff"] == 0, (key, t)
        assert t["max_ulp"] <= 1, (key, t)
        assert t["pad_zero"], (key, t)
        assert t["ctl"] > 0, ("rolled lse not caught", key, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--partials", required=True)
    a = p.parse_args()
    _run_worker(a.result_dir, a.partials)
