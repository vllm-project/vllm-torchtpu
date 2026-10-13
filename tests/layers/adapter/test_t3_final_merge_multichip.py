# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Trial-3 final stack, merge half: the DCP merge of real all-off vs all-on
attention partials over real collectives (8 ranks), backbone and drafter.

Reads `T3_FINAL_PARTIALS`, the npz test_t3_final_composition writes
(`T3_FINAL_DUMP`; steps mixed-256, decode-16, decode-32, mixed-2048). With
the s25 merge env (a2a merge <= 64 tokens, chunk8 + collapsed lse gather at
>= 1024), eager and compiled, per step:

* backbone: off = the off partials, PACK / REAL_ROWS / RS_BF16 off; on = the
  on partials with PACK + REAL_ROWS + TPU_DCP_MERGE_RS_BF16=1. Steps that do
  not take the chunked merge (decode a2a, 256 tokens) must be what s25
  allowed (eager bit for bit, compiled within one bf16 ulp). The chunked
  step (2048) is not bitwise by design: held to TPU_DCP_MERGE_RS_BF16's own
  bound, 7 bf16 scale-ulps, over today's error against the f64 merge.
* drafter (`is_backbone=False`): off = today's reduce-scatter merge; on =
  TPU_MTP_DCP_MERGE_SIZED (the a2a branch at <= 64 tokens, reduce-scatter op
  for op above): bit for bit above 64 tokens, within one bf16 ulp of off at
  or below.

Errors vs the f64 merge are in bf16 ulps of max(sum_r w_r |o_r|, 2^-10)
(real partials cancel to near zero, where an unfloored relative measure
fails today's merge too); each on arm may add at most its bound (chunked
backbone: RS_BF16's 7; everything else: 1) to the off arm's error.

Every arm reports its differing-element count vs off, its max direct bf16
ulp vs off and its scale-ulp error vs f64. Positive control: the on partials
with every rank's lse rolled by one token must break both bounds.
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
import test_t4_combined_merge_multichip as t4m  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
S25_MERGE_ENV = dict(t4m.S23_MERGE_ENV)
RS_BF16_MODE = 1
RS_BF16_MAX_SCALE_ULP = mm.RS_BF16_MAX_ULP[RS_BF16_MODE]
# Real attention partials have outputs near zero (cancellation), where any
# relative measure explodes for today's merge too: the scale sum_r w_r |o_r|
# is floored here before taking its bf16 ulp.
SCALE_FLOOR = 2.0**-10


def _scale_err(got, outputs, lses, rank, world):
    """max |got - f64| in bf16 ulps of max(sum_r w_r |o_r|, SCALE_FLOOR) over
    this rank's heads, and the max absolute error."""
    import numpy as np

    local = outputs.shape[2] // world
    sl = slice(rank * local, (rank + 1) * local)
    out = outputs[:, :, sl].float().numpy().astype(np.float64)
    lse = lses[:, :, sl].numpy().astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0.0, den, 1.0)
    ref = (out * w[..., None]).sum(axis=0)
    scale = np.maximum((np.abs(out) * w[..., None]).sum(axis=0), SCALE_FLOOR)
    err = np.abs(got.float().numpy().astype(np.float64) - ref)
    e = np.nan_to_num(err / np.exp2(np.floor(np.log2(scale)) - 7), nan=np.inf)
    return {
        "max_ulp": float(e.max()),
        "max_abs": float(np.nan_to_num(err, nan=np.inf).max()),
    }


def _all_partials(z, step, tag, dcp, real):
    """Every rank's [real] rows: outputs [world, real, H, V] bf16, lse f32."""
    import numpy as np
    import torch

    out = torch.from_numpy(z[f"s{step}_{tag}_out"].view(np.int16).copy())
    lse = torch.from_numpy(z[f"s{step}_{tag}_lse"].view(np.float32).copy())
    n = out.shape[0] // dcp
    o = out.view(torch.bfloat16).reshape(dcp, n, *out.shape[1:])[:, :real]
    lse = lse.reshape(dcp, n, -1)[:, :real]
    return o.contiguous(), lse.contiguous()


def _set_backbone(cp_mla, on):
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_PACK = on
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS = on
    cp_mla.envs.TPU_DCP_MERGE_RS_BF16 = RS_BF16_MODE if on else 0
    cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = False


def _set_drafter(cp_mla, on):
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_PACK = False
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS = False
    cp_mla.envs.TPU_DCP_MERGE_RS_BF16 = 0
    cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = on


def _check(rank, cp_mla, path):
    import numpy as np
    import torch

    z = np.load(path)
    dcp = int(z["dcp"])
    assert dcp == WORLD_SIZE, dcp
    for k, v in S25_MERGE_ENV.items():
        setattr(cp_mla.envs, k, v)
    merge = cp_mla.merge_lse_partials_scatter_heads
    local = None
    report = {}
    for step in range(int(z["steps"])):
        real = int(z[f"s{step}_real"])
        for model in ("backbone", "drafter"):
            for how in ("eager", "compiled"):
                key = f"s{step}/{model}/{how}"
                try:
                    qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
                    res = {}
                    for tag, on, roll in (
                        ("off", False, False),
                        ("on", True, False),
                        ("on", True, True),
                    ):
                        if model == "backbone":
                            _set_backbone(cp_mla, on)

                            def fn0(o, lse, qsl=qsl):
                                return merge(o, lse, query_start_loc=qsl)

                        else:
                            _set_drafter(cp_mla, on)

                            def fn0(o, lse):
                                return merge(o, lse, is_backbone=False)

                        fn = fn0 if how == "eager" else mm._compile(fn0)
                        o, lse = t4m._partials(z, step, tag, rank, dcp, roll)
                        res[(tag, roll)] = fn(o.to("tpu"), lse.to("tpu")).cpu()
                    want, got = res[("off", False)], res[("on", False)]
                    ctl = res[("on", True)]
                    tokens = int(want.shape[0])
                    outs, lses = _all_partials(z, step, "off", dcp, real)
                    local = outs.shape[2] // dcp
                    ref = mm._reference(outs, lses)[
                        :, rank * local : (rank + 1) * local
                    ]
                    u = t4m._ulp(got[:real], want[:real])
                    report[key] = {
                        "tokens": tokens,
                        "real": real,
                        "elements": int(u.numel()),
                        "n_diff": int((u > 0).sum()),
                        "max_ulp": int(u.max()) if u.numel() else 0,
                        "chunked": bool(
                            model == "backbone" and cp_mla._merge_chunks(tokens) > 1
                        ),
                        "a2a": bool(
                            tokens <= S25_MERGE_ENV["TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS"]
                        ),
                        "off_scale": _scale_err(want[:real], outs, lses, rank, dcp),
                        "on_scale": _scale_err(got[:real], outs, lses, rank, dcp),
                        "off_scale_raw": mm._scale_ulp_stats(
                            want[:real], outs, lses, rank, dcp
                        )["max_ulp"],
                        "on_f64": mm._error_stats(got[:real], ref),
                        "off_f64": mm._error_stats(want[:real], ref),
                        "ctl_scale": _scale_err(ctl[:real], outs, lses, rank, dcp),
                        "ctl_f64": mm._error_stats(ctl[:real], ref),
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
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="t3f_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
        cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS = S25_MERGE_ENV[
            "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS"
        ]
        cp_mla.envs.TPU_MTP_DCP_MERGE_SIZED = True
        cp_mla.prebuild_sized_merge(world)
        result["check"] = _check(rank, cp_mla, path)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_t3_merge_of_composed_partials_on_real_collectives(tmp_path):
    path = os.environ.get("T3_FINAL_PARTIALS")
    if not path or not os.path.exists(path):
        pytest.skip("T3_FINAL_PARTIALS (npz from test_t3_final_composition) unset")
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
                key,
                dict(
                    tokens=c["tokens"],
                    chunked=c["chunked"],
                    elements=0,
                    n_diff=0,
                    max_ulp=0,
                    on_scale_max=0.0,
                    off_scale_max=0.0,
                    on_abs_max=0.0,
                    off_abs_max=0.0,
                    on_within_1ulp_f64=True,
                    off_within_1ulp_f64=True,
                    ctl_scale_max=0.0,
                    ctl_within_1ulp_f64=True,
                ),
            )
            t["elements"] += c["elements"]
            t["n_diff"] += c["n_diff"]
            t["max_ulp"] = max(t["max_ulp"], c["max_ulp"])
            t["on_scale_max"] = max(t["on_scale_max"], c["on_scale"]["max_ulp"])
            t["off_scale_max"] = max(t["off_scale_max"], c["off_scale"]["max_ulp"])
            t["ctl_scale_max"] = max(t["ctl_scale_max"], c["ctl_scale"]["max_ulp"])
            t["on_abs_max"] = max(t["on_abs_max"], c["on_scale"]["max_abs"])
            t["off_abs_max"] = max(t["off_abs_max"], c["off_scale"]["max_abs"])
            t["on_within_1ulp_f64"] &= c["on_f64"]["within_1ulp"]
            t["off_within_1ulp_f64"] &= c["off_f64"]["within_1ulp"]
            t["ctl_within_1ulp_f64"] &= c["ctl_f64"]["within_1ulp"]
    failures = []
    for key, t in sorted(total.items()):
        print(f"T3MERGE {key}: {json.dumps(t)}")
        if "/backbone/" in key:
            if t["chunked"]:
                # bf16 wire: at most RS_BF16's bound over today's error.
                bound = t["off_scale_max"] + RS_BF16_MAX_SCALE_ULP
            else:
                bound = t["off_scale_max"] + 1
                ok0 = t["max_ulp"] <= 1 and (
                    not key.endswith("eager") or t["n_diff"] == 0
                )
        else:
            # Drafter: op for op above the a2a cap; rank-order sum at or below.
            bound = t["off_scale_max"] + 1
            ok0 = t["n_diff"] == 0 if t["tokens"] > 64 else t["max_ulp"] <= 1
        if "/backbone/" in key and t["chunked"]:
            ok0 = True
        ok = ok0 and t["on_scale_max"] <= bound and t["ctl_scale_max"] > bound
        if not ok:
            failures.append((key, t))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--partials", required=True)
    a = p.parse_args()
    _run_worker(a.result_dir, a.partials)
