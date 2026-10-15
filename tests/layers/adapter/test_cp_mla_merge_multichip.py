# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The DCP MLA lse merge over real collectives (`TPU_DCP_LSE_MERGE_A2A`).

The unit tests in `test_cp_mla_attention.py` pin the arithmetic against a
threaded stand-in for the exchange; this pins what only hardware can answer:

  1. `all_to_all_single` over a vLLM group lowers and runs on TorchTPU, eager
     and under `torch.compile`, and hands block `d` to rank `d`.
  2. Every rank's merged head block matches an f64 reference of the exact
     merge to within one bf16 ulp, including `-inf` rows and a rank that owns
     nothing -- and the same check fails when the send blocks are rotated by
     one rank (the positive control).

Run on one host (4 chips x 2 cores = 8 ranks). `--bench` adds a per-merge
timing of every path at decode and mixed-step token counts.
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
HEADS = 64
VDIM = 512
DV = 256  # GLM-5.2 v_head_dim
CHECK_TOKENS = 64
LAUNCH_TIMEOUT_SECONDS = 900

# name -> TPU_DCP_LSE_MERGE_A2A
ARMS = {
    "today": False,
    "a2a": True,
}


def _worker_command(result_dir: Path, extra: list[str]) -> list[str]:
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
        *extra,
    ]


def _prepare_worker_env() -> dict[str, str]:
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


# --------------------------------------------------------------------- worker
def _global_partials(world, tokens, seed=20261002):
    """Every rank's (bf16 partial, f32 lse), identical on all ranks."""
    import torch

    g = torch.Generator().manual_seed(seed)
    outputs = torch.randn(world, tokens, HEADS, VDIM, generator=g).to(torch.bfloat16)
    lses = torch.randn(world, tokens, HEADS, generator=g) * 4 + 3
    lses[:, 0] += 200.0  # exp overflows without the max shift
    lses[:, 1] -= 200.0  # exp underflows without the max shift
    lses[1:, 2] = float("-inf")  # rank 0 owns token 2 alone
    lses[:, 3] = float("-inf")  # nobody owns token 3
    lses[world - 1] = float("-inf")  # the last rank owns nothing
    lses[0, 4, : HEADS // world] = float("-inf")
    return outputs, lses


def _global_w_uv(seed=20261009):
    """A bf16 W_UV `[HEADS, VDIM, DV]` for the whole group, identical on all ranks."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return (torch.randn(HEADS, VDIM, DV, generator=g) / VDIM**0.5).to(torch.bfloat16)


def _uv_paths(cp_mla, merge, w_dev, rank, world):
    """W_UV after the merge (main: bf16 latent, bf16 bmm) and before it."""
    import torch

    local = HEADS // world
    w_local = w_dev[rank * local : (rank + 1) * local].contiguous().clone()

    def after(o, lse):
        merged = merge(o, lse).transpose(0, 1)
        return torch.bmm(merged, w_local).transpose(0, 1)

    def first(o, lse):
        return merge(cp_mla.project_uv_heads(o, w_dev, None), lse)

    # The path under test first, so a later device fault cannot hide it.
    return {"uv_first": first, "uv_after": after}


def _reference(outputs, lses):
    import numpy as np

    out = outputs.float().numpy().astype(np.float64)
    lse = lses.numpy().astype(np.float64)
    fin = np.isfinite(lse)
    shift = np.where(fin, lse, -np.inf).max(axis=0)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    w = np.where(fin, np.exp(lse - shift[None]), 0.0)
    den = w.sum(axis=0)
    w = w / np.where(den > 0.0, den, 1.0)
    return (out * w[..., None]).sum(axis=0)


def _error_stats(got, ref):
    import numpy as np

    got = got.float().numpy().astype(np.float64)
    err = np.abs(got - ref)
    tol = 2.0**-8 * np.abs(ref) + 1e-6
    return {
        "finite": bool(np.isfinite(got).all()),
        "within_1ulp": bool(np.isfinite(got).all() and (err <= tol).all()),
        "frac_over_1ulp": float((err > tol).mean()),
        "max_abs_err": float(np.nan_to_num(err, nan=np.inf).max()),
    }


def _weights(l_recv):
    import torch

    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    w = torch.where(finite, torch.exp(l_recv - shift), torch.zeros_like(l_recv))
    den = w.sum(dim=0)
    return w / torch.where(den > 0, den, torch.ones_like(den))


def _merge_reduce_sum(o_recv, l_recv):
    """The plain multiply-then-sum form (compiles to a default-precision dot)."""
    import torch

    w = _weights(l_recv)
    return (o_recv.to(torch.float32) * w.unsqueeze(-1)).sum(dim=0)


def _merge_unrolled(o_recv, l_recv):
    """An explicit elementwise f32 accumulation over the rank axis."""
    import torch

    w = _weights(l_recv)
    o_f32 = o_recv.to(torch.float32)
    acc = o_f32[0] * w[0].unsqueeze(-1)
    for r in range(1, o_recv.shape[0]):
        acc = acc + o_f32[r] * w[r].unsqueeze(-1)
    return acc


# Local-merge variants measured and checked beside the production one.
MERGE_VARIANTS = {"a2a_reduce_sum": _merge_reduce_sum, "a2a_unrolled": _merge_unrolled}


def _set_arm(cp_mla, arm):
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A = ARMS[arm]
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS = 1 << 30
    cp_mla.envs.TPU_DCP_RS_OUT_PRECISION = False  # "today" stays the f32 reference


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _check(rank, world, group, cp_mla):
    import numpy as np
    import torch

    outputs, lses = _global_partials(world, CHECK_TOKENS)
    local = HEADS // world
    ref = _reference(outputs, lses)[:, rank * local : (rank + 1) * local]
    o_dev = outputs[rank].to("tpu")
    l_dev = lses[rank].to("tpu")
    merge = cp_mla.merge_lse_partials_scatter_heads
    report = {}
    for arm in ARMS:
        _set_arm(cp_mla, arm)
        for how in ("eager", "compiled"):
            try:
                fn = merge if how == "eager" else _compile(merge)
                got = fn(o_dev, l_dev).cpu()
                report[f"{arm}/{how}"] = {
                    "shape": list(got.shape),
                    "dtype": str(got.dtype),
                    **_error_stats(got, ref),
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[f"{arm}/{how}"] = {"error": repr(exc)[:400]}

    # W_UV before vs after the merge, against the f64 reference projected.
    w_uv = _global_w_uv()
    ref_uv = np.einsum(
        "thv,hvd->thd",
        _reference(outputs, lses),
        w_uv.float().numpy().astype(np.float64),
    )[:, rank * local : (rank + 1) * local]
    w_dev = w_uv.to("tpu")
    for arm in ARMS:
        _set_arm(cp_mla, arm)
        for path, fn in _uv_paths(cp_mla, merge, w_dev, rank, world).items():
            for how in ("eager", "compiled"):
                try:
                    run = fn if how == "eager" else _compile(fn)
                    got = run(o_dev, l_dev).cpu()
                    report[f"{path}/{arm}/{how}"] = {
                        "shape": list(got.shape),
                        "dtype": str(got.dtype),
                        **_error_stats(got, ref_uv),
                    }
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[f"{path}/{arm}/{how}"] = {"error": repr(exc)[:400]}

    # Diagnostics, reported but not asserted: the compiled local merge with an
    # explicit per-rank f32 accumulation, and under HIGHEST op precision. They
    # tell a reduce-of-multiply rewritten into a default-precision (bf16) dot
    # apart from any other source of a compiled-only error.
    real_merge = cp_mla._merge_received

    _set_arm(cp_mla, "a2a")
    for name, variant in MERGE_VARIANTS.items():
        try:
            cp_mla._merge_received = variant
            got = _compile(merge)(o_dev, l_dev).cpu()
            report[f"diag/{name}/compiled"] = _error_stats(got, ref)
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            report[f"diag/{name}/compiled"] = {"error": repr(exc)[:400]}
        finally:
            cp_mla._merge_received = real_merge
    try:
        from torch_tpu._internal.precision import Precision, precision

        with precision(Precision.HIGHEST):
            got = _compile(merge)(o_dev, l_dev).cpu()
        report["diag/a2a_highest/compiled"] = _error_stats(got, ref)
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        report["diag/a2a_highest/compiled"] = {"error": repr(exc)[:400]}

    # Positive control: rotate the send blocks by one rank, so rank d merges
    # the heads meant for rank d-1. The same check must now fail.
    real = cp_mla._all_to_all
    cp_mla._all_to_all = lambda x, g: real(torch.roll(x, 1, dims=0), g)
    try:
        for arm in ("a2a",):
            _set_arm(cp_mla, arm)
            for how in ("eager", "compiled"):
                try:
                    fn = merge if how == "eager" else _compile(merge)
                    got = fn(o_dev, l_dev).cpu()
                    report[f"control_rotated/{arm}/{how}"] = _error_stats(got, ref)
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[f"control_rotated/{arm}/{how}"] = {"error": repr(exc)[:400]}
    finally:
        cp_mla._all_to_all = real
    return report


def _bench(rank, cp_mla, tokens_list, reps=10):
    """Per-merge device time: (wall of 1+K merges - wall of 1 merge) / K, each
    chained through the lse so none can be dropped or overlapped."""
    import torch

    def chain(k):
        def run(o, lse):
            for _ in range(k):
                y = cp_mla.merge_lse_partials_scatter_heads(o, lse)
                # A one-element update of both inputs: nothing is loop
                # invariant, and the update itself costs nothing.
                s = y.reshape(-1)[:1].float() * 0.0
                o = o.slice_scatter(o[:1] + s.to(o.dtype), dim=0, start=0, end=1)
                lse = lse.slice_scatter(lse[:1] + s, dim=0, start=0, end=1)
            return lse[:1, :1]

        return run

    out = {}
    world = int(os.environ["WORLD_SIZE"])
    w_dev = _global_w_uv().to("tpu")
    for tokens in tokens_list:
        torch.manual_seed(rank)
        o = torch.randn(tokens, HEADS, VDIM).to(torch.bfloat16).to("tpu")
        lse = (torch.randn(tokens, HEADS) * 3).to("tpu")
        real_merge = cp_mla._merge_received
        for arm in ("today", "a2a", *MERGE_VARIANTS):
            _set_arm(cp_mla, "a2a" if arm in MERGE_VARIANTS else arm)
            cp_mla._merge_received = MERGE_VARIANTS.get(arm, real_merge)
            try:
                times = {}
                for k in (1, 1 + reps):
                    fn = _compile(chain(k))
                    fn(o, lse).cpu()
                    fn(o, lse).cpu()
                    samples = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        fn(o, lse).cpu()
                        samples.append(time.perf_counter() - t0)
                    samples.sort()
                    times[k] = samples[len(samples) // 2]
                per = (times[1 + reps] - times[1]) / reps
                out[f"{arm}@{tokens}"] = {
                    "per_merge_us": per * 1e6,
                    "single_call_us": times[1] * 1e6,
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                out[f"{arm}@{tokens}"] = {"error": repr(exc)[:400]}
            finally:
                cp_mla._merge_received = real_merge
            if rank == 0:
                print(f"BENCH T={tokens} {arm}: {out[f'{arm}@{tokens}']}", flush=True)
        for arm in ARMS:
            _set_arm(cp_mla, arm)
            paths = _uv_paths(
                cp_mla, cp_mla.merge_lse_partials_scatter_heads, w_dev, rank, world
            )
            for path, fn in paths.items():
                key = f"{path}/{arm}@{tokens}"
                try:
                    out[key] = _time_chain(fn, o, lse, reps)
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    out[key] = {"error": repr(exc)[:400]}
                if rank == 0:
                    print(f"BENCH T={tokens} {path}/{arm}: {out[key]}", flush=True)
    return out


def _time_chain(fn, o, lse, reps):
    """Per-call device time of `fn(o, lse)`, chained like `_bench`'s merges."""

    def chain(k):
        def run(o, lse):
            for _ in range(k):
                y = fn(o, lse)
                s = y.reshape(-1)[:1].float() * 0.0
                o = o.slice_scatter(o[:1] + s.to(o.dtype), dim=0, start=0, end=1)
                lse = lse.slice_scatter(lse[:1] + s, dim=0, start=0, end=1)
            return lse[:1, :1]

        return run

    times = {}
    for k in (1, 1 + reps):
        run = _compile(chain(k))
        run(o, lse).cpu()
        run(o, lse).cpu()
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            run(o, lse).cpu()
            samples.append(time.perf_counter() - t0)
        samples.sort()
        times[k] = samples[len(samples) // 2]
    return {
        "per_merge_us": (times[1 + reps] - times[1]) / reps * 1e6,
        "single_call_us": times[1] * 1e6,
    }


def _run_worker(result_dir: Path, bench_tokens: list[int]) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        from vllm.distributed.parallel_state import init_model_parallel_group

        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        # The world group has no device communicator (vLLM builds it without
        # one), so `all_gather`/`reduce_scatter` would raise; a model-parallel
        # group over the same ranks is what the DCP group really is.
        world_group = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="dcp_merge_test",
        )
        cp_mla.get_dcp_group = lambda: group
        result["check"] = _check(rank, world, group, cp_mla)
        if bench_tokens:
            result["bench"] = _bench(rank, cp_mla, bench_tokens)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


# --------------------------------------------------------------------- pytest
def _launch(result_dir: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir, extra),
        cwd=Path(__file__).resolve().parents[3],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # libtpu logs can carry non-UTF-8 bytes; they must not fail the test.
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


def _results(result_dir: Path, world: int) -> list[dict]:
    out = []
    for rank in range(world):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_merge_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    for r in results:
        for key, c in sorted(r["check"].items()):
            if key.startswith("uv_") and key.endswith("/compiled"):
                print(f"UV rank={r['rank']} {key}: {c}", flush=True)
    for r in results:
        check = r["check"]
        for arm in ("today", "a2a"):
            for how in ("eager", "compiled"):
                c = check[f"{arm}/{how}"]
                assert "error" not in c, (r["rank"], arm, how, c)
                assert c["dtype"] == "torch.bfloat16", (r["rank"], arm, how, c)
                assert c["finite"], (r["rank"], arm, how, c)
        for how in ("eager", "compiled"):
            today, a2a = check[f"today/{how}"], check[f"a2a/{how}"]
            # The all-to-all merge must be exact to one bf16 ulp, or at worst
            # no less exact than today's path compiled the same way.
            assert a2a["within_1ulp"] or (
                a2a["frac_over_1ulp"] <= today["frac_over_1ulp"]
                and a2a["max_abs_err"] <= today["max_abs_err"]
            ), (r["rank"], how, today, a2a)
        for arm in ARMS:
            for how in ("eager", "compiled"):
                first = check[f"uv_first/{arm}/{how}"]
                after = check[f"uv_after/{arm}/{how}"]
                assert "error" not in first, (r["rank"], arm, how, first)
                assert "error" not in after, (r["rank"], arm, how, after)
                assert first["shape"] == after["shape"], (r["rank"], arm, how)
                # One rounding at the end: exact to one bf16 ulp, and never
                # less exact than projecting the rounded merge.
                assert first["within_1ulp"], (r["rank"], arm, how, first)
                assert first["max_abs_err"] <= after["max_abs_err"], (
                    r["rank"],
                    arm,
                    how,
                    first,
                    after,
                )
        for arm in ("a2a",):
            for how in ("eager", "compiled"):
                c = check[f"control_rotated/{arm}/{how}"]
                assert "error" not in c, (r["rank"], arm, how, c)
                assert not c["within_1ulp"], (
                    "positive control: rotated send blocks must fail the check",
                    r["rank"],
                    arm,
                    how,
                    c,
                )


def test_all_to_all_merge_on_real_collectives(tmp_path):
    result_dir = tmp_path / "dcp_merge"
    result_dir.mkdir()
    extra = []
    if os.environ.get("DCP_MERGE_BENCH_TOKENS"):
        extra = ["--bench-tokens", os.environ["DCP_MERGE_BENCH_TOKENS"]]
    completed = _launch(result_dir, extra)
    results = _results(result_dir, WORLD_SIZE)
    for line in (completed.stdout or "").splitlines():
        if line.startswith("BENCH "):
            print(line)
    print(json.dumps(results[0], indent=1))
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    assert_merge_results(results)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--bench-tokens", default="")
    parser.add_argument(
        "--assert-results",
        type=int,
        default=0,
        help="parent mode: assert the per-rank results of WORLD ranks",
    )
    args = parser.parse_args()
    if args.assert_results:
        results = _results(args.result_dir, args.assert_results)
        print(json.dumps(results[0], indent=1))
        assert_merge_results(results)
        print(f"ASSERT OK over {args.assert_results} ranks")
        return
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if not args.worker:
        raise ValueError("expected --worker")
    tokens = [int(t) for t in args.bench_tokens.split(",") if t]
    _run_worker(args.result_dir, tokens)


if __name__ == "__main__":
    _main()
