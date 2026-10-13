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
  3. A NaN partial of weight 0 (a rank that owns no position of a row ran on a
     dummy key) leaves the a2a merge finite and bit-identical to the merge of
     the same partials with finite values there; the guard changes no bit of an
     owned row on finite partials; and without the NaN zeroing (the
     positive control) the merge comes back NaN.
  4. The token-chunked reduce-scatter merge (`TPU_DCP_MERGE_CHUNKED`) changes
     no per-row arithmetic: it is reported bit for bit against the unchunked
     merge compiled the same way, must be no less exact, and the same check
     fails when every chunk is weighted with its neighbour row's lse.
  5. Collapsing the chunked merge's per-chunk lse gathers into one
     (`TPU_DCP_MERGE_CHUNKED_LSE_AG`) returns the per-chunk-gather merge bit
     for bit, eager and compiled, and the same check fails when the collapsed
     gather's rows are shifted by one.
  6. Packing the lse bits into the partial's all-to-all
     (`TPU_DCP_LSE_MERGE_A2A_PACK`) returns the two-exchange a2a merge bit for
     bit eager; compiled, where the merge fusion sees the received partial in
     another layout, it meets the a2a merge's own bar (one bf16 ulp of the f64
     reference, or no less exact than the a2a merge compiled the same way). The
     same check fails when the packed lse rows are rotated by one rank while
     the partial rows are left alone.
  7. Sending only the real tokens' rows (`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`, a
     ragged all-to-all sized from query_start_loc) returns the packed merge
     bit for bit on every real row and zero on padding rows, eager and
     compiled, at decode-sized buckets and real counts; the same check fails
     when the exchange is sized one row short.
  8. Sending the chunked merge's weighted partial in bf16
     (`TPU_DCP_MERGE_RS_BF16`; 1 = bf16 reduce-scatter, 2 = bf16 all-to-all
     plus a local f32 sum) is not bitwise: it is held to a max error in bf16
     ulps of sum_r w_r |o_r| against the f64 reference (1.5 for mode 2, a
     proof; 7 for mode 1) on hard and on spread partials, eager and compiled,
     and the same bound fails when one rank's weight is 25 % too large.

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
CHECK_TOKENS = 64
LAUNCH_TIMEOUT_SECONDS = 900

_ARM_DEFAULTS = {
    "TPU_DCP_LSE_MERGE_A2A": False,
    "TPU_DCP_MERGE_CHUNKED": 0,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": False,
    "TPU_DCP_LSE_MERGE_A2A_PACK": False,
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": False,
    "TPU_DCP_MERGE_RS_BF16": 0,
}
# name -> the env overrides of `_ARM_DEFAULTS` that make the arm.
ARMS = {
    "today": {},
    "a2a": {"TPU_DCP_LSE_MERGE_A2A": True},
    "chunk2": {"TPU_DCP_MERGE_CHUNKED": 2},
    "chunk4": {"TPU_DCP_MERGE_CHUNKED": 4},
    "chunk8": {"TPU_DCP_MERGE_CHUNKED": 8},
    "chunk8_ag1": {"TPU_DCP_MERGE_CHUNKED": 8, "TPU_DCP_MERGE_CHUNKED_LSE_AG": True},
    "a2a_pack": {"TPU_DCP_LSE_MERGE_A2A": True, "TPU_DCP_LSE_MERGE_A2A_PACK": True},
    "a2a_pack_real": {
        "TPU_DCP_LSE_MERGE_A2A": True,
        "TPU_DCP_LSE_MERGE_A2A_PACK": True,
        "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": True,
    },
    "chunk8_rsbf16": {"TPU_DCP_MERGE_CHUNKED": 8, "TPU_DCP_MERGE_RS_BF16": 1},
    "chunk8_ag1_rsbf16": {
        "TPU_DCP_MERGE_CHUNKED": 8,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
        "TPU_DCP_MERGE_RS_BF16": 1,
    },
    "chunk8_a2abf16": {"TPU_DCP_MERGE_CHUNKED": 8, "TPU_DCP_MERGE_RS_BF16": 2},
    "chunk8_ag1_a2abf16": {
        "TPU_DCP_MERGE_CHUNKED": 8,
        "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
        "TPU_DCP_MERGE_RS_BF16": 2,
    },
}
# bf16 wire (`TPU_DCP_MERGE_RS_BF16`): arm -> mode; held to a max error in bf16
# ulps of sum_r w_r |o_r| (mode 2: 1.5, a proof; mode 1: 7, 1.5x the CPU
# emulation's worst), not to today's merge.
RS_BF16_ARMS = {
    "chunk8_rsbf16": 1,
    "chunk8_ag1_rsbf16": 1,
    "chunk8_a2abf16": 2,
    "chunk8_ag1_a2abf16": 2,
}
RS_BF16_MAX_ULP = {1: 7.0, 2: 1.5}
# Arms that must return today's merge: same per-row arithmetic, other schedule.
CHUNK_ARMS = ("chunk2", "chunk4", "chunk8", "chunk8_ag1")
# Collapsed lse gathers (`TPU_DCP_MERGE_CHUNKED_LSE_AG`): bitwise chunk8.
LSE_AG_ARMS = ("chunk8_ag1",)
# lse packed into the partial's exchange (`TPU_DCP_LSE_MERGE_A2A_PACK`): bitwise a2a.
PACK_ARMS = ("a2a_pack",)


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


def _scale_ulp_stats(got, outputs, lses, rank, world):
    """|got - f64| in bf16 ulps of sum_r w_r |o_r| over this rank's heads."""
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
    scale = (np.abs(out) * w[..., None]).sum(axis=0)
    g = got.float().numpy().astype(np.float64)
    err = np.abs(g - ref)
    ulp = np.exp2(np.floor(np.log2(np.where(scale > 0, scale, 1.0))) - 7)
    e = np.where(scale > 0, err / ulp, np.where(err == 0, 0.0, np.inf))
    e = np.nan_to_num(e, nan=np.inf)
    return {
        "finite": bool(np.isfinite(g).all()),
        "max_ulp": float(e.max()),
        "mean_ulp": float(e[np.isfinite(e)].mean()) if np.isfinite(e).any() else 0.0,
        "frac_over_1ulp": float((e > 1.0).mean()),
    }


def _rs_bf16_checks(rank, world, cp_mla, merge):
    """TPU_DCP_MERGE_RS_BF16 arms (and today's chunk8 for scale) vs the f64
    reference on hard and on spread partials; the wrong-weight control."""
    import torch

    report = {}
    g = torch.Generator().manual_seed(4242)
    cases = {"hard": _global_partials(world, CHECK_TOKENS)}
    for spread in (0.5, 4.0):
        outputs = torch.randn(world, 256, HEADS, VDIM, generator=g).to(torch.bfloat16)
        lses = torch.randn(world, 256, HEADS, generator=g) * spread + 3
        cases[f"spread{spread}"] = (outputs, lses)
    for case, (outputs, lses) in cases.items():
        o_dev, l_dev = outputs[rank].to("tpu"), lses[rank].to("tpu")
        for arm in ("chunk8", "chunk8_ag1", *RS_BF16_ARMS):
            for how in ("eager", "compiled"):
                key = f"rsbf16/{case}/{arm}/{how}"
                try:
                    _set_arm(cp_mla, arm)
                    fn = merge if how == "eager" else _compile(merge)
                    got = fn(o_dev, l_dev).cpu()
                    report[key] = {
                        "dtype": str(got.dtype),
                        **_scale_ulp_stats(got, outputs, lses, rank, world),
                    }
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:400]}
    # Positive control: rank 1 weighs its partial 25 % too heavily.
    import math

    real = cp_mla._weighted_reduce_scatter

    def wrong(output, lse_f32, lses_g, group):
        if int(group.rank_in_group) == 1:
            lse_f32 = lse_f32 + math.log(1.25)
        return real(output, lse_f32, lses_g, group)

    cp_mla._weighted_reduce_scatter = wrong
    try:
        outputs, lses = cases["spread4.0"]
        o_dev, l_dev = outputs[rank].to("tpu"), lses[rank].to("tpu")
        for arm in RS_BF16_ARMS:
            for how in ("eager", "compiled"):
                key = f"control_wrong_weight/{arm}/{how}"
                try:
                    _set_arm(cp_mla, arm)
                    fn = merge if how == "eager" else _compile(merge)
                    got = fn(o_dev, l_dev).cpu()
                    report[key] = _scale_ulp_stats(got, outputs, lses, rank, world)
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:400]}
    finally:
        cp_mla._weighted_reduce_scatter = real
    return report


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


def _real_rows_checks(rank, world, cp_mla, merge):
    """TPU_DCP_LSE_MERGE_A2A_REAL_ROWS vs the packed merge, same compile mode:
    real rows bitwise, padding rows zero; and the one-row-short control."""
    import torch

    report = {}
    cases = [(16, 4), (16, 8), (16, 16), (32, 12), (64, 1), (16, 0)]
    for tokens, real in cases:
        outputs, lses = _global_partials(world, tokens)
        poisoned = outputs.clone()
        poisoned[~torch.isfinite(lses)] = float("nan")
        o_dev = poisoned[rank].to("tpu")
        l_dev = lses[rank].to("tpu")
        qsl = torch.tensor([0, real, real], dtype=torch.int32).to("tpu")
        for how in ("eager", "compiled"):
            key = f"real_rows/T{tokens}/n{real}/{how}"
            try:
                _set_arm(cp_mla, "a2a_pack")
                fn = merge if how == "eager" else _compile(merge)
                want = fn(o_dev, l_dev, query_start_loc=qsl).cpu()
                _set_arm(cp_mla, "a2a_pack_real")
                fn = merge if how == "eager" else _compile(merge)
                got = fn(o_dev, l_dev, query_start_loc=qsl).cpu()
                report[key] = {
                    "real_rows_bitwise_vs_pack": bool(
                        torch.equal(
                            got[:real].view(torch.int16), want[:real].view(torch.int16)
                        )
                    ),
                    "padding_rows_zero": bool((got[real:] == 0).all()),
                    "finite": bool(torch.isfinite(got.float()).all()),
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:400]}
    # Positive control: the exchange sized one row short.
    real_ragged = cp_mla._ragged_all_to_all
    cp_mla._ragged_all_to_all = lambda x, rows, g: real_ragged(x, rows - 1, g)
    try:
        tokens, real = 16, 5
        outputs, lses = _global_partials(world, tokens)
        o_dev, l_dev = outputs[rank].to("tpu"), lses[rank].to("tpu")
        qsl = torch.tensor([0, real], dtype=torch.int32).to("tpu")
        for how in ("eager", "compiled"):
            key = f"control_short_row/a2a_pack_real/{how}"
            try:
                _set_arm(cp_mla, "a2a_pack")
                fn = merge if how == "eager" else _compile(merge)
                want = fn(o_dev, l_dev, query_start_loc=qsl).cpu()
                _set_arm(cp_mla, "a2a_pack_real")
                fn = merge if how == "eager" else _compile(merge)
                got = fn(o_dev, l_dev, query_start_loc=qsl).cpu()
                report[key] = {
                    "real_rows_bitwise_vs_pack": bool(
                        torch.equal(
                            got[:real].view(torch.int16), want[:real].view(torch.int16)
                        )
                    )
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:400]}
    finally:
        cp_mla._ragged_all_to_all = real_ragged
    return report


def _nan_check(rank, world, cp_mla, merge, ref, outputs, lses, local):
    """The zero-weight NaN guard on hardware, a2a arm, eager and compiled."""
    import torch

    poisoned = outputs.clone()
    poisoned[~torch.isfinite(lses)] = float("nan")
    o_dev = outputs[rank].to("tpu")
    p_dev = poisoned[rank].to("tpu")
    l_dev = lses[rank].to("tpu")
    owned = torch.isfinite(lses[:, :, rank * local : (rank + 1) * local]).any(dim=0)
    report = {}
    real_guard = cp_mla._zero_nan
    _set_arm(cp_mla, "a2a")
    for how in ("eager", "compiled"):
        try:
            fn = merge if how == "eager" else _compile(merge)
            clean = fn(o_dev, l_dev).cpu()
            got = (merge if how == "eager" else _compile(merge))(p_dev, l_dev).cpu()
            c = _error_stats(got, ref)
            c["bitwise_vs_finite_input"] = bool(
                torch.equal(got.view(torch.int16), clean.view(torch.int16))
            )
            cp_mla._zero_nan = lambda o: o
            try:
                old = (merge if how == "eager" else _compile(merge))(o_dev, l_dev).cpu()
                ctl = (merge if how == "eager" else _compile(merge))(p_dev, l_dev).cpu()
            finally:
                cp_mla._zero_nan = real_guard
            same = clean.view(torch.int16) == old.view(torch.int16)
            c["owned_rows_bitwise_vs_unguarded"] = bool(same[owned].all())
            c["frac_bitwise_vs_unguarded"] = float(same.float().mean())
            c["unowned_rows_zero"] = bool((clean[~owned] == 0).all())
            report[f"nan/a2a/{how}"] = c
            report[f"control_unguarded_nan/a2a/{how}"] = _error_stats(ctl, ref)
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            traceback.print_exc()
            report[f"nan/a2a/{how}"] = {"error": repr(exc)[:400]}
        finally:
            cp_mla._zero_nan = real_guard
    return report


# Local-merge variants measured and checked beside the production one.
MERGE_VARIANTS = {"a2a_reduce_sum": _merge_reduce_sum, "a2a_unrolled": _merge_unrolled}


def _set_arm(cp_mla, arm):
    for name, value in {**_ARM_DEFAULTS, **ARMS[arm]}.items():
        setattr(cp_mla.envs, name, value)
    cp_mla.envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS = 1 << 30
    cp_mla.envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS = 0
    cp_mla.envs.TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS = 0


def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _check(rank, world, group, cp_mla):
    import torch

    outputs, lses = _global_partials(world, CHECK_TOKENS)
    local = HEADS // world
    ref = _reference(outputs, lses)[:, rank * local : (rank + 1) * local]
    o_dev = outputs[rank].to("tpu")
    l_dev = lses[rank].to("tpu")
    merge = cp_mla.merge_lse_partials_scatter_heads
    report = {}
    got_today = {}
    for arm in ARMS:  # "today" first
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
                if arm in ("today", "chunk8", "a2a"):
                    got_today[(arm, how)] = got
                if arm == "today":
                    got_today[how] = got
                if arm in PACK_ARMS and ("a2a", how) in got_today:
                    same = got.view(torch.int16) == got_today[("a2a", how)].view(
                        torch.int16
                    )
                    report[f"{arm}/{how}"]["bitwise_vs_a2a"] = bool(same.all())
                    report[f"{arm}/{how}"]["frac_bitwise_vs_a2a"] = float(
                        same.float().mean()
                    )
                elif arm in LSE_AG_ARMS and ("chunk8", how) in got_today:
                    report[f"{arm}/{how}"]["bitwise_vs_chunk8"] = bool(
                        torch.equal(
                            got.view(torch.int16),
                            got_today[("chunk8", how)].view(torch.int16),
                        )
                    )
                elif how in got_today and got.shape == got_today[how].shape:
                    same = got.view(torch.int16) == got_today[how].view(torch.int16)
                    report[f"{arm}/{how}"]["frac_bitwise_vs_today"] = float(
                        same.float().mean()
                    )
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[f"{arm}/{how}"] = {"error": repr(exc)[:400]}

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

    report.update(_nan_check(rank, world, cp_mla, merge, ref, outputs, lses, local))

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

    # Positive control for the packed exchange: the received lse rows (every
    # row after the partial's `T * local`) rotated by one source rank, the
    # partial rows untouched. The same check must now fail.
    real = cp_mla._all_to_all
    pack_rows = CHECK_TOKENS * local

    def _rotated_lse_rows(x, g):
        r = real(x, g)
        if r.dim() != 3 or r.shape[-1] != VDIM:  # only the packed exchange
            return r
        return torch.cat([r[:, :pack_rows], torch.roll(r[:, pack_rows:], 1, 0)], 1)

    cp_mla._all_to_all = _rotated_lse_rows
    try:
        for arm in PACK_ARMS:
            _set_arm(cp_mla, arm)
            for how in ("eager", "compiled"):
                key = f"control_lse_rows_rotated/{arm}/{how}"
                try:
                    fn = merge if how == "eager" else _compile(merge)
                    got = fn(o_dev, l_dev).cpu()
                    report[key] = _error_stats(got, ref)
                    report[key]["bitwise_vs_a2a"] = bool(
                        torch.equal(
                            got.view(torch.int16),
                            got_today[("a2a", how)].view(torch.int16),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:400]}
    finally:
        cp_mla._all_to_all = real

    report.update(_real_rows_checks(rank, world, cp_mla, merge))
    report.update(_rs_bf16_checks(rank, world, cp_mla, merge))

    # Positive control for the chunked merge: every chunk weighted with its
    # neighbour row's lse. The same check must now fail.
    real_rs = cp_mla.cp_lse_ag_out_rs
    cp_mla.cp_lse_ag_out_rs = lambda o, lse, g: real_rs(o, torch.roll(lse, 1, 0), g)
    try:
        _set_arm(cp_mla, "chunk4")
        for how in ("eager", "compiled"):
            try:
                fn = merge if how == "eager" else _compile(merge)
                got = fn(o_dev, l_dev).cpu()
                report[f"control_misaligned/chunk4/{how}"] = _error_stats(got, ref)
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[f"control_misaligned/chunk4/{how}"] = {"error": repr(exc)[:400]}
    finally:
        cp_mla.cp_lse_ag_out_rs = real_rs

    # Positive control for the collapsed gathers: the gathered lse rows
    # shifted by one, so every chunk is weighted with a neighbour's lse.
    real_ag = cp_mla._gather_lse
    cp_mla._gather_lse = lambda x, g: torch.roll(real_ag(x, g), 1, dims=1)
    try:
        for arm in LSE_AG_ARMS:
            _set_arm(cp_mla, arm)
            for how in ("eager", "compiled"):
                key = f"control_misaligned/{arm}/{how}"
                try:
                    fn = merge if how == "eager" else _compile(merge)
                    got = fn(o_dev, l_dev).cpu()
                    report[key] = _error_stats(got, ref)
                    report[key]["bitwise_vs_chunk8"] = bool(
                        torch.equal(
                            got.view(torch.int16),
                            got_today[("chunk8", how)].view(torch.int16),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    report[key] = {"error": repr(exc)[:400]}
    finally:
        cp_mla._gather_lse = real_ag
    return report


def _bench(rank, cp_mla, tokens_list, reps=10):
    """Per-merge device time: (wall of 1+K merges - wall of 1 merge) / K, each
    chained through the lse so none can be dropped or overlapped."""
    import torch

    def chain(k):
        def run(o, lse, qsl):
            for _ in range(k):
                y = cp_mla.merge_lse_partials_scatter_heads(o, lse, query_start_loc=qsl)
                # A one-element update of both inputs: nothing is loop
                # invariant, and the update itself costs nothing.
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
        real_merge = cp_mla._merge_received
        # Real rows: a b1-like step, a quarter of the bucket real.
        qsl = torch.tensor([0, max(1, tokens // 4)], dtype=torch.int32).to("tpu")
        arms = ("today", "a2a", *PACK_ARMS, "a2a_pack_real", *MERGE_VARIANTS)
        if tokens > 256:
            arms = os.environ.get("DCP_MERGE_BENCH_ARMS", "").split(",")
            arms = tuple(a for a in arms if a) or ("today", *CHUNK_ARMS, *RS_BF16_ARMS)
        for arm in arms:
            _set_arm(cp_mla, "a2a" if arm in MERGE_VARIANTS else arm)
            cp_mla._merge_received = MERGE_VARIANTS.get(arm, real_merge)
            try:
                times = {}
                for k in (1, 1 + reps):
                    fn = _compile(chain(k))
                    fn(o, lse, qsl).cpu()
                    fn(o, lse, qsl).cpu()
                    samples = []
                    for _ in range(5):
                        t0 = time.perf_counter()
                        fn(o, lse, qsl).cpu()
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
    return out


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
        # The real-rows exchange is a JAX op on the DCP mesh, which reads the
        # group from `distributed.dcp`; register it before any trace.
        from vllm_torchtpu.distributed import dcp as dcp_mod

        dcp_mod.get_dcp_group = lambda: group
        cp_mla.prebuild_ragged_all_to_all(world)
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
        for arm in CHUNK_ARMS:
            for how in ("eager", "compiled"):
                today, split = check[f"today/{how}"], check[f"{arm}/{how}"]
                assert "error" not in split, (r["rank"], arm, how, split)
                assert split["dtype"] == "torch.bfloat16", (r["rank"], arm, how)
                # Same per-row arithmetic: no less exact than today's merge.
                assert split["within_1ulp"] or (
                    split["frac_over_1ulp"] <= today["frac_over_1ulp"]
                    and split["max_abs_err"] <= today["max_abs_err"]
                ), (r["rank"], arm, how, today, split)
        for how in ("eager", "compiled"):
            c = check[f"nan/a2a/{how}"]
            assert "error" not in c, (r["rank"], how, c)
            assert c["finite"] and c["bitwise_vs_finite_input"], (r["rank"], how, c)
            assert c["owned_rows_bitwise_vs_unguarded"], (r["rank"], how, c)
            assert c["unowned_rows_zero"], (r["rank"], how, c)
            ctl = check[f"control_unguarded_nan/a2a/{how}"]
            assert not ctl["finite"], (
                "positive control: the unguarded merge must return NaN",
                r["rank"],
                how,
                ctl,
            )
        for arm in LSE_AG_ARMS:
            for how in ("eager", "compiled"):
                c = check[f"{arm}/{how}"]
                assert c.get("bitwise_vs_chunk8") is True, (r["rank"], arm, how, c)
                ctl = check[f"control_misaligned/{arm}/{how}"]
                assert "error" not in ctl, (r["rank"], arm, how, ctl)
                assert not ctl["within_1ulp"], (
                    "positive control: shifted collapsed lse must fail the check",
                    r["rank"],
                    arm,
                    how,
                    ctl,
                )
                assert not ctl["bitwise_vs_chunk8"], (r["rank"], arm, how, ctl)
        for how in ("eager", "compiled"):
            c = check[f"control_misaligned/chunk4/{how}"]
            assert "error" not in c, (r["rank"], how, c)
            assert not c["within_1ulp"], (
                "positive control: misaligned chunk lse must fail the check",
                r["rank"],
                how,
                c,
            )
        for arm in PACK_ARMS:
            for how in ("eager", "compiled"):
                c = check[f"{arm}/{how}"]
                assert "error" not in c, (r["rank"], arm, how, c)
                assert c["finite"], (r["rank"], arm, how, c)
                if how == "eager":
                    assert c.get("bitwise_vs_a2a") is True, (r["rank"], arm, how, c)
                a2a = check[f"a2a/{how}"]
                assert c["within_1ulp"] or (
                    c["frac_over_1ulp"] <= a2a["frac_over_1ulp"]
                    and c["max_abs_err"] <= a2a["max_abs_err"]
                ), (r["rank"], arm, how, a2a, c)
                ctl = check[f"control_lse_rows_rotated/{arm}/{how}"]
                assert "error" not in ctl, (r["rank"], arm, how, ctl)
                assert not ctl["within_1ulp"], (
                    "positive control: rotated packed lse rows must fail the check",
                    r["rank"],
                    arm,
                    how,
                    ctl,
                )
                assert not ctl["bitwise_vs_a2a"], (r["rank"], arm, how, ctl)
        for how in ("eager", "compiled"):
            keys = [k for k in check if k.startswith("real_rows/") and k.endswith(how)]
            assert len(keys) == 6, (r["rank"], how, keys)
            for k in keys:
                v = check[k]
                assert "error" not in v, (r["rank"], k, v)
                assert v["real_rows_bitwise_vs_pack"], (r["rank"], k, v)
                assert v["padding_rows_zero"] and v["finite"], (r["rank"], k, v)
            ctl = check[f"control_short_row/a2a_pack_real/{how}"]
            assert "error" not in ctl, (r["rank"], how, ctl)
            assert not ctl["real_rows_bitwise_vs_pack"], (
                "positive control: a one-row-short exchange must fail the check",
                r["rank"],
                how,
                ctl,
            )
        for arm, mode in RS_BF16_ARMS.items():
            bound = RS_BF16_MAX_ULP[mode]
            for how in ("eager", "compiled"):
                for case in ("hard", "spread0.5", "spread4.0"):
                    c = check[f"rsbf16/{case}/{arm}/{how}"]
                    assert "error" not in c, (r["rank"], case, arm, how, c)
                    assert c["dtype"] == "torch.bfloat16", (r["rank"], arm, how, c)
                    assert c["finite"], (r["rank"], case, arm, how, c)
                    assert c["max_ulp"] <= bound, (r["rank"], case, arm, how, c)
                ctl = check[f"control_wrong_weight/{arm}/{how}"]
                assert "error" not in ctl, (r["rank"], arm, how, ctl)
                assert ctl["max_ulp"] > bound, (
                    "positive control: a 25 % wrong weight must exceed the bound",
                    r["rank"],
                    arm,
                    how,
                    ctl,
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
