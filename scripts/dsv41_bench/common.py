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
"""Shared harness for server-free DeepSeek-V4.1 component replays.

Every ``bench_*.py`` in this directory is a single file that runs twice:

1. As the *parent* (no ``--worker``): prepares the TorchTPU slice environment
   once, spawns ``torch.distributed.run`` with ``--worker`` appended, then
   aggregates the per-rank JSON results and exits non-zero on any failure.
2. As a *worker* (``RANK=0..world_size-1``): calls :func:`init_worker`, builds
   its inputs, times them with :func:`time_steps`, optionally captures XProf
   via :func:`profile_steps`, and checks numerics with :func:`verify_outputs`.

Correctness is the gate: ``--verify`` is on by default and a mismatch raises
:class:`VerificationError`, which makes the worker (and so the parent) exit 1.

Wall-clock samples include host dispatch. Device time comes only from XProf
(:func:`summarize_xplanes`), never from host timers.

This module keeps torch/jax imports inside functions so that a worker can set
``LIBTPU_INIT_ARGS`` (``--dump-llo``) before libtpu is loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

WORLD_SIZE = 8
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH_STATE_ROOT = REPO_ROOT / ".dsv41_bench"
DEFAULT_GOLDEN_ROOT = pathlib.Path(
    os.environ.get("DSV41_BENCH_GOLDEN_DIR", BENCH_STATE_ROOT / "golden")
)
GIT_SHA_FILE = BENCH_STATE_ROOT / "git_sha"
LAUNCH_TIMEOUT_SECONDS = 1800
_PREPARED_KEYS = (
    "TORCH_TPU_XPROF_SESSION_ID",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
)
_PROFILE_SESSION_ENV = "DSV41_BENCH_PROFILE_SESSION_KEY"
sys.modules.setdefault("common", sys.modules[__name__])
sys.modules.setdefault("scripts.dsv41_bench.common", sys.modules[__name__])


# --------------------------------------------------------------------------
# Model geometry
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Dsv41Geometry:
    """DeepSeek-V4.1-Flash text-model geometry.

    Defaults mirror ``text_config`` of
    ``deepseek-ai/DeepSeek-V4.1-Flash`` snapshot
    ``2cba9e42aa026125f3ed06c6d98c1db82f7ca027``; use :meth:`from_hf_config`
    to re-derive them from a local ``config.json``.
    """

    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    num_key_value_heads: int = 1
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024
    o_groups: int = 8
    sliding_window: int = 128
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2304
    engram_head_dim: int = 256
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    engram_layer_ids: tuple[int, ...] = (1, 14)

    @classmethod
    def from_hf_config(cls, path: str | os.PathLike[str]) -> Dsv41Geometry:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        text = raw.get("text_config", raw)
        values: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name not in text:
                raise KeyError(
                    f"{path} has no text_config.{field.name}; pass a "
                    "DeepSeek-V4.1 config.json"
                )
            value = text[field.name]
            values[field.name] = tuple(value) if isinstance(value, list) else value
        return cls(**values)

    def heads_per_rank(self, tp_size: int) -> int:
        if self.num_attention_heads % tp_size:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} is not "
                f"divisible by tp_size={tp_size}"
            )
        return self.num_attention_heads // tp_size

    @property
    def candidate_positions(self) -> int:
        return self.candidate_topk_blocks * self.candidate_block_size


# --------------------------------------------------------------------------
# Roofline
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tpu7xRoofline:
    """Per-TensorCore tpu7x peak rates at the 1.87 GHz operational P-state.

    Source: ``tpu_knowledge/references/hardware_architecture_and_memory.md``
    section 4.1 (BF16/FP8 MXU peak, chip HBM bandwidth 6,273 GB/s split over
    2 TensorCores) and section 1 (intra-host ICI 186.68 GB/s per TensorCore,
    D2D partner DMA 600 GB/s per die). These are upper bounds; a bench must
    report measured time next to them, never substitute for it.

    [UNVERIFIED] The collective formulas charge every byte to the
    186.68 GB/s per-TensorCore ICI share and ignore the 600 GB/s D2D hop to
    the partner die. On TPU7x-8 an 8-rank all-gather to bf16[2048, 5120] took
    0.091 ms device time vs. 0.098 ms from :meth:`all_gather_s`, so treat
    collective numbers as a reference scale, not a strict lower bound.
    """

    clock_hz: float = 1.87e9
    bf16_flops_per_s: float = 980.48e12
    fp8_flops_per_s: float = 1960.95e12
    valu_flops_per_s: float = 3.83e12
    hbm_bytes_per_s: float = 6273.0e9 / 2
    ici_bytes_per_s: float = 186.68e9
    d2d_bytes_per_s: float = 600.0e9

    def compute_s(self, flops: float, dtype: Literal["bf16", "fp8"] = "bf16") -> float:
        peak = self.bf16_flops_per_s if dtype == "bf16" else self.fp8_flops_per_s
        return flops / peak

    def hbm_s(self, nbytes: float) -> float:
        return nbytes / self.hbm_bytes_per_s

    def all_reduce_s(self, nbytes: float, n: int) -> float:
        """Bandwidth-optimal ring all-reduce: ``2 (n-1)/n * bytes / bw``."""
        return 2 * (n - 1) / n * nbytes / self.ici_bytes_per_s

    def reduce_scatter_s(self, nbytes: float, n: int) -> float:
        """``nbytes`` is the full (pre-scatter) tensor size per rank."""
        return (n - 1) / n * nbytes / self.ici_bytes_per_s

    def all_gather_s(self, nbytes: float, n: int) -> float:
        """``nbytes`` is the full (post-gather) tensor size per rank."""
        return (n - 1) / n * nbytes / self.ici_bytes_per_s

    def all_to_all_s(self, nbytes: float, n: int) -> float:
        """``nbytes`` is the per-rank send buffer; ``1/n`` stays local."""
        return (n - 1) / n * nbytes / self.ici_bytes_per_s


@dataclasses.dataclass(frozen=True)
class RooflineEstimate:
    """Work of one op; the lower bound is the max of its resource times."""

    name: str
    flops: float = 0.0
    hbm_bytes: float = 0.0
    comm_s: float = 0.0
    dtype: Literal["bf16", "fp8"] = "bf16"

    def bound_s(self, hw: Tpu7xRoofline) -> dict[str, float]:
        parts = {
            "compute_s": hw.compute_s(self.flops, self.dtype),
            "hbm_s": hw.hbm_s(self.hbm_bytes),
            "comm_s": self.comm_s,
        }
        parts["lower_bound_s"] = max(parts.values())
        parts["arith_intensity"] = (
            self.flops / self.hbm_bytes if self.hbm_bytes else math.inf
        )
        return parts

    def report(self, hw: Tpu7xRoofline, measured_ms: float | None) -> dict[str, Any]:
        bound = self.bound_s(hw)
        out: dict[str, Any] = {"name": self.name, **bound}
        out["lower_bound_ms"] = bound["lower_bound_s"] * 1e3
        if measured_ms is not None:
            out["measured_ms"] = measured_ms
            out["roofline_efficiency"] = (
                out["lower_bound_ms"] / measured_ms if measured_ms > 0 else math.nan
            )
        return out


# --------------------------------------------------------------------------
# Numerical verification (``--verify``)
# --------------------------------------------------------------------------


class VerificationError(AssertionError):
    """A tensor failed its golden/reference comparison."""


@dataclasses.dataclass(frozen=True)
class MatchCriteria:
    """How strictly ``actual`` must agree with ``expected``.

    ``exact``: bitwise equal (NaN payloads and ``+0.0``/``-0.0`` included).
    ``lossless``: same non-finite mask and, on finite entries,
    ``cos_sim >= min_cos`` and ``rel_l2 <= max_rel_l2``.
    ``topk``: integer index rows; per-row overlap of valid (``>= 0``) indices
    averaged over rows must be ``>= min_topk_overlap``.
    """

    kind: Literal["exact", "lossless", "topk"] = "exact"
    min_cos: float = 0.9999
    max_rel_l2: float = 1e-3
    min_topk_overlap: float = 0.99


EXACT = MatchCriteria("exact")
LOSSLESS = MatchCriteria("lossless")
TOPK = MatchCriteria("topk")

_INT_VIEW = {1: "int8", 2: "int16", 4: "int32", 8: "int64"}


def _bitwise_equal(actual, expected) -> bool:
    import torch

    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    if actual.dtype.is_floating_point:
        view = getattr(torch, _INT_VIEW[actual.element_size()])
        return torch.equal(
            actual.contiguous().view(view), expected.contiguous().view(view)
        )
    return torch.equal(actual, expected)


def topk_overlap(actual, expected) -> tuple[float, float]:
    """Mean and min per-row overlap of valid indices (``-1`` is padding).

    Rows where ``expected`` has no valid index count as overlap 1.0 only if
    ``actual`` also has none.
    """
    import torch

    if actual.shape != expected.shape or actual.dim() != 2:
        raise ValueError(
            f"topk_overlap expects equal 2-D shapes, got {tuple(actual.shape)} "
            f"vs {tuple(expected.shape)}"
        )
    a = actual.to(torch.int64)
    e = expected.to(torch.int64)
    e_valid = e >= 0
    a_sorted, _ = torch.sort(
        torch.where(a >= 0, a, torch.iinfo(torch.int64).max), dim=1
    )
    pos = torch.searchsorted(a_sorted, e.contiguous()).clamp_max(a.shape[1] - 1)
    hit = (torch.gather(a_sorted, 1, pos) == e) & e_valid
    n_valid = e_valid.sum(dim=1)
    a_valid = (a >= 0).sum(dim=1)
    per_row = torch.where(
        n_valid > 0,
        hit.sum(dim=1).double() / n_valid.clamp_min(1).double(),
        (a_valid == 0).double(),
    )
    return float(per_row.mean()), float(per_row.min())


def compare_tensors(actual, expected, criteria: MatchCriteria) -> dict[str, Any]:
    """Return comparison metrics plus ``ok``; never raises on mismatch."""
    import torch

    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    m: dict[str, Any] = {
        "kind": criteria.kind,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
    }
    if actual.shape != expected.shape:
        m.update(
            ok=False, reason=f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
        return m
    if actual.dtype != expected.dtype:
        m.update(ok=False, reason=f"dtype {actual.dtype} != {expected.dtype}")
        return m
    m["bitwise_equal"] = _bitwise_equal(actual, expected)

    if criteria.kind == "exact":
        m["ok"] = m["bitwise_equal"]
        if not m["ok"]:
            diff = actual != expected
            if actual.dtype.is_floating_point:
                diff = diff & ~(actual.isnan() & expected.isnan())
            m["num_mismatch"] = int(diff.sum())
            if actual.dtype.is_floating_point:
                m["max_abs_diff"] = float(
                    (actual.double() - expected.double()).abs().nan_to_num(0.0).max()
                )
        return m

    if criteria.kind == "topk":
        mean, worst = topk_overlap(actual, expected)
        m.update(topk_overlap_mean=mean, topk_overlap_min=worst)
        m["ok"] = mean >= criteria.min_topk_overlap
        return m

    if not actual.dtype.is_floating_point:
        raise ValueError(f"lossless criteria needs a float tensor, got {actual.dtype}")
    a = actual.double()
    e = expected.double()
    a_fin = torch.isfinite(a)
    e_fin = torch.isfinite(e)
    m["nonfinite_mask_equal"] = bool(torch.equal(a_fin, e_fin))
    both = a_fin & e_fin
    av = a[both]
    ev = e[both]
    diff = av - ev
    m["max_abs_diff"] = float(diff.abs().max()) if diff.numel() else 0.0
    e_norm = float(ev.norm())
    a_norm = float(av.norm())
    m["rel_l2"] = float(diff.norm()) / e_norm if e_norm > 0 else float(diff.norm())
    if e_norm > 0 and a_norm > 0:
        m["cos_sim"] = float((av * ev).sum()) / (a_norm * e_norm)
    else:
        m["cos_sim"] = 1.0 if e_norm == a_norm else 0.0
    m["num_compared"] = int(both.sum())
    m["num_nan_expected"] = int(e.isnan().sum())
    m["num_nan_actual"] = int(a.isnan().sum())
    m["ok"] = (
        m["nonfinite_mask_equal"]
        and m["cos_sim"] >= criteria.min_cos
        and m["rel_l2"] <= criteria.max_rel_l2
    )
    # NaN is never a valid reference value (masked -inf is), and a check with
    # nothing finite to compare would pass vacuously.
    if m["num_nan_expected"] or m["num_nan_actual"]:
        m.update(ok=False, reason="NaN present in actual/expected")
    elif m["num_compared"] == 0 and expected.numel() > 0:
        m.update(ok=False, reason="no finite elements to compare")
    return m


def verify_outputs(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    criteria: MatchCriteria | Mapping[str, MatchCriteria],
    *,
    label: str = "",
) -> dict[str, dict[str, Any]]:
    """Compare every named tensor; raise :class:`VerificationError` on failure."""
    if set(actual) != set(expected):
        raise VerificationError(
            f"{label}: output names differ: actual={sorted(actual)} "
            f"expected={sorted(expected)}"
        )
    report = {}
    for name in sorted(expected):
        c = criteria[name] if isinstance(criteria, Mapping) else criteria
        report[name] = compare_tensors(actual[name], expected[name], c)
    failed = {k: v for k, v in report.items() if not v["ok"]}
    if failed:
        raise VerificationError(f"{label}: verification failed: {json.dumps(failed)}")
    return report


def golden_path(golden_dir: pathlib.Path, case: str, rank: int) -> pathlib.Path:
    return golden_dir / case / f"rank{rank}.pt"


def save_golden(
    path: pathlib.Path, tensors: Mapping[str, Any], meta: Mapping[str, Any]
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tensors": {k: v.detach().cpu() for k, v in tensors.items()},
        "meta": dict(meta),
    }
    torch.save(payload, path)


def load_golden(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not path.exists():
        raise FileNotFoundError(
            f"no golden at {path}; record one from the reference baseline with "
            "--update-golden"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["tensors"], payload["meta"]


def check_against_golden(
    ctx: WorkerContext,
    args: argparse.Namespace,
    case: str,
    outputs: Mapping[str, Any],
    criteria: MatchCriteria | Mapping[str, MatchCriteria],
) -> dict[str, Any]:
    """``--update-golden`` writes, ``--verify`` compares, otherwise skip."""
    path = golden_path(args.golden_dir, case, ctx.rank)
    meta = {"case": case, "git_sha": git_sha(), "argv": sys.argv[1:]}
    if args.update_golden:
        save_golden(path, outputs, meta)
        return {"golden": "written", "path": str(path)}
    if not args.verify:
        return {"golden": "skipped"}
    golden, golden_meta = load_golden(path)
    report = verify_outputs(outputs, golden, criteria, label=f"{case} rank{ctx.rank}")
    return {
        "golden": "verified",
        "path": str(path),
        "golden_meta": golden_meta,
        "report": report,
    }


def git_sha() -> str:
    """HEAD SHA (``+dirty`` if modified); synced copies read ``GIT_SHA_FILE``."""
    if not (REPO_ROOT / ".git").exists():
        if not GIT_SHA_FILE.exists():
            raise FileNotFoundError(
                f"{REPO_ROOT} is not a git checkout and has no {GIT_SHA_FILE}; "
                "write the target commit SHA to .dsv41_bench/git_sha"
            )
        return GIT_SHA_FILE.read_text(encoding="utf-8").strip()
    sha = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=9", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src", "scripts"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{sha}+dirty" if dirty else sha


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser, subops: Sequence[str]) -> None:
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-dir", type=pathlib.Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--num-seqs", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=90112)
    parser.add_argument("--tp-size", type=int, default=WORLD_SIZE)
    parser.add_argument("--subop", choices=list(subops), default=subops[0])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare outputs to golden/reference and exit 1 on mismatch",
    )
    parser.add_argument("--update-golden", action="store_true")
    parser.add_argument("--golden-dir", type=pathlib.Path, default=DEFAULT_GOLDEN_ROOT)
    parser.add_argument(
        "--profile",
        type=pathlib.Path,
        default=None,
        help="capture --profile-steps steps of XProf from all ranks into DIR",
    )
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument(
        "--trace-mode",
        choices=["TRACE_ONLY_XLA", "TRACE_COMPUTE"],
        default="TRACE_COMPUTE",
    )
    parser.add_argument(
        "--dump-llo",
        type=pathlib.Path,
        default=None,
        help="write HLO/LLO text dumps to DIR/rank<N>/ (disables compile cache)",
    )


def resolve_result_dir(args: argparse.Namespace, bench_name: str) -> pathlib.Path:
    if args.result_dir is not None:
        return args.result_dir.resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        BENCH_STATE_ROOT / "results" / f"{bench_name}_{args.subop}_{stamp}"
    ).resolve()


# --------------------------------------------------------------------------
# Parent launcher
# --------------------------------------------------------------------------


def _llo_env(dump_root: pathlib.Path, rank: int, env: dict[str, str]) -> None:
    """Point XLA (``XLA_FLAGS``) and libtpu (``LIBTPU_INIT_ARGS``) at a dir.

    Split per ``tpu_knowledge`` compiler_llo_and_cycle_analysis.md section
    7.1: ``--xla_jf_*`` flags in ``XLA_FLAGS`` abort the XLA flag parser.
    """
    llo_dir = dump_root / f"rank{rank}"
    llo_dir.mkdir(parents=True, exist_ok=True)
    env["XLA_FLAGS"] = (
        f"{env.get('XLA_FLAGS', '')} --xla_dump_to={llo_dir} --xla_dump_hlo_as_text"
    ).strip()
    env["LIBTPU_INIT_ARGS"] = (
        f"{env.get('LIBTPU_INIT_ARGS', '')} --xla_jf_dump_to={llo_dir} "
        "--xla_jf_dump_llo_text=true --xla_jf_dump_hlo_text=true "
        "--xla_xprof_register_llo_debug_info=true"
    ).strip()
    env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] = "disabled"


def prepare_worker_env(args: argparse.Namespace) -> dict[str, str]:
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    saved = {k: os.environ.get(k) for k in _PREPARED_KEYS}
    try:
        for key in _PREPARED_KEYS:
            os.environ.pop(key, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    env.setdefault("PJRT_DEVICE", "TPU")
    env.setdefault("VLLM_TARGET_DEVICE", "tpu")
    env.setdefault("VLLM_PLUGINS", "torchtpu")
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env["LIBTPU_INIT_ARGS"] = env.get("LIBTPU_INIT_ARGS", "").replace(
        "--xla_tpu_use_enhanced_launch_barrier=true",
        "--xla_tpu_use_enhanced_launch_barrier=false",
    )
    if args.profile is not None:
        env[_PROFILE_SESSION_ENV] = f"dsv41_{os.getpid()}_{time.time_ns()}"
    return env


def launch(script: str, argv: Sequence[str], result_dir: pathlib.Path, args) -> int:
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(pathlib.Path(script).resolve()),
        *argv,
        "--worker",
        "--result-dir",
        str(result_dir),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=prepare_worker_env(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_path = result_dir / "workers.log"
    assert process.stdout is not None
    deadline = time.monotonic() + LAUNCH_TIMEOUT_SECONDS
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            if line.startswith("[rank0]") or "Error" in line or "error" in line:
                sys.stdout.write(line)
            if time.monotonic() > deadline:
                os.killpg(process.pid, signal.SIGTERM)
                raise TimeoutError(
                    f"workers exceeded {LAUNCH_TIMEOUT_SECONDS}s; see {log_path}"
                )
    return process.wait()


def aggregate(result_dir: pathlib.Path) -> dict[str, Any]:
    """Merge rank JSONs; a distributed step ends when its slowest rank ends."""
    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    summary: dict[str, Any] = {"result_dir": str(result_dir), "errors": errors}
    timed = [r for r in results if "wall_samples_ms" in r]
    if len(timed) == WORLD_SIZE:
        n = min(len(r["wall_samples_ms"]) for r in timed)
        steps = [max(r["wall_samples_ms"][i] for r in timed) for i in range(n)]
        summary["wall_step_max_over_ranks_ms"] = steps
        summary["wall_median_ms"] = statistics.median(steps)
        summary["wall_min_ms"] = min(steps)
    if results and "error" not in results[0]:
        summary["rank0"] = {
            k: v for k, v in results[0].items() if k != "wall_samples_ms"
        }
    summary["hbm_by_rank"] = {
        str(r["rank"]): r.get("hbm") for r in results if r.get("hbm") is not None
    }
    profile_dir = results[0].get("profile_dir") if results else None
    if profile_dir and not errors:
        summary["xprof"] = summarize_xplanes(pathlib.Path(profile_dir))
    return summary


def run_main(
    bench_name: str,
    parser: argparse.ArgumentParser,
    worker_fn: Callable[[WorkerContext, argparse.Namespace], dict[str, Any]],
) -> None:
    args = parser.parse_args()
    if args.worker:
        _worker_entry(args, worker_fn)
        return
    result_dir = resolve_result_dir(args, bench_name)
    argv = [a for a in sys.argv[1:]]
    returncode = launch(sys.argv[0], argv, result_dir, args)
    summary = aggregate(result_dir)
    summary.update(
        bench=bench_name, returncode=returncode, git_sha=git_sha(), argv=argv
    )
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "xprof"}, indent=2))
    if "xprof" in summary:
        print_xprof_table(summary["xprof"])
    if returncode != 0 or summary["errors"]:
        print(f"FAILED: see {result_dir}/workers.log", file=sys.stderr)
        raise SystemExit(1)
    print(f"OK: {result_dir}/summary.json")


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------


@dataclasses.dataclass
class WorkerContext:
    rank: int
    world_size: int
    tp_size: int
    cpu_group: Any
    result_dir: pathlib.Path


def _worker_entry(args: argparse.Namespace, worker_fn) -> None:
    rank = int(os.environ["RANK"])
    if args.dump_llo is not None:
        _llo_env(args.dump_llo.resolve(), rank, os.environ)  # type: ignore[arg-type]
    result_path = args.result_dir / f"rank_{rank}.json"
    result: dict[str, Any]
    initialized = False
    try:
        ctx = init_worker(args)
        initialized = True
        result = {"rank": rank, **worker_fn(ctx, args)}
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result = {"rank": rank, "error": repr(exc)}
    finally:
        result_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        if initialized:
            import torch.distributed as dist

            dist.destroy_process_group()
    if "error" in result:
        raise SystemExit(1)


def init_worker(args: argparse.Namespace) -> WorkerContext:
    """Bring up ``tpu_dist`` plus vLLM's world and TP groups on this rank.

    vLLM collectives (``tensor_model_parallel_all_reduce`` etc.) used by the
    ``deepseek_v41/v2`` modules resolve through these groups.
    """
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % args.tp_size:
        raise ValueError(
            f"world_size={world_size} not divisible by tp_size={args.tp_size}"
        )
    dist.init_process_group(backend="tpu_dist")
    torch.empty((1,), device="tpu").cpu()
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(args.tp_size, 1)
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    torch.manual_seed(args.seed)
    return WorkerContext(
        rank=rank,
        world_size=world_size,
        tp_size=args.tp_size,
        cpu_group=cpu_group,
        result_dir=args.result_dir,
    )


def synchronize(tensors: Any) -> None:
    from torch_tpu._internal import sync

    flat = _flatten_tensors(tensors)
    sync.synchronize(flat, wait=True)


def model_wrapper_context():
    """``set_vllm_model_wrapper_context`` with the runner's worker mesh.

    Mirrors ``TPUModelRunner._create_mesh_for_parallelism``: a one-device
    ``("model",)`` mesh on this rank's local TPU, so ``deepseek_v41/v2`` ops
    that call ``get_vllm_model_wrapper_context().mesh`` build as in serving.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = Mesh(
        np.asarray(jax.local_devices()[:1]).reshape((1,)), axis_names=("model",)
    )
    return set_vllm_model_wrapper_context(mesh=mesh)


def _flatten_tensors(obj: Any) -> list[Any]:
    import torch

    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, Mapping):
        return [t for v in obj.values() for t in _flatten_tensors(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _flatten_tensors(v)]
    return []


def time_steps(
    ctx: WorkerContext,
    step: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> tuple[list[float], Any]:
    """Wall-clock per step (barrier -> dispatch -> device sync), host ms.

    Includes host dispatch; use XProf for device-kernel time.
    """
    import torch.distributed as dist

    out = None
    for _ in range(warmup):
        out = step()
        synchronize(out)
    samples = []
    for _ in range(iters):
        dist.barrier(group=ctx.cpu_group)
        start = time.perf_counter()
        out = step()
        synchronize(out)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples, out


def profile_steps(
    ctx: WorkerContext,
    args: argparse.Namespace,
    step: Callable[[], Any],
    *,
    label: str,
) -> str | None:
    """Capture ``args.profile_steps`` compiled steps from every rank.

    Uses ``torch.profiler`` + ``TpuProfilerConfig`` (never
    ``jax.profiler.start_trace``; see the tpu-profiling skill) and merges the
    per-rank captures into one run via ``vllm_torchtpu.profiler_trace``.
    """
    if args.profile is None:
        return None
    import torch
    import torch.distributed as dist
    from torch_tpu._internal.profiler import TpuProfilerConfig

    from vllm_torchtpu import profiler_trace

    profile_dir = str(args.profile.resolve())
    session_key = os.environ[_PROFILE_SESSION_ENV]
    pathlib.Path(profile_dir).mkdir(parents=True, exist_ok=True)
    capture_dir = profiler_trace.rank_capture_dir(profile_dir, ctx.rank)
    canonical_ts = profiler_trace.resolve_canonical_dst_ts(
        profile_dir, ctx.rank, session_key=session_key
    )
    pathlib.Path(capture_dir).mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"tpu_trace_mode": args.trace_mode}
    if args.trace_mode == "TRACE_COMPUTE":
        options.update(
            tpu_num_sparse_cores_to_trace=1, tpu_num_sparse_core_tiles_to_trace=1
        )
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            dir_name=capture_dir, use_gzip=True
        ),
        experimental_config=TpuProfilerConfig(
            run_dir=capture_dir,
            host_tracer_level=2,
            device_tracer_level=1,
            python_tracer_level=1,
            experimental_options=options,
        ),
    )
    dist.barrier(group=ctx.cpu_group)
    with profiler:
        dist.barrier(group=ctx.cpu_group)
        for i in range(args.profile_steps):
            with torch.profiler.record_function(f"{label}_step{i}"):
                synchronize(step())
    dist.barrier(group=ctx.cpu_group)
    profiler_trace.merge_rank_capture(
        capture_dir, profile_dir, canonical_ts, ctx.rank, world_size=ctx.world_size
    )
    dist.barrier(group=ctx.cpu_group)
    if ctx.rank == 0:
        profiler_trace.clear_canonical_ts_marker(profile_dir, session_key)
    return profile_dir


def hbm_stats() -> dict[str, int]:
    """HBM of this rank's TensorCore via ``torch.accelerator``."""
    import torch

    free, total = torch.accelerator.get_memory_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(total - free),
    }


def gather_object(ctx: WorkerContext, obj: Any) -> list[Any]:
    import torch.distributed as dist

    out: list[Any] = [None] * ctx.world_size
    dist.all_gather_object(out, obj, group=ctx.cpu_group)
    return out


# --------------------------------------------------------------------------
# XProf post-processing (offline, parent process)
# --------------------------------------------------------------------------


def summarize_xplanes(profile_dir: pathlib.Path, *, top: int = 25) -> dict[str, Any]:
    """Per-op device time from ``XLA Ops`` lines of every TensorCore plane.

    Returns, for each op name, the total ns per TensorCore plane, and the
    per-plane ``XLA Modules`` totals. Divide by ``--profile-steps`` for
    per-step numbers. SparseCore planes are reported separately.
    """
    from jax.profiler import ProfileData

    planes: dict[str, dict[str, Any]] = {}
    for xplane in sorted(profile_dir.rglob("*.xplane.pb")):
        data = ProfileData.from_file(str(xplane))
        for plane in data.planes:
            if not plane.name.startswith("/device:TPU:"):
                continue
            key = f"{xplane.name}:{plane.name}"
            ops: dict[str, int] = {}
            modules: dict[str, int] = {}
            for line in plane.lines:
                target = (
                    ops
                    if line.name == "XLA Ops"
                    else modules
                    if line.name == "XLA Modules"
                    else None
                )
                if target is None:
                    continue
                for event in line.events:
                    target[event.name] = target.get(event.name, 0) + int(
                        event.duration_ns
                    )
            if ops or modules:
                planes[key] = {
                    "sparse_core": "SparseCore" in plane.name,
                    "module_ns": modules,
                    "top_ops_ns": dict(
                        sorted(ops.items(), key=lambda kv: -kv[1])[:top]
                    ),
                    "total_op_ns": sum(ops.values()),
                }
    return {"profile_dir": str(profile_dir), "planes": planes}


def print_xprof_table(xprof: Mapping[str, Any]) -> None:
    for key, plane in xprof["planes"].items():
        if plane["sparse_core"]:
            continue
        print(f"\n== {key}  total_op={plane['total_op_ns'] / 1e6:.3f} ms")
        for name, ns in plane["module_ns"].items():
            print(f"   module {ns / 1e6:9.3f} ms  {name}")
        for name, ns in list(plane["top_ops_ns"].items())[:10]:
            print(f"   op     {ns / 1e6:9.3f} ms  {name}")
        break
