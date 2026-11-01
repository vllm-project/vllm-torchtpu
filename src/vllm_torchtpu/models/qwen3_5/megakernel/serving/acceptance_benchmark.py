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
"""Single-allocation interleaved serving acceptance benchmark for Qwen3.5.

Compares a reference server (`Arm A`: either a pinned `candidate30` container image
or stock `vllm serve` with the megakernel disabled) against the Qwen3.5 megakernel
template (`Arm B`) using **separate server processes** alternating `A / B / A / B`
on the same `v7x-8` TPU allocation across `c8`, `c16`, and `c32` (`900 s` per sweep).

Running separate `vllm serve` processes on one allocation exercises both the
megakernel and surrounding stock serving paths (prefill, `fp32` conv state in the
`640`-token unified slot with `--mamba-ssm-cache-dtype bfloat16`, rejection
sampling, prefix caching, async scheduling, and KV cache offloading). Arm B passes
template-specific settings (`profile`, `mode`) under `megakernel.options` alongside
top-level runtime keys (`plugin`, `stats_period_s`).

Accuracy (`GSM8K`, `n=1319`) MUST run as a **separate pass** (`run-gsm8k` or
`rejection_sample_mode="real"`) on **both** Arm A and Arm B with **real rejection
sampling** (`MTP=3` enabled, synthetic acceptance OFF, greedy `temperature=0.0`).
Synthetic acceptance (`rejection_sample_method="synthetic"`) fixes draft acceptance
rates for deterministic latency/throughput benchmarking and does NOT produce the
model's real token outputs.

This module uses only the Python standard library so importing the template
package never loads `torch`, `jax`, or `vllm`.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import pathlib
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_PROFILE = "qwen35-candidate30-fused"
DEFAULT_SWEEP_DURATION_S = 900
DEFAULT_PAIRS = 3
DEFAULT_SYNTHETIC_ACCEPTANCE_LEN = 3.39
DEFAULT_ACCEPTANCE_LEN_TOL = 0.04
DEFAULT_C16_BUNCHED_THRESHOLD_PCT = 6.0
REF_IMAGE_ENV = "MK_ACCEPTANCE_REFERENCE_IMAGE"
GSM8K_GREEDY_SAMPLING_PARAMS: dict[str, Any] = {"temperature": 0.0}

_GIB = 1024**3


@dataclasses.dataclass(frozen=True)
class ConcurrencySpec:
    """Serving configuration per client concurrency level matching `candidate30`."""

    name: str
    concurrency: int
    max_num_seqs: int
    kv_offload_enabled: bool
    kv_offload_bytes: int
    duration_s: int = DEFAULT_SWEEP_DURATION_S


CONCURRENCY_SPECS: dict[str, ConcurrencySpec] = {
    "c8": ConcurrencySpec(
        name="c8",
        concurrency=8,
        max_num_seqs=16,
        kv_offload_enabled=False,
        kv_offload_bytes=0,
    ),
    "c16": ConcurrencySpec(
        name="c16",
        concurrency=16,
        max_num_seqs=16,
        kv_offload_enabled=True,
        kv_offload_bytes=400 * _GIB,
    ),
    "c32": ConcurrencySpec(
        name="c32",
        concurrency=32,
        max_num_seqs=32,
        kv_offload_enabled=True,
        kv_offload_bytes=500 * _GIB,
    ),
}


@dataclasses.dataclass(frozen=True)
class StepMixSummary:
    """Aggregated megakernel step-mix counters parsed from a server sweep log."""

    mk_steps: int
    total_steps: int
    mk_pct: float
    bucket_counts: dict[str, int]
    bucket_shares_pct: dict[str, float]
    decline_codes: tuple[str, ...]
    mid_verify_share_pct: float
    packing_mode: str


@dataclasses.dataclass(frozen=True)
class SweepRecord:
    """Normalized metrics and validity metadata for one `900 s` serving sweep."""

    sweep_id: str
    concurrency: str
    pair_index: int
    arm: str  # "reference" (A) or "candidate" (B)
    arm_label: str
    numerics: str  # "approx" or "exact"
    expect_megakernel: bool
    duration_s: float
    output_tok_per_s: float
    itl_p50_ms: float
    itl_p90_ms: float
    itl_p99_ms: float
    itl_mean_ms: float
    records_error_dropped: int
    server_healthy_after: bool
    spec_mean_acceptance_length: float | None
    prefix_cache_hit_rate: float | None
    step_mix: StepMixSummary
    valid: bool = True
    invalid_reasons: tuple[str, ...] = ()


_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def t_crit_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for `df` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    z = 1.95996
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df * df)
    )


def default_plugin_module() -> str:
    """Infer the enclosing template package path so copied templates work unchanged."""
    pkg = __package__ or ""
    if pkg.endswith(".serving"):
        return pkg.rsplit(".", 1)[0]
    folder = pathlib.Path(__file__).resolve().parent.parent
    parts: list[str] = []
    cur = folder
    while (cur / "__init__.py").is_file() and cur.name.isidentifier():
        parts.append(cur.name)
        cur = cur.parent
    if parts:
        return ".".join(reversed(parts))
    return "vllm_torchtpu.models.qwen3_5.megakernel"


def profile_numerics(profile: str) -> str:
    """Classify numerics tier for a Qwen3.5 profile name (`exact` vs `approx`)."""
    if profile in ("default", "qwen35-baseline"):
        return "exact"
    return "approx"


def _common_vllm_serve_args(
    spec: ConcurrencySpec,
    *,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    rejection_sample_mode: str = "synthetic",
    synthetic_acceptance_length: float = DEFAULT_SYNTHETIC_ACCEPTANCE_LEN,
) -> list[str]:
    if rejection_sample_mode not in ("synthetic", "real"):
        raise ValueError(
            f"Unknown rejection_sample_mode {rejection_sample_mode!r}; "
            "expected 'synthetic' (perf sweeps) or 'real' (GSM8K accuracy pass)."
        )
    spec_dict: dict[str, Any] = {
        "method": "mtp",
        "num_speculative_tokens": 3,
    }
    if rejection_sample_mode == "synthetic":
        spec_dict["rejection_sample_method"] = "synthetic"
        spec_dict["synthetic_acceptance_length"] = round(
            float(synthetic_acceptance_length), 4
        )
    spec_cfg = json.dumps(
        spec_dict,
        separators=(",", ":"),
        sort_keys=True,
    )
    args = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--kv-cache-dtype",
        "fp8",
        "--mamba-cache-mode",
        "align",
        "--block-size",
        "256",
        "--max-model-len",
        "262144",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        str(spec.max_num_seqs),
        "--enable-prefix-caching",
        "--async-scheduling",
        "--prefill-schedule-interval",
        "1",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--speculative-config",
        spec_cfg,
    ]
    if spec.kv_offload_enabled and spec.kv_offload_bytes > 0:
        offload_cfg = json.dumps(
            {
                "kv_connector": "OffloadingConnector",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": int(spec.kv_offload_bytes),
                },
                "kv_role": "kv_both",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        args.extend(["--kv-transfer-config", offload_cfg])
    return args


def build_arm_env(
    arm: str,
    spec: ConcurrencySpec,
    *,
    reference_mode: str = "image",
    stats_period_s: int = 60,
    libtpu_pin: str = "0.0.50",
) -> dict[str, str]:
    """Build per-arm environment variables matching the `candidate30` recipe."""
    if arm not in ("reference", "candidate"):
        raise ValueError(f"Unknown arm {arm!r}; expected 'reference' or 'candidate'.")
    if reference_mode not in ("image", "stock"):
        raise ValueError(
            f"Unknown reference_mode {reference_mode!r}; expected 'image' or 'stock'."
        )
    env: dict[str, str] = {
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "ENABLE_PAW": "true",
        "TPU_QWEN35_AUTO_DEFAULTS": "1",
        "LIBTPU_PIN": libtpu_pin,
        "KV_OFFLOAD": "true" if spec.kv_offload_enabled else "false",
    }
    if spec.kv_offload_enabled and spec.kv_offload_bytes > 0:
        env["KV_OFFLOAD_CPU_BYTES"] = str(spec.kv_offload_bytes)
    if arm == "reference" and reference_mode == "image":
        env["VLLM_TPU_QWEN35_MEGAKERNEL"] = "1"
        env["VLLM_TPU_QWEN35_MK_MODE"] = "ep"
        env["VLLM_TPU_QWEN35_MK_STATS_PERIOD_S"] = str(int(stats_period_s))
    return env


def build_arm_server_command(
    arm: str,
    spec: ConcurrencySpec,
    *,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    reference_mode: str = "image",
    reference_image: str | None = None,
    plugin_module: str | None = None,
    profile: str = DEFAULT_PROFILE,
    mode: str = "ep",
    stats_period_s: int = 60,
    rejection_sample_mode: str = "synthetic",
    synthetic_acceptance_length: float = DEFAULT_SYNTHETIC_ACCEPTANCE_LEN,
    hf_cache_dir: str | None = None,
) -> list[str]:
    """Construct server command list for Arm A (`reference`) or Arm B (`candidate`).

    - Arm A (`reference`, `reference_mode='image'`): runs `vllm serve` inside
      `reference_image` (or `MK_ACCEPTANCE_REFERENCE_IMAGE`) with `candidate30`'s
      exact flags (`--block-size 256`, `MTP=3`, prefix caching, async scheduling,
      KV offload).
    - Arm A (`reference`, `reference_mode='stock'`): runs stock `vllm serve` on the
      current environment with `megakernel.enabled = False` and
      `--mamba-ssm-cache-dtype bfloat16`.
    - Arm B (`candidate`): runs `vllm serve` with `--mamba-ssm-cache-dtype bfloat16`
      (alongside `main`'s forced `fp32` conv state) and `--additional-config`
      pointing at the Qwen3.5 template plugin (`options.profile`, `options.mode`).
    - Pass `rejection_sample_mode='real'` for the separate GSM8K accuracy pass
      (`MTP=3` enabled, synthetic acceptance OFF; pair with greedy `temperature=0`).
    """
    if arm not in ("reference", "candidate"):
        raise ValueError(f"Unknown arm {arm!r}; expected 'reference' or 'candidate'.")
    if reference_mode not in ("image", "stock"):
        raise ValueError(
            f"Unknown reference_mode {reference_mode!r}; expected 'image' or 'stock'."
        )

    serve_args = _common_vllm_serve_args(
        spec,
        model=model,
        host=host,
        port=port,
        rejection_sample_mode=rejection_sample_mode,
        synthetic_acceptance_length=synthetic_acceptance_length,
    )

    if arm == "reference":
        if reference_mode == "image":
            img = reference_image or os.environ.get(REF_IMAGE_ENV, "").strip()
            if not img:
                raise ValueError(
                    "reference_mode='image' requires --reference-image or "
                    f"{REF_IMAGE_ENV} to be set (no hardcoded registry URI in src/)."
                )
            env_map = build_arm_env(
                "reference",
                spec,
                reference_mode="image",
                stats_period_s=stats_period_s,
            )
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--privileged",
                "--network",
                "host",
                "--ipc",
                "host",
                "--shm-size=64g",
            ]
            if hf_cache_dir:
                docker_cmd.extend(["-v", f"{hf_cache_dir}:{hf_cache_dir}"])
                env_map["HF_HOME"] = hf_cache_dir
            for k, v in sorted(env_map.items()):
                docker_cmd.extend(["-e", f"{k}={v}"])
            docker_cmd.append(img)
            docker_cmd.extend(serve_args)
            return docker_cmd

        # reference_mode == "stock"
        add_cfg = json.dumps(
            {"megakernel": {"enabled": False}},
            separators=(",", ":"),
            sort_keys=True,
        )
        return [
            *serve_args,
            "--mamba-ssm-cache-dtype",
            "bfloat16",
            "--additional-config",
            add_cfg,
        ]

    # Arm B ("candidate"): Qwen3.5 template on merged mk-dev
    resolved_plugin = plugin_module or default_plugin_module()
    add_cfg = json.dumps(
        {
            "megakernel": {
                "options": {
                    "mode": mode,
                    "profile": profile,
                },
                "plugin": resolved_plugin,
                "stats_period_s": int(stats_period_s),
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return [
        *serve_args,
        "--mamba-ssm-cache-dtype",
        "bfloat16",
        "--additional-config",
        add_cfg,
    ]


_STEP_MIX_RE = re.compile(
    r"megakernel step mix over \d+(?:\.\d+)?s:\s*(\d+)/(\d+)\s+MK\s+"
    r"\([0-9.]+%\)\s*(\{.*\})"
)
_MK_STATS_JSON_RE = re.compile(r"\[MK-STATS\].*?mix=(\{.*\})")
_DECLINE_RE = re.compile(r"megakernel declined \(([^)]+)\):\s*(.*)")


def classify_c16_packing_mode(
    bucket_counts: Mapping[str, int],
    *,
    bunched_threshold_pct: float = DEFAULT_C16_BUNCHED_THRESHOLD_PCT,
) -> tuple[float, str]:
    """Classify `c16` trajectory-packing phase from `[MK-STATS]` verify buckets.

    Returns `(mid_verify_share_pct, mode)` where `mid_verify_share_pct` is the
    combined step share of `mk_t4_b32`, `mk_t4_b40`, and `mk_t4_b48`, and `mode`
    is `'bunched'` (`>= 6.0%`) or `'staggered'` (`< 6.0%`).
    """
    total = sum(int(v) for v in bucket_counts.values())
    if total <= 0:
        return 0.0, "unknown"
    mid_steps = (
        int(bucket_counts.get("mk_t4_b32", 0))
        + int(bucket_counts.get("mk_t4_b40", 0))
        + int(bucket_counts.get("mk_t4_b48", 0))
    )
    mid_pct = (100.0 * mid_steps) / float(total)
    mode = "bunched" if mid_pct >= bunched_threshold_pct else "staggered"
    return round(mid_pct, 4), mode


def parse_mk_stats_log(
    log_text: str,
    *,
    bunched_threshold_pct: float = DEFAULT_C16_BUNCHED_THRESHOLD_PCT,
) -> StepMixSummary:
    """Aggregate all periodic step-mix windows and decline messages in a server log.

    `StepMixTracker` resets its window counters after every periodic emission, so
    every window dictionary in the sweep log is summed to obtain exact full-sweep
    bucket totals.
    """
    bucket_counts: dict[str, int] = {}
    decline_codes: list[str] = []

    for line in log_text.splitlines():
        m_decl = _DECLINE_RE.search(line)
        if m_decl:
            code = m_decl.group(1).strip()
            if code and code not in decline_codes:
                decline_codes.append(code)

        m_mix = _STEP_MIX_RE.search(line)
        raw_dict_str: str | None = None
        if m_mix:
            raw_dict_str = m_mix.group(3)
        else:
            m_json = _MK_STATS_JSON_RE.search(line)
            if m_json:
                raw_dict_str = m_json.group(1)

        if raw_dict_str is not None:
            try:
                parsed = ast.literal_eval(raw_dict_str)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(k, str) and isinstance(v, (int, float)):
                        bucket_counts[k] = bucket_counts.get(k, 0) + int(v)

    total_steps = sum(bucket_counts.values())
    mk_steps = sum(v for k, v in bucket_counts.items() if k.startswith("mk_"))
    mk_pct = (100.0 * mk_steps / float(total_steps)) if total_steps > 0 else 0.0
    shares = (
        {
            k: round(100.0 * v / float(total_steps), 4)
            for k, v in sorted(bucket_counts.items())
        }
        if total_steps > 0
        else {}
    )
    mid_pct, packing_mode = classify_c16_packing_mode(
        bucket_counts, bunched_threshold_pct=bunched_threshold_pct
    )
    return StepMixSummary(
        mk_steps=mk_steps,
        total_steps=total_steps,
        mk_pct=round(mk_pct, 4),
        bucket_counts=dict(sorted(bucket_counts.items())),
        bucket_shares_pct=shares,
        decline_codes=tuple(decline_codes),
        mid_verify_share_pct=mid_pct,
        packing_mode=packing_mode,
    )


_PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
)


def parse_prometheus_metrics(metrics_text: str) -> dict[str, float]:
    """Sum scalar/counter lines by metric family name from `/metrics` text."""
    totals: dict[str, float] = {}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        val = float(m.group(2))
        if math.isfinite(val):
            totals[name] = totals.get(name, 0.0) + val
    return totals


def _metric_delta(
    before: Mapping[str, float],
    after: Mapping[str, float],
    candidates: Sequence[str],
) -> float | None:
    for key in candidates:
        if key in after:
            return float(after[key]) - float(before.get(key, 0.0))
    return None


def compute_prometheus_deltas(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float | None]:
    """Compute speculative acceptance length and prefix-cache hit rate over a sweep."""
    accepted = _metric_delta(
        before,
        after,
        (
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_accepted_tokens",
        ),
    )
    drafts = _metric_delta(
        before,
        after,
        (
            "vllm:spec_decode_num_drafts_total",
            "vllm:spec_decode_num_drafts",
        ),
    )
    mean_acc_len: float | None = None
    if accepted is not None and drafts is not None and drafts > 0.0:
        mean_acc_len = round(1.0 + (accepted / drafts), 4)

    hits = _metric_delta(
        before,
        after,
        ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"),
    )
    queries = _metric_delta(
        before,
        after,
        ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"),
    )
    hit_rate: float | None = None
    if hits is not None and queries is not None and queries > 0.0:
        hit_rate = round(hits / queries, 4)

    return {
        "spec_mean_acceptance_length": mean_acc_len,
        "prefix_cache_hit_rate": hit_rate,
    }


def validate_sweep_record(
    record: SweepRecord,
    *,
    target_duration_s: float = DEFAULT_SWEEP_DURATION_S,
    expected_acceptance_length: float = DEFAULT_SYNTHETIC_ACCEPTANCE_LEN,
    acceptance_length_tol: float = DEFAULT_ACCEPTANCE_LEN_TOL,
) -> SweepRecord:
    """Enforce mandatory per-sweep validity invariants before paired statistics."""
    reasons: list[str] = []

    if not record.server_healthy_after:
        reasons.append("server_unhealthy_after_sweep")
    if record.records_error_dropped != 0:
        reasons.append(f"records_error_dropped={record.records_error_dropped}")

    if record.expect_megakernel:
        if record.step_mix.decline_codes:
            reasons.append(
                f"megakernel_declined:{','.join(record.step_mix.decline_codes)}"
            )
        if record.step_mix.mk_steps <= 0:
            reasons.append("mk_steps_zero")
    elif record.step_mix.mk_steps > 0:
        reasons.append(f"unexpected_mk_steps_on_stock_arm={record.step_mix.mk_steps}")

    if record.spec_mean_acceptance_length is None:
        reasons.append("missing_spec_acceptance_length")
    else:
        lo = expected_acceptance_length - acceptance_length_tol
        hi = expected_acceptance_length + acceptance_length_tol
        if not (lo <= record.spec_mean_acceptance_length <= hi):
            reasons.append(
                f"spec_acceptance_out_of_band={record.spec_mean_acceptance_length:.4f}"
                f" not_in_[{lo:.4f},{hi:.4f}]"
            )

    min_dur = 0.95 * float(target_duration_s)
    max_dur = 1.15 * float(target_duration_s)
    if not (min_dur <= record.duration_s <= max_dur):
        reasons.append(
            f"duration_out_of_window={record.duration_s:.1f}s"
            f" expected_[{min_dur:.1f},{max_dur:.1f}]s"
        )

    return dataclasses.replace(
        record,
        valid=(len(reasons) == 0),
        invalid_reasons=tuple(reasons),
    )


def paired_student_t(
    ref_values: Sequence[float],
    cand_values: Sequence[float],
    *,
    higher_is_better: bool,
    noise_floor_pct: float,
) -> dict[str, Any]:
    """Paired two-sided 95% Student-t evaluation.

    `noise_floor_pct` is strictly a measurement-precision target
    (`INCONCLUSIVE_HIGH_VARIANCE` when `0` is in the 95% CI and half-width exceeds
    `noise_floor_pct`), never an allowed slowdown budget. Any 95% CI excluding `0`
    in the unfavorable direction returns `SIGNIFICANT_REGRESSION`.
    """
    if len(ref_values) != len(cand_values) or len(ref_values) == 0:
        raise ValueError(
            "Expected equal non-empty paired sequences, got "
            f"{len(ref_values)} and {len(cand_values)}."
        )
    n = len(ref_values)
    deltas_pct = [
        ((c - r) / r) * 100.0 for r, c in zip(ref_values, cand_values, strict=True)
    ]
    mean_ref = statistics.fmean(ref_values)
    mean_cand = statistics.fmean(cand_values)
    mean_delta = statistics.fmean(deltas_pct)
    if n >= 2:
        sd = statistics.stdev(deltas_pct)
        se = sd / math.sqrt(n)
        hw = t_crit_95(n - 1) * se
    else:
        sd = 0.0
        se = float("inf")
        hw = float("inf")

    ci_lo = mean_delta - hw
    ci_hi = mean_delta + hw

    if higher_is_better:
        sig_reg = ci_hi < 0.0
        sig_imp = ci_lo > 0.0
    else:
        sig_reg = ci_lo > 0.0
        sig_imp = ci_hi < 0.0

    if n < 2:
        status = "INVALID_SINGLE_RUN"
    elif sig_reg:
        status = "SIGNIFICANT_REGRESSION"
    elif sig_imp:
        status = "SIGNIFICANT_IMPROVEMENT"
    elif hw > noise_floor_pct:
        status = "INCONCLUSIVE_HIGH_VARIANCE"
    else:
        status = "WITHIN_NOISE_EQUIVALENT"

    return {
        "pairs": n,
        "mean_reference": round(mean_ref, 4),
        "mean_candidate": round(mean_cand, 4),
        "mean_delta_pct": round(mean_delta, 3),
        "sd_delta_pct": round(sd, 3),
        "se_delta_pct": round(se, 3) if math.isfinite(se) else None,
        "ci95_lo_pct": round(ci_lo, 3) if math.isfinite(ci_lo) else None,
        "ci95_hi_pct": round(ci_hi, 3) if math.isfinite(ci_hi) else None,
        "ci95_half_width_pct": round(hw, 3) if math.isfinite(hw) else None,
        "noise_floor_pct": noise_floor_pct,
        "higher_is_better": higher_is_better,
        "status": status,
        "pairwise_deltas_pct": [round(x, 3) for x in deltas_pct],
    }


def _evaluate_pair_list(
    pairs: Sequence[tuple[SweepRecord, SweepRecord]],
) -> dict[str, Any]:
    return {
        "output_tok_per_s": paired_student_t(
            [a.output_tok_per_s for a, _ in pairs],
            [b.output_tok_per_s for _, b in pairs],
            higher_is_better=True,
            noise_floor_pct=1.5,
        ),
        "itl_p50_ms": paired_student_t(
            [a.itl_p50_ms for a, _ in pairs],
            [b.itl_p50_ms for _, b in pairs],
            higher_is_better=False,
            noise_floor_pct=2.0,
        ),
        "itl_p90_ms": paired_student_t(
            [a.itl_p90_ms for a, _ in pairs],
            [b.itl_p90_ms for _, b in pairs],
            higher_is_better=False,
            noise_floor_pct=2.0,
        ),
        "itl_p99_ms": paired_student_t(
            [a.itl_p99_ms for a, _ in pairs],
            [b.itl_p99_ms for _, b in pairs],
            higher_is_better=False,
            noise_floor_pct=3.0,
        ),
    }


def evaluate_concurrency(
    concurrency: str,
    records: Sequence[SweepRecord],
    *,
    max_arm_acceptance_spread: float = 0.02,
) -> dict[str, Any]:
    """Evaluate pairs at one concurrency, with `[MK-STATS]` mode splits at `c16`."""
    invalid = [r for r in records if not r.valid]
    by_pair: dict[int, dict[str, SweepRecord]] = {}
    for r in records:
        by_pair.setdefault(r.pair_index, {})[r.arm] = r

    ordered_pairs: list[tuple[SweepRecord, SweepRecord]] = []
    for p_idx in sorted(by_pair):
        slot = by_pair[p_idx]
        if "reference" in slot and "candidate" in slot:
            ordered_pairs.append((slot["reference"], slot["candidate"]))

    numerics_set = sorted({r.numerics for r in records})
    numerics_mixed = len(numerics_set) > 1

    if invalid or len(ordered_pairs) < 2:
        return {
            "concurrency": concurrency,
            "verdict": "INVALID_RUN",
            "numerics": "/".join(numerics_set) if numerics_set else "unknown",
            "numerics_mixed": numerics_mixed,
            "invalid_sweeps": [
                {"sweep_id": r.sweep_id, "reasons": list(r.invalid_reasons)}
                for r in invalid
            ],
            "pairs_count": len(ordered_pairs),
        }

    ref_acc = [
        a.spec_mean_acceptance_length
        for a, _ in ordered_pairs
        if a.spec_mean_acceptance_length is not None
    ]
    cand_acc = [
        b.spec_mean_acceptance_length
        for _, b in ordered_pairs
        if b.spec_mean_acceptance_length is not None
    ]
    arm_acc_delta = (
        abs(statistics.fmean(cand_acc) - statistics.fmean(ref_acc))
        if ref_acc and cand_acc
        else 0.0
    )
    if arm_acc_delta > max_arm_acceptance_spread:
        return {
            "concurrency": concurrency,
            "verdict": "INVALID_RUN",
            "numerics": "/".join(numerics_set),
            "numerics_mixed": numerics_mixed,
            "invalid_sweeps": [
                {
                    "sweep_id": "arm_acceptance_spread",
                    "reasons": [
                        f"arm_acceptance_delta={arm_acc_delta:.4f}>{max_arm_acceptance_spread:.4f}"
                    ],
                }
            ],
            "pairs_count": len(ordered_pairs),
        }

    raw_metrics = _evaluate_pair_list(ordered_pairs)
    stratification: dict[str, Any] | None = None
    primary_metrics = raw_metrics

    if concurrency == "c16":
        matched_pairs = [
            (a, b)
            for a, b in ordered_pairs
            if a.step_mix.packing_mode == b.step_mix.packing_mode
            and a.step_mix.packing_mode in ("staggered", "bunched")
        ]
        by_mode: dict[str, Any] = {}
        for mode_name in ("staggered", "bunched"):
            refs_m = [
                a for a, _ in ordered_pairs if a.step_mix.packing_mode == mode_name
            ]
            cands_m = [
                b for _, b in ordered_pairs if b.step_mix.packing_mode == mode_name
            ]
            if refs_m and cands_m:
                ref_tps = statistics.fmean(x.output_tok_per_s for x in refs_m)
                cand_tps = statistics.fmean(x.output_tok_per_s for x in cands_m)
                ref_p50 = statistics.fmean(x.itl_p50_ms for x in refs_m)
                cand_p50 = statistics.fmean(x.itl_p50_ms for x in cands_m)
                ref_p90 = statistics.fmean(x.itl_p90_ms for x in refs_m)
                cand_p90 = statistics.fmean(x.itl_p90_ms for x in cands_m)
                by_mode[mode_name] = {
                    "reference_sweeps": len(refs_m),
                    "candidate_sweeps": len(cands_m),
                    "output_tok_per_s_delta_pct": round(
                        100.0 * (cand_tps - ref_tps) / ref_tps, 3
                    ),
                    "itl_p50_ms_delta_pct": round(
                        100.0 * (cand_p50 - ref_p50) / ref_p50, 3
                    ),
                    "itl_p90_ms_delta_pct": round(
                        100.0 * (cand_p90 - ref_p90) / ref_p90, 3
                    ),
                }
        matched_eval = (
            _evaluate_pair_list(matched_pairs) if len(matched_pairs) >= 2 else None
        )
        stratification = {
            "sweep_modes": [
                {
                    "pair_index": a.pair_index,
                    "reference_mode": a.step_mix.packing_mode,
                    "reference_mid_verify_pct": a.step_mix.mid_verify_share_pct,
                    "candidate_mode": b.step_mix.packing_mode,
                    "candidate_mid_verify_pct": b.step_mix.mid_verify_share_pct,
                }
                for a, b in ordered_pairs
            ],
            "mode_matched_pairs": len(matched_pairs),
            "mode_matched_metrics": matched_eval,
            "by_mode": by_mode,
        }
        if matched_eval is not None:
            primary_metrics = matched_eval

    # Gate on primary metrics (throughput, p50 ITL, p90 ITL) AND fail if raw shows
    # any statistically significant regression.
    gated_keys = ("output_tok_per_s", "itl_p50_ms", "itl_p90_ms")
    any_regression = any(
        primary_metrics[k]["status"] == "SIGNIFICANT_REGRESSION"
        or raw_metrics[k]["status"] == "SIGNIFICANT_REGRESSION"
        for k in gated_keys
    )
    any_inconclusive = any(
        primary_metrics[k]["status"] == "INCONCLUSIVE_HIGH_VARIANCE"
        for k in ("output_tok_per_s", "itl_p50_ms")
    )

    if any_regression:
        verdict = "FAIL"
    elif any_inconclusive:
        verdict = "INCONCLUSIVE_HIGH_VARIANCE"
    else:
        verdict = "PASS"

    return {
        "concurrency": concurrency,
        "verdict": verdict,
        "numerics": "/".join(numerics_set),
        "numerics_mixed": numerics_mixed,
        "pairs_count": len(ordered_pairs),
        "arm_acceptance_spread": round(arm_acc_delta, 4),
        "reference_mk_pct_mean": round(
            statistics.fmean(a.step_mix.mk_pct for a, _ in ordered_pairs), 3
        ),
        "candidate_mk_pct_mean": round(
            statistics.fmean(b.step_mix.mk_pct for _, b in ordered_pairs), 3
        ),
        "raw_metrics": raw_metrics,
        "primary_metrics": primary_metrics,
        "c16_stratification": stratification,
    }


def build_gsm8k_server_command(
    arm: str,
    spec: ConcurrencySpec | None = None,
    *,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    reference_mode: str = "image",
    reference_image: str | None = None,
    plugin_module: str | None = None,
    profile: str = DEFAULT_PROFILE,
    mode: str = "ep",
    stats_period_s: int = 60,
    hf_cache_dir: str | None = None,
) -> list[str]:
    """Build server command for the separate GSM8K accuracy pass (`MTP=3`, real
    rejection sampling, synthetic acceptance OFF) on Arm A or Arm B.
    """
    eff_spec = spec if spec is not None else CONCURRENCY_SPECS["c32"]
    return build_arm_server_command(
        arm,
        eff_spec,
        model=model,
        host=host,
        port=port,
        reference_mode=reference_mode,
        reference_image=reference_image,
        plugin_module=plugin_module,
        profile=profile,
        mode=mode,
        stats_period_s=stats_period_s,
        rejection_sample_mode="real",
        hf_cache_dir=hf_cache_dir,
    )


def evaluate_gsm8k_gate(
    score_candidate: float,
    score_reference: float = 0.9242,
    *,
    n_questions: int = 1319,
    ref_only_correct: int | None = None,
    cand_only_correct: int | None = None,
    rejection_sample_mode: str = "real",
    temperature: float = 0.0,
    reference_acceptance_lengths: Sequence[float] | None = None,
    candidate_acceptance_lengths: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate GSM8K accuracy parity (`2 * SE` or paired McNemar) and acceptance
    length spread from a separate real-rejection-sampling greedy pass on both arms.
    """
    if rejection_sample_mode != "real":
        raise ValueError(
            "GSM8K accuracy must be measured with rejection_sample_mode='real' "
            "(MTP on, synthetic acceptance off); synthetic acceptance does not "
            "produce the model's tokens."
        )
    if float(temperature) != 0.0:
        raise ValueError(
            f"GSM8K accuracy pass requires greedy decoding (temperature=0.0); "
            f"got temperature={temperature!r}."
        )
    se = math.sqrt(score_reference * (1.0 - score_reference) / max(1, n_questions))
    band_2se = 2.0 * se
    delta = score_candidate - score_reference
    mcnemar_z: float | None = None
    if ref_only_correct is not None and cand_only_correct is not None:
        disc = ref_only_correct + cand_only_correct
        mcnemar_z = (
            (ref_only_correct - cand_only_correct) / math.sqrt(disc)
            if disc > 0
            else 0.0
        )
        passed = mcnemar_z <= 1.96
    else:
        passed = delta >= -band_2se

    ref_acc = [float(x) for x in (reference_acceptance_lengths or ())]
    cand_acc = [float(x) for x in (candidate_acceptance_lengths or ())]
    all_acc = ref_acc + cand_acc
    ref_acc_mean = round(statistics.fmean(ref_acc), 4) if ref_acc else None
    cand_acc_mean = round(statistics.fmean(cand_acc), 4) if cand_acc else None
    arm_acc_delta = (
        round(abs(cand_acc_mean - ref_acc_mean), 4)
        if ref_acc_mean is not None and cand_acc_mean is not None
        else None
    )
    acc_spread = (
        round(max(all_acc) - min(all_acc), 4)
        if len(all_acc) >= 2
        else (0.0 if len(all_acc) == 1 else None)
    )
    return {
        "n_questions": n_questions,
        "rejection_sample_mode": rejection_sample_mode,
        "temperature": 0.0,
        "score_reference": round(score_reference, 4),
        "score_candidate": round(score_candidate, 4),
        "delta": round(delta, 4),
        "band_2se": round(band_2se, 4),
        "mcnemar_z": round(mcnemar_z, 3) if mcnemar_z is not None else None,
        "reference_mean_acceptance_length": ref_acc_mean,
        "candidate_mean_acceptance_length": cand_acc_mean,
        "arm_acceptance_delta": arm_acc_delta,
        "acceptance_length_spread": acc_spread,
        "passed": bool(passed),
    }


def evaluate_acceptance_suite(
    records: Sequence[SweepRecord],
    *,
    required_concurrencies: Sequence[str] = ("c8", "c16", "c32"),
    gsm8k_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the full multi-concurrency acceptance suite."""
    by_conc: dict[str, list[SweepRecord]] = {}
    for r in records:
        by_conc.setdefault(r.concurrency, []).append(r)

    conc_reports: dict[str, Any] = {}
    for c_name in required_concurrencies:
        if c_name not in by_conc:
            conc_reports[c_name] = {
                "concurrency": c_name,
                "verdict": "MISSING",
                "pairs_count": 0,
            }
        else:
            conc_reports[c_name] = evaluate_concurrency(c_name, by_conc[c_name])

    verdicts = [conc_reports[c]["verdict"] for c in required_concurrencies]
    gsm8k_failed = gsm8k_result is not None and not bool(
        gsm8k_result.get("passed", False)
    )
    if any(v == "INVALID_RUN" for v in verdicts):
        overall = "INVALID_RUN"
    elif any(v == "FAIL" for v in verdicts) or gsm8k_failed:
        overall = "FAIL"
    elif any(v in ("MISSING", "INCONCLUSIVE_HIGH_VARIANCE") for v in verdicts):
        overall = "INCONCLUSIVE"
    else:
        overall = "PASS"

    return {
        "overall_verdict": overall,
        "required_concurrencies": list(required_concurrencies),
        "concurrencies": conc_reports,
        "gsm8k": dict(gsm8k_result) if gsm8k_result is not None else None,
    }


def format_acceptance_markdown(suite: Mapping[str, Any]) -> str:
    """Render a concise Markdown acceptance table separating `numerics` classes."""
    hdr = (
        "| Concurrency | Numerics | Pairs "
        "| Output Tok/s ($\\Delta \\pm \\text{CI}_{95}$) "
        "| `p50` ITL ms ($\\Delta \\pm \\text{CI}_{95}$) "
        "| `p90` ITL ms ($\\Delta \\pm \\text{CI}_{95}$) "
        "| `p99` ITL ms ($\\Delta \\pm \\text{CI}_{95}$) "
        "| MK % (`A` / `B`) | Verdict |"
    )
    lines = [
        "# Qwen3.5 Serving Acceptance Report (`Arm A` vs `Arm B`)",
        "",
        f"- **Overall Verdict:** **`{suite['overall_verdict']}`**",
        "",
        hdr,
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c_name in suite.get("required_concurrencies", ()):
        rep = suite["concurrencies"].get(c_name, {})
        verdict = rep.get("verdict", "MISSING")
        if "primary_metrics" not in rep:
            lines.append(
                f"| `{c_name}` | `{rep.get('numerics', 'n/a')}` | "
                f"`{rep.get('pairs_count', 0)}` | — | — | — | — | — | **`{verdict}`** |"
            )
            continue
        pm = rep["primary_metrics"]

        def _fmt(m: Mapping[str, Any]) -> str:
            hw = m.get("ci95_half_width_pct")
            hw_s = f"{hw:.2f}%" if hw is not None else "inf"
            return (
                f"`{m['mean_reference']:.2f} -> {m['mean_candidate']:.2f}` "
                f"(`{m['mean_delta_pct']:+.2f}% ± {hw_s}`)"
            )

        mk_pair_s = (
            f"`{rep['reference_mk_pct_mean']:.1f}% / "
            f"{rep['candidate_mk_pct_mean']:.1f}%`"
        )
        lines.append(
            f"| `{c_name}` | `{rep['numerics']}` | `{rep['pairs_count']}` | "
            f"{_fmt(pm['output_tok_per_s'])} | {_fmt(pm['itl_p50_ms'])} | "
            f"{_fmt(pm['itl_p90_ms'])} | {_fmt(pm['itl_p99_ms'])} | "
            f"{mk_pair_s} | **`{verdict}`** |"
        )
        strat = rep.get("c16_stratification")
        if strat and strat.get("by_mode"):
            for mode_name, m_stats in strat["by_mode"].items():
                cnt_s = (
                    f"`{m_stats['reference_sweeps']}A/{m_stats['candidate_sweeps']}B`"
                )
                lines.append(
                    f"| `{c_name}` (`{mode_name}`) | `{rep['numerics']}` | "
                    f"{cnt_s} | "
                    f"`{m_stats['output_tok_per_s_delta_pct']:+.2f}%` | "
                    f"`{m_stats['itl_p50_ms_delta_pct']:+.2f}%` | "
                    f"`{m_stats['itl_p90_ms_delta_pct']:+.2f}%` | — | — | stratified |"
                )

    gsm8k = suite.get("gsm8k")
    if gsm8k:
        mcnemar_part = (
            f", McNemar $z = {gsm8k['mcnemar_z']:.3f}$"
            if gsm8k.get("mcnemar_z") is not None
            else ""
        )
        rej_mode = gsm8k.get("rejection_sample_mode", "real")
        temp_val = gsm8k.get("temperature", 0.0)
        lines.extend(
            [
                "",
                (
                    "## Accuracy Gate (GSM8K — Separate Real Rejection "
                    "Sampling Pass, Greedy)"
                ),
                f"- Mode: `rejection_sample_mode={rej_mode}`, "
                f"`temperature={temp_val:.1f}`",
                f"- Reference: `{gsm8k['score_reference']:.4f}`, "
                f"Candidate: `{gsm8k['score_candidate']:.4f}`, "
                f"$\\Delta = {gsm8k['delta']:+.4f}$ "
                f"(`2*SE = ±{gsm8k['band_2se']:.4f}`{mcnemar_part}), "
                f"Passed: **`{gsm8k['passed']}`**",
            ]
        )
        if gsm8k.get("acceptance_length_spread") is not None:
            lines.append(
                f"- Real MTP acceptance length (`A` / `B`): "
                f"`{gsm8k.get('reference_mean_acceptance_length')} / "
                f"{gsm8k.get('candidate_mean_acceptance_length')}` "
                f"(spread across runs: `{gsm8k['acceptance_length_spread']:.4f}`)"
            )
    return "\n".join(lines) + "\n"


def _http_get_text(url: str, timeout_s: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, ""


def _wait_for_server_ready(
    proc: subprocess.Popen[str],
    log_path: pathlib.Path,
    *,
    health_url: str,
    expect_megakernel: bool,
    startup_timeout_s: float,
) -> None:
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited during startup (code {proc.returncode}); see {log_path}"
            )
        if expect_megakernel and log_path.exists():
            txt = log_path.read_text(encoding="utf-8", errors="replace")
            m = _DECLINE_RE.search(txt)
            if m:
                raise RuntimeError(
                    f"Server declined megakernel ({m.group(1)}): {m.group(2)}"
                )
        code, _ = _http_get_text(health_url, timeout_s=3.0)
        if code == 200:
            return
        time.sleep(2.0)
    raise TimeoutError(
        f"Server did not become healthy within {startup_timeout_s:.0f}s ({health_url})."
    )


def _terminate_process_group(
    proc: subprocess.Popen[str], grace_s: float = 30.0
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=10.0)


def _extract_client_metrics(
    payload: Mapping[str, Any], elapsed_s: float
) -> dict[str, Any]:
    """Normalize client benchmark JSON output into throughput and ITL percentiles."""
    duration = float(
        payload.get("duration_s")
        or payload.get("benchmark_duration_s")
        or payload.get("duration")
        or elapsed_s
    )
    out_tps = float(
        payload.get("output_tok_per_s")
        or payload.get("output_throughput")
        or payload.get("output_tokens_per_second")
        or 0.0
    )
    p50 = float(
        payload.get("itl_p50_ms")
        or payload.get("p50_itl_ms")
        or payload.get("median_itl_ms")
        or 0.0
    )
    p90 = float(payload.get("itl_p90_ms") or payload.get("p90_itl_ms") or 0.0)
    p99 = float(payload.get("itl_p99_ms") or payload.get("p99_itl_ms") or 0.0)
    mean_itl = float(payload.get("itl_mean_ms") or payload.get("mean_itl_ms") or 0.0)
    dropped = int(
        payload.get("records_error_dropped")
        or payload.get("failed_requests")
        or payload.get("num_failed")
        or 0
    )
    return {
        "duration_s": duration,
        "output_tok_per_s": out_tps,
        "itl_p50_ms": p50,
        "itl_p90_ms": p90,
        "itl_p99_ms": p99,
        "itl_mean_ms": mean_itl,
        "records_error_dropped": dropped,
    }


def run_interleaved_suite(
    *,
    concurrencies: Sequence[str],
    pairs: int,
    out_dir: pathlib.Path,
    client_cmd_template: str,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    reference_mode: str = "image",
    reference_image: str | None = None,
    arm_a_cmd_override: str | None = None,
    arm_b_cmd_override: str | None = None,
    plugin_module: str | None = None,
    profile: str = DEFAULT_PROFILE,
    startup_timeout_s: float = 1800.0,
    duration_override_s: int | None = None,
) -> dict[str, Any]:
    """Run separate-process `A/B/A/B` sweeps on a single TPU allocation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[SweepRecord] = []
    health_url = f"http://{host}:{port}/health"
    metrics_url = f"http://{host}:{port}/metrics"

    for c_name in concurrencies:
        if c_name not in CONCURRENCY_SPECS:
            valid_concs = sorted(CONCURRENCY_SPECS)
            raise ValueError(
                f"Unknown concurrency {c_name!r}; expected one of {valid_concs}"
            )
        spec = CONCURRENCY_SPECS[c_name]
        if duration_override_s is not None:
            spec = dataclasses.replace(spec, duration_s=int(duration_override_s))

        for pair_idx in range(1, pairs + 1):
            for arm in ("reference", "candidate"):
                sweep_id = f"{c_name}_p{pair_idx}_{arm}"
                server_log = out_dir / f"{sweep_id}.server.log"
                client_json = out_dir / f"{sweep_id}.client.json"

                if arm == "reference" and arm_a_cmd_override:
                    srv_cmd = shlex.split(arm_a_cmd_override)
                elif arm == "candidate" and arm_b_cmd_override:
                    srv_cmd = shlex.split(arm_b_cmd_override)
                else:
                    srv_cmd = build_arm_server_command(
                        arm,
                        spec,
                        model=model,
                        host=host,
                        port=port,
                        reference_mode=reference_mode,
                        reference_image=reference_image,
                        plugin_module=plugin_module,
                        profile=profile,
                    )

                srv_env = os.environ.copy()
                srv_env.update(build_arm_env(arm, spec, reference_mode=reference_mode))
                expect_mk = not (arm == "reference" and reference_mode == "stock")
                numerics = (
                    "exact"
                    if (arm == "reference" and reference_mode == "stock")
                    else ("approx" if arm == "reference" else profile_numerics(profile))
                )
                arm_label = (
                    f"reference:{reference_mode}"
                    if arm == "reference"
                    else f"candidate:{profile}"
                )

                with server_log.open("w", encoding="utf-8") as log_fp:
                    proc = subprocess.Popen(
                        srv_cmd,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=srv_env,
                        start_new_session=True,
                    )
                    try:
                        _wait_for_server_ready(
                            proc,
                            server_log,
                            health_url=health_url,
                            expect_megakernel=expect_mk,
                            startup_timeout_s=startup_timeout_s,
                        )
                        _, prom_before_txt = _http_get_text(metrics_url, timeout_s=10.0)
                        prom_before = parse_prometheus_metrics(prom_before_txt)

                        cli_cmd_str = client_cmd_template.format(
                            host=host,
                            port=port,
                            concurrency=spec.concurrency,
                            duration_s=spec.duration_s,
                            output_json=str(client_json),
                            sweep_id=sweep_id,
                            model=model,
                        )
                        t0 = time.monotonic()
                        subprocess.run(
                            shlex.split(cli_cmd_str),
                            check=True,
                        )
                        elapsed = time.monotonic() - t0

                        h_code, _ = _http_get_text(health_url, timeout_s=5.0)
                        _, prom_after_txt = _http_get_text(metrics_url, timeout_s=10.0)
                        prom_after = parse_prometheus_metrics(prom_after_txt)
                    finally:
                        _terminate_process_group(proc)

                log_text = server_log.read_text(encoding="utf-8", errors="replace")
                step_mix = parse_mk_stats_log(log_text)
                prom_deltas = compute_prometheus_deltas(prom_before, prom_after)
                cli_payload = json.loads(client_json.read_text(encoding="utf-8"))
                cli_metrics = _extract_client_metrics(cli_payload, elapsed)

                raw_rec = SweepRecord(
                    sweep_id=sweep_id,
                    concurrency=c_name,
                    pair_index=pair_idx,
                    arm=arm,
                    arm_label=arm_label,
                    numerics=numerics,
                    expect_megakernel=expect_mk,
                    duration_s=cli_metrics["duration_s"],
                    output_tok_per_s=cli_metrics["output_tok_per_s"],
                    itl_p50_ms=cli_metrics["itl_p50_ms"],
                    itl_p90_ms=cli_metrics["itl_p90_ms"],
                    itl_p99_ms=cli_metrics["itl_p99_ms"],
                    itl_mean_ms=cli_metrics["itl_mean_ms"],
                    records_error_dropped=cli_metrics["records_error_dropped"],
                    server_healthy_after=(h_code == 200),
                    spec_mean_acceptance_length=prom_deltas[
                        "spec_mean_acceptance_length"
                    ],
                    prefix_cache_hit_rate=prom_deltas["prefix_cache_hit_rate"],
                    step_mix=step_mix,
                )
                validated = validate_sweep_record(
                    raw_rec, target_duration_s=float(spec.duration_s)
                )
                records.append(validated)

    suite = evaluate_acceptance_suite(
        records, required_concurrencies=tuple(concurrencies)
    )
    summary_payload = {
        "suite": suite,
        "sweeps": [dataclasses.asdict(r) for r in records],
    }
    (out_dir / "acceptance_summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "acceptance_report.md").write_text(
        format_acceptance_markdown(suite),
        encoding="utf-8",
    )
    return summary_payload


def _record_from_dict(data: Mapping[str, Any]) -> SweepRecord:
    sm_raw = data.get("step_mix")
    if isinstance(sm_raw, Mapping):
        sm = StepMixSummary(
            mk_steps=int(sm_raw["mk_steps"]),
            total_steps=int(sm_raw["total_steps"]),
            mk_pct=float(sm_raw["mk_pct"]),
            bucket_counts={
                str(k): int(v) for k, v in dict(sm_raw["bucket_counts"]).items()
            },
            bucket_shares_pct={
                str(k): float(v) for k, v in dict(sm_raw["bucket_shares_pct"]).items()
            },
            decline_codes=tuple(str(x) for x in sm_raw.get("decline_codes", ())),
            mid_verify_share_pct=float(sm_raw["mid_verify_share_pct"]),
            packing_mode=str(sm_raw["packing_mode"]),
        )
    elif "server_log_path" in data:
        log_txt = pathlib.Path(str(data["server_log_path"])).read_text(
            encoding="utf-8", errors="replace"
        )
        sm = parse_mk_stats_log(log_txt)
    else:
        raise ValueError("Sweep dict must contain 'step_mix' or 'server_log_path'.")

    rec = SweepRecord(
        sweep_id=str(data["sweep_id"]),
        concurrency=str(data["concurrency"]),
        pair_index=int(data["pair_index"]),
        arm=str(data["arm"]),
        arm_label=str(data.get("arm_label", data["arm"])),
        numerics=str(data.get("numerics", "approx")),
        expect_megakernel=bool(data.get("expect_megakernel", True)),
        duration_s=float(data["duration_s"]),
        output_tok_per_s=float(data["output_tok_per_s"]),
        itl_p50_ms=float(data["itl_p50_ms"]),
        itl_p90_ms=float(data["itl_p90_ms"]),
        itl_p99_ms=float(data["itl_p99_ms"]),
        itl_mean_ms=float(data.get("itl_mean_ms", 0.0)),
        records_error_dropped=int(data.get("records_error_dropped", 0)),
        server_healthy_after=bool(data.get("server_healthy_after", True)),
        spec_mean_acceptance_length=(
            float(data["spec_mean_acceptance_length"])
            if data.get("spec_mean_acceptance_length") is not None
            else None
        ),
        prefix_cache_hit_rate=(
            float(data["prefix_cache_hit_rate"])
            if data.get("prefix_cache_hit_rate") is not None
            else None
        ),
        step_mix=sm,
    )
    target_dur = float(data.get("target_duration_s", DEFAULT_SWEEP_DURATION_S))
    return validate_sweep_record(rec, target_duration_s=target_dur)


def run_gsm8k_pass(
    *,
    out_dir: pathlib.Path,
    eval_cmd_template: str,
    concurrency_name: str = "c32",
    repeats: int = 1,
    model: str = DEFAULT_MODEL,
    host: str = "127.0.0.1",
    port: int = 8000,
    reference_mode: str = "image",
    reference_image: str | None = None,
    arm_a_cmd_override: str | None = None,
    arm_b_cmd_override: str | None = None,
    plugin_module: str | None = None,
    profile: str = DEFAULT_PROFILE,
    startup_timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run a separate GSM8K accuracy pass on both arms with REAL rejection sampling
    (`MTP=3` enabled, synthetic acceptance OFF, greedy `temperature=0.0`).
    """
    if concurrency_name not in CONCURRENCY_SPECS:
        valid_concs = sorted(CONCURRENCY_SPECS)
        raise ValueError(
            f"Unknown concurrency {concurrency_name!r}; expected one of {valid_concs}"
        )
    spec = CONCURRENCY_SPECS[concurrency_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    health_url = f"http://{host}:{port}/health"
    metrics_url = f"http://{host}:{port}/metrics"

    scores: dict[str, list[float]] = {"reference": [], "candidate": []}
    acc_lens: dict[str, list[float]] = {"reference": [], "candidate": []}
    n_questions = 1319
    ref_only_correct: int | None = None
    cand_only_correct: int | None = None
    arm_runs: list[dict[str, Any]] = []

    for rep_idx in range(1, max(1, int(repeats)) + 1):
        pair_correct: dict[str, list[bool]] = {}
        for arm in ("reference", "candidate"):
            run_id = f"gsm8k_r{rep_idx}_{arm}"
            server_log = out_dir / f"{run_id}.server.log"
            eval_json = out_dir / f"{run_id}.eval.json"

            if arm == "reference" and arm_a_cmd_override:
                srv_cmd = shlex.split(arm_a_cmd_override)
            elif arm == "candidate" and arm_b_cmd_override:
                srv_cmd = shlex.split(arm_b_cmd_override)
            else:
                srv_cmd = build_gsm8k_server_command(
                    arm,
                    spec,
                    model=model,
                    host=host,
                    port=port,
                    reference_mode=reference_mode,
                    reference_image=reference_image,
                    plugin_module=plugin_module,
                    profile=profile,
                )
            srv_str = " ".join(srv_cmd)
            if '"rejection_sample_method":"synthetic"' in srv_str.replace(" ", ""):
                raise ValueError(
                    f"GSM8K server command for {arm} has synthetic acceptance enabled; "
                    "real rejection sampling is required."
                )

            srv_env = os.environ.copy()
            srv_env.update(build_arm_env(arm, spec, reference_mode=reference_mode))
            expect_mk = not (arm == "reference" and reference_mode == "stock")

            with server_log.open("w", encoding="utf-8") as log_fp:
                proc = subprocess.Popen(
                    srv_cmd,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=srv_env,
                    start_new_session=True,
                )
                try:
                    _wait_for_server_ready(
                        proc,
                        server_log,
                        health_url=health_url,
                        expect_megakernel=expect_mk,
                        startup_timeout_s=startup_timeout_s,
                    )
                    _, prom_before_txt = _http_get_text(metrics_url, timeout_s=10.0)
                    prom_before = parse_prometheus_metrics(prom_before_txt)

                    eval_cmd_str = eval_cmd_template.format(
                        host=host,
                        port=port,
                        model=model,
                        output_json=str(eval_json),
                        run_id=run_id,
                        arm=arm,
                        temperature=0.0,
                    )
                    subprocess.run(shlex.split(eval_cmd_str), check=True)

                    _, prom_after_txt = _http_get_text(metrics_url, timeout_s=10.0)
                    prom_after = parse_prometheus_metrics(prom_after_txt)
                finally:
                    _terminate_process_group(proc)

            log_text = server_log.read_text(encoding="utf-8", errors="replace")
            step_mix = parse_mk_stats_log(log_text)
            if expect_mk and (step_mix.decline_codes or step_mix.mk_steps <= 0):
                raise RuntimeError(
                    f"GSM8K {run_id} failed megakernel execution check: "
                    f"mk_steps={step_mix.mk_steps}, declines={step_mix.decline_codes}"
                )
            prom_deltas = compute_prometheus_deltas(prom_before, prom_after)
            payload = json.loads(eval_json.read_text(encoding="utf-8"))
            score = float(
                payload.get("exact_match")
                if payload.get("exact_match") is not None
                else payload["accuracy"]
            )
            n_questions = int(payload.get("n_questions", n_questions))
            scores[arm].append(score)
            real_acc_len = prom_deltas.get("spec_mean_acceptance_length")
            if real_acc_len is not None:
                acc_lens[arm].append(float(real_acc_len))
            if isinstance(payload.get("per_question_correct"), list):
                pair_correct[arm] = [bool(x) for x in payload["per_question_correct"]]
            arm_runs.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "score": round(score, 4),
                    "spec_mean_acceptance_length": real_acc_len,
                    "mk_pct": step_mix.mk_pct,
                }
            )

        if "reference" in pair_correct and "candidate" in pair_correct:
            rc_list = pair_correct["reference"]
            cc_list = pair_correct["candidate"]
            if len(rc_list) == len(cc_list):
                ro = sum(1 for r_ok, c_ok in zip(rc_list, cc_list) if r_ok and not c_ok)
                co = sum(1 for r_ok, c_ok in zip(rc_list, cc_list) if c_ok and not r_ok)
                ref_only_correct = (ref_only_correct or 0) + ro
                cand_only_correct = (cand_only_correct or 0) + co

    gate = evaluate_gsm8k_gate(
        statistics.fmean(scores["candidate"]),
        statistics.fmean(scores["reference"]),
        n_questions=n_questions,
        ref_only_correct=ref_only_correct,
        cand_only_correct=cand_only_correct,
        rejection_sample_mode="real",
        temperature=0.0,
        reference_acceptance_lengths=acc_lens["reference"],
        candidate_acceptance_lengths=acc_lens["candidate"],
    )
    gate["runs"] = arm_runs
    (out_dir / "gsm8k_summary.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-allocation separate-process A/B/A/B serving acceptance "
            "benchmark for the Qwen3.5 megakernel template (plus separate "
            "real-rejection-sampling greedy GSM8K accuracy pass)."
        )
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_cmd = sub.add_parser(
        "print-commands",
        help="Print exact Arm A and Arm B server commands and environment variables.",
    )
    p_cmd.add_argument(
        "--pass-mode",
        choices=("perf", "gsm8k"),
        default="perf",
        help=(
            "'perf' prints synthetic-acceptance c8/c16/c32 sweep commands; "
            "'gsm8k' prints separate real-rejection-sampling (synthetic off, "
            "greedy temperature=0) commands for both arms."
        ),
    )
    p_cmd.add_argument(
        "--concurrencies",
        default="c8,c16,c32",
        help="Comma-separated concurrency specs (default: c8,c16,c32).",
    )
    p_cmd.add_argument("--model", default=DEFAULT_MODEL)
    p_cmd.add_argument(
        "--reference-mode",
        choices=("image", "stock"),
        default="image",
        help=(
            "'image' runs the reference container; "
            "'stock' runs stock vLLM on the current tree."
        ),
    )
    p_cmd.add_argument(
        "--reference-image",
        default=None,
        help=(
            "Container image URI for Arm A when --reference-mode=image "
            f"(or set {REF_IMAGE_ENV})."
        ),
    )
    p_cmd.add_argument("--plugin", default=None, help="Template plugin module path.")
    p_cmd.add_argument("--profile", default=DEFAULT_PROFILE)

    p_run = sub.add_parser(
        "run",
        help="Execute interleaved A/B/A/B sweeps with separate server processes.",
    )
    p_run.add_argument("--concurrencies", default="c8,c16,c32")
    p_run.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    p_run.add_argument("--out-dir", type=pathlib.Path, required=True)
    p_run.add_argument(
        "--client-cmd-template",
        required=True,
        help=(
            "Client command template supporting {host}, {port}, {concurrency}, "
            "{duration_s}, {output_json}, {sweep_id}, {model}."
        ),
    )
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--reference-mode", choices=("image", "stock"), default="image")
    p_run.add_argument("--reference-image", default=None)
    p_run.add_argument("--arm-a-cmd-override", default=None)
    p_run.add_argument("--arm-b-cmd-override", default=None)
    p_run.add_argument("--plugin", default=None)
    p_run.add_argument("--profile", default=DEFAULT_PROFILE)
    p_run.add_argument("--startup-timeout-s", type=float, default=1800.0)
    p_run.add_argument("--duration-s", type=int, default=None)

    p_gsm = sub.add_parser(
        "run-gsm8k",
        help=(
            "Execute separate GSM8K accuracy pass on both arms with REAL rejection "
            "sampling (MTP on, synthetic acceptance off, greedy temperature=0)."
        ),
    )
    p_gsm.add_argument("--out-dir", type=pathlib.Path, required=True)
    p_gsm.add_argument(
        "--eval-cmd-template",
        required=True,
        help=(
            "GSM8K client template supporting {host}, {port}, {model}, "
            "{output_json}, {run_id}, {arm}, {temperature}."
        ),
    )
    p_gsm.add_argument("--concurrency", default="c32")
    p_gsm.add_argument("--repeats", type=int, default=1)
    p_gsm.add_argument("--model", default=DEFAULT_MODEL)
    p_gsm.add_argument("--host", default="127.0.0.1")
    p_gsm.add_argument("--port", type=int, default=8000)
    p_gsm.add_argument("--reference-mode", choices=("image", "stock"), default="image")
    p_gsm.add_argument("--reference-image", default=None)
    p_gsm.add_argument("--arm-a-cmd-override", default=None)
    p_gsm.add_argument("--arm-b-cmd-override", default=None)
    p_gsm.add_argument("--plugin", default=None)
    p_gsm.add_argument("--profile", default=DEFAULT_PROFILE)
    p_gsm.add_argument("--startup-timeout-s", type=float, default=1800.0)

    p_ana = sub.add_parser(
        "analyze",
        help="Evaluate saved sweep JSON records and emit Markdown + JSON verdict.",
    )
    p_ana.add_argument("--sweeps-json", type=pathlib.Path, required=True)
    p_ana.add_argument("--gsm8k-json", type=pathlib.Path, default=None)
    p_ana.add_argument("--concurrencies", default="c8,c16,c32")
    p_ana.add_argument("--out-json", type=pathlib.Path, default=None)
    p_ana.add_argument("--out-md", type=pathlib.Path, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "print-commands":
        if args.pass_mode == "gsm8k":
            concs = [c.strip() for c in args.concurrencies.split(",") if c.strip()]
            c_name = concs[0] if len(concs) == 1 else "c32"
            spec = CONCURRENCY_SPECS[c_name]
            out_gsm = {
                "pass_mode": "gsm8k",
                "rejection_sample_mode": "real",
                "client_sampling_params": dict(GSM8K_GREEDY_SAMPLING_PARAMS),
                "concurrency_spec": c_name,
                "reference": {
                    "env": build_arm_env(
                        "reference", spec, reference_mode=args.reference_mode
                    ),
                    "cmd": build_gsm8k_server_command(
                        "reference",
                        spec,
                        model=args.model,
                        reference_mode=args.reference_mode,
                        reference_image=args.reference_image,
                        plugin_module=args.plugin,
                        profile=args.profile,
                    ),
                },
                "candidate": {
                    "env": build_arm_env(
                        "candidate", spec, reference_mode=args.reference_mode
                    ),
                    "cmd": build_gsm8k_server_command(
                        "candidate",
                        spec,
                        model=args.model,
                        reference_mode=args.reference_mode,
                        reference_image=args.reference_image,
                        plugin_module=args.plugin,
                        profile=args.profile,
                    ),
                },
            }
            sys.stdout.write(json.dumps(out_gsm, indent=2, sort_keys=True) + "\n")
            return 0

        concs = [c.strip() for c in args.concurrencies.split(",") if c.strip()]
        out: dict[str, Any] = {}
        for c_name in concs:
            spec = CONCURRENCY_SPECS[c_name]
            out[c_name] = {
                "reference": {
                    "env": build_arm_env(
                        "reference", spec, reference_mode=args.reference_mode
                    ),
                    "cmd": build_arm_server_command(
                        "reference",
                        spec,
                        model=args.model,
                        reference_mode=args.reference_mode,
                        reference_image=args.reference_image,
                        plugin_module=args.plugin,
                        profile=args.profile,
                    ),
                },
                "candidate": {
                    "env": build_arm_env(
                        "candidate", spec, reference_mode=args.reference_mode
                    ),
                    "cmd": build_arm_server_command(
                        "candidate",
                        spec,
                        model=args.model,
                        reference_mode=args.reference_mode,
                        reference_image=args.reference_image,
                        plugin_module=args.plugin,
                        profile=args.profile,
                    ),
                },
            }
        sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
        return 0

    if args.subcommand == "run":
        concs = [c.strip() for c in args.concurrencies.split(",") if c.strip()]
        res = run_interleaved_suite(
            concurrencies=concs,
            pairs=args.pairs,
            out_dir=args.out_dir,
            client_cmd_template=args.client_cmd_template,
            model=args.model,
            host=args.host,
            port=args.port,
            reference_mode=args.reference_mode,
            reference_image=args.reference_image,
            arm_a_cmd_override=args.arm_a_cmd_override,
            arm_b_cmd_override=args.arm_b_cmd_override,
            plugin_module=args.plugin,
            profile=args.profile,
            startup_timeout_s=args.startup_timeout_s,
            duration_override_s=args.duration_s,
        )
        sys.stdout.write(format_acceptance_markdown(res["suite"]))
        return 0 if res["suite"]["overall_verdict"] == "PASS" else 1

    if args.subcommand == "run-gsm8k":
        gsm = run_gsm8k_pass(
            out_dir=args.out_dir,
            eval_cmd_template=args.eval_cmd_template,
            concurrency_name=args.concurrency,
            repeats=args.repeats,
            model=args.model,
            host=args.host,
            port=args.port,
            reference_mode=args.reference_mode,
            reference_image=args.reference_image,
            arm_a_cmd_override=args.arm_a_cmd_override,
            arm_b_cmd_override=args.arm_b_cmd_override,
            plugin_module=args.plugin,
            profile=args.profile,
            startup_timeout_s=args.startup_timeout_s,
        )
        sys.stdout.write(json.dumps(gsm, indent=2, sort_keys=True) + "\n")
        return 0 if gsm["passed"] else 1

    if args.subcommand == "analyze":
        raw = json.loads(args.sweeps_json.read_text(encoding="utf-8"))
        items = raw["sweeps"] if isinstance(raw, dict) and "sweeps" in raw else raw
        recs = [_record_from_dict(item) for item in items]
        concs = [c.strip() for c in args.concurrencies.split(",") if c.strip()]
        gsm_res = (
            json.loads(args.gsm8k_json.read_text(encoding="utf-8"))
            if args.gsm8k_json is not None
            else None
        )
        suite = evaluate_acceptance_suite(
            recs, required_concurrencies=concs, gsm8k_result=gsm_res
        )
        md = format_acceptance_markdown(suite)
        if args.out_json is not None:
            args.out_json.write_text(
                json.dumps(suite, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if args.out_md is not None:
            args.out_md.write_text(md, encoding="utf-8")
        sys.stdout.write(md)
        return 0 if suite["overall_verdict"] == "PASS" else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
