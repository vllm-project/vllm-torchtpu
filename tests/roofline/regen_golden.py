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
"""Generate or verify the deterministic golden solve --json scenario grid."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vllm_torchtpu.roofline import cli

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = ROOT_DIR / "fixtures"
GOLDEN_DIR = ROOT_DIR / "golden"
GOLDEN_SCENARIOS_PATH = GOLDEN_DIR / "scenarios.json"

GOLDEN_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    ("qwen3_5_397b_fp8", "qwen3_5_397b_fp8.json"),
    ("qwen3_5_397b_nvfp4", "qwen3_5_397b_nvfp4.json"),
    ("glm_5_2_fp8", "glm_5_2_fp8.json"),
    ("glm_5_2_nvfp4", "glm_5_2_nvfp4.json"),
    ("deepseek_v4_1_flash", "deepseek_v4_1_flash.json"),
    ("minimax_m3_1_flash", "minimax_m3_1_flash.json"),
)
GOLDEN_SLICES: tuple[tuple[str, int], ...] = (
    ("2x2x1", 4),
    ("2x2x2", 8),
    ("2x2x4", 16),
)
GOLDEN_MODES: tuple[str, ...] = ("mixed", "disagg")
GOLDEN_PRESETS: tuple[str, ...] = (
    "chat-8k1k",
    "coding-agent",
    "reasoning-1k8k",
)
GOLDEN_CONCURRENCIES = "1,16,64"
GOLDEN_KV_DTYPE = "fp8"
GOLDEN_HYBRID_STATE_DTYPE = "bf16"
_FLOAT_DIGITS = 4
_MAX_GRID_WORKERS = 16


def _is_hybrid_recurrent_fixture(fixture_path: Path) -> bool:
    """Return True when a locked HF config defines hybrid linear-attention layers."""
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    cfg = raw.get("text_config", raw) if isinstance(raw, dict) else {}
    if not isinstance(cfg, dict):
        return False
    layer_types = cfg.get("layer_types")
    return isinstance(layer_types, list) and any(
        str(tok) in ("linear_attention", "gdn", "kda") for tok in layer_types
    )


def _round_value(value: object) -> object:
    """Recursively round floats for deterministic JSON serialization."""
    if isinstance(value, float):
        return round(value, _FLOAT_DIGITS)
    if isinstance(value, dict):
        return {str(k): _round_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_value(v) for v in value]
    return value


def _compact_pool(pool: object) -> dict[str, object] | None:
    """Retain structural and memory fields for one pool plan."""
    if not isinstance(pool, dict):
        return None
    return {
        "role": pool["role"],
        "instances": pool["instances"],
        "topology": pool["topology"],
        "tp": pool["tp"],
        "dp": pool["dp"],
        "ep": pool["ep"],
        "pp": pool["pp"],
        "pcp": pool["pcp"],
        "kv_split": pool["kv_split"],
        "hbm_used_gb": _round_value(pool["hbm_used_gb"]),
        "hbm_limit_gb": _round_value(pool["hbm_limit_gb"]),
    }


def _compact_point(point: dict[str, object]) -> dict[str, object]:
    """Retain full physics floor, ceiling, estimate, and verdict fields per point."""
    return {
        "mode": point["mode"],
        "concurrency": point["concurrency"],
        "total_chips": point["total_chips"],
        "prefill_pool": _compact_pool(point.get("prefill_pool")),
        "decode_pool": _compact_pool(point.get("decode_pool")),
        "bound_ttft_ms": _round_value(point.get("bound_ttft_ms")),
        "bound_tpot_ms": _round_value(point.get("bound_tpot_ms")),
        "est_ttft_ms": _round_value(point.get("est_ttft_ms")),
        "est_tpot_ms": _round_value(point.get("est_tpot_ms")),
        "est_interactivity_tok_s": _round_value(point.get("est_interactivity_tok_s")),
        "token_throughput_per_chip": _round_value(
            point.get("token_throughput_per_chip")
        ),
        "floor_throughput_per_chip": _round_value(
            point.get("floor_throughput_per_chip")
        ),
        "binding": point.get("binding"),
        "slo_verdict": point.get("slo_verdict"),
        "is_pareto": point.get("is_pareto"),
    }


def _compact_pool_record(record: dict[str, object]) -> dict[str, object]:
    """Retain per-pool floor, estimate, throughput, and Pareto fields."""
    return {
        "role": record["role"],
        "concurrency": record["concurrency"],
        "pool": _compact_pool(record.get("pool")),
        "ttft_floor_ms": _round_value(record.get("ttft_floor_ms")),
        "tpot_floor_ms": _round_value(record.get("tpot_floor_ms")),
        "ttft_ms": _round_value(record.get("ttft_ms")),
        "tpot_ms": _round_value(record.get("tpot_ms")),
        "interactivity_tok_s": _round_value(record.get("interactivity_tok_s")),
        "token_throughput_per_chip": _round_value(
            record.get("token_throughput_per_chip")
        ),
        "binding": record.get("binding"),
        "is_pareto": record.get("is_pareto"),
    }


def _digest_b64(items: object) -> str:
    """Return compact base64 SHA-256 digest over canonical JSON serialization."""
    canonical = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return base64.b64encode(hashlib.sha256(canonical.encode("utf-8")).digest()).decode(
        "ascii"
    )


def run_one_scenario(
    ckpt_id: str,
    fixture_name: str,
    slice_name: str,
    max_chips: int,
    mode_and_preset: tuple[str, str],
) -> dict[str, object]:
    """Run CLI solve --json for one golden grid scenario and normalize output."""
    mode, preset = mode_and_preset
    cfg_path = FIXTURES_DIR / fixture_name
    is_hybrid = _is_hybrid_recurrent_fixture(cfg_path)
    eff_max_chips = max_chips + 4 if mode == "disagg" else max_chips
    argv = [
        "solve",
        "--config",
        str(cfg_path),
        "--kv-dtype",
        GOLDEN_KV_DTYPE,
        *(("--state-dtype", GOLDEN_HYBRID_STATE_DTYPE) if is_hybrid else ()),
        "--example-scenario",
        preset,
        "--mode",
        mode,
        "--max-chips",
        str(eff_max_chips),
        "--concurrency",
        GOLDEN_CONCURRENCIES,
        "--json",
    ]
    if mode == "disagg":
        argv.extend(
            [
                "--pairs",
                "--prefill-topology",
                "2x2x1",
                "--decode-topology",
                slice_name,
                "--top-k",
                "16",
            ]
        )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(argv)
    if rc != 0:
        msg = f"solve failed (rc={rc}) for {ckpt_id} {slice_name} {mode} {preset}"
        raise RuntimeError(msg)
    raw = json.loads(buf.getvalue())
    record = raw.get("recommended")
    compact_candidates = [_compact_point(point) for point in raw["candidates"]]
    compact_prefill = [_compact_pool_record(r) for r in raw["prefill_pools"]]
    compact_decode = [_compact_pool_record(r) for r in raw["decode_pools"]]
    compact_frontier = [_compact_point(point) for point in raw.get("frontier", ())]
    return {
        "checkpoint": ckpt_id,
        "slice": slice_name,
        "max_chips": eff_max_chips,
        "mode": mode,
        "preset": preset,
        "cache_policy": {
            "kv_dtype": GOLDEN_KV_DTYPE,
            "state_dtype": GOLDEN_HYBRID_STATE_DTYPE if is_hybrid else None,
        },
        "schema": raw["schema"],
        "sort_by": raw["sort_by"],
        "top_k_limit": raw["top_k_limit"],
        "hbm_reserve_gb": _round_value(raw["hbm_reserve_gb"]),
        "search": raw["search"],
        "warnings": raw["warnings"],
        "recommended": _compact_point(record) if isinstance(record, dict) else None,
        "frontier_count": len(compact_frontier),
        "candidate_count": len(compact_candidates),
        "prefill_pool_count": len(compact_prefill),
        "decode_pool_count": len(compact_decode),
        "frontier_sha256_b64": _digest_b64(compact_frontier),
        "candidates_sha256_b64": _digest_b64(compact_candidates),
        "prefill_pools_sha256_b64": _digest_b64(compact_prefill),
        "decode_pools_sha256_b64": _digest_b64(compact_decode),
    }


def _run_grid_task(
    task: tuple[str, str, str, int, tuple[str, str]],
) -> tuple[str, dict[str, object]]:
    """Evaluate one golden grid task and return (scenario_key, payload)."""
    ckpt_id, fixture_name, slice_name, max_chips, mode_and_preset = task
    mode, preset = mode_and_preset
    key = f"{ckpt_id}__{slice_name}__{mode}__{preset}"
    return (
        key,
        run_one_scenario(
            ckpt_id,
            fixture_name,
            slice_name,
            max_chips,
            mode_and_preset,
        ),
    )


def build_golden_grid() -> dict[str, object]:
    """Compute the full deterministic golden grid dictionary."""
    tasks = [
        (ckpt_id, fixture_name, slice_name, max_chips, (mode, preset))
        for ckpt_id, fixture_name in GOLDEN_CHECKPOINTS
        for slice_name, max_chips in GOLDEN_SLICES
        for mode in GOLDEN_MODES
        for preset in GOLDEN_PRESETS
    ]
    workers = min(_MAX_GRID_WORKERS, max(1, len(tasks)), os.cpu_count() or 1)
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            scenarios = dict(pool.map(_run_grid_task, tasks))
    else:
        scenarios = dict(map(_run_grid_task, tasks))
    return {
        "grid_version": 1,
        "concurrencies": GOLDEN_CONCURRENCIES,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint supporting --write and --check modes."""
    parser = argparse.ArgumentParser(description="Generate or check roofline goldens.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="Write golden grid JSON.")
    group.add_argument(
        "--check", action="store_true", help="Verify golden grid matches current code."
    )
    args = parser.parse_args(argv)
    payload = build_golden_grid()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_SCENARIOS_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if not GOLDEN_SCENARIOS_PATH.exists():
        sys.stderr.write(f"Missing golden file: {GOLDEN_SCENARIOS_PATH}\n")
        return 1
    existing = GOLDEN_SCENARIOS_PATH.read_text(encoding="utf-8")
    if existing != rendered:
        sys.stderr.write(
            f"Golden grid mismatch at {GOLDEN_SCENARIOS_PATH}. "
            "Run regen_golden.py --write to inspect and update.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
