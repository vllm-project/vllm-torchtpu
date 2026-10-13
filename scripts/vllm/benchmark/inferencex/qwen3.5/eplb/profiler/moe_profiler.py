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
"""MoE Router Profiling Framework for EPLB on TPUs (Lock-free In-Graph Accumulation)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, Literal

import numpy as np
import torch

try:
    disable_compiler = torch.compiler.disable
except AttributeError:

    def disable_compiler(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


logger = logging.getLogger(__name__)

ENV_ENABLE_PROFILER = "VLLM_TPU_ENABLE_MOE_PROFILER"
ENV_PROFILE_DIR = "VLLM_TPU_MOE_PROFILE_DIR"
ENV_DUMP_INTERVAL = "VLLM_TPU_MOE_PROFILE_DUMP_INTERVAL"

PhaseType = Literal["prefill", "decode", "unknown"]


def is_profiling_enabled() -> bool:
    return os.getenv(ENV_ENABLE_PROFILER, "0").lower() in ("1", "true", "yes")


def get_rank() -> int:
    for env_key in ("RANK", "JOBSET_JOB_INDEX", "TPU_WORKER_ID", "LOCAL_RANK"):
        val = os.getenv(env_key)
        if val is not None and val.isdigit():
            return int(val)
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


class MoeProfiler:
    """Lock-free profiler that reads back in-graph accumulator buffers."""

    def __init__(self, output_dir: str | None = None) -> None:
        self.output_dir = output_dir or os.getenv(
            ENV_PROFILE_DIR, "/cdk-outputs/outputs/moe_profiles"
        )
        self.rank = get_rank()
        self._registered_layers: dict[int, tuple[torch.nn.Module, int]] = {}
        self.step_count = 0
        self.total_tokens_seen = 0
        self.valid_tokens_recorded = 0
        self.padding_tokens_dropped = 0
        # Set by hook_patch when the runner's token padding mask is in use.
        self.padding_mask_active = False
        # Host-side row counts from TokenPaddingState.update (this rank, local).
        self.host_steps = 0
        self.host_valid_rows = 0
        self.host_padded_rows = 0
        self._seen_all_padding_step = False
        # Histograms at the first real step after warmup (warmup and startup
        # padding), so runtime padding = padded - padded_startup.
        self._startup_snapshot: dict[str, np.ndarray] | None = None

        self._bg_thread = threading.Thread(
            target=self._periodic_flush_loop, daemon=True
        )
        self._bg_thread.start()

    def _periodic_flush_loop(self) -> None:
        while True:
            time.sleep(int(os.getenv(ENV_DUMP_INTERVAL, "120")))
            try:
                if self._registered_layers:
                    self.dump(step_suffix="latest")
            except Exception:
                pass

    @disable_compiler
    def register_layer(
        self, layer_idx: int, layer: torch.nn.Module, num_experts: int = 512
    ) -> None:
        self._registered_layers[layer_idx] = (layer, num_experts)

    @disable_compiler
    def compute_analytics(
        self,
        hist_total: np.ndarray,
        hist_prefill: np.ndarray,
        hist_decode: np.ndarray,
    ) -> dict[str, Any]:
        def _summarize_matrix(data: np.ndarray) -> dict[str, Any]:
            active_mask = data.sum(axis=-1) > 0
            if not np.any(active_mask):
                return {"imbalance_ratio_max_mean": 0.0, "cv": 0.0}
            active_data = data[active_mask]
            means = np.mean(active_data, axis=-1)
            maxs = np.max(active_data, axis=-1)
            stds = np.std(active_data, axis=-1)
            imbalance_ratios = np.where(means > 0, maxs / np.maximum(means, 1e-12), 0)
            cvs = np.where(means > 0, stds / np.maximum(means, 1e-12), 0)
            total_per_expert = active_data.sum(axis=0)
            return {
                "mean_imbalance_ratio": float(np.mean(imbalance_ratios)),
                "peak_imbalance_ratio": float(np.max(imbalance_ratios)),
                "peak_imbalance_layer": int(np.argmax(imbalance_ratios)),
                "mean_cv": float(np.mean(cvs)),
                "top_5_hottest_experts": [
                    int(x) for x in np.argsort(-total_per_expert)[:5]
                ],
                "top_5_coldest_experts": [
                    int(x) for x in np.argsort(total_per_expert)[:5]
                ],
            }

        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "total_workload": _summarize_matrix(hist_total),
            "prefill_phase": _summarize_matrix(hist_prefill),
            "decode_phase": _summarize_matrix(hist_decode),
        }

    _HIST_KINDS = ("total", "prefill", "decode", "valid", "padded")

    @disable_compiler
    def read_histograms(self) -> dict[str, np.ndarray] | None:
        """Copies every layer's on-device histograms to host arrays [L, E]."""
        layers_snapshot = dict(self._registered_layers)
        if not layers_snapshot:
            return None
        max_layer = max(layers_snapshot.keys()) + 1
        max_experts = max(num_exp for _, num_exp in layers_snapshot.values())
        hists = {
            k: np.zeros((max_layer, max_experts), dtype=np.int64)
            for k in self._HIST_KINDS
        }
        for idx, (layer, _) in layers_snapshot.items():
            for k in self._HIST_KINDS:
                buf = getattr(layer, f"_eplb_hist_{k}", None)
                if buf is None:
                    continue
                try:
                    v = buf.detach().cpu().numpy().astype(np.int64)
                    hists[k][idx, : len(v)] = v
                except Exception:
                    pass
        return hists

    @disable_compiler
    def read_decode_stats(self) -> dict[str, np.ndarray]:
        """Copies the decode co-activation buffers to host arrays [L, ...]."""
        from hook_patch import DECODE_BUFFERS  # pylint: disable=g-import-not-at-top

        layers_snapshot = dict(self._registered_layers)
        if not layers_snapshot:
            return {}
        max_layer = max(layers_snapshot.keys()) + 1
        out = {}
        for idx, (layer, _) in layers_snapshot.items():
            for name in DECODE_BUFFERS:
                buf = getattr(layer, name, None)
                if buf is None:
                    continue
                try:
                    v = buf.detach().cpu().numpy()
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                key = name.replace("_eplb_", "")
                if key not in out:
                    out[key] = np.zeros((max_layer,) + v.shape, dtype=v.dtype)
                out[key][idx] = v
        return out

    @disable_compiler
    def note_padding_update(self, local_valid: int, local_padded: int) -> None:
        """Called by the TokenPaddingState.update wrapper once per step (host).

        Warmup steps have no valid rows. The first step with valid rows after any
        all-padding step is the first real step; the histograms are snapshotted
        there so warmup and startup padding can be separated from runtime padding.
        vLLM's memory-profiling run is also all padding. The snapshot is taken
        once; later all-padding steps (dummy steps of idle DP ranks) count as
        runtime padding.

        Args:
          local_valid: Valid rows on this rank for the step.
          local_padded: Total (padded) rows on this rank for the step.
        """
        self.host_steps += 1
        self.host_valid_rows += local_valid
        self.host_padded_rows += max(local_padded - local_valid, 0)
        if local_valid == 0:
            self._seen_all_padding_step = True
            return
        if self._seen_all_padding_step and self._startup_snapshot is None:
            self._startup_snapshot = self.read_histograms() or {}
            print(
                f"[MoeProfiler Rank {self.rank} PID {os.getpid()}] First real step"
                f" after warmup (host step {self.host_steps}); startup histograms"
                " snapshotted.",
                flush=True,
            )

    @disable_compiler
    def dump(self, step_suffix: str = "final") -> str:
        hists = self.read_histograms()
        if hists is None:
            return ""
        hist_total = hists["total"]
        hist_prefill = hists["prefill"]
        hist_decode = hists["decode"]
        max_layer, max_experts = hist_total.shape

        total_tokens = int(hist_total.sum())
        if total_tokens == 0:
            return ""

        startup = self._startup_snapshot or {}
        zeros = np.zeros_like(hist_total)
        valid_events = int(hists["valid"].sum())
        padded_events = int(hists["padded"].sum())

        os.makedirs(self.output_dir, exist_ok=True)
        base_prefix = os.path.join(
            self.output_dir,
            f"moe_profile_rank_{self.rank}_pid_{os.getpid()}_{step_suffix}",
        )
        npz_path = f"{base_prefix}.npz"
        json_path = f"{base_prefix}_summary.json"

        decode_stats = self.read_decode_stats()
        # Kept for old readers: decode valid-row co-activation, as float32.
        affinity = decode_stats.get(
            "dec_tok_coact",
            np.zeros((max_layer, max_experts, max_experts), dtype=np.int32),
        ).astype(np.float32)

        payload = dict(
            histogram_total=hist_total,
            histogram_prefill=hist_prefill,
            histogram_decode=hist_decode,
            # Valid/padded split from the runner's padding mask (all zero when
            # padding_mask_active is 0). *_startup: values at the first real step
            # after warmup (all zero if that step was never seen).
            histogram_valid=hists["valid"],
            histogram_padded=hists["padded"],
            histogram_valid_startup=startup.get("valid", zeros),
            histogram_padded_startup=startup.get("padded", zeros),
            startup_snapshot_taken=int(self._startup_snapshot is not None),
            padding_mask_active=int(self.padding_mask_active),
            host_steps=self.host_steps,
            host_valid_rows=self.host_valid_rows,
            host_padded_rows=self.host_padded_rows,
            affinity_matrix=affinity,
            total_tokens=total_tokens,
            valid_tokens=valid_events if self.padding_mask_active else total_tokens,
            padding_tokens=padded_events if self.padding_mask_active else 0,
            steps=total_tokens // 6,
            rank=self.rank,
            **decode_stats,
        )
        np.savez_compressed(npz_path, **payload)

        rank_npz = os.path.join(
            self.output_dir, f"moe_profile_rank_{self.rank}_{step_suffix}.npz"
        )
        with contextlib.suppress(Exception):
            np.savez_compressed(rank_npz, **payload)

        analytics = self.compute_analytics(hist_total, hist_prefill, hist_decode)
        analytics["valid_tokens"] = payload["valid_tokens"]
        analytics["total_tokens_seen"] = total_tokens
        analytics["padding_mask_active"] = bool(self.padding_mask_active)
        analytics["host_rows"] = {
            "steps": self.host_steps,
            "valid": self.host_valid_rows,
            "padded": self.host_padded_rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2)

        pad_note = ""
        if self.padding_mask_active:
            host_rows = self.host_valid_rows + self.host_padded_rows
            pad_note = (
                f"; padded share: histogram {padded_events / total_tokens:.3f},"
                f" host {self.host_padded_rows / max(host_rows, 1):.3f}"
            )
        print(
            f"[MoeProfiler Rank {self.rank} PID {os.getpid()}] Dumped"
            f" {total_tokens} tokens across {max_layer} layers to"
            f" {npz_path}{pad_note}",
            flush=True,
        )
        return npz_path


_PROFILER: MoeProfiler | None = None


def get_moe_profiler() -> MoeProfiler:
    global _PROFILER
    if _PROFILER is None:
        _PROFILER = MoeProfiler()
    return _PROFILER


def record_moe_routing(
    layer_idx: int,
    topk_ids: torch.Tensor,
    num_experts: int = 384,
    is_padding: torch.Tensor | None = None,
    phase: PhaseType | None = None,
) -> None:
    pass
