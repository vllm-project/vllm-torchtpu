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
"""vLLM TPU adapter for the Qwen3.5-397B-A17B-FP8 whole-backbone decode megakernel.

Bridges vLLM's ``TPUModelRunner`` and ``UnifiedKVCache`` pool tensors to the
Pallas megakernel (`megakernel.py` + `pool_io.py`):
- Zero-HBM-overhead expert views (`stack_e=False`, `e_slots=72` or `64`) in EP
  mode (`VLLM_TPU_QWEN35_MK_MODE=ep`, default), or in-place EP->TP MoE weight
  resharding in TP mode (`VLLM_TPU_QWEN35_MK_MODE=tp`).
- Fused GDN `in_proj_qkvz` + `in_proj_ba` quantization into `lin_in` `[45, 8, 512, 2688]`.
- Dequantized BF16 router `[60, 2, 2048, 512]` and TP-sliced FP8 shared-expert
  `sh13` `[60, 1, 4096, 256]` and `sh2` `[60, 1, 128, 4096]` from `Cat_FP8_256`.
- Per-layer `smalls` packing (`[60, 1, S]` for the Pallas kernel + `[60, 16, 4096]`
  diagnostic table) and per-expert `es` scales.
- Sub-bucket compilation (`VLLM_TPU_QWEN35_MK_BUCKETS`, default `"8,16,32"`)
  over `sharded_jax_callable` with in-place donated raw pool updates.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax.sharding import PartitionSpec as P

from vllm.forward_context import get_forward_context
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_callable
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger

from . import megakernel as K
from . import pool_io

logger = init_logger(__name__)

_DEFAULT_BUCKETS = (8, 16, 32)
# MTP=3 verify rows (4 per sequence). Above 8 sequences the stock path was at
# or past parity in the per-bucket A/B, so larger batches stay on stock.
_DEFAULT_VERIFY_BUCKETS = (4, 8, 12, 16, 20, 24, 32)


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Qwen3.5 megakernel: {msg}")


# Persistent megakernel inputs must be torch_tpu base buffers. A tensor whose
# shape differs from its storage base (a reshape/unsqueeze/stack view) is
# re-materialized by a `tt_jit_execute_as_strided` copy on every compiled call
# that consumes it. Same idiom as `torch.empty(...).copy_(...)` in
# `layers/adapter/quantization/fp8.py`.
def _fresh(x: torch.Tensor) -> torch.Tensor:
    """Copy `x` into a newly allocated buffer of exactly `x.shape`."""
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)


def _stack(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """`torch.stack(xs, 0)` that returns a base buffer instead of a view.

    `torch.stack` lowers to `cat(xs, 0).view(...)`; concatenating unsqueezed
    inputs allocates the output with the stacked shape directly.
    """
    return torch.cat([x.unsqueeze(0) for x in xs], dim=0)


def _try_set(obj: Any, name: str, value: Any) -> None:
    """Set an existing attribute, skipping read-only properties."""
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except AttributeError:
            pass


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def _find_backbone(model: Any) -> Any:
    """Locate the 60-layer Qwen3NextModel backbone inside the vLLM model wrapper."""
    for path in (
        ("model", "language_model", "model"),
        ("language_model", "model"),
        ("model", "model"),
        ("model",),
        (),
    ):
        cur = model
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None and hasattr(cur, "layers") and hasattr(cur, "norm"):
            return cur
    raise ValueError(f"Cannot locate Qwen3.5 backbone in {type(model).__name__}")


def parse_mk_buckets(env_val: str | None = None) -> tuple[int, ...]:
    """Parse comma-separated T=1 decode batch buckets (e.g. '8,16,32' or '0' to disable)."""
    raw = (
        env_val
        if env_val is not None
        else os.environ.get("VLLM_TPU_QWEN35_MK_BUCKETS", "8,16,32")
    )
    if raw.strip().lower() in ("0", "none", "off", "false"):
        return ()
    buckets = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    _need(
        len(buckets) > 0 and all(1 <= b <= 64 for b in buckets),
        f"invalid VLLM_TPU_QWEN35_MK_BUCKETS={raw!r}",
    )
    return tuple(buckets)


def parse_mk_verify_buckets(env_val: str | None = None) -> tuple[int, ...]:
    """Parse comma-separated T=4 MTP verify row buckets (e.g. '8,16,24' or '0' to disable).

    The default compiles verify buckets up to 64 rows (16 sequences x 4 tokens), so
    all 1..16 active sequences run on the megakernel with zero stock fallbacks.
    """
    raw = (
        env_val
        if env_val is not None
        else os.environ.get(
            "VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS",
            ",".join(map(str, _DEFAULT_VERIFY_BUCKETS)),
        )
    )
    if raw.strip().lower() in ("0", "none", "off", "false"):
        return ()
    buckets = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    _need(
        len(buckets) > 0 and all(4 <= b <= 64 and b % 4 == 0 for b in buckets),
        f"invalid VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS={raw!r}",
    )
    return tuple(buckets)


def select_bucket(num_tokens: int, buckets: Sequence[int] = _DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket >= num_tokens."""
    for b in buckets:
        if b >= num_tokens:
            return b
    raise ValueError(f"num_tokens={num_tokens} exceeds max bucket in {buckets}")


def parse_mk_ab_enabled(env_val: str | None = None) -> bool:
    """Return True if per-bucket MK vs stock A/B latency calibration is enabled."""
    raw = (
        env_val
        if env_val is not None
        else os.environ.get("VLLM_TPU_QWEN35_MK_AB")
    )
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return os.path.exists(
        os.path.join(os.path.dirname(__file__), ".mk_ab_enabled")
    )


def parse_mk_fused_pool_io(env_val: str | None = None) -> bool:
    """Return True if fused P1+P2 pool I/O inside the megakernel is enabled (default True)."""
    raw = (
        env_val
        if env_val is not None
        else os.environ.get("VLLM_TPU_QWEN35_MK_FUSED_POOL_IO", "1")
    )
    return raw.strip().lower() not in ("0", "false", "no", "off", "none")



def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class _LatencySeries:
    """Bounded-memory sample accumulator keeping exact count/mean and up to `cap` samples for quantiles."""

    __slots__ = ("cap", "count", "sum_ms", "samples")

    def __init__(self, cap: int = 4000) -> None:
        self.cap = int(cap)
        self.count = 0
        self.sum_ms = 0.0
        self.samples: list[float] = []

    def add(self, dt_ms: float) -> None:
        v = float(dt_ms)
        idx = self.count
        self.count = idx + 1
        self.sum_ms += v
        if len(self.samples) < self.cap:
            self.samples.append(v)
        else:
            self.samples[idx % self.cap] = v

    def summary(self) -> list[int | float]:
        if self.count == 0:
            return [0, 0.0, 0.0, 0.0, 0.0]
        s = sorted(self.samples)
        mean_ms = self.sum_ms / self.count
        return [
            self.count,
            round(mean_ms, 3),
            round(_percentile(s, 0.10), 3),
            round(_percentile(s, 0.50), 3),
            round(_percentile(s, 0.90), 3),
        ]


class MKABCalibrator:
    """Deterministic per-key MK vs stock alternation and latency accumulator."""

    def __init__(
        self,
        enabled: bool = False,
        period_s: float = 60.0,
        cap: int = 4000,
    ) -> None:
        self.enabled = bool(enabled)
        self.period_s = float(period_s)
        self.cap = int(cap)
        self.counts: dict[str, int] = {}
        self.last_decision: tuple[str, str] = ("stock_early", "stock")
        self._window: dict[str, dict[str, _LatencySeries]] = {}
        self._cumulative: dict[str, dict[str, _LatencySeries]] = {}
        self._t0 = time.monotonic()

    def reset(self) -> None:
        """Clear alternation counters and latency samples (e.g. after warmup)."""
        self.counts.clear()
        self.last_decision = ("stock_early", "stock")
        self._window.clear()
        self._cumulative.clear()
        self._t0 = time.monotonic()

    def choose(self, key: str, eligible: bool) -> tuple[bool, str, str]:
        """Return `(actual_run_mk, mix_key, path)` and record `(key, path)` in `last_decision`.

        For MK-eligible steps (`eligible=True`, e.g. `key="mk_t4_b16"`), alternates
        deterministically per `key`: even counter -> `(True, "mk_t4_b16", "mk")`,
        odd counter -> `(False, "ab_stock_t4_b16", "stock")`.
        """
        if not eligible:
            self.last_decision = (key, "stock")
            return False, key, "stock"
        if not self.enabled:
            self.last_decision = (key, "mk")
            return True, key, "mk"
        cnt = self.counts.get(key, 0)
        self.counts[key] = cnt + 1
        if (cnt & 1) == 0:
            self.last_decision = (key, "mk")
            return True, key, "mk"
        suffix = key[3:] if key.startswith("mk_") else key
        mix_key = f"ab_stock_{suffix}"
        self.last_decision = (key, "stock")
        return False, mix_key, "stock"

    def _add_to(
        self,
        table: dict[str, dict[str, _LatencySeries]],
        key: str,
        subkey: str,
        dt_ms: float,
    ) -> None:
        bucket_map = table.get(key)
        if bucket_map is None:
            bucket_map = {}
            table[key] = bucket_map
        series = bucket_map.get(subkey)
        if series is None:
            series = _LatencySeries(self.cap)
            bucket_map[subkey] = series
        series.add(dt_ms)

    def record(
        self,
        dt_ms: float,
        padded_tokens: int | None = None,
        now: float | None = None,
    ) -> tuple[float, str, str] | None:
        """Record `dt_ms` for `last_decision`; return `(elapsed_s, window_json, cumulative_json)` when period elapses."""
        if not self.enabled:
            return None
        key, path = self.last_decision
        for tbl in (self._window, self._cumulative):
            self._add_to(tbl, key, path, dt_ms)
            if path == "stock" and padded_tokens is not None:
                self._add_to(tbl, key, f"stock_p{int(padded_tokens)}", dt_ms)
        ts = time.monotonic() if now is None else float(now)
        elapsed = ts - self._t0
        if elapsed >= self.period_s:
            win_json = self.format_json(window=True)
            cum_json = self.format_json(window=False)
            self._window.clear()
            self._t0 = ts
            return elapsed, win_json, cum_json
        return None

    def summary_dict(
        self, window: bool = False
    ) -> dict[str, dict[str, list[int | float]]]:
        src = self._window if window else self._cumulative
        out: dict[str, dict[str, list[int | float]]] = {}
        for k in sorted(src.keys()):
            sub = src[k]
            out[k] = {sk: sub[sk].summary() for sk in sorted(sub.keys())}
        return out

    def format_json(self, window: bool = False) -> str:
        return json.dumps(self.summary_dict(window=window), sort_keys=True)


# ---------------------------------------------------------------------------
# Weight Extraction & Packing Helpers
# ---------------------------------------------------------------------------


def extract_kn_fp8_weight(
    proj: Any, hidden_dim: int, out_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(w_kn [hidden_dim, out_dim] fp8, s_n [out_dim] f32)` from a linear projection."""
    w = proj.weight.data
    flipped = getattr(proj, WEIGHT_FLIPPED_ATTR, False)
    if flipped or tuple(w.shape) == (hidden_dim, out_dim):
        w_kn = w
    elif tuple(w.shape) == (out_dim, hidden_dim):
        w_kn = w.transpose(0, 1)
    else:
        raise ValueError(
            f"Unexpected weight shape {tuple(w.shape)}; expected "
            f"({hidden_dim}, {out_dim}) or ({out_dim}, {hidden_dim})"
        )

    s = getattr(proj, "weight_scale", None)
    if s is None:
        s = getattr(proj, "weight_scale_inv", None)

    if w_kn.dtype == torch.float8_e4m3fn:
        _need(s is not None, "FP8 linear projection is missing weight_scale")
        s_1d = s.data.float().reshape(-1).contiguous()
        _need(
            s_1d.shape == (out_dim,),
            f"FP8 weight_scale shape {tuple(s_1d.shape)} != ({out_dim},)",
        )
        return w_kn.contiguous(), s_1d

    # Fallback: quantize BF16/FP32 weight [hidden_dim, out_dim] per output channel.
    w_nk_f32 = w_kn.transpose(0, 1).float()
    amax = w_nk_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    w_fp8_kn = (
        (w_nk_f32 / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    return w_fp8_kn, scale.squeeze(-1).float().contiguous()


def quantize_ba_weight(
    in_proj_ba: Any, hidden_dim: int = 4096, vh: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 `in_proj_ba` (`2*vh` channels) per-channel to FP8 `[hidden_dim, 2*vh]`."""
    out_dim = 2 * vh
    w = in_proj_ba.weight.data
    flipped = getattr(in_proj_ba, WEIGHT_FLIPPED_ATTR, False)
    if flipped or tuple(w.shape) == (hidden_dim, out_dim):
        w_nk = w.transpose(0, 1)
    elif tuple(w.shape) == (out_dim, hidden_dim):
        w_nk = w
    else:
        raise ValueError(
            f"Unexpected in_proj_ba.weight shape {tuple(w.shape)}; "
            f"expected ({out_dim}, {hidden_dim}) or ({hidden_dim}, {out_dim})"
        )

    if w_nk.dtype == torch.float8_e4m3fn:
        s = getattr(in_proj_ba, "weight_scale", None)
        _need(s is not None, "FP8 in_proj_ba missing weight_scale")
        return w_nk.transpose(0, 1).contiguous(), s.data.float().reshape(out_dim).contiguous()

    w_f32 = w_nk.float()
    amax = w_f32.abs().amax(dim=-1, keepdim=True)
    ba_scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    ba_fp8_kn = (
        (w_f32 / ba_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    return ba_fp8_kn, ba_scale.squeeze(-1).float().contiguous()


def pack_lin_in_layer(
    gdn_layer: Any, c: K.MKConfig = K.FULL, tk_lin_in: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single GDN layer's `in_proj_qkvz` + `in_proj_ba` into tiled `lin_in` and `in_s`.

    Returns:
      - `w_lin_in`: `[H // tk_lin_in, tk_lin_in, NLI]` (`[8, 512, 2688]`) `float8_e4m3fn`
      - `in_s`: `[max(NLI, NAI)]` (`[2688]`) `float32`
    """
    qkvz_dim = c.QKV + c.Z  # 1536 + 1024 = 2560
    w_qkvz, s_qkvz = extract_kn_fp8_weight(gdn_layer.in_proj_qkvz, c.H, qkvz_dim)
    ba_fp8, ba_scale = quantize_ba_weight(gdn_layer.in_proj_ba, c.H, c.vh)

    ba_pad = torch.zeros((c.H, 128), dtype=torch.float8_e4m3fn, device=w_qkvz.device)
    ba_pad[:, 0 : c.vh] = ba_fp8[:, 0 : c.vh]
    ba_pad[:, 64 : 64 + c.vh] = ba_fp8[:, c.vh : 2 * c.vh]

    w_cat = torch.cat([w_qkvz, ba_pad], dim=1).contiguous()
    w_tiled = w_cat.view(c.H // tk_lin_in, tk_lin_in, c.NLI)

    s_width = max(c.NLI, c.NAI)
    in_s = torch.zeros((s_width,), dtype=torch.float32, device=s_qkvz.device)
    in_s[0:qkvz_dim] = s_qkvz
    in_s[c.BA : c.BA + c.vh] = ba_scale[0 : c.vh]
    in_s[c.BA + 64 : c.BA + 64 + c.vh] = ba_scale[c.vh : 2 * c.vh]
    return w_tiled, in_s


def pack_lin_out_layer(
    gdn_layer: Any, c: K.MKConfig = K.FULL, tk_lin_out: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single GDN layer's `out_proj` into `[Z // tk_lin_out, tk_lin_out, H]` and `out_s`."""
    w_kn, s_n = extract_kn_fp8_weight(gdn_layer.out_proj, c.Z, c.H)
    return w_kn.view(c.Z // tk_lin_out, tk_lin_out, c.H), s_n


def pack_att_in_layer(
    attn_layer: Any, c: K.MKConfig = K.FULL, tk_att_in: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single full-attention layer's `qkv_proj` into `[H // tk_att_in, tk_att_in, NAI]` and `in_s`."""
    if hasattr(attn_layer, "qkv_proj"):
        w_kn, s_n = extract_kn_fp8_weight(attn_layer.qkv_proj, c.H, c.NAI)
    else:
        q_dim = 2 * c.nh * c.HD
        kv_dim = c.HD
        wq, sq = extract_kn_fp8_weight(attn_layer.q_proj, c.H, q_dim)
        wk, sk = extract_kn_fp8_weight(attn_layer.k_proj, c.H, kv_dim)
        wv, sv = extract_kn_fp8_weight(attn_layer.v_proj, c.H, kv_dim)
        w_kn = torch.cat([wq, wk, wv], dim=1).contiguous()
        s_n = torch.cat([sq, sk, sv], dim=0).contiguous()
    s_width = max(c.NLI, c.NAI)
    in_s = torch.zeros((s_width,), dtype=torch.float32, device=s_n.device)
    in_s[0 : c.NAI] = s_n
    return w_kn.view(c.H // tk_att_in, tk_att_in, c.NAI), in_s


def pack_att_out_layer(
    attn_layer: Any, c: K.MKConfig = K.FULL, tk_att_out: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single full-attention layer's `o_proj` into `[nh*HD // tk_att_out, tk_att_out, H]` and `out_s`."""
    k_dim = c.nh * c.HD
    w_kn, s_n = extract_kn_fp8_weight(attn_layer.o_proj, k_dim, c.H)
    return w_kn.view(k_dim // tk_att_out, tk_att_out, c.H), s_n


def pack_router_and_shared_layer(
    mlp: Any,
    c: K.MKConfig = K.FULL,
    tp_rank: int = 0,
    tk_router: int = 2048,
    tk_sh13: int = 4096,
    tk_sh2: int = 128,
) -> dict[str, torch.Tensor]:
    """Extract router (`bf16`), shared expert `sh13`/`sh2` (`fp8`), and scales from `mlp`.

    Handles both `Cat_FP8_256` fused attributes (`_fused_gate_gu_fp8_weight`,
    `_fused_se_down_fp8_weight`, `_fused_se_gate_weight`) and unfused `mlp.gate` +
    `mlp.shared_expert` + `mlp.shared_expert_gate`.
    """
    w_gu = getattr(mlp, "_fused_gate_gu_fp8_weight", None)
    if w_gu is not None:
        s_gu = mlp._fused_gate_gu_fp8_scale.data.float().reshape(-1)
        if hasattr(mlp, "_fused_se_down_fp8_weight"):
            w_down = mlp._fused_se_down_fp8_weight.data
            s_down = mlp._fused_se_down_fp8_scale.data.float().reshape(c.H).contiguous()
        else:
            down_proj = mlp.shared_expert.down_proj
            down_rows = int(
                down_proj.weight.shape[0]
                if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
                or down_proj.weight.shape[1] == c.H
                else down_proj.weight.shape[1]
            )
            w_down, s_down = extract_kn_fp8_weight(down_proj, down_rows, c.H)
        if hasattr(mlp, "_fused_se_gate_weight"):
            sh_gate = mlp._fused_se_gate_weight.data.float().reshape(c.H).contiguous()
        else:
            sh_gate = (
                mlp.shared_expert_gate.weight.data.float().reshape(c.H).contiguous()
            )

        # Dequantize router columns [:, :E] back to BF16
        w_router_fp8 = w_gu.data[:, : c.E]
        s_router = s_gu[: c.E]
        w_router_bf16 = (
            (w_router_fp8.float() * s_router.unsqueeze(0))
            .to(torch.bfloat16)
            .contiguous()
            .view(c.H // tk_router, tk_router, c.E)
        )

        # Shared-expert gate|up: either replicated [4096, E + 2*IS] or TP-sliced [4096, E + 2*ISR]
        n_cols = int(w_gu.shape[1])
        if n_cols == c.E + 2 * c.IS:
            g0 = c.E + tp_rank * c.ISR
            u0 = c.E + c.IS + tp_rank * c.ISR
            w_sh13 = torch.cat(
                [w_gu.data[:, g0 : g0 + c.ISR], w_gu.data[:, u0 : u0 + c.ISR]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu[g0 : g0 + c.ISR], s_gu[u0 : u0 + c.ISR]], dim=0
            ).contiguous()
        elif n_cols == c.E + 2 * c.ISR:
            w_sh13 = w_gu.data[:, c.E : c.E + 2 * c.ISR].contiguous()
            s_sh13 = s_gu[c.E : c.E + 2 * c.ISR].contiguous()
        else:
            raise ValueError(
                f"Unexpected _fused_gate_gu_fp8_weight shape {tuple(w_gu.shape)}"
            )

        if int(w_down.shape[0]) == c.IS:
            r0 = tp_rank * c.ISR
            w_sh2 = w_down[r0 : r0 + c.ISR, :].contiguous()
        elif int(w_down.shape[0]) == c.ISR:
            w_sh2 = w_down.contiguous()
        else:
            raise ValueError(
                f"Unexpected _fused_se_down_fp8_weight shape {tuple(w_down.shape)}"
            )
    else:
        # Unfused fallback
        gate_w = mlp.gate.weight.data
        if tuple(gate_w.shape) == (c.E, c.H):
            gate_w = gate_w.transpose(0, 1)
        w_router_bf16 = (
            gate_w.to(torch.bfloat16)
            .contiguous()
            .view(c.H // tk_router, tk_router, c.E)
        )

        gu_proj = mlp.shared_expert.gate_up_proj
        down_proj = mlp.shared_expert.down_proj
        gu_cols = int(
            gu_proj.weight.shape[1]
            if getattr(gu_proj, WEIGHT_FLIPPED_ATTR, False)
            or gu_proj.weight.shape[0] == c.H
            else gu_proj.weight.shape[0]
        )
        w_gu_only, s_gu_only = extract_kn_fp8_weight(gu_proj, c.H, gu_cols)
        if gu_cols == 2 * c.IS:
            g0 = tp_rank * c.ISR
            u0 = c.IS + tp_rank * c.ISR
            w_sh13 = torch.cat(
                [w_gu_only[:, g0 : g0 + c.ISR], w_gu_only[:, u0 : u0 + c.ISR]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu_only[g0 : g0 + c.ISR], s_gu_only[u0 : u0 + c.ISR]], dim=0
            ).contiguous()
        elif gu_cols == 2 * c.ISR:
            w_sh13 = w_gu_only.contiguous()
            s_sh13 = s_gu_only.contiguous()
        else:
            raise ValueError(f"Unexpected shared_expert.gate_up_proj cols {gu_cols}")

        down_rows = int(
            down_proj.weight.shape[0]
            if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
            or down_proj.weight.shape[1] == c.H
            else down_proj.weight.shape[1]
        )
        w_down_only, s_down = extract_kn_fp8_weight(down_proj, down_rows, c.H)
        if down_rows == c.IS:
            r0 = tp_rank * c.ISR
            w_sh2 = w_down_only[r0 : r0 + c.ISR, :].contiguous()
        elif down_rows == c.ISR:
            w_sh2 = w_down_only.contiguous()
        else:
            raise ValueError(f"Unexpected shared_expert.down_proj rows {down_rows}")

        sh_gate = mlp.shared_expert_gate.weight.data.float().reshape(c.H).contiguous()

    return {
        "router": w_router_bf16,
        "sh13": w_sh13.view(c.H // tk_sh13, tk_sh13, 2 * c.ISR),
        "sh2": w_sh2.view(c.ISR // tk_sh2, tk_sh2, c.H),
        "sh13_s": s_sh13,
        "sh2_s": s_down,
        "sh_gate": sh_gate,
    }


def _get_routed_experts(mlp: Any) -> Any:
    """Return the underlying FusedMoE module holding `w13_weight` / `w2_weight`."""
    experts = mlp.experts
    return getattr(experts, "routed_experts", experts)


def extract_ecopy_layer(
    layer_idx: int | None,
    layer: Any,
    c: K.MKConfig,
    dev: torch.device,
) -> torch.Tensor | None:
    """Build the `[E]` float32 `ecopy` descriptor vector for `smalls['ecopy']` on `layer_idx`.

    For each virtual expert `v = r0 * ER + s0` (`0 <= v < E`):
      - `0.0` if `v` has no secondary copy slot
      - `float(1 + r1 * (EW - ER) + (s1 - ER))` if `v` is duplicated on rank `r1`
        at local slot `s1` (`ER <= s1 < EW`)
    """
    if c.moe_tp or c.e_slots <= c.ER:
        return None
    mlp = getattr(layer, "mlp", None)
    routed = _get_routed_experts(mlp) if mlp is not None and hasattr(mlp, "experts") else None
    if routed is not None:
        ec_buf = getattr(routed, "_ecopy", None)
        if ec_buf is not None:
            return ec_buf.to(device=dev, dtype=torch.float32).reshape(c.E)
        ec_cores = getattr(routed, "_ecopy_cores", None)
        if ec_cores is not None:
            return torch.tensor(
                K.encode_ecopy(c, ec_cores), dtype=torch.float32, device=dev
            )
        if layer_idx is None:
            prefix = getattr(routed, "layer_name", "") or ""
            from vllm_torchtpu import expert_placement

            layer_idx = expert_placement.layer_index(prefix)
    if layer_idx is not None:
        from vllm_torchtpu import expert_placement

        if (
            expert_placement.has_copies()
            and int(expert_placement.slots()) == int(c.EW)
            and int(expert_placement.global_count()) == int(c.E)
            and int(expert_placement.shards()) == int(c.R)
            and str(layer_idx) in expert_placement.data().get("layers", {})
        ):
            cores = expert_placement.cores(layer_idx)
            return torch.tensor(
                K.encode_ecopy(c, cores), dtype=torch.float32, device=dev
            )
    return None


def pack_smalls_layer(
    layer: Any,
    c: K.MKConfig,
    in_s: torch.Tensor,
    out_s: torch.Tensor,
    sh13_s: torch.Tensor,
    sh2_s: torch.Tensor,
    sh_gate: torch.Tensor,
    final_norm_w: torch.Tensor,
    *,
    layer_idx: int | None = None,
    ecopy: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a single layer's small parameters into both:
    1. `row_mk`: `[1, S] float32` following `megakernel.smalls_layout(c)` (`S = 28928` for `FULL`,
       or `29440` when `e_slots > ER` includes `ecopy[512]`).
    2. `row_16x4096`: `[16, H] float32` (`[16, 4096]`) diagnostic table.
    """
    lay, S = K.smalls_layout(c)
    dev = in_s.device

    in_norm_w = layer.input_layernorm.weight.data.float().reshape(c.H)
    post_norm_w = layer.post_attention_layernorm.weight.data.float().reshape(c.H)
    fn_w = final_norm_w.data.float().reshape(c.H)

    fields: dict[str, torch.Tensor] = {
        "in_norm": in_norm_w,
        "post_norm": post_norm_w,
        "in_s": in_s.float().reshape(-1),
        "out_s": out_s.float().reshape(c.H),
        "sh13_s": sh13_s.float().reshape(2 * c.ISR),
        "sh2_s": sh2_s.float().reshape(c.H),
        "sh_gate": sh_gate.float().reshape(c.H),
    }
    if "ecopy" in lay:
        if ecopy is None:
            ecopy = extract_ecopy_layer(layer_idx, layer, c, dev)
        if ecopy is not None:
            fields["ecopy"] = ecopy.float().reshape(c.E)

    row_16 = torch.zeros((16, c.H), dtype=torch.float32, device=dev)
    row_16[0] = in_norm_w
    row_16[1] = post_attention_w = post_norm_w
    row_16[2, : in_s.numel()] = in_s.float().reshape(-1)
    row_16[3] = out_s.float().reshape(c.H)
    row_16[4, : 2 * c.ISR] = sh13_s.float().reshape(2 * c.ISR)
    row_16[5] = sh2_s.float().reshape(c.H)
    row_16[6] = sh_gate.float().reshape(c.H)

    if getattr(layer, "layer_type", None) == "linear_attention":
        la = layer.linear_attn
        # conv1d.weight is [QKV, 1, 4] or [QKV, 4] -> transpose to [4, QKV] tap-major
        conv_w = la.conv1d.weight.data.float().reshape(c.QKV, 4).transpose(0, 1).contiguous()
        alog_w = la.A_log.data.float().reshape(c.vh)
        dtb_w = la.dt_bias.data.float().reshape(c.vh)
        gnorm_w = la.norm.weight.data.float().reshape(c.VD)

        fields["conv"] = conv_w.reshape(4 * c.QKV)
        fields["alog"] = F.pad(alog_w, (0, 128 - c.vh))
        fields["dtb"] = F.pad(dtb_w, (0, 128 - c.vh))
        fields["gnorm"] = gnorm_w

        for tap in range(4):
            row_16[7 + tap, : c.QKV] = conv_w[tap]
        row_16[11, : c.vh] = alog_w
        row_16[11, 64 : 64 + c.vh] = dtb_w
        row_16[12, : c.VD] = gnorm_w
    else:
        sa = layer.self_attn
        qn_w = sa.q_norm.weight.data.float().reshape(c.HD)
        kn_w = sa.k_norm.weight.data.float().reshape(c.HD)
        fields["qn"] = qn_w
        fields["kn"] = kn_w
        row_16[13, : c.HD] = qn_w
        row_16[14, : c.HD] = kn_w

    row_16[15] = fn_w

    parts = []
    for name, (_, width) in lay.items():
        v = fields.get(name)
        if v is None:
            v = torch.zeros((width,), dtype=torch.float32, device=dev)
        else:
            v = v.reshape(-1).float()
        _need(v.numel() == width, f"smalls field {name}: {v.numel()} != {width}")
        pad_len = K.rup(width) - width
        if pad_len > 0:
            v = F.pad(v, (0, pad_len))
        parts.append(v)

    row_mk = torch.cat(parts, dim=0).view(1, S).contiguous()
    return row_mk, row_16


def pack_expert_views_layer(
    mlp: Any,
    c: K.MKConfig,
    tk_e13: int,
    tk_e2: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create zero-copy 4D views of a layer's expert weights and pack its expert scales.

    Returns:
      - `e13_view`: `[EW, H // tk_e13, tk_e13, 2 * IR]` `float8_e4m3fn` (zero-copy view)
      - `e2_view`: `[EW, IR // tk_e2, tk_e2, H]` `float8_e4m3fn` (zero-copy view)
      - `es_mk`: `[EW, 1, 2 * IR + H]` `float32` (for `megakernel.py`)
      - `es_8x8192`: `[EW, 8, 8192]` `float32` (8-sublane broadcast table)
    """
    experts = _get_routed_experts(mlp)
    w13 = experts.w13_weight
    w2 = experts.w2_weight
    ew = int(w13.shape[0])
    _need(
        ew == c.EW,
        f"experts.w13_weight.shape[0]={ew} != c.EW={c.EW}",
    )
    _need(
        w13.is_contiguous() and w2.is_contiguous(),
        "expert weights must be contiguous for zero-copy views",
    )

    if not getattr(c, "stack_e", True):
        # Pass the base 3D Parameter tensors [EW, H, 2*IR] and [EW, IR, H]
        # directly without calling .view(...) in PyTorch. In torch_tpu, calling
        # .view(...) on a materialized PJRT tensor creates a deferred as_strided
        # node that triggers tt_jit_execute_as_strided (allocating duplicate 4D
        # buffers). Reshaping 3D -> 4D inside JAX shard_map
        # (pool_io.make_flat_pooled_step) is a zero-copy XLA bitcast.
        e13_view = w13
        e2_view = w2
    else:
        t13 = c.H // tk_e13
        t2 = c.IR // tk_e2
        e13_view = w13.view(ew, t13, tk_e13, 2 * c.IR)
        e2_view = w2.view(ew, t2, tk_e2, c.H)
    _need(
        e13_view.data_ptr() == w13.data_ptr() and e2_view.data_ptr() == w2.data_ptr(),
        "expert views must share storage with w13_weight / w2_weight",
    )

    s13_raw = getattr(experts, "w13_weight_scale_inv", None)
    if s13_raw is None:
        s13_raw = getattr(experts, "w13_weight_scale")
    s2_raw = getattr(experts, "w2_weight_scale_inv", None)
    if s2_raw is None:
        s2_raw = getattr(experts, "w2_weight_scale")

    s13 = s13_raw.data.float().reshape(ew, 2 * c.IR)
    s2 = s2_raw.data.float().reshape(ew, c.H)
    es_cat = torch.cat([s13, s2], dim=-1).contiguous()  # [ew, 2*IR + H]
    es_mk = es_cat.unsqueeze(1).contiguous()  # [ew, 1, 2*IR + H]

    if es_cat.device.type == "tpu":
        es_8x8192 = torch.empty((0,), dtype=torch.float32, device=es_cat.device)
    else:
        pad_cols = 8192 - es_cat.shape[-1]
        es_pad = F.pad(es_cat, (0, pad_cols)) if pad_cols > 0 else es_cat
        es_8x8192 = es_pad.unsqueeze(1).expand(ew, 8, es_pad.shape[-1]).contiguous()
    return e13_view, e2_view, es_mk, es_8x8192


def reshard_experts_ep_to_tp_local(
    w13_all: torch.Tensor,
    w2_all: torch.Tensor,
    s13_all: torch.Tensor,
    s2_all: torch.Tensor,
    tp_rank: int,
    c: K.MKConfig = K.FULL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice global 512-expert weights/scales into `tp_rank`'s TP-sharded expert tensors.

    Args:
      w13_all: `[E, H, 2*I]` (`[512, 4096, 2048]`) `float8_e4m3fn`
      w2_all: `[E, I, H]` (`[512, 1024, 4096]`) `float8_e4m3fn`
      s13_all: `[E, 2*I]` (`[512, 2048]`) `float32`
      s2_all: `[E, H]` (`[512, 4096]`) `float32`
      tp_rank: rank in `0..R-1`
      c: `MKConfig` with `moe_tp=True`

    Returns:
      - `w13_tp`: `[E, H, 2*IR]` (`[512, 4096, 256]`) `float8_e4m3fn`
      - `w2_tp`: `[E, IR, H]` (`[512, 128, 4096]`) `float8_e4m3fn`
      - `s13_tp`: `[E, 1, 1, 2*IR]` (`[512, 1, 1, 256]`) `float32`
      - `s2_tp`: `[E, 1, 1, H]` (`[512, 1, 1, H]`) `float32`
    """
    ir = c.I // c.R  # 128
    g0 = tp_rank * ir
    u0 = c.I + tp_rank * ir

    w13_tp = torch.cat(
        [w13_all[:, :, g0 : g0 + ir], w13_all[:, :, u0 : u0 + ir]], dim=-1
    ).contiguous()
    w2_tp = w2_all[:, g0 : g0 + ir, :].contiguous()

    s13_flat = s13_all.float().reshape(c.E, 2 * c.I)
    s2_flat = s2_all.float().reshape(c.E, c.H)
    s13_tp = _fresh(
        torch.cat(
            [s13_flat[:, g0 : g0 + ir], s13_flat[:, u0 : u0 + ir]], dim=-1
        ).reshape(c.E, 1, 1, 2 * ir)
    )
    s2_tp = _fresh(s2_flat.reshape(c.E, 1, 1, c.H))
    return w13_tp, w2_tp, s13_tp, s2_tp


def apply_tp_moe_flags(experts: Any) -> None:
    """Update a vLLM `FusedMoE` / `MoERunner` module's flags after in-place EP->TP weight resharding."""
    targets = [experts]
    routed = getattr(experts, "routed_experts", None)
    if routed is not None and routed is not experts:
        targets.append(routed)
    for tgt in targets:
        moe_cfg = getattr(tgt, "moe_config", None)
        if moe_cfg is not None:
            p_cfg = getattr(moe_cfg, "moe_parallel_config", None)
            if p_cfg is not None:
                _try_set(p_cfg, "use_ep", False)
                # `FusedMoEParallelConfig.make` stores an EP world as
                # `tp_size=1, ep_size=R`; TP MoE needs `tp_size=R, ep_size=1`.
                if getattr(p_cfg, "ep_size", 1) > 1 and getattr(p_cfg, "tp_size", 1) == 1:
                    _try_set(p_cfg, "tp_size", p_cfg.ep_size)
                    _try_set(p_cfg, "tp_rank", getattr(p_cfg, "ep_rank", 0))
                    _try_set(p_cfg, "ep_size", 1)
                    _try_set(p_cfg, "ep_rank", 0)
                _try_set(p_cfg, "sp_size", 1)
                _try_set(p_cfg, "is_sequence_parallel", False)
            _try_set(moe_cfg, "use_ep", False)
            _try_set(moe_cfg, "is_sequence_parallel", False)
        _try_set(tgt, "_tpu_fused_ep_eligible", False)
        # `expert_map` / `expert_mask` are read-only properties over these
        # buffers on vLLM's `MoERunner` / `RoutedExperts`.
        for buf in ("_experts_start", "_expert_map", "expert_mask"):
            if buf in getattr(tgt, "_buffers", {}):
                tgt.register_buffer(buf, None, persistent=False)
            else:
                _try_set(tgt, buf, None)
        _try_set(tgt, "_gmm_primary_experts", None)
        _try_set(tgt, "_lowtok_has_copies", False)
        if hasattr(tgt, "global_num_experts"):
            _try_set(tgt, "local_num_experts", tgt.global_num_experts)
        for qm_attr in ("quant_method", "_quant_method"):
            qm = getattr(tgt, qm_attr, None)
            if qm is not None:
                _try_set(qm, "_tpu_fused_ep_eligible", False)
                _try_set(qm, "_moe_lowtok_scores_op", None)
                _try_set(qm, "_fused_moe_ep_op", None)
                _try_set(qm, "_use_adaptive_fused_moe", False)
        if (
            moe_cfg is not None
            and hasattr(moe_cfg, "experts_per_token")
            and hasattr(tgt, "activation")
        ):
            from vllm_torchtpu.layers.adapter.fused_moe import (
                prebuild_fused_moe_kernel,
            )

            act_val = getattr(tgt.activation, "value", tgt.activation)
            prebuild_fused_moe_kernel(
                topk=moe_cfg.experts_per_token,
                activation=act_val,
                use_ep=False,
                num_primary_experts=None,
            )


def reshard_layer_experts_ep_to_tp(
    mlp: Any,
    tp_rank: int,
    c: K.MKConfig = K.FULL,
    all_gather_fn: Any = None,
) -> None:
    """Reshard a single layer's `mlp.experts` from EP `[e_slots, ...]` to TP `[512, ...]` in-place."""
    if all_gather_fn is None:
        from vllm.distributed import tensor_model_parallel_all_gather

        all_gather_fn = lambda x: tensor_model_parallel_all_gather(x, 0)

    experts = _get_routed_experts(mlp)
    er = c.E // c.R  # 64 primary owned experts per rank
    w13_own = experts.w13_weight.data[:er].contiguous()
    w2_own = experts.w2_weight.data[:er].contiguous()

    s13_param = getattr(experts, "w13_weight_scale_inv", None)
    if s13_param is None:
        s13_param = getattr(experts, "w13_weight_scale")
    s2_param = getattr(experts, "w2_weight_scale_inv", None)
    if s2_param is None:
        s2_param = getattr(experts, "w2_weight_scale")
    s13_own = s13_param.data[:er].reshape(er, 2 * c.I).contiguous()
    s2_own = s2_param.data[:er].reshape(er, c.H).contiguous()

    w13_all = all_gather_fn(w13_own)
    w2_all = all_gather_fn(w2_own)
    s13_all = all_gather_fn(s13_own)
    s2_all = all_gather_fn(s2_own)

    w13_tp, w2_tp, s13_tp, s2_tp = reshard_experts_ep_to_tp_local(
        w13_all, w2_all, s13_all, s2_all, tp_rank=tp_rank, c=c
    )
    del w13_all, w2_all, s13_all, s2_all
    del w13_own, w2_own, s13_own, s2_own

    experts.w13_weight = torch.nn.Parameter(w13_tp, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2_tp, requires_grad=False)
    if hasattr(experts, "w13_weight_scale_inv"):
        experts.w13_weight_scale_inv = torch.nn.Parameter(s13_tp, requires_grad=False)
        s13_new_param = experts.w13_weight_scale_inv
    else:
        experts.w13_weight_scale = torch.nn.Parameter(s13_tp, requires_grad=False)
        s13_new_param = experts.w13_weight_scale
    if hasattr(experts, "w2_weight_scale_inv"):
        experts.w2_weight_scale_inv = torch.nn.Parameter(s2_tp, requires_grad=False)
        s2_new_param = experts.w2_weight_scale_inv
    else:
        experts.w2_weight_scale = torch.nn.Parameter(s2_tp, requires_grad=False)
        s2_new_param = experts.w2_weight_scale

    if w13_tp.device.type == "tpu":
        from vllm_torchtpu.utils import synchronize_tensors

        synchronize_tensors(
            [experts.w13_weight, experts.w2_weight, s13_new_param, s2_new_param]
        )

    apply_tp_moe_flags(mlp.experts)


def pack_weights(
    backbone: Any,
    c: K.MKConfig,
    tp_rank: int = 0,
    tiles: K.Tiles | None = None,
) -> dict[str, Any]:
    """Pack all 60 layers of `backbone` into the megakernel's weight dictionary.

    When `c.stack_e` is `False` (default for vLLM integration), `weights["e13"]`
    and `weights["e2"]` are `L`-tuples of 4D zero-copy views into each layer's
    `mlp.experts.w13_weight` and `mlp.experts.w2_weight` (0 extra HBM).

    Returns a dict with:
      - All keys in `K.W_NAMES`: `"lin_in"`, `"lin_out"`, `"att_in"`, `"att_out"`,
        `"router"`, `"sh13"`, `"sh2"`, `"e13"`, `"e2"`, `"es"`, `"smalls"`, `"final_norm"`.
      - `"w_flat"`: flat list of `torch.Tensor` leaves in `K.W_NAMES` order (`130` tensors
        when `L=60, stack_e=False`, or `12` tensors when `stack_e=True`).
      - `"smalls_16x4096"`: `[L, 16, H]` `float32` table.
      - `"es_8x8192"`: `[L, EW, 8, 8192]` `float32` table.
    """
    if tiles is None:
        tiles = K.default_tiles(c)
    p = K.plan(c, tiles)
    tk = p["tk"]

    layers = backbone.layers
    _need(len(layers) == c.L, f"expected {c.L} layers, got {len(layers)}")

    lin_in_list: list[torch.Tensor] = []
    lin_out_list: list[torch.Tensor] = []
    att_in_list: list[torch.Tensor] = []
    att_out_list: list[torch.Tensor] = []
    router_list: list[torch.Tensor] = []
    sh13_list: list[torch.Tensor] = []
    sh2_list: list[torch.Tensor] = []
    e13_list: list[torch.Tensor] = []
    e2_list: list[torch.Tensor] = []
    es_mk_list: list[torch.Tensor] = []
    es_8_list: list[torch.Tensor] = []
    smalls_mk_list: list[torch.Tensor] = []
    smalls_16_list: list[torch.Tensor] = []

    final_norm_w = backbone.norm.weight.data.float().reshape(1, c.H).contiguous()

    for l in range(c.L):
        layer = layers[l]
        ltype = getattr(layer, "layer_type", None)
        if ltype is None:
            ltype = (
                "full_attention"
                if (l % c.att_every == c.att_every - 1)
                else "linear_attention"
            )

        if ltype == "linear_attention":
            w_in, in_s = pack_lin_in_layer(layer.linear_attn, c, tk["lin_in"])
            w_out, out_s = pack_lin_out_layer(layer.linear_attn, c, tk["lin_out"])
            lin_in_list.append(w_in)
            lin_out_list.append(w_out)
        elif ltype == "full_attention":
            w_in, in_s = pack_att_in_layer(layer.self_attn, c, tk["att_in"])
            w_out, out_s = pack_att_out_layer(layer.self_attn, c, tk["att_out"])
            att_in_list.append(w_in)
            att_out_list.append(w_out)
        else:
            raise ValueError(f"Layer {l}: unknown layer_type={ltype!r}")

        rs = pack_router_and_shared_layer(
            layer.mlp,
            c=c,
            tp_rank=tp_rank,
            tk_router=tk["router"],
            tk_sh13=tk["sh13"],
            tk_sh2=tk["sh2"],
        )
        router_list.append(rs["router"])
        sh13_list.append(rs["sh13"])
        sh2_list.append(rs["sh2"])

        e13_v, e2_v, es_mk, es_8 = pack_expert_views_layer(
            layer.mlp, c=c, tk_e13=tk["e13"], tk_e2=tk["e2"]
        )
        e13_list.append(e13_v)
        e2_list.append(e2_v)
        es_mk_list.append(es_mk)
        es_8_list.append(es_8)

        sm_mk, sm_16 = pack_smalls_layer(
            layer,
            c=c,
            in_s=in_s,
            out_s=out_s,
            sh13_s=rs["sh13_s"],
            sh2_s=rs["sh2_s"],
            sh_gate=rs["sh_gate"],
            final_norm_w=final_norm_w,
            layer_idx=l,
        )
        smalls_mk_list.append(sm_mk)
        smalls_16_list.append(sm_16)

    stack_e = getattr(c, "stack_e", True)
    e13_out: Any = _stack(e13_list) if stack_e else tuple(e13_list)
    e2_out: Any = _stack(e2_list) if stack_e else tuple(e2_list)

    def _stack_gdn(xs: list[torch.Tensor], kind: str) -> torch.Tensor:
        if xs:
            return _stack(xs)
        # An attention-only stack (the MTP draft layer) keeps one unread
        # GatedDeltaNet operand of the tiled shape (`K.weight_shapes`).
        shape, dt = K.weight_shapes(c, tiles)[kind]
        _need(jnp.dtype(dt) == jnp.dtype(jnp.float8_e4m3fn), f"{kind} dtype {dt}")
        return torch.zeros(shape, dtype=torch.float8_e4m3fn, device=final_norm_w.device)

    weights: dict[str, Any] = {
        "lin_in": _stack_gdn(lin_in_list, "lin_in"),
        "lin_out": _stack_gdn(lin_out_list, "lin_out"),
        "att_in": _stack(att_in_list),
        "att_out": _stack(att_out_list),
        "router": _stack(router_list),
        "sh13": _stack(sh13_list),
        "sh2": _stack(sh2_list),
        "e13": e13_out,
        "e2": e2_out,
        "es": _stack(es_mk_list),
        "smalls": _stack(smalls_mk_list),
        "final_norm": _fresh(final_norm_w),
    }

    w_flat: list[torch.Tensor] = []
    for k in K.W_NAMES:
        val = weights[k]
        if isinstance(val, (tuple, list)):
            w_flat.extend(t.detach() for t in val)
        else:
            w_flat.append(val.detach())

    weights["w_flat"] = w_flat
    weights["smalls_16x4096"] = _stack(smalls_16_list)
    weights["es_8x8192"] = _stack(es_8_list)
    return weights


def _slice_or_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Slice or zero-pad 1D tensor `x` to static length `target_len` without dynamic intermediate shapes."""
    x1 = x.reshape(-1)
    n = int(x1.shape[0])
    if n >= target_len:
        return x1[:target_len]
    return F.pad(x1, (0, target_len - n), value=0)


def _slice_or_pad_2d(
    x: torch.Tensor, target_rows: int, target_cols: int
) -> torch.Tensor:
    """Slice or zero-pad 2D tensor `x` to static `(target_rows, target_cols)` without dynamic intermediate shapes."""
    r, c = int(x.shape[0]), int(x.shape[1])
    out = x[: min(r, target_rows), : min(c, target_cols)]
    pad_r = max(0, target_rows - r)
    pad_c = max(0, target_cols - c)
    if pad_r > 0 or pad_c > 0:
        out = F.pad(out, (0, pad_c, 0, pad_r), value=0)
    return out


def _extract_group_mamba_tensors(
    m_md: Any,
    fallback_1d: torch.Tensor,
    fallback_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    """Extract `(state_ids, ckpt_2d, roff, has_roff, use_ckpt_2d)` with static shapes for compiled metadata prep."""
    state_ids_raw = getattr(m_md, "mamba_state_indices", None)
    ckpt_raw = getattr(m_md, "mamba_ckpt_indices", None)
    roff = getattr(m_md, "mamba_slot_read_offsets", None)

    if state_ids_raw is not None:
        s_1d = (
            state_ids_raw[:, 0]
            if state_ids_raw.ndim == 2
            else state_ids_raw.reshape(-1)
        )
    elif ckpt_raw is not None:
        s_1d = ckpt_raw[:, 0] if ckpt_raw.ndim == 2 else ckpt_raw.reshape(-1)
    else:
        s_1d = fallback_1d

    if ckpt_raw is not None and ckpt_raw.ndim == 2:
        c_2d = ckpt_raw
        use_ckpt_2d = True
    else:
        src_1d = ckpt_raw.reshape(-1) if ckpt_raw is not None else s_1d
        c_2d = src_1d.unsqueeze(1)
        use_ckpt_2d = False

    has_roff = roff is not None
    roff_t = roff.reshape(-1) if roff is not None else fallback_1d
    return s_1d, c_2d, roff_t, has_roff, use_ckpt_2d


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_prepare_step_inputs(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compiled on-device metadata + input_ids pack with zero `num_reqs` specialization."""
    T = tokens_per_seq
    B = bucket_t
    nseq_bkt = B // T

    bt = block_tables
    if bt.ndim == 1:
        if bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            bt = bt.reshape(int(s0.shape[0]), -1)
    pt_b = _slice_or_pad_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    for s_raw, c_raw, r_raw in ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)):
        s_b = _slice_or_pad_1d(s_raw, nseq_bkt).to(torch.int32)
        if has_roff:
            if r_raw.numel() > s_raw.shape[0]:
                s_idx = s_b.to(torch.long).clamp(min=0, max=r_raw.numel() - 1)
                roff_g = r_raw[s_idx].to(torch.int32)
            else:
                roff_g = _slice_or_pad_1d(r_raw, nseq_bkt).to(torch.int32)
        else:
            roff_g = torch.zeros_like(s_b)

        if T > 1:
            if use_ckpt_2d and c_raw.shape[1] >= T:
                ckpt_g = _slice_or_pad_2d(c_raw, nseq_bkt, T).to(torch.int32)
            else:
                ckpt_1d = _slice_or_pad_1d(c_raw[:, 0], nseq_bkt).to(torch.int32)
                ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T)
                roff_g = torch.zeros_like(roff_g)
        else:
            if use_ckpt_2d and has_roff and c_raw.shape[1] > 1:
                c_b = _slice_or_pad_2d(c_raw, nseq_bkt, int(c_raw.shape[1])).to(
                    torch.int32
                )
                r_idx = roff_g.to(torch.long).clamp(min=0, max=c_raw.shape[1] - 1)
                ckpt_g = c_b.gather(1, r_idx.unsqueeze(1))
            else:
                ckpt_g = s_b.unsqueeze(1)
            roff_g = torch.zeros_like(roff_g)

        active_g = ckpt_g[:, 0] > 0
        roff_g = torch.where(active_g, roff_g, torch.zeros_like(roff_g))
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    ckpt_b = torch.stack(ckpt_rows, dim=0)
    roff_b = torch.stack(roff_rows, dim=0)
    active_seq = ckpt_b[0, :, 0] > 0
    active_tok = active_seq.unsqueeze(1).expand(nseq_bkt, T).reshape(B)
    zero_b = torch.zeros((B,), dtype=torch.int32, device=input_ids.device)
    ids_b = torch.where(
        active_tok, _slice_or_pad_1d(input_ids, B).to(torch.int32), zero_b
    )
    pos_b = torch.where(
        active_tok, _slice_or_pad_1d(positions, B).to(torch.int32), zero_b
    )
    return ids_b, pos_b, pt_b, ckpt_b, roff_b


def prepare_step_metadata(
    attn_md: Any,
    mamba_mds: Sequence[Any],
    num_reqs: int,
    bucket_t: int,
    max_pages_per_seq: int,
    block_major_fn: Any = None,
    *,
    positions: torch.Tensor | None = None,
    tokens_per_seq: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `(pos, page_table, ckpt_indices, read_offsets)` padded to `bucket_t` rows.

    Args:
      attn_md: Full-attention `AttentionMetadata` (`seq_lens`, `block_tables`).
      mamba_mds: 3-element sequence of Mamba group `AttentionMetadata`
        (`mamba_state_indices`, `mamba_ckpt_indices`, `mamba_slot_read_offsets`).
      num_reqs: Number of active requests (`1 .. nseq_bkt`).
      bucket_t: Compiled row bucket `B` (`nseq_bkt * tokens_per_seq`).
      max_pages_per_seq: Maximum kernel pages per sequence (`2050`).
      block_major_fn: Unused legacy argument kept for caller compatibility.
      positions: Optional on-device `position_ids` (`[num_tokens]`, already
        adjusted by `_compiled_subtract_num_rejected_tokens` under async spec).
      tokens_per_seq: `T = 1` for 1-token decode or `T = 4` for MTP=3 verify.

    Returns:
      - `pos`: `int32[B]` token positions (`0` for padded rows).
      - `page_table`: `int32[nseq_bkt, max_pages_per_seq]` kernel page table (`0` for padded seqs).
      - `ckpt_indices`: `int32[3, nseq_bkt, T]` manager block IDs (`0` for padded seqs).
      - `read_offsets`: `int32[3, nseq_bkt]` checkpoint read offsets in `0..T-1` (`0` for padded seqs).
    """
    _ = block_major_fn
    T = int(tokens_per_seq)
    B = int(bucket_t)
    _need(T >= 1 and B % T == 0, f"bucket_t={B} must be divisible by tokens_per_seq={T}")
    nseq_bkt = B // T
    active_rows = num_reqs * T
    dev = attn_md.seq_lens.device
    active_seq_mask = (
        torch.arange(nseq_bkt, dtype=torch.int32, device=dev) < num_reqs
    )
    active_tok_mask = (
        torch.arange(B, dtype=torch.int32, device=dev) < active_rows
    )

    seq_lens_b = _slice_or_pad_1d(attn_md.seq_lens, nseq_bkt).to(torch.int32)
    if positions is not None and positions.numel() >= active_rows:
        pos_b = _slice_or_pad_1d(positions, B).to(torch.int32)
    elif T == 1:
        pos_b = (seq_lens_b - 1).clamp(min=0)
    else:
        base_pos = (seq_lens_b - T).clamp(min=0).unsqueeze(1)
        offs = torch.arange(T, dtype=torch.int32, device=dev).unsqueeze(0)
        pos_b = (base_pos + offs).reshape(B)
    pos = torch.where(
        active_tok_mask, pos_b, torch.zeros_like(pos_b)
    ).contiguous()

    bt = attn_md.block_tables
    if bt.ndim == 1:
        if bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            bt = bt.reshape(int(attn_md.seq_lens.shape[0]), -1)
    bt_b = _slice_or_pad_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)
    page_table = torch.where(
        active_seq_mask.unsqueeze(1), bt_b, torch.zeros_like(bt_b)
    ).contiguous()

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    zero_seq = torch.zeros((nseq_bkt,), dtype=torch.int32, device=dev)
    for g in range(3):
        m_md = mamba_mds[g]
        state_ids_raw = getattr(m_md, "mamba_state_indices", None)
        ckpt_raw = getattr(m_md, "mamba_ckpt_indices", None)
        roff = getattr(m_md, "mamba_slot_read_offsets", None)

        if roff is not None:
            if state_ids_raw is not None and roff.numel() > state_ids_raw.shape[0]:
                s_idx = (
                    _slice_or_pad_1d(state_ids_raw, nseq_bkt)
                    .to(torch.long)
                    .clamp(min=0, max=roff.numel() - 1)
                )
                roff_g = roff.reshape(-1)[s_idx].to(torch.int32)
            else:
                roff_g = _slice_or_pad_1d(roff, nseq_bkt).to(torch.int32)
        else:
            roff_g = zero_seq

        if T > 1:
            if ckpt_raw is not None and ckpt_raw.ndim == 2 and ckpt_raw.shape[1] >= T:
                ckpt_g = _slice_or_pad_2d(ckpt_raw, nseq_bkt, T).to(torch.int32)
            else:
                src = ckpt_raw if ckpt_raw is not None else state_ids_raw
                if src.ndim == 2:
                    src = src[:, 0]
                ckpt_1d = _slice_or_pad_1d(src, nseq_bkt).to(torch.int32).unsqueeze(1)
                ckpt_g = ckpt_1d.expand(nseq_bkt, T)
                roff_g = zero_seq
        else:
            if (
                ckpt_raw is not None
                and ckpt_raw.ndim == 2
                and roff is not None
                and ckpt_raw.shape[1] > 1
            ):
                r_idx = roff_g.to(torch.long).clamp(min=0, max=ckpt_raw.shape[1] - 1)
                ckpt_b_all = _slice_or_pad_2d(
                    ckpt_raw, nseq_bkt, int(ckpt_raw.shape[1])
                ).to(torch.int32)
                ckpt_g = ckpt_b_all.gather(1, r_idx.unsqueeze(1))
            else:
                src = state_ids_raw if state_ids_raw is not None else ckpt_raw
                if src.ndim == 2:
                    src = src[:, 0]
                ckpt_g = _slice_or_pad_1d(src, nseq_bkt).to(torch.int32).unsqueeze(1)
            roff_g = zero_seq

        ckpt_g = torch.where(
            active_seq_mask.unsqueeze(1), ckpt_g, torch.zeros_like(ckpt_g)
        )
        roff_g = torch.where(active_seq_mask, roff_g, zero_seq)
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    ckpt_indices = torch.stack(ckpt_rows, dim=0).contiguous()
    read_offsets = torch.stack(roff_rows, dim=0).contiguous()
    return pos, page_table, ckpt_indices, read_offsets


# ---------------------------------------------------------------------------
# Main vLLM Adapter Class
# ---------------------------------------------------------------------------


class Qwen35Megakernel:
    """Whole-backbone decode/verify megakernel driver for `Qwen3_5MoeForCausalLM`."""

    @classmethod
    def maybe_create(cls, runner: Any) -> Qwen35Megakernel | None:
        if os.environ.get("VLLM_TPU_QWEN35_MEGAKERNEL", "0") != "1":
            return None
        try:
            backbone = _find_backbone(runner.model)
        except ValueError as exc:
            logger.warning("Qwen3.5 megakernel disabled: %s", exc)
            return None

        if len(backbone.layers) != K.FULL.L:
            logger.warning(
                "Qwen3.5 megakernel disabled: layer count %d != %d",
                len(backbone.layers),
                K.FULL.L,
            )
            return None

        use_unified_attr = getattr(runner, "_unified_kv_layout", None)
        if use_unified_attr is not None:
            use_unified = bool(use_unified_attr)
        else:
            fn = getattr(runner, "_use_unified_kv_cache", None)
            use_unified = bool(fn() if callable(fn) else True)
        if not use_unified:
            logger.warning(
                "Qwen3.5 megakernel disabled: unified KV cache is not enabled"
            )
            return None

        mode = os.environ.get("VLLM_TPU_QWEN35_MK_MODE", "ep").strip().lower()
        spec_cfg = getattr(runner, "speculative_config", None)
        num_spec = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
        if num_spec == 3:
            verify_buckets = parse_mk_verify_buckets()
            decode_default = os.environ.get("VLLM_TPU_QWEN35_MK_BUCKETS", "0")
            buckets = parse_mk_buckets(decode_default)
        else:
            verify_buckets = (
                parse_mk_verify_buckets()
                if "VLLM_TPU_QWEN35_MK_VERIFY_BUCKETS" in os.environ
                else ()
            )
            buckets = parse_mk_buckets()
        return cls(
            runner,
            mode=mode,
            buckets=buckets,
            verify_buckets=verify_buckets,
        )

    def __init__(
        self,
        runner: Any,
        *,
        mode: str = "ep",
        buckets: Sequence[int] = _DEFAULT_BUCKETS,
        verify_buckets: Sequence[int] = (),
        mesh: jax.sharding.Mesh | None = None,
        fused_pool_io: bool | None = None,
    ) -> None:
        _need(mode in ("ep", "tp"), f"unsupported VLLM_TPU_QWEN35_MK_MODE={mode!r}")
        self.mode = mode
        self.buckets = tuple(sorted(buckets))
        self.verify_buckets = tuple(sorted(verify_buckets))
        self.max_bucket = max(self.buckets) if self.buckets else 0
        self.max_verify_bucket = max(self.verify_buckets) if self.verify_buckets else 0
        self.fused_pool_io = (
            parse_mk_fused_pool_io()
            if fused_pool_io is None
            else bool(fused_pool_io)
        )
        self._hlo_checked = False
        self.backbone = _find_backbone(runner.model)
        self.embed_tokens = self.backbone.embed_tokens

        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        _need(
            self.tp_size == K.FULL.R,
            f"tp_size={self.tp_size} != expected {K.FULL.R}",
        )

        if mesh is None:
            mesh = build_ep_mesh(axis_name="tp")
        if mesh is None:
            devices = jax.devices()[: self.tp_size]
            mesh = jax.sharding.Mesh(devices, ("tp",))
        self.mesh = mesh

        with torch.no_grad():
            if self.mode == "tp":
                logger.info(
                    "Qwen3.5 megakernel: resharding 60 MoE layers from EP to TP in-place"
                )
                for l in range(K.FULL.L):
                    layer_l = self.backbone.layers[l]
                    reshard_layer_experts_ep_to_tp(
                        layer_l.mlp,
                        tp_rank=self.tp_rank,
                        c=K.FULL,
                    )
                    _try_set(layer_l, "use_attn_reduce_scatter_for_moe", False)
                    linear_attn = getattr(layer_l, "linear_attn", None)
                    if linear_attn is not None:
                        _try_set(
                            getattr(linear_attn, "out_proj", None),
                            "reduce_results",
                            True,
                        )
                    self_attn = getattr(layer_l, "self_attn", None)
                    if self_attn is not None:
                        _try_set(
                            getattr(self_attn, "o_proj", None),
                            "reduce_results",
                            True,
                        )
                    _try_set(layer_l.mlp, "is_sequence_parallel", False)
                # Qwen3_5Model.use_sequence_parallel is a read-only property
                # derived from the layer flags above; only set it if writable.
                _try_set(self.backbone, "use_sequence_parallel", False)
                self.e_slots = 0
                self.base_cfg = dataclasses.replace(
                    K.FULL,
                    moe_tp=True,
                    stack_e=False,
                    e_slots=0,
                    paged_kv=True,
                    fused_pool_io=self.fused_pool_io,
                )
            else:
                e_slots = int(
                    _get_routed_experts(self.backbone.layers[0].mlp).w13_weight.shape[0]
                )
                self.e_slots = e_slots
                self.base_cfg = dataclasses.replace(
                    K.FULL_EP,
                    moe_tp=False,
                    stack_e=False,
                    e_slots=e_slots,
                    paged_kv=True,
                    fused_pool_io=self.fused_pool_io,
                )

            packed = pack_weights(
                self.backbone, self.base_cfg, tp_rank=self.tp_rank
            )
            self.weights_dict = packed
            self.w_flat: list[torch.Tensor] = packed["w_flat"]
            self.smalls_16x4096: torch.Tensor = packed["smalls_16x4096"]
            self.es_8x8192: torch.Tensor = packed["es_8x8192"]

        if self.w_flat and self.w_flat[0].device.type == "tpu":
            from vllm_torchtpu.utils import synchronize_tensors

            synchronize_tensors(self.w_flat)

        self.raw_pools: list[torch.Tensor] = []
        self.mamba_layer_names: list[str] = []
        self.attn_layer_name: str = ""
        self.num_pages: int = 0
        self.page_size: int = 128
        self.max_pages_per_seq: int = 0
        self._bucket_steps: dict[Any, Any] = {}
        self._bucket_out_rows: dict[tuple[int, int], int] = {}
        self._bucket_ops: dict[Any, Any] = {}
        self._bound = False
        self._zero_roff_1d: torch.Tensor | None = None
        self._ckpt_window: int = 1
        self._prepared_step: tuple[Any, ...] | None = None
        self._step_mix: dict[str, int] = {}
        self._step_mix_t0 = time.monotonic()
        self._step_mix_period_s = float(
            os.environ.get("VLLM_TPU_QWEN35_MK_STATS_PERIOD_S", "60")
        )
        self.ab_enabled = parse_mk_ab_enabled()
        self.ab_calibrator = MKABCalibrator(
            enabled=self.ab_enabled,
            period_s=self._step_mix_period_s,
        )

        logger.info(
            "Qwen3.5 megakernel initialized: mode=%s, e_slots=%d, decode_buckets=%s, "
            "verify_buckets=%s, fused_pool_io=%s, ab_enabled=%s, %d weight tensors (tp_rank=%d/%d)",
            self.mode,
            self.base_cfg.EW,
            self.buckets,
            self.verify_buckets,
            self.fused_pool_io,
            self.ab_enabled,
            len(self.w_flat),
            self.tp_rank,
            self.tp_size,
        )

    def bind_caches(self, runner: Any) -> None:
        """Bind the 15 target UnifiedKVCache raw pool tensors and build bucket ops."""
        raw_list = getattr(runner, "kv_cache_raw_tensors", None)
        pools: list[torch.Tensor] = []
        if raw_list is not None and len(raw_list) >= K.FULL.NATT:
            pools = [t for t in raw_list[: K.FULL.NATT]]
        else:
            for r in range(K.FULL.NATT):
                attn_mod = self.backbone.layers[4 * r + 3].self_attn.attn
                pools.append(_unwrap(attn_mod.kv_cache))

        _need(
            len(pools) == K.FULL.NATT,
            f"expected {K.FULL.NATT} raw pool tensors, got {len(pools)}",
        )
        num_pages = int(pools[0].shape[0])
        for r, pool in enumerate(pools):
            _need(
                isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
                f"pool[{r}] must be float8_e4m3fn tensor, got {type(pool)} {getattr(pool, 'dtype', None)}",
            )
            _need(
                pool.storage_offset() == 0,
                f"pool[{r}] has non-zero storage_offset={pool.storage_offset()}",
            )
            _need(
                tuple(pool.shape) == (num_pages, 2, 64, 4, 128),
                f"pool[{r}] shape {tuple(pool.shape)} != ({num_pages}, 2, 64, 4, 128)",
            )
            for g in range(3):
                la_cache = _unwrap(self.backbone.layers[4 * r + g].linear_attn.kv_cache)
                if isinstance(la_cache, torch.Tensor) and la_cache.numel() > 0:
                    _need(
                        la_cache.data_ptr() == pool.data_ptr(),
                        f"region {r} group {g} linear_attn.kv_cache does not alias pool[{r}]",
                    )

        self.raw_pools = pools
        self.num_pages = num_pages
        self.page_size = int(pools[0].shape[-1])
        self._zero_roff_1d = torch.zeros(
            (1,), dtype=torch.int32, device=pools[0].device
        )
        k_spec = (
            int(runner.speculative_config.num_speculative_tokens)
            if getattr(runner, "speculative_config", None) is not None
            else 0
        )
        self._ckpt_window = max(
            1, int(getattr(runner, "_mamba_ckpt_window", k_spec + 1))
        )
        attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
        if callable(attn_shapes_fn):
            _, max_pages_per_seq, _ = attn_shapes_fn(True)
            self.max_pages_per_seq = int(max_pages_per_seq)
        else:
            self.max_pages_per_seq = int(
                getattr(runner, "max_num_blocks_per_req", 2050)
            )
        self.mamba_layer_names = [
            self.backbone.layers[g].linear_attn.prefix for g in range(3)
        ]
        self.attn_layer_name = self.backbone.layers[3].self_attn.attn.layer_name

        self.base_cfg = dataclasses.replace(
            self.base_cfg,
            ctx=max(self.base_cfg.ctx, self.max_pages_per_seq * self.page_size),
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=self.max_pages_per_seq,
            paged_kv=True,
        )

        runner_token_paddings = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )

        def _runner_out_rows(b: int, m_rows: int) -> int:
            # The runner pads a b-row step to its first token bucket >= b and
            # takes back exactly that many hidden rows, which may be fewer
            # than the kernel's 16-row-aligned m_rows (e.g. 24 vs 32).
            for p in runner_token_paddings:
                if p >= b:
                    return p
            return m_rows

        self._runner_token_paddings = runner_token_paddings
        self._build_all_buckets()

        self._bound = True
        logger.info(
            "Qwen3.5 megakernel bound to %d unified pools (num_pages=%d, "
            "page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, verify_buckets=%s, fused_pool_io=%s)",
            len(self.raw_pools),
            self.num_pages,
            self.page_size,
            self.max_pages_per_seq,
            self.buckets,
            self.verify_buckets,
            self.fused_pool_io,
        )

    def _runner_out_rows(self, b: int, m_rows: int) -> int:
        # The runner pads a b-row step to its first token bucket >= b and
        # takes back exactly that many hidden rows, which may be fewer
        # than the kernel's 16-row-aligned m_rows (e.g. 24 vs 32).
        for p in getattr(self, "_runner_token_paddings", ()):
            if p >= b:
                return p
        return m_rows

    def _build_all_buckets(self) -> None:
        self._bucket_steps.clear()
        self._bucket_ops.clear()
        self._bucket_out_rows.clear()
        for b in self.buckets:
            m_rows = max(16, K.rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            step_1 = self._build_bucket_op(
                b, tokens_per_seq=1, out_pad_rows=out_rows
            )
            self._bucket_steps[(1, b)] = step_1
            self._bucket_steps[b] = step_1
            self._bucket_out_rows[(1, b)] = out_rows
        for b in self.verify_buckets:
            m_rows = max(16, K.rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            self._bucket_steps[(4, b)] = self._build_bucket_op(
                b, tokens_per_seq=4, out_pad_rows=out_rows
            )
            self._bucket_out_rows[(4, b)] = out_rows

    def _verify_fused_pool_hlo(
        self,
        step_fn: Any,
        in_specs: Sequence[P],
        out_specs: Sequence[P],
        donate_argnums: Sequence[int],
        c: K.MKConfig,
    ) -> tuple[bool, str]:
        """Verify at lower/compile time that fused pool I/O feeds raw pools directly to tpu_custom_call without S(1) copies."""
        try:
            from jax.sharding import NamedSharding

            T = int(c.T)
            B = int(c.B)
            nseq = B // T
            r_cnt = int(c.R)
            m_rows = int(c.M)
            max_pages = int(self.max_pages_per_seq)
            in_sh = tuple(NamedSharding(self.mesh, s) for s in in_specs)
            out_sh = tuple(NamedSharding(self.mesh, s) for s in out_specs)

            def _global_sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
                dt = (
                    jnp.float8_e4m3fn
                    if t.dtype == torch.float8_e4m3fn
                    else jnp.bfloat16
                    if t.dtype == torch.bfloat16
                    else jnp.int32
                    if t.dtype == torch.int32
                    else jnp.float32
                )
                return jax.ShapeDtypeStruct(
                    (r_cnt * int(t.shape[0]),) + tuple(t.shape[1:]), dt
                )

            avals = [
                jax.ShapeDtypeStruct((r_cnt * B,), jnp.int32),
                jax.ShapeDtypeStruct((r_cnt * m_rows, int(c.H)), jnp.bfloat16),
                *(_global_sds(w) for w in self.w_flat),
                *(_global_sds(p) for p in self.raw_pools),
                jax.ShapeDtypeStruct((r_cnt * nseq, max_pages), jnp.int32),
                jax.ShapeDtypeStruct((r_cnt * 3, nseq, T), jnp.int32),
                jax.ShapeDtypeStruct((r_cnt * 3, nseq), jnp.int32),
            ]
            j_fn = jax.jit(
                step_fn,
                in_shardings=in_sh,
                out_shardings=out_sh,
                donate_argnums=donate_argnums,
                keep_unused=True,
            )
            lowered = j_fn.lower(*avals)
            if os.environ.get("VLLM_TPU_QWEN35_MK_COMPILE_HLO_CHECK", "0") == "1":
                hlo_text = lowered.compile().as_text()
            else:
                hlo_text = lowered.as_text("hlo")
            return pool_io.check_fused_pool_hlo(
                hlo_text,
                num_pools=len(self.raw_pools),
                page_size=self.page_size,
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"HLO check exception: {exc}"

    def _build_bucket_op(
        self,
        bucket_t: int,
        tokens_per_seq: int = 1,
        out_pad_rows: int | None = None,
    ) -> Any:
        """Build the `sharded_jax_callable` custom op and fused `torch.compile` step for `(tokens_per_seq, bucket_t)`."""
        bucket_fused = bool(self.fused_pool_io and bucket_t <= 32)
        c = dataclasses.replace(
            self.base_cfg,
            B=bucket_t,
            T=tokens_per_seq,
            fused_pool_io=bucket_fused,
        )
        tiles = K.default_tiles(c)

        if hasattr(pool_io, "make_flat_pooled_step"):
            step_fn, in_specs, out_specs, donate_argnums = (
                pool_io.make_flat_pooled_step(
                    self.mesh, c, tiles, fused_pool_io=bucket_fused
                )
            )
            if bucket_fused and not self._hlo_checked:
                self._hlo_checked = True
                ok_hlo, reason_hlo = self._verify_fused_pool_hlo(
                    step_fn, in_specs, out_specs, donate_argnums, c
                )
                if not ok_hlo:
                    logger.warning(
                        "Qwen3.5 megakernel: fused_pool_io HLO check failed (%s); "
                        "falling back to 3-stage pool I/O",
                        reason_hlo,
                    )
                    self.fused_pool_io = False
                    self.base_cfg = dataclasses.replace(
                        self.base_cfg, fused_pool_io=False
                    )
                    c = dataclasses.replace(c, fused_pool_io=False)
                    step_fn, in_specs, out_specs, donate_argnums = (
                        pool_io.make_flat_pooled_step(
                            self.mesh, c, tiles, fused_pool_io=False
                        )
                    )
                else:
                    logger.info(
                        "Qwen3.5 megakernel: fused_pool_io HLO check passed (%s)",
                        reason_hlo,
                    )
        else:
            built = pool_io.make_pooled_step(self.mesh, c, tiles)
            if isinstance(built, tuple) and len(built) == 4:
                step_fn, in_specs, out_specs, donate_argnums = built
            else:
                step_fn = built
                n_w = len(self.w_flat)
                n_p = len(self.raw_pools)
                in_specs = (P("tp"),) * (2 + n_w + n_p + 3)
                out_specs = (P("tp"),) * (1 + n_p)
                donate_argnums = tuple(range(2 + n_w, 2 + n_w + n_p))

        n_p = len(self.raw_pools)
        n_in = len(in_specs)
        args = [f"x{i}" for i in range(n_in)]
        ret = "tuple[" + ", ".join(["jax.Array"] * len(out_specs)) + "]"
        io_tag = "" if self.fused_pool_io else "_3stage"
        fn_name = f"qwen35_mk_step_t{tokens_per_seq}_b{bucket_t}_{self.mode}{io_tag}"
        src = (
            f"def {fn_name}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
            f"    return _step_fn({', '.join(args)})\n"
        )
        ns: dict[str, Any] = {"jax": jax, "_step_fn": step_fn}
        exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102

        op_name = (
            f"qwen35_megakernel::step_t{tokens_per_seq}_b{bucket_t}_{self.mode}{io_tag}"
        )
        flat_callable = sharded_jax_callable(
            op_name,
            ns[fn_name],
            mesh=self.mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=donate_argnums,
        )

        # Wrap in a torch custom_op taking list[Tensor] for weights and pools
        # to stay well under PyTorch dispatcher's 64-argument schema cap.
        @torch.library.custom_op(op_name, mutates_args=())
        def op(
            pos: torch.Tensor,
            hidden: torch.Tensor,
            page_table: torch.Tensor,
            ckpt_indices: torch.Tensor,
            read_offsets: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return list(
                flat_callable(
                    pos,
                    hidden,
                    *weights,
                    *pools,
                    page_table,
                    ckpt_indices,
                    read_offsets,
                )
            )

        @op.register_fake
        def _(
            pos: torch.Tensor,
            hidden: torch.Tensor,
            page_table: torch.Tensor,
            ckpt_indices: torch.Tensor,
            read_offsets: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            return [torch.empty_like(hidden)] + [torch.empty_like(p) for p in pools]

        self._bucket_ops[(tokens_per_seq, bucket_t)] = op
        if tokens_per_seq == 1:
            self._bucket_ops[bucket_t] = op
        T = int(tokens_per_seq)
        B = int(bucket_t)
        nseq_bkt = B // T
        max_pages = int(self.max_pages_per_seq)
        m_rows = c.M
        target_out_rows = int(out_pad_rows or m_rows)

        def step(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            block_tables: torch.Tensor,
            s0: torch.Tensor,
            c0: torch.Tensor,
            s1: torch.Tensor,
            c1: torch.Tensor,
            s2: torch.Tensor,
            c2: torch.Tensor,
            roff: torch.Tensor,
            weights: list[torch.Tensor],
            pools: list[torch.Tensor],
        ) -> torch.Tensor:
            bt = block_tables
            if bt.ndim == 1:
                if bt.numel() % max_pages == 0:
                    bt = bt.reshape(-1, max_pages)
                else:
                    bt = bt.reshape(int(s0.shape[0]), -1)
            pt_b = _slice_or_pad_2d(bt, nseq_bkt, max_pages).to(torch.int32)

            roff_1d = roff.reshape(-1)
            has_roff = roff_1d.numel() > 1
            ckpt_rows: list[torch.Tensor] = []
            roff_rows: list[torch.Tensor] = []
            for s_raw, c_raw in ((s0, c0), (s1, c1), (s2, c2)):
                s_1d = s_raw[:, 0] if s_raw.ndim == 2 else s_raw.reshape(-1)
                s_b = _slice_or_pad_1d(s_1d, nseq_bkt).to(torch.int32)
                if has_roff:
                    if roff_1d.numel() > s_1d.shape[0]:
                        s_idx = s_b.to(torch.long).clamp(
                            min=0, max=roff_1d.numel() - 1
                        )
                        roff_g = roff_1d[s_idx].to(torch.int32)
                    else:
                        roff_g = _slice_or_pad_1d(roff_1d, nseq_bkt).to(torch.int32)
                else:
                    roff_g = torch.zeros_like(s_b)

                use_ckpt_2d = c_raw.ndim == 2
                if T > 1:
                    if use_ckpt_2d and c_raw.shape[1] >= T:
                        ckpt_g = _slice_or_pad_2d(c_raw, nseq_bkt, T).to(torch.int32)
                    else:
                        c_1d = c_raw[:, 0] if use_ckpt_2d else c_raw.reshape(-1)
                        ckpt_1d = _slice_or_pad_1d(c_1d, nseq_bkt).to(torch.int32)
                        ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T)
                        roff_g = torch.zeros_like(roff_g)
                else:
                    if use_ckpt_2d and has_roff and c_raw.shape[1] > 1:
                        c_b = _slice_or_pad_2d(
                            c_raw, nseq_bkt, int(c_raw.shape[1])
                        ).to(torch.int32)
                        r_idx = roff_g.to(torch.long).clamp(
                            min=0, max=c_raw.shape[1] - 1
                        )
                        ckpt_g = c_b.gather(1, r_idx.unsqueeze(1))
                    else:
                        ckpt_g = s_b.unsqueeze(1)
                    roff_g = torch.zeros_like(roff_g)

                active_g = ckpt_g[:, 0] > 0
                roff_g = torch.where(active_g, roff_g, torch.zeros_like(roff_g))
                ckpt_rows.append(ckpt_g)
                roff_rows.append(roff_g)

            ckpt_b = torch.stack(ckpt_rows, dim=0)
            roff_b = torch.stack(roff_rows, dim=0)
            active_seq = ckpt_b[0, :, 0] > 0
            active_tok = active_seq.unsqueeze(1).expand(nseq_bkt, T).reshape(B)
            zero_b = torch.zeros((B,), dtype=torch.int32, device=input_ids.device)
            ids_b = torch.where(
                active_tok, _slice_or_pad_1d(input_ids, B).to(torch.int32), zero_b
            )
            pos_1d = positions[0] if positions.ndim == 2 else positions
            pos_b = torch.where(
                active_tok, _slice_or_pad_1d(pos_1d, B).to(torch.int32), zero_b
            )

            h0 = self.embed_tokens(ids_b)
            if h0.shape[0] < m_rows:
                h0 = F.pad(h0, (0, 0, 0, m_rows - h0.shape[0]))
            outs = op(
                pos_b,
                h0,
                pt_b,
                ckpt_b,
                roff_b,
                list(weights),
                list(pools),
            )
            for dst, src in zip(pools, outs[1 : 1 + n_p]):
                dst.copy_(src)
            h_out = outs[0]
            if h_out.shape[0] > target_out_rows:
                # clone(): an output aliasing an intermediate is regenerated as
                # a view outside the compiled program.
                h_out = h_out[:target_out_rows].clone()
            elif h_out.shape[0] < target_out_rows:
                h_out = F.pad(h_out, (0, 0, 0, target_out_rows - h_out.shape[0]))
            return h_out

        return torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)

    def reset_stats(self) -> None:
        """Reset step-mix counters and A/B latency accumulators (e.g. after warmup)."""
        self._step_mix.clear()
        self._step_mix_t0 = time.monotonic()
        self.ab_calibrator.reset()

    def promote_fast_callables(self) -> int:
        """Replace `_qx_fastcall` wrappers in `_bucket_steps` with direct AOT callables when `QX_AOT_SKIP_GUARD_CHECK=1`."""
        self.reset_stats()
        if os.environ.get("QX_AOT_SKIP_GUARD_CHECK", "1") != "1":
            return 0
        promoted = 0
        for key, fn in list(self._bucket_steps.items()):
            st = getattr(fn, "_qx_fast_state", None)
            if isinstance(st, dict):
                entries = st.get("entries")
                entry = st.get("entry")
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 3
                    and callable(entry[1])
                    and (not isinstance(entries, list) or len(entries) <= 1)
                ):
                    self._bucket_steps[key] = entry[1]
                    promoted += 1
        if promoted > 0:
            logger.info(
                "Qwen3.5 megakernel: promoted %d bucket step callables to zero-guard direct dispatch",
                promoted,
            )
        return promoted

    def warmup(self, runner: Any) -> None:
        """Pre-compile and warm up all configured decode and MTP-verify buckets."""
        if not self._bound:
            return
        dev = self.raw_pools[0].device
        all_keys = [(1, b) for b in self.buckets] + [(4, b) for b in self.verify_buckets]
        if not all_keys:
            return
        logger.info("Qwen3.5 megakernel: warming up (T, B) buckets %s ...", all_keys)
        max_t = max(T for T, _ in all_keys)
        n_save = min(self.num_pages, max(5, 5 * (max_t + 1)))
        # Never slice-clone or slice-write raw_pools on TPU (PyTorch/XLA lowers
        # slice writes on a 2.30 GiB pool to as_strided_copy_inverse, allocating
        # a full 2.30 GiB temporary buffer per pool).
        saved_pages = (
            [p[:n_save].clone() for p in self.raw_pools]
            if dev.type != "tpu"
            else None
        )
        try:
            with torch.no_grad():
                runner_tok_pads = tuple(
                    int(x) for x in getattr(runner, "num_tokens_paddings", ())
                )
                runner_req_pads = tuple(
                    sorted(
                        {
                            int(x)
                            for x in (
                                getattr(runner, "num_reqs_max_model_len", None),
                                getattr(runner, "num_reqs_most_model_len", None),
                            )
                            if x is not None
                        }
                    )
                )
                roff_buf = getattr(runner, "mamba_slot_read_offsets", None)
                has_roff = isinstance(roff_buf, torch.Tensor)
                r_raw = roff_buf if has_roff else self._zero_roff_1d
                k_spec = (
                    int(runner.speculative_config.num_speculative_tokens)
                    if getattr(runner, "speculative_config", None) is not None
                    else 0
                )
                ckpt_cols = max(1, k_spec + 1)
                uses_mrope = bool(getattr(runner, "uses_mrope", False))
                sentinel_checked = False

                while True:
                    fallback_needed = False
                    for T, b in all_keys:
                        step_fn = (
                            self._bucket_steps.get((T, b))
                            or self._bucket_steps.get(b)
                        )
                        do_sentinel = (
                            self.fused_pool_io
                            and not sentinel_checked
                            and self.num_pages >= 5 * (T + 1)
                        )
                        if dev.type == "tpu" and runner_tok_pads and runner_req_pads:
                            tok_pad = next(
                                (p for p in runner_tok_pads if p >= b),
                                runner_tok_pads[-1],
                            )
                            pos_shape = (3, tok_pad) if uses_mrope else (tok_pad,)
                            for req_pad in runner_req_pads:
                                ids_raw = torch.zeros(
                                    (tok_pad,), dtype=torch.int32, device=dev
                                )
                                pos_raw = torch.zeros(
                                    pos_shape, dtype=torch.int32, device=dev
                                )
                                bt_raw_1d = torch.zeros(
                                    (req_pad * self.max_pages_per_seq,),
                                    dtype=torch.int32,
                                    device=dev,
                                )
                                s0_raw = torch.zeros(
                                    (req_pad,), dtype=torch.int32, device=dev
                                )
                                s1_raw = torch.zeros(
                                    (req_pad,), dtype=torch.int32, device=dev
                                )
                                s2_raw = torch.zeros(
                                    (req_pad,), dtype=torch.int32, device=dev
                                )
                                if ckpt_cols > 1:
                                    c0_raw = torch.zeros(
                                        (req_pad, ckpt_cols),
                                        dtype=torch.int32,
                                        device=dev,
                                    )
                                    c1_raw = torch.zeros(
                                        (req_pad, ckpt_cols),
                                        dtype=torch.int32,
                                        device=dev,
                                    )
                                    c2_raw = torch.zeros(
                                        (req_pad, ckpt_cols),
                                        dtype=torch.int32,
                                        device=dev,
                                    )
                                else:
                                    c0_raw, c1_raw, c2_raw = s0_raw, s1_raw, s2_raw

                                r_call = r_raw
                                if do_sentinel:
                                    sent_kv = (
                                        self.raw_pools[0][0]
                                        .view(torch.uint8)
                                        .cpu()
                                        .clone()
                                    )
                                    sent_gdn = (
                                        self.raw_pools[0][5:9]
                                        .view(torch.uint8)
                                        .cpu()
                                        .clone()
                                    )
                                    ids_raw[0] = 1
                                    s0_raw[0] = 1
                                    s1_raw[0] = 1
                                    s2_raw[0] = 1
                                    if ckpt_cols > 1:
                                        blk_ids = torch.arange(
                                            1,
                                            1 + min(T, ckpt_cols),
                                            dtype=torch.int32,
                                            device=dev,
                                        )
                                        c0_raw[0, : blk_ids.numel()] = blk_ids
                                        c1_raw[0, : blk_ids.numel()] = blk_ids
                                        c2_raw[0, : blk_ids.numel()] = blk_ids
                                    if r_raw.numel() > 1:
                                        r_call = torch.full_like(r_raw, -1)

                                out_b = step_fn(
                                    ids_raw,
                                    pos_raw,
                                    bt_raw_1d,
                                    s0_raw,
                                    c0_raw,
                                    s1_raw,
                                    c1_raw,
                                    s2_raw,
                                    c2_raw,
                                    r_call,
                                    self.w_flat,
                                    self.raw_pools,
                                )
                                from vllm_torchtpu.utils import synchronize_tensors

                                synchronize_tensors([out_b, *self.raw_pools])
                                if do_sentinel:
                                    do_sentinel = False
                                    sentinel_checked = True
                                    kv_ok = not torch.equal(
                                        self.raw_pools[0][0]
                                        .view(torch.uint8)
                                        .cpu(),
                                        sent_kv,
                                    )
                                    gdn_ok = not torch.equal(
                                        self.raw_pools[0][5:9]
                                        .view(torch.uint8)
                                        .cpu(),
                                        sent_gdn,
                                    )
                                    if not (kv_ok and gdn_ok):
                                        logger.warning(
                                            "Qwen3.5 megakernel: fused_pool_io "
                                            "warmup sentinel check failed "
                                            "(kv_ok=%s, gdn_ok=%s); "
                                            "falling back to 3-stage pool I/O",
                                            kv_ok,
                                            gdn_ok,
                                        )
                                        fallback_needed = True
                                        break
                                    logger.info(
                                        "Qwen3.5 megakernel: fused_pool_io "
                                        "warmup sentinel check passed "
                                        "(kv_ok=%s, gdn_ok=%s)",
                                        kv_ok,
                                        gdn_ok,
                                    )
                        else:
                            nseq = b // T
                            ids_b = torch.zeros((b,), dtype=torch.int32, device=dev)
                            pos_b = torch.zeros((b,), dtype=torch.int32, device=dev)
                            bt_b = torch.zeros(
                                (nseq * self.max_pages_per_seq,),
                                dtype=torch.int32,
                                device=dev,
                            )
                            s0_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                            s1_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                            s2_b = torch.zeros((nseq,), dtype=torch.int32, device=dev)
                            if ckpt_cols > 1:
                                c0_b = torch.zeros(
                                    (nseq, ckpt_cols), dtype=torch.int32, device=dev
                                )
                                c1_b = torch.zeros(
                                    (nseq, ckpt_cols), dtype=torch.int32, device=dev
                                )
                                c2_b = torch.zeros(
                                    (nseq, ckpt_cols), dtype=torch.int32, device=dev
                                )
                            else:
                                c0_b, c1_b, c2_b = s0_b, s1_b, s2_b
                            r_call = r_raw
                            if do_sentinel:
                                if dev.type != "tpu":
                                    self.raw_pools[0][0].view(torch.uint8).fill_(0x5A)
                                    self.raw_pools[0][5:9].view(torch.uint8).fill_(0x5A)
                                sent_kv = (
                                    self.raw_pools[0][0]
                                    .view(torch.uint8)
                                    .cpu()
                                    .clone()
                                )
                                sent_gdn = (
                                    self.raw_pools[0][5:9]
                                    .view(torch.uint8)
                                    .cpu()
                                    .clone()
                                )
                                ids_b[0] = 1
                                s0_b[0] = 1
                                s1_b[0] = 1
                                s2_b[0] = 1
                                if ckpt_cols > 1:
                                    blk_ids = torch.arange(
                                        1,
                                        1 + min(T, ckpt_cols),
                                        dtype=torch.int32,
                                        device=dev,
                                    )
                                    c0_b[0, : blk_ids.numel()] = blk_ids
                                    c1_b[0, : blk_ids.numel()] = blk_ids
                                    c2_b[0, : blk_ids.numel()] = blk_ids
                                if r_raw.numel() > 1:
                                    r_call = torch.full_like(r_raw, -1)
                            out_b = step_fn(
                                ids_b,
                                pos_b,
                                bt_b,
                                s0_b,
                                c0_b,
                                s1_b,
                                c1_b,
                                s2_b,
                                c2_b,
                                r_call,
                                self.w_flat,
                                self.raw_pools,
                            )
                            if dev.type == "tpu":
                                from vllm_torchtpu.utils import synchronize_tensors

                                synchronize_tensors([out_b, *self.raw_pools])
                            if do_sentinel:
                                sentinel_checked = True
                                kv_ok = not torch.equal(
                                    self.raw_pools[0][0].view(torch.uint8).cpu(),
                                    sent_kv,
                                )
                                gdn_ok = not torch.equal(
                                    self.raw_pools[0][5:9].view(torch.uint8).cpu(),
                                    sent_gdn,
                                )
                                if saved_pages is not None:
                                    for p, sv in zip(self.raw_pools, saved_pages):
                                        p[:n_save].copy_(sv)
                                if not (kv_ok and gdn_ok):
                                    logger.warning(
                                        "Qwen3.5 megakernel: fused_pool_io warmup "
                                        "sentinel check failed (kv_ok=%s, gdn_ok=%s); "
                                        "falling back to 3-stage pool I/O",
                                        kv_ok,
                                        gdn_ok,
                                    )
                                    fallback_needed = True
                                else:
                                    logger.info(
                                        "Qwen3.5 megakernel: fused_pool_io warmup "
                                        "sentinel check passed (kv_ok=%s, gdn_ok=%s)",
                                        kv_ok,
                                        gdn_ok,
                                    )
                        if fallback_needed:
                            break
                        logger.info(
                            "Qwen3.5 megakernel: warmed up bucket (T=%d, B=%d)", T, b
                        )
                    if fallback_needed:
                        self.fused_pool_io = False
                        self.base_cfg = dataclasses.replace(
                            self.base_cfg, fused_pool_io=False
                        )
                        self._build_all_buckets()
                        continue
                    break
                self.promote_fast_callables()
        finally:
            if saved_pages is not None:
                with torch.no_grad():
                    for p, sv in zip(self.raw_pools, saved_pages):
                        p[:n_save].copy_(sv)


    def _inspect_step_detail(
        self,
        attn_md: Any,
        mamba0: Any,
        runner: Any = None,
    ) -> tuple[bool, int, int, str]:
        """Return `(ok, num_reqs, tokens_per_seq, reason)` in O(1) without device-to-host sync."""
        if runner is not None:
            comb_key = getattr(runner, "_cached_comb_key", None)
            if isinstance(comb_key, tuple) and len(comb_key) == 3:
                _, num_windowed, num_reqs = comb_key
                if num_reqs <= 0:
                    return False, 0, 1, "stock_mixed"
                if num_windowed != num_reqs:
                    return (
                        False,
                        num_reqs,
                        1,
                        "stock_prefill" if num_windowed == 0 else "stock_mixed_prefill",
                    )
            else:
                comb_cpu = getattr(
                    runner, "_combined_request_distribution_cpu", None
                )
                req_dist_cpu = getattr(runner, "_request_distribution_cpu", None)
                bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
                if isinstance(comb_cpu, torch.Tensor) and comb_cpu.numel() >= 6:
                    num_windowed = int(comb_cpu[3].item())
                    num_reqs = int(comb_cpu[5].item())
                    if num_reqs <= 0:
                        return False, 0, 1, "stock_mixed"
                    if num_windowed != num_reqs:
                        return (
                            False,
                            num_reqs,
                            1,
                            "stock_prefill"
                            if num_windowed == 0
                            else "stock_mixed_prefill",
                        )
                elif (
                    isinstance(req_dist_cpu, torch.Tensor)
                    and req_dist_cpu.numel() >= 3
                ):
                    num_decode = int(req_dist_cpu[0].item())
                    num_reqs = int(req_dist_cpu[2].item())
                    if num_reqs <= 0:
                        return False, 0, 1, "stock_mixed"
                    if num_decode != num_reqs:
                        return (
                            False,
                            num_reqs,
                            1,
                            "stock_prefill"
                            if num_decode == 0
                            else "stock_mixed_prefill",
                        )
                elif bctx is not None and getattr(bctx, "num_reqs", None) is not None:
                    num_reqs = int(bctx.num_reqs)
                else:
                    num_reqs = int(attn_md.seq_lens.shape[0])

            qsl_np = getattr(runner, "query_start_loc_np", None)
            if qsl_np is not None and len(qsl_np) >= num_reqs + 1:
                if self._ckpt_window <= 4 and int(qsl_np[0]) == 0:
                    total_q = int(qsl_np[num_reqs])
                    if total_q == 4 * num_reqs:
                        return True, num_reqs, 4, "ok"
                    if total_q == num_reqs:
                        return True, num_reqs, 1, "ok"
                    return False, num_reqs, 0, "stock_mixed_qlen"
                q0 = int(qsl_np[1] - qsl_np[0])
                if q0 not in (1, 4):
                    return False, num_reqs, q0, "stock_mixed_qlen"
                for i in range(1, num_reqs):
                    if int(qsl_np[i + 1] - qsl_np[i]) != q0:
                        return False, num_reqs, q0, "stock_mixed_qlen"
                return True, num_reqs, q0, "ok"

        dist = getattr(mamba0, "mamba_request_distribution", None)
        if dist is None:
            dist = getattr(attn_md, "request_distribution", None)
        if dist is None:
            return False, 0, 1, "stock_mixed"

        if isinstance(dist, torch.Tensor):
            if dist.device.type == "cpu" and dist.numel() >= 3:
                d_end, _, total_reqs = (
                    int(dist[0].item()),
                    int(dist[1].item()),
                    int(dist[2].item()),
                )
                if total_reqs <= 0 or d_end != total_reqs:
                    return (
                        False,
                        0,
                        1,
                        "stock_prefill" if d_end == 0 else "stock_mixed_prefill",
                    )
                num_reqs = total_reqs
            else:
                num_reqs = int(attn_md.seq_lens.shape[0])
        else:
            num_reqs = int(attn_md.seq_lens.shape[0])

        max_q_len = getattr(attn_md, "max_query_len", None)
        if max_q_len is None:
            tokens_per_seq = 1
        else:
            tokens_per_seq = int(max_q_len)
            if tokens_per_seq not in (1, 4):
                return False, num_reqs, tokens_per_seq, "stock_mixed_qlen"
        return True, num_reqs, tokens_per_seq, "ok"

    def _inspect_step(
        self,
        attn_md: Any,
        mamba0: Any,
        runner: Any = None,
    ) -> tuple[bool, int, int]:
        """Return `(ok, num_reqs, tokens_per_seq)` without any device-to-host sync."""
        ok, num_reqs, tokens_per_seq, _ = self._inspect_step_detail(
            attn_md, mamba0, runner=runner
        )
        return ok, num_reqs, tokens_per_seq

    def can_run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        num_tokens: int | None = None,
        runner: Any = None,
    ) -> bool:
        """Return True iff the current forward step is a pure decode or MTP-verify step within compiled buckets."""
        _ = (positions, num_tokens)
        self._prepared_step = None
        self.ab_calibrator.last_decision = ("stock_early", "stock")
        if not self._bound:
            return False
        if (
            input_ids is None
            or inputs_embeds is not None
            or intermediate_tensors is not None
        ):
            return False

        ctx = get_forward_context()
        md = getattr(ctx, "attn_metadata", None)
        if not isinstance(md, dict) or not md:
            return False

        attn_md = md.get(self.attn_layer_name)
        if attn_md is None:
            return False
        m_names = self.mamba_layer_names
        m0 = md.get(m_names[0])
        m1 = md.get(m_names[1])
        m2 = md.get(m_names[2])
        if m0 is None or m1 is None or m2 is None:
            return False

        if getattr(m0, "mamba_state_indices", None) is None and getattr(
            m0, "mamba_ckpt_indices", None
        ) is None:
            return False

        ok, num_reqs, tokens_per_seq, reason = self._inspect_step_detail(
            attn_md, m0, runner=runner
        )
        if not ok or num_reqs <= 0:
            return self._note_step(reason, False)

        active_rows = num_reqs * tokens_per_seq
        if tokens_per_seq == 4:
            if not self.verify_buckets or active_rows > self.max_verify_bucket:
                return self._note_step(f"stock_t4_n{num_reqs}", False)
            bkt = select_bucket(active_rows, self.verify_buckets)
            step_fn = self._bucket_steps.get((4, bkt))
            if step_fn is None:
                return self._note_step("stock_t4_nobkt", False)
            if self._bucket_out_rows.get((4, bkt), input_ids.shape[0]) != int(
                input_ids.shape[0]
            ):
                return self._note_step(f"stock_t4_rows{bkt}", False)
            self._prepared_step = (
                step_fn,
                attn_md,
                m0,
                m1,
                m2,
                num_reqs,
                4,
                bkt,
            )
            return self._note_step(f"mk_t4_b{bkt}", True)
        if tokens_per_seq == 1:
            if not self.buckets or active_rows > self.max_bucket:
                return self._note_step(f"stock_t1_n{num_reqs}", False)
            bkt = select_bucket(active_rows, self.buckets)
            step_fn = (
                self._bucket_steps.get((1, bkt)) or self._bucket_steps.get(bkt)
            )
            if step_fn is None:
                return self._note_step("stock_t1_nobkt", False)
            self._prepared_step = (
                step_fn,
                attn_md,
                m0,
                m1,
                m2,
                num_reqs,
                1,
                bkt,
            )
            return self._note_step(f"mk_t1_b{bkt}", True)
        return self._note_step(f"stock_q{tokens_per_seq}", False)

    def _note_step(self, key: str, run: bool) -> bool:
        """Count a forward-step decision; log the MK/stock mix periodically."""
        actual_run, mix_key, _ = self.ab_calibrator.choose(key, run)
        self._step_mix[mix_key] = self._step_mix.get(mix_key, 0) + 1
        now = time.monotonic()
        if now - self._step_mix_t0 >= self._step_mix_period_s:
            if self.tp_rank == 0:
                total = sum(self._step_mix.values())
                mk = sum(v for k, v in self._step_mix.items() if k.startswith("mk_"))
                logger.info(
                    "Qwen3.5 megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                    now - self._step_mix_t0,
                    mk,
                    total,
                    100.0 * mk / max(total, 1),
                    dict(sorted(self._step_mix.items())),
                )
            self._step_mix = {}
            self._step_mix_t0 = now
        return actual_run

    def record_ab_step(
        self, dt_ms: float, padded_tokens: int | None = None
    ) -> None:
        """Record a timed `forward_model` step and periodically log window + cumulative A/B latency JSON on rank 0."""
        res = self.ab_calibrator.record(dt_ms, padded_tokens=padded_tokens)
        if res is not None and self.tp_rank == 0:
            elapsed, win_json, cum_json = res
            logger.info(
                "Qwen3.5 megakernel AB latency (window %.0fs): %s",
                elapsed,
                win_json,
            )
            logger.info(
                "Qwen3.5 megakernel AB latency (cumulative): %s",
                cum_json,
            )

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int | None = None,
        runner: Any = None,
    ) -> torch.Tensor:
        """Execute one full-backbone decode or MTP-verify step and return `[out_rows, H]` hidden states."""
        prepared = self._prepared_step
        self._prepared_step = None
        if prepared is not None:
            step_fn, attn_md, m0, m1, m2, num_reqs, tokens_per_seq, bucket_t = (
                prepared
            )
        else:
            md = get_forward_context().attn_metadata
            attn_md = md[self.attn_layer_name]
            m0 = md[self.mamba_layer_names[0]]
            m1 = md[self.mamba_layer_names[1]]
            m2 = md[self.mamba_layer_names[2]]
            _, num_reqs, tokens_per_seq = self._inspect_step(
                attn_md, m0, runner=runner
            )
            active_rows = num_reqs * tokens_per_seq
            if tokens_per_seq == 4:
                bucket_t = select_bucket(
                    max(4, min(active_rows, self.max_verify_bucket)),
                    self.verify_buckets,
                )
                step_fn = self._bucket_steps[(4, bucket_t)]
            else:
                bucket_t = select_bucket(
                    max(1, min(active_rows, self.max_bucket)), self.buckets
                )
                step_fn = (
                    self._bucket_steps.get((1, bucket_t))
                    or self._bucket_steps[bucket_t]
                )

        out_rows = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        fb_1d = attn_md.seq_lens

        s0 = m0.mamba_state_indices
        c0 = m0.mamba_ckpt_indices
        if s0 is None:
            s0 = c0 if c0 is not None else fb_1d
        if c0 is None:
            c0 = s0

        s1 = m1.mamba_state_indices
        c1 = m1.mamba_ckpt_indices
        if s1 is None:
            s1 = c1 if c1 is not None else fb_1d
        if c1 is None:
            c1 = s1

        s2 = m2.mamba_state_indices
        c2 = m2.mamba_ckpt_indices
        if s2 is None:
            s2 = c2 if c2 is not None else fb_1d
        if c2 is None:
            c2 = s2

        roff = getattr(m0, "mamba_slot_read_offsets", None)
        if roff is None and runner is not None:
            roff = getattr(runner, "mamba_slot_read_offsets", None)
        if roff is None:
            roff = self._zero_roff_1d

        active_rows = num_reqs * tokens_per_seq
        if positions is not None and positions.numel() >= active_rows:
            pos_in = positions
        elif tokens_per_seq > 1:
            offsets = torch.arange(
                1 - tokens_per_seq, 1, dtype=torch.int32, device=fb_1d.device
            ).unsqueeze(0)
            pos_in = (fb_1d.unsqueeze(1) + offsets).clamp(min=0).reshape(-1)
        else:
            pos_in = (fb_1d - 1).clamp(min=0)

        hidden_out = step_fn(
            input_ids,
            pos_in,
            attn_md.block_tables,
            s0,
            c0,
            s1,
            c1,
            s2,
            c2,
            roff,
            self.w_flat,
            self.raw_pools,
        )

        if out_rows == hidden_out.shape[0]:
            return hidden_out
        if out_rows <= hidden_out.shape[0]:
            return hidden_out[:out_rows]
        return F.pad(hidden_out, (0, 0, 0, out_rows - hidden_out.shape[0]))

