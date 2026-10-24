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
"""vLLM runtime driver (`MegakernelDriver`) for `megakernel`.

Integrates the modular `megakernel` library with `TPUModelRunner`:
- Extracts/packs model weights via `weight_collector.pack_weights` (preserving
  3D expert `Parameter` base buffers when `stack_e=False`).
- Binds unified KV+GDN `float8_e4m3fn` raw pools (`bind_caches`) and builds
  per-bucket ops via `exported_bridge.build_pooled_step_op` +
  `execute_pooled_step_inplace`.
- Verifies pool donation in lowered JAX HLO (`verify_pooled_step_hlo`) and
  outer `torch_tpu` compiled IR (`verify_torchtpu_compiled_step_ir`) across
  every compiled bucket, and runs a warmup sentinel check per `T` variant
  confirming both KV and GDN pages are updated in-place.
- Dispatches eligible `T=1` decode and `T=4` MTP verify steps from
  `TPUModelRunner.forward_model`, with optional A/B calibration (`MKABCalibrator`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import time
import types
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
import torch
import torch.nn.functional as F

try:
    from vllm.forward_context import get_forward_context
except ImportError:

    def get_forward_context() -> Any:  # type: ignore[misc]
        raise RuntimeError("vllm.forward_context is unavailable")


import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.cache import (
    should_fuse_pool_io,
    should_use_pool_gather,
)
from vllm_torchtpu.megakernel.errors import MegakernelLayoutError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.model_def import ModelDef, get_model
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.modules.gqa import GQAModule, GQATunables
from vllm_torchtpu.megakernel.ops import rup
from vllm_torchtpu.megakernel.pool.config import CacheExec, with_cache
from vllm_torchtpu.megakernel.profile import (
    Profile,
    get_profile,
    model_default_profile,
)
from vllm_torchtpu.megakernel.profiles.default import chip_budget
from vllm_torchtpu.megakernel_runtime.exported_bridge import (
    MegakernelConfigError,
    build_pooled_step_op,
    execute_pooled_step_inplace,
    verify_pooled_step_hlo,
    verify_torchtpu_compiled_step_ir,
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (
    find_backbone,
    get_routed_experts,
    pack_weights,
    reshard_layer_experts_ep_to_tp,
)

logger = init_logger(__name__)

# Default T=1 decode buckets (when speculative decoding is disabled) and T=4 MTP
# verify buckets (11 buckets covering NS in {1..6, 8, 10, 12, 14, 16} active
# sequences at T=4 to avoid rounding medium-concurrency steps up to the next
# power of 2). Cold XLA compilation takes ~12-15s per bucket on TPU v7x-8
# (~2.5 min total for all 11 verify buckets, <1s/bucket on warm start with
# Tier-3 PJRT cache); override via VLLM_TPU_MEGAKERNEL="verify_buckets=...".
_DEFAULT_BUCKETS = (8, 16, 32)
_DEFAULT_VERIFY_BUCKETS = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
_MTP_VERIFY_TOKENS = 4
_MTP_SPECULATIVE_TOKENS = _MTP_VERIFY_TOKENS - 1
_DEFAULT_FUSED_POOL_IO_MAX_B = 24
_DEFAULT_FUSED_POOL_IO_MIN_B_HI = 56


def _need_layout(cond: bool, msg: str) -> None:
    """Fail with `MegakernelLayoutError`: incompatible runtime cache layout."""
    if not cond:
        raise MegakernelLayoutError(f"MegakernelDriver: {msg}")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"MegakernelDriver: {msg}")


def _unwrap(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    return x


def _resolve_tp_size(runner: Any) -> int:
    """Resolve the tensor-parallel device count for `runner`."""
    p_cfg = getattr(runner, "parallel_config", None)
    tp_size = int(getattr(p_cfg, "tensor_parallel_size", 0) or 0)
    if tp_size > 0:
        return tp_size
    mesh = getattr(runner, "mesh", None)
    if mesh is not None and getattr(mesh, "devices", None) is not None:
        return int(mesh.devices.size)
    return 0


def _build_tp_rank_mesh(r_cnt: int, fallback_mesh: Any = None) -> Any:
    """Build an `r_cnt`-device JAX `Mesh` with axis `'tp'` in TP-rank order.

    When `fallback_mesh` already contains `r_cnt` devices (e.g. CPU unit-test
    runner), reuses its devices with axis `'tp'`. Otherwise queries
    `torch_tpu._internal.pallas.get_pallas_mesh(axis_names=('tp',))` so device
    ordering matches `TPUWorker` TP ranks on `tpu7x` (`2x2x1`) hosts rather than
    sorting by physical `device_id`.
    """
    import jax

    if (
        fallback_mesh is not None
        and getattr(fallback_mesh, "devices", None) is not None
        and int(fallback_mesh.devices.size) == r_cnt
    ):
        if tuple(fallback_mesh.axis_names) == ("tp",):
            return fallback_mesh
        return jax.sharding.Mesh(np.asarray(fallback_mesh.devices).reshape(-1), ("tp",))
    try:
        from torch_tpu._internal.pallas import get_pallas_mesh

        pallas_mesh = get_pallas_mesh(axis_names=("tp",))
        if int(pallas_mesh.devices.size) == r_cnt:
            return pallas_mesh
    except Exception as exc:
        logger.debug(
            "get_pallas_mesh unavailable (%s); falling back to jax.devices()",
            exc,
        )
    devs = jax.devices()[:r_cnt]
    return jax.sharding.Mesh(np.asarray(devs), ("tp",))


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", "none", ""):
        return False
    raise ValueError(f"MegakernelDriver: invalid boolean value {val!r}")


def _parse_fused_pool_io(val: Any) -> bool | str:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    if s in ("1", "true", "yes", "on", "fused"):
        return True
    if s in ("", "0", "false", "no", "off", "none", "split", "3stage"):
        return False
    raise ValueError(
        f"MegakernelDriver: invalid fused_pool_io={val!r}; "
        "expected 'auto', 'true', or 'false'"
    )


def _parse_pool_gather(val: Any) -> bool | str:
    """Parse `pool_gather` to `True`, `False`, or `"auto"` (empty string: off).

    As a profile setting, `True` means "every bucket where the pool backend can
    gather" (never a `T=1` bucket), so it cannot fail a bucket build.
    """
    if val is None:
        return "auto"
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    if s in ("1", "true", "yes", "on", "gpr"):
        return True
    if s in ("", "0", "false", "no", "off", "none", "split", "3stage"):
        return False
    raise ValueError(
        f"MegakernelDriver: invalid pool_gather={val!r}; "
        "expected 'auto', 'true', or 'false'"
    )


def parse_mk_buckets(val: str | Sequence[int] | None = None) -> tuple[int, ...]:
    """Parse `T=1` decode batch buckets from a string or sequence (`'0'` disables)."""
    if val is None:
        cfg = parse_megakernel_config()
        return cfg["buckets"]
    if not isinstance(val, str):
        buckets = sorted({int(x) for x in val})
        if not buckets:
            return ()
        _need(
            all(1 <= b <= 64 for b in buckets),
            f"invalid decode buckets={val!r}",
        )
        return tuple(buckets)
    raw = val.strip()
    if raw.lower() in ("0", "none", "off", "false", ""):
        return ()
    parts = raw.replace(":", ",").split(",")
    buckets = sorted({int(x.strip()) for x in parts if x.strip()})
    _need(
        len(buckets) > 0 and all(1 <= b <= 64 for b in buckets),
        f"invalid decode buckets={val!r}",
    )
    return tuple(buckets)


def parse_mk_verify_buckets(val: str | Sequence[int] | None = None) -> tuple[int, ...]:
    """Parse `T=4` MTP verify row buckets from a string or sequence (`'0'` disables)."""
    if val is None:
        cfg = parse_megakernel_config()
        return cfg["verify_buckets"]
    if not isinstance(val, str):
        buckets = sorted({int(x) for x in val})
        if not buckets:
            return ()
        _need(
            all(4 <= b <= 64 and b % 4 == 0 for b in buckets),
            f"invalid verify_buckets={val!r}",
        )
        return tuple(buckets)
    raw = val.strip()
    if raw.lower() in ("0", "none", "off", "false", ""):
        return ()
    parts = raw.replace(":", ",").split(",")
    buckets = sorted({int(x.strip()) for x in parts if x.strip()})
    _need(
        len(buckets) > 0 and all(4 <= b <= 64 and b % 4 == 0 for b in buckets),
        f"invalid verify_buckets={val!r}",
    )
    return tuple(buckets)


def parse_megakernel_config(raw: Any = None, runner: Any = None) -> dict[str, Any]:
    """Parse `VLLM_TPU_MEGAKERNEL` and `vllm_config.additional_config['megakernel']`.

    Supported `VLLM_TPU_MEGAKERNEL` formats:
    - `""` / `"0"` / `"false"`: disabled (default)
    - `"1"` / `"true"`: enabled with default settings
    - `"ep"` / `"tp"`: enabled with explicit sharding mode
    - `"mode=tp,buckets=8:16:32,verify_buckets=4:8:16,fused_pool_io=false,ab=1"`
    - `"profile=qwen35-candidate30"`: named execution profile (tiles, perf flags,
      buckets, pool I/O); explicit keys in the same config override the profile.
      Without a `profile` key, `cfg["profile"]` is `None` and the driver runs the
      model's default profile (`profile.model_default_profile`).
    - JSON object string, e.g. `'{"mode": "tp", "buckets": [8, 16], "ab": true}'`
    """
    cfg: dict[str, Any] = {
        "enabled": False,
        "mode": "",
        "profile": None,
        "buckets": _DEFAULT_BUCKETS,
        "verify_buckets": _DEFAULT_VERIFY_BUCKETS,
        "buckets_explicit": False,
        "verify_buckets_explicit": False,
        "fused_pool_io": False,
        "fused_pool_io_max_b": _DEFAULT_FUSED_POOL_IO_MAX_B,
        "fused_pool_io_min_b_hi": _DEFAULT_FUSED_POOL_IO_MIN_B_HI,
        "pool_gather": False,
        "ab": False,
        "stats_period_s": 60.0,
        "compile_hlo_check": False,
    }

    explicit: set[str] = set()

    def _apply_dict(d: dict[str, Any]) -> None:
        cfg["enabled"] = _parse_bool(d.get("enabled", True))
        if "profile" in d and d["profile"] is not None:
            cfg["profile"] = str(d["profile"]).strip() or None
        if "mode" in d and d["mode"] is not None:
            m = str(d["mode"]).strip().lower()
            _need(m in ("", "auto", "ep", "tp"), f"invalid mode={d['mode']!r}")
            cfg["mode"] = "" if m == "auto" else m
        if "buckets" in d and d["buckets"] is not None:
            cfg["buckets"] = parse_mk_buckets(d["buckets"])
            cfg["buckets_explicit"] = True
        if "verify_buckets" in d and d["verify_buckets"] is not None:
            cfg["verify_buckets"] = parse_mk_verify_buckets(d["verify_buckets"])
            cfg["verify_buckets_explicit"] = True
        if "fused_pool_io" in d and d["fused_pool_io"] is not None:
            cfg["fused_pool_io"] = _parse_fused_pool_io(d["fused_pool_io"])
            explicit.add("fused_pool_io")
        if "fused_pool_io_max_b" in d and d["fused_pool_io_max_b"] is not None:
            cfg["fused_pool_io_max_b"] = int(d["fused_pool_io_max_b"])
            explicit.add("fused_pool_io_max_b")
        if "fused_pool_io_min_b_hi" in d and d["fused_pool_io_min_b_hi"] is not None:
            cfg["fused_pool_io_min_b_hi"] = int(d["fused_pool_io_min_b_hi"])
            explicit.add("fused_pool_io_min_b_hi")
        if "pool_gather" in d and d["pool_gather"] is not None:
            cfg["pool_gather"] = _parse_pool_gather(d["pool_gather"])
            explicit.add("pool_gather")
        if "ab" in d and d["ab"] is not None:
            cfg["ab"] = _parse_bool(d["ab"])
        if "stats_period_s" in d and d["stats_period_s"] is not None:
            cfg["stats_period_s"] = float(d["stats_period_s"])
        if "compile_hlo_check" in d and d["compile_hlo_check"] is not None:
            cfg["compile_hlo_check"] = _parse_bool(d["compile_hlo_check"])

    sources: list[Any] = [envs.VLLM_TPU_MEGAKERNEL if raw is None else raw]
    if runner is not None:
        vllm_cfg = getattr(runner, "vllm_config", None)
        add_cfg = getattr(vllm_cfg, "additional_config", None)
        if isinstance(add_cfg, dict) and "megakernel" in add_cfg:
            sources.append(add_cfg["megakernel"])

    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            cfg["enabled"] = src
            continue
        if isinstance(src, dict):
            _apply_dict(src)
            continue
        s = str(src).strip()
        if not s or s.lower() in ("0", "false", "no", "off", "none"):
            cfg["enabled"] = False
            continue
        if s.lower() in ("1", "true", "yes", "on"):
            cfg["enabled"] = True
            continue
        if s.lower() in ("ep", "tp"):
            cfg["enabled"] = True
            cfg["mode"] = s.lower()
            continue
        if s.startswith("{"):
            parsed = json.loads(s)
            _need(isinstance(parsed, dict), f"expected JSON object in {s!r}")
            _apply_dict(parsed)
            continue
        kv_dict: dict[str, Any] = {}
        for item in s.split(","):
            part = item.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid VLLM_TPU_MEGAKERNEL entry {part!r} in {s!r}")
            k, v = part.split("=", 1)
            kv_dict[k.strip()] = v.strip()
        _apply_dict(kv_dict)

    _apply_profile_defaults(cfg, explicit)
    cfg["explicit"] = frozenset(explicit)
    return cfg


def _apply_profile_defaults(cfg: dict[str, Any], explicit: set[str]) -> None:
    """Fill profile-owned keys from `cfg["profile"]` unless set explicitly.

    With no named profile the keys keep their library defaults;
    `MegakernelDriver.maybe_create` passes only the explicit ones
    (`explicit_profile_settings`), so the model's default profile decides the
    rest.
    """
    if cfg["profile"] is None:
        return
    try:
        prof = get_profile(cfg["profile"])
    except KeyError as exc:
        raise ValueError(f"MegakernelDriver: {exc.args[0]}") from None
    cfg["profile"] = prof.name
    if "fused_pool_io" not in explicit:
        cfg["fused_pool_io"] = prof.fused_pool_io
    if "fused_pool_io_max_b" not in explicit:
        cfg["fused_pool_io_max_b"] = int(prof.fused_pool_io_max_b)
    if "fused_pool_io_min_b_hi" not in explicit:
        cfg["fused_pool_io_min_b_hi"] = int(prof.fused_pool_io_min_b_hi)
    if "pool_gather" not in explicit:
        cfg["pool_gather"] = "auto" if prof.pool_gather is None else prof.pool_gather
    if not cfg["buckets_explicit"] and prof.buckets is not None:
        cfg["buckets"] = parse_mk_buckets(prof.buckets)
    if not cfg["verify_buckets_explicit"] and prof.verify_buckets is not None:
        cfg["verify_buckets"] = parse_mk_verify_buckets(prof.verify_buckets)


def explicit_profile_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Profile-owned driver arguments from `cfg`: the value where the user set
    the key, else None.

    None lets `MegakernelDriver` take the setting from the effective profile
    (the named one, else the model's default profile) instead of the library
    defaults `parse_megakernel_config` fills in.
    """
    explicit = cfg.get("explicit", frozenset())
    out: dict[str, Any] = {
        "buckets": tuple(cfg["buckets"]) if cfg["buckets_explicit"] else None,
        "verify_buckets": (
            tuple(cfg["verify_buckets"]) if cfg["verify_buckets_explicit"] else None
        ),
    }
    for key in (
        "fused_pool_io",
        "fused_pool_io_max_b",
        "fused_pool_io_min_b_hi",
        "pool_gather",
    ):
        out[key] = cfg[key] if key in explicit else None
    return out


def resolve_driver_profile(
    profile: str | Profile | None,
    *,
    model_name: str,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
    pool_gather: bool | str | None = None,
) -> Profile:
    """Resolve the effective profile: named profile plus explicit overrides.

    `profile=None` (or `""`) selects the model's default profile
    (`model_default_profile(model_name)`, `default` if the model declares
    none). `None` overrides keep the profile's value. Fails closed
    (`ValueError`) for unknown profiles or a profile tuned for a different
    `ModelSpec.name`.
    """
    try:
        prof = get_profile(profile) if profile else model_default_profile(model_name)
    except KeyError as exc:
        raise ValueError(f"MegakernelDriver: {exc.args[0]}") from None
    _need(
        prof.model is None or prof.model == model_name,
        f"profile {prof.name!r} targets model {prof.model!r}, not {model_name!r}",
    )
    overrides: dict[str, Any] = {}
    if fused_pool_io is not None:
        overrides["fused_pool_io"] = _parse_fused_pool_io(fused_pool_io)
    if fused_pool_io_max_b is not None:
        overrides["fused_pool_io_max_b"] = int(fused_pool_io_max_b)
    if fused_pool_io_min_b_hi is not None:
        overrides["fused_pool_io_min_b_hi"] = int(fused_pool_io_min_b_hi)
    if pool_gather is not None:
        pg = _parse_pool_gather(pool_gather)
        overrides["pool_gather"] = None if pg == "auto" else pg
    # Real changes re-label the profile (`<name>+overrides`) and derive its
    # numerics class, so an override never silently keeps `exact`.
    return prof.with_overrides(**overrides)


def _resolve_driver_buckets(
    profile: Profile,
    buckets: Sequence[int] | None,
    verify_buckets: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve `(buckets, verify_buckets)` from explicit args, `profile` or defaults."""
    eff_buckets = (
        buckets
        if buckets is not None
        else (profile.buckets if profile.buckets is not None else _DEFAULT_BUCKETS)
    )
    eff_verify = (
        verify_buckets
        if verify_buckets is not None
        else (
            profile.verify_buckets
            if profile.verify_buckets is not None
            else _DEFAULT_VERIFY_BUCKETS
        )
    )
    return tuple(sorted(eff_buckets)), tuple(sorted(eff_verify))


def check_compile_hlo_mesh(mesh: Any, compile_hlo_check: bool) -> None:
    """Fail at config time if `compile_hlo_check=1` cannot work on `mesh`.

    Compiled-HLO verification needs real `jax.Device`s; under torch_tpu's
    `_PallasDevice` mesh XLA compilation belongs to torch_tpu and is verified via
    the compiled-IR check instead. Raises `MegakernelConfigError` (not a
    `ValueError`, which `bind_and_warmup` treats as an incompatible layout).
    """
    if not compile_hlo_check:
        return
    devs = getattr(mesh, "devices", None)
    first = devs.flat[0] if devs is not None and devs.size > 0 else None
    if not isinstance(first, jax.Device):
        raise MegakernelConfigError(
            "MegakernelDriver: compile_hlo_check=1 requires a mesh of real "
            f"jax.Device instances (got {type(first).__name__}); drop "
            "compile_hlo_check under torch_tpu (its compiled IR is checked instead)"
        )


def select_bucket(num_tokens: int, buckets: Sequence[int] = _DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket `>= num_tokens`."""
    for b in buckets:
        if b >= num_tokens:
            return b
    raise ValueError(f"num_tokens={num_tokens} exceeds max bucket in {buckets}")


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
    """Bounded-memory ring-buffer sample accumulator (exact mean + recent quantiles)."""

    __slots__ = ("cap", "count", "samples", "sum_ms")

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
        self.counts.clear()
        self.last_decision = ("stock_early", "stock")
        self._window.clear()
        self._cumulative.clear()
        self._t0 = time.monotonic()

    def choose(self, key: str, mk_eligible: bool) -> tuple[bool, str, str]:
        """Return `(actual_run_mk, mix_key, arm)` for the current step."""
        if not mk_eligible:
            self.last_decision = (key, "stock")
            return False, key, "stock"
        if not self.enabled:
            self.last_decision = (key, "mk")
            return True, key, "mk"
        c = self.counts.get(key, 0)
        self.counts[key] = c + 1
        if (c & 1) == 0:
            self.last_decision = (key, "mk")
            return True, f"{key}_ab_mk", "mk"
        self.last_decision = (key, "stock")
        return False, f"{key}_ab_stock", "stock"

    def record(
        self, dt_ms: float, padded_tokens: int | None = None
    ) -> tuple[float, str, str] | None:
        """Record a timed step and return `(elapsed_s, window_json, cum_json)`."""
        if not self.enabled:
            return None
        key, arm = self.last_decision
        rec_key = f"{key}_p{int(padded_tokens)}" if padded_tokens is not None else key
        for store in (self._window, self._cumulative):
            bucket_map = store.get(rec_key)
            if bucket_map is None:
                bucket_map = {}
                store[rec_key] = bucket_map
            series = bucket_map.get(arm)
            if series is None:
                series = _LatencySeries(self.cap)
                bucket_map[arm] = series
            series.add(dt_ms)

        now = time.monotonic()
        elapsed = now - self._t0
        if elapsed < self.period_s:
            return None
        win_json = self.format_summary(self._window)
        cum_json = self.format_summary(self._cumulative)
        self._window.clear()
        self._t0 = now
        return elapsed, win_json, cum_json

    @staticmethod
    def format_summary(store: dict[str, dict[str, _LatencySeries]]) -> str:
        out: dict[str, dict[str, Any]] = {}
        for k in sorted(store):
            arms = store[k]
            entry: dict[str, Any] = {}
            for arm in ("mk", "stock"):
                if arm in arms:
                    entry[arm] = arms[arm].summary()
            if "mk" in arms and "stock" in arms:
                mk_s = arms["mk"].summary()
                st_s = arms["stock"].summary()
                entry["d_mean_ms"] = round(float(mk_s[1]) - float(st_s[1]), 3)
                entry["d_p50_ms"] = round(float(mk_s[3]) - float(st_s[3]), 3)
            out[k] = entry
        return json.dumps(out, separators=(",", ":"))


def _slice_or_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x1 = x.reshape(-1)
    n = int(x1.shape[0])
    if n >= target_len:
        return x1[:target_len]
    return F.pad(x1, (0, target_len - n), value=0)


def _slice_or_pad_2d(
    x: torch.Tensor, target_rows: int, target_cols: int
) -> torch.Tensor:
    r, c = int(x.shape[0]), int(x.shape[1])
    out = x[: min(r, target_rows), : min(c, target_cols)]
    pad_r = max(0, target_rows - r)
    pad_c = max(0, target_cols - c)
    if pad_r > 0 or pad_c > 0:
        out = F.pad(out, (0, pad_c, 0, pad_r), value=0)
    return out


def prepare_step_inputs_eager(
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
    """Pack `(ids_b, pos_b, pt_b, ckpt_b, roff_b)` for a bucket of `bucket_t` tokens."""
    T = tokens_per_seq
    B = bucket_t
    nseq_bkt = B // T

    bt = block_tables
    if bt.ndim == 1:
        n_rows_bt = int(s0.shape[0])
        if n_rows_bt > 0 and bt.numel() % n_rows_bt == 0:
            bt = bt.reshape(n_rows_bt, bt.numel() // n_rows_bt)
        elif max_pages_per_seq > 0 and bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            raise ValueError(
                f"1-D block_tables of length {bt.numel()} is not divisible by "
                f"s0.shape[0]={n_rows_bt} or max_pages_per_seq={max_pages_per_seq}"
            )
    pt_b = _slice_or_pad_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    for s_raw, c_raw, r_raw in ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)):
        s_b = (
            _slice_or_pad_1d(s_raw[:, 0], nseq_bkt).to(torch.int32)
            if s_raw.ndim == 2
            else _slice_or_pad_1d(s_raw, nseq_bkt).to(torch.int32)
        )
        if has_roff:
            s_idx = s_b.to(torch.long).clamp(min=0, max=r_raw.numel() - 1)
            roff_g = r_raw.reshape(-1)[s_idx].to(torch.int32)
        else:
            roff_g = torch.zeros_like(s_b)

        if T > 1:
            if use_ckpt_2d and c_raw.ndim == 2 and c_raw.shape[1] >= T:
                ckpt_g = _slice_or_pad_2d(c_raw, nseq_bkt, T).to(torch.int32)
            else:
                src_1d = c_raw[:, 0] if c_raw.ndim == 2 else c_raw.reshape(-1)
                ckpt_1d = _slice_or_pad_1d(src_1d, nseq_bkt).to(torch.int32)
                ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T).contiguous()
                roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))
        else:
            # For T=1, build 2-column `[read_slot, write_slot]` so `gather_gdn`
            # reads from `c_b[s, roff[s]]` while `scatter_gdn` (`ckpt_t_off=1`)
            # writes the updated state back to the group base slot `s_b`.
            if use_ckpt_2d and has_roff and c_raw.ndim == 2 and c_raw.shape[1] > 1:
                c_b = _slice_or_pad_2d(c_raw, nseq_bkt, int(c_raw.shape[1])).to(
                    torch.int32
                )
                r_idx = roff_g.to(torch.long).clamp(min=0, max=c_raw.shape[1] - 1)
                read_col = c_b.gather(1, r_idx.unsqueeze(1))
            else:
                read_col = s_b.unsqueeze(1)
            write_col = s_b.unsqueeze(1)
            ckpt_g = torch.cat([read_col, write_col], dim=1)
            roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))

        active_g = (s_b > 0) & (ckpt_g[:, 0] > 0)
        ckpt_g = torch.where(active_g.unsqueeze(1), ckpt_g, torch.zeros_like(ckpt_g))
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
    pos_b = torch.where(active_tok, _slice_or_pad_1d(pos_1d, B).to(torch.int32), zero_b)
    return ids_b, pos_b, pt_b, ckpt_b, roff_b


_compiled_prepare_step_inputs: Callable[..., Any] | None = None


def _get_compiled_prepare_step_inputs() -> Callable[..., Any]:
    global _compiled_prepare_step_inputs
    fn = _compiled_prepare_step_inputs
    if fn is None:
        fn = torch.compile(
            prepare_step_inputs_eager, backend="tpu", fullgraph=True, dynamic=False
        )
        _compiled_prepare_step_inputs = fn
    return fn


def _extract_group_mamba_tensors(
    m_md: Any,
    fallback_1d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
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


def prepare_step_metadata(
    attn_md: Any,
    mamba_mds: Sequence[Any],
    positions: torch.Tensor | None,
    num_reqs: int,
    bucket_t: int,
    max_pages_per_seq: int,
    tokens_per_seq: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded `(pos_b, pt_b, ckpt_b, roff_b)` tensors for `bucket_t`."""
    _need(
        len(mamba_mds) == 3,
        f"expected 3 GDN group metadata objects, got {len(mamba_mds)}",
    )
    T = int(tokens_per_seq)
    B = int(bucket_t)
    _need(B % T == 0, f"bucket_t={B} must be divisible by tokens_per_seq={T}")
    bt = attn_md.block_tables
    dev = bt.device
    active_rows = num_reqs * T

    fb_1d = getattr(attn_md, "seq_lens", None)
    if fb_1d is None:
        fb_1d = torch.zeros((max(1, num_reqs),), dtype=torch.int32, device=dev)

    if positions is not None:
        pos_flat = positions[0] if positions.ndim == 2 else positions.reshape(-1)
    else:
        pos_flat = None
    if pos_flat is not None and pos_flat.numel() >= active_rows:
        pos_in = pos_flat[:active_rows]
    elif T > 1:
        seq_l = _slice_or_pad_1d(fb_1d, num_reqs).to(torch.int32)
        offsets = torch.arange(1 - T, 1, dtype=torch.int32, device=dev).unsqueeze(0)
        pos_in = (seq_l.unsqueeze(1) + offsets).clamp(min=0).reshape(-1)
    else:
        seq_l = _slice_or_pad_1d(fb_1d, num_reqs).to(torch.int32)
        pos_in = (seq_l - 1).clamp(min=0)

    s0, c0, r0, has_r0, use_c0 = _extract_group_mamba_tensors(mamba_mds[0], fb_1d)
    s1, c1, r1, has_r1, use_c1 = _extract_group_mamba_tensors(mamba_mds[1], fb_1d)
    s2, c2, r2, has_r2, use_c2 = _extract_group_mamba_tensors(mamba_mds[2], fb_1d)
    has_roff = bool(has_r0 or has_r1 or has_r2)
    use_ckpt_2d = bool(use_c0 or use_c1 or use_c2)

    if bt.ndim == 1:
        n_rows_bt = int(s0.shape[0])
        if n_rows_bt > 0 and bt.numel() % n_rows_bt == 0:
            bt = bt.reshape(n_rows_bt, bt.numel() // n_rows_bt)
        elif max_pages_per_seq > 0 and bt.numel() % int(max_pages_per_seq) == 0:
            bt = bt.reshape(-1, int(max_pages_per_seq))
        else:
            raise ValueError(
                f"1-D block_tables of length {bt.numel()} is not divisible by "
                f"s0.shape[0]={n_rows_bt} or max_pages_per_seq={max_pages_per_seq}"
            )
    bt = bt[:num_reqs]

    dummy_ids = torch.ones((active_rows,), dtype=torch.int32, device=dev)
    prep_fn = (
        _get_compiled_prepare_step_inputs()
        if dev.type == "tpu"
        else prepare_step_inputs_eager
    )
    _, pos_b, pt_b, ckpt_b, roff_b = prep_fn(
        dummy_ids,
        pos_in,
        bt,
        s0[:num_reqs],
        c0[:num_reqs],
        r0,
        s1[:num_reqs],
        c1[:num_reqs],
        r1,
        s2[:num_reqs],
        c2[:num_reqs],
        r2,
        B,
        T,
        int(max_pages_per_seq),
        has_roff,
        use_ckpt_2d,
    )
    return pos_b, pt_b, ckpt_b, roff_b


def _pcp_or_dcp_enabled(runner: Any) -> bool:
    if getattr(runner, "_pcp_enabled", False) or getattr(runner, "_dcp_enabled", False):
        return True
    p_cfg = getattr(runner, "parallel_config", None)
    if p_cfg is not None:
        if int(getattr(p_cfg, "prefill_context_parallel_size", 1) or 1) > 1:
            return True
        if int(getattr(p_cfg, "decode_context_parallel_size", 1) or 1) > 1:
            return True
    return False


class MegakernelDriver:
    """Model-agnostic megakernel runtime driver for `TPUModelRunner`.

    Subclasses (such as `Qwen35Megakernel`) or callers provide the declarative
    `ModelSpec` and weight packing via `_build_spec` and
    `_prepare_and_pack_weights` (or by passing `spec` / `weights` directly).
    """

    _model_name: str = qwen35.QWEN35_MODEL_NAME
    _expected_num_layers: int = get_model(_model_name).derive_geometry().L
    _total_routed_experts: int = get_model(_model_name).derive_geometry().E

    @classmethod
    def _model_def(cls) -> type[ModelDef]:
        mdef = get_model(cls._model_name)
        if mdef.segment_composed:
            raise MegakernelConfigError(
                f"{mdef.name!r} is segment-composed; {cls.__name__} runs "
                "whole-model kernels"
            )
        return mdef

    def __init__(
        self,
        model: Any,
        mesh: Any,
        *,
        tp_rank: int = 0,
        mode: str = "ep",
        spec: Any = None,
        geom: Any = None,
        base_exec: ExecSpec | None = None,
        weights: dict[str, Any] | None = None,
        buckets: Sequence[int] | None = None,
        verify_buckets: Sequence[int] | None = None,
        reshard_ep_to_tp: bool = False,
        expert_placement: Any = None,
        profile: str | Profile | None = None,
        fused_pool_io: bool | str | None = None,
        fused_pool_io_max_b: int | None = None,
        fused_pool_io_min_b_hi: int | None = None,
        pool_gather: bool | str | None = None,
        ab: bool | None = None,
        stats_period_s: float | None = None,
        compile_hlo_check: bool | None = None,
    ) -> None:
        _need(mode in ("ep", "tp"), f"invalid mode={mode!r}; expected 'ep' or 'tp'")
        self.model = model
        self.backbone = find_backbone(model)
        self.embed_tokens = self.backbone.embed_tokens
        self.mesh = mesh
        self.tp_rank = int(tp_rank)
        self.mode = mode

        r_cnt = int(mesh.devices.size)
        if spec is None:
            spec = self._build_spec(mode=mode, r_cnt=r_cnt, geom=geom)
        self.spec = spec
        self.geom = spec.config

        # Resolve the profile and validate the configuration before any
        # expensive or mutating work (EP->TP reshard, weight packing) so an
        # unknown profile or an impossible `compile_hlo_check` fails fast.
        default_cfg = parse_megakernel_config()
        self.profile: Profile = resolve_driver_profile(
            default_cfg["profile"] if profile is None else profile,
            model_name=self.spec.name,
            fused_pool_io=fused_pool_io,
            fused_pool_io_max_b=fused_pool_io_max_b,
            fused_pool_io_min_b_hi=fused_pool_io_min_b_hi,
            pool_gather=pool_gather,
        )
        self._compile_hlo_check: bool = bool(
            default_cfg["compile_hlo_check"]
            if compile_hlo_check is None
            else compile_hlo_check
        )
        check_compile_hlo_mesh(self.mesh, self._compile_hlo_check)
        self.fused_pool_io: bool | str = self.profile.fused_pool_io
        self.fused_pool_io_max_b: int = int(self.profile.fused_pool_io_max_b)
        self.fused_pool_io_min_b_hi: int = int(self.profile.fused_pool_io_min_b_hi)
        self.pool_gather: bool | None = self.profile.pool_gather

        # Bucket lists: explicit arguments, else the profile's, else defaults.
        self.buckets, self.verify_buckets = _resolve_driver_buckets(
            self.profile, buckets, verify_buckets
        )
        self.max_bucket = max(self.buckets) if self.buckets else 0
        self.max_verify_bucket = max(self.verify_buckets) if self.verify_buckets else 0
        _need(
            bool(self.buckets or self.verify_buckets),
            "at least one of decode buckets or verify_buckets must be non-empty",
        )

        if base_exec is None:
            base_exec = ExecSpec(
                R=r_cnt,
                B=max(self.max_bucket, self.max_verify_bucket, 8),
                T=1,
                stack_e=False,
                cache=CacheExec(paged_kv=True),
            )
        else:
            base_exec = with_cache(
                dataclasses.replace(base_exec, R=r_cnt, stack_e=False), paged_kv=True
            )
        self.base_exec = base_exec

        if weights is None:
            weights = self._prepare_and_pack_weights(
                mode=mode,
                reshard_ep_to_tp=reshard_ep_to_tp,
                expert_placement=expert_placement,
            )
        self.weights = weights
        self.w_flat: list[torch.Tensor] = self.weights["w_flat"]

        self.raw_pools: list[torch.Tensor] = []
        self.num_pages: int = 0
        self.page_size: int = 128
        self.max_pages_per_seq: int = 0
        self.mamba_layer_names: list[str] = []
        self.attn_layer_name: str = ""
        self._ckpt_window: int = 1
        self._zero_roff_1d: torch.Tensor | None = None
        self._runner_token_paddings: tuple[int, ...] = ()
        self._bucket_steps: dict[Any, Any] = {}
        self._bucket_ops: dict[tuple[int, int], Any] = {}
        self._bucket_out_rows: dict[tuple[int, int], int] = {}
        self._bucket_fused_io: dict[tuple[int, int], bool] = {}
        self._bucket_gpr: dict[tuple[int, int], bool] = {}
        self._bound = False
        self._hlo_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_checked_buckets: set[tuple[int, int]] = set()
        self._torchtpu_ir_error: str | None = None
        self._prepared_step: tuple[Any, ...] | None = None
        self._step_mix: dict[str, int] = {}
        self._step_mix_t0: float = time.monotonic()
        self._step_mix_period_s: float = float(
            default_cfg["stats_period_s"] if stats_period_s is None else stats_period_s
        )
        self.ab_calibrator = MKABCalibrator(
            enabled=bool(default_cfg["ab"] if ab is None else ab),
            period_s=self._step_mix_period_s,
        )

    @property
    def _torchtpu_ir_checked(self) -> bool:
        return bool(self._torchtpu_ir_checked_buckets)

    def _build_spec(self, mode: str, r_cnt: int, geom: Any = None) -> Any:
        del r_cnt
        model_def = self._model_def()
        if geom is None:
            e_slots = 0
            if mode == "ep":
                w13_0 = get_routed_experts(self.backbone.layers[0].mlp).w13_weight
                e_slots = int(w13_0.shape[0])
            geom = model_def.derive_geometry(
                model_def.default_hf_config,
                moe_tp=(mode == "tp"),
                e_slots=0 if mode == "tp" else e_slots,
            )
        return model_def.to_model_spec(geom)

    def _prepare_and_pack_weights(
        self,
        *,
        mode: str,
        reshard_ep_to_tp: bool,
        expert_placement: Any,
    ) -> dict[str, Any]:
        if reshard_ep_to_tp and mode == "tp":
            from vllm.distributed import tensor_model_parallel_all_gather

            from vllm_torchtpu.layers.adapter.fused_moe import (
                prebuild_fused_moe_kernel,
            )

            logger.info(
                "Resharding %s MoE experts from EP to TP across %d layers (rank=%d)...",
                self.spec.name,
                self.geom.L,
                self.tp_rank,
            )
            for l_idx in range(self.geom.L):
                reshard_layer_experts_ep_to_tp(
                    self.backbone.layers[l_idx].mlp,
                    tp_rank=self.tp_rank,
                    g=self.geom,
                    ex=self.base_exec,
                    all_gather_fn=lambda t: tensor_model_parallel_all_gather(t, dim=0),
                    prebuild_moe_fn=prebuild_fused_moe_kernel,
                )

        if expert_placement is None:
            try:
                from vllm_torchtpu import (
                    expert_placement as _ep_mod,  # type: ignore[attr-defined]
                )

                expert_placement = _ep_mod
            except ImportError:
                expert_placement = None

        t0 = time.perf_counter()
        weights = pack_weights(
            self.backbone,
            g=self.geom,
            ex=self.base_exec,
            tp_rank=self.tp_rank,
            expert_placement=expert_placement,
        )
        logger.info(
            "Packed %s megakernel weights (%d flat tensors, mode=%s) in %.2fs",
            self.spec.name,
            len(weights["w_flat"]),
            self.mode,
            time.perf_counter() - t0,
        )
        return weights

    @classmethod
    def _validate_runner_hf_config(cls, runner: Any) -> bool:
        hf_cfg = _runner_hf_config(runner)
        if hf_cfg is None:
            return True
        mdef = cls._model_def()
        tc = _cfg_get(hf_cfg, "text_config")
        root = tc if tc is not None else hf_cfg
        has_geom = (
            ("hidden_size" in root)
            if isinstance(root, dict)
            else (getattr(root, "hidden_size", None) is not None)
        )
        if has_geom:
            try:
                g_hf = mdef.derive_geometry(hf_cfg)
            except (MegakernelConfigError, ValueError, KeyError) as exc:
                logger.warning(
                    "Megakernel disabled for %s: %s; skipping.",
                    mdef.__name__,
                    exc,
                )
                return False
            g_def = mdef.derive_geometry(mdef.default_hf_config)
            if g_hf != g_def:
                diffs = [
                    f"{f.name}={getattr(g_hf, f.name)!r} "
                    f"(expected {getattr(g_def, f.name)!r})"
                    for f in dataclasses.fields(g_def)
                    if getattr(g_hf, f.name) != getattr(g_def, f.name)
                ]
                logger.warning(
                    "Megakernel disabled for %s: hf_config geometry differs in %s; "
                    "skipping.",
                    mdef.__name__,
                    ", ".join(diffs),
                )
                return False
            return True
        raw_e = _cfg_get(root, "num_experts")
        if raw_e is not None and int(raw_e) <= 0:
            logger.warning(
                "Megakernel disabled for %s: %s supports MoE checkpoints only; "
                "skipping.",
                mdef.__name__,
                mdef.__name__,
            )
            return False
        return True

    @classmethod
    def maybe_create(cls, runner: Any) -> MegakernelDriver | None:
        """Create a `MegakernelDriver` for `runner` if enabled and compatible."""
        cfg = parse_megakernel_config(runner=runner)
        if not cfg["enabled"]:
            return None
        if not cls._validate_runner_hf_config(runner):
            return None
        try:
            backbone = find_backbone(runner.model)
        except ValueError:
            return None
        if len(getattr(backbone, "layers", ())) != cls._expected_num_layers:
            return None
        if not getattr(runner, "_unified_kv_layout", False):
            logger.warning(
                "Megakernel requires unified KV+GDN cache layout "
                "(_unified_kv_layout=True); skipping."
            )
            return None
        if getattr(runner, "supports_mm_inputs", False):
            logger.warning(
                "Megakernel requires text-only input_ids "
                "(runner.supports_mm_inputs=True passes inputs_embeds; "
                'pass --limit-mm-per-prompt \'{"image":0,"video":0}\' for '
                "text-only serving); skipping."
            )
            return None
        if getattr(runner, "lora_config", None) is not None:
            logger.warning("Megakernel does not support LoRA adapters; skipping.")
            return None
        if not getattr(runner, "_pp_is_first", True) or not getattr(
            runner, "_pp_is_last", True
        ):
            logger.warning(
                "Megakernel requires single-stage pipeline (PP=1); skipping."
            )
            return None
        if _pcp_or_dcp_enabled(runner):
            logger.warning(
                "Megakernel does not support prefill/decode context "
                "parallelism (PCP/DCP); skipping."
            )
            return None
        dp_lockstep_fn = getattr(runner, "_dp_lockstep_enabled", None)
        if callable(dp_lockstep_fn) and dp_lockstep_fn():
            logger.warning("Megakernel does not support EP-DP lockstep; skipping.")
            return None
        if getattr(backbone, "aux_hidden_state_layers", ()):
            logger.warning(
                "Megakernel does not support aux_hidden_state_layers "
                "(EAGLE-3 target capture); skipping."
            )
            return None

        r_cnt = _resolve_tp_size(runner)
        if r_cnt != 8:
            logger.warning(
                "Megakernel requires 8 TPU devices in mesh, got %d; skipping.",
                r_cnt,
            )
            return None

        from vllm.distributed import get_tensor_model_parallel_rank

        tp_rank = int(get_tensor_model_parallel_rank())
        routed0 = get_routed_experts(backbone.layers[0].mlp)
        w13_0 = routed0.w13_weight
        loaded_ep = int(w13_0.shape[0]) < cls._total_routed_experts

        req_mode = str(cfg["mode"]).lower()
        if req_mode not in ("", "ep", "tp"):
            raise ValueError(
                f"Invalid megakernel mode={req_mode!r}; expected 'ep' or 'tp'"
            )
        mode = req_mode if req_mode else ("ep" if loaded_ep else "tp")
        reshard = mode == "tp" and loaded_ep
        mesh = _build_tp_rank_mesh(r_cnt, fallback_mesh=getattr(runner, "mesh", None))
        if int(mesh.devices.size) != r_cnt:
            logger.warning(
                "Megakernel TP-rank mesh has %d devices, expected %d; skipping.",
                int(mesh.devices.size),
                r_cnt,
            )
            return None
        dev_kind = getattr(mesh.devices.flat[0], "device_kind", "") or ""
        try:
            chip_budget(str(dev_kind))
        except MegakernelConfigError as exc:
            logger.warning("Megakernel disabled: %s; skipping.", exc)
            return None
        # Profile-owned settings the user didn't set stay None, so the driver
        # takes them from the effective profile.
        owned = explicit_profile_settings(cfg)
        prof = resolve_driver_profile(cfg["profile"], model_name=cls._model_name)
        buckets, verify_buckets = _resolve_driver_buckets(
            prof, owned["buckets"], owned["verify_buckets"]
        )
        if hasattr(runner, "speculative_config"):
            spec_cfg = getattr(runner, "speculative_config", None)
            num_spec = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
            # When MTP speculative decoding runs with 3 draft tokens (T=4 target
            # verify), steady-state target steps are all T=4 verify so default
            # T=1 decode buckets are skipped unless explicitly requested.
            # Conversely, when num_spec != 3 (e.g. non-speculative T=1 decode),
            # T=4 verify buckets are skipped unless explicitly requested.
            if num_spec == _MTP_SPECULATIVE_TOKENS and not cfg.get(
                "buckets_explicit", False
            ):
                buckets = ()
            elif num_spec != _MTP_SPECULATIVE_TOKENS and not cfg.get(
                "verify_buckets_explicit", False
            ):
                verify_buckets = ()
        if not buckets and not verify_buckets:
            logger.info("Megakernel disabled (both bucket lists empty).")
            return None

        return cls(
            model=runner.model,
            mesh=mesh,
            tp_rank=tp_rank,
            mode=mode,
            buckets=buckets,
            verify_buckets=verify_buckets,
            reshard_ep_to_tp=reshard,
            profile=cfg["profile"],
            fused_pool_io=owned["fused_pool_io"],
            fused_pool_io_max_b=owned["fused_pool_io_max_b"],
            fused_pool_io_min_b_hi=owned["fused_pool_io_min_b_hi"],
            pool_gather=owned["pool_gather"],
            ab=bool(cfg["ab"]),
            stats_period_s=float(cfg["stats_period_s"]),
            compile_hlo_check=bool(cfg["compile_hlo_check"]),
        )

    def bind_caches(self, runner: Any) -> None:
        """Bind the `NATT` shared `float8_e4m3fn` raw pools from `backbone.layers`."""
        _need_layout(
            bool(getattr(runner, "_unified_kv_layout", True)),
            "runner._unified_kv_layout must be True "
            "(unified KV+GDN cache layout required)",
        )
        pb = getattr(self.spec, "cache_backend", None)
        natt = int(pb.num_pools) if pb is not None else int(self.geom.NATT)
        g_per_att = (
            int(pb.recurrent_slots_per_pool)
            if pb is not None
            else int(self.geom.att_every - 1)
        )
        att_every = g_per_att + 1
        pools: list[torch.Tensor] = []
        for r in range(natt):
            att_layer = self.backbone.layers[att_every * r + g_per_att]
            raw = _unwrap(att_layer.self_attn.attn.kv_cache)
            if isinstance(raw, torch.Tensor) and raw.ndim == 6 and raw.shape[0] == 1:
                raw = raw[0]
            pools.append(raw)

        _need_layout(len(pools) == natt, f"expected {natt} pools, got {len(pools)}")
        num_pages = int(pools[0].shape[0])
        kvh_expected = max(1, self.geom.NKV // self.base_exec.R)
        hd4_expected = self.geom.HD // 4
        expected_shape = (num_pages, 2 * kvh_expected, hd4_expected, 4, 128)
        for r, pool in enumerate(pools):
            _need_layout(
                isinstance(pool, torch.Tensor) and pool.dtype == torch.float8_e4m3fn,
                f"pool[{r}] must be float8_e4m3fn tensor, got {type(pool)}",
            )
            _need_layout(
                pool.storage_offset() == 0,
                f"pool[{r}] has non-zero storage_offset={pool.storage_offset()}",
            )
            _need_layout(
                tuple(pool.shape) == expected_shape,
                f"pool[{r}] shape {tuple(pool.shape)} != {expected_shape}",
            )
            for g_idx in range(g_per_att):
                la_layer = self.backbone.layers[att_every * r + g_idx]
                la_cache = _unwrap(la_layer.linear_attn.kv_cache)
                _need_layout(
                    isinstance(la_cache, torch.Tensor)
                    and la_cache.numel() > 0
                    and la_cache.data_ptr() == pool.data_ptr(),
                    f"region {r} group {g_idx} linear_attn.kv_cache must "
                    f"alias pool[{r}] (unified KV+GDN cache layout is required)",
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
            self.max_pages_per_seq = int(getattr(runner, "max_num_blocks_per_req", 2))
        self.mamba_layer_names = [
            self.backbone.layers[g_idx].linear_attn.prefix for g_idx in range(g_per_att)
        ]
        self.attn_layer_name = self.backbone.layers[g_per_att].self_attn.attn.layer_name

        kv_blk = int(
            self.profile.exec_for(self.base_exec, B=self.base_exec.B, T=1, g=self.geom)
            .resolved_tiles.module(GQAModule.name, GQATunables)
            .kv_blk
        )
        aligned_ctx = ((self.max_pages_per_seq * self.page_size) // kv_blk) * kv_blk
        self.base_exec = with_cache(
            dataclasses.replace(
                self.base_exec, ctx=max(self.base_exec.ctx, aligned_ctx)
            ),
            num_pages=self.num_pages,
            page_size=self.page_size,
            max_pages_per_seq=self.max_pages_per_seq,
            paged_kv=True,
        )
        self._runner_token_paddings = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )
        self._build_all_buckets()
        self._bound = True
        fused_bkts = [k for k, v in sorted(self._bucket_fused_io.items()) if v]
        gpr_bkts = [k for k, v in sorted(self._bucket_gpr.items()) if v]
        logger.info(
            "Qwen3.5 megakernel bound to %d unified pools (profile=%s, "
            "num_pages=%d, page_size=%d, max_pages_per_seq=%d, decode_buckets=%s, "
            "verify_buckets=%s, fused_pool_io=%s, fused_buckets=%s, "
            "pool_gather=%s, gpr_buckets=%s)",
            len(self.raw_pools),
            self.profile.name,
            self.num_pages,
            self.page_size,
            self.max_pages_per_seq,
            self.buckets,
            self.verify_buckets,
            self.fused_pool_io,
            fused_bkts,
            self.pool_gather,
            gpr_bkts,
        )

    def _runner_out_rows(self, b: int, m_rows: int) -> int:
        for p in self._runner_token_paddings:
            if p >= b:
                return p
        return m_rows

    def _build_all_buckets(self) -> None:
        self._bucket_steps.clear()
        self._bucket_ops.clear()
        self._bucket_out_rows.clear()
        self._bucket_fused_io.clear()
        self._bucket_gpr.clear()
        self._hlo_checked_buckets.clear()
        self._torchtpu_ir_checked_buckets.clear()
        for b in self.buckets:
            m_rows = max(16, rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            step_1 = self._build_bucket_op(b, tokens_per_seq=1, out_pad_rows=out_rows)
            self._bucket_steps[(1, b)] = step_1
            self._bucket_steps[b] = step_1
            self._bucket_out_rows[(1, b)] = out_rows
        for b in self.verify_buckets:
            m_rows = max(16, rup(b, 16))
            out_rows = self._runner_out_rows(b, m_rows)
            self._bucket_steps[(4, b)] = self._build_bucket_op(
                b, tokens_per_seq=4, out_pad_rows=out_rows
            )
            self._bucket_out_rows[(4, b)] = out_rows

    def _build_bucket_op(
        self,
        bucket_t: int,
        tokens_per_seq: int = 1,
        out_pad_rows: int | None = None,
    ) -> Any:
        """Build the custom op and step wrapper for `(tokens_per_seq, bucket_t)`."""
        ex_cand = self.profile.exec_for(
            self.base_exec, B=bucket_t, T=tokens_per_seq, g=self.geom
        )
        use_fused = should_fuse_pool_io(
            self.spec,
            ex_cand,
            fused_pool_io=self.fused_pool_io,
            max_b_low=self.fused_pool_io_max_b,
            min_b_hi=self.fused_pool_io_min_b_hi,
        )
        ex_cand2 = with_cache(ex_cand, fused_pool_io=use_fused)
        pb = self.spec.cache_backend
        use_gpr = (
            False
            if (use_fused or pb is None)
            else should_use_pool_gather(
                self.spec,
                pb,
                ex_cand2,
                use_fused=use_fused,
            )
        )
        ex = with_cache(ex_cand2, pool_gather=use_gpr)
        is_tpu = self.raw_pools[0].device.type == "tpu"
        ckpt_cols_hlo = 2 if tokens_per_seq == 1 else tokens_per_seq
        bkt_key = (tokens_per_seq, bucket_t)
        if bkt_key not in self._hlo_checked_buckets:
            self._hlo_checked_buckets.add(bkt_key)
            compile_hlo = True if self._compile_hlo_check else None
            ok_hlo, reason_hlo = verify_pooled_step_hlo(
                self.mesh,
                self.spec,
                ex,
                self.w_flat,
                self.raw_pools,
                max_pages_per_seq=self.max_pages_per_seq,
                compile_hlo=compile_hlo,
                ckpt_cols=ckpt_cols_hlo,
            )
            if not ok_hlo:
                raise RuntimeError(
                    "Qwen3.5 megakernel HLO pool-donation check failed for "
                    f"(T={tokens_per_seq}, B={bucket_t}, "
                    f"fused_pool_io={use_fused}, pool_gather={use_gpr}): "
                    f"{reason_hlo}"
                )
            logger.info(
                "Qwen3.5 megakernel HLO pool-donation check passed for "
                "(T=%d, B=%d, fused_pool_io=%s, pool_gather=%s): %s",
                tokens_per_seq,
                bucket_t,
                use_fused,
                use_gpr,
                reason_hlo,
            )

        self._bucket_fused_io[bkt_key] = use_fused
        self._bucket_gpr[bkt_key] = use_gpr
        op_name = (
            f"megakernel_runtime::{self.spec.name}_t{tokens_per_seq}"
            f"_b{bucket_t}_{self.mode}"
        )
        op, _, _, _ = build_pooled_step_op(
            self.mesh,
            self.spec,
            ex,
            op_name=op_name,
            donate_pools=True,
        )
        self._bucket_ops[bkt_key] = op

        m_rows = ex.M
        out_rows = int(out_pad_rows) if out_pad_rows is not None else m_rows
        max_pages = self.max_pages_per_seq
        embed = self.embed_tokens

        def _step_impl(
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
            c0_2d = c0 if c0.ndim == 2 else c0.reshape(-1, 1)
            c1_2d = c1 if c1.ndim == 2 else c1.reshape(-1, 1)
            c2_2d = c2 if c2.ndim == 2 else c2.reshape(-1, 1)
            r_1d = roff.reshape(-1)
            use_ckpt_2d = c0_2d.shape[1] > 1
            ids_b, pos_b, pt_b, ckpt_b, roff_b = prepare_step_inputs_eager(
                input_ids,
                positions,
                block_tables,
                s0,
                c0_2d,
                r_1d,
                s1,
                c1_2d,
                r_1d,
                s2,
                c2_2d,
                r_1d,
                bucket_t,
                tokens_per_seq,
                max_pages,
                True,
                use_ckpt_2d,
            )
            hid = embed(ids_b).to(torch.bfloat16)
            if bucket_t < m_rows:
                hid = F.pad(hid, (0, 0, 0, m_rows - bucket_t))
            hid_out = execute_pooled_step_inplace(
                op,
                pos_b,
                hid,
                pt_b,
                ckpt_b,
                roff_b,
                weights,
                pools,
            )
            if hid_out.shape[0] > out_rows:
                hid_out = hid_out[:out_rows].clone()
            elif hid_out.shape[0] < out_rows:
                hid_out = F.pad(hid_out, (0, 0, 0, out_rows - hid_out.shape[0]))
            return hid_out

        # Give each bucket its own code object: nested closures share `__code__`,
        # and Dynamo caches compiled graphs per code object, so all buckets
        # would otherwise compete for one entry's `cache_size_limit`.
        bkt_name = f"step_t{tokens_per_seq}_b{bucket_t}"
        _step_impl = types.FunctionType(
            _step_impl.__code__.replace(co_name=bkt_name, co_qualname=bkt_name),
            _step_impl.__globals__,
            bkt_name,
            _step_impl.__defaults__,
            _step_impl.__closure__,
        )

        if is_tpu:

            def _on_torchtpu_compile(debug: Any) -> None:
                self._verify_torchtpu_compile_debug(
                    debug,
                    tokens_per_seq=tokens_per_seq,
                    bucket_t=bucket_t,
                )

            compile_opts: dict[str, Any] = {
                "name": bkt_name,
                "debug_callback": _on_torchtpu_compile,
                "serializable": False,
            }
            return torch.compile(
                _step_impl,
                backend="tpu",
                fullgraph=True,
                dynamic=False,
                options=compile_opts,
            )
        return _step_impl

    def _verify_torchtpu_compile_debug(
        self,
        debug: Any,
        *,
        tokens_per_seq: int,
        bucket_t: int,
    ) -> None:
        """Validate `TpuCompileDebug.stablehlo_forward_text` pool donation."""
        fwd_texts = getattr(debug, "stablehlo_forward_text", None) or ()
        if not fwd_texts or not any(str(t).strip() for t in fwd_texts):
            msg = (
                "Qwen3.5 megakernel torch_tpu compiled IR pool-donation "
                f"check failed for (T={tokens_per_seq}, B={bucket_t}): "
                "empty stablehlo_forward_text"
            )
            self._torchtpu_ir_error = msg
            raise RuntimeError(msg)

        ir_text = "\n".join(str(t) for t in fwd_texts)
        kvh = max(1, int(self.raw_pools[0].shape[1]) // 2)
        ok_ir, reason_ir = verify_torchtpu_compiled_step_ir(
            ir_text,
            num_pages=self.num_pages,
            page_size=self.page_size,
            kvh=kvh,
            hd=self.geom.HD,
            num_pools=self.geom.NATT,
        )
        if not ok_ir:
            msg = (
                "Qwen3.5 megakernel torch_tpu compiled IR pool-donation "
                f"check failed for (T={tokens_per_seq}, B={bucket_t}): "
                f"{reason_ir}"
            )
            self._torchtpu_ir_error = msg
            raise RuntimeError(msg)
        self._torchtpu_ir_checked_buckets.add((tokens_per_seq, bucket_t))

    def _ir_required_buckets(
        self, warmed_buckets: Sequence[tuple[int, int]], *, is_eager: bool
    ) -> list[tuple[int, int]]:
        """Buckets whose torch_tpu donation IR check must have run after warmup.

        Non-eager: every warmed bucket. Eager: fused buckets only (they are never
        skipped in eager warmup because fused pool I/O is unsafe without donation).
        """
        return [
            bkt
            for bkt in warmed_buckets
            if not is_eager or self._bucket_fused_io.get(bkt, False)
        ]

    def reset_stats(self) -> None:
        """Reset step-mix counters and A/B latency accumulators."""
        self._step_mix.clear()
        self._step_mix_t0 = time.monotonic()
        self.ab_calibrator.reset()

    @torch.no_grad()
    def warmup(self, runner: Any = None, *, eager_only: bool = False) -> None:
        """Warm compiled bucket steps and run the sentinel pool-write check."""
        if not self._bound:
            raise RuntimeError("bind_caches() must be called before warmup()")
        dev = self.raw_pools[0].device
        ckpt_cols = max(1, int(self._ckpt_window))
        # Sentinel pages for kernel variant index v (see the warmup loop): KV page
        # v and GDN pages [pps * (1 + spv * v), pps * (1 + spv * v + spp * T)),
        # with pps / spp the pool layout's pages and slots per pool and
        # spv = spp * max T slots per variant; snapshot them all. The step inputs
        # carry three recurrent sub-layers per pool (`s0..s2` / `c0..c2`, the
        # Qwen3.5 schedule), so the sentinel writes are wired for three too.
        pb = self.spec.cache_backend
        if pb is None or pb.layout is None:
            raise MegakernelLayoutError(
                f"MegakernelDriver: ModelSpec {self.spec.name!r} has no pool "
                "backend layout"
            )
        pool_lay = pb.layout
        pps = pool_lay.pages_per_slot
        spp = pool_lay.slots_per_pool
        rpp = pool_lay.rec_pages_per_slot
        spv = spp * _MTP_VERIFY_TOKENS
        n_variants = len(
            {
                (
                    T,
                    bool(self._bucket_fused_io.get((T, b), False)),
                    bool(self._bucket_gpr.get((T, b), False)),
                )
                for T, b_list in ((1, self.buckets), (4, self.verify_buckets))
                for b in b_list
            }
        )
        n_sent = pps * (1 + spv * max(n_variants - 1, 0) + spp * _MTP_VERIFY_TOKENS)
        n_save = min(self.num_pages, max(80, n_sent))
        # Never slice-clone or slice-write raw_pools on TPU (PyTorch/XLA lowers
        # slice writes on a 2.30 GiB pool to as_strided_copy_inverse, allocating
        # a full 2.30 GiB temporary buffer per pool).
        saved_page0 = (
            [p[:n_save].clone() for p in self.raw_pools] if dev.type != "tpu" else None
        )
        runner_tok_pads = tuple(
            int(x) for x in getattr(runner, "num_tokens_paddings", ())
        )
        req_block_pairs: list[tuple[int, int]] = []
        if runner is not None:
            attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
            if callable(attn_shapes_fn):
                r_max, b_max, _ = attn_shapes_fn(True)
                req_block_pairs.append((int(r_max), int(b_max)))
                if getattr(runner, "most_model_len", None) is not None:
                    r_most, b_most, _ = attn_shapes_fn(False)
                    pair = (int(r_most), int(b_most))
                    if pair not in req_block_pairs:
                        req_block_pairs.append(pair)
            else:
                r_max = getattr(runner, "num_reqs_max_model_len", None)
                b_max = getattr(runner, "max_num_blocks_per_req", None)
                if r_max is not None and b_max is not None:
                    req_block_pairs.append((int(r_max), int(b_max)))
                if getattr(runner, "most_model_len", None) is not None:
                    r_most = getattr(runner, "num_reqs_most_model_len", None)
                    b_most = getattr(runner, "num_blocks_per_most_len_req", None)
                    if r_most is not None and b_most is not None:
                        pair = (int(r_most), int(b_most))
                        if pair not in req_block_pairs:
                            req_block_pairs.append(pair)
        roff_buf = (
            getattr(runner, "mamba_slot_read_offsets", None)
            if runner is not None
            else None
        )
        assert self._zero_roff_1d is not None
        r_raw = roff_buf if isinstance(roff_buf, torch.Tensor) else self._zero_roff_1d
        uses_mrope = bool(getattr(runner, "uses_mrope", False))
        is_eager = bool(eager_only or getattr(runner, "enforce_eager", False))
        sentinel_checked_variants: set[tuple[int, bool, bool]] = set()
        warmed_buckets: list[tuple[int, int]] = []
        try:
            for T, b_list in ((1, self.buckets), (4, self.verify_buckets)):
                for b in b_list:
                    bkt_key = (T, b)
                    variant_key = (
                        T,
                        bool(self._bucket_fused_io.get(bkt_key, False)),
                        bool(self._bucket_gpr.get(bkt_key, False)),
                    )
                    # Eager mode warms one bucket per kernel variant, except fused
                    # buckets: each fused graph must compile here so its torch_tpu
                    # donation IR check runs (fused correctness needs donation).
                    eager_skip = is_eager and not variant_key[1]
                    if eager_skip and variant_key in sentinel_checked_variants:
                        continue
                    step_fn = self._bucket_steps[bkt_key]
                    warmed_buckets.append(bkt_key)
                    nseq = b // T
                    if runner_tok_pads and req_block_pairs:
                        tok_pad = next(
                            (p for p in runner_tok_pads if p >= b),
                            runner_tok_pads[-1],
                        )
                        shapes_to_warm = [
                            (tok_pad, req_pad, blk_pad)
                            for req_pad, blk_pad in req_block_pairs
                        ]
                    else:
                        shapes_to_warm = [(b, nseq, self.max_pages_per_seq)]

                    for tok_pad, req_pad, blk_pad in shapes_to_warm:
                        if eager_skip and variant_key in sentinel_checked_variants:
                            break
                        ids_b = torch.zeros((tok_pad,), dtype=torch.int32, device=dev)
                        pos_shape = (3, tok_pad) if uses_mrope else (tok_pad,)
                        pos_b = torch.zeros(pos_shape, dtype=torch.int32, device=dev)
                        bt_b = torch.zeros(
                            (req_pad * blk_pad,),
                            dtype=torch.int32,
                            device=dev,
                        )
                        s0_b = torch.zeros((req_pad,), dtype=torch.int32, device=dev)
                        s1_b = torch.zeros((req_pad,), dtype=torch.int32, device=dev)
                        s2_b = torch.zeros((req_pad,), dtype=torch.int32, device=dev)
                        c_cols = max(T, ckpt_cols)
                        if c_cols > 1:
                            c0_b = torch.zeros(
                                (req_pad, c_cols), dtype=torch.int32, device=dev
                            )
                            c1_b = torch.zeros(
                                (req_pad, c_cols), dtype=torch.int32, device=dev
                            )
                            c2_b = torch.zeros(
                                (req_pad, c_cols), dtype=torch.int32, device=dev
                            )
                        else:
                            c0_b, c1_b, c2_b = s0_b, s1_b, s2_b
                        r_call = r_raw

                        # Note: on TPU, the warmup sentinel writes to unallocated
                        # KV/GDN pages per (T, fused, gpr) kernel variant without
                        # restoring them, avoiding slice-writes
                        # (`as_strided_copy_inverse`) on the 2.30 GiB pools.
                        v_idx = len(sentinel_checked_variants)
                        kv_page = v_idx
                        gdn_base = 1 + v_idx * spv
                        if kv_page >= self.num_pages or self.num_pages < pps * (
                            gdn_base + spp * T
                        ):
                            has_t1 = any(k[0] == 1 for k in sentinel_checked_variants)
                            kv_page = 1 if (T > 1 and has_t1) else 0
                            gdn_base = 1 + spp if (T > 1 and has_t1) else 1
                        disjoint_subs = self.num_pages >= pps * (gdn_base + spp * T)
                        if not disjoint_subs:
                            kv_page = 0
                            gdn_base = 1
                        do_sentinel = (
                            variant_key not in sentinel_checked_variants
                            and self.num_pages >= pps * (gdn_base + T)
                        )
                        gdn_p0 = pps * gdn_base
                        sent_kv: list[torch.Tensor] | None = None
                        sent_gdn: list[torch.Tensor] | None = None
                        if do_sentinel:
                            if dev.type != "tpu":
                                for p in self.raw_pools:
                                    p[kv_page].view(torch.uint8).fill_(0x7F)
                                    p[gdn_p0 : gdn_p0 + pps].view(torch.uint8).fill_(
                                        0xFF
                                    )
                            sent_kv = [
                                p[kv_page].view(torch.uint8).cpu().clone()
                                for p in self.raw_pools
                            ]
                            sent_gdn = [
                                p[gdn_p0 : gdn_p0 + rpp].view(torch.uint8).cpu().clone()
                                for p in self.raw_pools
                            ]
                            ids_b[:T] = 1
                            bt_b[0] = kv_page
                            if T > 1:
                                pos_b[..., :T] = torch.arange(
                                    T, dtype=torch.int32, device=dev
                                )
                            sub_step = T if disjoint_subs else 0
                            s0_b[0] = gdn_base
                            s1_b[0] = gdn_base + sub_step
                            s2_b[0] = gdn_base + 2 * sub_step
                            if c_cols > 1:
                                n_c = min(T, c_cols)
                                for sub_i, c_sub in enumerate((c0_b, c1_b, c2_b)):
                                    b0 = gdn_base + sub_i * sub_step
                                    c_sub[0, :n_c] = torch.arange(
                                        b0, b0 + n_c, dtype=torch.int32, device=dev
                                    )
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
                            sentinel_checked_variants.add(variant_key)
                            assert sent_kv is not None and sent_gdn is not None
                            out_cpu = out_b[:b].detach().float().cpu()
                            out_fin_ok = bool(torch.isfinite(out_cpu).all())
                            for idx, p in enumerate(self.raw_pools):
                                kv_ok = not torch.equal(
                                    p[kv_page].view(torch.uint8).cpu(), sent_kv[idx]
                                )
                                gdn_ok = not torch.equal(
                                    p[gdn_p0 : gdn_p0 + rpp].view(torch.uint8).cpu(),
                                    sent_gdn[idx],
                                )
                                if not (kv_ok and gdn_ok and out_fin_ok):
                                    out_nz = int((out_cpu != 0).sum().item())
                                    out_fin = int(torch.isfinite(out_cpu).sum().item())
                                    pool_diag = [
                                        (
                                            i,
                                            int(
                                                (
                                                    rp[kv_page].view(torch.uint8).cpu()
                                                    != 0
                                                )
                                                .sum()
                                                .item()
                                            ),
                                            int(
                                                (
                                                    rp[gdn_p0 : gdn_p0 + rpp]
                                                    .view(torch.uint8)
                                                    .cpu()
                                                    != 0
                                                )
                                                .sum()
                                                .item()
                                            ),
                                        )
                                        for i, rp in enumerate(self.raw_pools[:4])
                                    ]
                                    raise RuntimeError(
                                        "MegakernelDriver: warmup sentinel check "
                                        f"failed on pool[{idx}] for "
                                        f"(T={T}, B={b}, variant={variant_key}) "
                                        f"(kv_ok={kv_ok}, gdn_ok={gdn_ok}, "
                                        f"out_nz={out_nz}/{out_cpu.numel()}, "
                                        f"out_fin={out_fin}, "
                                        f"pool0_3_nz={pool_diag})"
                                    )
            ir_required = (
                self._ir_required_buckets(warmed_buckets, is_eager=is_eager)
                if dev.type == "tpu"
                else []
            )
            if ir_required:
                if self._torchtpu_ir_error is not None:
                    raise RuntimeError(self._torchtpu_ir_error)
                missing_ir = [
                    bkt
                    for bkt in ir_required
                    if bkt not in self._torchtpu_ir_checked_buckets
                ]
                if missing_ir:
                    raise RuntimeError(
                        "Qwen3.5 megakernel torch_tpu compiled IR pool-donation "
                        f"check did not run for buckets {missing_ir} during warmup"
                    )
            self.reset_stats()
        finally:
            if saved_page0 is not None:
                for p, saved in zip(self.raw_pools, saved_page0, strict=True):
                    p[:n_save].copy_(saved)

    def bind_and_warmup(self, runner: Any) -> None:
        """Bind runner KV+GDN cache pools and warm up compiled megakernel buckets.

        Only `MegakernelLayoutError` (an incompatible runtime cache layout)
        disables the megakernel and falls back to the stock path. Configuration
        errors (`MegakernelConfigError`) and any other exception propagate.
        """
        precompile_timed = getattr(runner, "_precompile_timed", None)
        eager_only = bool(getattr(runner, "enforce_eager", False))
        ctx = (
            precompile_timed("megakernel")
            if callable(precompile_timed)
            else contextlib.nullcontext()
        )
        with ctx:
            try:
                self.bind_caches(runner)
            except MegakernelLayoutError as exc:
                logger.warning(
                    "Disabling Qwen3.5 megakernel due to incompatible KV cache "
                    "layout: %s",
                    exc,
                )
                self._bound = False
                if runner is not None and hasattr(runner, "_megakernel"):
                    runner._megakernel = None
                return
            self.warmup(runner, eager_only=eager_only)

    @staticmethod
    def _check_qsl_uniform(qsl_host: Any, num_reqs: int) -> tuple[bool, int, int, str]:
        if num_reqs <= 0 or len(qsl_host) <= num_reqs:
            return False, 0, 0, "stock_n0"
        qsl_slice = qsl_host[: num_reqs + 1]
        qsl_list = (
            qsl_slice.tolist()
            if hasattr(qsl_slice, "tolist")
            else [int(x) for x in qsl_slice]
        )
        if int(qsl_list[0]) != 0:
            return False, num_reqs, 0, "stock_mixed"
        q0 = int(qsl_list[1]) - int(qsl_list[0])
        for i in range(1, num_reqs):
            if int(qsl_list[i + 1]) - int(qsl_list[i]) != q0:
                return False, num_reqs, 0, "stock_mixed"
        if q0 == 1:
            return True, num_reqs, 1, "ok"
        if q0 == 4:
            return True, num_reqs, 4, "ok"
        return False, num_reqs, 0, f"stock_q{q0}"

    @classmethod
    def _inspect_step_detail(
        cls, attn_md: Any, m0: Any, runner: Any = None
    ) -> tuple[bool, int, int, str]:
        """Return `(ok, num_reqs, tokens_per_seq, reason)` without host-device sync."""
        num_reqs: int | None = None
        if runner is not None:
            bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
            if bctx is not None and getattr(bctx, "num_reqs", None) is not None:
                num_reqs = int(bctx.num_reqs)

            has_runner_roff = (
                getattr(runner, "mamba_slot_read_offsets", None) is not None
            )
            comb_key = getattr(runner, "_cached_comb_key", None)
            comb_cpu: Any = getattr(runner, "_combined_request_distribution_cpu", None)
            if has_runner_roff and isinstance(comb_key, tuple) and len(comb_key) == 3:
                _, num_windowed, dist_reqs = (
                    int(comb_key[0]),
                    int(comb_key[1]),
                    int(comb_key[2]),
                )
                if num_reqs is None:
                    num_reqs = dist_reqs
                if dist_reqs <= 0 or num_reqs <= 0:
                    return False, 0, 0, "stock_n0"
                if num_windowed != dist_reqs:
                    return (
                        False,
                        num_reqs,
                        0,
                        "stock_prefill" if num_windowed == 0 else "stock_mixed",
                    )
            elif (
                has_runner_roff
                and comb_cpu is not None
                and isinstance(comb_cpu, torch.Tensor)
                and comb_cpu.numel() >= 6
            ):
                num_windowed = int(comb_cpu[3].item())
                dist_reqs = int(comb_cpu[5].item())
                if num_reqs is None:
                    num_reqs = dist_reqs
                if dist_reqs <= 0 or num_reqs <= 0:
                    return False, 0, 0, "stock_n0"
                if num_windowed != dist_reqs:
                    return (
                        False,
                        num_reqs,
                        0,
                        "stock_prefill" if num_windowed == 0 else "stock_mixed",
                    )

            qsl_host = getattr(runner, "query_start_loc_np", None)
            if qsl_host is None:
                qsl_host = getattr(runner, "query_start_loc_cpu", None)
            if qsl_host is not None and num_reqs is not None:
                ok_qsl, n_qsl, t_qsl, reason_qsl = cls._check_qsl_uniform(
                    qsl_host, num_reqs
                )
                if ok_qsl and not has_runner_roff and t_qsl != 1:
                    return False, n_qsl, 0, "stock_prefill"
                return ok_qsl, n_qsl, t_qsl, reason_qsl

        # Fallback for unit tests passing CPU metadata without a full runner instance.
        # Never read device tensors on TPU; fail closed if host/CPU qsl is absent.
        dist: Any = getattr(m0, "mamba_request_distribution", None)
        if dist is None:
            dist = getattr(attn_md, "mamba_request_distribution", None)
        if dist is None:
            dist = getattr(attn_md, "request_distribution", None)
        if (
            dist is not None
            and isinstance(dist, torch.Tensor)
            and dist.device.type == "cpu"
            and dist.numel() >= 3
        ):
            d_end = int(dist[0].item())
            total_reqs = int(dist[2].item())
            if total_reqs <= 0:
                return False, 0, 0, "stock_n0"
            if d_end != total_reqs:
                return (
                    False,
                    total_reqs,
                    0,
                    "stock_prefill" if d_end == 0 else "stock_mixed",
                )
            if num_reqs is None:
                num_reqs = total_reqs

        num_seqs_attr = getattr(attn_md, "__dict__", {}).get("num_seqs")
        if num_reqs is None and num_seqs_attr is not None:
            if isinstance(num_seqs_attr, torch.Tensor):
                if num_seqs_attr.device.type == "cpu":
                    num_reqs = int(num_seqs_attr.item())
            else:
                num_reqs = int(num_seqs_attr)

        qsl_md = getattr(attn_md, "__dict__", {}).get("query_start_loc_cpu")
        if qsl_md is None:
            qsl_dev: Any = getattr(attn_md, "query_start_loc", None)
            if (
                qsl_dev is not None
                and isinstance(qsl_dev, torch.Tensor)
                and qsl_dev.device.type == "cpu"
            ):
                qsl_md = qsl_dev
        if qsl_md is not None and num_reqs is not None:
            return cls._check_qsl_uniform(qsl_md, num_reqs)

        return False, 0, 0, "stock_no_qsl"

    def can_run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        runner: Any = None,
    ) -> bool:
        """Return `True` if the current forward step should run on the megakernel.

        Note: Called once per `TPUModelRunner.forward_model` invocation; records
        step-mix statistics and A/B arm selection via `_note_step`.
        """
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
        if getattr(self.backbone, "aux_hidden_state_layers", ()):
            return False
        if runner is not None:
            bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
            if getattr(bctx, "position_ids_override", None) is not None:
                return False
            if not getattr(runner, "_unified_kv_layout", True):
                return False
            if getattr(runner, "supports_mm_inputs", False):
                return False
            if getattr(runner, "lora_config", None) is not None:
                return False
            if not getattr(runner, "_pp_is_first", True) or not getattr(
                runner, "_pp_is_last", True
            ):
                return False
            if _pcp_or_dcp_enabled(runner):
                return False
            dp_lockstep_fn = getattr(runner, "_dp_lockstep_enabled", None)
            if callable(dp_lockstep_fn) and dp_lockstep_fn():
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

        if (
            getattr(m0, "mamba_state_indices", None) is None
            and getattr(m0, "mamba_ckpt_indices", None) is None
        ):
            return False

        ok, num_reqs, tokens_per_seq, reason = self._inspect_step_detail(
            attn_md, m0, runner=runner
        )
        if not ok or num_reqs <= 0:
            return self._note_step(reason, False)

        active_rows = num_reqs * tokens_per_seq
        if positions is not None and positions.ndim == 2:
            if int(positions.shape[0]) != 3 or int(positions.shape[1]) < active_rows:
                return self._note_step("stock_mrope", False)
            mrope_buf = getattr(runner, "mrope_positions", None) if runner else None
            mrope_np = getattr(mrope_buf, "np", None)
            if mrope_np is not None and mrope_np.shape[1] >= active_rows:
                if not (
                    np.array_equal(mrope_np[0, :active_rows], mrope_np[1, :active_rows])
                    and np.array_equal(
                        mrope_np[0, :active_rows], mrope_np[2, :active_rows]
                    )
                ):
                    return self._note_step("stock_mrope", False)
            elif positions.device.type == "cpu":
                if not (
                    torch.equal(positions[0, :active_rows], positions[1, :active_rows])
                    and torch.equal(
                        positions[0, :active_rows], positions[2, :active_rows]
                    )
                ):
                    return self._note_step("stock_mrope", False)
            else:
                return self._note_step("stock_mrope", False)
        elif positions is not None and positions.ndim > 2:
            return self._note_step("stock_mrope", False)

        if tokens_per_seq == 4:
            if not self.verify_buckets or active_rows > self.max_verify_bucket:
                return self._note_step(f"stock_t4_n{num_reqs}", False)
            bkt = select_bucket(active_rows, self.verify_buckets)
            step_fn = self._bucket_steps.get((4, bkt))
            if step_fn is None:
                return self._note_step("stock_t4_nobkt", False)
            if self._runner_token_paddings and self._bucket_out_rows.get(
                (4, bkt), input_ids.shape[0]
            ) != int(input_ids.shape[0]):
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
            step_fn = self._bucket_steps.get((1, bkt)) or self._bucket_steps.get(bkt)
            if step_fn is None:
                return self._note_step("stock_t1_nobkt", False)
            if self._runner_token_paddings and self._bucket_out_rows.get(
                (1, bkt), input_ids.shape[0]
            ) != int(input_ids.shape[0]):
                return self._note_step(f"stock_t1_rows{bkt}", False)
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
        actual_run, mix_key, _ = self.ab_calibrator.choose(key, run)
        self._step_mix[mix_key] = self._step_mix.get(mix_key, 0) + 1
        now = time.monotonic()
        if now - self._step_mix_t0 >= self._step_mix_period_s:
            if self.tp_rank == 0:
                total = sum(self._step_mix.values())
                mk_cnt = sum(
                    v for k, v in self._step_mix.items() if k.startswith("mk_")
                )
                logger.info(
                    "Qwen3.5 megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                    now - self._step_mix_t0,
                    mk_cnt,
                    total,
                    100.0 * mk_cnt / max(total, 1),
                    dict(sorted(self._step_mix.items())),
                )
            self._step_mix = {}
            self._step_mix_t0 = now
        return actual_run

    def record_ab_step(self, dt_ms: float, padded_tokens: int | None = None) -> None:
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
        """Execute one full-backbone decode (`T=1`) or MTP verify (`T=4`) step."""
        prepared = self._prepared_step
        self._prepared_step = None
        if prepared is not None:
            step_fn, attn_md, m0, m1, m2, num_reqs, tokens_per_seq, _ = prepared
        else:
            md = get_forward_context().attn_metadata
            attn_md = md[self.attn_layer_name]
            m0 = md[self.mamba_layer_names[0]]
            m1 = md[self.mamba_layer_names[1]]
            m2 = md[self.mamba_layer_names[2]]
            _, num_reqs, tokens_per_seq, _ = self._inspect_step_detail(
                attn_md, m0, runner=runner
            )
            active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
            if tokens_per_seq == 4:
                bkt = select_bucket(
                    max(4, min(active_rows, self.max_verify_bucket)),
                    self.verify_buckets,
                )
                step_fn = self._bucket_steps[(4, bkt)]
            else:
                bkt = select_bucket(
                    max(1, min(active_rows, self.max_bucket)), self.buckets
                )
                step_fn = self._bucket_steps.get((1, bkt)) or self._bucket_steps[bkt]

        out_rows = int(num_tokens if num_tokens is not None else input_ids.shape[0])
        fb_1d = getattr(attn_md, "seq_lens", None)
        if fb_1d is None:
            fb_1d = torch.zeros(
                (max(1, num_reqs),), dtype=torch.int32, device=input_ids.device
            )

        s0 = getattr(m0, "mamba_state_indices", None)
        c0 = getattr(m0, "mamba_ckpt_indices", None)
        if s0 is None:
            s0 = c0 if c0 is not None else fb_1d
        if c0 is None:
            c0 = s0

        s1 = getattr(m1, "mamba_state_indices", None)
        c1 = getattr(m1, "mamba_ckpt_indices", None)
        if s1 is None:
            s1 = c1 if c1 is not None else fb_1d
        if c1 is None:
            c1 = s1

        s2 = getattr(m2, "mamba_state_indices", None)
        c2 = getattr(m2, "mamba_ckpt_indices", None)
        if s2 is None:
            s2 = c2 if c2 is not None else fb_1d
        if c2 is None:
            c2 = s2

        roff = getattr(m0, "mamba_slot_read_offsets", None)
        if roff is None and runner is not None:
            roff = getattr(runner, "mamba_slot_read_offsets", None)
        if roff is None:
            assert self._zero_roff_1d is not None
            roff = self._zero_roff_1d

        active_rows = max(tokens_per_seq, num_reqs * tokens_per_seq)
        if positions is not None and (
            (positions.ndim == 2 and positions.shape[1] >= active_rows)
            or (positions.ndim == 1 and positions.numel() >= active_rows)
        ):
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

    def try_forward(
        self,
        runner: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
    ) -> tuple[bool, Any]:
        """Try to execute the current forward step on the megakernel.

        Returns `(True, hidden_states)` when handled by the megakernel (or timed
        by the A/B calibrator), or `(False, None)` to fall through to the runner's
        default model forward.
        """
        use_mk = self.can_run(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
            runner=runner,
        )
        ab_t0 = time.perf_counter() if self.ab_calibrator.enabled else None
        if use_mk:
            assert input_ids is not None and positions is not None
            num_tok = int(positions.shape[-1])
            out = self(
                input_ids,
                positions,
                num_tokens=num_tok,
                runner=runner,
            )
            if ab_t0 is not None:
                if getattr(getattr(runner, "device", None), "type", None) == "tpu":
                    from vllm_torchtpu.utils import synchronize_tensors

                    synchronize_tensors([out])
                self.record_ab_step(
                    (time.perf_counter() - ab_t0) * 1000.0,
                    padded_tokens=num_tok,
                )
            return True, out

        if (
            ab_t0 is not None
            and self.ab_calibrator.last_decision[0] != "stock_early"
            and positions is not None
            and intermediate_tensors is None
            and callable(getattr(runner, "model", None))
            and not getattr(self.backbone, "aux_hidden_state_layers", ())
        ):
            out = runner.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if getattr(
                getattr(runner, "device", None), "type", None
            ) == "tpu" and isinstance(out, torch.Tensor):
                from vllm_torchtpu.utils import synchronize_tensors

                synchronize_tensors([out])
            self.record_ab_step(
                (time.perf_counter() - ab_t0) * 1000.0,
                padded_tokens=int(positions.shape[-1]),
            )
            return True, out

        return False, None


def _cfg_get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _runner_hf_config(runner: Any) -> Any:
    model_cfg = getattr(runner, "model_config", None) or getattr(
        getattr(runner, "vllm_config", None), "model_config", None
    )
    return getattr(model_cfg, "hf_config", None) if model_cfg is not None else None


class Qwen35Megakernel(MegakernelDriver):
    """Qwen3.5 (`Qwen3_5MoeForConditionalGeneration`) megakernel driver."""

    _model_name = qwen35.QWEN35_MODEL_NAME
    _expected_num_layers = get_model(_model_name).derive_geometry().L
    _total_routed_experts = get_model(_model_name).derive_geometry().E
    _runtime_registry_keys = (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe",
        "qwen3_5",
    )

    @classmethod
    def _validate_runner_hf_config(cls, runner: Any) -> bool:
        if not super()._validate_runner_hf_config(runner):
            return False
        hf_cfg = _runner_hf_config(runner)
        if hf_cfg is None:
            return True
        tc = _cfg_get(hf_cfg, "text_config")
        root = tc if tc is not None else hf_cfg
        raw_e = _cfg_get(root, "num_experts")
        archs = _cfg_get(hf_cfg, "architectures") or ()
        mtype = _cfg_get(hf_cfg, "model_type")
        if (
            (
                "Qwen3_5ForConditionalGeneration" in archs
                or "Qwen3_5ForCausalLM" in archs
                or mtype in ("qwen3_5", "qwen3_5_text")
            )
            and "Qwen3_5MoeForConditionalGeneration" not in archs
            and "Qwen3_5MoeForCausalLM" not in archs
            and mtype not in ("qwen3_5_moe", "qwen3_5_moe_text")
            and not raw_e
        ):
            mdef = cls._model_def()
            logger.warning(
                "Megakernel disabled for %s: %s supports MoE checkpoints only; "
                "skipping.",
                mdef.__name__,
                mdef.__name__,
            )
            return False
        return True


_MODEL_DRIVER_REGISTRY: dict[str, Callable[[Any], Any]] = {
    key: Qwen35Megakernel.maybe_create
    for key in Qwen35Megakernel._runtime_registry_keys
}


def register_megakernel_driver(
    arch_or_model_type: str, factory: Callable[[Any], Any]
) -> None:
    """Register a megakernel driver factory for a model architecture or `model_type`."""
    _MODEL_DRIVER_REGISTRY[arch_or_model_type] = factory


def maybe_create(runner: Any) -> Any:
    """Create a megakernel driver for `runner` if enabled for its model architecture."""
    hf_cfg = _runner_hf_config(runner)
    candidates: list[str] = []
    if hf_cfg is not None:
        archs = _cfg_get(hf_cfg, "architectures") or ()
        candidates.extend(str(a) for a in archs)
        if mtype := _cfg_get(hf_cfg, "model_type"):
            candidates.append(str(mtype))
        txt_cfg = _cfg_get(hf_cfg, "text_config")
        if txt_cfg is not None and (txt_mtype := _cfg_get(txt_cfg, "model_type")):
            candidates.append(str(txt_mtype))
    model_cls = type(getattr(runner, "model", None)).__name__
    if model_cls and model_cls != "NoneType":
        candidates.append(model_cls)
    for key in candidates:
        factory = _MODEL_DRIVER_REGISTRY.get(key)
        if factory is not None:
            return factory(runner)
    if hf_cfg is None:
        return Qwen35Megakernel.maybe_create(runner)
    return None
