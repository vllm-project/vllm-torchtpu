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
"""The runtime config parser and per-bucket `ExecSpec` setup: profile
resolution (the profile registry) and the cache-planner decisions per bucket.
The bucket lists and `BucketPolicy` are in `policy`.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from ..cache import should_fuse_pool_io, should_use_pool_gather
from ..exec_spec import ExecSpec
from ..modules.gqa import GQAModule, GQATunables
from ..pool.config import with_cache
from ..profile import Profile, get_profile, model_default_profile
from ..spec import ModelSpec
from .policy import (
    _FALSE_STRS,
    DEFAULT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
    _need,
    parse_mk_buckets,
    parse_mk_verify_buckets,
)

_TRUE_STRS = ("1", "true", "yes", "on")
DEFAULT_FUSED_POOL_IO_MAX_B: int = 24
DEFAULT_FUSED_POOL_IO_MIN_B_HI: int = 56
_SPLIT_STRS = (*_FALSE_STRS, "split", "3stage")


def _parse_bool(val: Any) -> bool:
    """Parse a boolean configuration value."""
    if isinstance(val, (bool, int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in _TRUE_STRS or s in _FALSE_STRS:
        return s in _TRUE_STRS
    raise ValueError(f"MegakernelDriver: invalid boolean value {val!r}")


def _parse_tri_flag(val: Any, *, true_extra: str, name: str) -> bool | str:
    if isinstance(val, (bool, int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    if s in (*_TRUE_STRS, true_extra) or s in _SPLIT_STRS:
        return s in (*_TRUE_STRS, true_extra)
    raise ValueError(
        f"MegakernelDriver: invalid {name}={val!r}; expected 'auto', 'true', or 'false'"
    )


def _parse_fused_pool_io(val: Any) -> bool | str:
    """Parse `fused_pool_io` to `True`, `False`, or `'auto'`."""
    return _parse_tri_flag(val, true_extra="fused", name="fused_pool_io")


def _parse_pool_gather(val: Any) -> bool | str:
    """Parse `pool_gather` to `True`, `False`, or `'auto'` (empty string: off)."""
    if val is None:
        return "auto"
    return _parse_tri_flag(val, true_extra="gpr", name="pool_gather")


def _apply_profile_defaults(cfg: dict[str, Any], explicit: set[str]) -> None:
    if cfg["profile"] is None:
        return
    try:
        prof = get_profile(cfg["profile"])
    except KeyError as exc:
        raise ValueError(f"MegakernelDriver: {exc.args[0]}") from None
    cfg["profile"] = prof.name
    for k in ("fused_pool_io", "fused_pool_io_max_b", "fused_pool_io_min_b_hi"):
        if k not in explicit:
            v = getattr(prof, k)
            cfg[k] = v if k == "fused_pool_io" else int(v)
    if "pool_gather" not in explicit:
        cfg["pool_gather"] = "auto" if prof.pool_gather is None else prof.pool_gather
    if not cfg["buckets_explicit"] and prof.buckets is not None:
        cfg["buckets"] = parse_mk_buckets(prof.buckets)
    if not cfg["verify_buckets_explicit"] and prof.verify_buckets is not None:
        cfg["verify_buckets"] = parse_mk_verify_buckets(prof.verify_buckets)


_DEFAULT_CFG: dict[str, Any] = {
    "enabled": False,
    "mode": "",
    "profile": None,
    "buckets": DEFAULT_BUCKETS,
    "verify_buckets": DEFAULT_VERIFY_BUCKETS,
    "buckets_explicit": False,
    "verify_buckets_explicit": False,
    "fused_pool_io": False,
    "fused_pool_io_max_b": DEFAULT_FUSED_POOL_IO_MAX_B,
    "fused_pool_io_min_b_hi": DEFAULT_FUSED_POOL_IO_MIN_B_HI,
    "pool_gather": False,
    "ab": False,
    "stats_period_s": 60.0,
    "compile_hlo_check": False,
}


def parse_runtime_config(*sources: Any) -> dict[str, Any]:
    """Parse one or more runtime config sources (str, bool, or mapping) into a dict."""
    cfg: dict[str, Any] = dict(_DEFAULT_CFG)
    explicit: set[str] = set()

    def _apply_dict(d: Mapping[str, Any]) -> None:
        cfg["enabled"] = _parse_bool(d.get("enabled", True))
        if d.get("profile") is not None:
            cfg["profile"] = str(d["profile"]).strip() or None
        if d.get("mode") is not None:
            m = str(d["mode"]).strip().lower()
            _need(m in ("", "auto", "ep", "tp"), f"invalid mode={d['mode']!r}")
            cfg["mode"] = "" if m == "auto" else m
        if d.get("buckets") is not None:
            cfg["buckets"] = parse_mk_buckets(d["buckets"])
            cfg["buckets_explicit"] = True
        if d.get("verify_buckets") is not None:
            cfg["verify_buckets"] = parse_mk_verify_buckets(d["verify_buckets"])
            cfg["verify_buckets_explicit"] = True
        for k, fn in (
            ("fused_pool_io", _parse_fused_pool_io),
            ("pool_gather", _parse_pool_gather),
            ("fused_pool_io_max_b", int),
            ("fused_pool_io_min_b_hi", int),
        ):
            if d.get(k) is not None:
                cfg[k], _ = fn(d[k]), explicit.add(k)
        for k in ("ab", "compile_hlo_check"):
            if d.get(k) is not None:
                cfg[k] = _parse_bool(d[k])
        if d.get("stats_period_s") is not None:
            cfg["stats_period_s"] = float(d["stats_period_s"])

    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            cfg["enabled"] = src
        elif isinstance(src, Mapping):
            _apply_dict(src)
        else:
            s, sl = str(src).strip(), str(src).strip().lower()
            if sl in _FALSE_STRS:
                cfg["enabled"] = False
            elif sl in _TRUE_STRS:
                cfg["enabled"] = True
            elif sl in ("ep", "tp"):
                cfg["enabled"], cfg["mode"] = True, sl
            elif s.startswith("{"):
                parsed = json.loads(s)
                _need(isinstance(parsed, dict), f"expected JSON object in {s!r}")
                _apply_dict(parsed)
            else:
                kv_dict: dict[str, Any] = {}
                for part in (p.strip() for p in s.split(",") if p.strip()):
                    if "=" not in part:
                        raise ValueError(
                            f"Invalid megakernel config entry {part!r} in {s!r}"
                        )
                    k, v = part.split("=", 1)
                    kv_dict[k.strip()] = v.strip()
                _apply_dict(kv_dict)

    _apply_profile_defaults(cfg, explicit)
    cfg["explicit"] = frozenset(explicit)
    return cfg


def explicit_profile_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Profile-owned driver arguments from `cfg`: value when user-set, else `None`."""
    exp = cfg.get("explicit", frozenset())
    v_bkts = tuple(cfg["verify_buckets"]) if cfg["verify_buckets_explicit"] else None
    out: dict[str, Any] = {
        "buckets": tuple(cfg["buckets"]) if cfg["buckets_explicit"] else None,
        "verify_buckets": v_bkts,
    }
    for k in (
        "fused_pool_io",
        "fused_pool_io_max_b",
        "fused_pool_io_min_b_hi",
        "pool_gather",
    ):
        out[k] = cfg[k] if k in exp else None
    return out


def resolve_driver_profile(
    profile: str | Profile | None,
    *,
    model_name: str = "",
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
    pool_gather: bool | str | None = None,
) -> Profile:
    """Resolve the effective profile: named profile plus explicit overrides."""
    try:
        prof = get_profile(profile) if profile else model_default_profile(model_name)
    except KeyError as exc:
        raise ValueError(f"MegakernelDriver: {exc.args[0]}") from None
    _need(
        prof.model is None or prof.model == model_name,
        f"profile {prof.name!r} targets model {prof.model!r}, not {model_name!r}",
    )
    ov: dict[str, Any] = {}
    if fused_pool_io is not None:
        ov["fused_pool_io"] = _parse_fused_pool_io(fused_pool_io)
    if fused_pool_io_max_b is not None:
        ov["fused_pool_io_max_b"] = int(fused_pool_io_max_b)
    if fused_pool_io_min_b_hi is not None:
        ov["fused_pool_io_min_b_hi"] = int(fused_pool_io_min_b_hi)
    if pool_gather is not None:
        pg = _parse_pool_gather(pool_gather)
        ov["pool_gather"] = None if pg == "auto" else pg
    return prof.with_overrides(**ov)


def align_cache_exec(
    base_exec: ExecSpec,
    profile: Profile,
    geom: Any,
    *,
    num_pages: int,
    page_size: int,
    max_pages_per_seq: int,
    dcp: int = 1,
) -> ExecSpec:
    """Align `base_exec.ctx` to `kv_blk` and bind paged KV cache geometry.

    `dcp` is the decode context parallel size: the page table is then the
    rank-local one, and `ctx` the rank-local context it addresses.
    """
    ex0 = profile.exec_for(base_exec, B=base_exec.B, T=1, g=geom)
    kv_blk = ex0.resolved_tiles.module(GQAModule.name, GQATunables).kv_blk
    aligned_ctx = ((max_pages_per_seq * page_size) // kv_blk) * kv_blk
    dcp_kw = {"dcp": int(dcp)} if dcp != 1 else {}
    return with_cache(
        dataclasses.replace(base_exec, ctx=max(base_exec.ctx, aligned_ctx)),
        num_pages=num_pages,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
        paged_kv=True,
        **dcp_kw,
    )


def bucket_exec_spec(
    spec: ModelSpec,
    base_exec: ExecSpec,
    profile: Profile,
    *,
    bucket_t: int,
    tokens_per_seq: int,
    fused_pool_io: bool | str | None = None,
    fused_pool_io_max_b: int | None = None,
    fused_pool_io_min_b_hi: int | None = None,
) -> tuple[ExecSpec, bool, bool]:
    """Build per-bucket `(ExecSpec, use_fused_pool_io, use_pool_gather)`."""
    eff_fused = profile.fused_pool_io if fused_pool_io is None else fused_pool_io
    p_max, p_min = profile.fused_pool_io_max_b, profile.fused_pool_io_min_b_hi
    max_b = p_max if fused_pool_io_max_b is None else fused_pool_io_max_b
    min_b = p_min if fused_pool_io_min_b_hi is None else fused_pool_io_min_b_hi
    ex_cand = profile.exec_for(base_exec, B=bucket_t, T=tokens_per_seq, g=spec.config)
    use_fused = should_fuse_pool_io(
        spec, ex_cand, fused_pool_io=eff_fused, max_b_low=max_b, min_b_hi=min_b
    )
    ex_cand2 = with_cache(ex_cand, fused_pool_io=use_fused)
    pb = spec.cache_backend
    use_gpr = (
        False
        if (use_fused or pb is None)
        else should_use_pool_gather(spec, pb, ex_cand2, use_fused=use_fused)
    )
    return with_cache(ex_cand2, pool_gather=use_gpr), use_fused, use_gpr
