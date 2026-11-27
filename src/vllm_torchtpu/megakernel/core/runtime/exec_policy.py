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
"""The runtime options parser and per-bucket `ExecSpec` setup: profile
resolution (the profile registry) and the cache-planner decisions per bucket.
The bucket lists and `BucketPolicy` are in `policy`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from ..cache import should_fuse_pool_io, should_use_pool_gather
from ..cache_common import require_pool_backend
from ..exec_spec import ExecSpec
from ..modules.gqa import GQAModule, GQATunables
from ..pool.config import FUSED_POOL_IO_MAX_B, FUSED_POOL_IO_MIN_B_HI, with_cache
from ..profile import Profile, get_profile, model_default_profile
from ..spec import ModelSpec
from ._parsing import _parse_bucket_list
from .constants import (
    DEFAULT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
    FALSE_STRS,
    MAX_BUCKET_ROWS,
    MIN_VERIFY_TOKENS,
)
from .policy import parse_mk_buckets

_TRUE_STRS = ("1", "true", "yes", "on")
DEFAULT_FUSED_POOL_IO_MAX_B: int = FUSED_POOL_IO_MAX_B
DEFAULT_FUSED_POOL_IO_MIN_B_HI: int = FUSED_POOL_IO_MIN_B_HI
_SPLIT_STRS = (*FALSE_STRS, "split", "3stage")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Megakernel: {msg}")


def _parse_bool(val: Any) -> bool:
    """Parse a boolean configuration value."""
    if isinstance(val, (bool, int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in _TRUE_STRS or s in FALSE_STRS:
        return s in _TRUE_STRS
    raise ValueError(f"Megakernel: invalid boolean value {val!r}")


def _parse_tri_flag(val: Any, *, true_extra: str, name: str) -> bool | str:
    if isinstance(val, (bool, int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s == "auto":
        return "auto"
    if s in (*_TRUE_STRS, true_extra) or s in _SPLIT_STRS:
        return s in (*_TRUE_STRS, true_extra)
    raise ValueError(
        f"Megakernel: invalid {name}={val!r}; expected 'auto', 'true', or 'false'"
    )


def _parse_int(val: Any, name: str) -> int:
    # A JSON bool or float is rejected like the string "4.5" is.
    if isinstance(val, (bool, float)):
        raise ValueError(f"Megakernel: invalid {name}={val!r}; expected an integer")
    try:
        return int(val)
    except (TypeError, ValueError):
        raise ValueError(
            f"Megakernel: invalid {name}={val!r}; expected an integer"
        ) from None


def _parse_verify_rows(val: Any) -> tuple[int, ...]:
    """Verify bucket rows, range-checked only: `BucketPolicy.resolve` checks
    them against the program's verify widths (`verify_tokens`)."""
    return _parse_bucket_list(
        val,
        valid_fn=lambda b: MIN_VERIFY_TOKENS <= b <= MAX_BUCKET_ROWS,
        label="verify_buckets",
    )


def _parse_fused_pool_io(val: Any) -> bool | str:
    """Parse `fused_pool_io` to `True`, `False`, or `'auto'`."""
    return _parse_tri_flag(val, true_extra="fused", name="fused_pool_io")


def _parse_pool_gather(val: Any) -> bool | str:
    """Parse `pool_gather` to `True`, `False`, or `'auto'` (empty string: off)."""
    return _parse_tri_flag(val, true_extra="gpr", name="pool_gather")


def _apply_profile_defaults(cfg: dict[str, Any], explicit: set[str]) -> None:
    if cfg["profile"] is None:
        return
    try:
        prof = get_profile(cfg["profile"])
    except KeyError as exc:
        raise ValueError(f"Megakernel: {exc.args[0]}") from None
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
        cfg["verify_buckets"] = _parse_verify_rows(prof.verify_buckets)


RUNTIME_OPTION_KEYS = frozenset(
    {
        "profile",
        "mode",
        "buckets",
        "verify_buckets",
        "fused_pool_io",
        "fused_pool_io_max_b",
        "fused_pool_io_min_b_hi",
        "pool_gather",
        "compile_hlo_check",
    }
)
_DEFAULT_OPTIONS: dict[str, Any] = {
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
    "compile_hlo_check": False,
}


def parse_runtime_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Type a program's runtime `options` (`RUNTIME_OPTION_KEYS`; others raise).

    `options` is what the adapter's config parser resolved from
    `VLLM_TPU_MEGAKERNEL` / `additional_config` (it owns that grammar and
    `enabled`); this parses each value (bucket lists, `auto` / on / off flags,
    ints) and fills the named profile's defaults for keys left unset.
    """
    if bad := sorted(set(options) - RUNTIME_OPTION_KEYS):
        raise ValueError(
            f"Megakernel: unknown runtime option {bad[0]!r}; known: "
            f"{sorted(RUNTIME_OPTION_KEYS)}"
        )
    cfg: dict[str, Any] = dict(_DEFAULT_OPTIONS)
    explicit: set[str] = set()
    d = {k: v for k, v in options.items() if v is not None}
    if "profile" in d:
        cfg["profile"] = str(d["profile"]).strip() or None
    if "mode" in d:
        m = str(d["mode"]).strip().lower()
        _need(m in ("", "auto", "ep", "tp"), f"invalid mode={d['mode']!r}")
        cfg["mode"] = "" if m == "auto" else m
    if "buckets" in d:
        cfg["buckets"] = parse_mk_buckets(d["buckets"])
        cfg["buckets_explicit"] = True
    if "verify_buckets" in d:
        cfg["verify_buckets"] = _parse_verify_rows(d["verify_buckets"])
        cfg["verify_buckets_explicit"] = True
    for k, fn in (
        ("fused_pool_io", _parse_fused_pool_io),
        ("pool_gather", _parse_pool_gather),
        ("fused_pool_io_max_b", lambda v: _parse_int(v, "fused_pool_io_max_b")),
        ("fused_pool_io_min_b_hi", lambda v: _parse_int(v, "fused_pool_io_min_b_hi")),
    ):
        if k in d:
            cfg[k] = fn(d[k])
            explicit.add(k)
    if "compile_hlo_check" in d:
        cfg["compile_hlo_check"] = _parse_bool(d["compile_hlo_check"])
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
        raise ValueError(f"Megakernel: {exc.args[0]}") from None
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
) -> ExecSpec:
    """Align `base_exec.ctx` to `kv_blk` and bind paged KV cache geometry."""
    ex0 = profile.exec_for(base_exec, B=base_exec.B, T=1, g=geom)
    kv_blk = ex0.resolved_tiles.module(GQAModule.name, GQATunables).kv_blk
    aligned_ctx = ((max_pages_per_seq * page_size) // kv_blk) * kv_blk
    return with_cache(
        dataclasses.replace(base_exec, ctx=max(base_exec.ctx, aligned_ctx)),
        num_pages=num_pages,
        page_size=page_size,
        max_pages_per_seq=max_pages_per_seq,
        paged_kv=True,
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
    use_gpr = not use_fused and should_use_pool_gather(
        spec, require_pool_backend(spec), ex_cand2, use_fused=False
    )
    return with_cache(ex_cand2, pool_gather=use_gpr), use_fused, use_gpr
