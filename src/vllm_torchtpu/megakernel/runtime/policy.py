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
"""Framework-free bucket policy, config parsing, and per-bucket ExecSpec setup."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..cache import should_fuse_pool_io, should_use_pool_gather
from ..exec_spec import ExecSpec
from ..modules.gqa import GQAModule, GQATunables
from ..ops import rup
from ..pool.config import with_cache
from ..profile import Profile, get_profile, model_default_profile
from ..spec import ModelSpec

DEFAULT_BUCKETS: tuple[int, ...] = (8, 16, 32)
DEFAULT_VERIFY_BUCKETS: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64)
MTP_VERIFY_TOKENS: int = 4
MTP_SPECULATIVE_TOKENS: int = MTP_VERIFY_TOKENS - 1
DEFAULT_FUSED_POOL_IO_MAX_B: int = 24
DEFAULT_FUSED_POOL_IO_MIN_B_HI: int = 56
_TRUE_STRS = ("1", "true", "yes", "on")
_FALSE_STRS = ("0", "false", "no", "off", "none", "")
_SPLIT_STRS = (*_FALSE_STRS, "split", "3stage")


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"MegakernelDriver: {msg}")


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


def _parse_bucket_list(
    val: str | Iterable[int], *, valid_fn: Any, label: str
) -> tuple[int, ...]:
    if not isinstance(val, str):
        if not isinstance(val, Iterable) or isinstance(val, (bytes, bytearray)):
            raise ValueError(f"MegakernelDriver: invalid {label}={val!r}")
        bs = sorted({int(x) for x in val})
        _need(all(valid_fn(b) for b in bs), f"invalid {label}={val!r}")
        return tuple(bs)
    raw = val.strip()
    if raw.lower() in _FALSE_STRS:
        return ()
    bs = sorted({int(x.strip()) for x in raw.replace(":", ",").split(",") if x.strip()})
    _need(bool(bs) and all(valid_fn(b) for b in bs), f"invalid {label}={val!r}")
    return tuple(bs)


def parse_mk_buckets(val: str | Iterable[int]) -> tuple[int, ...]:
    """Parse `T=1` decode batch buckets from a string or iterable (`'0'` disables)."""
    return _parse_bucket_list(
        val, valid_fn=lambda b: 1 <= b <= 64, label="decode buckets"
    )


def parse_mk_verify_buckets(val: str | Iterable[int]) -> tuple[int, ...]:
    """Parse `T=4` MTP verify row buckets from a string or iterable (`'0'` disables)."""
    return _parse_bucket_list(
        val, valid_fn=lambda b: 4 <= b <= 64 and b % 4 == 0, label="verify_buckets"
    )


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


def select_bucket(num_tokens: int, buckets: Sequence[int] = DEFAULT_BUCKETS) -> int:
    """Return the smallest compiled bucket `>= num_tokens`."""
    for b in buckets:
        if b >= num_tokens:
            return b
    raise ValueError(f"num_tokens={num_tokens} exceeds max bucket in {buckets}")


@dataclass(frozen=True)
class BucketPolicy:
    """Resolved `T=1` decode and multi-token verify buckets and output padding."""

    buckets: tuple[int, ...] = DEFAULT_BUCKETS
    verify_buckets: tuple[int, ...] = DEFAULT_VERIFY_BUCKETS
    token_paddings: tuple[int, ...] = ()
    verify_tokens: tuple[int, ...] = (MTP_VERIFY_TOKENS,)

    @property
    def max_bucket(self) -> int:
        return max(self.buckets) if self.buckets else 0

    @property
    def max_verify_bucket(self) -> int:
        return max(self.verify_buckets) if self.verify_buckets else 0

    @classmethod
    def resolve(
        cls,
        profile: Profile,
        buckets: Sequence[int] | None = None,
        verify_buckets: Sequence[int] | None = None,
        *,
        speculative_tokens: int | None = None,
        buckets_explicit: bool | None = None,
        verify_buckets_explicit: bool | None = None,
        token_paddings: Sequence[int] = (),
        verify_tokens: Sequence[int] = (MTP_VERIFY_TOKENS,),
    ) -> BucketPolicy:
        """Resolve `(buckets, verify_buckets)` from explicit args or `profile`."""
        db = DEFAULT_BUCKETS if profile.buckets is None else profile.buckets
        p_ver, vt = profile.verify_buckets, tuple(int(t) for t in verify_tokens)
        dv = DEFAULT_VERIFY_BUCKETS if p_ver is None else p_ver
        eff_v = [int(x) for x in (dv if verify_buckets is None else verify_buckets)]
        b_sorted = tuple(sorted(int(x) for x in (db if buckets is None else buckets)))
        v_sorted = tuple(sorted(x for x in eff_v if not any(x % t for t in vt)))
        b_ok, v_ok = buckets is not None, verify_buckets is not None
        b_exp = b_ok if buckets_explicit is None else buckets_explicit
        v_exp = v_ok if verify_buckets_explicit is None else verify_buckets_explicit
        bad = [x for x in eff_v if x not in v_sorted] if v_exp else []
        _need(not bad, f"verify_buckets {bad} not divisible by verify_tokens={vt}")
        if speculative_tokens is not None:
            if speculative_tokens + 1 in vt and not b_exp:
                b_sorted = ()
            elif speculative_tokens + 1 not in vt and not v_exp:
                v_sorted = ()
        return cls(b_sorted, v_sorted, tuple(int(x) for x in token_paddings), vt)

    def select(
        self, active_rows: int, tokens_per_seq: int = 1, *, clamp: bool = False
    ) -> int:
        """Select the smallest bucket for `(tokens_per_seq, active_rows)`."""
        is_v = tokens_per_seq in self.verify_tokens
        b_list = self.verify_buckets if is_v else self.buckets
        max_b = self.max_verify_bucket if is_v else self.max_bucket
        target = max(tokens_per_seq, min(active_rows, max_b)) if clamp else active_rows
        return select_bucket(target, b_list)

    def out_rows(self, b: int, m_rows: int | None = None) -> int:
        """Return output row count for bucket `b` given `token_paddings`."""
        eff_m = max(16, rup(b, 16)) if m_rows is None else int(m_rows)
        return next((p for p in self.token_paddings if p >= b), eff_m)

    def iter_buckets(self) -> Iterator[tuple[int, int]]:
        """Yield `(tokens_per_seq, bucket_t)` across decode and verify buckets."""
        for b in self.buckets:
            yield (1, b)
        yield from ((t, b) for t in self.verify_tokens for b in self.verify_buckets)


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
    kv_blk = int(ex0.resolved_tiles.module(GQAModule.name, GQATunables).kv_blk)
    aligned_ctx = ((int(max_pages_per_seq) * int(page_size)) // kv_blk) * kv_blk
    return with_cache(
        dataclasses.replace(base_exec, ctx=max(base_exec.ctx, aligned_ctx)),
        num_pages=int(num_pages),
        page_size=int(page_size),
        max_pages_per_seq=int(max_pages_per_seq),
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
    max_b = int(p_max if fused_pool_io_max_b is None else fused_pool_io_max_b)
    min_b = int(p_min if fused_pool_io_min_b_hi is None else fused_pool_io_min_b_hi)
    ex_cand = profile.exec_for(
        base_exec, B=int(bucket_t), T=int(tokens_per_seq), g=spec.config
    )
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
    return with_cache(ex_cand2, pool_gather=use_gpr), bool(use_fused), bool(use_gpr)
