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
"""Shared helpers of the pooled cache steps (argument checks, pool I/O choice)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import kernel
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .fused_io_hook import fused_io_available
from .model_def import find_model
from .planner import extra_inputs
from .pool.config import (
    FUSED_POOL_IO_MAX_B,
    FUSED_POOL_IO_MIN_B_HI,
    POOL_STATE_NAME,
    PoolBackend,
    cache_exec,
    resolved_max_pages_per_seq,
    with_cache,
)
from .spec import ModelSpec

_FUSED_TRUE = ("1", "true", "yes", "on", "fused")
_FUSED_FALSE = ("", "0", "false", "no", "off", "none", "split", "3stage")


def resolve_model_spec(g_or_spec: Any) -> ModelSpec:
    """`g_or_spec` itself if it is a `ModelSpec`, else its `ModelDef`'s spec.

    A geometry resolves through `model_def.find_model` (the `ModelDef` it was
    derived from); a `ModelDef` subclass resolves to its default spec. Model
    names are rejected: a name does not say which geometry to build.
    """
    if isinstance(g_or_spec, ModelSpec):
        return g_or_spec
    m_def = find_model(g_or_spec)
    if m_def is None or isinstance(g_or_spec, str):
        raise TypeError(
            "Expected a ModelSpec, a ModelDef or a geometry derived from a "
            f"ModelDef, got {type(g_or_spec).__name__}"
        )
    return m_def.to_model_spec(None if g_or_spec is m_def else g_or_spec)


def require_pool_backend(model_spec: ModelSpec) -> PoolBackend:
    if model_spec.cache_backend is None:
        raise ValueError(
            f"ModelSpec {model_spec.name!r} does not configure a cache_backend"
        )
    return model_spec.cache_backend


def validate_page_table_shape(
    page_table: Any, ex: ExecSpec, *, rank_folded: bool = True
) -> None:
    """Admit only the page-table shapes the pooled steps can run.

    Each rank needs one `(NS, K)` table with `K >= resolved_max_pages_per_seq`
    (wider tables are trimmed; narrower ones would make the kernel read past
    the table). `(1, NS, K)` is accepted too. With `rank_folded=True` (metadata
    sharded over `"tp"`, see `cache_pooled_local`), `(R * NS, K)` and
    `(R, NS, K)` per-rank tables are also accepted.
    """
    ns, r, max_pt = ex.NS, ex.R, resolved_max_pages_per_seq(ex)
    lead_2d = (ns, r * ns) if rank_folded else (ns,)
    lead_3d = ((1, ns), (r, ns)) if rank_folded else ((1, ns),)
    shape = tuple(page_table.shape)
    ok = (len(shape) == 2 and shape[0] in lead_2d) or (
        len(shape) == 3 and shape[:2] in lead_3d
    )
    if not ok or shape[-1] < max_pt:
        exp = [f"({n}, >={max_pt})" for n in dict.fromkeys(lead_2d)]
        exp += [f"({a}, {b}, >={max_pt})" for a, b in dict.fromkeys(lead_3d)]
        raise ValueError(f"page_table has shape {shape}, expected {' or '.join(exp)}")


def should_use_pool_gather(
    ms: ModelSpec,
    pb: PoolBackend,
    ex: ExecSpec,
    *,
    use_fused: bool,
) -> bool:
    """Return True when non-fused verify should gather GDN states in-kernel (`GPR`)."""
    if use_fused or cache_exec(ex).pool_gather in (False, "split"):
        return False
    if not pb.supports_pool_gather or not pb.has_recurrent_pool:
        if cache_exec(ex).pool_gather is True:
            raise MegakernelConfigError(
                f"ModelSpec {ms.name!r} cache_backend does not support pool_gather"
            )
        return False
    if pb.can_pool_gather is not None:
        ok = bool(pb.can_pool_gather(ex))
        if cache_exec(ex).pool_gather is True and not ok:
            raise MegakernelConfigError(
                f"pool_gather=True is not supported for ExecSpec(B={ex.B}, T={ex.T}, "
                f"page_size={cache_exec(ex).page_size})"
            )
        return ok
    return bool(cache_exec(ex).pool_gather)


def should_fuse_pool_io(
    model_spec: Any,
    ex: ExecSpec,
    *,
    fused_pool_io: bool | str | None = False,
    max_b_low: int = FUSED_POOL_IO_MAX_B,
    min_b_hi: int = FUSED_POOL_IO_MIN_B_HI,
) -> bool:
    """Resolve whether `(model_spec, ex)` should use opt-in 1-launch `fused_pool_io`.

    Args:
      model_spec: `ModelSpec`, or a geometry / `ModelDef` (see `resolve_model_spec`).
      ex: Execution specification (`ex.B`, `cache_exec(ex).fused_pool_io`).
      fused_pool_io:
        - `None`: defer to `bool(cache_exec(ex).fused_pool_io)` (default `False`).
        - `False` (default) / `"false"` / `"0"` / `"no"` / `"off"` / `"none"` /
          `""` / `"split"` / `"3stage"`: use the default split pool I/O pipeline.
        - `True` / `"true"` / `"1"` / `"yes"` / `"on"` / `"fused"`: force 1-launch
          fused I/O.
        - `"auto"`: opt-in selective heuristic that fuses when `ex.B <= max_b_low`
          or `ex.B >= min_b_hi` and uses split I/O in between.
      max_b_low: Upper batch threshold for low-`B` fusion in `"auto"` mode.
      min_b_hi: Lower batch threshold for high-`B` fusion in `"auto"` mode.

    Without a registered fused implementation (`fused_io_available()`), when
    the pool backend doesn't support fused I/O, or when the fused conv window
    (`PoolLayout.fused_conv_rows`) does not fit one conv page, `"auto"` resolves
    to split pool I/O and an explicit `True` raises `MegakernelConfigError`.
    """
    ms = resolve_model_spec(model_spec)
    pb = require_pool_backend(ms)
    if fused_pool_io is None:
        want_fused = explicit_true = bool(cache_exec(ex).fused_pool_io)
    elif isinstance(fused_pool_io, bool):
        want_fused = explicit_true = fused_pool_io
    elif isinstance(fused_pool_io, str):
        s = fused_pool_io.strip().lower()
        if s in _FUSED_TRUE or s in _FUSED_FALSE:
            want_fused = explicit_true = s in _FUSED_TRUE
        elif s == "auto":
            want_fused = max_b_low >= ex.B or min_b_hi <= ex.B
            explicit_true = False
        else:
            raise ValueError(
                f"Unsupported fused_pool_io={fused_pool_io!r}; expected a bool, "
                f"'auto', one of {_FUSED_TRUE} (fused) or one of {_FUSED_FALSE} "
                "(split)"
            )
    else:
        raise TypeError(
            f"Unsupported fused_pool_io type {type(fused_pool_io).__name__}"
        )

    if not pb.supports_fused_io:
        if explicit_true:
            raise MegakernelConfigError(
                f"ModelSpec {ms.name!r} cache_backend does not support fused_pool_io"
            )
        return False
    if want_fused and not fused_io_available():
        if explicit_true:
            raise MegakernelConfigError(
                "fused_pool_io=True needs a registered fused pool I/O "
                "implementation (the megakernel experimental package is not loaded)"
            )
        return False
    lay = pb.layout
    if want_fused and lay.has_recurrent:
        rows = lay.fused_conv_rows(ex.T, ex.R)
        if rows > lay.conv_page_rows:
            if explicit_true:
                raise MegakernelConfigError(
                    f"fused_pool_io=True: the fused conv window ({rows} rows at "
                    f"T={ex.T}, R={ex.R}) exceeds one conv page "
                    f"({lay.conv_page_rows} rows)"
                )
            return False
    return want_fused


def require_fused_donation(ms: ModelSpec, use_fused: bool, donate_pools: bool) -> None:
    """Fail closed: fused pool I/O is only correct with outer pool donation.

    On TPU the fused megakernel writes the pools through unaliased HBM DMAs and
    the step returns the donated input pools as its outputs. Without
    `donate_pools=True` XLA may hand the kernel a copy, silently dropping writes.
    """
    if use_fused and not donate_pools:
        raise MegakernelConfigError(
            f"fused_pool_io for ModelSpec {ms.name!r} requires donate_pools=True "
            "(the fused kernel relies on outer buffer donation for its pool writes)"
        )


@dataclass(frozen=True)
class PooledBuild:
    """What one pooled-step build fixes: the paged `ExecSpec`, the default
    compiled kernel (`call`, plan `p`), the kernel cache and the pool names."""

    mesh: Any
    ms: ModelSpec
    pb: PoolBackend
    ex_paged: ExecSpec
    use_fused: bool
    use_gpr: bool
    extras: Any
    n_extra: int
    get_compiled: Any
    call: Any
    p: Any
    w_names: Any
    s_names: Any
    o_names: Any
    wsh: Any
    has_rec: bool
    rec_slots: int
    pool_key: str
    pt_key: str
    max_pt: int
    info: dict[str, Any]


def default_act_mode(b: PooledBuild) -> str:
    """Sequence activity of a step called without an `active_seqs` input.

    This is `CacheExec.act_mode`, except that without recurrent pools the
    default `"from_ckpt"` (which reads checkpoint slots) means `"all"`.
    `"ref"` is returned as is; it needs a mask, so callers must raise.
    """
    mode = cache_exec(b.ex_paged).act_mode
    return "all" if mode == "from_ckpt" and not b.has_rec else mode


def build_pooled(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool,
    fused_pool_io: bool | str | None,
    max_b_low: int,
    min_b_hi: int,
) -> PooledBuild:
    """Resolve the pool I/O mode of `ex` and build the default kernel.

    Only the resolved paged `ExecSpec` is validated, so a caller override
    (`fused_pool_io`, or GPR turned off by fused I/O) is not rejected for the
    mode it replaces.
    """
    ms = resolve_model_spec(model_spec)
    pb = require_pool_backend(ms)
    use_fused = should_fuse_pool_io(
        ms, ex, fused_pool_io=fused_pool_io, max_b_low=max_b_low, min_b_hi=min_b_hi
    )
    use_gpr = should_use_pool_gather(ms, pb, ex, use_fused=use_fused)
    require_fused_donation(ms, use_fused, donate_pools)
    ex_paged = with_cache(
        ex, paged_kv=True, fused_pool_io=use_fused, pool_gather=use_gpr
    )
    pb.layout.validate(ex_paged)
    # Module extra inputs (`Module.extra_inputs`) follow the pool arguments.
    extras = extra_inputs(ms, ex_paged)
    cache: dict[tuple[str, bool, int, bool], tuple[Any, dict[str, Any]]] = {}

    def get_compiled(
        act_mode: str, ckpt_2d: bool, ckpt_k: int, roff_1d: bool
    ) -> tuple[Any, dict[str, Any]]:
        key = (act_mode, ckpt_2d, ckpt_k, roff_1d)
        if key not in cache:
            ex_k = with_cache(
                ex_paged,
                act_mode=act_mode,
                ckpt_2d=ckpt_2d,
                ckpt_k=ckpt_k,
                roff_1d=roff_1d,
            )
            cache[key] = kernel.build(ms, ex_k)
        return cache[key]

    # The default kernel fixes the weight names and shapes and runs the split
    # path; the fused / GPR paths build their variant from the metadata shapes.
    cx = cache_exec(ex_paged)
    default_ckpt_k = cx.ckpt_k if cx.ckpt_k > 0 else ex_paged.T
    call, info = get_compiled(cx.act_mode, cx.ckpt_2d, default_ckpt_k, cx.roff_1d)
    p = info["plan"]
    return PooledBuild(
        mesh=mesh,
        ms=ms,
        pb=pb,
        ex_paged=ex_paged,
        use_fused=use_fused,
        use_gpr=use_gpr,
        extras=extras,
        n_extra=len(extras),
        get_compiled=get_compiled,
        call=call,
        p=p,
        w_names=p.w_names,
        s_names=p.s_names,
        o_names=p.o_names,
        wsh=info["wshape"],
        has_rec=pb.has_recurrent_pool,
        rec_slots=pb.recurrent_slots_per_pool,
        pool_key=POOL_STATE_NAME,
        pt_key=pb.page_table_state_name,
        max_pt=resolved_max_pages_per_seq(ex_paged),
        info={**info, "use_fused": use_fused, "use_gpr": use_gpr},
    )


__all__ = [
    "PooledBuild",
    "build_pooled",
    "default_act_mode",
    "require_fused_donation",
    "require_pool_backend",
    "resolve_model_spec",
    "should_fuse_pool_io",
    "should_use_pool_gather",
    "validate_page_table_shape",
]
