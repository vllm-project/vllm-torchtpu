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
"""HLO/IR donation checks, pool schedule inspection, and warmup/sentinel plans."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding

from .. import cache
from ..cache_common import require_pool_backend
from ..errors import MegakernelConfigError, MegakernelLayoutError
from ..exec_spec import ExecSpec
from ..spec import ModelSpec
from .constants import MTP_VERIFY_TOKENS
from .contract import Bucket
from .layout_facts import dtype_name


@dataclass(frozen=True)
class PoolScheduleInfo:
    """Pool and recurrent-slot geometry derived from `ModelSpec`."""

    num_pools: int
    recurrent_slots_per_pool: int
    att_every: int
    pages_per_slot: int
    slots_per_pool: int
    rec_pages_per_slot: int


def pool_schedule_info(spec: ModelSpec) -> PoolScheduleInfo:
    """Extract `PoolScheduleInfo` from `spec.cache_backend`."""
    # A named layout decline (not require_pool_backend's ValueError), so the
    # driver records a named layout decline instead of bind_error.
    pb = spec.cache_backend
    if pb is None:
        raise MegakernelLayoutError(
            f"MegakernelDriver: ModelSpec {spec.name!r} has no pool backend layout"
        )
    lay = pb.layout
    rec, npools = pb.recurrent_slots_per_pool, pb.num_pools
    pps, spp = lay.pages_per_slot, lay.slots_per_pool
    # `att_every` counts real recurrent layers (`rec` is at least 1, for the
    # leading `ckpt_indices` dim, also on a KV-only pool).
    return PoolScheduleInfo(npools, rec, spp + 1, pps, spp, lay.rec_pages_per_slot)


def check_compile_hlo_mesh(mesh: jax.sharding.Mesh, compile_hlo_check: bool) -> None:
    """Fail at config time if `compile_hlo_check=1` cannot work on `mesh`."""
    if not compile_hlo_check:
        return
    devs = mesh.devices
    first = devs.flat[0] if devs.size > 0 else None
    if not isinstance(first, jax.Device):
        raise MegakernelConfigError(
            "MegakernelDriver: compile_hlo_check=1 requires a mesh of real "
            f"jax.Device instances (got {type(first).__name__}); drop "
            "compile_hlo_check on virtual device meshes (compiled IR is checked)"
        )


_Sds = jax.ShapeDtypeStruct
_BktMap = Mapping[tuple[int, int], bool]


def _any_dtype_to_jax(dt: Any) -> Any:
    """`dt` (torch / jax / numpy / str) as a jax dtype."""
    return jnp.dtype(dtype_name(dt))


def build_global_avals(
    mesh: Any,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Any,
    raw_pools: Any,
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
) -> list[_Sds]:
    """Construct global `jax.ShapeDtypeStruct` descriptors for HLO lowering/export."""
    r_cnt, g = mesh.devices.size, spec.config
    k_ckpt = ckpt_cols if ckpt_cols is not None else ex.T
    g_per_att = pool_schedule_info(spec).recurrent_slots_per_pool
    sds = [
        _Sds((r_cnt * int(t.shape[0]), *t.shape[1:]), _any_dtype_to_jax(t.dtype))
        for t in (*w_flat, *raw_pools)
    ]
    return [
        _Sds((ex.B,), jnp.int32),
        _Sds((r_cnt * ex.M, g.H), jnp.bfloat16),
        *sds,
        _Sds((ex.NS, max_pages_per_seq), jnp.int32),
        _Sds((g_per_att, ex.NS, k_ckpt), jnp.int32),
        _Sds((g_per_att, ex.NS), jnp.int32),
    ]


def verify_pooled_step_hlo(
    mesh: Any,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Any,
    raw_pools: Any,
    *,
    max_pages_per_seq: int,
    compile_hlo: bool | None = None,
    ckpt_cols: int | None = None,
) -> tuple[bool, str]:
    """Lower or compile `make_flat_pooled_step` and verify pool donation in HLO."""
    devs = mesh.devices
    has_real = devs.size > 0 and isinstance(devs.flat[0], jax.Device)
    if compile_hlo is True and not has_real:
        dev_t = type(devs.flat[0]).__name__ if devs.size > 0 else "empty"
        raise MegakernelConfigError(
            f"compile_hlo=True requires a Mesh of real jax.Device instances "
            f"(got {dev_t}); on virtual device meshes, XLA compilation is "
            "verified via verify_compiled_step_ir."
        )
    auto_comp = has_real and jax.default_backend() == "tpu" and not ex.interpret
    do_comp = auto_comp if compile_hlo is None else compile_hlo
    step_fn, in_sp, out_sp, don = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_sp)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_sp)
    av_kw = dict(max_pages_per_seq=max_pages_per_seq, ckpt_cols=ckpt_cols)
    avals = build_global_avals(mesh, spec, ex, w_flat, raw_pools, **av_kw)
    j_kw = dict(in_shardings=in_sh, out_shardings=out_sh, donate_argnums=don)
    lowered = jax.jit(step_fn, **j_kw, keep_unused=True).lower(*avals)
    hlo_text = lowered.compile().as_text() if do_comp else lowered.as_text("hlo")
    if hlo_text is None:
        raise MegakernelConfigError("pooled step HLO text is unavailable")
    num_pages, page_size = int(raw_pools[0].shape[0]), int(raw_pools[0].shape[-1])
    pb = require_pool_backend(spec)
    # The fused-I/O decision `make_flat_pooled_step` made for this `ex`.
    fio = cache.should_fuse_pool_io(spec, ex, fused_pool_io=None)
    return cache.check_pooled_step_hlo(
        hlo_text,
        page_size,
        layout=pb.layout,
        min_pool_pages=max(16, num_pages),
        fused_pool_io=fio,
        kernel_names=pb.kernel_names,
    )


_COPY_OPS = (
    *("stablehlo.copy", "mhlo.copy", "copy-start("),
    *("dynamic_update_slice", "dynamic-update-slice(", "as_strided_copy_inverse"),
)


def verify_compiled_step_ir(
    ir_text: str,
    *,
    num_pages: int,
    page_size: int,
    kvh: int,
    hd: int,
    num_pools: int,
    hlo_text: str | None = None,
) -> tuple[bool, str]:
    """Verify that a compiled outer forward step donates and aliases all pools."""
    if not ir_text or not ir_text.strip():
        return False, "empty compiled IR"
    calls = ("stablehlo.custom_call", "mhlo.custom_call", "custom-call(")
    if not any(tok in ir_text for tok in calls):
        return False, "missing custom kernel call in compiled IR"
    has_mlir, has_hlo = "func.func" in ir_text, "HloModule" in ir_text
    if not has_mlir and not has_hlo:
        return False, "unrecognized compiled IR format (missing func.func / HloModule)"
    if has_mlir:
        main_m = re.search(r"func\.func\b[^{]*@main\s*\((.*?)\)\s*->", ir_text, re.S)
        if main_m is None:
            return False, "missing func.func @main signature in compiled MLIR"
        pool_sh = f"{num_pages}x{2 * kvh}x{hd // 4}x4x{page_size}xf8E4M3FN"
        pool_pat = re.compile(rf"%arg\d+\s*:\s*tensor<{pool_sh}>")
        don_pat = re.compile(r"jax\.buffer_donor = true|tf\.aliasing_output\s*=\s*\d+")
        # One chunk per argument: its attribute dict may nest braces (sharding).
        args = re.split(r"(?=%arg\d+\s*:)", main_m.group(1))
        n_don = sum(bool(pool_pat.match(a) and don_pat.search(a)) for a in args)
        if n_don < num_pools:
            msg = "(jax.buffer_donor / tf.aliasing_output) on @main in compiled IR"
            return (
                False,
                f"expected >= {num_pools} donated pool arguments {msg}, found {n_don}",
            )

    extra_hlo = [hlo_text] if hlo_text and hlo_text.strip() else []
    for hlo_src in ([ir_text] if has_hlo else []) + extra_hlo:
        hdr = hlo_src.split("ENTRY", 1)[0]
        if "input_output_alias" not in hdr:
            return False, "missing input_output_alias in compiled HLO header"
        aliases = re.findall(r"(?:\{\d*\}\s*:|\{\{\d*\}\s*,\s*\d+)", hdr)
        if len(aliases) < num_pools:
            msg = f"entries in HloModule input_output_alias, found {len(aliases)}"
            return False, f"expected >= {num_pools} {msg}"

    mlir_pat = re.compile(rf"\b(\d+)x{2 * kvh}x{hd // 4}x4x{page_size}xf8E4M3FN\b")
    hlo_suf = f",{2 * kvh},{hd // 4},4,{page_size}]"
    for txt in [ir_text] + extra_hlo:
        for line in txt.splitlines():
            if "func.func" in line or "ENTRY" in line or "custom_call" in line:
                continue
            is_hlo_cp = "= f8e4m3fn[" in line and "copy(" in line
            if not (any(op in line for op in _COPY_OPS) or is_hlo_cp):
                continue
            s_ln = line.strip()
            if any(int(m.group(1)) >= num_pages for m in mlir_pat.finditer(line)):
                return False, f"unexpected full-pool copy in compiled IR: {s_ln}"
            if hlo_suf in line:
                pre = line.split(hlo_suf, 1)[0].rsplit("[", 1)[-1]
                dim_s = pre.rsplit(",", 1)[-1].strip()
                if not dim_s.isdigit() or int(dim_s) >= num_pages:
                    return False, f"unexpected full-pool copy in compiled HLO: {s_ln}"
    return True, "ok"


def ir_required_buckets(
    warmed: Sequence[tuple[int, int]], fused_io: _BktMap, *, is_eager: bool
) -> list[tuple[int, int]]:
    """Buckets whose compiled-IR donation check must have run after warmup."""
    return [b for b in warmed if not is_eager or fused_io.get(b, False)]


def verify_ir_coverage(
    warmed: Sequence[tuple[int, int]],
    fused_io: _BktMap,
    checked: Collection[tuple[int, int]],
    *,
    is_eager: bool = False,
    ir_error: str | None = None,
    model_label: str = "",
) -> None:
    """Verify that every required warmed bucket passed the compiled-IR check."""
    if not (req := ir_required_buckets(warmed, fused_io, is_eager=is_eager)):
        return
    if ir_error is not None:
        raise RuntimeError(ir_error)
    if missing := [bkt for bkt in req if bkt not in checked]:
        pfx = f"{model_label} " if model_label else ""
        raise RuntimeError(
            f"{pfx}megakernel compiled IR pool-donation "
            f"check did not run for buckets {missing} during warmup"
        )


@dataclass(frozen=True)
class SentinelVariantPlan:
    """Sentinel page and slot assignments for one kernel variant."""

    variant_key: tuple[int, bool, bool]
    kv_page: int
    gdn_base: int
    gdn_p0: int
    sub_step: int
    disjoint_subs: bool


@dataclass(frozen=True)
class WarmupStepPlan:
    """Execution plan for one warmup call `(T, B, tok_pad, req_pad, blk_pad)`."""

    tokens_per_seq: int
    bucket_t: int
    tok_pad: int
    req_pad: int
    blk_pad: int
    c_cols: int
    sentinel: SentinelVariantPlan | None


def build_warmup_plan(
    spec: ModelSpec,
    bucket_keys: Sequence[tuple[int, int]],
    bucket_fused_io: _BktMap,
    bucket_gpr: _BktMap,
    *,
    num_pages: int,
    max_pages_per_seq: int,
    ckpt_window: int = 1,
    runner_tok_pads: Sequence[int] = (),
    req_block_pairs: Sequence[tuple[int, int]] = (),
    is_eager: bool = False,
) -> tuple[list[WarmupStepPlan], list[tuple[int, int]], int]:
    """Build `(warmup_steps, warmed_buckets, n_save)` for `warmup()`."""
    pinfo = pool_schedule_info(spec)
    pps, spp = pinfo.pages_per_slot, pinfo.slots_per_pool
    all_bkts = list(bucket_keys)
    spv = spp * max((t for t, _ in all_bkts if t > 1), default=MTP_VERIFY_TOKENS)
    fio, gpr = bucket_fused_io, bucket_gpr
    n_var = len(
        {(k[0], bool(fio.get(k, False)), bool(gpr.get(k, False))) for k in all_bkts}
    )
    base_s = max(1, spp, (n_var + pps) // pps)
    n_save = min(num_pages, max(80, pps * (base_s + spv * max(n_var - 1, 0) + spv)))
    checked: set[tuple[int, bool, bool]] = set()
    warmed_buckets, steps = [], []
    for T, b in all_bkts:
        bkt = (T, b)
        v_key = (T, bool(fio.get(bkt, False)), bool(gpr.get(bkt, False)))
        eager_skip = is_eager and not v_key[1]
        if eager_skip and v_key in checked:
            continue
        warmed_buckets.append(bkt)
        shapes = Bucket(T, b).runner_shapes(
            runner_tok_pads,
            req_block_pairs,
            ckpt_window=ckpt_window,
            max_pages_per_seq=max_pages_per_seq,
        )
        for shape in shapes:
            if eager_skip and v_key in checked:
                break
            v_idx, has_t1 = len(checked), any(k[0] == 1 for k in checked)
            kv_page, gdn_base = 1 + v_idx, base_s + v_idx * spv
            if kv_page >= pps * gdn_base or num_pages < pps * (gdn_base + spp * T):
                kv_page, gdn_base = (
                    (2, base_s + spp) if (T > 1 and has_t1) else (1, base_s)
                )
            disjoint = num_pages >= pps * (gdn_base + spp * T)
            if not disjoint:
                kv_page, gdn_base = 1, 1
            sent: SentinelVariantPlan | None = None
            if v_key not in checked and num_pages >= pps * (gdn_base + T):
                checked.add(v_key)
                sub_s, gdn_p0 = (T if disjoint else 0), pps * gdn_base
                s_args = (v_key, kv_page, gdn_base, gdn_p0, sub_s, disjoint)
                sent = SentinelVariantPlan(*s_args)
            pads = (shape.tok_pad, shape.req_pad, shape.blk_pad, shape.ckpt_cols)
            steps.append(WarmupStepPlan(T, b, *pads, sent))
    return steps, warmed_buckets, n_save


def verify_sentinel_step(
    out_b: Any,
    kv_before: Sequence[Any],
    kv_after: Sequence[Any],
    gdn_before: Sequence[Any],
    gdn_after: Sequence[Any],
    *,
    tokens_per_seq: int,
    bucket_t: int,
    variant_key: tuple[int, bool, bool],
) -> None:
    """Verify that a sentinel warmup step wrote finite outputs and changed the
    bytes of every pool's sentinel KV page and recurrent slot."""
    n_pools = len(kv_before)
    if {len(kv_after), len(gdn_before), len(gdn_after)} != {n_pools}:
        raise ValueError("verify_sentinel_step: pool lists differ in length")
    out_arr = np.asarray(out_b, dtype=np.float32)
    out_fin_ok = bool(np.isfinite(out_arr).all())

    def _nz(x: Any) -> int:
        return int(np.count_nonzero(np.asarray(x)))

    def _changed(before: Any, after: Any) -> bool:
        # Bytes, not values: a NaN pattern left unchanged compares unequal.
        b0, b1 = np.asarray(before), np.asarray(after)
        return b0.shape != b1.shape or b0.tobytes() != b1.tobytes()

    for idx in range(max(n_pools, 1)):
        kv_ok = n_pools == 0 or _changed(kv_before[idx], kv_after[idx])
        gdn_ok = n_pools == 0 or _changed(gdn_before[idx], gdn_after[idx])
        if not (kv_ok and gdn_ok and out_fin_ok):
            out_nz, out_fin = _nz(out_arr), int(np.isfinite(out_arr).sum())
            n_diag = min(4, len(kv_after))
            diag = [(i, _nz(kv_after[i]), _nz(gdn_after[i])) for i in range(n_diag)]
            raise RuntimeError(
                f"MegakernelDriver: warmup sentinel check failed on pool[{idx}] for "
                f"(T={tokens_per_seq}, B={bucket_t}, variant={variant_key}) "
                f"(kv_ok={kv_ok}, gdn_ok={gdn_ok}, out_nz={out_nz}/{out_arr.size}, "
                f"out_fin={out_fin}, pool0_3_nz={diag})"
            )
