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
from ..errors import MegakernelConfigError, MegakernelLayoutError
from ..exec_spec import ExecSpec
from ..pool.config import cache_exec
from ..spec import ModelSpec
from .policy import MTP_VERIFY_TOKENS


@dataclass(frozen=True)
class PoolScheduleInfo:
    """Pool and recurrent-slot geometry derived from `ModelSpec`."""

    num_pools: int
    recurrent_slots_per_pool: int
    att_every: int
    pages_per_slot: int
    slots_per_pool: int
    rec_pages_per_slot: int

    def expected_pool_shape(
        self, num_pages: int, kvh: int, hd: int, page_size: int
    ) -> tuple[int, int, int, int, int]:
        """Return the expected 5-D `(num_pages, 2*kvh, hd//4, 4, page_size)` shape."""
        return (int(num_pages), 2 * int(kvh), int(hd) // 4, 4, int(page_size))


def pool_schedule_info(spec: ModelSpec) -> PoolScheduleInfo:
    """Extract `PoolScheduleInfo` from `spec.cache_backend`."""
    pb = getattr(spec, "cache_backend", None)
    lay = getattr(pb, "layout", None) if pb is not None else None
    if pb is None or lay is None:
        raise MegakernelLayoutError(
            f"MegakernelDriver: ModelSpec {getattr(spec, 'name', '')!r} has no pool "
            "backend layout"
        )
    rec, npools = int(pb.recurrent_slots_per_pool), int(pb.num_pools)
    pps, spp = int(lay.pages_per_slot), int(lay.slots_per_pool)
    return PoolScheduleInfo(npools, rec, rec + 1, pps, spp, int(lay.rec_pages_per_slot))


def check_compile_hlo_mesh(mesh: Any, compile_hlo_check: bool) -> None:
    """Fail at config time if `compile_hlo_check=1` cannot work on `mesh`."""
    if not compile_hlo_check:
        return
    devs = getattr(mesh, "devices", None)
    first = devs.flat[0] if devs is not None and devs.size > 0 else None
    if not isinstance(first, jax.Device):
        raise MegakernelConfigError(
            "MegakernelDriver: compile_hlo_check=1 requires a mesh of real "
            f"jax.Device instances (got {type(first).__name__}); drop "
            "compile_hlo_check on virtual device meshes (compiled IR is checked)"
        )


_KNOWN_DTYPES = (
    *("float8_e4m3fn", "bfloat16", "float32", "float16"),
    *("int32", "int64", "uint8", "bool"),
)
_Sds = jax.ShapeDtypeStruct
_BktMap = Mapping[tuple[int, int], bool]


def _any_dtype_to_jax(dt: Any) -> Any:
    if isinstance(dt, np.dtype):
        try:
            return jnp.dtype(dt)
        except TypeError:
            pass
    s = str(dt).strip()
    for pfx in ("torch.", "jax.numpy.", "numpy."):
        s = s.removeprefix(pfx)
    if s in _KNOWN_DTYPES:
        return getattr(jnp, "bool_" if s == "bool" else s)
    raise TypeError(f"Unsupported dtype for megakernel HLO check: {dt!r}")


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
    r_cnt, g = int(mesh.devices.size), spec.config
    k_ckpt = int(ckpt_cols) if ckpt_cols is not None else ex.T
    g_per_att = max(1, pool_schedule_info(spec).recurrent_slots_per_pool)
    sds = [
        _Sds((r_cnt * int(t.shape[0]), *t.shape[1:]), _any_dtype_to_jax(t.dtype))
        for t in (*w_flat, *raw_pools)
    ]
    return [
        _Sds((ex.B,), jnp.int32),
        _Sds((r_cnt * ex.M, int(getattr(g, "H", 0))), jnp.bfloat16),
        *sds,
        _Sds((ex.NS, int(max_pages_per_seq)), jnp.int32),
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
    has_real = bool(devs.size > 0 and isinstance(devs.flat[0], jax.Device))
    if compile_hlo is True and not has_real:
        dev_t = type(devs.flat[0]).__name__ if devs.size > 0 else "empty"
        raise MegakernelConfigError(
            f"compile_hlo=True requires a Mesh of real jax.Device instances "
            f"(got {dev_t}); on virtual device meshes, XLA compilation is "
            "verified via verify_compiled_step_ir."
        )
    auto_comp = bool(has_real and jax.default_backend() == "tpu" and not ex.interpret)
    do_comp = auto_comp if compile_hlo is None else bool(compile_hlo)
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
    num_pages, page_size = int(raw_pools[0].shape[0]), int(raw_pools[0].shape[-1])
    g, pb = spec.config, spec.cache_backend
    kvh = int(g.nkv(ex.R)) if hasattr(g, "nkv") else 1
    def_k = ("gather_gdn_states", "scatter_pool_states")
    k_names = pb.kernel_names if pb is not None else def_k
    hlo_kw = dict(kvh=kvh, hd=int(g.HD), min_pool_pages=max(16, num_pages))
    fio = cache_exec(ex).FUSED
    return cache.check_pooled_step_hlo(
        hlo_text, page_size, fused_pool_io=fio, kernel_names=k_names, **hlo_kw
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
    num_pages, page_size = int(num_pages), int(page_size)
    kvh, hd, num_pools = int(kvh), int(hd), int(num_pools)
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
        pool_pat = re.compile(rf"%arg\d+\s*:\s*tensor<{pool_sh}>(?:\s*(\{{[^}}]*\}}))?")
        don_pat = re.compile(r"jax\.buffer_donor = true|tf\.aliasing_output\s*=\s*\d+")
        matches = pool_pat.finditer(main_m.group(1))
        n_don = sum(bool(don_pat.search(m.group(1) or "")) for m in matches)
        if n_don < num_pools:
            msg = f"(jax.buffer_donor = true) on @main in compiled IR, found {n_don}"
            return False, f"expected >= {num_pools} donated pool arguments {msg}"

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
    if not (req := ir_required_buckets(warmed, fused_io, is_eager=bool(is_eager))):
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
    buckets: Sequence[int],
    verify_buckets: Sequence[int],
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
    num_pages, max_pages_per_seq = int(num_pages), int(max_pages_per_seq)
    ckpt_cols, is_eager = max(1, int(ckpt_window)), bool(is_eager)
    pinfo = pool_schedule_info(spec)
    pps, spp = pinfo.pages_per_slot, pinfo.slots_per_pool
    spv = spp * MTP_VERIFY_TOKENS
    all_bkts = [(1, b) for b in buckets] + [(4, b) for b in verify_buckets]
    fio, gpr = bucket_fused_io, bucket_gpr
    n_var = len(
        {(k[0], bool(fio.get(k, False)), bool(gpr.get(k, False))) for k in all_bkts}
    )
    base_s = max(1, spp, (n_var + pps) // pps)
    n_save = min(num_pages, max(80, pps * (base_s + spv * max(n_var - 1, 0) + spv)))
    checked: set[tuple[int, bool, bool]] = set()
    warmed_buckets, steps = [], []
    for T, b in all_bkts:
        bkt = (T, int(b))
        v_key = (T, bool(fio.get(bkt, False)), bool(gpr.get(bkt, False)))
        eager_skip = is_eager and not v_key[1]
        if eager_skip and v_key in checked:
            continue
        warmed_buckets.append(bkt)
        if runner_tok_pads and req_block_pairs:
            pads = [int(p) for p in runner_tok_pads if int(p) >= int(b)]
            tok_pad = pads[0] if pads else int(runner_tok_pads[-1])
            shapes = [(tok_pad, int(r), int(k)) for r, k in req_block_pairs]
        else:
            shapes = [(int(b), int(b) // T, max_pages_per_seq)]
        for tok_pad, req_pad, blk_pad in shapes:
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
            w_args = (T, int(b), tok_pad, req_pad, blk_pad, max(T, ckpt_cols), sent)
            steps.append(WarmupStepPlan(*w_args))
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
    """Verify that a sentinel warmup step wrote finite outputs and updated pools."""
    out_arr = np.asarray(out_b, dtype=np.float32)
    out_fin_ok = bool(np.isfinite(out_arr).all())

    def _nz(x: Any) -> int:
        return int(np.count_nonzero(np.asarray(x)))

    for idx in range(len(kv_before)):
        k0, k1 = np.asarray(kv_before[idx]), np.asarray(kv_after[idx])
        g0, g1 = np.asarray(gdn_before[idx]), np.asarray(gdn_after[idx])
        kv_ok, gdn_ok = not np.array_equal(k1, k0), not np.array_equal(g1, g0)
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
