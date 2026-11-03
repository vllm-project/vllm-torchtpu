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
"""First-principles fusion-scope advisor for TPU megakernel evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from math import ceil, gcd, log2
from pathlib import Path
from typing import Any, Literal, get_args

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import errors, exec_spec, ops, planner, spec
from .profiles import default as default_profiles

# fmt: off
FusionScope = Literal["sublayer", "single_layer", "stage", "cross_layer"]
Phase = Literal["decode", "verify", "prefill", "mixed"]
MoEMode = Literal["dense_masked", "binned_vmem"]
FIT_VMEM_HEADROOM_MIB = 6
_DT_A = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0}
_DTYPE_BYTES = {**_DT_A, "nvfp4": 0.5625, "mxfp4": 0.53125, "fp4": 0.5625}
_CHIPS = {"v4": (1, 135.0, 45.0), "v5e": (0, 200.0, 50.0), "v5p": (1, 300.0, 100.0),
          "v6e": (0, 400.0, 100.0), "tpu7x": (0, 300.0, 100.0)}
@dataclass(frozen=True)
class ChipSpec:
    name: str
    cores_per_device: int
    tpu_info: pltpu.TpuInfo
    vmem_capacity_bytes: int
    vmem_usable_bytes: int
    hbm_capacity_bytes: int
    hbm_bw_bytes_per_s: float
    bf16_flops_per_s: float
    native_ocp_fp8: bool
    ici_egress_bw_bytes_per_s: float
    ici_link_bw_bytes_per_s: float
    collective_latency_s: float
    def peak_flops(self, dtype: str) -> float:
        dt, c, t = dtype.lower(), self.cores_per_device, self.tpu_info
        f8 = dt in ("fp8", "nvfp4", "mxfp4", "fp4") and t.fp8_ops_per_second
        ops_s = f8 or (t.int8_ops_per_second if dt == "int8" else t.bf16_ops_per_second)
        return float(ops_s) * c * (0.5 if dt == "fp32" else 1.0)
    def lhs_row_floor(self, dtype: str) -> int:
        f8 = dtype.lower() in ("fp8", "nvfp4", "mxfp4", "fp4") and self.native_ocp_fp8
        return 32 if f8 else (16 if self.tpu_info.mxu_column_size >= 256 else 8)
@dataclass(frozen=True)
class ModelProfile:
    name: str
    num_layers: int
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    num_linear_layers: int = 0
    linear_key_heads: int = 0
    linear_value_heads: int = 0
    num_experts: int = 0
    top_k: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_moe_layers: int = -1
    num_attention_layers: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    linear_key_dim: int = 128
    linear_value_dim: int = 128
    linear_conv_kernel: int = 4
    is_periodic: bool = True
    ar_slots_per_layer: int = 2
    weight_dtype: str = "fp8"
    expert_dtype: str = ""
    kv_dtype: str = "fp8"
    state_dtype: str = "bf16"
    def __post_init__(self) -> None:
        L, ne, ml = self.num_layers, self.num_experts > 0, self.num_moe_layers
        al = self.num_attention_layers or (L - self.num_linear_layers)
        ml = L * ne if ml < 0 else ml
        vals = (al, ml, self.is_periodic and not (0 < ml < L - 1),
                self.moe_intermediate_size or self.intermediate_size * ne,
                self.expert_dtype or self.weight_dtype)
        ks = ("num_attention_layers", "num_moe_layers", "is_periodic",
              "moe_intermediate_size", "expert_dtype")
        for k, v in zip(ks, vals, strict=True):
            object.__setattr__(self, k, v)
@dataclass(frozen=True)
class WorkloadSpec:
    phase: Phase = "decode"
    batch: int = 16
    isl: int = 4096
    osl: int = 1024
    context: int = 0
    prefix_hit_rate: float = 0.0
    spec_tokens: int = 0
    chunk_size: int = 0
    microbatches: int = 0
    tp: int = 8
    pp: int = 1
    dp: int = 1
    ep: int = 1
    kv_split: int = 1
    segment_layers: int = 0
    moe_mode: MoEMode = "dense_masked"
    vmem_alias: bool = True
    vmem_fit_headroom: bool = False
    tiles: exec_spec.Tiles | None = None
    op_overhead_us: float = 0.0
@dataclass(frozen=True)
class ScopeEstimate:
    scope: FusionScope
    feasible: bool
    reason: str
    ops_per_step: int
    vmem_static_mib: float
    t_vmem_max: int
    n_splits: int
    w_step_gb: float
    w_crit_gb: float | None
    act_hbm_mb: float
    moe_flop_factor: float
    roof_floor_us: float
    step_bound_us: float
    break_even_op_overhead_us: float = 0.0
@dataclass(frozen=True)
class AdvisorReport:
    model: str
    chip: str
    phase: Phase
    active_tokens: int
    effective_context: int
    vmem_budget_mib: float
    hbm_fits: bool
    hbm_used_gb: float
    hbm_capacity_gb: float
    ridge_dense_flop_per_byte: float
    step_arithmetic_intensity: float
    asymptotic_decode_intensity: float | None
    recommended_scope: FusionScope
    build_megakernel: bool
    rationale: str
    scopes: tuple[ScopeEstimate, ...]
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
def fit_vmem_budget_bytes(tiles: exec_spec.Tiles | None = None) -> int:
    return max(0, (tiles or default_profiles.TPU_V7X).vmem_mib - 6) << 20
def _i(m: Mapping[str, Any], *ks: str, d: int = 0) -> int:
    return next((int(m[k]) for k in ks if m.get(k)), d)
_MLA: dict[str, Any] = {"kv_lora_rank": 512, "q_lora_rank": 1536,
                        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64}
_K3: dict[str, Any] = {**_MLA, "moe_intermediate_size": 3072, "num_moe_layers": 92,
                       "shared_expert_intermediate_size": 6144, "expert_dtype": "mxfp4"}
_DS: dict[str, Any] = {**_MLA, "num_experts": 256, "top_k": 8,
                       "moe_intermediate_size": 2048,
                       "shared_expert_intermediate_size": 2048}
MODEL_PRESETS: Mapping[str, ModelProfile] = {p.name: p for p in (
    ModelProfile("qwen3.5-0.8b", 24, 1024, 3584, 8, 2, 256, 18, 16, 16),
    ModelProfile("qwen3.5-397b", 60, 4096, 1024, 32, 2, 256, 45, 16, 64, 512, 10, 1024,
                 1024),
    ModelProfile("qwen3.8-2.4t", 92, 8192, 2048, 64, 4, 256, 69, 16, 128, 512, 10, 2048,
                 2048, expert_dtype="nvfp4"),
    *(ModelProfile(n, 93, 7168, 33792, 96, 96, 128, 69, 96, 96, 896, 16, **_K3)
      for n in ("kimi-k3.1-pro", "kimi-k3-pro")),
    ModelProfile("kimi-k3", 93, 7168, 33792, 96, 96, 128, 69, 96, 96, 384, 8, **_K3),
    ModelProfile("deepseek-v3", 61, 7168, 18432, 128, 128, 128, num_moe_layers=58,
                 **_DS),
    ModelProfile("glm-5.2", 78, 6144, 16384, 96, 96, 128, num_moe_layers=75, **_DS),
    ModelProfile("mistral-large", 88, 12288, 28672, 96),
    ModelProfile("llama-3.1-405b", 126, 16384, 53248, 128),
    ModelProfile("mixtral-8x22b", 56, 6144, 16384, 48, num_experts=8, top_k=2,
                 weight_dtype="bf16"),
    ModelProfile("llama-4-maverick", 48, 5120, 8192, 40, num_experts=128, top_k=1,
                 shared_expert_intermediate_size=8192),
    ModelProfile("gpt-oss-120b", 36, 2880, 2880, 64, head_dim=64, num_experts=128,
                 top_k=4, expert_dtype="mxfp4"))}
def resolve_chip_spec(chip: ChipSpec | str = "tpu7x", megacore: bool = False,
                      ici_bw_gbps: float | None = None,
                      collective_latency_us: float | None = None) -> ChipSpec:
    if type(chip) is ChipSpec:
        return chip
    raw, info = str(chip).strip().lower().removeprefix("tpu_"), None
    if raw == "auto":
        try:
            raw = (info := pltpu.get_tpu_info()).chip_version.value.lower()
        except Exception as exc:
            raise errors.MegakernelConfigError("chip='auto' requires TPU") from exc
    nc = int(info.num_cores) if info is not None else 1 + megacore
    canon = "tpu7x" if raw in ("7x", "v7x", "tpu7x") else raw
    if canon not in _CHIPS or (megacore and not _CHIPS[canon][0]):
        raise errors.MegakernelConfigError(f"Invalid chip={chip!r} or Megacore")
    _, eg_g, lk_g = _CHIPS[canon]
    cv = pltpu.ChipVersion(canon.removeprefix("tpu"))
    ti = info or pltpu.get_tpu_info_for_chip(cv, nc)
    vc, hb = int(ti.vmem_capacity_bytes) * nc, int(ti.hbm_capacity_bytes) * nc
    return ChipSpec(
        f"{canon}_megacore" if nc == 2 else canon, nc, ti, vc, vc - (2 << 20), hb,
        float(ti.mem_bw_bytes_per_second) * nc, float(ti.bf16_ops_per_second) * nc,
        bool(ti.is_matmul_supported(jnp.float8_e4m3fn, jnp.float8_e4m3fn)),
        (eg_g * nc if ici_bw_gbps is None else ici_bw_gbps) * 1e9, lk_g * 1e9,
        (1.0 if collective_latency_us is None else collective_latency_us) * 1e-6)
def model_profile_from_hf_config(
    raw_cfg: Mapping[str, Any], name: str = "model", weight_dtype: str | None = None,
    expert_dtype: str | None = None, kv_dtype: str = "fp8", state_dtype: str = "bf16",
) -> ModelProfile:
    c: Mapping[str, Any] = raw_cfg.get("text_config") or raw_cfg
    la, q = c.get("linear_attn_config") or {}, raw_cfg.get("quantization_config") or {}
    cm, L, H = {**la, **c}, _i(c, "num_hidden_layers"), _i(c, "hidden_size")
    nh, iv = _i(c, "num_attention_heads"), _i(c, "full_attention_interval")
    if min(L, H, nh) < 1:
        raise errors.MegakernelConfigError("Model layers, H, and heads must be >= 1")
    ltypes, kda = c.get("layer_types") or (), la.get("kda_layers") or ()
    n_lin = len(kda) or (L - L // iv if iv > 1 else 0)
    is_per = bool(kda) or iv < 2 or L % iv == 0
    if len(ltypes) == L:
        n_lin = sum(any(k in str(t) for k in ("lin", "mamba", "kda")) for t in ltypes)
        g = gcd(L - n_lin, L)
        is_per = list(ltypes[: L // g]) * g == list(ltypes) if 0 < n_lin < L else True
    ne = _i(c, "num_experts", "n_routed_experts", "num_local_experts")
    d_pfx = _i(c, "first_k_dense_replace") or len(c.get("mlp_only_layers") or ())
    td, bits = raw_cfg.get("torch_dtype", "bf16"), q.get("bits") or q.get("w_bit")
    t = f"{q.get('quant_method', '')} {q.get('format', '')} {td}".lower()
    fb = "fp8" if bits == 8 or any(k in t for k in ("fp8", "float8", "e4m3")) else (
        "fp32" if "32" in t else "bf16")
    inf = "nvfp4" if "nvfp4" in t else ("mxfp4" if "mxfp4" in t or bits == 4 else fb)
    wd = str(weight_dtype or raw_cfg.get("weight_dtype") or inf).lower()
    ed = str(expert_dtype or q.get("expert_dtype") or raw_cfg.get("expert_dtype") or wd)
    ffn_i = _i(c, "intermediate_size", "ffn_dim", d=4 * H)
    mi, n_sh = _i(c, "moe_intermediate_size", d=ffn_i * (ne > 0)), _i(
        c, "n_shared_experts", "num_shared_experts")
    return ModelProfile(
        name, L, H, ffn_i, nh, _i(c, "num_key_value_heads", d=nh),
        _i(c, "v_head_dim", "head_dim", d=H // nh), n_lin,
        _i(cm, "linear_num_key_heads", "num_k_heads", "num_heads"),
        _i(cm, "linear_num_value_heads", "num_v_heads", "num_heads"), ne,
        _i(c, "num_experts_per_tok", "num_selected_experts") * (ne > 0), mi,
        _i(c, "shared_expert_intermediate_size", d=n_sh * mi),
        max(0, L - d_pfx) * (ne > 0), 0, _i(c, "kv_lora_rank"), _i(c, "q_lora_rank"),
        _i(c, "qk_nope_head_dim"), _i(c, "qk_rope_head_dim"),
        _i(cm, "linear_key_head_dim", "head_k_dim", d=128),
        _i(cm, "linear_value_head_dim", "head_v_dim", d=128),
        _i(cm, "linear_conv_kernel_dim", "short_conv_kernel_size", d=4),
        is_per, 2, wd, ed.lower(), kv_dtype.lower(), state_dtype.lower())
def resolve_model_profile(
    target: Any, weight_dtype: str | None = None, expert_dtype: str | None = None,
    kv_dtype: str | None = None, state_dtype: str | None = None,
) -> ModelProfile:
    if type(target) is ModelProfile:
        return target
    dts = (weight_dtype, expert_dtype, kv_dtype or "fp8", state_dtype or "bf16")
    if type(target) is dict:
        return model_profile_from_hf_config(target, "model", *dts)
    s, p = str(target).strip(), Path(str(target).strip())
    if p.is_file() or (not s.endswith(".json") and (p := p / "config.json").is_file()):
        return model_profile_from_hf_config(
            json.loads(p.read_text("utf-8")), p.parent.name or p.stem, *dts)
    tl = s.lower().split("/")[-1]
    prof = MODEL_PRESETS.get(
        tl, next((v for k, v in MODEL_PRESETS.items() if tl.startswith(f"{k}-")), None))
    if prof is not None and not s.endswith(".json"):
        w, kv = (weight_dtype or prof.weight_dtype), (kv_dtype or prof.kv_dtype)
        ed = (expert_dtype or weight_dtype or prof.expert_dtype).lower()
        st = (state_dtype or prof.state_dtype).lower()
        return replace(prof, weight_dtype=w.lower(), expert_dtype=ed,
                       kv_dtype=kv.lower(), state_dtype=st)
    raise errors.MegakernelConfigError(f"Unknown model or missing path: {target!r}")
def advise(profile: ModelProfile, workload: WorkloadSpec,
           chip: ChipSpec | str = "tpu7x", plan_error: str = "") -> AdvisorReport:
    pr, wl, tl = profile, workload, workload.tiles
    L, H, va, ne = pr.num_layers, pr.hidden_size, wl.vmem_alias, pr.num_experts
    al, ll, ml = pr.num_attention_layers, pr.num_linear_layers, pr.num_moe_layers
    tp, pp, dp, ep, kvs = wl.tp, wl.pp, wl.dp, wl.ep, wl.kv_split
    ovh, vhd, lrk = wl.op_overhead_us, pr.head_dim, pr.kv_lora_rank
    nh, sh_i, tk = pr.num_attention_heads, pr.shared_expert_intermediate_size, pr.top_k
    sp, qr, qrk = wl.spec_tokens, pr.q_lora_rank, pr.qk_rope_head_dim
    pos = (L, H, nh, vhd, wl.batch, tp, pp, dp, ep, kvs, wl.isl)
    n0 = (al, ll, ml, wl.osl, wl.context, sp, wl.chunk_size, lrk, wl.microbatches,
          wl.segment_layers, qr, qrk, pr.qk_nope_head_dim)
    dts = (pr.weight_dtype, pr.expert_dtype, pr.kv_dtype, pr.state_dtype)
    if min(*pos) < 1 or ovh < 0.0:
        raise errors.MegakernelConfigError("Dimensions and parallelism must be >= 1")
    p_ok = min(*n0) >= 0 and (tp * dp) % ep == 0 and tp % kvs == 0 and pp <= L
    e_ok = tk == 0 if ne == 0 else (1 <= tk <= ne and ne % ep == 0)
    t_ok = 0.0 <= wl.prefix_hit_rate < 1.0 and not (tl and tl.vmem_mib < 2)
    if not (p_ok and e_ok and t_ok and al + ll <= L and ml <= L
            and all(d.lower() in _DTYPE_BYTES for d in dts)):
        raise errors.MegakernelConfigError("Invalid parallelism, layers, or dtype")
    bw_w, bw_e, bw_kv, bw_st = (_DTYPE_BYTES[d.lower()] for d in dts)
    cs, ar_hops = resolve_chip_spec(chip), round(log2(tp))
    c_unc, b_loc = max(1, round(wl.isl * (1 - wl.prefix_hit_rate))), ceil(wl.batch / dp)
    t_ch, l_loc = wl.chunk_size or c_unc, ceil(L / pp)
    eff_ctx = wl.context or (wl.isl + (0 if wl.phase == "prefill" else wl.osl // 2))
    tv_tok = max(1, sp) + 1 if wl.phase == "verify" else (sp + 1 if sp > 0 else 1)
    t_p = wl.chunk_size or ceil(b_loc * c_unc / max(1, wl.microbatches or pp))
    t_vm = b_loc * (1 if wl.phase == "decode" else tv_tok)
    act_t = t_p if wl.phase == "prefill" else t_vm + t_ch * (wl.phase == "mixed")
    l_attn, l_moe = min(l_loc, ceil(al * l_loc / L)), min(l_loc, ceil(ml * l_loc / L))
    tok_b, l_lin = act_t * H * 2.0, min(l_loc - l_attn, ceil(ll * l_loc / L))
    qn, kd, vd = pr.qk_nope_head_dim or vhd, pr.linear_key_dim, pr.linear_value_dim
    nh_r, kvh_r = max(1, ceil(nh / tp)), max(1, ceil(pr.num_kv_heads / (tp // kvs)))
    q_dim, q_tot = nh_r * vhd, nh_r * (qn + qrk)
    p_mla = (qr * (H + q_tot) if qr else H * q_tot) + H * (lrk + qrk + q_dim)
    p_attn = p_mla + lrk * nh_r * (qn + vhd) if lrk else H * 2 * (q_dim + kvh_r * vhd)
    l_hs = (pr.linear_key_heads, pr.linear_value_heads)
    kh_r, vh_r = (max(1, ceil(x / tp)) if l_lin else 0 for x in l_hs)
    z_r, qkv_r, ep_tp = vh_r * vd, 2 * kh_r * kd + vh_r * vd, gcd(ep, tp)
    p_lin, e_loc = H * (qkv_r + 2 * z_r) * (l_lin > 0), (ne // ep) * (l_moe > 0)
    p_dffn = 3 * H * max(1, ceil(pr.intermediate_size / tp))
    p_sh, ep_dp = 3 * H * ceil(sh_i / tp), ep // ep_tp
    i_exp_r = ceil(pr.moe_intermediate_size / (tp // ep_tp)) * (l_moe > 0)
    p_exp, ffn_s = 3 * H * i_exp_r, (1024 * max(1, i_exp_r or H)) if l_moe else 2 << 20
    p_base = l_attn * p_attn + l_lin * p_lin + (l_loc - l_moe) * p_dffn + l_moe * p_sh
    w_static, exp_b = p_base * bw_w + l_moe * H * ne * 2.0, p_exp * bw_e
    kv_tok = ((lrk + qrk) if lrk else 2.0 * kvh_r * vhd) * bw_kv / kvs
    s_kv_b, bw_hbm = l_attn * b_loc * eff_ctx * kv_tok, cs.hbm_bw_bytes_per_s
    s_rec_rw = 2.0 * l_lin * b_loc * (z_r * kd + qkv_r * pr.linear_conv_kernel) * bw_st
    vu = cs.vmem_usable_bytes
    v_raw = (tl.vmem_mib << 20 if tl else vu) - (6 << 20) * wl.vmem_fit_headroom
    hbm_used = w_static + l_moe * e_loc * exp_b + s_kv_b + 0.5 * s_rec_rw
    hbm_fits, v_bud = hbm_used <= cs.hbm_capacity_bytes, min(vu, max(0, v_raw))
    s1 = int((max(p_attn, p_lin) + (p_sh if l_moe else p_dffn)) * bw_w)
    gqa_s, gdn_s = (3072 * max(vhd, lrk) if l_attn else 0), 16 * min(vh_r, 8) * kd * vd
    r_lhs, t_pre = cs.lhs_row_floor(pr.expert_dtype if l_moe else pr.weight_dtype), (
        {"prefill": act_t, "mixed": t_ch}.get(wl.phase, 0))
    d_mla = (act_t - t_pre) * (2 * lrk + qrk) + t_pre * (qn + qrk + vhd)
    f_kv = 2.0 * l_attn * eff_ctx * (nh_r * d_mla if lrk else 2.0 * act_t * q_dim)
    f_dense = 2.0 * act_t * (p_base + l_moe * H * ne) + f_kv
    ar_vol, n_ar = 2.0 * (tp - 1) / tp * tok_b, pr.ar_slots_per_layer * l_loc
    eg_bw, lat_c = cs.ici_egress_bw_bytes_per_s, cs.collective_latency_s
    a2a_t = 2.0 * (ep_dp - 1) / ep_dp * max(1, tk) * tok_b / eg_bw if ep_dp > 1 else 0.0
    t_pp_s = (tok_b / cs.ici_link_bw_bytes_per_s + lat_c) if pp > 1 else 0.0
    n_seg = ceil(l_loc / wl.segment_layers) if 0 < wl.segment_layers < l_loc else 1
    f_moe = 2.0 * act_t * ep_dp * l_moe * tk * p_exp * e_loc / ne if e_loc else 0.0
    pk_w, pk_e = cs.peak_flops(pr.weight_dtype), cs.peak_flops(pr.expert_dtype)
    no_x = pp > 1 or n_seg > 1 or L == 1 or bool(plan_error) or not pr.is_periodic
    d_1l, e_1l = min(4 << 20, s1), min(4 << 20, int(2 * exp_b))
    d_m = (tl.nbd * tl.bank) if tl else min(28 << 20, 2 * s1)
    e_m = (tl.nbe * tl.ebank) if tl else (10 << 20)
    ests, raw_map, sub_ops, sub_pure = [], {}, 2 * l_loc, 0.0
    for sc in ("sublayer", "single_layer", "stage", "cross_layer"):
        feas, rsn = True, "ok"
        if sc == "stage" and pp == 1 and n_seg == 1:
            feas, rsn = False, "Identical to cross_layer when pp == 1"
        elif sc == "cross_layer" and no_x:
            feas, rsn = False, plan_error or "Bounded by stage/1L or non-periodic"
        if sc == "sublayer":
            v_st, alpha, n_sp = max(1 << 20, d_1l), 4 * H, 1
            t_max = max(act_t, (v_bud - v_st) // alpha)
            n_ops, act_b = 2 * l_loc, 4 * l_loc * tok_b
        else:
            is_1l, mul = sc == "single_layer", (n_seg if sc == "stage" else 1)
            d_r, e_r = (d_1l if is_1l else d_m), (e_1l if is_1l else e_m) * (l_moe > 0)
            s_a = max(gqa_s, gdn_s, ffn_s) if is_1l else max(gqa_s + gdn_s, ffn_s)
            p_m = (min(12 << 20, e_r) if tl else (12 << 20)) if l_moe else (2 << 20)
            pad = 0 if is_1l else p_m + 2304 * H * (l_moe == 0)
            v_st = d_r + e_r + (s_a if va else gqa_s + gdn_s + ffn_s) + pad
            act_b = 2 * (l_loc if is_1l else mul) * tok_b
            n_ops = l_loc if is_1l else (
                mul * (3 if tp * ep > 1 and l_attn + l_lin else 1))
            alpha = (28 * H + 4 * ne) if l_moe else (4 * H if va else 12 * H)
            t_max = 0 if v_st + alpha > v_bud else max(1, (v_bud - v_st) // alpha)
            n_sp = act_t if t_max == 0 else ceil(act_t / t_max)
            feas, rsn = (False, "Rings exceed VMEM") if t_max == 0 else (feas, rsn)
        w_step, phi, t_ep = w_static, 1.0, ceil(act_t / max(1, n_sp)) * ep_dp
        if e_loc > 0:
            e_hit = max(1.0, e_loc * (1.0 - (1.0 - min(1.0, tk / ne)) ** t_ep))
            pairs = max(1.0, t_ep * tk * e_loc / ne)
            avg_m, w_step = max(1.0, pairs / e_hit), w_static + l_moe * e_hit * exp_b
            phi_b = ceil(avg_m / r_lhs) * r_lhs / avg_m
            is_b = sc == "sublayer" or wl.moe_mode == "binned_vmem"
            phi = phi_b if is_b else max(1.0, (ops.rup(t_ep, 16) * e_hit) / pairs)
        t_comm = n_ar * (ar_vol / eg_bw + n_sp * ar_hops * lat_c) + t_pp_s
        t_comm += l_moe * (a2a_t + 2 * n_sp * lat_c) if l_moe and ep_dp > 1 else 0.0
        t_mxu, act_sv = f_dense / pk_w + (phi * f_moe) / pk_e, 4 * l_loc * tok_b - act_b
        hbm_int, d_ops = n_sp * w_step + s_kv_b + s_rec_rw, sub_ops - n_ops
        roof_u = (max(t_mxu, (hbm_int + act_b) / bw_hbm) + t_comm) * 1e6
        pure_u = (max(t_mxu, hbm_int / bw_hbm) + act_b / bw_hbm + t_comm) * 1e6
        sub_pure = pure_u if sc == "sublayer" else sub_pure
        step_u, d_b = pure_u + n_ops * ovh, bw_hbm * max(0, d_ops) * ovh * 1e-6
        be = round(max(0.0, (pure_u - sub_pure) / d_ops), 3) if d_ops > 0 else 0.0
        w_cr = round(((act_sv + d_b) / (n_sp - 1)) / 1e9, 4) if n_sp > 1 else None
        raw_map[sc] = step_u
        ests.append(ScopeEstimate(
            sc, feas, rsn, n_ops, round(v_st / (1 << 20), 3), t_max, n_sp,
            round(w_step / 1e9, 4), w_cr, round(act_b / 1e6, 3), round(phi, 3),
            round(roof_u, 2), round(step_u, 2), be))
    cands = [e for e in ests if e.feasible] if hbm_fits else [ests[0]]
    best = min(cands, key=lambda e: (raw_map[e.scope], e.ops_per_step))
    used_g, cap_g, ridge = hbm_used / 1e9, cs.hbm_capacity_bytes / 1e9, pk_w / bw_hbm
    b_wa = best.w_step_gb * 1e9 * best.n_splits + best.act_hbm_mb * 1e6
    st_b, v_q = s_kv_b + s_rec_rw, (nh_r * (2 * lrk + qrk) if lrk else 2.0 * q_dim)
    tot_b, v_p = max(1.0, b_wa + st_b), p_base + l_moe * H * ne + f_moe / (2.0 * act_t)
    v_flp = 2.0 * b_loc * tv_tok * (v_p + l_attn * eff_ctx * v_q)
    r_ok = f"{best.scope} ({best.step_bound_us:.1f} vs {ests[0].step_bound_us:.1f} us)"
    rat = r_ok if hbm_fits else f"Model+cache ({used_g:.2f} GB) > HBM ({cap_g:.2f} GB)"
    rec: FusionScope = best.scope if hbm_fits else "sublayer"
    ai_s = round((f_dense + best.moe_flop_factor * f_moe) / tot_b, 2)
    ai_i = round(v_flp / st_b, 2) if st_b else None
    return AdvisorReport(
        pr.name, cs.name, wl.phase, act_t, eff_ctx, round(v_bud / (1 << 20), 3),
        hbm_fits, round(used_g, 3), round(cap_g, 3), round(ridge, 2), ai_s, ai_i,
        rec, rec != "sublayer", rat, tuple(ests))
def advise_from_spec(
    model_spec: spec.ModelSpec, ex: exec_spec.ExecSpec, chip: ChipSpec | str = "tpu7x",
    moe_mode: MoEMode = "dense_masked", op_overhead_us: float = 0.0,
) -> AdvisorReport:
    plan_err, ar_slots, is_per, mlc, wd, ed, kv_dt = "", 2, True, {}, "fp8", "", "fp8"
    try:
        p = planner.plan(model_spec, ex)
        wds, mlc = tuple(p.weight_decls.values()), p.module_layer_counts
        ar_slots = sum(x is not None for x in p.slot_ar_index)
        rings = ([w for w in wds if w.stream_ring == r] for r in ("static", "dynamic"))
        kv_s = [s for k, s in p.state_decls.items() if "dn" not in p.state_owners[k]]
        wd, ed, kv_dt = [
            ("fp8" if jnp.dtype(x[0].dtype).itemsize == 1 else "bf16") if x else d
            for x, d in zip((*rings, kv_s), ("fp8", "", "fp8"), strict=True)]
    except errors.MegakernelConfigError as exc:
        plan_err, is_per = str(exc), False
    g = vars(model_spec.config)
    L, H, att_ev, n_exp = int(g["L"]), int(g["H"]), _i(g, "att_every", d=1), _i(g, "E")
    n_at = sum(v for k, v in mlc.items() if "gqa" in k or "att" in k)
    n_at = min(L, n_at or (L // att_ev if att_ev > 1 else L))
    prof = ModelProfile(
        model_spec.name, L, H, _i(g, "I", d=4 * H), _i(g, "NH", d=ex.R),
        _i(g, "NKV", d=ex.R), _i(g, "HD", d=128), max(0, L - n_at), _i(g, "KH"),
        _i(g, "VH"), n_exp, _i(g, "TOPK") * (n_exp > 0), _i(g, "I"), _i(g, "IS"),
        L * (n_exp > 0), n_at, linear_key_dim=_i(g, "KD", d=128),
        linear_value_dim=_i(g, "VD", d=128), is_periodic=is_per,
        ar_slots_per_layer=ar_slots, weight_dtype=wd, expert_dtype=ed or wd,
        kv_dtype=kv_dt)
    eff_ep = ex.R if (n_exp > 0 and not bool(g.get("moe_tp", True))) else 1
    wl = WorkloadSpec(
        "verify" if ex.T > 1 else "decode", ex.NS if ex.T > 1 else ex.B, ex.ctx, 0,
        ex.ctx, spec_tokens=max(0, ex.T - 1), tp=ex.R, ep=eff_ep, moe_mode=moe_mode,
        tiles=ex.tiles, op_overhead_us=op_overhead_us)
    return advise(prof, wl, chip, plan_err)
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m vllm_torchtpu.megakernel.advisor")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to local HuggingFace config.json")
    src.add_argument("--model", help="Built-in preset, HF repo ID, or local dir")
    w_f = [(f.name, int if f.type == "int" else float,
            None if f.name == "batch" else f.default)
           for f in fields(WorkloadSpec) if f.type in ("int", "float")]
    s_f = [(k, str, None) for k in (
        "weight_dtype", "expert_dtype", "kv_dtype", "state_dtype")]
    c_f = [("chip", str, "tpu7x"), ("ici_bw_gbps", float, None),
           ("collective_latency_us", float, None)]
    for k, tp, v in (*w_f, *s_f, *c_f):
        ap.add_argument(f"--{k.replace('_', '-')}", type=tp, default=v)
    ap.add_argument("--phase", choices=get_args(Phase), default="decode")
    ap.add_argument("--moe-mode", choices=get_args(MoEMode), default="dense_masked")
    ap.add_argument("--no-vmem-alias", dest="vmem_alias", action="store_false")
    for flag in ("--megacore", "--vmem-fit-headroom", "--json"):
        ap.add_argument(flag, action="store_true")
    return ap
def main(argv: Sequence[str] | None = None) -> int:
    a = build_arg_parser().parse_args(argv)
    if a.config and not Path(a.config).is_file():
        raise errors.MegakernelConfigError(f"Missing config file: {a.config!r}")
    dts = (a.weight_dtype, a.expert_dtype, a.kv_dtype, a.state_dtype)
    prof = resolve_model_profile(Path(a.config) if a.config else a.model, *dts)
    cs = resolve_chip_spec(a.chip, a.megacore, a.ici_bw_gbps, a.collective_latency_us)
    b_val = a.batch if a.batch is not None else (1 if a.phase == "prefill" else 16)
    kw = {f.name: vars(a)[f.name] for f in fields(WorkloadSpec) if f.name in vars(a)}
    rep = advise(prof, WorkloadSpec(**(kw | {"batch": b_val})), cs)
    ls = [f"  - {s.scope:12s} {s.step_bound_us:9.1f} us" for s in rep.scopes]
    hdr = f"Model: {rep.model} | Chip: {rep.chip} | Phase: {rep.phase}"
    rec = f"Recommended: {rep.recommended_scope} (build={rep.build_megakernel})"
    txt = "\n".join([hdr, rec, f"Rationale: {rep.rationale}", *ls])
    sys.stdout.write((json.dumps(rep.to_dict(), indent=2) if a.json else txt) + "\n")
    return 0
# fmt: on


if __name__ == "__main__":
    raise SystemExit(main())
