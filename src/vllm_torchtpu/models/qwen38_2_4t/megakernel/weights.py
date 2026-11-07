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
"""Weight extraction, NVFP4 dequantization/normalization, and state packing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from .shapes import Qwen38Config

    class _WeightHolder(Protocol):
        """Protocol for modules exposing a `weight` tensor."""

        weight: torch.Tensor | np.ndarray

    class _LinearProjLike(Protocol):
        """Protocol for quantized linear projection modules."""

        weight: torch.Tensor | np.ndarray
        weight_scale: torch.Tensor | np.ndarray

    class _AttnSublayerLike(Protocol):
        """Protocol for full-attention sublayers."""

        qkv_proj: _LinearProjLike
        o_proj: _LinearProjLike
        q_norm: _WeightHolder
        k_norm: _WeightHolder

    class _GdnSublayerLike(Protocol):
        """Protocol for GDN linear-attention sublayers."""

        in_proj_qkvz: _LinearProjLike
        out_proj: _LinearProjLike
        in_proj_ba: _WeightHolder
        conv1d: _WeightHolder
        A_log: torch.Tensor | np.ndarray
        dt_bias: torch.Tensor | np.ndarray
        norm: _WeightHolder

    class _SharedExpertLike(Protocol):
        """Protocol for shared MLP expert modules."""

        gate_up_proj: _LinearProjLike
        down_proj: _LinearProjLike

    class _MlpSublayerLike(Protocol):
        """Protocol for MoE MLP sublayers."""

        gate: _WeightHolder
        shared_expert_gate: _WeightHolder
        shared_expert: _SharedExpertLike
        experts: object

    class _DecoderLayerLike(Protocol):
        """Protocol for Qwen3.8 decoder layers."""

        input_layernorm: _WeightHolder
        post_attention_layernorm: _WeightHolder
        self_attn: _AttnSublayerLike
        linear_attn: _GdnSublayerLike
        mlp: _MlpSublayerLike

    class BackboneLike(Protocol):
        """Structural protocol for a Qwen3.8 decoder backbone."""

        layers: Sequence[_DecoderLayerLike]
        norm: _WeightHolder

    _BackboneLike = BackboneLike


def has_observed_weights() -> bool:
    """Return whether observed weights dictionary is populated."""
    return bool(_OBSERVED_WEIGHTS)


# Master registry of 28 weight and state tensors required by the Qwen3.8-2.4T megakernel
WEIGHT_KEYS: tuple[str, ...] = (
    "in_norm",
    "post_norm",
    "final_norm",
    "wqkv",
    "wqkv_s",
    "wo",
    "wo_s",
    "q_norm",
    "k_norm",
    "w_inproj",
    "w_inproj_s",
    "w_ba",
    "w_conv",
    "a_log",
    "dt_bias",
    "gdn_norm",
    "w_out",
    "w_out_s",
    "w_gate",
    "w_se13",
    "w_se13_s",
    "w_se2",
    "w_se2_s",
    "w_se_gate",
    "w_e13",
    "w_e13_s",
    "w_e2",
    "w_e2_s",
)

_OBSERVED_WEIGHTS: dict[str, torch.Tensor] = {}

_EXPERT_SUFFIXES = (
    "mlp.experts.w13_weight",
    "mlp.experts.w13_weight_scale",
    "mlp.experts.w13_weight_scale_2",
    "mlp.experts.w2_weight",
    "mlp.experts.w2_weight_scale",
    "mlp.experts.w2_weight_scale_2",
    "mlp.experts.gate_up_proj.weight",
    "mlp.experts.gate_up_proj.weight_scale",
    "mlp.experts.gate_up_proj.weight_scale_2",
    "mlp.experts.gate_up_proj",
    "mlp.experts.gate_up_proj_scale",
    "mlp.experts.gate_up_proj_scale_2",
    "mlp.experts.down_proj.weight",
    "mlp.experts.down_proj.weight_scale",
    "mlp.experts.down_proj.weight_scale_2",
    "mlp.experts.down_proj",
    "mlp.experts.down_proj_scale",
    "mlp.experts.down_proj_scale_2",
)

_EXPERT_ATTR_ALIASES: dict[str, tuple[str, ...]] = {
    "w13_weight": (
        "w13_weight",
        "gate_up_proj_weight",
        "gate_up_proj.weight",
        "gate_up_proj",
    ),
    "w13_weight_scale": (
        "w13_weight_scale",
        "gate_up_proj_scale",
        "gate_up_proj.weight_scale",
        "gate_up_proj.scale",
        "gate_up_proj_weight_scale",
    ),
    "w13_weight_scale_2": (
        "w13_weight_scale_2",
        "gate_up_proj_scale_2",
        "gate_up_proj.weight_scale_2",
        "gate_up_proj_weight_scale_2",
    ),
    "w2_weight": ("w2_weight", "down_proj_weight", "down_proj.weight", "down_proj"),
    "w2_weight_scale": (
        "w2_weight_scale",
        "down_proj_scale",
        "down_proj.weight_scale",
        "down_proj.scale",
        "down_proj_weight_scale",
    ),
    "w2_weight_scale_2": (
        "w2_weight_scale_2",
        "down_proj_scale_2",
        "down_proj_weight_scale_2",
    ),
}


def clear_observed_weights() -> None:
    """Clear the pre-pack observed weight cache."""
    _OBSERVED_WEIGHTS.clear()


def load_tensor(
    first: str | object,
    second: torch.Tensor | str | None = None,
    options: Mapping[str, object] | torch.Tensor | None = None,
    *extra: torch.Tensor | Mapping[str, object] | None,
) -> None:
    """Capture pre-pack expert weights and scales from the checkpoint stream.

    Supports both `load_tensor(name, tensor[, options])` (called by
    `megakernel_runtime.hooks.load_tensor_stream`) and
    `load_tensor(vllm_config, name, tensor[, options])`.
    """
    if isinstance(first, str):
        if (
            second is None
            or isinstance(second, str)
            or not isinstance(second, torch.Tensor)
        ):
            msg = f"Expected torch.Tensor for '{first}', got {type(second).__name__}"
            raise TypeError(msg)
        if any(first.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            _OBSERVED_WEIGHTS[first] = second.detach().cpu().clone()
        return
    if isinstance(second, str):
        cand = (
            options
            if (options is not None and not isinstance(options, Mapping))
            else (extra[0] if extra else None)
        )
        if (
            cand is None
            or isinstance(cand, Mapping)
            or not isinstance(cand, torch.Tensor)
        ):
            msg = f"Expected torch.Tensor for '{second}', got {type(cand).__name__}"
            raise TypeError(msg)
        if any(second.endswith(sfx) for sfx in _EXPERT_SUFFIXES):
            _OBSERVED_WEIGHTS[second] = cand.detach().cpu().clone()
        return
    msg = f"Expected str tensor name, got {type(first).__name__}"
    raise TypeError(msg)


def _observed_layer_expert_attr(layer_idx: int, attr: str) -> torch.Tensor | None:
    """Look up a pre-pack observed tensor for a specific layer and expert attribute."""
    candidates = _EXPERT_ATTR_ALIASES.get(attr, (attr,))
    for cand in candidates:
        for prefix in (
            f"model.layers.{layer_idx}.mlp.experts.{cand}",
            f"language_model.model.layers.{layer_idx}.mlp.experts.{cand}",
            f"layers.{layer_idx}.mlp.experts.{cand}",
        ):
            val = _OBSERVED_WEIGHTS.get(prefix)
            if val is not None:
                return val
    return None


def find_qwen38_2_4t_backbone(model: object) -> BackboneLike:
    """Locate the Qwen3.8-2.4T decoder backbone (`layers` + `norm`) inside a model."""
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
            return cast("BackboneLike", cur)
    msg = f"Cannot locate Qwen3.8-2.4T backbone in {type(model).__name__}"
    raise ValueError(msg)


find_qwen38_backbone = find_qwen38_2_4t_backbone


def _to_np_u8(t: torch.Tensor | np.ndarray | None) -> np.ndarray:
    """Convert tensor or array-like object to contiguous uint8 NumPy array."""
    if isinstance(t, np.ndarray) or t is None:
        return np.asarray(t, dtype=np.uint8)
    if hasattr(t, "detach"):
        return t.detach().cpu().contiguous().view(torch.uint8).numpy()
    return np.asarray(t, dtype=np.uint8)


def _to_np_f32(t: torch.Tensor | np.ndarray | None) -> np.ndarray:
    """Convert tensor or array-like object to contiguous float32 NumPy array."""
    if isinstance(t, np.ndarray) or t is None:
        return np.asarray(t, dtype=np.float32)
    if hasattr(t, "detach"):
        if t.dtype in (
            torch.float8_e4m3fn,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            return t.detach().cpu().to(torch.float32).numpy()
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _transpose_packed_fp4_2d(
    w_packed: np.ndarray, k_dim: int, n_dim: int
) -> np.ndarray:
    """Convert packed FP4 weight from `(n_dim, k_dim // 2)` to `(k_dim, n_dim // 2)`."""
    if w_packed.shape == (n_dim, k_dim // 2):
        lo = w_packed & 0x0F
        hi = (w_packed >> 4) & 0x0F
        nib_nk = np.empty((n_dim, k_dim), dtype=np.uint8)
        nib_nk[:, 0::2] = lo
        nib_nk[:, 1::2] = hi
        nib_kn = np.ascontiguousarray(nib_nk.T)
        out = (nib_kn[:, 0::2] & 0x0F) | ((nib_kn[:, 1::2] & 0x0F) << 4)
        return np.ascontiguousarray(out, dtype=np.uint8)
    if w_packed.shape == (k_dim, n_dim // 2):
        return np.ascontiguousarray(w_packed, dtype=np.uint8)
    msg = f"Unexpected packed FP4 2D shape {w_packed.shape} for K={k_dim}, N={n_dim}"
    raise ValueError(msg)


def _transpose_scale_2d(
    s_arr: np.ndarray, k_dim: int, n_dim: int, group_size: int = 16
) -> np.ndarray:
    """Convert FP4 block scale from `(n_dim, G)` to `(G, n_dim)` float32."""
    g_dim = k_dim // group_size
    if s_arr.shape == (n_dim, g_dim):
        return np.ascontiguousarray(s_arr.T, dtype=np.float32)
    if s_arr.shape == (g_dim, n_dim):
        return np.ascontiguousarray(s_arr, dtype=np.float32)
    msg = f"Unexpected FP4 scale 2D shape {s_arr.shape} for G={g_dim}, N={n_dim}"
    raise ValueError(msg)


def _extract_linear_fp4(
    proj: object, k_dim: int, n_dim: int, group_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Extract packed FP4 weights and scale factors from a linear projection."""
    w_attr = getattr(proj, "weight", getattr(proj, "qweight", None))
    if w_attr is None:
        msg = f"{type(proj).__name__} is missing required attribute 'weight'"
        raise ValueError(msg)
    s_attr = getattr(proj, "weight_scale", getattr(proj, "scales", None))
    if s_attr is None:
        msg = f"{type(proj).__name__} is missing required attribute 'weight_scale'"
        raise ValueError(msg)
    w_raw = _to_np_u8(w_attr)
    s_raw = _to_np_f32(s_attr)
    s2_attr = getattr(proj, "weight_scale_2", None)
    if s2_attr is not None:
        s_raw = s_raw * float(_to_np_f32(s2_attr).reshape(-1)[0])
    w_kn = _transpose_packed_fp4_2d(w_raw, k_dim, n_dim)
    s_gn = _transpose_scale_2d(s_raw, k_dim, n_dim, group_size=group_size)
    return w_kn, s_gn


def _resolve_expert_tensor(
    exps: object,
    layer_idx: int,
    attr: str,
    *,
    use_observed: bool,
    required: bool = True,
) -> torch.Tensor | np.ndarray | None:
    """Resolve an expert tensor from the pre-pack observed cache or live module."""
    if use_observed:
        obs = _observed_layer_expert_attr(layer_idx, attr)
        if obs is not None:
            return obs
    for cand in _EXPERT_ATTR_ALIASES.get(attr, (attr,)):
        val = getattr(exps, cand, None)
        if val is not None:
            return cast("torch.Tensor | np.ndarray", val)
    if required:
        msg = f"Layer {layer_idx} experts is missing required attribute '{attr}'"
        raise ValueError(msg)
    return None


def _transpose_expert_weights(
    w_raw: np.ndarray, e_cnt: int, k_dim: int, n_dim: int
) -> np.ndarray:
    """Transpose expert packed FP4 weights to `(e_cnt, k_dim, n_dim // 2)`."""
    if w_raw.shape == (e_cnt, n_dim, k_dim // 2):
        lo = w_raw & 0x0F
        hi = (w_raw >> 4) & 0x0F
        nib_enk = np.empty((e_cnt, n_dim, k_dim), dtype=np.uint8)
        nib_enk[:, :, 0::2] = lo
        nib_enk[:, :, 1::2] = hi
        nib_ekn = np.ascontiguousarray(np.transpose(nib_enk, (0, 2, 1)))
        return np.ascontiguousarray(
            (nib_ekn[:, :, 0::2] & 0x0F) | ((nib_ekn[:, :, 1::2] & 0x0F) << 4),
            dtype=np.uint8,
        )
    if w_raw.shape == (e_cnt, k_dim, n_dim // 2):
        return np.ascontiguousarray(w_raw, dtype=np.uint8)
    msg = f"Unexpected expert FP4 shape {w_raw.shape}"
    raise ValueError(msg)


def _transpose_expert_scales(
    s_raw: np.ndarray, e_cnt: int, g_dim: int, n_dim: int
) -> np.ndarray:
    """Transpose expert FP4 scales to `(e_cnt, g_dim, n_dim)`."""
    if s_raw.shape == (e_cnt, n_dim, g_dim):
        return np.ascontiguousarray(np.transpose(s_raw, (0, 2, 1)), dtype=np.float32)
    if s_raw.shape == (e_cnt, g_dim, n_dim):
        return np.ascontiguousarray(s_raw, dtype=np.float32)
    msg = f"Unexpected expert scale shape {s_raw.shape}"
    raise ValueError(msg)


def _extract_experts_fp4(
    exps: object,
    layer_idx: int,
    attrs: tuple[str, str, str],
    geom: tuple[int, int, int, int],
    *,
    use_observed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and transpose routed expert weights and scales for one sublayer."""
    w_attr, s_attr, s2_attr = attrs
    e_cnt, k_dim, n_dim, group_size = geom

    obs_w = _resolve_expert_tensor(
        exps, layer_idx, w_attr, use_observed=use_observed, required=True
    )
    obs_s = _resolve_expert_tensor(
        exps, layer_idx, s_attr, use_observed=use_observed, required=True
    )
    obs_s2 = _resolve_expert_tensor(
        exps, layer_idx, s2_attr, use_observed=use_observed, required=False
    )

    w_raw = _to_np_u8(obs_w)
    s_raw = _to_np_f32(obs_s)
    if obs_s2 is not None:
        s2_arr = _to_np_f32(obs_s2).reshape(-1, 1, 1)
        s_raw = s_raw * s2_arr

    g_dim = k_dim // group_size
    w_ekn = _transpose_expert_weights(w_raw, e_cnt, k_dim, n_dim)
    s_egn = _transpose_expert_scales(s_raw, e_cnt, g_dim, n_dim)
    return w_ekn, s_egn


def _extract_attn_sublayer(
    sa: _AttnSublayerLike, cfg: Qwen38Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract weight and scale arrays for one full attention sublayer."""
    wq, sq = _extract_linear_fp4(sa.qkv_proj, cfg.H, cfg.ATT_QKV_R, cfg.fp4_group)
    wo, so = _extract_linear_fp4(sa.o_proj, cfg.Q_R * cfg.D, cfg.H, cfg.fp4_group)
    qn = _to_np_f32(sa.q_norm.weight).reshape(cfg.D)
    kn = _to_np_f32(sa.k_norm.weight).reshape(cfg.D)
    return wq, sq, wo, so, qn, kn


def _extract_gdn_sublayer(
    la: _GdnSublayerLike, cfg: Qwen38Config
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract weight and state arrays for one GDN linear attention sublayer."""
    wi, si = _extract_linear_fp4(
        la.in_proj_qkvz, cfg.H, cfg.QKV + cfg.Z_R, cfg.fp4_group
    )
    wout, sout = _extract_linear_fp4(la.out_proj, cfg.Z_R, cfg.H, cfg.fp4_group)

    wba_raw = _to_np_f32(la.in_proj_ba.weight)
    if wba_raw.shape == (2 * cfg.VH_R, cfg.H):
        wba_raw = wba_raw.T
    wba = np.ascontiguousarray(wba_raw.reshape(cfg.H, 2 * cfg.VH_R))

    wconv_raw = _to_np_f32(la.conv1d.weight).reshape(-1, cfg.conv_kernel)
    if wconv_raw.shape == (cfg.QKV, cfg.conv_kernel):
        wconv_raw = wconv_raw.T
    wconv = np.ascontiguousarray(wconv_raw.reshape(cfg.conv_kernel, cfg.QKV))

    alog = _to_np_f32(la.A_log).reshape(cfg.VH_R)
    dt = _to_np_f32(la.dt_bias).reshape(cfg.VH_R)
    gnorm = _to_np_f32(la.norm.weight).reshape(cfg.VD)
    return wi, si, wout, sout, wba, wconv, alog, dt, gnorm


def _extract_mlp_sublayer(
    mlp: _MlpSublayerLike,
    l_idx: int,
    cfg: Qwen38Config,
    *,
    use_observed: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract routed and shared expert weights and scales for one MLP sublayer."""
    wg_raw = _to_np_f32(mlp.gate.weight)
    if wg_raw.shape == (cfg.E, cfg.H):
        wg_raw = wg_raw.T
    wgate = np.ascontiguousarray(wg_raw.reshape(cfg.H, cfg.E))

    wseg_obj = getattr(mlp, "shared_expert_gate", None)
    wseg_w = getattr(wseg_obj, "weight", None) if wseg_obj is not None else None
    if wseg_w is None:
        msg = f"Layer {l_idx} MLP is missing required shared_expert_gate.weight"
        raise ValueError(msg)
    wsegate = np.ascontiguousarray(_to_np_f32(wseg_w).reshape(cfg.H, 1))

    se = mlp.shared_expert
    wse13, sse13 = _extract_linear_fp4(
        se.gate_up_proj, cfg.H, 2 * cfg.SE_R, cfg.fp4_group
    )
    wse2, sse2 = _extract_linear_fp4(se.down_proj, cfg.SE_R, cfg.H, cfg.fp4_group)

    exps = mlp.experts
    we13, se13 = _extract_experts_fp4(
        exps,
        l_idx,
        ("w13_weight", "w13_weight_scale", "w13_weight_scale_2"),
        (cfg.E, cfg.H, 2 * cfg.I_R, cfg.fp4_group),
        use_observed=use_observed,
    )
    we2, se2 = _extract_experts_fp4(
        exps,
        l_idx,
        ("w2_weight", "w2_weight_scale", "w2_weight_scale_2"),
        (cfg.E, cfg.I_R, cfg.H, cfg.fp4_group),
        use_observed=use_observed,
    )
    return (
        wgate,
        wsegate,
        wse13,
        sse13,
        wse2,
        sse2,
        we13,
        se13,
        we2,
        se2,
    )


def _extract_single_rank_np(
    backbone: _BackboneLike, cfg: Qwen38Config, *, use_observed: bool = True
) -> dict[str, np.ndarray]:
    """Extract one rank's weight slices from a Qwen3.8 backbone into NumPy arrays."""
    in_norms, post_norms = [], []
    wqkv_l, wqkv_s_l = [], []
    wo_l, wo_s_l = [], []
    q_norms, k_norms = [], []

    win_l, win_s_l = [], []
    wba_l, wconv_l = [], []
    alog_l, dt_l, gnorm_l = [], [], []
    wout_l, wout_s_l = [], []

    wgate_l, wsegate_l = [], []
    wse13_l, wse13_s_l = [], []
    wse2_l, wse2_s_l = [], []
    we13_l, we13_s_l = [], []
    we2_l, we2_s_l = [], []

    layers = backbone.layers
    for l_idx in range(cfg.L):
        layer = layers[l_idx]
        in_norms.append(_to_np_f32(layer.input_layernorm.weight).reshape(cfg.H))
        post_norms.append(
            _to_np_f32(layer.post_attention_layernorm.weight).reshape(cfg.H)
        )

        if (l_idx + 1) % cfg.att_every == 0:
            wq, sq, wo, so, qn, kn = _extract_attn_sublayer(layer.self_attn, cfg)
            wqkv_l.append(wq)
            wqkv_s_l.append(sq)
            wo_l.append(wo)
            wo_s_l.append(so)
            q_norms.append(qn)
            k_norms.append(kn)
        else:
            wi, si, wout, sout, wba, wconv, alog, dt, gnorm = _extract_gdn_sublayer(
                layer.linear_attn, cfg
            )
            win_l.append(wi)
            win_s_l.append(si)
            wout_l.append(wout)
            wout_s_l.append(sout)
            wba_l.append(wba)
            wconv_l.append(wconv)
            alog_l.append(alog)
            dt_l.append(dt)
            gnorm_l.append(gnorm)

        (
            wgate,
            wsegate,
            wse13,
            sse13,
            wse2,
            sse2,
            we13,
            se13,
            we2,
            se2,
        ) = _extract_mlp_sublayer(layer.mlp, l_idx, cfg, use_observed=use_observed)
        wgate_l.append(wgate)
        wsegate_l.append(wsegate)
        wse13_l.append(wse13)
        wse13_s_l.append(sse13)
        wse2_l.append(wse2)
        wse2_s_l.append(sse2)
        we13_l.append(we13)
        we13_s_l.append(se13)
        we2_l.append(we2)
        we2_s_l.append(se2)

    final_norm_w = backbone.norm.weight
    return {
        "in_norm": np.stack(in_norms, axis=0),
        "post_norm": np.stack(post_norms, axis=0),
        "final_norm": _to_np_f32(final_norm_w).reshape(cfg.H),
        "wqkv": np.stack(wqkv_l, axis=0),
        "wqkv_s": np.stack(wqkv_s_l, axis=0),
        "wo": np.stack(wo_l, axis=0),
        "wo_s": np.stack(wo_s_l, axis=0),
        "q_norm": np.stack(q_norms, axis=0),
        "k_norm": np.stack(k_norms, axis=0),
        "w_inproj": np.stack(win_l, axis=0),
        "w_inproj_s": np.stack(win_s_l, axis=0),
        "w_ba": np.stack(wba_l, axis=0),
        "w_conv": np.stack(wconv_l, axis=0),
        "a_log": np.stack(alog_l, axis=0),
        "dt_bias": np.stack(dt_l, axis=0),
        "gdn_norm": np.stack(gnorm_l, axis=0),
        "w_out": np.stack(wout_l, axis=0),
        "w_out_s": np.stack(wout_s_l, axis=0),
        "w_gate": np.stack(wgate_l, axis=0),
        "w_se13": np.stack(wse13_l, axis=0),
        "w_se13_s": np.stack(wse13_s_l, axis=0),
        "w_se2": np.stack(wse2_l, axis=0),
        "w_se2_s": np.stack(wse2_s_l, axis=0),
        "w_se_gate": np.stack(wsegate_l, axis=0),
        "w_e13": np.stack(we13_l, axis=0),
        "w_e13_s": np.stack(we13_s_l, axis=0),
        "w_e2": np.stack(we2_l, axis=0),
        "w_e2_s": np.stack(we2_s_l, axis=0),
    }


def _to_rank_torch_tensors(
    rank_dicts: list[dict[str, np.ndarray]],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert per-rank NumPy weight dictionaries to stacked PyTorch tensors."""
    u8_keys = {"wqkv", "wo", "w_inproj", "w_out", "w_se13", "w_se2", "w_e13", "w_e2"}
    f32_keys = {"a_log", "dt_bias"}
    out: dict[str, torch.Tensor] = {}
    for k in WEIGHT_KEYS:
        stacked = np.stack([d[k] for d in rank_dicts], axis=0)
        if k in u8_keys:
            t = torch.from_numpy(stacked).to(dtype=torch.uint8, device=device)
        elif k in f32_keys:
            t = torch.from_numpy(stacked).to(dtype=torch.float32, device=device)
        else:
            t = torch.from_numpy(stacked).to(dtype=torch.bfloat16, device=device)
        out[k] = t.contiguous()
    return out


def extract_backbone_weights_torch(
    model: object,
    cfg: Qwen38Config,
    *,
    rank_backbones: Sequence[object] | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract Qwen3.8 weights into `(1, ...)` or `(R, ...)` PyTorch tensors."""
    has_obs = has_observed_weights()
    if rank_backbones is not None:
        rank_dicts = [
            _extract_single_rank_np(
                find_qwen38_2_4t_backbone(b), cfg, use_observed=has_obs
            )
            for b in rank_backbones
        ]
        return _to_rank_torch_tensors(rank_dicts, device=device)

    implicit_ranks = getattr(model, "_rank_backbones", None)
    if implicit_ranks is not None:
        rank_dicts = [
            _extract_single_rank_np(
                find_qwen38_2_4t_backbone(b),
                cfg,
                use_observed=(r_idx == 0 and has_obs),
            )
            for r_idx, b in enumerate(implicit_ranks)
        ]
        return _to_rank_torch_tensors(rank_dicts, device=device)

    backbone = find_qwen38_2_4t_backbone(model)
    single = _extract_single_rank_np(backbone, cfg, use_observed=has_obs)
    return _to_rank_torch_tensors([single], device=device)
