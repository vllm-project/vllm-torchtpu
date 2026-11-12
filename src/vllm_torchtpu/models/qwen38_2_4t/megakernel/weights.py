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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
import torch

from vllm_torchtpu.megakernel.adapter.backbone import find_backbone

from .constants import (
    EXPERT_ATTR_ALIASES,
    F32_WEIGHT_KEYS,
    U8_WEIGHT_KEYS,
    WEIGHT_KEYS,
)

if TYPE_CHECKING:
    from .shapes import Qwen38Config

    class _WeightHolder(Protocol):
        """Protocol for modules exposing a `weight` tensor."""

        weight: torch.Tensor | np.ndarray

    class _LinearProjLike(Protocol):
        """Protocol for quantized linear projection modules."""

        weight: torch.Tensor | np.ndarray
        weight_scale: torch.Tensor | np.ndarray
        weight_scale_2: torch.Tensor | np.ndarray | None = None

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


@runtime_checkable
class MultiRankBackboneHolder(Protocol):
    """Protocol for multi-rank test/host models exposing per-rank backbones."""

    @property
    def rank_backbones(self) -> Sequence[object]:
        """Return per-rank decoder backbones."""


find_qwen38_2_4t_backbone = find_backbone


def _get_candidate_attr(obj: object, cand: str) -> object:
    """Resolve an attribute or 2-part dotted submodule attribute."""
    if "." in cand:
        sub, name = cand.split(".", 1)
        sub_obj = getattr(obj, sub, None)
        return getattr(sub_obj, name, None) if sub_obj is not None else None
    return getattr(obj, cand, None)


def _to_np_u8(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert non-`None` tensor or array to contiguous `uint8` NumPy array."""
    if t is None:
        msg = "Expected Tensor or ndarray for uint8 weight, got None"
        raise TypeError(msg)
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t, dtype=np.uint8)
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().contiguous().view(torch.uint8).numpy()
    return np.ascontiguousarray(t, dtype=np.uint8)


def _to_np_f32(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert non-`None` tensor or array to contiguous `float32` NumPy array."""
    if t is None:
        msg = "Expected Tensor or ndarray for float32 tensor, got None"
        raise TypeError(msg)
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t, dtype=np.float32)
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().to(torch.float32).numpy()
    return np.ascontiguousarray(t, dtype=np.float32)


def _transpose_packed_fp4(
    w_packed: np.ndarray,
    k_dim: int,
    n_dim: int,
    *,
    already_kmajor: bool = False,
) -> np.ndarray:
    """Convert packed FP4 `(..., n_dim, k_dim // 2)` into `(..., k_dim, n_dim // 2)`."""
    prefix = w_packed.shape[:-2]
    raw_shape = (*prefix, n_dim, k_dim // 2)
    target_shape = (*prefix, k_dim, n_dim // 2)
    if already_kmajor and w_packed.shape == target_shape:
        return np.ascontiguousarray(w_packed, dtype=np.uint8)
    if w_packed.shape == raw_shape:
        lo = w_packed & 0x0F
        hi = (w_packed >> 4) & 0x0F
        nib_nk = np.empty((*prefix, n_dim, k_dim), dtype=np.uint8)
        nib_nk[..., 0::2] = lo
        nib_nk[..., 1::2] = hi
        nib_kn = np.ascontiguousarray(nib_nk.swapaxes(-2, -1))
        out = (nib_kn[..., 0::2] & 0x0F) | ((nib_kn[..., 1::2] & 0x0F) << 4)
        return np.ascontiguousarray(out, dtype=np.uint8)
    if w_packed.shape == target_shape:
        return np.ascontiguousarray(w_packed, dtype=np.uint8)
    msg = (
        f"Unexpected packed FP4 shape {w_packed.shape}; "
        f"expected {raw_shape} or {target_shape}"
    )
    raise ValueError(msg)


def _transpose_fp4_scale(
    s_arr: np.ndarray, k_dim: int, n_dim: int, group_size: int = 16
) -> np.ndarray:
    """Convert FP4 block scales `(..., n_dim, G)` into `(..., G, n_dim)` float32."""
    g_dim = k_dim // group_size
    prefix = s_arr.shape[:-2]
    raw_shape = (*prefix, n_dim, g_dim)
    target_shape = (*prefix, g_dim, n_dim)
    if s_arr.shape == raw_shape:
        return np.ascontiguousarray(s_arr.swapaxes(-2, -1), dtype=np.float32)
    if s_arr.shape == target_shape:
        return np.ascontiguousarray(s_arr, dtype=np.float32)
    msg = (
        f"Unexpected FP4 scale shape {s_arr.shape}; "
        f"expected {raw_shape} or {target_shape}"
    )
    raise ValueError(msg)


def _orient_2d(arr: np.ndarray, rows: int, cols: int, *, name: str) -> np.ndarray:
    """Orient a 2D weight matrix to shape `(rows, cols)`, transposing `(cols, rows)`."""
    if arr.shape == (cols, rows):
        return np.ascontiguousarray(arr.T)
    if arr.shape == (rows, cols):
        return np.ascontiguousarray(arr)
    msg = (
        f"Unexpected {name} shape {arr.shape}; "
        f"expected {(rows, cols)} or {(cols, rows)}"
    )
    raise ValueError(msg)


def _extract_linear_fp4(
    proj: _LinearProjLike, k_dim: int, n_dim: int, group_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Extract packed FP4 weights and scale factors from a linear projection."""
    w_attr = proj.weight
    if w_attr is None:
        msg = f"{type(proj).__name__} is missing required attribute 'weight'"
        raise ValueError(msg)
    s_attr = proj.weight_scale
    if s_attr is None:
        msg = f"{type(proj).__name__} is missing required attribute 'weight_scale'"
        raise ValueError(msg)
    w_raw = _to_np_u8(w_attr)
    s_raw = _to_np_f32(s_attr)
    is_post_load = s_raw.ndim == 4 and s_raw.shape[0] == 1 and s_raw.shape[2] == 1
    if is_post_load:
        # VllmNvFp4LinearMethod.process_weights_after_loading stores [1, G, 1, N].
        s_raw = s_raw[0, :, 0, :]
    fp4_x2 = getattr(torch, "float4_e2m1fn_x2", None)
    already_kmajor = is_post_load or (
        isinstance(w_attr, torch.Tensor)
        and fp4_x2 is not None
        and w_attr.dtype == fp4_x2
    )
    s2_attr = getattr(proj, "weight_scale_2", None)
    if s2_attr is not None:
        s_raw = s_raw * float(_to_np_f32(s2_attr).reshape(-1)[0])
    w_kn = _transpose_packed_fp4(w_raw, k_dim, n_dim, already_kmajor=already_kmajor)
    s_gn = _transpose_fp4_scale(s_raw, k_dim, n_dim, group_size=group_size)
    return w_kn, s_gn


def _resolve_expert_tensor(
    exps: object,
    layer_idx: int,
    attr: str,
    *,
    required: bool = True,
) -> torch.Tensor | np.ndarray | None:
    """Resolve an expert tensor from module attributes."""
    for cand in EXPERT_ATTR_ALIASES.get(attr, (attr,)):
        val = _get_candidate_attr(exps, cand)
        if val is not None:
            return cast("torch.Tensor | np.ndarray", val)
    if required:
        msg = f"Layer {layer_idx} experts is missing required attribute '{attr}'"
        raise ValueError(msg)
    return None


def _slice_tp_expert_shard(
    w_raw: np.ndarray,
    s_raw: np.ndarray,
    geom: tuple[int, int, int, int],
    tp_shard: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice global unsharded expert tensors for `(tp_rank, tp_size)` when needed."""
    _, k_dim, n_dim, group_size = geom
    tp_rank, tp_size = tp_shard
    if tp_size <= 1:
        return w_raw, s_raw
    k_packed = k_dim // 2
    g_dim = k_dim // group_size
    if (
        n_dim % 2 == 0
        and w_raw.shape[1:] == (tp_size * n_dim, k_packed)
        and s_raw.shape[1:] == (tp_size * n_dim, g_dim)
    ):
        half_r = n_dim // 2
        half_global = tp_size * half_r
        g_lo, g_hi = tp_rank * half_r, (tp_rank + 1) * half_r
        u_lo, u_hi = half_global + g_lo, half_global + g_hi
        w_sharded = np.concatenate(
            [w_raw[:, g_lo:g_hi, :], w_raw[:, u_lo:u_hi, :]], axis=1
        )
        s_sharded = np.concatenate(
            [s_raw[:, g_lo:g_hi, :], s_raw[:, u_lo:u_hi, :]], axis=1
        )
        return w_sharded, s_sharded
    if w_raw.shape[1:] == (n_dim, tp_size * k_packed) and s_raw.shape[1:] == (
        n_dim,
        tp_size * g_dim,
    ):
        w_lo, w_hi = tp_rank * k_packed, (tp_rank + 1) * k_packed
        s_lo, s_hi = tp_rank * g_dim, (tp_rank + 1) * g_dim
        return w_raw[:, :, w_lo:w_hi], s_raw[:, :, s_lo:s_hi]
    return w_raw, s_raw


def _extract_experts_fp4(
    exps: object,
    layer_idx: int,
    attrs: tuple[str, str, str],
    geom: tuple[int, int, int, int],
    *,
    tp_shard: tuple[int, int] = (0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and transpose routed expert weights and scales for one sublayer."""
    w_attr, s_attr, s2_attr = attrs
    e_cnt, k_dim, n_dim, group_size = geom
    tp_rank, tp_size = tp_shard

    obs_w = _resolve_expert_tensor(exps, layer_idx, w_attr, required=True)
    obs_s = _resolve_expert_tensor(exps, layer_idx, s_attr, required=True)
    obs_s2 = _resolve_expert_tensor(exps, layer_idx, s2_attr, required=False)
    if obs_w is None or obs_s is None:
        msg = f"Layer {layer_idx} experts failed to resolve required tensors {attrs}"
        raise ValueError(msg)

    w_raw = _to_np_u8(obs_w)
    s_raw = _to_np_f32(obs_s)
    if w_raw.shape[0] != e_cnt or s_raw.shape[0] != e_cnt:
        msg = (
            f"Layer {layer_idx} experts expected E={e_cnt}, "
            f"got weight={w_raw.shape}, scale={s_raw.shape}"
        )
        raise ValueError(msg)
    is_post_load = s_raw.ndim == 4 and s_raw.shape[2] == 1
    if is_post_load:
        # VllmNvFp4MoEMethod.process_weights_after_loading stores [E, G, 1, N].
        s_raw = s_raw[:, :, 0, :]
    fp4_x2 = getattr(torch, "float4_e2m1fn_x2", None)
    already_kmajor = is_post_load or (
        isinstance(obs_w, torch.Tensor) and fp4_x2 is not None and obs_w.dtype == fp4_x2
    )
    if obs_s2 is not None:
        s_raw = s_raw * _to_np_f32(obs_s2).reshape(e_cnt, -1, 1)
    w_raw, s_raw = _slice_tp_expert_shard(w_raw, s_raw, geom, (tp_rank, tp_size))

    w_ekn = _transpose_packed_fp4(w_raw, k_dim, n_dim, already_kmajor=already_kmajor)
    s_egn = _transpose_fp4_scale(s_raw, k_dim, n_dim, group_size=group_size)
    return w_ekn, s_egn


def _extract_attn_sublayer(
    sa: _AttnSublayerLike, cfg: Qwen38Config
) -> dict[str, np.ndarray]:
    """Extract weight and scale arrays for one full attention sublayer."""
    wq, sq = _extract_linear_fp4(sa.qkv_proj, cfg.H, cfg.ATT_QKV_R, cfg.fp4_group)
    wo, so = _extract_linear_fp4(sa.o_proj, cfg.Q_R * cfg.D, cfg.H, cfg.fp4_group)
    return {
        "wqkv": wq,
        "wqkv_s": sq,
        "wo": wo,
        "wo_s": so,
        "q_norm": _to_np_f32(sa.q_norm.weight).reshape(cfg.D),
        "k_norm": _to_np_f32(sa.k_norm.weight).reshape(cfg.D),
    }


def _extract_gdn_sublayer(
    la: _GdnSublayerLike, cfg: Qwen38Config
) -> dict[str, np.ndarray]:
    """Extract weight and state arrays for one GDN linear attention sublayer."""
    wi, si = _extract_linear_fp4(
        la.in_proj_qkvz, cfg.H, cfg.QKV + cfg.Z_R, cfg.fp4_group
    )
    wout, sout = _extract_linear_fp4(la.out_proj, cfg.Z_R, cfg.H, cfg.fp4_group)
    wba = _orient_2d(
        _to_np_f32(la.in_proj_ba.weight), cfg.H, 2 * cfg.VH_R, name="in_proj_ba"
    )
    wconv = _orient_2d(
        _to_np_f32(la.conv1d.weight).reshape(-1, cfg.conv_kernel),
        cfg.conv_kernel,
        cfg.QKV,
        name="conv1d",
    )
    return {
        "w_inproj": wi,
        "w_inproj_s": si,
        "w_out": wout,
        "w_out_s": sout,
        "w_ba": wba,
        "w_conv": wconv,
        "a_log": _to_np_f32(la.A_log).reshape(cfg.VH_R),
        "dt_bias": _to_np_f32(la.dt_bias).reshape(cfg.VH_R),
        "gdn_norm": _to_np_f32(la.norm.weight).reshape(cfg.VD),
    }


def _extract_mlp_sublayer(
    mlp: _MlpSublayerLike,
    l_idx: int,
    cfg: Qwen38Config,
    *,
    tp_rank: int = 0,
) -> dict[str, np.ndarray]:
    """Extract routed and shared expert weights and scales for one MLP sublayer."""
    wgate = _orient_2d(
        _to_np_f32(mlp.gate.weight), cfg.H, cfg.E, name=f"layer[{l_idx}].mlp.gate"
    )
    wseg_obj = mlp.shared_expert_gate
    wseg_w = getattr(wseg_obj, "weight", None) if wseg_obj is not None else None
    if wseg_w is None:
        msg = f"Layer {l_idx} MLP is missing required shared_expert_gate.weight"
        raise ValueError(msg)
    wsegate = np.ascontiguousarray(_to_np_f32(wseg_w).reshape(cfg.H, 1))

    shared_exp = mlp.shared_expert
    w_shared_13, s_shared_13 = _extract_linear_fp4(
        shared_exp.gate_up_proj, cfg.H, 2 * cfg.SE_R, cfg.fp4_group
    )
    w_shared_2, s_shared_2 = _extract_linear_fp4(
        shared_exp.down_proj, cfg.SE_R, cfg.H, cfg.fp4_group
    )
    tp_shard = (tp_rank, cfg.R)
    w_exp_13, s_exp_13 = _extract_experts_fp4(
        mlp.experts,
        l_idx,
        ("w13_weight", "w13_weight_scale", "w13_weight_scale_2"),
        (cfg.E, cfg.H, 2 * cfg.I_R, cfg.fp4_group),
        tp_shard=tp_shard,
    )
    w_exp_2, s_exp_2 = _extract_experts_fp4(
        mlp.experts,
        l_idx,
        ("w2_weight", "w2_weight_scale", "w2_weight_scale_2"),
        (cfg.E, cfg.I_R, cfg.H, cfg.fp4_group),
        tp_shard=tp_shard,
    )
    return {
        "w_gate": wgate,
        "w_se_gate": wsegate,
        "w_se13": w_shared_13,
        "w_se13_s": s_shared_13,
        "w_se2": w_shared_2,
        "w_se2_s": s_shared_2,
        "w_e13": w_exp_13,
        "w_e13_s": s_exp_13,
        "w_e2": w_exp_2,
        "w_e2_s": s_exp_2,
    }


def _extract_single_rank_np(
    backbone: BackboneLike,
    cfg: Qwen38Config,
    *,
    tp_rank: int = 0,
) -> dict[str, np.ndarray]:
    """Extract one rank's weight slices from a Qwen3.8 backbone into NumPy arrays."""
    buckets: dict[str, list[np.ndarray]] = {
        k: [] for k in WEIGHT_KEYS if k != "final_norm"
    }
    for l_idx in range(cfg.L):
        layer = backbone.layers[l_idx]
        buckets["in_norm"].append(
            _to_np_f32(layer.input_layernorm.weight).reshape(cfg.H)
        )
        buckets["post_norm"].append(
            _to_np_f32(layer.post_attention_layernorm.weight).reshape(cfg.H)
        )
        sub_weights = (
            _extract_attn_sublayer(layer.self_attn, cfg)
            if (l_idx + 1) % cfg.att_every == 0
            else _extract_gdn_sublayer(layer.linear_attn, cfg)
        )
        for k, arr in sub_weights.items():
            buckets[k].append(arr)
        for k, arr in _extract_mlp_sublayer(
            layer.mlp, l_idx, cfg, tp_rank=tp_rank
        ).items():
            buckets[k].append(arr)

    out = {k: np.stack(items, axis=0) for k, items in buckets.items()}
    out["final_norm"] = _to_np_f32(backbone.norm.weight).reshape(cfg.H)
    return out


def _torch_dtype_for_key(key: str) -> torch.dtype:
    """Return canonical PyTorch dtype for extracted weight key `key`."""
    if key in U8_WEIGHT_KEYS:
        return torch.uint8
    if key in F32_WEIGHT_KEYS:
        return torch.float32
    return torch.bfloat16


def _to_rank_torch_tensors(
    rank_dicts: list[dict[str, np.ndarray]],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert per-rank NumPy weight dictionaries to stacked PyTorch tensors."""
    out: dict[str, torch.Tensor] = {}
    for k in WEIGHT_KEYS:
        arr = (
            rank_dicts[0][k][np.newaxis, ...]
            if len(rank_dicts) == 1
            else np.stack([d[k] for d in rank_dicts], axis=0)
        )
        out[k] = (
            torch.from_numpy(arr)
            .to(dtype=_torch_dtype_for_key(k), device=device)
            .contiguous()
        )
    return out


def extract_backbone_weights_torch(
    model: object,
    cfg: Qwen38Config,
    *,
    rank_backbones: Sequence[object] | None = None,
    tp_rank: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Extract Qwen3.8 weights into `(1, ...)` or `(R, ...)` PyTorch tensors."""
    backbones = (
        rank_backbones
        if rank_backbones is not None
        else (
            model.rank_backbones if isinstance(model, MultiRankBackboneHolder) else None
        )
    )
    if backbones is not None:
        rank_dicts = [
            _extract_single_rank_np(
                find_backbone(b),
                cfg,
                tp_rank=r_idx,
            )
            for r_idx, b in enumerate(backbones)
        ]
        return _to_rank_torch_tensors(rank_dicts, device=device)

    backbone = find_backbone(model)
    single = _extract_single_rank_np(backbone, cfg, tp_rank=tp_rank)
    return _to_rank_torch_tensors([single], device=device)
