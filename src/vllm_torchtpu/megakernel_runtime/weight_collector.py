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
"""Generic vLLM weight collector (`VLLMWeightSource` and `load_vllm_weights`).

Walks live vLLM `torch.nn.Module` decoder backbones using the `HFParamDecl` and
`VLLMAlias` declarations on each megakernel `Module` and packs per-rank weights
via `vllm_torchtpu.megakernel.weights.load_weights`:
- When `ex.stack_e` is `False` (default for serving), dynamic routed-expert
  weights (`e13`, `e2` — ~45.0 GiB / 48.3 GB per rank at `R=8` for 60 layers)
  are passed through as their original 3D `torch.nn.Parameter` base buffers
  (`e13[l].data_ptr() == w13.data_ptr()`), avoiding duplicate expert HBM
  allocations; the 3D -> 4D reshape happens as a zero-copy XLA bitcast inside
  `cache.make_flat_pooled_step`.
- Non-expert layer weights are stacked across layers into contiguous base
  buffers (`storage_offset() == 0`, `_is_view() is False`) required by the
  single-call megakernel so `torch_tpu` never emits `tt_jit_execute_as_strided`
  copies on compiled steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.decls import HFParamDecl, VLLMAlias
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32, U8, rup
from vllm_torchtpu.megakernel.weights import (
    ArrayOps,
    load_weights,
    resolve_spec_and_norms,
    slice_alias_targets,
    target_axis_widths,
)

WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"

_TORCH_DTYPE_MAP: dict[Any, torch.dtype] = {
    jnp.dtype(FP8): torch.float8_e4m3fn,
    jnp.dtype(BF16): torch.bfloat16,
    jnp.dtype(F32): torch.float32,
    jnp.dtype(I32): torch.int32,
    jnp.dtype(U8): torch.uint8,
}


def _to_torch_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return _TORCH_DTYPE_MAP[jnp.dtype(dtype)]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"megakernel_runtime.weight_collector: {msg}")


def _own(x: torch.Tensor) -> torch.Tensor:
    """Return `x` as a contiguous base buffer (`storage_offset() == 0`, not a view)."""
    if not x._is_view() and x.storage_offset() == 0 and x.is_contiguous():
        return x
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)


def _stack(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """`torch.stack(xs, 0)` that returns a base buffer instead of a view."""
    return torch.cat([x.unsqueeze(0) for x in xs], dim=0)


def _try_set(obj: Any, name: str, value: Any) -> None:
    """Set an existing attribute on `obj`, skipping read-only properties."""
    if hasattr(obj, name):
        with suppress(AttributeError):
            setattr(obj, name, value)


class TorchArrayOps:
    """`ArrayOps` implementation backed by `torch.Tensor`."""

    def __init__(self, device: torch.device | None = None) -> None:
        self._dev = device

    def slice_axis(
        self, x: torch.Tensor, axis: int, start: int, stop: int
    ) -> torch.Tensor:
        ax = axis % x.ndim
        if start == 0 and stop == int(x.shape[ax]):
            return x
        return x.narrow(ax, start, stop - start)

    def concat(self, xs: Sequence[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.cat(list(xs), dim=axis)

    def reshape(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return x.view(*shape) if x.is_contiguous() else x.reshape(*shape)

    def transpose(self, x: torch.Tensor, perm: tuple[int, ...]) -> torch.Tensor:
        return x.permute(*perm)

    def pad(
        self, x: torch.Tensor, pad_widths: Sequence[tuple[int, int]]
    ) -> torch.Tensor:
        torch_pad: list[int] = []
        for before, after in reversed(pad_widths):
            torch_pad.extend((int(before), int(after)))
        if not any(torch_pad):
            return x
        if x.dtype == torch.float8_e4m3fn:
            return F.pad(x.view(torch.uint8), tuple(torch_pad), value=0).view(
                torch.float8_e4m3fn
            )
        return F.pad(x, tuple(torch_pad), value=0)

    def cast(self, x: torch.Tensor, dtype: Any) -> torch.Tensor:
        td = _to_torch_dtype(dtype)
        return x if x.dtype == td else x.to(td)

    def is_fp8(self, x: torch.Tensor) -> bool:
        return x.dtype == torch.float8_e4m3fn

    def dequantize(
        self, x: torch.Tensor, scale: torch.Tensor, dtype: Any
    ) -> torch.Tensor:
        return (x.float() * scale.float()).to(_to_torch_dtype(dtype))

    def quantize_fp8(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_nk_f32 = x.transpose(0, 1).float()
        amax = w_nk_f32.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
        w_fp8_kn = (
            (w_nk_f32 / scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .transpose(0, 1)
            .contiguous()
        )
        return w_fp8_kn, scale.squeeze(-1).float().contiguous()

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> torch.Tensor:
        return torch.zeros(shape, dtype=_to_torch_dtype(dtype), device=self._dev)

    def stack_layers(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return _stack(xs)

    def own(self, x: torch.Tensor) -> torch.Tensor:
        return _own(x)


def find_backbone(model: Any) -> Any:
    """Locate the decoder backbone (`layers` + `norm`) inside a vLLM model wrapper."""
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
            return cur
    raise ValueError(f"Cannot locate decoder backbone in {type(model).__name__}")


def get_routed_experts(mlp: Any) -> Any:
    """Return the underlying `FusedMoE` / `RoutedExperts` module on `mlp`."""
    experts = mlp.experts
    return getattr(experts, "routed_experts", experts)


def _orient_2d_kn(
    w: torch.Tensor,
    explicit_flip: bool | None,
    source_name: str,
    valid_k: set[int],
    valid_n: set[int],
) -> torch.Tensor:
    """Orient a 2D weight tensor `w` to `[K, N]` layout, raising on ambiguity."""
    r0, r1 = int(w.shape[0]), int(w.shape[1])
    if explicit_flip is True and r0 in valid_k and r1 in valid_n:
        return w
    if explicit_flip is False and r0 in valid_n and r1 in valid_k:
        return w.transpose(0, 1)
    if r0 == r1 and r0 in valid_k and r1 in valid_n:
        raise ValueError(
            f"Ambiguous square weight shape {tuple(w.shape)} for {source_name!r} "
            f"without explicit {WEIGHT_FLIPPED_ATTR!r} attribute"
        )
    if r0 in valid_k and r1 in valid_n:
        return w
    if r0 in valid_n and r1 in valid_k:
        return w.transpose(0, 1)
    raise ValueError(
        f"Unexpected weight shape {tuple(w.shape)} for {source_name!r}; "
        f"expected K in {sorted(valid_k)}, N in {sorted(valid_n)}"
    )


def extract_ecopy_layer(
    layer_idx: int | None,
    layer: Any,
    g: Any,
    ex: ExecSpec,
    dev: torch.device,
    *,
    expert_placement: Any = None,
    experts_path: str = "mlp.experts",
) -> torch.Tensor | None:
    """Build the `[E]` `float32` `ecopy` descriptor vector for `smalls['ecopy']`.

    Returns `None` when no explicit `_ecopy`, `_ecopy_cores`, or matching
    `expert_placement` manifest is present on the module, so `smalls['ecopy']`
    remains all zeros (primary-only routing).
    """
    er = g.ER(ex.R)
    ew = g.EW(ex.R)
    if g.moe_tp or g.e_slots <= er:
        return None
    cur: Any = layer
    for attr in experts_path.split("."):
        cur = getattr(cur, attr, None)
        if cur is None:
            break
    routed = getattr(cur, "routed_experts", cur) if cur is not None else None
    if routed is not None:
        ec_buf = getattr(routed, "_ecopy", None)
        if ec_buf is not None:
            return ec_buf.to(device=dev, dtype=torch.float32).reshape(g.E)
        ec_cores = getattr(routed, "_ecopy_cores", None)
        if ec_cores is not None:
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, ec_cores),
                dtype=torch.float32,
                device=dev,
            )
        if expert_placement is None:
            expert_placement = getattr(routed, "_expert_placement", None)
    if expert_placement is not None:
        if layer_idx is None and routed is not None:
            prefix = getattr(routed, "layer_name", "") or ""
            layer_idx = expert_placement.layer_index(prefix)
        if (
            layer_idx is not None
            and expert_placement.has_copies()
            and int(expert_placement.slots()) == int(ew)
            and int(expert_placement.global_count()) == int(g.E)
            and int(expert_placement.shards()) == int(ex.R)
            and str(layer_idx) in expert_placement.data().get("layers", {})
        ):
            cores = expert_placement.cores(layer_idx)
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, cores),
                dtype=torch.float32,
                device=dev,
            )
    return None


def _get_attr_or_scale_variant(obj: Any, leaf: str) -> tuple[str, Any] | None:
    val = getattr(obj, leaf, None)
    if val is not None:
        return leaf, val
    if leaf.endswith("_scale"):
        alt = f"{leaf}_inv"
        val = getattr(obj, alt, None)
        if val is not None:
            return alt, val
    elif leaf.endswith("_scale_inv"):
        alt = leaf[: -len("_inv")]
        val = getattr(obj, alt, None)
        if val is not None:
            return alt, val
    return None


def _walk_attr(root: Any, dotted: str) -> tuple[Any, str, Any] | None:
    """Resolve `dotted` path on `root` into `(parent_obj, leaf_name, leaf_val)`."""
    parts = dotted.split(".")
    leaf = parts[-1]
    cur = root
    for attr in parts[:-1]:
        nxt = getattr(cur, attr, None)
        if nxt is None:
            return None
        if attr == "experts" and hasattr(nxt, "routed_experts"):
            rexp = nxt.routed_experts
            if rexp is not None and _get_attr_or_scale_variant(rexp, leaf) is not None:
                nxt = rexp
        cur = nxt
    found = _get_attr_or_scale_variant(cur, leaf)
    if found is None:
        return None
    actual_leaf, val = found
    return cur, actual_leaf, val


def extract_kn_fp8_weight(
    proj: Any, hidden_dim: int, out_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(w_kn [hidden_dim, out_dim] fp8, s_n [out_dim] f32)` from `proj`."""
    w = proj.weight.data
    flipped = getattr(proj, WEIGHT_FLIPPED_ATTR, None)
    if flipped is True and tuple(w.shape) == (hidden_dim, out_dim):
        w_kn = w
    elif flipped is False and tuple(w.shape) == (out_dim, hidden_dim):
        w_kn = w.transpose(0, 1)
    elif tuple(w.shape) == (hidden_dim, out_dim) and hidden_dim != out_dim:
        w_kn = w
    elif tuple(w.shape) == (out_dim, hidden_dim):
        w_kn = w.transpose(0, 1)
    else:
        raise ValueError(
            f"Unexpected weight shape {tuple(w.shape)}; expected "
            f"({hidden_dim}, {out_dim}) or ({out_dim}, {hidden_dim})"
        )

    s = getattr(proj, "weight_scale", None)
    if s is None:
        s = getattr(proj, "weight_scale_inv", None)

    if w_kn.dtype == torch.float8_e4m3fn:
        _need(s is not None, "FP8 linear projection is missing weight_scale")
        assert s is not None
        s_1d = s.data.float().reshape(-1).contiguous()
        _need(
            s_1d.shape == (out_dim,),
            f"FP8 weight_scale shape {tuple(s_1d.shape)} != ({out_dim},)",
        )
        return w_kn.contiguous(), s_1d

    w_nk_f32 = w_kn.transpose(0, 1).float()
    amax = w_nk_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    w_fp8_kn = (
        (w_nk_f32 / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    return w_fp8_kn, scale.squeeze(-1).float().contiguous()


def _gather_replicated_sh13(
    w_gu_local: torch.Tensor,
    s_gu_local: torch.Tensor,
    *,
    g: Any,
    ex: Any,
    sh_rank: int,
    tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather partner TP shards of `shared_expert.gate_up_proj` when `sh_rep > 1`."""
    sh_rep = g.sh_rep(ex.R)
    local_isr = g.IS // ex.R
    isr = g.ISR(ex.R)
    _need(
        tp_all_gather_fn is not None,
        f"tp_all_gather_fn is required when shared_expert is TP-{ex.R} sharded "
        f"(local_isr={local_isr}) but megakernel uses isr={isr} (sh_rep={sh_rep})",
    )
    assert tp_all_gather_fn is not None
    w_gu_all = tp_all_gather_fn(w_gu_local)
    s_gu_all = tp_all_gather_fn(s_gu_local)
    _need(
        tuple(w_gu_all.shape) == (ex.R, g.H, 2 * local_isr),
        f"tp_all_gather_fn(gate_up) shape {tuple(w_gu_all.shape)} != "
        f"({ex.R}, {g.H}, {2 * local_isr})",
    )
    _need(
        tuple(s_gu_all.shape) == (ex.R, 2 * local_isr),
        f"tp_all_gather_fn(gate_up_scale) shape {tuple(s_gu_all.shape)} != "
        f"({ex.R}, {2 * local_isr})",
    )
    p0 = sh_rank * sh_rep
    p1 = p0 + sh_rep
    w_gate = w_gu_all[p0:p1, :, :local_isr].permute(1, 0, 2).reshape(g.H, isr)
    w_up = w_gu_all[p0:p1, :, local_isr:].permute(1, 0, 2).reshape(g.H, isr)
    s_gate = s_gu_all[p0:p1, :local_isr].reshape(isr)
    s_up = s_gu_all[p0:p1, local_isr:].reshape(isr)
    return (
        torch.cat([w_gate, w_up], dim=1).contiguous(),
        torch.cat([s_gate, s_up], dim=0).contiguous(),
    )


def _gather_replicated_sh2(
    w_down_local: torch.Tensor,
    s_down_local: torch.Tensor,
    *,
    g: Any,
    ex: Any,
    sh_rank: int,
    tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather partner TP shards of `shared_expert.down_proj` when `sh_rep > 1`."""
    sh_rep = g.sh_rep(ex.R)
    local_isr = g.IS // ex.R
    isr = g.ISR(ex.R)
    _need(
        tp_all_gather_fn is not None,
        f"tp_all_gather_fn is required when shared_expert.down_proj is TP-{ex.R} "
        f"sharded (rows={local_isr}) but megakernel uses isr={isr}",
    )
    assert tp_all_gather_fn is not None
    w_down_all = tp_all_gather_fn(w_down_local)
    s_down_all = tp_all_gather_fn(s_down_local)
    _need(
        tuple(w_down_all.shape) == (ex.R, local_isr, g.H),
        f"tp_all_gather_fn(down) shape {tuple(w_down_all.shape)} != "
        f"({ex.R}, {local_isr}, {g.H})",
    )
    _need(
        tuple(s_down_all.shape) == (ex.R, g.H),
        f"tp_all_gather_fn(down_scale) shape {tuple(s_down_all.shape)} != "
        f"({ex.R}, {g.H})",
    )
    p0 = sh_rank * sh_rep
    p1 = p0 + sh_rep
    w_grp_f32 = (w_down_all[p0:p1].float() * s_down_all[p0:p1].unsqueeze(1)).reshape(
        isr, g.H
    )
    amax = w_grp_f32.abs().amax(dim=0, keepdim=True)
    scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    w_sh2 = (
        (w_grp_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    )
    return w_sh2, scale.squeeze(0).float().contiguous()


def pack_router_and_shared_layer(
    mlp: Any,
    g: Any,
    ex: ExecSpec,
    *,
    tp_rank: int = 0,
    tk_router: int,
    tk_sh13: int,
    tk_sh2: int,
    tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract router (`bf16`), shared expert `sh13`/`sh2` (`fp8`), and scales."""
    isr = g.ISR(ex.R)
    sh_rep = g.sh_rep(ex.R)
    local_isr = g.IS // ex.R
    sh_rank = tp_rank // sh_rep
    w_gu = getattr(mlp, "_fused_gate_gu_fp8_weight", None)
    if w_gu is not None:
        s_gu = mlp._fused_gate_gu_fp8_scale.data.float().reshape(-1)
        if hasattr(mlp, "_fused_se_down_fp8_weight"):
            w_down = mlp._fused_se_down_fp8_weight.data
            s_down = mlp._fused_se_down_fp8_scale.data.float().reshape(g.H).contiguous()
        else:
            down_proj = mlp.shared_expert.down_proj
            down_rows = int(
                down_proj.weight.shape[0]
                if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
                or down_proj.weight.shape[1] == g.H
                else down_proj.weight.shape[1]
            )
            w_down, s_down = extract_kn_fp8_weight(down_proj, down_rows, g.H)
        if hasattr(mlp, "_fused_se_gate_weight"):
            sh_gate = mlp._fused_se_gate_weight.data.float().reshape(g.H).contiguous()
        else:
            sh_gate = (
                mlp.shared_expert_gate.weight.data.float().reshape(g.H).contiguous()
            )

        w_router_fp8 = w_gu.data[:, : g.E]
        s_router = s_gu[: g.E]
        w_router_bf16 = (
            (w_router_fp8.float() * s_router.unsqueeze(0))
            .to(torch.bfloat16)
            .contiguous()
            .view(g.H // tk_router, tk_router, g.E)
        )

        n_cols = int(w_gu.shape[1])
        if n_cols == g.E + 2 * g.IS:
            g0 = g.E + sh_rank * isr
            u0 = g.E + g.IS + sh_rank * isr
            w_sh13 = torch.cat(
                [w_gu.data[:, g0 : g0 + isr], w_gu.data[:, u0 : u0 + isr]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu[g0 : g0 + isr], s_gu[u0 : u0 + isr]], dim=0
            ).contiguous()
        elif n_cols == g.E + 2 * isr:
            w_sh13 = w_gu.data[:, g.E : g.E + 2 * isr].contiguous()
            s_sh13 = s_gu[g.E : g.E + 2 * isr].contiguous()
        elif sh_rep > 1 and n_cols == g.E + 2 * local_isr:
            w_sh13, s_sh13 = _gather_replicated_sh13(
                w_gu.data[:, g.E : g.E + 2 * local_isr].contiguous(),
                s_gu[g.E : g.E + 2 * local_isr].contiguous(),
                g=g,
                ex=ex,
                sh_rank=sh_rank,
                tp_all_gather_fn=tp_all_gather_fn,
            )
        else:
            raise ValueError(
                f"Unexpected _fused_gate_gu_fp8_weight shape {tuple(w_gu.shape)}"
            )

        if int(w_down.shape[0]) == g.IS:
            r0 = sh_rank * isr
            w_sh2 = w_down[r0 : r0 + isr, :].contiguous()
        elif int(w_down.shape[0]) == isr:
            w_sh2 = w_down.contiguous()
        elif sh_rep > 1 and int(w_down.shape[0]) == local_isr:
            w_sh2, s_down = _gather_replicated_sh2(
                w_down.contiguous(),
                s_down,
                g=g,
                ex=ex,
                sh_rank=sh_rank,
                tp_all_gather_fn=tp_all_gather_fn,
            )
        else:
            raise ValueError(
                f"Unexpected _fused_se_down_fp8_weight shape {tuple(w_down.shape)}"
            )
    else:
        gate_w = mlp.gate.weight.data
        if tuple(gate_w.shape) == (g.E, g.H):
            gate_w = gate_w.transpose(0, 1)
        w_router_bf16 = (
            gate_w.to(torch.bfloat16)
            .contiguous()
            .view(g.H // tk_router, tk_router, g.E)
        )

        gu_proj = mlp.shared_expert.gate_up_proj
        down_proj = mlp.shared_expert.down_proj
        gu_cols = int(
            gu_proj.weight.shape[1]
            if getattr(gu_proj, WEIGHT_FLIPPED_ATTR, False)
            or gu_proj.weight.shape[0] == g.H
            else gu_proj.weight.shape[0]
        )
        w_gu_only, s_gu_only = extract_kn_fp8_weight(gu_proj, g.H, gu_cols)
        if gu_cols == 2 * g.IS:
            g0 = sh_rank * isr
            u0 = g.IS + sh_rank * isr
            w_sh13 = torch.cat(
                [w_gu_only[:, g0 : g0 + isr], w_gu_only[:, u0 : u0 + isr]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu_only[g0 : g0 + isr], s_gu_only[u0 : u0 + isr]], dim=0
            ).contiguous()
        elif gu_cols == 2 * isr:
            w_sh13 = w_gu_only.contiguous()
            s_sh13 = s_gu_only.contiguous()
        elif sh_rep > 1 and gu_cols == 2 * local_isr:
            w_sh13, s_sh13 = _gather_replicated_sh13(
                w_gu_only,
                s_gu_only,
                g=g,
                ex=ex,
                sh_rank=sh_rank,
                tp_all_gather_fn=tp_all_gather_fn,
            )
        else:
            raise ValueError(f"Unexpected shared_expert.gate_up_proj cols {gu_cols}")

        down_rows = int(
            down_proj.weight.shape[0]
            if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
            or down_proj.weight.shape[1] == g.H
            else down_proj.weight.shape[1]
        )
        w_down_only, s_down = extract_kn_fp8_weight(down_proj, down_rows, g.H)
        if down_rows == g.IS:
            r0 = sh_rank * isr
            w_sh2 = w_down_only[r0 : r0 + isr, :].contiguous()
        elif down_rows == isr:
            w_sh2 = w_down_only.contiguous()
        elif sh_rep > 1 and down_rows == local_isr:
            w_sh2, s_down = _gather_replicated_sh2(
                w_down_only,
                s_down,
                g=g,
                ex=ex,
                sh_rank=sh_rank,
                tp_all_gather_fn=tp_all_gather_fn,
            )
        else:
            raise ValueError(f"Unexpected shared_expert.down_proj rows {down_rows}")

        sh_gate = mlp.shared_expert_gate.weight.data.float().reshape(g.H).contiguous()

    return {
        "router": w_router_bf16,
        "sh13": w_sh13.view(g.H // tk_sh13, tk_sh13, 2 * isr),
        "sh2": w_sh2.view(isr // tk_sh2, tk_sh2, g.H),
        "sh13_s": s_sh13,
        "sh2_s": s_down,
        "sh_gate": sh_gate,
    }


class VLLMWeightSource:
    """`WeightSource` backed by a live vLLM decoder `backbone` (`layers` + `norm`)."""

    def __init__(
        self,
        backbone: Any,
        g: Any,
        *,
        tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        _need(
            len(backbone.layers) == int(g.L),
            f"expected {g.L} layers, got {len(backbone.layers)}",
        )
        self._backbone = backbone
        self._dev = backbone.norm.weight.device
        self._ops = TorchArrayOps(device=self._dev)
        self._tp_all_gather_fn = tp_all_gather_fn
        self._g = g

    @property
    def ops(self) -> ArrayOps:
        return self._ops

    def has_layer_alias(self, lyr: int, ord_idx: int, source_name: str) -> bool:
        del ord_idx
        res = _walk_attr(self._backbone.layers[lyr], source_name)
        if res is None:
            return False
        _, _, val = res
        return isinstance(val, torch.Tensor) or isinstance(
            getattr(val, "weight", None), torch.Tensor
        )

    def _read_scale(
        self, layer: Any, proj_or_parent: Any, alias: VLLMAlias
    ) -> torch.Tensor | None:
        if hasattr(proj_or_parent, "weight"):
            for s_attr in ("weight_scale_inv", "weight_scale"):
                s = getattr(proj_or_parent, s_attr, None)
                if isinstance(s, torch.Tensor):
                    return s.data
        if alias.scale_source is not None:
            res = _walk_attr(layer, alias.scale_source)
            if res is not None and isinstance(res[2], torch.Tensor):
                return res[2].data
        return None

    def resolve_alias(
        self,
        lyr: int,
        ord_idx: int,
        alias: VLLMAlias,
        target_decls: Sequence[HFParamDecl],
        scale_decls: Mapping[str, HFParamDecl],
        *,
        R: int,
        tp_rank: int,
        fuse_groups: Mapping[str, tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        del ord_idx
        layer = self._backbone.layers[lyr]
        res = _walk_attr(layer, alias.source)
        _need(res is not None, f"Layer {lyr}: missing {alias.source!r}")
        assert res is not None
        parent, _, val = res
        is_direct_tensor = isinstance(val, torch.Tensor)
        if is_direct_tensor:
            proj_obj = parent
            w = val.detach()
        else:
            proj_obj = val
            w = val.weight.detach()

        d0 = target_decls[0]
        s_buf: torch.Tensor | None = None

        if w.ndim == 3 and d0.transform.expert_axis == 0:
            _need(
                w.is_contiguous(),
                "expert weights must be contiguous for zero-copy views",
            )
            leaf_name = alias.source.rsplit(".", 1)[-1]
            raw_stash = getattr(proj_obj, "_mk_nvfp4_raw", None)
            gscale_override = None
            if raw_stash is not None:
                w13_src, bs13_src, g13_src, w2_src, bs2_src, g2_src = raw_stash
                if leaf_name == "w13_weight":
                    w = w13_src
                    s_raw = bs13_src
                    gscale_override = g13_src
                elif leaf_name == "w2_weight":
                    w = w2_src
                    s_raw = bs2_src
                    gscale_override = g2_src
                else:
                    s_raw = self._read_scale(layer, proj_obj, alias)
            else:
                s_raw = self._read_scale(layer, proj_obj, alias)

            is_nvfp4 = (
                jnp.dtype(d0.dtype) == jnp.dtype(U8)
                or getattr(proj_obj, f"{leaf_name}_scale_2", None) is not None
                or gscale_override is not None
            )
            if is_nvfp4:
                ew = int(w.shape[0])
                k_dim = int(d0.shape[1]) * 2
                s_decl = scale_decls[alias.scale_targets[0]]
                _, n_dim = target_axis_widths(s_decl, R, -1)
                w_buf, pre_fused = _pack_nvfp4_w(
                    w, s_raw, ew, k_dim, n_dim, name=leaf_name
                )
                if gscale_override is not None:
                    gscale = gscale_override
                    if not gscale.is_floating_point() or gscale.numel() not in (
                        ew,
                        2 * ew,
                        ew * n_dim,
                    ):
                        raise MegakernelConfigError(
                            f"{leaf_name}_scale_2: expected float tensor of {ew} or "
                            f"{2 * ew} elements, got shape {tuple(gscale.shape)} and "
                            f"dtype {gscale.dtype} {_NVFP4_STOCK_MSG}"
                        )
                    if gscale.numel() == 2 * ew:
                        ir = n_dim // 2
                        s_buf = (
                            gscale.data.float()
                            .reshape(ew, 2, 1)
                            .expand(ew, 2, ir)
                            .reshape(ew, 2 * ir)
                            .contiguous()
                        )
                    else:
                        s_buf = (
                            gscale.data.float()
                            .reshape(ew, -1)
                            .expand(ew, n_dim)
                            .contiguous()
                        )
                else:
                    s_buf = _extract_nvfp4_global_scale(
                        proj_obj,
                        s_raw,
                        ew=ew,
                        n_dim=n_dim,
                        pre_fused=pre_fused,
                        name=leaf_name,
                    )
                target_dev = getattr(proj_obj, "w13_weight", w).device
                if w_buf.device != target_dev:
                    w_buf = _own(w_buf.to(target_dev))
                if s_buf.device != target_dev:
                    s_buf = s_buf.to(target_dev)

                resolved = getattr(proj_obj, "_mk_nvfp4_resolved", set())
                resolved.add(leaf_name)
                proj_obj._mk_nvfp4_resolved = resolved
                if {"w13_weight", "w2_weight"}.issubset(resolved):
                    if hasattr(proj_obj, "_mk_nvfp4_raw"):
                        delattr(proj_obj, "_mk_nvfp4_raw")
                    delattr(proj_obj, "_mk_nvfp4_resolved")
            else:
                _need(
                    s_raw is not None, f"Expert weight {alias.source!r} missing scale"
                )
                assert s_raw is not None
                exp_e, exp_n = int(w.shape[0]), int(w.shape[-1])
                _need(
                    s_raw.numel() == exp_e * exp_n,
                    f"Expert scale for {alias.source!r} has shape {tuple(s_raw.shape)} "
                    f"({s_raw.numel()} elems), expected ({exp_e}, {exp_n})",
                )
                s_buf = s_raw.float().reshape(exp_e, exp_n).contiguous()
                w_buf = w
        elif len(target_decls) == 1 and len(d0.shape) == 1:
            w_buf = w.reshape(-1)
        elif w.ndim == 2 and not d0.transform.flatten_tail:
            sh_rep = getattr(self._g, "sh_rep", lambda r: 1)(R)
            if self._tp_all_gather_fn is not None and sh_rep > 1:
                local_isr = self._g.IS // R
                sh_rank = tp_rank // sh_rep
                if "shared_expert.gate_up_proj" in alias.source:
                    gu_cols = int(
                        w.shape[1]
                        if getattr(proj_obj, WEIGHT_FLIPPED_ATTR, False)
                        or w.shape[0] == self._g.H
                        else w.shape[0]
                    )
                    w_gu, s_gu = extract_kn_fp8_weight(proj_obj, self._g.H, gu_cols)
                    if gu_cols == 2 * local_isr:
                        w_sh13, s_sh13 = _gather_replicated_sh13(
                            w_gu,
                            s_gu,
                            g=self._g,
                            ex=SimpleNamespace(R=R),
                            sh_rank=sh_rank,
                            tp_all_gather_fn=self._tp_all_gather_fn,
                        )
                        return slice_alias_targets(
                            self._ops,
                            alias,
                            w_sh13.contiguous(),
                            s_sh13.contiguous(),
                            target_decls,
                            scale_decls,
                            R=R,
                            tp_rank=tp_rank,
                            fuse_groups=fuse_groups,
                        )
                elif "shared_expert.down_proj" in alias.source:
                    down_rows = int(
                        w.shape[0]
                        if getattr(proj_obj, WEIGHT_FLIPPED_ATTR, False)
                        or w.shape[1] == self._g.H
                        else w.shape[1]
                    )
                    w_down, s_down = extract_kn_fp8_weight(
                        proj_obj, down_rows, self._g.H
                    )
                    if down_rows == local_isr:
                        w_sh2, s_down_g = _gather_replicated_sh2(
                            w_down,
                            s_down,
                            g=self._g,
                            ex=SimpleNamespace(R=R),
                            sh_rank=sh_rank,
                            tp_all_gather_fn=self._tp_all_gather_fn,
                        )
                        return slice_alias_targets(
                            self._ops,
                            alias,
                            w_sh2.contiguous(),
                            s_down_g.contiguous(),
                            target_decls,
                            scale_decls,
                            R=R,
                            tp_rank=tp_rank,
                            fuse_groups=fuse_groups,
                        )
            k_full, k_shard = target_axis_widths(d0, R, 0)
            valid_k = {k_full, k_shard}
            n_widths = [target_axis_widths(d, R, -1) for d in target_decls]
            valid_n = {
                sum(fw for fw, _ in n_widths),
                sum(sw for _, sw in n_widths),
            }
            if is_direct_tensor:
                explicit_flip = getattr(
                    val,
                    WEIGHT_FLIPPED_ATTR,
                    getattr(parent, WEIGHT_FLIPPED_ATTR, alias.flipped),
                )
            else:
                explicit_flip = getattr(proj_obj, WEIGHT_FLIPPED_ATTR, None)
            w_kn = _orient_2d_kn(w, explicit_flip, alias.source, valid_k, valid_n)

            if w_kn.dtype == torch.float8_e4m3fn:
                s_raw = self._read_scale(layer, proj_obj, alias)
                _need(
                    s_raw is not None,
                    f"FP8 projection {alias.source!r} is missing weight_scale",
                )
                assert s_raw is not None
                exp_n = int(w_kn.shape[1])
                _need(
                    s_raw.numel() == exp_n,
                    f"FP8 weight_scale for {alias.source!r} has shape "
                    f"{tuple(s_raw.shape)} ({s_raw.numel()} elems), expected {exp_n}",
                )
                s_buf = s_raw.float().reshape(exp_n).contiguous()
            w_buf = w_kn.contiguous()
        else:
            w_buf = w

        return slice_alias_targets(
            self._ops,
            alias,
            w_buf,
            s_buf,
            target_decls,
            scale_decls,
            R=R,
            tp_rank=tp_rank,
            fuse_groups=fuse_groups,
        )

    def read_model_param(self, decl: HFParamDecl, *, R: int, tp_rank: int) -> Any:
        del R, tp_rank
        candidates = [al.source for al in decl.vllm_aliases] + [decl.hf_name]
        for cand in candidates:
            for path in (cand, cand.removeprefix("model.")):
                res = _walk_attr(self._backbone, path)
                if res is not None:
                    val = res[2]
                    t = (
                        val.detach()
                        if isinstance(val, torch.Tensor)
                        else val.weight.detach()
                    )
                    self._dev = t.device
                    self._ops._dev = t.device
                    return t
        raise ValueError(f"Cannot resolve model parameter {decl.name!r} on backbone")

    def read_layer_ecopy(
        self,
        lyr: int,
        g: Any,
        ex: ExecSpec,
        *,
        expert_placement: Any = None,
        experts_path: str = "mlp.experts",
    ) -> torch.Tensor | None:
        return extract_ecopy_layer(
            lyr,
            self._backbone.layers[lyr],
            g,
            ex,
            self._dev,
            expert_placement=expert_placement,
            experts_path=experts_path,
        )

    def supports_passthrough(self) -> bool:
        return True


def load_vllm_weights(
    model_def: Any,
    backbone: Any,
    ex: ExecSpec,
    *,
    g: Any = None,
    tp_rank: int = 0,
    expert_placement: Any = None,
    tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Extract and pack all layers of `backbone` via `load_weights`."""
    _, geom, _, _, _ = resolve_spec_and_norms(model_def, ex, g=g)
    source = VLLMWeightSource(backbone, geom, tp_all_gather_fn=tp_all_gather_fn)
    return load_weights(
        model_def,
        ex,
        source,
        g=geom,
        tp_rank=tp_rank,
        expert_placement=expert_placement,
    )


_NVFP4_STOCK_MSG = (
    "(stock process_weights_after_loading already requantized; see unit B)"
)


def _pack_nvfp4_w(
    w_raw: torch.Tensor | None,
    bscale_raw: torch.Tensor | None,
    ew: int,
    k_dim: int,
    n_dim: int,
    *,
    name: str,
) -> tuple[torch.Tensor, bool]:
    """Pack raw uint8 NVFP4 weights and block scales into `[ew, K // 2, n_raw]`."""
    k_half = k_dim // 2
    n_raw = n_dim + rup(n_dim // 8, 128)
    if w_raw is None or w_raw.dtype != torch.uint8:
        got_w = None if w_raw is None else (tuple(w_raw.shape), w_raw.dtype)
        raise MegakernelConfigError(
            f"{name}: expected raw torch.uint8 weight tensor, "
            f"got {got_w} {_NVFP4_STOCK_MSG}"
        )
    w_u8 = w_raw.data
    if tuple(w_u8.shape) == (ew, k_half, n_raw):
        return w_u8.contiguous(), True
    if tuple(w_u8.shape) == (ew, n_dim, k_half):
        w_kn = w_u8.transpose(1, 2).contiguous()
    elif tuple(w_u8.shape) == (ew, k_half, n_dim):
        w_kn = w_u8.contiguous()
    else:
        raise MegakernelConfigError(
            f"{name}: expected raw uint8 shape {(ew, n_dim, k_half)} or "
            f"{(ew, k_half, n_dim)}, got {tuple(w_u8.shape)} {_NVFP4_STOCK_MSG}"
        )
    if bscale_raw is None or bscale_raw.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        got_bs = (
            None if bscale_raw is None else (tuple(bscale_raw.shape), bscale_raw.dtype)
        )
        raise MegakernelConfigError(
            f"{name}_scale: expected raw uint8 or float8_e4m3fn "
            f"block scale, got {got_bs} {_NVFP4_STOCK_MSG}"
        )
    bs_u8 = bscale_raw.data.view(torch.uint8)
    if tuple(bs_u8.shape) == (ew, n_dim, k_dim // 16):
        bs_kn = bs_u8.transpose(1, 2).contiguous()
    elif tuple(bs_u8.shape) == (ew, k_dim // 16, n_dim):
        bs_kn = bs_u8.contiguous()
    else:
        raise MegakernelConfigError(
            f"{name}_scale: expected raw block scale shape "
            f"{(ew, n_dim, k_dim // 16)} or {(ew, k_dim // 16, n_dim)}, "
            f"got {tuple(bs_u8.shape)} {_NVFP4_STOCK_MSG}"
        )
    bs_cols = bs_kn.reshape(ew, k_half, n_dim // 8)
    pad_cols = n_raw - n_dim - (n_dim // 8)
    parts = [w_kn, bs_cols]
    if pad_cols > 0:
        parts.append(
            torch.zeros(
                (ew, k_half, pad_cols),
                dtype=torch.uint8,
                device=w_u8.device,
            )
        )
    return _own(torch.cat(parts, dim=-1)), False


def _extract_nvfp4_global_scale(
    proj_obj: Any,
    bscale_raw: torch.Tensor | None,
    *,
    ew: int,
    n_dim: int,
    pre_fused: bool,
    name: str,
) -> torch.Tensor:
    gscale = getattr(proj_obj, f"{name}_scale_2", None)
    if not pre_fused and gscale is None:
        raise MegakernelConfigError(
            "w13_weight_scale_2 and w2_weight_scale_2 must be present for raw "
            f"NVFP4 packing {_NVFP4_STOCK_MSG}"
        )
    if gscale is not None:
        if not gscale.is_floating_point() or gscale.numel() not in (
            ew,
            2 * ew,
            ew * n_dim,
        ):
            raise MegakernelConfigError(
                f"{name}_scale_2: expected float tensor of {ew} or {2 * ew} elements, "
                f"got shape {tuple(gscale.shape)} and dtype {gscale.dtype} "
                f"{_NVFP4_STOCK_MSG}"
            )
        if gscale.numel() == 2 * ew:
            ir = n_dim // 2
            return (
                gscale.data.float()
                .reshape(ew, 2, 1)
                .expand(ew, 2, ir)
                .reshape(ew, 2 * ir)
                .contiguous()
            )
        return gscale.data.float().reshape(ew, -1).expand(ew, n_dim).contiguous()
    _need(bscale_raw is not None, f"Expert weight {name!r} missing scale")
    assert bscale_raw is not None
    return bscale_raw.data.float().reshape(ew, n_dim).contiguous()


def pack_expert_views_layer(
    mlp: Any,
    g: Any,
    ex: ExecSpec,
    tk_e13: int,
    tk_e2: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-layer routed expert weights (`e13`, `e2`) and scales (`es_mk`)."""
    experts = get_routed_experts(mlp)
    w13 = experts.w13_weight
    w2 = experts.w2_weight
    ew = g.EW(ex.R)
    ir = g.IR(ex.R)
    _need(
        int(w13.shape[0]) == ew,
        f"experts.w13_weight.shape[0]={w13.shape[0]} != EW={ew}",
    )
    _need(
        w13.is_contiguous() and w2.is_contiguous(),
        "expert weights must be contiguous for zero-copy views",
    )

    is_nvfp4 = (
        str(getattr(g, "expert_quant", "fp8")) == "nvfp4"
        or getattr(experts, "w13_weight_scale_2", None) is not None
        or getattr(experts, "_mk_nvfp4_raw", None) is not None
    )
    if is_nvfp4:
        target_dev = w13.device
        raw_stash = getattr(experts, "_mk_nvfp4_raw", None)
        if raw_stash is not None:
            delattr(experts, "_mk_nvfp4_raw")
            w13_src, bs13_src, g13, w2_src, bs2_src, g2 = raw_stash
        else:
            w13_src = w13
            bs13_src = getattr(experts, "w13_weight_scale", None)
            g13 = getattr(experts, "w13_weight_scale_2", None)
            w2_src = w2
            bs2_src = getattr(experts, "w2_weight_scale", None)
            g2 = getattr(experts, "w2_weight_scale_2", None)

        e13_fused, pre13 = _pack_nvfp4_w(
            w13_src, bs13_src, ew, g.H, 2 * ir, name="w13_weight"
        )
        e2_fused, pre2 = _pack_nvfp4_w(w2_src, bs2_src, ew, ir, g.H, name="w2_weight")
        if not ex.stack_e:
            e13_view, e2_view = e13_fused, e2_fused
        else:
            k13_half, n13_raw = int(e13_fused.shape[1]), int(e13_fused.shape[2])
            k2_half, n2_raw = int(e2_fused.shape[1]), int(e2_fused.shape[2])
            e13_view = e13_fused.view(ew, k13_half // tk_e13, tk_e13, n13_raw)
            e2_view = e2_fused.view(ew, k2_half // tk_e2, tk_e2, n2_raw)

        if not (pre13 and pre2) and (g13 is None or g2 is None):
            raise MegakernelConfigError(
                "w13_weight_scale_2 and w2_weight_scale_2 must be present for raw "
                f"NVFP4 packing {_NVFP4_STOCK_MSG}"
            )
        if g13 is not None and (
            not g13.is_floating_point() or g13.numel() not in (ew, 2 * ew, ew * 2 * ir)
        ):
            raise MegakernelConfigError(
                f"w13_weight_scale_2: expected float tensor of {ew} or "
                f"{2 * ew} elements, got shape {tuple(g13.shape)} and "
                f"dtype {g13.dtype} {_NVFP4_STOCK_MSG}"
            )
        if g2 is not None and (
            not g2.is_floating_point() or g2.numel() not in (ew, ew * g.H)
        ):
            raise MegakernelConfigError(
                f"w2_weight_scale_2: expected float tensor of {ew} elements, "
                f"got shape {tuple(g2.shape)} and dtype {g2.dtype} {_NVFP4_STOCK_MSG}"
            )
        if g13 is None:
            s13 = experts.w13_weight_scale.data.float().reshape(ew, 2 * ir)
        elif g13.numel() == 2 * ew:
            s13 = (
                g13.data.float()
                .reshape(ew, 2, 1)
                .expand(ew, 2, ir)
                .reshape(ew, 2 * ir)
                .contiguous()
            )
        else:
            s13 = g13.data.float().reshape(ew, -1).expand(ew, 2 * ir).contiguous()

        s2 = (
            g2.data.float().reshape(ew, -1).expand(ew, g.H).contiguous()
            if g2 is not None
            else experts.w2_weight_scale.data.float().reshape(ew, g.H)
        )
        if e13_view.device != target_dev:
            e13_view = _own(e13_view.to(target_dev))
        if e2_view.device != target_dev:
            e2_view = _own(e2_view.to(target_dev))
        if s13.device != target_dev:
            s13 = s13.to(target_dev)
        if s2.device != target_dev:
            s2 = s2.to(target_dev)
    else:
        if not ex.stack_e:
            e13_view = w13
            e2_view = w2
        else:
            t13 = g.H // tk_e13
            t2 = ir // tk_e2
            e13_view = w13.view(ew, t13, tk_e13, 2 * ir)
            e2_view = w2.view(ew, t2, tk_e2, g.H)
        _need(
            e13_view.data_ptr() == w13.data_ptr()
            and e2_view.data_ptr() == w2.data_ptr(),
            "expert views must share storage with w13_weight / w2_weight",
        )
        s13_raw = getattr(experts, "w13_weight_scale_inv", None)
        if s13_raw is None:
            s13_raw = experts.w13_weight_scale
        s2_raw = getattr(experts, "w2_weight_scale_inv", None)
        if s2_raw is None:
            s2_raw = experts.w2_weight_scale
        s13 = s13_raw.data.float().reshape(ew, 2 * ir)
        s2 = s2_raw.data.float().reshape(ew, g.H)
    es_cat = torch.cat([s13, s2], dim=-1).contiguous()
    es_mk = es_cat.unsqueeze(1).contiguous()
    return e13_view, e2_view, es_mk


def reshard_experts_ep_to_tp_local(
    w13_all: torch.Tensor,
    w2_all: torch.Tensor,
    s13_all: torch.Tensor,
    s2_all: torch.Tensor,
    tp_rank: int,
    g: Any,
    ex: ExecSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice global `E`-expert weights/scales into `tp_rank`'s TP-sharded tensors."""
    ir = g.I // ex.R
    g0 = tp_rank * ir
    u0 = g.I + tp_rank * ir

    w13_tp = _own(
        torch.cat([w13_all[:, :, g0 : g0 + ir], w13_all[:, :, u0 : u0 + ir]], dim=-1)
    )
    w2_tp = _own(w2_all[:, g0 : g0 + ir, :])

    s13_flat = s13_all.float().reshape(g.E, 2 * g.I)
    s2_flat = s2_all.float().reshape(g.E, g.H)
    s13_tp = _own(
        torch.cat(
            [s13_flat[:, g0 : g0 + ir], s13_flat[:, u0 : u0 + ir]], dim=-1
        ).reshape(g.E, 1, 1, 2 * ir)
    )
    s2_tp = _own(s2_flat.reshape(g.E, 1, 1, g.H))
    return w13_tp, w2_tp, s13_tp, s2_tp


def apply_tp_moe_flags(experts: Any, *, prebuild_moe_fn: Any = None) -> None:
    """Update a `FusedMoE` module's flags after in-place EP->TP resharding."""
    targets = [experts]
    routed = getattr(experts, "routed_experts", None)
    if routed is not None and routed is not experts:
        targets.append(routed)
    for tgt in targets:
        moe_cfg = getattr(tgt, "moe_config", None)
        if moe_cfg is not None:
            p_cfg = getattr(moe_cfg, "moe_parallel_config", None)
            if p_cfg is not None:
                _try_set(p_cfg, "use_ep", False)
                if (
                    getattr(p_cfg, "ep_size", 1) > 1
                    and getattr(p_cfg, "tp_size", 1) == 1
                ):
                    _try_set(p_cfg, "tp_size", p_cfg.ep_size)
                    _try_set(p_cfg, "tp_rank", getattr(p_cfg, "ep_rank", 0))
                    _try_set(p_cfg, "ep_size", 1)
                    _try_set(p_cfg, "ep_rank", 0)
                _try_set(p_cfg, "sp_size", 1)
                _try_set(p_cfg, "is_sequence_parallel", False)
            _try_set(moe_cfg, "use_ep", False)
            _try_set(moe_cfg, "is_sequence_parallel", False)
        _try_set(tgt, "_tpu_fused_ep_eligible", False)
        for buf in ("_experts_start", "_expert_map", "expert_mask"):
            if buf in getattr(tgt, "_buffers", {}):
                tgt.register_buffer(buf, None, persistent=False)
            else:
                _try_set(tgt, buf, None)
        _try_set(tgt, "_gmm_primary_experts", None)
        _try_set(tgt, "_lowtok_has_copies", False)
        if hasattr(tgt, "global_num_experts"):
            _try_set(tgt, "local_num_experts", tgt.global_num_experts)
        for qm_attr in ("quant_method", "_quant_method"):
            qm = getattr(tgt, qm_attr, None)
            if qm is not None:
                _try_set(qm, "_tpu_fused_ep_eligible", False)
                _try_set(qm, "_moe_lowtok_scores_op", None)
                _try_set(qm, "_fused_moe_ep_op", None)
                _try_set(qm, "_use_adaptive_fused_moe", False)
        if (
            prebuild_moe_fn is not None
            and moe_cfg is not None
            and hasattr(moe_cfg, "experts_per_token")
            and hasattr(tgt, "activation")
        ):
            act_val = getattr(tgt.activation, "value", tgt.activation)
            prebuild_moe_fn(
                topk=moe_cfg.experts_per_token,
                activation=act_val,
                use_ep=False,
            )


def reshard_layer_experts_ep_to_tp(
    mlp: Any,
    tp_rank: int,
    g: Any,
    ex: ExecSpec,
    *,
    all_gather_fn: Any = None,
    prebuild_moe_fn: Any = None,
) -> None:
    """Reshard `mlp.experts` from EP (`[ER, H, 2*I]`) to TP (`[E, H, 2*IR]`)."""
    experts = get_routed_experts(mlp)
    er = g.ER(ex.R)
    w13_ep = experts.w13_weight.data[:er].contiguous()
    w2_ep = experts.w2_weight.data[:er].contiguous()

    s13_attr = (
        "w13_weight_scale_inv"
        if hasattr(experts, "w13_weight_scale_inv")
        else "w13_weight_scale"
    )
    s2_attr = (
        "w2_weight_scale_inv"
        if hasattr(experts, "w2_weight_scale_inv")
        else "w2_weight_scale"
    )
    s13_ep = getattr(experts, s13_attr).data[:er].reshape(er, 2 * g.I).contiguous()
    s2_ep = getattr(experts, s2_attr).data[:er].reshape(er, g.H).contiguous()

    if all_gather_fn is None:
        _need(
            ex.R == 1,
            "all_gather_fn must be provided to reshard_layer_experts_ep_to_tp "
            "when R > 1",
        )
        w13_all, w2_all, s13_all, s2_all = w13_ep, w2_ep, s13_ep, s2_ep
    else:
        w13_all = all_gather_fn(w13_ep)
        w2_all = all_gather_fn(w2_ep)
        s13_all = all_gather_fn(s13_ep)
        s2_all = all_gather_fn(s2_ep)

    w13_tp, w2_tp, s13_tp, s2_tp = reshard_experts_ep_to_tp_local(
        w13_all, w2_all, s13_all, s2_all, tp_rank=tp_rank, g=g, ex=ex
    )
    del w13_all, w2_all, s13_all, s2_all
    del w13_ep, w2_ep, s13_ep, s2_ep

    experts.w13_weight = torch.nn.Parameter(w13_tp, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2_tp, requires_grad=False)
    s13_new_param = torch.nn.Parameter(s13_tp, requires_grad=False)
    s2_new_param = torch.nn.Parameter(s2_tp, requires_grad=False)
    setattr(experts, s13_attr, s13_new_param)
    setattr(experts, s2_attr, s2_new_param)

    if w13_tp.device.type == "tpu":
        from torch_tpu._internal import sync

        sync.synchronize(
            tensors=[
                experts.w13_weight,
                experts.w2_weight,
                s13_new_param,
                s2_new_param,
            ],
            wait=True,
        )

    apply_tp_moe_flags(mlp.experts, prebuild_moe_fn=prebuild_moe_fn)


def pack_weights(
    backbone: Any,
    g: Any,
    ex: ExecSpec,
    *,
    tp_rank: int = 0,
    expert_placement: Any = None,
    tp_all_gather_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Pack all layers of `backbone` into the megakernel's weight dictionary."""
    mdef = getattr(g, "_model_def", None)
    if mdef is None:
        from vllm_torchtpu.megakernel.models.qwen35 import Qwen35

        mdef = Qwen35
    return load_vllm_weights(
        mdef,
        backbone,
        ex,
        g=g,
        tp_rank=tp_rank,
        expert_placement=expert_placement,
        tp_all_gather_fn=tp_all_gather_fn,
    )


__all__ = [
    "pack_weights",
    "pack_router_and_shared_layer",
    "_gather_replicated_sh13",
    "_gather_replicated_sh2",
    "TorchArrayOps",
    "VLLMWeightSource",
    "WEIGHT_FLIPPED_ATTR",
    "apply_tp_moe_flags",
    "extract_ecopy_layer",
    "find_backbone",
    "get_routed_experts",
    "load_vllm_weights",
    "reshard_experts_ep_to_tp_local",
    "reshard_layer_experts_ep_to_tp",
]
