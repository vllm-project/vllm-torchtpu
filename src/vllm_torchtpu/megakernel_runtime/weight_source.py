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
"""Generic vLLM weight source (`TorchArrayOps`, `VLLMWeightSource`,
`load_vllm_weights`, the backbone lookup `find_backbone`).

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

No `megakernel` library imports at module level: the library's weight
packing (`megakernel.weights`) and the MoE expert-copy helper
(`weight_collector.extract_ecopy_layer`) are imported when a weight load
runs. The MoE / library-geometry helpers are in `weight_collector`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel_runtime.backbone import find_backbone

if TYPE_CHECKING:
    from vllm_torchtpu.megakernel.decls import HFParamDecl, VLLMAlias
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.weights import ArrayOps


WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"


_TORCH_DTYPE_MAP: dict[Any, torch.dtype] = {
    jnp.dtype(jnp.float8_e4m3fn): torch.float8_e4m3fn,
    jnp.dtype(jnp.bfloat16): torch.bfloat16,
    jnp.dtype(jnp.float32): torch.float32,
    jnp.dtype(jnp.int32): torch.int32,
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


class VLLMWeightSource:
    """`WeightSource` backed by a live vLLM decoder `backbone` (`layers` + `norm`)."""

    def __init__(self, backbone: Any, g: Any) -> None:
        _need(
            len(backbone.layers) == int(g.L),
            f"expected {g.L} layers, got {len(backbone.layers)}",
        )
        self._backbone = backbone
        self._dev = backbone.norm.weight.device
        self._ops = TorchArrayOps(device=self._dev)

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
        from vllm_torchtpu.megakernel.weights import (
            slice_alias_targets,
            target_axis_widths,
        )

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
            # bf16 (unquantized) experts carry no scale; fp8 experts must.
            if self._ops.is_fp8(w):
                s_raw = self._read_scale(layer, proj_obj, alias)
                _need(
                    s_raw is not None,
                    f"FP8 expert weight {alias.source!r} is missing its scale "
                    "(weight_scale / weight_scale_inv)",
                )
                assert s_raw is not None
                exp_e, exp_n = int(w.shape[0]), int(w.shape[-1])
                _need(
                    s_raw.numel() == exp_e * exp_n,
                    f"Expert scale for {alias.source!r} has shape "
                    f"{tuple(s_raw.shape)} ({s_raw.numel()} elems), "
                    f"expected ({exp_e}, {exp_n})",
                )
                s_buf = s_raw.float().reshape(exp_e, exp_n).contiguous()
            w_buf = w
        elif len(target_decls) == 1 and len(d0.shape) == 1:
            w_buf = w.reshape(-1)
        elif w.ndim == 2 and not d0.transform.flatten_tail:
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
        from .weight_collector import extract_ecopy_layer

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
) -> dict[str, Any]:
    """Extract and pack all layers of `backbone` via `load_weights`."""
    from vllm_torchtpu.megakernel.weights import load_weights, resolve_spec_and_norms

    _, geom, _, _, _ = resolve_spec_and_norms(model_def, ex, g=g)
    source = VLLMWeightSource(backbone, geom)
    return load_weights(
        model_def,
        ex,
        source,
        g=geom,
        tp_rank=tp_rank,
        expert_placement=expert_placement,
    )


__all__ = [
    "WEIGHT_FLIPPED_ATTR",
    "TorchArrayOps",
    "VLLMWeightSource",
    "find_backbone",
    "load_vllm_weights",
]
