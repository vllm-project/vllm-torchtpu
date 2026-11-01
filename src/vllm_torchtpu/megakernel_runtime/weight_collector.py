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

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

import jax.numpy as jnp
import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.decls import HFParamDecl, VLLMAlias
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, I32
from vllm_torchtpu.megakernel.weights import (
    ArrayOps,
    load_weights,
    resolve_spec_and_norms,
    slice_alias_targets,
    target_axis_widths,
)
from vllm_torchtpu.megakernel_runtime.backbone import find_backbone

WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"

_TORCH_DTYPE_MAP: dict[Any, torch.dtype] = {
    jnp.dtype(FP8): torch.float8_e4m3fn,
    jnp.dtype(BF16): torch.bfloat16,
    jnp.dtype(F32): torch.float32,
    jnp.dtype(I32): torch.int32,
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


def reshard_experts_ep_to_tp_local(
    w13_all: torch.Tensor,
    w2_all: torch.Tensor,
    s13_all: torch.Tensor | None,
    s2_all: torch.Tensor | None,
    tp_rank: int,
    g: Any,
    ex: ExecSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Slice global `E`-expert weights/scales into `tp_rank`'s TP-sharded tensors
    (scales `None` for bf16 experts)."""
    ir = g.I // ex.R
    g0 = tp_rank * ir
    u0 = g.I + tp_rank * ir

    w13_tp = _own(
        torch.cat([w13_all[:, :, g0 : g0 + ir], w13_all[:, :, u0 : u0 + ir]], dim=-1)
    )
    w2_tp = _own(w2_all[:, g0 : g0 + ir, :])
    if s13_all is None or s2_all is None:
        return w13_tp, w2_tp, None, None

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


def _expert_scale_attr(experts: Any, weight: str) -> str:
    for attr in (f"{weight}_scale_inv", f"{weight}_scale"):
        if hasattr(experts, attr):
            return attr
    raise ValueError(f"FP8 experts are missing {weight}_scale (or _scale_inv)")


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

    # bf16 (unquantized) experts carry no scales; fp8 experts must.
    fp8 = w13_ep.dtype == torch.float8_e4m3fn
    _need(
        fp8 == (w2_ep.dtype == torch.float8_e4m3fn),
        f"w13/w2 expert dtypes differ ({w13_ep.dtype} vs {w2_ep.dtype})",
    )
    s13_attr = s2_attr = ""
    s13_ep = s2_ep = None
    if fp8:
        s13_attr = _expert_scale_attr(experts, "w13_weight")
        s2_attr = _expert_scale_attr(experts, "w2_weight")
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
        s13_all = all_gather_fn(s13_ep) if fp8 else None
        s2_all = all_gather_fn(s2_ep) if fp8 else None

    w13_tp, w2_tp, s13_tp, s2_tp = reshard_experts_ep_to_tp_local(
        w13_all, w2_all, s13_all, s2_all, tp_rank=tp_rank, g=g, ex=ex
    )
    del w13_all, w2_all, s13_all, s2_all
    del w13_ep, w2_ep, s13_ep, s2_ep

    experts.w13_weight = torch.nn.Parameter(w13_tp, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2_tp, requires_grad=False)
    synced = [experts.w13_weight, experts.w2_weight]
    if s13_tp is not None and s2_tp is not None:
        s13_new_param = torch.nn.Parameter(s13_tp, requires_grad=False)
        s2_new_param = torch.nn.Parameter(s2_tp, requires_grad=False)
        setattr(experts, s13_attr, s13_new_param)
        setattr(experts, s2_attr, s2_new_param)
        synced += [s13_new_param, s2_new_param]

    if w13_tp.device.type == "tpu":
        from torch_tpu._internal import sync

        sync.synchronize(tensors=synced, wait=True)

    apply_tp_moe_flags(mlp.experts, prebuild_moe_fn=prebuild_moe_fn)


__all__ = [
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
