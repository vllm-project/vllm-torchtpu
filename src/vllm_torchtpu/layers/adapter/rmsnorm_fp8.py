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
"""Torch adapter fusing pre-MLP RMSNorm + dynamic FP8 quant ahead of gate_up_proj."""

import ast
import functools
import inspect
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax.experimental.pallas.tpu as pltpu
import torch
import torch.nn as nn
from torch_tpu._internal import pallas
from vllm import envs as vllm_envs
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)

from vllm_torchtpu.kernels.rmsnorm_fp8 import (
    FP8_RMSNORM_MIN_TOKENS,
    fused_rmsnorm_fp8_matmul,
    select_block_m,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tpu_info import get_chip_version

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

# Speculative drafter / MTP modules are traced once across token counts rather
# than per backbone compile_sizes bucket; keep them on the stock unfused path.
_DRAFTER_PREFIXES = ("draft_model.", "mtp.", "eagle_head.")


def _fake_fused_rmsnorm_fp8_matmul(
    x: torch.Tensor,
    gamma: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
    w8a16_max_rows: int = 0,
    vmem_capacity_bytes: int = 0,
) -> torch.Tensor:
    del gamma, w_s, eps, gamma_offset, w8a16_max_rows, vmem_capacity_bytes
    return torch.empty(x.shape[0], w_q.shape[-1], dtype=x.dtype, device=x.device)


# Registered at import time rather than lazily behind a Lock because Dynamo
# traces through decoder-layer forwards under fullgraph compilation.
_rmsnorm_fp8_mm_op = pallas.jax_op(
    "pallas::fused_rmsnorm_fp8_matmul", fused_rmsnorm_fp8_matmul
)
_rmsnorm_fp8_mm_op.register_fake(_fake_fused_rmsnorm_fp8_matmul)


def _resolve_gamma_offset(norm: nn.Module) -> float | None:
    if type(norm) is RMSNorm and isinstance(norm, RMSNorm):
        if not norm.has_weight or norm.variance_size_override is not None:
            return None
        return 0.0
    if type(norm) is GemmaRMSNorm and isinstance(norm, GemmaRMSNorm):
        return 1.0
    return None


def can_fuse_rmsnorm_fp8_linear(
    x: torch.Tensor,
    norm: nn.Module,
    linear: nn.Module,
    vmem_capacity_bytes: int,
) -> bool:
    """True when ``norm -> linear`` can safely dispatch to ``_rmsnorm_fp8_mm_op``.

    Scoped to exact ``MergedColumnParallelLinear`` instances (gated-MLP
    ``gate_up_proj``) whose weights have been transposed to ``[K, N]`` FP8
    runtime layout. Only static layer/weight properties and hidden size ``K``
    are checked here so TorchDynamo traces a single FX graph across all token
    counts; per-bucket ``M`` and ``w8a16_max_rows`` gating runs inside JAX
    lowering.
    """
    if vllm_envs.VLLM_BATCH_INVARIANT or _resolve_gamma_offset(norm) is None:
        return False
    assert isinstance(norm, (RMSNorm, GemmaRMSNorm))
    if (
        type(linear) is not MergedColumnParallelLinear
        or not isinstance(linear, MergedColumnParallelLinear)
        or MergedColumnParallelLinear.forward is not ColumnParallelLinear.forward
        or linear.prefix.startswith(_DRAFTER_PREFIXES)
        or linear.gather_output
        or getattr(linear, KEEP_VLLM_LAYOUT_ATTR, False)
        or not getattr(linear, WEIGHT_FLIPPED_ATTR, False)
        or not isinstance(linear.quant_method, VllmFp8LinearMethodTPU)
        or x.device.type != "tpu"
        or x.dtype != torch.bfloat16
        or x.ndim < 2
    ):
        return False

    # Load-time FP8 quantization registers `weight_scale` only during
    # `process_weights_after_loading`, so it may be absent before post-load.
    weight_scale = getattr(linear, "weight_scale", None)
    if not isinstance(weight_scale, torch.Tensor):
        return False

    norm_weight = norm.weight
    weight = linear.weight
    k = x.shape[-1]
    if (
        not isinstance(k, int)
        or k < 128
        or k % 128 != 0
        or norm_weight.ndim != 1
        or norm_weight.shape[0] != k
    ):
        return False

    if (
        weight.ndim != 2
        or weight.shape[0] != k
        or weight.shape[1] <= 0
        or weight.dtype != torch.float8_e4m3fn
        or weight_scale.ndim != 1
        or weight_scale.shape[0] not in (1, weight.shape[1])
    ):
        return False

    return select_block_m(FP8_RMSNORM_MIN_TOKENS, k, vmem_capacity_bytes) is not None


def _finish_column_linear_output(
    out_2d: torch.Tensor,
    orig_shape: torch.Size,
    linear: MergedColumnParallelLinear,
) -> Any:
    out = (
        out_2d
        if len(orig_shape) == 2
        else out_2d.reshape(*orig_shape[:-1], out_2d.shape[-1])
    )
    bias = linear.bias if not linear.skip_bias_add else None
    if bias is not None:
        out = out + bias
    if not linear.return_bias:
        return out
    output_bias = linear.bias if linear.skip_bias_add else None
    return out, output_bias


def fused_rmsnorm_fp8_linear(
    x: torch.Tensor,
    norm: RMSNorm | GemmaRMSNorm,
    linear: MergedColumnParallelLinear,
    vmem_capacity_bytes: int,
) -> Any:
    """Executes ``linear(norm(x))`` via ``pallas::fused_rmsnorm_fp8_matmul``."""
    gamma_offset = _resolve_gamma_offset(norm)
    assert gamma_offset is not None
    quant_method = linear.quant_method
    assert isinstance(quant_method, VllmFp8LinearMethodTPU)
    k = x.shape[-1]
    x_2d = x if x.ndim == 2 else x.reshape(-1, k)
    out_2d = _rmsnorm_fp8_mm_op(
        x_2d,
        norm.weight,
        linear.weight,
        linear.weight_scale,
        float(norm.variance_epsilon),
        gamma_offset,
        int(quant_method.w8a16_max_rows),
        vmem_capacity_bytes,
    )
    return _finish_column_linear_output(out_2d, x.shape, linear)


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _count_name_loads(nodes: list[ast.stmt] | ast.AST, var_name: str) -> int:
    roots = nodes if isinstance(nodes, list) else [nodes]
    total = 0
    for root in roots:
        for node in ast.walk(root):
            if (
                isinstance(node, ast.Name)
                and node.id == var_name
                and isinstance(node.ctx, ast.Load)
            ):
                total += 1
    return total


def _match_self_call(expr: ast.expr, self_name: str) -> tuple[str, ast.Call] | None:
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Attribute):
        return None
    if not isinstance(expr.func.value, ast.Name) or expr.func.value.id != self_name:
        return None
    return expr.func.attr, expr


def _match_norm_stmt(stmt: ast.stmt, self_name: str) -> tuple[str, str] | None:
    """Matches 1-arg or 2-arg norm assignment and returns ``(out_var, norm_attr)``."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    tgt = stmt.targets[0]
    m = _match_self_call(stmt.value, self_name)
    if m is None:
        return None
    norm_attr, call = m
    if call.keywords:
        return None
    # 1-arg: `h = self.<norm_attr>(x)`
    if (
        isinstance(tgt, ast.Name)
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
    ):
        return tgt.id, norm_attr
    # 2-arg: `h, res = self.<norm_attr>(h, res)`
    if (
        isinstance(tgt, ast.Tuple)
        and len(tgt.elts) == 2
        and all(isinstance(e, ast.Name) for e in tgt.elts)
        and len(call.args) == 2
        and all(isinstance(a, ast.Name) for a in call.args)
    ):
        out_el, res_el = tgt.elts
        assert isinstance(out_el, ast.Name) and isinstance(res_el, ast.Name)
        if out_el.id != res_el.id:
            return out_el.id, norm_attr
    return None


def _match_consumer_stmt(
    stmt: ast.stmt, self_name: str, norm_var: str
) -> tuple[str, int | str] | None:
    """Matches ``norm_var = self.<submod_attr>(...)`` consuming ``norm_var`` once."""
    if (
        not isinstance(stmt, ast.Assign)
        or len(stmt.targets) != 1
        or not isinstance(stmt.targets[0], ast.Name)
        or stmt.targets[0].id != norm_var
    ):
        return None
    m = _match_self_call(stmt.value, self_name)
    if m is None:
        return None
    submod_attr, call = m
    if _count_name_loads(call, norm_var) != 1:
        return None

    for idx, arg in enumerate(call.args):
        if isinstance(arg, ast.Name) and arg.id == norm_var:
            return submod_attr, idx
    for kw in call.keywords:
        if (
            kw.arg is not None
            and isinstance(kw.value, ast.Name)
            and kw.value.id == norm_var
        ):
            return submod_attr, kw.arg
    return None


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _find_fusible_norm_submod_pairs(
    fn: Callable,
) -> list[tuple[str, str, int | str]]:
    """Finds provably single-use ``(norm_attr, submod_attr, arg_spec)`` pairs."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return []

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return []
    fdef = tree.body[0]
    if not fdef.args.args:
        return []
    self_name = fdef.args.args[0].arg
    body = _strip_docstring(fdef.body)

    pairs: list[tuple[str, str, int | str]] = []
    i = 0
    while i + 1 < len(body):
        nm = _match_norm_stmt(body[i], self_name)
        if nm is not None:
            norm_var, norm_attr = nm
            cm = _match_consumer_stmt(body[i + 1], self_name, norm_var)
            if cm is not None:
                submod_attr, arg_spec = cm
                pairs.append((norm_attr, submod_attr, arg_spec))
                i += 2
                continue
        i += 1
    return pairs


def _match_unpack_proj_call(
    stmt: ast.stmt, self_name: str, in_param: str
) -> tuple[str, str] | None:
    """Matches ``lhs, _ = self.<proj>(in_param)`` -> ``(proj, lhs)``."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    tgt = stmt.targets[0]
    if not isinstance(tgt, ast.Tuple) or len(tgt.elts) != 2:
        return None
    lhs, ignored = tgt.elts
    if (
        not isinstance(lhs, ast.Name)
        or not isinstance(ignored, ast.Name)
        or ignored.id != "_"
    ):
        return None
    m = _match_self_call(stmt.value, self_name)
    if m is None:
        return None
    proj_attr, call = m
    if (
        len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Name)
        or call.args[0].id != in_param
    ):
        return None
    return proj_attr, lhs.id


def _unconditionally_rebinds(stmt: ast.stmt, var_name: str) -> bool:
    """True when ``stmt`` overwrites ``var_name`` without reading ``var_name``."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        tgt = stmt.targets[0]
        if _count_name_loads(stmt.value, var_name) != 0 or (
            _count_name_loads(tgt, var_name) != 0
        ):
            return False
        if isinstance(tgt, ast.Name) and tgt.id == var_name:
            return True
        if isinstance(tgt, ast.Tuple):
            return any(isinstance(e, ast.Name) and e.id == var_name for e in tgt.elts)
    if (
        isinstance(stmt, ast.AnnAssign)
        and stmt.value is not None
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id == var_name
    ):
        return (
            _count_name_loads(stmt.value, var_name) == 0
            and _count_name_loads(stmt.annotation, var_name) == 0
        )
    return False


def _has_live_load_before_rebind(stmts: list[ast.stmt], var_name: str) -> bool:
    """True if ``var_name`` can be read before an unconditional rebinding."""
    for stmt in stmts:
        if _unconditionally_rebinds(stmt, var_name):
            return False
        if _count_name_loads(stmt, var_name) != 0:
            return True
    return False


def _entry_projection_attr(fn: Callable, arg_spec: int | str) -> str | None:
    """Verifies ``submod.forward`` passes its input exclusively to ``self.<proj>``."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return None

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fdef = tree.body[0]
    non_self_args = fdef.args.args[1:]
    if isinstance(arg_spec, int):
        if arg_spec >= len(non_self_args):
            return None
        in_param = non_self_args[arg_spec].arg
    else:
        all_named = [*non_self_args, *fdef.args.kwonlyargs]
        matched = [a.arg for a in all_named if a.arg == arg_spec]
        if len(matched) != 1:
            return None
        in_param = matched[0]

    self_name = fdef.args.args[0].arg
    body = _strip_docstring(fdef.body)
    if not body:
        return None

    up = _match_unpack_proj_call(body[0], self_name, in_param)
    if up is None:
        return None
    proj_attr, lhs = up
    if lhs != in_param and _has_live_load_before_rebind(body[1:], in_param):
        return None
    return proj_attr


def _bind_norm_and_proj(
    norm: RMSNorm | GemmaRMSNorm,
    proj: MergedColumnParallelLinear,
    vmem_capacity_bytes: int,
) -> bool:
    if (
        vllm_envs.VLLM_BATCH_INVARIANT
        or getattr(norm, "_tpu_rmsnorm_fp8_bound", False)
        or getattr(proj, "_tpu_rmsnorm_fp8_bound", False)
    ):
        return False
    if _resolve_gamma_offset(norm) is None:
        return False
    if (
        type(proj) is not MergedColumnParallelLinear
        or MergedColumnParallelLinear.forward is not ColumnParallelLinear.forward
        or "forward" in proj.__dict__
        or proj.prefix.startswith(_DRAFTER_PREFIXES)
        or proj.gather_output
        or not isinstance(proj.quant_method, VllmFp8LinearMethodTPU)
    ):
        return False

    orig_norm_fwd = norm.forward
    orig_proj_fwd = proj.forward

    def _deferred_norm_forward(
        x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return x
        # Match ir_ops.fused_add_rms_norm: round the residual sum to x.dtype
        # once; _fused_proj_forward normalizes that rounded sum.
        new_res = (x.to(torch.float32) + residual.to(torch.float32)).to(x.dtype)
        return new_res, new_res

    def _fused_proj_forward(x: torch.Tensor) -> Any:
        if can_fuse_rmsnorm_fp8_linear(x, norm, proj, vmem_capacity_bytes):
            return fused_rmsnorm_fp8_linear(x, norm, proj, vmem_capacity_bytes)
        if not torch.compiler.is_compiling():
            logger.info_once(
                "Falling back to unfused pre-MLP RMSNorm+Linear on %s "
                "(device=%s, dtype=%s, shape=%s)",
                proj.prefix or type(proj).__name__,
                x.device.type,
                x.dtype,
                tuple(x.shape),
            )
        return orig_proj_fwd(orig_norm_fwd(x))

    norm.forward = _deferred_norm_forward  # type: ignore[method-assign]
    proj.forward = _fused_proj_forward  # type: ignore[method-assign]
    norm._tpu_rmsnorm_fp8_bound = True  # type: ignore[attr-defined]
    proj._tpu_rmsnorm_fp8_bound = True  # type: ignore[attr-defined]
    logger.info_once(
        "Bound TPU fused Pallas RMSNorm+FP8 pre-MLP projection (first seen on %s)",
        proj.prefix or type(proj).__name__,
    )
    return True


def _make_patched_decoder_init(
    orig_init: Callable,
    pairs: list[tuple[str, str, int | str]],
    vmem_capacity_bytes: int,
) -> Callable:
    @functools.wraps(orig_init)
    def _patched_init(self: nn.Module, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        for norm_attr, submod_attr, arg_spec in pairs:
            norm = getattr(self, norm_attr, None)
            submod = getattr(self, submod_attr, None)
            if not isinstance(norm, (RMSNorm, GemmaRMSNorm)) or not isinstance(
                submod, nn.Module
            ):
                continue
            submod_fwd = type(submod).__dict__.get("forward")
            if submod_fwd is None:
                continue
            proj_attr = _entry_projection_attr(submod_fwd, arg_spec)
            if proj_attr is None:
                continue
            proj = getattr(submod, proj_attr, None)
            if type(proj) is MergedColumnParallelLinear and isinstance(
                proj, MergedColumnParallelLinear
            ):
                _bind_norm_and_proj(norm, proj, vmem_capacity_bytes)

    _patched_init._tpu_fused_rmsnorm_fp8 = True  # type: ignore[attr-defined]
    return _patched_init


def maybe_patch_rmsnorm_fp8(model_config: "ModelConfig | None" = None) -> bool | None:
    """Discovers and binds pre-MLP ``RMSNorm -> MergedColumnParallelLinear`` on v7x."""
    if model_config is None:
        return False
    if vllm_envs.VLLM_BATCH_INVARIANT or get_chip_version() != pltpu.ChipVersion.TPU_7X:
        return None

    # Query static chip VMEM once outside Dynamo tracing so the compiled forward
    # path sees a plain integer constant rather than calling into tpu_info.
    vmem_capacity_bytes = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    from vllm.model_executor.models.registry import ModelRegistry

    supported_archs = ModelRegistry.get_supported_archs()
    archs = [a for a in (model_config.architectures or ()) if a in supported_archs]
    if archs:
        ModelRegistry.resolve_model_cls(archs, model_config=model_config)

    patched_names: list[str] = []
    for cls in sorted(_all_subclasses(nn.Module), key=lambda c: c.__qualname__):
        if not cls.__module__.startswith("vllm.model_executor.models."):
            continue
        fwd = cls.__dict__.get("forward")
        init_fn = cls.__dict__.get("__init__")
        if (
            fwd is None
            or init_fn is None
            or getattr(init_fn, "_tpu_fused_rmsnorm_fp8", False)
        ):
            continue
        pairs = _find_fusible_norm_submod_pairs(fwd)
        if pairs:
            cls.__init__ = _make_patched_decoder_init(  # type: ignore[method-assign]
                init_fn, pairs, vmem_capacity_bytes
            )
            patched_names.append(f"{cls.__module__}.{cls.__qualname__}")

    if patched_names:
        logger.info(
            "Applied TPU patch: pre-MLP RMSNorm+FP8 decoder-init hook (%s)",
            ", ".join(patched_names),
        )
    return None
