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
"""Torch adapter for the fused Pallas gated-MLP activation + dynamic FP8 quant op."""

import ast
import functools
import importlib
import inspect
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import jax.experimental.pallas.tpu as pltpu
import torch
import torch.nn as nn
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gated_mlp_fp8 import (
    GatedActivationKind,
    fused_gated_act_fp8_matmul,
)
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tpu_info import get_chip_version

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


class _GatedMLPTarget(NamedTuple):
    architectures: frozenset[str]
    module: str
    class_name: str


class _FusedDownProj(NamedTuple):
    act_kind: int
    weight: torch.Tensor
    weight_scale: torch.Tensor
    w8a16_max_rows: int
    tp_size: int
    tp_rank: int
    reduce_results: bool
    bias: torch.Tensor | None


# Restricted to Gemma4MLP where end-to-end serving, GSM8K accuracy, and greedy
# parity are validated on TPU v7x.
_ALLOWED_GATED_MLP_TARGETS: tuple[_GatedMLPTarget, ...] = (
    _GatedMLPTarget(
        architectures=frozenset(
            {"Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"}
        ),
        module="vllm.model_executor.models.gemma4",
        class_name="Gemma4MLP",
    ),
)


def _fake_fused_gated_act_fp8_matmul(
    gate_up: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    act_kind: int = int(GatedActivationKind.GELU_TANH),
    w8a16_max_rows: int = 0,
    vmem_capacity_bytes: int = 0,
) -> torch.Tensor:
    del w_s, act_kind, w8a16_max_rows, vmem_capacity_bytes
    return torch.empty(
        gate_up.shape[0],
        w_q.shape[-1],
        dtype=gate_up.dtype,
        device=gate_up.device,
    )


# Registered at import time rather than lazily behind a Lock because Dynamo
# traces through MLP forward under fullgraph compilation (same pattern as router_topk).
_fp8_op = pallas.jax_op(
    "pallas::fused_gated_act_fp8_matmul", fused_gated_act_fp8_matmul
)
_fp8_op.register_fake(_fake_fused_gated_act_fp8_matmul)


_ACT_GELU_TANH = int(GatedActivationKind.GELU_TANH)
_ACT_SILU = int(GatedActivationKind.SILU)


def _resolve_act_kind(act_fn: nn.Module) -> int | None:
    if type(act_fn) is GeluAndMul:
        if act_fn.approximate == "tanh":
            return _ACT_GELU_TANH
        # Exact GeLU lowers to erfc, which Pallas TPU lowering does not implement.
        return None
    if type(act_fn) is SiluAndMul:
        return _ACT_SILU
    return None


def _is_standard_unpack_call(
    stmt: ast.stmt, self_name: str, attr_name: str, arg_name: str
) -> str | None:
    """Matches ``lhs, _ = self.<attr_name>(<arg_name>)`` and returns ``lhs``."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Tuple) or len(target.elts) != 2:
        return None
    lhs, ignored = target.elts
    if not isinstance(lhs, ast.Name) or not isinstance(ignored, ast.Name):
        return None
    if ignored.id != "_":
        return None
    call = stmt.value
    if (
        not isinstance(call, ast.Call)
        or len(call.args) != 1
        or call.keywords
        or not isinstance(call.args[0], ast.Name)
        or call.args[0].id != arg_name
    ):
        return None
    func = call.func
    if (
        not isinstance(func, ast.Attribute)
        or func.attr != attr_name
        or not isinstance(func.value, ast.Name)
        or func.value.id != self_name
    ):
        return None
    return lhs.id


def _matches_standard_gated_mlp_forward(fn: Callable) -> bool:
    """True iff ``fn`` is strictly ``gate_up, _ = self.gate_up_proj(x);
    x = self.act_fn(gate_up); x, _ = self.down_proj(x); return x``."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return False

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    fdef = tree.body[0]
    if len(fdef.args.args) != 2 or fdef.args.vararg or fdef.args.kwarg:
        return False
    self_name = fdef.args.args[0].arg
    in_name = fdef.args.args[1].arg

    body = fdef.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 4:
        return False

    v_gate_up = _is_standard_unpack_call(body[0], self_name, "gate_up_proj", in_name)
    if v_gate_up is None:
        return False

    stmt1 = body[1]
    if (
        not isinstance(stmt1, ast.Assign)
        or len(stmt1.targets) != 1
        or not isinstance(stmt1.targets[0], ast.Name)
    ):
        return False
    v_act = stmt1.targets[0].id
    call1 = stmt1.value
    if (
        not isinstance(call1, ast.Call)
        or len(call1.args) != 1
        or call1.keywords
        or not isinstance(call1.args[0], ast.Name)
        or call1.args[0].id != v_gate_up
        or not isinstance(call1.func, ast.Attribute)
        or call1.func.attr != "act_fn"
        or not isinstance(call1.func.value, ast.Name)
        or call1.func.value.id != self_name
    ):
        return False

    v_out = _is_standard_unpack_call(body[2], self_name, "down_proj", v_act)
    if v_out is None:
        return False

    stmt3 = body[3]
    return (
        isinstance(stmt3, ast.Return)
        and isinstance(stmt3.value, ast.Name)
        and stmt3.value.id == v_out
    )


# Upstream Gemma4MTP (vllm/model_executor/models/gemma4_mtp.py) constructs
# Gemma4MultiTokenPredictor(..., prefix=maybe_prefix(prefix, "draft_model")),
# giving drafter MLPs prefix "draft_model.layers.{i}.mlp.down_proj".
_DRAFTER_PREFIX = "draft_model."
_LOGGED_FALLBACK_REASONS: set[str] = set()


def _log_fallback_once(down_proj: object, reason: str) -> None:
    # Deduplicate by reason so a 60-layer model logs once (naming the first seen
    # layer) rather than 60 times; skip under TorchDynamo symbolic tracing, which
    # rejects custom Logger methods under fullgraph=True (see moe_routing).
    if not torch.compiler.is_compiling() and reason not in _LOGGED_FALLBACK_REASONS:
        _LOGGED_FALLBACK_REASONS.add(reason)
        prefix = getattr(down_proj, "prefix", "") or type(down_proj).__name__
        logger.info_once(
            "Skipping TPU fused gated-MLP FP8 (first seen on %s): %s",
            prefix,
            reason,
        )


def _resolve_fused_down_proj(
    act_fn: nn.Module, down_proj: nn.Module
) -> _FusedDownProj | str:
    """Validate static layer/weight properties once and return resolved params."""
    act_kind = _resolve_act_kind(act_fn)
    if act_kind is None:
        return f"unsupported activation {type(act_fn).__name__}"
    if not isinstance(down_proj, RowParallelLinear):
        return f"down_proj is {type(down_proj).__name__}, expected RowParallelLinear"
    prefix = getattr(down_proj, "prefix", "")
    if prefix.startswith(_DRAFTER_PREFIX):
        return f"drafter prefix ({prefix}) excluded"
    quant_method = down_proj.quant_method
    if not isinstance(quant_method, VllmFp8LinearMethodTPU):
        return f"unsupported quant_method {type(quant_method).__name__}"
    weight = getattr(down_proj, "weight", None)
    weight_scale = getattr(down_proj, "weight_scale", None)
    if (
        weight is None
        or weight_scale is None
        or not isinstance(weight, torch.Tensor)
        or not isinstance(weight_scale, torch.Tensor)
    ):
        return "missing down_proj weight or weight_scale"
    if not down_proj.input_is_parallel:
        return "down_proj.input_is_parallel is False"
    if down_proj.return_bias is False:
        return "down_proj.return_bias is False"
    if down_proj.skip_bias_add:
        return "down_proj.skip_bias_add is True"
    if getattr(down_proj, KEEP_VLLM_LAYOUT_ATTR, False):
        return f"down_proj has {KEEP_VLLM_LAYOUT_ATTR}=True"
    if weight.ndim != 2 or weight.shape[0] < 128 or weight.shape[0] % 128 != 0:
        return f"unsupported down_proj weight shape {tuple(weight.shape)}"
    if weight.dtype != torch.float8_e4m3fn:
        return f"unsupported down_proj weight dtype {weight.dtype}"
    if weight_scale.ndim != 1 or weight_scale.shape[0] not in (1, weight.shape[1]):
        return (
            "non-channel/tensor down_proj weight_scale shape "
            f"{tuple(weight_scale.shape)}"
        )
    return _FusedDownProj(
        act_kind=act_kind,
        weight=weight,
        weight_scale=weight_scale,
        w8a16_max_rows=int(quant_method.w8a16_max_rows),
        tp_size=int(down_proj.tp_size),
        tp_rank=int(down_proj.tp_rank),
        reduce_results=bool(down_proj.reduce_results),
        bias=getattr(down_proj, "bias", None),
    )


def _run_unfused_act_down(
    gate_up: torch.Tensor, act_fn: nn.Module, down_proj: nn.Module
) -> torch.Tensor:
    act_out = act_fn(gate_up)
    res = down_proj(act_out)
    return res[0] if isinstance(res, tuple) else res


def _run_fused_or_fallback(
    gate_up: torch.Tensor,
    act_fn: nn.Module,
    down_proj: nn.Module,
    spec: _FusedDownProj,
    vmem_capacity_bytes: int,
) -> torch.Tensor:
    if (
        gate_up.device.type != "tpu"
        or gate_up.dtype != torch.bfloat16
        or gate_up.shape[-1] != 2 * spec.weight.shape[0]
    ):
        _log_fallback_once(
            down_proj,
            f"incompatible gate_up (device={gate_up.device.type}, "
            f"dtype={gate_up.dtype}, shape={tuple(gate_up.shape)})",
        )
        return _run_unfused_act_down(gate_up, act_fn, down_proj)

    orig_out_shape = (*gate_up.shape[:-1], spec.weight.shape[1])
    gate_up_2d = gate_up.reshape(-1, gate_up.shape[-1])
    out_2d = _fp8_op(
        gate_up_2d,
        spec.weight,
        spec.weight_scale,
        spec.act_kind,
        spec.w8a16_max_rows,
        vmem_capacity_bytes,
    )
    out = out_2d.reshape(orig_out_shape)
    if spec.reduce_results and spec.tp_size > 1:
        out = tensor_model_parallel_all_reduce(out)
    # Always add bias after the TP all-reduce when reduce_results is True so
    # per-rank FX graphs stay identical (matching TpuRowParallelLinear).
    if spec.bias is not None and (spec.reduce_results or spec.tp_rank == 0):
        out = out + spec.bias
    return out


def apply_gated_mlp_act_down(
    gate_up: torch.Tensor,
    act_fn: nn.Module,
    down_proj: nn.Module,
    vmem_capacity_bytes: int,
) -> torch.Tensor:
    """Runs ``down_proj(act_fn(gate_up))``, dispatching to ``_fp8_op`` when eligible.

    Only static layer/weight properties are inspected here so the Dynamo FX
    graph stays completely independent of ``num_tokens`` across piecewise
    subgraphs; token-bucket selection (``m >= FP8_MIN_TOKENS``,
    ``w8a16_max_rows``, ``select_block_m``) happens inside JAX lowering per
    compiled bucket, matching ``linear_common.quantized_matmul``.
    """
    spec_or_reason = _resolve_fused_down_proj(act_fn, down_proj)
    if isinstance(spec_or_reason, str):
        _log_fallback_once(down_proj, spec_or_reason)
        return _run_unfused_act_down(gate_up, act_fn, down_proj)
    return _run_fused_or_fallback(
        gate_up, act_fn, down_proj, spec_or_reason, vmem_capacity_bytes
    )


def _make_fused_mlp_forward(
    orig_forward: Callable, vmem_capacity_bytes: int
) -> Callable:
    @functools.wraps(orig_forward)
    def _fused_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate_up_proj = self.gate_up_proj
        act_fn = self.act_fn
        down_proj = self.down_proj
        if (
            not isinstance(gate_up_proj, MergedColumnParallelLinear)
            or type(gate_up_proj) is not MergedColumnParallelLinear
        ):
            _log_fallback_once(
                down_proj,
                f"gate_up_proj is {type(gate_up_proj).__name__}, "
                "expected MergedColumnParallelLinear",
            )
            return orig_forward(self, x)
        if not isinstance(act_fn, nn.Module) or not isinstance(down_proj, nn.Module):
            _log_fallback_once(down_proj, "act_fn or down_proj is not an nn.Module")
            return orig_forward(self, x)

        spec_or_reason = _resolve_fused_down_proj(act_fn, down_proj)
        if isinstance(spec_or_reason, str):
            _log_fallback_once(down_proj, spec_or_reason)
            return orig_forward(self, x)

        gate_up, _ = gate_up_proj(x)
        return _run_fused_or_fallback(
            gate_up, act_fn, down_proj, spec_or_reason, vmem_capacity_bytes
        )

    _fused_forward._tpu_fused_gated_mlp = True  # type: ignore[attr-defined]
    return _fused_forward


def maybe_patch_gated_mlp(model_config: "ModelConfig | None" = None) -> bool | None:
    """Patches allowlisted dense gated-MLP classes (``Gemma4MLP``) on TPU v7x."""
    if model_config is None:
        return False
    if (
        not envs.TPU_FUSED_GATED_MLP_FP8
        or get_chip_version() != pltpu.ChipVersion.TPU_7X
    ):
        return None
    if (
        getattr(model_config, "quantization", "fp8") is None
        and not envs.QUANTIZE_ON_LOAD_PREFIXES
    ):
        return None

    archs = set(getattr(model_config, "architectures", None) or ())
    matched_targets = [
        target for target in _ALLOWED_GATED_MLP_TARGETS if archs & target.architectures
    ]
    if not matched_targets:
        return None

    # Query static chip VMEM once outside Dynamo tracing so the compiled forward
    # path sees a plain integer constant rather than calling into tpu_info.
    vmem_capacity_bytes = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )

    patched_names: list[str] = []
    for target in matched_targets:
        mod = importlib.import_module(target.module)
        cls = getattr(mod, target.class_name, None)
        if cls is None:
            continue
        fwd = cls.__dict__.get("forward")
        if fwd is None or getattr(fwd, "_tpu_fused_gated_mlp", False):
            continue
        if not _matches_standard_gated_mlp_forward(fwd):
            logger.warning(
                "Skipping TPU fused gated-MLP patch for %s.%s: "
                "forward signature changed.",
                target.module,
                target.class_name,
            )
            continue
        cls.forward = _make_fused_mlp_forward(fwd, vmem_capacity_bytes)  # type: ignore[method-assign]
        patched_names.append(f"{target.module}.{target.class_name}")

    if patched_names:
        logger.info(
            "Applied TPU patch: fused Pallas gated-MLP activation+FP8 (%s)",
            ", ".join(patched_names),
        )
    return None
