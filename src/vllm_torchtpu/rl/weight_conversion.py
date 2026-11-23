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
"""Pure-PyTorch 12-op executor, schema validator, and manifest inverter.

Implements the Distributed RL Weight Conversion Protocol using only ``torch.*``
operations (zero JAX imports).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RuleSpec = Mapping[str, Any] | str

SCHEMA_VERSION = "weight_mapping"
_INTERMEDIATE_KEY_PREFIX = "__tmp__:"


class OpName(StrEnum):
    """Supported self-invertible primitive op names."""

    STACK = "stack"
    UNSTACK = "unstack"
    CONCAT = "concat"
    SPLIT = "split"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    SCALE = "scale"
    PAD = "pad"
    UNPAD = "unpad"
    CAST = "cast"
    REPEAT = "repeat"
    UNREPEAT = "unrepeat"


_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegInt = Annotated[int, Field(ge=0)]
_NonEmptyStrSeq = Annotated[Sequence[str], Field(min_length=1)]
_StrPair = Annotated[Sequence[str], Field(min_length=2, max_length=2)]
_NonEmptyPosIntSeq = Annotated[Sequence[_PositiveInt], Field(min_length=1)]
_NonNegIntPair = Annotated[Sequence[_NonNegInt], Field(min_length=2, max_length=2)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_mapping(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and not isinstance(data, dict):
            return dict(data)
        return data


class _AxisKwargs(_StrictModel):
    axis: int


class _ConcatSplitKwargs(_StrictModel):
    axis: int
    sizes: _NonEmptyPosIntSeq


class _ReshapeKwargs(_StrictModel):
    src_shape: Sequence[_NonNegInt]
    dst_shape: Sequence[_NonNegInt]

    @model_validator(mode="after")
    def _check_element_count(self) -> Self:
        if math.prod(self.src_shape) != math.prod(self.dst_shape):
            raise ValueError(
                f"reshape: element count mismatch between "
                f"src_shape={list(self.src_shape)} and "
                f"dst_shape={list(self.dst_shape)}"
            )
        return self


class _TransposeKwargs(_StrictModel):
    axes: Sequence[_NonNegInt]

    @field_validator("axes")
    @classmethod
    def _check_permutation(cls, v: Sequence[int]) -> Sequence[int]:
        if sorted(v) != list(range(len(v))):
            raise ValueError(
                f"transpose.axes: must be a permutation of 0..{len(v) - 1}, "
                f"got {list(v)}"
            )
        return v


class _ScaleKwargs(_StrictModel):
    factor: Annotated[float, Field(allow_inf_nan=False)]

    @field_validator("factor")
    @classmethod
    def _check_non_zero(cls, v: float) -> float:
        if v == 0:
            raise ValueError(f"scale.factor: must be finite and non-zero, got {v}")
        return v


class _PadUnpadKwargs(_StrictModel):
    axis: int
    sizes: _NonNegIntPair


class _CastKwargs(_StrictModel):
    src_dtype: str
    dst_dtype: str

    @field_validator("src_dtype", "dst_dtype")
    @classmethod
    def _check_dtype(cls, v: str) -> str:
        if v not in _DTYPE_MAP:
            raise ValueError(f"cast: unsupported dtype {v!r}")
        return v


class _RepeatUnrepeatKwargs(_StrictModel):
    axis: int
    times: _PositiveInt


_OP_KWARGS_MODELS: dict[str, type[_StrictModel]] = {
    OpName.STACK: _AxisKwargs,
    OpName.UNSTACK: _AxisKwargs,
    OpName.CONCAT: _ConcatSplitKwargs,
    OpName.SPLIT: _ConcatSplitKwargs,
    OpName.RESHAPE: _ReshapeKwargs,
    OpName.TRANSPOSE: _TransposeKwargs,
    OpName.SCALE: _ScaleKwargs,
    OpName.PAD: _PadUnpadKwargs,
    OpName.UNPAD: _PadUnpadKwargs,
    OpName.CAST: _CastKwargs,
    OpName.REPEAT: _RepeatUnrepeatKwargs,
    OpName.UNREPEAT: _RepeatUnrepeatKwargs,
}


class _RuleModel(_StrictModel):
    source_keys: _NonEmptyStrSeq
    target_keys: _NonEmptyStrSeq
    ops: Sequence[Mapping[str, Any]]
    sharding: Sequence[str | None] = Field(default_factory=list)

    @field_validator("ops")
    @classmethod
    def _validate_ops(
        cls, v: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        for idx, op in enumerate(v):
            if len(op) != 1:
                raise ValueError(
                    f"ops[{idx}]: each op must be a single-key mapping "
                    f"{{op_name: kwargs}}"
                )
            op_name, kwargs = next(iter(op.items()))
            if op_name not in _OP_KWARGS_MODELS:
                raise ValueError(f"ops[{idx}]: unknown op {op_name!r}")
            _OP_KWARGS_MODELS[op_name].model_validate(kwargs)
        return v


class _ManifestModel(_StrictModel):
    schema_version: Literal["weight_mapping"]
    expert_axis_grouped: bool = False
    stripes: Mapping[str, _PositiveInt] = Field(default_factory=dict)
    tied_keys: Sequence[_StrPair] = Field(default_factory=list)
    dropped_keys: Sequence[str] = Field(default_factory=list)
    rules: Sequence[_RuleModel]


def validate_manifest(spec: Mapping[str, Any]) -> None:
    """Strictly validate a ``weight_mapping`` manifest dictionary.

    Args:
        spec: Parsed manifest mapping to validate.

    Raises:
        ValueError: On any unknown key, unknown op, missing required field, or
            type/schema mismatch.
    """
    _ManifestModel.model_validate(spec)


def _parse_manifest(spec: RuleSpec) -> dict[str, Any]:
    if isinstance(spec, str):
        try:
            parsed = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(
                f"Manifest JSON must decode to a JSON object, "
                f"got {type(parsed).__name__}"
            )
        return dict(parsed)
    if isinstance(spec, Mapping):
        return dict(spec)
    raise ValueError(
        f"Manifest must be a mapping or JSON string, got {type(spec).__name__}"
    )


def _normalize_axis(
    axis: int,
    ndim: int,
    op_name: str,
    *,
    allow_end: bool = False,
) -> int:
    upper = ndim if allow_end else ndim - 1
    lower = -(ndim + 1) if allow_end else -ndim
    if axis < lower or axis > upper:
        raise ValueError(
            f"{op_name}: axis {axis} out of range for tensor with ndim={ndim}"
        )
    return axis + (ndim + 1 if allow_end else ndim) if axis < 0 else axis


def _apply_single_op(
    op_name: str,
    kwargs: Mapping[str, Any],
    tensor: torch.Tensor,
) -> torch.Tensor:
    if op_name == OpName.RESHAPE:
        src_shape = list(kwargs["src_shape"])
        dst_shape = list(kwargs["dst_shape"])
        if list(tensor.shape) != src_shape:
            raise ValueError(
                f"reshape: expected input shape {src_shape}, got {list(tensor.shape)}"
            )
        return tensor.reshape(dst_shape)

    if op_name == OpName.TRANSPOSE:
        axes = list(kwargs["axes"])
        if len(axes) != tensor.ndim:
            raise ValueError(
                f"transpose: len(axes)={len(axes)} does not match "
                f"tensor.ndim={tensor.ndim}"
            )
        return tensor.permute(*axes)

    if op_name == OpName.SCALE:
        factor = float(kwargs["factor"])
        orig_dtype = tensor.dtype
        return (tensor.to(torch.float32) * factor).to(orig_dtype)

    if op_name == OpName.PAD:
        axis = _normalize_axis(kwargs["axis"], tensor.ndim, OpName.PAD)
        left, right = kwargs["sizes"]
        if left == 0 and right == 0:
            return tensor
        out_shape = list(tensor.shape)
        out_shape[axis] += left + right
        out = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(left, left + tensor.shape[axis])
        out[tuple(slices)] = tensor
        return out

    if op_name == OpName.UNPAD:
        axis = _normalize_axis(kwargs["axis"], tensor.ndim, OpName.UNPAD)
        left, right = kwargs["sizes"]
        dim = tensor.shape[axis]
        if left + right > dim:
            raise ValueError(
                f"unpad: sizes {[left, right]} exceed dim={dim} on axis {axis}"
            )
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(left, dim - right)
        return tensor[tuple(slices)]

    if op_name == OpName.CAST:
        expected_src = _DTYPE_MAP[kwargs["src_dtype"]]
        dst_dtype = _DTYPE_MAP[kwargs["dst_dtype"]]
        if tensor.dtype != expected_src:
            raise ValueError(
                f"cast: expected input dtype {expected_src}, got {tensor.dtype}"
            )
        return tensor.to(dst_dtype)

    if op_name == OpName.REPEAT:
        axis = _normalize_axis(kwargs["axis"], tensor.ndim, OpName.REPEAT)
        return torch.repeat_interleave(tensor, kwargs["times"], dim=axis)

    if op_name == OpName.UNREPEAT:
        axis = _normalize_axis(kwargs["axis"], tensor.ndim, OpName.UNREPEAT)
        times = kwargs["times"]
        dim = tensor.shape[axis]
        if dim % times != 0:
            raise ValueError(
                f"unrepeat: dim={dim} on axis {axis} is not divisible by times={times}"
            )
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(None, None, times)
        return tensor[tuple(slices)]

    raise ValueError(f"Unknown op: {op_name!r}")


def _apply_op(
    op_name: str,
    kwargs: Mapping[str, Any],
    tensors: list[torch.Tensor],
) -> list[torch.Tensor]:
    if op_name == OpName.STACK:
        if not tensors:
            raise ValueError("stack: requires at least one input tensor")
        axis = _normalize_axis(
            kwargs["axis"], tensors[0].ndim, OpName.STACK, allow_end=True
        )
        ref_shape = tensors[0].shape
        for idx, t in enumerate(tensors[1:], start=1):
            if t.shape != ref_shape:
                raise ValueError(
                    f"stack: tensor[{idx}] shape {list(t.shape)} != {list(ref_shape)}"
                )
        return [torch.stack(tensors, dim=axis)]

    if op_name == OpName.UNSTACK:
        if len(tensors) != 1:
            raise ValueError(f"unstack: expected exactly 1 tensor, got {len(tensors)}")
        axis = _normalize_axis(kwargs["axis"], tensors[0].ndim, OpName.UNSTACK)
        return list(torch.unbind(tensors[0], dim=axis))

    if op_name == OpName.CONCAT:
        sizes = list(kwargs["sizes"])
        if len(tensors) != len(sizes):
            raise ValueError(
                f"concat: expected {len(sizes)} tensors for sizes={sizes}, "
                f"got {len(tensors)}"
            )
        axis = _normalize_axis(kwargs["axis"], tensors[0].ndim, OpName.CONCAT)
        ref_other_shape = tensors[0].shape[:axis] + tensors[0].shape[axis + 1 :]
        for idx, (t, expected_size) in enumerate(zip(tensors, sizes)):
            if t.ndim != tensors[0].ndim:
                raise ValueError(
                    f"concat: tensor[{idx}] ndim={t.ndim} != {tensors[0].ndim}"
                )
            if t.shape[axis] != expected_size:
                raise ValueError(
                    f"concat: tensor[{idx}] width {t.shape[axis]} on axis {axis} "
                    f"!= expected size {expected_size}"
                )
            other_shape = t.shape[:axis] + t.shape[axis + 1 :]
            if other_shape != ref_other_shape:
                raise ValueError(
                    f"concat: tensor[{idx}] non-concat shape {list(other_shape)} "
                    f"!= {list(ref_other_shape)}"
                )
        return [torch.cat(tensors, dim=axis)]

    if op_name == OpName.SPLIT:
        if len(tensors) != 1:
            raise ValueError(f"split: expected exactly 1 tensor, got {len(tensors)}")
        sizes = list(kwargs["sizes"])
        t = tensors[0]
        axis = _normalize_axis(kwargs["axis"], t.ndim, OpName.SPLIT)
        if t.shape[axis] != sum(sizes):
            raise ValueError(
                f"split: tensor width {t.shape[axis]} on axis {axis} "
                f"!= sum(sizes)={sum(sizes)}"
            )
        return list(torch.split(t, sizes, dim=axis))

    return [_apply_single_op(op_name, kwargs, t) for t in tensors]


def execute_manifest(
    manifest: RuleSpec,
    src_weights: Mapping[str, torch.Tensor],
    mesh: Any = None,
) -> dict[str, torch.Tensor]:
    """Execute a ``weight_mapping`` manifest over PyTorch tensors.

    Args:
        manifest: Validated manifest mapping or JSON string.
        src_weights: Mapping from source weight names to ``torch.Tensor``.
        mesh: Optional mesh argument accepted for protocol signature
            compatibility (unused by the pure-PyTorch executor).

    Returns:
        Dictionary mapping target weight names to converted ``torch.Tensor``.
    """
    del mesh
    spec = _parse_manifest(manifest)
    validate_manifest(spec)

    # Rule chaining invariant: ``src_weights`` always takes precedence over
    # ``out`` so a key written to the same name (``x -> x``) and read by a later
    # rule (``x -> y``) reads the original source tensor and is never treated as
    # an intermediate. Only keys absent from ``src_weights`` that are produced by
    # an earlier rule and consumed downstream (or prefixed with ``__tmp__:``) are
    # pruned before returning, including when a downstream rule is skipped via
    # ``dropped_keys``.
    dropped: set[str] = set(spec.get("dropped_keys", ()))
    out: dict[str, torch.Tensor] = {}
    intermediate_keys: set[str] = set()

    for rule_idx, rule in enumerate(spec["rules"]):
        source_keys: Sequence[str] = rule["source_keys"]
        target_keys: Sequence[str] = rule["target_keys"]

        if any(k in dropped for k in source_keys):
            for k in source_keys:
                if k not in src_weights and k in out:
                    intermediate_keys.add(k)
            dropped.update(target_keys)
            continue

        tensors: list[torch.Tensor] = []
        for k in source_keys:
            if k in src_weights:
                tensors.append(src_weights[k])
            elif k in out:
                tensors.append(out[k])
                intermediate_keys.add(k)
            else:
                raise KeyError(f"rules[{rule_idx}]: missing source weight key {k!r}")

        for op in rule["ops"]:
            op_name, kwargs = next(iter(op.items()))
            tensors = _apply_op(op_name, kwargs, tensors)

        if len(tensors) != len(target_keys):
            raise ValueError(
                f"rules[{rule_idx}]: produced {len(tensors)} tensor(s), "
                f"expected {len(target_keys)} for target_keys={list(target_keys)}"
            )
        for k, tensor in zip(target_keys, tensors):
            out[k] = tensor
            intermediate_keys.discard(k)

    for k in list(out.keys()):
        if k in intermediate_keys or k.startswith(_INTERMEDIATE_KEY_PREFIX):
            out.pop(k, None)

    for src_key, dst_key in spec.get("tied_keys", ()):
        if src_key in out and dst_key not in dropped:
            out[dst_key] = out[src_key].clone()

    return out


def from_hf_impl(
    dst_rule: RuleSpec,
    hf_weights: Mapping[str, torch.Tensor],
    mesh: Any = None,
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace weights to target engine layout using ``dst_rule``."""
    return execute_manifest(dst_rule, hf_weights, mesh=mesh)


def from_hf(
    dst_rule: str,
    hf_weights: Mapping[str, torch.Tensor],
    mesh: Any = None,
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace weights using a serialized JSON ``dst_rule`` string."""
    if not isinstance(dst_rule, str):
        raise TypeError(
            f"from_hf expects dst_rule to be str, got {type(dst_rule).__name__}"
        )
    return from_hf_impl(dst_rule, hf_weights, mesh=mesh)


def to_hf_impl(
    src_rule: RuleSpec,
    src_weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert engine weights to HuggingFace layout using ``src_rule``."""
    return execute_manifest(src_rule, src_weights)


def to_hf(
    src_rule: str,
    src_weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert engine weights using a serialized JSON ``src_rule`` string."""
    if not isinstance(src_rule, str):
        raise TypeError(
            f"to_hf expects src_rule to be str, got {type(src_rule).__name__}"
        )
    return to_hf_impl(src_rule, src_weights)


def _invert_op(op: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    op_name, kwargs = next(iter(op.items()))
    if op_name == OpName.STACK:
        return {OpName.UNSTACK.value: {"axis": kwargs["axis"]}}
    if op_name == OpName.UNSTACK:
        return {OpName.STACK.value: {"axis": kwargs["axis"]}}
    if op_name == OpName.CONCAT:
        return {
            OpName.SPLIT.value: {"axis": kwargs["axis"], "sizes": list(kwargs["sizes"])}
        }
    if op_name == OpName.SPLIT:
        return {
            OpName.CONCAT.value: {
                "axis": kwargs["axis"],
                "sizes": list(kwargs["sizes"]),
            }
        }
    if op_name == OpName.RESHAPE:
        return {
            OpName.RESHAPE.value: {
                "src_shape": list(kwargs["dst_shape"]),
                "dst_shape": list(kwargs["src_shape"]),
            }
        }
    if op_name == OpName.TRANSPOSE:
        axes = list(kwargs["axes"])
        inv_axes = [0] * len(axes)
        for new_pos, old_pos in enumerate(axes):
            inv_axes[old_pos] = new_pos
        return {OpName.TRANSPOSE.value: {"axes": inv_axes}}
    if op_name == OpName.SCALE:
        return {OpName.SCALE.value: {"factor": 1.0 / float(kwargs["factor"])}}
    if op_name == OpName.PAD:
        return {
            OpName.UNPAD.value: {"axis": kwargs["axis"], "sizes": list(kwargs["sizes"])}
        }
    if op_name == OpName.UNPAD:
        return {
            OpName.PAD.value: {"axis": kwargs["axis"], "sizes": list(kwargs["sizes"])}
        }
    if op_name == OpName.CAST:
        return {
            OpName.CAST.value: {
                "src_dtype": kwargs["dst_dtype"],
                "dst_dtype": kwargs["src_dtype"],
            }
        }
    if op_name == OpName.REPEAT:
        return {
            OpName.UNREPEAT.value: {"axis": kwargs["axis"], "times": kwargs["times"]}
        }
    if op_name == OpName.UNREPEAT:
        return {OpName.REPEAT.value: {"axis": kwargs["axis"], "times": kwargs["times"]}}
    raise ValueError(f"Unknown op: {op_name!r}")


def invert_manifest(spec: RuleSpec) -> dict[str, Any]:
    """Mechanically derive the exact inverse of a ``weight_mapping`` manifest.

    Swaps ``source_keys`` and ``target_keys``, reverses each rule's ``ops``
    sequence, and replaces every op with its mathematical inverse.

    Args:
        spec: Validated manifest mapping or JSON string.

    Returns:
        Inverse manifest dictionary conforming to ``weight_mapping`` schema.
    """
    parsed = _parse_manifest(spec)
    validate_manifest(parsed)

    dropped = set(parsed.get("dropped_keys", ()))
    inv_dropped: list[str] = []
    inv_rules: list[dict[str, Any]] = []

    for rule in parsed["rules"]:
        source_keys = list(rule["source_keys"])
        target_keys = list(rule["target_keys"])
        if any(k in dropped for k in source_keys):
            dropped.update(target_keys)
            for tk in target_keys:
                if tk not in inv_dropped:
                    inv_dropped.append(tk)

    for rule in reversed(parsed["rules"]):
        source_keys = list(rule["source_keys"])
        target_keys = list(rule["target_keys"])
        inv_ops = [_invert_op(op) for op in reversed(rule["ops"])]
        inv_rules.append(
            {
                "source_keys": target_keys,
                "target_keys": source_keys,
                "ops": inv_ops,
            }
        )

    stripes = parsed.get("stripes")
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": parsed.get("expert_axis_grouped", False),
        "stripes": dict(stripes) if isinstance(stripes, Mapping) else {},
        "tied_keys": [],
        "dropped_keys": inv_dropped,
        "rules": inv_rules,
    }


__all__ = [
    "OpName",
    "RuleSpec",
    "SCHEMA_VERSION",
    "execute_manifest",
    "from_hf",
    "from_hf_impl",
    "invert_manifest",
    "to_hf",
    "to_hf_impl",
    "validate_manifest",
]
