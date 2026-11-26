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
"""Pure-PyTorch 12-op executor, validator, and inverter for ``weight_mapping``.

Contract summary (zero JAX imports):
- ``stripes`` records parallel axis sizes (e.g. ``tp_size``, ``ep_size``,
  ``moe_tp_size``) baked into the manifest; each rule's ``sharding`` names the
  axis (or ``None``) per output dimension.
- Rules run left-to-right; only ``__tmp__:*`` target keys may be consumed by a
  later rule, and ``__tmp__:*`` keys are pruned before returning.
- ``dropped_keys`` skips any rule reading a dropped source (transitively).
- ``tied_keys`` lists ``[src, dst]`` alias pairs copied after rule execution.
- ``invert_manifest`` reverses rules/ops for checkpoint export and round-trips.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeGuard

import torch

SCHEMA_VERSION = "weight_mapping"
TMP_KEY_PREFIX = "__tmp__:"


class OpName(str, Enum):
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    SPLIT = "split"
    CONCAT = "concat"
    STACK = "stack"
    UNSTACK = "unstack"
    UNPAD = "unpad"
    PAD = "pad"
    REPEAT = "repeat"
    UNREPEAT = "unrepeat"
    SCALE = "scale"
    CAST = "cast"


OpSpec = dict[str, dict[str, Any]]
RuleSpec = str | Mapping[str, Any]

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

_TOP_KEYS = frozenset(
    {
        "schema_version",
        "expert_axis_grouped",
        "stripes",
        "tied_keys",
        "dropped_keys",
        "rules",
    }
)
_RULE_KEYS = frozenset({"source_keys", "target_keys", "ops", "sharding"})


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_seq(x: Any) -> TypeGuard[Sequence[Any]]:
    return isinstance(x, Sequence) and not isinstance(x, (str, bytes))


def _check_keys(obj: Mapping[str, Any], expected: set[str], ctx: str) -> None:
    keys = set(obj.keys())
    if keys != expected:
        raise ValueError(f"{ctx}: expected keys {sorted(expected)}, got {sorted(keys)}")


def _check_pos_ints(seq: Any, ctx: str) -> list[int]:
    if not _is_seq(seq) or len(seq) == 0 or not all(_is_int(v) and v > 0 for v in seq):
        raise ValueError(f"{ctx}: expected non-empty sequence of positive ints")
    return [int(v) for v in seq]


def _validate_op(op: Any, ctx: str) -> None:
    if not isinstance(op, Mapping) or len(op) != 1:
        raise ValueError(f"{ctx}: each op must be a single-key mapping {{op: kwargs}}")
    op_name, kw = next(iter(op.items()))
    if not isinstance(kw, Mapping):
        raise ValueError(f"{ctx}: kwargs for {op_name!r} must be a mapping")

    if op_name == OpName.RESHAPE:
        _check_keys(kw, {"src_shape", "dst_shape"}, f"{ctx}.reshape")
        for field in ("src_shape", "dst_shape"):
            shape = kw[field]
            if not _is_seq(shape) or not all(_is_int(v) and v > 0 for v in shape):
                raise ValueError(f"{ctx}.reshape.{field}: expected positive int dims")
        if math.prod(kw["src_shape"]) != math.prod(kw["dst_shape"]):
            raise ValueError(
                f"{ctx}.reshape: src_shape and dst_shape element counts differ"
            )
    elif op_name == OpName.TRANSPOSE:
        _check_keys(kw, {"axes"}, f"{ctx}.transpose")
        axes = kw["axes"]
        if not _is_seq(axes) or not all(_is_int(a) for a in axes):
            raise ValueError(f"{ctx}.transpose.axes: expected int sequence")
        if sorted(int(a) for a in axes) != list(range(len(axes))):
            raise ValueError(f"{ctx}.transpose.axes: must be a permutation of 0..N-1")
    elif op_name in (OpName.SPLIT, OpName.CONCAT):
        _check_keys(kw, {"axis", "sizes"}, f"{ctx}.{op_name}")
        if not _is_int(kw["axis"]):
            raise ValueError(f"{ctx}.{op_name}.axis: expected int")
        _check_pos_ints(kw["sizes"], f"{ctx}.{op_name}.sizes")
    elif op_name in (OpName.STACK, OpName.UNSTACK):
        _check_keys(kw, {"axis"}, f"{ctx}.{op_name}")
        if not _is_int(kw["axis"]):
            raise ValueError(f"{ctx}.{op_name}.axis: expected int")
    elif op_name in (OpName.UNPAD, OpName.PAD):
        _check_keys(kw, {"axis", "sizes"}, f"{ctx}.{op_name}")
        if not _is_int(kw["axis"]):
            raise ValueError(f"{ctx}.{op_name}.axis: expected int")
        sizes = kw["sizes"]
        if (
            not _is_seq(sizes)
            or len(sizes) != 2
            or not all(_is_int(v) and v >= 0 for v in sizes)
        ):
            raise ValueError(f"{ctx}.{op_name}.sizes: expected [left, right] >= 0")
    elif op_name in (OpName.REPEAT, OpName.UNREPEAT):
        _check_keys(kw, {"axis", "times"}, f"{ctx}.{op_name}")
        if not _is_int(kw["axis"]) or not (_is_int(kw["times"]) and kw["times"] > 0):
            raise ValueError(f"{ctx}.{op_name}: expected int axis and times > 0")
    elif op_name == OpName.SCALE:
        _check_keys(kw, {"factor"}, f"{ctx}.scale")
        f = kw["factor"]
        if (
            not isinstance(f, (int, float))
            or isinstance(f, bool)
            or not math.isfinite(f)
            or f == 0
        ):
            raise ValueError(f"{ctx}.scale.factor: expected finite non-zero number")
    elif op_name == OpName.CAST:
        _check_keys(kw, {"src_dtype", "dst_dtype"}, f"{ctx}.cast")
        for field in ("src_dtype", "dst_dtype"):
            if not isinstance(kw[field], str) or kw[field] not in _DTYPE_MAP:
                raise ValueError(f"{ctx}.cast.{field}: unknown dtype {kw[field]!r}")
    else:
        raise ValueError(f"{ctx}: unknown op {op_name!r}")


def _parse_manifest(manifest: RuleSpec) -> dict[str, Any]:
    if isinstance(manifest, str):
        try:
            data = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {exc}") from exc
    elif isinstance(manifest, Mapping):
        data = dict(manifest)
    else:
        raise TypeError(f"Expected str or Mapping, got {type(manifest).__name__}")

    if not isinstance(data, dict):
        raise ValueError("Manifest JSON must decode to an object")
    extra = set(data.keys()) - _TOP_KEYS
    if extra:
        raise ValueError(f"Unknown top-level manifest keys: {sorted(extra)}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {data.get('schema_version')!r}"
        )

    eag = data.get("expert_axis_grouped", False)
    if not isinstance(eag, bool):
        raise ValueError("expert_axis_grouped must be a bool")

    stripes_raw = data.get("stripes", {})
    if not isinstance(stripes_raw, Mapping):
        raise ValueError("stripes must be a mapping")
    stripes: dict[str, int] = {}
    for k, v in stripes_raw.items():
        if not isinstance(k, str) or not (_is_int(v) and v > 0):
            raise ValueError(f"stripes[{k!r}] must be a positive int, got {v!r}")
        stripes[k] = int(v)

    tied_raw = data.get("tied_keys", [])
    if not _is_seq(tied_raw):
        raise ValueError("tied_keys must be a sequence")
    tied_keys: list[list[str]] = []
    for pair in tied_raw:
        if (
            not _is_seq(pair)
            or len(pair) != 2
            or not all(isinstance(s, str) for s in pair)
        ):
            raise ValueError(
                f"tied_keys entries must be [src, dst] pairs, got {pair!r}"
            )
        tied_keys.append([pair[0], pair[1]])

    dropped_raw = data.get("dropped_keys", [])
    if not _is_seq(dropped_raw) or not all(isinstance(s, str) for s in dropped_raw):
        raise ValueError("dropped_keys must be a sequence of strings")

    rules_raw = data.get("rules")
    if not _is_seq(rules_raw):
        raise ValueError("rules is required and must be a sequence")
    rules: list[dict[str, Any]] = []
    for idx, rule in enumerate(rules_raw):
        ctx = f"rules[{idx}]"
        if not isinstance(rule, Mapping):
            raise ValueError(f"{ctx}: must be a mapping")
        extra_r = set(rule.keys()) - _RULE_KEYS
        if extra_r:
            raise ValueError(f"{ctx}: unknown keys {sorted(extra_r)}")
        for req in ("source_keys", "target_keys"):
            vals = rule.get(req)
            if (
                not _is_seq(vals)
                or not vals
                or not all(isinstance(s, str) for s in vals)
            ):
                raise ValueError(
                    f"{ctx}.{req}: must be a non-empty sequence of strings"
                )
        if "ops" not in rule or not _is_seq(rule["ops"]):
            raise ValueError(f"{ctx}.ops: is required and must be a sequence")
        for op_i, op in enumerate(rule["ops"]):
            _validate_op(op, f"{ctx}.ops[{op_i}]")
        entry: dict[str, Any] = {
            "source_keys": list(rule["source_keys"]),
            "target_keys": list(rule["target_keys"]),
            "ops": [dict(op) for op in rule["ops"]],
        }
        if "sharding" in rule:
            sh = rule["sharding"]
            if not _is_seq(sh) or not all(a is None or isinstance(a, str) for a in sh):
                raise ValueError(f"{ctx}.sharding: must be a sequence of str | None")
            entry["sharding"] = list(sh)
        rules.append(entry)

    produced: set[str] = set()
    consumed_tmp: set[str] = set()
    for idx, entry in enumerate(rules):
        for sk in entry["source_keys"]:
            if sk.startswith(TMP_KEY_PREFIX):
                if sk not in produced:
                    raise ValueError(
                        f"rules[{idx}]: {sk!r} consumed before being produced"
                    )
                consumed_tmp.add(sk)
        for tk in entry["target_keys"]:
            if tk in produced:
                raise ValueError(f"rules[{idx}]: duplicate target key {tk!r}")
            produced.add(tk)
    dangling = sorted(
        k for k in produced if k.startswith(TMP_KEY_PREFIX) and k not in consumed_tmp
    )
    if dangling:
        raise ValueError(f"Unconsumed temporary target key(s): {dangling}")

    return {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": eag,
        "stripes": stripes,
        "tied_keys": tied_keys,
        "dropped_keys": list(dropped_raw),
        "rules": rules,
    }


def validate_manifest(manifest: RuleSpec) -> dict[str, Any]:
    """Validate a ``weight_mapping`` manifest and return its normalized dict."""
    return _parse_manifest(manifest)


def _norm_axis(axis: int, ndim: int, op: str) -> int:
    norm = axis + ndim if axis < 0 else axis
    if not (0 <= norm < ndim):
        raise ValueError(f"{op}: axis {axis} out of range for ndim={ndim}")
    return norm


def _apply_unary(op_name: str, kw: Mapping[str, Any], t: torch.Tensor) -> torch.Tensor:
    if op_name == OpName.RESHAPE:
        if list(t.shape) != list(kw["src_shape"]):
            raise ValueError(
                f"reshape: shape {list(t.shape)} != {list(kw['src_shape'])}"
            )
        return t.reshape(tuple(kw["dst_shape"]))
    if op_name == OpName.TRANSPOSE:
        axes = list(kw["axes"])
        if len(axes) != t.ndim:
            raise ValueError(f"transpose: len(axes)={len(axes)} != ndim={t.ndim}")
        return t.permute(*axes)
    if op_name == OpName.UNPAD:
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        left, right = kw["sizes"]
        length = int(t.shape[axis]) - left - right
        if length <= 0:
            raise ValueError(
                f"unpad: invalid [{left}, {right}] for dim={t.shape[axis]}"
            )
        return t.narrow(axis, left, length)
    if op_name == OpName.PAD:
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        left, right = kw["sizes"]
        if left == 0 and right == 0:
            return t
        out_shape = list(t.shape)
        out_shape[axis] += left + right
        out = torch.zeros(out_shape, dtype=t.dtype, device=t.device)
        out.narrow(axis, left, int(t.shape[axis])).copy_(t)
        return out
    if op_name == OpName.REPEAT:
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        return t.repeat_interleave(int(kw["times"]), dim=axis)
    if op_name == OpName.UNREPEAT:
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        times = int(kw["times"])
        if int(t.shape[axis]) % times != 0:
            raise ValueError(f"unrepeat: dim={t.shape[axis]} not divisible by {times}")
        idx = [slice(None)] * t.ndim
        idx[axis] = slice(0, int(t.shape[axis]), times)
        return t[tuple(idx)].contiguous()
    if op_name == OpName.SCALE:
        compute_dt = torch.float64 if t.dtype == torch.float64 else torch.float32
        return (t.to(compute_dt) * float(kw["factor"])).to(t.dtype)
    if op_name == OpName.CAST:
        src_dt, dst_dt = _DTYPE_MAP[kw["src_dtype"]], _DTYPE_MAP[kw["dst_dtype"]]
        if t.dtype != src_dt:
            raise ValueError(f"cast: tensor dtype {t.dtype} != src_dtype {src_dt}")
        return t.to(dst_dt)
    raise ValueError(f"Unknown op {op_name!r}")


def _apply_op(
    op_name: str, kw: Mapping[str, Any], tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    if op_name == OpName.CONCAT:
        axis = _norm_axis(kw["axis"], tensors[0].ndim, op_name)
        if [int(t.shape[axis]) for t in tensors] != list(kw["sizes"]):
            raise ValueError(
                f"concat: tensor sizes {[int(t.shape[axis]) for t in tensors]} "
                f"!= {list(kw['sizes'])}"
            )
        return [torch.cat(tensors, dim=axis)]
    if op_name == OpName.STACK:
        axis = _norm_axis(kw["axis"], tensors[0].ndim + 1, op_name)
        return [torch.stack(tensors, dim=axis)]
    if op_name == OpName.SPLIT:
        if len(tensors) != 1:
            raise ValueError(f"split: expected 1 tensor, got {len(tensors)}")
        t = tensors[0]
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        if sum(kw["sizes"]) != int(t.shape[axis]):
            raise ValueError(
                f"split: sum({kw['sizes']}) != shape[{axis}]={t.shape[axis]}"
            )
        return list(torch.split(t, list(kw["sizes"]), dim=axis))
    if op_name == OpName.UNSTACK:
        if len(tensors) != 1:
            raise ValueError(f"unstack: expected 1 tensor, got {len(tensors)}")
        t = tensors[0]
        axis = _norm_axis(kw["axis"], t.ndim, op_name)
        return list(torch.unbind(t, dim=axis))
    return [_apply_unary(op_name, kw, t) for t in tensors]


def _invert_op(op: Mapping[str, Any]) -> OpSpec:
    op_name, kw = next(iter(op.items()))
    if op_name == OpName.RESHAPE:
        return {
            "reshape": {
                "src_shape": list(kw["dst_shape"]),
                "dst_shape": list(kw["src_shape"]),
            }
        }
    if op_name == OpName.TRANSPOSE:
        axes = list(kw["axes"])
        inv = [0] * len(axes)
        for i, a in enumerate(axes):
            inv[a] = i
        return {"transpose": {"axes": inv}}
    if op_name == OpName.CONCAT:
        return {"split": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
    if op_name == OpName.SPLIT:
        return {"concat": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
    if op_name == OpName.STACK:
        return {"unstack": {"axis": kw["axis"]}}
    if op_name == OpName.UNSTACK:
        return {"stack": {"axis": kw["axis"]}}
    if op_name == OpName.PAD:
        return {"unpad": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
    if op_name == OpName.UNPAD:
        return {"pad": {"axis": kw["axis"], "sizes": list(kw["sizes"])}}
    if op_name == OpName.REPEAT:
        return {"unrepeat": {"axis": kw["axis"], "times": kw["times"]}}
    if op_name == OpName.UNREPEAT:
        return {"repeat": {"axis": kw["axis"], "times": kw["times"]}}
    if op_name == OpName.SCALE:
        return {"scale": {"factor": 1.0 / float(kw["factor"])}}
    if op_name == OpName.CAST:
        return {"cast": {"src_dtype": kw["dst_dtype"], "dst_dtype": kw["src_dtype"]}}
    raise ValueError(f"Unknown op {op_name!r}")


def invert_manifest(manifest: RuleSpec) -> dict[str, Any]:
    """Invert a ``weight_mapping`` manifest (reversing chained rules and ops)."""
    spec = _parse_manifest(manifest)
    fwd_dropped = set(spec["dropped_keys"])
    inv_dropped: list[str] = []
    inv_dropped_seen: set[str] = set()
    for r in spec["rules"]:
        if any(k in fwd_dropped for k in r["source_keys"]):
            fwd_dropped.update(r["target_keys"])
            for tk in r["target_keys"]:
                if not tk.startswith(TMP_KEY_PREFIX) and tk not in inv_dropped_seen:
                    inv_dropped.append(tk)
                    inv_dropped_seen.add(tk)
    inv_rules = [
        {
            "source_keys": list(r["target_keys"]),
            "target_keys": list(r["source_keys"]),
            "ops": [_invert_op(op) for op in reversed(r["ops"])],
        }
        for r in reversed(spec["rules"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": spec["expert_axis_grouped"],
        "stripes": dict(spec["stripes"]),
        "tied_keys": [],
        "dropped_keys": inv_dropped,
        "rules": inv_rules,
    }


def execute_manifest(
    manifest: RuleSpec,
    src_weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Execute a ``weight_mapping`` manifest over PyTorch tensors (with chaining)."""
    spec = _parse_manifest(manifest)
    dropped = set(spec["dropped_keys"])
    out: dict[str, torch.Tensor] = {}
    tmp: dict[str, torch.Tensor] = {}

    for idx, rule in enumerate(spec["rules"]):
        if any(k in dropped for k in rule["source_keys"]):
            dropped.update(rule["target_keys"])
            continue

        tensors: list[torch.Tensor] = []
        for k in rule["source_keys"]:
            if k in src_weights:
                tensors.append(src_weights[k])
            elif k.startswith(TMP_KEY_PREFIX) and k in tmp:
                tensors.append(tmp[k])
            else:
                raise KeyError(f"rules[{idx}]: missing source weight key {k!r}")

        for op in rule["ops"]:
            op_name, kw = next(iter(op.items()))
            tensors = _apply_op(op_name, kw, tensors)

        if len(tensors) != len(rule["target_keys"]):
            raise ValueError(
                f"rules[{idx}]: produced {len(tensors)} tensor(s) for "
                f"{len(rule['target_keys'])} target key(s)"
            )
        for tk, tensor in zip(rule["target_keys"], tensors, strict=True):
            if tk.startswith(TMP_KEY_PREFIX):
                tmp[tk] = tensor
            else:
                out[tk] = tensor

    for src_k, dst_k in spec["tied_keys"]:
        if src_k in out and dst_k not in out and dst_k not in dropped:
            out[dst_k] = out[src_k]
    return out


def from_hf(
    dst_rule: str,
    hf_weights: Mapping[str, torch.Tensor],
    mesh: Any = None,
) -> dict[str, torch.Tensor]:
    """Convert HuggingFace weights using a serialized JSON ``dst_rule`` string.

    When ``mesh`` (``ShardingCoords`` or axis-map ``Mapping``) is provided,
    slices each output tensor to the local rank's shard.
    """
    if not isinstance(dst_rule, str):
        raise TypeError(
            f"from_hf expects dst_rule to be str, got {type(dst_rule).__name__}"
        )
    spec = _parse_manifest(dst_rule)
    out = execute_manifest(spec, hf_weights)
    if mesh is not None:
        from vllm_torchtpu.rl.weight_installer import (
            extract_sharding_map,
            slice_tensor_for_shard,
        )

        smap = extract_sharding_map(spec)
        out = {
            k: slice_tensor_for_shard(k, t, smap.get(k), mesh) for k, t in out.items()
        }
    return out


def to_hf(
    src_rule: str,
    src_weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Execute a framework-to-HF JSON manifest (e.g. ``invert_manifest(dst_rule)``)."""
    if not isinstance(src_rule, str):
        raise TypeError(
            f"to_hf expects src_rule to be str, got {type(src_rule).__name__}"
        )
    return execute_manifest(src_rule, src_weights)
