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
"""Public PyTree while-loop wrapper and step-slice update helpers."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.while_loop import while_loop_op

_TPU_DEVICE_TYPES = frozenset({"tpu", "privateuseone"})
_INTEGER_INDEX_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
)

__all__ = ["masked_update_step_slice", "tpu_while_loop"]


class _WhileCarryStorageOverlapError(ValueError):
    """Raised when distinct while-loop carry leaves share underlying storage."""


def _is_wrapped_or_symbolic_tensor(tensor: torch.Tensor) -> bool:
    """Returns True for Tensor subclasses, functorch/FakeTensor wrappers, or proxies."""
    return (
        type(tensor) is not torch.Tensor
        or torch._C._functorch.is_functorch_wrapped_tensor(tensor)
        or torch._subclasses.fake_tensor.is_fake(tensor)
        or bool(torch.overrides.has_torch_function_unary(tensor))
    )


def _tensor_storage_base_ptr(tensor: torch.Tensor) -> int | None:
    """Returns physical storage base pointer for non-empty concrete tensors."""
    if (
        tensor.numel() == 0
        or tensor.element_size() == 0
        or tensor.is_meta
        or torch._subclasses.fake_tensor.is_fake(tensor)
    ):
        return None
    with contextlib.suppress(RuntimeError):
        offset_elems = int(tensor.storage_offset())
        storage_base = 0
        with contextlib.suppress(RuntimeError):
            storage_base = int(tensor.untyped_storage().data_ptr())
        if storage_base != 0:
            return storage_base
        data_ptr = int(tensor.data_ptr())
        if data_ptr != 0:
            return data_ptr - offset_elems * tensor.element_size()
    return None


def _check_disjoint_concrete_storages(
    carries: Sequence[torch.Tensor],
    leaf_indices: Sequence[int] | None = None,
) -> None:
    seen: list[tuple[int, torch.Tensor, int]] = []
    for pos, tensor in enumerate(carries):
        label_idx = leaf_indices[pos] if leaf_indices is not None else pos
        base_ptr = _tensor_storage_base_ptr(tensor)
        if base_ptr is None:
            continue
        for prev_label, prev_tensor, prev_base in seen:
            if tensor is not prev_tensor and prev_base == base_ptr:
                raise _WhileCarryStorageOverlapError(
                    f"tpu_while_loop carry leaf {label_idx} shares underlying "
                    f"storage with distinct carry leaf {prev_label} "
                    "(overlapping or disjoint views of the same buffer); pass "
                    "identical Tensor objects for cross-layer KV sharing or "
                    "clone separate buffers."
                )
        seen.append((label_idx, tensor, base_ptr))


@torch.library.custom_op("vllm_torchtpu::_validate_while_carries", mutates_args=())
def _validate_while_carries(carries: list[torch.Tensor]) -> list[torch.Tensor]:
    _check_disjoint_concrete_storages(carries)
    return [
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device).copy_(
            t
        )
        for t in carries
    ]


@_validate_while_carries.register_fake
def _(carries: list[torch.Tensor]) -> list[torch.Tensor]:
    return [
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        for t in carries
    ]


def _validate_cond_output(pred: Any) -> torch.Tensor:
    if not isinstance(pred, torch.Tensor):
        raise TypeError(
            f"while_loop cond_fn must return a 0-D torch.bool Tensor, got {type(pred)}"
        )
    if pred.dtype != torch.bool or pred.ndim != 0:
        raise ValueError(
            "while_loop cond_fn must return a 0-D torch.bool Tensor, "
            f"got dtype={pred.dtype}, shape={tuple(pred.shape)}"
        )
    return pred


def _evaluate_eager_cond(pred: Any) -> bool:
    return pred if isinstance(pred, bool) else bool(_validate_cond_output(pred).item())


def _validate_body_outputs(
    raw_out: Any,
    reference_carries: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if not isinstance(raw_out, (tuple, list)):
        raise TypeError(
            f"while_loop body_fn must return a tuple or list, got {type(raw_out)}"
        )
    if len(raw_out) != len(reference_carries):
        raise ValueError(
            f"while_loop body_fn returned {len(raw_out)} values, expected "
            f"{len(reference_carries)}"
        )
    validated: list[torch.Tensor] = []
    for idx, (ref, out) in enumerate(zip(reference_carries, raw_out, strict=True)):
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"while_loop body_fn output[{idx}] must be a Tensor, got {type(out)}"
            )
        if out.shape != ref.shape or out.dtype != ref.dtype:
            raise ValueError(
                f"while_loop body_fn output[{idx}] metadata mismatch: expected "
                f"({tuple(ref.shape)}, {ref.dtype}), got "
                f"({tuple(out.shape)}, {out.dtype})"
            )
        if out.device != ref.device:
            raise ValueError(
                f"while_loop body_fn output[{idx}] device mismatch: expected "
                f"{ref.device}, got {out.device}"
            )
        validated.append(out)
    return validated


def _clone_preserving_strides(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous() or type(tensor) is not torch.Tensor:
        return tensor.clone()
    strides = tuple(int(s) for s in tensor.stride())
    active = sorted(
        ((int(d), int(s)) for d, s in zip(tensor.shape, strides, strict=True) if d > 1),
        key=lambda pair: pair[1],
    )
    if (
        0 in strides
        or any(s <= 0 for _, s in active)
        or any(
            active[i][0] * active[i][1] > active[i + 1][1]
            for i in range(len(active) - 1)
        )
    ):
        return tensor.clone()
    try:
        out = torch.empty_strided(
            tensor.shape, strides, dtype=tensor.dtype, device=tensor.device
        )
        return out.copy_(tensor)
    except NotImplementedError:
        # Backends without aten::empty_strided (e.g. XLA/PrivateUse1) fall back
        # to a contiguous clone.
        return tensor.clone()


def _clone_aliased_eager_outputs(
    outputs: Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    forbidden_bases = {
        base for t in inputs if (base := _tensor_storage_base_ptr(t)) is not None
    }
    result: list[torch.Tensor] = []
    for out in outputs:
        base = _tensor_storage_base_ptr(out)
        if base is None or base in forbidden_bases:
            out = _clone_preserving_strides(out)
            base = _tensor_storage_base_ptr(out)
        if base is not None:
            forbidden_bases.add(base)
        result.append(out)
    return tuple(result)


def _call_eager_checking_version(
    fn: Callable[..., Any],
    args: Sequence[torch.Tensor],
    label: str,
) -> Any:
    call_args = list(args)
    before = [
        (None if a.is_inference() else a._version, a.data_ptr()) for a in call_args
    ]
    raw_out = fn(*call_args)
    for idx, (arg, (ver, ptr)) in enumerate(zip(call_args, before, strict=True)):
        cur_ver = None if arg.is_inference() else arg._version
        if (ver is not None and cur_ver != ver) or arg.data_ptr() != ptr:
            raise RuntimeError(
                f"In-place mutation of {label} input argument {idx} is not "
                "supported in TPU while_loop; use functional tensor updates."
            )
    return raw_out


def _run_eager_cpu_while_loop(
    cond_fn: Callable[..., Any],
    body_fn: Callable[..., Any],
    carried_list: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    carries = list(carried_list)
    while _evaluate_eager_cond(
        _call_eager_checking_version(cond_fn, carries, "cond_fn")
    ):
        carries = _validate_body_outputs(
            _call_eager_checking_version(body_fn, carries, "body_fn"),
            carried_list,
        )
    return _clone_aliased_eager_outputs(carries, carried_list)


def _deduplicate_carry_leaves(
    flat_carries: Sequence[torch.Tensor],
) -> tuple[list[torch.Tensor], list[int], list[list[int]], list[int]]:
    unique_carries: list[torch.Tensor] = []
    leaf_to_unique_idx: list[int] = []
    unique_to_leaf_indices: list[list[int]] = []
    primary_leaf_indices: list[int] = []

    for leaf_idx, leaf in enumerate(flat_carries):
        existing = next(
            (idx for idx, uniq in enumerate(unique_carries) if leaf is uniq), None
        )
        if existing is not None:
            leaf_to_unique_idx.append(existing)
            unique_to_leaf_indices[existing].append(leaf_idx)
        else:
            leaf_to_unique_idx.append(len(unique_carries))
            unique_carries.append(leaf)
            unique_to_leaf_indices.append([leaf_idx])
            primary_leaf_indices.append(leaf_idx)

    return (
        unique_carries,
        leaf_to_unique_idx,
        unique_to_leaf_indices,
        primary_leaf_indices,
    )


def tpu_while_loop[T](
    cond_fn: Callable[..., torch.Tensor | bool],
    body_fn: Callable[..., T],
    carried_inputs: T,
) -> T:
    """PyTree-aware ``torch.while_loop`` wrapper for TPU execution.

    Deduplicates identical ``Tensor`` leaves (``a is b``, e.g. cross-layer
    shared KV pools) into one physical carry slot so XLA buffer donation is
    preserved, and rejects distinct carry views sharing underlying storage.
    Default-generator RNG ops inside ``body_fn`` advance across iterations via
    ``2xui64`` Philox carry splicing. Under ``torch.compile``, carries must
    satisfy ``is_non_overlapping_and_dense()``.
    """
    flat_carries, tree_spec = pytree.tree_flatten(carried_inputs)
    if not flat_carries:
        raise ValueError("tpu_while_loop carried_inputs must contain >= 1 Tensor")
    for idx, leaf in enumerate(flat_carries):
        if not isinstance(leaf, torch.Tensor):
            raise TypeError(
                f"tpu_while_loop carry leaf {idx} must be a torch.Tensor, "
                f"got {type(leaf)}"
            )

    (
        unique_carries,
        leaf_to_unique_idx,
        unique_to_leaf_indices,
        primary_leaf_indices,
    ) = _deduplicate_carry_leaves(flat_carries)

    first_dev = unique_carries[0].device
    for idx, carry in enumerate(unique_carries[1:], start=1):
        if carry.device != first_dev:
            raise ValueError(
                f"All tpu_while_loop carried_inputs must reside on {first_dev}, "
                f"got {carry.device} at unique carry {idx}."
            )

    if torch.compiler.is_compiling():
        if len(unique_carries) >= 2:
            unique_carries = list(_validate_while_carries(unique_carries))
    else:
        _check_disjoint_concrete_storages(unique_carries, primary_leaf_indices)

    has_shared_leaves = len(unique_carries) != len(flat_carries)
    is_plain_tuple = type(carried_inputs) is tuple
    is_flat_tuple = (
        not has_shared_leaves
        and is_plain_tuple
        and all(
            isinstance(x, torch.Tensor) for x in cast(tuple[Any, ...], carried_inputs)
        )
    )

    def _wrapped_cond(*unique_args: torch.Tensor) -> torch.Tensor | bool:
        restored = pytree.tree_unflatten(
            [unique_args[u_idx] for u_idx in leaf_to_unique_idx], tree_spec
        )
        return cond_fn(*restored) if is_plain_tuple else cond_fn(restored)

    def _wrapped_body(*unique_args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        restored = pytree.tree_unflatten(
            [unique_args[u_idx] for u_idx in leaf_to_unique_idx], tree_spec
        )
        out = body_fn(*restored) if is_plain_tuple else body_fn(restored)
        out_leaves, out_spec = pytree.tree_flatten(out)
        if out_spec != tree_spec:
            raise ValueError(
                "tpu_while_loop body_fn output PyTree structure does not match "
                f"carried_inputs: expected {tree_spec}, got {out_spec}"
            )
        if not has_shared_leaves:
            return tuple(out_leaves)
        dedup_outs: list[torch.Tensor] = []
        for u_idx, group_indices in enumerate(unique_to_leaf_indices):
            if len(group_indices) == 1:
                dedup_outs.append(out_leaves[group_indices[0]])
                continue
            in_tensor = unique_args[u_idx]
            updated = [
                out_leaves[l_idx]
                for l_idx in group_indices
                if out_leaves[l_idx] is not in_tensor
            ]
            if any(t is not updated[0] for t in updated[1:]):
                raise ValueError(
                    f"tpu_while_loop carry leaves {group_indices} shared the same "
                    "input Tensor and were deduplicated, but body_fn returned "
                    "distinct updated Tensors across those slots."
                )
            dedup_outs.append(updated[0] if updated else in_tensor)
        return tuple(dedup_outs)

    w_cond: Callable[..., Any] = cond_fn if is_flat_tuple else _wrapped_cond
    w_body: Callable[..., Any] = body_fn if is_flat_tuple else _wrapped_body
    carry_tuple = tuple(unique_carries)

    in_eager = (
        not torch.compiler.is_compiling()
        and not torch.fx._symbolic_trace.is_fx_symbolic_tracing()
        and torch.utils._python_dispatch._get_current_dispatch_mode() is None
    )
    if in_eager and all(
        c.device.type == "cpu"
        and not c.requires_grad
        and not _is_wrapped_or_symbolic_tensor(c)
        for c in unique_carries
    ):
        raw_out: Sequence[torch.Tensor] = _run_eager_cpu_while_loop(
            w_cond, w_body, carry_tuple
        )
    elif in_eager and first_dev.type in _TPU_DEVICE_TYPES:
        while_hop: Callable[..., Any] = while_loop_op
        raw_out = while_hop(w_cond, w_body, carry_tuple, ())
    else:
        raw_out = torch.while_loop(w_cond, w_body, carry_tuple)

    if is_flat_tuple:
        return cast(T, tuple(raw_out))
    return pytree.tree_unflatten(
        [raw_out[u_idx] for u_idx in leaf_to_unique_idx], tree_spec
    )


def masked_update_step_slice(
    buffer: torch.Tensor,
    update: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Functionally writes ``update`` into a small step ``buffer`` via ``where``.

    Lowers to ``O(buffer)`` ``int32``-indexed ``torch.where`` selection (plus
    ``torch.index_select`` when ``slice_len > 1``), clamping ``index`` to
    ``[0, buffer.shape[dim] - slice_len]``. Intended for small decode step
    windows (``[N_max, B, ...]``), not large KV-cache pools.
    """
    if not isinstance(buffer, torch.Tensor) or buffer.ndim == 0:
        raise ValueError(
            "masked_update_step_slice requires buffer Tensor with ndim >= 1, "
            f"got {type(buffer)} (ndim={getattr(buffer, 'ndim', None)})"
        )
    if not isinstance(update, torch.Tensor):
        raise ValueError(
            f"masked_update_step_slice requires update Tensor, got {type(update)}"
        )
    if update.dtype != buffer.dtype:
        raise ValueError(
            "masked_update_step_slice requires matching buffer and update dtypes, "
            f"got {buffer.dtype} vs {update.dtype}"
        )
    if not isinstance(index, torch.Tensor) or index.ndim != 0:
        raise ValueError(
            f"index must be a 0-D integer Tensor, got {type(index)} "
            f"with shape {getattr(index, 'shape', None)}"
        )
    if index.dtype not in _INTEGER_INDEX_DTYPES:
        raise ValueError(f"index must have an integer dtype, got {index.dtype}")
    if update.device != buffer.device or index.device != buffer.device:
        raise ValueError(
            "masked_update_step_slice requires buffer, update, and index on the "
            f"same device, got {buffer.device}, {update.device}, {index.device}"
        )

    norm_dim = dim if dim >= 0 else buffer.ndim + dim
    if norm_dim < 0 or norm_dim >= buffer.ndim:
        raise ValueError(f"dim={dim} out of range for buffer of rank {buffer.ndim}")

    src = update.unsqueeze(norm_dim) if update.ndim == buffer.ndim - 1 else update
    if src.ndim != buffer.ndim:
        raise ValueError(
            f"update rank must be {buffer.ndim - 1} or {buffer.ndim}, got {update.ndim}"
        )
    if any(
        ax != norm_dim and src.shape[ax] != buffer.shape[ax]
        for ax in range(buffer.ndim)
    ):
        raise ValueError(
            f"update shape {tuple(update.shape)} incompatible with buffer "
            f"shape {tuple(buffer.shape)} along dim={dim}"
        )

    dim_size = buffer.shape[norm_dim]
    slice_len = src.shape[norm_dim]
    if slice_len > dim_size:
        raise ValueError(
            f"update slice length {slice_len} exceeds buffer size {dim_size} "
            f"along dim={dim}"
        )
    if slice_len == 0:
        return buffer.clone()

    max_start = dim_size - slice_len
    if index.dtype == torch.int32:
        start_i32 = index.clamp(0, max_start)
    elif index.dtype in (torch.int8, torch.int16, torch.uint8, torch.uint16):
        start_i32 = index.to(dtype=torch.int32).clamp(0, max_start)
    elif index.dtype == torch.int64:
        start_i32 = index.clamp(0, max_start).to(dtype=torch.int32)
    else:
        signed_i64 = index.to(dtype=torch.int64)
        max_i64 = torch.tensor(max_start, device=index.device, dtype=torch.int64)
        start_i32 = torch.where(
            signed_i64 < 0, max_i64, signed_i64.clamp(0, max_start)
        ).to(dtype=torch.int32)

    coords = torch.arange(dim_size, device=buffer.device, dtype=torch.int32)
    bcast_shape = [dim_size if ax == norm_dim else 1 for ax in range(buffer.ndim)]
    if slice_len == 1:
        return torch.where((coords == start_i32).reshape(bcast_shape), src, buffer)

    rel_idx = (coords - start_i32).clamp(0, slice_len - 1)
    gathered = torch.index_select(src, norm_dim, rel_idx)
    mask = ((coords >= start_i32) & (coords < start_i32 + slice_len)).reshape(
        bcast_shape
    )
    return torch.where(mask, gathered, buffer)
