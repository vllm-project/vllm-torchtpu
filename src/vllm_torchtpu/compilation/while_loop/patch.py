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
"""TorchTPU ``while_loop`` HigherOrderOp lowering and FX pass patches."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import operator
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.while_loop import while_loop_op
from torch._library import utils as library_utils

from vllm_torchtpu.compilation.while_loop._mlir import (
    _assemble_while_loop_mlir,
    _parse_while_subgraphs,
)
from vllm_torchtpu.compilation.while_loop.ops import (
    _TPU_DEVICE_TYPES,
    _check_disjoint_concrete_storages,
    _clone_aliased_eager_outputs,
    _evaluate_eager_cond,
    _tensor_storage_base_ptr,
    _validate_body_outputs,
    _validate_cond_output,
    _WhileCarryStorageOverlapError,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_WHILE_CUSTOM_KERNEL_NAME = "vllm_torchtpu_while_loop"
_FX_ALL_REDUCE_SCHEMA_NAMES = frozenset(
    {
        "_c10d_functional::all_reduce",
        "_c10d_functional::all_reduce_",
        "_c10d_functional::all_reduce_coalesced",
        "_c10d_functional::all_reduce_coalesced_",
    }
)
_FX_BENIGN_C10D_SCHEMA_NAMES = frozenset({"_c10d_functional::wait_tensor"})
_FX_ORDER_SENSITIVE_COLLECTIVE_SCHEMA_NAMES = frozenset(
    {
        "_c10d_functional::all_gather_into_tensor",
        "_c10d_functional::all_gather_into_tensor_out",
        "_c10d_functional::all_gather_into_tensor_coalesced",
        "_c10d_functional::reduce_scatter_tensor",
        "_c10d_functional::reduce_scatter_tensor_out",
        "_c10d_functional::reduce_scatter_tensor_coalesced",
        "_c10d_functional::all_to_all_single",
        "_c10d_functional::broadcast",
        "_c10d_functional::broadcast_",
        "_c10d_functional::isend",
        "_c10d_functional::irecv",
        "_c10d_functional::batch_p2p_ops",
        "tpu::_all_gather_dim",
        "tpu::_reduce_scatter_dim",
        "tpu::_ragged_all_to_all",
    }
)

__all__ = ["install_while_loop_patch"]


def _find_overlapping_pair_indices(
    carries: Sequence[torch.Tensor],
) -> tuple[int, int] | None:
    seen: list[tuple[int, int]] = []
    for idx, tensor in enumerate(carries):
        base_ptr = _tensor_storage_base_ptr(tensor)
        if base_ptr is None:
            continue
        for prev_idx, prev_base in seen:
            if prev_base == base_ptr:
                return (prev_idx, idx)
        seen.append((idx, base_ptr))
    return None


def _format_carry_overlap_message(prev_idx: int, cur_idx: int) -> str:
    return (
        f"tpu_while_loop carry leaf {cur_idx} shares underlying storage with "
        f"distinct carry leaf {prev_idx} (overlapping or disjoint views of the "
        "same buffer); pass identical Tensor objects for cross-layer KV "
        "sharing or clone separate buffers."
    )


@torch.library.custom_op("vllm_torchtpu::_check_while_carry_args", mutates_args=())
def _check_while_carry_args(carries: list[torch.Tensor]) -> None:
    overlap = _find_overlapping_pair_indices(carries)
    if overlap is not None:
        raise _WhileCarryStorageOverlapError(_format_carry_overlap_message(*overlap))


@_check_while_carry_args.register_fake
def _(carries: list[torch.Tensor]) -> None:
    return None


class _DeferredCarryOverlapErrorModule(torch.nn.Module):
    """Picklable leaf ``nn.Module`` raising deferred ``ValueError`` at call time."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def forward(self, *args: Any) -> None:
        del args
        raise _WhileCarryStorageOverlapError(self.message)


@dataclasses.dataclass
class _PallasHookState:
    orig_call: Callable[..., Any]
    wrapper_call: Callable[..., Any]
    active_count: int = 0


class _CapturedDeviceAssignments(list[list[int]]):
    def __init__(self) -> None:
        super().__init__()
        self.saw_unassigned_custom_kernel: bool = False


@dataclasses.dataclass(frozen=True)
class _SubgraphTraceResult:
    mlir_text: str
    updates_rng: bool
    captured_da: list[int] | None
    saw_unassigned_custom_kernel: bool = False
    fx_collectives: tuple[bool, frozenset[int]] | None = None


@dataclasses.dataclass
class _SubgraphRngCapture:
    rng_begin: torch.Tensor | None = None
    rng_end: torch.Tensor | None = None
    updates_rng: bool = False


@dataclasses.dataclass(frozen=True)
class _RegionDeviceAssignmentReq:
    da: list[int] | None
    has_requirement: bool
    desc: str


class _WhileLoopThreadState(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.da_stacks_by_mod: dict[int, list[_CapturedDeviceAssignments]] = {}


_capture_hook_lock = threading.Lock()
_capture_hook_states: dict[int, _PallasHookState] = {}
_thread_local_state = _WhileLoopThreadState()


def _parse_mutated_arg_indices(mutated_arg_indices: Any) -> set[int]:
    if not mutated_arg_indices:
        return set()
    if isinstance(mutated_arg_indices, str):
        return {int(tok) for tok in mutated_arg_indices.split(",") if tok.strip()}
    if isinstance(mutated_arg_indices, Collection) and not isinstance(
        mutated_arg_indices, (str, bytes)
    ):
        return {
            int(idx)
            for idx in mutated_arg_indices
            if isinstance(idx, (int, str)) and not isinstance(idx, bool)
        }
    return set()


def _fx_nodes_in_value(value: Any) -> set[torch.fx.Node]:
    return {
        leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.fx.Node)
    }


def _aliases_of(
    placeholder_aliases: Mapping[torch.fx.Node, Collection[torch.fx.Node]],
    value: Any,
) -> set[torch.fx.Node]:
    return set().union(
        *(placeholder_aliases.get(n, ()) for n in _fx_nodes_in_value(value))
    )


def _getitem_aliases_source(arg0: torch.fx.Node, idx: Any) -> bool:
    if arg0.op != "call_function" or not isinstance(arg0.target, torch._ops.OpOverload):
        return False
    returns = arg0.target._schema.returns
    if len(returns) == 1:
        return returns[0].alias_info is not None
    if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(returns):
        return returns[idx].alias_info is not None
    return any(r.alias_info is not None for r in returns)


def _check_fx_graph_no_input_mutation(fn: Any, label: str) -> None:
    if not isinstance(fn, torch.fx.GraphModule):
        return

    aliases: dict[torch.fx.Node, set[torch.fx.Node]] = {}
    for node in fn.graph.nodes:
        if node.op == "placeholder":
            aliases[node] = {node}
            continue
        if node.op != "call_function":
            continue
        target = node.target
        if target is while_loop_op:
            aliases[node] = set()
            mut_indices = _parse_mutated_arg_indices(
                node.kwargs.get("mutated_arg_indices")
            )
            if mut_indices:
                carried = (
                    node.args[2]
                    if len(node.args) >= 3 and isinstance(node.args[2], (tuple, list))
                    else ()
                )
                additional = (
                    node.args[3]
                    if len(node.args) >= 4 and isinstance(node.args[3], (tuple, list))
                    else ()
                )
                flat_in = [*carried, *additional]
                written = set().union(
                    *(
                        _aliases_of(aliases, flat_in[i])
                        for i in mut_indices
                        if 0 <= i < len(flat_in)
                    )
                )
                if written:
                    names = ", ".join(p.name for p in written)
                    raise RuntimeError(
                        f"In-place mutation (while_loop) of {label} input "
                        f"({names}) is not supported in TPU while_loop."
                    )
            continue
        if target is torch.ops.aten._unsafe_view.default:
            aliases[node] = _aliases_of(aliases, node.args[0] if node.args else ())
            continue
        if target is operator.getitem:
            arg0 = node.args[0] if node.args else None
            idx = node.args[1] if len(node.args) > 1 else None
            aliases[node] = (
                _aliases_of(aliases, arg0)
                if isinstance(arg0, torch.fx.Node)
                and _getitem_aliases_source(arg0, idx)
                else set()
            )
            continue
        if not isinstance(target, torch._ops.OpOverload):
            continue

        schema = target._schema
        annotated = [
            (arg.alias_info, _aliases_of(aliases, val))
            for arg, val in library_utils.zip_schema(schema, node.args, node.kwargs)
            if arg.alias_info is not None
        ]
        written_phs = set().union(*(a for info, a in annotated if info.is_write))
        if written_phs:
            names = ", ".join(p.name for p in written_phs)
            raise RuntimeError(
                f"In-place mutation ({target}) of {label} input "
                f"({names}) is not supported in TPU while_loop."
            )
        if any(r.alias_info is not None for r in schema.returns):
            aliases[node] = set().union(*(a for _, a in annotated))


def _has_partial_storage(tensor: torch.Tensor, *, fail_closed: bool = False) -> bool:
    if tensor.numel() == 0 or tensor.element_size() == 0:
        return False
    if tensor.is_meta or torch._subclasses.fake_tensor.is_fake(tensor):
        return fail_closed
    try:
        storage_nbytes = int(tensor.untyped_storage().nbytes())
    except RuntimeError:
        return fail_closed
    return fail_closed if storage_nbytes == 0 else storage_nbytes != tensor.nbytes


def _safe_storage_offset(tensor: torch.Tensor) -> int:
    with contextlib.suppress(RuntimeError):
        return int(tensor.storage_offset())
    return 0


def _compute_allowed_donor_indices(
    num_carries: int,
    *tensor_sequences: Sequence[torch.Tensor],
) -> set[int]:
    """Returns carry indices proven to own full storage disjoint from all inputs."""
    allowed = set(range(num_carries))
    for seq in tensor_sequences:
        bases = [_tensor_storage_base_ptr(t) for t in seq]
        nonempty = [t.numel() > 0 and t.element_size() > 0 for t in seq]
        unresolved_nonempty = [
            is_ne and base is None for is_ne, base in zip(nonempty, bases, strict=True)
        ]
        for carry_idx in list(allowed):
            carry_t = seq[carry_idx]
            if any(
                other_idx != carry_idx and seq[other_idx] is carry_t
                for other_idx in range(len(seq))
            ) or _has_partial_storage(carry_t, fail_closed=True):
                allowed.discard(carry_idx)
                continue
            if not nonempty[carry_idx]:
                continue
            c_base = bases[carry_idx]
            if c_base is None or any(
                other_idx != carry_idx
                and (unresolved_nonempty[other_idx] or bases[other_idx] == c_base)
                for other_idx in range(len(seq))
            ):
                allowed.discard(carry_idx)
    return allowed


def _materialize_zero_offset_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if _safe_storage_offset(tensor) != 0 or _has_partial_storage(tensor):
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _restore_tensor_strides(
    tpu_torch_compile: Any,
    tensor: torch.Tensor,
    target_stride: tuple[int, ...],
) -> torch.Tensor:
    if tuple(int(s) for s in tensor.stride()) == target_stride:
        return tensor
    return tpu_torch_compile.force_strides(tensor, list(target_stride), 0)


def _validate_tpu_state_tensor_device(
    state_tensor: Any,
    expected_dev: torch.device,
) -> None:
    if not isinstance(state_tensor, torch.Tensor):
        raise TypeError(
            f"Expected TPU generator state Tensor, got {type(state_tensor)}"
        )
    same_type = (
        state_tensor.device.type in _TPU_DEVICE_TYPES
        and expected_dev.type in _TPU_DEVICE_TYPES
    ) or state_tensor.device.type == expected_dev.type
    if not same_type or (state_tensor.device.index or 0) != (expected_dev.index or 0):
        raise ValueError(
            f"TPU generator state device ({state_tensor.device}) does not match "
            f"while_loop carry device ({expected_dev})."
        )


def _validate_while_loop_inputs(
    carried_inputs: Any,
    additional_inputs: Any,
    stack_output: bool,
    mutated_arg_indices: str | Collection[int | str] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if stack_output:
        raise NotImplementedError(
            "stack_output=True is not supported by TPU while_loop"
        )
    if _parse_mutated_arg_indices(mutated_arg_indices):
        raise RuntimeError(
            "In-place input buffer mutation inside while_loop cond_fn or "
            "body_fn is not supported on TPU; return updated carries instead."
        )
    if not isinstance(carried_inputs, (tuple, list)) or len(carried_inputs) == 0:
        raise (
            TypeError(
                "while_loop carried_inputs must be a tuple/list, "
                f"got {type(carried_inputs)}"
            )
            if not isinstance(carried_inputs, (tuple, list))
            else ValueError("while_loop carried_inputs must be non-empty")
        )
    if not isinstance(additional_inputs, (tuple, list)):
        raise TypeError(
            "while_loop additional_inputs must be a tuple/list, "
            f"got {type(additional_inputs)}"
        )

    carried_list = list(carried_inputs)
    additional_list = list(additional_inputs)
    for name, seq in (
        ("carried_inputs", carried_list),
        ("additional_inputs", additional_list),
    ):
        for idx, item in enumerate(seq):
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    f"while_loop {name}[{idx}] must be a Tensor, got {type(item)}"
                )

    expected_device = carried_list[0].device
    for name, seq in (
        ("carried_inputs", carried_list),
        ("additional_inputs", additional_list),
    ):
        for idx, tensor in enumerate(seq):
            if tensor.device != expected_device:
                raise ValueError(
                    f"All while_loop {name} must reside on {expected_device}, "
                    f"got {tensor.device} at index {idx}"
                )
    return carried_list, additional_list


def _run_callable_checking_no_mutation(
    fn: Any,
    args: Sequence[torch.Tensor],
    *,
    leaf_placeholders: Sequence[torch.Tensor] = (),
    export_mod: Any | None,
    label: str,
) -> Any:
    if isinstance(fn, torch.export.ExportedProgram):
        fn = fn.module()
    _check_fx_graph_no_input_mutation(fn, label)

    call_args = list(args)
    watched = [*call_args, *leaf_placeholders]
    before = [(None if a.is_inference() else a._version, a.data_ptr()) for a in watched]
    raw_out = (
        export_mod.EagerLikeFxInterpreter(fn).run(*call_args)
        if isinstance(fn, torch.fx.GraphModule) and export_mod is not None
        else fn(*call_args)
    )
    for idx, (arg, (ver, ptr)) in enumerate(zip(watched, before, strict=True)):
        cur_ver = None if arg.is_inference() else arg._version
        if (ver is not None and cur_ver != ver) or arg.data_ptr() != ptr:
            arg_idx = idx if idx < len(call_args) else idx - len(call_args)
            raise RuntimeError(
                f"In-place mutation of {label} input argument {arg_idx} is not "
                "supported in TPU while_loop; use functional tensor updates."
            )
    return raw_out


@contextlib.contextmanager
def _forbid_eager_tpu_rng_advance(
    tpu_torch_compile: Any,
    expected_device: torch.device,
) -> Iterator[None]:
    device_idx = expected_device.index or 0
    tpu_device_mod = getattr(torch, "tpu", None)
    default_gen: Any | None = None
    if tpu_device_mod is not None:
        with contextlib.suppress(RuntimeError, IndexError):
            default_gen = tpu_device_mod.default_generators[device_idx]
    if default_gen is None:
        yield
        return

    gen_before = tpu_torch_compile.get_device_state_tensor(default_gen)
    _validate_tpu_state_tensor_device(gen_before, expected_device)
    gen_after: torch.Tensor | None = None
    try:
        yield
        gen_after = tpu_torch_compile.get_device_state_tensor(default_gen)
    finally:
        cur_state = tpu_torch_compile.get_device_state_tensor(default_gen)
        if cur_state.data_ptr() != gen_before.data_ptr():
            tpu_torch_compile.set_device_state_tensor(default_gen, gen_before)

    if gen_after is not None and gen_after.data_ptr() != gen_before.data_ptr():
        raise RuntimeError(
            "Random number generation inside while_loop cond_fn is not "
            "supported on TPU."
        )


def _run_eager_tpu_while_loop(
    cond_fn: Callable[..., Any],
    body_fn: Callable[..., Any],
    carried_list: Sequence[torch.Tensor],
    additional_list: Sequence[torch.Tensor],
    *,
    tpu_torch_compile: Any,
) -> tuple[torch.Tensor, ...]:
    carries = list(carried_list)
    expected_device = carries[0].device
    while True:
        with _forbid_eager_tpu_rng_advance(tpu_torch_compile, expected_device):
            should_continue = _evaluate_eager_cond(
                _run_callable_checking_no_mutation(
                    cond_fn,
                    [*carries, *additional_list],
                    export_mod=None,
                    label="cond_fn",
                )
            )
        if not should_continue:
            break
        carries = _validate_body_outputs(
            _run_callable_checking_no_mutation(
                body_fn,
                [*carries, *additional_list],
                export_mod=None,
                label="body_fn",
            ),
            carried_list,
        )
    return _clone_aliased_eager_outputs(carries, [*carried_list, *additional_list])


def _make_placeholder_like(
    tpu_torch_compile: Any,
    compiler_mod: Any,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if compiler_mod.is_pinned_host_tensor(tensor):
        return tpu_torch_compile.placeholder_like(tensor, True)
    return tpu_torch_compile.placeholder_like(tensor)


@contextlib.contextmanager
def _preserve_outer_export_generator(
    tpu_torch_compile: Any,
    execution_mode_mod: Any,
    device_idx: int,
) -> Iterator[tuple[Any | None, bool]]:
    tpu_device_mod = getattr(torch, "tpu", None)
    outer_gen: Any | None = None
    had_outer_export_gen = False
    if tpu_device_mod is not None:
        try:
            outer_gen = tpu_device_mod.default_generators[device_idx]
            had_outer_export_gen = True
        except (RuntimeError, IndexError):
            with (
                contextlib.suppress(RuntimeError, IndexError),
                execution_mode_mod.set_eager_mode(
                    execution_mode_mod.EagerMode.DEFER_AND_FUSE
                ),
            ):
                outer_gen = tpu_device_mod.default_generators[device_idx]
    try:
        yield outer_gen, had_outer_export_gen
    finally:
        tpu_torch_compile.set_export_mode_default_device_generator(
            outer_gen if had_outer_export_gen else None, device_idx
        )


@contextlib.contextmanager
def _capture_inner_device_assignments(
    tpu_torch_pallas: Any,
) -> Iterator[_CapturedDeviceAssignments]:
    """Thread-safely captures explicit ``device_assignment``s on the current thread."""
    captured = _CapturedDeviceAssignments()
    mod_key = id(tpu_torch_pallas)
    stacks_by_mod = _thread_local_state.da_stacks_by_mod

    with _capture_hook_lock:
        state = _capture_hook_states.get(mod_key)
        if state is None:
            orig_call = tpu_torch_pallas.call_custom_kernel

            @functools.wraps(orig_call)
            def _intercepted_call_custom_kernel(*args: Any, **kwargs: Any) -> Any:
                mod_stack = _thread_local_state.da_stacks_by_mod.get(mod_key)
                if mod_stack:
                    active_captured = mod_stack[-1]
                    raw_da = kwargs.get("device_assignment")
                    if raw_da:
                        normalized = [int(x) for x in raw_da]
                        if active_captured and active_captured[0] != normalized:
                            raise ValueError(
                                "Conflicting device_assignment values across "
                                "custom kernels inside while_loop: "
                                f"{active_captured[0]} vs {normalized}"
                            )
                        if not active_captured:
                            active_captured.append(normalized)
                    else:
                        kernel_name = args[0] if args else kwargs.get("name")
                        if kernel_name != _WHILE_CUSTOM_KERNEL_NAME:
                            active_captured.saw_unassigned_custom_kernel = True
                return orig_call(*args, **kwargs)

            state = _PallasHookState(orig_call, _intercepted_call_custom_kernel, 1)
            _capture_hook_states[mod_key] = state
            tpu_torch_pallas.call_custom_kernel = _intercepted_call_custom_kernel
        else:
            state.active_count += 1

    thread_stack = stacks_by_mod.setdefault(mod_key, [])
    thread_stack.append(captured)
    try:
        yield captured
    finally:
        thread_stack.pop()
        if not thread_stack:
            stacks_by_mod.pop(mod_key, None)
        with _capture_hook_lock:
            cur_state = _capture_hook_states.get(mod_key)
            if cur_state is not None:
                cur_state.active_count -= 1
                if cur_state.active_count <= 0:
                    _capture_hook_states.pop(mod_key, None)
                    if tpu_torch_pallas.call_custom_kernel is cur_state.wrapper_call:
                        tpu_torch_pallas.call_custom_kernel = cur_state.orig_call


def _deduplicate_repeated_outputs(
    tensors: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    deduped: list[torch.Tensor] = []
    seen_ptrs: set[int] = set()
    for tensor in tensors:
        ptr = tensor.data_ptr()
        if ptr in seen_ptrs:
            tensor = tensor.clone()
            ptr = tensor.data_ptr()
        seen_ptrs.add(ptr)
        deduped.append(tensor)
    return deduped


@contextlib.contextmanager
def _subgraph_rng_trace_scope(
    *,
    is_cond: bool,
    tpu_torch_compile: Any,
    expected_device: torch.device,
    outer_gen: Any | None,
    had_outer_export_gen: bool,
) -> Iterator[_SubgraphRngCapture]:
    device_idx = expected_device.index or 0
    sub_gen: Any | None = None
    rng_begin: torch.Tensor | None = None
    if outer_gen is not None:
        sub_gen = tpu_torch_compile.make_export_mode_device_generator(outer_gen)
        rng_begin = tpu_torch_compile.get_device_state_tensor(sub_gen)
        assert rng_begin is not None
        _validate_tpu_state_tensor_device(rng_begin, expected_device)
        tpu_torch_compile.set_export_mode_default_device_generator(sub_gen, device_idx)

    capture = _SubgraphRngCapture(rng_begin=rng_begin)
    try:
        yield capture
    finally:
        if sub_gen is not None:
            tpu_torch_compile.set_export_mode_default_device_generator(
                outer_gen if had_outer_export_gen else None, device_idx
            )

    if rng_begin is not None and sub_gen is not None:
        sub_end = tpu_torch_compile.get_device_state_tensor(sub_gen)
        if sub_end.data_ptr() != rng_begin.data_ptr():
            capture.updates_rng = True
            capture.rng_end = sub_end

    if is_cond and capture.updates_rng:
        raise RuntimeError(
            "Random number generation inside while_loop cond_fn is not supported "
            "on TPU because stablehlo.while condition regions cannot persist RNG state."
        )
    if not is_cond and capture.updates_rng and not had_outer_export_gen:
        raise RuntimeError(
            "while_loop body_fn updated default TPU generator state during FX "
            "graph compilation without an active export-mode generator override."
        )


def _resolve_all_reduce_group_size(
    node: torch.fx.Node,
    schema: torch._C.FunctionSchema,
) -> int | None:
    group: Any = next(
        (
            val
            for arg, val in library_utils.zip_schema(schema, node.args, node.kwargs)
            if arg.name in ("group_name", "group")
        ),
        None,
    )
    if group is None:
        return None
    if isinstance(group, (list, tuple, range)):
        ranks = {int(r) for r in group}
        return len(ranks) if ranks else None
    if not dist.is_available() or not dist.is_initialized():
        return None
    with contextlib.suppress(RuntimeError, ValueError, KeyError):
        if isinstance(group, (str, int)):
            pg = dist.distributed_c10d._resolve_process_group(
                dist.distributed_c10d.GroupName(str(group))
            )
            return len(dist.get_process_group_ranks(pg))
        if isinstance(group, dist.ProcessGroup):
            return len(dist.get_process_group_ranks(group))
    return None


def _inspect_fx_subgraph_collectives(fn: Any) -> tuple[bool, frozenset[int]] | None:
    """Classifies FX collectives as order-sensitive vs full-group ``all_reduce``."""
    if isinstance(fn, torch.export.ExportedProgram):
        fn = fn.module()
    if not isinstance(fn, torch.fx.GraphModule):
        return None

    reduce_sizes: set[int] = set()
    for mod in fn.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op != "call_function" or not isinstance(
                node.target, torch._ops.OpOverload
            ):
                continue
            schema = node.target._schema
            schema_name = schema.name
            if schema_name in _FX_ORDER_SENSITIVE_COLLECTIVE_SCHEMA_NAMES:
                return True, frozenset()
            if schema_name in _FX_ALL_REDUCE_SCHEMA_NAMES:
                gsize = _resolve_all_reduce_group_size(node, schema)
                if gsize is None:
                    return True, frozenset()
                reduce_sizes.add(gsize)
                continue
            if (
                schema_name.startswith("_c10d_functional::")
                and schema_name not in _FX_BENIGN_C10D_SCHEMA_NAMES
            ):
                return True, frozenset()
    return False, frozenset(reduce_sizes)


def _merge_region_collective_info(
    trace: _SubgraphTraceResult,
    *,
    parsed_order_sens: bool,
    parsed_red_sizes: frozenset[int],
) -> tuple[bool, frozenset[int]]:
    if trace.fx_collectives is None:
        return parsed_order_sens, parsed_red_sizes
    fx_order_sens, fx_red_sizes = trace.fx_collectives
    if trace.saw_unassigned_custom_kernel:
        return fx_order_sens or parsed_order_sens, fx_red_sizes | parsed_red_sizes
    return fx_order_sens, fx_red_sizes


@dataclasses.dataclass(frozen=True)
class _SubgraphTracer:
    trace_carries: Sequence[torch.Tensor]
    orig_carry_strides: Sequence[tuple[int, ...]]
    norm_additional_list: Sequence[torch.Tensor]
    orig_add_strides: Sequence[tuple[int, ...]]
    tpu_torch_compile: Any
    tpu_torch_pallas: Any
    compiler_mod: Any
    export_mod: Any
    expected_device: torch.device
    outer_gen: Any | None
    had_outer_export_gen: bool

    def trace_cond(self, cond_fn: Callable[..., Any]) -> _SubgraphTraceResult:
        return self._trace(cond_fn, is_cond=True, label="cond_fn")

    def trace_body(self, body_fn: Callable[..., Any]) -> _SubgraphTraceResult:
        return self._trace(body_fn, is_cond=False, label="body_fn")

    def _trace(
        self, fn: Callable[..., Any], *, is_cond: bool, label: str
    ) -> _SubgraphTraceResult:
        carry_phs = [
            _make_placeholder_like(self.tpu_torch_compile, self.compiler_mod, t)
            for t in self.trace_carries
        ]
        add_phs = [
            _make_placeholder_like(self.tpu_torch_compile, self.compiler_mod, t)
            for t in self.norm_additional_list
        ]

        with (
            _capture_inner_device_assignments(self.tpu_torch_pallas) as captured_das,
            _subgraph_rng_trace_scope(
                is_cond=is_cond,
                tpu_torch_compile=self.tpu_torch_compile,
                expected_device=self.expected_device,
                outer_gen=self.outer_gen,
                had_outer_export_gen=self.had_outer_export_gen,
            ) as rng_capture,
        ):
            call_carry_args = [
                _restore_tensor_strides(self.tpu_torch_compile, ph, stride)
                for ph, stride in zip(carry_phs, self.orig_carry_strides, strict=True)
            ]
            call_add_args = [
                _restore_tensor_strides(self.tpu_torch_compile, ph, stride)
                for ph, stride in zip(add_phs, self.orig_add_strides, strict=True)
            ]
            raw_out = _run_callable_checking_no_mutation(
                fn,
                [*call_carry_args, *call_add_args],
                leaf_placeholders=[*carry_phs, *add_phs],
                export_mod=self.export_mod,
                label=label,
            )
            validated_outs = (
                [_validate_cond_output(raw_out)]
                if is_cond
                else _deduplicate_repeated_outputs(
                    [
                        _materialize_zero_offset_contiguous(o)
                        for o in _validate_body_outputs(raw_out, carry_phs)
                    ]
                )
            )

        if not is_cond and rng_capture.updates_rng:
            assert rng_capture.rng_begin is not None and rng_capture.rng_end is not None
            arg_tensors = [*carry_phs, rng_capture.rng_begin, *add_phs]
            res_tensors = [*validated_outs, rng_capture.rng_end]
        else:
            arg_tensors = [*carry_phs, *add_phs]
            res_tensors = validated_outs

        mlir_mod = self.tpu_torch_compile.build_mlir(
            result_tensors=res_tensors,
            argument_tensors=arg_tensors,
            use_stablehlo_bounds=False,
        )
        mlir_text = self.tpu_torch_compile.serialize_mlir_text(
            mlir_mod, enable_debug_info=True
        )
        return _SubgraphTraceResult(
            mlir_text=mlir_text,
            updates_rng=rng_capture.updates_rng,
            captured_da=captured_das[0] if captured_das else None,
            saw_unassigned_custom_kernel=captured_das.saw_unassigned_custom_kernel,
            fx_collectives=_inspect_fx_subgraph_collectives(fn),
        )


def _require_default_device_assignment(
    device_mesh_mod: Any, region_label: str
) -> list[int]:
    try:
        default_da = [int(x) for x in device_mesh_mod.get_device_assignment(None)]
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unable to resolve default device_assignment for multi-device ops "
            f"inside while_loop {region_label}: {exc}"
        ) from exc
    if not default_da:
        raise ValueError(
            f"Default device_assignment is empty for multi-device ops inside "
            f"while_loop {region_label}."
        )
    return default_da


def _region_da_requirement(
    *,
    region_label: str,
    captured_da: list[int] | None,
    has_multi_device_ops: bool,
    device_mesh_mod: Any,
    allow_permuted_default_da: bool = False,
) -> _RegionDeviceAssignmentReq:
    if captured_da is not None:
        if has_multi_device_ops:
            norm_cap = list(captured_da)
            norm_def = _require_default_device_assignment(device_mesh_mod, region_label)
            if sorted(norm_cap) != sorted(norm_def) or (
                not allow_permuted_default_da and norm_cap != norm_def
            ):
                raise ValueError(
                    f"Mismatched device_assignment inside while_loop "
                    f"{region_label}: custom kernel ({norm_cap}) vs "
                    f"implicit multi-device ops ({norm_def})."
                )
        return _RegionDeviceAssignmentReq(
            da=captured_da,
            has_requirement=True,
            desc=f"{region_label} custom kernel ({captured_da})",
        )
    if has_multi_device_ops:
        default_da = _require_default_device_assignment(device_mesh_mod, region_label)
        return _RegionDeviceAssignmentReq(
            da=default_da,
            has_requirement=True,
            desc=f"{region_label} implicit multi-device ops ({default_da})",
        )
    return _RegionDeviceAssignmentReq(da=None, has_requirement=False, desc=region_label)


def _resolve_while_device_assignment(
    *,
    cond_captured_da: list[int] | None,
    body_captured_da: list[int] | None,
    cond_has_multi_device_ops: bool,
    body_has_multi_device_ops: bool,
    device_mesh_mod: Any,
    cond_allow_permuted_default_da: bool = False,
    body_allow_permuted_default_da: bool = False,
) -> list[int]:
    cond_req = _region_da_requirement(
        region_label="cond_fn",
        captured_da=cond_captured_da,
        has_multi_device_ops=cond_has_multi_device_ops,
        device_mesh_mod=device_mesh_mod,
        allow_permuted_default_da=cond_allow_permuted_default_da,
    )
    body_req = _region_da_requirement(
        region_label="body_fn",
        captured_da=body_captured_da,
        has_multi_device_ops=body_has_multi_device_ops,
        device_mesh_mod=device_mesh_mod,
        allow_permuted_default_da=body_allow_permuted_default_da,
    )
    if cond_req.has_requirement and body_req.has_requirement:
        if cond_req.da == body_req.da:
            return cond_req.da or []
        raise ValueError(
            f"Mismatched device_assignment between while_loop "
            f"{cond_req.desc} and {body_req.desc}."
        )
    if cond_req.has_requirement:
        return cond_req.da or []
    if body_req.has_requirement:
        return body_req.da or []
    return []


def _handle_while_loop_impl(
    cond_fn: Callable[..., Any],
    body_fn: Callable[..., Any],
    carried_inputs: tuple[Any, ...] | list[Any],
    additional_inputs: tuple[Any, ...] | list[Any] = (),
    stack_output: bool = False,
    mutated_arg_indices: str | Collection[int | str] = "",
) -> tuple[torch.Tensor, ...]:
    """TPU dispatch implementation for ``torch.ops.higher_order.while_loop``."""
    orig_carried_list, orig_additional_list = _validate_while_loop_inputs(
        carried_inputs, additional_inputs, stack_output, mutated_arg_indices
    )

    from torch_tpu._internal import execution_mode
    from torch_tpu._internal.compile import compiler, tpu_torch_compile
    from torch_tpu._internal.distributed import device_mesh
    from torch_tpu._internal.export import export
    from torch_tpu._internal.pallas import tpu_torch_pallas

    _check_disjoint_concrete_storages(orig_carried_list)

    if execution_mode.eager_mode != execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH:
        return _run_eager_tpu_while_loop(
            cond_fn,
            body_fn,
            orig_carried_list,
            orig_additional_list,
            tpu_torch_compile=tpu_torch_compile,
        )

    expected_device = orig_carried_list[0].device
    device_idx = expected_device.index or 0
    orig_carry_strides = [tuple(int(s) for s in t.stride()) for t in orig_carried_list]
    orig_add_strides = [tuple(int(s) for s in t.stride()) for t in orig_additional_list]
    trace_carries = [_materialize_zero_offset_contiguous(t) for t in orig_carried_list]
    norm_additional_list = [
        _materialize_zero_offset_contiguous(t) for t in orig_additional_list
    ]

    with _preserve_outer_export_generator(
        tpu_torch_compile, execution_mode, device_idx
    ) as (outer_gen, had_outer_export_gen):
        outer_rng_before = (
            tpu_torch_compile.get_device_state_tensor(outer_gen)
            if outer_gen is not None
            else None
        )
        if outer_rng_before is not None:
            _validate_tpu_state_tensor_device(outer_rng_before, expected_device)
        tracer = _SubgraphTracer(
            trace_carries=trace_carries,
            orig_carry_strides=orig_carry_strides,
            norm_additional_list=norm_additional_list,
            orig_add_strides=orig_add_strides,
            tpu_torch_compile=tpu_torch_compile,
            tpu_torch_pallas=tpu_torch_pallas,
            compiler_mod=compiler,
            export_mod=export,
            expected_device=expected_device,
            outer_gen=outer_gen,
            had_outer_export_gen=had_outer_export_gen,
        )
        cond_trace = tracer.trace_cond(cond_fn)
        body_trace = tracer.trace_body(body_fn)

    with _parse_while_subgraphs(
        cond_trace.mlir_text,
        body_trace.mlir_text,
        num_carries=len(orig_carried_list),
    ) as parsed:
        for donor_idx in parsed.raw_donated_carry_indices:
            orig_carry = orig_carried_list[donor_idx]
            if (
                not orig_carry.is_contiguous()
                or _safe_storage_offset(orig_carry) != 0
                or _has_partial_storage(orig_carry)
            ):
                raise ValueError(
                    f"Donated while_loop carried_inputs[{donor_idx}] must own its "
                    "full storage contiguously with storage_offset == 0; pass "
                    ".contiguous().clone() before the loop."
                )

        raw_donor_set = set(parsed.raw_donated_carry_indices)
        exec_carried_list = [
            orig_c
            if idx in raw_donor_set
            else _materialize_zero_offset_contiguous(orig_c)
            for idx, orig_c in enumerate(orig_carried_list)
        ]
        extra_rng = (
            [outer_rng_before]
            if body_trace.updates_rng and outer_rng_before is not None
            else []
        )
        allowed_donors = _compute_allowed_donor_indices(
            len(exec_carried_list),
            [*orig_carried_list, *orig_additional_list, *extra_rng],
            [*exec_carried_list, *norm_additional_list, *extra_rng],
        )
        cond_order_sens, cond_red_sizes = _merge_region_collective_info(
            cond_trace,
            parsed_order_sens=parsed.cond_has_order_sensitive_ops,
            parsed_red_sizes=parsed.cond_world_reduce_sizes,
        )
        body_order_sens, body_red_sizes = _merge_region_collective_info(
            body_trace,
            parsed_order_sens=parsed.body_has_order_sensitive_ops,
            parsed_red_sizes=parsed.body_world_reduce_sizes,
        )
        resolved_da = _resolve_while_device_assignment(
            cond_captured_da=cond_trace.captured_da,
            body_captured_da=body_trace.captured_da,
            cond_has_multi_device_ops=parsed.cond_has_multi_device_ops,
            body_has_multi_device_ops=parsed.body_has_multi_device_ops,
            device_mesh_mod=device_mesh,
            cond_allow_permuted_default_da=(
                cond_trace.captured_da is not None
                and not cond_order_sens
                and all(gs == len(cond_trace.captured_da) for gs in cond_red_sizes)
            ),
            body_allow_permuted_default_da=(
                body_trace.captured_da is not None
                and not body_order_sens
                and all(gs == len(body_trace.captured_da) for gs in body_red_sizes)
            ),
        )
        artifact = _assemble_while_loop_mlir(
            parsed,
            num_carries=len(exec_carried_list),
            num_additional_inputs=len(norm_additional_list),
            has_rng_state=body_trace.updates_rng,
            num_partitions=max(1, len(resolved_da)),
            allowed_donor_indices=allowed_donors,
        )

    if not tpu_torch_pallas.lookup_custom_kernel(
        _WHILE_CUSTOM_KERNEL_NAME, artifact.kernel_key
    ):
        tpu_torch_pallas.register_custom_kernel(
            _WHILE_CUSTOM_KERNEL_NAME,
            artifact.kernel_key,
            serialized_mlir_module=artifact.normalized_mlir_text,
        )

    kernel_inputs = list(exec_carried_list)
    output_shapes = [
        _make_placeholder_like(tpu_torch_compile, compiler, t)
        for t in exec_carried_list
    ]
    if body_trace.updates_rng:
        assert outer_gen is not None and outer_rng_before is not None
        kernel_inputs.append(outer_rng_before)
        output_shapes.append(
            _make_placeholder_like(tpu_torch_compile, compiler, outer_rng_before)
        )
    kernel_inputs.extend(norm_additional_list)

    raw_results = tpu_torch_pallas.call_custom_kernel(
        _WHILE_CUSTOM_KERNEL_NAME,
        artifact.kernel_key,
        inputs=kernel_inputs,
        output_shapes=output_shapes,
        donate_argnums=list(artifact.donated_carry_indices),
        device_assignment=resolved_da,
    )

    if body_trace.updates_rng:
        assert outer_gen is not None
        tpu_torch_compile.set_device_state_tensor(outer_gen, raw_results[-1])
        user_results = raw_results[:-1]
    else:
        user_results = raw_results

    return tuple(
        _restore_tensor_strides(tpu_torch_compile, res, orig_stride)
        for res, orig_stride in zip(user_results, orig_carry_strides, strict=True)
    )


def _patch_clone_mutated_returned_placeholders(cmrp_mod: Any) -> None:
    """Exempts pure ``while_loop_op`` users from placeholder cloning in 0.2.0."""
    orig_is_compute = cmrp_mod._is_returned_compute_placeholder
    if getattr(orig_is_compute, "_vllm_while_loop_patched", False):
        return

    @functools.wraps(orig_is_compute)
    def _patched_check(node: torch.fx.Node, returned_nodes: set[torch.fx.Node]) -> bool:
        if not orig_is_compute(node, returned_nodes):
            return False
        non_output_users = [u for u in node.users if u.op != "output"]
        return not (
            non_output_users
            and all(
                u.op == "call_function"
                and u.target is while_loop_op
                and not _parse_mutated_arg_indices(u.kwargs.get("mutated_arg_indices"))
                for u in non_output_users
            )
        )

    cast(Any, _patched_check)._vllm_while_loop_patched = True
    cmrp_mod._is_returned_compute_placeholder = _patched_check


def _node_is_nonempty_tensor(node: torch.fx.Node) -> bool:
    val = node.meta.get("val")
    return (
        bool(val.numel() > 0 and val.element_size() > 0)
        if isinstance(val, torch.Tensor)
        else True
    )


def _make_deferred_value_error_gm(
    graph_module: torch.fx.GraphModule,
    message: str,
) -> torch.fx.GraphModule:
    err_graph = torch.fx.Graph()
    phs = [
        err_graph.placeholder(ph.name)
        for ph in graph_module.graph.find_nodes(op="placeholder")
    ]
    err_graph.call_module("_raise_carry_overlap", args=tuple(phs[:1]))
    err_graph.output(())
    return torch.fx.GraphModule(
        {"_raise_carry_overlap": _DeferredCarryOverlapErrorModule(message)}, err_graph
    )


@dataclasses.dataclass(frozen=True)
class _CarryValidationPlan:
    uncond_err: str | None
    uncond_ph_groups: tuple[tuple[int, ...], ...]


def _resolve_child_gm(
    gm: torch.fx.GraphModule,
    arg: Any,
) -> torch.fx.GraphModule | None:
    if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
        cur: Any = gm
        for atom in str(arg.target).split("."):
            cur = getattr(cur, atom)
        if isinstance(cur, torch.fx.GraphModule):
            return cur
    return None


def _check_and_erase_validate_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    storage_roots: dict[torch.fx.Node, set[Any]],
    top_ph_to_index: Mapping[torch.fx.Node, int],
    uncond_ph_groups: list[tuple[int, ...]],
) -> str | None:
    raw_carries = node.args[0] if node.args else ()
    carry_nodes = [
        c
        for c in (raw_carries if isinstance(raw_carries, (tuple, list)) else ())
        if isinstance(c, torch.fx.Node)
    ]
    for i, na in enumerate(carry_nodes):
        if not _node_is_nonempty_tensor(na):
            continue
        roots_a = storage_roots.get(na, {na})
        for j in range(i + 1, len(carry_nodes)):
            nb = carry_nodes[j]
            if _node_is_nonempty_tensor(nb) and (roots_a & storage_roots.get(nb, {nb})):
                return _format_carry_overlap_message(i, j)

    group_indices: list[int] = []
    for c_node in carry_nodes:
        if not _node_is_nonempty_tensor(c_node):
            continue
        for root in storage_roots.get(c_node, ()):
            if (
                isinstance(root, torch.fx.Node)
                and root in top_ph_to_index
                and top_ph_to_index[root] not in group_indices
            ):
                group_indices.append(top_ph_to_index[root])
    if len(group_indices) >= 2 and tuple(group_indices) not in uncond_ph_groups:
        uncond_ph_groups.append(tuple(group_indices))

    for user in list(node.users):
        if (
            user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) == 2
            and isinstance(user.args[1], int)
        ):
            src_node = carry_nodes[user.args[1]]
            storage_roots[user] = set(storage_roots.get(src_node, {src_node}))
            user.replace_all_uses_with(src_node)
            gm.graph.erase_node(user)
    gm.graph.erase_node(node)
    return None


def _record_fx_node_storage_roots(
    node: torch.fx.Node,
    storage_roots: dict[torch.fx.Node, set[Any]],
    getitem_roots: dict[tuple[torch.fx.Node, Any], object],
) -> None:
    target = node.target
    if target is torch.ops.aten._unsafe_view.default:
        arg0 = node.args[0] if node.args else None
        storage_roots[node] = (
            set(storage_roots.get(arg0, {arg0}))
            if isinstance(arg0, torch.fx.Node)
            else {node}
        )
        return
    if target is operator.getitem:
        arg0 = node.args[0] if node.args else None
        idx = node.args[1] if len(node.args) > 1 else None
        if isinstance(arg0, torch.fx.Node):
            if _getitem_aliases_source(arg0, idx):
                storage_roots[node] = set(storage_roots.get(arg0, {arg0}))
            else:
                try:
                    hash((arg0, idx))
                except TypeError:
                    storage_roots[node] = {object()}
                else:
                    storage_roots[node] = {
                        getitem_roots.setdefault((arg0, idx), object())
                    }
        else:
            storage_roots[node] = {node}
        return
    if isinstance(target, torch._ops.OpOverload):
        schema = target._schema
        if any(r.alias_info is not None for r in schema.returns):
            aliased_vals = [
                val
                for arg, val in library_utils.zip_schema(schema, node.args, node.kwargs)
                if arg.alias_info is not None
            ]
            storage_roots[node] = set().union(
                *(storage_roots.get(n, {n}) for n in _fx_nodes_in_value(aliased_vals))
            )
            return
    storage_roots[node] = {node}


def _rewire_while_carry_checks_in_gm(
    gm: torch.fx.GraphModule,
    ph_roots_by_idx: Mapping[int, set[Any]],
    top_ph_to_index: Mapping[torch.fx.Node, int],
    uncond_ph_groups: list[tuple[int, ...]],
    visited: set[int],
) -> str | None:
    gm_id = id(gm)
    if gm_id in visited:
        return None
    visited.add(gm_id)

    validate_op = torch.ops.vllm_torchtpu._validate_while_carries.default
    cond_hop = getattr(torch.ops.higher_order, "cond", None)
    storage_roots: dict[torch.fx.Node, set[Any]] = {
        ph: set(ph_roots_by_idx.get(idx, (ph,)))
        for idx, ph in enumerate(gm.graph.find_nodes(op="placeholder"))
    }
    getitem_roots: dict[tuple[torch.fx.Node, Any], object] = {}
    mutated = False

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        target = node.target
        if target is while_loop_op and len(node.args) >= 4:
            cond_arg, body_arg, carried_arg, additional_arg = node.args[:4]
            carried_list = (
                list(carried_arg) if isinstance(carried_arg, (tuple, list)) else []
            )
            add_list = (
                list(additional_arg)
                if isinstance(additional_arg, (tuple, list))
                else []
            )
            sub_ph_roots = {
                idx: set(storage_roots.get(inp, {inp}))
                for idx, inp in enumerate([*carried_list, *add_list])
                if isinstance(inp, torch.fx.Node)
            }
            for sub_arg in (cond_arg, body_arg):
                sub_gm = _resolve_child_gm(gm, sub_arg)
                if sub_gm is not None:
                    err = _rewire_while_carry_checks_in_gm(
                        sub_gm, sub_ph_roots, top_ph_to_index, uncond_ph_groups, visited
                    )
                    if err is not None:
                        return err
            storage_roots[node] = {node}
            continue

        if cond_hop is not None and target is cond_hop and len(node.args) >= 4:
            _, true_arg, false_arg, operands_arg = node.args[:4]
            op_list = (
                list(operands_arg) if isinstance(operands_arg, (tuple, list)) else []
            )
            sub_ph_roots = {
                idx: set(storage_roots.get(inp, {inp}))
                for idx, inp in enumerate(op_list)
                if isinstance(inp, torch.fx.Node)
            }
            for branch_arg in (true_arg, false_arg):
                branch_gm = _resolve_child_gm(gm, branch_arg)
                if branch_gm is not None:
                    err = _rewire_while_carry_checks_in_gm(
                        branch_gm,
                        sub_ph_roots,
                        top_ph_to_index,
                        uncond_ph_groups,
                        visited,
                    )
                    if err is not None:
                        return err
            storage_roots[node] = {node}
            continue

        if target is validate_op:
            err = _check_and_erase_validate_node(
                gm, node, storage_roots, top_ph_to_index, uncond_ph_groups
            )
            if err is not None:
                return err
            mutated = True
            continue

        _record_fx_node_storage_roots(node, storage_roots, getitem_roots)

    if mutated:
        gm.graph.lint()
        gm.recompile()
    return None


def _extract_and_rewire_while_carry_checks(
    graph_module: torch.fx.GraphModule,
) -> _CarryValidationPlan:
    meta = graph_module.meta
    cached = meta.get("_vllm_while_carry_plan")
    if isinstance(cached, _CarryValidationPlan):
        return cached

    validate_op = torch.ops.vllm_torchtpu._validate_while_carries.default
    all_gms = [m for m in graph_module.modules() if isinstance(m, torch.fx.GraphModule)]
    if not any(
        n.op == "call_function" and n.target is validate_op
        for m in all_gms
        for n in m.graph.nodes
    ):
        return _CarryValidationPlan(None, ())

    top_ph_to_index = {
        ph: idx
        for idx, ph in enumerate(graph_module.graph.find_nodes(op="placeholder"))
    }
    uncond_ph_groups: list[tuple[int, ...]] = []
    visited: set[int] = set()

    err = _rewire_while_carry_checks_in_gm(
        graph_module, {}, top_ph_to_index, uncond_ph_groups, visited
    )
    if err is None:
        for sub_gm in all_gms:
            err = _rewire_while_carry_checks_in_gm(
                sub_gm, {}, top_ph_to_index, uncond_ph_groups, visited
            )
            if err is not None:
                break

    plan = (
        _CarryValidationPlan(err, ())
        if err is not None
        else _CarryValidationPlan(None, tuple(uncond_ph_groups))
    )
    meta["_vllm_while_carry_plan"] = plan
    return plan


def _patch_split_compiler_while_carry_validation(split_compiler_mod: Any) -> None:
    """Patches ``SplitCompiler._compile_graph`` to enforce carry disjointness."""
    split_cls = split_compiler_mod.SplitCompiler
    orig_compile_graph = split_cls._compile_graph
    if getattr(orig_compile_graph, "_vllm_while_loop_patched", False):
        return

    @functools.wraps(orig_compile_graph)
    def _patched_compile_graph(
        self: Any,
        graph_module: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        plan = _extract_and_rewire_while_carry_checks(graph_module)
        if plan.uncond_err is not None:
            return split_compiler_mod._SplitCompiledExecutable(
                target=_make_deferred_value_error_gm(graph_module, plan.uncond_err)
            )
        try:
            exe = orig_compile_graph(
                self, graph_module, example_inputs, *args, **kwargs
            )
        except _WhileCarryStorageOverlapError as exc:
            return split_compiler_mod._SplitCompiledExecutable(
                target=_make_deferred_value_error_gm(graph_module, str(exc))
            )

        if plan.uncond_ph_groups:
            split_gm = exe._split_gm
            split_phs = list(split_gm.graph.find_nodes(op="placeholder"))
            orig_phs = list(graph_module.graph.find_nodes(op="placeholder"))
            if len(split_phs) != len(orig_phs):
                raise RuntimeError(
                    "SplitCompiler changed top-level placeholder count from "
                    f"{len(orig_phs)} to {len(split_phs)}; cannot bind carry checks."
                )
            first_exec = next(
                (n for n in split_gm.graph.nodes if n.op != "placeholder"), None
            )
            if first_exec is not None:
                for group in plan.uncond_ph_groups:
                    with split_gm.graph.inserting_before(first_exec):
                        split_gm.graph.call_function(
                            torch.ops.vllm_torchtpu._check_while_carry_args.default,
                            args=([split_phs[idx] for idx in group],),
                        )
                split_gm.graph.lint()
                split_gm.recompile()
        return exe

    cast(Any, _patched_compile_graph)._vllm_while_loop_patched = True
    split_cls._compile_graph = _patched_compile_graph


def install_while_loop_patch() -> None:
    """Installs the TPU ``while_loop`` compiler patch for ``torch-tpu==0.2.0``."""
    try:
        import torch_tpu  # noqa: F401
    except ImportError:
        logger.debug("Skipping install_while_loop_patch: torch_tpu is not installed.")
        return

    from torch_tpu._internal.compile import split_compiler
    from torch_tpu._internal.compile.fx_passes import (
        clone_mutated_returned_placeholders as cmrp_mod,
    )

    for key in (DispatchKey.PrivateUse1, DispatchKey.XLA):
        if key not in while_loop_op.py_kernels:
            while_loop_op.py_impl(key)(_handle_while_loop_impl)
        else:
            logger.info(
                "Skipping while_loop_op patch for %s: native kernel already "
                "registered.",
                key,
            )

    _patch_clone_mutated_returned_placeholders(cmrp_mod)
    _patch_split_compiler_while_carry_validation(split_compiler)
