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
"""Typed StableHLO and Shardy MLIR AST assembly for TPU ``while_loop``."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import Any

from jax._src.interpreters.mlir import make_ir_context
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func, hlo, sdy

_MANUAL_MESH_AXIS = "__d"
_SHARDING_OP_ATTRS = frozenset({"mhlo.sharding", "sdy.sharding", "sdy.sharding_rule"})
_SHARDING_ARG_RES_ATTRS = frozenset({"mhlo.sharding", "sdy.sharding"})
_MULTI_DEVICE_OP_NAMES = frozenset(
    {
        "all_gather",
        "all_reduce",
        "all_to_all",
        "collective_broadcast",
        "collective_permute",
        "partition_id",
        "ragged_all_to_all",
        "recv",
        "reduce_scatter",
        "replica_id",
        "send",
    }
)


@dataclasses.dataclass(frozen=True)
class _WhileLoopMlirArtifact:
    mlir_text: str
    normalized_mlir_text: str
    kernel_key: str
    donated_carry_indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _ParsedWhileSubgraphs:
    cond_mod: Any
    cond_main: Any
    cond_helpers: list[Any]
    body_mod: Any
    body_main: Any
    body_helpers: list[Any]
    raw_donated_carry_indices: tuple[int, ...]
    cond_has_multi_device_ops: bool
    body_has_multi_device_ops: bool
    cond_has_order_sensitive_ops: bool = False
    body_has_order_sensitive_ops: bool = False
    cond_world_reduce_sizes: frozenset[int] = frozenset()
    body_world_reduce_sizes: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True)
class _ValidatedWhileSignatures:
    cond_args: list[Any]
    body_user_carry_args: list[Any]
    rng_arg_in_body: Any | None
    body_add_args: list[Any]
    effective_carry_types: list[Any]
    wrapper_arg_types: list[Any]


def _is_i1_scalar_tensor_type(mlir_type: Any) -> bool:
    return (
        isinstance(mlir_type, ir.RankedTensorType)
        and mlir_type.rank == 0
        and isinstance(mlir_type.element_type, ir.IntegerType)
        and mlir_type.element_type.width == 1
    )


def _is_rng_state_tensor_type(mlir_type: Any) -> bool:
    return (
        isinstance(mlir_type, ir.RankedTensorType)
        and mlir_type.rank == 1
        and list(mlir_type.shape) == [2]
        and isinstance(mlir_type.element_type, ir.IntegerType)
        and mlir_type.element_type.width == 64
        and mlir_type.element_type.is_unsigned
    )


def _extract_donated_carry_indices(body_main: Any, num_carries: int) -> tuple[int, ...]:
    arg_attrs = body_main.arg_attrs
    if arg_attrs is None:
        return ()
    donated: list[int] = []
    for idx in range(min(num_carries, len(arg_attrs))):
        for named_attr in ir.DictAttr(arg_attrs[idx]):
            if (
                named_attr.name == "jax.buffer_donor"
                and isinstance(named_attr.attr, ir.BoolAttr)
                and named_attr.attr.value
            ) or (
                named_attr.name == "tf.aliasing_output"
                and isinstance(named_attr.attr, ir.IntegerAttr)
                and int(named_attr.attr.value) == idx
            ):
                donated.append(idx)
                break
    return tuple(donated)


def _op_is_multi_device_custom_call(raw_op: Any, short_name: str) -> bool:
    if short_name != "custom_call" or "call_target_name" not in raw_op.attributes:
        return False
    target_attr = raw_op.attributes["call_target_name"]
    return (
        isinstance(target_attr, ir.StringAttr)
        and target_attr.value.rsplit(".", 1)[-1] in _MULTI_DEVICE_OP_NAMES
    )


def _region_has_multi_device_ops(region: Any) -> bool:
    for block in region.blocks:
        for op in block.operations:
            raw_op = op.operation
            short_name = raw_op.name.rsplit(".", 1)[-1]
            if short_name in _MULTI_DEVICE_OP_NAMES or _op_is_multi_device_custom_call(
                raw_op, short_name
            ):
                return True
            if any(_region_has_multi_device_ops(r) for r in raw_op.regions):
                return True
    return False


def _single_group_world_all_reduce_size(raw_op: Any) -> int | None:
    """Returns ``N`` when ``raw_op`` is a single-group ``[0..N-1]`` ``all_reduce``."""
    if (
        raw_op.name.rsplit(".", 1)[-1] != "all_reduce"
        or "replica_groups" not in raw_op.attributes
        or any(_region_has_multi_device_ops(r) for r in raw_op.regions)
    ):
        return None
    rg_attr = raw_op.attributes["replica_groups"]
    if (
        not isinstance(rg_attr.type, ir.RankedTensorType)
        or rg_attr.type.rank != 2
        or int(rg_attr.type.shape[0]) != 1
    ):
        return None
    group_size = int(rg_attr.type.shape[1])
    if group_size < 1:
        return None
    try:
        vals = [int(v) for v in ir.DenseIntElementsAttr(rg_attr)]
    except (TypeError, ValueError):
        return None
    return group_size if sorted(vals) == list(range(group_size)) else None


def _inspect_main_implicit_multi_device_ops(region: Any) -> tuple[bool, frozenset[int]]:
    """Inspects ``@main`` for order-sensitive ops and world ``all_reduce`` sizes."""
    has_order_sensitive = False
    reduce_sizes: set[int] = set()

    def _walk(reg: Any) -> None:
        nonlocal has_order_sensitive
        for block in reg.blocks:
            for op in block.operations:
                raw_op = op.operation
                short_name = raw_op.name.rsplit(".", 1)[-1]
                if short_name in _MULTI_DEVICE_OP_NAMES:
                    world_sz = _single_group_world_all_reduce_size(raw_op)
                    if world_sz is not None:
                        reduce_sizes.add(world_sz)
                        continue
                    has_order_sensitive = True
                    return
                if _op_is_multi_device_custom_call(raw_op, short_name):
                    has_order_sensitive = True
                    return
                for child_region in raw_op.regions:
                    _walk(child_region)
                    if has_order_sensitive:
                        return

    _walk(region)
    return has_order_sensitive, frozenset(reduce_sizes)


def _strip_sharding_from_dict_array(
    array_attr: Any, names_to_strip: frozenset[str]
) -> Any:
    cleaned_dicts = []
    changed = False
    for elem in array_attr:
        kept = {}
        for named_attr in ir.DictAttr(elem):
            if named_attr.name in names_to_strip:
                changed = True
            else:
                kept[named_attr.name] = named_attr.attr
        cleaned_dicts.append(ir.DictAttr.get(kept))
    return ir.ArrayAttr.get(cleaned_dicts) if changed else array_attr


def _unwrap_manual_computation_op(raw_op: Any) -> None:
    """Inlines a replicated inner ``sdy.manual_computation`` into its parent."""
    inner_block = raw_op.regions[0].blocks[0]
    inner_ops = list(inner_block.operations)
    if not inner_ops or inner_ops[-1].operation.name != "sdy.return":
        raise ValueError("Expected inner sdy.manual_computation to end with sdy.return")
    term_op = inner_ops[-1].operation

    for idx, (block_arg, operand) in enumerate(
        zip(inner_block.arguments, raw_op.operands, strict=True)
    ):
        if block_arg.type != operand.type:
            raise ValueError(
                f"Cannot unwrap sdy.manual_computation with shape-changing "
                f"input {idx}: outer {operand.type} vs inner {block_arg.type}"
            )
    for idx, (res, ret_val) in enumerate(
        zip(raw_op.results, term_op.operands, strict=True)
    ):
        if res.type != ret_val.type:
            raise ValueError(
                f"Cannot unwrap sdy.manual_computation with shape-changing "
                f"result {idx}: outer {res.type} vs inner {ret_val.type}"
            )

    for attr_name in ("in_shardings", "out_shardings"):
        if attr_name not in raw_op.attributes:
            continue
        per_val = sdy.TensorShardingPerValueAttr(raw_op.attributes[attr_name])
        for sh_idx, raw_sh in enumerate(per_val.shardings):
            tensor_sh = sdy.TensorShardingAttr(raw_sh)
            if len(tensor_sh.unreduced_axes) > 0 or any(
                len(sdy.DimensionShardingAttr(d).axes) > 0
                for d in tensor_sh.dimension_shardings
            ):
                raise ValueError(
                    f"Cannot unwrap sdy.manual_computation with non-replicated "
                    f"sharding in {attr_name}[{sh_idx}]: {tensor_sh}"
                )

    for block_arg, operand in zip(inner_block.arguments, raw_op.operands, strict=True):
        block_arg.replace_all_uses_with(operand)
    for inner_op in inner_ops[:-1]:
        inner_op.operation.move_before(raw_op)
    for res, ret_val in zip(raw_op.results, term_op.operands, strict=True):
        res.replace_all_uses_with(ret_val)
    raw_op.erase()


def _strip_sharding_annotations_in_region(region: Any) -> None:
    for block in region.blocks:
        for op in list(block.operations):
            raw_op = op.operation
            if raw_op.name == "sdy.mesh":
                raw_op.erase()
                continue
            if raw_op.name in ("sdy.sharding_constraint", "sdy.reshard"):
                raw_op.results[0].replace_all_uses_with(raw_op.operands[0])
                raw_op.erase()
                continue
            for attr_name in _SHARDING_OP_ATTRS:
                if attr_name in raw_op.attributes:
                    del raw_op.attributes[attr_name]
            if raw_op.name == "func.func":
                fn_view = raw_op.opview
                if fn_view.arg_attrs is not None:
                    fn_view.arg_attrs = _strip_sharding_from_dict_array(
                        fn_view.arg_attrs, _SHARDING_ARG_RES_ATTRS
                    )
                if fn_view.res_attrs is not None:
                    fn_view.res_attrs = _strip_sharding_from_dict_array(
                        fn_view.res_attrs, _SHARDING_ARG_RES_ATTRS
                    )
            for child_region in raw_op.regions:
                _strip_sharding_annotations_in_region(child_region)
            if raw_op.name == "sdy.manual_computation":
                _unwrap_manual_computation_op(raw_op)


def _find_main_and_helpers(module: Any, label: str) -> tuple[Any, list[Any]]:
    main_fn = None
    helpers: list[Any] = []
    for op in list(module.body.operations):
        if op.operation.name != "func.func":
            continue
        if ir.SymbolTable.get_symbol_name(op).value == "main":
            main_fn = op.opview
        else:
            helpers.append(op)
    if main_fn is None:
        raise ValueError(f"{label} MLIR module is missing a '@main' func.func")
    return main_fn, helpers


def _import_private_helpers(
    src_module: Any,
    helpers: Sequence[Any],
    dst_module: Any,
    prefix: str,
    used_symbols: set[str],
) -> None:
    for idx, helper_op in enumerate(helpers):
        old_sym = ir.SymbolTable.get_symbol_name(helper_op).value
        candidate = f"{prefix}{idx}_{old_sym}"
        suffix = 0
        while candidate in used_symbols:
            suffix += 1
            candidate = f"{prefix}{idx}_{suffix}_{old_sym}"
        used_symbols.add(candidate)
        ir.SymbolTable.replace_all_symbol_uses(old_sym, candidate, src_module.operation)
        ir.SymbolTable.set_symbol_name(helper_op, candidate)
        ir.SymbolTable.set_visibility(helper_op, "private")
        dst_module.body.append(helper_op.operation.clone())


def _remap_region_operands(region: Any, value_map: dict[Any, Any]) -> None:
    for block in region.blocks:
        for child_op in block.operations:
            raw_child = child_op.operation
            for idx, operand in enumerate(raw_child.operands):
                if operand in value_map:
                    raw_child.operands[idx] = value_map[operand]
            for sub_region in raw_child.regions:
                _remap_region_operands(sub_region, value_map)


def _splice_entry_block_into_while_region(
    src_func: Any,
    src_to_dst_args: Sequence[tuple[Any, Any]],
    dst_block: Any,
    make_return_operands: Callable[[Sequence[Any]], list[Any]],
    label: str,
) -> None:
    blocks = list(src_func.regions[0].blocks)
    if len(blocks) != 1:
        raise ValueError(
            f"{label} '@main' must have exactly 1 block, got {len(blocks)}"
        )
    ops = list(blocks[0].operations)
    if not ops or ops[-1].operation.name != "func.return":
        raise ValueError(f"{label} '@main' must end with 'func.return'")

    value_map: dict[Any, Any] = dict(src_to_dst_args)
    for src_op in ops[:-1]:
        raw_src = src_op.operation
        cloned = raw_src.clone()
        dst_block.append(cloned)
        for idx, operand in enumerate(cloned.operands):
            if operand in value_map:
                cloned.operands[idx] = value_map[operand]
        for child_region in cloned.regions:
            _remap_region_operands(child_region, value_map)
        for old_res, new_res in zip(raw_src.results, cloned.results, strict=True):
            value_map[old_res] = new_res

    ret_op = ops[-1].operation
    ret_operands = make_return_operands([value_map.get(v, v) for v in ret_op.operands])
    with ir.InsertionPoint(dst_block), ret_op.location:
        hlo.ReturnOp(ret_operands)


@contextlib.contextmanager
def _parse_while_subgraphs(
    cond_mlir_text: str,
    body_mlir_text: str,
    *,
    num_carries: int,
) -> Iterator[_ParsedWhileSubgraphs]:
    if num_carries < 1:
        raise ValueError(f"num_carries must be >= 1, got {num_carries}")

    with make_ir_context():
        cond_mod = ir.Module.parse(cond_mlir_text)
        body_mod = ir.Module.parse(body_mlir_text)
        cond_main, cond_helpers = _find_main_and_helpers(cond_mod, "cond")
        body_main, body_helpers = _find_main_and_helpers(body_mod, "body")
        raw_donated = _extract_donated_carry_indices(body_main, num_carries)

        for mod in (cond_mod, body_mod):
            for region in mod.operation.regions:
                _strip_sharding_annotations_in_region(region)

        cond_multi = any(
            _region_has_multi_device_ops(r) for r in cond_mod.operation.regions
        )
        body_multi = any(
            _region_has_multi_device_ops(r) for r in body_mod.operation.regions
        )
        cond_order_sens, cond_red_sizes = _inspect_main_implicit_multi_device_ops(
            cond_main.regions[0]
        )
        body_order_sens, body_red_sizes = _inspect_main_implicit_multi_device_ops(
            body_main.regions[0]
        )
        yield _ParsedWhileSubgraphs(
            cond_mod=cond_mod,
            cond_main=cond_main,
            cond_helpers=cond_helpers,
            body_mod=body_mod,
            body_main=body_main,
            body_helpers=body_helpers,
            raw_donated_carry_indices=raw_donated,
            cond_has_multi_device_ops=cond_multi,
            body_has_multi_device_ops=body_multi,
            cond_has_order_sensitive_ops=cond_order_sens,
            body_has_order_sensitive_ops=body_order_sens,
            cond_world_reduce_sizes=cond_red_sizes,
            body_world_reduce_sizes=body_red_sizes,
        )


def _validate_while_subgraph_signatures(
    cond_main: Any,
    body_main: Any,
    *,
    num_carries: int,
    num_additional_inputs: int,
    has_rng_state: bool,
) -> _ValidatedWhileSignatures:
    cond_args = list(cond_main.regions[0].blocks[0].arguments)
    body_args = list(body_main.regions[0].blocks[0].arguments)
    cond_results = list(cond_main.type.results)
    body_results = list(body_main.type.results)

    if len(cond_results) != 1 or not _is_i1_scalar_tensor_type(cond_results[0]):
        raise ValueError(
            "while_loop cond_fn '@main' must return a single scalar "
            f"tensor<i1>, got {cond_results}"
        )
    if len(cond_args) != num_carries + num_additional_inputs:
        raise ValueError(
            f"while_loop cond_fn '@main' expected "
            f"{num_carries + num_additional_inputs} arguments, got {len(cond_args)}"
        )

    rng_extra = 1 if has_rng_state else 0
    if len(body_results) != num_carries + rng_extra:
        raise ValueError(
            f"while_loop body_fn '@main' must return {num_carries + rng_extra} "
            f"results, got {len(body_results)}"
        )
    if len(body_args) != num_carries + num_additional_inputs + rng_extra:
        raise ValueError(
            f"while_loop body_fn '@main' must accept "
            f"{num_carries + num_additional_inputs + rng_extra} arguments, "
            f"got {len(body_args)}"
        )

    body_user_carry_args = body_args[:num_carries]
    rng_arg_in_body: Any | None = None
    if has_rng_state:
        if not _is_rng_state_tensor_type(body_results[num_carries]):
            raise ValueError(
                "Expected trailing RNG state result of type tensor<2xui64>, "
                f"got {body_results[num_carries]}"
            )
        rng_arg = body_args[num_carries]
        if not _is_rng_state_tensor_type(rng_arg.type):
            raise ValueError(
                "Expected RNG state argument of type tensor<2xui64> at "
                f"body '@main' index {num_carries}, got {rng_arg.type}"
            )
        rng_arg_in_body = rng_arg
        body_add_args = body_args[num_carries + 1 :]
    else:
        body_add_args = body_args[num_carries:]

    for idx in range(num_carries):
        if body_user_carry_args[idx].type != body_results[idx]:
            raise ValueError(
                f"while_loop body carry {idx} type mismatch: input "
                f"{body_user_carry_args[idx].type} vs output {body_results[idx]}"
            )
        if cond_args[idx].type != body_user_carry_args[idx].type:
            raise ValueError(
                f"while_loop carry {idx} type mismatch between cond "
                f"({cond_args[idx].type}) and body ({body_user_carry_args[idx].type})"
            )

    for idx in range(num_additional_inputs):
        if cond_args[num_carries + idx].type != body_add_args[idx].type:
            raise ValueError(
                f"while_loop additional_input {idx} type mismatch between cond "
                f"({cond_args[num_carries + idx].type}) and body "
                f"({body_add_args[idx].type})"
            )

    effective_carry_types = [a.type for a in body_user_carry_args]
    if rng_arg_in_body is not None:
        effective_carry_types.append(rng_arg_in_body.type)
    additional_types = [a.type for a in body_add_args]

    return _ValidatedWhileSignatures(
        cond_args=cond_args,
        body_user_carry_args=body_user_carry_args,
        rng_arg_in_body=rng_arg_in_body,
        body_add_args=body_add_args,
        effective_carry_types=effective_carry_types,
        wrapper_arg_types=[*effective_carry_types, *additional_types],
    )


def _emit_while_op(
    *,
    cond_main: Any,
    body_main: Any,
    sigs: _ValidatedWhileSignatures,
    num_carries: int,
    has_rng_state: bool,
    while_operands: list[Any],
) -> list[Any]:
    num_effective_carries = len(sigs.effective_carry_types)
    while_op = hlo.WhileOp(sigs.wrapper_arg_types, while_operands)

    cond_block = while_op.cond.blocks.append(*sigs.wrapper_arg_types)
    cond_dst_args = list(cond_block.arguments)
    cond_mapped_dst = (
        cond_dst_args[:num_carries] + cond_dst_args[num_effective_carries:]
    )
    _splice_entry_block_into_while_region(
        cond_main,
        list(zip(sigs.cond_args, cond_mapped_dst, strict=True)),
        cond_block,
        lambda outs: [outs[0]],
        "cond",
    )

    body_block = while_op.body.blocks.append(*sigs.wrapper_arg_types)
    body_dst_args = list(body_block.arguments)
    body_arg_map: list[tuple[Any, Any]] = list(
        zip(sigs.body_user_carry_args, body_dst_args[:num_carries], strict=True)
    )
    if has_rng_state:
        assert sigs.rng_arg_in_body is not None
        body_arg_map.append((sigs.rng_arg_in_body, body_dst_args[num_carries]))
    body_arg_map.extend(
        zip(sigs.body_add_args, body_dst_args[num_effective_carries:], strict=True)
    )
    passthrough_add_args = body_dst_args[num_effective_carries:]
    _splice_entry_block_into_while_region(
        body_main,
        body_arg_map,
        body_block,
        lambda outs: [*outs, *passthrough_add_args],
        "body",
    )
    return list(while_op.results[:num_effective_carries])


def _make_replicated_sdy_sharding(mesh_sym: str, mlir_type: Any) -> Any:
    ranked = ir.RankedTensorType(mlir_type)
    dims = [
        sdy.DimensionShardingAttr.get([], is_closed=True) for _ in range(ranked.rank)
    ]
    return sdy.TensorShardingAttr.get(mesh_sym, dims, [])


def _assemble_while_loop_mlir(
    parsed: _ParsedWhileSubgraphs,
    *,
    num_carries: int,
    num_additional_inputs: int,
    has_rng_state: bool,
    num_partitions: int,
    allowed_donor_indices: Collection[int] | None,
) -> _WhileLoopMlirArtifact:
    if num_additional_inputs < 0:
        raise ValueError(
            f"num_additional_inputs must be >= 0, got {num_additional_inputs}"
        )
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

    if allowed_donor_indices is not None:
        allowed_set = {int(i) for i in allowed_donor_indices}
        donated_carry_indices = tuple(
            idx for idx in parsed.raw_donated_carry_indices if idx in allowed_set
        )
    else:
        donated_carry_indices = parsed.raw_donated_carry_indices

    sigs = _validate_while_subgraph_signatures(
        parsed.cond_main,
        parsed.body_main,
        num_carries=num_carries,
        num_additional_inputs=num_additional_inputs,
        has_rng_state=has_rng_state,
    )

    wrapper_mod = ir.Module.create(ir.Location.unknown())
    used_symbols: set[str] = {"main"} | {
        ir.SymbolTable.get_symbol_name(op).value
        for op in (*parsed.cond_helpers, *parsed.body_helpers)
    }
    _import_private_helpers(
        parsed.cond_mod, parsed.cond_helpers, wrapper_mod, "__while_cond_", used_symbols
    )
    _import_private_helpers(
        parsed.body_mod, parsed.body_helpers, wrapper_mod, "__while_body_", used_symbols
    )

    with ir.InsertionPoint(wrapper_mod.body), ir.Location.unknown():
        mesh_sym = "__while_mesh"
        if num_partitions > 1:
            suffix = 0
            while mesh_sym in used_symbols:
                suffix += 1
                mesh_sym = f"__while_mesh_{suffix}"
            used_symbols.add(mesh_sym)
            sdy.MeshOp(
                mesh_sym,
                sdy.MeshAttr.get(
                    [sdy.MeshAxisAttr.get(_MANUAL_MESH_AXIS, num_partitions)]
                ),
            )

        main_fn = func.FuncOp(
            "main",
            ir.FunctionType.get(sigs.wrapper_arg_types, sigs.effective_carry_types),
        )
        main_fn.sym_visibility = ir.StringAttr.get("public")

        if donated_carry_indices:
            donated_set = set(donated_carry_indices)
            i32_ty = ir.IntegerType.get_signless(32)
            main_fn.arg_attrs = ir.ArrayAttr.get(
                [
                    ir.DictAttr.get(
                        {"tf.aliasing_output": ir.IntegerAttr.get(i32_ty, idx)}
                    )
                    if idx in donated_set
                    else ir.DictAttr.get({})
                    for idx in range(len(sigs.wrapper_arg_types))
                ]
            )

        entry_block = main_fn.add_entry_block()
        with ir.InsertionPoint(entry_block):
            if num_partitions > 1:
                in_sh = sdy.TensorShardingPerValueAttr.get(
                    [
                        _make_replicated_sdy_sharding(mesh_sym, t)
                        for t in sigs.wrapper_arg_types
                    ]
                )
                out_sh = sdy.TensorShardingPerValueAttr.get(
                    [
                        _make_replicated_sdy_sharding(mesh_sym, t)
                        for t in sigs.effective_carry_types
                    ]
                )
                manual_op = sdy.ManualComputationOp(
                    sigs.effective_carry_types,
                    list(entry_block.arguments),
                    in_sh,
                    out_sh,
                    sdy.ManualAxesAttr.get([ir.StringAttr.get(_MANUAL_MESH_AXIS)]),
                )
                manual_block = manual_op.body.blocks.append(*sigs.wrapper_arg_types)
                with ir.InsertionPoint(manual_block):
                    carry_outs = _emit_while_op(
                        cond_main=parsed.cond_main,
                        body_main=parsed.body_main,
                        sigs=sigs,
                        num_carries=num_carries,
                        has_rng_state=has_rng_state,
                        while_operands=list(manual_block.arguments),
                    )
                    sdy.ReturnOp(carry_outs)
                func.ReturnOp(list(manual_op.results))
            else:
                carry_outs = _emit_while_op(
                    cond_main=parsed.cond_main,
                    body_main=parsed.body_main,
                    sigs=sigs,
                    num_carries=num_carries,
                    has_rng_state=has_rng_state,
                    while_operands=list(entry_block.arguments),
                )
                func.ReturnOp(carry_outs)

    if not wrapper_mod.operation.verify():
        raise ValueError("Generated stablehlo.while MLIR module failed verification")

    mlir_text = str(wrapper_mod.operation.get_asm(enable_debug_info=True))
    normalized_mlir_text = str(wrapper_mod.operation.get_asm(enable_debug_info=False))
    kernel_key = hashlib.blake2b(
        normalized_mlir_text.encode("utf-8"), digest_size=16
    ).hexdigest()
    return _WhileLoopMlirArtifact(
        mlir_text=mlir_text,
        normalized_mlir_text=normalized_mlir_text,
        kernel_key=kernel_key,
        donated_carry_indices=donated_carry_indices,
    )


def _build_while_loop_mlir(
    cond_mlir_text: str,
    body_mlir_text: str,
    *,
    num_carries: int,
    num_additional_inputs: int = 0,
    has_rng_state: bool = False,
    num_partitions: int = 1,
    allowed_donor_indices: Collection[int] | None = None,
) -> _WhileLoopMlirArtifact:
    """Assembles ``cond`` and ``body`` StableHLO modules into ``stablehlo.while``."""
    with _parse_while_subgraphs(
        cond_mlir_text, body_mlir_text, num_carries=num_carries
    ) as parsed:
        return _assemble_while_loop_mlir(
            parsed,
            num_carries=num_carries,
            num_additional_inputs=num_additional_inputs,
            has_rng_state=has_rng_state,
            num_partitions=num_partitions,
            allowed_donor_indices=allowed_donor_indices,
        )
