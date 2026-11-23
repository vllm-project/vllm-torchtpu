# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve linear, aliased buffers at the Engram HostExecute boundary."""

import logging
import re
from functools import cache

from jaxlib import _hlo

logger = logging.getLogger(__name__)


@cache
def _graph_type():
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from moe_schedule import Module

    proto = descriptor_pb2.FileDescriptorProto.FromString(
        Module.DESCRIPTOR.file.serialized_pb
    )
    instruction = next(m for m in proto.message_type if m.name == "Instruction")
    instruction.field.add(name="called_computation_ids", number=38, type=3, label=3)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(proto)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("edge.Module"))


def _start_before_compute(blob: bytes, names: list[str]) -> bytes:
    from moe_schedule import _ancestors, _sort

    module = _graph_type().FromString(blob)
    entry = next(c for c in module.computations if c.id == module.entry_computation_id)
    by_id = {n.id: n for n in entry.instructions}
    starts = [n for n in entry.instructions if f"%{n.name}" in names]
    assert len(starts) == len(names) and all(n.opcode == "async-start" for n in starts)
    prerequisites = set().union(*(_ancestors(n, by_id) | {n.id} for n in starts))
    dones = {
        n.id
        for n in entry.instructions
        if n.opcode == "async-done" and n.operand_ids[0] in {s.id for s in starts}
    }

    def matrix(node, matrix_computations):
        return (
            node.opcode in ("dot", "convolution")
            or (node.opcode == "custom-call" and "matmul" in node.metadata.op_name)
            or any(c in matrix_computations for c in node.called_computation_ids)
        )

    matrix_computations = set()
    while True:
        expanded = {
            c.id
            for c in module.computations
            if any(matrix(n, matrix_computations) for n in c.instructions)
        }
        if expanded == matrix_computations:
            break
        matrix_computations = expanded
    for node in entry.instructions:
        if (
            node.id not in prerequisites
            and matrix(node, matrix_computations)
            and not (_ancestors(node, by_id) & dones)
        ):
            node.control_predecessor_ids.extend(s.id for s in starts)
            sequence = next(
                (s.value for s in module.schedule.sequences if s.key == entry.id), None
            )
            _sort(entry, sequence, {})
            logger.info("Engram FFI submissions precede compute: %s", node.name)
            return module.SerializeToString()
    return blob


def rewrite_text(text: str) -> tuple[str, list[str]]:
    """Reinterpret flat int32 Host buffers without changing entry/DMA layouts."""
    blocks = re.split(r"(?=^(?:ENTRY )?%[\w.-]+ \()", text, flags=re.MULTILINE)
    targets = {}
    for block in blocks:
        if not block.startswith("%"):
            continue
        calls = re.findall(
            r"(%[\w.-]+) = .* custom-call\(([^)]+)\), "
            r'custom_call_target="engram_host_gather_256"',
            block,
        )
        if not calls:
            continue
        aliases = {}
        for result, operands in calls:
            staging = operands.split(",")[1].strip()
            parameter = re.search(
                rf"  {re.escape(staging)} = .* parameter\((\d+)\)", block
            )
            assert parameter is not None, "FFI staging must be a Host argument"
            aliases[result] = int(parameter[1])
        computation = re.match(r"%([\w.-]+)", block)
        root = re.search(r"  ROOT (%[\w.-]+) = (.+)", block)
        assert computation is not None and root is not None
        elements = re.search(r" tuple\(([^)]+)\)", root[2])
        outputs = elements[1].split(", ") if elements else [root[1]]
        targets[computation[1]] = (bool(elements), [aliases[n] for n in outputs])
    changed = []
    for index, block in enumerate(blocks):
        if not block.startswith("ENTRY "):
            continue
        lines = block.splitlines()
        output = []
        starts = set()
        for line in lines:
            match = re.match(
                r"  (%[\w.-]+) = (.+) custom-call-start\(([^)]+)\)(.*)", line
            )
            if (
                match
                and 'custom_call_target="HostExecute"' in match[4]
                and any(
                    f"called_computations={{%{target}}}" in match[4]
                    for target in targets
                )
            ):
                name, shape, operands, attributes = match.groups()
                ((is_tuple, aliases),) = [
                    description
                    for target, description in targets.items()
                    if f"called_computations={{%{target}}}" in attributes
                ]
                args = [arg.strip() for arg in operands.split(",")]
                shapes = re.findall(r"s32\[\d+\]\{0:T\(1024\)S\(5\)\}", shape)
                assert len(shapes) == len(args) + len(aliases), line
                assert all(
                    shapes[arg] == shapes[len(args) + result]
                    for result, arg in enumerate(aliases)
                ), line
                assert all(
                    int(shape.split("[")[1].split("]")[0]) % 1024 == 0
                    for shape in shapes
                )
                assert "output_to_operand_aliasing" not in attributes, line
                mapping = ", ".join(
                    f"{{{result if is_tuple else ''}}}: ({arg}, {{}})"
                    for result, arg in enumerate(aliases)
                )
                attributes += f", output_to_operand_aliasing={{{mapping}}}"
                views = []
                for arg_index, (arg, arg_shape) in enumerate(
                    zip(args, shapes, strict=False)
                ):
                    view = f"{name}.host_linear_{arg_index}"
                    linear = arg_shape.replace("T(1024)S(5)", "T(1)L(1024)S(5)")
                    output.append(f"  {view} = {linear} bitcast({arg})")
                    views.append(view)
                shape = shape.replace("T(1024)S(5)", "T(1)L(1024)S(5)")
                output.append(
                    f"  {name} = {shape} custom-call-start({', '.join(views)})"
                    f"{attributes}"
                )
                starts.add(name)
                changed.append(name)
                continue
            match = re.match(
                r"  (%[\w.-]+) = (.+) "
                r"custom-call-done\((%[\w.-]+)\)(.*)",
                line,
            )
            if match and match[3] in starts:
                name, shape, start, attributes = match.groups()
                linear = shape.replace("T(1024)S(5)", "T(1)L(1024)S(5)")
                output.append(
                    f"  {name}.host_linear = {linear} custom-call-done({start})"
                    f"{attributes}"
                )
                if shape.startswith("("):
                    elements = re.findall(r"s32\[\d+\]\{[^}]+\}", shape)
                    views = []
                    for result, element in enumerate(elements):
                        view = f"{name}.element_{result}"
                        linear = element.replace("T(1024)S(5)", "T(1)L(1024)S(5)")
                        output.append(
                            f"  {view}.linear = {linear} "
                            f"get-tuple-element({name}.host_linear), index={result}"
                        )
                        output.append(f"  {view} = {element} bitcast({view}.linear)")
                        views.append(view)
                    output.append(f"  {name} = {shape} tuple({', '.join(views)})")
                else:
                    output.append(f"  {name} = {shape} bitcast({name}.host_linear)")
                continue
            output.append(line)
        blocks[index] = "\n".join(output) + "\n"
    return "".join(blocks), changed


@cache
def install_host_ffi() -> None:
    """Install the experimental linear-view correction before model compilation."""
    import libtpu
    from jax.extend import xla
    from jaxlib import xla_client

    if not xla_client.pjrt_plugin_loaded("tpu"):
        xla_client.load_pjrt_plugin_dynamically("tpu", libtpu.get_library_path())

    def rewrite(blob: bytes) -> bytes | None:
        module = _hlo.HloModule.from_serialized_hlo_module_proto(blob)
        after, changed = rewrite_text(module.to_string())
        if not changed:
            return None
        logger.info("Engram FFI linear HostExecute views: %s", changed)
        result = _hlo.hlo_module_from_text(after).as_serialized_hlo_module_proto()
        return _start_before_compute(result, changed)

    xla.register_hlo_module_transformation(
        rewrite,
        name="engram_ffi_linear_views_v1",
        platforms="tpu",
        stage=xla.PipelineStage.PRE_SCHEDULER,
    )
