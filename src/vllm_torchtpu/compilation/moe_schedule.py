# SPDX-License-Identifier: Apache-2.0
"""Order split-SC MoE submissions and independent scale layout copies.

Graph-only protobuf fields follow openxla/xla's xla/service/hlo.proto.
Unknown fields (including layouts and backend configs) round-trip unchanged.
"""

import heapq
import json
import threading
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from jax.extend import xla


def module_type():
    file = descriptor_pb2.FileDescriptorProto(
        name="edge_hlo.proto", package="edge", syntax="proto3"
    )

    def message(name, fields):
        msg = file.message_type.add(name=name)
        for name, number, typ, repeated, target in fields:
            f = msg.field.add(
                name=name, number=number, type=typ, label=3 if repeated else 1
            )
            if target:
                f.type_name = ".edge." + target

    message("Metadata", [("op_name", 2, 9, False, None)])
    message(
        "Instruction",
        [
            ("name", 1, 9, False, None),
            ("opcode", 2, 9, False, None),
            ("metadata", 7, 11, False, "Metadata"),
            ("id", 35, 3, False, None),
            ("operand_ids", 36, 3, True, None),
            ("control_predecessor_ids", 37, 3, True, None),
            ("backend_config", 43, 12, False, None),
        ],
    )
    message(
        "Computation",
        [
            ("name", 1, 9, False, None),
            ("instructions", 2, 11, True, "Instruction"),
            ("id", 5, 3, False, None),
            ("root_id", 6, 3, False, None),
        ],
    )
    message("Sequence", [("instruction_ids", 1, 3, True, None)])
    message(
        "ScheduleEntry",
        [("key", 1, 3, False, None), ("value", 2, 11, False, "Sequence")],
    )
    message("Schedule", [("sequences", 1, 11, True, "ScheduleEntry")])
    message(
        "Module",
        [
            ("name", 1, 9, False, None),
            ("computations", 3, 11, True, "Computation"),
            ("entry_computation_id", 6, 3, False, None),
            ("schedule", 7, 11, False, "Schedule"),
        ],
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("edge.Module"))


Module = module_type()


def _ancestors(node: Any, by_id: dict[int, Any]) -> set[int]:
    seen = set()
    stack = list(node.operand_ids) + list(node.control_predecessor_ids)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        parent = by_id[current]
        stack.extend(parent.operand_ids)
        stack.extend(parent.control_predecessor_ids)
    return seen


def _nearest(
    node: Any,
    by_id: dict[int, Any],
    marker: str | tuple[str, ...],
    opcode: str,
    *,
    stop_at_other_starts: bool = False,
) -> Any:
    """Stop at the first matching depth, not at a previous layer's collective."""
    markers = (marker,) if isinstance(marker, str) else marker
    frontier = [node.id]
    visited = set()
    while frontier:
        found = [
            by_id[n]
            for n in frontier
            if by_id[n].opcode == opcode
            and any(m in by_id[n].metadata.op_name for m in markers)
        ]
        if found:
            return found[0] if len(found) == 1 else None
        if stop_at_other_starts and any(
            by_id[n].opcode == "async-start" for n in frontier
        ):
            return None
        visited.update(frontier)
        frontier = list(
            {p for n in frontier for p in by_id[n].operand_ids if p not in visited}
        )
    return None


def _sort(comp: Any, sequence: Any, priority_ids: dict[int, int]) -> None:
    by_id = {n.id: n for n in comp.instructions}
    nodes = (
        [by_id[n] for n in sequence.instruction_ids]
        if sequence is not None
        else list(comp.instructions)
    )
    position = {n.id: i for i, n in enumerate(nodes)}
    priority = {n: min(pos, priority_ids.get(n, pos)) for n, pos in position.items()}
    users: dict[int, list[int]] = {n.id: [] for n in nodes}
    pending = {}
    for node in nodes:
        deps = set(node.operand_ids) | set(node.control_predecessor_ids)
        pending[node.id] = len(deps)
        for predecessor in deps:
            users[predecessor].append(node.id)
    ready = [(priority[n], position[n]) for n, count in pending.items() if not count]
    heapq.heapify(ready)
    ordered = []
    while ready:
        node = nodes[heapq.heappop(ready)[1]]
        ordered.append(node)
        for user in users[node.id]:
            pending[user] -= 1
            if pending[user] == 0:
                heapq.heappush(ready, (priority[user], position[user]))
    if len(ordered) != len(nodes):
        raise ValueError("MoE scheduling control edges introduce a cycle")
    del comp.instructions[:]
    comp.instructions.extend(ordered)
    if sequence is not None:
        del sequence.instruction_ids[:]
        sequence.instruction_ids.extend(n.id for n in ordered)


def _pin_sparse_core(node: Any, core_id: int) -> bool:
    """Pin an existing SC offload before scheduling, preserving other settings."""
    if not node.backend_config:
        return False
    config = json.loads(node.backend_config)
    sparse = config.get("sparse_core_config")
    if sparse is None:
        return False
    if (
        sparse.get("core_ids") == [str(core_id)]
        and sparse.get("core_id_mutability") is False
    ):
        return False
    sparse["core_ids"] = [str(core_id)]
    sparse["core_id_mutability"] = False
    node.backend_config = json.dumps(config).encode()
    return True


def rewrite_moe_schedule(blob: bytes) -> bytes | None:
    """Match by kernel operands, never frontend argument names or module order.

    FP4 MoE's ABI has activation rows at 11, token indices at 10, and weight
    scales at 15/16 (see make_call in fused_moe/v2/kernel.py). A compiled
    graph can contain several MoE layers and unrelated collectives. Unsupported
    or ambiguous patterns are left untouched rather than ordered speculatively.
    """
    module = Module.FromString(blob)
    changed = False
    for comp in module.computations:
        by_id = {n.id: n for n in comp.instructions}
        sequence = next(
            (s.value for s in module.schedule.sequences if s.key == comp.id), None
        )
        positions = {
            n: i
            for i, n in enumerate(
                sequence.instruction_ids if sequence is not None else list(by_id)
            )
        }
        # Order independent multi-SC completions before advancing an AG.
        # Independent offloads otherwise can straddle the split-SC submissions
        # and stall libtpu's resource-aware scheduler in a full serving graph.
        multicore_done = []
        for node in comp.instructions:
            if node.opcode != "async-done" or len(node.operand_ids) != 1:
                continue
            start = by_id[node.operand_ids[0]]
            if start.opcode != "async-start":
                continue
            config = json.loads(start.backend_config or "{}")
            cores = config.get("sparse_core_config", {}).get("core_ids", [])
            if len(cores) > 1:
                multicore_done.append(node)
        priorities: dict[int, int] = {}
        comp_changed = False

        def edge(before: Any, after: Any, by_id=by_id) -> bool:
            nonlocal comp_changed, changed
            if before.id == after.id or after.id in _ancestors(before, by_id):
                return False
            if before.id not in after.control_predecessor_ids:
                after.control_predecessor_ids.append(before.id)
                comp_changed = changed = True
            return True

        for kernel in list(comp.instructions):
            if (
                kernel.opcode != "custom-call"
                or "fused_ep_moe_v2_fp4w_" not in kernel.metadata.op_name
                or len(kernel.operand_ids) < 17
            ):
                continue
            activation = _nearest(
                by_id[kernel.operand_ids[11]],
                by_id,
                ("moe_activation_ag", "topology_three_phase_ag"),
                "async-start",
                stop_at_other_starts=True,
            )
            metadata_done = _nearest(
                by_id[kernel.operand_ids[10]],
                by_id,
                ("moe_metadata_ag", "all_gather"),
                "async-done",
            )
            if activation is None or metadata_done is None:
                continue
            metadata_start = by_id[metadata_done.operand_ids[0]]
            if metadata_start.opcode not in ("async-start", "async-update"):
                continue
            scatter = _nearest(
                by_id[kernel.operand_ids[10]],
                by_id,
                "moe_preprocess_sc_scatter",
                "async-start",
            )
            for node, core_id in ((activation, 1), (metadata_start, 0), (scatter, 0)):
                if node is not None and _pin_sparse_core(node, core_id):
                    comp_changed = changed = True
            activation_deps = _ancestors(activation, by_id)
            for done in multicore_done:
                # Serialized instruction order is not execution order. edge()
                # rejects downstream completions that would create a cycle.
                if done.id not in activation_deps:
                    edge(done, activation)
            activation_deps = _ancestors(activation, by_id)
            # Only this metadata producer's router nodes; prior layers and
            # activation input producers cannot be placed behind submission.
            for n in _ancestors(metadata_start, by_id):
                node = by_id[n]
                if (
                    n not in activation_deps
                    and "moe_early_ag_route" in node.metadata.op_name
                ):
                    edge(activation, node)
            edge(activation, metadata_start)
            for index in (15, 16):
                copy = by_id[kernel.operand_ids[index]]
                while (
                    copy.opcode in ("bitcast", "reshape") and len(copy.operand_ids) == 1
                ):
                    copy = by_id[copy.operand_ids[0]]
                if copy.opcode != "copy":
                    continue
                if metadata_done.id in _ancestors(copy, by_id) or copy.id in _ancestors(
                    metadata_start, by_id
                ):
                    continue
                edge(metadata_start, copy)
                edge(copy, metadata_done)
                for n in _ancestors(copy, by_id) | {copy.id}:
                    priorities[n] = min(
                        priorities.get(n, positions[n]), positions[metadata_done.id]
                    )
        if comp_changed:
            _sort(comp, sequence, priorities)
    return module.SerializeToString() if changed else None


_install_lock = threading.Lock()
_installed = False


def install_moe_schedule() -> None:
    """Register once per process; unrelated compilations are unchanged."""
    global _installed
    with _install_lock:
        if not _installed:
            xla.register_hlo_module_transformation(
                rewrite_moe_schedule,
                name="split_sc_moe_schedule",
                platforms="tpu",
                stage=xla.PipelineStage.PRE_SCHEDULER,
            )
            _installed = True
