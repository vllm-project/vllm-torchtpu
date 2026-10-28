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
"""Bind scheduling identities to FX call sites."""

import copy

import torch
import torch.fx as fx


def _bind_scheduling_group_namespaces(graph: fx.GraphModule) -> None:
    """Bind static native-op namespaces to actual call sites in this program."""
    argument = "scheduling_group_namespace"
    namespace = 0
    seen: set[int] = set()

    def parameter(node: fx.Node) -> int | None:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return None
        for index, item in enumerate(node.target._schema.arguments):
            if item.name == argument:
                return index
        return None

    def contains_groups(module: fx.GraphModule) -> bool:
        return any(
            parameter(node) is not None
            for child in module.modules()
            if isinstance(child, fx.GraphModule)
            for node in child.graph.nodes
        )

    def bind(module: fx.GraphModule) -> None:
        nonlocal namespace
        seen.add(id(module))
        changed = False
        for node in module.graph.nodes:
            if node.op == "call_module":
                child = module.get_submodule(str(node.target))
                if isinstance(child, fx.GraphModule) and contains_groups(child):
                    if id(child) in seen:
                        occupied = (
                            module.__dict__.keys()
                            | module._modules.keys()
                            | module._parameters.keys()
                            | module._buffers.keys()
                        )
                        name = f"_scheduling_{node.name}"
                        while name in occupied:
                            name += "_"
                        child = fx.GraphModule(child, copy.deepcopy(child.graph))
                        module.add_module(name, child)
                        node.target = name
                    bind(child)
                    changed = True
            index = parameter(node)
            if index is None:
                continue
            namespace += 1
            args, kwargs = list(node.args), dict(node.kwargs)
            if index < len(args):
                args[index] = namespace
                kwargs.pop(argument, None)
            else:
                kwargs[argument] = namespace
            node.args, node.kwargs = tuple(args), kwargs
            changed = True
        if changed:
            module.graph.lint()
            module.recompile()

    bind(graph)

