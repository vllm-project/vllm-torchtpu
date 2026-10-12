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
"""Shared check for the precompile-sweep rotation tests: whether a traced
graph calls anything that may communicate."""

import torch


def ops_outside_aten(graph_module: torch.fx.GraphModule) -> list[str]:
    """Calls in a Dynamo graph to anything that may communicate: a registered
    op outside aten/prims (custom and Pallas ops included, since a collective
    can hide inside them) or a torch.distributed function."""
    found = []
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
                if str(target).split(".")[0] not in ("aten", "prims"):
                    found.append(str(target))
            elif getattr(target, "__module__", "").startswith("torch.distributed"):
                found.append(str(target))
    return found
