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
"""Graph-input order of a traced module, to guard flag-gated fused paths.

The shape-variant hand-over (``compilation.shape_variants``) reuses one trace's
compiled bucket from another trace of the same model only when their subgraph
inputs line up. Piecewise compilation splits the Dynamo graph and orders each
subgraph's inputs by their first use inside it, so a size-gated branch that
first uses a module's weights in a different order than the other branch moves
the inputs between traces: the hand-over refuses the bucket at engine init, or,
when only same-shaped weights trade places, accepts it and binds the wrong
weights. Trace both branches with :func:`parameters_by_first_use` and assert
the lists are equal.
"""

import torch


def parameters_by_first_use(module: torch.nn.Module, *args) -> list[str]:
    """Placeholder names of the parameters in ``module(*args)``'s graph (one
    full graph, static shapes), in order of their first use, as a piecewise
    subgraph would take them."""
    names = []

    def backend(gm, example_inputs):
        position = {node: index for index, node in enumerate(gm.graph.nodes)}
        params = [
            node
            for node in gm.graph.nodes
            if node.op == "placeholder" and "_parameters_" in node.name and node.users
        ]
        params.sort(key=lambda node: min(position[user] for user in node.users))
        names.extend(node.name for node in params)
        return gm

    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(module, backend=backend, fullgraph=True, dynamic=False)(*args)
    torch._dynamo.reset()
    return names
