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
"""Host critical-path helpers for the serving decode loop."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _dynamo_graph_count() -> int:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


class GuardEvalSkipper:
    """Arms Dynamo's `skip_guard_eval_unsafe` stance once the runner is warm.

    Every `torch.compile`d call in a decode cycle (sampler, verify, drafter
    inputs, glue) first runs Dynamo's cache lookup, which evaluates the
    entry's full guard set. Under the stance Dynamo runs only the guards that
    ever told its cache entries apart. That is only correct when no new
    specialization appears afterwards, so the stance is armed after
    `quiet_steps` consecutive steps without a new Dynamo graph. A recompile
    after arming raises in Dynamo rather than compiling.
    """

    def __init__(self, quiet_steps: int):
        self.quiet_steps = int(quiet_steps)
        self.armed = False
        self._graphs = -1
        self._quiet = 0

    @property
    def pending(self) -> bool:
        return self.quiet_steps > 0 and not self.armed

    def step(self) -> bool:
        """Count one runner step; returns True on the step that arms."""
        if not self.pending:
            return False
        graphs = _dynamo_graph_count()
        if graphs != self._graphs:
            self._graphs = graphs
            self._quiet = 0
            return False
        self._quiet += 1
        if self._quiet < self.quiet_steps:
            return False
        torch.compiler.set_stance("default", skip_guard_eval_unsafe=True)
        self.armed = True
        logger.info(
            "TPU_HOST_SKIP_GUARD_EVAL_AFTER_STEPS=%d: no new Dynamo graph for "
            "%d steps (%d graphs); skip_guard_eval_unsafe stance armed.",
            self.quiet_steps,
            self._quiet,
            graphs,
        )
        return True

    def disarm(self) -> None:
        if self.armed:
            torch.compiler.set_stance("default", skip_guard_eval_unsafe=False)
            self.armed = False
            self._quiet = 0


# --- TPU_HOST_KEEP_INPUT_MUTATIONS ------------------------------------------
#
# torch_tpu's backend runs AOTAutograd with keep_inference_input_mutations=False:
# a compiled graph returns the new value of every input it writes (the MiniMax
# forward writes 60 kv caches + 60 index caches), and AOT's runtime epilogue
# then runs one eager `inp.copy_(new)` per written input on the host after the
# executable returns (120 host copy_ dispatches per decode forward). With 1 the
# copies stay in the graph as `aten.copy_(placeholder, new)`. With 2 those
# placeholders are also passed to torch_tpu's StaticCompiler as donated inputs
# (its make_fx in-place path), so the executable may update them in place.


def mutated_placeholder_indices(gm: torch.fx.GraphModule) -> list[int]:
    """Placeholder positions that the graph overwrites with `aten.copy_`."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    index = {n: i for i, n in enumerate(placeholders)}
    out = set()
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.copy_.default
            and node.args[0] in index
        ):
            out.add(index[node.args[0]])
    return sorted(out)


def keep_input_mutations_aot_autograd(aot_autograd_fn):
    """Wraps `aot_autograd(**kwargs)` to keep inference input mutations."""

    def aot_autograd_keep(**kwargs):
        kwargs["keep_inference_input_mutations"] = True
        return aot_autograd_fn(**kwargs)

    aot_autograd_keep._tpu_keep_input_mutations = True
    return aot_autograd_keep


def install_keep_input_mutations(mode: int | None = None) -> None:
    """Patch torch_tpu's TpuBackend per TPU_HOST_KEEP_INPUT_MUTATIONS."""
    if mode is None:
        from vllm_torchtpu import envs

        mode = int(envs.TPU_HOST_KEEP_INPUT_MUTATIONS)
    if mode <= 0:
        return
    from torch_tpu._internal.compile import _backend, compiler

    if not getattr(_backend.aot_autograd, "_tpu_keep_input_mutations", False):
        _backend.aot_autograd = keep_input_mutations_aot_autograd(
            _backend.aot_autograd
        )
    if mode >= 2 and not getattr(
        compiler.StaticCompiler.__call__, "_tpu_donate_mutated", False
    ):
        original_call = compiler.StaticCompiler.__call__

        def call_donating(self, graph_module, example_inputs, *args, **kwargs):
            if kwargs.get("donated_inputs") is None and len(args) < 5:
                donated = mutated_placeholder_indices(graph_module)
                if donated:
                    kwargs["donated_inputs"] = donated
            return original_call(self, graph_module, example_inputs, *args, **kwargs)

        call_donating._tpu_donate_mutated = True
        compiler.StaticCompiler.__call__ = call_donating
    logger.warning(
        "TPU_HOST_KEEP_INPUT_MUTATIONS=%d: TpuBackend patched. On the pinned "
        "torch_tpu the in-graph cache writes are lost (exos j-d93796de); "
        "outputs will be wrong.",
        mode,
    )
