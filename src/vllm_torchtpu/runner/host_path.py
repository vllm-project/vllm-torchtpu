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
