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
"""capture_model warms programs in the grad mode serving calls them in: dynamo
guards on grad mode, so a mismatch retraces every warmed shape on its first
serving call."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Served(Exception):
    pass


def test_warmed_graph_is_reused_by_execute_model_and_sample_tokens():
    torch._dynamo.reset()
    compiles = []

    def backend(gm, _inputs):
        compiles.append(gm)
        return gm.forward

    program = torch.compile(lambda x: x + 1, backend=backend, dynamic=False)

    def run(_self, *_args):
        program(torch.zeros(4))

    def run_then_stop(_self, *_args):
        run(_self)
        raise _Served

    runner = object.__new__(TPUModelRunner)
    runner._pp_wave = None
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    runner.enforce_eager = True
    runner.execute_model_state = None
    runner._pp_is_last = False
    with (
        patch.object(TPUModelRunner, "_add_pipeline_chunk_buckets", run),
        patch.object(TPUModelRunner, "_update_states", run_then_stop),
        patch.object(TPUModelRunner, "_sample_tokens_pp_intermediate_stage", run),
    ):
        runner.capture_model()
        with pytest.raises(_Served):
            runner.execute_model(SimpleNamespace())
        runner.sample_tokens(None)
    assert len(compiles) == 1
