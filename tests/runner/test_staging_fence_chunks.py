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
"""The staging fence orders each chunk's host rewrites after the previous
chunk's uploads, and releases held host tensors only once a fence recorded
after them has been waited on."""

import inspect
import types
from types import SimpleNamespace

import torch
from runner_input_test_utils import _make_runner, _scheduler_output

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_second_chunk_waits_for_the_first_chunks_fence(monkeypatch):
    """Chunk 2's input preparation synchronizes chunk 1's fence before it
    rewrites the host staging tensors, and only then releases chunk 1's
    held uploads."""
    waits = []
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=[4, 4],
        scheduled_tokens=[4, 4],
        token_paddings=[8, 16],
        pcp_size=1,
        max_num_reqs=1,
    )

    class _Event:
        def record(self):
            pass

        def synchronize(self):
            waits.append(
                (
                    runner.input_ids_cpu[:4].tolist(),
                    [id(t) for t in runner._fenced_staging_sources],
                )
            )

    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=_Event), raising=False)
    runner._record_input_staging_fence = types.MethodType(
        TPUModelRunner._record_input_staging_fence, runner
    )
    scheduler_output = _scheduler_output([4, 4])

    result = TPUModelRunner._prepare_inputs(
        runner, scheduler_output, 0, num_decode_reqs=0
    )
    end_index = result[4]
    assert end_index == 1
    first_ids = runner.input_ids_cpu[:4].tolist()
    held = [id(t) for t in runner._staging_sources]
    assert held, "chunk 1's uploads are held"
    runner._record_input_staging_fence()
    assert [id(t) for t in runner._fenced_staging_sources] == held

    TPUModelRunner._prepare_inputs(
        runner, scheduler_output, end_index, num_decode_reqs=0
    )
    assert waits == [(first_ids, held)]
    assert runner.input_ids_cpu[:4].tolist() != first_ids
    assert runner._fenced_staging_sources == []


def _source_index(method, needle):
    source = inspect.getsource(method)
    assert needle in source, needle
    return source.index(needle)


def test_steps_wait_before_rewriting_and_record_before_the_forward():
    """execute_model waits on the previous fence before the state update and
    records each chunk's fence before its forward; a dummy run, which shares
    the host staging tensors, does the same."""
    step = TPUModelRunner.execute_model
    assert _source_index(step, "self._wait_input_staging_fence()") < _source_index(
        step, "self._update_states(scheduler_output)"
    )
    assert _source_index(step, "self._record_input_staging_fence()") < _source_index(
        step, "self.forward_model("
    )
    dummy = TPUModelRunner._dummy_run
    assert _source_index(dummy, "self._wait_input_staging_fence()") < _source_index(
        dummy, "self._build_attention_metadata("
    )
    assert _source_index(dummy, "self._record_input_staging_fence()") < _source_index(
        dummy, "self.forward_model("
    )


def test_runner_registers_for_held_uploads():
    """The runner registers its upload and hold once its held lists exist,
    and wraps the loaded model's encoder metadata."""
    init = TPUModelRunner.__init__
    assert _source_index(init, "self._staging_sources: list") < _source_index(
        init, "host_uploads.set_uploader(self._upload, self.device, self._hold)"
    )
    load = TPUModelRunner.load_model
    assert _source_index(load, "self.model = model") < _source_index(
        load, "host_uploads.hold_encoder_metadata_uploads(self.model)"
    )
