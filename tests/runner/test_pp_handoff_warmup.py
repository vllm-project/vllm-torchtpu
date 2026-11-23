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
"""The pipeline hand-off warmup runs only the buckets the backbone compiled.

Under the PCP manager a rank's local token count stops at
max_local_num_tokens, and the standalone backbone pass compiles only those
buckets; a hand-off warmup over the full ladder reaches a bucket no graph was
compiled for ("Shape: 4096 out of considered ranges").
"""

from types import SimpleNamespace

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(cap):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.num_tokens_paddings = [512, 1024, 2048, 4096, 8192]
    runner.pcp_manager = (
        None if cap is None else SimpleNamespace(max_local_num_tokens=cap)
    )
    runner._pp_wave = SimpleNamespace(settle=lambda: None)
    runner.num_reqs_max_model_len = 1
    runner.max_num_blocks_per_req = 1
    runner.most_model_len = None
    runner.ran = []
    runner._dummy_run = lambda num_tokens, *a, **k: runner.ran.append(num_tokens)
    return runner


def test_handoff_warmup_stops_at_the_pcp_local_cap(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_device", lambda: None)
    runner = _runner(cap=2048)
    runner._warmup_handoff_forwards()
    assert runner.ran == [512, 1024, 2048]


def test_handoff_warmup_runs_the_full_ladder_without_pcp(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_device", lambda: None)
    runner = _runner(cap=None)
    runner._warmup_handoff_forwards()
    assert runner.ran == [512, 1024, 2048, 4096, 8192]


def _capture_runner(monkeypatch, cap):
    runner = _runner(cap)
    runner._pp_wave = None
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    runner.max_num_tokens = 8192
    runner.device = "cpu"
    runner.enforce_eager = True
    runner._pp_topk_buffer = None
    runner._pp_intermediate_template = {}
    runner._add_pipeline_chunk_buckets = lambda: None
    runner._pp_intermediate_tensors = lambda *a, **k: None
    built = []

    class _Wave:
        def __init__(self, device, max_rows, template):
            built.append(max_rows)

        def warmup(self):
            pass

    monkeypatch.setattr(tpu_runner, "PPWave", _Wave)
    runner.capture_model()
    return built


def test_handoff_buffer_is_the_pcp_local_cap(monkeypatch):
    """Each device hands off only its own rows, so the buffer stops at the
    largest local bucket rather than the global step."""
    assert _capture_runner(monkeypatch, cap=1100) == [2048]


def test_handoff_buffer_is_the_full_step_without_pcp(monkeypatch):
    assert _capture_runner(monkeypatch, cap=None) == [8192]
