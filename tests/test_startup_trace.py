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
"""Unit tests for startup Perfetto trace slices and fingerprint dumps."""

import pytest
import torch

from vllm_torchtpu import startup_trace
from vllm_torchtpu.compilation.graph_fingerprint import _graph_fingerprint


@pytest.fixture(autouse=True)
def _reset_startup_trace():
    startup_trace._installed.clear()
    startup_trace._active[0] = True
    yield
    startup_trace._installed.clear()
    startup_trace._active[0] = True


def test_startup_trace_wraps_and_stops_after_compile_or_warm_up(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_TPU_STARTUP_TRACE", "1")
    monkeypatch.setenv("VLLM_TPU_STARTUP_TRACE_ROLE", "decode")
    monkeypatch.setenv("RANK", "3")

    class DummyWorker:
        def load_model(self):
            return "loaded"

        def compile_or_warm_up_model(self):
            return "warmed"

    startup_trace.install_worker(DummyWorker)
    w = DummyWorker()
    assert w.load_model() == "loaded"
    assert w.compile_or_warm_up_model() == "warmed"
    # Subsequent calls after compile_or_warm_up_model returns emit nothing.
    assert w.load_model() == "loaded"

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 4
    assert "trace-start-event: name=worker.load_model" in out[0]
    assert "pid=vllm-decode tid=rank3" in out[0]
    assert "trace-end-event: name=worker.load_model" in out[1]
    assert "trace-start-event: name=worker.compile_or_warm_up_model" in out[2]
    assert "trace-end-event: name=worker.compile_or_warm_up_model" in out[3]


def test_fingerprint_dump_writes_for_selected_ranks(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_TPU_FINGERPRINT_DUMP", str(tmp_path))
    monkeypatch.setenv("VLLM_TPU_FINGERPRINT_DUMP_RANKS", "2")

    gm = torch.fx.symbolic_trace(lambda x: x + 1)

    monkeypatch.setenv("LOCAL_RANK", "0")
    fp0 = _graph_fingerprint(gm)
    assert fp0 is not None
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv("LOCAL_RANK", "2")
    fp2 = _graph_fingerprint(gm)
    assert fp2 == fp0
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == f"rank2-{fp2[:16]}.txt"
    assert "### module" in files[0].read_text()
