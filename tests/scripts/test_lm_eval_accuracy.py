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
"""Unit tests for scripts/vllm/integration/lm_eval_accuracy.py."""

import importlib.util
import pathlib
import sys
import types

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "integration"
    / "lm_eval_accuracy.py"
)
_spec = importlib.util.spec_from_file_location("lm_eval_accuracy", _SCRIPT)
lm_eval_accuracy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm_eval_accuracy)


def _run(monkeypatch, **evaluation_args):
    calls = {"shutdown": 0}

    def simple_evaluate(**kwargs):
        calls["kwargs"] = kwargs
        return {"results": {}}

    def shutdown():
        calls["shutdown"] += 1

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            llm_engine=types.SimpleNamespace(
                engine_core=types.SimpleNamespace(shutdown=shutdown)
            )
        )
    )
    monkeypatch.setitem(
        sys.modules, "lm_eval", types.SimpleNamespace(simple_evaluate=simple_evaluate)
    )
    monkeypatch.setattr(lm_eval_accuracy, "_create_vllm_model", lambda *a: model)
    lm_eval_accuracy.evaluate_with_vllm(
        model_args="pretrained=m", tasks="gsm8k", **evaluation_args
    )
    return calls


def test_torch_seed_skipped_by_default(monkeypatch):
    calls = _run(monkeypatch)
    assert calls["kwargs"]["torch_random_seed"] is None
    assert calls["shutdown"] == 1


def test_explicit_torch_seed_passed_through(monkeypatch):
    calls = _run(monkeypatch, torch_random_seed=7)
    assert calls["kwargs"]["torch_random_seed"] == 7
