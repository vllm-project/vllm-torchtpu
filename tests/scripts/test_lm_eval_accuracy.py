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


def _install_fake_vllm(monkeypatch):
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    entrypoints_mod = types.ModuleType("vllm.entrypoints")
    entrypoints_mod.__path__ = []
    vllm_mod.entrypoints = entrypoints_mod

    renderers_mod = types.ModuleType("vllm.renderers")
    renderers_mod.__path__ = []
    renderers_chat_utils = types.ModuleType("vllm.renderers.chat_utils")
    renderers_mod.chat_utils = renderers_chat_utils
    hf_mod = types.ModuleType("vllm.renderers.hf")

    def resolve_chat_template(
        tokenizer, chat_template=None, tools=None, *, model_config=None
    ):
        return ("upstream_resolve_chat_template", tokenizer, chat_template)

    hf_mod.resolve_chat_template = resolve_chat_template
    renderers_mod.hf = hf_mod
    vllm_mod.renderers = renderers_mod

    engine_mod = types.ModuleType("vllm.engine")
    engine_mod.__path__ = []
    arg_utils_mod = types.ModuleType("vllm.engine.arg_utils")

    class FakeEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    arg_utils_mod.EngineArgs = FakeEngineArgs
    engine_mod.arg_utils = arg_utils_mod
    vllm_mod.engine = engine_mod

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints_mod)
    monkeypatch.delitem(sys.modules, "vllm.entrypoints.chat_utils", raising=False)
    monkeypatch.setitem(sys.modules, "vllm.renderers", renderers_mod)
    monkeypatch.setitem(sys.modules, "vllm.renderers.chat_utils", renderers_chat_utils)
    monkeypatch.setitem(sys.modules, "vllm.renderers.hf", hf_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils_mod)
    return entrypoints_mod, renderers_chat_utils, resolve_chat_template, FakeEngineArgs


def _install_fake_lm_eval_vllm(monkeypatch):
    lm_eval_mod = types.ModuleType("lm_eval")
    lm_eval_mod.__path__ = []
    models_mod = types.ModuleType("lm_eval.models")
    models_mod.__path__ = []
    causallms_mod = types.ModuleType("lm_eval.models.vllm_causallms")

    class FakeVLLM:
        @staticmethod
        def create_from_arg_string(model_args, additional_config):
            from vllm.entrypoints.chat_utils import resolve_hf_chat_template

            return {
                "model_args": model_args,
                "additional_config": additional_config,
                "resolve_hf_chat_template": resolve_hf_chat_template,
            }

    causallms_mod.VLLM = FakeVLLM
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models.vllm_causallms", causallms_mod)


def test_create_vllm_model_wires_upstream_chat_utils_and_engine_args(monkeypatch):
    entrypoints_mod, renderers_chat_utils, resolve_chat_template, engine_args_cls = (
        _install_fake_vllm(monkeypatch)
    )
    _install_fake_lm_eval_vllm(monkeypatch)

    result = lm_eval_accuracy._create_vllm_model("pretrained=m", "auto", None, None)

    assert sys.modules["vllm.entrypoints.chat_utils"] is renderers_chat_utils
    assert entrypoints_mod.chat_utils is renderers_chat_utils
    assert result["resolve_hf_chat_template"] is resolve_chat_template
    assert engine_args_cls(model="m", swap_space=4).kwargs == {"model": "m"}
