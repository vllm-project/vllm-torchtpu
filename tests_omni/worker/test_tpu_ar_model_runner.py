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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_ar_model_runner.py."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni


def _stage_config(**fields: Any) -> OmniModelConfig:
    defaults = dict(
        model_arch_config=SimpleNamespace(
            architectures=["Qwen3OmniMoeForConditionalGeneration"]
        ),
        model_stage="thinker",
        stage_id=0,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
        stage_connector_config=None,
    )
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**defaults, **fields}.items():
        object.__setattr__(config, name, value)
    return config


@pytest.fixture(autouse=True)
def _stub_tpu_model_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_init(self: Any, vllm_config: Any, *args: Any, **kwargs: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config

    monkeypatch.setattr(TPUModelRunner, "__init__", _fake_init)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.worker.tpu_ar_model_runner.maybe_patch_qwen3_omni_thinker",
        lambda *_: False,
    )


@pytest.fixture
def tpu_ar_model_runner(request: pytest.FixtureRequest) -> TPUARModelRunner:
    model_config = getattr(request, "param", None)
    if model_config is None:
        model_config = _stage_config()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
    )
    return TPUARModelRunner(vllm_config, "tpu")


class _StageModel:
    def __init__(
        self,
        result: Any,
        *,
        has_preprocess: bool = False,
        has_postprocess: bool = False,
    ):
        self.result = result
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.wrapped: list[Any] = []
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.result

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        self.wrapped.append(model_outputs)
        hidden = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"embed": {}})


@pytest.mark.parametrize(
    ("tpu_ar_model_runner", "expected"),
    [
        (_stage_config(model_stage="thinker", stage_id=0), ("thinker", 0)),
        (SimpleNamespace(architectures=[]), (None, None)),
    ],
    indirect=["tpu_ar_model_runner"],
    ids=["omni_stage_config", "plain_vllm_config"],
)
def test_tpu_ar_model_runner_stage_metadata(
    tpu_ar_model_runner: TPUARModelRunner,
    expected: tuple[str | None, int | None],
) -> None:
    assert (tpu_ar_model_runner.model_stage, tpu_ar_model_runner.stage_id) == expected


@pytest.mark.parametrize(
    ("model_config", "speculative_config", "match"),
    [
        (
            _stage_config(
                custom_process_next_stage_input_func="pkg.mod.thinker2talker"
            ),
            None,
            "connectors",
        ),
        (
            _stage_config(),
            SimpleNamespace(method="eagle3"),
            "speculative_config",
        ),
    ],
    ids=["connectors", "speculative_config"],
)
def test_tpu_ar_model_runner_init_rejects_unsupported_config(
    model_config: Any,
    speculative_config: Any,
    match: str,
) -> None:
    vllm_config = SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
    )
    with pytest.raises(NotImplementedError, match=match):
        TPUARModelRunner(vllm_config, "tpu")


@pytest.mark.parametrize(
    ("has_preprocess", "has_postprocess", "error_match"),
    [
        (False, False, None),
        (True, False, "has_preprocess"),
        (False, True, "has_postprocess"),
        (True, True, "has_preprocess, has_postprocess"),
    ],
    ids=["thinker", "preprocess_only", "postprocess_only", "talker_both"],
)
def test_load_model_checks_per_step_hook_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tpu_ar_model_runner: TPUARModelRunner,
    has_preprocess: bool,
    has_postprocess: bool,
    error_match: str | None,
) -> None:
    model = _StageModel(
        torch.zeros(1, 4),
        has_preprocess=has_preprocess,
        has_postprocess=has_postprocess,
    )
    monkeypatch.setattr(
        TPUModelRunner,
        "load_model",
        lambda self: setattr(self, "model", model),
    )
    if error_match is None:
        tpu_ar_model_runner.load_model()
        assert tpu_ar_model_runner.get_model() is model
    else:
        with pytest.raises(NotImplementedError, match=error_match):
            tpu_ar_model_runner.load_model()


_HIDDEN = torch.arange(8.0).reshape(2, 4)
_STAGED = (_HIDDEN, {"layer_0": torch.zeros(2, 4)})
_INTERMEDIATE = IntermediateTensors({"hidden_states": _HIDDEN})


@pytest.mark.parametrize(
    ("raw_output", "intermediate_tensors", "expected_out", "expected_wrapped"),
    [
        (_HIDDEN, None, _HIDDEN, [_HIDDEN]),
        ([_HIDDEN], None, _HIDDEN, [_HIDDEN]),
        (_STAGED, None, _HIDDEN, [_STAGED]),
        (OmniOutput(text_hidden_states=_HIDDEN), None, _HIDDEN, []),
        (_INTERMEDIATE, _INTERMEDIATE, _INTERMEDIATE, []),
    ],
    ids=[
        "tensor",
        "single_elem_list",
        "staged_tuple",
        "omni_output",
        "intermediate_tensors",
    ],
)
def test_forward_model_normalizes_stage_outputs(
    tpu_ar_model_runner: TPUARModelRunner,
    raw_output: Any,
    intermediate_tensors: IntermediateTensors | None,
    expected_out: Any,
    expected_wrapped: list[Any],
) -> None:
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    model = _StageModel(raw_output)
    tpu_ar_model_runner.model = model

    out, aux = tpu_ar_model_runner.forward_model(
        ids, pos, intermediate_tensors=intermediate_tensors
    )
    assert out is expected_out
    assert aux is None
    assert model.wrapped == expected_wrapped
    if intermediate_tensors is None:
        assert "intermediate_tensors" not in model.kwargs
    else:
        assert model.kwargs["intermediate_tensors"] is intermediate_tensors
