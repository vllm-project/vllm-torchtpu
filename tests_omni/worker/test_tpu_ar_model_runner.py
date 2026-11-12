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
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni


def _stage_config(**fields) -> OmniModelConfig:
    defaults = dict(
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


def _build(model_config=None, model=None, speculative_config=None) -> TPUARModelRunner:
    if model_config is None:
        model_config = _stage_config()

    def _fake_init(self, vllm_config, *args, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config

    with patch.object(TPUModelRunner, "__init__", _fake_init):
        runner = TPUARModelRunner(
            SimpleNamespace(
                model_config=model_config,
                speculative_config=speculative_config,
            ),
            "tpu",
        )
    if model is not None:
        with (
            patch(
                "vllm_torchtpu.omni.worker.tpu_ar_model_runner."
                "maybe_patch_qwen3_omni_thinker"
            ) as mock_patch,
            patch.object(
                TPUModelRunner, "load_model", lambda self: setattr(self, "model", model)
            ),
        ):
            runner.load_model()
            mock_patch.assert_called_once_with(runner.model_config)
    return runner


class _StageModel:
    def __init__(self, result, *, has_preprocess=False, has_postprocess=False):
        self.result = result
        self.has_preprocess = has_preprocess
        self.has_postprocess = has_postprocess
        self.wrapped = []

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.result

    def make_omni_output(self, model_outputs, **kwargs) -> OmniOutput:
        self.wrapped.append(model_outputs)
        hidden = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"embed": {}})


def test_tpu_ar_model_runner_init_and_inheritance():
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert OmniConnectorModelRunnerMixin not in OmniTPUModelRunner.__mro__

    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    assert (runner.model_stage, runner.stage_id) == ("thinker", 0)
    plain = _build(SimpleNamespace())
    assert (plain.model_stage, plain.stage_id) == (None, None)

    with pytest.raises(NotImplementedError, match="connectors"):
        _build(
            _stage_config(custom_process_next_stage_input_func="pkg.mod.thinker2talker")
        )

    with pytest.raises(NotImplementedError, match="speculative_config"):
        _build(speculative_config=SimpleNamespace(method="eagle3"))


def test_load_model_checks_per_step_hook_capabilities():
    thinker = _StageModel(torch.zeros(1, 4))
    assert _build(model=thinker).get_model() is thinker

    talker = _StageModel(torch.zeros(1, 4), has_preprocess=True, has_postprocess=True)
    with pytest.raises(NotImplementedError, match="has_preprocess, has_postprocess"):
        _build(model=talker)


def test_forward_model_normalizes_stage_outputs():
    ids, pos = torch.tensor([1, 2]), torch.tensor([0, 1])
    hidden = torch.arange(8.0).reshape(2, 4)
    staged = (hidden, {"layer_0": torch.zeros(2, 4)})

    for raw, expected_wrapped in [
        (hidden, [hidden]),
        ([hidden], [hidden]),
        (staged, [staged]),
        (OmniOutput(text_hidden_states=hidden), []),
    ]:
        runner = _build(model=(model := _StageModel(raw)))
        assert runner.forward_model(ids, pos) == (hidden, None)
        assert model.wrapped == expected_wrapped
        assert "intermediate_tensors" not in model.kwargs

    tensors = IntermediateTensors({"hidden_states": hidden})
    runner = _build(model=(model := _StageModel(tensors)))
    assert runner.forward_model(ids, pos, intermediate_tensors=tensors) == (
        tensors,
        None,
    )
    assert model.kwargs["intermediate_tensors"] is tensors


def test_tpu_ar_model_runner_mixed_dim_mm_slicing():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.supports_mm_inputs = True

    vid_emb, aud_emb = torch.ones(6, 32), torch.ones(4, 8)
    vid_emb.modality, aud_emb.modality = "video", "audio"
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        OmniTPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = runner._gather_mm_embeddings("sched_out")
    assert len(idx_list) == 1 and torch.equal(idx_list[0], torch.arange(10))
    assert out_mask is mask

    captured_mm = []

    def _dummy_embed_input_ids(ids, **kwargs):
        captured_mm.append(kwargs.get("multimodal_embeddings"))
        return torch.zeros(ids.shape[0], 8)

    runner.model = SimpleNamespace(embed_input_ids=_dummy_embed_input_ids)
    _, embeds_out = runner._get_model_inputs(
        torch.zeros(5, dtype=torch.long),
        ([torch.arange(4, 9, dtype=torch.int64)], torch.ones(5, dtype=torch.bool)),
    )
    assert embeds_out.shape == (5, 8) and len(captured_mm[0]) == 2
    assert captured_mm[0][0].shape == (2, 32)
    assert captured_mm[0][0].modality == "video"
    assert captured_mm[0][1].shape == (3, 8)
    assert captured_mm[0][1].modality == "audio"
    assert len(runner._omni_mm_embeds_list) == 2

    # Slicing through the final embedding row clears _omni_mm_embeds_list.
    runner._get_model_inputs(
        torch.zeros(1, dtype=torch.long),
        ([torch.arange(9, 10, dtype=torch.int64)], torch.ones(1, dtype=torch.bool)),
    )
    assert runner._omni_mm_embeds_list == []


def test_tpu_ar_model_runner_capture_model_precompiles_mm_buckets():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    runner.device = torch.device("cpu")
    runner.enforce_eager = False
    runner.supports_mm_inputs = True
    runner._pp_is_first = True
    runner.num_tokens_paddings = [16, 64]
    runner._update_num_xla_graphs = MagicMock()

    seen_modalities: list[str] = []
    seen_merges: list[tuple[int, tuple[tuple[int, ...], str] | None]] = []
    reset_cache = MagicMock()
    runner.mm_budget = SimpleNamespace(
        mm_max_toks_per_item={"video": 65536, "audio": 390, "image": 4096},
        reset_cache=reset_cache,
    )
    runner._get_mm_dummy_batch = lambda modality, count: {"modality": modality}

    def _embed_multimodal(**kwargs):
        modality = kwargs["modality"]
        seen_modalities.append(modality)
        if modality == "empty":
            return ()
        width = 8 if modality == "audio" else 32
        return (torch.ones(4, width),)

    def _embed_input_ids(input_ids, multimodal_embeddings=None, *, is_multimodal=None):
        mm_info = (
            None
            if multimodal_embeddings is None
            else (
                tuple(multimodal_embeddings[0].shape),
                multimodal_embeddings[0].modality,
            )
        )
        seen_merges.append((int(input_ids.shape[0]), mm_info))
        return torch.zeros(input_ids.shape[0], 8)

    runner.model = SimpleNamespace(
        embed_multimodal=_embed_multimodal,
        embed_input_ids=_embed_input_ids,
    )

    with patch.object(OmniTPUModelRunner, "capture_model") as mock_super_capture:
        runner.capture_model()
        mock_super_capture.assert_called_once_with()
        assert seen_modalities == ["audio", "image"]
        reset_cache.assert_called_once_with()
        assert seen_merges == [
            (16, ((1, 8), "audio")),
            (64, ((1, 8), "audio")),
            (16, ((1, 32), "image")),
            (64, ((1, 32), "image")),
            (16, None),
        ]

        # Empty embed_multimodal logs warning_once and skips merge warmup
        seen_modalities.clear()
        seen_merges.clear()
        runner.mm_budget.mm_max_toks_per_item = {"empty": 100}
        with patch(
            "vllm_torchtpu.omni.worker.tpu_ar_model_runner.logger.warning_once"
        ) as mock_warn:
            runner.capture_model()
            mock_warn.assert_called_once()
        assert seen_modalities == ["empty"]
        assert seen_merges == []

        # Non-MM, enforce_eager, or non-first PP stage skips MM precompilation
        runner.mm_budget.mm_max_toks_per_item = {"audio": 390}
        for attr in ("supports_mm_inputs", "_pp_is_first"):
            seen_modalities.clear()
            seen_merges.clear()
            setattr(runner, attr, False)
            runner.capture_model()
            assert seen_modalities == []
            assert seen_merges == []
            setattr(runner, attr, True)

        seen_modalities.clear()
        seen_merges.clear()
        runner.enforce_eager = True
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []
