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
"""Unit tests for src/vllm_torchtpu/omni/worker/tpu_ar_model_runner.py.

``OmniTPUModelRunner`` only records the stage it serves, so it is covered
through the AR runner here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.omni

# The connector fields needs_omni_connector() looks at, all off.
_SINGLE_STAGE = dict(
    requires_full_payload_input=False,
    custom_process_next_stage_input_func=None,
    stage_connector_config=None,
)


def _fake_base_init(self, vllm_config, *args, **kwargs):
    """Stand-in for TPUModelRunner.__init__: only the config plumbing."""
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config


def _stage_config(**fields) -> OmniModelConfig:
    config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {**_SINGLE_STAGE, **fields}.items():
        object.__setattr__(config, name, value)
    return config


def _build(model_config) -> TPUARModelRunner:
    with patch.object(TPUModelRunner, "__init__", _fake_base_init):
        return TPUARModelRunner(SimpleNamespace(model_config=model_config), "tpu")


def test_tpu_ar_model_runner_inheritance():
    # Same shape as upstream GPUARModelRunner: the connector mixin sits on the
    # leaf runner, not on the shared OmniTPUModelRunner base.
    assert TPUARModelRunner.__bases__ == (
        OmniTPUModelRunner,
        OmniConnectorModelRunnerMixin,
    )
    assert OmniConnectorModelRunnerMixin not in OmniTPUModelRunner.__mro__


def test_thinker_stage_is_accepted():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    assert runner.model_stage == "thinker"
    assert runner.stage_id == 0


def test_plain_vllm_config_is_accepted():
    # No stage metadata at all: a single-stage deployment outside vLLM-Omni.
    runner = _build(SimpleNamespace())
    assert runner.model_stage is None
    assert runner.stage_id is None


def test_talker_stage_is_refused_at_construction():
    with pytest.raises(NotImplementedError, match="talker_mtp"):
        _build(_stage_config(model_stage="talker", stage_id=1))


def test_multi_stage_thinker_is_refused_at_construction():
    # The thinker of a multi-stage pipeline hands its output to the next stage
    # (custom_process_next_stage_input_func is set), which needs connectors.
    with pytest.raises(NotImplementedError, match="connectors"):
        _build(
            _stage_config(
                model_stage="thinker",
                stage_id=0,
                custom_process_next_stage_input_func="pkg.mod.thinker2talker",
            )
        )


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
            (16, ((1, 32), "image")),
            (16, None),
        ]

        # Non-MM stage skips MM encoder precompilation
        seen_modalities.clear()
        seen_merges.clear()
        runner.supports_mm_inputs = False
        runner.capture_model()
        assert seen_modalities == []
        assert seen_merges == []


def test_tpu_ar_model_runner_forward_model_unwraps_omni_output():
    runner = _build(_stage_config(model_stage="thinker", stage_id=0))
    expected_hs = torch.ones(3, 8)
    omni_out = OmniOutput(text_hidden_states=expected_hs)
    with patch.object(
        OmniTPUModelRunner, "forward_model", return_value=(omni_out, None)
    ):
        hs, aux = runner.forward_model(
            torch.zeros(3, dtype=torch.long), torch.arange(3)
        )
    assert hs is expected_hs
    assert aux is None
