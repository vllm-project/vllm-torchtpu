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
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.multimodal.utils import set_mm_embedding_modality
from vllm.sequence import IntermediateTensors
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

from vllm_torchtpu.omni.worker.tpu_ar_model_runner import (
    TPUARModelRunner,
    _expand_mm_dummy_batches,
)
from vllm_torchtpu.omni.worker.tpu_model_runner import OmniTPUModelRunner
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
    for name, value in (defaults | fields).items():
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


def test_tpu_ar_model_runner_mixed_dim_mm_slicing(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    """Verify _gather_mm_embeddings and _get_model_inputs preserve mixed widths."""
    vid_emb = set_mm_embedding_modality(torch.ones(4, 32), "video")
    aud_emb = set_mm_embedding_modality(torch.full((3, 8), 2.0), "audio")
    mask = torch.ones(10, dtype=torch.bool)

    with patch.object(
        TPUModelRunner,
        "_gather_mm_embeddings",
        return_value=([vid_emb, aud_emb], mask),
    ):
        idx_list, out_mask = tpu_ar_model_runner._gather_mm_embeddings(
            SimpleNamespace()
        )

    assert out_mask is mask
    assert len(idx_list) == 1
    assert torch.equal(idx_list[0], torch.arange(7, dtype=torch.long))

    # Simulate TPUModelRunner.execute_model slicing row indices [2:6] across the
    # video->audio boundary (last 2 rows of video + first 2 rows of audio).
    captured_mm: list[Any] = []

    def _fake_parent_get_model_inputs(self, input_ids, mm_embed_inputs):
        captured_mm.append(mm_embed_inputs)
        return None, input_ids

    with patch.object(
        TPUModelRunner, "_get_model_inputs", _fake_parent_get_model_inputs
    ):
        tpu_ar_model_runner._get_model_inputs(
            torch.zeros(10, dtype=torch.long),
            ([idx_list[0][2:6]], mask),
        )

    resolved_embeds, resolved_mask = captured_mm[0]
    assert resolved_mask is mask
    assert len(resolved_embeds) == 2
    assert resolved_embeds[0].shape == (2, 32)
    assert resolved_embeds[0].modality == "video"
    assert resolved_embeds[1].shape == (2, 8)
    assert resolved_embeds[1].modality == "audio"


def test_tpu_ar_model_runner_capture_model_precompiles_mm_buckets(
    tpu_ar_model_runner: TPUARModelRunner,
) -> None:
    runner = tpu_ar_model_runner
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

    def _embed_multimodal(**kwargs: Any) -> tuple[torch.Tensor, ...]:
        modality = kwargs["modality"]
        seen_modalities.append(modality)
        if modality == "empty":
            return ()
        width = 8 if modality == "audio" else 32
        return (torch.ones(4, width),)

    def _embed_input_ids(
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        *,
        is_multimodal: Any = None,
    ) -> torch.Tensor:
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

    # Verify _expand_mm_dummy_batches expands audio and image dummy inputs
    # across all static buckets in _AUDIO_CHUNK_BUCKETS and _VISION_PATCH_BUCKETS.
    audio_batches = _expand_mm_dummy_batches(
        "audio",
        {
            "input_features": torch.zeros(128, 3000),
            "audio_feature_lengths": torch.tensor([3000], dtype=torch.long),
        },
    )
    assert len(audio_batches) == 1
    assert audio_batches[0]["input_features"].shape == (128, 800 + 1600 + 3200)
    assert audio_batches[0]["audio_feature_lengths"].tolist() == [800, 1600, 3200]

    image_batches = _expand_mm_dummy_batches(
        "image",
        {
            "pixel_values": torch.zeros(16384, 16),
            "image_grid_thw": torch.tensor([[1, 128, 128]], dtype=torch.int32),
        },
    )
    assert [int(b["pixel_values"].shape[0]) for b in image_batches] == [
        4096,
        8192,
        16384,
    ]
    assert [b["image_grid_thw"].tolist() for b in image_batches] == [
        [[1, 64, 64]],
        [[1, 64, 128]],
        [[1, 64, 256]],
    ]
