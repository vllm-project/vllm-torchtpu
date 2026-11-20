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
"""CPU unit tests for Muse Spark Open multimodal support."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.model_executor.models.utils as vllm_utils
import vllm.models.muse_spark.common.omni as up_omni
from PIL import Image
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.models.muse_spark.common.audio import AutoEncoder
from vllm.models.muse_spark.common.omni import MuseSpark1xOmniMixin
from vllm.models.muse_spark.common.processor import preprocess_image
from vllm.models.muse_spark.common.vision import NaViT
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataParser
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.models.vllm.test_muse_spark_text import (
    _use_cpu_platform,
    build_synthetic_hf_weights,
    make_tiny_omni_config_dict,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
)
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.muse_spark.mm_processing import (
    MAX_AUDIO_SECONDS,
    TPUMuseSparkOmniMultiModalProcessor,
    TPUMuseSparkOmniProcessingInfo,
    _check_media_dimensions,
)
from vllm_torchtpu.models.vllm.muse_spark.model import (
    TPUMuseSparkModel,
    TPUMuseSparkTextOnlyForConditionalGeneration,
)
from vllm_torchtpu.models.vllm.muse_spark.multimodal import (
    TPUMuseSparkForConditionalGeneration,
)


def _make_mm_vllm_config(
    tmp_path: Path,
    *,
    limit_per_prompt: dict[str, int] | None = None,
) -> VllmConfig:
    import json

    cfg_dict = make_tiny_omni_config_dict()
    model_dir = tmp_path / "mm_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(cfg_dict))
    register_models()
    limits = (
        limit_per_prompt
        if limit_per_prompt is not None
        else {"image": 2, "video": 1, "audio": 2}
    )
    model_config = ModelConfig(
        model=str(model_dir),
        tokenizer_mode="skip",
        dtype=torch.bfloat16,
        max_model_len=8192,
        limit_mm_per_prompt=limits,
    )
    vc = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=128, cache_dtype="auto"),
        parallel_config=ParallelConfig(tensor_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
        quant_config=VllmUnquantizedConfig(),
    )
    VllmQuantConfig.set_configs(vc)
    return vc


def test_registry_resolves_tpu_mm_class_and_retains_rollback() -> None:
    """ModelRegistry resolves to TPUMuseSparkForConditionalGeneration."""
    register_models()
    resolved = ModelRegistry.models[
        "MuseSparkForConditionalGeneration"
    ].load_model_cls()
    assert resolved is TPUMuseSparkForConditionalGeneration
    assert issubclass(TPUMuseSparkTextOnlyForConditionalGeneration, torch.nn.Module)


def test_embed_input_ids_uses_static_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """TPUMuseSparkForConditionalGeneration.embed_input_ids uses static merge."""
    _use_cpu_platform(monkeypatch, request)
    assert (
        up_omni._merge_multimodal_embeddings is vllm_utils._merge_multimodal_embeddings
    )
    orig = getattr(vllm_utils, "_original_merge_multimodal_embeddings", None)
    assert callable(orig)

    vc = _make_mm_vllm_config(
        tmp_path, limit_per_prompt={"image": 0, "video": 0, "audio": 0}
    )
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        model = TPUMuseSparkForConditionalGeneration(vllm_config=vc)

    torch.manual_seed(9)
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_(0.0, 0.02)

    hidden_size = vc.model_config.hf_config.text_config.hidden_size
    input_ids = torch.arange(10, dtype=torch.long)
    mm_embeds = [
        torch.randn(3, hidden_size, dtype=torch.bfloat16),
        torch.randn(2, hidden_size, dtype=torch.bfloat16),
    ]
    is_mm = torch.tensor(
        [False, True, True, True, False, False, True, True, False, False]
    )

    called_static: list[bool] = []
    static_merge = up_omni._merge_multimodal_embeddings

    def _spy_merge(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: object,
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        called_static.append(True)
        return static_merge(inputs_embeds, multimodal_embeddings, is_multimodal)

    monkeypatch.setattr(up_omni, "_merge_multimodal_embeddings", _spy_merge)
    base_embeds = model.embed_input_ids(input_ids)
    got = model.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_mm)
    ref = orig(base_embeds.clone(), mm_embeds, is_mm)
    assert called_static == [True]
    assert torch.equal(got, ref)


def test_mm_class_builds_on_tpu_text_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """TPUMuseSparkForConditionalGeneration composes Omni towers over TPU text stack."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_mm_vllm_config(tmp_path)
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        model = TPUMuseSparkForConditionalGeneration(vllm_config=vc)

    assert isinstance(model, MuseSpark1xOmniMixin)
    assert isinstance(model.model, TPUMuseSparkModel)
    assert isinstance(model.vision_model, NaViT)
    assert isinstance(model.audio_encoder, AutoEncoder)


def test_zero_mm_limits_skip_towers_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Zero MM limits turn towers into StageMissingLayer and skip tower weights."""
    _use_cpu_platform(monkeypatch, request)
    vc = _make_mm_vllm_config(
        tmp_path, limit_per_prompt={"image": 0, "video": 0, "audio": 0}
    )
    with set_default_torch_dtype(torch.bfloat16), set_current_vllm_config(vc):
        model = TPUMuseSparkForConditionalGeneration(vllm_config=vc)

    assert isinstance(model.vision_model, StageMissingLayer)
    assert isinstance(model.audio_encoder, StageMissingLayer)

    weights = build_synthetic_hf_weights(
        vc.model_config.hf_config.text_config, include_towers=True
    )
    loaded = model.load_weights(weights.items())
    expected = {name for name, _ in model.named_parameters()} | {
        "model.layers.1.feed_forward.expert_bias"
    }
    assert loaded == expected


@pytest.mark.parametrize(
    ("height", "width"),
    [(10, 50_000), (50_000, 10), (1, 10_000), (14, 14), (3, 5000)],
)
def test_resize_extreme_aspect_ratios(tmp_path: Path, height: int, width: int) -> None:
    """Extreme aspect-ratio images resize within [1, 4096] tokens cleanly."""
    vc = _make_mm_vllm_config(tmp_path)
    proc = MULTIMODAL_REGISTRY.create_processor(
        vc.model_config, tokenizer=SimpleNamespace(get_vocab=dict)
    )
    assert isinstance(proc, TPUMuseSparkOmniMultiModalProcessor)
    pre_cfg = proc.info.preprocess_config

    # All parametrized shapes stay <= 1.5 MB RGB while exercising extreme aspect ratios.
    img = Image.new("RGB", (width, height))
    tensor, num_tokens = preprocess_image(img, pre_cfg)
    assert 1 <= num_tokens <= pre_cfg.max_num_image_tokens
    assert tensor.shape[-1] % 28 == 0
    assert tensor.shape[-2] % 28 == 0


def test_audio_600s_cap_and_empty_guards(tmp_path: Path) -> None:
    """Audio budget is 7,500 tokens (600 s); cap-length clips fit; empty rejected."""
    vc = _make_mm_vllm_config(tmp_path)
    proc = MULTIMODAL_REGISTRY.create_processor(
        vc.model_config, tokenizer=SimpleNamespace(get_vocab=dict)
    )
    assert isinstance(proc, TPUMuseSparkOmniMultiModalProcessor)
    assert isinstance(proc.info, TPUMuseSparkOmniProcessingInfo)

    assert proc.info.max_audio_tokens() == 7500
    max_tokens = proc.info.get_mm_max_tokens_per_item(
        8192, {"image": 1, "video": 1, "audio": 1}
    )
    assert max_tokens["audio"] == 7500

    parser = MultiModalDataParser(target_sr=proc.info.audio_sample_rate)

    # 1. Zero-dimension image raises ValueError (not ZeroDivisionError).
    zero_img_items = parser.parse_mm_data({"image": [Image.new("RGB", (0, 64))]})
    with pytest.raises(ValueError, match="empty image"):
        proc._apply_hf_processor_main(zero_img_items, {})

    # 2. Empty audio waveform raises ValueError.
    empty_audio_items = parser.parse_mm_data(
        {"audio": [np.zeros((0,), dtype=np.float32)]}
    )
    with pytest.raises(ValueError, match="empty audio"):
        proc._apply_hf_processor_main(empty_audio_items, {})

    # 3. Cap-length audio (600 s) fits exactly within the 7,500-token cap.
    cap_samples = MAX_AUDIO_SECONDS * proc.info.audio_sample_rate
    cap_audio_items = parser.parse_mm_data(
        {"audio": [np.zeros((cap_samples,), dtype=np.float32)]}
    )
    cap_out = proc._apply_hf_processor_main(cap_audio_items, {})
    assert list(cap_out["audio_token_counts"]) == [7500]

    # 4. Multi-clip prompts (2+ audios) succeed without ambiguous truthiness errors.
    two_clips = parser.parse_mm_data(
        {
            "audio": [
                np.zeros((1920,), dtype=np.float32),
                np.zeros((3840,), dtype=np.float32),
            ]
        }
    )
    two_out = proc._apply_hf_processor_main(two_clips, {})
    assert list(two_out["audio_token_counts"]) == [1, 2]

    # 5. Audio exceeding 600 s raises ValueError.
    over_cap_samples = MAX_AUDIO_SECONDS * proc.info.audio_sample_rate + 1920
    over_audio_items = parser.parse_mm_data(
        {"audio": [np.zeros((over_cap_samples,), dtype=np.float32)]}
    )
    with pytest.raises(ValueError, match="exceeding the 7500-token"):
        proc._apply_hf_processor_main(over_audio_items, {})


def test_check_media_dimensions_hwc_images_and_sequence_videos() -> None:
    """Dimension guard rejects zero-extent HWC/CHW images and sequence videos."""

    class _FakeItems:
        def __init__(self, data: dict[str, object]) -> None:
            self._data = data

        def get_processor_data(self) -> dict[str, object]:
            return self._data

    class _FakeMMItems:
        def __init__(self, data: dict[str, object]) -> None:
            self._items = {"media": _FakeItems(data)}

        def get_all_counts(self) -> dict[str, int]:
            return {"media": 1}

        def select(self, keys: set[str]) -> dict[str, _FakeItems]:
            return {k: v for k, v in self._items.items() if k in keys}

    for bad_img in (
        np.zeros((0, 16, 3), dtype=np.uint8),
        np.zeros((16, 0, 3), dtype=np.uint8),
        np.zeros((3, 0, 16), dtype=np.uint8),
    ):
        with pytest.raises(ValueError, match="empty image"):
            _check_media_dimensions(_FakeMMItems({"images": [bad_img]}))  # type: ignore[arg-type]

    for bad_video, pattern in (
        (np.zeros((0, 16, 16, 3), dtype=np.uint8), "empty video input"),
        (np.zeros((2, 0, 16, 3), dtype=np.uint8), "empty video frame"),
        ([], "empty video input"),
        (
            [
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((0, 16, 3), dtype=np.uint8),
            ],
            "empty video frame",
        ),
        ([Image.new("RGB", (0, 16))], "empty video frame"),
    ):
        with pytest.raises(ValueError, match=pattern):
            _check_media_dimensions(_FakeMMItems({"videos": [bad_video]}))  # type: ignore[arg-type]

    # Valid HWC image and sequence video pass without raising.
    _check_media_dimensions(
        _FakeMMItems(  # type: ignore[arg-type]
            {
                "images": [np.zeros((16, 16, 3), dtype=np.uint8)],
                "videos": [[np.zeros((16, 16, 3), dtype=np.uint8)]],
            }
        )
    )


def test_registered_mm_config_rejects_unsupported_flags(tmp_path: Path) -> None:
    """Registered multimodal config hook rejects unsupported MM flags."""
    register_models()
    assert (
        ModelRegistry.models["MuseSparkForConditionalGeneration"].load_model_cls()
        is TPUMuseSparkForConditionalGeneration
    )
    cfg_hook = MODELS_CONFIG_MAP["MuseSparkForConditionalGeneration"]

    vc_dp = _make_mm_vllm_config(tmp_path / "dp")
    vc_dp.parallel_config.data_parallel_size = 2
    vc_dp.parallel_config.enable_expert_parallel = True
    with pytest.raises(NotImplementedError, match="data_parallel_size=1"):
        cfg_hook.verify_and_update_config(vc_dp)

    vc_tp_data = _make_mm_vllm_config(tmp_path / "tp_data")
    assert vc_tp_data.model_config.multimodal_config is not None
    vc_tp_data.model_config.multimodal_config.mm_encoder_tp_mode = "data"
    with pytest.raises(NotImplementedError, match="mm_encoder_tp_mode='data'"):
        cfg_hook.verify_and_update_config(vc_tp_data)

    vc_no_chunk = _make_mm_vllm_config(tmp_path / "no_chunk")
    vc_no_chunk.scheduler_config.disable_chunked_mm_input = True
    with pytest.raises(NotImplementedError, match="disable-chunked-mm-input"):
        cfg_hook.verify_and_update_config(vc_no_chunk)
