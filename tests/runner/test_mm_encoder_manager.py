# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs, EncoderCudaGraphConfig,
    EncoderCudaGraphPathConfig, EncoderCudaGraphReplayBuffers)

from vllm_torchtpu.runner.mm_encoder_manager import (
    MMEncoderManager, maybe_create_mm_encoder_manager)


@pytest.fixture
def mock_vllm_config():
    config = MagicMock()
    config.compilation_config.cudagraph_mm_encoder = True
    config.compilation_config.encoder_cudagraph_token_budgets = [256, 512]
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 2
    config.compilation_config.encoder_cudagraph_max_frames_per_batch = 16
    config.model_config.dtype = torch.bfloat16

    mm_config = MagicMock()
    mm_config.get_limit_per_prompt.return_value = 0
    mm_config.mm_encoder_tp_mode = None
    config.model_config.multimodal_config = mm_config
    config.parallel_config.tensor_parallel_size = 1
    return config


@pytest.fixture
def mock_model():
    model = MagicMock(spec=[
        "get_encoder_cudagraph_config",
        "prepare_encoder_cudagraph_capture_inputs",
        "prepare_encoder_cudagraph_replay_buffers",
        "encoder_cudagraph_forward",
        "encoder_eager_forward",
    ])

    encoder_config = EncoderCudaGraphConfig(
        modalities=["image"],
        buffer_keys=["pixel_values", "image_grid_thw"],
        out_hidden_size=64,
        paths={"default": EncoderCudaGraphPathConfig()},
    )
    model.get_encoder_cudagraph_config.return_value = encoder_config

    def prepare_capture(budget,
                        max_batch,
                        max_frames,
                        device,
                        dtype,
                        path="default"):
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixel_values":
                torch.zeros((budget, 64), dtype=dtype, device=device),
                "image_grid_thw":
                torch.zeros((max_batch, 3), dtype=torch.int32, device=device),
                "scalar_meta":
                torch.tensor(42, device=device),
            })

    model.prepare_encoder_cudagraph_capture_inputs.side_effect = prepare_capture
    model.encoder_cudagraph_forward.side_effect = lambda t, path="default": torch.ones(
        (256, 64), dtype=torch.bfloat16)
    return model


def test_maybe_create_mm_encoder_manager_disabled(mock_vllm_config,
                                                  mock_model):
    mock_vllm_config.compilation_config.cudagraph_mm_encoder = False
    manager = maybe_create_mm_encoder_manager(mock_vllm_config,
                                              torch.device("cpu"), mock_model)
    assert manager is None


@patch("vllm_torchtpu.runner.mm_encoder_manager.supports_encoder_cudagraph",
       return_value=False)
def test_maybe_create_mm_encoder_manager_unsupported(mock_supports,
                                                     mock_vllm_config,
                                                     mock_model):
    manager = maybe_create_mm_encoder_manager(mock_vllm_config,
                                              torch.device("cpu"), mock_model)
    assert manager is None


@patch("vllm_torchtpu.runner.mm_encoder_manager.supports_encoder_cudagraph",
       return_value=True)
@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_maybe_create_mm_encoder_manager_success(mock_compile, mock_supports,
                                                 mock_vllm_config, mock_model):
    manager = maybe_create_mm_encoder_manager(mock_vllm_config,
                                              torch.device("cpu"), mock_model)
    assert isinstance(manager, MMEncoderManager)
    assert manager.token_budgets == [256, 512]
    assert 256 in manager.budget_templates["default"]
    assert 512 in manager.budget_templates["default"]


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_pad_to_template_exact_shape(mock_compile, mock_vllm_config,
                                     mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    replay_values = {
        "pixel_values": torch.ones((256, 64), dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 2, 3], [4, 5, 6]],
                                       dtype=torch.int32),
        "scalar_meta": torch.tensor(42),
    }

    padded = manager._pad_to_template(replay_values, budget=256)
    assert torch.equal(padded["pixel_values"], replay_values["pixel_values"])
    assert torch.equal(padded["image_grid_thw"],
                       replay_values["image_grid_thw"])
    assert torch.equal(padded["scalar_meta"], replay_values["scalar_meta"])


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_pad_to_template_dynamic_padding(mock_compile, mock_vllm_config,
                                         mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    # Dynamic input has fewer tokens (100) than the budget (256)
    dynamic_pixel_values = torch.ones((100, 64), dtype=torch.bfloat16)
    replay_values = {
        "pixel_values": dynamic_pixel_values,
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int32),
    }

    padded = manager._pad_to_template(replay_values, budget=256)

    # Output should be padded to budget template shape (256, 64)
    assert padded["pixel_values"].shape == (256, 64)
    # The first 100 rows should contain the dynamic input
    assert torch.equal(padded["pixel_values"][:100], dynamic_pixel_values)
    # The remaining rows should be zero-padded
    assert torch.equal(padded["pixel_values"][100:],
                       torch.zeros((156, 64), dtype=torch.bfloat16))

    # Grid THW should match template shape (2, 3) with first row filled
    assert padded["image_grid_thw"].shape == (2, 3)
    assert torch.equal(padded["image_grid_thw"][:1],
                       replay_values["image_grid_thw"])
    assert torch.equal(padded["image_grid_thw"][1:],
                       torch.zeros((1, 3), dtype=torch.int32))

    # Missing key defaults to template
    assert "scalar_meta" in padded
    assert padded["scalar_meta"].item() == 42


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
@patch("vllm_torchtpu.runner.mm_encoder_manager.synchronize_tensors")
def test_capture_budget_graph(mock_sync, mock_compile, mock_vllm_config,
                              mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    manager._capture_budget_graph(256)
    assert 256 in manager.budget_graphs["default"]
    mock_sync.assert_called_once()


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_run_budget_graph_hit(mock_compile, mock_vllm_config, mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    mock_model.prepare_encoder_cudagraph_replay_buffers.return_value = (
        EncoderCudaGraphReplayBuffers(
            values={
                "pixel_values": torch.ones((100, 64), dtype=torch.bfloat16),
                "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int32),
            }))

    # Mock item specs (e.g. 1 vision item)
    item_spec = MagicMock()
    manager._get_item_specs = MagicMock(return_value=[item_spec])

    mm_kwargs = {"image": torch.zeros(1)}
    output = manager._run_budget_graph(mm_kwargs, token_budget=256)

    assert output is not None
    assert manager.graph_hits == 1
    assert manager.graph_misses == 0
    mock_model.prepare_encoder_cudagraph_replay_buffers.assert_called_once_with(
        mm_kwargs,
        manager.max_batch_size,
        manager.max_frames_per_batch,
        path="default")


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_run_budget_graph_miss(mock_compile, mock_vllm_config, mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    item_spec = MagicMock()
    manager._get_item_specs = MagicMock(return_value=[item_spec])

    mm_kwargs = {"image": torch.zeros(1)}
    # Unbudgeted token count
    output = manager._run_budget_graph(mm_kwargs, token_budget=9999)

    assert output is None
    assert manager.graph_hits == 0
    assert manager.graph_misses == 1


@patch("torch.compile", side_effect=lambda fn, **kwargs: fn)
def test_precompile_vision_encoder(mock_compile, mock_vllm_config, mock_model):
    manager = MMEncoderManager(mock_vllm_config, torch.device("cpu"),
                               mock_model)

    manager._capture_budget_graph = MagicMock()
    manager.precompile_vision_encoder()

    assert manager._capture_budget_graph.call_count == len(
        manager.token_budgets)
    for budget in manager.token_budgets:
        manager._capture_budget_graph.assert_any_call(budget, path="default")
