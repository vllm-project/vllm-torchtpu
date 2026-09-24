# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm_torchtpu.runner import mm_encoder_manager as mm


@pytest.fixture
def compile_mock(monkeypatch):
    result = Mock(return_value=Mock())
    monkeypatch.setattr(mm.torch, "compile", result)
    return result


@pytest.fixture
def manager(compile_mock):
    model = Mock()
    model.get_encoder_cudagraph_config.return_value = SimpleNamespace(
        padding_logics={},
        paths={
            "global": SimpleNamespace(min_token_budget=None, allow_zero_tokens=False),
            "local": SimpleNamespace(min_token_budget=8, allow_zero_tokens=True),
        },
    )
    model.prepare_encoder_cudagraph_capture_inputs.side_effect = (
        lambda budget, *args, path: SimpleNamespace(
            values={
                "pixels": torch.zeros(budget, 2),
                "scalar": torch.tensor(0),
                "positions": torch.arange(budget),
            }
        )
    )
    model.get_encoder_cudagraph_item_specs.return_value = [Mock(), Mock()]
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda _: 0, mm_encoder_tp_mode="weights"
            ),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=[8, 4],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
    )
    return mm.MMEncoderManager(config, torch.device("cpu"), model)


def test_initialization_compiles_encoder_forward(manager, compile_mock):
    compile_mock.assert_called_once_with(
        manager.model.encoder_cudagraph_forward,
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )


def test_initialization_builds_positive_budgets_per_path(manager):
    assert {
        path: list(templates) for path, templates in manager.budget_templates.items()
    } == {"global": [4, 8], "local": [8]}
    assert manager.graph_hits == manager.graph_misses == 0
    assert manager.model.prepare_encoder_cudagraph_capture_inputs.call_args_list == [
        call(4, 2, 0, torch.device("cpu"), torch.float32, path="global"),
        call(8, 2, 0, torch.device("cpu"), torch.float32, path="global"),
        call(8, 2, 0, torch.device("cpu"), torch.float32, path="local"),
    ]
    assert (
        manager.budget_templates["global"][8]["pixels"]
        is not (manager.budget_templates["local"][8]["pixels"])
    )


def test_padding_clears_previous_request_and_preserves_defaults(manager):
    template = manager.budget_templates["global"][4]
    template["pixels"].fill_(99)
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = manager._pad_to_template(
        {"pixels": source, "scalar": torch.tensor(7)}, 4, "global"
    )
    torch.testing.assert_close(
        result["pixels"], torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]])
    )
    assert result["scalar"].item() == 7
    assert result["positions"] is template["positions"]
    assert result["pixels"] is template["pixels"]
    torch.testing.assert_close(source, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    result = manager._pad_to_template({"pixels": torch.ones(1, 2)}, 4, "global")
    torch.testing.assert_close(result["pixels"][1:], torch.zeros(3, 2))


def test_exact_shape_uses_input_without_overwriting_template(manager):
    source = torch.ones(4, 2)
    result = manager._pad_to_template({"pixels": source}, 4, "global")
    assert result["pixels"] is source
    torch.testing.assert_close(
        manager.budget_templates["global"][4]["pixels"], torch.zeros(4, 2)
    )


def test_custom_padding_is_selected_per_key_and_path(manager):
    def pad(dst, src):
        dst.fill_(-1)
        dst[-src.shape[0] :].copy_(src)

    manager.config.padding_logics["pixels"] = pad
    result = manager._pad_to_template({"pixels": torch.ones(1, 2)}, 8, "local")
    torch.testing.assert_close(result["pixels"][:-1], -torch.ones(7, 2))
    torch.testing.assert_close(result["pixels"][-1:], torch.ones(1, 2))
    torch.testing.assert_close(
        manager.budget_templates["global"][8]["pixels"], torch.zeros(8, 2)
    )


def test_capture_preserves_template_mapping_and_registers_after_sync(
    manager, monkeypatch
):
    template = manager.budget_templates["global"][4]
    output = torch.ones(4, 2)

    def forward(values, *, path):
        assert not torch.is_grad_enabled()
        assert values is not template
        assert path == "global"
        values.pop("pixels")
        return output

    def synchronize(values, *, wait):
        assert values is output
        assert wait
        assert 4 not in manager._get_graph_set("global")

    manager._compiled_budget_forward.side_effect = forward
    monkeypatch.setattr(mm, "synchronize_tensors", synchronize)
    manager._capture_budget_graph(4, "global")
    assert manager._get_graph_set("global")[4] is template
    assert "pixels" in template


@pytest.mark.parametrize("failure", ["forward", "synchronize"])
def test_capture_failure_leaves_budget_unregistered(manager, monkeypatch, failure):
    synchronize = Mock()
    target = manager._compiled_budget_forward if failure == "forward" else synchronize
    target.side_effect = RuntimeError("capture failed")
    monkeypatch.setattr(mm, "synchronize_tensors", synchronize)
    manager._capture_budget_graph(4, "global")
    assert 4 not in manager._get_graph_set("global")
    assert manager.graph_hits == manager.graph_misses == 0


def test_precompile_continues_after_one_budget_fails(manager, monkeypatch):
    manager._compiled_budget_forward.side_effect = [
        RuntimeError("compile"),
        torch.ones(8, 2),
        torch.ones(8, 2),
    ]
    synchronize = Mock()
    monkeypatch.setattr(mm, "synchronize_tensors", synchronize)
    manager.precompile_vision_encoder()
    assert manager._compiled_budget_forward.call_count == 3
    assert synchronize.call_count == 2
    assert set(manager._get_graph_set("global")) == {8}
    assert set(manager._get_graph_set("local")) == {8}


@pytest.mark.parametrize("path, budget", [("missing", 4), ("global", 16), ("local", 0)])
def test_missing_template_counts_items_without_replay(manager, path, budget):
    assert manager._run_budget_graph({}, budget, path) is None
    assert manager.graph_hits == 0
    assert manager.graph_misses == 2
    manager.model.prepare_encoder_cudagraph_replay_buffers.assert_not_called()
    manager._compiled_budget_forward.assert_not_called()


def test_replay_counts_items_and_passes_padded_values(manager):
    mm_kwargs = {"images": object()}
    manager.model.prepare_encoder_cudagraph_replay_buffers.return_value = (
        SimpleNamespace(values={"pixels": torch.ones(2, 2)})
    )
    output = torch.ones(4, 3)

    def forward(values, *, path):
        assert not torch.is_grad_enabled()
        assert path == "global"
        pixels = values.pop("pixels")
        torch.testing.assert_close(pixels[:2], torch.ones(2, 2))
        torch.testing.assert_close(pixels[2:], torch.zeros(2, 2))
        return output

    manager._compiled_budget_forward.side_effect = forward
    for _ in range(2):
        assert manager._run_budget_graph(mm_kwargs, 4, "global") is output
    assert manager.graph_hits == 4
    assert manager.graph_misses == 0
    assert "pixels" in manager.budget_templates["global"][4]
    manager.model.prepare_encoder_cudagraph_replay_buffers.assert_called_with(
        mm_kwargs, 2, 0, path="global"
    )
    manager.model.encoder_eager_forward.assert_not_called()


def test_replay_failure_uses_original_inputs_for_eager_fallback(manager):
    mm_kwargs = {"images": object()}
    manager.model.prepare_encoder_cudagraph_replay_buffers.return_value = (
        SimpleNamespace(values={"pixels": torch.ones(2, 2)})
    )
    manager._compiled_budget_forward.side_effect = RuntimeError("execute")
    output = torch.ones(2, 3)

    def eager(values, *, path):
        assert values is mm_kwargs
        assert path == "local"
        assert torch.is_inference_mode_enabled()
        return output

    manager.model.encoder_eager_forward.side_effect = eager
    assert manager._run_budget_graph(mm_kwargs, 8, "local") is output
    assert manager.graph_hits == 0
    assert manager.graph_misses == 2


@pytest.mark.parametrize(
    "enabled, supported", [(False, True), (True, False), (True, True)]
)
def test_factory_requires_enabled_and_supported(monkeypatch, enabled, supported):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=enabled)
    )
    model = Mock()
    device = torch.device("cpu")
    supports = Mock(return_value=supported)
    constructor = Mock()
    monkeypatch.setattr(mm, "supports_encoder_cudagraph", supports)
    monkeypatch.setattr(mm, "MMEncoderManager", constructor)
    result = mm.maybe_create_mm_encoder_manager(config, device, model)
    if enabled and supported:
        assert result is constructor.return_value
        constructor.assert_called_once_with(config, device, model)
    else:
        assert result is None
        constructor.assert_not_called()
    if not enabled:
        supports.assert_not_called()
