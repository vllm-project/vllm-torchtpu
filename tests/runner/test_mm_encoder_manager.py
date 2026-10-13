# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from vllm.distributed import parallel_state
from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

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
        capture_axes=(),
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


@pytest.mark.parametrize("allow_eager_fallback", [True, False])
def test_replay_failure_uses_original_inputs_for_eager_fallback(
    manager, allow_eager_fallback
):
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
    if not allow_eager_fallback:
        with pytest.raises(RuntimeError, match="execute"):
            manager._run_budget_graph(mm_kwargs, 8, "local", allow_eager_fallback=False)
        manager.model.encoder_eager_forward.assert_not_called()
        return
    assert manager._run_budget_graph(mm_kwargs, 8, "local") is output
    assert manager.graph_hits == 0
    assert manager.graph_misses == 2


@pytest.fixture
def coordinated_manager(manager, monkeypatch):
    manager.path_token_budgets = {"default": [4, 8]}
    manager.budget_templates = {"default": manager.budget_templates["global"]}
    manager.config.out_hidden_size = 2
    monkeypatch.setattr(mm, "synchronize_tensors", Mock())
    manager.model.get_encoder_cudagraph_item_specs.side_effect = lambda kwargs: [
        EncoderItemSpec(input_size=4 * count, output_tokens=count)
        for count in kwargs["tokens"]
    ]
    manager.model.select_encoder_cudagraph_items.side_effect = lambda kwargs, indices: {
        "tokens": [kwargs["tokens"][i] for i in indices]
    }
    manager._run_budget_graph = Mock(return_value=torch.zeros(8, 2))
    manager._compiled_budget_forward.return_value = torch.zeros(8, 2)

    def postprocess(graphs, indices, tokens, outputs, **kwargs):
        for index, count in zip(
            indices, kwargs["batch_mm_kwargs"]["tokens"], strict=True
        ):
            outputs[index] = torch.full((count, 2), float(count))

    manager.model.postprocess_encoder_output.side_effect = postprocess
    return manager


@pytest.mark.parametrize("active", [True, False])
def test_coordinated_encoder_matches_peer_buckets_and_restores_item_order(
    coordinated_manager, monkeypatch, active
):
    manager = coordinated_manager
    monkeypatch.setattr(
        parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=3, cpu_group=None),
    )

    def gather(plans, local, *, group):
        assert local == (None, [4, 4] if active else [], 8 if active else 0)
        plans[:] = [local, (None, [8, 8, 8], 24), (None, [], 0)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    counts = [3, 1, 4] if active else []
    outputs = manager.execute_dp_coordinated({"tokens": counts} if active else None)
    assert [o.shape[0] for o in outputs] == counts
    for count, output in zip(counts, outputs, strict=True):
        assert (output == count).all()
    assert manager._run_budget_graph.call_args_list == (
        [
            call(
                {"tokens": [1, 3]},
                8,
                axis_keys=(),
                allow_eager_fallback=False,
            ),
            call(
                {"tokens": [4]},
                8,
                axis_keys=(),
                allow_eager_fallback=False,
            ),
        ]
        if active
        else []
    )
    assert manager._compiled_budget_forward.call_count == (1 if active else 3)
    assert mm.synchronize_tensors.call_count == 3


def test_coordinated_encoder_propagates_peer_input_errors(
    coordinated_manager, monkeypatch
):
    monkeypatch.setattr(
        parallel_state,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=None),
    )

    def gather(plans, local, *, group):
        plans[:] = [local, ("image exceeds largest bucket", [], 0)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="image exceeds largest bucket"):
        coordinated_manager.execute_dp_coordinated(None)
    coordinated_manager._compiled_budget_forward.assert_not_called()


@pytest.mark.parametrize("counts", [[3, 1, 4], [3], []])
@pytest.mark.parametrize("rank", [0, 1])
def test_coordinated_encoder_dp_shards_and_gathers_empty_and_uneven_ranks(
    coordinated_manager, monkeypatch, counts, rank
):
    manager = coordinated_manager
    manager.use_dp = True
    tp = SimpleNamespace(world_size=2, rank_in_group=rank, all_gather=Mock())
    ep = SimpleNamespace(world_size=4, cpu_group=object())
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: tp)
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: ep)
    assignments = (
        [[2], [0, 1]] if len(counts) == 3 else [[0], []] if counts else [[], []]
    )

    def plans_gather(plans, local, *, group):
        assert group is ep.cpu_group
        local_tokens = sum(counts[i] for i in assignments[rank])
        assert local == (None, [4] if local_tokens else [], local_tokens)
        plans[:] = [local, (None, [8, 8], 12), (None, [], 0), (None, [], 0)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", plans_gather)
    gathered = torch.zeros(24, 2)
    for peer, assigned in enumerate(assignments):
        offset = peer * 12
        for index in assigned:
            gathered[offset : offset + counts[index]] = counts[index]
            offset += counts[index]

    def tensors_gather(padded, *, dim):
        assert dim == 0
        torch.testing.assert_close(padded, gathered[rank * 12 : (rank + 1) * 12])
        assert mm.synchronize_tensors.call_args == call(padded, wait=True)
        return gathered

    tp.all_gather.side_effect = tensors_gather
    outputs = manager.execute_dp_coordinated({"tokens": counts} if counts else None)
    assert [o.shape[0] for o in outputs] == counts
    for count, output in zip(counts, outputs, strict=True):
        assert (output == count).all()
    assert manager._compiled_budget_forward.call_count == (
        1 if assignments[rank] else 2
    )
    assert mm.synchronize_tensors.call_args == call(gathered, wait=True)
    tp.all_gather.assert_called_once()


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


def test_execute_local_chunks_oversized_video_into_compiled_budgets(compile_mock):
    """Chunk an oversized video so each chunk hits a compiled budget graph."""
    model = Mock()
    model.get_encoder_cudagraph_config.return_value = SimpleNamespace(
        padding_logics={},
        capture_axes=(),
        paths={
            "default": SimpleNamespace(min_token_budget=None, allow_zero_tokens=False),
        },
    )
    # 4 patches per output token (e.g. spatial_merge_size = 2)
    model.prepare_encoder_cudagraph_capture_inputs.side_effect = (
        lambda budget, *args, path: SimpleNamespace(
            values={"pixel_values": torch.zeros(budget * 4, 2)}
        )
    )

    def get_item_specs(mm_kwargs):
        grid = mm_kwargs["video_grid_thw"]
        grid_list = grid.tolist() if hasattr(grid, "tolist") else list(grid)
        return [
            SimpleNamespace(
                input_size=int(t) * int(h) * int(w),
                output_tokens=int(t) * (int(h) // 2) * (int(w) // 2),
                get_path_output_tokens=lambda path, t=t, h=h, w=w: (
                    int(t) * (int(h) // 2) * (int(w) // 2)
                ),
            )
            for t, h, w in grid_list
        ]

    def select_items(mm_kwargs, indices):
        grid = mm_kwargs["video_grid_thw"]
        grid_list = grid.tolist() if hasattr(grid, "tolist") else list(grid)
        pixels = mm_kwargs["pixel_values_videos"]
        selected_grid = [grid_list[i] for i in indices]
        return {
            "pixel_values_videos": pixels,
            "video_grid_thw": torch.tensor(selected_grid, dtype=torch.int64),
        }

    def prepare_replay(mm_kwargs, *args, path):
        return SimpleNamespace(
            values={"pixel_values": mm_kwargs["pixel_values_videos"]}
        )

    def postprocess(
        outputs, indices, per_item_out_tokens, dest, clone, batch_mm_kwargs
    ):
        out = outputs["default"]
        offset = 0
        for idx in indices:
            n = per_item_out_tokens[idx]
            dest[idx] = out[offset : offset + n].clone()
            offset += n

    model.get_encoder_cudagraph_item_specs.side_effect = get_item_specs
    model.select_encoder_cudagraph_items.side_effect = select_items
    model.prepare_encoder_cudagraph_replay_buffers.side_effect = prepare_replay
    model.postprocess_encoder_output.side_effect = postprocess

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda _: 0,
                mm_encoder_tp_mode="weights",
                video_pruning_rate=None,
            ),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=[4, 8],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
    )
    mgr = mm.MMEncoderManager(config, torch.device("cpu"), model)

    # Forward projects every group of 4 patch rows to 1 output token by summing
    def compiled_forward(values, *, path):
        pv = values["pixel_values"]
        return pv.reshape(-1, 4, 2).sum(dim=1)

    mgr._compiled_budget_forward.side_effect = compiled_forward

    # Video with t=5, h=4, w=4 -> 80 patches -> 20 output tokens (> max budget 8).
    # Max budget 8 tokens = 32 patches = 2 frames of 4x4, so chunks into:
    # t=2 (8 tokens, budget 8), t=2 (8 tokens, budget 8), t=1 (4 tokens, budget 4).
    pixels = torch.arange(160, dtype=torch.float32).reshape(80, 2)
    mm_kwargs = {
        "pixel_values_videos": pixels,
        "video_grid_thw": torch.tensor([[5, 4, 4]], dtype=torch.int64),
    }
    outputs = mgr._execute_local(mm_kwargs)
    assert len(outputs) == 1
    expected = pixels.reshape(20, 4, 2).sum(dim=1)
    torch.testing.assert_close(outputs[0], expected)
    assert mgr.graph_hits == 3
    assert mgr.graph_misses == 0
    model.encoder_eager_forward.assert_not_called()
