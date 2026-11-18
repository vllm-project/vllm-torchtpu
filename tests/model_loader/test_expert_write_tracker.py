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
import contextlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm_torchtpu.model_loader_patches import ExpertParamStager, ExpertWriteTracker


def test_w13_completes_after_both_shards_of_every_expert():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(4, 6, 8)
    for expert in range(4):
        assert not tracker.record(w13, expert, "w1")
        last = tracker.record(w13, expert, "w3")
        assert last == (expert == 3)


def test_w2_completes_after_every_expert():
    tracker = ExpertWriteTracker()
    w2 = torch.empty(3, 8, 6)
    assert [tracker.record(w2, e, "w2") for e in range(3)] == [False, False, True]


def test_w13_waits_for_w3_when_every_w1_arrives_first():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(4, 6, 8)
    assert not any(tracker.record(w13, e, "w1") for e in range(4))
    assert [tracker.record(w13, e, "w3") for e in range(4)] == [
        False,
        False,
        False,
        True,
    ]


def test_one_local_expert_needs_both_shards():
    tracker = ExpertWriteTracker()
    w13 = torch.empty(1, 6, 8)
    assert not tracker.record(w13, 5, "w1")
    assert tracker.record(w13, 5, "w3")


def test_whole_tensor_writes_never_complete_a_parameter():
    # gpt-oss MXFP4 checkpoints copy every expert in one write with no
    # shard or expert id; such a parameter is written at the end of loading.
    tracker = ExpertWriteTracker()
    w13 = torch.empty(1, 6, 8)
    assert not tracker.record(w13, None, None)
    assert not tracker.record(w13, None, None)


def test_parameters_are_tracked_apart():
    tracker = ExpertWriteTracker()
    a, b = torch.empty(2, 4, 4), torch.empty(2, 4, 4)
    assert not tracker.record(a, 0, "w2")
    assert not tracker.record(b, 0, "w2")
    assert tracker.record(a, 1, "w2")
    assert tracker.record(b, 1, "w2")


def test_a_completed_parameter_can_be_written_again():
    tracker = ExpertWriteTracker()
    w2 = torch.empty(2, 4, 4)
    assert tracker.record(w2, 0, "w2") is False
    assert tracker.record(w2, 1, "w2") is True
    assert tracker.record(w2, 0, "w2") is False


def _param(*shape):
    return torch.nn.Parameter(torch.zeros(*shape), requires_grad=False)


def test_stager_writes_the_parameter_once_its_last_expert_lands():
    stager = ExpertParamStager()
    w2 = _param(2, 4, 4)
    host = stager.host_param(w2)
    assert stager.host_param(w2) is host
    host[0] = 1
    assert not stager.written(w2, 0, "w2")
    assert torch.equal(w2.data, torch.zeros(2, 4, 4))
    host[1] = 2
    assert stager.written(w2, 1, "w2")
    assert torch.equal(w2.data[0], torch.ones(4, 4))
    assert torch.equal(w2.data[1], torch.full((4, 4), 2.0))
    assert len(stager) == 0


def test_stand_in_keeps_the_parameter_attributes():
    stager = ExpertParamStager()
    w13 = _param(2, 8, 4)
    w13.is_transposed = True
    host = stager.host_param(w13)
    assert host.is_transposed is True
    assert host.shape == w13.shape and host.device.type == "cpu"


def test_flush_all_writes_incomplete_parameters():
    stager = ExpertParamStager()
    w13 = _param(3, 8, 4)
    stager.host_param(w13)[0] = 5
    assert not stager.written(w13, 0, "w1")
    stager.flush_all()
    assert len(stager) == 0
    assert stager.flushed == 1
    assert torch.equal(w13.data[0], torch.full((8, 4), 5.0))


def test_flush_all_drops_the_buffers_flushes_left_behind():
    stager = ExpertParamStager()
    w2 = _param(1, 8, 4)
    stager.host_param(w2)
    assert stager.written(w2, 0, "w2")
    assert stager._pool
    stager.flush_all()
    assert not stager._pool


def test_flush_uploads_in_chunks(monkeypatch):
    from vllm_torchtpu import model_loader_patches

    # two experts per upload
    monkeypatch.setattr(model_loader_patches, "_UPLOAD_CHUNK_BYTES", 2 * 8 * 4 * 4)
    stager = ExpertParamStager()
    w2 = _param(5, 8, 4)
    host = stager.host_param(w2)
    for expert in range(5):
        host[expert] = expert + 1
        stager.written(w2, expert, "w2")
    assert w2.shape == (5, 8, 4)
    for expert in range(5):
        assert torch.equal(w2.data[expert], torch.full((8, 4), expert + 1.0))


def test_the_oldest_parameter_is_written_early_when_too_many_are_staged(monkeypatch):
    from vllm_torchtpu import model_loader_patches

    monkeypatch.setattr(model_loader_patches, "_MAX_STAGED", 2)
    stager = ExpertParamStager()
    first, second, third = _param(2, 4, 4), _param(2, 4, 4), _param(2, 4, 4)
    stager.host_param(first)[0] = 7
    stager.host_param(second)
    assert stager.staging(first)
    stager.host_param(third)
    # staging a third parameter wrote the first one early
    assert stager.flushed == 1
    assert torch.equal(first.data[0], torch.full((4, 4), 7.0))
    assert not stager.staging(first)
    assert stager.staging(second) and stager.staging(third)
    assert len(stager) == 2


def test_interleaved_parameters_sharing_pooled_buffers_keep_their_experts():
    """Every w1 of two same-shaped parameters lands before any w3; neither
    may be written early and hand its buffer to the other."""
    stager = ExpertParamStager()
    a, b = _param(2, 4, 2), _param(2, 4, 2)
    for param, w1 in ((a, 11), (b, 21)):
        for expert in range(2):
            stager.host_param(param)[expert, :2] = w1
            assert not stager.written(param, expert, "w1")
    assert stager.flushed == 0
    for param, w3 in ((a, 13), (b, 23)):
        for expert in range(2):
            stager.host_param(param)[expert, 2:] = w3
            assert stager.written(param, expert, "w3") == (expert == 1)
    assert stager.flushed == 2 and len(stager) == 0
    for param, w1, w3 in ((a, 11, 13), (b, 21, 23)):
        assert torch.equal(param.data[:, :2], torch.full((2, 2, 2), float(w1)))
        assert torch.equal(param.data[:, 2:], torch.full((2, 2, 2), float(w3)))


@pytest.fixture
def staging_patch_undo(monkeypatch):
    """Undo patch_moe_expert_write_staging after the test.

    The patch is process-wide; undoing it lets later tests load expert weights
    through the stock loader and start from a fresh stager.
    """
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.model_loader import base_loader

    monkeypatch.setattr(RoutedExperts, "weight_loader", RoutedExperts.weight_loader)
    monkeypatch.setattr(
        RoutedExperts, "_tpu_expert_staging_patch", False, raising=False
    )
    monkeypatch.setattr(RoutedExperts, "_tpu_expert_stager", None, raising=False)
    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        base_loader.process_weights_after_loading,
    )


def test_the_patched_loader_keeps_the_upstream_loader_attributes(staging_patch_undo):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from vllm_torchtpu.model_loader_patches import patch_moe_expert_write_staging

    patch_moe_expert_write_staging()
    loader = RoutedExperts.weight_loader
    assert loader.supports_moe_loading is True
    assert loader.__name__ == "weight_loader"


_MODEL_CONFIG = SimpleNamespace(dtype=torch.bfloat16, quantization=None)


def _quant_method(process):
    """A quant method whose process_weights_after_loading calls `process`."""
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    class _QuantMethod(QuantizeMethodBase):
        def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any):
            return None

        def apply(self, layer: torch.nn.Module, *args: Any, **kwargs: Any):
            return None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            process(layer)

    return _QuantMethod()


def _tagged(module: Any, quant_method: Any, tag: str) -> Any:
    module.quant_method = quant_method
    module.tag = tag
    return module


def _linear(quant_method: Any, tag: str) -> Any:
    from vllm.model_executor.layers.linear import LinearBase

    linear = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(linear)
    return _tagged(linear, quant_method, tag)


def _experts(quant_method: Any = None) -> Any:
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    return _tagged(experts, quant_method, "experts")


def _model(**children: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    for name, child in children.items():
        model.add_module(name, child)
    return model


@pytest.fixture
def post_load(staging_patch_undo, monkeypatch):
    """Upstream's process_weights_after_loading wrapped by the staging patch."""
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.model_loader import utils as loader_utils

    from vllm_torchtpu.model_loader_patches import patch_moe_expert_write_staging

    monkeypatch.setattr(
        base_loader,
        "process_weights_after_loading",
        loader_utils.process_weights_after_loading,
    )
    monkeypatch.setattr(
        loader_utils, "maybe_retie_word_embeddings", lambda *a, **kw: None
    )
    patch_moe_expert_write_staging()
    return lambda model: base_loader.process_weights_after_loading(
        model, _MODEL_CONFIG, torch.device("cpu")
    )


def test_linear_weights_processed_before_expert_flush_and_not_rerun(post_load):
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    events: list[str] = []

    def process_experts(layer: Any) -> None:
        # RoutedExperts runs after stager.flush_all() has written w2.
        assert torch.equal(layer.w2.data[0], torch.full((4, 4), 9.0))
        events.append(layer.tag)

    experts = _experts(_quant_method(process_experts))
    experts.w2 = _param(2, 4, 4)

    def process_linear(layer: Any) -> None:
        # LinearBase runs while expert writes are still staged on the host.
        assert torch.equal(experts.w2.data[0], torch.zeros((4, 4)))
        events.append(layer.tag)

    shared_qm = _quant_method(process_linear)
    model = _model(
        experts=experts,
        lin_a=_linear(shared_qm, "a"),
        lin_b=_linear(shared_qm, "b"),
    )
    stager = cast(Any, RoutedExperts)._tpu_expert_stager
    # Stage only expert 0 of 2 so w2 stays on the host until flush_all().
    stager.host_param(experts.w2)[0] = 9.0
    assert not stager.written(experts.w2, 0, "w2")

    post_load(model)

    assert events == ["a", "b", "experts"]
    # The overrides are removed rather than shadowed by instance attributes.
    assert "process_weights_after_loading" not in vars(shared_qm)
    assert "requires_device_loading" not in vars(shared_qm)


def test_offloaded_linear_modules_are_not_moved_to_the_device_twice(
    post_load, monkeypatch
):
    """Under CPU offloading, upstream's pass skips the device round trip for
    linear modules the early pass already processed."""
    from vllm.model_executor.model_loader import utils as loader_utils

    entered: list[str] = []

    @contextlib.contextmanager
    def spy_device_loading(module, target_device):
        entered.append(module.tag)
        yield module

    monkeypatch.setattr(loader_utils, "device_loading_context", spy_device_loading)
    shared_qm = _quant_method(lambda layer: None)
    model = _model(
        experts=_experts(_quant_method(lambda layer: None)),
        lin_a=_linear(shared_qm, "a"),
        lin_b=_linear(shared_qm, "b"),
    )

    post_load(model)

    assert entered == ["a", "b", "experts"]


def test_quant_method_shared_with_a_non_linear_module_keeps_device_loading(
    post_load,
):
    events: list[tuple[str, bool]] = []

    def process(layer: Any) -> None:
        events.append((layer.tag, shared_qm.requires_device_loading))

    shared_qm = _quant_method(process)
    model = _model(
        experts=_experts(),
        lin=_linear(shared_qm, "lin"),
        other=_tagged(torch.nn.Module(), shared_qm, "other"),
    )

    post_load(model)

    assert events == [("lin", True), ("other", True)]


def test_dense_model_takes_the_upstream_path(staging_patch_undo, monkeypatch):
    from vllm.model_executor.model_loader import base_loader

    from vllm_torchtpu.model_loader_patches import patch_moe_expert_write_staging

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        base_loader, "process_weights_after_loading", lambda *a: calls.append(a)
    )
    patch_moe_expert_write_staging()
    events: list[str] = []
    qm = _quant_method(lambda layer: events.append(layer.tag))
    model = _model(lin=_linear(qm, "lin"))
    device = torch.device("cpu")

    base_loader.process_weights_after_loading(model, _MODEL_CONFIG, device)

    assert calls == [(model, _MODEL_CONFIG, device)]
    assert events == []


def test_failed_expert_processing_restores_quant_methods(post_load):
    def process_experts(layer: Any) -> None:
        # The early pass has finished, so the overrides are in place.
        assert shared_qm.requires_device_loading is False
        raise RuntimeError("boom in expert pwal")

    shared_qm = _quant_method(lambda layer: None)
    model = _model(
        experts=_experts(_quant_method(process_experts)),
        lin=_linear(shared_qm, "lin"),
    )

    with pytest.raises(RuntimeError, match="boom in expert pwal"):
        post_load(model)

    assert "process_weights_after_loading" not in vars(shared_qm)
    assert "requires_device_loading" not in vars(shared_qm)
