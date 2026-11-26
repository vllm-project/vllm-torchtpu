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
"""The optional model-owned runner adapter: off by default, delegated when set."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.runner import kv_cache_manager, model_adapters, tpu_runner
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.model_adapters import (
    ModelAdapter,
    register_model_adapter,
    select_model_adapter,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

ARCH = "ToyAdapterForCausalLM"


def _runner(arch=ARCH):
    return NS(model_config=NS(architecture=arch))


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(model_adapters, "_FACTORIES", {})


class _Adapter(ModelAdapter):
    def __init__(self, runner):
        self.runner = runner
        self.calls = []

    def bind_model(self, model):
        self.calls.append(("bind_model", model))

    def begin_step(self):
        return torch.zeros(1)

    def prepare_inputs(self, snapshot):
        return {}

    def get_kv_cache_spec(self):
        return {"layer.0": "spec"}

    def initialize_kv_cache(self, config):
        self.calls.append(("init", config))
        caches = {"layer.0": torch.zeros(1)}
        self.runner.kv_cache_config = config
        self.runner.attn_groups = [["group"]]
        self.runner.kv_caches = list(caches.values())
        return caches

    def dummy_run(self, num_tokens, num_reqs, profile_prefix=None, *, dp_lockstep):
        self.calls.append(("dummy", num_tokens, num_reqs, profile_prefix, dp_lockstep))
        return 1.5


def test_no_registration_selects_no_adapter():
    assert select_model_adapter(_runner()) is None


def test_registered_factory_serves_only_its_architectures():
    register_model_adapter([ARCH, "ToyAdapterForConditionalGeneration"], _Adapter)
    runner = _runner()
    adapter = select_model_adapter(runner)
    assert isinstance(adapter, _Adapter) and adapter.runner is runner
    assert isinstance(
        select_model_adapter(_runner("ToyAdapterForConditionalGeneration")), _Adapter
    )
    assert select_model_adapter(_runner("OtherForCausalLM")) is None
    # A factory may still decline a runner.
    register_model_adapter([ARCH], lambda runner: None)
    assert select_model_adapter(runner) is None


def test_a_factory_must_return_a_model_adapter():
    register_model_adapter([ARCH], lambda runner: NS())
    with pytest.raises(TypeError, match="does not subclass ModelAdapter"):
        select_model_adapter(_runner())


def test_optional_hooks_default_to_no_ops():
    adapter = _Adapter(None)
    assert adapter.bind_draft(object()) is None
    assert adapter.precompile_fused_inputs([16], [4, None]) is None
    assert adapter.kv_cache_memory_budget() is None


@pytest.mark.parametrize("model_kwargs", [None, {"extra": 7}])
def test_forward_model_passes_adapter_kwargs(model_kwargs):
    seen = {}

    def model(**kwargs):
        seen.update(kwargs)
        return torch.zeros(2, 4)

    out, aux = TPUModelRunner.forward_model(
        NS(model=model),
        input_ids=torch.zeros(2, dtype=torch.int32),
        positions=torch.zeros(2, dtype=torch.int32),
        model_kwargs=model_kwargs,
    )
    assert out.shape == (2, 4) and aux is None
    assert seen.get("extra") == (None if model_kwargs is None else 7)
    if model_kwargs is not None:
        assert model_kwargs == {"extra": 7}  # not mutated


def test_dummy_run_delegates_to_the_adapter():
    adapter = _Adapter(None)
    fenced = []
    runner = NS(
        model_adapter=adapter, _wait_input_staging_fence=lambda: fenced.append(1)
    )
    ms = TPUModelRunner._dummy_run(
        runner, 64, 4, 8, dp_lockstep=True, profile_prefix=128
    )
    assert ms == 1.5 and fenced == [1]
    assert adapter.calls == [("dummy", 64, 4, 128, True)]


@pytest.mark.parametrize(
    "mode", [{"through_handoff": True}, {"profile_decode_query_lens": [1, 2]}]
)
def test_dummy_run_rejects_modes_the_adapter_does_not_take(mode):
    adapter = _Adapter(None)
    runner = NS(model_adapter=adapter, _wait_input_staging_fence=lambda: None)
    with pytest.raises(NotImplementedError):
        TPUModelRunner._dummy_run(runner, 64, 4, 8, profile_prefix=128, **mode)
    assert adapter.calls == []


@pytest.mark.parametrize("connector", [False, True])
def test_kv_cache_manager_delegates_spec_and_cache(monkeypatch, connector):
    registered = []
    fake_connector = NS(
        register_kv_caches=registered.append,
        register_runner=lambda runner: registered.append(("runner", runner)),
    )
    monkeypatch.setattr(kv_cache_manager, "has_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(
        kv_cache_manager, "get_kv_transfer_group", lambda: fake_connector
    )
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.runner = _fresh_runner()
    manager.model_adapter = adapter = _Adapter(manager.runner)
    assert manager.get_kv_cache_spec() == {"layer.0": "spec"}
    config = NS(num_blocks=None)  # the ordinary path would assert on it
    manager.initialize_kv_cache(config)
    assert adapter.calls == [("init", config)]
    if connector:
        assert list(registered[0]) == ["layer.0"]
        assert registered[1] == ("runner", manager.runner)
    else:
        assert registered == []


def test_kv_connector_needs_the_adapter_caches(monkeypatch):
    monkeypatch.setattr(kv_cache_manager, "has_kv_transfer_group", lambda: True)
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.runner = NS()
    with pytest.raises(NotImplementedError):
        manager._register_kv_connector(None)


def _fresh_runner():
    return NS(kv_cache_config=None, attn_groups=[], kv_caches=[])


@pytest.mark.parametrize("skipped", ["kv_cache_config", "attn_groups", "kv_caches"])
def test_kv_cache_manager_checks_the_runner_state_the_adapter_must_set(
    monkeypatch, skipped
):
    monkeypatch.setattr(kv_cache_manager, "has_kv_transfer_group", lambda: False)
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.runner = _fresh_runner()
    adapter = _Adapter(manager.runner)
    full_init = adapter.initialize_kv_cache

    def init_but_skip(config):
        caches = full_init(config)
        setattr(manager.runner, skipped, getattr(_fresh_runner(), skipped))
        return caches

    adapter.initialize_kv_cache = init_but_skip
    manager.model_adapter = adapter
    with pytest.raises(RuntimeError, match=skipped):
        manager.initialize_kv_cache(NS(num_blocks=None))


def _support_runner(pp=1, lora=None, spec=None, async_drafter=False, mm=False):
    return NS(
        parallel_config=NS(pipeline_parallel_size=pp),
        lora_config=lora,
        supports_mm_inputs=mm,
        speculative_config=spec,
        _is_async_drafter=async_drafter,
        model_config=NS(architecture=ARCH),
    )


def test_adapter_supports_one_stage_no_lora_and_the_async_drafter():
    TPUModelRunner._check_model_adapter_support(_support_runner())
    TPUModelRunner._check_model_adapter_support(
        _support_runner(
            spec=NS(method="mtp", enable_adaptive_verification=False),
            async_drafter=True,
        )
    )


@pytest.mark.parametrize(
    "kwargs, what",
    [
        ({"pp": 2}, "pipeline parallelism"),
        ({"lora": object()}, "LoRA"),
        ({"mm": True}, "multimodal inputs"),
        (
            {"spec": NS(method="ngram", enable_adaptive_verification=False)},
            "'ngram'",
        ),
        (
            {
                "spec": NS(method="dspark", enable_adaptive_verification=True),
                "async_drafter": True,
            },
            "adaptive verification",
        ),
    ],
)
def test_adapter_rejects_unsupported_configurations_at_load(kwargs, what):
    with pytest.raises(NotImplementedError, match=what):
        TPUModelRunner._check_model_adapter_support(_support_runner(**kwargs))


def test_a_different_factory_for_the_same_architecture_warns(monkeypatch):
    warned = []
    monkeypatch.setattr(
        model_adapters.logger, "warning", lambda msg, *args: warned.append(args[0])
    )
    register_model_adapter([ARCH], _Adapter)
    register_model_adapter([ARCH], _Adapter)  # the same factory again: quiet
    register_model_adapter([ARCH], lambda runner: None)
    assert warned == [ARCH]


def test_capture_model_precompiles_the_adapter_inputs(monkeypatch):
    """capture_model hands the adapter the token buckets and both row options
    after the backbone, inside the same precompile pass."""
    monkeypatch.setattr(
        tpu_runner, "maybe_create_mm_encoder_manager", lambda *_args: None
    )
    runner = MagicMock(spec=TPUModelRunner)
    runner.parallel_config = NS(pipeline_parallel_size=1)
    runner.enforce_eager = False
    runner.lora_config = None
    runner.vllm_config = runner.device = runner.model = None
    runner._mtp_prefill_only = False
    runner._is_async_drafter = False
    runner.num_tokens_paddings = [16, 32]
    runner.num_reqs_max_model_len, runner.num_reqs_most_model_len = 4, None
    calls = []
    runner._precompile_backbone.side_effect = lambda: calls.append("backbone")
    runner.model_adapter = MagicMock(spec=ModelAdapter)
    runner.model_adapter.precompile_fused_inputs.side_effect = (
        lambda paddings, rows: calls.append(("adapter", paddings, rows))
    )
    TPUModelRunner.capture_model(runner)
    assert calls == ["backbone", ("adapter", [16, 32], (4, None))]


class _StopAfterAdapterSelection(RuntimeError):
    pass


def _load_model_runner(monkeypatch, *, pp=1):
    """The CPU stand-in of `load_model` (as in
    test_load_model_eagerly_initializes_sampling_generator_before_transfers),
    stopped at the quantization config right before weight loading."""
    real_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda dev, *args, **kwargs: (
            real_device("cpu") if dev == "tpu" else real_device(dev, *args, **kwargs)
        ),
    )
    reached = []

    def stop(_vllm_config):
        reached.append("get_tpu_quantization_config")
        raise _StopAfterAdapterSelection

    monkeypatch.setattr(tpu_runner, "get_tpu_quantization_config", stop)
    runner = NS(
        device_config=NS(device=real_device("cpu")),
        device=real_device("cpu"),
        vllm_config=NS(device_config=None, quant_config=None),
        model_config=NS(quantization=None, seed=0, architecture=ARCH),
        kv_cache_manager=NS(model_adapter=None),
        is_pooling_model=True,
        parallel_config=NS(pipeline_parallel_size=pp),
        lora_config=None,
        supports_mm_inputs=False,
        speculative_config=None,
        _is_async_drafter=False,
    )
    runner._check_model_adapter_support = (
        TPUModelRunner._check_model_adapter_support.__get__(runner)
    )
    return runner, reached


def test_load_model_selects_the_registered_adapter(monkeypatch):
    """load_model builds the registered adapter for the runner, checks its
    configuration and hands the same adapter to the KV cache manager before
    weight loading."""
    register_model_adapter([ARCH], _Adapter)
    runner, reached = _load_model_runner(monkeypatch)
    with pytest.raises(_StopAfterAdapterSelection):
        TPUModelRunner.load_model(runner)
    assert reached == ["get_tpu_quantization_config"]
    assert isinstance(runner.model_adapter, _Adapter)
    assert runner.model_adapter.runner is runner
    assert runner.kv_cache_manager.model_adapter is runner.model_adapter


def test_load_model_rejects_an_unsupported_adapter_configuration(monkeypatch):
    """The support check runs in load_model, before weight loading and before
    the KV cache manager gets the adapter."""
    register_model_adapter([ARCH], _Adapter)
    runner, reached = _load_model_runner(monkeypatch, pp=2)
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        TPUModelRunner.load_model(runner)
    assert reached == []
    assert runner.kv_cache_manager.model_adapter is None
