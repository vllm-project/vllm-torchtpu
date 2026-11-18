# SPDX-License-Identifier: Apache-2.0
"""The optional model-owned runner adapter: off by default, delegated when set."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_torchtpu.runner import kv_cache_manager, model_adapters
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.model_adapters import (
    register_model_adapter,
    select_model_adapter,
    unregister_model_adapter,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

ARCH = "ToyAdapterForCausalLM"


def _runner(arch=ARCH):
    return NS(model_config=NS(architecture=arch))


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(model_adapters, "_FACTORIES", {})


class _Adapter:
    def __init__(self, runner):
        self.runner = runner
        self.calls = []

    def get_kv_cache_spec(self):
        return {"layer.0": "spec"}

    def initialize_kv_cache(self, config):
        self.calls.append(("init", config))
        return {"layer.0": torch.zeros(1)}

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
    unregister_model_adapter([ARCH])
    assert select_model_adapter(runner) is None


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
    manager.runner = NS()
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
