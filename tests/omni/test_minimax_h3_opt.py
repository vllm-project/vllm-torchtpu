# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 DiT step-time TPU patches.

The vLLM-Omni modules under test are stand-ins exposing the attributes the
patches replace, registered in ``sys.modules`` under the upstream names. The
AdaLN cache check also loads vLLM-Omni's model-independent
exact_projection_cache.py from sys.path by file location (skipped if absent).
TPU dispatch is exercised by forcing ``minimax_h3._is_tpu``; the math stays on
CPU.
"""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.omni import minimax_h3

pytestmark = pytest.mark.omni

_TPU_DEVICE = SimpleNamespace(type="tpu")


def _install(monkeypatch, module: types.ModuleType) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _force_tpu(monkeypatch) -> None:
    """Dispatch every tensor as a torch_tpu tensor (the math stays on CPU)."""
    monkeypatch.setattr(minimax_h3, "_is_tpu", lambda value: True)


def _fake_pipeline(monkeypatch) -> types.ModuleType:
    """Minimal pipeline_minimax_h3 stand-in so apply_minimax_h3_patches runs."""
    pipeline = types.ModuleType(minimax_h3.PIPELINE_MODULE)
    pipeline.dist = torch.distributed

    def _broadcast_tensor(tensor, *, dtype, device):
        raise AssertionError("unpatched _broadcast_tensor")

    class MiniMaxH3Pipeline:
        def _encoder_group_broadcast_tensor(self, tensor, *, dtype, device):
            raise AssertionError("unpatched")

        def _distribute_encode_inputs(self, ids, vision_kwargs):
            raise AssertionError("unpatched")

    pipeline._broadcast_tensor = _broadcast_tensor
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    monkeypatch.setitem(sys.modules, minimax_h3.INDEXED_MODULATION_MODULE, None)
    return _install(monkeypatch, pipeline)


def test_is_tpu():
    assert minimax_h3._is_tpu(SimpleNamespace(device=_TPU_DEVICE))
    assert minimax_h3._is_tpu(_TPU_DEVICE)
    assert minimax_h3._is_tpu("tpu") and minimax_h3._is_tpu("tpu:3")
    assert not minimax_h3._is_tpu(torch.zeros(1))
    assert not minimax_h3._is_tpu(torch.device("cpu"))
    assert not minimax_h3._is_tpu(torch.device("meta"))
    assert not minimax_h3._is_tpu("cpu")
    assert not minimax_h3._is_tpu(None)


# -----------------------------------------------------------------------------
# (a) AdaLN projection cache
# -----------------------------------------------------------------------------


def _fake_adaln_module(monkeypatch, base: type = object):
    """adaln_cache stand-in; ``prepare`` mirrors adaln_cache.py:70-82."""
    module = types.ModuleType(minimax_h3.ADALN_CACHE_MODULE)
    calls = []

    class MiniMaxH3RuntimeAdalnCache(base):
        def prepare(self, embedding):
            """Upstream prepare."""
            calls.append(embedding)
            self._sidecar_plan = None
            if base is object:
                self._input = embedding
                self._key = "digest"
            else:
                super().prepare(embedding)
            self._sidecar_plan = "plan"

    module.MiniMaxH3RuntimeAdalnCache = MiniMaxH3RuntimeAdalnCache
    return _install(monkeypatch, module), calls


def test_adaln_cache_prepare_dispatch(monkeypatch):
    module, calls = _fake_adaln_module(monkeypatch)
    cls = module.MiniMaxH3RuntimeAdalnCache
    original = cls.prepare
    assert minimax_h3.patch_adaln_projection_cache()
    assert cls._tpu_orig_prepare is original
    assert cls.prepare.__wrapped__.__wrapped__ is original
    assert cls.prepare.__doc__ == "Upstream prepare."

    # CPU (and CUDA) tensors arm the cache exactly as upstream does.
    cache = cls()
    embedding = torch.randn(2, 4)
    cache.prepare(embedding)
    assert calls == [embedding]
    assert cache._key == "digest" and cache._input is embedding
    assert cache._sidecar_plan == "plan"

    # TPU tensors leave it unarmed without running the host digest.
    _force_tpu(monkeypatch)
    cache.prepare(torch.randn(2, 4))
    assert calls == [embedding]
    assert cache._key is None and cache._input is None
    assert cache._sidecar_plan is None


def test_adaln_cache_patch_idempotent_and_noop(monkeypatch):
    module, _ = _fake_adaln_module(monkeypatch)
    assert minimax_h3.patch_adaln_projection_cache()
    patched = module.MiniMaxH3RuntimeAdalnCache.prepare
    assert not minimax_h3.patch_adaln_projection_cache()
    assert module.MiniMaxH3RuntimeAdalnCache.prepare is patched

    monkeypatch.setitem(sys.modules, minimax_h3.ADALN_CACHE_MODULE, None)
    assert minimax_h3.patch_adaln_projection_cache() is False
    monkeypatch.delitem(sys.modules, minimax_h3.ADALN_CACHE_MODULE)
    assert minimax_h3.patch_adaln_projection_cache() is False
    assert minimax_h3.ADALN_CACHE_MODULE not in sys.modules


@pytest.fixture(scope="module")
def upstream_exact_projection_cache():
    """vLLM-Omni's exact_projection_cache.py, loaded by path under a private name.

    tests/omni/mock_omni.py may replace ``vllm_omni`` in ``sys.modules``; the
    file only imports torch (and vllm.distributed inside ``project``).
    """
    relative = os.path.join(
        "vllm_omni", "diffusion", "cache", "exact_projection_cache.py"
    )
    for entry in sys.path:
        path = os.path.join(entry or os.getcwd(), relative)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(
                "_upstream_exact_projection_cache", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip(f"{relative} not found on sys.path")


def test_adaln_cache_tpu_projects_without_tp_vote(
    monkeypatch, upstream_exact_projection_cache
):
    distributed = pytest.importorskip("vllm.distributed")

    def no_vote():
        raise AssertionError("TP hit vote")

    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(distributed, "get_tp_group", no_vote)
    module, _ = _fake_adaln_module(
        monkeypatch, upstream_exact_projection_cache.ExactProjectionCache
    )
    cls = module.MiniMaxH3RuntimeAdalnCache
    linear = torch.nn.Linear(4, 6)
    embedding = torch.randn(3, 4)
    computed = []

    def compute():
        computed.append(1)
        return linear(embedding)

    with torch.inference_mode():
        # Upstream: an armed cache votes over the TP group before projecting.
        cache = cls(max_bytes=1 << 20)
        cache.prepare(embedding)
        assert cache._key is not None
        with pytest.raises(AssertionError, match="TP hit vote"):
            cache.project("blocks.0.adaln_proj.linear", linear, embedding, compute)

        assert minimax_h3.patch_adaln_projection_cache()
        _force_tpu(monkeypatch)
        cache = cls(max_bytes=1 << 20)
        cache.prepare(embedding)
        expected = linear(embedding)
        for _ in range(2):
            out = cache.project(
                "blocks.0.adaln_proj.linear", linear, embedding, compute
            )
            assert torch.equal(out, expected)
    assert computed == [1, 1]
    assert cache.hits == 0 and cache.misses == 0


def test_apply_installs_adaln_cache_patch(monkeypatch):
    _fake_pipeline(monkeypatch)
    module, _ = _fake_adaln_module(monkeypatch)
    assert minimax_h3.apply_minimax_h3_patches()
    assert "_tpu_orig_prepare" in module.MiniMaxH3RuntimeAdalnCache.__dict__
    patched = module.MiniMaxH3RuntimeAdalnCache.prepare
    assert not minimax_h3.apply_minimax_h3_patches()
    assert module.MiniMaxH3RuntimeAdalnCache.prepare is patched
