# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 Ulysses full-span embed TPU patch.

The vLLM-Omni transformer and SP parallel_state modules are stand-ins
registered in ``sys.modules`` under the upstream names. The stand-in
``MiniMaxH3DiTModel._rope_local_span`` returns a 1/U slice like upstream's
(minimax_h3_transformer.py:1457-1467 at vllm-omni e1d77a3). TPU dispatch is
exercised by forcing ``minimax_h3._is_tpu``; everything runs on CPU.
"""

import sys
import types

import pytest
import torch

from vllm_torchtpu.omni import minimax_h3

pytestmark = pytest.mark.omni

SEQ_LEN = 2432


def _install(monkeypatch, module: types.ModuleType) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def _fake_transformer_module(monkeypatch) -> types.ModuleType:
    module = types.ModuleType(minimax_h3.TRANSFORMER_MODULE)

    class MiniMaxH3DiTModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)
            self.span_calls = []

        def _rope_local_span(self, seq_len: int) -> tuple[int, int]:
            self.span_calls.append(seq_len)
            world = minimax_h3.sequence_parallel_world_size()
            return 0, seq_len // world

    module.MiniMaxH3DiTModel = MiniMaxH3DiTModel
    return _install(monkeypatch, module)


def _set_sp_world(monkeypatch, world: int | None) -> None:
    """Install a parallel_state stand-in; ``None`` means SP not initialized."""
    module = types.ModuleType(minimax_h3.SP_PARALLEL_STATE_MODULE)

    def get_sequence_parallel_world_size() -> int:
        assert world is not None, "pipeline model parallel group is not initialized"
        return world

    module.get_sequence_parallel_world_size = get_sequence_parallel_world_size
    _install(monkeypatch, module)


def _force_tpu(monkeypatch, on_tpu: bool = True) -> None:
    monkeypatch.setattr(minimax_h3, "_is_tpu", lambda value: on_tpu)


def test_sequence_parallel_world_size(monkeypatch):
    monkeypatch.setitem(sys.modules, minimax_h3.SP_PARALLEL_STATE_MODULE, None)
    assert minimax_h3.sequence_parallel_world_size() == 1
    _set_sp_world(monkeypatch, None)
    assert minimax_h3.sequence_parallel_world_size() == 1
    for world in (1, 2, 4, 8):
        _set_sp_world(monkeypatch, world)
        assert minimax_h3.sequence_parallel_world_size() == world


def test_patch_installs_idempotent_and_noop(monkeypatch):
    module = _fake_transformer_module(monkeypatch)
    cls = module.MiniMaxH3DiTModel
    original = cls._rope_local_span
    assert minimax_h3.patch_ulysses_full_span_embed()
    assert cls.__dict__["_tpu_orig_rope_local_span"] is original
    assert cls._rope_local_span.__wrapped__ is original
    patched = cls._rope_local_span
    assert not minimax_h3.patch_ulysses_full_span_embed()
    assert cls._rope_local_span is patched

    monkeypatch.setitem(sys.modules, minimax_h3.TRANSFORMER_MODULE, None)
    assert minimax_h3.patch_ulysses_full_span_embed() is False
    _install(monkeypatch, types.ModuleType(minimax_h3.TRANSFORMER_MODULE))
    assert minimax_h3.patch_ulysses_full_span_embed() is False


def test_subclass_of_patched_class_is_patched_once(monkeypatch):
    module = _fake_transformer_module(monkeypatch)
    assert minimax_h3.patch_ulysses_full_span_embed()
    patched = module.MiniMaxH3DiTModel._rope_local_span

    class Sub(module.MiniMaxH3DiTModel):
        pass

    module.MiniMaxH3DiTModel = Sub
    # The marker is looked up in the class __dict__, not inherited.
    assert minimax_h3.patch_ulysses_full_span_embed()
    assert Sub.__dict__["_tpu_orig_rope_local_span"] is patched


@pytest.mark.parametrize("on_tpu", [True, False])
def test_sp_world_one_calls_upstream(monkeypatch, on_tpu):
    module = _fake_transformer_module(monkeypatch)
    assert minimax_h3.patch_ulysses_full_span_embed()
    _force_tpu(monkeypatch, on_tpu)
    model = module.MiniMaxH3DiTModel()
    for world in (None, 1):
        _set_sp_world(monkeypatch, world)
        want = model._tpu_orig_rope_local_span(SEQ_LEN)
        assert model._rope_local_span(SEQ_LEN) == want == (0, SEQ_LEN)
    monkeypatch.setitem(sys.modules, minimax_h3.SP_PARALLEL_STATE_MODULE, None)
    assert model._rope_local_span(SEQ_LEN) == (0, SEQ_LEN)
    assert model.span_calls == [SEQ_LEN] * 5


@pytest.mark.parametrize("world", [2, 4, 8])
def test_cpu_keeps_upstream_local_span(monkeypatch, world):
    module = _fake_transformer_module(monkeypatch)
    assert minimax_h3.patch_ulysses_full_span_embed()
    _force_tpu(monkeypatch, False)
    _set_sp_world(monkeypatch, world)
    model = module.MiniMaxH3DiTModel()
    assert model._rope_local_span(SEQ_LEN) == (0, SEQ_LEN // world)
    assert model.span_calls == [SEQ_LEN]


@pytest.mark.parametrize("world", [2, 4, 8])
def test_tpu_sp_returns_full_span(monkeypatch, world):
    module = _fake_transformer_module(monkeypatch)
    assert minimax_h3.patch_ulysses_full_span_embed()
    _force_tpu(monkeypatch)
    _set_sp_world(monkeypatch, world)
    model = module.MiniMaxH3DiTModel()
    assert model._rope_local_span(SEQ_LEN) == (0, SEQ_LEN)
    assert model.span_calls == []


def test_tpu_dispatch_reads_model_parameters(monkeypatch):
    module = _fake_transformer_module(monkeypatch)
    assert minimax_h3.patch_ulysses_full_span_embed()
    _set_sp_world(monkeypatch, 4)
    seen = []

    def is_tpu(value):
        seen.append(value)
        return False

    monkeypatch.setattr(minimax_h3, "_is_tpu", is_tpu)
    model = module.MiniMaxH3DiTModel()
    model._rope_local_span(SEQ_LEN)
    assert len(seen) == 1 and seen[0] is next(model.parameters())


def test_apply_installs_full_span_patch(monkeypatch):
    # Minimal pipeline_minimax_h3 stand-in (as in test_minimax_h3_opt.py).
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
    _install(monkeypatch, pipeline)
    monkeypatch.setitem(sys.modules, minimax_h3.INDEXED_MODULATION_MODULE, None)
    for name in (
        minimax_h3.ADALN_CACHE_MODULE,
        minimax_h3.DENOISE_LOOP_MODULE,
    ):
        monkeypatch.setitem(sys.modules, name, None)
    module = _fake_transformer_module(monkeypatch)
    original = module.MiniMaxH3DiTModel._rope_local_span
    assert minimax_h3.apply_minimax_h3_patches()
    assert module.MiniMaxH3DiTModel._tpu_orig_rope_local_span is original
    assert not minimax_h3.apply_minimax_h3_patches()
