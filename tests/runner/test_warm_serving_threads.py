# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""VLLM_TPU_WARM_SERVING_THREADS and VLLM_TPU_SAMPLING_COPIES_WHOLE: the
precompile passes run at serving's torch thread count (dynamo guards on it),
and InputBatch's device sampling refresh writes one shape per buffer instead
of a slice per live request count."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _copy_slice_whole,
    _serving_num_threads,
)


def _timed_runner():
    return SimpleNamespace(_update_num_xla_graphs=lambda name: None)


@pytest.fixture
def restore_threads():
    saved = torch.get_num_threads()
    yield
    torch.set_num_threads(saved)


def test_serving_num_threads_follows_vllm_runtime_rule(monkeypatch, restore_threads):
    torch.set_num_threads(3)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert _serving_num_threads() == 1
    # vLLM's own startup value is dropped for serving ...
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
    assert _serving_num_threads() == 1
    # ... a user-set OMP_NUM_THREADS is kept.
    monkeypatch.delenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM")
    assert _serving_num_threads() == 3


@pytest.mark.parametrize("flag", ["0", "1"])
def test_precompile_timed_runs_at_serving_threads(monkeypatch, restore_threads, flag):
    monkeypatch.setenv("VLLM_TPU_WARM_SERVING_THREADS", flag)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    torch.set_num_threads(3)
    with TPUModelRunner._precompile_timed(_timed_runner(), "x"):
        inside = torch.get_num_threads()
    assert inside == (1 if flag == "1" else 3)
    assert torch.get_num_threads() == 3  # restored for the rest of startup


@pytest.mark.parametrize("flag", ["0", "1"])
def test_warmed_region_is_not_recompiled_after_thread_drop(
    monkeypatch, restore_threads, flag
):
    """The served failure on CPU: a static region warmed inside a precompile
    pass at the startup thread count, then served after the drop to 1 thread
    (compile_or_warm_up_model). Without the flag dynamo recompiles
    (GLOBAL_STATE changed: num_threads); with it the warm entry is reused."""
    monkeypatch.setenv("VLLM_TPU_WARM_SERVING_THREADS", flag)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    torch._dynamo.reset()
    compiles = []

    def backend(gm, example_inputs):
        compiles.append(1)
        return gm.forward

    # Same body as _substitute_placeholder_token.
    fn = torch.compile(
        tpu_runner._substitute_placeholder_token.__wrapped__
        if hasattr(tpu_runner._substitute_placeholder_token, "__wrapped__")
        else tpu_runner._substitute_placeholder_token._torchdynamo_orig_callable,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )

    def args():
        return (
            torch.zeros(8, dtype=torch.int32),
            torch.arange(8, dtype=torch.int32),
            torch.full((8,), -1, dtype=torch.int32),
            torch.zeros(128, dtype=torch.int64),
        )

    torch.set_num_threads(3)  # startup count, as in initialize_kv_cache
    with torch.no_grad(), TPUModelRunner._precompile_timed(_timed_runner(), "x"):
        fn(*args())
    assert len(compiles) == 1
    torch.set_num_threads(1)  # set_torch_threads_for_runtime
    with torch.no_grad():
        out = fn(*args())
    assert torch.equal(out, torch.zeros(8, dtype=torch.int32))
    assert len(compiles) == (1 if flag == "1" else 2)
    torch._dynamo.reset()


def test_copy_slice_whole_same_visible_values():
    """The returned view and every slot below `length` match upstream's
    copy_slice; the whole destination is written (one shape)."""
    from vllm.v1.utils import copy_slice

    gen = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.int32):
        src = (torch.rand(32, generator=gen) * 100).to(dtype)
        for length in (1, 7, 31, 32):
            ref_dst = torch.full((32,), 5, dtype=dtype)
            new_dst = torch.full((32,), 5, dtype=dtype)
            want = copy_slice(src, ref_dst, length)
            writes = []
            real_copy = torch.Tensor.copy_

            def spy(self, other, non_blocking=False, _w=writes, _c=real_copy):
                _w.append(tuple(self.shape))
                return _c(self, other, non_blocking=non_blocking)

            torch.Tensor.copy_ = spy
            try:
                got = _copy_slice_whole(src, new_dst, length)
            finally:
                torch.Tensor.copy_ = real_copy
            assert writes == [(32,)]  # one whole-buffer write, never [:length]
            assert got.shape == want.shape == (length,)
            assert torch.equal(got, want)
            assert torch.equal(new_dst[:length], ref_dst[:length])
            assert torch.equal(new_dst, src)


def test_copy_slice_whole_keeps_upstream_path_for_other_tensors():
    src = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    dst = torch.zeros(4, 3)
    out = _copy_slice_whole(src, dst, 2)
    assert out.shape == (2, 3) and torch.equal(dst[:2], src[:2])
    assert torch.equal(dst[2:], torch.zeros(2, 3))


def test_install_points_input_batch_at_whole_copy(monkeypatch):
    from vllm.v1.worker import gpu_input_batch

    monkeypatch.setattr(gpu_input_batch, "copy_slice", gpu_input_batch.copy_slice)
    assert tpu_runner._install_sampling_copies_whole()
    assert gpu_input_batch.copy_slice is _copy_slice_whole


@pytest.mark.parametrize("flag", ["0", "1"])
def test_precompile_sampling_copies(monkeypatch, flag):
    monkeypatch.setenv("VLLM_TPU_SAMPLING_COPIES_WHOLE", flag)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a: None)
    seen = []

    def record(src, dst, length):
        seen.append((dst.shape[0], length, src.dtype, torch.is_grad_enabled()))
        return dst[:length]

    monkeypatch.setattr(tpu_runner, "_copy_slice_whole", record)
    ib = SimpleNamespace(
        temperature=torch.zeros(32),
        temperature_cpu_tensor=torch.ones(32),
        top_k=torch.zeros(32, dtype=torch.int32),
        top_k_cpu_tensor=torch.ones(32, dtype=torch.int32),
    )
    runner = SimpleNamespace(
        input_batch=ib,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    TPUModelRunner._precompile_sampling_copies(runner)
    if flag == "0":
        assert seen == []
        return
    assert seen == [
        (32, 32, torch.float32, False),
        (32, 32, torch.int32, False),
    ]
