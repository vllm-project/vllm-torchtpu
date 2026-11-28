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
"""Unit tests for TpuOmniTorchProfilerWrapper on the real vllm-omni base class.

torch.profiler.profile and TpuProfilerConfig are stubbed, so no TPU is used.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.config import ProfilerConfig

from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.profiler import TpuOmniTorchProfilerWrapper

pytestmark = pytest.mark.omni

_PROFILER_TS = "2026_01_01_12_00_00"


def _profiler_config(tmp_path, with_stack=False):
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_with_stack=with_stack,
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=False,
        torch_profiler_with_flops=True,
    )


@pytest.fixture
def tpu_profiler(monkeypatch, tmp_path):
    """Stubs torch.profiler and the TPU profiler config for the wrapper tests.

    torch.distributed starts out uninitialized and profiler_trace is the real
    module. ``fake_rank(rank, world_size)`` fakes a distributed run, stubs the
    profiler_trace helpers and returns those stubs.
    """
    # The python_tracer_level asserts expect the default level.
    monkeypatch.delenv("PYTHON_TRACER_LEVEL", raising=False)
    mocks = SimpleNamespace(profile=MagicMock(), tpu_config=MagicMock())
    monkeypatch.setattr("torch.profiler.profile", mocks.profile)
    monkeypatch.setattr(
        "vllm_torchtpu.omni.profiler.TpuProfilerConfig", mocks.tpu_config
    )
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

    def fake_rank(rank, world_size):
        monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
        monkeypatch.setattr("torch.distributed.get_rank", lambda: rank)
        monkeypatch.setattr("torch.distributed.get_world_size", lambda: world_size)
        trace = SimpleNamespace(
            resolve=MagicMock(return_value=_PROFILER_TS),
            capture=MagicMock(return_value=str(tmp_path / f"rank_{rank}")),
            merge=MagicMock(),
            clear=MagicMock(),
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.resolve_canonical_dst_ts", trace.resolve
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.rank_capture_dir", trace.capture
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.merge_rank_capture", trace.merge
        )
        monkeypatch.setattr(
            "vllm_torchtpu.profiler_trace.clear_canonical_ts_marker", trace.clear
        )
        return trace

    mocks.fake_rank = fake_rank
    return mocks


def test_platform_returns_tpu_profiler_cls():
    assert (
        OmniTpuPlatform().get_profiler_cls()
        == "vllm_torchtpu.omni.profiler.TpuOmniTorchProfilerWrapper"
    ), "the Omni TPU platform must hand Omni the TPU profiler wrapper"


def test_tpu_omni_profiler_init(tmp_path):
    wrapper = TpuOmniTorchProfilerWrapper(
        _profiler_config(tmp_path, with_stack=True), "worker_0", 0
    )

    assert wrapper._activities == ["CPU", "TPU"], (
        "default activities must be ['CPU', 'TPU']; 'TPU' is a placeholder, "
        "PrivateUse1 is added in _start"
    )
    assert wrapper.dump_cpu_time_total is False, (
        "two default activities must keep the base CPU-only time table "
        "(dump_cpu_time_total) off"
    )
    assert wrapper.profiler is None, (
        "torch.profiler must not be built in __init__; _start builds it once "
        "the capture dir is known"
    )
    assert wrapper.profiler_config.torch_profiler_with_stack is False, (
        "the base config must get with_stack=False even when stacks are "
        "requested; the TPU python tracer records them instead"
    )


def test_tpu_omni_profiler_start_stop_merges_rank(tmp_path, tpu_profiler):
    wrapper = TpuOmniTorchProfilerWrapper(
        _profiler_config(tmp_path, with_stack=True), "worker_0", 0
    )
    trace = tpu_profiler.fake_rank(2, world_size=4)

    wrapper._start()
    assert trace.resolve.called, "resolve_canonical_dst_ts was not called in _start()"
    assert trace.capture.called, "rank_capture_dir was not called in _start()"
    tpu_profiler.profile.assert_called_once()
    _, tpu_kwargs = tpu_profiler.tpu_config.call_args
    assert tpu_kwargs.get("python_tracer_level") == 1, (
        "with_stack=True must keep the TPU python tracer on (default level 1), "
        f"got {tpu_kwargs!r}"
    )

    wrapper._stop()
    tpu_profiler.profile.return_value.stop.assert_called_once()
    trace.merge.assert_called_once_with(
        str(tmp_path / "rank_2"),
        str(tmp_path / "worker_0"),
        _PROFILER_TS,
        2,
        world_size=4,
    )
    assert not trace.clear.called, (
        "only rank 0 clears the canonical ts marker; this is rank 2"
    )


def test_tpu_omni_profiler_shared_run_root(tmp_path, tpu_profiler):
    # Same trace filename on two workers in one process: one shared run root.
    cfg = _profiler_config(tmp_path)
    w1 = TpuOmniTorchProfilerWrapper(cfg, "worker_1", 0)
    w1.set_trace_filename("my_trace")
    w2 = TpuOmniTorchProfilerWrapper(cfg, "worker_2", 0)
    w2.set_trace_filename("my_trace")

    w1._start()
    w2._start()

    run_root = w1._capture_state.run_root
    assert run_root == w2._capture_state.run_root, (
        "workers with the same trace filename must share one run root so "
        "their ranks merge into one capture"
    )
    assert run_root == os.path.join(str(tmp_path), "my_trace"), (
        f"run root must be <torch_profiler_dir>/<trace filename>, got {run_root}"
    )

    session_dir = w1._ensure_session_dir()
    assert os.path.dirname(session_dir) == str(tmp_path), (
        f"Omni session dir must sit directly under torch_profiler_dir, "
        f"got {session_dir}"
    )
    assert os.path.dirname(run_root) == str(tmp_path), (
        f"TPU run root must sit directly under torch_profiler_dir, got {run_root}"
    )
    assert os.path.basename(session_dir).endswith("_my_trace"), (
        f"session dir must be named <ts>_<trace filename>, got {session_dir}"
    )

    _, tpu_kwargs = tpu_profiler.tpu_config.call_args
    assert tpu_kwargs.get("python_tracer_level") == 0, (
        "with_stack=False must turn the TPU python tracer off (checked on "
        f"w2's _start, the last TpuProfilerConfig call), got {tpu_kwargs!r}"
    )

    w1._stop()
    w2._stop()


def test_tpu_omni_profiler_stop_error_still_merges(tmp_path, tpu_profiler):
    wrapper = TpuOmniTorchProfilerWrapper(_profiler_config(tmp_path), "w3", 0)
    trace = tpu_profiler.fake_rank(0, world_size=1)
    wrapper._start()
    wrapper.profiler.stop.side_effect = ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        wrapper._stop()

    # _stop's finally must still merge and (rank 0) clear the marker when
    # super()._stop() raises.
    trace.merge.assert_called_once()
    trace.clear.assert_called_once()


def test_tpu_omni_profiler_canonical_ts_sync(tmp_path, tpu_profiler, monkeypatch):
    cfg = _profiler_config(tmp_path)
    w0 = TpuOmniTorchProfilerWrapper(cfg, "w0", 0)
    w0.set_trace_filename("my_trace")
    w1 = TpuOmniTorchProfilerWrapper(cfg, "w1", 0)
    w1.set_trace_filename("my_trace")

    # Two ranks of one distributed run, sharing the real profiler_trace marker.
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    w0._start()
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 1)
    w1._start()

    ts0, ts1 = w0._capture_state.canonical_ts, w1._capture_state.canonical_ts
    assert ts0 == ts1, (
        "both ranks of one run must resolve the same canonical ts so their "
        f"XSpaces land in one plugins/profile/<ts> dir, got {ts0} vs {ts1}"
    )


def test_tpu_omni_profiler_forwards_profiler_kwargs(tmp_path, tpu_profiler):
    wrapper = TpuOmniTorchProfilerWrapper(_profiler_config(tmp_path), "w_kwargs", 0)
    wrapper._profiler_kwargs = {
        "host_tracer_level": 3,
        "python_tracer_level": 1,
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "e2e_enable_fw_throttle_event": True,
    }

    wrapper._start()

    _, tpu_kwargs = tpu_profiler.tpu_config.call_args
    assert tpu_kwargs["host_tracer_level"] == 3
    assert tpu_kwargs["python_tracer_level"] == 1
    assert (
        tpu_kwargs["experimental_options"]["tpu_trace_mode"] == "TRACE_COMPUTE_AND_SYNC"
    )
    assert tpu_kwargs["experimental_options"]["e2e_enable_fw_throttle_event"] is True
    wrapper._stop()
