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
"""Tests for TPUWorker"""

import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import ParallelConfig
from vllm.v1.engine.core import EngineCoreProc

import vllm_torchtpu as plugin
from vllm_torchtpu import patch_registry
from vllm_torchtpu.worker.tpu_worker import (TPUWorker,
                                             _configure_tpu_process_env)


def test_configure_tpu_process_env_sets_world_size_only_for_multiple_ranks(
        monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")

    _configure_tpu_process_env(rank=0,
                               local_rank=0,
                               world_size=1,
                               local_world_size=1)

    assert os.environ["RANK"] == "0"
    assert os.environ["LOCAL_RANK"] == "0"
    assert os.environ["LOCAL_WORLD_SIZE"] == "1"
    assert "WORLD_SIZE" not in os.environ

    _configure_tpu_process_env(rank=1,
                               local_rank=1,
                               world_size=8,
                               local_world_size=8)

    assert os.environ["RANK"] == "1"
    assert os.environ["LOCAL_RANK"] == "1"
    assert os.environ["LOCAL_WORLD_SIZE"] == "8"
    assert os.environ["WORLD_SIZE"] == "8"


def _make_vllm_config(profiler_torch_dir=None):
    """Build a minimal mock VllmConfig for TPUWorker.__init__."""
    cfg = MagicMock()
    cfg.model_config.dtype = torch.bfloat16
    cfg.cache_config.cache_dtype = "auto"
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.compilation_config.compile_ranges_endpoints = []
    cfg.profiler_config.torch_profiler_dir = profiler_torch_dir
    cfg.additional_config = {}
    return cfg


def _build_worker(vllm_config, rank=0):
    """Construct a TPUWorker with heavy side-effects mocked out."""
    # envs is deliberately not patched: __init__ reads nothing from it, and
    # profile() must see the real USE_PHASED_PROFILER that tests monkeypatch.
    with (
            patch("vllm_torchtpu.patch_registry.apply"),
            patch(
                "vllm_torchtpu.worker.tpu_worker.WorkerBase.__init__",
                return_value=None,
            ),
    ):

        worker = TPUWorker.__new__(TPUWorker)
        # Set attributes that WorkerBase.__init__ would normally set.
        worker.vllm_config = vllm_config
        worker.model_config = vllm_config.model_config
        worker.cache_config = vllm_config.cache_config
        worker.parallel_config = vllm_config.parallel_config
        worker.compilation_config = vllm_config.compilation_config
        worker.rank = rank
        worker.local_rank = 0

        # Now call __init__ (WorkerBase.__init__ is a no-op mock).
        worker.__init__(
            vllm_config=vllm_config,
            local_rank=0,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:29500",
            is_driver_worker=True,
        )
        return worker


class TestProfilerDir:
    """Verify profiler_config.torch_profiler_dir logic."""

    def test_uses_profiler_config_when_set(self):
        """profiler_config.torch_profiler_dir is used when set."""
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg)
        assert worker.profile_dir == "/config/profiler/dir"

    def test_every_rank_captures(self):
        """A worker only sees its own chips, so all ranks must capture."""
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg, rank=3)
        assert worker.profile_dir == "/config/profiler/dir"

    def test_no_profiler_when_not_set(self):
        """profile_dir is None when config is not set."""
        cfg = _make_vllm_config(profiler_torch_dir=None)
        worker = _build_worker(cfg)
        assert worker.profile_dir is None


class TestProfileDispatchesToPhasedProfiler:
    """USE_PHASED_PROFILER makes profile() arm/disarm TPUModelRunner's phase
    profiler instead of opening a worker-owned profiler context."""

    def test_start_arms_the_model_runners_phased_profiler(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg)
        worker.model_runner = MagicMock()

        worker.profile(is_start=True)

        worker.model_runner.start_phased_profiling.assert_called_once_with(
            None)
        worker.model_runner.stop_phased_profiling.assert_not_called()
        assert worker.profile_context is None

    def test_start_forwards_the_profile_prefix(self, monkeypatch):
        """The prefix must not be dropped just because this is a phased run."""
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg)
        worker.model_runner = MagicMock()

        worker.profile(is_start=True, profile_prefix="decode")

        worker.model_runner.start_phased_profiling.assert_called_once_with(
            "decode")

    def test_stop_disarms_the_model_runners_phased_profiler(self, monkeypatch):
        monkeypatch.setenv("USE_PHASED_PROFILER", "true")
        cfg = _make_vllm_config(profiler_torch_dir="/config/profiler/dir")
        worker = _build_worker(cfg)
        worker.model_runner = MagicMock()

        worker.profile(is_start=False)

        worker.model_runner.stop_phased_profiling.assert_called_once_with()
        worker.model_runner.start_phased_profiling.assert_not_called()


class TestProfileCaptureAndMerge:
    """Standard torch_profiler_dir flow: per-rank capture, merged run dir."""

    @staticmethod
    def _make_worker(profile_dir, rank, world_size):
        cfg = _make_vllm_config(profiler_torch_dir=str(profile_dir))
        worker = _build_worker(cfg, rank=rank)
        # init_device() normally derives these from the TPU rank binding.
        worker.profile_rank = rank
        worker.profile_world_size = world_size
        return worker

    @staticmethod
    def _stage_capture(worker, filename, content):
        """Stand in for the xprof trace handler writing on profiler exit."""
        ts_dir = (Path(worker.profile_capture_dir) / "plugins" / "profile" /
                  "pt_ts")
        ts_dir.mkdir(parents=True)
        (ts_dir / filename).write_text(content)

    def _run_profile_cycle(self, workers, filename="t1v-n-host-w-0.xplane.pb"):
        """Drive one start/stop cycle on every rank; return their run ts."""
        with patch("vllm_torchtpu.worker.tpu_worker.torch.profiler"):
            for worker in workers:
                worker.profile(is_start=True)
            canonical_ts = [w.profile_canonical_ts for w in workers]
            for worker in workers:
                self._stage_capture(worker, filename,
                                    f"rank_{worker.profile_rank}")
            for worker in workers:
                worker.profile(is_start=False)
        return canonical_ts

    def test_single_rank_capture_is_merged_into_run_dir(self, tmp_path):
        worker = self._make_worker(tmp_path, rank=0, world_size=1)

        with patch("vllm_torchtpu.worker.tpu_worker.torch.profiler"):
            worker.profile(is_start=True)
            capture_dir = Path(worker.profile_capture_dir)
            canonical_ts = worker.profile_canonical_ts
            assert capture_dir == tmp_path / "rank_0"
            self._stage_capture(worker, "t1v-n-host-w-0.xplane.pb", "trace")
            worker.profile(is_start=False)

        dst = tmp_path / "plugins" / "profile" / canonical_ts
        assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "trace"
        assert not (capture_dir / "plugins").exists()
        # The marker is dropped once the run it describes is merged.
        assert not list(tmp_path.glob(".canonical_ts_*"))

    def test_all_ranks_merge_into_one_run_dir(self, tmp_path):
        workers = [
            self._make_worker(tmp_path, rank=rank, world_size=4)
            for rank in range(4)
        ]

        canonical_ts = set(self._run_profile_cycle(workers))

        # Every rank agreed on rank 0's run directory...
        assert len(canonical_ts) == 1
        dst = tmp_path / "plugins" / "profile" / canonical_ts.pop()
        # ...and same-named per-host xplane files did not collide.
        for rank in range(4):
            assert (dst /
                    f"rank{rank}_t1v-n-host-w-0.xplane.pb").read_text() == (
                        f"rank_{rank}")

    def test_back_to_back_runs_get_separate_run_dirs(self, tmp_path):
        workers = [
            self._make_worker(tmp_path, rank=rank, world_size=2)
            for rank in range(2)
        ]

        first_ts = self._run_profile_cycle(workers,
                                           filename="first.xplane.pb")[0]
        # The second cycle must not reuse the first cycle's marker; force a
        # distinct timestamp so the two runs are distinguishable.
        with patch("vllm_torchtpu.profiler_trace.datetime") as mock_dt:
            mock_dt.datetime.now.return_value.strftime.return_value = (
                "2026_05_06_04_47_36")
            second_ts = self._run_profile_cycle(workers,
                                                filename="second.xplane.pb")

        assert second_ts == ["2026_05_06_04_47_36"] * 2
        assert second_ts[0] != first_ts
        profiles = tmp_path / "plugins" / "profile"
        assert (profiles / first_ts / "rank0_first.xplane.pb").exists()
        assert (profiles / second_ts[0] / "rank1_second.xplane.pb").exists()

    def test_profile_prefix_scopes_the_run_dir(self, tmp_path):
        worker = self._make_worker(tmp_path, rank=0, world_size=1)

        with patch("vllm_torchtpu.worker.tpu_worker.torch.profiler"):
            worker.profile(is_start=True, profile_prefix="decode")
            canonical_ts = worker.profile_canonical_ts
            assert Path(worker.profile_capture_dir) == (tmp_path / "decode" /
                                                        "rank_0")
            self._stage_capture(worker, "t1v-n-host-w-0.xplane.pb", "trace")
            # The merge follows the run the capture was started under, so a
            # stop that forgets the prefix still lands in the right place.
            worker.profile(is_start=False)

        dst = tmp_path / "decode" / "plugins" / "profile" / canonical_ts
        assert (dst / "t1v-n-host-w-0.xplane.pb").read_text() == "trace"

    def test_stop_without_start_is_a_noop(self, tmp_path):
        worker = self._make_worker(tmp_path, rank=0, world_size=1)

        with patch("vllm_torchtpu.worker.tpu_worker.torch.profiler"):
            worker.profile(is_start=False)

        assert not list(tmp_path.iterdir())


def test_initialize_from_config_updates_num_gpu_blocks():
    worker = TPUWorker.__new__(TPUWorker)
    worker.cache_config = MagicMock(num_gpu_blocks=None)
    worker.vllm_config = MagicMock()
    worker.model_runner = MagicMock()
    kv_cache_config = MagicMock(num_blocks=2048)

    with patch("vllm_torchtpu.worker.tpu_worker."
               "ensure_kv_transfer_initialized"):
        worker.initialize_from_config(kv_cache_config)

    assert worker.cache_config.num_gpu_blocks == 2048
    worker.model_runner.initialize_kv_cache.assert_called_once_with(
        kv_cache_config)


@patch("vllm_torchtpu.worker.tpu_worker.get_pp_group")
@patch("vllm_torchtpu.worker.tpu_worker.get_tensor_model_parallel_rank",
       return_value=3)
@patch("vllm_torchtpu.worker.tpu_worker.get_kv_transfer_group")
@patch("vllm_torchtpu.worker.tpu_worker.has_kv_transfer_group",
       return_value=True)
def test_kv_connector_handshake_metadata_uses_pp_tp_rank_key(
        _has_group, get_group, _get_tp_rank, get_pp_group):
    metadata = object()
    get_group.return_value.get_handshake_metadata.return_value = metadata
    get_pp_group.return_value.rank_in_group = 2
    worker = TPUWorker.__new__(TPUWorker)

    assert worker.get_kv_connector_handshake_metadata() == {(2, 3): metadata}


def test_model_load_stage_precedes_runner(monkeypatch):
    events = []
    config = object()
    monkeypatch.setattr(patch_registry, "apply",
                        lambda stage, **kwargs: events.append((stage, kwargs)))
    worker = SimpleNamespace(
        model_config=config,
        model_runner=SimpleNamespace(load_model=lambda: events.append("load")))
    TPUWorker.load_model(worker)
    assert events == [("model_load", {"model_config": config}), "load"]


@pytest.mark.parametrize("managed_by_vllm, runtime_threads", [(True, 1),
                                                              (False, 2)])
def test_warmup_restores_runtime_threads_without_overriding_user_choice(
        monkeypatch, managed_by_vllm, runtime_threads):
    from vllm.utils.torch_utils import OMP_NUM_THREADS_SET_BY_VLLM

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    if managed_by_vllm:
        monkeypatch.setenv(OMP_NUM_THREADS_SET_BY_VLLM, "1")
    else:
        monkeypatch.delenv(OMP_NUM_THREADS_SET_BY_VLLM, raising=False)

    warmup_threads = []

    def capture_model():
        warmup_threads.append(torch.get_num_threads())

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(capture_model=capture_model),
        compilation_config=SimpleNamespace(compilation_time=0.0),
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = TPUWorker.compile_or_warm_up_model(worker)
        assert warmup_threads == [2]
        assert torch.get_num_threads() == runtime_threads
        assert result.language_model == worker.compilation_config.compilation_time
        assert result.encoder == 0.0
    finally:
        torch.set_num_threads(previous_threads)


def test_real_patch_entrypoints_in_spawned_processes():
    import subprocess
    import sys

    probe = Path(__file__).resolve()
    source = probe.parents[2] / "src"
    env = dict(os.environ, PYTHONPATH=str(source), JAX_PLATFORMS="tpu")
    result = subprocess.run([sys.executable, str(probe)],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: spawned engine_core applied real patches" in result.stdout
    assert "PASS: spawned worker_init applied real patches" in result.stdout


_original_engine_run = EngineCoreProc.run_engine_core
_PARENT_MARKER = "parent_process_only"


def _check_engine_patches():
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import core

    assert _PARENT_MARKER not in patch_registry._applied
    expected = {
        p.target
        for p in patch_registry.PATCHES if "engine_core" in p.stages
    }
    assert expected <= patch_registry._applied
    assert core.resolve_kv_cache_block_sizes is kv_cache_utils.resolve_kv_cache_block_sizes
    assert Scheduler._mamba_block_aligned_split._tpu_scheduler_block_size_patch
    assert Scheduler._tpu_hybrid_producer_prefix_hit_patch
    assert EngineCoreProc.run_engine_core is plugin._run_engine_core_with_tpu_patches
    assert EngineCoreProc._tpu_original_run_engine_core is _original_engine_run
    wrapper = Scheduler.__init__
    patch_registry.apply("engine_core")
    assert Scheduler.__init__ is wrapper


def _engine_init(self, connection, *, engine_index, vllm_config):
    _check_engine_patches()
    assert engine_index == 0
    connection.send(("engine_core", os.getpid()))
    connection.close()
    raise SystemExit(0)


def _worker_entry(connection):
    from vllm.config.compilation import CompilationConfig

    from vllm_torchtpu.worker.tpu_worker import TPUWorker, WorkerBase

    assert _PARENT_MARKER not in patch_registry._applied
    assert not any(
        "worker_init" in p.stages and p.target in patch_registry._applied
        for p in patch_registry.PATCHES)

    def before_worker_base(self, **kwargs):
        _check_engine_patches()
        assert "vllm_torchtpu:_patch_vllm_disable_compile_ranges" in patch_registry._applied
        assert CompilationConfig.get_compile_ranges(None) == []
        wrapper = CompilationConfig.get_compile_ranges
        patch_registry.apply("worker_init")
        assert CompilationConfig.get_compile_ranges is wrapper
        connection.send(("worker_init", os.getpid()))
        connection.close()
        raise SystemExit(0)

    with patch.object(WorkerBase, "__init__", before_worker_base):
        TPUWorker(vllm_config=None,
                  local_rank=0,
                  rank=0,
                  distributed_init_method="tcp://127.0.0.1:29500")


if __name__ == "__mp_main__":
    # The constructor is the boundary preceding executor and TPU initialization.
    EngineCoreProc.__init__ = _engine_init


def _run_spawn_probes():
    patch_registry.apply("platform_activation")
    patch_registry._applied.add(_PARENT_MARKER)
    context = multiprocessing.get_context("spawn")
    processes = []
    connections = []
    try:
        for stage, target in (("engine_core", EngineCoreProc.run_engine_core),
                              ("worker_init", _worker_entry)):
            parent, child = context.Pipe(duplex=False)
            kwargs = {}
            if stage == "engine_core":
                kwargs["vllm_config"] = SimpleNamespace(
                    parallel_config=ParallelConfig(numa_bind=False))
            process = context.Process(target=target,
                                      args=(child, ),
                                      kwargs=kwargs)
            process.start()
            child.close()
            processes.append(process)
            connections.append((stage, parent))
        for process, (stage, parent) in zip(processes, connections):
            assert parent.poll(60), f"{stage} did not reach its entrypoint"
            actual_stage, pid = parent.recv()
            assert actual_stage == stage and pid != os.getpid()
            process.join(10)
            assert process.exitcode == 0, (stage, process.exitcode)
            print(f"PASS: spawned {stage} applied real patches", flush=True)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for _, connection in connections:
            connection.close()


if __name__ == "__main__":
    _run_spawn_probes()
