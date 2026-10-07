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
"""100% full-module line and branch coverage tests for auxiliary bench tools.

Targets:
- scripts/dsv41_bench/hbm_ledger/sitecustomize.py
- scripts/dsv41_bench/xplane_summary.py
- scripts/dsv41_bench/bench_mhc.py
- scripts/dsv41_bench/needle_32k.py
- scripts/dsv41_bench/bench_kv_compressor.py
- scripts/dsv41_bench/bench_harness_selftest.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import random
import re
import runpy
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

BENCH_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import bench_harness_selftest  # noqa: E402
import bench_kv_compressor  # noqa: E402
import bench_mhc  # noqa: E402
import common  # noqa: E402
import needle_32k  # noqa: E402

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc as v2_mhc  # noqa: E402
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops  # noqa: E402

pytestmark = pytest.mark.cpu_test


# ==============================================================================
# 1. Tests for scripts/dsv41_bench/hbm_ledger/sitecustomize.py
# ==============================================================================


class TestHbmLedgerSitecustomize:
    """Comprehensive line and branch tests for hbm_ledger/sitecustomize.py."""

    @pytest.fixture
    def hbm_sitecustomize(self):
        """Loads a fresh copy of hbm_ledger/sitecustomize.py."""
        path = BENCH_DIR / "hbm_ledger" / "sitecustomize.py"
        spec = importlib.util.spec_from_file_location("test_hbm_sitecustomize", path)
        mod = importlib.util.module_from_spec(spec)
        return mod, path

    def test_chain_next_sitecustomize_no_next(
        self, monkeypatch: pytest.MonkeyPatch, hbm_sitecustomize
    ) -> None:
        """Test _chain_next_sitecustomize when no other sitecustomize is found."""
        _, path = hbm_sitecustomize
        monkeypatch.setattr(
            importlib.machinery.PathFinder, "find_spec", lambda name, path=None: None
        )

        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "0"}):
            orig_meta = list(sys.meta_path)
            try:
                spec = importlib.util.spec_from_file_location("test_mod_no_next", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
            finally:
                sys.meta_path[:] = orig_meta

    def test_chain_next_sitecustomize_with_next(
        self, monkeypatch: pytest.MonkeyPatch, hbm_sitecustomize
    ) -> None:
        """Test _chain_next_sitecustomize loads next module when found."""
        _, path = hbm_sitecustomize
        fake_loader = Mock()
        fake_spec = importlib.machinery.ModuleSpec("sitecustomize", fake_loader)
        monkeypatch.setattr(
            importlib.machinery.PathFinder,
            "find_spec",
            lambda name, path=None: fake_spec if name == "sitecustomize" else None,
        )

        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "0"}):
            orig_meta = list(sys.meta_path)
            try:
                spec = importlib.util.spec_from_file_location(
                    "test_mod_with_next", path
                )
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                fake_loader.exec_module.assert_called_once()
            finally:
                sys.meta_path[:] = orig_meta

    def test_hbm_ledger_category_matching(self, hbm_sitecustomize) -> None:
        """Test _category matches all keywords and falls back to other."""
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
                spec = importlib.util.spec_from_file_location("test_hbm_cat", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                cat_fn = m._category

                assert cat_fn("model.engram_layer.weight") == "engram"
                assert cat_fn("model.layers.0.experts.0.w1") == "routed_experts"
                assert cat_fn("model.layers.0.shared_expert.w1") == "shared_experts"
                assert cat_fn("model.embed_tokens.weight") == "embed"
                assert cat_fn("model.lm_head.weight") == "lm_head"
                assert cat_fn("model.layers.0.indexer.weights") == "indexer"
                assert cat_fn("model.layers.0.compressor.proj") == "compressor"
                assert cat_fn("model.layers.0.attn.q_proj") == "attn"
                assert cat_fn("model.layers.0.mhc.mix") == "mhc"
                assert cat_fn("model.layers.0.hc_post.mix") == "mhc"
                assert cat_fn("model.layers.0.gate.weight") == "router"
                assert cat_fn("model.layers.0.norm.weight") == "norm"
                assert cat_fn("model.something_unrelated") == "other"
        finally:
            sys.meta_path[:] = orig_meta

    def test_hbm_ledger_mem_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hbm_sitecustomize,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test _mem_line under normal and exception handling paths."""
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
                spec = importlib.util.spec_from_file_location("test_hbm_mem", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                mem_fn = m._mem_line

                # 1. Normal path
                worker = SimpleNamespace(rank=0, devices=[torch.device("cpu")])
                monkeypatch.setattr(
                    torch.accelerator,
                    "get_memory_info",
                    lambda dev: (1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024),
                )
                monkeypatch.setattr(
                    torch.accelerator,
                    "memory_stats",
                    lambda dev: {
                        "peak_allocated": 2 * 1024 * 1024 * 1024,
                        "other_stat": 0,
                    },
                )
                mem_fn(worker, "stage_test")
                out = capsys.readouterr().out
                assert "[HBM_LEDGER] rank=0 stage=stage_test" in out
                assert "used=3.000GiB limit=4.000GiB" in out
                assert "peak_allocated=2.000GiB" in out

                # 2. Exception in get_memory_info and memory_stats
                worker_no_dev = SimpleNamespace()  # no devices, no rank
                monkeypatch.setattr(
                    torch.accelerator,
                    "get_memory_info",
                    Mock(side_effect=RuntimeError("no accelerator")),
                )
                monkeypatch.setattr(
                    torch.accelerator,
                    "memory_stats",
                    Mock(side_effect=ValueError("stats unavailable")),
                )
                mem_fn(worker_no_dev, "stage_err")
                out2 = capsys.readouterr().out
                assert "[HBM_LEDGER] rank=? stage=stage_err" in out2
                assert "get_memory_info failed: RuntimeError('no accelerator')" in out2
                assert "memory_stats n/a (ValueError)" in out2
        finally:
            sys.meta_path[:] = orig_meta

    def test_hbm_ledger_param_ledger(
        self, hbm_sitecustomize, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test _param_ledger with missing model, duplicate storage,
        and device suffixes.
        """
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
                spec = importlib.util.spec_from_file_location("test_hbm_param", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                param_fn = m._param_ledger

                # 1. Missing model
                worker_empty = SimpleNamespace(model_runner=None, rank=0)
                param_fn(worker_empty)
                assert "model_runner.model missing" in capsys.readouterr().out

                # 2. Normal model with TPU and CPU parameters
                t1_mock = Mock()
                t1_mock.untyped_storage.return_value.data_ptr.return_value = 12345
                t1_mock.untyped_storage.return_value.nbytes.return_value = 4096
                t1_mock.device.type = "tpu"
                t1_mock.numel.return_value = 1024
                t1_mock.element_size.return_value = 4

                # Duplicate key
                t1_dup = Mock()
                t1_dup.untyped_storage.return_value.data_ptr.return_value = 12345
                t1_dup.untyped_storage.return_value.nbytes.return_value = 4096

                # CPU param
                t2_mock = Mock()
                t2_mock.untyped_storage.return_value.data_ptr.return_value = 67890
                t2_mock.untyped_storage.return_value.nbytes.return_value = 4096
                t2_mock.device.type = "cpu"
                t2_mock.numel.return_value = 2048
                t2_mock.element_size.return_value = 2

                # Param with untyped_storage raising exception
                t3_mock = Mock()
                t3_mock.untyped_storage.side_effect = RuntimeError("no storage")
                t3_mock.device.type = "tpu"
                t3_mock.numel.return_value = 512
                t3_mock.element_size.return_value = 2

                model = SimpleNamespace(
                    named_parameters=lambda remove_duplicate=False: [
                        ("embed.weight", t1_mock),
                        ("embed.view", t1_dup),
                        ("engram.table", t2_mock),
                    ],
                    named_buffers=lambda remove_duplicate=False: [
                        ("norm.running_mean", t3_mock),
                    ],
                )
                worker = SimpleNamespace(
                    model_runner=SimpleNamespace(model=model), rank=1
                )
                param_fn(worker)
                out = capsys.readouterr().out
                assert "[HBM_LEDGER] rank=1 stage=param_ledger" in out
                assert "param:embed" in out
                assert "param:engram@cpu" in out
                assert "buffer:norm" in out
        finally:
            sys.meta_path[:] = orig_meta

    def test_hbm_ledger_patch_and_wrappers(
        self, hbm_sitecustomize, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test _patch wraps TPUWorker methods and executes probes."""
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(
                os.environ,
                {
                    "DSV41_HBM_LEDGER": "1",
                    "DSV41_HBM_LEDGER_EVERY": "2",
                    "DSV41_HBM_LEDGER_MAX_STEPS": "4",
                    "DSV41_HBM_LEDGER_ALL_RANKS": "0",
                },
            ):
                spec = importlib.util.spec_from_file_location("test_hbm_patch", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)

                class FakeTPUWorker:
                    def __init__(self, rank=0):
                        self.rank = rank
                        self.model_runner = SimpleNamespace(
                            model=SimpleNamespace(
                                named_parameters=lambda remove_duplicate=False: [],
                                named_buffers=lambda remove_duplicate=False: [],
                            )
                        )
                        self.devices = [torch.device("cpu")]

                    def load_model(self):
                        return "loaded"

                    def determine_available_memory(self):
                        return 8 * (1 << 30)

                    def initialize_from_config(self):
                        return "init"

                    def compile_or_warm_up_model(self):
                        return "warm"

                    def execute_model(self):
                        return "exec"

                fake_mod = SimpleNamespace(TPUWorker=FakeTPUWorker)
                m._patch(fake_mod)

                # Instantiate patched worker rank 0
                w0 = FakeTPUWorker(rank=0)
                assert w0.load_model() == "loaded"
                assert w0.determine_available_memory() == 8 * (1 << 30)
                assert w0.initialize_from_config() == "init"
                assert w0.compile_or_warm_up_model() == "warm"

                # Execute 5 times: step 2 and 4 trigger mem_line, step 5 exceeds max
                for _ in range(5):
                    assert w0.execute_model() == "exec"

                out = capsys.readouterr().out
                assert "after_load_model" in out
                assert "after_determine_avail(kv=8.000GiB)" in out
                assert "after_kv_alloc" in out
                assert "after_warmup" in out
                assert "execute_model#2" in out
                assert "execute_model#4" in out
                assert "execute_model#5" not in out

                # Test rank != 0 when all_ranks is 0: execute_model is skipped
                w1 = FakeTPUWorker(rank=1)
                w1.execute_model()
                out1 = capsys.readouterr().out
                assert "execute_model" not in out1
        finally:
            sys.meta_path[:] = orig_meta

    def test_hbm_ledger_patch_exception_handling(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hbm_sitecustomize,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test exception in after callback is caught and logged."""
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
                spec = importlib.util.spec_from_file_location("test_hbm_exc", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)

                class FaultyWorker:
                    def load_model(self):
                        return "done"

                    def determine_available_memory(self):
                        pass

                    def initialize_from_config(self):
                        pass

                    def compile_or_warm_up_model(self):
                        pass

                    def execute_model(self):
                        pass

                fake_mod = SimpleNamespace(TPUWorker=FaultyWorker)
                m._patch(fake_mod)

                # Force _mem_line to raise inside load_model probe
                monkeypatch.setattr(
                    m,
                    "_mem_line",
                    Mock(side_effect=RuntimeError("probe exploded")),
                )
                fw = FaultyWorker()
                assert fw.load_model() == "done"
                out = capsys.readouterr().out
                assert (
                    "[HBM_LEDGER] load_model probe failed: "
                    "RuntimeError('probe exploded')" in out
                )
        finally:
            sys.meta_path[:] = orig_meta

    def test_hbm_ledger_finder_and_loader(
        self, monkeypatch: pytest.MonkeyPatch, hbm_sitecustomize
    ) -> None:
        """Test MetaPathFinder and Loader hooks."""
        _, path = hbm_sitecustomize
        orig_meta = list(sys.meta_path)
        try:
            with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
                spec = importlib.util.spec_from_file_location("test_hbm_finder", path)
                assert spec is not None
                assert spec.loader is not None
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)

                finder = sys.meta_path[0]
                assert isinstance(finder, m._Finder)

                # 1. Non-target spec returns None
                assert finder.find_spec("some_other_module", []) is None

                # 2. Target spec where find_spec returns None or loader is None
                monkeypatch.setattr(importlib.util, "find_spec", lambda fullname: None)
                assert finder.find_spec("vllm_torchtpu.worker.tpu_worker", []) is None

                spec_no_loader = SimpleNamespace(loader=None)
                monkeypatch.setattr(
                    importlib.util, "find_spec", lambda fullname: spec_no_loader
                )
                assert (
                    finder.find_spec("vllm_torchtpu.worker.tpu_worker", [])
                    is spec_no_loader
                )

                # 3. Target spec with valid loader
                fake_inner_loader = Mock()
                fake_spec = SimpleNamespace(loader=fake_inner_loader)
                monkeypatch.setattr(
                    importlib.util, "find_spec", lambda fullname: fake_spec
                )
                wrapped_spec = finder.find_spec("vllm_torchtpu.worker.tpu_worker", [])
                assert wrapped_spec is not None
                assert isinstance(wrapped_spec.loader, m._Loader)

                # Loader methods
                loader = wrapped_spec.loader
                loader.create_module(wrapped_spec)
                fake_inner_loader.create_module.assert_called_once_with(wrapped_spec)

                class DummyWorker:
                    def load_model(self):
                        pass

                    def determine_available_memory(self):
                        pass

                    def initialize_from_config(self):
                        pass

                    def compile_or_warm_up_model(self):
                        pass

                    def execute_model(self):
                        pass

                fake_mod = SimpleNamespace(TPUWorker=DummyWorker)
                loader.exec_module(fake_mod)
                fake_inner_loader.exec_module.assert_called_once_with(fake_mod)
        finally:
            sys.meta_path[:] = orig_meta


# ==============================================================================
# 2. Tests for scripts/dsv41_bench/xplane_summary.py
# ==============================================================================


class TestXplaneSummary:
    """Comprehensive line and branch tests for xplane_summary.py."""

    def test_xplane_summary_full_flow(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test xplane_summary CLI parsing, planes/lines/modules/ops summaries."""
        xplane_path = BENCH_DIR / "xplane_summary.py"

        class MockEvent:
            def __init__(self, name, start_ns, duration_ns):
                self.name = name
                self.start_ns = start_ns
                self.duration_ns = duration_ns

        # Modules: 3 events, two big (>5ms) -> test gaps
        mod_events = [
            MockEvent("v41_encoder", 1_000_000_000, 10_000_000),  # 10ms (big)
            MockEvent(
                "v41_encoder", 1_020_000_000, 8_000_000
            ),  # 8ms (big) -> gap = 10ms
            MockEvent("short_mod", 1_030_000_000, 1_000_000),  # 1ms
        ]
        # Ops: 2 ops
        op_events = [
            MockEvent("custom_kernel.1 = dot()", 0, 5_000_000),
            MockEvent("all_reduce.2 = all_reduce()", 0, 3_000_000),
        ]

        mock_plane = SimpleNamespace(
            name="/device:TPU:0",
            lines=[
                SimpleNamespace(name="XLA Modules", events=mod_events),
                SimpleNamespace(name="XLA Ops", events=op_events),
            ],
        )
        mock_pd = SimpleNamespace(planes=[mock_plane])

        mock_profile_data_cls = Mock()
        mock_profile_data_cls.from_file.return_value = mock_pd

        monkeypatch.setitem(
            sys.modules,
            "jax._src.profiler",
            SimpleNamespace(ProfileData=mock_profile_data_cls),
        )

        test_argv = ["xplane_summary.py", "/tmp/fake.xplane.pb", "/device:TPU:0"]
        with patch.object(sys, "argv", test_argv):
            runpy.run_path(str(xplane_path), run_name="__main__")

        out = capsys.readouterr().out
        assert "planes: ['/device:TPU:0']" in out
        assert "lines: ['XLA Modules', 'XLA Ops']" in out
        assert "modules: n=3 span=31.0ms busy=19.0ms" in out
        assert "v41_encoder" in out
        assert "big-module (>5ms) n=2 gap med=10.00ms mean=10.00ms max=10.00ms" in out
        assert "ops total=8.0ms" in out
        assert "custom_kernel" in out
        assert "all_reduce" in out

    def test_xplane_summary_default_plane_and_no_ops_or_gaps(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test xplane_summary with default plane, single big module, and no ops."""
        xplane_path = BENCH_DIR / "xplane_summary.py"

        class MockEvent:
            def __init__(self, name, start_ns, duration_ns):
                self.name = name
                self.start_ns = start_ns
                self.duration_ns = duration_ns

        # Only 1 big module -> len(big) == 1 -> gaps is empty (branch 39->46)
        mod_events = [
            MockEvent("v41_encoder", 1_000_000_000, 10_000_000),
        ]

        mock_plane = SimpleNamespace(
            name="/device:TPU:0",
            lines=[
                SimpleNamespace(name="XLA Modules", events=mod_events),
            ],
        )
        mock_pd = SimpleNamespace(planes=[mock_plane])

        mock_profile_data_cls = Mock()
        mock_profile_data_cls.from_file.return_value = mock_pd

        monkeypatch.setitem(
            sys.modules,
            "jax._src.profiler",
            SimpleNamespace(ProfileData=mock_profile_data_cls),
        )

        test_argv = ["xplane_summary.py", "/tmp/fake.xplane.pb"]
        with patch.object(sys, "argv", test_argv):
            runpy.run_path(str(xplane_path), run_name="__main__")

        out = capsys.readouterr().out
        assert "planes: ['/device:TPU:0']" in out
        assert "modules: n=1" in out
        assert "gap med=" not in out
        assert "ops total=" not in out

    def test_xplane_summary_no_modules_only_ops(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test xplane_summary when 'XLA Modules' is absent (covers branch 20->46)."""
        xplane_path = BENCH_DIR / "xplane_summary.py"

        class MockEvent:
            def __init__(self, name, start_ns, duration_ns):
                self.name = name
                self.start_ns = start_ns
                self.duration_ns = duration_ns

        op_events = [
            MockEvent("dot.1", 0, 2_000_000),
        ]

        mock_plane = SimpleNamespace(
            name="/device:TPU:0",
            lines=[
                SimpleNamespace(name="XLA Ops", events=op_events),
            ],
        )
        mock_pd = SimpleNamespace(planes=[mock_plane])
        mock_profile_data_cls = Mock()
        mock_profile_data_cls.from_file.return_value = mock_pd

        monkeypatch.setitem(
            sys.modules,
            "jax._src.profiler",
            SimpleNamespace(ProfileData=mock_profile_data_cls),
        )

        test_argv = ["xplane_summary.py", "/tmp/fake.xplane.pb"]
        with patch.object(sys, "argv", test_argv):
            runpy.run_path(str(xplane_path), run_name="__main__")

        out = capsys.readouterr().out
        assert "modules:" not in out
        assert "ops total=2.0ms" in out


# ==============================================================================
# 3. Tests for scripts/dsv41_bench/bench_mhc.py
# ==============================================================================


class TestBenchMhc:
    """Comprehensive line and branch tests for bench_mhc.py."""

    def test_bench_mhc_full_inputs_cpu(self) -> None:
        """Verify _full_inputs constructs all required tensor keys and shapes."""
        geo = common.Dsv41Geometry()
        inputs = bench_mhc._full_inputs(torch, tokens=128, geo=geo, seed=42)
        assert inputs["x"].shape == (128, geo.hidden_size)
        assert inputs["residual"].shape == (128, geo.hc_mult * geo.hidden_size)
        assert inputs["post_mix"].shape == (128, geo.hc_mult, 1)
        assert inputs["comb"].shape == (128, geo.hc_mult, geo.hc_mult)
        assert inputs["pre_mix"].shape == (128, geo.hc_mult)
        assert inputs["hc_scale"].shape == (3,)
        assert inputs["hc_base"].shape == (geo.hc_mult * (geo.hc_mult + 2),)
        assert inputs["norm_weight"].shape == (geo.hidden_size,)

    def test_bench_mhc_slice_rows_and_conversions(self) -> None:
        """Verify _slice_rows, _to_tpu, and _cpu helper dictionaries."""
        inputs = {
            "x": torch.zeros(10, 4),
            "residual": torch.zeros(10, 16),
            "post_mix": torch.zeros(10, 4, 1),
            "comb": torch.zeros(10, 4, 4),
            "pre_mix": torch.zeros(10, 4),
            "mixes": torch.zeros(10, 24),
            "sqrsum": torch.zeros(10, 1),
            "hc_scale": torch.zeros(3),  # Not row-sharded
        }
        sliced = bench_mhc._slice_rows(inputs, 2, 6)
        assert sliced["x"].shape[0] == 4
        assert sliced["residual"].shape[0] == 4
        assert sliced["hc_scale"].shape[0] == 3

        # _to_tpu / _cpu
        t = torch.zeros(2)
        with patch.object(t, "to", return_value=t) as mock_to:
            bench_mhc._to_tpu({"t": t})
            mock_to.assert_called_once_with("tpu")

        with patch.object(t, "cpu", return_value=t) as mock_cpu:
            bench_mhc._cpu({"t": t})
            mock_cpu.assert_called_once()

    def test_bench_mhc_maybe_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify _maybe_compile disabled vs enabled."""

        def fn(x):
            return x + 1

        assert bench_mhc._maybe_compile(fn, False) is fn

        mock_compile = Mock(return_value="compiled_fn")
        monkeypatch.setattr(torch, "compile", mock_compile)
        res = bench_mhc._maybe_compile(fn, True)
        assert res == "compiled_fn"
        mock_compile.assert_called_once()

    def test_bench_mhc_make_fn_all_subops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify _make_fn generates valid callable for each subop."""
        geo = common.Dsv41Geometry()
        inputs = bench_mhc._full_inputs(torch, tokens=4, geo=geo, seed=0)

        # Mock mhc functions directly on module to ensure deterministic CPU returns
        dummy_residual = torch.zeros(4, geo.hc_mult, geo.hidden_size)
        dummy_post = torch.zeros(4, geo.hc_mult, 1)
        dummy_comb = torch.zeros(4, geo.hc_mult, geo.hc_mult)
        dummy_val = torch.zeros(4, geo.hidden_size)
        dummy_next = torch.zeros(4, geo.hc_mult)

        monkeypatch.setattr(
            v2_mhc,
            "shifted_post_pre",
            lambda *args, **kwargs: (
                dummy_residual,
                dummy_post,
                dummy_comb,
                dummy_val,
                dummy_next,
                None,
            ),
        )
        monkeypatch.setattr(
            v2_mhc,
            "post",
            lambda *args, **kwargs: dummy_residual,
        )
        monkeypatch.setattr(
            v2_mhc,
            "delayed_pre",
            lambda *args, **kwargs: (
                dummy_post,
                dummy_comb,
                dummy_val,
                dummy_next,
            ),
        )
        monkeypatch.setattr(
            v2_ops,
            "rms_norm",
            lambda *args, **kwargs: dummy_val,
        )

        # 1. 'all'
        fn_all = bench_mhc._make_fn("all", geo)
        out_all = fn_all(inputs)
        assert "residual" in out_all and "post" in out_all and "value" in out_all

        # 2. 'mhc_post_only'
        fn_post = bench_mhc._make_fn("mhc_post_only", geo)
        out_post = fn_post(inputs)
        assert "residual" in out_post

        # 3. 'delayed_pre_only'
        fn_pre = bench_mhc._make_fn("delayed_pre_only", geo)
        out_pre = fn_pre(inputs)
        assert "post" in out_pre and "value" in out_pre

        # 4. 'rms_norm_only'
        fn_norm = bench_mhc._make_fn("rms_norm_only", geo)
        out_norm = fn_norm(inputs)
        assert "value" in out_norm

        # 5. 'sinkhorn_only' (mock _gates_op)
        mock_op = Mock(
            return_value=(
                torch.zeros(4, 4),
                torch.zeros(4, 4),
                torch.zeros(4, 4, 4),
            )
        )
        monkeypatch.setattr(bench_mhc, "_gates_op", lambda g: mock_op)
        fn_sink = bench_mhc._make_fn("sinkhorn_only", geo)
        out_sink = fn_sink(inputs)
        assert "pre" in out_sink and "post" in out_sink

        # 6. Invalid subop
        with pytest.raises(ValueError, match="unknown_subop"):
            bench_mhc._make_fn("unknown_subop", geo)

    def test_bench_mhc_gates_op_construction_and_call(self) -> None:
        """Verify _gates_op constructs jax_op and registers fake."""
        geo = common.Dsv41Geometry()
        with common.model_wrapper_context():
            op = bench_mhc._gates_op(geo)
            assert op is not None

            # Directly invoke the inner gates function to cover line 111
            gates_fn = getattr(getattr(op, "_init_fn", None), "jit_fn", None)
            inner_fn = getattr(gates_fn, "_fun", None)
            if inner_fn is not None:
                with patch.object(
                    bench_mhc,
                    "_gates_jax",
                    return_value=("pre", "post", "comb"),
                ) as mock_gj:
                    res = inner_fn("mixes", "sqrsum", "scale", "base")
                    assert res == ("pre", "post", "comb")
                    mock_gj.assert_called_once()

            # Test fake implementation registered on op
            mixes = torch.zeros(8, 24)
            sqrsum = torch.zeros(8, 1)
            scale = torch.zeros(3)
            base = torch.zeros(24)

            # Call with meta tensors to exercise fake
            meta_mixes = mixes.to(device="meta")
            meta_sqrsum = sqrsum.to(device="meta")
            meta_scale = scale.to(device="meta")
            meta_base = base.to(device="meta")
            pre, post, comb = op(meta_mixes, meta_sqrsum, meta_scale, meta_base)
            assert pre.shape == (8, geo.hc_mult)
            assert post.shape == (8, geo.hc_mult)
            assert comb.shape == (8, geo.hc_mult, geo.hc_mult)

    def test_bench_mhc_gates_jax(self) -> None:
        """Verify _gates_jax delegates to pre_gates with correct constants."""
        pre_gates_mock = Mock(return_value="gates_result")
        res = bench_mhc._gates_jax(
            pre_gates_mock,
            mult=4,
            dim=5120,
            repeats=1,
            mixes="mixes",
            sqrsum="sqrsum",
            scale="scale",
            base="base",
        )
        assert res == "gates_result"
        pre_gates_mock.assert_called_once_with(
            "mixes",
            "sqrsum",
            4,
            5120,
            "scale",
            "base",
            bench_mhc.RMS_EPS,
            bench_mhc.HC_EPS,
            bench_mhc.HC_EPS,
            bench_mhc.HC_POST_ALPHA,
            1,
        )

    def test_bench_mhc_roofline(self) -> None:
        """Verify _roofline estimates across modes and subops."""
        geo = common.Dsv41Geometry()

        # Collective only
        est_sp = bench_mhc._roofline(
            "collective_only", tokens=256, geo=geo, tp=8, mode="sp"
        )
        assert est_sp.name == "sp collective"
        assert est_sp.comm_s > 0

        est_tp = bench_mhc._roofline(
            "collective_only", tokens=2048, geo=geo, tp=8, mode="tp"
        )
        assert est_tp.name == "tp collective"
        assert est_tp.comm_s > 0

        # All
        est_all = bench_mhc._roofline("all", tokens=2048, geo=geo, tp=8, mode="tp")
        assert est_all.name == "all"
        assert est_all.flops > 0
        assert est_all.hbm_bytes > 0

        # Specific subops
        for subop in (
            "mhc_post_only",
            "delayed_pre_only",
            "rms_norm_only",
            "sinkhorn_only",
        ):
            est = bench_mhc._roofline(subop, tokens=2048, geo=geo, tp=8, mode="tp")
            assert est.name in ("post", "delayed_pre", "rms_norm", "sinkhorn")
            assert est.hbm_bytes > 0

    @pytest.mark.parametrize("mode", ["tp", "sp"])
    def test_bench_mhc_collective_helper(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """Verify _collective helper executes AR and RS across tp and sp modes."""
        geo = common.Dsv41Geometry()
        args = argparse.Namespace(seed=0, mode=mode, compile=False)
        ctx = SimpleNamespace(tp_size=8)

        monkeypatch.setattr(
            "vllm.distributed.get_tp_group",
            lambda: SimpleNamespace(rank_in_group=0),
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_all_reduce",
            lambda x: x.clone(),
        )
        monkeypatch.setattr(
            "vllm.distributed.tensor_model_parallel_reduce_scatter",
            lambda x, dim=0: x[: x.shape[0] // 8].clone(),
        )
        monkeypatch.setattr(common, "synchronize", lambda x: None)
        monkeypatch.setattr(torch.Tensor, "to", lambda self, dev: self)

        run = bench_mhc._collective(ctx, args, tokens_full=128, geo=geo)
        assert "step" in run
        assert "outputs" in run
        assert "diagnostics_vs_fp32_truth" in run
        if mode == "sp":
            assert run["tokens_local"] == 128 // 8
        else:
            assert run["tokens_local"] == 128

    def test_bench_mhc_worker_execution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """Test bench_mhc.worker execution across modes, verification,
        and golden paths.
        """
        geo = common.Dsv41Geometry()
        ctx = SimpleNamespace(
            tp_size=1,
            rank=0,
            local_rank=0,
            device=torch.device("cpu"),
        )
        monkeypatch.setattr(
            "vllm.distributed.get_tp_group",
            lambda: SimpleNamespace(rank_in_group=0),
        )
        monkeypatch.setattr(common, "model_wrapper_context", MagicMock)
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.mhc.initialize_delayed_mhc",
            lambda: None,
        )
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.mhc.initialize_mhc_post",
            lambda: None,
        )
        monkeypatch.setattr(torch.Tensor, "to", lambda self, dev: self)
        monkeypatch.setattr(common, "synchronize", lambda x: None)
        monkeypatch.setattr(common, "hbm_stats", lambda: {"used": 100})

        # Make time_steps actually execute step() once to cover line 397
        def mock_time_steps(ctx, step, warmup, iters):
            step()
            return ([1.5, 1.4], {})

        monkeypatch.setattr(common, "time_steps", mock_time_steps)
        monkeypatch.setattr(
            common, "profile_steps", lambda ctx, args, step, label: None
        )

        # 1. Invalid chunk size
        invalid_args = argparse.Namespace(
            chunk_size=10,  # not multiple of 64
            mode="tp",
            subop="all",
        )
        with pytest.raises(ValueError, match="must be a multiple of"):
            bench_mhc.worker(ctx, invalid_args)

        # 2. subop == "all", mode == "tp", update_golden == False, verify == False
        tp_args = argparse.Namespace(
            chunk_size=64,
            mode="tp",
            subop="all",
            seed=1,
            compile=False,
            warmup=1,
            iters=2,
            verify=False,
            update_golden=False,
            golden_dir=tmp_path,
        )
        dummy_out = {
            "residual": torch.zeros(64, geo.hc_mult * geo.hidden_size),
            "post": torch.zeros(64, geo.hc_mult, 1),
            "comb": torch.zeros(64, geo.hc_mult, geo.hc_mult),
            "value": torch.zeros(64, geo.hidden_size),
            "next_pre": torch.zeros(64, geo.hc_mult),
        }
        monkeypatch.setattr(
            bench_mhc, "_make_fn", lambda subop, geo: lambda inp: dummy_out
        )
        monkeypatch.setattr(
            common,
            "check_against_golden",
            lambda ctx, args, case, out, match: {"status": "ok"},
        )

        res_tp = bench_mhc.worker(ctx, tp_args)
        assert res_tp["mode"] == "tp"
        assert res_tp["golden"] == {"status": "ok"}
        assert res_tp["roofline_vs_wall"]["measured_ms"] == 1.45

        # 3. mode == "sp" with update_golden=True raises ValueError
        sp_update_args = argparse.Namespace(
            chunk_size=64,
            mode="sp",
            subop="all",
            seed=1,
            compile=False,
            warmup=1,
            iters=2,
            verify=True,
            update_golden=True,
            golden_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="record goldens with --mode tp"):
            bench_mhc.worker(ctx, sp_update_args)

        # 4. mode == "sp" with verify=True
        sp_verify_args = argparse.Namespace(
            chunk_size=64,
            mode="sp",
            subop="all",
            seed=1,
            compile=False,
            warmup=1,
            iters=2,
            verify=True,
            update_golden=False,
            golden_dir=tmp_path,
        )
        monkeypatch.setattr(
            common,
            "load_golden",
            lambda path: (dummy_out, {"meta": "data"}),
        )
        res_sp = bench_mhc.worker(ctx, sp_verify_args)
        assert res_sp["mode"] == "sp"
        assert "report" in res_sp["golden"]

        # 5. mode == "sp" with verify=False (covers line 436: skipped golden)
        sp_no_verify_args = argparse.Namespace(
            chunk_size=64,
            mode="sp",
            subop="all",
            seed=1,
            compile=False,
            warmup=1,
            iters=2,
            verify=False,
            update_golden=False,
            golden_dir=tmp_path,
        )
        res_sp_skipped = bench_mhc.worker(ctx, sp_no_verify_args)
        assert res_sp_skipped["golden"] == {"golden": "skipped"}

        # 6. subop == "collective_only" (covers lines 375-378)
        coll_args = argparse.Namespace(
            chunk_size=64,
            mode="sp",
            subop="collective_only",
            seed=1,
            compile=False,
            warmup=1,
            iters=2,
            verify=True,
            update_golden=False,
            golden_dir=tmp_path,
        )
        fake_coll_run = {
            "step": lambda: None,
            "outputs": {"out": torch.zeros(64)},
            "reference": {"out": torch.zeros(64)},
            "tokens_local": 64,
            "diagnostics_vs_fp32_truth": {"sp": {"ok": True}},
        }
        monkeypatch.setattr(
            bench_mhc,
            "_collective",
            lambda ctx, args, tokens_full, geo: fake_coll_run,
        )
        monkeypatch.setattr(
            common,
            "load_golden",
            lambda path: ({"out": torch.zeros(64)}, {"meta": "coll_meta"}),
        )
        res_coll = bench_mhc.worker(ctx, coll_args)
        assert res_coll["diagnostics_vs_fp32_truth"] == {"sp": {"ok": True}}

    def test_bench_mhc_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify bench_mhc.main configures parser and calls common.run_main."""
        run_main_mock = Mock()
        monkeypatch.setattr(common, "run_main", run_main_mock)
        monkeypatch.setattr(
            sys, "argv", ["bench_mhc.py", "--mode", "sp", "--no-compile"]
        )
        bench_mhc.main()
        run_main_mock.assert_called_once()
        assert run_main_mock.call_args[0][0] == "mhc"

        # Also test __main__ execution block via runpy
        with patch.object(sys, "argv", ["bench_mhc.py", "--mode", "tp"]):
            runpy.run_path(str(BENCH_DIR / "bench_mhc.py"), run_name="__main__")


# ==============================================================================
# 4. Tests for scripts/dsv41_bench/needle_32k.py
# ==============================================================================


class TestNeedle32k:
    """Comprehensive line and branch tests for needle_32k.py."""

    def test_post_success(self) -> None:
        mock_resp_data = {"count": 42}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_resp_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            res = needle_32k._post(
                "http://127.0.0.1:8012/test", {"key": "value"}, timeout=5.0
            )
            assert res == mock_resp_data
            mock_urlopen.assert_called_once()
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://127.0.0.1:8012/test"
            assert req.data == b'{"key": "value"}'
            assert req.headers["Content-type"] == "application/json"

    def test_count_tokens(self) -> None:
        with patch.object(
            needle_32k, "_post", return_value={"count": 128}
        ) as mock_post:
            count = needle_32k._count_tokens(
                "http://127.0.0.1:8012", "model-x", "text", timeout=10.0
            )
            assert count == 128
            mock_post.assert_called_once_with(
                "http://127.0.0.1:8012/tokenize",
                {"model": "model-x", "prompt": "text"},
                10.0,
            )

    def test_filler(self) -> None:
        rng = random.Random(42)
        sentences = needle_32k._filler(rng, 5)
        assert len(sentences) == 5
        assert all(s.endswith(".") for s in sentences)

    def test_build_prompt_boundaries_and_validation(self) -> None:
        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        prompt = needle_32k.build_prompt(sentences, 0.0, "NEEDLE", "QUESTION")
        assert prompt.startswith("Below is a long document.")
        assert "NEEDLE" in prompt
        assert prompt.endswith("QUESTION")

        prompt_end = needle_32k.build_prompt(sentences, 1.0, "NEEDLE", "QUESTION")
        assert "NEEDLE" in prompt_end

        with pytest.raises(ValueError, match=r"depth must be in \[0, 1\]"):
            needle_32k.build_prompt(sentences, -0.1, "NEEDLE", "QUESTION")
        with pytest.raises(ValueError, match=r"depth must be in \[0, 1\]"):
            needle_32k.build_prompt(sentences, 1.1, "NEEDLE", "QUESTION")

    def test_main_all_pass_with_out(self, tmp_path: pathlib.Path) -> None:
        out_file = tmp_path / "needle_out.json"

        def fake_post(url: str, payload: dict, timeout: float) -> dict:
            if "/tokenize" in url:
                return {"count": 400}
            if "/v1/chat/completions" in url:
                prompt = payload["messages"][0]["content"]
                match = re.search(
                    r"The special magic number for \w+ is (\d+)\.", prompt
                )
                key = match.group(1) if match else "1234567"
                return {
                    "choices": [{"message": {"content": f"The number is {key}."}}],
                    "usage": {"prompt_tokens": 100},
                }
            return {}

        with (
            patch.object(needle_32k, "_post", side_effect=fake_post),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.5",
                    f"--out={out_file}",
                    "--target-tokens=100",
                ],
            ),
        ):
            rc = needle_32k.main()
            assert rc == 0
            assert out_file.is_file()
            data = json.loads(out_file.read_text(encoding="utf-8"))
            assert data["all_pass"] is True
            assert data["score"] == "1/1"
            assert len(data["trials"]) == 1
            assert data["trials"][0]["pass"] is True

    def test_main_failure_and_content_none(self) -> None:
        def fake_post(url: str, payload: dict, timeout: float) -> dict:
            if "/tokenize" in url:
                return {"count": 400}
            return {
                "choices": [{"message": {"content": None}}],
                "usage": {"prompt_tokens": 50},
            }

        with (
            patch.object(needle_32k, "_post", side_effect=fake_post),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.2",
                    "--target-tokens=50",
                ],
            ),
        ):
            rc = needle_32k.main()
            assert rc == 1

    def test_needle_entrypoint_via_runpy(self) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = (
            b'{"count": 400, "choices": [{"message": {"content": "1234567"}}], '
            b'"usage": {"prompt_tokens": 100}}'
        )
        mock_resp.__enter__.return_value = mock_resp
        with (
            patch("urllib.request.urlopen", return_value=mock_resp),
            patch.object(
                sys,
                "argv",
                [
                    "needle_32k.py",
                    "--rounds=1",
                    "--depths=0.5",
                    "--target-tokens=100",
                ],
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(str(BENCH_DIR / "needle_32k.py"), run_name="__main__")
            assert exc_info.value.code in (0, 1)


# ==============================================================================
# 5. Tests for scripts/dsv41_bench/bench_kv_compressor.py
# ==============================================================================


class TestBenchKVCompressor:
    """Comprehensive line and branch tests for bench_kv_compressor.py."""

    def test_sys_path_insertion(self) -> None:
        bench_dir_str = str(BENCH_DIR)
        assert bench_dir_str in sys.path

    def test_resolve_device(self) -> None:
        assert bench_kv_compressor._resolve_device("cpu") == torch.device("cpu")
        assert bench_kv_compressor._resolve_device("meta") == torch.device("meta")

        # auto without TPU -> returns cpu
        with patch.dict(sys.modules, {"torch_tpu": None}):
            assert bench_kv_compressor._resolve_device("auto") == torch.device("cpu")

        # tpu without TPU -> raises
        with (
            patch.dict(sys.modules, {"torch_tpu": None}),
            pytest.raises(ModuleNotFoundError),
        ):
            bench_kv_compressor._resolve_device("tpu")

        # auto/tpu with mock TPU working
        mock_tpu_dev = SimpleNamespace(type="tpu")
        orig_device = torch.device
        with (
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
        ):
            assert bench_kv_compressor._resolve_device("auto") == mock_tpu_dev
            assert bench_kv_compressor._resolve_device("tpu") == mock_tpu_dev

    def test_sync_if_tpu(self) -> None:
        cpu_dev = torch.device("cpu")
        bench_kv_compressor._sync_if_tpu(cpu_dev, torch.zeros(1))

        tpu_dev = SimpleNamespace(type="tpu")
        # None tensors -> no-op
        bench_kv_compressor._sync_if_tpu(tpu_dev, None)
        # Non-none tensors -> call common.synchronize
        with patch.object(common, "synchronize") as mock_sync:
            t = torch.zeros(1)
            bench_kv_compressor._sync_if_tpu(tpu_dev, t)
            mock_sync.assert_called_once_with(t)

    def test_bench_ms(self) -> None:
        calls = []

        def dummy() -> None:
            calls.append(1)

        dur = bench_kv_compressor._bench_ms(dummy, warmup=2, iters=3)
        assert len(calls) == 5
        assert dur >= 0.0

    def test_rms_norm_ref(self) -> None:
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        w = torch.ones(64, dtype=torch.bfloat16)
        out = bench_kv_compressor._rms_norm_ref(x, w, eps=1e-6)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16

    def test_run_project_only_verify_and_no_verify(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")
        # verify=True
        r1, v1 = bench_kv_compressor.run_project_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True
        assert r1["name"] == "project_only"

        # verify=False (covers branch 176->189)
        r2, v2 = bench_kv_compressor.run_project_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=False
        )
        assert v2 is None
        assert r2["name"] == "project_only"

    def test_run_save_compress_norm_only_verify_and_no_verify(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")
        # verify=True
        r1, v1 = bench_kv_compressor.run_save_compress_norm_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True
        assert r1["name"] == "save_compress_norm_only"

        # verify=False (covers branch 243->259)
        r2, v2 = bench_kv_compressor.run_save_compress_norm_only(
            dev, hw, num_tokens=16, warmup=1, iters=1, verify=False
        )
        assert v2 is None
        assert r2["name"] == "save_compress_norm_only"

    def test_run_write_fp8_only_branches(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")

        # total_slots < num_tokens assertion
        with pytest.raises(AssertionError, match="must be >= num_tokens"):
            bench_kv_compressor.run_write_fp8_only(
                dev,
                hw,
                num_tokens=64,
                num_pages=2,
                page_size=16,
                warmup=1,
                iters=1,
                verify=False,
            )

        # verify=True on CPU
        r1, v1 = bench_kv_compressor.run_write_fp8_only(
            dev,
            hw,
            num_tokens=32,
            num_pages=4,
            page_size=16,
            warmup=1,
            iters=1,
            verify=True,
        )
        assert v1 is not None
        assert v1["ok"] is True

        # verify=False on CPU (covers branch 305->321)
        r2, v2 = bench_kv_compressor.run_write_fp8_only(
            dev,
            hw,
            num_tokens=32,
            num_pages=4,
            page_size=16,
            warmup=1,
            iters=1,
            verify=False,
        )
        assert v2 is None

        # non-CPU device covers lines 278-280 (device.type != "cpu")
        meta_dev = torch.device("meta")
        with (
            patch.object(bench_kv_compressor, "initialize_cache_ops") as mock_init_ops,
            patch.object(bench_kv_compressor.PagedCacheRows, "write"),
        ):
            r3, v3 = bench_kv_compressor.run_write_fp8_only(
                meta_dev,
                hw,
                num_tokens=16,
                num_pages=2,
                page_size=16,
                warmup=1,
                iters=1,
                verify=False,
            )
            assert mock_init_ops.called
            assert v3 is None

    def test_run_engram_host_vs_hbm_branches(self) -> None:
        hw = common.Tpu7xRoofline()
        dev = torch.device("cpu")

        # verify=True on CPU
        rh1, rm1, v1 = bench_kv_compressor.run_engram_host_vs_hbm(
            dev, hw, num_tokens=16, engram_rows=128, warmup=1, iters=1, verify=True
        )
        assert v1 is not None
        assert v1["ok"] is True

        # verify=False on CPU (covers branch 428->439)
        rh2, rm2, v2 = bench_kv_compressor.run_engram_host_vs_hbm(
            dev, hw, num_tokens=16, engram_rows=128, warmup=1, iters=1, verify=False
        )
        assert v2 is None

        # non-CPU device covers lines 373-374 (emb_hbm.to(device))
        meta_dev = torch.device("meta")
        with (
            patch.object(
                bench_kv_compressor.ParallelEngramEmbedding,
                "to",
                return_value=MagicMock(),
            ) as mock_to,
            patch.object(bench_kv_compressor.ParallelEngramEmbedding, "lookup"),
            patch.object(torch.Tensor, "copy_"),
        ):
            rh3, rm3, v3 = bench_kv_compressor.run_engram_host_vs_hbm(
                meta_dev,
                hw,
                num_tokens=16,
                engram_rows=128,
                warmup=1,
                iters=1,
                verify=False,
            )
            assert mock_to.called
            assert v3 is None

    def test_main_individual_subops_and_rank_nonzero(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Run individual subops to cover 542->550, 550->558, 558->573, 573->587
        subops = [
            "project_only",
            "save_compress_norm_only",
            "write_fp8_only",
            "engram_host_vs_hbm",
        ]
        for subop in subops:
            summary = bench_kv_compressor.main(
                [
                    "--device=cpu",
                    f"--subop={subop}",
                    "--num-tokens=16",
                    "--num-pages=2",
                    "--page-size=16",
                    "--engram-rows=128",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary["subop"] == subop

        # Run with rank != 0 and --dump-llo (covers lines 536 and 596->619)
        llo_dir = tmp_path / "llo"
        with (
            patch.dict(os.environ, {"RANK": "1"}),
            patch.object(common, "_llo_env") as mock_llo,
        ):
            summary_rank1 = bench_kv_compressor.main(
                [
                    "--device=cpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    f"--dump-llo={llo_dir}",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_rank1["subop"] == "project_only"
            mock_llo.assert_called_once()

    def test_main_tpu_orchestration_and_subprocess_paths(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Case 1: device=auto, RANK not in os.environ, tpu_count > 0, json_out exists
        out_json = tmp_path / "summary_tpu.json"
        out_json.write_text(json.dumps({"from": "subprocess"}), encoding="utf-8")

        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=8,
            ),
            patch.object(common, "prepare_worker_env", return_value={"FAKE": "1"}),
            patch("subprocess.run") as mock_subproc,
        ):
            res1 = bench_kv_compressor.main(["--device=auto", f"--json-out={out_json}"])
            assert res1 == {"from": "subprocess"}
            mock_subproc.assert_called_once()

        # Case 2: device=auto, RANK not in os.environ, tpu_count > 0, no json_out
        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=8,
            ),
            patch.object(common, "prepare_worker_env", return_value={"FAKE": "1"}),
            patch("subprocess.run"),
        ):
            res2 = bench_kv_compressor.main(["--device=auto", "--subop=write_fp8_only"])
            assert res2 == {"device": "tpu", "subop": "write_fp8_only"}

        # Case 3: device=tpu, RANK not in os.environ, tpu_count == 0 -> raises
        with (
            patch(
                "torch_tpu._internal.utils.hardware.get_tpu_device_count",
                return_value=0,
            ),
            pytest.raises(RuntimeError, match="No TPU devices found."),
        ):
            bench_kv_compressor.main(["--device=tpu"])

        # Case 4: device=auto, tpu_count == 0 -> sets device = cpu (line 522)
        with patch(
            "torch_tpu._internal.utils.hardware.get_tpu_device_count",
            return_value=0,
        ):
            summary_auto = bench_kv_compressor.main(
                [
                    "--device=auto",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_auto["device"] == "cpu"

        # Case 4b: hardware.get_tpu_device_count raises exception (lines 499-500)
        with patch(
            "torch_tpu._internal.utils.hardware.get_tpu_device_count",
            side_effect=RuntimeError("simulated error"),
        ):
            summary_err = bench_kv_compressor.main(
                [
                    "--device=auto",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_err["device"] == "cpu"

        # Case 5a: worker branch with dist uninitialized initially
        mock_tpu_dev = SimpleNamespace(type="tpu")
        orig_device = torch.device
        init_state = {"init": False}

        def mock_is_initialized() -> bool:
            return init_state["init"]

        def mock_init_pg(*args: Any, **kwargs: Any) -> None:
            init_state["init"] = True

        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", side_effect=mock_is_initialized),
            patch("torch.distributed.init_process_group", side_effect=mock_init_pg),
            patch("torch.distributed.barrier") as mock_barrier,
            patch("torch.distributed.destroy_process_group") as mock_destroy,
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=(
                    {"name": "project_only", "lower_bound_ms": 0.01},
                    None,
                ),
            ),
        ):
            summary_worker = bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            assert summary_worker["subop"] == "project_only"
            assert mock_barrier.called
            assert mock_destroy.called

        # Case 5b: worker branch with dist already initialized (line 527 False branch)
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.init_process_group") as mock_init_pg_skip,
            patch("torch.distributed.barrier"),
            patch("torch.distributed.destroy_process_group"),
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=(
                    {"name": "project_only", "lower_bound_ms": 0.01},
                    None,
                ),
            ),
        ):
            bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            mock_init_pg_skip.assert_not_called()

        # Case 5c: worker branch where dist is uninitialized at exit (lines 622->625)
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.barrier") as mock_barrier_skip,
            patch(
                "torch.device",
                side_effect=lambda d: mock_tpu_dev if d == "tpu" else orig_device(d),
            ),
            patch("torch.empty", return_value=MagicMock(cpu=MagicMock())),
            patch.object(
                bench_kv_compressor,
                "run_project_only",
                return_value=(
                    {"name": "project_only", "lower_bound_ms": 0.01},
                    None,
                ),
            ),
        ):
            bench_kv_compressor.main(
                [
                    "--device=tpu",
                    "--subop=project_only",
                    "--num-tokens=16",
                    "--warmup=1",
                    "--iters=1",
                ]
            )
            mock_barrier_skip.assert_not_called()

    def test_main_json_out_write(self, tmp_path: pathlib.Path) -> None:
        out_file = tmp_path / "nested" / "summary.json"
        summary = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=project_only",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                f"--json-out={out_file}",
            ]
        )
        assert out_file.is_file()
        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert loaded["subop"] == summary["subop"]
        assert loaded["device"] == "cpu"

    def test_main_all_subops_with_and_without_verify(self) -> None:
        summary_verify = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=all",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                "--verify",
            ]
        )
        assert len(summary_verify["verifications"]) == 4
        assert len(summary_verify["rooflines"]) == 5

        summary_no_verify = bench_kv_compressor.main(
            [
                "--device=cpu",
                "--subop=all",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
                "--no-verify",
            ]
        )
        assert len(summary_no_verify["verifications"]) == 0
        assert len(summary_no_verify["rooflines"]) == 5

    def test_bench_kv_compressor_sys_path_insert(self) -> None:
        bench_dir_str = str(BENCH_DIR.resolve())
        orig_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p != bench_dir_str]
            code = compile(
                (BENCH_DIR / "bench_kv_compressor.py").read_text(encoding="utf-8"),
                str(BENCH_DIR / "bench_kv_compressor.py"),
                "exec",
            )
            globs = {
                "__name__": "bench_kv_compressor_test_mod",
                "__file__": str(BENCH_DIR / "bench_kv_compressor.py"),
            }
            exec(code, globs)
            assert bench_dir_str in sys.path
        finally:
            sys.path = orig_path

    def test_bench_kv_compressor_entrypoint_via_runpy(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "bench_kv_compressor.py",
                "--device=cpu",
                "--subop=project_only",
                "--num-tokens=16",
                "--warmup=1",
                "--iters=1",
            ],
        ):
            runpy.run_path(
                str(BENCH_DIR / "bench_kv_compressor.py"), run_name="__main__"
            )


# ==============================================================================
# 6. Tests for scripts/dsv41_bench/bench_harness_selftest.py
# ==============================================================================


class TestBenchHarnessSelftest:
    """Comprehensive line and branch tests for bench_harness_selftest.py."""

    @pytest.fixture(autouse=True)
    def _mock_tpu_to_cpu(self):
        orig_to = torch.Tensor.to

        def mock_to(self, *args, **kwargs):
            if args and args[0] == "tpu":
                return self
            if "device" in kwargs and kwargs["device"] == "tpu":
                kwargs["device"] = "cpu"
            return orig_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", mock_to):
            yield

    def test_rank_input(self) -> None:
        t = bench_harness_selftest._rank_input(torch, rank=1, tokens=8, hidden=16)
        assert t.shape == (8, 16)
        assert t.dtype == torch.bfloat16
        assert t[0, 0] == float((0 + 0 + 1) % 7)

    def test_reference_all_subops(self) -> None:
        # all_reduce
        ref_ar = bench_harness_selftest._reference(torch, "all_reduce", 0, 4, 8, 16)
        assert ref_ar.shape == (8, 16)
        assert ref_ar.dtype == torch.bfloat16

        # reduce_scatter
        ref_rs = bench_harness_selftest._reference(torch, "reduce_scatter", 1, 4, 8, 16)
        assert ref_rs.shape == (2, 16)
        assert ref_rs.dtype == torch.bfloat16

        # all_gather
        ref_ag = bench_harness_selftest._reference(torch, "all_gather", 0, 4, 8, 16)
        assert ref_ag.shape == (8, 16)

    def test_worker_all_reduce_subop(self, tmp_path: pathlib.Path) -> None:
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)
        args = argparse.Namespace(
            chunk_size=16,
            subop="all_reduce",
            warmup=1,
            iters=1,
            verify=True,
            golden_dir=tmp_path / "golden",
            update_golden=False,
            profile=None,
        )

        with (
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm.distributed.tensor_model_parallel_all_reduce",
                side_effect=lambda x: x,
            ),
            patch.object(common, "synchronize"),
            patch.object(common, "hbm_stats", return_value={"hbm": 1}),
            patch.object(common, "time_steps", return_value=([1.0], None)),
            patch.object(
                common, "check_against_golden", return_value={"golden": "skipped"}
            ),
            patch.object(common, "verify_outputs", return_value={"ok": True}),
        ):
            mock_tp.return_value.rank_in_group = 0
            res = bench_harness_selftest.worker(ctx, args)
            assert "roofline_vs_wall" in res
            assert res["roofline_vs_wall"]["x41_per_step_lower_bound_ms"] > 0

    def test_worker_reduce_scatter_and_all_gather(self, tmp_path: pathlib.Path) -> None:
        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)
        args_rs = argparse.Namespace(
            chunk_size=16,
            subop="reduce_scatter",
            warmup=1,
            iters=1,
            verify=False,
            golden_dir=tmp_path / "golden",
            update_golden=False,
            profile=None,
        )

        with (
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm.distributed.tensor_model_parallel_reduce_scatter",
                side_effect=lambda x, dim: x,
            ),
            patch.object(common, "synchronize"),
            patch.object(common, "hbm_stats", return_value={}),
            patch.object(common, "time_steps", return_value=([1.0], None)),
            patch.object(common, "profile_steps", return_value=None),
            patch.object(common, "check_against_golden", return_value={}),
        ):
            mock_tp.return_value.rank_in_group = 0
            res_rs = bench_harness_selftest.worker(ctx, args_rs)
            assert "roofline_vs_wall" in res_rs

        args_ag = argparse.Namespace(
            chunk_size=16,
            subop="all_gather",
            warmup=1,
            iters=1,
            verify=False,
            golden_dir=tmp_path / "golden",
            update_golden=False,
            profile=None,
        )

        with (
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch(
                "vllm.distributed.tensor_model_parallel_all_gather",
                side_effect=lambda x, dim: x,
            ),
            patch.object(common, "synchronize"),
            patch.object(common, "hbm_stats", return_value={}),
            patch.object(common, "time_steps", return_value=([1.0], None)),
            patch.object(common, "profile_steps", return_value=None),
            patch.object(common, "check_against_golden", return_value={}),
        ):
            mock_tp.return_value.rank_in_group = 0
            res_ag = bench_harness_selftest.worker(ctx, args_ag)
            assert "roofline_vs_wall" in res_ag

    def test_worker_chunk_size_validation(self, tmp_path: pathlib.Path) -> None:
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)
        args = argparse.Namespace(chunk_size=15, subop="all_reduce")
        with pytest.raises(ValueError, match="must divide by tp"):
            bench_harness_selftest.worker(ctx, args)

    def test_main_and_entrypoint_via_runpy(self) -> None:
        with (
            patch.object(common, "run_main") as mock_run_main,
            patch.object(
                sys, "argv", ["bench_harness_selftest.py", "--subop", "all_reduce"]
            ),
        ):
            bench_harness_selftest.main()
            mock_run_main.assert_called_once()

        script_path = str(BENCH_DIR / "bench_harness_selftest.py")
        with (
            patch.object(common, "run_main"),
            patch.object(
                sys, "argv", ["bench_harness_selftest.py", "--subop", "all_reduce"]
            ),
        ):
            res = runpy.run_path(script_path, run_name="__main__")
            assert "main" in res
