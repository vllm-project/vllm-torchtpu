# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for benchmarking tools.

Covers all lines and branches in:
- scripts/dsv41_bench/hbm_ledger/sitecustomize.py
- scripts/dsv41_bench/xplane_summary.py
- scripts/dsv41_bench/bench_mhc.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import pathlib
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_mhc  # noqa: E402
import common  # noqa: E402

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc as v2_mhc  # noqa: E402
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ops as v2_ops  # noqa: E402

# ==============================================================================
# 1. Tests for scripts/dsv41_bench/hbm_ledger/sitecustomize.py
# ==============================================================================


@pytest.fixture
def hbm_sitecustomize():
    """Loads a fresh copy of hbm_ledger/sitecustomize.py."""
    path = _SCRIPTS_DIR / "hbm_ledger" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("test_hbm_sitecustomize", path)
    mod = importlib.util.module_from_spec(spec)
    return mod, path


def test_chain_next_sitecustomize_no_next(monkeypatch, hbm_sitecustomize):
    """Test _chain_next_sitecustomize when no other sitecustomize is found."""
    mod, path = hbm_sitecustomize
    monkeypatch.setattr(
        importlib.machinery.PathFinder, "find_spec", lambda name, path=None: None
    )

    with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "0"}):
        orig_meta = list(sys.meta_path)
        try:
            spec = importlib.util.spec_from_file_location("test_mod_no_next", path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        finally:
            sys.meta_path[:] = orig_meta


def test_chain_next_sitecustomize_with_next(monkeypatch, hbm_sitecustomize):
    """Test _chain_next_sitecustomize loads next module when found."""
    mod, path = hbm_sitecustomize
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
            spec = importlib.util.spec_from_file_location("test_mod_with_next", path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            fake_loader.exec_module.assert_called_once()
        finally:
            sys.meta_path[:] = orig_meta


def test_hbm_ledger_category_matching(monkeypatch, hbm_sitecustomize):
    """Test _category matches all keywords and falls back to other."""
    _, path = hbm_sitecustomize
    orig_meta = list(sys.meta_path)
    try:
        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
            spec = importlib.util.spec_from_file_location("test_hbm_cat", path)
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


def test_hbm_ledger_mem_line(monkeypatch, hbm_sitecustomize, capsys):
    """Test _mem_line under normal and exception handling paths."""
    _, path = hbm_sitecustomize
    orig_meta = list(sys.meta_path)
    try:
        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
            spec = importlib.util.spec_from_file_location("test_hbm_mem", path)
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
                lambda dev: {"peak_allocated": 2 * 1024 * 1024 * 1024, "other_stat": 0},
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


def test_hbm_ledger_param_ledger(monkeypatch, hbm_sitecustomize, capsys):
    """Test _param_ledger with missing model, duplicate storage, and device suffixes."""
    _, path = hbm_sitecustomize
    orig_meta = list(sys.meta_path)
    try:
        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
            spec = importlib.util.spec_from_file_location("test_hbm_param", path)
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
            worker = SimpleNamespace(model_runner=SimpleNamespace(model=model), rank=1)
            param_fn(worker)
            out = capsys.readouterr().out
            assert "[HBM_LEDGER] rank=1 stage=param_ledger" in out
            assert "param:embed" in out
            assert "param:engram@cpu" in out
            assert "buffer:norm" in out
    finally:
        sys.meta_path[:] = orig_meta


def test_hbm_ledger_patch_and_wrappers(monkeypatch, hbm_sitecustomize, capsys):
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
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)

            class FakeTPUWorker:
                def __init__(self, rank=0):
                    self.rank = rank
                    self.devices = [torch.device("cpu")]
                    self.model_runner = None

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


def test_hbm_ledger_patch_exception_handling(monkeypatch, hbm_sitecustomize, capsys):
    """Test exception in after callback is caught and logged."""
    _, path = hbm_sitecustomize
    orig_meta = list(sys.meta_path)
    try:
        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
            spec = importlib.util.spec_from_file_location("test_hbm_exc", path)
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
                "[HBM_LEDGER] load_model probe failed: RuntimeError('probe exploded')"
                in out
            )
    finally:
        sys.meta_path[:] = orig_meta


def test_hbm_ledger_finder_and_loader(monkeypatch, hbm_sitecustomize):
    """Test MetaPathFinder and Loader hooks."""
    _, path = hbm_sitecustomize
    orig_meta = list(sys.meta_path)
    try:
        with patch.dict(os.environ, {"DSV41_HBM_LEDGER": "1"}):
            spec = importlib.util.spec_from_file_location("test_hbm_finder", path)
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
            monkeypatch.setattr(importlib.util, "find_spec", lambda fullname: fake_spec)
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


def test_xplane_summary_full_flow(monkeypatch, capsys):
    """Test xplane_summary CLI parsing, planes/lines/modules/ops summaries."""
    xplane_path = _SCRIPTS_DIR / "xplane_summary.py"

    class MockEvent:
        def __init__(self, name, start_ns, duration_ns):
            self.name = name
            self.start_ns = start_ns
            self.duration_ns = duration_ns

    # Modules: 3 events, two big (>5ms) -> test gaps
    mod_events = [
        MockEvent("v41_encoder", 1_000_000_000, 10_000_000),  # 10ms (big)
        MockEvent("v41_encoder", 1_020_000_000, 8_000_000),  # 8ms (big) -> gap = 10ms
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

    # Run script with custom args
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


def test_xplane_summary_default_plane_and_no_ops_or_gaps(monkeypatch, capsys):
    """Test xplane_summary with default plane and single big module."""
    xplane_path = _SCRIPTS_DIR / "xplane_summary.py"

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


# ==============================================================================
# 3. Tests for scripts/dsv41_bench/bench_mhc.py
# ==============================================================================


def test_bench_mhc_full_inputs_cpu():
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


def test_bench_mhc_slice_rows_and_conversions():
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


def test_bench_mhc_maybe_compile(monkeypatch):
    """Verify _maybe_compile disabled vs enabled."""

    def fn(x):
        return x + 1

    assert bench_mhc._maybe_compile(fn, False) is fn

    mock_compile = Mock(return_value="compiled_fn")
    monkeypatch.setattr(torch, "compile", mock_compile)
    res = bench_mhc._maybe_compile(fn, True)
    assert res == "compiled_fn"
    mock_compile.assert_called_once()


def test_bench_mhc_make_fn_all_subops(monkeypatch):
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
        lambda *args, **kwargs: (dummy_post, dummy_comb, dummy_val, dummy_next),
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
        return_value=(torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4, 4))
    )
    monkeypatch.setattr(bench_mhc, "_gates_op", lambda g: mock_op)
    fn_sink = bench_mhc._make_fn("sinkhorn_only", geo)
    out_sink = fn_sink(inputs)
    assert "pre" in out_sink and "post" in out_sink

    # 6. Invalid subop
    with pytest.raises(ValueError, match="unknown_subop"):
        bench_mhc._make_fn("unknown_subop", geo)


def test_bench_mhc_gates_op_construction_and_call():
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
                bench_mhc, "_gates_jax", return_value=("pre", "post", "comb")
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


def test_bench_mhc_gates_jax():
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


def test_bench_mhc_roofline():
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
def test_bench_mhc_collective_helper(monkeypatch, mode):
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


def test_bench_mhc_worker_execution(monkeypatch, tmp_path):
    """Test bench_mhc.worker execution across modes, verification, and golden paths."""
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
    monkeypatch.setattr(common, "profile_steps", lambda ctx, args, step, label: None)

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
    monkeypatch.setattr(bench_mhc, "_make_fn", lambda subop, geo: lambda inp: dummy_out)
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
        bench_mhc, "_collective", lambda ctx, args, tokens_full, geo: fake_coll_run
    )
    monkeypatch.setattr(
        common,
        "load_golden",
        lambda path: ({"out": torch.zeros(64)}, {"meta": "coll_meta"}),
    )
    res_coll = bench_mhc.worker(ctx, coll_args)
    assert res_coll["diagnostics_vs_fp32_truth"] == {"sp": {"ok": True}}


def test_bench_mhc_main(monkeypatch):
    """Verify bench_mhc.main configures parser and calls common.run_main."""
    run_main_mock = Mock()
    monkeypatch.setattr(common, "run_main", run_main_mock)
    monkeypatch.setattr(sys, "argv", ["bench_mhc.py", "--mode", "sp", "--no-compile"])
    bench_mhc.main()
    run_main_mock.assert_called_once()
    assert run_main_mock.call_args[0][0] == "mhc"

    # Also test __main__ execution block via runpy
    with patch.object(sys, "argv", ["bench_mhc.py", "--mode", "tp"]):
        runpy.run_path(str(_SCRIPTS_DIR / "bench_mhc.py"), run_name="__main__")
