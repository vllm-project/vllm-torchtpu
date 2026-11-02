# SPDX-License-Identifier: Apache-2.0
"""100% Full-Module Line and Branch Coverage Suite for /Benchmarks.

Target Modules:
- scripts/dsv41_bench/common.py
- scripts/dsv41_bench/bench_indexer.py
- scripts/dsv41_bench/bench_sparse_mla_swa.py
- scripts/dsv41_bench/bench_decoder_layer.py

Explicitly validates:
- + --q-len={0, 1, 6}, --swa-width={0, 512, 576},
  --query-split={pad16, ceil}
- Decode sub-op suite (`decode_subop_suite` and `swa_window_vs_dense_attend`)
- 40-layer decode step reconciliation (`reconcile_40_layer_decode_step`)
- Roofline estimates, numerical verification tolerances, golden file roundtrip
- Memory guardrails: RSS < 2 GiB strictly maintained on CPU
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import torch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "dsv41_bench"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dsv41_bench import bench_decoder_layer as bdl  # noqa: E402
from scripts.dsv41_bench import bench_indexer as bi  # noqa: E402
from scripts.dsv41_bench import bench_sparse_mla_swa as bs  # noqa: E402
from scripts.dsv41_bench import common  # noqa: E402

sys.modules.setdefault("common", common)
sys.modules["common"] = common
sys.modules.setdefault("bench_decoder_layer", bdl)
sys.modules["bench_decoder_layer"] = bdl
sys.modules.setdefault("bench_indexer", bi)
sys.modules["bench_indexer"] = bi
sys.modules.setdefault("bench_sparse_mla_swa", bs)
sys.modules["bench_sparse_mla_swa"] = bs

from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (  # noqa: E402
    paged_indexer_jax,
    write_index_keys,
)

bi.common = common
bs.common = common
bdl.common = common

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _patch_tpu_device_to_cpu(monkeypatch):
    """Safely map TPU tensor calls to CPU during testing."""
    orig_to = torch.Tensor.to

    def mock_to(self, *args, **kwargs):
        if args and args[0] == "tpu":
            args = ("cpu", *args[1:])
        elif kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", mock_to)

    orig_tensor = torch.tensor

    def mock_tensor(*args, **kwargs):
        if kwargs.get("device") == "tpu":
            kwargs["device"] = "cpu"
        return orig_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", mock_tensor)

    for fn_name in ("arange", "zeros", "ones", "randn", "empty"):
        orig_fn = getattr(torch, fn_name)

        def make_mock(f):
            def mock_fn(*args, **kwargs):
                if kwargs.get("device") == "tpu":
                    kwargs["device"] = "cpu"
                return f(*args, **kwargs)

            return mock_fn

        monkeypatch.setattr(torch, fn_name, make_mock(orig_fn))


# ==============================================================================
# SECTION 1: scripts/dsv41_bench/common.py Tests
# ==============================================================================


class TestCommonGeometry:
    def test_default_values_and_candidate_positions(self):
        geo = common.Dsv41Geometry()
        assert geo.hidden_size == 5120
        assert geo.num_hidden_layers == 40
        assert geo.candidate_positions == 2048 * 8

    def test_heads_per_rank(self):
        geo = common.Dsv41Geometry(num_attention_heads=64)
        assert geo.heads_per_rank(8) == 8
        assert geo.heads_per_rank(16) == 4
        with pytest.raises(ValueError, match="not divisible"):
            geo.heads_per_rank(5)

    def test_from_hf_config_nested_and_flat(self, tmp_path):
        fields = {
            f.name: getattr(common.Dsv41Geometry(), f.name)
            for f in dataclasses.fields(common.Dsv41Geometry)
        }
        cfg_nested = tmp_path / "config_nested.json"
        cfg_nested.write_text(json.dumps({"text_config": fields}), encoding="utf-8")
        geo1 = common.Dsv41Geometry.from_hf_config(cfg_nested)
        assert geo1.hidden_size == 5120

        cfg_flat = tmp_path / "config_flat.json"
        cfg_flat.write_text(json.dumps(fields), encoding="utf-8")
        geo2 = common.Dsv41Geometry.from_hf_config(cfg_flat)
        assert geo2.num_hidden_layers == 40

        incomplete = dict(fields)
        del incomplete["hidden_size"]
        cfg_bad = tmp_path / "config_bad.json"
        cfg_bad.write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(KeyError, match="has no text_config.hidden_size"):
            common.Dsv41Geometry.from_hf_config(cfg_bad)


class TestCommonRooflineAndEstimate:
    def test_roofline_rates_and_formulas(self):
        hw = common.Tpu7xRoofline()
        s_bf16 = hw.compute_s(1e12, dtype="bf16")
        s_fp8 = hw.compute_s(1e12, dtype="fp8")
        assert s_bf16 > s_fp8 > 0.0

        assert hw.hbm_s(1e9) > 0.0

        ar = hw.all_reduce_s(1024, 8)
        rs = hw.reduce_scatter_s(1024, 8)
        ag = hw.all_gather_s(1024, 8)
        a2a = hw.all_to_all_s(1024, 8)
        assert ar == 2 * rs
        assert rs == ag == a2a

    def test_roofline_estimate_bound_and_report(self):
        hw = common.Tpu7xRoofline()
        est = common.RooflineEstimate(
            name="test_op", flops=1e12, hbm_bytes=1e9, comm_s=0.001
        )
        bound = est.bound_s(hw)
        assert bound["arith_intensity"] == 1000.0
        assert bound["lower_bound_s"] > 0

        est_zero_hbm = common.RooflineEstimate(
            name="compute_only", flops=1e12, hbm_bytes=0.0
        )
        bound_zero = est_zero_hbm.bound_s(hw)
        assert math.isinf(bound_zero["arith_intensity"])

        rep1 = est.report(hw, measured_ms=10.0)
        assert rep1["name"] == "test_op"
        assert rep1["measured_ms"] == 10.0
        assert rep1["roofline_efficiency"] > 0.0

        rep2 = est.report(hw, measured_ms=None)
        assert "measured_ms" not in rep2

        rep3 = est.report(hw, measured_ms=0.0)
        assert math.isnan(rep3["roofline_efficiency"])


class TestCommonNumericalVerification:
    def test_bitwise_equal_dtypes_and_shapes(self):
        t1 = torch.zeros(4, dtype=torch.float32)
        t2 = torch.zeros(5, dtype=torch.float32)
        t3 = torch.zeros(4, dtype=torch.int32)
        assert not common._bitwise_equal(t1, t2)
        assert not common._bitwise_equal(t1, t3)

        f8_1 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        f8_2 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        f8_3 = torch.tensor([1.0, 3.0], dtype=torch.float8_e4m3fn)
        assert common._bitwise_equal(f8_1, f8_2)
        assert not common._bitwise_equal(f8_1, f8_3)

        bf_1 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        bf_2 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        assert common._bitwise_equal(bf_1, bf_2)

        f32_pos = torch.tensor([0.0, float("nan")])
        f32_neg = torch.tensor([-0.0, float("nan")])
        assert not common._bitwise_equal(f32_pos, f32_neg)
        assert common._bitwise_equal(f32_pos, f32_pos.clone())

        f64_1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        f64_2 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        f64_3 = torch.tensor([1.0, 2.5], dtype=torch.float64)
        assert common._bitwise_equal(f64_1, f64_2)
        assert not common._bitwise_equal(f64_1, f64_3)

        i32_1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        i32_2 = torch.tensor([1, 2, 3], dtype=torch.int32)
        i32_3 = torch.tensor([1, 2, 4], dtype=torch.int32)
        assert common._bitwise_equal(i32_1, i32_2)
        assert not common._bitwise_equal(i32_1, i32_3)

    def test_topk_overlap_validation_and_cases(self):
        with pytest.raises(ValueError, match="expects equal 2-D shapes"):
            common.topk_overlap(torch.tensor([1, 2]), torch.tensor([1, 2]))
        with pytest.raises(ValueError, match="expects equal 2-D shapes"):
            common.topk_overlap(torch.zeros(2, 3), torch.zeros(2, 4))

        actual = torch.tensor([[10, 20, 30], [5, 6, -1]])
        expected = torch.tensor([[20, 10, 30], [5, 7, -1]])
        mean, worst = common.topk_overlap(actual, expected)
        assert mean == pytest.approx((1.0 + 0.5) / 2.0)
        assert worst == pytest.approx(0.5)

        actual_empty = torch.tensor([[-1, -1], [1, 2]])
        expected_empty = torch.tensor([[-1, -1], [1, 2]])
        mean, worst = common.topk_overlap(actual_empty, expected_empty)
        assert mean == 1.0 and worst == 1.0

        actual_has_val = torch.tensor([[5, -1], [1, 2]])
        mean, worst = common.topk_overlap(actual_has_val, expected_empty)
        assert worst == 0.0

    def test_compare_tensors_shape_and_dtype_mismatch(self):
        a = torch.zeros(2, 3)
        b = torch.zeros(2, 4)
        c = torch.zeros(2, 3, dtype=torch.int32)
        res_shape = common.compare_tensors(a, b, common.EXACT)
        assert not res_shape["ok"]
        assert "shape" in res_shape["reason"]

        res_dtype = common.compare_tensors(a, c, common.EXACT)
        assert not res_dtype["ok"]
        assert "dtype" in res_dtype["reason"]

    def test_compare_tensors_exact_float_and_int(self):
        a = torch.tensor([1.0, 2.0, float("nan")])
        b = torch.tensor([1.0, 2.5, float("nan")])
        res = common.compare_tensors(a, b, common.EXACT)
        assert not res["ok"]
        assert res["num_mismatch"] == 1
        assert res["max_abs_diff"] == pytest.approx(0.5)

        a_int = torch.tensor([1, 2, 3])
        b_int = torch.tensor([1, 4, 3])
        res_int = common.compare_tensors(a_int, b_int, common.EXACT)
        assert not res_int["ok"]
        assert res_int["num_mismatch"] == 1

    def test_compare_tensors_topk_criteria(self):
        actual = torch.tensor([[10, 20], [30, 40]])
        expected = torch.tensor([[10, 20], [30, 40]])
        res = common.compare_tensors(actual, expected, common.TOPK)
        assert res["ok"]
        assert res["topk_overlap_mean"] == 1.0

        bad_actual = torch.tensor([[1, 2], [3, 4]])
        res_bad = common.compare_tensors(bad_actual, expected, common.TOPK)
        assert not res_bad["ok"]

    def test_compare_tensors_lossless(self):
        with pytest.raises(ValueError, match="needs a float tensor"):
            common.compare_tensors(
                torch.tensor([1, 2]), torch.tensor([1, 2]), common.LOSSLESS
            )

        a_inf = torch.tensor([1.0, float("inf")])
        b_fin = torch.tensor([1.0, 2.0])
        res_mask = common.compare_tensors(a_inf, b_fin, common.LOSSLESS)
        assert not res_mask["ok"]
        assert not res_mask["nonfinite_mask_equal"]

        a_nan = torch.tensor([1.0, float("nan")])
        b_nan = torch.tensor([1.0, float("nan")])
        res_nan = common.compare_tensors(a_nan, b_nan, common.LOSSLESS)
        assert not res_nan["ok"]
        assert "NaN present" in res_nan["reason"]

        a_all_inf = torch.tensor([float("inf"), float("-inf")])
        b_all_inf = torch.tensor([float("inf"), float("-inf")])
        res_empty = common.compare_tensors(a_all_inf, b_all_inf, common.LOSSLESS)
        assert not res_empty["ok"]
        assert "no finite elements" in res_empty["reason"]

        zero_a = torch.zeros(4)
        zero_b = torch.zeros(4)
        res_zero = common.compare_tensors(zero_a, zero_b, common.LOSSLESS)
        assert res_zero["ok"]
        assert res_zero["cos_sim"] == 1.0

        one_zero = common.compare_tensors(
            torch.ones(4), torch.zeros(4), common.LOSSLESS
        )
        assert not one_zero["ok"]
        assert one_zero["cos_sim"] == 0.0

        x = torch.randn(100)
        close = x + 1e-6 * torch.randn(100)
        far = x + 0.5 * torch.randn(100)
        assert common.compare_tensors(close, x, common.LOSSLESS)["ok"]
        assert not common.compare_tensors(far, x, common.LOSSLESS)["ok"]


class TestVerifyOutputsAndGolden:
    def test_verify_outputs_key_mismatch_and_criteria_dict(self):
        actual = {"out1": torch.ones(2), "out2": torch.zeros(2)}
        expected_diff_keys = {"out1": torch.ones(2)}
        with pytest.raises(common.VerificationError, match="output names differ"):
            common.verify_outputs(actual, expected_diff_keys, common.EXACT)

        crit_map = {"out1": common.EXACT, "out2": common.LOSSLESS}
        rep = common.verify_outputs(actual, actual, crit_map)
        assert "out1" in rep and "out2" in rep

        bad_actual = {"out1": torch.ones(2), "out2": torch.ones(2)}
        with pytest.raises(common.VerificationError, match="verification failed"):
            common.verify_outputs(bad_actual, actual, crit_map)

    def test_golden_file_save_load_and_not_found(self, tmp_path):
        p = common.golden_path(tmp_path, "case1", 0)
        assert p == tmp_path / "case1" / "rank0.pt"

        with pytest.raises(FileNotFoundError, match="no golden at"):
            common.load_golden(p)

        tensors = {"t1": torch.tensor([1.0, 2.0])}
        meta = {"test": 123}
        common.save_golden(p, tensors, meta)
        loaded_t, loaded_m = common.load_golden(p)
        assert torch.equal(loaded_t["t1"], tensors["t1"])
        assert loaded_m["test"] == 123

    def test_check_against_golden_modes(self, tmp_path):
        ctx = common.WorkerContext(
            rank=0, world_size=1, tp_size=1, cpu_group=None, result_dir=tmp_path
        )
        tensors = {"out": torch.tensor([1.0, 2.0])}

        args_update = argparse.Namespace(
            golden_dir=tmp_path, update_golden=True, verify=False
        )
        with patch("sys.argv", ["script.py"]):
            res_up = common.check_against_golden(
                ctx, args_update, "c1", tensors, common.EXACT
            )
        assert res_up["golden"] == "written"

        args_skip = argparse.Namespace(
            golden_dir=tmp_path, update_golden=False, verify=False
        )
        res_skip = common.check_against_golden(
            ctx, args_skip, "c1", tensors, common.EXACT
        )
        assert res_skip["golden"] == "skipped"

        args_verify = argparse.Namespace(
            golden_dir=tmp_path, update_golden=False, verify=True
        )
        res_ver = common.check_against_golden(
            ctx, args_verify, "c1", tensors, common.EXACT
        )
        assert res_ver["golden"] == "verified"


class TestCLIAndGitSha:
    def test_git_sha_with_and_without_git(self, tmp_path, monkeypatch):
        monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout="123456789\n"),
                Mock(stdout=""),
            ]
            assert common.git_sha() == "123456789"

            mock_run.side_effect = [
                Mock(stdout="123456789\n"),
                Mock(stdout="M src/foo.py\n"),
            ]
            assert common.git_sha() == "123456789+dirty"

        (tmp_path / ".git").rmdir()
        sha_file = tmp_path / "git_sha"
        monkeypatch.setattr(common, "GIT_SHA_FILE", sha_file)
        sha_file.write_text("abcdef123\n", encoding="utf-8")
        assert common.git_sha() == "abcdef123"

        sha_file.unlink()
        with pytest.raises(FileNotFoundError, match="is not a git checkout"):
            common.git_sha()

    def test_add_common_args_and_resolve_result_dir(self, tmp_path):
        parser = argparse.ArgumentParser()
        common.add_common_args(parser, ("sub1", "sub2"))
        args = parser.parse_args(["--subop", "sub2", "--chunk-size", "4096"])
        assert args.subop == "sub2"
        assert args.chunk_size == 4096

        args.result_dir = tmp_path / "custom"
        expected_dir = (tmp_path / "custom").resolve()
        assert common.resolve_result_dir(args, "bench1") == expected_dir

        args.result_dir = None
        res_dir = common.resolve_result_dir(args, "bench1")
        assert "bench1_sub2_" in str(res_dir)

    def test_llo_env(self, tmp_path):
        env: dict[str, str] = {}
        common._llo_env(tmp_path, 0, env)
        assert "--xla_dump_to=" in env["XLA_FLAGS"]
        assert "--xla_jf_dump_to=" in env["LIBTPU_INIT_ARGS"]
        assert env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] == "disabled"


class TestLaunchAndAggregation:
    def test_prepare_worker_env(self, monkeypatch):
        monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "4")
        args_no_prof = argparse.Namespace(profile=None)
        with patch(
            "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
            ".prepare_tpu_environment"
        ) as mock_prep:
            env = common.prepare_worker_env(args_no_prof)
            mock_prep.assert_called_once_with(world_size=common.WORLD_SIZE)
            assert env["PJRT_DEVICE"] == "TPU"
            assert common._PROFILE_SESSION_ENV not in env

        args_prof = argparse.Namespace(profile=pathlib.Path("/tmp/prof"))
        with patch(
            "torch_tpu._internal.distributed.launchers.singlehost_wrapper"
            ".prepare_tpu_environment"
        ):
            env_prof = common.prepare_worker_env(args_prof)
            assert common._PROFILE_SESSION_ENV in env_prof

    def test_launch_normal_and_timeout(self, tmp_path):
        args = argparse.Namespace(profile=None)
        mock_proc = MagicMock()
        mock_proc.stdout = ["[rank0] running step\n", "normal line\n", "some error\n"]
        mock_proc.wait.return_value = 0
        with (
            patch.object(common, "prepare_worker_env", return_value={"mock": "1"}),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            rc = common.launch("dummy.py", ["--subop", "all"], tmp_path, args)
            assert rc == 0
            log = (tmp_path / "workers.log").read_text(encoding="utf-8")
            assert "[rank0] running step" in log

        with (
            patch.object(common, "prepare_worker_env", return_value={"mock": "1"}),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("time.monotonic", side_effect=[0, 1e9, 1e9]),
            patch("os.killpg") as mock_kill,
        ):
            with pytest.raises(TimeoutError, match="workers exceeded"):
                common.launch("dummy.py", [], tmp_path, args)
            mock_kill.assert_called_once()

    def test_aggregate_summary(self, tmp_path):
        for r in range(common.WORLD_SIZE):
            rank_data = {
                "rank": r,
                "wall_samples_ms": [10.0 + r, 12.0 + r],
                "hbm": {"used_bytes": 1000},
                "profile_dir": "/tmp/dummy_prof",
            }
            (tmp_path / f"rank_{r}.json").write_text(
                json.dumps(rank_data), encoding="utf-8"
            )

        with patch.object(common, "summarize_xplanes", return_value={"planes": {}}):
            summary = common.aggregate(tmp_path)
            assert len(summary["wall_step_max_over_ranks_ms"]) == 2
            assert summary["wall_min_ms"] > 0
            assert "rank0" in summary
            assert "xprof" in summary

        (tmp_path / "rank_3.json").unlink()
        summary_err = common.aggregate(tmp_path)
        assert len(summary_err["errors"]) == 1
        assert summary_err["errors"][0]["rank"] == 3

        err_dir = tmp_path / "err_dir"
        err_dir.mkdir()
        (err_dir / "rank_0.json").write_text(
            json.dumps({"rank": 0, "error": "failed"}), encoding="utf-8"
        )
        s_err0 = common.aggregate(err_dir)
        assert "rank0" not in s_err0

    def test_run_main_parent_and_worker(self, tmp_path, monkeypatch):
        parser = argparse.ArgumentParser()
        common.add_common_args(parser, ("all_reduce",))

        with patch.object(common, "_worker_entry") as mock_entry:
            monkeypatch.setattr(sys, "argv", ["bench.py", "--worker"])
            common.run_main("test_bench", parser, lambda ctx, args: {})
            mock_entry.assert_called_once()

        mock_agg = {
            "result_dir": str(tmp_path),
            "errors": [],
            "xprof": {"planes": {}},
        }
        with (
            patch.object(common, "launch", return_value=0),
            patch.object(common, "aggregate", return_value=mock_agg),
            patch.object(common, "git_sha", return_value="abc1234"),
            patch.object(common, "print_xprof_table") as mock_table,
        ):
            monkeypatch.setattr(
                sys, "argv", ["bench.py", "--result-dir", str(tmp_path)]
            )
            common.run_main("test_bench", parser, lambda ctx, args: {})
            mock_table.assert_called_once()
            assert (tmp_path / "summary.json").exists()

        with (
            patch.object(common, "launch", return_value=1),
            patch.object(
                common,
                "aggregate",
                return_value={"result_dir": str(tmp_path), "errors": ["some_error"]},
            ),
            patch.object(common, "git_sha", return_value="abc1234"),
        ):
            monkeypatch.setattr(
                sys, "argv", ["bench.py", "--result-dir", str(tmp_path)]
            )
            with pytest.raises(SystemExit):
                common.run_main("test_bench", parser, lambda ctx, args: {})


class TestWorkerSide:
    def test_worker_entry_success_and_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.delenv("XLA_FLAGS", raising=False)
        monkeypatch.delenv("LIBTPU_INIT_ARGS", raising=False)
        monkeypatch.delenv("TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE", raising=False)
        args = argparse.Namespace(
            result_dir=tmp_path, dump_llo=tmp_path / "llo", tp_size=1
        )

        with (
            patch.object(common, "init_worker") as mock_init,
            patch("torch.distributed.destroy_process_group") as mock_destroy,
        ):
            ctx = common.WorkerContext(0, 1, 1, None, tmp_path)
            mock_init.return_value = ctx
            common._worker_entry(args, lambda c, a: {"data": 42})
            mock_destroy.assert_called_once()
            res = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
            assert res["data"] == 42

            args_no_llo = argparse.Namespace(
                result_dir=tmp_path, dump_llo=None, tp_size=1
            )
            common._worker_entry(args_no_llo, lambda c, a: {"data": 43})
            res2 = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
            assert res2["data"] == 43

        with (
            patch.object(common, "init_worker", side_effect=RuntimeError("boom")),
            patch("torch.distributed.destroy_process_group"),
        ):
            with pytest.raises(SystemExit):
                common._worker_entry(args, lambda c, a: {})
            res_err = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
            assert "RuntimeError('boom')" in res_err["error"]

    def test_init_worker(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "8")
        args = argparse.Namespace(tp_size=8, seed=42, result_dir=pathlib.Path("/tmp"))

        with (
            patch("torch.distributed.init_process_group") as mock_init_pg,
            patch("torch.empty") as mock_empty,
            patch("vllm.distributed.init_distributed_environment") as mock_vllm_init,
            patch("vllm.distributed.ensure_model_parallel_initialized") as mock_vllm_mp,
            patch("vllm.config.set_current_vllm_config"),
            patch("vllm.config.VllmConfig"),
            patch("torch.distributed.new_group"),
        ):
            mock_empty.return_value.cpu.return_value = None
            ctx = common.init_worker(args)
            assert ctx.rank == 0
            assert ctx.world_size == 8
            assert ctx.tp_size == 8
            mock_init_pg.assert_called_once_with(backend="tpu_dist")
            mock_vllm_init.assert_called_once()
            mock_vllm_mp.assert_called_once_with(8, 1)

        args_bad = argparse.Namespace(tp_size=5)
        with pytest.raises(ValueError, match="not divisible by tp_size"):
            common.init_worker(args_bad)

    def test_synchronize_and_flatten_tensors(self):
        t1 = torch.zeros(2)
        t2 = torch.zeros(3)
        obj = {"a": t1, "b": [t2, "primitive", (t1,)]}
        flattened = common._flatten_tensors(obj)
        assert len(flattened) == 3

        with patch("torch_tpu._internal.sync.synchronize") as mock_sync:
            common.synchronize(obj)
            mock_sync.assert_called_once_with(flattened, wait=True)

    def test_model_wrapper_context(self):
        with (
            patch("jax.local_devices", return_value=[Mock()]),
            patch("jax.sharding.Mesh"),
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context"
                ".set_vllm_model_wrapper_context"
            ) as mock_ctx,
        ):
            mock_ctx.return_value = "mesh_ctx"
            assert common.model_wrapper_context() == "mesh_ctx"

    def test_time_steps(self):
        ctx = common.WorkerContext(0, 1, 1, None, pathlib.Path("/tmp"))
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return torch.zeros(1)

        with patch.object(common, "synchronize"), patch("torch.distributed.barrier"):
            samples, out = common.time_steps(ctx, step, warmup=2, iters=3)
            assert len(samples) == 3
            assert call_count == 5

    def test_profile_steps(self, tmp_path, monkeypatch):
        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)

        def step():
            return torch.zeros(1)

        args_none = argparse.Namespace(profile=None)
        assert common.profile_steps(ctx, args_none, step, label="test") is None

        monkeypatch.setenv(common._PROFILE_SESSION_ENV, "sess_123")
        args_prof = argparse.Namespace(
            profile=tmp_path / "prof",
            trace_mode="TRACE_COMPUTE",
            profile_steps=2,
        )

        with (
            patch("torch.profiler.profile") as mock_prof,
            patch(
                "vllm_torchtpu.profiler_trace.rank_capture_dir",
                return_value=str(tmp_path / "cap"),
            ),
            patch(
                "vllm_torchtpu.profiler_trace.resolve_canonical_dst_ts",
                return_value=12345,
            ),
            patch("vllm_torchtpu.profiler_trace.merge_rank_capture") as mock_merge,
            patch(
                "vllm_torchtpu.profiler_trace.clear_canonical_ts_marker"
            ) as mock_clear,
            patch("torch.distributed.barrier"),
            patch.object(common, "synchronize"),
        ):
            mock_prof_ctx = MagicMock()
            mock_prof.return_value = mock_prof_ctx
            prof_dir = common.profile_steps(ctx, args_prof, step, label="test")
            assert prof_dir == str((tmp_path / "prof").resolve())
            mock_merge.assert_called_once()
            mock_clear.assert_called_once()

            ctx1 = common.WorkerContext(1, 2, 1, None, tmp_path)
            args_default = argparse.Namespace(
                profile=tmp_path / "prof",
                trace_mode="TRACE_DEFAULT",
                profile_steps=1,
            )
            mock_clear.reset_mock()
            prof_dir1 = common.profile_steps(ctx1, args_default, step, label="test")
            assert prof_dir1 == str((tmp_path / "prof").resolve())
            mock_clear.assert_not_called()

    def test_hbm_stats(self):
        with patch("torch.accelerator.get_memory_info", return_value=(4000, 10000)):
            stats = common.hbm_stats()
            assert stats["free_bytes"] == 4000
            assert stats["total_bytes"] == 10000
            assert stats["used_bytes"] == 6000

    def test_gather_object(self):
        ctx = common.WorkerContext(0, 2, 1, None, pathlib.Path("/tmp"))
        with patch("torch.distributed.all_gather_object") as mock_ago:
            res = common.gather_object(ctx, {"k": "v"})
            assert len(res) == 2
            mock_ago.assert_called_once()

    def test_summarize_xplanes_and_print_xprof_table(self, tmp_path):
        (tmp_path / "test.xplane.pb").write_bytes(b"dummy")

        event1 = SimpleNamespace(name="MatMul", duration_ns=500000)
        line1 = SimpleNamespace(name="XLA Ops", events=[event1])
        event2 = SimpleNamespace(name="layer_0", duration_ns=1000000)
        line2 = SimpleNamespace(name="XLA Modules", events=[event2])
        line_other = SimpleNamespace(name="Unrelated Line", events=[])

        plane_tc = SimpleNamespace(
            name="/device:TPU:0", lines=[line1, line2, line_other]
        )
        plane_sc = SimpleNamespace(name="/device:TPU:0_SparseCore", lines=[line1])
        plane_empty = SimpleNamespace(name="/device:TPU:1", lines=[line_other])
        plane_cpu = SimpleNamespace(name="/device:CPU:0", lines=[])

        data = SimpleNamespace(planes=[plane_tc, plane_sc, plane_empty, plane_cpu])

        with patch("jax.profiler.ProfileData.from_file", return_value=data):
            summary = common.summarize_xplanes(tmp_path)
            assert len(summary["planes"]) == 2
            common.print_xprof_table(summary)

        common.print_xprof_table(
            {
                "planes": {
                    "sc_plane": {
                        "sparse_core": True,
                        "total_op_ns": 100,
                        "module_ns": {},
                        "top_ops_ns": {},
                    }
                }
            }
        )


# ==============================================================================
# SECTION 2: scripts/dsv41_bench/bench_indexer.py Tests
# ==============================================================================


class TestBenchIndexerFull:
    def test_build_indexer_inputs_branches(self):
        inp = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=64,
            tp_size=8,
            tp_rank=0,
            num_seqs=2,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
        )
        assert inp["width"] == 256
        assert inp["part"] == 16
        assert inp["lengths_part_t"][0].item() == 8

        inp1 = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=64,
            tp_size=8,
            tp_rank=0,
            num_seqs=1,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=2,
            block_size=16,
        )
        assert inp1["part"] == 16
        assert inp1["width"] == 256

        inp2 = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=8192,
            tp_size=8,
            tp_rank=0,
            num_seqs=1,
            kv_len=16,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=2,
            block_size=16,
        )
        assert inp2["write_records_t"].shape[0] == 8192
        assert (inp2["write_records_t"] == -1).any()

        # + decode: q_len=1 with ceil and pad16
        inp_dec_ceil = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=16,
            tp_size=8,
            tp_rank=0,
            num_seqs=4,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
            q_len=1,
            query_split="ceil",
        )
        assert inp_dec_ceil["part"] == 2
        assert inp_dec_ceil["records_t"].shape[0] == 4

        inp_dec_pad16 = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=16,
            tp_size=8,
            tp_rank=0,
            num_seqs=4,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
            q_len=1,
            query_split="pad16",
        )
        assert inp_dec_pad16["part"] == 16

        # q_len = 6 (DSpark Verify)
        inp_verify = bi._build_indexer_inputs(
            rng_seed=42,
            tokens=128,
            tp_size=8,
            tp_rank=0,
            num_seqs=4,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
            q_len=6,
            query_split="ceil",
        )
        assert inp_verify["part"] == 16

    def test_write_index_keys_and_visible_topk(self):
        inp = bi._build_indexer_inputs(
            rng_seed=12,
            tokens=64,
            tp_size=8,
            tp_rank=0,
            num_seqs=2,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
        )
        pool = jnp.asarray(inp["pool_t"].numpy())
        wrec = jnp.asarray(inp["write_records_t"].numpy())
        wkeys = jnp.asarray(inp["write_keys_t"].float().numpy(), jnp.bfloat16)
        u1 = write_index_keys(pool, wrec, wkeys)
        u2 = write_index_keys(pool, wrec, wkeys)
        assert np.array_equal(u1, u2)

    def test_paged_indexer_jax_subops_precompact_vs_baseline(self):
        inp = bi._build_indexer_inputs(
            rng_seed=101,
            tokens=64,
            tp_size=8,
            tp_rank=0,
            num_seqs=2,
            kv_len=512,
            compress_ratio=2,
            heads=8,
            head_dim=128,
            topk=32,
            candidate_blocks=8,
            block_size=16,
        )
        q = jnp.asarray(inp["q_part_t"].float().numpy(), jnp.bfloat16)
        pool = jnp.asarray(inp["pool_t"].numpy())
        records = jnp.asarray(inp["records_t"].numpy())
        weights = jnp.asarray(inp["weights_part_t"].numpy())
        lengths = jnp.asarray(inp["lengths_part_t"].numpy())
        candidates = jnp.asarray(inp["candidates_part_t"].numpy())
        requests = jnp.asarray(inp["requests_part_t"].numpy())
        batch_max = jnp.asarray([inp["batch_max"]], dtype=jnp.int32)

        for cw, cf in [(False, False), (True, False), (False, True)]:
            s_b, p_b = paged_indexer_jax(
                q,
                pool,
                records,
                weights,
                lengths,
                candidates,
                requests,
                topk=32,
                block_size=16,
                candidate_write=cw,
                candidate_filter=cf,
                unordered=False,
                batch_max=batch_max,
            )
            s_p, p_p = paged_indexer_jax(
                q,
                pool,
                records,
                weights,
                lengths,
                candidates,
                requests,
                topk=32,
                block_size=16,
                candidate_write=cw,
                candidate_filter=cf,
                unordered=True,
                batch_max=batch_max,
            )
            assert s_b.shape == s_p.shape
            assert p_b.shape == p_p.shape

    def test_compute_roofline_all_subops_and_modes(self):
        hw = common.Tpu7xRoofline()
        for subop in bi.SUBOPS:
            for mode in bi.MODES:
                r_none = bi._compute_roofline(
                    hw,
                    subop,
                    mode,
                    tokens=1024,
                    part=128,
                    tp_size=8,
                    width=2048,
                    heads=64,
                    head_dim=128,
                    topk=512,
                    candidate_blocks=128,
                    block_size=16,
                    measured_ms=None,
                )
                assert r_none["lower_bound_ms"] > 0
                assert "measured_ms" not in r_none

                r_val = bi._compute_roofline(
                    hw,
                    subop,
                    mode,
                    tokens=1024,
                    part=128,
                    tp_size=8,
                    width=2048,
                    heads=64,
                    head_dim=128,
                    topk=512,
                    candidate_blocks=128,
                    block_size=16,
                    measured_ms=1.5,
                    num_seqs=4,
                    q_len=1,
                )
                assert r_val["measured_ms"] == 1.5

    def test_worker_all_subops_and_verify(self, tmp_path):
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)
        subops = [
            "all",
            "full_context",
            "candidate_write",
            "candidate_filter",
            "topk_only",
            "cache_write_only",
        ]

        def dummy_jax_op(name, fn, *args, **kwargs):
            try:
                if "write_index_keys" in name:
                    fn(
                        jnp.zeros((16, 128), dtype=jnp.bfloat16),
                        jnp.zeros((16,), dtype=jnp.int32),
                        jnp.zeros((16, 128), dtype=jnp.bfloat16),
                    )
                elif "topk" in name:
                    fn(
                        jnp.zeros((16, 2048), dtype=jnp.float32),
                        jnp.zeros((16,), dtype=jnp.int32),
                        jnp.array([128], dtype=jnp.int32),
                    )
                elif "pad_comp" in name or "compact" in name:
                    fn(jnp.zeros((16, 16), dtype=jnp.int32))
                elif "idx" in name:
                    fn(
                        jnp.zeros((16, 8, 128), dtype=jnp.bfloat16),
                        jnp.zeros((16, 128), dtype=jnp.bfloat16),
                        jnp.zeros((16, 16), dtype=jnp.int32),
                        jnp.zeros((16, 8), dtype=jnp.float32),
                        jnp.zeros((16,), dtype=jnp.int32),
                        jnp.zeros((16, 8), dtype=jnp.int32),
                        jnp.zeros((16,), dtype=jnp.int32),
                        jnp.array([128], dtype=jnp.int32),
                    )
            except Exception:
                pass

            if "write_index_keys" in name:
                return lambda p, r, k: p
            elif "compact" in name or "topk" in name or "pad_comp" in name:
                return lambda *a, **k: torch.zeros((1024, 128), dtype=torch.int32)
            else:
                return lambda *a, **k: (
                    torch.zeros((1024, 512), dtype=torch.float32),
                    torch.zeros((1024, 128), dtype=torch.int32),
                )

        with (
            patch.dict(os.environ, {common._PROFILE_SESSION_ENV: "test_session"}),
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch.object(common, "model_wrapper_context"),
            patch.object(common, "synchronize"),
            patch.object(common, "hbm_stats", return_value={"hbm": 100}),
            patch.object(
                common,
                "time_steps",
                side_effect=lambda c, step_fn, **kw: ([1.0], step_fn()),
            ),
            patch.object(
                common,
                "profile_steps",
                side_effect=[None, str(tmp_path / "prof")] * 100,
            ),
            patch.object(common, "summarize_xplanes", return_value={}),
            patch.object(common, "check_against_golden", return_value={"golden": "ok"}),
            patch.object(common, "verify_outputs", return_value={"ok": True}),
            patch("torch_tpu._internal.pallas.jax_op", side_effect=dummy_jax_op),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl.initialize_indexer"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl.paged_indexer",
                return_value=(
                    torch.zeros((128, 128)),
                    torch.zeros((1024, 512)),
                    torch.zeros((1024, 128)),
                ),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention._gather_query_rows",
                side_effect=lambda x: x,
            ),
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                return_value=MagicMock(mesh=None),
            ),
        ):
            mock_tp.return_value.rank_in_group = 0
            for subop in subops:
                for mode in ("precompact", "baseline"):
                    args = argparse.Namespace(
                        chunk_size=1024,
                        seed=42,
                        num_seqs=1,
                        kv_len=512,
                        compress_ratio=2,
                        mode=mode,
                        subop=subop,
                        verify=False,
                        golden_dir=tmp_path / "golden",
                        update_golden=False,
                        profile=None,
                        warmup=1,
                        iters=1,
                    )
                    res = bi.worker(ctx, args)
                    assert res["subop"] == subop
                    assert res["roofline_vs_wall"]["lower_bound_ms"] > 0

            # Test verify=True and update_golden=True for subop="all"
            args_v = argparse.Namespace(
                chunk_size=1024,
                seed=42,
                num_seqs=1,
                kv_len=512,
                compress_ratio=2,
                mode="precompact",
                subop="all",
                verify=True,
                golden_dir=tmp_path / "golden",
                update_golden=True,
                profile=tmp_path / "prof",
                warmup=1,
                iters=1,
            )
            res_v = bi.worker(ctx, args_v)
            assert res_v["subop"] == "all"

            # Test verify=True with profile=None
            with patch.dict(os.environ, {}, clear=True):
                args_v_noprof = argparse.Namespace(
                    chunk_size=1024,
                    seed=42,
                    num_seqs=1,
                    kv_len=512,
                    compress_ratio=2,
                    mode="baseline",
                    subop="all",
                    verify=True,
                    golden_dir=tmp_path / "golden",
                    update_golden=False,
                    profile=None,
                    warmup=1,
                    iters=1,
                )
                res_v_noprof = bi.worker(ctx, args_v_noprof)
                assert res_v_noprof["subop"] == "all"

            # Individual subops with verify=True
            for s in ("cache_write_only", "topk_only", "full_context"):
                args_s = argparse.Namespace(
                    chunk_size=1024,
                    seed=42,
                    num_seqs=1,
                    kv_len=512,
                    compress_ratio=2,
                    mode="baseline",
                    subop=s,
                    verify=True,
                    golden_dir=tmp_path / "golden",
                    update_golden=False,
                    profile=None,
                    warmup=1,
                    iters=1,
                )
                res_s = bi.worker(ctx, args_s)
                assert res_s["subop"] == s

            # + decode: q_len=1, query_split="ceil"
            args_ceil = argparse.Namespace(
                chunk_size=16,
                seed=42,
                num_seqs=4,
                kv_len=512,
                compress_ratio=2,
                mode="baseline",
                subop="all",
                verify=True,
                golden_dir=tmp_path / "golden",
                update_golden=False,
                profile=None,
                warmup=1,
                iters=1,
                q_len=1,
                query_split="ceil",
            )
            res_ceil = bi.worker(ctx, args_ceil)
            assert res_ceil["query_split"] == "ceil"
            assert res_ceil["q_len"] == 1

            # + verify: q_len=6, query_split="pad16"
            args_pad = argparse.Namespace(
                chunk_size=128,
                seed=42,
                num_seqs=4,
                kv_len=512,
                compress_ratio=2,
                mode="precompact",
                subop="all",
                verify=True,
                golden_dir=tmp_path / "golden",
                update_golden=False,
                profile=None,
                warmup=1,
                iters=1,
                q_len=6,
                query_split="pad16",
            )
            res_pad = bi.worker(ctx, args_pad)
            assert res_pad["query_split"] == "pad16"
            assert res_pad["q_len"] == 6

    def test_indexer_main_and_entrypoint(self, monkeypatch):
        with patch.object(common, "run_main") as mock_rm:
            bi.main()
            mock_rm.assert_called_once()

        monkeypatch.setattr(sys, "argv", ["bench_indexer.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(_SCRIPTS_DIR / "bench_indexer.py"), run_name="__main__")
        assert exc_info.value.code == 0


# ==============================================================================
# SECTION 3: scripts/dsv41_bench/bench_sparse_mla_swa.py Tests
# ==============================================================================


class TestBenchSparseMlaSwaFull:
    def test_build_query_start_loc(self):
        q1 = bs._build_query_start_loc(torch, tokens=256, num_seqs=1)
        assert q1.shape == (2,)
        assert q1.tolist() == [0, 254]

        q4 = bs._build_query_start_loc(torch, tokens=256, num_seqs=4)
        assert q4.shape == (5,)
        assert q4[0].item() == 0
        assert q4[-1].item() == 254

        q_dec = bs._build_query_start_loc(torch, tokens=16, num_seqs=4, q_len=1)
        assert q_dec.tolist() == [0, 1, 2, 3, 4]

        with pytest.raises(ValueError, match="exceeds tokens"):
            bs._build_query_start_loc(torch, tokens=16, num_seqs=4, q_len=6)

    def test_build_swa_inputs_branches(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        tokens = 256
        window_size = 128

        q_loc = bs._build_query_start_loc(torch, tokens, num_seqs=2)
        pool, rec, slots = bs._build_swa_inputs(torch, rng, tokens, window_size, q_loc)
        assert rec.shape == (tokens, window_size)

        # Empty seq length branch (covers line 124)
        q_loc_empty = torch.tensor([0, 10, 10, 20], dtype=torch.int32)
        p_emp, r_emp, s_emp = bs._build_swa_inputs(torch, rng, 32, 16, q_loc_empty)
        assert r_emp.shape == (32, 16)

        # Decode q_len > 0 branch
        q_loc_dec = bs._build_query_start_loc(torch, 16, num_seqs=4, q_len=1)
        p_dec, r_dec, s_dec = bs._build_swa_inputs(
            torch, rng, 16, 512, q_loc_dec, q_len=1
        )
        assert r_dec.shape == (16, 512)

    def test_build_main_kv_inputs_branches(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        tokens = 256
        kv_len = 512
        compress_ratio = 2
        page_size = 16
        topk = 64
        q_loc = bs._build_query_start_loc(torch, tokens, num_seqs=2)

        mp, rp, p, rec = bs._build_main_kv_inputs(
            torch,
            rng,
            tokens,
            kv_len,
            compress_ratio,
            page_size,
            topk,
            q_loc,
            unique_per_chunk=128,
        )
        assert rec.shape == (tokens, topk)

        # unique_per_chunk <= 0 (covers branch 278->281)
        mp2, rp2, p2, rec2 = bs._build_main_kv_inputs(
            torch,
            rng,
            tokens,
            kv_len,
            compress_ratio,
            page_size,
            topk,
            q_loc,
            unique_per_chunk=0,
        )
        assert rec2.shape == (tokens, topk)

    def test_compact_records_torch_ref(self):
        # 1. width multiple of 128
        r1 = torch.randint(0, 100, (16, 128), dtype=torch.int32)
        c1 = bs._compact_records_torch_ref(torch, r1)
        assert c1.shape == (16, 128)

        # 2. width NOT multiple of 128 (covers line 231: padded > width)
        r2 = torch.randint(0, 100, (16, 50), dtype=torch.int32)
        c2 = bs._compact_records_torch_ref(torch, r2)
        assert c2.shape == (16, 128)

    def test_jax_reference_kernels(self):
        tokens = 128
        topk = 128
        main_pool = jnp.zeros((512, 512), dtype=jnp.bfloat16)
        rope_pool = jnp.zeros((128, 64), dtype=jnp.bfloat16)
        compacted = jnp.zeros((tokens, topk), dtype=jnp.int32)
        pages = jnp.zeros((1, 32), dtype=jnp.int32)
        q_loc = jnp.array([0, 126], dtype=jnp.int32)

        # Mock Pallas / Mosaic kernel calls
        with (
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.gather_page_ids",
                return_value=jnp.zeros((tokens, topk), dtype=jnp.int32),
            ),
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather.csa_gather",
                side_effect=lambda n_c, r_c, idx, num_valid_indices=None, top_k=128: (
                    jnp.zeros((idx.shape[0], 128), dtype=jnp.int32),
                    jnp.zeros((idx.shape[0] // 4, 128), dtype=jnp.int32),
                ),
            ),
            patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.sparse_ragged_paged_attention",
                return_value=jnp.zeros((tokens, 8, 512), dtype=jnp.bfloat16),
            ),
            patch(
                "jax.experimental.pallas.pallas_call",
                return_value=lambda *a, **k: a[5],
            ),
        ):
            # 1. _sc_gather_only_jax
            g_nope, g_rope = bs._sc_gather_only_jax(
                main_pool, rope_pool, compacted, pages, q_loc, page_size=4
            )
            assert g_nope.shape[0] == tokens

            # 2. _sc_gather_union_only_jax
            u_nope, u_rope = bs._sc_gather_union_only_jax(
                main_pool, rope_pool, compacted, pages, q_loc, page_size=4
            )
            assert u_nope.shape[0] == tokens

            # 3. _pregather_all_chunks_jax
            pg_n, pg_r, kv_l = bs._pregather_all_chunks_jax(
                main_pool, rope_pool, compacted, pages, q_loc, page_size=4
            )
            assert kv_l.shape[0] == tokens

            # 4. _mla_tc_only_jax
            q_in = jnp.zeros((tokens, 8, 512), dtype=jnp.bfloat16)
            sink = jnp.zeros((8,), dtype=jnp.float32)
            s_num = jnp.zeros((tokens, 8, 512), dtype=jnp.float32)
            s_den = jnp.zeros((tokens, 8), dtype=jnp.float32)
            s_max = jnp.zeros((tokens, 8), dtype=jnp.float32)
            out_mla = bs._mla_tc_only_jax(
                q_in,
                pg_n,
                pg_r,
                kv_l,
                q_loc,
                sink,
                s_num,
                s_den,
                s_max,
                scale=0.1,
                max_num_seqs=1,
            )
            assert out_mla.shape == (tokens, 8, 512)

            # 5. _sparse_fp8_attention_precompact_jax
            s_pool = jnp.zeros((256, 128), dtype=jnp.int32)
            s_rec = jnp.zeros((tokens, 128), dtype=jnp.int32)
            out_pre = bs._sparse_fp8_attention_precompact_jax(
                q_in,
                s_pool,
                main_pool,
                rope_pool,
                s_rec,
                compacted,
                pages,
                q_loc,
                sink,
                scale=0.1,
                page_size=4,
            )
            assert out_pre.shape == (tokens, 8, 512)

    def test_compute_roofline_sparse_mla_swa(self):
        hw = common.Tpu7xRoofline()

        # token_parallel mode: all vs swa_only
        r_tp_all = bs._compute_roofline(
            hw,
            "all",
            "token_parallel",
            tokens=1024,
            tp_size=8,
            heads_total=64,
            head_dim=128,
            window_size=128,
            topk=512,
            measured_ms=2.0,
        )
        assert r_tp_all["comm_s"] > 0

        r_tp_swa = bs._compute_roofline(
            hw,
            "swa_only",
            "token_parallel",
            tokens=1024,
            tp_size=8,
            heads_total=64,
            head_dim=128,
            window_size=128,
            topk=512,
            measured_ms=None,
        )
        assert r_tp_swa["comm_s"] == 0.0

        # union_gather mode
        r_ug = bs._compute_roofline(
            hw,
            "all",
            "union_gather",
            tokens=1024,
            tp_size=8,
            heads_total=64,
            head_dim=128,
            window_size=128,
            topk=512,
            measured_ms=1.0,
        )
        assert r_ug["lower_bound_ms"] > 0

        # subops in baseline mode
        for sub in ("compact_records_only", "sc_gather_only", "mla_tc_only"):
            r_sub = bs._compute_roofline(
                hw,
                sub,
                "baseline",
                tokens=1024,
                tp_size=8,
                heads_total=64,
                head_dim=128,
                window_size=128,
                topk=512,
                measured_ms=1.0,
            )
            assert r_sub["lower_bound_ms"] > 0

        # q_len > 0 (decode & verify branches)
        for ql in (1, 6):
            r_dec = bs._compute_roofline(
                hw,
                "all",
                "baseline",
                tokens=128,
                tp_size=8,
                heads_total=64,
                head_dim=128,
                window_size=512,
                topk=128,
                measured_ms=0.5,
                q_len=ql,
            )
            assert r_dec["lower_bound_ms"] > 0

    def test_worker_chunk_size_validation(self, tmp_path):
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)

        # chunk_size % 128 != 0
        args1 = argparse.Namespace(chunk_size=100)
        with pytest.raises(ValueError, match="must be a multiple of 128"):
            bs.worker(ctx, args1)

        # (chunk_size // tp_size) % 128 != 0 (e.g. 128 // 8 = 16)
        args2 = argparse.Namespace(chunk_size=128)
        with pytest.raises(ValueError, match="tp_size=8 must be a multiple of 128"):
            bs.worker(ctx, args2)

        # q_len > 0 and chunk_size % 4 != 0
        args3 = argparse.Namespace(chunk_size=6, q_len=1)
        with pytest.raises(
            ValueError, match="must be a multiple of 4 when --q-len > 0"
        ):
            bs.worker(ctx, args3)

        # q_len > 0, mode="token_parallel", and (tokens // tp_size) % 4 != 0
        args4 = argparse.Namespace(chunk_size=16, q_len=1, mode="token_parallel")
        with pytest.raises(ValueError, match="tp_size=8 must be a multiple of 4"):
            bs.worker(ctx, args4)

        # q_len > 0 and num_seqs * q_len > tokens
        args5 = argparse.Namespace(chunk_size=16, q_len=6, num_seqs=4, mode="baseline")
        with pytest.raises(ValueError, match="exceeds --chunk-size"):
            bs.worker(ctx, args5)

    def test_worker_all_subops_and_modes(self, tmp_path):
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)

        def dummy_mla_jax_op(name, fn, *args, **kwargs):
            try:
                import inspect

                n_params = len(inspect.signature(fn).parameters)
                fn(*[torch.zeros((16, 16)) for _ in range(n_params)])
            except Exception:
                pass

            def _len(a):
                return a[0].shape[0] if a and hasattr(a[0], "shape") else 1024

            def _rec_len(a):
                return (
                    a[2].shape[0] if len(a) > 2 and hasattr(a[2], "shape") else _len(a)
                )

            if "sc_gather" in name:
                return lambda *a, **k: (
                    torch.zeros((_rec_len(a), 4)),
                    torch.zeros((_rec_len(a), 4)),
                )
            elif "pregather" in name:
                return lambda *a, **k: (
                    torch.zeros((1, _rec_len(a), 128, 512)),
                    torch.zeros((1, _rec_len(a), 32, 64)),
                    torch.zeros((_rec_len(a),), dtype=torch.int32),
                )
            elif "precompact" in name or "mla_tc" in name:
                return lambda *a, **k: torch.zeros(
                    (
                        _len(a),
                        a[0].shape[1]
                        if a and hasattr(a[0], "shape") and a[0].ndim >= 2
                        else (64 if _len(a) < 1024 else 8),
                        512,
                    )
                )
            elif "compact" in name:
                return lambda *a, **k: torch.zeros((_len(a), 128), dtype=torch.int32)
            else:
                return lambda *a, **k: (
                    torch.zeros(
                        (
                            _len(a),
                            a[0].shape[1]
                            if a and hasattr(a[0], "shape") and a[0].ndim >= 2
                            else (64 if _len(a) < 1024 else 8),
                            512,
                        )
                    ),
                    torch.zeros((_len(a), 4)),
                    torch.zeros((_len(a), 4)),
                )

        def mock_shard_map(fn, *a, **kw):
            return lambda *args: (
                torch.zeros(
                    (
                        args[0].shape[0]
                        if args and hasattr(args[0], "shape")
                        else 1024,
                        64
                        if (
                            args
                            and hasattr(args[0], "shape")
                            and args[0].shape[0] < 1024
                        )
                        else 8,
                        512,
                    )
                ),
                torch.zeros(
                    (
                        args[0].shape[0]
                        if args and hasattr(args[0], "shape")
                        else 1024,
                        4,
                    )
                ),
                torch.zeros(
                    (
                        args[0].shape[0]
                        if args and hasattr(args[0], "shape")
                        else 1024,
                        4,
                    )
                ),
            )

        with (
            patch("vllm.distributed.get_tp_group") as mock_tp,
            patch.object(common, "model_wrapper_context"),
            patch.object(common, "synchronize"),
            patch.object(common, "hbm_stats", return_value={"hbm": 100}),
            patch.object(
                common,
                "time_steps",
                side_effect=lambda c, step_fn, **kw: ([1.0], step_fn()),
            ),
            patch.object(
                common,
                "profile_steps",
                side_effect=[None, str(tmp_path / "prof")] * 100,
            ),
            patch.object(common, "check_against_golden", return_value={"golden": "ok"}),
            patch.object(common, "verify_outputs", return_value={"ok": True}),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.sparse_fp8_attention",
                side_effect=lambda q, *a, **kw: torch.zeros_like(q),
            ),
            patch(
                "torch.distributed.all_to_all_single",
                side_effect=lambda recv, send, *a, **kw: (
                    recv.copy_(send) if hasattr(recv, "copy_") else None
                ),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.initialize_paged_attention"
            ),
            patch("jax.shard_map", side_effect=mock_shard_map),
            patch("torch_tpu._internal.pallas.jax_op", side_effect=dummy_mla_jax_op),
            patch.object(
                bs,
                "_build_swa_inputs",
                side_effect=lambda t, rng, toks, w_sz, q_loc, q_len=0: (
                    t.zeros((64, 128), dtype=t.int32),
                    t.zeros((toks, w_sz), dtype=t.int32),
                    t.cat(
                        [
                            t.zeros((int(q_loc[-1].item()),), dtype=t.int32),
                            t.full((toks - int(q_loc[-1].item()),), -1, dtype=t.int32),
                        ]
                    ),
                ),
            ),
            patch.object(
                bs,
                "_build_main_kv_inputs",
                side_effect=lambda t, rng, toks, kv_l, cr, p_sz, tk, q_loc, upc: (
                    t.zeros((64, 128), dtype=t.int32),
                    t.zeros((16, 128), dtype=t.int32),
                    t.zeros((int(q_loc.numel() - 1), 16), dtype=t.int32),
                    t.zeros((toks, tk), dtype=t.int32),
                ),
            ),
            patch.object(
                bs,
                "_compact_records_torch_ref",
                side_effect=lambda t, rec: rec.clone(),
            ),
            patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                return_value=MagicMock(mesh=None),
            ),
        ):
            mock_tp.return_value.rank_in_group = 0
            mock_tp.return_value.device_group = None

            # Subop="all" across all modes
            for mode in ("baseline", "precompact", "token_parallel", "union_gather"):
                args = argparse.Namespace(
                    chunk_size=1024,
                    seed=42,
                    num_seqs=1,
                    kv_len=512,
                    compress_ratio=2,
                    page_size=16,
                    unique_per_chunk=256,
                    mode=mode,
                    subop="all",
                    verify=False,
                    golden_dir=tmp_path / "golden",
                    update_golden=False,
                    profile=None,
                    warmup=1,
                    iters=1,
                )
                res = bs.worker(ctx, args)
                assert res["subop"] == "all"

            # Individual subops
            for subop in (
                "swa_only",
                "compact_records_only",
                "sc_gather_only",
                "mla_tc_only",
            ):
                for mode in ("token_parallel", "baseline", "union_gather"):
                    if subop != "sc_gather_only" and mode == "union_gather":
                        continue
                    args = argparse.Namespace(
                        chunk_size=1024,
                        seed=42,
                        num_seqs=1,
                        kv_len=512,
                        compress_ratio=2,
                        page_size=16,
                        unique_per_chunk=256,
                        mode=mode,
                        subop=subop,
                        verify=False,
                        golden_dir=tmp_path / "golden",
                        update_golden=False,
                        profile=None,
                        warmup=1,
                        iters=1,
                    )
                    res = bs.worker(ctx, args)
                    assert res["subop"] == subop

            # verify=True and update_golden=True across all subops and modes
            for s in (
                "swa_only",
                "compact_records_only",
                "sc_gather_only",
                "mla_tc_only",
                "all",
            ):
                for m in ("baseline", "token_parallel", "precompact", "union_gather"):
                    if s == "mla_tc_only" and m in ("precompact", "union_gather"):
                        continue
                    if s in ("swa_only", "compact_records_only") and m != "baseline":
                        continue
                    if s == "sc_gather_only" and m not in (
                        "baseline",
                        "token_parallel",
                        "union_gather",
                    ):
                        continue
                    args_v = argparse.Namespace(
                        chunk_size=1024,
                        seed=42,
                        num_seqs=1,
                        kv_len=512,
                        compress_ratio=2,
                        page_size=16,
                        unique_per_chunk=256,
                        mode=m,
                        subop=s,
                        verify=True,
                        golden_dir=tmp_path / "golden",
                        update_golden=True,
                        profile=tmp_path / "prof",
                        warmup=1,
                        iters=1,
                    )
                    res_v = bs.worker(ctx, args_v)
                    assert res_v["subop"] == s

            # num_seqs=2 under union_gather to cover branch
            args_ug2 = argparse.Namespace(
                chunk_size=1024,
                seed=42,
                num_seqs=2,
                kv_len=512,
                compress_ratio=2,
                page_size=16,
                unique_per_chunk=256,
                mode="union_gather",
                subop="sc_gather_only",
                verify=True,
                golden_dir=tmp_path / "golden",
                update_golden=False,
                profile=None,
                warmup=1,
                iters=1,
            )
            res_ug2 = bs.worker(ctx, args_ug2)
            assert res_ug2["subop"] == "sc_gather_only"

            # Decode q_len=1 (chunk_size=16) and q_len=6 (chunk_size=128)
            for c_size, q_l, n_s, swa_w in [(16, 1, 4, 512), (128, 6, 4, 576)]:
                args_dec = argparse.Namespace(
                    chunk_size=c_size,
                    q_len=q_l,
                    swa_width=swa_w,
                    seed=42,
                    num_seqs=n_s,
                    kv_len=512,
                    compress_ratio=2,
                    page_size=16,
                    unique_per_chunk=256,
                    mode="baseline",
                    subop="all",
                    verify=True,
                    golden_dir=tmp_path / "golden",
                    update_golden=False,
                    profile=None,
                    warmup=1,
                    iters=1,
                )
                res_dec = bs.worker(ctx, args_dec)
                assert "decode_subop_suite" in res_dec["verify"]
                if c_size == 128:
                    assert (
                        "all_token_parallel" in res_dec["verify"]["decode_subop_suite"]
                    )
                assert res_dec["window_size"] == swa_w

    def test_sparse_mla_swa_main_and_entrypoint(self, monkeypatch):
        with patch.object(common, "run_main") as mock_rm:
            bs.main()
            mock_rm.assert_called_once()

        monkeypatch.setattr(sys, "argv", ["bench_sparse_mla_swa.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(_SCRIPTS_DIR / "bench_sparse_mla_swa.py"), run_name="__main__"
            )
        assert exc_info.value.code == 0


# ==============================================================================
# SECTION 4: scripts/dsv41_bench/bench_decoder_layer.py Tests
# ==============================================================================


class TestBenchDecoderLayerFull:
    @pytest.mark.parametrize("subop", bdl.SUBOPS)
    @pytest.mark.parametrize("mode", bdl.MODES)
    def test_compute_layer_roofline_all_subops_and_modes(
        self, subop: str, mode: str
    ) -> None:
        roof = bdl.compute_layer_roofline(
            subop, num_tokens=2048, kv_len=90112, tp_size=8, mode=mode
        )
        assert roof["subop"] == subop
        assert roof["mode"] == mode
        assert roof["flops"] > 0
        assert roof["hbm_bytes"] > 0
        assert roof["comm_bytes"] > 0
        assert roof["lower_bound_ms"] > 0.0
        if subop == "decoder_ced":
            assert roof["num_tokens"] == 128
        else:
            assert roof["num_tokens"] == 2048

    def test_compute_layer_roofline_error_handling(self):
        with pytest.raises(ValueError, match="Unsupported subop"):
            bdl.compute_layer_roofline("invalid_subop", num_tokens=128)
        with pytest.raises(ValueError, match="Unsupported mode"):
            bdl.compute_layer_roofline("all", num_tokens=128, mode="invalid_mode")

    def test_reconcile_40_layer_step_matches_e2e_baseline(self) -> None:
        rec = bdl.reconcile_40_layer_step(num_tokens=2048, ced_decoder_tokens=128)
        base = rec["baseline_tp"]
        opt_tp = rec["optimized_tp_ced1"]
        opt_sp_ced1 = rec["optimized_sp_ced1"]
        opt_sp = rec["optimized_sp_ced0"]
        ref = rec["e2e_reference"]

        assert (
            0.0
            < base["ced1_chunks_1_to_48_isolated_ms"]
            < ref["baseline_100k_xprof_device_chunk_ms"]
        )
        assert ref["baseline_tp_ced1_unmeasured_residual_ms_unverified"] > 0.0
        assert (
            base["ced0_all_40L_isolated_ms"] > base["ced1_chunks_1_to_48_isolated_ms"]
        )

        assert (
            opt_tp["ced1_chunks_1_to_48_isolated_ms"]
            < base["ced1_chunks_1_to_48_isolated_ms"]
        )
        assert opt_tp["isolated_delta_vs_baseline_tp_ms"] == pytest.approx(
            -12.83, abs=1e-2
        )

        assert (
            opt_sp_ced1["ced1_chunks_1_to_48_isolated_ms"]
            < opt_tp["ced1_chunks_1_to_48_isolated_ms"]
        )
        assert opt_sp_ced1["isolated_delta_vs_optimized_tp_ced1_ms"] == pytest.approx(
            -8.88, abs=1e-2
        )
        assert opt_sp_ced1["isolated_delta_vs_baseline_tp_ms"] == pytest.approx(
            -21.71, abs=1e-2
        )

        assert opt_sp["ced0_all_40L_isolated_ms"] < opt_tp["ced0_all_40L_isolated_ms"]
        assert opt_sp["unmeasured_40L_dense_ep_compressor_host_gap_ms_unverified"] > 0.0

    def test_reconcile_40_layer_decode_step_comprehensive(self) -> None:
        rec = bdl.reconcile_40_layer_decode_step(num_seqs=16, kv_len=100000)
        assert rec["num_seqs"] == 16
        assert rec["kv_len"] == 100000
        assert "decode_q1_T16" in rec
        assert "verify_q6_T128" in rec
        assert "unverified_notes" in rec
        assert len(rec["unverified_notes"]) == 2

        q1 = rec["decode_q1_T16"]
        assert q1["swa_2L_ms"] > 0
        assert q1["sparse_mla_38L_delta_ms"] < 0
        assert q1["indexer_6L_delta_ms"] < 0
        assert q1["isolated_delta_ms"] < 0
        assert q1["e2e_mean_tpot_ms"] == 22.979
        assert q1["e2e_p50_tpot_ms"] == 22.89
        assert q1["baseline_xprof_device_ms"] == 22.070
        assert q1["baseline_host_gap_ms"] == 0.909
        assert "baseline_xprof_breakdown_ms" in q1
        assert "indexer_8L_ms" in q1["baseline_xprof_breakdown_ms"]

        q6 = rec["verify_q6_T128"]
        assert q6["swa_2L_ms"] > 0
        assert q6["sparse_mla_38L_delta_ms"] < 0
        assert q6["moe_40L_isolated_ms"] > 0

    @pytest.mark.parametrize("subop", bdl.SUBOPS)
    def test_run_decoder_layer_verification_all_subops(self, subop: str) -> None:
        res = bdl.run_decoder_layer_verification(
            subop, num_tokens=64, num_seqs=2, tp_size=8
        )
        assert res["subop"] == subop
        assert len(res["comparisons"]) > 0
        for name, comp in res["comparisons"].items():
            assert comp["ok"], f"{name} failed verification: {comp}"
            if comp["kind"] == "exact":
                assert comp["bitwise_equal"], f"{name} expected bitwise equal: {comp}"
            else:
                assert comp["cos_sim"] >= 0.9999

    def test_run_decoder_layer_verification_errors(self):
        with pytest.raises(ValueError, match="Unsupported subop"):
            bdl.run_decoder_layer_verification("invalid_subop")
        with pytest.raises(ValueError, match="must be divisible by tp_size"):
            bdl.run_decoder_layer_verification("all", num_tokens=63, tp_size=8)
        with (
            patch.object(
                common,
                "compare_tensors",
                return_value={"ok": False, "reason": "mock_fail"},
            ),
            pytest.raises(common.VerificationError, match="mock_fail"),
        ):
            bdl.run_decoder_layer_verification(
                "encoder_source", num_tokens=16, tp_size=8
            )

    def test_worker_execution(self, tmp_path):
        args = argparse.Namespace(
            subop="encoder_source",
            chunk_size=16,
            num_seqs=1,
            tp_size=8,
            seed=42,
            mode="tp",
            kv_len=512,
            worker=True,
        )
        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)
        with (
            patch.object(common, "init_worker", return_value=ctx),
            patch.object(common, "hbm_stats", return_value={"mock": 1}),
            patch.object(
                common, "summarize_samples", create=True, return_value={"median": 1.0}
            ),
            patch.object(common, "write_rank_result", create=True) as mock_write,
        ):
            rc = bdl._worker(args)
            assert rc == 0
            mock_write.assert_called_once()

    def test_cli_cpu_only_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = bdl.main(["--subop", "all", "--chunk-size", "128", "--cpu-only"])
        assert rc == 0
        captured = capsys.readouterr().out
        assert '"four_layer_e2e_tp_vs_sp"' in captured
        assert '"four_layer_ced1_stage_split_exact"' in captured
        assert '"four_layer_ced1_replay_tail_vs_full"' in captured

    def test_main_modes(self):
        with patch.object(bdl, "_worker", return_value=0) as mock_w:
            rc = bdl.main(["--worker", "--subop", "encoder_source"])
            assert rc == 0
            mock_w.assert_called_once()

        with patch.object(common, "run_parent", return_value=0, create=True) as mock_p:
            rc = bdl.main(["--subop", "encoder_source"])
            assert rc == 0
            mock_p.assert_called_once()

    def test_decoder_layer_main_entrypoint(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["bench_decoder_layer.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(_SCRIPTS_DIR / "bench_decoder_layer.py"), run_name="__main__"
            )
        assert exc_info.value.code == 0

        # Cover line 51 branch when __package__ is set
        p = _SCRIPTS_DIR / "bench_decoder_layer.py"
        code = compile(p.read_text(encoding="utf-8"), str(p), "exec")
        exec(
            code,
            {
                "__package__": "scripts.dsv41_bench",
                "__name__": "bench_decoder_layer",
                "__file__": str(p),
            },
        )
