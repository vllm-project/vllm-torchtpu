# SPDX-License-Identifier: Apache-2.0
"""Unit test suite for bench common utilities and harness selftest.

Covers all lines and branches in:
- scripts/dsv41_bench/common.py
- scripts/dsv41_bench/bench_harness_selftest.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import runpy
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "dsv41_bench"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bench_harness_selftest  # noqa: E402
import common  # noqa: E402

pytestmark = pytest.mark.cpu_test


# ==============================================================================
# 1. Tests for Dsv41Geometry
# ==============================================================================


class TestDsv41Geometry:
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
        # Test nested text_config
        cfg_nested = tmp_path / "config_nested.json"
        cfg_nested.write_text(json.dumps({"text_config": fields}), encoding="utf-8")
        geo1 = common.Dsv41Geometry.from_hf_config(cfg_nested)
        assert geo1.hidden_size == 5120

        # Test flat config
        cfg_flat = tmp_path / "config_flat.json"
        cfg_flat.write_text(json.dumps(fields), encoding="utf-8")
        geo2 = common.Dsv41Geometry.from_hf_config(cfg_flat)
        assert geo2.num_hidden_layers == 40

        # Test missing key raises KeyError
        incomplete = dict(fields)
        del incomplete["hidden_size"]
        cfg_bad = tmp_path / "config_bad.json"
        cfg_bad.write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(KeyError, match="has no text_config.hidden_size"):
            common.Dsv41Geometry.from_hf_config(cfg_bad)


# ==============================================================================
# 2. Tests for Tpu7xRoofline and RooflineEstimate
# ==============================================================================


class TestTpu7xRooflineAndEstimate:
    def test_roofline_rates_and_formulas(self):
        hw = common.Tpu7xRoofline()
        # compute_s
        s_bf16 = hw.compute_s(1e12, dtype="bf16")
        s_fp8 = hw.compute_s(1e12, dtype="fp8")
        assert s_bf16 > s_fp8 > 0.0

        # hbm_s
        assert hw.hbm_s(1e9) > 0.0

        # collectives
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

        # Zero hbm_bytes -> inf arith_intensity
        est_zero_hbm = common.RooflineEstimate(
            name="compute_only", flops=1e12, hbm_bytes=0.0
        )
        bound_zero = est_zero_hbm.bound_s(hw)
        assert math.isinf(bound_zero["arith_intensity"])

        # report with measured_ms
        rep1 = est.report(hw, measured_ms=10.0)
        assert rep1["name"] == "test_op"
        assert rep1["measured_ms"] == 10.0
        assert rep1["roofline_efficiency"] > 0.0

        # report with measured_ms is None
        rep2 = est.report(hw, measured_ms=None)
        assert "measured_ms" not in rep2

        # report with measured_ms <= 0
        rep3 = est.report(hw, measured_ms=0.0)
        assert math.isnan(rep3["roofline_efficiency"])


# ==============================================================================
# 3. Tests for _bitwise_equal, topk_overlap, and compare_tensors
# ==============================================================================


class TestNumericalVerification:
    def test_bitwise_equal_dtypes_and_shapes(self):
        # Mismatched shapes and dtypes
        t1 = torch.zeros(4, dtype=torch.float32)
        t2 = torch.zeros(5, dtype=torch.float32)
        t3 = torch.zeros(4, dtype=torch.int32)
        assert not common._bitwise_equal(t1, t2)
        assert not common._bitwise_equal(t1, t3)

        # 1-byte float (float8_e4m3fn)
        f8_1 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        f8_2 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn)
        f8_3 = torch.tensor([1.0, 3.0], dtype=torch.float8_e4m3fn)
        assert common._bitwise_equal(f8_1, f8_2)
        assert not common._bitwise_equal(f8_1, f8_3)

        # 2-byte float (bfloat16)
        bf_1 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        bf_2 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        assert common._bitwise_equal(bf_1, bf_2)

        # 4-byte float (float32) with signed zero and NaN
        f32_pos = torch.tensor([0.0, float("nan")])
        f32_neg = torch.tensor([-0.0, float("nan")])
        assert not common._bitwise_equal(f32_pos, f32_neg)
        assert common._bitwise_equal(f32_pos, f32_pos.clone())

        # 8-byte float (float64)
        f64_1 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        f64_2 = torch.tensor([1.0, 2.0], dtype=torch.float64)
        f64_3 = torch.tensor([1.0, 2.5], dtype=torch.float64)
        assert common._bitwise_equal(f64_1, f64_2)
        assert not common._bitwise_equal(f64_1, f64_3)

        # Integer tensors
        i32_1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        i32_2 = torch.tensor([1, 2, 3], dtype=torch.int32)
        i32_3 = torch.tensor([1, 2, 4], dtype=torch.int32)
        assert common._bitwise_equal(i32_1, i32_2)
        assert not common._bitwise_equal(i32_1, i32_3)

    def test_topk_overlap_validation_and_cases(self):
        # 1D or mismatched shape raises ValueError
        with pytest.raises(ValueError, match="expects equal 2-D shapes"):
            common.topk_overlap(torch.tensor([1, 2]), torch.tensor([1, 2]))
        with pytest.raises(ValueError, match="expects equal 2-D shapes"):
            common.topk_overlap(torch.zeros(2, 3), torch.zeros(2, 4))

        # Normal valid overlap
        actual = torch.tensor([[10, 20, 30], [5, 6, -1]])
        expected = torch.tensor([[20, 10, 30], [5, 7, -1]])
        mean, worst = common.topk_overlap(actual, expected)
        assert mean == pytest.approx((1.0 + 0.5) / 2.0)
        assert worst == pytest.approx(0.5)

        # Expected has row with no valid index
        actual_empty = torch.tensor([[-1, -1], [1, 2]])
        expected_empty = torch.tensor([[-1, -1], [1, 2]])
        mean, worst = common.topk_overlap(actual_empty, expected_empty)
        assert mean == 1.0 and worst == 1.0

        # Actual has valid index when expected has none
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
        # Float mismatch
        a = torch.tensor([1.0, 2.0, float("nan")])
        b = torch.tensor([1.0, 2.5, float("nan")])
        res = common.compare_tensors(a, b, common.EXACT)
        assert not res["ok"]
        assert res["num_mismatch"] == 1
        assert res["max_abs_diff"] == pytest.approx(0.5)

        # Int mismatch
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
        # Non-float tensor raises ValueError
        with pytest.raises(ValueError, match="needs a float tensor"):
            common.compare_tensors(
                torch.tensor([1, 2]), torch.tensor([1, 2]), common.LOSSLESS
            )

        # Non-finite mask mismatch
        a_inf = torch.tensor([1.0, float("inf")])
        b_fin = torch.tensor([1.0, 2.0])
        res_mask = common.compare_tensors(a_inf, b_fin, common.LOSSLESS)
        assert not res_mask["ok"]
        assert not res_mask["nonfinite_mask_equal"]

        # NaN present in expected or actual
        a_nan = torch.tensor([1.0, float("nan")])
        b_nan = torch.tensor([1.0, float("nan")])
        res_nan = common.compare_tensors(a_nan, b_nan, common.LOSSLESS)
        assert not res_nan["ok"]
        assert "NaN present" in res_nan["reason"]

        # No finite elements to compare
        a_all_inf = torch.tensor([float("inf"), float("-inf")])
        b_all_inf = torch.tensor([float("inf"), float("-inf")])
        res_empty = common.compare_tensors(a_all_inf, b_all_inf, common.LOSSLESS)
        assert not res_empty["ok"]
        assert "no finite elements" in res_empty["reason"]

        # Zero norm branches
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

        # Normal close vs far
        x = torch.randn(100)
        close = x + 1e-6 * torch.randn(100)
        far = x + 0.5 * torch.randn(100)
        assert common.compare_tensors(close, x, common.LOSSLESS)["ok"]
        assert not common.compare_tensors(far, x, common.LOSSLESS)["ok"]


# ==============================================================================
# 4. Tests for verify_outputs and Golden file lifecycle
# ==============================================================================


class TestVerifyOutputsAndGolden:
    def test_verify_outputs_key_mismatch_and_criteria_dict(self):
        actual = {"out1": torch.ones(2), "out2": torch.zeros(2)}
        expected_diff_keys = {"out1": torch.ones(2)}
        with pytest.raises(common.VerificationError, match="output names differ"):
            common.verify_outputs(actual, expected_diff_keys, common.EXACT)

        # Using Mapping of criteria
        crit_map = {"out1": common.EXACT, "out2": common.LOSSLESS}
        rep = common.verify_outputs(actual, actual, crit_map)
        assert "out1" in rep and "out2" in rep

        # Failure raises VerificationError
        bad_actual = {"out1": torch.ones(2), "out2": torch.ones(2)}
        with pytest.raises(common.VerificationError, match="verification failed"):
            common.verify_outputs(bad_actual, actual, crit_map)

    def test_golden_file_save_load_and_not_found(self, tmp_path):
        p = common.golden_path(tmp_path, "case1", 0)
        assert p == tmp_path / "case1" / "rank0.pt"

        # FileNotFoundError on missing golden
        with pytest.raises(FileNotFoundError, match="no golden at"):
            common.load_golden(p)

        # Save and load
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

        # update_golden mode
        args_update = argparse.Namespace(
            golden_dir=tmp_path, update_golden=True, verify=False
        )
        with patch("sys.argv", ["script.py"]):
            res_up = common.check_against_golden(
                ctx, args_update, "c1", tensors, common.EXACT
            )
        assert res_up["golden"] == "written"

        # verify=False mode -> skipped
        args_skip = argparse.Namespace(
            golden_dir=tmp_path, update_golden=False, verify=False
        )
        res_skip = common.check_against_golden(
            ctx, args_skip, "c1", tensors, common.EXACT
        )
        assert res_skip["golden"] == "skipped"

        # verify=True mode -> verified
        args_verify = argparse.Namespace(
            golden_dir=tmp_path, update_golden=False, verify=True
        )
        res_ver = common.check_against_golden(
            ctx, args_verify, "c1", tensors, common.EXACT
        )
        assert res_ver["golden"] == "verified"


# ==============================================================================
# 5. Tests for git_sha, CLI args, result_dir, and _llo_env
# ==============================================================================


class TestCLIAndGitSha:
    def test_git_sha_with_and_without_git(self, tmp_path, monkeypatch):
        # Case 1: REPO_ROOT/.git exists
        monkeypatch.setattr(common, "REPO_ROOT", tmp_path)
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout="123456789\n"),
                Mock(stdout=""),  # clean
            ]
            assert common.git_sha() == "123456789"

            mock_run.side_effect = [
                Mock(stdout="123456789\n"),
                Mock(stdout="M src/foo.py\n"),  # dirty
            ]
            assert common.git_sha() == "123456789+dirty"

        # Case 2: No .git, with GIT_SHA_FILE
        (tmp_path / ".git").rmdir()
        sha_file = tmp_path / "git_sha"
        monkeypatch.setattr(common, "GIT_SHA_FILE", sha_file)
        sha_file.write_text("abcdef123\n", encoding="utf-8")
        assert common.git_sha() == "abcdef123"

        # Case 3: No .git and no GIT_SHA_FILE
        sha_file.unlink()
        with pytest.raises(FileNotFoundError, match="is not a git checkout"):
            common.git_sha()

    def test_add_common_args_and_resolve_result_dir(self, tmp_path):
        parser = argparse.ArgumentParser()
        common.add_common_args(parser, ("sub1", "sub2"))
        args = parser.parse_args(["--subop", "sub2", "--chunk-size", "4096"])
        assert args.subop == "sub2"
        assert args.chunk_size == 4096

        # resolve_result_dir with explicit dir
        args.result_dir = tmp_path / "custom"
        expected_dir = (tmp_path / "custom").resolve()
        assert common.resolve_result_dir(args, "bench1") == expected_dir

        # resolve_result_dir with None
        args.result_dir = None
        res_dir = common.resolve_result_dir(args, "bench1")
        assert "bench1_sub2_" in str(res_dir)

    def test_llo_env(self, tmp_path):
        env: dict[str, str] = {}
        common._llo_env(tmp_path, 0, env)
        assert "--xla_dump_to=" in env["XLA_FLAGS"]
        assert "--xla_jf_dump_to=" in env["LIBTPU_INIT_ARGS"]
        assert env["TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE"] == "disabled"


# ==============================================================================
# 6. Tests for prepare_worker_env, launch, aggregate, and run_main
# ==============================================================================


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
        # Normal launch
        mock_proc = MagicMock()
        mock_proc.stdout = ["[rank0] running step\n", "normal line\n", "some error\n"]
        mock_proc.wait.return_value = 0
        with (
            patch("common.prepare_worker_env", return_value={"mock": "1"}),
            patch("subprocess.Popen", return_value=mock_proc),
        ):
            rc = common.launch("dummy.py", ["--subop", "all"], tmp_path, args)
            assert rc == 0
            log = (tmp_path / "workers.log").read_text(encoding="utf-8")
            assert "[rank0] running step" in log

        # Timeout launch
        with (
            patch("common.prepare_worker_env", return_value={"mock": "1"}),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("time.monotonic", side_effect=[0, 1e9, 1e9]),
            patch("os.killpg") as mock_kill,
        ):
            with pytest.raises(TimeoutError, match="workers exceeded"):
                common.launch("dummy.py", [], tmp_path, args)
            mock_kill.assert_called_once()

    def test_aggregate_summary(self, tmp_path):
        # Create rank files for 8 ranks
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

        with patch("common.summarize_xplanes", return_value={"planes": {}}):
            summary = common.aggregate(tmp_path)
            assert len(summary["wall_step_max_over_ranks_ms"]) == 2
            assert summary["wall_min_ms"] > 0
            assert "rank0" in summary
            assert "xprof" in summary

        # Missing rank file
        (tmp_path / "rank_3.json").unlink()
        summary_err = common.aggregate(tmp_path)
        assert len(summary_err["errors"]) == 1
        assert summary_err["errors"][0]["rank"] == 3

        # Results with error in rank0 -> branch 649->653
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

        # Test worker branch
        with patch("common._worker_entry") as mock_entry:
            monkeypatch.setattr(sys, "argv", ["bench.py", "--worker"])
            common.run_main("test_bench", parser, lambda ctx, args: {})
            mock_entry.assert_called_once()

        # Test parent branch success
        mock_agg = {
            "result_dir": str(tmp_path),
            "errors": [],
            "xprof": {"planes": {}},
        }
        with (
            patch("common.launch", return_value=0),
            patch("common.aggregate", return_value=mock_agg),
            patch("common.git_sha", return_value="abc1234"),
            patch("common.print_xprof_table") as mock_table,
        ):
            monkeypatch.setattr(
                sys, "argv", ["bench.py", "--result-dir", str(tmp_path)]
            )
            common.run_main("test_bench", parser, lambda ctx, args: {})
            mock_table.assert_called_once()
            assert (tmp_path / "summary.json").exists()

        # Test parent branch failure raises SystemExit
        with (
            patch("common.launch", return_value=1),
            patch(
                "common.aggregate",
                return_value={"result_dir": str(tmp_path), "errors": ["some_error"]},
            ),
            patch("common.git_sha", return_value="abc1234"),
        ):
            monkeypatch.setattr(
                sys, "argv", ["bench.py", "--result-dir", str(tmp_path)]
            )
            with pytest.raises(SystemExit):
                common.run_main("test_bench", parser, lambda ctx, args: {})


# ==============================================================================
# 7. Tests for Worker Side Functions
# ==============================================================================


class TestWorkerSide:
    def test_worker_entry_success_and_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        args = argparse.Namespace(
            result_dir=tmp_path, dump_llo=tmp_path / "llo", tp_size=1
        )

        # Worker success
        with (
            patch("common.init_worker") as mock_init,
            patch("torch.distributed.destroy_process_group") as mock_destroy,
        ):
            ctx = common.WorkerContext(0, 1, 1, None, tmp_path)
            mock_init.return_value = ctx
            common._worker_entry(args, lambda c, a: {"data": 42})
            mock_destroy.assert_called_once()
            res = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
            assert res["data"] == 42

            # dump_llo is None -> branch 706->708
            args_no_llo = argparse.Namespace(
                result_dir=tmp_path, dump_llo=None, tp_size=1
            )
            common._worker_entry(args_no_llo, lambda c, a: {"data": 43})
            res2 = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
            assert res2["data"] == 43

        # Worker error
        with (
            patch("common.init_worker", side_effect=RuntimeError("boom")),
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

        # World size not divisible raises ValueError
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

        with patch("common.synchronize"), patch("torch.distributed.barrier"):
            samples, out = common.time_steps(ctx, step, warmup=2, iters=3)
            assert len(samples) == 3
            assert call_count == 5

    def test_profile_steps(self, tmp_path, monkeypatch):
        ctx = common.WorkerContext(0, 1, 1, None, tmp_path)

        def step():
            return torch.zeros(1)

        # args.profile is None
        args_none = argparse.Namespace(profile=None)
        assert common.profile_steps(ctx, args_none, step, label="test") is None

        # args.profile is set
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
            patch("common.synchronize"),
        ):
            mock_prof_ctx = MagicMock()
            mock_prof.return_value = mock_prof_ctx
            prof_dir = common.profile_steps(ctx, args_prof, step, label="test")
            assert prof_dir == str((tmp_path / "prof").resolve())
            mock_merge.assert_called_once()
            mock_clear.assert_called_once()

            # Test trace_mode != TRACE_COMPUTE and rank != 0
            # (branches 870->874, 901->903)
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
        from types import SimpleNamespace

        # Create a mock xplane file
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
            # print_xprof_table normal
            common.print_xprof_table(summary)

        # print_xprof_table where all planes are sparse_core
        # (covers line 978 and branch 976->exit)
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
# 8. Tests for scripts/dsv41_bench/bench_harness_selftest.py
# ==============================================================================


class TestBenchHarnessSelftest:
    @pytest.fixture(autouse=True)
    def patch_tpu_device(self):
        orig_to = torch.Tensor.to

        def mock_to(self, *args, **kwargs):
            if args and args[0] == "tpu":
                return self
            if "device" in kwargs and kwargs["device"] == "tpu":
                kwargs["device"] = "cpu"
            return orig_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", mock_to):
            yield

    def test_rank_input(self):
        t = bench_harness_selftest._rank_input(torch, rank=1, tokens=8, hidden=16)
        assert t.shape == (8, 16)
        assert t.dtype == torch.bfloat16
        assert t[0, 0] == float((0 + 0 + 1) % 7)

    def test_reference_all_subops(self):
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

    def test_worker_all_reduce_subop(self, tmp_path):
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
            patch("common.synchronize"),
            patch("common.hbm_stats", return_value={"hbm": 1}),
            patch("common.time_steps", return_value=([1.0], None)),
            patch("common.profile_steps", return_value=None),
            patch("common.check_against_golden", return_value={"golden": "skipped"}),
            patch("common.verify_outputs", return_value={"ok": True}),
        ):
            mock_tp.return_value.rank_in_group = 0
            res = bench_harness_selftest.worker(ctx, args)
            assert "roofline_vs_wall" in res
            assert res["roofline_vs_wall"]["x41_per_step_lower_bound_ms"] > 0

    def test_worker_reduce_scatter_and_all_gather(self, tmp_path):
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
            patch("common.synchronize"),
            patch("common.hbm_stats", return_value={}),
            patch("common.time_steps", return_value=([1.0], None)),
            patch("common.profile_steps", return_value=None),
            patch("common.check_against_golden", return_value={}),
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
            patch("common.synchronize"),
            patch("common.hbm_stats", return_value={}),
            patch("common.time_steps", return_value=([1.0], None)),
            patch("common.profile_steps", return_value=None),
            patch("common.check_against_golden", return_value={}),
        ):
            mock_tp.return_value.rank_in_group = 0
            res_ag = bench_harness_selftest.worker(ctx, args_ag)
            assert "roofline_vs_wall" in res_ag

    def test_worker_chunk_size_validation(self, tmp_path):
        ctx = common.WorkerContext(0, 8, 8, None, tmp_path)
        args = argparse.Namespace(chunk_size=15, subop="all_reduce")
        with pytest.raises(ValueError, match="must divide by tp"):
            bench_harness_selftest.worker(ctx, args)

    def test_main_and_entrypoint_via_runpy(self):
        with patch("common.run_main") as mock_run_main:
            bench_harness_selftest.main()
            mock_run_main.assert_called_once()

        script_path = str(_SCRIPTS_DIR / "bench_harness_selftest.py")
        with patch("common.run_main"):
            res = runpy.run_path(script_path, run_name="__main__")
            assert "main" in res
