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
"""Comprehensive unit test suite for DeepSeek-V4.1 .

Achieves 100% full-module statement and branch coverage across:
1. scripts/vllm/benchmarking/bench_decode_sliding_window.py
2. smoke_e2e_dsv41_tp8.py
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import math
import pathlib
import runpy
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import requests
import torch

pytestmark = pytest.mark.cpu_test

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT_SMOKE = _REPO_ROOT / "smoke_e2e_dsv41_tp8.py"
_SCRIPT_BENCH = (
    _REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "bench_decode_sliding_window.py"
)

# Load smoke_e2e_dsv41_tp8 module
if "smoke_e2e_dsv41_tp8" in sys.modules:
    smoke = sys.modules["smoke_e2e_dsv41_tp8"]
else:
    _spec_smoke = importlib.util.spec_from_file_location(
        "smoke_e2e_dsv41_tp8", _SCRIPT_SMOKE
    )
    smoke = importlib.util.module_from_spec(_spec_smoke)
    sys.modules["smoke_e2e_dsv41_tp8"] = smoke
    _spec_smoke.loader.exec_module(smoke)

# Load bench_decode_sliding_window module
if "bench_decode_sliding_window" in sys.modules:
    bench = sys.modules["bench_decode_sliding_window"]
else:
    _spec_bench = importlib.util.spec_from_file_location(
        "bench_decode_sliding_window", _SCRIPT_BENCH
    )
    bench = importlib.util.module_from_spec(_spec_bench)
    sys.modules["bench_decode_sliding_window"] = bench
    _spec_bench.loader.exec_module(bench)


def _write_fake_meta_dir(meta_dir: pathlib.Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "text_config": {
            "engram_max_ngram_size": 4,
            "engram_n_heads": 8,
            "num_hidden_layers": 40,
        }
    }
    (meta_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (meta_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (meta_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (meta_dir / "chat_template.jinja").write_text("", encoding="utf-8")


# ==============================================================================
# Class 1: Comprehensive Unit Tests for smoke_e2e_dsv41_tp8.py
# ==============================================================================


class TestSmokeE2EDsv41:
    """Covers 100% lines & branches of smoke_e2e_dsv41_tp8.py."""

    def test_sys_path_insert_branch(self):
        repo_root_str = str(_SCRIPT_SMOKE.resolve().parent)
        orig_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if p != repo_root_str]
            assert repo_root_str not in sys.path
            runpy.run_path(str(_SCRIPT_SMOKE), run_name="smoke_test_sys_path")
            assert sys.path[0] == repo_root_str
        finally:
            sys.path = orig_path

    def test_deterministic_dummy_weights_all_types(self):
        import vllm.model_executor.model_loader.weight_utils as weight_utils

        smoke._install_deterministic_dummy_weights()

        # 1. Meta tensor -> immediate return
        meta_param = torch.nn.Parameter(
            torch.empty((4, 4), device="meta", dtype=torch.float32)
        )
        weight_utils.initialize_single_dummy_weight(meta_param)

        # 2. Explicit dummy_weight_value
        val_param = torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32))
        val_param.dummy_weight_value = 0.5
        weight_utils.initialize_single_dummy_weight(val_param)
        assert torch.all(val_param == 0.5)

        # 3. uint8 with shape[-1] <= 160 (scale: fills 120)
        u8_scale = torch.nn.Parameter(
            torch.empty((2, 128), dtype=torch.uint8), requires_grad=False
        )
        weight_utils.initialize_single_dummy_weight(u8_scale)
        assert torch.all(u8_scale == 120)

        # 4. uint8 with shape[-1] > 160 (weight: fills 0x21)
        u8_weight = torch.nn.Parameter(
            torch.empty((2, 512), dtype=torch.uint8), requires_grad=False
        )
        weight_utils.initialize_single_dummy_weight(u8_weight)
        assert torch.all(u8_weight == 0x21)

        # 5. uint8 with ndim < 2 (fills 0x21)
        u8_1d = torch.nn.Parameter(
            torch.empty((100,), dtype=torch.uint8), requires_grad=False
        )
        weight_utils.initialize_single_dummy_weight(u8_1d)
        assert torch.all(u8_1d == 0x21)

        # 6. Non-floating non-uint8 (e.g. int32: zero_())
        i32_param = torch.nn.Parameter(
            torch.empty((4, 4), dtype=torch.int32), requires_grad=False
        )
        weight_utils.initialize_single_dummy_weight(i32_param)
        assert torch.all(i32_param == 0)

        # 7. Floating point parameter (bfloat16 uniform_)
        bf16_param = torch.nn.Parameter(torch.empty((8, 16), dtype=torch.bfloat16))
        weight_utils.initialize_single_dummy_weight(bf16_param)
        assert torch.isfinite(bf16_param).all()

    def test_logits_capture_all_branches(self, tmp_path, monkeypatch):
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        # 1. No DSV41_SMOKE_LOGITS_DIR -> early return
        monkeypatch.delenv("DSV41_SMOKE_LOGITS_DIR", raising=False)
        smoke._install_logits_capture()

        # 2. Reset hooked flag and mock original runner methods
        logits_dir = tmp_path / "logits_test"
        monkeypatch.setenv("DSV41_SMOKE_LOGITS_DIR", str(logits_dir))
        monkeypatch.setattr(
            TPUModelRunner, "_dsv41_smoke_logits_hooked", False, raising=False
        )

        orig_from_hidden = mock.MagicMock(
            return_value=torch.ones((1, 16), dtype=torch.float32)
        )
        orig_selected = mock.MagicMock(
            return_value=torch.ones((1, 16), dtype=torch.float32)
        )
        monkeypatch.setattr(
            TPUModelRunner, "compute_logits_from_hidden_states", orig_from_hidden
        )
        monkeypatch.setattr(TPUModelRunner, "compute_selected_logits", orig_selected)

        smoke._install_logits_capture()
        assert getattr(TPUModelRunner, "_dsv41_smoke_logits_hooked", False)

        # 3. Subsequent call returns immediately
        smoke._install_logits_capture()

        # 4. Exercise wrapped methods and _save_logits
        runner = TPUModelRunner.__new__(TPUModelRunner)
        fake_hidden = torch.zeros((1, 10))

        # Test rank 0 valid logits
        monkeypatch.setenv("RANK", "0")
        TPUModelRunner.compute_logits_from_hidden_states(runner, fake_hidden)
        TPUModelRunner.compute_selected_logits(runner, fake_hidden, [0])

        assert (logits_dir / "steps" / "step_0000.pt").exists()
        assert (logits_dir / "steps" / "step_0001.pt").exists()

        # Test rank != 0 (e.g. 1) -> does not save
        monkeypatch.setenv("RANK", "1")
        TPUModelRunner.compute_logits_from_hidden_states(runner, fake_hidden)
        assert not (logits_dir / "steps" / "step_0002.pt").exists()

        # Test rank 0 with logits is None or numel == 0
        monkeypatch.setenv("RANK", "0")
        orig_from_hidden.return_value = None
        TPUModelRunner.compute_logits_from_hidden_states(runner, fake_hidden)
        orig_from_hidden.return_value = torch.empty((0, 0))
        TPUModelRunner.compute_logits_from_hidden_states(runner, fake_hidden)
        assert not (logits_dir / "steps" / "step_0002.pt").exists()

    def test_active_env_and_warning_branch(self, monkeypatch, capsys):
        # 1. Success path inside smoke_e2e_dsv41_tp8.py
        monkeypatch.setenv("DSV41_SMOKE_ACTIVE", "1")
        monkeypatch.delenv("DSV41_SMOKE_LOGITS_DIR", raising=False)
        runpy.run_path(str(_SCRIPT_SMOKE), run_name="smoke_active_ok")

        # 2. Exception path inside smoke_e2e_dsv41_tp8.py
        with mock.patch.dict(
            sys.modules, {"vllm.model_executor.model_loader.dummy_loader": None}
        ):
            runpy.run_path(str(_SCRIPT_SMOKE), run_name="smoke_active_err")

        captured = capsys.readouterr()
        assert "[smoke_e2e] warning:" in captured.err

    def test_build_smoke_model_dir_all_layers_and_files(self, tmp_path):
        meta_dir = tmp_path / "meta"
        _write_fake_meta_dir(meta_dir)

        out_dir = tmp_path / "smoke_out"
        # 1. First build creates out_dir
        smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=4)
        assert (out_dir / "config.json").exists()
        assert (out_dir / "tokenizer.json").exists()
        assert (out_dir / "chat_template.jinja").exists()

        # 2. Second build exercises out_dir.exists() shutil.rmtree branch
        smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=6)
        cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))[
            "text_config"
        ]
        assert cfg["num_hidden_layers"] == 6

        # 3. Missing tokenizer files branch (src.exists() == False)
        sparse_meta = tmp_path / "sparse_meta"
        sparse_meta.mkdir()
        (sparse_meta / "config.json").write_text(
            json.dumps(
                {
                    "text_config": {
                        "engram_max_ngram_size": 4,
                        "engram_n_heads": 8,
                        "num_hidden_layers": 40,
                    }
                }
            ),
            encoding="utf-8",
        )
        # Only tokenizer.json exists, other two don't
        (sparse_meta / "tokenizer.json").write_text("{}", encoding="utf-8")
        smoke.build_smoke_model_dir(
            sparse_meta, tmp_path / "smoke_sparse", num_layers=4
        )

        # 4. Invalid num_layers raises ValueError
        with pytest.raises(ValueError, match="Unsupported --num-layers=8"):
            smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=8)

    def test_extract_top_logprobs_all_branches(self):
        # 1. Empty or None step_logprobs
        empty_ids, empty_vals = smoke._extract_top_logprobs(None)
        assert empty_ids.shape == (0, 0)
        assert empty_vals.shape == (0, 0)

        empty_ids2, empty_vals2 = smoke._extract_top_logprobs([])
        assert empty_ids2.shape == (0, 0)
        assert empty_vals2.shape == (0, 0)

        # 2. Mixed normal and NaN logprobs
        step_lps = [
            {
                1: SimpleNamespace(logprob=-0.5),
                2: SimpleNamespace(logprob=-1.2),
                3: SimpleNamespace(logprob=float("nan")),
            }
        ]
        ids, vals = smoke._extract_top_logprobs(step_lps, top_k=2)
        assert ids.tolist() == [[1, 2]]
        assert math.isclose(vals[0, 0].item(), -0.5, rel_tol=1e-5)

    def test_load_captured_step_logits_all_branches(self, tmp_path):
        # 1. Non-existent steps_dir
        missing_dir = tmp_path / "does_not_exist"
        res = smoke._load_captured_step_logits(missing_dir)
        assert res.shape == (0, 0)

        # 2. Exists but empty
        empty_dir = tmp_path / "empty_steps"
        empty_dir.mkdir()
        res2 = smoke._load_captured_step_logits(empty_dir)
        assert res2.shape == (0, 0)

        # 3. Contains 1D tensor (ndim != 2), all-zero row, valid rows, max_steps cutoff
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir()
        # Invalid ndim
        torch.save(torch.randn(10), steps_dir / "step_0000.pt")
        # Empty row (shape[0] < 1)
        torch.save(torch.empty((0, 10)), steps_dir / "step_0001.pt")
        # All zeros (CED=1 partial prefill chunk)
        torch.save(torch.zeros((1, 10)), steps_dir / "step_0002.pt")
        # Valid row 1
        torch.save(torch.ones((1, 10)), steps_dir / "step_0003.pt")
        # Valid row 2
        torch.save(torch.full((1, 10), 2.0), steps_dir / "step_0004.pt")

        # Load with max_steps=1
        loaded = smoke._load_captured_step_logits(steps_dir, max_steps=1)
        assert loaded.shape == (1, 10)
        assert torch.all(loaded == 1.0)

        # Load all
        loaded_all = smoke._load_captured_step_logits(steps_dir)
        assert loaded_all.shape == (2, 10)
        assert torch.all(loaded_all[1] == 2.0)

    def test_verify_golden_all_branches(self, tmp_path):
        from scripts.dsv41_bench import common

        golden_path = tmp_path / "golden.pt"

        # 1. Non-existent golden_path -> FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Golden file not found"):
            smoke.verify_golden({}, tmp_path / "not_found.pt")

        # Setup baseline golden payload
        torch.manual_seed(42)
        logits = torch.randn(2, 64)
        golden_payload = {
            "token_ids": torch.tensor([5, 12], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[5, 1], [12, 2]], dtype=torch.int64),
            "top_logprob_values": torch.tensor(
                [[-0.1, -1.0], [-0.2, -1.5]], dtype=torch.float32
            ),
            "step_logits": logits,
        }
        torch.save(golden_payload, golden_path)

        # 2. Perfect match
        report = smoke.verify_golden(golden_payload, golden_path)
        assert report["token_ids"]["ok"]
        assert report["step_logits"]["ok"]

        # 3. Missing step_logits in actual or expected
        no_logits_actual = {
            "token_ids": golden_payload["token_ids"],
            "top_logprob_ids": golden_payload["top_logprob_ids"],
            "top_logprob_values": golden_payload["top_logprob_values"],
        }
        report_no_logits = smoke.verify_golden(no_logits_actual, golden_path)
        assert "step_logits" not in report_no_logits

        # 4. Empty step_logits in actual
        empty_logits_actual = {
            **golden_payload,
            "step_logits": torch.empty((0, 0)),
        }
        with pytest.raises(common.VerificationError, match="step_logits is empty"):
            smoke.verify_golden(empty_logits_actual, golden_path)

        # 5. top1_match_rate < 1.0
        bad_top1_logits = logits.clone()
        bad_top1_logits[0, 60] = 50.0  # alters argmax
        bad_top1_actual = {**golden_payload, "step_logits": bad_top1_logits}
        with pytest.raises(common.VerificationError, match="top1_match_rate"):
            smoke.verify_golden(bad_top1_actual, golden_path)

        # 6. Tied argmax match branch
        # Construct logits with identical max at indices 5 and 6
        tied_logits = torch.zeros((1, 32), dtype=torch.float32)
        tied_logits[0, 5] = 2.0
        tied_logits[0, 6] = 2.0
        tied_golden_path = tmp_path / "tied_golden.pt"
        torch.save(
            {
                "token_ids": torch.tensor([5], dtype=torch.int64),
                "top_logprob_ids": torch.tensor([[5, 6]], dtype=torch.int64),
                "top_logprob_values": torch.tensor(
                    [[-11.7697, -11.7697]], dtype=torch.float32
                ),
                "step_logits": tied_logits,
            },
            tied_golden_path,
        )

        # Device chose tied index 6 instead of 5, and TP shifted logprobs
        tied_actual = {
            "token_ids": torch.tensor([6], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[6, 5]], dtype=torch.int64),
            "top_logprob_values": torch.tensor(
                [[-13.1644, -13.1644]], dtype=torch.float32
            ),
            "step_logits": tied_logits.clone(),
        }
        tied_rep = smoke.verify_golden(tied_actual, tied_golden_path)
        assert tied_rep["token_ids"]["ok"]
        assert tied_rep["token_ids"]["tied_argmax_match"]
        assert tied_rep["top_logprob_values"]["ok"]
        assert tied_rep["top_logprob_values"]["shift_invariant_match"]

        # Tied argmax failure: token_id is out of range or not tied max
        bad_tied_actual = {
            **tied_actual,
            "token_ids": torch.tensor([999], dtype=torch.int64),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(bad_tied_actual, tied_golden_path)

        # Tied argmax failure: act_ids shape mismatch
        bad_shape_tied = {
            **tied_actual,
            "token_ids": torch.tensor([6, 5], dtype=torch.int64),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(bad_shape_tied, tied_golden_path)

        # Shift-invariant failure: centered diff > 1e-3
        # (cos_sim >= 0.999, covers line 363->370)
        bad_shift_actual = {
            **tied_actual,
            "top_logprob_values": torch.tensor(
                [[-13.1644, -13.1744]], dtype=torch.float32
            ),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(bad_shift_actual, tied_golden_path)

        # Shift-invariant failure: shape mismatch
        bad_shape_shift = {
            **tied_actual,
            "top_logprob_values": torch.ones((1, 1), dtype=torch.float32),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(bad_shape_shift, tied_golden_path)

        # Shift-invariant failure: empty
        bad_empty_shift = {
            **tied_actual,
            "top_logprob_values": torch.empty((0, 0), dtype=torch.float32),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(bad_empty_shift, tied_golden_path)

        # Shift-invariant failure: cos_sim < 0.999
        anti_sim_shift = {
            **tied_actual,
            "top_logprob_values": torch.tensor(
                [[13.1644, -13.1644]], dtype=torch.float32
            ),
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(anti_sim_shift, tied_golden_path)

        # 8. top1 matches 100% but step_logits is not ok (covers line 330->370)
        exp_logits = torch.zeros((1, 64), dtype=torch.float32)
        exp_logits[0, 10] = 10.0
        act_logits_l2 = exp_logits.clone()
        act_logits_l2[0, 20] = 5.0
        l2_golden_path = tmp_path / "l2_golden.pt"
        torch.save(
            {
                "token_ids": torch.tensor([10], dtype=torch.int64),
                "top_logprob_ids": torch.tensor([[10, 20]], dtype=torch.int64),
                "top_logprob_values": torch.tensor([[-0.1, -1.0]], dtype=torch.float32),
                "step_logits": exp_logits,
            },
            l2_golden_path,
        )
        l2_actual = {
            "token_ids": torch.tensor([10], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[10, 20]], dtype=torch.int64),
            "top_logprob_values": torch.tensor([[-0.1, -1.0]], dtype=torch.float32),
            "step_logits": act_logits_l2,
        }
        with pytest.raises(common.VerificationError):
            smoke.verify_golden(l2_actual, l2_golden_path)

        # 9. cos_sim >= 0.999 but shape mismatch or numel == 0 (covers line 353->370)
        with mock.patch.object(
            common,
            "compare_tensors",
            side_effect=[
                {"ok": True},
                {"ok": False, "cos_sim": 1.0},
                {"ok": True, "top1_match_rate": 1.0},
            ],
        ):
            bad_shape_actual = {
                **tied_actual,
                "top_logprob_values": torch.ones((2, 5), dtype=torch.float32),
            }
            with pytest.raises(common.VerificationError):
                smoke.verify_golden(bad_shape_actual, tied_golden_path)

        with mock.patch.object(
            common,
            "compare_tensors",
            side_effect=[
                {"ok": True},
                {"ok": False, "cos_sim": 1.0},
                {"ok": True, "top1_match_rate": 1.0},
            ],
        ):
            bad_zero_actual = {
                **tied_actual,
                "top_logprob_values": torch.empty((0, 0), dtype=torch.float32),
            }
            with pytest.raises(common.VerificationError):
                smoke.verify_golden(bad_zero_actual, tied_golden_path)

    def test_build_operator_shape_ledger_branches(self):
        # 1. dec_tokens > 16 (default)
        ledger_large = smoke.build_operator_shape_ledger(
            num_layers=4, batch_tokens=2048, kv_len=90112, ced_window=128
        )
        assert ledger_large["decoder_tokens_padded"] == 2048

        # 2. dec_tokens <= 16
        ledger_small = smoke.build_operator_shape_ledger(
            num_layers=4, batch_tokens=8, kv_len=1024, ced_window=8
        )
        assert ledger_small["decoder_tokens_padded"] == 16

    def test_parse_args(self, monkeypatch):
        test_argv = [
            "smoke_e2e_dsv41_tp8.py",
            "--num-layers",
            "6",
            "--max-tokens",
            "3",
            "--prompt-tokens",
            "1024",
        ]
        monkeypatch.setattr(sys, "argv", test_argv)
        args = smoke.parse_args()
        assert args.num_layers == 6
        assert args.max_tokens == 3
        assert args.prompt_tokens == 1024

    def test_main_and_entrypoint(self, tmp_path, monkeypatch):
        meta_dir = tmp_path / "meta"
        _write_fake_meta_dir(meta_dir)
        smoke_dir = tmp_path / "smoke_run"
        prof_dir = tmp_path / "profile"
        golden_file = tmp_path / "golden.pt"

        # Mock LLM and SamplingParams
        class FakeOutput:
            def __init__(self):
                self.token_ids = [42]
                self.logprobs = [{42: SimpleNamespace(logprob=-0.2)}]

        class FakeOutputs:
            def __init__(self):
                self.outputs = [FakeOutput()]

        class FakeLLM:
            _created = False

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                # Create warmup steps_dir on first run so line 558 executes.
                # Skip on second run so False branch 557->560 is also covered.
                if not FakeLLM._created:
                    FakeLLM._created = True
                    warmup_dir = (
                        pathlib.Path(kwargs["model"]) / "logits_capture" / "steps"
                    )
                    warmup_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        torch.zeros((1, 32), dtype=torch.float32),
                        warmup_dir / "warmup.pt",
                    )

            def start_profile(self):
                pass

            def stop_profile(self):
                pass

            def generate(self, prompts, sampling_params):
                # Write fake captured step logits to steps_dir
                steps_dir = smoke_dir / "logits_capture" / "steps"
                steps_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    torch.ones((1, 32), dtype=torch.float32),
                    steps_dir / "step_0000.pt",
                )
                return [FakeOutputs()]

        # Patch vllm components
        import vllm

        monkeypatch.setattr(vllm, "LLM", FakeLLM)
        monkeypatch.setattr(
            vllm,
            "SamplingParams",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        monkeypatch.setattr(
            vllm,
            "TokensPrompt",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )

        # Mock xprof helpers
        from scripts.dsv41_bench import common

        monkeypatch.setattr(
            common, "summarize_xplanes", lambda p: {"total_events": 100}
        )
        monkeypatch.setattr(common, "print_xprof_table", lambda x: None)

        # Run 1: main with profile_dir, save_golden, and verify_against_baseline
        # Save a reference golden file first
        torch.save(
            {
                "token_ids": torch.tensor([42], dtype=torch.int64),
                "top_logprob_ids": torch.tensor([[42]], dtype=torch.int64),
                "top_logprob_values": torch.tensor([[-0.2]], dtype=torch.float32),
                "step_logits": torch.ones((1, 32), dtype=torch.float32),
            },
            golden_file,
        )

        argv1 = [
            "smoke_e2e_dsv41_tp8.py",
            "--meta-dir",
            str(meta_dir),
            "--smoke-dir",
            str(smoke_dir),
            "--num-layers",
            "4",
            "--max-tokens",
            "1",
            "--profile-dir",
            str(prof_dir),
            "--save-golden",
            str(golden_file),
            "--verify-against-baseline",
            str(golden_file),
        ]
        monkeypatch.setattr(sys, "argv", argv1)
        monkeypatch.delenv("PYTHONPATH", raising=False)
        smoke.main()

        assert golden_file.exists()
        assert (prof_dir / "xprof_summary.json").exists()

        # Run 2: main with PYTHONPATH already set and no profiling
        monkeypatch.setenv("PYTHONPATH", "/custom/path")
        argv2 = [
            "smoke_e2e_dsv41_tp8.py",
            "--meta-dir",
            str(meta_dir),
            "--smoke-dir",
            str(smoke_dir),
            "--num-layers",
            "6",
            "--max-tokens",
            "1",
        ]
        monkeypatch.setattr(sys, "argv", argv2)
        smoke.main()

        # Run 3: exercise if __name__ == "__main__": main() via runpy
        monkeypatch.setattr(sys, "argv", ["smoke_e2e_dsv41_tp8.py", "--help"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPT_SMOKE), run_name="__main__")


# ==============================================================================
# Class 2: Comprehensive Unit Tests for bench_decode_sliding_window.py
# ==============================================================================


class TestBenchDecodeSlidingWindow:
    """Covers 100% lines & branches of bench_decode_sliding_window.py."""

    def test_arg_validators_positive_int(self):
        assert bench.positive_int("5") == 5
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive integer"
        ):
            bench.positive_int("0")
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive integer"
        ):
            bench.positive_int("-3")

    def test_arg_validators_positive_float(self):
        assert bench.positive_float("3.14") == 3.14
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive number"
        ):
            bench.positive_float("0.0")
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive number"
        ):
            bench.positive_float("-1.5")
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive number"
        ):
            bench.positive_float("nan")
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a positive number"
        ):
            bench.positive_float("inf")

    def test_arg_validators_nonnegative_float(self):
        assert bench.nonnegative_float("0.0") == 0.0
        assert bench.nonnegative_float("2.5") == 2.5
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a non-negative number"
        ):
            bench.nonnegative_float("-0.1")
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a non-negative number"
        ):
            bench.nonnegative_float("nan")

    def test_arg_validators_unit_interval_float(self):
        assert bench.unit_interval_float("0.5") == 0.5
        assert bench.unit_interval_float("1.0") == 1.0
        with pytest.raises(argparse.ArgumentTypeError, match="must be in the interval"):
            bench.unit_interval_float("0.0")
        with pytest.raises(argparse.ArgumentTypeError, match="must be in the interval"):
            bench.unit_interval_float("1.1")
        with pytest.raises(argparse.ArgumentTypeError, match="must be in the interval"):
            bench.unit_interval_float("-0.2")
        with pytest.raises(argparse.ArgumentTypeError, match="must be in the interval"):
            bench.unit_interval_float("nan")

    def test_arg_validators_nonnegative_int(self):
        assert bench.nonnegative_int("0") == 0
        assert bench.nonnegative_int("8") == 8
        with pytest.raises(
            argparse.ArgumentTypeError, match="must be a non-negative integer"
        ):
            bench.nonnegative_int("-1")

    def test_repeat_and_truncate_token_ids(self):
        with pytest.raises(ValueError, match="fixed prompt text produced no tokens"):
            bench.repeat_and_truncate_token_ids([], 10)

        res = bench.repeat_and_truncate_token_ids([1, 2, 3], 5)
        assert res == [1, 2, 3, 1, 2]

    def test_build_unique_prompt_token_ids_all_branches(self):
        # 1. Source tokens empty
        with pytest.raises(ValueError, match="fixed prompt text produced no tokens"):
            bench.build_unique_prompt_token_ids(
                encode_text=lambda _: [],
                target_tokens=10,
                concurrency=2,
            )

        # 2. Prefix tokens empty
        def encode_empty_prefix(text):
            return [] if "Unique benchmark prefix" in text else [10, 20, 30, 40, 50]

        with pytest.raises(ValueError, match="unique prefix produced no tokens"):
            bench.build_unique_prompt_token_ids(
                encode_text=encode_empty_prefix,
                target_tokens=10,
                concurrency=2,
            )

        # 3. Normal path with prefix >= target_tokens
        counter = {"val": 0}

        def encode_varying(text):
            if "Unique benchmark prefix" in text:
                counter["val"] += 1
                return [counter["val"], 100, 200, 300]
            return [10, 20, 30, 40]

        prompts_short, count = bench.build_unique_prompt_token_ids(
            encode_text=encode_varying,
            target_tokens=2,
            concurrency=2,
        )
        assert len(prompts_short) == 2
        assert len(prompts_short[0]) == 2
        assert count == 4

        # 4. Normal path with prefix < target_tokens
        counter["val"] = 0
        prompts_long, _ = bench.build_unique_prompt_token_ids(
            encode_text=encode_varying,
            target_tokens=8,
            concurrency=2,
        )
        assert len(prompts_long) == 2
        assert len(prompts_long[0]) == 8

        # 5. Non-unique prompts error
        def encode_constant(text):
            return [1, 2, 3]

        with pytest.raises(
            ValueError, match="tokenizer did not produce unique request prompts"
        ):
            bench.build_unique_prompt_token_ids(
                encode_text=encode_constant,
                target_tokens=5,
                concurrency=2,
            )

    def test_load_unique_prompt_token_ids(self, tmp_path, monkeypatch):
        import transformers

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                # Hash text to return stable tokens
                return [abs(hash(text + str(i))) % 1000 + 1 for i in range(5)]

        fake_tok = FakeTokenizer()
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: fake_tok,
        )

        # Directory path (local_files_only=True)
        local_dir = tmp_path / "tok_dir"
        local_dir.mkdir()
        prompts1, src_cnt1 = bench.load_unique_prompt_token_ids(
            tokenizer=str(local_dir),
            target_tokens=10,
            concurrency=2,
        )
        assert len(prompts1) == 2
        assert src_cnt1 == 5

        # HF hub id (local_files_only=False)
        prompts2, _ = bench.load_unique_prompt_token_ids(
            tokenizer="meta-llama/Llama-3-8B",
            target_tokens=10,
            concurrency=2,
        )
        assert len(prompts2) == 2

    def test_check_full_overlap_possible(self):
        # 1. Decode tokens outlasts prefill -> True
        possible, tok = bench.check_full_overlap_possible(
            concurrency=16,
            prefill_tokens=8192,
            decode_tokens=1024,
            max_num_batched_tokens=2048,
        )
        assert possible
        assert tok == 15 * 4  # (16-1) * ceil(8192/2048) = 60

        # 2. Decode tokens too short -> False
        possible_bad, _ = bench.check_full_overlap_possible(
            concurrency=16,
            prefill_tokens=8192,
            decode_tokens=50,
            max_num_batched_tokens=2048,
        )
        assert not possible_bad

    def test_parse_prometheus_sums_all_branches(self):
        text = """
# HELP vllm:num_requests_running Number of requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model="foo"} 3.5
vllm:num_requests_running{model="bar"} 1.5

# Ignored metric
unrelated_metric 10.0
# Invalid value
vllm:num_requests_waiting invalid_number
vllm:num_requests_waiting 2.0
"""
        names = {"vllm:num_requests_running", "vllm:num_requests_waiting"}
        sums = bench.parse_prometheus_sums(text, names)
        assert sums["vllm:num_requests_running"] == 5.0
        assert sums["vllm:num_requests_waiting"] == 2.0

    def test_server_metrics_sampler_all_branches(self, monkeypatch):
        # 1. Scrape raises RequestException -> graceful handling
        def fake_get_fail(*args, **kwargs):
            raise requests.RequestException("connection failed")

        monkeypatch.setattr(requests, "get", fake_get_fail)
        sampler = bench.ServerMetricsSampler("http://localhost:8000/metrics", 0.01)
        sampler._scrape()
        assert sampler.summary()["server_metrics_samples"] == 0

        # 2. Scrape succeeds with gauges and preemptions
        scrapes = [
            """
vllm:num_requests_running 4.0
vllm:num_requests_waiting 1.0
vllm:kv_cache_usage_perc 25.0
vllm:num_preemptions_total 10.0
""",
            """
vllm:num_requests_running 8.0
vllm:num_requests_waiting 0.0
vllm:kv_cache_usage_perc 50.0
vllm:num_preemptions_total 14.0
""",
        ]
        scrape_iter = iter(scrapes)

        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        def fake_get_success(*args, **kwargs):
            return FakeResponse(next(scrape_iter, scrapes[-1]))

        monkeypatch.setattr(requests, "get", fake_get_success)

        with bench.ServerMetricsSampler("http://localhost:8000/metrics", 0.01) as s:
            s._scrape()

        summary = s.summary()
        assert summary["server_metrics_samples"] >= 2
        assert summary["server_running_requests_max"] == 8.0
        assert summary["server_waiting_requests_max"] == 1.0
        assert summary["server_kv_cache_usage_max"] == 50.0
        assert summary["server_preemptions"] == 4.0

        # Scrape with missing names (covers 264->263 and 266->exit)
        monkeypatch.setattr(
            requests,
            "get",
            lambda *args, **kwargs: FakeResponse("vllm:num_requests_running 1.0\n"),
        )
        s_partial = bench.ServerMetricsSampler("http://localhost:8000/metrics", 0.01)
        s_partial._scrape()
        assert s_partial.summary()["server_running_requests_max"] == 1.0
        assert s_partial.summary()["server_waiting_requests_max"] is None
        assert s_partial.summary()["server_preemptions"] is None

        # 3. Direct execution of _run() loop (covers lines 272-274)
        s_run = bench.ServerMetricsSampler("http://localhost:8000/metrics", 0.001)
        s_run._stop.set()
        s_run._run()
        s_run._stop.clear()
        with mock.patch.object(s_run, "_scrape", side_effect=s_run._stop.set):
            s_run._run()

    def test_percentile_and_metric_stats(self):
        # Empty list
        assert bench.percentile([], 50) is None
        stats_empty = bench.metric_stats([])
        assert stats_empty["count"] == 0
        assert stats_empty["avg"] is None
        assert stats_empty["stddev"] == 0.0

        # Single element
        stats_single = bench.metric_stats([10.0])
        assert stats_single["count"] == 1
        assert stats_single["avg"] == 10.0
        assert stats_single["stddev"] == 0.0

        # Multi-element
        stats_multi = bench.metric_stats([10.0, 20.0, 30.0, 40.0])
        assert stats_multi["count"] == 4
        assert stats_multi["avg"] == 25.0
        assert stats_multi["min"] == 10.0
        assert stats_multi["max"] == 40.0
        assert stats_multi["p50"] == 20.0

    def test_build_request_tpot_rows(self):
        results = [
            # Error result (skipped)
            bench.RequestResult(
                request_id=0,
                started_s=1.0,
                finished_s=2.0,
                token_times_s=[],
                error="HTTP 500",
            ),
            # Too few tokens (skipped)
            bench.RequestResult(
                request_id=1,
                started_s=1.0,
                finished_s=2.0,
                token_times_s=[1.5],
                error=None,
            ),
            # Valid result
            bench.RequestResult(
                request_id=2,
                started_s=1.0,
                finished_s=4.0,
                token_times_s=[1.5, 2.5, 3.5],
                error=None,
            ),
        ]
        rows = bench.build_request_tpot_rows(
            round_index=1,
            results=results,
            batch_start_s=1.0,
        )
        assert len(rows) == 1
        assert rows[0]["request_id"] == 2
        assert rows[0]["received_tokens"] == 3
        assert rows[0]["first_token_after_batch_start_s"] == 0.5
        assert rows[0]["last_token_after_batch_start_s"] == 2.5
        assert rows[0]["decode_duration_s"] == 2.0
        assert math.isclose(rows[0]["tpot_ms"], 1000.0)

    def test_completion_token_delta(self):
        # 1. No choices
        assert bench.completion_token_delta({}, 5) == (0, 5)
        assert bench.completion_token_delta({"choices": []}, 5) == (0, 5)

        # 2. Choices with no text
        assert bench.completion_token_delta({"choices": [{"text": ""}]}, 5) == (
            0,
            5,
        )

        # 3. Usage with completion_tokens integer
        event_usage = {
            "choices": [{"text": "abc"}],
            "usage": {"completion_tokens": 8},
        }
        assert bench.completion_token_delta(event_usage, 5) == (3, 8)
        # Delta bounded below by 1
        assert bench.completion_token_delta(event_usage, 8) == (1, 8)

        # 4. Usage without completion_tokens integer
        event_no_usage = {"choices": [{"text": "abc"}]}
        assert bench.completion_token_delta(event_no_usage, 5) == (1, 6)

    def test_iter_sse_data(self):
        class FakeResponse:
            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        "",
                        ": heartbeat comment",
                        "invalid format",
                        'data: {"test": 1}',
                        "data: [DONE]",
                    ]
                )

        lines = list(bench.iter_sse_data(FakeResponse()))
        assert lines == ['{"test": 1}', "[DONE]"]

    def test_streamed_completion_all_branches(self, monkeypatch):
        # 1. Success with streaming data and DP rank
        class MockSuccessResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        (
                            'data: {"choices": [{"text": "a"}], '
                            '"usage": {"completion_tokens": 1}}'
                        ),
                        (
                            'data: {"choices": [{"text": "b"}], '
                            '"usage": {"completion_tokens": 2}}'
                        ),
                        "data: [DONE]",
                    ]
                )

        monkeypatch.setattr(
            requests, "post", lambda *args, **kwargs: MockSuccessResponse()
        )
        res_success = bench.streamed_completion(
            request_id=0,
            endpoint="http://localhost:8000/v1/completions",
            request_payload={},
            data_parallel_rank=2,
            timeout_s=10.0,
        )
        assert res_success.error is None
        assert len(res_success.token_times_s) == 2

        # 2. HTTP error response
        class MockErrorResponse:
            status_code = 500
            text = "Internal Server Error"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            requests, "post", lambda *args, **kwargs: MockErrorResponse()
        )
        res_error = bench.streamed_completion(
            request_id=1,
            endpoint="http://localhost:8000/v1/completions",
            request_payload={},
            data_parallel_rank=None,
            timeout_s=10.0,
        )
        assert res_error.error is not None
        assert "HTTP 500" in res_error.error

        # 3. Connection / Exception error
        def mock_raise(*args, **kwargs):
            raise requests.Timeout("connection timed out")

        monkeypatch.setattr(requests, "post", mock_raise)
        res_timeout = bench.streamed_completion(
            request_id=2,
            endpoint="http://localhost:8000/v1/completions",
            request_payload={},
            data_parallel_rank=None,
            timeout_s=10.0,
        )
        assert res_timeout.error is not None
        assert "Timeout" in res_timeout.error

        # 4. Empty SSE stream (covers line 411->424)
        class MockEmptyStreamResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def iter_lines(self, decode_unicode=True):
                return iter([])

        monkeypatch.setattr(
            requests, "post", lambda *args, **kwargs: MockEmptyStreamResponse()
        )
        res_empty = bench.streamed_completion(
            request_id=3,
            endpoint="http://localhost:8000/v1/completions",
            request_payload={},
            data_parallel_rank=None,
            timeout_s=10.0,
        )
        assert len(res_empty.token_times_s) == 0

    def test_run_round_all_branches(self, monkeypatch):
        # 1. Payload count mismatch
        with pytest.raises(ValueError, match="expected 2 request payloads"):
            bench.run_round(
                endpoint="http://localhost:8000/v1/completions",
                request_payloads=[{}],
                concurrency=2,
                data_parallel_size=1,
                timeout_s=5.0,
            )

        # 2. Successful run with DP size > 1
        fake_result = bench.RequestResult(
            request_id=0,
            started_s=1.0,
            finished_s=2.0,
            token_times_s=[1.2, 1.8],
            error=None,
        )
        monkeypatch.setattr(
            bench,
            "streamed_completion",
            lambda request_id, **kwargs: dataclasses.replace(
                fake_result, request_id=request_id
            ),
        )
        results = bench.run_round(
            endpoint="http://localhost:8000/v1/completions",
            request_payloads=[{}, {}],
            concurrency=2,
            data_parallel_size=2,
            timeout_s=5.0,
        )
        assert len(results) == 2
        assert results[0].request_id == 0
        assert results[1].request_id == 1

    def test_analyze_round_incomplete_requests(self):
        results = [
            bench.RequestResult(0, 1.0, 2.0, [1.5], None),  # < 2 tokens
            bench.RequestResult(1, 1.0, 2.0, [1.5, 2.0], "Error!"),  # error
        ]
        analysis = bench.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            window_s=1.0,
            step_s=0.5,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.9,
        )
        assert not analysis.summary["valid"]
        assert analysis.summary["invalid_reason"] == "incomplete_requests"

    def test_analyze_round_duration_and_span_checks(self):
        # Two requests with overlapping decode intervals
        # Req 0: tokens at [1.0, 1.5, 2.0] -> decode interval (1.5, 2.0)
        # Req 1: tokens at [1.0, 1.8, 2.5] -> decode interval (1.8, 2.5)
        # Full overlap start: max(1.5, 1.8) = 1.8, end: min(2.0, 2.5) = 2.0
        # duration = 0.2s
        results = [
            bench.RequestResult(0, 0.5, 2.0, [1.0, 1.5, 2.0], None),
            bench.RequestResult(1, 0.5, 2.5, [1.0, 1.8, 2.5], None),
        ]

        # 1. full_duration_s < min_full_overlap_s
        a1 = bench.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            window_s=0.1,
            step_s=0.05,
            min_full_overlap_s=1.0,  # 0.2 < 1.0
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.9,
        )
        assert a1.summary["invalid_reason"] == "full_overlap_shorter_than_minimum"

        # 2. full_overlap_token_count < min_full_overlap_tokens
        a2 = bench.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            window_s=0.1,
            step_s=0.05,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=50,  # only ~1 token
            min_peak_active_fraction=0.9,
        )
        assert a2.summary["invalid_reason"] == "full_overlap_has_too_few_tokens"

        # 3. decode_span_shorter_than_window
        a3 = bench.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            window_s=10.0,  # decode span is 2.5 - 1.5 = 1.0 < 10.0
            step_s=0.5,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.9,
        )
        assert a3.summary["invalid_reason"] == "decode_span_shorter_than_window"

    def test_analyze_round_windows_and_intervals(self):
        # Craft results that exercise:
        # - multiple sliding windows
        # - active_requests == 0 branch during stepping
        # - peak window candidate filtering
        # - overlapping and non-overlapping peak intervals merging
        # Req 0: starts at 0.0, tokens [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Req 1: starts at 0.0, tokens [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tokens = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        results = [
            bench.RequestResult(0, 0.0, 6.0, list(tokens), None),
            bench.RequestResult(1, 0.0, 6.0, list(tokens), None),
        ]

        # Valid round with windows
        a = bench.analyze_round(
            round_index=1,
            results=results,
            concurrency=2,
            window_s=1.0,
            step_s=0.5,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.9,
        )
        assert a.summary["valid"]
        assert a.summary["active_requests_max"] == 2
        assert a.summary["window_count"] > 0
        assert a.summary["peak_window_throughput_tok_s"] is not None
        assert a.summary["timeline_valid_full_concurrency_decode"]

        # Case where peak_window is None
        # (min_peak_active_fraction=1.0 but requests never reached it)
        # Create requests with non-overlapping decode intervals:
        r_disjoint = [
            bench.RequestResult(0, 0.0, 3.0, [1.0, 1.5, 2.5], None),
            bench.RequestResult(1, 0.0, 6.0, [1.0, 4.5, 5.5], None),
        ]
        # Full overlap between req 0 and req 1 is empty, so full_duration_s == 0
        # If min_full_overlap_s is 0, full_duration_s < 0 is False
        a_disjoint = bench.analyze_round(
            round_index=1,
            results=r_disjoint,
            concurrency=2,
            window_s=0.5,
            step_s=0.5,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=1.0,  # requires 2 active requests; max is 1
        )
        assert a_disjoint.summary["peak_window_active_requests"] == 0
        assert a_disjoint.summary["peak_window_throughput_tok_s"] is None

        # Case where windows is empty (no_samples)
        # All candidate windows have 0 active requests
        # (covers line 590 active_requests == 0 and line 676)
        r_no_samples = [
            bench.RequestResult(0, 0.5, 1.4, [0.8, 1.0, 1.4], None),
            bench.RequestResult(1, 0.5, 3.0, [2.4, 2.6, 3.0], None),
        ]
        a_empty = bench.analyze_round(
            round_index=1,
            results=r_no_samples,
            concurrency=2,
            window_s=1.0,
            step_s=0.2,
            min_full_overlap_s=0.0,
            min_full_overlap_tokens=0,
            min_peak_active_fraction=0.5,
        )
        assert not a_empty.summary["valid"]
        assert a_empty.summary["invalid_reason"] == "no_samples"
        assert a_empty.summary["window_count"] == 0

    def test_file_writers_and_formatters(self, tmp_path):
        # 1. write_raw_requests
        raw_path = tmp_path / "raw.jsonl"
        results = [
            bench.RequestResult(0, 1.0, 3.0, [1.5, 2.5], None),
            bench.RequestResult(1, 1.0, 2.0, [], "Error!"),
        ]
        bench.write_raw_requests(raw_path, round_index=1, results=results)
        assert raw_path.exists()
        raw_lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(raw_lines) == 2

        # 2. write_request_tpot_csv
        tpot_path = tmp_path / "request_tpot.csv"
        tpots = [
            {
                "round": 1,
                "request_id": 0,
                "received_tokens": 10,
                "first_token_after_batch_start_s": 0.5,
                "last_token_after_batch_start_s": 2.5,
                "decode_duration_s": 2.0,
                "tpot_ms": 200.0,
            }
        ]
        bench.write_request_tpot_csv(tpot_path, tpots)
        assert tpot_path.exists()

        # 3. write_rounds_csv
        rounds_path = tmp_path / "rounds.csv"
        round_row = {
            "round": 1,
            "valid": True,
            "successful_requests": 2,
            "failed_requests": 0,
            "usage_tokens": 20,
            "submit_skew_s": 0.01,
            "first_token_skew_s": 0.02,
            "ttft_s": {"p50": 0.5, "p90": 0.6},
            "end_to_end_throughput_tok_s": 10.0,
            "full_overlap_duration_s": 1.0,
            "full_overlap_token_count": 8,
            "full_overlap_throughput_tok_s": 8.0,
            "active_requests_max": 2,
            "timeline_valid_full_concurrency_decode": True,
            "full_concurrency_window_count": 2,
            "window_count": 2,
            "peak_window_min_active_requests": 2,
            "peak_window_active_requests": 2,
            "peak_window_start_after_batch_start_s": 0.5,
            "peak_window_end_after_batch_start_s": 1.5,
            "peak_window_throughput_tok_s": 8.0,
            "peak_window_tpot_ms": {
                "count": 4,
                "avg": 250.0,
                "p50": 250.0,
                "p90": 250.0,
                "p99": 250.0,
            },
            "window_throughput_tok_s": {
                "count": 2,
                "avg": 8.0,
                "min": 8.0,
                "max": 8.0,
                "stddev": 0.0,
                "p50": 8.0,
                "p90": 8.0,
                "p99": 8.0,
            },
            "peak_active_tpot_ms": {
                "count": 4,
                "avg": 250.0,
                "min": 250.0,
                "max": 250.0,
                "stddev": 0.0,
                "p50": 250.0,
                "p90": 250.0,
                "p99": 250.0,
            },
            "request_tpot_ms": {
                "count": 2,
                "avg": 200.0,
                "min": 200.0,
                "max": 200.0,
                "stddev": 0.0,
                "p50": 200.0,
                "p90": 200.0,
                "p99": 200.0,
            },
            "server_metrics_samples": 5,
            "server_running_requests_max": 2.0,
            "server_waiting_requests_max": 0.0,
            "server_kv_cache_usage_max": 10.0,
            "server_preemptions": 0.0,
            "invalid_reason": "",
        }
        bench.write_rounds_csv(rounds_path, [round_row])
        assert rounds_path.exists()

        # 4. write_timeline_csv
        tl_path = tmp_path / "timeline.csv"
        windows = [
            {
                "round": 1,
                "window": 1,
                "start_after_batch_start_s": 0.5,
                "end_after_batch_start_s": 1.5,
                "active_requests": 2,
                "token_count": 8,
                "throughput_tok_s": 8.0,
            }
        ]
        bench.write_timeline_csv(tl_path, windows)
        assert tl_path.exists()

        # 5. format_metric and stats_row
        assert bench.format_metric(None) == "-"
        assert bench.format_metric(3.14159) == "3.142"
        s_row = bench.stats_row("TestMetric", round_row["window_throughput_tok_s"])
        assert "|TestMetric|2|8.000|" in s_row

        # 6. write_markdown
        md_path = tmp_path / "summary.md"
        summary_payload = {
            "benchmark": {
                "window_s": 1.0,
                "concurrency": 2,
                "prefill_tokens": 100,
                "decode_tokens": 50,
                "model": "dsv41",
                "data_parallel_size": 1,
                "tensor_parallel_size": 8,
                "tokenizer": "dsv41",
                "step_s": 0.5,
                "min_peak_active_fraction": 0.9,
                "min_full_overlap_s": 0.0,
                "min_full_overlap_tokens": 0,
                "metrics_interval_s": 1.0,
                "base_url": "http://localhost:8000",
                "max_num_batched_tokens": 2048,
            },
            "valid_rounds": 1,
            "requested_rounds": 1,
            "best": {
                "output_throughput": 8.0,
                "mean_tpot_ms": 250.0,
            },
            "aggregate": {
                "window_throughput_tok_s": round_row["window_throughput_tok_s"],
                "peak_window_throughput_tok_s": round_row["window_throughput_tok_s"],
                "peak_window_tpot_p50_ms": bench.metric_stats([250.0, 250.0]),
                "full_overlap_throughput_tok_s": round_row["window_throughput_tok_s"],
                "peak_active_tpot_p50_ms": round_row["peak_active_tpot_ms"],
                "request_tpot_ms": round_row["request_tpot_ms"],
                "ttft_p50_s": bench.metric_stats([0.5]),
            },
            "results": [round_row],
        }
        bench.write_markdown(md_path, summary_payload)
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert "Decode Sliding-Window Benchmark" in md_text
        assert "Best round:" in md_text

        # write_markdown with best=None and max_num_batched_tokens=None
        summary_no_best = dict(summary_payload)
        summary_no_best["best"] = None
        summary_no_best["benchmark"] = dict(summary_payload["benchmark"])
        summary_no_best["benchmark"]["max_num_batched_tokens"] = None
        bench.write_markdown(md_path, summary_no_best)
        assert md_path.exists()

    def test_parse_args_all_branches(self, monkeypatch):
        # 1. decode_tokens < 2 raises error
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--decode-tokens",
                "1",
            ],
        )
        with pytest.raises(SystemExit):
            bench.parse_args()

        # 2. tokenizer is None -> defaults to model
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "my_model",
                "--output-dir",
                "/tmp/out",
                "--decode-tokens",
                "10",
            ],
        )
        args = bench.parse_args()
        assert args.tokenizer == "my_model"

        # 2b. tokenizer is not None (covers line 1174->1176)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "my_model",
                "--tokenizer",
                "custom_tok",
                "--output-dir",
                "/tmp/out",
                "--decode-tokens",
                "10",
            ],
        )
        args_custom = bench.parse_args()
        assert args_custom.tokenizer == "custom_tok"

        # 3. max_num_batched_tokens overlap check impossible -> error
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--concurrency",
                "16",
                "--prefill-tokens",
                "8192",
                "--decode-tokens",
                "10",  # 10 < 60 tokens needed
                "--max-num-batched-tokens",
                "2048",
            ],
        )
        with pytest.raises(SystemExit):
            bench.parse_args()

        # 4. max_num_batched_tokens overlap check possible -> succeeds
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--concurrency",
                "16",
                "--prefill-tokens",
                "8192",
                "--decode-tokens",
                "100",  # 100 > 60 tokens needed
                "--max-num-batched-tokens",
                "2048",
            ],
        )
        args_ok = bench.parse_args()
        assert args_ok.decode_tokens == 100

    def test_main_success_single_and_multi_round(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "bench_out"

        class FakeHealthResponse:
            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            requests, "get", lambda *args, **kwargs: FakeHealthResponse()
        )
        monkeypatch.setattr(
            bench,
            "load_unique_prompt_token_ids",
            lambda **kwargs: ([[1, 2, 3], [4, 5, 6]], 3),
        )

        # Mock ServerMetricsSampler context manager
        class FakeSampler:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def summary(self):
                return {
                    "server_metrics_samples": 2,
                    "server_running_requests_max": 2.0,
                    "server_waiting_requests_max": 0.0,
                    "server_kv_cache_usage_max": 10.0,
                    "server_preemptions": 0.0,
                }

        monkeypatch.setattr(
            bench, "ServerMetricsSampler", lambda *args, **kwargs: FakeSampler()
        )

        # Mock run_round
        tokens = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        fake_results = [
            bench.RequestResult(0, 0.0, 6.0, list(tokens), None),
            bench.RequestResult(1, 0.0, 6.0, list(tokens), None),
        ]
        monkeypatch.setattr(bench, "run_round", lambda **kwargs: fake_results)

        # 1. Single round run
        argv_single = [
            "bench_decode_sliding_window.py",
            "--model",
            "dsv41",
            "--output-dir",
            str(out_dir),
            "--concurrency",
            "2",
            "--prefill-tokens",
            "10",
            "--decode-tokens",
            "6",
            "--rounds",
            "1",
            "--window-seconds",
            "1.0",
            "--cache-salt-prefix",
            "test-salt",
        ]
        monkeypatch.setattr(sys, "argv", argv_single)
        bench.main()

        assert (out_dir / "summary.json").exists()
        assert (out_dir / "rounds.csv").exists()
        assert (out_dir / "summary.md").exists()

        # 2. Multi-round run with cooldown
        argv_multi = [
            "bench_decode_sliding_window.py",
            "--model",
            "dsv41",
            "--output-dir",
            str(out_dir),
            "--concurrency",
            "2",
            "--prefill-tokens",
            "10",
            "--decode-tokens",
            "6",
            "--rounds",
            "2",
            "--cooldown-seconds",
            "0.01",
            "--window-seconds",
            "1.0",
        ]
        monkeypatch.setattr(sys, "argv", argv_multi)
        bench.main()

    def test_main_failure_cases(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "bench_fail_out"

        class FakeHealthResponse:
            def raise_for_status(self):
                pass

        monkeypatch.setattr(
            requests, "get", lambda *args, **kwargs: FakeHealthResponse()
        )
        monkeypatch.setattr(
            bench,
            "load_unique_prompt_token_ids",
            lambda **kwargs: ([[1, 2]], 2),
        )

        class FakeSampler:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def summary(self):
                return {}

        monkeypatch.setattr(
            bench, "ServerMetricsSampler", lambda *args, **kwargs: FakeSampler()
        )

        # 1. failed_requests > 0 raises SystemExit
        fail_results = [
            bench.RequestResult(0, 0.0, 1.0, [], "Request failed!"),
        ]
        monkeypatch.setattr(bench, "run_round", lambda **kwargs: fail_results)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                str(out_dir),
                "--concurrency",
                "1",
                "--decode-tokens",
                "5",
            ],
        )
        with pytest.raises(SystemExit, match="failed requests"):
            bench.main()

        # 2. invalid_rounds raises SystemExit
        # Results succeed but analyze_round marks invalid
        # (e.g. min_full_overlap_s not met)
        short_results = [
            bench.RequestResult(0, 0.0, 1.0, [0.1, 0.2, 0.3], None),
        ]
        monkeypatch.setattr(bench, "run_round", lambda **kwargs: short_results)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                str(out_dir),
                "--concurrency",
                "1",
                "--decode-tokens",
                "5",
                "--min-full-overlap-seconds",
                "100.0",  # causes invalid round
            ],
        )
        with pytest.raises(SystemExit, match="invalid rounds"):
            bench.main()

        # 3. throughput_p50 is None branch (prints "no complete sliding window")
        # Mock analyze_round to return valid=True but with empty stats
        fake_analysis = bench.RoundAnalysis(
            summary={
                "round": 1,
                "valid": True,
                "failed_requests": 0,
                "successful_requests": 1,
                "concurrency": 1,
                "usage_tokens": 10,
                "submit_skew_s": 0.0,
                "first_token_skew_s": 0.0,
                "ttft_s": bench.metric_stats([0.5]),
                "end_to_end_throughput_tok_s": 1.0,
                "full_overlap_duration_s": 1.0,
                "full_overlap_token_count": 5,
                "full_overlap_throughput_tok_s": None,
                "active_requests_max": 1,
                "timeline_valid_full_concurrency_decode": True,
                "full_concurrency_window_count": 0,
                "window_count": 0,
                "peak_window_min_active_requests": 1,
                "peak_window_active_requests": 0,
                "peak_window_start_after_batch_start_s": None,
                "peak_window_end_after_batch_start_s": None,
                "peak_window_throughput_tok_s": None,
                "peak_window_tpot_ms": bench.metric_stats([]),
                "window_throughput_tok_s": bench.metric_stats([10.0]),
                "peak_active_tpot_ms": bench.metric_stats([]),
                "request_tpot_ms": bench.metric_stats([]),
                "invalid_reason": "",
            },
            windows=[],
            request_tpots=[],
        )
        monkeypatch.setattr(bench, "analyze_round", lambda **kwargs: fake_analysis)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bench_decode_sliding_window.py",
                "--model",
                "m",
                "--output-dir",
                str(out_dir),
                "--concurrency",
                "1",
                "--decode-tokens",
                "5",
            ],
        )
        bench.main()

    def test_main_entrypoint_runpy(self, monkeypatch):
        # Exercises if __name__ == "__main__": main()
        monkeypatch.setattr(sys, "argv", ["bench_decode_sliding_window.py", "--help"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPT_BENCH), run_name="__main__")
