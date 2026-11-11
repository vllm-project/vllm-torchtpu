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
"""Unit tests for the indexer benchmark and TP=8 smoke.

Targets:
1. smoke_e2e_dsv41_tp8.py
"""

from __future__ import annotations

import json
import os
import pathlib
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure CPU execution defaults before importing jax or TPU runtime libraries.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_BENCH_DIR = _REPO_ROOT / "scripts" / "dsv41_bench"

if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse existing common module if already loaded (e.g., from selftest) to avoid
# module shadowing.
existing_common = sys.modules.get("common")
if existing_common is not None:
    common = existing_common
else:
    from scripts.dsv41_bench import common  # noqa: E402

    sys.modules.setdefault("common", common)
    sys.modules["common"] = common

import smoke_e2e_dsv41_tp8 as smoke  # noqa: E402

smoke.common = common


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
# 1. Tests for smoke_e2e_dsv41_tp8.py
# ==============================================================================


class TestSmokeE2E:
    def test_deterministic_single_dummy_weight_branches(self):
        smoke._install_deterministic_dummy_weights()
        import vllm.model_executor.model_loader.weight_utils as wu

        # 1. Meta device early return
        meta_param = torch.nn.Parameter(
            torch.empty((4, 4), device="meta", dtype=torch.float32)
        )
        wu.initialize_single_dummy_weight(meta_param)
        assert meta_param.device.type == "meta"

        # 2. Custom dummy_weight_value
        p_val = torch.nn.Parameter(torch.empty((2, 2), dtype=torch.float32))
        p_val.dummy_weight_value = 0.5
        wu.initialize_single_dummy_weight(p_val)
        assert torch.all(p_val == 0.5)

        # 3. Non-floating point: uint8 with shape[-1] <= 160 (UE8M0 scales)
        p_scale = torch.nn.Parameter(
            torch.empty((2, 128), dtype=torch.uint8), requires_grad=False
        )
        wu.initialize_single_dummy_weight(p_scale)
        assert torch.all(p_scale == 120)

        # 4. Non-floating point: uint8 with shape[-1] > 160 (FP4 weights)
        p_fp4 = torch.nn.Parameter(
            torch.empty((2, 256), dtype=torch.uint8), requires_grad=False
        )
        wu.initialize_single_dummy_weight(p_fp4)
        assert torch.all(p_fp4 == 0x21)

        # 5. Non-floating point: uint8 with ndim < 2
        p_u1 = torch.nn.Parameter(
            torch.empty((10,), dtype=torch.uint8), requires_grad=False
        )
        wu.initialize_single_dummy_weight(p_u1)
        assert torch.all(p_u1 == 0x21)

        # 6. Non-floating point: int32/int64 -> zero_()
        p_int = torch.nn.Parameter(
            torch.ones((4, 4), dtype=torch.int32), requires_grad=False
        )
        wu.initialize_single_dummy_weight(p_int)
        assert torch.all(p_int == 0)

        # 7. Standard floating point parameter
        p_bf16 = torch.nn.Parameter(torch.empty((4, 8), dtype=torch.bfloat16))
        wu.initialize_single_dummy_weight(p_bf16)
        assert torch.isfinite(p_bf16).all()

    def test_install_logits_capture_branches(self, tmp_path, monkeypatch):
        from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

        # 1. No DSV41_SMOKE_LOGITS_DIR
        monkeypatch.delenv("DSV41_SMOKE_LOGITS_DIR", raising=False)
        smoke._install_logits_capture()

        # 2. Set DSV41_SMOKE_LOGITS_DIR
        monkeypatch.setenv("DSV41_SMOKE_LOGITS_DIR", str(tmp_path / "logits"))
        if hasattr(TPUModelRunner, "_dsv41_smoke_logits_hooked"):
            delattr(TPUModelRunner, "_dsv41_smoke_logits_hooked")

        orig_from = TPUModelRunner.compute_logits_from_hidden_states
        orig_sel = TPUModelRunner.compute_selected_logits
        try:
            TPUModelRunner.compute_logits_from_hidden_states = (
                lambda self, hs, *a, **k: torch.ones((1, 10))
            )
            TPUModelRunner.compute_selected_logits = lambda self, hs, ids, *a, **k: (
                torch.ones((1, 10))
            )
            smoke._install_logits_capture()
            assert getattr(TPUModelRunner, "_dsv41_smoke_logits_hooked", False)

            # Test already hooked branch
            smoke._install_logits_capture()

            runner = MagicMock()
            l1 = TPUModelRunner.compute_logits_from_hidden_states(
                runner, torch.ones((1, 10))
            )
            assert l1 is not None
            l2 = TPUModelRunner.compute_selected_logits(
                runner, torch.ones((1, 10)), torch.tensor([0])
            )
            assert l2 is not None
            # Check logits files were saved
            assert (tmp_path / "logits" / "steps" / "step_0000.pt").exists()
            assert (tmp_path / "logits" / "steps" / "step_0001.pt").exists()

            # Empty logits: covers line 115->exit (logits.numel() == 0)
            TPUModelRunner.compute_logits_from_hidden_states(
                runner, torch.empty((0, 0))
            )
            # RANK=1: covers line 115->exit (rank != 0)
            monkeypatch.setenv("RANK", "1")
            TPUModelRunner.compute_logits_from_hidden_states(
                runner, torch.ones((1, 10))
            )
            monkeypatch.setenv("RANK", "0")
        finally:
            TPUModelRunner.compute_logits_from_hidden_states = orig_from
            TPUModelRunner.compute_selected_logits = orig_sel
            if hasattr(TPUModelRunner, "_dsv41_smoke_logits_hooked"):
                delattr(TPUModelRunner, "_dsv41_smoke_logits_hooked")

    def test_build_smoke_model_dir_branches(self, tmp_path):
        meta_dir = tmp_path / "meta"
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

        # Pre-existing out_dir
        out_dir = tmp_path / "smoke_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "old_file.txt").write_text("old", encoding="utf-8")

        # 1. 4-layer build
        smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=4)
        c4 = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))[
            "text_config"
        ]
        assert c4["num_hidden_layers"] == 4
        assert (out_dir / "tokenizer.json").exists()

        # 2. 6-layer build
        smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=6)
        c6 = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))[
            "text_config"
        ]
        assert c6["num_hidden_layers"] == 6

        # 3. Invalid layer count raises ValueError
        with pytest.raises(ValueError, match="Unsupported --num-layers"):
            smoke.build_smoke_model_dir(meta_dir, out_dir, num_layers=5)

    def test_extract_top_logprobs_branches(self):
        # Empty logprobs
        ids_e, vals_e = smoke._extract_top_logprobs(None)
        assert ids_e.numel() == 0 and vals_e.numel() == 0

        # Logprobs with NaN
        lps = [
            {
                1: SimpleNamespace(logprob=float("nan")),
                2: SimpleNamespace(logprob=-0.5),
            }
        ]
        ids, vals = smoke._extract_top_logprobs(lps, top_k=2)
        assert ids.tolist() == [[2, 1]]

    def test_load_captured_step_logits_branches(self, tmp_path):
        # 1. Non-existent dir
        missing_dir = tmp_path / "non_existent"
        loaded_empty = smoke._load_captured_step_logits(missing_dir)
        assert loaded_empty.numel() == 0

        # 2. Steps with invalid ndim, zero rows, and max_steps
        steps_dir = tmp_path / "steps"
        steps_dir.mkdir()
        # Invalid ndim
        torch.save(torch.zeros((10,)), steps_dir / "step_0000.pt")
        # All zero row (CED=1 partial prefill)
        torch.save(torch.zeros((1, 8)), steps_dir / "step_0001.pt")
        # Valid rows
        torch.save(torch.ones((1, 8)), steps_dir / "step_0002.pt")
        torch.save(torch.ones((1, 8)) * 2, steps_dir / "step_0003.pt")

        loaded = smoke._load_captured_step_logits(steps_dir, max_steps=1)
        assert loaded.shape == (1, 8)
        assert torch.all(loaded == 1)

        # 3. max_steps > available steps: loop finishes without break
        # (covers lines 271->282 and 280->271)
        loaded_all = smoke._load_captured_step_logits(steps_dir, max_steps=10)
        assert loaded_all.shape == (2, 8)

    def test_verify_golden_all_branches(self, tmp_path):
        golden_path = tmp_path / "golden.pt"

        # 1. Golden file not found
        with pytest.raises(FileNotFoundError, match="Golden file not found"):
            smoke.verify_golden({}, golden_path)

        # 2. Empty step_logits
        logits = torch.ones((1, 16), dtype=torch.float32)
        golden_payload = {
            "token_ids": torch.tensor([5], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[5]], dtype=torch.int64),
            "top_logprob_values": torch.tensor([[-0.1]], dtype=torch.float32),
            "step_logits": torch.empty((0, 0)),
        }
        torch.save(golden_payload, golden_path)
        with pytest.raises(common.VerificationError, match="step_logits is empty"):
            smoke.verify_golden(
                {**golden_payload, "step_logits": torch.empty((0, 0))}, golden_path
            )

        # 3. top1_match_rate < 1.0
        golden_payload["step_logits"] = logits
        torch.save(golden_payload, golden_path)
        bad_logits = torch.zeros((1, 16), dtype=torch.float32)
        bad_logits[0, 1] = 10.0  # argmax 1 != 0
        actual_bad = {**golden_payload, "step_logits": bad_logits}
        with pytest.raises(common.VerificationError, match="top1_match_rate"):
            smoke.verify_golden(actual_bad, golden_path)

        # 4. Successful verify without step_logits
        payload_no_logits = {
            "token_ids": torch.tensor([5], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[5]], dtype=torch.int64),
            "top_logprob_values": torch.tensor([[-0.1]], dtype=torch.float32),
        }
        g_no_logits = tmp_path / "g_no_logits.pt"
        torch.save(payload_no_logits, g_no_logits)
        rep = smoke.verify_golden(payload_no_logits, g_no_logits)
        assert rep["token_ids"]["ok"]
        assert rep["top_logprob_values"]["ok"]

    def test_verify_golden_tied_argmax_and_shift_invariant(self, tmp_path):
        golden_path = tmp_path / "golden_tied.pt"
        exp_logits = torch.zeros((1, 16), dtype=torch.float32)
        exp_logits[0, 0] = 10.0
        exp_logits[0, 1] = 10.0
        golden_payload = {
            "token_ids": torch.tensor([0], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[0, 1]], dtype=torch.int64),
            "top_logprob_values": torch.tensor([[100.0, 200.0]], dtype=torch.float32),
            "step_logits": exp_logits,
        }
        torch.save(golden_payload, golden_path)

        actual_payload = {
            "token_ids": torch.tensor([1], dtype=torch.int64),
            "top_logprob_ids": torch.tensor([[1, 0]], dtype=torch.int64),
            "top_logprob_values": torch.tensor([[105.0, 205.0]], dtype=torch.float32),
            "step_logits": exp_logits.clone(),
        }
        rep = smoke.verify_golden(actual_payload, golden_path)
        assert rep["token_ids"]["ok"]
        assert rep["token_ids"].get("tied_argmax_match", False)
        assert rep["top_logprob_values"]["ok"]
        assert rep["top_logprob_values"].get("shift_invariant_match", False)

    def test_verify_golden_edge_branches(self, tmp_path):
        # 1. top1_match == 1.0 but step_logits ok is False (covers 330->370)
        p1 = tmp_path / "g1.pt"
        torch.save(
            {
                "token_ids": torch.tensor([0]),
                "top_logprob_ids": torch.tensor([[0]]),
                "top_logprob_values": torch.tensor([[-0.1]]),
                "step_logits": torch.tensor([[10.0, 1.0]]),
            },
            p1,
        )
        act1 = {
            "token_ids": torch.tensor([0]),
            "top_logprob_ids": torch.tensor([[0]]),
            "top_logprob_values": torch.tensor([[-0.1]]),
            "step_logits": torch.tensor([[1000.0, 1.0]]),
        }
        with pytest.raises(common.VerificationError, match="step_logits"):
            smoke.verify_golden(act1, p1)

        # 2. token_ids already ok (covers 335->347) and top_logprob_values already
        # ok (covers 347->370)
        p2 = tmp_path / "g2.pt"
        torch.save(
            {
                "token_ids": torch.tensor([0]),
                "top_logprob_ids": torch.tensor([[0]]),
                "top_logprob_values": torch.tensor([[-0.1]]),
                "step_logits": torch.tensor([[10.0, 1.0]]),
            },
            p2,
        )
        act2 = {
            "token_ids": torch.tensor([0]),
            "top_logprob_ids": torch.tensor([[0]]),
            "top_logprob_values": torch.tensor([[-0.1]]),
            "step_logits": torch.tensor([[10.0, 1.0]]),
        }
        r2 = smoke.verify_golden(act2, p2)
        assert r2["token_ids"]["ok"]
        assert not r2["token_ids"].get("tied_argmax_match", False)

        # 3. act_ids.numel() != exp_logits.shape[0] (covers 337->347)
        p3 = tmp_path / "g3.pt"
        torch.save(
            {
                "token_ids": torch.tensor([0]),
                "top_logprob_ids": torch.tensor([[0]]),
                "top_logprob_values": torch.tensor([[-0.1]]),
                "step_logits": torch.tensor([[10.0, 1.0]]),
            },
            p3,
        )
        act3 = {
            "token_ids": torch.tensor([1, 2]),
            "top_logprob_ids": torch.tensor([[0]]),
            "top_logprob_values": torch.tensor([[-0.1]]),
            "step_logits": torch.tensor([[10.0, 1.0]]),
        }
        with pytest.raises(common.VerificationError, match="token_ids"):
            smoke.verify_golden(act3, p3)

        # 4. top_logprob_values shape mismatch with cos_sim >= 0.999
        # (covers False branch 353->370)
        p4 = tmp_path / "g4.pt"
        torch.save(
            {
                "token_ids": torch.tensor([0]),
                "top_logprob_ids": torch.tensor([[0, 1]]),
                "top_logprob_values": torch.tensor([[100.0, 200.0]]),
                "step_logits": torch.tensor([[10.0, 1.0]]),
            },
            p4,
        )
        act4 = {
            "token_ids": torch.tensor([0]),
            "top_logprob_ids": torch.tensor([[0, 1]]),
            "top_logprob_values": torch.tensor([[105.0]]),
            "step_logits": torch.tensor([[10.0, 1.0]]),
        }
        orig_comp = common.compare_tensors

        def mock_comp(a, e, crit):
            res = orig_comp(a, e, crit)
            if a.shape != e.shape:
                res["cos_sim"] = 1.0
            return res

        with (
            patch(
                "scripts.dsv41_bench.common.compare_tensors",
                side_effect=mock_comp,
            ),
            pytest.raises(common.VerificationError, match="top_logprob_values"),
        ):
            smoke.verify_golden(act4, p4)

        # 5. centered_diff > 1e-3 (covers 363->370)
        p5 = tmp_path / "g5.pt"
        torch.save(
            {
                "token_ids": torch.tensor([0]),
                "top_logprob_ids": torch.tensor([[0, 1]]),
                "top_logprob_values": torch.tensor([[100.0, 200.0]]),
                "step_logits": torch.tensor([[10.0, 1.0]]),
            },
            p5,
        )
        act5 = {
            "token_ids": torch.tensor([0]),
            "top_logprob_ids": torch.tensor([[0, 1]]),
            "top_logprob_values": torch.tensor([[105.0, 204.0]]),
            "step_logits": torch.tensor([[10.0, 1.0]]),
        }
        with pytest.raises(common.VerificationError, match="top_logprob_values"):
            smoke.verify_golden(act5, p5)

    def test_operator_shape_ledger_4l_and_6l(self):
        l4 = smoke.build_operator_shape_ledger(
            num_layers=4, batch_tokens=1024, kv_len=2048
        )
        assert l4["num_layers"] == 4
        assert "L0_swa_only" in l4["layers"]

        l6 = smoke.build_operator_shape_ledger(
            num_layers=6, batch_tokens=1024, kv_len=2048, ced_window=10
        )
        assert l6["num_layers"] == 6

    def test_parse_args(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["smoke.py", "--num-layers", "6", "--max-tokens", "2"]
        )
        args = smoke.parse_args()
        assert args.num_layers == 6
        assert args.max_tokens == 2

    def test_smoke_e2e_main_and_entrypoint(self, tmp_path, monkeypatch):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "text_config": {
                "engram_max_ngram_size": 4,
                "engram_n_heads": 8,
                "num_hidden_layers": 40,
            }
        }
        (meta_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

        prof_dir = tmp_path / "prof"
        golden_file = tmp_path / "golden.pt"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "smoke_e2e.py",
                "--meta-dir",
                str(meta_dir),
                "--smoke-dir",
                str(tmp_path / "smoke_run"),
                "--num-layers",
                "4",
                "--prompt-tokens",
                "16",
                "--max-tokens",
                "1",
                "--profile-dir",
                str(prof_dir),
                "--save-golden",
                str(golden_file),
            ],
        )

        mock_out = MagicMock()
        mock_out.token_ids = [42]
        mock_out.logprobs = [{42: SimpleNamespace(logprob=-0.1)}]
        mock_gen = SimpleNamespace(outputs=[mock_out])
        mock_llm_inst = MagicMock()
        mock_llm_inst.generate.return_value = [mock_gen]

        def mock_llm_ctor(*a, **kw):
            # Pre-create steps_dir under DSV41_SMOKE_LOGITS_DIR to test line 558 rmtree
            ldir = pathlib.Path(os.environ["DSV41_SMOKE_LOGITS_DIR"])
            (ldir / "steps").mkdir(parents=True, exist_ok=True)
            return mock_llm_inst

        with (
            patch("vllm.LLM", side_effect=mock_llm_ctor),
            patch("smoke_e2e_dsv41_tp8._install_deterministic_dummy_weights"),
            patch("smoke_e2e_dsv41_tp8._install_logits_capture"),
            patch(
                "smoke_e2e_dsv41_tp8._load_captured_step_logits",
                return_value=torch.ones((1, 32)),
            ),
            patch("scripts.dsv41_bench.common.summarize_xplanes", return_value={}),
            patch("scripts.dsv41_bench.common.print_xprof_table"),
        ):
            smoke.main()
            assert golden_file.exists()

        # Run with --verify-against-baseline and --speculative-config
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "smoke_e2e.py",
                "--meta-dir",
                str(meta_dir),
                "--smoke-dir",
                str(tmp_path / "smoke_run2"),
                "--prompt-tokens",
                "16",
                "--speculative-config",
                '{"method":"dspark","num_speculative_tokens":5}',
                "--verify-against-baseline",
                str(golden_file),
            ],
        )
        with (
            patch("vllm.LLM", return_value=mock_llm_inst) as mock_llm_cls,
            patch("smoke_e2e_dsv41_tp8._install_deterministic_dummy_weights"),
            patch("smoke_e2e_dsv41_tp8._install_logits_capture"),
            patch(
                "smoke_e2e_dsv41_tp8._load_captured_step_logits",
                return_value=torch.ones((1, 32)),
            ),
            patch("smoke_e2e_dsv41_tp8.verify_golden", return_value={"ok": True}),
        ):
            smoke.main()
            assert mock_llm_cls.call_args.kwargs["speculative_config"] == {
                "method": "dspark",
                "num_speculative_tokens": 5,
            }

        # Entrypoint via runpy: remove _REPO_ROOT from sys.path to cover line 58
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(_REPO_ROOT)])
        monkeypatch.setattr(sys, "argv", ["smoke_e2e_dsv41_tp8.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(_REPO_ROOT / "smoke_e2e_dsv41_tp8.py"), run_name="__main__"
            )
        assert exc_info.value.code == 0

    def test_smoke_active_hooks_exception(self, monkeypatch):
        # Covers lines 147-148
        monkeypatch.setenv("DSV41_SMOKE_ACTIVE", "1")
        with patch.dict(
            sys.modules, {"vllm.model_executor.model_loader.dummy_loader": None}
        ):
            runpy.run_path(str(_REPO_ROOT / "smoke_e2e_dsv41_tp8.py"))
