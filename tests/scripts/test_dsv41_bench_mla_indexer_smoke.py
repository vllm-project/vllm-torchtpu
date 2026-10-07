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
"""Unit tests for sparse MLA/SWA benchmark, indexer benchmark, and TP=8 smoke.

Targets:
1. scripts/dsv41_bench/bench_sparse_mla_swa.py
2. scripts/dsv41_bench/bench_indexer.py
3. smoke_e2e_dsv41_tp8.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Ensure CPU execution defaults before importing jax or TPU runtime libraries.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import numpy as np
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

import bench_indexer as bi  # noqa: E402
import bench_sparse_mla_swa as bs  # noqa: E402

import smoke_e2e_dsv41_tp8 as smoke  # noqa: E402

bi.common = common
bs.common = common
smoke.common = common

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (  # noqa: E402
    paged_indexer_jax,
    write_index_keys,
)


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


# ==============================================================================
# 2. Tests for scripts/dsv41_bench/bench_indexer.py
# ==============================================================================


class TestBenchIndexer:
    def test_build_indexer_inputs_branches(self):
        # 1. num_seqs > 1, candidate_blocks >= 4
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

        # 2. num_seqs == 1, candidate_blocks < 4 (covers branches 119->122, 127->130)
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

        # 3. phys_rows < tokens (covers lines 152-153)
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

        # 4. q_len > 0 with query_split="ceil" and query_split="pad16"
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
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.initialize_indexer"
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.paged_indexer",
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

            # Test verify=True with profile=None to cover lines 891->902 and 974->976
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

            # Test verify=True for individual subops
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

    def test_main_and_entrypoint(self, monkeypatch):
        with patch.object(common, "run_main") as mock_rm:
            bi.main()
            mock_rm.assert_called_once()

        monkeypatch.setattr(sys, "argv", ["bench_indexer.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(_BENCH_DIR / "bench_indexer.py"), run_name="__main__")
        assert exc_info.value.code == 0


# ==============================================================================
# 3. Tests for scripts/dsv41_bench/bench_sparse_mla_swa.py
# ==============================================================================


class TestBenchSparseMlaSwa:
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

        # Normal q_loc
        q_loc = bs._build_query_start_loc(torch, tokens, num_seqs=2)
        pool, rec, slots = bs._build_swa_inputs(torch, rng, tokens, window_size, q_loc)
        assert rec.shape == (tokens, window_size)

        # Decode q_len > 0 branch
        q_loc_dec = bs._build_query_start_loc(torch, 16, num_seqs=4, q_len=1)
        p_dec, r_dec, s_dec = bs._build_swa_inputs(
            torch, rng, 16, 512, q_loc_dec, q_len=1
        )
        assert r_dec.shape == (16, 512)
        assert (s_dec[:4] >= 0).all()
        assert (s_dec[4:] == -1).all()

        # q_loc with empty length (start == end) to cover line 114 (length <= 0)
        q_loc_empty = torch.tensor([0, 100, 100, 254], dtype=torch.int32)
        p2, r2, s2 = bs._build_swa_inputs(torch, rng, tokens, window_size, q_loc_empty)
        assert r2.shape == (tokens, window_size)

    def test_build_main_kv_inputs_branches(self):
        rng = torch.Generator(device="cpu").manual_seed(42)
        tokens = 256
        q_loc = bs._build_query_start_loc(torch, tokens, num_seqs=2)
        mp, rp, pgs, mrec = bs._build_main_kv_inputs(
            torch,
            rng,
            tokens=tokens,
            kv_len=1024,
            compress_ratio=2,
            page_size=16,
            topk=128,
            query_start_loc=q_loc,
            unique_per_chunk=256,
        )
        assert mrec.shape == (tokens, 128)

    def test_compact_records_torch_ref(self):
        # 1. width multiple of 128
        r1 = torch.randint(0, 100, (16, 128), dtype=torch.int32)
        c1 = bs._compact_records_torch_ref(torch, r1)
        assert c1.shape == (16, 128)

        # 2. width NOT multiple of 128 (covers line 225: padded > width)
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
            patch.object(common, "profile_steps", return_value=None),
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

            # num_seqs=2 under union_gather to cover branch 1410->1455
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
            for c_size, q_l, n_s in [(16, 1, 4), (128, 6, 4)]:
                args_dec = argparse.Namespace(
                    chunk_size=c_size,
                    q_len=q_l,
                    swa_width=512,
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

    def test_main_and_entrypoint(self, monkeypatch):
        with patch.object(common, "run_main") as mock_rm:
            bs.main()
            mock_rm.assert_called_once()

        monkeypatch.setattr(sys, "argv", ["bench_sparse_mla_swa.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(_BENCH_DIR / "bench_sparse_mla_swa.py"), run_name="__main__"
            )
        assert exc_info.value.code == 0


# ==============================================================================
# 4. Tests for 93e011ff9 benchmarking config scripts
# ==============================================================================
class TestBenchmarkingConfigs:
    """Unit tests verifying 100k prefill/decode benchmarking config scripts."""

    _CONFIG_DIR = _REPO_ROOT / "scripts" / "vllm" / "benchmarking" / "configs"
    _PREFILL_SCRIPT = "deepseek-v41-flash-tp8-ep-100k-prefill.sh"
    _DECODE_SCRIPT = "deepseek-v41-flash-tp8-ep-100k-decode.sh"

    @classmethod
    def _source_config(
        cls, script_name: str, env_overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        config_path = cls._CONFIG_DIR / script_name
        assert config_path.is_file(), f"Config script not found: {config_path}"

        bash_code = f"""
        source "{config_path}"
        vars=(
            MODEL TENSOR_PARALLELISM DATA_PARALLELISM ENABLE_EP
            MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS
            ASYNC_SCHEDULING GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE
            NUM_PROMPTS ISL_OSL_CONFIGS CONCURRENCY_OPTIONS
            RANDOM_RANGE_RATIO BENCHMARK_TEMPERATURE BENCHMARK_WARMUP_RUNS
            EVAL_TOLERANCE SERVER_READY_WAIT_MIN
            OMP_NUM_THREADS NEW_MODEL_DESIGN MODEL_IMPL_TYPE
            VLLM_USE_BREAKABLE_CUDAGRAPH VLLM_USE_AOT_COMPILE
            TPU_DEEPSEEK_V41_IMPL TPU_DEEPSEEK_V41_CED
            EXTRA_SERVE_ARGS LIBTPU_INIT_ARGS
        )
        for v in "${{vars[@]}}"; do
            echo "__VAR__:$v=${{!v}}"
        done
        """
        env = os.environ.copy()
        for k in (
            "DEEPSEEK_V41_MODEL",
            "KV_CACHE_DTYPE",
            "OMP_NUM_THREADS",
            "TPU_DEEPSEEK_V41_IMPL",
            "TPU_DEEPSEEK_V41_CED",
            "EXTRA_SERVE_ARGS",
            "LIBTPU_INIT_ARGS",
        ):
            env.pop(k, None)
        if env_overrides:
            env.update(env_overrides)
        res = subprocess.run(
            ["bash", "-c", bash_code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        result: dict[str, str] = {}
        for line in res.stdout.splitlines():
            if line.startswith("__VAR__:"):
                k, v = line[len("__VAR__:") :].split("=", 1)
                result[k] = v
        return result

    def test_config_scripts_exist_and_pass_syntax_check(self):
        for script_name in (self._PREFILL_SCRIPT, self._DECODE_SCRIPT):
            script_path = self._CONFIG_DIR / script_name
            assert script_path.is_file(), f"{script_name} must exist"
            assert script_path.stat().st_size > 0, f"{script_name} must not be empty"

            res = subprocess.run(
                ["bash", "-n", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            assert res.returncode == 0

    def test_deepseek_v41_flash_100k_prefill_default_config(self):
        cfg = self._source_config(self._PREFILL_SCRIPT)

        assert cfg["MODEL"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        assert cfg["TENSOR_PARALLELISM"] == "8"
        assert cfg["DATA_PARALLELISM"] == "1"
        assert cfg["ENABLE_EP"] == "true"
        assert cfg["MAX_MODEL_LEN"] == "103424"
        assert cfg["MAX_NUM_BATCHED_TOKENS"] == "2048"
        assert cfg["MAX_NUM_SEQS"] == "1"
        assert cfg["ASYNC_SCHEDULING"] == "true"
        assert cfg["GPU_MEMORY_UTILIZATION"] == "0.85"

        assert cfg["KV_CACHE_DTYPE"] == "fp8"

        assert cfg["NUM_PROMPTS"] == "4"
        assert cfg["ISL_OSL_CONFIGS"] == "100000:1"
        assert cfg["CONCURRENCY_OPTIONS"] == "1"
        assert cfg["RANDOM_RANGE_RATIO"] == "0.0"
        assert cfg["BENCHMARK_TEMPERATURE"] == "0"
        assert cfg["BENCHMARK_WARMUP_RUNS"] == "1"
        assert cfg["EVAL_TOLERANCE"] == "0.03"
        assert cfg["SERVER_READY_WAIT_MIN"] == "120"

        assert cfg["OMP_NUM_THREADS"] == "1"
        assert cfg["NEW_MODEL_DESIGN"] == "1"
        assert cfg["MODEL_IMPL_TYPE"] == "vllm"
        assert cfg["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
        assert cfg["VLLM_USE_AOT_COMPILE"] == "0"
        assert cfg["TPU_DEEPSEEK_V41_IMPL"] == "v2"
        assert cfg["TPU_DEEPSEEK_V41_CED"] == "1"

        extra_args = cfg["EXTRA_SERVE_ARGS"]
        assert "--no-enable-prefix-caching" in extra_args
        assert "--enable-ep-weight-filter" in extra_args
        assert "--safetensors-load-strategy=prefetch" in extra_args
        assert "--safetensors-prefetch-num-threads=8" in extra_args
        assert "--block-size=16" in extra_args
        assert "--kv-cache-memory-bytes=1342177280" in extra_args
        assert '--compilation-config={"compile_sizes":[2,2048]}' in extra_args

        assert (
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
            in cfg["LIBTPU_INIT_ARGS"]
        )

    def test_deepseek_v41_flash_100k_prefill_env_overrides(self):
        overrides = {
            "DEEPSEEK_V41_MODEL": "custom-org/Custom-V41",
            "KV_CACHE_DTYPE": "auto",
            "OMP_NUM_THREADS": "4",
            "TPU_DEEPSEEK_V41_IMPL": "v1",
            "TPU_DEEPSEEK_V41_CED": "0",
            "EXTRA_SERVE_ARGS": "--test-flag",
            "LIBTPU_INIT_ARGS": (
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1024"
            ),
        }
        cfg = self._source_config(self._PREFILL_SCRIPT, env_overrides=overrides)

        assert cfg["MODEL"] == "custom-org/Custom-V41"
        assert cfg["KV_CACHE_DTYPE"] == "auto"
        assert cfg["OMP_NUM_THREADS"] == "4"
        assert cfg["TPU_DEEPSEEK_V41_IMPL"] == "v1"
        assert cfg["TPU_DEEPSEEK_V41_CED"] == "0"
        assert cfg["EXTRA_SERVE_ARGS"].startswith("--test-flag ")
        assert (
            cfg["LIBTPU_INIT_ARGS"]
            == "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1024"
        )

        cfg_append = self._source_config(
            self._PREFILL_SCRIPT,
            env_overrides={"LIBTPU_INIT_ARGS": "--xla_tpu_enable_llo_profiling=true"},
        )
        assert (
            cfg_append["LIBTPU_INIT_ARGS"].strip()
            == "--xla_tpu_enable_llo_profiling=true"
            " --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
        )

    def test_deepseek_v41_flash_100k_decode_default_config(self):
        cfg = self._source_config(self._DECODE_SCRIPT)

        assert cfg["MODEL"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        assert cfg["TENSOR_PARALLELISM"] == "8"
        assert cfg["DATA_PARALLELISM"] == "1"
        assert cfg["ENABLE_EP"] == "true"
        assert cfg["MAX_MODEL_LEN"] == "103424"
        assert cfg["MAX_NUM_BATCHED_TOKENS"] == "2048"
        assert cfg["MAX_NUM_SEQS"] == "16"
        assert cfg["ASYNC_SCHEDULING"] == "true"
        assert cfg["GPU_MEMORY_UTILIZATION"] == "0.85"

        assert cfg["KV_CACHE_DTYPE"] == "fp8"

        assert cfg["ISL_OSL_CONFIGS"] == "100000:2048"
        assert cfg["CONCURRENCY_OPTIONS"] == "4 8 16"
        assert cfg["RANDOM_RANGE_RATIO"] == "0.0"
        assert cfg["BENCHMARK_TEMPERATURE"] == "0"
        assert cfg["EVAL_TOLERANCE"] == "0.03"
        assert cfg["SERVER_READY_WAIT_MIN"] == "120"

        assert cfg["OMP_NUM_THREADS"] == "1"
        assert cfg["NEW_MODEL_DESIGN"] == "1"
        assert cfg["MODEL_IMPL_TYPE"] == "vllm"
        assert cfg["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
        assert cfg["VLLM_USE_AOT_COMPILE"] == "0"
        assert cfg["TPU_DEEPSEEK_V41_IMPL"] == "v2"
        assert cfg["TPU_DEEPSEEK_V41_CED"] == "1"

        extra_args = cfg["EXTRA_SERVE_ARGS"]
        assert "--no-enable-prefix-caching" in extra_args
        assert "--enable-ep-weight-filter" in extra_args
        assert "--safetensors-load-strategy=prefetch" in extra_args
        assert "--safetensors-prefetch-num-threads=8" in extra_args
        assert "--block-size=16" in extra_args
        assert "--kv-cache-memory-bytes=21474836480" in extra_args
        assert '--compilation-config={"compile_sizes":[4,8,16,2048]}' in extra_args

        assert (
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
            in cfg["LIBTPU_INIT_ARGS"]
        )

    def test_deepseek_v41_flash_100k_decode_env_overrides(self):
        overrides = {
            "DEEPSEEK_V41_MODEL": "custom-org/Custom-Decode-V41",
            "KV_CACHE_DTYPE": "bf16",
            "EXTRA_SERVE_ARGS": "--decode-extra-flag",
            "LIBTPU_INIT_ARGS": (
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=512"
            ),
        }
        cfg = self._source_config(self._DECODE_SCRIPT, env_overrides=overrides)

        assert cfg["MODEL"] == "custom-org/Custom-Decode-V41"
        assert cfg["KV_CACHE_DTYPE"] == "bf16"
        assert cfg["EXTRA_SERVE_ARGS"].startswith("--decode-extra-flag ")
        assert (
            cfg["LIBTPU_INIT_ARGS"]
            == "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=512"
        )

    def test_config_script_static_invariants(self):
        for script_name in (self._PREFILL_SCRIPT, self._DECODE_SCRIPT):
            content = (self._CONFIG_DIR / script_name).read_text(encoding="utf-8")
            assert content.startswith("#!/bin/bash\n")
            assert "# shellcheck disable=SC2034" in content
            assert 'TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"' in content
            assert 'TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"' in content
            assert (
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes" in content
            )
