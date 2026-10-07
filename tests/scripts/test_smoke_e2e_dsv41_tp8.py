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
"""CPU unit tests for smoke_e2e_dsv41_tp8.py."""

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_test

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "smoke_e2e_dsv41_tp8.py"
_spec = importlib.util.spec_from_file_location("smoke_e2e_dsv41_tp8", _SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
sys.modules["smoke_e2e_dsv41_tp8"] = smoke
_spec.loader.exec_module(smoke)


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


def test_build_smoke_model_dir_4l_and_6l(tmp_path):
    meta_dir = tmp_path / "meta"
    _write_fake_meta_dir(meta_dir)

    out_4l = smoke.build_smoke_model_dir(meta_dir, tmp_path / "smoke_4l", num_layers=4)
    cfg_4l = json.loads((out_4l / "config.json").read_text(encoding="utf-8"))[
        "text_config"
    ]
    assert cfg_4l["num_hidden_layers"] == 4
    assert cfg_4l["compress_ratios"] == [0, 2, 1, 1]
    assert cfg_4l["kv_source_layer_ids"] == [1, 2]
    assert cfg_4l["index_source_layer_ids"] == [1, 2, 3]
    assert cfg_4l["candidate_source_layer_id"] == 2
    assert cfg_4l["engram_layer_ids"] == [1]
    assert (out_4l / "sitecustomize.py").exists()
    assert (out_4l / "usercustomize.py").exists()

    out_6l = smoke.build_smoke_model_dir(meta_dir, tmp_path / "smoke_6l", num_layers=6)
    cfg_6l = json.loads((out_6l / "config.json").read_text(encoding="utf-8"))[
        "text_config"
    ]
    assert cfg_6l["num_hidden_layers"] == 6
    assert cfg_6l["compress_ratios"] == [0, 2, 2, 1, 1, 1]
    assert cfg_6l["kv_source_layer_ids"] == [1, 3]
    assert cfg_6l["index_source_layer_ids"] == [1, 3, 4]
    assert cfg_6l["candidate_source_layer_id"] == 3


def test_extract_top_logprobs_and_verify_golden(tmp_path):
    from scripts.dsv41_bench import common

    step_lps = [
        {10: SimpleNamespace(logprob=-0.1), 20: SimpleNamespace(logprob=-1.5)},
        {30: SimpleNamespace(logprob=-0.2), 40: SimpleNamespace(logprob=-2.0)},
    ]
    ids, vals = smoke._extract_top_logprobs(step_lps)
    assert ids.tolist() == [[10, 20], [30, 40]]

    torch.manual_seed(0)
    logits = torch.randn(4, 128)
    logits[:, 10] = 5.0
    golden_payload = {
        "token_ids": torch.tensor([10, 30, 15, 25], dtype=torch.int64),
        "top_logprob_ids": ids,
        "top_logprob_values": vals,
        "step_logits": logits,
    }
    golden_path = tmp_path / "golden.pt"
    torch.save(golden_payload, golden_path)

    report = smoke.verify_golden(golden_payload, golden_path)
    assert report["token_ids"]["ok"]
    assert report["step_logits"]["ok"]
    assert report["step_logits"]["top1_match_rate"] == 1.0

    # Perturb top-1 logit -> must fail verify_golden
    bad_logits = logits.clone()
    bad_logits[0, 99] = 10.0
    bad_payload = {**golden_payload, "step_logits": bad_logits}
    with pytest.raises(common.VerificationError, match="top1_match_rate"):
        smoke.verify_golden(bad_payload, golden_path)

    # Empty step_logits -> must fail verify_golden
    empty_payload = {
        **golden_payload,
        "step_logits": torch.empty((0, 0), dtype=torch.float32),
    }
    with pytest.raises(common.VerificationError, match="step_logits is empty"):
        smoke.verify_golden(empty_payload, golden_path)


def test_build_operator_shape_ledger():
    ledger = smoke.build_operator_shape_ledger(
        num_layers=4, batch_tokens=2048, kv_len=90112
    )
    assert ledger["num_layers"] == 4
    assert ledger["encoder_tokens"] == 2048
    assert ledger["kv_len"] == 90112
    assert ledger["layers"]["L0_swa_only"]["mhc_pre_delayed"] == [2048, 4, 5120]
    assert ledger["layers"]["L1_c2_engram_indexer_full"]["indexer_k_compressed"] == [
        45056,
        128,
    ]
    assert (
        ledger["layers"]["L3_c1_reuse_candidate_filter"]["candidate_tokens_expanded"]
        == 16384
    )


def test_deterministic_single_dummy_weight_handles_requires_grad_parameter():
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    smoke._install_deterministic_dummy_weights()
    p = torch.nn.Parameter(torch.empty((8, 16), dtype=torch.bfloat16))
    assert p.requires_grad
    weight_utils.initialize_single_dummy_weight(p)
    assert torch.isfinite(p).all()

    p_val = torch.nn.Parameter(torch.empty((4, 4), dtype=torch.float32))
    p_val.dummy_weight_value = 0.25
    weight_utils.initialize_single_dummy_weight(p_val)
    assert torch.all(p_val == 0.25)


def test_extract_top_logprobs_truncates_extra_sampled_token_and_loads_non_zero_steps(
    tmp_path,
):
    step_lps = [
        {10: SimpleNamespace(logprob=-0.1), 20: SimpleNamespace(logprob=-1.5)},
        {
            30: SimpleNamespace(logprob=-0.2),
            40: SimpleNamespace(logprob=-2.0),
            99: SimpleNamespace(logprob=float("nan")),
        },
    ]
    ids, vals = smoke._extract_top_logprobs(step_lps, top_k=2)
    assert ids.shape == (2, 2)
    assert vals.shape == (2, 2)
    assert ids.tolist() == [[10, 20], [30, 40]]

    steps_dir = tmp_path / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    # Simulate CED=1 partial-prefill zero logits followed by final non-zero logits
    torch.save(torch.zeros((1, 32), dtype=torch.float32), steps_dir / "step_0002.pt")
    torch.save(torch.zeros((1, 32), dtype=torch.float32), steps_dir / "step_0044.pt")
    real_logits = torch.arange(32, dtype=torch.float32).unsqueeze(0) + 1.0
    torch.save(real_logits, steps_dir / "step_0045.pt")

    loaded = smoke._load_captured_step_logits(steps_dir, max_steps=1)
    assert loaded.shape == (1, 32)
    assert torch.equal(loaded[0], real_logits[0])


def test_verify_golden_tied_argmax_and_tp_shifted_logprobs(tmp_path):
    from scripts.dsv41_bench import common

    logits = torch.zeros((1, 64), dtype=torch.float32)
    # Tokens 11 and 59 share the exact maximum logit
    logits[0, 11] = 8.44e-5
    logits[0, 59] = 8.44e-5
    golden_payload = {
        "token_ids": torch.tensor([59], dtype=torch.int64),
        "top_logprob_ids": torch.tensor([[59, 11]], dtype=torch.int64),
        "top_logprob_values": torch.tensor([[-11.7697, -11.7697]], dtype=torch.float32),
        "step_logits": logits,
    }
    golden_path = tmp_path / "golden_tied.pt"
    torch.save(golden_payload, golden_path)

    # Device greedy sampler picks lowest tied index 11 (= argmax) and shifted logprobs
    actual_payload = {
        "token_ids": torch.tensor([11], dtype=torch.int64),
        "top_logprob_ids": torch.tensor([[11, 59]], dtype=torch.int64),
        "top_logprob_values": torch.tensor([[-13.1644, -13.1644]], dtype=torch.float32),
        "step_logits": logits.clone(),
    }
    report = smoke.verify_golden(actual_payload, golden_path)
    assert report["step_logits"]["ok"]
    assert report["token_ids"]["ok"]
    assert report["token_ids"]["tied_argmax_match"]
    assert report["top_logprob_values"]["ok"]
    assert report["top_logprob_values"]["shift_invariant_match"]

    # Non-maximum token_id must still fail
    bad_actual = {
        **actual_payload,
        "token_ids": torch.tensor([7], dtype=torch.int64),
    }
    with pytest.raises(common.VerificationError):
        smoke.verify_golden(bad_actual, golden_path)
