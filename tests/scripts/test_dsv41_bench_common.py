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
"""CPU tests for scripts/dsv41_bench/common.py verification and roofline."""

import importlib.util
import math
import pathlib
import sys

import pytest
import torch

pytestmark = pytest.mark.cpu_test

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "dsv41_bench"
    / "common.py"
)
_spec = importlib.util.spec_from_file_location("dsv41_bench_common", _SCRIPT)
common = importlib.util.module_from_spec(_spec)
sys.modules["dsv41_bench_common"] = common
_spec.loader.exec_module(common)


def test_exact_distinguishes_signed_zero_and_nan_payload():
    a = torch.tensor([0.0, 1.0, float("nan")])
    b = torch.tensor([-0.0, 1.0, float("nan")])
    assert not common.compare_tensors(a, b, common.EXACT)["ok"]
    assert common.compare_tensors(a, a.clone(), common.EXACT)["ok"]


def test_exact_rejects_dtype_and_shape_mismatch():
    a = torch.zeros(4, dtype=torch.bfloat16)
    assert not common.compare_tensors(a, a.float(), common.EXACT)["ok"]
    assert not common.compare_tensors(a, a[:3], common.EXACT)["ok"]


def test_lossless_thresholds_and_nonfinite_mask():
    torch.manual_seed(0)
    e = torch.randn(1024, 64)
    close = e + 1e-5 * torch.randn_like(e)
    m = common.compare_tensors(close, e, common.LOSSLESS)
    assert m["ok"] and m["cos_sim"] > 0.9999 and m["rel_l2"] < 1e-3
    far = e + 0.1 * torch.randn_like(e)
    assert not common.compare_tensors(far, e, common.LOSSLESS)["ok"]
    masked = e.clone()
    masked[0, 0] = float("inf")
    m = common.compare_tensors(masked, e, common.LOSSLESS)
    assert not m["nonfinite_mask_equal"] and not m["ok"]


def test_lossless_rejects_nan_and_vacuous_comparisons():
    neg_inf = torch.tensor([float("-inf"), 1.0, 2.0])
    assert common.compare_tensors(neg_inf, neg_inf.clone(), common.LOSSLESS)["ok"]
    nan = torch.full((4, 3), float("nan"))
    m = common.compare_tensors(nan, nan.clone(), common.LOSSLESS)
    assert m["nonfinite_mask_equal"] and not m["ok"]
    inf = torch.full((4,), float("-inf"))
    m = common.compare_tensors(inf, inf.clone(), common.LOSSLESS)
    assert m["num_compared"] == 0 and not m["ok"]


def test_topk_overlap_ignores_padding_and_order():
    expected = torch.tensor([[3, 1, 2, -1], [5, 6, 7, 8], [-1, -1, -1, -1]])
    actual = torch.tensor([[2, 3, 1, -1], [8, 7, 6, 9], [-1, -1, -1, -1]])
    mean, worst = common.topk_overlap(actual, expected)
    assert worst == pytest.approx(0.75)
    assert mean == pytest.approx((1.0 + 0.75 + 1.0) / 3)
    m = common.compare_tensors(
        actual, expected, common.MatchCriteria("topk", min_topk_overlap=0.95)
    )
    assert not m["ok"]


def test_topk_empty_expected_row_requires_empty_actual():
    expected = torch.tensor([[-1, -1]])
    assert common.topk_overlap(torch.tensor([[-1, -1]]), expected) == (1.0, 1.0)
    assert common.topk_overlap(torch.tensor([[4, -1]]), expected) == (0.0, 0.0)


def test_verify_outputs_raises_with_names():
    a = {"x": torch.ones(2)}
    with pytest.raises(common.VerificationError, match="output names differ"):
        common.verify_outputs(a, {"y": torch.ones(2)}, common.EXACT)
    with pytest.raises(common.VerificationError, match='"x"'):
        common.verify_outputs(a, {"x": torch.zeros(2)}, common.EXACT)
    report = common.verify_outputs(a, {"x": torch.ones(2)}, common.EXACT)
    assert report["x"]["ok"]


def test_golden_roundtrip(tmp_path):
    path = common.golden_path(tmp_path, "case", 3)
    t = {"x": torch.arange(6, dtype=torch.bfloat16)}
    common.save_golden(path, t, {"git_sha": "abc"})
    got, meta = common.load_golden(path)
    assert meta["git_sha"] == "abc"
    assert common.verify_outputs(got, t, common.EXACT)["x"]["bitwise_equal"]
    with pytest.raises(FileNotFoundError, match="--update-golden"):
        common.load_golden(tmp_path / "missing.pt")


def test_roofline_collectives_match_ring_formulas():
    hw = common.Tpu7xRoofline()
    nbytes = 2048 * 5120 * 2
    assert hw.all_reduce_s(nbytes, 8) == pytest.approx(2 * 7 / 8 * nbytes / 186.68e9)
    assert hw.reduce_scatter_s(nbytes, 8) * 2 == pytest.approx(
        hw.all_reduce_s(nbytes, 8)
    )
    est = common.RooflineEstimate("mm", flops=2 * 2048 * 5120 * 5120, hbm_bytes=nbytes)
    rep = est.report(hw, measured_ms=1.0)
    assert rep["lower_bound_ms"] == pytest.approx(
        max(est.flops / 980.48e12, nbytes / (6273e9 / 2)) * 1e3
    )
    assert 0 < rep["roofline_efficiency"] < 1
    assert math.isinf(
        common.RooflineEstimate("c", comm_s=1.0).bound_s(hw)["arith_intensity"]
    )


def test_geometry_from_hf_config(tmp_path):
    import dataclasses
    import json

    geo = common.Dsv41Geometry()
    text = {
        f.name: list(v) if isinstance(v := getattr(geo, f.name), tuple) else v
        for f in dataclasses.fields(geo)
    }
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"text_config": text}))
    assert common.Dsv41Geometry.from_hf_config(cfg) == geo
    assert geo.heads_per_rank(8) == 8
    assert geo.candidate_positions == 16384
    del text["index_topk"]
    cfg.write_text(json.dumps({"text_config": text}))
    with pytest.raises(KeyError, match="index_topk"):
        common.Dsv41Geometry.from_hf_config(cfg)
