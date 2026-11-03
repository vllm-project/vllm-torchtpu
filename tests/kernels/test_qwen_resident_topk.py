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
"""Qwen-owned resident resource and native score-prefix contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_dsv4_sparsecore_topk import _mismatch
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import MAX_SLICE_WORDS
from vllm_torchtpu.kernels.qwen4_exp import resident_topk as resident_module
from vllm_torchtpu.kernels.qwen4_exp.resident_topk import (
    qwen_sparsecore_topk as sparsecore_topk,
)


def test_resident_prefix_budget_keeps_short_rows_and_reserves_buffers():
    """No extra variant for short inputs; both score ABIs fit named buffers."""
    for return_scores in (False, True):
        for k in (512, 4096):
            args = dict(
                return_scores=return_scores, capacity_bytes=512 * 1024,
                num_lanes=16, dma_granule_bytes=32, pipeline_lanes=128,
            )
            assert resident_module._resident_prefix_words(33, 2048, k, **args) is None
            resident = resident_module._resident_prefix_words(33, 65792, k, **args)
            assert resident == MAX_SLICE_WORDS
            small = resident_module._resident_prefix_words(
                33, 65792, k, **dict(args, capacity_bytes=256 * 1024)
            )
            assert small is not None and small < resident
            assert small >= k
    # A positive resident extent smaller than k cannot implement top-k. Keep
    # the original full path, including its existing k > physical-n behavior.
    assert resident_module._resident_prefix_words(
        3, 65792, 4096, return_scores=True, capacity_bytes=128 * 1024,
        num_lanes=16, dma_granule_bytes=32, pipeline_lanes=128,
    ) is None



@pytest.mark.tpu_test
@pytest.mark.parametrize("b,n", [(33, 65792), (33, 32784), (3, 65792)])
@pytest.mark.parametrize(
    "raw_bits,return_scores",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_resident_prefix_pitched_rows_and_full_fallback(b, n, raw_bits, return_scores):
    """An out-of-prefix winner must force fallback, not disappear in padding."""
    k = 512
    rng = np.random.default_rng(681103)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    resident = resident_module._resident_prefix_words(
        b, n, k, return_scores=return_scores, capacity_bytes=512 * 1024,
        num_lanes=16, dma_granule_bytes=32, pipeline_lanes=128,
    )
    assert resident is not None
    lengths = np.resize(
        np.array([0, -1, 15, resident, resident + 1, n, n + 16], np.int32), b
    )
    if b > 4:
        scores[4, resident] = 1000
        scores[3, resident - 1] = 1000
    if b > 7:
        scores[7] = -np.inf
        scores[8] = rng.integers(0, 17, n).astype(np.float32)
    data = jnp.asarray(scores.view(np.int32) if raw_bits else scores)
    frozen = np.asarray(data).tobytes()
    for lens in (np.minimum(lengths, resident), lengths):
        result = jax.block_until_ready(sparsecore_topk(
            data, k, jnp.asarray(lens), return_scores=return_scores,

        ))
        idx = np.asarray(result[0] if return_scores else result)
        effective = np.clip(lens, 0, n)
        assert _mismatch(scores, idx, k, effective) is None
        if b > 4 and lens[4] > resident:
            assert resident in idx[4]
        if return_scores:
            bits = np.full(idx.shape, sc_module.BITS_NEG_INF, np.int32)
            for row in range(b):
                valid = idx[row] >= 0
                bits[row, valid] = scores.view(np.int32)[row, idx[row, valid]]
            np.testing.assert_array_equal(result[1], bits)
        repeat = jax.block_until_ready(sparsecore_topk(
            data, k, jnp.asarray(lens), return_scores=return_scores,

        ))
        repeat_idx = np.asarray(repeat[0] if return_scores else repeat)
        assert _mismatch(scores, repeat_idx, k, effective) is None
        assert np.asarray(data).tobytes() == frozen
    if b == 33 and n == 65792 and not raw_bits:
        default = np.asarray(sparsecore_topk(data, k))
        assert _mismatch(scores, default, k, np.full(b, n, np.int32)) is None
        assert resident in default[4]
        bad = default.copy()
        bad[4, 0] = bad[4, 1]
        assert _mismatch(scores, bad, k, np.full(b, n, np.int32)) is not None



@pytest.mark.tpu_test
def test_resident_prefix_keeps_one_caller_scheduling_boundary():
    """Runtime extent dispatch must not wrap SC calls in an HLO conditional."""
    compiled = sparsecore_topk.lower(
        jnp.zeros((33, 65792), jnp.float32), 512,
        row_lengths=jnp.full((33,), 32, jnp.int32),
        scheduling_group_id=9,
    ).compile()
    lines = compiled.as_text().splitlines()
    calls = [line for line in lines
             if "custom-call(" in line and "sc_topk_direct" in line]
    assert len(calls) == 1
    assert '_scheduling_group_id="9"' in calls[0]
    assert not any("conditional(" in line for line in lines)



@pytest.mark.tpu_test
def test_resident_prefix_big_k_returned_bits(tmp_path):
    """Exercise the admitted cooperative prefix, including returned raw bits."""
    import hashlib
    import json

    b, n, k = 3, 65792, 4096
    resident = resident_module._resident_prefix_words(
        b, n, k, return_scores=True, capacity_bytes=512 * 1024,
        num_lanes=16, dma_granule_bytes=32, pipeline_lanes=128,
    )
    assert resident is not None and resident >= k
    lengths = np.array([resident - 1, resident, resident - 16], np.int32)
    assert np.all(np.clip(lengths, 0, n) <= resident)
    rng = np.random.default_rng(691103)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    scores[:, resident - 17] = 1000
    scores[:, resident:] = 2000  # Ineligible, despite outranking every winner.
    raw = scores.view(np.int32).copy()
    data, lens = jnp.asarray(raw), jnp.asarray(lengths)
    frozen = raw.tobytes(), lengths.tobytes()
    compiled = sparsecore_topk.lower(
        data, k, row_lengths=lens, return_scores=True,
    ).compile()
    hlo = compiled.as_text()
    calls = [line for line in hlo.splitlines()
             if "custom-call(" in line and "sc_topk_direct" in line]
    assert len(calls) == 1 and f"resident{resident}" in calls[0]
    for _ in range(2):
        idx, bits = jax.block_until_ready(compiled(data, row_lengths=lens))
        idx, bits = np.asarray(idx), np.asarray(bits)
        assert _mismatch(scores, idx, k, lengths) is None
        np.testing.assert_array_equal(bits, np.take_along_axis(raw, idx, axis=1))
        assert np.all(np.any(idx == resident - 17, axis=1))
        assert np.asarray(data).tobytes() == frozen[0]
        assert np.asarray(lens).tobytes() == frozen[1]
    bad = idx.copy()
    bad[0, 0] = bad[0, 1]
    assert _mismatch(scores, bad, k, lengths) is not None
    (tmp_path / "bigk.hlo.txt").write_text(hlo)
    receipt = dict(
        B=b, physical_N=n, k=k, resident_N=resident, return_scores=True,
        resident_statically_enabled=True, runtime_prefix_predicate=True,
        row_lengths=lengths.tolist(), repeats=2, independent_oracle=True,
        returned_bits_exact=True, input_readonly=True, negative_control=True,
        raw_input_SHA256=hashlib.sha256(raw.tobytes()).hexdigest(),
        HLO_SHA256=hashlib.sha256(hlo.encode()).hexdigest(), custom_call=calls[0],
    )
    (tmp_path / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("BIGK_RECEIPT", tmp_path / "receipt.json", json.dumps(receipt))
