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
"""Shared default and Qwen opt-in selection on pitched, padded raw scores."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.kernels.test_dsv4_sparsecore_topk import _mismatch
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)
from vllm_torchtpu.kernels.qwen4_exp.resident_topk import QwenSparseCoreTopK
from vllm_torchtpu.kernels.qwen4_exp import resident_topk as resident_module


@pytest.mark.tpu_test
@pytest.mark.parametrize("qwen", [False, True])
@pytest.mark.parametrize(
    "b,n,k,raw,returned",
    [(33, 65792, 512, False, False),
     (33, 32784, 512, True, True),
     (3, 65792, 4096, True, True)],
)
def test_public_policy_pitched_rows_and_fallback(
    tmp_path, qwen, b, n, k, raw, returned
):
    rng = np.random.default_rng(721103)
    scores = rng.integers(-8, 9, (b, n)).astype(np.float32)
    scores[0] = -np.inf
    resident = resident_module._resident_prefix_words(
        b, n, k, return_scores=returned, capacity_bytes=512 * 1024,
        num_lanes=16, dma_granule_bytes=32, pipeline_lanes=128,
    )
    assert resident is not None
    lengths = np.resize(np.array([0, resident, resident - 1], np.int32), b)
    scores[1:, resident - 17] = 1000
    scores[:, resident:] = 2000
    data = jnp.asarray(scores.view(np.int32) if raw else scores)
    frozen = np.asarray(data).tobytes()
    adapter = QwenSparseCoreTopK() if qwen else SparseCoreTopK()

    def select(values, lens):
        return adapter.select(
            values, k, lens, scheduling_group_id=9, return_scores=returned
        )

    lens = jnp.asarray(lengths)
    compiled = jax.jit(select).lower(data, lens).compile()
    hlo = compiled.as_text()
    (tmp_path / "selector.hlo.txt").write_text(hlo)
    calls = [line for line in hlo.splitlines()
             if "custom-call(" in line and "sc_topk_direct" in line]
    assert len(calls) == 1
    assert ("_resident" in calls[0]) is qwen
    assert '_scheduling_group_id="9"' in calls[0]
    assert not any("conditional(" in line for line in hlo.splitlines())
    for actual_lengths in (lengths, np.full(b, n, np.int32)):
        for _ in range(2):
            result = jax.block_until_ready(compiled(data, jnp.asarray(actual_lengths)))
            ids = np.asarray(result[0] if returned else result)
            assert _mismatch(scores, ids, k, actual_lengths) is None
            if returned:
                bits = np.full(ids.shape, sc.BITS_NEG_INF, np.int32)
                valid = ids >= 0
                expected = np.take_along_axis(
                    scores.view(np.int32), np.maximum(ids, 0), axis=1
                )
                bits[valid] = expected[valid]
                np.testing.assert_array_equal(result[1], bits)
            assert np.asarray(data).tobytes() == frozen
        bad = ids.copy()
        bad[1, 0] = bad[1, 1]
        assert _mismatch(scores, bad, k, actual_lengths) is not None
