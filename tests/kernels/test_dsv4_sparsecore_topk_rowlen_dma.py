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
"""Tests for `TPU_SC_TOPK_ROWLEN_DMA` in the SparseCore top-k.

With the flag on, the cooperative path (`p > 1`) of `sparsecore_topk` copies
each score slice from HBM only up to the row length, rounded up to
`_dma_chunk_words`, instead of the whole slice. The selection must not change.

The TPU tests run the kernel with the flag off and on and require the same
indices and scores bit for bit. Positions at or past the row length hold a
large finite score, so a copy that is too short (stale words read as scores)
or a mask that is wrong (padding selected) changes the result. The CPU tests
check the DMA size helper and the flag plumbing; SparseCore kernels cannot
run on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_module
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    LANES,
    _dma_chunk_words,
    _pick_partition,
    sparsecore_topk,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_64K = 65536  # DSA indexer score row at --max-model-len 1,048,576 with DCP16

# (b, n, k, expected p). b16 n64K is the decode shape; b1024 has many waves,
# as the mixed step does; n133120 has p = 8 and a slice length (16,640) whose
# DMA size is 256, not 2,048.
SHAPES = [
    pytest.param(16, N_64K, 2048, 2, id="p2-b16-n65536"),
    pytest.param(1024, N_64K, 2048, 2, id="p2-b1024-n65536"),
    pytest.param(16, 133120, 2048, 8, id="p8-b16-n133120"),
]


def _row_lengths(b: int, n: int, p: int) -> np.ndarray:
    """Lengths at the DMA chunk and slice edges, cycled over the rows."""
    slice_len = n // p
    chunk = _dma_chunk_words(slice_len)
    edges = sorted(
        {
            min(e, n)
            for e in (
                0,
                1,
                LANES - 1,
                LANES,
                LANES + 1,
                chunk - 1,
                chunk,
                chunk + 1,
                2047,
                2048,
                2836,
                slice_len - 1,
                slice_len,
                slice_len + 1,
                slice_len + chunk + 3,
                n - 1,
                n,
            )
        }
    )
    return np.array([edges[i % len(edges)] for i in range(b)], np.int32)


def _scores(b: int, n: int, row_lengths: np.ndarray) -> np.ndarray:
    """Random scores, with 1e30 at every position at or past the row length."""
    rng = np.random.default_rng(20261123)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    past_end = np.arange(n)[None, :] >= row_lengths[:, None]
    scores[past_end] = 1e30
    return scores


@pytest.mark.tpu_test
@pytest.mark.parametrize("return_scores", [False, True])
@pytest.mark.parametrize("b,n,k,expected_p", SHAPES)
def test_rowlen_dma_is_bit_identical(b, n, k, expected_p, return_scores):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the SparseCore top-k requires a TPU")
    assert _pick_partition(b, n, k) == expected_p
    row_lengths = _row_lengths(b, n, expected_p)
    scores = jnp.asarray(_scores(b, n, row_lengths))
    lengths = jnp.asarray(row_lengths)

    def run(rowlen_dma):
        out = sparsecore_topk(
            scores, k, lengths, return_scores=return_scores, rowlen_dma=rowlen_dma
        )
        return jax.tree.map(np.asarray, jax.block_until_ready(out))

    want, got = run(False), run(True)
    for w, g in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        np.testing.assert_array_equal(g, w)
    idx = got[0] if return_scores else got
    # No index at or past the row length: the 1e30 padding was never selected.
    assert not np.any(idx >= row_lengths[:, None])
    # The rows are full where the row length allows it.
    np.testing.assert_array_equal(
        (idx >= 0).sum(axis=1), np.minimum(row_lengths, k)
    )


@pytest.mark.parametrize(
    "slice_len,chunk",
    [(32768, 2048), (65536, 2048), (16640, 256), (400, 16), (2048, 2048), (16, 16)],
)
def test_dma_chunk_divides_the_slice(slice_len, chunk):
    assert _dma_chunk_words(slice_len) == chunk
    assert slice_len % chunk == 0 and chunk % LANES == 0


def _capture_direct(monkeypatch):
    seen = {}

    def fake_direct(scores, k, row_lengths, **kwargs):
        seen.update(kwargs)
        return jnp.zeros((scores.shape[0], k), jnp.int32)

    monkeypatch.setattr(sc_module, "_sc_topk_direct", fake_direct)
    return seen


@pytest.mark.parametrize("env,expected", [("0", False), ("1", True)])
def test_env_selects_rowlen_dma(monkeypatch, env, expected):
    """`rowlen_dma=None` reads the env at trace time. The undecorated function
    is called, because the jit cache would keep the first trace."""
    seen = _capture_direct(monkeypatch)
    monkeypatch.setenv("TPU_SC_TOPK_ROWLEN_DMA", env)
    sparsecore_topk.__wrapped__(jnp.zeros((16, N_64K), jnp.float32), 2048)
    assert seen["rowlen_dma"] is expected and seen["p"] == 2


def test_explicit_argument_overrides_env(monkeypatch):
    seen = _capture_direct(monkeypatch)
    monkeypatch.setenv("TPU_SC_TOPK_ROWLEN_DMA", "1")
    sparsecore_topk.__wrapped__(
        jnp.zeros((16, N_64K), jnp.float32), 2048, rowlen_dma=False
    )
    assert seen["rowlen_dma"] is False


def test_rowlen_dma_env_defaults_to_false(monkeypatch):
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    monkeypatch.delenv("TPU_SC_TOPK_ROWLEN_DMA", raising=False)
    assert envs.environment_variables["TPU_SC_TOPK_ROWLEN_DMA"]() is False
    assert "TPU_SC_TOPK_ROWLEN_DMA" in TpuPlatform.additional_env_vars
