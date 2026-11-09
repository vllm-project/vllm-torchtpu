# SPDX-License-Identifier: Apache-2.0
"""The one-hot table lookups equal the gather / bucketize / repeat they replace."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import _requests
from vllm_torchtpu.layers.core import table_lookup


def _starts(rng, requests, tokens, first=0):
    counts = rng.integers(0, 2 * tokens // requests + 2, requests)
    counts[rng.random(requests) < 0.2] = 0
    starts = np.concatenate(([first], first + np.cumsum(counts)))
    return np.minimum(starts, first + tokens)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("requests,tokens", [(48, 288), (48, 240), (1, 7), (5, 64)])
def test_torch_lookups_match(monkeypatch, seed, requests, tokens):
    rng = np.random.default_rng(seed)
    starts = torch.tensor(_starts(rng, requests, tokens), dtype=torch.int32)
    rows = torch.arange(tokens)
    index = torch.tensor(rng.integers(-requests, requests, tokens))
    table = torch.tensor(rng.integers(-1000, 1000, requests + 1), dtype=torch.int32)
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("TPU_ONEHOT_TABLE_LOOKUP", flag)
        out[flag] = (
            table_lookup.bucketize_right(rows, starts[1:]),
            table_lookup.bucketize_right(rows.int(), starts[1:], out_int32=True),
            table_lookup.take(table, index),
            table_lookup.take(table, index.clamp(min=0)[:, None]),
        )
    for ref, new in zip(out["0"], out["1"], strict=True):
        assert ref.dtype == new.dtype and ref.shape == new.shape
        assert torch.equal(ref, new)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("first", [0, 3])
@pytest.mark.parametrize("requests,tokens", [(48, 288), (48, 96), (2, 9)])
def test_csa2_requests_match(seed, first, requests, tokens):
    starts = jnp.asarray(_starts(np.random.default_rng(seed), requests, tokens, first))
    ref = _requests(starts, tokens, onehot=False)
    new = _requests(starts, tokens, onehot=True)
    assert ref.dtype == new.dtype
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(new))
