# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_DCP_SPLIT_LSE: qsa_attention(dcp=...) routing the partial
output and its lse as two arrays gives the same bits as the [.., D + 1]
channel, on the decode-only, prefix-slice and row-gather routes."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import qsa_rpa


def _fake_attend(scale):
    def attend(q, gate, blocks, lengths, table, cache, *rest, **kwargs):
        assert kwargs.get("return_lse")
        out = q.astype(jnp.float32) * scale + lengths[:, None, None]
        lse = jnp.where(
            (lengths > 0)[:, None], q.astype(jnp.float32).sum(-1) + scale, -jnp.inf
        )
        return out, lse, cache + 1

    return attend


@pytest.mark.parametrize("prefix_slice", ["1", "0"])
@pytest.mark.parametrize(
    "tokens,starts", [(16, [0, 1, 2, 3, 16]), (16, [0, 16]), (8, [0, 2, 4, 6, 8])]
)
def test_split_lse_matches_channel(monkeypatch, prefix_slice, tokens, starts):
    monkeypatch.setattr(qsa_rpa, "decode_attention", _fake_attend(2.0))
    monkeypatch.setattr(qsa_rpa, "_prefill_attention", _fake_attend(3.0))
    monkeypatch.setenv("VLLM_TPU_QSA_PREFIX_SLICE", prefix_slice)
    rng = np.random.default_rng(tokens + len(starts))
    q = jnp.asarray(rng.normal(size=(tokens, 6, 8)), jnp.float32)
    lengths = jnp.asarray(rng.integers(0, 5, size=tokens), jnp.int32)
    blocks = jnp.zeros((tokens, 4), jnp.int32)
    args = (
        q,
        q,
        blocks,
        lengths,
        blocks,
        jnp.zeros((2,), jnp.float32),
        jnp.asarray(starts, jnp.int32),
    )
    kwargs = dict(kv_heads=1, ratio=4, max_query_len=4, dcp=(0, 2))
    results = []
    for split in ("0", "1"):
        monkeypatch.setenv("VLLM_TPU_QSA_DCP_SPLIT_LSE", split)
        results.append(qsa_rpa.qsa_attention(*args, **kwargs))
    for want, got in zip(*results):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
