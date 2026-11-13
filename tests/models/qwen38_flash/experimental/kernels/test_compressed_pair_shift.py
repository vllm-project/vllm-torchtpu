# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT: compressed_pairs' pair destinations,
writers and masks are unchanged and every half it marks for writing carries
the same key row as the gather, for mixed prefill/decode steps with and
without compressed DCP, and when a written row's offset forces the fallback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental import settings
from vllm_torchtpu.models.qwen38_flash.experimental.kernels import pooled_qsa

RATIO = 4


def _step(lens, contexts, seed):
    rng = np.random.default_rng(seed)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    tokens = int(starts[-1])
    req = np.repeat(np.arange(len(lens)), lens).astype(np.int32)
    offset = np.arange(tokens) - starts[req]
    logical = (np.asarray(contexts)[req] + offset).astype(np.int32)
    complete = (logical % RATIO) == RATIO - 1
    keys = jnp.asarray(rng.normal(size=(tokens, 128)), jnp.bfloat16)
    return (
        keys,
        jnp.asarray(complete),
        jnp.asarray(req),
        jnp.asarray(logical),
        jnp.asarray(starts),
        jnp.asarray(contexts, jnp.int32),
    )


def _pairs(flag, monkeypatch, pool, table, step, dcp):
    monkeypatch.setattr(settings, "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT", flag)
    keys, complete, req, logical, starts, context = step
    fn = jax.jit(
        lambda *a: pooled_qsa.compressed_pairs(
            pool, *a, table, RATIO, **({"dcp": dcp} if dcp else {})
        )
    )
    return [np.asarray(x) for x in fn(keys, complete, req, logical, starts, context)]


@pytest.mark.parametrize("dcp", [False, True])
@pytest.mark.parametrize(
    "lens,contexts",
    [
        ((1000,), (0,)),
        ((1, 1, 1, 509), (37, 4, 130, 6)),
        ((3, 777, 1, 250), (9, 0, 1023, 61)),
    ],
)
def test_pair_shift_matches_gather(lens, contexts, dcp, monkeypatch):
    pool = jnp.zeros((64, 256, 1, 128), jnp.bfloat16)
    table = jnp.arange(4 * 16, dtype=jnp.int32).reshape(4, 16) % 64
    step = _step(lens, contexts, seed=sum(lens))
    dcp_arg = (jnp.asarray([1], jnp.int32), 2) if dcp else None
    want = _pairs(False, monkeypatch, pool, table, step, dcp_arg)
    got = _pairs(True, monkeypatch, pool, table, step, dcp_arg)
    for g, w in zip(got[:3], want[:3]):
        np.testing.assert_array_equal(g, w)
    mask = want[2]
    np.testing.assert_array_equal(got[3][mask], want[3][mask])


def test_pair_shift_falls_back_on_out_of_range_rows(monkeypatch):
    # A row whose logical position does not follow its token index: its
    # group members are far away, so the shift path must hand over to the gather.
    pool = jnp.zeros((64, 256, 1, 128), jnp.bfloat16)
    table = jnp.arange(64, dtype=jnp.int32).reshape(1, 64)
    keys, complete, req, logical, starts, context = _step((64,), (0,), seed=1)
    logical = logical.at[10].set(43)
    complete = (logical % RATIO) == RATIO - 1
    step = (keys, complete, req, logical, starts, context)
    want = _pairs(False, monkeypatch, pool, table, step, None)
    got = _pairs(True, monkeypatch, pool, table, step, None)
    mask = want[2]
    assert mask[10].any()
    np.testing.assert_array_equal(got[3][mask], want[3][mask])
