# SPDX-License-Identifier: Apache-2.0
"""A rank-major [pair, T, C] q/gate (VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR) gives
the QSA step the same outputs and pool as the head-joined [T, pair * C] one."""

import jax.numpy as jnp
import numpy as np

from tests.models.qwen38_flash.experimental.kernels.test_fused_kv_write import (  # noqa: F401 (fixture)
    interpret,
    make_step,
    raw,
    run,
)


def test_rank_major_qgate_matches_joined(interpret, monkeypatch):  # noqa: F811
    args = make_step(
        np.random.default_rng(4), [40, 4, 4], [3, 509, 7], jnp.float8_e4m3fn
    )
    joined = args[0]
    half = joined.shape[1] // 2
    rank_major = jnp.stack([joined[:, :half], joined[:, half:]])
    options = dict(main_slot=1, max_query_len=40)
    want = run(monkeypatch, False, args, **options)
    got = run(monkeypatch, False, (rank_major,) + tuple(args[1:]), **options)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(raw(a), raw(b))
