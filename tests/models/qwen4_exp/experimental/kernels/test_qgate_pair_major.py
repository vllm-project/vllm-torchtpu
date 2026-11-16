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
"""A rank-major [pair, T, C] q/gate (VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR) gives
the QSA step the same outputs and pool as the head-joined [T, pair * C] one."""

import jax.numpy as jnp
import numpy as np

from tests.models.qwen4_exp.experimental.kernels.qsa_step_harness import (  # noqa: F401 (fixture)
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
    want = run(args, **options)
    got = run((rank_major,) + tuple(args[1:]), **options)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(raw(a), raw(b))
