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
"""Selection of the few-token MoE kernel inside the routed GMM path.

The kernel runs in Pallas interpret mode off a TPU. The routed GMM path it
replaces is TPU-only, so "not selected" is shown by reaching it: its first
step is stubbed to raise.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.moe_lowtok import kernel as lowtok_kernel
from vllm_torchtpu.layers.core import fused_moe_gmm

FP8 = jnp.float8_e4m3fn
LOCAL, HIDDEN, INTER, TOPK, START = 8, 512, 256, 4, 16


class _ReachedGmm(Exception):
    pass


@pytest.fixture
def hooked(monkeypatch):
    """fused_moe_func with the kernel in interpret mode off a TPU and the
    routed GMM path's entry stubbed to raise."""
    interpret = jax.default_backend() != "tpu"
    monkeypatch.setattr(
        fused_moe_gmm,
        "moe_lowtok",
        functools.partial(lowtok_kernel.moe_lowtok, interpret=interpret),
    )

    def reached_gmm(*args, **kwargs):
        raise _ReachedGmm

    monkeypatch.setattr(fused_moe_gmm, "prepare_routed_gmm_inputs", reached_gmm)
    monkeypatch.setattr(fused_moe_gmm, "_DENSE_EXPERT_THRESHOLD", 0)
    # The undecorated function: jit would cache the first trace's decision.
    return fused_moe_gmm.fused_moe_func.__wrapped__


def _inputs(tokens, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(
        rng.standard_normal((tokens, HIDDEN)).astype(np.float32), jnp.bfloat16
    )
    # Global ids over three shards' experts, so some land outside this one.
    ids = np.stack(
        [rng.choice(3 * LOCAL, size=TOPK, replace=False) for _ in range(tokens)]
    )
    weights = rng.random((tokens, TOPK)).astype(np.float32)
    weights = jnp.asarray(weights / weights.sum(axis=1, keepdims=True), jnp.bfloat16)
    w1 = jnp.asarray(
        np.clip(rng.standard_normal((LOCAL, HIDDEN, 2 * INTER)) * 40, -448, 448), FP8
    )
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((LOCAL, INTER, HIDDEN)) * 40, -448, 448), FP8
    )
    s1 = jnp.asarray(
        rng.random((LOCAL, 1, 1, 2 * INTER)).astype(np.float32) / (40 * np.sqrt(HIDDEN))
    )
    s2 = jnp.asarray(
        rng.random((LOCAL, 1, 1, HIDDEN)).astype(np.float32) / (40 * np.sqrt(INTER))
    )
    return dict(
        hidden_states=x,
        w1=w1,
        w2=w2,
        w1_scale=s1,
        w2_scale=s2,
        w1_bias=None,
        w2_bias=None,
        topk_weights=weights,
        topk_ids=jnp.asarray(ids.astype(np.int32)),
        experts_start=jnp.int32(START),
        topk=TOPK,
        activation="silu",
        use_ep=True,
    )


def test_selected_for_few_tokens_and_matches_reference(hooked, monkeypatch):
    monkeypatch.setattr(fused_moe_gmm, "_LOWTOK_MAX_TOKENS", 16)
    args = _inputs(tokens=5)
    got = np.asarray(hooked(**args), np.float32)

    # This shard's experts are START .. START + LOCAL - 1; the rest are masked.
    local = args["topk_ids"] - START
    valid = (local >= 0) & (local < LOCAL)
    want = np.asarray(
        lowtok_kernel.moe_lowtok_reference(
            args["hidden_states"],
            jnp.where(valid, local, -1),
            jnp.where(valid, args["topk_weights"], 0),
            args["w1"],
            args["w1_scale"],
            args["w2"],
            args["w2_scale"],
        ),
        np.float32,
    )
    assert np.abs(want).max() > 0
    np.testing.assert_allclose(got, want, atol=2e-2 * np.abs(want).max(), rtol=0)


@pytest.mark.parametrize(
    "max_tokens,tokens,overrides",
    [
        (0, 4, {}),  # off by default
        (16, 17, {}),  # above the cap
        (16, 4, {"use_ep": False, "experts_start": None}),  # no expert parallelism
        (16, 4, {"activation": "gelu"}),  # unsupported layer
    ],
)
def test_other_calls_keep_the_routed_path(
    hooked, monkeypatch, max_tokens, tokens, overrides
):
    monkeypatch.setattr(fused_moe_gmm, "_LOWTOK_MAX_TOKENS", max_tokens)
    args = _inputs(tokens) | overrides
    with pytest.raises(_ReachedGmm):
        hooked(**args)
