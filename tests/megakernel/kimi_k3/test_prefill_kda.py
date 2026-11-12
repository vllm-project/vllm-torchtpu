# SPDX-License-Identifier: Apache-2.0
"""The prefill KDA layer's pieces on CPU: the chunked delta rule's sub-block
factors."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

pytestmark = pytest.mark.cpu_test


def _delta_rule(q, k, v, gate, beta, state):
    """The Kimi delta rule token by token in FP64."""
    q, k, v, gate, beta = (np.asarray(x, np.float64) for x in (q, k, v, gate, beta))
    s = np.asarray(state, np.float64)
    outputs = []
    for r in range(q.shape[1]):
        s = s * np.exp(gate[:, r, :, None])
        delta = beta[:, r, None] * (v[:, r] - np.einsum("hk,hkv->hv", k[:, r], s))
        s = s + np.einsum("hk,hv->hkv", k[:, r], delta)
        outputs.append(np.einsum("hk,hkv->hv", q[:, r], s))
    return np.stack(outputs, axis=1), s


def _gates(kind, rng, shape):
    if kind == "realistic":  # lower bound * sigmoid, as kda_decay
        return -5 / (1 + np.exp(-(2 * rng.standard_normal(shape) + 1)))
    if kind == "mixed":
        return rng.choice([0.0, -0.01, -1.0, -5.0], size=shape)
    gate = np.full(shape, -5.0)  # every row at the bound: the widest factors
    if kind == "boundary":  # but no decay between sub-blocks
        gate[:, 32::32] = 0.0
    return gate


@pytest.mark.parametrize("kind", ["realistic", "mixed", "saturated", "boundary"])
@pytest.mark.parametrize("fast", [True, False])
def test_chunk_sub_blocks_against_the_sequential_delta_rule(kind, fast):
    assert kernels._KDA_SUB_BLOCK == 32  # the kernel's default
    rng = np.random.default_rng(7)
    heads, chunks, channels = 3, 3, 128
    shape = (heads, 64, channels)
    state = 0.1 * rng.standard_normal((heads, channels, channels))
    initial = state
    outputs = {16: [], 32: []}
    states = {16: state, 32: state}
    references = []
    for _ in range(chunks):
        q = rng.standard_normal(shape)
        q *= channels**-0.5 / np.linalg.norm(q, axis=-1, keepdims=True)
        k = rng.standard_normal(shape)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.standard_normal(shape)
        gate = _gates(kind, rng, shape)
        beta = 1 / (1 + np.exp(-rng.standard_normal(shape[:2])))
        reference, initial = _delta_rule(q, k, v, gate, beta, initial)
        references.append(reference)
        args = [jnp.asarray(x, jnp.float32) for x in (q, k, v, gate, beta)]
        for sub_block in (16, 32):
            invert = None if fast else kernels._invert_delta_rule_two_level
            out, states[sub_block] = kernels.kda_chunk(
                *args,
                jnp.asarray(states[sub_block], jnp.float32),
                sub_block=sub_block,
                invert=invert,
            )
            outputs[sub_block].append(np.asarray(out))
    reference = np.stack(references)
    scale = np.abs(reference).max()
    errors = {
        sub_block: np.abs(np.stack(outputs[sub_block]) - reference).max() / scale
        for sub_block in (16, 32)
    }
    for sub_block in (16, 32):
        state_error = np.abs(np.asarray(states[sub_block]) - initial).max()
        assert state_error <= 5e-5 * np.abs(initial).max()
    assert errors[16] <= 2e-5
    # Rows at the decay bound reach the dropped-term bound of 32-row
    # sub-blocks (``_kda_chunk_core``); other gates agree with 16 rows.
    saturated = kind in ("saturated", "boundary")
    assert errors[32] <= (1e-3 if saturated else 2e-5)
