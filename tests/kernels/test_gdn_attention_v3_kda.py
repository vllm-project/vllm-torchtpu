# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Correctness tests for the KDA path of the fused conv1d + GDN v3 kernel.

What these tests exist for is the wiring: the per-channel gate is `kq_head_dim`
times wider than GDN's per-head scalar, so it travels through a different
BlockSpec and a different VMEM load on its way into the kernel.

The oracle is a float64 token-at-a-time recurrence, preceded by the conv1d
and silu that the fused kernel folds in. Comparing against a recurrence rather
than against another Pallas kernel keeps this independent of the implementation
it is meant to check.

Both gate forms are covered. `gate_lower_bound=None` gives
`-exp(A_log) * softplus(g + dt_bias)`, which Kimi-Linear-48B uses; a float
gives `gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, which every
K3 variant uses with -5.0. They are different functions of the same
weights, so each needs its own oracle.
"""

from __future__ import annotations

from typing import NamedTuple
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from kernels.gdn.v3.vmem_test_utils import (
    get_vmem_declared_bytes,
    weight_refs_vmem_bytes,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import (
    compute_kda,
    config,
    memory_ref,
    tiling,
    wrapper,
)
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op


class _TracedKdaInputs(NamedTuple):
    """Typed pytree container for tracing fused_conv1d_gdn on CPU."""

    qkv: jax.Array
    b: jax.Array
    a: jax.Array
    states: tuple[jax.Array, jax.Array]
    conv_weight: jax.Array
    a_log: jax.Array
    dt_bias: jax.Array
    query_start_loc: jax.Array
    state_indices: jax.Array
    distribution: jax.Array
    seq_lens: jax.Array
    read_offsets: jax.Array | None


def _fresh_jit_fused_conv1d_gdn() -> jax.stages.Wrapped:
    """Return an isolated jax.jit around fused_conv1d_gdn.__wrapped__."""
    return jax.jit(
        wrapper.fused_conv1d_gdn.__wrapped__,
        static_argnames=wrapper.FUSED_CONV1D_GDN_STATIC_ARGNAMES,
    )


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Mirror the kernel's `_l2_norm_f32`, which accumulates in float32."""
    x_f32 = x.astype(np.float32)
    inv_norm = 1.0 / np.sqrt(np.sum(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6)
    return (x_f32 * inv_norm).astype(np.float64)


def _recurrence(
    q: np.ndarray,  # [T, H, K], normalised
    k: np.ndarray,  # [T, H, K], normalised
    v: np.ndarray,  # [T, H, V]
    gate: np.ndarray,  # [T, H, K], activated (natural log)
    beta: np.ndarray,  # [T, H], through the sigmoid
    state: np.ndarray,  # [H, K, V]
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One token at a time, in float64.

    Transposed to this kernel's token-major activation layout.
    """
    running = state.astype(np.float64).copy()
    output = np.zeros((q.shape[0], q.shape[1], v.shape[-1]), np.float64)
    for token in range(q.shape[0]):
        # The per-channel gate decays along K, hence the trailing axis.
        running *= np.exp(gate[token])[:, :, None]
        prediction = np.einsum("hk,hkv->hv", k[token], running)
        delta = beta[token][:, None] * (v[token] - prediction)
        running += k[token][:, :, None] * delta[:, None, :]
        output[token] = scale * np.einsum("hk,hkv->hv", q[token], running)
    return output, running


def kda_attention_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    states: tuple[jnp.ndarray, jnp.ndarray],
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.KDA,
    gate_lower_bound: float | None = None,
    beta_scale: float = 1.0,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Reference conv1d + silu + KDA, sequence by sequence in eager mode.

    Structured like `test_gdn_attention_v3.gdn_attention_ref` so the two can
    be read side by side; the conv half is identical and the recurrence half
    is the KDA one.
    """
    del attention_mode  # Only the KDA form is implemented here.

    num_tokens = qkv.shape[0]
    num_valid_seqs = int(distribution[2])

    out_mixed_qkv = jnp.zeros_like(qkv)
    conv_state, recurrent_state = states
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = np.zeros((num_tokens, n_v * d_v), np.float64)

    # Part 1: Conv1D, over the raw activations.
    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        if has_init:
            c_state = new_conv_state[s]
        else:
            c_state = jnp.zeros_like(new_conv_state[s])
            new_recurrent_state = new_recurrent_state.at[s].set(
                jnp.zeros_like(new_recurrent_state[s])
            )

        x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k_idx in range(kernel_size):
            acc += (
                x_full[k_idx : k_idx + query_len].astype(jnp.float32)
                * conv_weight[:, 0, k_idx].astype(jnp.float32)[None, :]
            )
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1) :])
        out_mixed_qkv = out_mixed_qkv.at[start:end].set(acc.astype(qkv.dtype))

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    # Part 2: KDA, in float64.
    a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
    dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)
    scale = float(d_k**-0.5)

    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        qkv_seq = np.asarray(out_mixed_qkv[start:end], np.float64)
        key_dim = n_kq * d_k
        q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = qkv_seq[:, key_dim : 2 * key_dim].reshape(query_len, n_kq, d_k)
        v_seq = qkv_seq[:, 2 * key_dim :].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = np.repeat(q_seq, repeat_factor, axis=1)
            k_seq = np.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)

        beta_seq = beta_scale / (1.0 + np.exp(-np.asarray(b[start:end], np.float64)))

        # The gate is per (head, channel): [T, H, K].
        a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
        if gate_lower_bound is None:
            gate_seq = -np.exp(a_log_np) * np.logaddexp(0.0, a_seq + dt_bias_np)
        else:
            # sigmoid(x) written as exp(-logaddexp(0, -x)) to avoid overflow
            # in the exponential for large negative x.
            gate_seq = gate_lower_bound * np.exp(
                -np.logaddexp(0.0, -(np.exp(a_log_np) * (a_seq + dt_bias_np)))
            )

        out_seq, final_state = _recurrence(
            q_seq,
            k_seq,
            v_seq,
            gate_seq,
            beta_seq,
            np.asarray(new_recurrent_state[s], np.float64),
            scale,
        )
        new_recurrent_state = new_recurrent_state.at[s].set(
            jnp.asarray(final_state, new_recurrent_state.dtype)
        )
        output[start:end] = out_seq.reshape(query_len, n_v * d_v)

    return (new_conv_state, new_recurrent_state), output


def kda_attention_spec_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    states: tuple[jnp.ndarray, jnp.ndarray],
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    gate_lower_bound: float | None,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Token reference for KDA verify windows and their checkpoints."""
    del distribution
    conv_state, recurrent_state = states
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = np.zeros((qkv.shape[0], n_v * d_v), np.float64)
    a_log_np = np.asarray(a_log, np.float64).reshape(n_v, 1)
    dt_bias_np = np.asarray(dt_bias, np.float64).reshape(n_v, d_k)

    for req_idx in range(num_spec_seqs):
        base = int(state_indices[req_idx])
        read_slot = base + int(read_offsets[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        has_init = bool((seq_lens[req_idx] - query_len) > 0)
        c_state = (
            conv_state[read_slot] if has_init else jnp.zeros_like(conv_state[read_slot])
        )
        running = np.asarray(
            recurrent_state[read_slot]
            if has_init
            else jnp.zeros_like(recurrent_state[read_slot]),
            np.float64,
        ).copy()

        x_full = jnp.concatenate([c_state, qkv[start:end]], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k_idx in range(kernel_size):
            acc += (
                x_full[k_idx : k_idx + query_len].astype(jnp.float32)
                * conv_weight[:, 0, k_idx].astype(jnp.float32)[None, :]
            )
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = jax.nn.silu(acc).astype(qkv.dtype)
        for t in range(query_len):
            new_conv_state = new_conv_state.at[base + t].set(
                x_full[t + 1 : t + kernel_size]
            )

        conv_out_np = np.asarray(conv_out, np.float64)
        key_dim = n_kq * d_k
        q_seq = conv_out_np[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = conv_out_np[:, key_dim : 2 * key_dim].reshape(query_len, n_kq, d_k)
        v_seq = conv_out_np[:, 2 * key_dim :].reshape(query_len, n_v, d_v)
        if n_v != n_kq:
            repeat_factor = n_v // n_kq
            q_seq = np.repeat(q_seq, repeat_factor, axis=1)
            k_seq = np.repeat(k_seq, repeat_factor, axis=1)
        q_seq = _l2_normalize(q_seq) * float(d_k**-0.5)
        k_seq = _l2_normalize(k_seq)
        # Production activates beta in fp32, then stores it in the model
        # activation dtype before the recurrent update.
        beta_seq = np.asarray(
            jax.nn.sigmoid(b[start:end].astype(jnp.float32)).astype(b.dtype),
            np.float64,
        )
        a_seq = np.asarray(a[start:end], np.float64).reshape(query_len, n_v, d_k)
        if gate_lower_bound is None:
            gate_seq = -np.exp(a_log_np) * np.logaddexp(0.0, a_seq + dt_bias_np)
        else:
            gate_seq = gate_lower_bound * np.exp(
                -np.logaddexp(0.0, -(np.exp(a_log_np) * (a_seq + dt_bias_np)))
            )

        for t in range(query_len):
            running *= np.exp(gate_seq[t])[:, :, None]
            prediction = np.einsum("hk,hkv->hv", k_seq[t], running)
            delta = beta_seq[t][:, None] * (v_seq[t] - prediction)
            running += k_seq[t][:, :, None] * delta[:, None, :]
            output[start + t] = np.einsum("hk,hkv->hv", q_seq[t], running).reshape(-1)
            new_recurrent_state = new_recurrent_state.at[base + t].set(
                jnp.asarray(running, recurrent_state.dtype)
            )

    return (new_conv_state, new_recurrent_state), jnp.asarray(output, qkv.dtype)


_SHAPES = (
    dict(
        name="decode_only",
        max_reqs=8,
        q_loc=list(range(9)),
        distribution=[8, 8, 8],
    ),
    dict(
        name="single_prefill",
        max_reqs=1,
        q_loc=[0, 256],
        distribution=[0, 1, 1],
    ),
    dict(
        name="mixed_prefill",
        max_reqs=3,
        q_loc=[0, 192, 320, 384],
        distribution=[0, 3, 3],
    ),
    dict(
        name="ragged_prefill",
        max_reqs=4,
        q_loc=[0, 65, 130, 131, 195],
        distribution=[0, 4, 4],
    ),
    # Kimi-Linear-48B's head count. `triangular_block_size` is a step
    # function of `num_v_heads`, so this covers a different sub-block
    # size (8) than the cases above (16).
    dict(
        name="kimi_linear_heads",
        max_reqs=2,
        q_loc=[0, 192, 256],
        distribution=[0, 2, 2],
        n_v=32,
    ),
    # OLMo 3.5: value heads twice as wide as key heads (256 / 512), one head
    # per TP8 rank, and beta in [0, 2) (allow_neg_eigval).
    dict(
        name="olmo35_heads",
        max_reqs=3,
        q_loc=[0, 1, 129, 200],
        distribution=[1, 3, 3],
        n_v=1,
        d_k=256,
        d_v=512,
        beta_scale=2.0,
    ),
    # OLMo 3.5 at TP2 (four heads per rank), resuming from random conv and
    # recurrent states: seq_lens exceed the query lengths, so the wide state
    # is DMA'd in from HBM rather than zero-initialized.
    dict(
        name="olmo35_heads_tp2_warm_state",
        max_reqs=3,
        q_loc=[0, 1, 2, 130],
        distribution=[2, 3, 3],
        n_v=4,
        d_k=256,
        d_v=512,
        beta_scale=2.0,
        initial_state=True,
    ),
)

# Kimi-Linear-48B sets no bound; every K3 variant sets -5.0.
_GATES = (("softplus", None), ("bounded", -5.0))


def _cases():
    """Every shape under both gate forms."""
    for shape in _SHAPES:
        for gate_name, lower_bound in _GATES:
            case = dict(shape)
            case["testcase_name"] = f"{case.pop('name')}_{gate_name}"
            case["gate_lower_bound"] = lower_bound
            yield case
            if shape["distribution"][0] == 0:
                yield dict(
                    case,
                    testcase_name=case["testcase_name"] + "_prefill_only",
                    prefill_only=True,
                )


@pytest.mark.tpu_test
class KdaAttentionV3Test(parameterized.TestCase):
    @parameterized.named_parameters(*_cases())
    def test_kda_matches_the_recurrence(
        self,
        max_reqs,
        q_loc,
        distribution,
        gate_lower_bound,
        n_v=4,
        d_k=128,
        d_v=128,
        beta_scale=1.0,
        prefill_only=False,
        initial_state=False,
    ):
        # K3, Kimi-Linear and OLMo 3.5 all run KDA without GQA: n_kq == n_v.
        n_kq = n_v
        kernel_size = 4

        q_loc = jnp.array(q_loc, dtype=jnp.int32)
        distribution = jnp.array(distribution, dtype=jnp.int32)
        num_tokens = int(q_loc[max_reqs])

        # Slot 0 is the reserved null block for invalid / padded tokens.
        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1
        dim_size = 2 * n_kq * d_k + n_v * d_v

        rngs = iter(jax.random.split(jax.random.key(0), 10))
        qkv = jax.random.normal(next(rngs), (num_tokens, dim_size))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        # The KDA gate is per-channel: n_v * d_k wide, not n_v.
        a = jax.random.normal(next(rngs), (num_tokens, n_v * d_k))

        if initial_state:
            conv_state = jax.random.normal(
                next(rngs), (num_blocks, kernel_size - 1, dim_size)
            )
            recurrent_state = 0.1 * jax.random.normal(
                next(rngs), (num_blocks, n_v, d_k, d_v)
            )
        else:
            conv_state = jnp.zeros((num_blocks, kernel_size - 1, dim_size))
            recurrent_state = jnp.zeros((num_blocks, n_v, d_k, d_v))
        conv_weight = jax.random.normal(next(rngs), (dim_size, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (dim_size,))

        a_log = jax.random.normal(next(rngs), (n_v,))
        # Per-channel, and kept 2D: the kernel reshapes it to broadcast over
        # (head, channel), which Mosaic will not do from a flat vector.
        dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

        seq_lens = jnp.asarray(
            q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32
        )
        if initial_state:
            # Earlier context: every sequence resumes from its stored state.
            seq_lens = seq_lens + 64

        common_kwargs = dict(
            qkv=qkv,
            b=b,
            a=a,
            states=(conv_state, recurrent_state),
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            attention_mode=config.AttentionMode.KDA,
            gate_lower_bound=gate_lower_bound,
            beta_scale=beta_scale,
        )

        (_, ref_state), ref_out = kda_attention_ref(**common_kwargs)

        kda_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=[
                "n_kq",
                "n_v",
                "d_k",
                "d_v",
                "kernel_size",
                "attention_mode",
                "gate_lower_bound",
                "beta_scale",
                "prefill_only",
            ],
        )
        (_, out_state), out = kda_jitted(**common_kwargs, prefill_only=prefill_only)

        # Tighter than the GDN v3 suite's 2e-2/2e-2 on purpose. KDA feeds the
        # output through an L2 norm and a d_k**-0.5 scale, so |out| peaks
        # around 0.3 and 2e-2 would swallow the gate-layout errors this test
        # exists to catch. The observed fp32-vs-fp64 gap is ~1e-3.
        np.testing.assert_allclose(
            np.asarray(out, np.float64), ref_out, rtol=2e-2, atol=5e-3
        )
        # The state is looser than the output because it accumulates the
        # delta rule over the whole chunk without the output's L2 norm and
        # d_k**-0.5 scale to damp it, and `v - prediction` cancels hard on a
        # handful of entries. Still 2x tighter than the GDN v3 suite.
        np.testing.assert_allclose(
            np.asarray(out_state, np.float64),
            np.asarray(ref_state, np.float64),
            rtol=2e-2,
            atol=1e-2,
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="tp32_multi_tile_recurrent_window",
            num_heads=3,
            spec_lengths=[1, 6, 3, 8, 5, 2, 7, 4] * 4,
            read_offsets=[0, 2, 4, 6, 0, 2, 4, 6] * 4,
        ),
        dict(
            testcase_name="tp8_large_recurrent_window",
            num_heads=12,
            spec_lengths=[1, 6, 3, 8, 5, 2, 7, 4],
            read_offsets=[0, 2, 4, 6, 0, 2, 4, 6],
        ),
    )
    def test_speculative_window_checkpoints(
        self, num_heads, spec_lengths, read_offsets
    ):
        """An 8-token K3 verify window stays in one kernel invocation.

        This covers ragged windows, rollback reads from non-zero checkpoints,
        TP8's large FP32 recurrent-state window, and enough TP32 sequences to
        pipeline multiple tiles. Every real token must write its conv and
        recurrent state to ``base + t``; padded tail slots must remain
        untouched.
        """
        n_kq = n_v = num_heads
        d_k = d_v = 128
        kernel_size = 4
        num_spec_tokens = 7
        window = num_spec_tokens + 1
        num_seqs = len(spec_lengths)
        q_loc = jnp.array(
            np.concatenate([[0], np.cumsum(spec_lengths)]), dtype=jnp.int32
        )
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(read_offsets, dtype=jnp.int32)
        seq_lens = jnp.array([32 + length for length in spec_lengths], dtype=jnp.int32)
        dim_size = 2 * n_kq * d_k + n_v * d_v

        rngs = iter(jax.random.split(jax.random.key(11), 8))
        qkv = jax.random.normal(
            next(rngs), (sum(spec_lengths), dim_size), dtype=jnp.bfloat16
        )
        b = jax.random.normal(next(rngs), (sum(spec_lengths), n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(
            next(rngs), (sum(spec_lengths), n_v * d_k), dtype=jnp.bfloat16
        )
        # Production KDA keeps conv state in fp32 (layers/adapter/kda_attention.py
        # kda_state_dtype).
        conv_state = jax.random.normal(
            next(rngs),
            (num_blocks, kernel_size - 1, dim_size),
            dtype=jnp.float32,
        )
        recurrent_state = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v))
        conv_weight = jax.random.normal(
            next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
        )
        a_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v, d_k))

        common_kwargs = dict(
            qkv=qkv,
            b=b,
            a=a,
            states=(conv_state, recurrent_state),
            conv_weight=conv_weight,
            conv_bias=None,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            gate_lower_bound=-5.0,
        )

        (ref_conv, ref_rec), ref_out = kda_attention_spec_ref(
            **common_kwargs,
            read_offsets=read_offsets_arr,
            num_spec_seqs=num_seqs,
        )
        # `fused_conv1d_gdn` already owns its jit/static-argument contract and
        # donates both cache inputs. Build the reference first: reading either
        # original cache after this call is invalid once JAX reuses it.
        (new_conv, new_rec), out = wrapper.fused_conv1d_gdn(
            **common_kwargs,
            read_offsets=read_offsets_arr,
            num_spec_tokens=num_spec_tokens,
            batched_only=True,
            attention_mode=config.AttentionMode.KDA,
        )

        np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=5e-3)
        for slot in range(num_blocks):
            np.testing.assert_allclose(
                new_conv[slot],
                ref_conv[slot],
                rtol=2e-2,
                atol=5e-3,
                err_msg=f"conv checkpoint mismatch at slot {slot}",
            )
            np.testing.assert_allclose(
                new_rec[slot],
                ref_rec[slot],
                rtol=2e-2,
                atol=1e-2,
                err_msg=f"recurrent checkpoint mismatch at slot {slot}",
            )

    def test_kda_spec_decode_dynamic_tiling_head_splits(self) -> None:
        """Verify KDA speculative verify (W=4) with dynamic tiling."""
        d_k = d_v = 128
        kernel_size = 4
        num_spec_tokens = 3
        window = num_spec_tokens + 1
        spec_lengths = [1, 4, 2, 4, 3, 1, 4, 2]
        read_offsets = [0, 1, 2, 3, 0, 1, 2, 3]
        num_seqs = len(spec_lengths)
        query_start_loc = jnp.array(
            np.concatenate([[0], np.cumsum(spec_lengths)]), dtype=jnp.int32
        )
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(read_offsets, dtype=jnp.int32)
        seq_lens = jnp.array([16 + length for length in spec_lengths], dtype=jnp.int32)
        total_tokens = sum(spec_lengths)

        for n_kq, n_v, gate_lower_bound in [
            (3, 3, -5.0),
            (12, 12, -5.0),
            (4, 4, None),
            (32, 32, None),
        ]:
            with (
                self.subTest(n_kq=n_kq, n_v=n_v, gate_lower_bound=gate_lower_bound),
                pytest.MonkeyPatch.context() as mp,
            ):
                mp.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
                dim_size = 2 * n_kq * d_k + n_v * d_v
                rngs = iter(jax.random.split(jax.random.key(n_v), 8))
                init_conv = jax.random.normal(
                    next(rngs),
                    (num_blocks, kernel_size - 1, dim_size),
                    dtype=jnp.float32,
                )
                init_rec = jax.random.normal(
                    next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
                )
                common_kwargs = {
                    "qkv": jax.random.normal(
                        next(rngs), (total_tokens, dim_size), dtype=jnp.bfloat16
                    ),
                    "b": jax.random.normal(
                        next(rngs), (total_tokens, n_v), dtype=jnp.bfloat16
                    ),
                    "a": jax.random.normal(
                        next(rngs), (total_tokens, n_v * d_k), dtype=jnp.bfloat16
                    ),
                    "conv_weight": jax.random.normal(
                        next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
                    ),
                    "conv_bias": None,
                    "a_log": jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
                    "dt_bias": jax.random.normal(
                        next(rngs), (n_v, d_k), dtype=jnp.float32
                    ),
                    "query_start_loc": query_start_loc,
                    "state_indices": state_indices,
                    "distribution": distribution,
                    "seq_lens": seq_lens,
                    "n_kq": n_kq,
                    "n_v": n_v,
                    "d_k": d_k,
                    "d_v": d_v,
                    "kernel_size": kernel_size,
                    "gate_lower_bound": gate_lower_bound,
                }
                (ref_conv, ref_rec), ref_out = kda_attention_spec_ref(
                    states=(init_conv, init_rec),
                    read_offsets=read_offsets_arr,
                    num_spec_seqs=num_seqs,
                    **common_kwargs,
                )
                jit_fn = _fresh_jit_fused_conv1d_gdn()
                (new_conv, new_rec), out = jit_fn(
                    states=(init_conv.copy(), init_rec.copy()),
                    read_offsets=read_offsets_arr,
                    num_spec_tokens=num_spec_tokens,
                    batched_only=True,
                    attention_mode=config.AttentionMode.KDA,
                    **common_kwargs,
                )
                np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=5e-3)
                for slot in range(num_blocks):
                    np.testing.assert_allclose(
                        new_conv[slot],
                        ref_conv[slot],
                        rtol=2e-2,
                        atol=5e-3,
                        err_msg=f"n_v={n_v} conv slot {slot}",
                    )
                    np.testing.assert_allclose(
                        new_rec[slot],
                        ref_rec[slot],
                        rtol=2e-2,
                        atol=1e-2,
                        err_msg=f"n_v={n_v} rec slot {slot}",
                    )

    def test_kda_spec_window_multi_sequence_tile(self) -> None:
        """Verify KDA speculative verify across multi-sequence tiles (W=4)."""
        n_kq = n_v = 3
        d_k = d_v = 128
        kernel_size = 4
        num_spec_tokens = 3
        window = num_spec_tokens + 1
        num_seqs = 24
        spec_lengths = [1 + (i % window) for i in range(num_seqs)]
        read_offsets = [i % window for i in range(num_seqs)]
        total_tokens = sum(spec_lengths)
        dim_size = 2 * n_kq * d_k + n_v * d_v
        tpu_info = pltpu.get_tpu_info()

        decode_tile, _ = tiling.get_tile_sizes(
            batch_size=total_tokens,
            num_seqs=num_seqs,
            padded_batch_size=total_tokens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim_size,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=tpu_info.num_lanes,
            decode_vmem_limit_bytes=config.GDNConfig.get_tiling_vmem_limit_bytes(
                window
            ),
            mixed_vmem_limit_bytes=config.GDNConfig.get_tiling_vmem_limit_bytes(),
            window_size=window,
            attention_mode=config.AttentionMode.KDA,
            num_sublanes=tpu_info.num_sublanes,
        )
        steps = (num_seqs + decode_tile - 1) // decode_tile
        self.assertGreaterEqual(decode_tile, 2)
        self.assertGreaterEqual(steps, 2)

        query_start_loc = jnp.array(
            np.concatenate([[0], np.cumsum(spec_lengths)]), dtype=jnp.int32
        )
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(read_offsets, dtype=jnp.int32)
        seq_lens = jnp.array([16 + length for length in spec_lengths], dtype=jnp.int32)

        rngs = iter(jax.random.split(jax.random.key(0x5EC3), 8))
        init_conv = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, dim_size), dtype=jnp.float32
        )
        init_rec = jax.random.normal(
            next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
        )
        common_kwargs = {
            "qkv": jax.random.normal(
                next(rngs), (total_tokens, dim_size), dtype=jnp.bfloat16
            ),
            "b": jax.random.normal(next(rngs), (total_tokens, n_v), dtype=jnp.bfloat16),
            "a": jax.random.normal(
                next(rngs), (total_tokens, n_v * d_k), dtype=jnp.bfloat16
            ),
            "conv_weight": jax.random.normal(
                next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
            ),
            "conv_bias": None,
            "a_log": jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
            "dt_bias": jax.random.normal(next(rngs), (n_v, d_k), dtype=jnp.float32),
            "query_start_loc": query_start_loc,
            "state_indices": state_indices,
            "distribution": distribution,
            "seq_lens": seq_lens,
            "n_kq": n_kq,
            "n_v": n_v,
            "d_k": d_k,
            "d_v": d_v,
            "kernel_size": kernel_size,
            "gate_lower_bound": -5.0,
        }
        (ref_conv, ref_rec), ref_out = kda_attention_spec_ref(
            states=(init_conv, init_rec),
            read_offsets=read_offsets_arr,
            num_spec_seqs=num_seqs,
            **common_kwargs,
        )
        jit_fn = _fresh_jit_fused_conv1d_gdn()
        for repeat_idx in range(3):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
                (new_conv, new_rec), out = jit_fn(
                    states=(init_conv.copy(), init_rec.copy()),
                    read_offsets=read_offsets_arr,
                    num_spec_tokens=num_spec_tokens,
                    batched_only=True,
                    attention_mode=config.AttentionMode.KDA,
                    **common_kwargs,
                )
            np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=5e-3)
            for slot in range(num_blocks):
                np.testing.assert_allclose(
                    new_conv[slot],
                    ref_conv[slot],
                    rtol=2e-2,
                    atol=5e-3,
                    err_msg=f"rep {repeat_idx} conv slot {slot}",
                )
                np.testing.assert_allclose(
                    new_rec[slot],
                    ref_rec[slot],
                    rtol=2e-2,
                    atol=1e-2,
                    err_msg=f"rep {repeat_idx} rec slot {slot}",
                )

    def test_kda_head_dense_prefill_c64_execution(self) -> None:
        """Kimi-Linear-48B DP8/TP1 (n_v=32, B=256) selects C=64 and matches ref."""
        n_kq = n_v = 32
        d_k = d_v = 128
        kernel_size = 4
        dim_size = 2 * n_kq * d_k + n_v * d_v
        tokens = 256
        tpu_info = pltpu.get_tpu_info()
        mixed_limit = config.GDNConfig.get_tiling_vmem_limit_bytes()
        _, mixed_tile = tiling.get_tile_sizes(
            batch_size=tokens,
            num_seqs=1,
            padded_batch_size=tokens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim_size,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=tpu_info.num_lanes,
            decode_vmem_limit_bytes=mixed_limit,
            mixed_vmem_limit_bytes=mixed_limit,
            window_size=1,
            attention_mode=config.AttentionMode.KDA,
            num_sublanes=tpu_info.num_sublanes,
        )
        self.assertEqual(mixed_tile, 64)
        rngs = iter(jax.random.split(jax.random.key(0x32), 8))
        query_start_loc = jnp.array([0, tokens], dtype=jnp.int32)
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        state_indices = jnp.array([0], dtype=jnp.int32)
        seq_lens = jnp.array([tokens], dtype=jnp.int32)
        init_conv = jnp.zeros((2, kernel_size - 1, dim_size), dtype=jnp.float32)
        init_rec = jnp.zeros((2, n_v, d_k, d_v), dtype=jnp.float32)
        kwargs = {
            "qkv": jax.random.normal(
                next(rngs), (tokens, dim_size), dtype=jnp.bfloat16
            ),
            "b": jax.random.normal(next(rngs), (tokens, n_v), dtype=jnp.bfloat16),
            "a": jax.random.normal(next(rngs), (tokens, n_v * d_k), dtype=jnp.bfloat16),
            "conv_weight": jax.random.normal(
                next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
            ),
            "conv_bias": None,
            "a_log": jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
            "dt_bias": jax.random.normal(next(rngs), (n_v, d_k), dtype=jnp.float32),
            "query_start_loc": query_start_loc,
            "state_indices": state_indices,
            "distribution": distribution,
            "seq_lens": seq_lens,
            "n_kq": n_kq,
            "n_v": n_v,
            "d_k": d_k,
            "d_v": d_v,
            "kernel_size": kernel_size,
        }
        (ref_conv, ref_rec), ref_out = kda_attention_ref(
            states=(init_conv, init_rec),
            **kwargs,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
            jit_fn = _fresh_jit_fused_conv1d_gdn()
            (new_conv, new_rec), out = jit_fn(
                states=(init_conv.copy(), init_rec.copy()),
                attention_mode=config.AttentionMode.KDA,
                **kwargs,
            )
        np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(new_conv[0], ref_conv[0], rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(new_rec[0], ref_rec[0], rtol=2e-2, atol=1e-2)

    @parameterized.named_parameters(
        {
            "testcase_name": "kimi_k3_tp32_unbounded",
            "num_heads": 3,
            "gate_lower_bound": None,
        },
        {
            "testcase_name": "kimi_k3_tp32_bounded",
            "num_heads": 3,
            "gate_lower_bound": -5.0,
        },
        {
            "testcase_name": "kimi_linear_tp8_unbounded",
            "num_heads": 4,
            "gate_lower_bound": None,
        },
        {
            "testcase_name": "kimi_linear_tp8_bounded",
            "num_heads": 4,
            "gate_lower_bound": -5.0,
        },
    )
    def test_kda_small_head_prefill_c128_execution(
        self, num_heads: int, gate_lower_bound: float | None
    ) -> None:
        """Verify small-head KDA prefill selects C=128 and matches reference."""
        d_k = d_v = 128
        kernel_size = 4
        tokens = 256
        dim_size = 2 * num_heads * d_k + num_heads * d_v
        tpu_info = pltpu.get_tpu_info()
        mixed_limit = config.GDNConfig.get_tiling_vmem_limit_bytes()
        _, mixed_tile = tiling.get_tile_sizes(
            batch_size=tokens,
            num_seqs=1,
            padded_batch_size=tokens,
            n_kq=num_heads,
            n_v=num_heads,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim_size,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=tpu_info.num_lanes,
            decode_vmem_limit_bytes=mixed_limit,
            mixed_vmem_limit_bytes=mixed_limit,
            window_size=1,
            attention_mode=config.AttentionMode.KDA,
            num_sublanes=tpu_info.num_sublanes,
        )
        self.assertEqual(mixed_tile, 128)

        seed = num_heads * 10 + (0 if gate_lower_bound is None else 1)
        rngs = iter(jax.random.split(jax.random.key(seed), 8))
        init_conv = jax.random.normal(
            next(rngs), (2, kernel_size - 1, dim_size), dtype=jnp.float32
        )
        init_rec = jax.random.normal(
            next(rngs), (2, num_heads, d_k, d_v), dtype=jnp.float32
        )
        kwargs = {
            "qkv": jax.random.normal(
                next(rngs), (tokens, dim_size), dtype=jnp.bfloat16
            ),
            "b": jax.random.normal(next(rngs), (tokens, num_heads), dtype=jnp.bfloat16),
            "a": jax.random.normal(
                next(rngs), (tokens, num_heads * d_k), dtype=jnp.bfloat16
            ),
            "conv_weight": jax.random.normal(
                next(rngs), (dim_size, 1, kernel_size), dtype=jnp.bfloat16
            ),
            "conv_bias": None,
            "a_log": jax.random.normal(next(rngs), (num_heads,), dtype=jnp.float32),
            "dt_bias": jax.random.normal(
                next(rngs), (num_heads, d_k), dtype=jnp.float32
            ),
            "query_start_loc": jnp.array([0, tokens], dtype=jnp.int32),
            "state_indices": jnp.array([1], dtype=jnp.int32),
            "distribution": jnp.array([0, 1, 1], dtype=jnp.int32),
            "seq_lens": jnp.array([tokens + 16], dtype=jnp.int32),
            "n_kq": num_heads,
            "n_v": num_heads,
            "d_k": d_k,
            "d_v": d_v,
            "kernel_size": kernel_size,
            "gate_lower_bound": gate_lower_bound,
        }
        (ref_conv, ref_rec), ref_out = kda_attention_ref(
            states=(init_conv, init_rec),
            **kwargs,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
            jit_fn = _fresh_jit_fused_conv1d_gdn()
            (new_conv, new_rec), out = jit_fn(
                states=(init_conv.copy(), init_rec.copy()),
                attention_mode=config.AttentionMode.KDA,
                **kwargs,
            )
        np.testing.assert_allclose(out, ref_out, rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(new_conv[1], ref_conv[1], rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(new_rec[1], ref_rec[1], rtol=2e-2, atol=1e-2)


class KdaTilingAndValidationTest(parameterized.TestCase):
    """CPU tests for KDA input guards, VMEM tiling, and wrapper dispatch."""

    @parameterized.named_parameters(
        {
            "testcase_name": "bound_under_gdn",
            "overrides": {
                "attention_mode": config.AttentionMode.GDN,
                "gate_lower_bound": -5.0,
            },
            "error": ValueError,
        },
        {
            "testcase_name": "non_negative_bound",
            "overrides": {"gate_lower_bound": 0.0},
            "error": ValueError,
        },
        {
            "testcase_name": "beta_scale_under_gdn",
            "overrides": {
                "attention_mode": config.AttentionMode.GDN,
                "beta_scale": 2.0,
            },
            "error": NotImplementedError,
        },
        {
            "testcase_name": "non_positive_beta_scale",
            "overrides": {"beta_scale": 0.0},
            "error": ValueError,
        },
    )
    def test_rejects(
        self, overrides: dict[str, object], error: type[Exception]
    ) -> None:
        """The entry point's guards, which all fail silently if dropped."""
        n_kq = n_v = 4
        d_k = d_v = 128
        kernel_size = 4
        dim_size = 2 * n_kq * d_k + n_v * d_v

        kwargs: dict[str, object] = {
            "qkv": jnp.zeros((8, dim_size)),
            "b": jnp.zeros((8, n_v)),
            "a": jnp.zeros((8, n_v * d_k)),
            "states": (
                jnp.zeros((9, kernel_size - 1, dim_size)),
                jnp.zeros((9, n_v, d_k, d_v)),
            ),
            "conv_weight": jnp.zeros((dim_size, 1, kernel_size)),
            "conv_bias": None,
            "a_log": jnp.zeros((n_v,)),
            "dt_bias": jnp.zeros((n_v, d_k)),
            "query_start_loc": jnp.arange(9, dtype=jnp.int32),
            "state_indices": jnp.arange(1, 9),
            "distribution": jnp.array([8, 8, 8], dtype=jnp.int32),
            "seq_lens": jnp.ones((8,), dtype=jnp.int32),
            "n_kq": n_kq,
            "n_v": n_v,
            "d_k": d_k,
            "d_v": d_v,
            "kernel_size": kernel_size,
            "attention_mode": config.AttentionMode.KDA,
        }
        kwargs.update(overrides)
        with self.assertRaises(error):
            wrapper.fused_conv1d_gdn(**kwargs)

    def test_kda_vmem_and_tile_sizes_across_parallelism(self) -> None:
        """Verify KDA VMEM estimate and tile selection across served regimes."""
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            d_k = d_v = 128
            kernel_size = 4
            tpu_info = pltpu.get_tpu_info()
            num_lanes = tpu_info.num_lanes
            num_sublanes = tpu_info.num_sublanes
            limit_w1 = config.GDNConfig.get_tiling_vmem_limit_bytes(1)

            # 1. KDA n_v=32, C=128 exceeds the 51.20 MiB W=1 limit (57.11 MiB).
            num_heads_dense = 32
            chunk_128 = 128
            dim_32 = 2 * num_heads_dense * d_k + num_heads_dense * d_v
            aligned_v_32 = tiling.align_to(num_heads_dense, num_lanes)
            aligned_gate_32 = tiling.align_to(num_heads_dense * d_k, num_lanes)
            aligned_out_32 = tiling.align_to(num_heads_dense * d_v, num_lanes)
            f32_bytes = 4
            bf16_bytes = 2
            expected_kda_c128_bytes = (
                # Double-buffered f32 inputs: qkv, b, a.
                2 * chunk_128 * (dim_32 + aligned_v_32 + aligned_gate_32) * f32_bytes
                # Double-buffered conv and recurrent states (2 slots/seq at W=1).
                + 4 * (kernel_size - 1) * dim_32 * f32_bytes
                + 4 * num_heads_dense * d_k * d_v * f32_bytes
                # Double-buffered bf16 output.
                + 2 * chunk_128 * aligned_out_32 * bf16_bytes
                # Prefill f32 conv and recurrent carry scratch.
                + (kernel_size - 1) * dim_32 * f32_bytes
                + num_heads_dense * d_k * d_v * f32_bytes
                # Static weights: conv [K+1, dim], a_log [1D], dt_bias [2D].
                + (kernel_size + 1) * dim_32 * f32_bytes
                + (aligned_v_32 + tiling.align_to(num_heads_dense, num_sublanes) * d_k)
                * f32_bytes
                # Intermediates: GDN chunk scratch + 3 live KDA decay tensors + temps.
                + (
                    num_heads_dense
                    * (
                        5 * chunk_128 * chunk_128
                        + 3 * chunk_128 * (d_v + d_k)
                        + 3 * chunk_128 * d_k
                    )
                    + num_sublanes * num_sublanes * num_lanes
                )
                * f32_bytes
            )
            est_kda_c128 = tiling.get_vmem_estimate_bytes(
                tile_b=1,
                chunk_sz=chunk_128,
                n_kq=num_heads_dense,
                n_v=num_heads_dense,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                act_out_bytes=bf16_bytes,
                rec_state_bytes=f32_bytes,
                num_lanes=num_lanes,
                conv_state_dim_size=dim_32,
                is_decode=False,
                window_size=1,
                attention_mode=config.AttentionMode.KDA,
                num_sublanes=num_sublanes,
            )
            self.assertEqual(est_kda_c128, expected_kda_c128_bytes)
            self.assertGreater(est_kda_c128, limit_w1)

            # 2. Verify tile picks across served Kimi-K3 and Kimi-Linear splits
            # at W=1 and W=4: Kimi-K3 TP32 (3, 3) and TP8 (12, 12);
            # Kimi-Linear-48B TP8 (4, 4) and DP8 (32, 32).
            regimes = [
                (3, 3, 1, (16, 128)),
                (3, 3, 4, (16, 128)),
                (12, 12, 1, (8, 64)),
                (12, 12, 4, (4, 64)),
                (4, 4, 1, (16, 128)),
                (4, 4, 4, (16, 128)),
                (32, 32, 1, (4, 64)),
                (32, 32, 4, (2, 64)),
            ]
            for n_kq, n_v, window_size, expected_tiles in regimes:
                dim_size = 2 * n_kq * d_k + n_v * d_v
                decode_limit = config.GDNConfig.get_tiling_vmem_limit_bytes(window_size)
                decode_tile, mixed_tile = tiling.get_tile_sizes(
                    batch_size=512,
                    num_seqs=64,
                    padded_batch_size=512,
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    conv_state_dim_size=dim_size,
                    act_out_dtype=jnp.dtype(jnp.bfloat16),
                    recurrent_state_dtype=jnp.dtype(jnp.float32),
                    num_lanes=num_lanes,
                    decode_vmem_limit_bytes=decode_limit,
                    mixed_vmem_limit_bytes=limit_w1,
                    window_size=window_size,
                    attention_mode=config.AttentionMode.KDA,
                    num_sublanes=num_sublanes,
                )
                self.assertEqual((decode_tile, mixed_tile), expected_tiles)

            # 3. Verify 0 B delta between declared VMEM bytes and create_allocs.
            f32 = jnp.dtype(jnp.float32)
            bf16 = jnp.dtype(jnp.bfloat16)
            dummy_meta = memory_ref.MetadataRef(
                num_tiles=None,
                p_id_to_s_idx=None,
                p_id_to_r_base=None,
                p_id_to_r_size=None,
                p_id_is_first_tile=None,
                p_id_is_last_tile=None,
                s_idx_has_initial_state=None,
                s_idx_to_state_indices=None,
                s_idx_to_read_offset=None,
                s_idx_to_ckpt_indices=None,
            )
            cases = [
                (mode, n_kq, n_v, tile_size, window_size)
                for mode in (config.GDNMode.BATCHED, config.GDNMode.PER_SEQ)
                for n_kq, n_v, tile_size, window_size in (
                    (3, 3, 8, 4),
                    (12, 12, 4, 4),
                    (4, 4, 16, 1),
                    (4, 4, 16, 4),
                    (32, 32, 2, 4),
                    (32, 32, 64, 1),
                )
                if not (mode == config.GDNMode.PER_SEQ and window_size > 1)
            ]
            for mode, n_kq, n_v, tile_size, window_size in cases:
                dim_size = 2 * n_kq * d_k + n_v * d_v
                cfg = config.GDNConfig(
                    mode=mode,
                    dtypes=config.Dtypes(
                        act_in=f32,
                        act_out=bf16,
                        compute=f32,
                        recurrent_state=f32,
                        conv_state=f32,
                    ),
                    batch_size=512,
                    dim_size=dim_size,
                    kernel_size=kernel_size,
                    tile_size=tile_size,
                    num_kq_heads=n_kq,
                    num_v_heads=n_v,
                    kq_head_dim=d_k,
                    v_head_dim=d_v,
                    attention_mode=config.AttentionMode.KDA,
                    window_size=window_size,
                )
                allocs = memory_ref.create_allocs(
                    metadata_ref=dummy_meta,
                    qkv_ref=f32,
                    b_ref=f32,
                    a_ref=f32,
                    out_ref=bf16,
                    conv_state_ref=f32,
                    recurrent_state_ref=f32,
                    cfg=cfg,
                )
                alloc_bytes = sum(
                    int(np.prod(a.window_ref.shape))
                    * jnp.dtype(a.window_ref.dtype).itemsize
                    for a in allocs
                    if a.window_ref is not None
                )
                scratch_bytes = sum(
                    int(np.prod(s.shape)) * jnp.dtype(s.dtype).itemsize
                    for s in cfg.get_scratch_shape_dict().values()
                    if s is not None
                )
                weights_bytes = weight_refs_vmem_bytes(cfg, num_lanes, num_sublanes)
                declared = get_vmem_declared_bytes(
                    cfg,
                    conv_state_dim_size=dim_size,
                    num_lanes=num_lanes,
                    num_sublanes=num_sublanes,
                )
                self.assertEqual(declared, alloc_bytes + scratch_bytes + weights_bytes)

    def test_kda_wrapper_compiles_dynamic_and_static_tiles(self) -> None:
        """Verify fused_conv1d_gdn lowers KDA with dynamic tiles on and T=1 off."""
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        d_k = d_v = 128
        kernel_size = 4

        def _trace_kernel_names(
            *,
            dynamic_tiling: bool,
            num_heads: int,
            tokens: int,
            num_seqs: int,
            num_spec_tokens: int,
        ) -> list[str]:
            """Trace fused_conv1d_gdn and return lowered Pallas kernel names."""
            dim_size = 2 * num_heads * d_k + num_heads * d_v
            window = num_spec_tokens + 1
            num_blocks = 1 + num_seqs * window
            is_spec = num_spec_tokens > 0
            traced_inputs = _TracedKdaInputs(
                qkv=jnp.zeros((tokens, dim_size), dtype=jnp.bfloat16),
                b=jnp.zeros((tokens, num_heads), dtype=jnp.bfloat16),
                a=jnp.zeros((tokens, num_heads * d_k), dtype=jnp.bfloat16),
                states=(
                    jnp.zeros(
                        (num_blocks, kernel_size - 1, dim_size), dtype=jnp.float32
                    ),
                    jnp.zeros((num_blocks, num_heads, d_k, d_v), dtype=jnp.float32),
                ),
                conv_weight=jnp.zeros((dim_size, 1, kernel_size), dtype=jnp.bfloat16),
                a_log=jnp.zeros((num_heads,), dtype=jnp.float32),
                dt_bias=jnp.zeros((num_heads, d_k), dtype=jnp.float32),
                query_start_loc=jnp.zeros((num_seqs + 1,), dtype=jnp.int32),
                state_indices=jnp.zeros((num_seqs,), dtype=jnp.int32),
                distribution=jnp.zeros((3,), dtype=jnp.int32),
                seq_lens=jnp.zeros((num_seqs,), dtype=jnp.int32),
                read_offsets=jnp.zeros((num_seqs,), dtype=jnp.int32)
                if is_spec
                else None,
            )

            def _wrapped_call(
                inputs: _TracedKdaInputs,
            ) -> tuple[jax.Array | tuple[jax.Array, jax.Array], jax.Array]:
                """Invoke fused_conv1d_gdn.__wrapped__ with traced tensor inputs."""
                return wrapper.fused_conv1d_gdn.__wrapped__(
                    qkv=inputs.qkv,
                    b=inputs.b,
                    a=inputs.a,
                    states=inputs.states,
                    conv_weight=inputs.conv_weight,
                    conv_bias=None,
                    a_log=inputs.a_log,
                    dt_bias=inputs.dt_bias,
                    query_start_loc=inputs.query_start_loc,
                    state_indices=inputs.state_indices,
                    distribution=inputs.distribution,
                    seq_lens=inputs.seq_lens,
                    n_kq=num_heads,
                    n_v=num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                    read_offsets=inputs.read_offsets,
                    num_spec_tokens=num_spec_tokens,
                    batched_only=is_spec,
                    prefill_only=not is_spec,
                    attention_mode=config.AttentionMode.KDA,
                    gate_lower_bound=-5.0 if is_spec else None,
                )

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", dynamic_tiling)
                mp.setattr(wrapper, "logger", mock.Mock())
                traced = jax.make_jaxpr(_wrapped_call)(traced_inputs)
            return [
                eqn.params["name"]
                for eqn in traced.jaxpr.eqns
                if "pallas" in eqn.primitive.name
            ]

        with use_abstract_mesh(mesh):
            self.assertEqual(
                _trace_kernel_names(
                    dynamic_tiling=True,
                    num_heads=3,
                    tokens=60,
                    num_seqs=24,
                    num_spec_tokens=3,
                ),
                ["fused_conv1d_kda_batched_b8_c4_w4_bounded"],
            )
            self.assertEqual(
                _trace_kernel_names(
                    dynamic_tiling=False,
                    num_heads=3,
                    tokens=60,
                    num_seqs=24,
                    num_spec_tokens=3,
                ),
                ["fused_conv1d_kda_batched_b1_c4_w4_bounded"],
            )
            self.assertEqual(
                _trace_kernel_names(
                    dynamic_tiling=True,
                    num_heads=3,
                    tokens=256,
                    num_seqs=1,
                    num_spec_tokens=0,
                ),
                ["fused_conv1d_kda_per_seq_b1_c128"],
            )
            self.assertEqual(
                _trace_kernel_names(
                    dynamic_tiling=True,
                    num_heads=32,
                    tokens=256,
                    num_seqs=1,
                    num_spec_tokens=0,
                ),
                ["fused_conv1d_kda_per_seq_b1_c64"],
            )

    @staticmethod
    def _l2_norm_f32_oracle(x: jax.Array, eps: float = 1e-6) -> jax.Array:
        """L2-normalize along the last dimension using float32 accumulation."""
        x_f32 = x.astype(jnp.float32)
        inv_norm = jax.lax.rsqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
        return (x_f32 * inv_norm).astype(x.dtype)

    @staticmethod
    def _per_seq_kda_token_loop_oracle(
        qkv: tuple[jax.Array, jax.Array, jax.Array],
        gates: tuple[jax.Array, jax.Array],
        state_prev: jax.Array,
        cfg: config.GDNConfig,
    ) -> tuple[jax.Array, jax.Array]:
        """Run un-fused per-sequence 2-dot reference KDA recurrence."""
        q_masked, k_masked, v_masked = qkv
        gating_decay, beta_masked = gates
        contract_dk = (((2,), (1,)), ((0,), (0,)))
        ref_outs, ref_states = [], []
        for seq_idx in range(cfg.seq_tile_size):
            state = state_prev[seq_idx]
            out_seq, state_seq = [], []
            for chunk_idx in range(cfg.chunk_size):
                q_heads = jnp.repeat(
                    q_masked[seq_idx, :, chunk_idx], cfg.v_per_kq_head, axis=0
                )[:, None, :]
                k_heads = jnp.repeat(
                    k_masked[seq_idx, :, chunk_idx], cfg.v_per_kq_head, axis=0
                )[:, None, :]
                k_heads_t = jnp.swapaxes(k_heads, 1, 2)
                state_updated = state * gating_decay[seq_idx, :, chunk_idx, :, None]
                v_updated = jax.lax.dot(
                    k_heads,
                    state_updated,
                    dimension_numbers=contract_dk,
                    preferred_element_type=jnp.float32,
                )
                v_new = beta_masked[seq_idx, :, chunk_idx : chunk_idx + 1] * (
                    v_masked[seq_idx, :, chunk_idx : chunk_idx + 1] - v_updated
                )
                state = state_updated + k_heads_t * v_new
                out = jax.lax.dot(
                    q_heads,
                    state,
                    dimension_numbers=contract_dk,
                    preferred_element_type=jnp.float32,
                )
                out_seq.append(out[:, 0, :])
                state_seq.append(state)
            ref_outs.append(jnp.stack(out_seq, axis=0))
            ref_states.append(jnp.stack(state_seq, axis=0))
        return jnp.stack(ref_outs, axis=0), jnp.stack(ref_states, axis=0)

    def test_recurrent_kda_batched_multi_token_window_with_ragged_masks(self) -> None:
        """Batched recurrent_kda matches per-seq 2-dot reference across W=1, 4."""
        spec_window_size = 4
        seq_tile_size = 8
        num_kq_heads = 4
        num_v_heads = 8
        kq_head_dim = v_head_dim = 128
        dim_size = 2 * num_kq_heads * kq_head_dim + num_v_heads * v_head_dim
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh), jax.default_matmul_precision("highest"):
            cases = [
                (window_size, act_dtype, gate_lower_bound)
                for window_size in (1, spec_window_size)
                for act_dtype in (jnp.float32, jnp.bfloat16)
                for gate_lower_bound in (None, -5.0)
            ]
            for window_size, act_dtype, gate_lower_bound in cases:
                cfg = config.GDNConfig(
                    mode=config.GDNMode.BATCHED,
                    dtypes=config.Dtypes(
                        act_in=jnp.dtype(act_dtype),
                        act_out=jnp.dtype(act_dtype),
                        compute=jnp.dtype(jnp.float32),
                        recurrent_state=jnp.dtype(jnp.float32),
                        conv_state=jnp.dtype(jnp.float32),
                    ),
                    batch_size=seq_tile_size * window_size,
                    dim_size=dim_size,
                    kernel_size=4,
                    tile_size=seq_tile_size,
                    num_kq_heads=num_kq_heads,
                    num_v_heads=num_v_heads,
                    kq_head_dim=kq_head_dim,
                    v_head_dim=v_head_dim,
                    num_buffers=2,
                    window_size=window_size,
                    attention_mode=config.AttentionMode.KDA,
                    gate_lower_bound=gate_lower_bound,
                )
                real_sizes = jnp.array(
                    [
                        window_size,
                        max(1, window_size - 1),
                        window_size,
                        1,
                        window_size,
                        window_size,
                        0,
                        window_size,
                    ],
                    dtype=jnp.int32,
                )
                q_compact = jax.random.normal(
                    jax.random.key(42),
                    (seq_tile_size, num_kq_heads, window_size, 1, kq_head_dim),
                    dtype=act_dtype,
                )
                k_compact = jax.random.normal(
                    jax.random.key(43),
                    (seq_tile_size, num_kq_heads, window_size, 1, kq_head_dim),
                    dtype=act_dtype,
                )
                v_compact = jax.random.normal(
                    jax.random.key(44),
                    (seq_tile_size, num_v_heads, window_size, 1, v_head_dim),
                    dtype=act_dtype,
                )
                b_compact = jax.random.normal(
                    jax.random.key(45),
                    (seq_tile_size, 1, window_size, 1, cfg.aligned_num_v_heads),
                    dtype=act_dtype,
                )
                a_compact = jax.random.normal(
                    jax.random.key(46),
                    (seq_tile_size, num_v_heads, window_size, 1, kq_head_dim),
                    dtype=act_dtype,
                )
                state_prev = jax.random.normal(
                    jax.random.key(47),
                    (seq_tile_size, num_v_heads, kq_head_dim, v_head_dim),
                    dtype=jnp.float32,
                )
                a_log = jax.random.normal(
                    jax.random.key(48), (num_v_heads,), dtype=jnp.float32
                )
                dt_bias = jax.random.normal(
                    jax.random.key(49), (num_v_heads, kq_head_dim), dtype=jnp.float32
                )

                jaxpr = jax.make_jaxpr(compute_kda.recurrent_kda, static_argnums=(9,))(
                    real_sizes,
                    q_compact,
                    k_compact,
                    v_compact,
                    b_compact,
                    a_compact,
                    state_prev,
                    a_log,
                    dt_bias,
                    cfg,
                )
                dot_eqns = [
                    eqn
                    for eqn in jaxpr.jaxpr.eqns
                    if eqn.primitive.name == "dot_general"
                ]
                self.assertLen(dot_eqns, window_size)
                for eqn in dot_eqns:
                    self.assertEqual(
                        eqn.invars[0].aval.shape,
                        (cfg.tile_size * cfg.num_v_heads, 2, cfg.kq_head_dim),
                    )
                    self.assertEqual(
                        eqn.invars[1].aval.shape,
                        (
                            cfg.tile_size * cfg.num_v_heads,
                            cfg.kq_head_dim,
                            cfg.v_head_dim,
                        ),
                    )

                out, state = compute_kda.recurrent_kda(
                    real_sizes,
                    q_compact,
                    k_compact,
                    v_compact,
                    b_compact,
                    a_compact,
                    state_prev,
                    a_log,
                    dt_bias,
                    cfg,
                )
                iota = jax.lax.broadcasted_iota(
                    jnp.int32, (seq_tile_size, 1, window_size, 1, 1), 2
                )
                mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1)
                q_masked = self._l2_norm_f32_oracle(
                    jnp.where(mask, q_compact.astype(jnp.float32), 0.0)
                ) * (kq_head_dim**-0.5)
                k_masked = self._l2_norm_f32_oracle(
                    jnp.where(mask, k_compact.astype(jnp.float32), 0.0)
                )
                v_masked = jnp.where(mask, v_compact.astype(jnp.float32), 0.0)
                beta_masked = jnp.where(
                    mask, jax.nn.sigmoid(b_compact.astype(jnp.float32)), 0.0
                )
                beta_masked = jnp.swapaxes(beta_masked, 1, 4)[:, :num_v_heads, :, 0, :]
                gating_log = jnp.where(
                    mask,
                    compute_kda.activate_gate(
                        a_compact,
                        a_log.reshape(1, num_v_heads, 1, 1, 1),
                        dt_bias.reshape(1, num_v_heads, 1, 1, kq_head_dim),
                        cfg,
                    ),
                    0.0,
                )
                gating_decay = jnp.exp(gating_log)[:, :, :, 0, :]
                ref_out, ref_state = self._per_seq_kda_token_loop_oracle(
                    (
                        q_masked[:, :, :, 0, :],
                        k_masked[:, :, :, 0, :],
                        v_masked[:, :, :, 0, :],
                    ),
                    (gating_decay, beta_masked),
                    state_prev,
                    cfg,
                )

                np.testing.assert_allclose(
                    np.asarray(out, np.float32),
                    np.asarray(ref_out, np.float32),
                    rtol=1e-4,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    np.asarray(state, np.float32),
                    np.asarray(ref_state, np.float32),
                    rtol=1e-4,
                    atol=1e-4,
                )
                # Masked-out rows and trailing checkpoints are bit-exact.
                np.testing.assert_allclose(
                    np.asarray(out[6], np.float32), 0.0, rtol=0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    np.asarray(state[6, -1], np.float32),
                    np.asarray(state_prev[6], np.float32),
                    rtol=0.0,
                    atol=0.0,
                )
                if window_size == spec_window_size:
                    np.testing.assert_allclose(
                        np.asarray(state[1, 3], np.float32),
                        np.asarray(state[1, 2], np.float32),
                        rtol=0.0,
                        atol=0.0,
                    )
                    np.testing.assert_allclose(
                        np.asarray(state[3, 3], np.float32),
                        np.asarray(state[3, 0], np.float32),
                        rtol=0.0,
                        atol=0.0,
                    )

    def _wide_value_head_kwargs(self, states, **overrides):
        n_kq = n_v = 1
        d_k, d_v = 256, 512
        kernel_size = 4
        dim_size = 2 * n_kq * d_k + n_v * d_v
        kwargs = dict(
            qkv=jnp.zeros((8, dim_size)),
            b=jnp.zeros((8, n_v)),
            a=jnp.zeros((8, n_v * d_k)),
            states=states,
            conv_weight=jnp.zeros((dim_size, 1, kernel_size)),
            conv_bias=None,
            a_log=jnp.zeros((n_v,)),
            dt_bias=jnp.zeros((n_v, d_k)),
            query_start_loc=jnp.arange(9, dtype=jnp.int32),
            state_indices=jnp.arange(1, 9),
            distribution=jnp.array([8, 8, 8], dtype=jnp.int32),
            seq_lens=jnp.ones((8,), dtype=jnp.int32),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            attention_mode=config.AttentionMode.KDA,
        )
        kwargs.update(overrides)
        return kwargs

    def test_rejects_wide_value_heads_on_the_unified_pool(self):
        """The unified pool's lane-row granules need d_v == 128."""
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        # fp32 unified pool: [blocks, granules, packing=1, lanes].
        kwargs = self._wide_value_head_kwargs(
            jnp.zeros((9, 64, 1, 128)),
            conv_dtype=jnp.float32,
            recurrent_dtype=jnp.float32,
        )
        with (
            use_abstract_mesh(mesh),
            self.assertRaisesRegex(NotImplementedError, "d_v == 128"),
        ):
            wrapper.fused_conv1d_gdn(**kwargs)

    def test_rejects_wide_value_heads_with_a_bf16_recurrent_state(self):
        """The full-row granule for d_v > 128 is only used with fp32 states."""
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        kwargs = self._wide_value_head_kwargs(
            (
                jnp.zeros((9, 3, 1024)),
                jnp.zeros((9, 1, 256, 512), dtype=jnp.bfloat16),
            )
        )
        with (
            use_abstract_mesh(mesh),
            self.assertRaisesRegex(NotImplementedError, "bf16 recurrent"),
        ):
            wrapper.fused_conv1d_gdn(**kwargs)


@pytest.mark.tpu_test
class KdaCustomOpWideValueHeadTest(parameterized.TestCase):
    """The fused KDA custom op, not just the kernel, with d_v != d_k and a
    beta scale: the relaxed ABI check, the `head_v_dim` output reshape, the
    verify path's `empty_out` shape and the `beta_scale` plumbing through
    `_build_fused_core`.
    """

    @parameterized.named_parameters(
        dict(
            testcase_name="decode_and_prefill",
            q_loc=[0, 1, 129, 200],
            distribution=[1, 3, 3],
            num_spec_tokens=0,
        ),
        # With speculative decoding on and no verify window in the batch, the
        # verify segment is skipped and `empty_out` stands in for its output;
        # a wrong width would fail to combine with the prefill output.
        dict(
            testcase_name="speculative_empty_verify_window",
            q_loc=[0, 70, 129],
            distribution=[0, 2, 2],
            num_spec_tokens=1,
        ),
    )
    def test_op_matches_the_recurrence(self, q_loc, distribution, num_spec_tokens):
        num_heads, d_k, d_v, kernel_size = 1, 256, 512, 4
        beta_scale, eps = 2.0, 1e-5
        max_reqs = len(q_loc) - 1
        q_loc = jnp.array(q_loc, dtype=jnp.int32)
        distribution = jnp.array(distribution, dtype=jnp.int32)
        num_tokens = int(q_loc[-1])
        dim = num_heads * (2 * d_k + d_v)
        num_slots = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(1), 9))
        mixed_qkv = jax.random.normal(next(rngs), (num_tokens, dim))
        raw_gate = jax.random.normal(next(rngs), (num_tokens, num_heads * d_k))
        beta = jax.random.normal(next(rngs), (num_tokens, num_heads))
        output_gate = jax.random.normal(next(rngs), (num_tokens, num_heads * d_v))
        # The op takes [kernel_size, dim] with channels in qkv order.
        conv_weight = jax.random.normal(next(rngs), (kernel_size, dim))
        a_log = jax.random.normal(next(rngs), (num_heads,))
        dt_bias = jax.random.normal(next(rngs), (num_heads * d_k,))
        norm_weight = 1.0 + 0.1 * jax.random.normal(next(rngs), (d_v,))
        conv_state = jnp.zeros((num_slots, kernel_size - 1, dim))
        recurrent_state = jnp.zeros((num_slots, num_heads, d_k, d_v))
        state_indices = jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
        seq_lens = (q_loc[1:] - q_loc[:-1]).astype(jnp.int32)

        (ref_conv, ref_state), ref_out = kda_attention_ref(
            qkv=mixed_qkv,
            b=beta,
            a=raw_gate,
            states=(conv_state, recurrent_state),
            conv_weight=jnp.transpose(conv_weight)[:, None, :],
            conv_bias=None,
            a_log=a_log,
            dt_bias=dt_bias.reshape(num_heads, d_k),
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=num_heads,
            n_v=num_heads,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            gate_lower_bound=None,
            beta_scale=beta_scale,
        )
        ref_out = kda_attention_op._gated_output_norm(
            jnp.asarray(ref_out, jnp.float32).reshape(num_tokens, num_heads, d_v),
            output_gate,
            norm_weight,
            eps,
            jnp.float32,
        )

        core = kda_attention_op._build_fused_core(
            None, eps, num_spec_tokens=num_spec_tokens, beta_scale=beta_scale
        )
        out, new_conv, new_state = jax.jit(core)(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            conv_state,
            recurrent_state,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            q_loc,
            state_indices,
            seq_lens,
            distribution,
            slot_read_offsets=jnp.zeros((num_slots,), jnp.int32),
        )

        self.assertEqual(out.shape, (num_tokens, num_heads, d_v))
        np.testing.assert_allclose(
            np.asarray(out, np.float64),
            np.asarray(ref_out, np.float64),
            rtol=2e-2,
            atol=1e-2,
        )
        # The conv state update is pure data movement.
        np.testing.assert_array_equal(np.asarray(new_conv), np.asarray(ref_conv))
        np.testing.assert_allclose(
            np.asarray(new_state, np.float64),
            np.asarray(ref_state, np.float64),
            rtol=2e-2,
            atol=1e-2,
        )
