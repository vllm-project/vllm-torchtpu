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

import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import config, tiling, wrapper


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def gdn_attention_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
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
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Runs reference GDN attention sequence-by-sequence in eager mode.

    Bypasses XLA ragged masking complexity required for jax.jit by slicing
    sequence chunks directly in Python loop using query_start_loc boundaries.
    """

    num_tokens = qkv.shape[0]
    num_valid_seqs = int(distribution[2])

    out_mixed_qkv = jnp.zeros_like(qkv)
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

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

        # Part 1: Conv1D
        X = qkv[start:end]
        x_full = jnp.concatenate([c_state, X], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k in range(kernel_size):
            acc += (
                x_full[k : k + query_len].astype(jnp.float32)
                * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
            )
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = acc.astype(qkv.dtype)
        new_conv_state = new_conv_state.at[s].set(x_full[-(kernel_size - 1) :])
        out_mixed_qkv = out_mixed_qkv.at[start:end].set(conv_out)

    out_mixed_qkv = jax.nn.silu(out_mixed_qkv)

    for req_idx in range(num_valid_seqs):
        s = int(state_indices[req_idx])
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        r_state = new_recurrent_state[s]
        qkv_seq = out_mixed_qkv[start:end]
        key_dim = n_kq * d_k
        q_seq = qkv_seq[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = qkv_seq[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = qkv_seq[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        b_seq = b[start:end]
        a_seq = a[start:end]

        beta_seq = jax.nn.sigmoid(b_seq.astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a_seq.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
        )

        def step_fn(carry_state, xs):
            q_t, k_t, v_t, beta_t, g_t = xs
            q_t = q_t * (d_k**-0.5)
            exp_g = jnp.exp(g_t)

            k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
            v_diff = v_t - exp_g[:, None] * k_state
            v_new = beta_t[:, None] * v_diff

            q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
            q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            out_step = exp_g[:, None] * q_state + q_k * v_new

            k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
            new_state = carry_state * exp_g[:, None, None] + k_v_new

            return new_state.astype(recurrent_state.dtype), out_step.astype(qkv.dtype)

        final_r_state, out_seq = jax.lax.scan(
            step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
        )
        new_recurrent_state = new_recurrent_state.at[s].set(final_r_state)
        output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))

    return (new_conv_state, new_recurrent_state), output


def gdn_attention_spec_ref(
    qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    conv_state: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: jnp.ndarray | None,
    a_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    read_offsets: jnp.ndarray,
    num_spec_seqs: int,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Reference for SPEC mode: per-token state checkpoints with read offsets.

    Sequence s reads its initial state from slot
    `state_indices[s] + read_offsets[s]` and writes the state after window
    position t to slot `state_indices[s] + t`.
    """
    num_tokens = qkv.shape[0]
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((num_tokens, n_v * d_v), dtype=qkv.dtype)

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
        r_state = (
            recurrent_state[read_slot]
            if has_init
            else jnp.zeros_like(recurrent_state[read_slot])
        )

        # Conv1D with per-token windows.
        X = qkv[start:end]
        x_full = jnp.concatenate([c_state, X], axis=0)
        acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
        for k in range(kernel_size):
            acc += (
                x_full[k : k + query_len].astype(jnp.float32)
                * conv_weight[:, 0, k].astype(jnp.float32)[None, :]
            )
        if conv_bias is not None:
            acc += conv_bias.astype(jnp.float32)[None, :]
        conv_out = jax.nn.silu(acc).astype(qkv.dtype)
        for t in range(query_len):
            new_conv_state = new_conv_state.at[base + t].set(
                x_full[t + 1 : t + kernel_size]
            )

        # Recurrent GDN with per-token states.
        key_dim = n_kq * d_k
        q_seq = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
        k_seq = conv_out[:, key_dim : key_dim * 2].reshape(query_len, n_kq, d_k)
        v_seq = conv_out[:, key_dim * 2 :].reshape(query_len, n_v, d_v)

        repeat_factor = n_v // n_kq
        if repeat_factor > 1:
            q_seq = jnp.repeat(q_seq, repeat_factor, axis=1)
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)

        q_seq = _l2_normalize(q_seq)
        k_seq = _l2_normalize(k_seq)
        v_seq = v_seq.astype(jnp.float32)

        beta_seq = jax.nn.sigmoid(b[start:end].astype(jnp.float32))
        g_seq = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
            a[start:end].astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
        )

        def step_fn(carry_state, xs):
            q_t, k_t, v_t, beta_t, g_t = xs
            q_t = q_t * (d_k**-0.5)
            exp_g = jnp.exp(g_t)

            k_state = jnp.einsum("hd, hdm -> hm", k_t, carry_state)
            v_diff = v_t - exp_g[:, None] * k_state
            v_new = beta_t[:, None] * v_diff

            q_state = jnp.einsum("hd, hdm -> hm", q_t, carry_state)
            q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
            out_step = exp_g[:, None] * q_state + q_k * v_new

            k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
            new_state = carry_state * exp_g[:, None, None] + k_v_new
            new_state = new_state.astype(recurrent_state.dtype)

            return new_state, (out_step.astype(qkv.dtype), new_state)

        _, (out_seq, states_seq) = jax.lax.scan(
            step_fn, r_state, (q_seq, k_seq, v_seq, beta_seq, g_seq)
        )
        for t in range(query_len):
            new_recurrent_state = new_recurrent_state.at[base + t].set(states_seq[t])
        output = output.at[start:end].set(out_seq.reshape(query_len, n_v * d_v))

    return (new_conv_state, new_recurrent_state), output


class GDNAttentionTileSelectionTest(parameterized.TestCase):
    """Selects optimal (decode_tile, mixed_tile) across Qwen3.5-397B parallelisms."""

    def test_qwen3_5_397b_parallelism_tile_selection(self):
        d_k, d_v, kernel_size = 128, 128, 4
        f32 = jnp.dtype(jnp.float32)
        bf16 = jnp.dtype(jnp.bfloat16)

        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            limit_w1 = config.GDNConfig.get_vmem_limit_bytes(1)
            limit_w4 = config.GDNConfig.get_vmem_limit_bytes(4)
            mixed_limit = config.GDNConfig.get_vmem_limit_bytes()

            # (label, n_kq, n_v, b, n, w, state_dtype, expected)
            cases = [
                # TP8 + DCP=4 (n_kq=2, n_v=8, W=4 for MTP=3): warmup & verify buckets
                ("tp8_dcp4_c16_b4", 2, 8, 4, 16, 4, f32, (2, 4)),
                ("tp8_dcp4_c16_b8", 2, 8, 8, 16, 4, f32, (4, 8)),
                ("tp8_dcp4_c16_b16", 2, 8, 16, 16, 4, f32, (4, 16)),
                ("tp8_dcp4_c16_b32", 2, 8, 32, 16, 4, f32, (4, 32)),
                ("tp8_dcp4_c16_b64", 2, 8, 64, 16, 4, f32, (4, 64)),
                ("tp8_dcp4_c32_b128", 2, 8, 128, 32, 4, f32, (8, 128)),
                ("tp8_dcp4_c64_b256", 2, 8, 256, 64, 4, f32, (8, 128)),
                # TP8 + DCP=1 (n_kq=2, n_v=8, W=1 and W=4) and padded slot table
                ("tp8_dcp1_c8_mtp3_b32", 2, 8, 32, 8, 4, f32, (4, 32)),
                ("tp8_dcp1_c32_b16", 2, 8, 16, 32, 1, f32, (4, 16)),
                ("tp8_dcp1_c64_b64", 2, 8, 64, 64, 1, f32, (8, 64)),
                ("tp8_padded_n512_b64", 2, 8, 64, 512, 1, bf16, (8, 64)),
                # PCP8 + TP1 disagg prefill (n_kq=2, n_v=8, W=1)
                ("pcp8_prefill_b4096", 2, 8, 4096, 16, 1, f32, (4, 128)),
                # DP4 + TP2 + EP8 (n_kq=8, n_v=32, W=1 and W=4)
                ("dp4tp2_c64_b8", 8, 32, 8, 32, 1, f32, (2, 8)),
                ("dp4tp2_c64_b32", 8, 32, 32, 32, 1, f32, (4, 32)),
                ("dp4tp2_c64_b2048", 8, 32, 2048, 32, 1, f32, (4, 128)),
                ("dp4tp2_mtp3_c16_b64_f32", 8, 32, 64, 16, 4, f32, (2, 64)),
                ("dp4tp2_mtp3_c32_b128_f32", 8, 32, 128, 32, 4, f32, (2, 128)),
                ("dp4tp2_mtp3_c32_b128_bf16", 8, 32, 128, 32, 4, bf16, (4, 128)),
                # DP8 + TP1 + EP8 (n_kq=16, n_v=64, W=1 and W=4)
                ("dp8_c128_b8", 16, 64, 8, 16, 1, f32, (1, 8)),
                ("dp8_c128_b16", 16, 64, 16, 16, 1, f32, (2, 16)),
                ("dp8_c256_b32_f32", 16, 64, 32, 32, 1, f32, (2, 32)),
                ("dp8_c256_b32_bf16", 16, 64, 32, 32, 1, bf16, (2, 32)),
                ("dp8_c256_b320_f32", 16, 64, 320, 32, 1, f32, (2, 64)),
                ("dp8_mtp3_c32_b128_f32", 16, 64, 128, 32, 4, f32, (1, 64)),
                ("dp8_mtp3_c32_b128_bf16", 16, 64, 128, 32, 4, bf16, (2, 64)),
            ]
            for label, n_kq, n_v, b, n, w, state_dtype, expected in cases:
                with self.subTest(label=label):
                    dim = 2 * n_kq * d_k + n_v * d_v
                    got = tiling.get_tile_sizes(
                        batch_size=b,
                        num_seqs=n,
                        padded_batch_size=b,
                        n_kq=n_kq,
                        n_v=n_v,
                        d_k=d_k,
                        d_v=d_v,
                        kernel_size=kernel_size,
                        conv_state_dim_size=dim,
                        act_out_dtype=bf16,
                        recurrent_state_dtype=state_dtype,
                        num_lanes=128,
                        decode_vmem_limit_bytes=limit_w4 if w > 1 else limit_w1,
                        mixed_vmem_limit_bytes=mixed_limit,
                        window_size=w,
                    )
                    self.assertEqual(got, expected, msg=label)

    def test_tile_sizes_with_split_state_slots_at_w1(self):
        """W=1 decode tiles count 1 input and 1 output state slot per sequence."""
        d_k, d_v, kernel_size = 128, 128, 4
        bf16 = jnp.dtype(jnp.bfloat16)
        f32 = jnp.dtype(jnp.float32)
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        with use_abstract_mesh(mesh):
            limit_w1 = config.GDNConfig.get_vmem_limit_bytes(1)
            # One state slot per sequence would pick T=8 and T=4 here.
            cases = [
                ("dp4tp2_c128_b128_f32", 8, 32, 128, 128, (4, 128)),
                ("dp8_c64_b64_f32", 16, 64, 64, 64, (2, 64)),
            ]
            for label, n_kq, n_v, b, n, expected in cases:
                with self.subTest(label=label):
                    got = tiling.get_tile_sizes(
                        batch_size=b,
                        num_seqs=n,
                        padded_batch_size=b,
                        n_kq=n_kq,
                        n_v=n_v,
                        d_k=d_k,
                        d_v=d_v,
                        kernel_size=kernel_size,
                        conv_state_dim_size=2 * n_kq * d_k + n_v * d_v,
                        act_out_dtype=bf16,
                        recurrent_state_dtype=f32,
                        num_lanes=128,
                        decode_vmem_limit_bytes=limit_w1,
                        mixed_vmem_limit_bytes=limit_w1,
                        window_size=1,
                    )
                    self.assertEqual(got, expected, msg=label)


class GDNStatePipelineRaceTest(parameterized.TestCase):
    """CPU interpret-mode checks of the state DMA pipeline."""

    def test_state_pipeline_has_no_dma_race(self):
        """Pallas interpret-mode vector-clock detector finds zero DMA races."""
        n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        # 6 decode sequences, then 3 one-token prefill sequences (one PER_SEQ tile
        # each), so both kernels reuse a slot two steps later.
        num_decode = 6
        num_seqs = 9
        decode_tile = 2  # 3 pipeline steps: step 2 reuses step 0's slot.

        rngs = iter(jax.random.split(jax.random.key(0), 9))
        qkv = jax.random.normal(next(rngs), (num_seqs, dim), dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16)
        conv_w = jax.random.normal(
            next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
        )
        conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
        a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        init_conv = jax.random.normal(
            next(rngs), (num_seqs, kernel_size - 1, dim), dtype=jnp.bfloat16
        )
        init_rec = jax.random.normal(
            next(rngs), (num_seqs, n_v, d_k, d_v), dtype=jnp.float32
        )

        q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
        distribution = jnp.array([num_decode, num_seqs, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.full((num_seqs,), 11, dtype=jnp.int32)

        # OSS jax 0.11.x has no InterpretParams.on_race, so read the detector's
        # verdict before each pallas_call clears the shared state.
        mosaic_interpret = sys.modules[pltpu.reset_tpu_interpret_mode_state.__module__]
        races_per_call: list[bool] = []
        orig_clear = mosaic_interpret._clear_shared_memory

        def _record_and_clear():
            races = getattr(mosaic_interpret, "races", None)
            if races is not None:
                races_per_call.append(bool(races.races_found))
            orig_clear()

        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu")
        )
        pltpu.reset_tpu_interpret_mode_state()
        mosaic_interpret._clear_shared_memory = _record_and_clear
        try:
            with (
                use_abstract_mesh(mesh),
                mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
                pltpu.force_tpu_interpret_mode(
                    pltpu.InterpretParams(
                        detect_races=True,
                        dma_execution_mode="eager",
                        skip_floating_point_ops=False,
                        vector_clock_size=128,
                    )
                ),
            ):
                (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                    qkv=qkv,
                    b=b,
                    a=a,
                    conv_state=init_conv,
                    recurrent_state=init_rec,
                    conv_weight=conv_w,
                    conv_bias=conv_b,
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
                    decode_tile_size=decode_tile,
                    # CPU interpret mode does not support ref.reshape on the C > 1
                    # chunked prefill path; each 1-token prefill is one tile at C = 1.
                    mixed_tile_size=1,
                )
                jax.block_until_ready((conv_out, rec_out, out))
            self.assertTrue(races_per_call)
            self.assertFalse(
                any(races_per_call),
                f"DMA race detected across pallas_call phases: {races_per_call}",
            )
        finally:
            mosaic_interpret._clear_shared_memory = orig_clear
            pltpu.reset_tpu_interpret_mode_state()


@pytest.mark.tpu_test
class GDNAttentionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(
            testcase_name="prefill",
            max_reqs=1,
            lengths=[8192],
            q_loc=[0, 8192],
            distribution=[0, 0, 3],
        ),
        dict(
            testcase_name="mixed",
            max_reqs=3,
            lengths=[256, 128, 128],
            q_loc=[0, 256, 384, 512],
            distribution=[0, 3, 3],
        ),
        dict(
            testcase_name="decode_only",
            max_reqs=64,
            lengths=[1] * 64,
            q_loc=list(range(65)),
            distribution=[64, 64, 64],
        ),
        dict(
            testcase_name="mixed_prefill_decode",
            max_reqs=11,
            lengths=[1] * 8 + [128, 128, 256],
            q_loc=[0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520],
            distribution=[8, 11, 11],
        ),
        dict(
            testcase_name="padded_mixed_prefill",
            max_reqs=16,
            lengths=[128, 64, 32, 16, 8],
            q_loc=[0, 128, 192, 224, 240, 248] + [1] * 11,
            distribution=[0, 5, 5],
        ),
        dict(
            testcase_name="padded_decode_only",
            max_reqs=512,
            lengths=[1] * 64,
            q_loc=list(range(65)) + [1] * 448,
            distribution=[64, 64, 64],
        ),
        dict(
            testcase_name="prefill_fused",
            max_reqs=1,
            lengths=[8192],
            q_loc=[0, 8192],
            distribution=[0, 0, 3],
        ),
        dict(
            testcase_name="mixed_fused",
            max_reqs=3,
            lengths=[256, 128, 128],
            q_loc=[0, 256, 384, 512],
            distribution=[0, 3, 3],
        ),
    )
    def test_run_jax_gdn_attention_local(self, max_reqs, lengths, q_loc, distribution):
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        num_tokens = sum(lengths)

        q_loc = jnp.array(q_loc)
        distribution = jnp.array(distribution, dtype=jnp.int32)

        # recurrent_state[0] and conv_state[0] are reserved for null blocks
        # (invalid / padded tokens). so start with index 1
        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(0), 12))

        query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_state_q = jnp.zeros((num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_k = jnp.zeros((num_blocks, kernel_size - 1, n_kq * kq_head_dim))
        conv_state_v = jnp.zeros((num_blocks, kernel_size - 1, n_v * v_head_dim))
        recurrent_state = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

        conv_weight_q = jax.random.normal(
            next(rngs), (n_kq * kq_head_dim, 1, kernel_size)
        )
        conv_weight_k = jax.random.normal(
            next(rngs), (n_kq * kq_head_dim, 1, kernel_size)
        )
        conv_weight_v = jax.random.normal(
            next(rngs), (n_v * v_head_dim, 1, kernel_size)
        )

        conv_bias_q = jax.random.normal(next(rngs), (n_kq * kq_head_dim,))
        conv_bias_k = jax.random.normal(next(rngs), (n_kq * kq_head_dim,))
        conv_bias_v = jax.random.normal(next(rngs), (n_v * v_head_dim,))

        A_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(jax.random.key(0), (n_v,))

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)
        conv_state = jnp.concatenate(
            [conv_state_q, conv_state_k, conv_state_v], axis=-1
        )
        conv_weight = jnp.concatenate(
            [conv_weight_q, conv_weight_k, conv_weight_v], axis=0
        )
        conv_bias = jnp.concatenate([conv_bias_q, conv_bias_k, conv_bias_v], axis=-1)

        gdn_attention_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )

        # All sequences in this test start from a fresh slot; the existing
        # parametrizations don't exercise prefix-cache-hit / chunked-prefill
        # continuation. ``seq_lens == query_lens`` (context_len = 0)
        # reproduces the prior behavior (zero initial state regardless of
        # slot contents).
        seq_lens = jnp.asarray(
            q_loc[1 : max_reqs + 1] - q_loc[:max_reqs], dtype=jnp.int32
        )

        common_kwargs = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Run ref
        new_states_ref, output_ref = gdn_attention_ref(**common_kwargs)

        # Run chunked
        new_states_chunked, output_chunked = gdn_attention_jitted(**common_kwargs)

        # Compare results
        np.testing.assert_allclose(output_chunked, output_ref, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(
            new_states_chunked[0], new_states_ref[0], rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            new_states_chunked[1], new_states_ref[1], rtol=2e-2, atol=2e-2
        )

    @parameterized.named_parameters(
        dict(
            testcase_name="spec_windows",
            # 3 verify windows: full (5), partial drafts (3), no drafts (1).
            spec_lengths=[5, 3, 1],
            read_offsets=[2, 0, 4],
            prefill_lengths=[],
        ),
        dict(
            testcase_name="spec_windows_padded",
            # More windows than one tile (decode_tile_size=4).
            spec_lengths=[5, 1, 3, 2, 4],
            read_offsets=[0, 1, 2, 3, 4],
            prefill_lengths=[],
        ),
        dict(
            testcase_name="spec_and_prefill",
            spec_lengths=[5, 2],
            read_offsets=[3, 1],
            prefill_lengths=[64],
        ),
    )
    def test_spec_mode_checkpoints(self, spec_lengths, read_offsets, prefill_lengths):
        """SPEC mode: initial state comes from `base + read_offset` and one
        state checkpoint per window position lands at `base + t`, matching a
        token-by-token eager reference. Prefill sequences in the same batch
        keep the original PER_SEQ behavior (read/write the base slot)."""
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4
        num_spec_tokens = 4
        window = num_spec_tokens + 1

        lengths = list(spec_lengths) + list(prefill_lengths)
        num_seqs = len(lengths)
        num_spec_seqs = len(spec_lengths)
        num_tokens = sum(lengths)
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]), dtype=jnp.int32)
        distribution = jnp.array(
            [num_spec_seqs, num_spec_seqs, num_seqs], dtype=jnp.int32
        )

        # Slot groups of `window` consecutive slots per sequence; slot 0 is
        # the null block.
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        num_blocks = 1 + num_seqs * window
        read_offsets_arr = jnp.array(
            list(read_offsets) + [0] * len(prefill_lengths), dtype=jnp.int32
        )

        rngs = iter(jax.random.split(jax.random.key(3), 12))
        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim

        mixed_qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        # Every slot holds a distinct random state so a read from the wrong
        # slot (or a write to the wrong slot) is caught.
        conv_state = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, conv_dim)
        )
        recurrent_state = jax.random.normal(
            next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
        )

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        A_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        # All sequences continue an existing context (has_initial=True) for
        # the spec windows; prefills start fresh (context_len = 0).
        seq_lens = jnp.array(
            [16 + sl for sl in spec_lengths] + list(prefill_lengths), dtype=jnp.int32
        )

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=[
                "n_kq",
                "n_v",
                "d_k",
                "d_v",
                "kernel_size",
                "num_spec_tokens",
            ],
        )

        (new_conv, new_rec), output = run_jitted(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            q_loc,
            state_indices,
            distribution,
            seq_lens,
            read_offsets_arr,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
        )

        # Reference: spec windows token-by-token with checkpoints.
        (ref_conv, ref_rec), ref_output = gdn_attention_spec_ref(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            read_offsets=read_offsets_arr,
            num_spec_seqs=num_spec_seqs,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )
        # Reference for the prefill tail: original eager reference over the
        # same buffers (spec sequences excluded via the distribution).
        if prefill_lengths:
            spec_token_end = int(q_loc[num_spec_seqs])
            (ref_conv, ref_rec), prefill_output = gdn_attention_ref(
                qkv=mixed_qkv[spec_token_end:],
                b=b[spec_token_end:],
                a=a[spec_token_end:],
                conv_state=ref_conv,
                recurrent_state=ref_rec,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                a_log=A_log,
                dt_bias=dt_bias,
                query_start_loc=q_loc[num_spec_seqs:] - q_loc[num_spec_seqs],
                state_indices=state_indices[num_spec_seqs:],
                distribution=jnp.array(
                    [0, 0, num_seqs - num_spec_seqs], dtype=jnp.int32
                ),
                seq_lens=seq_lens[num_spec_seqs:],
                n_kq=n_kq,
                n_v=n_v,
                d_k=kq_head_dim,
                d_v=v_head_dim,
                kernel_size=kernel_size,
            )
            ref_output = ref_output.at[spec_token_end:].set(prefill_output)

        np.testing.assert_allclose(output, ref_output, rtol=2e-2, atol=2e-2)

        # Verify every written checkpoint slot; untouched slots must keep
        # their original contents.
        touched = set()
        for i, sl in enumerate(spec_lengths):
            base = int(state_indices[i])
            touched.update(range(base, base + sl))
        for i in range(len(prefill_lengths)):
            touched.add(int(state_indices[num_spec_seqs + i]))
        for slot in range(num_blocks):
            expected_conv = ref_conv[slot] if slot in touched else conv_state[slot]
            expected_rec = ref_rec[slot] if slot in touched else (recurrent_state[slot])
            np.testing.assert_allclose(
                new_conv[slot],
                expected_conv,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"conv checkpoint mismatch at slot {slot}",
            )
            np.testing.assert_allclose(
                new_rec[slot],
                expected_rec,
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"recurrent checkpoint mismatch at slot {slot}",
            )

    def test_has_initial_state_zeros_stale_slot(self):
        """Ensure stale states are ignore by new request.

        A new prefill landing on a slot whose previous tenant left
        non-zero state must produce the same output and final state as it
        would on a fresh-zero slot. This exercises the production bug fixed
        by the `has_initial_state` plumbing: vLLM's mamba pool reuses
        freed slots without clearing them, and previously the TPU GDN
        kernel consumed the stale state, silently corrupting the new
        request's recurrent trajectory.
        """
        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        # Two requests, one prefill of 64 tokens each.
        max_reqs = 2
        lengths = [64, 64]
        q_loc = jnp.array([0, 64, 128])
        distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
        num_tokens = sum(lengths)

        state_indices = jnp.arange(1, max_reqs + 1)
        num_blocks = max_reqs + 1

        rngs = iter(jax.random.split(jax.random.key(7), 12))
        query = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (num_tokens, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (num_tokens, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))

        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
        conv_state_fresh = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
        recurrent_state_fresh = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

        # Build a "stale" pair where the slots that the two new requests
        # land on are filled with arbitrary nonzero values (simulating a
        # prior request that finished without the pool clearing the slot).
        stale_conv = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, conv_dim)
        )
        stale_recurrent = jax.random.normal(
            next(rngs), (num_blocks, n_v, kq_head_dim, v_head_dim)
        )
        # Slot 0 is the null block; leave it zero.
        conv_state_stale = conv_state_fresh.at[1:].set(stale_conv[1:])
        recurrent_state_stale = recurrent_state_fresh.at[1:].set(stale_recurrent[1:])

        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        A_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        mixed_qkv = jnp.concatenate([query, key, value], axis=-1)

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )

        # Both requests are brand new: no prior context. seq_lens equals
        # query_lens so context_len = 0, which gives has_initial_state = False.
        seq_lens_new = jnp.asarray(lengths, dtype=jnp.int32)

        common_kwargs = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens_new,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Reference run: fresh-zero slots.
        (new_conv_fresh, new_rec_fresh), output_fresh = run_jitted(
            conv_state=conv_state_fresh,
            recurrent_state=recurrent_state_fresh,
            **common_kwargs,
        )
        # Stale-slot run: same inputs, but the slots already contain a
        # prior request's state. With the fix, has_initial_state=False
        # masks that out; outputs and the writeback at the active slots
        # must match the fresh-zero run.
        (new_conv_stale, new_rec_stale), output_stale = run_jitted(
            conv_state=conv_state_stale,
            recurrent_state=recurrent_state_stale,
            **common_kwargs,
        )

        np.testing.assert_allclose(output_fresh, output_stale, rtol=1e-5, atol=1e-5)
        # Compare the active slots (1..max_reqs+1); slot 0 (null) is
        # untouched in both runs and inactive slots are unused.
        np.testing.assert_allclose(
            new_conv_fresh[1 : max_reqs + 1],
            new_conv_stale[1 : max_reqs + 1],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            new_rec_fresh[1 : max_reqs + 1],
            new_rec_stale[1 : max_reqs + 1],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_has_initial_state_preserves_continuation(self):
        """Ensure existing slots are read during multi-step execution.

        When `has_initial_state[i]` is True, the kernel must use the
        slot's existing state as the prefill's initial state. This is the
        chunked-prefill / prefix-cache continuation path: a prior step
        wrote a recurrent/conv state to the slot, and the next prefill
        chunk for the same request must continue from that state, not
        from zero. Compares against running the equivalent single-shot
        prefill on the concatenated token stream from a zero state.
        """

        kq_head_dim = 128
        v_head_dim = 128
        n_kq = 2
        n_v = 8
        kernel_size = 4

        # One request, prefill split into two halves of 32 tokens each.
        # Step A: tokens [0, 32) starting from zero state.
        # Step B: tokens [32, 64) starting from Step A's final state.
        # Reference: tokens [0, 64) in a single shot from zero state.
        half = 32
        full = 64
        state_indices = jnp.array([1])
        num_blocks = 2

        rngs = iter(jax.random.split(jax.random.key(11), 12))
        query = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
        key = jax.random.normal(next(rngs), (full, n_kq * kq_head_dim))
        value = jax.random.normal(next(rngs), (full, n_v * v_head_dim))
        b = jax.random.normal(next(rngs), (full, n_v))
        a = jax.random.normal(next(rngs), (full, n_v))

        conv_dim = (n_kq * kq_head_dim) * 2 + n_v * v_head_dim
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        A_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        mixed_qkv_full = jnp.concatenate([query, key, value], axis=-1)
        mixed_qkv_a = mixed_qkv_full[:half]
        mixed_qkv_b = mixed_qkv_full[half:]

        conv_state_zero = jnp.zeros((num_blocks, kernel_size - 1, conv_dim))
        recurrent_state_zero = jnp.zeros((num_blocks, n_v, kq_head_dim, v_head_dim))

        run_jitted = jax.jit(
            wrapper.fused_conv1d_gdn,
            static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
        )
        common_static = dict(
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            n_kq=n_kq,
            n_v=n_v,
            d_k=kq_head_dim,
            d_v=v_head_dim,
            kernel_size=kernel_size,
        )

        # Single-shot reference (all 64 tokens, zero state, has_initial=False
        # encoded as seq_lens == query_lens == [full]).
        (_, _), output_ref = run_jitted(
            qkv=mixed_qkv_full,
            b=b,
            a=a,
            conv_state=conv_state_zero,
            recurrent_state=recurrent_state_zero,
            query_start_loc=jnp.array([0, full]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **common_static,
        )

        # Step A: first 32 tokens, zero state, has_initial=False.
        (conv_after_a, rec_after_a), output_a = run_jitted(
            qkv=mixed_qkv_a,
            b=b[:half],
            a=a[:half],
            conv_state=conv_state_zero,
            recurrent_state=recurrent_state_zero,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([half], dtype=jnp.int32),
            **common_static,
        )

        # Step B: next 32 tokens, slot now holds Step A's state.
        # seq_lens=[full] with query_lens=[half] gives context_len=half>0,
        # i.e., has_initial=True so the kernel continues from that state.
        (_, _), output_b = run_jitted(
            qkv=mixed_qkv_b,
            b=b[half:],
            a=a[half:],
            conv_state=conv_after_a,
            recurrent_state=rec_after_a,
            query_start_loc=jnp.array([0, half]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([full], dtype=jnp.int32),
            **common_static,
        )

        # Step A's output must match the first half of the single-shot
        # reference; Step B (continuation) must match the second half.
        np.testing.assert_allclose(output_a, output_ref[:half], rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(output_b, output_ref[half:], rtol=2e-2, atol=2e-2)

    def test_dynamic_tiling_with_mixed_and_decode_parity(self):
        """Verifies dynamic tiling matches static output and state."""

        d_k, d_v, kernel_size = 128, 128, 4
        query_lens = [1] * 32 + [128, 160]
        num_reqs = len(query_lens)
        total_tokens = sum(query_lens)
        num_blocks = num_reqs + 4

        qsl = jnp.array([0] + list(np.cumsum(query_lens)), dtype=jnp.int32)
        state_indices = jnp.arange(1, num_reqs + 1, dtype=jnp.int32)
        distribution = jnp.array([32, num_reqs, num_reqs], dtype=jnp.int32)
        seq_lens = jnp.array([64] * 32 + [128, 224], dtype=jnp.int32)

        for n_kq, n_v in [(2, 8), (8, 32), (16, 64)]:
            rngs = iter(jax.random.split(jax.random.key(99), 16))
            conv_dim = 2 * n_kq * d_k + n_v * d_v
            qkv = jax.random.normal(
                next(rngs), (total_tokens, conv_dim), dtype=jnp.bfloat16
            )
            b = jax.random.normal(next(rngs), (total_tokens, n_v), dtype=jnp.bfloat16)
            a = jax.random.normal(next(rngs), (total_tokens, n_v), dtype=jnp.bfloat16)
            conv_state = jax.random.normal(
                next(rngs), (num_blocks, kernel_size - 1, conv_dim), dtype=jnp.float32
            )
            recurrent_state = jax.random.normal(
                next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
            )
            conv_weight = jax.random.normal(
                next(rngs), (conv_dim, 1, kernel_size), dtype=jnp.bfloat16
            )
            conv_bias = jax.random.normal(next(rngs), (conv_dim,), dtype=jnp.bfloat16)
            a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
            dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)

            kwargs = dict(
                qkv=qkv,
                b=b,
                a=a,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                conv_weight=conv_weight,
                conv_bias=conv_bias,
                a_log=a_log,
                dt_bias=dt_bias,
                query_start_loc=qsl,
                state_indices=state_indices,
                distribution=distribution,
                seq_lens=seq_lens,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
            )
            with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False):
                fn_static = jax.jit(
                    wrapper.fused_conv1d_gdn.__wrapped__,
                    static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
                )
                (conv_static, rec_static), out_static = fn_static(**kwargs)

            with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True):
                fn_dyn = jax.jit(
                    wrapper.fused_conv1d_gdn.__wrapped__,
                    static_argnames=["n_kq", "n_v", "d_k", "d_v", "kernel_size"],
                )
                (conv_dyn, rec_dyn), out_dyn = fn_dyn(**kwargs)

            np.testing.assert_allclose(out_dyn, out_static, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(conv_dyn, conv_static, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(rec_dyn, rec_static, rtol=1e-2, atol=1e-2)

    def test_gdn_v3_dynamic_tiling_spec_verify_matches_static(self):
        """Dynamic tiling at num_spec_tokens=3 (W=4) matches flag-off tiling."""

        d_k, d_v, kernel_size = 128, 128, 4
        num_spec_tokens = 3
        window = num_spec_tokens + 1
        spec_lengths = [window] * 4
        num_seqs = len(spec_lengths)
        num_tokens = sum(spec_lengths)
        num_blocks = 1 + num_seqs * window
        qsl = jnp.array([0] + list(np.cumsum(spec_lengths)), dtype=jnp.int32)
        state_indices = jnp.array(
            [1 + i * window for i in range(num_seqs)], dtype=jnp.int32
        )
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.array([16 + sl for sl in spec_lengths], dtype=jnp.int32)
        read_offsets = jnp.zeros(num_seqs, dtype=jnp.int32)

        n_kq, n_v = 2, 8
        rngs = iter(jax.random.split(jax.random.key(123), 12))
        conv_dim = (n_kq * d_k) * 2 + n_v * d_v

        mixed_qkv = jax.random.normal(next(rngs), (num_tokens, conv_dim))
        b = jax.random.normal(next(rngs), (num_tokens, n_v))
        a = jax.random.normal(next(rngs), (num_tokens, n_v))
        conv_state = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, conv_dim)
        )
        recurrent_state = jax.random.normal(next(rngs), (num_blocks, n_v, d_k, d_v))
        conv_weight = jax.random.normal(next(rngs), (conv_dim, 1, kernel_size))
        conv_bias = jax.random.normal(next(rngs), (conv_dim,))
        a_log = jax.random.normal(next(rngs), (n_v,))
        dt_bias = jax.random.normal(next(rngs), (n_v,))

        kwargs = dict(
            qkv=mixed_qkv,
            b=b,
            a=a,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=qsl,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            read_offsets=read_offsets,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            num_spec_tokens=num_spec_tokens,
        )

        with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False):
            fn_static = jax.jit(
                wrapper.fused_conv1d_gdn.__wrapped__,
                static_argnames=[
                    "n_kq",
                    "n_v",
                    "d_k",
                    "d_v",
                    "kernel_size",
                    "num_spec_tokens",
                ],
            )
            (conv_static, rec_static), out_static = fn_static(**kwargs)

        with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True):
            fn_dyn = jax.jit(
                wrapper.fused_conv1d_gdn.__wrapped__,
                static_argnames=[
                    "n_kq",
                    "n_v",
                    "d_k",
                    "d_v",
                    "kernel_size",
                    "num_spec_tokens",
                ],
            )
            (conv_dyn, rec_dyn), out_dyn = fn_dyn(**kwargs)

        np.testing.assert_allclose(out_dyn, out_static, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(conv_dyn, conv_static, rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(rec_dyn, rec_static, rtol=1e-2, atol=1e-2)

    def _run_batched_decode_initial_state_parity(self, state_dtype: jnp.dtype):
        """Multi-tile batched decode preserves per-sequence initial states."""
        n_kq, n_v, d_k, d_v, kernel_size = 16, 64, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        num_seqs = 256
        decode_tile, _ = tiling.get_tile_sizes(
            batch_size=num_seqs,
            num_seqs=num_seqs,
            padded_batch_size=num_seqs,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=state_dtype,
            num_lanes=pltpu.get_tpu_info().num_lanes,
            decode_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(1),
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
        )
        steps = num_seqs // decode_tile
        self.assertGreaterEqual(steps, 4)

        block_ids = jnp.arange(1, num_seqs + 1, dtype=jnp.float32)
        init_conv = jnp.broadcast_to(
            block_ids[:, None, None] * jnp.float32(1e-2),
            (num_seqs, kernel_size - 1, dim),
        ).astype(jnp.bfloat16)
        init_rec = jnp.broadcast_to(
            block_ids[:, None, None, None] * jnp.float32(1e-2),
            (num_seqs, n_v, d_k, d_v),
        ).astype(state_dtype)

        qkv = jnp.zeros((num_seqs, dim), dtype=jnp.bfloat16)
        b = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
        a = jnp.full((num_seqs, n_v), -20.0, dtype=jnp.bfloat16)
        conv_weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.bfloat16)
        conv_bias = jnp.zeros((dim,), dtype=jnp.bfloat16)
        a_log = jnp.full((n_v,), -20.0, dtype=jnp.float32)
        dt_bias = jnp.full((n_v,), -20.0, dtype=jnp.float32)

        q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.full((num_seqs,), 11, dtype=jnp.int32)

        (conv_ref, rec_ref), out_ref = gdn_attention_ref(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=init_conv,
            recurrent_state=init_rec,
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
        )

        for rep in range(3):
            (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                qkv=qkv,
                b=b,
                a=a,
                conv_state=init_conv.copy(),
                recurrent_state=init_rec.copy(),
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
                decode_tile_size=decode_tile,
            )
            rec_np = np.asarray(rec_out, dtype=np.float32)
            ref_np = np.asarray(rec_ref, dtype=np.float32)
            init_np = np.asarray(init_rec, dtype=np.float32)
            bad_blocks = np.where(
                np.max(np.abs(rec_np - ref_np), axis=(1, 2, 3)) > 1e-5
            )[0]
            if bad_blocks.size > 0:
                blk = int(bad_blocks[0])
                src_blk = blk + 2 * decode_tile
                corrupted_mask = np.abs(rec_np[blk] - ref_np[blk]) > 1e-5
                hit = (
                    float(
                        np.mean(
                            rec_np[blk][corrupted_mask]
                            == init_np[src_blk][corrupted_mask]
                        )
                    )
                    if src_blk < num_seqs and np.any(corrupted_mask)
                    else 0.0
                )
                self.fail(
                    f"rep={rep} state_dtype={state_dtype} tile={decode_tile}: "
                    f"{bad_blocks.size}/{num_seqs} blocks corrupted "
                    f"(blocks={bad_blocks.tolist()}); block {blk} matches "
                    f"init_rec[{src_blk}] on {hit:.1%} of corrupted elements "
                    f"({int(np.sum(corrupted_mask))} elements)."
                )
            np.testing.assert_allclose(conv_out, conv_ref, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(out, out_ref, rtol=1e-5, atol=1e-5)

    # The flag-off clamp would shrink `decode_tile` below the tiling pick.
    @mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
    def test_batched_decode_initial_state_parity_fp32(self):
        self._run_batched_decode_initial_state_parity(jnp.dtype(jnp.float32))

    @mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
    def test_batched_decode_initial_state_parity_bf16(self):
        self._run_batched_decode_initial_state_parity(jnp.dtype(jnp.bfloat16))

    def test_spec_window_checkpoints_multi_tile(self):
        """Multi-tile spec verify (W=4) matches reference across repeats."""
        n_kq, n_v, d_k, d_v, kernel_size = 4, 8, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        window = 4
        num_seqs = 12
        batch_size = num_seqs * window
        num_blocks = num_seqs * window + 8

        decode_tile, _ = tiling.get_tile_sizes(
            batch_size=batch_size,
            num_seqs=num_seqs,
            padded_batch_size=batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim,
            act_out_dtype=jnp.dtype(jnp.bfloat16),
            recurrent_state_dtype=jnp.dtype(jnp.float32),
            num_lanes=pltpu.get_tpu_info().num_lanes,
            decode_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(window),
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window,
        )
        self.assertGreaterEqual(decode_tile, 2)
        self.assertGreaterEqual((num_seqs + decode_tile - 1) // decode_tile, 3)

        rngs = iter(jax.random.split(jax.random.key(314), 10))
        qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        conv_w = jax.random.normal(
            next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
        )
        conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
        a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        init_conv = jax.random.normal(
            next(rngs), (num_blocks, kernel_size - 1, dim), dtype=jnp.bfloat16
        )
        init_rec = jax.random.normal(
            next(rngs), (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
        )

        q_loc = jnp.arange(0, batch_size + 1, window, dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window
        read_offsets = jnp.array([s % window for s in range(num_seqs)], dtype=jnp.int32)
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.full((num_seqs,), window + 10, dtype=jnp.int32)

        (conv_ref, rec_ref), out_ref = gdn_attention_spec_ref(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=init_conv,
            recurrent_state=init_rec,
            conv_weight=conv_w,
            conv_bias=conv_b,
            a_log=a_log,
            dt_bias=dt_bias,
            query_start_loc=q_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            num_spec_seqs=num_seqs,
            seq_lens=seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )

        for _ in range(3):
            (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                qkv=qkv,
                b=b,
                a=a,
                conv_state=init_conv.copy(),
                recurrent_state=init_rec.copy(),
                conv_weight=conv_w,
                conv_bias=conv_b,
                a_log=a_log,
                dt_bias=dt_bias,
                query_start_loc=q_loc,
                state_indices=state_indices,
                distribution=distribution,
                seq_lens=seq_lens,
                read_offsets=read_offsets,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                num_spec_tokens=window - 1,
            )
            np.testing.assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
            for s in range(num_seqs):
                base = int(state_indices[s])
                np.testing.assert_allclose(
                    conv_out[base : base + window],
                    conv_ref[base : base + window],
                    rtol=1e-2,
                    atol=1e-2,
                )
                np.testing.assert_allclose(
                    rec_out[base : base + window],
                    rec_ref[base : base + window],
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_per_seq_boundary_read_after_write(self):
        """PER_SEQ multi-tile sequences preserve initial state and final writeback."""
        n_kq, n_v, d_k, d_v, kernel_size = 4, 8, 128, 128, 4
        dim = 2 * n_kq * d_k + n_v * d_v
        # One-tile sequences between two-tile ones put a store at tile p and a
        # load at tile p + 2 on the same slot of the old shared ring.
        lengths = [64, 32, 64, 32, 64, 32]
        num_seqs = len(lengths)
        batch_size = sum(lengths)

        rngs = iter(jax.random.split(jax.random.key(2718), 10))
        qkv = jax.random.normal(next(rngs), (batch_size, dim), dtype=jnp.bfloat16)
        b = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        a = jax.random.normal(next(rngs), (batch_size, n_v), dtype=jnp.bfloat16)
        conv_w = jax.random.normal(
            next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16
        )
        conv_b = jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16)
        a_log = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        dt_bias = jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32)
        init_conv = jax.random.normal(
            next(rngs), (num_seqs, kernel_size - 1, dim), dtype=jnp.bfloat16
        )
        init_rec = jax.random.normal(
            next(rngs), (num_seqs, n_v, d_k, d_v), dtype=jnp.float32
        )

        q_loc = jnp.array([0, *np.cumsum(lengths)], dtype=jnp.int32)
        state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
        distribution = jnp.array([0, num_seqs, num_seqs], dtype=jnp.int32)
        seq_lens = jnp.array(lengths, dtype=jnp.int32) + 10

        (conv_ref, rec_ref), out_ref = gdn_attention_ref(
            qkv=qkv,
            b=b,
            a=a,
            conv_state=init_conv,
            recurrent_state=init_rec,
            conv_weight=conv_w,
            conv_bias=conv_b,
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
        )

        for _ in range(3):
            (conv_out, rec_out), out = wrapper.fused_conv1d_gdn(
                qkv=qkv,
                b=b,
                a=a,
                conv_state=init_conv.copy(),
                recurrent_state=init_rec.copy(),
                conv_weight=conv_w,
                conv_bias=conv_b,
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
                mixed_tile_size=32,
            )
            np.testing.assert_allclose(out, out_ref, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(conv_out, conv_ref, rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(rec_out, rec_ref, rtol=1e-2, atol=1e-2)

    def test_qwen3_5_397b_parallelism_execution(self):
        """Executes dynamic-tiled GDN across Qwen3.5-397B head splits and DCP=4."""
        d_k, d_v, kernel_size = 128, 128, 4
        rng = jax.random.key(42)

        for n_kq, n_v, num_seqs, num_spec_tokens, prefill_len, num_decodes in [
            (2, 8, 16, 3, 16, 0),  # TP8 DCP=4 c=16 MTP=3: 16-tok prefill in B=16
            (8, 32, 16, 0, 12, 4),  # DP4TP2: 4 decodes + 12-tok prefill in B=16
            (16, 64, 16, 0, 12, 4),  # DP8: 4 decodes + 12-tok prefill in B=16
        ]:
            with self.subTest(n_kq=n_kq, n_v=n_v, num_spec_tokens=num_spec_tokens):
                dim = 2 * n_kq * d_k + n_v * d_v
                window = num_spec_tokens + 1
                batch_size = num_decodes + prefill_len
                num_blocks = num_seqs * window + 4
                k1, k2, k3, k4, k5, k6, k7, k8, k9, rng = jax.random.split(rng, 10)
                init_conv = jax.random.normal(
                    k8, (num_blocks, kernel_size - 1, dim), dtype=jnp.bfloat16
                )
                init_rec = jax.random.normal(
                    k9, (num_blocks, n_v, d_k, d_v), dtype=jnp.float32
                )
                active_seqs = num_decodes + 1
                state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window
                common_kwargs = dict(
                    qkv=jax.random.normal(k1, (batch_size, dim), dtype=jnp.bfloat16),
                    b=jax.random.normal(k2, (batch_size, n_v), dtype=jnp.bfloat16),
                    a=jax.random.normal(k3, (batch_size, n_v), dtype=jnp.bfloat16),
                    conv_weight=jax.random.normal(
                        k4, (dim, 1, kernel_size), dtype=jnp.bfloat16
                    ),
                    conv_bias=jax.random.normal(k5, (dim,), dtype=jnp.bfloat16),
                    a_log=jax.random.normal(k6, (n_v,), dtype=jnp.float32),
                    dt_bias=jax.random.normal(k7, (n_v,), dtype=jnp.float32),
                    query_start_loc=jnp.array(
                        list(range(num_decodes + 1))
                        + [batch_size] * (num_seqs - num_decodes),
                        dtype=jnp.int32,
                    ),
                    state_indices=state_indices,
                    distribution=jnp.array(
                        [num_decodes, num_decodes, active_seqs], dtype=jnp.int32
                    ),
                    seq_lens=jnp.array(
                        [10] * num_decodes
                        + [prefill_len + 10]
                        + [0] * (num_seqs - active_seqs),
                        dtype=jnp.int32,
                    ),
                    n_kq=n_kq,
                    n_v=n_v,
                    d_k=d_k,
                    d_v=d_v,
                    kernel_size=kernel_size,
                )
                (conv_ref, rec_ref), out_ref = gdn_attention_ref(
                    conv_state=init_conv,
                    recurrent_state=init_rec,
                    **common_kwargs,
                )
                with mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True):
                    # Fresh jit: the shared jitted wrapper caches by static
                    # args and shapes, not by the flag it reads at trace time.
                    fn_dyn = jax.jit(
                        wrapper.fused_conv1d_gdn.__wrapped__,
                        static_argnames=[
                            "n_kq",
                            "n_v",
                            "d_k",
                            "d_v",
                            "kernel_size",
                            "num_spec_tokens",
                        ],
                    )
                    (conv_out, rec_out), out = fn_dyn(
                        conv_state=init_conv.copy(),
                        recurrent_state=init_rec.copy(),
                        read_offsets=(
                            jnp.zeros((num_seqs,), dtype=jnp.int32)
                            if num_spec_tokens > 0
                            else None
                        ),
                        num_spec_tokens=num_spec_tokens,
                        **common_kwargs,
                    )
                np.testing.assert_allclose(out, out_ref, rtol=2e-2, atol=2e-2)
                for s in range(active_seqs):
                    slot = int(state_indices[s])
                    np.testing.assert_allclose(
                        conv_out[slot], conv_ref[slot], rtol=2e-2, atol=2e-2
                    )
                    np.testing.assert_allclose(
                        rec_out[slot], rec_ref[slot], rtol=3e-2, atol=3e-2
                    )
