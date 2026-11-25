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
"""CPU unit tests for head-grouped BF16 and FP8 block-doubling GDN prefill."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.gdn.v3 import (
    block_doubling,
    compute_gdn,
    config,
    vmem_ldst,
)


def _rel_l2(ref_x: jax.Array, act_x: jax.Array) -> float:
    r = np.asarray(ref_x, dtype=np.float32)
    y = np.asarray(act_x, dtype=np.float32)
    return float(np.linalg.norm(r - y) / (np.linalg.norm(r) + 1e-12))


def _make_cfg(
    *,
    n_kq: int,
    n_v: int,
    d: int = 128,
    chunk: int = 64,
    mode: config.GDNPrefillMode = config.GDNPrefillMode.BLOCK_DOUBLING_BF16,
) -> config.GDNConfig:
    f32 = jnp.dtype(jnp.float32)
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(f32, f32, f32, f32, f32),
        batch_size=chunk,
        dim_size=(2 * n_kq + n_v) * d,
        kernel_size=4,
        tile_size=chunk,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d,
        v_head_dim=d,
        gdn_prefill_mode=mode,
    )


class BlockDoublingComputeTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="ratio_1_chunk_64", n_kq=4, n_v=4),
        dict(testcase_name="ratio_2_chunk_64", n_kq=2, n_v=4),
        dict(testcase_name="ratio_3_unpaired_chunk_64", n_kq=2, n_v=6),
        dict(testcase_name="chunk_32_partial", n_kq=2, n_v=8, chunk=32, valid_len=25),
        dict(
            testcase_name="ratio_8_chunk_128_partial",
            n_kq=2,
            n_v=16,
            chunk=128,
            valid_len=95,
        ),
        dict(
            testcase_name="ratio_8_chunk_128_strong_decay",
            n_kq=2,
            n_v=16,
            chunk=128,
            gate_scale=30.0,
        ),
        dict(
            testcase_name="ratio_4_outlier_state_1000x",
            n_kq=2,
            n_v=8,
            state_scale=1000.0,
        ),
        dict(
            testcase_name="chunk_256_multi_block",
            n_kq=1,
            n_v=4,
            chunk=256,
            valid_len=210,
        ),
        dict(testcase_name="multi_group_64_heads", n_kq=8, n_v=64),
    )
    def test_chunked_per_seq_matches_sequential(
        self,
        n_kq,
        n_v,
        chunk=64,
        gate_scale=0.1,
        valid_len=None,
        state_scale=1.0,
    ):
        d = 128
        if valid_len is None:
            valid_len = chunk
        k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(41), 6)
        q = compute_gdn.l2_norm(jax.random.normal(k0, (n_kq, chunk, d))) * (d**-0.5)
        k = compute_gdn.l2_norm(jax.random.normal(k1, (n_kq, chunk, d)))
        v = jax.random.normal(k2, (n_v, chunk, d))
        gating_log = -jnp.abs(jax.random.normal(k3, (1, chunk, n_v))) * gate_scale
        beta = jax.nn.sigmoid(jax.random.normal(k4, (1, chunk, n_v)))
        if valid_len < chunk:
            valid_mask = (jnp.arange(chunk) < valid_len)[None, :, None]
            gating_log = jnp.where(valid_mask, gating_log, 0.0)
            beta = jnp.where(valid_mask, beta, 0.0)
        state_prev = jax.random.normal(k5, (n_v, d, d)) * jnp.float32(state_scale)

        with jax.default_matmul_precision("float32"):
            ref_out, ref_state = compute_gdn.chunked_gdn_per_seq(
                q,
                k,
                v,
                gating_log,
                beta,
                state_prev,
                _make_cfg(
                    n_kq=n_kq,
                    n_v=n_v,
                    chunk=chunk,
                    mode=config.GDNPrefillMode.SEQUENTIAL,
                ),
            )
            for mode, max_rel in (
                (config.GDNPrefillMode.BLOCK_DOUBLING_BF16, 1.5e-2),
                (config.GDNPrefillMode.BLOCK_DOUBLING_FP8, 10.0 ** (-30.0 / 20.0)),
            ):
                out, state = compute_gdn.chunked_gdn_per_seq(
                    q,
                    k,
                    v,
                    gating_log,
                    beta,
                    state_prev,
                    _make_cfg(n_kq=n_kq, n_v=n_v, chunk=chunk, mode=mode),
                )
                self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(state))))
                out_err = _rel_l2(ref_out, out)
                state_err = _rel_l2(ref_state, state)
                # Confirm block doubling ran rather than falling back to sequential.
                self.assertGreater(out_err, 0.0)
                self.assertLessEqual(out_err, max_rel)
                self.assertLessEqual(state_err, max_rel)

    @parameterized.named_parameters(
        dict(testcase_name="streamed_16_heads_bf16_out", n_kq=2, n_v=16, out_bytes=2),
        dict(testcase_name="streamed_16_heads_f32_out", n_kq=2, n_v=16, out_bytes=4),
        dict(testcase_name="unstreamed_6_heads_bf16_out", n_kq=2, n_v=6, out_bytes=2),
    )
    def test_run_chunk_streamed_matches_in_memory(self, n_kq, n_v, out_bytes):
        chunk, d, num_lanes = 64, 128, 128
        cfg = _make_cfg(
            n_kq=n_kq,
            n_v=n_v,
            chunk=chunk,
            mode=config.GDNPrefillMode.BLOCK_DOUBLING_BF16,
        )
        rng = np.random.default_rng(19)
        dim = (2 * n_kq + n_v) * d
        qkv_compact = jnp.asarray(
            rng.standard_normal((chunk, 1, dim)), dtype=jnp.float32
        )
        # Pad with one NaN spacer column to verify strided load_qkv_head skips it.
        qkv_padded = jnp.concat(
            [qkv_compact, jnp.full((chunk, 1, num_lanes), jnp.nan, dtype=jnp.float32)],
            axis=-1,
        )
        b_lanes = jnp.asarray(rng.standard_normal((1, chunk, num_lanes)), jnp.float32)
        a_lanes = jnp.asarray(rng.standard_normal((1, chunk, num_lanes)), jnp.float32)
        a_log = jnp.asarray(rng.uniform(-1.0, 1.0, (num_lanes,)), jnp.float32)
        dt_bias = jnp.asarray(rng.uniform(-2.0, -0.5, (num_lanes,)), jnp.float32)
        state_init = jnp.asarray(rng.standard_normal((n_v, d, d)), jnp.float32)
        real_sizes = jnp.array([53], dtype=jnp.int32)

        def _load(kind: vmem_ldst.QKVKind, head: int) -> jax.Array:
            return vmem_ldst.load_qkv_head(
                cast(jax.Ref, qkv_padded), kind, head, cfg, num_lanes
            )

        def load_raw_kq(kqs: list[int]):
            return (
                {kq: _load(vmem_ldst.QKVKind.Q, kq) for kq in kqs},
                {kq: _load(vmem_ldst.QKVKind.K, kq) for kq in kqs},
            )

        def load_raw_v(vs: list[int]):
            return {v: _load(vmem_ldst.QKVKind.V, v) for v in vs}

        stored_states: dict[int, jax.Array] = {}
        streamed_outs: dict[int, jax.Array] = {}
        out_step = block_doubling.out_stream_step(cfg, out_bytes)
        out_dtype = (
            jnp.dtype(jnp.bfloat16) if out_bytes == 2 else jnp.dtype(jnp.float32)
        )

        def store_out_slice(v0: int, heads: list[jax.Array]) -> None:
            for offset, head_out in enumerate(heads):
                streamed_outs[v0 + offset] = head_out

        out_ret, state_ret = block_doubling.run_chunk_streamed(
            load_raw_kq=load_raw_kq,
            load_raw_v=load_raw_v,
            cfg=cfg,
            load_state_v=lambda v: state_init[v],
            store_state_v=stored_states.__setitem__,
            real_sizes=real_sizes,
            b_lanes=b_lanes,
            a_lanes=a_lanes,
            a_log=a_log,
            dt_bias=dt_bias,
            store_out_slice=store_out_slice if out_step else None,
            out_step=out_step,
            out_dtype=out_dtype if out_step else None,
        )
        self.assertIsNone(state_ret)
        got_state = jnp.stack([stored_states[v] for v in range(n_v)], axis=0)
        if out_step:
            self.assertIsNone(out_ret)
            got_out = jnp.stack([streamed_outs[v] for v in range(n_v)], axis=0)
        else:
            self.assertIsNotNone(out_ret)
            assert out_ret is not None
            got_out = out_ret.astype(out_dtype)

        # Reference via in-memory run_chunk_per_seq with identical masked inputs.
        mask = (jnp.arange(chunk) < real_sizes[0])[:, None]
        q_raw = qkv_compact[:, 0, : n_kq * d].reshape(chunk, n_kq, d).swapaxes(0, 1)
        k_raw = (
            qkv_compact[:, 0, n_kq * d : 2 * n_kq * d]
            .reshape(chunk, n_kq, d)
            .swapaxes(0, 1)
        )
        v_raw = qkv_compact[:, 0, 2 * n_kq * d :].reshape(chunk, n_v, d).swapaxes(0, 1)
        q_ref = compute_gdn.l2_norm(jnp.where(mask[None], q_raw, 0.0)) * (d**-0.5)
        k_ref = compute_gdn.l2_norm(jnp.where(mask[None], k_raw, 0.0))
        v_ref = jnp.where(mask[None], v_raw, 0.0)
        beta = jnp.where(mask[None], jax.nn.sigmoid(b_lanes), 0.0)
        gating_log = jnp.where(
            mask[None],
            -jnp.exp(a_log.reshape(1, 1, -1))
            * jax.nn.softplus(a_lanes + dt_bias.reshape(1, 1, -1)),
            0.0,
        )
        ref_out, ref_state = block_doubling.run_chunk_per_seq(
            q_ref, k_ref, v_ref, gating_log, beta, state_init, cfg
        )
        assert ref_out is not None and ref_state is not None
        self.assertLessEqual(_rel_l2(ref_out.astype(out_dtype), got_out), 1e-4)
        self.assertLessEqual(_rel_l2(ref_state, got_state), 1e-4)

    def test_error_paths(self):
        cfg_seq = _make_cfg(
            n_kq=2, n_v=8, chunk=64, mode=config.GDNPrefillMode.SEQUENTIAL
        )
        z = jnp.zeros((1, 64, 128), jnp.float32)
        with self.assertRaisesRegex(ValueError, "Expected active block-doubling mode"):
            block_doubling.run_chunk_streamed(
                load_raw_kq=lambda kqs: ({}, {}),
                load_raw_v=lambda vs: {},
                cfg=cfg_seq,
                load_state_v=lambda v: jnp.zeros((128, 128), jnp.float32),
                store_state_v=lambda v, s: None,
                real_sizes=jnp.array([64], jnp.int32),
                b_lanes=z,
                a_lanes=z,
                a_log=jnp.zeros((128,), jnp.float32),
                dt_bias=jnp.zeros((128,), jnp.float32),
            )

        cfg_bd = _make_cfg(
            n_kq=2, n_v=8, chunk=64, mode=config.GDNPrefillMode.BLOCK_DOUBLING_BF16
        )
        for bad_step in (0, 6):
            with self.assertRaisesRegex(ValueError, "out_step"):
                block_doubling.BlockDoublingChunk(
                    cfg_bd,
                    z,
                    jnp.ones_like(z),
                    store_out_slice=lambda v0, heads: None,
                    out_step=bad_step,
                )

        with self.assertRaisesRegex(ValueError, "Expected active block-doubling mode"):
            block_doubling.run_chunk_per_seq(
                None, None, None, z, jnp.ones_like(z), None, cfg_seq
            )
        with self.assertRaisesRegex(ValueError, "state_prev or load_state_v"):
            block_doubling.run_chunk_per_seq(
                None, None, None, z, jnp.ones_like(z), None, cfg_bd
            )

        with self.assertRaisesRegex(ValueError, "QKVKind"):
            vmem_ldst.load_qkv_head(
                cast(jax.Ref, jnp.zeros((64, 1, 12 * 128), jnp.float32)),
                "z",
                0,
                cfg_bd,
                128,
            )
        cfg_unaligned = _make_cfg(
            n_kq=2,
            n_v=8,
            d=64,
            chunk=64,
            mode=config.GDNPrefillMode.BLOCK_DOUBLING_BF16,
        )
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            vmem_ldst.load_qkv_head(
                cast(jax.Ref, jnp.zeros((64, 1, 12 * 64), jnp.float32)),
                vmem_ldst.QKVKind.Q,
                0,
                cfg_unaligned,
                128,
            )


if __name__ == "__main__":
    absltest.main()
