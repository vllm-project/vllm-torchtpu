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


class BlockDoublingPhaseHooksTest(parameterized.TestCase):
    @parameterized.parameters(
        config.GDNPrefillMode.BLOCK_DOUBLING_BF16,
        config.GDNPrefillMode.BLOCK_DOUBLING_FP8,
    )
    def test_streamed_prescaled_gates_and_phase_hook(self, mode):
        rng = np.random.default_rng(17)
        n_kq, n_v, chunk, d = 2, 8, 32, 128
        cfg = _make_cfg(n_kq=n_kq, n_v=n_v, chunk=chunk, d=d, mode=mode)
        q = jnp.asarray(rng.standard_normal((n_kq, chunk, d)), jnp.float32)
        k = jnp.asarray(rng.standard_normal((n_kq, chunk, d)), jnp.float32)
        v = jnp.asarray(rng.standard_normal((n_v, chunk, d)), jnp.float32)
        state_prev = jnp.asarray(rng.standard_normal((n_v, d, d)), jnp.float32)
        b_lanes = jnp.asarray(rng.standard_normal((1, chunk, 128)), jnp.float32)
        a_lanes = jnp.asarray(rng.standard_normal((1, chunk, 128)), jnp.float32)
        a_log = jnp.asarray(rng.uniform(-2.0, 0.0, (128,)), jnp.float32)
        dt_bias = jnp.asarray(rng.normal(0.0, 0.5, (128,)), jnp.float32)

        # Reference run with raw Q/K, raw gate inputs, and explicit real_sizes.
        ref_states: dict[int, jax.Array] = {}
        ref_out, _ = block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: q[i] for i in kqs},
                {i: k[i] for i in kqs},
            ),
            load_raw_v=lambda vs: {i: v[i] for i in vs},
            cfg=cfg,
            load_state_v=lambda h: state_prev[h],
            store_state_v=ref_states.__setitem__,
            real_sizes=jnp.array([chunk], jnp.int32),
            b_lanes=b_lanes,
            a_lanes=a_lanes,
            a_log=a_log,
            dt_bias=dt_bias,
        )
        self.assertIsNotNone(ref_out)

        # Precomputed gates + prescaled Q/K + real_sizes=None + phase_hook +
        # streamed output slices.
        q_scale = d**-0.5
        inv_q = jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        inv_k = jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        q_pre = q * (inv_q * jnp.float32(q_scale * 256.0))
        k_pre = k * (inv_k * jnp.float32(256.0))
        beta_lanes = jax.nn.sigmoid(b_lanes)
        glog_lanes = -jnp.exp(a_log.reshape(1, 1, -1)) * jax.nn.softplus(
            a_lanes + dt_bias.reshape(1, 1, -1)
        )

        events: list[tuple[block_doubling.BlockDoublingPhase, int, int]] = []
        got_states: dict[int, jax.Array] = {}
        got_out_slices: dict[int, list[jax.Array]] = {}
        out_none, state_none = block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: q_pre[i] for i in kqs},
                {i: k_pre[i] for i in kqs},
            ),
            load_raw_v=lambda vs: {i: v[i] for i in vs},
            cfg=cfg,
            load_state_v=lambda h: state_prev[h],
            store_state_v=got_states.__setitem__,
            gating_log_lanes=glog_lanes,
            beta_lanes=beta_lanes,
            kq_prescaled_256=True,
            store_out_slice=got_out_slices.__setitem__,
            out_step=4,
            out_dtype=jnp.dtype(jnp.float32),
            phase_hook=lambda phase, idx, tot: events.append((phase, idx, tot)),
        )
        self.assertIsNone(out_none)
        self.assertIsNone(state_none)
        self.assertEqual(sorted(got_out_slices), [0, 4])
        got_out = jnp.stack(got_out_slices[0] + got_out_slices[4], axis=0)
        np.testing.assert_allclose(
            np.asarray(got_out), np.asarray(ref_out), rtol=1e-5, atol=1e-5
        )
        for h in range(n_v):
            np.testing.assert_allclose(
                np.asarray(got_states[h]),
                np.asarray(ref_states[h]),
                rtol=1e-5,
                atol=1e-5,
            )
        phase = block_doubling.BlockDoublingPhase
        # 1 group of 8 heads (4 paired V heads in front, 8 heads in tail).
        want_events = (
            [(phase.GROUP_START, 0, 1)]
            + [(phase.STATE_FIRST_PAIR, i, 4) for i in range(4)]
            + [(phase.TAIL_HEAD, i, 8) for i in range(8)]
            + [(phase.GROUP_END, 0, 1)]
        )
        self.assertEqual(events, want_events)

    def test_phase_hook_multi_group_and_unpaired_tail_head(self):
        phase = block_doubling.BlockDoublingPhase
        d, chunk = 128, 32

        # 1) Multi-group (n_kq=8, n_v=64 -> 2 groups of 32 heads, 16 pairs/group).
        cfg_multi = _make_cfg(n_kq=8, n_v=64, chunk=chunk, d=d)
        events_multi: list[tuple[block_doubling.BlockDoublingPhase, int, int]] = []
        zeros_1d = jnp.zeros((chunk, d), jnp.float32)
        zeros_s = jnp.zeros((d, d), jnp.float32)
        zeros_lanes = jnp.zeros((1, chunk, 128), jnp.float32)
        block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: zeros_1d for i in kqs},
                {i: zeros_1d for i in kqs},
            ),
            load_raw_v=lambda vs: {i: zeros_1d for i in vs},
            cfg=cfg_multi,
            load_state_v=lambda h: zeros_s,
            store_state_v=lambda h, s: None,
            gating_log_lanes=zeros_lanes,
            beta_lanes=zeros_lanes,
            kq_prescaled_256=True,
            phase_hook=lambda p, idx, tot: events_multi.append((p, idx, tot)),
        )
        want_multi = []
        for g_idx in range(2):
            want_multi.append((phase.GROUP_START, g_idx, 2))
            want_multi.extend([(phase.STATE_FIRST_PAIR, i, 16) for i in range(16)])
            want_multi.extend([(phase.TAIL_HEAD, i, 32) for i in range(32)])
            want_multi.append((phase.GROUP_END, g_idx, 2))
        self.assertEqual(events_multi, want_multi)

        # 2) Odd v_per_kq (n_kq=2, n_v=6 -> v_per_kq=3 disables V-head pairing,
        #    so front visits 6 singleton heads and tail visits 6 heads).
        cfg_odd = _make_cfg(n_kq=2, n_v=6, chunk=chunk, d=d)
        events_odd: list[tuple[block_doubling.BlockDoublingPhase, int, int]] = []
        block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: zeros_1d for i in kqs},
                {i: zeros_1d for i in kqs},
            ),
            load_raw_v=lambda vs: {i: zeros_1d for i in vs},
            cfg=cfg_odd,
            load_state_v=lambda h: zeros_s,
            store_state_v=lambda h, s: None,
            gating_log_lanes=zeros_lanes,
            beta_lanes=zeros_lanes,
            kq_prescaled_256=True,
            phase_hook=lambda p, idx, tot: events_odd.append((p, idx, tot)),
        )
        want_odd = (
            [(phase.GROUP_START, 0, 1)]
            + [(phase.STATE_FIRST_PAIR, i, 6) for i in range(6)]
            + [(phase.TAIL_HEAD, i, 6) for i in range(6)]
            + [(phase.GROUP_END, 0, 1)]
        )
        self.assertEqual(events_odd, want_odd)

    @parameterized.parameters(False, True)
    def test_precomputed_gates_respect_real_sizes_mask(self, kq_prescaled_256):
        rng = np.random.default_rng(23)
        n_kq, n_v, chunk, d = 1, 4, 32, 128
        valid_len = 20
        cfg = _make_cfg(n_kq=n_kq, n_v=n_v, chunk=chunk, d=d)
        q_clean = jnp.asarray(rng.standard_normal((n_kq, chunk, d)), jnp.float32)
        k_clean = jnp.asarray(rng.standard_normal((n_kq, chunk, d)), jnp.float32)
        if kq_prescaled_256:
            q_scale = d**-0.5
            inv_q = jax.lax.rsqrt(jnp.sum(q_clean**2, axis=-1, keepdims=True) + 1e-6)
            inv_k = jax.lax.rsqrt(jnp.sum(k_clean**2, axis=-1, keepdims=True) + 1e-6)
            q_clean = q_clean * (inv_q * jnp.float32(q_scale * 256.0))
            k_clean = k_clean * (inv_k * jnp.float32(256.0))
        v_clean = jnp.asarray(rng.standard_normal((n_v, chunk, d)), jnp.float32)
        state_prev = jnp.asarray(rng.standard_normal((n_v, d, d)), jnp.float32)
        beta_clean = jax.nn.sigmoid(
            jnp.asarray(rng.standard_normal((1, chunk, 128)), jnp.float32)
        )
        glog_clean = -jnp.abs(
            jnp.asarray(rng.standard_normal((1, chunk, 128)), jnp.float32)
        )
        mask_2d = (jnp.arange(chunk) < valid_len)[None, :, None]
        # Poison padded rows with NaN so any missed masking leaks immediately.
        q_poison = jnp.where(mask_2d, q_clean, jnp.nan)
        k_poison = jnp.where(mask_2d, k_clean, jnp.nan)
        v_poison = jnp.where(mask_2d, v_clean, jnp.nan)
        glog_poison = jnp.where(mask_2d, glog_clean, jnp.nan)
        beta_poison = jnp.where(mask_2d, beta_clean, jnp.nan)

        states_masked: dict[int, jax.Array] = {}
        states_premasked: dict[int, jax.Array] = {}
        out_masked, _ = block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: q_poison[i] for i in kqs},
                {i: k_poison[i] for i in kqs},
            ),
            load_raw_v=lambda vs: {i: v_poison[i] for i in vs},
            cfg=cfg,
            load_state_v=lambda h: state_prev[h],
            store_state_v=states_masked.__setitem__,
            real_sizes=jnp.array([valid_len], jnp.int32),
            gating_log_lanes=glog_poison,
            beta_lanes=beta_poison,
            kq_prescaled_256=kq_prescaled_256,
        )
        out_premasked, _ = block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: jnp.where(mask_2d[0], q_clean[i], 0.0) for i in kqs},
                {i: jnp.where(mask_2d[0], k_clean[i], 0.0) for i in kqs},
            ),
            load_raw_v=lambda vs: {
                i: jnp.where(mask_2d[0], v_clean[i], 0.0) for i in vs
            },
            cfg=cfg,
            load_state_v=lambda h: state_prev[h],
            store_state_v=states_premasked.__setitem__,
            real_sizes=jnp.array([valid_len], jnp.int32),
            gating_log_lanes=jnp.where(mask_2d, glog_clean, 0.0),
            beta_lanes=jnp.where(mask_2d, beta_clean, 0.0),
            kq_prescaled_256=kq_prescaled_256,
        )
        self.assertIsNotNone(out_masked)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out_masked))))
        np.testing.assert_array_equal(np.asarray(out_masked), np.asarray(out_premasked))
        for h in range(n_v):
            np.testing.assert_array_equal(
                np.asarray(states_masked[h]), np.asarray(states_premasked[h])
            )

        # Must-fail control: omitting real_sizes lets poisoned padded rows leak NaN.
        states_unmasked: dict[int, jax.Array] = {}
        out_unmasked, _ = block_doubling.run_chunk_streamed(
            load_raw_kq=lambda kqs: (
                {i: q_poison[i] for i in kqs},
                {i: k_poison[i] for i in kqs},
            ),
            load_raw_v=lambda vs: {i: v_poison[i] for i in vs},
            cfg=cfg,
            load_state_v=lambda h: state_prev[h],
            store_state_v=states_unmasked.__setitem__,
            real_sizes=None,
            gating_log_lanes=glog_poison,
            beta_lanes=beta_poison,
            kq_prescaled_256=kq_prescaled_256,
        )
        self.assertIsNotNone(out_unmasked)
        self.assertTrue(bool(jnp.any(jnp.isnan(out_unmasked))))

    def test_phase_hook_validation_errors(self):
        cfg = _make_cfg(n_kq=2, n_v=8, chunk=32, d=128)
        zeros_lanes = jnp.zeros((1, 32, 128), jnp.float32)

        def _dummy_kq(kqs):
            del kqs
            return {}, {}

        def _dummy_v(vs):
            del vs
            return {}

        def _dummy_load(h):
            del h
            return jnp.zeros((128, 128), jnp.float32)

        def _dummy_store(h, s):
            del h, s

        with self.assertRaisesRegex(ValueError, "both be set or both be None"):
            block_doubling.run_chunk_streamed(
                load_raw_kq=_dummy_kq,
                load_raw_v=_dummy_v,
                cfg=cfg,
                load_state_v=_dummy_load,
                store_state_v=_dummy_store,
                gating_log_lanes=zeros_lanes,
            )
        with self.assertRaisesRegex(ValueError, "both be set or both be None"):
            block_doubling.run_chunk_streamed(
                load_raw_kq=_dummy_kq,
                load_raw_v=_dummy_v,
                cfg=cfg,
                load_state_v=_dummy_load,
                store_state_v=_dummy_store,
                beta_lanes=zeros_lanes,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined with"):
            block_doubling.run_chunk_streamed(
                load_raw_kq=_dummy_kq,
                load_raw_v=_dummy_v,
                cfg=cfg,
                load_state_v=_dummy_load,
                store_state_v=_dummy_store,
                gating_log_lanes=zeros_lanes,
                beta_lanes=zeros_lanes,
                b_lanes=zeros_lanes,
            )
        with self.assertRaisesRegex(ValueError, "b_lanes, a_lanes"):
            block_doubling.run_chunk_streamed(
                load_raw_kq=_dummy_kq,
                load_raw_v=_dummy_v,
                cfg=cfg,
                load_state_v=_dummy_load,
                store_state_v=_dummy_store,
            )


if __name__ == "__main__":
    absltest.main()
