"""Batching regression test for the GLM 5.3 Flash TPU KDA layer (runs on CPU).

Drives `TpuKimiGatedDeltaNetAttention._forward` with vllm_torchtpu-style padded
`AttentionMetadata` (no `*_cpu` fields, stale `seq_lens` tail, junk padding
rows, reused state slots) over several scheduler steps, and checks every
scheduled token against a per-request, per-token reference.

The two Pallas kernels are replaced by torch versions of the same recurrence
(the prefill one poisons masked rows with NaN), so this covers the layer's
gather / scatter / state-slot logic, not kernel numerics.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_decode_torch as kdt
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_vllm_decode as V
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_vllm_layer_tpu as L
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_vllm_prefill as P

pytestmark = pytest.mark.cpu_test

H, D, W = 2, 16, 4
SL = W - 1
PROJ = H * D
DIM = 3 * PROJ
LB = -5.0
NUM_SLOTS = 24
MAX_REQS = 16


def _l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _ref_step(state, q, k, v, g_raw, beta_sig, a_log, dt_bias, scale):
    """One KDA token. state [H, dv, dk] fp32; returns (state, out [H, dv])."""
    qn, kn = _l2n(q.float()) * scale, _l2n(k.float())
    g = g_raw.float() + dt_bias.float()
    decay = torch.exp(LB * torch.sigmoid(torch.exp(a_log.float())[:, None] * g))
    state = state * decay[:, None, :]
    pred = (kn[:, None, :] * state).sum(-1)
    bk = beta_sig.float()[:, None] * kn
    state = state + (v.float() - pred)[:, :, None] * bk[:, None, :]
    return state, (qn[:, None, :] * state).sum(-1)


def _fake_kda_decode(
    q,
    k,
    v,
    g_raw,
    beta,
    a_log,
    dt_bias,
    recurrent_state,
    state_indices,
    lower_bound,
    qk_scale,
    h_block=0,
    null_slot=-1,
):
    """Torch twin of the Pallas decode kernel, including its padding rule."""
    del h_block
    assert lower_bound == LB
    assert state_indices.dtype == torch.int32
    out = torch.empty(q.shape[0], H, D)
    for i in range(q.shape[0]):
        s = int(state_indices[i])
        if null_slot >= 0 and s == null_slot:
            out[i] = 0  # Padding: the slot is left alone.
            continue
        recurrent_state[s], out[i] = _ref_step(
            recurrent_state[s].clone(),
            q[i],
            k[i],
            v[i],
            g_raw[i],
            beta[i],
            a_log,
            dt_bias,
            qk_scale,
        )
    return out


def _fake_kda_prefill_chunk(
    q,
    k,
    v,
    g_raw,
    beta,
    a_log,
    dt_bias,
    recurrent_state,
    valid_lens,
    lower_bound,
    qk_scale,
    h_block=0,
):
    del h_block
    assert lower_bound == LB
    out = torch.full((*q.shape[:2], H, D), float("nan"))
    for s in range(q.shape[0]):
        st = recurrent_state[s].clone()
        for t in range(int(valid_lens[s])):
            st, out[s, t] = _ref_step(
                st,
                q[s, t],
                k[s, t],
                v[s, t],
                g_raw[s, t],
                beta[s, t],
                a_log,
                dt_bias,
                qk_scale,
            )
        recurrent_state[s] = st
    return out


class _Layer:
    """The attributes of `TpuKimiGatedDeltaNetAttention` that `_forward` uses."""

    _forward = L.TpuKimiGatedDeltaNetAttention._forward
    _forward_tpu_metadata = L.TpuKimiGatedDeltaNetAttention._forward_tpu_metadata
    _forward_gdn_metadata = L.TpuKimiGatedDeltaNetAttention._forward_gdn_metadata

    def __init__(self, dim_first):
        self.prefix = "model.layers.0.self_attn"
        self.local_projection_size = PROJ
        self.local_num_heads = H
        self.head_dim = D
        self.gate_lower_bound = LB
        self.conv1d = types.SimpleNamespace(
            weight=torch.randn(DIM, 1, W) * 0.5, bias=torch.randn(DIM) * 0.1
        )
        self.A_log = torch.randn(H) * 0.3
        self.dt_bias = torch.randn(H * D) * 0.3
        self.o_norm = L.FusedRMSNormGated(D, eps=1e-5, activation="sigmoid")
        with torch.no_grad():
            self.o_norm.weight.copy_(1 + 0.1 * torch.randn(D))
        conv_shape = (NUM_SLOTS, DIM, SL) if dim_first else (NUM_SLOTS, SL, DIM)
        # Garbage everywhere: a fresh request that reads its slot would fail.
        self.kv_cache = (torch.randn(conv_shape), torch.randn(NUM_SLOTS, H, D, D))


class _Request:
    def __init__(self, prompt_len, max_new=12):
        total = prompt_len + max_new
        self.prompt_len = prompt_len
        self.x = torch.randn(total, DIM)
        self.g1 = torch.randn(total, H, D)
        self.beta = torch.randn(total, H)
        self.g2 = torch.randn(total, H, D)
        self.computed = 0
        self.slot = None
        self.window = torch.zeros(SL, DIM)
        self.state = torch.zeros(H, D, D)

    def ref_outputs(self, layer, n):
        """Advances the reference by n tokens; returns normed outputs [n, H, D]."""
        w = layer.conv1d.weight.view(DIM, W)
        outs = []
        for t in range(self.computed, self.computed + n):
            window = torch.cat([self.window, self.x[t][None]], 0)
            conv = torch.nn.functional.silu((window * w.T).sum(0) + layer.conv1d.bias)
            self.window = window[1:]
            q, k, v = conv.split(PROJ)
            self.state, o = _ref_step(
                self.state,
                q.view(H, D),
                k.view(H, D),
                v.view(H, D),
                self.g1[t],
                torch.sigmoid(self.beta[t]),
                layer.A_log,
                layer.dt_bias.view(H, D),
                D**-0.5,
            )
            outs.append(o)
        o = torch.stack(outs)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)
        o = o * layer.o_norm.weight
        return o * torch.sigmoid(self.g2[self.computed : self.computed + n])


class TestGlm53KdaLayerBatching(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        for name, fake in (
            ("kda_decode", _fake_kda_decode),
            ("kda_prefill_chunk", _fake_kda_prefill_chunk),
        ):
            patcher = mock.patch.object(kdt, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(L, "_PLAN_LOG_BUDGET", [0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _step(self, layer, free_slots, sched, t_pad):
        """sched: [(request, num_tokens)] in batch order."""
        for req, _ in sched:
            if req.slot is None:
                req.slot = free_slots.pop()
        q_lens = [n for _, n in sched]
        n_tok = sum(q_lens)
        qsl = np.zeros(MAX_REQS + 1, np.int32)
        qsl[1 : len(sched) + 1] = np.cumsum(q_lens)
        qsl[len(sched) + 1 :] = qsl[len(sched)]
        seq_lens = self.rng.integers(0, 10**6, MAX_REQS).astype(np.int32)
        slots = np.zeros(MAX_REQS, np.int32)
        for i, (req, n) in enumerate(sched):
            seq_lens[i] = req.computed + n
            slots[i] = req.slot
        num_dec = 0
        while (
            num_dec < len(sched)
            and sched[num_dec][1] == 1
            and sched[num_dec][0].computed > 0
        ):
            num_dec += 1
        meta = types.SimpleNamespace(
            query_start_loc=torch.from_numpy(qsl),
            seq_lens=torch.from_numpy(seq_lens),
            request_distribution=torch.tensor(
                [num_dec, num_dec, len(sched)], dtype=torch.int32
            ),
            mamba_state_indices=torch.from_numpy(slots),
            block_tables=torch.zeros(MAX_REQS * 4, dtype=torch.int32),
        )
        ctx = types.SimpleNamespace(attn_metadata={layer.prefix: meta})

        # Scheduled tokens in order, then large junk in the padding rows.
        mixed = torch.randn(t_pad, DIM) * 50
        g1 = torch.randn(t_pad, H, D) * 50
        beta = torch.randn(t_pad, H) * 50
        g2 = torch.randn(t_pad, H, D)
        row = 0
        for req, n in sched:
            rows, toks = slice(row, row + n), slice(req.computed, req.computed + n)
            mixed[rows], g1[rows] = req.x[toks], req.g1[toks]
            beta[rows], g2[rows] = req.beta[toks], req.g2[toks]
            row += n
        # Like the base class: mixed_qkv is a split view, g1/beta get a batch dim.
        mixed_view = torch.cat([mixed, torch.zeros(t_pad, 5)], -1).split(
            [DIM, 5], dim=-1
        )[0]
        core = torch.full((1, t_pad, H, D), float("nan"))
        with mock.patch.object(L, "get_forward_context", lambda: ctx):
            layer._forward(mixed_view, g1.unsqueeze(0), g2, beta.unsqueeze(0), core)

        row = 0
        for req, n in sched:
            torch.testing.assert_close(
                core[0, row : row + n],
                req.ref_outputs(layer, n),
                rtol=0,
                atol=2e-4,
                msg=f"tokens {req.computed}..{req.computed + n}",
            )
            req.computed += n
            row += n
        self.assertTrue((core[0, n_tok:] == 0).all(), "padded rows not zero")

    def _scenario(self, dim_first):
        layer = _Layer(dim_first)
        free = list(range(NUM_SLOTS - 1, 0, -1))  # Slot 0 is the null slot.
        a, b, c, d = _Request(70), _Request(45), _Request(1), _Request(600)
        with mock.patch.object(L, "is_conv_state_dim_first", lambda: dim_first):
            self._step(layer, free, [(a, 70), (b, 5)], 128)  # Fresh prefills.
            self._step(layer, free, [(a, 1), (b, 40), (c, 1)], 64)  # Resumed.
            self._step(layer, free, [(a, 1), (b, 1), (c, 1), (d, 600)], 1024)
            self._step(layer, free, [(a, 1), (b, 1), (c, 1), (d, 1)], 8)
            # `a` finishes; `e` reuses its slot, which still holds a's state.
            free.append(a.slot)
            e = _Request(3)
            self._step(layer, free, [(b, 1), (c, 1), (d, 1), (e, 3)], 16)
            self.assertEqual(e.slot, a.slot)
            # Not decode-first: a resumed 1-token request after a prefill.
            f = _Request(20)
            self._step(layer, free, [(b, 1), (f, 20), (c, 1), (e, 1)], 32)
            # 11 requests (bucket 16); ragged prefills 100 + 28 sum to 128.
            g, h = _Request(100), _Request(28)
            others = [_Request(int(self.rng.integers(2, 9))) for _ in range(4)]
            sched = [(r, 1) for r in (b, c, d, e, f)] + [(g, 100), (h, 28)]
            self._step(layer, free, sched + [(r, r.prompt_len) for r in others], 256)
            for _ in range(2):
                self._step(
                    layer, free, [(r, 1) for r in [b, c, d, e, f, g, h] + others], 16
                )

    def test_batching_conv_state_seq_first(self):
        self._scenario(dim_first=False)

    def test_batching_conv_state_dim_first(self):
        self._scenario(dim_first=True)

    def test_ragged_prefill_whose_total_divides_the_chunk(self):
        # 100 + 28 = 128 tokens used to take the equal-length fast path.
        bounds = [0, 100, 128]
        q, k, v, g = (torch.randn(1, 128, H, D) for _ in range(4))
        b = torch.randn(1, 128, H)
        a_log, dt_bias = torch.randn(H) * 0.3, torch.randn(H * D) * 0.3
        init = torch.randn(2, H, D, D)
        out, final = P.chunk_kda_with_fused_gate(
            q,
            k,
            v,
            g,
            b,
            a_log,
            dt_bias,
            LB,
            init.clone(),
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
        )
        for s in range(2):
            st = init[s].clone()
            for t in range(bounds[s], bounds[s + 1]):
                st, o = _ref_step(
                    st,
                    q[0, t],
                    k[0, t],
                    v[0, t],
                    g[0, t],
                    torch.sigmoid(b[0, t]),
                    a_log,
                    dt_bias.view(H, D),
                    D**-0.5,
                )
                torch.testing.assert_close(out[0, t], o, rtol=0, atol=1e-4)
            torch.testing.assert_close(final[s], st, rtol=0, atol=1e-4)

    def test_single_sequence_prefill_uses_s_pad_1_without_scatter(self):
        meta = types.SimpleNamespace(
            query_start_loc=torch.tensor([0, 600, 600, 600], dtype=torch.int32),
            seq_lens=torch.tensor([600, 0, 0], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
            mamba_state_indices=torch.tensor([3, 0, 0], dtype=torch.int32),
            block_tables=torch.zeros(12, dtype=torch.int32),
        )
        plan = L._build_tpu_step_plan(
            meta, padded_tokens=1024, state_len=SL, device=torch.device("cpu")
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.num_prefills, 1)
        self.assertEqual(plan.prefill_slots.numel(), 1)
        self.assertEqual(plan.prefill_chunk_len, 256)
        self.assertEqual(len(plan.prefill_chunk_rows), 3)
        for rows in plan.prefill_chunk_rows:
            self.assertEqual(rows.numel(), 256)

        layer = _Layer(dim_first=False)
        free = [5]
        req = _Request(600)
        with mock.patch.object(L, "is_conv_state_dim_first", lambda: False):
            self._step(layer, free, [(req, 600)], 1024)

    def _spy_decode_calls(self):
        """Records `(state, slots, null_slot)` for every decode kernel call."""
        calls = []

        def spy(*args, **kwargs):
            calls.append((args[7], args[8].clone(), kwargs.get("null_slot", -1)))
            return _fake_kda_decode(*args, **kwargs)

        patcher = mock.patch.object(kdt, "kda_decode", spy)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def test_pure_decode_fast_path_updates_cache_in_place(self):
        # 5 decodes -> plan bucket 8, in a 16-row step: 3 null-slot entries
        # plus 8 rows past the plan, all of which must come out zero.
        layer = _Layer(dim_first=False)
        free = list(range(NUM_SLOTS - 1, 0, -1))
        reqs = [_Request(int(self.rng.integers(2, 9))) for _ in range(5)]
        with mock.patch.object(L, "is_conv_state_dim_first", lambda: False):
            self._step(layer, free, [(r, r.prompt_len) for r in reqs], 64)
            with torch.no_grad():
                layer.kv_cache[1][0].copy_(torch.randn(H, D, D))
            null_before = layer.kv_cache[1][0].clone()
            calls = self._spy_decode_calls()
            for _ in range(3):
                self._step(layer, free, [(r, 1) for r in reqs], 16)
        self.assertEqual(len(calls), 3)
        for state, slots, null_slot in calls:
            self.assertIs(state, layer.kv_cache[1])
            self.assertEqual(null_slot, L._NULL_SLOT)
            self.assertEqual(slots.tolist(), [r.slot for r in reqs] + [0] * 3)
        torch.testing.assert_close(layer.kv_cache[1][0], null_before, rtol=0, atol=0)

    def test_decode_bucket_wider_than_step_takes_gather_path(self):
        # 3 decodes -> plan bucket 8, but the step holds only 4 rows.
        layer = _Layer(dim_first=True)
        free = list(range(NUM_SLOTS - 1, 0, -1))
        reqs = [_Request(4) for _ in range(3)]
        with mock.patch.object(L, "is_conv_state_dim_first", lambda: True):
            self._step(layer, free, [(r, 4) for r in reqs], 16)
            calls = self._spy_decode_calls()
            self._step(layer, free, [(r, 1) for r in reqs], 4)
        self.assertEqual(len(calls), 1)
        state, slots, null_slot = calls[0]
        self.assertIs(state, layer.kv_cache[1])
        self.assertEqual(null_slot, L._NULL_SLOT)
        self.assertEqual(slots.numel(), 8)

    def test_packed_decode_in_place_matches_gather_scatter(self):
        n = 4
        mixed = torch.randn(n, DIM)
        g = torch.randn(1, n, H, D)
        b = torch.randn(1, n, H)
        a_log, dt_bias = torch.randn(H) * 0.3, torch.randn(H * D) * 0.3
        slots = torch.tensor([7, 3, 11, 5], dtype=torch.int32)
        cache = torch.randn(NUM_SLOTS, H, D, D)
        # Same values behind a view, which must take gather -> kernel -> scatter.
        base = torch.cat([torch.zeros(1, H, D, D), cache])
        view = base[1:]
        self.assertTrue(V.can_update_in_place(cache))
        self.assertFalse(V.can_update_in_place(view))

        out_a, state_a = V.fused_recurrent_kda_packed_decode(
            mixed, g, b, a_log, dt_bias, LB, cache, slots.long(), null_slot=0
        )
        out_b, _ = V.fused_recurrent_kda_packed_decode(
            mixed, g, b, a_log, dt_bias, LB, view, slots, null_slot=0
        )
        self.assertIs(state_a, cache)
        torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
        torch.testing.assert_close(cache, view, rtol=0, atol=0)
        self.assertTrue((base[0] == 0).all())

        # Null-slot entries: zero output, slot untouched, others still advance.
        null_before = cache[0].clone()
        slot7_before = cache[7].clone()
        out_c, _ = V.fused_recurrent_kda_packed_decode(
            mixed,
            g,
            b,
            a_log,
            dt_bias,
            LB,
            cache,
            torch.tensor([7, 3, 0, 0], dtype=torch.int32),
            null_slot=0,
        )
        self.assertTrue((out_c[0, 2:] == 0).all())
        self.assertFalse((out_c[0, :2] == 0).all())
        torch.testing.assert_close(cache[0], null_before, rtol=0, atol=0)
        self.assertFalse(torch.equal(cache[7], slot7_before))

        with self.assertRaisesRegex(ValueError, "state indices"):
            V.fused_recurrent_kda_packed_decode(
                mixed, g, b, a_log, dt_bias, LB, cache, slots[:2]
            )


if __name__ == "__main__":
    unittest.main()
