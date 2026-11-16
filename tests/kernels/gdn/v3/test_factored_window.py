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
"""The factored LOG-mode window body (`compute_gdn.factored_gdn`) against the
recurrence.

`TestFactoredBody` runs the body eagerly next to today's recurrent body on
the folded state, and both next to a float64 reference, for every offset and
real window size, with decays drawn normally and with zero, tiny and
underflowing decays (logged and in the window).

`test_factored_numerics_tpu` drives the pooled kernel on TPU, SELECT and the
factored LOG kernel each on its own pool, through 64 steps of 32 sequences
with drawn offsets (verify windows, shorter windows, one-token steps, PER_SEQ
chunks, fresh starts, skips), and compares the outputs, the stored base and
the state the next step resumes from against a float64 reference of the
recurrence carried per sequence. The outputs are float32 here (the served
kernel writes bf16, whose rounding would hide the arithmetic's error).
"""

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, vmem_ldst, wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

N_KQ, N_V, D, K, DIM, KS = 2, 6, 128, 3, 1280, 4
W = K + 1
F32 = jnp.dtype(jnp.float32)
LANES = D  # aligned_num_v_heads on TPU

SELECT_PLAN = config.StateSourcePlan(
    stride=1,
    conv=config.StateRegion(0, 1, 384, 16, F32, 1, 15),
    recurrent=config.StateRegion(0, 1, 0, 384, F32, 2, 768),
)
LOG_PLAN = dataclasses.replace(
    SELECT_PLAN,
    log=config.StateRegion(0, 1, 400, 14, F32, 2, 27),
    log_conv=config.StateRegion(0, 1, 414, 15, F32, 1, 15),
)


def _cfg(seqs, factored):
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=F32, act_out=F32, compute=F32, recurrent_state=F32, conv_state=F32
        ),
        batch_size=seqs * W,
        dim_size=DIM,
        kernel_size=KS,
        tile_size=seqs,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D,
        v_head_dim=D,
        state_plan=LOG_PLAN,
        window_size=W,
        checkpoint_mode=config.CheckpointMode.LOG,
        log_positions=K,
        factored_window=factored,
    )


# --------------------------------------------------------------------------
# Float64 reference of the recurrence (compute_gdn.recurrent_gdn_per_seq).
# --------------------------------------------------------------------------


def _softplus(x):
    return np.logaddexp(0.0, x)


def _prep(q, k, b, a, a_log, dt_bias):
    """Activations after l2_norm, scale, sigmoid and the decay, float64."""
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * D**-0.5
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    beta = 1.0 / (1.0 + np.exp(-b))
    g = np.exp(-np.exp(a_log) * _softplus(a + dt_bias))
    return q, k, beta, g


def _ref_token(state, q, k, v, beta, g):
    """One token: q, k [N_KQ, D]; v [N_V, D]; beta, g [N_V]. Returns (out, state, v_new)."""
    kr = np.repeat(k, N_V // N_KQ, axis=0)
    qr = np.repeat(q, N_V // N_KQ, axis=0)
    state = state * g[:, None, None]
    u = np.einsum("hk,hkv->hv", kr, state)
    v_new = beta[:, None] * (v - u)
    state = state + kr[:, :, None] * v_new[:, None, :]
    return np.einsum("hk,hkv->hv", qr, state), state, v_new


# --------------------------------------------------------------------------
# CPU: the body against the recurrent body and the reference.
# --------------------------------------------------------------------------

REGIMES = ("normal", "zero", "tiny", "underflow")


def _window_a(rng, regime, shape, a_log, dt_bias):
    """Gate inputs `a` whose decays follow the regime."""
    a = rng.normal(0, 1, shape)
    if regime == "normal":
        return a

    def a_for(log_decay):
        # softplus(a + dt_bias) * exp(a_log) = -log_decay
        y = -log_decay / np.exp(a_log)
        return np.log(np.expm1(y)) - dt_bias

    pick = rng.random(shape) < 0.5
    if regime == "zero":
        target = np.full(shape, 300.0 + 0 * a)
        return np.where(pick, target, a)
    if regime == "tiny":
        log_decay = rng.choice([-86.0, -80.0, -60.0], size=shape)
        return np.where(pick, a_for(log_decay), a)
    # Every decay small enough that a product of a few underflows.
    return a_for(rng.uniform(-40.0, -20.0, size=shape))


def _log_decays(rng, regime, shape):
    g = rng.uniform(0.7, 1.0, shape)
    if regime == "zero":
        return np.where(rng.random(shape) < 0.5, 0.0, g)
    if regime == "tiny":
        return np.where(rng.random(shape) < 0.5, rng.choice([1.5e-38, 1e-35, 1e-26], size=shape), g)
    if regime == "underflow":
        return np.exp(rng.uniform(-40.0, -20.0, shape))
    return g


def _err(x, ref):
    """max |x - ref| relative to the reference's RMS."""
    x = np.asarray(x, np.float64)
    scale = max(float(np.sqrt(np.mean(ref * ref))), 1e-30)
    return float(np.max(np.abs(x - ref))) / scale


def _rms_err(x, ref):
    """RMS of x - ref relative to the reference's RMS."""
    x = np.asarray(x, np.float64)
    scale = max(float(np.sqrt(np.mean(ref * ref))), 1e-30)
    return float(np.sqrt(np.mean((x - ref) ** 2))) / scale


UPDATE_PASSES = {
    "split": "factored_update_pass",
    "bf16": "factored_update_pass_bf16",
    "vpu": "factored_update_pass_vpu",
}


class TestFactoredBody:
    @pytest.mark.parametrize("update", ["split", "vpu"])
    @pytest.mark.parametrize("regime", REGIMES)
    def test_matches_recurrence(self, regime, update, monkeypatch):
        monkeypatch.setattr(
            compute_gdn, "factored_update_pass", getattr(compute_gdn, UPDATE_PASSES[update])
        )
        rng = np.random.default_rng(REGIMES.index(regime))
        seqs = 8
        rec_cfg, fac_cfg = _cfg(seqs, False), _cfg(seqs, True)
        a_log = np.log(rng.uniform(1, 16, N_V))
        dt_bias = rng.normal(0, 1, N_V)
        a_log_p = jnp.pad(jnp.asarray(a_log, jnp.float32), (0, LANES - N_V))
        dt_bias_p = jnp.pad(jnp.asarray(dt_bias, jnp.float32), (0, LANES - N_V))
        worst = {"out_rec": 0.0, "out_fac": 0.0, "base_rec": 0.0, "base_fac": 0.0, "f32_diff": 0.0}
        for trial in range(3):
            real = rng.integers(1, W + 1, seqs)
            offsets = rng.integers(0, K + 1, seqs)
            real[0], offsets[0] = W, K  # the longest chain every trial
            base = rng.normal(0, 0.3, (seqs, N_V, D, D)).astype(np.float32)
            q = rng.normal(0, 1, (seqs, N_KQ, W, D)).astype(np.float32)
            k = rng.normal(0, 1, (seqs, N_KQ, W, D)).astype(np.float32)
            v = rng.normal(0, 1, (seqs, N_V, W, D)).astype(np.float32)
            b = np.zeros((seqs, 1, W, LANES), np.float32)
            a = np.zeros((seqs, 1, W, LANES), np.float32)
            b[..., :N_V] = rng.normal(0, 1, (seqs, 1, W, N_V))
            a[..., :N_V] = _window_a(rng, regime, (seqs, 1, W, N_V), a_log, dt_bias)
            # Logged updates: decays, unit keys, update vectors.
            lg = _log_decays(rng, regime, (seqs, K, N_V)).astype(np.float32)
            lk = rng.normal(0, 1, (seqs, K, N_KQ, D))
            lk = (lk / np.linalg.norm(lk, axis=-1, keepdims=True)).astype(np.float32)
            lu = rng.normal(0, 1, (seqs, K, N_V, D)).astype(np.float32)

            # Typed log rows, NaN past the offset and in the unused gate lanes.
            log_rows, counts = [], []
            for s in range(seqs):
                gates = np.full((K, D), np.nan, np.float32)
                gates[:, :N_V] = lg[s]
                rows = np.concatenate([gates, lk[s].reshape(-1, D), lu[s].reshape(-1, D)])
                for j in range(offsets[s], K):
                    rows[j] = np.nan
                    rows[K + j * N_KQ : K + (j + 1) * N_KQ] = np.nan
                    rows[K + K * N_KQ + j * N_V : K + K * N_KQ + (j + 1) * N_V] = np.nan
                log_rows.append(jnp.asarray(rows))
                counts.append(jnp.int32(offsets[s]))

            fac_out, fac_state, fac_log = compute_gdn.factored_gdn(
                jnp.asarray(real, jnp.int32),
                jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.asarray(b), jnp.asarray(a),
                jnp.asarray(base), log_rows, counts, a_log_p, dt_bias_p, fac_cfg,
            )

            # Today's body on the folded state.
            folded = []
            for s in range(seqs):
                gates = np.ones((K, 1, 1, D), np.float32)
                gates[:, 0, 0, :N_V] = lg[s]
                rec, _ = vmem_ldst.fold_window_log(
                    jnp.asarray(base[s]),
                    jnp.zeros((KS - 1, 1, DIM), jnp.float32),
                    jnp.asarray(gates),
                    jnp.asarray(lk[s][:, :, None, :]),
                    jnp.asarray(lu[s][:, :, None, :]),
                    jnp.zeros((K, 1, DIM), jnp.float32),
                    jnp.int32(offsets[s]),
                    rec_cfg,
                )
                folded.append(rec)
            rec_out, rec_state, (rec_gates, rec_keys, rec_upd) = compute_gdn.recurrent_gdn(
                jnp.asarray(real, jnp.int32),
                jnp.asarray(q[:, :, :, None, :]), jnp.asarray(k[:, :, :, None, :]),
                jnp.asarray(v[:, :, :, None, :]),
                jnp.asarray(b[:, :, :, None, :]), jnp.asarray(a[:, :, :, None, :]),
                jnp.stack(folded), a_log_p.reshape(-1), dt_bias_p.reshape(-1), rec_cfg,
            )

            # Reference, float64.
            qn, kn, beta, g = _prep(
                q.astype(np.float64), k.astype(np.float64),
                b[..., :N_V].astype(np.float64), a[..., :N_V].astype(np.float64), a_log, dt_bias,
            )
            for s in range(seqs):
                state = base[s].astype(np.float64)
                for j in range(offsets[s]):
                    kr = np.repeat(lk[s, j].astype(np.float64), N_V // N_KQ, axis=0)
                    state = state * lg[s, j].astype(np.float64)[:, None, None]
                    state = state + kr[:, :, None] * lu[s, j].astype(np.float64)[:, None, :]
                outs, state0 = [], None
                for c in range(real[s]):
                    o, state, _ = _ref_token(
                        state, qn[s, :, c], kn[s, :, c], v[s, :, c].astype(np.float64),
                        beta[s, 0, c], g[s, 0, c],
                    )
                    outs.append(o)
                    if c == 0:
                        state0 = state
                outs = np.stack(outs)
                n = real[s]
                fo, ro = np.asarray(fac_out[s, :n]), np.asarray(rec_out[s, :n])
                assert np.all(np.isfinite(fo)) and np.all(np.isfinite(np.asarray(fac_state[s])))
                worst["out_fac"] = max(worst["out_fac"], _err(fo, outs))
                worst["out_rec"] = max(worst["out_rec"], _err(ro, outs))
                worst["base_fac"] = max(worst["base_fac"], _err(fac_state[s, 0], state0))
                worst["base_rec"] = max(worst["base_rec"], _err(rec_state[s, 0], state0))
                worst["f32_diff"] = max(worst["f32_diff"], _err(fo, ro.astype(np.float64)))
                # The log of positions 1..K matches today's (keys and gates
                # exactly; updates to float32 rounding).
                rows = np.asarray(fac_log[s])
                np.testing.assert_array_equal(rows[:K, :N_V], np.asarray(rec_gates[s])[:, :N_V])
                np.testing.assert_array_equal(
                    rows[K : K + K * N_KQ], np.asarray(rec_keys[s]).reshape(-1, D)
                )
                upd = np.asarray(rec_upd[s]).reshape(-1, D)
                np.testing.assert_allclose(
                    rows[K + K * N_KQ :], upd, rtol=0, atol=1e-5 * max(1.0, np.abs(upd).max())
                )
        print(f"FACTORED_CPU regime={regime} update={update} " + " ".join(f"{k}={v:.3g}" for k, v in worst.items()))
        # Same arithmetic class as the recurrence: errors at float32 level,
        # not more than a few times the recurrent body's own.
        assert worst["out_fac"] < max(4 * worst["out_rec"], 1e-5), worst
        assert worst["base_fac"] < max(4 * worst["base_rec"], 1e-5), worst


def test_stage_order_is_the_same_arithmetic():
    """`factored_gdn(order=...)` only reorders the trace: equal bits."""
    rng = np.random.default_rng(11)
    seqs = 4
    cfg = _cfg(seqs, True)
    a_log = jnp.pad(jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32), (0, LANES - N_V))
    dt_bias = jnp.pad(jnp.asarray(rng.normal(0, 1, N_V), jnp.float32), (0, LANES - N_V))
    args = (
        jnp.asarray([4, 3, 1, 4], jnp.int32),
        jnp.asarray(rng.normal(0, 1, (seqs, N_KQ, W, D)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (seqs, N_KQ, W, D)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (seqs, N_V, W, D)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (seqs, 1, W, LANES)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (seqs, 1, W, LANES)), jnp.float32),
        jnp.asarray(rng.normal(0, 0.3, (seqs, N_V, D, D)), jnp.float32),
        [jnp.asarray(rng.normal(0, 0.5, (K + K * N_KQ + K * N_V, D)), jnp.float32) for _ in range(seqs)],
        [jnp.int32(c) for c in (3, 0, 2, 1)],
        a_log,
        dt_bias,
        cfg,
    )
    one = compute_gdn.factored_gdn(*args)
    for order in ("phased", "pipelined"):
        two = compute_gdn.factored_gdn(*args, order=order)
        for x, y in zip(jax.tree.leaves(one), jax.tree.leaves(two)):
            np.testing.assert_array_equal(np.asarray(x).view(np.uint32), np.asarray(y).view(np.uint32))


# --------------------------------------------------------------------------
# TPU: 64 steps of 32 sequences, SELECT and factored LOG against float64.
# --------------------------------------------------------------------------

NUM_SEQS = 32
NUM_STEPS = int(os.environ.get("F7_FACT_STEPS", "64"))
TOKENS = 512
ROWS, PAYLOAD, POOL_LANES = 512, (1, 4), 256
NUM_BLOCKS = 1 + 4 * W * NUM_SEQS
PROBES = NUM_SEQS * W


@functools.partial(jax.jit, static_argnames=("plan", "factored"), donate_argnums=(3,))
def _run(qkv, b, a, pool, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck, *, plan, factored):
    return wrapper.fused_conv1d_gdn(
        Conv1DInput(qkv, jnp.float32),
        b, a, None, None, conv_w, None, a_log, dt_bias, qsl, si, dist, sl, ro, ck,
        n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=KS,
        state_source=pool, state_plan=plan, num_spec_tokens=K,
        conv_weight_kmajor=True, factored_window=factored,
    )


def _a_for(log_decay, a_log, dt_bias):
    """The gate input `a` whose decay is exp(log_decay)."""
    y = -log_decay / np.exp(a_log)
    return np.log(np.expm1(y)) - dt_bias


def _a_with_regimes(rng, a_log, dt_bias, tokens):
    """Gate inputs: mostly N(0, 1), with zero, tiny, subnormal and small
    decays mixed in per (token, head)."""
    a = rng.normal(0, 1, (tokens, N_V))
    u = rng.random((tokens, N_V))
    a = np.where(u < 0.02, 300.0, a)  # decay exactly 0
    a = np.where((u >= 0.02) & (u < 0.04), _a_for(np.full((tokens, N_V), -86.0), a_log, dt_bias), a)
    a = np.where((u >= 0.04) & (u < 0.05), _a_for(np.full((tokens, N_V), -95.0), a_log, dt_bias), a)
    small = _a_for(rng.uniform(-40.0, -20.0, (tokens, N_V)), a_log, dt_bias)
    return np.where((u >= 0.05) & (u < 0.10), small, a)


@dataclasses.dataclass
class _Seq:
    group: list[int]
    seq_len: int = 0
    ro: int = 0
    last_r: int = 0
    conv: np.ndarray | None = None  # [KS - 1, DIM] f64, state to resume from
    state: np.ndarray | None = None  # [N_V, D, D] f64
    # Per position of its last BATCHED window: (conv, state) after it.
    window: list = dataclasses.field(default_factory=list)


def _ref_chunk(seq, x, b, a, conv_w, a_log, dt_bias):
    """Runs the reference over a chunk of tokens from the sequence's resume
    point. Returns outputs [n, N_V, D] and (conv, state) after every token."""
    rows = np.concatenate([seq.conv, x], axis=0)
    outs, after = [], []
    state = seq.state
    for t in range(x.shape[0]):
        conv_out = sum(rows[t + j] * conv_w[j] for j in range(KS))
        act = conv_out / (1.0 + np.exp(-conv_out))
        q = act[: N_KQ * D].reshape(N_KQ, D)
        k = act[N_KQ * D : 2 * N_KQ * D].reshape(N_KQ, D)
        v = act[2 * N_KQ * D :].reshape(N_V, D)
        q, k, beta, g = _prep(q, k, b[t], a[t], a_log, dt_bias)
        o, state, _ = _ref_token(state, q, k, v, beta, g)
        outs.append(o)
        after.append((rows[t + 1 : t + KS].copy(), state))
    return np.stack(outs), after


def test_factored_numerics_tpu(monkeypatch):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    monkeypatch.setenv("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")
    update = os.environ.get("F7_FACT_UPDATE", "split")
    monkeypatch.setattr(
        compute_gdn, "factored_update_pass", getattr(compute_gdn, UPDATE_PASSES[update])
    )
    jax.clear_caches()
    print(f"FACT_NUM_UPDATE {update}")
    from test_window_log import _f8, _pool_copy, _probe

    rng = np.random.default_rng(int(os.environ.get("F7_FACT_SEED", "20261112")))
    raw = rng.integers(0, 256, size=(NUM_BLOCKS, ROWS, *PAYLOAD, POOL_LANES), dtype=np.uint8)
    pool_sel = _f8(jnp.asarray(raw))
    pool_fac = _pool_copy(pool_sel)
    del raw
    conv_w64 = rng.normal(0, 0.5, (KS, 1, DIM))
    a_log64 = np.log(rng.uniform(1, 16, N_V))
    dt_bias_bf = jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16)
    conv_w = jnp.asarray(conv_w64, jnp.float32)
    a_log = jnp.asarray(a_log64, jnp.float32)
    # Reference parameters: exactly the float32 / bf16 values the kernels read.
    conv_w64 = np.asarray(conv_w, np.float64)[:, 0, :]
    a_log64 = np.asarray(a_log, np.float64)
    dt_bias64 = np.asarray(dt_bias_bf.astype(jnp.float32), np.float64)

    free = list(range(1, NUM_BLOCKS))
    rng.shuffle(free)

    def new_group():
        return [free.pop() for _ in range(W)]

    def fresh(seq):
        seq.seq_len, seq.ro, seq.last_r, seq.window = 0, 0, 0, []
        seq.conv = np.zeros((KS - 1, DIM))
        seq.state = np.zeros((N_V, D, D))

    seqs = []
    for _ in range(NUM_SEQS):
        seq = _Seq(group=new_group())
        fresh(seq)
        seqs.append(seq)
    # Per step: max over sequences of max |x - ref| / RMS(ref), and max and
    # median over sequences of RMS(x - ref) / RMS(ref).
    worst, worst_rms, med_rms = {}, {}, {}
    zero_decays = 0

    for step in range(NUM_STEPS):
        for s, seq in enumerate(seqs):
            if step and rng.random() < 0.04:
                free.extend(seq.group)
                seq.group = new_group()
                fresh(seq)
        batched, per_seq = [], []
        for s, seq in enumerate(seqs):
            if seq.seq_len and rng.random() < 0.1:
                continue
            u = rng.random()
            if u < 0.55:
                batched.append((s, W, "verify"))
            elif u < 0.70:
                batched.append((s, int(rng.integers(2, W)), "verify"))
            elif u < 0.85 or len(per_seq) >= 6:
                batched.append((s, 1, "one_token"))
            else:
                per_seq.append((s, int(rng.integers(W + 1, 34)), "per_seq"))
        order = batched + per_seq
        qlens = np.zeros(NUM_SEQS, np.int32)
        si = np.zeros(NUM_SEQS, np.int32)
        ck = np.zeros((NUM_SEQS, W), np.int32)
        sl = np.zeros(NUM_SEQS, np.int32)
        ro = np.zeros(NUM_SEQS, np.int32)
        for slot, (s, n, _) in enumerate(order):
            seq = seqs[s]
            qlens[slot], si[slot], ck[slot] = n, seq.group[0], seq.group
            sl[slot], ro[slot] = seq.seq_len + n, seq.ro
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        dist = np.array([len(batched), len(order), len(order)], np.int32)
        qkv = rng.normal(0, 1, (TOKENS, 1, DIM)).astype(np.float32)
        b = rng.normal(0, 1, (TOKENS, N_V)).astype(np.float32)
        a = _a_with_regimes(rng, a_log64, dt_bias64, TOKENS)
        # Some sequences decay by e^-20..e^-40 on every token and head, so
        # products over a window (and its logged updates) underflow.
        for slot, (s, n, _) in enumerate(order):
            if rng.random() < 0.1:
                lo = qsl[slot]
                a[lo : lo + n] = _a_for(rng.uniform(-40.0, -20.0, (n, N_V)), a_log64, dt_bias64)
        a = a.astype(np.float32)
        args = (
            jnp.asarray(qkv), jnp.asarray(b), jnp.asarray(a),
        )
        common = (conv_w, a_log, dt_bias_bf, jnp.asarray(qsl), jnp.asarray(si),
                  jnp.asarray(dist), jnp.asarray(sl), jnp.asarray(ro), jnp.asarray(ck))
        pool_sel, out_sel = _run(*args, pool_sel, *common, plan=SELECT_PLAN, factored=False)
        pool_fac, out_fac = _run(*args, pool_fac, *common, plan=LOG_PLAN, factored=True)
        out_sel, out_fac = np.asarray(out_sel), np.asarray(out_fac)

        keys = ("out_sel", "out_fac", "base_sel", "base_fac", "next_sel", "next_fac")
        errs = {key: [] for key in keys}
        rms = {key: [] for key in keys}
        base_ref, probes = [], []
        for slot, (s, n, kind) in enumerate(order):
            seq = seqs[s]
            lo = qsl[slot]
            g = np.exp(-np.exp(a_log64) * _softplus(a[lo : lo + n].astype(np.float64) + dt_bias64))
            zero_decays += int((g == 0).sum())
            outs, after = _ref_chunk(
                seq, qkv[lo : lo + n, 0].astype(np.float64), b[lo : lo + n].astype(np.float64),
                a[lo : lo + n].astype(np.float64), conv_w64, a_log64, dt_bias64,
            )
            for key, out in (("out_sel", out_sel), ("out_fac", out_fac)):
                errs[key].append(_err(out[lo : lo + n].reshape(n, N_V, D), outs))
                rms[key].append(_rms_err(out[lo : lo + n].reshape(n, N_V, D), outs))
            seq.seq_len += n
            # Checkpoint 0 / block 0: after window position 0, or after the
            # PER_SEQ chunk.
            base_ref.append(after[-1][1] if kind == "per_seq" else after[0][1])
            if kind == "verify":
                seq.ro = int(rng.integers(1, n + 1)) - 1
                seq.window = after
            else:
                seq.ro = 0
                seq.window = [after[-1]]
            seq.conv, seq.state = seq.window[seq.ro]
            probes.append((s, seq.group[seq.ro], seq.group[0], seq.ro))
        n_run = len(order)
        pad = PROBES - n_run
        zeros = jnp.zeros((PROBES,), jnp.int32)
        blocks0 = jnp.asarray([p[2] for p in probes] + [0] * pad, jnp.int32)
        # Block 0 as stored (no fold), both pools.
        sel0 = np.asarray(_probe(pool_sel, blocks0, zeros)[0])
        fac0 = np.asarray(_probe(pool_fac, blocks0, zeros)[0])
        # The state the next step resumes from: SELECT's checkpoint[ro], the
        # factored pool's block 0 with ro logged updates folded on.
        sel_next = np.asarray(
            _probe(pool_sel, jnp.asarray([p[1] for p in probes] + [0] * pad, jnp.int32), zeros)[0]
        )
        fac_next = np.asarray(
            _probe(pool_fac, blocks0, jnp.asarray([p[3] for p in probes] + [0] * pad, jnp.int32))[0]
        )
        for i, (s, _, _, _) in enumerate(probes):
            for key, x, ref in (
                ("base_sel", sel0[i], base_ref[i]),
                ("base_fac", fac0[i], base_ref[i]),
                ("next_sel", sel_next[i], seqs[s].state),
                ("next_fac", fac_next[i], seqs[s].state),
            ):
                errs[key].append(_err(x, ref))
                rms[key].append(_rms_err(x, ref))
        line = []
        for key in keys:
            vals, rvals = np.asarray(errs[key]), np.asarray(rms[key])
            worst.setdefault(key, []).append(vals.max())
            worst_rms.setdefault(key, []).append(rvals.max())
            med_rms.setdefault(key, []).append(np.median(rvals))
            line.append(f"{key}={vals.max():.3g}/{np.median(vals):.3g}/{rvals.max():.3g}/{np.median(rvals):.3g}")
        print(f"FACT_NUM step={step} seqs={n_run} " + " ".join(line), flush=True)

    print(f"FACT_NUM_ZERO_DECAYS {zero_decays}")
    for key, vals in worst.items():
        vals = np.asarray(vals)
        q = max(1, len(vals) // 4)
        print(f"FACT_NUM_SUMMARY {key} max={vals.max():.3g} first_quarter_max={vals[:q].max():.3g} "
              f"last_quarter_max={vals[-q:].max():.3g} median_step_max={np.median(vals):.3g}")
    for key in worst_rms:
        hi, med = np.asarray(worst_rms[key]), np.asarray(med_rms[key])
        q = max(1, len(hi) // 4)
        print(f"FACT_NUM_RMS {key} max={hi.max():.3g} first_quarter_max={hi[:q].max():.3g} "
              f"last_quarter_max={hi[-q:].max():.3g} median_first_quarter={np.median(med[:q]):.3g} "
              f"median_last_quarter={np.median(med[-q:]):.3g}")
    out_ratio = max(worst["out_fac"]) / max(max(worst["out_sel"]), 1e-30)
    base_ratio = max(worst["base_fac"]) / max(max(worst["base_sel"]), 1e-30)
    print(f"FACT_NUM_RATIO out={out_ratio:.3g} base={base_ratio:.3g}")
    assert np.all(np.isfinite(np.asarray(worst["out_fac"])))
    assert np.all(np.isfinite(np.asarray(worst["base_fac"])))
