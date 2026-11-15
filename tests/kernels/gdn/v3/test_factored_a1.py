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
"""Gate A1 of the factored GDN window (gdn-kernel/design.md): kernel numerics
on TPU against float64 over a long run.

SELECT and the factored LOG kernel each run on their own pool at the served
Qwen3.8 Flash decode geometry through F7_A1_STEPS (2,000) steps of 32
sequences: verify windows of 4, shorter windows, one-token steps, PER_SEQ
chunks (so a sequence moves between the two kernels both ways), skips and
rare fresh restarts, so a sequence lives about 500 steps. Each sequence draws
a decay regime at its start: benign (decays e^-0.05..e^-0.001, long memory),
served (the served expression on N(0, 1) inputs) or adverse (decays of
exactly 0, ~1e-38, subnormal, and runs of e^-20..e^-40 whose products
underflow).

Every step compares, against a float64 recurrence carried per sequence, the
outputs (float32 here), the stored base, the recurrent state rebuilt at every
read offset the last window can name, and the conv state, for BATCHED and
PER_SEQ steps apart; checks every factored output and stored value is finite;
and checks the factored kernel writes no row outside its sequences' block 0
(state, conv and, BATCHED, log rows). Optional shadow bodies
(F7_A1_SHADOWS, comma separated variants of bench_gdn_factored.py) run on
their own pool copy and must equal the factored kernel bit for bit, outputs
and every pool row.

Pass: for each quantity and step kind, the factored kernel's error is at most
1.25 times SELECT's in max (max |x - ref| / rms(ref)) and in rms (rms(x -
ref) / rms(ref)), over the first 64 steps and over the last 64.
"""

import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput

NUM_SEQS = 32
NUM_STEPS = int(os.environ.get("F7_A1_STEPS", "2000"))
TOKENS = 512
ROWS, PAYLOAD, POOL_LANES = 512, (1, 4), 256
STATE_ROWS, LOG_END = 400, 429
WINDOW_STEPS = 64
LIMIT = 1.25


def _call(qkv, b, a, pool, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck, *, plan, factored):
    from test_factored_window import D, K, KS, N_KQ, N_V

    return wrapper.fused_conv1d_gdn(
        Conv1DInput(qkv, jnp.float32),
        b, a, None, None, conv_w, None, a_log, dt_bias, qsl, si, dist, sl, ro, ck,
        n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=KS,
        state_source=pool, state_plan=plan, num_spec_tokens=K,
        conv_weight_kmajor=True, factored_window=factored,
    )


def _compile(variant, plan, factored, args):
    """An executable of one kernel body, compiled under the variant's patches
    (bench_gdn_factored._patched) and kept, so bodies do not share a trace."""
    from bench_gdn_factored import _patched

    with _patched(variant):
        fn = functools.partial(_call, plan=plan, factored=factored)
        return jax.jit(fn, donate_argnums=(3,)).lower(*args).compile()


@dataclasses.dataclass
class _Seq:
    group: list[int]
    regime: str = "served"
    seq_len: int = 0
    ro: int = 0
    last_r: int = 0
    conv: np.ndarray | None = None
    state: np.ndarray | None = None
    window: list = dataclasses.field(default_factory=list)
    prev_kind: str = "fresh"


def test_factored_a1_tpu(monkeypatch):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel")
    monkeypatch.setenv("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")
    from test_factored_window import (
        D, DIM, KS, LOG_PLAN, N_V, SELECT_PLAN, W, _a_for, _err, _ref_chunk, _rms_err, _softplus,
    )
    from test_window_log import _f8, _pool_copy, _probe, _row_mask_equal, _rows_unchanged

    num_blocks = 1 + 4 * W * NUM_SEQS
    probes_n = NUM_SEQS * W
    rng = np.random.default_rng(int(os.environ.get("F7_A1_SEED", "20261113")))
    shadows = [v for v in os.environ.get("F7_A1_SHADOWS", "").split(",") if v]

    raw = rng.integers(0, 256, size=(num_blocks, ROWS, *PAYLOAD, POOL_LANES), dtype=np.uint8)
    pool_sel = _f8(jnp.asarray(raw))
    pool_fac = _pool_copy(pool_sel)
    pool_sh = {v: _pool_copy(pool_sel) for v in shadows}
    del raw
    conv_w = jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32)
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32)
    dt_bias = jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16)
    conv_w64 = np.asarray(conv_w, np.float64)[:, 0, :]
    a_log64 = np.asarray(a_log, np.float64)
    dt_bias64 = np.asarray(dt_bias.astype(jnp.float32), np.float64)

    free = list(range(1, num_blocks))
    rng.shuffle(free)

    def start(seq):
        seq.seq_len, seq.ro, seq.last_r, seq.window = 0, 0, 0, []
        seq.conv, seq.state = np.zeros((KS - 1, DIM)), np.zeros((N_V, D, D))
        seq.regime = str(rng.choice(["benign", "served", "adverse"]))
        seq.prev_kind = "fresh"

    seqs = []
    for _ in range(NUM_SEQS):
        seq = _Seq(group=[free.pop() for _ in range(W)])
        start(seq)
        seqs.append(seq)

    def gate_inputs(regime, n):
        if regime == "benign":
            return _a_for(rng.uniform(-0.05, -0.001, (n, N_V)), a_log64, dt_bias64)
        a = rng.normal(0, 1, (n, N_V))
        if regime == "served":
            return a
        if rng.random() < 0.3:
            return _a_for(rng.uniform(-40.0, -20.0, (n, N_V)), a_log64, dt_bias64)
        u = rng.random((n, N_V))
        a = np.where(u < 0.1, 300.0, a)
        a = np.where((u >= 0.1) & (u < 0.2), _a_for(np.full((n, N_V), -86.0), a_log64, dt_bias64), a)
        a = np.where((u >= 0.2) & (u < 0.25), _a_for(np.full((n, N_V), -95.0), a_log64, dt_bias64), a)
        small = _a_for(rng.uniform(-40.0, -20.0, (n, N_V)), a_log64, dt_bias64)
        return np.where((u >= 0.25) & (u < 0.4), small, a)

    counts = dict.fromkeys(
        ["verify_full", "verify_short", "one_token", "per_seq", "one_token_then_verify",
         "per_seq_then_batched", "batched_then_per_seq", "skipped", "restarts", "loads_offset_gt0",
         "zero_decays", "nonfinite", "stray_rows", "shadow_mismatch_steps"],
        0,
    )
    quantities = ("out", "base", "rebuilt", "conv")
    kinds = ("batched", "per_seq")
    # stats[(kernel, quantity, kind)] = per step (max of max-metric, max of
    # rms-metric, median of rms-metric) over that step's sequences.
    stats = {}
    compiled = {}

    for step in range(NUM_STEPS):
        for seq in seqs:
            if step and rng.random() < 0.002:
                free.extend(seq.group)
                seq.group = [free.pop() for _ in range(W)]
                start(seq)
                counts["restarts"] += 1
        batched, per_seq = [], []
        for s, seq in enumerate(seqs):
            if seq.seq_len and rng.random() < 0.1:
                counts["skipped"] += 1
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
        for slot, (s, n, kind) in enumerate(order):
            seq = seqs[s]
            qlens[slot], si[slot], ck[slot] = n, seq.group[0], seq.group
            sl[slot], ro[slot] = seq.seq_len + n, seq.ro
            counts["loads_offset_gt0"] += int(seq.ro > 0 and seq.seq_len > 0)
            if kind == "verify":
                counts["verify_full" if n == W else "verify_short"] += 1
            else:
                counts[kind] += 1
            if seq.prev_kind == "one_token" and kind == "verify":
                counts["one_token_then_verify"] += 1
            if seq.prev_kind == "per_seq" and kind != "per_seq":
                counts["per_seq_then_batched"] += 1
            if seq.prev_kind in ("verify", "one_token") and kind == "per_seq":
                counts["batched_then_per_seq"] += 1
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        dist = np.array([len(batched), len(order), len(order)], np.int32)
        qkv = rng.normal(0, 1, (TOKENS, 1, DIM)).astype(np.float32)
        b = rng.normal(0, 1, (TOKENS, N_V)).astype(np.float32)
        a = rng.normal(0, 1, (TOKENS, N_V))
        for slot, (s, n, _) in enumerate(order):
            a[qsl[slot] : qsl[slot] + n] = gate_inputs(seqs[s].regime, n)
        a = a.astype(np.float32)
        args = (
            jnp.asarray(qkv), jnp.asarray(b), jnp.asarray(a), None, conv_w, a_log, dt_bias,
            jnp.asarray(qsl), jnp.asarray(si), jnp.asarray(dist), jnp.asarray(sl),
            jnp.asarray(ro), jnp.asarray(ck),
        )

        def with_pool(pool):
            return args[:3] + (pool,) + args[4:]

        if not compiled:
            compiled["sel"] = _compile("select", SELECT_PLAN, False, with_pool(pool_sel))
            compiled["fac"] = _compile("fact", LOG_PLAN, True, with_pool(pool_fac))
            for v in shadows:
                compiled[v] = _compile(v, LOG_PLAN, True, with_pool(pool_sh[v]))
            print(f"A1_COMPILED {sorted(compiled)}", flush=True)

        before_fac = _pool_copy(pool_fac)
        pool_sel, out_sel = compiled["sel"](*with_pool(pool_sel))
        pool_fac, out_fac = compiled["fac"](*with_pool(pool_fac))
        for v in shadows:
            pool_sh[v], out_sh = compiled[v](*with_pool(pool_sh[v]))
            same_out = bool(jnp.all(
                jax.lax.bitcast_convert_type(out_sh, jnp.uint32)
                == jax.lax.bitcast_convert_type(out_fac, jnp.uint32)
            ))
            blocks = jnp.arange(num_blocks, dtype=jnp.int32)
            same_pool = bool(jnp.all(
                _row_mask_equal(pool_sh[v], blocks, pool_fac, blocks, r0=0, r1=ROWS) == 1
            ))
            if not (same_out and same_pool):
                counts["shadow_mismatch_steps"] += 1
                if counts["shadow_mismatch_steps"] <= 5:
                    print(f"A1_SHADOW_MISMATCH step={step} variant={v} outputs_equal={same_out} "
                          f"pools_equal={same_pool}", flush=True)
                pool_sh[v] = _pool_copy(pool_fac)
        out_sel, out_fac = np.asarray(out_sel), np.asarray(out_fac)

        # Rows the factored kernel may write: block 0's state and conv rows,
        # and BATCHED its log rows.
        allowed = np.zeros((num_blocks, ROWS), bool)
        for s, n, kind in order:
            allowed[seqs[s].group[0], : (STATE_ROWS if kind == "per_seq" else LOG_END)] = True
        stray = np.argwhere(~np.asarray(_rows_unchanged(before_fac, pool_fac)) & ~allowed)
        del before_fac
        if stray.size:
            counts["stray_rows"] += len(stray)
            print(f"A1_STRAY step={step} rows={len(stray)} first={tuple(int(x) for x in stray[0])}")

        errs = {}

        def add(kernel, quantity, kind, x, ref):
            key = (kernel, quantity, kind)
            errs.setdefault(key, []).append((_err(x, ref), _rms_err(x, ref)))

        probes, refs = [], []
        for slot, (s, n, kind) in enumerate(order):
            seq = seqs[s]
            lo = qsl[slot]
            g = np.exp(-np.exp(a_log64) * _softplus(a[lo : lo + n].astype(np.float64) + dt_bias64))
            counts["zero_decays"] += int((g == 0).sum())
            outs, after = _ref_chunk(
                seq, qkv[lo : lo + n, 0].astype(np.float64), b[lo : lo + n].astype(np.float64),
                a[lo : lo + n].astype(np.float64), conv_w64, a_log64, dt_bias64,
            )
            k = "per_seq" if kind == "per_seq" else "batched"
            fo = out_fac[lo : lo + n].reshape(n, N_V, D)
            if not np.all(np.isfinite(fo)):
                counts["nonfinite"] += 1
            add("sel", "out", k, out_sel[lo : lo + n].reshape(n, N_V, D), outs)
            add("fac", "out", k, fo, outs)
            seq.seq_len += n
            window = after if kind == "verify" else [after[-1]]
            # Every offset the next step may name; offset 0 is the stored base.
            for j, (conv_j, state_j) in enumerate(window):
                probes.append((seq.group[j], seq.group[0], j, k))
                refs.append((conv_j, state_j))
            seq.ro = int(rng.integers(1, n + 1)) - 1 if kind == "verify" else 0
            seq.window = window
            seq.conv, seq.state = window[seq.ro]
            seq.last_r = len(window)
            seq.prev_kind = kind
        for p0 in range(0, len(probes), probes_n):
            chunk = probes[p0 : p0 + probes_n]
            pad = probes_n - len(chunk)
            sel_blocks = jnp.asarray([p[0] for p in chunk] + [0] * pad, jnp.int32)
            fac_blocks = jnp.asarray([p[1] for p in chunk] + [0] * pad, jnp.int32)
            offs = jnp.asarray([p[2] for p in chunk] + [0] * pad, jnp.int32)
            zeros = jnp.zeros((probes_n,), jnp.int32)
            sel_rec, sel_conv = (np.asarray(x) for x in _probe(pool_sel, sel_blocks, zeros)[:2])
            fac_rec, fac_conv = (np.asarray(x) for x in _probe(pool_fac, fac_blocks, offs)[:2])
            for i, (_, _, j, k) in enumerate(chunk):
                conv_ref, state_ref = refs[p0 + i]
                if not (np.all(np.isfinite(fac_rec[i])) and np.all(np.isfinite(fac_conv[i]))):
                    counts["nonfinite"] += 1
                for kernel, rec, conv in (("sel", sel_rec[i], sel_conv[i]), ("fac", fac_rec[i], fac_conv[i])):
                    add(kernel, "rebuilt", k, rec, state_ref)
                    if j == 0:
                        add(kernel, "base", k, rec, state_ref)
                    add(kernel, "conv", k, conv[:, 0], conv_ref)
        for key, vals in errs.items():
            vals = np.asarray(vals)
            stats.setdefault(key, []).append(
                (step, vals[:, 0].max(), vals[:, 1].max(), float(np.median(vals[:, 1])))
            )
        if step % 50 == 0 or step == NUM_STEPS - 1:
            line = " ".join(
                f"{q}/{k}={stats[('fac', q, k)][-1][2]:.3g}vs{stats[('sel', q, k)][-1][2]:.3g}"
                for q in quantities for k in kinds
                if ("fac", q, k) in stats and stats[("fac", q, k)][-1][0] == step
            )
            print(f"A1_STEP {step} rms_max fac vs sel: {line}", flush=True)

    print("A1_COUNTS " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
    windows = {"first": (0, min(WINDOW_STEPS, NUM_STEPS)), "last": (max(0, NUM_STEPS - WINDOW_STEPS), NUM_STEPS)}
    failures = []
    for q in quantities:
        for k in kinds:
            for wname, (lo, hi) in windows.items():
                row = {}
                for kernel in ("sel", "fac"):
                    pts = [p for p in stats.get((kernel, q, k), []) if lo <= p[0] < hi]
                    if not pts:
                        continue
                    arr = np.asarray([p[1:] for p in pts])
                    row[kernel] = (arr[:, 0].max(), arr[:, 1].max(), float(np.median(arr[:, 2])))
                if len(row) < 2:
                    continue
                (s_max, s_rms, s_med), (f_max, f_rms, f_med) = row["sel"], row["fac"]
                ok = f_max <= LIMIT * s_max + 1e-12 and f_rms <= LIMIT * s_rms + 1e-12
                print(f"A1_SUMMARY {q} {k} {wname}{hi} max fac={f_max:.3g} sel={s_max:.3g} "
                      f"ratio={f_max / max(s_max, 1e-30):.3g} | rms fac={f_rms:.3g} sel={s_rms:.3g} "
                      f"ratio={f_rms / max(s_rms, 1e-30):.3g} | median rms fac={f_med:.3g} "
                      f"sel={s_med:.3g} | {'ok' if ok else 'OVER'}", flush=True)
                if not ok:
                    failures.append(f"{q} {k} {wname}")
    for name in ("nonfinite", "stray_rows", "shadow_mismatch_steps"):
        if counts[name]:
            failures.append(f"{name}={counts[name]}")
    print(f"A1_RESULT {'PASS' if not failures else 'FAIL'} steps={NUM_STEPS} shadows={shadows} "
          f"failures={failures}", flush=True)
    assert not failures, failures
