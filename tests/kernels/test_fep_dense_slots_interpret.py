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
"""The dense-slots layer (v3/dense_slots.py) on CPU, every program in Pallas TPU interpret mode
(remote DMAs, semaphores and barriers simulated) over 8 fake devices, against a per-token reference
of the same math: v3's select, fused EP v3's per-row FFN (ffn.expert_ffn_fp8 / result_rows with
bf16 result rows) and its k-ordered weighted sum.

Needs 8 CPU devices, i.e. XLA_FLAGS=--xla_force_host_platform_device_count=8 before JAX starts
(skipped otherwise). Small shapes: hidden 1024 (whole 8-sublane staging tiles), intermediate 128, 16 or 32 experts, top-k 4.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.fused_moe.v3 import dense_slots as DS
from vllm_torchtpu.kernels.fused_moe.v3 import device as DEV
from vllm_torchtpu.kernels.fused_moe.v3 import ffn
from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8, quantize_rows

D, H, I, K = 8, 1024, 128, 4

VARIANTS = {
    "plain": {},
    "fused": dict(fused=True),
    "pallas_tables": dict(fused=True, pallas_tables=True),
    "merged": dict(fused=True, merged=True),
    "merged_unpadded": dict(fused=True, merged=True, unpadded=True),
    # The unrouted-expert skip (Config.dense_slots_skip_unrouted) and other weight slot counts.
    "plain_skip": dict(skip_unrouted=True),
    "fused_skip": dict(fused=True, skip_unrouted=True),
    "merged_unpadded_skip": dict(fused=True, merged=True, unpadded=True, skip_unrouted=True),
    "merged_unpadded_w4": dict(fused=True, merged=True, unpadded=True, weight_slots=4),
    "merged_unpadded_skip_w2": dict(fused=True, merged=True, unpadded=True, skip_unrouted=True,
                                    weight_slots=2),
}


@pytest.fixture(autouse=True)
def _cpu_budget(monkeypatch):
    # The device check runs here, not at import: the suite refuses a JAX backend started
    # during collection.
    if len(jax.devices()) < D or jax.devices()[0].platform != "cpu":
        pytest.skip(f"needs {D} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count={D})")
    # No TPU to ask for its VMEM size: the programs plan for a 64 MiB core.
    monkeypatch.setattr(DS, "VMEM_LIMIT", 64 << 20)
    monkeypatch.setattr(DEV, "vmem_limit", lambda *a, **k: 64 << 20)
    monkeypatch.setattr(DS, "TABLE_HOOK", None)
    monkeypatch.setattr(DS, "MERGED_POS_SHIFT", 0)
    monkeypatch.setattr(DS, "SKIP_MUTANT", "")


def _world(e, real, hot, seed=0):
    rng = np.random.default_rng(seed)
    tokens = real * D
    x = (rng.standard_normal((tokens, H)) / 10).astype(np.float32)
    logits = rng.standard_normal((tokens, e)).astype(np.float32)
    if hot == "cold":
        # Local experts 1, 3, 5 and 6 of every shard never routed (with 8 per shard: a
        # prefetched expert without rows, cold runs between live ones, and live ones after).
        g = e // D
        cold = [d * g + c for d in range(D) for c in (1, 3, 5, 6) if c < g]
        logits[:, cold] -= 30.0
    elif hot:
        logits[:, :K] += 20.0  # every token -> experts 0..K-1, all on shard 0
    w1f = rng.standard_normal((e, H, 2 * I)).astype(np.float32) / 10
    w2f = rng.standard_normal((e, I, H)).astype(np.float32) / 10
    s1 = np.abs(w1f).max(1) / 448.0
    s2 = np.abs(w2f).max(1) / 448.0
    w1 = jnp.asarray(w1f / s1[:, None, :]).astype(FP8)
    w2 = jnp.asarray(w2f / s2[:, None, :]).astype(FP8)
    return jnp.asarray(x).astype(jnp.bfloat16), jnp.asarray(logits), w1, w2, jnp.asarray(s1), jnp.asarray(s2)


def _reference(x, logits, w1, w2, s1, s2, real, valid):
    """Per token: v3's select on its shard's rows, then each selected expert's fp8 FFN on the
    quantized row, bf16 result rows, summed over k in order (a zero weight's term dropped)."""
    out = np.zeros(x.shape, np.float32)
    q, qs = quantize_rows(x, jnp.float32, block=0, dtype=jnp.bfloat16)
    for d in range(D):
        rows = slice(d * real, (d + 1) * real)
        w, idx = v3_layer._select(logits[rows], top_k=K, tokens_per_shard=real, renormalize=True,
                                  drop_nan_rows=True, message_rows=None,
                                  valid_tokens=jnp.asarray([valid[d]], jnp.int32))
        for t in range(valid[d]):
            g = d * real + t
            acc = None
            for k in range(K):
                e = int(idx[t, k])
                a2, ms = ffn.expert_ffn_fp8(q[g:g + 1], qs[g:g + 1], w1[e], w2[e], s1[e:e + 1],
                                            activation="silu", w1_bias=None, no_gate=False, block=0,
                                            elementwise=jnp.bfloat16, token_block=0)
                r, _ = ffn.result_rows(a2, ms, w2_scales=s2[e:e + 1], w2_bias=None,
                                       dtype=jnp.bfloat16, elementwise=jnp.bfloat16)
                c = w[t, k]
                term = jnp.where(c != 0, r.astype(jnp.float32) * c, 0.0)
                acc = term if k == 0 else acc + term
            out[g] = np.asarray(acc.astype(jnp.bfloat16).astype(jnp.float32))[0]
    return out


def _run(variant, e, real, hot, short):
    x, logits, w1, w2, s1, s2 = _world(e, real, hot)
    mesh = Mesh(np.asarray(jax.devices()[:D]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    args = [jax.device_put(a, sh) for a in (x, w1, w2, s1, s2, logits)]
    valid = [real - (short if d % 2 else 0) for d in range(D)]
    kw = dict(VARIANTS[variant])
    ws = kw.pop("weight_slots", 0)
    if ws:
        kw["config"] = Config(dense_slots_weight_slots=ws)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        got = jax.jit(lambda *a: DS.dense_slots_moe(
            *a, top_k=K, renormalize=True, mesh=mesh, tile_rows=32,
            shard_index=jnp.arange(D, dtype=jnp.int32).reshape(D, 1),
            valid_tokens=jnp.asarray(valid, jnp.int32), drop_nan_rows=True, **kw))(*args)
        want = _reference(x, logits, w1, w2, s1, s2, real, valid)
    return np.asarray(got.astype(jnp.float32)), want


def _close(got, want):
    """The kernel and the reference are the same ops, but CPU matmuls of different shapes (a
    32-row tile against one row) can round an f32 sum differently, and an fp8 requantization
    can carry that to a few bf16 steps on a handful of values (measured: 15 of 196,608). On
    the device the layer is bit-identical to fused_ep_moe with bf16 result rows
    (test_fep_dense_slots.py). A wrong row moves most of the output (a mutant: ~90%)."""
    off = np.abs(got - want) > 2.0**-7 * np.abs(want) + 1e-5
    return bool(np.isfinite(got).all() and off.mean() < 1e-3)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("e,real,hot,short", [(16, 24, False, 0), (16, 24, True, 0),
                                              (32, 32, False, 5), (64, 24, "cold", 0),
                                              (64, 32, "cold", 5), (64, 24, True, 0)])
def test_matches_reference(variant, e, real, hot, short):
    if e == 64 and "skip" not in variant and variant not in ("merged_unpadded", "fused"):
        pytest.skip("64-expert worlds cover the skip variants and their bases")
    got, want = _run(variant, e, real, hot, short)
    assert _close(got, want), float(np.abs(got - want).max())


def _run_kernel_only(variant, e, real, hot, short):
    x, logits, w1, w2, s1, s2 = _world(e, real, hot)
    mesh = Mesh(np.asarray(jax.devices()[:D]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    args = [jax.device_put(a, sh) for a in (x, w1, w2, s1, s2, logits)]
    valid = [real - (short if d % 2 else 0) for d in range(D)]
    kw = dict(VARIANTS[variant])
    ws = kw.pop("weight_slots", 0)
    if ws:
        kw["config"] = Config(dense_slots_weight_slots=ws)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        got = jax.jit(lambda *a: DS.dense_slots_moe(
            *a, top_k=K, renormalize=True, mesh=mesh, tile_rows=32,
            shard_index=jnp.arange(D, dtype=jnp.int32).reshape(D, 1),
            valid_tokens=jnp.asarray(valid, jnp.int32), drop_nan_rows=True, **kw))(*args)
    return np.asarray(got.astype(jnp.float32))


@pytest.mark.parametrize("base,variant", [("merged_unpadded", "merged_unpadded_skip"),
                                          ("merged_unpadded", "merged_unpadded_w4"),
                                          ("merged_unpadded", "merged_unpadded_skip_w2"),
                                          ("fused", "fused_skip")])
@pytest.mark.parametrize("hot,short", [("cold", 0), ("cold", 5), (True, 0), (False, 0)])
def test_skip_bit_identical(base, variant, hot, short):
    """The skip and the slot count change only which weight DMAs run and when: the same
    program on the same CPU gives every value equal."""
    a = _run_kernel_only(base, 64, 24, hot, short)
    b = _run_kernel_only(variant, 64, 24, hot, short)
    assert np.array_equal(a, b), int((a != b).sum())


def test_skip_slot_mutant_is_caught(monkeypatch):
    """Computing with the old expert-indexed weight slot on the compacted list reads another
    expert's (or a not yet landed) weights: the output must move."""
    a = _run_kernel_only("merged_unpadded", 64, 24, "cold", 0)
    monkeypatch.setattr(DS, "SKIP_MUTANT", "slot")
    DS.build_kernel.cache_clear()
    b = _run_kernel_only("merged_unpadded_skip", 64, 24, "cold", 0)
    monkeypatch.setattr(DS, "SKIP_MUTANT", "")
    DS.build_kernel.cache_clear()
    assert not np.array_equal(a, b)


def test_mutants_are_caught(monkeypatch):
    """Each mutant breaks the output: the destination reads the neighbouring pair's row (pos),
    each expert gathers its rows one token off (onehot), the merged program's arrival rows shift
    by one (merged pos)."""
    cases = [("plain", "pos"), ("fused", "onehot"), ("merged_unpadded", "merged_pos")]
    for variant, mutant in cases:
        if mutant == "pos":
            monkeypatch.setattr(DS, "TABLE_HOOK", lambda t: dict(
                t, pos=jnp.roll(t["pos"].reshape(-1), 1).reshape(t["pos"].shape)))
        elif mutant == "onehot":
            monkeypatch.setattr(DS, "TABLE_HOOK", lambda t: dict(t, mp=jnp.roll(t["mp"], 1, axis=1)))
        else:
            monkeypatch.setattr(DS, "MERGED_POS_SHIFT", 1)
        DS.build_kernel.cache_clear()
        got, want = _run(variant, 16, 24, False, 0)
        assert not _close(got, want), (variant, mutant)
        monkeypatch.setattr(DS, "TABLE_HOOK", None)
        monkeypatch.setattr(DS, "MERGED_POS_SHIFT", 0)
        DS.build_kernel.cache_clear()
