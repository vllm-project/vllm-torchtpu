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
"""The dense-slots layer (models/qwen3_5/kernels/dense_slots.py) on CPU, every program
in Pallas TPU interpret mode
(remote DMAs, semaphores and barriers simulated) over 8 fake devices, against a
per-token reference
of the same math: v3's select, fused EP v3's per-row FFN (ffn.expert_ffn_fp8 /
result_rows with
bf16 result rows) and its k-ordered weighted sum.

Small shapes: hidden 1024 (whole 8-sublane staging tiles), intermediate 128, 16, 32 or
64 experts, top-k 4.

The checks need 8 CPU devices, and the host device count (XLA_FLAGS) must be set before
JAX starts, so the tests run this file as a script in a subprocess, once per session,
for the cases pytest selected, and each test reports its case. One case per variant
and the two mutant checks run in regular CI; the rest of the grid is marked nightly.
"""

import json
import os
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v3 import Config, ffn
from vllm_torchtpu.kernels.fused_moe.v3 import device as DEV
from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8, quantize_rows
from vllm_torchtpu.models.qwen3_5.kernels import dense_slots as DS

D, H, INTER, K = 8, 1024, 128, 4

VARIANTS = {
    "plain": {},
    "fused": dict(fused=True),
    "pallas_tables": dict(fused=True, pallas_tables=True),
    "merged": dict(fused=True, merged=True),
    "merged_unpadded": dict(fused=True, merged=True, unpadded=True),
    # The unrouted-expert skip (Config.dense_slots_skip_unrouted) and other weight slot
    # counts.
    "plain_skip": dict(skip_unrouted=True),
    "fused_skip": dict(fused=True, skip_unrouted=True),
    "merged_unpadded_skip": dict(
        fused=True, merged=True, unpadded=True, skip_unrouted=True
    ),
    "merged_unpadded_w4": dict(fused=True, merged=True, unpadded=True, weight_slots=4),
    "merged_unpadded_skip_w2": dict(
        fused=True, merged=True, unpadded=True, skip_unrouted=True, weight_slots=2
    ),
}


def _cpu_budget(monkeypatch):
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
        # prefetched expert without rows, cold runs between live ones, and live ones
        # after).
        g = e // D
        cold = [d * g + c for d in range(D) for c in (1, 3, 5, 6) if c < g]
        logits[:, cold] -= 30.0
    elif hot:
        logits[:, :K] += 20.0  # every token -> experts 0..K-1, all on shard 0
    w1f = rng.standard_normal((e, H, 2 * INTER)).astype(np.float32) / 10
    w2f = rng.standard_normal((e, INTER, H)).astype(np.float32) / 10
    s1 = np.abs(w1f).max(1) / 448.0
    s2 = np.abs(w2f).max(1) / 448.0
    w1 = jnp.asarray(w1f / s1[:, None, :]).astype(FP8)
    w2 = jnp.asarray(w2f / s2[:, None, :]).astype(FP8)
    return (
        jnp.asarray(x).astype(jnp.bfloat16),
        jnp.asarray(logits),
        w1,
        w2,
        jnp.asarray(s1),
        jnp.asarray(s2),
    )


def _reference(x, logits, w1, w2, s1, s2, real, valid):
    """Per token: v3's select on its shard's rows, then each selected expert's fp8 FFN
    on the
    quantized row, bf16 result rows, summed over k in order (a zero weight's term
    dropped)."""
    out = np.zeros(x.shape, np.float32)
    q, qs = quantize_rows(x, jnp.float32, block=0, dtype=jnp.bfloat16)
    for d in range(D):
        rows = slice(d * real, (d + 1) * real)
        w, idx = v3_layer._select(
            logits[rows],
            top_k=K,
            tokens_per_shard=real,
            renormalize=True,
            drop_nan_rows=True,
            message_rows=None,
            valid_tokens=jnp.asarray([valid[d]], jnp.int32),
        )
        for t in range(valid[d]):
            g = d * real + t
            acc = None
            for k in range(K):
                e = int(idx[t, k])
                a2, ms = ffn.expert_ffn_fp8(
                    q[g : g + 1],
                    qs[g : g + 1],
                    w1[e],
                    w2[e],
                    s1[e : e + 1],
                    activation="silu",
                    w1_bias=None,
                    no_gate=False,
                    block=0,
                    elementwise=jnp.bfloat16,
                    token_block=0,
                )
                r, _ = ffn.result_rows(
                    a2,
                    ms,
                    w2_scales=s2[e : e + 1],
                    w2_bias=None,
                    dtype=jnp.bfloat16,
                    elementwise=jnp.bfloat16,
                )
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
        got = jax.jit(
            lambda *a: DS.dense_slots_moe(
                *a,
                top_k=K,
                renormalize=True,
                mesh=mesh,
                tile_rows=32,
                shard_index=jnp.arange(D, dtype=jnp.int32).reshape(D, 1),
                valid_tokens=jnp.asarray(valid, jnp.int32),
                drop_nan_rows=True,
                **kw,
            )
        )(*args)
        want = _reference(x, logits, w1, w2, s1, s2, real, valid)
    return np.asarray(got.astype(jnp.float32)), want


def _close(got, want):
    """The kernel and the reference are the same ops, but CPU matmuls of different
    shapes (a
    32-row tile against one row) can round an f32 sum differently, and an fp8
    requantization
    can carry that to a few bf16 steps on a handful of values (measured: 15 of
    196,608). On
    the device the layer is bit-identical to fused_ep_moe with bf16 result rows
    (test_fep_dense_slots.py). A wrong row moves most of the output (a mutant: ~90%)."""
    off = np.abs(got - want) > 2.0**-7 * np.abs(want) + 1e-5
    return bool(np.isfinite(got).all() and off.mean() < 1e-3)


def _check_matches_reference(variant, e, real, hot, short):
    if (
        e == 64
        and "skip" not in variant
        and variant not in ("merged_unpadded", "fused")
    ):
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
        got = jax.jit(
            lambda *a: DS.dense_slots_moe(
                *a,
                top_k=K,
                renormalize=True,
                mesh=mesh,
                tile_rows=32,
                shard_index=jnp.arange(D, dtype=jnp.int32).reshape(D, 1),
                valid_tokens=jnp.asarray(valid, jnp.int32),
                drop_nan_rows=True,
                **kw,
            )
        )(*args)
    return np.asarray(got.astype(jnp.float32))


def _check_skip_bit_identical(base, variant, hot, short):
    """The skip and the slot count change only which weight DMAs run and when: the same
    program on the same CPU gives every value equal."""
    a = _run_kernel_only(base, 64, 24, hot, short)
    b = _run_kernel_only(variant, 64, 24, hot, short)
    assert np.array_equal(a, b), int((a != b).sum())


def _check_skip_slot_mutant_is_caught(monkeypatch):
    """Computing with the old expert-indexed weight slot on the compacted list reads
    another
    expert's (or a not yet landed) weights: the output must move."""
    a = _run_kernel_only("merged_unpadded", 64, 24, "cold", 0)
    monkeypatch.setattr(DS, "SKIP_MUTANT", "slot")
    DS.build_kernel.cache_clear()
    b = _run_kernel_only("merged_unpadded_skip", 64, 24, "cold", 0)
    monkeypatch.setattr(DS, "SKIP_MUTANT", "")
    DS.build_kernel.cache_clear()
    assert not np.array_equal(a, b)


def _check_mutants_are_caught(monkeypatch):
    """Each mutant breaks the output: the destination reads the neighbouring pair's row
    (pos),
    each expert gathers its rows one token off (onehot), the merged program's arrival
    rows shift
    by one (merged pos)."""
    cases = [("plain", "pos"), ("fused", "onehot"), ("merged_unpadded", "merged_pos")]
    for variant, mutant in cases:
        if mutant == "pos":
            monkeypatch.setattr(
                DS,
                "TABLE_HOOK",
                lambda t: dict(
                    t, pos=jnp.roll(t["pos"].reshape(-1), 1).reshape(t["pos"].shape)
                ),
            )
        elif mutant == "onehot":
            monkeypatch.setattr(
                DS, "TABLE_HOOK", lambda t: dict(t, mp=jnp.roll(t["mp"], 1, axis=1))
            )
        else:
            monkeypatch.setattr(DS, "MERGED_POS_SHIFT", 1)
        DS.build_kernel.cache_clear()
        got, want = _run(variant, 16, 24, False, 0)
        assert not _close(got, want), (variant, mutant)
        monkeypatch.setattr(DS, "TABLE_HOOK", None)
        monkeypatch.setattr(DS, "MERGED_POS_SHIFT", 0)
        DS.build_kernel.cache_clear()


WORLDS = [
    (16, 24, False, 0),
    (16, 24, True, 0),
    (32, 32, False, 5),
    (64, 24, "cold", 0),
    (64, 32, "cold", 5),
    (64, 24, True, 0),
]
# Regular CI runs each variant on one world: a skip variant where experts go unrouted,
# the others on the smallest world. The rest of the grid runs nightly.
REGULAR_WORLD = {
    v: (64, 24, "cold", 0) if "skip" in v else (16, 24, False, 0) for v in VARIANTS
}
SKIP_PAIRS = [
    ("merged_unpadded", "merged_unpadded_skip"),
    ("merged_unpadded", "merged_unpadded_w4"),
    ("merged_unpadded", "merged_unpadded_skip_w2"),
    ("fused", "fused_skip"),
]
SKIP_WORLDS = [("cold", 0), ("cold", 5), (True, 0), (False, 0)]


def _case_id(name, *params):
    return name + "[" + "-".join(map(str, params)) + "]"


def _cases():
    """Every case of the grid: id -> the check it runs with a MonkeyPatch."""
    cases = {}
    for variant in VARIANTS:
        for world in WORLDS:
            cases[_case_id("matches_reference", variant, *world)] = (
                lambda mp, v=variant, w=world: _check_matches_reference(v, *w)
            )
    for base, variant in SKIP_PAIRS:
        for hot, short in SKIP_WORLDS:
            cases[_case_id("skip_bit_identical", base, variant, hot, short)] = (
                lambda mp, b=base, v=variant, h=hot, s=short: _check_skip_bit_identical(
                    b, v, h, s
                )
            )
    cases["skip_slot_mutant_is_caught"] = _check_skip_slot_mutant_is_caught
    cases["mutants_are_caught"] = _check_mutants_are_caught
    return cases


def _worker(case_ids):
    devices = jax.devices()
    # Fail, not skip, without the 8 devices: the tests set them up themselves.
    assert len(devices) >= D and devices[0].platform == "cpu", devices
    cases = _cases()
    results = {}
    for case in case_ids:
        start = time.monotonic()
        with pytest.MonkeyPatch.context() as mp:
            _cpu_budget(mp)
            try:
                cases[case](mp)
                outcome, message = "passed", ""
            except pytest.skip.Exception as e:
                outcome, message = "skipped", str(e)
            except Exception:
                outcome, message = "failed", traceback.format_exc()[-3000:]
        results[case] = dict(
            outcome=outcome, message=message, seconds=time.monotonic() - start
        )
    print("RESULT " + json.dumps(results))


_TESTS = {
    "test_matches_reference": "matches_reference",
    "test_skip_bit_identical": "skip_bit_identical",
}


def _item_case(item):
    name = item.originalname
    if name in _TESTS:
        return _case_id(_TESTS[name], *item.callspec.params.values())
    return name[len("test_") :]


@pytest.fixture(scope="module")
def on_8_devices(request):
    """Runs the cases pytest selected from this file in one subprocess with 8 CPU
    devices, and returns each case's outcome."""
    case_ids = [
        _item_case(item)
        for item in request.session.items
        if getattr(item, "module", None) is request.module
    ]
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={D}",
    )
    proc = subprocess.run(
        [sys.executable, __file__, json.dumps(case_ids)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300 + 300 * len(case_ids),
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


def _report(results, case, may_skip=False):
    got = results[case]
    if got["outcome"] == "skipped" and may_skip:
        pytest.skip(got["message"])
    assert got["outcome"] == "passed", got["message"]


def _matches_params():
    for variant in VARIANTS:
        for world in WORLDS:
            regular = REGULAR_WORLD[variant] == world
            marks = () if regular else (pytest.mark.nightly,)
            yield pytest.param(variant, *world, marks=marks)


@pytest.mark.parametrize("variant,e,real,hot,short", list(_matches_params()))
def test_matches_reference(on_8_devices, variant, e, real, hot, short):
    # The 64-expert worlds run only the skip variants and their bases (the check skips
    # the rest itself).
    may_skip = (
        e == 64
        and "skip" not in variant
        and variant not in ("merged_unpadded", "fused")
    )
    _report(
        on_8_devices,
        _case_id("matches_reference", variant, e, real, hot, short),
        may_skip,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("base,variant", SKIP_PAIRS)
@pytest.mark.parametrize("hot,short", SKIP_WORLDS)
def test_skip_bit_identical(on_8_devices, base, variant, hot, short):
    _report(on_8_devices, _case_id("skip_bit_identical", base, variant, hot, short))


def test_skip_slot_mutant_is_caught(on_8_devices):
    _report(on_8_devices, "skip_slot_mutant_is_caught")


def test_mutants_are_caught(on_8_devices):
    _report(on_8_devices, "mutants_are_caught")


if __name__ == "__main__":
    _worker(json.loads(sys.argv[1]))
