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
"""kernels/collectives/pair_lse_merge on 8 CPU devices (pairs (0, 1), (2, 3),
...) in interpret mode with the race detector, and its collective-permute
twin, against cp_lse_ag_out_rs's math (a subprocess: the device count is fixed
when JAX starts)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

N_RANKS = 8
# (tokens, 2 * heads): the c4 draft decode buckets; 7 rows (not a tile).
CASES = ((8, 16), (24, 16), (7, 16))
# exchange_block (rows, 2 * heads): the c4 b1 and b4 draft blocks (8 and 32
# rows, 1 and 4 requests of 7 rows plus padding).
BLOCK_CASES = ((8, 16), (32, 16))
KV_SCALE = 0.5  # the draft fp8 cache's K/V scale in exchange_block's test


def _stored(x):
    """x as the fp8 draft cache holds it, back in bf16."""
    import jax.numpy as jnp

    x = (x.astype(jnp.float32) / KV_SCALE).astype(jnp.float8_e4m3fn)
    return (x.astype(jnp.float32) * KV_SCALE).astype(jnp.bfloat16)


D = 128


def _inputs(t, two_h, seed):
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((N_RANKS, t, two_h, D)).astype(np.float32)
    lse = (rng.standard_normal((N_RANKS, t, two_h)) * 3).astype(np.float32)
    # A token with no context on one rank (LSE -inf, its partial NaN).
    lse[0, 0, :] = -np.inf
    out[0, 0, :, :] = np.nan
    return out, lse


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives import pair_lse_merge as k

    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    peers = np.arange(N_RANKS) ^ 1
    meta = jnp.asarray(np.stack([peers, np.arange(N_RANKS) % 2], 1), jnp.int32)
    perm = [(r, int(peers[r])) for r in range(N_RANKS)]
    spec = P("d")

    def sharded(fn):
        return jax.jit(
            jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=(spec,) * 3,
                out_specs=(spec, spec),
                check_vma=False,
            )
        )

    kernel = sharded(lambda o, s, m: k.exchange(o[0], s[0], m[0]))
    xla = sharded(lambda o, s, m: k.exchange_xla(o[0], s[0], m[0, 1], "d", perm))
    result = {}
    for t, two_h in CASES:
        out, lse = _inputs(t, two_h, t)
        out_b = jnp.asarray(out, jnp.bfloat16)
        want = [
            k.reference([out_b[r & ~1], out_b[r | 1]], [lse[r & ~1], lse[r | 1]], r % 2)
            for r in range(N_RANKS)
        ]
        for name, run in (("kernel", kernel), ("xla", xla)):
            pltpu.reset_tpu_interpret_mode_state()
            o, ol = run(out_b, jnp.asarray(lse), meta)
            o = np.asarray(o, np.float32).reshape(N_RANKS, t, two_h // 2, D)
            ol = np.asarray(ol).reshape(N_RANKS, t, two_h // 2)
            ulp = max(
                float(
                    np.nanmax(
                        np.abs(o[r] - np.asarray(w[0], np.float32))
                        / np.maximum(np.abs(np.asarray(w[0], np.float32)), 1e-3)
                    )
                )
                for r, w in enumerate(want)
            )
            result[f"{name}-{t}"] = dict(
                rel=ulp,
                lse=bool(
                    all(
                        np.allclose(ol[r], np.asarray(w[1]), rtol=1e-6, atol=1e-6)
                        for r, w in enumerate(want)
                    )
                ),
                finite=bool(np.isfinite(o).all()),
                races=bool(getattr(interpret_pallas_call.races, "races_found", False)),
            )
    # The block's own attention and the final merge folded in
    # (`exchange_block`, on the attention kernel's own LSE layout and the
    # block's K/V before the fp8 cache round trip) vs the unfused path:
    # token-order LSEs, the pair merge, the round trip,
    # cp_draft_attention.attend_block (uniform blocks of 7) and merge.
    from vllm_torchtpu.layers.adapter.cp_draft_attention import attend_block, merge

    for t, two_h in BLOCK_CASES:
        heads, q_block = two_h // 2, 7
        reqs = t // q_block
        valid = reqs * q_block
        rng = np.random.default_rng(100 + t)
        out, _ = _inputs(t, two_h, t)
        out[:, valid:] = np.nan  # rows no request owns: undefined
        out_b = jnp.asarray(out, jnp.bfloat16)
        lse_native = (
            rng.standard_normal((N_RANKS, reqs, 1, q_block * two_h)) * 3
        ).astype(np.float32)
        lse_native[0, 0, 0, :two_h] = -np.inf  # a token with no context
        lse_t = np.full((N_RANKS, t, two_h), -np.inf, np.float32)
        lse_t[:, :valid] = lse_native.reshape(N_RANKS, valid, two_h)
        out_t = jnp.where(jnp.arange(t)[None, :, None, None] < valid, out_b, 0)
        q = jnp.asarray(rng.standard_normal((N_RANKS, t, heads, D)), jnp.bfloat16)
        kv = jnp.asarray(rng.standard_normal((2, N_RANKS, t, 1, D)), jnp.bfloat16)

        for causal in (True, False):
            block = jax.jit(
                jax.shard_map(
                    lambda o, s, m, q_, kk, vv, c=causal, qb=q_block: k.exchange_block(
                        o[0],
                        s[0],
                        m[0],
                        q_[0],
                        kk[0],
                        vv[0],
                        q_block=qb,
                        sm_scale=0.125,
                        causal=c,
                        k_scale=KV_SCALE,
                        v_scale=KV_SCALE,
                        cache_dtype=jnp.float8_e4m3fn,
                    )[None],
                    mesh=mesh,
                    in_specs=(spec,) * 6,
                    out_specs=spec,
                    check_vma=False,
                )
            )
            pltpu.reset_tpu_interpret_mode_state()
            got = block(out_b, jnp.asarray(lse_native), meta, q, kv[0], kv[1])
            got = np.asarray(got, np.float32).reshape(N_RANKS, t, heads, D)
            rel = 0.0
            seq_lens = jnp.zeros((reqs,), jnp.int32)
            for r in range(N_RANKS):
                lo, hi = r & ~1, r | 1
                ctx, ctx_lse = k.reference(
                    [out_t[lo], out_t[hi]], [lse_t[lo], lse_t[hi]], r % 2
                )
                ob, lb = attend_block(
                    q[r],
                    _stored(kv[0, r]),
                    _stored(kv[1, r]),
                    seq_lens,
                    None,
                    max_query_len=q_block,
                    sm_scale=0.125,
                    causal=causal,
                    uniform_blocks=True,
                )
                want = np.asarray(merge(ctx.astype(ob.dtype), ctx_lse, ob, lb)[0])
                want = want.astype(np.float32)
                err = np.abs(got[r] - want) / np.maximum(np.abs(want), 1e-2)
                rel = max(rel, float(np.nanmax(err)))
            result[f"block-{t}-{int(causal)}"] = dict(
                rel=rel,
                finite=bool(np.isfinite(got).all()),
                races=bool(getattr(interpret_pallas_call.races, "races_found", False)),
            )
    # Fault control: the other block's heads must miss the bar.
    t, two_h = CASES[1]
    out, lse = _inputs(t, two_h, t)
    out_b = jnp.asarray(out, jnp.bfloat16)
    pltpu.reset_tpu_interpret_mode_state()
    o, _ = kernel(out_b, jnp.asarray(lse), meta.at[:, 1].set(1 - meta[:, 1]))
    o = np.asarray(o, np.float32).reshape(N_RANKS, t, two_h // 2, D)
    w = k.reference([out_b[2], out_b[3]], [lse[2], lse[3]], 0)[0]
    result["fault_caught"] = bool(not np.allclose(o[2], np.asarray(w, np.float32)))
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=540
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("causal", [1, 0])
@pytest.mark.parametrize("t", [t for t, _ in BLOCK_CASES])
def test_merge_block_on_8_devices(on_8_devices, t, causal):
    case = on_8_devices[f"block-{t}-{causal}"]
    # Within two bf16 roundings of the unfused ops.
    assert case["rel"] <= 2**-6 and case["finite"], case
    assert not case["races"], case


@pytest.mark.parametrize("name", ["kernel", "xla"])
@pytest.mark.parametrize("t", [t for t, _ in CASES])
def test_merge_on_8_devices(on_8_devices, name, t):
    case = on_8_devices[f"{name}-{t}"]
    # Within one bf16 rounding of the reference.
    assert case["rel"] <= 2**-7 and case["lse"] and case["finite"], case
    assert not case["races"], case


def test_fault_is_caught(on_8_devices):
    assert on_8_devices["fault_caught"]


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
