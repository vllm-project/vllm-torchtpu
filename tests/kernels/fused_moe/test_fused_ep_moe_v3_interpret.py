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
"""Fused EP MoE v3 (``kernels/fused_moe/v3``) end to end on CPU.

``fused_ep_moe`` runs over an 8-shard expert-parallel mesh of CPU devices with
every Pallas program in TPU interpret mode (remote DMAs, semaphores and
barriers simulated, the race detector on), against a per-token reference of
the same math: the layer's own top-k selection on each shard's rows, each
selected expert's FP8 FFN on the FP8-quantized row (``ffn.expert_ffn_fp8`` and
``ffn.result_rows`` with bf16 result rows), and the k-ordered weighted sum.

The routing tables, the token exchange, the FFN and the combine are the
kernel's; only the per-expert arithmetic is shared with the reference, so a
row routed to the wrong expert, dropped, or combined with the wrong weight
moves most of the output. The SparseCore row gather has no interpreter, so
these cases run the kernel's Pallas transport (``Config.sorted_rows=False``);
the sorted-rows path is covered on TPU only.

The kernel runs in a subprocess with 8 CPU devices, so the suite needs no
XLA flags of its own. Small shapes keep the interpreter fast: hidden 1024,
intermediate 128, 16 or 32 experts, top-k 4.
"""

import contextlib
import io
import json
import os
import sys

import numpy as np
import pytest

D, H, INTER, K = 8, 1024, 128, 4
TILE_ROWS = 32

# name: (experts, rows per shard, routing, rows dropped on odd shards)
CASES = {
    "spread": (16, 24, "spread", 0),
    "hot": (16, 24, "hot", 0),
    "short_shards": (32, 32, "spread", 5),
}


def _tpu7x():
    from jax.experimental.pallas import tpu as pltpu

    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _world(experts, real, routing, seed=0):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

    rng = np.random.default_rng(seed)
    tokens = real * D
    x = (rng.standard_normal((tokens, H)) / 10).astype(np.float32)
    logits = rng.standard_normal((tokens, experts)).astype(np.float32)
    if routing == "hot":
        # Every token picks experts 0..K-1, all on shard 0: one shard receives
        # every row, the others none.
        logits[:, :K] += 20.0
    w1f = rng.standard_normal((experts, H, 2 * INTER)).astype(np.float32) / 10
    w2f = rng.standard_normal((experts, INTER, H)).astype(np.float32) / 10
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
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v3 import ffn
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer
    from vllm_torchtpu.kernels.fused_moe.v3.rowquant import quantize_rows

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


def _run_case(name):
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe

    experts, real, routing, short = CASES[name]
    x, logits, w1, w2, s1, s2 = _world(experts, real, routing)
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)[:D]
    mesh = Mesh(np.asarray(devices), (MESH_AXIS,))
    sharded = NamedSharding(mesh, P(MESH_AXIS))
    args = [jax.device_put(a, sharded) for a in (x, w1, w2, s1, s2, logits)]
    valid = [real - (short if d % 2 else 0) for d in range(D)]
    config = Config(result_rows="bf16", sorted_rows=False)

    def run(x, w1, w2, s1, s2, logits):
        return fused_ep_moe(
            x,
            w1,
            w2,
            s1,
            s2,
            logits,
            top_k=K,
            renormalize=True,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format="fp8",
            activation="silu",
            config=config,
            experts_per_shard=experts // D,
            shard_index=jnp.arange(D, dtype=jnp.int32).reshape(D, 1),
            valid_tokens=jnp.asarray(valid, jnp.int32),
            drop_nan_rows=True,
        )

    pltpu.reset_tpu_interpret_mode_state()
    with contextlib.redirect_stdout(io.StringIO()):
        got = np.asarray(jax.block_until_ready(jax.jit(run)(*args)))
    races = bool(interpret_pallas_call.races.races_found)
    got = got.astype(np.float32)
    want = _reference(x, logits, w1, w2, s1, s2, real, valid)
    rows = np.concatenate([np.arange(d * real, d * real + valid[d]) for d in range(D)])
    got_valid, want_valid = got[rows], want[rows]
    off = np.abs(got_valid - want_valid) > 2.0**-7 * np.abs(want_valid) + 1e-5
    return {
        "finite": bool(np.isfinite(got_valid).all()),
        "off_fraction": float(off.mean()),
        "max_abs_diff": float(np.abs(got_valid - want_valid).max()),
        "want_rms": float(np.sqrt((want_valid**2).mean())),
        "races_found": races,
    }


def _worker(names):
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    pltpu.get_tpu_info = _tpu7x
    tpu_info.get_tpu_info = _tpu7x
    print("RESULT " + json.dumps({n: _run_case(n) for n in names}))


def _run_worker(names):
    import subprocess

    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={D}",
    )
    proc = subprocess.run(
        [sys.executable, __file__, *names],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def results():
    return _run_worker(list(CASES))


@pytest.mark.parametrize("name", list(CASES))
def test_fused_ep_moe_matches_the_per_token_reference(results, name):
    """CPU matmuls of different shapes (a tile against one row) can round an
    f32 sum differently, and an FP8 requantization can carry that to a few bf16
    steps on a handful of values; a misrouted or dropped row moves most of its
    output, so the bound is one value in a thousand."""
    case = results[name]
    assert case["finite"], case
    assert case["off_fraction"] < 1e-3, case
    assert case["want_rms"] > 0, case


@pytest.mark.parametrize("name", list(CASES))
def test_fused_ep_moe_has_no_dma_races(results, name):
    assert not results[name]["races_found"], results[name]


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker(sys.argv[1:])
