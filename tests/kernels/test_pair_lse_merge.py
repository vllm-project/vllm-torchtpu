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


@pytest.mark.parametrize("name", ["kernel", "xla"])
@pytest.mark.parametrize("t", [t for t, _ in CASES])
def test_merge_on_8_devices(on_8_devices, name, t):
    case = on_8_devices[f"{name}-{t}"]
    # Within one bf16 rounding of the reference.
    assert case["rel"] <= 2**-7 and case["lse"] and case["finite"], case
    assert not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()


def test_fault_is_caught(on_8_devices):
    assert on_8_devices["fault_caught"]
