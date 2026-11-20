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
"""moe_tp_decode_rg is bitwise moe_tp_decode's fast schedule on real TPU cores
(every local device as one mesh), at GLM-5.2 per-rank shapes on the plain-TP
and the expert-range layouts, including an expert routed more rows than the
slab holds (the dense overflow finish)."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROUTED_SCALE = 2.5
ROWS = (96, 128, 160, 192)


@pytest.mark.parametrize(
    "layout, rows, n_valid, hot",
    [
        ("tp", 96, 96, False),
        ("tp", 192, 171, False),
        ("tp", 128, 128, True),
        ("ep", 96, 77, False),
        ("ep", 160, 160, False),
        ("ep", 192, 192, True),
    ],
)
def test_row_gathered_is_bitwise(monkeypatch, layout, rows, n_valid, hot):
    import test_moe_tp_decode_kernel as base
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel, kernel_rg

    if jax.default_backend() != "tpu" or len(jax.devices()) not in (8, 16):
        pytest.skip("needs a TPU host (8 or 16 cores)")
    monkeypatch.setenv("TPU_MOE_TP_DECODE_EXTRA_ROWS", ",".join(map(str, ROWS)))
    monkeypatch.setenv("TPU_MOE_TP_DECODE_VMEM_FRACTION", "1.0")
    devices = jax.devices()
    n = len(devices)
    experts = 256
    ep = layout == "ep"
    c = dict(
        h=6144,
        e=experts // n if ep else experts,
        i=2048 if ep else 2048 // n,
        fs=2048 // n,
        topk=8,
    )
    mesh = Mesh(np.asarray(devices), ("d",))
    keys = jax.random.split(jax.random.key(2), n + 3)
    ranks = [base._rank_weights(keys[r], c) for r in range(n)]
    stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*ranks, strict=True)]
    gate = jax.random.normal(keys[-1], (experts, c["h"])) / np.sqrt(c["h"])
    bias = jax.random.uniform(keys[-2], (experts,), jnp.float32, -0.05, 0.05)
    x = jax.random.normal(keys[-3], (rows, c["h"])).astype(jnp.bfloat16)
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)
    if hot:
        # Every row picks expert 3: more rows than the slab holds.
        scores = scores.at[:, 3].set(1.0)
    starts = jnp.arange(n, dtype=jnp.int32) * c["e"] if ep else jnp.zeros(n, jnp.int32)
    specs = (P("d"), P("d"), P(), P(), P("d")) + (P("d"),) * 8 + (P("d"),)

    def make(fn, **kw):
        def body(x, s, b, nv, rank, *rest):
            *w, e0 = rest
            if fn is kernel_rg.moe_tp_decode_rg:
                assert kernel_rg.can_use_moe_tp_decode_rg(
                    x, s, b, *w, topk=c["topk"], n_ranks=n
                )
            return fn(
                x,
                s,
                b,
                nv[0],
                rank.reshape(()),
                *w,
                e0.reshape(()),
                routed_scale=ROUTED_SCALE,
                topk=c["topk"],
                n_ranks=n,
                **kw,
            )

        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )

    ops = (
        jnp.tile(x, (n, 1)),
        jnp.tile(scores, (n, 1)),
        bias,
        jnp.asarray([n_valid], jnp.int32),
        jnp.arange(n, dtype=jnp.int32).reshape(n, 1),
        *stacked,
        starts,
    )

    def put(ops):
        return [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(ops, specs)]

    args = put(ops)
    want = np.asarray(make(kernel.moe_tp_decode, fast=True)(*args)).view(np.uint16)
    got = np.asarray(make(kernel_rg.moe_tp_decode_rg)(*args)).view(np.uint16)
    print(
        f"RESULT {layout} rows {rows} n_valid {n_valid} hot {hot}: "
        f"bitwise {np.array_equal(got, want)}"
    )
    assert np.array_equal(got, want)
    bad = list(ops)
    bad[8] = bad[8] * 1.5  # s2: every routed output row moves
    rg = make(kernel_rg.moe_tp_decode_rg)
    assert not np.array_equal(np.asarray(rg(*put(bad))).view(np.uint16), got)
