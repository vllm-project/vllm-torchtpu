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
"""moe_tp_decode with hbm_operands=True (TPU_MOE_TP_DECODE_HBM_OPERANDS) is
bitwise the default on real TPU cores (every local device as one TP mesh), at
GLM-5.2 per-rank shapes; a changed shared-expert scale is caught. Also prints
how many whole-weight VMEM staging copies each arm's compiled HLO keeps."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROUTED_SCALE = 2.5


def _staged(hlo: str) -> int:
    return sum(
        1
        for ln in hlo.splitlines()
        if " copy-start(" in ln
        and "S(1)" in ln
        and ("f32[256,1,1," in ln or "f8e4m3fn[" in ln)
    )


# fs: the shared expert's columns per rank (None: 2048 / n). At 128 rows the
# 8-rank slice (fs 256) does not fit the VMEM plan, so that case takes GLM-5.2's
# served TP16 slice (fs 128).
@pytest.mark.parametrize(
    "rows, n_valid, fs",
    [(16, 4, None), (16, 16, None), (32, 12, None), (64, 57, None), (128, 117, 128)],
)
def test_hbm_operands_bitwise(rows, n_valid, fs):
    import test_moe_tp_decode_kernel as base
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel

    if jax.default_backend() != "tpu" or len(jax.devices()) not in (8, 16):
        pytest.skip("needs a TPU host (8 or 16 cores)")
    devices = jax.devices()
    n = len(devices)
    c = dict(h=6144, e=256, i=2048 // n, fs=fs or 2048 // n, topk=8)
    mesh = Mesh(np.asarray(devices), ("d",))
    keys = jax.random.split(jax.random.key(1), n + 3)
    ranks = [base._rank_weights(keys[r], c) for r in range(n)]
    stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*ranks, strict=True)]
    gate = jax.random.normal(keys[-1], (c["e"], c["h"])) / np.sqrt(c["h"])
    bias = jax.random.uniform(keys[-2], (c["e"],), jnp.float32, -0.05, 0.05)
    x = jax.random.normal(keys[-3], (rows, c["h"])).astype(jnp.bfloat16)
    scores = jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)
    specs = (P("d"), P("d"), P(), P(), P("d")) + (P("d"),) * 8

    def make(hbm):
        def body(x, s, b, nv, rank, *w):
            assert kernel.can_use_moe_tp_decode(x, s, b, *w, topk=c["topk"], n_ranks=n)
            return kernel.moe_tp_decode(
                x,
                s,
                b,
                nv[0],
                rank.reshape(()),
                *w,
                routed_scale=ROUTED_SCALE,
                topk=c["topk"],
                n_ranks=n,
                fast=True,
                hbm_operands=hbm,
            )

        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )

    def put(ws):
        ops = (
            jnp.tile(x, (n, 1)),
            jnp.tile(scores, (n, 1)),
            bias,
            jnp.asarray([n_valid], jnp.int32),
            jnp.arange(n, dtype=jnp.int32).reshape(n, 1),
            *ws,
        )
        return [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(ops, specs)]

    args = put(stacked)
    f0, f1 = make(False), make(True)
    want = np.asarray(f0(*args)).view(np.uint16)
    got = np.asarray(f1(*args)).view(np.uint16)
    s0 = _staged(f0.lower(*args).compile().as_text())
    s1 = _staged(f1.lower(*args).compile().as_text())
    print(
        f"RESULT rows {rows} n_valid {n_valid}: bitwise {np.array_equal(got, want)}; "
        f"whole-weight S(1) copy-starts default {s0} hbm {s1}"
    )
    assert np.array_equal(got, want)
    if n_valid:
        bad = list(stacked)
        bad[7] = bad[7] * 1.5  # shared_s2: every output row moves
        ctrl = np.asarray(f1(*put(bad))).view(np.uint16)
        assert not np.array_equal(ctrl, got)
