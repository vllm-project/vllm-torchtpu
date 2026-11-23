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
"""The fused DCP-2 decode attention call (`msa_dcp_fused`,
TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS) against the unfused
`msa_dcp.attend_and_merge`, in Pallas interpret mode on a 2-device CPU
platform (a subprocess: the device count is fixed at JAX start-up): bit for
bit at 8/16/32/128 decode rows (8 verify rows per request, contexts past the
top-k so the selection is exercised, a short request, padding rows), for fp8
caches (c4 / c16) and 4-bit caches; a positive control (the partner's
candidates matter); the eligibility rule; and deviceless tpu7x compiles at
the served shapes."""

import os
import subprocess
import sys

import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_fused

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dcp_fused_spmd.py")


def _run(*argv, devices=2, timeout=3000):
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        PYTHONPATH=f"{ROOT}:{ROOT}/src",
    )
    res = subprocess.run(
        [sys.executable, _WORKER, *argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-4000:]


@pytest.mark.parametrize(
    "kind,ctx,pad,control",
    [
        ("fp8", "5000", 0, "1"),  # T 8 (c4: one request)
        ("fp8", "6000,2600", 0, "0"),  # T 16 (c16: two requests)
        ("fp8", "5000,3000,700", 8, "0"),  # T 32: a short request, a padding tile
        ("fp4", "6000,2600", 0, "1"),  # 4-bit caches (c48 layout), T 16
        ("fp4", "5000,3000,700", 8, "0"),
    ],
)
def test_fused_matches_unfused_bitwise(kind, ctx, pad, control):
    _run(kind, ctx, str(pad), control)


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_fused_matches_unfused_bitwise_max_free(kind):
    """With the max-free softmax (TPU_MINIMAX_M3_MSA_MAX_FREE) the fused call
    still matches the unfused one bit for bit (both run the same body)."""
    _run(kind, "5000,3000,700", "8", "0", "maxfree")


def test_fused_matches_unfused_bitwise_shared_scale_page():
    """TPU_MINIMAX_M3_KV4_SHARED_SCALES: with the layer's K/V scales at rows
    6 / 7 of a scale page shared with three layers, the fused call matches
    the unfused one over the layer's own page bit for bit."""
    _run("fp4", "5000,3000,700", "8", "0", "shared")


def test_fused_matches_unfused_bitwise_128_rows():
    """16 requests (the c16 128-row bucket): 16 tiles, the work list's
    bitonic unions (the unfused call's XLA work list)."""
    ctx = ",".join(str(1500 + 300 * i) for i in range(16))
    _run("fp8", ctx, "0", "0")


def test_eligible():
    e = msa_dcp_fused.eligible
    assert e(8, 32, dcp=2, kv_heads=1)
    assert e(32, 32, dcp=2, kv_heads=1)
    assert not e(40, 32, dcp=2, kv_heads=1)  # over the switch
    assert not e(12, 32, dcp=2, kv_heads=1)  # not whole 8-row tiles
    assert not e(8, 0, dcp=2, kv_heads=1)  # off
    assert not e(8, 32, dcp=4, kv_heads=1)
    assert not e(8, 32, dcp=8, kv_heads=4)
    assert not e(256, 512, dcp=2, kv_heads=1)  # past MAX_TOKENS
    assert e(128, 512, dcp=2, kv_heads=1)


@pytest.mark.parametrize("kind", ["fp8", "fp4"])
@pytest.mark.parametrize("tokens", [8, 16, 32, 128])
def test_compiles_for_tpu7x(kind, tokens):
    """Mosaic lowers the fused call at the served shapes (TP 8: 8 query heads
    per rank, top-16, 128-token blocks) within its 32 MiB VMEM limit."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    mesh = Mesh(np.array(topo.devices[:2]), ("dcp",))
    hl, d, topk, pages, seqs, pps = 8, 128, 16, 4096, 32, 1024
    sh, rep = P("dcp"), P()

    def s(shape, dt, spec):
        return jax.ShapeDtypeStruct(shape, dt, sharding=NamedSharding(mesh, spec))

    fp4 = kind == "fp4"
    kv = (
        s((2 * pages, 16, 4, 2 * d), jnp.uint8, sh)
        if fp4
        else s((2 * pages, 32, 4, 2 * d), jnp.float8_e4m3fn, sh)
    )
    args = [
        s((2,), jnp.int32, sh),
        s((2 * hl, tokens, d), jnp.bfloat16, sh),
        kv,
        s((2 * tokens, 2 * topk), jnp.float32, sh),
        s((seqs,), jnp.int32, rep),
        s((seqs * pps,), jnp.int32, rep),
        s((seqs + 1,), jnp.int32, rep),
    ]
    specs = (sh,) * 4 + (rep,) * 3
    if fp4:
        args += [s((2 * pages, 8, 128), jnp.float32, sh), s((d, d), jnp.bfloat16, rep)]
        specs += (sh, rep)
    f = jax.jit(
        jax.shard_map(
            lambda *a: msa_dcp_fused.fused_attend_and_merge(
                *a, axis="dcp", dcp=2, sm_scale=d**-0.5, topk=topk
            ),
            mesh=mesh,
            in_specs=specs,
            out_specs=sh,
            check_vma=False,
        )
    )
    assert "minimax_m3_msa_dcp_fused_decode" in f.lower(*args).compile().as_text()
