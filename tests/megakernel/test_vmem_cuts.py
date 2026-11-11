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
"""Whole-step VMEM cuts (`Tiles.hbm_io`, `Tiles.io_chunk`, `Tiles.acc_fit`) are bitwise.

Each case builds the whole-step kernel (two `[GDN, GDN, GDN, QSA]` groups, R = 2,
Pallas interpret mode) with the stock tiles and with the cuts, and requires every
output byte-identical: the hidden output, the HC residual, the new KV rows and (served
indexer case) the selection ids and cache rows.

Each case runs in a subprocess with `--xla_allow_excess_precision=false` and FMA off
(`--xla_cpu_max_isa=AVX`): by default XLA:CPU drops some of the kernel's f32 -> bf16 ->
f32 roundings depending on the surrounding graph, which would differ between the two
builds for reasons unrelated to the cuts (see test_moe_rowbucket.py). Mosaic keeps
every rounding.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

CASES = ("plain_b8", "plain_b32", "plain_b48", "served_b8", "served_b8_long")
# io_chunk 8: two or more row chunks at every case (MB >= 16).
CUTS = dict(hbm_io=True, io_chunk=8, acc_fit=True, lean_scratch=True)


def _leaves(x):
    import jax
    import numpy as np

    return [np.asarray(v) for v in jax.tree_util.tree_leaves(x)]


def _plain(B: int, tiles):
    """All outputs of the plain whole-step kernel at `B` rows (T = 4)."""
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from tests.megakernel import test_wholestep as tw
    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

    R, T = 2, 4
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=tiles)
    kv_len = (tw.KV_LEN * (B // len(tw.KV_LEN) + 1))[:B]
    d = wi.random_wholestep_inputs(
        11, tw.G, ex, tw.L, budget=tw.BUDGET, n_pages=tw.PAGES, kv_len=kv_len
    )
    spec = wholestep_spec(tw.G, R, L=tw.L, qsa=dict(budget=tw.BUDGET))
    w_mk, states = wi.pack_wholestep(spec, tw.G, ex, d, tw.L)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(
        mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    hid, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras(d, mesh, R))
    return _leaves(hid) + _leaves(outs)


def _served(tiles, full: bool):
    """Outputs of the indexed kernel (HBM tables; `full`: + sharded indexer, union
    attention, request loop as served) at 2 requests x 4 rows."""
    from tests.megakernel import test_wholestep as tw

    tw._TILES = tiles
    kw = dict(union=True, shard=True, loop=True) if full else {}
    out = tw._run_c(long=True, rank_ids=True, **kw)
    res, _, _, ck, ring, _, iq, ids, d = out
    return _leaves([res, ck, ring, iq, ids, d["_kr_new"], d["_ir_new"]])


def _child(case: str) -> None:
    import numpy as np

    from tests.megakernel import test_wholestep as tw

    base = tw._TILES
    cut = dataclasses.replace(base, **CUTS)
    if case.startswith("plain"):
        B = int(case.split("_b")[1])
        a, b = _plain(B, base), _plain(B, cut)
    else:
        full = case == "served_b8"
        a, b = _served(base, full), _served(cut, full)
    assert len(a) == len(b) and a, (len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape, (i, x.shape, y.shape)
        assert np.isfinite(x.astype(np.float32)).all() or x.dtype.kind in "iu", i
        np.testing.assert_array_equal(x, y, err_msg=f"{case} output {i}")
    print(f"VMEM_CUTS_OK {case} ({len(a)} outputs)", flush=True)


@pytest.mark.parametrize("case", CASES)
def test_vmem_cuts_bitwise(case):
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + " --xla_allow_excess_precision=false --xla_cpu_max_isa=AVX"
        + " --xla_force_host_platform_device_count=8"
    )
    env["JAX_PLATFORMS"] = "cpu"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), case],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    assert r.returncode == 0 and f"VMEM_CUTS_OK {case}" in r.stdout, (
        r.stdout[-3000:] + r.stderr[-6000:]
    )


if __name__ == "__main__":
    _child(sys.argv[1])
