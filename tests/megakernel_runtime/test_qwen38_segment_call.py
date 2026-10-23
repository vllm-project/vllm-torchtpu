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
"""The composed Qwen3.8 segment call (pool gather -> kernel -> scatter) on CPU.

Real per-rank GDN shapes (2 k-heads, 6 v-heads, 128x128 state) at R=2 and the
real pool byte layout. The residual must match `segment_reference` fed with the
pool's states, and every verify token's GDN checkpoint must land in its block.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import (  # noqa: E402
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (  # noqa: E402
    T_VERIFY,
    build_segment_call,
)

_spec = importlib.util.spec_from_file_location(
    "gdn_state_test", os.path.join(os.path.dirname(__file__), "test_qwen38_gdn_state.py")
)
_gst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gst)

import pytest  # noqa: E402

R, NS = 2, 2
B = NS * T_VERIFY
NMB = 1 + 3 * NS * T_VERIFY  # block 0 = null; up to 3 GDN layers
POOL_SHAPE = (NMB * 4, 128, 1, 4, 256)
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


def _blocks(NL):
    """ckpt [NL, NS, T] distinct per (layer, request, token); reads at offset 2."""
    ck = 1 + np.arange(NL * NS * T_VERIFY, dtype=np.int32).reshape(NL, NS, T_VERIFY)
    off = np.zeros((NL, NMB), np.int32)
    for l in range(NL):
        for s in range(NS):
            off[l, ck[l, s, 0]] = 2
    return ck, off


@pytest.mark.parametrize("L,head,pool_gdn", [(4, False, False), (4, False, True), (2, True, True)])
def test_segment_call_matches_reference_and_writes_checkpoints(L, head, pool_gdn):
    NL = L if head else L - 1
    rng = np.random.default_rng(0)
    ck, off = _blocks(NL)
    raws = []
    for _ in range(R):
        raw = np.zeros(POOL_SHAPE, np.uint8)
        for m in range(1, NMB):
            _gst.ref_write(
                raw,
                m,
                rng.standard_normal((6, 128, 128)).astype(np.float32) * 0.05,
                rng.standard_normal((3, 1280)).astype(np.float32) * 0.05,
            )
        raws.append(raw)
    ex = ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, interpret=True, tiles=TILES)
    spec = qwen38.segment_spec(G, R, L=L, head=head)
    inp = random_segment_inputs(jax.random.PRNGKey(5), G, ex, L, gain=0.25, head=head)

    # The reference reads the same initial states the kernel gathers.
    conv0 = np.zeros((R, NL, NS, 3, 1280), np.float32)
    rec0 = np.zeros((R, NL, NS, 6, 128, 128), np.float32)
    for r in range(R):
        for l in range(NL):
            for s in range(NS):
                ssm, conv = _gst.ref_read(raws[r], ck[l, s, 2])
                rec0[r, l, s], conv0[r, l, s] = ssm, conv
    inp.st_l = {"conv": jnp.asarray(conv0), "rec": jnp.asarray(rec0).astype(jnp.bfloat16)}
    ref, ref_states = segment_reference(
        G, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L, head=head
    )

    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    w_mk, _, _ = pack_segment(spec, G, ex, inp, head=head)
    spec = qwen38.segment_spec(G, R, L=L, head=head, pool_gdn=pool_gdn)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names:
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    flat = [jax.device_put(kernel.fold(x), tp) for x in flat]
    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn = jax.device_put(inp.attn.reshape(R * B, -1), tp)
    pool = jax.device_put(
        jnp.concatenate([_gst._to_pool(r) for r in raws], axis=0), tp
    )
    meta = []
    for l in range(NL):
        meta += [
            jnp.asarray(ck[l, :, 0]),  # state ids
            jnp.asarray(off[l]),
            jnp.asarray(ck[l]),
            jnp.arange(NS + 1, dtype=jnp.int32) * T_VERIFY,
            jnp.full((NS,), 100, jnp.int32),
        ]
    meta = [jax.device_put(m, rep) for m in meta]

    layout = gs.Qwen38GdnPoolLayout(pool_shape=POOL_SHAPE, pool_dtype="float8_e4m3fn")
    fn, in_specs, out_specs, donate = build_segment_call(
        mesh, G, L, B, layout, interpret=True, tiles=TILES, head=head, pool_gdn=pool_gdn
    )
    lead = (hidden,) if head else (hidden, attn)
    assert len(in_specs) == len(lead) + len(flat) + 1 + len(meta)
    assert donate == (len(lead) + len(flat),)
    # Through the fixed-arity wrapper the TPU bridge needs (no *args signature).
    from vllm_torchtpu.megakernel_runtime.exported_bridge import make_typed_step_wrapper

    typed = make_typed_step_wrapper("seg_call_test", fn, len(in_specs), len(out_specs))
    res, new_pool = jax.jit(typed)(*lead, *flat, pool, *meta)

    res0 = np.asarray(res.reshape(R, B, -1)[0], np.float32)
    refn = np.asarray(ref, np.float32)
    scale = float(np.abs(refn).mean())
    assert float(np.std(refn - np.asarray(inp.x, np.float32))) > 0.05
    assert float(np.abs(res0 - refn).mean()) < 1e-2 * scale

    new_raw = np.asarray(jax.lax.bitcast_convert_type(new_pool, jnp.uint8)).reshape(
        R, *POOL_SHAPE
    )
    for l in range(NL):
        rec_ck = np.asarray(ref_states[l]["rec_ckpt"], np.float32).reshape(
            R, NS, T_VERIFY, 6, 128, 128
        )
        for r in range(R):
            for s in range(NS):
                for t in range(T_VERIFY):
                    ssm, _ = _gst.ref_read(new_raw[r], ck[l, s, t])
                    np.testing.assert_allclose(
                        ssm, rec_ck[r, s, t], atol=2e-2, rtol=2e-2,
                        err_msg=f"layer {l} rank {r} req {s} token {t}",
                    )
