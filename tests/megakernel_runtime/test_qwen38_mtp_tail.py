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
"""The MTP draft tail past the largest bucket runs in bucket-row chunks and
matches the L=1 segment call built at the full rows count.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

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
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.megakernel import kernel  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles  # noqa: E402
from vllm_torchtpu.megakernel.models import qwen38  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_inputs import (  # noqa: E402
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel_runtime.exported_bridge import (  # noqa: E402
    make_typed_step_wrapper,
)
from vllm_torchtpu.megakernel_runtime.qwen38_mtp_tail import build_tail_call  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (  # noqa: E402
    T_VERIFY,
    build_segment_call,
)

R = 2
CHUNK = 4
G = qwen38.TINY
TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


def _run(fn, in_specs, out_specs, args):
    typed = make_typed_step_wrapper("mtp_tail_test", fn, len(in_specs), len(out_specs))
    (res,) = jax.jit(typed)(*args)
    return np.asarray(res, np.float32)


@pytest.mark.parametrize("B", [12, 14])
def test_chunked_tail_matches_full_call(B):
    T = T_VERIFY if B % T_VERIFY == 0 else 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=TILES)
    spec = qwen38.segment_spec(G, R, L=1)
    inp = random_segment_inputs(jax.random.PRNGKey(B), G, ex, 1, gain=0.25)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp = NamedSharding(mesh, P("tp"))
    w_mk, _, _ = pack_segment(spec, G, ex, inp)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names:
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    flat = [jax.device_put(kernel.fold(x), tp) for x in flat]
    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn = jax.device_put(inp.attn.reshape(R * B, -1), tp)
    args = (hidden, attn, *flat)

    full = build_segment_call(mesh, G, 1, B, None, interpret=True, tiles=TILES)
    chunked = build_tail_call(mesh, G, B, None, CHUNK, interpret=True, tiles=TILES)
    assert chunked[1:] == full[1:]  # same specs, so the torch op signature is unchanged
    want = _run(*full[:3], args)
    got = _run(*chunked[:3], args)
    assert got.shape == want.shape == (R * B, hidden.shape[-1])
    scale = float(np.abs(want).mean())
    assert float(np.abs(got - want).max()) < 1e-2 * scale


def test_small_counts_build_at_their_own_rows():
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    fn, in_specs, out_specs, donate = build_tail_call(
        mesh, G, CHUNK, None, CHUNK, interpret=True, tiles=TILES
    )
    assert donate == () and len(out_specs) == 1 and len(in_specs) > 2
