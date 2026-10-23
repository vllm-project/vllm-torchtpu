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
"""The manual-DMA checkpoint scatter writes the same pool bytes as the region adapter.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast  # noqa: E402

NL, NS, T = 2, 3, 4
NMB = 1 + NL * NS * T
SHAPE = (NMB * 4, 128, 1, 4, 256)
LAYOUT = gs.Qwen38GdnPoolLayout(pool_shape=SHAPE, pool_dtype="float8_e4m3fn")


@pytest.mark.parametrize("pad_requests,ring", [(False, 4), (True, 4), (False, 2), (False, 8)])
def test_fast_scatter_matches_region_scatter(pad_requests: bool, ring: int):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)
    wb = 1 + np.arange(NL * NS * T, dtype=np.int32).reshape(NL, NS, T)
    if pad_requests:
        wb[:, -1] = 0  # a padded request writes the null block only
    wb = jnp.asarray(wb)
    ref = gs.scatter_gdn_ckpts(pool, conv, rec, wb, layout=LAYOUT, interpret=True)
    got = scatter_gdn_ckpts_fast(pool, conv, rec, wb, layout=LAYOUT, interpret=True, ring=ring)
    a = np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8))
    b = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint8))
    # Manager blocks other than the written ones must be untouched by both.
    live = sorted(set(np.asarray(wb).reshape(-1).tolist()) - {0})
    for m in range(NMB):
        blk = slice(4 * m, 4 * m + 4)
        if m == 0:
            continue  # null block: last writer wins, order differs
        if m in live:
            np.testing.assert_array_equal(b[4 * m : 4 * m + 3], a[4 * m : 4 * m + 3], err_msg=f"ssm {m}")
            np.testing.assert_array_equal(b[4 * m + 3, :15], a[4 * m + 3, :15], err_msg=f"conv {m}")
        else:
            np.testing.assert_array_equal(b[blk], raw[blk], err_msg=f"untouched {m}")
