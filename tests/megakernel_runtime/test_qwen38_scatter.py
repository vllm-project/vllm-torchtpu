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


@pytest.mark.parametrize(
    "pad_requests,ring,kb,prefetch",
    [(False, 4, 128, 1), (True, 4, 128, 1), (False, 2, 128, 1), (False, 8, 128, 1), (False, 4, 512, 1),
     (True, 4, 512, 1), (False, 8, 128, 3), (True, 8, 128, 7), (False, 12, 128, 6), (True, 4, 512, 3)],
)
def test_fast_scatter_matches_region_scatter(pad_requests: bool, ring: int, kb: int, prefetch: int):
    """Served pools use 128-row (QSA kernel) or 512-row (one per manager block) blocks."""
    SHAPE = (NMB * (512 // kb), kb, 1, 4, 256)
    LAYOUT = gs.Qwen38GdnPoolLayout(pool_shape=SHAPE, pool_dtype="float8_e4m3fn")
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)
    wb = 1 + np.arange(NL * NS * T, dtype=np.int32).reshape(NL, NS, T)
    if pad_requests:
        wb[:, -1] = 0  # a padded request writes the null block only
    wb = jnp.asarray(wb)
    if kb == 128:
        ref = gs.scatter_gdn_ckpts(pool, conv, rec, wb, layout=LAYOUT, interpret=True)
    else:
        # The region adapters do not run under the CPU interpreter at kb=512;
        # the reference is the kb=128 fast scatter (checked against them above)
        # on the same bytes, which are the same rows per manager block.
        shape128 = (NMB * 4, 128, 1, 4, 256)
        ref = scatter_gdn_ckpts_fast(
            pool.reshape(shape128), conv, rec, wb, interpret=True,
            layout=gs.Qwen38GdnPoolLayout(pool_shape=shape128, pool_dtype="float8_e4m3fn"),
        )
    got = scatter_gdn_ckpts_fast(
        pool, conv, rec, wb, layout=LAYOUT, interpret=True, ring=ring, prefetch=prefetch
    )
    a = np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8))
    b = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint8))
    # Manager blocks other than the written ones must be untouched by both.
    live = sorted(set(np.asarray(wb).reshape(-1).tolist()) - {0})
    a, b, r = (x.reshape(NMB, 512, -1) for x in (a, b, raw))
    for m in range(1, NMB):  # null block 0: last writer wins, order differs
        if m in live:
            np.testing.assert_array_equal(b[m, :384], a[m, :384], err_msg=f"ssm {m}")
            np.testing.assert_array_equal(b[m, 384:399], a[m, 384:399], err_msg=f"conv {m}")
        else:
            np.testing.assert_array_equal(b[m], r[m], err_msg=f"untouched {m}")


def test_fast_scatter_check_catches_a_misaddressed_write():
    """Fault control: the deep-prefetch scatter with two requests' blocks swapped must fail the
    byte comparison (a slot / index mix-up in the ring shows up the same way)."""
    with pytest.raises(AssertionError):
        _swapped_blocks_compare()


def _swapped_blocks_compare():
    SHAPE = (NMB * 4, 128, 1, 4, 256)
    LAYOUT = gs.Qwen38GdnPoolLayout(pool_shape=SHAPE, pool_dtype="float8_e4m3fn")
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    conv = jnp.asarray(rng.standard_normal((NL, NS, 8, 1280)), jnp.float32)
    rec = jnp.asarray(rng.standard_normal((NL, NS * T * 6, 128, 128)), jnp.float32).astype(jnp.bfloat16)
    wb = 1 + np.arange(NL * NS * T, dtype=np.int32).reshape(NL, NS, T)
    bad = wb.copy()
    bad[0, 0], bad[0, 1] = wb[0, 1].copy(), wb[0, 0].copy()
    ref = gs.scatter_gdn_ckpts(pool, conv, rec, jnp.asarray(wb), layout=LAYOUT, interpret=True)
    got = scatter_gdn_ckpts_fast(pool, conv, rec, jnp.asarray(bad), layout=LAYOUT, interpret=True, ring=8, prefetch=3)
    a = np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint8)).reshape(NMB, 512, -1)
    b = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint8)).reshape(NMB, 512, -1)
    np.testing.assert_array_equal(b[1:, :399], a[1:, :399])
