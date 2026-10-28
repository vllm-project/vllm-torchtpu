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
"""On-TPU time of one segment's GDN checkpoint scatter (one device): adapter vs ring sizes."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast


def timed(f, pool, *a, reps=20):
    p = f(pool, *a)
    jax.block_until_ready(p)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            p = f(p, *a)
        jax.block_until_ready(p)
        best = min(best, (time.perf_counter() - t) / reps)
    return round(best * 1e6, 1)


for NS in (2, 4, 8, 16):
    NL, T = 3, 4
    nmb = 1 + NL * NS * T + 32
    shape = (nmb * 4, 128, 1, 4, 256)
    lay = gs.Qwen38GdnPoolLayout(pool_shape=shape, pool_dtype="float8_e4m3fn")
    pool = jnp.zeros(shape, jnp.float8_e4m3fn)
    conv = jnp.ones((NL, NS, 8, 1280), jnp.float32)
    rec = jnp.ones((NL, NS * T * 6, 128, 128), jnp.bfloat16)
    wb = jnp.asarray(1 + np.arange(NL * NS * T, dtype=np.int32).reshape(NL, NS, T))
    out = {"NS": NS, "rows": NS * T}
    out["adapter_us"] = timed(
        jax.jit(lambda p, c, r, w: gs.scatter_gdn_ckpts(p, c, r, w, layout=lay), donate_argnums=0),
        pool, conv, rec, wb,
    )
    for ring, pf in ((4, 1), (8, 1), (4, 2), (4, 3), (8, 3), (8, 6), (12, 6), (12, 10), (16, 12)):
        pool = jnp.zeros(shape, jnp.float8_e4m3fn)
        out[f"fast_r{ring}_p{pf}_us"] = timed(
            jax.jit(
                lambda p, c, r, w, ring=ring, pf=pf: scatter_gdn_ckpts_fast(
                    p, c, r, w, layout=lay, ring=ring, prefetch=pf
                ),
                donate_argnums=0,
            ),
            pool, conv, rec, wb,
        )
    print("RESULT", json.dumps(out), flush=True)
