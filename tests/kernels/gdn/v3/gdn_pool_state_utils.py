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
"""Host-side view of the GDN V3 unified-pool state layout, for tests.

The V3 kernel owns how a state block is laid out in the unified pool;
production callers only hand it the pool. Tests still need to place and
inspect state bytes from outside the kernel API, so this module mirrors the
kernel's layout:

* The kernel addresses one state block as a stream of uint32 words. A ref
  `bitcast(uint32)` packs along the pool's second-minor (sublane) axis, so a
  pool row `(..., packing, lanes)` yields one word per lane.
* The recurrent (SSM) state starts at granule 0. An fp32 state is stored as
  is; a bf16 state packs row pairs per word (row 2r in the low half, row
  2r + 1 in the high half of lane l), the way a ref bitcast packs it.
* The fp32 conv state starts at the granule right after the recurrent
  state; see `GDNConfig.conv_hbm_offset`.

Import it as `kernels.gdn.v3.gdn_pool_state_utils`: `tests/` has no
`__init__.py`, so pytest puts it on `sys.path` for `tests/conftest.py`, and
`tests/kernels` resolves as the namespace package `kernels`. A regular
package named `kernels` installed in site-packages would shadow it.
"""

import math

import jax
import jax.numpy as jnp

# One granule is a row of 128 uint32 lanes, the unit the kernel moves state in.
GRANULE = 512
WORDS_PER_GRANULE = GRANULE // 4


def on_cpu(fn, *args):
    """Run a pure bit-shuffle on CPU, then hand the result back to the TPU.

    TPU canonicalises NaN payloads whenever float bytes pass through a vector
    register, so bitcasting or transposing a bf16/fp8 pool on device silently
    rewrites about a percent of its words -- measured on v7/jax 0.10.2: 71 of
    4096 uint32 words through bfloat16, 77 through float8_e4m3fn, 0 through
    int8. The kernel never meets this, because its granule DMA moves the pool
    as uint32 and no float register ever sees it; a host-side harness that
    builds or reads the pool *through the pool's own dtype* does. CPU bitcasts
    are exact, so the byte layout these tests assert on is the real one.
    """
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        out = jax.jit(fn)(*jax.device_put(args, cpu))
    return jax.device_put(out, jax.devices()[0])


def state_words(pool, split):
    """The pool the way the kernel addresses it: one uint32 row per state.

    A ref's `bitcast(uint32)` packs along the *second-minor* (sublane) axis,
    which for a pool row `(..., packing, lanes)` means one word per lane, not
    one word per adjacent pair. Moving that axis minormost first is what makes
    the host view agree with the kernel's granule stream.
    """
    words = jnp.moveaxis(pool, -2, -1).view(jnp.uint32)
    return words.reshape(pool.shape[0] // split, -1)


def words_to_pool(words, pool):
    """Inverse of `state_words`, back to `pool`'s shape and dtype."""
    moved = words.reshape(*pool.shape[:-2], pool.shape[-1], 1).view(pool.dtype)
    return jnp.moveaxis(moved, -2, -1).reshape(pool.shape)


def put_words(words, idx, granule0, payload):
    """Drop `payload`'s raw bytes at a granule offset of each named state."""
    body = payload.reshape(idx.shape[0], -1).view(jnp.uint32)
    start = granule0 * WORDS_PER_GRANULE
    return words.at[idx, start : start + body.shape[1]].set(body)


def take_words(words, idx, granule0, shape, dtype):
    """Read `shape`-shaped `dtype` values at a granule offset of each state."""
    nwords = math.prod(shape) * jnp.dtype(dtype).itemsize // 4
    start = granule0 * WORDS_PER_GRANULE
    return words[idx, start : start + nwords].view(dtype).reshape(idx.shape[0], *shape)


def pack_bf16_rows(ssm):
    """bf16 [..., D_K, D_V] -> uint32 [..., D_K // 2, D_V], packed the way a
    ref bitcast packs it: word (r, l) holds row 2r (low half) and row 2r + 1
    (high half) of lane l."""
    pairs = ssm.astype(jnp.bfloat16).reshape(
        *ssm.shape[:-2], ssm.shape[-2] // 2, 2, ssm.shape[-1]
    )
    return jax.lax.bitcast_convert_type(jnp.moveaxis(pairs, -2, -1), jnp.uint32)


def unpack_bf16_rows(words):
    """Inverse of `pack_bf16_rows`."""
    pairs = jax.lax.bitcast_convert_type(words, jnp.bfloat16)
    rows = jnp.moveaxis(pairs, -1, -2)
    return rows.reshape(*words.shape[:-2], words.shape[-2] * 2, words.shape[-1])


def recurrent_granules(recurrent_shape, recurrent_dtype=jnp.float32) -> int:
    """Granules taken by one recurrent state, i.e. where the conv state starts."""
    nbytes = math.prod(recurrent_shape) * jnp.dtype(recurrent_dtype).itemsize
    assert nbytes % GRANULE == 0, (recurrent_shape, recurrent_dtype)
    return nbytes // GRANULE


def _recurrent_words(ssm, recurrent_dtype):
    if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
        return pack_bf16_rows(ssm)
    assert jnp.dtype(recurrent_dtype) == jnp.float32, recurrent_dtype
    return jnp.asarray(ssm, jnp.float32).view(jnp.uint32)


def write_states(pool, ssm, conv, idx, *, split, recurrent_dtype=jnp.float32):
    """Place recurrent and conv states of the states named by `idx`.

    `ssm` is `[len(idx), num_v_heads, d_k, d_v]` and `conv` is
    `[len(idx), kernel_size - 1, conv_dim]`. Region tails are left alone --
    the kernel moves live granules only, and nothing reads the padding.
    """

    def build(pool, ssm, conv, idx):
        words = state_words(pool, split)
        words = put_words(words, idx, 0, _recurrent_words(ssm, recurrent_dtype))
        words = put_words(
            words,
            idx,
            recurrent_granules(ssm.shape[1:], recurrent_dtype),
            jnp.asarray(conv, jnp.float32),
        )
        return words_to_pool(words, pool)

    return on_cpu(build, pool, ssm, conv, idx)


def read_states(
    pool, idx, *, split, recurrent_shape, conv_shape, recurrent_dtype=jnp.float32
):
    """Read back `(ssm, conv)` of the states named by `idx`.

    `recurrent_shape` is `(num_v_heads, d_k, d_v)` and `conv_shape` is
    `(kernel_size - 1, conv_dim)`. The conv state is fp32.
    """

    def take(pool, idx):
        words = state_words(pool, split)
        if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
            num_v_heads, d_k, d_v = recurrent_shape
            packed = take_words(words, idx, 0, (num_v_heads, d_k // 2, d_v), jnp.uint32)
            ssm = unpack_bf16_rows(packed)
        else:
            ssm = take_words(words, idx, 0, recurrent_shape, recurrent_dtype)
        conv = take_words(
            words,
            idx,
            recurrent_granules(recurrent_shape, recurrent_dtype),
            conv_shape,
            jnp.float32,
        )
        return ssm, conv

    return on_cpu(take, pool, idx)
