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
"""`fence_output` and `prepared_inputs` of the batched RPA wrapper.

`fence_output`: the token-major output goes through one optimization barrier.
`prepared_inputs`: handing the wrapper what prepare_inputs returns launches the
same kernels. The lowering halves run on CPU (StableHLO lowered for an abstract
TPU7x device); the kernel half needs a TPU and checks bitwise equality.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB

_DECODE_BLOCKS = KBC.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = KBC.BlockSizes(bq_sz=64, bq_c_sz=16, bkv_sz=256, batch_size=2)
_NUM_Q_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 2, 1, 256
_PAGE_SIZE, _PAGES_PER_SEQ, _MAX_SEQS, _TOKENS = 16, 8, 16, 16


def _args():
    cache_shape = KB.get_kv_cache_shape(
        _MAX_SEQS * _PAGES_PER_SEQ,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        jnp.float8_e4m3fn,
    )
    rng = np.random.default_rng(0)
    q = rng.standard_normal((_TOKENS, _NUM_Q_HEADS, _HEAD_DIM))
    kv = rng.standard_normal((2, _TOKENS, _NUM_KV_HEADS, _HEAD_DIM))
    kv_lens = (100, 40) + (0,) * (_MAX_SEQS - 2)
    cu_q_lens = (0, 1, 13) + (13,) * (_MAX_SEQS - 2)
    return (
        jnp.asarray(q, jnp.bfloat16),
        # New K/V arrive in the cache dtype, as the attention layer quantizes them.
        jnp.asarray(kv[0], jnp.float8_e4m3fn),
        jnp.asarray(kv[1], jnp.float8_e4m3fn),
        jnp.asarray(rng.standard_normal(cache_shape), jnp.float8_e4m3fn),
        jnp.asarray(kv_lens, jnp.int32),
        jnp.arange(_MAX_SEQS * _PAGES_PER_SEQ, dtype=jnp.int32),
        jnp.asarray(cu_q_lens, jnp.int32),
        jnp.asarray((1, 1, 2), jnp.int32),
    )


def _call(fence_output, *args, prepared=False):
    q, k, v, cache = args[:4]
    prepared_inputs = (
        KB.prepare_inputs(q, k, v, q.dtype, cache.dtype) if prepared else None
    )
    return KB.ragged_paged_attention(
        *args,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        fence_output=fence_output,
        prepared_inputs=prepared_inputs,
    )


def _lowered_text(fence_output, prepared=False):
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh):
        return (
            jax.jit(lambda *a: _call(fence_output, *a, prepared=prepared))
            .trace(*_args())
            .lower(lowering_platforms=("tpu",))
            .as_text()
        )


@pytest.mark.parametrize("fence_output", [False, True])
def test_lowering_has_one_output_barrier(fence_output):
    barriers = _lowered_text(fence_output).count("stablehlo.optimization_barrier")
    assert barriers == int(fence_output)


def test_prepared_inputs_launch_the_same_kernels():
    def kernels(text):
        return sorted(re.findall(r"backend_config = \"([^\"]*)\"", text))

    plain, prepared = _lowered_text(False), _lowered_text(False, prepared=True)
    assert kernels(plain) and kernels(prepared) == kernels(plain)


def test_prepared_inputs_must_match_prepare_inputs():
    device = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    q, k, v, cache, *rest = _args()

    def call(*args):
        good_q, good_kv = KB.prepare_inputs(q, k, v, q.dtype, cache.dtype)
        return KB.ragged_paged_attention(
            *args,
            decode_block_sizes=_DECODE_BLOCKS,
            prefill_block_sizes=_PREFILL_BLOCKS,
            prepared_inputs=(good_q[:, :8], good_kv),
        )

    mesh = jax.sharding.AbstractMesh((), (), abstract_device=device)
    with jax.sharding.use_abstract_mesh(mesh), pytest.raises(ValueError, match="prep"):
        jax.jit(call).trace(q, k, v, cache, *rest)


@pytest.mark.tpu_test
def test_outputs_bitwise_equal():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("the batched RPA kernel needs a TPU")
    ref_out, ref_cache = _call(False, *_args())
    out, cache = _call(True, *_args())
    np.testing.assert_array_equal(
        np.asarray(out.view(jnp.uint16)), np.asarray(ref_out.view(jnp.uint16))
    )
    np.testing.assert_array_equal(
        np.asarray(cache.view(jnp.uint8)), np.asarray(ref_cache.view(jnp.uint8))
    )
