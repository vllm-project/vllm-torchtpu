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
"""CPU contract tests for shared-KV pages (`shared_kv_rotary_dims`).

A shared-KV page holds a KV head's value in head slot 0 and, leading slot 1,
the key's rotated dims; the key equals the value elsewhere. The kernels' values
need a device; these check the page rows the wrapper writes, what the kernels
move, the refusals, and that both kernels lower for tpu7x.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_FP8 = jnp.float8_e4m3fn
_LANE = configs.KVLayout.SEQ_ALONG_LANE
_MESH = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))


def _rows(x):
    """[tokens, d] -> [d / 4, 4, tokens], one SEQ_ALONG_LANE head slot."""
    tokens, d = x.shape
    return np.asarray(x).T.reshape(d // 4, 4, tokens)


def test_prepare_inputs_writes_the_value_and_the_rotated_key_rows():
    tokens, d, rot = 40, 512, 64
    key = jax.random.key(0)
    q = jax.random.normal(key, (tokens, 8, d)).astype(jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (tokens, 1, d)).astype(_FP8)
    v = jax.random.normal(jax.random.fold_in(key, 2), (tokens, 1, d)).astype(_FP8)
    with use_abstract_mesh(_MESH):
        q_hbm, new_kv = wrapper.prepare_inputs(
            q,
            k,
            v,
            jnp.bfloat16,
            _FP8,
            kv_layout=_LANE,
            page_size=256,
            shared_kv_rotary_dims=rot,
        )
        q_std, _ = wrapper.prepare_inputs(
            q, k, v, jnp.bfloat16, _FP8, kv_layout=_LANE, page_size=256
        )
    assert new_kv.shape == (2, d // 4, 4, 256)
    bits = lambda x: np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))  # noqa: E731
    np.testing.assert_array_equal(bits(q_hbm), bits(q_std))
    new_kv = bits(new_kv)
    np.testing.assert_array_equal(new_kv[0, :, :, :tokens], _rows(bits(v[:, 0])))
    k8 = bits(k[:, 0])
    rotated = np.concatenate([k8[:, :rot], k8[:, d // 2 : d // 2 + rot]], axis=1)
    np.testing.assert_array_equal(new_kv[1, : 2 * rot // 4, :, :tokens], _rows(rotated))
    assert not new_kv[1, 2 * rot // 4 :].any() and not new_kv[:, :, :, tokens:].any()


def _cfgs(rot, mode=configs.RpaCase.DECODE, bkv=1536, batch=8):
    model = configs.ModelConfigs(
        num_q_heads=8, num_kv_heads=1, head_dim=512, mask_value=-1e30
    )
    serve = configs.ServingConfigs(
        num_seqs=64,
        page_size=256,
        total_q_tokens=16,
        num_page_indices=64 * 640,
        dtype_q=jnp.bfloat16,
        dtype_kv=_FP8,
        dtype_out=jnp.bfloat16,
        kv_layout=_LANE,
        shared_kv_rotary_dims=rot,
    )
    return configs.RpaConfigs(
        block=configs.BlockSizes(1, 1, bkv, batch),
        model=model,
        serve=serve,
        mode=mode,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


def test_kernels_move_the_value_and_rotated_key_rows_only():
    with use_abstract_mesh(_MESH):
        shared, std = _cfgs(64), _cfgs(0)
        assert (shared.kv_rows, std.kv_rows) == (160, 256)
        assert (shared.kv_bytes_per_token, std.kv_bytes_per_token) == (640, 1024)
        assert shared.kv_vmem_shape == (8, 1, 160, 4, 1536 + 256)
        assert std.kv_vmem_shape == (8, 2, 128, 4, 1536 + 256)


def test_default_decode_block_grows_with_the_smaller_rows():
    with use_abstract_mesh(_MESH):
        shared, std = _cfgs(64), _cfgs(0)
        sizes = [
            wrapper.calculate_block_sizes(c.model, c.serve, 64 * 1024 * 1024)[0]
            for c in (shared, std)
        ]
    assert sizes[0].bkv_sz > sizes[1].bkv_sz


@pytest.mark.parametrize(
    "rot,heads,d,layout,per_token,match",
    [
        (64, 2, 512, _LANE, False, "one KV head"),
        (64, 1, 512, configs.KVLayout.HEAD_ALONG_SUBLANE, False, "one KV head"),
        (48, 1, 512, _LANE, False, "tiles"),
        (512, 1, 512, _LANE, False, "is not in"),
        (64, 1, 512, _LANE, True, "per-token"),
    ],
)
def test_shared_kv_refusals(rot, heads, d, layout, per_token, match):
    with use_abstract_mesh(_MESH), pytest.raises(ValueError, match=match):
        wrapper.validate_shared_kv(rot, heads, d, layout, per_token)


def test_shared_kv_kernels_lower_for_tpu7x():
    """Decode and the fp8 mixed kernel on shared-KV pages pass Mosaic's verifier."""
    num_pages, pages_per_seq, max_seqs, tokens = 64, 16, 8, 1024
    with use_abstract_mesh(_MESH):
        kv_shape = wrapper.get_kv_cache_shape(
            num_pages, 256, 1, 512, _FP8, kv_layout=_LANE
        )
        args = [
            jax.ShapeDtypeStruct((tokens, 8, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens, 1, 512), _FP8),
            jax.ShapeDtypeStruct((tokens, 1, 512), _FP8),
            jax.ShapeDtypeStruct(kv_shape, _FP8),
            jax.ShapeDtypeStruct((max_seqs,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs * pages_per_seq,), jnp.int32),
            jax.ShapeDtypeStruct((max_seqs + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ]

        def f(*a):
            return wrapper.ragged_paged_attention(
                *a,
                sm_scale=512**-0.5,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=_LANE,
                fp8_compute=True,
                shared_kv_rotary_dims=64,
            )

        text = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "RPAd" in text and "RPAm" in text
