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
"""Tests for batched RPA config and block size calculations across TPU generations."""

import functools
from unittest.mock import MagicMock, patch

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


def _create_mock_tpu_info(
    generation: int,
    chip_version: str = "v8i",
    num_lanes: int = 128,
    mxu_column_size: int = 128,
    vmem_capacity_bytes: int = 64 * 1024 * 1024,
    fp8_ops_per_second: int = 2,
    bf16_ops_per_second: int = 1,
):
    mock = MagicMock()
    mock.generation = generation
    mock.chip_version = chip_version
    mock.num_lanes = num_lanes
    mock.mxu_column_size = mxu_column_size
    mock.vmem_capacity_bytes = vmem_capacity_bytes
    mock.fp8_ops_per_second = fp8_ops_per_second
    mock.bf16_ops_per_second = bf16_ops_per_second
    return mock


@pytest.mark.parametrize(
    "generation,chip_version,is_8bit,expected_bq_c_min",
    [
        (8, "v8i", True, 1),
        (8, "v8i", False, 1),
        (8, "v8t", True, 1),
        (8, "v8t", False, 1),
        (7, "v7x", True, 1),
        (7, "v7x", False, 1),
        (5, "v5p", False, 1),
    ],
)
def test_calculate_block_sizes_generation(
    generation, chip_version, is_8bit, expected_bq_c_min
):
    dtype = jnp.float8_e4m3fn if is_8bit else jnp.bfloat16
    model_cfg = configs.ModelConfigs(
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask_value=-1e9,
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=8,
        page_size=16,
        total_q_tokens=8,
        num_page_indices=128,
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_out=jnp.bfloat16,
    )
    vmem_limit = 64 * 1024 * 1024
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(generation, chip_version=chip_version),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, vmem_limit
        )
        assert decode_blocks.bq_sz >= decode_blocks.bq_c_sz >= expected_bq_c_min
        assert prefill_blocks.bq_sz >= prefill_blocks.bq_c_sz >= expected_bq_c_min


@pytest.mark.parametrize("scale", ["dynamic_k_scale", "dynamic_v_scale"])
def test_prepared_inputs_rejects_dynamic_scales(scale):
    """Dynamic KV scales are applied by prepare_inputs, not alongside it."""
    q = jnp.zeros((8, 2, 128), jnp.bfloat16)
    kv = jnp.zeros((8, 1, 128), jnp.float8_e4m3fn)
    tpu_info = _create_mock_tpu_info(7, chip_version="v7x")
    tpu_info.num_sublanes = 8
    with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=tpu_info):
        prepared = wrapper.prepare_inputs(q, kv, kv, q.dtype, kv.dtype)
        cache = jnp.zeros(
            wrapper.get_kv_cache_shape(1, 128, 1, 128, kv.dtype), kv.dtype
        )
        with pytest.raises(ValueError, match=scale):
            wrapper.ragged_paged_attention(
                q,
                kv,
                kv,
                cache,
                jnp.array([8], jnp.int32),
                jnp.array([0], jnp.int32),
                jnp.array([0, 8], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
                prepared_inputs=prepared,
                **{scale: jnp.ones((8, 1), jnp.float32)},
            )


@pytest.mark.parametrize("decode_batch_size", [0, -1])
def test_calculate_block_sizes_rejects_a_decode_batch_below_one(decode_batch_size):
    model_cfg = configs.ModelConfigs(
        num_q_heads=32, num_kv_heads=8, head_dim=128, mask_value=-1e9
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=64,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=1024,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    with (
        patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            return_value=_create_mock_tpu_info(7, chip_version="v7x"),
        ),
        pytest.raises(
            ValueError, match=f"decode_batch_size .* got {decode_batch_size}"
        ),
    ):
        wrapper.calculate_block_sizes(
            model_cfg,
            serve_cfg,
            32 * 1024 * 1024,
            decode_batch_size=decode_batch_size,
        )


@pytest.mark.parametrize("decode_batch_size", [8, 4, 1])
def test_calculate_block_sizes_caps_the_decode_batch(decode_batch_size):
    model_cfg = configs.ModelConfigs(
        num_q_heads=32, num_kv_heads=8, head_dim=128, mask_value=-1e9
    )
    serve_cfg = configs.ServingConfigs(
        num_seqs=64,
        page_size=16,
        total_q_tokens=64,
        num_page_indices=1024,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.bfloat16,
        dtype_out=jnp.bfloat16,
    )
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=_create_mock_tpu_info(7, chip_version="v7x"),
    ):
        decode_blocks, prefill_blocks = wrapper.calculate_block_sizes(
            model_cfg,
            serve_cfg,
            32 * 1024 * 1024,
            decode_batch_size=decode_batch_size,
        )
        default_prefill = wrapper.calculate_block_sizes(
            model_cfg, serve_cfg, 32 * 1024 * 1024
        )[1]
    assert decode_blocks.batch_size == decode_batch_size
    assert prefill_blocks == default_prefill


def _rpa_kernel_names(**kwargs):
    """The names of the RPA kernels (not the schedule kernels) the jitted
    wrapper traces on TPU7x for a 16-token bucket with one new sequence; a
    name carries the kernel's batch and tiles."""
    heads, kv_heads, head_dim, seqs, pages_per_seq, page, tokens = (
        12,
        2,
        128,
        8,
        16,
        128,
        16,
    )
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((tokens, heads, head_dim)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((tokens, kv_heads, head_dim)), jnp.bfloat16)
    cache = jnp.zeros(
        wrapper.get_kv_cache_shape(
            seqs * pages_per_seq + 4, page, kv_heads, head_dim, jnp.bfloat16
        ),
        jnp.bfloat16,
    )
    kv_lens = np.zeros(seqs, np.int32)
    kv_lens[0] = 10 + tokens
    cu_q_lens = np.zeros(seqs + 1, np.int32)
    cu_q_lens[1:] = tokens
    call = functools.partial(
        wrapper.ragged_paged_attention, sm_scale=head_dim**-0.5, **kwargs
    )
    jaxpr = jax.make_jaxpr(call)(
        q,
        k,
        k,
        cache,
        jnp.asarray(kv_lens),
        jnp.zeros(seqs * pages_per_seq, jnp.int32),
        jnp.asarray(cu_q_lens),
        jnp.asarray([0, 1, 1], jnp.int32),
    ).jaxpr
    names = []

    def walk(j):
        for eqn in j.eqns:
            if eqn.primitive.name == "pallas_call":
                names.append(eqn.params["name"])
                continue
            for value in eqn.params.values():
                for sub in value if isinstance(value, (list, tuple)) else [value]:
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jaxpr)
    return [n for n in names if n.startswith("RPA")]


@pytest.mark.parametrize("decode_batch_size", [8, 4])
def test_the_jitted_wrapper_passes_the_decode_batch_size_to_its_decode_kernel(
    monkeypatch, decode_batch_size
):
    """decode_batch_size is a static argument of ragged_paged_attention: the
    DECODE kernel the wrapper traces works on that many sequences, and the
    MIXED kernel keeps its batch of 2."""
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda *args, **kwargs: info)
    names = _rpa_kernel_names(decode_batch_size=decode_batch_size)
    decode = f"RPA{configs.RpaCase.DECODE.symbol}-"
    mixed = f"RPA{configs.RpaCase.MIXED.symbol}-"
    assert [n for n in names if n.startswith(decode)] == [
        n for n in names if f"-b{decode_batch_size}-" in n
    ], names
    assert len([n for n in names if n.startswith(decode)]) == 1, names
    assert all("-b2-" in n for n in names if n.startswith(mixed)), names
