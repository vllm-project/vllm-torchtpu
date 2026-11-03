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
"""CPU contract tests for `ragged_paged_attention(unmasked_interior=True)`.

The mask-free step body must run exactly on the steps whose `generate_mask`
keeps every (query, key) pair; the kernel's values need a device.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, kernel, wrapper


@dataclasses.dataclass
class _Schedule:
    s_idx: np.ndarray


def _cfgs(bq, bq_c, bkv, sliding_window, scope):
    model = configs.ModelConfigs(
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=512,
        mask_value=-1e30,
        sliding_window=sliding_window,
    )
    serve = configs.ServingConfigs(
        num_seqs=8,
        page_size=256,
        total_q_tokens=16384,
        num_page_indices=8 * 640,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        attention_scope=scope,
        unmasked_interior=True,
    )
    return configs.RpaConfigs(
        block=configs.BlockSizes(bq, bq_c, bkv, 1),
        model=model,
        serve=serve,
        mode=configs.RpaCase.MIXED,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


def _masks_nothing(cfgs, meta):
    rows = cfgs.bq_c_sz * cfgs.aligned_num_q_heads_per_kv_head
    shape = (1, cfgs.model.num_kv_heads, rows, cfgs.bkv_sz)
    return all(
        bool(m.all())
        for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz)
        for m in kernel.generate_mask(
            shape, bq_start=bq_start, step_meta=meta, cfgs=cfgs
        )
    )


@pytest.mark.parametrize(
    "bq,bq_c,bkv,sliding_window",
    [
        (512, 32, 512, None),  # Gemma4-31B TP2 global layer, fp8 geometry
        (256, 32, 2048, None),
        (256, 64, 256, 1024),  # a sliding-window layer
        (64, 16, 128, 200),
    ],
)
@pytest.mark.parametrize(
    "scope",
    [
        configs.AttentionScope.FULL,
        configs.AttentionScope.NEW_TOKENS_ONLY,
        configs.AttentionScope.CACHE_ONLY,
    ],
)
def test_step_needs_mask_matches_generate_mask(bq, bq_c, bkv, sliding_window, scope):
    cfgs = _cfgs(bq, bq_c, bkv, sliding_window, scope)
    schedule = _Schedule(s_idx=np.zeros((1, 1), np.int32))
    seen = set()
    for offset in range(-3 * bkv - bq, 2 * bkv, 16):
        for local in (-bkv, 0, 16, bkv, 2 * bkv):
            meta = kernel.ComputeMetadata(
                causal_offset=[jnp.int32(offset)],
                local_k_start=(
                    [jnp.int32(local)]
                    if scope == configs.AttentionScope.NEW_TOKENS_ONLY
                    else None
                ),
                local_k_end=(
                    [jnp.int32(local)]
                    if scope == configs.AttentionScope.CACHE_ONLY
                    else None
                ),
            )
            needs = bool(kernel.step_needs_mask(0, schedule, meta, cfgs=cfgs))
            assert needs == (not _masks_nothing(cfgs, meta)), (offset, local)
            seen.add(needs)
    assert seen == {True, False}


def test_padding_steps_take_the_masked_body():
    cfgs = _cfgs(512, 32, 512, None, configs.AttentionScope.FULL)
    meta = kernel.ComputeMetadata(causal_offset=[jnp.int32(-4096)])
    padding = _Schedule(s_idx=np.full((1, 1), -1, np.int32))
    assert bool(kernel.step_needs_mask(0, padding, meta, cfgs=cfgs))
    valid = _Schedule(s_idx=np.zeros((1, 1), np.int32))
    assert not bool(kernel.step_needs_mask(0, valid, meta, cfgs=cfgs))


def test_mixed_kernel_with_unmasked_interior_lowers_for_tpu7x():
    """Both step bodies trace and pass Mosaic's verifier for tpu7x (no device)."""
    import jax

    num_pages, pages_per_seq, max_seqs, tokens = 64, 16, 8, 1024
    fp8 = jnp.float8_e4m3fn
    kv_shape = wrapper.get_kv_cache_shape(num_pages, 256, 2, 512, fp8)
    args = [
        jax.ShapeDtypeStruct((tokens, 16, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 2, 512), fp8),
        jax.ShapeDtypeStruct((tokens, 2, 512), fp8),
        jax.ShapeDtypeStruct(kv_shape, fp8),
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
            fp8_compute=True,
            unmasked_interior=True,
        )

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        text = jax.jit(f).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
    assert "RPAm" in text
