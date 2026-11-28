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
"""CPU unit tests for RPA v3 VMEM-aware default block-size clamping."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa_v3

_MIB = 1024 * 1024


@pytest.mark.parametrize(
    ("vmem_mib", "tpu_version"),
    [(16, 5), (32, 6), (64, 7)],
)
def test_vmem_budget_and_static_validate_sweep(
    monkeypatch: pytest.MonkeyPatch,
    vmem_mib: int,
    tpu_version: int,
) -> None:
    vmem_limit = vmem_mib * _MIB
    monkeypatch.setattr(rpa_v3, "get_tpu_version", lambda: tpu_version)
    grid = itertools.product(
        (1, 2, 4, 8, 16, 32),
        (1, 4),
        (128, 256, 512),
        ((16, 256), (16, 3), (64, 64)),
    )
    for kv_heads, gqa_ratio, head_dim, (page_size, pages_per_seq) in grid:
        q_heads = kv_heads * gqa_ratio
        q_shape = jax.ShapeDtypeStruct((2048, q_heads, head_dim), jnp.bfloat16)
        kv_shape = jax.ShapeDtypeStruct((2048, kv_heads, head_dim), jnp.bfloat16)
        cache_shape = rpa_v3.get_kv_cache_shape(
            pages_per_seq, page_size, kv_heads, head_dim, jnp.bfloat16
        )
        cache = jax.ShapeDtypeStruct(cache_shape, jnp.bfloat16)
        blocks = {}
        for case in (
            rpa_v3.RpaCase.DECODE,
            rpa_v3.RpaCase.PREFILL,
            rpa_v3.RpaCase.MIXED,
        ):
            bs = rpa_v3.get_default_block_sizes(
                jnp.bfloat16,
                jnp.bfloat16,
                q_heads,
                kv_heads,
                head_dim,
                page_size,
                2048,
                16,
                pages_per_seq,
                case=case,
                vmem_limit_bytes=vmem_limit,
            )
            est = rpa_v3.get_vmem_estimate_bytes(
                kv_heads,
                gqa_ratio,
                head_dim,
                bs["bq_sz"],
                bs["bkv_sz"],
                jnp.bfloat16,
                jnp.bfloat16,
            )
            min_est = rpa_v3.get_vmem_estimate_bytes(
                kv_heads,
                gqa_ratio,
                head_dim,
                1,
                page_size,
                jnp.bfloat16,
                jnp.bfloat16,
            )
            budget = int(vmem_limit * rpa_v3._VMEM_CLAMP_UTILIZATION)
            assert est <= budget if min_est <= budget else (
                (bs["bq_sz"], bs["bkv_sz"]) == (1, page_size)
            )
            blocks[case] = (
                bs["bq_sz"],
                bs["bkv_sz"],
                bs["bq_csz"],
                bs["bkv_csz"],
            )

        rpa_v3.static_validate_inputs(
            q_shape,
            kv_shape,
            kv_shape,
            cache,
            jax.ShapeDtypeStruct((16,), jnp.int32),
            jax.ShapeDtypeStruct((16 * pages_per_seq,), jnp.int32),
            jax.ShapeDtypeStruct((17,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            d_block_sizes=blocks[rpa_v3.RpaCase.DECODE],
            p_block_sizes=blocks[rpa_v3.RpaCase.PREFILL],
            m_block_sizes=blocks[rpa_v3.RpaCase.MIXED],
            vmem_limit_bytes=vmem_limit,
        )


def test_get_default_block_sizes_propagates_tpu_info_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When vmem_limit_bytes is omitted, pltpu.get_tpu_info() errors propagate."""
    monkeypatch.setattr(rpa_v3, "get_tpu_version", lambda: 7)

    def _fail():
        raise ValueError("No TPU device")

    monkeypatch.setattr(rpa_v3.pltpu, "get_tpu_info", _fail)
    with pytest.raises(ValueError, match="No TPU device"):
        rpa_v3.get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            actual_num_q_heads=16,
            actual_num_kv_heads=8,
            head_dim=128,
            page_size=16,
            max_num_tokens=4096,
            max_num_seqs=16,
            pages_per_seq=256,
            case=rpa_v3.RpaCase.MIXED,
            vmem_limit_bytes=None,
        )


@pytest.mark.parametrize(
    ("tpu_version", "vmem_mib", "kv_heads", "head_dim"),
    [
        (5, 16, 8, 256),
        (6, 32, 8, 512),
        (7, 64, 16, 256),
    ],
)
@pytest.mark.parametrize("case", [rpa_v3.RpaCase.PREFILL, rpa_v3.RpaCase.MIXED])
def test_vmem_clamp_shrinks_bkv_before_collapsing_bq(
    monkeypatch: pytest.MonkeyPatch,
    tpu_version: int,
    vmem_mib: int,
    kv_heads: int,
    head_dim: int,
    case: rpa_v3.RpaCase,
) -> None:
    """When bkv_x2_ref dominates VMEM, clamping shrinks bkv_sz so bq_sz stays > 1."""
    vmem_limit = vmem_mib * _MIB
    monkeypatch.setattr(rpa_v3, "get_tpu_version", lambda: tpu_version)
    bs = rpa_v3.get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        actual_num_q_heads=kv_heads,
        actual_num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=16,
        max_num_tokens=2048,
        max_num_seqs=16,
        pages_per_seq=256,
        case=case,
        vmem_limit_bytes=vmem_limit,
    )
    est = rpa_v3.get_vmem_estimate_bytes(
        kv_heads,
        1,
        head_dim,
        bs["bq_sz"],
        bs["bkv_sz"],
        jnp.bfloat16,
        jnp.bfloat16,
    )
    assert bs["bq_sz"] > 1
    assert est <= int(vmem_limit * rpa_v3._VMEM_CLAMP_UTILIZATION)
    assert bs["bq_sz"] % bs["bq_csz"] == 0
    assert bs["bkv_sz"] % bs["bkv_csz"] == 0
