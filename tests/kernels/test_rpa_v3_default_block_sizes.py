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
"""VMEM fitting of the RPA v3 default block sizes (runs on CPU)."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as rpa

pytestmark = pytest.mark.cpu_test

_TPU7X_VMEM = 64 * 1024 * 1024


def _defaults(
    monkeypatch,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_tokens,
    case=rpa.RpaCase.MIXED,
):
    monkeypatch.setattr(rpa, "get_tpu_version", lambda: 7)
    return rpa.get_default_block_sizes(
        jnp.bfloat16,
        jnp.bfloat16,
        num_q_heads,
        num_kv_heads,
        head_dim,
        16,  # page_size
        max_tokens,
        64,  # max_num_seqs
        512,  # pages_per_seq -> max_kv = 8192
        case=case,
        vmem_limit_bytes=_TPU7X_VMEM,
    )


def _estimate(bs, *, num_q_heads, num_kv_heads, head_dim):
    return rpa.get_vmem_estimate_bytes(
        num_kv_heads,
        num_q_heads // num_kv_heads,
        head_dim,
        bs["bq_sz"],
        bs["bkv_sz"],
        jnp.bfloat16,
        jnp.bfloat16,
    )


@pytest.mark.parametrize(
    "max_tokens, expected",
    [
        # Unfitted: bq 512/512, bkv 2048/512 -> 71.34 MiB, fails to compile.
        (1024, dict(bq_sz=512, bkv_sz=1024, bq_csz=512, bkv_csz=512)),
        (2048, dict(bq_sz=512, bkv_sz=512, bq_csz=512, bkv_csz=512)),
    ],
)
def test_glm53_nope_mla_mixed_blocks_fit_tpu7x_vmem(monkeypatch, max_tokens, expected):
    # GLM 5.3 Flash NoPE MLA as MHA at TP=8: 8 q / 8 kv heads, head_dim 256.
    shape = dict(num_q_heads=8, num_kv_heads=8, head_dim=256)
    bs = _defaults(monkeypatch, max_tokens=max_tokens, **shape)
    assert bs == expected
    assert _estimate(bs, **shape) <= _TPU7X_VMEM
    assert bs["bq_sz"] % bs["bq_csz"] == 0
    assert bs["bkv_sz"] % bs["bkv_csz"] == 0
    assert bs["bkv_sz"] % 16 == 0


@pytest.mark.parametrize("case", list(rpa.RpaCase))
def test_blocks_that_fit_are_unchanged(monkeypatch, case):
    # 1 kv head / 8 q heads at head_dim 128 is well inside VMEM.
    bs = _defaults(
        monkeypatch,
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=128,
        max_tokens=2048,
        case=case,
    )
    if case == rpa.RpaCase.DECODE:
        assert bs == dict(bq_sz=1, bkv_sz=8192, bq_csz=1, bkv_csz=8192)
    else:
        assert bs == dict(bq_sz=256, bkv_sz=2048, bq_csz=128, bkv_csz=512)


def test_fit_shrinks_kv_block_before_query_block():
    args = dict(
        q_dtype=jnp.bfloat16,
        kv_dtype=jnp.bfloat16,
        actual_num_q_heads=8,
        actual_num_kv_heads=8,
        head_dim=256,
    )
    # A generous limit leaves everything alone.
    assert rpa.fit_block_sizes_to_vmem(
        512, 2048, 512, 512, vmem_limit_bytes=1 << 40, **args
    ) == (512, 2048, 512, 512)
    # 68 MiB estimate -> only the KV fetch block shrinks.
    assert rpa.fit_block_sizes_to_vmem(
        512, 2048, 512, 512, vmem_limit_bytes=_TPU7X_VMEM, **args
    ) == (512, 1024, 512, 512)
    # A tiny limit bottoms out at bkv_csz and bq 1 instead of looping.
    assert rpa.fit_block_sizes_to_vmem(
        512, 2048, 512, 512, vmem_limit_bytes=1, **args
    ) == (1, 512, 1, 512)
