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
import math
from types import SimpleNamespace

import pytest
import torch
from vllm_omni.attention.fish_kvcache_backend import (
    get_fish_kvcache_attn_stats,
    install_fish_kvcache_attn_backend,
    reset_fish_kvcache_attn_stats,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 2e-2
MAX_ABS = 5e-2


def _gqa_paged_decode_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """CPU fp32 grouped-query paged KV-cache decode attention reference."""
    batch, num_q_heads, head_dim = query.shape
    num_kv_heads = kv_cache.shape[1]
    group = num_q_heads // num_kv_heads
    q_fp32 = query.detach().cpu().float()
    k_cache, v_cache = kv_cache.detach().cpu().float().split(head_dim, dim=-1)
    outputs = []
    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        blocks = block_table[b]
        keys = torch.cat([k_cache[int(blk)] for blk in blocks], dim=1)[:, :seq_len]
        vals = torch.cat([v_cache[int(blk)] for blk in blocks], dim=1)[:, :seq_len]
        keys = keys.repeat_interleave(group, dim=0)
        vals = vals.repeat_interleave(group, dim=0)
        scores = torch.einsum("hd,hsd->hs", q_fp32[b], keys) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hs,hsd->hd", probs, vals))
    return torch.stack(outputs, dim=0)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "fish_kvcache_attn.is_available() gates on current_platform.is_cuda() "
        "and raises RuntimeError when required on TPU"
    ),
)
def test_fish_kvcache_triton_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fish KV-Cache Triton Attention

    - TPU path: require Fish KV-cache attention and install on 1-layer TPU wrapper.
    - Setup check: assert installed == 1 and small_hit_count == 1 after decode forward.
    - Outcome check: compare TPU bf16 output against CPU fp32 GQA reference.
    """
    monkeypatch.setenv("VLLM_OMNI_FISH_KVCACHE_ATTN", "required")
    reset_fish_kvcache_attn_stats()

    torch.manual_seed(0)
    batch, q_heads, kv_heads, block_size, head_dim = 2, 8, 2, 16, 128
    num_blocks = 4
    scale = 1.0 / math.sqrt(head_dim)

    def _fallback_forward(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError(
            "fallback forward should not be called on Fish KV-cache hit"
        )

    impl = SimpleNamespace(
        forward=_fallback_forward,
        dcp_world_size=1,
        alibi_slopes=None,
        sliding_window=None,
        scale=scale,
    )
    model = SimpleNamespace(
        layers=[
            SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(impl=impl)))
        ]
    )

    installed = install_fish_kvcache_attn_backend(model)
    assert installed == 1, f"expected 1 installed layer, got {installed}"

    query_cpu = torch.randn((batch, q_heads, head_dim), dtype=torch.bfloat16)
    kv_cache_cpu = torch.randn(
        (num_blocks, kv_heads, block_size, 2 * head_dim), dtype=torch.bfloat16
    )
    block_table_cpu = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([12, 20], dtype=torch.int32)

    ref = _gqa_paged_decode_reference(
        query_cpu, kv_cache_cpu, block_table_cpu, seq_lens_cpu, scale
    )

    tpu = torch.device("tpu")
    query_tpu = query_cpu.to(tpu)
    kv_cache_tpu = kv_cache_cpu.to(tpu)
    output_tpu = torch.empty_like(query_tpu)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=batch,
        block_table=block_table_cpu.to(tpu),
        seq_lens=seq_lens_cpu.to(tpu),
        max_query_len=1,
        max_seq_len=20,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
    )

    out = impl.forward(
        None,
        query_tpu,
        None,
        None,
        kv_cache_tpu,
        attn_metadata,
        output_tpu,
    )
    stats = get_fish_kvcache_attn_stats()
    assert stats["small_hit_count"] == 1, (
        f"expected small_hit_count == 1, got {stats['small_hit_count']}"
    )
    assert_parity(
        out,
        ref,
        name="fish_kvcache_triton_attention",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
