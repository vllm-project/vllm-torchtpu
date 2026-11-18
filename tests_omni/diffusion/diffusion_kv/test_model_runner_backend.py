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
"""TPU component tests for DiffusionKVModelRunnerBackend and prefix caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.model_runner_backend import (
    DiffusionKVModelRunnerBackend,
)
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionRow,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

PAGED_KV_MAX_REL_L2 = 2.0e-2
NUM_HEADS = 4
HEAD_SIZE = 64
BLOCK_SIZE = 16


def _build_minimal_vllm_config() -> SimpleNamespace:
    attention_geometry: dict[str, int] = {}

    def set_attention_geometry(
        *, num_heads: int, num_kv_heads: int, head_size: int
    ) -> None:
        attention_geometry["num_heads"] = num_heads
        attention_geometry["num_kv_heads"] = num_kv_heads
        attention_geometry["head_size"] = head_size

    model_config = SimpleNamespace(
        max_model_len=64,
        dtype=torch.bfloat16,
        set_attention_geometry=set_attention_geometry,
    )
    scheduler_config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
        tensor_parallel_size=1,
    )
    cache_config = SimpleNamespace(cache_dtype="auto")
    attention_config = SimpleNamespace(use_non_causal=False)
    compilation_config = SimpleNamespace(static_forward_context={})
    device_config = SimpleNamespace(device=torch.device("tpu"))
    return SimpleNamespace(
        model_config=model_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        attention_config=attention_config,
        compilation_config=compilation_config,
        device_config=device_config,
    )


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "TPU diffusion attention backends do not support paged KV "
        "(supports_paged_kv is False / PallasAttentionBackend.get_builder_cls "
        "raises NotImplementedError)"
    ),
)
def test_diffusion_paged_kv_and_prefix_caching() -> None:
    """Diffusion paged KV cache and prefix caching

    - TPU path: Configure OmniDiffusionConfig(diffusion_kv_mode="paged_scheduler",
      enable_prefix_caching=True), register a paged diffusion Attention layer on
      DiffusionKVModelRunnerBackend with device=torch.device("tpu"), initialize
      native KV cache pools, and run prefill + prefix-cached suffix forward_paged.
    - Setup check: Assert od_config.diffusion_kv_mode is
      DiffusionKVCacheMode.PAGED_SCHEDULER, od_config.enable_prefix_caching is
      True, and backend.kv_cache_config.num_blocks > 0.
    - Outcome check: Assert paged prefill and prefix-cached suffix outputs on
      "tpu" match CPU float32 non-causal SDPA reference within bf16 relative-L2
      tolerance.
    """
    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        diffusion_kv_mode="paged_scheduler",
        enable_prefix_caching=True,
        diffusion_kv_max_rows_per_request=2,
    )
    assert od_config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER, (
        f"Expected PAGED_SCHEDULER, got {od_config.diffusion_kv_mode}"
    )
    assert od_config.enable_prefix_caching is True, (
        "Expected enable_prefix_caching to be True"
    )

    vllm_config = _build_minimal_vllm_config()
    backend = DiffusionKVModelRunnerBackend(
        vllm_config=vllm_config,
        od_config=od_config,
        device=torch.device("tpu"),
    )
    layer = Attention(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        causal=False,
        softmax_scale=HEAD_SIZE**-0.5,
        num_kv_heads=NUM_HEADS,
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        non_causal=True,
    )
    specs = backend.register_kv_cache_layers({"layer-0": (layer, spec)})
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8 * specs["layer-0"].page_size_bytes, shared_by=["layer-0"]
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer-0"], kv_cache_spec=specs["layer-0"])
        ],
    )
    backend.initialize_kv_cache(kv_cache_config)
    assert backend.kv_cache_config is not None, (
        "backend.initialize_kv_cache did not populate backend.kv_cache_config"
    )
    assert backend.kv_cache_config.num_blocks > 0, (
        f"Expected kv_cache_config.num_blocks > 0, "
        f"got {backend.kv_cache_config.num_blocks}"
    )

    torch.manual_seed(0)
    q = torch.randn(19, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    k = torch.randn(19, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    v = torch.randn(19, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)

    adapter = backend.paged_attention_adapter
    assert adapter is not None, (
        "Expected backend.paged_attention_adapter to be initialized, got None"
    )
    impl = layer.attn_backend.get_impl_cls()(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        num_kv_heads=NUM_HEADS,
    )
    q_tpu = q.to(device="tpu", dtype=torch.bfloat16)
    k_tpu = k.to(device="tpu", dtype=torch.bfloat16)
    v_tpu = v.to(device="tpu", dtype=torch.bfloat16)

    prefix_batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0", sequence_id=0, query_len=17, seq_len=17
            )
        ]
    )
    with adapter.activate(prefix_batch):
        prefix_ctx = adapter.prepare_layer_context(
            "layer-0", q_tpu[:17], k_tpu[:17], v_tpu[:17]
        )
        prefix_out = impl.forward_paged(prefix_ctx)

    suffix_batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=19,
                kv_start_pos=17,
            )
        ]
    )
    with adapter.activate(suffix_batch):
        suffix_ctx = adapter.prepare_layer_context(
            "layer-0", q_tpu[17:], k_tpu[17:], v_tpu[17:]
        )
        suffix_out = impl.forward_paged(suffix_ctx)

    prefix_ref = F.scaled_dot_product_attention(
        q[:17].unsqueeze(0).transpose(1, 2),
        k[:17].unsqueeze(0).transpose(1, 2),
        v[:17].unsqueeze(0).transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)[0]
    suffix_ref = F.scaled_dot_product_attention(
        q[17:].unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)[0]

    assert_parity(
        prefix_out.cpu(),
        prefix_ref,
        name="Diffusion paged KV prefill TPU bf16 vs CPU fp32",
        max_rel_l2=PAGED_KV_MAX_REL_L2,
    )
    assert_parity(
        suffix_out.cpu(),
        suffix_ref,
        name="Diffusion paged KV prefix-cached suffix TPU bf16 vs CPU fp32",
        max_rel_l2=PAGED_KV_MAX_REL_L2,
    )
