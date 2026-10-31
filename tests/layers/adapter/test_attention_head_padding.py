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
"""q/k/v padding against the KV page a Pallas attention layer writes.

The kernel is picked from the query width (`q.shape[-1] == 64` selects the
head_dim=64 kernel), so padding a head_dim=64 query to its own packed page
width (128) silently switches to the v3 kernel and its layout check fails.
Padding is only for a page defined by another spec.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
    kv_page_head_padding,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


def _page(num_kv_heads: int, head_size: int, dtype: torch.dtype):
    return PallasAttentionBackend.get_kv_cache_shape(
        1, 1, num_kv_heads, head_size, dtype
    )[2:]


@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize(
    "num_kv_heads,head_size",
    [(8, 64), (2, 64), (8, 128), (4, 256), (8, 80)],
)
def test_native_page_needs_no_padding(num_kv_heads, head_size, dtype):
    page = _page(num_kv_heads, head_size, dtype)

    assert kv_page_head_padding(page, page, page[-1], head_size) == 0


def test_hd64_native_page_is_wider_than_the_head():
    # The packed head_dim=64 page is 128 lanes wide, twice the head width.
    assert _page(8, 64, BF16)[-1] == 128


@pytest.mark.parametrize("dtype", [BF16, FP8])
@pytest.mark.parametrize(
    "draft_heads,draft_head_size,target_heads,target_head_size,expected_padding",
    [
        (8, 64, 8, 128, 64),  # head_dim=64 draft in a v3 head_dim=128 pool
        (4, 128, 4, 256, 128),  # narrower draft in a wider target pool
    ],
)
def test_page_from_another_spec_is_padded_to_its_width(
    draft_heads,
    draft_head_size,
    target_heads,
    target_head_size,
    expected_padding,
    dtype,
):
    pool = _page(target_heads, target_head_size, dtype)
    native = _page(draft_heads, draft_head_size, dtype)

    assert kv_page_head_padding(pool, native, pool[-1], draft_head_size) == (
        expected_padding
    )


def test_page_no_wider_than_the_head_is_never_padded():
    pool = _page(8, 128, BF16)
    native = _page(4, 128, BF16)

    assert kv_page_head_padding(pool, native, 128, 128) == 0


@pytest.fixture
def cpu_vllm_config():
    """A current vLLM config pinned to CPU.

    CPU CI does not always register torch's TPU device type, so the default
    `VllmConfig()` device detection can fail there.
    """
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _forward_query_width(head_size: int, pool_heads: int, pool_head_size: int):
    """Run `forward` and return the width of the query the kernel receives."""
    num_heads = 4
    num_kv_heads = 4
    q_len = 3
    impl = PallasAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bf16",
        kv_cache_quantized_dtype=None,
    )
    kv_cache = torch.zeros(
        PallasAttentionBackend.get_kv_cache_shape(
            2, 16, pool_heads, pool_head_size, BF16
        ),
        dtype=BF16,
    )
    query = torch.ones(q_len, num_heads * head_size, dtype=BF16)
    key = torch.ones(q_len, num_kv_heads * head_size, dtype=BF16)
    seen = {}

    def fake_kernel(_kv_cache, q, *_args):
        seen["width"] = q.shape[-1]
        return torch.ones_like(q)

    metadata = AttentionMetadata(
        input_positions=torch.arange(q_len, dtype=torch.int32),
        block_tables=torch.zeros(1, dtype=torch.int32),
        seq_lens=torch.tensor([q_len], dtype=torch.int32),
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    layer = SimpleNamespace()
    with (
        patch.object(
            PallasAttentionBackendImpl, "_build_rpa_kernel", return_value=fake_kernel
        ),
        patch(
            "vllm_torchtpu.layers.adapter.attention.synchronize_tensors",
            lambda *_args, **_kwargs: None,
        ),
        set_vllm_model_wrapper_context(mesh=SimpleNamespace()),
    ):
        out = impl.forward(
            query=query,
            key=key,
            value=key.clone(),
            kv_cache=kv_cache,
            attn_metadata=metadata,
            layer=layer,
        )
    assert out.shape == query.shape
    return seen["width"]


def test_forward_keeps_hd64_query_on_its_own_page(cpu_vllm_config):
    # Padding to 128 here would pick the v3 kernel for a head_dim=64 layer.
    assert _forward_query_width(64, pool_heads=4, pool_head_size=64) == 64


def test_forward_pads_hd64_query_in_a_hd128_pool(cpu_vllm_config):
    # A head_dim=64 draft layer writing into a head_dim=128 target's pool.
    assert _forward_query_width(64, pool_heads=4, pool_head_size=128) == 128
