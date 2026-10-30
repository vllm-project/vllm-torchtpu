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
"""Default KV page size for dense attention models.

KV offload moves one page of one layer per transfer, so the page size sets the
transfer size and the number of transfers per reloaded prefix.
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend


def _page_size(max_model_len: int) -> int:
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len))
    return PallasAttentionBackend.get_page_size(config)


@pytest.mark.parametrize(
    "max_model_len,page_size",
    [(128, 16), (256, 16), (1024, 64), (2048, 128), (4096, 256), (8192, 256)],
)
def test_short_context_page_size(max_model_len, page_size):
    assert _page_size(max_model_len) == page_size


@pytest.mark.parametrize("max_model_len", [8193, 16384, 131072, 1 << 20])
def test_long_context_uses_128_token_pages(max_model_len):
    assert _page_size(max_model_len) == 128
