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
"""TPU component tests for PromptEmbedCache."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

PROMPT_CACHE_MAX_REL_L2 = 2.0e-2


class _TinyPromptEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(token_ids))


class _TinyPromptPipeline:
    def __init__(
        self, encoder: _TinyPromptEncoder, device: str, dtype: torch.dtype
    ) -> None:
        self.encoder = encoder
        self.device = device
        self.dtype = dtype
        self.calls = 0

    def encode_prompt(
        self, prompt: str, extra_tensor: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.calls += 1
        token_ids = torch.tensor(
            [[ord(c) % 32 for c in prompt[:6]]],
            device=self.device,
            dtype=torch.long,
        )
        out = self.encoder(token_ids)
        if extra_tensor is not None:
            out = out + 0.01 * extra_tensor.to(device=self.device, dtype=self.dtype)
        return out


def test_prompt_embedding_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt-Embedding Cache

    - TPU path: Resolve OMNI_DIFFUSION_PROMPT_EMBED_CACHE=1 and
      OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE=2 via
      resolve_prompt_embed_cache_config() and wrap a pipeline whose
      encode_prompt runs a tiny nn.Embedding + nn.Linear encoder on "tpu"
      (bfloat16) via install_prompt_embed_cache.
    - Setup check: Assert pipeline._prompt_embed_cache has max_size == 2,
      repeated prompt records misses == 1 and hits == 1 with encoder call
      count == 1, tensor kwargs record bypassed == 1, and a third prompt
      evicts the oldest entry while keeping size == 2.
    - Outcome check: Assert cached "tpu" embedding tensor is bitwise equal
      (torch.equal) to the miss output and matches CPU float32 encoder
      reference within bf16 relative-L2 tolerance.
    """
    monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE", "1")
    monkeypatch.setenv("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE", "2")
    enabled, max_size = resolve_prompt_embed_cache_config()
    assert enabled is True, f"Expected enabled=True, got {enabled}"
    assert max_size == 2, f"Expected max_size=2, got {max_size}"

    torch.manual_seed(0)
    base_encoder = _TinyPromptEncoder()
    tpu_encoder = copy.deepcopy(base_encoder).to(device="tpu")
    tpu_encoder.proj = tpu_encoder.proj.to(dtype=torch.bfloat16)
    tpu_encoder.embed = tpu_encoder.embed.to(dtype=torch.bfloat16)
    cpu_encoder = copy.deepcopy(base_encoder).to(device="cpu", dtype=torch.float32)

    tpu_pipe = _TinyPromptPipeline(tpu_encoder, "tpu", torch.bfloat16)
    cpu_pipe = _TinyPromptPipeline(cpu_encoder, "cpu", torch.float32)

    cache = install_prompt_embed_cache(tpu_pipe, max_size=max_size, enabled=enabled)
    assert cache is not None, (
        "Expected PromptEmbedCache instance from install_prompt_embed_cache, got None"
    )
    assert cache.max_size == 2, (
        f"Expected PromptEmbedCache max_size == 2, got {cache.max_size}"
    )

    with torch.no_grad():
        first_out = tpu_pipe.encode_prompt("a red cat")
        second_out = tpu_pipe.encode_prompt("a red cat")
        cpu_ref = cpu_pipe.encode_prompt("a red cat")

    stats = cache.stats()
    assert stats["misses"] == 1, f"Expected misses == 1, got {stats['misses']}"
    assert stats["hits"] == 1, f"Expected hits == 1, got {stats['hits']}"
    assert tpu_pipe.calls == 1, f"Expected 1 encoder call, got {tpu_pipe.calls}"
    assert first_out.device.type == "tpu", (
        f"Expected cached tensor on tpu, got {first_out.device.type}"
    )
    assert torch.equal(second_out.cpu(), first_out.cpu()), (
        "Cached TPU prompt embedding must be bitwise equal to miss output"
    )
    assert_parity(
        second_out.cpu(),
        cpu_ref,
        name="PromptEmbedCache TPU bf16 vs CPU fp32",
        max_rel_l2=PROMPT_CACHE_MAX_REL_L2,
    )

    with torch.no_grad():
        tpu_pipe.encode_prompt(
            "a red cat",
            extra_tensor=torch.ones(1, 6, 16, device="tpu", dtype=torch.bfloat16),
        )
        assert cache.stats()["bypassed"] == 1, (
            f"Expected bypassed == 1 for tensor kwarg, got {cache.stats()['bypassed']}"
        )
        tpu_pipe.encode_prompt("a blue dog")
        tpu_pipe.encode_prompt("a green bird")

    evict_stats = cache.stats()
    assert evict_stats["size"] == 2, (
        f"Expected LRU cache size == 2 after 3rd prompt, got {evict_stats['size']}"
    )
    assert evict_stats["misses"] == 3, (
        f"Expected misses == 3 after 3rd prompt, got {evict_stats['misses']}"
    )
