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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# tools/megakernel/k3.py
"""Kimi-K3 fixtures for the standalone tools.

Stand-in unified pools laid out as vLLM's, their KV cache groups, and the fixed
prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from .. import config, layout
from ..decode import KimiDecoder
from ..program import kernel_modules
from ..runtime_shim import program as mk
from ..runtime_shim.config import MegakernelEngineConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..decode_ops import DecodeOps

PAGE, DCP = 256, 32
LOGICAL = PAGE * DCP
CACHE_DTYPE = "auto"
_ROPE_START = 576
_BASE_TILE_TOKENS = 256
_NUM_MLA_POOLS = 24
_KDA_GROUPS = 3
_DEFAULT_MAX_BATCHED_TOKENS = 512
_DEFAULT_SPEC_TOKENS = 7


def configure_cache(dtype: str) -> None:
    """Configure standalone fixtures from the same cache option as serving."""
    page_tokens = layout.cache_layout(dtype).tokens_per_block
    globals().update(
        PAGE=page_tokens,
        LOGICAL=page_tokens * DCP,
        CACHE_DTYPE=dtype,
    )


def logical() -> int:
    """Return tokens per DCP block under the installed kernel config's DCP group."""
    return PAGE * config.active().dcp_group


def fixed_prompt(count: int, checkpoint: str) -> tuple[list[int], object]:
    """Return ``(tokens, tokenizer)`` from the first ``count`` tokens."""
    (load,) = kernel_modules("load")
    tokenizer = load.load_tokenizer(checkpoint)
    text = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path(checkpoint).glob("*.py"))
    )
    return list(tokenizer.encode(text, disallowed_special=())[:count]), tokenizer


class Pools:
    """24 unified pools laid out per ``cache_layout(CACHE_DTYPE)`` and KV groups."""

    def __init__(
        self,
        num_blocks: int,
        mla_width: int,
        state_width: int,
        *,
        pattern: bool = False,
    ) -> None:
        """Allocate ``num_blocks`` stand-in unified pool blocks on TPU."""
        cache = layout.cache_layout(CACHE_DTYPE)
        dtype = getattr(torch, cache.dtype)
        if pattern:
            generator = torch.Generator().manual_seed(0)
            page = (
                torch.randn((1, 128, 2, 640), dtype=torch.float32, generator=generator)
                * 0.5
            )
            page[..., _ROPE_START:] = 0
            page = page.to(torch.bfloat16)
            page = (
                page.reshape(1, _BASE_TILE_TOKENS, 640)
                .repeat(1, cache.tokens_per_block // _BASE_TILE_TOKENS, 1)
                .reshape(1, *cache.block_shape)
                .to(dtype)
            )
            self.tensors = [
                page.to("tpu").repeat(num_blocks, 1, 1, 1)
                for _ in range(_NUM_MLA_POOLS)
            ]
        else:
            self.tensors = [
                torch.zeros((num_blocks, *cache.block_shape), dtype=dtype).to("tpu")
                for _ in range(_NUM_MLA_POOLS)
            ]
        mla_names = [
            f"model.layers.{layer}.self_attn.mla_attn.attn" for layer in layout.MLA
        ]
        kda_names = [f"model.layers.{layer}.self_attn" for layer in layout.KDA]
        self.caches: dict[str, torch.Tensor] = {
            name: self.tensors[m] for m, name in enumerate(mla_names)
        }
        self.caches.update(
            {name: self.tensors[k // _KDA_GROUPS] for k, name in enumerate(kda_names)}
        )
        self.groups = [mk.KVGroup(tuple(mla_names), logical(), mla_width)]
        self.groups += [
            mk.KVGroup(
                tuple(n for k, n in enumerate(kda_names) if k % _KDA_GROUPS == g),
                PAGE,
                state_width,
            )
            for g in range(_KDA_GROUPS)
        ]
        self.num_blocks = num_blocks
        self.mla_width, self.state_width = mla_width, state_width
        self._next = 1

    def scheduled(self, table: Sequence[int] | np.ndarray, index: int) -> np.ndarray:
        """Return MLA ``table`` offset for the ``index``-th request's DP group."""
        group = index % config.active().attention_dp
        arr = np.asarray(table, np.int32)
        return np.where(arr > 0, arr + group * self.num_blocks, 0).astype(np.int32)

    def fresh(self, n: int) -> list[int]:
        """Allocate and return ``n`` fresh pool block indices."""
        out = list(range(self._next, self._next + n))
        self._next += n
        return out

    def tables(
        self,
        mla: Sequence[int] | np.ndarray,
        kda: Sequence[Sequence[int] | np.ndarray],
    ) -> tuple[np.ndarray, ...]:
        """Pad ``mla`` and ``kda`` block lists to table width and return tuple."""

        def padded(table: Sequence[int] | np.ndarray, width: int) -> np.ndarray:
            """Right-pad ``table`` with zeros to ``width`` int32 entries."""
            items = [int(x) for x in table]
            return np.asarray(items + [0] * (width - len(items)), np.int32)

        return (
            padded(mla, self.mla_width),
            *(padded(t, self.state_width) for t in kda),
        )


def engine_config(
    role: str,
    *,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    **spec_kwargs: int | tuple[tuple[int, int, int], ...] | None,
) -> MegakernelEngineConfig:
    """Build ``MegakernelEngineConfig`` for the TP32 / DCP32 deployment."""
    dp = config.active().attention_dp
    num_spec = cast("int", spec_kwargs.get("num_speculative_tokens", 0))
    sched = cast(
        "tuple[tuple[int, int, int], ...] | None",
        spec_kwargs.get("num_speculative_tokens_per_batch_size"),
    )
    return MegakernelEngineConfig(
        role=role,
        tensor_parallel_size=DCP,
        decode_context_parallel_size=DCP // dp,
        block_size=PAGE,
        kv_cache_dtype=CACHE_DTYPE,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=num_spec,
        num_speculative_tokens_per_batch_size=sched,
        attention_dp=dp,
    )


def decoder(
    ops: DecodeOps,
    pools: Pools,
    *,
    max_num_seqs: int,
    max_model_len: int,
    **spec_options: object,
) -> KimiDecoder:
    """Build and bind a ``KimiDecoder`` over ``ops`` and fixture ``pools``."""
    draft = cast("Path | str | None", spec_options.get("draft"))
    schedule = cast(
        "tuple[tuple[int, int, int], ...] | None",
        spec_options.get("schedule"),
    )
    cfg = engine_config(
        "decode",
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=_DEFAULT_MAX_BATCHED_TOKENS,
        num_speculative_tokens=_DEFAULT_SPEC_TOKENS if schedule else 0,
        num_speculative_tokens_per_batch_size=schedule,
    )
    dec = KimiDecoder(cfg, ops, draft if schedule else None, tp_rank=ops.rank)
    dec.bind_kv_caches(pools.caches, pools.groups)
    return dec


def request(
    req_id: str,
    tokens: Sequence[int] | np.ndarray,
    table: np.ndarray,
    sampling: tuple[float, float, int, int] = (0.0, 1.0, 0, 0),
    computed: int = 0,
) -> mk.RequestInput:
    """Build a decode ``RequestInput`` scheduling ``tokens`` after ``computed``."""
    tok_arr = np.asarray(tokens, np.int32)
    end = computed + len(tok_arr)
    token_ids = np.zeros(end, np.int32)
    token_ids[computed:] = tok_arr
    null = np.zeros(1, np.int32)
    return mk.RequestInput(
        req_id=req_id,
        computed=computed,
        scheduled=len(tok_arr),
        token_ids=token_ids,
        num_tokens=end,
        num_prompt_tokens=end,
        block_tables=(np.asarray(table, np.int32), null, null, null),
        sampling=mk.SamplingSpec(*sampling),
    )
