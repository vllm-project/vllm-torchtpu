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
"""QSA cache writes after a Qwen3.8 whole-step megakernel call.

The whole-step kernel attends each QSA layer's own step tokens from VMEM and writes no
cache. Afterwards, in the same JAX callable, every QSA layer's step tokens are written
through the stock writer sequence (`pooled_qsa.qsa_cache_writes`, the code
`_pooled_qsa` runs) from the kernel's k / v / raw-indexer-key projections, so the pool
holds the bytes the stock layers would have left for the same projections: main K/V
(FP8-quantised, paired head slot), the completed groups' compressed keys (packed slot)
and the ring rows + tags, verify drafts included.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import qsa_cache_writes
from vllm_torchtpu.kernels.qwen4_exp.qsa import produce_kv


@dataclasses.dataclass(frozen=True)
class QSALayerCache:
    """Static per-QSA-layer cache addressing (`Qwen4ExpQSA.bind_pool` keywords)."""

    compressed_slot: int = 0
    main_slot: int = -1
    k_scale: float = 1.0
    v_scale: float = 1.0


def wholestep_qsa_writes(
    pool: Any,
    rows: list[tuple[Any, Any, Any]],
    norms: list[tuple[Any, Any]],
    tables: list[tuple[Any, Any, Any]],
    layers: list[QSALayerCache],
    positions: Any,
    trig: Any,
    starts: Any,
    context: Any,
    *,
    compress_ratio: int,
    norm_eps: float,
    rotary_dim: int,
    max_query_len: int,
    num_requests: int | None = None,
) -> Any:
    """Write every QSA layer's step tokens, in layer order; returns the pool.

    rows[i]: layer i's `(key, value, index_k)` projections `[T, kv*D]`, `[T, kv*D]`,
      `[T, index_dim]` (the raw indexer key: `index_qk[:, -index_dim:]`).
    norms[i]: `(k_norm, index_k_norm)` weights. tables[i]: the runner's flat
      `(ring_ids, main_table, compressed_table)` of layer i's cache groups.
    num_requests: the kernel's static request count (`M // T`). Requests past it are idle
      padding (the driver calls the kernel only for steps that fit its bucket), so they are
      cropped before the writers rather than read, merged and written back as no-ops.
    """
    n = len(context)
    nreq = n if num_requests is None else min(n, num_requests)
    context, starts = context[:nreq], starts[: nreq + 1]
    for (key, value, index_k), (nk, nik), (ring, main, comp), lay in zip(
        rows, norms, tables, layers, strict=True
    ):
        k, v = produce_kv(key, value, nk, positions, trig, norm_eps, rotary_dim)
        pool, _, _, _ = qsa_cache_writes(
            pool,
            k,
            v,
            index_k,
            positions,
            nik,
            trig,
            starts,
            context,
            ring.reshape(n, -1)[:nreq, 0],
            main.reshape(n, -1)[:nreq],
            comp.reshape(n, -1)[:nreq],
            compress_ratio=compress_ratio,
            norm_eps=norm_eps,
            rotary_dim=rotary_dim,
            k_scale=lay.k_scale,
            v_scale=lay.v_scale,
            compressed_slot=lay.compressed_slot,
            main_slot=lay.main_slot,
            max_query_len=max_query_len,
        )
    return pool
