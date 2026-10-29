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
"""Torch custom-op boundaries for sparse attention and its native pool."""

import functools
import hashlib

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import TopKBackend
from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import (
    compact_pooled_qsa_step,
)

_backend_keys: dict[str, TopKBackend] = {}


def _op_name(kind: str, backend: TopKBackend | None, backend_key: str | None) -> str:
    if backend is None:
        if backend_key is not None:
            raise ValueError("A backend key requires an explicit topk_backend")
        return f"qwen4_exp::{kind}"
    if not isinstance(backend_key, str) or not backend_key:
        raise ValueError("Explicit topk_backend requires a stable topk_backend_key")
    previous = _backend_keys.setdefault(backend_key, backend)
    if previous != backend:
        raise ValueError("topk_backend_key already identifies a different backend")
    suffix = hashlib.sha256(backend_key.encode("utf-8")).hexdigest()
    return f"qwen4_exp::{kind}_{suffix}"


def build_compact_pooled_qsa_op(
    *, topk_backend: TopKBackend | None = None, topk_backend_key: str | None = None
):
    """Bind a selector under a stable persisted-graph identity.

    Explicit keys must identify the implementation/version and all static
    configuration, consistently across processes. Different backends must not
    reuse a key. The default selector retains its legacy operator name.
    The TPU compiler binds scheduling_group_namespace per FX call site; zero
    keeps direct native calls safe without cross-call scheduling annotations.
    """
    return _build_compact_pooled_qsa_op(topk_backend, topk_backend_key)


@functools.cache
def _build_compact_pooled_qsa_op(
    topk_backend: TopKBackend | None, topk_backend_key: str | None
):
    op = pallas.jax_op(
        _op_name("compact_pooled_qsa", topk_backend, topk_backend_key),
        functools.partial(compact_pooled_qsa_step, topk_backend=topk_backend),
        donate_argnums=(10, 11),  # pool, index_pool
    )

    @op.register_fake
    def fake(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        index_pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        request_distribution: torch.Tensor,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale=1.0,
        v_scale=1.0,
        scheduling_group_namespace=0,
    ):
        return (
            torch.empty_like(pool),
            torch.empty_like(index_pool),
            qgate.new_empty((qgate.shape[0], qgate.shape[-1] // 2)),
            positions.new_empty(
                (qgate.shape[0], token_topk + compress_ratio - 1), dtype=torch.int32
            ),
        )

    def run(
        qgate: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        index_qk: torch.Tensor,
        nq: torch.Tensor,
        nk: torch.Tensor,
        niq: torch.Tensor,
        nik: torch.Tensor,
        positions: torch.Tensor,
        trig: torch.Tensor,
        pool: torch.Tensor,
        index_pool: torch.Tensor,
        starts: torch.Tensor,
        context: torch.Tensor,
        ring_ids: torch.Tensor,
        main_table: torch.Tensor,
        compressed_table: torch.Tensor,
        request_distribution: torch.Tensor,
        compress_ratio: int,
        token_topk: int,
        norm_eps: float,
        rotary_dim: int,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        scheduling_group_namespace: int = 0,
    ):
        # The native QSA kernels consume positions and request/page metadata as int32.
        positions = positions.to(torch.int32)
        starts = starts.to(torch.int32)
        context = context.to(torch.int32)
        ring_ids = ring_ids.to(torch.int32)
        main_table = main_table.to(torch.int32)
        compressed_table = compressed_table.to(torch.int32)
        request_distribution = request_distribution.to(torch.int32)
        updated, index_updated, output, blocks = op(
            qgate,
            key,
            value,
            index_qk,
            nq,
            nk,
            niq,
            nik,
            positions,
            trig,
            pool,
            index_pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            request_distribution,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale,
            v_scale,
            scheduling_group_namespace,
        )
        pool.copy_(updated)
        index_pool.copy_(index_updated)
        return output, blocks

    return run
