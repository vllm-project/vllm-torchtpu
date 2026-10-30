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
"""Tests for DeepSeek-V4.1 Pallas indexer custom-op bridges."""

from collections.abc import Iterator
from typing import Any

import jax
import pytest
import torch
from jax.sharding import (
    AbstractDevice,
    AbstractMesh,
    PartitionSpec,
    use_abstract_mesh,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    dsv41_paged_indexer_topk,
    indexer_op,
)


@pytest.fixture(autouse=True)
def _tpu7x_abstract_mesh() -> Iterator[None]:
    """Bind a TPU7x abstract mesh on CPU so Pallas kernels query TPU7x info."""
    if jax.default_backend() != "tpu":
        mesh = AbstractMesh((), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
        with use_abstract_mesh(mesh):
            yield
    else:
        yield


def test_fake_paged_indexer_topk_shapes() -> None:
    """Verify FakeTensor abstract shapes for dsv41_paged_indexer_topk."""
    with FakeTensorMode() as mode:
        q = torch.empty((6, 4, 128), dtype=torch.bfloat16)
        weights = torch.empty((6, 4, 1), dtype=torch.float32)
        pool_lo = torch.empty((4, 8, 128), dtype=torch.int32)
        pool_hi = torch.empty((2, 8, 128), dtype=torch.int32)
        block_tables = torch.empty((2, 2), dtype=torch.int32)
        seq_lens = torch.empty((2,), dtype=torch.int32)
        cu_seqlens_q = torch.empty((3,), dtype=torch.int32)
        candidate_indices = torch.empty((6, 8), dtype=torch.int32)

        out = dsv41_paged_indexer_topk(
            q,
            weights,
            pool_lo,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            topk=16,
            k_pool_hi=pool_hi,
            candidate_indices=candidate_indices,
            compress_ratio=2,
        )
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert out.shape == (6, 16)
        assert out.dtype == torch.int32


def test_indexer_partition_specs_and_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify partition-spec constants and thin JAX wrapper argument forwarding."""
    replicated_spec = PartitionSpec()
    assert (
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        None,
        replicated_spec,
        replicated_spec,
        None,
    ) == indexer_op.PAGED_INDEXER_TOPK_PARTITION_SPECS

    calls: dict[str, tuple[object, ...]] = {}

    def fake_topk(
        q: object,
        weights: object,
        k_pool_lo: object,
        block_tables: object,
        seq_lens: object,
        cu_seqlens_q: object,
        *,
        topk: int,
        k_pool_hi: object = None,
        candidate_indices: object = None,
        compress_ratio: int = 1,
    ) -> str:
        calls["topk"] = (
            q,
            weights,
            k_pool_lo,
            block_tables,
            seq_lens,
            cu_seqlens_q,
            topk,
            k_pool_hi,
            candidate_indices,
            compress_ratio,
        )
        return "topk_out"

    monkeypatch.setattr(indexer_op, "paged_indexer_topk", fake_topk)

    topk_args: tuple[Any, ...] = (
        "q",
        "weights",
        "pool_lo",
        "block_tables",
        "seq_lens",
        "cu_seqlens_q",
        16,
        "pool_hi",
        "candidates",
        2,
    )
    assert indexer_op._paged_indexer_topk_jax(*topk_args) == "topk_out"
    assert calls["topk"] == topk_args


@pytest.mark.tpu_test
def test_tpu_indexer_custom_ops_end_to_end() -> None:
    """Execute dsv41 indexer custom op on physical TPU7x hardware."""
    import jax.numpy as jnp
    import numpy as np
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer import (
        paged_indexer_topk,
    )

    device = torch.device("tpu")
    gen = torch.Generator().manual_seed(77)
    q_cpu = torch.randn((4, 32, 128), dtype=torch.float32, generator=gen).to(
        torch.bfloat16
    )
    weights_cpu = torch.rand((4, 32, 1), dtype=torch.float32, generator=gen) + 0.1
    pool_lo_cpu = (
        torch.randn((2, 64, 256), dtype=torch.float32, generator=gen)
        .to(torch.bfloat16)
        .view(torch.int32)
    )
    pool_hi_cpu = (
        torch.randn((2, 64, 256), dtype=torch.float32, generator=gen)
        .to(torch.bfloat16)
        .view(torch.int32)
    )
    block_tables_cpu = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([20, 256], dtype=torch.int32)
    cu_seqlens_q_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)

    topk_idx = dsv41_paged_indexer_topk(
        q_cpu.to(device),
        weights_cpu.to(device),
        pool_lo_cpu.to(device),
        block_tables_cpu.to(device),
        seq_lens_cpu.to(device),
        cu_seqlens_q_cpu.to(device),
        topk=16,
        k_pool_hi=pool_hi_cpu.to(device),
        compress_ratio=2,
    ).cpu()
    ref_idx = np.asarray(
        paged_indexer_topk(
            jnp.asarray(q_cpu.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(weights_cpu.numpy()),
            jnp.asarray(pool_lo_cpu.numpy()),
            jnp.asarray(block_tables_cpu.numpy()),
            jnp.asarray(seq_lens_cpu.numpy()),
            jnp.asarray(cu_seqlens_q_cpu.numpy()),
            topk=16,
            k_pool_hi=jnp.asarray(pool_hi_cpu.numpy()),
            compress_ratio=2,
            interpret=True,
        )
    )
    assert topk_idx.shape == (4, 16) and topk_idx.dtype == torch.int32
    np.testing.assert_array_equal(
        np.sort(topk_idx.numpy(), axis=1), np.sort(ref_idx, axis=1)
    )
