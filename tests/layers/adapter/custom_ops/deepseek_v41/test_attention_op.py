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
"""Tests for DeepSeek-V4.1 Pallas sparse MLA + SWA custom-op bridges."""

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
    attention_op,
    dsv41_gather_sparse_fp8_kv,
    dsv41_sparse_mla_swa_attention,
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


def test_fake_gather_sparse_fp8_kv_shapes() -> None:
    """Verify FakeTensor abstract shapes for dsv41_gather_sparse_fp8_kv."""
    with FakeTensorMode() as mode:
        kv_lo = torch.empty((4, 8, 128), dtype=torch.int32)
        rope_lo = torch.empty((4, 2, 128), dtype=torch.int32)
        kv_hi = torch.empty((2, 8, 128), dtype=torch.int32)
        rope_hi = torch.empty((2, 2, 128), dtype=torch.int32)
        topk = torch.empty((6, 32), dtype=torch.int32)
        block_tables = torch.empty((2, 3), dtype=torch.int32)
        cu_seqlens_q = torch.empty((3,), dtype=torch.int32)

        nope, rope = dsv41_gather_sparse_fp8_kv(
            kv_lo,
            rope_lo,
            topk,
            block_tables,
            cu_seqlens_q,
            page_size=8,
            kv_pool_hi=kv_hi,
            rope_pool_hi=rope_hi,
        )
        assert isinstance(nope, FakeTensor) and nope.fake_mode is mode
        assert isinstance(rope, FakeTensor) and rope.fake_mode is mode
        assert nope.shape == (6, 32, 128) and rope.shape == (6, 8, 128)
        assert nope.dtype == torch.int32 and rope.dtype == torch.int32


def test_fake_sparse_mla_swa_attention_shapes() -> None:
    """Verify FakeTensor abstract shapes for dsv41_sparse_mla_swa_attention."""
    with FakeTensorMode() as mode:
        q = torch.empty((6, 8, 512), dtype=torch.bfloat16)
        nope = torch.empty((6, 32, 128), dtype=torch.int32)
        rope = torch.empty((6, 8, 128), dtype=torch.int32)
        topk = torch.empty((6, 32), dtype=torch.int32)
        sinks = torch.empty((8,), dtype=torch.float32)
        seq_lens = torch.empty((2,), dtype=torch.int32)
        cu_seqlens_q = torch.empty((3,), dtype=torch.int32)
        swa_kv = torch.empty((6, 16, 512), dtype=torch.bfloat16)
        swa_kv_2d = torch.empty((2 * 16 + 6, 512), dtype=torch.bfloat16)
        swa_min_pos = torch.empty((2,), dtype=torch.int32)

        out = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk,
            sinks,
            seq_lens,
            cu_seqlens_q,
            sm_scale=0.125,
            swa_window=16,
            swa_kv=swa_kv,
            kv_compress_ratio=2,
        )
        out_2d = dsv41_sparse_mla_swa_attention(
            q,
            nope,
            rope,
            topk,
            sinks,
            seq_lens,
            cu_seqlens_q,
            sm_scale=0.125,
            swa_window=16,
            swa_kv=swa_kv_2d,
            kv_compress_ratio=2,
            swa_min_pos=swa_min_pos,
        )
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert isinstance(out_2d, FakeTensor) and out_2d.fake_mode is mode
        assert out.shape == (6, 8, 512) and out_2d.shape == (6, 8, 512)
        assert out.dtype == torch.bfloat16 and out_2d.dtype == torch.bfloat16
        assert out is not q and out_2d is not q


def test_attention_partition_specs_and_wrappers(
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
        None,
        replicated_spec,
        replicated_spec,
    ) == attention_op.GATHER_SPARSE_FP8_KV_PARTITION_SPECS
    assert (
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        replicated_spec,
        None,
        None,
        replicated_spec,
        None,
        replicated_spec,
    ) == attention_op.SPARSE_MLA_SWA_PARTITION_SPECS

    calls: dict[str, tuple[object, ...]] = {}

    def fake_gather(
        kv_pool_lo: object,
        rope_pool_lo: object,
        topk_indices: object,
        block_tables: object,
        cu_seqlens_q: object,
        *,
        page_size: int,
        kv_pool_hi: object = None,
        rope_pool_hi: object = None,
    ) -> tuple[str, str]:
        calls["gather"] = (
            kv_pool_lo,
            rope_pool_lo,
            topk_indices,
            block_tables,
            cu_seqlens_q,
            page_size,
            kv_pool_hi,
            rope_pool_hi,
        )
        return ("nope", "rope")

    def fake_attn(
        q: object,
        gathered_nope: object,
        gathered_rope: object,
        topk_indices: object,
        sinks: object,
        seq_lens: object,
        cu_seqlens_q: object,
        *,
        sm_scale: float,
        swa_window: int,
        swa_kv: object = None,
        kv_compress_ratio: int = 1,
        swa_min_pos: object = None,
    ) -> str:
        calls["attn"] = (
            q,
            gathered_nope,
            gathered_rope,
            topk_indices,
            sinks,
            seq_lens,
            cu_seqlens_q,
            sm_scale,
            swa_window,
            swa_kv,
            kv_compress_ratio,
            swa_min_pos,
        )
        return "attn_out"

    monkeypatch.setattr(attention_op, "gather_sparse_fp8_kv", fake_gather)
    monkeypatch.setattr(attention_op, "sparse_mla_swa_attention", fake_attn)

    gather_args: tuple[Any, ...] = (
        "kv_lo",
        "rope_lo",
        "topk",
        "block_tables",
        "cu_seqlens_q",
        8,
        "kv_hi",
        "rope_hi",
    )
    assert attention_op._gather_sparse_fp8_kv_jax(*gather_args) == (
        "nope",
        "rope",
    )
    assert calls["gather"] == gather_args
    attn_args: tuple[Any, ...] = (
        "q",
        "gathered_nope",
        "gathered_rope",
        "topk",
        "sinks",
        "seq_lens",
        "cu_seqlens_q",
        0.25,
        16,
        "swa",
        2,
        "min_pos",
    )
    assert attention_op._sparse_mla_swa_attention_jax(*attn_args) == "attn_out"
    assert calls["attn"] == attn_args


@pytest.mark.tpu_test
def test_tpu_attention_custom_ops_end_to_end() -> None:
    """Execute dsv41 sparse MLA + SWA custom ops on physical TPU7x hardware."""
    import jax.numpy as jnp
    import numpy as np
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
        pack_fp8_kv_rows,
    )
    from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
        gather_sparse_fp8_kv,
        sparse_mla_swa_attention,
    )

    device = torch.device("tpu")
    gen = torch.Generator().manual_seed(17)
    raw_kv = (torch.randn((64, 512), dtype=torch.float32, generator=gen) * 0.2).to(
        torch.bfloat16
    )
    nope_j, rope_j = pack_fp8_kv_rows(
        jnp.asarray(raw_kv.float().numpy(), dtype=jnp.bfloat16), interpret=True
    )
    kv_all = torch.from_numpy(np.asarray(nope_j).reshape(4, 16, 128))
    rope_all = torch.from_numpy(np.asarray(rope_j).reshape(4, 4, 128))
    topk_cpu = torch.randint(-1, 28, (6, 36), dtype=torch.int32, generator=gen)
    bt_cpu = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    cu_cpu = torch.tensor([0, 2, 5], dtype=torch.int32)
    sl_cpu = torch.tensor([24, 30], dtype=torch.int32)
    q_cpu = (torch.randn((6, 12, 512), dtype=torch.float32, generator=gen) * 0.15).to(
        torch.bfloat16
    )
    sinks_cpu = torch.rand((12,), dtype=torch.float32, generator=gen) - 0.5
    swa_cpu = (torch.randn((6, 18, 512), dtype=torch.float32, generator=gen) * 0.2).to(
        torch.bfloat16
    )
    swa_2d_cpu = (
        torch.randn((2 * 18 + 6, 512), dtype=torch.float32, generator=gen) * 0.2
    ).to(torch.bfloat16)
    min_pos_cpu = torch.tensor([20, 26], dtype=torch.int32)

    nope, rope = dsv41_gather_sparse_fp8_kv(
        kv_all[:2].to(device),
        rope_all[:2].to(device),
        topk_cpu.to(device),
        bt_cpu.to(device),
        cu_cpu.to(device),
        page_size=16,
        kv_pool_hi=kv_all[2:].to(device),
        rope_pool_hi=rope_all[2:].to(device),
    )
    out = dsv41_sparse_mla_swa_attention(
        q_cpu.to(device),
        nope,
        rope,
        topk_cpu.to(device),
        sinks_cpu.to(device),
        sl_cpu.to(device),
        cu_cpu.to(device),
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_cpu.to(device),
        kv_compress_ratio=2,
    )
    out_min = dsv41_sparse_mla_swa_attention(
        q_cpu.to(device),
        nope,
        rope,
        topk_cpu.to(device),
        sinks_cpu.to(device),
        sl_cpu.to(device),
        cu_cpu.to(device),
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_cpu.to(device),
        kv_compress_ratio=2,
        swa_min_pos=min_pos_cpu.to(device),
    )
    out_2d = dsv41_sparse_mla_swa_attention(
        q_cpu.to(device),
        nope,
        rope,
        topk_cpu.to(device),
        sinks_cpu.to(device),
        sl_cpu.to(device),
        cu_cpu.to(device),
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_2d_cpu.to(device),
        kv_compress_ratio=2,
        swa_min_pos=min_pos_cpu.to(device),
    )
    ref_gn, ref_gr = gather_sparse_fp8_kv(
        jnp.asarray(kv_all[:2].numpy()),
        jnp.asarray(rope_all[:2].numpy()),
        jnp.asarray(topk_cpu.numpy()),
        jnp.asarray(bt_cpu.numpy()),
        jnp.asarray(cu_cpu.numpy()),
        page_size=16,
        kv_pool_hi=jnp.asarray(kv_all[2:].numpy()),
        rope_pool_hi=jnp.asarray(rope_all[2:].numpy()),
        interpret=True,
    )
    q_j = jnp.asarray(q_cpu.float().numpy(), dtype=jnp.bfloat16)
    tk_j = jnp.asarray(topk_cpu.numpy())
    sinks_j = jnp.asarray(sinks_cpu.numpy())
    sl_j = jnp.asarray(sl_cpu.numpy())
    cu_j = jnp.asarray(cu_cpu.numpy())
    swa_j = jnp.asarray(swa_cpu.float().numpy(), dtype=jnp.bfloat16)
    swa_2d_j = jnp.asarray(swa_2d_cpu.float().numpy(), dtype=jnp.bfloat16)
    min_pos_j = jnp.asarray(min_pos_cpu.numpy())
    ref_out = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_j,
        kv_compress_ratio=2,
        interpret=True,
    )
    ref_min = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_j,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_j,
        interpret=True,
    )
    ref_2d = sparse_mla_swa_attention(
        q_j,
        ref_gn,
        ref_gr,
        tk_j,
        sinks_j,
        sl_j,
        cu_j,
        sm_scale=0.0625,
        swa_window=8,
        swa_kv=swa_2d_j,
        kv_compress_ratio=2,
        swa_min_pos=min_pos_j,
        interpret=True,
    )
    np.testing.assert_array_equal(nope.cpu().numpy(), np.asarray(ref_gn))
    np.testing.assert_array_equal(rope.cpu().numpy(), np.asarray(ref_gr))
    np.testing.assert_allclose(
        out.cpu().float().numpy(),
        np.asarray(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        out_min.cpu().float().numpy(),
        np.asarray(ref_min, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        out_2d.cpu().float().numpy(),
        np.asarray(ref_2d, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    assert not np.allclose(
        out_min.cpu().float().numpy()[:5],
        out.cpu().float().numpy()[:5],
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_array_equal(out.cpu().float().numpy()[5:], 0.0)
    np.testing.assert_array_equal(out_2d.cpu().float().numpy()[5:], 0.0)
