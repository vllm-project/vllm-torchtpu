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
"""Tests for DeepSeek-V4.1 Pallas cache custom-op bridges."""

from collections.abc import Iterator

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
    cache_op,
    dsv41_gather_cache_rows,
    dsv41_pack_fp8_kv_rows,
    dsv41_write_sub_rows,
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


@pytest.mark.parametrize("slice_words", [32, 64, 128])
@pytest.mark.parametrize("page_tokens", [8, 16])
@pytest.mark.parametrize(
    "after_dtype", [None, torch.int32, torch.bfloat16, torch.uint8]
)
def test_fake_write_sub_rows_shapes(
    slice_words: int,
    page_tokens: int,
    after_dtype: torch.dtype | None,
) -> None:
    """Verify FakeTensor shape/dtype propagation, after= dispatch, and copy_ pattern."""
    with FakeTensorMode() as mode:
        cache = torch.empty((16, 128), dtype=torch.int32)
        updates = torch.empty((8, slice_words), dtype=torch.int32)
        slots = torch.empty((8,), dtype=torch.int32)
        after = (
            None if after_dtype is None else torch.empty((4, 128), dtype=after_dtype)
        )
        out = dsv41_write_sub_rows(
            cache,
            updates,
            slots,
            slot_offset=64,
            after=after,
            page_tokens=page_tokens,
        )
        assert isinstance(out, FakeTensor) and out.fake_mode is mode
        assert out.shape == (16, 128)
        assert out.dtype == torch.int32
        assert out is not cache
        cache.copy_(out)
        assert cache.shape == (16, 128)


def test_fake_pack_and_gather_shapes() -> None:
    """Verify FakeTensor abstract shapes for pack_fp8_kv_rows and gather_cache_rows."""
    with FakeTensorMode() as mode:
        latent = torch.empty((10, 512), dtype=torch.bfloat16)
        nope, rope = dsv41_pack_fp8_kv_rows(latent)
        assert isinstance(nope, FakeTensor) and nope.fake_mode is mode
        assert isinstance(rope, FakeTensor) and rope.fake_mode is mode
        assert nope.shape == (10, 128) and rope.shape == (10, 32)
        assert nope.dtype == torch.int32 and rope.dtype == torch.int32

        pool_lo = torch.empty((8, 128), dtype=torch.int32)
        pool_hi = torch.empty((4, 128), dtype=torch.int32)
        indices = torch.empty((16,), dtype=torch.int32)
        gathered_1bank = dsv41_gather_cache_rows(pool_lo, indices)
        gathered_2bank = dsv41_gather_cache_rows(pool_lo, indices, pool_hi=pool_hi)
        assert gathered_1bank.shape == (16, 128) and gathered_1bank.dtype == torch.int32
        assert gathered_2bank.shape == (16, 128) and gathered_2bank.dtype == torch.int32


def test_cache_partition_specs_and_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify partition-spec constants and thin JAX wrapper argument forwarding."""
    replicated_spec = PartitionSpec()
    assert (
        replicated_spec,
        replicated_spec,
        replicated_spec,
        None,
        None,
    ) == cache_op.WRITE_SUB_ROWS_PARTITION_SPECS
    assert (
        replicated_spec,
        replicated_spec,
        replicated_spec,
        None,
        None,
        replicated_spec,
    ) == cache_op.WRITE_SUB_ROWS_AFTER_PARTITION_SPECS
    assert (replicated_spec,) == cache_op.PACK_FP8_KV_PARTITION_SPECS
    assert (
        replicated_spec,
        replicated_spec,
        replicated_spec,
    ) == cache_op.GATHER_CACHE_ROWS_PARTITION_SPECS

    calls: dict[str, tuple[object, ...]] = {}

    def fake_write(
        cache_words: object,
        updates: object,
        slot_mapping: object,
        *,
        slot_offset: int = 0,
        after: object = None,
        page_tokens: int = cache_op.DEFAULT_PAGE_TOKENS,
    ) -> str:
        calls["write"] = (
            cache_words,
            updates,
            slot_mapping,
            slot_offset,
            after,
            page_tokens,
        )
        return "write_out"

    def fake_pack(latent: object) -> tuple[str, str]:
        calls["pack"] = (latent,)
        return ("nope", "rope")

    def fake_gather(pool_lo: object, indices: object, *, pool_hi: object = None) -> str:
        calls["gather"] = (pool_lo, indices, pool_hi)
        return "gather_out"

    monkeypatch.setattr(cache_op, "write_sub_rows", fake_write)
    monkeypatch.setattr(cache_op, "pack_fp8_kv_rows", fake_pack)
    monkeypatch.setattr(cache_op, "gather_cache_rows", fake_gather)

    assert (
        cache_op._write_sub_rows_jax("cache", "updates", "slots", 16)  # type: ignore[arg-type]
        == "write_out"
    )
    assert calls["write"] == ("cache", "updates", "slots", 16, None, 16)
    assert (
        cache_op._write_sub_rows_jax("cache", "updates", "slots", 16, 8)  # type: ignore[arg-type]
        == "write_out"
    )
    assert calls["write"] == ("cache", "updates", "slots", 16, None, 8)
    assert (
        cache_op._write_sub_rows_after_jax(
            "cache",
            "updates",
            "slots",
            16,
            8,
            "dep",  # type: ignore[arg-type]
        )
        == "write_out"
    )
    assert calls["write"] == ("cache", "updates", "slots", 16, "dep", 8)
    assert cache_op._pack_fp8_kv_rows_jax("latent") == ("nope", "rope")  # type: ignore[arg-type]
    assert calls["pack"] == ("latent",)
    assert cache_op._gather_cache_rows_jax("lo", "indices", "hi") == "gather_out"  # type: ignore[arg-type]
    assert calls["gather"] == ("lo", "indices", "hi")


@pytest.mark.tpu_test
def test_tpu_cache_custom_ops_end_to_end() -> None:
    """Execute dsv41 cache custom ops on physical TPU7x hardware."""
    import jax.numpy as jnp
    import numpy as np
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
        pack_fp8_kv_rows,
    )

    device = torch.device("tpu")
    cache = torch.zeros((8, 128), dtype=torch.int32, device=device)
    updates = torch.arange(8 * 32, dtype=torch.int32, device=device).reshape(8, 32)
    slots = torch.tensor(
        [16, 17, 18, 19, -1, 100, 16, 20], dtype=torch.int32, device=device
    )
    cache.copy_(dsv41_write_sub_rows(cache, updates, slots, slot_offset=16))
    cache_cpu = cache.cpu()
    updates_cpu = updates.cpu()
    assert torch.equal(cache_cpu[0, 0:32], updates_cpu[6])
    assert torch.equal(cache_cpu[0, 32:64], updates_cpu[1])
    assert torch.equal(cache_cpu[0, 64:96], updates_cpu[2])
    assert torch.equal(cache_cpu[0, 96:128], updates_cpu[3])
    assert torch.equal(cache_cpu[1, 0:32], updates_cpu[7])
    assert torch.all(cache_cpu[1, 32:] == 0)

    gen = torch.Generator().manual_seed(42)
    latent_f32 = torch.randn((8, 512), dtype=torch.float32, generator=gen)
    latent_cpu = latent_f32.to(torch.bfloat16)
    nope_w, rope_w = dsv41_pack_fp8_kv_rows(latent_cpu.to(device))
    nope_cpu, rope_cpu = nope_w.cpu(), rope_w.cpu()
    ref_nope, ref_rope = pack_fp8_kv_rows(
        jnp.asarray(latent_cpu.float().numpy(), dtype=jnp.bfloat16), interpret=True
    )
    np.testing.assert_array_equal(nope_cpu.numpy(), np.asarray(ref_nope))
    np.testing.assert_array_equal(rope_cpu.numpy(), np.asarray(ref_rope))

    indices = torch.tensor(
        [0, 1, 8, -1, 2, 3, 9, 100], dtype=torch.int32, device=device
    )
    gathered_tpu = dsv41_gather_cache_rows(cache, indices, pool_hi=nope_w)
    gathered_cpu = gathered_tpu.cpu()
    assert torch.equal(gathered_cpu[0], cache_cpu[0])
    assert torch.equal(gathered_cpu[1], cache_cpu[1])
    assert torch.equal(gathered_cpu[2], nope_cpu[0])
    assert torch.all(gathered_cpu[3] == 0)
    assert torch.equal(gathered_cpu[6], nope_cpu[1])
    assert torch.all(gathered_cpu[7] == 0)

    for dep in (
        gathered_tpu[:1],
        latent_cpu[:2].to(device),
        torch.tensor([7, 13], dtype=torch.uint8, device=device),
    ):
        cache_after = torch.zeros((8, 128), dtype=torch.int32, device=device)
        cache_after.copy_(
            dsv41_write_sub_rows(cache_after, updates, slots, slot_offset=16, after=dep)
        )
        assert torch.equal(cache_after.cpu(), cache_cpu)
