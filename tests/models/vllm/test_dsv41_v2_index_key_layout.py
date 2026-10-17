# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index-only cache interpretation must preserve other records and owners."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def _cache(raw=None, *, layout=None, mode="halfplane"):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

    if raw is None:
        raw = torch.zeros((8, 128), dtype=torch.int32)
    if layout is None:
        layout = ((4, 4, 128), torch.bfloat16, 1024, 0)
    return PagedCacheRows(raw, layout, index_key_layout=mode)


def test_halfplane_cpu_write_read_preserves_sibling_and_shared_regions():
    raw = torch.full((8, 128), 731, dtype=torch.int32)
    cache = _cache(raw)
    keys = torch.arange(4 * 128).reshape(4, 128).to(torch.bfloat16)
    slots = torch.tensor([0, 1, 7, -1])
    cache.write(slots, keys)
    before = raw.clone()
    cache.write(slots, keys)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)
    actual = cache[torch.tensor([0, 0, 1]), torch.tensor([0, 1, 3])]
    torch.testing.assert_close(actual, keys[:3], rtol=0, atol=0)
    torch.testing.assert_close(
        raw[1:3], torch.full((2, 128), 731, dtype=torch.int32), rtol=0, atol=0
    )
    ordinary = _cache(mode="ordinary")
    ordinary.write(torch.tensor([0]), keys[:1])
    torch.testing.assert_close(ordinary[torch.tensor([0]), torch.tensor([0])], keys[:1])
    assert cache.raw is raw


def test_halfplane_cpu_bank_boundary_and_invalid_metadata():
    banks = (
        torch.zeros((4, 128), dtype=torch.int32),
        torch.zeros((4, 128), dtype=torch.int32),
    )
    cache = _cache(banks)
    slots = torch.tensor([7, 8, 15])
    values = torch.arange(3 * 128).reshape(3, 128).to(torch.bfloat16)
    cache.write(slots, values)
    torch.testing.assert_close(cache[slots // 4, slots % 4], values, rtol=0, atol=0)
    for record in (-1, 16):
        with pytest.raises(IndexError, match="outside its owner"):
            cache[torch.tensor([record // 4]), torch.tensor([record % 4])]


@pytest.mark.parametrize(
    "shape,stride,offset",
    [((4, 3, 128), 1024, 0), ((4, 4, 128), 1280, 0), ((4, 4, 128), 1024, 256)],
)
def test_halfplane_refuses_partial_owner_rows_before_mutation(shape, stride, offset):
    raw = torch.zeros((16, 128), dtype=torch.int32)
    with pytest.raises(ValueError, match="whole 512-byte owner rows"):
        _cache(raw, layout=(shape, torch.bfloat16, stride, offset))
    assert not raw.any()


def test_same_process_full_page_copy_preserves_halfplane_bits():
    cache = _cache()
    keys = torch.arange(4 * 128).reshape(4, 128).to(torch.bfloat16)
    cache.write(torch.arange(4), keys)
    # Prefix/CoW copies entire manager pages; channel order is opaque bytes.
    cache.raw[2:4].copy_(cache.raw[:2])
    slots = torch.arange(4, 8)
    torch.testing.assert_close(cache[slots // 4, slots % 4], keys, rtol=0, atol=0)


def test_eager_native_index_cache_uses_tagged_interpretation():
    # The real repository collection hook must run in a fresh interpreter.
    path = Path(__file__).resolve()
    root = path.parents[3]
    gate_files = (
        path,
        root / "tests/kernels/deepseek_v41/test_index_key_layout.py",
        root / "tests/kernels/test_dsv41_prefill_index_scores.py",
        root / "tests/kernels/test_dsv41_decode_index_scores.py",
        root / "tests/kernels/test_dsv41_decode_tuning.py",
        root / "tests/scripts/test_dsv41_single_k_smoke_placement.py",
    )
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "cpu_test",
            *(str(file) for file in gate_files),
        ],
        cwd=root,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    output = collected.stdout + collected.stderr
    assert collected.returncode == 0, output
    assert "test_eager_native_index_cache_uses_tagged_interpretation" in output
    assert "A JAX backend was initialized during test collection" not in output

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        NativeCacheView,
        PagedCacheRows,
    )

    view = NativeCacheView()
    view._tpu_index_key_layout = "halfplane"
    view._tpu_raw_cache = torch.zeros((8, 128), dtype=torch.int32)
    view._tpu_cache_layout = ((4, 4, 128), torch.bfloat16, 1024, 0)
    assert isinstance(view.kv_cache, PagedCacheRows)
    assert view.kv_cache.index_key_layout == "halfplane"


def test_index_layout_is_process_lifetime_static(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import index_key_layout

    index_key_layout.cache_clear()
    try:
        monkeypatch.setenv("TPU_DEEPSEEK_V41_INDEX_KEY_LAYOUT", "halfplane")
        assert index_key_layout() == "halfplane"
        monkeypatch.setenv("TPU_DEEPSEEK_V41_INDEX_KEY_LAYOUT", "ordinary")
        assert index_key_layout() == "halfplane"
    finally:
        index_key_layout.cache_clear()


def test_halfplane_refuses_unversioned_external_kv_transfer(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention

    config = SimpleNamespace(
        kv_transfer_config=object(), cache_config=SimpleNamespace(kv_offloading_size=0)
    )
    monkeypatch.setattr(attention, "index_key_layout", lambda: "halfplane")
    monkeypatch.setattr(attention, "get_current_vllm_config", lambda: config)
    with pytest.raises(ValueError, match="external KV transfer"):
        attention.IndexerCache()


def test_halfplane_cpu_writer_matches_legacy_nan_canonicalization():
    cache = _cache()
    bits = torch.tensor([0x7F81, 0x7FC1, -127, -63] * 32, dtype=torch.int16)
    keys = bits.view(torch.bfloat16).reshape(1, 128)
    cache.write(torch.tensor([0]), keys)
    actual = cache[torch.tensor([0]), torch.tensor([0])].view(torch.int16)
    wanted = torch.tensor([0x7FC0, 0x7FC0, -64, -64] * 32, dtype=torch.int16).reshape(
        1, 128
    )
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def test_halfplane_cpu_all_bf16_patterns_normalize_only_nan_payloads():
    bits = torch.arange(65536, dtype=torch.int32)
    values = bits.to(torch.int16).view(torch.bfloat16).reshape(512, 128)
    cache = _cache(
        torch.zeros((256, 128), dtype=torch.int32),
        layout=((128, 4, 128), torch.bfloat16, 1024, 0),
    )
    records = torch.arange(512)
    cache.write(records, values)
    actual = cache[records // 4, records % 4].view(torch.int16).flatten()
    is_nan = ((bits & 0x7F80) == 0x7F80) & ((bits & 127) != 0)
    expected = torch.where(is_nan, (bits & 0x8000) | 0x7FC0, bits).to(torch.int16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bound_cache_cannot_change_interpretation():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import NativeCacheView

    view = NativeCacheView()
    view._tpu_index_key_layout = "halfplane"
    view._tpu_bound_index_key_layout = "ordinary"
    with pytest.raises(ValueError, match="Cannot change Index-K layout"):
        _ = view.kv_cache


def test_model_startup_registers_real_jax_ops_on_cpu(monkeypatch):
    import jax
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import (
        initialize_delayed_mhc,
        initialize_mhc_post,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        attention_kernels,
        cache,
        candidate_cache,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    monkeypatch.setenv("TPU_DEEPSEEK_V41_ATTENTION", "auto")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("registration",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        # The former callback must fail TorchTPU's real registration verifier.
        with pytest.raises(ValueError, match="Missing argument type annotation"):
            pallas.jax_op(
                "pallas::dsv41_test_untyped_read_regression",
                lambda pool, records, elements, fp32=False, halfplane=False: (
                    cache.gather_cache_values(
                        pool, records, elements, fp32, halfplane=halfplane
                    )
                ),
                mesh=mesh,
                input_partition_specs=(P(), P(), None, None, None),
            )
        # Match the six startup registration calls in v2/model.py. These are
        # real jax_op registrations, with no model weights or TPU execution.
        cache.initialize_cache_ops()
        initialize_delayed_mhc()
        deepseek_v41_mhc_op.initialize_shifted_mhc()
        initialize_mhc_post()
        attention_kernels.initialize_paged_attention()
        attention_kernels.initialize_indexer()
        assert all(
            operation is not None
            for operation in (
                cache._read_values_op,
                cache._read_values_2bank_op,
                cache._write_index_values_op,
                cache._write_index_values_after_op,
                attention_kernels._indexer_op,
                attention_kernels._indexer_2bank_op,
                candidate_cache._prepare_op,
                candidate_cache._prepare_2bank_op,
                candidate_cache._select_op,
                candidate_cache._select_2bank_op,
            )
        )
        # Re-entering initialization must retain the already registered ops.
        read = cache._read_values_op
        indexer = attention_kernels._indexer_op
        cache.initialize_cache_ops()
        attention_kernels.initialize_indexer()
        assert cache._read_values_op is read
        assert attention_kernels._indexer_op is indexer
