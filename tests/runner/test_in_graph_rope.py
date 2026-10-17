# SPDX-License-Identifier: Apache-2.0
"""Tests for computing rotary embeddings in-graph instead of from a cache."""

import copy
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding as rope_lib
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_torchtpu.runner import utils as runner_utils

pytestmark = pytest.mark.cpu_test

_HEAD_SIZE = 128
_NUM_HEADS = 4
_NUM_KV_HEADS = 2


@pytest.fixture(autouse=True)
def _fresh_rope_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # `get_rope` memoizes modules; each test needs its own instances.
    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})


def _make_rope(
    rope_parameters: dict,
    max_position: int = 4096,
    head_size: int = _HEAD_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["none"]
    with set_current_vllm_config(vllm_config):
        return rope_lib.get_rope(
            head_size,
            max_position=max_position,
            is_neox_style=True,
            rope_parameters=rope_parameters,
            dtype=dtype,
        )


def _qk(
    num_tokens: int, head_size: int = _HEAD_SIZE, dtype: torch.dtype = torch.float32
):
    gen = torch.Generator().manual_seed(0)
    query = torch.randn(num_tokens, _NUM_HEADS * head_size, generator=gen)
    key = torch.randn(num_tokens, _NUM_KV_HEADS * head_size, generator=gen)
    return query.to(dtype), key.to(dtype)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    int_dtype = {2: torch.int16, 4: torch.int32}[actual.element_size()]
    mismatched = actual.view(int_dtype) != expected.view(int_dtype)
    assert not mismatched.any(), f"{int(mismatched.sum())} elements differ bitwise"


class _Model(torch.nn.Module):
    def __init__(self, rope: torch.nn.Module) -> None:
        super().__init__()
        self.rope = rope


@pytest.mark.parametrize(
    "rope_parameters",
    [
        pytest.param({"rope_type": "default", "rope_theta": 1e6}, id="default"),
        pytest.param(
            {
                "rope_type": "llama3",
                "rope_theta": 5e5,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 1024,
            },
            id="llama3",
        ),
    ],
)
def test_in_graph_rope_matches_cached_rope(rope_parameters: dict) -> None:
    rope = _make_rope(rope_parameters)
    positions = torch.tensor([0, 1, 5, 23, 1024, 4095])
    query, key = _qk(positions.numel())
    ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())

    num_patched = runner_utils.patch_in_graph_rotary_embedding(
        _Model(rope), query.dtype
    )

    out_q, out_k = rope(positions, query.clone(), key.clone())
    _assert_bitwise_equal(out_q, ref_q)
    _assert_bitwise_equal(out_k, ref_k)
    assert num_patched == 1
    assert "cos_sin_cache" not in rope._buffers


@pytest.mark.parametrize(
    ("rope_parameters", "head_size", "max_position", "positions"),
    [
        pytest.param(
            {"rope_type": "default", "rope_theta": 1e6, "mrope_section": [16, 24, 24]},
            _HEAD_SIZE,
            4096,
            torch.tensor([[0, 1, 2, 3], [0, 4, 4, 5], [0, 7, 8, 9]]),
            id="mrope",
        ),
        pytest.param(
            {
                "rope_type": "yarn",
                "rope_theta": 1e6,
                "factor": 4.0,
                "original_max_position_embeddings": 1024,
            },
            _HEAD_SIZE,
            4096,
            torch.tensor([0, 1, 5, 2000]),
            id="yarn",
        ),
        pytest.param(
            {
                "rope_type": "deepseek_yarn",
                "rope_theta": 1e4,
                "factor": 40.0,
                "original_max_position_embeddings": 1024,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            },
            64,
            4096,
            torch.tensor([0, 1, 5, 2000]),
            id="deepseek_yarn",
        ),
    ],
)
def test_cache_derived_rope_variants_are_left_untouched(
    rope_parameters: dict,
    head_size: int,
    max_position: int,
    positions: torch.Tensor,
) -> None:
    rope = _make_rope(rope_parameters, max_position, head_size)
    query, key = _qk(positions.shape[-1], head_size)
    ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())

    num_patched = runner_utils.patch_in_graph_rotary_embedding(
        _Model(rope), query.dtype
    )

    assert num_patched == 0
    assert isinstance(rope.cos_sin_cache, torch.Tensor)
    out_q, out_k = rope.forward_native(positions, query.clone(), key.clone())
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)


def test_patch_is_scoped_to_the_given_model() -> None:
    patched = _make_rope({"rope_type": "default", "rope_theta": 1e6})
    runner_utils.patch_in_graph_rotary_embedding(_Model(patched), torch.float32)

    untouched = _make_rope({"rope_type": "default", "rope_theta": 1e4})

    assert RotaryEmbedding.forward_native.__qualname__ == (
        "RotaryEmbedding.forward_native"
    )
    assert isinstance(untouched.cos_sin_cache, torch.Tensor)


def test_patch_rejects_cache_that_does_not_match_inv_freq() -> None:
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6})
    rope.cos_sin_cache.mul_(2.0)

    with pytest.raises(RuntimeError, match="differ from the cache"):
        runner_utils.patch_in_graph_rotary_embedding(_Model(rope), torch.float32)
    assert isinstance(rope.cos_sin_cache, torch.Tensor)


def test_in_graph_rope_is_bit_exact_where_computed_cos_sin_differs_from_cache() -> None:
    # Device sin/cos can round a few entries differently from the CPU-built
    # cache; simulate that by moving some bf16 cache entries by one ulp.
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6}, dtype=torch.bfloat16)
    cache_bits = rope.cos_sin_cache.view(torch.int16)
    cache_bits[3, 5] += 1
    cache_bits[777, 70] += 1
    cache_bits[4000, 0] -= 1
    positions = torch.tensor([0, 3, 4, 777, 4000, 4095, 3])
    query, key = _qk(positions.numel(), dtype=torch.bfloat16)
    ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())

    assert runner_utils.patch_in_graph_rotary_embedding(_Model(rope), query.dtype) == 1

    out_q, out_k = rope(positions, query.clone(), key.clone())
    _assert_bitwise_equal(out_q, ref_q)
    _assert_bitwise_equal(out_k, ref_k)


def test_patch_rejects_too_many_corrected_positions() -> None:
    # 1100 of 4096 rows exceeds the limit of max(64, 25% of positions) = 1024.
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6}, dtype=torch.bfloat16)
    rope.cos_sin_cache.view(torch.int16)[:1100, 1] += 1

    with pytest.raises(RuntimeError, match="limit 1024"):
        runner_utils.patch_in_graph_rotary_embedding(_Model(rope), torch.bfloat16)
    assert isinstance(rope.cos_sin_cache, torch.Tensor)


def test_fp32_cache_is_compared_in_the_dtype_forward_runs_in() -> None:
    # Serving keeps an fp32 cache for a bf16 model. Device fp32 sin/cos differs
    # from the CPU cache by about one ulp in almost every row, which the bf16
    # conversion on the cached path almost always rounds away.
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6})
    rope.cos_sin_cache.view(torch.int32)[:, 7] += 1
    positions = torch.tensor([0, 1, 100, 2047, 4095])
    query, key = _qk(positions.numel(), dtype=torch.bfloat16)
    # An eager forward replaces an fp32 cache with its bf16 conversion, so take
    # the reference from a copy and keep the fp32 cache that serving patches.
    ref_q, ref_k = copy.deepcopy(rope).forward_native(
        positions, query.clone(), key.clone()
    )
    assert rope.cos_sin_cache.dtype == torch.float32

    assert (
        runner_utils.patch_in_graph_rotary_embedding(_Model(rope), torch.bfloat16) == 1
    )

    out_q, out_k = rope(positions, query.clone(), key.clone())
    _assert_bitwise_equal(out_q, ref_q)
    _assert_bitwise_equal(out_k, ref_k)


def test_patched_forward_rejects_a_different_query_dtype() -> None:
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6})
    runner_utils.patch_in_graph_rotary_embedding(_Model(rope), torch.bfloat16)
    query, key = _qk(2)

    with pytest.raises(ValueError, match="bfloat16"):
        rope(torch.tensor([0, 1]), query, key)


def test_cache_built_with_different_cpu_cos_is_reproduced_bit_exactly() -> None:
    # CPU torch.cos can return values up to ~1.5e-4 apart for identical inputs
    # in different processes, so the cache need not match a CPU recompute.
    rope = _make_rope({"rope_type": "default", "rope_theta": 1e6})
    rope.cos_sin_cache[:50, 3] += 1e-4
    positions = torch.tensor([0, 7, 49, 50, 4095])
    query, key = _qk(positions.numel())
    ref_q, ref_k = rope.forward_native(positions, query.clone(), key.clone())

    assert (
        runner_utils.patch_in_graph_rotary_embedding(_Model(rope), torch.float32) == 1
    )

    out_q, out_k = rope(positions, query.clone(), key.clone())
    _assert_bitwise_equal(out_q, ref_q)
    _assert_bitwise_equal(out_k, ref_k)
