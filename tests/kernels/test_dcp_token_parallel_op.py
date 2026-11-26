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
"""Unit tests for DCP token-parallel prefill gating and row-split geometry."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import dcp_token_parallel
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.models.vllm import glm_dsa_dispatch


def _make_vllm_config(
    *,
    arch: str = "GlmMoeDsaForCausalLM",
    model_type: str = "glm_moe_dsa",
    max_model_len: int = 1_048_576,
    dcp_size: int = 16,
    interleave_size: int = 4,
    block_size: int = 256,
    enable_prefix_caching: bool = False,
    max_num_seqs: int = 64,
    speculative_config=None,
    kv_transfer_config=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            hf_config=SimpleNamespace(architectures=[arch], model_type=model_type),
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave_size,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size, enable_prefix_caching=enable_prefix_caching
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )


@pytest.fixture(autouse=True)
def _reset_state():
    st = dcp_token_parallel_op._state
    prev = (
        glm_dsa_dispatch._ACTIVE_ARCH,
        glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN,
        st.enabled,
        st.dcp_size,
        st.over_capacity,
    )
    try:
        yield
    finally:
        (
            glm_dsa_dispatch._ACTIVE_ARCH,
            glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN,
            st.enabled,
            st.dcp_size,
            st.over_capacity,
        ) = prev


def test_gather_caps_cover_512k_ruler_and_one_million_mrcr():
    assert dcp_token_parallel_op.GATHER_CAPS[-3:] == (384, 512, 768)
    for kw, max_p, tail in (
        (dict(max_model_len=1_048_576, block_size=256), 324, (260, 324)),
        (dict(max_model_len=524_288, block_size=64), 580, (384, 512, 580)),
        (dict(max_model_len=1_048_576, block_size=128), 580, (384, 512, 580)),
        (dict(max_model_len=1_048_576, block_size=64), 1092, (384, 512, 768, 1092)),
    ):
        lo = dcp_token_parallel_op.layout_for(_make_vllm_config(**kw))
        assert lo.max_pages == max_p and lo.gather_caps[-len(tail) :] == tail


@pytest.mark.parametrize("num_pages", [0, 516, 1028])
def test_page_gather_wave_geometry_and_zero_page_fastpath(num_pages: int):
    from vllm_torchtpu.kernels.mla import page_gather

    cache = jnp.zeros((1200, 4, 8), dtype=jnp.uint32)
    if num_pages == 0:
        out0 = page_gather.gather_pages(cache, jnp.zeros((0,), jnp.int32), 8)
        assert out0.shape == (0, 8) and out0.dtype == jnp.uint32
        return
    with pytest.raises(ValueError, match="expected uint32 pages"):
        page_gather.gather_pages(cache, jnp.zeros((4,), jnp.int32), 7)
    total_subcores, lanes = 32, 16
    num_waves = -(-num_pages // (total_subcores * lanes))
    covered = [
        (w * total_subcores + c) * lanes + i
        for w in range(num_waves)
        for c in range(total_subcores)
        for i in range(lanes)
        if (w * total_subcores + c) * lanes + i < num_pages
    ]
    assert sorted(covered) == list(range(num_pages))


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_pages", [16, 516, 1028])
def test_page_gather_tpu_multi_wave_execution(num_pages: int):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.mla import page_gather

    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("Requires SparseCore TPU device")
    rng = np.random.default_rng(7 + num_pages)
    cache_np = rng.integers(0, 1 << 30, size=(1200, 4, 128), dtype=np.uint32)
    page_ids_np = rng.integers(0, 1200, size=(num_pages,), dtype=np.int32)
    got = page_gather.gather_pages(jnp.asarray(cache_np), jnp.asarray(page_ids_np), 128)
    assert got.shape == (num_pages * 4, 128)
    np.testing.assert_array_equal(
        np.asarray(got), cache_np[page_ids_np].reshape(num_pages * 4, 128)
    )


def test_supported_for_config_falls_back_cleanly_on_apc_and_non_glm(monkeypatch):
    monkeypatch.setattr(glm_dsa_dispatch, "is_tpu_v7x", lambda: True)
    valid = _make_vllm_config(enable_prefix_caching=False, max_num_seqs=64)
    glm_dsa_dispatch.set_active_model_config(valid.model_config)
    assert dcp_token_parallel_op.configure_for(valid) is True
    assert dcp_token_parallel_op.applies(8192) and dcp_token_parallel_op.applies(1024)
    for bad_tok in (1032, 768, 512, 16):
        assert not dcp_token_parallel_op.applies(bad_tok)
    assert dcp_token_parallel_op.configure_for(
        _make_vllm_config(enable_prefix_caching=True)
    )
    assert dcp_token_parallel_op.applies(8192)
    for kw in (
        dict(speculative_config=object()),
        dict(kv_transfer_config=object()),
        dict(max_num_seqs=512),
        dict(dcp_size=1),
        dict(arch="DeepseekV3ForCausalLM", model_type="deepseek_v3"),
    ):
        assert not dcp_token_parallel_op.supported_for_config(_make_vllm_config(**kw))
    from vllm_torchtpu.platforms.tpu_platform import _configure_glm_dsa_dcp_defaults

    raw = _make_vllm_config(interleave_size=1)
    raw.cache_config.block_size = 1024
    raw.cache_config.user_specified_block_size = False
    _configure_glm_dsa_dcp_defaults(raw)  # type: ignore[arg-type]
    assert raw.parallel_config.cp_kv_cache_interleave_size == 4
    assert raw.cache_config.block_size == 256


def test_token_parallel_rank_rows_partition_covers_active_batch_exactly():
    dcp, split_rows, num_tokens = 4, 8, 32
    layout = dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=4,
        page_size=16,
        split_rows=split_rows,
        max_pages=16,
        gather_caps=(4, 8, 16),
    )
    seq_lens, qsl = (
        jnp.array([19, 7, 0, 0], jnp.int32),
        jnp.array([0, 19, 26, 26, 26], jnp.int32),
    )
    rows_per_rank = layout.rows_per_rank(num_tokens)
    owned_rows, total_valid = [], 0
    for r in range(dcp):
        rr = dcp_token_parallel.rank_rows(
            layout, jnp.int32(r), num_tokens, seq_lens, qsl, jnp.int32(2)
        )
        n_r = int(rr.num_rows)
        total_valid += n_r
        owned_rows.extend(
            range(split_rows + r * rows_per_rank, split_rows + r * rows_per_rank + n_r)
        )
        cu = np.asarray(rr.cu_q_lens)
        assert np.all(np.diff(cu) >= 0) and int(cu[int(rr.num_seqs)]) == n_r
    assert total_valid == 26 - split_rows and owned_rows == list(range(split_rows, 26))

    cfg = _make_vllm_config(
        max_model_len=65_536, dcp_size=16, block_size=256, max_num_seqs=16
    )
    glm_dsa_dispatch._ACTIVE_ARCH, glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN = (
        glm_dsa_dispatch.GLM_DSA_ARCH,
        65_536,
    )
    orig_v7x = glm_dsa_dispatch.is_tpu_v7x
    glm_dsa_dispatch.is_tpu_v7x = lambda: True  # type: ignore[assignment]
    try:
        assert dcp_token_parallel_op.configure_for(cfg)
        assert dcp_token_parallel_op.check_step_capacity(
            cfg,
            padded_num_tokens=4096,
            query_start_loc=np.array([0, 2048, 4096], np.int32),
            seq_lens=np.array([40_000, 40_000], np.int32),
            num_reqs=2,
        )
        assert dcp_token_parallel_op.applies(4096)
        assert not dcp_token_parallel_op.check_step_capacity(
            cfg,
            padded_num_tokens=4096,
            query_start_loc=np.array([0, 4096], np.int32),
            seq_lens=np.array([1_048_576], np.int32),
            num_reqs=1,
        )
        assert not dcp_token_parallel_op.applies(4096)
    finally:
        glm_dsa_dispatch.is_tpu_v7x = orig_v7x  # type: ignore[assignment]


def test_should_pass_only_supported_kwargs_to_sparse_ragged_paged_attention() -> None:
    import ast
    import inspect
    import pathlib

    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

    target = getattr(
        sparse_mla_kernel.sparse_ragged_paged_attention,
        "__wrapped__",
        sparse_mla_kernel.sparse_ragged_paged_attention,
    )
    valid_params = set(inspect.signature(target).parameters)
    adapter_dir = pathlib.Path(dcp_token_parallel_op.__file__).resolve().parent
    checked_calls = 0
    for fname in (
        "dcp_nonidx_qb_mla_op.py",
        "dcp_unified_idx_qb_mla_op.py",
        "dcp_token_parallel_op.py",
    ):
        tree = ast.parse((adapter_dir / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sparse_ragged_paged_attention"
            ):
                checked_calls += 1
                kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                unexpected = sorted(kw_names - valid_params)
                assert not unexpected, (
                    f"{fname}:{node.lineno} unexpected kwargs: {unexpected}"
                )
    assert checked_calls >= 4


def test_should_narrow_64bit_buffers_for_jax_keep_input() -> None:
    import torch

    from vllm_torchtpu.layers.adapter.linear_common import coerce_jax_keep_dtype

    assert (
        coerce_jax_keep_dtype(torch.tensor(7, dtype=torch.int64)).dtype == torch.int32
    )
    assert (
        coerce_jax_keep_dtype(torch.tensor(7, dtype=torch.uint64)).dtype == torch.uint32
    )
    assert (
        coerce_jax_keep_dtype(torch.tensor(1.5, dtype=torch.float64)).dtype
        == torch.float32
    )
    bf16 = torch.ones((2, 4), dtype=torch.bfloat16)
    assert coerce_jax_keep_dtype(bf16) is bf16
