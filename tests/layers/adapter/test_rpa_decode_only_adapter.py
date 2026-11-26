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
"""Unit tests for PallasAttentionBackendImpl decode-only op pre-build and dispatch."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.layers.adapter import attention as attn_module
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _make_impl(
    cls: type[PallasAttentionBackendImpl], *, head_size: int = 128
) -> PallasAttentionBackendImpl:
    return cls(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _install_fake_jax_op(monkeypatch: pytest.MonkeyPatch):
    built: list[tuple[str, Any, list[Any]]] = []

    def _fake_jax_op(op_name: str, fn: Any, **kwargs: Any) -> MagicMock:
        calls: list[Any] = []

        def _invoke(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kw):
            calls.append((op_name, query.shape))
            return kv_cache.clone(), torch.zeros_like(query)

        op = MagicMock(side_effect=_invoke)
        built.append((op_name, fn, calls))
        return op

    monkeypatch.setattr(attn_module.pallas, "jax_op", _fake_jax_op)
    monkeypatch.setattr(attn_module, "synchronize_tensors", lambda *a, **kw: None)
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    return built


def _dummy_forward_args(num_tokens: int = 16, head_size: int = 128):
    q = torch.zeros((num_tokens, 8, head_size), dtype=torch.bfloat16)
    k = torch.zeros((num_tokens, 2, head_size), dtype=torch.bfloat16)
    v = torch.zeros((num_tokens, 2, head_size), dtype=torch.bfloat16)
    # HEAD_ALONG_SUBLANE bf16 shape: [num_pages, page_size, kv_heads*2//2, 2, head_dim]
    kv_cache = torch.zeros((4, 128, 2, 2, head_size), dtype=torch.bfloat16)
    md = AttentionMetadata(
        input_positions=torch.zeros((num_tokens,), dtype=torch.int32),
        block_tables=torch.zeros((8,), dtype=torch.int32),
        seq_lens=torch.ones((4,), dtype=torch.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32),
        request_distribution=torch.tensor([4, 4, 4], dtype=torch.int32),
    )
    return q, k, v, kv_cache, md


def test_initialize_kernel_prebuilds_decode_only_op_and_forward_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """initialize_kernel pre-builds both ops; forward builds none and dispatches."""
    built = _install_fake_jax_op(monkeypatch)
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    impl = _make_impl(PallasBatchedRPAAttentionBackendImpl, head_size=128)
    layer = cast(Any, SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0))

    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        impl.initialize_kernel(layer)

    assert impl.has_rpa_decode_only_kernel() is True
    assert len(built) == 2
    mixed_name, mixed_fn, mixed_calls = built[0]
    decode_name, decode_fn, decode_calls = built[1]
    assert mixed_name != decode_name
    assert "decode_only" not in mixed_fn.keywords
    assert decode_fn.keywords["decode_only"] is True

    q, k, v, kv_cache, md = _dummy_forward_args()
    # Forward under rpa_decode_only=True must not register any new jax_op.
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), rpa_decode_only=True):
        impl.forward(layer, q, k, v, kv_cache, md)
    assert len(built) == 2
    assert len(decode_calls) == 1
    assert len(mixed_calls) == 0

    # Forward under rpa_decode_only=False routes to the mixed op with zero new ops.
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), rpa_decode_only=False):
        impl.forward(layer, q, k, v, kv_cache, md)
    assert len(built) == 2
    assert len(decode_calls) == 1
    assert len(mixed_calls) == 1


def test_set_subpage_split_clears_stale_decode_only_op(
    monkeypatch: pytest.MonkeyPatch,
):
    """Changing subpage_split invalidates the stored decode-only op until re-init."""
    built = _install_fake_jax_op(monkeypatch)
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    impl = _make_impl(PallasBatchedRPAAttentionBackendImpl, head_size=128)
    layer = cast(Any, SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0))

    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        impl.initialize_kernel(layer)
        assert impl.has_rpa_decode_only_kernel() is True
        old_decode = impl.rpa_kernel_decode_only

        impl.set_subpage_split(2)
        assert impl.has_rpa_decode_only_kernel() is False

        impl.initialize_kernel(layer)
        assert impl.has_rpa_decode_only_kernel() is True
        assert impl.rpa_kernel_decode_only is not old_decode
        assert built[-1][1].keywords["subpage_split"] == 2
        assert built[-1][1].keywords["decode_only"] is True


@pytest.mark.parametrize(
    ("flag", "cls", "head_size"),
    [
        ("0", PallasBatchedRPAAttentionBackendImpl, 128),
        ("1", PallasBatchedRPAAttentionBackendImpl, 64),
        ("1", PallasAttentionBackendImpl, 128),
    ],
)
def test_ineligible_layers_do_not_build_decode_only_op(
    monkeypatch: pytest.MonkeyPatch, flag: str, cls, head_size: int
):
    """Flag off, hd64, and default RPA impl build only the mixed op."""
    built = _install_fake_jax_op(monkeypatch)
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", flag)
    impl = _make_impl(cls, head_size=head_size)
    layer = cast(Any, SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0))

    with set_vllm_model_wrapper_context(mesh=cast(Any, None)):
        impl.initialize_kernel(layer)

    assert impl.has_rpa_decode_only_kernel() is False
    assert len(built) == 1
    assert "decode_only" not in built[0][1].keywords


def test_pcp_and_dcp_do_not_build_decode_only_op(monkeypatch: pytest.MonkeyPatch):
    """PCP streaming and DCP return early before building a decode-only op."""
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    layer = cast(Any, SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0))

    # PCP path.
    pcp_built: list[str] = []
    monkeypatch.setattr(
        attn_module,
        "pcp_streaming_jax_op",
        lambda name, fn, **kw: pcp_built.append(name) or MagicMock(),
    )
    monkeypatch.setattr(
        attn_module, "make_pcp_streaming_rpa_kernel", lambda **kw: lambda *a, **k: None
    )
    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    pcp_impl = _make_impl(PallasBatchedRPAAttentionBackendImpl, head_size=128)
    monkeypatch.setattr(pcp_impl, "_validate_pcp_streaming_support", lambda _s: None)
    monkeypatch.setattr(
        pcp_impl,
        "_select_kernel_mesh",
        lambda mesh, _pcp: (mesh, None, (attn_module.P(None),) * 9),
    )
    pcp_cfg = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            cp_kv_cache_interleave_size=128,
            decode_context_parallel_size=1,
        ),
    )
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), vllm_config=pcp_cfg):
        pcp_impl.initialize_kernel(layer)
    assert pcp_impl.has_rpa_decode_only_kernel() is False
    assert len(pcp_built) == 1

    # DCP path.
    monkeypatch.setattr(
        attn_module,
        "_get_dcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    monkeypatch.setattr(
        attn_module, "_build_dcp_kernels", lambda **kw: (MagicMock(), MagicMock())
    )
    dcp_impl = _make_impl(PallasBatchedRPAAttentionBackendImpl, head_size=128)
    dcp_cfg = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=0,
            decode_context_parallel_size=2,
        ),
    )
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), vllm_config=dcp_cfg):
        dcp_impl.initialize_kernel(layer)
    assert dcp_impl.has_rpa_decode_only_kernel() is False


def test_default_impl_build_rpa_kernel_rejects_decode_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """Requesting decode_only=True on an impl whose entry lacks it raises."""
    _install_fake_jax_op(monkeypatch)
    impl = _make_impl(PallasAttentionBackendImpl, head_size=128)
    with (
        set_vllm_model_wrapper_context(mesh=cast(Any, None)),
        pytest.raises(NotImplementedError, match="decode_only"),
    ):
        impl._build_rpa_kernel(None, None, None, decode_only=True)
