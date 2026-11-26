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
"""CPU unit tests for the opt-in decode-only batched RPA graph path.

Covers:
- batched RPA wrapper JAXPR (`decode_only=True` omits `RPAm` and keeps aliasing);
- `PallasAttentionBackendImpl` decode-only op pre-build, dispatch, and invalidation;
- `PiecewiseBackend` patch registration and decode-only overlay dispatch;
- runner bucket planning, self-disable guards, PP agreement, and DP lockstep.

Reference-backed TPU functional coverage for `decode_only=True` lives in
`tests/kernels/experimental/batched_rpa/correctness_test.py`, and cold/warm
two-pass precompile pipeline coverage lives in
`tests/compilation/test_shape_variants.py`.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.compilation.piecewise_backend import PiecewiseBackend
from vllm.distributed import parallel_state

from vllm_torchtpu import _patch_vllm_piecewise_backend
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper
from vllm_torchtpu.layers.adapter import attention as attn_module
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner import rpa_decode_only_graph
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _get_padded_token_len

_TPU_7X_INFO = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
_DP_SIZE = 4


# ---------------------------------------------------------------------------
# 1. Batched RPA wrapper JAXPR (decode_only=True)
# ---------------------------------------------------------------------------


def _find_pallas_eqns(jaxpr):
    eqns = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            eqns.append(eqn)
        for val in eqn.params.values():
            if hasattr(val, "jaxpr"):
                eqns.extend(_find_pallas_eqns(val.jaxpr))
    return eqns


def _make_dummy_wrapper_inputs(kv_layout: configs.KVLayout):
    max_tokens = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    num_pages = 16
    page_size = 128
    max_seqs = 8
    pages_per_seq = 2

    q = jnp.zeros((max_tokens, num_q_heads, head_dim), dtype=jnp.bfloat16)
    k = jnp.zeros((max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    v = jnp.zeros((max_tokens, num_kv_heads, head_dim), dtype=jnp.bfloat16)
    kv_shape = wrapper.get_kv_cache_shape(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        jnp.bfloat16,
        kv_layout=kv_layout,
        chip_version=pltpu.ChipVersion.TPU_7X,
    )
    kv_cache = jnp.zeros(kv_shape, dtype=jnp.bfloat16)
    kv_lens = jnp.ones((max_seqs,), dtype=jnp.int32)
    page_indices = jnp.zeros((max_seqs * pages_per_seq,), dtype=jnp.int32)
    cu_q_lens = jnp.arange(max_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([max_seqs, max_seqs, max_seqs], dtype=jnp.int32)
    return q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution


@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_wrapper_decode_only_omits_mixed_and_preserves_aliasing(kv_layout, return_lse):
    """decode_only=True statically omits RPAm and preserves input-output aliases."""
    inputs = _make_dummy_wrapper_inputs(kv_layout=kv_layout)
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_TPU_7X_INFO):
        mixed = jax.make_jaxpr(
            lambda *args: wrapper.ragged_paged_attention(
                *args, kv_layout=kv_layout, return_lse=return_lse, decode_only=False
            )
        )(*inputs)
        decode = jax.make_jaxpr(
            lambda *args: wrapper.ragged_paged_attention(
                *args, kv_layout=kv_layout, return_lse=return_lse, decode_only=True
            )
        )(*inputs)

    mixed_calls = [
        (e.params.get("name"), e.params.get("input_output_aliases"))
        for e in _find_pallas_eqns(mixed.jaxpr)
    ]
    decode_calls = [
        (e.params.get("name"), e.params.get("input_output_aliases"))
        for e in _find_pallas_eqns(decode.jaxpr)
    ]

    assert len(mixed_calls) == 4
    assert mixed_calls[0][0] == "rpa_metadata_schedule"
    assert mixed_calls[1][0].startswith("RPAd-")
    assert mixed_calls[2][0] == "rpa_metadata_schedule"
    assert mixed_calls[3][0].startswith("RPAm-")

    assert len(decode_calls) == 2
    assert decode_calls[0][0] == "rpa_metadata_schedule"
    assert decode_calls[1][0].startswith("RPAd-")
    assert decode_calls[1][1] == mixed_calls[1][1]
    assert len(decode_calls[1][1]) == (3 if return_lse else 2)


# ---------------------------------------------------------------------------
# 2. Attention adapter: pre-build, forward dispatch, invalidation, eligibility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. PiecewiseBackend patch: overlay dispatch and warmup registration
# ---------------------------------------------------------------------------


def _patched_piecewise_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PiecewiseBackend, "__init__", PiecewiseBackend.__init__)
    monkeypatch.setattr(PiecewiseBackend, "__call__", PiecewiseBackend.__call__)
    _patch_vllm_piecewise_backend()


def test_piecewise_backend_decode_only_overlay_dispatch_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """Decode-only overlay dispatches attached buckets and falls back otherwise."""
    _patched_piecewise_backend(monkeypatch)
    entries = {
        n: SimpleNamespace(
            compiled=True,
            compile_range=SimpleNamespace(start=n, end=n, is_single_size=lambda: True),
            runnable=lambda *a, n=n, **k: n,
        )
        for n in (16, 128)
    }
    backend = SimpleNamespace(
        sym_shape_indices=[],
        range_entries=entries,
        compile_sizes=[16, 128],
        compile_ranges=[(16, 16), (128, 128)],
        _find_range_for_shape=entries.get,
        _tpu_decode_only_runnables={16: lambda *a, **k: "decode_16"},
    )

    assert PiecewiseBackend.__call__(backend, torch.zeros(16, 4)) == 16
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), rpa_decode_only=True):
        assert PiecewiseBackend.__call__(backend, torch.zeros(16, 4)) == "decode_16"
        # Bucket without a decode-only entry falls back to the mixed runnable.
        assert PiecewiseBackend.__call__(backend, torch.zeros(128, 4)) == 128
        with set_vllm_model_wrapper_context(
            mesh=cast(Any, None), rpa_decode_only=False
        ):
            assert PiecewiseBackend.__call__(backend, torch.zeros(16, 4)) == 16

    # Single-bucket backend also dispatches decode-only when attached.
    single_entry = {
        32: SimpleNamespace(
            compiled=True,
            compile_range=SimpleNamespace(
                start=32, end=32, is_single_size=lambda: True
            ),
            runnable=lambda *a, **k: "mixed_32",
        )
    }
    single = SimpleNamespace(
        sym_shape_indices=[],
        range_entries=single_entry,
        compile_sizes=[32],
        compile_ranges=[(32, 32)],
        _find_range_for_shape=single_entry.get,
        _tpu_decode_only_runnables={32: lambda *a, **k: "decode_32"},
    )
    assert PiecewiseBackend.__call__(single, torch.zeros(32, 4)) == "mixed_32"
    with set_vllm_model_wrapper_context(mesh=cast(Any, None), rpa_decode_only=True):
        assert PiecewiseBackend.__call__(single, torch.zeros(32, 4)) == "decode_32"


def test_piecewise_backend_init_registers_in_shape_variants_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """Patched PiecewiseBackend.__init__ registers live backends in warmup session."""
    recorded: list[object] = []

    def fake_init(self, *args, **kwargs):
        recorded.append(self)

    monkeypatch.setattr(PiecewiseBackend, "__init__", fake_init)
    monkeypatch.setattr(PiecewiseBackend, "__call__", PiecewiseBackend.__call__)
    _patch_vllm_piecewise_backend()
    inst = object.__new__(PiecewiseBackend)
    with shape_variants.warmup() as session:
        PiecewiseBackend.__init__(inst)
        assert session.backends == [(0, inst)]
    assert recorded == [inst]


# ---------------------------------------------------------------------------
# 4. Runner bucket planning, self-disable guards, PP and DP agreement
# ---------------------------------------------------------------------------


def _fake_impl(*, has_decode_only: bool) -> PallasAttentionBackendImpl:
    impl = object.__new__(PallasAttentionBackendImpl)
    impl.rpa_kernel_decode_only = object() if has_decode_only else None
    return impl


def _resolve(monkeypatch, *, has_decode_only: bool = True, **overrides):
    monkeypatch.setattr(
        rpa_decode_only_graph,
        "get_layers_from_vllm_config",
        lambda _cfg, _cls: {
            "attn.0": SimpleNamespace(impl=_fake_impl(has_decode_only=has_decode_only))
        },
    )
    kwargs = dict(
        vllm_config=None,
        token_paddings=[16, 32, 64, 128, 256],
        pad_token_len=_get_padded_token_len,
        num_reqs_max_model_len=64,
        num_reqs_most_model_len=None,
        reorder_batch_threshold=1,
        enforce_eager=False,
        has_pcp=False,
        has_kv_cache_bundle=False,
        pipeline_parallel_size=1,
    )
    kwargs.update(overrides)
    return rpa_decode_only_graph.resolve_decode_only_buckets(**kwargs)


def test_disabled_when_flag_off(monkeypatch):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "0")
    assert _resolve(monkeypatch) == []


@pytest.mark.parametrize(
    ("max_reqs", "threshold", "expected"),
    [
        (64, 1, [16, 32, 64]),
        # Non-aligned max_reqs rounds up to the covering bucket (128).
        (100, 1, [16, 32, 64, 128]),
        # Spec decode widens the decode cap by reorder_batch_threshold.
        (64, 2, [16, 32, 64, 128]),
        # Cap above largest bucket covers every bucket without IndexError.
        (512, 1, [16, 32, 64, 128, 256]),
    ],
)
def test_covering_buckets(monkeypatch, max_reqs, threshold, expected):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    assert (
        _resolve(
            monkeypatch,
            num_reqs_max_model_len=max_reqs,
            reorder_batch_threshold=threshold,
        )
        == expected
    )


def test_mega_aot_raises(monkeypatch):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    monkeypatch.setattr(
        rpa_decode_only_graph.vllm_envs, "VLLM_USE_MEGA_AOT_ARTIFACT", True
    )
    with pytest.raises(ValueError, match="VLLM_USE_MEGA_AOT_ARTIFACT"):
        _resolve(monkeypatch)


@pytest.mark.parametrize(
    ("overrides", "has_decode_only"),
    [
        ({"enforce_eager": True}, True),
        ({"has_pcp": True}, True),
        ({"has_kv_cache_bundle": True}, True),
        ({}, False),
    ],
)
def test_self_disable_logs_warning(monkeypatch, overrides, has_decode_only):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    warnings: list[str] = []
    monkeypatch.setattr(
        rpa_decode_only_graph.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )
    assert _resolve(monkeypatch, has_decode_only=has_decode_only, **overrides) == []
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("local_max_reqs", "peer_top_bucket", "expected"),
    [
        # Both stages agree on 64.
        (64, 64, [16, 32, 64]),
        # Local derives 128, peer derives 64 -> MIN reduction agrees on 64.
        (100, 64, [16, 32, 64]),
        # Peer disables (0) -> all stages disable.
        (64, 0, []),
    ],
)
def test_pp_stage_agreement_reduces_top_bucket(
    monkeypatch, local_max_reqs, peer_top_bucket, expected
):
    """Under PP, stages reduce top_bucket via min_over_group to stay in lockstep."""
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    pp_group = SimpleNamespace(cpu_group="pp")
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)

    def _all_reduce(t, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MIN
        assert group == "pp"
        t.clamp_(max=peer_top_bucket)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    assert (
        _resolve(
            monkeypatch,
            num_reqs_max_model_len=local_max_reqs,
            pipeline_parallel_size=2,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mixed_ranks", "expected"),
    [
        (set(), [False] * _DP_SIZE),
        ({2}, [True] * _DP_SIZE),
        ({0, 3}, [True] * _DP_SIZE),
    ],
)
def test_dp_coordinated_step_propagates_rpa_mixed(monkeypatch, mixed_ranks, expected):
    """Every DP rank agrees on whether any rank has mixed/prefill RPA work."""
    runners: list[TPUModelRunner] = []
    for _ in range(_DP_SIZE):
        runner = object.__new__(TPUModelRunner)
        runner._dp_lockstep_enabled = lambda: True
        runner.num_tokens_paddings = [16, 32, 64]
        runners.append(runner)
    sent: list[torch.Tensor] = []

    def _run_rank(rank: int, runner: TPUModelRunner, all_reduce):
        group = SimpleNamespace(rank_in_group=rank, world_size=_DP_SIZE, cpu_group=None)
        monkeypatch.setattr(parallel_state, "get_dp_group", lambda: group)
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
        runner._dp_coordinated_step(16, 1, local_has_rpa_mixed=(rank in mixed_ranks))

    for rank, runner in enumerate(runners):
        _run_rank(rank, runner, lambda t, op=None, group=None: sent.append(t.clone()))
    reduced = torch.stack(sent).amax(dim=0)
    for rank, runner in enumerate(runners):
        _run_rank(rank, runner, lambda t, op=None, group=None: t.copy_(reduced))
    assert [runner._dp_step_has_rpa_mixed for runner in runners] == expected
