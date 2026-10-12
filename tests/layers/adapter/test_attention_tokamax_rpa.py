# SPDX-License-Identifier: Apache-2.0
"""Routing of batched RPA to tokamax's operator, and its fallbacks.

CPU only: tokamax's operator and the in-tree `jax_op` are both faked, so these
check the dispatch in `PallasAttentionBackendImpl._build_rpa_kernel`, not the
kernels. The kernels are compared on TPU in
tests/kernels/experimental/batched_rpa/test_tokamax_parity.py.
"""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa import configs as batched_rpa_configs
from vllm_torchtpu.layers.adapter import attention
from vllm_torchtpu.layers.adapter.attention import (
    PallasAttentionBackendImpl,
    PallasBatchedRPAAttentionBackendImpl,
    _tokamax_rpa_unsupported_reason,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

_TOKAMAX_RPA = "tokamax.experimental.torch_tpu.ops.batched_rpa.torch_pallas_mosaic_tpu"


@pytest.fixture
def vllm_config_context():
    """A CPU-device config, so this runs on hosts without a TPU as well."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls=PallasBatchedRPAAttentionBackendImpl, **kwargs):
    return cls(
        num_heads=2,
        head_size=kwargs.pop("head_size", 128),
        scale=0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=kwargs.pop("sliding_window", None),
        kv_cache_dtype="bfloat16",
        **kwargs,
    )


def _mesh(model=1):
    return SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": model})


@pytest.fixture
def fake_tokamax(monkeypatch):
    """Installs a fake tokamax batched RPA module, parents included.

    Its op class stands in for tokamax's `_PallasMosaicTpuBatchedRpa`: vLLM
    subclasses it to register a donating copy, and calls go through its
    `__call__`, which records the call and bumps the pool. Returns a namespace
    with every call (`calls`, each tagged with the op that took it) and the
    module singleton (`singleton`).
    """
    calls = []

    class FakeBatchedRpa:
        def __init__(self):
            self.jax_op_name = "torch_tpu_pallas_mosaic_tpu_batched_rpa"
            self.donate_argnums = None
            self.fake_impl = None
            self._torch_tokamax_op = mock.Mock()

        def __call__(self, queries, keys, values, kv_cache, *args, **kwargs):
            calls.append((self, queries, keys, values, kv_cache, args, kwargs))
            return torch.full_like(queries, 7.0), kv_cache + 1

    parts = _TOKAMAX_RPA.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for i in range(1, len(parts)):
        parent, child = ".".join(parts[:i]), parts[i]
        setattr(sys.modules[parent], child, sys.modules[f"{parent}.{child}"])
    singleton = FakeBatchedRpa()
    sys.modules[_TOKAMAX_RPA].PallasMosaicTpuBatchedRpa = singleton
    # vLLM builds its donating subclass once per process.
    monkeypatch.setattr(attention, "_donating_tokamax_rpa_op", None)
    return SimpleNamespace(calls=calls, singleton=singleton)


@pytest.fixture
def in_tree(monkeypatch):
    """Fakes the in-tree `jax_op`; returns the calls its op receives."""
    calls = []

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

        def __call__(self, kv_cache, query, *args, **kwargs):
            calls.append((kv_cache, query, args))
            return kv_cache, torch.full_like(query, 3.0)

    def fake_jax_op(name, fn, **kwargs):
        assert name.startswith("pallas::rpa_kernel"), (
            f"only the in-tree op registers through vLLM's jax_op, not {name}"
        )
        return FakeOp()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(PallasAttentionBackendImpl, "_tokamax_kernel_registry", {})
    # One mesh object, as in a real process: the in-tree registry keys on its id.
    mesh = object()
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (mesh, None, None)),
    )
    monkeypatch.setattr(attention.pallas, "jax_op", fake_jax_op)
    return calls


def _build(impl, mesh=None, **kwargs):
    with set_vllm_model_wrapper_context(mesh=mesh or _mesh(), vllm_config=None):
        return impl._build_rpa_kernel(None, None, None, **kwargs)


def _call(kernel, kv_cache):
    query = torch.ones(3, 2, 128)
    key = torch.ones(3, 1, 128)
    i32 = lambda *s: torch.zeros(s, dtype=torch.int32)  # noqa: E731
    return kernel(kv_cache, query, key, key, i32(2), i32(8), i32(3), i32(3), None)


# --- _tokamax_rpa_unsupported_reason -------------------------------------------


def test_supported_layer_has_no_reason(vllm_config_context):
    assert _tokamax_rpa_unsupported_reason(_impl(), _mesh(), block_major=False) is None


@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda impl, mp: {"mesh": _mesh(model=2)}, "model-axis"),
        (lambda impl, mp: {"block_major": True}, "block-major"),
        (lambda impl, mp: setattr(impl, "head_size", 64), "head_dim 64"),
        (lambda impl, mp: setattr(impl, "sinks", torch.zeros(2)), "sinks"),
        (
            lambda impl, mp: setattr(
                impl, "kv_layout", batched_rpa_configs.KVLayout.SEQ_ALONG_LANE
            ),
            "HND",
        ),
        (lambda impl, mp: setattr(impl, "use_causal_mask", False), "causal"),
    ],
    ids=["tp-in-process", "block-major", "hd64", "sinks", "hnd", "non-causal"],
)
def test_unsupported_layer_names_its_reason(
    vllm_config_context, monkeypatch, setup, expected
):
    impl = _impl()
    overrides = setup(impl, monkeypatch) or {}
    reason = _tokamax_rpa_unsupported_reason(
        impl,
        overrides.get("mesh", _mesh()),
        block_major=overrides.get("block_major", False),
    )
    assert reason is not None and expected in reason


# --- dispatch ------------------------------------------------------------------


def test_batched_layer_runs_tokamax_and_writes_the_pool_back(
    vllm_config_context, fake_tokamax, in_tree
):
    impl = _impl(sliding_window=64)
    kernel = _build(impl)
    kv_cache = torch.zeros(8, 16, 2, 1, 128)
    out = _call(kernel, kv_cache)

    assert len(fake_tokamax.calls) == 1 and not in_tree
    op, queries, keys, _values, pool, _args, kwargs = fake_tokamax.calls[0]
    # queries/keys/values first, then the pool, in the runner's own 5-D shape.
    assert queries.shape == (3, 2, 128) and keys.shape == (3, 1, 128)
    assert pool is kv_cache
    assert kwargs["sm_scale"] == 0.5
    assert kwargs["sliding_window"] == 64
    assert kwargs["use_causal_mask"] is True
    assert kwargs["decode_query_size"] == impl.decode_query_size
    assert torch.all(out == 7.0)
    # The new pool's contents land in the runner's tensor.
    assert torch.all(kv_cache == 1.0)
    # The call took tokamax's `__call__` on vLLM's subclass, which donates the
    # pool (argument 3) and carries vLLM's fake in place of jax_op's default,
    # which refuses a symbolic token count. The singleton is left as it was.
    assert op is not fake_tokamax.singleton
    assert isinstance(op, type(fake_tokamax.singleton))
    assert "__call__" not in type(op).__dict__
    assert op.donate_argnums == (3,)
    assert op.fake_impl is attention._fake_tokamax_rpa
    assert op.jax_op_name != fake_tokamax.singleton.jax_op_name
    assert fake_tokamax.singleton.donate_argnums is None


def test_kernel_iteration_mode_takes_the_non_donating_singleton(
    vllm_config_context, fake_tokamax, in_tree, monkeypatch
):
    """As for the in-tree op: kernel-iteration mode runs the op eagerly, where
    donation would kill the buffer the writeback targets."""
    monkeypatch.setenv("TPU_KERNEL_ITER_MODE", "1")
    _call(_build(_impl()), torch.zeros(8, 16, 2, 1, 128))
    assert not in_tree
    assert fake_tokamax.calls[0][0] is fake_tokamax.singleton
    fake_tokamax.singleton._torch_tokamax_op.register_fake.assert_called_with(
        attention._fake_tokamax_rpa
    )


def test_the_donating_subclass_is_built_once(vllm_config_context, fake_tokamax, in_tree):
    _call(_build(_impl(sliding_window=64)), torch.zeros(8, 16, 2, 1, 128))
    _call(_build(_impl()), torch.zeros(8, 16, 2, 1, 128))
    assert fake_tokamax.calls[0][0] is fake_tokamax.calls[1][0]


def test_fake_takes_a_symbolic_token_count():
    from torch._subclasses import fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    mode = fake_tensor.FakeTensorMode(shape_env=ShapeEnv())
    with mode:
        tokens = mode.shape_env.create_unbacked_symint()
        torch._check_is_size(tokens)
        queries = torch.empty((tokens, 2, 128), dtype=torch.bfloat16)
        pool = torch.empty((8, 16, 2, 1, 128), dtype=torch.bfloat16)
        out, kv = attention._fake_tokamax_rpa(queries, queries, queries, pool)
        out32, _ = attention._fake_tokamax_rpa(
            queries, queries, queries, pool, out_dtype="float32"
        )
    assert out.shape == queries.shape and out.dtype == torch.bfloat16
    assert kv.shape == pool.shape
    assert out32.dtype == torch.float32


def test_layers_with_one_config_share_one_closure(
    vllm_config_context, fake_tokamax, in_tree
):
    assert _build(_impl()) is _build(_impl())


def test_flash_attn_backend_keeps_the_in_tree_kernel(
    vllm_config_context, fake_tokamax, in_tree
):
    kernel = _build(_impl(cls=PallasAttentionBackendImpl))
    _call(kernel, torch.zeros(8, 16, 2, 1, 128))
    assert in_tree and not fake_tokamax.calls


def test_unsupported_layer_logs_the_fallback_the_e2e_recipe_greps_for(
    vllm_config_context, fake_tokamax, in_tree, monkeypatch
):
    logger = mock.Mock()
    monkeypatch.setattr(attention, "logger", logger)
    kernel = _build(_impl(), mesh=_mesh(model=2))
    _call(kernel, torch.zeros(8, 16, 2, 1, 128))

    assert in_tree and not fake_tokamax.calls
    (fmt, reason), _ = logger.warning_once.call_args
    # recipes/experimental/stzeng/vllm-torchtpu-tokamax-rpa-e2e.yml greps for this.
    assert "falling back to the in-tree kernel because" in fmt % reason
    assert "model-axis" in reason


def test_missing_tokamax_falls_back_instead_of_failing(
    vllm_config_context, in_tree, monkeypatch
):
    monkeypatch.setitem(sys.modules, _TOKAMAX_RPA, None)
    # A package that already imported the module would hand it back as an
    # attribute, e.g. after an earlier test in the session built tokamax's op.
    parent, _, child = _TOKAMAX_RPA.rpartition(".")
    if parent in sys.modules:
        monkeypatch.delattr(sys.modules[parent], child, raising=False)
    logger = mock.Mock()
    monkeypatch.setattr(attention, "logger", logger)
    kernel = _build(_impl())
    _call(kernel, torch.zeros(8, 16, 2, 1, 128))

    assert in_tree
    (_fmt, reason), _ = logger.warning_once.call_args
    assert "tokamax not being importable" in reason


def test_shared_4d_pool_is_left_to_the_in_tree_kernel(
    vllm_config_context, fake_tokamax, in_tree
):
    """Only the in-tree kernel can view `[N, bs, packing, words]` as pages."""
    kernel = _build(_impl())
    _call(kernel, torch.zeros(8, 16, 1, 256))
    assert in_tree and not fake_tokamax.calls
