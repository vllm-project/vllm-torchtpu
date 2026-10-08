"""CPU tests for the GLM-5.3 KDA layer's Kimi backend.

The kernels are covered on TPU by `tests/kernels/test_glm53_kimi_kda.py`.
These tests cover the glue around them, with the Kimi ops replaced by
recording fakes: backend selection, the state shape/dtype contract between
the layer and the model classmethods that size the unified pool, the
conv-weight fold, the row packing the pooled op builder forwards, and
`_forward`'s routing to the pooled or the dispatched op.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import gdn_pool_layout
from vllm_torchtpu.layers.adapter.custom_ops import glm53_kda_vllm_layer_tpu as L
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op
from vllm_torchtpu.models.vllm.glm5_next import model as glm5_next_model

H, D, K = 2, 8, 4  # Local heads, head dim, conv taps.
PROJ = H * D
DIM = 3 * PROJ
T = 6  # Tokens in a step.
LB = -5.0
EPS = 1e-5
PREFIX = "model.layers.0.self_attn"
UNIFIED_POOL_ENV = "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL"
ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM")
MODEL_CLASSES = (
    glm5_next_model.Glm5NextForConditionalGeneration,
    glm5_next_model.Glm5NextForCausalLM,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _default_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(L.GLM53_KDA_BACKEND_ENV, raising=False)
    monkeypatch.delenv(UNIFIED_POOL_ENV, raising=False)


class _Conv(nn.Module):
    """`conv1d` as the base class builds it: weight `[3 * proj, 1, K]`."""

    def __init__(self, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(DIM, 1, K))
        self.bias = nn.Parameter(torch.randn(DIM)) if bias else None


def _base_init(conv_bias: bool, num_spec: int):
    """Stands in for `KimiGatedDeltaNetAttention.__init__`, which needs a full
    vLLM config and TP groups. Sets only what the TPU subclass reads."""

    def init(self, config, vllm_config, prefix=""):
        del config, vllm_config
        nn.Module.__init__(self)
        self.prefix = prefix
        self.conv_size = K
        self.local_num_heads = H
        self.head_dim = D
        self.local_projection_size = PROJ
        self.num_spec = num_spec
        self.gate_lower_bound = LB
        self.conv1d = _Conv(conv_bias)
        self.o_norm = L.FusedRMSNormGated(D, eps=EPS, activation="sigmoid")
        self.A_log = nn.Parameter(torch.randn(H))
        self.dt_bias = nn.Parameter(torch.randn(PROJ))

    return init


def _tripwire(name: str):
    """An op stand-in that fails the test if the layer calls it."""

    def op(*args, **kwargs):
        raise AssertionError(f"{name} must not run on this path")

    return op


class _RecordingOp:
    """An op stand-in that records its arguments and returns a distinctive
    `[T, H, D]` output."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.output = torch.randn(T, H, D)

    def __call__(self, *args):
        self.calls.append(args)
        return self.output.clone()


@pytest.fixture
def builders(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replaces the Kimi op builders and records how the layer calls them."""
    calls: dict = {}

    def recorder(kind: str):
        def build(prefix, **kwargs):
            calls[kind] = (prefix, kwargs)
            return _tripwire(f"{kind}_kda_op")

        return build

    monkeypatch.setattr(L, "build_kimi_dispatched_kda_op", recorder("dispatched"))
    monkeypatch.setattr(L, "build_kimi_pooled_kda_op", recorder("pooled"))
    monkeypatch.setattr(L, "is_conv_state_dim_first", lambda: False)
    return calls


def _vllm_config(architecture=ARCHITECTURES[0], tp=1, hf_config=None):
    if hf_config is None:
        hf_config = SimpleNamespace(num_hidden_layers=4)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture, hf_config=hf_config, dtype=torch.bfloat16
        ),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto", block_size=128),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        speculative_config=None,
    )


def _make_layer(
    monkeypatch: pytest.MonkeyPatch,
    backend: str | None = None,
    *,
    conv_bias: bool = False,
    num_spec: int = 0,
    vllm_config=None,
):
    if backend is not None:
        monkeypatch.setenv(L.GLM53_KDA_BACKEND_ENV, backend)
    monkeypatch.setattr(
        L.KimiGatedDeltaNetAttention, "__init__", _base_init(conv_bias, num_spec)
    )
    if vllm_config is None:
        vllm_config = _vllm_config()
    return L.TpuKimiGatedDeltaNetAttention(SimpleNamespace(), vllm_config, PREFIX)


# Backend selection and pool membership.


def test_backend_defaults_to_kimi() -> None:
    assert L.glm53_kda_backend() == L.KDA_BACKEND_KIMI


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("kimi", "kimi"),
        ("legacy", "legacy"),
        (" KIMI ", "kimi"),
        ("Legacy\n", "legacy"),
        ("", "kimi"),
        ("  ", "kimi"),
    ],
)
def test_backend_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv(L.GLM53_KDA_BACKEND_ENV, raw)
    assert L.glm53_kda_backend() == expected


@pytest.mark.parametrize("raw", ["fla", "kimi-linear", "pooled"])
def test_backend_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(L.GLM53_KDA_BACKEND_ENV, raw)
    with pytest.raises(ValueError, match=L.GLM53_KDA_BACKEND_ENV):
        L.glm53_kda_backend()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_glm53_runs_on_the_unified_pool_by_default(
    monkeypatch: pytest.MonkeyPatch, architecture: str
) -> None:
    assert architecture in gdn_pool_layout.POOLED_GDN_ARCHITECTURES
    assert gdn_pool_layout.unified_kv_layout_enabled_for_architecture(architecture)
    monkeypatch.setenv(UNIFIED_POOL_ENV, "0")
    assert not gdn_pool_layout.unified_kv_layout_enabled_for_architecture(architecture)


# The state contract between the layer and the pool sizing.


def test_kimi_state_shape_and_dtype() -> None:
    assert L.kimi_kda_state_shape(4, 8, 128) == ((3, 1, 3 * 8 * 128), (8, 128, 128))
    assert L.kimi_kda_state_dtype() == (torch.float32, torch.float32)


def test_state_dtype_matches_kimi_k3() -> None:
    """GLM-5.3 and Kimi-K3 hand their state to the same op."""
    from vllm_torchtpu.models.vllm.kimi_k3 import attention as kimi_attention

    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )
    assert kimi_attention.kda_state_dtype(config) == L.kimi_kda_state_dtype()


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_model_state_spec_matches_the_layer(
    monkeypatch: pytest.MonkeyPatch, builders: dict, model_cls, tp: int
) -> None:
    """The platform sizes the unified pool from the model classmethods and
    vLLM allocates from the layer, so the two must declare the same state."""
    config = _vllm_config(tp=tp)
    shape = model_cls.get_mamba_state_shape_from_config(config)
    dtype = model_cls.get_mamba_state_dtype_from_config(config)
    assert shape == L.kimi_kda_state_shape(4, 64 // tp, 128)
    assert dtype == L.kimi_kda_state_dtype()

    layer = _make_layer(monkeypatch)
    layer.conv_size, layer.local_num_heads, layer.head_dim = 4, 64 // tp, 128
    assert layer.get_state_shape() == shape
    assert layer.get_state_dtype() == dtype


def test_model_state_spec_follows_the_checkpoint_kda_geometry() -> None:
    hf_config = SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=4,
            linear_attn_config={
                "num_heads": 16,
                "head_dim": 64,
                "short_conv_kernel_size": 3,
            },
        )
    )
    config = _vllm_config(tp=4, hf_config=hf_config)
    shape = MODEL_CLASSES[0].get_mamba_state_shape_from_config(config)
    assert shape == ((2, 1, 3 * 4 * 64), (4, 64, 64))


def test_model_state_spec_rejects_heads_not_divisible_by_tp() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        MODEL_CLASSES[0].get_mamba_state_shape_from_config(_vllm_config(tp=3))


def test_model_state_spec_is_unchanged_for_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(L.GLM53_KDA_BACKEND_ENV, "legacy")
    config = _vllm_config(tp=8)
    model_cls = MODEL_CLASSES[0]
    assert model_cls.get_mamba_state_shape_from_config(config) == (
        glm5_next_model.MambaStateShapeCalculator.kda_state_shape(
            8, 64, 128, conv_kernel_size=4, num_spec=0
        )
    )
    assert model_cls.get_mamba_state_dtype_from_config(config) == (
        glm5_next_model.MambaStateDtypeCalculator.kda_state_dtype(
            torch.bfloat16, "auto"
        )
    )


# Construction.


def test_kimi_construction_builds_both_ops(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    config = _vllm_config()
    layer = _make_layer(monkeypatch, num_spec=2, vllm_config=config)

    assert layer._kda_backend == L.KDA_BACKEND_KIMI
    assert not layer._conv_weight_folded
    assert builders["dispatched"] == (
        PREFIX,
        dict(lower_bound=LB, eps=EPS, state_dim_first=False, num_spec_tokens=2),
    )
    prefix, kwargs = builders["pooled"]
    assert prefix == PREFIX
    assert kwargs.pop("vllm_config") is config
    assert kwargs == dict(lower_bound=LB, eps=EPS, pool_tokens_per_row=1)
    buffer = dict(layer.named_buffers())["conv_weight_fused"]
    assert buffer.shape == (K, 3, H, D)
    assert buffer.dtype == torch.float32
    # Derived from `conv1d.weight`; not a checkpoint entry.
    assert "conv_weight_fused" not in layer.state_dict()


def test_kimi_construction_rejects_a_conv_bias(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    with pytest.raises(NotImplementedError, match="bias-free"):
        _make_layer(monkeypatch, conv_bias=True)


def test_legacy_construction_skips_the_kimi_ops(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    monkeypatch.setenv(UNIFIED_POOL_ENV, "0")
    shape, dtype = ((3, DIM), (H, D, D)), (torch.bfloat16, torch.float32)
    monkeypatch.setattr(
        L.KimiGatedDeltaNetAttention, "get_state_shape", lambda self: shape
    )
    monkeypatch.setattr(
        L.KimiGatedDeltaNetAttention, "get_state_dtype", lambda self: dtype
    )
    # The legacy kernels take a conv bias, so it must not be rejected.
    layer = _make_layer(monkeypatch, "legacy", conv_bias=True)

    assert layer._kda_backend == L.KDA_BACKEND_LEGACY
    assert builders == {}
    assert "conv_weight_fused" not in dict(layer.named_buffers())
    assert layer.get_state_shape() == shape
    assert layer.get_state_dtype() == dtype
    layer.fold_conv_weight()
    assert not layer._conv_weight_folded


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_legacy_construction_refuses_the_unified_pool(
    monkeypatch: pytest.MonkeyPatch, builders: dict, architecture: str
) -> None:
    with pytest.raises(NotImplementedError, match=f"{UNIFIED_POOL_ENV}=0"):
        _make_layer(monkeypatch, "legacy", vllm_config=_vllm_config(architecture))


# The conv-weight fold.


def test_fold_conv_weight_layout(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _make_layer(monkeypatch)
    # Encode (channel, tap) in every value: w[c, 0, k] = 100 * c + k.
    channel = torch.arange(DIM, dtype=torch.float32)[:, None, None]
    tap = torch.arange(K, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        layer.conv1d.weight.copy_(100 * channel + tap)

    layer.fold_conv_weight()

    fused = layer.conv_weight_fused
    assert layer._conv_weight_folded
    assert fused.shape == (K, 3, H, D)
    assert fused.is_contiguous()
    # mixed_qkv's channel order: q | k | v, each laid out (head, head_dim).
    j, h, d = torch.meshgrid(
        torch.arange(3), torch.arange(H), torch.arange(D), indexing="ij"
    )
    c = (j * PROJ + h * D + d).float()
    for k in range(K):
        torch.testing.assert_close(fused[k], 100 * c + k, rtol=0, atol=0)
    # Still the registered non-persistent buffer, so `.to()` moves it.
    assert dict(layer.named_buffers())["conv_weight_fused"] is fused
    assert "conv_weight_fused" not in layer.state_dict()


def test_folded_weight_reproduces_the_causal_depthwise_conv(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    """Folded tap k multiplies input t - (K - 1) + k, which is what
    `conv1d.weight` tap k means to vLLM's causal conv1d, and what the Kimi
    kernel applies (see the fp64 reference in the kernel tests)."""
    layer = _make_layer(monkeypatch)
    layer.fold_conv_weight()
    x = torch.randn(T, DIM)

    expected = F.conv1d(
        F.pad(x.t()[None], (K - 1, 0)), layer.conv1d.weight.detach(), groups=DIM
    )[0].t()
    padded = F.pad(x, (0, 0, K - 1, 0))
    fused = layer.conv_weight_fused.reshape(K, DIM)
    actual = sum(fused[k] * padded[k : k + T] for k in range(K))

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fold_is_idempotent(monkeypatch: pytest.MonkeyPatch, builders: dict) -> None:
    layer = _make_layer(monkeypatch)
    layer.fold_conv_weight()
    first = layer.conv_weight_fused.clone()

    layer.process_weights_after_loading(torch.bfloat16)

    torch.testing.assert_close(layer.conv_weight_fused, first, rtol=0, atol=0)


def test_model_post_load_hook_folds_every_kda_layer(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layers = [_make_layer(monkeypatch) for _ in range(2)]
    model = SimpleNamespace(modules=lambda: [layers[0], nn.Linear(2, 2), layers[1]])

    glm5_next_model.Glm5NextForConditionalGeneration.process_weights_after_loading(
        model
    )

    assert all(layer._conv_weight_folded for layer in layers)


# The pooled op builder.


@pytest.mark.parametrize("rows", [None, 1])
def test_pooled_op_builder_forwards_the_row_packing(
    monkeypatch: pytest.MonkeyPatch, rows: int | None
) -> None:
    captured: dict = {}

    class FakeJaxOp:
        def __init__(self, name, function, donate_argnums=()):
            captured.update(name=name, function=function, donate=donate_argnums)

        def register_fake(self, fake):
            del fake

    seen: dict = {}

    def fake_core(*args, **kwargs):
        seen.update(kwargs, num_args=len(args))
        return None, None

    monkeypatch.setattr(kda_attention_op.pallas, "jax_op", FakeJaxOp)
    monkeypatch.setattr(kda_attention_op, "_pooled_kda_core", fake_core)
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=128))
    extra = {} if rows is None else {"pool_tokens_per_row": rows}

    kda_attention_op.build_kimi_pooled_kda_op(
        PREFIX, lower_bound=LB, eps=EPS, vllm_config=config, **extra
    )
    captured["function"](*range(13))

    # The packing decides every region offset, so it is part of the program
    # name; unset keeps the Kimi name (and its inference) unchanged.
    assert captured["name"].endswith("_rows1") == (rows == 1)
    assert ("_rows" in captured["name"]) == (rows is not None)
    assert captured["donate"] == (4,)
    assert seen["num_args"] == 13
    assert seen["pool_tokens_per_row"] == rows
    assert seen["pool_block_tokens"] == 128


# `_forward` routing.


def _metadata(**overrides) -> SimpleNamespace:
    """vllm_torchtpu per-layer metadata: one decode, then one prefill."""
    fields = dict(
        query_start_loc=torch.tensor([0, 1, T], dtype=torch.int32),
        seq_lens=torch.tensor([9, T - 1], dtype=torch.int32),
        request_distribution=torch.tensor([1, 1, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([3, 5], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _inputs():
    """What the base class hands `_forward`: a split view of the fused
    projection, g1 and beta with a batch dim, g2 without one."""
    mixed_qkv = torch.randn(T, DIM + 5).split([DIM, 5], dim=-1)[0]
    return (
        mixed_qkv,
        torch.randn(1, T, H, D),
        torch.randn(T, H, D),
        torch.randn(1, T, H),
    )


def _kimi_layer(monkeypatch: pytest.MonkeyPatch, kv_cache):
    layer = _make_layer(monkeypatch)
    layer.fold_conv_weight()
    layer.kv_cache = kv_cache
    layer.pooled_kda_op = _tripwire("pooled_kda_op")
    layer.dispatched_kda_op = _tripwire("dispatched_kda_op")
    return layer


def _pool() -> torch.Tensor:
    return torch.zeros(4, 16, 2, 2, 256)


def _run_forward(monkeypatch, layer, metadata, inputs, out_ndim: int = 4):
    context = SimpleNamespace(
        attn_metadata=None if metadata is None else {PREFIX: metadata}
    )
    monkeypatch.setattr(L, "get_forward_context", lambda: context)
    mixed_qkv, g1, g2, beta = inputs
    core = torch.full((1, T, H, D)[4 - out_ndim :], float("nan"))
    layer._forward(mixed_qkv, g1, g2, beta, core)
    return core


@pytest.mark.parametrize("out_ndim", [3, 4])
def test_forward_sends_a_pool_buffer_to_the_pooled_op(
    monkeypatch: pytest.MonkeyPatch, builders: dict, out_ndim: int
) -> None:
    pool = _pool()
    layer = _kimi_layer(monkeypatch, [pool])
    op = layer.pooled_kda_op = _RecordingOp()
    metadata = _metadata()
    inputs = mixed_qkv, g1, g2, beta = _inputs()

    core = _run_forward(monkeypatch, layer, metadata, inputs, out_ndim)

    (args,) = op.calls
    assert len(args) == 13
    assert args[0].is_contiguous()
    torch.testing.assert_close(args[0], mixed_qkv, rtol=0, atol=0)
    torch.testing.assert_close(args[1], g1.reshape(T, PROJ), rtol=0, atol=0)
    torch.testing.assert_close(args[2], beta.reshape(T, H), rtol=0, atol=0)
    torch.testing.assert_close(args[3], g2.reshape(T, PROJ), rtol=0, atol=0)
    expected_identities = [
        pool,
        layer.conv_weight_fused,
        layer.A_log,
        layer.dt_bias,
        layer.o_norm.weight,
        metadata.query_start_loc,
        metadata.mamba_state_indices,
        metadata.seq_lens,
        metadata.request_distribution,
    ]
    assert len(args[4:]) == len(expected_identities)
    for got, want in zip(args[4:], expected_identities):
        assert got is want
    expected = op.output if out_ndim == 3 else op.output[None]
    torch.testing.assert_close(core, expected, rtol=0, atol=0)


@pytest.mark.parametrize("speculative", [False, True])
def test_forward_sends_per_layer_caches_to_the_dispatched_op(
    monkeypatch: pytest.MonkeyPatch, builders: dict, speculative: bool
) -> None:
    conv, recurrent = torch.zeros(8, K - 1, 1, DIM), torch.zeros(8, H, D, D)
    layer = _kimi_layer(monkeypatch, [conv, recurrent])
    op = layer.dispatched_kda_op = _RecordingOp()
    overrides = {}
    if speculative:
        overrides = dict(
            mamba_request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
            mamba_slot_read_offsets=torch.tensor([0, 1], dtype=torch.int32),
        )
    metadata = _metadata(**overrides)

    core = _run_forward(monkeypatch, layer, metadata, _inputs())

    (args,) = op.calls
    assert len(args) == 16
    expected_identities = [
        conv,
        recurrent,
        layer.conv_weight_fused,
        layer.A_log,
        layer.dt_bias,
        layer.o_norm.weight,
        metadata.query_start_loc,
        metadata.mamba_state_indices,
        metadata.seq_lens,
        metadata.request_distribution,
    ]
    assert len(args[4:14]) == len(expected_identities)
    for got, want in zip(args[4:14], expected_identities):
        assert got is want
    if speculative:
        assert args[14] is metadata.mamba_request_distribution
        assert args[15] is metadata.mamba_slot_read_offsets
    else:
        # No verify windows: the step's own distribution, and a placeholder
        # the num_spec_tokens=0 op never reads.
        assert args[14] is metadata.request_distribution
        assert args[15] is metadata.mamba_state_indices
    torch.testing.assert_close(core, op.output[None], rtol=0, atol=0)


def test_pooled_forward_rejects_speculative_verify(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _kimi_layer(monkeypatch, [_pool()])
    metadata = _metadata(mamba_slot_read_offsets=torch.zeros(2, dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="Speculative decoding"):
        _run_forward(monkeypatch, layer, metadata, _inputs())


def test_forward_requires_the_folded_conv_weight(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _kimi_layer(monkeypatch, [_pool()])
    layer._conv_weight_folded = False
    with pytest.raises(RuntimeError, match="never folded"):
        _run_forward(monkeypatch, layer, _metadata(), _inputs())


def test_forward_requires_mamba_state_indices(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _kimi_layer(monkeypatch, [_pool()])
    with pytest.raises(RuntimeError, match="mamba_state_indices"):
        _run_forward(monkeypatch, layer, _metadata(mamba_state_indices=None), _inputs())


def test_forward_rejects_gdn_metadata(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    layer = _kimi_layer(monkeypatch, [_pool()])
    metadata = object.__new__(L.GDNAttentionMetadata)
    with pytest.raises(NotImplementedError, match="GLM53_KDA_BACKEND=legacy"):
        _run_forward(monkeypatch, layer, metadata, _inputs())


@pytest.mark.parametrize(
    "setup", ["no_metadata", "no_cache", "empty_cache_list", "empty_cache"]
)
def test_forward_runs_no_op_before_the_cache_exists(
    monkeypatch: pytest.MonkeyPatch, builders: dict, setup: str
) -> None:
    """Profiling runs before the runner binds the state cache; neither op
    may run then (both ops are tripwires here)."""
    kv_cache = {
        "no_metadata": [_pool()],
        "no_cache": None,
        "empty_cache_list": [],
        "empty_cache": [torch.empty(0)],
    }[setup]
    layer = _kimi_layer(monkeypatch, kv_cache)
    metadata = None if setup == "no_metadata" else _metadata()

    core = _run_forward(monkeypatch, layer, metadata, _inputs())

    assert torch.isfinite(core).all()


def test_legacy_forward_rejects_the_unified_pool(
    monkeypatch: pytest.MonkeyPatch, builders: dict
) -> None:
    """Defense in depth behind the construction check: a pool buffer bound
    at run time still fails loudly."""
    monkeypatch.setenv(UNIFIED_POOL_ENV, "0")
    layer = _make_layer(monkeypatch, "legacy")
    layer.kv_cache = [_pool()]
    with pytest.raises(NotImplementedError, match="GLM53_KDA_BACKEND=kimi"):
        _run_forward(monkeypatch, layer, _metadata(), _inputs())
