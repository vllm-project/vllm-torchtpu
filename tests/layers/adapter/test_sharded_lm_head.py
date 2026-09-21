"""Host tests for the sharded LM head and its model compatibility checks."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod)

from vllm_torchtpu.layers.adapter import sharded_lm_head as sh


@pytest.fixture(autouse=True)
def _on_the_host():
    """Every JAX array on the CPU device: the mesh below is a CPU mesh, and the
    reference matmul must run on the same device with the same (full f32)
    precision, whatever backend and matmul precision an earlier test in the
    session left as the default."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_body_matches_the_matmul_on_a_one_device_mesh():
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), (sh.EP_AXIS_NAME, ))
    fn = sh.head_fn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 32), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (64, 32), jnp.float32)
    with mesh:
        out = fn(x, w)
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(x @ w.T),
                               rtol=1e-4,
                               atol=1e-4)


def test_method_pads_and_trims(monkeypatch):
    calls = []

    def stub(x, w):
        calls.append(tuple(x.shape))
        return x @ w.T

    monkeypatch.setattr(sh, "_OP", stub)
    layer = types.SimpleNamespace(weight_slice=torch.randn(24, 8))
    m = sh.ShardedHeadMethod(rows=16)
    x = torch.randn(5, 8)
    out = m.apply(layer, x)
    assert calls == [(16, 8)] and out.shape == (5, 24)
    torch.testing.assert_close(out, x @ layer.weight_slice.T)
    try:
        m.apply(layer, torch.randn(17, 8))
        raise AssertionError("17 rows must be refused")
    except ValueError:
        pass


def test_the_head_needs_one_request_bucket():
    assert sh.unsupported_reason([40]) is None
    assert "request buckets" in sh.unsupported_reason([8, 16, 32])


def test_install_leaves_the_head_alone_outside_the_limit():
    head = torch.nn.Linear(256, 1024, bias=False)
    model = torch.nn.Module()
    model.lm_head = head
    before = head.quant_method if hasattr(head, "quant_method") else None
    assert sh.install(model, [8, 16, 32]) is False
    assert getattr(head, "quant_method", None) is before


def test_a_tensor_parallel_head_is_left_alone():
    head = torch.nn.Linear(256, 1024, bias=False)
    head.tp_size = 2
    model = torch.nn.Module()
    model.lm_head = head
    before = getattr(head, "quant_method", None)
    assert sh.install(model, [16]) is False
    assert getattr(head, "quant_method", None) is before


def _loaded_head(dtype=torch.bfloat16, head_type=ParallelLMHead):
    # Model loading has already populated the head before installation. Build
    # that state without initializing a distributed vLLM model on the host.
    head = object.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(torch.randn(64, 32).to(dtype),
                                     requires_grad=False)
    head.quant_method = UnquantizedEmbeddingMethod()
    head.tp_size = 1
    return head


@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
def test_install_accepts_unquantized_heads_without_a_model_name(
        monkeypatch, dtype, nested):

    class CustomHead(ParallelLMHead):
        pass

    head = _loaded_head(dtype, CustomHead)
    model = torch.nn.Module()
    if nested:
        model.language_model = torch.nn.Module()
        model.language_model.lm_head = head
    else:
        model.lm_head = head
    original = head.weight.detach().clone()
    mesh = types.SimpleNamespace(shape={sh.EP_AXIS_NAME: 8})
    monkeypatch.setattr(sh, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(sh, "ep_mesh_index", lambda: 3)
    monkeypatch.setattr(sh, "build_op", lambda _: object())
    monkeypatch.setattr(sh, "_OP", None)

    assert sh.install(model, [16])
    torch.testing.assert_close(head.weight_slice, original[24:32])
    assert head.weight.shape == (1, 32)
    assert isinstance(head.quant_method, sh.ShardedHeadMethod)


@pytest.mark.parametrize(
    "invalid", ["method", "method_subclass", "dtype", "shape", "interface"])
def test_unsupported_head_is_unchanged(monkeypatch, invalid):
    head = _loaded_head()
    if invalid == "method":
        head.quant_method = object()
    elif invalid == "method_subclass":

        class CustomMethod(UnquantizedEmbeddingMethod):

            def apply(self, layer, x, bias=None):
                return 2 * (x @ layer.weight.T)

        head.quant_method = CustomMethod()
    elif invalid == "dtype":
        head.weight = torch.nn.Parameter(torch.zeros(64, 32, dtype=torch.int8),
                                         requires_grad=False)
    elif invalid == "shape":
        head.weight = torch.nn.Parameter(torch.zeros(64, 32, 1),
                                         requires_grad=False)
    else:
        head = torch.nn.Linear(32, 64, bias=False)
        head.quant_method = UnquantizedEmbeddingMethod()
    model = torch.nn.Module()
    model.lm_head = head
    weight, method = head.weight, head.quant_method
    monkeypatch.setattr(
        sh, "build_ep_mesh",
        lambda: pytest.fail("unsupported head reached mesh setup"))

    assert not sh.install(model, [16])
    assert head.weight is weight
    assert head.quant_method is method
    assert not hasattr(head, "weight_slice")


@pytest.mark.parametrize("same_module", [False, True])
def test_shared_embedding_weights_are_unchanged(same_module):
    head = _loaded_head()
    model = torch.nn.Module()
    model.lm_head = head
    if same_module:
        model.embed_tokens = head
    else:
        model.embed_tokens = torch.nn.Embedding(64, 32)
        model.embed_tokens.weight = head.weight
    weight = head.weight

    assert not sh.install(model, [16])
    assert head.weight is model.embed_tokens.weight is weight
    assert isinstance(head.quant_method, UnquantizedEmbeddingMethod)


@pytest.mark.parametrize("invalid", ["mesh", "vocabulary"])
def test_unsupported_mesh_or_vocabulary_keeps_original_weights(
        monkeypatch, invalid):
    model = torch.nn.Module()
    model.lm_head = _loaded_head()
    if invalid == "vocabulary":
        model.lm_head.weight = torch.nn.Parameter(torch.zeros(65, 32))
    weight, method = model.lm_head.weight, model.lm_head.quant_method
    mesh = None if invalid == "mesh" else types.SimpleNamespace(
        shape={sh.EP_AXIS_NAME: 8})
    monkeypatch.setattr(sh, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(sh, "ep_mesh_index", lambda: 0)

    assert not sh.install(model, [16])
    assert model.lm_head.weight is weight
    assert model.lm_head.quant_method is method


def test_disabled_flag_does_not_install(monkeypatch):
    monkeypatch.setattr(sh.envs, "TPU_SHARDED_LM_HEAD", False)
    monkeypatch.setattr(sh, "install",
                        lambda *_: pytest.fail("disabled flag installed head"))
    assert not sh.install_if_enabled(torch.nn.Module(), [16])


@pytest.mark.parametrize("ranks", [2, 8])
@pytest.mark.parametrize("rows", [5, 16])
def test_collective_preserves_request_and_vocabulary_order(ranks, rows):
    devices = jax.devices("cpu")
    if len(devices) < ranks:
        pytest.skip(f"requires {ranks} CPU devices")
    mesh = Mesh(np.array(devices[:ranks]), (sh.EP_AXIS_NAME, ))
    x = jax.random.normal(jax.random.PRNGKey(1), (ranks * rows, 32))
    weight = jax.random.normal(jax.random.PRNGKey(2), (64, 32))
    sharding = NamedSharding(mesh, P(sh.EP_AXIS_NAME))
    out = sh.head_fn(mesh)(jax.device_put(x, sharding),
                           jax.device_put(weight, sharding))
    np.testing.assert_allclose(np.asarray(out),
                               np.asarray(x @ weight.T),
                               rtol=1e-4,
                               atol=1e-4)
