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
"""The vocabulary-sharded LM head: the JAX body on a one-device CPU mesh, the
head method's padding, and what `install` does to the model."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from vllm_torchtpu.layers.adapter import sharded_lm_head

pytestmark = pytest.mark.cpu_test

VOCAB = 256
HIDDEN = 32


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _model(tied: bool = False) -> torch.nn.Module:
    model = torch.nn.Module()
    model.lm_head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    torch.nn.init.normal_(model.lm_head.weight)
    if tied:
        model.embed_tokens = VocabParallelEmbedding(VOCAB, HIDDEN, disable_tp=True)
        model.lm_head.weight = model.embed_tokens.weight
    return model


def _one_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices("cpu")[:1]), (sharded_lm_head.EP_AXIS_NAME,))


def _jax_op(rows: int):
    """The real op body on a one-device CPU mesh, called with torch tensors."""
    fn = sharded_lm_head.head_fn(_one_device_mesh(), rows)
    return lambda x, w: torch.from_numpy(np.asarray(fn(x.numpy(), w.numpy())))


def _fake_ep(monkeypatch, ep_size: int, index: int) -> None:
    """An EP group of `ep_size` with this rank at `index`. With one rank the op
    is the real body; otherwise it is the local matmul, which is what one rank
    of the real op contributes."""
    mesh = SimpleNamespace(shape={sharded_lm_head.EP_AXIS_NAME: ep_size})
    monkeypatch.setattr(sharded_lm_head, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(sharded_lm_head, "ep_mesh_index", lambda: index)
    monkeypatch.setattr(
        sharded_lm_head,
        "build_op",
        lambda mesh, rows: _jax_op(rows) if ep_size == 1 else lambda x, w: x @ w.T,
    )


def _drafter(model: torch.nn.Module, shares_collectives: bool, monkeypatch):
    """A drafter whose draft model shares the target head, as MTP does."""
    draft_model = torch.nn.Module()
    draft_model.model = torch.nn.Module()
    draft_model.model.lm_head = model.lm_head
    monkeypatch.setattr(
        sharded_lm_head,
        "_draft_shares_collectives",
        lambda drafter: shares_collectives,
    )
    return SimpleNamespace(draft_model=draft_model)


# 16 requests with 3 speculative tokens: 64 collective rows. Verify steps pass
# token buckets up to 64, plain and draft steps pass request counts.
@pytest.mark.parametrize("num_rows", [64, 48, 16, 5])
def test_body_matches_the_matmul_at_verify_row_counts(num_rows):
    x = jax.random.normal(jax.random.PRNGKey(0), (num_rows, HIDDEN), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (VOCAB, HIDDEN), jnp.float32)
    out = sharded_lm_head.head_fn(_one_device_mesh(), rows=64)(x, w)
    assert out.shape == (num_rows, VOCAB)
    np.testing.assert_allclose(np.asarray(out), np.asarray(x @ w.T), rtol=1e-4)


def test_method_projects_up_to_the_collective_rows():
    layer = SimpleNamespace(weight_slice=torch.randn(24, 8))
    method = sharded_lm_head.ShardedLMHeadMethod(_jax_op(16), rows=16)
    x = torch.randn(5, 8)
    torch.testing.assert_close(
        method.apply(layer, x), x @ layer.weight_slice.T, rtol=1e-4, atol=1e-4
    )
    with pytest.raises(ValueError, match="exceed"):
        method.apply(layer, torch.randn(17, 8))


@pytest.mark.parametrize(
    ("num_speculative_tokens", "expected"), [(None, 48), (1, 128), (3, 256)]
)
def test_logits_rows_cover_the_verify_bucket(num_speculative_tokens, expected):
    spec = (
        None
        if num_speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    )
    paddings = [16, 32, 64, 128, 256, 512]
    assert sharded_lm_head.logits_rows(48, paddings, spec) == expected


def test_install_keeps_this_ranks_vocabulary_rows(monkeypatch):
    _fake_ep(monkeypatch, ep_size=4, index=2)
    model = _model()
    weight = model.lm_head.weight.detach().clone()
    sharded_lm_head.install(model, 16)
    head = model.lm_head
    part = VOCAB // 4
    torch.testing.assert_close(head.weight_slice, weight[2 * part : 3 * part])
    assert head.weight.shape == (1, HIDDEN)
    x = torch.randn(5, HIDDEN)
    out = head.quant_method.apply(head, x)
    torch.testing.assert_close(out, x @ weight[2 * part : 3 * part].T)


@pytest.mark.parametrize("shares_collectives", [True, False])
def test_draft_head_logits_equal_the_unsharded_head(monkeypatch, shares_collectives):
    _fake_ep(monkeypatch, ep_size=1, index=0)
    model = _model()
    weight = model.lm_head.weight.detach().clone()
    drafter = _drafter(model, shares_collectives, monkeypatch)
    sharded_lm_head.install(model, 64, drafter=drafter)

    draft_head = drafter.draft_model.model.lm_head
    assert (draft_head is model.lm_head) is shares_collectives
    assert (
        isinstance(draft_head.quant_method, sharded_lm_head.ShardedLMHeadMethod)
        is shares_collectives
    )
    x = torch.randn(16, HIDDEN)
    logits = draft_head.quant_method.apply(draft_head, x)
    torch.testing.assert_close(logits, x @ weight.T, rtol=1e-4, atol=1e-4)


def test_unshared_draft_keeps_the_full_head_while_the_target_is_sliced(monkeypatch):
    _fake_ep(monkeypatch, ep_size=4, index=1)
    model = _model()
    weight = model.lm_head.weight.detach().clone()
    drafter = _drafter(model, False, monkeypatch)
    sharded_lm_head.install(model, 64, drafter=drafter)
    assert model.lm_head.weight.shape == (1, HIDDEN)
    torch.testing.assert_close(drafter.draft_model.model.lm_head.weight, weight)


def test_only_a_lockstep_eagle_draft_shares_the_collectives():
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    for lockstep in (True, False):
        drafter = MagicMock(spec=Eagle3Proposer)
        drafter._dp_lockstep_sharded.return_value = lockstep
        assert sharded_lm_head._draft_shares_collectives(drafter) is lockstep
    assert not sharded_lm_head._draft_shares_collectives(SimpleNamespace())


def test_one_request_bucket():
    assert sharded_lm_head.request_buckets(40) == [40]


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("no_head", "no ParallelLMHead"),
        ("tied", "tied"),
        ("tensor_parallel", "tensor-parallel"),
        ("quantized", "unquantized"),
        ("no_ep", "expert-parallel"),
        ("vocab_split", "does not split"),
    ],
)
def test_install_refuses_an_unsupported_head(monkeypatch, case, match):
    _fake_ep(monkeypatch, ep_size=3 if case == "vocab_split" else 4, index=0)
    if case == "no_ep":
        monkeypatch.setattr(sharded_lm_head, "build_ep_mesh", lambda: None)
    model = torch.nn.Module() if case == "no_head" else _model(tied=case == "tied")
    if case == "tensor_parallel":
        model.lm_head.tp_size = 2
    if case == "quantized":
        model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match=match):
        sharded_lm_head.install(model, 16)
