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
"""The sharded vocabulary projection (layers/adapter/sharded_lm_head.py).

The engines are emulated on CPU JAX devices: each emulated rank runs `install`
on its own copy of the head (its mesh index patched in), and the op's JAX body
runs over a CPU mesh of that many devices. The sharded logits and greedy
tokens are compared with the unsharded head's.
"""

import importlib.util
import os
import pathlib
import sys
import types

# Eight host devices for the emulated DP8 mesh; only takes effect when JAX has
# not initialized its backends yet in this process.
if "jax" not in sys.modules:
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    )

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.sharding import Mesh  # noqa: E402
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    UnquantizedEmbeddingMethod,
)


def _load_module():
    """The module by path: the `layers` package imports torch_tpu kernels,
    which a CPU-only environment does not have."""
    path = (
        pathlib.Path(__file__).parents[3]
        / "src/vllm_torchtpu/layers/adapter/sharded_lm_head.py"
    )
    spec = importlib.util.spec_from_file_location("_sharded_lm_head_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sh = _load_module()

VOCAB, HIDDEN, ROWS = 256, 64, 16


@pytest.fixture(autouse=True)
def _on_the_host():
    """Every JAX array on the CPU device, at full f32 matmul precision,
    whatever backend an earlier test in the session left as the default."""
    with (
        jax.default_device(jax.devices("cpu")[0]),
        jax.default_matmul_precision("highest"),
    ):
        yield


def _cpu_mesh(n):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(f"needs {n} CPU devices, have {len(devices)}")
    return Mesh(np.array(devices[:n]), (sh.EP_AXIS_NAME,))


def _model(weight):
    """A model holding `lm_head` as vLLM's unquantized head does."""
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    head.tp_size = 1
    head.quant_method = UnquantizedEmbeddingMethod()
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.lm_head = head
    return model


def _install_on_ranks(monkeypatch, weight, mesh):
    """Run `install` once per emulated engine; returns each rank's head."""
    n = mesh.shape[sh.EP_AXIS_NAME]
    monkeypatch.setattr(sh, "build_ep_mesh", lambda: mesh)
    monkeypatch.setattr(sh, "build_op", lambda m: "op")
    heads = []
    for idx in range(n):
        monkeypatch.setattr(sh, "ep_mesh_index", lambda idx=idx: idx)
        model = _model(weight)
        assert sh.install(model, [ROWS]) is True
        heads.append(model.language_model.lm_head)
    return heads


def _emulated_op(mesh, heads, xs):
    """The op as rank `r` sees it: its padded rows go in at mesh index r,
    beside the other ranks' rows, and its own block of the output comes back."""
    fn = jax.jit(sh.head_fn(mesh))

    def to_jax(t):
        # Exact: every value of the torch dtype is a value of the JAX one.
        dtype = jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
        return jnp.asarray(t.float().numpy(), dtype=dtype)

    w_global = to_jax(torch.cat([h.weight_slice for h in heads]))

    def op_for(rank):
        def op(x, w_slice):
            assert w_slice is heads[rank].weight_slice
            rows = [
                torch.nn.functional.pad(o, (0, 0, 0, ROWS - o.shape[0])) for o in xs
            ]
            rows[rank] = x
            out = fn(to_jax(torch.cat(rows)), w_global)
            out = np.asarray(out.astype(jnp.float32))[rank * ROWS : (rank + 1) * ROWS]
            return torch.from_numpy(out.copy()).to(x.dtype)

        return op

    return op_for


@pytest.mark.parametrize("n", [8, 4])
def test_body_matches_the_matmul_over_emulated_engines(n):
    mesh = _cpu_mesh(n)
    fn = sh.head_fn(mesh)
    x = jax.random.normal(jax.random.PRNGKey(0), (n * ROWS, HIDDEN), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (VOCAB, HIDDEN), jnp.float32)
    out = jax.jit(fn)(x, w)
    ref = x @ w.T
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-4)
    np.testing.assert_array_equal(
        np.asarray(out).argmax(-1), np.asarray(ref).argmax(-1)
    )


def test_install_slices_by_mesh_index_and_drops_the_full_weight(monkeypatch):
    mesh = _cpu_mesh(8)
    weight = torch.randn(VOCAB, HIDDEN)
    heads = _install_on_ranks(monkeypatch, weight, mesh)
    part = VOCAB // 8
    for idx, head in enumerate(heads):
        torch.testing.assert_close(
            head.weight_slice, weight[idx * part : (idx + 1) * part], rtol=0, atol=0
        )
        assert tuple(head.weight.shape) == (1, HIDDEN)
        assert isinstance(head.quant_method, sh.ShardedHeadMethod)
        assert head.quant_method.rows == ROWS


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sharded_logits_and_greedy_tokens_match_the_unsharded_head(monkeypatch, dtype):
    """Through vLLM's LogitsProcessor, every emulated engine's sharded logits
    (fewer rows than the bucket on most ranks) equal its unsharded logits."""
    n = 8
    mesh = _cpu_mesh(n)
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, HIDDEN).to(dtype)
    heads = _install_on_ranks(monkeypatch, weight, mesh)
    # Uneven engines, as in DP8: one idle-sized, one full, the rest between.
    counts = [1, ROWS, 5, 9, 16, 3, 12, 7]
    xs = [torch.randn(c, HIDDEN).to(dtype) for c in counts]
    op_for = _emulated_op(mesh, heads, xs)

    plain = torch.nn.Module()
    plain.weight = torch.nn.Parameter(weight, requires_grad=False)
    plain.tp_size = 1
    plain.quant_method = types.SimpleNamespace(
        apply=lambda layer, x, bias=None: torch.nn.functional.linear(
            x.float(), layer.weight.float(), bias
        ).to(x.dtype)
    )
    with set_current_vllm_config(VllmConfig()):
        processor = LogitsProcessor(VOCAB)
    for rank in range(n):
        monkeypatch.setattr(sh, "_OP", op_for(rank))
        got = processor(heads[rank], xs[rank])
        want = processor(plain, xs[rank])
        assert got.shape == (counts[rank], VOCAB) and got.dtype == dtype
        tol = 0 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=1e-4)
        torch.testing.assert_close(got.argmax(-1), want.argmax(-1))


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
    with pytest.raises(ValueError):
        m.apply(layer, torch.randn(17, 8))


def test_the_head_needs_one_request_bucket():
    assert sh.unsupported_reason([40]) is None
    assert "request buckets" in sh.unsupported_reason([8, 16, 32])


def _assert_left_alone(model, buckets):
    head = model.language_model.lm_head
    before = (head.quant_method, head.weight)
    assert sh.install(model, buckets) is False
    assert (head.quant_method, head.weight) == before
    assert not hasattr(head, "weight_slice")


def test_install_leaves_the_head_alone_with_several_buckets():
    _assert_left_alone(_model(torch.randn(VOCAB, HIDDEN)), [8, 16, 32])


def test_a_tensor_parallel_head_is_left_alone():
    model = _model(torch.randn(VOCAB, HIDDEN))
    model.language_model.lm_head.tp_size = 2
    _assert_left_alone(model, [ROWS])


def test_a_quantized_head_is_left_alone():
    model = _model(torch.randn(VOCAB, HIDDEN))
    model.language_model.lm_head.quant_method = object()
    _assert_left_alone(model, [ROWS])


def test_a_head_dtype_override_is_left_alone():
    model = _model(torch.randn(VOCAB, HIDDEN))
    model.language_model.logits_processor = types.SimpleNamespace(
        head_dtype=torch.float64
    )
    _assert_left_alone(model, [ROWS])


def test_a_tied_head_is_left_alone():
    model = _model(torch.randn(VOCAB, HIDDEN))
    model.language_model.embed_tokens = model.language_model.lm_head
    _assert_left_alone(model, [ROWS])


def test_no_expert_parallel_mesh_leaves_the_head_alone(monkeypatch):
    monkeypatch.setattr(sh, "build_ep_mesh", lambda: None)
    monkeypatch.setattr(sh, "ep_mesh_index", lambda: None)
    _assert_left_alone(_model(torch.randn(VOCAB, HIDDEN)), [ROWS])


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_SHARDED_LM_HEAD", raising=False)
    model = _model(torch.randn(VOCAB, HIDDEN))
    assert sh.install_if_enabled(model, [ROWS]) is False
    assert isinstance(
        model.language_model.lm_head.quant_method, UnquantizedEmbeddingMethod
    )


def test_request_bucket_lookup_is_unchanged_with_the_head_off():
    """The runner now pads the request count through `num_reqs_paddings`;
    with the default bucket list that is the old power-of-two padding."""
    runner = (
        pathlib.Path(__file__).parents[3] / "src/vllm_torchtpu/runner/tpu_runner.py"
    )
    import ast

    tree = ast.parse(runner.read_text())
    names = {
        "_get_req_paddings",
        "_get_padded_num_reqs_with_upper_limit",
        "_get_padded_token_len",
    }
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in funcs} == names
    import bisect
    import logging

    ns = {"bisect": bisect, "logger": logging.getLogger(__name__)}
    min_seqs = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "MIN_NUM_SEQS" for t in n.targets)
    )
    exec(compile(ast.Module([min_seqs, *funcs], []), str(runner), "exec"), ns)
    for max_reqs in (8, 16, 20, 64, 144, 256):
        paddings = ns["_get_req_paddings"](ns["MIN_NUM_SEQS"], max_reqs)
        for n in range(1, max_reqs + 1):
            assert ns["_get_padded_token_len"](paddings, n) == ns[
                "_get_padded_num_reqs_with_upper_limit"
            ](n, max_reqs)
        # With the head on, every request count runs at the one bucket.
        assert ns["_get_padded_token_len"]([max_reqs], n) == max_reqs
