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

"""The GDN kernel dispatch: the Qwen GDN architectures run the Qwen3.5 copy of
the fused Conv1D-GDN v3 kernel (models/qwen3_5/kernels/gdn), every other
architecture runs the shared kernel in kernels/gdn/v3, and nothing in the GDN
layer calls either kernel except through `gdn_kernel_wrapper`."""

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.gdn_pool_layout import (
    POOLED_GDN_ARCHITECTURES,
    QWEN_GDN_ARCHITECTURES,
)
from vllm_torchtpu.kernels.gdn.v3 import wrapper as main_gdn_wrapper
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.core import gdn_attention
from vllm_torchtpu.models.qwen3_5.kernels.gdn import wrapper as qwen3_5_gdn_wrapper
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

OTHER_ARCHITECTURES = [
    *sorted(POOLED_GDN_ARCHITECTURES - QWEN_GDN_ARCHITECTURES),
    "LlamaForCausalLM",
    None,
]
QWEN = "Qwen3_5MoeForConditionalGeneration"
KIMI = "KimiK3ForConditionalGeneration"

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 2, 4, 8, 8, 4
DIM = 2 * N_KQ * D_K + N_V * D_V
NUM_TOKENS, NUM_SEQS = 6, 2


@pytest.mark.parametrize("architecture", sorted(QWEN_GDN_ARCHITECTURES))
def test_qwen_gdn_architectures_run_the_copy(architecture):
    assert gdn_attention.gdn_kernel_wrapper(architecture) is qwen3_5_gdn_wrapper


@pytest.mark.parametrize("architecture", OTHER_ARCHITECTURES)
def test_every_other_architecture_runs_the_shared_kernel(architecture):
    assert gdn_attention.gdn_kernel_wrapper(architecture) is main_gdn_wrapper


def test_the_layer_calls_a_kernel_only_through_the_dispatch():
    """`gdn_v3_wrapper` is named only inside `gdn_kernel_wrapper`, so no call
    site can bypass the architecture choice."""
    tree = ast.parse(inspect.getsource(gdn_attention))
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gdn_kernel_wrapper":
            allowed = {id(n) for n in ast.walk(node)}
    assert allowed
    uses = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "gdn_v3_wrapper" and id(n) not in allowed
    ]
    assert uses == []


class _Spy:
    """Stands in for one module's fused_conv1d_gdn and records its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, qkv, b, a, states, *args, n_v, d_v, **kwargs):
        self.calls += 1
        return states, jnp.zeros((qkv.shape[0], n_v * d_v), qkv.dtype)


@pytest.fixture
def spies(monkeypatch):
    main_spy, copy_spy = _Spy(), _Spy()
    monkeypatch.setattr(main_gdn_wrapper, "fused_conv1d_gdn", main_spy)
    monkeypatch.setattr(qwen3_5_gdn_wrapper, "fused_conv1d_gdn", copy_spy)
    return SimpleNamespace(main=main_spy, copy=copy_spy)


def _operands():
    return dict(
        b=jnp.zeros((NUM_TOKENS, N_V), jnp.float32),
        a=jnp.zeros((NUM_TOKENS, N_V), jnp.float32),
        conv_weight=jnp.zeros((DIM, 1, KERNEL_SIZE), jnp.float32),
        A_log=jnp.zeros((N_V,), jnp.float32),
        dt_bias=jnp.zeros((N_V,), jnp.float32),
        query_start_loc=jnp.array([0, 3, 6], jnp.int32),
        state_indices=jnp.array([1, 2], jnp.int32),
        distribution=jnp.array([0, 2, 2], jnp.int32),
        seq_lens=jnp.array([3, 3], jnp.int32),
    )


def _run_pooled_local(architecture):
    o = _operands()
    pool = jnp.zeros((4, 16, 2, 128), jnp.float32)
    return gdn_attention.run_jax_gdn_attention_pooled_local(
        jnp.zeros((NUM_TOKENS, DIM), jnp.float32),
        o["b"],
        o["a"],
        pool,
        o["conv_weight"],
        None,
        o["A_log"],
        o["dt_bias"],
        o["query_start_loc"],
        o["state_indices"],
        o["distribution"],
        o["seq_lens"],
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        pool_block_tokens=16,
        architecture=architecture,
    )


def _run_dense(architecture):
    o = _operands()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    return gdn_attention.run_jax_gdn_attention(
        jnp.zeros((NUM_TOKENS, DIM), jnp.float32),
        o["b"],
        o["a"],
        jnp.zeros((4, KERNEL_SIZE - 1, DIM), jnp.float32),
        jnp.zeros((4, N_V, D_K, D_V), jnp.float32),
        o["conv_weight"],
        None,
        o["A_log"],
        o["dt_bias"],
        o["state_indices"],
        o["query_start_loc"],
        o["distribution"],
        o["seq_lens"],
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        mesh=mesh,
        architecture=architecture,
    )


def _run_pooled_sharded(architecture):
    o = _operands()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    return gdn_attention.run_jax_gdn_attention_pooled(
        jnp.zeros((NUM_TOKENS, DIM), jnp.float32),
        o["b"],
        o["a"],
        jnp.zeros((4, 16, 2, 128), jnp.float32),
        o["conv_weight"],
        None,
        o["A_log"],
        o["dt_bias"],
        o["state_indices"],
        o["query_start_loc"],
        o["distribution"],
        o["seq_lens"],
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        pool_block_tokens=16,
        mesh=mesh,
        architecture=architecture,
    )


@pytest.mark.parametrize(
    "run",
    [_run_pooled_local, _run_dense, _run_pooled_sharded],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize(
    ("architecture", "expect_copy"),
    [(QWEN, True), ("Qwen3NextForCausalLM", True), (KIMI, False), (None, False)],
)
def test_each_call_path_runs_the_architectures_kernel(
    spies, run, architecture, expect_copy
):
    _, out = run(architecture)
    assert np.asarray(out).shape == (NUM_TOKENS, N_V * D_V)
    assert (spies.copy.calls, spies.main.calls) == ((1, 0) if expect_copy else (0, 1))


def _attn(architecture):
    attn = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)
    attn.prefix = "model.layers.0.linear_attn"
    attn.num_k_heads = N_KQ
    attn.gqa_interleaved_layout = False
    attn.num_v_heads = N_V
    attn.tp_size = 1
    attn.head_k_dim = D_K
    attn.head_v_dim = D_V
    attn.conv_kernel_size = KERNEL_SIZE
    attn.num_spec = 0
    attn.conv1d = SimpleNamespace(bias=None)
    attn.model_config = SimpleNamespace(dtype=None, architecture=architecture)
    attn.cache_config = SimpleNamespace(
        mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float32"
    )
    return attn


def _vllm_config(architecture):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            cp_kv_cache_interleave_size=16,
            data_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=16, mamba_cache_mode=None),
        model_config=SimpleNamespace(architecture=architecture),
    )


@pytest.mark.parametrize("architecture", [QWEN, KIMI])
def test_the_layer_hands_its_architecture_to_both_ops(architecture):
    """The dense op and the pooled op of a GDN layer pass the layer's model
    architecture down to the dispatch."""
    attn = _attn(architecture)
    built = {}

    def fake_jax_op(name, wrapped_fn, **_kwargs):
        built[name] = wrapped_fn
        return MagicMock()

    with (
        set_vllm_model_wrapper_context(mesh=None, vllm_config=_vllm_config(None)),
        patch.object(gdn_attention_op.pallas, "jax_op", side_effect=fake_jax_op),
        patch.object(gdn_attention_op, "gdn_attention_pooled_core_tpu") as core,
    ):
        attn._build_gdn_op()
        attn._build_pooled_gdn_op()
        (pooled_fn,) = [
            fn
            for name, fn in built.items()
            if name.startswith("pallas::gdn_attention_pooled_")
        ]
        pooled_fn(*([None] * 12))

    (dense_fn,) = [
        fn
        for name, fn in built.items()
        if not name.startswith("pallas::gdn_attention_pooled_")
    ]
    assert dense_fn.keywords["architecture"] == architecture
    assert core.call_args.kwargs["architecture"] == architecture


def test_the_engine_architecture_is_used_when_the_layer_has_none():
    assert (
        gdn_attention_op._model_architecture(
            SimpleNamespace(), _vllm_config("Qwen3NextForCausalLM")
        )
        == "Qwen3NextForCausalLM"
    )
    assert gdn_attention_op._model_architecture(SimpleNamespace(), None) is None
