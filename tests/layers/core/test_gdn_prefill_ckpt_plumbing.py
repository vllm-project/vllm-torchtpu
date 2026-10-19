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
"""The jax-level pooled GDN wrappers hand the internal prefill checkpoint
operands to the kernel entries unchanged and replicated.

The kernels are replaced by stand-ins that record what they receive, so these
run on CPU with a one-device mesh.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.layers.core import gdn_attention

NUM_SEQS = 4
NUM_TOKENS = 8


def _model_mesh():
    return Mesh(np.asarray(jax.devices()[:1]), ("model",))


def _pooled_inputs():
    return dict(
        j_mixed_qkv=jnp.zeros((NUM_TOKENS, 16), jnp.bfloat16),
        j_b=jnp.zeros((NUM_TOKENS, 2), jnp.bfloat16),
        j_a=jnp.zeros((NUM_TOKENS, 2), jnp.bfloat16),
        recurrent_state=jnp.zeros((6, 4, 2, 8), jnp.bfloat16),
        j_conv_weight=jnp.zeros((16, 1, 4), jnp.bfloat16),
        j_conv_bias=None,
        j_A_log=jnp.zeros((2,), jnp.float32),
        j_dt_bias=jnp.zeros((2,), jnp.float32),
        state_indices=jnp.array([1, 2, 3, 0], jnp.int32),
        query_start_loc=jnp.array([0, 1, 2, 8, 8], jnp.int32),
        distribution=jnp.array([2, 2, 3], jnp.int32),
        seq_lens=jnp.array([10, 20, 8191, 0], jnp.int32),
    )


@pytest.mark.parametrize("with_spec_operands", [False, True])
def test_pooled_wrapper_passes_checkpoint_operands_after_the_spec_ones(
    monkeypatch, with_spec_operands
):
    seen = {}

    def fake_local(*args, **_kwargs):
        # Inside the shard_map the operands are tracers: record which slots
        # are filled, and route the checkpoint pair's values to the output.
        seen["is_none"] = [arg is None for arg in args]
        tag = args[14].sum() + 1000 * args[15].sum()
        return args[3], jnp.full((NUM_TOKENS, 4), tag, jnp.float32)

    monkeypatch.setattr(gdn_attention, "run_jax_gdn_attention_pooled_local", fake_local)
    offsets = jnp.array([0, 0, 3072, 0], jnp.int32)
    ids = jnp.array([-1, -1, 5, -1], jnp.int32)
    spec_kwargs = {}
    if with_spec_operands:
        spec_kwargs = dict(
            slot_read_offsets=jnp.zeros((6,), jnp.int32),
            ckpt_indices=jnp.zeros((NUM_SEQS, 2), jnp.int32),
        )

    _, output = gdn_attention.run_jax_gdn_attention_pooled(
        **_pooled_inputs(),
        **spec_kwargs,
        n_kq=2,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=4,
        mesh=_model_mesh(),
        prefill_ckpt_offsets=offsets,
        prefill_ckpt_indices=ids,
    )

    # 12 required operands, read_offsets, ckpt_indices, then the checkpoint pair.
    is_none = seen["is_none"]
    assert len(is_none) == 16
    assert is_none[12:14] == [not with_spec_operands] * 2
    assert is_none[14:] == [False, False]
    # Offsets sum to 3072 and ids to 2, in that order.
    assert float(output[0, 0]) == 3072 + 1000 * 2


def test_pooled_wrapper_without_checkpoints_keeps_its_operands(monkeypatch):
    seen = {}

    def fake_local(*args, **_kwargs):
        seen["num_args"] = len(args)
        return args[3], jnp.zeros((NUM_TOKENS, 4), jnp.bfloat16)

    monkeypatch.setattr(gdn_attention, "run_jax_gdn_attention_pooled_local", fake_local)

    gdn_attention.run_jax_gdn_attention_pooled(
        **_pooled_inputs(),
        n_kq=2,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=4,
        mesh=_model_mesh(),
    )

    assert seen["num_args"] == 12


def test_pooled_local_hands_checkpoint_operands_to_the_kernel(monkeypatch):
    seen = {}

    def fake_kernel(*args, **kwargs):
        seen["kwargs"] = kwargs
        return kwargs["state_source"], jnp.zeros((NUM_TOKENS, 4), jnp.bfloat16)

    monkeypatch.setattr(gdn_attention.gdn_v3_wrapper, "fused_conv1d_gdn", fake_kernel)
    monkeypatch.setattr(
        gdn_attention, "_build_v3_pool_state_plan", lambda *_a, **_k: "plan"
    )
    inputs = _pooled_inputs()
    offsets = jnp.array([0, 0, 3072, 0], jnp.int32)
    ids = jnp.array([-1, -1, 5, -1], jnp.int32)

    gdn_attention.run_jax_gdn_attention_pooled_local(
        inputs["j_mixed_qkv"],
        inputs["j_b"],
        inputs["j_a"],
        inputs["recurrent_state"],
        inputs["j_conv_weight"],
        None,
        inputs["j_A_log"],
        inputs["j_dt_bias"],
        inputs["query_start_loc"],
        inputs["state_indices"],
        inputs["distribution"],
        inputs["seq_lens"],
        None,
        None,
        offsets,
        ids,
        n_kq=2,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=4,
    )

    assert seen["kwargs"]["prefill_ckpt_offsets"] is offsets
    assert seen["kwargs"]["prefill_ckpt_indices"] is ids
    assert seen["kwargs"]["state_plan"] == "plan"


@pytest.mark.parametrize("with_checkpoints", [False, True])
def test_pcp_projection_wrapper_hands_checkpoint_operands_to_the_kernel(
    monkeypatch, with_checkpoints
):
    seen = {}

    def fake_kernel(*args, **kwargs):
        seen["offsets_is_none"] = kwargs["prefill_ckpt_offsets"] is None
        seen["ids_is_none"] = kwargs["prefill_ckpt_indices"] is None
        pool = args[5]
        out = jnp.zeros((NUM_TOKENS, 2, 4), jnp.float32)
        if not with_checkpoints:
            return pool, out, out
        tag = (
            kwargs["prefill_ckpt_offsets"].sum()
            + 1000 * kwargs["prefill_ckpt_indices"].sum()
        )
        return pool, out + tag, out

    monkeypatch.setattr(
        gdn_attention.gdn_v3_pcp_wrapper, "fused_qkvz_projection_pcp_gdn", fake_kernel
    )
    monkeypatch.setattr(
        gdn_attention, "_build_v3_pool_state_plan", lambda *_a, **_k: "plan"
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("pcp", "tp"))
    ckpt_kwargs = {}
    if with_checkpoints:
        ckpt_kwargs = dict(
            prefill_ckpt_offsets=jnp.array([0, 0, 3072, 0], jnp.int32),
            prefill_ckpt_indices=jnp.array([-1, -1, 5, -1], jnp.int32),
        )

    _, output, _ = gdn_attention.run_jax_gdn_attention_pooled_pcp_prefill_projection(
        jnp.zeros((NUM_TOKENS, 8), jnp.bfloat16),
        jnp.zeros((32, 8), jnp.bfloat16),
        None,
        jnp.zeros((NUM_TOKENS, 2), jnp.bfloat16),
        jnp.zeros((NUM_TOKENS, 2), jnp.bfloat16),
        jnp.zeros((6, 4, 2, 8), jnp.bfloat16),
        jnp.zeros((24, 1, 4), jnp.bfloat16),
        None,
        jnp.zeros((2,), jnp.float32),
        jnp.zeros((2,), jnp.float32),
        jnp.array([1, 2, 3, 0], jnp.int32),
        jnp.array([0, 1, 2, 8, 8], jnp.int32),
        jnp.array([0, 0, 3], jnp.int32),
        jnp.array([10, 20, 8191, 0], jnp.int32),
        n_kq=2,
        n_v=2,
        d_k=4,
        d_v=4,
        kernel_size=4,
        pool_block_tokens=4,
        pcp_size=1,
        interleave_size=4,
        mesh=mesh,
        **ckpt_kwargs,
    )

    assert seen["offsets_is_none"] is not with_checkpoints
    assert seen["ids_is_none"] is not with_checkpoints
    if with_checkpoints:
        assert float(output[0, 0, 0]) == 3072 + 1000 * 2
