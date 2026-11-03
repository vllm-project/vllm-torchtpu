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
"""VLLM_TPU_GDN_HOIST_METADATA: the step metadata computed once per step
gives the GDN kernels the same scalar-prefetch metadata, bit for bit, as
the per-layer derivation (read-offset gather, int32 casts, PER_SEQ roll).

CPU only: compares the metadata the kernels would receive, both directly
and through `run_jax_gdn_attention_pooled`'s shard_map plumbing with the
kernel call replaced by a probe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper
from vllm_torchtpu.layers.core import gdn_attention

NUM_SPEC = 3
WINDOW = NUM_SPEC + 1
N_KQ, N_V, D_K, D_V = 2, 6, 128, 128
DIM = N_KQ * D_K * 2 + N_V * D_V
NUM_BLOCKS = 1000


def _step(kind: str, seed: int = 0, num_seqs: int = 48):
    """Per-step metadata as the runner hands it to every GDN layer.

    `decode`: 1-token decodes; `spec`: MTP K=3 verify windows of 1..4
    tokens; `mixed`: windows, then prefill chunks, then idle padded slots.
    """
    rng = np.random.default_rng(seed)
    if kind == "decode":
        qlens = np.ones(num_seqs, np.int32)
        distribution = [num_seqs] * 3
    elif kind == "spec":
        qlens = rng.integers(1, WINDOW + 1, num_seqs).astype(np.int32)
        distribution = [num_seqs] * 3
    elif kind == "mixed":
        n_win, n_pre = 30, 10
        qlens = np.zeros(num_seqs, np.int32)
        qlens[:n_win] = rng.integers(1, WINDOW + 1, n_win)
        qlens[n_win : n_win + n_pre] = rng.integers(5, 300, n_pre)
        distribution = [n_win, n_win + n_pre, n_win + n_pre]
    else:
        raise ValueError(kind)
    live = qlens > 0
    query_start_loc = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    seq_lens = np.where(live, qlens + rng.integers(0, 5000, num_seqs), 0)
    if kind == "mixed":
        # Some prefill chunks are fresh (no initial state).
        seq_lens[30:33] = qlens[30:33]
    blocks = rng.permutation(np.arange(1, NUM_BLOCKS))
    state_indices = np.where(live, blocks[:num_seqs], 0).astype(np.int32)
    ckpt = blocks[num_seqs : num_seqs + num_seqs * WINDOW].reshape(num_seqs, WINDOW)
    ckpt[:, 0] = state_indices
    ckpt = np.where(live[:, None], ckpt, 0).astype(np.int32)
    slot_read_offsets = rng.integers(0, WINDOW, NUM_BLOCKS).astype(np.int32)
    num_tokens = int(query_start_loc[-1])
    num_tokens = max(num_tokens, num_seqs * WINDOW)
    return dict(
        slot_read_offsets=jnp.asarray(slot_read_offsets),
        state_indices=jnp.asarray(state_indices),
        query_start_loc=jnp.asarray(query_start_loc),
        seq_lens=jnp.asarray(seq_lens.astype(np.int32)),
        distribution=jnp.asarray(np.asarray(distribution, np.int32)),
        ckpt_indices=jnp.asarray(ckpt),
        num_tokens=num_tokens,
    )


def _cfg(mode: config.GDNMode, batch_size: int) -> config.GDNConfig:
    per_seq = mode == config.GDNMode.PER_SEQ
    return config.GDNConfig(
        mode=mode,
        batch_size=batch_size,
        kernel_size=4,
        tile_size=64 if per_seq else 4,
        window_size=1 if per_seq else WINDOW,
        dim_size=DIM,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D_K,
        v_head_dim=D_V,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.float32),
        ),
    )


def _all_metadata(batch_size, *, seq_lens, query_start_loc, state_indices,
                  read_offsets, distribution, ckpt_indices, per_seq_inputs=None):
    """Both kernel calls' metadata, after `fused_conv1d_gdn`'s int32 casts."""
    read_offsets = read_offsets.astype(jnp.int32)
    ckpt_indices = ckpt_indices.astype(jnp.int32)
    return [
        wrapper.kernel_metadata(
            _cfg(mode, batch_size),
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            distribution=distribution,
            ckpt_indices=ckpt_indices,
            per_seq_inputs=per_seq_inputs,
        )
        for mode in (config.GDNMode.BATCHED, config.GDNMode.PER_SEQ)
    ]


def _assert_bit_equal(got, want):
    got_leaves, got_tree = jax.tree.flatten(got)
    want_leaves, want_tree = jax.tree.flatten(want)
    assert got_tree == want_tree
    for i, (g, w) in enumerate(zip(got_leaves, want_leaves, strict=True)):
        g, w = np.asarray(g), np.asarray(w)
        assert g.dtype == w.dtype, (i, g.dtype, w.dtype)
        np.testing.assert_array_equal(g, w, err_msg=f"leaf {i}")


@pytest.mark.parametrize("kind", ["decode", "spec", "mixed"])
def test_step_metadata_matches_per_layer(kind):
    s = _step(kind)
    per_layer = _all_metadata(
        s["num_tokens"],
        seq_lens=s["seq_lens"],
        query_start_loc=s["query_start_loc"],
        state_indices=s["state_indices"],
        read_offsets=s["slot_read_offsets"][s["state_indices"]],
        distribution=s["distribution"],
        ckpt_indices=s["ckpt_indices"],
    )
    read_offsets, ckpt, *per_seq = gdn_attention.gdn_step_metadata(
        s["slot_read_offsets"],
        s["state_indices"],
        s["query_start_loc"],
        s["seq_lens"],
        s["distribution"],
        s["ckpt_indices"],
    )
    hoisted = _all_metadata(
        s["num_tokens"],
        seq_lens=s["seq_lens"],
        query_start_loc=s["query_start_loc"],
        state_indices=s["state_indices"],
        read_offsets=read_offsets,
        distribution=s["distribution"],
        ckpt_indices=ckpt,
        per_seq_inputs=tuple(per_seq),
    )
    _assert_bit_equal(hoisted, per_layer)
    # The PER_SEQ segment is non-empty only in the mixed step, so the roll
    # is exercised with a non-zero shift there.
    if kind == "mixed":
        assert int(hoisted[1].num_tiles) > 0


@pytest.mark.parametrize("kind", ["decode", "spec", "mixed"])
def test_pooled_op_routes_step_metadata(kind, monkeypatch):
    """Through the shard_map: the kernel call sees the same metadata."""
    s = _step(kind, seed=1)
    num_tokens = s["num_tokens"]
    width = N_V * D_V

    def probe(qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
              a_log, dt_bias, query_start_loc, state_indices, distribution,
              seq_lens, read_offsets=None, ckpt_indices=None, *, state_source,
              per_seq_inputs=None, **_):
        md = _all_metadata(
            qkv.shape[0],
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            distribution=distribution,
            ckpt_indices=ckpt_indices,
            per_seq_inputs=per_seq_inputs,
        )
        flat = jnp.concatenate(
            [jnp.ravel(x).astype(jnp.int32) for x in jax.tree.leaves(md)]
        )
        assert flat.size <= num_tokens * width
        out = jnp.pad(flat, (0, num_tokens * width - flat.size))
        return state_source, out.reshape(num_tokens, width)

    monkeypatch.setattr(gdn_attention.gdn_v3_wrapper, "fused_conv1d_gdn", probe)
    monkeypatch.setattr(gdn_attention, "_build_v3_pool_state_plan",
                        lambda *a, **k: None)
    mesh = Mesh(np.array(jax.devices()[:1]), ("model",))
    common = (
        jnp.zeros((num_tokens, DIM), jnp.bfloat16),
        jnp.zeros((num_tokens, N_V), jnp.bfloat16),
        jnp.zeros((num_tokens, N_V), jnp.bfloat16),
        jnp.zeros((8, 16), jnp.bfloat16),
        jnp.zeros((DIM, 1, 4), jnp.bfloat16),
        None,
        jnp.zeros((N_V,), jnp.float32),
        jnp.zeros((N_V,), jnp.float32),
        s["state_indices"],
        s["query_start_loc"],
        s["distribution"],
        s["seq_lens"],
    )
    kwargs = dict(n_kq=N_KQ, n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=4,
                  pool_block_tokens=4352, mesh=mesh, num_spec_tokens=NUM_SPEC)
    _, per_layer = gdn_attention.run_jax_gdn_attention_pooled(
        *common, s["slot_read_offsets"], s["ckpt_indices"], **kwargs
    )
    step = gdn_attention.gdn_step_metadata(
        s["slot_read_offsets"],
        s["state_indices"],
        s["query_start_loc"],
        s["seq_lens"],
        s["distribution"],
        s["ckpt_indices"],
    )
    _, hoisted = gdn_attention.run_jax_gdn_attention_pooled(
        *common, None, None, step_metadata=step, **kwargs
    )
    np.testing.assert_array_equal(np.asarray(hoisted), np.asarray(per_layer))
    assert np.asarray(per_layer).any()
