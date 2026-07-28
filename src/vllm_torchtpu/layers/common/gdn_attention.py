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
"""
Bridge the torch gdn_attention_core op for gated deltanet attention TPU impl

"""
import dataclasses
import functools
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.layers.common.ragged_gated_delta_rule_wrapper as ragged_gated_delta_rule_wrapper
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.causal_conv1d import causal_conv1d
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_v3_wrapper
from vllm_torchtpu.utils import get_mesh_shape_product

RaggedGatedDeltaRuleImpl = ragged_gated_delta_rule_wrapper.RaggedGatedDeltaRuleImpl


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GdnAttentionConfig:
    ragged_gated_delta_rule_impl: RaggedGatedDeltaRuleImpl = (
        RaggedGatedDeltaRuleImpl.REF)


@functools.lru_cache(maxsize=None)
def _pool_state_ops(ssm_ntok: int, n_v: int, d_k: int, d_v: int, split: int):
    """StateOps reading/writing the ssm f32 token-range of the pool.
    Cached per geometry so the delta-rule jit sees one stable static
    instance across steps."""

    def _read(pool, idx):
        # out_lanes=d_v: the kernel splits pool lanes in place, so this
        # reshape is lane-preserving (free) instead of a lane-crossing
        # relayout of the whole gathered state.
        gathered = pool_adapters.gather_region(pool,
                                               idx,
                                               tok0=0,
                                               ntok=ssm_ntok,
                                               out_dtype=jnp.float32,
                                               out_lanes=d_v,
                                               split=split)
        return gathered.reshape(idx.shape[0], n_v, d_k, d_v)

    def _write(pool, states, idx):
        rows = (n_v * d_k * d_v) // d_v
        return pool_adapters.scatter_region(pool,
                                            states.astype(jnp.float32).reshape(
                                                idx.shape[0], rows, d_v),
                                            idx,
                                            tok0=0,
                                            ntok=ssm_ntok,
                                            split=split)

    return ragged_gated_delta_rule_wrapper.jax_impl.StateOps(read=_read,
                                                             write=_write)


def run_jax_gdn_attention_pooled_local(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    conv_weight: jnp.ndarray,
    conv_bias: Optional[jnp.ndarray],
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """GDN attention over the unified block pool.

    ``recurrent_state`` is the attention-shaped pool holding the f32 ssm
    region at token offset 0 and the conv region in the tokens right after
    it. All pool knowledge lives here: for the chunked impls, conv state
    is gathered/scattered through `pool_adapters` around the stock conv
    kernel and the ssm region is threaded into the stock delta-rule via
    pluggable `StateOps`; the fused V3 kernel instead streams both regions
    in place through a `v3_state_source` copy-plan — the model kernels'
    math is unmodified either way.
    """
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    # Backends with a fixed kernel block (batched RPA) get the pool born
    # at kernel granularity; a manager block is `split` consecutive kernel
    # blocks (upstream map_to_kernel_blocks). State regions address
    # manager-block token ranges; the adapters route them (split kwarg) —
    # the pool itself is never reshaped (an XLA reshape of the pool
    # materializes as a pool-sized relayout copy per step).
    assert pool_block_tokens % recurrent_state.shape[1] == 0, (
        pool_block_tokens, recurrent_state.shape)
    split = pool_block_tokens // recurrent_state.shape[1]

    conv_dim = conv_weight.shape[0]
    lanes = recurrent_state.shape[-1]
    per_tok_elems = math.prod(recurrent_state.shape[2:])
    tok_bytes = per_tok_elems * jnp.dtype(recurrent_state.dtype).itemsize
    ssm_bytes = n_v * d_k * d_v * 4
    conv_bytes = (kernel_size - 1) * conv_dim * 2
    assert ssm_bytes % tok_bytes == 0, (ssm_bytes, tok_bytes)
    ssm_ntok = ssm_bytes // tok_bytes
    assert ssm_ntok <= pool_block_tokens, (
        "ssm state does not fit the attention page", ssm_ntok,
        pool_block_tokens)
    conv_data_rows = conv_bytes // (2 * lanes)

    # The conv slot occupies whole tokens right after the ssm region,
    # padded up so the slot's token range satisfies the tok0 % ntok == 0
    # layout rule; the pad tokens are dead bytes inside the slot.
    conv_ntok = 1
    while (conv_ntok * tok_bytes < conv_bytes or ssm_ntok % conv_ntok != 0):
        conv_ntok *= 2
    assert ssm_ntok + conv_ntok <= pool_block_tokens, (
        "mamba slot exceeds the attention page", ssm_ntok, conv_ntok,
        pool_block_tokens)
    conv_pool, conv_tok0 = recurrent_state, ssm_ntok
    conv_slot_rows = (conv_ntok * tok_bytes) // (2 * lanes)

    if (config.ragged_gated_delta_rule_impl ==
            RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_V3_PD):
        # Fused conv+GDN kernel: both state regions stream directly
        # between the pool and the kernel's double-buffered pipeline (one
        # contiguous DMA per slot per region) — no external gather/scatter
        # round trip. The kernel masks fresh slots via has_initial_state,
        # so a newly-allocated block's bytes are never read, and padded
        # slots move no bytes in either direction.
        plan = pool_adapters.v3_state_source(
            recurrent_state,
            split=split,
            ssm_ntok=ssm_ntok,
            conv_tok0=conv_tok0,
            conv_ntok=conv_ntok,
            conv_dim=conv_dim,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
        )
        recurrent_state, output = gdn_v3_wrapper.fused_conv1d_gdn(
            mixed_qkv,
            b,
            a,
            None,
            None,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            query_start_loc,
            state_indices,
            distribution,
            seq_lens,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            state_source=recurrent_state,
            state_plan=plan,
        )
        return recurrent_state, output

    def _write_conv(pool, new_conv):
        rows = new_conv.reshape(-1, conv_data_rows, lanes)
        pad = conv_slot_rows - conv_data_rows
        if pad:
            rows = jnp.pad(rows, ((0, 0), (0, pad), (0, 0)))
        return pool_adapters.scatter_region(pool,
                                            rows.astype(jnp.bfloat16),
                                            state_indices,
                                            tok0=conv_tok0,
                                            ntok=conv_ntok,
                                            split=split)

    conv_gathered = pool_adapters.gather_region(conv_pool,
                                                state_indices,
                                                tok0=conv_tok0,
                                                ntok=conv_ntok,
                                                split=split,
                                                out_dtype=jnp.bfloat16)
    conv_state = conv_gathered[:, :conv_data_rows, :].reshape(
        -1, kernel_size - 1, conv_dim)
    identity_indices = jnp.arange(state_indices.shape[0], dtype=jnp.int32)

    out_mixed_qkv, new_conv_state = causal_conv1d.ragged_causal_conv1d(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        identity_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    recurrent_state = _write_conv(conv_pool, new_conv_state)

    if config.ragged_gated_delta_rule_impl == RaggedGatedDeltaRuleImpl.REF:
        raise NotImplementedError(
            "the ref impl reads recurrent state natively and is not wired "
            "to the unified block pool; use chunked_jax_pd or "
            "chunked_kernel_v3_pd.")

    wrapper_config = config.ragged_gated_delta_rule_impl.to_config()
    new_recurrent_state, output = ragged_gated_delta_rule_wrapper.ragged_gated_delta_rule_wrapper(
        mixed_qkv=out_mixed_qkv,
        b=b,
        a=a,
        recurrent_state=recurrent_state,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        config=wrapper_config,
        chunk_size=64,
        has_initial_state=has_initial_state,
        state_ops=_pool_state_ops(ssm_ntok, n_v, d_k, d_v, split),
    )

    return new_recurrent_state, output


def run_jax_gdn_attention_pooled(
    j_mixed_qkv: jnp.ndarray,
    j_b: jnp.ndarray,
    j_a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    j_conv_weight: jnp.ndarray,
    j_conv_bias: Optional[jnp.ndarray],
    j_A_log: jnp.ndarray,
    j_dt_bias: jnp.ndarray,
    state_indices: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    distribution: jnp.ndarray,
    seq_lens: jnp.ndarray,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    mesh: jax.sharding.Mesh,
    config: GdnAttentionConfig = GdnAttentionConfig(),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Runs GDN attention over the unified block pool, sharded on the mesh.

    Args:
        j_mixed_qkv: Input tensor of shape `(num_tokens, dim)`.
        j_b: Input tensor of shape `(num_tokens, n_v)`.
        j_a: Input tensor of shape `(num_tokens, n_v)`.
        recurrent_state: The attention-shaped pool
          `(num_blocks, block_size, num_kv_heads * 2, head_size)` in the KV
          dtype, carrying the f32 ssm region and the bf16 conv slot as
          token-ranges of each state block. Block 0 is the null block, only
          used for padded / invalid tokens.
        j_conv_weight: Convolutional weight tensor of shape `(dim, 1,
          kernel_size)`.
        j_conv_bias: Optional convolutional bias tensor of shape `(dim,)`.
        j_A_log: Log of A parameter tensor of shape `(n_v,)`.
        j_dt_bias: Delta T bias tensor of shape `(n_v,)`.
        state_indices: Tensor of shape `(max_reqs,)` mapping request index to
          state block index.
        query_start_loc: Tensor of shape `(num_seqs + 1,)` with start locations of
          each sequence.
        distribution: Tensor of shape `(3,)` int32 — `(decode_end, prefill_end,
          mixed_end)`.
        seq_lens: Tensor of shape `(max_reqs,)` with the total sequence length
          per request (computed + scheduled). Used inside the local function
          to derive ``has_initial_state``.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Dimension of key.
        d_v: Dimension of value.
        kernel_size: Convolution kernel size.
        mesh: The device mesh for distributed computation.
        config: Configuration for implementation selection.

    Returns:
        A tuple containing:
        - The updated pool (the in-place written state regions).
        - The output tensor of shape `(num_tokens, n_v * d_v)`.
    """
    pool_spec = P(None, None, None, None)  # attention-shaped pool
    in_specs = (
        P(None, "model"),  # j_mixed_qkv
        P(None, "model"),  # j_b
        P(None, "model"),  # j_a
        pool_spec,  # recurrent_state (attention-shaped pool)
        P("model", None, None),  # j_conv_weight
        P("model") if j_conv_bias is not None else None,  # j_conv_bias
        P("model"),  # j_A_log
        P("model"),  # j_dt_bias
        P(None),  # query_start_loc
        P(None),  # state_indices
        P(None),  # distribution
        P(None),  # seq_lens
    )

    out_specs = (
        pool_spec,  # new_recurrent_state (attention-shaped pool)
        P(None, "model"),  # output
    )

    tp_size = get_mesh_shape_product(mesh, "model")

    p_run_jax_gdn_attention_pooled_local = functools.partial(
        run_jax_gdn_attention_pooled_local,
        n_kq=n_kq // tp_size,
        n_v=n_v // tp_size,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        config=config,
    )
    mapped_fn = jax.shard_map(
        p_run_jax_gdn_attention_pooled_local,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )

    outputs = mapped_fn(
        j_mixed_qkv,
        j_b,
        j_a,
        recurrent_state,
        j_conv_weight,
        j_conv_bias,
        j_A_log,
        j_dt_bias,
        query_start_loc,
        state_indices,
        distribution,
        seq_lens,
    )

    return outputs
