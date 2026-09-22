# SPDX-License-Identifier: Apache-2.0
"""Native 3:1 GDN with private FP32 state and isolated recurrence precision."""
import functools

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.gdn.v3.wrapper import fused_conv1d_gdn
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def gdn_state_shapes(tp_size: int):
    if tp_size not in (1, 2, 4, 8):
        raise ValueError("Qwen4Exp GDN supports TP1/2/4/8")
    return (3, 10240 // tp_size), (48 // tp_size, 128, 128)


def gdn_local(qkv: jax.Array, b: jax.Array, a: jax.Array,
              conv_state: jax.Array, recurrent_state: jax.Array,
              conv_weight: jax.Array, conv_bias: jax.Array | None,
              a_log: jax.Array, dt_bias: jax.Array, state_indices: jax.Array,
              query_start_loc: jax.Array, distribution: jax.Array,
              seq_lens: jax.Array, *, n_kq: int,
              n_v: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    # This context must be INSIDE the separately named custom op: a context
    # around its caller is lost when TorchTPU traces/caches the JAX function.
    with jax.default_matmul_precision("highest"):
        (conv, state), output = fused_conv1d_gdn(qkv,
                                                 b,
                                                 a,
                                                 conv_state,
                                                 recurrent_state,
                                                 conv_weight,
                                                 conv_bias,
                                                 a_log,
                                                 dt_bias,
                                                 query_start_loc,
                                                 state_indices,
                                                 distribution,
                                                 seq_lens,
                                                 n_kq=n_kq,
                                                 n_v=n_v,
                                                 d_k=128,
                                                 d_v=128,
                                                 kernel_size=4,
                                                 decode_tile_size=1)
    return conv, state, output


def build_gdn_op(prefix: str, tp_size: int, *, mesh=None, has_conv_bias=False):
    """Build outside Dynamo; returned callable mutates the two bound caches.

    TP refers to vLLM's rank-local head partition. A provided TorchTPU mesh
    must have model-axis size one: concatenated Q|K|V cannot be subdivided
    by a contiguous width partition without separately sharding each field.
    """
    gdn_state_shapes(tp_size)
    if mesh is not None and mesh.shape["model"] != 1:
        raise ValueError("Qwen4Exp GDN requires model-axis size 1")
    n_kq, n_v = 16 // tp_size, 48 // tp_size
    local = functools.partial(gdn_local, n_kq=n_kq, n_v=n_v)
    fn = local if mesh is None else jax.shard_map(
        local,
        mesh=mesh,
        check_vma=False,
        in_specs=(P(None, "model"), P(None, "model"), P(None, "model"),
                  P(None, None, "model"), P(None, "model", None, None),
                  P("model", None, None),
                  P("model") if has_conv_bias else None, P("model"),
                  P("model"), P(), P(), P(), P()),
        out_specs=(P(None, None, "model"), P(None, "model", None, None),
                   P(None, "model")))
    op = pallas.jax_op(f"qwen4_exp::gdn_highest_{prefix.replace('.', '_')}",
                       fn,
                       donate_argnums=(3, 4))

    @op.register_fake
    def fake(qkv, b, a, conv_state, recurrent_state, *args):
        return (torch.empty_like(conv_state),
                torch.empty_like(recurrent_state),
                qkv.new_empty((qkv.shape[0], n_v * 128)))

    def run(qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            state_indices,
            query_start_loc,
            distribution,
            seq_lens,
            slot_read_offsets=None):
        new_conv, new_state, output = op(qkv, b, a, conv_state,
                                         recurrent_state, conv_weight,
                                         conv_bias, a_log, dt_bias,
                                         state_indices, query_start_loc,
                                         distribution, seq_lens)
        conv_state.copy_(new_conv)
        recurrent_state.copy_(new_state)
        return output

    return run


class Qwen4ExpGatedDeltaNetAttention(VllmGatedDeltaNetAttention):
    """Reuse projections, layer identity, cache binding and output gating.

    Model admission owns the S1 no-PCP/no-MTP/private-cache restriction.
    This class is deliberately not registered as a global OOT replacement.
    """

    def get_state_dtype(self):
        return torch.float32, torch.float32

    def get_state_shape(self):
        return gdn_state_shapes(self.tp_size)

    def _build_gdn_op(self, *, pcp_streaming=False):
        assert not pcp_streaming, "Qwen4Exp admission must reject PCP"
        assert self.num_spec == 0, "Qwen4Exp S1 admission must reject MTP"
        assert (self.num_k_heads, self.num_v_heads, self.head_k_dim,
                self.head_v_dim, self.conv_kernel_size) == (16, 48, 128, 128,
                                                            4)
        return build_gdn_op(self.prefix,
                            self.tp_size,
                            mesh=get_vllm_model_wrapper_context().mesh,
                            has_conv_bias=self.conv1d.bias is not None)

    def _build_pooled_gdn_op(self):
        return None
