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

import dataclasses
import functools

import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import \
    is_conv_state_dim_first
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

from vllm_torchtpu.distributed.pcp import (get_or_create_pcp_mesh,
                                           get_pcp_rank, get_pcp_world_size)
from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    unified_kv_layout_enabled_for_architecture)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import \
    pcp_streaming_jax_op
from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.core.gdn_attention import (
    run_jax_gdn_attention, run_jax_gdn_attention_pcp_tp_prefill,
    run_jax_gdn_attention_pooled,
    run_jax_gdn_attention_pooled_pcp_prefill_projection)
from vllm_torchtpu.layers.core.sequence_layout import \
    is_pcp_streaming_attention_metadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def _to_jax_ssm_state_dtype(dtype: torch.dtype) -> jnp.dtype:
    if dtype == torch.float32:
        return jnp.dtype(jnp.float32)
    if dtype == torch.bfloat16:
        return jnp.dtype(jnp.bfloat16)
    raise ValueError(f"Unsupported mamba_ssm_cache_dtype for TPU GDN: "
                     f"{dtype}; expected float32 or bfloat16")


def gdn_attention_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    slot_read_offsets: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Speculative decoding: vLLM's MambaSpec widens the conv state by
    # `num_spec` columns per slot. The extra columns are unused on TPU
    # (rollback keeps one full checkpoint per group slot instead of a
    # rolling window); only the first kernel_size - 1 columns hold data, so
    # slice them out for the kernel and write the result back into the same
    # (donated) buffer afterwards.
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, :kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    (new_conv_state, new_recurrent_state), output = run_jax_gdn_attention(
        mixed_qkv,
        b,
        a,
        conv_state_in,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        slot_read_offsets,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        mesh=mesh,
        num_spec_tokens=num_spec_tokens,
    )
    if state_len > kernel_size - 1:
        # Write the kernel result into the first kernel_size - 1 columns of the
        # donated conv_state in place and leave the unused spec tail untouched.
        # A concat here would materialize a fresh full-width array that no
        # longer derives from the donated buffer, defeating the conv-cache
        # input_output_alias (a full-width alloc + copy per GDN layer per step);
        # the dynamic-update-slice keeps the write inside the donated buffer.
        new_conv_state = conv_state.at[:, :kernel_size -
                                       1, :].set(new_conv_state)

    return new_conv_state, new_recurrent_state, output


def gdn_attention_pooled_core_tpu(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    slot_read_offsets: jax.Array | None = None,
    ckpt_indices: jax.Array | None = None,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    recurrent_state_dtype: jnp.dtype,
    num_spec_tokens: int = 0,
) -> tuple[jax.Array, jax.Array]:
    return run_jax_gdn_attention_pooled(
        mixed_qkv,
        b,
        a,
        recurrent_state,
        conv_weight,
        conv_bias,
        A_log,
        dt_bias,
        state_indices,
        query_start_loc,
        distribution,
        seq_lens,
        slot_read_offsets,
        ckpt_indices,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        pool_block_tokens=pool_block_tokens,
        recurrent_state_dtype=recurrent_state_dtype,
        mesh=mesh,
        num_spec_tokens=num_spec_tokens,
    )


def gdn_attention_core_tpu_pcp_prefill(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    A_log: jax.Array,
    dt_bias: jax.Array,
    state_indices: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pcp_size: int,
    interleave_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    state_len = conv_state.shape[1]
    if state_len > kernel_size - 1:
        conv_state_in = conv_state[:, :kernel_size - 1, :]
    else:
        conv_state_in = conv_state

    (new_conv_state,
     new_recurrent_state), output = run_jax_gdn_attention_pcp_tp_prefill(
         mixed_qkv,
         b,
         a,
         conv_state_in,
         recurrent_state,
         conv_weight,
         conv_bias,
         A_log,
         dt_bias,
         state_indices,
         query_start_loc,
         distribution,
         seq_lens,
         n_kq=n_kq,
         n_v=n_v,
         d_k=d_k,
         d_v=d_v,
         kernel_size=kernel_size,
         pcp_size=pcp_size,
         interleave_size=interleave_size,
         mesh=mesh,
     )
    if state_len > kernel_size - 1:
        new_conv_state = conv_state.at[:, :kernel_size -
                                       1, :].set(new_conv_state)
    return new_conv_state, new_recurrent_state, output


def _get_pcp_size(vllm_config: VllmConfig) -> int:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    return pcp_size if pcp_size > 1 else 1


def _localize_gdn_mamba_spec_for_pcp(
    spec: MambaSpec,
    pcp_size: int,
) -> MambaSpec:
    if pcp_size <= 1 or len(spec.shapes) != 2:
        return spec
    if is_conv_state_dim_first():
        raise NotImplementedError("TPU GDN PCP state sharding requires "
                                  "VLLM_SSM_CONV_STATE_LAYOUT=SD.")

    conv_shape = tuple(spec.shapes[0])
    recurrent_shape = tuple(spec.shapes[1])
    if not conv_shape or not recurrent_shape:
        return spec
    if conv_shape[-1] % pcp_size != 0:
        raise ValueError("GDN conv state width must be divisible by PCP size: "
                         f"shape={conv_shape}, pcp_size={pcp_size}")
    if recurrent_shape[0] % pcp_size != 0:
        raise ValueError(
            "GDN recurrent state head count must be divisible by PCP size: "
            f"shape={recurrent_shape}, pcp_size={pcp_size}")

    unpadded_page_size = dataclasses.replace(
        spec, page_size_padded=None).page_size_bytes
    page_size_padded = spec.page_size_padded
    if page_size_padded == unpadded_page_size:
        page_size_padded = None

    return dataclasses.replace(
        spec,
        shapes=(
            (*conv_shape[:-1], conv_shape[-1] // pcp_size),
            (recurrent_shape[0] // pcp_size, *recurrent_shape[1:]),
        ),
        page_size_padded=page_size_padded,
    )


@QwenGatedDeltaNetAttention.register_oot
class VllmGatedDeltaNetAttention(QwenGatedDeltaNetAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # `_require_pcp_projection_parameters` hands this weight straight to
        # `fused_qkvz_projection_pcp_gdn`, which validates it as [n_out, n_in].
        if hasattr(self, "in_proj_qkvz"):
            setattr(self.in_proj_qkvz, KEEP_VLLM_LAYOUT_ATTR, True)
        # Bound by the runner during KV-cache initialization; None until then
        # (warmup/profiling runs check this).
        self.kv_cache = None
        self.gdn_op = self._build_gdn_op()
        self.gdn_pooled_op = self._build_pooled_gdn_op()
        pcp_enabled = self._pcp_streaming_enabled()
        self._pcp_streaming_configured = pcp_enabled
        self.gdn_pcp_op = (self._build_gdn_op(
            pcp_streaming=True) if pcp_enabled else None)
        self.gdn_pooled_pcp_op = (self._build_pooled_pcp_gdn_op()
                                  if pcp_enabled else None)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        assert isinstance(spec, MambaSpec)
        return _localize_gdn_mamba_spec_for_pcp(spec,
                                                _get_pcp_size(vllm_config))

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        conv_state_dtype, temporal_state_dtype = super().get_state_dtype()
        if unified_kv_layout_enabled_for_architecture(
                self.model_config.architecture):
            # Declare exactly what the pool physically stores. vLLM sizes the
            # mamba page from the dtype declared HERE and asserts the padded
            # page covers it, while the block-slot derivation budgets the page
            # from DEFAULT_POOLED_GDN_CONV_STATE_DTYPE. Declaring anything
            # wider is not conservative, it is a contradiction: the two
            # disagreed once and every hybrid GDN engine failed to start on
            # that assert.
            conv_state_dtype = getattr(
                torch,
                DEFAULT_POOLED_GDN_CONV_STATE_DTYPE.split(".")[1])
        else:
            # TODO: Support bf16 conv state.
            # Per-layer cache: unlike the pool, the declared dtype IS the
            # storage dtype (vLLM strides the raw tensor with it), and this
            # kernel path still reads fp32 conv state. Nothing pads this
            # layout, so the wider declaration costs only its own bytes.
            conv_state_dtype = torch.float32
        return conv_state_dtype, temporal_state_dtype

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Back the conv weight with a buffer of its own 3-D shape.

        The upstream layer rebinds `conv1d.weight` to an `unsqueeze(1)` view
        of the 2-D ColumnParallelLinear buffer and the sharded loader fills it
        through row slices, so the device buffer behind the Parameter stays
        2-D. torch_tpu hands a compiled executable the buffer behind each
        argument and re-materializes a shape-mismatched argument as a
        standalone `tt_jit_as_strided` program before every forward. One
        contiguous copy at load time removes that per-step program.
        """
        del act_dtype
        w = self.conv1d.weight
        self.conv1d.weight.data = torch.empty(w.shape,
                                              dtype=w.dtype,
                                              device=w.device).copy_(w)

    def get_state_shape(self, ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        conv_state_shape, temporal_state_shape = super().get_state_shape()
        conv_state_shape = conv_state_shape[:-1] + (1, conv_state_shape[-1])
        return conv_state_shape, temporal_state_shape

    @staticmethod
    def _pcp_streaming_enabled() -> bool:
        vllm_context = get_vllm_model_wrapper_context()
        if vllm_context.vllm_config is None:
            return False
        parallel_config = vllm_context.vllm_config.parallel_config
        return parallel_config.prefill_context_parallel_size > 1

    def _build_gdn_op(self, *, pcp_streaming: bool = False):
        local_num_v_heads = self.num_v_heads // self.tp_size
        local_num_kq_heads = self.num_k_heads // self.tp_size
        has_conv_bias = self.conv1d.bias is not None
        vllm_context = get_vllm_model_wrapper_context()
        # The non-PCP op uses num_spec_tokens for verify/rollback. PCP is a
        # prefill-only path admitted at the platform boundary; its widened
        # conv-state tail is preserved by gdn_attention_core_tpu_pcp_prefill.
        num_spec_tokens = self.num_spec

        def _fake_gdn(mixed_qkv, _b, _a, conv_state, recurrent_state, *args,
                      **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            return torch.empty_like(conv_state), torch.empty_like(
                recurrent_state), torch.empty(out_shape,
                                              dtype=mixed_qkv.dtype,
                                              device=mixed_qkv.device)

        if pcp_streaming:
            parallel_config = vllm_context.vllm_config.parallel_config
            interleave_size = parallel_config.cp_kv_cache_interleave_size
            if not isinstance(interleave_size, int) or interleave_size <= 0:
                raise ValueError("GDN PCP streaming requires "
                                 "cp_kv_cache_interleave_size > 0.")
            pcp_mesh = get_or_create_pcp_mesh(axis_name="pcp")
            pcp_size = get_pcp_world_size()
            if pcp_size != pcp_mesh.shape["pcp"]:
                raise ValueError(
                    f"PCP world size {pcp_size} does not match mesh shape "
                    f"{pcp_mesh.shape['pcp']}.")
            if local_num_kq_heads % pcp_size != 0:
                raise ValueError("local GDN K/Q heads "
                                 f"{local_num_kq_heads} must be divisible by "
                                 f"pcp_size={pcp_size}.")
            if local_num_v_heads % pcp_size != 0:
                raise ValueError(
                    f"local GDN V heads {local_num_v_heads} must be divisible "
                    f"by pcp_size={pcp_size}.")
            op_name = ("pallas::gdn_attention_pcp_"
                       f"{self.prefix.replace('.', '_')}")
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu_pcp_prefill,
                mesh=pcp_mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
            )
            input_partition_specs = (
                PartitionSpec("pcp"),  # mixed_qkv
                PartitionSpec("pcp"),  # b
                PartitionSpec("pcp"),  # a
                PartitionSpec(None, None, "pcp"),  # conv_state
                PartitionSpec(None, "pcp", None, None),  # recurrent_state
                PartitionSpec(),  # conv_weight
                PartitionSpec() if has_conv_bias else None,  # conv_bias
                PartitionSpec(),  # A_log
                PartitionSpec(),  # dt_bias
                PartitionSpec(),  # state_indices
                PartitionSpec(),  # query_start_loc
                PartitionSpec(),  # request_distribution
                PartitionSpec(),  # seq_lens
            )
            output_partition_specs = (
                PartitionSpec(None, None, "pcp"),  # new_conv_state
                PartitionSpec(None, "pcp", None, None),  # new_recurrent_state
                PartitionSpec("pcp"),  # output
            )
            gdn_jax_op = pcp_streaming_jax_op(
                op_name,
                wrapped_fn,
                donate_argnums=(3, 4),
                mesh=pcp_mesh,
                input_partition_specs=input_partition_specs,
                output_partition_specs=output_partition_specs,
            )
        else:
            op_name = f"pallas::gdn_attention_{self.prefix.replace('.', '_')}"
            wrapped_fn = functools.partial(
                gdn_attention_core_tpu,
                mesh=vllm_context.mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=self.head_k_dim,
                d_v=self.head_v_dim,
                kernel_size=self.conv_kernel_size,
                num_spec_tokens=num_spec_tokens,
            )
            gdn_jax_op = pallas.jax_op(op_name,
                                       wrapped_fn,
                                       donate_argnums=(3, 4))

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            conv_state: torch.Tensor,
            recurrent_state: torch.Tensor,
            conv_weight: torch.Tensor,
            conv_bias: torch.Tensor | None,
            A_log: torch.Tensor,
            dt_bias: torch.Tensor,
            state_indices: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            seq_lens: torch.Tensor,
            slot_read_offsets: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # The PCP prefill op keeps its original 13-arg signature. Its
            # speculative state tail is not a rollback operand.
            extra_args = (() if pcp_streaming else (slot_read_offsets, ))
            new_conv, new_rec, outputs = gdn_jax_op(
                mixed_qkv, b, a, conv_state, recurrent_state, conv_weight,
                conv_bias, A_log, dt_bias, state_indices, query_start_loc,
                request_distribution, seq_lens, *extra_args)

            conv_state.copy_(new_conv)
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def _build_pooled_gdn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        local_num_v_heads = self.num_v_heads // self.tp_size
        vllm_config = vllm_context.vllm_config
        mesh = vllm_context.mesh
        n_kq = self.num_k_heads // self.tp_size
        d_k = self.head_k_dim
        d_v = self.head_v_dim
        kernel_size = self.conv_kernel_size
        recurrent_state_dtype = _to_jax_ssm_state_dtype(
            self.get_state_dtype()[1])
        # Speculative decoding: verify windows run in the GDN kernel's SPEC
        # mode with one state checkpoint per window position kept inside
        # the request's state block; rejected drafts are rolled back by
        # checkpoint selection (see TPUModelRunner.mamba_slot_read_offsets).
        num_spec_tokens = self.num_spec

        # Written out rather than functools.partial so pool_block_tokens is
        # read per call instead of at build time; pallas.jax_op requires a
        # fully annotated signature, so the operands are spelled out.
        def wrapped_fn(
            mixed_qkv: jax.Array,
            b: jax.Array,
            a: jax.Array,
            recurrent_state: jax.Array,
            conv_weight: jax.Array,
            conv_bias: jax.Array | None,
            A_log: jax.Array,
            dt_bias: jax.Array,
            state_indices: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            slot_read_offsets: jax.Array | None = None,
            ckpt_indices: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array]:
            return gdn_attention_pooled_core_tpu(
                mixed_qkv,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                slot_read_offsets,
                ckpt_indices,
                mesh=mesh,
                n_kq=n_kq,
                n_v=local_num_v_heads,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                # Manager block size: the pool may be born at a smaller
                # kernel granularity for backends with a fixed kernel
                # block. Read lazily at first call: the executor finalizes
                # cache_config.block_size only AFTER load_model, and hybrid
                # models construct GDN layers before the first
                # full-attention layer would ever see the adjusted value.
                pool_block_tokens=vllm_config.cache_config.block_size,
                recurrent_state_dtype=recurrent_state_dtype,
                num_spec_tokens=num_spec_tokens,
            )

        op_name = f"pallas::gdn_attention_pooled_{self.prefix.replace('.', '_')}"
        # The recurrent state (arg 3) is the attention-shaped pool: the ssm
        # and conv byte-regions are read/written through it by the pool
        # adapters, so it is the only donated input. inplace-donation
        # aliases its program-level output to the input buffer so the
        # ~pool-sized update is not double-counted at compile.
        gdn_jax_op = pallas.jax_op(op_name, wrapped_fn, donate_argnums=(3, ))

        def _fake_gdn(mixed_qkv, _b, _a, recurrent_state, *args, **kwargs):
            num_tokens = mixed_qkv.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            out = torch.empty(out_shape,
                              dtype=mixed_qkv.dtype,
                              device=mixed_qkv.device)
            return torch.empty_like(recurrent_state), out

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(mixed_qkv: torch.Tensor,
                     b: torch.Tensor,
                     a: torch.Tensor,
                     recurrent_state: torch.Tensor,
                     conv_weight: torch.Tensor,
                     conv_bias: torch.Tensor | None,
                     A_log: torch.Tensor,
                     dt_bias: torch.Tensor,
                     state_indices: torch.Tensor,
                     query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor,
                     seq_lens: torch.Tensor,
                     slot_read_offsets: torch.Tensor | None = None,
                     ckpt_indices: torch.Tensor | None = None) -> torch.Tensor:
            new_rec, outputs = gdn_jax_op(mixed_qkv, b, a, recurrent_state,
                                          conv_weight, conv_bias, A_log,
                                          dt_bias, state_indices,
                                          query_start_loc,
                                          request_distribution, seq_lens,
                                          slot_read_offsets, ckpt_indices)

            # Plain donation + copy_ writeback (aliased in-place by XLA).
            recurrent_state.copy_(new_rec)

            return outputs

        return gdn_impl

    def _build_pooled_pcp_gdn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        local_num_v_heads = self.num_v_heads // self.tp_size
        local_num_kq_heads = self.num_k_heads // self.tp_size
        has_conv_bias = self.conv1d.bias is not None
        parallel_config = vllm_context.vllm_config.parallel_config
        interleave_size = parallel_config.cp_kv_cache_interleave_size
        if not isinstance(interleave_size, int) or interleave_size <= 0:
            raise ValueError("GDN pooled PCP prefill requires "
                             "cp_kv_cache_interleave_size > 0.")
        pcp_mesh = get_or_create_pcp_mesh(axis_name="pcp")
        pcp_size = get_pcp_world_size()
        if pcp_size != pcp_mesh.shape["pcp"]:
            raise ValueError(
                f"PCP world size {pcp_size} does not match mesh shape "
                f"{pcp_mesh.shape['pcp']}.")
        if local_num_kq_heads % pcp_size != 0:
            raise ValueError("local GDN K/Q heads "
                             f"{local_num_kq_heads} must be divisible by "
                             f"pcp_size={pcp_size}.")
        if local_num_v_heads % pcp_size != 0:
            raise ValueError(
                f"local GDN V heads {local_num_v_heads} must be divisible "
                f"by pcp_size={pcp_size}.")

        vllm_config = vllm_context.vllm_config
        d_k = self.head_k_dim
        d_v = self.head_v_dim
        kernel_size = self.conv_kernel_size
        recurrent_state_dtype = _to_jax_ssm_state_dtype(
            self.get_state_dtype()[1])

        def wrapped_fn(
            hidden_states: jax.Array,
            qkvz_weight: jax.Array,
            b: jax.Array,
            a: jax.Array,
            recurrent_state: jax.Array,
            conv_weight: jax.Array,
            conv_bias: jax.Array | None,
            A_log: jax.Array,
            dt_bias: jax.Array,
            state_indices: jax.Array,
            query_start_loc: jax.Array,
            distribution: jax.Array,
            seq_lens: jax.Array,
            qkvz_weight_scale: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return run_jax_gdn_attention_pooled_pcp_prefill_projection(
                hidden_states,
                qkvz_weight,
                qkvz_weight_scale,
                b,
                a,
                recurrent_state,
                conv_weight,
                conv_bias,
                A_log,
                dt_bias,
                state_indices,
                query_start_loc,
                distribution,
                seq_lens,
                mesh=pcp_mesh,
                n_kq=local_num_kq_heads,
                n_v=local_num_v_heads,
                d_k=d_k,
                d_v=d_v,
                kernel_size=kernel_size,
                pool_block_tokens=vllm_config.cache_config.block_size,
                pcp_size=pcp_size,
                interleave_size=interleave_size,
                recurrent_state_dtype=recurrent_state_dtype,
            )

        op_name = ("pallas::gdn_attention_pooled_pcp_fused_"
                   f"{self.prefix.replace('.', '_')}")
        input_partition_specs = (
            PartitionSpec("pcp"),  # hidden states
            PartitionSpec(),  # QKVZ weight
            PartitionSpec("pcp"),  # b
            PartitionSpec("pcp"),  # a
            PartitionSpec("pcp"),  # recurrent_state pool (rank-local, axis 0)
            PartitionSpec(),  # conv_weight
            PartitionSpec() if has_conv_bias else None,  # conv_bias
            PartitionSpec(),  # A_log
            PartitionSpec(),  # dt_bias
            PartitionSpec(),  # state_indices
            PartitionSpec(),  # query_start_loc
            PartitionSpec(),  # request_distribution
            PartitionSpec(),  # seq_lens
            PartitionSpec(),  # optional QKVZ weight scale
        )
        output_partition_specs = (
            PartitionSpec("pcp"),  # new pool
            PartitionSpec("pcp"),  # output
            PartitionSpec("pcp"),  # z
        )
        gdn_jax_op = pcp_streaming_jax_op(
            op_name,
            wrapped_fn,
            # Keep optional operands after the pool so filtering None at the
            # TorchTPU boundary cannot change the donated tensor's index.
            donate_argnums=(4, ),
            mesh=pcp_mesh,
            input_partition_specs=input_partition_specs,
            output_partition_specs=output_partition_specs,
        )

        def _fake_gdn(hidden_states, _weight, _b, _a, recurrent_state, *args,
                      **kwargs):
            num_tokens = hidden_states.size(0)
            out_shape = (num_tokens, local_num_v_heads, self.head_v_dim)
            output = torch.empty(out_shape,
                                 dtype=hidden_states.dtype,
                                 device=hidden_states.device)
            return torch.empty_like(recurrent_state), output, torch.empty_like(
                output)

        gdn_jax_op.register_fake(_fake_gdn)

        def gdn_impl(
                hidden_states: torch.Tensor, qkvz_weight: torch.Tensor,
                qkvz_weight_scale: torch.Tensor | None, b: torch.Tensor,
                a: torch.Tensor, recurrent_state: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor | None,
                A_log: torch.Tensor, dt_bias: torch.Tensor,
                state_indices: torch.Tensor, query_start_loc: torch.Tensor,
                request_distribution: torch.Tensor,
                seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            new_rec, outputs, z = gdn_jax_op(hidden_states, qkvz_weight, b, a,
                                             recurrent_state, conv_weight,
                                             conv_bias, A_log, dt_bias,
                                             state_indices, query_start_loc,
                                             request_distribution, seq_lens,
                                             qkvz_weight_scale)
            # Match the non-PCP pooled path: donation aliases the Pallas
            # result to the input pool, while copy_ exposes the mutation to
            # the surrounding compiled graph without rebinding its storage.
            recurrent_state.copy_(new_rec)
            return outputs, z

        return gdn_impl

    def _require_pcp_projection_parameters(
            self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return BF16 or tensor/channel/block-scaled FP8 QKVZ parameters."""
        if (hasattr(self, "in_proj_qkv") or not hasattr(self, "in_proj_qkvz")
                or self.gqa_interleaved_layout):
            raise RuntimeError("GDN pooled PCP prefill requires the Qwen3.5 "
                               "non-interleaved QKVZ layout.")

        projection = self.in_proj_qkvz
        weight = getattr(projection, "weight", None)
        weight_scale = getattr(projection, "weight_scale", None)
        bias = getattr(projection, "bias", None)
        if weight is None or weight.ndim != 2 or bias is not None:
            raise RuntimeError(
                "GDN pooled PCP prefill requires bias-free rank-2 QKVZ weights."
            )
        if weight.dtype == torch.bfloat16:
            if weight_scale is not None:
                raise RuntimeError("BF16 QKVZ weights must not have a scale.")
        elif weight.dtype == torch.float8_e4m3fn:
            if weight_scale is None or weight_scale.dtype != torch.float32:
                raise RuntimeError("FP8 QKVZ weights require an FP32 scale.")
            from vllm_torchtpu.kernels.gdn.v3.projection_scale import \
                projection_scale_layout
            try:
                projection_scale_layout(weight.shape, weight_scale.shape)
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
        else:
            raise RuntimeError("GDN pooled PCP prefill requires BF16 or "
                               "float8_e4m3fn QKVZ weights.")
        return weight, weight_scale

    def _has_active_pooled_pcp_state(self, kv_cache) -> bool:
        """Whether a PCP-configured worker owns an allocated unified pool."""
        return (kv_cache is not None and len(kv_cache) == 1
                and kv_cache[0].numel() > 0 and self._pcp_streaming_configured)

    def _apply_output_projection(self, core_attn_out: torch.Tensor,
                                 z: torch.Tensor,
                                 num_tokens: int) -> torch.Tensor:
        local_num_v_heads = self.num_v_heads // self.tp_size
        core_attn_out = core_attn_out.view(num_tokens, local_num_v_heads,
                                           self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output, _ = self.out_proj(core_attn_out)
        return output

    def _forward_fused_pooled_pcp(self, hidden_states: torch.Tensor,
                                  state_pool: torch.Tensor,
                                  attn_metadata) -> torch.Tensor:
        """Run the mandatory fused path for an allocated pooled PCP call."""
        if not is_pcp_streaming_attention_metadata(attn_metadata):
            raise RuntimeError(
                "PCP is configured with a unified pool, but attention "
                "metadata does not use the PCP streaming sequence layout.")

        gdn_pooled_pcp_op = self.gdn_pooled_pcp_op
        assert gdn_pooled_pcp_op is not None
        qkvz_weight, qkvz_weight_scale = (
            self._require_pcp_projection_parameters())
        ba, _ = self.in_proj_ba(hidden_states)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        state_indices = attn_metadata.mamba_state_indices
        assert state_indices is not None
        core_attn_out, z = gdn_pooled_pcp_op(
            hidden_states,
            qkvz_weight,
            qkvz_weight_scale,
            b,
            a,
            state_pool,
            self.conv1d.weight,
            self.conv1d.bias,
            self.A_log,
            self.dt_bias,
            state_indices.to(torch.int32),
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
            attn_metadata.seq_lens,
        )
        num_tokens = hidden_states.size(0)
        if core_attn_out.shape[0] != num_tokens:
            raise RuntimeError("GDN op returned an incompatible output shape.")
        return self._apply_output_projection(core_attn_out, z, num_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        kv_cache = self.kv_cache
        if self._has_active_pooled_pcp_state(kv_cache):
            attn_metadata = get_forward_context().attn_metadata[self.prefix]
            return self._forward_fused_pooled_pcp(
                hidden_states,
                kv_cache[0],
                attn_metadata,
            )

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkv"):
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            if self.gqa_interleaved_layout:
                # Qwen3-Next: unpack the interleaved GQA layout
                query, key, value, z, b, a = self.fix_query_key_value_ordering(
                    mixed_qkvz, ba)
                query, key, value = map(
                    lambda x: rearrange(x, "l p d -> l (p d)"),
                    (query, key, value))
                mixed_qkv = torch.cat((query, key, value), dim=-1)
            else:
                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
                z_size = self.value_dim // self.tp_size
                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
                z = z.reshape(z.size(0), -1, self.head_v_dim)
                b, a = ba.chunk(2, dim=-1)
                b = b.contiguous()
                a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        local_num_v_heads = self.num_v_heads // self.tp_size

        # During warmup or memory profiling, the kv_cache might not be allocated yet
        if kv_cache is None or kv_cache[0].numel() == 0:
            core_attn_out = torch.zeros(
                (num_tokens, local_num_v_heads, self.head_v_dim),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )
            recurrent_state = None
        else:
            attn_metadata = get_forward_context().attn_metadata[self.prefix]

            if len(kv_cache) == 1:
                # Unified block pool: the attention-shaped pool carries the
                # ssm and conv byte-regions; state ids come from the pool
                # metadata builder.
                (recurrent_state, ) = kv_cache
                assert attn_metadata.mamba_state_indices is not None
                state_indices = attn_metadata.mamba_state_indices.to(
                    torch.int32)
                # Speculative decoding: the GDN kernel's windowed segment
                # covers both 1-token decodes and speculative verify
                # windows (the batch is ordered [decode][verify][prefill]);
                # ragged paged attention keeps `request_distribution` with
                # its 1-token decode front segment.
                request_distribution = attn_metadata.request_distribution
                slot_read_offsets = getattr(attn_metadata,
                                            "mamba_slot_read_offsets", None)
                mamba_request_distribution = getattr(
                    attn_metadata, "mamba_request_distribution", None)
                if mamba_request_distribution is not None:
                    request_distribution = mamba_request_distribution
                if self.num_spec > 0:
                    assert slot_read_offsets is not None, (
                        "Speculative decoding with GDN layers requires "
                        "mamba_slot_read_offsets in the attention metadata.")
                ckpt_indices = getattr(attn_metadata, "mamba_ckpt_indices",
                                       None)
                core_attn_out = self.gdn_pooled_op(
                    mixed_qkv, b, a, recurrent_state, self.conv1d.weight,
                    self.conv1d.bias, self.A_log, self.dt_bias, state_indices,
                    attn_metadata.query_start_loc, request_distribution,
                    attn_metadata.seq_lens, slot_read_offsets, ckpt_indices)
                if core_attn_out.shape[0] != num_tokens:
                    raise RuntimeError(
                        "GDN op returned an incompatible output shape.")
            else:
                conv_state, recurrent_state = kv_cache
                # Recurrent-state slot id per persistent-batch position.
                # Compact-mamba: the mamba pool has only `_mamba_num_blocks` slots
                # (< attention `num_blocks`), so the slot id is carried explicitly
                # in `mamba_state_indices` (in [0, _mamba_num_blocks)) rather than
                # derived from the attention `block_tables[:, 0]`. Fall back to
                # `block_tables[:, 0]` only when compact sizing was skipped and
                # mamba shares the attention block pool (uniform layout).
                if attn_metadata.mamba_state_indices is not None:
                    state_indices = attn_metadata.mamba_state_indices.to(
                        torch.int32)
                else:
                    max_reqs = attn_metadata.seq_lens.shape[0]
                    max_blocks_per_req = (
                        attn_metadata.block_tables.shape[0] // max_reqs)
                    block_tables_2d = torch.reshape(
                        attn_metadata.block_tables,
                        (max_reqs, max_blocks_per_req),
                    )
                    state_indices = block_tables_2d[:, 0].to(torch.int32)

                # Speculative decoding: the GDN kernel's windowed segment
                # covers both 1-token decodes and speculative verify windows
                # (the batch is ordered [decode][verify][prefill/mixed]);
                # ragged paged attention keeps using `request_distribution`
                # with its 1-token decode front segment.
                request_distribution = attn_metadata.request_distribution
                slot_read_offsets = getattr(attn_metadata,
                                            "mamba_slot_read_offsets", None)
                mamba_request_distribution = getattr(
                    attn_metadata, "mamba_request_distribution", None)
                if mamba_request_distribution is not None:
                    request_distribution = mamba_request_distribution
                use_pcp_streaming = is_pcp_streaming_attention_metadata(
                    attn_metadata)
                if not use_pcp_streaming and self.num_spec > 0:
                    assert slot_read_offsets is not None, (
                        "Speculative decoding with GDN layers requires "
                        "mamba_slot_read_offsets in the attention metadata.")

                # Execute the TorchTPU custom op
                if not use_pcp_streaming:
                    core_attn_out = self.gdn_op(
                        mixed_qkv, b, a, conv_state, recurrent_state,
                        self.conv1d.weight, self.conv1d.bias, self.A_log,
                        self.dt_bias, state_indices,
                        attn_metadata.query_start_loc, request_distribution,
                        attn_metadata.seq_lens, slot_read_offsets)
                else:
                    gdn_pcp_op = self.gdn_pcp_op
                    if gdn_pcp_op is None:
                        raise RuntimeError(
                            "GDN PCP prefill op was not initialized during model "
                            "loading.")
                    core_attn_out = gdn_pcp_op(
                        mixed_qkv, b, a, conv_state, recurrent_state,
                        self.conv1d.weight, self.conv1d.bias, self.A_log,
                        self.dt_bias, state_indices,
                        attn_metadata.query_start_loc,
                        attn_metadata.request_distribution,
                        attn_metadata.seq_lens)
                if core_attn_out.shape[0] != num_tokens:
                    if not use_pcp_streaming:
                        raise RuntimeError(
                            "GDN op returned an incompatible output shape.")
                    start = get_pcp_rank() * num_tokens
                    core_attn_out = core_attn_out[start:start + num_tokens]

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        return self._apply_output_projection(core_attn_out, z, num_tokens)
