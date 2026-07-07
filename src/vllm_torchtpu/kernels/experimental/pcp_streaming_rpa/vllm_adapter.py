# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""vLLM custom-op adapter for PCP streaming RPA."""

import inspect
from collections.abc import Callable, Sequence

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed.pcp import get_or_create_pcp_mesh
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME, sharded_pcp_ragged_paged_attention)

_PCP_STREAMING_RPA_TENSOR_ARG_COUNT = 9

PCP_STREAMING_RPA_INPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME),  # kv_cache: local cache blocks
    PartitionSpec(PCP_AXIS_NAME),  # query: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME),  # key: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME),  # value: local prefill tokens
    PartitionSpec(),  # seq_lens
    PartitionSpec(),  # block_tables
    PartitionSpec(),  # query_start_loc
    PartitionSpec(),  # request_distribution
)
PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME),  # new_kv_cache
    PartitionSpec(PCP_AXIS_NAME),  # output
)


def _torch_placeholder_with_sharding(
        aval: jax.core.ShapedArray | None,
        sharding: jax.sharding.NamedSharding | None,
        mesh: jax.sharding.Mesh | None) -> torch.Tensor | None:
    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(
            f"Unsupported dtype for pallas kernels: {aval.dtype}")
    spec = getattr(sharding, "spec", None)
    if spec is None:
        spec = getattr(getattr(aval, "sharding", None), "spec", None)
    return torch.empty(pallas_impl.get_local_shape(aval.shape, mesh, spec),
                       dtype=torch_dtype,
                       device="tpu")


class _PcpStreamingJaxCallable(pallas_impl.JaxCallable):
    """JaxCallable variant that allocates shard-map outputs with out_shardings."""

    def __call__(self, *args, **kwargs):
        self._validate_args(*args)

        kernel_key = pallas_impl._get_kernel_invocation_key(
            self.trace_key, args, kwargs, self.static_argnums)
        output_shapes, out_tree = self.output_shapes.get(
            kernel_key, (None, None))
        kernel_exists = pallas_impl.tpu_torch_pallas.lookup_custom_kernel(
            self.name, kernel_key)
        if not output_shapes or not kernel_exists:
            jax_args = pallas_impl.jax_placeholders(
                args,
                mesh=self.mesh,
                partition_specs=self.input_partition_specs,
            )
            with jax._src.config.export_ignore_forward_compatibility(True):
                lowered = self.exported(*jax_args, **kwargs)
            pallas_impl.tpu_torch_pallas.register_custom_kernel(
                self.name,
                kernel_key,
                serialized_mlir_module=lowered.mlir_module_serialized,
            )
            out_shardings = getattr(lowered, "_out_named_shardings", None)
            if (out_shardings is None
                    or len(out_shardings) != len(lowered.out_avals)):
                out_shardings = [None] * len(lowered.out_avals)
            output_shapes = [
                _torch_placeholder_with_sharding(aval, sharding, self.mesh)
                for aval, sharding in zip(lowered.out_avals, out_shardings)
            ]
            out_tree = lowered.out_tree
            self.output_shapes[kernel_key] = (output_shapes, out_tree)

        tensor_args = [
            arg for i, arg in enumerate(args)
            if arg is not None and i not in self.static_argnums
        ]
        results = pallas_impl.tpu_torch_pallas.call_custom_kernel(
            self.name,
            kernel_key,
            inputs=tensor_args,
            output_shapes=output_shapes,
            donate_argnums=self.donate_argnums,
        )

        for in_idx, out_idx in self.input_output_aliases.items():
            tensor_args[in_idx].copy_(results[out_idx])

        return out_tree.unflatten(results)


def _named_shardings(mesh: jax.sharding.Mesh,
                     partition_specs: Sequence[PartitionSpec]):
    return tuple(
        jax.sharding.NamedSharding(mesh, spec) for spec in partition_specs)


def pcp_streaming_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Sequence[int] | None,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
):
    """Register a PCP streaming custom op with explicit output sharding.

    TorchTPU's stock pallas.jax_op derives output placeholder shapes from
    lowered.out_avals, but jax.export records shard_map outputs there as
    replicated. Explicit out_shardings are preserved in _out_named_shardings;
    the PCP-specific JaxCallable uses that to allocate local torch outputs.
    """
    signature = inspect.signature(fn, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    output_shardings = _named_shardings(
        mesh, PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS)
    jit_fn = jax.jit(fn,
                     static_argnums=static_argnums,
                     donate_argnums=donate_argnums_tuple,
                     out_shardings=output_shardings)
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums":
            static_argnums,
            "donate_argnums":
            donate_argnums_tuple,
            "output_partition_specs":
            tuple(map(str, PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS)),
        },
    )
    wrapped_fn = _PcpStreamingJaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )

    result = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args, **kwargs):
        jax_args = pallas_impl.jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        out_shardings = getattr(lowered, "_out_named_shardings", None)
        if out_shardings is None or len(out_shardings) != len(
                lowered.out_avals):
            out_shardings = [None] * len(lowered.out_avals)
        return lowered.out_tree.unflatten(
            _torch_placeholder_with_sharding(aval, sharding, mesh)
            for aval, sharding in zip(lowered.out_avals, out_shardings))

    result.register_fake(fake_fn)
    return result


def get_pcp_streaming_mesh():
    return get_or_create_pcp_mesh(axis_name=PCP_AXIS_NAME)


def make_pcp_streaming_rpa_kernel(
    *,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    mesh: jax.sharding.Mesh,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    skip_kv_update: bool,
    cp_kv_cache_interleave_size: int,
    max_model_len: int | None = None,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a PCP streaming RPA entry with only tensor args in its signature."""
    if soft_cap is not None:
        raise NotImplementedError(
            "PCP streaming RPA does not support logits soft cap.")
    if skip_kv_update:
        raise NotImplementedError(
            "PCP streaming RPA does not support skip_kv_update.")

    def _pcp_streaming_rpa_kernel(
        kv_cache: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        output, new_kv_cache = sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=query,
            k=key,
            v=value,
            kv_cache=kv_cache,
            kv_lens=seq_lens,
            page_indices=block_tables,
            cu_q_lens=query_start_loc,
            distribution=request_distribution,
            attention_sink=None,
            sm_scale=sm_scale,
            attention_chunk_size=sliding_window,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            max_context_tokens=max_model_len,
            update_kv_cache=True,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        return new_kv_cache, output

    return _pcp_streaming_rpa_kernel


def invoke_pcp_streaming_op(rpa_kernel_op, kv_cache, args, kwargs):
    """Call the 8-tensor PCP streaming custom op from generic RPA args."""
    if kwargs:
        raise ValueError(
            "PCP streaming RPA kernel does not accept keyword tensor args: "
            f"{tuple(kwargs)}.")
    if len(args) != _PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1:
        raise ValueError(
            "PCP streaming RPA kernel expects "
            f"{_PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1} tensor args "
            f"after kv_cache, got {len(args)}.")
    if args[7] is not None:
        raise NotImplementedError(
            "PCP streaming RPA does not support attention sinks.")
    pcp_streaming_args = (
        args[0],  # query
        args[1],  # key
        args[2],  # value
        args[3],  # seq_lens
        args[4],  # block_tables
        args[5],  # query_start_loc
        args[6],  # request_distribution
    )
    return rpa_kernel_op(kv_cache, *pcp_streaming_args)
