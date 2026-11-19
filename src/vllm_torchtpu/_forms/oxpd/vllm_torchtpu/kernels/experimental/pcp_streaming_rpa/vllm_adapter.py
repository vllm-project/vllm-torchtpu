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

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import get_or_create_pcp_mesh
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME,
    TP_AXIS_NAME,
    sharded_pcp_ragged_paged_attention,
)

_PCP_STREAMING_RPA_TENSOR_ARG_COUNT = 9

PCP_STREAMING_RPA_INPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME, None, TP_AXIS_NAME),  # kv_cache
    PartitionSpec(PCP_AXIS_NAME, TP_AXIS_NAME),  # query: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME, TP_AXIS_NAME),  # key: local prefill tokens
    PartitionSpec(PCP_AXIS_NAME, TP_AXIS_NAME),  # value: local prefill tokens
    PartitionSpec(),  # seq_lens
    PartitionSpec(),  # block_tables
    PartitionSpec(),  # query_start_loc
    PartitionSpec(),  # request_distribution
)
PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS = (
    PartitionSpec(PCP_AXIS_NAME, None, TP_AXIS_NAME),  # new_kv_cache
    PartitionSpec(PCP_AXIS_NAME, TP_AXIS_NAME),  # output
)


def build_pcp_streaming_callable(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Sequence[int] | None,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec] | None = None,
) -> pallas_impl.JaxCallable:
    """Build the eager JaxCallable for a PCP streaming kernel fn.

    Split out of ``pcp_streaming_jax_op`` so kernel-iteration hot-reload can
    rebuild the callable from freshly reloaded kernel modules and swap it in
    behind the already-registered torch op.

    This assembles exactly what ``pallas.jax_op`` assembles internally, so a
    hot-reloaded callable behaves like the one the normal path registers.

    It used to return a local JaxCallable subclass that sized shard_map
    outputs itself, because Torchtpu used to derive them from
    ``lowered.out_avals`` and jax.export records shard_map outputs there as
    replicated. Torchtpu now supports ``output_partition_specs``, which is
    the same information by a shorter route, so a subclass is not needed.
    The subclass had also overridden ``__call__`` outright, so it silently
    missed whatever Torchtpu's own ``__call__`` did -- including, before
    google-pytorch/torch_tpu#3522, the step that made the kernel's
    DeviceAssignment follow the caller's mesh rather than PJRT enumeration
    order. #3522 makes that assignment canonical, so there is no longer a
    per-call argument to miss; using the base JaxCallable unmodified is what
    keeps a future one from being missed the same way.
    """
    signature = inspect.signature(fn, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    donate_argnums_tuple = tuple(donate_argnums or ())
    output_partition_specs_tuple = tuple(
        output_partition_specs or PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS
    )
    # No out_shardings on the jit, because pallas.jax_op does not set it
    # either and this has to build the same callable it does. The shard_map
    # body already declares its out_specs, so pinning them again only added an
    # advisory sdy.sharding result attribute to the exported module -- and the
    # partitioner is a no-op here because shard_map is already manual.
    jit_fn = jax.jit(
        fn, static_argnums=static_argnums, donate_argnums=donate_argnums_tuple
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
        },
    )
    return pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        output_partition_specs=output_partition_specs_tuple,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
    )


def pcp_streaming_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Sequence[int] | None,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec] | None = None,
):
    """Register a PCP streaming custom op behind a hot-reload dispatcher.

    Everything else here is what ``pallas.jax_op`` does; the one thing it
    does not do is register the op against ``kernel_reload``'s dispatcher, so
    that reload_kernels() can swap the callable without re-registering the
    torch op. Callers that do not use TPU_KERNEL_ITER_MODE -- the GDN PCP ops,
    for instance -- should call ``pallas.jax_op`` directly rather than this.
    """
    wrapped_fn = build_pcp_streaming_callable(
        name,
        fn,
        donate_argnums=donate_argnums,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
    )

    # Kernel-iteration mode registers the op against a dispatcher indirection
    # so hot-reload can swap the callable without re-registering the op.
    if envs.TPU_KERNEL_ITER_MODE:
        from vllm_torchtpu.compilation import kernel_reload

        kernel_reload.set_live(name, wrapped_fn)
        op_target = kernel_reload.make_dispatcher(name, wrapped_fn)
    else:
        op_target = wrapped_fn

    result = torch.library.custom_op(name, op_target, mutates_args=())

    output_specs = tuple(
        output_partition_specs or PCP_STREAMING_RPA_OUTPUT_PARTITION_SPECS
    )

    def fake_fn(*args, **kwargs):
        # Mirrors stock's fake kernel: size the outputs from the specs the
        # caller declared, not from lowered.out_avals, which jax.export
        # records as replicated for shard_map results.
        jax_args = pallas_impl.jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            pallas_impl.torch_placeholder(aval, mesh=mesh, partition_spec=spec)
            for aval, spec in zip(lowered.out_avals, output_specs)
        )

    result.register_fake(fake_fn)
    return result


def get_pcp_streaming_mesh():
    return get_or_create_pcp_mesh(axis_name=PCP_AXIS_NAME, tp_axis_name=TP_AXIS_NAME)


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
    q_block_size: int = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    q_compute_size: int | None = None,
    kv_layout: batched_rpa_configs.KVLayout = (
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    block_major: bool = False,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a PCP streaming RPA entry with only tensor args in its signature.

    When ``block_major`` is True, ``kv_cache`` is the merged block-major pool
    ``(num_blocks, rows_per_block, *page)``, which the wrapper folds into flat
    kernel blocks along dim 0 via zero-copy bitcasts before and after the kernel.
    """
    if soft_cap is not None:
        raise NotImplementedError("PCP streaming RPA does not support logits soft cap.")
    if skip_kv_update:
        raise NotImplementedError("PCP streaming RPA does not support skip_kv_update.")

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
        # Fold the block-major pool so kernel blocks run along dim 0
        # (zero-copy bitcast).
        pool = kv_cache.reshape((-1,) + kv_cache.shape[2:]) if block_major else kv_cache
        output, new_pool = sharded_pcp_ragged_paged_attention(
            mesh=mesh,
            q=query,
            k=key,
            v=value,
            kv_cache=pool,
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
            update_kv_cache=True,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            q_block_size=q_block_size,
            q_compute_size=q_compute_size,
            kv_layout=kv_layout,
        )
        if block_major:
            new_pool = new_pool.reshape(kv_cache.shape)
        return new_pool, output

    return _pcp_streaming_rpa_kernel


def invoke_pcp_streaming_op(rpa_kernel_op, kv_cache, args, kwargs):
    """Call the 8-tensor PCP streaming custom op from generic RPA args."""
    if kwargs:
        raise ValueError(
            "PCP streaming RPA kernel does not accept keyword tensor args: "
            f"{tuple(kwargs)}."
        )
    if len(args) != _PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1:
        raise ValueError(
            "PCP streaming RPA kernel expects "
            f"{_PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1} tensor args "
            f"after kv_cache, got {len(args)}."
        )
    if args[7] is not None:
        raise NotImplementedError("PCP streaming RPA does not support attention sinks.")
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
