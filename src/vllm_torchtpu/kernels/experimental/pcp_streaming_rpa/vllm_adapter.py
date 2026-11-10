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
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import meta_pack
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME,
    TP_AXIS_NAME,
    sharded_pcp_all_rpa_metadata,
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
# TPU_PCP_RPA_HOIST_META: the per-forward metadata op takes the attention op's
# inputs minus value and returns the packed metadata (one [pcp, total] int32
# tensor, row r = PCP rank r's metadata); the hoisted attention op takes it as
# a ninth tensor.
PCP_META_PACKED_PARTITION_SPEC = PartitionSpec(PCP_AXIS_NAME)
PCP_META_INPUT_PARTITION_SPECS = (
    PCP_STREAMING_RPA_INPUT_PARTITION_SPECS[:3]
    + PCP_STREAMING_RPA_INPUT_PARTITION_SPECS[4:]
)
PCP_META_OUTPUT_PARTITION_SPECS = (PCP_META_PACKED_PARTITION_SPEC,)


def build_pcp_streaming_callable(
    name: str,
    fn: Callable[..., object],
    *,
    donate_argnums: Sequence[int] | None,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Sequence[PartitionSpec] | None = None,
    keep_unused: bool = False,
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

    ``keep_unused`` keeps every tensor argument in the exported module, as
    ``pallas.custom_jax_kernel`` always does. torch_tpu calls the module with
    every tensor the op takes, so an op whose fn reads some tensors only for
    their shapes (the TPU_PCP_RPA_HOIST_META meta op reads kv_cache, query and
    key that way) must keep them: jax.jit's default prunes them, and torch_tpu
    then fails with "'func.call' op incorrect number of operands for callee:
    expected 4, but got 7" (glgt2 j-561ee46d). The 8-tensor entry reads all of
    its tensors, so it keeps the default and its module is unchanged.
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
    jit_kwargs = {}
    trace_kwargs = {}
    if keep_unused:
        jit_kwargs["keep_unused"] = True
        trace_kwargs["keep_unused"] = True
    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        **jit_kwargs,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            **trace_kwargs,
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
    keep_unused: bool = False,
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
        keep_unused=keep_unused,
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
    # The JaxCallable behind the op, for tests that export it as torch_tpu does.
    result.pcp_jax_callable = wrapped_fn
    return result


def pcp_meta_jax_op(
    name: str,
    meta_kernel: Callable[..., jax.Array],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
):
    """Register the TPU_PCP_RPA_HOIST_META metadata op (see make_pcp_meta_kernel).

    * keep_unused: kv_cache, query and key are read for their shapes only;
      jax.jit's default prunes them and torch_tpu's 7-operand call then fails
      ("expected 4, but got 7", glgt2 j-561ee46d).
    * A fake that never reads the token dim: the default fake turns the
      dynamic token SymInt into an int, which specializes it ("You marked
      L['input_ids'].size()[0] as dynamic but your code specialized it to be a
      constant (512)", stagebc j-e49c50b5). The packed output is padded to a
      width computed at the configured token cap (``meta_kernel.width_cap``),
      so its shape depends on static dims only.
    """
    if getattr(meta_kernel, "width_cap", None) is None:
        raise ValueError("pcp_meta_jax_op needs make_pcp_meta_kernel(..., max_tokens=...)")
    op = pcp_streaming_jax_op(
        name,
        meta_kernel,
        donate_argnums=None,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=PCP_META_OUTPUT_PARTITION_SPECS,
        keep_unused=True,
    )
    specs = tuple(input_partition_specs)

    def _static_global(t, spec, token_dim):
        # Global jax shape of a torch tensor without reading its token dim.
        dims = [1 if token_dim and i == 0 else int(d) for i, d in enumerate(t.shape)]
        g = list(pallas_impl.get_global_shape(tuple(dims), mesh, spec))
        if token_dim:
            g[0] = 0  # width_cap replaces it with the cap
        return jax.ShapeDtypeStruct(tuple(g), pallas_impl.TORCH_TO_JAX_DTYPE_MAP[t.dtype])

    def _fake_pcp_meta_op(*tensors):
        shapes = [
            _static_global(t, spec, token_dim=i in (1, 2))
            for i, (t, spec) in enumerate(zip(tensors, specs))
        ]
        width = meta_kernel.width_cap(*shapes)
        local_rows = 1  # [pcp, width] sharded over pcp
        return tensors[0].new_empty((local_rows, width), dtype=torch.int32)

    op.register_fake(_fake_pcp_meta_op)
    return op


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
    q_block_size: int | None = None,
    q_compute_size: int | None = None,
    kv_layout: batched_rpa_configs.KVLayout = (
        batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    block_major: bool = False,
    hoisted: bool = False,
    meta_max_tokens: int | None = None,
    final_rows_history: bool = False,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Build a PCP streaming RPA entry with only tensor args in its signature.

    With ``hoisted`` (TPU_PCP_RPA_HOIST_META, all-RPA only) the entry takes a
    ninth tensor, the packed per-forward metadata from
    ``make_pcp_meta_kernel``'s op, and passes it as ``precomputed``. With
    ``meta_max_tokens`` that tensor is padded to the width at that global token
    cap (it must be exactly that wide) and only its leading part is read.

    When ``block_major`` is True, ``kv_cache`` is the merged block-major pool
    ``(num_blocks, rows_per_block, *page)``, which the wrapper folds into flat
    kernel blocks along dim 0 via zero-copy bitcasts before and after the kernel.
    """
    if soft_cap is not None:
        raise NotImplementedError("PCP streaming RPA does not support logits soft cap.")
    if skip_kv_update:
        raise NotImplementedError("PCP streaming RPA does not support skip_kv_update.")
    if final_rows_history and hoisted:
        raise ValueError("TPU_PCP_MTP_FINAL_ROWS_HISTORY runs on the per-call (non-hoisted) entry.")
    # Read once when the op is built (trace time), like the other kernel flags.
    history_rpa_mode = bool(envs.TPU_PCP_HISTORY_RPA)
    current_rpa_mode = bool(envs.TPU_PCP_CURRENT_RPA)

    def _local_q_block_size(query):
        local_q_block_size = q_block_size
        if local_q_block_size is None:
            # Bound the FP32 output accumulator to 16 MiB, preserving the
            # established 512-row tile for 32 heads x 256 channels. Q and
            # output DMA buffers each use half that storage in BF16.
            tp_size = 1 if mesh is None else mesh.shape.get(TP_AXIS_NAME, 1)
            local_heads = query.shape[-2] // tp_size
            bytes_per_row = local_heads * query.shape[-1] * 4
            local_q_block_size = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE
            while (
                local_q_block_size * bytes_per_row > 16 * 1024 * 1024
                and local_q_block_size // 2 >= cp_kv_cache_interleave_size
            ):
                local_q_block_size //= 2
        return local_q_block_size

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
        local_q_block_size = _local_q_block_size(query)
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
            q_block_size=local_q_block_size,
            q_compute_size=q_compute_size,
            kv_layout=kv_layout,
            history_rpa_mode=history_rpa_mode,
            current_rpa_mode=current_rpa_mode,
            **({"final_rows_history": True} if final_rows_history else {}),
        )
        if block_major:
            new_pool = new_pool.reshape(kv_cache.shape)
        return new_pool, output

    if not hoisted:
        return _pcp_streaming_rpa_kernel
    if not (history_rpa_mode and current_rpa_mode):
        raise ValueError("TPU_PCP_RPA_HOIST_META needs TPU_PCP_HISTORY_RPA=1 and TPU_PCP_CURRENT_RPA=1.")

    def _pcp_streaming_rpa_kernel_hoisted(
        kv_cache: jax.Array,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        meta_packed: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        pool = kv_cache.reshape((-1,) + kv_cache.shape[2:]) if block_major else kv_cache
        md = (seq_lens, block_tables, query_start_loc, request_distribution)
        layout = jax.eval_shape(meta_fn, pool, query, key, *md)
        if meta_max_tokens is not None:
            cap = _meta_width_cap(meta_fn, meta_max_tokens, pool, query, key, *md)
            if meta_packed.shape[1] != cap:
                raise ValueError(
                    f"packed PCP metadata width {meta_packed.shape[1]} != padded width {cap}"
                )
            if not meta_pack.body_unpack_enabled():
                meta_packed = meta_packed[:, : meta_pack.packed_width(layout)]
        if meta_pack.body_unpack_enabled():
            # TPU_PCP_META_BODY_UNPACK: unpacked per rank inside the shard_map (wrapper.py).
            precomputed = meta_pack.PackedMeta(meta_packed, layout)
        else:
            precomputed = meta_pack.unpack(meta_packed, layout)
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
            q_block_size=_local_q_block_size(query),
            q_compute_size=q_compute_size,
            kv_layout=kv_layout,
            history_rpa_mode=True,
            current_rpa_mode=True,
            precomputed=precomputed,
        )
        if block_major:
            new_pool = new_pool.reshape(kv_cache.shape)
        return new_pool, output

    meta_fn = _pcp_meta_fn(
        mesh=mesh, sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size, kv_layout=kv_layout,
    )
    return _pcp_streaming_rpa_kernel_hoisted


_META_WIDTH_CAP_CACHE: dict = {}


def _meta_width_cap(meta_fn, max_tokens, pool, query, key, *md):
    """Packed metadata width at the global token cap ``max_tokens``, from the
    other inputs' static shapes. Raises if this call's token count exceeds the
    cap or its packed width exceeds the cap's (so padding can never truncate)."""
    if query.shape[0] > max_tokens:
        raise ValueError(
            f"PCP metadata: {query.shape[0]} tokens exceed the padded cap {max_tokens}"
        )
    sds = lambda x, n0=None: jax.ShapeDtypeStruct(
        x.shape if n0 is None else (n0,) + tuple(x.shape[1:]), x.dtype
    )
    key_ = (id(meta_fn), max_tokens) + tuple(
        (tuple(x.shape[1:]) if i in (1, 2) else tuple(x.shape), str(x.dtype))
        for i, x in enumerate((pool, query, key) + md)
    )
    cap = _META_WIDTH_CAP_CACHE.get(key_)
    if cap is None:
        cap = meta_pack.packed_width(jax.eval_shape(
            meta_fn, sds(pool), sds(query, max_tokens), sds(key, max_tokens),
            *(sds(x) for x in md),
        ))
        _META_WIDTH_CAP_CACHE[key_] = cap
    if query.shape[0]:
        width = meta_pack.packed_width(jax.eval_shape(
            meta_fn, sds(pool), sds(query), sds(key), *(sds(x) for x in md)))
        if width > cap:
            raise ValueError(
                f"PCP metadata width {width} at {query.shape[0]} tokens exceeds the "
                f"width {cap} at the cap {max_tokens}"
            )
    return cap


def _pcp_meta_fn(*, mesh, sm_scale, k_scale, v_scale, cp_kv_cache_interleave_size, kv_layout):
    """The metadata function both hoisted ops share (one layout, by construction)."""

    def fn(pool, query, key, seq_lens, block_tables, query_start_loc, request_distribution):
        return sharded_pcp_all_rpa_metadata(
            mesh, query, key, pool, seq_lens, block_tables, query_start_loc,
            request_distribution, sm_scale, k_scale=k_scale, v_scale=v_scale,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size, kv_layout=kv_layout,
        )

    return fn


def make_pcp_meta_kernel(
    *,
    k_scale: float | None,
    v_scale: float | None,
    mesh: jax.sharding.Mesh,
    sm_scale: float | None,
    cp_kv_cache_interleave_size: int,
    kv_layout: batched_rpa_configs.KVLayout,
    block_major: bool = False,
    max_tokens: int | None = None,
) -> Callable[..., jax.Array]:
    """TPU_PCP_RPA_HOIST_META: the once-per-forward metadata op's entry.

    Returns the all-RPA metadata (row maps, write-back mapping, segment
    descriptors, current-pool indices, history per-source metadata, capture
    index and the three MIXED RPA schedules) packed as [pcp, total] int32.
    With ``max_tokens`` (a global token cap) the result is zero-padded to the
    width at that cap, and the entry carries ``width_cap(*global_shapes)``, so
    the torch op's output shape does not depend on the token count.
    """
    meta_fn = _pcp_meta_fn(
        mesh=mesh, sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size, kv_layout=kv_layout,
    )

    def _pcp_meta_kernel(
        kv_cache: jax.Array,
        query: jax.Array,
        key: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> jax.Array:
        pool = kv_cache.reshape((-1,) + kv_cache.shape[2:]) if block_major else kv_cache
        md = (seq_lens, block_tables, query_start_loc, request_distribution)
        packed = meta_pack.pack(meta_fn(pool, query, key, *md))
        if max_tokens is not None:
            cap = _meta_width_cap(meta_fn, max_tokens, pool, query, key, *md)
            packed = jax.numpy.pad(packed, ((0, 0), (0, cap - packed.shape[1])))
        return packed

    if max_tokens is not None:

        def width_cap(kv_cache, query, key, seq_lens, block_tables, query_start_loc,
                      request_distribution):
            pool = kv_cache
            if block_major:
                pool = jax.ShapeDtypeStruct(
                    (kv_cache.shape[0] * kv_cache.shape[1],) + tuple(kv_cache.shape[2:]),
                    kv_cache.dtype,
                )
            return _meta_width_cap(
                meta_fn, max_tokens, pool, query, key, seq_lens, block_tables,
                query_start_loc, request_distribution,
            )

        _pcp_meta_kernel.width_cap = width_cap
    return _pcp_meta_kernel


def invoke_pcp_streaming_op(rpa_kernel_op, kv_cache, args, kwargs):
    """Call the 8-tensor PCP streaming custom op from generic RPA args."""
    if kwargs:
        raise ValueError(
            "PCP streaming RPA kernel does not accept keyword tensor args: "
            f"{tuple(kwargs)}."
        )
    hoisted = len(args) == _PCP_STREAMING_RPA_TENSOR_ARG_COUNT
    if len(args) != _PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1 and not hoisted:
        raise ValueError(
            "PCP streaming RPA kernel expects "
            f"{_PCP_STREAMING_RPA_TENSOR_ARG_COUNT - 1} tensor args "
            f"after kv_cache (plus the packed metadata when hoisted), got {len(args)}."
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
    if hoisted:
        pcp_streaming_args += (args[8],)  # packed per-forward metadata
    return rpa_kernel_op(kv_cache, *pcp_streaming_args)
