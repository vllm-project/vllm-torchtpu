# SPDX-License-Identifier: Apache-2.0
"""Experimental sorted FP8 writer preparation; production writer is unchanged."""

import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import cache


def prepare(keys, columns, order):
    """Quantize and reorder, yielding the original writer's exact layout."""
    count = keys.shape[0]
    padded = pl.cdiv(count, 128) * 128

    def kernel(key_ref, column_ref, order_ref, column_out, value_out):
        order = order_ref[:]
        values = jnp.take_along_axis(
            key_ref[:, :], jnp.broadcast_to(order[:, None], (padded, 128)), axis=0
        ).astype(jnp.float32)
        amax = jnp.maximum(jnp.max(jnp.abs(values), axis=1), 1e-4)
        exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
        scale = jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)
        quantized = (values / scale[:, None]).astype(jnp.float8_e4m3fn)
        bits = jax.lax.bitcast_convert_type(quantized, jnp.uint8).astype(jnp.uint32)
        bits = bits.reshape(padded, 32, 4)
        words = bits[:, :, 0] | (bits[:, :, 1] << 8)
        words |= (bits[:, :, 2] << 16) | (bits[:, :, 3] << 24)
        packed = jnp.concatenate(
            (
                words.astype(jnp.int32),
                jax.lax.bitcast_convert_type(scale, jnp.int32)[:, None],
            ),
            axis=1,
        )
        active = jnp.arange(padded) < count
        value_out[:, :] = jnp.where(active[:, None], packed, 0).T
        column_out[:] = jnp.where(active, columns_gather(column_ref[:], order), 0)

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded,), jnp.int32),
            jax.ShapeDtypeStruct((33, padded), jnp.int32),
        ),
        in_specs=(
            pl.BlockSpec((padded, 128), lambda: (0, 0)),
            pl.BlockSpec((padded,), lambda: (0,)),
            pl.BlockSpec((padded,), lambda: (0,)),
        ),
        out_specs=(pl.BlockSpec(), pl.BlockSpec()),
        name="indexer_sorted_fp8_prepare",
    )(
        keys,
        jnp.pad(columns, (0, padded - count)),
        jnp.pad(order, (0, padded - count)),
    )


def columns_gather(columns, order):
    # A singleton non-gather axis becomes an offset dimension in JAX gather,
    # which Mosaic rejects. Eight rows retain the supported batching contract.
    shape = (8, columns.shape[0])
    return jnp.take_along_axis(
        jnp.broadcast_to(columns[None, :], shape),
        jnp.broadcast_to(order[None, :], shape),
        axis=1,
    )[0]


def original_writer_with_prepared_inputs():
    """AST substitution changes only preparation, preserving kernel and aliases."""
    tree = ast.parse(inspect.getsource(cache.write_packed_pages))
    fn = tree.body[0]
    start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "valid"
    )
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.FunctionDef))
    original_kernel = ast.dump(fn.body[end], include_attributes=False)
    fn.args.kwonlyargs.append(ast.arg(arg="prepared"))
    fn.args.kw_defaults.append(ast.Constant(None))
    fn.body[start:end] = ast.parse("bases, columns, values = prepared").body
    assert original_kernel == ast.dump(
        next(n for n in fn.body if isinstance(n, ast.FunctionDef)),
        include_attributes=False,
    )
    namespace = dict(vars(cache))
    exec(
        compile(ast.fix_missing_locations(tree), str(Path(__file__)), "exec"), namespace
    )
    return namespace["write_packed_pages"]


_writer = original_writer_with_prepared_inputs()


def write_fp8_records(banks, records, keys, layout):
    layout.validate(tuple(bank.shape[0] for bank in banks))
    assert keys.shape == (288, 128) and records.shape == (288,)
    base, columns, valid = cache._record_addresses(records, layout)
    base = jnp.where(valid, base, -1)
    total_rows = sum(b.shape[0] for b in banks)
    first_rows = banks[0].shape[0]
    valid = (base >= 0) & (base + layout.page_rows <= first_rows)
    if len(banks) == 2:
        valid |= (base >= first_rows) & (base + layout.page_rows <= total_rows)
    bases, order = jax.lax.sort_key_val(
        jnp.where(valid, base, total_rows), jnp.arange(288, dtype=jnp.int32)
    )
    sorted_columns, values = prepare(keys, columns, order)
    bases = jnp.pad(bases, (0, 97), constant_values=total_rows)
    return _writer(
        banks,
        base,
        columns,
        jnp.zeros((288, 33), jnp.int32),
        page_size=layout.page_size,
        prepared=(bases, sorted_columns, values),
    )
