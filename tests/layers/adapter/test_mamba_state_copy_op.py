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
"""A host-provided Mamba copy pair with an index outside the pool becomes the
`(0, 0)` padding pair before the copy kernel, which does no bounds checks, uses
it: in `pool_adapters.copy_blocks`, which both copy programs call, and in the
row copy before a manager row expands into kernel blocks. The programs run on
the host with the Pallas kernel replaced by a NumPy copy, and the tests check
the pool's bytes: each valid pair copies its row and every other row keeps its
bytes.
"""

import base64
import re
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl

from tests.distributed.tpu_test_utils import run_in_isolated_process
from tests.layers.adapter.mamba_state_copy_test_utils import (
    numpy_copy_blocks,
    numpy_copy_kernel,
    row_copy_program,
)
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op

POOL_ROWS = 10
# In range, then below zero and past the pool: the first row past it and the
# int32 extremes.
INDICES = [0, 5, 9, -1, POOL_ROWS, 2**31 - 1, -(2**31)]

# Two valid pairs (one reads the last row), then pairs that each join a live
# row to an index outside the pool, then the padding pair. Clamping the
# outside index into the pool would change row 0, 3, 8 or 9.
VALID = [(2, 7), (9, 5)]
OUTSIDE = [
    (POOL_ROWS, 3),
    (2**31 - 1, 3),
    (-1, 8),
    (6, POOL_ROWS),
    (6, -(2**31)),
    (6, 2**31 - 1),
]
PAIRS = VALID + OUTSIDE + [(0, 0)]
# A manager row whose `row * steps` wraps in int32 to kernel block 4 when a row
# has 4 steps, that is, onto manager row 1.
WRAPS = 2**30 + 1


def _indices(pairs):
    src = jnp.array([s for s, _ in pairs], dtype=jnp.int32)
    dst = jnp.array([d for _, d in pairs], dtype=jnp.int32)
    return src, dst


def _rows_pool():
    """A pool whose every int32 differs, so any moved byte shows."""
    return jnp.arange(POOL_ROWS * 4 * 3, dtype=jnp.int32).reshape(POOL_ROWS, 4, 3)


def _after_valid_copies(pool):
    out = np.array(pool)
    for s, d in VALID:
        out[d] = np.asarray(pool)[s]
    return out


def test_a_pair_with_an_index_outside_the_pool_becomes_the_null_pair():
    pairs = [(s, d) for s in INDICES for d in INDICES]
    src, dst = _indices(pairs)
    new_src, new_dst = pool_adapters.null_out_of_range_pairs(src, dst, POOL_ROWS)
    assert new_src.dtype == new_dst.dtype == jnp.int32
    in_pool = {0, 5, 9}
    expected = [(s, d) if {s, d} <= in_pool else (0, 0) for s, d in pairs]
    assert list(zip(new_src.tolist(), new_dst.tolist())) == expected


def test_the_block_copy_moves_its_valid_pairs_and_no_other_row():
    pool = _rows_pool()
    with numpy_copy_kernel():
        new_pool, _ = mamba_state_copy_op._copy_fn(pool, *_indices(PAIRS))
    np.testing.assert_array_equal(np.asarray(new_pool), _after_valid_copies(pool))


@pytest.mark.parametrize("step_rows", [1, 2, 4], ids=["steps=4", "steps=2", "steps=1"])
def test_the_row_copy_moves_its_valid_rows_and_no_other_row(step_rows):
    """Each manager row expands into `steps` kernel-block steps. A row outside
    the pool is replaced before that expansion, so one that would wrap in
    int32 back into the pool moves nothing either."""
    pool = _rows_pool()  # 2 regions x 2 kernel blocks per manager row
    pairs = PAIRS + [(WRAPS, 3), (6, WRAPS)]
    program = row_copy_program(num_pools=2, split=2, step_rows=step_rows)
    with numpy_copy_kernel():
        new_pool, _ = program(pool, *_indices(pairs))
    np.testing.assert_array_equal(np.asarray(new_pool), _after_valid_copies(pool))


def test_valid_pairs_reach_the_copy_kernel_unchanged():
    """With every pair inside the pool the guard changes no index, so both
    programs copy exactly what they copied without it."""
    seen = []

    def kernel(pool, src, dst, in_pool=None):
        seen.append((src, dst))
        return numpy_copy_blocks(pool, src, dst)

    src, dst = _indices(VALID + [(0, 0), (9, 9)])
    with patch.object(pool_adapters, "_copy_blocks_unchecked", kernel):
        mamba_state_copy_op._copy_fn(_rows_pool(), src, dst)
        row_copy_program(num_pools=2, split=2, step_rows=2)(_rows_pool(), src, dst)
    (block_src, block_dst), (row_src, row_dst) = seen
    assert block_src.dtype == block_dst.dtype == jnp.int32
    assert (block_src.tolist(), block_dst.tolist()) == (src.tolist(), dst.tolist())

    def expanded(rows, steps=2):
        return [r * steps + o for r in rows.tolist() for o in range(steps)]

    assert (row_src.tolist(), row_dst.tolist()) == (expanded(src), expanded(dst))


@pytest.mark.parametrize("program", ["block", "row"])
def test_the_debug_check_reports_a_replaced_pair(program):
    """On the host the report is a host callback raising `RuntimeError` that
    names the pairs as passed in. The row copy's bad pairs are reported by
    `copy_blocks` on the expanded pool of `POOL_ROWS * steps` kernel-block
    steps, as the marked steps (-1 * steps + offset)."""
    if program == "block":
        run, num_blocks = mamba_state_copy_op._copy_fn, POOL_ROWS
        count, first = 6, r"\(10, 3\)"  # the first pair outside, as passed in
    else:
        run = row_copy_program(num_pools=2, split=2, step_rows=2)
        num_blocks = POOL_ROWS * 2
        count, first = 12, r"\(-2, -2\)"  # 2 steps per marked row; the first step
    # On a TPU device the programs would take the TPU branch, which has no
    # callback: pin the host report to the CPU device.
    with jax.default_device(jax.devices("cpu")[0]), numpy_copy_kernel():
        pool = _rows_pool()
        run(pool, *_indices(PAIRS))  # debug checks off: replaced, not reported
        jax.effects_barrier()
        with pl.enable_debug_checks(True):
            run(pool, *_indices(VALID))  # nothing replaced, nothing reported
            jax.effects_barrier()
            with pytest.raises(
                (RuntimeError, jax.errors.JaxRuntimeError),
                match=re.escape(pool_adapters._replaced_pair_message(num_blocks))
                + rf": {count} pair\(s\), first ones: \[{first}, ",
            ):
                run(pool, *_indices(PAIRS))
                jax.effects_barrier()


def _tpu_kernels(program, pool):
    """Lower `program` for TPU from this host; return the text and each
    Pallas kernel's name and serialized body."""
    idx = jax.ShapeDtypeStruct((8,), jnp.int32)
    lowered = jax.jit(program).trace(pool, idx, idx).lower(lowering_platforms=("tpu",))
    text = lowered.as_text()
    kernels = []
    for config, name in re.findall(
        r'backend_config = "(.*?)", kernel_name = "(\w+)"', text
    ):
        body = re.search(r"body\\22: \\22([A-Za-z0-9+/=]+)", config).group(1)
        kernels.append((name, base64.b64decode(body)))
    return text, kernels


@pytest.mark.parametrize("program", ["block", "row"])
def test_on_tpu_the_debug_check_is_in_the_copy_kernel(program):
    """A host callback made the copy programs fail to compile on TPU. Lowered
    for TPU, the programs hold only the copy kernel and no callback; with the
    flag the copy kernel holds the debug check's message, without it not."""
    if program == "block":
        run, num_blocks = mamba_state_copy_op._copy_fn, POOL_ROWS
        pool = jax.ShapeDtypeStruct((POOL_ROWS, 16, 2, 2, 128), jnp.bfloat16)
    else:
        run = row_copy_program(num_pools=2, split=2, step_rows=2)
        num_blocks = POOL_ROWS * 2  # the copy runs on the expanded pool
        pool = jax.ShapeDtypeStruct((POOL_ROWS, 4, 16, 2, 2, 128), jnp.bfloat16)
    message = pool_adapters._replaced_pair_message(num_blocks).encode()
    for flag in (False, True):
        with pl.enable_debug_checks(flag):
            text, kernels = _tpu_kernels(run, pool)
        assert "callback" not in text
        assert [name for name, _ in kernels] == ["_copy_kernel"]
        assert (message in kernels[0][1]) == flag


def _copies_with_debug_checks_on_tpu():
    # Set before anything is traced, so both programs compile the copy kernel
    # with its debug check.
    jax.config.update("jax_pallas_enable_debug_checks", True)
    gen = torch.Generator().manual_seed(0)

    def host_pool(shape):
        return torch.randn(shape, generator=gen).to(torch.bfloat16)

    def indices(pairs):
        pairs = pairs + [(0, 0)] * (8 - len(pairs))
        return (
            torch.tensor([s for s, _ in pairs], dtype=torch.int32).to("tpu"),
            torch.tensor([d for _, d in pairs], dtype=torch.int32).to("tpu"),
        )

    def expected(host, pairs):
        out = host.clone()
        for s, d in pairs:
            out[d] = host[s]
        return out

    block_pairs = [(5, 7), (9, 12), (63, 3), (20, 21)]
    hosts = [host_pool((64, 16, 2, 2, 128)) for _ in range(2)]
    pools = [host.to("tpu") for host in hosts]
    row_pairs = [(5, 7), (63, 2), (11, 40)]
    host_rows = host_pool((64, 6, 16, 2, 2, 128))
    rows = host_rows.to("tpu")
    with torch.no_grad():
        mamba_state_copy_op.copy_mamba_state_blocks(pools, *indices(block_pairs))
        mamba_state_copy_op.copy_mamba_state_rows(
            rows, *indices(row_pairs), num_pools=3, split=2
        )
    torch.accelerator.synchronize()
    for pool, host in zip(pools, hosts):
        assert torch.equal(pool.cpu(), expected(host, block_pairs))
    assert torch.equal(rows.cpu(), expected(host_rows, row_pairs))


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_with_debug_checks_both_programs_copy_valid_pairs_on_tpu():
    """With `pl.enable_debug_checks`, both programs compile the copy kernel
    with its `pl.debug_check` and run it on TPU. Only valid pairs run: a check
    that fired on them would halt the core and leave the pools uncopied, so
    the test fails. A pair outside the pool is not run here, because the halt
    would fail every later program in the session."""
    run_in_isolated_process(_copies_with_debug_checks_on_tpu)
