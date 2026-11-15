# SPDX-License-Identifier: Apache-2.0
"""Attention-DP verify DCP layout: 4-row chunks with 3 ring slots."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.megakernel.kimi_k3.kernels import decode_megakernel as dm
from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("rows", [16, 32, 48, 64])
def test_verify_layout_is_four_rows_three_slots(rows):
    assert dm._verify_dcp_layout(rows, 6, 4) == {"chunk_rows": 4, "ring": 3}


@pytest.mark.parametrize(
    "rows,heads,sequence_rows",
    [(64, 6, 1), (64, 3, 4), (8, 6, 4), (64, 6, 8)],
)
def test_other_shapes_keep_the_default(rows, heads, sequence_rows):
    assert dm._verify_dcp_layout(rows, heads, sequence_rows) == {}


def test_ring_override():
    assert mla_dcp.chunk_layout(64, 6) == (4, 2)
    assert mla_dcp.chunk_layout(64, 6, ring=3) == (4, 3)
    assert mla_dcp.chunk_layout(32, 6, chunk_rows=4, ring=3) == (4, 3)


@pytest.mark.parametrize("rows", [16, 32, 48, 64])
def test_persistent_scratch_uses_three_slots(rows):
    layout = dm._verify_dcp_layout(rows, 6, 4)
    gathered = mla_dcp.persistent_scratch(
        rows, heads=6, cache_dtype=jnp.float8_e4m3fn, sequence_rows=4, **layout
    )[0]
    assert gathered.shape[0] == 3
