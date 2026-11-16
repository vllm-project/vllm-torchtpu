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
"""CPU-interpret harness for the SparseCore-layout sparse MLA kernel.

Lets kernel tests run `sparse_ragged_paged_attention` with the SparseCore
cache layout on CPU (Pallas TPU interpret mode) by replacing the SparseCore
gather with a CPU stand-in.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather


def dsa_gather_cpu_interpret(
    nope_cache,
    rope_cache,
    indices,
    *,
    out_size=None,
    atoms_per_batch=16,
    num_live_indices=None,
    live_multiple=1,
    live_vec=None,
    live_vec_offset=None,
    nope_words=128,
    **_unused,
):
    """CPU stand-in for `native_sc_gather.dsa_gather_native_sc`.

    Gathers NoPE rows and banded RoPE quads for `indices`, and poisons rows
    past the gathered live bound with `0x7F7F7F7F`. It re-derives the live
    bound and occupancy math of the SparseCore gather. It only needs to be
    self-consistent: tests compare two kernel paths that both read through
    this stand-in, so drift in its live-bound math cannot hide a word-layout
    or merge bug.
    """
    if out_size is None:
        out_size = indices.size
    num_cores, num_subcores, atoms_per_batch, occ, num_batches = (
        native_sc_gather.live_vec_geometry(out_size, atoms_per_batch)
    )
    num_owners = num_cores * num_subcores
    guard_tail = live_multiple % (num_owners * occ) != 0
    if live_vec_offset is not None:
        lv = indices.reshape(-1)[
            live_vec_offset : live_vec_offset + 2 * native_sc_gather.NUM_LANES
        ]
        eff = lv[0] if guard_tail else lv[native_sc_gather.NUM_LANES] * num_owners
        gathered = jnp.minimum(eff * occ, out_size)
    elif live_vec is not None:
        eff = (
            live_vec[0]
            if guard_tail
            else live_vec[native_sc_gather.NUM_LANES] * num_owners
        )
        gathered = jnp.minimum(eff * occ, out_size)
    elif num_live_indices is not None:
        lb = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occ - 1) // occ, 0, num_batches
        )
        ob = (lb + num_owners - 1) // num_owners
        gathered = jnp.minimum((lb if guard_tail else ob * num_owners) * occ, out_size)
    else:
        gathered = jnp.int32(out_size)
    poison = np.uint32(0x7F7F7F7F)
    flat_idx = indices.reshape(-1)[:out_size]
    nope_flat = nope_cache.reshape(-1, nope_words)
    rope_flat = rope_cache.reshape(-1, native_sc_gather.ROPE_WORDS)
    safe = jnp.clip(flat_idx, 0, nope_flat.shape[0] - 1)
    nope_out = jnp.where(
        jnp.arange(out_size, dtype=jnp.int32)[:, None] < gathered,
        nope_flat[safe],
        poison,
    )
    quads = rope_flat[safe].reshape(
        out_size // native_sc_gather.ROPE_PACKING,
        native_sc_gather.ROPE_PACKING,
        native_sc_gather.ROPE_WORDS,
    )
    bands = native_sc_gather.transpose_quad(
        [quads[:, s, :] for s in range(native_sc_gather.ROPE_PACKING)]
    )
    rope_out = jnp.concatenate(bands, axis=-1)
    rope_out = jnp.where(
        jnp.arange(rope_out.shape[0], dtype=jnp.int32)[:, None]
        < (gathered // native_sc_gather.ROPE_PACKING),
        rope_out,
        poison,
    )
    return nope_out, rope_out


@contextlib.contextmanager
def cpu_interpret_context():
    """Runs the sparse MLA kernel in Pallas TPU interpret mode on CPU.

    Patches the chip info to TPU7x and the SparseCore gather with
    `dsa_gather_cpu_interpret`. On a TPU it does nothing.
    """
    if jax.devices()[0].platform == "tpu":
        yield
        return
    pltpu.reset_tpu_interpret_mode_state()
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    with (
        mock.patch.object(pltpu, "get_tpu_info", return_value=info),
        mock.patch.object(
            native_sc_gather, "dsa_gather_native_sc", dsa_gather_cpu_interpret
        ),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield
