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
"""CPU tests for the native-SC gather's per-row live-atom schedule."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather as G


def _visited(lanes, atoms_per_batch, row_atoms):
    """Batch atoms the kernel's inner grid visits, per logical batch."""
    out = []
    for vals in np.asarray(lanes):
        starts = [jnp.int32(v) for v in vals[1:]]
        steps = range(int(vals[0]))
        out.append(
            [int(G.live_atom_of(jnp.int32(s), starts, row_atoms)) for s in steps]
        )
    return out


def _expected(per_row, num_batches, atoms_per_batch, row_atoms):
    live = np.arange(row_atoms)[None, :] < np.clip(per_row, 0, row_atoms)[:, None]
    flat = live.reshape(num_batches, atoms_per_batch)
    return [list(np.nonzero(b)[0]) for b in flat]


@pytest.mark.parametrize(
    "row_atoms,atoms_per_batch",
    [(8, 16), (8, 32), (4, 16), (2, 16), (8, 8), (8, 1), (8, 4), (4, 2)],
)
def test_visits_exactly_the_live_atoms(row_atoms, atoms_per_batch):
    rng = np.random.default_rng(row_atoms * 100 + atoms_per_batch)
    num_rows = 64
    num_batches = num_rows * row_atoms // atoms_per_batch
    # 0 live, 1, a full row, and everything between; plus out-of-range clamps.
    per_row = rng.integers(0, row_atoms + 1, size=num_rows)
    per_row[:4] = [0, 1, row_atoms, 0]
    per_row[4] = row_atoms + 3
    per_row[5] = -2
    lanes = G.live_atom_lanes(
        jnp.asarray(per_row, jnp.int32), num_batches, atoms_per_batch, row_atoms
    )
    assert lanes.shape == (num_batches, max(1, atoms_per_batch // row_atoms))
    got = _visited(lanes, atoms_per_batch, row_atoms)
    want = _expected(per_row, num_batches, atoms_per_batch, row_atoms)
    assert got == want
    # Positive control: a shifted profile must disagree.
    assert want != _expected(
        np.roll(per_row, 1), num_batches, atoms_per_batch, row_atoms
    )


def test_all_live_and_all_dead():
    for fill, total in ((8, 16), (0, 0)):
        lanes = np.asarray(
            G.live_atom_lanes(jnp.full((32,), fill, jnp.int32), 16, 16, 8)
        )
        assert (lanes[:, 0] == total).all()


def test_rejects_non_dividing_batch():
    with pytest.raises(ValueError, match="divide"):
        G.live_atom_lanes(jnp.zeros((12,), jnp.int32), 8, 12, 8)
