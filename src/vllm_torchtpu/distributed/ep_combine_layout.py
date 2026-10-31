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
"""Row pieces of the EP combine's reduce-scatter.

The combine (``TpuDeviceCommunicator.combine``) can reduce-scatter its input in
pieces, so that a piece is sent while the rows of the next one are still being
produced. The token rows are cut into ``COMBINE_BLOCKS`` equal blocks and piece
``j`` takes sub-block ``j`` of every block: for a group size that divides
``COMBINE_BLOCKS``, every piece holds an equal share of every rank's rows, so
it is a reduce-scatter on its own. A split combine's input is piece-major
(the pieces of ``split_pieces`` one after another), so every piece is a plain
slice of it. Torch-free so the JAX unpermute that writes the combine's input
can produce that order.
"""

# Share of every block each piece takes, in _SHARE_UNITS. The first piece is
# small, so its reduce-scatter starts soon after the unpermute ends, and the
# rest is finished while it is sent.
COMBINE_PIECE_SIXTEENTHS = (4, 12)
_SHARE_UNITS = 16
# Row blocks a piece takes one sub-block of: two per rank of an eight-rank
# group, so every piece spans both halves of every rank's slice.
COMBINE_BLOCKS = 16
# Combine inputs that are split, in rows: prefill-sized batches. Other sizes
# are left whole: on tpu7x (libtpu 0.0.50) XLA compiled some of their pieces
# to all-reduces instead of SparseCore reduce-scatters. Re-check the pieces'
# collectives in the compiled HLO before widening this or on a libtpu bump.
COMBINE_SPLIT_ROWS = (4096, 8192)
# Rows of a bf16 (16, 128) tile; sub-blocks stay tile-aligned.
_ROW_TILE = 16


def combine_piece_rows(num_rows: int) -> tuple[int, ...]:
    """Rows each piece takes from every block; empty when not split."""
    if num_rows not in COMBINE_SPLIT_ROWS:
        return ()
    block_rows = num_rows // COMBINE_BLOCKS
    leading = [
        block_rows * share // _SHARE_UNITS // _ROW_TILE * _ROW_TILE
        for share in COMBINE_PIECE_SIXTEENTHS[:-1]
    ]
    return (*leading, block_rows - sum(leading))


def split_pieces(x, piece_rows: tuple[int, ...]) -> list:
    """The pieces of token-ordered rows of ``x`` (torch or JAX), in order.

    ``piece_rows`` counts rows of ``x`` per block, as ``combine_piece_rows``
    does for the combine's input.
    """
    rest = tuple(x.shape[1:])
    blocks = x.reshape(COMBINE_BLOCKS, -1, *rest)
    pieces = []
    start = 0
    for rows in piece_rows:
        pieces.append(blocks[:, start : start + rows].reshape(-1, *rest))
        start += rows
    return pieces
