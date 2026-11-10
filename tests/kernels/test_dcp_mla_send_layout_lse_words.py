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
"""CPU-interpret regression test for `build_send_lse_words` when `T < W`.

When `num_tokens < send_world`, `send_rows = num_tokens * (num_q_heads //
send_world) < num_q_heads` (for example `T=2, W=8, H=64 -> send_rows=16 < 64`,
`T=4, W=8, H=64 -> send_rows=32 < 64` and `T=4, W=8, H=32 -> send_rows=16 <
32`). `build_send_lse_words` must keep
the block-reversed `hi` (`[0, num_q_heads)`) and `lo` (`[send_rows, send_rows +
num_q_heads)`) LSE word sources separate instead of packing them into a single
shared row where lanes `[send_rows, num_q_heads)` overwrite `lo` with `hi`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import sparse_mla_cpu_interpret as cpu_interpret  # noqa: E402
import test_dcp_mla_send_layout as send_layout_test  # noqa: E402
import test_mla_decode_prologue as prologue  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla  # noqa: E402

PAD = sparse_mla.SEND_PAD_ROWS


@pytest.mark.parametrize(
    ("world", "heads", "cell"),
    [
        (8, 64, (2, (2,))),
        (8, 64, (4, (3,))),
        (8, 32, (4, (3,))),
        (16, 64, (8, (4, 3))),
        (8, 64, (8, (4, 3))),
        (8, 64, (16, (4, 4, 4, 1))),
    ],
    ids=[
        "w8_h64_t2",
        "w8_h64_t4",
        "w8_h32_t4",
        "w16_h64_t8",
        "w8_h64_t8_ctrl",
        "w8_h64_t16_ctrl",
    ],
)
def test_send_layout_lse_words_match_packed_reference(
    world: int, heads: int, cell: tuple[int, tuple[int, ...]]
) -> None:
    query_tokens, seq_q = cell
    real = int(sum(seq_q))
    # The small-chunk grid step, as the adapter derives it (2 at T=2).
    batch = sparse_mla._largest_divisor(query_tokens, 4)
    assert sparse_mla.send_layout_supported(query_tokens, heads, world, 512, batch)
    p = cpu_interpret.compact_problem(query_tokens, seq_q, heads)
    with cpu_interpret.cpu_interpret_context():
        out, lse = prologue._kernel(p, candidate=True)
        send = send_layout_test._kernel_send(p, world)

    per = heads // world
    rows = query_tokens * per
    assert send.shape == (world, PAD + rows, out.shape[-1])

    nw, no = send_layout_test._diffs(
        send, prologue._bits16(out), np.asarray(lse), world, real
    )
    assert nw == 0 and no == 0, f"W={world} H={heads} T={query_tokens}: {nw} / {no}"

    # Verify unpacked LSE bit patterns on real rows contain no spurious NaNs and
    # preserve both -inf (empty row 0) and finite LSEs bit for bit.
    words = send.view(np.int16)[:, PAD - 1]
    hi = words[:, :rows].astype(np.int32)
    lo = words[:, rows : 2 * rows].astype(np.int32)
    lse_unpacked = ((hi << 16) | (lo & 0xFFFF)).view(np.float32)[:, : real * per]
    assert not np.isnan(lse_unpacked).any()
    assert np.isneginf(lse_unpacked).any()
    assert np.isfinite(lse_unpacked).any()
    assert not send.view(np.int16)[:, : PAD - 1].any()


def test_send_layout_lse_lo_skip_fault_caught_at_t4() -> None:
    p = cpu_interpret.compact_problem(4, (3,), 64)
    with cpu_interpret.cpu_interpret_context():
        out, lse = prologue._kernel(p, candidate=True)
        send = send_layout_test._kernel_send(p, 8, fault="send_lse_lo_skip")
    nw, no = send_layout_test._diffs(send, prologue._bits16(out), np.asarray(lse), 8, 3)
    assert nw > 0 and no == 0
