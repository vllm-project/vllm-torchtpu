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
"""The exchange schedule simulated on the CPU for every width: each
representative ends with every block exactly once, the pieces a link
carries are the same on every link in every round, the bytes per link meet
the all-gather lower bound, and the landing order the receiver assumes is
the order the sender issues."""

import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import transport

pytestmark = pytest.mark.cpu_test

WIDTHS = (1, 2, 4, 8, 16, 32)


def simulate(width):
    """Runs the static schedule for every representative. Returns the
    pieces each representative holds per pair and the pieces per link per
    round."""
    n, n_pieces = transport.dimensions(width), transport.pieces(width)
    representatives = range(0, width, 2)
    held = {rep: {rep >> 1: set(range(n_pieces))} for rep in representatives}
    loads = {}
    for t in range(1, n + 1):
        landed = {rep: {} for rep in representatives}
        for rep in representatives:
            for link, relative, pieces in transport.sends(n, t, n_pieces):
                pair = (rep >> 1) ^ relative
                assert set(pieces) <= held[rep].get(pair, set()), (
                    "a piece forwarded before it was held",
                    width,
                    t,
                    rep,
                )
                loads[(t, link)] = loads.get((t, link), 0) + len(pieces)
                landed[rep ^ (2 << link)].setdefault(pair, set()).update(pieces)
        for rep in representatives:
            for link in range(n):
                expected = transport.arrivals(n, t, n_pieces, link)
                got = {
                    (pair ^ (rep >> 1), p)
                    for pair, ps in landed[rep].items()
                    for p in ps
                    if transport.landing_link(pair ^ (rep >> 1), p, n_pieces) == link
                }
                assert set(expected) == got and len(expected) == len(got), (
                    "arrivals() disagrees with the simulation",
                    width,
                    t,
                )
            for pair, ps in landed[rep].items():
                assert not held[rep].get(pair, set()) & ps, (
                    "a piece delivered twice",
                    width,
                    t,
                    rep,
                    pair,
                )
                held[rep].setdefault(pair, set()).update(ps)
    return held, loads


class TransportScheduleTest(parameterized.TestCase):
    @parameterized.parameters(*WIDTHS)
    def test_every_representative_holds_every_block_once(self, width):
        held, _ = simulate(width)
        n_pieces = transport.pieces(width)
        for rep, blocks in held.items():
            self.assertEqual(set(blocks), set(range((width + 1) // 2)), rep)
            for pieces in blocks.values():
                self.assertEqual(pieces, set(range(n_pieces)))

    @parameterized.parameters(*WIDTHS)
    def test_links_carry_the_same_bytes_at_the_bound(self, width):
        n, n_pieces = transport.dimensions(width), transport.pieces(width)
        _, loads = simulate(width)
        representatives = width // 2
        for t in range(1, n + 1):
            per_link = {loads[(t, link)] for link in range(n)}
            self.assertLen(per_link, 1, f"round {t} links differ")
        blocks_per_link = (
            sum(loads.get((t, 0), 0) for t in range(1, n + 1))
            / n_pieces
            / representatives
            if n
            else 0
        )
        bound = (2**n - 1) / n if n else 0
        self.assertAlmostEqual(blocks_per_link, bound)

    @parameterized.parameters(*WIDTHS)
    def test_parts_are_whole_pieces(self, width):
        n, n_pieces = transport.dimensions(width), transport.pieces(width)
        for t in range(1, n + 1):
            self.assertEqual(n_pieces % t, 0)
            for _, _, pieces in transport.sends(n, t, n_pieces):
                self.assertEqual(len(pieces), n_pieces // t)
        self.assertEqual(transport.receive_semaphores(width), 1 + n * n + n)
        self.assertEqual(transport.token_multiple(width), n_pieces // 2)

    def test_eight_shards_is_the_shipped_schedule(self):
        # Four pieces per block, two rounds, six receive semaphores beyond the
        # mate tile semaphore, as the 8-shard program had.
        self.assertEqual(transport.pieces(8), 4)
        self.assertEqual(transport.dimensions(8), 2)
        self.assertEqual(transport.receive_semaphores(8), 7)
        self.assertEqual(
            transport.sends(2, 2, 4), [(0, 2, range(0, 2)), (1, 1, range(2, 4))]
        )

    @parameterized.parameters(0, 3, 6, 12, 64)
    def test_refuses_other_widths(self, width):
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            transport.check_width(width)


if __name__ == "__main__":
    absltest.main()
