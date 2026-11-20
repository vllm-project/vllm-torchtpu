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
"""Fused EP MoE v3's token exchange (``transport``): its static schedule
simulated in plain Python, and its start program traced on one CPU device.

The exchange's rounds, the pieces each one forwards and the waits that
gate them are trace-time Python over relative addresses, so the schedule is
run here for every supported mesh width without a device: every block
reaches every representative exactly once, every forwarded piece has landed
first, and the waits on each round's semaphores add up to what lands on
them.

The start program quantizes a shard's rows a QUANT_TILE_ROWS tile at a time
and writes each tile's scale column into the routing message; a shard whose
rows are not a whole number of tiles ends in a partial tile. The program
returns DMA semaphores, which the TPU interpreter cannot do, so these tests
trace it: the shapes are what is under test.
"""

import collections
import math
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v3 import layout, transport
from vllm_torchtpu.kernels.fused_moe.v3.layout import MESH_AXIS

pytestmark = pytest.mark.cpu_test

HIDDEN, PAIRS = 256, 1024


def _trace_start(tokens, **kwargs):
    """The start program over `tokens` rows on a one-shard mesh, traced, with
    start_transport's `kwargs`."""
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), (MESH_AXIS,))
    message_rows = transport.message_rows(tokens, PAIRS)

    def start(rows, message):
        return transport.start_transport(
            rows,
            message,
            scale_base=transport.scale_row_offset(PAIRS),
            num_shards=1,
            **kwargs,
        )[3]

    program = jax.shard_map(
        start,
        mesh=mesh,
        in_specs=(P(MESH_AXIS), P(MESH_AXIS)),
        out_specs=P(MESH_AXIS),
        check_vma=False,
    )
    return jax.make_jaxpr(program)(
        jax.ShapeDtypeStruct((tokens, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((message_rows, transport.MESSAGE_LANES), jnp.int32),
    )


@pytest.mark.parametrize("tokens", [128, 136, 192])
def test_start_traces_with_a_partial_last_tile(tokens):
    """136 and 192 rows end in a partial tile of 8 and 64 rows; each tile's
    scale column is a whole tile tall and fills one message row."""
    assert transport.quant_tiles(tokens) == -(-tokens // transport.QUANT_TILE_ROWS)
    _trace_start(tokens)


@pytest.mark.parametrize("side_effects", [True, False])
@pytest.mark.parametrize("with_index", [False, True], ids=["mesh_axis", "operand"])
def test_the_start_program_takes_its_lowering_form_from_its_arguments(
    monkeypatch, side_effects, with_index
):
    """`side_effects` is what the program declares to XLA. A shard index
    handed in is read from its SMEM operand, never the mesh axis, and every
    peer address takes the operand form (the torch bridge's); without one
    the program reads the mesh axis and addresses peers in the axis form."""
    forms = []
    device_coords = transport.device_coords

    def record(*args, shard_index_operand):
        forms.append(shard_index_operand)
        return device_coords(*args, shard_index_operand=shard_index_operand)

    monkeypatch.setattr(transport, "device_coords", record)
    index = jnp.zeros((1,), jnp.int32) if with_index else None
    text = str(_trace_start(128, side_effects=side_effects, shard_index=index))
    assert text.count("has_side_effects=True") == int(side_effects)
    assert forms and set(forms) == {with_index}
    assert ("axis_index" in text) == (not with_index)


def test_a_shard_index_operand_addresses_a_mesh_of_the_expert_axis_alone():
    """The operand form names MESH_AXIS alone; a replica axis would need
    lax.axis_index, which the operand form exists to avoid."""
    assert layout.device_coords(3, shard_index_operand=True) == {MESH_AXIS: 3}
    assert layout.device_coords(3, shard_index_operand=False) == (3,)
    with pytest.raises(ValueError, match="one axis"):
        layout.device_coords(3, (MESH_AXIS, "replica"), shard_index_operand=True)


# ---- the exchange schedule, simulated ----

WIDTHS = [2, 4, 8, 16, 32]


def _schedule(width):
    return transport.dimensions(width), transport.pieces(width)


@pytest.mark.parametrize("width", WIDTHS)
def test_every_block_reaches_every_representative_once(width):
    """A representative receives each other pair's block, every piece of it,
    exactly once, in the round its distance names, and every link carries
    the same share each round."""
    n, n_pieces = _schedule(width)
    received = collections.Counter()
    for t in range(1, n + 1):
        per_link = []
        for link in range(n):
            landed = transport.arrivals(n, t, n_pieces, link)
            per_link.append(len(landed))
            for r, p in landed:
                assert bin(r).count("1") == t, (t, link, r)
                received[(r, p)] += 1
        # C(n - 1, t - 1) / t blocks a link in round t.
        assert per_link == [math.comb(n - 1, t - 1) * n_pieces // t] * n
    want = {(r, p) for r in range(1, 1 << n) for p in range(n_pieces)}
    assert set(received) == want
    assert set(received.values()) <= {1}


@pytest.mark.parametrize("width", WIDTHS)
def test_a_forwarded_piece_landed_on_the_link_landing_link_names(width):
    """Round t forwards only pieces that arrived in round t - 1, and
    landing_link names the link each one arrived on."""
    n, n_pieces = _schedule(width)
    for t in range(2, n + 1):
        for _link, r, pieces in transport.sends(n, t, n_pieces):
            for p in pieces:
                link = transport.landing_link(r, p, n_pieces)
                assert (r, p) in transport.arrivals(n, t - 1, n_pieces, link)


class _Copy:
    def __init__(self, log, rows, sem, peer):
        self.log, self.key = log, (rows, sem, peer)

    def start(self):
        self.log.append(("send", *self.key))

    def wait_send(self):
        self.log.append(("wait_send", *self.key))


class _RecordingExchange:
    """A representative's side of ``transport.Exchange`` that records what
    the forwarding and finishing steps issue, in program order."""

    mate = "mate"

    def __init__(self, width):
        self.n, self.n_pieces = _schedule(width)
        self.log = []

    def representative_only(self, body):
        body()

    def mate_only(self, body):
        pass

    def piece(self, r, p):
        return ("piece", r, p)

    def block(self, r):
        return ("block", r)

    def peer(self, link):
        return link

    def round_sem(self, t, link):
        return ("round", t, link)

    def mate_sem(self, t):
        return ("mate", t)

    def remote_copy(self, rows, sem, peer, src=None):
        return _Copy(self.log, rows, sem, peer)

    def wait_landed(self, rows, sem):
        self.log.append(("wait", rows, sem))


@pytest.mark.parametrize("width", WIDTHS)
def test_the_waits_match_what_lands_on_each_semaphore(width):
    """Run the forwarding rounds and the finish against a recorder: a piece
    is forwarded only after a wait for it on the semaphore of the round and
    link it landed on, each round semaphore is waited for exactly the pieces
    that land on it, every block goes to the mate once, and every send is
    waited for."""
    x = _RecordingExchange(width)
    transport.forward_middle_rounds(x)
    transport.finish(x)
    n, n_pieces = x.n, x.n_pieces
    waited = collections.defaultdict(list)
    for event in x.log:
        kind, rows, sem = event[:3]
        if kind == "wait":
            waited[sem].append(rows)
        elif kind == "send" and rows[0] == "piece" and sem[1] >= 2:
            _, r, p = rows
            landed_on = ("round", sem[1] - 1, transport.landing_link(r, p, n_pieces))
            assert rows in waited[landed_on], (event, landed_on)
    for t in range(1, n + 1):
        for link in range(n):
            want = [
                ("piece", r, p) for r, p in transport.arrivals(n, t, n_pieces, link)
            ]
            assert waited[("round", t, link)] == want, (t, link)
    mate_sends = [e[1] for e in x.log if e[0] == "send" and e[2][0] == "mate"]
    assert sorted(mate_sends) == sorted(
        ("block", r) for t in range(1, n + 1) for r in transport.blocks_at(n, t)
    )
    forwarded = collections.Counter(e[1:] for e in x.log if e[0] == "send")
    finished = collections.Counter(e[1:] for e in x.log if e[0] == "wait_send")
    # finish waits for round 1's sends too, which the start call issues.
    round_one = collections.Counter(
        (("piece", r, p), ("round", 1, link), link)
        for link, r, pieces in transport.sends(n, 1, n_pieces)
        for p in pieces
    )
    assert finished == forwarded + round_one


@pytest.mark.parametrize("width", WIDTHS)
def test_the_receive_semaphores_are_distinct_and_in_range(width):
    """Each round's links and each round's mate copies wait on a semaphore
    of their own, all inside receive_semaphores(width)."""
    n, _ = _schedule(width)

    class _Index:
        def __getitem__(self, i):
            return i

    exchange = transport.Exchange(
        None,
        None,
        types.SimpleNamespace(at=_Index()),
        0,
        transport.token_multiple(width),
        width,
        shard_index_operand=False,
    )
    ids = [exchange.round_sem(t, link) for t in range(1, n + 1) for link in range(n)]
    ids += [exchange.mate_sem(t) for t in range(1, n + 1)]
    assert len(set(ids)) == len(ids)
    assert transport.MATE_TILE_SEMAPHORE not in ids
    assert all(0 <= i < transport.receive_semaphores(width) for i in ids)
