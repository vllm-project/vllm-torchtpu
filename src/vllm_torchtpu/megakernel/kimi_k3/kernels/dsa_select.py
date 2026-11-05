# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1's exact DSA top-k select inside the K3 decode megakernel (DCP).

Each row's candidates are its tokens ``0 .. position`` (the new token
included), spread over the DCP group's members page by page (local page ``j``
of the member with page owner ``o`` is global page ``j group + o``). A row of
more than ``INDEX_TOPK`` tokens selects its ``INDEX_TOPK`` highest scores
over the whole group, ties to the lower global token index; every member
marks the selected tokens it holds. The scores never leave their member: a
radix select over order-preserving int32 keys exchanges only digit counts.

Layout and bounds. :meth:`TopkSelect.run` reads the scorer's FP32 buffer
``[planes, V, 128]`` (``dsa_decode.score_shape``) in place and turns it into
keys. Below eight rows a row splits into ``g`` parts of ``P = planes x 128``
local tokens (``dsa_decode.virtual_rows``); virtual row ``r g + p`` is bounded
by ``clip(L - p P, 0, P)``, ``L = dsa_decode.local_length`` of the row on
this member, and ``L = 0`` for a row of at most ``INDEX_TOPK`` tokens (its
"early" flag set, its mask empty: its fold is dense). Entries at or past a
bound become the lowest key, so the scorer's padding and unwritten entries
never count. Every plane loop runs to the call's largest bound, not the
table width.

Rounds. ``ceil(32 / bits)`` rounds, each a count of the keys' next digit
below the prefix found so far (``histogram``: one compare per key and
threshold, or four 8-bit counters per lane), the counts' sum over the group
(``exchange``: one flat all-to-all, or a host's lanes then the lane's hosts)
and the digit decision. When ``bits`` does not divide 32 the last round's
digit is the remaining low bits: its count pass still bins ``bits`` bits at
shift 0, and the decision drops the digit values past the remaining width
(those keys lie above the prefix, already in the count above it). Below
eight rows, ``row_sum="before"`` sums a row's virtual rows before the
exchange, into one tile. :data:`SELECT_FORMS` gives the form per batch class.

Stop. A row is decided once the count at or above the lower edge of its
current bin is exactly ``INDEX_TOPK``: every key in the bin is selected, so
the edge serves as ``tau`` with every tie taken, and no later round changes
the row's mark. With ``stop`` set, the rounds end after the first round
(from the one that has resolved ``STOP_BITS`` bits) after which every row is
decided; the counts are the group's sums, so every member stops at the same
round.

No barrier. The first call relies on every member having entered the kernel
(the decode megakernel's entry barrier). Every member of the group calls
:meth:`TopkSelect.run` the same number of times in the same order. A peer
sends round ``j``'s packet only after receiving this member's round ``j - 1``
packet, which this member sent after reading its round ``j - 2`` slots (the
same parity), so slots never mix. Every call starts at parity 0: after a
call of an even number of rounds by the same argument, after an odd number
(``stop="any"``) because a member enters its next call only after every
member finished this one (the decode megakernel's DCP combine between MLA
layers needs every member's partials, sent after its select; a caller with
no such exchange between two calls uses ``stop="even"``, which stops only
after an even number of rounds). The tie inbox is written only after every
member finished the previous call.

Ties. The last round leaves the threshold key ``tau``, the count above it
and the count at or above it, the same on every member. When some row's ties
at ``tau`` overflow its quota, one tie round sends each member's per-page tie
counts into slot ``owner`` (its page owner) of every peer's tie inbox, so the
sum over lower slots counts the ties on the same local page at lower global
indices under any owner table; a tie is taken when the ties ahead of it in
global token order are under the quota. Without the round every tie is taken.

The new token. It is never in a mask: its owner clears its bit after the
marking (every fold excludes local tokens at or past ``position``).
:meth:`TopkSelect.new_selected` decides its flag from its score on every
member: above ``tau``, or equal and every tie taken (ties go to the lower
global indices and the new token has the highest). A row of at most
``INDEX_TOPK`` tokens selects it; the caller decides that from the live
position, since the select's state is valid only after :meth:`TopkSelect.run`
in the same step and layer.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_TOPK
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, mla_dcp

SORT = _config().dsa_select_sort
PACK16 = _config().dsa_select_pack16
DIGIT_BITS = _config().dsa_select_digit_bits
STOP = _config().dsa_select_stop
COUNT_UNROLL = _config().dsa_select_count_unroll
PACKED_UNROLL = _config().dsa_select_packed_unroll


def sorted_rows(batch: int) -> bool:
    """Whether a call of ``batch`` rows sorts them (KernelConfig.dsa_select_sort;
    one virtual row a row, more than one 8-row block)."""
    return SORT and batch >= 16


LANES = 128
SUBLANES = 8
HOST_LANES = mla_dcp.HOST_LANES
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
BIN_CHUNK = 16  # digit values counted per pass over the keys
FIELD_PLANES = 255  # planes an 8-bit packed counter may count before widening
STATE_PLANES = 5  # early, base (tau ^ INT32_MIN), upper (count > tau), offset (tie ranks), at (count >= tau)
EARLY, BASE, UPPER, OFFSET, AT = range(STATE_PLANES)
MAX_PAGES = 128  # a tie packet's lanes: tables over this many local pages are refused
SCOPED_NAMES = ("dsa select scalars",)
BLOCK_PLANES = 2  # the select scalars' first entry of each 8-row block's plane bound (0: the call's, 1: the tie flag)
STOP_SLOT = 15  # the select scalars' entry of the stop flag (every row decided)
DECIDED_SLOT = 14  # with rounds_cap (timing only), where the decided flag goes instead
STOP_BITS = 20  # bits resolved before the first stop check: below it no row was ever decided on real text


@dataclasses.dataclass(frozen=True)
class SelectForm:
    """A select's trace-time form: ``bits`` the digit width (``ceil(32 /
    bits)`` rounds), ``exchange`` the histogram rounds' exchange (``flat`` or
    ``host``), ``histogram`` the count pass (``cumulative`` or ``packed``),
    ``row_sum`` whether below eight rows a row's virtual rows are summed
    ``before`` the exchange or ``after``, ``wire`` the counts' exchange words
    (``int32``, or ``pack16``: two 16-bit counts a lane, for a flat exchange of
    packed counts per virtual row only), ``stop`` whether the rounds end
    once every row is decided (``off``, ``any`` round, or only after an
    ``even`` number of rounds; the module docstring) and ``unroll`` the packed
    count pass's key planes a loop trip (1: today's loop; 2 or 4: the reworked
    pass, ``TopkSelect._packed_pass_unrolled``)."""

    bits: int
    exchange: str
    histogram: str
    row_sum: str
    wire: str = "int32"
    stop: str = "off"
    unroll: int = 1

    def __post_init__(self):
        if self.bits not in (4, 6, 7, 8) or self.exchange not in ("flat", "host") or (
                self.histogram not in ("cumulative", "packed")) or self.row_sum not in ("after", "before") or (
                self.wire not in ("int32", "pack16")) or self.stop not in ("off", "any", "even") or (
                self.unroll not in (1, 2, 4)) or (self.unroll > 1 and self.histogram != "packed"):
            raise ValueError(f"unknown select form {self}")
        if self.wire == "pack16" and (self.exchange, self.histogram, self.row_sum) != ("flat", "packed", "after") or (
                self.wire == "pack16" and self.bits < 6):
            raise ValueError(f"16-bit counts need a flat exchange of 6- to 8-bit packed counts per virtual row, "
                             f"not {self}")

    @property
    def rounds(self) -> int:
        return -(-32 // self.bits)

    @property
    def first_stop(self) -> int:
        """The first round after which the stop is checked: the one that has
        resolved ``STOP_BITS`` bits (its index rounded up to odd for
        ``even``)."""
        first = -(-STOP_BITS // self.bits) - 1
        return first + (first % 2 == 0) if self.stop == "even" else first


SELECT_FORMS: dict[int, SelectForm] = {
    1: SelectForm(8, "flat", "packed", "before"),
    2: SelectForm(8, "flat", "packed", "before"),
    4: SelectForm(8, "flat", "packed", "before"),
    8: SelectForm(8, "host", "packed", "after"),
    16: SelectForm(4, "flat", "cumulative", "after"),
    32: SelectForm(4, "flat", "cumulative", "after"),
    64: SelectForm(4, "flat", "cumulative", "after"),
}


def select_form(rows: int, local_tokens: int, passes: int = 1) -> SelectForm:
    """The form of a call of ``rows`` rows of at most ``local_tokens`` local
    tokens, one of ``passes`` back-to-back calls a layer:
    ``SELECT_FORMS[rows]``, or with ``KernelConfig.dsa_select_pack16`` at eight
    rows (one virtual row a row) the flat exchange of 16-bit counts while the
    local tokens, and so every member's count, stay below 2^16; at eight rows
    ``KernelConfig.dsa_select_digit_bits`` (when set) the digit width; with
    ``KernelConfig.dsa_select_stop`` the stop once every row is decided
    (``even`` when passes follow each other with no exchange between them);
    ``KernelConfig.dsa_select_packed_unroll`` packed counts at 16 and 32 rows,
    and ``KernelConfig.dsa_select_count_unroll`` the unroll of the forms
    packed today (eight rows and below)."""
    stop = "off" if not STOP else "any" if passes == 1 else "even"
    bits = DIGIT_BITS or 8
    if PACK16 and rows == 8 and local_tokens < 2**16:
        return SelectForm(bits, "flat", "packed", "after", "pack16", stop, COUNT_UNROLL)
    if rows == 8:
        return dataclasses.replace(SELECT_FORMS[rows], bits=bits, stop=stop, unroll=COUNT_UNROLL)
    if PACKED_UNROLL and rows in (16, 32):
        return dataclasses.replace(SELECT_FORMS[rows], histogram="packed", stop=stop, unroll=PACKED_UNROLL)
    unroll = COUNT_UNROLL if SELECT_FORMS[rows].histogram == "packed" else 1
    return dataclasses.replace(SELECT_FORMS[rows], stop=stop, unroll=unroll)


def tie_pages(local_tokens: int, page_tokens: int) -> int:
    """The tie packet's lanes per virtual row: the next power of two at or
    above the local pages ``ceil(local_tokens / page_tokens)``; a ValueError
    over ``MAX_PAGES`` pages."""
    pages = -(-local_tokens // page_tokens)
    if pages > MAX_PAGES:
        raise ValueError(f"select: {pages} local pages of {page_tokens} tokens exceed the tie packet's "
                         f"{MAX_PAGES} lanes")
    return 1 << max(pages - 1, 0).bit_length()


def scratch_shapes(batch: int, local_tokens: int, form: SelectForm,
                   words: bool = False) -> dict[str, tuple[tuple[int, ...], Any]]:
    """The select's VMEM views by name, ``(shape, dtype)``: ``mask`` (byte
    ``i`` of word ``w`` is plane ``4 w + i``'s 0/1), or with ``words`` (one
    virtual row a row) ``words`` (lane-major: bit ``s % 32`` of lane ``l`` of
    tile ``s // 32``, ``[s // 32, v, l]``, is plane ``s``'s lane ``l``, local
    token ``128 s + l``), ``summary`` (bit ``w % 32`` of lane ``w // 32``
    whether word ``w`` = ``[w // 128, v, w % 128]`` is nonzero) and ``count``
    (the row's selected local tokens, lane-broadcast); then ``bounds`` (each
    virtual row's local tokens), ``state`` (``STATE_PLANES``) and
    ``histogram`` (the rounds' digit counts)."""
    planes, vrows, _ = dsa_decode.score_shape(local_tokens, batch)
    shapes: dict[str, tuple[tuple[int, ...], Any]] = {}
    if words:
        tiles = -(-planes // 32)
        if tiles > LANES // 4:
            raise ValueError(f"select: {planes} planes' words exceed the summary's {LANES * 32} words")
        shapes.update(words=((tiles, vrows, LANES), jnp.int32), summary=((1, vrows, LANES), jnp.int32),
                      count=((vrows, LANES), jnp.int32))
    else:
        shapes["mask"] = ((-(-planes // 4), vrows, LANES), jnp.int32)
    shapes.update(bounds=((vrows, LANES), jnp.int32), state=((STATE_PLANES, vrows, LANES), jnp.int32),
                  histogram=((vrows, max(2**form.bits, LANES)), jnp.int32))
    return shapes


def scoped_scratch() -> tuple:
    """The select's scoped SMEM (``SCOPED_NAMES``): the call's largest plane
    bound, the tie round's predicate and each 8-row block's largest plane
    bound (from ``BLOCK_PLANES``, at most eight blocks)."""
    return (pltpu.SMEM((16,), jnp.int32),)


def _packet_shape(vrows: int, width: int) -> tuple[int, int]:
    """The ``[rows, lanes]`` packet of ``[vrows, width]`` counts. Below 128
    lanes, row block ``f`` of ``rows`` rows takes lanes ``[f width, (f + 1)
    width)``, so packets stay dense."""
    if width >= LANES:
        return vrows, width
    return max(8, vrows * width // LANES), LANES


def _group_sum(x, groups: int):
    """Sum over each row's ``groups`` virtual rows (aligned runs of sublanes of
    an 8-sublane tile), on each of them."""
    sub = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    step = 1
    while step < groups:
        mate = jnp.where((sub & step) != 0, pltpu.roll(x, step, 0), pltpu.roll(x, 8 - step, 0))
        x, step = x + mate, 2 * step
    return x


def _group_exclusive(x, groups: int):
    """Sum over each row's earlier virtual rows (:func:`_group_sum`'s runs)."""
    part_of = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0) % groups
    out = jnp.zeros_like(x)
    for offset in range(1, groups):
        out = out + jnp.where(part_of >= offset, pltpu.roll(x, offset, 0), 0)
    return out


def _select_stages(exchange: str, group: int) -> list[tuple[int, int]]:
    """An exchange's stages over a group of ``group`` ranks, ``(slots,
    stride)`` each: ``slots`` ranks ``me ^ (offset * stride)`` swap packets
    (``flat``: the whole group; ``host``: a host's lanes, then the lane's
    hosts, the second stage omitted within one host); none for any other
    name (the microbenchmark's timing-only forms and its merge)."""
    if group & (group - 1):
        raise ValueError(f"select: peers are rank ^ offset, so the group must be a power of two, not {group}")
    if exchange == "flat":
        return [(group, 1)]
    if exchange == "host":
        if group % HOST_LANES:
            raise ValueError(f"select: the host exchange needs whole hosts of {HOST_LANES} ranks, not {group}")
        return [(HOST_LANES, 1)] + ([(group // HOST_LANES, HOST_LANES)] if group > HOST_LANES else [])
    return []


def _inbox_exchange(me, inbox, sems, packet, parity, slots: int, stride: int, slot, base=None,
                    wait_sent: bool = True):
    """One exchange stage: ``packet`` goes into ``slot`` (returned) of this
    rank's persistent ``inbox [parity, slots, ...]`` and, by one remote DMA
    each, of its ``slots - 1`` peers', mesh ranks ``base + (me ^ (offset *
    stride))`` (``base`` None: the group starts at mesh rank 0); then one
    byte-counted wait for the peers' packets in and one for this rank's out
    (``sems [2, parity]``: the sends' and the receives' semaphores, the
    remote copies' ``send_sem`` and ``recv_sem``). ``packet`` None: it is in
    the slot already. ``wait_sent`` False leaves the wait for this rank's
    sends to the caller (:func:`_wait_sent`), before it next writes the slot
    or ends."""
    if packet is not None:
        inbox[parity, slot] = packet
    for offset in range(1, slots):
        peer = me ^ (offset * stride)
        pltpu.make_async_remote_copy(
            inbox.at[parity, slot],
            inbox.at[parity, slot],
            sems.at[0, parity],
            sems.at[1, parity],
            device_id=(peer if base is None else base + peer,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    peers = inbox.at[parity, pl.ds(0, slots - 1)]
    pltpu.make_async_copy(peers, peers, sems.at[1, parity]).wait()
    if wait_sent:
        pltpu.make_async_copy(peers, peers, sems.at[0, parity]).wait()
    return slot


def _wait_sent(stages, stage_wires, parity):
    """Wait for this rank's sends of the exchange with ``parity`` through
    ``stages`` (one byte-counted wait each), left by
    :func:`_exchange_sum` with ``wait_sent`` False."""
    for (slots, _), (inbox, sems) in zip(stages, stage_wires):
        peers = inbox.at[parity, pl.ds(0, slots - 1)]
        pltpu.make_async_copy(peers, peers, sems.at[0, parity]).wait()


def _exchange_sum(me, stages, stage_wires, packet, parity, lower_too: bool, slot, base=None,
                  wait_sent: bool = True, unpack=None):
    """``(sum over ranks, sum over lower slots)`` of ``packet`` through
    ``stages`` (:func:`_select_stages`) and their ``(inbox, sems)`` wires, one
    :func:`_inbox_exchange` into slot ``slot(slots, stride)`` and a sum of the
    slots each; without stages ``(packet, 0)``. No barrier between rounds (the
    module docstring). ``unpack`` (one stage only): each member's packet is
    summed as ``unpack(packet)`` (the sum only)."""
    if unpack is not None:
        if len(stage_wires) != 1 or lower_too:
            raise ValueError("select: unpacked sums take one exchange stage and no lower sums")
        (slots, stride), (inbox, sems) = stages[0], stage_wires[0]
        _inbox_exchange(me, inbox, sems, packet, parity, slots, stride, slot(slots, stride), base, wait_sent)
        zero = jnp.zeros_like(unpack(packet))
        return jax.lax.fori_loop(0, slots, lambda i, total: total + unpack(inbox[parity, i]), zero), zero
    if not stage_wires:
        return packet, jnp.zeros_like(packet)
    lower = jnp.zeros_like(packet)
    for (slots, stride), (inbox, sems) in zip(stages, stage_wires):
        mine = _inbox_exchange(me, inbox, sems, packet, parity, slots, stride, slot(slots, stride), base,
                               wait_sent)

        def add(i, carry, inbox=inbox, mine=mine):
            total, below = carry
            got = inbox[parity, i]
            if lower_too:
                below = below + got * (i < mine).astype(jnp.int32)
            return total + got, below

        packet, below = jax.lax.fori_loop(0, slots, add, (jnp.zeros_like(packet), jnp.zeros_like(packet)))
        lower = lower + below
    return packet, lower


def _row_pieces(rows: int, bins: int):
    """``row_sum="before"``'s packet layout: per row ``r`` and 128-lane chunk
    ``c`` of its ``bins`` counts, ``(r, c, sublane, lane offset, width)`` of
    the chunk's place in the ``[8, 128]`` packet (flat position ``r bins +
    d``)."""
    if rows * bins > SUBLANES * LANES:
        raise ValueError(f"select: {rows} rows of {bins} counts exceed one packet")
    pieces = []
    for r in range(rows):
        for c in range(max(1, bins // LANES)):
            start = r * bins + c * LANES
            pieces.append((r, c, start // LANES, start % LANES, min(bins, LANES)))
    return pieces


def _pack16_shape(vrows: int, half: int) -> tuple[int, int]:
    """The 16-bit counts' packet of ``[vrows, half]`` words: as is from 128
    lanes, else row blocks side by side in 128 lanes (``[vrows half / 128,
    128]``, under one tile: 1 KiB at eight rows of 6-bit digits)."""
    return (vrows, half) if half >= LANES else (vrows * half // LANES, LANES)


def _histogram_packet(vrows: int, parts: int, form: SelectForm) -> tuple[int, int]:
    if form.wire == "pack16":
        return _pack16_shape(vrows, 2**form.bits // 2)
    if form.row_sum == "before" and parts > 1:
        return SUBLANES, LANES
    return _packet_shape(vrows, 2**form.bits)


def persistent_scratch(batch: int, local_tokens: int, page_tokens: int, group: int, form: SelectForm,
                       order_rows: int = 0) -> tuple:
    """The select's persistent VMEM, which peers write (:func:`persistent_names`):
    per histogram exchange stage an inbox ``[parity, slots, packet]`` and its
    received / sent DMA semaphores, then the tie inbox ``[1, group, packet]``
    and its semaphores."""
    vrows, parts = dsa_decode.virtual_rows(batch)
    packet = _histogram_packet(vrows, parts, form)
    wires = []
    for slots, _ in _select_stages(form.exchange, group):
        wires += [pltpu.VMEM((2, slots, *packet), jnp.int32), pltpu.SemaphoreType.DMA((2, 2))]
    if order_rows and sorted_rows(order_rows):
        wires += [pltpu.SMEM((3, order_rows), jnp.int32)]  # sort_rows's order
    tie_packet = _packet_shape(vrows, tie_pages(local_tokens, page_tokens))
    return (*wires, pltpu.VMEM((1, group, *tie_packet), jnp.int32), pltpu.SemaphoreType.DMA((2, 1)))


def persistent_names(form: SelectForm, group: int = mla_dcp.GROUP, order_rows: int = 0) -> tuple[str, ...]:
    """:func:`persistent_scratch`'s entries by name."""
    count = len(_select_stages(form.exchange, group))
    stages = {"flat": ("histogram",), "host": ("lane", "host")}[form.exchange][:count]
    names = []
    for stage in stages:
        names += [f"dsa select {stage} inbox", f"dsa select {stage} semaphores"]
    if order_rows and sorted_rows(order_rows):
        names += ["dsa select order"]
    return (*names, "dsa select tie inbox", "dsa select tie semaphores")


def sort_rows(order, position_ref, batch: int, sequence_rows: int):
    """Into SMEM ``order [3, batch]``: ``order[0, row]`` its slot (virtual
    row), ``order[1, slot]`` its row and ``order[2, slot]`` its position.
    Sequences (``sequence_rows`` consecutive rows) go in descending order of
    their first row's position, ties by row, each sequence's rows in order:
    the same on every member. Once a step, before the first MLA layer."""
    sequences = batch // sequence_rows

    @pl.loop(0, sequences)
    def rank(s):
        key = position_ref[s * sequence_rows]

        def ahead(t, count):
            other = position_ref[t * sequence_rows]
            return count + ((other > key) | ((other == key) & (t < s))).astype(jnp.int32)

        place = jax.lax.fori_loop(0, sequences, ahead, jnp.int32(0))
        for w in range(sequence_rows):
            row, slot = s * sequence_rows + w, place * sequence_rows + w
            order[0, row] = slot
            order[1, slot] = row
            order[2, slot] = position_ref[row]


def _key(raw):
    """The order-preserving int32 key of FP32 bits ``raw`` (signed order)."""
    return raw ^ (jax.lax.shift_right_arithmetic(raw, 31) & 0x7FFFFFFF)


def _score_key(score):
    """The select's int32 key of FP32 ``score``."""
    return _key(pltpu.bitcast(score, jnp.int32))


def _new_chosen(score, base, early, at):
    """Int32 ``[rows, 1]`` 0/1: whether each row's new token (FP32 ``score``)
    is selected, from its state planes ``BASE``, ``EARLY`` and ``AT``."""
    tau = base ^ INT32_MIN
    key = _score_key(score)
    chosen = (early != 0) | (key > tau) | ((key == tau) & (at == INDEX_TOPK))
    return jnp.max(chosen.astype(jnp.int32), axis=1, keepdims=True)


def _row_sum(x):
    return jnp.sum(x, axis=1, keepdims=True)


def _strict_lanes():
    """``[128, 128]`` BF16 ``i < j``: a 0/1 row times it counts the ones before
    each lane."""
    return (jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 0)
            < jax.lax.broadcasted_iota(jnp.int32, (LANES, LANES), 1)).astype(jnp.bfloat16)


def _lane_exclusive(x):
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    total, step = x, 1
    while step < LANES:
        total = total + jnp.where(lane >= step, pltpu.roll(total, step, 1), 0)
        step *= 2
    return total - x


def _lane_suffix(x):
    lane = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    width, step = x.shape[1], 1
    while step < width:
        later = pltpu.roll(x, width - step, 1)
        x = x + jnp.where(lane + step < width, later, 0)
        step *= 2
    return x


def _roll(x, shift, axis):
    shift %= x.shape[axis]
    return pltpu.roll(x, shift, axis) if shift else x


class _Keys:
    """The int32 keys held in the scores' FP32 ref, by value bitcasts (Pallas
    interpret mode does not read through a ref bitcast)."""

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, index):
        return pltpu.bitcast(self.ref[index], jnp.int32)

    def __setitem__(self, index, value):
        self.ref[index] = pltpu.bitcast(value, jnp.float32)


@dataclasses.dataclass(frozen=True)
class SelectScratch:
    """One MLA layer's select buffers: the :func:`scratch_shapes` views, the
    :func:`scoped_scratch` SMEM and the :func:`persistent_scratch` refs
    (``wires``, in its order)."""

    mask: Any
    bounds: Any
    state: Any
    histogram: Any
    smem: Any
    wires: tuple
    order: Any = None  # sort_rows's order (persistent SMEM) when the call's rows are sorted
    words: Any = None  # with bit words: the words, summary and count views (mask None)
    summary: Any = None
    count: Any = None


class TopkSelect:
    """The select of one MLA layer on this member, for ``batch`` rows of at
    most ``local_tokens`` local tokens at ``position_ref`` (SMEM).

    The member holds the global pages ``g % group == owner``; it is mesh rank
    ``base + member`` of a group of ``group`` consecutive ranks. ``form``
    (:class:`SelectForm`) is trace-time. ``skip``: the ``KernelConfig.dcp_skip``
    names; with ``index_select`` among them :meth:`run` is compiled, never run
    (timing only). ``scope(name)``: the context ``run`` runs in (default
    none). ``rounds_cap`` (timing only, a ``stop`` form): exactly this many
    rounds, the stop check computed but not obeyed, then every row taken as
    decided (no tie round, as after a real stop). The caller runs :meth:`run`
    only in steps that score (``dsa_decode.scores_needed``), on every member
    alike."""

    def __init__(self, scratch: SelectScratch, *, batch: int, local_tokens: int, position_ref, owner, member, base,
                 group: int, page_tokens: int, form: SelectForm, skip: tuple[str, ...] = (), scope: Any = None,
                 rounds_cap: int = 0):
        self.s, self.batch, self.position_ref, self.form = scratch, batch, position_ref, form
        self.owner, self.member, self.base, self.group, self.page_tokens = owner, member, base, group, page_tokens
        self.vrows, self.parts = dsa_decode.virtual_rows(batch)
        self.planes = dsa_decode.score_shape(local_tokens, batch)[0]
        self.part_tokens = self.planes * LANES
        self.words = -(-self.planes // 4)
        self.blocks = self.vrows // SUBLANES
        if self.blocks > DECIDED_SLOT - BLOCK_PLANES:
            raise ValueError(f"select: {self.vrows} virtual rows exceed the scalars' block bounds")
        self.bins = 2**form.bits
        self.bin_lanes = max(self.bins, LANES)
        # Digit values per pass over the keys (packed: four per counter).
        self.chunk = min(self.bins, BIN_CHUNK * (4 if form.histogram == "packed" else 1))
        lane_rows = page_tokens // LANES
        if page_tokens % LANES or lane_rows & (lane_rows - 1):
            raise ValueError(f"select: {page_tokens}-token pages must be a power-of-two number of lane rows")
        self.page_shift = lane_rows.bit_length() - 1
        self.tie_width = tie_pages(local_tokens, page_tokens)
        self.stages = _select_stages(form.exchange, group)
        if scratch.wires:
            # Each stage's inbox, then the tie inbox, each followed by its semaphores; a sorted call's order is no inbox.
            wires = [wire for wire in scratch.wires if wire is not scratch.order]
            held = [inbox.shape[1] for inbox in wires[::2]]
            wanted = [slots for slots, _ in self.stages] + [group]
            if len(wires) != 2 * len(wanted) or held != wanted:
                raise ValueError(f"select: inboxes of {held} members' slots, not {wanted} for a group of {group}")
        # At eight rows and more a row is one virtual row: there is nothing to sum.
        self.sum_first = form.row_sum == "before" and self.parts > 1
        self.pieces = _row_pieces(self.vrows // self.parts, self.bins) if self.sum_first else []
        self.hist_packet = _histogram_packet(self.vrows, self.parts, form)
        self.pack16 = form.wire == "pack16"
        if self.pack16 and self.part_tokens >= 2**16:
            raise ValueError(f"select: 16-bit counts need fewer than 2^16 local tokens a virtual row, not "
                             f"{self.part_tokens} (select_form)")
        self.tie_packet = _packet_shape(self.vrows, self.tie_width)
        self.scope = scope or (lambda name: contextlib.nullcontext())
        self.runs = position_ref[0] < 0 if "index_select" in skip else None
        if rounds_cap and (form.stop == "off" or not form.first_stop < rounds_cap <= form.rounds):
            raise ValueError(f"select: a rounds cap of {rounds_cap} needs a stop form whose checks reach it, not {form}")
        self.rounds_cap = rounds_cap
        self.word_mode = scratch.words is not None
        if self.word_mode and self.parts != 1:
            raise ValueError("select: bit words need one virtual row a row")

    def run(self, scores) -> None:
        """Select every row from ``scores`` (the scorer's FP32 ``[planes, V,
        128]`` ref, int32 keys after) into the mask words and the state."""
        if scores.shape != (self.planes, self.vrows, LANES):
            raise ValueError(f"select: scores {scores.shape}, expected {(self.planes, self.vrows, LANES)}")
        with self.scope("index_select"):
            if self.runs is None:
                self._run(scores)
            else:
                pl.when(self.runs)(lambda: self._run(scores))

    def _state(self, plane):
        return self.s.state.at[plane]

    def _bounds(self):
        """Each virtual row's bound and early flag (a rolled loop over rows,
        one aligned tile store each) and the call's largest plane bound."""
        s, g, part = self.s, self.parts, self.part_tokens
        s.bounds[...] = jnp.zeros(s.bounds.shape, jnp.int32)
        self._state(EARLY)[...] = jnp.ones(s.bounds.shape, jnp.int32)  # padding rows never need ties
        s.smem[0] = 0
        for b in range(self.blocks):
            s.smem[BLOCK_PLANES + b] = 0
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)

        @pl.loop(0, self.batch)
        def row(r):
            position = self.position_ref[r]
            sparse = position + 1 > INDEX_TOPK
            length = jnp.where(sparse, dsa_decode.local_length(position, self.owner, self.page_tokens, self.group), 0)
            first = r * g
            tile = pl.ds(pl.multiple_of(first // SUBLANES * SUBLANES, SUBLANES), SUBLANES)
            p = sub - first % SUBLANES
            mine = (p >= 0) & (p < g)
            s.bounds[tile, :] = jnp.where(mine, jnp.clip(length - p * part, 0, part), s.bounds[tile, :])
            early = self._state(EARLY)
            early[tile, :] = jnp.where(mine, jnp.logical_not(sparse).astype(jnp.int32), early[tile, :])
            planes = (jnp.minimum(length, part) + LANES - 1) // LANES
            s.smem[0] = jnp.maximum(s.smem[0], planes)
            s.smem[BLOCK_PLANES + first // SUBLANES] = jnp.maximum(s.smem[BLOCK_PLANES + first // SUBLANES], planes)

    def _pack(self, x, width, shape):
        if width >= LANES:
            return x
        block = shape[0]
        packet = x[:block]
        for f in range(1, self.vrows // block):
            packet = packet + pltpu.roll(x[f * block:(f + 1) * block], f * width, 1)
        return packet

    def _unpack(self, packet, width, shape):
        if width >= LANES:
            return packet
        block = shape[0]
        keep = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1) < width
        parts = [jnp.where(keep, pltpu.roll(packet, (LANES - f * width) % LANES, 1), 0)
                 for f in range(self.vrows // block)]
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)

    def _to_row_packet(self, local):
        """Each row's counts, summed on all its virtual rows of ``local [8,
        bin_lanes]``, from its first one into its place in the packet."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
        packet = jnp.zeros((SUBLANES, LANES), jnp.int32)
        for r, c, s, offset, width in self.pieces:
            piece = local[:, c * LANES:(c + 1) * LANES]
            if width < LANES:
                piece = jnp.where(lane < width, piece, 0)
            piece = _roll(_roll(piece, s - r * self.parts, 0), offset, 1)
            packet = packet + jnp.where(sub == s, piece, 0)
        return packet

    def _from_row_packet(self, packet):
        """The summed packet's counts of each row, on all its virtual rows."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)
        chunks = [jnp.zeros((SUBLANES, LANES), jnp.int32)] * (self.bin_lanes // LANES)
        for r, c, s, offset, width in self.pieces:
            piece = _roll(_roll(packet, r * self.parts - s, 0), -offset, 1)
            if width < LANES:
                piece = jnp.where(lane < width, piece, 0)
            chunks[c] = jnp.where(sub == r * self.parts, piece, chunks[c])
        # Each row's counts sit on its first virtual row: the group sum copies them.
        chunks = [_group_sum(x, self.parts) for x in chunks]
        return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=1)

    def _block_planes(self, b):
        """8-row block ``b``'s largest plane bound (traced ``b``)."""
        return self.s.smem[BLOCK_PLANES + b]

    def _plane_loop(self, lo, hi, step, carry):
        """``step`` over key planes ``lo .. hi`` (traced): ``form.unroll``
        planes a trip, then a tail of the remaining ones a trip."""
        unroll = self.form.unroll
        trips = jnp.maximum(hi - lo, 0) // unroll  # a short block's later span has lo > hi: no planes

        def trip(i, carry):
            first = lo + unroll * i
            for u in range(unroll):
                carry = step(first + u, carry)
            return carry

        carry = jax.lax.fori_loop(0, trips, trip, carry)
        return jax.lax.fori_loop(lo + unroll * trips, hi, step, carry)

    def _count_pass(self, keys, shift):
        """Counts at or above each of the round's digit thresholds, per
        virtual row, over its 8-row block's planes of ``keys``."""
        hist, base_ref, chunk = self.s.histogram, self._state(BASE), self.chunk
        bin_lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, self.bin_lanes), 1)

        def per_chunk(c, _):
            def per_block(b, _):
                block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
                base = base_ref[block, :]
                thresholds = [(base | ((c * chunk + d) << shift)) ^ INT32_MIN for d in range(chunk)]

                def step(s, acc):
                    x = keys[s, block, :]
                    return tuple(a + (x >= t).astype(jnp.int32) for a, t in zip(acc, thresholds))

                zero = jnp.zeros((SUBLANES, LANES), jnp.int32)
                acc = jax.lax.fori_loop(0, self._block_planes(b), step, (zero,) * chunk)
                h = hist[block, :]
                for d in range(chunk):
                    h = jnp.where(bin_lane == c * chunk + d, _row_sum(acc[d]), h)
                hist[block, :] = h

            jax.lax.fori_loop(0, self.vrows // SUBLANES, per_block, None)

        jax.lax.fori_loop(0, self.bins // chunk, per_chunk, None)

    def _packed_pass(self, keys, shift):
        """Counts of each digit value (not cumulative) among the keys of the
        round's prefix, as four 8-bit counters per int32 lane, one
        compare-and-add per key and four digit values."""
        if self.form.unroll > 1:
            return self._packed_pass_unrolled(keys, shift)
        hist, base_ref, chunk = self.s.histogram, self._state(BASE), self.chunk
        bin_lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, self.bin_lanes), 1)
        spans = -(-self.planes // FIELD_PLANES)
        span = -(-self.planes // spans)

        def per_chunk(c, _):
            def per_block(b, _):
                block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
                base = base_ref[block, :] ^ INT32_MIN
                planes = self._block_planes(b)
                first = c * chunk // 4
                targets = [first + g for g in range(chunk // 4)]
                mine = (bin_lane >= c * chunk) & (bin_lane < (c + 1) * chunk)

                def per_span(p, h):
                    def step(s, acc):
                        # Keys outside the prefix give digits >= 2^b (or < 0).
                        digit = jax.lax.shift_right_logical(keys[s, block, :] - base, shift)
                        group = jax.lax.shift_right_logical(digit, 2)
                        one = jnp.left_shift(1, (digit & 3) << 3)
                        return tuple(a + jnp.where(group == t, one, 0) for a, t in zip(acc, targets))

                    zero = jnp.zeros((SUBLANES, LANES), jnp.int32)
                    acc = jax.lax.fori_loop(p * span, jnp.minimum((p + 1) * span, planes), step,
                                            (zero,) * (chunk // 4))
                    for g in range(chunk // 4):
                        for f in range(4):
                            field = jax.lax.shift_right_logical(acc[g], 8 * f) & 255
                            h = jnp.where(bin_lane == c * chunk + 4 * g + f, h + _row_sum(field), h)
                    return h

                h = jnp.where(mine, 0, hist[block, :])
                hist[block, :] = jax.lax.fori_loop(0, spans, per_span, h)

            jax.lax.fori_loop(0, self.vrows // SUBLANES, per_block, None)

        jax.lax.fori_loop(0, self.bins // chunk, per_chunk, None)

    def _packed_pass_unrolled(self, keys, shift):
        """:meth:`_packed_pass` with ``form.unroll`` planes a trip, its targets
        and shift as vectors made once a block (no scalar broadcast a plane),
        and each counter word's four 8-bit fields folded into two 16-bit lane
        pairs before the row sums (a span's field is at most 255 a lane, so a
        pair's 128-lane sum is at most 32640 in each half: exact)."""
        hist, base_ref, chunk = self.s.histogram, self._state(BASE), self.chunk
        bin_lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, self.bin_lanes), 1)
        spans = -(-self.planes // FIELD_PLANES)
        span = -(-self.planes // spans)
        tile = (SUBLANES, LANES)

        def per_chunk(c, _):
            def per_block(b, _):
                block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
                base = base_ref[block, :] ^ INT32_MIN
                planes = self._block_planes(b)
                first = c * chunk // 4
                shifts = jnp.full(tile, shift, jnp.int32)
                targets = [jnp.full(tile, first + g, jnp.int32) for g in range(chunk // 4)]
                mine = (bin_lane >= c * chunk) & (bin_lane < (c + 1) * chunk)

                def per_span(p, h):
                    def step(s, acc):
                        # Keys outside the prefix give digits >= 2^b (or < 0).
                        digit = jax.lax.shift_right_logical(keys[s, block, :] - base, shifts)
                        group = jax.lax.shift_right_logical(digit, 2)
                        one = jnp.left_shift(1, (digit & 3) << 3)
                        return tuple(a + jnp.where(group == t, one, 0) for a, t in zip(acc, targets))

                    zero = jnp.zeros(tile, jnp.int32)
                    acc = self._plane_loop(p * span, jnp.minimum((p + 1) * span, planes), step,
                                           (zero,) * (chunk // 4))
                    for g in range(chunk // 4):
                        even = _row_sum(acc[g] & 0x00FF00FF)
                        odd = _row_sum(jax.lax.shift_right_logical(acc[g], 8) & 0x00FF00FF)
                        fields = (even & 0xFFFF, odd & 0xFFFF, jax.lax.shift_right_logical(even, 16),
                                  jax.lax.shift_right_logical(odd, 16))
                        for f in range(4):
                            h = jnp.where(bin_lane == c * chunk + 4 * g + f, h + fields[f], h)
                    return h

                h = jnp.where(mine, 0, hist[block, :])
                hist[block, :] = jax.lax.fori_loop(0, spans, per_span, h)

            jax.lax.fori_loop(0, self.vrows // SUBLANES, per_block, None)

        jax.lax.fori_loop(0, self.bins // chunk, per_chunk, None)

    def _round_wires(self):
        return [(self.s.wires[2 * i], self.s.wires[2 * i + 1]) for i in range(len(self.stages))]

    def _round(self, j, keys):
        """Round ``j``: count, exchange (parity ``j % 2``) and decide the digit."""
        bits, g = self.form.bits, self.parts
        base_ref, upper_ref, at_ref = self._state(BASE), self._state(UPPER), self._state(AT)
        shift = 32 - bits * (j + 1)
        # Digit values the round may choose: all 2^bits, except a last round narrower than bits.
        limit = None
        if 32 % bits:
            shift = jnp.maximum(shift, 0)
            limit = jnp.left_shift(1, 32 - bits * j - shift)
        if self.form.histogram == "packed":
            self._packed_pass(keys, shift)
        else:
            self._count_pass(keys, shift)
        hist = self.s.histogram[...]
        unpack = None
        if self.pack16 and self.bins > LANES:
            # Every member's count is below 2^16: digit values d and d + 128 share lane d.
            half = self.bins // 2
            packet = hist[:, :half] | jnp.left_shift(hist[:, half:self.bins], 16)

            def unpack(words):
                return jnp.concatenate([words & 0xFFFF, jax.lax.shift_right_logical(words, 16)], axis=1)
        elif self.pack16:
            # Digit values d and d + half share lane d of a row's half words, row blocks side by side.
            half = self.bins // 2
            lane = jax.lax.broadcasted_iota(jnp.int32, hist.shape, 1)
            words = jnp.where(lane < half, hist | jnp.left_shift(pltpu.roll(hist, LANES - half, 1), 16), 0)
            packet = self._pack(words, half, self.hist_packet)

            def unpack_rows(packet):
                words = self._unpack(packet, half, self.hist_packet)
                lane = jax.lax.broadcasted_iota(jnp.int32, words.shape, 1)
                high = pltpu.roll(jax.lax.shift_right_logical(words, 16), half, 1)
                return jnp.where(lane < half, words & 0xFFFF, high)

            unpack = unpack_rows
        else:
            packet = self._to_row_packet(_group_sum(hist, g)) if self.sum_first else self._pack(
                hist, self.bins, self.hist_packet)
        wires = self._round_wires()
        member = self.member
        # This parity's slot was last sent from two rounds ago: those sends must be done before it is rewritten.
        pl.when(j >= 2)(lambda: _wait_sent(self.stages, wires, j % 2))
        total, _ = _exchange_sum(member, self.stages, wires, packet, j % 2, False,
                                 lambda slots, stride: (member // stride) % slots, self.base, wait_sent=False,
                                 unpack=unpack)
        if self.sum_first:
            counts = self._from_row_packet(total)
        elif self.pack16:
            counts = total
        else:
            counts = self._unpack(total, self.bins, self.hist_packet)
            if g > 1:
                counts = _group_sum(counts, g)
        in_bins = jax.lax.broadcasted_iota(jnp.int32, counts.shape, 1) < (self.bins if limit is None else limit)
        if self.form.histogram == "packed":
            # Cumulative from the top, plus the keys above the prefix (a narrow last round's dropped digit values
            # hold keys above the prefix: they are in it already).
            if limit is not None:
                counts = jnp.where(in_bins, counts, 0)
            counts = jnp.where(in_bins, _lane_suffix(counts) + upper_ref[...][:, :1], 0)
        shape = (self.vrows, LANES)
        reach = _row_sum(jnp.where(in_bins & (counts >= INDEX_TOPK), 1, 0))
        below = jnp.max(jnp.where(in_bins & (counts < INDEX_TOPK), counts, -1), axis=1, keepdims=True)
        below = jnp.broadcast_to(below, shape)
        upper_ref[...] = jnp.where(below >= 0, below, upper_ref[...])
        base_ref[...] = base_ref[...] | ((reach - 1) << shift)
        # The chosen digit's count: after the last round, count(>= tau).
        at = jnp.min(jnp.where(in_bins & (counts >= INDEX_TOPK), counts, INT32_MAX), axis=1, keepdims=True)
        at_ref[...] = jnp.broadcast_to(at, shape)

    def _page_of(self, s, shape=None):
        part_of = jax.lax.broadcasted_iota(jnp.int32, shape or (self.vrows, LANES), 0) % self.parts
        return jax.lax.shift_right_logical(part_of * self.planes + s, self.page_shift)

    def _tie_round(self, keys, planes):
        """Each tie's rank offset in global token order, per local page: the
        ties of the other members on earlier local pages and of the lower page
        owners on the same one (a flat exchange into slot ``owner``)."""
        shape = (self.vrows, LANES)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        tau = self._state(BASE)[...] ^ INT32_MIN
        bound = self.s.bounds[...]

        def page_ties(s, counts):
            tie = ((keys[s] == tau) & (s * LANES + lane < bound)).astype(jnp.int32)
            return counts + jnp.where(lane == self._page_of(s), _row_sum(tie), 0)

        own = jax.lax.fori_loop(0, planes, page_ties, jnp.zeros(shape, jnp.int32))
        owner = self.owner
        total, lower = _exchange_sum(self.member, [(self.group, 1)], [self.s.wires[-2:]],
                                     self._pack(own, self.tie_width, self.tie_packet), 0, True,
                                     lambda slots, stride: owner, self.base)
        total, lower = (self._unpack(x, self.tie_width, self.tie_packet) for x in (total, lower))
        if self.parts > 1:
            total, lower, own = (_group_sum(x, self.parts) for x in (total, lower, own))
        self._state(OFFSET)[...] = _lane_exclusive(total - own) + lower

    def _mark(self, keys, with_ties):
        """The mask words below each 8-row block's largest bound: keys above
        ``tau``, and ties at ``tau`` (``with_ties``: those whose rank in global
        token order is under the quota); with bit words :meth:`_mark_words`."""
        if self.word_mode:
            return self._mark_words(keys, with_ties)
        shape = (SUBLANES, LANES)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        strict = _strict_lanes()

        def per_block(b, _):
            block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
            planes = self._block_planes(b)
            tau = self._state(BASE)[block, :] ^ INT32_MIN
            quota = INDEX_TOPK - self._state(UPPER)[block, :]
            bound = self.s.bounds[block, :]
            before = jnp.zeros(shape, jnp.int32)
            if with_ties and self.parts > 1:
                row_ties = jax.lax.fori_loop(
                    0, planes, lambda s, acc: acc + _row_sum(
                        ((keys[s, block, :] == tau) & (s * LANES + lane < bound)).astype(jnp.int32)), before)
                before = _group_exclusive(row_ties, self.parts)

            def word(w, before):
                out = jnp.zeros(shape, jnp.int32)
                for i in range(4):
                    s = 4 * w + i
                    x = keys[jnp.minimum(s, self.planes - 1) if 4 * self.words > self.planes else s, block, :]
                    valid = s * LANES + lane < bound
                    if with_ties:
                        tie = (x == tau) & valid
                        count = tie.astype(jnp.int32)
                        inner = jnp.dot(count.astype(jnp.bfloat16), strict,
                                        preferred_element_type=jnp.float32).astype(jnp.int32)
                        offset = self._state(OFFSET)[block, :]
                        ahead = _row_sum(jnp.where(lane == self._page_of(s, shape), offset, 0))
                        chosen = (x > tau) | (tie & (ahead + before + inner < quota))
                        before = before + _row_sum(count)
                    else:
                        chosen = x >= tau
                    out = out | ((chosen & valid).astype(jnp.int32) << (8 * i))
                self.s.mask[w, block, :] = out
                return before

            jax.lax.fori_loop(0, (planes + 3) // 4, word, before)

        jax.lax.fori_loop(0, self.blocks, per_block, None)

    def _chosen_plane(self, keys, s, block, tau, quota, bound, with_ties, before, strict):
        """Plane ``s`` of a block's chosen keys (and the ties counted so far), as :meth:`_mark` chooses them."""
        shape = (SUBLANES, LANES)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        x = keys[jnp.minimum(s, self.planes - 1), block, :]
        valid = s * LANES + lane < bound
        if not with_ties:
            return (x >= tau) & valid, before
        tie = (x == tau) & valid
        count = tie.astype(jnp.int32)
        inner = jnp.dot(count.astype(jnp.bfloat16), strict, preferred_element_type=jnp.float32).astype(jnp.int32)
        ahead = _row_sum(jnp.where(lane == self._page_of(s, shape), self._state(OFFSET)[block, :], 0))
        chosen = (x > tau) | (tie & (ahead + before + inner < quota))
        return chosen & valid, before + _row_sum(count)

    def _mark_words(self, keys, with_ties):
        """The selection as lane-major bit words (``scratch_shapes``'
        ``words``) below each 8-row block's largest bound, 32 planes a words
        tile (plane ``s``'s chosen lanes on bit ``s % 32``: lane-wise shifts
        and ors, no cross-lane work), then each row's ``summary`` bits of its
        nonzero words and its ``count``. Words past a row's bound are not
        written in a step."""
        shape = (SUBLANES, LANES)
        lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        in_word = lane % 32
        strict = _strict_lanes()

        def pack(bits):
            for step in (1, 2, 4, 8, 16):  # lane 32 k + 31: the or of lanes 32 k ..
                bits = bits | jnp.where(in_word >= step, pltpu.roll(bits, step, 1), 0)
            return bits

        def place(bits, i, acc):
            # Lane 32 k + 31's word into lane 4 i + k.
            source = jnp.clip(32 * (lane - 4 * i) + 31, 0, LANES - 1)
            placed = jnp.take_along_axis(bits, source, axis=1, mode="promise_in_bounds")
            return jnp.where((lane >= 4 * i) & (lane < 4 * i + 4), placed, acc)

        def per_block(b, _):
            block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
            planes = self._block_planes(b)
            tau = self._state(BASE)[block, :] ^ INT32_MIN
            quota = INDEX_TOPK - self._state(UPPER)[block, :]
            bound = self.s.bounds[block, :]
            zero = jnp.zeros(shape, jnp.int32)

            def tile(g, carry):
                before, total = carry

                def plane(i, inner):
                    acc, before, total = inner
                    chosen, before = self._chosen_plane(keys, 32 * g + i, block, tau, quota, bound, with_ties, before,
                                                        strict)
                    chosen = chosen.astype(jnp.int32)
                    return acc | jnp.left_shift(chosen, i), before, total + chosen

                acc, before, total = jax.lax.fori_loop(0, jnp.minimum(32, planes - 32 * g), plane, (zero, before, total))
                self.s.words[g, block, :] = acc
                return before, total

            tiles = (planes + 31) // 32
            _, total = jax.lax.fori_loop(0, tiles, tile, (zero, zero))
            self.s.count[block, :] = jnp.broadcast_to(_row_sum(total), shape)

            def summarize(g, summary):
                nonzero = jnp.left_shift((self.s.words[g, block, :] != 0).astype(jnp.int32), in_word)
                return place(pack(nonzero), g, summary)

            self.s.summary[0, block, :] = jax.lax.fori_loop(0, tiles, summarize, zero)

        jax.lax.fori_loop(0, self.blocks, per_block, None)

    def _clear_new_tokens(self):
        """On the owner of each sparse row's new token, its bit (local token
        ``L - 1``)."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)

        @pl.loop(0, self.batch)
        def row(r):
            position = self.position_ref[r]

            @pl.when((position + 1 > INDEX_TOPK) & (position // self.page_tokens % self.group == self.owner))
            def _():
                token = dsa_decode.local_length(position, self.owner, self.page_tokens, self.group) - 1
                vrow = r * self.parts + token // self.part_tokens
                plane = token % self.part_tokens // LANES
                tile = pl.ds(pl.multiple_of(vrow // SUBLANES * SUBLANES, SUBLANES), SUBLANES)
                if self.word_mode:
                    words = self.s.words.at[token // LANES // 32]
                    hit = (sub == vrow % SUBLANES) & (lane == token % LANES)
                    bit = jnp.left_shift(1, token // LANES % 32)
                    held = jnp.max(jnp.where(hit & ((words[tile, :] & bit) != 0), 1, 0))
                    words[tile, :] = jnp.where(hit, words[tile, :] & ~bit, words[tile, :])
                    self.s.count[tile, :] = self.s.count[tile, :] - jnp.where(sub == vrow % SUBLANES, held, 0)
                    # Its word now empty: the summary's bit too.
                    empty = jnp.max(jnp.where(hit & (words[tile, :] == 0), 1, 0))
                    word = token // LANES // 32 * LANES + token % LANES
                    marks = self.s.summary.at[0]
                    spot = (sub == vrow % SUBLANES) & (lane == word // 32)
                    marks[tile, :] = jnp.where(spot & (empty != 0), marks[tile, :] & ~jnp.left_shift(1, word % 32),
                                               marks[tile, :])
                    return
                words = self.s.mask.at[plane // 4]
                hit = (sub == vrow % SUBLANES) & (lane == token % LANES)
                words[tile, :] = jnp.where(hit, words[tile, :] & ~jnp.left_shift(1, 8 * (plane % 4)),
                                           words[tile, :])

    def _run(self, scores):
        keys = _Keys(scores)
        self._state(BASE)[...] = jnp.zeros((self.vrows, LANES), jnp.int32)
        self._state(UPPER)[...] = jnp.zeros((self.vrows, LANES), jnp.int32)
        self.s.histogram[...] = jnp.zeros(self.s.histogram.shape, jnp.int32)
        self._bounds()
        planes = self.s.smem[0]
        lane = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 1)

        def convert(b, _):
            block = pl.ds(pl.multiple_of(SUBLANES * b, SUBLANES), SUBLANES)
            bound = self.s.bounds[block, :]

            @pl.loop(0, self._block_planes(b))
            def to_keys(s):
                keys[s, block, :] = jnp.where(s * LANES + lane < bound, _score_key(keys.ref[s, block, :]), INT32_MIN)

        jax.lax.fori_loop(0, self.blocks, convert, None)
        form = self.form
        if form.stop == "off":
            jax.lax.fori_loop(0, form.rounds, lambda j, _: self._round(j, keys), None)
        else:
            self.s.smem[STOP_SLOT] = 0

            def stopping_round(j, _):
                @pl.when(self.s.smem[STOP_SLOT] == 0)
                def _():
                    self._round(j, keys)
                    check = (j >= form.first_stop) & (j < form.rounds - 1)
                    if form.stop == "even":
                        check = check & (j % 2 == 1)

                    @pl.when(check)
                    def _():
                        # Decided: count(>= the current bin's edge) is exactly k, so the edge is tau and every
                        # tie is taken (padding and early rows never need more).
                        open_rows = (self._state(AT)[...] != INDEX_TOPK) & (self._state(EARLY)[...] == 0)
                        decided = 1 - jnp.max(open_rows.astype(jnp.int32))
                        if self.rounds_cap:
                            self.s.smem[DECIDED_SLOT] = decided
                            self.s.smem[STOP_SLOT] = (j >= self.rounds_cap - 1).astype(jnp.int32)
                        else:
                            self.s.smem[STOP_SLOT] = decided

            jax.lax.fori_loop(0, form.rounds, stopping_round, None)
        # The last two rounds' sends (one of each parity; the first stop check follows the third round).
        for parity in (0, 1):
            _wait_sent(self.stages, self._round_wires(), parity)
        if self.rounds_cap:
            # Timing only: a capped stop goes on as a real one does (every row decided, no tie round).
            self._state(AT)[...] = jnp.full((self.vrows, LANES), INDEX_TOPK, jnp.int32)
        # Rows whose threshold ties overflow the quota: the same on every member (summed counts).
        need = (self._state(AT)[...] != INDEX_TOPK) & (self._state(EARLY)[...] == 0)
        self.s.smem[1] = jnp.max(need.astype(jnp.int32))
        ties = self.s.smem[1] != 0
        pl.when(ties)(lambda: self._tie_round(keys, planes))
        # Without the round every tie is taken: exactly the quota.
        pl.when(ties)(lambda: self._mark(keys, True))
        pl.when(jnp.logical_not(ties))(lambda: self._mark(keys, False))
        self._clear_new_tokens()

    def _pick(self, plane, vrow):
        """``[1, 128]``: state plane ``plane`` of virtual row ``vrow`` (traced)."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        tile = self._state(plane)[pl.ds(pl.multiple_of(vrow // SUBLANES * SUBLANES, SUBLANES), SUBLANES), :]
        return jnp.sum(jnp.where(sub == vrow % SUBLANES, tile, 0), axis=0, keepdims=True)

    def new_selected(self, row, score):
        """Whether row ``row``'s new token (FP32 ``score``, lane-broadcast
        ``[1, 128]``) is selected: a bool scalar, valid only after :meth:`run`
        in the same step and layer (True on an early row)."""
        return jnp.max(self.new_selected_lanes(row, score)) > 0

    def new_selected_lanes(self, row, score):
        """:meth:`new_selected` as int32 ``[1, 1]`` 0/1, with no scalar."""
        vrow = row * self.parts
        return _new_chosen(score, self._pick(BASE, vrow), self._pick(EARLY, vrow), self._pick(AT, vrow))

    def new_selected_tile(self, first, score):
        """:meth:`new_selected` of rows ``first .. first + 8`` (one part a row;
        ``first`` a multiple of 8, may be traced) from their scores ``score``
        ``[8, 128]``: int32 ``[8, 1]`` 0/1."""
        tile = pl.ds(first if isinstance(first, int) else pl.multiple_of(first, SUBLANES), SUBLANES)
        return _new_chosen(score, self._state(BASE)[tile, :], self._state(EARLY)[tile, :], self._state(AT)[tile, :])

    def lanes(self, row, first, tokens: int):
        """Int32 ``[1, tokens]`` 0/1: row ``row``'s selection of local tokens
        ``first .. first + tokens`` (both traced; ``first`` and the static
        ``tokens`` multiples of 128, ``tokens`` at most 1024); ones on an early
        row. Chunks past the row's last part read its last plane: those tokens
        are past the row's limit, which every fold masks."""
        if tokens % LANES or not 0 < tokens <= 8 * LANES:
            raise ValueError(f"select: {tokens} tokens a call, a multiple of {LANES} up to {8 * LANES}")
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        if self.word_mode:
            return self._word_lanes(row, first, tokens)
        pieces = []
        for c in range(tokens // LANES):
            token = first + LANES * c
            part = token // self.part_tokens
            plane = jnp.where(part < self.parts, token % self.part_tokens // LANES, self.planes - 1)
            vrow = row * self.parts + jnp.minimum(part, self.parts - 1)
            tile = self.s.mask[plane // 4, pl.ds(pl.multiple_of(vrow // SUBLANES * SUBLANES, SUBLANES), SUBLANES), :]
            word = jnp.sum(jnp.where(sub == vrow % SUBLANES, tile, 0), axis=0, keepdims=True)
            pieces.append(jax.lax.shift_right_logical(word, 8 * (plane % 4)) & 255)
        selected = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=1)
        return jnp.where(self.position_ref[row] + 1 <= INDEX_TOPK, 1, selected)


    def _word_lanes(self, row, first, tokens: int):
        """:meth:`lanes` from the lane-major bit words: one word tile load
        (``first`` a multiple of 128 whose ``tokens // 128`` planes stay in one
        32-plane tile, as a fold step's tokens do), each 128-token chunk its
        plane's bit of the row's lanes (lane-wise shifts, no cross-lane
        work)."""
        sub = jax.lax.broadcasted_iota(jnp.int32, (SUBLANES, LANES), 0)
        plane = first // LANES
        tile = jnp.minimum(plane // 32, self.s.words.shape[0] - 1)
        block = self.s.words[tile, pl.ds(pl.multiple_of(row // SUBLANES * SUBLANES, SUBLANES), SUBLANES), :]
        words = jnp.sum(jnp.where(sub == row % SUBLANES, block, 0), axis=0, keepdims=True)
        pieces = [jax.lax.shift_right_logical(words, (plane + c) % 32) & 1 for c in range(tokens // LANES)]
        selected = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=1)
        return jnp.where(self.position_ref[row] + 1 <= INDEX_TOPK, 1, selected)


def unpack_words(words, rows: int, local_tokens: int):
    """numpy: the lane-major bit words (int32 ``[tiles, V, 128]``,
    ``scratch_shapes``' ``words``) -> the bool mask ``[rows, local_tokens]``."""
    words = np.ascontiguousarray(words, np.int32).view(np.uint32)
    bits = (words[:, None] >> np.arange(32, dtype=np.uint32)[None, :, None, None]) & 1  # [tiles, 32, V, 128]
    by_row = bits.transpose(2, 0, 1, 3).reshape(words.shape[1], -1)
    return by_row[:rows, :local_tokens] != 0


def select_rows(scores, positions, new_scores, *, page_tokens: int, group: int, form: SelectForm, owners=None,
                words: bool = False, interpret=None, stats: bool = False):
    """The select as a standalone call (tests), inside a ``shard_map`` over
    ``"tp"`` whose groups of ``group`` ranks start at mesh rank 0, member =
    the rank, page owner ``owners[member]`` (an int32 ``[group]`` permutation;
    default the member): this rank's ``scores`` ``[planes, V, 128]`` FP32
    (:func:`dsa_decode.score_shape`), the rows' ``positions`` ``[rows]`` and
    their new tokens' scores ``new_scores`` ``[rows]`` FP32 (every rank's
    copy). Returns ``(selection, new_selected [rows] int32)``: the mask words
    ``[words, V, 128]`` int32 (past the call's largest bound zero), or with
    ``words`` (one virtual row a row) ``(bit words, summary, count)``
    (:func:`scratch_shapes`), and with ``stats`` whether the rounds stopped
    early (``stop``, ``[1]``). Interpret mode only (``interpret``: its
    parameters): the call has no entry barrier."""
    planes, vrows, _ = scores.shape
    rows = positions.shape[0]
    local_tokens = planes * dsa_decode.virtual_rows(rows)[1] * LANES
    shapes = scratch_shapes(rows, local_tokens, form, words=words)
    selection_names = ("words", "summary", "count") if words else ("mask",)

    def kernel(position_ref, owner_ref, scores_ref, new_ref, *refs):
        outs, refs = refs[:len(selection_names)], refs[len(selection_names):]
        flags_ref, _, bounds, state, histogram, smem, *wires = refs
        member = jax.lax.axis_index("tp")
        views = dict(zip(selection_names, outs))
        select = TopkSelect(SelectScratch(views.get("mask"), bounds, state, histogram, smem, tuple(wires),
                                          words=views.get("words"), summary=views.get("summary"),
                                          count=views.get("count")),
                            batch=rows, local_tokens=local_tokens, position_ref=position_ref, owner=owner_ref[member],
                            member=member, base=0, group=group, page_tokens=page_tokens, form=form)
        for out in outs:
            out[...] = jnp.zeros(out.shape, jnp.int32)
        smem[STOP_SLOT] = 0
        pl.when(dsa_decode.scores_needed(position_ref, rows))(lambda: select.run(scores_ref))
        flags_ref[rows] = smem[STOP_SLOT]

        @pl.loop(0, rows)
        def flag(r):
            chosen = select.new_selected(r, new_ref[pl.ds(r, 1), :])
            flags_ref[r] = jnp.where(position_ref[r] + 1 <= INDEX_TOPK, 1, chosen.astype(jnp.int32))

    vmem, smem = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.SMEM)
    *selection, flags, _ = pl.pallas_call(
        kernel,
        out_shape=(*(jax.ShapeDtypeStruct(*shapes[name]) for name in selection_names),
                   jax.ShapeDtypeStruct((rows + 1,), jnp.int32), jax.ShapeDtypeStruct(scores.shape, jnp.float32)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, vmem],
            out_specs=[*(vmem for _ in selection_names), smem, vmem],
            scratch_shapes=(*(pltpu.VMEM(*shapes[name]) for name in ("bounds", "state", "histogram")),
                            *scoped_scratch(), *persistent_scratch(rows, local_tokens, page_tokens, group, form)),
        ),
        input_output_aliases={2: len(selection_names) + 1},
        interpret=interpret,
        name="k31_dsa_select",
    )(jnp.asarray(positions, jnp.int32), jnp.arange(group, dtype=jnp.int32) if owners is None
      else jnp.asarray(owners, jnp.int32), scores,
      jnp.broadcast_to(jnp.asarray(new_scores, jnp.float32)[:, None], (rows, LANES)))
    selection = selection[0] if len(selection) == 1 else tuple(selection)
    return (selection, flags[:rows], flags[rows:]) if stats else (selection, flags[:rows])


def unpack_mask(words, rows: int, local_tokens: int):
    """numpy: the mask words (int32 ``[words, V, 128]``; byte ``i`` of word
    ``w`` is plane ``4 w + i``'s 0/1) -> the bool mask ``[rows, local_tokens]``."""
    planes = dsa_decode.score_shape(local_tokens, rows)[0]
    words = np.ascontiguousarray(words, np.int32)
    by_plane = words.view(np.uint8).reshape(*words.shape, 4).transpose(0, 3, 1, 2)
    by_plane = by_plane.reshape(-1, *words.shape[1:])[:planes]
    return dsa_decode.scores_by_row(by_plane, rows)[:, :local_tokens] != 0
