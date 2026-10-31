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
"""Moves expert slabs between ranks for the dynamic re-layout.

A `Plan` (expert_relayout.py) names, per layer, which expert moves from
which rank's slot into which rank's staging slot. The mover runs it in a
background thread, three legs per slab, none of them on the serving steps'
critical path:

1. device -> host on the source: tpu-sync's raw transfer of exactly the
   slot's major-dimension slice of the weight tensor into pinned host memory
   (`transfer_d2h`, paced so the copy never parks the step's kernels);
2. host -> host over a gloo process group of its own (never the group the
   step's all-reduce uses), `isend` / `irecv` of the pinned buffers, tagged
   by (group, layer, destination slot, part);
3. host -> device on the destination into the staging slot (`transfer_h2d`),
   which no table references, so the write is invisible to the steps until
   the controller installs the table.

The per-contraction-block scale tables are small (about 100 KB per slot)
and flat ([slots * rows, N]); the source sends the slot's rows, the
destination writes them with a torch index copy. The bytes copied are the
tensor's own, so the destination's slot holds exactly the source's and the
layer's output after the install is the static layout's bit for bit.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The expert tensors of a placed layer, by part id (what move_tag carries).
# Each part names its attribute candidates in order: the fused EP kernel reads
# the row-padded flat scale tables (`_tpu_fused_*_scale`, mxfp4.py), not the
# quantizer's [E, blocks, 1, N] scale Parameters, so those are what move.
PART_ALIASES = (
    ("w13_weight",),
    ("w2_weight",),
    ("_tpu_fused_w13_scale", "w13_weight_scale"),
    ("_tpu_fused_w2_scale", "w2_weight_scale"),
    ("w13_bias",),
    ("w2_bias",),
)
PARTS = tuple(aliases[0] for aliases in PART_ALIASES)
WEIGHT_PARTS = PARTS[0:2]
SCALE_PARTS = PARTS[2:4]
BIAS_PARTS = PARTS[4:6]
_GROUP_IDS = {"target": 0, "draft": 1}


def move_tag(
    group: str, layer: int, dst_slot: int, part: int, generation: int = 0
) -> int:
    """A unique int32 tag per (group, layer, destination slot, part), and per
    generation modulo 128: a receive a dropped generation left posted can
    never match a later generation's send (the base fits in 23 bits)."""
    base = ((_GROUP_IDS.get(group, 2) * 1024 + layer) * 256 + dst_slot) * 8 + part + 1
    return ((int(generation) % 128) << 23) + base


class _ExchangeState:
    """One generation's link between the pump and its exchange thread. A
    new one per generation: a thread a dropped generation left behind keeps
    its own queues and flags and never touches the next generation's."""

    def __init__(self, generation: int = 0, incoming=None) -> None:
        self.generation: int = generation
        self.incoming = incoming if incoming is not None else []
        self.send_q: queue.Queue = queue.Queue()
        self.recv_q: queue.Queue = queue.Queue()
        self.sends_closed = threading.Event()
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.error: BaseException | None = None
        # Counters the thread keeps for `progress()` (plain ints: a racy read
        # is fine for a diagnostic).
        self.recvs_posted = 0
        self.recvs_done = 0
        self.sends_posted = 0
        self.sends_done = 0


# ---------------------------------------------------------------------------
# Device <-> host, by major-dimension slices of a tensor
#
# tpu-sync's raw transfer copies a tensor's major-dimension slices as the
# PHYSICAL bytes of its device buffer (tiled, packed), paced, asynchronously,
# straight into the live storage the compiled program reads (compiled mode
# looks each argument's buffer up per call, so a DMA into a staging slot is
# what the next step sees once the slot table points at it). A slot of a
# slot-major weight is one slice; its bytes mean nothing on the host but land
# byte-for-byte in a slot of a same-shaped tensor on another rank.
#
# The rows of a flat scale table ([slots * rows, N]) cannot be a partial copy
# (rank-2 partial copies are refused: one row is a fraction of a tile row), so
# a table moves whole: D2H the table, cut or patch the slot's rows in the host
# image (rows is a multiple of the f32 tile's 8 rows, so a slot's rows are one
# contiguous physical range), H2D the table back. The slice byte size comes
# from raw_transfer.major_slice_bytes: the device layout can pad the minor
# dims, and the fp4 pair dtype's storage is accounted at one byte per nibble,
# so nbytes / shape[0] is not it in general.
#
# torch_tpu dispatches device tensors from one thread only, so every device
# leg is ISSUED from the controller's step hook on the main thread,
# asynchronously, and polled there. Only host memory crosses to the mover's
# thread.


class _Done:
    """A future that is already complete (synchronous transports)."""

    def is_ready(self) -> bool:
        return True


def _pinned(shape) -> torch.Tensor:
    try:
        return torch.empty(shape, dtype=torch.uint8, pin_memory=True)
    except (RuntimeError, TypeError):
        return torch.empty(shape, dtype=torch.uint8)


_TPU_SYNC_REQUIREMENT = (
    "TPU_MOE_EPLB_DYNAMIC=1 needs a tpu-sync-torch build whose "
    "tpu_sync.api.torch.raw_transfer supports major_slice_bytes and fp4 "
    "(float4_e2m1fn_x2) raw transfers; "
    "unset TPU_MOE_EPLB_DYNAMIC if that build is not installed"
)


class RawTransferTransport:
    """Slices through tpu-sync's raw PJRT copies into pinned host memory,
    asynchronously (a PjRtCopyFuture per copy), paced by the library so a copy
    never parks the step's kernel launches. Host buffers are [count, slice
    bytes] uint8, pooled by shape across generations (pinning costs
    milliseconds per buffer); `release` hands one back."""

    def __init__(self) -> None:
        try:
            from tpu_sync.api.torch import raw_transfer

            # The module's __getattr__ loads the extension for unknown names,
            # so an old tpu-sync answers this with an import error.
            usable = callable(raw_transfer.major_slice_bytes)
        except Exception as e:  # noqa: BLE001 - any failure means "not usable"
            raise ImportError(_TPU_SYNC_REQUIREMENT + f": {e!r}") from e
        if not usable:
            raise ImportError(_TPU_SYNC_REQUIREMENT)
        self._rt = raw_transfer
        self._pool: dict[tuple[int, ...], list[torch.Tensor]] = {}
        self._owned: set[int] = set()  # data pointers of buffers this pool made
        self._lock = threading.Lock()
        # Pinned copies made for unpinned h2d sources; `release_pending`
        # returns them to the pool once the mover's generation is done.
        self._pending_release: list[torch.Tensor] = []
        self._img: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.free_d2h: bool = False

    def slice_bytes(self, t: torch.Tensor) -> int:
        return int(self._rt.major_slice_bytes(t))

    def _image(self, t: torch.Tensor) -> torch.Tensor:
        hit = self._img.get(id(t))
        if hit is not None and hit[0] is t:
            return hit[1]
        rows = int(t.shape[0])
        host = _pinned((rows, self.slice_bytes(t)))
        if hasattr(self._rt, "transfer_d2h"):
            try:
                self._rt.transfer_d2h(t, host, unsafe_skip_buffer_lock=True)
            except TypeError:
                self._rt.transfer_d2h(t, host)
        else:
            fut = self._rt.transfer_d2h_async(
                t,
                host,
                src_offsets_major_dim=[0],
                dst_offsets_major_dim=[0],
                copy_sizes_major_dim=[rows],
            )
            if hasattr(fut, "block_until_ready"):
                fut.block_until_ready()
        self._img[id(t)] = (t, host)
        return host

    def prewarm(self, modules) -> int:
        """Copy every placed tensor's physical bytes into a pinned host image
        once at arm, so runtime D2H slices are served from host memory with
        zero device access and zero step budget cost."""
        if not hasattr(self._rt, "transfer_d2h") and not hasattr(
            self._rt, "transfer_d2h_async"
        ):
            return 0
        total = 0
        t0 = time.monotonic()
        for module in modules:
            for _, _name, t, _kind in _parts(module):
                total += self._image(t).numel()
        self.free_d2h = True
        logger.info(
            "expert mover | raw host image of %d placed tensors: %.2f GB in %.1f s",
            len(self._img),
            total / 2**30,
            time.monotonic() - t0,
        )
        return total

    def _take(self, shape: tuple[int, ...]) -> torch.Tensor:
        with self._lock:
            free = self._pool.get(shape)
            if free:
                return free.pop()
        host = _pinned(shape)
        with self._lock:
            self._owned.add(host.data_ptr())
        return host

    def _ours(self, host: torch.Tensor) -> bool:
        with self._lock:
            return host.data_ptr() in self._owned

    def release(self, host: torch.Tensor) -> None:
        if self._ours(host):
            with self._lock:
                self._pool.setdefault(tuple(host.shape), []).append(host)

    def release_pending(self) -> None:
        pending, self._pending_release = self._pending_release, []
        for host in pending:
            self.release(host)

    def d2h(self, t: torch.Tensor, first: int, count: int):
        hit = self._img.get(id(t))
        if hit is not None and hit[0] is t:
            view = hit[1][first : first + count]
            if count == 1:
                return _Done(), view
            return _Done(), view.clone()
        host = self._take((count, self.slice_bytes(t)))
        future = self._rt.transfer_d2h_async(
            t,
            host,
            src_offsets_major_dim=[first],
            dst_offsets_major_dim=[0],
            copy_sizes_major_dim=[count],
        )
        return future, host

    def h2d(self, host: torch.Tensor, t: torch.Tensor, first: int):
        hit = self._img.get(id(t))
        if hit is not None and hit[0] is t:
            count = int(host.shape[0])
            img = hit[1]
            img[first : first + count].copy_(host.reshape(count, -1))
            if (
                count == int(t.shape[0])
                and first == 0
                and hasattr(self._rt, "transfer_h2d_async")
            ):
                try:
                    return self._rt.transfer_h2d_async(
                        img, t, unsafe_skip_buffer_lock=True
                    )
                except TypeError:
                    pass
            try:
                return self._rt.transfer_h2d_async(
                    img,
                    t,
                    src_offsets_major_dim=[first],
                    dst_offsets_major_dim=[first],
                    copy_sizes_major_dim=[count],
                    unsafe_skip_buffer_lock=True,
                )
            except TypeError:
                return self._rt.transfer_h2d_async(
                    img,
                    t,
                    src_offsets_major_dim=[first],
                    dst_offsets_major_dim=[first],
                    copy_sizes_major_dim=[count],
                )
        if not self._ours(host):
            pinned = self._take(tuple(host.shape))
            pinned.copy_(host)
            host = pinned
            self._pending_release.append(pinned)
        return self._rt.transfer_h2d_async(
            host,
            t,
            src_offsets_major_dim=[0],
            dst_offsets_major_dim=[first],
            copy_sizes_major_dim=[int(host.shape[0])],
        )

    @staticmethod
    def ready(future) -> bool:
        return bool(future.is_ready())


class CpuTransport:
    """Slices of CPU tensors through torch, as logical bytes: the transport
    the tests drive the mover with (torch cannot move the fp4 pair dtype on a
    TPU, which is why serving uses tpu-sync)."""

    def slice_bytes(self, t: torch.Tensor) -> int:
        return t.nbytes // int(t.shape[0])

    def release(self, host: torch.Tensor) -> None:
        pass

    def release_pending(self) -> None:
        pass

    def d2h(self, t: torch.Tensor, first: int, count: int):
        rows = t[first : first + count].detach().contiguous()
        return _Done(), rows.view(torch.uint8).reshape(count, -1).clone()

    def h2d(self, host: torch.Tensor, t: torch.Tensor, first: int):
        count = int(host.shape[0])
        dst = t[first : first + count]
        dst.copy_(host.reshape(count, -1).contiguous().view(t.dtype).reshape(dst.shape))
        return _Done()

    @staticmethod
    def ready(future) -> bool:
        return bool(future.is_ready())


# ---------------------------------------------------------------------------
# Host <-> host


class GlooP2P:
    """Point-to-point tensors over a dedicated gloo group. `rank_map` turns
    the plan's EP ranks into the global ranks torch.distributed addresses."""

    def __init__(self, group, rank_map: list[int] | None = None) -> None:
        self.group = group
        self.rank_map = rank_map

    def _global(self, rank: int) -> int:
        return int(self.rank_map[rank]) if self.rank_map is not None else int(rank)

    def send(self, dst: int, tag: int, tensor: torch.Tensor):
        """Non-blocking: two ranks sending to each other must not rendezvous
        on blocking sends. Returns the work handle to wait on."""
        return torch.distributed.isend(
            tensor.contiguous(), dst=self._global(dst), group=self.group, tag=tag
        )

    def recv(self, src: int, tag: int, tensor: torch.Tensor) -> None:
        torch.distributed.recv(tensor, src=self._global(src), group=self.group, tag=tag)

    def irecv(self, src: int, tag: int, tensor: torch.Tensor):
        """Non-blocking receive into `tensor`; the handle answers
        `is_completed()` (the exchange thread polls it) and `wait()`."""
        return torch.distributed.irecv(
            tensor, src=self._global(src), group=self.group, tag=tag
        )


def make_gloo_group(global_ranks: list[int]):
    """The mover's own process group over the EP ranks (a collective: every
    rank calls it, once, at arm time)."""
    return torch.distributed.new_group(
        ranks=[int(r) for r in global_ranks], backend="gloo"
    )


class LocalExchange:
    """An in-process mailbox standing in for the gloo group in tests: ranks
    are threads, a send parks the tensor under (src, dst, tag) and a recv
    waits for it."""

    def __init__(self) -> None:
        self._box: dict[tuple[int, int, int], torch.Tensor] = {}
        self._cv = threading.Condition()

    def endpoint(self, rank: int) -> _LocalEndpoint:
        return _LocalEndpoint(self, rank)

    def _send(self, src, dst, tag, tensor):
        with self._cv:
            self._box[(src, dst, tag)] = tensor.clone()
            self._cv.notify_all()

    def _recv(self, src, dst, tag, tensor, timeout=30.0):
        deadline = time.monotonic() + timeout
        with self._cv:
            while (src, dst, tag) not in self._box:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(f"no message {(src, dst, tag)}")
                self._cv.wait(left)
            tensor.copy_(self._box.pop((src, dst, tag)))


class _LocalEndpoint:
    def __init__(self, exchange: LocalExchange, rank: int) -> None:
        self.exchange = exchange
        self.rank = rank

    def send(self, dst, tag, tensor):
        self.exchange._send(self.rank, dst, tag, tensor)
        return None

    def recv(self, src, tag, tensor):
        self.exchange._recv(src, self.rank, tag, tensor)

    def irecv(self, src, tag, tensor):
        return _LocalRecv(self.exchange, src, self.rank, tag, tensor)


class _LocalRecv:
    """A non-blocking receive on the mailbox: completes when the message is
    there, copying it out."""

    def __init__(self, exchange, src, dst, tag, tensor) -> None:
        self.exchange, self.key, self.tensor = exchange, (src, dst, tag), tensor
        self.done = False

    def is_completed(self) -> bool:
        if not self.done:
            with self.exchange._cv:
                msg = self.exchange._box.pop(self.key, None)
            if msg is not None:
                self.tensor.copy_(msg)
                self.done = True
        return self.done

    def wait(self) -> None:
        if not self.done:
            self.exchange._recv(*self.key, self.tensor)
            self.done = True


# ---------------------------------------------------------------------------
# The mover

SLOT, ROWS = "slot", "rows"


def _parts(module) -> list[tuple[int, str, torch.Tensor, str]]:
    """(part id, attribute, tensor, kind) for every expert tensor the module
    has: SLOT for slot-major tensors (rank >= 3, one slot = one major slice),
    ROWS for the flat scale tables (rank 2, a slot = a block of rows). A module
    with none is a plan error, not a module with nothing to move."""
    out = []
    for part, aliases in enumerate(PART_ALIASES):
        for name in aliases:
            t = getattr(module, name, None)
            if t is None:
                continue
            # Keep the attribute object itself (a Parameter stays the same
            # object across calls, `.data` does not), so a transport may key a
            # per-tensor host image by identity.
            if t.dim() >= 3:
                kind = SLOT
            elif t.dim() == 2 and aliases[0] in SCALE_PARTS:
                kind = ROWS
            else:
                raise RuntimeError(
                    f"{name} of shape {tuple(t.shape)} is not movable: slots are "
                    "major slices of a rank >= 3 tensor or row blocks of a flat "
                    "scale table"
                )
            out.append((part, name, t, kind))
            break
    if not out:
        raise RuntimeError(f"{module!r} has none of the expert tensors {PARTS}")
    return out


def _slot_count(module) -> int:
    """Physical slots on this rank: the weights' leading dimension."""
    for name in WEIGHT_PARTS:
        t = getattr(module, name, None)
        if t is not None:
            return int(t.shape[0])
    return int(module.local_num_experts)


def move_bytes(module, transport) -> int:
    """Device bytes one slab move of this module costs on the wire: a slot of
    every slot part plus a slot's rows of every table part."""
    slots = _slot_count(module)
    total = 0
    for _, _name, t, kind in _parts(module):
        per_slice = transport.slice_bytes(t)
        total += per_slice if kind == SLOT else per_slice * (int(t.shape[0]) // slots)
    return int(total)


def describe_parts(module, transport=None) -> str:
    """One line per expert tensor: shape, dtype, kind and (with a transport)
    the device bytes of one slot -- logged at arm for the first layer."""
    lines = []
    slots = _slot_count(module)
    for _, name, t, kind in _parts(module):
        line = f"{name}: {tuple(t.shape)} {str(t.dtype).split('.')[-1]} {kind}"
        if transport is not None:
            per_slice = transport.slice_bytes(t)
            per_slot = (
                per_slice if kind == SLOT else per_slice * (int(t.shape[0]) // slots)
            )
            line += f", {per_slot / 1e6:.2f} MB/slot"
        lines.append(line)
    return "; ".join(lines)


class SlabMover:
    """Runs a plan's moves for one rank (the `Mover` protocol of
    expert_relayout.py) as one streaming pipeline driven by `pump()` from the
    runner right after each step's forward is dispatched:

    * device -> host: the source slots' copies and the whole-table copies of
      every scale table a move touches are issued under `step_budget_bytes`
      per pump, tables first; a payload is handed to the exchange thread the
      pump it lands (a table's outgoing rows are cut from its image);
    * host -> host: a thread posts every receive up front, sends payloads as
      they come, and reports arrivals;
    * host -> device: arrived slots go to their staging slots and arrived
      rows into their table's image, whose whole-table copy goes back once
      every patch is in, under the same budget.

    The generation is staged when every leg has drained.
    """

    def __init__(
        self,
        rank: int,
        transport,
        p2p,
        *,
        generation: int = 0,
        step_budget_bytes: int = 16 << 20,
    ):
        self.rank = rank
        self.transport = transport
        self.p2p = p2p
        # Bytes of DMA one pump may issue (0 = unbounded): the device runs
        # DMAs and program launches through one in-order queue, so a pump's
        # copies delay whatever is launched after them by their transfer time.
        self.step_budget_bytes = max(int(step_budget_bytes), 0)
        self._staged = generation
        self._failed = False
        self._phase = "idle"
        self._pumps = 0
        self._pump_s = 0.0
        self._pump_max_s = 0.0
        self._thread: threading.Thread | None = None
        self._x = _ExchangeState()
        self.last_report: dict[str, Any] = {}
        self._reset_work()

    def _reset_work(self) -> None:
        self._gen: int = 0
        self._t0 = 0.0
        self._out_moves: list = []
        self._in_moves: list = []
        # Outgoing SLOT copies: (move, part, tensor, slot) -> payload hosts.
        self._out_todo: list = []
        self._out_inflight: list = []  # (future, host, move, part)
        self._out_sent = 0  # payloads handed to the exchange thread
        self._out_expected = 0
        self._sent_total = 0
        # Incoming: (move, part, tensor, kind, slot, payload host).
        self._incoming: list = []
        self._in_received = 0
        # Incoming ROWS payloads waiting for their table's image: (index, key).
        self._rows_waiting: list = []
        # Whole-table images, keyed by (group, layer, part): {"tensor", "rows",
        # "host", "future", "ready", "out_rows": [(move, part)], "expected",
        # "patched"}; a table with incoming rows goes back whole once patched.
        self._tables: dict = {}
        self._tables_todo: list = []
        self._h2d_todo: list = []  # (host, tensor, first)
        self._h2d_inflight: list = []
        self._to_release: list = []  # transport buffers handed back at the end
        # The exchange thread: payloads in, arrivals (indices into
        # `_incoming`) out, and "nothing more to send" from the pump.
        self._x = _ExchangeState()
        # A dropped generation finishing its exchange without device writes
        # (`drain`), and one given up on with copies possibly in flight.
        self._draining = False
        self._abandoned = False
        self._marks: dict[str, float] = {}

    # -- Mover protocol

    def start(self, plan, modules) -> None:
        """Set the generation up; a plan the modules cannot serve marks the
        generation failed rather than raising into the step."""
        if self._phase != "idle":
            raise RuntimeError("a generation is still staging")
        self._reset_work()
        self._failed = False
        self._gen = int(plan.generation)
        self._t0 = time.monotonic()
        try:
            self._prepare(plan, modules)
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %s cannot start",
                self.rank,
                plan.generation,
            )
            self._failed = True
            self._phase = "idle"

    def _table(self, m, part, tensor, module):
        key = (m.group, m.layer, part)
        if key not in self._tables:
            rows = int(tensor.shape[0]) // _slot_count(module)
            self._tables[key] = {
                "tensor": tensor,
                "rows": rows,
                "host": None,
                "future": None,
                "ready": False,
                "out_rows": [],
                "expected": 0,
                "patched": 0,
            }
            self._tables_todo.append(key)
        return self._tables[key]

    def _prepare(self, plan, modules) -> None:
        incoming, outgoing = plan.moves_for(self.rank)
        for m in outgoing:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == SLOT:
                    self._out_todo.append((m, part, tensor, m.src_slot))
                else:
                    self._table(m, part, tensor, module)["out_rows"].append((m, part))
                self._out_expected += 1
        for m in incoming:
            module = modules[(m.group, m.layer)]
            for part, _, tensor, kind in _parts(module):
                if kind == SLOT:
                    nbytes = self.transport.slice_bytes(tensor)
                else:
                    table = self._table(m, part, tensor, module)
                    table["expected"] += 1
                    nbytes = self.transport.slice_bytes(tensor) * table["rows"]
                self._incoming.append(
                    (
                        m,
                        part,
                        tensor,
                        kind,
                        m.dst_slot,
                        torch.empty(nbytes, dtype=torch.uint8),
                    )
                )
        self._out_moves = outgoing
        self._in_moves = incoming
        if not outgoing and not incoming:
            self._phase = "stream"
            self._finish()
            return
        # The host leg streams from the start: receives are posted now and
        # every payload is sent the step it lands on the host.
        self._x = _ExchangeState(self._gen, self._incoming)
        self._thread = threading.Thread(
            target=self._exchange, args=(self._x,), name="expert-mover", daemon=True
        )
        self._thread.start()
        self._phase = "stream"

    def pump(self) -> None:
        """One bounded slice of device work, from the main thread."""
        if self._phase == "idle" or self._failed:
            return
        if self._x.cancel.is_set():
            self._phase = "idle"
            return
        t0 = time.perf_counter()
        try:
            getattr(self, f"_pump_{self._phase}")()
        except Exception:  # noqa: BLE001 - reported through failed()
            logger.exception(
                "expert mover | rank %d: generation %s failed", self.rank, self._gen
            )
            self._failed = True
            self._phase = "idle"
        finally:
            dt = time.perf_counter() - t0
            self._pumps += 1
            self._pump_s += dt
            self._pump_max_s = max(self._pump_max_s, dt)

    def staged_generation(self) -> int:
        return self._staged

    def failed(self) -> bool:
        return self._failed

    def cancel(self) -> None:
        self._x.cancel.set()

    def drain(self, timeout_s: float) -> bool:
        """Finish a dropped generation's host exchange without writing the
        device: every remaining device-to-host copy is issued (no budget),
        every payload sent, every posted receive absorbed and discarded.
        Every rank drops a timed-out generation at the same step and drains
        it, so every receive a peer posted gets its send; nothing is left to
        match a later generation's messages. True when the exchange and its
        copies finished within `timeout_s`; otherwise the generation is
        abandoned: its thread is told to stop and its host buffers are never
        reused (a copy may still be landing in them)."""
        if self._phase != "stream" or self._thread is None:
            return True
        x = self._x
        self._draining = True
        end = time.monotonic() + max(float(timeout_s), 0.0)
        try:
            while time.monotonic() < end and x.error is None:
                self._pump_stream()
                if self._drained():
                    break
                time.sleep(0.002)
        except Exception:  # noqa: BLE001 - an abandoned drain, reported below
            logger.exception(
                "expert mover | rank %d: draining generation %s failed",
                self.rank,
                self._gen,
            )
        ok = x.error is None and self._drained()
        if not ok:
            logger.warning(
                "expert mover | rank %d: drain gave up after %.0f s: %s",
                self.rank,
                timeout_s,
                self.progress(),
            )
            self._abandoned = True
            x.cancel.set()
        return ok

    def progress(self) -> str:
        """Where a generation stands, for the logs of a timed-out or abandoned
        one: device-to-host copies, the exchange's sends and receives, the
        arrivals applied and the host-to-device copies."""
        x = self._x
        tables = self._tables.values()
        alive = self._thread is not None and self._thread.is_alive()
        marks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self._marks.items()))
        return (
            f"gen {self._gen} phase {self._phase} pumps {self._pumps}"
            f" | d2h slabs {self._out_expected - len(self._out_todo)}"
            f"/{self._out_expected} issued, {len(self._out_inflight)} in flight"
            f" | tables {sum(t['future'] is not None for t in tables)}"
            f"/{len(self._tables)} issued, {sum(t['ready'] for t in tables)} landed"
            f" | handed {self._out_sent}/{self._out_expected}, queue"
            f" {x.send_q.qsize()}, sends {x.sends_done}/{x.sends_posted} done,"
            f" closed {x.sends_closed.is_set()}"
            f" | recvs {x.recvs_done}/{x.recvs_posted} done, applied"
            f" {self._in_received}/{len(self._incoming)}, rows waiting"
            f" {len(self._rows_waiting)}"
            f" | h2d {len(self._h2d_todo)} queued, {len(self._h2d_inflight)} in flight"
            f" | thread {'alive' if alive else 'gone'}, done {x.done.is_set()},"
            f" error {x.error!r}"
            f" | marks {marks}"
        )

    def _drained(self) -> bool:
        return (
            self._x.done.is_set()
            and not self._out_inflight
            and not self._h2d_inflight
            and all(tb["future"] is None or tb["ready"] for tb in self._tables.values())
        )

    def settle(self, generation: int) -> None:
        self._staged = generation
        if self._phase == "stream":  # dropped mid-generation
            self._x.sends_closed.set()
            if self._abandoned:
                # A device copy may still land in these: never reuse them.
                logger.warning(
                    "expert mover | rank %d: generation %s abandoned; %d host "
                    "buffers left unreleased",
                    self.rank,
                    self._gen,
                    len(self._to_release),
                )
            else:
                for host in self._to_release:
                    self.transport.release(host)
            self._to_release = []
        self._phase = "idle"

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -- phases

    def _budget(self) -> float:
        if self._draining or self.step_budget_bytes == 0:
            return float("inf")
        return self.step_budget_bytes

    def _pump_stream(self) -> None:
        """One step's slice of every leg, under one byte budget: issue D2H
        (tables first), hand landed payloads to the exchange thread, apply
        arrived payloads (slots to the H2D queue, rows into their table's
        image), issue H2D, and finish when every leg has drained."""
        x = self._x
        if x.error is not None:
            raise RuntimeError(f"host exchange failed: {x.error!r}")
        budget = self._budget()
        while self._tables_todo and budget > 0:
            table = self._tables[self._tables_todo.pop(0)]
            t = table["tensor"]
            table["future"], table["host"] = self.transport.d2h(t, 0, int(t.shape[0]))
            self._to_release.append(table["host"])
            if not getattr(self.transport, "free_d2h", False):
                budget -= table["host"].numel()
        while self._out_todo and budget > 0:
            m, part, tensor, slot = self._out_todo.pop(0)
            future, host = self.transport.d2h(tensor, slot, 1)
            self._out_inflight.append((future, host, m, part))
            self._to_release.append(host)
            if not getattr(self.transport, "free_d2h", False):
                budget -= host.numel()
        still = []
        for future, host, m, part in self._out_inflight:
            if self.transport.ready(future):
                self._send((m, part, host.reshape(-1)))
            else:
                still.append((future, host, m, part))
        self._out_inflight = still
        for key, table in self._tables.items():
            if table["ready"] or table["future"] is None:
                continue
            if not self.transport.ready(table["future"]):
                continue
            table["ready"] = True
            self._marks.setdefault("first_image_s", time.monotonic() - self._t0)
            image = table["host"].reshape(-1)
            span = table["rows"] * (image.numel() // int(table["tensor"].shape[0]))
            for m, part in table["out_rows"]:
                a = m.src_slot * span
                self._send((m, part, image[a : a + span].clone()))
            self._apply_waiting_rows(key)
        if (
            not x.sends_closed.is_set()
            and self._out_sent == self._out_expected
            and all(tb["ready"] or not tb["out_rows"] for tb in self._tables.values())
        ):
            self._marks["d2h_done_s"] = time.monotonic() - self._t0
            x.sends_closed.set()
        while True:
            try:
                i = x.recv_q.get_nowait()
            except queue.Empty:
                break
            self._in_received += 1
            if self._draining:
                continue  # absorbed, never applied
            self._marks.setdefault("first_arrival_s", time.monotonic() - self._t0)
            m, part, tensor, kind, dst_slot, payload = self._incoming[i]
            if kind == SLOT:
                self._h2d_todo.append((payload.reshape(1, -1), tensor, dst_slot))
                continue
            key = (m.group, m.layer, part)
            self._rows_waiting.append(i)
            self._apply_waiting_rows(key)
        while self._h2d_todo and budget > 0 and not self._draining:
            host, tensor, first = self._h2d_todo.pop(0)
            self._h2d_inflight.append(self.transport.h2d(host, tensor, first))
            budget -= host.numel()
        self._h2d_inflight = [
            f for f in self._h2d_inflight if not self.transport.ready(f)
        ]
        if (
            not self._draining
            and x.done.is_set()
            and x.sends_closed.is_set()
            and self._in_received == len(self._incoming)
            and not self._rows_waiting
            and not self._h2d_todo
            and not self._h2d_inflight
        ):
            if x.error is not None:
                raise RuntimeError(f"host exchange failed: {x.error!r}")
            self._finish()

    def _send(self, item) -> None:
        self._marks.setdefault("first_send_s", time.monotonic() - self._t0)
        self._out_sent += 1
        self._sent_total += int(item[2].numel())
        self._x.send_q.put(item)

    def _sent_bytes(self) -> int:
        return self._sent_total

    def _apply_waiting_rows(self, key) -> None:
        """Patch the arrived row payloads of one table into its image once
        the image is on the host; queue the table's H2D when complete."""
        table = self._tables[key]
        if not table["ready"] or self._draining:
            return
        image = table["host"].reshape(-1)
        span = table["rows"] * (image.numel() // int(table["tensor"].shape[0]))
        keep = []
        for i in self._rows_waiting:
            m, part, _tensor, _kind, dst_slot, payload = self._incoming[i]
            if (m.group, m.layer, part) != key:
                keep.append(i)
                continue
            a = dst_slot * span
            image[a : a + span].copy_(payload)
            table["patched"] += 1
        self._rows_waiting = keep
        if table["expected"] and table["patched"] == table["expected"]:
            table["expected"] = 0  # queued once
            self._h2d_todo.append((table["host"], table["tensor"], 0))

    def _exchange(self, x: _ExchangeState) -> None:
        """The host-to-host leg on the mover's thread, CPU memory only: every
        receive is posted up front (non-blocking), payloads are sent as the
        pump hands them over, arrivals are reported by index. Ends when the
        pump has closed the sends and every send and receive has completed."""
        try:
            recvs = []
            for i, (m, part, _t, _kind, dst_slot, payload) in enumerate(x.incoming):
                recvs.append(
                    (
                        i,
                        self.p2p.irecv(
                            m.src_rank,
                            move_tag(m.group, m.layer, dst_slot, part, x.generation),
                            payload,
                        ),
                    )
                )
            x.recvs_posted = len(recvs)
            sends = []
            while True:
                if x.cancel.is_set():
                    return
                while True:
                    try:
                        m, part, payload = x.send_q.get_nowait()
                    except queue.Empty:
                        break
                    h = self.p2p.send(
                        m.dst_rank,
                        move_tag(m.group, m.layer, m.dst_slot, part, x.generation),
                        payload,
                    )
                    x.sends_posted += 1
                    if h is not None:
                        sends.append(h)
                    else:
                        x.sends_done += 1
                pending = []
                for i, h in recvs:
                    if h.is_completed():
                        x.recvs_done += 1
                        x.recv_q.put(i)
                    else:
                        pending.append((i, h))
                recvs = pending
                still = [h for h in sends if not h.is_completed()]
                x.sends_done += len(sends) - len(still)
                sends = still
                if (
                    x.sends_closed.is_set()
                    and x.send_q.empty()
                    and not recvs
                    and not sends
                ):
                    return
                # gloo's work handles only report completion from wait(), so
                # polling is_completed() alone moved one slab per idle second.
                # Once every send is out, drain on this (background) thread:
                # wait each receive in order, handing it over as it lands,
                # then wait the sends. wait() releases the GIL.
                if x.sends_closed.is_set() and x.send_q.empty():
                    for i, h in recvs:
                        if x.cancel.is_set():
                            return
                        h.wait()
                        x.recvs_done += 1
                        x.recv_q.put(i)
                    recvs = []
                    for h in sends:
                        if x.cancel.is_set():
                            return
                        h.wait()
                        x.sends_done += 1
                    sends = []
                    continue
                time.sleep(0.0005)
        except BaseException as e:  # noqa: BLE001 - handed to the main thread
            x.error = e
        finally:
            x.done.set()

    def _finish(self) -> None:
        received = sum(p.numel() for *_, p in self._incoming)
        sent = self._sent_bytes()
        seconds = time.monotonic() - self._t0
        marks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self._marks.items()))
        self.last_report = {
            "generation": self._gen,
            "incoming": len(self._in_moves),
            "outgoing": len(self._out_moves),
            "received_bytes": received,
            "sent_bytes": sent,
            "tables": len(self._tables),
            "seconds": seconds,
            "pumps": self._pumps,
            "pump_seconds": self._pump_s,
            "pump_max_seconds": self._pump_max_s,
        }
        logger.info(
            "expert mover | rank %d staged generation %s: %d in / %d out, "
            "%.1f MB received, %.1f MB sent, %d tables re-imaged, %.2f s; "
            "%d pumps on the step thread, %.0f ms total, %.1f ms max; marks %s",
            self.rank,
            self._gen,
            len(self._in_moves),
            len(self._out_moves),
            received / 1e6,
            sent / 1e6,
            len(self._tables),
            seconds,
            self._pumps,
            self._pump_s * 1e3,
            self._pump_max_s * 1e3,
            marks,
        )
        self.last_report["marks"] = dict(self._marks)
        self._pumps, self._pump_s, self._pump_max_s = 0, 0.0, 0.0
        for host in self._to_release:
            self.transport.release(host)
        self._to_release = []
        self.transport.release_pending()
        assert self._gen is not None
        self._staged = self._gen
        self._phase = "idle"
