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
"""The fused EP kernel's own per-slot routed row counts, accumulated on the
device for the dynamic expert re-layout (TPU_MOE_EPLB_DYNAMIC,
expert_relayout.py).

The fused EP layer all-gathers every rank's per-slot pair counts on every
step (`rows_by_dest`, [ep, physical slots]) to plan its dispatch, so every
rank already knows the whole step's load per physical slot with no further
collective. Each placed layer carries a [physical slots] int32 accumulator
that the op takes as a donated operand and returns updated; the controller
reads the accumulators back asynchronously ahead of its window boundaries.
The counts are replicated sums, identical on every rank after the same step.
"""

import numpy as np
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

LOAD_ATTR = "_tpu_expert_load"

# (layer name, layer) for every registered layer of this process, in
# registration order. The layers own the buffers; this only lists them.
_REGISTERED: list[tuple[str, torch.nn.Module]] = []
_STATE: dict = {"steps": 0}


def enabled() -> bool:
    """The accumulator exists when the dynamic re-layout needs it."""
    return bool(envs.TPU_MOE_EPLB_DYNAMIC)


def register(layer, *, physical: int, device: torch.device | None = None) -> None:
    """Give an armed layer its [physical] accumulator. Call after weight load."""
    if not enabled() or operand(layer) is not None:
        return
    if device is None:
        w = getattr(layer, "w13_weight", None)
        if w is None:
            w = getattr(layer, "w13_weight_packed", None)
        device = w.device if w is not None else torch.device("cpu")
    layer.register_buffer(
        LOAD_ATTR,
        torch.zeros((physical,), dtype=torch.int32, device=device),
        persistent=False,
    )
    _REGISTERED.append((layer.layer_name, layer))


def operand(layer):
    """The layer's accumulator, or None when the tap is off."""
    return getattr(layer, LOAD_ATTR, None)


def commit(layer, updated: torch.Tensor) -> None:
    """Write the op's returned accumulator back: the operand was donated."""
    getattr(layer, LOAD_ATTR).copy_(updated)


def registered_layers() -> int:
    """How many layers carry an accumulator in this process."""
    return len(_REGISTERED)


def _read_all_sync() -> np.ndarray:
    """Every registered layer's accumulator as one [layers, widest] int64 host
    array through torch: a stack program and a synchronous `.cpu()`. The
    fallback where the asynchronous reader is unavailable (CPU tests)."""
    buffers = [getattr(layer, LOAD_ATTR) for _, layer in _REGISTERED]
    width = max(b.numel() for b in buffers)
    stacked = torch.stack(
        [torch.nn.functional.pad(b, (0, width - b.numel())) for b in buffers]
    )
    return stacked.cpu().numpy().astype(np.int64)


class _AsyncReader:
    """Reads the accumulators the way the runner fetches sampled tokens: one
    non-blocking device-to-host copy per accumulator enqueued on the device
    stream right after the step's dispatch (`.to("cpu", non_blocking=True)`)
    and a recorded `torch.tpu.Event`, polled at the next hook. Nothing on the
    host waits at issue: the copies queue behind the program that produces
    the accumulators. (A tpu-sync raw transfer per accumulator worked but its
    constructor awaits that program on the host, ~170 ms per read at DP32.)"""

    def __init__(self) -> None:
        self._srcs: list[torch.Tensor] = []
        self._cpu: list[torch.Tensor] | None = None
        self._event = None
        self.latest: np.ndarray | None = None
        self.polls_block = False  # the event lacks query(): poll synchronizes

    @property
    def in_flight(self) -> bool:
        return self._event is not None

    def bind(self) -> None:
        self._srcs = [getattr(layer, LOAD_ATTR) for _, layer in _REGISTERED]
        event = torch.tpu.Event()  # raises where the device module is absent
        self.polls_block = not callable(getattr(event, "query", None))

    def issue(self) -> None:
        if self._event is not None:
            return
        self._cpu = [t.to("cpu", non_blocking=True) for t in self._srcs]
        event = torch.tpu.Event()
        event.record()
        self._event = event

    def _done(self, wait: bool = False) -> bool:
        assert self._event is not None
        if self.polls_block or wait:
            self._event.synchronize()
            return True
        return bool(self._event.query())

    def poll(self, wait: bool = False) -> bool:
        """Adopt a completed read; True when `latest` changed."""
        if self._event is None or self._cpu is None or not self._done(wait=wait):
            return False
        width = max(t.numel() for t in self._cpu)
        host = np.zeros((len(self._cpu), width), dtype=np.int64)
        for i, c in enumerate(self._cpu):
            host[i, : c.numel()] = c.numpy().astype(np.int64, copy=False)
        self._event, self._cpu = None, None
        self.latest = host
        return True


_READER: _AsyncReader | None = None


def _disable_reader(e: Exception) -> None:
    global _READER
    logger.warning(
        "expert load | asynchronous readback failed (%s); reading synchronously", e
    )
    _STATE["raw_disabled"] = True
    _READER = None


def _reader() -> _AsyncReader | None:
    """The asynchronous reader once bound, or None (synchronous torch path)."""
    global _READER
    if _READER is not None or _STATE.get("raw_disabled"):
        return _READER
    if not _REGISTERED or getattr(_REGISTERED[0][1], LOAD_ATTR).device.type == "cpu":
        _STATE["raw_disabled"] = True
        return None
    try:
        reader = _AsyncReader()
        reader.bind()
    except Exception as e:  # noqa: BLE001 - the torch path stays
        _disable_reader(e)
        return None
    logger.info(
        "expert load | asynchronous readback armed (%s)",
        "poll synchronizes" if reader.polls_block else "event query",
    )
    _READER = reader
    return reader


def _read_all() -> np.ndarray | None:
    """The counts for a consumer (a window boundary): the reader's latest
    adopted read, or None when none has landed yet. Never a wait: the hook
    runs before the step is dispatched, and a read's DMA is chained behind
    the program that produces the accumulators, which the hook itself has yet
    to dispatch (waiting there deadlocks the step). The torch path, where the
    asynchronous reader is unavailable, reads synchronously."""
    reader = _reader()
    if reader is None:
        return _read_all_sync()
    return reader.latest


def on_step(step: int) -> None:
    """Called once per coordinated step on the host, before the step is
    dispatched: records the step and adopts a completed read."""
    if not _REGISTERED:
        return
    _STATE["steps"] = int(step)
    reader = _reader()
    if reader is not None:
        try:
            reader.poll()
        except Exception as e:  # noqa: BLE001 - fall back for good
            _disable_reader(e)


def issue_now() -> None:
    """Issue an asynchronous readback right after the current step's
    dispatch, first adopting any read still in flight so a phase-entry or
    pre-window read is never skipped."""
    reader = _reader()
    if reader is None:
        return
    try:
        if reader.in_flight:
            reader.poll(wait=True)
        reader.issue()
    except Exception as e:  # noqa: BLE001
        logger.warning("expert load | on-demand readback issue failed (%s)", e)


def flush() -> None:
    """Adopt any in-flight read issued in an earlier step, waiting for it, so
    every lockstep rank takes its controller snapshot from the same read."""
    reader = _reader()
    if reader is not None and reader.in_flight:
        try:
            reader.poll(wait=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("expert load | asynchronous readback flush failed (%s)", e)


def snapshot() -> dict[tuple[str, int], np.ndarray]:
    """Every registered layer's cumulative counts, keyed by its layout
    (group, layer index), as int64 host arrays. The counts are replicated
    sums, identical on every rank after the same step, which is what lets
    the dynamic re-layout solve without a collective (expert_relayout.py)."""
    from vllm_torchtpu import expert_placement

    if not _REGISTERED:
        return {}
    host = _read_all()
    if host is None:
        return {}
    out = {}
    for i, (_name, layer) in enumerate(_REGISTERED):
        key = expert_placement.layout_key(layer)
        if key is None:
            continue
        out[key] = host[i, : getattr(layer, LOAD_ATTR).numel()].copy()
    return out
