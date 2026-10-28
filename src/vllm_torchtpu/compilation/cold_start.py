# SPDX-License-Identifier: Apache-2.0
"""Opt-in cold executable compaction before a serving worker becomes ready."""

import pickle
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.parallel_state import get_world_group

from vllm_torchtpu import envs


@dataclass(slots=True)
class ColdProgram:
    runtime: Callable[..., Any] | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        runtime = self.runtime
        assert runtime is not None, "Cold program called during startup compaction"
        return runtime(*args, **kwargs)


@dataclass(slots=True)
class _PendingProgram:
    slot: ColdProgram
    payload: bytes

    def restore(self) -> None:
        from vllm_torchtpu.compilation.tpu_compiler import _deserialize_entry

        fake_mode = (
            torch._guards.detect_fake_mode() or torch._subclasses.FakeTensorMode()
        )
        with fake_mode:
            entry = pickle.loads(self.payload)
        self.slot.runtime = _deserialize_entry(entry)
        self.payload = b""


@dataclass
class ColdStartCompaction:
    programs: int = 0
    backend_calls: int = 0
    _pending: list[_PendingProgram] = field(default_factory=list)

    def any_rank_compacted(self) -> bool:
        # A partial disk-cache hit must not split collective peers between
        # rewarm and ready. EP workers share a DP-inclusive vLLM world.
        flag = torch.tensor([int(self.programs > 0)], dtype=torch.int32, device="cpu")
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MAX, group=get_world_group().cpu_group
        )
        return bool(flag.item())

    def register(self, runtime: Callable[..., Any], entry: Any) -> ColdProgram:
        slot = ColdProgram(runtime)
        payload = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
        self._pending.append(_PendingProgram(slot, payload))
        self.programs += 1
        return slot

    def drop_runnables(self) -> None:
        # Drop every old AOT chain before loading any replacement so the
        # startup purge can return freed compiler allocations before replay.
        for pending in self._pending:
            pending.slot.runtime = None

    def restore_runnables(self) -> None:
        for pending in self._pending:
            pending.restore()
        self._pending.clear()


_current: ContextVar[ColdStartCompaction | None] = ContextVar(
    "cold_start_compaction", default=None
)


def current_compaction() -> ColdStartCompaction | None:
    return _current.get()


@contextmanager
def compact_cold_start() -> Iterator[ColdStartCompaction | None]:
    if not envs.TPU_COMPACT_COLD_START:
        yield None
        return
    assert _current.get() is None, "Cold-start compaction cannot be nested"
    state = ColdStartCompaction()
    token = _current.set(state)
    try:
        yield state
    finally:
        _current.reset(token)
