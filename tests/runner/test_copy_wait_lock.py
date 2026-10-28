# SPDX-License-Identifier: Apache-2.0
import threading
import time

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState


class _SlowEvent:
    def __init__(self):
        self.calls = 0

    def synchronize(self):
        self.calls += 1
        time.sleep(0.05)


def test_serialized_wait_synchronizes_once_across_threads(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_SERIALIZE_COPY_WAIT", True, raising=False)
    event = _SlowEvent()
    state = AsyncTPUCopyState([torch.zeros(2)], [2], event)
    threads = [threading.Thread(target=state.wait) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert event.calls == 1 and state.completed


def test_pinned_copy_reuses_a_ring_and_copies_tokens_out(monkeypatch):
    from vllm_torchtpu.runner import tpu_runner_async_output as mod

    monkeypatch.setattr(envs, "VLLM_TPU_PINNED_TOKEN_COPY", True, raising=False)
    monkeypatch.setattr(mod, "_pinned", {})
    real_empty = torch.empty
    monkeypatch.setattr(
        mod.torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k)
    )

    class _Event:
        def record(self):
            pass

        def synchronize(self):
            pass

    monkeypatch.setattr(mod.torch, "tpu", type("T", (), {"Event": _Event}), raising=False)
    states = []
    for step in range(mod._PINNED_RING + 2):
        src = torch.full((6,), step, dtype=torch.int32)
        state = mod.AsyncTPUCopyState.from_device_chunks([src], [4])
        state.wait()
        states.append((step, state.sampled_token_ids_cpu))
    # The ring holds _PINNED_RING buffers; tokens read at wait time stay intact
    # after their buffer is reused.
    assert len(mod._pinned[((6,), torch.int32)][0]) == mod._PINNED_RING
    for step, tokens in states:
        assert tokens.tolist() == [step] * 4
