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
