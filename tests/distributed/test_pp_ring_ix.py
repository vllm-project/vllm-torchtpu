# SPDX-License-Identifier: Apache-2.0
"""TPU_PP_HANDOFF=ring on the IX PP4 x PCP2 layout: the ring is what carries
the forwards, and nothing falls back to the slice-wide wave.

Each of the 8 devices (4 stages x 2 one-chip PCP columns, global rank =
stage * 2 + column) builds its hand-off through the runner's capture_model,
runs the self-test, then serving steps through begin_step / start_forward /
end_forward over a fake point-to-point transport, as in test_pp_ring. The
wave's constructor, its slice-wide launch and its drain all raise, so a
fallback anywhere on that path fails the test instead of passing quietly.
"""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import pp_push, pp_ring, pp_wave

HIDDEN = 8
MAX_ROWS = 64
STAGES = 4
COLUMNS = 2
TIMEOUT = 20
TEMPLATE = {
    "hidden_states": ((HIDDEN,), torch.float32),
    "residual": ((HIDDEN,), torch.float32),
}


class _Work:
    def wait(self):
        return True


class _Fence:
    def synchronize(self):
        pass


class _Transport:
    """isend/irecv between device threads; a send waits for its receive."""

    def __init__(self, stride):
        self.stride = stride
        self.cv = threading.Condition()
        self.posted = {}
        self.landed = {}
        self.sends = []

    def irecv(self, tensor, src, tag):
        with self.cv:
            key = (src, src + self.stride, tag)
            assert key not in self.posted, f"receive {key} posted twice"
            self.posted[key] = tensor
            self.cv.notify_all()
        return _Work()

    def isend(self, tensor, dst, tag):
        key = (dst - self.stride, dst, tag)
        with self.cv:
            if not self.cv.wait_for(lambda: key in self.posted, timeout=TIMEOUT):
                raise TimeoutError(f"send {key} never found its receive")
            target = self.posted.pop(key)
            target.copy_(tensor)
            self.landed[id(target)] = target
            self.sends.append(key)
            self.cv.notify_all()
        return _Work()

    def wait_landed(self, tensor):
        with self.cv:
            assert self.cv.wait_for(
                lambda: self.landed.get(id(tensor)) is tensor, timeout=TIMEOUT
            ), "a received tensor never landed"


def _forbid_the_wave(monkeypatch):
    """Every entry into the slice-wide hand-off raises."""
    used = []

    def forbidden(name):
        def fail(*args, **kwargs):
            used.append(name)
            raise AssertionError(
                f"TPU_PP_HANDOFF=ring reached the slice-wide wave ({name})"
            )

        return fail

    monkeypatch.setattr(pp_wave.PPWave, "__init__", forbidden("PPWave"))
    monkeypatch.setattr(pp_wave, "_launch_now", forbidden("_launch_now"))
    monkeypatch.setattr(pp_wave, "_drain_slice", forbidden("_drain_slice"))
    monkeypatch.setattr(
        pp_wave, "drain_before_handoff", forbidden("drain_before_handoff")
    )
    return used


def _install_fakes(monkeypatch, transport):
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import pcp

    monkeypatch.setattr(torch.distributed, "isend", transport.isend)
    monkeypatch.setattr(torch.distributed, "irecv", transport.irecv)
    monkeypatch.setattr(pp_ring, "_flush", lambda tensors: None)
    monkeypatch.setattr(pp_ring, "_fence", _Fence)
    read = pp_ring._first_and_last

    def first_and_last(x):
        transport.wait_landed(x)
        return read(x)

    monkeypatch.setattr(pp_ring, "_first_and_last", first_and_last)
    local = threading.local()
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(
            rank_in_group=local.stage,
            world_size=STAGES,
            ranks=[s * COLUMNS + local.column for s in range(STAGES)],
        ),
    )
    monkeypatch.setattr(
        pcp,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=local.column, world_size=COLUMNS),
    )
    return local


def _fake_runner():
    runner = SimpleNamespace(
        _add_pipeline_chunk_buckets=lambda: None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=STAGES),
        _pp_wave=None,
        _pp_intermediate_template=None,
        _pp_topk_buffer=None,
        max_num_tokens=MAX_ROWS,
        device=torch.device("cpu"),
        enforce_eager=True,
        _qwen35_mtp_mk=None,
    )

    def intermediate_tensors(num_tokens, zeros):
        runner._pp_intermediate_template = TEMPLATE

    runner._pp_intermediate_tensors = intermediate_tensors
    return runner


def _device(local, stage, column, transport, steps, out, errors):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    try:
        local.stage, local.column = stage, column
        runner = _fake_runner()
        TPUModelRunner.capture_model(runner)
        ring = runner._pp_wave
        out["types"][stage, column] = type(ring).__name__
        out["drains"][stage, column] = TPUWorker.pp_drains_before_handoff(
            SimpleNamespace(model_runner=runner)
        )
        g = 0
        for rows_per_forward in steps:
            if stage > 0:
                ring.begin_step(rows_per_forward)
            for rows in rows_per_forward:
                received = ring.start_forward(rows)
                if stage == 0:
                    h = torch.full((rows, HIDDEN), float(g + 100 * column))
                    r = torch.full((rows, HIDDEN), 0.5)
                else:
                    transport.wait_landed(received["hidden_states"])
                    h = received["hidden_states"] + 1
                    r = received["residual"]
                if stage == STAGES - 1:
                    ring.end_forward(None, rows)
                    out["last"].append((column, g, rows, h.clone()))
                else:
                    ring.end_forward({"hidden_states": h, "residual": r}, rows)
                g += 1
        ring.close()
    except BaseException as exc:  # reported by the test thread
        errors.append(((stage, column), exc))


def _run_pipeline(monkeypatch, steps):
    transport = _Transport(stride=COLUMNS)
    local = _install_fakes(monkeypatch, transport)
    out = {"types": {}, "drains": {}, "last": []}
    errors = []
    threads = [
        threading.Thread(
            target=_device,
            args=(local, s, c, transport, steps, out, errors),
            daemon=True,
        )
        for s in range(STAGES)
        for c in range(COLUMNS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in threads), "a device never finished"
    return transport, out, errors


STEPS = [[64], [32, 16], [8], [64, 64, 32]]


def test_the_ring_flag_never_reaches_the_slice_wide_wave(monkeypatch):
    monkeypatch.setenv("TPU_PP_HANDOFF", "ring")
    monkeypatch.delenv("TPU_PP_RING_DEPTH", raising=False)
    monkeypatch.delenv("VLLM_PP_WAVE_ALWAYS_DRAIN", raising=False)
    used = _forbid_the_wave(monkeypatch)

    transport, out, errors = _run_pipeline(monkeypatch, STEPS)

    assert not errors, errors
    assert used == []
    # Every device built the ring, and none drains before a hand-off.
    assert out["types"] == {
        (s, c): "PPRing" for s in range(STAGES) for c in range(COLUMNS)
    }
    assert out["drains"] == {
        (s, c): False for s in range(STAGES) for c in range(COLUMNS)
    }
    # Every forward reached the last stage of its own column, carrying the
    # sum (hidden + residual) plus one per later stage.
    forwards = [rows for step in STEPS for rows in step]
    assert len(out["last"]) == len(forwards) * COLUMNS
    for column, g, rows, h in out["last"]:
        assert rows == forwards[g]
        want = g + 100 * column + 0.5 + (STAGES - 1)
        assert torch.equal(h, torch.full((rows, HIDDEN), want))
    # Point to point, column to column: every transfer goes two global ranks
    # on; the self-test plus every forward crosses each of the 3 boundaries
    # in each of the 2 columns.
    assert {dst - src for src, dst, _ in transport.sends} == {COLUMNS}
    assert len(transport.sends) == (1 + len(forwards)) * (STAGES - 1) * COLUMNS


def test_the_engine_side_of_the_ring_sends_no_push_or_settle(monkeypatch):
    from vllm.distributed.device_communicators import shm_broadcast
    from vllm.v1.executor import multiproc_executor

    monkeypatch.setenv("TPU_PP_HANDOFF", "ring")
    monkeypatch.setattr(pp_push, "_ring_chunks_needed", 0)

    class _Queue:
        def __init__(self, *args, **kwargs):
            pass

    class _Executor:
        def execute_model(self, scheduler_output, non_block=False):
            return "forward"

        def take_draft_token_ids(self):
            raise AssertionError("asked the workers for draft tokens")

    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)
    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _Executor)
    pp_push.patch_executor_for_pp_handoff(
        SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=4))
    )
    assert getattr(_Executor, "_tpu_pp_ring_patch", False)
    assert not getattr(_Executor, "_tpu_pp_wave_patch", False)
    assert _Executor().execute_model(None) == "forward"


def test_each_worker_logs_that_the_ring_carries_its_forwards(monkeypatch):
    monkeypatch.setenv("TPU_PP_HANDOFF", "ring")
    monkeypatch.delenv("TPU_PP_RING_DEPTH", raising=False)
    _forbid_the_wave(monkeypatch)
    lines = []
    lock = threading.Lock()

    def info(msg, *args):
        with lock:
            lines.append(msg % args)

    monkeypatch.setattr(pp_ring.logger, "info", info)

    _, _, errors = _run_pipeline(monkeypatch, STEPS)

    assert not errors, errors

    def count(fragment):
        return sum(fragment in line for line in lines)

    devices = STAGES * COLUMNS
    senders = (STAGES - 1) * COLUMNS
    assert count("PP ring: rank") == devices
    assert count("PP ring ACTIVE") == count("self-test passed") + count(
        "first scheduler step"
    ) + count("first forward received") + count("first forward sent")
    assert count("self-test passed") == devices
    assert count("first scheduler step") == senders  # stages 1-3 post ahead
    assert count("first forward received") == senders
    assert count("first forward sent") == senders
    for s in range(STAGES):
        for c in range(COLUMNS):
            assert count(f"rank {s}/{STAGES} column {c}/{COLUMNS} ") >= 2


def test_the_wave_flag_still_builds_the_wave(monkeypatch):
    """The control: with the flag off the runner builds the wave (the DR
    arm), so the ring test above would notice a swapped default."""
    from vllm_torchtpu.runner import tpu_runner

    monkeypatch.delenv("TPU_PP_HANDOFF", raising=False)
    built = []

    class _Wave:
        def __init__(self, *args, **kwargs):
            built.append("PPWave")

        def warmup(self):
            pass

    monkeypatch.setattr(tpu_runner, "PPWave", _Wave)
    monkeypatch.setattr(
        tpu_runner, "PPRing", lambda *a, **k: pytest.fail("built the ring")
    )
    tpu_runner.TPUModelRunner.capture_model(_fake_runner())
    assert built == ["PPWave"]
