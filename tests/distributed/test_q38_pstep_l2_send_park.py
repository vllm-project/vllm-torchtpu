# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-L2: event-gated, parked Stage-3 send registration
(VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION; stage3_send_park.py).

The fake device below models what the gate must respect: the KV writes of a
finished request are pending until the device runs them. The blocking sync
(wait=True) runs them before it returns; the non-blocking sync (wait=False)
only enqueues them, and they land when the device reaches the event recorded
behind them (FakeDevice.run_to, or the event's synchronize()). Every
registration snapshots the cache values, so a registration that runs before
the writes landed reads zeros. Every test line prints a Q38_PSTEP_L2 marker.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu.distributed.kv_transfer import stage3_send_park
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

from .test_stage3_grouped_ahead import TABLES, _send, _worker, env  # noqa: F401
from .test_tpu_connector import _MOD

KEY = "VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION"
WANT = [1.0, 2.0, 3.0, 4.0]
UUID_A, UUID_B, UUID_C = 101, 202, 303


def _mark(name, **rows):
    print(
        "Q38_PSTEP_L2 %s %s"
        % (name, " ".join(f"{k}={v}" for k, v in sorted(rows.items()))),
        flush=True,
    )


class FakeDevice:
    """Pending KV writes and the events recorded behind them, in order."""

    def __init__(self, caches):
        self.caches = caches
        self.pending = []  # one write batch per sync, oldest first
        self.enqueued = 0
        self.landed = 0
        self.events = []
        self.trace = []

    def write_values(self):
        return [float(v) for v in range(1, len(self.caches) + 1)]

    def land_through(self, seq):
        """Every batch enqueued before sequence number `seq` lands."""
        while self.landed < seq and self.pending:
            batch = self.pending.pop(0)
            for tensor, value in zip(self.caches, batch):
                tensor.fill_(value)
            self.landed += 1

    def sync(self, tensors, *, wait):
        assert [id(t) for t in tensors] == [id(t) for t in self.caches]
        self.trace.append(("sync", wait))
        self.pending.append(self.write_values())
        self.enqueued += 1
        if wait:
            self.land_through(self.enqueued)

    def new_event(self):
        event = FakeEvent(self, self.enqueued)
        self.events.append(event)
        self.trace.append(("event", self.enqueued))
        return event

    def run_to(self, event):
        """The device reaches `event`: every write enqueued before it lands."""
        event.complete()


class FakeEvent:
    def __init__(self, device, covers):
        self.device = device
        self.covers = covers
        self.done = False
        self.queries = 0
        self.waits = 0

    def complete(self):
        if not self.done:
            self.device.land_through(self.covers)
            self.done = True

    def query(self):
        self.queries += 1
        return self.done

    def synchronize(self):
        self.waits += 1
        self.complete()


def _setup(env, monkeypatch, *, key, pipeline=True, spec="mtp"):  # noqa: F811
    if key:
        monkeypatch.setenv(KEY, "1")
    else:
        monkeypatch.delenv(KEY, raising=False)
    worker, facade = _worker("qwen", env)
    worker.vllm_config.speculative_config = (
        SimpleNamespace(method=spec) if spec else None
    )
    caches = [torch.zeros(1) for _ in range(4)]
    worker.runner = SimpleNamespace(kv_caches=[caches[0], caches[1:3], caches[3]])
    device = FakeDevice(caches)
    monkeypatch.setattr(f"{_MOD}.synchronize_tensors", device.sync)
    monkeypatch.setattr(f"{_MOD}._stage3_park_fence_event", device.new_event)
    env.enter_context(patch(f"{_MOD}._engine_is_pipeline", return_value=pipeline))
    seen = []
    original = facade.register_request_blocks

    def register(**kwargs):
        seen.append((kwargs["req_id"], kwargs["uuid"], [t.item() for t in caches]))
        device.trace.append(("register", kwargs["req_id"]))
        return original(**kwargs)

    monkeypatch.setattr(facade, "register_request_blocks", register)
    return worker, facade, device, seen


def _meta(sends=(), release=None):
    meta = TPUConnectorMetadata()
    ahead_tables, _, num_tokens = TABLES["qwen"]
    for req_id, uuid, ahead in sends:
        meta.reqs_to_send[req_id] = _send(uuid, ahead_tables, num_tokens, ahead=ahead)
    for req_id, uuid in (release or {}).items():
        meta.reqs_to_release[req_id] = uuid
    return meta


# ---------------------------------------------------------------------------
# Key off: today's path (blocking sync, registration in the same call).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pipeline", [True, False])
def test_key_off_is_the_blocking_path(env, monkeypatch, pipeline):  # noqa: F811
    worker, facade, device, seen = _setup(env, monkeypatch, key=False, pipeline=pipeline)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    assert device.trace == [("sync", True), ("register", "a")]
    assert seen == [("a", UUID_A, WANT)]
    assert device.events == []
    assert "_stage3_send_park" in worker.__dict__ and worker._stage3_send_park is None
    worker.get_finished({"a"})
    _mark("KEYOFF", pipeline=int(pipeline), trace=len(device.trace), ok=1)


# ---------------------------------------------------------------------------
# Key on, pipeline producer: non-blocking sync, event, parked registration.
# ---------------------------------------------------------------------------


def test_parked_registration_waits_for_device_writes(env, monkeypatch):  # noqa: F811
    """The send is parked behind its event; the next call (a no-forward step:
    empty metadata) registers it only after the writes landed."""
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    assert device.trace == [("sync", False), ("event", 1)]
    assert seen == [] and facade.register_request_blocks_calls == []
    assert [t.item() for t in device.caches] == [0.0] * 4  # writes still pending
    park = worker._stage3_send_park
    assert len(park) == 1
    worker.process_send_load(_meta())  # no-forward step: nothing new
    assert seen == [("a", UUID_A, WANT)]
    assert len(park) == 0
    assert device.events[0].waits >= 1
    assert park.counts["waited"] == 1 and park.counts["ready"] == 0
    _mark(
        "GATE",
        registered=len(seen),
        values=",".join(str(v) for v in seen[0][2]),
        waited=park.counts["waited"],
    )


def test_completed_event_registers_without_waiting(env, monkeypatch):  # noqa: F811
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    device.run_to(device.events[0])  # the device ran past the event
    worker.process_send_load(_meta())
    assert seen == [("a", UUID_A, WANT)]
    park = worker._stage3_send_park
    assert park.counts["ready"] == 1 and park.counts["waited"] == 0
    _mark("READY", registered=len(seen), ready=park.counts["ready"])


def test_parked_send_drains_at_the_next_forward_step_before_its_own_work(
    env, monkeypatch  # noqa: F811
):
    """A forward step that brings new sends registers the parked step first
    (order of record), then handles its own."""
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    worker.process_send_load(_meta([("b", UUID_B, True)]))  # ahead: no sync
    assert [s[:2] for s in seen] == [("a", UUID_A), ("b", UUID_B)]
    assert seen[0][2] == WANT
    assert device.trace == [
        ("sync", False),
        ("event", 1),
        ("register", "a"),
        ("register", "b"),
    ]
    worker.process_send_load(_meta([("c", UUID_C, False)]))  # parks again
    assert [s[0] for s in seen] == ["a", "b"]
    worker.process_send_load(_meta())
    assert [s[0] for s in seen] == ["a", "b", "c"]
    _mark("ORDER", order="".join(s[0] for s in seen))


def test_finish_announced_while_parked_is_applied_after_registration(
    env, monkeypatch  # noqa: F811
):
    """finished_req_ids names the request in the same step its send parks
    (both come from the finish); the TTL clock starts at registration."""
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    with patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30.0):
        worker.process_send_load(_meta([("a", UUID_A, False)]))
        worker.get_finished({"a"})
        assert "a" not in worker._stage3_registered_sends
        worker.process_send_load(_meta())
        reg = worker._stage3_registered_sends["a"]
        assert not reg.finish_pending
        assert time.time() < reg.expiration_time < time.time() + 31.0
    assert worker._stage3_send_park.counts["finishes"] == 1
    _mark("FINISH", finish_pending=int(reg.finish_pending), armed=1)


def test_unfinished_parked_send_stays_finish_pending(env, monkeypatch):  # noqa: F811
    """Negative control of the finish row: without finished_req_ids the
    pipeline registration stays finish_pending (as on today's path)."""
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    worker.process_send_load(_meta())
    reg = worker._stage3_registered_sends["a"]
    assert reg.finish_pending and reg.expiration_time == float("inf")
    _mark("FINISHNEG", finish_pending=int(reg.finish_pending))


def test_release_and_fresh_send_keep_their_order_when_parked(
    env, monkeypatch  # noqa: F811
):
    """request_finished minted a fresh send and released the ahead uuid in the
    same metadata: parked together, applied in today's order."""
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, True)]))  # ahead, registered
    worker.process_send_load(_meta([("a", UUID_B, False)], release={"a": UUID_A}))
    assert worker._stage3_registered_sends["a"].uuid == UUID_A
    worker.process_send_load(_meta())
    assert worker._stage3_registered_sends["a"].uuid == UUID_B
    assert [(c["req_id"], c["uuid"]) for c in facade.release_request_blocks_calls] == [
        ("a", UUID_A)
    ]
    _mark("RELEASE", released=len(facade.release_request_blocks_calls))


def test_ahead_only_step_registers_at_once(env, monkeypatch):  # noqa: F811
    worker, facade, device, seen = _setup(env, monkeypatch, key=True)
    worker.process_send_load(_meta([("a", UUID_A, True)]))
    assert device.trace == [("register", "a")]
    assert device.events == []
    _mark("AHEAD", trace=len(device.trace))


def test_no_drafter_never_parks(env, monkeypatch):  # noqa: F811
    worker, facade, device, seen = _setup(env, monkeypatch, key=True, spec=None)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    assert device.trace == [("register", "a")]
    _mark("NOSPEC", trace=len(device.trace))


def test_key_on_without_pipeline_keeps_the_blocking_path(env, monkeypatch):  # noqa: F811
    worker, facade, device, seen = _setup(env, monkeypatch, key=True, pipeline=False)
    worker.process_send_load(_meta([("a", UUID_A, False)]))
    assert device.trace == [("sync", True), ("register", "a")]
    assert worker._stage3_send_park is None
    _mark("NOTPIPE", trace=len(device.trace))


def test_engagement_line_format_matches_the_reader():
    """The kit's eng_literal is the source format string of the ENGAGED line."""
    import inspect

    src = inspect.getsource(stage3_send_park)
    lit = (
        "q38 PSTEP-L2 stage-3 send park ENGAGED: transfer rank %d, sends %d "
        "(VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION)"
    )
    assert src.count(lit) == 1
    _mark("ENGFMT", count=src.count(lit))


# ---------------------------------------------------------------------------
# Unit rows on the FIFO itself.
# ---------------------------------------------------------------------------


class _UnitEvent:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


def test_unit_fifo_waits_and_registers_in_order():
    park = stage3_send_park.Stage3SendParking(3)
    events = [_UnitEvent(), _UnitEvent(done=True), _UnitEvent()]
    for index, event in enumerate(events):
        meta = TPUConnectorMetadata()
        meta.reqs_to_send[f"r{index}"] = None
        park.park(meta, event)
    order = []

    def register(meta):
        (req_id,) = meta.reqs_to_send
        assert events[int(req_id[1:])].done, "registered before its event completed"
        order.append(req_id)

    assert park.drain(register, lambda r: None) == 3
    assert order == ["r0", "r1", "r2"] and len(park) == 0
    assert park.counts["ready"] == 1 and park.counts["waited"] == 2
    assert park.drain(register, lambda r: None) == 0
    _mark("UNITFIFO", order="".join(order), waited=park.counts["waited"])


def test_unit_note_finished_only_for_parked():
    park = stage3_send_park.Stage3SendParking(0)
    meta = TPUConnectorMetadata()
    meta.reqs_to_send["x"] = None
    park.park(meta, _UnitEvent())
    park.note_finished({"x", "y"})
    armed = []
    park.drain(lambda m: None, armed.append)
    assert armed == ["x"]
    park.note_finished({"x"})  # nothing parked: ignored
    assert park.drain(lambda m: None, armed.append) == 0 and armed == ["x"]
    _mark("UNITFIN", armed=",".join(armed))
