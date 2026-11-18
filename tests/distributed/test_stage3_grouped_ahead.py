# SPDX-License-Identifier: Apache-2.0
"""Grouped Stage-3 producer registration of sends published ahead of the finish.

A producer that registers ahead (every pipeline producer; a TP / DP producer
under TPU_RAIDEN_STAGE3_REGISTER_AHEAD) publishes a request's send in the step
that completes its prompt. The scheduler later releases that uuid when the
request is preempted, or when its finish mints a fresh send instead of
adopting it. The grouped worker path (every Qwen PD producer, and DSv4) must
then retire the ahead registration like the ungrouped path does.

j-37f0b38b (PP8 prefill, c112, prefill KV pool at 99-100 %) died on
'Conflicting duplicate Stage-3 producer registration': the request was
preempted after its ahead send (uuid 347766198785141), resumed with
num_computed_tokens 0, and re-published ahead (uuid 256075575488569, the same
3009 tokens, new pages) while the grouped workers still held the first
registration. They had dropped the scheduler's release because the grouped
path never read reqs_to_release, and the replace branch could not fire because
the grouped path never set finish_pending.
"""

import time
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import SendMeta
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

from .test_tpu_connector import (
    _MOD,
    _dsv4_worker_plan,
    _FakeRaidenControllerFacade,
    _FakeRaidenEngine,
    _make_raiden_worker,
)

REQ = "chatcmpl-34e4fd96-d9c9-4cbb-87a5-fd42f3004e61-8d0d914b"
UUID_AHEAD = 347766198785141
UUID_FRESH = 256075575488569
DUP = "Conflicting duplicate Stage-3 producer registration"

# The resumed send's ten one-page group tables, as dumped at the crash; the
# first send's pages are unknown (the pool reallocated them on preemption).
QWEN_TABLES_FRESH = [[3710], [5617], [7553], [9106], [5010], [7596], [7044], [7691],
                     [6586], [1467]]
QWEN_TABLES_AHEAD = [[2710], [4617], [6553], [8106], [4010], [6596], [6044], [6691],
                     [5586], [467]]
DSV4_TABLES_AHEAD = [[10], [20], [30, 31, 32, 33], [0] * 7 + [47], [50], [60],
                     [70, 71, 72, 73]]
DSV4_TABLES_FRESH = [[11], [21], [24, 25, 26, 27], [0] * 7 + [46], [52], [62],
                     [64, 65, 66, 67]]
TABLES = {
    "qwen": (QWEN_TABLES_AHEAD, QWEN_TABLES_FRESH, 3009),
    "dsv4": (DSV4_TABLES_AHEAD, DSV4_TABLES_FRESH, 1024),
}


@pytest.fixture
def env():
    with ExitStack() as stack:
        for name, value in {
            "TPU_KV_RESHARD_TRANSPORT": "raiden",
            "TPU_RAIDEN_TRANSFER_PARALLELISM": 1,
            "TPU_RAIDEN_DSV4_ADMISSION": False,
        }.items():
            stack.enter_context(patch(f"{_MOD}.tpu_envs.{name}", value, create=True))
        yield stack


def _qwen_spans(req_meta, num_tokens):
    """Qwen PD's shape: one page per group, every group owned by this stage."""
    groups = tuple(tuple(ids) for ids in req_meta.group_block_ids)
    spans = [
        SimpleNamespace(
            tag=f"qwen4.g{index}", block_ids=ids, spans=(object(),), declared_bytes=4096
        )
        for index, ids in enumerate(groups)
    ]
    return groups, spans


def _worker(kind, env):
    if kind == "dsv4":
        env.enter_context(
            patch(f"{_MOD}.tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER", True, create=True)
        )
        env.enter_context(patch(f"{_MOD}.tpu_envs.TPU_RAIDEN_DSV4_ADMISSION", True))
        worker = _make_raiden_worker(
            tp_rank=0, tp_size=1, is_producer=True, dp_size=8, block_size=1024
        )
        worker._stage3_dsv4_plan, _ = _dsv4_worker_plan(worker)
    else:
        worker = _make_raiden_worker(
            tp_rank=0, tp_size=1, is_producer=True, block_size=3200
        )
        # Qwen4ExpRaidenWorker: grouped unconditionally, its own span lowering.
        worker._uses_stage3_grouped_pages = lambda: True
        worker._stage3_grouped_spans = _qwen_spans
        worker._local_raiden_transfer_rank = MagicMock(return_value=0)
        worker._raiden_transfer_parallelism = MagicMock(return_value=1)
    engine = _FakeRaidenEngine()
    engine.poll_results = []
    facade = _FakeRaidenControllerFacade()
    worker._raiden_transfer_engine = engine
    worker._raiden_controller_facade = facade
    worker._raiden_controller_address = "source:27000"
    worker._raiden_work_unit = SimpleNamespace(data_name="grouped")
    return worker, facade


def _send(uuid, tables, num_tokens, *, ahead):
    return SendMeta(
        uuid=uuid,
        local_block_ids=list(tables[0]),
        expiration_time=time.time() + 300.0,
        num_tokens=num_tokens,
        group_block_ids=[list(ids) for ids in tables],
        registered_ahead=ahead,
    )


def _step(worker, *, send=None, release=None):
    meta = TPUConnectorMetadata()
    if send is not None:
        meta.reqs_to_send[REQ] = send
    if release is not None:
        meta.reqs_to_release[REQ] = release
    worker.process_send_load(meta)


def _registered_uuids(facade):
    return [call["uuid"] for call in facade.register_request_blocks_calls]


def _released(facade):
    return [(call["req_id"], call["uuid"]) for call in facade.release_request_blocks_calls]


def test_qwen_pd_worker_always_takes_the_grouped_path():
    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    assert Qwen4ExpRaidenWorker._uses_stage3_grouped_pages() is True


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_incident_sequence_does_not_raise(kind, env):
    """The three steps of j-37f0b38b's request, end to end. On the unfixed
    grouped path the third step raises the production error."""
    worker, facade = _worker(kind, env)
    ahead, fresh, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    _step(worker, release=UUID_AHEAD)
    _step(worker, send=_send(UUID_FRESH, fresh, num_tokens, ahead=True))
    assert worker._stage3_registered_sends[REQ].uuid == UUID_FRESH
    assert _registered_uuids(facade) == [UUID_AHEAD, UUID_FRESH]
    assert _released(facade) == [(REQ, UUID_AHEAD)]


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_preempted_request_re_published_ahead_registers_its_fresh_send(kind, env):
    """The j-37f0b38b sequence: ahead send, preemption release, resumed ahead send."""
    worker, facade = _worker(kind, env)
    ahead, fresh, num_tokens = TABLES[kind]

    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    assert worker._stage3_registered_sends[REQ].uuid == UUID_AHEAD

    # Preemption step: the scheduler releases the ahead uuid.
    _step(worker, release=UUID_AHEAD)
    assert REQ not in worker._stage3_registered_sends
    assert _released(facade) == [(REQ, UUID_AHEAD)]

    # Resumed from 0 computed tokens: a new ahead send on new pages.
    _step(worker, send=_send(UUID_FRESH, fresh, num_tokens, ahead=True))
    registration = worker._stage3_registered_sends[REQ]
    assert registration.uuid == UUID_FRESH
    assert registration.group_block_ids == tuple(tuple(ids) for ids in fresh)
    assert registration.finish_pending
    assert registration.expiration_time == float("inf")
    assert _registered_uuids(facade) == [UUID_AHEAD, UUID_FRESH]
    assert _released(facade) == [(REQ, UUID_AHEAD)]


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_finish_minted_send_with_its_release_in_the_same_step(kind, env):
    """request_finished minted a fresh send and released the ahead uuid; both
    ride one metadata. The release is applied before the sends."""
    worker, facade = _worker(kind, env)
    ahead, fresh, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    _step(
        worker,
        send=_send(UUID_FRESH, fresh, num_tokens, ahead=False),
        release=UUID_AHEAD,
    )
    registration = worker._stage3_registered_sends[REQ]
    assert registration.uuid == UUID_FRESH
    # A finish-time send of a non-pipeline producer keeps its TTL.
    assert not registration.finish_pending
    assert registration.expiration_time < float("inf")
    assert _registered_uuids(facade) == [UUID_AHEAD, UUID_FRESH]
    assert _released(facade) == [(REQ, UUID_AHEAD)]


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_finish_minted_send_ahead_of_its_release_replaces_the_ahead(kind, env):
    """cf0345411b's case: the fresh send registers before the release arrives.
    The ahead registration is finish_pending, so the replace branch retires it;
    the trailing release then names a uuid no longer registered."""
    worker, facade = _worker(kind, env)
    ahead, fresh, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    assert worker._stage3_registered_sends[REQ].finish_pending
    _step(worker, send=_send(UUID_FRESH, fresh, num_tokens, ahead=False))
    assert worker._stage3_registered_sends[REQ].uuid == UUID_FRESH
    assert _released(facade) == [(REQ, UUID_AHEAD)]
    _step(worker, release=UUID_AHEAD)
    assert worker._stage3_registered_sends[REQ].uuid == UUID_FRESH
    assert _released(facade) == [(REQ, UUID_AHEAD)]
    assert _registered_uuids(facade) == [UUID_AHEAD, UUID_FRESH]


@pytest.mark.parametrize("kind", ["qwen"])
def test_adopted_ahead_registration_is_armed_by_finished_req_ids(kind, env):
    """A finish that adopts the ahead uuid sends nothing new: the registration
    waits (no TTL, no status probe) until finished_req_ids names the request."""
    worker, facade = _worker(kind, env)
    ahead, _, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    with (
        patch(f"{_MOD}.dist_utils.get_stage3_status_probe_s", return_value=1e-9),
        patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=30.0),
    ):
        pending = worker._stage3_registered_sends[REQ]
        assert pending.finish_pending
        assert pending.expiration_time == float("inf")
        worker.get_finished(set())
        worker.get_finished(set())
        assert facade.get_request_block_status_calls == []
        assert facade.cancel_request_blocks_calls == []
        worker.get_finished({REQ})
        armed = worker._stage3_registered_sends[REQ]
        assert not armed.finish_pending
        assert time.time() < armed.expiration_time < time.time() + 31.0
        assert armed.group_block_ids == pending.group_block_ids


def test_pipeline_stage_registers_every_send_finish_pending(env):
    """Every pipeline stage registers ahead of the finish (a TP1 PP prefill,
    the Q38 shape), whatever the send's own flag says."""
    worker, _ = _worker("qwen", env)
    ahead, _, num_tokens = TABLES["qwen"]
    with patch(f"{_MOD}._engine_is_pipeline", return_value=True):
        _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=False))
    registration = worker._stage3_registered_sends[REQ]
    assert registration.finish_pending
    assert registration.expiration_time == float("inf")


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_finish_time_send_keeps_its_ttl(kind, env):
    worker, facade = _worker(kind, env)
    ahead, _, num_tokens = TABLES[kind]
    send = _send(UUID_AHEAD, ahead, num_tokens, ahead=False)
    _step(worker, send=send)
    registration = worker._stage3_registered_sends[REQ]
    assert not registration.finish_pending
    assert registration.expiration_time == send.expiration_time


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_release_of_another_uuid_keeps_the_registration(kind, env):
    worker, facade = _worker(kind, env)
    ahead, _, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=True))
    _step(worker, release=UUID_FRESH)
    assert worker._stage3_registered_sends[REQ].uuid == UUID_AHEAD
    assert _released(facade) == []


@pytest.mark.parametrize("kind", ["qwen", "dsv4"])
def test_conflicting_send_without_release_or_ahead_still_raises(kind, env):
    """Negative control: a second finish-time uuid for a live finish-time
    registration is still a conflict (the fix only retires ahead sends)."""
    worker, facade = _worker(kind, env)
    ahead, fresh, num_tokens = TABLES[kind]
    _step(worker, send=_send(UUID_AHEAD, ahead, num_tokens, ahead=False))
    with pytest.raises(ValueError, match=DUP):
        _step(worker, send=_send(UUID_FRESH, fresh, num_tokens, ahead=False))
    assert _registered_uuids(facade) == [UUID_AHEAD]
