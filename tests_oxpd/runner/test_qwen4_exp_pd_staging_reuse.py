# SPDX-License-Identifier: Apache-2.0
"""A staged decode must never repack one load from staging pages another load
has already landed in (lane_xfer §11.21, M1).

On TPU the repack (and S4's landing gather) is only *enqueued* when
``repack_landed`` returns, and ``release`` used to free its staging pages at
once. The step's admission then handed the lowest free pages, the ones just
freed, to the next load, whose bytes land through tpu_sync's raw
host-to-device copies without waiting for queued device reads. If they landed
before the queued repack ran, that repack copied the NEW load's KV into the
OLD load's blocks: one request answered another's question (INT, int-2 G1 B
arm, j-6f7110fa).

CPU torch runs everything synchronously, so these tests model the device
queue: a repack captures its indices when enqueued and reads staging memory
only when the queue is flushed. Reservation, release, the quarantine and the
worker's repack_landed are the real code.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.qwen4_exp import pd_connector as pc
from vllm_torchtpu.models.vllm.qwen4_exp.pd_staging import RepackPlan, StagedConsumer

ROWS = 4  # rows per staging page / pool block in this model
PAGES = 16
OWNER = "model.layers.3.self_attn"


class DeviceQueue:
    """Enqueued device work, run in order at ``flush`` (the TPU stream)."""

    def __init__(self):
        self.work = []

    def flush(self):
        for fn in self.work:
            fn()
        self.work.clear()


def consumer(queue, pages_per_load=2, shared=False):
    """A real StagedConsumer (reserve / release / quarantine), assembled
    without __init__ as D-MK's admission tests do, with a one-owner layout and
    a repack that runs on the modelled device queue."""
    st = object.__new__(StagedConsumer)
    st.pages = PAGES
    st.free = torch.ones(PAGES, dtype=torch.int32)
    st.free[0] = 0  # the null page
    st.reservations, st.skips = {}, {}
    st.staging = torch.zeros(PAGES, ROWS, dtype=torch.int64)
    st.pool = torch.zeros(64, ROWS, dtype=torch.int64)
    st.shared = object() if shared else None
    st._owner_counts = lambda num_tokens, skip_tokens: ([0], [OWNER], [pages_per_load])

    def plan_repack(uuid, dst_tables, num_tokens):
        # Indices are fixed when the plan is built, as on TPU.
        return RepackPlan(passes=[], chunks=[(list(st.reservations[uuid][OWNER]), list(dst_tables[0]))])

    def run_repack(plan):
        (src, dst), = plan.chunks

        def copy():  # reads staging memory when the device runs it
            st.pool[dst] = st.staging[src]

        queue.work.append(copy)

    st.plan_repack = plan_repack
    st.run_repack = run_repack
    st.repack = lambda uuid, tables, num_tokens: run_repack(plan_repack(uuid, tables, num_tokens))
    return st


def worker(staging):
    w = object.__new__(pc.Qwen4ExpRaidenWorker)
    w._staging = staging
    w._stage3_submitted_load_metas = {}
    w._pending_repacks = {}
    w._sync_repack = True
    w._sync_on = lambda: True
    w._tp_cpu_group = lambda: None
    w.tp_rank = 0
    return w


def admit_and_land(w, req_id, uuid, dst, value):
    """Admission reserves staging pages; the native landing writes the load's
    bytes into them (no device ordering); the worker records the landed load."""
    st = w._staging
    assert st.reserve(uuid, 1024)
    st.staging[st.reservations[uuid][OWNER]] = value
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(uuid=uuid, group_block_ids=[list(dst)], num_tokens=1024)
    w._stage3_finish_destination(req_id, True)  # L1: pending until named
    return list(st.reservations[uuid][OWNER])


def step(w, named=()):
    """One engine step's start, as the connector's start_load_kv runs it."""
    w._staging.advance_quarantine()
    w.repack_landed(list(named))


@pytest.mark.parametrize("shared", [False, True], ids=["routed", "s4-shared"])
def test_a_load_is_never_repacked_from_pages_another_load_landed_in(shared, monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", raising=False)
    q = DeviceQueue()
    w = worker(consumer(q, shared=shared))
    a_pages = admit_and_land(w, "a", 1, dst=[10, 11], value=101)
    step(w, named=["a"])  # A's repack is enqueued and A's pages released...
    b_pages = admit_and_land(w, "b", 2, dst=[20, 21], value=202)  # ...then B is admitted and lands
    q.flush()  # the device finally runs A's repack
    assert not set(a_pages) & set(b_pages), "B was given A's staging pages before A's repack ran"
    assert torch.all(w._staging.pool[[10, 11]] == 101), "A's blocks hold B's KV"


def test_s4_two_shared_loads_in_one_step(monkeypatch):
    """Two shared loads named in one step, a third admitted after both."""
    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", raising=False)
    q = DeviceQueue()
    w = worker(consumer(q, shared=True))
    admit_and_land(w, "a1", 1, dst=[10, 11], value=101)
    admit_and_land(w, "a2", 2, dst=[12, 13], value=102)
    step(w, named=["a1", "a2"])
    admit_and_land(w, "b", 3, dst=[20, 21], value=303)
    q.flush()
    assert torch.all(w._staging.pool[[10, 11]] == 101)
    assert torch.all(w._staging.pool[[12, 13]] == 102)
    assert torch.all(w._staging.pool[[20, 21]] == 0)  # B is not named yet


def test_quarantined_pages_return_after_the_configured_steps(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", "3")
    q = DeviceQueue()
    w = worker(consumer(q))
    a_pages = admit_and_land(w, "a", 1, dst=[10, 11], value=101)
    step(w, named=["a"])  # released in this step
    st = w._staging
    assert st.quarantined_pages() == 2 and not any(int(st.free[p]) for p in a_pages)
    step(w)
    step(w)
    assert not any(int(st.free[p]) for p in a_pages), "freed before 3 step starts"
    step(w)
    assert all(int(st.free[p]) for p in a_pages) and st.quarantined_pages() == 0


def test_zero_steps_frees_at_once(monkeypatch):
    """The escape hatch restores the old behaviour (and its hazard)."""
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", "0")
    q = DeviceQueue()
    w = worker(consumer(q))
    a_pages = admit_and_land(w, "a", 1, dst=[10, 11], value=101)
    step(w, named=["a"])
    assert all(int(w._staging.free[p]) for p in a_pages)


def test_dropped_and_failed_loads_are_quarantined_too(monkeypatch):
    """Unrepacked frees (drop, cleanup) also wait: a dropped load's landing
    may still be in flight on another rank's queue."""
    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", raising=False)
    q = DeviceQueue()
    w = worker(consumer(q))
    a_pages = admit_and_land(w, "a", 1, dst=[10, 11], value=101)
    w._staging.advance_quarantine()
    w.repack_landed([], drop=["a"])
    assert not any(int(w._staging.free[p]) for p in a_pages)


def test_start_load_kv_advances_before_repacking(monkeypatch):
    """The connector counts the quarantine down at every step start, before
    the step's repacks and admission (and with L1 off too)."""
    calls = []
    staging = SimpleNamespace(advance_quarantine=lambda: calls.append("advance"))
    w = SimpleNamespace(_staging=staging, repack_landed=lambda names, drop=(): calls.append("repack"))
    conn = object.__new__(pc.Qwen4ExpRaidenConnector)
    conn.connector_worker = w
    conn._connector_metadata = SimpleNamespace(qwen_repack_req_ids=["a"], qwen_drop_req_ids=[])
    monkeypatch.setattr(pc.Qwen4ExpRaidenConnector.__mro__[1], "start_load_kv", lambda self, *a, **k: calls.append("admit"))
    for sync in (True, False):
        calls.clear()
        conn._sync_repack = sync
        conn.start_load_kv()
        assert calls == (["advance", "repack", "admit"] if sync else ["advance", "admit"])


def test_env_default_and_compile_key():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert envs.environment_variables["QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS"]() == 3
    assert "QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS" in _TPU_COMPILE_ENV_IGNORED


# --- q38 F2 (QWEN4_EXP_PD_DRAFT_KV): a target owner and the MTP draft's per load.
# The draft's pages are entries of the same reservation, so release() must
# quarantine them with the target's and advance_quarantine() return them
# together; a draft page freed at release would let the next load land in it
# before the old load's repack reads it.

DRAFT = "mtp.layers.48.self_attn"


def draft_consumer(queue):
    """``consumer`` whose loads stage a target owner (2 pages) and the MTP
    draft's owner (1 page); one plan repacks both (pd_staging's one indices
    plan per load), run on the modelled device queue."""
    st = consumer(queue)
    st._owner_counts = lambda num_tokens, skip_tokens: ([0], [OWNER, DRAFT], [2, 1])
    run_repack = st.run_repack

    def plan_repack(uuid, dst_tables, num_tokens):
        pages = st.reservations[uuid]
        src = list(pages[OWNER]) + list(pages[DRAFT])
        return RepackPlan(passes=[], chunks=[(src, list(dst_tables[0]) + list(dst_tables[1]))])

    st.plan_repack = plan_repack
    st.repack = lambda uuid, tables, num_tokens: run_repack(plan_repack(uuid, tables, num_tokens))
    return st


def admit_and_land_draft(w, req_id, uuid, dst, draft_dst, value):
    """``admit_and_land`` for both owners: the target's pages hold ``value``,
    the draft's ``value + 50``."""
    st = w._staging
    assert st.reserve(uuid, 1024)
    target, draft = list(st.reservations[uuid][OWNER]), list(st.reservations[uuid][DRAFT])
    st.staging[target] = value
    st.staging[draft] = value + 50
    w._stage3_submitted_load_metas[req_id] = SimpleNamespace(
        uuid=uuid, group_block_ids=[list(dst), list(draft_dst)], num_tokens=1024
    )
    w._stage3_finish_destination(req_id, True)  # L1: pending until named
    return target, draft


def test_draft_pages_are_quarantined_and_returned_with_the_targets(monkeypatch):
    monkeypatch.setenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", "3")
    q = DeviceQueue()
    w = worker(draft_consumer(q))
    st = w._staging
    target, draft = admit_and_land_draft(w, "a", 1, dst=[10, 11], draft_dst=[12], value=101)
    step(w, named=["a"])  # released in this step
    assert st.quarantined_pages() == 3
    assert not any(int(st.free[p]) for p in target + draft), "a page left quarantine at release"
    step(w)
    step(w)
    assert not any(int(st.free[p]) for p in target + draft), "freed before 3 step starts"
    step(w)
    assert all(int(st.free[p]) for p in target + draft) and st.quarantined_pages() == 0


def test_a_load_is_never_repacked_from_draft_pages_another_load_landed_in(monkeypatch):
    monkeypatch.delenv("QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", raising=False)
    q = DeviceQueue()
    w = worker(draft_consumer(q))
    a_target, a_draft = admit_and_land_draft(w, "a", 1, dst=[10, 11], draft_dst=[12], value=101)
    step(w, named=["a"])  # A's repack is enqueued and A's pages released...
    b_target, b_draft = admit_and_land_draft(w, "b", 2, dst=[20, 21], draft_dst=[22], value=202)
    q.flush()  # ...B landed first; the device runs A's repack now
    assert not set(a_target + a_draft) & set(b_target + b_draft), "B was given A's pages"
    assert torch.all(w._staging.pool[[10, 11]] == 101), "A's target blocks hold B's KV"
    assert torch.all(w._staging.pool[[12]] == 151), "A's draft block holds B's draft KV"
