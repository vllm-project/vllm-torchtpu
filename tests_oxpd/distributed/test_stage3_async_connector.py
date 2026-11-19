# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC wiring on the real TPURaidenConnectorWorker (sharded consumer,
TP 2, one process; the coordinator's collectives see a two-rank group whose
peer always agrees, as in test_tpu_connector's sharded-consumer tests).

- ON vs OFF: the same passes (new loads, a release-only, a fail-only, a duplicate
  replay, a parked load, an expired load, a local staging failure) leave the same
  tables and make the same controller calls with byte-identical arguments.
- The apply pass: a finished job changes no scheduler-visible table before
  ``pass + LAG``; at that pass every decision lands.
- TP0 holds the RPC outcome of a load whose job is not applied yet.
- A follower marks the load in flight when its job finishes (native terminals
  park) and joins the TP submit set only at the apply pass; it makes no RPC.
- A pass with the lever off first applies every outstanding job.
- An abort while a load is inside a job resolves (parked -> failed terminal; in
  flight -> the pre-RPC abort check), and no load is reported done before its
  native terminal.
- Sub-marks: today's pass charges ~s3_* only with VLLM_TPU_S3_SUBMARKS.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.distributed.kv_transfer import stage3_async as _s3a
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    TPUConnectorMetadata,
    _Stage3LoadMeta,
)

from .test_tpu_connector import (
    _MOD,
    _FakeRaidenEngine,
    _flush_stage3_submits,
    _make_raiden_worker,
)

ADDR = "prefill-controller.test:27000"


def _meta(uuid, src, blocks, **kw):
    return _Stage3LoadMeta(
        uuid=uuid,
        source_req_id=src,
        local_block_ids=list(blocks),
        num_tokens=2048,
        src_controller_address=ADDR,
        src_job_name="prefill",
        src_engine_id="producer-engine",
        src_data_replica_idx=0,
        src_parallelism=8,
        **kw,
    )


def _md(**loads):
    md = TPUConnectorMetadata()
    for dest, meta in loads.items():
        md.reqs_to_load[dest] = meta
    return md


class _Rig:
    def __init__(self, tp_rank=0, lever=True, states=None, no_room=(), fail_stage=(), budget=150.0):
        self.worker = w = _make_raiden_worker(
            is_producer=False, tp_rank=tp_rank, tp_size=2, block_size=1024
        )
        w._raiden_work_unit = MagicMock()
        w._raiden_transfer_engine = self.engine = _FakeRaidenEngine()
        self.engine.poll_results = []
        self.facade = MagicMock()
        self.facade.start_transfer.return_value = True
        self.facade.cancel_request_blocks_if_unclaimed.return_value = True
        w._stage3_source_facades[ADDR] = self.facade
        self.lever = [lever]
        self.states = states  # follower: what TP0 broadcasts
        self.no_room = set(no_room)
        self.fail_stage = set(fail_stage)
        self.budget = budget
        w._s3a_state().group = SimpleNamespace(world_size=2, cpu_group=object())
        self.group = SimpleNamespace(
            world_size=2,
            cpu_group=object(),
            broadcast_object=(lambda obj, src: obj) if states is None else (lambda obj, src: dict(self.states)),
        )

    @contextlib.contextmanager
    def env(self):
        w = self.worker

        def reserve_reduced(meta, reduced_free):
            return meta.source_req_id not in self.no_room

        def source_units(meta):
            if meta.source_req_id in self.fail_stage:
                raise RuntimeError("staging failed (test)")
            return ["src-unit-0"]

        with (
            patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden", create=True),
            patch(f"{_MOD}.tpu_envs.QWEN4_EXP_PD_BATCHED_ADMISSION", True, create=True),
            patch(f"{_MOD}.tpu_envs.QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES", 0, create=True),
            patch(f"{_MOD}.tpu_envs.QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK", 0, create=True),
            patch(f"{_MOD}.dist_utils.get_p2p_wait_pull_timeout", return_value=2 * self.budget),
            patch(f"{_MOD}.get_tp_group", return_value=self.group),
            patch.object(w, "_stage3_tp_group", return_value=self.group),
            patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, value, group: out.__setitem__(slice(None), [value] * len(out)),
            ),
            patch("torch.distributed.all_reduce", side_effect=lambda t, op=None, group=None: None),
            patch.object(w, "_require_stage3_controller", return_value=(MagicMock(), "10.0.0.2:28000")),
            patch.object(w, "_stage3_source_work_units", side_effect=source_units),
            patch.object(w, "_raiden_destination_work_units", return_value=["dst-unit-0", "dst-unit-1"]),
            patch.object(w, "_raiden_hbm_memory_type", return_value=3),
            patch.object(w, "_stage3_reserve_destination_reduced", side_effect=reserve_reduced),
            patch.object(_s3a, "lever_on", side_effect=lambda: self.lever[0]),
            patch.object(_s3a, "CONFIGURED", True),
            patch.object(_s3a, "LAG", 2),
        ):
            yield

    def settle(self):
        """Every outstanding job finished and its RPCs posted (deterministic tests)."""
        for job in list(self.worker._s3a_state().jobs):
            assert job.done.wait(10), "coordinator job did not finish"
            assert job.error is None, job.error
        _flush_stage3_submits(self.worker)

    def step(self, md=None, finished=None):
        self.worker._submit_stage3_loads(md or TPUConnectorMetadata(), self.engine)
        self.settle()
        return self.worker.get_finished(finished)

    def calls(self):
        return sorted(
            (tuple(sorted((k, repr(v)) for k, v in c.kwargs.items())) for c in self.facade.start_transfer.call_args_list)
        )

    def tables(self):
        w = self.worker
        return dict(
            submitted=dict(w._stage3_submitted_loads),
            accepted=set(w._stage3_controller_accepted),
            tp_submits=set(w._stage3_tp_submits),
            inflight=set(w._stage3_inflight_submits),
            failed=set(w._failed_recving),
            terminal=set(w._stage3_terminal_loads),
            waits=sorted(w._stage3_destination_waits),
            blocks={k: list(v) for k, v in w._load_block_ids.items()},
            pending_cleanup=set(w._stage3_finished_loads_pending_cleanup),
        )


def _script():
    return [
        _md(
            dA=_meta(11, "sA", [1, 2]),
            dB=_meta(12, "sB", [3]),
            dC=_meta(13, "sC", [4], release_only=True, report_completion=True),
            dD=_meta(14, "sD", [5], fail_only=True),
            dP=_meta(15, "sP", [6]),  # no destination room: parks, then expires
            dF=_meta(16, "sF", [7]),  # local staging fails
        ),
        _md(dA=_meta(11, "sA", [1, 2]), dE=_meta(17, "sE", [8, 9])),  # dA: duplicate replay
    ]


def _run(lever):
    rig = _Rig(lever=lever, no_room={"sP"}, fail_stage={"sF"}, budget=150.0)
    reported = set()
    with rig.env():
        for md in _script():
            _, rec = rig.step(md)
            reported |= rec
        for _ in range(4):
            _, rec = rig.step()
            reported |= rec
        # The parked load's wait budget runs out: both paths expire it.
        with patch(f"{_MOD}.time.time", return_value=10**12):
            for _ in range(6):
                _, rec = rig.step()
                reported |= rec
    return rig, reported


def test_on_matches_off_tables_and_controller_calls():
    off, rep_off = _run(False)
    on, rep_on = _run(True)
    assert on.worker._s3a_state().stats.jobs >= 2, "the ON run never used the coordinator"
    assert off.worker._s3a_state().stats.jobs == 0
    assert on.calls() == off.calls() and len(on.calls()) == 3  # dA, dB, dE
    t_on, t_off = on.tables(), off.tables()
    assert t_on == t_off, (t_on, t_off)
    assert rep_on == rep_off, (rep_on, rep_off)
    # Release-only and fail-only resolved; the staging failure and the expiry failed.
    assert {"dC", "dD", "dF", "dP"} <= rep_on
    assert on.facade.cancel_request_blocks_if_unclaimed.call_count == off.facade.cancel_request_blocks_if_unclaimed.call_count


def test_decisions_land_only_at_the_apply_pass():
    rig = _Rig(lever=True)
    with rig.env():
        rig.worker._submit_stage3_loads(_md(dA=_meta(21, "sA", [1])), rig.engine)  # pass 1 -> applies at 3
        rig.settle()
        st = rig.worker._s3a_state()
        assert st.jobs and st.jobs[0].done.is_set() and st.jobs[0].apply_pass == 3
        # Finished, RPC answered, but not applied: no TP submit, no acceptance yet.
        assert rig.worker.get_finished() == (set(), set())
        assert rig.worker._stage3_tp_submits == set()
        assert rig.worker._stage3_controller_accepted == set()
        assert len(rig.worker._stage3_submit_outcomes) == 1  # held by TP0
        rig.step()  # pass 2: still not due
        assert rig.worker._stage3_tp_submits == set() and rig.worker._stage3_controller_accepted == set()
        with patch(f"{_MOD}.logger") as log:
            rig.worker._submit_stage3_loads(TPUConnectorMetadata(), rig.engine)  # pass 3: applied
        lines = [c.args[0] % c.args[1:] for c in log.info.call_args_list if "S3ASYNC_JOB" in str(c.args[0])]
        assert len(lines) == 1 and "seq=1 pass=3" in lines[0] and "kinds=dispatched" in lines[0], lines
        assert rig.worker._stage3_tp_submits == {"dA"}
        assert rig.worker.get_finished() == (set(), set())  # accepted, not landed
        assert rig.worker._stage3_controller_accepted == {"dA"}
        assert rig.worker._stage3_tp_submits == set()  # the submit-state broadcast resolved it
        # No load is reported done before its native terminal.
        rig.engine.poll_results = [([], ["sA"], [])]
        assert rig.worker.get_finished() == (set(), {"dA"})


def test_follower_parks_terminals_and_joins_tp_submits_at_the_apply_pass():
    rig = _Rig(tp_rank=1, lever=True, states={})
    with rig.env():
        rig.worker._submit_stage3_loads(_md(dA=_meta(31, "sA", [1])), rig.engine)
        rig.settle()
        assert rig.worker._stage3_inflight_submits == {"dA"}  # terminals park from the job on
        assert rig.worker._stage3_tp_submits == set()
        rig.engine.poll_results = [([], ["sA"], [])]  # an early native terminal
        assert rig.worker.get_finished() == (set(), set())  # parked, not dropped
        rig.step()
        rig.worker._submit_stage3_loads(TPUConnectorMetadata(), rig.engine)  # pass 3
        assert rig.worker._stage3_tp_submits == {"dA"}
        rig.states = {"dA": "accepted"}
        assert rig.worker.get_finished() == (set(), {"dA"})  # the parked terminal replays
        assert rig.facade.start_transfer.call_count == 0  # only TP0 calls the controller


def test_lever_off_applies_outstanding_jobs_first():
    rig = _Rig(lever=True)
    with rig.env():
        rig.worker._submit_stage3_loads(_md(dA=_meta(41, "sA", [1])), rig.engine)
        rig.settle()
        rig.lever[0] = False
        rig.worker._submit_stage3_loads(_md(dB=_meta(42, "sB", [2])), rig.engine)
        st = rig.worker._s3a_state()
        assert st.jobs == [] and st.stats.drains == 1
        assert rig.worker._stage3_tp_submits == {"dA", "dB"}
        _flush_stage3_submits(rig.worker)
        rig.worker.get_finished()
        assert rig.worker._stage3_controller_accepted == {"dA", "dB"}


def test_abort_inside_a_job_resolves():
    rig = _Rig(lever=True, no_room={"sP"})
    with rig.env():
        rig.worker._submit_stage3_loads(_md(dP=_meta(51, "sP", [5]), dQ=_meta(52, "sQ", [6])), rig.engine)
        rig.settle()
        # Both aborted while their job is outstanding.
        assert rig.worker.get_finished({"dP", "dQ"}) == (set(), set())
        assert rig.worker._stage3_finished_loads_pending_cleanup == {"dP", "dQ"}
        rig.step()
        rig.worker._submit_stage3_loads(TPUConnectorMetadata(), rig.engine)  # pass 3: applied
        _flush_stage3_submits(rig.worker)
        # dP (parked when aborted) gets a failed terminal; dQ's RPC had already
        # run, so it resolves through the native terminal path like today.
        _, rec = rig.worker.get_finished()
        assert "dP" in rec
        rig.engine.poll_results = [([], [], ["sQ"])]
        _, rec2 = rig.worker.get_finished()
        assert "dQ" in rec | rec2
        _, _ = rig.worker.get_finished()
        assert rig.worker._stage3_finished_loads_pending_cleanup == set()
        assert rig.worker._stage3_destination_waits == {}


@pytest.mark.parametrize("marks", [False, True])
def test_submarks_only_with_the_flag(marks):
    rig = _Rig(lever=False)
    charged = []
    with rig.env(), patch.object(_s3a, "SUBMARKS", marks), patch(
        f"{_MOD}._hp.charge", side_effect=lambda name, ns: charged.append(name)
    ):
        rig.step(_md(dA=_meta(61, "sA", [1])))
        _flush_stage3_submits(rig.worker)
        rig.worker.get_finished()
    names = set(charged)
    if marks:
        assert {"s3_pass", "s3_adm", "s3_res", "s3_stage", "s3_ag", "s3_disp", "s3_after", "s3_sync"} <= names, names
    else:
        assert not names
    # On passes charge ~s3a_pass (and never today's ~s3_* pass parts).
    rig2 = _Rig(lever=True)
    charged.clear()
    with rig2.env(), patch.object(_s3a, "SUBMARKS", marks), patch(
        f"{_MOD}._hp.charge", side_effect=lambda name, ns: charged.append(name)
    ):
        rig2.step(_md(dA=_meta(62, "sA", [1])))
    names = set(charged)
    if marks:
        assert "s3a_pass" in names and not names & {"s3_pass", "s3_adm", "s3_res", "s3_ag"}, names
    else:
        assert not names


def test_qwen_coordinator_reserve_matches_the_reduced_reserve():
    """Qwen4ExpRaidenWorker.s3a_reserve (the coordinator's) against
    _stage3_reserve_destination_reduced (today's batched pass) on twin stagings:
    the same arguments reach the reservation (uuid, tokens, prefix skip, F2's
    draft-zero prefix), so every table ends equal."""
    import random as _random

    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp.pd_connector import Qwen4ExpRaidenWorker

    from . import test_stage3_batched_admission as tba

    for seed in range(40):
        g = _random.Random(seed)
        free = torch.ones(96, dtype=torch.int32)
        free[0] = 0
        for p in g.sample(range(1, 96), g.randint(0, 40)):
            free[p] = 0
        a = SimpleNamespace(_staging=tba._staging(free))
        b = SimpleNamespace(_staging=tba._staging(free))
        rf_a, rf_b = free.clone(), free.clone()
        for i in range(g.randint(1, 6)):
            tokens = g.choice([16, 64, 160, 320])
            meta = SimpleNamespace(
                uuid=seed * 10 + i,
                num_tokens=tokens,
                skip_tokens=g.choice([0, 16]) if tokens > 16 else 0,
                src_cached_tokens=g.choice([0, 0, 32]),
            )
            ra = Qwen4ExpRaidenWorker._stage3_reserve_destination_reduced(a, meta, rf_a)
            rb = Qwen4ExpRaidenWorker.s3a_reserve(b, meta, rf_b)
            assert ra == rb
            sa, sb = a._staging, b._staging
            assert torch.equal(sa.free, sb.free) and torch.equal(rf_a, rf_b)
            assert sa.reservations == sb.reservations and sa.skips == sb.skips
            assert sa.__dict__.get("tokens", {}) == sb.__dict__.get("tokens", {})
            assert sa.__dict__.get("draft_zero", {}) == sb.__dict__.get("draft_zero", {})
    assert Qwen4ExpRaidenWorker.s3a_reserve(SimpleNamespace(_staging=None), None, None) is True
