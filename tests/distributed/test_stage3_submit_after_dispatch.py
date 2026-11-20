# SPDX-License-Identifier: Apache-2.0
"""Stage-3 consumer loads held back until after the forward dispatch
(TPU_RAIDEN_STAGE3_SUBMIT_AFTER_DISPATCH), on simulated TP ranks.

Each rank is its own TPURaidenConnectorWorker on its own thread. The TP group
is a barrier exchange, so a rank that issues a collective the others do not
breaks the barrier and fails the test instead of passing silently.
"""

import threading
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    TPUConnectorMetadata,
)

from .test_tpu_connector import (
    _MOD,
    _FakeRaidenControllerFacade,
    _FakeRaidenEngine,
    _flush_stage3_submits,
    _make_raiden_worker,
)

from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (  # isort: skip
    _split_held_stage3_loads,
    _Stage3LoadMeta,
)

_SRC = "prefill-controller.test:27000"
_TP = 4


class _BarrierTPGroup:
    """A TP group whose collectives meet at a barrier; logs every collective
    each rank issues, in order."""

    def __init__(self, world_size: int):
        self.world_size = world_size
        self.cpu_group = object()
        self._barrier = threading.Barrier(world_size, timeout=10)
        self._slots: list = [None] * world_size
        self._local = threading.local()
        self.log: list[list[tuple]] = [[] for _ in range(world_size)]

    def bind(self, rank: int) -> None:
        self._local.rank = rank

    def _exchange(self, kind: str, value):
        rank = self._local.rank
        self.log[rank].append((kind, value))
        self._slots[rank] = value
        self._barrier.wait()
        out = list(self._slots)
        self._barrier.wait()
        return out

    def all_gather_object(self, outputs, value, group=None):
        outputs[:] = self._exchange("gather", value)

    def broadcast_object(self, obj=None, src=0):
        return self._exchange("broadcast", obj)[src]


def _shape(log: list[tuple]) -> list[tuple]:
    """A rank's collectives with its vote values; a broadcast's payload is
    rank 0's view of RPCs still in flight, so only its place counts."""
    return [(kind, value) if kind == "gather" else (kind,) for kind, value in log]


def _load(req_id: str, uuid: int, blocks: list[int], **kwargs) -> _Stage3LoadMeta:
    return _Stage3LoadMeta(
        uuid=uuid,
        source_req_id=f"src-{req_id}",
        local_block_ids=blocks,
        num_tokens=2048,
        src_controller_address=_SRC,
        src_job_name="prefill",
        src_engine_id="producer-engine",
        src_data_replica_idx=0,
        src_parallelism=_TP,
        **kwargs,
    )


def _step_metadata() -> TPUConnectorMetadata:
    """Two loads around a release-only and a fail-only entry."""
    meta = TPUConnectorMetadata()
    meta.reqs_to_load["a"] = _load("a", 11, [1, 2])
    meta.reqs_to_load["rel"] = _load(
        "rel", 12, [], release_only=True, report_completion=True
    )
    meta.reqs_to_load["b"] = _load("b", 13, [3])
    meta.reqs_to_load["fail"] = _load("fail", 14, [4, 5], fail_only=True)
    return meta


class _Ranks:
    """``_TP`` consumer workers sharing one barrier TP group."""

    def __init__(self, stack: ExitStack, *, bad_staging: tuple[int, str] | None):
        self.group = _BarrierTPGroup(_TP)
        self.facade = _FakeRaidenControllerFacade()
        stack.enter_context(
            patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden")
        )
        stack.enter_context(patch(f"{_MOD}.get_tp_group", return_value=self.group))
        stack.enter_context(
            patch(
                f"{_MOD}.torch.distributed.all_gather_object",
                side_effect=self.group.all_gather_object,
            )
        )
        self.workers = []
        # Membership of the TP submit set each time get_finished tests it.
        self.tested_sets: list[list[list[str]]] = [[] for _ in range(_TP)]
        for rank in range(_TP):
            worker = _make_raiden_worker(
                is_producer=False, tp_rank=rank, tp_size=_TP, block_size=1024
            )
            assert worker._stage3_sharded_consumer()
            worker._raiden_work_unit = MagicMock()
            engine = _FakeRaidenEngine()
            engine.poll_results = []
            worker._raiden_transfer_engine = engine
            worker._stage3_source_facades[_SRC] = self.facade
            stack.enter_context(
                patch.object(worker, "_stage3_tp_group", return_value=self.group)
            )
            stack.enter_context(
                patch.object(
                    worker,
                    "_require_stage3_controller",
                    return_value=(MagicMock(), "10.0.0.2:28000"),
                )
            )
            units = MagicMock(return_value=[MagicMock()])
            if bad_staging is not None and bad_staging[0] == rank:
                bad_req = bad_staging[1]

                def units(req_meta, _bad=f"src-{bad_req}"):
                    if req_meta.source_req_id == _bad:
                        raise RuntimeError("source facade unreachable on this rank")
                    return [MagicMock()]

            stack.enter_context(
                patch.object(worker, "_stage3_source_work_units", side_effect=units)
            )
            stack.enter_context(
                patch.object(
                    worker,
                    "_raiden_destination_work_units",
                    return_value=[MagicMock() for _ in range(_TP)],
                )
            )
            stack.enter_context(
                patch.object(worker, "_raiden_hbm_memory_type", return_value=3)
            )
            drain = worker._drain_stage3_submit_outcomes

            def recording_drain(_rank=rank, _worker=worker, _drain=drain):
                self.tested_sets[_rank].append(sorted(_worker._stage3_tp_submits))
                _drain()

            worker._drain_stage3_submit_outcomes = recording_drain
            self.workers.append(worker)

    def run(self, step) -> list:
        """Runs ``step(worker)`` on every rank at once; returns the results."""
        results: list = [None] * _TP
        errors: list = []

        def body(rank):
            self.group.bind(rank)
            try:
                results[rank] = step(self.workers[rank])
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                self.group._barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(_TP)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        assert not any(t.is_alive() for t in threads), "a rank hung in a collective"
        if errors:
            raise errors[0]
        return results


def _step(hold: bool):
    """One engine step's connector work on a worker: start_load_kv (or its
    holding form, then the held submit once the forward is dispatched), then
    the poll."""

    def step(worker):
        meta = _step_metadata()
        if hold:
            held = worker.process_send_load(meta, hold_stage3_loads=True)
            calls_before_submit = list(worker._stage3_tp_submits)
            assert held is not None
            assert list(held.reqs_to_load) == ["a", "b"]
            # The forward and the drafter are dispatched here.
            worker.submit_held_stage3_loads(held)
        else:
            worker.process_send_load(meta)
            calls_before_submit = None
        finished = worker.get_finished(set())
        return calls_before_submit, finished

    return step


def _settle(ranks: _Ranks) -> list:
    """Lets rank 0's submit thread finish, then polls once more everywhere so
    the leader's outcomes fan out."""
    _flush_stage3_submits(ranks.workers[0])
    return ranks.run(lambda worker: worker.get_finished(set()))


@pytest.mark.parametrize("hold", [False, True])
def test_held_loads_join_the_submit_set_before_get_finished_tests_it(hold):
    with ExitStack() as stack:
        ranks = _Ranks(stack, bad_staging=None)
        results = ranks.run(_step(hold))
        if hold:
            # Holding runs no vote: the release-only and fail-only entries
            # are handled, the loads are not yet in the set.
            assert all(before == [] for before, _ in results)
        # At the test, every rank holds both loads.
        assert ranks.tested_sets == [[["a", "b"]]] * _TP
        # Same collectives, same order, on every rank: one vote a load, in
        # scheduler order, then the submit-state fan-out.
        assert [_shape(log) for log in ranks.group.log] == [
            [("gather", True), ("gather", True), ("broadcast",)]
        ] * _TP
        # The fail-only entry is reported by every rank in the same poll.
        for _, (sent, recvd) in results:
            assert "fail" in recvd and "rel" in recvd
        _settle(ranks)
        for worker in ranks.workers:
            assert worker._stage3_controller_accepted >= {"a", "b"}
            assert not worker._stage3_inflight_submits
        # Rank 0 alone coordinates, in scheduler order.
        assert [c["req_id"] for c in ranks.facade.start_transfer_calls] == [
            "src-a",
            "src-b",
        ]
        assert ranks.facade.cancel_request_blocks_calls == [
            {"req_id": "src-rel", "uuid": 12},
            {"req_id": "src-fail", "uuid": 14},
        ]


def test_hold_changes_no_outcome_or_order():
    """The held submit leaves the same tables, terminals and controller calls
    as the submit before the forward."""

    def snapshot(ranks: _Ranks, results):
        return [
            (
                sorted(w._stage3_submitted_loads.items()),
                sorted(w._failed_recving),
                sorted(w._stage3_controller_accepted),
                sorted(w._failed_block_ids),
                results[i][1],
            )
            for i, w in enumerate(ranks.workers)
        ], [c["req_id"] for c in ranks.facade.start_transfer_calls]

    outcomes = {}
    for hold in (False, True):
        with ExitStack() as stack:
            ranks = _Ranks(stack, bad_staging=None)
            results = ranks.run(_step(hold))
            _settle(ranks)
            outcomes[hold] = (
                snapshot(ranks, results),
                [_shape(log) for log in ranks.group.log],
            )
    assert outcomes[True] == outcomes[False]


@pytest.mark.parametrize("hold", [False, True])
def test_one_rank_failing_local_staging_fails_the_load_everywhere(hold):
    with ExitStack() as stack:
        ranks = _Ranks(stack, bad_staging=(2, "a"))
        results = ranks.run(_step(hold))
        # Rank 2 voted False for "a": no rank put it in the set or sent it.
        assert ranks.tested_sets == [[["b"]]] * _TP
        assert ranks.group.log[0][0] == ("gather", True)
        assert ranks.group.log[2][0] == ("gather", False)
        for worker, (_, (_, recvd)) in zip(ranks.workers, results):
            assert "a" in worker._failed_recving
            assert {1, 2} <= worker._failed_block_ids
            assert "a" in recvd
        _settle(ranks)
        assert [c["req_id"] for c in ranks.facade.start_transfer_calls] == ["src-b"]


def test_held_loads_of_one_step_precede_the_next_steps_connector_work():
    """Step N's held loads are submitted in step N's poll, before step N+1's
    entries; a load in step N+1 for the same request is a duplicate."""
    with ExitStack() as stack:
        ranks = _Ranks(stack, bad_staging=None)

        def two_steps(worker):
            first = TPUConnectorMetadata(reqs_to_load={"a": _load("a", 11, [1, 2])})
            held = worker.process_send_load(first, hold_stage3_loads=True)
            worker.submit_held_stage3_loads(held)
            worker.get_finished(set())
            second = TPUConnectorMetadata(
                reqs_to_load={"a": _load("a", 11, [1, 2]), "b": _load("b", 13, [3])}
            )
            held = worker.process_send_load(second, hold_stage3_loads=True)
            worker.submit_held_stage3_loads(held)
            worker.get_finished(set())
            return None

        ranks.run(two_steps)
        _settle(ranks)
        assert [c["req_id"] for c in ranks.facade.start_transfer_calls] == [
            "src-a",
            "src-b",
        ]
        # "a" again in step N+1 is a duplicate: no second vote.
        kinds = [kind for kind, _ in ranks.group.log[0]]
        assert kinds.count("gather") == 2


def test_split_keeps_scheduler_order_and_leaves_non_transfers_in_place():
    meta = _step_metadata()
    now, held = _split_held_stage3_loads(meta)
    assert list(now.reqs_to_load) == ["rel", "fail"]
    assert held is not None and list(held.reqs_to_load) == ["a", "b"]
    assert held.reqs_to_load["a"] is meta.reqs_to_load["a"]
    only_release = TPUConnectorMetadata(reqs_to_load={"rel": meta.reqs_to_load["rel"]})
    assert _split_held_stage3_loads(only_release) == (only_release, None)


def test_producer_holds_nothing():
    worker = _make_raiden_worker(is_producer=True, tp_rank=0, tp_size=1)
    worker._raiden_transfer_engine = _FakeRaidenEngine()
    with ExitStack() as stack:
        stack.enter_context(
            patch(f"{_MOD}.tpu_envs.TPU_KV_RESHARD_TRANSPORT", "raiden")
        )
        registers = [
            stack.enter_context(patch.object(worker, name))
            for name in (
                "_register_stage3_request_blocks",
                "_register_stage3_request_row_spans",
                "_register_stage3_request_blocks_dsv4",
            )
        ]
        meta = TPUConnectorMetadata()
        assert worker.process_send_load(meta, hold_stage3_loads=True) is None
        assert sum(r.call_count for r in registers) == 1
