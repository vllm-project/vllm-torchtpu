# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC S3-DEFER-AR on the real TPURaidenConnectorWorker (sharded consumer, TP 2,
one process; the peer always agrees, as in test_stage3_async_connector's rig).

- On vs off: s3d_issue (in start_load_kv) + the deferred _submit_stage3_loads leave the same
  tables, make the same controller calls with the same arguments and report the same loads as
  today's pass, over new / release-only / fail-only / duplicate / parked / expired / staging-failed loads.
- The deferred pass uses fast_reserve (s3a_reserve) and Benliang Wang's all_staged; today's does not.
- The candidates guard: if the loads changed between the issue and the pass, the issued
  collective is still waited, a synchronous prepare decides, the mismatch is counted.
- Never both levers: with s3async configured, s3defer refuses (one ERROR, today's path).
- Sub-mark ~s3d_issue and the S3DEFER engagement line.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm_torchtpu.distributed.kv_transfer import stage3_async as _s3a
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPUConnectorMetadata

from .test_stage3_async_connector import _MOD, _Rig, _flush_stage3_submits, _md, _meta, _script


class _Both:
    """Two patches as one context manager (the rig's all_reduce stub returns a waitable for async_op)."""

    def __init__(self, *ps):
        self.ps = ps

    def __enter__(self):
        for p in self.ps:
            p.__enter__()

    def __exit__(self, *exc):
        for p in reversed(self.ps):
            p.__exit__(*exc)


def _defer_env(rig):
    del rig
    return (
        patch.object(_s3a, "CONFIGURED", False),
        _Both(
            patch.object(_s3a, "DEFER_CONFIGURED", True),
            # The peer always agrees (as the rig's all_reduce stub); async_op returns a work handle.
            patch(
                "torch.distributed.all_reduce",
                side_effect=lambda t, op=None, group=None, async_op=False: (
                    SimpleNamespace(wait=lambda: None) if async_op else None
                ),
            ),
        ),
    )


def _run(defer):
    rig = _Rig(lever=False, no_room={"sP"}, fail_stage={"sF"}, budget=150.0)
    reported = set()
    c1, c2 = _defer_env(rig)
    with rig.env(), c1, c2:
        def step(md=None):
            md = md or TPUConnectorMetadata()
            if defer:
                assert rig.worker.s3d_issue(md)
            rig.worker._submit_stage3_loads(md, rig.engine)
            _flush_stage3_submits(rig.worker)
            return rig.worker.get_finished()[1]
        for md in _script():
            reported |= step(md)
        for _ in range(3):
            reported |= step()
        with patch(f"{_MOD}.time.time", return_value=10**12):
            for _ in range(3):
                reported |= step()
    return rig, reported


def test_defer_on_matches_off():
    off, rep_off = _run(False)
    on, rep_on = _run(True)
    st = on.worker.__dict__["_s3d"]
    assert st.eligible and st.mismatches == 0
    assert on.calls() == off.calls() and len(on.calls()) == 3
    assert on.tables() == off.tables(), (on.tables(), off.tables())
    assert rep_on == rep_off and {"dC", "dD", "dF", "dP"} <= rep_on
    assert on.facade.cancel_request_blocks_if_unclaimed.call_count == off.facade.cancel_request_blocks_if_unclaimed.call_count


def test_deferred_pass_uses_fast_reserve_and_all_staged():
    for defer in (False, True):
        rig = _Rig(lever=False)
        c1, c2 = _defer_env(rig)
        calls = {"staged": 0, "fast": 0}
        real_staged = _s3a_adm_all_staged = None
        from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm

        real_staged = adm.all_staged
        with rig.env(), c1, c2, patch.object(adm, "all_staged", side_effect=lambda *a, **k: (calls.__setitem__("staged", calls["staged"] + 1), real_staged(*a, **k))[1]), \
                patch.object(rig.worker, "s3a_reserve", side_effect=lambda m, rf: (calls.__setitem__("fast", calls["fast"] + 1), True)[1]):
            md = _md(dA=_meta(71, "sA", [1]))
            if defer:
                rig.worker.s3d_issue(md)
            rig.worker._submit_stage3_loads(md, rig.engine)
            _flush_stage3_submits(rig.worker)
        if defer:
            assert calls == {"staged": 1, "fast": 1}, calls
        else:
            assert calls == {"staged": 0, "fast": 0}, calls


def test_candidates_guard_waits_and_decides_synchronously():
    rig = _Rig(lever=False)
    c1, c2 = _defer_env(rig)
    with rig.env(), c1, c2:
        assert rig.worker.s3d_issue(_md(dA=_meta(81, "sA", [1])))
        rig.worker._submit_stage3_loads(_md(dB=_meta(82, "sB", [2])), rig.engine)  # a different pass
        _flush_stage3_submits(rig.worker)
        rig.worker.get_finished()
        st = rig.worker.__dict__["_s3d"]
        assert st.mismatches == 1 and len(st.wait_ms) == 1
        assert rig.worker._stage3_controller_accepted == {"dB"}
        assert "_s3d_issued" not in rig.worker.__dict__ and "_s3d_pass" not in rig.worker.__dict__


def test_never_both_levers():
    rig = _Rig(lever=False)
    with rig.env(), patch.object(_s3a, "DEFER_CONFIGURED", True), patch(f"{_MOD}.logger") as log:
        # _Rig.env configures s3async: s3defer must refuse
        assert rig.worker.s3d_issue(_md(dA=_meta(91, "sA", [1]))) is False
    assert any("stays off" in str(c.args[0]) for c in log.error.call_args_list)


def test_issue_submark_and_stats_line():
    rig = _Rig(lever=False)
    c1, c2 = _defer_env(rig)
    charged = []
    with rig.env(), c1, c2, patch.object(_s3a, "SUBMARKS", True), patch.object(_s3a, "STATS_EVERY", 2), \
            patch(f"{_MOD}._hp.charge", side_effect=lambda name, ns: charged.append(name)), patch(f"{_MOD}.logger") as log:
        for i in range(2):
            md = _md(**{f"d{i}": _meta(100 + i, f"s{i}", [i + 1])})
            rig.worker.s3d_issue(md)
            rig.worker._submit_stage3_loads(md, rig.engine)
            _flush_stage3_submits(rig.worker)
    assert "s3d_issue" in charged and "s3_pass" in charged
    lines = [c.args[0] % c.args[1:] for c in log.info.call_args_list if str(c.args[0]).startswith("S3DEFER rank")]
    assert lines and "passes=2 on=2 share_on=1.000 issued=2" in lines[0], lines
