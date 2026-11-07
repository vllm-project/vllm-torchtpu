# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rank agreement: decline votes, tokens, the agreement group and the
declined driver."""

from __future__ import annotations

from typing import Any

import pytest

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
from vllm_torchtpu.megakernel.runtime.contract import EligibilityResult, TpuMonoSpec
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError


class _FakeTPGroup:
    """Stands in for the TP group's CPU process group: `all_reduce(MAX)` folds
    in the votes of the other ranks, which the test sets."""

    def __init__(self, peer_votes: list[int], peer_tokens: list[int] | None = None):
        self.peer_votes = peer_votes
        self.peer_tokens = peer_tokens
        self.calls = 0

    def all_reduce_max(self, t: Any, group: Any) -> None:
        assert group is self
        self.calls += 1
        mine = int(t[1])
        peers = self.peer_tokens
        toks = [mine] * (1 + len(self.peer_votes)) if peers is None else [mine, *peers]
        t[0] = max([int(t[0]), *self.peer_votes])
        t[1] = max(toks)
        t[2] = max(-x for x in toks)


@pytest.fixture
def fake_tp_group(monkeypatch: pytest.MonkeyPatch) -> Any:
    def make(
        peer_votes: list[int], peer_tokens: list[int] | None = None
    ) -> _FakeTPGroup:
        g = _FakeTPGroup(peer_votes, peer_tokens)
        monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: g)
        monkeypatch.setattr(agreement_mod, "_all_reduce_max", g.all_reduce_max)
        return g

    return make


def _vote(code: str) -> int:
    return agreement_mod.DECLINE_VOTE_CODES.index(code) + 1


def test_agree_on_decline_is_rank_uniform(fake_tp_group: Any) -> None:
    mine = EligibilityResult(False, "pallas_mesh_mismatch", "local")
    # Nobody declines.
    g = fake_tp_group([0, 0, 0])
    assert agreement_mod.agree_on_decline(None) is None and g.calls == 1
    # This rank declines: it keeps its own reason.
    fake_tp_group([0, 0])
    assert agreement_mod.agree_on_decline(mine) is mine
    # A peer declines: this rank declines too, naming the peer's code.
    fake_tp_group([0, _vote("pool_layout_mismatch")])
    res = agreement_mod.agree_on_decline(None)
    assert res is not None and not res.eligible
    assert res.code == "peer_declined:pool_layout_mismatch"
    # Unlisted codes vote as `other`.
    fake_tp_group([0])
    odd = EligibilityResult(False, "something_new", "x")
    assert agreement_mod.agree_on_decline(odd) is odd
    assert agreement_mod._decline_vote(odd) == _vote("other")


def test_agree_on_decline_single_rank_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_reduce(*_a: Any) -> None:
        raise AssertionError("no collective on a single rank")

    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)
    monkeypatch.setattr(agreement_mod, "_all_reduce_max", no_reduce)
    mine = EligibilityResult(False, "pallas_mesh_mismatch", "local")
    assert agreement_mod.agree_on_decline(mine) is mine
    assert agreement_mod.agree_on_decline(None) is None


def test_agree_on_decline_detects_bucket_set_mismatch(fake_tp_group: Any) -> None:
    tok = agreement_mod.buckets_token([(4, 4), (1, 8)])
    assert tok == agreement_mod.buckets_token([(1, 8), (4, 4)])
    assert tok != agreement_mod.buckets_token([(1, 8)])
    fake_tp_group([0], peer_tokens=[tok])
    assert agreement_mod.agree_on_decline(None, token=tok) is None
    fake_tp_group([0], peer_tokens=[tok + 1])
    res = agreement_mod.agree_on_decline(None, token=tok)
    assert res is not None and res.code == "bucket_set_mismatch"


def test_startup_agreement_fingerprint_mismatch_wins_over_votes(
    fake_tp_group: Any,
) -> None:
    """With `mismatch_code` (the startup agreement), differing tokens decline
    every rank with that code, even when a peer also votes a decline."""
    fake_tp_group([_vote("create_error")], peer_tokens=[7])
    res = agreement_mod.agree_on_decline(None, token=8, mismatch_code="plugin_mismatch")
    assert res is not None and res.code == "plugin_mismatch"
    assert "plugin_mismatch" in res.message
    # Equal tokens: the vote decides as usual.
    fake_tp_group([_vote("create_error")], peer_tokens=[8])
    res = agreement_mod.agree_on_decline(None, token=8, mismatch_code="plugin_mismatch")
    assert res is not None and res.code == "peer_declined:create_error"


def test_vote_codes_add_a_plugin_s_codes_before_other(fake_tp_group: Any) -> None:
    codes = agreement_mod.vote_codes(("my_code", "bind_error", "other"))
    base = agreement_mod.DECLINE_VOTE_CODES
    assert codes == (*base[:-1], "my_code", "other")
    # A peer voting the plugin's code: this rank declines naming it.
    fake_tp_group([codes.index("my_code") + 1])
    res = agreement_mod.agree_on_decline(None, codes=codes)
    assert res is not None and res.code == "peer_declined:my_code"
    # Without the plugin's codes the same decline votes as `other`.
    mine = EligibilityResult(False, "my_code", "m")
    assert agreement_mod._decline_vote(mine) == _vote("other")
    assert agreement_mod._decline_vote(mine, codes) == codes.index("my_code") + 1


def test_declined_driver_joins_the_bind_agreement(fake_tp_group: Any) -> None:
    from types import SimpleNamespace

    g = fake_tp_group([0])
    dd = agreement_mod.DeclinedDriver(
        EligibilityResult(False, "pallas_mesh_unavailable", "m")
    )
    dd.bind_and_warmup(SimpleNamespace())
    assert g.calls == 1


def test_declined_driver_counts_stock_steps(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every forward falls through to the stock model and counts as
    `stock_<code>` (the periodic summary names it); a suppressed run (dummy /
    precompile) isn't counted; `reset_stats` clears the counts."""
    import logging

    from vllm_torchtpu.megakernel_runtime import hooks, testing

    try:
        agreement_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            agreement_mod.logger,
            "_logger",
            logging.getLogger(agreement_mod.logger._name),
        )
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    dd = agreement_mod.DeclinedDriver(
        EligibilityResult(False, "pallas_mesh_unavailable", "m"),
        display_name="Toy",
        stats_period_s=3600.0,
    )
    runner = testing.fake_runner()
    with hooks.suppressed(runner):
        assert dd.try_forward(runner, hooks.StepContext(None, None)) == (False, None)
    assert dd.step_mix_tracker.counts == {}
    assert dd.try_forward(runner, hooks.StepContext(None, None)) == (False, None)
    assert dd.step_mix_tracker.counts == {"stock_pallas_mesh_unavailable": 1}
    # A due summary unpacks note_step's tuple and logs it on rank 0.
    dd.step_mix_tracker.period_s = 0.0
    with caplog.at_level(logging.INFO):
        assert dd.try_forward(runner, hooks.StepContext(None, None)) == (False, None)
    assert "Toy megakernel step mix" in caplog.text
    assert "0/2 MK (0.0%)" in caplog.text
    assert "'stock_pallas_mesh_unavailable': 2" in caplog.text
    dd.step_mix_tracker.period_s = 3600.0
    dd.try_forward(runner, hooks.StepContext(None, None))
    dd.reset_stats()
    assert dd.step_mix_tracker.counts == {}


def test_agreement_group_resolves_the_tp_group(monkeypatch: pytest.MonkeyPatch) -> None:
    tp = object()
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: tp)
    assert agreement_mod.agreement_group("tp", None) is tp
    for domain in ("tp_x_dcp", "kernel_mesh"):
        with pytest.raises(MegakernelPluginError, match=domain):
            agreement_mod.agreement_group(domain, None)


def _spec(**kw: Any) -> TpuMonoSpec:
    return TpuMonoSpec("t", "1", create="m:c", **kw)


def test_collective_domain_is_fingerprinted() -> None:
    doc = {"module": "m", "attr": "SPEC", "source_hash": "s", "config": {}}
    tp = fp.plugin_fingerprint(plugin=_spec(), raw_env="", **doc)
    mesh = _spec(collective_domain="kernel_mesh")
    assert fp.plugin_fingerprint(plugin=mesh, raw_env="", **doc) != tp
