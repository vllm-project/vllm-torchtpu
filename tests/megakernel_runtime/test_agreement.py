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
        toks = [mine] if self.peer_tokens is None else [mine, *self.peer_tokens]
        if self.peer_tokens is None:
            toks = [mine] * (1 + len(self.peer_votes))
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


def test_declined_driver_joins_the_bind_agreement(fake_tp_group: Any) -> None:
    from types import SimpleNamespace

    g = fake_tp_group([0])
    dd = agreement_mod.DeclinedDriver(
        EligibilityResult(False, "pallas_mesh_unavailable", "m")
    )
    dd.bind_and_warmup(SimpleNamespace())
    assert g.calls == 1


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
