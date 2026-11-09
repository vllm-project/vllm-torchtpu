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
"""`ProgramDriver` with a toy plugin that serves: startup order, the startup
and bind agreements, declines before `create`, warmup shapes, routing, the
weight-update guard and a 2-rank gloo run."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import json
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel.adapter.agreement as agreement_mod
import vllm_torchtpu.megakernel.adapter.driver as driver_mod
from tests.megakernel.adapter import toy_plugin
from tests.megakernel.adapter.toy_plugin import TOY_PLUGIN
from vllm_torchtpu.megakernel import adapter as mkr
from vllm_torchtpu.megakernel.adapter import fingerprint as fp
from vllm_torchtpu.megakernel.adapter import stepping, testing
from vllm_torchtpu.megakernel.adapter.hooks import StepContext
from vllm_torchtpu.megakernel.adapter.plugin import RunnerView
from vllm_torchtpu.megakernel.core import errors as mk_errors
from vllm_torchtpu.megakernel.core.runtime.contract import (
    Bucket,
    RunnerShape,
    TpuMonoSpec,
)

_SRC = Path(driver_mod.__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _std_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


mk_toy_plugin = toy_plugin.mk_toy_plugin


@pytest.fixture
def toy(mk_toy_plugin: Path) -> Path:
    """The toy plugin as a single-file module `toy_mk_plugin` on `sys.path`."""
    return mk_toy_plugin


def _no_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)


def _runner(options: dict[str, Any] | None = None, **kw: Any) -> Any:
    mk = {"plugin": "toy_mk_plugin", "options": dict(options or {})}
    return testing.fake_runner(megakernel=mk, **kw)


def _fwd_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(stepping, "get_forward_context", lambda: ctx)


def test_toy_plugin_serves_through_try_forward(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_group(monkeypatch)
    _fwd_ctx(monkeypatch)
    runner = _runner({"scale": 3.0})
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    assert isinstance(drv, driver_mod.ProgramDriver) and drv.serving
    prog = drv.program
    assert prog.events == ["load"] and prog.mesh is not None
    runner._megakernel = drv
    drv.bind_and_warmup(runner)
    assert prog.events == ["load", "bind", ("buckets", (16, 32, 64))]
    # The program hook runs from the worker's on_warmup_done, once per bind.
    drv.on_warmup_done(runner)
    drv.on_warmup_done(runner)
    assert prog.events[-1] == "warm" and prog.events.count("warm") == 1
    # Runtime-enumerated runner shapes: first padding >= B, x every
    # (req, blk) pair; ckpt_cols = max(T, ckpt_window).
    assert [(k, s) for k, s, _ in prog.warmed] == [
        ((1, 4), RunnerShape(16, 8, 4, 1)),
        ((1, 8), RunnerShape(16, 8, 4, 1)),
    ]
    assert [v for _, _, v in prog.warmed] == [3.0 + 4, 3.0 + 8]
    assert drv._buckets == [Bucket(1, 4, 16), Bucket(1, 8, 8)]
    # Each bucket has its own code object.
    names = {fn.__code__.co_name for fn in drv._steps.values()}
    assert names == {"step_t1_b4", "step_t1_b8"}

    ids = torch.arange(3, dtype=torch.int32)
    hit, out = drv.try_forward(runner, StepContext(ids, torch.zeros(3)))
    assert hit and torch.equal(out, ids.float() * 3.0 + 4)
    big = torch.arange(9, dtype=torch.int32)
    assert drv.try_forward(runner, StepContext(big, torch.zeros(9))) == (False, None)
    with mkr.suppressed(runner):
        assert drv.try_forward(runner, StepContext(ids, torch.zeros(3))) == (
            False,
            None,
        )
    assert drv.step_mix_tracker.counts == {"mk_t1_b4": 1, "stock_big": 1}


class _StockModel:
    """A stand-in for `runner.model`: records its calls, returns `-input_ids`
    wrapped in a one-element list (as `@support_torch_compile` may)."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, *, input_ids: Any, positions: Any, inputs_embeds: Any) -> Any:
        del positions, inputs_embeds
        self.calls.append(int(input_ids.shape[0]))
        return [-input_ids.float()]


def test_ab_calibration_alternates_and_times_the_stock_arm(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `ab` on, an eligible step alternates between the megakernel and
    the stock model (both timed, both `hit=True`); an ineligible step is timed
    on the stock model; a step the program doesn't classify stays with the
    runner."""
    _no_group(monkeypatch)
    _fwd_ctx(monkeypatch)
    runner = testing.fake_runner(megakernel={"plugin": "toy_mk_plugin", "ab": True})
    stock = _StockModel()
    runner.model = stock
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    assert isinstance(drv, driver_mod.ProgramDriver) and drv.ab_calibrator.enabled
    runner._megakernel = drv
    drv.bind_and_warmup(runner)

    ids = torch.arange(3, dtype=torch.int32)
    outs = []
    for _ in range(4):
        hit, out = drv.try_forward(runner, StepContext(ids, torch.zeros(3)))
        assert hit
        # A stock-arm step leaves no megakernel step pending.
        assert drv._pending is None
        outs.append(out)
    mk_out, stock_out = ids.float() * 2.0 + 4, -ids.float()
    for out, want in zip(outs, (mk_out, stock_out, mk_out, stock_out), strict=True):
        assert torch.equal(out, want)
    big = torch.arange(9, dtype=torch.int32)
    hit, out = drv.try_forward(runner, StepContext(big, torch.zeros(9)))
    assert hit and torch.equal(out, -big.float())
    assert stock.calls == [3, 3, 9]
    assert drv.step_mix_tracker.counts == {
        "mk_t1_b4_ab_mk": 2,
        "mk_t1_b4_ab_stock": 2,
        "stock_big": 1,
    }
    cal = drv.ab_calibrator
    cum = json.loads(cal.format_summary(cal._cumulative))
    assert (cum["mk_t1_b4_p3"]["mk"][0], cum["mk_t1_b4_p3"]["stock"][0]) == (2, 2)
    assert "d_mean_ms" in cum["mk_t1_b4_p3"]
    assert cum["stock_big_p9"]["stock"][0] == 1 and "mk" not in cum["stock_big_p9"]

    # Not classified by the program (`decide` returns None), or not a
    # megakernel step at all (`inputs_embeds`): the runner's own forward.
    monkeypatch.setattr(drv.program, "decide", lambda ctx, fwd: None)
    assert drv.try_forward(runner, StepContext(ids, torch.zeros(3))) == (False, None)
    embeds = StepContext(None, torch.zeros(3), inputs_embeds=torch.zeros(3, 4))
    assert drv.try_forward(runner, embeds) == (False, None)
    assert stock.calls == [3, 3, 9]


def test_toy_fingerprint_changes_with_options_and_source(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_group(monkeypatch)
    f1 = driver_mod.ProgramDriver.maybe_create(_runner({"scale": 2.0})).fingerprint
    assert (
        f1 == driver_mod.ProgramDriver.maybe_create(_runner({"scale": 2.0})).fingerprint
    )
    f2 = driver_mod.ProgramDriver.maybe_create(_runner({"scale": 5.0})).fingerprint
    assert f2 != f1
    toy.write_text(TOY_PLUGIN + "\n# edited\n", encoding="utf-8")
    fp.clear_caches()
    f3 = driver_mod.ProgramDriver.maybe_create(_runner({"scale": 2.0})).fingerprint
    assert f3 != f1


def test_toy_create_decline_gives_declined_driver(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_group(monkeypatch)
    drv = driver_mod.ProgramDriver.maybe_create(_runner({"decline": True}))
    assert isinstance(drv, driver_mod.DeclinedDriver) and not drv.serving
    assert drv.decline.code == "toy_declined"
    assert "toy_declined" in drv._codes and drv._codes[-1] == "other"


class _VoteGroup:
    def __init__(self, peer_votes: list[int], peer_token: int | None = None) -> None:
        self.peer_votes, self.peer_token = peer_votes, peer_token
        self.seen: list[tuple[int, int]] = []

    def all_reduce_max(self, t: Any, group: Any) -> None:
        assert group is self
        self.seen.append((int(t[0]), int(t[1])))
        toks = [int(t[1]), int(t[1]) if self.peer_token is None else self.peer_token]
        t[0] = max([int(t[0]), *self.peer_votes])
        t[1], t[2] = max(toks), max(-x for x in toks)


def _vote_group(monkeypatch: pytest.MonkeyPatch, *args: Any) -> _VoteGroup:
    g = _VoteGroup(*args)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: g)
    monkeypatch.setattr(agreement_mod, "_all_reduce_max", g.all_reduce_max)
    return g


def test_toy_create_error_votes_then_raises(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = _vote_group(monkeypatch, [0])
    with pytest.raises(RuntimeError, match="toy create bug"):
        driver_mod.ProgramDriver.maybe_create(_runner({"raise_rank": 0}))
    codes = driver_mod.vote_codes(("toy_declined",))
    assert len(g.seen) == 1 and g.seen[0][0] == codes.index("create_error") + 1
    assert g.seen[0][1] > 0  # the fingerprint token


def test_agreement_one_peer_decline_and_plugin_mismatch(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codes = driver_mod.vote_codes(("toy_declined",))
    _vote_group(monkeypatch, [codes.index("toy_declined") + 1])
    drv = driver_mod.ProgramDriver.maybe_create(_runner())
    assert isinstance(drv, driver_mod.DeclinedDriver)
    assert drv.decline.code == "peer_declined:toy_declined"
    _vote_group(monkeypatch, [0], 12345)
    drv = driver_mod.ProgramDriver.maybe_create(_runner())
    assert drv.decline.code == "plugin_mismatch"


def test_pre_agreement_failures_vote_before_raising(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rank-local failure from the plugin import to the mesh joins
    the startup agreement as create_error (token 0 before the fingerprint exists), so
    the peers decline instead of waiting."""
    from vllm_torchtpu.megakernel.adapter.errors import MegakernelPluginError

    codes = driver_mod.vote_codes(("toy_declined",))
    vote = codes.index("create_error") + 1

    def unhashable(*_a: Any, **_k: Any) -> Any:
        raise MegakernelPluginError("plugin_source_unhashable:zip: nope")

    g = _vote_group(monkeypatch, [0])
    with monkeypatch.context() as m:
        m.setattr(driver_mod, "source_hash", unhashable)
        with pytest.raises(MegakernelPluginError, match="unhashable"):
            driver_mod.ProgramDriver.maybe_create(_runner())
    assert g.seen == [(vote, 0)]

    def mesh_boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("device enumeration failed")

    g = _vote_group(monkeypatch, [0])
    with monkeypatch.context() as m:
        m.setattr(driver_mod, "_build_tp_rank_mesh", mesh_boom)
        with pytest.raises(RuntimeError, match="device enumeration"):
            driver_mod.ProgramDriver.maybe_create(_runner())
    assert len(g.seen) == 1 and g.seen[0][0] == vote and g.seen[0][1] > 0

    g = _vote_group(monkeypatch, [0])
    with pytest.raises(Exception, match="failed to import"):
        driver_mod.ProgramDriver.maybe_create(
            testing.fake_runner(megakernel={"plugin": "no_such_mk_plugin_xyz"})
        )
    base_vote = driver_mod.vote_codes().index("create_error") + 1
    assert g.seen == [(base_vote, 0)] and base_vote == vote

    # A peer that failed before its fingerprint: plugin_mismatch here.
    _vote_group(monkeypatch, [vote], 0)
    drv = driver_mod.ProgramDriver.maybe_create(_runner())
    assert drv.decline.code == "plugin_mismatch"


def test_create_result_must_be_a_program_or_decline(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_torchtpu.megakernel.adapter.errors import MegakernelPluginError

    toy_mod = importlib.import_module("toy_mk_plugin")
    monkeypatch.setattr(toy_mod, "create", lambda view, options: object())
    monkeypatch.setattr(
        toy_mod,
        "SPEC",
        dataclasses.replace(toy_mod.SPEC, create=toy_mod.create),
    )
    g = _vote_group(monkeypatch, [0])
    with pytest.raises(MegakernelPluginError, match="neither a Decline nor"):
        driver_mod.ProgramDriver.maybe_create(_runner())
    codes = driver_mod.vote_codes(("toy_declined",))
    assert [v for v, _ in g.seen] == [codes.index("create_error") + 1]


def test_create_runs_no_collective(toy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("create ran a collective")

    for name in (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "broadcast",
        "barrier",
        "reduce_scatter",
        "all_to_all",
    ):
        monkeypatch.setattr(torch.distributed, name, boom)
    toy_mod = importlib.import_module("toy_mk_plugin")
    view = RunnerView(_runner(), tp_rank=0)
    prog = toy_mod.SPEC.create(view, {"scale": 1.0})
    assert prog.mesh == "unloaded"


def test_bind_agreement_bucket_mismatch_installs_declined_driver(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_group(monkeypatch)
    runner = _runner()
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    runner._megakernel = drv
    g = _vote_group(monkeypatch, [0], 7)
    drv.bind_and_warmup(runner)
    assert len(g.seen) == 1 and drv.program.warmed == []
    assert isinstance(runner._megakernel, driver_mod.DeclinedDriver)
    assert runner._megakernel.decline.code == "bucket_set_mismatch"
    # The bind token is the bucket set salted with the fingerprint.
    keys = [(1, 4), (1, 8)]
    assert g.seen[0][1] == driver_mod.buckets_token(keys, salt=drv.fingerprint)
    assert g.seen[0][1] != driver_mod.buckets_token(keys)


def _spec(**kw: Any) -> TpuMonoSpec:
    return TpuMonoSpec("t", "1", create="m:c", **kw)


def _mm_cfg(lm_only: bool = False, limits: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(language_model_only=lm_only, limit_per_prompt=limits or {})


def test_text_only_server_serves_from_token_ids(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `supports_mm_inputs=True` the runner feeds the stock model
    `inputs_embeds`; on a text-only server the megakernel serves the step
    from `ctx.token_ids`, and without them (multimodal embeddings in the
    step) the step stays on the stock model."""
    _no_group(monkeypatch)
    _fwd_ctx(monkeypatch)
    runner = _runner(
        {"scale": 3.0},
        supports_mm_inputs=True,
        multimodal_config=_mm_cfg(lm_only=True),
    )
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    assert isinstance(drv, driver_mod.ProgramDriver) and drv._text_embeds
    runner._megakernel = drv
    drv.bind_and_warmup(runner)
    ids = torch.arange(3, dtype=torch.int32)
    emb = torch.zeros(3, 4)
    pos = torch.zeros(3)
    hit, out = drv.try_forward(runner, StepContext(None, pos, emb, token_ids=ids))
    assert hit and torch.equal(out, ids.float() * 3.0 + 4)
    assert drv.try_forward(runner, StepContext(None, pos, emb)) == (False, None)
    assert drv.step_mix_tracker.counts == {"mk_t1_b4": 1}
    # A driver for a server that may get multimodal inputs never swaps.
    drv._text_embeds = False
    ctx = StepContext(None, pos, emb, token_ids=ids)
    assert drv.try_forward(runner, ctx) == (False, None)


def test_unknown_requires_key_fails_at_spec_load(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = toy.read_text("utf-8").replace(
        'decline_codes=("toy_declined",)',
        'decline_codes=("toy_declined",), requires={"no_such_key": "1"}',
    )
    toy.write_text(src, encoding="utf-8")
    _no_group(monkeypatch)
    with pytest.raises(mk_errors.MegakernelConfigError, match="no_such_key"):
        driver_mod.ProgramDriver.maybe_create(_runner())


def _forbid_create_and_collectives(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_collective(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("must decline before create / any collective")

    monkeypatch.setattr(driver_mod, "agree_on_decline", no_collective)
    monkeypatch.setattr(driver_mod, "_build_tp_rank_mesh", no_collective)
    monkeypatch.setattr(driver_mod, "resolve_callable", no_collective)
    monkeypatch.setattr(torch.distributed, "all_reduce", no_collective)


def test_undeclared_feature_declines_before_create(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_create_and_collectives(monkeypatch)
    assert driver_mod.ProgramDriver.maybe_create(_runner(speculative_tokens=3)) is None


def test_requires_unmet_declines_before_create(
    toy: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    src = toy.read_text("utf-8").replace(
        'decline_codes=("toy_declined",)',
        'decline_codes=("toy_declined",), '
        'requires={"mamba_ssm_cache_dtype": "bfloat16"}',
    )
    toy.write_text(src, encoding="utf-8")
    _no_group(monkeypatch)
    with monkeypatch.context() as m:
        _forbid_create_and_collectives(m)
        with caplog.at_level(logging.WARNING):
            assert driver_mod.ProgramDriver.maybe_create(_runner()) is None
    assert "--mamba-ssm-cache-dtype bfloat16" in caplog.text
    drv = driver_mod.ProgramDriver.maybe_create(
        _runner(mamba_ssm_cache_dtype="bfloat16")
    )
    assert isinstance(drv, driver_mod.ProgramDriver)


def test_numerics_is_logged_and_fingerprinted(
    toy: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _no_group(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert driver_mod.ProgramDriver.maybe_create(_runner()) is not None
    assert "numerics=exact" in caplog.text
    doc = {"module": "m", "attr": "SPEC", "source_hash": "s", "config": {}}
    exact = fp.plugin_fingerprint(plugin=_spec(), raw_env="", **doc)
    approx = fp.plugin_fingerprint(plugin=_spec(numerics="approx"), raw_env="", **doc)
    assert exact != approx


def test_loaded_line_says_whose_numerics_it_reports(
    toy: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A spec's `numerics` is its default profile's; a requested profile's own
    numerics are only known to the program, which logs them at bind."""
    note = driver_mod.loaded_numerics_note
    assert note("approx", {}) == "spec numerics=approx (default profile)"
    assert note("approx", {"profile": "p-exact"}) == (
        "spec numerics=approx (default profile); requested profile=p-exact, "
        "its numerics are logged at bind"
    )
    _no_group(monkeypatch)
    with caplog.at_level(logging.INFO):
        drv = driver_mod.ProgramDriver.maybe_create(_runner({"profile": "p-exact"}))
    assert drv is not None
    assert "loaded from toy_mk_plugin (spec numerics=exact (default profile); " in (
        caplog.text
    )
    assert "requested profile=p-exact, its numerics are logged at bind" in caplog.text


# --- collective_domain ------------------------------------------------------


def _record_agreements(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
    """Resolve every domain to a sentinel group; record the domains asked for
    and the group each all-reduce runs over."""
    group, log = object(), []

    def resolve(domain: str, runner: Any) -> Any:
        log.append(("resolve", domain))
        return group

    def all_reduce_max(t: torch.Tensor, g: Any) -> None:
        log.append(("all_reduce", g))

    monkeypatch.setattr(driver_mod, "agreement_group", resolve)
    monkeypatch.setattr(agreement_mod, "agreement_group", resolve)
    monkeypatch.setattr(agreement_mod, "_all_reduce_max", all_reduce_max)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)
    return group, log


@pytest.mark.parametrize("options", [{}, {"decline": True}])
def test_agreements_run_over_the_declared_domain_group(
    toy: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    """The startup and bind agreements (serving or declined driver) use
    the group `agreement_group` resolves for the spec's domain."""
    group, log = _record_agreements(monkeypatch)
    runner = _runner(options)
    drv = driver_mod.ProgramDriver.maybe_create(runner)
    assert drv.serving is not options.get("decline", False)
    runner._megakernel = drv
    drv.bind_and_warmup(runner)
    assert log == [("resolve", "tp"), ("all_reduce", group)] * 2


@pytest.mark.parametrize("domain", ["tp_x_dcp", "kernel_mesh"])
def test_unsupported_collective_domain_declines_before_any_collective(
    toy: Path, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    src = toy.read_text("utf-8").replace(
        'decline_codes=("toy_declined",)',
        f'decline_codes=("toy_declined",), collective_domain="{domain}"',
    )
    toy.write_text(src, encoding="utf-8")
    _forbid_create_and_collectives(monkeypatch)

    def no_group(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("must decline before resolving an agreement group")

    monkeypatch.setattr(driver_mod, "agreement_group", no_group)
    monkeypatch.setattr(agreement_mod, "agreement_group", no_group)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", no_group)
    assert driver_mod.ProgramDriver.maybe_create(_runner()) is None


_ENTRY_OFF = """
import json, sys
from vllm_torchtpu.megakernel import adapter as mkr
from vllm_torchtpu.megakernel.adapter import testing
out = [mkr.maybe_create(testing.fake_runner(megakernel=mk)) for mk in (None, False)]
print(json.dumps({
    "results": [repr(o) for o in out],
    "driver_loaded": "vllm_torchtpu.megakernel.adapter.driver" in sys.modules,
}))
"""


def test_package_maybe_create_is_off_without_config_and_skips_the_driver() -> None:
    """`megakernel.adapter.maybe_create` with the megakernel off returns
    `None` without importing the driver (or JAX through it)."""
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "VLLM_TPU_MEGAKERNEL"}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_SRC), env.get("PYTHONPATH", "")) if p
    )
    res = subprocess.run(
        [sys.executable, "-c", _ENTRY_OFF],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got == {"results": ["None", "None"], "driver_loaded": False}


def test_package_maybe_create_and_lazy_export_reach_the_driver(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a plugin configured, `megakernel.adapter.maybe_create` returns the
    driver's `ProgramDriver`; the package's lazy `ProgramDriver` is it."""
    _no_group(monkeypatch)
    drv = mkr.maybe_create(_runner())
    assert isinstance(drv, driver_mod.ProgramDriver) and drv.serving
    assert mkr.ProgramDriver is driver_mod.ProgramDriver


def test_weight_update_guard() -> None:
    mkr.guard_weight_update(SimpleNamespace(_megakernel=None))
    mkr.guard_weight_update(SimpleNamespace())
    dd = driver_mod.DeclinedDriver(driver_mod.EligibilityResult(False, "x", "m"))
    mkr.guard_weight_update(SimpleNamespace(_megakernel=dd))
    serving = object.__new__(driver_mod.ProgramDriver)
    with pytest.raises(mkr.MegakernelUnsupported, match="weight_update"):
        mkr.guard_weight_update(SimpleNamespace(_megakernel=serving))


def test_toy_plugin_source_is_valid_python() -> None:
    ast.parse(textwrap.dedent(TOY_PLUGIN))
