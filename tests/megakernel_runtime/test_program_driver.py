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
import logging
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from vllm_torchtpu import megakernel_runtime as mkr
from vllm_torchtpu.megakernel import errors as mk_errors
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime import testing, testkit
from vllm_torchtpu.megakernel_runtime.hooks import StepContext
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket,
    RunnerShape,
    RunnerView,
    TpuMonoSpec,
)
from vllm_torchtpu.megakernel_runtime.testkit import (
    TOY_PLUGIN,
    assert_rank_uniform_declines,
    run_two_ranks,
)

_SRC = Path(driver_mod.__file__).resolve().parents[2]


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


mk_toy_plugin = testkit.mk_toy_plugin


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
    monkeypatch.setattr(driver_mod, "get_forward_context", lambda: ctx)


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
    from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError

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
    from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError

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


def test_weight_update_guard() -> None:
    mkr.guard_weight_update(SimpleNamespace(_megakernel=None))
    mkr.guard_weight_update(SimpleNamespace())
    dd = driver_mod.DeclinedDriver(driver_mod.EligibilityResult(False, "x", "m"))
    mkr.guard_weight_update(SimpleNamespace(_megakernel=dd))
    serving = object.__new__(driver_mod.ProgramDriver)
    with pytest.raises(mkr.MegakernelUnsupported, match="weight_update"):
        mkr.guard_weight_update(SimpleNamespace(_megakernel=serving))


def test_runner_shapes_without_paddings() -> None:
    shapes = driver_mod.runner_shapes(
        Bucket(4, 16), (), (), ckpt_window=1, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(16, 4, 9, 4)]
    shapes = driver_mod.runner_shapes(
        Bucket(1, 64), (16, 32), ((4, 2), (1, 8)), ckpt_window=3, max_pages_per_seq=9
    )
    assert shapes == [RunnerShape(32, 4, 2, 3), RunnerShape(32, 1, 8, 3)]


# --- 2-rank gloo ----------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", ["serve", "decline", "mismatch", "create_error", "import_error"]
)
def test_two_rank_gloo_agreement(tmp_path: Path, scenario: str) -> None:
    """Real 2-process gloo agreement: never hangs, always rank-uniform."""
    src = (TOY_PLUGIN, TOY_PLUGIN)
    options: dict[str, Any] = {}
    if scenario == "decline":
        options = {"decline_rank": 1}
    elif scenario == "mismatch":
        src = (TOY_PLUGIN, TOY_PLUGIN + "\n# skewed on rank 1\n")
    elif scenario == "create_error":
        options = {"raise_rank": 1}
    elif scenario == "import_error":
        src = (TOY_PLUGIN, 'raise ImportError("plugin missing on this host")\n')
    if scenario == "serve":
        r0, r1 = run_two_ranks(tmp_path, options=options, sources=src)
    else:
        r0, r1 = assert_rank_uniform_declines(tmp_path, options=options, sources=src)
    if scenario == "serve":
        assert r0 == r1 == {"type": "ProgramDriver", "code": None, "warmed": 2}
    elif scenario == "decline":
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:toy_declined"}
        assert r1 == {"type": "DeclinedDriver", "code": "toy_declined"}
    elif scenario == "mismatch":
        assert r0 == r1 == {"type": "DeclinedDriver", "code": "plugin_mismatch"}
    elif scenario == "create_error":
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:create_error"}
        assert r1 == {"raised": "RuntimeError: toy create bug"}
    else:
        # Rank 1 votes before raising (token 0): rank 0 declines, no hang.
        assert r0 == {"type": "DeclinedDriver", "code": "plugin_mismatch"}
        assert r1["raised"].startswith("MegakernelConfigError")
        assert "plugin missing on this host" in r1["raised"]


def test_toy_plugin_source_is_valid_python() -> None:
    ast.parse(textwrap.dedent(TOY_PLUGIN))
