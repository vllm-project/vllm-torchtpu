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
"""`ProgramDriver` with a toy plugin that serves: startup order, agreements,
capability guards, warmup shapes, routing, and a 2-rank gloo run."""

from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from vllm_torchtpu import megakernel_runtime as mkr
from vllm_torchtpu.megakernel_runtime import fingerprint as fp
from vllm_torchtpu.megakernel_runtime import testing
from vllm_torchtpu.megakernel_runtime.hooks import StepContext
from vllm_torchtpu.megakernel_runtime.plugin import Bucket, RunnerShape, RunnerView

_SRC = Path(driver_mod.__file__).resolve().parents[2]

TOY_PLUGIN = '''
"""A toy megakernel plugin: y = x * scale + B, in pure torch."""
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket, Decline, MegakernelPlugin, WarmupCase,
)


class ToyProgram:
    display_name = "Toy"

    def __init__(self, view, options):
        self.scale = float(options.get("scale", 2.0))
        self.mesh = "unloaded"
        self.warmed = []
        self.events = []

    def buckets(self, token_paddings):
        self.events.append(("buckets", tuple(token_paddings)))
        return [Bucket(1, 4), Bucket(1, 8, out_rows=8)]

    def load(self, mesh):
        self.mesh = mesh
        self.events.append("load")

    def bind(self, view):
        self.events.append("bind")
        return None

    def build(self, bucket):
        scale, b = self.scale, bucket.B

        def step(x):
            return x * scale + b

        return step

    def warmup_inputs(self, bucket, shape):
        import torch

        def check(out):
            self.warmed.append((bucket.key, shape, float(out[0])))

        return WarmupCase((torch.ones(shape.tok_pad),), check)

    def decide(self, ctx, fwd):
        from vllm_torchtpu.megakernel_runtime.plugin import StepDecision

        n = int(ctx.input_ids.shape[0])
        b = next((b for b in (4, 8) if b >= n), None)
        if b is None:
            return StepDecision(False, "stock_big", n, 1)
        return StepDecision(True, f"mk_t1_b{b}", n, 1, b)

    def forward(self, decision, ctx, fwd):
        step = fwd.steps[(1, decision.bucket_t)]
        return step(ctx.input_ids.float())[: fwd.num_tokens]

    def on_warmup_done(self):
        self.events.append("warm")


def create(view, options):
    rank = view.tp_rank
    if options.get("decline") or options.get("decline_rank") == rank:
        return Decline("toy_declined", "toy says no")
    if options.get("raise_rank") == rank:
        raise RuntimeError("toy create bug")
    return ToyProgram(view, options)


PLUGIN = MegakernelPlugin(
    "toy", "1", create=create, decline_codes=("toy_declined",)
)
'''


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


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The toy plugin as a single-file module `toy_mk_plugin` on `sys.path`."""
    (tmp_path / "toy_mk_plugin.py").write_text(TOY_PLUGIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("toy_mk_plugin", None)
    importlib.invalidate_caches()
    fp.clear_caches()
    yield tmp_path / "toy_mk_plugin.py"
    sys.modules.pop("toy_mk_plugin", None)
    fp.clear_caches()


def _no_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(driver_mod, "_tp_cpu_group", lambda: None)


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
    assert prog.events == ["load", "bind", ("buckets", (16, 32, 64)), "warm"]
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
    monkeypatch.setattr(driver_mod, "_tp_cpu_group", lambda: g)
    monkeypatch.setattr(driver_mod, "_all_reduce_max", g.all_reduce_max)
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
    prog = toy_mod.PLUGIN.create(view, {"scale": 1.0})
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


# --- Capability guards --------------------------------------------------------

_QWEN_CAPS = frozenset({"spec_decode", "mrope", "prefix_caching", "kv_offload"})


def _kv(name: str, members: list[str] | None = None) -> Any:
    extra = {"connectors": [{"kv_connector": m} for m in members or []]}
    return SimpleNamespace(kv_connector=name, kv_connector_extra_config=extra)


@pytest.mark.parametrize(
    "kv, code",
    [
        (None, None),
        (_kv("OffloadingConnector"), None),
        (_kv("TPURaidenOffloadingConnector"), None),
        (
            _kv(
                "MultiConnector",
                ["OffloadingConnector", "TPURaidenOffloadingConnector"],
            ),
            None,
        ),
        (
            _kv("MultiConnector", ["OffloadingConnector", "NixlConnector"]),
            "kv_connector_unsupported:NixlConnector",
        ),
        (_kv("TPURaidenConnector"), "kv_connector_unsupported:TPURaidenConnector"),
        (_kv("DecodeBenchConnector"), "kv_connector_unsupported:DecodeBenchConnector"),
        (_kv("MultiConnector"), "kv_connector_unsupported:MultiConnector"),
    ],
)
def test_kv_connector_guard(kv: Any, code: str | None) -> None:
    runner = testing.fake_runner(kv_transfer_config=kv)
    res = driver_mod.check_runner_capabilities(runner, _QWEN_CAPS, tp_size=1)
    assert (res.code if res else None) == code
    if code is None and kv is not None:
        res = driver_mod.check_runner_capabilities(runner, frozenset(), tp_size=1)
        assert res is not None and res.code == "kv_offload_unsupported"
    if code is not None:
        caps = _QWEN_CAPS | {"kv_transfer"}
        assert driver_mod.check_runner_capabilities(runner, caps, tp_size=1) is None


@pytest.mark.parametrize(
    "kw, code, cap",
    [
        ({"speculative_tokens": 3}, "spec_decode_unsupported", "spec_decode"),
        ({"uses_mrope": True}, "mrope_unsupported", "mrope"),
        (
            {"enable_prefix_caching": True},
            "prefix_caching_unsupported",
            "prefix_caching",
        ),
        ({"enable_eplb": True}, "eplb_dynamic_unsupported", "eplb_dynamic"),
        ({"supports_mm_inputs": True}, "multimodal_inputs", "multimodal"),
        ({"lora_config": object()}, "lora_enabled", "lora"),
        ({"_pp_is_last": False}, "pipeline_parallel", "pipeline_parallel"),
        ({"_dcp_enabled": True}, "context_parallel", "dcp"),
        ({"_pcp_enabled": True}, "context_parallel", "pcp"),
    ],
)
def test_undeclared_capability_is_a_named_decline(
    kw: dict[str, Any], code: str, cap: str
) -> None:
    runner = testing.fake_runner(**kw)
    res = driver_mod.check_runner_capabilities(runner, frozenset(), tp_size=1)
    assert res is not None and res.code == code and res.message
    caps = frozenset({cap})
    assert driver_mod.check_runner_capabilities(runner, caps, tp_size=1) is None


def test_static_expert_placement_and_sleep_mode() -> None:
    # A static placement table is not dynamic EPLB.
    runner = testing.fake_runner(TPU_EXPERT_PLACEMENT="table")
    assert driver_mod.check_runner_capabilities(runner, _QWEN_CAPS, tp_size=1) is None
    runner.vllm_config.model_config.enable_sleep_mode = True
    res = driver_mod.check_runner_capabilities(runner, _QWEN_CAPS, tp_size=1)
    assert res is not None and res.code == "sleep_mode_unsupported"
    # Never-declarable ones: DP lockstep and DP > 1.
    runner = testing.fake_runner(_dp_lockstep_enabled=lambda: True)
    res = driver_mod.check_runner_capabilities(runner, _QWEN_CAPS, tp_size=1)
    assert res is not None and res.code == "dp_lockstep"
    runner = testing.fake_runner()
    runner.parallel_config.data_parallel_size = 2
    res = driver_mod.check_runner_capabilities(runner, _QWEN_CAPS, tp_size=1)
    assert res is not None and res.code == "dp_unsupported"


def test_undeclared_capability_declines_before_create(
    toy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_collective(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("must decline before create / any collective")

    monkeypatch.setattr(driver_mod, "agree_on_decline", no_collective)
    monkeypatch.setattr(driver_mod, "_build_tp_rank_mesh", no_collective)
    assert driver_mod.ProgramDriver.maybe_create(_runner(speculative_tokens=3)) is None


def test_weight_update_guard() -> None:
    mkr.guard_weight_update(SimpleNamespace(_megakernel=None))
    mkr.guard_weight_update(SimpleNamespace())
    dd = driver_mod.DeclinedDriver(driver_mod._rt.EligibilityResult(False, "x", "m"))
    mkr.guard_weight_update(SimpleNamespace(_megakernel=dd))
    serving = object.__new__(driver_mod.ProgramDriver)
    with pytest.raises(mkr.MegakernelUnsupported, match="weight_update"):
        mkr.guard_weight_update(SimpleNamespace(_megakernel=serving))
    # `TPUWorker.sync_weights` runs the guard before syncing.
    src = (_SRC / "vllm_torchtpu" / "worker" / "tpu_worker.py").read_text("utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "sync_weights"
    )
    first = ast.unparse(fn.body[0])
    assert first == "megakernel_runtime.guard_weight_update(self.model_runner)"


def test_fake_runner_implements_every_runner_view_property() -> None:
    props = [n for n, v in vars(RunnerView).items() if isinstance(v, property)]
    assert len(props) == 17
    model = object()
    runner = testing.fake_runner(
        model=model,
        hf_config={"architectures": ["X"]},
        tp_size=2,
        speculative_tokens=3,
        req_block_pairs=((8, 4), (2, 16)),
        max_pages_per_seq=32,
        mamba_slot_read_offsets=torch.zeros(2),
        uses_mrope=True,
        enforce_eager=True,
    )
    view = RunnerView(runner, tp_rank=1)
    got = {n: getattr(view, n) for n in props}
    assert got["model"] is model and got["vllm_config"] is runner.vllm_config
    assert got["device"] == torch.device("cpu")
    assert got["hf_config"] == {"architectures": ["X"]}
    assert (got["tp_size"], got["tp_rank"]) == (2, 1)
    assert got["speculative_tokens"] == 3 and got["ckpt_window"] == 4
    assert got["cache_block_size"] == 256
    assert got["token_paddings"] == (16, 32, 64)
    assert got["req_block_pairs"] == ((8, 32), (2, 16))
    assert got["max_pages_per_seq"] == 32
    assert got["mamba_slot_read_offsets"] is runner.mamba_slot_read_offsets
    assert got["uses_mrope"] and got["enforce_eager"] and got["unified_kv_layout"]
    assert got["device_kind"] == "cpu"
    assert RunnerView(testing.fake_runner(ckpt_window=6)).ckpt_window == 6


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

_RANK_SCRIPT = """
import datetime, json, logging, sys
rank, out, init, plugin_dir, options = sys.argv[1:6]
rank = int(rank)
sys.path.insert(0, plugin_dir)
import torch.distributed as dist
dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2,
                        timeout=datetime.timedelta(seconds=60))
from vllm_torchtpu.megakernel_runtime import driver as d, testing
try:
    d.logger._resolve()
except ModuleNotFoundError:
    d.logger._logger = logging.getLogger("mk")
d._tp_cpu_group = lambda: dist.group.WORLD
runner = testing.fake_runner(
    tp_size=2,
    megakernel={"plugin": "toy_mk_plugin", "options": json.loads(options)},
)
res = {}
try:
    drv = d.ProgramDriver.maybe_create(runner)
    res["type"] = type(drv).__name__
    res["code"] = getattr(getattr(drv, "decline", None), "code", None)
    if isinstance(drv, d.ProgramDriver):
        runner._megakernel = drv
        drv.bind_and_warmup(runner)
        res["warmed"] = len(drv.program.warmed)
except Exception as exc:
    res["raised"] = f"{type(exc).__name__}: {exc}"
json.dump(res, open(out, "w"))
dist.destroy_process_group()
"""


def _run_two_ranks(
    tmp_path: Path, options: dict[str, Any], sources: tuple[str, str]
) -> list[dict[str, Any]]:
    script = tmp_path / "rank.py"
    script.write_text(_RANK_SCRIPT, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(_SRC),
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    env.pop("VLLM_TPU_MEGAKERNEL", None)
    init = f"file://{tmp_path / 'pg_init'}"
    procs, outs = [], []
    for rank in (0, 1):
        pdir = tmp_path / f"plugin{rank}"
        pdir.mkdir()
        (pdir / "toy_mk_plugin.py").write_text(sources[rank], encoding="utf-8")
        out = tmp_path / f"out{rank}.json"
        outs.append(out)
        args = [str(rank), str(out), init, str(pdir), json.dumps(options)]
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script), *args],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    logs = []
    try:
        for p in procs:
            logs.append(p.communicate(timeout=240)[0])
    finally:
        for p in procs:
            p.kill()
    for p, log in zip(procs, logs, strict=True):
        assert p.returncode == 0, log
    return [json.loads(o.read_text(encoding="utf-8")) for o in outs]


@pytest.mark.parametrize("scenario", ["serve", "decline", "mismatch", "create_error"])
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
    r0, r1 = _run_two_ranks(tmp_path, options, src)
    if scenario == "serve":
        assert r0 == r1 == {"type": "ProgramDriver", "code": None, "warmed": 2}
    elif scenario == "decline":
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:toy_declined"}
        assert r1 == {"type": "DeclinedDriver", "code": "toy_declined"}
    elif scenario == "mismatch":
        assert r0 == r1 == {"type": "DeclinedDriver", "code": "plugin_mismatch"}
    else:
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:create_error"}
        assert r1 == {"raised": "RuntimeError: toy create bug"}


def test_toy_plugin_source_is_valid_python() -> None:
    ast.parse(textwrap.dedent(TOY_PLUGIN))
