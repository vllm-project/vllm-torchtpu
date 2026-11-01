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
"""Test kit for megakernel model folders and plugins (CPU, no TPU).

- `check_spec(plugin)`: the plugin's `SPEC` is valid, its `create` is a lazy
  `"module:attr"` string, and importing the plugin module loads no torch /
  jax / vLLM.
- `run_cpu_e2e(plugin, runner, steps)`: what the runner does at startup and
  per step, on a fake runner: `maybe_create`, `bind_and_warmup`, then one
  `try_forward` per step; returns the outputs and the step mix.
- `run_two_ranks(...)` / `assert_rank_uniform_declines(...)`: a plugin's
  startup on two real gloo ranks; a decline on any rank declines every rank,
  and nothing hangs.
- `route_lazy_loggers(monkeypatch, *packages)`: route the runtime's (and a
  folder's) lazy vLLM loggers to `logging` when vLLM is absent.
- `TOY_PLUGIN` / `write_plugin`: a single-file toy plugin (pure torch).
- Fixtures `mk_fake_runner` (the `testing.fake_runner` factory) and
  `mk_toy_plugin` (the toy plugin importable as `toy_mk_plugin`); import them
  into a `conftest.py` to use them.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from .plugin import TpuMonoSpec
from .testing import fake_runner

_SRC = Path(__file__).resolve().parents[2]
_RUNTIME_PREFIX = "vllm_torchtpu.megakernel_runtime."
# Top-level packages a plugin module must not load at import.
HEAVY_IMPORTS = ("jax", "torch", "vllm")

TOY_PLUGIN = '''
"""A toy megakernel plugin: y = x * scale + B, in pure torch."""
from vllm_torchtpu.megakernel_runtime.plugin import (
    Bucket, Decline, TpuMonoSpec, WarmupCase,
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


SPEC = TpuMonoSpec(
    "toy", "1", create=create, decline_codes=("toy_declined",)
)
'''
TOY_MODULE = "toy_mk_plugin"


def write_plugin(directory: Path, source: str, module: str = TOY_MODULE) -> Path:
    """Write a single-file plugin `module` into `directory`; returns its path."""
    path = Path(directory) / f"{module}.py"
    path.write_text(source, encoding="utf-8")
    return path


def route_lazy_loggers(monkeypatch: pytest.MonkeyPatch, *packages: str) -> None:
    """Route the lazy vLLM loggers of the runtime and of the already imported
    modules of `packages` to `logging` when vLLM is absent (CPU)."""
    importlib.import_module(_RUNTIME_PREFIX + "driver")
    prefixes = (_RUNTIME_PREFIX, *(f"{p}." for p in packages), *packages)
    for name in sorted(sys.modules):
        if not name.startswith(prefixes):
            continue
        lazy = getattr(sys.modules[name], "logger", None)
        if not callable(getattr(lazy, "_resolve", None)):
            continue
        try:
            lazy._resolve()
        except ModuleNotFoundError:
            monkeypatch.setattr(lazy, "_logger", logging.getLogger(lazy._name))


# --- check_spec -----------------------------------------------------------------

_IMPORT_PROBE = """
import importlib, json, sys
heavy, module = json.loads(sys.argv[1]), sys.argv[2]
for name in heavy:
    sys.modules[name] = None
importlib.import_module(module)
print(json.dumps(sorted(n for n in heavy if sys.modules.get(n) is not None)))
"""


def _plugin_ref(plugin: str | ModuleType) -> tuple[str, str]:
    if isinstance(plugin, ModuleType):
        return plugin.__name__, "SPEC"
    mod, _, attr = plugin.partition(":")
    return mod, attr or "SPEC"


def check_spec(
    plugin: str | ModuleType | TpuMonoSpec, *, lazy_create: bool | None = None
) -> TpuMonoSpec:
    """Check a plugin's `SPEC` (a module name, `"module:attr"`, the module or
    the spec itself) and return it; raises `AssertionError` on a problem.

    The spec must be a `TpuMonoSpec` of this runtime's API with well-formed
    `decline_codes`, and its `create` (and `layout_request`, if a string)
    must resolve. `create` must be a lazy `"module:attr"` string when
    `lazy_create` is true (default: when the plugin is a package, i.e. a
    model folder). Given a module, it must import in a fresh interpreter with
    torch, jax and vLLM blocked.
    """
    from .config import PluginRef, import_plugin, resolve_callable

    module: ModuleType | None = None
    if isinstance(plugin, TpuMonoSpec):
        spec = plugin
    else:
        mod_name, attr = _plugin_ref(plugin)
        spec, module = import_plugin(PluginRef(mod_name, attr))
        probe = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE, json.dumps(HEAVY_IMPORTS), mod_name],
            capture_output=True,
            text=True,
            env=_subprocess_env(sys.path),
            timeout=300,
        )
        if probe.returncode != 0:
            raise AssertionError(
                f"importing {mod_name} with {', '.join(HEAVY_IMPORTS)} blocked "
                f"failed; keep the plugin module stdlib-only and import heavy "
                f"modules from `create`:\n{probe.stderr[-3000:]}"
            )
    if lazy_create is None:
        lazy_create = module is not None and hasattr(module, "__path__")
    if lazy_create and (not isinstance(spec.create, str) or ":" not in spec.create):
        raise AssertionError(
            f"SPEC {spec.name!r}: create={spec.create!r} should be a lazy "
            "'module:attr' string (e.g. '.adapter:create'), so importing the "
            "plugin loads no torch / jax / vLLM"
        )
    codes = list(spec.decline_codes)
    if any(not isinstance(c, str) or not c or c == "other" for c in codes):
        raise AssertionError(f"SPEC {spec.name!r}: bad decline_codes {codes}")
    if len(set(codes)) != len(codes):
        raise AssertionError(f"SPEC {spec.name!r}: duplicate decline_codes {codes}")
    if module is not None:
        resolve_callable(spec.create, module, what="create")
        if isinstance(spec.layout_request, str):
            resolve_callable(spec.layout_request, module, what="layout_request")
    return spec


# --- run_cpu_e2e ----------------------------------------------------------------


@dataclasses.dataclass
class CpuE2EResult:
    """What `run_cpu_e2e` saw: the driver, and per step whether the megakernel
    served it (`hits`) and its output (`None` for a stock step)."""

    driver: Any
    runner: Any
    hits: list[bool]
    outputs: list[Any]

    @property
    def program(self) -> Any:
        return self.driver.program

    @property
    def step_mix(self) -> dict[str, int]:
        return dict(self.driver.step_mix_tracker.counts)


def _set_plugin(runner: Any, plugin: str | ModuleType | None, options: Any) -> None:
    add_cfg = runner.vllm_config.additional_config
    mk_cfg = add_cfg.get("megakernel")
    mk_cfg = dict(mk_cfg) if isinstance(mk_cfg, Mapping) else {}
    if plugin is not None:
        mod, attr = _plugin_ref(plugin)
        mk_cfg["plugin"] = mod if attr == "SPEC" else f"{mod}:{attr}"
    if options is not None:
        mk_cfg["options"] = {**mk_cfg.get("options", {}), **dict(options)}
    add_cfg["megakernel"] = mk_cfg


def _step_contexts(steps: Iterable[Any]) -> Iterator[Any]:
    from .hooks import StepContext

    for s in steps:
        yield s if isinstance(s, StepContext) else StepContext(*s)


def run_cpu_e2e(
    plugin: str | ModuleType | None,
    runner: Any,
    steps: Iterable[Any] | Callable[[Any], Iterable[Any]],
    *,
    metadata: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] | None = None,
    options: Mapping[str, Any] | None = None,
) -> CpuE2EResult:
    """Start `plugin` on `runner` and serve `steps`, as the runner would.

    `runner` is a fake runner (or a zero-argument factory of one); `plugin`
    (a module name, `"module:attr"` or module; `None` keeps the runner's) and
    `options` are written into its megakernel config. `steps` are
    `StepContext`s or `(input_ids, positions)` pairs, or a callable of the
    program returning them; `metadata` (or a callable of the program) is the
    forward context's attention metadata during the steps. Raises
    `AssertionError` if the megakernel doesn't serve after startup or bind.
    """
    from . import driver as driver_mod
    from . import maybe_create

    if not hasattr(runner, "vllm_config") and callable(runner):
        runner = runner()
    if plugin is not None or options is not None:
        _set_plugin(runner, plugin, options)
    drv = maybe_create(runner)
    if drv is None or not drv.serving:
        decline = getattr(drv, "decline", None)
        raise AssertionError(f"megakernel not serving after startup: {decline}")
    runner._megakernel = drv
    drv.bind_and_warmup(runner)
    if runner._megakernel is not drv:
        decline = getattr(runner._megakernel, "decline", None)
        raise AssertionError(f"megakernel declined at bind: {decline}")
    program = drv.program
    md = metadata(program) if callable(metadata) else dict(metadata or {})
    ctxs = steps(program) if callable(steps) else steps
    hits: list[bool] = []
    outputs: list[Any] = []
    fwd_ctx = SimpleNamespace(attn_metadata=md)
    with mock.patch.object(driver_mod, "get_forward_context", lambda: fwd_ctx):
        for ctx in _step_contexts(ctxs):
            hit, out = drv.try_forward(runner, ctx)
            hits.append(bool(hit))
            outputs.append(out)
    return CpuE2EResult(drv, runner, hits, outputs)


# --- two gloo ranks -------------------------------------------------------------

_RANK_SCRIPT = """
import datetime, json, logging, sys
rank, out, init, plugin_dir, plugin, options, runner_kw = sys.argv[1:8]
rank = int(rank)
if plugin_dir:
    sys.path.insert(0, plugin_dir)
import torch.distributed as dist
dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2,
                        timeout=datetime.timedelta(seconds=60))
from vllm_torchtpu.megakernel_runtime import agreement as a, driver as d, testing
for lazy in (d.logger, a.logger):
    try:
        lazy._resolve()
    except ModuleNotFoundError:
        lazy._logger = logging.getLogger("mk")
a._tp_cpu_group = lambda: dist.group.WORLD
runner = testing.fake_runner(
    tp_size=2,
    megakernel={"plugin": plugin, "options": json.loads(options)},
    **json.loads(runner_kw),
)
res = {}
try:
    drv = d.ProgramDriver.maybe_create(runner)
    res["type"] = type(drv).__name__
    res["code"] = getattr(getattr(drv, "decline", None), "code", None)
    if isinstance(drv, d.ProgramDriver):
        runner._megakernel = drv
        drv.bind_and_warmup(runner)
        cur = runner._megakernel
        res["type"] = type(cur).__name__
        res["code"] = getattr(getattr(cur, "decline", None), "code", None)
        warmed = getattr(drv.program, "warmed", None)
        if res["type"] == "ProgramDriver" and warmed is not None:
            res["warmed"] = len(warmed)
except Exception as exc:
    res["raised"] = f"{type(exc).__name__}: {exc}"
json.dump(res, open(out, "w"))
dist.destroy_process_group()
"""


def _subprocess_env(paths: Sequence[str] = ()) -> dict[str, str]:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    entries = [str(_SRC), *paths, env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(p for p in entries if p)
    env.pop("VLLM_TPU_MEGAKERNEL", None)
    return env


def run_two_ranks(
    tmp_path: Path,
    *,
    plugin: str = TOY_MODULE,
    options: Mapping[str, Any] | None = None,
    sources: Sequence[str] | None = None,
    runner_kwargs: Mapping[str, Any] | None = None,
    timeout: float = 240,
) -> list[dict[str, Any]]:
    """Start `plugin` on two gloo ranks (a TP=2 fake runner each) and return
    each rank's outcome: `{"type", "code"[, "warmed"]}` or `{"raised"}`.

    `sources` (one per rank) write the single-file plugin module `plugin`
    into a per-rank directory first, e.g. to skew it on one rank;
    `runner_kwargs` (JSON values) go to `fake_runner`.
    """
    script = tmp_path / "rank.py"
    script.write_text(_RANK_SCRIPT, encoding="utf-8")
    env = _subprocess_env()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    init = f"file://{tmp_path / 'pg_init'}"
    procs, outs = [], []
    for rank in (0, 1):
        pdir = ""
        if sources is not None:
            pdir_path = tmp_path / f"plugin{rank}"
            pdir_path.mkdir()
            write_plugin(pdir_path, sources[rank], plugin)
            pdir = str(pdir_path)
        out = tmp_path / f"out{rank}.json"
        outs.append(out)
        args = [
            str(rank),
            str(out),
            init,
            pdir,
            plugin,
            json.dumps(dict(options or {})),
            json.dumps(dict(runner_kwargs or {})),
        ]
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
            logs.append(p.communicate(timeout=timeout)[0])
    finally:
        for p in procs:
            p.kill()
    for p, log in zip(procs, logs, strict=True):
        assert p.returncode == 0, log
    return [json.loads(o.read_text(encoding="utf-8")) for o in outs]


def assert_rank_uniform_declines(tmp_path: Path, **kwargs: Any) -> list[dict[str, Any]]:
    """`run_two_ranks(tmp_path, **kwargs)` and assert no rank serves: each one
    declines or raises, at least one declines, and nothing hung."""
    results = run_two_ranks(tmp_path, **kwargs)
    for rank, res in enumerate(results):
        assert res.get("type") == "DeclinedDriver" or "raised" in res, (rank, results)
    assert any(r.get("type") == "DeclinedDriver" for r in results), results
    return results


# --- fixtures -------------------------------------------------------------------


@pytest.fixture
def mk_fake_runner(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    """The `fake_runner` factory, with `VLLM_TPU_MEGAKERNEL` unset and the
    runtime's lazy loggers routed to `logging`."""
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    route_lazy_loggers(monkeypatch)
    return fake_runner


@pytest.fixture
def mk_toy_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """`TOY_PLUGIN` as the single-file module `toy_mk_plugin` on `sys.path`;
    yields its path (edit it to change the plugin source)."""
    from . import fingerprint

    path = write_plugin(tmp_path, TOY_PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(TOY_MODULE, None)
    importlib.invalidate_caches()
    fingerprint.clear_caches()
    yield path
    sys.modules.pop(TOY_MODULE, None)
    fingerprint.clear_caches()


__all__ = [
    "HEAVY_IMPORTS",
    "TOY_MODULE",
    "TOY_PLUGIN",
    "CpuE2EResult",
    "assert_rank_uniform_declines",
    "check_spec",
    "mk_fake_runner",
    "mk_toy_plugin",
    "route_lazy_loggers",
    "run_cpu_e2e",
    "run_two_ranks",
    "write_plugin",
]
