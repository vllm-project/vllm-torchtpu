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
"""The tpu_compile decorator's fast dispatch and additional_config["tpu_compile"].

Functions are compiled with ``tpu_compile`` at import, before the options are
known; these tests check that its wrapper does nothing until the options arm
it, and that each option does what it says once armed. Functions are compiled
with the eager backend, which runs on CPU and keeps Dynamo's guards and cache
entries.
"""

import functools
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu.compilation import compile_options, fast_dispatch
from vllm_torchtpu.compilation.compile_options import CompileOptions


@pytest.fixture(autouse=True)
def _disarmed():
    """Every test starts and ends with the options off (main's default)."""
    fast_dispatch.configure(fast=False, drop_global_state=False)
    torch._dynamo.reset()
    yield
    fast_dispatch.configure(fast=False, drop_global_state=False)
    torch._dynamo.reset()


def _config(section=None):
    additional = {} if section is None else {compile_options.SECTION: section}
    return SimpleNamespace(additional_config=additional)


def _compiled(fn):
    """What tpu_compile builds for a function, on the eager backend."""
    compiled = torch.compile(
        fn,
        backend="eager",
        fullgraph=True,
        dynamic=False,
        options={"guard_filter_fn": fast_dispatch._drop_global_state},
    )
    return fast_dispatch._wrap(compiled, fn)


def _calls(wrapped, n):
    x = torch.arange(4.0)
    before = list(fast_dispatch._STATS)
    for _ in range(n):
        assert torch.equal(wrapped(x), x * 2 + 1)
    return [a - b for a, b in zip(fast_dispatch._STATS, before)]


def test_options_default_off():
    assert compile_options.parse(None) == CompileOptions()
    assert not any(vars(CompileOptions()).values())


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"fast_compiled_call": 1}, "must be a bool"),
        ({"fast_call": True}, "unknown keys"),
        ("on", "must be a JSON object"),
    ],
)
def test_options_are_strict(raw, message):
    with pytest.raises(ValueError, match=message):
        compile_options.parse(raw)


def test_platform_import_leaves_torch_compile_alone():
    """Nothing in the package replaces torch.compile: importing the platform,
    the worker and the runner, which decorate their compiled functions, keeps
    the torch.compile the process had."""
    code = """
import torch
before = torch.compile
import vllm_torchtpu.platforms  # noqa: F401
from vllm_torchtpu.runner import tpu_runner  # noqa: F401
from vllm_torchtpu.worker import tpu_worker  # noqa: F401
assert torch.compile is before, torch.compile
assert hasattr(tpu_runner.TPUModelRunner.sample_from_logits, "_fast_dispatch_state")
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


def _stand_in_compile(monkeypatch, result=None):
    """torch.compile replaced by a stand-in that records the keyword arguments
    of each call and compiles the "tpu" backend as "eager" (or returns
    ``result`` when given)."""
    real_compile = torch.compile
    calls = []

    def stand_in(model=None, *args, **kwargs):
        calls.append(
            {k: dict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        )
        if result is not None:
            return result(model)
        if kwargs.get("backend") == "tpu":
            kwargs = {**kwargs, "backend": "eager"}
        return real_compile(model, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", stand_in)
    return calls


ELIGIBLE = {"backend": "tpu", "fullgraph": True, "dynamic": False}


def test_tpu_compile_compiles_with_the_filter_and_wraps(monkeypatch):
    calls = _stand_in_compile(monkeypatch)

    def f(x):
        return x * 2 + 1

    wrapped = fast_dispatch.tpu_compile(f)
    assert calls == [
        {**ELIGIBLE, "options": {"guard_filter_fn": fast_dispatch._drop_global_state}}
    ]
    assert wrapped._fast_dispatch_state["entries"] == []
    assert wrapped._torchdynamo_orig_callable is f
    # Decorated before the options are known: disarmed it calls the compiled
    # function; armed, the first call goes through Dynamo and records the
    # entry, and the next ones dispatch directly.
    assert _calls(wrapped, 2) == [0, 0]
    fast_dispatch.configure(fast=True, drop_global_state=False)
    assert _calls(wrapped, 3) == [2, 1]


def test_tpu_compile_returns_the_function_when_dynamo_is_disabled(monkeypatch):
    """With TORCHDYNAMO_DISABLE=1, torch.compile returns the function itself."""
    _stand_in_compile(monkeypatch, result=lambda model: model)

    def f(x):
        return x * 2 + 1

    assert fast_dispatch.tpu_compile(f) is f


def test_decorated_modules_import_with_dynamo_disabled():
    """A module-level decorated site imports and runs with Dynamo disabled, as
    on main: the decorator leaves the plain function."""
    code = """
import torch
from vllm_torchtpu.runner import min_tokens
fn = min_tokens.mask_stop_tokens
assert not hasattr(fn, "_fast_dispatch_state"), fn
assert fn.__code__.co_name == "mask_stop_tokens", fn
print("ok")
"""
    env = dict(os.environ, TORCHDYNAMO_DISABLE="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")


class _Module(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("target", ["module", "bound method", "partial"])
def test_tpu_compile_takes_only_plain_functions(target):
    """The wrapper binds a plain function's own code and closure; anything
    else is refused at decoration rather than dispatched wrongly."""
    if target == "module":
        model = _Module()
    elif target == "bound method":
        model = _Module().forward
    else:
        model = functools.partial(lambda x: x)
    with pytest.raises(TypeError, match="plain function"):
        fast_dispatch.tpu_compile(model)


def test_disarmed_wrapper_only_calls_the_compiled_function():
    def f(x):
        return x * 2 + 1

    wrapped = _compiled(f)
    assert _calls(wrapped, 4) == [0, 0]
    # Nothing was cached for direct dispatch.
    assert wrapped._fast_dispatch_state["entries"] == []
    # torch._dynamo.disable() unwraps it to the plain function.
    assert wrapped._torchdynamo_orig_callable is f
    assert torch._dynamo.eval_frame.innermost_fn(wrapped) is f


def test_fast_compiled_call_dispatches_directly_once_armed():
    def f(x):
        return x * 2 + 1

    compile_options.apply(_config({"fast_compiled_call": True}))
    wrapped = _compiled(f)
    fast, slow = _calls(wrapped, 4)
    assert slow == 1  # the first call compiles and caches the entry
    assert fast == 3


def test_a_traced_caller_takes_the_compiled_path():
    """An outer compiled function that calls an armed wrapper with cached
    entries traces through the compiled function, not the direct dispatch's
    cache-entry and eval-frame queries, so a fullgraph trace holds."""

    def f(x):
        return x * 2 + 1

    compile_options.apply(_config({"fast_compiled_call": True}))
    wrapped = _compiled(f)
    assert _calls(wrapped, 3) == [2, 1]
    assert wrapped._fast_dispatch_state["entries"]

    def outer(x):
        return wrapped(x) - 1

    traced = torch.compile(outer, backend="eager", fullgraph=True, dynamic=False)
    x = torch.arange(4.0)
    before = list(fast_dispatch._STATS)
    assert torch.equal(traced(x), x * 2)
    assert before == fast_dispatch._STATS


def test_fast_dispatch_binds_keyword_only_defaults():
    """A positional call of a function with a keyword-only default: the frame
    Dynamo guards holds the default, so the guards must see it too."""

    def f(x, *, scale=2):
        return x * scale + 1

    compile_options.apply(_config({"fast_compiled_call": True}))
    wrapped = _compiled(f)
    fast, slow = _calls(wrapped, 4)
    assert (fast, slow) == (3, 1)
    # An explicit keyword still takes the bound path and its own guards.
    x = torch.arange(4.0)
    assert torch.equal(wrapped(x, scale=3), x * 3 + 1)


def test_fast_dispatch_keeps_every_compiled_entry():
    """Two shapes alternating both dispatch directly once each is compiled."""

    def f(x):
        return x * 2 + 1

    compile_options.apply(_config({"fast_compiled_call": True}))
    wrapped = _compiled(f)
    small, large = torch.arange(4.0), torch.arange(8.0)
    before = list(fast_dispatch._STATS)
    for x in (small, large, small, large, small, large):
        assert torch.equal(wrapped(x), x * 2 + 1)
    fast, slow = (a - b for a, b in zip(fast_dispatch._STATS, before))
    assert (fast, slow) == (4, 2)


def _num_threads_recompiles():
    def g(x):
        return x * 2 + 1

    compiled = _compiled(g)
    x = torch.arange(4.0)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        compiled(x)
        torch.set_num_threads(2)
        compiled(x)
    finally:
        torch.set_num_threads(threads)
    return fast_dispatch._entry_count(g.__code__)


def test_disarmed_filter_keeps_the_global_state_guard():
    """Off by default: a CPU thread-count change recompiles, as on main."""
    entries = [
        SimpleNamespace(guard_type="GLOBAL_STATE"),
        SimpleNamespace(guard_type="TENSOR_MATCH"),
    ]
    assert fast_dispatch._drop_global_state(entries) == [True, True]
    assert _num_threads_recompiles() == 2


def test_drop_global_state_guard_reuses_the_entry_across_threads():
    compile_options.apply(_config({"drop_global_state_guard": True}))
    entries = [
        SimpleNamespace(guard_type="GLOBAL_STATE"),
        SimpleNamespace(guard_type="TENSOR_MATCH"),
    ]
    assert fast_dispatch._drop_global_state(entries) == [False, True]
    assert _num_threads_recompiles() == 1


def test_worker_arms_the_options_before_anything_else():
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    class Stop(Exception):
        pass

    config = _config({"fast_compiled_call": True})
    with (
        patch("vllm_torchtpu.patch_registry.apply", side_effect=Stop),
        pytest.raises(Stop),
    ):
        TPUWorker(
            vllm_config=config,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29500",
        )
    assert fast_dispatch._CONFIG == {"fast": True, "drop_global_state": False}


class _HashStub:
    """What VllmConfig.compute_hash reads, as in
    tests/compilation/test_additional_config_hash_patch.py."""

    def __init__(self, additional_config):
        self.additional_config = additional_config
        self.observability_config = SimpleNamespace(compute_hash=lambda: "o")
        self.aux_output_config = SimpleNamespace(compute_hash=lambda: "a")

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None


def test_section_stays_in_the_compile_cache_key():
    """The options change how compiled programs are guarded and called, so
    turning one on must change the compile-cache key."""
    from vllm.config import VllmConfig

    from vllm_torchtpu import _patch_vllm_config_hash_ignore_diagnostics
    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    _patch_vllm_config_hash_ignore_diagnostics()
    assert compile_options.SECTION not in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
    off = VllmConfig.compute_hash(_HashStub({}))
    on = VllmConfig.compute_hash(
        _HashStub({compile_options.SECTION: {"fast_compiled_call": True}})
    )
    assert on != off
