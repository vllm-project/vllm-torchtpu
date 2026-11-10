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
"""The torch.compile fast-call hook and additional_config["tpu_compile"].

The hook is installed when the TPU platform module loads, so it is always in
place; these tests check that it does nothing until the options arm it, and
that each option does what it says once armed. Functions are compiled with
the eager backend, which runs on CPU and keeps Dynamo's guards and cache
entries.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_torchtpu import _qx_fastcall
from vllm_torchtpu.compilation import compile_options
from vllm_torchtpu.compilation.compile_options import CompileOptions


@pytest.fixture(autouse=True)
def _disarmed():
    """Every test starts and ends with the hook disarmed (main's default)."""
    _qx_fastcall.configure(fast=False, drop_global_state=False)
    torch._dynamo.reset()
    yield
    _qx_fastcall.configure(fast=False, drop_global_state=False)
    torch._dynamo.reset()


def _config(section=None):
    additional = {} if section is None else {compile_options.SECTION: section}
    return SimpleNamespace(additional_config=additional)


def _compiled(fn, **options):
    """What the hook builds for an eligible function, on the eager backend."""
    compiled = torch.compile(
        fn,
        backend="eager",
        fullgraph=True,
        dynamic=False,
        options={"guard_filter_fn": _qx_fastcall._drop_global_state},
    )
    return _qx_fastcall._wrap(compiled, fn)


def _calls(wrapped, n):
    x = torch.arange(4.0)
    before = list(_qx_fastcall._STATS)
    for _ in range(n):
        assert torch.equal(wrapped(x), x * 2 + 1)
    return [a - b for a, b in zip(_qx_fastcall._STATS, before)]


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


def test_hook_is_installed_at_platform_import():
    import vllm_torchtpu.platforms  # noqa: F401

    assert _qx_fastcall._INSTALLED
    assert torch.compile.__name__ == "compile"
    assert torch.compile.__code__.co_name == "qx_compile"


def test_disarmed_wrapper_only_calls_the_compiled_function():
    def f(x):
        return x * 2 + 1

    wrapped = _compiled(f)
    assert _calls(wrapped, 4) == [0, 0]
    # Nothing was cached for direct dispatch.
    assert wrapped._qx_fast_state["entries"] == []
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


def test_fast_dispatch_keeps_every_compiled_entry():
    """Two shapes alternating both dispatch directly once each is compiled."""

    def f(x):
        return x * 2 + 1

    compile_options.apply(_config({"fast_compiled_call": True}))
    wrapped = _compiled(f)
    small, large = torch.arange(4.0), torch.arange(8.0)
    before = list(_qx_fastcall._STATS)
    for x in (small, large, small, large, small, large):
        assert torch.equal(wrapped(x), x * 2 + 1)
    fast, slow = (a - b for a, b in zip(_qx_fastcall._STATS, before))
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
    return _qx_fastcall._entry_count(g.__code__)


def test_disarmed_filter_keeps_the_global_state_guard():
    """Off by default: a CPU thread-count change recompiles, as on main."""
    entries = [
        SimpleNamespace(guard_type="GLOBAL_STATE"),
        SimpleNamespace(guard_type="TENSOR_MATCH"),
    ]
    assert _qx_fastcall._drop_global_state(entries) == [True, True]
    assert _num_threads_recompiles() == 2


def test_drop_global_state_guard_reuses_the_entry_across_threads():
    compile_options.apply(_config({"drop_global_state_guard": True}))
    entries = [
        SimpleNamespace(guard_type="GLOBAL_STATE"),
        SimpleNamespace(guard_type="TENSOR_MATCH"),
    ]
    assert _qx_fastcall._drop_global_state(entries) == [False, True]
    assert _num_threads_recompiles() == 1


class _Module:
    def __init__(self, *children):
        self.aot_compiled_fn = SimpleNamespace(disable_guard_check=False)
        self._children = children

    def modules(self):
        yield self
        for child in self._children:
            yield from child.modules()


def _aot_runner(section):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    target = _Module(_Module())
    draft = _Module()
    runner = SimpleNamespace(
        vllm_config=_config(section),
        model=target,
        drafter=SimpleNamespace(draft_model=draft),
    )
    TPUModelRunner._qx_maybe_disable_aot_guard_checks(runner)
    return [m.aot_compiled_fn.disable_guard_check for m in (*target.modules(), draft)]


def test_aot_guard_checks_stay_on_by_default():
    assert _aot_runner(None) == [False, False, False]


def test_skip_aot_guard_checks_covers_target_and_drafter():
    assert _aot_runner({"skip_aot_guard_checks": True}) == [True, True, True]


def test_worker_arms_the_hook_before_anything_else():
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
    assert _qx_fastcall._CONFIG == {"fast": True, "drop_global_state": False}


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
