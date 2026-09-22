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
"""TpuMoERunner against vLLM's real MoERunner and out-of-tree registry.

Runs without TorchTPU: `vllm_torchtpu.layers` imports TPU attention kernels,
so the module under test is loaded by file path and the fused EP bridge it
imports lazily is replaced by a stub in `sys.modules`.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import vllm
from vllm.model_executor.custom_op import (maybe_get_oot_by_class,
                                           op_registry_oot)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner import moe_runner as upstream

from vllm_torchtpu import envs

pytestmark = pytest.mark.cpu_test

_REPO = Path(__file__).resolve().parents[2]
_RUNNER_PATH = _REPO / "src/vllm_torchtpu/layers/adapter/moe_runner.py"
_LAYERS_INIT_PATH = _REPO / "src/vllm_torchtpu/layers/__init__.py"
_REGISTRY_PATH = _REPO / "src/vllm_torchtpu/patch_registry.py"
_BRIDGE = "vllm_torchtpu.layers.adapter.fused_moe_ep"

_OVERRIDES = ("_select_forward", "_fused_output_is_reduced", "_maybe_dispatch",
              "_maybe_combine")
_OLD_MARKERS = ("_tpu_select_forward_patch", "_tpu_fused_output_reduced_patch",
                "_tpu_fused_ep_pcp_collectives_patch")
_UPSTREAM_SUBCLASSES = {
    "TransformersMoERunner", "LatentMoERunner", "ROCmLatentMoERunner"
}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def fused_ep_flag():
    """Set ``envs.USE_MOE_FUSED_EP_KERNEL`` for the duration of one test.

    ``envs`` resolves names lazily through a module ``__getattr__``.
    ``monkeypatch.setattr`` would undo itself by assigning the old value back,
    which leaves a concrete module attribute that shadows the lazy lookup for
    every later test in the process. ``patch.object`` deletes it instead.
    """
    patchers = []

    def set_flag(value: bool) -> None:
        patcher = patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", value)
        patcher.start()
        patchers.append(patcher)

    yield set_flag
    for patcher in reversed(patchers):
        patcher.stop()


@pytest.fixture
def tpu_runner(monkeypatch):
    """Register TpuMoERunner for one test; yield it and the bridge stub.

    `register_oot` asserts a name is registered once per process, so whatever
    is registered already (the real class, when the package is importable) is
    set aside and put back afterwards.
    """
    before = {name: upstream.MoERunner.__dict__[name] for name in _OVERRIDES}
    previous = op_registry_oot.pop("MoERunner", None)
    bridge = ModuleType(_BRIDGE)
    bridge.fused_moe_ep_supported = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, _BRIDGE, bridge)
    try:
        module = _load_by_path("_test_tpu_moe_runner", _RUNNER_PATH)
        yield module.TpuMoERunner, bridge.fused_moe_ep_supported
    finally:
        op_registry_oot.pop("MoERunner", None)
        if previous is not None:
            op_registry_oot["MoERunner"] = previous
    # Nothing a test did may leave the upstream class changed either.
    for name, attr in before.items():
        assert upstream.MoERunner.__dict__[name] is attr, name


def _runner(*, shared=False, pcp_size=2, skip=False, upstream_reduced=False):
    """A runner built through `__new__`, which is where the registry acts.

    `__init__` is skipped: it needs a router, experts and a vLLM config. The
    attributes set are the ones the overrides and upstream's getter read.
    """
    runner = upstream.MoERunner.__new__(upstream.MoERunner)
    torch.nn.Module.__init__(runner)
    kernel = (SimpleNamespace(
        output_is_reduced=lambda: True) if upstream_reduced else None)
    runner.routed_experts = SimpleNamespace(quant_method=SimpleNamespace(
        moe_kernel=kernel))
    runner._shared_experts = torch.nn.Identity() if shared else None
    runner.moe_config = SimpleNamespace(pcp_size=pcp_size,
                                        skip_final_all_reduce=skip)
    return runner


def test_registered_under_the_upstream_class_name(tpu_runner):
    cls, _ = tpu_runner
    assert op_registry_oot["MoERunner"] is cls
    # The lookup LoRA's can_replace_layer uses.
    assert maybe_get_oot_by_class(upstream.MoERunner) is cls
    assert issubclass(cls, upstream.MoERunner)
    # Guard against vacuous checks below: all four really are overridden.
    for name in _OVERRIDES:
        assert name in cls.__dict__, name
        assert cls.__dict__[name] is not upstream.MoERunner.__dict__[name]


def test_registration_does_not_mutate_upstream_moe_runner():
    """The point of registering: vLLM's class stays vLLM's."""
    before = dict(upstream.MoERunner.__dict__)
    previous = op_registry_oot.pop("MoERunner", None)
    try:
        _load_by_path("_test_tpu_moe_runner_identity", _RUNNER_PATH)
    finally:
        op_registry_oot.pop("MoERunner", None)
        if previous is not None:
            op_registry_oot["MoERunner"] = previous
    after = dict(upstream.MoERunner.__dict__)
    assert after.keys() == before.keys()
    for name in _OVERRIDES:
        assert after[name] is before[name], name
    for marker in _OLD_MARKERS:
        assert not hasattr(upstream.MoERunner, marker), marker


def test_upstream_constructor_builds_tpu_runner(tpu_runner, monkeypatch):
    """`MoERunner(...)`, as `FusedMoE` calls it, runs upstream's `__init__` on
    our class, so `_forward_entry` is chosen by our `_select_forward`."""
    cls, _ = tpu_runner
    context = SimpleNamespace(static_forward_context={},
                              static_all_moe_layers=[])
    monkeypatch.setattr(upstream, "get_current_vllm_config",
                        lambda: SimpleNamespace(compilation_config=context))
    runner = upstream.MoERunner("test.moe", SimpleNamespace(), object(),
                                SimpleNamespace())
    assert type(runner) is cls
    assert isinstance(runner, upstream.MoERunner)
    assert runner._forward_entry is upstream._moe_forward
    assert context.static_forward_context == {"test.moe": runner}
    assert context.static_all_moe_layers == ["test.moe"]


@pytest.mark.parametrize("shared", [False, True])
def test_select_forward_is_the_direct_function(tpu_runner, shared):
    cls, _ = tpu_runner
    runner = _runner(shared=shared)
    assert type(runner) is cls
    expected = (upstream._moe_forward_shared
                if shared else upstream._moe_forward)
    assert runner._select_forward() is expected
    # The op wrappers are distinct objects, so `is` above cannot pass on them.
    assert runner._select_forward() is not torch.ops.vllm.moe_forward
    assert runner._select_forward() is not torch.ops.vllm.moe_forward_shared


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("upstream_reduced", [False, True])
def test_fused_output_is_reduced_truth_table(tpu_runner, fused_ep_flag,
                                             enabled, armed, skip,
                                             upstream_reduced):
    _, supported = tpu_runner
    supported.return_value = armed
    fused_ep_flag(enabled)
    runner = _runner(skip=skip, upstream_reduced=upstream_reduced)

    # The stand-in really drives upstream's own getter.
    assert upstream.MoERunner._fused_output_is_reduced.fget(runner) is (
        upstream_reduced)

    assert runner._fused_output_is_reduced is (upstream_reduced
                                               or (enabled and armed
                                                   and not skip))
    if not enabled or skip:
        # Upstream's answer, without asking whether the layer is armed.
        supported.assert_not_called()
    else:
        # Arming is per layer: asked of this layer's own quant method.
        supported.assert_called_once_with(runner._quant_method)


@pytest.mark.parametrize("upstream_reduced", [False, True])
def test_fused_output_is_reduced_without_the_bridge(tpu_runner, monkeypatch,
                                                    fused_ep_flag,
                                                    upstream_reduced):
    """If the bridge cannot be imported, the answer is upstream's."""
    fused_ep_flag(True)
    # `None` in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, _BRIDGE, None)
    runner = _runner(upstream_reduced=upstream_reduced)
    assert runner._fused_output_is_reduced is upstream_reduced


@pytest.mark.parametrize("shared", [False, True])
def test_armed_pcp_layer_bypasses_outer_collectives(tpu_runner, monkeypatch,
                                                    shared):
    _, supported = tpu_runner
    supported.return_value = True
    forbidden = Mock(side_effect=AssertionError("duplicate PCP collective"))
    monkeypatch.setattr(upstream, "get_pcp_group", forbidden)
    monkeypatch.setattr(upstream, "get_ep_group", forbidden)
    runner = _runner(shared=shared, pcp_size=2)
    hidden, logits, shared_output = (torch.randn(2, 4) for _ in range(3))

    dispatched = runner._maybe_dispatch(hidden, logits)
    assert isinstance(dispatched, tuple) and len(dispatched) == 2
    assert dispatched[0] is hidden
    assert dispatched[1] is logits

    combined = runner._maybe_combine(shared_output, hidden)
    if shared:
        # Upstream's shape when a shared expert was computed separately.
        assert isinstance(combined, tuple) and len(combined) == 2
        assert combined[0] is shared_output
        assert combined[1] is hidden
    else:
        assert combined is hidden
    forbidden.assert_not_called()
    supported.assert_called_with(runner._quant_method)


@pytest.mark.parametrize("pcp_size,armed", [(1, True), (2, False), (1, False)])
def test_other_layers_delegate_dispatch_and_combine(tpu_runner, monkeypatch,
                                                    pcp_size, armed):
    _, supported = tpu_runner
    supported.return_value = armed
    runner = _runner(pcp_size=pcp_size)
    # Observe delegation on the real parent class; undone before the fixture
    # checks that the upstream class is unchanged.
    with monkeypatch.context() as m:
        dispatch = Mock(return_value=("dispatched", "logits"))
        combine = Mock(return_value="combined")
        m.setattr(upstream.MoERunner, "_maybe_dispatch", dispatch)
        m.setattr(upstream.MoERunner, "_maybe_combine", combine)
        assert runner._maybe_dispatch("hidden",
                                      "logits") == ("dispatched", "logits")
        assert runner._maybe_combine("shared", "hidden") == "combined"
        dispatch.assert_called_once_with("hidden", "logits")
        combine.assert_called_once_with("shared", "hidden")
    if pcp_size <= 1:
        # Without PCP there is no outer pair to own, so arming is not asked.
        supported.assert_not_called()


def test_upstream_subclasses_of_moe_runner_are_the_known_ones():
    """Name-keyed registration does not cover subclasses; pin which exist."""
    found = set()
    for path in Path(vllm.__file__).parent.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "MoERunner" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.id if isinstance(base, ast.Name) else getattr(
                    base, "attr", None)
                for base in node.bases
            }
            if "MoERunner" in bases:
                found.add(node.name)
    assert found == _UPSTREAM_SUBCLASSES, (
        f"vLLM's direct subclasses of MoERunner changed: {sorted(found)} vs "
        f"{sorted(_UPSTREAM_SUBCLASSES)}. MoERunner.register_oot replaces "
        "only the exact class name, so TpuMoERunner's overrides do not reach "
        "a subclass. For each new one, decide whether it can be constructed "
        "on TPU (which model passes it as runner_cls, and does this plugin "
        "serve that model through upstream's implementation?). If it can, it "
        "needs its own out-of-tree registration carrying the same overrides. "
        "Then update this set and the TpuMoERunner docstring.")


def test_layers_package_imports_the_runner_module():
    """The one line that registers TpuMoERunner in every process: both
    `load_general_plugins()` (entry point `vllm_torchtpu.layers`) and the
    `_register_custom_ops` patch callback import this package first."""
    tree = ast.parse(_LAYERS_INIT_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body if isinstance(node, ast.ImportFrom)
        and node.module == "vllm_torchtpu.layers.adapter" and node.level == 0
        for alias in node.names
    }
    assert "moe_runner" in imported, sorted(imported)


def test_patch_registry_no_longer_lists_the_moe_runner_patches():
    registry = _load_by_path("_test_tpu_moe_runner_registry", _REGISTRY_PATH)
    targets = [patch.target for patch in registry.PATCHES]
    for removed in ("_patch_default_moe_runner_select_forward",
                    "_patch_moe_runner_fused_output_is_reduced"):
        assert not any(removed in target for target in targets), removed
    # Its process-wide half is still a patch, and still registered.
    assert "vllm_torchtpu:_patch_moe_explicit_pcp_collectives" in targets


def test_remaining_pcp_patch_leaves_moe_runner_alone(monkeypatch):
    """`_patch_moe_explicit_pcp_collectives` now only replaces
    `FusedMoEParallelConfig.use_all2all_kernels`."""
    import vllm_torchtpu

    for removed in ("_patch_default_moe_runner_select_forward",
                    "_patch_moe_runner_fused_output_is_reduced"):
        assert not hasattr(vllm_torchtpu, removed), removed

    before = dict(upstream.MoERunner.__dict__)
    marker = "_tpu_explicit_pcp_collectives_patch"
    # Restore the dataclass after the test, whether or not it was patched.
    monkeypatch.setattr(FusedMoEParallelConfig, "use_all2all_kernels",
                        FusedMoEParallelConfig.use_all2all_kernels)
    if hasattr(FusedMoEParallelConfig, marker):
        monkeypatch.setattr(FusedMoEParallelConfig, marker, True)
    else:
        monkeypatch.setattr(FusedMoEParallelConfig,
                            marker,
                            False,
                            raising=False)

    vllm_torchtpu._patch_moe_explicit_pcp_collectives()

    assert getattr(FusedMoEParallelConfig, marker) is True
    assert dict(upstream.MoERunner.__dict__) == before
    for old_marker in _OLD_MARKERS:
        assert not hasattr(upstream.MoERunner, old_marker), old_marker
