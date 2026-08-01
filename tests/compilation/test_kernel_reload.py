# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the kernel-iteration hot-reload registry."""

import importlib
import sys
import textwrap

import pytest

from vllm_torchtpu.compilation import kernel_reload


@pytest.fixture(autouse=True)
def _clean_registry():
    saved_live = dict(kernel_reload._LIVE)
    saved_builders = dict(kernel_reload._BUILDERS)
    kernel_reload._LIVE.clear()
    kernel_reload._BUILDERS.clear()
    yield
    kernel_reload._LIVE.clear()
    kernel_reload._LIVE.update(saved_live)
    kernel_reload._BUILDERS.clear()
    kernel_reload._BUILDERS.update(saved_builders)


def _template(x: int, y: int) -> int:
    """Template callable with a real signature."""
    return x + y


def test_dispatcher_copies_function_introspection():
    dispatcher = kernel_reload.make_dispatcher("ns::op_a", _template)

    assert dispatcher.__name__ == "_template"
    assert dispatcher.__globals__ is _template.__globals__
    assert dispatcher.__annotations__ == _template.__annotations__
    # inspect.signature must resolve the template's signature (via
    # __wrapped__), not __call__(*args, **kwargs) — torch.library's
    # infer_schema rejects varargs signatures.
    import inspect
    assert list(inspect.signature(dispatcher).parameters) == ["x", "y"]


def test_dispatcher_follows_live_swaps():
    kernel_reload.set_live("ns::op_b", lambda x, y: x + y)
    dispatcher = kernel_reload.make_dispatcher("ns::op_b", _template)

    assert dispatcher(2, 3) == 5
    kernel_reload.set_live("ns::op_b", lambda x, y: x * y)
    assert dispatcher(2, 3) == 6


def test_reload_kernels_reimports_and_rebuilds(tmp_path, monkeypatch):
    mod_file = tmp_path / "reloadable_kernel_mod.py"
    mod_file.write_text(
        textwrap.dedent("""
        def kernel(x):
            return x + 1
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("reloadable_kernel_mod")
    try:

        def builder():
            fresh = importlib.import_module("reloadable_kernel_mod")
            return fresh.kernel

        kernel_reload.set_live("ns::op_c", mod.kernel)
        kernel_reload.register_builder("ns::op_c", builder)
        dispatcher = kernel_reload.make_dispatcher("ns::op_c", mod.kernel)
        assert dispatcher(1) == 2

        mod_file.write_text(
            textwrap.dedent("""
            def kernel(x):
                return x + 100
            """))
        stats = kernel_reload.reload_kernels(modules=["reloadable_kernel_mod"])

        assert stats["rebuilt_ops"] == ["ns::op_c"]
        assert stats["reloaded_modules"] == ["reloadable_kernel_mod"]
        assert dispatcher(1) == 101
    finally:
        sys.modules.pop("reloadable_kernel_mod", None)


def test_reload_module_list_env_override(monkeypatch):
    monkeypatch.setenv("TPU_KERNEL_RELOAD_MODULES", "pkg.mod_a, pkg.mod_b")
    assert kernel_reload.default_reload_modules() == ("pkg.mod_a", "pkg.mod_b")

    monkeypatch.delenv("TPU_KERNEL_RELOAD_MODULES")
    assert (kernel_reload.default_reload_modules() ==
            kernel_reload._DEFAULT_RELOAD_MODULES)


def test_reload_skips_unimported_modules():
    stats = kernel_reload.reload_kernels(modules=["never.imported.module"])
    assert stats["rebuilt_ops"] == []
    assert stats["reloaded_modules"] == ["never.imported.module"]
