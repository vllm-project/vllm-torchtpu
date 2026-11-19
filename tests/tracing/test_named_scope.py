import contextlib

import pytest

from vllm_torchtpu.tracing import named_scope as ns

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def entered(monkeypatch):
    """Replace TorchTPU's scope with one that records the names it enters."""
    names: list[str] = []

    @contextlib.contextmanager
    def fake_scope(name):
        names.append(name)
        yield

    monkeypatch.setattr(ns, "_scope", None)
    monkeypatch.setattr(ns, "_import_scope", lambda: fake_scope)
    return names


def test_scopes_are_noops_before_initialization(entered):
    @ns.scoped("phase")
    def double(x):
        return 2 * x

    with ns.named_scope("outer"):
        assert double(3) == 6
    assert entered == []
    assert not ns.named_scopes_enabled()


def test_initialization_enables_scopes(monkeypatch, entered):
    monkeypatch.setenv("TPU_PROFILER_NAMED_SCOPES", "1")

    @ns.scoped("phase")
    def double(x):
        return 2 * x

    assert ns.initialize_named_scopes()
    with ns.named_scope("outer"):
        assert double(3) == 6
    assert double(4) == 8
    # A function decorated before initialization picks the scope up, and
    # each call enters a new scope.
    assert entered == ["outer", "phase", "phase"]


def test_env_switch_keeps_scopes_off(monkeypatch, entered):
    monkeypatch.setenv("TPU_PROFILER_NAMED_SCOPES", "0")
    assert not ns.initialize_named_scopes()
    with ns.named_scope("outer"):
        pass
    assert entered == []


def test_missing_api_keeps_scopes_off(monkeypatch):
    monkeypatch.setattr(ns, "_scope", None)
    monkeypatch.setattr(ns, "_import_scope", lambda: None)
    monkeypatch.setenv("TPU_PROFILER_NAMED_SCOPES", "1")
    assert not ns.initialize_named_scopes()
    assert isinstance(ns.named_scope("outer"), contextlib.nullcontext)


def test_scoped_keeps_the_function_metadata():
    @ns.scoped("phase")
    def documented(x):
        """Doc."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Doc."
