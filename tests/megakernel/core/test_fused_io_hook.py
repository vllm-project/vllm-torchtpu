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
"""Unit tests for `fused_io_hook` registration and lookup."""

from __future__ import annotations

import pytest

from vllm_torchtpu.megakernel.core import _builtins, fused_io_hook
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError


def test_register_and_lookup_fused_io_impl(monkeypatch):
    """`register_fused_io` installs an implementation returned by `fused_io()`."""
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", None)
    assert not fused_io_hook.fused_io_available()

    sentinel = object()
    fused_io_hook.register_fused_io(sentinel)  # type: ignore[arg-type]
    assert fused_io_hook.fused_io_available()
    assert fused_io_hook.fused_io() is sentinel


def test_fused_io_without_implementation_raises_framework_neutral_error(monkeypatch):
    """`fused_io()` raises `MegakernelConfigError` with a framework-neutral message."""
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", None)
    assert not fused_io_hook.fused_io_available()
    with pytest.raises(MegakernelConfigError, match="fused pool I/O") as err:
        fused_io_hook.fused_io()
    assert "vllm" not in str(err.value)


def test_fused_io_lookup_calls_ensure_builtins(monkeypatch):
    """First lookup requests `_builtins.ensure_builtins(*FUSED_IO)`."""
    called: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        fused_io_hook, "ensure_builtins", lambda *names: called.append(names)
    )
    sentinel = object()
    monkeypatch.setattr(fused_io_hook, "_IMPL", sentinel)
    assert fused_io_hook.fused_io_available()
    assert fused_io_hook.fused_io() is sentinel
    assert called == [fused_io_hook.FUSED_IO, fused_io_hook.FUSED_IO]
