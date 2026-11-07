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
"""_patch_skip_unread_aot_autograd_save: torch_tpu's never-read
AOTAutogradCache entries are not written, every other save still is."""

from types import SimpleNamespace

import pytest

import vllm_torchtpu


@pytest.fixture
def autograd_cache(monkeypatch):
    """AOTAutogradCache with a recording save in place of the real one."""
    from torch._functorch._aot_autograd import autograd_cache as ac

    calls = []

    def recording_save(key, entry, remote):
        calls.append((key, remote))

    monkeypatch.setattr(ac.AOTAutogradCache, "save", staticmethod(recording_save))
    vllm_torchtpu._patch_skip_unread_aot_autograd_save()
    return ac, calls


def _entry(precompile_backend_id=None):
    return SimpleNamespace(
        sanitized_aot_config=SimpleNamespace(
            precompile_backend_id=precompile_backend_id
        )
    )


@pytest.mark.parametrize(
    "key, bundling, backend_id, remote, saved",
    [
        ("torchtpu_abc", False, None, False, False),  # torch_tpu key, no reader
        ("torchtpu_abc", True, None, False, False),  # capture-only bundling
        ("torchtpu_abc", True, "pc0", False, True),  # precompile artifact
        ("torchtpu_abc", False, None, True, True),  # remote cache
        ("aabc123", False, None, False, True),  # torch's own key: looked up
    ],
)
def test_save_is_skipped_only_when_nothing_reads_it(
    monkeypatch, autograd_cache, key, bundling, backend_id, remote, saved
):
    ac, calls = autograd_cache
    monkeypatch.setattr(ac, "should_bundle_autograd_cache", lambda: bundling)
    # torch calls save(key, entry, remote=...); call it the same way.
    ac.AOTAutogradCache.save(key, _entry(backend_id), remote=remote)
    assert calls == ([(key, remote)] if saved else [])


def test_patched_save_stays_a_staticmethod(autograd_cache):
    ac, _ = autograd_cache
    assert isinstance(ac.AOTAutogradCache.__dict__["save"], staticmethod)


def test_applying_twice_wraps_once(autograd_cache):
    ac, _ = autograd_cache
    first = ac.AOTAutogradCache.__dict__["save"]
    vllm_torchtpu._patch_skip_unread_aot_autograd_save()
    assert ac.AOTAutogradCache.__dict__["save"] is first


def test_still_wraps_once_after_torch_tpu_rewraps_it(monkeypatch, autograd_cache):
    """torch_tpu restores the method as staticmethod(<the function it found>)."""
    ac, calls = autograd_cache
    patched_fn = ac.AOTAutogradCache.save
    ac.AOTAutogradCache.save = staticmethod(patched_fn)
    vllm_torchtpu._patch_skip_unread_aot_autograd_save()
    assert ac.AOTAutogradCache.__dict__["save"].__func__ is patched_fn
    monkeypatch.setattr(ac, "should_bundle_autograd_cache", lambda: False)
    ac.AOTAutogradCache.save("aabc123", _entry(), remote=False)
    assert calls == [("aabc123", False)]


def test_registered_for_every_process():
    from vllm_torchtpu import patch_registry

    stages = [
        p.stages
        for p in patch_registry.PATCHES
        if p.target == "vllm_torchtpu:_patch_skip_unread_aot_autograd_save"
    ]
    assert stages == [patch_registry._ACTIVE]
