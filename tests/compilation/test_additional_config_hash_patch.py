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
"""The diagnostics-only additional_config carve-out for compile cache keys.

``VllmConfig.compute_hash()`` folds the whole ``additional_config`` dict in,
and that hash feeds both the AOT compile cache key and the piecewise cache dir.
The phased profiler is configured through ``additional_config`` with a per-run
trace directory, which would otherwise force a full recompile on every start.
"""

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig

from vllm_torchtpu import _patch_vllm_config_hash_ignore_diagnostics
from vllm_torchtpu.runner.utils import (
    HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
    PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY, PHASED_PROFILING_DIR_KEY)

# Class attributes the patch stamps onto VllmConfig, which monkeypatch cannot
# roll back on its own because they did not exist beforehand.
_PATCH_STAMPS = ("_tpu_additional_config_hash_patch",
                 "_tpu_upstream_compute_hash")


def _upstream_compute_hash():
    """The genuine implementation, whether or not the patch already ran.

    Another test may have built a VllmConfig, which runs
    ``TpuPlatform.check_and_update_config()`` -> ``apply_tpu_patches()``.
    """
    return getattr(VllmConfig, "_tpu_upstream_compute_hash",
                   None) or VllmConfig.__dict__["compute_hash"]


class _StubVllmConfig:
    """Minimal stand-in for what ``compute_hash()`` reads off a VllmConfig.

    A real VllmConfig needs a fully materialized ModelConfig to construct.
    ``compute_hash()`` guards every sub-config with ``if self.x:``, so ``None``
    sends them all down the "absent" branch -- except ``observability_config``,
    which is dereferenced unconditionally.
    """

    def __init__(self, additional_config=None):
        self.additional_config = additional_config
        self.observability_config = SimpleNamespace(
            compute_hash=lambda: "observability")

    def __getattr__(self, name):
        # Let dunders (``__setstate__``/``__deepcopy__``/...) miss normally so
        # copy.copy() -- which the patch uses -- falls back to its defaults.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None


@pytest.fixture
def apply_patch_over(monkeypatch):
    """Re-apply the patch on top of a chosen trunk, then undo it."""

    def _apply(trunk):
        monkeypatch.setattr(VllmConfig, "compute_hash", trunk)
        # apply_tpu_patches() may already have run in this process; the patch
        # is idempotent by design, so clear its stamps to re-wrap the trunk.
        for stamp in _PATCH_STAMPS:
            monkeypatch.delattr(VllmConfig, stamp, raising=False)
        _patch_vllm_config_hash_ignore_diagnostics()
        return VllmConfig.compute_hash

    yield _apply

    for stamp in _PATCH_STAMPS:
        if stamp in VllmConfig.__dict__:
            delattr(VllmConfig, stamp)


@pytest.fixture
def patched_compute_hash(apply_patch_over):
    return apply_patch_over(_upstream_compute_hash())


def test_unpatched_hash_changes_with_the_profile_dir():
    """Guards the premise: without the carve-out this busts the cache."""
    assert (_upstream_compute_hash()(_StubVllmConfig(
        {PHASED_PROFILING_DIR_KEY: "/tmp/profile_0723"}))
            != _upstream_compute_hash()(_StubVllmConfig(
                {PHASED_PROFILING_DIR_KEY: "/tmp/profile_0724"})))


def test_profile_dir_no_longer_changes_the_hash(patched_compute_hash):
    """The PR's own usage example: a dated dir must not force a recompile."""
    assert (patched_compute_hash(
        _StubVllmConfig({PHASED_PROFILING_DIR_KEY:
                         "/tmp/profile_0723"})) == patched_compute_hash(
                             _StubVllmConfig({
                                 PHASED_PROFILING_DIR_KEY:
                                 "/tmp/profile_0724"
                             })))


def test_enabling_phased_profiling_reuses_the_unprofiled_cache(
        patched_compute_hash):
    baseline = patched_compute_hash(_StubVllmConfig())

    assert patched_compute_hash(
        _StubVllmConfig({
            PHASED_PROFILING_DIR_KEY: "/tmp/phased",
            PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY: 128,
        })) == baseline


def test_non_diagnostic_keys_still_change_the_hash(patched_compute_hash):
    """Only the profiling knobs are carved out; the rest still count."""
    assert (patched_compute_hash(
        _StubVllmConfig({
            "some_plugin_knob": 1,
            PHASED_PROFILING_DIR_KEY: "/tmp/phased",
        })) != patched_compute_hash(
            _StubVllmConfig({
                "some_plugin_knob": 2,
                PHASED_PROFILING_DIR_KEY: "/tmp/phased",
            })))


def test_untouched_configs_keep_their_upstream_hash(patched_compute_hash):
    config = _StubVllmConfig({"some_plugin_knob": 1})

    assert patched_compute_hash(config) == _upstream_compute_hash()(config)


def test_additional_config_is_restored_after_hashing(patched_compute_hash):
    additional_config = {PHASED_PROFILING_DIR_KEY: "/tmp/phased"}
    config = _StubVllmConfig(additional_config)

    patched_compute_hash(config)

    assert config.additional_config is additional_config


def test_additional_config_is_restored_when_hashing_raises(apply_patch_over):

    def exploding_compute_hash(self):
        raise RuntimeError("boom")

    compute_hash = apply_patch_over(exploding_compute_hash)
    additional_config = {PHASED_PROFILING_DIR_KEY: "/tmp/phased"}
    config = _StubVllmConfig(additional_config)

    with pytest.raises(RuntimeError):
        compute_hash(config)

    assert config.additional_config is additional_config


def test_supports_hash_additional_config_is_left_alone(patched_compute_hash):
    """``additional_config`` may be a SupportsHash object rather than a dict."""
    additional_config = SimpleNamespace(compute_hash=lambda: "custom")
    config = _StubVllmConfig(additional_config)

    assert patched_compute_hash(config) == _upstream_compute_hash()(config)
    assert config.additional_config is additional_config


def test_patch_is_idempotent(patched_compute_hash):
    _patch_vllm_config_hash_ignore_diagnostics()

    assert VllmConfig.compute_hash is patched_compute_hash


def test_ignored_keys_match_the_keys_the_runner_reads():
    assert HASH_IGNORED_ADDITIONAL_CONFIG_KEYS == {
        PHASED_PROFILING_DIR_KEY,
        PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
    }
