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

``VllmConfig.compile_factors()`` folds a dict ``additional_config`` in whole,
and the hash of those factors feeds both the AOT compile cache key and the
piecewise cache dir. The phased profiler's remaining ``additional_config``
knobs only steer profiling, so letting them into the hash would force
needless recompiles.
"""

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig

from vllm_torchtpu import _patch_vllm_config_hash_ignore_diagnostics
from vllm_torchtpu.runner.utils import (
    HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
    PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
    PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,
)

KEY = PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY
PREFILL_KEY = PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY

# Class attributes the patch stamps onto VllmConfig, which monkeypatch cannot
# roll back on its own because they did not exist beforehand.
_PATCH_STAMPS = (
    "_tpu_additional_config_hash_patch",
    "_tpu_upstream_compile_factors",
)


def _upstream_compile_factors():
    """The genuine implementation, whether or not the patch already ran.

    Another test may have built a VllmConfig, which runs
    ``TpuPlatform.check_and_update_config()`` ->
    ``patch_registry.apply("platform_activation")``.
    """
    return (
        getattr(VllmConfig, "_tpu_upstream_compile_factors", None)
        or VllmConfig.__dict__["compile_factors"]
    )


class _StubVllmConfig:
    """Minimal stand-in for what ``compile_factors()`` reads off a VllmConfig.

    A real VllmConfig needs a fully materialized ModelConfig to construct.
    ``compile_factors()`` maps a ``None`` sub-config to ``{}`` and skips the
    multimodal factors when ``model_config`` is ``None``.
    """

    def __init__(self, additional_config=None):
        self.additional_config = additional_config

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None


@pytest.fixture
def apply_patch_over(monkeypatch):
    """Re-apply the patch on top of a chosen trunk, then undo it."""

    def _apply(trunk):
        monkeypatch.setattr(VllmConfig, "compile_factors", trunk)
        # Platform activation may already have run in this process; the patch
        # is idempotent by design, so clear its stamps to re-wrap the trunk.
        for stamp in _PATCH_STAMPS:
            monkeypatch.delattr(VllmConfig, stamp, raising=False)
        _patch_vllm_config_hash_ignore_diagnostics()
        return VllmConfig.compile_factors

    yield _apply

    for stamp in _PATCH_STAMPS:
        if stamp in VllmConfig.__dict__:
            delattr(VllmConfig, stamp)


@pytest.fixture
def patched_compile_factors(apply_patch_over):
    return apply_patch_over(_upstream_compile_factors())


def test_unpatched_factors_change_with_the_threshold():
    """Guards the premise: without the carve-out this busts the cache."""
    upstream = _upstream_compile_factors()

    assert upstream(_StubVllmConfig({KEY: 128})) != upstream(
        _StubVllmConfig({KEY: 256})
    )


def test_threshold_no_longer_changes_the_factors(patched_compile_factors):
    """Retuning a profiling threshold must not force a recompile."""
    assert patched_compile_factors(
        _StubVllmConfig({KEY: 128})
    ) == patched_compile_factors(_StubVllmConfig({KEY: 256}))


def test_setting_the_threshold_reuses_the_unprofiled_cache(patched_compile_factors):
    """Adding the knob at all must not invalidate a cache built without it."""
    baseline = patched_compile_factors(_StubVllmConfig())

    assert patched_compile_factors(_StubVllmConfig({KEY: 128})) == baseline


def test_non_diagnostic_keys_still_change_the_factors(patched_compile_factors):
    """Only the profiling knob is carved out; the rest still count."""
    assert patched_compile_factors(
        _StubVllmConfig({"some_plugin_knob": 1, KEY: 128})
    ) != patched_compile_factors(_StubVllmConfig({"some_plugin_knob": 2, KEY: 128}))


def test_untouched_configs_keep_their_upstream_factors(patched_compile_factors):
    config = _StubVllmConfig({"some_plugin_knob": 1})

    assert patched_compile_factors(config) == _upstream_compile_factors()(config)


@pytest.mark.parametrize("additional_config", [None, {"option": 1}, {KEY: 128}])
@pytest.mark.parametrize("include_version", [False, True])
def test_factors_forward_include_version(
    patched_compile_factors, additional_config, include_version
):
    config = _StubVllmConfig(additional_config)
    filtered = _StubVllmConfig(
        None if additional_config == {KEY: 128} else additional_config
    )
    expected = _upstream_compile_factors()(filtered, include_version=include_version)
    assert patched_compile_factors(config, include_version=include_version) == expected
    assert patched_compile_factors(config, include_version) == expected


def test_additional_config_is_left_unchanged(patched_compile_factors):
    additional_config = {"some_plugin_knob": 1, KEY: 128}
    config = _StubVllmConfig(additional_config)

    patched_compile_factors(config)

    assert config.additional_config is additional_config
    assert additional_config == {"some_plugin_knob": 1, KEY: 128}


def test_compile_factors_additional_config_is_left_alone(patched_compile_factors):
    """``additional_config`` may be a SupportsCompileFactors object."""
    additional_config = SimpleNamespace(compile_factors=lambda: {KEY: "custom"})
    config = _StubVllmConfig(additional_config)

    assert patched_compile_factors(config) == _upstream_compile_factors()(config)


def test_patch_is_idempotent(patched_compile_factors):
    _patch_vllm_config_hash_ignore_diagnostics()

    assert VllmConfig.compile_factors is patched_compile_factors


def test_ignored_keys_match_the_keys_the_runner_reads():
    assert {KEY, PREFILL_KEY} == HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
