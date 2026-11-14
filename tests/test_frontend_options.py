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
"""additional_config["tpu_frontend"]: strict parsing, off by default, and each
option installs exactly its own API-server patch from the platform hook."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu import frontend_fix, frontend_options, frontend_tokcache
from vllm_torchtpu.frontend_options import FrontendOptions

_INSTALLERS = {
    "tokenize_prefix_cache": (frontend_tokcache, "maybe_install_tokenize_prefix_cache"),
    "lazy_reasoning_count": (frontend_fix, "maybe_install_lazy_reasoning_count"),
    "detok_prompt_tail": (frontend_fix, "maybe_install_detok_prompt_tail"),
}


def _config(section=None):
    additional = {} if section is None else {frontend_options.SECTION: section}
    return SimpleNamespace(additional_config=additional)


@pytest.fixture
def installed(monkeypatch):
    """Record which installers ran instead of patching vLLM."""
    calls = []
    for name, (module, attr) in _INSTALLERS.items():
        monkeypatch.setattr(module, attr, lambda name=name: calls.append(name))
    return calls


def test_defaults_are_all_off():
    assert frontend_options.parse(None) == FrontendOptions()
    assert frontend_options.parse({}) == FrontendOptions()
    assert not any(vars(FrontendOptions()).values())


def test_get_reads_the_section():
    options = frontend_options.get(_config({"lazy_reasoning_count": True}))
    assert options == FrontendOptions(lazy_reasoning_count=True)
    assert frontend_options.get(SimpleNamespace(additional_config=None)) == (
        FrontendOptions()
    )


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"tokenize_prefix_cache": 1}, "must be a bool"),
        ({"tokenize_prefix_cache": "true"}, "must be a bool"),
        ({"tokenize_prefix_cach": True}, "unknown keys"),
        ([True], "must be a JSON object"),
    ],
)
def test_parse_is_strict(raw, message):
    with pytest.raises(ValueError, match=message):
        frontend_options.parse(raw)


def test_install_by_default_patches_nothing(installed):
    assert frontend_options.install(_config()) == FrontendOptions()
    assert installed == []


@pytest.mark.parametrize("name", sorted(_INSTALLERS))
def test_each_option_installs_only_its_patch(installed, name):
    frontend_options.install(_config({name: True}))
    assert installed == [name]


def test_failed_install_does_not_stop_startup(monkeypatch):
    def broken():
        raise RuntimeError("vLLM moved")

    monkeypatch.setattr(frontend_fix, "maybe_install_detok_prompt_tail", broken)
    options = frontend_options.install(_config({"detok_prompt_tail": True}))
    assert options.detok_prompt_tail


def test_section_is_kept_out_of_the_compile_cache_key():
    from vllm_torchtpu.runner.utils import (
        FRONTEND_OPTIONS_SECTION,
        HASH_IGNORED_ADDITIONAL_CONFIG_KEYS,
    )

    assert frontend_options.SECTION == FRONTEND_OPTIONS_SECTION
    assert FRONTEND_OPTIONS_SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


def _platform_patches():
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    return (
        patch("vllm_torchtpu.patch_registry.apply"),
        patch.object(TpuPlatform, "_prepare_singlehost_tpu_env"),
        # Keep DeviceConfig from building torch.device("tpu") off a TPU.
        patch.object(TpuPlatform, "uses_host_device_handling", return_value=True),
    )


def test_platform_hook_installs_from_the_engine_config(installed):
    from vllm.config import VllmConfig

    p1, p2, p3 = _platform_patches()
    with p1, p2, p3:
        VllmConfig(additional_config={"tpu_frontend": {"tokenize_prefix_cache": True}})
    assert installed == ["tokenize_prefix_cache"]


def test_platform_hook_rejects_an_unknown_option(installed):
    from vllm.config import VllmConfig

    p1, p2, p3 = _platform_patches()
    with p1, p2, p3, pytest.raises(ValueError, match="unknown keys"):
        VllmConfig(additional_config={"tpu_frontend": {"tokenize_cache": True}})
    assert installed == []
