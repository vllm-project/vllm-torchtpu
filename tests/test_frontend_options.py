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
option installs exactly its own API-server patch through patch_registry."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import VllmConfig

from vllm_torchtpu import (
    frontend_fix,
    frontend_options,
    frontend_tokcache,
    patch_registry,
)
from vllm_torchtpu.frontend_options import FrontendOptions
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform
from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

_INSTALLERS = {
    "tokenize_prefix_cache": (frontend_tokcache, "maybe_install_tokenize_prefix_cache"),
    "detok_prompt_tail": (frontend_fix, "maybe_install_detok_prompt_tail"),
}
# The patch_registry callback of each option.
_CALLBACKS = {name: f"install_{name}" for name in _INSTALLERS}


def _run_callbacks(config):
    """Call every option's registry callback the way patch_registry does;
    returns their results by option."""
    return {
        name: getattr(frontend_options, callback)(vllm_config=config)
        for name, callback in _CALLBACKS.items()
    }


def _config(section=None):
    additional = {} if section is None else {frontend_options.SECTION: section}
    return SimpleNamespace(additional_config=additional)


@pytest.fixture
def installed(monkeypatch):
    """Record which installers ran instead of patching vLLM."""
    calls = []
    for name, (module, attr) in _INSTALLERS.items():
        monkeypatch.setattr(
            module, attr, lambda name=name, **kwargs: calls.append(name)
        )
    return calls


def test_defaults_are_all_off():
    assert frontend_options.parse(None) == FrontendOptions()
    assert frontend_options.parse({}) == FrontendOptions()
    switches = {k: v for k, v in vars(FrontendOptions()).items() if type(v) is bool}
    assert switches and not any(switches.values())


def test_cache_keeps_256m_token_ids_by_default_and_honours_the_option(monkeypatch):
    """256M is the default size; a deployment can lower it (its prefill legs
    ran 64M)."""
    assert FrontendOptions().tokenize_prefix_cache_max_tokens == 256_000_000
    seen = []
    monkeypatch.setattr(
        frontend_tokcache,
        "maybe_install_tokenize_prefix_cache",
        lambda **kwargs: seen.append(kwargs),
    )
    on = {"tokenize_prefix_cache": True}
    frontend_options.install_tokenize_prefix_cache(vllm_config=_config(on))
    frontend_options.install_tokenize_prefix_cache(
        vllm_config=_config({**on, "tokenize_prefix_cache_max_tokens": 64_000_000})
    )
    assert seen == [{"max_tokens": 256_000_000}, {"max_tokens": 64_000_000}]


def test_get_reads_the_section():
    options = frontend_options.get(_config({"detok_prompt_tail": True}))
    assert options == FrontendOptions(detok_prompt_tail=True)
    assert frontend_options.get(SimpleNamespace(additional_config=None)) == (
        FrontendOptions()
    )


def test_get_needs_an_engine_config():
    """An object without additional_config (a model config passed by mistake,
    or an upstream rename) fails instead of reading as all options off."""
    with pytest.raises(AttributeError, match="additional_config"):
        frontend_options.get(SimpleNamespace(model="Qwen/Qwen3.5-397B-A17B"))


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"tokenize_prefix_cache": 1}, "must be a bool"),
        ({"tokenize_prefix_cache": "true"}, "must be a bool"),
        ({"tokenize_prefix_cach": True}, "unknown keys"),
        ([True], "must be a JSON object"),
        ({"tokenize_prefix_cache_max_tokens": "64M"}, "must be a int"),
        ({"tokenize_prefix_cache_max_tokens": True}, "must be a int"),
        ({"tokenize_prefix_cache_max_tokens": 0}, "must be positive"),
    ],
)
def test_parse_is_strict(raw, message):
    with pytest.raises(ValueError, match=message):
        frontend_options.parse(raw)


@pytest.mark.parametrize("config", [_config(), None])
def test_by_default_nothing_is_patched_and_every_callback_stays_eligible(
    installed, config
):
    # False keeps a registry entry eligible, so a later engine config built in
    # the same process can still turn its option on.
    assert _run_callbacks(config) == dict.fromkeys(_INSTALLERS, False)
    assert installed == []


@pytest.mark.parametrize("name", sorted(_INSTALLERS))
def test_each_option_installs_only_its_patch(installed, name):
    results = _run_callbacks(_config({name: True}))
    assert installed == [name]
    # The installed option is complete; the others stay eligible.
    assert results == {other: None if other == name else False for other in results}


def test_failed_install_does_not_stop_startup(monkeypatch):
    def broken():
        raise RuntimeError("vLLM moved")

    monkeypatch.setattr(frontend_fix, "maybe_install_detok_prompt_tail", broken)
    config = _config({"detok_prompt_tail": True})
    assert frontend_options.install_detok_prompt_tail(vllm_config=config) is None


def test_each_callback_is_registered_for_platform_activation_with_the_config():
    entries = {
        p.target: p
        for p in patch_registry.PATCHES
        if p.target.startswith("vllm_torchtpu.frontend_options:")
    }
    assert sorted(entries) == sorted(
        f"vllm_torchtpu.frontend_options:{cb}" for cb in _CALLBACKS.values()
    )
    for entry in entries.values():
        assert entry.stages == ("platform_activation",)
        assert entry.vllm_config and not entry.model_config


def test_section_is_kept_out_of_the_compile_cache_key():
    assert frontend_options.SECTION in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS


def _platform_patches():
    return (
        patch("vllm_torchtpu.patch_registry.apply"),
        patch.object(TpuPlatform, "_prepare_singlehost_tpu_env"),
        # Keep DeviceConfig from building torch.device("tpu") off a TPU.
        patch.object(TpuPlatform, "uses_host_device_handling", return_value=True),
    )


def test_platform_hook_applies_the_registry_with_the_engine_config():
    p1, p2, p3 = _platform_patches()
    with p1 as apply, p2, p3:
        config = VllmConfig(
            additional_config={"tpu_frontend": {"tokenize_prefix_cache": True}}
        )
    calls = [c for c in apply.call_args_list if c.args == ("platform_activation",)]
    assert calls and all(c.kwargs["vllm_config"] is config for c in calls)


@pytest.mark.parametrize(
    "additional",
    [
        {"tpu_frontend": {"tokenize_cache": True}},
        {"tpu_worker": {"numa": True}},
        {"tpu_worker": {"numa_bind": 1}},
    ],
)
def test_platform_hook_rejects_a_bad_host_only_section(installed, additional):
    """Before any patch installs and before a worker starts."""
    p1, p2, p3 = _platform_patches()
    with p1 as apply, p2, p3, pytest.raises(ValueError, match="additional_config"):
        VllmConfig(additional_config=additional)
    apply.assert_not_called()
    assert installed == []
