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
"""TPU tests for vllm_omni.plugins entry point discovery and filtering."""

from importlib.metadata import entry_points

import pytest
import vllm.envs as vllm_envs
import vllm_omni.plugins as omni_plugins
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_omni.plugins import (
    OMNI_PLATFORM_PLUGINS_GROUP,
    load_omni_general_plugins,
    load_omni_plugins_by_group,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_EXPECTED_ENTRYPOINT = "vllm_torchtpu.omni:register_omni_tpu_platform"
_EXPECTED_QUALNAME = "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_platform_plugins_entrypoint_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Platform Plugins Entrypoint Extension

    - TPU path: Exercises ``load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)``
      and ``load_omni_general_plugins()`` for ``vllm_omni.platform_plugins``.
    - Setup check: Confirms ``torchtpu`` entry point targets
      ``vllm_torchtpu.omni:register_omni_tpu_platform``.
    - Outcome check: Verifies ``VLLM_PLUGINS`` filtering (``None``, ``["torchtpu"]``,
      ``["other"]``), class resolution to ``OmniTpuPlatform``, and idempotent
      ``omni_plugins_loaded`` state.
    """
    discovered = [
        ep
        for ep in entry_points(group=OMNI_PLATFORM_PLUGINS_GROUP)
        if ep.name == "torchtpu"
    ]
    assert len(discovered) == 1, (
        f"expected 1 'torchtpu' entry point, found {len(discovered)}"
    )
    assert discovered[0].value == _EXPECTED_ENTRYPOINT, (
        f"entry point value {discovered[0].value!r} != {_EXPECTED_ENTRYPOINT!r}"
    )

    monkeypatch.setattr(vllm_envs, "VLLM_PLUGINS", None)
    plugins_all = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)
    assert "torchtpu" in plugins_all, (
        f"'torchtpu' missing when VLLM_PLUGINS=None; got {set(plugins_all.keys())}"
    )

    monkeypatch.setattr(vllm_envs, "VLLM_PLUGINS", ["torchtpu"])
    plugins_only = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)
    assert set(plugins_only.keys()) == {"torchtpu"}, (
        f"expected {{'torchtpu'}}, got {set(plugins_only.keys())}"
    )

    monkeypatch.setattr(vllm_envs, "VLLM_PLUGINS", ["other"])
    plugins_none = load_omni_plugins_by_group(OMNI_PLATFORM_PLUGINS_GROUP)
    assert plugins_none == {}, f"expected empty dict, got {plugins_none!r}"

    qualname = plugins_only["torchtpu"]()
    assert qualname == _EXPECTED_QUALNAME, (
        f"resolved qualname {qualname!r} != {_EXPECTED_QUALNAME!r}"
    )
    resolved_cls = resolve_obj_by_qualname(qualname)
    assert resolved_cls is OmniTpuPlatform, (
        f"expected OmniTpuPlatform, got {resolved_cls!r}"
    )

    load_omni_general_plugins()
    assert omni_plugins.omni_plugins_loaded is True, (
        "omni_plugins_loaded must be True after load_omni_general_plugins()"
    )
    load_omni_general_plugins()
    assert omni_plugins.omni_plugins_loaded is True, (
        "omni_plugins_loaded must remain True across repeated calls"
    )
