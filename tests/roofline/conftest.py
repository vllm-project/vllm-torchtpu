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
"""Shared locked-checkpoint fixture loaders for roofline unit tests."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

from vllm_torchtpu.roofline import (
    Dtypes,
    ModelGeometry,
    ModelIdentity,
    Parallel,
    StepShape,
    apply_hf_overrides,
    dtypes_from_hf_config,
    from_hf_config,
    pass_costs,
    resolve_model_identity,
)
from vllm_torchtpu.roofline.modules import ModuleCost

FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"
LOCKED_FIXTURE_FILES: tuple[str, ...] = (
    "qwen3_5_397b_fp8.json",
    "qwen3_5_397b_nvfp4.json",
    "glm_5_2_fp8.json",
    "glm_5_2_nvfp4.json",
    "deepseek_v4_1_flash.json",
    "minimax_m3_1_flash.json",
    "qwen3_8_flash_next_fp8.json",
)


@dataclass(frozen=True, slots=True)
class LockedBundle:
    """Bundled config, identity, geometry, and dtypes for a locked fixture."""

    config: dict[str, object]
    identity: ModelIdentity
    geometry: ModelGeometry
    dtypes: Dtypes


@lru_cache(maxsize=16)
def _read_fixture_json(fixture_name: str) -> dict[str, object]:
    """Read and cache one locked HuggingFace config JSON from fixtures/."""
    return json.loads((FIXTURES_DIR / fixture_name).read_text(encoding="utf-8"))


def load_locked_config(
    fixture_name: str = "qwen3_5_397b_fp8.json",
) -> dict[str, object]:
    """Return a fresh copy of one of the locked basis HF checkpoint configs."""
    return copy.deepcopy(_read_fixture_json(fixture_name))


@lru_cache(maxsize=1)
def _locked_draft_fixture_map() -> dict[str, str]:
    """Derive checkpoint fixture -> draft_fixture mapping from manifest.json."""
    manifest = _read_fixture_json("manifest.json")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list):
        return {}
    return {
        str(entry["fixture"]): str(entry["draft_fixture"])
        for entry in checkpoints
        if isinstance(entry, dict)
        and entry.get("fixture")
        and entry.get("draft_fixture")
    }


def load_locked_draft_config(
    fixture_name: str = "minimax_m3_1_flash.json",
) -> dict[str, object] | None:
    """Return a fresh copy of a locked fixture's companion draft config if defined."""
    draft_name = _locked_draft_fixture_map().get(fixture_name)
    if draft_name is None:
        return None
    return copy.deepcopy(_read_fixture_json(draft_name))


def load_locked_geometry(
    fixture_name: str = "qwen3_5_397b_fp8.json",
) -> ModelGeometry:
    """Parse one of the locked basis HF checkpoint configs into ModelGeometry."""
    return from_hf_config(_read_fixture_json(fixture_name))


def load_locked_bundle(
    fixture_name: str = "qwen3_5_397b_fp8.json",
) -> LockedBundle:
    """Load config copy, canonical identity, geometry, and dtypes for a fixture."""
    config = load_locked_config(fixture_name)
    identity = resolve_model_identity(config)
    geometry = from_hf_config(config)
    dtypes = dtypes_from_hf_config(config)
    return LockedBundle(
        config=config, identity=identity, geometry=geometry, dtypes=dtypes
    )


def override_locked_config(
    fixture_name: str,
    overrides: dict[str, object],
) -> dict[str, object]:
    """Return a locked HF config copy patched via apply_hf_overrides."""
    return dict(apply_hf_overrides(load_locked_config(fixture_name), overrides))


def modules_by_name(
    geometry: ModelGeometry,
    parallel: Parallel,
    shape: StepShape,
    dtypes: Dtypes,
) -> dict[str, ModuleCost]:
    """Return pass_costs indexed by module name."""
    return {
        module.name: module for module in pass_costs(geometry, parallel, shape, dtypes)
    }


@pytest.fixture
def locked_config_path(tmp_path: Path) -> Callable[[str], Path]:
    """Return a helper writing a locked fixture config JSON into tmp_path."""

    def _write(fixture_name: str = "qwen3_5_397b_fp8.json") -> Path:
        target = tmp_path / fixture_name
        target.write_text(
            (FIXTURES_DIR / fixture_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return target

    return _write
