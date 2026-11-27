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
"""Model fixture provenance manifest and geometry lock-in tests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from vllm_torchtpu.roofline import (
    SUPPORTED_MODELS,
    count_layer_types,
    from_hf_config,
    resolve_model_identity,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_MANIFEST_PATH = _FIXTURES_DIR / "manifest.json"
_NON_MODEL_FIXTURE_FILES = frozenset({"manifest.json", "representative_layouts.json"})


def verify_model_manifest(manifest_path: Path, fixtures_dir: Path) -> None:
    """Verify supported models, 1:1 fixture parity, SHA-256 digests, and geometry."""
    assert manifest_path.is_file(), f"Missing model manifest at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["checkpoints"]
    assert entries

    manifest_repos = {entry["hf_repo"] for entry in entries}
    supported_repos = {
        repo for spec in SUPPORTED_MODELS.values() for repo in spec.hf_repos
    }
    assert manifest_repos == supported_repos

    manifest_canonical_ids = {entry["canonical_id"] for entry in entries}
    assert manifest_canonical_ids == set(SUPPORTED_MODELS)

    # Every model and optional draft JSON under fixtures/ must appear in manifest 1:1.
    manifest_fixtures = {entry["fixture"] for entry in entries}
    for entry in entries:
        draft_fixture = entry.get("draft_fixture")
        if draft_fixture is not None:
            manifest_fixtures.add(draft_fixture)
    disk_model_fixtures = {
        path.name
        for path in fixtures_dir.glob("*.json")
        if path.name not in _NON_MODEL_FIXTURE_FILES
    }
    assert disk_model_fixtures == manifest_fixtures

    for entry in entries:
        fixture_path = fixtures_dir / entry["fixture"]
        assert fixture_path.is_file(), f"Missing fixture {fixture_path}"
        actual_digest = base64.b64encode(
            hashlib.sha256(fixture_path.read_bytes()).digest()
        ).decode("ascii")
        assert actual_digest == entry["sha256_b64"]
        assert entry["revision"]

        draft_cfg: dict[str, object] | None = None
        draft_fixture = entry.get("draft_fixture")
        if draft_fixture is not None:
            draft_path = fixtures_dir / draft_fixture
            assert draft_path.is_file(), f"Missing draft fixture {draft_path}"
            actual_draft_digest = base64.b64encode(
                hashlib.sha256(draft_path.read_bytes()).digest()
            ).decode("ascii")
            assert actual_draft_digest == entry["draft_sha256_b64"]
            draft_cfg = json.loads(draft_path.read_text(encoding="utf-8"))

        raw_cfg = json.loads(fixture_path.read_text(encoding="utf-8"))
        identity = resolve_model_identity(raw_cfg, model_ref=entry["hf_repo"])
        assert identity.canonical_id == entry["canonical_id"]

        geometry = from_hf_config(
            raw_cfg, model_ref=entry["hf_repo"], draft_config=draft_cfg
        )
        expected_geom = entry["geometry"]
        for field_name, expected_value in expected_geom.items():
            actual_value = getattr(geometry, field_name)
            if isinstance(expected_value, list):
                expected_value = tuple(expected_value)
            assert actual_value == expected_value, (
                f"{entry['fixture']}.{field_name}: "
                f"{actual_value!r} != {expected_value!r}"
            )

        structure = entry["structure"]
        counts = count_layer_types(geometry)
        for family, expected_count in structure["layer_counts"].items():
            assert getattr(counts, family) == expected_count
        assert geometry.indexer_types.count("full") == structure["indexer_full"]
        assert geometry.indexer_types.count("shared") == structure["indexer_shared"]
        assert geometry.ffn_layer_counts() == (
            structure["ffn_moe_layers"],
            structure["ffn_dense_layers"],
        )


def test_model_manifest_matches_supported_models_and_fixture_hashes() -> None:
    """Verify locked manifest fixtures, digests, and extracted geometry."""
    verify_model_manifest(_MANIFEST_PATH, _FIXTURES_DIR)
