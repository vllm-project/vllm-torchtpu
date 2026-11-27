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
"""Checks on the Qwen3.5 validation entries (models/qwen3_5/validation/)."""

from __future__ import annotations

import json
import pathlib
import re

import pytest
import yaml

from vllm_torchtpu import additional_config_sections, envs
from vllm_torchtpu.distributed.kv_transfer import connector_options
from vllm_torchtpu.models import qwen3_5
from vllm_torchtpu.models.qwen3_5 import settings

VALIDATION = pathlib.Path(qwen3_5.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))
GATES = sorted(VALIDATION.glob("gate-*.yaml"))
# The merge validation cap: no single deployment above 32 chips.
MAX_CHIPS = 32
# Identifiers outside readers cannot resolve: job ids, full commit ids and
# launcher files of a private deployment.
INTERNAL_REFERENCE = re.compile(r"\bj-[0-9a-f]{8}|\b[0-9a-f]{40}\b|\.jsonnet\b")


def _load(path: pathlib.Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def test_all_entries_are_present():
    assert [p.name.split("-")[0] for p in POINTS] == ["p01", "p05", "p08", "p10", "p12"]
    assert {p.stem for p in GATES} == {"gate-tp8-dcp4-1p1d", "gate-dp4tp2-1p1d"}


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_point_entry_is_well_formed(path):
    entry = _load(path)
    assert entry["name"] == path.stem
    assert entry["point"] == int(path.stem[1:3])
    record = entry["record"]
    assert record["mtok_per_dollar"] > 0
    assert record["p90_interactivity"] > 0
    chips = 0
    for leg_name in ("prefill", "decode"):
        leg = entry["legs"][leg_name]
        assert all(isinstance(a, str) for a in leg["serve_args"])
        assert all(isinstance(v, str) for v in leg["env"].values())
        assert isinstance(leg["additional_config"], dict)
        chips += leg["replicas"] * leg["chips_per_replica"]
    assert entry["chips"] == chips <= MAX_CHIPS


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_point_env_uses_declared_variables_only(path):
    """An entry may only set variables envs.py declares: knobs that are not
    declared there are settings or additional_config options."""
    entry = _load(path)
    for leg_name, leg in entry["legs"].items():
        undeclared = sorted(set(leg["env"]) - set(envs.environment_variables))
        assert not undeclared, f"{leg_name}: {undeclared}"


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_point_env_values_are_accepted_by_this_tree(path, monkeypatch):
    """Each value a leg sets is accepted by this tree's envs.py reader, except
    the ones listed under not_yet_supported, which name the capability they
    need from a later change."""
    entry = _load(path)
    pending = entry.get("not_yet_supported", {})
    assert set(pending) <= set(entry["legs"])
    for leg_name, leg in entry["legs"].items():
        skipped = pending.get(leg_name, {})
        assert set(skipped) <= set(leg["env"]), f"{leg_name}: {sorted(skipped)}"
        assert all(isinstance(need, str) and need for need in skipped.values())
        with monkeypatch.context() as m:
            for name, value in leg["env"].items():
                if name in skipped:
                    m.delenv(name, raising=False)
                else:
                    m.setenv(name, value)
            for name, value in leg["env"].items():
                if name in skipped:
                    continue
                try:
                    envs.environment_variables[name]()
                except ValueError as e:
                    pytest.fail(f"{leg_name}: {name}={value!r} is rejected: {e}")


@pytest.mark.parametrize("path", POINTS + GATES, ids=lambda p: p.stem)
def test_entries_carry_no_internal_references(path):
    text = path.read_text()
    assert not INTERNAL_REFERENCE.findall(text)


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_entries_name_no_absolute_paths(path):
    """Paths of one deployment's filesystem do not resolve elsewhere; an entry
    names a file through a placeholder (README.md)."""
    for leg_name, leg in _load(path)["legs"].items():
        values = list(leg["env"].values()) + leg["serve_args"]
        absolute = [v for v in values if v.startswith("/")]
        assert not absolute, f"{leg_name}: {absolute}"


def test_readme_carries_no_internal_references():
    assert not INTERNAL_REFERENCE.findall((VALIDATION / "README.md").read_text())


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_point_does_not_pass_additional_config_inline(path):
    entry = _load(path)
    for leg in entry["legs"].values():
        assert "--additional-config" not in leg["serve_args"]


def _serve_arg(leg: dict, flag: str) -> str | None:
    args = leg["serve_args"]
    return args[args.index(flag) + 1] if flag in args else None


def _connector_configs(config: dict):
    """A --kv-transfer-config, then each child of a multi-connector."""
    yield config
    for child in config.get("kv_connector_extra_config", {}).get("connectors", ()):
        yield from _connector_configs(child)


# The entries' kv_connector_extra_config keys that are no connector option;
# their own readers take them.
_CONNECTOR_SETTINGS = {
    # vLLM's MultiConnector.
    "TPUMultiConnector": {"connectors"},
    # vLLM's offloading config and TPURaidenStoreOffloadingSpec.
    "TPURaidenOffloadingConnector": {"cpu_bytes_to_use", "raiden_controller_port"},
    # _raiden_dst_dcp_size: the decode DCP size of the TP8 + DCP4 points.
    "TPURaidenConnector": {"destination_decode_context_parallel_size"},
}


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_point_connector_options_are_known(path):
    """Every kv_connector_extra_config key of a leg is an option of its
    connector or one of the settings above, and the options parse."""
    for leg_name, leg in _load(path)["legs"].items():
        raw = _serve_arg(leg, "--kv-transfer-config")
        assert raw is not None, leg_name
        for config in _connector_configs(json.loads(raw)):
            name = config["kv_connector"]
            options_cls = connector_options.CONNECTOR_OPTIONS[name]
            extra = config.get("kv_connector_extra_config", {})
            unknown = (
                set(extra)
                - connector_options.option_names(options_cls)
                - _CONNECTOR_SETTINGS[name]
            )
            assert not unknown, f"{leg_name} {name}: {sorted(unknown)}"
            connector_options.read_options(options_cls, extra)


def _qwen3_5_sections(entry: dict):
    for leg in entry.get("legs", {}).values():
        if "additional_config" in leg:
            yield leg["additional_config"].get(settings.SECTION)
    for arm in entry.get("arms", {}).values():
        yield arm["additional_config"].get(settings.SECTION)


@pytest.mark.parametrize("path", POINTS + GATES, ids=lambda p: p.stem)
def test_qwen3_5_settings_in_entries_exist(path):
    """A qwen3_5 section may only name settings this tree implements."""
    for section in _qwen3_5_sections(_load(path)):
        settings.parse(section)


@pytest.mark.parametrize("path", POINTS + GATES, ids=lambda p: p.stem)
def test_entry_sections_are_read_by_this_tree(path):
    """Every qwen3_5 / tpu_* section an entry passes has a reader in this tree;
    the config hook would refuse it at engine start otherwise."""
    entry = _load(path)
    configs = [leg.get("additional_config", {}) for leg in entry["legs"].values()]
    configs += [arm["additional_config"] for arm in entry.get("arms", {}).values()]
    for config in configs:
        assert not additional_config_sections.unknown_sections(config)


@pytest.mark.parametrize("path", GATES, ids=lambda p: p.stem)
def test_gate_entry_is_well_formed(path):
    entry = _load(path)
    assert entry["name"] == path.stem
    base = _load(next(p for p in POINTS if _load(p)["point"] == entry["base_point"]))
    assert base["legs"].keys() == entry["legs"].keys()
    assert entry["chips"] == sum(
        leg["replicas"] * base["legs"][name]["chips_per_replica"]
        for name, leg in entry["legs"].items()
    )
    assert set(entry["arms"]) == {"tradeoffs_off", "tradeoffs_on"}
    assert entry["suite"]["gsm8k"]["samples"] == 1319
    assert entry["suite"]["mtp_acceptance_length"] == "real"
    override = entry["legs"]["decode"]["serve_args_override"]["--speculative-config"]
    assert "synthetic" not in override
