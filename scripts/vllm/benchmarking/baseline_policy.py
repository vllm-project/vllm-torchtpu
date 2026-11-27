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
"""Freeze policy and strict JSON loading for benchmark baselines.

Baseline files keep metadata under top-level keys that start with "_"; the
gates ignore them. The update policy lives under "_policy":
  "_policy": {
    "state": "development" | "frozen",   (absent block means development)
    "owner": "<handle>",                 (required when frozen)
    "approval": {"approver": "<handle>", "reference": "<PR or issue URL>"}
  }
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, NamedTuple

POLICY_KEY = "_policy"
DEVELOPMENT = "development"
FROZEN = "frozen"

_POLICY_FIELDS = frozenset({"state", "owner", "approval"})
_APPROVAL_FIELDS = frozenset({"approver", "reference"})


class BaselineApproval(NamedTuple):
    approver: str
    reference: str


@dataclasses.dataclass(frozen=True)
class BaselinePolicy:
    """Who may change a baseline file, and whether it is frozen."""

    state: str = DEVELOPMENT
    owner: str | None = None
    approval: BaselineApproval | None = None


def reject_json_constant(token: str) -> Any:
    """Reject non-standard JSON numeric constants (NaN, Infinity, -Infinity)."""
    raise ValueError(f"non-standard JSON constant {token!r} is not allowed")


def load_json_object(path: Path, *, label: str = "") -> dict[str, Any]:
    """Read a JSON object file; raise ValueError on read/parse/type errors."""
    target = f"{label} {path}" if label else str(path)
    try:
        data = json.loads(path.read_text(), parse_constant=reject_json_constant)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read {target}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{target} must hold a JSON object")
    return data


def gated_entries(baseline: dict[str, Any]) -> dict[str, Any]:
    """Return top-level entries that are gated (excluding '_'-prefixed metadata)."""
    return {k: v for k, v in baseline.items() if not k.startswith("_")}


def _policy_str(block: dict[str, Any], field: str, where: str) -> str:
    value = block.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}.{field} must be a non-empty string")
    return value.strip()


def _check_known_fields(
    block: dict[str, Any], allowed: frozenset[str], where: str
) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        names = ", ".join(repr(k) for k in unknown)
        raise ValueError(f"unknown {where} field(s): {names}")


def parse_policy(baseline: dict[str, Any]) -> BaselinePolicy:
    """Read the `_policy` block; raise ValueError when it is malformed."""
    if POLICY_KEY not in baseline:
        return BaselinePolicy()
    raw = baseline[POLICY_KEY]
    if not isinstance(raw, dict):
        raise ValueError(f"{POLICY_KEY} must be an object")
    _check_known_fields(raw, _POLICY_FIELDS, POLICY_KEY)
    state = raw.get("state", DEVELOPMENT)
    if state not in (DEVELOPMENT, FROZEN):
        raise ValueError(
            f"{POLICY_KEY}.state must be {DEVELOPMENT!r} or {FROZEN!r}, got {state!r}"
        )
    owner = None
    if state == FROZEN or "owner" in raw:
        owner = _policy_str(raw, "owner", POLICY_KEY)
    approval = None
    if "approval" in raw:
        block = raw["approval"]
        where = f"{POLICY_KEY}.approval"
        if not isinstance(block, dict):
            raise ValueError(f"{where} must be an object")
        _check_known_fields(block, _APPROVAL_FIELDS, where)
        approval = BaselineApproval(
            _policy_str(block, "approver", where),
            _policy_str(block, "reference", where),
        )
    return BaselinePolicy(state, owner, approval)


def _json_equal(left: Any, right: Any) -> bool:
    """Type-aware JSON equality where booleans never equal integers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def check_baseline_update(
    old: dict[str, Any] | None, new: dict[str, Any]
) -> str | None:
    """Return why replacing baseline `old` with `new` is not allowed, or None.

    A development baseline (or a new file, `old` None) changes freely. A frozen
    baseline changes its gated values, state, owner, or approval record only
    when `new` carries a new approval record whose approver is the owner named
    in the old file.
    """
    new_policy = parse_policy(new)
    if old is None:
        return None
    old_policy = parse_policy(old)
    if old_policy.state != FROZEN:
        return None
    if _json_equal(gated_entries(old), gated_entries(new)) and new_policy == old_policy:
        return None
    approval = new_policy.approval
    if approval is None or approval == old_policy.approval:
        return (
            "frozen baseline changed without a new "
            f"{POLICY_KEY}.approval record from owner {old_policy.owner!r}"
        )
    if approval.approver != old_policy.owner:
        return (
            f"frozen baseline changed with approval from {approval.approver!r}, "
            f"but its owner is {old_policy.owner!r}"
        )
    return None


def format_update_result(baseline_path: Path, violation: str | None) -> str:
    """Format the one-line PASS or FAILED summary for update mode."""
    if violation:
        return f"FAILED: {baseline_path}: {violation}"
    return "PASS: baseline update follows its freeze policy."
