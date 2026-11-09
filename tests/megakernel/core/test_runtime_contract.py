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
"""The runtime contract: `TpuMonoSpec` validation, the decline and result
types, and the framework-free import of `megakernel.runtime`."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from vllm_torchtpu.megakernel.core import errors
from vllm_torchtpu.megakernel.core.runtime import contract

_SRC = Path(contract.__file__).resolve().parents[4]


def _spec(**kw: Any) -> contract.TpuMonoSpec:
    return contract.TpuMonoSpec("t", "1", create="m:c", **kw)


def test_spec_validates_and_normalizes() -> None:
    plug = contract.TpuMonoSpec(
        "toy", "1", create="x:y", features={"spec_decode", "mrope"}
    )
    assert plug.features == frozenset({"spec_decode", "mrope"})
    with pytest.raises(errors.MegakernelConfigError, match="unknown features"):
        contract.TpuMonoSpec("toy", "1", create="x:y", features={"warp"})
    with pytest.raises(errors.MegakernelConfigError):
        contract.TpuMonoSpec("", "1", create="x:y")


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"requires": {"no_such_key": "1"}}, "unknown requires key 'no_such_key'"),
        ({"numerics": "bitwise"}, "numerics='bitwise'"),
        ({"parallelism": {"tp", "sp"}}, "unknown parallelism"),
        ({"parallelism": {"ep_in_tp"}}, "must include 'tp'"),
        ({"features": {"telepathy"}}, "unknown features"),
        ({"collective_domain": "dp"}, "collective_domain='dp'"),
    ],
)
def test_invalid_spec_fields_raise(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(errors.MegakernelConfigError, match=match):
        _spec(**kw)


@pytest.mark.parametrize(
    "field",
    [
        "chips",
        "tp_sizes",
        "parallelism",
        "features",
        "decline_codes",
        "top_level_options",
    ],
)
def test_bare_string_for_a_tuple_field_raises(field: str) -> None:
    # "TPU7x" must not split into characters ("T" would match every TPU).
    with pytest.raises(errors.MegakernelConfigError, match=f"{field} must be a tuple"):
        _spec(**{field: "TPU7x"})
    with pytest.raises(errors.MegakernelConfigError, match=f"{field} must be a tuple"):
        _spec(**{field: 4})


def test_top_level_options_is_a_frozenset_of_names() -> None:
    spec = _spec(top_level_options=["mode", "buckets"])
    assert spec.top_level_options == frozenset({"mode", "buckets"})
    assert _spec().top_level_options == frozenset()


def test_tuple_fields_keep_whole_items() -> None:
    spec = _spec(chips=["TPU7x"], tp_sizes=[4, "8"], decline_codes=["a"])
    assert spec.chips == ("TPU7x",) and spec.tp_sizes == (4, 8)
    assert spec.decline_codes == ("a",)


def test_requires_is_frozen_and_lowercase() -> None:
    src = {"mamba_ssm_cache_dtype": "BFloat16", "enable_prefix_caching": False}
    spec = _spec(requires=src)
    assert spec.requires == {
        "enable_prefix_caching": "false",
        "mamba_ssm_cache_dtype": "bfloat16",
    }
    assert list(spec.requires) == sorted(src)
    with pytest.raises(TypeError):
        spec.requires["block_size"] = "16"  # type: ignore[index]
    src["block_size"] = 16  # the caller's dict is copied, not aliased
    assert "block_size" not in spec.requires
    assert dataclasses.replace(spec, requires={}).requires == {}
    with pytest.raises(
        errors.MegakernelConfigError, match="requires must be a mapping"
    ):
        _spec(requires=[("block_size", "16")])


def test_decline_types_and_one_eligibility_result() -> None:
    assert errors.EligibilityResult is contract.EligibilityResult
    assert issubclass(contract.LayoutDecline, errors.MegakernelLayoutError)
    dec = contract.Decline("too_small", "msg")
    assert dec.result == contract.EligibilityResult(False, "too_small", "msg")
    with pytest.raises(AttributeError):
        _ = errors.NoSuchName


_BLOCKED_IMPORT = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None
import vllm_torchtpu.megakernel.core.errors as errors
import vllm_torchtpu.megakernel.core.runtime as rt
import vllm_torchtpu.megakernel.core.runtime.contract as contract
for name in contract.__all__:
    if name in rt.__all__:
        assert getattr(rt, name) is getattr(contract, name), name
assert errors.EligibilityResult is contract.EligibilityResult
assert "vllm_torchtpu.megakernel.core.runtime.stats" not in sys.modules
{extra}
print("OK")
"""
_STATS_ON_FIRST_ACCESS = """
import vllm_torchtpu.megakernel.core.runtime.stats as stats
for name in stats.__all__:
    assert getattr(rt, name) is getattr(stats, name), name
assert "numpy" in sys.modules
"""


@pytest.mark.parametrize(
    "blocked, extra",
    [
        (("jax", "jaxlib", "numpy", "torch", "vllm"), ""),
        (("jax", "jaxlib", "torch", "vllm"), _STATS_ON_FIRST_ACCESS),
    ],
    ids=["contract_stdlib_only", "stats_numpy_on_first_access"],
)
def test_runtime_package_imports_without_frameworks(
    blocked: tuple[str, ...], extra: str
) -> None:
    """The contract imports with jax, numpy, torch and vLLM blocked; the
    statistics need numpy only, on first access."""
    out = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT.format(blocked=blocked, extra=extra)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr[-4000:]
