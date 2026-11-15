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
"""The Qwen3.5 SPEC's startup guards on a prefill / decode disaggregated (PD)
decode leg: TP8 with DCP4 that consumes KV through the TPU PD connector
(`TPURaidenConnector`, `kv_role` `kv_consumer`), MTP and prefix caching, with
`--mamba-ssm-cache-dtype bfloat16`. The leg passes; a producer, `kv_both` or
the default SSM cache dtype is still declined, and the configurations the
template served before still pass."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_torchtpu.megakernel.adapter.guards as guards_mod
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.adapter.plugin import RunnerView
from vllm_torchtpu.models.qwen3_5 import megakernel as qwen_plugin

TP = 8


@pytest.fixture(autouse=True)
def _tpu_connectors_registered() -> None:
    from vllm_torchtpu.platforms import tpu_platform

    tpu_platform._register_tpu_kv_connectors()


def _raiden(role: str = "kv_consumer") -> Any:
    return SimpleNamespace(
        kv_connector="TPURaidenConnector",
        kv_role=role,
        kv_connector_extra_config={"destination_decode_context_parallel_size": 4},
    )


def _runner(**kw: Any) -> Any:
    base: dict[str, Any] = dict(
        tp_size=TP,
        speculative_tokens=3,
        enable_prefix_caching=True,
        block_size=640,
        mamba_ssm_cache_dtype="bfloat16",
        decode_context_parallel_size=4,
        cp_kv_cache_interleave_size=128,
        kv_transfer_config=_raiden(),
    )
    base.update(kw)
    return testing.fake_runner(**base)


def _code(runner: Any) -> str | None:
    res = guards_mod.check_spec(
        runner, qwen_plugin.SPEC, view=RunnerView(runner, tp_rank=0)
    )
    return None if res is None else res.code


def test_pd_decode_leg_passes_the_spec_guards() -> None:
    assert _code(_runner()) is None


@pytest.mark.parametrize("role", ["kv_producer", "kv_both"])
def test_producer_and_kv_both_legs_are_still_declined(role: str) -> None:
    assert _code(_runner(kv_transfer_config=_raiden(role))) == (
        "kv_connector_unsupported:TPURaidenConnector"
    )


@pytest.mark.parametrize(
    "dtype, code",
    [
        ("bfloat16", None),
        (torch.bfloat16, None),
        ("float32", "requires_unmet:mamba_ssm_cache_dtype"),
        ("auto", "requires_unmet:mamba_ssm_cache_dtype"),
    ],
)
def test_mamba_ssm_cache_dtype_requirement(dtype: Any, code: str | None) -> None:
    """The SPEC still requires `--mamba-ssm-cache-dtype bfloat16` (unchanged);
    the PD decode legs pass it, and without it they are declined by name."""
    assert qwen_plugin.SPEC.requires == {"mamba_ssm_cache_dtype": "bfloat16"}
    assert _code(_runner(mamba_ssm_cache_dtype=dtype)) == code


@pytest.mark.parametrize(
    "kw",
    [
        # what the template served before: TP8, no DCP, no transfer connector
        dict(decode_context_parallel_size=1, kv_transfer_config=None),
        dict(
            decode_context_parallel_size=1,
            kv_transfer_config=SimpleNamespace(
                kv_connector="TPURaidenOffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={},
            ),
        ),
        # a DCP4 decode leg without a connector, and a TP8 consumer without DCP
        dict(kv_transfer_config=None),
        dict(decode_context_parallel_size=1),
    ],
)
def test_other_supported_configs_pass(kw: dict[str, Any]) -> None:
    assert _code(_runner(**kw)) is None
