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
"""The GDN state settings of the Qwen3.5 validation entries on main's unified
GDN state pool: the bf16 SSM opt-in, the block sizes it implies, and no QK
pair-blocked conv layout."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from vllm_torchtpu.models import qwen3_5

VALIDATION = pathlib.Path(qwen3_5.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))
GATES = sorted(VALIDATION.glob("gate-*.yaml"))

# One block holds a request's whole GDN state: ssm tokens + conv tokens at
# 1024 bytes per fp8 KV token row, conv state in float32 rounded up to a power
# of two of rows, SSM state in bf16 (tests/models/qwen3_5/
# test_qwen3_5_gdn_pool_sizing.py derives these through main's sizing).
_GDN_BLOCK_FIT = {
    # (tensor parallel size, MTP tokens): tokens
    (1, 1): 2048 + 256,
    (2, 3): 1024 + 256,
    (8, 3): 256 + 64,
}


def _load(path: pathlib.Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _arg(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1, flag
    return args[args.index(flag) + 1]


def _num_spec(args: list[str]) -> int:
    return yaml.safe_load(_arg(args, "--speculative-config"))["num_speculative_tokens"]


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_no_point_sets_the_qk_pair_conv_layout(path):
    """main's Raiden pool manifest refuses TPU_GDN_CONV_QK_PAIR_LAYOUT."""
    for leg in _load(path)["legs"].values():
        assert "TPU_GDN_CONV_QK_PAIR_LAYOUT" not in leg["env"]


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_every_point_leg_opts_into_the_bf16_ssm_state(path):
    for leg in _load(path)["legs"].values():
        assert _arg(leg["serve_args"], "--mamba-ssm-cache-dtype") == "bfloat16"


@pytest.mark.parametrize("path", GATES, ids=lambda p: p.stem)
def test_gates_run_the_bf16_ssm_state_as_a_trade_off(path):
    entry = _load(path)
    overrides = entry["arm_serve_args_override"]
    assert set(overrides) == set(entry["arms"])
    assert overrides["tradeoffs_on"] == {"--mamba-ssm-cache-dtype": "bfloat16"}
    assert overrides["tradeoffs_off"] == {"--mamba-ssm-cache-dtype": "auto"}


# The record jobs' --block-size per leg (cloud-devkit-recipes 90863fb2).
_RECORD_BLOCK = {"prefill": 2176, "decode": 640}


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_block_size_is_what_main_runs(path):
    """main raises a --block-size below the GDN state fit (aligned to the RPA
    kernel's 128-token blocks) and keeps a larger one, so each leg passes
    max(aligned fit, record value): the value the server runs."""
    for name, leg in _load(path)["legs"].items():
        args = leg["serve_args"]
        tp = int(_arg(args, "--tensor-parallel-size"))
        fit = _GDN_BLOCK_FIT[(tp, _num_spec(args))]
        aligned_fit = -(-fit // 128) * 128
        expected = max(aligned_fit, _RECORD_BLOCK[name])
        assert int(_arg(args, "--block-size")) == expected, name
