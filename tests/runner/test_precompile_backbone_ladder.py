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
"""The backbone warmup only fails on refused buckets the ladder can run."""

import contextlib
from types import SimpleNamespace

import pytest

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Buckets:
    def __init__(self, refused):
        self.refused = set(refused)
        self.traces = 0

    def needs_retrace(self, size):
        return False


def _run(monkeypatch, ladder, refused):
    buckets = _Buckets(refused)

    @contextlib.contextmanager
    def warmup():
        yield buckets

    monkeypatch.setattr(shape_variants, "warmup", warmup)
    monkeypatch.setattr(tpu_runner.envs, "TPU_PRECOMPILE_TRACE_FIRST", False)
    ran = []
    runner = SimpleNamespace(
        _backbone_token_paddings=lambda: list(ladder),
        _backbone_dummy_runs=ran.append,
        _pp_wave=None,
        model=None,
        vllm_config=None,
    )
    TPUModelRunner._precompile_backbone(runner)
    return ran


def test_refused_buckets_above_the_pcp_ladder_do_not_fail_the_start(monkeypatch):
    # PCP caps the local tokens at 1536; compile_sizes still lists 2048-4096.
    ran = _run(monkeypatch, [512, 1024, 1536], refused={2048, 3072, 4096})
    assert ran == [512, 1024, 1536]


def test_a_refused_bucket_the_ladder_can_run_still_fails(monkeypatch):
    with pytest.raises(
        shape_variants.ShapeSpecializationError, match=r"uncompiled: \[1024\]"
    ):
        _run(monkeypatch, [512, 1024, 1536], refused={1024, 4096})
