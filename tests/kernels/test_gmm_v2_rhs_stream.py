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
"""The grouped matmul's hand-streamed whole-expert rhs slabs are off unless
a model rule sets gmm_v2.RHS_STREAM, and then cover only whole-expert tiles
of non-fused projections without bias or packed weights."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.megablox import gmm_v2

pytestmark = pytest.mark.cpu_test


def _cfgs(fuse_act=None, tile_n=1024, tile_k=4096, has_bias=False, unpack=False):
    """The fields use_rhs_stream reads: a GMM2 of 4096 x 1024 per expert."""
    return SimpleNamespace(
        fuse_act=fuse_act,
        out_size_n=1024,
        tiles=SimpleNamespace(tile_n=tile_n, tile_k=tile_k),
        dims=SimpleNamespace(size_k=4096),
        rhs_cfgs=SimpleNamespace(has_bias=has_bias, should_unpack=unpack),
    )


def test_the_default_is_emit_pipelines_rhs_path():
    assert gmm_v2.RHS_STREAM is False


def test_off_whole_expert_tiles_keep_emit_pipeline(monkeypatch):
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", False)
    assert gmm_v2.use_rhs_stream(_cfgs()) is False


@pytest.mark.parametrize(
    "cfgs,streamed",
    [
        (_cfgs(), True),
        (_cfgs(fuse_act="silu"), False),
        (_cfgs(tile_n=512), False),
        (_cfgs(tile_k=2048), False),
        (_cfgs(has_bias=True), False),
        (_cfgs(unpack=True), False),
    ],
    ids=["whole_expert", "fused", "split_n", "split_k", "bias", "packed"],
)
def test_on_only_whole_expert_tiles_stream(monkeypatch, cfgs, streamed):
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", True)
    assert bool(gmm_v2.use_rhs_stream(cfgs)) is streamed


def test_set_rhs_stream_switches_it(monkeypatch):
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", False)
    gmm_v2.set_rhs_stream(True)
    assert gmm_v2.RHS_STREAM is True
    gmm_v2.set_rhs_stream(False)
    assert gmm_v2.RHS_STREAM is False
