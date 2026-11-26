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
"""Unit tests for decode-only batched RPA bucket planning and PP agreement."""

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.runner import rpa_decode_only_graph
from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len


def _fake_impl(*, has_decode_only: bool) -> PallasAttentionBackendImpl:
    impl = object.__new__(PallasAttentionBackendImpl)
    impl.rpa_kernel_decode_only = object() if has_decode_only else None
    return impl


def _resolve(monkeypatch, *, has_decode_only: bool = True, **overrides):
    monkeypatch.setattr(
        rpa_decode_only_graph,
        "get_layers_from_vllm_config",
        lambda _cfg, _cls: {
            "attn.0": SimpleNamespace(impl=_fake_impl(has_decode_only=has_decode_only))
        },
    )
    kwargs = dict(
        vllm_config=None,
        token_paddings=[16, 32, 64, 128, 256],
        pad_token_len=_get_padded_token_len,
        num_reqs_max_model_len=64,
        num_reqs_most_model_len=None,
        reorder_batch_threshold=1,
        enforce_eager=False,
        has_pcp=False,
        has_kv_cache_bundle=False,
        pipeline_parallel_size=1,
    )
    kwargs.update(overrides)
    return rpa_decode_only_graph.resolve_decode_only_buckets(**kwargs)


def test_disabled_when_flag_off(monkeypatch):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "0")
    assert _resolve(monkeypatch) == []


@pytest.mark.parametrize(
    ("max_reqs", "threshold", "expected"),
    [
        (64, 1, [16, 32, 64]),
        # Non-aligned max_reqs rounds up to the covering bucket (128).
        (100, 1, [16, 32, 64, 128]),
        # Spec decode widens the decode cap by reorder_batch_threshold.
        (64, 2, [16, 32, 64, 128]),
        # Cap above largest bucket covers every bucket without IndexError.
        (512, 1, [16, 32, 64, 128, 256]),
    ],
)
def test_covering_buckets(monkeypatch, max_reqs, threshold, expected):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    assert (
        _resolve(
            monkeypatch,
            num_reqs_max_model_len=max_reqs,
            reorder_batch_threshold=threshold,
        )
        == expected
    )


def test_mega_aot_raises(monkeypatch):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    monkeypatch.setattr(
        rpa_decode_only_graph.vllm_envs, "VLLM_USE_MEGA_AOT_ARTIFACT", True
    )
    with pytest.raises(ValueError, match="VLLM_USE_MEGA_AOT_ARTIFACT"):
        _resolve(monkeypatch)


@pytest.mark.parametrize(
    ("overrides", "has_decode_only"),
    [
        ({"enforce_eager": True}, True),
        ({"has_pcp": True}, True),
        ({"has_kv_cache_bundle": True}, True),
        ({}, False),
    ],
)
def test_self_disable_logs_warning(monkeypatch, overrides, has_decode_only):
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    warnings: list[str] = []
    monkeypatch.setattr(
        rpa_decode_only_graph.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )
    assert _resolve(monkeypatch, has_decode_only=has_decode_only, **overrides) == []
    assert len(warnings) == 1


@pytest.mark.parametrize(
    ("local_max_reqs", "peer_top_bucket", "expected"),
    [
        # Both stages agree on 64.
        (64, 64, [16, 32, 64]),
        # Local derives 128, peer derives 64 -> MIN reduction agrees on 64.
        (100, 64, [16, 32, 64]),
        # Peer disables (0) -> all stages disable.
        (64, 0, []),
    ],
)
def test_pp_stage_agreement_reduces_top_bucket(
    monkeypatch, local_max_reqs, peer_top_bucket, expected
):
    """Under PP, stages reduce top_bucket via min_over_group to stay in lockstep."""
    monkeypatch.setenv("TPU_RPA_DECODE_ONLY_GRAPH", "1")
    pp_group = SimpleNamespace(cpu_group="pp")
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: pp_group)

    def _all_reduce(t, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MIN
        assert group == "pp"
        t.clamp_(max=peer_top_bucket)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    assert (
        _resolve(
            monkeypatch,
            num_reqs_max_model_len=local_max_reqs,
            pipeline_parallel_size=2,
        )
        == expected
    )
