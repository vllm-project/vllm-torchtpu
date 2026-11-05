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
"""TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS: which buckets apply W_UV before
the DCP merge.

0 keeps every bucket (backbone and MTP drafter). m > 0 takes the backbone's
buckets of at least m tokens on the power-of-2 ladder (`T > m // 2`) and never
the drafter, whose one graph serves every bucket. `uv_before_merge_runs` adds the
layer and query-replication terms; TPU_FUSED_UV_OPROJ takes a bucket only where
it is False.
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import attention


@pytest.mark.parametrize("backbone", [True, False])
@pytest.mark.parametrize("tokens", [4, 16, 32, 512, 1024, 4096])
def test_min_tokens_zero_takes_every_bucket(monkeypatch, tokens, backbone):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 0)
    assert attention._uv_before_merge_applies(tokens, backbone)


def test_min_tokens_takes_backbone_buckets_from_the_edge(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 1024)
    taken = [
        t
        for t in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        if attention._uv_before_merge_applies(t, True)
    ]
    assert taken == [1024, 2048, 4096]


@pytest.mark.parametrize("tokens", [4, 1024, 4096])
def test_min_tokens_never_takes_the_drafter(monkeypatch, tokens):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 1024)
    assert not attention._uv_before_merge_applies(tokens, False)


def test_retain_buffers_reads_each_buffer_in_warmup_only(monkeypatch):
    """Off TPU (the TPU path lifts through `_keep_input`, which reads
    nothing), each buffer is read once, by one-element index, never
    reshaped (a flattened read relayouts the whole weight)."""
    import torch

    from vllm_torchtpu.compilation import shape_variants

    class Buffer:
        """A buffer stand-in recording how it is read (no reshape method)."""

        def __init__(self, t):
            self.t, self.reads = t, []

        def dim(self):
            return self.t.dim()

        def __getitem__(self, idx):
            self.reads.append(idx)
            return self.t[idx]

    out = torch.ones(4, 8, dtype=torch.bfloat16)
    w = Buffer(torch.full((2, 3, 5), 2.0))
    scale = Buffer(torch.full((1,), 0.5))

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: False)
    assert attention._retain_buffers(out, w, scale, None) is out
    assert w.reads == [] and scale.reads == []

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
    got = attention._retain_buffers(out, w, scale, None)
    assert torch.equal(got, out)
    assert w.reads == [(0, 0, 0)]
    assert scale.reads == [(0,)]


# Every row count the fused W_UV + o_proj kernel takes (fp8_oproj_all_reduce.LIMIT_M).
_FUSED_ROWS = [1, 4, 8, 16, 32, 64]
_W_UV_LAYER = SimpleNamespace(W_UV_dcp_qrep=object())


def _uv_runs(
    monkeypatch,
    tokens,
    *,
    min_tokens=0,
    sized=False,
    tag="backbone",
    layer=_W_UV_LAYER,
    use_qrep=True,
):
    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", min_tokens)
    monkeypatch.setattr(envs, "TPU_MTP_DCP_MERGE_SIZED", sized)
    monkeypatch.setattr(attention.vllm_compilation_backends, "model_tag", tag)
    return attention.uv_before_merge_runs(layer, tokens, use_qrep)


@pytest.mark.parametrize("tokens", _FUSED_ROWS)
def test_uv_before_merge_runs_on_fused_rows_with_min_tokens_zero(monkeypatch, tokens):
    """The fused kernel must not ask for the latent here: the backend returns
    the projected rows and the kernel raised on their width at warmup."""
    assert _uv_runs(monkeypatch, tokens)


@pytest.mark.parametrize("tokens", _FUSED_ROWS)
def test_uv_before_merge_skips_fused_rows_below_min_tokens(monkeypatch, tokens):
    """With MIN_TOKENS=1024 the decode buckets keep the latent merge, so the
    fused kernel must keep taking them."""
    assert not _uv_runs(monkeypatch, tokens, min_tokens=1024)


@pytest.mark.parametrize(
    "layer", [SimpleNamespace(), SimpleNamespace(W_UV_dcp_qrep=None)]
)
def test_uv_before_merge_needs_the_replicated_w_uv(monkeypatch, layer):
    """A layer built without TPU_DCP_MLA_UV_BEFORE_MERGE projects after the
    merge."""
    assert not _uv_runs(monkeypatch, 16, layer=layer)


def test_uv_before_merge_needs_query_replication(monkeypatch):
    """Without qrep the backend attends this rank's heads and projects after the
    merge, even when the layer holds the replicated W_UV."""
    assert not _uv_runs(monkeypatch, 16, use_qrep=False)


@pytest.mark.parametrize(
    "tag,sized,runs",
    [
        ("eagle_head", True, False),
        ("eagle_head", False, True),
        ("backbone", True, True),
    ],
)
def test_sized_mtp_merge_keeps_the_drafter_latent(monkeypatch, tag, sized, runs):
    """TPU_MTP_DCP_MERGE_SIZED takes only a 2-byte partial, so the drafter must
    not project its f32 partial first; the backbone is unaffected."""
    assert _uv_runs(monkeypatch, 16, sized=sized, tag=tag) == runs


@pytest.mark.parametrize(
    "dcp_size,replicated,topk,runs",
    [
        (4, True, True, True),
        (1, True, True, False),
        (4, False, True, False),
        (4, True, False, False),
    ],
)
def test_the_wrapper_asks_with_the_backend_s_dcp_size(
    monkeypatch, dcp_size, replicated, topk, runs
):
    """TPU_FUSED_UV_OPROJ's gate in the attention wrapper asks the backend's
    question with the backend's `dcp_size`; the wrapper has none of its own."""
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper as Wrapper,
    )

    monkeypatch.setattr(envs, "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 0)
    monkeypatch.setattr(envs, "TPU_MTP_DCP_MERGE_SIZED", False)
    monkeypatch.setattr(attention.vllm_compilation_backends, "model_tag", "backbone")
    mla_attn = SimpleNamespace(
        W_UV_dcp_qrep=object(), impl=SimpleNamespace(dcp_size=dcp_size)
    )
    wrapper = SimpleNamespace(mla_attn=mla_attn)
    got = Wrapper._uv_before_merge(
        wrapper, 16, object() if replicated else None, object() if topk else None
    )
    assert got == runs
