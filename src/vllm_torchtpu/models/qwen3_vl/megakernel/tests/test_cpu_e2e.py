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
"""CPU end-to-end test of this megakernel folder (fake runner, no TPU).

The adapter loads `SPEC` by this folder's package name, runs its startup
checks on a fake multimodal M-RoPE runner, and `create` builds a program for
a tiny geometry; the program packs the fake vLLM backbone's bf16 weights,
binds its per-layer bf16 KV caches at their page size (128 tokens, or 512 as
vLLM picks at 256 sequences of 262144 tokens), warms up and serves a decode
step through the CPU-interpret kernel, and leaves a step with prompt tokens to
the stock model.
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest  # pyrefly: ignore[missing-import]
import torch
from jax.sharding import Mesh

from vllm_torchtpu.megakernel.adapter.testkit import check_plugin_spec, run_cpu_e2e
from vllm_torchtpu.megakernel.core import CacheExec, ExecSpec

from .. import adapter, model, profiles
from .tiny import tiny_geom, tiny_tiles

FOLDER = (__package__ or "").rpartition(".")[0]
NUM_PAGES = 16


def _w(*shape: int) -> SimpleNamespace:
    return SimpleNamespace(weight=torch.randn(shape, dtype=torch.bfloat16) * 0.05)


def _backbone(g: Any, page_size: int) -> SimpleNamespace:
    """A vLLM-like Qwen3-VL text backbone: fused qkv / gate_up projections in
    `(out, in)` layout and one bf16 KV cache of HND pages per layer."""
    lay = model.Qwen3VLText.pool_layout(g, R=1)
    page = dataclasses.replace(lay, page_size=page_size).kv_page_shape
    half = g.ROT // 2
    inv = g.theta ** (-torch.arange(half, dtype=torch.float32) / half)
    ang = torch.arange(4096, dtype=torch.float32)[:, None] * inv
    cos_sin = torch.cat([ang.cos(), ang.sin()], -1).to(torch.bfloat16)
    rope = SimpleNamespace(cos_sin_cache=cos_sin, is_neox_style=True)
    layers = []
    for i in range(g.L):
        kv = torch.zeros((NUM_PAGES, *page), dtype=torch.bfloat16)
        attn = SimpleNamespace(
            layer_name=f"model.layers.{i}.self_attn.attn", kv_cache=[kv]
        )
        layers.append(
            SimpleNamespace(
                input_layernorm=_w(g.H),
                post_attention_layernorm=_w(g.H),
                self_attn=SimpleNamespace(
                    qkv_proj=_w((g.NH + 2 * g.NKV) * g.HD, g.H),
                    o_proj=_w(g.H, g.NH * g.HD),
                    q_norm=_w(g.HD),
                    k_norm=_w(g.HD),
                    rotary_emb=rope,
                    attn=attn,
                ),
                mlp=SimpleNamespace(
                    gate_up_proj=_w(2 * g.FF, g.H), down_proj=_w(g.H, g.FF)
                ),
            )
        )
    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=_w(g.H))


@pytest.fixture(params=[128, 512], ids=["p128", "p512"])
def tiny_runner(
    request: pytest.FixtureRequest, mk_fake_runner: Any, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """A fake TP=1 multimodal runner over a tiny backbone of `request.param`-token
    KV pages, with this folder as the megakernel plugin (one `(1, 1)` bucket)."""
    page_size = request.param
    g = tiny_geom(R=1)
    ex = ExecSpec(
        R=1,
        B=1,
        T=1,
        ctx=256,
        tiles=tiny_tiles(kv_blk=256),
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=NUM_PAGES, max_pages_per_seq=2),
    )
    folder = sys.modules[FOLDER]
    # The tiny geometry stands in for the checkpoint's (TP=1 on CPU).
    monkeypatch.setattr(folder, "SPEC", dataclasses.replace(folder.SPEC, tp_sizes=(1,)))
    monkeypatch.setattr(
        model.Qwen3VLText, "derive_geometry", classmethod(lambda cls, *a, **k: g)
    )
    real_cls = adapter.Qwen3VLProgram
    monkeypatch.setattr(
        adapter,
        "Qwen3VLProgram",
        lambda m, **kw: real_cls(m, geom=g, base_exec=ex, **kw),
    )
    torch.manual_seed(0)
    qsl = np.zeros(16, dtype=np.int32)
    qsl[:2] = [0, 1]
    return mk_fake_runner(
        model=SimpleNamespace(model=_backbone(g, page_size)),
        hf_config=SimpleNamespace(
            architectures=list(model.Qwen3VLText.hf_architectures)
        ),
        megakernel={
            "plugin": FOLDER,
            "options": {"buckets": "1"},
            "stats_period_s": 3600,
        },
        mesh=Mesh(np.array(jax.devices("cpu")[:1]), ("tp",)),
        block_size=page_size,
        supports_mm_inputs=True,
        uses_mrope=True,
        token_paddings=(1, 2, 4),
        req_block_pairs=((2, 2),),
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=1, start_index=0),
        query_start_loc_np=qsl,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.array([129, 0]),
            num_prompt_tokens=np.array([100, 0]),
        ),
    )


def test_spec_loads_without_registering_anything() -> None:
    spec = check_plugin_spec(FOLDER)
    assert spec.create == ".adapter:create"
    assert profiles.DEFAULT_PROFILE.name in profiles.PROFILES
    assert model.register() is model.Qwen3VLText
    assert model.register() is model.Qwen3VLText  # idempotent


def test_serves_decode_and_leaves_prompt_steps_to_stock(tiny_runner: Any) -> None:
    ids = torch.tensor([5], dtype=torch.int32)
    pos = torch.full((3, 1), 129 - 40, dtype=torch.int32)  # M-RoPE delta -40
    batch = tiny_runner.input_batch

    def steps(prog: Any) -> Any:
        yield ids, pos
        batch.num_computed_tokens_cpu[0] = 99  # a 1-token final prefill chunk
        yield ids, pos

    def metadata(prog: Any) -> dict[str, Any]:
        bt = torch.tensor([3, 4, 0, 0], dtype=torch.int32)
        seq = torch.tensor([130, 0], dtype=torch.int32)
        return {prog.attn_layer_name: SimpleNamespace(block_tables=bt, seq_lens=seq)}

    res = run_cpu_e2e(None, tiny_runner, steps, metadata=metadata)
    assert res.program.profile.name == profiles.PROFILE.name
    assert sorted(res.driver._steps) == [(1, 1)]
    assert res.hits == [True, False]
    assert res.step_mix == {"mk_t1_b1": 1, "stock_prompt": 1}
    out = res.outputs[0]
    assert out.shape == (1, 256) and bool(torch.isfinite(out.float()).all())
    # The kernel took the pool's pages, and the new token's K/V went to KV
    # index 129 of every layer: page 4 slot 1 at 128 tokens, page 3 slot 129 at 512.
    layers = tiny_runner.model.model.layers
    ps = layers[0].self_attn.attn.kv_cache[0].shape[-1]
    assert res.program.spec.cache_backend.layout.page_size == ps
    page_id, slot = (3, 4)[129 // ps], 129 % ps
    for layer in layers:
        page = layer.self_attn.attn.kv_cache[0][page_id].float()
        assert page[..., slot].abs().sum() > 0
        assert page[..., :slot].abs().sum() == 0
        assert page[..., slot + 1 :].abs().sum() == 0
