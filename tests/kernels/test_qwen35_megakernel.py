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
"""CPU unit tests for top-level Qwen3.5 megakernel modular package structure,
step metadata preparation, and A/B calibrator."""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from vllm_torchtpu.kernels import qwen35_megakernel as pkg
from vllm_torchtpu.kernels.qwen35_megakernel import attention_block
from vllm_torchtpu.kernels.qwen35_megakernel import blocks
from vllm_torchtpu.kernels.qwen35_megakernel import collectives
from vllm_torchtpu.kernels.qwen35_megakernel import config
from vllm_torchtpu.kernels.qwen35_megakernel import context
from vllm_torchtpu.kernels.qwen35_megakernel import gdn_block
from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import moe_block
from vllm_torchtpu.kernels.qwen35_megakernel import pool_dma
from vllm_torchtpu.kernels.qwen35_megakernel import rings
from vllm_torchtpu.kernels.qwen35_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.qwen35_megakernel.vllm_adapter import (
    MKABCalibrator,
    parse_mk_ab_enabled,
    prepare_step_metadata,
)

from qwen35_megakernel_test_utils import bare_mk


def test_modular_reexports_and_symbols():
    """Verify that the modular submodules expose their expected classes/functions
    and `megakernel.py` / `__init__.py` preserve backward-compatible re-exports."""
    assert K.MKConfig is config.MKConfig
    assert K.Tiles is config.Tiles
    assert K.default_tiles is config.default_tiles
    assert K.weight_shapes is config.weight_shapes
    assert K.state_shapes is config.state_shapes
    assert K.smalls_layout is config.smalls_layout
    assert K.encode_ecopy is config.encode_ecopy
    assert K.plan is config.plan

    assert K.MKBuildContext is context.MKBuildContext
    assert K.PallasAllReduce is collectives.PallasAllReduce
    assert K.WeightRings is rings.WeightRings
    assert K.DenseWeightRing is rings.DenseWeightRing
    assert K.ExpertWeightRing is rings.ExpertWeightRing
    assert K.KVPoolDMA is pool_dma.KVPoolDMA
    assert K.GDNPoolDMA is pool_dma.GDNPoolDMA
    assert K.GprRecIn is pool_dma.GprRecIn
    assert K.GDNBlock is gdn_block.GDNBlock
    assert K.PagedAttentionBlock is attention_block.PagedAttentionBlock
    assert K.MoEBlock is moe_block.MoEBlock
    assert blocks.GDNBlock is gdn_block.GDNBlock

    assert pkg.MKConfig is config.MKConfig
    assert pkg.Tiles is config.Tiles
    assert pkg.PallasAllReduce is collectives.PallasAllReduce
    assert pkg.WeightRings is rings.WeightRings
    assert pkg.KVPoolDMA is pool_dma.KVPoolDMA
    assert pkg.GDNPoolDMA is pool_dma.GDNPoolDMA
    assert pkg.GDNBlock is gdn_block.GDNBlock
    assert pkg.PagedAttentionBlock is attention_block.PagedAttentionBlock
    assert pkg.MoEBlock is moe_block.MoEBlock


def test_prepare_step_metadata_t1_and_t4():
    num_reqs = 3
    max_pages = 16
    bt_1d = torch.arange(1, 1 + 8 * max_pages, dtype=torch.int32)
    attn_md = SimpleNamespace(
        seq_lens=torch.tensor([100, 250, 40, 0, 0, 0, 0, 0], dtype=torch.int32),
        block_tables=bt_1d,
    )
    roff_full = torch.zeros((512,), dtype=torch.int32)
    roff_full[10] = 2
    roff_full[20] = 3
    roff_full[30] = 1
    mamba_mds = []
    for g in range(3):
        state_ids = torch.tensor(
            [10 + g, 20 + g, 30 + g, 0, 0, 0, 0, 0], dtype=torch.int32
        )
        ckpt_2d = torch.zeros((8, 4), dtype=torch.int32)
        for s in range(num_reqs):
            for t in range(4):
                ckpt_2d[s, t] = 100 * (g + 1) + 10 * s + t
        mamba_mds.append(
            SimpleNamespace(
                mamba_state_indices=state_ids,
                mamba_ckpt_indices=ckpt_2d,
                mamba_slot_read_offsets=roff_full,
            )
        )

    positions_t4 = torch.tensor(
        [96, 97, 98, 99, 246, 247, 248, 249, 36, 37, 38, 39, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    pos4, pt4, ckpt4, roff4 = prepare_step_metadata(
        attn_md,
        mamba_mds,
        num_reqs=num_reqs,
        bucket_t=16,
        max_pages_per_seq=max_pages,
        positions=positions_t4,
        tokens_per_seq=4,
    )
    assert pos4.shape == (16,)
    assert torch.equal(pos4[:12], positions_t4[:12])
    assert torch.all(pos4[12:] == 0)
    assert pt4.shape == (4, max_pages)
    assert torch.all(pt4[3] == 0)
    assert ckpt4.shape == (3, 4, 4)
    assert torch.all(ckpt4[:, 3, :] == 0)
    assert roff4.shape == (3, 4)
    assert roff4[0, :4].tolist() == [2, 3, 1, 0]

    pos1, pt1, ckpt1, roff1 = prepare_step_metadata(
        attn_md,
        mamba_mds,
        num_reqs=num_reqs,
        bucket_t=8,
        max_pages_per_seq=max_pages,
        tokens_per_seq=1,
    )
    assert pos1.shape == (8,)
    assert pos1[:3].tolist() == [99, 249, 39]
    assert pt1.shape == (8, max_pages)
    assert ckpt1.shape == (3, 8, 1)
    assert roff1.shape == (3, 8)
    assert torch.all(roff1 == 0)


def test_mk_ab_calibrator_and_hooks(monkeypatch):
    assert parse_mk_ab_enabled("1") is True
    assert parse_mk_ab_enabled("0") is False

    cal = MKABCalibrator(enabled=True, period_s=10.0, cap=4)
    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    assert cal.last_decision == ("mk_t4_b16", "mk")
    cal.record(10.0, padded_tokens=16, now=cal._t0 + 1.0)

    assert cal.choose("mk_t4_b16", True) == (False, "ab_stock_t4_b16", "stock")
    assert cal.last_decision == ("mk_t4_b16", "stock")
    cal.record(14.0, padded_tokens=16, now=cal._t0 + 2.0)

    assert cal.choose("mk_t4_b8", True) == (True, "mk_t4_b8", "mk")
    cal.record(8.0, padded_tokens=16, now=cal._t0 + 3.0)

    assert cal.choose("stock_t4_n13", False) == (False, "stock_t4_n13", "stock")
    cal.record(19.5, padded_tokens=64, now=cal._t0 + 4.0)

    assert cal.choose("mk_t4_b16", True) == (True, "mk_t4_b16", "mk")
    res = cal.record(12.0, padded_tokens=16, now=cal._t0 + 11.0)
    assert res is not None
    elapsed, _, _ = res
    assert elapsed >= 10.0
    cum = cal.summary_dict(window=False)
    assert cum["mk_t4_b16"]["mk"] == [2, 11.0, 10.2, 11.0, 11.8]
    assert cum["mk_t4_b16"]["stock"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["mk_t4_b16"]["stock_p16"] == [1, 14.0, 14.0, 14.0, 14.0]
    assert cum["stock_t4_n13"]["stock"] == [1, 19.5, 19.5, 19.5, 19.5]
    assert cum["stock_t4_n13"]["stock_p64"] == [1, 19.5, 19.5, 19.5, 19.5]
    assert cal.summary_dict(window=True) == {}

    cal.reset()
    assert cal.counts == {}
    assert cal.summary_dict(window=False) == {}

    mk = bare_mk(
        monkeypatch,
        tp_rank=0,
        _step_mix={},
        _step_mix_t0=time.monotonic(),
        _step_mix_period_s=1e9,
        ab_calibrator=VA.MKABCalibrator(enabled=True, period_s=1e9),
    )
    assert mk._note_step("mk_t4_b16", True) is True
    assert mk._note_step("mk_t4_b16", True) is False
    assert mk._note_step("stock_t4_n13", False) is False
    assert mk._step_mix == {"mk_t4_b16": 1, "ab_stock_t4_b16": 1, "stock_t4_n13": 1}
    mk.record_ab_step(12.5, padded_tokens=16)
    mk.reset_stats()
    assert mk._step_mix == {} and mk.ab_calibrator.counts == {}


def test_pallas_allreduce_hier1_uses_cross_core_offsets_357(monkeypatch):
    class _FakeRef:
        def __init__(self, tag="ref"):
            self.tag = tag

        @property
        def at(self):
            return self

        def __getitem__(self, item):
            return _FakeRef(f"{self.tag}[{item}]")

    recorded = []

    def _fake_remote_copy(
        *, src_ref, dst_ref, send_sem, recv_sem, device_id, device_id_type
    ):
        recorded.append((src_ref.tag, dst_ref.tag, device_id[0]))
        return SimpleNamespace(
            start=lambda: None, wait_recv=lambda: None, wait_send=lambda: None
        )

    monkeypatch.setattr(
        collectives.pltpu, "make_async_remote_copy", _fake_remote_copy
    )
    scr = {
        k: _FakeRef(k)
        for k in (
            "send",
            "recv",
            "ag_send",
            "ag_recv",
            "s2_send",
            "s2_recv",
            "ar_ssem",
            "ar_rsem",
        )
    }

    for me in range(8):
        recorded.clear()
        ctx_h1 = SimpleNamespace(
            HIER=1, RSAG=False, R=8, HC2=2048, ARP_H1=False, AR0S=False, scr=scr, me=me
        )
        collectives.PallasAllReduce(ctx_h1).ar_copies(slot=0, rnd_i=1)
        peers_h1 = [dev_id for _, _, dev_id in recorded]
        assert peers_h1 == [me ^ 3, me ^ 5, me ^ 7]
        # Every cross-chip peer is on the opposite core (bit 0 flipped) and
        # covers the other 3 chips in order (chip ^ 1, chip ^ 2, chip ^ 3).
        assert [(p & 1) for p in peers_h1] == [1 - (me & 1)] * 3
        assert [p // 2 for p in peers_h1] == [
            (me // 2) ^ 1,
            (me // 2) ^ 2,
            (me // 2) ^ 3,
        ]

    recorded.clear()
    ctx_h2 = SimpleNamespace(
        HIER=2, RSAG=False, R=8, HC2=2048, ARP_H1=False, AR0S=False, scr=scr, me=0
    )
    collectives.PallasAllReduce(ctx_h2).ar_copies(slot=0, rnd_i=1)
    assert [dev_id for _, _, dev_id in recorded] == [2, 4, 6]
