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
"""CPU unit checks for `oxpd` whole-step row limit, DCP trim, scratch alias, and draft operands."""

from __future__ import annotations

import pytest

from vllm_torchtpu.megakernel_runtime import (
    qwen38_driver as drv,
    qwen38_wholestep_call as wsc,
)


def test_whole_max_rows_defaults_and_override(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_MAX_ROWS", raising=False)
    assert drv._whole_max_rows() == drv.WHOLE_MAX_ROWS == 96
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_MAX_ROWS", "0")
    assert drv._whole_max_rows() == 96
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_MAX_ROWS", "128")
    assert drv._whole_max_rows() == 128
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_MAX_ROWS", "129")
    with pytest.raises(ValueError, match="kernel limit"):
        drv._whole_max_rows()


@pytest.mark.parametrize(
    ("dcp", "B", "trimmed"),
    [
        (1, 128, False),
        (2, 96, False),
        (2, 128, True),
    ],
)
def test_dcp_trim_drops_one_expert_slot_at_128_rows(monkeypatch, dcp, B, trimmed):
    monkeypatch.setattr(wsc, "main_kv_dcp", lambda: dcp)
    monkeypatch.delenv("VLLM_TPU_MK_VMEM_CUTS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_RINGS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_DCP_TRIM_NBE", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_MOE_AR2D", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_MOE_E7", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_AR2D_SINGLE_SLOT", raising=False)
    off = wsc.wholestep_tiles(B, 16384)
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_DCP_TRIM_NBE", "1")
    on = wsc.wholestep_tiles(B, 16384)
    assert on.nbe == (max(2, off.nbe - 1) if trimmed else off.nbe)
    assert on.nbd == off.nbd


def test_dcp_trim_skipped_with_ar2d_single_slot(monkeypatch):
    monkeypatch.setattr(wsc, "main_kv_dcp", lambda: 2)
    monkeypatch.delenv("VLLM_TPU_MK_VMEM_CUTS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_RINGS", raising=False)
    monkeypatch.setenv("VLLM_TPU_MK_MOE_AR2D", "1")
    monkeypatch.setenv("VLLM_TPU_MK_AR2D_SINGLE_SLOT", "1")
    monkeypatch.delenv("VLLM_TPU_MK_WHOLE_DCP_TRIM_NBE", raising=False)
    off = wsc.wholestep_tiles(128, 16384)
    monkeypatch.setenv("VLLM_TPU_MK_WHOLE_DCP_TRIM_NBE", "1")
    on = wsc.wholestep_tiles(128, 16384)
    assert off.hier == 4 and off.ar2d_single_slot is True
    assert on.nbe == off.nbe


def test_scratch_alias_tile_flag(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_SCRATCH_ALIAS", raising=False)
    monkeypatch.delenv("VLLM_TPU_MK_VMEM_CUTS", raising=False)
    assert wsc.wholestep_tiles(64, 16384).scratch_alias is False
    monkeypatch.setenv("VLLM_TPU_MK_SCRATCH_ALIAS", "1")
    assert wsc.wholestep_tiles(16, 16384).scratch_alias is True
    assert wsc.wholestep_tiles(64, 16384).scratch_alias is True
    monkeypatch.delenv("VLLM_TPU_MK_SCRATCH_ALIAS", raising=False)
    monkeypatch.setenv("VLLM_TPU_MK_VMEM_CUTS", "alias")
    assert wsc.wholestep_tiles(16, 16384).scratch_alias is False
    assert wsc.wholestep_tiles(64, 16384).scratch_alias is True


def test_unsupported_vmem_cut_token_raises(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MK_VMEM_CUTS", "io,ring")
    with pytest.raises(ValueError, match="unsupported tokens"):
        wsc.wholestep_tiles(64, 16384)


def test_draft_attn_clean_operands_flag(monkeypatch):
    from types import SimpleNamespace

    import torch
    from torch import nn

    from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as da

    monkeypatch.setattr(da.qd, "draft_attn_spec", lambda *a, **k: None)
    monkeypatch.setattr(da, "draft_indexed", lambda *a, **k: {})
    monkeypatch.setattr(
        da,
        "pack_draft_attn_operands",
        lambda *a, **k: [torch.zeros((2, 4), dtype=torch.bfloat16)],
    )
    mesh = SimpleNamespace(size=1)
    kind = SimpleNamespace(opts={"budget": 1, "max_groups": 1, "pages_per_seq": 1, "dcp": 1})

    monkeypatch.delenv("VLLM_TPU_MK_DRAFT_CLEAN_OPERANDS", raising=False)
    mod_buf = da.DraftAttn(None, None, mesh, kind, 4)
    assert "w_0" in dict(mod_buf.named_buffers())
    assert "w_0" not in dict(mod_buf.named_parameters())

    monkeypatch.setenv("VLLM_TPU_MK_DRAFT_CLEAN_OPERANDS", "1")
    mod_par = da.DraftAttn(None, None, mesh, kind, 4)
    assert isinstance(mod_par.w_0, nn.Parameter)
    assert not mod_par.w_0.requires_grad
    assert "w_0" in dict(mod_par.named_parameters())

