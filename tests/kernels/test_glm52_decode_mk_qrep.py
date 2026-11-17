# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The megakernel's q_b operand under VLLM_DCP_Q_REPLICATE=1 equals the qrep 0 one, byte for byte (CPU analogue of
best-config-compat check K3).

One synthetic GLM-5.2-FP8 q_b checkpoint in the real layout (fp8_e4m3 [NH*DQ, QL] = [16384, 2048], 128 x 128 block
scales [128, 16]) goes through the real stock load path twice per rank:
- qrep 0: vLLM `ColumnParallelLinear` at TP16 rank r (this rank's 4 heads);
- qrep 1: vLLM `DCPGroupColumnParallelLinear` with DCP 16 (all 64 heads, tp_size 1);
each with the real weight loaders, the real `VllmFp8LinearMethodTPU.process_weights_after_loading` (block dequant,
per-channel requant, [K, N] flip; TPU_FP8_LINEAR_PALLAS sites as served) and, NeoX on, the real
`permute_rope_to_half_split` (over the wrapper's num_heads * dcp_group_size heads). The adapter's `q_b_local` then
builds the megakernel's operands: the stacked qb dense operand and the smalls row's q_b_s field.

Needs glm-agentx's vLLM (the image this branch serves on); skipped where vllm_torchtpu's quantization package does not
import (main's vLLM).
"""

import contextlib
import types
from unittest import mock

import pytest
import torch

try:
    import vllm.model_executor.layers.linear as LIN
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        DCPGroupColumnParallelLinear,
    )

    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmMultiHeadLatentAttentionWrapper,
    )
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
except (
    ImportError
) as e:  # main's vLLM: vllm_torchtpu.layers.adapter.quantization does not import
    pytest.skip(f"needs glm-agentx's vLLM stack: {e}", allow_module_level=True)

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as VA
from vllm_torchtpu.kernels.glm52_decode_mk.config import rup

CF = C.FULL
R, QL, NH, DQ, HR = CF.R, CF.QL, CF.NH, CF.DQ, CF.HR
DN, DR, KL = CF.DN, CF.DR, CF.KL
N = NH * DQ
NQB = HR * DQ
PREFIX = "model.layers.0.self_attn.q_b_proj"


@pytest.fixture(scope="module")
def checkpoint():
    g = torch.Generator().manual_seed(52)
    raw = torch.randint(0, 256, (N, QL), dtype=torch.uint8, generator=g)
    raw[(raw & 0x7F) == 0x7F] = 0  # no NaN codes
    s_inv = torch.rand(N // 128, QL // 128, generator=g) * 2e-3 + 1e-5
    return raw.view(torch.float8_e4m3fn), s_inv


@contextlib.contextmanager
def _served(rank, dcp=R):
    """TP16 rank `rank`, DCP `dcp` (16, or 8 for V16): what get_tensor_model_parallel_* / get_current_vllm_config say
    on that rank."""
    vc = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=dcp)
    )
    with contextlib.ExitStack() as st:
        for t in (
            "vllm.distributed",
            "vllm.distributed.parallel_state",
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.parameter",
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
        ):
            st.enter_context(
                mock.patch(
                    f"{t}.get_tensor_model_parallel_world_size", lambda: R, create=True
                )
            )
            st.enter_context(
                mock.patch(
                    f"{t}.get_tensor_model_parallel_rank", lambda: rank, create=True
                )
            )
        st.enter_context(
            mock.patch.object(LIN, "get_current_vllm_config", lambda: vc, create=True)
        )
        yield


def _q_b_proj(cls, rank, checkpoint, dcp=R):
    w, s_inv = checkpoint
    quant = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    with _served(rank, dcp):
        layer = cls(
            input_size=QL,
            output_size=N,
            bias=False,
            quant_config=quant,
            params_dtype=torch.bfloat16,
            gather_output=False,
            prefix=PREFIX,
        )
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, s_inv)
        layer.quant_method.process_weights_after_loading(layer)
    return layer


def _permute_neox(q_b_proj, group_size):
    """The real TPU_MLA_ROPE_NEOX_PERMUTE post-load hook on a wrapper holding this q_b_proj (its kv side is a stub
    fused_qkv_a_proj of the real width)."""
    wr = VllmMultiHeadLatentAttentionWrapper.__new__(
        VllmMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wr)
    wr.kimi_preprocess = None
    wr.q_lora_rank, wr.kv_lora_rank = QL, KL
    wr.qk_nope_head_dim, wr.qk_rope_head_dim, wr.qk_head_dim = DN, DR, DQ
    wr.num_heads, wr.dcp_group_size = HR, group_size
    wr.dcp_q_replicate = group_size > 1
    wr.q_b_proj, wr.q_proj, wr.kv_a_proj_with_mqa = q_b_proj, None, None
    kv = torch.nn.Module()
    kv.weight = torch.nn.Parameter(
        torch.zeros(QL + KL + DR, 8, dtype=torch.bfloat16), requires_grad=False
    )
    wr.fused_qkv_a_proj = kv
    wr._rope_neox_permuted = False
    wr.permute_rope_to_half_split()
    assert wr._rope_neox_permuted


def _operands(layer, rank, qrep, cfg=CF):
    """The megakernel's q_b operands from a processed q_b_proj, exactly as Glm52DecodeMK builds them: `q_b_local`
    (via `_layer`), the stacked dense copy (`_collect_weights`: _own(torch.stack)) and the q_b_s field of a smalls
    row (`_build_smalls`: f32, zero-padded to a lane multiple)."""
    n = layer.weight.numel() // QL
    qw, qs = VA.q_b_local(
        layer.weight, VA._scale(layer, n, "q_b_proj"), cfg, rank, qrep=qrep
    )
    qb = VA._own(torch.stack([qw.detach()]))
    q_b_s = torch.nn.functional.pad(qs.reshape(-1).float(), (0, rup(NQB) - NQB))
    return qb, q_b_s


def _bytes(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("neox", [False, True], ids=["interleaved", "neox"])
@pytest.mark.parametrize("rank", [0, 5, R - 1])
def test_qrep1_q_b_operand_equals_qrep0(checkpoint, rank, neox):
    tp = _q_b_proj(ColumnParallelLinear, rank, checkpoint)
    rep = _q_b_proj(DCPGroupColumnParallelLinear, rank, checkpoint)
    assert not getattr(tp, "qrep_active", False)
    assert rep.qrep_active and rep.group_size == R and rep.rank_in_group == rank
    assert tuple(tp.weight.shape) == (QL, NQB) and tuple(rep.weight.shape) == (QL, N)
    if neox:
        _permute_neox(tp, 1)
        _permute_neox(rep, R)
    assert VA.qrep_layer(rep, R, rank, 0) and not VA.qrep_layer(tp, R, rank, 0)
    qb0, s0 = _operands(tp, rank, qrep=False)
    qb1, s1 = _operands(rep, rank, qrep=True)
    assert qb0.shape == qb1.shape == (1, QL, NQB) and qb1.dtype == torch.float8_e4m3fn
    assert torch.equal(_bytes(qb1), _bytes(qb0))
    assert torch.equal(_bytes(s1), _bytes(s0))
    # positive controls: the neighbour's heads, and the other rope layout, are different operands
    qbn, _ = _operands(rep, (rank + 1) % R, qrep=True)
    assert not torch.equal(_bytes(qbn), _bytes(qb0))
    if neox:
        plain, _ = _operands(
            _q_b_proj(ColumnParallelLinear, rank, checkpoint), rank, qrep=False
        )
        assert not torch.equal(_bytes(plain), _bytes(qb1))


@pytest.mark.parametrize("neox", [False, True], ids=["interleaved", "neox"])
@pytest.mark.parametrize("rank", [0, 5, 8, R - 1])
def test_v16_dcp8_qrep1_q_b_operand_equals_qrep0(checkpoint, rank, neox):
    """V16 (DCP 8, VLLM_DCP_Q_REPLICATE=1): the real DCPGroupColumnParallelLinear holds lane rank // 8's 32 heads;
    q_b_local's slice [(rank % 8) * HR, +HR) of it is byte-identical to the TP16 column-parallel shard."""
    import dataclasses

    cfg8 = dataclasses.replace(CF, DCP=8)
    tp = _q_b_proj(ColumnParallelLinear, rank, checkpoint)
    rep = _q_b_proj(DCPGroupColumnParallelLinear, rank, checkpoint, dcp=8)
    assert rep.qrep_active and rep.group_size == 8 and rep.rank_in_group == rank % 8
    assert tuple(rep.weight.shape) == (QL, N // 2)
    if neox:
        _permute_neox(tp, 1)
        _permute_neox(rep, 8)
    assert VA.qrep_layer(rep, R, rank, 0, 8)
    qb0, s0 = _operands(tp, rank, qrep=False)
    qb1, s1 = _operands(rep, rank, qrep=True, cfg=cfg8)
    assert torch.equal(_bytes(qb1), _bytes(qb0))
    assert torch.equal(_bytes(s1), _bytes(s0))
    # positive control: the replica's neighbour in the group holds other heads
    qbn, _ = _operands(rep, rank + 1 if rank % 8 != 7 else rank - 1, qrep=True, cfg=cfg8)
    assert not torch.equal(_bytes(qbn), _bytes(qb0))
