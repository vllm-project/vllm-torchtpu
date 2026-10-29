# SPDX-License-Identifier: Apache-2.0
"""Tests for ``transforms.microbatch`` context manager and FX region rewriter pass."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.fx as fx
from torch import nn

from vllm_torchtpu import envs, transforms
from vllm_torchtpu.layers.adapter import cp_mla_attention


# ---------------------------------------------------------------------------
# Mock Collective & Compute Custom Ops for Deterministic FX & Eager Testing
# ---------------------------------------------------------------------------


@torch.library.custom_op("tpu_transforms_test::all_gather_dim0", mutates_args=())
def _mock_all_gather_dim0(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


@_mock_all_gather_dim0.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("tpu_transforms_test::all_reduce", mutates_args=())
def _mock_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x * 1.5


@_mock_all_reduce.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "tpu_transforms_test::reduce_scatter_heads", mutates_args=()
)
def _mock_reduce_scatter_heads(
    x: torch.Tensor, world_size: int
) -> torch.Tensor:
    # x is heads-major [H, T, V] -> returns [H // world_size, T, V]
    h = x.shape[0] // world_size
    pieces = [x[r * h : (r + 1) * h] for r in range(world_size)]
    return sum(pieces)


@_mock_reduce_scatter_heads.register_fake
def _(x: torch.Tensor, world_size: int) -> torch.Tensor:
    h, t, v = x.shape
    return x.new_empty((h // world_size, t, v))


@torch.library.custom_op("tpu_transforms_test::wait_tensor", mutates_args=())
def _mock_wait_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_mock_wait_tensor.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def tensor_model_parallel_all_gather(
    x_shard: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    del dim
    out = torch.ops.tpu_transforms_test.all_gather_dim0(x_shard)
    return torch.ops.tpu_transforms_test.wait_tensor(out)


def tensor_model_parallel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    out = torch.ops.tpu_transforms_test.all_reduce(x)
    return torch.ops.tpu_transforms_test.wait_tensor(out)


class _MockGateUpProj(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.nn.functional.linear(x, self.weight), None


class _MockRowParallelOProj(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with transforms.microbatching.pipeline_stage():
            partial = torch.nn.functional.linear(x, self.weight)
        with transforms.microbatching.pipeline_stage():
            out = tensor_model_parallel_all_reduce(partial)
        return out, None


class _MockDcpGroup:
    def __init__(self, world_size: int = 2) -> None:
        self.world_size = world_size

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        del dim
        # Repeat x `[1, T, H]` -> `[world_size, T, H]` via a custom collective op.
        gathered = torch.ops.tpu_transforms_test.all_gather_dim0(
            x.expand(self.world_size, -1, -1).contiguous()
        )
        return torch.ops.tpu_transforms_test.wait_tensor(gathered)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if dim == 0:
            out = torch.ops.tpu_transforms_test.reduce_scatter_heads(
                x, self.world_size
            )
            return torch.ops.tpu_transforms_test.wait_tensor(out)
        # dim == 1: [T, H, V] -> transpose to [H, T, V], reduce_scatter, transpose back
        out = torch.ops.tpu_transforms_test.reduce_scatter_heads(
            x.transpose(0, 1).contiguous(), self.world_size
        )
        return torch.ops.tpu_transforms_test.wait_tensor(out).transpose(0, 1)


def _capture_dynamic_fx_graph(
    fn: Any, *example_args: torch.Tensor
) -> tuple[fx.GraphModule, list[Any]]:
    """Capture a single dynamic-token FX graph via ``torch.compile``."""
    torch._dynamo.reset()
    captured: list[tuple[fx.GraphModule, list[Any]]] = []

    def _capture_backend(gm: fx.GraphModule, example_inputs: list[Any]):
        captured.append((copy.deepcopy(gm), list(example_inputs)))
        return gm.forward

    for arg in example_args:
        if isinstance(arg, torch.Tensor) and arg.ndim > 0:
            torch._dynamo.mark_dynamic(arg, 0)

    compiled = torch.compile(fn, backend=_capture_backend)
    compiled(*example_args)
    assert len(captured) == 1, f"Expected 1 captured FX graph, got {len(captured)}"
    return captured[0]


def _count_targets(gm: fx.GraphModule, substring: str) -> int:
    count = 0
    for n in gm.graph.nodes:
        if n.op in ("call_function", "call_method"):
            name = getattr(n.target, "__name__", str(n.target)).lower()
            if substring in name:
                count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Collective -> Compute: AllGather -> gate_up_proj
# ---------------------------------------------------------------------------


def test_collective_to_compute_all_gather_gate_up_proj() -> None:
    torch.manual_seed(0)

    class AllGatherMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = _MockGateUpProj(16, 32)

        def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
            with transforms.microbatch(min_len=1024, groups=2):
                with transforms.microbatching.pipeline_stage():
                    x_full = tensor_model_parallel_all_gather(x_shard, dim=0)
                with transforms.microbatching.pipeline_stage():
                    gate_up, _ = self.gate_up_proj(x_full)
            return gate_up + 1.0

    model = AllGatherMLP().eval()
    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        model, torch.randn(64, 16)
    )

    # Bucket 512 (< min_len=1024): markers stripped, unsplit graph.
    gm_512 = copy.deepcopy(gm_orig)
    rewritten_512 = transforms.apply_microbatch_pass(
        gm_512, example_inputs, size=512
    )
    assert rewritten_512 == 0
    assert _count_targets(gm_512, "microbatch_enter") == 0
    assert _count_targets(gm_512, "microbatch_exit") == 0
    assert _count_targets(gm_512, "all_gather_dim0") == 1
    assert _count_targets(gm_512, "linear") == 1
    assert _count_targets(gm_512, "microbatch_barrier_2") == 0

    x_512 = torch.randn(512, 16)
    ref_512 = (tensor_model_parallel_all_gather(x_512, dim=0) @ model.gate_up_proj.weight.T) + 1.0
    out_512 = model(x_512)
    torch.testing.assert_close(out_512, ref_512)

    # Bucket 1024 (>= min_len=1024): rewritten into 2-stage diamond pipeline!
    gm_1024 = copy.deepcopy(gm_orig)
    rewritten_1024 = transforms.apply_microbatch_pass(
        gm_1024, example_inputs, size=1024
    )
    assert rewritten_1024 == 1
    assert _count_targets(gm_1024, "microbatch_enter") == 0
    assert _count_targets(gm_1024, "microbatch_exit") == 0
    assert _count_targets(gm_1024, "slice") == 2
    assert _count_targets(gm_1024, "all_gather_dim0") == 2
    assert _count_targets(gm_1024, "linear") == 2
    assert _count_targets(gm_1024, "microbatch_barrier_2") == 2
    assert _count_targets(gm_1024, "cat") == 1

    # Verify topological order of the diamond:
    # Stage1(0) [all_gather_0] -> Barrier1 -> Stage2(0) [linear_0] -> Stage1(1) [all_gather_1] -> Barrier2 -> Stage2(1) [linear_1]
    op_sequence = [
        getattr(n.target, "__name__", str(n.target)).lower()
        for n in gm_1024.graph.nodes
        if n.op == "call_function"
        and any(
            k in getattr(n.target, "__name__", str(n.target)).lower()
            for k in ("all_gather_dim0", "linear", "microbatch_barrier_2", "cat")
        )
    ]
    assert len(op_sequence) == 7
    assert "all_gather_dim0" in op_sequence[0]
    assert "microbatch_barrier_2" in op_sequence[1]
    assert "linear" in op_sequence[2]
    assert "all_gather_dim0" in op_sequence[3]
    assert "microbatch_barrier_2" in op_sequence[4]
    assert "linear" in op_sequence[5]
    assert "cat" in op_sequence[6]

    # Verify numerical equivalence on concrete 1024-token input!
    x_1024 = torch.randn(1024, 16)
    ref_1024 = (tensor_model_parallel_all_gather(x_1024, dim=0) @ model.gate_up_proj.weight.T) + 1.0
    # Execute the rewritten FX GraphModule directly with concrete args
    weight_param = model.gate_up_proj.weight
    # Find placeholder order in gm_1024
    phs = [n for n in gm_1024.graph.nodes if n.op == "placeholder"]
    call_args = []
    for ph in phs:
        val = ph.meta.get("example_value")
        if isinstance(val, torch.SymInt):
            call_args.append(1024)
        elif "weight" in ph.name:
            call_args.append(weight_param)
        else:
            call_args.append(x_1024)
    out_1024 = gm_1024(*call_args)
    if isinstance(out_1024, tuple):
        out_1024 = out_1024[0]
    torch.testing.assert_close(out_1024, ref_1024)


# ---------------------------------------------------------------------------
# 2. Compute -> Collective: GLM-5.2 o_proj (GEMM -> AllReduce)
# ---------------------------------------------------------------------------


def test_compute_to_collective_o_proj_all_reduce() -> None:
    torch.manual_seed(1)

    class OProjBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = _MockRowParallelOProj(16, 32)

        def forward(self, attn_out: torch.Tensor) -> torch.Tensor:
            with transforms.microbatch(min_len=1024, groups=2):
                out, _ = self.o_proj(attn_out)
            return out

    model = OProjBlock().eval()
    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        model, torch.randn(64, 16)
    )

    gm_1024 = copy.deepcopy(gm_orig)
    rewritten = transforms.apply_microbatch_pass(
        gm_1024, example_inputs, size=1024
    )
    assert rewritten == 1

    # Verify topological order of Compute -> Collective diamond:
    # Stage1(0) [linear_0] -> Barrier1 -> Stage2(0) [all_reduce_0] -> Stage1(1) [linear_1] -> Barrier2 -> Stage2(1) [all_reduce_1] -> cat
    op_sequence = [
        getattr(n.target, "__name__", str(n.target)).lower()
        for n in gm_1024.graph.nodes
        if n.op == "call_function"
        and any(
            k in getattr(n.target, "__name__", str(n.target)).lower()
            for k in ("all_reduce", "linear", "microbatch_barrier_2", "cat")
        )
    ]
    assert len(op_sequence) == 7
    assert "linear" in op_sequence[0]
    assert "microbatch_barrier_2" in op_sequence[1]
    assert "all_reduce" in op_sequence[2]
    assert "linear" in op_sequence[3]
    assert "microbatch_barrier_2" in op_sequence[4]
    assert "all_reduce" in op_sequence[5]
    assert "cat" in op_sequence[6]

    x_1024 = torch.randn(1024, 16)
    ref_1024 = tensor_model_parallel_all_reduce(x_1024 @ model.o_proj.weight.T)
    phs = [n for n in gm_1024.graph.nodes if n.op == "placeholder"]
    call_args = [
        1024
        if isinstance(ph.meta.get("example_value"), torch.SymInt)
        else (model.o_proj.weight if "weight" in ph.name else x_1024)
        for ph in phs
    ]
    out_1024 = gm_1024(*call_args)
    if isinstance(out_1024, tuple):
        out_1024 = out_1024[0]
    torch.testing.assert_close(out_1024, ref_1024)


# ---------------------------------------------------------------------------
# 3. GLM-5.2 DCP V-Space Merge via transforms.microbatch (groups=2 and groups=8)
# ---------------------------------------------------------------------------


def test_glm52_cp_mla_attention_vspace_merge_with_transforms(monkeypatch) -> None:
    torch.manual_seed(2)
    dcp_group = _MockDcpGroup(world_size=2)
    monkeypatch.setattr(cp_mla_attention, "get_dcp_group", lambda: dcp_group)
    monkeypatch.setattr(envs, "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS", 1024)
    monkeypatch.setattr(envs, "TPU_DCP_MERGE_CHUNKED_LSE_AG", False)
    monkeypatch.setattr(envs, "TPU_DCP_MLA_MERGE_PIECES", 0)

    num_heads = 8
    kv_lora_rank = 16
    v_head_dim = 12
    w_uv = torch.randn(num_heads, kv_lora_rank, v_head_dim)
    w_uv_scale = torch.rand(num_heads, 1, v_head_dim)

    def _fn_groups_4(
        partial: torch.Tensor, lse: torch.Tensor
    ) -> torch.Tensor:
        return cp_mla_attention._merge_chunked(
            partial,
            lse,
            dcp_group,
            chunks=4,
            rs_dtype=torch.bfloat16,
            vspace=(w_uv, w_uv_scale),
        )

    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        _fn_groups_4,
        torch.randn(128, num_heads, kv_lora_rank, dtype=torch.bfloat16),
        torch.randn(128, num_heads, dtype=torch.float32),
    )

    gm_2048 = copy.deepcopy(gm_orig)
    rewritten = transforms.apply_microbatch_pass(
        gm_2048, example_inputs, size=2048
    )
    assert rewritten == 1
    # 2 sliced live-ins (`partial` and `lse_f32`) x 4 groups = 8 slice ops.
    # Weights `w_uv` and `w_uv_scale` must NOT be sliced!
    assert _count_targets(gm_2048, "slice") == 8
    assert _count_targets(gm_2048, "bmm") == 4
    assert _count_targets(gm_2048, "reduce_scatter_heads") == 4
    # For groups=4 and 2 sliced inputs feeding Stage 1:
    # Each of the 3 diamond transitions emits 2 Barrier-1 calls (for partial and lse)
    # + 1 Barrier-2 call (synchronizing Stage2(g) with Stage1(g+1)) = 9 barrier calls.
    assert _count_targets(gm_2048, "microbatch_barrier_2") == 9
    assert _count_targets(gm_2048, "cat.default") == 1

    # When TPU_DCP_MLA_MERGE_PIECES=2, barrier=False so XLA's SliceConcatForwarding
    # can fuse directly with the Pallas MLA pieces.
    monkeypatch.setattr(envs, "TPU_DCP_MLA_MERGE_PIECES", 2)
    gm_pieces, example_inputs_pieces = _capture_dynamic_fx_graph(
        _fn_groups_4,
        torch.randn(128, num_heads, kv_lora_rank, dtype=torch.bfloat16),
        torch.randn(128, num_heads, dtype=torch.float32),
    )
    rewritten_pieces = transforms.apply_microbatch_pass(
        gm_pieces, example_inputs_pieces, size=2048
    )
    assert rewritten_pieces == 1
    assert _count_targets(gm_pieces, "microbatch_barrier_2") == 0
    assert _count_targets(gm_pieces, "reduce_scatter_heads") == 4

    # Compare numerical output against the eager reference execution.
    partial_2048 = torch.randn(
        2048, num_heads, kv_lora_rank, dtype=torch.bfloat16
    )
    lse_2048 = torch.randn(2048, num_heads, dtype=torch.float32)

    ref_2048 = cp_mla_attention._merge_chunked(
        partial_2048,
        lse_2048,
        dcp_group,
        chunks=4,
        rs_dtype=torch.bfloat16,
        vspace=(w_uv, w_uv_scale),
    )

    phs = [n for n in gm_2048.graph.nodes if n.op == "placeholder"]
    call_args = []
    for ph in phs:
        val = ph.meta.get("example_value")
        if isinstance(val, torch.SymInt):
            call_args.append(2048)
        elif "partial" in ph.name:
            call_args.append(partial_2048)
        elif "lse" in ph.name:
            call_args.append(lse_2048)
        elif "scale" in ph.name:
            call_args.append(w_uv_scale)
        else:
            call_args.append(w_uv)

    out_2048 = gm_2048(*call_args)
    if isinstance(out_2048, tuple):
        out_2048 = out_2048[0]
    torch.testing.assert_close(out_2048, ref_2048)


# ---------------------------------------------------------------------------
# 4. Qwen3.5 GDN + Full-Attention + MoE Decoder Layers via transforms.microbatch
# ---------------------------------------------------------------------------


@torch.library.custom_op("tpu_transforms_test::gdn_pooled_op", mutates_args=())
def _mock_gdn_pooled_op(
    mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    return mixed_qkv * torch.sigmoid(b + a)


@_mock_gdn_pooled_op.register_fake
def _(
    mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    del b, a
    return torch.empty_like(mixed_qkv)


@torch.library.custom_op("tpu_transforms_test::fused_moe", mutates_args=())
def _mock_fused_moe(
    hidden_states: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    return hidden_states * torch.sigmoid(router_logits[:, :1])


@_mock_fused_moe.register_fake
def _(
    hidden_states: torch.Tensor, router_logits: torch.Tensor
) -> torch.Tensor:
    del router_logits
    return torch.empty_like(hidden_states)


def test_qwen35_gdn_and_moe_layer_microbatch_pipeline(monkeypatch) -> None:
    """Verify Qwen3.5 GDN + MoE + Next-Layer Pre-Attn microbatching:
    - Coalesces `input_layernorm` + `in_proj_qkvz`/`in_proj_ba` into 1 region
    - Hoists `local_padding_mask[:num_tokens]` and slices it per chunk `[g*step:(g+1)*step]`
    - Forwards `z` from Pre-Attn to Post-Attn and `(hidden_states, residual)` from
      Post-Attn -> MoE -> Next-Layer Pre-Attn with zero intermediate `cat`/`slice`
    - Forms all 4 diamond barriers across Attn-OutProj, MoE, and Next-Layer Pre-Attn
    """
    from vllm_torchtpu.models.vllm.qwen3_next_patch import (
        patched_attention_forward,
        patched_decoder_layer_forward,
    )

    torch.manual_seed(3)
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 2)
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_MIN_TOKENS", 1024)

    class _MockRMSNorm(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(
            self, x: torch.Tensor, residual: torch.Tensor | None = None
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if residual is None:
                return x * self.weight
            res = x + residual
            return res * self.weight, res

    class _MockGdnAttn(nn.Module):
        def __init__(self, dim: int, padding_mask: torch.Tensor) -> None:
            super().__init__()
            self.in_proj_qkvz = _MockGateUpProj(dim, dim * 2)
            self.in_proj_ba = _MockGateUpProj(dim, dim * 2)
            self.out_proj = _MockRowParallelOProj(dim, dim)
            self.local_padding_mask = padding_mask
            self.dim = dim

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            with transforms.microbatch(
                min_len=envs.TPU_QWEN35_MICROBATCH_MIN_TOKENS,
                groups=envs.TPU_QWEN35_MICROBATCH_GROUPS,
            ):
                qkvz, _ = self.in_proj_qkvz(hidden_states)
                ba, _ = self.in_proj_ba(hidden_states)
                mask = self.local_padding_mask[: hidden_states.shape[0]]
                qkvz = qkvz * mask
                mixed_qkv, z = qkvz.split([self.dim, self.dim], dim=-1)
                b, a = ba.chunk(2, dim=-1)

            # Unsplit stateful Pallas GDN kernel on [0:N]
            core_out = torch.ops.tpu_transforms_test.gdn_pooled_op(
                mixed_qkv, b, a
            )

            with transforms.microbatch(
                min_len=envs.TPU_QWEN35_MICROBATCH_MIN_TOKENS,
                groups=envs.TPU_QWEN35_MICROBATCH_GROUPS,
            ):
                gated = core_out * torch.nn.functional.silu(z)
                out, _ = self.out_proj(gated)
            return out

    class _MockFullAttn(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.qkv_proj = _MockGateUpProj(dim, dim * 4)
            self.o_proj = _MockRowParallelOProj(dim, dim)
            self.dim = dim

        def _project_qkv_gate(
            self, qkv: torch.Tensor, positions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del positions
            q, k, v, gate = qkv.split(
                [self.dim, self.dim, self.dim, self.dim], dim=-1
            )
            return q, k, v, gate

        def attn(
            self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
        ) -> torch.Tensor:
            # Unsplit Pallas RPAm attention kernel on [0:N]
            return torch.ops.tpu_transforms_test.gdn_pooled_op(q, k, v)

        forward = patched_attention_forward

    class _MockMoEBlock(nn.Module):
        def __init__(self, dim: int, padding_mask: torch.Tensor) -> None:
            super().__init__()
            self.gate = nn.Linear(dim, 4, bias=False)
            self.shared_expert = nn.Linear(dim, dim, bias=False)
            self.local_padding_mask = padding_mask

        def forward(
            self,
            hidden_states: torch.Tensor,
            already_sequence_parallel: bool = False,
        ) -> torch.Tensor:
            del already_sequence_parallel
            with transforms.microbatching.pipeline_stage():
                mask = self.local_padding_mask[: hidden_states.shape[0]]
                router_logits = self.gate(hidden_states) * mask
                shared_out = self.shared_expert(hidden_states) * mask
                routed_out = torch.ops.tpu_transforms_test.fused_moe(
                    hidden_states, router_logits
                )
                combined = shared_out + routed_out
            with transforms.microbatching.pipeline_stage():
                return tensor_model_parallel_all_reduce(combined)

    class _MockDecoderLayer(nn.Module):
        def __init__(
            self, dim: int, layer_type: str, padding_mask: torch.Tensor
        ) -> None:
            super().__init__()
            self.use_attn_reduce_scatter_for_moe = False
            self.layer_type = layer_type
            self.layer_scale = False
            self.input_layernorm = _MockRMSNorm(dim)
            self.post_attention_layernorm = _MockRMSNorm(dim)
            if layer_type == "linear_attention":
                self.linear_attn = _MockGdnAttn(dim, padding_mask)
            else:
                self.self_attn = _MockFullAttn(dim)
            self.mlp = _MockMoEBlock(dim, padding_mask)

        forward = patched_decoder_layer_forward

    class _TwoLayerQwen35(nn.Module):
        def __init__(self, dim: int, padding_mask: torch.Tensor) -> None:
            super().__init__()
            self.layer0 = _MockDecoderLayer(
                dim, "linear_attention", padding_mask
            )
            self.layer1 = _MockDecoderLayer(dim, "full_attention", padding_mask)

        def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            hidden_states, residual = self.layer0(
                hidden_states, residual, positions
            )
            hidden_states, residual = self.layer1(
                hidden_states, residual, positions
            )
            return hidden_states, residual

    # Non-uniform padding mask across the two 512-token halves of a 1024-token bucket
    # so any bug that slices `[0:512]` twice instead of `[0:512]` and `[512:1024]` fails!
    padding_mask = torch.ones(2048, 1)
    padding_mask[768:1024] = 0.0

    dim = 16
    model = _TwoLayerQwen35(dim, padding_mask).eval()
    gm_orig, example_inputs = _capture_dynamic_fx_graph(
        model,
        torch.randn(64, dim),
        torch.randn(64, dim),
        torch.arange(64),
    )

    gm_1024 = copy.deepcopy(gm_orig)
    rewritten = transforms.apply_microbatch_pass(
        gm_1024, example_inputs, size=1024
    )
    # 2 layers x 3 regions/layer (coalesced Pre-Attn, Post-Attn, MoE) = 6 rewritten regions
    assert rewritten == 6
    assert _count_targets(gm_1024, "microbatch_enter") == 0
    assert _count_targets(gm_1024, "microbatch_exit") == 0
    # Core GDN and Full Attention Pallas ops remain unsplit (1 per layer = 2 total)
    assert _count_targets(gm_1024, "gdn_pooled_op") == 2
    # MoE fused kernel and all_reduce are microbatched into 2 chunks per region
    assert _count_targets(gm_1024, "fused_moe") == 4
    assert _count_targets(gm_1024, "all_reduce") == 8
    # 2-stage diamonds (4 regions x 2 barriers) + single-stage Pre-Attn (2 regions x 1 barrier) = 10 barriers
    assert _count_targets(gm_1024, "microbatch_barrier_2") == 10

    # Verify numerical equivalence against the unsplit eager model on 1024 tokens
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 0)
    hs_1024 = torch.randn(1024, dim)
    res_1024 = torch.randn(1024, dim)
    pos_1024 = torch.arange(1024)
    ref_hs, ref_res = model(hs_1024, res_1024, pos_1024)

    # Execute gm_1024 with concrete 1024-token inputs
    gm_512 = copy.deepcopy(gm_orig)
    transforms.apply_microbatch_pass(gm_512, example_inputs, size=512)
    # Build concrete args matching gm_orig's placeholders using a 1024-token capture
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 2)
    gm_1024_cap, inputs_1024 = _capture_dynamic_fx_graph(
        model, hs_1024, res_1024, pos_1024
    )
    transforms.apply_microbatch_pass(gm_1024_cap, inputs_1024, size=1024)
    out_hs, out_res = gm_1024_cap(*inputs_1024)
    torch.testing.assert_close(out_hs, ref_hs, rtol=1e-4, atol=2e-2)
    torch.testing.assert_close(out_res, ref_res, rtol=1e-4, atol=2e-2)


def test_qwen35_sequence_parallel_moe_microbatch(monkeypatch) -> None:
    """Verify Qwen3.5 SP (`use_attn_reduce_scatter_for_moe=True`) microbatching
    where the MoE region slices `[s0 // tp, H]` inputs using `_eval_dim_for_bucket`."""
    from vllm_torchtpu.models.vllm import qwen3_next_patch as patch_mod
    from vllm_torchtpu.models.vllm.qwen3_next_patch import (
        patched_decoder_layer_forward,
    )

    torch.manual_seed(4)
    tp = 2
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 2)
    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_MIN_TOKENS", 1024)
    monkeypatch.setattr(
        patch_mod, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: torch.cat([x] * tp, dim=dim),
    )
    monkeypatch.setattr(
        patch_mod,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: x.narrow(dim, 0, x.shape[dim] // tp),
    )

    class _MockRMSNorm(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(
            self, x: torch.Tensor, residual: torch.Tensor | None = None
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if residual is None:
                return x * self.weight
            res = x + residual
            return res * self.weight, res

    class _MockSPMoEBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gate = nn.Linear(dim, 4, bias=False)
            self.shared_expert = nn.Linear(dim, dim, bias=False)

        def forward(
            self,
            hidden_states: torch.Tensor,
            already_sequence_parallel: bool = False,
        ) -> torch.Tensor:
            assert already_sequence_parallel
            with transforms.microbatching.pipeline_stage():
                router_logits = self.gate(hidden_states)
                shared_out = self.shared_expert(hidden_states)
                routed_out = torch.ops.tpu_transforms_test.fused_moe(
                    hidden_states, router_logits
                )
                combined = shared_out + routed_out
            with transforms.microbatching.pipeline_stage():
                return tensor_model_parallel_all_reduce(combined)

    class _MockSPLayer(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.use_attn_reduce_scatter_for_moe = True
            self.layer_type = "linear_attention"
            self.layer_scale = False
            self.input_layernorm = _MockRMSNorm(dim)
            self.post_attention_layernorm = _MockRMSNorm(dim)
            self.mlp = _MockSPMoEBlock(dim)

        def linear_attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states * 0.5

        forward = patched_decoder_layer_forward

    dim = 16
    layer = _MockSPLayer(dim).eval()
    # SP inputs have first dimension `num_tokens // tp` (`512` when `positions` has `1024`)
    hs_shard = torch.randn(512, dim)
    res_shard = torch.randn(512, dim)
    pos_full = torch.arange(1024)

    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 0)
    ref_hs, ref_res = layer(hs_shard, res_shard, pos_full)

    monkeypatch.setattr(envs, "TPU_QWEN35_MICROBATCH_GROUPS", 2)
    gm_sp, inputs_sp = _capture_dynamic_fx_graph(
        layer, hs_shard, res_shard, pos_full
    )
    rewritten = transforms.apply_microbatch_pass(gm_sp, inputs_sp, size=1024)
    assert rewritten == 1
    assert _count_targets(gm_sp, "fused_moe") == 2
    assert _count_targets(gm_sp, "all_reduce") == 2
    assert _count_targets(gm_sp, "microbatch_barrier_2") == 2

    out_hs, out_res = gm_sp(*inputs_sp)
    torch.testing.assert_close(out_hs, ref_hs, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_res, ref_res, rtol=1e-4, atol=1e-4)


@torch.library.custom_op("tpu_transforms_test::arbitrary_op_alpha", mutates_args=())
def _arbitrary_op_alpha(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 + 1.0


@_arbitrary_op_alpha.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("tpu_transforms_test::arbitrary_op_beta", mutates_args=())
def _arbitrary_op_beta(x: torch.Tensor) -> torch.Tensor:
    return x - 0.5


@_arbitrary_op_beta.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def test_explicit_pipeline_stage_without_op_name_heuristics() -> None:
    """Verify that `transforms.microbatching.pipeline_stage` partitions arbitrary
    operations into Stage 1 and Stage 2 purely from explicit stage scopes, with
    zero reliance on collective or compute op-name heuristics."""

    def _fn(x: torch.Tensor) -> torch.Tensor:
        with transforms.microbatch(min_len=1024, groups=2):
            with transforms.microbatching.pipeline_stage():
                y = torch.ops.tpu_transforms_test.arbitrary_op_alpha(x)
            with transforms.microbatching.pipeline_stage():
                z = torch.ops.tpu_transforms_test.arbitrary_op_beta(y)
        return z

    gm, inputs = _capture_dynamic_fx_graph(_fn, torch.randn(1024, 16))
    rewritten = transforms.apply_microbatch_pass(gm, inputs, size=1024)
    assert rewritten == 1
    assert _count_targets(gm, "microbatch_stage_enter") == 0
    assert _count_targets(gm, "microbatch_stage_exit") == 0

    op_sequence = [
        getattr(n.target, "__name__", str(n.target)).lower()
        for n in gm.graph.nodes
        if n.op == "call_function"
        and any(
            k in getattr(n.target, "__name__", str(n.target)).lower()
            for k in (
                "arbitrary_op_alpha",
                "arbitrary_op_beta",
                "microbatch_barrier_2",
                "cat",
            )
        )
    ]
    assert len(op_sequence) == 7
    assert "arbitrary_op_alpha" in op_sequence[0]
    assert "microbatch_barrier_2" in op_sequence[1]
    assert "arbitrary_op_beta" in op_sequence[2]
    assert "arbitrary_op_alpha" in op_sequence[3]
    assert "microbatch_barrier_2" in op_sequence[4]
    assert "arbitrary_op_beta" in op_sequence[5]
    assert "cat" in op_sequence[6]

    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    x_1024 = torch.randn(1024, 16)
    ref = (x_1024 * 2.0 + 1.0) - 0.5
    call_args = [
        1024
        if isinstance(ph.meta.get("example_value"), torch.SymInt)
        else x_1024
        for ph in phs
    ]
    out = gm(*call_args)
    if isinstance(out, tuple):
        out = out[0]
    torch.testing.assert_close(out, ref)



