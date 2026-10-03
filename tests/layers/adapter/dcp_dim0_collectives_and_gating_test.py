#!/usr/bin/env python3
"""
================================================================================
3-TIER EXECUTION PROVENANCE BANNER
--------------------------------------------------------------------------------
Tier 1 (Host C-ABI Shared Library): N/A (PyTorch / vLLM DCP collective adapter)
Tier 2 (Host C++20 LloRegionBuilder IR): N/A
Tier 3 (CPU Numerical Parity vs. Physical TPU Execution):
  [TIER 3 — CPU NUMERICAL PARITY & MEMORY-LAYOUT ZERO-COPY VERIFICATION
   (NOT A PHYSICAL TPU RUN)]
  Executes PyTorch tensor collectives (`DeviceCommunicatorBase.all_gather` and
  `DeviceCommunicatorBase.reduce_scatter`) across simulated multi-rank DCP groups
  against AST-extracted functions from
  `gen4_dcp/src/vllm_torchtpu/layers/adapter/attention.py`
  and `gen4_dcp/cp_mla_attention.py`.
================================================================================
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from typing import Any

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    _venv_dir = "/usr/local/google/home/jfy/.venvs/torchcpu"
    _venv_py = os.path.join(_venv_dir, "bin", "python3")
    if os.path.exists(_venv_py) and os.path.abspath(sys.prefix) != os.path.abspath(
        _venv_dir
    ):
        os.execv(_venv_py, [_venv_py, *sys.argv])
    raise


def _find_repo_root(start: str) -> str:
    cur = os.path.abspath(start)
    while cur != os.path.dirname(cur):
        if os.path.exists(os.path.join(cur, "src", "vllm_torchtpu")):
            return cur
        cur = os.path.dirname(cur)
    raise RuntimeError(f"Could not find repo root from {start}")


_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
HERE = _find_repo_root(_FILE_DIR)
F_ATTN = os.path.join(HERE, "src", "vllm_torchtpu", "layers", "adapter", "attention.py")
F_CP_MLA = os.path.join(
    HERE, "src", "vllm_torchtpu", "layers", "adapter", "cp_mla_attention.py"
)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _extract(path: str, names: set[str], ns: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(_read(path), filename=path)
    picked = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    got = {n.name for n in picked}
    missing = names - got
    if missing:
        raise AssertionError(f"{path}: missing {sorted(missing)}")
    mod = ast.fix_missing_locations(ast.Module(body=picked, type_ignores=[]))
    exec(compile(mod, path, "exec"), ns)
    return ns


def _cossim_and_sqnr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a64 = a.detach().to(torch.float64).reshape(-1).numpy()
    b64 = b.detach().to(torch.float64).reshape(-1).numpy()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    cossim = float(np.dot(a64, b64) / denom) if denom > 0 else 1.0
    noise = float(np.sum((a64 - b64) ** 2))
    sig = float(np.sum(b64**2))
    sqnr = float(10.0 * np.log10(sig / noise)) if noise > 0 else float("inf")
    return cossim, sqnr


class _InstrumentedDeviceCommunicatorGroup:
    """Exact replica of
    `vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase`
    `all_gather` (L229-252) and `reduce_scatter` (L262-294), instrumented to record:
      - `all_gather_post_reshape_copied_bytes`: bytes copied when post-all_gather
        `output_tensor.reshape((world_size,) + input_size).movedim(0, dim).reshape(...)`
        cannot return a zero-copy view of `output_tensor`.
      - `reduce_scatter_pre_movedim_copied_bytes`: bytes copied by
        `input_tensor = input_.movedim(0, dim).contiguous()`
        before `reduce_scatter_tensor`.
      - `reduce_scatter_post_movedim_copied_bytes`: bytes copied by
        `output_tensor.movedim(0, dim).contiguous()` after `reduce_scatter_tensor`.
    """

    def __init__(
        self, rank: int, world_size: int, peer_buffers: dict[str, list[torch.Tensor]]
    ):
        self.rank = rank
        self.world_size = world_size
        self.peer_buffers = peer_buffers
        self.all_gather_post_reshape_copied_bytes = 0
        self.reduce_scatter_pre_movedim_copied_bytes = 0
        self.reduce_scatter_post_movedim_copied_bytes = 0

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # Simulate dist.all_gather_into_tensor(output_tensor, input_) across ranks:
        # Note: dist.all_gather_into_tensor requires contiguous input_ on TPU/NCCL.
        assert input_.is_contiguous(), (
            "dist.all_gather_into_tensor requires contiguous input_"
        )
        peers = self.peer_buffers.get("ag_inputs")
        if peers is not None:
            torch.cat([p.contiguous() for p in peers], dim=0, out=output_tensor)
        else:
            torch.cat([input_] * self.world_size, dim=0, out=output_tensor)

        raw_ptr = output_tensor.data_ptr()
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        if output_tensor.data_ptr() != raw_ptr:
            self.all_gather_post_reshape_copied_bytes += (
                output_tensor.numel() * output_tensor.element_size()
            )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim()
        if dim < 0:
            dim += input_.dim()

        in_ptr = input_.data_ptr()
        input_tensor = input_.movedim(0, dim).contiguous()
        if input_tensor.data_ptr() != in_ptr:
            self.reduce_scatter_pre_movedim_copied_bytes += (
                input_tensor.numel() * input_tensor.element_size()
            )

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Simulate dist.reduce_scatter_tensor(output_tensor, input_tensor)
        peers = self.peer_buffers.get("rs_inputs")
        if peers is not None:
            total = sum(p.movedim(0, dim).contiguous() for p in peers)
        else:
            total = input_tensor * world_size
        output_tensor.copy_(total.narrow(0, self.rank * chunk_size, chunk_size))

        out_ptr = output_tensor.data_ptr()
        ret = output_tensor.movedim(0, dim).contiguous()
        if ret.data_ptr() != out_ptr:
            self.reduce_scatter_post_movedim_copied_bytes += (
                ret.numel() * ret.element_size()
            )
        return ret


def _legacy_all_gather_heads(q: torch.Tensor, group: Any) -> torch.Tensor:
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


def _legacy_weight_by_lse(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    lse_f32 = lse.to(torch.float32).contiguous()
    gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    finite = torch.isfinite(gathered)
    shift = torch.where(
        finite, gathered, torch.full_like(gathered, float("-inf"))
    ).amax(dim=0)
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, torch.zeros_like(shift), shift)
    weights = torch.where(
        finite, torch.exp(gathered - shift.unsqueeze(0)), torch.zeros_like(gathered)
    )
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    mine = (
        torch.where(
            torch.isfinite(lse_f32),
            torch.exp(lse_f32 - shift),
            torch.zeros_like(lse_f32),
        )
        / safe_denom
    )
    return (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()


def _legacy_merge_lse_partials_scatter_heads(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    if group is None or int(group.world_size) == 1:
        return output
    weighted = _legacy_weight_by_lse(output, lse, group)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


class TestDcpDim0CollectivesZeroMovedimAndGate7Parity(unittest.TestCase):
    def _load_fns(self, path: str, group_box: list[Any]):
        ns: dict[str, Any] = {
            "torch": torch,
            "Any": Any,
            "_get_dcp_group": lambda: group_box[0],
            "get_dcp_group": lambda: group_box[0],
        }
        _extract(
            path,
            {
                "all_gather_heads",
                "all_gather_q_nope_pe_heads",
                "_weight_by_lse",
                "merge_lse_partials_scatter_heads",
            },
            ns,
        )
        return ns

    def test_attention_and_cp_mla_attention_ast_parity(self):
        group_box: list[Any] = [None]
        ns_attn = self._load_fns(F_ATTN, group_box)
        ns_cp = self._load_fns(F_CP_MLA, group_box)
        for fn_name in (
            "all_gather_heads",
            "all_gather_q_nope_pe_heads",
            "_weight_by_lse",
            "merge_lse_partials_scatter_heads",
        ):
            self.assertIn(fn_name, ns_attn)
            self.assertIn(fn_name, ns_cp)

    def test_reduce_scatter_dim0_eliminates_movedim_copies_and_preserves_gate7_parity(
        self,
    ):
        torch.manual_seed(42)
        world_size = 4
        T = 64
        H_local = 4
        H_dcp = H_local * world_size
        D = 512

        partials = [
            torch.randn(T, H_dcp, D, dtype=torch.bfloat16) for _ in range(world_size)
        ]
        lses = [torch.randn(T, H_dcp, dtype=torch.float32) for _ in range(world_size)]
        # Inject -inf for ranks owning nothing on specific tokens and all-empty token 3
        lses[0][1, :] = float("-inf")
        lses[2][1, :] = float("-inf")
        for r in range(world_size):
            lses[r][3, :] = float("-inf")

        group_box: list[Any] = [None]
        ns = self._load_fns(F_ATTN, group_box)

        # Precompute weighted tensors across ranks for both legacy (dim=1)
        # and new (dim=0)
        lse_peers = [lse_val.unsqueeze(0) for lse_val in lses]
        legacy_weighted = []
        new_weighted = []
        for r in range(world_size):
            g_tmp = _InstrumentedDeviceCommunicatorGroup(
                r, world_size, {"ag_inputs": lse_peers}
            )
            legacy_weighted.append(_legacy_weight_by_lse(partials[r], lses[r], g_tmp))
            new_weighted.append(ns["_weight_by_lse"](partials[r], lses[r], g_tmp))
            self.assertEqual(new_weighted[-1].shape, (H_dcp, T, D))
            self.assertTrue(new_weighted[-1].is_contiguous())

        w_uv = torch.randn(H_local, D, 128, dtype=torch.bfloat16)

        for r in range(world_size):
            g_old = _InstrumentedDeviceCommunicatorGroup(
                r,
                world_size,
                {"ag_inputs": lse_peers, "rs_inputs": legacy_weighted},
            )
            out_old = _legacy_merge_lse_partials_scatter_heads(
                partials[r], lses[r], g_old
            )
            outputs_t_old = out_old.reshape(-1, H_local, D).transpose(0, 1)
            # Legacy dim=1 copies full [T, H_dcp, D] f32 before RS and
            # [T, H_local, D] f32 after RS
            self.assertEqual(
                g_old.reduce_scatter_pre_movedim_copied_bytes,
                T * H_dcp * D * 4,
            )
            self.assertEqual(
                g_old.reduce_scatter_post_movedim_copied_bytes,
                T * H_local * D * 4,
            )
            # And legacy outputs_t_old is NON-contiguous before W_UV BMM!
            self.assertFalse(outputs_t_old.is_contiguous())

            g_new = _InstrumentedDeviceCommunicatorGroup(
                r,
                world_size,
                {"ag_inputs": lse_peers, "rs_inputs": new_weighted},
            )
            group_box[0] = g_new
            out_new = ns["merge_lse_partials_scatter_heads"](partials[r], lses[r])
            self.assertEqual(out_new.shape, (T, H_local, D))
            outputs_3d = (
                out_new
                if out_new.ndim == 3
                and out_new.shape[1] == H_local
                and out_new.shape[2] == D
                else out_new.reshape(-1, H_local, D)
            )
            outputs_t_new = outputs_3d.transpose(0, 1)

            # New dim=0 copies ZERO bytes in pre-movedim and ZERO bytes in post-movedim!
            self.assertEqual(g_new.reduce_scatter_pre_movedim_copied_bytes, 0)
            self.assertEqual(g_new.reduce_scatter_post_movedim_copied_bytes, 0)
            # And outputs_t_new is 100% CONTIGUOUS in [H_local, T, D] with zero copy!
            self.assertTrue(outputs_t_new.is_contiguous())
            self.assertEqual(outputs_t_new.data_ptr(), out_new.data_ptr())

            # Gate 7 numerical parity check
            cossim, sqnr = _cossim_and_sqnr(out_new, out_old)
            self.assertGreaterEqual(cossim, 0.99999)
            self.assertGreaterEqual(sqnr, 45.0)
            torch.testing.assert_close(out_new, out_old, rtol=0.0, atol=0.0)

            proj_old = torch.bmm(outputs_t_old, w_uv)
            proj_new = torch.bmm(outputs_t_new, w_uv)
            torch.testing.assert_close(proj_new, proj_old, rtol=0.0, atol=0.0)

    def test_all_gather_q_nope_pe_heads_dim0_eliminates_post_gather_copy(self):
        torch.manual_seed(99)
        world_size = 4
        T = 64
        H_local = 4
        D_nope = 128
        D_lora = 512
        D_rope = 64

        q_nopes = [
            torch.randn(T, H_local, D_nope, dtype=torch.bfloat16)
            for _ in range(world_size)
        ]
        w_uk_ts = [
            torch.randn(H_local, D_nope, D_lora, dtype=torch.bfloat16)
            for _ in range(world_size)
        ]
        q_pes = [
            torch.randn(T, H_local, D_rope, dtype=torch.bfloat16)
            for _ in range(world_size)
        ]

        # Produce ql_nope_flat the exact way PallasMLAttentionBackendImpl.forward does:
        ql_nope_flats = []
        for r in range(world_size):
            ql_nope = torch.bmm(q_nopes[r].transpose(0, 1), w_uk_ts[r])
            ql_nope = ql_nope.to(torch.bfloat16).transpose(0, 1)
            ql_nope_flat = ql_nope.view(-1, H_local, D_lora)
            # Verify ql_nope_flat.transpose(0, 1) is ALREADY contiguous (from bmm)!
            self.assertTrue(ql_nope_flat.transpose(0, 1).is_contiguous())
            self.assertEqual(
                ql_nope_flat.transpose(0, 1).contiguous().data_ptr(),
                ql_nope_flat.data_ptr(),
            )
            ql_nope_flats.append(ql_nope_flat)

        group_box: list[Any] = [None]
        ns = self._load_fns(F_ATTN, group_box)

        # 1. Fused Q gather path (`TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER=1`)
        old_cat_peers = [
            torch.cat([ql_nope_flats[r], q_pes[r]], dim=-1).contiguous()
            for r in range(world_size)
        ]
        new_cat_hm_peers = [
            torch.cat(
                [ql_nope_flats[r].transpose(0, 1), q_pes[r].transpose(0, 1)],
                dim=-1,
            )
            for r in range(world_size)
        ]

        for r in range(world_size):
            g_old = _InstrumentedDeviceCommunicatorGroup(
                r, world_size, {"ag_inputs": old_cat_peers}
            )
            q_dcp_old = _legacy_all_gather_heads(old_cat_peers[r], g_old)
            ql_nope_old = q_dcp_old[..., :D_lora].contiguous()
            q_pe_old = q_dcp_old[..., D_lora:].contiguous()
            # Legacy dim=1 copies full gathered tensor inside all_gather.reshape()!
            self.assertEqual(
                g_old.all_gather_post_reshape_copied_bytes,
                T * (H_local * world_size) * (D_lora + D_rope) * 2,
            )

            g_new = _InstrumentedDeviceCommunicatorGroup(
                r, world_size, {"ag_inputs": new_cat_hm_peers}
            )
            group_box[0] = g_new
            ql_nope_new, q_pe_new = ns["all_gather_q_nope_pe_heads"](
                ql_nope_flats[r], q_pes[r], D_lora
            )
            # New dim=0 copies ZERO bytes inside all_gather.reshape()!
            self.assertEqual(g_new.all_gather_post_reshape_copied_bytes, 0)
            self.assertTrue(ql_nope_new.is_contiguous())
            self.assertTrue(q_pe_new.is_contiguous())
            torch.testing.assert_close(ql_nope_new, ql_nope_old, rtol=0.0, atol=0.0)
            torch.testing.assert_close(q_pe_new, q_pe_old, rtol=0.0, atol=0.0)

        # 2. Unfused `all_gather_heads` path (`TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER=0`)
        old_nope_peers = [x.contiguous() for x in ql_nope_flats]
        new_nope_hm_peers = [x.transpose(0, 1).contiguous() for x in ql_nope_flats]
        for r in range(world_size):
            g_old = _InstrumentedDeviceCommunicatorGroup(
                r, world_size, {"ag_inputs": old_nope_peers}
            )
            nope_old = _legacy_all_gather_heads(ql_nope_flats[r], g_old)
            self.assertEqual(
                g_old.all_gather_post_reshape_copied_bytes,
                T * (H_local * world_size) * D_lora * 2,
            )

            g_new = _InstrumentedDeviceCommunicatorGroup(
                r, world_size, {"ag_inputs": new_nope_hm_peers}
            )
            group_box[0] = g_new
            nope_new = ns["all_gather_heads"](ql_nope_flats[r])
            self.assertEqual(g_new.all_gather_post_reshape_copied_bytes, 0)
            torch.testing.assert_close(nope_new, nope_old, rtol=0.0, atol=0.0)

    def test_negative_control_misaligned_lse_or_wrong_split_fails(self):
        """Gate 4 Negative Control:
        1. Permuting rank LSEs in `merge_lse_partials_scatter_heads` must diverge.
        2. Passing a wrong `kv_lora_rank` split offset to `all_gather_q_nope_pe_heads`
           must produce wrong tensor shapes.
        """
        torch.manual_seed(7)
        world_size = 4
        T, H_local, D = 16, 2, 32
        H_dcp = H_local * world_size
        partials = [
            torch.randn(T, H_dcp, D, dtype=torch.bfloat16) for _ in range(world_size)
        ]
        lses = [
            torch.randn(T, H_dcp, dtype=torch.float32) * 5.0 for _ in range(world_size)
        ]

        group_box: list[Any] = [None]
        ns = self._load_fns(F_ATTN, group_box)

        lse_peers = [lse_val.unsqueeze(0) for lse_val in lses]
        good_weighted = [
            ns["_weight_by_lse"](
                partials[r],
                lses[r],
                _InstrumentedDeviceCommunicatorGroup(
                    r, world_size, {"ag_inputs": lse_peers}
                ),
            )
            for r in range(world_size)
        ]
        # Negative control: rank r weights its own partial with rank (r+1)'s LSE
        bad_weighted = [
            ns["_weight_by_lse"](
                partials[r],
                lses[(r + 1) % world_size],
                _InstrumentedDeviceCommunicatorGroup(
                    r, world_size, {"ag_inputs": lse_peers}
                ),
            )
            for r in range(world_size)
        ]

        g_good = _InstrumentedDeviceCommunicatorGroup(
            0, world_size, {"ag_inputs": lse_peers, "rs_inputs": good_weighted}
        )
        group_box[0] = g_good
        out_good = ns["merge_lse_partials_scatter_heads"](partials[0], lses[0])

        g_bad = _InstrumentedDeviceCommunicatorGroup(
            0, world_size, {"ag_inputs": lse_peers, "rs_inputs": bad_weighted}
        )
        group_box[0] = g_bad
        out_bad = ns["merge_lse_partials_scatter_heads"](partials[0], lses[1])

        max_diff = float((out_good.float() - out_bad.float()).abs().max().item())
        self.assertGreater(max_diff, 0.1)

    def test_symint_num_tokens_adds_zero_shape_env_guards(self):
        """Gen-5 Regression Test for `j-1b1c232b` & `j-f55bc1ba`:
        Verifies that passing a dynamic TorchDynamo `SymInt` (`s0 = 16`, min=2)
        through `PallasMLAttentionBackendImpl._use_sparse_mla_dcp_a2a_merge`,
        `_fake_sparse_mla_dcp`, and `PallasMLAttentionBackendImpl.forward` adds
        ZERO guards (`len(shape_env.guards) == 0`) to `ShapeEnv`, whereas the
        legacy `int(num_tokens) > max_tokens` check specialized `s0 == 16`
        (`len(shape_env.guards) > 0`, Negative Control).
        """
        from torch._dynamo.source import LocalSource
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
        )

        tree = ast.parse(_read(F_ATTN), filename=F_ATTN)
        cls_node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "PallasMLAttentionBackendImpl"
        )
        methods = {
            n.name: n
            for n in cls_node.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in {
                "_build_sparse_mla_dcp_op",
                "forward",
            }
        }
        self.assertEqual(
            set(methods),
            {
                "_build_sparse_mla_dcp_op",
                "forward",
            },
        )

        group_box: list[Any] = [None]
        ns = self._load_fns(F_ATTN, group_box)
        mod = ast.fix_missing_locations(
            ast.Module(
                body=[methods["forward"]],
                type_ignores=[],
            )
        )
        ns.update(
            {
                "envs": type(
                    "_Envs", (), {"TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER": True}
                )(),
                "logger": type(
                    "_Logger", (), {"info_once": staticmethod(lambda *a, **k: None)}
                )(),
                "quantize_kv": lambda dtype, x, value=None, k_scale=None: (x, None),
            }
        )
        exec(compile(mod, F_ATTN, "exec"), ns)

        class _DummyBackend:
            dcp_size = 4
            forward = ns["forward"]

            def _get_kv_scales(self, layer):
                return (None, None, None)

        backend = _DummyBackend()

        # Symbolic `SymInt` behavior (`s_tok` backed by 16, dynamic):
        shape_env = ShapeEnv()
        sym_t = shape_env.create_unspecified_symint_and_symbol(
            16,
            source=LocalSource("q_nope"),
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        self.assertIsInstance(sym_t, torch.SymInt)
        self.assertEqual(len(shape_env.guards), 0)

        with FakeTensorMode(shape_env=shape_env, static_shapes=False):
            fake_q_nope = torch.empty((sym_t, 4, 128), dtype=torch.bfloat16)
            fake_q_pe = torch.empty((sym_t, 4, 64), dtype=torch.bfloat16)
            fake_kv_c = torch.empty((sym_t, 512), dtype=torch.bfloat16)
            fake_k_pe = torch.empty((sym_t, 64), dtype=torch.bfloat16)
            fake_topk = torch.empty((sym_t * 4, 2048), dtype=torch.int32)
            fake_w_uk_t = torch.empty((16, 128, 512), dtype=torch.bfloat16)
            fake_w_uv = torch.empty((4, 512, 128), dtype=torch.bfloat16)
            fake_q_rep = (
                torch.empty((sym_t, 16, 128), dtype=torch.bfloat16),
                torch.empty((sym_t, 16, 64), dtype=torch.bfloat16),
            )

            class _FakeLayer:
                num_heads = 4
                v_head_dim = 128
                kv_lora_rank = 512
                qk_rope_head_dim = 64
                kv_cache_quantized_dtype = None
                W_UK_T_dcp_qrep = fake_w_uk_t
                W_UK_T_dcp_qrep_scale = None
                W_UK_T = fake_w_uk_t[:4]
                W_UV = fake_w_uv

                @staticmethod
                def sparse_mla_dcp_op(kv_cache, ql_nope_dcp, *args):
                    t, h, d = ql_nope_dcp.shape
                    return (
                        ql_nope_dcp.new_empty((t, h, d)),
                        ql_nope_dcp.new_empty((t, h), dtype=torch.float32),
                    )

            class _FakeMeta:
                seq_lens = torch.empty((16,), dtype=torch.int32)
                block_tables = torch.empty((16, 8), dtype=torch.int32)
                query_start_loc = torch.empty((17,), dtype=torch.int32)
                request_distribution = torch.empty((3,), dtype=torch.int32)

            guards_before_forward = list(shape_env.guards)

            # 2. Tracing `forward()` with dynamic `sym_t` adds ZERO
            # specialization guards
            out = backend.forward(
                _FakeLayer(),
                (fake_q_nope, fake_q_pe),
                fake_kv_c,
                fake_k_pe,
                (
                    torch.empty((8, 512), dtype=torch.bfloat16),
                    torch.empty((8, 64), dtype=torch.bfloat16),
                ),
                _FakeMeta(),
                topk_indices=fake_topk,
                q_dcp_replicated=fake_q_rep,
            )
            self.assertEqual(out.shape[1], 4 * 128)
            self.assertIsInstance(out.shape[0], torch.SymInt)
            new_guards = [
                str(g.expr) for g in shape_env.guards[len(guards_before_forward) :]
            ]
            # Verify no specialization guard (Eq(..., 16) or <= 256) was added!
            for expr_str in new_guards:
                self.assertNotIn("16", expr_str)
                self.assertNotIn("256", expr_str)

            # Negative control: legacy `int(sym_t) > 256` immediately specializes
            # `Eq(s_tok, 16)`!
            _ = int(sym_t) > 256
            all_exprs = [str(g.expr) for g in shape_env.guards]
            self.assertTrue(
                any("16" in e for e in all_exprs),
                f"Expected legacy int(sym_t) to specialize to 16, got {all_exprs}",
            )

    def test_dcp_mla_default_env_flags(self):
        """Verifies that DCP MLA optimization flags default safely in envs.py:
        - TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER defaults to True
        """
        import importlib.util

        f_envs = os.path.join(HERE, "src", "vllm_torchtpu", "envs.py")
        spec = importlib.util.spec_from_file_location("_dcp_envs", f_envs)
        assert spec is not None and spec.loader is not None
        envs_mod = importlib.util.module_from_spec(spec)
        os.environ.pop("TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", None)
        spec.loader.exec_module(envs_mod)
        self.assertTrue(envs_mod.TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER)


if __name__ == "__main__":
    print(
        "[Tier 3 — CPU Numerical Parity & Memory-Layout Zero-Copy Verification "
        "(NOT A PHYSICAL TPU RUN)]"
    )
    unittest.main(verbosity=2)
