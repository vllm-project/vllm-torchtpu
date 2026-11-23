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
"""Public contract and CPU torch.compile unit tests for while_loop.ops."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest
import torch
from torch._dynamo.testing import CompileCounter
from torch.fx.experimental.proxy_tensor import make_fx

import vllm_torchtpu.compilation.while_loop as while_loop_pkg
from vllm_torchtpu.compilation.while_loop import (
    masked_update_step_slice,
    tpu_while_loop,
)
from vllm_torchtpu.compilation.while_loop import (
    ops as wl_ops,
)


@dataclasses.dataclass
class _DecodeState:
    step: torch.Tensor
    active: torch.Tensor
    buffers: dict[str, torch.Tensor]


torch.utils._pytree.register_dataclass(_DecodeState)


class TestPublicWhileLoopModuleHygiene:
    def test_public_exports_and_zero_patch_or_mlir_imports(self) -> None:
        assert set(while_loop_pkg.__all__) == {
            "masked_update_step_slice",
            "tpu_while_loop",
        }
        pkg_dir = Path(wl_ops.__file__).resolve().parent
        forbidden = ("_mlir", "patch", "torch_tpu", "jax")
        for fname in ("__init__.py", "ops.py"):
            tree = ast.parse((pkg_dir / fname).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or "", *(a.name for a in node.names)]
                else:
                    continue
                assert not any(f in n for n in names for f in forbidden)


class TestTpuWhileLoopContracts:
    def test_pytree_carries_zero_trip_strides_and_inference_mode(self) -> None:
        init = _DecodeState(
            step=torch.tensor(0, dtype=torch.int32),
            active=torch.tensor([True, True, False]),
            buffers={"tok": torch.zeros((4, 3), dtype=torch.int32)},
        )

        def body(st: _DecodeState) -> _DecodeState:
            row = torch.where(st.active, st.step + 10, torch.zeros_like(st.step))
            return _DecodeState(
                step=st.step + 1,
                active=st.active & (st.step < 1),
                buffers={
                    "tok": masked_update_step_slice(
                        st.buffers["tok"], row, st.step, dim=0
                    )
                },
            )

        with torch.inference_mode():
            out = tpu_while_loop(lambda st: (st.step < 4) & st.active.any(), body, init)
        assert int(out.step.item()) == 2
        assert out.buffers["tok"][:2].tolist() == [[10, 10, 0], [11, 11, 0]]

        # 0-trip eager loop clones outputs while preserving both dense transposed
        # and non-dense strided views.
        trans = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        gapped = torch.arange(24, dtype=torch.float32).reshape(4, 6)[:, ::2]
        _, out_t, out_g = tpu_while_loop(
            lambda s, _t, _g: s < 0,
            lambda s, t, g: (s + 1, t + 1.0, g + 1.0),
            (torch.tensor(0, dtype=torch.int32), trans, gapped),
        )
        assert out_t.data_ptr() != trans.data_ptr() and out_t.stride() == (1, 6)
        assert out_g.data_ptr() != gapped.data_ptr() and out_g.stride() == (6, 2)
        torch.testing.assert_close(out_t, trans)
        torch.testing.assert_close(out_g, gapped)

    def test_validation_errors_and_generator_kwarg_rejected(self) -> None:
        s0 = torch.tensor(0, dtype=torch.int32)
        x0 = torch.zeros((2, 2), dtype=torch.float32)

        with pytest.raises(TypeError):
            tpu_while_loop(  # type: ignore[call-arg]
                lambda s: s < 1,
                lambda s: (s + 1,),
                (s0,),
                generator=torch.Generator(),
            )
        with pytest.raises(ValueError, match=">= 1 Tensor"):
            tpu_while_loop(lambda _: True, lambda x: x, ())
        with pytest.raises(TypeError, match="torch.Tensor"):
            tpu_while_loop(lambda s, _n: s < 1, lambda s, n: (s + 1, n), (s0, 3))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="reside on"):
            tpu_while_loop(
                lambda s, _m: s < 1,
                lambda s, m: (s + 1, m),
                (s0, torch.zeros((2,), device="meta")),
            )
        with pytest.raises(ValueError, match="0-D torch.bool"):
            tpu_while_loop(lambda s: torch.tensor([True]), lambda s: (s + 1,), (s0,))
        with pytest.raises(ValueError, match="PyTree structure"):
            tpu_while_loop(
                lambda d: d["s"] < 1,
                lambda d: {"s": d["s"] + 1, "extra": d["s"]},
                {"s": s0},
            )
        with pytest.raises(RuntimeError, match="In-place mutation"):
            tpu_while_loop(
                lambda s, _x: s < 1,
                lambda s, x: (s + 1, x.add_(1.0)),
                (s0, x0.clone()),
            )

    def test_default_rng_and_shared_carry_deduplication_under_compile(self) -> None:
        @torch.compile(backend="eager", fullgraph=True)
        def sample_steps(n: torch.Tensor) -> torch.Tensor:
            s0 = torch.tensor(0, dtype=torch.int32)
            draws0 = torch.zeros((3, 4), dtype=torch.float32)
            _, draws = tpu_while_loop(
                lambda s, _d: s < n,
                lambda s, d: (
                    s + 1,
                    masked_update_step_slice(d, torch.rand((4,)), s, dim=0),
                ),
                (s0, draws0),
            )
            return draws

        torch.manual_seed(123)
        draws = sample_steps(torch.tensor(3, dtype=torch.int32))
        assert not torch.allclose(draws[0], draws[1])
        assert not torch.allclose(draws[1], draws[2])

        # Identical Tensor object deduplicates cleanly; divergent updates raise.
        kv = torch.zeros((2, 4), dtype=torch.float32)

        def valid_shared(s: torch.Tensor, k0: torch.Tensor, _k1: torch.Tensor):
            nk = k0 + 1.5
            return s + 1, nk, nk

        _, r0, r1 = torch.compile(backend="eager", fullgraph=True)(
            lambda k: tpu_while_loop(
                lambda s, _a, _b: s < 2,
                valid_shared,
                (torch.tensor(0, dtype=torch.int32), k, k),
            )
        )(kv)
        torch.testing.assert_close(r0, torch.full((2, 4), 3.0))
        torch.testing.assert_close(r1, torch.full((2, 4), 3.0))

        with pytest.raises(ValueError, match="deduplicated"):
            tpu_while_loop(
                lambda s, _a, _b: s < 1,
                lambda s, a, b: (s + 1, a + 1.0, b + 2.0),
                (torch.tensor(0, dtype=torch.int32), kv, kv),
            )

    def test_recompilation_stability_and_noncontiguous_strides(self) -> None:
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def run_loop(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
            s0 = torch.tensor(0, dtype=torch.int32)

            def body(s: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
                nx = x + 1.0
                return (s + 1, nx, nx) if x is y else (s + 1, nx, y + 2.0)

            _, oa, ob = tpu_while_loop(lambda s, _x, _y: s < 2, body, (s0, a, b))
            return oa, ob

        for _ in range(6):
            run_loop(torch.ones((4, 4)), torch.zeros((4, 4)))
        assert counter.frame_count == 1
        for _ in range(6):
            sh = torch.ones((4, 4))
            run_loop(sh, sh)
        assert counter.frame_count == 2

        # Non-contiguous transposed strides survive inside body_fn and epilogue.
        @torch.compile(backend="aot_eager", fullgraph=True)
        def transposed_loop(x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            s0 = torch.tensor(0, dtype=torch.int32)

            def body(s: torch.Tensor, h: torch.Tensor):
                flat = h.t().view(-1)
                nxt = (h + flat.mean() * 0.1).t().contiguous().t()
                return s + 1, nxt

            _, out_h = tpu_while_loop(lambda s, _h: s < 3, body, (s0, x_t))
            return out_h, out_h.t().view(-1)

        in_t = torch.arange(32, dtype=torch.float32).reshape(8, 4).t()
        got_h, got_flat = transposed_loop(in_t)
        assert got_h.stride() == (1, 4) and got_flat.shape == (32,)

    @pytest.mark.parametrize("backend", ["eager", "aot_eager"])
    def test_overlapping_and_same_storage_views_raise_value_error(
        self, backend: str
    ) -> None:
        @torch.compile(backend=backend, fullgraph=True)
        def run_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            s0 = torch.tensor(0, dtype=torch.int32)
            _, oa, ob = tpu_while_loop(
                lambda s, _x, _y: s < 2,
                lambda s, x, y: (s + 1, x + 1.0, y + 1.0),
                (s0, a, b),
            )
            return oa + ob

        pool = torch.zeros((8, 4), dtype=torch.float32)
        for va, vb in ((pool[0:4], pool[2:6]), (pool[0:4], pool[4:8])):
            with pytest.raises(ValueError, match="storage"):
                tpu_while_loop(
                    lambda s, _x, _y: s < 1,
                    lambda s, x, y: (s + 1, x + 1.0, y + 1.0),
                    (torch.tensor(0, dtype=torch.int32), va, vb),
                )
            with pytest.raises(ValueError, match="storage"):
                run_pair(va, vb)


class TestMaskedUpdateStepSlice:
    @pytest.mark.parametrize(
        ("buf_dtype", "upd_dtype"),
        [
            (torch.int32, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.int32, torch.int64),
        ],
    )
    def test_rejects_mismatched_dtype_and_invalid_inputs(
        self, buf_dtype: torch.dtype, upd_dtype: torch.dtype
    ) -> None:
        idx = torch.tensor(1, dtype=torch.int32)
        with pytest.raises(ValueError, match="dtype"):
            masked_update_step_slice(
                torch.zeros((4, 2), dtype=buf_dtype),
                torch.ones((2,), dtype=upd_dtype),
                idx,
            )

        base = torch.zeros((2, 5), dtype=torch.int32)
        upd = torch.ones((2, 2), dtype=torch.int32)
        for bad_call, pat in (
            (
                lambda: masked_update_step_slice(
                    base, torch.ones((2, 2), dtype=torch.int32, device="meta"), idx, 1
                ),
                "device",
            ),
            (
                lambda: masked_update_step_slice(torch.tensor(0), torch.tensor(1), idx),
                "ndim >= 1",
            ),
            (
                lambda: masked_update_step_slice(base, upd, torch.tensor([1]), 1),
                "0-D",
            ),
            (
                lambda: masked_update_step_slice(base, upd, torch.tensor(True), 1),
                "integer",
            ),
            (
                lambda: masked_update_step_slice(base, upd, idx, 3),
                "out of range",
            ),
            (
                lambda: masked_update_step_slice(
                    base, torch.ones((2, 2, 1), dtype=torch.int32), idx, 1
                ),
                "rank",
            ),
            (
                lambda: masked_update_step_slice(
                    base, torch.ones((3, 2), dtype=torch.int32), idx, 1
                ),
                "incompatible",
            ),
            (
                lambda: masked_update_step_slice(
                    base, torch.ones((2, 6), dtype=torch.int32), idx, 1
                ),
                "exceeds",
            ),
        ):
            with pytest.raises(ValueError, match=pat):
                bad_call()

    def test_clamping_64bit_unsigned_dtypes_and_pure_int32_graph(self) -> None:
        buf = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32)
        upd = torch.tensor([99, 98], dtype=torch.int32)
        for raw_idx, dtype, expected_row in (
            (-3, torch.int32, 0),
            (9, torch.int32, 3),
            (-(1 << 33), torch.int64, 0),
            (1 << 32, torch.int64, 3),
            (2, torch.uint8, 2),
            ((1 << 31) + 9, torch.uint32, 3),
            ((1 << 63) + 5, torch.uint64, 3),
        ):
            got = masked_update_step_slice(
                buf, upd, torch.tensor(raw_idx, dtype=dtype), dim=0
            )
            exp = buf.clone()
            exp[expected_row] = upd
            assert torch.equal(got, exp)

        # Multi-row clamped write, negative dim, 0-length slice, and fp8/bool.
        buf5 = torch.arange(10, dtype=torch.int32).reshape(5, 2)
        upd3 = torch.tensor([[100, 101], [200, 201], [300, 301]], dtype=torch.int32)
        assert torch.equal(
            masked_update_step_slice(buf5, upd3, torch.tensor(10, dtype=torch.int32)),
            torch.tensor(
                [[0, 1], [2, 3], [100, 101], [200, 201], [300, 301]], dtype=torch.int32
            ),
        )
        z_out = masked_update_step_slice(
            buf5,
            torch.zeros((0, 2), dtype=torch.int32),
            torch.tensor(1, dtype=torch.int32),
        )
        assert torch.equal(z_out, buf5) and z_out.data_ptr() != buf5.data_ptr()

        fp8_buf = torch.zeros((3, 2), dtype=torch.float8_e4m3fn)
        fp8_upd = torch.tensor([1.5, -2.0]).to(torch.float8_e4m3fn)
        fp8_out = masked_update_step_slice(
            fp8_buf, fp8_upd, torch.tensor(1, dtype=torch.int32), dim=0
        )
        torch.testing.assert_close(fp8_out[1].float(), fp8_upd.float())

        # Pure int32 indexing emits zero int64 tensors in FX graph.
        for u in (torch.ones((4,)), torch.ones((3, 4))):
            gm = make_fx(lambda b, s, i: masked_update_step_slice(b, s, i, dim=0))(
                torch.zeros((6, 4)), u, torch.tensor(2, dtype=torch.int32)
            )
            for node in gm.graph.nodes:
                val = node.meta.get("val")
                if isinstance(val, torch.Tensor):
                    assert val.dtype != torch.int64
