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
"""Real-TPU compilation, KV donation, and multichip tests for while_loop."""

import argparse
import dataclasses
import functools
import gc
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.distributed.tpu_test_utils import (
    prepare_tpu_environment,
    run_in_isolated_process,
)

_GOOGLE_PCI_VENDOR = "0x1ae0"
_MULTICHIP_LAUNCH_TIMEOUT_S = 600


def _has_tpu_hardware() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    pci_root = Path("/sys/bus/pci/devices")
    if pci_root.is_dir():
        for dev in pci_root.iterdir():
            try:
                if (dev / "vendor").read_text().strip() != _GOOGLE_PCI_VENDOR:
                    continue
                group = (dev / "iommu_group").readlink().name
                if (Path("/dev/vfio") / group).exists():
                    return True
            except OSError:
                continue
    return any(Path("/dev").glob("accel*"))


pytestmark = [
    pytest.mark.tpu_test,
    pytest.mark.skipif(
        not _has_tpu_hardware(),
        reason="TPU hardware and torch_tpu are required",
    ),
]


def _activate_patch() -> None:
    import torch_tpu  # noqa: F401

    from vllm_torchtpu import patch_registry

    patch_registry.apply("worker_init")
    torch.set_float32_matmul_precision("highest")


def _release_tpu_memory() -> None:
    gc.collect()
    torch.accelerator.synchronize()


@dataclasses.dataclass
class _DecodeCarry:
    step: torch.Tensor
    positions_i64: torch.Tensor
    active_mask: torch.Tensor
    buffers: dict[str, torch.Tensor]


torch.utils._pytree.register_dataclass(_DecodeCarry)


def _check_zero_trip_eos_rng_and_strides() -> None:
    from vllm_torchtpu.compilation.while_loop import (
        masked_update_step_slice,
        tpu_while_loop,
    )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def burst_with_eos_and_rng(
        x_t: torch.Tensor, w_t: torch.Tensor, max_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = x_t.device
        s0 = torch.zeros((), dtype=torch.int32, device=dev)
        active0 = torch.tensor([True, True, False, True], device=dev)
        noise_log0 = torch.zeros((5, 4), dtype=torch.float32, device=dev)

        def cond_fn(
            s: torch.Tensor, _h: torch.Tensor, act: torch.Tensor, _nl: torch.Tensor
        ):
            return (s < max_steps) & act.any()

        def body_fn(
            s: torch.Tensor, h: torch.Tensor, act: torch.Tensor, nl: torch.Tensor
        ):
            flat_sum = h.t().view(-1).mean() + w_t.t().view(-1).mean()
            nxt_h = torch.tanh(h @ w_t + flat_sum * 0.01).t().contiguous().t()
            draws = torch.rand((4,), dtype=torch.float32, device=dev)
            nxt_nl = masked_update_step_slice(nl, draws, s, dim=0)
            return s + 1, nxt_h, act & (s < 2), nxt_nl

        out_s, out_h, _, out_nl = tpu_while_loop(
            cond_fn, body_fn, (s0, x_t, active0, noise_log0)
        )
        return out_s, out_h, out_h.t().view(-1), out_nl

    x_in = torch.arange(64, dtype=torch.float32, device="tpu").reshape(16, 4).t() * 0.02
    w_in = (torch.eye(16, dtype=torch.float32, device="tpu") * 0.5).t()

    # 0-trip preserves (1, 4) stride; multi-step exits early at step 3 with RNG.
    z_s, z_h, z_flat, _ = burst_with_eos_and_rng(
        x_in, w_in, torch.tensor(0, dtype=torch.int32, device="tpu")
    )
    assert (
        int(z_s.cpu().item()) == 0 and z_h.stride() == (1, 4) and z_flat.shape == (64,)
    )

    tpu_mod: Any = torch.tpu
    tpu_mod.manual_seed(42)
    m_s, m_h, _, m_nl = burst_with_eos_and_rng(
        x_in, w_in, torch.tensor(5, dtype=torch.int32, device="tpu")
    )
    assert int(m_s.cpu().item()) == 3 and m_h.stride() == (1, 4)
    nl_cpu = m_nl.cpu()
    assert not torch.allclose(nl_cpu[0], nl_cpu[1])
    assert not torch.allclose(nl_cpu[1], nl_cpu[2])
    assert torch.equal(nl_cpu[3:], torch.zeros((2, 4)))


def _check_pytree_int64_nested_and_chained_loops() -> None:
    from vllm_torchtpu.compilation.while_loop import (
        masked_update_step_slice,
        tpu_while_loop,
    )

    def pytree_prog(st: _DecodeCarry, lim: torch.Tensor) -> _DecodeCarry:
        def body(cur: _DecodeCarry) -> _DecodeCarry:
            npos = torch.where(
                cur.active_mask, cur.positions_i64 + 1, cur.positions_i64
            )
            w = masked_update_step_slice(
                cur.buffers["win"],
                torch.full((2, 4), 9.0, dtype=torch.float32, device=cur.step.device),
                cur.step + 50,
                dim=0,
            )
            return _DecodeCarry(
                step=cur.step + 1,
                positions_i64=npos,
                active_mask=cur.active_mask & (cur.step < 2),
                buffers={"win": w},
            )

        return tpu_while_loop(lambda c: (c.step < lim) & c.active_mask.any(), body, st)

    def make_st(dev: str) -> _DecodeCarry:
        return _DecodeCarry(
            step=torch.zeros((), dtype=torch.int64, device=dev),
            positions_i64=((1 << 36) + torch.arange(4, dtype=torch.int64)).to(dev),
            active_mask=torch.tensor([True, False, True, True], device=dev),
            buffers={"win": torch.zeros((4, 4), dtype=torch.float32, device=dev)},
        )

    ref = pytree_prog(make_st("cpu"), torch.tensor(5, dtype=torch.int64))
    got = torch.compile(pytree_prog, backend="tpu", fullgraph=True, dynamic=False)(
        make_st("tpu"), torch.tensor(5, dtype=torch.int64, device="tpu")
    )
    assert int(got.step.cpu().item()) == int(ref.step.item()) == 3
    assert torch.equal(got.positions_i64.cpu(), ref.positions_i64)
    assert torch.equal(got.buffers["win"].cpu(), ref.buffers["win"])

    # Nested + chained + topk multi-output loops match CPU reference.
    def nested_and_topk(
        x: torch.Tensor, lim: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = x.device
        z = torch.zeros((), dtype=torch.int32, device=dev)
        vals, idxs = torch.topk(x, k=2, dim=-1)
        _, v_chained, idx_out = tpu_while_loop(
            lambda s, _v, _i: s < 2,
            lambda s, v, idx: (s + 1, v + 0.5, idx + 1),
            (z, vals, idxs),
        )

        def outer_body(i: torch.Tensor, acc: torch.Tensor):
            _, inner_v = tpu_while_loop(
                lambda j, _a: j < 2, lambda j, a: (j + 1, a * 1.25), (z, acc)
            )
            return i + 1, inner_v

        _, final_v = tpu_while_loop(lambda i, _a: i < lim, outer_body, (z, v_chained))
        return final_v, idx_out

    x_cpu = torch.tensor([[1.0, 4.0, 2.0, 3.0], [5.0, 0.5, 2.5, 1.5]])
    ref_v, ref_i = nested_and_topk(x_cpu, torch.tensor(2, dtype=torch.int32))
    got_v, got_i = torch.compile(
        nested_and_topk, backend="tpu", fullgraph=True, dynamic=False
    )(x_cpu.to("tpu"), torch.tensor(2, dtype=torch.int32, device="tpu"))
    torch.testing.assert_close(got_v.cpu(), ref_v, atol=1e-4, rtol=1e-4)
    assert torch.equal(got_i.cpu(), ref_i)


def _check_disjointness_and_recompile_stability() -> None:
    from vllm_torchtpu.compilation.while_loop import tpu_while_loop

    pool = torch.ones((8, 8), dtype=torch.float32, device="tpu")
    s0 = torch.zeros((), dtype=torch.int32, device="tpu")

    with pytest.raises(ValueError, match="storage"):
        tpu_while_loop(
            lambda s, _a, _b: s < 1,
            lambda s, a, b: (s + 1, a + 1.0, b + 1.0),
            (s0, pool[0:4], pool[2:6]),
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def nested_hoisted(lim: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        i0 = torch.zeros((), dtype=torch.int32, device=x.device)

        def outer_body(i: torch.Tensor, cx: torch.Tensor):
            _, nx, _ = tpu_while_loop(
                lambda j, _u, _v: j < 2,
                lambda j, u, v: (j + 1, u + 1.0, v + 2.0),
                (torch.zeros((), dtype=torch.int32, device=x.device), cx, y),
            )
            return i + 1, nx

        return tpu_while_loop(lambda i, _cx: i < lim, outer_body, (i0, x))

    lim0 = torch.tensor(0, dtype=torch.int32, device="tpu")
    lim1 = torch.tensor(1, dtype=torch.int32, device="tpu")
    _, ok_x = nested_hoisted(
        lim1,
        torch.ones((4, 8), device="tpu"),
        torch.full((4, 8), 2.0, device="tpu"),
    )
    torch.testing.assert_close(ok_x.cpu(), torch.full((4, 8), 3.0))
    for lim in (lim0, lim1):
        with pytest.raises(ValueError, match="storage"):
            nested_hoisted(lim, pool[0:4], pool[4:8])

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def distinct_or_shared(a: torch.Tensor, b: torch.Tensor):
        z = torch.zeros((), dtype=torch.int32, device=a.device)

        def body(s: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
            nx = (x + 1.0).t().contiguous().t()
            return (
                (s + 1, nx, nx)
                if x is y
                else (s + 1, nx, (y + 2.0).t().contiguous().t())
            )

        _, oa, ob = tpu_while_loop(lambda s, _x, _y: s < 2, body, (z, a, b))
        return oa, ob

    oa = ob = sa = torch.empty(0)
    for _ in range(16):
        oa, ob = distinct_or_shared(
            torch.ones((8, 4), device="tpu").t(),
            torch.full((8, 4), 3.0, device="tpu").t(),
        )
    torch.testing.assert_close(oa.cpu(), torch.full((4, 8), 3.0))
    torch.testing.assert_close(ob.cpu(), torch.full((4, 8), 7.0))
    for _ in range(16):
        sh = torch.ones((8, 4), device="tpu").t()
        sa, _ = distinct_or_shared(sh, sh)
    torch.testing.assert_close(sa.cpu(), torch.full((4, 8), 3.0))


def _run_single_chip_semantics_suite() -> None:
    _activate_patch()
    for check in (
        _check_zero_trip_eos_rng_and_strides,
        _check_pytree_int64_nested_and_chained_loops,
        _check_disjointness_and_recompile_stability,
    ):
        check()
        _release_tpu_memory()


# ---------------------------------------------------------------------------
# Single-Chip Pallas KV Buffer Donation & Continue Decode Suite
# ---------------------------------------------------------------------------


def _build_paged_kv_write_op(op_name: str) -> Any:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from torch_tpu._internal import pallas as torch_pallas

    def _kv_write_kernel(
        _slot_ref: Any, vals_ref: Any, _pool_in_ref: Any, pool_out_ref: Any
    ) -> None:
        pool_out_ref[...] = vals_ref[...]

    def _paged_kv_write_jax(
        kv_pool: jax.Array, slot_indices: jax.Array, new_kv: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        slots_i32 = slot_indices.astype(jnp.int32)
        block = (1, kv_pool.shape[1], kv_pool.shape[2])
        updated = pl.pallas_call(
            _kv_write_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(slots_i32.shape[0],),
                in_specs=[
                    pl.BlockSpec(block, lambda i, s: (i, 0, 0)),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=pl.BlockSpec(block, lambda i, s: (s[i], 0, 0)),
            ),
            out_shape=jax.ShapeDtypeStruct(kv_pool.shape, kv_pool.dtype),
            input_output_aliases={2: 0},
        )(slots_i32, new_kv, kv_pool)
        return updated, new_kv.astype(jnp.float32).mean(axis=(1, 2)).sum()

    op = torch_pallas.jax_op(op_name, _paged_kv_write_jax, donate_argnums=(0,))
    op.register_fake(
        lambda kv_pool, _s, _v: (
            torch.empty_like(kv_pool),
            torch.empty((), dtype=torch.float32, device=kv_pool.device),
        )
    )
    return op


def _check_zero_copy_384mb_and_fp8_kv_donation(kv_f32_op: Any, kv_fp8_op: Any) -> None:
    from torch_tpu._internal.pallas import tpu_torch_pallas

    from vllm_torchtpu.compilation.while_loop import tpu_while_loop

    num_steps, batch = 4, 4

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def donated_loop(
        kv_cache: torch.Tensor,
        slots_per_step: torch.Tensor,
        writes: torch.Tensor,
        max_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = kv_cache.device
        s0 = torch.zeros((), dtype=torch.int32, device=dev)
        csum0 = torch.zeros((), dtype=torch.float32, device=dev)
        grid = torch.arange(num_steps, dtype=torch.int32, device=dev)

        def body(
            s: torch.Tensor, k0: torch.Tensor, _k1: torch.Tensor, cs: torch.Tensor
        ):
            sel = grid == s
            sl = (slots_per_step * sel.unsqueeze(1).to(torch.int32)).sum(
                0, dtype=torch.int32
            )
            vl = (writes * sel.view(num_steps, 1, 1, 1).to(writes.dtype)).sum(0)
            op = kv_fp8_op if k0.dtype == torch.float8_e4m3fn else kv_f32_op
            nk, step_sum = op(k0, sl, vl)
            return s + 1, nk, nk, cs + step_sum

        out_s, out_k0, _, out_cs = tpu_while_loop(
            lambda s, _k0, _k1, _cs: s < max_steps,
            body,
            (s0, kv_cache, kv_cache, csum0),
        )
        kv_cache.copy_(out_k0)
        return out_s, out_cs.clamp_(-1e6, 1e6)

    # 384 MiB float32 KV pool (24576 * 32 * 128 * 4B) verifies zero-copy HBM donation.
    kv_cpu = torch.zeros((24576, 32, 128), dtype=torch.float32)
    slots_cpu = torch.tensor(
        [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]],
        dtype=torch.int32,
    )
    writes_cpu = torch.ones((num_steps, batch, 32, 128), dtype=torch.float32) * 1.5
    kv_tpu = kv_cpu.to("tpu")
    slots_tpu, writes_tpu = slots_cpu.to("tpu"), writes_cpu.to("tpu")
    lim_tpu = torch.tensor(num_steps, dtype=torch.int32, device="tpu")

    donors: list[list[int]] = []
    orig_ck = tpu_torch_pallas.call_custom_kernel

    def spy_ck(name: str, fp: str, *args: Any, **kwargs: Any) -> Any:
        if not name.startswith("pallas::while_loop_test"):
            donors.append(list(kwargs.get("donate_argnums") or ()))
        return orig_ck(name, fp, *args, **kwargs)

    tpu_torch_pallas.call_custom_kernel = spy_ck
    try:
        with torch.inference_mode():
            donated_loop(kv_tpu, slots_tpu, writes_tpu, lim_tpu)
            torch.accelerator.synchronize()
    finally:
        tpu_torch_pallas.call_custom_kernel = orig_ck
    assert donors and 1 in donors[-1]

    alloc_base = torch.accelerator.memory_allocated("tpu")
    torch.accelerator.reset_peak_memory_stats("tpu")
    with torch.inference_mode():
        got_s, _ = donated_loop(kv_tpu, slots_tpu, writes_tpu, lim_tpu)
        torch.accelerator.synchronize()
    peak_extra = torch.accelerator.max_memory_allocated("tpu") - alloc_base
    assert int(got_s.cpu().item()) == num_steps
    assert peak_extra < kv_tpu.nbytes // 8
    torch.testing.assert_close(kv_tpu[1].cpu(), torch.full((32, 128), 1.5))

    # FP8 (float8_e4m3fn) KV donation path.
    kv_fp8 = torch.zeros((64, 8, 128), dtype=torch.float8_e4m3fn, device="tpu")
    w_fp8 = torch.full((num_steps, batch, 8, 128), 2.0, device="tpu").to(
        torch.float8_e4m3fn
    )
    with torch.inference_mode():
        s_fp8, _ = donated_loop(kv_fp8, slots_tpu, w_fp8, lim_tpu)
    assert int(s_fp8.cpu().item()) == num_steps
    torch.testing.assert_close(kv_fp8[1].cpu().float(), torch.full((8, 128), 2.0))


def _check_continue_decode_burst_parity(kv_f32_op: Any) -> None:
    from vllm_torchtpu.compilation.while_loop import (
        masked_update_step_slice,
        tpu_while_loop,
    )

    max_burst, batch, heads, dim, eos_id = 4, 3, 4, 128, 3
    gen = torch.Generator().manual_seed(99)
    embed_cpu = torch.randn((8, heads * dim), generator=gen) * 0.3
    head_cpu = torch.randn((heads * dim, 8), generator=gen) * 0.3

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def decode_burst(
        kv_pool: torch.Tensor,
        embed: torch.Tensor,
        lm_head: torch.Tensor,
        tok0: torch.Tensor,
        pos0: torch.Tensor,
        act0: torch.Tensor,
        base_blocks: torch.Tensor,
        max_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = kv_pool.device
        s0 = torch.zeros((), dtype=torch.int32, device=dev)
        win0 = torch.full((max_burst, batch), -1, dtype=torch.int32, device=dev)
        scratch = torch.zeros_like(base_blocks)

        def cond(s, _t, _p, act, _kv, _w):
            return (s < max_steps) & act.any()

        def body(s, tok, pos, act, kv, win):
            slots = torch.where(act, base_blocks + pos, scratch)
            h = torch.tanh(embed.index_select(0, tok))
            nkv, _ = kv_f32_op(kv, slots, h.view(batch, heads, dim))
            logits = h @ lm_head
            sampled = logits.argmax(dim=-1).to(torch.int32)
            ntok = torch.where(act, sampled, torch.zeros_like(sampled))
            nwin = masked_update_step_slice(win, ntok, s, dim=0)
            nact = act & (ntok != eos_id)
            return s + 1, ntok, pos + act.to(torch.int32), nact, nkv, nwin

        out_s, _, _, out_act, out_kv, out_win = tpu_while_loop(
            cond, body, (s0, tok0, pos0, act0, kv_pool, win0)
        )
        kv_pool.copy_(out_kv)
        return out_s, out_act, out_win

    tok_init = torch.tensor([1, 2, 4], dtype=torch.int32)
    pos_init = torch.zeros((batch,), dtype=torch.int32)
    act_init = torch.tensor([True, True, False])
    blocks = torch.tensor([4, 12, 20], dtype=torch.int32)
    kv_init = torch.zeros((32, heads, dim), dtype=torch.float32)

    # CPU step-by-step reference.
    r_s, r_tok, r_pos, r_act = 0, tok_init.clone(), pos_init.clone(), act_init.clone()
    r_kv = kv_init.clone()
    r_win = torch.full((max_burst, batch), -1, dtype=torch.int32)
    while r_s < max_burst and bool(r_act.any()):
        sl = torch.where(r_act, blocks + r_pos, torch.zeros_like(blocks))
        h = torch.tanh(embed_cpu.index_select(0, r_tok))
        for b in range(batch):
            r_kv[int(sl[b].item())] = h[b].view(heads, dim)
        samp = (h @ head_cpu).argmax(dim=-1).to(torch.int32)
        r_tok = torch.where(r_act, samp, torch.zeros_like(samp))
        r_win[r_s] = r_tok
        r_act = r_act & (r_tok != eos_id)
        r_pos = r_pos + r_act.to(torch.int32)
        r_s += 1

    kv_tpu = kv_init.to("tpu")
    got_s, got_act, got_win = decode_burst(
        kv_tpu,
        embed_cpu.to("tpu"),
        head_cpu.to("tpu"),
        tok_init.to("tpu"),
        pos_init.to("tpu"),
        act_init.to("tpu"),
        blocks.to("tpu"),
        torch.tensor(max_burst, dtype=torch.int32, device="tpu"),
    )
    assert int(got_s.cpu().item()) == r_s
    assert torch.equal(got_act.cpu(), r_act)
    assert torch.equal(got_win.cpu(), r_win)
    torch.testing.assert_close(kv_tpu[4:].cpu(), r_kv[4:], atol=1e-3, rtol=1e-3)


def _run_single_chip_kv_donation_suite() -> None:
    _activate_patch()
    kv_f32 = _build_paged_kv_write_op("pallas::while_loop_test_kv_f32")
    kv_fp8 = _build_paged_kv_write_op("pallas::while_loop_test_kv_fp8")
    _check_zero_copy_384mb_and_fp8_kv_donation(kv_f32, kv_fp8)
    _release_tpu_memory()
    _check_continue_decode_burst_parity(kv_f32)
    _release_tpu_memory()


class TestSingleChipWhileLoopSuites:
    @pytest.mark.spawns_tpu_workers
    def test_single_chip_while_loop_semantics_and_contracts(self) -> None:
        run_in_isolated_process(_run_single_chip_semantics_suite)

    @pytest.mark.spawns_tpu_workers
    def test_single_chip_pallas_kv_donation_and_continue_decode(self) -> None:
        run_in_isolated_process(_run_single_chip_kv_donation_suite)


# ---------------------------------------------------------------------------
# Multi-Chip Collectives & Permuted ep_mesh Suite
# ---------------------------------------------------------------------------


def _select_multichip_world_size() -> int:
    from torch_tpu._internal.utils import hardware

    count = hardware.get_tpu_device_count()
    if count >= 8:
        return 8
    if count >= 4:
        return 4
    pytest.skip(f"Multi-chip test requires >= 4 TPU devices, found {count}")


def _prepare_multichip_worker_env(world_size: int) -> dict[str, str]:
    saved_env, saved_argv = os.environ.copy(), sys.argv[:]
    try:
        for k in (
            "JOB_COMPLETION_INDEX",
            "TPU_WORKER_HOSTNAMES",
            "TORCH_TPU_XPROF_SESSION_ID",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "TORCH_TPU_TOPOLOGY",
        ):
            os.environ.pop(k, None)
        sys.argv = [sys.argv[0]]
        prepare_tpu_environment(world_size=world_size)
        env = os.environ.copy()
    finally:
        sys.argv = saved_argv
        os.environ.clear()
        os.environ.update(saved_env)
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def _multichip_cpu_reference(
    world_size: int,
    mesh_idx_by_rank: list[int],
    max_steps: int,
    eos_rank: int,
    eos_step: int,
    routing_order: list[int] | None = None,
) -> dict[int, dict[str, Any]]:
    route = mesh_idx_by_rank if routing_order is None else routing_order
    rank_at_pos = [0] * world_size
    for r, pos in enumerate(route):
        rank_at_pos[pos] = r

    gen = torch.Generator().manual_seed(707)
    init_h = torch.randn((world_size, 8, 128), generator=gen) * 0.3
    w_tp = torch.randn((world_size, 128, 128), generator=gen) * (
        0.2 / math.sqrt(world_size)
    )
    h_cur = [init_h[r].clone() for r in range(world_size)]
    step, any_eos = 0, False

    while step < max_steps and not any_eos:
        h_mid = torch.tanh(
            torch.stack([h_cur[r] @ w_tp[r] for r in range(world_size)]).sum(0)
        )
        x_in = [h_mid + 0.2 * h_cur[r] for r in range(world_size)]
        tagged = [
            x_in[r] + float(mesh_idx_by_rank[r]) * 0.25 for r in range(world_size)
        ]
        csum = torch.stack(x_in).sum(0)
        h_cur = [
            0.5 * h_mid
            + 0.1 * (csum + 0.5 * tagged[rank_at_pos[(route[r] - 1) % world_size]])
            for r in range(world_size)
        ]
        any_eos = step == eos_step and 0 <= eos_rank < world_size
        step += 1

    return {
        r: {
            "steps": step,
            "any_eos": any_eos,
            "h_init": init_h[r],
            "w_local": w_tp[r],
            "h_final": h_cur[r],
        }
        for r in range(world_size)
    }


def _build_permuted_ep_mesh_and_op(
    world_size: int,
) -> tuple[Any, list[int]]:
    import jax
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.sharding import Mesh, PartitionSpec
    from torch_tpu._internal import pallas as torch_pallas
    from torch_tpu._internal.pallas import get_pallas_mesh

    ranked_devs = list(get_pallas_mesh(axis_names=("d",)).devices.flat)
    ranked_ids = [int(d.id) for d in ranked_devs]
    by_id = {int(d.id): d for d in ranked_devs}
    custom_ids = sorted(ranked_ids)
    if custom_ids == ranked_ids:
        custom_ids = [ranked_ids[i ^ 1] for i in range(world_size)]
    custom_mesh = Mesh(np.asarray([by_id[d] for d in custom_ids]), axis_names=("d",))
    mesh_idx_by_rank = [custom_ids.index(d_id) for d_id in ranked_ids]
    ring_perm = [(i, (i + 1) % world_size) for i in range(world_size)]

    def _permuted_kernel(x_global: jax.Array, idx_global: jax.Array) -> jax.Array:
        @functools.partial(
            jax.shard_map,
            mesh=custom_mesh,
            in_specs=(PartitionSpec("d", None), PartitionSpec("d", None)),
            out_specs=PartitionSpec("d", None),
            check_vma=False,
        )
        def _shard(x_s: jax.Array, idx_s: jax.Array) -> jax.Array:
            tagged = pl.pallas_call(
                lambda xr, ir, or_: or_.__setitem__(Ellipsis, xr[...] + ir[...] * 0.25),
                grid=(1,),
                out_shape=jax.ShapeDtypeStruct(x_s.shape, x_s.dtype),
                in_specs=[
                    pl.BlockSpec(x_s.shape, lambda _: (0, 0)),
                    pl.BlockSpec(idx_s.shape, lambda _: (0, 0)),
                ],
                out_specs=pl.BlockSpec(x_s.shape, lambda _: (0, 0)),
            )(x_s, idx_s)
            return jax.lax.psum(x_s, "d") + 0.5 * jax.lax.ppermute(
                tagged, "d", perm=ring_perm
            )

        return _shard(x_global, idx_global)

    op = torch_pallas.jax_op(
        "pallas::while_loop_test_ep_mesh_op",
        _permuted_kernel,
        mesh=custom_mesh,
        input_partition_specs=(PartitionSpec("d", None), PartitionSpec("d", None)),
        output_partition_specs=(PartitionSpec("d", None),),
    )
    op.register_fake(lambda x_local, _idx: torch.empty_like(x_local))
    return op, mesh_idx_by_rank


def _verify_multichip_ep_and_nested_loops(
    rank: int,
    world_size: int,
    mesh_idx_by_rank: list[int],
    custom_mesh_op: Any,
    ref_all: dict[int, dict[str, Any]],
) -> tuple[int, bool, float]:
    import torch.distributed as dist

    from vllm_torchtpu.compilation.while_loop import tpu_while_loop

    my_ref = ref_all[rank]

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def ep_tp_while(
        h0: torch.Tensor,
        w_loc: torch.Tensor,
        idx_tile: torch.Tensor,
        r_t: torch.Tensor,
        trig_r: torch.Tensor,
        trig_s: torch.Tensor,
        lim: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = h0.device
        s0 = torch.zeros((), dtype=torch.int32, device=dev)
        eos0 = torch.zeros((), dtype=torch.bool, device=dev)
        one_v = torch.ones((1,), dtype=torch.int32, device=dev)
        zero_v = torch.zeros((1,), dtype=torch.int32, device=dev)

        def body(s: torch.Tensor, h: torch.Tensor, eos: torch.Tensor):
            part = (h @ w_loc).clone()
            dist.all_reduce(part, op=dist.ReduceOp.SUM)
            h_mid = torch.tanh(part)
            h_nxt = 0.5 * h_mid + 0.1 * custom_mesh_op(h_mid + 0.2 * h, idx_tile)
            ev = torch.where((r_t == trig_r) & (s == trig_s), one_v, zero_v).clone()
            dist.all_reduce(ev, op=dist.ReduceOp.MAX)
            return s + 1, h_nxt, eos | (ev[0] > 0)

        return tpu_while_loop(lambda s, _h, e: (s < lim) & (~e), body, (s0, h0, eos0))

    h_tpu = my_ref["h_init"].to("tpu")
    w_tpu = my_ref["w_local"].to("tpu")
    tile_tpu = torch.full_like(h_tpu, float(mesh_idx_by_rank[rank]))
    got_s, got_h, got_eos = ep_tp_while(
        h_tpu,
        w_tpu,
        tile_tpu,
        torch.tensor(rank, dtype=torch.int32, device="tpu"),
        torch.tensor(world_size - 3, dtype=torch.int32, device="tpu"),
        torch.tensor(2, dtype=torch.int32, device="tpu"),
        torch.tensor(7, dtype=torch.int32, device="tpu"),
    )
    torch.accelerator.synchronize()
    dist.barrier()

    got_h_cpu = got_h.cpu()
    torch.testing.assert_close(got_h_cpu, my_ref["h_final"], atol=2e-3, rtol=2e-3)

    # Outer ep_mesh + pure inner while_loop + nested 8-chip while_loop.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def mixed_and_nested(h0: torch.Tensor, w_loc: torch.Tensor, tile: torch.Tensor):
        z = torch.zeros((), dtype=torch.int32, device=h0.device)

        def outer_body(i: torch.Tensor, h_out: torch.Tensor):
            m = custom_mesh_op(h_out, tile)
            _, h_in = tpu_while_loop(
                lambda j, _x: j < 2,
                lambda j, x: (j + 1, torch.tanh(x @ w_loc)),
                (z, 0.2 * m),
            )
            return i + 1, 0.5 * h_out + 0.5 * h_in

        return tpu_while_loop(lambda i, _h: i < 2, outer_body, (z, h0))

    ns, _ = mixed_and_nested(h_tpu, w_tpu, tile_tpu)
    torch.accelerator.synchronize()
    dist.barrier()
    assert int(ns.cpu().item()) == 2
    return (
        int(got_s.cpu().item()),
        bool(got_eos.cpu().item()),
        float((got_h_cpu - my_ref["h_final"]).abs().max().item()),
    )


def _run_multichip_worker(result_dir: Path) -> None:
    import torch.distributed as dist

    _activate_patch()
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    report: dict[str, Any] = {"rank": rank}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        custom_op, mesh_idx_by_rank = _build_permuted_ep_mesh_and_op(world_size)
        ref_all = _multichip_cpu_reference(
            world_size, mesh_idx_by_rank, 7, world_size - 3, 2
        )
        steps, eos, err = _verify_multichip_ep_and_nested_loops(
            rank, world_size, mesh_idx_by_rank, custom_op, ref_all
        )
        report.update(
            {
                "my_mesh_idx": mesh_idx_by_rank[rank],
                "steps": steps,
                "any_eos": eos,
                "max_abs_err": err,
            }
        )
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        report["error"] = f"{exc!r}\n{traceback.format_exc()}"
    finally:
        (result_dir / f"rank_{rank}.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        if initialized:
            dist.destroy_process_group()
    if "error" in report:
        raise SystemExit(1)


@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
def test_multichip_collectives_and_custom_ep_mesh_device_assignment(
    tmp_path: Path,
) -> None:
    world_size = _select_multichip_world_size()
    result_dir = tmp_path / "multichip_while_loop"
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={world_size}",
        str(Path(__file__).resolve()),
        "--multichip-worker",
        "--result-dir",
        str(result_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=Path(__file__).resolve().parents[2],
        env=_prepare_multichip_worker_env(world_size),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=_MULTICHIP_LAUNCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        output, _ = proc.communicate()
        pytest.fail(f"Multichip worker timed out:\n{(output or '')[-6000:]}")

    results = [
        json.loads((result_dir / f"rank_{r}.json").read_text(encoding="utf-8"))
        for r in range(world_size)
        if (result_dir / f"rank_{r}.json").exists()
    ]
    assert len(results) == world_size and proc.returncode == 0, output[-8000:]
    assert not [r for r in results if "error" in r], output[-8000:]
    assert sum(r["my_mesh_idx"] != r["rank"] for r in results) >= 2
    for r in results:
        assert r["steps"] == 3 and r["any_eos"] is True and r["max_abs_err"] < 2e-3


def _cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multichip-worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.multichip_worker:
        raise ValueError("Expected --multichip-worker")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    _run_multichip_worker(args.result_dir)


if __name__ == "__main__":
    _cli_main()
