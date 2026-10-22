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
"""Independent protocol, resource, routing and native causal contracts."""

import dataclasses
import functools
import hashlib
import math
import runpy
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from test_sparse_gqa_decode import _specs

from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import qsa_attention
from vllm_torchtpu.kernels.sparse_gqa_decode import abi, schedule
from vllm_torchtpu.kernels.sparse_gqa_decode.eligibility import plan_decode
from vllm_torchtpu.kernels.sparse_gqa_decode.planning import (
    make_config,
    resource_usage,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.tuned_block_sizes import DecodePolicy


def test_benchmark_provenance_records_late_loaded_sources(monkeypatch, tmp_path):
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    script = (
        Path(qsa_rpa.__file__).parents[4] / "scripts/benchmark_sparse_gqa_decode.py"
    )
    provenance = runpy.run_path(str(script))["provenance"]
    before = provenance([])
    source = tmp_path / "late_kernel.py"
    source.write_text("# controlled late-loaded source\n")
    name = "vllm_torchtpu.kernels.sparse_gqa_decode._late_probe"
    module = types.ModuleType(name)
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, name, module)
    after = provenance([])
    assert name not in before["loaded_sources"]
    assert after["loaded_sources"][name] == dict(
        path=str(source.resolve()),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def test_protocol_field_growth_fails_closed(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    cfg = make_config(3, 12, 128, 2, 4, 17, jnp.dtype(jnp.bfloat16))
    shape = schedule.schedule.BaseMetadataComputer.get_rpa_schedule(cfg)
    assert set(schedule.field_names(shape)) == {
        f.name for f in dataclasses.fields(shape)
    } - {"cfgs", "actual_steps"}
    extended = dataclasses.make_dataclass(
        "ExtendedSchedule",
        [(f.name, object) for f in dataclasses.fields(shape)]
        + [("future_dma", object)],
    )
    with pytest.raises(ValueError, match="Unhandled RpaSchedule"):
        schedule.field_names(extended(*([None] * len(dataclasses.fields(extended)))))


def test_resource_bytes_and_boundary_are_independent_of_policy_names(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    cfg = make_config(64, 24, 256, 2, 4, 512, jnp.dtype(jnp.bfloat16))
    usage = resource_usage(cfg)
    # Enumerate actual scalar ABI shapes, independent of the planner formula.
    assert (
        usage.scalar_bytes
        == sum(np.prod(s) for s in [(65,), (64,), (64,), (64, 513)])
        * np.dtype(np.int32).itemsize
    )
    assert usage.schedule_bytes == sum(
        x.size * x.dtype.itemsize
        for x in jax.tree.leaves(
            schedule.schedule.BaseMetadataComputer.get_rpa_schedule(cfg)
        )
    )
    assert usage.worst_steps == math.ceil(64 * math.ceil(513 * 4 / 256) / 8)
    assert usage.fits(info)
    assert not usage.fits(
        dataclasses.replace(
            info,
            smem_capacity_bytes=usage.scalar_bytes
            + usage.schedule_bytes
            + 32 * 1024
            - 1,
        )
    )
    assert not usage.fits(
        dataclasses.replace(
            info, vmem_capacity_bytes=math.ceil(usage.vmem_estimate_bytes / 0.8) - 1
        )
    )


def test_planned_routing_reuses_plan_and_actual_small_chunk(monkeypatch):
    import vllm_torchtpu.kernels.sparse_gqa_decode as public
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = _specs()
    calls = []
    real_plan = public.plan_decode

    def counted(*a, **kw):
        result = real_plan(*a, **kw)
        calls.append(result)
        return result

    def execute(*a, launch, **kw):
        assert launch is calls[0]
        smaller = launch.config_for_batch(3)
        assert smaller.serve.num_seqs == 3 and smaller.serve.num_page_indices == 3 * 513
        return a[0], a[5]

    monkeypatch.setattr(public, "plan_decode", counted)
    monkeypatch.setattr(decode, "execute_decode", execute)
    qsa_rpa.qsa_attention(
        *args, jax.ShapeDtypeStruct((9,), jnp.int32), kv_heads=2, ratio=4
    )
    assert len(calls) == 1


def test_policy_static_identity_and_layout_constraints(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    args = _specs()
    plans = [
        plan_decode(
            *args,
            kv_heads=2,
            ratio=4,
            k_scale=1.0,
            v_scale=1.0,
            policy=DecodePolicy(schedule_copy_words=n),
        )
        for n in [128, 256, 1024]
    ]
    assert all(x is not None for x in plans)
    assert len({x.policy for x in plans}) == 3
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    def check_page(*inputs, ratio, page_size):
        assert ratio == 4 and page_size == args[5].shape[1]
        raise RuntimeError("metadata checked before tensor operations")

    monkeypatch.setattr(decode, "virtual_page_metadata", check_page)
    for launch in plans:
        with pytest.raises(RuntimeError, match="metadata checked"):
            decode._attend(*args, launch=launch, k_scale=1.0, v_scale=1.0)
    assert (
        plan_decode(
            *args,
            kv_heads=2,
            ratio=4,
            k_scale=1.0,
            v_scale=1.0,
            policy=DecodePolicy(schedule_copy_words=129),
        )
        is None
    )
    with pytest.raises(ValueError, match="scalar"):
        abi.output_aliases((None,), ())
    assert abi.output_aliases((None,) * 4, (None,)) == {4: 0, 6: 1}


def test_module_entry_signature_remains_compatible(monkeypatch):
    import vllm_torchtpu.kernels.sparse_gqa_decode as public
    from vllm_torchtpu.kernels.sparse_gqa_decode import decode

    args = _specs()

    def checked(*inputs, kv_heads, ratio, k_scale, v_scale):
        assert (kv_heads, ratio, k_scale, v_scale) == (2, 4, 0.375, 2.5)
        return inputs[0], inputs[5]

    monkeypatch.setattr(public, "qsa_decode_attention", checked)
    assert decode.qsa_decode_attention(
        *args, kv_heads=2, ratio=4, k_scale=0.375, v_scale=2.5
    ) == (args[0], args[5])


@pytest.mark.tpu_test
@pytest.mark.parametrize("alignment", [128, 256, 1024])
def test_schedule_prefix_tail_capacity(alignment):
    prefix = np.array([0, 1, 127, 128, 129, 255, 256, 1023, 1024, 1025], np.int32)
    for capacity in [128, 256, 1152]:
        actual = np.asarray(
            jax.jit(
                lambda p, capacity=capacity: abi.schedule_copy_size(
                    p, capacity, alignment
                )
            )(jnp.asarray(prefix))
        )
        expected = np.array(
            [min(math.ceil(int(p) / alignment) * alignment, capacity) for p in prefix]
        )
        np.testing.assert_array_equal(actual, expected)
    assert np.dtype(np.uint32).itemsize * 128 == abi.RPA_WAIT_ROW_BYTES


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("batch", [6, 12, 24])
def test_schedule_fields_against_scalar_task_oracle(fp8, batch):
    dtype = jnp.dtype(jnp.float8_e4m3fn if fp8 else jnp.bfloat16)
    cfg = make_config(batch, 8, 128, 1, 8, 65, dtype)
    lengths = np.tile(np.array([0, 1, 255, 256, 257, 520], np.int32), batch // 6)
    cu = np.arange(batch + 1, dtype=np.int32)
    actual = schedule.generate(jnp.asarray(cu), jnp.asarray(lengths), cfg)
    tasks = [
        (row, tile, int(length))
        for row, length in enumerate(lengths)
        for tile in range(math.ceil(int(length) / 256))
    ]
    steps = math.ceil(len(tasks) / 8)
    assert np.asarray(actual.actual_steps)[0] == steps
    for step in range(steps):
        tokens = 0
        live = 0
        lasts = 0
        for lane in range(8):
            i = step * 8 + lane
            if i >= len(tasks):
                assert np.asarray(actual.s_idx.data).reshape(-1, 8)[step, lane] == -1
                continue
            row, tile, length = tasks[i]
            live += 1
            tokens += min(256, length - tile * 256)
            last = tile == math.ceil(length / 256) - 1
            lasts += last
            assert np.asarray(actual.s_idx.data).reshape(-1, 8)[step, lane] == row
            assert np.asarray(actual.k_idx.data).reshape(-1, 8)[step, lane] == tile
            assert np.asarray(actual.is_last_k.data).reshape(-1, 8)[step, lane] == last
            np.testing.assert_array_equal(
                np.asarray(actual.dma_q.data).reshape(-1, 8, 2)[step, lane],
                [row, 1],
            )
            dma = np.asarray(actual.dma_kv_cache.data).reshape(
                cfg.max_steps_ub, 8, cfg.bkv_p_cache, 3
            )
            for page in range(cfg.bkv_p_cache):
                np.testing.assert_array_equal(
                    dma[step, lane, page],
                    [
                        min(row * 66 + tile * 32 + page, batch * 66 - 1),
                        page * 8,
                        max(0, min(length - tile * 256 - page * 8, 8)),
                    ],
                )
        assert (
            np.asarray(actual.total_wait_kv_in.data)[step]
            == tokens * cfg.kv_bytes_per_token // 512
        )
        assert (
            np.asarray(actual.total_wait_q_in.data)[step]
            == live * cfg.q_bytes_per_token // 512
        )
        assert (
            np.asarray(actual.total_wait_o_out.data)[step]
            == lasts * cfg.o_bytes_per_token // 512
        )
    for name in schedule.UNUSED_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(actual, name).data), 0)


@pytest.mark.tpu_test
def test_integer_rne_midpoints_are_observable():
    from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

    # Exact FP32 BF16 midpoints: even low retained bit rounds down, odd up.
    bits = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], np.uint32)
    actual = np.asarray(jax.jit(bf16)(jnp.asarray(bits.view(np.float32)))).view(
        np.uint16
    )
    np.testing.assert_array_equal(actual, [0x3F80, 0x3F82, 0xBF80, 0xBF82])


@pytest.mark.tpu_test
def test_all_head_analytic_causal_duplicates_and_cache():
    # Independent token-space oracle: ratio virtual pages cross physical page 136.
    batch, heads, kv, dim, ratio, groups, page = 6, 8, 1, 128, 8, 17, 136
    lengths = np.array([7, 8, 9, 137, 155, 0], np.int32)
    columns = 2
    packing = 2
    table = (
        np.random.default_rng(2123)
        .permutation(batch * columns)
        .reshape(batch, columns)
        .astype(np.int32)
    )
    host_cache = np.zeros((batch * columns, page, 1, packing, dim), np.float32)
    blocks = np.full((batch, groups), -1, np.int32)
    for row, L in enumerate(lengths):
        n = min(int(L) // ratio, groups)
        if n:
            blocks[row, :n] = np.arange(n)
        if row == 4:
            blocks[row, :n] = np.array([18, 0, 0, *range(1, 15)], np.int32)
        for logical in range(columns):
            for token in range(page):
                pos = logical * page + token
                physical = table[row, logical]
                host_cache[physical, token, 0, 0] = np.float32((pos % 13) / 16)
                host_cache[physical, token, 0, 1] = np.float32(
                    (pos % 23) / 16 + (row + 1) / 8
                )
            # Poison every future token, including padded final tail groups.
            first = max(0, int(L) - logical * page)
            if first < page:
                host_cache[int(table[row, logical]), first:] = np.nan
    cache = jnp.asarray(host_cache, jnp.bfloat16)
    cache_host = np.asarray(cache).copy()
    q = jnp.zeros((batch, heads, dim), jnp.bfloat16)
    gate = jnp.zeros_like(q)
    q = q.at[-1].set(jnp.nan)
    gate = gate.at[-1].set(jnp.nan)
    ks, vs = 1.25, 0.5
    args = (
        q,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
    )
    exe = (
        jax.jit(
            functools.partial(
                qsa_attention,
                starts=jnp.arange(batch + 1, dtype=jnp.int32),
                kv_heads=kv,
                ratio=ratio,
                k_scale=ks,
                v_scale=vs,
            ),
            donate_argnums=(5,),
        )
        .lower(*args)
        .compile()
    )
    hlo = exe.as_text()
    assert "fused_qsa_decode" in hlo
    stats = exe.memory_analysis()
    mem = {
        k: getattr(stats, k)
        for k in [
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        ]
    }
    # Compiled alias size tests outer donation, not every internal operation.
    assert mem["alias_size_in_bytes"] >= cache_host.nbytes
    actual, returned = jax.block_until_ready(exe(*args))
    actual = np.asarray(actual, np.float32)
    wanted = np.zeros_like(actual, np.float64)
    row_errors = []
    for row, L in enumerate(lengths):
        if not L:
            continue
        ids = []
        for g in blocks[row, : min(int(L) // ratio, groups)]:
            ids.extend(range(int(g) * ratio, (int(g) + 1) * ratio))
        ids.extend(range(int(L) // ratio * ratio, int(L)))
        v = np.stack(
            [cache_host[table[row, t // page], t % page, 0, 1] for t in ids]
        ).astype(np.float64)
        target = (v.mean(axis=0) * vs).astype(np.float32).astype(
            ml_dtypes.bfloat16
        ).astype(np.float32) * np.float32(0.5)
        target = target.astype(ml_dtypes.bfloat16).astype(np.float32)
        wanted[row] = target
        for head in range(heads):
            err = float(
                np.linalg.norm(actual[row, head] - target)
                / max(np.linalg.norm(target), 1e-12)
            )
            assert err < 0.01, (row, head, err)
        row_errors.append(
            float(
                np.linalg.norm(actual[row] - wanted[row])
                / max(np.linalg.norm(wanted[row]), 1e-12)
            )
        )
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[-1], 0)
    np.testing.assert_array_equal(
        np.asarray(returned).view(np.uint8), cache_host.view(np.uint8)
    )
