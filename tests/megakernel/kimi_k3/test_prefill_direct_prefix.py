# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for K3 prefill zero-prefix fast path and direct rank-local arena."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

if "torch_tpu" not in sys.modules:
    for mod_name in (
        "torch_tpu",
        "torch_tpu._internal",
        "torch_tpu._internal.pallas",
        "torch_tpu._internal.pallas.pallas",
    ):
        sys.modules.setdefault(mod_name, ModuleType(mod_name))
    sys.modules["torch_tpu._internal.pallas"].pallas = sys.modules[  # type: ignore[attr-defined]
        "torch_tpu._internal.pallas.pallas"
    ]

from vllm_torchtpu.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_megakernel import (
    DirectPrefix,
    append_direct_prefix,
)
from vllm_torchtpu.megakernel.kimi_k3.prefill import (
    KDA_LAYERS,
    MLA_LAYERS,
    KimiPrefiller,
)
from vllm_torchtpu.megakernel.program import GREEDY, RequestInput, StepInput

pytestmark = pytest.mark.cpu_test


def test_direct_prefix_split_returns_shared_multi_layer_pair():
    arena = DirectPrefix("keys_hbm", "values_hbm")
    first, second = arena.split()
    assert first is arena
    assert second is arena


def test_append_direct_matches_reference_up_projection_bit_for_bit():
    rng = np.random.default_rng(42)
    window = 64
    capacity = 256
    pool_dtype = jnp.float8_e4m3fn
    arena_dtype = jnp.float8_e4m3fn

    latents = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, window, 576), dtype=np.float32) * 0.25,
        dtype=jnp.bfloat16,
    )
    m_kb = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, 512, 768), dtype=np.float32) * 0.05,
        dtype=jnp.bfloat16,
    )
    arena_k = jnp.zeros((1, MLA_LAYERS, 3, capacity, 256), dtype=arena_dtype)
    arena_v = jnp.zeros((1, MLA_LAYERS, 3, 128, capacity), dtype=arena_dtype)
    pos = jnp.int32(128)

    def step(m, carry):
        ak, av = carry
        x = jax.lax.dynamic_index_in_dim(latents[0], m, axis=0, keepdims=False)
        x = x.astype(pool_dtype).astype(jnp.bfloat16)
        up = jax.lax.dynamic_index_in_dim(m_kb[0], m, axis=0, keepdims=False)
        rope = x[:, 512:576]
        expanded = jnp.dot(x[:, :512], up, preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        )
        eh = expanded.reshape(window, 3, 256).transpose(1, 0, 2)
        k_new = jnp.concatenate(
            (
                eh[:, :, :128],
                jnp.broadcast_to(rope[None], (3, window, 64)),
                jnp.zeros((3, window, 64), jnp.bfloat16),
            ),
            axis=-1,
        ).astype(ak.dtype)
        v_new = jnp.swapaxes(eh[:, :, 128:], 1, 2).astype(av.dtype)
        ak = jax.lax.dynamic_update_slice(ak, k_new[None, None], (0, m, 0, pos, 0))
        av = jax.lax.dynamic_update_slice(av, v_new[None, None], (0, m, 0, 0, pos))
        return ak, av

    out_k, out_v = jax.jit(
        lambda ak, av, lat, kb, p: jax.lax.fori_loop(0, MLA_LAYERS, step, (ak, av))
    )(arena_k, arena_v, latents, m_kb, pos)

    # Compare every MLA layer and head against the exact _expand_pages_kernel formula.
    for m in range(MLA_LAYERS):
        x_all = latents[0, m].astype(pool_dtype).astype(jnp.bfloat16)
        rope_all = x_all[:, 512:576]
        expanded_all = jnp.dot(
            x_all[:, :512], m_kb[0, m], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        zeros_all = jnp.zeros((window, 64), jnp.bfloat16)
        for h in range(3):
            eh = expanded_all[:, h * 256 : (h + 1) * 256]
            ref_k = jnp.concatenate((eh[:, :128], rope_all, zeros_all), axis=-1).astype(
                arena_dtype
            )
            ref_v = eh[:, 128:].T.astype(arena_dtype)
            np.testing.assert_array_equal(
                np.asarray(out_k[0, m, h, 128 : 128 + window].astype(jnp.float32)),
                np.asarray(ref_k.astype(jnp.float32)),
            )
            np.testing.assert_array_equal(
                np.asarray(out_v[0, m, h, :, 128 : 128 + window].astype(jnp.float32)),
                np.asarray(ref_v.astype(jnp.float32)),
            )


def test_pallas_append_direct_prefix_interpret_matches_reference():
    rng = np.random.default_rng(7)
    window = 512
    capacity = 1536
    pool_dtype = jnp.float8_e4m3fn
    arena_dtype = jnp.float8_e4m3fn

    latents = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, window, 576), dtype=np.float32) * 0.25,
        dtype=jnp.bfloat16,
    )
    m_kb = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, 512, 768), dtype=np.float32) * 0.05,
        dtype=jnp.bfloat16,
    )
    sentinel_k = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, 3, capacity, 256), dtype=np.float32) * 0.1,
        dtype=arena_dtype,
    )
    sentinel_v = jnp.asarray(
        rng.standard_normal((1, MLA_LAYERS, 3, 128, capacity), dtype=np.float32) * 0.1,
        dtype=arena_dtype,
    )
    start = jnp.array([512], dtype=jnp.int32)

    out_k, out_v = append_direct_prefix(
        sentinel_k,
        sentinel_v,
        latents,
        m_kb,
        start,
        pool_dtype=pool_dtype,
        interpret=True,
    )

    # Sentinel regions before 512 and after 1024 must be completely untouched.
    np.testing.assert_array_equal(
        np.asarray(out_k[:, :, :, :512, :].astype(jnp.float32)),
        np.asarray(sentinel_k[:, :, :, :512, :].astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(out_k[:, :, :, 1024:, :].astype(jnp.float32)),
        np.asarray(sentinel_k[:, :, :, 1024:, :].astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(out_v[:, :, :, :, :512].astype(jnp.float32)),
        np.asarray(sentinel_v[:, :, :, :, :512].astype(jnp.float32)),
    )
    np.testing.assert_array_equal(
        np.asarray(out_v[:, :, :, :, 1024:].astype(jnp.float32)),
        np.asarray(sentinel_v[:, :, :, :, 1024:].astype(jnp.float32)),
    )

    for m in range(MLA_LAYERS):
        x_all = latents[0, m].astype(pool_dtype).astype(jnp.bfloat16)
        rope_all = x_all[:, 512:576]
        expanded_all = jnp.dot(
            x_all[:, :512], m_kb[0, m], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        zeros_all = jnp.zeros((window, 64), jnp.bfloat16)
        for h in range(3):
            eh = expanded_all[:, h * 256 : (h + 1) * 256]
            ref_k = jnp.concatenate((eh[:, :128], rope_all, zeros_all), axis=-1).astype(
                arena_dtype
            )
            ref_v = eh[:, 128:].T.astype(arena_dtype)
            np.testing.assert_array_equal(
                np.asarray(out_k[0, m, h, 512:1024].astype(jnp.float32)),
                np.asarray(ref_k.astype(jnp.float32)),
            )
            np.testing.assert_array_equal(
                np.asarray(out_v[0, m, h, :, 512:1024].astype(jnp.float32)),
                np.asarray(ref_v.astype(jnp.float32)),
            )


def _make_harness(monkeypatch) -> tuple[KimiPrefiller, list, list]:
    """Build a CPU-only KimiPrefiller harness recording run_packed & append calls."""
    monkeypatch.setattr(
        "vllm_torchtpu.megakernel.kimi_k3.prefill._device_i32",
        lambda vals: torch.tensor(vals, dtype=torch.int32),
    )
    monkeypatch.setattr("vllm_torchtpu.megakernel.kimi_k3.prefill.logger", MagicMock())
    orig_to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *args, **kwargs: (
            self if args and args[0] == "tpu" else orig_to(self, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        k3_config,
        "_ACTIVE",
        k3_config.KernelConfig(prefill_prefix_fp8=True, prefill_direct_prefix=True),
    )

    prefiller = object.__new__(KimiPrefiller)
    prefiller.config = SimpleNamespace(max_num_seqs=64, block_size=512)
    prefiller.page_tokens = 512
    prefiller.cp_rank = 0
    prefiller.compute_logits = False
    prefiller.slots = 1
    prefiller.cp_capacity = 5120
    prefiller.rank = 0
    prefiller._requests = {}
    prefiller._zero_pair = (torch.zeros(1), torch.zeros(1))
    prefiller._mla_group = 0
    prefiller._kda_groups = [(1, 512)] * KDA_LAYERS
    prefiller._zero_logits = None
    prefiller._direct_capacity = 65536
    prefiller._direct_arena = (torch.zeros(1), torch.zeros(1))
    prefiller._direct_req_id = None
    prefiller._direct_upto = 0
    prefiller._direct_ops = {}
    prefiller.ring = None
    prefiller.draft_latents = None

    run_calls: list[tuple[int, str, bool]] = []
    append_calls: list[tuple[int, int]] = []

    program = MagicMock()
    program.windows = (512, 1024, 2048, 4096, 8192, 16384)
    program.capacity = 131072
    program.window_slots.side_effect = lambda w: 1
    program.has_mode.side_effect = lambda w, mode="cp": True
    program.sp_order.side_effect = lambda w: np.arange(w, dtype=np.int32)
    program.stack_states.side_effect = lambda states: (torch.zeros(1), torch.zeros(1))
    program.split_states.side_effect = lambda r, c: [(r, c)]
    program.last_routes = None

    def fake_run_packed(
        shard,
        extent,
        segments,
        rec,
        conv,
        pools,
        ids,
        local,
        ring_hook,
        *,
        prefix_mode="cp",
        direct_arena=None,
    ):
        window = int(shard.shape[-1]) * 32
        run_calls.append((window, prefix_mode, ids is not None))
        latents = torch.zeros((1, MLA_LAYERS, window, 576), dtype=torch.bfloat16)
        return None, latents, rec, conv, rec, conv

    program.run_packed.side_effect = fake_run_packed
    prefiller.program = program

    writer = MagicMock()
    writer.mla_pools.return_value = [torch.zeros(1)] * MLA_LAYERS
    prefiller._writer = writer

    def fake_append_direct(window: int, latents: torch.Tensor, start: int) -> None:
        append_calls.append((window, start))

    prefiller._append_direct = fake_append_direct  # type: ignore[method-assign]
    return prefiller, run_calls, append_calls


def _req_input(req_id: str, prompt_len: int, computed: int, end: int) -> RequestInput:
    blocks = [np.arange(1, 300, dtype=np.int32), np.arange(1, 300, dtype=np.int32)]
    return RequestInput(
        req_id=req_id,
        token_ids=np.ones(prompt_len, dtype=np.int32),
        num_tokens=prompt_len,
        num_prompt_tokens=prompt_len,
        computed=computed,
        scheduled=end - computed,
        block_tables=blocks,
        sampling=GREEDY,
    )


def test_prefiller_80k_uses_none_then_direct_and_skips_final_append(monkeypatch):
    prefiller, run_calls, append_calls = _make_harness(monkeypatch)
    prompt = 80127
    # 5 chunked scheduler steps of 16384 (final step finishes at 80127)
    bounds = [
        (0, 16384),
        (16384, 32768),
        (32768, 49152),
        (49152, 65536),
        (65536, prompt),
    ]
    for computed, end in bounds:
        prefiller.step(
            StepInput(
                requests=[_req_input("req-80k", prompt, computed, end)],
                copies=[],
                finished=[],
            )
        )

    assert run_calls == [
        (16384, "none", False),
        (16384, "direct", False),
        (16384, "direct", False),
        (16384, "direct", False),
        (16384, "direct", False),
    ]
    assert append_calls == [
        (16384, 0),
        (16384, 16384),
        (16384, 32768),
        (16384, 49152),
    ]
    assert prefiller._direct_req_id is None
    assert prefiller._direct_upto == 0


def test_prefiller_short_and_prefix_hit_and_invalidation(monkeypatch):
    prefiller, run_calls, append_calls = _make_harness(monkeypatch)

    # 1. Start a multi-window request (chunk 0: 0..16384 of 32768)
    prefiller.step(
        StepInput(
            requests=[_req_input("req-long", 32768, 0, 16384)],
            copies=[],
            finished=[],
        )
    )
    assert prefiller._direct_req_id == "req-long"
    assert prefiller._direct_upto == 16384

    # 2. Interleave a short 1-window request (0..1024): uses "none", skips append,
    #    and preserves req-long's direct arena!
    prefiller.step(
        StepInput(
            requests=[_req_input("req-short", 1024, 0, 1024)],
            copies=[],
            finished=[],
        )
    )
    assert prefiller._direct_req_id == "req-long"
    assert prefiller._direct_upto == 16384

    # 3. Finish req-long (16384..32768): still uses "direct"!
    prefiller.step(
        StepInput(
            requests=[_req_input("req-long", 32768, 16384, 32768)],
            copies=[],
            finished=[],
        )
    )
    assert prefiller._direct_req_id is None

    # 4. Prefix-cache hit starting at 16384 falls back to "cp" (with CP operands)
    prefiller.program.split_states.side_effect = lambda r, c: [(r, c)]
    prefiller._writer.read_states.return_value = (torch.zeros(1), torch.zeros(1))
    prefiller.step(
        StepInput(
            requests=[_req_input("req-hit", 32768, 16384, 32768)],
            copies=[],
            finished=[],
        )
    )

    assert run_calls == [
        (16384, "none", False),
        (1024, "none", False),
        (16384, "direct", False),
        (16384, "cp", True),
    ]
    assert append_calls == [(16384, 0)]
