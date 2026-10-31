# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU_DEEPSEEK_V41_ENGRAM_HBM device lookup against the host exchange's layout.

The SPMD lookup check needs 8 devices; without them it reruns itself in a
subprocess on 8 forced CPU devices, so it also runs on the CPU lane.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from jax.sharding import Mesh

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram_hbm

WORLD, NODE, TOKENS, PART_COLS, HEAD_ROWS, DIM, BLOCK = 8, 4, 5, 3, 40, 8, 4
COLS = PART_COLS * NODE


def _reference(raw, scale):
    values = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    power = (scale.astype(np.int32) << 23).view(np.float32)
    power = np.repeat(power, BLOCK, -1)
    invalid = np.repeat(scale == 255, BLOCK, -1)
    return np.where(invalid, 0.0, values * power).astype(ml_dtypes.bfloat16)


def test_lookup_matches_host_exchange_layout():
    if len(jax.devices()) >= WORLD:
        _check_lookup()
        return
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('t', {__file__!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "m._check_lookup()"
    )
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={WORLD}",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


def _check_lookup():
    assert len(jax.devices()) >= WORLD
    rng = np.random.default_rng(0)
    offsets = np.arange(COLS) * HEAD_ROWS  # global row id = head offset + id
    # Rank r holds the heads of node-local shard r % NODE, as the host does.
    meta = tuple(
        (
            PART_COLS * (r % NODE),
            int(offsets[PART_COLS * (r % NODE)]),
            int(offsets[PART_COLS * (r % NODE)] + PART_COLS * HEAD_ROWS),
            PART_COLS * HEAD_ROWS,
        )
        for r in range(WORLD)
    )
    rows = PART_COLS * HEAD_ROWS
    full_raw = rng.integers(0, 255, (COLS * HEAD_ROWS, DIM), dtype=np.uint8)
    full_scale = rng.integers(100, 150, (COLS * HEAD_ROWS, DIM // BLOCK), np.uint8)
    table = np.concatenate([full_raw[m[1] : m[2]] for m in meta])
    scales = np.concatenate([full_scale[m[1] : m[2]] for m in meta])
    hashes = np.stack(
        [offsets + rng.integers(0, HEAD_ROWS, (TOKENS, COLS)) for _ in range(WORLD)]
    ).astype(np.int32)
    hashes[3, 1, 5] = -1  # a dead n-gram reads as zeros

    mesh = Mesh(np.array(jax.devices()[:WORLD]), (engram_hbm._AXIS,))
    lookup = engram_hbm.make_lookup(
        mesh, TOKENS, COLS, PART_COLS, rows, DIM, BLOCK, NODE, meta
    )
    out = lookup(
        jnp.asarray(hashes.reshape(WORLD * TOKENS, COLS)),
        jnp.asarray(table),
        jnp.asarray(scales),
        jnp.arange(WORLD, dtype=jnp.int32).reshape(WORLD, 1),
    )
    out = np.asarray(out).reshape(WORLD, TOKENS, COLS, DIM)

    expected = np.zeros_like(out)
    for r in range(WORLD):
        for t in range(TOKENS):
            for c in range(COLS):
                g = hashes[r, t, c]
                if g >= 0:
                    expected[r, t, c] = _reference(full_raw[g], full_scale[g])
    np.testing.assert_array_equal(out.view(np.uint16), expected.view(np.uint16))


def test_dequant_matches_reference():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 255, (6, DIM), dtype=np.uint8)
    scale = rng.integers(100, 150, (6, DIM // BLOCK), dtype=np.uint8)
    scale[2] = 255
    got = np.asarray(engram_hbm.dequant(jnp.asarray(raw), jnp.asarray(scale), BLOCK))
    want = _reference(raw, scale)
    same = got.view(np.uint16) == want.view(np.uint16)
    both_nan = np.isnan(got.astype(np.float32)) & np.isnan(want.astype(np.float32))
    assert (same | both_nan).all()


def test_default_env_and_uses_hbm_lookup_gating(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.delenv("TPU_DEEPSEEK_V41_ENGRAM_HBM", raising=False)
    assert envs.environment_variables["TPU_DEEPSEEK_V41_ENGRAM_HBM"]() == "first"

    def _check(layer_idx: int, *, dev_type: str = "tpu", dp_size: int = 8) -> bool:
        staged: object = SimpleNamespace(device=SimpleNamespace(type=dev_type))
        return engram_hbm.should_use_hbm_lookup(
            staged_raw=staged,  # type: ignore[arg-type]
            dp_size=dp_size,
            layer_hash_index=layer_idx,
        )

    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: object())
    monkeypatch.setattr(engram_hbm, "_tp_size", lambda: 1)
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HBM", "first")
    assert _check(0) is True
    assert _check(1) is False
    assert _check(0, dev_type="cpu") is False
    assert _check(0, dp_size=1) is False

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HBM", "all")
    assert _check(0) is True
    assert _check(1) is True

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HBM", "off")
    assert _check(0) is False

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_ENGRAM_HBM", "first")
    monkeypatch.setattr(engram_hbm, "_tp_size", lambda: 2)
    assert _check(0) is False
    monkeypatch.setattr(engram_hbm, "_tp_size", lambda: 1)
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: None)
    assert _check(0) is False


def test_upload_table_pads_with_invalid_scale():
    weight = torch.arange(3 * DIM, dtype=torch.uint8).reshape(3, DIM)
    scales = torch.full((3, DIM // BLOCK), 127, dtype=torch.uint8)
    table, table_scales = engram_hbm.upload_table(
        weight, scales, torch.device("cpu"), rows=5
    )
    assert torch.equal(table[:3], weight) and not table[3:].any()
    assert torch.equal(table_scales[:3], scales)
    assert (table_scales[3:] == 255).all()


def _hbm_engram(monkeypatch, *, capacity: int = 8):
    """A host-offload Engram with HBM state preset and the lookup op faked."""
    from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common

    seen: list[torch.Tensor] = []

    def build_lookup_op(layer, num_tokens, n_cols, part_cols, rows, dim, *rest):
        def op(hashes, table, scales, rank):
            seen.append(hashes)
            # Row t of token t, so slicing back to local tokens is visible.
            return (
                torch.arange(num_tokens, dtype=torch.bfloat16)
                .reshape(-1, 1, 1)
                .expand(num_tokens, part_cols * NODE, dim)
                .clone()
            )

        return op

    monkeypatch.setattr(engram_hbm, "build_lookup_op", build_lookup_op)
    monkeypatch.setattr("vllm_torchtpu.distributed.ep_mesh.ep_mesh_index", lambda: 0)
    e = common.Engram.__new__(common.Engram)
    torch.nn.Module.__init__(e)
    e.layer_hash_index = 0
    e.embed_tokens = SimpleNamespace(
        n_hash_cols=COLS, part_n_hash_cols=PART_COLS, block_size=BLOCK, dp_size=NODE
    )
    e._staged_raw = torch.zeros(1)
    e._host_raw = torch.zeros((capacity, COLS, DIM), dtype=torch.uint8)
    e._active_upload = None
    e._uses_hbm = True
    e._hbm_rank = None
    e._hbm_rows = None
    table = torch.zeros((HEAD_ROWS, DIM), dtype=torch.uint8)
    e._hbm_table = (table, table, (), HEAD_ROWS)
    return e, seen


def _dp_sizes(monkeypatch, sizes):
    import vllm.forward_context as fc

    monkeypatch.setattr(fc, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        fc,
        "get_forward_context",
        lambda: SimpleNamespace(
            additional_kwargs={"tpu_num_tokens_across_dp_cpu": tuple(sizes)}
        ),
    )


def test_hbm_prepare_pads_to_dp_max_and_slices_back(monkeypatch):
    from vllm.models.deepseek_v41.common.engram import DEAD_ID

    e, seen = _hbm_engram(monkeypatch)
    _dp_sizes(monkeypatch, (3, 5, 1, 2))
    e._hbm_prepare(torch.ones((3, COLS), dtype=torch.int32))
    (hashes,) = seen
    assert hashes.shape == (5, COLS)  # every rank runs the DP-wide max
    assert (hashes[:3] == 1).all() and (hashes[3:] == DEAD_ID).all()
    rows = e._ready_rows(3)
    assert rows.shape == (3, COLS, DIM)
    assert torch.equal(rows[:, 0, 0], torch.arange(3, dtype=torch.bfloat16))


def test_hbm_prepare_idle_rank_still_joins_lookup(monkeypatch):
    e, seen = _hbm_engram(monkeypatch)
    _dp_sizes(monkeypatch, (0, 4, 0, 0))
    e._hbm_prepare(torch.ones((0, COLS), dtype=torch.int32))
    assert seen[0].shape == (4, COLS)  # the collective needs every rank
    assert e._ready_rows(0).shape == (0, COLS, DIM)

    seen.clear()
    _dp_sizes(monkeypatch, (0, 0, 0, 0))
    e._hbm_prepare(torch.ones((0, COLS), dtype=torch.int32))
    assert not seen and e._ready_rows(0).shape == (0, COLS, DIM)


def test_step_hashes_gathers_host_layers_once():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import Engram
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import _StepHashes

    def engram(uses_hbm: bool) -> Engram:
        e = Engram.__new__(Engram)
        torch.nn.Module.__init__(e)
        e._uses_hbm = uses_hbm
        return e

    calls: list[torch.Tensor] = []
    device = torch.arange(4)
    step = _StepHashes(device, lambda h: calls.append(h) or h + 100, eager=False)
    assert step.for_layer(engram(True)) is device
    assert not calls  # no D2H/gather while only HBM layers ran
    host = step.for_layer(engram(False))
    assert torch.equal(host, device + 100)
    assert step.for_layer(engram(False)) is host and len(calls) == 1


def test_step_hashes_eager_gathers_up_front():
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import Engram
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import _StepHashes

    host_layer = Engram.__new__(Engram)
    torch.nn.Module.__init__(host_layer)
    host_layer._uses_hbm = False
    calls: list[torch.Tensor] = []
    device = torch.arange(4)
    step = _StepHashes(device, lambda h: calls.append(h) or h + 100, eager=True)
    assert len(calls) == 1  # gathered before any layer is staged
    host = step.for_layer(host_layer)
    assert torch.equal(host, device + 100)
    assert step.for_layer(host_layer) is host and len(calls) == 1
