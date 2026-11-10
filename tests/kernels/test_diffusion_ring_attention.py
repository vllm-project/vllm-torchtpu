# Copyright 2026 The SGLang Authors.
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
#
# Adapted from sglang-torchtpu test/registered/tpu/unit/test_diffusion_ring_attention.py
# (Apache-2.0).
"""Unit, interpret-mode, and on-device tests for TPU Pallas diffusion ring attention."""

import functools
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

if "--xla_force_host_platform_device_count=" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        f"{os.environ.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count=8"
    ).strip()

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.diffusion_attention import ring_kernel


def _fp32_attention_reference(
    q_bf16: object, k_bf16: object, v_bf16: object, scale: float
) -> np.ndarray:
    qf = np.asarray(q_bf16, dtype=np.float32).transpose(0, 2, 1, 3)
    rep = qf.shape[1] // np.asarray(k_bf16).shape[2]
    kf = np.repeat(np.asarray(k_bf16, dtype=np.float32), rep, axis=2).transpose(
        0, 2, 1, 3
    )
    vf = np.repeat(np.asarray(v_bf16, dtype=np.float32), rep, axis=2).transpose(
        0, 2, 1, 3
    )
    scores = np.matmul(qf, np.swapaxes(kf, -1, -2)) * scale
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.matmul(probs, vf).transpose(0, 2, 1, 3)


def test_tuned_wan22_blocks_match_table() -> None:
    blocks = ring_kernel.Blocks.for_shard(18900, 4, use_fixed_m=True)
    assert (
        blocks.block_q,
        blocks.block_kv,
        blocks.block_kv_compute,
        blocks.block_q_sub,
    ) == (19200, 3328, 3328, 3200)

    online = ring_kernel.Blocks.for_shard(18900, 4, use_fixed_m=False)
    assert (
        online.block_q,
        online.block_kv,
        online.block_kv_compute,
        online.block_q_sub,
    ) == (19200, 2816, 2816, 3840)


def test_untuned_wan22_sweep_shapes_fit_vmem() -> None:
    # All [480p, 720p] x [1..15]s shards at ring=2,4,8 fit 64 MiB VMEM.
    frame_counts = sorted(
        {16 * sec + 1 for sec in range(1, 16)} | {17, 41, 81, 121, 161, 201, 241}
    )
    for ring_size in (2, 4, 8):
        for tokens_per_latent_frame in (390 * 4, 900 * 4):
            for frames in frame_counts:
                latent_frames = (frames - 1) // 4 + 1
                shard_len = (latent_frames * tokens_per_latent_frame) // ring_size
                for use_fixed_m in (True, False):
                    blocks = ring_kernel.Blocks.for_shard(
                        shard_len, ring_size, use_fixed_m=use_fixed_m
                    )
                    q_sub = blocks.block_q_sub or ring_kernel.auto_block_q_sub(
                        blocks.block_q
                    )
                    assert blocks.block_q % q_sub == 0
                    assert q_sub % 128 == 0
                    assert blocks.block_kv % blocks.block_kv_compute == 0
                    ubq, ubkv, ubkvc, uqs = ring_kernel.select_untuned_blocks(
                        shard_len, use_fixed_m=use_fixed_m
                    )
                    if shard_len > 34_000:
                        assert ubq < shard_len
                    single_buf = ring_kernel.is_single_buffered_q(ubq, shard_len)
                    est = ring_kernel.estimate_vmem_bytes(
                        ubq,
                        uqs,
                        ubkv,
                        ubkvc,
                        single_buffered_q=single_buf,
                    )
                    assert est <= ring_kernel.VMEM_LIMIT_BYTES


def test_single_buffered_q_and_head_dim_scaling() -> None:
    assert not ring_kernel.is_single_buffered_q(4608, 4500)
    assert ring_kernel.is_single_buffered_q(5120, 5070)
    assert not ring_kernel.is_single_buffered_q(23040, 45900)
    v64 = ring_kernel.estimate_vmem_bytes(8192, 2048, 2560, head_dim=64)
    v128 = ring_kernel.estimate_vmem_bytes(8192, 2048, 2560, head_dim=128)
    assert v64 < v128
    b64 = ring_kernel.Blocks.for_shard(5070, 4, head_dim=64, use_fixed_m=True)
    assert b64.block_q % (b64.block_q_sub or 128) == 0


def test_constant_row0_and_gate_eligibility() -> None:
    import jax.numpy as jnp

    qn = jnp.array([5.0, 12.0, 10.65], dtype=jnp.float32)
    kn = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float32)
    mk = np.asarray(ring_kernel.ring_fixed_m_operands(qn, kn))
    assert mk.shape == (2, 3)
    np.testing.assert_allclose(mk[0], [106.0, 106.0, 106.0])
    np.testing.assert_allclose(mk[1], [1.0, 0.0, 1.0])


def test_ring4_interpret_multiblock_matches_fp32_reference() -> None:
    import jax

    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 4:
        xla_flags = " ".join(
            tok
            for tok in os.environ.get("XLA_FLAGS", "").split()
            if not tok.startswith("--xla_force_host_platform_device_count=")
        )
        env = {
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": (
                f"{xla_flags} --xla_force_host_platform_device_count=8".strip()
            ),
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                (
                    f"{os.path.abspath(__file__)}"
                    "::test_ring4_interpret_multiblock_matches_fp32_reference"
                ),
                "-q",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        return

    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    # ring_size=4 separates upstream from downstream; shard_len=310 with
    # bq=256, q_sub=256, bkv=128 exercises 3 KV blocks (_run_full + per-block
    # _reload_own + ragged KV tail), 2 Q passes, n_q == 1 hybrid last body,
    # batch=2, and GQA 4:2 with mixed fixed-m / online heads.
    rng = np.random.default_rng(0)
    ring_size = 4
    shard_len = 310
    batch = 2
    seq_len = ring_size * shard_len
    head_dim = 128
    scale = head_dim**-0.5
    pinned = {
        "block_q": 256,
        "block_kv": 128,
        "block_kv_compute": 128,
        "block_q_sub": 256,
    }

    q_np = rng.standard_normal((batch, seq_len, 4, head_dim), dtype=np.float32)
    q_np[:, :, 3, :] *= 8.0
    k_np = rng.standard_normal((batch, seq_len, 2, head_dim), dtype=np.float32)
    v_np = rng.standard_normal((batch, seq_len, 2, head_dim), dtype=np.float32)

    q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
    k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
    ref = _fp32_attention_reference(q_bf16, k_bf16, v_bf16, scale)

    axis_names = ("batch", "ring", "uly")
    mesh = Mesh(
        np.asarray(cpu_devices[:ring_size]).reshape(1, ring_size, 1), axis_names
    )
    spec = PartitionSpec(*axis_names)
    sharding = NamedSharding(mesh, spec)

    def _put(x: object, heads: int) -> object:
        reshaped = jnp.swapaxes(
            jnp.asarray(x).reshape(batch, ring_size, shard_len, heads, head_dim), 0, 1
        )
        return jax.device_put(reshaped[None, :, None, :, :, :, :], sharding)

    q_in, k_in, v_in = _put(q_bf16, 4), _put(k_bf16, 2), _put(v_bf16, 2)

    for use_fixed_m in (True, False):
        pltpu.reset_tpu_interpret_mode_state()
        fn = jax.jit(
            jax.shard_map(
                functools.partial(
                    ring_kernel.ring_attention_sharded,
                    ring_axis="ring",
                    axis_names=axis_names,
                    scale=scale,
                    use_fixed_m=use_fixed_m,
                    pinned=pinned,
                    interpret=pltpu.InterpretParams(detect_races=True),
                ),
                mesh=mesh,
                in_specs=(spec, spec, spec),
                out_specs=spec,
                check_vma=False,
            )
        )
        out_sharded = np.asarray(fn(q_in, k_in, v_in), dtype=np.float32)[0, :, 0]
        out = np.swapaxes(out_sharded, 0, 1).reshape(batch, seq_len, 4, head_dim)
        assert not (ipc.races is not None and ipc.races.races_found), (
            f"Interpret detected DMA/semaphore race (use_fixed_m={use_fixed_m})"
        )
        assert np.all(np.isfinite(out))
        rel_l2 = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        assert rel_l2 < 7e-3


def test_omni_ring_dispatch_and_env_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("vllm_omni")
    from vllm_torchtpu.omni import attention as omni_attn

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    calls: list[tuple[str, tuple[object, ...]]] = []
    # Cases 1-6 run as TPU v7; case 7 covers the other side of the gate.
    monkeypatch.setattr(omni_attn, "_ici_ring_kernel_hw_ok", None)
    monkeypatch.setattr(
        omni_attn.hardware, "get_tpu_version", lambda: omni_attn.hardware.TpuVersion.V7
    )
    assert omni_attn._ici_ring_kernel_hw_supported() is True

    monkeypatch.setattr(
        omni_attn,
        "_pallas_ring_attention_op",
        lambda outer, ring, uly, fuse, scale, fm: (
            lambda q, k, v: (
                calls.append(
                    ("ici_ring", (outer, ring, uly, fuse, round(scale, 6), fm))
                )
                or (q + 1)
            )
        ),
    )
    monkeypatch.setattr(
        omni_attn,
        "_short_2d_ring_attention_op",
        lambda outer, ring, scale: (
            lambda q, k, v: (
                calls.append(("short_2d", (outer, ring, round(scale, 6)))) or (q + 2)
            )
        ),
    )
    monkeypatch.setattr(
        omni_attn,
        "_ring_ulysses_attention_op",
        lambda outer, ring, uly, fuse: (
            lambda q, k, v, scale: (
                calls.append(("fallback_ppermute", (outer, ring, uly, fuse))) or (q + 3)
            )
        ),
    )

    # 1. Long shard (q_seq=4290 > 4096) at c2r4 uses ICI ring op (fixed_m=True).
    q_long = torch.zeros((1, 4290, 4, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    k_long = torch.zeros((1, 4290, 2, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    out = omni_attn.tpu_ring_flash_attention(
        q_long, k_long, k_long, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    assert tuple(out.shape) == (1, 4290, 4, 128)
    assert torch.all(out == 1)
    assert calls == [("ici_ring", (2, 4, 1, False, 0.125, True))]

    # 2. VLLM_TORCHTPU_OMNI_RING_FIXED_M=0 switches use_fixed_m to False.
    calls.clear()
    with patch.dict(os.environ, {"VLLM_TORCHTPU_OMNI_RING_FIXED_M": "0"}):
        out_fm0 = omni_attn.tpu_ring_flash_attention(
            q_long, k_long, k_long, 0.125, ring_size=4, uly_size=1, outer_size=2
        )
    assert torch.all(out_fm0 == 1)
    assert calls == [("ici_ring", (2, 4, 1, False, 0.125, False))]

    # 3. Short shard (q_seq=1950 <= 4096, ring=4, uly=1) uses short 2D ring op.
    calls.clear()
    q_short = torch.zeros((1, 1950, 4, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    out_short = omni_attn.tpu_ring_flash_attention(
        q_short, q_short, q_short, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    assert torch.all(out_short == 2)
    assert calls == [("short_2d", (2, 4, 0.125))]

    # 4. Odd head count on short shard stays on _pallas_ring_attention_op.
    calls.clear()
    q_odd = torch.zeros((1, 1950, 3, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    k_odd = torch.zeros((1, 1950, 1, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    out_odd = omni_attn.tpu_ring_flash_attention(
        q_odd, k_odd, k_odd, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    assert torch.all(out_odd == 1)
    assert calls == [("ici_ring", (2, 4, 1, False, 0.125, True))]

    # 5. VLLM_TORCHTPU_OMNI_RING_KERNEL=0 opts out to fallback ppermute ring.
    calls.clear()
    with patch.dict(os.environ, {"VLLM_TORCHTPU_OMNI_RING_KERNEL": "0"}):
        out_fb = omni_attn.tpu_ring_flash_attention(
            q_long, k_long, k_long, 0.125, ring_size=4, uly_size=1, outer_size=2
        )
        out_short_fb = omni_attn.tpu_ring_flash_attention(
            q_short, q_short, q_short, 0.125, ring_size=4, uly_size=1, outer_size=2
        )
    assert torch.all(out_fb == 3)
    assert torch.all(out_short_fb == 3)
    assert calls == [
        ("fallback_ppermute", (2, 4, 1, False)),
        ("fallback_ppermute", (2, 4, 1, False)),
    ]

    # 6. Very short shard (< 256 rows) falls back cleanly.
    calls.clear()
    q_tiny = torch.zeros((1, 128, 4, 128), dtype=torch.bfloat16).as_subclass(
        _FakeTpuTensor
    )
    out_tiny = omni_attn.tpu_ring_flash_attention(
        q_tiny, q_tiny, q_tiny, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    assert torch.all(out_tiny == 3)
    assert calls == [("fallback_ppermute", (2, 4, 1, False))]

    # 7. Other TPU generations (and hosts without a known TPU) use the ppermute
    # ring path; the generation is probed once and cached.
    calls.clear()
    probes: list[str] = []

    def _v6e():
        probes.append("v6e")
        return omni_attn.hardware.TpuVersion.V6E

    monkeypatch.setattr(omni_attn, "_ici_ring_kernel_hw_ok", None)
    monkeypatch.setattr(omni_attn.hardware, "get_tpu_version", _v6e)
    out_v6e = omni_attn.tpu_ring_flash_attention(
        q_long, k_long, k_long, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    out_v6e_short = omni_attn.tpu_ring_flash_attention(
        q_short, q_short, q_short, 0.125, ring_size=4, uly_size=1, outer_size=2
    )
    assert torch.all(out_v6e == 3)
    assert torch.all(out_v6e_short == 3)
    assert calls == [("fallback_ppermute", (2, 4, 1, False))] * 2
    assert probes == ["v6e"]

    monkeypatch.setattr(omni_attn, "_ici_ring_kernel_hw_ok", None)
    monkeypatch.setattr(
        omni_attn.hardware,
        "get_tpu_version",
        lambda: omni_attn.hardware.TpuVersion.UNKNOWN,
    )
    assert omni_attn._ici_ring_kernel_hw_supported() is False


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "ring_size,shard_len,block_kv",
    [
        (2, 390, None),
        (4, 390, None),
        (4, 4290, None),
        # block_kv=1024 splits each hop into 5 KV blocks with a ragged tail, so
        # the multi-block DMA pipeline (_reload_own, credit semaphores) runs.
        (4, 4290, 1024),
        # Tuned Wan 2.2 shard: several KV blocks per hop, ragged tail.
        (4, 18900, None),
    ],
)
def test_sharded_ring_on_device_matches_fp32_reference(
    ring_size: int, shard_len: int, block_kv: int | None
) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if jax.default_backend() != "tpu" or len(jax.devices()) < ring_size:
        pytest.skip(f"Requires TPU with >= {ring_size} TensorCores")

    rng = np.random.default_rng(0)
    seq_len = ring_size * shard_len
    head_dim = 128
    scale = head_dim**-0.5

    q_np = rng.standard_normal((1, seq_len, 4, head_dim), dtype=np.float32)
    # Head 3 exceeds FIXED_M_RING_SAFE_BOUND to exercise hybrid fallback.
    q_np[:, :, 3, :] *= 8.0
    k_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)
    v_np = rng.standard_normal((1, seq_len, 2, head_dim), dtype=np.float32)

    q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
    k_bf16 = jnp.asarray(k_np, dtype=jnp.bfloat16)
    v_bf16 = jnp.asarray(v_np, dtype=jnp.bfloat16)
    # A full fp32 (seq x seq) score matrix does not fit host memory at 75600
    # rows, so long cases check a fixed sample of query rows.
    rows = (
        np.arange(seq_len)
        if seq_len <= 20_000
        else np.sort(rng.choice(seq_len, size=2048, replace=False))
    )
    ref = _fp32_attention_reference(q_bf16[:, rows], k_bf16, v_bf16, scale)
    pinned = None if block_kv is None else {"block_kv": block_kv}

    axis_names = ("batch", "ring", "uly")
    mesh = Mesh(
        np.asarray(jax.devices()[:ring_size]).reshape(1, ring_size, 1), axis_names
    )
    spec = PartitionSpec(*axis_names)
    sharding = NamedSharding(mesh, spec)

    q_in = jax.device_put(
        q_bf16.reshape(1, ring_size, shard_len, 4, head_dim)[:, :, None, None, :, :, :],
        sharding,
    )
    k_in = jax.device_put(
        k_bf16.reshape(1, ring_size, shard_len, 2, head_dim)[:, :, None, None, :, :, :],
        sharding,
    )
    v_in = jax.device_put(
        v_bf16.reshape(1, ring_size, shard_len, 2, head_dim)[:, :, None, None, :, :, :],
        sharding,
    )

    for use_fixed_m in (True, False):
        fn = jax.jit(
            jax.shard_map(
                functools.partial(
                    ring_kernel.ring_attention_sharded,
                    ring_axis="ring",
                    axis_names=axis_names,
                    scale=scale,
                    use_fixed_m=use_fixed_m,
                    pinned=pinned,
                ),
                mesh=mesh,
                in_specs=(spec, spec, spec),
                out_specs=spec,
                check_vma=False,
            )
        )
        if pinned is not None or shard_len == 18900:
            blocks = ring_kernel.Blocks.for_shard(
                shard_len, ring_size, pinned, use_fixed_m=use_fixed_m
            )
            assert blocks.block_kv < shard_len
        out = np.asarray(fn(q_in, k_in, v_in), dtype=np.float32)[0, :, 0, 0].reshape(
            1, seq_len, 4, head_dim
        )[:, rows]
        assert np.all(np.isfinite(out))
        rel_l2 = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        assert rel_l2 < 7e-3
