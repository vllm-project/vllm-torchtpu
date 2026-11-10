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
"""CPU/gloo unit tests for the Wan VAE spatial-shard all-gather halo patch."""

import datetime
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu.omni import patches

pytestmark = pytest.mark.omni

# Compact Wan VAE architecture that exercises all 3D causal convolutions, 2D
TINY_CONFIG = dict(
    base_dim=8,
    decoder_base_dim=8,
    z_dim=8,
    dim_mult=[1, 2, 2, 2],
    num_res_blocks=1,
    temperal_downsample=[False, True, True],
    is_residual=True,
    patch_size=2,
    in_channels=12,
    out_channels=12,
    scale_factor_temporal=4,
    scale_factor_spatial=16,
)


def _strip_broken_cv2_sys_path() -> None:
    """Remove any site-packages/cv2 entry left on sys.path by a failed cv2 import.

    Non-headless opencv-python inserts its package directory onto sys.path
    before loading libGL.so.1; if that shared library is missing on a CPU-only
    host, the leftover directory shadows stdlib ``typing`` with ``cv2/typing``
    in child processes started by ``torch.multiprocessing.spawn``.
    """
    sys.path[:] = [p for p in sys.path if not p.rstrip("/").endswith("/cv2")]


@pytest.fixture(scope="module")
def wan_vae_modules():
    """Import the real vLLM-Omni Wan VAE modules."""
    wan_spatial_shard = pytest.importorskip(
        "vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard"
    )
    autoencoder_kl_wan = pytest.importorskip(
        "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan"
    )
    _strip_broken_cv2_sys_path()
    return SimpleNamespace(
        wan_spatial_shard=wan_spatial_shard,
        autoencoder_kl_wan=autoencoder_kl_wan,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_patch_idempotent_and_kill_switch_on_real_omni(wan_vae_modules, monkeypatch):
    wan_spatial_shard = wan_vae_modules.wan_spatial_shard
    orig_fn = getattr(
        wan_spatial_shard,
        "_tpu_orig_halo_exchange_p2p",
        wan_spatial_shard._halo_exchange_p2p,
    )
    orig_ag = getattr(
        wan_spatial_shard,
        "_tpu_orig_all_gather_along_dim",
        wan_spatial_shard.all_gather_along_dim,
    )
    monkeypatch.setattr(wan_spatial_shard, "_halo_exchange_p2p", orig_fn)
    monkeypatch.setattr(wan_spatial_shard, "all_gather_along_dim", orig_ag)
    monkeypatch.delattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p", raising=False)
    monkeypatch.delattr(
        wan_spatial_shard, "_tpu_orig_all_gather_along_dim", raising=False
    )

    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "0")
    patches._patch_wan_vae_halo_allgather()
    assert wan_spatial_shard._halo_exchange_p2p is orig_fn
    assert not hasattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p")

    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    patches._patch_wan_vae_halo_allgather()
    patched_fn = wan_spatial_shard._halo_exchange_p2p
    assert patched_fn is not orig_fn
    assert wan_spatial_shard._tpu_orig_halo_exchange_p2p is orig_fn

    # Idempotent: second call preserves both the sentinel and the wrapper.
    patches._patch_wan_vae_halo_allgather()
    assert wan_spatial_shard._halo_exchange_p2p is patched_fn
    assert wan_spatial_shard._tpu_orig_halo_exchange_p2p is orig_fn

    # Single-rank call zeroes both receive buffers without any collective.
    def _forbid_collective(*args, **kwargs):
        raise AssertionError("No collective should run for world_size=1")

    monkeypatch.setattr(wan_spatial_shard.dist, "all_gather", _forbid_collective)
    monkeypatch.setattr(wan_spatial_shard.dist, "batch_isend_irecv", _forbid_collective)
    top_ref = torch.ones(1, 2, 3, 4)
    bottom_ref = torch.ones(1, 2, 3, 4)
    recv_top = torch.full((1, 2, 3, 4), 5.0)
    recv_bottom = torch.full((1, 2, 3, 4), 7.0)
    wan_spatial_shard._halo_exchange_p2p(
        rank=0,
        world_size=1,
        group=object(),
        top_row_ref=top_ref,
        bottom_row_ref=bottom_ref,
        recv_top_buf=recv_top,
        recv_bottom_buf=recv_bottom,
    )
    assert torch.equal(recv_top, torch.zeros_like(recv_top))
    assert torch.equal(recv_bottom, torch.zeros_like(recv_bottom))


def _record_warnings(monkeypatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(
        patches.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )
    return warnings


def _drifted_halo_exchange_p2p(
    *, rank, world_size, group, top_ref, bottom_ref, recv_top_buf, recv_bottom_buf
):
    raise AssertionError("never called")


@pytest.mark.parametrize("drift", ["renamed_keyword", "missing_symbol"])
def test_patch_warns_and_skips_on_upstream_drift(wan_vae_modules, monkeypatch, drift):
    wan_spatial_shard = wan_vae_modules.wan_spatial_shard
    monkeypatch.delattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p", raising=False)
    if drift == "renamed_keyword":
        monkeypatch.setattr(
            wan_spatial_shard, "_halo_exchange_p2p", _drifted_halo_exchange_p2p
        )
    else:
        monkeypatch.delattr(wan_spatial_shard, "_halo_exchange_p2p")
    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    before = getattr(wan_spatial_shard, "_halo_exchange_p2p", None)
    warnings = _record_warnings(monkeypatch)

    patches._patch_wan_vae_halo_allgather()
    expected = "update vllm_torchtpu.omni.patches._patch_wan_vae_halo_allgather"
    assert len(warnings) == 1 and expected in warnings[0]
    assert getattr(wan_spatial_shard, "_halo_exchange_p2p", None) is before
    assert not hasattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p")


def _check_halo_bit_for_bit(rank: int, world_size: int, group) -> None:
    from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard

    orig_p2p = getattr(
        wan_spatial_shard,
        "_tpu_orig_halo_exchange_p2p",
        wan_spatial_shard._halo_exchange_p2p,
    )
    shapes = (
        (2, 4, 7, 9),
        (1, 3, 2, 7, 9),
    )

    for split_dim in ("height", "width"):
        dim = wan_spatial_shard._spatial_dim(split_dim)
        for shape in shapes:
            for halo_size in (1, 2):
                gen = torch.Generator().manual_seed(
                    1000 * world_size + 100 * rank + 10 * len(shape) + halo_size
                )
                x = torch.randn(shape, generator=gen, dtype=torch.float32)

                # 1. Run upstream p2p halo exchange on gloo as the reference.
                wan_spatial_shard._halo_exchange_p2p = orig_p2p
                if hasattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p"):
                    delattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p")

                ref_top_init = torch.full_like(
                    wan_spatial_shard._narrow_along_dim(x, dim, 0, halo_size),
                    -123.0,
                ).contiguous()
                ref_bottom_init = torch.full_like(
                    wan_spatial_shard._narrow_along_dim(
                        x, dim, x.shape[dim] - halo_size, halo_size
                    ),
                    -456.0,
                ).contiguous()
                ref_out, ref_top, ref_bottom = wan_spatial_shard.halo_exchange(
                    x,
                    group=group,
                    halo_size=halo_size,
                    split_dim=split_dim,
                    recv_top_buf=ref_top_init.clone(),
                    recv_bottom_buf=ref_bottom_init.clone(),
                )

                # 2. Install the TPU all-gather patch (and verify idempotency),
                patches._patch_wan_vae_halo_allgather()
                patched_fn = wan_spatial_shard._halo_exchange_p2p
                patches._patch_wan_vae_halo_allgather()
                assert wan_spatial_shard._halo_exchange_p2p is patched_fn

                def _forbid_p2p(*args, **kwargs):
                    raise RuntimeError("P2P must not be called under all-gather patch")

                with (
                    patch.object(
                        wan_spatial_shard.dist, "batch_isend_irecv", _forbid_p2p
                    ),
                    patch.object(wan_spatial_shard.dist, "P2POp", _forbid_p2p),
                ):
                    got_out, got_top, got_bottom = wan_spatial_shard.halo_exchange(
                        x,
                        group=group,
                        halo_size=halo_size,
                        split_dim=split_dim,
                        recv_top_buf=ref_top_init.clone(),
                        recv_bottom_buf=ref_bottom_init.clone(),
                    )
                    assert torch.equal(got_top, ref_top)
                    assert torch.equal(got_bottom, ref_bottom)
                    assert torch.equal(got_out, ref_out)

                    # Also verify a direct _halo_exchange_p2p call with
                    token = wan_spatial_shard._SPATIAL_SHARD_CONTEXT.set(
                        wan_spatial_shard.SpatialShardContext(
                            input_extent=x.shape[dim] * world_size,
                            local_input_extent=x.shape[dim],
                            split_dim=split_dim,
                            rank=rank,
                            world_size=world_size,
                        )
                    )
                    try:
                        ctx_top = ref_top_init.clone()
                        ctx_bottom = ref_bottom_init.clone()
                        wan_spatial_shard._halo_exchange_p2p(
                            rank=rank,
                            world_size=world_size,
                            group=group,
                            top_row_ref=wan_spatial_shard._narrow_along_dim(
                                x, dim, 0, halo_size
                            ),
                            bottom_row_ref=wan_spatial_shard._narrow_along_dim(
                                x, dim, x.shape[dim] - halo_size, halo_size
                            ),
                            recv_top_buf=ctx_top,
                            recv_bottom_buf=ctx_bottom,
                        )
                    finally:
                        wan_spatial_shard._SPATIAL_SHARD_CONTEXT.reset(token)
                    assert torch.equal(ctx_top, ref_top)
                    assert torch.equal(ctx_bottom, ref_bottom)


def _check_tiny_wan_decoder(rank: int, world_size: int, group) -> None:
    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
        wan_spatial_shard,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
        OmniAutoencoderKLWan,
    )

    patches._patch_wan_vae_halo_allgather()
    patched_halo_fn = wan_spatial_shard._halo_exchange_p2p

    torch.manual_seed(0)
    ref_vae = OmniAutoencoderKLWan(**TINY_CONFIG).eval()
    ref_state = ref_vae.state_dict()

    # Spatial extent 5 is uneven across both 2 ranks (3 + 2, padded to 6)
    gen = torch.Generator().manual_seed(42)
    z = torch.randn(
        1, TINY_CONFIG["z_dim"], 2, 5, 5, generator=gen, dtype=torch.float32
    )

    with torch.no_grad():
        expected = ref_vae.decode(z, return_dict=False)[0]

    def _forbid_p2p(*args, **kwargs):
        raise RuntimeError("P2P must not be called during Wan spatial-shard decode")

    fake_world_group = SimpleNamespace(device_group=group, cpu_group=group)
    for split_dim in ("height", "width"):
        vae = DistributedAutoencoderKLWan(**TINY_CONFIG).eval()
        vae.load_state_dict(ref_state)
        with patch.object(
            distributed_vae_executor,
            "get_world_group",
            return_value=fake_world_group,
        ):
            vae.init_distributed()
        vae.use_tiling = True
        # Lower tiling threshold so small unit-test latents enter
        vae.tile_sample_min_height = 16
        vae.tile_sample_min_width = 16
        vae.set_parallel_size(world_size, mode=f"spatial_shard_{split_dim}")

        halo_calls = 0

        def _counted_halo(**kwargs):
            nonlocal halo_calls
            halo_calls += 1
            return patched_halo_fn(**kwargs)

        with (
            patch.object(wan_spatial_shard.dist, "batch_isend_irecv", _forbid_p2p),
            patch.object(wan_spatial_shard.dist, "P2POp", _forbid_p2p),
            patch.object(wan_spatial_shard, "_halo_exchange_p2p", _counted_halo),
            torch.no_grad(),
        ):
            sharded = vae.decode(z, return_dict=False)[0]

        assert halo_calls > 0
        if rank == 0:
            assert sharded.shape == expected.shape
            max_abs = (sharded - expected).abs().max().item()
            assert max_abs <= 1e-5, (
                f"world_size={world_size} split_dim={split_dim} "
                f"max_abs={max_abs} > 1e-5"
            )
        else:
            assert sharded.numel() == 0


def _gloo_worker(rank: int, world_size: int, port: int) -> None:
    _strip_broken_cv2_sys_path()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        _check_halo_bit_for_bit(rank, world_size, dist.group.WORLD)
        if world_size in (2, 3):
            _check_tiny_wan_decoder(rank, world_size, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_wan_vae_halo_allgather_on_gloo_ranks(wan_vae_modules, world_size: int):
    _strip_broken_cv2_sys_path()
    mp.spawn(
        _gloo_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


# Wan 2.1 VAE decoder topology (the VAE of Wan 2.2 T2V-A14B): 16 latent
WAN21_SMALL_CONFIG = dict(
    base_dim=4,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True],
)
# Wan 2.2 720x1280 latent: 90 rows do not divide over 8 ranks (padded to 96).
WAN_720P_LATENT_SHAPE = (1, 16, 2, 90, 160)


def _check_wan21_720p_on_8_ranks(
    rank: int, world_size: int, device_group, cpu_group
) -> None:
    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
    )
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
        DistributedAutoencoderKLWan,
        OmniAutoencoderKLWan,
    )

    # Fresh spawned process: enable the gate; the patches are idempotent.
    os.environ["VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER"] = "1"
    patches._patch_wan_vae_halo_allgather()
    patches._patch_distributed_vae_host_metadata_collectives()

    torch.manual_seed(0)
    ref_vae = OmniAutoencoderKLWan(**WAN21_SMALL_CONFIG).eval()
    ref_state = ref_vae.state_dict()
    z = torch.randn(
        WAN_720P_LATENT_SHAPE,
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float32,
    )
    expected = {}
    if rank == 0:
        with torch.no_grad():
            expected["full"] = ref_vae.decode(z, return_dict=False)[0]
            ref_vae.use_tiling = True
            expected["tile"] = ref_vae.decode(z, return_dict=False)[0]

    orig_all_gather = dist.all_gather
    orig_all_reduce = dist.all_reduce
    fake_world_group = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)

    def _forbid_p2p(*args, **kwargs):
        raise RuntimeError("P2P must not be called during distributed decode")

    calls: list[tuple[str, str, tuple[int, ...], str]] = []
    for mode in ("spatial_shard_height", "spatial_shard_width", "tile"):
        vae = DistributedAutoencoderKLWan(**WAN21_SMALL_CONFIG).eval()
        vae.load_state_dict(ref_state)
        with patch.object(
            distributed_vae_executor,
            "get_world_group",
            return_value=fake_world_group,
        ):
            vae.init_distributed()
        vae.use_tiling = True
        vae.set_parallel_size(world_size, mode=mode)

        calls.clear()

        def _logged_all_gather(tensor_list, tensor, group=None, async_op=False):
            on_device = group is device_group
            calls.append(
                (
                    "all_gather",
                    "device" if on_device else "cpu",
                    tuple(tensor.shape),
                    str(tensor.dtype),
                )
            )
            if on_device and not tensor.is_floating_point():
                raise RuntimeError(
                    f"integer all_gather on the device group: {tensor.dtype}"
                )
            return orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)

        def _logged_all_reduce(
            tensor, op=dist.ReduceOp.SUM, group=None, async_op=False
        ):
            calls.append(
                (
                    "all_reduce",
                    "device" if group is device_group else "cpu",
                    tuple(tensor.shape),
                    str(tensor.dtype),
                )
            )
            if group is not cpu_group:
                raise RuntimeError(
                    "all_reduce (ReduceOp.MAX) must run on the CPU group"
                )
            return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

        with (
            patch.object(dist, "all_gather", _logged_all_gather),
            patch.object(dist, "all_reduce", _logged_all_reduce),
            patch.object(dist, "batch_isend_irecv", _forbid_p2p),
            torch.no_grad(),
        ):
            out = vae.decode(z, return_dict=False)[0]

        # Every rank must issue the same collectives with the same shapes in
        all_calls = [None] * world_size
        dist.all_gather_object(all_calls, calls, group=cpu_group)
        for other_rank, other_calls in enumerate(all_calls):
            assert other_calls == all_calls[0], (
                f"mode={mode}: rank {other_rank} collective sequence differs "
                f"from rank 0 ({len(other_calls)} vs {len(all_calls[0])} calls)"
            )
        device_gathers = [c for c in calls if c[:2] == ("all_gather", "device")]
        assert device_gathers, f"mode={mode}: no device-group all_gather"

        if rank == 0:
            ref = expected["tile" if mode == "tile" else "full"]
            assert out.shape == ref.shape == (1, 3, 5, 720, 1280), (
                mode,
                out.shape,
                ref.shape,
            )
            max_abs = (out - ref).abs().max().item()
            print(
                f"[wan21-720p-8rank] mode={mode} collectives={len(calls)} "
                f"distinct={len(set(calls))} device_all_gathers="
                f"{len(device_gathers)} max_abs={max_abs:.3e}",
                flush=True,
            )
            # Width shards change the innermost conv extent, so CPU conv
            tol = 1e-4 if mode == "spatial_shard_width" else 1e-5
            assert max_abs <= tol, f"mode={mode} max_abs={max_abs} > {tol}"
        else:
            assert out.numel() == 0


def _gloo_worker_8rank_720p(rank: int, world_size: int, port: int) -> None:
    _strip_broken_cv2_sys_path()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(1)

    timeout = datetime.timedelta(seconds=600)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)
    try:
        cpu_group = dist.new_group(backend="gloo", timeout=timeout)
        _check_wan21_720p_on_8_ranks(rank, world_size, dist.group.WORLD, cpu_group)
    finally:
        dist.destroy_process_group()


def test_wan21_vae_720p_latent_on_8_gloo_ranks(wan_vae_modules):
    """Wan 2.2 720x1280 geometry (latent 90x160) on 8 ranks: spatial height
    (90 -> 8 x 12, padded), spatial width (160 -> 8 x 20) and tile mode."""
    _strip_broken_cv2_sys_path()
    mp.spawn(
        _gloo_worker_8rank_720p,
        args=(8, _free_port()),
        nprocs=8,
        join=True,
    )


def test_all_gather_along_dim_and_tile_gather_sync_on_tpu_device(
    wan_vae_modules, monkeypatch
):
    from contextlib import nullcontext

    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
    )

    wan_spatial_shard = wan_vae_modules.wan_spatial_shard
    monkeypatch.delattr(wan_spatial_shard, "_tpu_orig_halo_exchange_p2p", raising=False)
    monkeypatch.delattr(
        wan_spatial_shard, "_tpu_orig_all_gather_along_dim", raising=False
    )
    monkeypatch.delattr(
        distributed_vae_executor.DistributedVaeExecutor,
        "_tpu_orig_compute_global_padding_shape",
        raising=False,
    )
    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    patches._patch_wan_vae_halo_allgather()
    patches._patch_distributed_vae_host_metadata_collectives()

    sync_calls = []
    monkeypatch.setattr(
        patches,
        "synchronize_tensors",
        lambda tensors=None, wait=True: sync_calls.append((tensors, wait)),
    )
    monkeypatch.setattr(wan_spatial_shard, "_rank_world", lambda _g: (1, 2))
    monkeypatch.setattr(
        wan_spatial_shard.dist,
        "all_gather",
        lambda out_list, inp, group=None: [t.copy_(inp) for t in out_list],
    )

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="tpu")

    x = torch.ones((1, 2, 1, 4, 4)).as_subclass(_FakeTpuTensor)
    out_non_dst = wan_spatial_shard.all_gather_along_dim(
        x, group=object(), dim=3, dst=0
    )
    assert out_non_dst.numel() == 0
    assert len(sync_calls) == 1
    assert sync_calls[0] == (None, False)

    # dst=None (attention gather) does not force a chunk boundary sync.
    sync_calls.clear()
    out_all = wan_spatial_shard.all_gather_along_dim(x, group=object(), dim=3, dst=None)
    assert out_all.shape == (1, 2, 1, 8, 4)
    assert len(sync_calls) == 0

    # Tile-parallel float gather on TPU synchronizes gather_list on all ranks.
    monkeypatch.setattr(
        distributed_vae_executor.dist,
        "all_gather",
        lambda out_list, inp, group=None: [t.copy_(inp) for t in out_list],
    )
    fake_exec = SimpleNamespace(
        world_size=2,
        rank=1,
        group=object(),
        _error_group=object(),
        _sync_errors=lambda _stage: nullcontext(),
    )
    res = distributed_vae_executor.DistributedVaeExecutor.gather_tensors(fake_exec, x)
    assert res is None
    assert len(sync_calls) == 1
    assert len(sync_calls[0][0]) == 2
    assert sync_calls[0][1] is False


def test_sync_final_result_broadcasts_shape_on_host_group(wan_vae_modules, monkeypatch):
    from contextlib import nullcontext

    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
    )

    executor_cls = distributed_vae_executor.DistributedVaeExecutor
    monkeypatch.delattr(
        executor_cls, "_tpu_orig_compute_global_padding_shape", raising=False
    )
    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    patches._patch_distributed_vae_host_metadata_collectives()
    assert hasattr(executor_cls, "_tpu_orig_sync_final_result")

    sync_calls = []
    monkeypatch.setattr(
        patches,
        "synchronize_tensors",
        lambda tensors=None, wait=True: sync_calls.append((tensors, wait)),
    )
    device_group, cpu_group = object(), object()
    calls = []

    def _fake_broadcast(tensor, src=0, group=None):
        calls.append((group, tensor.dtype, tensor.device.type))
        if group is cpu_group:
            tensor.copy_(torch.tensor([1, 16, 2, 4, 4]))
        else:
            tensor.fill_(3.0)

    monkeypatch.setattr(distributed_vae_executor.dist, "broadcast", _fake_broadcast)

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="tpu")

    monkeypatch.setattr(
        torch,
        "empty",
        lambda shape, device=None, dtype=None: torch.zeros(
            shape, dtype=dtype
        ).as_subclass(_FakeTpuTensor),
    )
    fake_exec = SimpleNamespace(
        world_size=2,
        rank=1,
        group=device_group,
        _error_group=cpu_group,
        _sync_errors=lambda _stage: nullcontext(),
    )
    out = executor_cls._sync_final_result(
        fake_exec, torch.empty(0), 5, "tpu", torch.float32
    )
    assert calls[0] == (cpu_group, torch.int64, "cpu")
    assert calls[1][0] is device_group and calls[1][1] == torch.float32
    assert tuple(out.shape) == (1, 16, 2, 4, 4)
    assert len(sync_calls) == 1 and sync_calls[0][0][0] is out
    assert sync_calls[0][1] is False


def test_host_metadata_patch_skips_without_sync_errors(wan_vae_modules, monkeypatch):
    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
    )

    class _ExecutorWithoutSyncErrors:
        def _compute_global_padding_shape(self, local_results, output_ndim, device):
            return [0] * (output_ndim + 1)

        def gather_tensors(self, tensor):
            return None

    executor_cls = _ExecutorWithoutSyncErrors
    orig_compute = executor_cls._compute_global_padding_shape
    orig_gather = executor_cls.gather_tensors
    monkeypatch.setattr(
        distributed_vae_executor, "DistributedVaeExecutor", executor_cls
    )
    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    warnings = _record_warnings(monkeypatch)

    patches._patch_distributed_vae_host_metadata_collectives()
    assert len(warnings) == 1 and "_sync_errors" in warnings[0]
    assert executor_cls._compute_global_padding_shape is orig_compute
    assert executor_cls.gather_tensors is orig_gather
    assert not hasattr(executor_cls, "_tpu_orig_compute_global_padding_shape")


def test_host_metadata_patch_falls_back_without_error_group(
    wan_vae_modules, monkeypatch
):
    from contextlib import nullcontext

    from vllm_omni.diffusion.distributed.autoencoders import (
        distributed_vae_executor,
    )

    calls = []

    class _ExecutorWithoutErrorGroup:
        def _sync_errors(self, stage):
            return nullcontext()

        def _compute_global_padding_shape(self, local_results, output_ndim, device):
            calls.append("compute")
            return [0] * (output_ndim + 1)

        def gather_tensors(self, tensor):
            calls.append("gather")

        def _sync_final_result(
            self, rank0_result, output_ndim, output_device, output_dtype
        ):
            calls.append("sync")
            return rank0_result

    def _forbid_collective(*args, **kwargs):
        raise AssertionError("the upstream method must run the collective")

    for name in ("all_reduce", "all_gather", "broadcast"):
        monkeypatch.setattr(distributed_vae_executor.dist, name, _forbid_collective)
    monkeypatch.setattr(
        distributed_vae_executor,
        "DistributedVaeExecutor",
        _ExecutorWithoutErrorGroup,
    )
    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    warnings = _record_warnings(monkeypatch)

    patches._patch_distributed_vae_host_metadata_collectives()
    assert hasattr(_ExecutorWithoutErrorGroup, "_tpu_orig_compute_global_padding_shape")
    executor = _ExecutorWithoutErrorGroup()
    result = torch.zeros(2)
    assert executor._compute_global_padding_shape([], 5, "cpu") == [0] * 6
    assert executor.gather_tensors(torch.zeros(2, dtype=torch.int64)) is None
    assert executor.gather_tensors(result) is None
    assert executor._sync_final_result(result, 1, "cpu", torch.float32) is result
    assert calls == ["compute", "gather", "gather", "sync"]
    assert len(warnings) == 1 and "_error_group" in warnings[0]


@pytest.mark.parametrize("num_frames", [1, 5])
def test_wan_vae_replicated_encode_skips_distributed_executor(
    wan_vae_modules, monkeypatch, num_frames: int
):
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    autoencoder_kl_wan = wan_vae_modules.autoencoder_kl_wan
    cls = autoencoder_kl_wan.DistributedAutoencoderKLWan
    orig_encode = getattr(cls, "_tpu_orig_encode", cls._encode)
    orig_tiled_encode = getattr(cls, "_tpu_orig_tiled_encode", cls.tiled_encode)
    monkeypatch.setattr(cls, "_encode", orig_encode)
    monkeypatch.setattr(cls, "tiled_encode", orig_tiled_encode)
    monkeypatch.delattr(cls, "_tpu_orig_encode", raising=False)
    monkeypatch.delattr(cls, "_tpu_orig_tiled_encode", raising=False)

    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "0")
    patches._patch_wan_vae_replicated_encode()
    assert cls._encode is orig_encode
    assert cls.tiled_encode is orig_tiled_encode
    assert not hasattr(cls, "_tpu_orig_encode")
    assert not hasattr(cls, "_tpu_orig_tiled_encode")

    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    patches._patch_wan_vae_replicated_encode()
    patched_encode = cls._encode
    patched_tiled = cls.tiled_encode
    assert patched_encode is not orig_encode
    assert patched_tiled is not orig_tiled_encode
    assert cls._tpu_orig_encode is orig_encode
    assert cls._tpu_orig_tiled_encode is orig_tiled_encode

    # Idempotent on repeated application.
    patches._patch_wan_vae_replicated_encode()
    assert cls._encode is patched_encode
    assert cls.tiled_encode is patched_tiled
    assert cls._tpu_orig_encode is orig_encode
    assert cls._tpu_orig_tiled_encode is orig_tiled_encode

    torch.manual_seed(0)
    ref_untiled = autoencoder_kl_wan.OmniAutoencoderKLWan(**WAN21_SMALL_CONFIG).eval()
    ref_untiled.use_tiling = False
    ref_tiled = autoencoder_kl_wan.OmniAutoencoderKLWan(**WAN21_SMALL_CONFIG).eval()
    ref_tiled.load_state_dict(ref_untiled.state_dict())
    ref_tiled.use_tiling = True

    vae = cls(**WAN21_SMALL_CONFIG).eval()
    vae.load_state_dict(ref_untiled.state_dict())
    vae.use_tiling = True

    def _forbid_execute(*args, **kwargs):
        raise AssertionError(
            "DistributedVaeExecutor.execute must not run during encode"
        )

    vae.distributed_executor = SimpleNamespace(execute=_forbid_execute)
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: True)

    # Exceeds tile_sample_min (256): distributed VAE keeps decode tiling enabled
    # while running full-frame untiled encode locally on each rank.
    x = torch.randn(
        1, 3, num_frames, 320, 320, generator=torch.Generator().manual_seed(11)
    )
    with torch.no_grad():
        ref_dist = ref_untiled.encode(x).latent_dist.mean
        got_dist = vae.encode(x).latent_dist.mean
        assert torch.equal(got_dist, ref_dist)
        assert vae.use_tiling is True

        # Direct tiled_encode calls also stay rank-local without executor collectives.
        x_patch = (
            patchify(x, patch_size=vae.config.patch_size)
            if vae.config.patch_size is not None
            else x
        )
        ref_tiled_out = ref_tiled.tiled_encode(x_patch)
        got_tiled_out = vae.tiled_encode(x_patch)
        assert torch.equal(got_tiled_out, ref_tiled_out)


def test_wan_vae_replicated_encode_restores_tiling_and_respects_single_rank(
    wan_vae_modules, monkeypatch
):
    autoencoder_kl_wan = wan_vae_modules.autoencoder_kl_wan
    cls = autoencoder_kl_wan.DistributedAutoencoderKLWan
    orig_encode = getattr(cls, "_tpu_orig_encode", cls._encode)
    orig_tiled_encode = getattr(cls, "_tpu_orig_tiled_encode", cls.tiled_encode)
    monkeypatch.setattr(cls, "_encode", orig_encode)
    monkeypatch.setattr(cls, "tiled_encode", orig_tiled_encode)
    monkeypatch.delattr(cls, "_tpu_orig_encode", raising=False)
    monkeypatch.delattr(cls, "_tpu_orig_tiled_encode", raising=False)

    monkeypatch.setenv("VLLM_TORCHTPU_OMNI_WAN_VAE_HALO_ALLGATHER", "1")
    patches._patch_wan_vae_replicated_encode()

    torch.manual_seed(1)
    ref_tiled = autoencoder_kl_wan.OmniAutoencoderKLWan(**WAN21_SMALL_CONFIG).eval()
    ref_tiled.use_tiling = True
    vae = cls(**WAN21_SMALL_CONFIG).eval()
    vae.load_state_dict(ref_tiled.state_dict())
    vae.use_tiling = True

    # Single-rank explicit tiling remains tiled when distributed VAE is disabled.
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: False)
    x = torch.randn(1, 3, 1, 320, 320, generator=torch.Generator().manual_seed(19))
    with torch.no_grad():
        assert torch.equal(
            vae.encode(x).latent_dist.mean,
            ref_tiled.encode(x).latent_dist.mean,
        )

    # Error path restores use_tiling=True in finally when distributed VAE is enabled.
    monkeypatch.setattr(vae, "is_distributed_enabled", lambda: True)

    def _boom(*_args, **_kwargs):
        assert vae.use_tiling is False
        raise RuntimeError("boom")

    monkeypatch.setattr(vae.encoder, "forward", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        vae.encode(x)
    assert vae.use_tiling is True
