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
"""TPU tests for DiffusersPipelineLoader GGUF and multithreaded/broadcast loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gguf
import portpicker
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch_tpu._internal.utils import hardware
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
)
from vllm_omni.diffusion.model_loader import diffusers_loader as diffusers_loader_mod
from vllm_omni.diffusion.model_loader.checkpoint_adapters import get_checkpoint_adapter
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, PR_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [pytest.mark.omni_feature, pytest.mark.omni]

GGUF_MAX_REL_L2 = 5.0e-2
GGUF_MAX_ABS = 1.0e-1
BROADCAST_MAX_REL_L2 = 2.0e-2
BROADCAST_MAX_ABS = 4.0e-2


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "GGUF quantization is not registered in vLLM/vllm-omni without vllm-gguf-plugin"
    ),
)
def test_gguf_quantization(tmp_path: Path) -> None:
    """GGUF Quantization.

    - TPU path: Write a tiny local GGUF weight file via ``gguf.GGUFWriter``,
      build ``OmniDiffusionConfig`` with ``method="gguf"``, load into a tiny
      ``ReplicatedLinear`` via ``DiffusersPipelineLoader`` on ``"tpu"``, and
      run forward.
    - Setup check: Assert the local GGUF file exists,
      ``od_config.quantization_config.get_name() == "gguf"``, and
      ``get_checkpoint_adapter`` resolves a GGUF adapter.
    - Outcome check: Load GGUF weights onto ``"tpu"``, run ``layer(x_tpu)`` in
      bfloat16, and assert relative-L2 / max-abs parity against CPU float32
      ``F.linear`` on the unquantized source weights.
    """
    torch.default_generator.manual_seed(0)
    src_weight = torch.randn(32, 64, dtype=torch.float32) * 0.1
    gguf_path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(gguf_path), "flux")
    writer.add_tensor("transformer.proj.weight", src_weight.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert gguf_path.exists(), f"Expected GGUF file at {gguf_path}"

    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        quantization_config={"method": "gguf", "gguf_model": str(gguf_path)},
    )
    assert od_config.quantization_config is not None, (
        "Expected non-None quantization_config for gguf"
    )
    assert od_config.quantization_config.get_name() == "gguf", (
        f"Expected get_name() == 'gguf', "
        f"got {od_config.quantization_config.get_name()!r}"
    )
    adapter = get_checkpoint_adapter(od_config)
    assert adapter is not None, "Expected GGUF checkpoint adapter"

    layer = ReplicatedLinear(
        64,
        32,
        bias=False,
        quant_config=od_config.quantization_config,
        params_dtype=torch.bfloat16,
        prefix="transformer.proj",
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config)
    source = DiffusersPipelineLoader.ComponentSource(
        model_or_path=str(tmp_path),
        subfolder=None,
        revision=None,
    )
    for name, tensor in loader._get_weights_iterator(source, layer):
        if name.endswith("weight"):
            with torch.no_grad():
                layer.weight.copy_(tensor)
    layer = layer.to("tpu")

    x_cpu = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    out_tpu, _ = layer(x_cpu.to("tpu"))
    assert_parity(
        out_tpu,
        F.linear(x_cpu.float(), src_weight),
        name="gguf_linear",
        max_rel_l2=GGUF_MAX_REL_L2,
        max_abs=GGUF_MAX_ABS,
    )


class _TinyShardModel(nn.Module):
    """Minimal module with bf16/fp32 parameters and an int64 buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        self.proj2 = nn.Linear(32, 16, bias=False, dtype=torch.float32)
        self.register_buffer("step_ids", torch.zeros(4, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj1(x).float()
        return self.proj2(h) + self.step_ids[:1].float()


def _broadcast_worker(
    rank: int,
    world_size: int,
    master_port: int,
    shard_dir: str,
    enable_multithread: bool,
    num_threads: int,
    x_cpu: torch.Tensor,
    result_dir: str,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        OmniTpuPlatform.set_device(0)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{master_port}",
            local_rank=rank,
        )

        target_device = torch.device("tpu")
        od_config = OmniDiffusionConfig(
            model=shard_dir,
            enable_multithread_weight_load=enable_multithread,
            num_weight_load_threads=num_threads,
            enable_broadcast_weight_load=True,
        )
        loader = DiffusersPipelineLoader(LoadConfig(), od_config)
        model = _TinyShardModel()
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            for buf in model.buffers():
                buf.zero_()
        model = model.to(target_device)

        spy_workers: list[int | None] = []
        orig_iter = diffusers_loader_mod.multi_thread_safetensors_weights_iterator

        def _spy_iter(
            hf_weights_files: list[str],
            use_tqdm_on_load: bool = True,
            max_workers: int = 4,
        ) -> Any:
            spy_workers.append(max_workers)
            return orig_iter(
                hf_weights_files,
                use_tqdm_on_load=use_tqdm_on_load,
                max_workers=max_workers,
            )

        diffusers_loader_mod.multi_thread_safetensors_weights_iterator = _spy_iter
        try:
            if rank == 0:
                source = DiffusersPipelineLoader.ComponentSource(
                    model_or_path=shard_dir,
                    subfolder=None,
                    revision=None,
                )
                state = dict(model.named_parameters()) | dict(model.named_buffers())
                with torch.no_grad():
                    for name, tensor in loader._get_weights_iterator(source, model):
                        state[name].copy_(tensor.to(target_device))
            else:
                zero_ok = all(
                    bool(p.detach().cpu().eq(0).all())
                    for p in list(model.parameters()) + list(model.buffers())
                )
                assert zero_ok, (
                    "Rank 1 parameters/buffers were not zeroed before broadcast"
                )

            dist.barrier()
            loader._broadcast_model_weights(
                model, target_device=target_device, src_rank=0
            )
            torch.accelerator.synchronize()
        finally:
            diffusers_loader_mod.multi_thread_safetensors_weights_iterator = orig_iter

        with torch.no_grad():
            out = model(x_cpu.to(target_device)).cpu()
        state_cpu = {
            name: tensor.detach().cpu()
            for name, tensor in (
                list(model.named_parameters()) + list(model.named_buffers())
            )
        }
        torch.save(
            {"spy_workers": spy_workers, "state": state_cpu, "output": out},
            Path(result_dir) / f"rank_{rank}.pt",
        )
    finally:
        if dist.is_initialized():
            destroy_distributed_env()


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.spawns_tpu_workers
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
@pytest.mark.parametrize(
    ("enable_multithread", "num_threads"),
    [(True, 4), (False, 1)],
    ids=["multithread_4", "sequential"],
)
def test_multithreaded_and_broadcast_weight_loading(
    tmp_path: Path,
    enable_multithread: bool,
    num_threads: int,
) -> None:
    """Multi-Threaded & Broadcast Weight Loading.

    - TPU path: Spawn 2 TPU worker processes; rank 0 loads multi-shard
      safetensors via ``DiffusersPipelineLoader`` (multithreaded or sequential)
      while rank 1 skips disk I/O, then both run ``_broadcast_model_weights``.
    - Setup check: Spy on ``multi_thread_safetensors_weights_iterator`` on
      rank 0 to assert ``max_workers=4`` when multithreaded and 0 calls when
      disabled, and assert rank 1 starts with zeroed TPU weights/buffers.
    - Outcome check: Assert every bf16/fp32 parameter and int64 buffer on both
      TPU ranks equals the shard tensors after broadcast and forward outputs on
      both ranks match the CPU fp32 reference via ``assert_parity``.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= 2, f"Test requires 2 TPU devices, found {device_count}"

    # A CPU generator: torch.manual_seed() also seeds the TPU generators, which
    # starts the TPU runtime in this parent and takes a device from the workers.
    gen = torch.Generator().manual_seed(7)
    w1 = torch.randn(32, 32, dtype=torch.bfloat16, generator=gen) * 0.1
    w2 = torch.randn(16, 32, dtype=torch.float32, generator=gen) * 0.1
    step_ids = torch.tensor([3, 5, 7, 11], dtype=torch.int64)

    shard_1 = tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file({"proj1.weight": w1}, str(shard_1))
    save_file({"proj2.weight": w2, "step_ids": step_ids}, str(shard_2))
    index_payload = {
        "weight_map": {
            "proj1.weight": shard_1.name,
            "proj2.weight": shard_2.name,
            "step_ids": shard_2.name,
        }
    }
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index_payload),
        encoding="utf-8",
    )

    x_cpu = torch.randn(2, 32, dtype=torch.bfloat16, generator=gen)
    ref_model = _TinyShardModel()
    with torch.no_grad():
        ref_model.proj1.weight.copy_(w1)
        ref_model.proj2.weight.copy_(w2)
        ref_model.step_ids.copy_(step_ids)
        ref_out = ref_model(x_cpu)

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    master_port = portpicker.pick_unused_port()
    context = mp.spawn(
        _broadcast_worker,
        args=(
            2,
            master_port,
            str(tmp_path),
            enable_multithread,
            num_threads,
            x_cpu,
            str(result_dir),
        ),
        nprocs=2,
        join=False,
    )
    try:
        while not context.join():
            pass
    finally:
        # Reap both workers even if the join is interrupted (e.g. by the timeout),
        # so no process holding a TPU device outlives the test.
        for proc in context.processes:
            proc.kill()
            proc.join()

    by_rank = {
        rank: torch.load(result_dir / f"rank_{rank}.pt", weights_only=True)
        for rank in (0, 1)
    }
    expected_workers = [4] if enable_multithread else []
    assert by_rank[0]["spy_workers"] == expected_workers, (
        f"Expected rank 0 spy_workers={expected_workers}, "
        f"got {by_rank[0]['spy_workers']}"
    )
    assert by_rank[1]["spy_workers"] == [], (
        f"Expected rank 1 not to read from disk, got {by_rank[1]['spy_workers']}"
    )

    expected_state = {
        "proj1.weight": w1,
        "proj2.weight": w2,
        "step_ids": step_ids,
    }
    for rank in (0, 1):
        for name, expected_tensor in expected_state.items():
            actual_tensor = by_rank[rank]["state"][name]
            assert torch.equal(actual_tensor, expected_tensor), (
                f"Rank {rank} tensor {name!r} mismatch after broadcast"
            )
        assert_parity(
            by_rank[rank]["output"],
            ref_out,
            name=f"broadcast_rank_{rank}_multithread_{enable_multithread}",
            max_rel_l2=BROADCAST_MAX_REL_L2,
            max_abs=BROADCAST_MAX_ABS,
        )
