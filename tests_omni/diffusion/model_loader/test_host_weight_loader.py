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
"""TPU tests for Host Weight Runtime loader integration in vllm-omni."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.host_weight_loader import HWRLoaderMixin
from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan
from vllm_omni.diffusion.model_loader.host_weights import (
    FinalLayoutBF16Producer,
    PreparedWeightSource,
    WeightSourceKind,
)
from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.diffusion.offloader import (
    DistributedLayerwiseOffloadBackend,
    OffloadStartupState,
    enable_offload_backend,
)
from vllm_omni.diffusion.offloader.startup import attach_offload_startup_state
from vllm_omni.host_weight_runtime import (
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    PostLoadPublicationOutcome,
    ProductionPolicy,
    ResolutionOutcome,
    RuntimeMode,
    StorageDomainPolicy,
)
from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


class _TinyHWRDiT(nn.Module):
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="test-tiny-hwr-dit",
        version="1",
    )
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.restore_validations = 0
        self.blocks = nn.ModuleList(
            [nn.Linear(4, 4, bias=False, dtype=dtype, device=device) for _ in range(2)]
        )

    def validate_restored_host_weights(self) -> None:
        self.restore_validations += 1
        for block in self.blocks:
            assert not block.weight.is_meta, (
                f"Restored block weight is still on meta device: {block.weight.device}"
            )
            assert block.weight.dtype is torch.bfloat16, (
                "Expected restored block weight dtype bfloat16, "
                f"got {block.weight.dtype}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _TinyHWRPipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.transformer = _TinyHWRDiT(dtype=dtype, device=device)


class _TestHWRLoader(HWRLoaderMixin):
    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.quant_config = None

    def _prepare_hwr_sources(self, model, modules, sources):
        del model, modules
        return tuple(
            PreparedWeightSource(
                model_or_path=str(src.folder),
                subfolder=None,
                requested_revision=None,
                prefix=src.prefix,
                resolved_root=Path(src.folder),
                weight_files=(Path(src.folder) / "model.safetensors",),
                use_safetensors=True,
                source_kind=WeightSourceKind.LOCAL_PATH,
            )
            for src in sources
        )


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "OmniTpuPlatform.supports_cpu_offload() is False so "
        "enable_offload_backend raises RuntimeError for HWR host_weight_plan"
    ),
)
def test_host_weight_runtime(tmp_path: Path) -> None:
    """Host Weight Runtime.

    - TPU path: Publish a tiny contract-enabled bf16 DiT into a local
      HostWeightRuntime store via HWRLoaderMixin, resolve a warm LOCAL_HIT
      HostWeightPlan, attach OffloadStartupState, and call enable_offload_backend.
    - Setup check: Assert _resolve_hwr returns ResolutionOutcome.LOCAL_HIT with
      backing_kind == "host_weight_runtime" and restore_validations == 1.
    - Outcome check: Assert DistributedLayerwiseOffloadBackend is enabled, run
      restored DiT forward on TPU, and compare output with CPU fp32 reference.
    """
    torch.manual_seed(42)
    ref_pipeline = _TinyHWRPipeline(dtype=torch.float32, device=torch.device("cpu"))
    x_cpu = torch.randn(2, 4, dtype=torch.float32)

    ckpt_dir = tmp_path / "transformer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "model.safetensors"
    save_file(
        {
            k: v.to(dtype=torch.bfloat16)
            for k, v in ref_pipeline.transformer.state_dict().items()
        },
        str(ckpt_file),
    )

    cold_pipeline = _TinyHWRPipeline(dtype=torch.bfloat16, device=torch.device("cpu"))
    with torch.no_grad():
        for cold_blk, ref_blk in zip(
            cold_pipeline.transformer.blocks, ref_pipeline.transformer.blocks
        ):
            cold_blk.weight.copy_(ref_blk.weight.to(dtype=torch.bfloat16))

    store_root = tmp_path / "hwr_store"
    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=False,
        host_weight_runtime_mode="required",
        host_weight_runtime_root=str(store_root),
    )
    loader = _TestHWRLoader(od_config)
    sources = (SimpleNamespace(prefix="transformer.", folder=str(ckpt_dir)),)
    modules = SimpleNamespace(
        dit_names=("transformer",), dits=(cold_pipeline.transformer,)
    )

    cold_ctx = loader._build_hwr_context(
        cold_pipeline,
        modules,
        load_format="default",
        sources=sources,
        source_digest_cache=None,
    )
    runtime = HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.REQUIRED,
            domain=StorageDomainPolicy(
                root=store_root, storage_class=detect_storage_class(store_root)
            ),
            production=ProductionPolicy(
                allow_local_build=False,
                allow_post_load_publish=True,
            ),
        )
    )
    pub = runtime.publish_after_load(
        cold_ctx.identity,
        producer=FinalLayoutBF16Producer(
            cold_ctx, cold_pipeline, (("transformer", cold_pipeline.transformer),)
        ),
    )
    assert pub.outcome is PostLoadPublicationOutcome.PUBLISHED, (
        f"Expected PUBLISHED, got {pub.outcome}"
    )

    warm_pipeline = _TinyHWRPipeline(dtype=torch.bfloat16, device=torch.device("meta"))
    warm_modules = SimpleNamespace(
        dit_names=("transformer",), dits=(warm_pipeline.transformer,)
    )
    state = loader._resolve_hwr(
        warm_pipeline,
        warm_modules,
        dist_offload=True,
        use_allgather=False,
        load_format="default",
        sources=sources,
    )
    assert state is not None, "_resolve_hwr returned None"
    assert state["outcome"] is ResolutionOutcome.LOCAL_HIT, (
        f"Expected LOCAL_HIT, got {state['outcome']}"
    )
    plan = state["plan"]
    assert isinstance(plan, HostWeightPlan), (
        f"Expected HostWeightPlan from _resolve_hwr, got {type(plan).__name__}"
    )
    assert plan.backing_kind == "host_weight_runtime", (
        f"Expected backing_kind='host_weight_runtime', got {plan.backing_kind!r}"
    )
    assert warm_pipeline.transformer.restore_validations == 1, (
        f"Expected restore_validations == 1, got "
        f"{warm_pipeline.transformer.restore_validations}"
    )

    attach_offload_startup_state(
        warm_pipeline, OffloadStartupState(host_weight_plan=plan)
    )
    tpu_device = torch.device("tpu")
    warm_pipeline, backend = enable_offload_backend(
        od_config, warm_pipeline, device=tpu_device
    )
    assert isinstance(backend, DistributedLayerwiseOffloadBackend), (
        f"Expected DistributedLayerwiseOffloadBackend, got {type(backend).__name__}"
    )
    assert backend.enabled, "DistributedLayerwiseOffloadBackend was not enabled"

    tpu_out = warm_pipeline.transformer(
        x_cpu.to(dtype=torch.bfloat16, device=tpu_device)
    )
    ref_out = ref_pipeline.transformer(x_cpu)
    assert_parity(
        tpu_out,
        ref_out,
        name="host_weight_runtime",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
