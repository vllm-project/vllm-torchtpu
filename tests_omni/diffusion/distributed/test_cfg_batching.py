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
"""TPU tests for single-rank diffusion CFG batching."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2
GUIDANCE_SCALE = 7.0


class _TinyCFGBatchedPipeline(CFGParallelMixin):
    supports_cfg_batching = True

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        proj: torch.nn.Linear,
    ) -> None:
        self.od_config = od_config
        self.transformer = proj
        self.transformer.cfg_branch = None
        self.calls: list[tuple[int, str | None]] = []

    def predict_noise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        assert return_dict is False, "Expected return_dict=False"
        self.calls.append((hidden_states.shape[0], self.transformer.cfg_branch))
        ctx = encoder_hidden_states.mean(dim=1)
        projected = self.transformer(hidden_states + ctx)
        return projected * timestep.view(-1, 1)


def _branch_kwargs(
    batch: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "hidden_states": torch.randn(batch, 16, generator=gen).to(
            dtype=torch.bfloat16, device=device
        ),
        "timestep": torch.full((batch,), 0.5, dtype=torch.bfloat16, device=device),
        "encoder_hidden_states": torch.randn(batch, seq_len, 16, generator=gen).to(
            dtype=torch.bfloat16, device=device
        ),
        "return_dict": False,
    }


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_diffusion_cfg_batching(omni_ctx: None) -> None:
    """Diffusion CFG Batching.

    - TPU path: configure ``OmniDiffusionConfig(..., enable_cfg_batching=True)``
      on a single CFG rank and run ``CFGParallelMixin.predict_noise_maybe_with_cfg``
      with a TPU bf16 linear denoiser.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and batched
      execution makes 1 denoiser call of size ``2 * batch`` with no branch stamp
      vs 2 sequential calls (``"positive"``, ``"negative"``).
    - Outcome check: batched output matches sequential output via
      ``assert_parity``, and mismatched prompt lengths fall back sequentially.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    model_dir = tiny_diffusion_dir()
    batched_cfg = OmniDiffusionConfig(model=model_dir, enable_cfg_batching=True)
    seq_cfg = OmniDiffusionConfig(model=model_dir, enable_cfg_batching=False)

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16).to(tpu_device)

    pos = _branch_kwargs(2, 12, seed=1, device=tpu_device)
    neg = _branch_kwargs(2, 12, seed=2, device=tpu_device)
    neg["hidden_states"] = pos["hidden_states"]

    batched_pipe = _TinyCFGBatchedPipeline(batched_cfg, proj)
    seq_pipe = _TinyCFGBatchedPipeline(seq_cfg, proj)

    batched_out = batched_pipe.predict_noise_maybe_with_cfg(
        True, GUIDANCE_SCALE, pos, neg, cfg_normalize=False
    )
    seq_out = seq_pipe.predict_noise_maybe_with_cfg(
        True, GUIDANCE_SCALE, pos, neg, cfg_normalize=False
    )

    assert batched_pipe.calls == [(4, None)], (
        f"Expected one batched call [(4, None)], got {batched_pipe.calls}"
    )
    assert seq_pipe.calls == [(2, "positive"), (2, "negative")], (
        f"Expected two sequential calls, got {seq_pipe.calls}"
    )
    assert_parity(
        batched_out,
        seq_out,
        name="diffusion_cfg_batching",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )

    neg_short = _branch_kwargs(2, 8, seed=3, device=tpu_device)
    neg_short["hidden_states"] = pos["hidden_states"]
    fallback_pipe = _TinyCFGBatchedPipeline(batched_cfg, proj)
    fallback_out = fallback_pipe.predict_noise_maybe_with_cfg(
        True, GUIDANCE_SCALE, pos, neg_short, cfg_normalize=False
    )
    ref_fallback = seq_pipe.predict_noise_maybe_with_cfg(
        True, GUIDANCE_SCALE, pos, neg_short, cfg_normalize=False
    )
    assert [n for n, _ in fallback_pipe.calls] == [2, 2], (
        f"Expected sequential fallback [2, 2], got {fallback_pipe.calls}"
    )
    assert_parity(
        fallback_out,
        ref_fallback,
        name="diffusion_cfg_batching_fallback",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
