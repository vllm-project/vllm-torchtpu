# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Qwen3.8 as a megakernel program: the prefill role's pipeline stage
(:class:`QwenPrefillStage`), and what it supports.

The prefill runs as one stage of vLLM's PP x PCP engine layout: a stage is
one host's ``PCP_SIZE`` cores (vLLM's prefill context parallel group), which
holds a contiguous run of the 92 layers (vLLM's pipeline partition) and
computes every scheduled token of a step, sequence parallel over its cores.
:func:`from_vllm` builds it from vLLM's configuration; the standalone tools
build it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.program import Capabilities

NUM_LAYERS = 92
HIDDEN_SIZE = 8192
VOCAB = 248320
PCP_SIZE = 8  # the cores of a stage (one host)
FULL_ATTENTION_INTERVAL = 4  # layers 3, 7, ..., 91 are full attention, the rest GDN
# Sampling parameters the first-token op implements (beyond greedy).
SAMPLING: frozenset[str] = frozenset()
# --model-loader-extra-config keys, by role: the prefill reads the model's
# checkpoint itself (``synthetic_seed``: random weights instead, for timing).
LOADER_KEYS = {"prefill": frozenset({"synthetic_seed"}), "decode": frozenset()}


def is_full_attention(layer: int) -> bool:
    return layer % FULL_ATTENTION_INTERVAL == FULL_ATTENTION_INTERVAL - 1


def check(config: MegakernelEngineConfig) -> None:
    """Raise ``NotImplementedError`` for a configuration the Qwen3.8
    programs do not implement."""
    if config.role != "prefill":
        raise NotImplementedError(
            "the Qwen3.8 megakernel serves the prefill role of a P/D pair"
        )
    if config.prefill_context_parallel_size != PCP_SIZE:
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel runs a stage on one host's cores: "
            f"--prefill-context-parallel-size {PCP_SIZE}"
        )
    if config.tensor_parallel_size != 1 or config.decode_context_parallel_size != 1:
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel runs --tensor-parallel-size 1 and "
            "--decode-context-parallel-size 1"
        )
    if not config.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel writes an FP8 KV cache: "
            "--kv-cache-dtype fp8"
        )
    if config.mamba_ssm_cache_dtype != "float32":
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel keeps FP32 GDN states: "
            "--mamba-ssm-cache-dtype float32 (the checkpoint's mamba_ssm_dtype)"
        )
    interleave = config.cp_kv_cache_interleave_size
    if interleave is None or interleave % 128:
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel needs --cp-kv-cache-interleave-size "
            f"a multiple of 128, not {interleave}"
        )


def capabilities(config: MegakernelEngineConfig) -> Capabilities:
    check(config)
    return Capabilities(
        roles=frozenset({"prefill"}),
        sampling=SAMPLING,
        speculation=None,
        num_speculative_tokens=0,
        max_step_tokens=config.max_num_batched_tokens,
        # A GDN state is checkpointed where a chunk ends: on a state block.
        page_aligned_chunks=True,
        block_copies=False,
        # The prefill's first tokens are not served; it accepts the requests.
        structured_outputs=True,
        pipeline=True,
    )


def from_vllm(vllm_config: Any, tp_rank: int):
    """This worker's stage of the prefill (``tp_rank`` is 0: TP 1)."""
    from vllm.distributed import get_pp_group
    from vllm.distributed.parallel_state import get_pcp_group
    from vllm.distributed.utils import get_pp_indices

    from vllm_torchtpu.distributed.pcp import get_or_create_pcp_mesh
    from vllm_torchtpu.megakernel import backend
    from vllm_torchtpu.megakernel.qwen38.stage import AXIS, QwenPrefillStage

    hf_config = vllm_config.model_config.hf_text_config
    if (
        int(hf_config.num_hidden_layers) != NUM_LAYERS
        or int(hf_config.hidden_size) != HIDDEN_SIZE
        or int(hf_config.full_attention_interval) != FULL_ATTENTION_INTERVAL
    ):
        raise NotImplementedError("the Qwen3.8 megakernel implements full Qwen3.8 only")
    config = MegakernelEngineConfig.from_vllm_config(vllm_config)
    check(config)
    pp = get_pp_group()
    first, last = get_pp_indices(NUM_LAYERS, pp.rank_in_group, pp.world_size)
    return QwenPrefillStage(
        config,
        Path(vllm_config.model_config.model),
        layers=range(first, last),
        first_stage=pp.is_first_rank,
        last_stage=pp.is_last_rank,
        mesh=get_or_create_pcp_mesh(AXIS),
        core=get_pcp_group().rank_in_group,
        synthetic_seed=backend.synthetic_seed(vllm_config),
    )
