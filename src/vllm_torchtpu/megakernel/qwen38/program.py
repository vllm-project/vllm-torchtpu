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
(:class:`QwenPrefillStage`), the decode role's program (:class:`QwenDecoder`),
and what they support.

The prefill runs as one stage of vLLM's PP x PCP engine layout: a stage is
one host's ``PCP_SIZE`` cores (vLLM's prefill context parallel group), which
holds a contiguous run of the 92 layers (vLLM's pipeline partition) and
computes every scheduled token of a step, sequence parallel over its cores.
The decoder runs TP32 with DCP8 attention over the pools the KV transfer
reshards the prefill's into. :func:`from_vllm` builds either from vLLM's
configuration; the standalone tools build the stage directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.program import Capabilities

NUM_LAYERS = 92
MAX_DECODE_REQUESTS = 16  # the decode megakernel's largest batch
HIDDEN_SIZE = 8192
VOCAB = 248320
PCP_SIZE = 8  # the cores of a stage (one host)
MTP_LAYER = 92  # the MTP layer's attention: a KV cache layer after the target's
MTP_DRAFTS = 3  # proposals per speculative round
TP_SIZE = 32  # the decoder's ranks
DCP_SIZE = 8  # the decoder's attention context parallel group
FULL_ATTENTION_INTERVAL = 4  # layers 3, 7, ..., 91 are full attention, the rest GDN
# Sampling parameters the decoder's ops implement (beyond greedy); the prefill
# accepts them too (its first tokens are not served).
SAMPLING = frozenset({"temperature", "top_p", "top_k", "seed"})
# --model-loader-extra-config keys, by role: the prefill reads the model's
# checkpoint itself (``synthetic_seed``: random weights instead, for timing;
# ``mtp``: write the MTP layer's KV for a decoder that speculates with it);
# the decoder reads its rank cache (``decode_weights``).
LOADER_KEYS = {
    "prefill": frozenset({"synthetic_seed", "mtp"}),
    "decode": frozenset({"presharded"}),
}
MTP_METHODS = frozenset({"mtp", "qwen3_5_mtp", "qwen3_next_mtp"})


def is_full_attention(layer: int) -> bool:
    return layer % FULL_ATTENTION_INTERVAL == FULL_ATTENTION_INTERVAL - 1


def check(config: MegakernelEngineConfig) -> None:
    """Raise ``NotImplementedError`` for a configuration the Qwen3.8
    programs do not implement."""
    if not config.kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "the Qwen3.8 megakernel keeps an FP8 KV cache: --kv-cache-dtype fp8"
        )
    if config.mamba_ssm_cache_dtype != "float32":
        raise NotImplementedError(
            "the Qwen3.8 megakernel keeps FP32 GDN states in the pools: "
            "--mamba-ssm-cache-dtype float32 (the checkpoint's mamba_ssm_dtype)"
        )
    interleave = config.cp_kv_cache_interleave_size
    if config.role == "decode":
        if (
            config.tensor_parallel_size != TP_SIZE
            or config.decode_context_parallel_size != DCP_SIZE
        ):
            raise NotImplementedError(
                f"the Qwen3.8 decode megakernel runs --tensor-parallel-size {TP_SIZE} "
                f"--decode-context-parallel-size {DCP_SIZE}"
            )
        if interleave not in (128, 256):
            raise NotImplementedError(
                "the Qwen3.8 decode megakernel needs --cp-kv-cache-interleave-size "
                f"128 or 256 (its KV page), not {interleave}"
            )
        if config.max_num_seqs > MAX_DECODE_REQUESTS:
            raise NotImplementedError(
                f"the Qwen3.8 decode megakernel steps at most {MAX_DECODE_REQUESTS} "
                f"requests: --max-num-seqs <= {MAX_DECODE_REQUESTS}"
            )
        return
    if config.role != "prefill":
        raise NotImplementedError(f"the Qwen3.8 megakernel has no {config.role} role")
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
    if interleave is None or interleave % 128:
        raise NotImplementedError(
            "the Qwen3.8 prefill megakernel needs --cp-kv-cache-interleave-size "
            f"a multiple of 128, not {interleave}"
        )


def mtp(vllm_config: Any) -> bool:
    """Whether the pair speculates with the checkpoint's MTP layer: the
    prefill's loader config ``mtp``, the decoder's speculative config."""
    from vllm_torchtpu.megakernel import backend

    if backend.role(vllm_config) == "prefill":
        return bool(backend.loader_option(vllm_config, "mtp"))
    spec = vllm_config.speculative_config
    if spec is None:
        return False
    if spec.method not in MTP_METHODS or spec.num_speculative_tokens != MTP_DRAFTS:
        raise NotImplementedError(
            f"the Qwen3.8 decoder speculates with the MTP layer, {MTP_DRAFTS} tokens"
        )
    if not vllm_config.scheduler_config.async_scheduling:
        # Synchronous scheduling never schedules the drafts, which stay on the device.
        raise NotImplementedError("the Qwen3.8 MTP decoder needs async scheduling")
    return True


def kv_layers(vllm_config: Any) -> list[Any]:
    """KV cache layers the programs keep beyond the model's: the MTP layer's
    attention (``model.layers.92``, as the target's attention layers, so vLLM
    groups it with them: the same block tables and pages, and the KV transfer
    moves it with the target's). Keep the returned modules alive."""
    if not mtp(vllm_config):
        return []
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention

    context = vllm_config.compilation_config.static_forward_context
    # Any of this worker's target attention layers (a stage holds only its own).
    target = next(
        context[f"model.layers.{layer}.self_attn.attn"]
        for layer in range(NUM_LAYERS)
        if is_full_attention(layer)
        and f"model.layers.{layer}.self_attn.attn" in context
    )
    with set_current_vllm_config(vllm_config):
        return [
            Attention(
                target.num_heads,
                target.head_size,
                target.impl.scale,
                num_kv_heads=target.num_kv_heads,
                cache_config=vllm_config.cache_config,
                prefix=f"model.layers.{MTP_LAYER}.self_attn.attn",
            )
        ]


def capabilities(config: MegakernelEngineConfig) -> Capabilities:
    check(config)
    if config.role == "decode":
        speculating = config.num_speculative_tokens > 0
        return Capabilities(
            roles=frozenset({"decode"}),
            sampling=SAMPLING,
            speculation="mtp" if speculating else None,
            num_speculative_tokens=MTP_DRAFTS if speculating else 0,
            max_step_tokens=config.max_num_batched_tokens,
            page_aligned_chunks=False,
            block_copies=False,
        )
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
    """This worker's stage of the prefill (``tp_rank`` is 0: TP 1), or its
    rank of the decoder."""
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
    if config.role == "decode":
        from vllm_torchtpu.megakernel.qwen38.decode import QwenDecoder

        presharded = backend.presharded(vllm_config)
        if presharded is None:
            raise ValueError("the Qwen3.8 decoder loads its rank cache: set presharded")
        return QwenDecoder(
            config,
            presharded,
            tp_rank=tp_rank,
            mtp_checkpoint=Path(vllm_config.model_config.model)
            if mtp(vllm_config)
            else None,
        )
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
        mtp=mtp(vllm_config),
    )
