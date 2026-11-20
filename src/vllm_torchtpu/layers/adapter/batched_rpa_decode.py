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
"""Batched RPA buffers sized for decode programs (TPU_RPA_DECODE_BATCHED).

A program of at most :data:`DECODE_PROGRAM_MAX_ROWS` query rows runs the
batched attention kernel with half the chip's VMEM, leaving the rest for the
compiler to prefetch the next layers' weights across the kernel, and with a
decode batch no larger than the requests the engine can hold.

Half the VMEM also halves the KV block the kernel streams, so a sequence takes
twice the blocks. That cost grows with the sequence's length and the gain does
not, so the settings are taken only while ``max_model_len`` is at most
:data:`DECODE_MAX_MODEL_LEN`.
"""

import dataclasses

from jax.experimental.pallas import tpu as pltpu
from vllm.config import VllmConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The largest token bucket treated as a decode program.
DECODE_PROGRAM_MAX_ROWS = 64
# The wrapper's default decode batch, and the cap here.
MAX_DECODE_BATCH = 8
# The longest max_model_len that takes the decode settings, placed on tpu7x
# (Qwen3.5 TP8, fp8 KV): the slowdown of MTP decode passes grows with context,
# and past 12k tokens it exceeds the 8k slowdown by more than ~2% of a step.
DECODE_MAX_MODEL_LEN = 12288


@dataclasses.dataclass(frozen=True)
class DecodeTuning:
    vmem_limit_bytes: int
    decode_batch_size: int


def decode_tuning(vllm_config: VllmConfig) -> DecodeTuning | None:
    """The decode programs' kernel settings, or None when the flag is off."""
    if not envs.TPU_RPA_DECODE_BATCHED:
        return None
    max_model_len = vllm_config.model_config.max_model_len
    if max_model_len > DECODE_MAX_MODEL_LEN:
        logger.warning_once(
            "[Attention] TPU_RPA_DECODE_BATCHED: max_model_len %d > %d; batched "
            "RPA keeps its default settings, whose larger KV blocks serve long "
            "sequences faster.",
            max_model_len,
            DECODE_MAX_MODEL_LEN,
        )
        return None
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    tuning = DecodeTuning(
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2,
        decode_batch_size=min(MAX_DECODE_BATCH, max_num_seqs),
    )
    spec = vllm_config.speculative_config
    rows_per_request = 1 + (spec.num_speculative_tokens if spec is not None else 0)
    max_decode_rows = max_num_seqs * rows_per_request
    logger.info_once(
        "[Attention] TPU_RPA_DECODE_BATCHED: buckets of <= %d rows run batched "
        "RPA with %d MiB of VMEM and a decode batch of %d, i.e. decode steps of "
        "<= %d requests at %d rows each; the largest decode step has %d rows "
        "and runs the %s settings.",
        DECODE_PROGRAM_MAX_ROWS,
        tuning.vmem_limit_bytes >> 20,
        tuning.decode_batch_size,
        DECODE_PROGRAM_MAX_ROWS // rows_per_request,
        rows_per_request,
        max_decode_rows,
        "decode" if max_decode_rows <= DECODE_PROGRAM_MAX_ROWS else "default",
    )
    return tuning


def kernel_kwargs(tuning: DecodeTuning | None, num_q_rows: int) -> dict[str, int]:
    """Keyword arguments for the batched kernel of a program of ``num_q_rows``."""
    if tuning is None or num_q_rows > DECODE_PROGRAM_MAX_ROWS:
        return {}
    return dataclasses.asdict(tuning)
