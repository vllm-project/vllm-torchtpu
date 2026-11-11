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
"""Opt-in MSA kernel variants, set from `--additional-config` keys
(`vllm_torchtpu.opt_in_config`) before the first trace.

The kernels are plain JAX and read module-level flags; the attention layer
calls `apply` from its constructor. Every key defaults off (the kernels'
default paths), and an enabled one logs once per process with its key.
"""

from typing import Any

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_indexer,
)
from vllm_torchtpu.opt_in_config import option

logger = init_logger(__name__)

_logged: set[str] = set()

# (additional_config key, module, module-level flag)
TOGGLES = (
    ("minimax_msa_unstable_sort", msa_attention, "UNSTABLE_UNION_SORT"),
    ("minimax_dcp_fine_scan_widths", msa_dcp, "FINE_SCAN_WIDTHS"),
    ("minimax_msa_onehot_lookup", msa_indexer, "ONEHOT_LOOKUP"),
    ("minimax_indexer_dma_ring3", msa_indexer, "DMA_RING3"),
    ("minimax_indexer_fp8_transposed", msa_indexer, "FP8_TRANSPOSED"),
    ("minimax_msa_lse_r6", msa_attention_lse, "R6_KERNEL"),
)


def _set(additional_config: Any, key: str, module, attr: str) -> None:
    value = bool(option(additional_config, key))
    setattr(module, attr, value)
    if value and key not in _logged:
        _logged.add(key)
        logger.info("MiniMax-M3 MSA kernel variant on: additional_config.%s", key)


def apply(additional_config: Any) -> None:
    for key, module, attr in TOGGLES:
        _set(additional_config, key, module, attr)
    # The DCP op factory (`ops.msa`) reads this one when it builds the op.
    from vllm_torchtpu.models.minimax_m3.ops import msa as msa_ops

    _set(additional_config, "minimax_dcp_heads_attend", msa_ops, "DCP_HEADS_ATTEND")
