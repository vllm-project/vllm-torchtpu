# SPDX-License-Identifier: Apache-2.0
"""q38 L2 O1s-HOIST: the rank operands of the DCP / request-sharded / token-sharded QSA ops.

Port of MM's DSTOCK-B HOIST H-A (q38mm-dstockb 581cfffd3, test fix 2c9309247; on MM int-5 as
97b9b6e17 + d1e888ee1; author SHRIHARI MURALI with Claude), adapted to PD's O1s
(VLLM_TPU_QSA_DECODE_REQUEST_SHARD) and read from the registered flag
``envs.VLLM_TPU_QSA_SHARED_RANK_OPERANDS``.

Each QSA layer passes its DCP rank (``dcp_rank``) and TP rank (``request_shard_index`` /
``token_shard_index``) as an int32 [1] operand so that every rank traces the same program. With the
flag on, all layers on a rank register the same tensor object for each (kind, value, device), so the
compiled step takes one input instead of one per layer and XLA can merge the per-step selection
metadata derived from it (O1s: ``request_shard_step`` and the shared-stream ``_request_window``,
12x per step today, M: G2r j-92d0cc68 rank 0, 0.183 ms/step at 64 rows). Off (default): a fresh
tensor per call, exactly as the layers built them before.
"""

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SHARED: dict[tuple[str, int, str], torch.Tensor] = {}
# One engagement line per process (every TP rank; plain logger.info, as the O1s bind line), at the first
# reuse of a shared tensor: the proof that two layers hold one object, read by the STEP car's reader.
_LOGGED = False


def rank_operand(kind: str, value: int, device=None) -> torch.Tensor:
    """``torch.full((1,), value, dtype=int32, device=device)``; one shared object per
    (kind, value, device) when VLLM_TPU_QSA_SHARED_RANK_OPERANDS is on."""
    if not envs.VLLM_TPU_QSA_SHARED_RANK_OPERANDS:
        return torch.full((1,), value, dtype=torch.int32, device=device)
    key = (kind, int(value), str(torch.device(device)) if device is not None else "default")
    tensor = _SHARED.get(key)
    if tensor is None:
        tensor = torch.full((1,), value, dtype=torch.int32, device=device)
        _SHARED[key] = tensor
        return tensor
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info("q38 O1s-HOIST shared rank operands: on (%s %d reused across QSA layers)", kind, int(value))
    return tensor
