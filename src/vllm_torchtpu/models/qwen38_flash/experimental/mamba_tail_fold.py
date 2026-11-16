# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/core/mamba_tail_fold.py at
# 6d4183b4bff9ccd19fe1d88e68fbd09c9ee45d94.
#   Owned by models/qwen38_flash from this copy on.
"""Internal Mamba prefill checkpoints from the shared step metadata.

A ``MambaSpec`` with ``num_prefill_checkpoint_blocks`` makes vLLM's manager
reserve a block for the state at the prompt's last block boundary inside a
step; the GDN kernel and the PLE layer then write that state. This model's
pooled specs request none (the tail fold is retired), so the metadata carries
no checkpoints and ``prefill_checkpoint_of`` returns None.
"""

import torch


def prefill_checkpoint_of(md, num_reqs: int):
    """In the compiled forward: this step's per-request (offset, destination
    block), int32, from the shared metadata's internal prefill checkpoints
    (``mamba_prefill_ckpt_offsets`` and the group's resolved
    ``mamba_prefill_ckpt_indices``, -1 for none), or None when the step has
    none. Rows without a checkpoint get block 0, which the writers skip."""
    offsets = getattr(md, "mamba_prefill_ckpt_offsets", None)
    ids = getattr(md, "mamba_prefill_ckpt_indices", None)
    if offsets is None or ids is None:
        return None
    offsets = offsets[:num_reqs].to(torch.int32)
    ids = ids[:num_reqs].to(torch.int32)
    live = (offsets > 0) & (ids >= 0)
    blocks = torch.where(live, ids, torch.zeros_like(ids))
    return torch.where(live, offsets, torch.zeros_like(offsets)), blocks
