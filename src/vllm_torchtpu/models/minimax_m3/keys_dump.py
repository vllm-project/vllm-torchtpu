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
"""Torch entry point for the JAX MSA forward."""
"""Debug only: save the MSA prefill selections of one layer to disk.

With ``--additional-config '{"minimax_msa_dump_keys": <layer>}'`` the DCP MSA
op of that layer copies this rank's selected page keys into a device buffer
(`attention.MiniMaxM3SparseAttention.keys_dump`); `maybe_dump` runs after a
forward, outside the compiled graph, and saves the buffer for the first
`MAX_DUMPS` prefill forwards (more than `SMALL_BATCH_TOKENS` tokens) of the
ranks in `DUMP_RANKS`. Each file holds ``packed [T, 3 + G * topk]`` int32
(local_key, pos, valid, keys; ``-1`` = not held on this rank).
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_DUMPS = int(os.environ.get("MINIMAX_MSA_DUMP_MAX", "40"))
DUMP_RANKS = tuple(
    int(r) for r in os.environ.get("MINIMAX_MSA_DUMP_RANKS", "0,3").split(",")
)
SMALL_BATCH_TOKENS = 1024  # msa_attention.SMALL_BATCH_TOKENS

# Attention modules holding a keys-dump buffer (registered at init).
LAYERS: list[Any] = []

_state = {"dumps": 0, "forwards": 0, "rank": None, "dir": None}


def _rank() -> int:
    if _state["rank"] is None:
        try:
            import torch.distributed as dist

            _state["rank"] = dist.get_rank() if dist.is_initialized() else 0
        except Exception:  # noqa: BLE001
            _state["rank"] = 0
    return _state["rank"]


def _out_dir() -> str:
    if _state["dir"] is None:
        base = os.environ.get("CDK_OUTPUT_DIR") or "/cdk-outputs/outputs"
        path = os.path.join(base, "msa_keys_dump")
        os.makedirs(path, exist_ok=True)
        _state["dir"] = path
    return _state["dir"]


def maybe_dump(
    num_tokens_padded: int, num_reqs: int, extra: dict | None = None
) -> None:
    """Save the dump buffers after a real (served) forward."""
    if not LAYERS:
        return
    _state["forwards"] += 1
    if _state["dumps"] >= MAX_DUMPS or num_tokens_padded <= SMALL_BATCH_TOKENS:
        return
    rank = _rank()
    if rank not in DUMP_RANKS:
        return
    for layer in LAYERS:
        meta = layer.keys_dump_meta.cpu().numpy()
        rows = int(meta[0])
        packed = layer.keys_dump[: min(rows, layer.keys_dump.shape[0])].cpu().numpy()
        layer_id = "".join(c for c in layer.dump_layer_prefix.split("layers.")[-1].split(".")[0] if c.isdigit())
        name = f"rank{rank}_L{layer_id}_dump{_state['dumps']:03d}_fwd{_state['forwards']:06d}.npz"
        np.savez_compressed(
            os.path.join(_out_dir(), name),
            packed=packed,
            meta=meta,
            num_tokens_padded=np.int64(num_tokens_padded),
            num_reqs=np.int64(num_reqs),
            forward_index=np.int64(_state["forwards"]),
            rank=np.int64(rank),
            dcp_rank=np.int64(layer.dump_dcp_rank),
            layer=layer.dump_layer_prefix,
            is_prefill=np.bool_(True),
            **(extra or {}),
        )
        logger.warning("MSA keys dump: wrote %s (T=%d)", name, rows)
    _state["dumps"] += 1
