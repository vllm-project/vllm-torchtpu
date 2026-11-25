# SPDX-License-Identifier: Apache-2.0
"""Debug-only comparison of the DCP attention merges on served steps.

``VLLM_TPU_QSA_DCP_MERGE_CHECK=1``: every main-KV DCP QSA layer (target and
MTP draft) keeps the all-gather merge as its output and also runs the rs,
rs_bf16 and proj merges of VLLM_TPU_QSA_DCP_MERGE on the same partials
(``qsa_op._dcp_finish``), folding each one's disagreement with the all-gather
result into an f32 [4, 8] module buffer (row 3: the locally projected group
q/gate of VLLM_TPU_QSA_DCP_LOCAL_QGATE against the all-gathered one, when that
flag is set too) (``qsa_rpa.dcp_merge_check_stats``:
max abs error, max |reference|, max error relative to the head vector's
largest |reference|, fraction of differing elements, NaN count, merged-LSE
difference, call count). Every ``INTERVAL_S`` the runner reads the buffers on
every rank, logs one ``DCPMERGE_CHECK`` line per layer and mode, and zeroes
them, so each line covers the steps since the previous one.
"""

import time

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

INTERVAL_S = 60.0
MODES = ("rs", "rs_bf16", "proj", "qgate")

_layers: list = []
_state = {"last": None, "index": 0}


def register(attention, device) -> None:
    """Gives a DCP QSA layer its check buffer and lists it for the runner."""
    attention.register_buffer(
        "_merge_check",
        torch.zeros((len(MODES), 8), dtype=torch.float32, device=device),
        persistent=False,
    )
    _layers.append(attention)


def after_forward(runner) -> None:
    """Logs and zeroes the check buffers once every ``INTERVAL_S``."""
    del runner
    if not _layers:
        return
    now = time.monotonic()
    if _state["last"] is None:
        _state["last"] = now
        return
    if now - _state["last"] < INTERVAL_S:
        return
    _state["last"] = now
    index = _state["index"]
    _state["index"] += 1
    for layer in _layers:
        stats = layer._merge_check.cpu()
        layer._merge_check.zero_()
        if not float(stats[0, 6]):
            continue
        for row, mode in enumerate(MODES):
            s = stats[row].tolist()
            if not s[6]:
                continue
            logger.info(
                "DCPMERGE_CHECK interval=%d tp_rank=%d layer=%s mode=%s max_abs=%.3e "
                "ref_max=%.3e max_rel=%.3e frac_diff=%.5f nan=%d lse_err=%.3e calls=%d",
                index,
                layer.tp_rank,
                layer.prefix,
                mode,
                s[0],
                s[1],
                s[2],
                s[3],
                int(s[4]),
                s[5],
                int(s[6]),
            )
