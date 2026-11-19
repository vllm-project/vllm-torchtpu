# SPDX-License-Identifier: Apache-2.0
"""q38 L4 EAGER-FIX E-c: full-length sampling-parameter copies in vLLM's
InputBatch (VLLM_TPU_SAMPLING_COPY_FULL, default off; compile-ignored).

vLLM's ``InputBatch._make_sampling_metadata`` copies temperature, top_p and
top_k (and penalties when used) host -> device with
``copy_slice(cpu, dev, num_reqs)`` = ``dev[:num_reqs].copy_(cpu[:num_reqs])``
on every request add / remove. Under torch_tpu a write into a view compiles an
eager program keyed on the view's length (``copy__as_strided_inverse``), so a
new running-batch size compiles one at serving time. The TPU runner never reads
``sampling_metadata`` (it builds its own sampling tensors), but to be safe
against any reader this keeps the values: for 1-D buffers of equal shape it
copies the whole fixed-length buffer (a full overwrite: no view, one layout),
then returns the same ``dev[:num_reqs]`` view as before. Rows below num_reqs
are bitwise the same; rows past it are stale either way. Other copies
(the 2-D allowed-token mask) keep the stock slice copy.

With VLLM_TPU_EAGER_SHAPE_LOG on, every slice copy is also noted (site
``copy_slice``), so a baseline run names the lengths it compiles.
"""

import functools

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import eager_shape_log
from vllm_torchtpu.runner import l6_fskip as _l6_fskip

logger = init_logger(__name__)

_MARK = "_q38_sampling_copy_full"


def make_copy_slice(original, full: bool):
    @functools.wraps(original)
    def copy_slice(
        from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
    ) -> torch.Tensor:
        same_1d = (
            from_tensor.ndim == 1
            and to_tensor.ndim == 1
            and from_tensor.shape == to_tensor.shape
        )
        if eager_shape_log.enabled() and length != to_tensor.shape[0]:
            eager_shape_log.note(
                "copy_slice", (str(to_tensor.dtype), tuple(to_tensor.shape), length)
            )
        if full and same_1d:
            _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)
            to_tensor.copy_(from_tensor, non_blocking=True)
            return to_tensor[:length]
        _l6_fskip.S.n += 1  # q38 L6 fskip: vLLM's slice copy is an upload too (counted conservatively)
        return original(from_tensor, to_tensor, length)

    setattr(copy_slice, _MARK, True)
    return copy_slice


def install() -> bool:
    """Patch ``vllm.v1.worker.gpu_input_batch.copy_slice`` (idempotent)."""
    full = envs.VLLM_TPU_SAMPLING_COPY_FULL
    if not (full or eager_shape_log.enabled()):
        return False
    from vllm.v1.worker import gpu_input_batch

    if getattr(gpu_input_batch.copy_slice, _MARK, False):
        return True
    gpu_input_batch.copy_slice = make_copy_slice(gpu_input_batch.copy_slice, full)
    logger.info(
        "q38 EAGER-FIX: InputBatch copy_slice patched (full-length 1-D copies=%s, "
        "shape log=%s)",
        full,
        eager_shape_log.enabled(),
    )
    return True
