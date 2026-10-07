"""Host copies of a step's Engram hash inputs, for Host-DRAM Engram tables.

With ``TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1`` the Engram rows are gathered on
the host, so the n-gram hashes must reach the host every step. Hashing on the
device and copying back (``gathered_hashes.to("cpu")``) blocks on the previous
step and stops async scheduling from overlapping host work with the device.
The runner already holds every hash input on the host before uploading it, so
V4.1 publishes those host copies for the model instead.

Like ``async_fence.py``, this wraps the runner's lookback upload rather than
editing the shared runner: ``forward_model`` calls
``_prepare_lookback_token_ids`` inside the step's forward context, after
``input_ids_cpu``, ``positions_cpu`` and ``query_start_loc_cpu`` are final.
"""

from dataclasses import dataclass

import numpy as np
import torch

HOST_ENGRAM_KEY = "deepseek_v41_host_engram"


@dataclass(frozen=True)
class HostEngramInputs:
    """Host views matching the device tensors of one forward step.

    ``input_ids``/``positions`` cover the padded token count from the start
    of the runner's host buffers (full-batch layout), ``query_start_loc`` the
    whole padded request table and ``lookback_token_ids`` the whole lookback
    buffer, so callers slice them to the device tensors' static shapes.
    ``all_prompt`` is True when every scheduled token of every request is a
    prompt token (none is a generated token, whose host id may be an async
    placeholder).
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    lookback_token_ids: torch.Tensor
    all_prompt: bool


def capture_host_engram_inputs(runner, num_reqs: int) -> HostEngramInputs | None:
    """Snapshot the runner's host hash inputs, or None if they do not apply."""
    from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind

    plan = getattr(runner, "_last_sequence_layout_plan", None)
    if plan is None or plan.kind is not SequenceLayoutKind.ALL:
        return None
    batch = runner.input_batch
    seq_lens = runner.seq_lens_np[:num_reqs]
    all_prompt = bool(np.all(seq_lens <= batch.num_prompt_tokens[:num_reqs]))
    return HostEngramInputs(
        input_ids=runner.input_ids_cpu,
        positions=runner.positions_cpu,
        query_start_loc=runner.query_start_loc_cpu,
        lookback_token_ids=runner.lookback_token_ids.cpu,
        all_prompt=all_prompt,
    )


def install_host_engram_inputs() -> None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    original = TPUModelRunner._prepare_lookback_token_ids
    if getattr(original, "_deepseek_v41_host_engram", False):
        return

    def prepare(self, num_reqs: int) -> torch.Tensor:
        ids = original(self, num_reqs)
        if is_forward_context_available():
            get_forward_context().additional_kwargs[HOST_ENGRAM_KEY] = (
                capture_host_engram_inputs(self, num_reqs)
            )
        return ids

    prepare._deepseek_v41_host_engram = True  # type: ignore[attr-defined]
    TPUModelRunner._prepare_lookback_token_ids = prepare  # type: ignore[method-assign]
