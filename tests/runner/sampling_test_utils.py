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
"""Stand-ins for testing the chunk samplers (``runner/chunk_sampling.py``)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import chunk_sampling
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _sampling_runner(
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    spec_k: int | None = None,
) -> TPUModelRunner:
    """A runner with only the state the chunk samplers read; the class binds
    its methods.

    Every request samples at ``temperature``, ``top_k`` and ``top_p``, with no
    top-k candidate blocks, and the batch's top-k and top-p flags follow those
    values. With ``spec_k``, drafts of up to ``spec_k`` tokens are verified
    strictly.
    """
    runner = object.__new__(TPUModelRunner)
    runner.input_batch = SimpleNamespace(no_top_k=top_k == 0, no_top_p=top_p == 1.0)
    runner._build_padded_sampling_params = lambda _start, _end, logits: (
        torch.full((logits.shape[0], 1), temperature),
        torch.full((logits.shape[0], 1), top_k, dtype=torch.int32),
        torch.full((logits.shape[0], 1), top_p),
    )
    runner._sampler_topk_candidates = lambda *_args: 0
    runner.sample_from_logits_func = runner.sample_from_logits
    runner._stage_timing = MagicMock()
    if spec_k is not None:
        runner.speculative_config = SimpleNamespace(
            num_speculative_tokens=spec_k,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        runner.rejection_sampler = RejectionSampler(
            runner.speculative_config, torch.device("cpu")
        )
    return runner


def _sampling_step(**fields) -> chunk_sampling.StepContext:
    """The context of a greedy step without grammar, min_tokens or repetition
    penalty, with ``fields`` replaced."""
    return chunk_sampling.StepContext(
        **{
            "top_tokens": False,
            "all_greedy": True,
            "sampling_generator": None,
            "grammar_output": None,
            "scheduler_output": SimpleNamespace(scheduled_spec_decode_tokens={}),
            "is_spec_step": False,
            "min_toks": {},
            "rep_pen": False,
            **fields,
        }
    )


def _verify_md(
    drafts_per_req: list[list[int]], pad_tokens: int = 4, pad_reqs: int = 4
) -> SimpleNamespace:
    """Verify metadata for requests with ``drafts_per_req``, whose logits rows
    are each request's draft positions and then its bonus row. Draft slots pad
    to ``pad_tokens`` (as segment ``len(drafts_per_req)``) and requests to
    ``pad_reqs``."""
    flat_drafts, tgt_idx, bonus_idx, seg, grp = [], [], [], [], []
    row_cursor = 0
    for req_i, drafts in enumerate(drafts_per_req):
        flat_drafts.extend(drafts)
        tgt_idx.extend(range(row_cursor, row_cursor + len(drafts)))
        bonus_idx.append(row_cursor + len(drafts))
        seg.extend([req_i] * len(drafts))
        grp.extend(range(len(drafts)))
        row_cursor += len(drafts) + 1
    pad = pad_tokens - len(flat_drafts)
    lengths = [len(d) for d in drafts_per_req]
    return SimpleNamespace(
        draft_token_ids=torch.tensor(flat_drafts + [0] * pad, dtype=torch.int32),
        draft_lengths=torch.tensor(
            lengths + [0] * (pad_reqs - len(lengths)), dtype=torch.int32
        ),
        draft_lengths_cpu=np.array(lengths),
        target_logits_indices=torch.tensor(tgt_idx + [0] * pad, dtype=torch.int32),
        bonus_logits_indices=torch.tensor(
            bonus_idx + [0] * (pad_reqs - len(bonus_idx)), dtype=torch.int32
        ),
        final_logits_indices=torch.arange(
            max(pad_tokens, row_cursor), dtype=torch.int32
        ),
        segment_ids=torch.tensor(seg + [len(drafts_per_req)] * pad, dtype=torch.int64),
        group_indices=torch.tensor(grp + [0] * pad, dtype=torch.int32),
    )
