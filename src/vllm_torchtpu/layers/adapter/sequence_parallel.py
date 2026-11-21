# Copyright 2025 Google LLC
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
"""Row gating of sequence-parallel MoE.

A step at or above its layout's row threshold (``min_rows``) slices its rows
over TP; a smaller one keeps them replicated. The runner records the form of
the step about to run (``set_active``), and the TPU communicator reads it to
dispatch and combine the MoE rows accordingly.
"""

from vllm.config import ParallelConfig
from vllm.distributed import parallel_state

from vllm_torchtpu import envs

# The form of the step about to run. Sliced (main's form) until the runner
# records one, so a forward traced before the first step, or by a caller that
# records no form, takes main's path.
_ACTIVE = True

# Smallest step, in rows, that runs sequence-parallel. Below it the step keeps
# its rows replicated over TP: slicing a few rows saves less in the MoE than the
# extra all-gather and reduce-scatter cost.
#
# Only a model whose forward follows each step's form (it slices a step's rows
# exactly when the runner marks the step sequence-parallel) can run a step
# replicated, so the threshold applies once such a model calls
# follow_step_form(). Until then it is DEFAULT_MIN_ROWS, 0: every
# sequence-parallel step stays sliced and every compile size follows main's
# alignment rule, as on main.
DEFAULT_MIN_ROWS = 0
# The threshold for a model that follows the step's form: STEP_FORM_MIN_ROWS,
# or the entry of a layout measured to cross over at a different row count,
# keyed on (data_parallel_size, tensor_parallel_size). The values are the
# Qwen3.5-397B decode crossovers: 64 rows at DP4 x TP2 and 512 at TP8.
STEP_FORM_MIN_ROWS = 512
_MIN_ROWS_BY_LAYOUT: dict[tuple[int, int], int] = {(4, 2): 64, (1, 8): 512}
_FOLLOWS_STEP_FORM = False

# A speculative drafter is compiled apart from the model, so its forward and
# MoE read whatever step form was recorded last. Only an MTP proposer that
# records each draft forward's own form with set_active before running it
# lets a draft run replicated rows; with any other, a draft would run on the
# form of the target step before it. So for an engine with a drafter the row
# threshold stays DEFAULT_MIN_ROWS (every sequence-parallel step sliced, the
# draft included, as on main) until the proposer records the draft's form.
# The MTP proposer records it (`spec_decode/eagle3.py`, `_draft_rows_sliced`
# and `sequence_parallel.set_active` before each draft forward). It is the
# only proposer that can meet the threshold: the platform admits
# sequence-parallel MoE with no speculative method but MTP
# (`_configure_sequence_parallel_moe`), and without sequence-parallel MoE no
# MoE layer reads the recorded form.
DRAFT_RECORDS_STEP_FORM = True
# With the switch on, _HAS_DRAFTER no longer changes gates_rows(); it and
# set_has_drafter stay with their two callers until a follow-up drops them.
_HAS_DRAFTER = False


def parallel_layout(
    parallel_config: ParallelConfig | None = None,
) -> tuple[int, int] | None:
    """(data_parallel_size, tensor_parallel_size) of ``parallel_config``, or of
    this process's initialized model-parallel groups; None when neither is
    available (for example, before the workers set up their groups)."""
    if parallel_config is not None:
        return (
            int(parallel_config.data_parallel_size),
            int(parallel_config.tensor_parallel_size),
        )
    if not parallel_state.model_parallel_is_initialized():
        return None
    return (
        int(parallel_state.get_dp_group().world_size),
        int(parallel_state.get_tp_group().world_size),
    )


def follow_step_form() -> None:
    """Record that the model's forward follows each step's form.

    Called by a model patch that slices a step's rows only when the runner
    marks the step sequence-parallel (``set_active``); from then on a step
    below ``min_rows()`` runs replicated, and the runner writes the per-rank
    padding mask only for a sliced step (``step_rows_are_sliced``).
    """
    global _FOLLOWS_STEP_FORM
    _FOLLOWS_STEP_FORM = True


def follows_step_form() -> bool:
    return _FOLLOWS_STEP_FORM


def set_has_drafter(has_drafter: bool) -> None:
    """Record whether the engine runs a speculative drafter. The platform's
    sequence-parallel check records it in the engine process, and the runner
    in each worker, before any compile size or step is sized."""
    global _HAS_DRAFTER
    _HAS_DRAFTER = bool(has_drafter)


def gates_rows() -> bool:
    """Whether steps below the layout's threshold run replicated: the model
    follows each step's form, and the engine has no drafter or one whose
    proposer records each draft forward's form (DRAFT_RECORDS_STEP_FORM)."""
    return _FOLLOWS_STEP_FORM and (DRAFT_RECORDS_STEP_FORM or not _HAS_DRAFTER)


def min_rows(parallel_config: ParallelConfig | None = None) -> int:
    """The row threshold for the layout of ``parallel_config``, or of this
    process's model-parallel groups when it is None: DEFAULT_MIN_ROWS until a
    model follows each step's form (and, with a drafter, until its proposer
    records each draft forward's form; see gates_rows), then the layout's
    entry or STEP_FORM_MIN_ROWS."""
    if not gates_rows():
        return DEFAULT_MIN_ROWS
    layout = parallel_layout(parallel_config)
    if layout is None:
        return STEP_FORM_MIN_ROWS
    return _MIN_ROWS_BY_LAYOUT.get(layout, STEP_FORM_MIN_ROWS)


def enabled() -> bool:
    """Whether sequence-parallel MoE is on at all (`TPU_MOE_SEQUENCE_PARALLEL`).

    The row threshold (see min_rows) only decides which steps of an enabled
    model run sliced; it never turns SP on by itself.
    """
    return bool(envs.TPU_MOE_SEQUENCE_PARALLEL)


def fp8_gather() -> bool:
    """Whether the attention-side all-gather carries fp8 rows and scales.

    Always true. It acts only on a sequence-parallel step, whose gathered rows
    feed FP8 projections that quantize their input per row: quantizing each row
    before the gather instead of after it gives the same inputs and moves about
    half the bytes of a bf16 gather.
    """
    return True


def step_is_sequence_parallel(full_num_tokens: int) -> bool:
    return enabled() and int(full_num_tokens) >= min_rows()


def step_rows_are_sliced(parallel_config: ParallelConfig, full_num_tokens: int) -> bool:
    """Whether the forward about to run slices this step's rows over TP: the
    model follows each step's form, sequence-parallel MoE is admitted
    (``use_sequence_parallel_moe``, which vLLM's Qwen3.5 decoder layers set
    ``use_attn_reduce_scatter_for_moe`` from), and the step is
    sequence-parallel. Otherwise the MoE layers get every row of the step."""
    return (
        _FOLLOWS_STEP_FORM
        and parallel_config.use_sequence_parallel_moe
        and step_is_sequence_parallel(full_num_tokens)
    )


def set_active(active: bool) -> None:
    """Record the form of the step about to run.

    The runner calls it before the compiled model, never a traced forward:
    a global written inside the trace becomes a graph output.
    """
    global _ACTIVE
    _ACTIVE = bool(active)


def is_active() -> bool:
    return _ACTIVE


def rows_are_replicated(is_sequence_parallel: bool) -> bool:
    """Whether a sequence-parallel MoE layer sees the step's rows replicated
    on every rank (a step below min_rows)."""
    return is_sequence_parallel and not _ACTIVE
