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
"""Decode-owning megakernel programs: the runner side of the contract.

A `MonoProgram` replaces one stock forward with a compiled step and keeps no
state between steps. Some megakernels instead keep each request's decode
state (its attention latents and recurrent state) in their own buffers: vLLM
runs the request's prefill, and once the prompt is complete the megakernel
*owns* the request and runs every later decode step for it, until the
request finishes. Such a program implements `DecodeOwner` below; the plugin's
`create(view, options)` returns it like any other program.

The runner calls a decode owner through a fixed set of hooks, all of them
no-ops when no owner is installed (`runner.decode_owner is None`):

- `load(model, draft_model)` once after the model (and draft) loaded;
  `bind_runner(runner)` after `initialize_kv_cache`;
- `release(req_ids)` for finished or preempted requests;
- per step: `owns(req_id)` (the runner moves owned requests to the front of
  the batch and runs vLLM's forward on the rest), `begin_step(num_chunks)`,
  `chunk_tokens(...)` before and `flush_chunk()` after each vLLM forward
  chunk (prefill side outputs the owner keeps), `flush_draft_chunk(i)` after
  the drafter's first pass over chunk `i`, `run_decode(...)` for the owned
  requests, and `take_ownership(...)` after sampling.

The helpers below are the parts of the runner integration that don't need
the runner's internals: building the owner from the megakernel config,
moving owned requests to the front, and merging the owner's tokens and
logprobs into vLLM's `ModelRunnerOutput`. Only the stdlib is imported at
module load; numpy and vLLM are imported by the helpers that need them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.outputs import LogprobsLists

logger = init_logger(__name__)


class DecodeResult(NamedTuple):
    """One owned request's tokens from one decode step.

    `tokens` are the generated (accepted) tokens, `top_ids` / `top_logprobs`
    the owner's top-k candidates per generated token (`[len(tokens), k]`,
    sorted by logprob), `drafts` the next draft tokens when the owner also
    drafts (speculative decoding), else `None`.
    """

    tokens: list[int]
    top_ids: Any
    top_logprobs: Any
    drafts: list[int] | None


@runtime_checkable
class DecodeOwner(Protocol):
    """A megakernel program that owns the decode of some requests.

    Every method is called identically on every TP rank, in the same order,
    with the same host arguments (all ranks see the same scheduler output).
    `drafts` is True when owned requests carry draft tokens.
    """

    drafts: bool

    def load(self, model: Any, draft_model: Any | None) -> None: ...

    def bind_runner(self, runner: Any) -> None: ...

    def release(self, req_ids: Iterable[str]) -> None: ...

    def owns(self, req_id: str) -> bool: ...

    def begin_step(self, num_chunks: int) -> None: ...

    def chunk_tokens(
        self,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_tokens_padded: int,
    ) -> None: ...

    def flush_chunk(self) -> None: ...

    def flush_draft_chunk(self, chunk_index: int) -> None: ...

    def take_ownership(
        self,
        runner: Any,
        scheduler_output: Any,
        sampled: Mapping[str, tuple[int, int]],
        drafts: Mapping[str, list[int]] | None,
    ) -> None: ...

    def run_decode(
        self, runner: Any, indices: Sequence[int], scheduler_output: Any
    ) -> Mapping[str, DecodeResult]: ...


def _config_decline(
    vllm_config: Any, spec: Any, options: Mapping[str, Any]
) -> str | None:
    """Return a config-level decline code if `vllm_config` cannot use `spec`."""
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    if bool(getattr(sched_cfg, "async_scheduling", False)):
        return "async_scheduling_unsupported"
    p_cfg = getattr(vllm_config, "parallel_config", None)
    tp_size = int(getattr(p_cfg, "tensor_parallel_size", 1) or 1)
    if spec.tp_sizes and tp_size not in spec.tp_sizes:
        return "unsupported_tp_size"
    if getattr(p_cfg, "pipeline_parallel_size", 1) > 1 and "pp" not in spec.parallelism:
        return "parallelism_unsupported:pp"
    if getattr(p_cfg, "data_parallel_size", 1) > 1 and "dp" not in spec.parallelism:
        return "parallelism_unsupported:dp"
    if (
        getattr(p_cfg, "decode_context_parallel_size", 1) > 1
        and "dcp" not in spec.parallelism
    ):
        return "parallelism_unsupported:dcp"
    if (
        getattr(p_cfg, "prefill_context_parallel_size", 1) > 1
        and "pcp" not in spec.parallelism
    ):
        return "parallelism_unsupported:pcp"
    if getattr(p_cfg, "enable_expert_parallel", False) and not (
        {"ep_in_tp", "ep_wider_than_tp"} & set(spec.parallelism)
    ):
        return "parallelism_unsupported:ep_in_tp"
    if getattr(p_cfg, "enable_eplb", False) and "eplb_dynamic" not in spec.features:
        return "eplb_dynamic_unsupported"
    if (
        getattr(vllm_config, "lora_config", None) is not None
        and "lora" not in spec.features
    ):
        return "lora_enabled"
    model_cfg = getattr(vllm_config, "model_config", None)
    if (
        bool(getattr(model_cfg, "enable_sleep_mode", False))
        and "sleep_mode" not in spec.features
    ):
        return "sleep_mode_unsupported"
    if (
        str(
            getattr(vllm_config, "load_config", None)
            and getattr(vllm_config.load_config, "load_format", "")
            or ""
        ).lower()
        == "dummy"
    ):
        return "dummy_weights_unsupported"
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    spec_toks = (
        int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0) if spec_cfg else 0
    )
    if spec_toks > 0 and "spec_decode" not in spec.features:
        return "spec_decode_unsupported"
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is not None:
        from .guards import _OFFLOADING_CONNECTORS, kv_connector_names

        kv_names = kv_connector_names(kv_cfg)
        if any(n in _OFFLOADING_CONNECTORS for n in kv_names) and (
            "kv_offload" not in spec.features
        ):
            return "kv_offload_unsupported"
        if any(n not in _OFFLOADING_CONNECTORS for n in kv_names) and (
            "kv_transfer" not in spec.features
        ):
            return "kv_connector_unsupported"
    return None


def create(runner: Any) -> DecodeOwner | None:
    """The decode owner the megakernel config selects for `runner`'s model.

    Reads `VLLM_TPU_MEGAKERNEL` and `additional_config["megakernel"]`
    (`config.select_plugin`). Returns `None` when the megakernel is off, no
    template claims the checkpoint, the template isn't installed, or the
    plugin declines (logged). A plugin whose program does not own decode
    raises: the runner serves only decode-owning programs today.
    """
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel.core.runtime.contract import (
        Decline,
        EligibilityResult,
    )

    from . import agreement, guards
    from . import config as mk_config
    from .plugin import RunnerView

    vllm_config = runner.vllm_config
    additional = getattr(vllm_config, "additional_config", None) or {}
    cfg, resolved = mk_config.select_plugin(
        envs.VLLM_TPU_MEGAKERNEL,
        additional.get("megakernel"),
        runner.model_config.hf_config,
    )
    if resolved is None:
        if cfg.enabled:
            logger.info("Megakernel: no template claims this checkpoint; skipping.")
        return None
    ref, options = resolved
    missing = mk_config.missing_template(ref, runner.model_config.hf_config)
    if missing is not None:
        logger.info(missing)
        return None
    spec, module = mk_config.import_plugin(ref)
    view = RunnerView(runner)

    local_decline: EligibilityResult | None = None
    if bool(
        getattr(
            getattr(vllm_config, "scheduler_config", None), "async_scheduling", False
        )
    ):
        local_decline = EligibilityResult(
            False,
            "async_scheduling_unsupported",
            f"Megakernel {spec.name} requires --no-async-scheduling; skipping.",
        )
    elif hasattr(runner, "supports_mm_inputs"):
        local_decline = guards.check_spec(runner, spec, view=view)
    else:
        cfg_code = _config_decline(vllm_config, spec, options)
        if cfg_code is not None:
            local_decline = EligibilityResult(
                False,
                cfg_code,
                f"Megakernel {spec.name} declined by config ({cfg_code}); skipping.",
            )

    program: Any = None
    if local_decline is None:
        create_fn = mk_config.resolve_callable(spec.create, module, what="create")
        program = create_fn(view, dict(options))
        if isinstance(program, Decline):
            local_decline = EligibilityResult(False, program.code, program.message)

    codes = agreement.vote_codes(
        (local_decline.code,) if local_decline is not None else ()
    )
    agreed_decline = agreement.agree_on_decline(local_decline, codes=codes)
    if agreed_decline is not None:
        logger.warning(
            "Megakernel %s declined (%s): %s",
            spec.name,
            agreed_decline.code,
            agreed_decline.message,
        )
        return None
    if not isinstance(program, DecodeOwner):
        raise mk_config.MegakernelConfigError(
            f"megakernel plugin {spec.name!r} returned {type(program).__name__}, "
            "which does not own decode; the TPU runner serves decode-owning "
            "megakernel programs only"
        )
    logger.info("Megakernel %s %s: owns decode.", spec.name, spec.version)
    return program


def configure_scheduler(vllm_config: Any) -> None:
    """Select `DecodeOwnerScheduler` when `vllm_config` activates a decode owner."""
    from vllm_torchtpu import envs

    from . import config as mk_config
    from .decode_owner_scheduler import SCHEDULER_CLS

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None) if model_config else None
    if hf_config is None:
        return
    additional = getattr(vllm_config, "additional_config", None) or {}
    try:
        _, resolved = mk_config.select_plugin(
            envs.VLLM_TPU_MEGAKERNEL, additional.get("megakernel"), hf_config
        )
    except mk_config.MegakernelConfigError:
        return
    if resolved is None:
        return
    ref, options = resolved
    if mk_config.missing_template(ref, hf_config) is not None:
        return
    try:
        spec, _ = mk_config.import_plugin(ref)
    except Exception:  # noqa: BLE001
        return
    decline_code = _config_decline(vllm_config, spec, options)
    if decline_code is not None:
        if decline_code == "async_scheduling_unsupported":
            logger.warning(
                "Megakernel %s requires --no-async-scheduling; skipping "
                "DecodeOwnerScheduler.",
                spec.name,
            )
        return
    scheduler_config = vllm_config.scheduler_config
    default_scheds = (
        None,
        "vllm.v1.core.sched.scheduler.Scheduler",
        SCHEDULER_CLS,
    )
    if scheduler_config.scheduler_cls not in default_scheds:
        logger.warning(
            "DecodeOwnerScheduler is off: scheduler %s is configured.",
            scheduler_config.scheduler_cls,
        )
        return
    scheduler_config.scheduler_cls = SCHEDULER_CLS
    logger.info("Enabled DecodeOwnerScheduler for megakernel plugin %s.", spec.name)


def move_owned_to_front(
    input_batch: Any, owner: DecodeOwner, num_decode: int, num_windowed: int
) -> tuple[int, int, int]:
    """Put the owned requests first, so vLLM's forward can start after them.

    `num_decode` / `num_windowed` are the runner's batch-order counts after
    its decode-first reorder (`[decodes][verify windows][prefills]`). Owned
    requests are single-token decodes or verify windows; everything after
    them keeps that order. Returns `(num_owned, num_decode, num_windowed)`
    with the counts measured from batch index 0, owned requests included.
    """
    n = input_batch.num_reqs
    ids = input_batch.req_ids
    flags = [ids[i] is not None and owner.owns(ids[i]) for i in range(n)]
    num_owned = sum(flags)
    if num_owned == 0:
        return 0, num_decode, num_windowed
    if any(flags[i] for i in range(num_windowed, n)):
        raise RuntimeError("a request the megakernel owns was scheduled as a prefill")
    order = [i for i in range(n) if flags[i]] + [i for i in range(n) if not flags[i]]
    held = list(range(n))  # held[slot] = original index now in that slot
    where = list(range(n))  # where[original index] = its current slot
    for dst, orig in enumerate(order):
        src = where[orig]
        if src != dst:
            input_batch.swap_states(dst, src)
            displaced = held[dst]
            held[dst], held[src] = orig, displaced
            where[orig], where[displaced] = dst, src
    rest_decode = sum(1 for i in range(num_decode) if not flags[i])
    rest_windowed = sum(1 for i in range(num_decode, num_windowed) if not flags[i])
    return (
        num_owned,
        num_owned + rest_decode,
        num_owned + rest_decode + rest_windowed,
    )


def logprobs_lists(results: Sequence[DecodeResult], num_logprobs: int) -> LogprobsLists:
    """vLLM logprob lists for the owner's tokens: per generated token
    `[sampled, top 1..k]` and the sampled token's 1-based rank."""
    import numpy as np
    from vllm.v1.outputs import LogprobsLists

    ids_out, lps_out, ranks, cu = [], [], [], [0]
    for res in results:
        for j, tok in enumerate(res.tokens):
            ids_j = np.asarray(res.top_ids[j])
            lps_j = np.asarray(res.top_logprobs[j], dtype=np.float32)
            hit = np.nonzero(ids_j == tok)[0]
            if hit.size == 0:
                raise RuntimeError(
                    f"megakernel token {tok} is not in its own top {ids_j.size}"
                )
            ids_out.append(
                np.concatenate(([tok], ids_j[:num_logprobs])).astype(np.int32)
            )
            lps_out.append(
                np.concatenate(([lps_j[hit[0]]], lps_j[:num_logprobs])).astype(
                    np.float32
                )
            )
            ranks.append(int(hit[0]) + 1)
        cu.append(cu[-1] + len(res.tokens))
    return LogprobsLists(
        logprob_token_ids=np.stack(ids_out),
        logprobs=np.stack(lps_out),
        sampled_token_ranks=np.asarray(ranks, np.int32),
        cu_num_generated_tokens=cu,
    )


def merge_logprobs(
    vllm_logprobs: Any,
    results: Sequence[DecodeResult],
    num_vllm_reqs: int,
    max_num_logprobs: int | None,
) -> Any:
    """A mixed step's logprobs: the owned requests' rows first, then vLLM's,
    with one `cu_num_generated_tokens` entry per request.

    vLLM-side requests without rows (a discarded partial-prefill sample, or
    no list at all) get zero-length entries, because the scheduler slices
    the logprobs of every request that asked for them.
    """
    import numpy as np
    from vllm.v1.outputs import LogprobsLists

    has_vllm = isinstance(vllm_logprobs, LogprobsLists)
    if max_num_logprobs is None and not has_vllm:
        return vllm_logprobs
    if has_vllm:
        k = vllm_logprobs.logprob_token_ids.shape[1] - 1
    else:
        assert max_num_logprobs is not None
        k = int(max_num_logprobs)
    owned = logprobs_lists(results, k)
    owned_cu = list(owned.cu_num_generated_tokens or [0])
    end = owned_cu[-1]
    if not has_vllm:
        return owned._replace(cu_num_generated_tokens=owned_cu + [end] * num_vllm_reqs)
    cu_v = vllm_logprobs.cu_num_generated_tokens or list(range(num_vllm_reqs + 1))
    cu = owned_cu + [end + c for c in cu_v[1:]]
    cu += [cu[-1]] * (len(owned_cu) + num_vllm_reqs - len(cu))
    return LogprobsLists(
        logprob_token_ids=np.concatenate(
            [owned.logprob_token_ids, vllm_logprobs.logprob_token_ids]
        ),
        logprobs=np.concatenate([owned.logprobs, vllm_logprobs.logprobs]),
        sampled_token_ranks=np.concatenate(
            [owned.sampled_token_ranks, vllm_logprobs.sampled_token_ranks]
        ),
        cu_num_generated_tokens=cu,
    )


def check_output(
    sampled_token_ids: Sequence[Sequence[int]],
    req_ids: Sequence[str | None],
    discard: Iterable[int],
    where: str,
) -> None:
    """Every scheduled, non-discarded request got at least one token, and
    there is one token list per request."""
    n = len(req_ids)
    if len(sampled_token_ids) != n:
        raise RuntimeError(
            f"megakernel {where} step: {len(sampled_token_ids)} token lists for "
            f"{n} requests"
        )
    skip = set(discard)
    empty = [
        (i, req_ids[i]) for i in range(n) if i not in skip and not sampled_token_ids[i]
    ]
    if empty:
        raise RuntimeError(
            f"megakernel {where} step: no token for requests {empty} "
            f"(discarded: {sorted(skip)})"
        )


def chunk_discards_all_samples(runner: Any, scheduler_output: Any, start: int) -> bool:
    """Whether no request in `[start, num_reqs)` keeps this step's sample.

    True when every one of them is a partial prefill (its prompt continues
    next step), so a chunk need not compute logits at all. Only then, and
    only when no logits processor reads token ids and the step is
    synchronous: an async step publishes its sampled tokens before the host
    knows which are discarded.
    """
    batch = runner.input_batch
    n = batch.num_reqs
    if runner.scheduler_config.async_scheduling:
        return False
    if getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    if batch.logits_processing_needs_token_ids[:n].any():
        return False
    for req_id in batch.req_ids[start:n]:
        if req_id is None:
            continue
        req = runner.requests[req_id]
        scheduled = scheduler_output.num_scheduled_tokens[req_id]
        if req.num_computed_tokens + scheduled >= req.num_tokens:
            return False
    return True


def owned_step_output(runner: Any, owner: DecodeOwner, scheduler_output: Any) -> Any:
    """One decode step for a batch the owner owns entirely, as vLLM's
    `ModelRunnerOutput` (vLLM's forward and sampling don't run)."""
    from vllm.v1.outputs import ModelRunnerOutput

    batch = runner.input_batch
    num_reqs = batch.num_reqs
    req_ids = [str(r) for r in batch.req_ids[:num_reqs]]
    res = owner.run_decode(runner, list(range(num_reqs)), scheduler_output)
    tokens = [res[r].tokens for r in req_ids]
    logprobs: Any = None
    if batch.max_num_logprobs is not None:
        logprobs = logprobs_lists(
            [res[r] for r in req_ids], int(batch.max_num_logprobs)
        )
    for b, req_id in enumerate(req_ids):
        idx = batch.req_id_to_index[req_id]
        start = batch.num_tokens_no_spec[idx]
        end = start + len(tokens[b])
        batch.token_ids_cpu[idx, start:end] = tokens[b]
        batch.num_tokens_no_spec[idx] = end
        runner.requests[req_id].output_token_ids.extend(tokens[b])
    if owner.drafts:
        runner.spec_decode_manager.set_draft_token_ids(
            [res[r].drafts or [] for r in req_ids]
        )
    out = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=dict(batch.req_id_to_index),
        sampled_token_ids=tokens,
        logprobs=logprobs,
        prompt_logprobs_dict={req_id: None for req_id in req_ids},
        pooler_output=[],
        kv_connector_output=None,
    )
    check_output(out.sampled_token_ids, req_ids, (), "owned")
    return out


def prepend_owned(
    runner: Any,
    owner: DecodeOwner,
    final_output: Any,
    num_owned: int,
    scheduler_output: Any,
) -> None:
    """A mixed step: vLLM sampled requests `[num_owned, num_reqs)`. Run one
    owner step for the front requests and put their tokens, logprobs and
    drafts ahead of vLLM's in `final_output`."""
    if num_owned == 0:
        return
    batch = runner.input_batch
    req_ids = [str(r) for r in batch.req_ids[:num_owned]]
    res = owner.run_decode(runner, list(range(num_owned)), scheduler_output)
    results = [res[r] for r in req_ids]
    final_output.logprobs = merge_logprobs(
        final_output.logprobs,
        results,
        len(final_output.sampled_token_ids),
        batch.max_num_logprobs,
    )
    final_output.sampled_token_ids = [r.tokens for r in results] + list(
        final_output.sampled_token_ids
    )
    if owner.drafts:
        manager = runner.spec_decode_manager
        vllm_drafts = manager.peek_draft_token_ids() or []
        manager.set_draft_token_ids([r.drafts or [] for r in results] + vllm_drafts)


def hand_over(
    runner: Any,
    owner: DecodeOwner,
    scheduler_output: Any,
    request_seq_lens: Sequence[tuple[int, Any, int, str]],
    discard: Iterable[int],
    sampled_token_ids: Sequence[Sequence[int]],
) -> None:
    """After a vLLM step: requests whose prompt just completed (vLLM sampled
    their first token) hand their decode state to the owner, which takes the
    ones it can serve."""
    skip = set(discard)
    sampled: dict[str, tuple[int, int]] = {}
    for i, _req_state, seq_len, req_id in request_seq_lens:
        if i in skip:
            continue
        toks = sampled_token_ids[i]
        if toks:
            sampled[req_id] = (int(toks[0]), int(seq_len))
    if not sampled:
        return
    drafts = None
    if owner.drafts:
        current = runner.spec_decode_manager.peek_draft_token_ids()
        if current is not None:
            ids = runner.input_batch.req_ids[: runner.input_batch.num_reqs]
            drafts = {str(r): list(d) for r, d in zip(ids, current)}
    owner.take_ownership(runner, scheduler_output, sampled, drafts)


__all__ = [
    "DecodeOwner",
    "DecodeResult",
    "check_output",
    "chunk_discards_all_samples",
    "create",
    "hand_over",
    "logprobs_lists",
    "merge_logprobs",
    "move_owned_to_front",
    "owned_step_output",
    "prepend_owned",
]
