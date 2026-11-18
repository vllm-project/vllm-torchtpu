# SPDX-License-Identifier: Apache-2.0
"""q38 PPARF (VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER, default off): PP last
async output admits a prefill-only MTP producer, its drafts stay on device,
and the copy gate is re-armed behind them before the unwaited return.

Port of q38 HIGHC-PPAE1's tests/runner/test_pp_copygate_fence.py (the stage-7
copy-gate fence, TPU_PP_COPYGATE_FENCE mode 1) onto Benliang Wang's R tree.
The sample_tokens tests drive the real TPUModelRunner.sample_tokens (the
sampled-token copy, the drafter's proposal, the gate fence, the KV save, the
return) on a runner stub with his ``_pp_last_async_output`` set, with an
in-order device model that logs every graph enqueue (synchronize_tensors),
every event record and every event wait. The gate is right when the event the
worker's output thread waits on (get_output) was recorded after every
KV-writing root of the step was enqueued. Each assertion message starts with
a name the CDK CPU proof's mutants are matched against (ADMIT, FLAGSOFF,
DEVICE, EAGERSHAPE, ASYNC, ORDER, ROOTS, COVER, NODRAFT, STEP, ENGAGE).
"""

import inspect
import logging
import textwrap
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner import pp_last_async_drafter as pparf
from vllm_torchtpu.runner import pp_last_async_output as plao
from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)

FLAG = pparf.ENV
PPA = "VLLM_TPU_PP_LAST_ASYNC_OUTPUT"
ENGAGED = (
    "q38 PPARF on: prefill-only drafter admitted under PP last async output "
    "(VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER=1)"
)
_MTP = NS(method="mtp", num_speculative_tokens=1)
# The drafter's per-chunk row buckets in these tests: the device drafts may only
# ever take these [p, K] shapes (the ones the synchronous .cpu() copies).
_BUCKETS = {(16, 1), (32, 1)}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in (FLAG, PPA):
        monkeypatch.delenv(key, raising=False)
        # An earlier test's monkeypatch.setattr(envs, ...) undo leaves a real
        # module attribute that shadows envs.__getattr__ (see the qsa_rpa
        # compile-key test); read the registry entry itself.
        envs.__dict__.pop(key, None)
    monkeypatch.setattr(pparf, "_COUNT", {"fences": 0, "roots": 0})
    monkeypatch.setattr(pparf, "_DECISION", {"logged": False})


def _flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)


def _lines(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(pparf.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    return lines


# -- decide(): which runners return unwaited ---------------------------------------------------------------


def _config(pp=8, tp=1, pcp=1, async_scheduling=False, spec=None, kv=object()):
    return NS(
        parallel_config=NS(
            pipeline_parallel_size=pp, tensor_parallel_size=tp, prefill_context_parallel_size=pcp
        ),
        scheduler_config=NS(async_scheduling=async_scheduling),
        speculative_config=spec,
        kv_transfer_config=kv,
    )


def test_decide_admits_a_prefill_only_drafter_with_the_flag(monkeypatch):
    monkeypatch.setenv(PPA, "1")
    _flag(monkeypatch, "1")
    assert plao.decide(_config(spec=_MTP), True, False, prefill_only_drafter=True) == (True, "on"), (
        "ADMIT: the flag must admit a prefill-only MTP producer"
    )
    assert plao.decide(_config(spec=_MTP), True, False, prefill_only_drafter=False) == (
        False, "speculative decoding"), "ADMIT: a verifying drafter must still be refused"
    assert plao.decide(_config(spec=_MTP), True, False) == (False, "speculative decoding"), (
        "ADMIT: the default is a verifying drafter"
    )
    # Every other refusal still applies to an admitted drafter.
    assert plao.decide(_config(spec=_MTP), False, False, prefill_only_drafter=True) == (
        False, "not the last stage"), "ADMIT: only the last stage"
    assert plao.decide(_config(spec=_MTP, pp=1), True, False, prefill_only_drafter=True) == (
        False, "no pipeline"), "ADMIT: only a pipeline"
    assert not plao.decide(_config(spec=_MTP, async_scheduling=True), True, False, prefill_only_drafter=True)[0]
    assert not plao.decide(_config(spec=_MTP), True, True, prefill_only_drafter=True)[0]
    assert not plao.decide(_config(spec=_MTP, tp=2, kv=None), True, False, prefill_only_drafter=True)[0]
    assert plao.decide(_config(spec=_MTP, tp=2), True, False, prefill_only_drafter=True)[0]
    # No drafter: his decision, unchanged.
    assert plao.decide(_config(), True, False) == (True, "on")
    monkeypatch.setenv(PPA, "0")
    assert plao.decide(_config(spec=_MTP), True, False, prefill_only_drafter=True) == (False, "flag off")


@pytest.mark.parametrize("value", [None, "0"])
def test_decide_flag_off_is_his_decision(monkeypatch, value):
    """Off (unset or 0), a drafter is refused whatever kind it is, and no
    answer depends on the new parameter."""
    _flag(monkeypatch, value)
    monkeypatch.setenv(PPA, "1")
    for pod in (True, False):
        assert plao.decide(_config(spec=_MTP), True, False, prefill_only_drafter=pod) == (
            False, "speculative decoding"), f"FLAGSOFF: a drafter must be refused with the flag off (pod={pod})"
    for ppa in ("0", "1"):
        monkeypatch.setenv(PPA, ppa)
        for cfg in (_config(), _config(spec=_MTP), _config(pp=1, spec=_MTP), _config(tp=2, kv=None, spec=_MTP),
                    _config(async_scheduling=True, spec=_MTP)):
            for last in (True, False):
                for pooling in (True, False):
                    base = plao.decide(cfg, last, pooling)
                    assert plao.decide(cfg, last, pooling, prefill_only_drafter=True) == base, (
                        f"FLAGSOFF: decide must not depend on prefill_only_drafter with the flag off ({ppa}, {last}, {pooling})"
                    )


def _decide_fragment():
    """The runner's own decision lines in __init__, from the decide call up to
    the pooler fields, executed as written."""
    src = inspect.getsource(TPUModelRunner.__init__)
    a = src.index("self._pp_last_async_output, reason = pp_last_async_output.decide(")
    b = src.index("# Resolved in load_model")
    return textwrap.dedent(src[src.rindex("\n", 0, a) + 1 : src.rindex("\n", 0, b) + 1])


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("prefill_only", [True, False])
def test_runner_init_passes_its_drafter_kind_and_logs_once(monkeypatch, flag, prefill_only):
    monkeypatch.setenv(PPA, "1")
    _flag(monkeypatch, flag)
    lines = _lines(monkeypatch)
    stub = NS(_pp_is_last=True, is_pooling_model=False, _mtp_prefill_only=prefill_only)
    ns = {
        "self": stub, "vllm_config": _config(spec=_MTP), "pp_last_async_output": plao, "envs": envs,
        "logger": logging.getLogger("q38.pparf.test"), "_pparf": pparf,
    }
    exec(compile(_decide_fragment(), "<TPUModelRunner.__init__ decision>", "exec"), ns)
    on = flag == "1" and prefill_only
    assert stub._pp_last_async_output is on, (
        f"ADMIT: the runner must pass its prefill-only kind to decide (flag {flag}, prefill_only {prefill_only})"
    )
    mine = [line for line in lines if "PPARF" in line]
    if on:
        assert mine == [ENGAGED], f"ENGAGE: one engagement line when admitted, got {mine}"
    else:
        assert mine == [], f"ENGAGE: no line unless the flag admitted the drafter, got {mine}"


# -- the device model and the runner stub ------------------------------------------------------------------


class _Device:
    """An in-order device stream: enqueues, event records and waits, logged."""

    def __init__(self):
        self.log: list[tuple] = []
        self.n = 0
        self.drafts = None
        device = self

        class Event:
            def __init__(self):
                device.n += 1
                self.id = device.n

            def record(self):
                device.log.append(("record", self.id))

            def query(self):
                return False

            def synchronize(self):
                device.log.append(("wait", self.id))

        self.Event = Event

    def sync(self, tensors=None, wait=True):
        tensors = list(tensors or ())
        self.log.append(("enqueue", tuple(id(t) for t in tensors), wait, tuple(tuple(t.shape) for t in tensors)))

    def index(self, entry):
        return self.log.index(entry)

    def records(self):
        return [e[1] for e in self.log if e[0] == "record"]


def _runner(device, *, rows=None, drafter=True, prefill_only=True, ppa=False, producer=True, pp=8, buckets=(16,)):
    """The state sample_tokens reads on the last stage of a ``pp``-stage
    producer: requests ``rows`` = [(req_id, num_computed, num_scheduled,
    num_tokens)], greedy (row i samples 7 + i), with an MTP drafter whose
    drafts are ``seed + 100`` and which, like the device drafter under the
    flag (eagle3 propose's PPARF return), hands back its bucketed per-chunk
    [p, K] tensors (rows padded to each bucket in ``buckets``) only when asked
    to; the synchronous form is a host list after a blocking copy. The
    predicates are the real properties, evaluated on this stub under the
    test's env."""
    rows = rows or [("r0", 0, 1, 1)]
    n = len(rows)
    scheduler_output = NS(
        num_scheduled_tokens={r[0]: r[2] for r in rows},
        finished_req_ids=set(),
        has_structured_output_requests=False,
        total_num_scheduled_tokens=sum(r[2] for r in rows),
        kv_connector_metadata=None,
    )
    runner = NS(
        _pp_wave=None,
        _pp_is_last=True,
        parallel_config=NS(pipeline_parallel_size=pp),
        vllm_config=NS(kv_transfer_config=NS(is_kv_producer=producer)),
        execute_model_state=NS(
            scheduler_output=scheduler_output,
            draft_chunks=[],
            logits_list=[torch.zeros(n, 16)],
            num_reqs_list=[n],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
        ),
        input_batch=NS(
            max_num_logprobs=None,
            num_reqs=n,
            req_ids=[r[0] for r in rows],
            req_id_to_index={r[0]: i for i, r in enumerate(rows)},
            all_greedy=True,
            generators={},
            num_tokens_no_spec=[r[3] for r in rows],
            token_ids_cpu=np.zeros((n, 16), dtype=np.int32),
        ),
        requests={r[0]: NS(num_computed_tokens=r[1], num_tokens=r[3], output_token_ids=[]) for r in rows},
        speculative_config=_MTP if drafter else None,
        scheduler_config=NS(async_scheduling=False),
        _is_async_drafter=drafter,
        _mtp_prefill_only=bool(drafter and prefill_only),
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        _pp_last_async_output=ppa,
        _get_sampling_generator=lambda: None,
        _greedy_sample=lambda logits: torch.arange(7, 7 + logits.shape[0]),
        _update_mamba_slot_read_offsets=lambda *_a: None,
        _update_num_xla_graphs=lambda *_a: None,
        _modify_prev_results=lambda: None,
        _stage_timing=MagicMock(),
    )
    runner._stage_timing.take_samples.return_value = ([], [])
    runner.get_finished_kv_transfers = TPUModelRunner.get_finished_kv_transfers.__get__(runner)
    runner.maybe_wait_for_kv_save = lambda: device.log.append(("kv_save",))
    if drafter:

        def propose(**kwargs):
            device.log.append(("propose", kwargs["return_device"]))
            seed = (kwargs["device_seed"].view(-1, 1) + 100).to(torch.int32)
            chunks = []
            for p in buckets:
                chunk = torch.zeros(p, 1, dtype=torch.int32)
                chunk[: min(p, seed.shape[0])] = seed[:p]
                chunks.append(chunk)
            device.drafts = chunks
            return chunks if kwargs["return_device"] else None

        runner.drafter = NS()
        runner._async_drafter = runner.drafter
        runner.spec_decode_manager = NS(propose_draft_token_ids=propose)
    prop = getattr(TPUModelRunner, "_copygate_registers_ahead", None)
    if prop is not None:
        runner._copygate_registers_ahead = prop.fget(runner)
    runner._fence_spec_kv_under_token_copy = TPUModelRunner._fence_spec_kv_under_token_copy.fget(runner)
    return runner


def _sample(device, runner, monkeypatch):
    """sample_tokens with the device model: the token copy records the
    original ready event, as AsyncTPUCopyState.from_device_chunks does."""
    monkeypatch.setattr(torch, "tpu", NS(Event=device.Event), raising=False)
    copies = []

    def device_copy(chunks, lengths, spec_logprobs=None):
        device.log.append(("copy",))
        event = device.Event()
        event.record()
        state = AsyncTPUCopyState(
            chunks_cpu=[c.to(torch.int32) for c in chunks], chunk_real_lens=lengths, copy_ready_event=event
        )
        copies.append(state)
        return state

    with (
        patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False),
        patch("vllm_torchtpu.runner.tpu_runner.synchronize_tensors", side_effect=device.sync),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=device_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    return output, copies[0]


def _ahead(monkeypatch, value):
    monkeypatch.setattr(runner_mod.dist_utils, "get_stage3_register_ahead_enabled", lambda: value)


# -- 2. device drafts --------------------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("ppa", [False, True])
def test_device_drafts_only_with_the_flag_and_ppa(monkeypatch, flag, ppa):
    _flag(monkeypatch, flag)
    device = _Device()
    runner = _runner(device, ppa=ppa)
    _sample(device, runner, monkeypatch)
    want = flag == "1" and ppa
    proposals = [e for e in device.log if e[0] == "propose"]
    assert proposals == [("propose", want)], (
        f"DEVICE: return_device must be {want} with flag {flag} and PPA {ppa}, got {device.log}"
    )


# -- 3. the copy-gate fence --------------------------------------------------------------------------------


@pytest.mark.parametrize("buckets", [(16,), (16, 32)])
def test_gate_rearmed_after_drafts_and_roots_before_the_unwaited_return(monkeypatch, buckets):
    """On, under PPA: each bucketed draft chunk is enqueued in its own
    non-blocking call (chunk order, the graphs the synchronous .cpu()
    materializes), then every registered root in one call; the gate is
    re-recorded after them and before the KV save, sample_tokens returns
    without waiting, and the output thread's get_output then waits on the
    re-armed gate only. No enqueued draft takes a shape outside the buckets."""
    _flag(monkeypatch, "1")
    _ahead(monkeypatch, False)  # a pipeline producer registers ahead whatever the flag says
    device = _Device()
    runner = _runner(device, ppa=True, rows=[("r0", 0, 1, 1), ("r1", 4, 2, 6)], buckets=buckets)
    extra = torch.zeros(3)
    pparf.register_root(runner, extra)

    output, copy = _sample(device, runner, monkeypatch)

    enq = [e for e in device.log if e[0] == "enqueue"]
    drafts = device.drafts or []
    assert len(drafts) == len(buckets) and [e[1] for e in enq[: len(drafts)]] == [(id(d),) for d in drafts], (
        f"ORDER: each bucketed draft chunk must be enqueued in its own call, in chunk order, got {device.log}"
    )
    assert {e[3][0] for e in enq[: len(drafts)]} <= _BUCKETS, f"EAGERSHAPE: draft enqueues off the bucket set: {device.log}"
    assert len(enq) == len(drafts) + 1 and enq[-1][1] == (id(extra),), (
        f"ROOTS: the registered root must be enqueued after the drafts, before the gate, got {device.log}"
    )
    assert all(e[2] is False for e in enq), f"ORDER: no enqueue may wait, got {device.log}"
    gate = copy.copy_ready_event
    first = device.records()[0]
    assert gate.id != first, f"ORDER: the gate must be re-armed, got {device.log}"
    assert device.index(("record", gate.id)) > device.index(enq[-1]), (
        f"ORDER: the gate must be recorded after every enqueue, got {device.log}"
    )
    assert device.index(("record", gate.id)) < device.index(("kv_save",)), (
        f"ORDER: the gate fence precedes the KV save, got {device.log}"
    )
    assert isinstance(output, AsyncTPUModelRunnerOutput) and not [e for e in device.log if e[0] == "wait"], (
        f"ORDER: the step must return unwaited, got {device.log}"
    )
    result = output.get_output()
    assert ("wait", gate.id) in device.log and ("wait", first) not in device.log, (
        f"ORDER: the output thread must wait on the re-armed gate only, got {device.log}"
    )
    assert result.sampled_token_ids == [[7], [8]]
    assert runner.requests["r0"].output_token_ids == [], "ORDER: the runner commits through the scheduler's tokens"
    assert pparf.take_roots(runner) == [], "STEP: the fence takes the step's roots"


@pytest.mark.parametrize("flag", [None, "0"])
@pytest.mark.parametrize("ahead", [False, True])
def test_flag_off_keeps_the_base_fence(monkeypatch, flag, ahead):
    """Off, the drafter path is the base one: host drafts, no enqueue, the
    base predicate's fence (TPU_RAIDEN_STAGE3_REGISTER_AHEAD alone), the
    blocking wait, the immediate commit; the new module is never used."""
    _flag(monkeypatch, flag)
    _ahead(monkeypatch, ahead)

    def forbidden(*_a, **_k):
        raise AssertionError("FLAGSOFF: pp_last_async_drafter used with the flag off")

    monkeypatch.setattr(pparf, "fence", forbidden)
    monkeypatch.setattr(pparf, "take_roots", forbidden)
    device = _Device()
    runner = _runner(device)
    output, copy = _sample(device, runner, monkeypatch)

    assert [e for e in device.log if e[0] == "propose"] == [("propose", False)], f"FLAGSOFF: host drafts, got {device.log}"
    assert not [e for e in device.log if e[0] == "enqueue"], f"FLAGSOFF: no enqueue, got {device.log}"
    records = device.records()
    assert len(records) == (2 if ahead else 1), f"FLAGSOFF: the base predicate decides the fence, got {device.log}"
    if ahead:
        assert device.index(("record", records[1])) < device.index(("kv_save",)), f"FLAGSOFF: got {device.log}"
    assert not isinstance(output, AsyncTPUModelRunnerOutput), "FLAGSOFF: the synchronous return"
    assert [e for e in device.log if e[0] == "wait"] == [("wait", records[-1])], f"FLAGSOFF: one wait, got {device.log}"
    assert output.sampled_token_ids == [[7]] and runner.requests["r0"].output_token_ids == [7], "FLAGSOFF: commit"


def _eagle3_tail():
    """Eagle3Proposer.propose's own return block (from the final
    ``if return_device:`` to the host copy), executed as written on the
    per-chunk [p, K] drafts."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    src = inspect.getsource(Eagle3Proposer.propose)
    s0 = src.index("per_chunk_stacked = [")
    b = src.index("        result = []\n", s0)
    a = src.rindex("        if return_device:\n", s0, b)
    code = ("def _tail(per_chunk_stacked, chunks, runner, return_device):\n"
            + textwrap.indent(textwrap.dedent(src[a:b]), "    ") + "    return 'host'\n")
    ns = {"torch": torch, "_pparf": pparf}
    exec(compile(code, "<Eagle3Proposer.propose return>", "exec"), ns)
    return ns["_tail"]


@pytest.mark.parametrize("flag", [None, "1"])
@pytest.mark.parametrize("ppa", [False, True])
def test_device_drafts_keep_the_bucketed_shapes(monkeypatch, flag, ppa):
    """Under the flag and PPA the drafter's device return is its bucketed
    per-chunk [p, K] tensors themselves (no slice / cat to the real request
    count: no device program of a new shape). Otherwise (flag off, or an
    async-scheduling drafter without PPA) it is the base [num_reqs, K] form."""
    _flag(monkeypatch, flag)
    tail = _eagle3_tail()
    stacked = [torch.arange(16).view(16, 1), torch.arange(32).view(32, 1)]
    chunks = [NS(num_reqs=5), NS(num_reqs=20)]
    out = tail(stacked, chunks, NS(_pp_last_async_output=ppa), True)
    if flag == "1" and ppa:
        assert isinstance(out, list) and len(out) == 2 and all(o is t for o, t in zip(out, stacked)), (
            f"EAGERSHAPE: the PPARF device return must be the bucketed chunks themselves, got {out}"
        )
        assert {tuple(o.shape) for o in out} <= _BUCKETS, f"EAGERSHAPE: shapes off the bucket set: {[o.shape for o in out]}"
    else:
        tag = "ASYNC" if flag == "1" else "FLAGSOFF"
        assert isinstance(out, torch.Tensor) and tuple(out.shape) == (25, 1), (
            f"{tag}: the base device return is one [num_reqs, K] tensor, got {out if not isinstance(out, torch.Tensor) else out.shape}"
        )
        assert out.view(-1).tolist() == list(range(5)) + list(range(20)), f"{tag}: the base rows"
    assert tail(stacked, chunks, NS(_pp_last_async_output=ppa), False) == "host", "FLAGSOFF: the host form is untouched"


def _stub(*, producer=True, pp=8, drafter=True, kv=True):
    return NS(
        _is_async_drafter=drafter,
        vllm_config=NS(kv_transfer_config=NS(is_kv_producer=producer) if kv else None),
        parallel_config=NS(pipeline_parallel_size=pp),
    )


def _fence_prop(stub):
    stub._copygate_registers_ahead = TPUModelRunner._copygate_registers_ahead.fget(stub)
    return TPUModelRunner._fence_spec_kv_under_token_copy.fget(stub)


def test_coverage_follows_the_connector_rule(monkeypatch):
    """On, a pipeline producer is fenced whatever TPU_RAIDEN_STAGE3_REGISTER_AHEAD
    says (the connector registers ahead for every pipeline producer); a
    non-pipeline producer follows the flag; a consumer never. Off, the base
    keys on the flag alone."""
    _ahead(monkeypatch, False)
    _flag(monkeypatch, "1")
    assert _fence_prop(_stub(pp=8)) is True, "COVER: a pipeline producer must be fenced with the flag 0"
    assert _fence_prop(_stub(pp=1)) is False, "COVER: a non-pipeline producer with the flag 0 registers at finish"
    assert _fence_prop(_stub(producer=False)) is False, "COVER: a consumer is never fenced"
    assert _fence_prop(_stub(kv=False)) is False, "COVER: no KV transfer, no fence"
    assert _fence_prop(_stub(drafter=False)) is False, "COVER: the drafter property needs a drafter"
    _ahead(monkeypatch, True)
    assert _fence_prop(_stub(pp=1)) is True, "COVER: the flag on fences a non-pipeline producer"
    _flag(monkeypatch, "0")
    _ahead(monkeypatch, False)
    assert TPUModelRunner._fence_spec_kv_under_token_copy.fget(_stub(pp=8)) is False, (
        "FLAGSOFF: the base predicate is the flag alone"
    )


def test_producer_without_a_drafter_is_fenced(monkeypatch):
    """On, a register-ahead producer with no drafter still re-arms the gate
    after its registered roots (the base fences only drafter producers)."""
    _flag(monkeypatch, "1")
    _ahead(monkeypatch, False)
    device = _Device()
    runner = _runner(device, drafter=False, ppa=True)
    root = torch.zeros(2)
    pparf.register_root(runner, root)
    _output, copy = _sample(device, runner, monkeypatch)
    enq = [e for e in device.log if e[0] == "enqueue"]
    assert enq and enq[-1][1] == (id(root),), f"NODRAFT: the registered root must be enqueued, got {device.log}"
    assert device.index(("record", copy.copy_ready_event.id)) > device.index(enq[-1]), (
        f"NODRAFT: the gate must be re-armed after the root, got {device.log}"
    )
    assert device.index(("record", copy.copy_ready_event.id)) < device.index(("kv_save",)), (
        f"ORDER: the gate fence precedes the KV save, got {device.log}"
    )


@pytest.mark.parametrize("flag", [None, "1"])
def test_roots_are_cleared_at_execute_model_start_after_the_strip(monkeypatch, flag):
    _flag(monkeypatch, flag)
    log: list[str] = []
    real_take = pparf.take_roots
    monkeypatch.setattr(pparf, "take_roots", lambda runner: (log.append("take"), real_take(runner))[1])

    class _Stop(Exception):
        pass

    def update(_so):
        log.append("update")
        raise _Stop

    runner = NS(execute_model_state=None, _pp_wave=None, _mtp_prefill_only=True)
    runner._strip_prefill_only_padding = lambda _so: log.append("strip")
    runner._update_states = update
    stale = torch.zeros(1)
    pparf.register_root(runner, stale)
    with pytest.raises(_Stop):
        TPUModelRunner.execute_model(runner, NS())
    if flag == "1":
        assert log == ["strip", "take", "update"], f"STEP: the roots are cleared after the strip, got {log}"
        assert runner.__dict__["_copygate_roots"] == [], "STEP: a stale root must not reach this step's gate"
    else:
        assert log == ["strip", "update"], f"FLAGSOFF: execute_model start must be the base one, got {log}"
        assert runner.__dict__["_copygate_roots"] == [stale]


def test_roots_are_per_step():
    runner = NS()
    a, b = torch.zeros(1), torch.zeros(1)
    pparf.register_root(runner, a)
    pparf.register_root(runner, None)
    pparf.register_root(runner, b)
    assert pparf.take_roots(runner) == [a, b], "STEP: registered roots in order, None dropped"
    assert pparf.take_roots(runner) == [], "STEP: taking the roots clears them"


# -- the flag and the engagement lines ---------------------------------------------------------------------


@pytest.mark.parametrize("raw,want", [(None, False), ("0", False), ("", False), ("1", True), ("true", True)])
def test_flag_parse_and_default_off(monkeypatch, raw, want):
    _flag(monkeypatch, raw)
    assert pparf.enabled() is want, f"FLAGSOFF: {FLAG}={raw!r} -> {pparf.enabled()}, want {want}"


def test_flag_is_registered_and_not_a_compile_key():
    """Rule 13 (registered in the plugin) and host-only: listed in the compile
    ignore set (read from the source: importing tpu_compiler needs torch_tpu)."""
    import ast
    import pathlib

    assert FLAG in envs.environment_variables, "FLAGSOFF: the flag must be registered in envs.py"
    path = pathlib.Path(plao.__file__).parents[1] / "compilation" / "tpu_compiler.py"
    ignored = next(
        ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_TPU_COMPILE_ENV_IGNORED" for t in n.targets)
    )
    assert FLAG in ignored, "FLAGSOFF: the flag is host-only, not a compile-key factor"


def test_engagement_line_once_when_on(monkeypatch):
    _flag(monkeypatch, "1")
    lines = _lines(monkeypatch)
    pparf.note_decision(True, "on", is_last_stage=True, prefill_only_drafter=True)
    pparf.note_decision(True, "on", is_last_stage=True, prefill_only_drafter=True)
    assert lines == [ENGAGED], f"ENGAGE: one line per process when admitted, got {lines}"
    monkeypatch.setattr(pparf, "_DECISION", {"logged": False})
    lines.clear()
    pparf.note_decision(False, "not the last stage", is_last_stage=False, prefill_only_drafter=True)
    pparf.note_decision(True, "on", is_last_stage=True, prefill_only_drafter=False)
    assert lines == [], f"ENGAGE: ranks that do not draft print nothing, got {lines}"
    pparf.note_decision(False, "flag off", is_last_stage=True, prefill_only_drafter=True)
    assert lines == [
        "q38 PPARF NOT engaged: PP last async output is off (flag off) (VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER=1)"
    ], f"ENGAGE: the drafting rank says when the flag did not engage, got {lines}"
    assert getattr(pparf.logger, "_name", "").startswith("vllm."), "ENGAGE: the line must go to vLLM's logger"


def test_fence_line_first_and_every_report(monkeypatch):
    _flag(monkeypatch, "1")
    lines = _lines(monkeypatch)
    monkeypatch.setattr(pparf, "REPORT_EVERY", 3)
    state = NS(fence_pending_device_work=lambda: None)
    pparf.fence(state, [], [torch.zeros(1)], synchronize=lambda ts: None, skip_fence=True)
    assert lines == [], f"ENGAGE: a skipped fence must not log, got {lines}"
    for _ in range(3):
        pparf.fence(state, [torch.zeros(1)], [torch.zeros(1)], synchronize=lambda ts: None)
    assert len(lines) == 2, f"ENGAGE: the first re-arm and the REPORT_EVERY-th must log, got {lines}"
    assert lines[0].startswith(
        "q38 PPARF copy-gate fence (VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER=1): 1 gate re-arm(s), 3 root(s)"
    ), f"ENGAGE: got {lines[0]!r}"
    assert "3 gate re-arm(s), 7 root(s)" in lines[1], f"ENGAGE: got {lines[1]!r}"


@pytest.mark.parametrize("flag", [None, "0"])
def test_flag_off_prints_nothing(monkeypatch, flag):
    """Off, a full drafter step on a register-ahead producer under PPA logs no
    PPARF line (fence() and note_decision are never reached)."""
    _flag(monkeypatch, flag)
    _ahead(monkeypatch, True)
    lines = _lines(monkeypatch)
    device = _Device()
    _sample(device, _runner(device, ppa=True), monkeypatch)
    assert lines == [], f"ENGAGE: nothing with the flag off, got {lines}"
