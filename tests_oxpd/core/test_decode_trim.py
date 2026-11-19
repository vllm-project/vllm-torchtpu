# SPDX-License-Identifier: Apache-2.0
"""q38 L1-C48 DECODE-TRIM (core/decode_trim.py, VLLM_TPU_DECODE_TRIM).

Three groups:
  P*  the policy alone (cost table, throughput rule with the re-entry loss, margin, must);
  S*  the per-step state machine on a fake scheduler (rotation, the one-step starvation bound at adversarial n,
      counters, row bound of trimmed steps);
  R*  the real vLLM AsyncScheduler (env-off identity, engagement, held-back requests stay RUNNING with unchanged
      accounting, re-entry ships all_token_ids, scheduler-side async rejection correction under emulated k=3 spec
      decode against an untrimmed twin), the install guards, the engine-core patch and the compile-key entry.
The TPU runner's own re-entry handling is not CPU-testable except the host-side substitution index builder (R9).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import decode_trim as dt
from vllm_torchtpu.core.decode_trim import DecodeTrim, TrimTable

# T98g c48 served fits (M, L5 census): D14 b64 14.749; stock b80 17.470, b96 20.206; b112 / b128 linear (I).
TABLE_A = "16:6.76/32:9.79/48:12.27/64:14.75/80:17.47/96:20.21/112:22.94/128:25.68/al:2.297"
# Same with S1 on D14 at 0.82 x T98i's bench delta (I).
TABLE_B = "16:6.51/32:9.10/48:11.35/64:13.38/80:17.47/96:20.21/112:22.94/128:25.68/al:2.297"


# ----------------------------------------------------------------------------- P: policy
def test_p1_parse_and_describe():
    t = TrimTable.parse("64:13.38/80:17.47/margin:0.05/al:2.3/reentry:0.5")
    assert t.costs == {64: 13.38, 80: 17.47}
    assert (t.margin, t.al, t.reentry) == (0.05, 2.3, 0.5)
    assert t.bucket(1) == 64 and t.bucket(64) == 64 and t.bucket(65) == 80 and t.bucket(81) is None
    assert "reentry 0.5" in t.describe()


@pytest.mark.parametrize(
    "raw",
    ["64:13.38,80:17.47", "64:13.38+80:17.47", "64:13.38 80:17.47", "64=13.38", "64:13.38/64:14",
     "64:0", "", "64:13/margin:1.5", "64:13/al:0.5", "64:13/reentry:2", "x:1"],
)
def test_p2_parse_rejects(raw):
    with pytest.raises(ValueError):
        TrimTable.parse(raw)


def _trims(raw, top=40):
    t = TrimTable.parse(raw)
    return {n: m for n in range(1, top + 1) if (m := t.choose(n, 4)) != n}


def test_p3_choose_today_table():
    """T98g as measured (no S1), re-entry charged (scratch/decL1/c48/bin/expect_choose.py, independent arithmetic)."""
    assert _trims(TABLE_A) == {9: 8, 13: 12, 17: 16, 21: 20, 25: 24, 29: 28}
    assert _trims(TABLE_A + "/above:64") == {17: 16, 21: 20, 25: 24, 29: 28}
    assert TrimTable.parse(TABLE_A).choose(40, 4) == 40  # past the table: never trimmed


def test_p4_choose_s1_table_and_reentry_off():
    assert _trims(TABLE_B) == {9: 8, 13: 12, 17: 16, 18: 16, 21: 20, 25: 24, 29: 28}
    want = {n: 16 for n in (17, 18, 19, 20, 21, 22, 23, 25, 26)} | {29: 28, 30: 28}
    assert _trims(TABLE_B + "/reentry:0/above:64") == want


def test_p5_margin_and_must():
    t = TrimTable.parse(TABLE_A)
    assert t.choose(21, 4) == 20
    assert TrimTable.parse(TABLE_A + "/margin:0.1").choose(21, 4) == 21  # gain 7.0 % < 10 %
    assert t.choose(17, 4, must=17) == 17
    tb = TrimTable.parse(TABLE_B + "/reentry:0")
    assert tb.choose(22, 4) == 16
    assert tb.choose(22, 4, must=18) == 20  # 16 is below must; 20 (b80) is the best allowed
    assert tb.choose(22, 4, must=21) == 22


# ----------------------------------------------------------------------------- S: state machine on a fake scheduler
class _FakeSched:
    """Calls the hook for every running request in order (vLLM's running loop) and schedules the ones it allows."""

    def __init__(self):
        self.running: list = []
        self.waiting: list = []
        self.requests: dict = {}
        self.num_spec_tokens = 3

    def add(self, rid):
        r = SimpleNamespace(request_id=rid, num_computed_tokens=100, num_prompt_tokens=100, status=None)
        self.running.append(r)
        self.requests[rid] = r

    def drop(self, rid):
        self.running = [r for r in self.running if r.request_id != rid]
        self.requests.pop(rid, None)

    def run(self, trim: DecodeTrim):
        trim.begin()
        sched = {}
        for r in self.running:
            n = trim.hook(r, 4)
            if n:
                sched[r.request_id] = n
        out = SimpleNamespace(num_scheduled_tokens=sched, total_num_scheduled_tokens=sum(sched.values()))
        trim.end(out)
        return out


def _trim(table=TABLE_B + "/reentry:0", log_every=0):
    s = _FakeSched()
    return s, DecodeTrim(s, TrimTable.parse(table), 4, log_every)


def test_s1_rotation_is_fair_at_17():
    s, trim = _trim()
    for i in range(17):
        s.add(f"r{i}")
    served = {f"r{i}": 0 for i in range(17)}
    for _ in range(17 * 8):
        out = s.run(trim)
        assert len(out.num_scheduled_tokens) == 16 and out.total_num_scheduled_tokens == 64
        for rid in out.num_scheduled_tokens:
            served[rid] += 1
    assert max(served.values()) - min(served.values()) <= 1
    assert trim.totals["max_consec_skip"] == 1


def test_s2_starvation_bound_one_step_at_adversarial_n():
    """Random arrivals / finishes with n swinging 1..40: no request is ever held back in two consecutive steps,
    checked from the outputs (independently of the module's counters), and every trimmed step fits its bucket."""
    rng = random.Random(1234)
    for table in (TABLE_A, TABLE_B, TABLE_B + "/reentry:0"):
        s, trim = _trim(table)
        nxt = 0
        skipped_last: set = set()
        worst = 0
        for step in range(3000):
            target = rng.choice([4, 12, 16, 17, 18, 20, 21, 24, 26, 30, 33, 40])
            while len(s.running) < target and rng.random() < 0.7:
                s.add(f"q{nxt}")
                nxt += 1
            if s.running and rng.random() < 0.3:
                s.drop(rng.choice(s.running).request_id)
            ids = {r.request_id for r in s.running}
            out = s.run(trim)
            got = set(out.num_scheduled_tokens)
            skipped = ids - got
            assert not (skipped & skipped_last), (step, skipped & skipped_last)
            if skipped:
                worst = max(worst, 1)
                b_got, b_all = trim.table.bucket(len(got) * 4), trim.table.bucket(len(ids) * 4)
                assert b_all is not None and b_got < b_all, (step, len(ids), len(got))
            skipped_last = skipped
        assert trim.totals["max_consec_skip"] <= 1 and worst == 1, table


class _Grab(__import__("logging").Handler):
    def __init__(self):
        super().__init__(level=0)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _grab():
    import logging

    g = _Grab()
    lg = logging.getLogger("vllm.vllm_torchtpu.core.decode_trim")
    lg.addHandler(g)
    lg.setLevel(logging.INFO)
    return lg, g


def test_s3_counters_and_window_line():
    s, trim = _trim(log_every=50)
    for i in range(20):
        s.add(f"r{i}")
    lg, g = _grab()
    try:
        for _ in range(50):
            s.run(trim)
    finally:
        lg.removeHandler(g)
    lines = [x for x in g.lines if "q38 DECODE-TRIM steps=" in x]
    assert len(lines) == 1
    line = lines[0]
    assert "steps=50 trimmed=50 skipped_req_steps=200" in line
    assert "unhonored=0" in line and "max_consec_skip=1" in line and "nm=20>16:50" in line
    assert "gap_ms_after_skip" in line


def test_s6_forced_rule_binds_when_more_are_held_than_served():
    """With 32 rows almost free and 128 rows expensive, n = 20 would serve 8 and hold 12. Least-recently-served
    rotation alone would then hold 4 of those 12 again; the forced rule serves all 12 (here: no trim that step)."""
    s, trim = _trim("32:1/128:100/al:1/reentry:0")
    for i in range(20):
        s.add(f"r{i}")
    skipped_last: set = set()
    for step in range(40):
        ids = {r.request_id for r in s.running}
        got = set(s.run(trim).num_scheduled_tokens)
        skipped = ids - got
        assert not (skipped & skipped_last), (step, sorted(skipped & skipped_last))
        skipped_last = skipped
    assert trim.totals["trimmed"] == 20 and trim.totals["max_consec_skip"] == 1


def test_s4_prefill_in_step_is_never_trimmed():
    s, trim = _trim()
    for i in range(20):
        s.add(f"r{i}")
    s.running[3].num_computed_tokens = 10  # a prefill chunk
    out = s.run(trim)
    assert len(out.num_scheduled_tokens) == 20
    assert trim.w["mixed"] == 1


def test_s5_ready_waiting_requests_count_and_reserve():
    from vllm.v1.request import RequestStatus

    s, trim = _trim()
    for i in range(16):
        s.add(f"r{i}")
    s.waiting = [SimpleNamespace(request_id="w0", status=RequestStatus.WAITING, num_computed_tokens=50)]
    out = s.run(trim)
    # n = 16 running + 1 ready waiting = 17 -> m = 16: one running request is held back for the arrival.
    assert len(out.num_scheduled_tokens) == 15
    assert trim.planned_n == 17 and trim.planned_m == 16


# ----------------------------------------------------------------------------- R: the real vLLM scheduler
MODEL = "facebook/opt-125m"
BLOCK = 16


@pytest.fixture(scope="module")
def model_config():
    from vllm.config import ModelConfig

    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _real(model_config, spec_k=0, num_blocks=256, max_num_seqs=32):
    import torch
    from vllm.config import CacheConfig, DeviceConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=2048, max_model_len=2048,
            enable_chunked_prefill=True, is_encoder_decoder=False, watermark=0.0, async_scheduling=True,
        ),
        model_config=model_config,
        cache_config=CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
                                 enable_prefix_caching=False),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0"], FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=64, dtype=torch.float16))],
    )
    cfg.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(cfg)
    s = AsyncScheduler(vllm_config=cfg, kv_cache_config=kv,
                       structured_output_manager=StructuredOutputManager(cfg), block_size=BLOCK)
    if spec_k:
        # Emulated MTP k: the scheduler-side half of speculative decoding (placeholder drafts, 1 + k rows per
        # decode request, rejection rollback in update_from_output). No drafter exists on CPU.
        s.num_spec_tokens = spec_k
        s._spec_token_placeholders = [-1] * spec_k
    init_none_hash(sha256)
    return s, get_request_block_hasher(BLOCK, sha256)


def _req(rid, hasher, seed, prompt_len=16, max_tokens=64):
    from vllm import SamplingParams
    from vllm.v1.request import Request

    p = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    p.update_from_generation_config({}, 2)
    return Request(rid, [seed * 1000 + i for i in range(prompt_len)], p, None, block_hasher=hasher)


def _tok(rid: str, pos: int) -> int:
    return 50 + (sum(map(ord, rid)) * 7 + pos * 13) % 900


def _accepted(rid: str, d: int, k: int) -> int:
    return (sum(map(ord, rid)) + 3 * d) % (k + 1)


def _drive(s, steps, k, record=None):
    """Synchronous engine loop: schedule -> deterministic outputs -> update_from_output. A request's sampled tokens
    depend only on (request, its own decode-step index), so trimmed and untrimmed twins produce the same streams."""
    from vllm.v1.outputs import ModelRunnerOutput

    dsteps: dict = {}
    outs = []
    for _ in range(steps):
        if not s.has_requests():
            break
        before = {rid: (r.num_computed_tokens, r.num_output_placeholders, r.status) for rid, r in s.requests.items()}
        out = s.schedule()
        outs.append(out)
        ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in ids:
            r = s.requests[rid]
            if r.num_computed_tokens < r.num_prompt_tokens:  # a partial prefill chunk: nothing sampled
                sampled.append([])
                continue
            spec = out.scheduled_spec_decode_tokens.get(rid)
            if spec:
                d = dsteps.get(rid, 0)
                dsteps[rid] = d + 1
                a = _accepted(rid, d, len(spec))
            else:
                a = 0
            base = r.num_output_tokens
            sampled.append([_tok(rid, base + j) for j in range(a + 1)])
        if record is not None:
            record(out, before)
        s.update_from_output(out, ModelRunnerOutput(req_ids=ids, req_id_to_index={x: i for i, x in enumerate(ids)},
                                                    sampled_token_ids=sampled))
    return outs


def test_r1_env_unset_installs_nothing(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    monkeypatch.delenv("VLLM_TPU_DECODE_TRIM", raising=False)
    init = EngineCore.__init__
    assert dt.patch_engine_core_for_decode_trim() is False
    assert EngineCore.__init__ is init and not getattr(EngineCore, "_q38_decode_trim_patch", False)
    for off in ("0", "off", ""):
        monkeypatch.setenv("VLLM_TPU_DECODE_TRIM", off)
        assert dt.patch_engine_core_for_decode_trim() is False and EngineCore.__init__ is init


def test_r2_env_set_patches_engine_core_and_bad_table_raises(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    init = EngineCore.__init__
    monkeypatch.setattr(EngineCore, "__init__", init)
    monkeypatch.setenv("VLLM_TPU_DECODE_TRIM", "64:13.38,80:17.47")
    with pytest.raises(ValueError):
        dt.patch_engine_core_for_decode_trim()
    monkeypatch.setenv("VLLM_TPU_DECODE_TRIM", TABLE_B)
    try:
        assert dt.patch_engine_core_for_decode_trim() is True
        assert EngineCore.__init__ is not init and EngineCore._q38_decode_trim_patch is True
    finally:
        EngineCore.__init__ = init
        if hasattr(EngineCore, "_q38_decode_trim_patch"):
            del EngineCore._q38_decode_trim_patch


def test_r3_identity_never_trimming_table_matches_bare_scheduler(model_config):
    """Env off = nothing installed (R1). With the hook installed but a table under which trimming never pays, the
    real scheduler's outputs are identical step by step to a bare twin's: the hook path itself is transparent."""
    seqs = []
    for install in (False, True):
        s, h = _real(model_config, spec_k=3)
        assert "schedule" not in vars(s) and "_mamba_block_aligned_split" not in vars(s)
        if install:
            assert dt.install(s, TrimTable.parse("1024:1"), 0) is not None
        for i in range(20):
            s.add_request(_req(f"r{i}", h, i, max_tokens=24))
        outs = _drive(s, 60, 3)
        seqs.append([(dict(o.num_scheduled_tokens), {k: list(v) for k, v in o.scheduled_spec_decode_tokens.items()},
                      o.total_num_scheduled_tokens) for o in outs])
    assert seqs[0] == seqs[1] and len(seqs[0]) > 10


def test_r4_trim_engages_held_back_stay_running(model_config):
    from vllm.v1.request import RequestStatus

    s, h = _real(model_config, spec_k=3)
    lg, g = _grab()
    try:
        trim = dt.install(s, TrimTable.parse("8:1/16:3/al:1/reentry:0"), 0)
    finally:
        lg.removeHandler(g)
    assert trim is not None and trim.rows_per_req == 4
    assert any("q38 DECODE-TRIM on: scheduler AsyncScheduler, rows/request 4" in x for x in g.lines), g.lines
    for i in range(3):
        s.add_request(_req(f"r{i}", h, i, max_tokens=40))
    seen = {"trimmed": 0}

    def record(out, before):
        if trim.skipped_prev:  # end() just stored this step's honored held-back set
            seen["trimmed"] += 1
            assert len(out.num_scheduled_tokens) == 2 and out.total_num_scheduled_tokens == 8
            (held,) = trim.skipped_prev
            assert held not in out.num_scheduled_tokens
            r = s.requests[held]
            assert r.status == RequestStatus.RUNNING and r in s.running
            assert (r.num_computed_tokens, r.num_output_placeholders) == before[held][:2]

    _drive(s, 25, 3, record)
    assert seen["trimmed"] >= 10
    assert trim.totals["max_consec_skip"] == 1 and trim.totals["unhonored"] == 0


def test_r5_reentry_ships_all_token_ids(model_config):
    s, h = _real(model_config, spec_k=3)
    dt.install(s, TrimTable.parse("8:1/16:3/al:1/reentry:0"), 0)
    for i in range(3):
        s.add_request(_req(f"r{i}", h, i, max_tokens=40))
    prev: set = set()
    checked = 0

    def record(out, before):
        nonlocal prev, checked
        now = set(out.num_scheduled_tokens)
        cached = out.scheduled_cached_reqs
        for rid in now - prev:
            if rid in cached.req_ids:
                assert list(cached.all_token_ids[rid]) == list(s.requests[rid].all_token_ids)
                checked += 1
        for rid in now & prev:
            if rid in cached.req_ids:
                assert rid not in cached.all_token_ids
        prev = now

    _drive(s, 25, 3, record)
    assert checked >= 8


def test_r6_async_rejection_accounting_matches_untrimmed_twin(model_config):
    """Emulated k=3: after every update the held-back and re-entering requests satisfy the scheduler's invariants
    (no placeholders left, num_computed_tokens == num_tokens - 1), and every request's final token stream equals the
    untrimmed twin's."""
    finals = []
    for install in (False, True):
        s, h = _real(model_config, spec_k=3)
        if install:
            dt.install(s, TrimTable.parse("8:1/16:3/al:1/reentry:0"), 0)
        reqs = [_req(f"r{i}", h, i, max_tokens=30) for i in range(3)]
        for r in reqs:
            s.add_request(r)

        def record(out, before, s=s):
            pass

        for _ in range(200):
            if not s.has_requests():
                break
            _drive(s, 1, 3)
            for rid, r in s.requests.items():
                if not r.is_finished() and r.num_output_tokens:
                    assert r.num_output_placeholders == 0, rid
                    assert r.num_computed_tokens == r.num_tokens - 1, (rid, r.num_computed_tokens, r.num_tokens)
        finals.append({r.request_id: list(r.output_token_ids) for r in reqs})
    assert finals[0] == finals[1]
    assert all(len(v) == 30 for v in finals[1].values())
    assert all(v == [_tok(k, j) for j in range(30)] for k, v in finals[1].items())


def test_r7_install_refuses_hook_owner_and_double_install(model_config):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    class Owner(AsyncScheduler):
        def _mamba_block_aligned_split(self, request, num_new_tokens, *a, **k):
            return num_new_tokens

    s, _ = _real(model_config)
    s.__class__ = Owner
    assert dt.install(s, TrimTable.parse(TABLE_B), 0) is None
    assert "schedule" not in vars(s)
    s2, _ = _real(model_config)
    assert dt.install(s2, TrimTable.parse(TABLE_B), 0) is not None
    assert dt.install(s2, TrimTable.parse(TABLE_B), 0) is None


def test_r8_compile_env_ignored():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_DECODE_TRIM" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "VLLM_TPU_DECODE_TRIM_LOG" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_r9_runner_substitution_skips_request_absent_last_step(monkeypatch):
    """Pins the re-entry contract. The TPU runner's regular substitution index builder maps only requests present in
    the previous step's async results, so without the stash a held-back request re-enters with the scheduler's -1
    placeholder drafts (the loss reentry=1 charges). q38 L1-DSTASH: with VLLM_TPU_DECODE_TRIM_DRAFT_STASH the builder
    is unchanged, and _dstash_apply fills the re-entering request's slots from the [bonus, draft_1..K] row it had in
    the results of the step it last ran in (the stash arm runs reentry:0); the other requests are untouched by it."""
    import numpy as np
    import torch

    from vllm_torchtpu.runner import draft_stash
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    r = object.__new__(TPUModelRunner)
    r.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    r._last_sequence_layout_plan = None
    r.input_batch = SimpleNamespace(req_ids=["a", "held", "b"])
    r._pre_async_results = SimpleNamespace(req_id_to_index_copy={"a": 0, "b": 1}, spec_decode_num_rejected_tokens=object())
    cur, src = r._prepare_async_token_substitution_indices(0, 3, np.array([4, 4, 4]))
    assert list(cur) == [0, 1, 2, 3, 8, 9, 10, 11]
    assert list(src) == [0, 1, 2, 3, 4, 5, 6, 7]
    # the stash: 'held' sat at row 2 of the step it last ran in
    comp = tpu_runner._substitute_placeholder_token
    monkeypatch.setattr(tpu_runner, "_substitute_placeholder_token",
                        getattr(comp, "_torchdynamo_orig_callable", None) or comp.__wrapped__)
    monkeypatch.delenv("VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", raising=False)
    r.device = torch.device("cpu")
    r._dstash = draft_stash.DraftStash(3, 0)
    stashed = torch.arange(1000, 1032, dtype=torch.int64)
    r._dstash.entries = {"held": (stashed, 2)}
    r.input_ids = torch.tensor([7] + [-1] * 3 + [8] + [-1] * 3 + [9] + [-1] * 3, dtype=torch.int32)
    r._dstash_apply(SimpleNamespace(has_structured_output_requests=False), 0, 3, np.array([4, 4, 4]))
    assert r.input_ids.tolist() == [7, -1, -1, -1, 1008, 1009, 1010, 1011, 9, -1, -1, -1]
    assert r._dstash.entries == {} and r._dstash_chunk == 0
