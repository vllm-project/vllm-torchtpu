# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VPSHOST host levers (l6_ab classes). CDK CPU pod only (never the Mac or the VM).

T-2 stage 0 FB-ORDER (lever ``sampfb``, VLLM_TPU_L4_SAMP_FB_ORDER): on the VPS fallback path, execute_model skips the
top-k candidate scan whose result the fallback block overwrites.
T-2 stage 1 SAMP-SNAP (lever ``sampsnap``, VLLM_TPU_L4_SAMP_SNAP): the VPS verify's F, padded sampling-parameter
uploads and num_active upload come from a per-batch snapshot (runner/l4_vpshost.py SampSnap). The runner's real
`_sample_spec_verify_vps` / `_vps_verify_fb` / `_vps_fallback_rows` / `_padded_sampling_params` / upload helpers run on
a fake runner; the compiled verify is replaced by a recorder, and every input it receives (values, dtype, shape,
stride, storage offset, device) must be the same with the lever on as off.
T-1(i) STATS-ASYNC (lever ``statsasync``, VLLM_TPU_L4_VPS_STATS_ASYNC): the VPS_FALLBACK counts are copied to the host
without blocking (l4_vpshost.AsyncCounts); the runner's real `_vps_verify_fb` runs with the counts as a device-tensor
stand-in that counts every blocking read, and the lines must equal the lever-off lines.

The decision block is the real execute_model source (extracted with ast, executed with the real l6_ab module). The
runner's real `_sampler_topk_candidates`, `_topk_candidate_blocks` and `_vps_fallback_rows` run bound to a fake runner
over a fake InputBatch with vLLM v1's add / remove / swap / condense semantics for the sampling rows
(vllm/v1/worker/gpu_input_batch.py: greedy -> temperature 0.0, top_k outside (0, vocab) -> vocab).
Every result is compared with the lever off and with an oracle written here from the base (b551e9b84) semantics.
"""

import ast
import functools
import os
import random
from types import SimpleNamespace as NS

import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as tr
from vllm_torchtpu.runner import l4_vpshost
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import l6_ab

V = 248320          # Qwen3.8 vocabulary (a multiple of CANDIDATE_BLOCK 128)
K = 3               # served num_speculative_tokens
ROWS = (4, 8, 16, 32, 64)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---- import origin ----------------------------------------------------------------------------------------------
def test_imports_the_tree_under_test():
    """The modules under test resolve inside the tree this test file belongs to (or Q38_VPSHOST_EXPECT_ROOT); the
    CPU image also carries an installed vllm_torchtpu, which must not be the one imported."""
    root = os.path.realpath(os.environ.get("Q38_VPSHOST_EXPECT_ROOT") or os.path.join(HERE, "..", "..", "src"))
    for mod in (tr, l6_ab, tpu_compiler, l4_vpshost):
        where = os.path.realpath(mod.__file__)
        assert where.startswith(root + os.sep), f"{mod.__name__} imported from {where}, not under {root}"


# ---- l6_ab arming -------------------------------------------------------------------------------------------------
@pytest.fixture
def ab(monkeypatch):
    """arm(**env): set the L6 / L4 lever env (everything else of theirs unset) and re-resolve l6_ab; the module is
    re-resolved with the original env afterwards, so later tests see every lever off."""
    saved = dict(os.environ)
    lines = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))

    def arm(**env):
        for k in list(os.environ):
            if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_")) or k in (
                    "VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS", "VLLM_TPU_SERVED_AB"):
                del os.environ[k]
        os.environ["TPU_SAMPLER_TOPK_CANDIDATES"] = "64"     # the served value
        os.environ.update(env)
        l6_ab._resolve()
        return lines

    yield arm
    os.environ.clear()
    os.environ.update(saved)
    l6_ab._resolve()


# Step-count windows long enough that no test ever rolls one: the window's engagement counts (l6_ab.note) are then
# the whole arm's, so every lever-on arm can assert its lever really engaged (no vacuous on arm: s1v1 m2 j-4a352f1a).
ONE_WINDOW = {"VLLM_TPU_L6_AB_STEPS": "1000000"}


def engaged(key):
    """How many times `key` (lever.event) was noted in the current l6_ab window."""
    return int(l6_ab._state["counts"].get(key, 0))


# ---- fake InputBatch with vLLM v1's sampling-row semantics ---------------------------------------------------------
class FakeBatch:
    def __init__(self, max_reqs=256):
        self._req_ids = []
        self.req_id_to_index = {}
        self._removed = []
        self.temperature_cpu_tensor = torch.zeros(max_reqs, dtype=torch.float32)
        self.top_k_cpu_tensor = torch.zeros(max_reqs, dtype=torch.int32)
        self.top_p_cpu_tensor = torch.ones(max_reqs, dtype=torch.float32)

    @property
    def req_ids(self):
        return self._req_ids

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)

    def add_request(self, req_id, params):
        if self._removed:
            idx = min(self._removed)
            self._removed.remove(idx)
            self._req_ids[idx] = req_id
        else:
            idx = len(self._req_ids)
            self._req_ids.append(req_id)
        self.req_id_to_index[req_id] = idx
        t, k, p = params
        self.temperature_cpu_tensor[idx] = 0.0 if t == 0 else t
        self.top_k_cpu_tensor[idx] = k if 0 < k < V else V
        self.top_p_cpu_tensor[idx] = p
        return idx

    def remove_request(self, req_id):
        idx = self.req_id_to_index.pop(req_id)
        self._req_ids[idx] = None
        self._removed.append(idx)

    def _move(self, src, dst):
        rid = self._req_ids[src]
        self._req_ids[dst], self._req_ids[src] = rid, None
        self.req_id_to_index[rid] = dst
        for t in (self.temperature_cpu_tensor, self.top_k_cpu_tensor, self.top_p_cpu_tensor):
            t[dst] = t[src]

    def condense(self):
        """vLLM: the largest live rows fill the smallest holes below num_reqs; the tail is dropped."""
        n = self.num_reqs
        holes = sorted(i for i in self._removed if i < n)
        tail = [i for i in range(len(self._req_ids) - 1, n - 1, -1) if self._req_ids[i] is not None]
        for dst, src in zip(holes, tail):
            self._move(src, dst)
        del self._req_ids[n:]
        self._removed.clear()

    def swap_states(self, i, j):
        a, b = self._req_ids[i], self._req_ids[j]
        self._req_ids[i], self._req_ids[j] = b, a
        self.req_id_to_index[b], self.req_id_to_index[a] = i, j
        for t in (self.temperature_cpu_tensor, self.top_k_cpu_tensor, self.top_p_cpu_tensor):
            t[i], t[j] = t[j].clone(), t[i].clone()


def rand_params(rng, p_range=0.03):
    """Served-like mix: mostly sampled with an in-range top_k; a few greedy, unset (0 / >= vocab) or > KC top_k."""
    t = rng.choice([0.6, 0.6, 0.7, 1.0, 0.0])
    if rng.random() < p_range:
        k = rng.choice([0, V + 5, 33, 40, 100])
    else:
        k = rng.choice([20, 20, 20, 1, 5, 32])
    return t, k, rng.choice([1.0, 0.95, 0.8])


class Driver:
    """Random add / remove / swap / same-id param-change sequences keeping the live rows near `rows`."""

    def __init__(self, rows, seed, p_range=0.03):
        self.rng = random.Random(seed)
        self.rows, self.p_range = rows, p_range
        self.batch = FakeBatch()
        self.params = {}
        self.nid = 0
        self.ops = {"add": 0, "remove": 0, "swap": 0, "readd": 0}
        while self.batch.num_reqs < rows:
            self._add()

    def _add(self):
        rid = f"req-{self.nid}"
        self.nid += 1
        self.params[rid] = rand_params(self.rng, self.p_range)
        self.batch.add_request(rid, self.params[rid])
        self.ops["add"] += 1

    def step(self):
        b, rng = self.batch, self.rng
        live = [r for r in b.req_ids if r is not None]
        op = rng.random()
        if op < 0.10 and b.num_reqs > max(1, self.rows - 3):
            for rid in rng.sample(live, rng.randint(1, min(2, len(live) - 1) or 1)):
                b.remove_request(rid)
                self.ops["remove"] += 1
            b.condense()
        elif op < 0.18 and b.num_reqs < self.rows:
            self._add()
        elif op < 0.23 and b.num_reqs > 1:
            i, j = rng.sample(range(b.num_reqs), 2)
            b.swap_states(i, j)
            self.ops["swap"] += 1
        elif op < 0.26:
            # the same request id leaves and re-enters with other sampling params (an id reused by a client)
            rid = rng.choice(live)
            b.remove_request(rid)
            b.condense()
            self.params[rid] = rand_params(rng, self.p_range)
            b.add_request(rid, self.params[rid])
            self.ops["readd"] += 1


# ---- T-2 stage 0: the execute_model decision block ------------------------------------------------------------------
# as ast.unparse renders it (it parenthesises the comparison inside the "and" chain)
NOTE_TEST = ast.unparse(ast.parse("vps_skip_topk and vps_step and spec_decode_metadata is not None", mode="eval").body)


def _is_note(st):
    return isinstance(st, ast.If) and any(
        isinstance(c, ast.Call) and ast.unparse(c.func) == "_l6_ab.note" and "sampfb" in ast.unparse(c)
        for c in ast.walk(st))


@functools.lru_cache(maxsize=None)
def _decision_block():
    """(code, body, i, j): execute_model's statements from `vps_skip_topk = ...` through the sampfb note (or through
    the fallback block when no note follows). Built lazily so a broken tree fails a test rather than collection."""
    src = open(tr.__file__).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "execute_model")
    found = []
    for node in ast.walk(fn):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, st in enumerate(body):
                if isinstance(st, ast.Assign) and getattr(st.targets[0], "id", None) == "vps_skip_topk":
                    j = next((j for j in range(i, min(i + 6, len(body))) if _is_note(body[j])), i + 3)
                    found.append((body, i, j))
    assert len(found) == 1, f"vps_skip_topk assigned {len(found)} times in execute_model"
    body, i, j = found[0]
    code = compile(ast.Module(body=body[i:j + 1], type_ignores=[]), "<execute_model:vps_decision>", "exec")
    return code, body, i, j


class DecRunner:
    _topk_candidate_blocks = tr.TPUModelRunner._topk_candidate_blocks
    _vps_fallback_rows = tr.TPUModelRunner._vps_fallback_rows
    _real_topk = tr.TPUModelRunner._sampler_topk_candidates

    def __init__(self, batch, fb_max=8, tiegap=0, cand=False):
        self.input_batch = batch
        self.vocab_size = V
        self.speculative_config = NS(num_speculative_tokens=K)
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = fb_max, tiegap, cand
        self._vps_fb_path = fb_max > 0 or tiegap > 0 or cand        # vps_wiring.fallback_path()'s rule
        self.topk_calls = 0

    def _sampler_topk_candidates(self, s, e, grammar):
        self.topk_calls += 1
        return self._real_topk(s, e, grammar)


def decide(r, step, md, s, e):
    loc = {"self": r, "vps_step": step, "spec_decode_metadata": md, "start_index": s, "end_index": e}
    exec(_decision_block()[0], {"_l6_ab": tr._l6_ab}, loc)
    return loc["vps_chunk"], loc["vps_fb"]


def oracle(r, step, md, s, e):
    """b551e9b84's decision: the candidate test, overwritten on the fallback path by the fallback's own rule."""
    chunk = step and md is not None and r._real_topk(s, e, None) > 0
    fb = 0
    if step and md is not None and r._vps_fb_path:
        rows = r._vps_fallback_rows(s, e)
        chunk, fb = rows is not None, rows or 0
    return chunk, fb


CONFIGS = {"fb8": dict(fb_max=8), "fb8_tie1": dict(fb_max=8, tiegap=1), "cand": dict(fb_max=0, cand=True),
           "nofb": dict(fb_max=0)}


def _run_decisions(arm, lever, rows, cfg, seed, steps=240):
    """(decisions, topk calls on fallback steps, topk calls elsewhere, step kinds) for one arm."""
    arm(**ONE_WINDOW, **({"VLLM_TPU_L4_SAMP_FB_ORDER": lever} if lever else {}))
    d = Driver(rows, seed, p_range=min(0.3, 3.0 / rows))     # about 2-3 range requests per batch: F 0 / 4 / 8 / None
    r = DecRunner(d.batch, **CONFIGS[cfg])
    rng = random.Random(seed + 1)
    out, fb_calls, other_calls, kinds = [], 0, 0, {"fb": 0, "nostep": 0, "nomd": 0, "other": 0}
    for _ in range(steps):
        d.step()
        l6_ab.begin_step()
        step = rng.random() > 0.05
        md = object() if rng.random() > 0.05 else None
        n = d.batch.num_reqs
        for s, e in ((0, n), (n // 2, n)) if n > 1 else ((0, n),):
            before = r.topk_calls
            got = decide(r, step, md, s, e)
            assert type(got[0]) is bool and type(got[1]) is int
            assert got == oracle(r, step, md, s, e), (step, md is None, s, e, got)
            fb_step = step and md is not None and r._vps_fb_path
            kind = "fb" if fb_step else ("nostep" if not step else ("nomd" if md is None else "other"))
            kinds[kind] += 1
            if fb_step:
                fb_calls += r.topk_calls - before
            else:
                other_calls += r.topk_calls - before
            out.append(got)
    kinds["skips"] = engaged("sampfb.skip")
    kinds["lever_on"] = l6_ab.on("sampfb")
    return out, fb_calls, other_calls, kinds, d.ops


@pytest.mark.parametrize("cfg", ["fb8", "fb8_tie1", "cand"])
@pytest.mark.parametrize("rows", ROWS)
def test_sampfb_bitwise_and_census(ab, rows, cfg):
    """Fallback path: decisions equal (lever on vs off vs the b551e9b84 oracle) over random batches; the top-k scan
    runs once per fallback chunk with the lever off and never with it on (census)."""
    off, fb_off, oth_off, kinds, ops = _run_decisions(ab, None, rows, cfg, seed=rows * 7)
    on, fb_on, oth_on, kinds_on, _ = _run_decisions(ab, "1", rows, cfg, seed=rows * 7)
    print(f"\nSAMPFB_CENSUS rows={rows} cfg={cfg} chunks={len(off)} fb_chunks={kinds['fb']} "
          f"topk_calls_on_fb off={fb_off} on={fb_on} elsewhere off={oth_off} on={oth_on} "
          f"F={sorted(set(f for _, f in off))} vps={sum(c for c, _ in off)} ops={ops}")
    assert on == off
    # vacuity guard: the on arm's lever was on and skipped exactly the fallback chunks; the off arm's never engaged
    assert kinds_on["lever_on"] and kinds_on["skips"] == kinds["fb"] > 0
    assert not kinds["lever_on"] and kinds["skips"] == 0
    kinds_on = {k: v for k, v in kinds_on.items() if k not in ("skips", "lever_on")}
    assert kinds_on == {k: v for k, v in kinds.items() if k not in ("skips", "lever_on")}
    assert kinds["fb"] > 0 and kinds["nostep"] > 0 and kinds["nomd"] > 0
    assert fb_off == kinds["fb"] and fb_on == 0
    assert oth_on == oth_off
    assert ops["readd"] > 0 and ops["swap"] > 0 and ops["remove"] > 0
    if cfg == "fb8" and rows >= 16:
        assert {0, 4, 8} <= set(f for _, f in off) and any(not c for c, _ in off)


@pytest.mark.parametrize("rows", ROWS)
def test_sampfb_non_fallback_path_keeps_the_scan(ab, rows):
    """Without the fallback path the candidate test decides vps_chunk, so the lever must not skip it."""
    off, fb_off, oth_off, kinds, _ = _run_decisions(ab, None, rows, "nofb", seed=rows * 11)
    on, fb_on, oth_on, kinds_on, _ = _run_decisions(ab, "1", rows, "nofb", seed=rows * 11)
    assert kinds_on["lever_on"] and kinds_on["skips"] == 0         # the lever was on (not vacuous) and skipped nothing
    assert kinds["fb"] == 0 and on == off and oth_on == oth_off == kinds["other"]
    assert any(c for c, _ in off) and any(not c for c, _ in off)


def test_sampfb_windows_and_engagement(ab):
    """ab:1 with 20-step windows: the scan is skipped exactly in class-1 windows; each window line counts its skips
    (sampfb.skip, only for VPS fallback chunks); decisions equal the oracle throughout."""
    lines = ab(VLLM_TPU_L4_SAMP_FB_ORDER="ab:1", VLLM_TPU_L6_AB_STEPS="20")
    d = Driver(16, seed=5)
    r = DecRunner(d.batch, fb_max=8)
    want_skips = {}
    for j in range(1, 101):
        d.step()
        l6_ab.begin_step()
        k = j // 20
        step = j % 7 != 0                              # a non-VPS step now and then
        before = r.topk_calls
        assert decide(r, step, object(), 0, d.batch.num_reqs) == oracle(r, step, object(), 0, d.batch.num_reqs)
        called = r.topk_calls - before
        on = l6_ab.on("sampfb")
        assert on == (k % 2 == 1)
        assert called == (0 if (on or not step) else 1)
        if on and step:
            want_skips[k] = want_skips.get(k, 0) + 1
    got = {}
    for ln in lines:
        if ln.startswith("q38 L6AB role=worker window="):
            f = dict(kv.split("=", 1) for kv in ln.split()[2:])
            counts = dict(c.split("=") for c in f["counts"].split(",")) if f["counts"] != "-" else {}
            got[int(f["window"])] = (f["on"], int(counts.get("sampfb.skip", 0)))
    print("\nSAMPFB_WINDOWS", got)
    assert sorted(got) == [0, 1, 2, 3, 4]
    for k, (on, n) in got.items():
        assert on == ("sampfb" if k % 2 else "-") and n == want_skips.get(k, 0)
    assert sum(n for _, n in got.values()) > 0


def test_sampfb_block_sits_between_the_layout_plan_and_the_logits():
    """The decision block (5 statements) follows the layout plan and precedes the logits dispatch, and vps_chunk /
    vps_fb are consumed as on b551e9b84."""
    _, body, i, j = _decision_block()
    stmts = [ast.unparse(s) for s in body[i:j + 1]]
    assert len(stmts) == 5, stmts
    assert stmts[0] == "vps_skip_topk = self._vps_fb_path and _l6_ab.on('sampfb')"
    assert ast.unparse(body[i - 1]) == "layout_plan = self._last_sequence_layout_plan"
    nxt = j + 1
    if isinstance(body[nxt], ast.Assign) and getattr(body[nxt].targets[0], "id", None) == "vfy_defer":
        # q38 L4 v5 composition (V5COMP-BUILD; tests only): VFY-ONE's defer gate (tests/l4vfy) reads vps_chunk
        # after this block and sits between the sampfb note and the logits dispatch.
        nxt += 1
    assert ast.unparse(body[nxt].test) == "self.is_pooling_model"
    assert stmts[4] == f"if {NOTE_TEST}:\n    _l6_ab.note('sampfb', 'skip')"
    src = open(tr.__file__).read()
    assert src.count("vps_fb_list.append(int(vps_fb) if vps_chunk else 0)") == 1
    assert src.count('_l6_ab.on("sampfb")') == 1 and src.count('_l6_ab.note("sampfb", "skip")') == 1


# ---- registration ---------------------------------------------------------------------------------------------------
LEVERS = {"sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER", "sampsnap": "VLLM_TPU_L4_SAMP_SNAP",
          "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC"}


@pytest.mark.parametrize("lever,env", sorted(LEVERS.items()))
def test_lever_registered_host_only(ab, lever, env):
    """In l6_ab.LEVERS, parsed as 1 / ab:<c>+<c> (the served ','-free form), registered in envs.py and in the
    compile-key ignore list."""
    from vllm_torchtpu import envs

    assert l6_ab.LEVERS[lever] == env
    assert env in envs.environment_variables and env in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    ab(**{env: "1"})
    l6_ab.begin_step()
    assert l6_ab.ACTIVE and l6_ab.on(lever)
    ab(**{env: "ab:1+3", "VLLM_TPU_L6_AB_STEPS": "20", "VLLM_TPU_L6_AB_CLASSES": "4"})
    assert l6_ab._cfg[lever] == (False, frozenset({1, 3}))
    ab()
    l6_ab.begin_step()
    assert not l6_ab.on(lever)


# ---- T-2 stage 1: SAMP-SNAP -------------------------------------------------------------------------------------------
FULLV, LOCALV, TP = 2048, 256, 2      # the head's full / local widths only size the draws here
MAX_REQS = 256


class SnapRunner:
    _upload_keyed = tr.TPUModelRunner._upload_keyed
    _vps_fallback_rows = tr.TPUModelRunner._vps_fallback_rows
    _vps_verify_fb = tr.TPUModelRunner._vps_verify_fb
    _log_vps_fallback = tr.TPUModelRunner._log_vps_fallback
    _sample_spec_verify_vps = tr.TPUModelRunner._sample_spec_verify_vps
    _real_padded = tr.TPUModelRunner._padded_sampling_params
    _real_upload = tr.TPUModelRunner._upload_unless_unchanged

    def __init__(self, batch, fb_max=8, tiegap=0, cand=True):
        self.input_batch = batch
        self.vocab_size = V
        self.device = torch.device("cpu")
        self.speculative_config = NS(num_speculative_tokens=K)
        self.rejection_sampler = NS(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0]), synthetic_mode=True)
        self._vps_head = NS(full_vocab=FULLV, tp_size=TP, shard_offset=0, shard_width=LOCALV)
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = fb_max, tiegap, cand
        self._vps_fb_path = True
        self._vps_fb_stats, self._vps_fb_chunks, self._vps_checks_left = None, 0, 0
        self._h2d_reuse = {}
        self._samp_snap = l4_vpshost.SampSnap()
        self._vps_fb_async = l4_vpshost.AsyncCounts()
        self.calls = {"padded": 0, "nactive": 0}

    def _padded_sampling_params(self, *a):
        self.calls["padded"] += 1
        return self._real_padded(*a)

    def _upload_unless_unchanged(self, slot, host):
        if slot == "vps_num_active":
            self.calls["nactive"] += 1
        return self._real_upload(slot, host)


@pytest.fixture
def snapenv(ab, monkeypatch):
    """The compiled verify records its inputs; vps_wiring.range_rows counts F computations."""
    rec, frows = [], [0]

    def fake_fb(*args):
        rec.append(args)
        return torch.zeros(4, dtype=torch.int32), torch.tensor([1, 1, 0], dtype=torch.int32)

    real_rr = tr.vps_wiring.range_rows

    def counting_rr(*a, **k):
        frows[0] += 1
        return real_rr(*a, **k)

    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", fake_fb)
    monkeypatch.setattr(tr.vps_wiring, "range_rows", counting_rr)
    return NS(arm=ab, rec=rec, frows=frows)


def pad_of(n):
    return tr._get_padded_num_reqs_with_upper_limit(n, MAX_REQS)


def make_md(P):
    return NS(bonus_logits_indices=torch.arange(P, dtype=torch.int32) * (K + 1) + K,
              target_logits_indices=torch.arange(P * K, dtype=torch.int32),
              draft_token_ids=torch.zeros(P * K, dtype=torch.int32),
              draft_lengths=torch.full((P,), K, dtype=torch.int32),
              segment_ids=torch.arange(P, dtype=torch.int32).repeat_interleave(K),
              group_indices=torch.arange(K, dtype=torch.int32).repeat(P))


def same(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
                and a.storage_offset() == b.storage_offset() and a.device == b.device and torch.equal(a, b))
    return type(a) is type(b) and a == b


def verify_step(r, env, n, P, dtype, seed, s=0):
    """execute_model's fallback F for the chunk [s, n), then the chunk's VPS verify; returns (F, compiled inputs)."""
    F = r._vps_fallback_rows(s, n)
    if F is None:
        return None, None
    g = torch.Generator().manual_seed(seed)
    before = len(env.rec)
    r._sample_spec_verify_vps(torch.zeros(P * (K + 1), LOCALV, dtype=dtype), make_md(P), s, n, g, F)
    assert len(env.rec) == before + 1
    return F, env.rec[-1] + (g.get_state(),)


def run_snap_arm(env, levers, rows, seed, steps=200, reuse="1", fb_max=8, cand=True):
    env.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE=reuse, **ONE_WINDOW, **levers)
    d = Driver(rows, seed)
    r = SnapRunner(d.batch, fb_max=fb_max, cand=cand)
    rng = random.Random(seed + 3)
    out, keys = [], {"F": set(), "params": set(), "n": set(), "lookups": 0, "verifies": 0}
    adds = lambda: d.ops["add"] + d.ops["readd"]                                  # noqa: E731
    f0 = env.frows[0]
    for i in range(steps):
        d.step()
        l6_ab.begin_step()
        n = d.batch.num_reqs
        P = pad_of(n) * (2 if rng.random() < 0.1 and pad_of(n) < MAX_REQS else 1)
        dtype = torch.float32 if rng.random() < 0.1 else torch.bfloat16
        ids = tuple(d.batch.req_ids[:n])
        F, got = verify_step(r, env, n, P, dtype, seed * 1000 + i)
        keys["lookups"] += 1
        keys["F"].add((adds(), ids, n))
        if F is not None:
            keys["verifies"] += 1
            keys["params"].add((adds(), ids, n, P, dtype))
            keys["n"].add(n)
        out.append((F, got))
    keys["hit"], keys["miss"] = engaged("sampsnap.hit"), engaged("sampsnap.miss")
    return out, r, keys, env.frows[0] - f0, d.ops


@pytest.mark.parametrize("reuse", ["1", "0"])
@pytest.mark.parametrize("rows", ROWS)
def test_sampsnap_bitwise_and_census(snapenv, rows, reuse):
    """Random add / remove / swap / same-id re-add sequences, padding doubled on 10 % of steps, f32 logits on 10 %:
    F and every input of the compiled verify are the same with sampsnap on as off (SAMPLING_PARAMS_REUSE on and off).
    Census: off builds F / params / num_active on every lookup; on builds each exactly once per distinct
    (batch generation, request ids, rows[, padded rows, dtype]) key, and num_active once per value."""
    off, roff, koff, foff, ops = run_snap_arm(snapenv, {}, rows, seed=rows * 13, reuse=reuse)
    on, ron, kon, fon, _ = run_snap_arm(snapenv, {"VLLM_TPU_L4_SAMP_SNAP": "1"}, rows, seed=rows * 13, reuse=reuse)
    assert len(on) == len(off)
    for i, ((fa, a), (fb, b)) in enumerate(zip(off, on)):
        assert fa == fb, (i, fa, fb)
        if a is not None:
            assert len(a) == len(b) and all(same(x, y) for x, y in zip(a, b)), (i, [j for j, (x, y) in enumerate(zip(a, b)) if not same(x, y)])
    hits = kon["verifies"] - ron.calls["padded"]
    print(f"\nSAMPSNAP_CENSUS rows={rows} reuse={reuse} steps={len(off)} verifies={koff['verifies']} "
          f"F_builds off={foff} on={fon} (keys {len(kon['F'])}) params_builds off={roff.calls['padded']} "
          f"on={ron.calls['padded']} (keys {len(kon['params'])}) nactive_builds off={roff.calls['nactive']} "
          f"on={ron.calls['nactive']} (values {len(kon['n'])}) hits={hits} gen={ron._samp_snap.gen} ops={ops}")
    assert foff == koff["lookups"] and roff.calls["padded"] == roff.calls["nactive"] == koff["verifies"]
    assert fon == len(kon["F"]) and ron.calls["padded"] == len(kon["params"]) and ron.calls["nactive"] == len(kon["n"])
    assert hits > 0 and ops["readd"] > 0 and ops["swap"] > 0
    # vacuity guard: the engagement counts agree with the builds (on) and are absent (off)
    assert kon["hit"] == hits > 0 and kon["miss"] == ron.calls["padded"] and koff["hit"] == koff["miss"] == 0
    assert roff._samp_snap.gen == 0 and not roff._samp_snap._d                 # lever off never touches the snapshot


def _fixed(env, levers, rows=16, reuse="1"):
    """A fixed batch of `rows` requests with varied in-range params and one unset top_k (row 1): F = 4."""
    env.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE=reuse, **ONE_WINDOW, **levers)
    d = NS(batch=FakeBatch())
    for i in range(rows):
        d.batch.add_request(f"f{i}", ((0.6, 0.7, 1.0, 0.0)[i % 4], 0 if i == 1 else (20, 5, 32, 1)[i % 4],
                                      (0.95, 1.0, 0.8)[i % 3]))
    return d, SnapRunner(d.batch)


@pytest.mark.parametrize("seq", [(8, 16, 8, 32, 16, 8), (16, 32, 64, 16)])
def test_sampsnap_same_ids_different_padding(snapenv, seq):
    """Same request ids, the padded row count changes: each call equals the lever-off call (padding is in the key)."""
    res = {}
    for name, lev in (("off", {}), ("on", {"VLLM_TPU_L4_SAMP_SNAP": "1"})):
        d, r = _fixed(snapenv, lev, rows=6)
        res[name] = []
        for i, P in enumerate(seq):
            l6_ab.begin_step()
            res[name].append(verify_step(r, snapenv, 6, P, torch.bfloat16, 50 + i))
        res[name + "_eng"] = (engaged("sampsnap.hit"), engaged("sampsnap.miss"), r.calls["padded"])
    # vacuity guard: on served repeats from the snapshot (one build per distinct padding), off built every call
    assert res["on_eng"] == (len(seq) - len(set(seq)), len(set(seq)), len(set(seq))) and res["off_eng"] == (0, 0, len(seq))
    for (fa, a), (fb, b) in zip(res["off"], res["on"]):
        assert fa == fb and all(same(x, y) for x, y in zip(a, b))
    assert len({r[1][4].shape[0] for r in res["on"]}) == len(set(seq))      # temps follows the padding


def test_sampsnap_same_ids_changed_params(snapenv):
    """Same request ids in the same order, one request re-added with other sampling params (remove the last row and
    add it back: it lands on the same row): the snapshot is dropped and the new params reach the verify."""
    res = {}
    for name, lev in (("off", {}), ("on", {"VLLM_TPU_L4_SAMP_SNAP": "1"})):
        d, r = _fixed(snapenv, lev, rows=8)
        res[name] = []
        for i in range(6):
            l6_ab.begin_step()
            if i in (2, 4):
                rid = d.batch.req_ids[d.batch.num_reqs - 1]
                d.batch.remove_request(rid)
                d.batch.condense()
                d.batch.add_request(rid, (0.3 + 0.1 * i, 7 + i, 0.5))
            res[name].append((tuple(d.batch.req_ids), verify_step(r, snapenv, 8, 8, torch.bfloat16, 70 + i)))
        res[name + "_eng"] = (engaged("sampsnap.hit"), engaged("sampsnap.miss"))
    # vacuity guard: 6 lookups, rebuilt after each of the 2 re-adds: 3 misses, 3 hits; off notes nothing
    assert res["on_eng"] == (3, 3) and res["off_eng"] == (0, 0)
    assert len({ids for ids, _ in res["on"]}) == 1
    for (ia, (fa, a)), (ib, (fb, b)) in zip(res["off"], res["on"]):
        assert ia == ib and fa == fb and all(same(x, y) for x, y in zip(a, b))
    temps = [x[1][4] for _, x in res["on"]]
    assert not torch.equal(temps[1], temps[2]) and not torch.equal(temps[3], temps[4])


def test_sampsnap_dtype_in_key(snapenv):
    """Same ids and padding, bf16 then f32 then bf16 logits: temps takes the logits dtype every time."""
    res = {}
    for name, lev in (("off", {}), ("on", {"VLLM_TPU_L4_SAMP_SNAP": "1"})):
        d, r = _fixed(snapenv, lev, rows=8)
        res[name] = []
        for i, dt in enumerate((torch.bfloat16, torch.float32, torch.bfloat16, torch.float32)):
            l6_ab.begin_step()                     # the lever is read per step: off before the first begin_step
            res[name].append(verify_step(r, snapenv, 8, 8, dt, 90 + i))
        res[name + "_eng"] = (engaged("sampsnap.hit"), engaged("sampsnap.miss"), r.calls["padded"])
    # vacuity guard (s1v1 m2 j-4a352f1a: this test ran with the lever off): on built once per dtype and hit twice
    assert res["on_eng"] == (2, 2, 2) and res["off_eng"] == (0, 0, 4)
    for (fa, a), (fb, b) in zip(res["off"], res["on"]):
        assert fa == fb and all(same(x, y) for x, y in zip(a, b))
    assert [x[1][4].dtype for x in res["on"]] == [torch.bfloat16, torch.float32] * 2


def test_sampsnap_new_input_batch_object(snapenv):
    """KV-cache re-initialisation builds a new InputBatch: same request ids, other params; the snapshot follows it."""
    snapenv.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE="1", VLLM_TPU_L4_SAMP_SNAP="1", **ONE_WINDOW)
    l6_ab.begin_step()
    b1, b2 = FakeBatch(), FakeBatch()
    for i in range(8):
        b1.add_request(f"r{i}", (0.6, 20, 0.95))
        b2.add_request(f"r{i}", (0.9, 5 if i else 0, 0.8))         # r0 unset top_k: one range request, F = 4
    r = SnapRunner(b1)
    f1, a1 = verify_step(r, snapenv, 8, 8, torch.bfloat16, 1)
    r.input_batch = b2
    f2, a2 = verify_step(r, snapenv, 8, 8, torch.bfloat16, 2)
    # vacuity guard: both lookups went through the snapshot (two misses: b1, then the re-watched b2)
    assert (engaged("sampsnap.miss"), r._samp_snap.gen) == (2, 2)
    ref = SnapRunner(b2)
    snapenv.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE="1")
    l6_ab.begin_step()
    f3, a3 = verify_step(ref, snapenv, 8, 8, torch.bfloat16, 2)
    assert (f1, f2, f3) == (0, 4, 4) and all(same(x, y) for x, y in zip(a2, a3))


def test_sampsnap_fallback_settings_in_key(snapenv):
    """F is kept per (fb_max, tiegap): changing them on the same batch gives the lever-off F."""
    snapenv.arm(VLLM_TPU_L4_SAMP_SNAP="1", **ONE_WINDOW)
    l6_ab.begin_step()
    b = FakeBatch()
    for i in range(8):
        b.add_request(f"r{i}", (0.6, 0 if i < 2 else 20, 0.95))              # two range requests: 8 rows
    r = SnapRunner(b, fb_max=8)
    got = [r._vps_fallback_rows(0, 8)]
    r._vps_fb_max = 4
    got.append(r._vps_fallback_rows(0, 8))
    r._vps_fb_max, r._vps_tiegap = 16, 1
    got.append(r._vps_fallback_rows(0, 8))
    assert got == [8, None, 16]
    # vacuity guard: F came through the snapshot (one entry, three fallback settings, each computed once)
    assert l6_ab.on("sampsnap") and len(r._samp_snap._d) == 1 and snapenv.frows[0] == 3
    assert sorted(map(str, next(iter(r._samp_snap._d.values())).rows)) == ["(16, 1)", "(4, 0)", "(8, 0)"]
    assert r._vps_fallback_rows(0, 8) == 16 and snapenv.frows[0] == 3   # a hit computes nothing


def test_sampsnap_check_mode_catches_a_stale_snapshot(snapenv, monkeypatch):
    """A parameter written behind add_request's back (never done by vLLM) is served stale without the check mode;
    with VLLM_TPU_L4_SAMP_SNAP_CHECK=1 the hit is checked, counted stale, logged and rebuilt (= lever off)."""
    errs = []
    monkeypatch.setattr(l4_vpshost.logger, "error", lambda fmt, *a: errs.append(fmt % a))
    out = {}
    for check in ("0", "1"):
        snapenv.arm(VLLM_TPU_L4_SAMP_SNAP="1", VLLM_TPU_L4_SAMP_SNAP_CHECK=check, **ONE_WINDOW)
        l6_ab.begin_step()
        b = FakeBatch()
        for i in range(8):
            b.add_request(f"r{i}", (0.6, 20, 0.95))
        r = SnapRunner(b)
        verify_step(r, snapenv, 8, 8, torch.bfloat16, 1)
        b.temperature_cpu_tensor[3] = 0.25                                      # behind the snapshot's back
        f, a = verify_step(r, snapenv, 8, 8, torch.bfloat16, 2)
        out[check] = (a[4].clone(), r._samp_snap.checks, r._samp_snap.stale, l6_ab._state["counts"].copy())
        assert engaged("sampsnap.miss") >= 1 and (check == "1" or engaged("sampsnap.hit") == 1)   # vacuity guard
    stale_t, fresh_t = out["0"][0], out["1"][0]
    assert float(stale_t[3, 0]) == float(torch.tensor(0.6, dtype=torch.bfloat16))
    assert float(fresh_t[3, 0]) == float(torch.tensor(0.25, dtype=torch.bfloat16))
    assert out["0"][1:3] == (0, 0) and out["1"][1] >= 1 and out["1"][2] == 1
    assert out["1"][3].get("sampsnap.stale") == 1 and len(errs) == 1 and "stale snapshot" in errs[0]


def test_sampsnap_check_mode_follows_l6_check_allowed(snapenv, monkeypatch):
    """q38-l6-next bounds CHECK modes by l6_ab.check_allowed() (VLLM_TPU_L6_CHECK_STEPS); when it says no, no check."""
    monkeypatch.setattr(l6_ab, "check_allowed", lambda: False, raising=False)
    snapenv.arm(VLLM_TPU_L4_SAMP_SNAP="1", VLLM_TPU_L4_SAMP_SNAP_CHECK="1", **ONE_WINDOW)
    l6_ab.begin_step()
    b = FakeBatch()
    for i in range(8):
        b.add_request(f"r{i}", (0.6, 20, 0.95))
    r = SnapRunner(b)
    for i in range(4):
        verify_step(r, snapenv, 8, 8, torch.bfloat16, i)
    assert r._samp_snap.checks == 0 and r._samp_snap.hits >= 3
    assert engaged("sampsnap.hit") == 3 and engaged("sampsnap.check") == 0        # vacuity guard: hits, no checks


def test_sampsnap_windows_and_engagement(snapenv):
    """ab:1 with 20-step windows: params are built on every verify in class-0 windows and served from the snapshot in
    class-1 windows; each class-1 window line counts sampsnap.hit + sampsnap.miss = its verifies, class-0 lines none."""
    lines = snapenv.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE="1", VLLM_TPU_L4_SAMP_SNAP="ab:1", VLLM_TPU_L6_AB_STEPS="20")
    d = Driver(16, seed=9)
    r = SnapRunner(d.batch)
    per = {}
    for j in range(1, 101):
        d.step()
        l6_ab.begin_step()
        k = j // 20
        n = d.batch.num_reqs
        before = r.calls["padded"]
        F, _ = verify_step(r, snapenv, n, pad_of(n), torch.bfloat16, j)
        if F is not None:
            p = per.setdefault(k, [0, 0])
            p[0] += 1
            p[1] += r.calls["padded"] - before
        assert l6_ab.on("sampsnap") == (k % 2 == 1)
    got = {}
    for ln in lines:
        if ln.startswith("q38 L6AB role=worker window="):
            f = dict(kv.split("=", 1) for kv in ln.split()[2:])
            counts = dict(c.split("=") for c in f["counts"].split(",")) if f["counts"] != "-" else {}
            got[int(f["window"])] = (f["on"], int(counts.get("sampsnap.hit", 0)), int(counts.get("sampsnap.miss", 0)))
    print("\nSAMPSNAP_WINDOWS", got, per)
    assert sorted(got) == [0, 1, 2, 3, 4]
    for k, (on, hit, miss) in got.items():
        verifies, builds = per.get(k, [0, 0])
        if k % 2:
            assert on == "sampsnap" and hit + miss == verifies and miss == builds and hit > 0
        else:
            assert on == "-" and hit == miss == 0 and builds == verifies


def test_sampsnap_runner_sites():
    """The lever is read at exactly three sites: _vps_fallback_rows (F), _sample_spec_verify_vps (the parameter
    uploads, fallback path only) and _vps_verify_fb (num_active); never inside a compiled function."""
    src = open(tr.__file__).read()
    tree = ast.parse(src)
    sites = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for c in ast.walk(fn):
                if isinstance(c, ast.Call) and ast.unparse(c) == "_l6_ab.on('sampsnap')":
                    sites.append(fn.name)
    # q38 L4 v5 composition (V5COMP-BUILD; tests only): VFY-ONE's own copies of the D83 chain carry the same lever
    # branch (786f7bac2: `_vfy_verify` parameter + num_active uploads, `_vfy_chain` num_active upload); they are
    # proved bitwise and site for site in tests/q38_l4_v5c/test_v5c_compose.py.
    vfy_sites = ["_vfy_chain", "_vfy_verify", "_vfy_verify"]
    assert sorted(sites) == sorted(["_sample_spec_verify_vps", "_vps_fallback_rows", "_vps_verify_fb"] + vfy_sites)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sample_spec_verify_vps")
    assert "if self._vps_fb_path and _l6_ab.on('sampsnap'):" in ast.unparse(fn)
    assert src.count("self._samp_snap = _l4_vpshost.SampSnap()") == 1


# ---- T-2 stage 0: sampled tokens (SAMPFB-TOKEN, L1's request before v5 windows sampfb) -------------------------------
TV = 10240      # 64 VP_BLOCKs of 160 and 80 CANDIDATE_BLOCKs of 128: both the VPS candidates and the top-k scan work


class TokRunner(SnapRunner):
    """The decision block, then the runner's real _sample_spec_verify_chunk -> _sample_spec_verify_vps ->
    _vps_verify_fb with the real _vps_verify_fb_impl run eagerly (TP1 head, so the all-gather is the identity)."""

    _topk_candidate_blocks = tr.TPUModelRunner._topk_candidate_blocks
    _real_topk = tr.TPUModelRunner._sampler_topk_candidates
    _sample_spec_verify_chunk = tr.TPUModelRunner._sample_spec_verify_chunk

    def __init__(self, batch, cand):
        super().__init__(batch, fb_max=8, cand=cand)
        self.vocab_size = TV
        self._vps_head = NS(full_vocab=TV, tp_size=1, shard_offset=0, shard_width=TV)
        self.topk_calls = 0

    def _sampler_topk_candidates(self, s, e, grammar):
        self.topk_calls += 1
        return self._real_topk(s, e, grammar)


@pytest.fixture
def tokenv(ab, monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", tr._vps_verify_fb_impl)
    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x)
    return NS(arm=ab)


def tok_batch(rows, n_range, rng):
    """`rows` requests with served-like in-range params, `n_range` of them sampled with top_k unset or > KC."""
    b = FakeBatch()
    idx = set(rng.sample(range(rows), n_range))
    for i in range(rows):
        t = rng.choice([0.6, 0.7, 1.0]) if i in idx else rng.choice([0.6, 0.7, 1.0, 0.0])
        k = rng.choice([0, TV + 5, 40, 100]) if i in idx else rng.choice([20, 20, 1, 5, 32])
        b.add_request(f"t{i}", (t, k, rng.choice([1.0, 0.95, 0.8])))
    return b


def tok_problem(P, seed):
    g = torch.Generator().manual_seed(seed)
    N, T = P * (K + 1), P * K
    logits = torch.randn(N, TV, generator=g) * 3
    logits.scatter_add_(1, torch.randint(0, TV, (N, 6), generator=g), torch.rand(N, 6, generator=g) * 8)
    rows = torch.arange(N).view(P, K + 1)
    md = NS(bonus_logits_indices=rows[:, K].contiguous(), target_logits_indices=rows[:, :K].reshape(-1).contiguous(),
            draft_token_ids=torch.randint(0, TV, (T,), generator=g).to(torch.int32),
            draft_lengths=torch.full((P,), K, dtype=torch.int32),
            segment_ids=torch.arange(P).repeat_interleave(K).to(torch.int32),
            group_indices=torch.arange(K).repeat(P).to(torch.int32))
    return logits.bfloat16(), md


def run_tok_arm(env, lever, rows, cand, steps=24):
    env.arm(VLLM_TPU_SAMPLING_PARAMS_REUSE="1", **ONE_WINDOW, **({"VLLM_TPU_L4_SAMP_FB_ORDER": lever} if lever else {}))
    rng = random.Random(rows * 31 + cand)
    gen = torch.Generator().manual_seed(4321)                  # the runner's sampling generator, shared by the steps
    recs, calls = [], 0
    for i in range(steps):
        l6_ab.begin_step()
        r = TokRunner(tok_batch(rows, (0, 1, 2, 3)[i % 4], rng), cand)        # F = 0 / 4 / 8 / None in turn
        logits, md = tok_problem(pad_of(rows), rows * 1000 + i)
        vps_chunk, vps_fb = decide(r, True, md, 0, rows)
        toks = None                     # vps_chunk False: today's (non-VPS) verify, a different program
        if vps_chunk:
            toks = r._sample_spec_verify_chunk(logits, md, None, None, 0, rows, False, gen, False, vps=True,
                                               vps_fallback=vps_fb)
        calls += r.topk_calls
        recs.append((vps_chunk, vps_fb, toks, gen.get_state().clone()))
    return recs, calls, engaged("sampfb.skip")


@pytest.mark.parametrize("cand", [True, False])
@pytest.mark.parametrize("rows", (4, 8, 16))
def test_sampfb_sampled_tokens_bitwise(tokenv, rows, cand):
    """VPS fallback steps with 0 / 1 / 2 / 3 range requests in turn (F = 0 / 4 / 8 / None), one sampling generator
    across the steps (fixed seed), per-candidate and full-vocabulary draws: the sampled tokens [B, K+1] and the
    generator state after every step are bitwise equal with sampfb on as off, and so is the decision. The on arm
    really skipped the scan on every step (vacuity guard)."""
    off, calls_off, skip_off = run_tok_arm(tokenv, None, rows, cand)
    on, calls_on, skip_on = run_tok_arm(tokenv, "1", rows, cand)
    tok_bad = [i for i, (a, b) in enumerate(zip(off, on))
               if not ((a[2] is None and b[2] is None) or (a[2] is not None and b[2] is not None and same(a[2], b[2])))
               or not torch.equal(a[3], b[3])]
    vps = sum(1 for a in off if a[0])
    with_f = sum(1 for a in off if a[0] and a[1] > 0)
    distinct = len({tuple(a[2].reshape(-1).tolist()) for a in off if a[2] is not None})
    print(f"\nSAMPFBTOK_CENSUS rows={rows} cand={cand} steps={len(off)} vps={vps} vps_with_F={with_f} "
          f"F={sorted(set(a[1] for a in off))} distinct_token_rows={distinct} topk_calls off={calls_off} on={calls_on} "
          f"skips on={skip_on} token_mismatch_steps={tok_bad}")
    assert tok_bad == []
    assert [(a[0], a[1]) for a in off] == [(b[0], b[1]) for b in on]
    assert calls_off == len(off) and skip_off == 0 and calls_on == 0 and skip_on == len(on)     # vacuity guard
    assert vps == 3 * len(off) // 4 and with_f == len(off) // 2 and distinct == vps


# ---- T-1(i): STATS-ASYNC ------------------------------------------------------------------------------------------------
class Dev(torch.Tensor):
    """A device tensor stand-in on CPU: every read that would block the host on the device is counted in `reads`;
    `.to("cpu", non_blocking=True)` returns a plain copy taken at that moment (the in-flight copy's contents)."""

    reads: list = []
    BLOCKING = {torch.Tensor.tolist, torch.Tensor.item, torch.Tensor.numpy, torch.Tensor.cpu, torch.Tensor.__bool__,
                torch.Tensor.__int__, torch.Tensor.__float__, torch.Tensor.__index__}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in cls.BLOCKING:
            cls.reads.append(getattr(func, "__name__", str(func)))
        elif func is torch.Tensor.to:
            dev = kwargs.get("device", next((a for a in args[1:] if isinstance(a, (str, torch.device))), None))
            if dev is not None and torch.device(dev).type == "cpu":
                if kwargs.get("non_blocking", False):
                    with torch._C.DisableTorchFunctionSubclass():
                        return args[0].clone()
                cls.reads.append("to_cpu_blocking")
        return super().__torch_function__(func, types, args, kwargs)


class FakeEvents:
    """l4_vpshost._event stand-in: each event reports ready after `lag` unready queries (or once synchronized)."""

    def __init__(self, lag):
        self.lag, self.made = lag, []

    def __call__(self):
        ev = NS(left=self.lag, queries=0)

        def query():
            ev.queries += 1
            if ev.left <= 0:
                return True
            ev.left -= 1
            return False

        ev.query = query
        ev.synchronize = lambda: setattr(ev, "left", 0)
        self.made.append(ev)
        return ev


class StatsRunner:
    _vps_verify_fb = tr.TPUModelRunner._vps_verify_fb
    _log_vps_fallback = tr.TPUModelRunner._log_vps_fallback
    _upload_unless_unchanged = tr.TPUModelRunner._upload_unless_unchanged

    def __init__(self):
        self.device = torch.device("cpu")
        self._vps_head = NS(full_vocab=FULLV, tp_size=TP, shard_offset=0, shard_width=LOCALV)
        self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = 8, 0, False
        self._vps_fb_stats, self._vps_fb_chunks = None, 0
        self._vps_fb_async = l4_vpshost.AsyncCounts()
        self._samp_snap = l4_vpshost.SampSnap()
        self._h2d_reuse = {}


@pytest.fixture
def statsenv(ab, monkeypatch):
    """The compiled fallback verify returns per-chunk [flagged, covered, uncovered] counts as a Dev tensor; the runner's
    logger records VPS_FALLBACK lines; l4_vpshost._event is a FakeEvents (set per test with .events(lag))."""
    lines = []
    rng = random.Random(17)

    def fake_fb(*args):
        f = rng.randint(0, 6)
        c = rng.randint(0, f)
        return torch.zeros(4, dtype=torch.int32), torch.tensor([f, c, f - c], dtype=torch.int32).as_subclass(Dev)

    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", fake_fb)
    monkeypatch.setattr(tr, "logger", NS(info=lambda fmt, *a: lines.append(fmt % a), warning=lambda *a, **k: None,
                                         warning_once=lambda *a, **k: None, info_once=lambda *a, **k: None))
    st = NS(arm=ab, lines=lines, rng=rng, ev=None)

    def events(lag):
        st.ev = FakeEvents(lag)
        monkeypatch.setattr(l4_vpshost, "_event", st.ev)

    st.events = events
    return st


def run_chunks(env, r, n, start=0):
    """n fallback chunks through the runner's real _vps_verify_fb; (VPS_FALLBACK lines, blocking reads) of the span."""
    l0, r0 = len(env.lines), len(Dev.reads)
    for _ in range(n):
        l6_ab.begin_step()
        r._vps_verify_fb(torch.zeros(4, LOCALV), (None,) * 15, 8, 0, None)
    return [ln for ln in env.lines[l0:] if ln.startswith("VPS_FALLBACK")], Dev.reads[r0:]


def stats_arm(env, levers, lag, chunks=1000, tail=6):
    env.arm(VLLM_TPU_VPS_STATS_EVERY="200", VLLM_TPU_L6_AB_STEPS="100000", **levers)
    env.rng.seed(17)
    env.events(lag)
    r = StatsRunner()
    del Dev.reads[:]
    lines, reads = run_chunks(env, r, chunks)
    more, reads2 = run_chunks(env, r, tail)
    return lines, lines + more, reads, reads2, r


@pytest.mark.parametrize("lag", [0, 1, 3])
def test_statsasync_same_lines_zero_blocking_reads(statsenv, lag):
    """1000 fallback chunks, VLLM_TPU_VPS_STATS_EVERY=200: lever off reads the counts 5 times with a blocking
    .tolist() (the census's positive control); lever on never blocks, and once each copy has landed (lag unready
    queries) the same 5 VPS_FALLBACK lines are printed, in order. Engagement: statsasync.issued = 5."""
    off, off_all, reads_off, _, _ = stats_arm(statsenv, {}, lag)
    on, on_all, reads_on, reads_on2, r = stats_arm(statsenv, {"VLLM_TPU_L4_VPS_STATS_ASYNC": "1"}, lag)
    print(f"\nSTATSASYNC_CENSUS rows=8 lag={lag} chunks=1000 lines off={len(off)} on={len(on)}+{len(on_all) - len(on)} "
          f"blocking_reads off={len(reads_off)} ({sorted(set(reads_off))}) on={len(reads_on) + len(reads_on2)} "
          f"events={len(statsenv.ev.made)} queries={sum(e.queries for e in statsenv.ev.made)}")
    assert len(off) == 5 and off_all == off and reads_off == ["tolist"] * 5
    assert reads_on == [] and reads_on2 == [] and on_all == off and not r._vps_fb_async.pending
    assert len(statsenv.ev.made) == 5 and l6_ab._state["counts"].get("statsasync.issued") == 5
    assert len(on) == 4 and on == off[:4]       # the window of chunk 1000 lands after chunk 1000 (in the tail)


def test_statsasync_line_waits_for_its_copy(statsenv):
    """A copy still in flight prints nothing; the chunk after it lands prints the line of ITS window (chunks=200 and
    the counts as of chunk 200), not the current one."""
    off, *_ = stats_arm(statsenv, {}, 0, chunks=200, tail=0)
    statsenv.arm(VLLM_TPU_VPS_STATS_EVERY="200", VLLM_TPU_L6_AB_STEPS="100000", VLLM_TPU_L4_VPS_STATS_ASYNC="1")
    statsenv.rng.seed(17)
    statsenv.events(10 ** 9)
    r = StatsRunner()
    a, _ = run_chunks(statsenv, r, 200)
    b, _ = run_chunks(statsenv, r, 30)
    assert a == [] and b == [] and len(r._vps_fb_async.pending) == 1
    statsenv.ev.made[0].left = 0                                            # the copy lands
    c, reads = run_chunks(statsenv, r, 1)
    assert c == off and reads == [] and not r._vps_fb_async.pending
    assert engaged("statsasync.issued") == 1                                # vacuity guard
    assert c[0].startswith("VPS_FALLBACK chunks=200 ")


def test_statsasync_windows_flush_before_a_blocking_line(statsenv):
    """ab:1 with 150-step windows: chunks 200 and 800 fall in on windows (issued, never landing by themselves),
    400 / 600 / 1000 in off windows (blocking). The off path first waits for and logs the copies still in flight, so
    the lines are the lever-off lines in order; the only blocking reads are the off windows' own .tolist()."""
    off, *_ = stats_arm(statsenv, {}, 0)
    statsenv.arm(VLLM_TPU_VPS_STATS_EVERY="200", VLLM_TPU_L6_AB_STEPS="150", VLLM_TPU_L4_VPS_STATS_ASYNC="ab:1")
    statsenv.rng.seed(17)
    statsenv.events(10 ** 9)
    r = StatsRunner()
    del Dev.reads[:]
    lines, reads = run_chunks(statsenv, r, 1000)
    assert lines == off and reads == ["tolist"] * 3 and len(statsenv.ev.made) == 2 and not r._vps_fb_async.pending


def test_statsasync_runner_sites():
    """statsasync is read once, in _vps_verify_fb; the blocking .tolist() sits only on its lever-off branch."""
    src = open(tr.__file__).read()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_vps_verify_fb")
    body = ast.unparse(fn)
    # q38 L4 v5 composition (V5COMP-BUILD; tests only): VFY-ONE's `_vfy_fb_tick` (its copy of this tail) carries the
    # same branch (786f7bac2); tests/q38_l4_v5c/test_v5c_compose.py proves the two tails statement for statement.
    tick = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_vfy_fb_tick")
    tbody = ast.unparse(tick)
    assert src.count('_l6_ab.on("statsasync")') == 2 and "_l6_ab.on('statsasync')" in body
    assert "_l6_ab.on('statsasync')" in tbody and tbody.count("self._vps_fb_stats.tolist()") == 1
    assert src.count("self._vps_fb_stats.tolist()") == 2 and body.count("self._vps_fb_stats.tolist()") == 1
    assert src.count("self._vps_fb_async = _l4_vpshost.AsyncCounts()") == 1
