# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VRNG-VPS x VFY-ONE (T1-NODEFER-VFY route (a); SAMP2-BUILD, helper of L4-E). CDK CPU pod only.

Real modules (PYTHONPATH=src), TP1 VPS head with a fake vocab shard (hidden @ W^T in bf16, as tests/l4vfy), the
runner's real methods bound to a fake runner, compiled programs replaced by their eager bodies.

1. `_vfy_one_rng_impl` (the VFY program drawing in-graph) is bitwise `_vfy_one_impl` fed with the counter's full
   draws (vps_rng.full_draws): tokens, accumulated stats, read-offset operands, seed outputs; it returns counter + 1
   and the AL counts of its own tokens. The same for `_vps_verify_rng_impl` with D83's operands against
   `_vps_verify_fb_impl` / `_vps_verify_impl`.
2. `_vfy_verify` with both flags: in "vfy" mode one composed call (no eager draw, no base VFY call, the generator
   untouched, the counter + 1) whose tokens / stats / read-offset buffer equal the base VFY step on the counter's
   draws; cold base key -> the D83 chain's VRNG program; no read-offset groups -> base draws (a never-warmed VRNG key);
   VRNG window off -> the base VFY program on eager draws, bitwise flag-off.
3. Composed CHECK (VLLM_TPU_VPS_VFY_ONE_CHECK): the chain's VRNG result serves, the composed program runs beside it on
   the same counter value, 0 mismatches; a broken composed program is counted.
4. Warm-up: `_vfy_precompile` warms the composed program beside every VFY call and registers ("vfy", key); a steady
   serving step's `_vfy_vrng_key` is among them; `_precompile_vps_vrng` registers the D83-chain keys with the
   read-offset part `_vrng_ro_key` computes in serving.
5. AOT census of a VFY step: eager RNG dispatches 3 -> 0 with both flags; the composed program has no RNG op.
"""

import contextlib
import os
from types import SimpleNamespace as NS

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_torchtpu.runner.tpu_runner as tr
from vllm_torchtpu.layers.adapter.sample import vps_rng
from vllm_torchtpu.runner import l4_agree, l4_samp2, l4_vpshost, vfy_one, vps_wiring

HERE = os.path.dirname(os.path.abspath(__file__))
K, V, HID = 3, 10240, 64
CPU = torch.device("cpu")


def test_imports_the_tree_under_test():
    root = os.path.realpath(os.environ.get("Q38_SAMP2_EXPECT_ROOT") or os.path.join(HERE, "..", "..", "src"))
    for mod in (tr, vps_rng, l4_samp2, vfy_one, vps_wiring):
        assert os.path.realpath(mod.__file__).startswith(root + os.sep), mod.__file__


class Proc:
    def _get_logits(self, hidden, head, bias, skip_gather=False):
        assert bias is None and skip_gather
        return torch.nn.functional.linear(hidden, head.weight)


_gw = torch.Generator().manual_seed(4242)
HEAD = NS(weight=(torch.randn(V, HID, generator=_gw) * 0.5).bfloat16())
PROC = Proc()


class AlwaysWarm(set):
    def __contains__(self, k):
        return True


def seed_body(tokens, bump):
    """A stand-in for the drafter's fused seed traced in the VFY program (value 2): two outputs of the tokens."""
    return tokens.sum(dim=1) + bump, (tokens * 2).to(torch.int64)


METHODS = ["_sample_spec_verify_vps", "_vps_verify_fb", "_vps_verify_vrng", "_vrng_ro_key", "_apply_folded_read_offsets",
           "_check_folded_read_offsets", "_read_offset_fold_geometry", "_vps_ro_fold_warmup_args",
           "_warm_folded_scatter", "_num_reqs_tensor", "compute_selected_local_logits",
           "compute_local_logits_from_hidden_states", "_vfy_chunk_logits", "_vfy_fb_draw", "_vfy_fb_tick", "_vfy_chain",
           "_vfy_verify", "_vfy_check", "_vfy_precompile", "_vfy_meta_draws", "_vfy_rng_call", "_vfy_vrng_key",
           "_precompile_vps_vrng", "_log_vps_fallback"]


class Runner:
    def __init__(self, B=8, *, cand=False, mode=1, vrng=None, seed=0, spec=None):
        self.rejection_sampler = NS(synthetic_conditional_rates=torch.tensor([1.0, 0.32, 0.0]), synthetic_mode=False)
        self.speculative_config = NS(num_speculative_tokens=K)
        self._vps_head = vps_wiring.VpsHead(PROC, HEAD, 0, V, V, 1)
        self._vps_checks_left, self._vps_check_faults, self._vps_checks_done = 0, 0, 0
        self._vps_mismatches = None
        self._vps_fb_path, self._vps_fb_max, self._vps_tiegap, self._vps_cand_draws = True, 8, 0, cand
        self._vps_fb_stats, self._vps_fb_chunks = None, 0
        self.device = CPU
        self._num_reqs_device = {}
        self._fused_read_offsets = (None, 0)
        self._fused_read_offsets_applied = False
        self._vps_ro_fold = True
        self._vps_ro_checks_left, self._vps_ro_check_faults, self._vps_ro_checks_done = 0, 0, 0
        self._vps_ro_mismatches = None
        self._vps_ro_folds = 0
        self._vps_ro_warmed = set()
        self._has_mamba_state = False
        g = torch.Generator().manual_seed(7000 + seed)
        buf = torch.randint(0, 9, (4096,), generator=g, dtype=torch.int32)
        buf[0] = 0
        self.mamba_slot_read_offsets = buf
        self.top_k = torch.full((256,), 20, dtype=torch.int32)
        self.top_k[1] = self.top_k[5] = V                       # fallback rows (flagged)
        self.temps = torch.full((256,), 0.6)
        self.warmup_done = True
        self._vfy = vfy_one.VfyOne(mode, 0, "") if mode else None
        self._vrng = vrng
        # v5 composition (V5COMP-BUILD; tests only): VPSHOST T-1(i)'s (idle) async-copy state, and q38 L4 AGREE's port:
        # VRNG-VPS registers on the runner's engine-step agreement clock, as the runner does at its build.
        self._vps_fb_async = l4_vpshost.AsyncCounts()
        if vrng is not None:
            l4_agree.register(self, vrng)
        self._spec = spec
        self._eager_shape_note = None

    def _padded_sampling_params(self, s, e, n, dtype, device):
        k = torch.where(self.top_k >= V, 0, self.top_k)
        return (self.temps.to(dtype).view(-1, 1)[:n], k.view(-1, 1)[:n].clone(),
                torch.full((n, 1), 0.95, dtype=torch.float32))

    def _upload_unless_unchanged(self, slot, host):
        return host

    def _vfy_seed_spec(self, num_reqs):
        return self._spec

    def _precompile_timed(self, name):
        return contextlib.nullcontext()


def _eager(name):
    """The runner method's eager body. compute_selected_local_logits and compute_local_logits_from_hidden_states are
    `@torch.compile(backend="tpu")` wrappers: bound as they are, they compile for TPU on the CPU pod (smoke v1
    j-07b875aa: the cold-key step's logits were torch_tpu tensors and pytest aborted in torch_tpu while printing them).
    tests/l4vfy binds the same eager bodies from source. A failed unwrap fails here, at import."""
    f = getattr(tr.TPUModelRunner, name)
    f = getattr(f, "_torchdynamo_orig_callable", None) or getattr(f, "__wrapped__", None) or f
    code = f.__code__
    assert code.co_name == name and os.path.realpath(code.co_filename) == os.path.realpath(tr.__file__), (
        name, code.co_name, code.co_filename)
    return f


for _n in METHODS:
    setattr(Runner, _n, _eager(_n))


@pytest.fixture
def rt(monkeypatch):
    import vllm.distributed

    for k in [k for k in os.environ if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_", "VLLM_TPU_VPS_INGRAPH"))]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(vllm.distributed, "tensor_model_parallel_all_gather", lambda x, dim=-1: x)
    monkeypatch.setattr(tr, "_vps_verify_fb_compiled", tr._vps_verify_fb_impl)
    monkeypatch.setattr(tr, "_vps_verify_compiled", tr._vps_verify_impl)
    monkeypatch.setattr(tr, "_vps_verify_rng_compiled", tr._vps_verify_rng_impl)
    monkeypatch.setattr(tr, "_vfy_one_compiled", tr._vfy_one_impl)
    monkeypatch.setattr(tr, "_vfy_one_rng_compiled", tr._vfy_one_rng_impl)
    monkeypatch.setattr(tr, "synchronize_tensors", lambda *a, **k: None)
    lines = []
    monkeypatch.setattr(l4_samp2.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l4_samp2.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    return NS(lines=lines)


def mkctl(seed=3, ab="1"):
    ctl = l4_samp2.VrngCtl(seed, CPU, ab=ab)
    ctl.event_fn = lambda: NS(query=lambda: True)
    return ctl


def problem(seed, B, R=None, extra_rows=8):
    R = B if R is None else R
    g = torch.Generator().manual_seed(seed)
    N = B * (K + 1)
    T = N + extra_rows
    hidden = torch.randn(T, HID, generator=g).bfloat16()
    idx = torch.randperm(T, generator=g)[:N].to(torch.int32)
    logits = PROC._get_logits(hidden[idx.long()], HEAD, None, skip_gather=True)
    rows = torch.arange(N).view(B, K + 1)
    am = logits[rows[:, :K].reshape(-1)].float().argmax(-1)
    draft = torch.where(torch.rand(B * K, generator=g) < 0.6, am,
                        torch.randint(0, V, (B * K,), generator=g)).to(torch.int32)
    md = NS(bonus_logits_indices=rows[:, K].contiguous(), target_logits_indices=rows[:, :K].reshape(-1).contiguous(),
            draft_token_ids=draft, draft_lengths=torch.full((B,), K, dtype=torch.int32),
            draft_lengths_cpu=[K] * R,
            segment_ids=torch.arange(B).repeat_interleave(K).to(torch.int32),
            group_indices=torch.arange(K).repeat(B).to(torch.int32))
    return hidden, idx, md


def slot_groups(seed, G, W, R, nslots=4096):
    g = torch.Generator().manual_seed(9000 + seed)
    slots = (torch.randperm(nslots - 1, generator=g)[: G * W] + 1).to(torch.int32).view(G, W)
    slots[:, R:] = 0
    return [slots[i].clone() for i in range(G)]


def tick(r):
    """q38 L4 AGREE port (v5 composition): one VPS verify = one engine step; the runner's clock ticks first."""
    agree = r.__dict__.get("_l4_agree")
    if agree is not None:
        agree.tick(r)


def vfy_step(r, hidden, idx, md, groups, R, F, gen, mode="vfy"):
    r._fused_read_offsets = (groups, R)
    r._fused_read_offsets_applied = False
    r._vfy.step_mode = mode
    tick(r)
    return r._sample_spec_verify_vps(vfy_one.VfyChunk(hidden, idx), md, 0, R, gen, F)


def warm_all(r):
    r._vfy.logits_dtype = torch.bfloat16
    r._vfy.warm = AlwaysWarm()
    if r._vrng is not None:
        r._vrng.warm = AlwaysWarm()


def full_args(r, md, n, counter):
    temps, top_k, top_p = r._padded_sampling_params(0, n, md.bonus_logits_indices.shape[0], torch.bfloat16, CPU)
    bu, au, ru = vps_rng.full_draws(counter, md.bonus_logits_indices.shape[0], tuple(md.draft_token_ids.shape),
                                    md.target_logits_indices.shape[0], V)
    return (md.bonus_logits_indices, md.target_logits_indices, temps, top_k, top_p, bu, au, ru, md.draft_token_ids,
            md.draft_lengths, md.segment_ids, md.group_indices, r.rejection_sampler.synthetic_conditional_rates, K,
            False)


# ---------------------------------------------------------------- 1. program-level bitwise identity

@pytest.mark.parametrize("F", [0, 8])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("c", [0, 2**31 - 1, -12345])
def test_vfy_rng_impl_equals_vfy_impl_on_counter_draws(rt, F, seeded, c):
    B = R = 8
    hidden, idx, md = problem(10 + F + 3 * seeded, B, R)
    groups = slot_groups(5, 3, 64, R)
    r = Runner(B)
    ct = torch.tensor(c, dtype=torch.int32)
    args = full_args(r, md, R, ct)
    ro = (groups, torch.tensor(R), torch.int32)
    acc = torch.tensor([1, 2, 3], dtype=torch.int64)
    na = torch.tensor([R], dtype=torch.int32)
    seed = (seed_body, torch.tensor(7)) if seeded else ()
    base = tr._vfy_one_impl(r._vps_head, hidden, idx, *args, na, F, False, None, *ro, acc, *seed)
    rng = tr._vfy_one_rng_impl(r._vps_head, hidden, idx, *args[:5], ct, *args[8:], na, F, *ro, acc, *seed)
    assert len(rng) == len(base) + 2
    assert all(torch.equal(a, b) for a, b in zip(base[:4], rng[:4]))
    assert all(torch.equal(a, b) for a, b in zip(base[4:], rng[6:]))
    al, nxt = rng[4], rng[5]
    assert int(nxt) == (c + 1 if c != 2**31 - 1 else -2**31)
    emitted = (rng[0] != -1).sum(dim=1)
    assert al.tolist() == [int((emitted[:R] - 1).clamp(min=0).sum()), R * K, R]


@pytest.mark.parametrize("fb", [True, False])
def test_chain_rng_impl_with_read_offsets_equals_base(rt, fb):
    B = R = 8
    hidden, idx, md = problem(31, B, R)
    groups = slot_groups(6, 3, 64, R)
    r = Runner(B)
    local = PROC._get_logits(hidden[idx.long()], HEAD, None, skip_gather=True)
    ct = torch.tensor(99, dtype=torch.int32)
    args = full_args(r, md, R, ct)
    ro = (groups, torch.tensor(R), torch.int32)
    na = torch.tensor([R], dtype=torch.int32)
    if fb:
        base = tr._vps_verify_fb_impl(local, 0, *args, na, 8, False, None, *ro)
    else:
        base = tr._vps_verify_impl(local, 0, *args, *ro)
        base = (base[0], torch.zeros(3, dtype=torch.int32), *base[1:])
    rng = tr._vps_verify_rng_impl(local, 0, *args[:5], ct, *args[8:], na, 8 if fb else 0, fb, V, *ro)
    assert len(rng) == 6 and int(rng[5]) == 100
    assert all(torch.equal(a, b) for a, b in zip(base, rng[:4]))


# ---------------------------------------------------------------- 2. the VFY path with both flags

class Calls(TorchDispatchMode):
    NAMES = ("uniform", "rand", "bernoulli", "normal", "exponential", "random_")

    def __init__(self):
        super().__init__()
        self.rng = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(t in str(func) for t in self.NAMES):
            self.rng += 1
        return func(*args, **(kwargs or {}))


def count_programs(monkeypatch):
    seen = {"vfy": 0, "vfy_rng": 0, "chain": 0, "chain_rng": 0}
    for name, attr, impl in (("vfy", "_vfy_one_compiled", tr._vfy_one_impl),
                             ("vfy_rng", "_vfy_one_rng_compiled", tr._vfy_one_rng_impl),
                             ("chain", "_vps_verify_fb_compiled", tr._vps_verify_fb_impl),
                             ("chain_rng", "_vps_verify_rng_compiled", tr._vps_verify_rng_impl)):
        def wrapped(*a, _n=name, _f=impl, **k):
            seen[_n] += 1
            return _f(*a, **k)
        monkeypatch.setattr(tr, attr, wrapped)
    return seen


@pytest.mark.parametrize("seeded", [False, True])
def test_vfy_mode_runs_one_composed_call(rt, monkeypatch, seeded):
    B = R = 8
    hidden, idx, md = problem(41, B, R)
    groups = slot_groups(7, 3, 64, R)
    spec = vfy_one.VfySeedSpec(seed_body, (torch.tensor(5),), (torch.tensor(5),)) if seeded else None
    ctl = mkctl()
    on = Runner(B, vrng=ctl, spec=spec)
    ref = Runner(B, spec=spec)                       # the base VFY program, fed the counter's draws below
    for r in (on, ref):
        warm_all(r)
    seen = count_programs(monkeypatch)
    gen = torch.Generator().manual_seed(1)
    g0 = gen.get_state().clone()
    c0 = ctl.counter_word()
    mode = Calls()
    with mode:
        out = vfy_step(on, hidden, idx, md, groups, R, 8, gen)
    assert seen == {"vfy": 0, "vfy_rng": 1, "chain": 0, "chain_rng": 0} and mode.rng == 0
    assert torch.equal(gen.get_state(), g0) and ctl.counter_word() == c0 + 1 and int(ctl.counter) == c0 + 1
    assert on._vfy.stats["vfy_vrng"] == 1 and ctl.stats["vrng"] == 1
    # reference: the base VFY program on the same counter's full draws
    ref._fused_read_offsets = (groups, R)
    args = full_args(ref, md, R, torch.tensor(c0, dtype=torch.int32))
    acc0 = torch.zeros(3, dtype=torch.int64)
    res = tr._vfy_one_impl(ref._vps_head, hidden, idx, *args, torch.tensor([R], dtype=torch.int32), 8, False, None,
                           list(groups), ref._num_reqs_tensor(R), torch.int32, acc0,
                           *((spec.fn, *spec.args) if seeded else ()))
    ref._apply_folded_read_offsets(res[0], (list(groups), ref._num_reqs_tensor(R), torch.int32), res[2], res[3])
    assert torch.equal(out, res[0]) and torch.equal(on._vps_fb_stats, res[1])
    assert torch.equal(on.mamba_slot_read_offsets, ref.mamba_slot_read_offsets)
    if seeded:
        stash = on.__dict__.get("_vfy_seed")
        assert stash is not None and all(torch.equal(a, b) for a, b in zip(stash[2], res[4:]))


def test_vrng_window_off_is_bitwise_flag_off(rt, monkeypatch):
    """VRNG warm but its window off ('0'): the VFY step is the base VFY program on eager draws, bitwise the
    flag-off step (tokens, random stream, stats, read-offset buffer)."""
    B = R = 8
    hidden, idx, md = problem(43, B, R)
    groups = slot_groups(8, 3, 64, R)
    on, off = Runner(B, vrng=mkctl(ab="0")), Runner(B)
    for r in (on, off):
        warm_all(r)
    seen = count_programs(monkeypatch)
    g1, g2 = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2)
    o1 = vfy_step(on, hidden, idx, md, groups, R, 8, g1)
    o2 = vfy_step(off, hidden, idx, md, groups, R, 8, g2)
    assert torch.equal(o1, o2) and torch.equal(g1.get_state(), g2.get_state())
    assert torch.equal(on._vps_fb_stats, off._vps_fb_stats)
    assert torch.equal(on.mamba_slot_read_offsets, off.mamba_slot_read_offsets)
    assert seen["vfy"] == 2 and seen["vfy_rng"] == 0 and on._vrng.stats["window_off"] == 1


def test_cold_base_key_runs_the_chain_vrng_program(rt, monkeypatch):
    B = R = 8
    hidden, idx, md = problem(45, B, R)
    groups = slot_groups(9, 3, 64, R)
    ctl = mkctl()
    r = Runner(B, vrng=ctl)
    warm_all(r)
    r._vfy.warm = set()                             # no VFY key warmed: the D83 chain serves
    seen = count_programs(monkeypatch)
    gen = torch.Generator().manual_seed(3)
    g0 = gen.get_state().clone()
    vfy_step(r, hidden, idx, md, groups, R, 8, gen)
    assert seen == {"vfy": 0, "vfy_rng": 0, "chain": 0, "chain_rng": 1}
    assert torch.equal(gen.get_state(), g0) and ctl.vrng_calls == 1 and r._vfy.stats["cold"] == 1


def test_no_read_offsets_takes_the_base_draws(rt, monkeypatch):
    B = R = 8
    hidden, idx, md = problem(47, B, R)
    ctl = mkctl()
    ctl.warm = set()                                # real warm set: ("vfy", "no_ro") is never in it
    ctl.warm_key(("anything",))
    r = Runner(B, vrng=ctl)
    r._vfy.logits_dtype = torch.bfloat16
    r._vfy.warm = AlwaysWarm()
    seen = count_programs(monkeypatch)
    gen = torch.Generator().manual_seed(4)
    g0 = gen.get_state().clone()
    vfy_step(r, hidden, idx, md, [], R, 8, gen)
    assert seen["chain"] == 1 and seen["chain_rng"] == 0 and ctl.stats["cold"] == 1
    assert not torch.equal(gen.get_state(), g0) and ctl.vrng_calls == 0


# ---------------------------------------------------------------- 3. composed CHECK

@pytest.mark.parametrize("broken", [False, True])
def test_composed_check(rt, monkeypatch, broken):
    """CHECK step with both flags: the chain's VRNG program serves (one counter step) and the composed program runs
    beside it on the same counter value; 0 mismatches, or counted when the composed program is broken."""
    B = R = 8
    hidden, idx, md = problem(51, B, R)
    groups = slot_groups(10, 3, 64, R)
    ctl = mkctl()
    r = Runner(B, vrng=ctl)
    warm_all(r)
    seen = count_programs(monkeypatch)
    if broken:
        real = tr._vfy_one_rng_impl

        def bad(*a, **k):
            res = list(real(*a, **k))
            res[0] = res[0].clone()
            res[0][0, 0] += 1
            return tuple(res)

        monkeypatch.setattr(tr, "_vfy_one_rng_compiled", bad)
    gen = torch.Generator().manual_seed(5)
    c0 = ctl.counter_word()
    vfy_step(r, hidden, idx, md, groups, R, 8, gen, mode="check")
    assert seen["chain_rng"] == 1 and (broken or seen["vfy_rng"] == 1) and seen["vfy"] == 0
    assert ctl.counter_word() == c0 + 1 and r._vfy.stats["check"] == 1
    assert (r._vfy.check_mismatches > 0) == broken


# ---------------------------------------------------------------- 4. warm-up keys

class WarmRunner(Runner):
    def __init__(self, vrng):
        super().__init__(8, vrng=vrng)
        self.attn_groups = [[NS(metadata_builders=[NS(is_mamba_group=True)])]] * 3
        self.num_reqs_max_model_len, self.most_model_len, self.num_reqs_most_model_len = 16, None, None
        self.max_num_reqs = 16
        self.num_tokens_paddings = [16, 32, 64]
        self.num_reqs_paddings = [8, 16]
        self._hidden_states_dtype = torch.bfloat16
        self.model_config = NS(get_hidden_size=lambda: HID)
        self._is_async_drafter = False
        self.drafter = self._async_drafter = None


def test_warmup_registers_the_composed_keys(rt, monkeypatch):
    monkeypatch.setattr(tr.envs, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", False, raising=False)
    ctl = mkctl()
    r = WarmRunner(ctl)
    seen = count_programs(monkeypatch)
    r._vfy_precompile()
    vkeys = [k for k in ctl.warm if k[0] == "vfy"]
    assert r._vfy.programs > 0 and len(vkeys) == r._vfy.programs == seen["vfy_rng"] == seen["vfy"]
    # a steady serving step's key is among them: R live requests at their buckets
    R = 5
    B = tr._get_padded_num_reqs_with_upper_limit(R, r.max_num_reqs)
    T = tr._get_padded_token_len(r.num_tokens_paddings, R * (K + 1))
    hidden = torch.zeros(T, HID, dtype=torch.bfloat16)
    idx = torch.zeros(T, dtype=torch.int32)
    md = NS(bonus_logits_indices=torch.zeros(B, dtype=torch.int32), target_logits_indices=torch.zeros(T, dtype=torch.int32),
            draft_token_ids=torch.zeros(T, dtype=torch.int32), draft_lengths=torch.zeros(B, dtype=torch.int32),
            segment_ids=torch.zeros(T, dtype=torch.int64), group_indices=torch.zeros(T, dtype=torch.int32))
    ro_w = next(w for w in r._read_offset_fold_geometry()[1] if w >= R)
    r._fused_read_offsets = ([torch.zeros(ro_w, dtype=torch.int32) for _ in range(r._read_offset_fold_geometry()[0])], R)
    temps, top_k, top_p = r._padded_sampling_params(0, R, B, torch.bfloat16, CPU)
    F = vps_wiring.fallback_variants(r._vps_fb_max, r._vps_tiegap)[0]
    key = r._vfy_vrng_key(vfy_one.VfyChunk(hidden, idx), md, temps, top_k, top_p, F, None)
    assert key in ctl.warm, key


# ---------------------------------------------------------------- 5. AOT census of a VFY step

def test_vfy_step_census(rt, monkeypatch):
    B = R = 8
    hidden, idx, md = problem(61, B, R)
    groups = slot_groups(11, 3, 64, R)
    counts = {}
    for arm, ctl in (("base", None), ("vrng", mkctl())):
        r = Runner(B, vrng=ctl)
        warm_all(r)
        mode = Calls()
        with mode:
            vfy_step(r, hidden, idx, md, groups, R, 8, torch.Generator().manual_seed(6))
        counts[arm] = mode.rng
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    r = Runner(B)
    args = full_args(r, md, R, torch.tensor(1, dtype=torch.int32))
    torch._dynamo.reset()
    fn = torch.compile(tr._vfy_one_rng_impl, backend=capture, fullgraph=True, dynamic=False)
    fn(r._vps_head, hidden, idx, *args[:5], torch.tensor(1, dtype=torch.int32), *args[8:],
       torch.tensor([R], dtype=torch.int32), 8, groups, torch.tensor(R), torch.int32, torch.zeros(3, dtype=torch.int64))
    nodes = [n for g in graphs for n in g.graph.nodes if n.op in ("call_function", "call_method")]
    rng_nodes = [n for n in nodes if any(t in str(n.target) for t in Calls.NAMES)]
    print(f"\nVRNG_VFY_CENSUS base_eager_rng={counts['base']} vrng_eager_rng={counts['vrng']} aot_graphs={len(graphs)} "
          f"aot_nodes={len(nodes)} aot_rng_nodes={len(rng_nodes)}")
    assert counts == {"base": 3, "vrng": 0} and len(graphs) == 1 and rng_nodes == []


def test_served_step_after_warmup_runs_the_composed_program(rt, monkeypatch):
    """After the real warm-up (no AlwaysWarm), a steady serving step at a warmed shape takes the composed program:
    the VFY program's meta-draw key and the VRNG key computed in serving match the warmed ones."""
    monkeypatch.setattr(tr.envs, "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY", False, raising=False)
    ctl = mkctl()
    r = WarmRunner(ctl)
    r._vfy_precompile()
    seen = count_programs(monkeypatch)
    R = 5
    B = tr._get_padded_num_reqs_with_upper_limit(R, r.max_num_reqs)
    T = tr._get_padded_token_len(r.num_tokens_paddings, R * (K + 1))
    g = torch.Generator().manual_seed(71)
    hidden = torch.randn(T, HID, generator=g).bfloat16()
    idx = torch.randperm(T, generator=g).to(torch.int32)
    md = NS(bonus_logits_indices=(torch.arange(B) * (K + 1) + K).clamp(max=T - 1).to(torch.int32),
            target_logits_indices=torch.arange(T, dtype=torch.int32),
            draft_token_ids=torch.randint(0, V, (T,), generator=g, dtype=torch.int32),
            draft_lengths=torch.tensor([K] * R + [0] * (B - R), dtype=torch.int32), draft_lengths_cpu=[K] * R,
            segment_ids=(torch.arange(T) // K).clamp(max=R).to(torch.int64),
            group_indices=(torch.arange(T) % K).to(torch.int32))
    ng, widths = r._read_offset_fold_geometry()
    groups = [torch.randint(1, 4096, (widths[0],), generator=g, dtype=torch.int32) for _ in range(ng)]
    F = vps_wiring.fallback_variants(r._vps_fb_max, r._vps_tiegap)[0]
    r._fused_read_offsets = (groups, R)
    r._vfy.step_mode = "vfy"
    gen = torch.Generator().manual_seed(8)
    g0 = gen.get_state().clone()
    tick(r)
    r._sample_spec_verify_vps(vfy_one.VfyChunk(hidden, idx), md, 0, R, gen, F)
    assert seen == {"vfy": 0, "vfy_rng": 1, "chain": 0, "chain_rng": 0}, (seen, dict(ctl.stats), dict(r._vfy.stats))
    assert torch.equal(gen.get_state(), g0) and ctl.vrng_calls == 1
