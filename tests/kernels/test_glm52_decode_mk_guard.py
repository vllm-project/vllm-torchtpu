# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the glm52_decode_mk W4 step guard and the served cache-form binding (WHITEBOARD.v3 §8 W4).

The predicate is `classify_step` (pure), the served cache form is `stacked_groups_plan` + `bind_caches`, and the
runner contract is `step_ok` (one counted decision per scheduled step) + `try_call` (consumes it, falls back to
stock on a call-time mismatch). None of this needs a TPU: the megakernel's step program is stubbed.
"""

import dataclasses
import types

import pytest
import torch

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as VA
from vllm_torchtpu.kernels.glm52_decode_mk.vllm_adapter import (
    REASONS,
    Glm52DecodeMK,
    Unsupported,
    classify_step,
    stacked_groups_plan,
)

ROWS = [16, 32]


def _cls(reqs, **kw):
    a = dict(num_chunks=1, has_encoder=False, qmax=4, rows=ROWS, bmax=8)
    a.update(kw)
    return classify_step(reqs, **a)


def V(n):  # a verify request with n rows (n - 1 drafts), past its prompt
    return (n, n - 1, False)


# ------------------------------------------------------------------ classify_step


@pytest.mark.parametrize(
    "reqs, bucket",
    [
        ([V(4)] * 4, 16),  # dec-b16: 4 requests x 4 rows
        ([V(4)], 16),
        ([V(2), V(3), V(4)], 16),  # ragged verify rows
        ([V(4)] * 5, 32),  # 20 tokens
        ([V(4)] * 8, 32),  # dec-b32
        (
            [V(3)] * 5,
            32,
        ),  # 15 tokens fit 16, but 5 requests > 16 // QMAX: the kernel's row contract
        ([V(2)] * 4, 16),
    ],
)
def test_eligible_steps_engage(reqs, bucket):
    assert _cls(reqs)[:2] == (bucket, "mk")


@pytest.mark.parametrize(
    "reqs, kw, reason",
    [
        ([(1500, 0, True)], {}, "prefill"),  # prefill chunk
        (
            [(3, 0, True)],
            {},
            "prefill",
        ),  # a 3-token prompt tail after a prefix-cache hit
        (
            [(4, 0, True), (4, 0, True)],
            {},
            "prefill",
        ),  # verify-sized prefill chunks: no drafts
        ([V(4), V(4), (900, 0, True)], {}, "mixed"),  # decode + a new request's prefill
        ([V(4), (1, 0, False)], {}, "mixed"),  # one request lost its drafts
        ([(1, 0, False)] * 3, {}, "no_draft"),  # plain decode, q_len 1
        (
            [(4, 3, True)],
            {},
            "prefill",
        ),  # drafts on a request still in its prompt: never verify
        ([(3, 3, False)], {}, "prefill"),  # scheduled != 1 + drafts
        ([V(5)], {}, "q_len"),  # more rows than MTP 3 verify
        ([V(4)] * 9, {"bmax": 8}, "over_reqs"),
        ([V(4)] * 4, {"rows": [16]}, "mk"),
        ([V(4)] * 5, {"rows": [16]}, "over_bucket"),  # only the 16 bucket built
        ([V(4)] * 4, {"num_chunks": 2}, "multi_chunk"),
        ([V(4)] * 4, {"has_encoder": True}, "encoder"),
        ([], {}, "no_draft"),
    ],
)
def test_ineligible_steps_decline(reqs, kw, reason):
    bucket, why, detail = _cls(reqs, **kw)
    assert why == reason, detail
    assert (bucket is None) == (reason != "mk")
    assert why in REASONS


def test_every_reason_is_named():
    assert len(set(REASONS)) == len(REASONS)


# ------------------------------------------------------------------ stacked_groups_plan


def _binds(n_layers, gs):
    return [(i // gs, i % gs) for i in range(n_layers)]


def test_served_groups_2x39_1x21():
    b = {"nope": _binds(78, 39), "rope": _binds(78, 39), "idx": _binds(21, 21)}
    s = {"nope": [39, 39], "rope": [39, 39], "idx": [21]}
    assert stacked_groups_plan(b, s) == {"nope": 2, "rope": 2, "idx": 1}


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (
            lambda b, s: s.update(nope=[40, 38], rope=[40, 38]),
            "not equal",
        ),  # unequal split
        (
            lambda b, s: b["nope"].__setitem__(5, (0, 6)),
            "want 0 row 5",
        ),  # a layer on the wrong row
        (lambda b, s: b["idx"].__setitem__(0, (1, 0)), "want 0 row 0"),
        (
            lambda b, s: (s.update(rope=[78]), b.update(rope=_binds(78, 78))),
            "nope groups",
        ),  # nope / rope differ
        (lambda b, s: s.pop("idx"), "no idx"),
        (lambda b, s: b["nope"].pop(), "77 layers bound"),
        (
            lambda b, s: (
                s.update(nope=[13] * 6, rope=[13] * 6),
                b.update(nope=_binds(78, 13), rope=_binds(78, 13)),
            ),
            "untimed",
        ),
    ],
)
def test_bad_groups_decline(mutate, msg):
    b = {"nope": _binds(78, 39), "rope": _binds(78, 39), "idx": _binds(21, 21)}
    s = {"nope": [39, 39], "rope": [39, 39], "idx": [21]}
    mutate(b, s)
    with pytest.raises(Unsupported, match=msg):
        stacked_groups_plan(b, s)


# ------------------------------------------------------------------ the A0-10 pool guard (load time)


def _owner_pages_bruteforce(seq, page=256, R=16, ilv=4):
    # rank 0 owns position p iff (p % (R * ilv)) // ilv == 0 (config.FullCfg.owner): count, then pages
    n = sum(1 for p in range(seq) if (p % (R * ilv)) // ilv == 0)
    return -(-n // page)


@pytest.mark.parametrize(
    "seq",
    [
        1,
        3,
        4,
        5,
        63,
        64,
        65,
        68,
        69,
        4095,
        4096,
        4097,
        16383,
        16384,
        16385,
        65536,
        65537,
        131072,
        131073,
    ],
)
def test_rank0_pages_matches_ownership(seq):
    assert VA.rank0_pages(seq, page=256, R=16, ilv=4) == _owner_pages_bruteforce(seq)
    assert VA.rank0_pages(seq, page=1, R=16, ilv=4) == _owner_pages_bruteforce(
        seq, page=1
    )  # token-exact


def test_rank0_pages_monotone():
    p = [VA.rank0_pages(s, page=256, R=16, ilv=4) for s in range(1, 40000, 7)]
    assert all(a <= b for a, b in zip(p, p[1:]))


def test_pool_guard_served_lengths_pass():
    # 640K served, the 1M twin: 160 / 256 rank-0 pages against the production pool 1344
    assert VA.check_pool(655360, {16: 1344, 32: 1344}, page=256, R=16, ilv=4) == 160
    assert VA.check_pool(1 << 20, {16: 1344, 32: 1344}, page=256, R=16, ilv=4) == 256


def test_pool_guard_refuses_overflow():
    # a request one page past the pool: the DSA wave would overflow b_keys / b_blk (A0-10) -> no megakernel
    mml = 1344 * 256 * 16 + 1  # 1345 rank-0 pages
    assert VA.rank0_pages(mml, page=256, R=16, ilv=4) == 1345
    with pytest.raises(Unsupported, match="pool guard"):
        VA.check_pool(mml, {16: 1344, 32: 1344}, page=256, R=16, ilv=4)
    with pytest.raises(Unsupported, match=r"b32"):  # any bucket's smaller pool refuses
        VA.check_pool(1 << 20, {16: 1344, 32: 255}, page=256, R=16, ilv=4)
    assert (
        VA.check_pool(1344 * 256 * 16, {16: 1344}, page=256, R=16, ilv=4) == 1344
    )  # exactly full: allowed


def test_pool_guard_is_wired_at_load():
    src = __import__("inspect").getsource(VA.Glm52DecodeMK.__init__)
    flat = "".join(
        src.split()
    )  # whitespace-free: independent of the formatter's line breaks
    assert (
        "self.pool_pages=check_pool(mml,{t:tl.poolfort,tlinself.tiles_by_rows.items()}"
        in flat
    )


# ------------------------------------------------------------------ load guard: qrep is judged per layer


class _PastLoadGuards(Exception):
    """Raised by the stubbed dist.get_rank, which __init__ first calls right after its last load-time guard."""


def _glm52_hf(cfg=C.FULL):
    return types.SimpleNamespace(
        model_type="glm_moe_dsa",
        num_hidden_layers=cfg.L,
        first_k_dense_replace=cfg.first_dense,
        hidden_size=cfg.H,
        num_attention_heads=cfg.NH,
        q_lora_rank=cfg.QL,
        index_topk_freq=cfg.idx_freq,
        index_skip_topk_offset=cfg.idx_skip,
        rms_norm_eps=cfg.eps,
        routed_scaling_factor=cfg.routed_scale,
        kv_lora_rank=cfg.KL,
        qk_nope_head_dim=cfg.DN,
        qk_rope_head_dim=cfg.DR,
        v_head_dim=cfg.DV,
        n_routed_experts=cfg.E,
        num_experts_per_tok=cfg.TOPK,
        moe_intermediate_size=cfg.I,
        n_shared_experts=cfg.IS // cfg.I,
        intermediate_size=cfg.ID,
        index_n_heads=cfg.IH,
        index_head_dim=cfg.IDIM,
        index_topk=cfg.KT,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        rope_interleave=True,
        indexer_rope_interleave=True,
        rope_parameters={"rope_type": "default"},
    )


def _glm52_runner(
    qrep_layers=(), dcp_q_replicate=True, cfg=C.FULL, q_group=None, dcp=None
):
    """A served GLM-5.2 runner (TP16 x DCP16 il4, EP off, MTP 3) whose layer l's q_b_proj has qrep_active = l in
    qrep_layers, over a q group of `q_group` (default the world, rank_in_group 0 = the stubbed TP rank).
    dcp_q_replicate is the parallel_config field vLLM's GLM config hook sets True by default."""
    layers = torch.nn.ModuleList()
    for l in range(cfg.L):
        ly = torch.nn.Module()
        ly.self_attn = torch.nn.Module()
        ly.self_attn.q_b_proj = torch.nn.Module()
        ly.self_attn.q_b_proj.qrep_active = l in qrep_layers
        if l in qrep_layers:  # vLLM DCPGroupColumnParallelLinear's attributes
            ly.self_attn.q_b_proj.group_size = cfg.R if q_group is None else q_group
            ly.self_attn.q_b_proj.rank_in_group = 0
        if cfg.full_indexer(l):
            ly.self_attn.indexer = torch.nn.Module()
        layers.append(ly)
    bb = torch.nn.Module()
    bb.layers, bb.embed_tokens, bb.norm = (
        layers,
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    model = torch.nn.Module()
    model.model = bb
    pc = types.SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=cfg.R,
        decode_context_parallel_size=cfg.R if dcp is None else dcp,
        cp_kv_cache_interleave_size=cfg.ILV,
        enable_expert_parallel=False,
        dcp_q_replicate=dcp_q_replicate,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=_glm52_hf(cfg), max_model_len=1 << 20
        ),
        parallel_config=pc,
        speculative_config=types.SimpleNamespace(
            method="mtp", num_speculative_tokens=cfg.QMAX - 1
        ),
        block_size=cfg.page,
        scheduler_config=types.SimpleNamespace(max_num_seqs=cfg.BM),
        model=model,
    )


@pytest.fixture
def load_env(monkeypatch):
    import torch.distributed as dist

    def past_guards():
        raise _PastLoadGuards

    monkeypatch.setenv("GLM52_DECODE_MK", "1")
    monkeypatch.setenv("GLM52_DECODE_MK_ROWS", "16,32")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: C.FULL.R)
    monkeypatch.setattr(dist, "get_rank", past_guards)
    monkeypatch.setattr(
        Glm52DecodeMK,
        "_check_rank_layout",
        staticmethod(lambda R, dcp=None: {"rank": 0, "tp": 0, "dcp": 0}),
    )


@pytest.mark.parametrize("field", [True, False])
def test_load_guard_ignores_config_qrep_field(load_env, field):
    # the production template sets VLLM_DCP_Q_REPLICATE=0 (runtime qrep off, q_b_proj column-parallel) while vLLM's
    # GLM config hook leaves parallel_config.dcp_q_replicate True: the megakernel must still build
    with pytest.raises(_PastLoadGuards):
        Glm52DecodeMK(_glm52_runner(qrep_layers=(), dcp_q_replicate=field))


QREP_SETS = [(0,), (40,), (C.FULL.L - 1,), tuple(range(C.FULL.L))]


@pytest.mark.parametrize("qrep_layers", QREP_SETS)
def test_load_guard_accepts_qrep_over_the_world(load_env, qrep_layers):
    # VLLM_DCP_Q_REPLICATE=1 at DCP16: the replicated q_b is sliced to this rank's heads (q_b_local), not refused
    with pytest.raises(_PastLoadGuards):
        Glm52DecodeMK(_glm52_runner(qrep_layers=qrep_layers))


@pytest.mark.parametrize("qrep_layers", QREP_SETS)
@pytest.mark.parametrize("q_group", [8, 1, None])
def test_load_guard_refuses_qrep_group_not_the_world(load_env, qrep_layers, q_group):
    runner = _glm52_runner(qrep_layers=qrep_layers, q_group=q_group or 8)
    if q_group is None:  # a replicated layer without the DCPGroup attributes
        for l in qrep_layers:
            del runner.model.model.layers[l].self_attn.q_b_proj.group_size
    with pytest.raises(
        Unsupported,
        match=rf"layer {qrep_layers[0]}: q_b_proj is replicated \(qrep_active\) over a q group of",
    ):
        Glm52DecodeMK(runner)
    assert Glm52DecodeMK.maybe_create(runner) is None  # logged, stock forever


@pytest.mark.parametrize("dcp", [4, 2, 1])
def test_load_guard_refuses_dcp_not_tp_with_a_clear_reason(load_env, dcp, monkeypatch):
    # DCP 16 (= TP) and DCP 8 (V16) build; any other DCP is refused, loudly
    with pytest.raises(
        Unsupported, match=rf"DCP {dcp}: the megakernel is built for DCP = TP = 16 or, as V16 .*DCP = 8"
    ):
        Glm52DecodeMK(_glm52_runner(dcp=dcp))
    warned = []
    monkeypatch.setattr(VA.logger, "warning", lambda f, *a: warned.append(f % a))
    assert Glm52DecodeMK.maybe_create(_glm52_runner(dcp=dcp, qrep_layers=(0,))) is None
    assert len(warned) == 1, warned
    assert "disabled (not built; every step runs the stock graphs)" in warned[0]
    assert f"DCP {dcp}: the megakernel is built for" in warned[0]


@pytest.mark.parametrize("qrep_layers", [(), (0,), tuple(range(C.FULL.L))])
def test_load_guard_accepts_dcp8_v16(load_env, qrep_layers):
    # the best served config's DECODE_CONTEXT_PARALLEL_SIZE=8 (+ VLLM_DCP_Q_REPLICATE=1: q group = the DCP8 group,
    # rank_in_group = rank % 8 = 0 for the stubbed rank 0) builds as V16
    with pytest.raises(_PastLoadGuards):
        Glm52DecodeMK(_glm52_runner(dcp=8, qrep_layers=qrep_layers, q_group=8))


def test_load_guard_refuses_dcp8_with_a_world_q_group(load_env):
    with pytest.raises(Unsupported, match=r"q group of 16 at rank_in_group 0"):
        Glm52DecodeMK(_glm52_runner(dcp=8, qrep_layers=(0,), q_group=16))


def test_load_guard_does_not_read_the_config_qrep_field():
    src = __import__("inspect").getsource(VA.Glm52DecodeMK.__init__)
    assert (
        'getattr(pc, "dcp_q_replicate"' not in src and "pc.dcp_q_replicate" not in src
    )
    # judged per layer from the module (qrep_layer reads q_b_proj.qrep_active)
    assert "qrep_layer(" in src
    assert '"qrep_active"' in __import__("inspect").getsource(VA.qrep_layer)


# ------------------------------------------------------------------ bind_caches on fake modules

L, NB, NBI = 6, 2, 3
CFG = dataclasses.replace(
    C.FULL, L=L, first_dense=1, idx_freq=4, idx_skip=3
)  # fresh indexer layers 0, 1, 2
IDX = [l for l in range(L) if CFG.full_indexer(l)]


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_UK_T = self.W_UV = None
        self.kv_cache = None
        self.kv_stack = None


class _KCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_cache = None
        self.kv_stack = None


class _SelfAttn(torch.nn.Module):
    def __init__(self, has_idx):
        super().__init__()
        self.attn = _Attn()
        if has_idx:
            self.indexer = torch.nn.Module()
            self.indexer.k_cache = _KCache()


def _shape(kind, nb):
    return {
        "nope": (nb, 256, 128),
        "rope": (nb, 64, 128),
        "idx": (nb, CFG.IREC // 4, 4, 256),
    }[kind]


def _dtype(kind):
    return torch.uint8 if kind == "idx" else torch.uint32


def _mk(stack=None, perlayer=False, groups_nope=2):
    """A Glm52DecodeMK with fake layers; stack: runner-stacked groups (nope / rope `groups_nope`, idx 1)."""
    mk = object.__new__(Glm52DecodeMK)
    mk.cfg = CFG
    mk.rows = ROWS
    mk.cfgs = {t: dataclasses.replace(CFG, TMAX=t) for t in ROWS}
    mk.rank = 0
    mk.enabled = True
    mk.counts = {r: 0 for r in REASONS}
    mk.decisions = mk.consumed = mk.calls = 0
    mk.log_every = 0
    mk.unbound = "capture_model has not bound the caches"
    mk.cache_form = None
    mk.call_shapes, mk.warm_shapes = {}, {}
    mk._steps = {}
    # levers without LV (bind_caches reads lv_full since a66f6c89f)
    mk.lv_full, mk._steps_full = False, {}
    mk.dense, mk.experts, mk.max_pages = [], [], 256
    mk.smalls = torch.zeros(1)
    layers = [types.SimpleNamespace(self_attn=_SelfAttn(l in IDX)) for l in range(L)]
    mk.backbone = types.SimpleNamespace(layers=layers)
    runner = types.SimpleNamespace(glm5_kv_stack_groups={})
    if stack is None:
        for l, ly in enumerate(layers):
            sa = ly.self_attn
            sa.attn.kv_cache = (
                torch.zeros(_shape("nope", NB), dtype=torch.uint32),
                torch.zeros(_shape("rope", NB), dtype=torch.uint32),
            )
            if l in IDX:
                sa.indexer.k_cache.kv_cache = torch.zeros(
                    _shape("idx", NBI), dtype=torch.uint8
                )
    else:
        gs = L // groups_nope
        g = {k: [] for k in ("nope", "rope", "idx")}
        for gi in range(groups_nope):
            names = list(range(gi * gs, (gi + 1) * gs))
            for k in ("nope", "rope"):
                g[k].append(
                    (torch.zeros((gs,) + _shape(k, NB), dtype=torch.uint32), names)
                )
        g["idx"].append(
            (torch.zeros((len(IDX),) + _shape("idx", NBI), dtype=torch.uint8), IDX)
        )
        for l, ly in enumerate(layers):
            sa = ly.self_attn
            sa.attn.kv_cache = (g["nope"][l // gs][0], g["rope"][l // gs][0])
            sa.attn.kv_stack = (l % gs, NB)
            if l in IDX:
                sa.indexer.k_cache.kv_cache = g["idx"][0][0]
                sa.indexer.k_cache.kv_stack = (IDX.index(l), NBI)
        runner.glm5_kv_stack_groups = g
        stack(g, layers)
    built = []
    mk._build_op = lambda cfg: built.append(cfg.TMAX) or _FakeStep(mk, cfg.TMAX)
    return mk, runner, built


class _FakeStep:
    def __init__(self, mk, T):
        self.mk, self.T = mk, T
        self.n = 0

    def __call__(self, ids, pos, qsl, seq, bt, smalls, weights, caches):
        self.n += 1
        assert caches is self.mk.caches
        return torch.full((self.T, 8), 7.0)


def test_bind_stacked_served_form():
    mk, runner, built = _mk(stack=lambda g, layers: None)
    mk.bind_caches(runner)
    assert mk.unbound is None and mk.cache_form == "stacked"
    assert mk.groups == {"nope": 2, "rope": 2, "idx": 1}
    g = runner.glm5_kv_stack_groups
    want = [t for k in ("nope", "rope", "idx") for t, _ in g[k]]
    assert len(mk.caches) == len(want) and all(
        a is b for a, b in zip(mk.caches, want)
    )  # operand order
    assert built == ROWS


def test_bind_stacked_one_group():
    mk, runner, _ = _mk(stack=lambda g, layers: None, groups_nope=1)
    mk.bind_caches(runner)
    assert mk.unbound is None and mk.groups == {"nope": 1, "rope": 1, "idx": 1}


def _swap_rows(g, layers):
    layers[1].self_attn.attn.kv_stack, layers[2].self_attn.attn.kv_stack = (
        (2, NB),
        (1, NB),
    )


def _partial(g, layers):
    layers[4].self_attn.attn.kv_stack = None


def _foreign(g, layers):
    t = g["nope"][0][0]
    layers[0].self_attn.attn.kv_cache = (
        t.clone(),
        layers[0].self_attn.attn.kv_cache[1],
    )


@pytest.mark.parametrize(
    "stack, msg",
    [
        (_swap_rows, "row 2, want 0 row 1"),
        (_partial, "partly stacked"),
        (_foreign, "not bound to a runner group tensor"),
    ],
)
def test_bind_stacked_mismatch_stays_unbound(stack, msg):
    mk, runner, built = _mk(stack=stack)
    mk.bind_caches(runner)  # never raises
    assert mk.unbound is not None and msg in mk.unbound, mk.unbound
    assert mk.caches is None and built == []
    sched = _sched([4, 4])
    assert mk.step_ok(_runner_for(sched), sched, 1) is None
    assert mk.counts["unbound"] == 1 and mk.counts["mk"] == 0


def test_bind_perlayer_small_model():
    mk, runner, built = _mk()
    mk.bind_caches(runner)
    assert mk.unbound is None and mk.cache_form == "per-layer"
    assert mk.groups == {"nope": L, "rope": L, "idx": len(IDX)}


def test_bind_perlayer_blocked_past_limit(monkeypatch):
    monkeypatch.setenv("GLM52_DECODE_MK_PERLAYER_MAX_L", str(L - 1))
    mk, runner, built = _mk()
    mk.bind_caches(runner)
    assert mk.unbound is not None and "263,806 bundles" in mk.unbound
    assert built == []


# ------------------------------------------------------------------ step_ok + try_call (the runner contract)


def _sched(nums, spec=None, prompt=None, chunks_encoder=None):
    """A SchedulerOutput stand-in: nums = scheduled tokens per request; spec = drafts per request (default n - 1);
    prompt = still-in-prompt flags."""
    ids = [f"r{i}" for i in range(len(nums))]
    spec = [n - 1 for n in nums] if spec is None else spec
    return types.SimpleNamespace(
        num_scheduled_tokens=dict(zip(ids, nums)),
        scheduled_spec_decode_tokens={i: [-1] * k for i, k in zip(ids, spec) if k},
        scheduled_encoder_inputs={},
        total_num_scheduled_tokens=sum(nums),
        _prompt=[False] * len(nums) if prompt is None else prompt,
    )


def _runner_for(sched):
    ids = list(sched.num_scheduled_tokens)
    return types.SimpleNamespace(
        input_batch=types.SimpleNamespace(
            req_id_to_index={r: i for i, r in enumerate(ids)},
            num_computed_tokens_cpu=[0 if p else 100 for p in sched._prompt],
            num_prompt_tokens=[50] * len(ids),
        )
    )


class _MD:
    def __init__(self, S=8, W=256):
        self.query_start_loc = torch.zeros(S + 1, dtype=torch.int32)
        self.seq_lens = torch.zeros(S, dtype=torch.int32)
        self.block_tables = torch.zeros(S * W, dtype=torch.int32)


def _bound():
    mk, runner, built = _mk(stack=lambda g, layers: None)
    mk.bind_caches(runner)
    md = {"md": _MD()}
    mk._metadata = lambda: md["md"]
    return mk, md


def _forward(mk, pending, n_pad, **kw):
    """runner.forward_model's hook: consume the pending decision, MK or stock."""
    ids = torch.zeros(n_pad, dtype=torch.int32)
    out = mk.try_call(None, pending[1], ids, ids, **kw)
    return "stock" if out is None else ("mk", tuple(out.shape))


def test_runner_contract_counts_every_step():
    mk, md = _bound()
    steps = [
        (
            _sched([1500, 700], spec=[0, 0], prompt=[True, True]),
            2048,
            "stock",
        ),  # prefill
        (_sched([4, 4]), 8, ("mk", (8, 8))),  # B16
        (
            _sched([4, 4, 900], spec=[3, 3, 0], prompt=[False, False, True]),
            1024,
            "stock",
        ),  # mixed
        (_sched([4] * 6), 24, ("mk", (24, 8))),  # B32
        (_sched([3], spec=[0], prompt=[True]), 16, "stock"),  # prefix-hit prompt tail
        (_sched([1, 1], spec=[0, 0]), 8, "stock"),  # no drafts
        (_sched([4] * 3), 16, ("mk", (16, 8))),
    ]
    for sched, n_pad, want in steps:
        pending = ("step", mk.step_ok(_runner_for(sched), sched, 1))
        assert _forward(mk, pending, n_pad) == want, sched.num_scheduled_tokens
    c = mk.counts
    assert (c["mk"], c["mk_b16"], c["mk_b32"]) == (3, 2, 1)
    assert (c["prefill"], c["mixed"], c["no_draft"]) == (2, 1, 1)
    assert (
        mk.decisions
        == mk.consumed
        == len(steps)
        == sum(c[r] for r in REASONS if r not in ("embeds", "shape"))
    )
    assert mk.calls == 3


def test_toggle_off_declines_and_counts():
    mk, _ = _bound()
    mk.enabled = False
    sched = _sched([4, 4])
    assert mk.step_ok(_runner_for(sched), sched, 1) is None
    assert mk.counts["off"] == 1


def test_call_time_declines_fall_back_to_stock():
    mk, md = _bound()
    sched = _sched([4, 4])
    b = mk.step_ok(_runner_for(sched), sched, 1)
    assert _forward(mk, ("step", b), 8) == ("mk", (8, 8))  # fixes b16's shapes
    b = mk.step_ok(_runner_for(sched), sched, 1)
    assert _forward(mk, ("step", b), 8, inputs_embeds=torch.zeros(1)) == "stock"
    md["md"] = _MD(S=16)  # metadata no longer the bucket's
    b = mk.step_ok(_runner_for(sched), sched, 1)
    assert _forward(mk, ("step", b), 8) == "stock"
    assert (mk.counts["embeds"], mk.counts["shape"], mk.calls) == (1, 1, 1)
    assert mk.decisions == mk.consumed == 3


def test_harness_path_is_not_a_decision():
    mk, _ = _bound()
    ids = torch.zeros(16, dtype=torch.int32)
    out = mk.try_call(None, mk.bucket_for(16), ids, ids, harness=True)
    assert out is not None and mk.consumed == 0 and mk.decisions == 0 and mk.calls == 1


def test_rows_ok_matches_classify():
    mk, _ = _bound()
    assert mk.rows_ok([4, 4, 4, 4], 16) and mk.rows_ok([4] * 8, 32)
    assert not mk.rows_ok([4, 4, 4, 4], 32)  # the twin pads exactly to the bucket
    assert (
        not mk.rows_ok([5], 16)
        and not mk.rows_ok([1, 4], 16)
        and not mk.rows_ok([3] * 5, 16)
    )


def test_periodic_log(caplog):
    mk, _ = _bound()
    mk.log_every = 2
    seen = []
    mk.log_counts = lambda: seen.append(dict(mk.counts))
    sched = _sched([4, 4])
    for _ in range(5):
        mk.step_ok(_runner_for(sched), sched, 1)
    assert len(seen) == 2 and seen[-1]["mk"] == 4


def test_adapter_module_has_no_silent_paths():
    # every early return of _decide maps to a REASONS entry (guards against an added branch that returns a bare
    # None without counting)
    src = __import__("inspect").getsource(VA.Glm52DecodeMK._decide)
    for why in ("off", "unbound"):
        assert f'"{why}"' in src


# ------------------------------------------------------------------ token rows: runner buckets below the MK's (4, 8)


class _RecStep:
    """A stand-in compiled step that records the token operands it is handed and returns a row function of them
    (row i = ids[i] + 1000 * pos[i]), so the rows the runner keeps can be compared with an unpadded computation."""

    def __init__(self, T):
        self.T, self.seen = T, []

    def __call__(self, ids, pos, qsl, seq, bt, smalls, weights, caches):
        self.seen.append((tuple(ids.shape), tuple(pos.shape), ids.dtype, pos.dtype))
        assert ids.shape == (self.T,) and pos.shape == (self.T,)
        return (ids.float() + 1000.0 * pos.float())[:, None].expand(self.T, 8).clone()


def _bound_rec():
    mk, md = _bound()
    mk._steps = {t: _RecStep(t) for t in ROWS}
    return mk, md


@pytest.mark.parametrize(
    "nums, n_pad, bucket",
    [
        ([4], 4, 16),  # TPU_TOKEN_BUCKET_EXTRA=4: one request
        ([3], 4, 16),
        ([4, 4], 8, 16),  # TPU_TOKEN_BUCKET_EXTRA=8: two requests
        ([2, 3, 2], 8, 16),
        ([4] * 4, 16, 16),  # the default 16 bucket
        # 15 tokens but 5 requests > 16 // 4: the b32 program on a 16-row runner bucket
        ([3] * 5, 16, 32),
        ([4] * 8, 32, 32),
        ([2] * 8, 16, 32),
    ],
)
def test_runner_buckets_map_to_warmed_programs(nums, n_pad, bucket):
    mk, _ = _bound_rec()
    sched = _sched(nums)
    b = mk.step_ok(_runner_for(sched), sched, 1)
    assert b == bucket
    T = sum(nums)
    ids = torch.zeros(n_pad, dtype=torch.int32)
    pos = torch.zeros(n_pad, dtype=torch.int32)
    ids[:T] = torch.arange(1, T + 1, dtype=torch.int32)
    pos[:T] = torch.arange(100, 100 + T, dtype=torch.int32)
    out = mk.try_call(None, b, ids, pos)
    assert out is not None and tuple(out.shape) == (n_pad, 8)
    assert mk._steps[bucket].seen == [((bucket,), (bucket,), torch.int32, torch.int32)]
    assert all(not s.seen for t, s in mk._steps.items() if t != bucket)
    want = (ids.float() + 1000.0 * pos.float())[:, None].expand(n_pad, 8)
    # every row the runner keeps, incl. its own padding rows (zeros in -> zeros out)
    assert torch.equal(out, want)
    assert mk.counts["shape"] == 0 and mk.calls == 1


def test_to_bucket_rows_pads_like_the_in_step_fit():
    x = torch.arange(1, 5, dtype=torch.int32)
    for t in (16, 32):
        y = VA.to_bucket_rows(x, t)
        assert y.shape == (t,) and y.dtype == torch.int32 and y.is_contiguous()
        assert torch.equal(y, VA.fit_rows(x, t))
        assert not y[4:].any()
    full = torch.arange(16, dtype=torch.int32)
    # the runner's tensor untouched at n == t
    assert VA.to_bucket_rows(full, 16) is full


@pytest.mark.parametrize(
    "ids, pos, bucket, msg",
    [
        (torch.zeros(32, dtype=torch.int32), None, 16, "32 token rows not in 1..16"),
        (torch.zeros(33, dtype=torch.int32), None, 32, "33 token rows not in 1..32"),
        (torch.zeros(0, dtype=torch.int32), None, 16, "0 token rows not in 1..16"),
        (torch.zeros(2, 8, dtype=torch.int32), None, 16, "is not 1-D"),
        (
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
            16,
            "positions shape (16,) != input_ids (8,)",
        ),
        (
            torch.zeros(8, dtype=torch.int64),
            None,
            16,
            "input_ids dtype torch.int64 != int32",
        ),
        (
            torch.zeros(8, dtype=torch.int32),
            torch.zeros(8, dtype=torch.int64),
            16,
            "positions dtype torch.int64 != int32",
        ),
    ],
)
def test_unexpected_token_rows_decline_with_reason(ids, pos, bucket, msg, monkeypatch):
    mk, _ = _bound_rec()
    pos = ids if pos is None else pos
    warned = []
    # the module logger directly: vLLM's logging config may not propagate to caplog's root handler
    monkeypatch.setattr(VA.logger, "warning", lambda f, *a: warned.append(f % a))
    assert mk.try_call(None, bucket, ids, pos) is None
    assert mk.try_call(None, bucket, ids, pos) is None
    assert mk.counts["shape"] == 2 and mk.calls == 0
    # never reached a compiled program
    assert all(not s.seen for s in mk._steps.values())
    why = VA.token_rows_mismatch(bucket, ids, pos)
    assert why and msg in why
    # loud, once per distinct reason
    assert len(warned) == 1 and msg in warned[0], warned
    assert "unexpected shape would recompile" in warned[0]


def test_token_rows_padding_never_recompiles_the_step():
    """The real mechanism on CPU: a torch.compile(fullgraph, dynamic=False) step that pads with the adapter's own
    `fit_rows`, warmed at 16 and 32 rows like `warmup`. Through `to_bucket_rows`, runner buckets 4 / 8 / 16 / 32 run
    the two warmed graphs (no new compile) and keep exactly the rows of an in-step pad; handing the step the raw 4 rows
    (the e8301f284 behaviour) compiles a third graph (positive control)."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(
            tuple(tuple(x.shape) for x in example_inputs if isinstance(x, torch.Tensor))
        )
        return gm.forward

    torch._dynamo.reset()
    emb = torch.nn.Embedding(64, 8)
    steps = {}
    for T in ROWS:

        def step(ids, positions, T=T):
            p = VA.fit_rows(positions.reshape(-1).to(torch.int32), T)
            h = emb(VA.fit_rows(ids.reshape(-1), T)).float()
            return h * (1.0 + p.float()[:, None])

        steps[T] = torch.compile(step, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for T in ROWS:  # warmup: [T] int32
            steps[T](
                torch.zeros(T, dtype=torch.int32),
                torch.arange(T, dtype=torch.int32) % 4,
            )
        warm = len(graphs)
        assert warm == 2
        for n, T in ((4, 16), (8, 16), (16, 16), (16, 32), (8, 32), (32, 32)):
            ids = torch.randint(1, 64, (n,), dtype=torch.int32)
            pos = torch.randint(0, 1000, (n,), dtype=torch.int32)
            got = steps[T](VA.to_bucket_rows(ids, T), VA.to_bucket_rows(pos, T))[:n]
            ref = emb(ids).float() * (1.0 + pos.float()[:, None])
            assert torch.equal(got, ref), (n, T)
            assert len(graphs) == warm, (n, T, graphs)
        raw = torch.zeros(4, dtype=torch.int32)
        steps[16](raw, raw)  # the raw 4 rows: a new graph
        assert len(graphs) == warm + 1
    torch._dynamo.reset()
