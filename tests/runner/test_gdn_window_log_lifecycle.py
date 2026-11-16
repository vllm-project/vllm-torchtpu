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
"""Lifecycle of the GDN window log (VLLM_TPU_GDN_WINDOW_LOG) through the runner.

The runner's own metadata and copy paths run on CPU: the seed-copy collector
and flush (`_collect_mamba_state_seed_copies`, `_flush_mamba_state_seed_copies`),
the staged metadata walk (`attention_metadata._staged_walk_products`, class
gather), the new-owner offset reset, the post-sampling offset scatter and the
copy-on-write path (`_apply_kv_cache_block_copies`). The pool copy is the
Pallas block copy (`pool_adapters.copy_blocks`) in TPU interpret mode, or a
whole-row torch copy where the test says so.

The GDN kernel is a NumPy model of its contract (f7-kernel/result.md, "Log
layout contract"): in log mode a step loads checkpoint block 0 and applies the
first read-offset logged positions, a BATCHED step stores the state after
window position 0 to block 0 and the operands of positions 1..K to the log
(identity on masked rows), a PER_SEQ step stores its final state to block 0
and leaves the log. In select mode (today) a step loads checkpoint
`read_offset` and a BATCHED step stores one state per window position. The
PLE group always uses select mode.

The check: every load, for every GDN and PLE group, equals a reference
recurrence over the request's accepted tokens, bit for bit.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu.layers.core.attention_metadata import _staged_walk_products
from vllm_torchtpu.runner.tpu_runner import INVALID_TOKEN_ID, TPUModelRunner

K = 3  # draft positions; window K + 1
B = 8  # mamba block size in tokens (the column stride)
N_KQ, N_V, D = 1, 2, 4  # key heads, value heads, head dim (d_k == d_v)
C = 2 * N_KQ * D + N_V * D  # conv channels
P = 6  # PLE history width
LANES = 128
ROWS = 24
NUM_BLOCKS = 192
MAX_LEN = 136
WIDTH = -(-MAX_LEN // B) + K

# Rows of a block page. GDN: state, conv rows, then the log in the contract's
# order (gates, keys, updates, conv inputs). PLE: its history rows.
S_ROW = 0
CONV_ROW = 1  # 3 rows
G_ROW = 4  # K rows: lanes [0, N_V)
K_ROW = G_ROW + K  # K * N_KQ rows
U_ROW = K_ROW + K * N_KQ  # K * N_V rows
X_ROW = U_ROW + K * N_V  # K rows: lanes [0, C)
assert X_ROW + K <= ROWS
PLE_ROW = 0  # 3 rows

GDN_GIDS = (1, 2)
PLE_GID = 3
MAMBA_GIDS = GDN_GIDS + (PLE_GID,)


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _bits_equal(a, b) -> bool:
    return np.array_equal(_f32(a).view(np.uint32), _f32(b).view(np.uint32))


def _seed(*parts) -> int:
    value = 0x9E3779B9
    for part in parts:
        value = (value * 1_000_003 + int(part)) % (1 << 63)
    return value


@dataclasses.dataclass
class Ops:
    """One token's operands for one GDN layer."""

    g: np.ndarray  # [N_V]
    k: np.ndarray  # [N_KQ, D]
    v: np.ndarray  # [N_V, D]
    beta: np.ndarray  # [N_V]
    x: np.ndarray  # [C] conv input


def token_ops(layer: int, pos: int, tok: int) -> Ops:
    rng = np.random.default_rng(_seed(layer, pos, tok))
    k = _f32(rng.standard_normal((N_KQ, D)))
    k = _f32(k / np.sqrt((k * k).sum(-1, keepdims=True) + np.float32(1e-6)))
    return Ops(
        g=_f32(rng.uniform(0.6, 1.0, N_V)),
        k=k,
        v=_f32(rng.standard_normal((N_V, D))),
        beta=_f32(rng.uniform(0.1, 0.9, N_V)),
        x=_f32(rng.standard_normal(C)),
    )


def ple_input(pos: int, tok: int) -> np.ndarray:
    return _f32(np.random.default_rng(_seed(99, pos, tok)).standard_normal(P))


def _repeat_k(k):
    return np.repeat(k, N_V // N_KQ, axis=0)  # [N_V, D]


def gdn_step(S, conv, ops: Ops):
    """One token, in the kernel's order: scale, then the rank-1 term.

    Returns the new state and conv, and the update vector the log stores."""
    k_rep = _repeat_k(ops.k)
    s_upd = _f32(S * ops.g[:, None, None])
    v_hat = _f32(np.einsum("hk,hkv->hv", k_rep, s_upd))
    u = _f32(ops.beta[:, None] * _f32(ops.v - v_hat))
    S_new = _f32(s_upd + _f32(k_rep[:, :, None] * u[:, None, :]))
    conv_new = _f32(np.concatenate([conv, ops.x[None]])[-3:])
    return S_new, conv_new, u


def gdn_fold(S, conv, g, k, u, x):
    """One logged position: the same two statements as the kernel's update."""
    k_rep = _repeat_k(k)
    S = _f32(S * g[:, None, None])
    S = _f32(S + _f32(k_rep[:, :, None] * u[:, None, :]))
    conv = _f32(np.concatenate([conv, x[None]])[-3:])
    return S, conv


def ple_step(hist, p):
    return _f32(np.concatenate([hist, p[None]])[-3:])


# --- pool page access ------------------------------------------------------


def read_gdn(pool, block):
    page = pool[block].numpy()
    S = page[S_ROW, : N_V * D * D].reshape(N_V, D, D).copy()
    conv = page[CONV_ROW : CONV_ROW + 3, :C].copy()
    return S, conv


def write_gdn(pool, block, S, conv):
    pool[block, S_ROW, : N_V * D * D] = torch.from_numpy(_f32(S).reshape(-1))
    pool[block, CONV_ROW : CONV_ROW + 3, :C] = torch.from_numpy(_f32(conv))


def read_log(pool, block, j):
    page = pool[block].numpy()
    g = page[G_ROW + j, :N_V].copy()
    k = page[K_ROW + j * N_KQ : K_ROW + (j + 1) * N_KQ, :D].copy()
    u = page[U_ROW + j * N_V : U_ROW + (j + 1) * N_V, :D].copy()
    x = page[X_ROW + j, :C].copy()
    return g, k, u, x


def write_log(pool, block, j, g, k, u, x):
    pool[block, G_ROW + j, :N_V] = torch.from_numpy(_f32(g))
    pool[block, K_ROW + j * N_KQ : K_ROW + (j + 1) * N_KQ, :D] = torch.from_numpy(
        _f32(k)
    )
    pool[block, U_ROW + j * N_V : U_ROW + (j + 1) * N_V, :D] = torch.from_numpy(
        _f32(u)
    )
    pool[block, X_ROW + j, :C] = torch.from_numpy(_f32(x))


def _assert_changed_within(before, after, allowed, what) -> None:
    changed = set((before != after).nonzero().flatten().tolist())
    assert changed <= set(allowed) | {0}, (what, sorted(changed - set(allowed)))


def log_nonempty(pool, block) -> bool:
    return bool(pool[block, G_ROW:X_ROW + K].abs().sum() > 0)


def read_ple(pool, block):
    return pool[block, PLE_ROW : PLE_ROW + 3, :P].numpy().copy()


def write_ple(pool, block, hist):
    pool[block, PLE_ROW : PLE_ROW + 3, :P] = torch.from_numpy(_f32(hist))


# --- copies ----------------------------------------------------------------


def pallas_block_copy(raws, src, dst):
    """`copy_mamba_state_blocks` with its Pallas body in TPU interpret mode."""
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels import pool_adapters

    for pool in raws:
        with pltpu.force_tpu_interpret_mode():
            out = pool_adapters.copy_blocks(
                jnp.asarray(pool.numpy()),
                jnp.asarray(src.numpy().astype(np.int32)),
                jnp.asarray(dst.numpy().astype(np.int32)),
            )
        pool.copy_(torch.from_numpy(np.array(out)))


def torch_block_copy(raws, src, dst):
    for pool in raws:
        pool[dst.long()] = pool[src.long()].clone()


# --- the engine ------------------------------------------------------------


@dataclasses.dataclass
class Req:
    rid: str
    prompt: list[int]
    kind: str  # "local", "landing" or "prefix"
    committed: list[int] = dataclasses.field(default_factory=list)
    next_token: int | None = None
    phase: str = "prefill"
    rows: dict = dataclasses.field(default_factory=dict)  # gid -> [WIDTH] ids
    held_col: int | None = None
    preempted: bool = False
    # Ref states over `committed`, per GDN gid, and the PLE history.
    ref: dict = dataclasses.field(default_factory=dict)
    force_plain: bool = False
    force_per_seq: bool = False

    @property
    def computed(self) -> int:
        return len(self.committed)


class Engine:
    """Scheduler, runner metadata/copy paths and a kernel model, on CPU."""

    def __init__(self, *, window_log: bool, runner_log: bool, copy, seed: int):
        self.window_log = window_log  # the kernel model's mode
        self.rng = np.random.default_rng(seed)
        self.pool = torch.zeros((NUM_BLOCKS, ROWS, LANES), dtype=torch.float32)
        self.offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
        self.free = list(range(NUM_BLOCKS - 1, 0, -1))
        self.reqs: dict[str, Req] = {}
        self.next_id = 0
        self.stats: dict[str, int] = {}
        self.copy = copy
        self.tables = {
            gid: torch.zeros((16, WIDTH), dtype=torch.int32) for gid in MAMBA_GIDS
        }
        input_batch = SimpleNamespace(
            req_ids=[None] * 16,
            num_computed_tokens_cpu=np.zeros(16, dtype=np.int64),
            block_table={
                gid: SimpleNamespace(get_cpu_tensor=lambda t=t: t)
                for gid, t in self.tables.items()
            },
        )
        r = SimpleNamespace(
            requests={},
            _mamba_copy_plan=[(gid, [self.pool]) for gid in MAMBA_GIDS],
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=SimpleNamespace(num_speculative_tokens=K),
            _pre_async_results=None,
            _mamba_state_pos={},
            _mamba_offset_seeded=set(),
            _mamba_state_block_size=B,
            _pool_block_split=1,
            bm_pool_layout=None,
            _mamba_ckpt_window=K + 1,
            _mamba_window_log_gids=frozenset(GDN_GIDS if runner_log else ()),
            cache_config=SimpleNamespace(block_size=B),
            device=torch.device("cpu"),
            mamba_slot_read_offsets=self.offsets,
            _pending_mamba_state_copies=[],
            input_batch=input_batch,
            kv_cache_raw_tensors=[self.pool],
            _read_offset_scratch={},
            _attn_metadata_builder_ctx=None,
        )
        r._bucket_len = TPUModelRunner._bucket_len
        r._pad_to_bucket = TPUModelRunner._pad_to_bucket
        r._dp_lockstep_enabled = lambda: False
        for name in (
            "_spec_seed_copies",
            "_flush_mamba_state_seed_copies",
            "_copy_mamba_state_blocks",
            "_collect_mamba_state_seed_copies",
            "_mamba_state_index_groups",
            "_read_offset_reset_scratch",
            "_reset_read_offsets_for_new_requests",
            "_update_mamba_slot_read_offsets",
            "_apply_kv_cache_block_copies",
            "is_unified_pool_used",
        ):
            setattr(r, name, getattr(TPUModelRunner, name).__get__(r))
        self.runner = r

    # -- bookkeeping --

    def count(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    def alloc(self) -> int:
        return self.free.pop()

    def release(self, block: int) -> None:
        if block:
            self.free.append(block)

    def free_rows(self, req: Req) -> None:
        for gid, row in req.rows.items():
            for col, block in enumerate(row):
                self.release(block)
                row[col] = 0

    def _ref_init(self, req: Req) -> None:
        req.ref = {
            gid: (np.zeros((N_V, D, D), np.float32), np.zeros((3, C), np.float32))
            for gid in GDN_GIDS
        }
        req.ref[PLE_GID] = np.zeros((3, P), np.float32)

    def _ref_commit(self, req: Req, tokens: list[int]) -> None:
        for tok in tokens:
            pos = req.computed
            for gid in GDN_GIDS:
                S, conv = req.ref[gid]
                S, conv, _ = gdn_step(S, conv, token_ops(gid, pos, tok))
                req.ref[gid] = (S, conv)
            req.ref[PLE_GID] = ple_step(req.ref[PLE_GID], ple_input(pos, tok))
            req.committed.append(tok)

    def new_token(self) -> int:
        return int(self.rng.integers(1, 1 << 30))

    # -- request arrival --

    def admit(self, kind: str) -> Req:
        rid = f"r{self.next_id}"
        self.next_id += 1
        n = int(self.rng.integers(B + 2, 5 * B))
        req = Req(rid, [self.new_token() for _ in range(n)], kind)
        req.rows = {gid: [0] * WIDTH for gid in MAMBA_GIDS}
        self._ref_init(req)
        self.reqs[rid] = req
        self.runner.requests[rid] = object()
        if kind == "landing":
            self._land(req, len(req.prompt) - 1)
            req.phase = "decode"
            req.next_token = req.prompt[-1]
            req.force_plain = True
        elif kind == "prefix":
            self._land(req, (len(req.prompt) - 1) // B * B)
        return req

    def _land(self, req: Req, n_tokens: int) -> None:
        """A state at `n_tokens` landed (PD repack) or found in the prefix cache.

        Writes the state and conv rows (or the PLE history) of the running
        entry, leaving its log rows and read offset as the previous owner
        left them, which the test records."""
        self._ref_commit(req, req.prompt[:n_tokens])
        col = (n_tokens - 1) // B
        for gid in MAMBA_GIDS:
            for t in range(K + 1):
                req.rows[gid][col + t] = self.alloc()
            block = req.rows[gid][col]
            if gid in GDN_GIDS:
                if log_nonempty(self.pool, block) and int(self.offsets[block]):
                    self.count(f"{req.kind}_on_stale_log_and_offset")
                write_gdn(self.pool, block, *req.ref[gid])
            else:
                write_ple(self.pool, block, req.ref[PLE_GID])
        req.held_col = col

    # -- one engine step --

    def step(self, events: bool = True) -> None:
        finished: set[str] = set()
        preempted: set[str] = set()
        resumed: set[str] = set()
        if events:
            self._events(finished, preempted, resumed)
        # _update_states' mamba invalidation, as the runner runs it.
        r = self.runner
        for rid in finished | preempted | resumed:
            r._mamba_state_pos.pop(rid, None)
        r._mamba_offset_seeded.difference_update(finished | preempted | resumed)
        for rid in finished:
            r.requests.pop(rid, None)
        if events:
            self._maybe_cow()

        running = [q for q in self.reqs.values() if not q.preempted]
        if not running:
            return
        plan = []
        for q in running:
            if q.phase == "prefill":
                left = len(q.prompt) - q.computed
                n = min(left, int(self.rng.integers(1, 3 * B)))
                plan.append((q, "per_seq", q.prompt[q.computed : q.computed + n]))
            elif q.force_per_seq:
                q.force_per_seq = False
                plan.append((q, "per_seq_decode", [q.next_token]))
            elif q.force_plain or self.rng.random() < 0.1:
                q.force_plain = False
                plan.append((q, "plain", [q.next_token]))
            else:
                drafts = [self.new_token() for _ in range(K)]
                plan.append((q, "verify", [q.next_token, *drafts]))
        # [decode][verify][prefill], the runner's batch order.
        rank = {"plain": 0, "per_seq_decode": 0, "verify": 1, "per_seq": 2}
        plan.sort(key=lambda e: rank[e[1]])
        for q, _, toks in plan:
            self._allocate(q, q.computed + len(toks))

        n = len(plan)
        ib = r.input_batch
        ib.req_ids = [q.rid for q, _, _ in plan] + [None] * (16 - n)
        ib.num_computed_tokens_cpu[:] = 0
        for i, (q, _, _) in enumerate(plan):
            ib.num_computed_tokens_cpu[i] = q.computed
        for gid, table in self.tables.items():
            table.zero_()
            for i, (q, _, _) in enumerate(plan):
                table[i] = torch.tensor(q.rows[gid], dtype=torch.int32)
        sched = SimpleNamespace(
            num_scheduled_tokens={q.rid: len(t) for q, _, t in plan}
        )
        before = dict(r._mamba_state_pos)
        offsets_before = self.offsets.clone()
        r._collect_mamba_state_seed_copies(sched, 0, n)
        self._record_crossings(plan, before)
        # Padded entries (bucket padding of the copy and offset programs) touch
        # only the null block: every other staged pair is a real crossing.
        crossing_dsts = set()
        for q, _, toks in plan:
            prev = before.get(q.rid, (q.computed - 1) // B)
            curr = (q.computed + len(toks) - 1) // B
            if 0 <= prev != curr:
                crossing_dsts |= {q.rows[gid][curr] for gid in MAMBA_GIDS}
        for _, src_t, dst_t in r._pending_mamba_state_copies:
            for src, dst in zip(src_t.tolist(), dst_t.tolist()):
                assert dst in crossing_dsts or (src, dst) == (0, 0), (src, dst)
        _assert_changed_within(offsets_before, self.offsets, crossing_dsts, "offsets")

        width = 16
        seq_lens = torch.zeros(width, dtype=torch.int64)
        for i, (q, _, toks) in enumerate(plan):
            seq_lens[i] = q.computed + len(toks)
        flat = torch.cat([self.tables[gid].reshape(-1) for gid in MAMBA_GIDS])
        table_geom = tuple((j * width * WIDTH, width, WIDTH) for j in range(3))
        mamba_geom = tuple((j, WIDTH, B) for j in range(3))
        products = _staged_walk_products(
            flat, seq_lens, table_geom, mamba_geom, K + 1,
            include_tables=False, class_gather=True,
        )
        states = dict(zip(MAMBA_GIDS, products[:3]))
        ckpts = dict(zip(MAMBA_GIDS, products[3:]))
        # Padded request slots name the null block only, so the kernel's
        # zero-size DMAs read and write no state page for them.
        assert n < width
        for gid in MAMBA_GIDS:
            assert not states[gid][n:].any() and not ckpts[gid][n:].any(), gid
        self.count("padded_rows", width - n)
        real_states = {int(states[gid][i]) for gid in MAMBA_GIDS for i in range(n)}

        stale = {
            int(states[gid][i]): int(self.offsets[int(states[gid][i])])
            for gid in GDN_GIDS
            for i in range(n)
        }
        pool_before = self.pool.clone()
        r._flush_mamba_state_seed_copies()
        changed = (self.pool != pool_before).flatten(1).any(1).nonzero().flatten()
        assert set(changed.tolist()) <= crossing_dsts, (changed.tolist(), crossing_dsts)
        offsets_before = self.offsets.clone()
        r._attn_metadata_builder_ctx = SimpleNamespace(
            mamba_state_indices=None,
            unified_mamba_state_indices=[states[gid] for gid in MAMBA_GIDS],
        )
        r._reset_read_offsets_for_new_requests(0, n)

        accepted = []
        for i, (q, kind, toks) in enumerate(plan):
            offs = {gid: int(self.offsets[int(states[gid][i])]) for gid in MAMBA_GIDS}
            if kind == "per_seq_decode" and offs[GDN_GIDS[0]] > 0:
                self.count("per_seq_with_offset")
            for gid in GDN_GIDS:
                block = int(states[gid][i])
                if q.computed and stale.get(block) and not offs[gid]:
                    self.count("stale_offset_cleared")
            self._run_request(q, kind, toks, i, states, ckpts, offs)
            if kind == "verify":
                a = int(self.rng.integers(1, K + 2))
            elif kind == "per_seq" and q.computed + len(toks) < len(q.prompt):
                a = 0
            else:
                a = 1
            accepted.append(a)

        next_tokens = torch.full((width, K + 1), INVALID_TOKEN_ID, dtype=torch.int64)
        for i, a in enumerate(accepted):
            next_tokens[i, :a] = 7
        r._update_mamba_slot_read_offsets(
            [states[gid] for gid in MAMBA_GIDS], next_tokens, n
        )
        _assert_changed_within(offsets_before, self.offsets, real_states, "offsets")

        for (q, kind, toks), a in zip(plan, accepted):
            q.held_col = (q.computed + len(toks) - 1) // B
            if kind == "per_seq":
                self._ref_commit(q, toks)
                if q.computed == len(q.prompt):
                    q.phase = "decode"
                    q.next_token = self.new_token()
                continue
            self._ref_commit(q, toks[:a])
            if kind == "verify":
                self.count(f"accepted_{a}")
                if a > 1 and self.rng.random() < 0.15:
                    q.force_per_seq = True
            q.next_token = self.new_token()

    def _allocate(self, q: Req, num_tokens: int) -> None:
        """Align-mode MambaManager with K speculative blocks: the state column
        and the K after it; the previous step's state column stays one more
        step, earlier entries are freed."""
        curr = (num_tokens - 1) // B
        for gid in MAMBA_GIDS:
            row = q.rows[gid]
            for col in range(WIDTH):
                keep = curr <= col <= curr + K or col == q.held_col
                if not keep and row[col]:
                    self.release(row[col])
                    row[col] = 0
            for col in range(curr, curr + K + 1):
                if not row[col]:
                    row[col] = self.alloc()

    def _record_crossings(self, plan, before) -> None:
        for q, kind, toks in plan:
            prev = before.get(q.rid, (q.computed - 1) // B)
            curr = (q.computed + len(toks) - 1) // B
            if prev < 0 or prev == curr:
                continue
            self.count("crossings")
            if kind == "verify":
                # Window position of the first token in the new column.
                self.count(f"crossing_align_{curr * B - q.computed}")
            src = q.rows[GDN_GIDS[0]][prev]
            if int(self.offsets[src]) > 0:
                self.count("crossing_with_offset")
            if q.rid not in self.runner._mamba_offset_seeded:
                self.count("crossing_new_owner")

    def _run_request(self, q, kind, toks, i, states, ckpts, offs) -> None:
        has_initial = q.computed > 0
        batched = kind in ("plain", "verify")
        for gid in GDN_GIDS:
            block = int(states[gid][i])
            row_ckpts = [int(c) for c in ckpts[gid][i]]
            off = offs[gid]
            if not has_initial:
                S = np.zeros((N_V, D, D), np.float32)
                conv = np.zeros((3, C), np.float32)
            elif self.window_log:
                S, conv = read_gdn(self.pool, block)
                for j in range(off):
                    S, conv = gdn_fold(S, conv, *read_log(self.pool, block, j))
            else:
                S, conv = read_gdn(self.pool, row_ckpts[off])
            ref_S, ref_conv = q.ref[gid]
            assert _bits_equal(S, ref_S) and _bits_equal(conv, ref_conv), (
                f"{q.rid} gid {gid} {kind} at {q.computed}: loaded state differs "
                f"from the reference (offset {off})"
            )
            outs = []
            for t, tok in enumerate(toks):
                ops = token_ops(gid, q.computed + t, tok)
                S, conv, u = gdn_step(S, conv, ops)
                outs.append((S, conv, ops, u))
            if not batched:
                write_gdn(self.pool, row_ckpts[0], outs[-1][0], outs[-1][1])
            elif self.window_log:
                write_gdn(self.pool, block, outs[0][0], outs[0][1])
                for j in range(K):
                    if j + 1 < len(outs):
                        _, _, ops, u = outs[j + 1]
                        write_log(self.pool, block, j, ops.g, ops.k, u, ops.x)
                    else:
                        pad = _f32(self.rng.standard_normal(C))
                        write_log(
                            self.pool, block, j,
                            np.ones(N_V, np.float32),
                            np.zeros((N_KQ, D), np.float32),
                            np.zeros((N_V, D), np.float32),
                            pad,
                        )
            else:
                for t, (S_t, conv_t, _, _) in enumerate(outs):
                    write_gdn(self.pool, row_ckpts[t], S_t, conv_t)
        # PLE: select mode in both runs.
        row_ckpts = [int(c) for c in ckpts[PLE_GID][i]]
        off = offs[PLE_GID]
        hist = (
            read_ple(self.pool, row_ckpts[off])
            if has_initial
            else np.zeros((3, P), np.float32)
        )
        assert _bits_equal(hist, q.ref[PLE_GID]), (
            f"{q.rid} PLE {kind} at {q.computed}: loaded history differs (offset {off})"
        )
        hists = []
        for t, tok in enumerate(toks):
            hist = ple_step(hist, ple_input(q.computed + t, tok))
            hists.append(hist)
        if batched:
            for t, h in enumerate(hists):
                write_ple(self.pool, row_ckpts[t], h)
        else:
            write_ple(self.pool, row_ckpts[0], hists[-1])

    def _events(self, finished, preempted, resumed) -> None:
        live = [q for q in self.reqs.values() if not q.preempted]
        for q in list(live):
            if q.phase == "decode" and q.computed + K + 2 >= MAX_LEN:
                finished.add(q.rid)
        for q in [q for q in self.reqs.values() if q.preempted]:
            if self.rng.random() < 0.3:
                q.preempted = False
                resumed.add(q.rid)
                # Recompute from scratch: the accepted tokens become the prompt.
                q.committed, q.prompt, q.phase = [], list(q.committed), "prefill"
                self._ref_init(q)
                q.held_col = None
                self.count("resumed")
        decoding = [q for q in live if q.phase == "decode" and q.rid not in finished]
        if decoding and self.rng.random() < 0.04:
            q = decoding[int(self.rng.integers(len(decoding)))]
            q.preempted = True
            preempted.add(q.rid)
            self.free_rows(q)
            q.held_col = None
            self.count("preempted")
        if decoding and self.rng.random() < 0.05:
            q = decoding[int(self.rng.integers(len(decoding)))]
            if q.rid not in preempted:
                finished.add(q.rid)
        for rid in finished:
            self.free_rows(self.reqs.pop(rid))
            self.count("finished")
        while len([q for q in self.reqs.values() if not q.preempted]) < 6:
            kind = ("local", "landing", "prefix")[int(self.rng.integers(3))]
            self.admit(kind)
            self.count(f"admitted_{kind}")

    def _maybe_cow(self) -> None:
        """Copy-on-write of a decoding request's state blocks.

        Today's checkpoint selection only survives a copy of the state block
        when the offset is 0, so a select-mode group is copied only then; the
        window log carries its pending positions in the block, so a log-mode
        group is copied at any offset."""
        cands = [
            q
            for q in self.reqs.values()
            if not q.preempted and q.phase == "decode" and q.held_col is not None
        ]
        if not cands or self.rng.random() > 0.08:
            return
        q = cands[int(self.rng.integers(len(cands)))]
        col = q.held_col
        copies = []
        for gid in MAMBA_GIDS:
            src = q.rows[gid][col]
            off = int(self.offsets[src])
            if off and not (self.window_log and gid in GDN_GIDS):
                continue
            copies.append((gid, src, off))
        if not copies:
            return
        pairs = []
        for gid, src, off in copies:
            dst = self.alloc()
            q.rows[gid][col] = dst
            pairs.append(SimpleNamespace(src_block_id=src, dst_block_id=dst))
            if off and gid in GDN_GIDS:
                self.count("cow_with_offset")
        self.runner._apply_kv_cache_block_copies(pairs)
        for p in pairs:
            self.release(p.src_block_id)
        self.count("cow")


def run_engine(*, window_log, runner_log, copy, seed, steps):
    engine = Engine(window_log=window_log, runner_log=runner_log, copy=copy, seed=seed)
    for _ in range(steps):
        engine.step()
    return engine.stats


@pytest.fixture
def block_copy(monkeypatch, request):
    copy = request.param
    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", copy)
    return copy


@pytest.mark.parametrize("block_copy", [pallas_block_copy], indirect=True)
def test_window_log_lifecycle_matches_reference(block_copy):
    stats = run_engine(
        window_log=True, runner_log=True, copy=block_copy, seed=1, steps=260
    )
    # Every case of gate 2 happened, several times.
    for key in (
        "crossing_align_0",
        "crossing_align_1",
        "crossing_align_2",
        "crossing_align_3",
        "crossing_with_offset",
        "crossing_new_owner",
        "landing_on_stale_log_and_offset",
        "stale_offset_cleared",
        "preempted",
        "resumed",
        "cow_with_offset",
        "per_seq_with_offset",
        "admitted_local",
        "admitted_landing",
        "admitted_prefix",
        "padded_rows",
    ):
        assert stats.get(key, 0) >= 2, (key, stats)


@pytest.mark.parametrize("block_copy", [torch_block_copy], indirect=True)
@pytest.mark.parametrize("seed", [2, 3, 4])
def test_window_log_lifecycle_more_seeds(block_copy, seed):
    stats = run_engine(
        window_log=True, runner_log=True, copy=block_copy, seed=seed, steps=400
    )
    assert stats.get("crossing_with_offset", 0) >= 2, stats


@pytest.mark.parametrize("block_copy", [torch_block_copy], indirect=True)
def test_select_mode_lifecycle_matches_reference(block_copy):
    """The harness on today's checkpoint selection: the reference holds too."""
    run_engine(window_log=False, runner_log=False, copy=block_copy, seed=1, steps=260)


@pytest.mark.parametrize("block_copy", [torch_block_copy], indirect=True)
def test_window_log_kernel_with_select_runner_fails(block_copy):
    """The log kernel behind today's crossing (seed checkpoint[offset], reset
    the offset) resumes from a wrong state: the harness catches it."""
    with pytest.raises(AssertionError, match="differs"):
        run_engine(
            window_log=True, runner_log=False, copy=block_copy, seed=1, steps=260
        )


@pytest.mark.parametrize("block_copy", [torch_block_copy], indirect=True)
def test_padding_that_names_a_page_fails(block_copy):
    """The padded-row checks have teeth: bucket padding that names a real
    block instead of the null block is caught."""
    engine = Engine(window_log=True, runner_log=True, copy=block_copy, seed=1)
    pad_list = TPUModelRunner._pad_to_bucket.__func__

    def pad_with_block_one(cls, values, pad=0):
        bad = [1] * len(pad) if isinstance(pad, (list, tuple)) else 1
        return pad_list(cls, values, type(pad)(bad) if isinstance(pad, tuple) else bad)

    engine.runner._pad_to_bucket = classmethod(pad_with_block_one).__get__(
        None, TPUModelRunner
    )
    with pytest.raises(AssertionError):
        for _ in range(200):
            engine.step()


# --- the crossing site, pair by pair ---------------------------------------


def _crossing_runner(offsets, rows_by_gid, *, log_gids, seeded=True, computed=8):
    tables = {}
    for gid, rows in rows_by_gid.items():
        t = torch.zeros((4, 8), dtype=torch.int32)
        t[0, : len(rows)] = torch.tensor(rows, dtype=torch.int32)
        tables[gid] = SimpleNamespace(get_cpu_tensor=lambda t=t: t)
    pool = torch.zeros((32, 4))
    r = SimpleNamespace(
        requests={"a": object()},
        _mamba_copy_plan=[(gid, [pool]) for gid in rows_by_gid],
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        _pre_async_results=None,
        _mamba_state_pos={"a": 0},
        _mamba_offset_seeded={"a"} if seeded else set(),
        _mamba_state_block_size=B,
        _pool_block_split=1,
        bm_pool_layout=None,
        _mamba_ckpt_window=K + 1,
        _mamba_window_log_gids=frozenset(log_gids),
        device=torch.device("cpu"),
        mamba_slot_read_offsets=offsets,
        _pending_mamba_state_copies=[],
        input_batch=SimpleNamespace(
            req_ids=["a", None, None, None],
            num_computed_tokens_cpu=np.array([computed, 0, 0, 0]),
            block_table=tables,
        ),
    )
    r._bucket_len = TPUModelRunner._bucket_len
    r._pad_to_bucket = TPUModelRunner._pad_to_bucket
    r._spec_seed_copies = TPUModelRunner._spec_seed_copies.__get__(r)
    return r


def _pairs(r):
    out = []
    for _, src, dst in r._pending_mamba_state_copies:
        out += [(int(s), int(d)) for s, d in zip(src, dst) if (int(s), int(d)) != (0, 0)]
    return out


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_log_group_copies_checkpoint_zero_and_migrates_offset(offset):
    # GDN (gid 1, log) and PLE (gid 2, select) cross in the same step.
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = offsets[15] = offset
    r = _crossing_runner(
        offsets, {1: [5, 6, 7, 8, 9], 2: [15, 16, 17, 18, 19]}, log_gids={1}
    )
    TPUModelRunner._collect_mamba_state_seed_copies(
        r, SimpleNamespace(num_scheduled_tokens={"a": 1}), 0, 1
    )
    assert _pairs(r) == [(5, 6), (15 + offset, 16)]
    assert int(offsets[6]) == offset  # GDN: the offset moves with the log
    assert int(offsets[16]) == 0  # PLE: the selected checkpoint is new block 0
    assert int(offsets[5]) == offset  # sources untouched


@pytest.mark.parametrize("offset", [1, 2, 3])
def test_log_group_new_owner_crossing_zeroes_offset(offset):
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = offsets[15] = offset
    r = _crossing_runner(
        offsets,
        {1: [5, 6, 7, 8, 9], 2: [15, 16, 17, 18, 19]},
        log_gids={1},
        seeded=False,
    )
    r._mamba_state_pos.clear()
    TPUModelRunner._collect_mamba_state_seed_copies(
        r, SimpleNamespace(num_scheduled_tokens={"a": 1}), 0, 1
    )
    assert _pairs(r) == [(5, 6), (15, 16)]
    assert int(offsets[6]) == 0 and int(offsets[16]) == 0
    assert int(offsets[5]) == offset


def test_log_group_off_keeps_select_crossing():
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = 2
    r = _crossing_runner(offsets, {1: [5, 6, 7, 8, 9]}, log_gids=set())
    TPUModelRunner._collect_mamba_state_seed_copies(
        r, SimpleNamespace(num_scheduled_tokens={"a": 1}), 0, 1
    )
    assert _pairs(r) == [(7, 6)]
    assert int(offsets[6]) == 0


def test_cow_copy_keeps_log_rows_and_offset(monkeypatch):
    """Copy-on-write through the runner's path and the Pallas block copy:
    every row of the block moves, the log included, with its offset."""
    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", pallas_block_copy)
    pool = torch.zeros((NUM_BLOCKS, ROWS, LANES), dtype=torch.float32)
    pool[9] = torch.randn(ROWS, LANES)
    offsets = torch.zeros(NUM_BLOCKS, dtype=torch.int32)
    offsets[9] = 3
    r = SimpleNamespace(
        mamba_slot_read_offsets=offsets,
        kv_cache_raw_tensors=[pool],
        _pool_block_split=1,
        bm_pool_layout=None,
        device=torch.device("cpu"),
    )
    r._pad_to_bucket = TPUModelRunner._pad_to_bucket
    r._bucket_len = TPUModelRunner._bucket_len
    r.is_unified_pool_used = TPUModelRunner.is_unified_pool_used.__get__(r)
    r._copy_mamba_state_blocks = TPUModelRunner._copy_mamba_state_blocks.__get__(r)
    TPUModelRunner._apply_kv_cache_block_copies(
        r, [SimpleNamespace(src_block_id=9, dst_block_id=40)]
    )
    assert torch.equal(pool[40], pool[9])
    assert int(offsets[40]) == 3


def _async_previous(r, accepted: int) -> None:
    """Async scheduling: the previous step verified 3 drafts and accepted
    `accepted` tokens; the host positions assume all 4."""
    sampled = torch.tensor([[7] * accepted + [INVALID_TOKEN_ID] * (K + 1 - accepted)])
    r.scheduler_config = SimpleNamespace(async_scheduling=True)
    r.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    r._pre_async_results = SimpleNamespace(
        req_id_to_index_copy={"a": 0},
        num_draft_per_req={0: K},
        num_chunks=1,
        wait_for_copy=lambda: sampled,
    )


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_log_group_async_correction_decides_the_crossing(accepted):
    """The host assumes the previous verify accepted all drafts (window ends
    at token 9, column 1); the correction moves it back by the rejections.
    Only an actual crossing copies: GDN copies checkpoint 0 with its log and
    migrates the offset (accepted - 1), PLE selects checkpoint[offset]."""
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = offsets[15] = accepted - 1
    r = _crossing_runner(
        offsets,
        {1: [5, 6, 7, 8, 9], 2: [15, 16, 17, 18, 19]},
        log_gids={1},
        computed=6,
    )
    _async_previous(r, accepted)
    TPUModelRunner._collect_mamba_state_seed_copies(
        r, SimpleNamespace(num_scheduled_tokens={"a": K + 1}), 0, 1
    )
    if accepted <= 2:  # corrected window ends at token 6 or 7: column 0
        assert _pairs(r) == [] and r._mamba_state_pos["a"] == 0
        assert int(offsets[6]) == 0 and int(offsets[16]) == 0
        return
    assert r._mamba_state_pos["a"] == 1
    assert _pairs(r) == [(5, 6), (15 + accepted - 1, 16)]
    assert int(offsets[6]) == accepted - 1
    assert int(offsets[16]) == 0


def test_log_group_async_rejection_returns_to_the_previous_column():
    """The previous step crossed into column 1 and its drafts were rejected,
    so the state column goes back to 0: GDN copies column 1's checkpoint 0
    (state after window position 0, its log) into column 0's block and the
    offset moves with it over the stale one; PLE selects and resets."""
    offsets = torch.zeros(32, dtype=torch.int32)
    offsets[5] = 2  # stale, left by column 0's last verify
    offsets[15] = 2
    pool_rows = {1: [5, 6, 7, 8, 9], 2: [15, 16, 17, 18, 19]}
    r = _crossing_runner(offsets, pool_rows, log_gids={1}, computed=10)
    r._mamba_state_pos["a"] = 1
    _async_previous(r, accepted=1)
    TPUModelRunner._collect_mamba_state_seed_copies(
        r, SimpleNamespace(num_scheduled_tokens={"a": 1}), 0, 1
    )
    assert r._mamba_state_pos["a"] == 0
    assert _pairs(r) == [(6, 5), (16, 15)]
    assert int(offsets[5]) == 0 and int(offsets[15]) == 0


def test_window_log_groups_are_the_gdn_mamba_groups():
    """The runner's log groups are exactly the GDN mamba groups (the groups
    whose pooled kernel runs log mode), not attention or the PLE short conv;
    a kernel in log mode behind a select-mode crossing resumes from a wrong
    state (test_window_log_kernel_with_select_runner_fails)."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    attn = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
    )

    def mamba(kind):
        return MambaSpec(
            block_size=16,
            shapes=((2, 8), (2, 4, 8)),
            dtypes=(torch.float32, torch.float32),
            mamba_type=kind,
            num_speculative_blocks=K,
        )

    gdn = mamba(MambaAttentionBackendEnum.GDN_ATTN)
    ple = mamba(MambaAttentionBackendEnum.SHORT_CONV)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["l0.attn"], kv_cache_spec=attn),
            KVCacheGroupSpec(layer_names=["l1.gdn"], kv_cache_spec=gdn),
            KVCacheGroupSpec(layer_names=["l2.gdn"], kv_cache_spec=gdn),
            KVCacheGroupSpec(layer_names=["l1.ple"], kv_cache_spec=ple),
        ]
    )
    assert runner_mod.gdn_window_log_group_ids(config, K + 1) == {1, 2}
    # No speculative checkpoint window: no log groups.
    assert runner_mod.gdn_window_log_group_ids(config, 1) == frozenset()
