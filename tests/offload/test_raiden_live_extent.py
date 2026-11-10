# SPDX-License-Identifier: Apache-2.0
"""Live-extent host offload (TPU_RAIDEN_LIVE_EXTENT).

A key stores only the kernel-block sub-hashes that hold live bytes:
- a GDN state key, its state region (9 of 17 under PP4 x PCP2, where each rank
  holds a head-sharded half state at the start of a 2,176-row slot);
- a full-attention partial-tail key, the rows below its end position.

Geometry used here is #1's PCP prefill stage: 17 kernel blocks of 128 rows per
device block, one full-attention group (0) and three GDN groups (1, 2, 3),
4,352-token logical blocks over PCP 2 with interleave 128.
"""

import collections
import os
import unittest
from unittest.mock import patch

from test_raiden_store_offload import (
    FAKE_STORE_TYPES,
    FakeBlockStatus,
    FakeKVCacheStore,
    _Entry,
    make_manager,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

from vllm_torchtpu.offload.raiden_store import (
    LiveExtent,
    RaidenOffloadingManager,
    live_extent_requested,
    resolve_live_extent,
)

F = 17
KB = 128
BLK = 4352
FA = 0
STATE_GROUPS = (1, 2, 3)


def key(n: int, group: int) -> bytes:
    return make_offload_key(b"blockhash_%06d" % n, group)


def subs(k: bytes, n: int) -> list[bytes]:
    return [k + i.to_bytes(4, "big") for i in range(n)]


def ctx_with(positions: dict[bytes, int], req_id: str = "r") -> ReqContext:
    c = ReqContext(req_id=req_id)
    for k, pos in positions.items():
        c.set_offload_key_position(k, pos)
    return c


def pcp_rule(classes=frozenset({"state", "tail"}), **kw) -> LiveExtent | None:
    args = dict(
        factor=F,
        kernel_block_tokens=KB,
        state_group_ids=STATE_GROUPS,
        state_required_tokens=1088,
        fa_group_ids=(FA,),
        tail_block_tokens=BLK,
        pcp_size=2,
        dcp_size=1,
        interleave=128,
        layer_major=True,
    )
    args.update(kw)
    return resolve_live_extent(frozenset(classes), **args)


def manager(store, live, num_blocks=64):
    return make_manager(
        store, device_block_size_factor=F, num_blocks=num_blocks, live_extent=live
    )


def seed_host(store, k: bytes, n: int) -> None:
    for h in subs(k, n):
        store.entries[h] = _Entry(status=FakeBlockStatus.HOST)


class TestFlag(unittest.TestCase):
    def test_default_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TPU_RAIDEN_LIVE_EXTENT", None)
            self.assertEqual(live_extent_requested(), frozenset())
        for off in ("", "0", "off", "false"):
            with patch.dict(os.environ, {"TPU_RAIDEN_LIVE_EXTENT": off}):
                self.assertEqual(live_extent_requested(), frozenset(), off)

    def test_values(self):
        cases = {
            "state": {"state"},
            "tail": {"tail"},
            "state,tail": {"state", "tail"},
            " tail , state ": {"state", "tail"},
            "1": {"state", "tail"},
            "all": {"state", "tail"},
        }
        for raw, want in cases.items():
            with patch.dict(os.environ, {"TPU_RAIDEN_LIVE_EXTENT": raw}):
                self.assertEqual(live_extent_requested(), frozenset(want), raw)
        with patch.dict(os.environ, {"TPU_RAIDEN_LIVE_EXTENT": "state,bogus"}):
            with self.assertRaises(ValueError):
                live_extent_requested()

    def test_no_classes_resolves_to_none(self):
        self.assertIsNone(pcp_rule(classes=frozenset()))


class TestResolve(unittest.TestCase):
    def test_pcp_geometry(self):
        rule = pcp_rule()
        self.assertIsNotNone(rule)
        self.assertTrue(rule.state_on)
        self.assertEqual(rule.state_count(), 9)  # ceil(1088 / 128)
        self.assertTrue(rule.tail_on)
        self.assertEqual(rule.tail_block_tokens, BLK)

    def test_pp8_state_fills_the_slot(self):
        # PP8: 2,176 required rows = all 17 kernel blocks; nothing to save.
        self.assertIsNone(
            pcp_rule(
                classes={"state"},
                state_required_tokens=2176,
                tail_block_tokens=2176,
                pcp_size=1,
                interleave=1,
            )
        )
        rule = pcp_rule(
            state_required_tokens=2176, tail_block_tokens=2176, pcp_size=1, interleave=1
        )
        self.assertFalse(rule.state_on)
        self.assertTrue(rule.tail_on)

    def test_block_major_disables(self):
        self.assertIsNone(pcp_rule(layer_major=False))

    def test_unknown_state_extent_disables_state_only(self):
        rule = pcp_rule(state_required_tokens=None)
        self.assertFalse(rule.state_on)
        self.assertTrue(rule.tail_on)

    def test_dcp_disables_tail(self):
        rule = pcp_rule(dcp_size=2, pcp_size=1)
        self.assertTrue(rule.state_on)
        self.assertFalse(rule.tail_on)

    def test_misaligned_interleave_disables_tail(self):
        rule = pcp_rule(interleave=3)
        self.assertFalse(rule.tail_on)

    def test_mutant_is_one_block_short(self):
        self.assertEqual(pcp_rule(mutant_short_state=True).state_count(), 8)


class TestTailCount(unittest.TestCase):
    def test_pcp2_interleave128(self):
        rule = pcp_rule()
        base = 22 * BLK
        cases = {
            base + 128: 1,  # 1 chunk on rank 0
            base + 256: 1,  # chunks 0 (rank 0) and 1 (rank 1)
            base + 384: 2,  # rank 0 holds chunks 0 and 2
            base + 2176: 9,  # 17 chunks -> 9 on rank 0
            base + 4224: 17,  # 33 chunks -> 17 on rank 0
            base + BLK: F,  # block aligned: a complete chunk
            base: F,
        }
        for pos, want in cases.items():
            self.assertEqual(rule.tail_count(pos), want, pos - base)

    def test_pcp1(self):
        rule = pcp_rule(
            state_required_tokens=2176, tail_block_tokens=2176, pcp_size=1, interleave=1
        )
        base = 40 * 2176
        cases = {base + 128: 1, base + 129: 2, base + 2048: 16, base + 2175: 17}
        for pos, want in cases.items():
            self.assertEqual(rule.tail_count(pos), want, pos - base)

    def test_counts_cover_every_rank_row(self):
        """Brute force: rows below the boundary fit in the counted blocks on every rank."""
        rule = pcp_rule()
        for live in range(128, BLK, 128):
            n = rule.tail_count(22 * BLK + live)
            for rank in range(2):
                rows = [
                    (o // 256) * 128 + o % 128
                    for o in range(live)
                    if (o // 128) % 2 == rank
                ]
                if rows:
                    self.assertLess(max(rows) // KB, n, (live, rank))


class TestDefaultPathUnchanged(unittest.TestCase):
    def test_manager_without_rule_keeps_the_factor(self):
        store = FakeKVCacheStore()
        m = manager(store, None)
        sk, fk = key(1, 1), key(2, FA)
        m.submit_store_job(1, [sk, fk], [5, 6])
        (call,) = store.calls_named("insert")
        self.assertEqual(call.args[0], subs(sk, F) + subs(fk, F))
        self.assertEqual(
            [s.device_block_id for s in call.args[1]],
            list(range(5 * F, 6 * F)) + list(range(6 * F, 7 * F)),
        )

    def test_env_alone_does_not_change_a_manager(self):
        with patch.dict(os.environ, {"TPU_RAIDEN_LIVE_EXTENT": "state,tail"}):
            store = FakeKVCacheStore()
            m = manager(store, None)
            m.submit_store_job(1, [key(1, 1)], [5])
            (call,) = store.calls_named("insert")
            self.assertEqual(len(call.args[0]), F)


class TestStateKeys(unittest.TestCase):
    def setUp(self):
        self.store = FakeKVCacheStore()
        self.m = manager(self.store, pcp_rule())

    def test_store_job_stores_the_state_region_only(self):
        sk, fk = key(1, 2), key(2, FA)
        self.m.submit_store_job(1, [sk, fk], [5, 6])
        (call,) = self.store.calls_named("insert")
        self.assertEqual(call.args[0], subs(sk, 9) + subs(fk, F))
        self.assertEqual(
            [s.device_block_id for s in call.args[1]],
            list(range(5 * F, 5 * F + 9)) + list(range(6 * F, 7 * F)),
        )
        (save,) = self.store.calls_named("save")
        self.assertEqual(save.args[0], call.args[0])
        tel = self.m.telemetry
        self.assertEqual((tel["live_state_keys"], tel["live_state_sub"]), (1, 9))
        self.assertEqual((tel["live_other_keys"], tel["live_other_sub"]), (1, F))
        self.assertEqual(tel["live_saved_sub"], F - 9)

    def test_load_job_loads_the_state_region_only(self):
        sk = key(3, 1)
        seed_host(self.store, sk, 9)
        self.assertIs(self.m.lookup(sk, ReqContext(req_id="r")), LookupResult.HIT)
        self.m.prepare_load([sk], ReqContext(req_id="r"))
        self.m.submit_load_job(7, [sk], [11], "r")
        (load,) = self.store.calls_named("load")
        self.assertEqual(load.args[0], subs(sk, 9))
        self.assertEqual(load.args[1], list(range(11 * F, 11 * F + 9)))

    def test_lookup_needs_every_live_sub_hash(self):
        sk = key(4, 3)
        seed_host(self.store, sk, 8)
        self.assertIs(self.m.lookup(sk, ReqContext(req_id="r")), LookupResult.MISS)
        seed_host(self.store, sk, 9)
        self.assertIs(self.m.lookup(sk, ReqContext(req_id="r")), LookupResult.HIT)

    def test_resident_state_key_is_not_restored(self):
        sk = key(5, 1)
        seed_host(self.store, sk, 9)
        out = self.m.prepare_store([sk], ReqContext(req_id="r"))
        self.assertEqual(out.keys_to_store, [])


class TestTailKeys(unittest.TestCase):
    def setUp(self):
        self.store = FakeKVCacheStore()
        self.m = manager(self.store, pcp_rule())
        self.tk = key(10, FA)
        self.pos = 22 * BLK + 384  # 3 chunks -> 2 blocks on rank 0

    def test_store_count_comes_from_the_prepare_context(self):
        out = self.m.prepare_store([self.tk], ctx_with({self.tk: self.pos}))
        self.assertEqual(out.keys_to_store, [self.tk])
        # submit_store_job has no request context.
        self.m.submit_store_job(1, [self.tk], [7])
        (call,) = self.store.calls_named("insert")
        self.assertEqual(call.args[0], subs(self.tk, 2))
        self.assertEqual([s.device_block_id for s in call.args[1]], [7 * F, 7 * F + 1])
        self.assertEqual(self.m.telemetry["live_tail_sub"], 2)

    def test_unknown_position_keeps_the_factor(self):
        out = self.m.prepare_store([self.tk], ReqContext(req_id="r"))
        self.m.submit_store_job(1, out.keys_to_store, [7])
        (call,) = self.store.calls_named("insert")
        self.assertEqual(len(call.args[0]), F)

    def test_lookup_and_load_use_the_position(self):
        seed_host(self.store, self.tk, 2)
        c = ctx_with({self.tk: self.pos})
        self.assertIs(self.m.lookup(self.tk, c), LookupResult.HIT)
        self.m.prepare_load([self.tk], c)
        self.m.submit_load_job(3, [self.tk], [9], "r")
        (load,) = self.store.calls_named("load")
        self.assertEqual(load.args[0], subs(self.tk, 2))
        self.assertEqual(load.args[1], [9 * F, 9 * F + 1])

    def test_complete_chunk_key_keeps_the_factor(self):
        ck = key(11, FA)
        self.m.prepare_store([ck], ctx_with({ck: 23 * BLK}))
        self.m.submit_store_job(1, [ck], [4])
        (call,) = self.store.calls_named("insert")
        self.assertEqual(len(call.args[0]), F)

    def test_count_map_is_bounded(self):
        self.m._tail_counts_cap = 8
        keys = [key(100 + i, FA) for i in range(20)]
        self.m.touch(keys, ctx_with({k: 22 * BLK + 128 for k in keys}))
        self.assertEqual(len(self.m._tail_counts), 8)
        self.assertEqual(list(self.m._tail_counts), keys[-8:])


class TestMultiKeyJobs(unittest.TestCase):
    """Per-key counts must keep every sub-hash paired with its own device block."""

    def test_load_job_pairs_each_key_with_its_blocks(self):
        store = FakeKVCacheStore()
        m = manager(store, pcp_rule())
        sk, fk, tk = key(50, 2), key(51, FA), key(52, FA)
        seed_host(store, sk, 9)
        seed_host(store, fk, F)
        seed_host(store, tk, 2)
        c = ctx_with({fk: 23 * BLK, tk: 22 * BLK + 384})
        m.prepare_load([sk, fk, tk], c)
        m.submit_load_job(4, [sk, fk, tk], [3, 4, 5], "r")
        (load,) = store.calls_named("load")
        self.assertEqual(load.args[0], subs(sk, 9) + subs(fk, F) + subs(tk, 2))
        self.assertEqual(
            load.args[1],
            list(range(3 * F, 3 * F + 9)) + list(range(4 * F, 5 * F)) + [5 * F, 5 * F + 1],
        )

    def test_store_job_pairs_each_key_with_its_blocks(self):
        store = FakeKVCacheStore()
        m = manager(store, pcp_rule())
        tk, sk = key(53, FA), key(54, 3)
        m.prepare_store([tk, sk], ctx_with({tk: 22 * BLK + 896}))  # 7 chunks -> 4 blocks
        m.submit_store_job(1, [tk, sk], [8, 2])
        (call,) = store.calls_named("insert")
        self.assertEqual(call.args[0], subs(tk, 4) + subs(sk, 9))
        self.assertEqual(
            [s.device_block_id for s in call.args[1]],
            list(range(8 * F, 8 * F + 4)) + list(range(2 * F, 2 * F + 9)),
        )


class TestMixedFlagSafety(unittest.TestCase):
    def test_full_copy_serves_a_live_extent_lookup(self):
        store = FakeKVCacheStore()
        sk = key(20, 1)
        seed_host(store, sk, F)  # stored by an engine with the flag off
        m = manager(store, pcp_rule())
        self.assertIs(m.lookup(sk, ReqContext(req_id="r")), LookupResult.HIT)
        m.prepare_load([sk], ReqContext(req_id="r"))
        m.submit_load_job(1, [sk], [2], "r")
        (load,) = store.calls_named("load")
        self.assertEqual(load.args[0], subs(sk, 9))

    def test_live_copy_is_a_miss_without_the_flag(self):
        store = FakeKVCacheStore()
        sk = key(21, 1)
        seed_host(store, sk, 9)  # stored with the flag on
        m = manager(store, None)
        self.assertIs(m.lookup(sk, ReqContext(req_id="r")), LookupResult.MISS)


class TestEvictionBookkeeping(unittest.TestCase):
    def test_state_key_retires_after_its_live_sub_hashes(self):
        store = FakeKVCacheStore()
        m = manager(store, pcp_rule())
        sk = key(30, 2)
        store.simulate_evict(subs(sk, 9))
        (event,) = m.take_events()
        self.assertEqual(event.keys, [sk])
        self.assertNotIn(sk, m._evicted_sub_hashes)

    def test_default_needs_the_factor(self):
        store = FakeKVCacheStore()
        m = manager(store, None)
        sk = key(31, 2)
        store.simulate_evict(subs(sk, 9))
        self.assertIn(sk, m._evicted_sub_hashes)
        store.simulate_evict(subs(sk, F)[9:])
        self.assertNotIn(sk, m._evicted_sub_hashes)


class _LruStore(FakeKVCacheStore):
    """FakeKVCacheStore with a capacity in entries and LRU eviction of unpinned entries."""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.order: collections.OrderedDict[bytes, None] = collections.OrderedDict()

    def insert(self, block_hashes, slices, on_host) -> bool:
        ok = super().insert(block_hashes, slices, on_host)
        for h in block_hashes:
            self.order[h] = None
            self.order.move_to_end(h)
        evicted = []
        while len(self.entries) > self.capacity:
            victim = next(
                (h for h in self.order if self.entries[h].pin_count == 0), None
            )
            if victim is None:
                break
            del self.entries[victim]
            del self.order[victim]
            evicted.append(victim)
        if evicted:
            self.simulate_evict(evicted)
        return ok


class TestCapacity(unittest.TestCase):
    def _resident_after_storing(self, live, n_keys=3, capacity=2 * F):
        store = _LruStore(capacity)
        m = make_manager(
            store,
            device_block_size_factor=F,
            num_blocks=capacity // F,
            live_extent=live,
        )
        keys = [key(40 + i, 1 + i % 3) for i in range(n_keys)]
        for i, k in enumerate(keys):
            m.submit_store_job(i, [k], [i])
            store.complete_save([h for h in list(store._saving)])
            m.poll_finished_jobs()
        return sum(
            m.lookup(k, ReqContext(req_id="probe")) is LookupResult.HIT for k in keys
        )

    def test_live_extent_fits_more_state_keys(self):
        # 34 entries: 2 full state keys, or 3 live-extent ones (27 entries).
        self.assertEqual(self._resident_after_storing(None), 2)
        self.assertEqual(self._resident_after_storing(pcp_rule()), 3)


class _Qwen35GdnModel:
    """Qwen3.5-397B GDN state shapes per TP shard, with the MTP-widened conv
    (kernel 4 -> 3 taps + 1 speculative token), as the TPU layer declares it."""

    @staticmethod
    def get_mamba_state_shape_from_config(vllm_config):
        hf = vllm_config.model_config.hf_text_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        kq, v = hf.linear_num_key_heads // tp, hf.linear_num_value_heads // tp
        conv_dim = 2 * kq * hf.linear_key_head_dim + v * hf.linear_value_head_dim
        return (
            (hf.linear_conv_kernel_dim - 1 + 1, conv_dim),
            (v, hf.linear_value_head_dim, hf.linear_key_head_dim),
        )

    @staticmethod
    def get_mamba_state_dtype_from_config(_):
        import torch

        return (torch.bfloat16, torch.float32)


def _qwen35_vllm_config(pcp: int):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vllm.config import CacheConfig, ModelConfig, VllmConfig

    cfg = MagicMock(spec=VllmConfig)
    cfg.model_config = MagicMock(spec=ModelConfig)
    cfg.model_config.is_hybrid = True
    cfg.model_config.architecture = "Qwen3_5MoeForCausalLM"
    cfg.model_config.hf_text_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    cfg.cache_config = MagicMock(spec=CacheConfig)
    cfg.parallel_config = MagicMock()
    cfg.parallel_config.tensor_parallel_size = 1
    cfg.parallel_config.prefill_context_parallel_size = pcp
    cfg.parallel_config.cp_kv_cache_interleave_size = 128 if pcp > 1 else 1
    return cfg


class TestStateExtentDerivation(unittest.TestCase):
    """Ties the derivation to the measured block-size log lines (M):
    PP8 mamba_raw_state_bytes=2195456 / required 2228224 (2,176 rows);
    PP4xPCP2 1097728 / 1114112 (1,088 rows, mamba_fit_block_size=1088)."""

    def _required(self, pcp):
        from vllm_torchtpu.offload.raiden_store import _gdn_state_required_tokens
        from vllm_torchtpu.platforms.tpu_block_size_utils import (
            _hybrid_mamba_state_layout,
        )

        cfg = _qwen35_vllm_config(pcp)
        with patch(
            "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
            return_value=(_Qwen35GdnModel, None),
        ):
            layout = _hybrid_mamba_state_layout(cfg, 1024)
            rows = _gdn_state_required_tokens(
                cfg, bytes_per_pool_row=1024, slot_bytes=2176 * 1024
            )
        return layout, rows

    def test_pp8(self):
        layout, rows = self._required(1)
        self.assertEqual(layout.conv_bytes + layout.ssm_bytes, 2195456)
        self.assertEqual(layout.required_bytes, 2228224)
        self.assertEqual(rows, 2176)

    def test_pcp2(self):
        layout, rows = self._required(2)
        self.assertEqual(layout.conv_bytes + layout.ssm_bytes, 1097728)
        self.assertEqual(layout.required_bytes, 1114112)
        self.assertEqual(rows, 1088)
        self.assertEqual(layout.ssm_tokens, 1024)  # SSM = kernel blocks 0-7


class TestSpecResolver(unittest.TestCase):
    """TPURaidenStoreOffloadingSpec._resolve_live_extent on #1's PCP stage layout."""

    def _resolve(self, env, pcp=2, block_major=None):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        from vllm_torchtpu.offload.raiden_store import TPURaidenStoreOffloadingSpec

        groups = [SimpleNamespace(kv_cache_spec=MagicMock(spec=FullAttentionSpec))]
        groups += [
            SimpleNamespace(kv_cache_spec=MagicMock(spec=MambaSpec)) for _ in range(3)
        ]
        fake_self = SimpleNamespace(
            vllm_config=_qwen35_vllm_config(pcp),
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            block_major_contract=block_major,
            device_block_size_factor=F,
        )
        config = SimpleNamespace(
            groups=[SimpleNamespace(group_id=i) for i in range(4)],
            parallel=SimpleNamespace(pcp_size=pcp, dcp_size=1),
        )
        with patch.dict(os.environ, env), patch(
            "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
            return_value=(_Qwen35GdnModel, None),
        ):
            return TPURaidenStoreOffloadingSpec._resolve_live_extent(
                fake_self,
                config,
                kernel_block_size=KB,
                per_block_shape=(4, 64, 4, 128),
                kv_dtype=torch.float8_e4m3fn,
                device_block_size=2176,
            )

    def test_off_by_default(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("TPU_RAIDEN_LIVE_EXTENT", None)
            self.assertIsNone(self._resolve({}))

    def test_pcp_stage(self):
        rule = self._resolve({"TPU_RAIDEN_LIVE_EXTENT": "state,tail"})
        self.assertEqual(rule.state_group_ids, frozenset({1, 2, 3}))
        self.assertEqual(rule.state_count(), 9)
        self.assertEqual(rule.tail_group_ids, frozenset({0}))
        self.assertEqual(rule.tail_block_tokens, BLK)
        self.assertEqual((rule.pcp_size, rule.interleave), (2, 128))

    def test_pp8_stage_has_no_state_saving(self):
        rule = self._resolve({"TPU_RAIDEN_LIVE_EXTENT": "state"}, pcp=1)
        self.assertIsNone(rule)

    def test_block_major_is_refused(self):
        rule = self._resolve(
            {"TPU_RAIDEN_LIVE_EXTENT": "state,tail"}, block_major=object()
        )
        self.assertIsNone(rule)

    def test_mutant_env(self):
        rule = self._resolve(
            {
                "TPU_RAIDEN_LIVE_EXTENT": "state",
                "TPU_RAIDEN_LIVE_EXTENT_MUTANT": "short_state",
            }
        )
        self.assertEqual(rule.state_count(), 8)


class TestManagerSignature(unittest.TestCase):
    def test_factor_mismatch_is_refused(self):
        with self.assertRaises(AssertionError):
            RaidenOffloadingManager(
                kernel_physical_blocks_capacity=8 * 4,
                offload_logical_blocks_capacity=8,
                device_block_size_factor=4,
                world_size=2,
                raiden_controller_port=0,
                raiden_job_name="t",
                live_extent=pcp_rule(),
                store=FakeKVCacheStore(),
                store_types=FAKE_STORE_TYPES,
            )


if __name__ == "__main__":
    unittest.main()
