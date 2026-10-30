# SPDX-License-Identifier: Apache-2.0

import inspect
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import synthetic_decode_connector as module
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
    DecodeBenchConnectorScheduler,
)

C = module.SyntheticTPUDecodeConnector


class TestSynthetic(unittest.TestCase):
    def test_logging_enabled(self):
        self.assertTrue(module.logger.isEnabledFor(20))

    def test_concrete(self):
        self.assertFalse(inspect.isabstract(C))

    def test_requires_dummy(self):
        cfg = NS(
            load_config=NS(load_format="auto"),
            kv_transfer_config=NS(
                kv_connector_extra_config={"synthetic_decode_benchmark": True}
            ),
        )
        with self.assertRaises(ValueError):
            C(cfg, None, None)

    def test_requires_optin(self):
        cfg = NS(
            load_config=NS(load_format="dummy"),
            kv_transfer_config=NS(kv_connector_extra_config={}),
        )
        with self.assertRaises(ValueError):
            C(cfg, None, None)

    def test_long_prompt_and_all_groups(self):
        s = DecodeBenchConnectorScheduler(NS(cache_config=NS(block_size=2176)))
        req = NS(request_id="r", num_tokens=131072)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (131071, False))
        self.assertEqual(s.get_num_new_matched_tokens(req, 130560), (511, False))
        groups = tuple(list(range(70)) for _ in range(4))
        s.update_state_after_alloc(req, NS(get_block_ids=lambda: groups), 131071)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (0, False))
        meta = s.build_connector_meta(None)
        self.assertEqual([len(g) for g in meta.reqs_to_fill["r"][0]], [61] * 4)
        self.assertFalse(s.build_connector_meta(None).reqs_to_fill)
        s.request_finished(req)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (131071, False))

    def test_pool_reset_once(self):
        class Tensor:
            zeros = 0

            def zero_(self):
                self.zeros += 1

        c = C.__new__(C)
        c._runner = NS(kv_cache_raw_tensors=[Tensor(), Tensor()])
        c._pool_initialized = False
        c._connector_metadata = DecodeBenchConnectorMetadata(
            reqs_to_fill={"a": (([],), 131071)}
        )
        calls = []
        with patch.dict(
            sys.modules,
            {
                "vllm_torchtpu.utils": NS(
                    synchronize_tensors=lambda t, wait: calls.append(wait)
                )
            },
        ):
            c.start_load_kv(None)
            c._connector_metadata = DecodeBenchConnectorMetadata(
                reqs_to_fill={"b": (([],), 131071)}
            )
            c.start_load_kv(None)
        self.assertEqual([t.zeros for t in c._runner.kv_cache_raw_tensors], [1, 1])
        self.assertEqual(calls, [True, True])


if __name__ == "__main__":
    unittest.main()
