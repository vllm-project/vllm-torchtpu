# SPDX-License-Identifier: Apache-2.0
"""q38 victim identity run, addendum 1 (j-266f4814): the harness's pool and staging constants follow the planner.

j-266f4814 expected 128 usable IDs and got 247: with QWEN4_EXP_STATIC_POOL_FILL=1 the static planner takes
max(plan, cap pages), and at --max-model-len 131072 / --max-num-seqs 1 the plan is 1 + one request's blocks over
every group, far above the 129-page cap. Its 16 staging leases x hint 86-87 also exceeded the 248-block pool, so
tpu_sync kept the full mirror (a storage is bounded only when its slot count is below num_blocks,
lane_xfer.md:865). These tests pin the planner rule and tie the harness constants to the measured pool.
"""

import importlib.util
import os

from tests.distributed.test_qwen4_exp_pd_draft_kv import real_cache_config
from tests.runner.test_qwen4_exp_host_tier import pooled, served_prefill  # noqa: F401  (pooled: autouse env fixture)
from vllm_torchtpu.models.vllm.qwen4_exp import config as qconfig

CAP = 845_414_400  # the launcher's PREFILL_QWEN4_EXP_MAX_POOL_BYTES (129 blocks x 2 pages)


def _harness():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "q38_hosttier", "q38_hosttier_victim_identity.py")
    spec = importlib.util.spec_from_file_location("q38_victim_identity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _identity_cfg():
    cfg = real_cache_config(served_prefill())
    cfg.model_config.max_model_len = 131072
    cfg.scheduler_config.max_num_seqs = 1
    return cfg


def _plan_and_bpb(monkeypatch, cfg):
    """The minimal static plan (fill off, uncapped) and the busiest stage's bytes per block, from the real planner."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    monkeypatch.delenv("QWEN4_EXP_STATIC_POOL_FILL", raising=False)
    monkeypatch.setenv("QWEN4_EXP_MAX_POOL_BYTES", str(1 << 50))
    plan = qconfig.native_cache_capacity(cfg)
    qconfig.ensure_kv_group_size_rule()
    specs = qconfig.native_cache_specs(cfg)
    return plan, qconfig._stage_bytes_per_block(cfg, get_kv_cache_groups(cfg, specs), specs)


def test_a_cap_below_the_plan_does_not_shrink_a_filled_pool(monkeypatch):
    """Revision 2's error: it expected fill 1 + a cap below the plan to give the cap. The planner takes
    max(plan, cap pages) with fill on and min(plan, cap pages) with fill off (config.py:395-405)."""
    cfg = _identity_cfg()
    plan, bpb = _plan_and_bpb(monkeypatch, cfg)
    small = plan // 2
    monkeypatch.setenv("QWEN4_EXP_MAX_POOL_BYTES", str(small * bpb))
    monkeypatch.setenv("QWEN4_EXP_STATIC_POOL_FILL", "1")
    assert qconfig.native_cache_capacity(cfg) == plan
    monkeypatch.setenv("QWEN4_EXP_STATIC_POOL_FILL", "0")
    assert qconfig.native_cache_capacity(cfg) == small


def test_harness_constants_follow_the_served_pool():
    """The served PP8 geometry is not reproducible on CPU (the test model plans another group structure), so the
    pool comes from j-266f4814's own log: "Qwen4Exp unified pool: ... blocks=248" with fill 1 and the 129-block cap
    (M). The harness must expect it and keep bounded staging feasible on it (lane_xfer.md:865)."""
    harness = _harness()
    assert harness.POOL_BLOCKS == 248
    assert harness.POOL_IDS[0] <= harness.POOL_BLOCKS - 1 <= harness.POOL_IDS[1]
    assert harness.LEASES * harness.HINT_MAX < harness.POOL_BLOCKS
    assert harness.POLICY_ENV["QWEN4_EXP_PD_STAGING_LEASES"] == str(harness.LEASES)
