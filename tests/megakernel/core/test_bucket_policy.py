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
"""The bucket policy (`runtime.policy`) and the eligibility rules
(`runtime.eligibility`): bucket-list parsing and selection, `BucketPolicy`,
and the startup / post-mesh / step checks. No model folder, cache planner or
profile registry."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from tests.megakernel.core import constants as test_constants
from tests.megakernel.core.constants import IMPORT_PROBE, REPO_SRC, STARTUP_OK
from vllm_torchtpu.megakernel.core import runtime as mk_runtime
from vllm_torchtpu.megakernel.core.profile import Profile
from vllm_torchtpu.megakernel.core.runtime import constants as runtime_constants
from vllm_torchtpu.megakernel.core.runtime.contract import (
    MESH_SIZE_MISMATCH,
    UNSUPPORTED_TP_SIZE,
    UNSUPPORTED_TPU_CHIP,
    EligibilityResult,
)
from vllm_torchtpu.megakernel.core.runtime.eligibility import (
    check_post_mesh_eligibility,
    check_startup_eligibility,
    check_step_eligibility,
)
from vllm_torchtpu.megakernel.core.runtime.policy import (
    DEFAULT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
    MTP_SPECULATIVE_TOKENS,
    MTP_VERIFY_TOKENS,
    BucketPolicy,
    parse_mk_buckets,
    parse_mk_verify_buckets,
    select_bucket,
)
from vllm_torchtpu.megakernel.core.runtime.stats import BatchShape, inspect_batch_shape


@pytest.mark.parametrize(
    "module, blocked",
    [
        # constants is framework-free: no jax either.
        ("constants", ("jax", "jaxlib", "torch", "vllm")),
        # policy is framework-free: no jax either.
        ("policy", ("jax", "jaxlib", "torch", "vllm")),
        # eligibility imports no torch or vLLM (jax through the chip table).
        ("eligibility", ("torch", "vllm")),
    ],
)
def test_import_without_frameworks(module: str, blocked: tuple[str, ...]) -> None:
    """The module imports with `blocked` unimportable (an import of any of them
    would fail the subprocess)."""
    out = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE.format(blocked=blocked, module=module)],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": str(REPO_SRC),
            "PATH": "/usr/bin:/bin",
            "JAX_PLATFORMS": "cpu",
        },
        timeout=300,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr[-3000:]


def test_bucket_list_parsing_and_selection() -> None:
    assert parse_mk_buckets("8, 32, 16") == (8, 16, 32)
    assert parse_mk_buckets("8:16") == (8, 16)
    assert parse_mk_buckets("none") == ()
    assert parse_mk_buckets(x for x in (16, 8)) == (8, 16)
    assert parse_mk_buckets({32, 8, 16}) == (8, 16, 32)
    assert parse_mk_verify_buckets("16, 32") == (16, 32)
    assert parse_mk_verify_buckets(x for x in (32, 16)) == (16, 32)
    assert parse_mk_verify_buckets({32, 16}) == (16, 32)
    for bad in ("65", "0, 8", "8, 8", "8, abc", b"8", (8, 8), [True], [8.5]):
        with pytest.raises(ValueError, match="decode buckets"):
            parse_mk_buckets(bad)
    for bad in ("6", "16, 16", "16, abc", (16, 16), [True], [16.5]):
        with pytest.raises(ValueError, match="verify_buckets"):
            parse_mk_verify_buckets(bad)
    assert select_bucket(5, (8, 16, 32)) == 8
    assert select_bucket(32, (8, 16, 32)) == 32
    with pytest.raises(ValueError, match="exceeds max bucket"):
        select_bucket(33, (8, 16, 32))


def test_bucket_policy_resolve_select_and_out_rows() -> None:
    assert MTP_SPECULATIVE_TOKENS == MTP_VERIFY_TOKENS - 1
    prof = Profile(name="t-buckets", numerics="exact")
    pol = BucketPolicy.resolve(prof)
    assert (pol.buckets, pol.verify_buckets) == (
        DEFAULT_BUCKETS,
        DEFAULT_VERIFY_BUCKETS,
    )
    # The profile's lists, unless explicit ones are given.
    prof = Profile(name="t-buckets", numerics="exact", buckets=(16, 8))
    assert BucketPolicy.resolve(prof).buckets == (8, 16)
    assert BucketPolicy.resolve(prof, buckets=(32,)).buckets == (32,)
    # With a draft of MTP_SPECULATIVE_TOKENS, implicit decode buckets drop;
    # without one, implicit verify buckets drop.
    mtp = BucketPolicy.resolve(prof, speculative_tokens=MTP_SPECULATIVE_TOKENS)
    assert mtp.buckets == () and mtp.verify_buckets == DEFAULT_VERIFY_BUCKETS
    pol = BucketPolicy.resolve(
        prof,
        buckets=(8, 16),
        verify_buckets=(16, 32),
        speculative_tokens=0,
        buckets_explicit=True,
        verify_buckets_explicit=False,
    )
    assert pol.buckets == (8, 16)
    assert pol.verify_buckets == ()
    assert list(pol.iter_buckets()) == [(1, 8), (1, 16)]
    assert (pol.max_bucket, pol.max_verify_bucket) == (16, 0)
    assert pol.select(5, 1) == 8

    # Both direct construction and resolve() validate buckets (even when
    # speculative_tokens drops an implicit list).
    with pytest.raises(ValueError, match="decode buckets"):
        BucketPolicy((8, 8), ())
    with pytest.raises(ValueError, match="verify_buckets"):
        BucketPolicy((8,), (6,))
    bad_prof = Profile(name="t-bad-verify", numerics="exact", verify_buckets=(6,))
    with pytest.raises(ValueError, match="verify_buckets"):
        BucketPolicy.resolve(bad_prof, speculative_tokens=0)

    pol_pad = BucketPolicy((8, 16), (16, 32), token_paddings=(64, 16, 32))
    assert pol_pad.token_paddings == (16, 32, 64)
    assert pol_pad.out_rows(8, 16) == 16
    assert pol_pad.out_rows(32, 32) == 32
    assert BucketPolicy((8,), ()).out_rows(8) == 16
    assert pol_pad.select(16, 4) == 16
    assert pol_pad.select(64, 4, clamp=True) == 32
    assert list(pol_pad.iter_buckets()) == [(1, 8), (1, 16), (4, 16), (4, 32)]
    for bad_pad in ((0, 16), (True,), "16"):
        with pytest.raises(ValueError, match="token_paddings"):
            BucketPolicy((8,), (), token_paddings=bad_pad)


@pytest.mark.parametrize(
    ("overrides", "expected_reason", "expected_level", "expected_substr"),
    [
        (
            {"num_layers": 30, "expected_num_layers": 60},
            "layer_count_mismatch",
            "debug",
            "",
        ),
        (
            {"unified_cache_layout": False},
            "unaligned_cache_layout",
            "warning",
            "unified KV and recurrent cache layout",
        ),
        (
            {"multimodal_inputs": True},
            "multimodal_inputs",
            "warning",
            "multimodal inputs unsupported",
        ),
        (
            {"lora_enabled": True},
            "lora_enabled",
            "warning",
            "does not support LoRA",
        ),
        (
            {"pipeline_parallel": True},
            "pipeline_parallel",
            "warning",
            "single-stage pipeline",
        ),
        (
            {"context_parallel": True},
            "context_parallel",
            "warning",
            "context parallelism",
        ),
        (
            {"dp_lockstep": True},
            "dp_lockstep",
            "warning",
            "data-parallel lockstep",
        ),
        (
            {"aux_hidden_states": True},
            "aux_hidden_states",
            "warning",
            "auxiliary hidden-state",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": (8,)},
            "unsupported_tp_size",
            "warning",
            "8 TPU devices in mesh, got 4",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": (2, 8)},
            "unsupported_tp_size",
            "warning",
            "TPU mesh size in (2, 8), got 4",
        ),
        (
            {"tp_size": 4, "supported_tp_sizes": ()},
            "unsupported_tp_size",
            "warning",
            "does not declare any supported TPU mesh sizes",
        ),
    ],
)
def test_startup_eligibility_table_driven(
    overrides: dict,
    expected_reason: str,
    expected_level: str,
    expected_substr: str,
):
    """Table-driven test covering `check_startup_eligibility` and post-mesh checks."""
    base_kwargs = dict(
        num_layers=60,
        expected_num_layers=60,
        unified_cache_layout=True,
        multimodal_inputs=False,
        lora_enabled=False,
        pipeline_parallel=False,
        context_parallel=False,
        dp_lockstep=False,
        aux_hidden_states=False,
        tp_size=8,
        supported_tp_sizes=(8,),
    )
    res_ok = check_startup_eligibility(**base_kwargs)
    assert res_ok.eligible and res_ok.reason == "ok" and res_ok.code == "ok"

    kw = {**base_kwargs, **overrides}
    res = check_startup_eligibility(**kw)
    assert not res.eligible
    assert res.reason == expected_reason
    assert res.code == expected_reason
    assert res.level == expected_level
    assert expected_substr in res.message


def test_post_mesh_eligibility() -> None:
    post_ok = check_post_mesh_eligibility(
        tp_size=8,
        mesh_size=8,
        device_kind="TPU7x",
        bucket_policy=BucketPolicy((8,), ()),
    )
    assert post_ok.eligible and post_ok.reason == "ok"
    res_mesh = check_post_mesh_eligibility(tp_size=4, mesh_size=2)
    assert not res_mesh.eligible and res_mesh.reason == MESH_SIZE_MISMATCH
    assert "mesh has 2 devices, expected 4" in res_mesh.message
    res_chip = check_post_mesh_eligibility(tp_size=8, mesh_size=8, device_kind="TPU v5")
    assert not res_chip.eligible and res_chip.reason == UNSUPPORTED_TPU_CHIP
    res_bkts = check_post_mesh_eligibility(
        tp_size=8, mesh_size=8, bucket_policy=BucketPolicy((), ())
    )
    assert not res_bkts.eligible and res_bkts.reason == "empty_buckets"
    assert res_bkts.level == "info"


def test_step_eligibility_cases():
    """Table-driven test for `check_step_eligibility`."""
    bs_runner = BatchShape(True, 2, 4, "ok")
    pol = BucketPolicy((8, 16), (16, 32), token_paddings=(16, 32))
    steps = {(1, 8): object(), (1, 16): object(), (4, 16): object(), (4, 32): object()}

    cases = [
        (
            dict(
                batch_shape=bs_runner,
                input_rows=16,
                mrope_ok=False,
            ),
            False,
            "stock_mrope",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(False, 2, 0, "stock_mixed"),
                input_rows=16,
            ),
            False,
            "stock_mixed",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 2, 2, "ok"),
                input_rows=16,
            ),
            False,
            "stock_q2",
            0,
        ),
        (
            # An unserved step size is reported before the M-RoPE check.
            dict(
                batch_shape=BatchShape(True, 2, 8, "ok"),
                input_rows=16,
                mrope_ok=False,
            ),
            False,
            "stock_q8",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 20, 1, "ok"),
                input_rows=32,
            ),
            False,
            "stock_t1_n20",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 10, 4, "ok"),
                input_rows=64,
            ),
            False,
            "stock_t4_n10",
            0,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 4, 1, "ok"),
                input_rows=64,
            ),
            False,
            "stock_t1_rows8",
            8,
        ),
        (
            dict(
                batch_shape=BatchShape(True, 4, 1, "ok"),
                input_rows=16,
            ),
            True,
            "mk_t1_b8",
            8,
        ),
        (
            dict(
                batch_shape=bs_runner,
                input_rows=16,
            ),
            True,
            "mk_t4_b16",
            16,
        ),
    ]
    for kw, exp_el, exp_reason, exp_bkt in cases:
        bs_case = kw["batch_shape"]
        dec = check_step_eligibility(
            bs_case,
            pol,
            steps,
            input_rows=kw["input_rows"],
            mrope_ok=kw.get("mrope_ok", True),
        )
        assert dec.eligible is exp_el, kw
        assert dec.reason == exp_reason, kw
        assert dec.bucket_t == exp_bkt, kw

    # Missing compiled bucket key returns stock_t1_nobkt (bare int key 8 never matched)
    dec_nobkt = check_step_eligibility(
        BatchShape(True, 4, 1, "ok"),
        pol,
        {8: object()},
        input_rows=16,
    )
    assert not dec_nobkt.eligible and dec_nobkt.reason == "stock_t1_nobkt"


def test_startup_disabled_and_hf_config_decline_come_first() -> None:
    res = check_startup_eligibility(enabled=False, num_layers=1, expected_num_layers=2)
    assert not res.eligible and (res.code, res.level) == ("disabled", "debug")
    hf_bad = EligibilityResult(False, "non_moe_checkpoint", "dense")
    res = check_startup_eligibility(
        hf_config_result=hf_bad, num_layers=1, expected_num_layers=2
    )
    assert res is hf_bad
    hf_ok = EligibilityResult(True, "ok", "")
    res = check_startup_eligibility(hf_config_result=hf_ok, **STARTUP_OK)
    assert res.eligible


def test_startup_without_tp_size_skips_the_tp_and_mesh_checks() -> None:
    """`tp_size=None` checks neither the TP size nor the mesh size (the
    runtime does), but still the chip and the buckets; a `mesh_size` without
    `tp_size` is an error, since nothing would check it."""
    assert check_startup_eligibility(supported_tp_sizes=(), **STARTUP_OK).eligible
    res = check_startup_eligibility(device_kind="TPU v5", **STARTUP_OK)
    assert not res.eligible and res.code == UNSUPPORTED_TPU_CHIP
    res = check_startup_eligibility(bucket_policy=BucketPolicy((), ()), **STARTUP_OK)
    assert not res.eligible and res.code == "empty_buckets"
    with pytest.raises(ValueError, match="mesh_size=8 needs tp_size"):
        check_startup_eligibility(mesh_size=8, **STARTUP_OK)
    # With tp_size, the mesh size is checked.
    res = check_startup_eligibility(
        tp_size=8, supported_tp_sizes=(8,), mesh_size=4, **STARTUP_OK
    )
    assert not res.eligible and res.code == MESH_SIZE_MISMATCH
    res = check_startup_eligibility(tp_size=4, supported_tp_sizes=(8,), **STARTUP_OK)
    assert res.code == UNSUPPORTED_TP_SIZE


def test_verify_tokens_eight_bucket_policy_and_eligibility() -> None:
    """Verify generalized verify_tokens in BucketPolicy and eligibility."""
    prof_8 = Profile(
        name="test_8", numerics="exact", buckets=(2,), verify_buckets=(8, 16)
    )
    pol_8 = BucketPolicy.resolve(prof_8, verify_tokens=(8,))
    assert pol_8.verify_tokens == (8,)
    pol_spec7 = BucketPolicy.resolve(prof_8, speculative_tokens=7, verify_tokens=(8,))
    assert (pol_spec7.buckets, pol_spec7.verify_buckets) == ((), (8, 16))
    pol_spec7_implicit = BucketPolicy.resolve(prof_8, speculative_tokens=7)
    assert pol_spec7_implicit.verify_tokens == (8,)
    assert (pol_spec7_implicit.buckets, pol_spec7_implicit.verify_buckets) == (
        (),
        (8, 16),
    )

    pol_4_8 = BucketPolicy.resolve(
        Profile(
            name="test_4_8",
            numerics="exact",
            buckets=(2,),
            verify_buckets=(4, 8, 16),
        ),
        verify_tokens=(4, 8),
    )
    assert pol_4_8.verify_tokens == (4, 8)
    assert list(pol_4_8.iter_buckets()) == [
        (1, 2),
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
    ]
    pol_def_8 = BucketPolicy.resolve(
        Profile(name="test_def", numerics="exact"), verify_tokens=(8,)
    )
    assert pol_def_8.verify_buckets == (8, 16, 24, 32, 40, 48, 56, 64)
    expected_bucket_16 = 16
    assert pol_def_8.select(16, 8) == expected_bucket_16

    b_shape_8 = BatchShape(eligible=True, num_reqs=1, tokens_per_req=8, reason="ok")
    dec_8 = check_step_eligibility(
        b_shape_8,
        pol_8,
        compiled_buckets={(8, 8)},
        input_rows=8,
    )
    assert (dec_8.eligible, dec_8.tokens_per_seq, dec_8.bucket_t) == (True, 8, 8)

    pol_def = BucketPolicy()
    dec_def = check_step_eligibility(
        b_shape_8,
        pol_def,
        compiled_buckets={(1, 8)},
        input_rows=8,
    )
    assert (dec_def.eligible, dec_def.reason) == (False, "stock_q8")

    pol_unaligned_max = BucketPolicy(verify_buckets=(4, 12), verify_tokens=(4, 8))
    dec_unaligned = check_step_eligibility(
        b_shape_8,
        pol_unaligned_max,
        compiled_buckets={(4, 12)},
        input_rows=8,
    )
    assert (dec_unaligned.eligible, dec_unaligned.reason) == (False, "stock_t8_n1")


def test_resolve_explicit_verify_buckets_validation() -> None:
    """Verify explicit verify_buckets raise ValueError when unaligned."""
    prof = Profile(name="test_exp", numerics="exact", buckets=(2,))
    with pytest.raises(ValueError, match=r"verify_buckets \[4\] not divisible"):
        BucketPolicy.resolve(prof, verify_buckets=(4, 8), verify_tokens=(8,))
    with pytest.raises(ValueError, match="invalid verify_tokens"):
        BucketPolicy.resolve(prof, verify_tokens=(0,))
    pol_non_exp = BucketPolicy.resolve(
        prof,
        verify_buckets=(4, 8),
        verify_buckets_explicit=False,
        verify_tokens=(8,),
    )
    assert pol_non_exp.verify_buckets == (8,)


def test_verify_tokens_eight_from_query_start_loc() -> None:
    """Verify batch inspection and step eligibility for 8-token verify steps."""
    batch_shape = inspect_batch_shape(
        num_reqs=4,
        has_slot_read_offsets=True,
        query_start_loc=[0, 8, 16, 24, 32],
    )
    pol = BucketPolicy(verify_tokens=(4, 8))
    dec = check_step_eligibility(
        batch_shape,
        pol,
        compiled_buckets={(8, 32)},
        input_rows=32,
    )
    assert (dec.eligible, dec.tokens_per_seq, dec.bucket_t) == (True, 8, 32)


def test_verify_tokens_heterogeneous_resolve_keeps_per_width_buckets() -> None:
    """Mixed verify_tokens=(4, 8) retains per-width buckets (e.g. b=4)."""
    prof = Profile(name="test_hetero", numerics="exact", buckets=(2,))
    pol = BucketPolicy.resolve(prof, verify_buckets=(4, 8, 16), verify_tokens=(4, 8))
    expected_bucket_4 = 4
    assert pol.verify_buckets == (4, 8, 16)
    assert pol.verify_buckets_for(4) == (4, 8, 16)
    assert pol.verify_buckets_for(8) == (8, 16)
    assert pol.select(4, 4) == expected_bucket_4


def test_verify_tokens_rejects_one() -> None:
    """`verify_tokens` requires widths >= 2 since T=1 routes to decode."""
    prof = Profile(name="test_one", numerics="exact")
    with pytest.raises(ValueError, match="invalid verify_tokens"):
        BucketPolicy((8,), (4,), verify_tokens=(1,))
    with pytest.raises(ValueError, match="invalid verify_tokens"):
        BucketPolicy.resolve(prof, verify_tokens=(1,))


def test_runtime_constants_module_and_iter_buckets_not_generator() -> None:
    """Validate un-prefixed constants.py exports and non-generator iter_buckets."""
    assert hasattr(runtime_constants, "DEFAULT_BUCKETS")
    assert hasattr(runtime_constants, "DEFAULT_VERIFY_BUCKETS")
    assert hasattr(runtime_constants, "MTP_VERIFY_TOKENS")
    assert hasattr(runtime_constants, "MTP_SPECULATIVE_TOKENS")
    assert hasattr(runtime_constants, "MAX_BUCKET_ROWS")
    assert hasattr(runtime_constants, "MIN_VERIFY_TOKENS")
    assert hasattr(runtime_constants, "FALSE_STRS")
    assert hasattr(runtime_constants, "STARTUP_FLAG_RULES")

    assert hasattr(test_constants, "REPO_SRC")
    assert hasattr(test_constants, "IMPORT_PROBE")
    assert hasattr(test_constants, "STARTUP_OK")

    assert not inspect.isgeneratorfunction(BucketPolicy.iter_buckets)

    pol = BucketPolicy(buckets=(2, 4), verify_buckets=(4, 8), verify_tokens=(4,))
    bucket_iter = pol.iter_buckets()
    assert list(bucket_iter) == [(1, 2), (1, 4), (4, 4), (4, 8)]


def test_parse_mk_verify_buckets_custom_widths() -> None:
    """Validate parameterized verify widths and runtime lazy constant exports."""
    assert parse_mk_verify_buckets("6, 12", verify_tokens=(6,)) == (6, 12)
    assert parse_mk_verify_buckets("8, 16", verify_tokens=(8,)) == (8, 16)
    with pytest.raises(ValueError, match="invalid verify_buckets"):
        parse_mk_verify_buckets("4, 8", verify_tokens=(8,))

    prof = Profile(name="test_widths", numerics="exact", buckets=(2,))
    for bad_buckets in ((4, 8), (6, 8)):
        with pytest.raises(ValueError, match="not divisible by verify_tokens"):
            BucketPolicy.resolve(
                prof,
                verify_buckets=bad_buckets,
                verify_tokens=(8,),
            )

    assert mk_runtime.MIN_VERIFY_TOKENS == runtime_constants.MIN_VERIFY_TOKENS
    assert mk_runtime.MAX_BUCKET_ROWS == runtime_constants.MAX_BUCKET_ROWS
    assert mk_runtime.constants is runtime_constants
