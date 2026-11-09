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
"""TPU_MLA_FUSED_SMALL_A2A: the decode MLA kernel runs the DCP small
all-to-all itself.

CPU: the bucket gate (`fused_small_a2a_applies`) and the kernel's static
geometry.

The bit-identity claim -- the fused kernel's receive buffer equals
`small_all_to_all(send, row0=SEND_SLAB_ROW0)` of the unfused kernel's send
buffer in every row it writes, so the unpack and merge see the same bits --
needs 8 TensorCores with remote DMA and a barrier. It was checked on 2x2x2
(16 ranks = two DCP8 groups) at the served per-rank shapes (T = 4 / 8 and the
T = 16 guard, edge cells with padding tokens and whole ranks at lse -inf),
with NaN-poisoned receive buffers and positive controls (misrouted peer
block, dropped peer block, misrouted destination slab, dropped lse low
halves) that must fail, by the write-tpu-kernels harness of close-gaps trial 6
(kernel_work/t6-mla-fuse, probe 02-m1).
"""

import os
from unittest import mock

import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

GATE_ENV = {
    "TPU_MLA_FUSED_SMALL_A2A": "1",
    "TPU_DCP_MLA_SEND_SMALL_A2A": "1",
    "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS": "8",
    "TPU_DCP_INDEXER_SMALL_AG": "1",
    "TPU_SC_TOPK_DCP_MERGE_FLAT": "1",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_MLA_DECODE_BODY_LEVERS": "1",
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_FUSED_PREP": "1",
    "TPU_MLA_DECODE_ROW_WIDTHS": "128,256,384",
    "TPU_DCP_MLA_SMALL_CHUNK_TOKENS": "32",
}


def _applies(tokens, **over):
    env = {k: v for k, v in dict(GATE_ENV, **over).items() if v is not None}
    with mock.patch.dict(os.environ, env):
        for k, v in over.items():
            if v is None:
                os.environ.pop(k, None)
        return cp_mla.fused_small_a2a_applies(tokens)


def test_gate_buckets():
    # T <= cap (8) with a one-step grid; never the T = 16 bucket.
    assert [t for t in (1, 2, 4, 8, 16, 32) if _applies(t)] == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "key,off",
    [
        # Explicit off values: each of these is on by default on this branch.
        ("TPU_MLA_FUSED_SMALL_A2A", "0"),
        ("TPU_DCP_MLA_SEND_SMALL_A2A", "0"),
        ("TPU_DCP_RANK_OPERAND", "0"),
        ("TPU_DCP_MLA_FUSED_PREP", "0"),
        ("TPU_MLA_DECODE_ROW_WIDTHS", ""),
        ("TPU_DCP_MLA_TOPK_CAP", "0"),
    ],
)
def test_gate_off_without(key, off):
    assert not _applies(4, **{key: off})


def test_gate_needs_step8():
    # Without "step8" the decode grid walks the small-chunk batch: not one step.
    assert not _applies(8, TPU_MLA_DECODE_BODY_LEVERS="tr,zu,pv512,quad")
    assert _applies(8, TPU_MLA_DECODE_BODY_LEVERS="tr,step8")


def test_gate_cap():
    assert not _applies(8, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="4")
    assert _applies(4, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="4")
    # A cap above the step8 step would leave a two-step grid: off everywhere.
    assert not _applies(4, TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS="16")
    assert not _applies(4, TPU_DCP_MLA_SMALL_CHUNK_TOKENS="4")


def test_gate_compares_tokens_once():
    # The bucket size is symbolic under torch.compile: one `T <= cap`
    # comparison and nothing else (no modulo, no lower bound), or the
    # backbone trace specializes on the first bucket.
    seen = []

    class T(int):
        def __le__(self, other):
            seen.append(("<=", other))
            return int(self) <= other

        def __gt__(self, other):
            seen.append((">", other))
            return int(self) > other

        def __mod__(self, other):
            raise AssertionError("modulo on the token count")

        def __lt__(self, other):
            raise AssertionError("< on the token count")

        def __ge__(self, other):
            raise AssertionError(">= on the token count")

    assert _applies(T(4))
    assert seen == [("<=", 8)], seen


def test_slab_geometry():
    # The slab starts on a 16-row int16 tile and its last head row is the lse
    # word row, which lands on row SEND_PAD_ROWS - 1 of the receiving block.
    assert sparse_mla.SEND_SLAB_ROW0 % 8 == 0
    assert sparse_mla.SEND_SLAB_ROW0 + (sparse_mla.SEND_SLAB_ROW0 - 1) == (
        sparse_mla.SEND_PAD_ROWS - 1
    )
    assert sparse_mla.SEND_EXCHANGE_COLLECTIVE_ID == 13


def test_bucket_config():
    from vllm_torchtpu.layers.core import attention_interface as ai

    def cfg(tokens, **env):
        with mock.patch.dict(os.environ, env):
            for k in ("TPU_MLA_FUSED_SMALL_A2A_BUCKETS",):
                if k not in env:
                    os.environ.pop(k, None)
            return ai.fused_small_a2a_bucket_config(tokens)

    # The default bucket spec (trial 6): T = 4 pins only the gathered KV at
    # 16 MiB; T = 8 is not listed and takes the knob defaults.
    assert cfg(4) == (("kv",), 16) and cfg(8) == (True, 32)
    # Knob defaults (M1e, no spec): T < 8 unpinned at 16 MiB, T >= 8 everything
    # pinned at 32.
    no_spec = {"TPU_MLA_FUSED_SMALL_A2A_BUCKETS": ""}
    assert cfg(4, **no_spec) == (False, 16) and cfg(8, **no_spec) == (True, 32)
    spec = "4=kv@16,8=kv+tab@32"
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS=spec) == (("kv",), 16)
    assert cfg(8, TPU_MLA_FUSED_SMALL_A2A_BUCKETS=spec) == (("kv", "tab"), 32)
    # A bucket the spec does not list keeps the defaults.
    assert cfg(8, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=all@24") == (True, 32)
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=all@24") == (True, 24)
    assert cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=none@20") == (False, 20)
    with pytest.raises(ValueError):
        cfg(4, TPU_MLA_FUSED_SMALL_A2A_BUCKETS="4=weights@16")


def test_pin_roles_cover_every_operand_class():
    assert set(sparse_mla.SEND_EXCHANGE_PIN_ROLES) == {"kv", "q", "q_pe", "tab", "recv"}


def test_min_tokens_default_fuses_every_gated_bucket():
    from vllm_torchtpu import envs

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS == 1
    with mock.patch.dict(os.environ, {"TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS": "8"}):
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS == 8


def test_merge_mode_constants():
    from vllm_torchtpu import envs

    assert "send_merge_lo_skip" in sparse_mla.SEND_EXCHANGE_FAULTS
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MERGE", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE is True  # default on (trial 6)
    with mock.patch.dict(os.environ, {"TPU_MLA_FUSED_SMALL_A2A_MERGE": "0"}):
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE is False
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS", None)
        assert envs.TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS == 4


_CPU8_WORKER_ENV = "_MLA_FUSED_SMALL_A2A_CPU8_WORKER"
_N_DCP = 8
_CPU8_ENV = {
    **GATE_ENV,
    "TPU_MLA_SPARSE_SPLIT_Q": "1",
    "TPU_DCP_MLA_MASK_IN_KERNEL": "1",
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": "256",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
    "TPU_MLA_DECODE_PROLOGUE": "1",
    "TPU_DCP_MLA_SEND_LAYOUT": "1",
    "TPU_MLA_FUSED_SMALL_A2A_BUCKETS": "4=kv@16",
    "TPU_MLA_FUSED_SMALL_A2A_MERGE": "1",
    "TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS": "4",
    "TPU_MLA_KV_WRITE_IN_GATHER": "0",
    "TPU_MLA_TC_KV_WRITER": "1",
}


@pytest.mark.parametrize("off_val", [None, "", "0", "false", "off", "FALSE", "OFF"])
def test_resolve_config_off_equals_defaults(off_val):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.core import attention_interface as ai

    clean_keys = (
        "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS",
        "TPU_MLA_FUSED_SMALL_A2A_BUCKETS",
        "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS",
        "TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS",
        "TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS",
        "TPU_MLA_DECODE_BODY_LEVERS",
    )
    with mock.patch.dict(os.environ, dict(GATE_ENV), clear=False):
        for k in clean_keys:
            os.environ.pop(k, None)
        if off_val is not None:
            os.environ["TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS"] = off_val
        actual_raw = envs.TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS
        assert actual_raw == (off_val or "")
        cfg = ai.resolve_fused_small_a2a_config(
            num_q_heads=64, dcp_size=8, fused_bucket=True
        )
        assert cfg == ai.FusedSmallA2AConfig(
            bucket_spec="4=kv@16",
            send_max_tokens=8,
            merge_max_tokens=4,
            mtp_merge_max_tokens=8,
            decode_levers=sparse_mla.DECODE_BODY_DEFAULT,
        )
        assert "step16" not in cfg.decode_levers
        assert [
            t for t in (1, 2, 4, 8, 16, 32) if cp_mla.fused_small_a2a_applies(t)
        ] == [
            1,
            2,
            4,
            8,
        ]
        assert ai.fused_small_a2a_bucket_config(4) == (("kv",), 16)
        assert ai.fused_small_a2a_bucket_config(8) == (True, 32)


def test_4chip_buckets_on_gate_and_bucket_config():
    from vllm_torchtpu.layers.core import attention_interface as ai

    for spec_val in ("1", "4=kv@16,8=kv@24,16=kv@32"):
        with mock.patch.dict(
            os.environ,
            {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": spec_val},
        ):
            cfg = ai.resolve_fused_small_a2a_config(
                num_q_heads=64, dcp_size=8, fused_bucket=True
            )
            assert cfg == ai.FusedSmallA2AConfig(
                bucket_spec="4=kv@16,8=kv@24,16=kv@32",
                send_max_tokens=16,
                merge_max_tokens=16,
                mtp_merge_max_tokens=16,
                decode_levers=(*sparse_mla.DECODE_BODY_DEFAULT, "step16"),
            )
            # Unfused buckets (e.g. T=32) keep DECODE_BODY_DEFAULT without step16.
            unfused_cfg = ai.resolve_fused_small_a2a_config(
                num_q_heads=64, dcp_size=8, fused_bucket=False
            )
            assert unfused_cfg.decode_levers == sparse_mla.DECODE_BODY_DEFAULT
            assert [
                t for t in (1, 2, 4, 8, 16, 32) if cp_mla.fused_small_a2a_applies(t)
            ] == [1, 2, 4, 8, 16]
            assert ai.fused_small_a2a_bucket_config(4) == (("kv",), 16)
            assert ai.fused_small_a2a_bucket_config(8) == (("kv",), 24)
            assert ai.fused_small_a2a_bucket_config(16) == (("kv",), 32)

    # Symbolic token count still performs a single `T <= 16` check.
    seen = []

    class SymbolicT(int):
        def __le__(self, other):
            seen.append(("<=", other))
            return int(self) <= other

        def __gt__(self, other):
            seen.append((">", other))
            return int(self) > other

        def __mod__(self, other):
            raise AssertionError("modulo on symbolic token count")

        def __lt__(self, other):
            raise AssertionError("< on symbolic token count")

        def __ge__(self, other):
            raise AssertionError(">= on symbolic token count")

    assert _applies(SymbolicT(16), TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS="1")
    assert seen == [("<=", 16)], seen

    # Custom spec override and T<=8-only spec (keeps step8 without step16).
    with mock.patch.dict(
        os.environ,
        {
            **GATE_ENV,
            "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "4=kv@16,8=kv+tab@20,16=all@28",
        },
    ):
        assert ai.fused_small_a2a_bucket_config(8) == (("kv", "tab"), 20)
        assert ai.fused_small_a2a_bucket_config(16) == (True, 28)

    with mock.patch.dict(
        os.environ,
        {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "4=kv@16,8=kv@24"},
    ):
        cfg8 = ai.resolve_fused_small_a2a_config(
            num_q_heads=64, dcp_size=8, fused_bucket=True
        )
        assert cfg8.send_max_tokens == 8
        assert cfg8.merge_max_tokens == 8
        assert cfg8.mtp_merge_max_tokens == 8
        assert cfg8.decode_levers == sparse_mla.DECODE_BODY_DEFAULT


def test_4chip_buckets_rejects_tl_above_128_and_bad_specs():
    import inspect

    from vllm_torchtpu.layers.core import attention_interface as ai

    assert ai.IN_KERNEL_MERGE_MAX_SEND_ROWS == 128
    assert sparse_mla.DECODE_STEP16_TOKENS == 16

    # No model-specific magic defaults (e.g. 64 or 8) in helper/gate signatures.
    sig_resolve = inspect.signature(ai.resolve_fused_small_a2a_config)
    assert sig_resolve.parameters["num_q_heads"].default is None
    assert sig_resolve.parameters["dcp_size"].default is None
    sig_bucket = inspect.signature(ai.fused_small_a2a_bucket_config)
    assert sig_bucket.parameters["num_q_heads"].default is None
    assert sig_bucket.parameters["dcp_size"].default is None
    assert (
        inspect.signature(cp_mla.send_small_a2a_applies)
        .parameters["num_q_heads"]
        .default
        is None
    )
    assert (
        inspect.signature(cp_mla.fused_small_a2a_applies)
        .parameters["num_q_heads"]
        .default
        is None
    )

    with mock.patch.dict(
        os.environ, {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "1"}
    ):
        # Cached keyed on env + shape args: repeated calls return same instance.
        c1 = ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=8)
        c2 = ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=8)
        assert c1 is c2

        # When num_q_heads or dcp_size is unknown (None), skip tl check cleanly.
        assert ai.resolve_fused_small_a2a_config().send_max_tokens == 16
        assert ai.resolve_fused_small_a2a_config(dcp_size=4).send_max_tokens == 16
        assert ai.resolve_fused_small_a2a_config(num_q_heads=128).send_max_tokens == 16

        # Gates use real DCP group world_size when present (and skip tl when None).
        fake_grp8 = mock.Mock(world_size=8)
        fake_grp4 = mock.Mock(world_size=4)
        with mock.patch.object(cp_mla, "get_dcp_group", return_value=fake_grp8):
            assert cp_mla.fused_small_a2a_applies(16)
            assert cp_mla.fused_small_a2a_applies(16, num_q_heads=64)
        with mock.patch.object(cp_mla, "get_dcp_group", return_value=fake_grp4):
            assert cp_mla.fused_small_a2a_applies(16)  # num_q_heads unknown -> skip tl
            with pytest.raises(ValueError, match="tl=256"):
                cp_mla.fused_small_a2a_applies(16, num_q_heads=64)
            with pytest.raises(ValueError, match="tl=256"):
                cp_mla.send_small_a2a_applies(16, num_q_heads=64)

        # H=128, W=8 -> send_per=16 -> T=16 needs tl=256 > 128.
        with pytest.raises(ValueError, match="tl=256"):
            ai.resolve_fused_small_a2a_config(num_q_heads=128, dcp_size=8)
        # H=64, W=4 -> send_per=16 -> T=16 needs tl=256 > 128.
        with pytest.raises(ValueError, match="tl=256"):
            ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=4)
        with pytest.raises(ValueError, match="dcp_size"):
            ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=0)
        with pytest.raises(ValueError, match="num_q_heads"):
            ai.resolve_fused_small_a2a_config(num_q_heads=0, dcp_size=8)
        with pytest.raises(ValueError, match="divisible"):
            ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=6)

    # T=32 at H=64, W=8 -> tl=256 > 128.
    with (
        mock.patch.dict(
            os.environ,
            {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "32=kv@48"},
        ),
        pytest.raises(ValueError, match="tl=256"),
    ):
        ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=8)

    # T=32 at H=32, W=8 -> tl=128 <= 128, but T=32 > DECODE_STEP16_TOKENS (16).
    with (
        mock.patch.dict(
            os.environ,
            {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "32=kv@32"},
        ),
        pytest.raises(ValueError, match="exceeds step16 bound"),
    ):
        ai.resolve_fused_small_a2a_config(num_q_heads=32, dcp_size=8)

    for bad_spec in (
        "bogus",
        ",",
        "0=kv@16",
        "-4=kv@16",
        "16=weights@32",
        "16=kv@0",
        "16=kv@abc",
    ):
        with (
            mock.patch.dict(
                os.environ,
                {**GATE_ENV, "TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": bad_spec},
            ),
            pytest.raises(ValueError),
        ):
            ai.resolve_fused_small_a2a_config(num_q_heads=64, dcp_size=8)


def test_cpu8_step16_chunk_requires_capped_row_widths(_cpu8_worker_run: bool):
    if not _cpu8_worker_run:
        return
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

    ql = jax.ShapeDtypeStruct((16, 64, 512), jnp.bfloat16)
    qp = jax.ShapeDtypeStruct((16, 64, 256), jnp.bfloat16)
    nc = jax.ShapeDtypeStruct((8, 256, 128), jnp.uint32)
    rc = jax.ShapeDtypeStruct((8, 64, 128), jnp.uint32)
    kvl = jax.ShapeDtypeStruct((16,), jnp.int32)
    pidx = jax.ShapeDtypeStruct((16, 2048), jnp.int32)
    tend = jax.ShapeDtypeStruct((), jnp.int32)
    rt = jax.ShapeDtypeStruct((16, 2, 128), jnp.float32)

    @jax.jit
    def bad_fn(ql, qp, nc, rc, kvl, pidx, tend, rt):
        return sparse_mla.sparse_attention_chunk(
            ql,
            nc,
            rc,
            kvl,
            pidx,
            None,
            start_pos=0,
            end_pos=16,
            token_end=tend,
            sm_scale=0.125,
            k_scale=1.0,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            small_chunk_tokens=32,
            small_chunk_batch_size=4,
            topk_cap=512,
            q_pe=qp,
            q_rope_table=rt,
            lse_fill=True,
            lse_lane_dense=True,
            lse_mask_empty=True,
            row_widths=(),
            row_levers=(*sparse_mla.DECODE_BODY_DEFAULT, "step16"),
        )

    with (
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
        mock.patch.object(pltpu, "get_tpu_info", return_value=None),
        mock.patch.object(
            native_sc_gather, "dsa_gather_native_sc", _fake_gather_poisoned
        ),
        pytest.raises(ValueError, match="requires capped row_widths"),
    ):
        bad_fn.lower(ql, qp, nc, rc, kvl, pidx, tend, rt)


@pytest.fixture(scope="module")
def _cpu8_worker_run(request: pytest.FixtureRequest) -> bool:
    """Run 8-CPU-device Pallas interpret tests in an isolated subprocess."""
    import re
    import subprocess
    import sys

    if os.environ.get(_CPU8_WORKER_ENV) == "1":
        return True
    targets = [
        f"{__file__}::{item.nodeid.split('::', 1)[1]}"
        for item in request.session.items
        if item.path == request.path and "_cpu8_worker_run" in item.fixturenames
    ]
    assert targets
    flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+",
        "",
        os.environ.get("XLA_FLAGS", ""),
    ).strip()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(f"{flags} --xla_force_host_platform_device_count={_N_DCP}".strip()),
    )
    env[_CPU8_WORKER_ENV] = "1"
    cache_dir = str(request.config.getini("cache_dir"))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            f"cache_dir={cache_dir}",
            "-q",
            "--tb=short",
            *targets,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"8-CPU-device subprocess failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        )
    return False


def _fake_gather_poisoned(
    nope_cache,
    rope_cache,
    indices,
    *,
    out_size,
    atoms_per_batch=16,
    live_vec_offset=None,
    num_live_indices=None,
    packed_rope=False,
    **_,
):
    """Reference SparseCore gather poisoning rows >= num_live_indices with NaN."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather

    del atoms_per_batch, live_vec_offset
    flat = indices.reshape(-1)[:out_size]
    total = nope_cache.shape[0] * nope_cache.shape[1]
    safe = jnp.clip(flat, 0, total - 1)
    nope = nope_cache.reshape(-1, 128)[safe]
    words = (
        native_sc_gather.PACKED_ROPE_WORDS
        if packed_rope
        else native_sc_gather.ROPE_WORDS
    )
    r_rows = rope_cache.reshape(-1, words)[safe].reshape(
        -1, native_sc_gather.ROPE_PACKING, words
    )
    bands = native_sc_gather.transpose_quad(
        [r_rows[:, t] for t in range(native_sc_gather.ROPE_PACKING)]
    )
    rope = jnp.zeros((r_rows.shape[0], 128), jnp.uint32)
    for b, data in enumerate(bands):
        rope = rope.at[:, b * words : (b + 1) * words].set(data)
    if num_live_indices is not None:
        n_live = jnp.reshape(num_live_indices, ())
        row_idx = jnp.arange(out_size, dtype=jnp.int32)[:, None]
        poison = jnp.uint32(0x7F7F7F7F)
        nope = jnp.where(row_idx < n_live, nope, poison)
        rope_idx = jnp.arange(rope.shape[0], dtype=jnp.int32)[:, None]
        pack = native_sc_gather.ROPE_PACKING
        live_rope_rows = (n_live + pack - 1) // pack
        rope = jnp.where(rope_idx < live_rope_rows, rope, poison)
    return nope, rope


def _bf16_ulp_diff(a_bf16, b_bf16) -> tuple[int, int]:
    import jax.numpy as jnp
    import numpy as np

    a_np = np.asarray(jnp.asarray(a_bf16, jnp.bfloat16).view(jnp.uint16), np.int32)
    b_np = np.asarray(jnp.asarray(b_bf16, jnp.bfloat16).view(jnp.uint16), np.int32)

    def to_ord(u):
        return np.where((u & 0x8000) != 0, 0x8000 - (u & 0x7FFF), u + 0x8000)

    diff = np.abs(to_ord(a_np) - to_ord(b_np))
    return int((a_np != b_np).sum()), int(diff.max()) if diff.size else 0


def _sequential_merge_jax(o_recv, l_recv):
    """Sequential w=0..W-1 two-part (w_hi + w_lo) merge matching xchg_merge_store."""
    import jax.numpy as jnp

    n = o_recv.shape[0]
    finite = [jnp.isfinite(l_recv[w]) for w in range(n)]
    shift = jnp.where(finite[0], l_recv[0], -jnp.inf)
    for w in range(1, n):
        shift = jnp.maximum(shift, jnp.where(finite[w], l_recv[w], -jnp.inf))
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = [jnp.where(finite[w], jnp.exp(l_recv[w] - shift), 0.0) for w in range(n)]
    denom = weights[0]
    for w in range(1, n):
        denom = denom + weights[w]
    denom = jnp.where(denom > 0.0, denom, 1.0)
    acc_hi = jnp.zeros_like(o_recv[0], dtype=jnp.float32)
    acc_lo = jnp.zeros_like(o_recv[0], dtype=jnp.float32)
    for w in range(n):
        wt = (weights[w] / denom)[..., None]
        w_hi = wt.astype(jnp.bfloat16).astype(jnp.float32)
        w_lo = wt - w_hi
        part = o_recv[w].astype(jnp.float32)
        part = jnp.where(jnp.isnan(part), 0.0, part)
        acc_hi = acc_hi + part * w_hi
        acc_lo = acc_lo + part * w_lo
    return (acc_hi + acc_lo).astype(jnp.bfloat16)


def _cpu8_dcp_mesh():
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk

    devs = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devs) >= _N_DCP
    return Mesh(
        np.asarray(devs[:_N_DCP]).reshape(1, _N_DCP),
        ("model", streamindex_topk.DCP_AXIS_NAME),
    )


def test_cpu8_hlo_byte_identical_off_and_step16_subpass(_cpu8_worker_run: bool):
    if not _cpu8_worker_run:
        return
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.kernels.mla import kv_cache_utils
    from vllm_torchtpu.kernels.mla.sparse import native_sc_gather
    from vllm_torchtpu.layers.core import attention_interface as ai

    mesh = _cpu8_dcp_mesh()
    axis = streamindex_topk.DCP_AXIS_NAME
    num_pages, page_size = 8, 256
    nope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages,
        page_size,
        512,
        4,
    )
    rope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages,
        page_size,
        64,
        4,
    )

    with (
        mock.patch.dict(os.environ, dict(_CPU8_ENV), clear=False),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
        mock.patch.object(pltpu, "get_tpu_info", return_value=None),
        mock.patch.object(
            native_sc_gather, "dsa_gather_native_sc", _fake_gather_poisoned
        ),
    ):
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS", None)

        # 1. Condition 5: step16 leaves T=4 and T=8 StableHLO byte-identical.
        def lower_srpa(t_sub: int, levers: tuple[str, ...]) -> str:
            ql = jax.ShapeDtypeStruct((t_sub, 64, 512), jnp.bfloat16)
            qp = jax.ShapeDtypeStruct((t_sub, 64, 256), jnp.bfloat16)
            nc = jax.ShapeDtypeStruct((num_pages, 256, 128), jnp.uint32)
            rc = jax.ShapeDtypeStruct((num_pages, 64, 128), jnp.uint32)
            tk = jax.ShapeDtypeStruct((t_sub, 2048), jnp.int32)
            pt = jax.ShapeDtypeStruct((num_pages,), jnp.int32)
            cq = jax.ShapeDtypeStruct((2,), jnp.int32)
            dist = jax.ShapeDtypeStruct((3,), jnp.int32)
            rt = jax.ShapeDtypeStruct((t_sub, 2, 128), jnp.float32)

            @jax.jit
            def fn(ql, qp, nc, rc, tk, pt, cq, dist, rt):
                return sparse_mla.sparse_ragged_paged_attention(
                    ql,
                    nc,
                    rc,
                    tk,
                    pt,
                    cq,
                    dist,
                    q_pe=qp,
                    q_rope_table=rt,
                    lse_in_kernel=True,
                    sm_scale=0.125,
                    k_scale=1.0,
                    return_lse=True,
                    cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
                    gather_and_attention_chunk_size=256,
                    topk_cap=512,
                    small_chunk_batch_size=4,
                    small_chunk_tokens=32,
                    lse_lane_dense=True,
                    lse_mask_empty=True,
                    fused_window_prep=True,
                    decode_row_widths=(128, 256, 384),
                    decode_body_levers=levers,
                )

            return fn.lower(ql, qp, nc, rc, tk, pt, cq, dist, rt).as_text()

        for t_sub in (4, 8):
            assert lower_srpa(t_sub, sparse_mla.DECODE_BODY_DEFAULT) == lower_srpa(
                t_sub, (*sparse_mla.DECODE_BODY_DEFAULT, "step16")
            )

        # 2. Control buckets T=4 and T=32 lower identically with knob OFF vs ON.
        sh_rep = NamedSharding(mesh, P())
        sh_dcp = NamedSharding(mesh, P(axis))
        sh_dcp2 = NamedSharding(mesh, P(axis, None))
        sh_dcp3 = NamedSharding(mesh, P(axis, None, None))

        def lower_dcp(t_ctrl: int, knob_val: str | None) -> str:
            ql_s = jax.ShapeDtypeStruct(
                (t_ctrl, 64, 512), jnp.bfloat16, sharding=sh_rep
            )
            qp_s = jax.ShapeDtypeStruct(
                (t_ctrl, 64, 256), jnp.bfloat16, sharding=sh_rep
            )
            kvc_s = jax.ShapeDtypeStruct(
                (t_ctrl, 512), jnp.float8_e4m3fn, sharding=sh_rep
            )
            kpe_s = jax.ShapeDtypeStruct(
                (t_ctrl, 64), jnp.float8_e4m3fn, sharding=sh_rep
            )
            nc_s = jax.ShapeDtypeStruct(
                (8 * num_pages, 256, 128), jnp.uint32, sharding=sh_dcp3
            )
            rc_s = jax.ShapeDtypeStruct(
                (8 * num_pages, 64, 128), jnp.uint32, sharding=sh_dcp3
            )
            tk_s = jax.ShapeDtypeStruct((8 * t_ctrl, 2048), jnp.int32, sharding=sh_dcp2)
            sl_s = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=sh_rep)
            bt_s = jax.ShapeDtypeStruct((num_pages,), jnp.int32, sharding=sh_rep)
            qsl_s = jax.ShapeDtypeStruct((2,), jnp.int32, sharding=sh_rep)
            dist_s = jax.ShapeDtypeStruct((3,), jnp.int32, sharding=sh_rep)
            rt_s = jax.ShapeDtypeStruct((t_ctrl, 2, 128), jnp.float32, sharding=sh_rep)
            r_s = jax.ShapeDtypeStruct((8,), jnp.int32, sharding=sh_dcp)
            l_s = jax.ShapeDtypeStruct((8,), jnp.int32, sharding=sh_dcp)
            env_patch = (
                {}
                if knob_val is None
                else {"TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": knob_val}
            )
            with mock.patch.dict(os.environ, env_patch, clear=False):
                if knob_val is None:
                    os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS", None)
                fuse = cp_mla.fused_small_a2a_applies(t_ctrl)
                cfg = ai.resolve_fused_small_a2a_config(64, 8, fused_bucket=fuse)
                fuse_mrg = fuse and t_ctrl <= cfg.merge_max_tokens

                @jax.jit
                def fn(ql, qp, kvc, kpe, nc, rc, tk, sl, bt, qsl, dist, rt, r, ln):
                    return ai.sparse_mla_attention_dcp(
                        ql,
                        qp,
                        kvc,
                        kpe,
                        nc,
                        rc,
                        tk,
                        sl,
                        bt,
                        qsl,
                        dist,
                        mesh,
                        nope_spec,
                        rope_spec,
                        sm_scale=0.125,
                        k_scale=1.0,
                        dcp_size=8,
                        interleave_size=4,
                        q_rope_table=rt,
                        dcp_rank_ids=r,
                        send_layout=t_ctrl <= 16,
                        fused_exchange=fuse,
                        dcp_lane_ids=ln if fuse else None,
                        fused_merge=fuse_mrg,
                    )

                return fn.lower(
                    ql_s,
                    qp_s,
                    kvc_s,
                    kpe_s,
                    nc_s,
                    rc_s,
                    tk_s,
                    sl_s,
                    bt_s,
                    qsl_s,
                    dist_s,
                    rt_s,
                    r_s,
                    l_s,
                ).as_text()

        for t_ctrl in (4, 32):
            assert lower_dcp(t_ctrl, None) == lower_dcp(t_ctrl, "1")


def _run_dcp8_backbone_and_mtp_suite(mesh, *, is_cpu_interpret: bool) -> None:
    import contextlib

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.kernels.mla import kv_cache_utils
    from vllm_torchtpu.kernels.mla.sparse import (
        native_sc_cache_layout,
        native_sc_gather,
    )
    from vllm_torchtpu.layers.core import attention_interface as ai

    axis = streamindex_topk.DCP_AXIS_NAME
    sh_dcp = NamedSharding(mesh, P(axis))
    sh_dcp2 = NamedSharding(mesh, P(axis, None))
    sh_dcp3 = NamedSharding(mesh, P(axis, None, None))
    sh_rep = NamedSharding(mesh, P())

    num_pages, page_size = 8, 256
    cache_tokens = num_pages * page_size
    nope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.NOPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages,
        page_size,
        512,
        4,
    )
    rope_spec = kv_cache_utils.SparseMLAKVCacheSpec.create(
        kv_cache_utils.KVCacheType.ROPE,
        kv_cache_utils.KVCacheLayout.SPARSECORE,
        num_pages,
        page_size,
        64,
        4,
    )

    rng = np.random.default_rng(20261108)
    nope_ranks, rope_ranks = [], []
    for _ in range(8):
        nb = jax.lax.bitcast_convert_type(
            jnp.asarray(
                rng.uniform(-1.0, 1.0, (num_pages, page_size, 512)), jnp.float32
            ).astype(jnp.float8_e4m3fn),
            jnp.uint8,
        )
        rb = jax.lax.bitcast_convert_type(
            jnp.asarray(
                rng.uniform(-1.0, 1.0, (num_pages, page_size, 64)), jnp.float32
            ).astype(jnp.float8_e4m3fn),
            jnp.uint8,
        )
        rb_pad = np.pad(np.asarray(rb), ((0, 0), (0, 0), (0, 64)))
        nope_ranks.append(
            np.asarray(
                native_sc_cache_layout.pack_nope(
                    np.asarray(nb).reshape(num_pages, page_size, 4, 128)
                )
            )
        )
        rope_ranks.append(
            np.asarray(
                native_sc_cache_layout.pack_rope_banded(
                    rb_pad.reshape(num_pages, page_size // 4, 4, 128)
                )
            )
        )
    nope_sharded = jax.device_put(np.concatenate(nope_ranks, axis=0), sh_dcp3)
    rope_sharded = jax.device_put(np.concatenate(rope_ranks, axis=0), sh_dcp3)
    r_ids = jax.device_put(np.arange(8, dtype=np.int32), sh_dcp)
    l_ids = jax.device_put(np.zeros(8, dtype=np.int32), sh_dcp)
    bt = jax.device_put(np.arange(num_pages, dtype=np.int32), sh_rep)

    class _GroupStub:
        world_size = 8

    def _call(fn, *fargs):
        if is_cpu_interpret:
            pltpu.reset_tpu_interpret_mode_state()
        res = fn(*fargs)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), res)
        return res

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, dict(_CPU8_ENV), clear=False))
        os.environ.pop("TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS", None)
        stack.enter_context(
            mock.patch.object(cp_mla, "get_dcp_group", return_value=_GroupStub())
        )
        if is_cpu_interpret:
            stack.enter_context(pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()))
            stack.enter_context(
                mock.patch.object(pltpu, "get_tpu_info", return_value=None)
            )
            stack.enter_context(
                mock.patch.object(
                    native_sc_gather, "dsa_gather_native_sc", _fake_gather_poisoned
                )
            )

        for num_tokens, reals in ((8, (8, 5, 0)), (16, (16, 11, 0))):
            ql_np = rng.normal(0.0, 0.5, (num_tokens, 64, 512)).astype(np.float32)
            qp_np = rng.normal(0.0, 0.5, (num_tokens, 64, 256)).astype(np.float32)
            kvc_np = rng.uniform(-0.5, 0.5, (num_tokens, 512)).astype(np.float32)
            kpe_np = rng.uniform(-0.5, 0.5, (num_tokens, 64)).astype(np.float32)
            ang = np.arange(num_tokens, dtype=np.float32)[:, None] * 0.03
            cos = np.cos(np.repeat(ang, 32, axis=1))
            sin = np.sin(np.repeat(ang, 32, axis=1))
            rt_np = np.stack(
                [
                    np.concatenate([cos, cos, cos, cos], axis=1),
                    np.concatenate([-sin, sin, -sin, sin], axis=1),
                ],
                axis=1,
            ).astype(np.float32)

            topk_ranks = []
            base_counts = [0, 1, 64, 129, 257, 384, 512, 600]
            for r in range(8):
                rk_idx = np.full((num_tokens, 2048), -1, dtype=np.int32)
                if r != 3:  # rank 3 owns nothing -> lse = -inf across all tokens
                    for t in range(num_tokens):
                        cnt = base_counts[(t + r) % len(base_counts)]
                        if t == 0 and r == 0:
                            cnt = 600  # capped overflow (>512)
                        if cnt > 0:
                            rk_idx[t, :cnt] = rng.choice(
                                cache_tokens, size=cnt, replace=False
                            )
                topk_ranks.append(rk_idx)
            topk_sharded = jax.device_put(np.concatenate(topk_ranks, axis=0), sh_dcp2)

            ql = jax.device_put(jnp.asarray(ql_np, jnp.bfloat16), sh_rep)
            qp = jax.device_put(jnp.asarray(qp_np, jnp.bfloat16), sh_rep)
            kvc = jax.device_put(jnp.asarray(kvc_np).astype(jnp.float8_e4m3fn), sh_rep)
            kpe = jax.device_put(jnp.asarray(kpe_np).astype(jnp.float8_e4m3fn), sh_rep)
            rtab = jax.device_put(jnp.asarray(rt_np, jnp.float32), sh_rep)

            off_fuse_xchg = cp_mla.fused_small_a2a_applies(num_tokens)
            off_cfg = ai.resolve_fused_small_a2a_config(
                64, 8, fused_bucket=off_fuse_xchg
            )
            off_fuse_mrg = off_fuse_xchg and num_tokens <= off_cfg.merge_max_tokens
            assert off_fuse_xchg == (num_tokens == 8)
            assert not off_fuse_mrg

            @jax.jit
            def run_off(
                ql,
                qp,
                kvc,
                kpe,
                nc,
                rc,
                tk,
                sl,
                bt,
                qsl,
                dist,
                rtab,
                r_ids,
                l_ids,
                off_fuse_xchg=off_fuse_xchg,
            ):
                return ai.sparse_mla_attention_dcp(
                    ql,
                    qp,
                    kvc,
                    kpe,
                    nc,
                    rc,
                    tk,
                    sl,
                    bt,
                    qsl,
                    dist,
                    mesh,
                    nope_spec,
                    rope_spec,
                    sm_scale=0.125,
                    k_scale=1.0,
                    dcp_size=8,
                    interleave_size=4,
                    q_rope_table=rtab,
                    dcp_rank_ids=r_ids,
                    send_layout=True,
                    fused_exchange=off_fuse_xchg,
                    dcp_lane_ids=l_ids if off_fuse_xchg else None,
                    fused_merge=False,
                )

            with mock.patch.dict(
                os.environ, {"TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "1"}
            ):
                assert cp_mla.fused_small_a2a_applies(num_tokens)
                on_cfg = ai.resolve_fused_small_a2a_config(64, 8, fused_bucket=True)
                assert num_tokens <= on_cfg.merge_max_tokens

                @jax.jit
                def run_on(
                    ql, qp, kvc, kpe, nc, rc, tk, sl, bt, qsl, dist, rtab, r_ids, l_ids
                ):
                    return ai.sparse_mla_attention_dcp(
                        ql,
                        qp,
                        kvc,
                        kpe,
                        nc,
                        rc,
                        tk,
                        sl,
                        bt,
                        qsl,
                        dist,
                        mesh,
                        nope_spec,
                        rope_spec,
                        sm_scale=0.125,
                        k_scale=1.0,
                        dcp_size=8,
                        interleave_size=4,
                        q_rope_table=rtab,
                        dcp_rank_ids=r_ids,
                        send_layout=True,
                        fused_exchange=True,
                        dcp_lane_ids=l_ids,
                        fused_merge=True,
                    )

                @jax.jit
                def run_step16_send(
                    ql, qp, kvc, kpe, nc, rc, tk, sl, bt, qsl, dist, rtab, r_ids
                ):
                    return ai.sparse_mla_attention_dcp(
                        ql,
                        qp,
                        kvc,
                        kpe,
                        nc,
                        rc,
                        tk,
                        sl,
                        bt,
                        qsl,
                        dist,
                        mesh,
                        nope_spec,
                        rope_spec,
                        sm_scale=0.125,
                        k_scale=1.0,
                        dcp_size=8,
                        interleave_size=4,
                        q_rope_table=rtab,
                        dcp_rank_ids=r_ids,
                        send_layout=True,
                        fused_exchange=False,
                        fused_merge=False,
                    )

            tl = num_tokens * 8
            on_full_all = None
            for real in reals:
                live_flag = 1 if real > 0 else 0
                sl = jax.device_put(
                    np.array([1500 if real > 0 else 0], dtype=np.int32), sh_rep
                )
                qsl = jax.device_put(np.array([0, real], dtype=np.int32), sh_rep)
                dist = jax.device_put(
                    np.array([live_flag, live_flag, live_flag], dtype=np.int32), sh_rep
                )

                off_ret = _call(
                    run_off,
                    ql,
                    qp,
                    kvc,
                    kpe,
                    nope_sharded,
                    rope_sharded,
                    topk_sharded,
                    sl,
                    bt,
                    qsl,
                    dist,
                    rtab,
                    r_ids,
                    l_ids,
                )
                with mock.patch.dict(
                    os.environ, {"TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "1"}
                ):
                    _, _, on_merged = _call(
                        run_on,
                        ql,
                        qp,
                        kvc,
                        kpe,
                        nope_sharded,
                        rope_sharded,
                        topk_sharded,
                        sl,
                        bt,
                        qsl,
                        dist,
                        rtab,
                        r_ids,
                        l_ids,
                    )
                    if num_tokens == 16 and real == 16:
                        _, _, s16_send = _call(
                            run_step16_send,
                            ql,
                            qp,
                            kvc,
                            kpe,
                            nope_sharded,
                            rope_sharded,
                            topk_sharded,
                            sl,
                            bt,
                            qsl,
                            dist,
                            rtab,
                            r_ids,
                        )
                        pad_r = sparse_mla.SEND_PAD_ROWS
                        s_off = np.asarray(off_ret[2]).reshape(8, 8, pad_r + tl, 512)
                        s_on = np.asarray(s16_send).reshape(8, 8, pad_r + tl, 512)
                        assert (
                            int(
                                (
                                    s_off[:, :, pad_r - 1 : pad_r + tl]
                                    != s_on[:, :, pad_r - 1 : pad_r + tl]
                                ).sum()
                            )
                            == 0
                        )

                on_ranks = np.asarray(on_merged).reshape(8, num_tokens, 8, 512)
                on_full = on_ranks.transpose(1, 0, 2, 3).reshape(num_tokens, 64, 512)
                if real == num_tokens:
                    on_full_all = on_full

                # Extract per-destination-rank received (o_recv, l_recv) from OFF.
                if num_tokens == 8:
                    o_recv_all = jnp.asarray(off_ret[2]).reshape(8, 8, tl, 512)
                    l_recv_all = jnp.asarray(off_ret[3]).reshape(8, 8, tl)
                    # Mask padding rows >= real*8 as send_small_unpack_local does.
                    row_i = jnp.arange(tl)[None, None, :]
                    o_recv_all = jnp.where(
                        (row_i < real * 8)[..., None],
                        o_recv_all,
                        jnp.bfloat16(0.0),
                    )
                else:
                    pad_r = sparse_mla.SEND_PAD_ROWS
                    # Transpose [sender=8, dest=8, block, 512] -> [dest=8, sender=8]
                    recv_blocks = (
                        jnp.asarray(off_ret[2])
                        .reshape(8, 8, pad_r + tl, 512)
                        .transpose(1, 0, 2, 3)
                    )
                    rows_arr = jnp.array([1 + real * 8], dtype=jnp.int32)
                    unpacked = [
                        cp_mla._mask_unpack(recv_blocks[d], rows_arr, tl)
                        for d in range(8)
                    ]
                    o_recv_all = jnp.stack([u[0] for u in unpacked], axis=0)
                    l_recv_all = jnp.stack([u[1] for u in unpacked], axis=0)

                seq_ranks, xla_ranks, torch_ranks = [], [], []
                for d in range(8):
                    o_d = o_recv_all[d]
                    l_d = l_recv_all[d]
                    seq_ranks.append(
                        np.asarray(_sequential_merge_jax(o_d, l_d)).reshape(
                            num_tokens, 8, 512
                        )
                    )
                    xla_ranks.append(
                        np.asarray(
                            cp_mla.merge_received_jax(o_d, l_d).astype(jnp.bfloat16)
                        ).reshape(num_tokens, 8, 512)
                    )
                    o_t = (
                        torch.from_numpy(np.asarray(o_d.astype(jnp.float32)).copy())
                        .to(torch.bfloat16)
                        .reshape(8, num_tokens, 8, 512)
                    )
                    l_t = torch.from_numpy(np.asarray(l_d).copy()).reshape(
                        8, num_tokens, 8
                    )
                    t_mrg = (
                        cp_mla._merge_received(o_t, l_t)
                        .to(torch.bfloat16)
                        .view(torch.uint16)
                        .numpy()
                    )
                    torch_ranks.append(
                        np.asarray(
                            jax.lax.bitcast_convert_type(
                                jnp.asarray(t_mrg, jnp.uint16), jnp.bfloat16
                            )
                        )
                    )

                def _reassemble(rk_list, n_tok=num_tokens, real_n=real):
                    arr = (
                        np.stack(rk_list, axis=0)
                        .transpose(1, 0, 2, 3)
                        .reshape(n_tok, 64, 512)
                    )
                    return np.where(
                        (np.arange(n_tok)[:, None, None] < real_n),
                        arr,
                        jnp.bfloat16(0.0),
                    )

                # Padding rows >= real in fused output must be strictly zero bits.
                pad_bits = np.asarray(jnp.asarray(on_full[real:]).view(jnp.uint16))
                assert int((pad_bits != 0).sum()) == 0

                seq_full = _reassemble(seq_ranks)
                xla_full = _reassemble(xla_ranks)
                off_full = _reassemble(torch_ranks)
                nd_seq, ulp_seq = _bf16_ulp_diff(on_full, seq_full)
                assert nd_seq == 0 and ulp_seq == 0

                # Against XLA reduce_f32 and PyTorch _merge_received, only W=8 f32
                # sum associativity (plus torch.exp vs jnp.exp) can differ:
                # bf16 outputs match within <=2 bf16 ULP + 1e-5 atol, differing
                # elements are < 1e-4, and non-cancelled elements (|ref| >= 1e-3)
                # stay within <=1 / <=2 bf16 ULP.
                on_f32 = np.asarray(jnp.asarray(on_full, jnp.bfloat16), np.float32)
                for ref_full, max_ulp_non_cancel in ((xla_full, 1), (off_full, 2)):
                    ref_f32 = np.asarray(
                        jnp.asarray(ref_full, jnp.bfloat16), np.float32
                    )
                    np.testing.assert_allclose(on_f32, ref_f32, rtol=2**-6, atol=1e-5)
                    nd_ref, _ = _bf16_ulp_diff(on_full, ref_full)
                    assert nd_ref <= max(2, on_full.size // 10_000)
                    keep = np.abs(ref_f32) >= 1e-3
                    if bool(keep.any()):
                        _, ulp_nc = _bf16_ulp_diff(on_full[keep], ref_full[keep])
                        assert ulp_nc <= max_ulp_non_cancel

            # Positive fault controls on fused exchange + merge.
            if num_tokens == 8:
                assert on_full_all is not None
                sl = jax.device_put(np.array([1500], dtype=np.int32), sh_rep)
                qsl = jax.device_put(np.array([0, 8], dtype=np.int32), sh_rep)
                dist = jax.device_put(np.array([1, 1, 1], dtype=np.int32), sh_rep)
                with mock.patch.dict(
                    os.environ, {"TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "1"}
                ):
                    for fault in ("send_merge_lo_skip", "send_peer_drop"):

                        @jax.jit
                        def run_fault(
                            ql,
                            qp,
                            kvc,
                            kpe,
                            nc,
                            rc,
                            tk,
                            sl,
                            bt,
                            qsl,
                            dist,
                            rtab,
                            r_ids,
                            l_ids,
                            fault=fault,
                        ):
                            return ai.sparse_mla_attention_dcp(
                                ql,
                                qp,
                                kvc,
                                kpe,
                                nc,
                                rc,
                                tk,
                                sl,
                                bt,
                                qsl,
                                dist,
                                mesh,
                                nope_spec,
                                rope_spec,
                                sm_scale=0.125,
                                k_scale=1.0,
                                dcp_size=8,
                                interleave_size=4,
                                q_rope_table=rtab,
                                dcp_rank_ids=r_ids,
                                send_layout=True,
                                fused_exchange=True,
                                dcp_lane_ids=l_ids,
                                fused_merge=True,
                                prologue_fault=fault,
                            )

                        _, _, bad_mrg = _call(
                            run_fault,
                            ql,
                            qp,
                            kvc,
                            kpe,
                            nope_sharded,
                            rope_sharded,
                            topk_sharded,
                            sl,
                            bt,
                            qsl,
                            dist,
                            rtab,
                            r_ids,
                            l_ids,
                        )
                        bad_full = (
                            np.asarray(bad_mrg)
                            .reshape(8, 8, 8, 512)
                            .transpose(1, 0, 2, 3)
                            .reshape(8, 64, 512)
                        )
                        nd_bad, ulp_bad = _bf16_ulp_diff(bad_full, on_full_all)
                        assert nd_bad > 0 and ulp_bad >= 1

        # Condition 3: MTP drafter sized_merge_local at T=16 (OFF vs ON).
        t_mtp = 16
        o_mtp_np = rng.normal(0.0, 1.0, (8 * t_mtp, 64, 512)).astype(np.float32)
        l_mtp_np = rng.normal(-2.0, 1.5, (8 * t_mtp, 64)).astype(np.float32)
        # Rank 2 owns nothing (-inf LSE + NaN partials); row 0 has partial -inf heads.
        l_mtp_np[2 * t_mtp : 3 * t_mtp, :] = -np.inf
        o_mtp_np[2 * t_mtp : 3 * t_mtp, :, :] = np.nan
        l_mtp_np[0, :16] = -np.inf
        o_mtp_np[0, :16, :] = np.nan

        o_mtp = jax.device_put(jnp.asarray(o_mtp_np, jnp.bfloat16), sh_dcp3)
        l_mtp = jax.device_put(jnp.asarray(l_mtp_np, jnp.float32), sh_dcp2)

        def make_mtp_fn(small_cap: int):
            def body(o, lse, rank, lane):
                return cp_mla.sized_merge_local(
                    o,
                    lse,
                    world=8,
                    a2a_max_tokens=32,
                    axis_name=axis,
                    small={
                        "max_tokens": small_cap,
                        "rank": rank,
                        "lane": lane,
                        "lane_axis": "model",
                    },
                )

            return jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(P(axis, None, None), P(axis, None), P(axis), P(axis)),
                    out_specs=P(axis, None, None),
                    check_vma=False,
                )
            )

        cap_off = ai.resolve_fused_small_a2a_config(dcp_size=8).mtp_merge_max_tokens
        with mock.patch.dict(
            os.environ, {"TPU_MLA_FUSED_SMALL_A2A_4CHIP_BUCKETS": "1"}
        ):
            cap_on = ai.resolve_fused_small_a2a_config(dcp_size=8).mtp_merge_max_tokens
        assert cap_off == 8 and cap_on == 16

        mtp_off_fn = make_mtp_fn(cap_off)
        mtp_on_fn = make_mtp_fn(cap_on)
        hlo_off = mtp_off_fn.lower(o_mtp, l_mtp, r_ids, l_ids).as_text()
        hlo_on = mtp_on_fn.lower(o_mtp, l_mtp, r_ids, l_ids).as_text()
        assert "all_to_all" in hlo_off and "pallas_call" not in hlo_off
        assert "all_to_all" not in hlo_on
        if not is_cpu_interpret:
            assert "pallas_call" in hlo_on
        else:
            assert "callback" in hlo_on or "pallas_call" in hlo_on

        y_off = np.asarray(_call(mtp_off_fn, o_mtp, l_mtp, r_ids, l_ids))
        y_on = np.asarray(_call(mtp_on_fn, o_mtp, l_mtp, r_ids, l_ids))
        assert np.isfinite(y_on.astype(np.float32)).all()
        nd_mtp, ulp_mtp = _bf16_ulp_diff(y_on, y_off)
        assert nd_mtp == 0 and ulp_mtp == 0


def test_cpu8_backbone_t8_t16_and_mtp_t16_numerics(_cpu8_worker_run: bool):
    if not _cpu8_worker_run:
        return
    _run_dcp8_backbone_and_mtp_suite(_cpu8_dcp_mesh(), is_cpu_interpret=True)


@pytest.mark.multichip
def test_tpu_dcp8_fused_small_a2a_t8_t16_and_mtp_t16():
    import jax
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk

    if jax.default_backend() != "tpu" or len(jax.devices("tpu")) < _N_DCP:
        pytest.skip(f"needs a TPU host with >= {_N_DCP} TensorCores")
    mesh = Mesh(
        np.asarray(jax.devices("tpu")[:_N_DCP]).reshape(1, _N_DCP),
        ("model", streamindex_topk.DCP_AXIS_NAME),
    )
    _run_dcp8_backbone_and_mtp_suite(mesh, is_cpu_interpret=False)
