# SPDX-License-Identifier: Apache-2.0
"""q38 PORT-BL UNION: the union route's kernels and the HBM-typed schedule lower to Mosaic for TPU7x on CPU
(ported from Benliang Wang <benliangw@google.com>, 2ce02bb1a on benliangw/q38fh-integ-pin-5b44702aea).

Neither the SparseCore metadata kernel nor the selection-mask batched RPA runs on CPU; lowering is the CPU check
that both build for the served shapes, inside the real ``pooled_qsa_request_attend_dcp_step``, at served dims
(64 request slots, main-KV DCP 4, 512-token fp8 pages, 512 groups a row, 12 query heads).

  LW1 key on: the O1s attend lowers with the SparseCore kernel (``qsa_union_meta_r<covered>_p2080``) and the
      union RPA kernels (``-sel-ro``) and keeps the read-only per-row fallback; key off: none of them.
  LW2 the schedule key changes only the schedule kernel's output placement: same RPA kernel names, a different
      lowered program, and the HBM memory space on the schedule outputs.
"""

from __future__ import annotations

import functools
import json
import re

import jax
import jax.numpy as jnp
import pytest
from jax._src.mesh import AbstractDevice, AbstractMesh
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa, qsa_rpa

TPU7X = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
SERVED = {
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": "1",
    "VLLM_TPU_QSA_DECODE_SLOT_READ": "1",
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": "1",
    "VLLM_TPU_QSA_DECODE_RPA_BKV": "512",
    "VLLM_TPU_QSA_DECODE_RPA_LANES": "4",
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": "128",
    "VLLM_TPU_QSA_DECODE_RPA_FETCH": "fast",
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": "1",
}
UNION = "VLLM_TPU_QSA_DECODE_RPA_UNION_O1S"
SCHED = "VLLM_TPU_RPA_SCHEDULE_HBM_O1S"
SLOTS, COLS, PAGES = 64, 20, 2048


def _lower_attend(rows, max_query_len=4):
    spec = jax.ShapeDtypeStruct
    specs = (
        spec((rows, 12, 256), jnp.bfloat16),  # q
        spec((rows, 12, 256), jnp.bfloat16),  # gate
        spec((rows,), jnp.int32),  # lengths
        spec((rows,), jnp.int32),  # req
        spec((SLOTS + 1,), jnp.int32),  # starts
        spec((SLOTS, COLS), jnp.int32),  # main_table
        spec((SLOTS,), jnp.int32),  # context
        spec((PAGES, 512, 1, 4, 256), jnp.float8_e4m3fn),  # pool
        spec((8, rows // 8, 512), jnp.int32),  # gathered
        spec((1,), jnp.int32),  # dcp_rank
    )
    fn = functools.partial(pooled_qsa.pooled_qsa_request_attend_dcp_step, kv_heads=1, compress_ratio=4,
                           token_topk=2048, dcp_size=4, k_scale=0.375, v_scale=2.5, max_query_len=max_query_len)
    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((), (), abstract_device=device)):
        text = jax.jit(fn, donate_argnums=(7,)).trace(*specs).lower(lowering_platforms=("tpu",)).as_text()
    jax.clear_caches()
    return text


def _kernel_names(text):
    return sorted(set(re.findall(r'kernel_name = "([^"]+)"', text)))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: TPU7X)
    for k, v in SERVED.items():
        monkeypatch.setenv(k, v)
    for k in (UNION, SCHED):
        monkeypatch.delenv(k, raising=False)
    qsa_rpa.UNION_TRACED.clear()
    jax.clear_caches()
    return monkeypatch


@pytest.mark.parametrize("rows", [96, 256])
def test_lw1_union_route_lowers_for_tpu7x(env, rows):
    off = _lower_attend(rows)
    env.setenv(UNION, "640")
    on = _lower_attend(rows)
    names_on, names_off = _kernel_names(on), _kernel_names(off)
    covered = 32 if rows <= 128 else 64
    meta = [n for n in names_on if n.startswith("qsa_union_meta")]
    sel = [n for n in names_on if "-sel" in n]
    print(f"PBLU LW1 rows {rows}: on kernels {names_on}; off kernels {names_off}; on {len(on)} chars off {len(off)}")
    assert meta and all(m.startswith(f"qsa_union_meta_r{covered}_p2080") for m in meta), meta
    assert sel and all(n.endswith("-sel-ro") for n in sel), sel
    assert any(n.startswith("RPAd") and "-ro" in n and "-sel" not in n for n in names_on), names_on
    assert not any("qsa_union_meta" in n or "-sel" in n or n.endswith("-ro") for n in names_off), names_off


def test_lw1_one_row_programs_are_untouched(env):
    """A one-row-a-request program (max_query_len 1: the draft / non-spec shape) lowers to the same program with
    the union key on as off."""
    off = _lower_attend(64, max_query_len=1)
    env.setenv(UNION, "640")
    on = _lower_attend(64, max_query_len=1)
    print(f"PBLU LW1 max_query_len 1: identical {on == off}; kernels {_kernel_names(on)}")
    assert on == off and not qsa_rpa.UNION_TRACED


def _custom_calls(text):
    """[(kernel name, custom_call_config without its body, body)] of every tpu_custom_call, in program order."""
    out = []
    for m in re.finditer(r'backend_config = "((?:[^"\\]|\\.)*)"', text):
        raw = re.sub(r"\\([0-9A-Fa-f]{2})", lambda x: chr(int(x.group(1), 16)), m.group(1))
        cc = json.loads(raw)["custom_call_config"]
        name = re.search(r'kernel_name = "([^"]+)"', text[m.end() : m.end() + 20000])
        body = cc.pop("body", "")
        out.append((name.group(1) if name else "?", cc, body))
    return out


def test_lw2_schedule_hbm_changes_only_the_schedule_placement(env):
    """Same kernels and the same Mosaic bodies (byte for byte); the only change is the schedule kernel's
    ``output_memory_space_colors``, every output on color 0 (HBM) for its array outputs (PORT-BL-UNION diag1
    j-213fcb5f found exactly this)."""
    off = _custom_calls(_lower_attend(128))
    env.setenv(SCHED, "1")
    on = _custom_calls(_lower_attend(128))
    assert [n for n, *_ in on] == [n for n, *_ in off], ([n for n, *_ in on], [n for n, *_ in off])
    changed = []
    for (name, cc_on, body_on), (_, cc_off, body_off) in zip(on, off):
        assert body_on == body_off, f"{name}: Mosaic body changed"
        if cc_on != cc_off:
            changed.append(name)
            extra = {k: v for k, v in cc_on.items() if cc_off.get(k) != v}
            print(f"PBLU LW2 {name}: config keys changed {sorted(extra)}; "
                  f"colors {[c.get('color') for c in cc_on.get('output_memory_space_colors', [])]}")
            assert sorted(extra) == ["output_memory_space_colors"] and "output_memory_space_colors" not in cc_off
            colors = cc_on["output_memory_space_colors"]
            assert colors and all(c["color"] == 0 for c in colors), colors
    print(f"PBLU LW2 kernels {[n for n, *_ in on]}; configs changed only for {changed}")
    assert changed == ["rpa_metadata_schedule"], changed
