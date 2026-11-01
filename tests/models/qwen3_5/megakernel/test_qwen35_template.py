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
"""The Qwen3.5 template as a plugin: shorthand equivalence, a collective-free
`create` and the stdlib-only plugin module (the public-surface guard over
every model folder is `tests/megakernel_runtime/test_template_surface.py`)."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm_torchtpu.megakernel_runtime.agreement as agreement_mod
import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from tests.models.qwen3_5.megakernel.test_vllm_qwen35_e2e import (
    _allocate_aliased_pools,
    _cpu_mesh,
    _tiny_exec,
    _tiny_geom,
)
from tests.models.qwen3_5.megakernel.test_weight_collector import (
    make_synthetic_qwen35_backbone,
)
from vllm_torchtpu.megakernel import Profile, register_profile
from vllm_torchtpu.megakernel import profile as profile_mod
from vllm_torchtpu.megakernel.runtime import EligibilityResult, facts_from_pool_layout
from vllm_torchtpu.megakernel_runtime import testing
from vllm_torchtpu.megakernel_runtime.hooks import StepContext
from vllm_torchtpu.megakernel_runtime.plugin import SHORTHANDS, RunnerView
from vllm_torchtpu.models.qwen3_5 import megakernel as qwen_plugin
from vllm_torchtpu.models.qwen3_5.megakernel import adapter
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35

_RT = Path(driver_mod.__file__).resolve().parent
_SRC = _RT.parents[1]
_QWEN_HF = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])
_MODULE = SHORTHANDS["qwen35"].module


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    monkeypatch.setattr(agreement_mod, "_tp_cpu_group", lambda: None)


def test_shorthand_table_matches_the_template() -> None:
    sh = SHORTHANDS["qwen35"]
    assert sh.architectures == qwen35.Qwen35.hf_architectures
    assert sh.model_types == qwen35.Qwen35.hf_model_type
    assert sh.legacy_keys == adapter.OPTION_KEYS
    assert qwen_plugin.SPEC.name == "qwen35"
    assert qwen_plugin.SPEC.create == ".adapter:create"
    assert qwen_plugin.SPEC.features == frozenset(
        {"spec_decode", "mrope", "prefix_caching", "kv_offload"}
    )
    assert qwen_plugin.SPEC.layout_request is None
    # The runtime enforces these before `create`.
    assert qwen_plugin.SPEC.chips == ("TPU7x",)
    assert qwen_plugin.SPEC.tp_sizes == (8,)
    assert qwen_plugin.SPEC.parallelism == frozenset({"tp", "ep_in_tp"})
    assert qwen_plugin.SPEC.requires == {"mamba_ssm_cache_dtype": "bfloat16"}
    assert qwen_plugin.SPEC.numerics == "approx"


_FRESH = """
import sys
for name in ("torch", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
import vllm_torchtpu.models.qwen3_5.megakernel as t
from vllm_torchtpu.megakernel_runtime.salt import compile_cache_salt
assert t.SPEC.name == "qwen35"
cfg = type("C", (), {"additional_config": {"megakernel": {"plugin": "qwen35"}}})()
assert compile_cache_salt(cfg) is None
print("OK")
"""


def test_template_plugin_imports_without_torch_jax_or_vllm() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _FRESH],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_SRC), "PATH": "/usr/bin:/bin"},
        timeout=120,
    )
    assert out.returncode == 0 and out.stdout.strip() == "OK", out.stderr


# --- create: no collective, no weights ------------------------------------------


def _full_size_runner(**kw: Any) -> Any:
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=8, B=4, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex)
    backbone.layers = list(backbone.layers) * 15
    return testing.fake_runner(
        model=SimpleNamespace(model=backbone), hf_config=None, tp_size=8, **kw
    )


def test_qwen_create_runs_no_collective_and_loads_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("create ran a collective")

    for name in ("all_reduce", "all_gather", "broadcast", "barrier", "all_to_all"):
        monkeypatch.setattr(torch.distributed, name, boom)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(tensor_model_parallel_all_gather=boom),
    )
    view = RunnerView(_full_size_runner(), tp_rank=3)
    prog = adapter.create(view, {"buckets": "8"})
    assert isinstance(prog, adapter.Qwen35Program)
    assert prog.mesh is None and not hasattr(prog, "w_flat")
    assert (prog.tp_rank, prog.decode_buckets) == (3, (8,))
    dec = adapter.create(view, {"buckets": "0"})
    assert dec.code == "empty_buckets"
    with pytest.raises(ValueError, match="unknown option 'profiel'"):
        adapter.create(view, {"profiel": "x"})


def test_qwen_create_declines_are_named() -> None:
    view = RunnerView(_full_size_runner(unified_kv_layout=False), tp_rank=0)
    dec = adapter.create(view, {})
    assert dec.code == "unaligned_cache_layout"
    assert dec.message == adapter.UNALIGNED_LAYOUT_MESSAGE
    view = RunnerView(testing.fake_runner(model=object(), tp_size=8), tp_rank=0)
    assert adapter.create(view, {}).code == "backbone_not_found"


# --- pool slot check ---------------------------------------------------------


def _slot_check_driver(layers: list[Any]) -> Any:
    drv = object.__new__(adapter.Qwen35Program)
    drv.backbone = SimpleNamespace(layers=layers)
    drv.spec = SimpleNamespace(
        cache_backend=SimpleNamespace(layout=qwen35.Qwen35.pool_layout(R=8))
    )
    drv.base_exec = SimpleNamespace(R=8)
    drv._display_name = "test"
    drv.seen = []
    return drv


def test_pool_slot_check_uses_first_recurrent_layer_and_skips_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    la = object()
    layers = [
        SimpleNamespace(linear_attn=None),
        SimpleNamespace(),
        SimpleNamespace(linear_attn=la),
    ]
    drv = _slot_check_driver(layers)

    def facts(runner: Any, linear_attn: Any, pool: Any) -> Any:
        drv.seen.append(linear_attn)
        return facts_from_pool_layout(drv.spec.cache_backend.layout, 8)

    monkeypatch.setattr(drv, "_framework_slot_facts", facts, raising=False)
    drv._check_pool_layout_facts(SimpleNamespace(), object())
    assert drv.seen == [la] and drv.pool_layout_facts is not None

    # No recurrent layers: no slot to check, and no pool_layout_unknown decline.
    drv = _slot_check_driver([SimpleNamespace(linear_attn=None), SimpleNamespace()])

    def never(*_a: Any) -> Any:
        raise AssertionError("no framework slot facts without recurrent layers")

    monkeypatch.setattr(drv, "_framework_slot_facts", never, raising=False)
    drv._check_pool_layout_facts(SimpleNamespace(), object())
    assert drv.pool_layout_facts is None


# --- shorthand vs plugin key: identical programs ----------------------------------


@pytest.fixture
def tiny_template(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Serve the template on a tiny CPU geometry (TP=1, one (1, 2) bucket)."""
    g = _tiny_geom(L=4, att_every=4)
    ex = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    monkeypatch.setattr(
        qwen35.Qwen35, "derive_geometry", classmethod(lambda cls, *a, **k: g)
    )
    monkeypatch.setattr(
        qwen_plugin, "SPEC", dataclasses.replace(qwen_plugin.SPEC, tp_sizes=(1,))
    )
    monkeypatch.setattr(
        adapter, "validate_hf_config", lambda *_a, **_k: EligibilityResult(True)
    )
    real_cls = adapter.Qwen35Program
    monkeypatch.setattr(
        adapter,
        "Qwen35Program",
        lambda model, **kw: real_cls(model, geom=g, base_exec=ex, **kw),
    )
    monkeypatch.setattr(profile_mod, "_REGISTRY", dict(profile_mod._REGISTRY))
    monkeypatch.setattr(
        profile_mod, "_MODEL_DEFAULTS", dict(profile_mod._MODEL_DEFAULTS)
    )
    register_profile(
        Profile(
            name="qwen35-test-tiny",
            model="qwen3_5",
            buckets=(2,),
            verify_buckets=(),
        ),
        replace=True,
    )
    monkeypatch.setattr(adapter, "DEFAULT_PROFILE", "qwen35-test-tiny")

    def _facts(self: Any, view: Any, linear_attn: Any, pool: Any) -> Any:
        return facts_from_pool_layout(self.spec.cache_backend.layout, self.base_exec.R)

    monkeypatch.setattr(real_cls, "_framework_slot_facts", _facts)

    def make(env: Any, megakernel: Any) -> Any:
        if env is None:
            monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
        else:
            monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", env)
        torch.manual_seed(0)
        backbone = make_synthetic_qwen35_backbone(g, ex)
        _allocate_aliased_pools(backbone, g, 96)
        qsl = np.zeros(16, dtype=np.int32)
        qsl[:3] = [0, 1, 2]
        runner = testing.fake_runner(
            model=SimpleNamespace(model=backbone),
            hf_config=_QWEN_HF,
            megakernel=megakernel,
            mesh=_cpu_mesh(1),
            mamba_ssm_cache_dtype="bfloat16",
            token_paddings=(2, 4, 8),
            req_block_pairs=((4, 2),),
            max_pages_per_seq=2,
            _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
            query_start_loc_np=qsl,
            _cached_comb_key=None,
            _combined_request_distribution_cpu=None,
        )
        return driver_mod.ProgramDriver.maybe_create(runner), runner

    return make


def _w_flat_digest(drv: Any) -> str:
    h = hashlib.sha256()
    for t in drv.program.w_flat:
        arr = t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy()
        h.update(str(tuple(t.shape)).encode() + str(t.dtype).encode() + arr.tobytes())
    return h.hexdigest()


def _program_state(drv: Any) -> tuple[Any, ...]:
    p = drv.program
    return (
        p.profile.name,
        p.mode,
        p.decode_buckets,
        p.verify_buckets,
        p._reshard_ep_to_tp,
        p.fused_pool_io,
        p.pool_gather,
        p._compile_hlo_check,
    )


_SAME_PROGRAM = [
    ("1", None),
    ("true", None),
    ("qwen35", None),
    (None, True),
    (None, "qwen35"),
    (None, {"plugin": "qwen35"}),
    (None, {"plugin": _MODULE}),
    (None, {"plugin": f"{_MODULE}:SPEC", "options": {}}),
]
_SAME_EP = [("ep", None), (None, {"plugin": "qwen35", "options": {"mode": "ep"}})]


def test_shorthand_and_plugin_key_build_identical_programs(tiny_template: Any) -> None:
    for group in (_SAME_PROGRAM, _SAME_EP):
        states, digests = set(), set()
        for env, add in group:
            drv, _ = tiny_template(env, add)
            assert isinstance(drv, driver_mod.ProgramDriver), (env, add)
            states.add(_program_state(drv))
            digests.add(_w_flat_digest(drv))
        assert len(states) == 1 and len(digests) == 1, (group, states)
    assert next(iter(states))[1] == "ep"


def _md_dict(drv: Any) -> dict[str, Any]:
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([10, 11, 12, 13, 0, 0, 0, 0], dtype=torch.int32),
        seq_lens=torch.tensor([6, 10, 0, 0], dtype=torch.int32),
    )
    md = {drv.program.attn_layer_name: attn_md}
    for sub, name in enumerate(drv.program.mamba_layer_names):
        md[name] = SimpleNamespace(
            mamba_state_indices=torch.tensor(
                [1 + 2 * sub, 2 + 2 * sub, 0, 0], dtype=torch.int32
            ),
            mamba_ckpt_indices=None,
            mamba_slot_read_offsets=None,
        )
    return md


def test_shorthand_and_plugin_key_serve_identical_steps(
    tiny_template: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    outs, keys = [], []
    for env, add in (("1", None), (None, {"plugin": _MODULE})):
        drv, runner = tiny_template(env, add)
        runner._megakernel = drv
        drv.bind_and_warmup(runner)
        assert runner._megakernel is drv and drv._bound
        keys.append(sorted(drv._steps))
        md = _md_dict(drv)
        monkeypatch.setattr(
            driver_mod,
            "get_forward_context",
            lambda md=md: SimpleNamespace(attn_metadata=md),
        )
        ids = torch.tensor([5, 12], dtype=torch.int32)
        hit, out = drv.try_forward(
            runner, StepContext(ids, torch.tensor([5, 9], dtype=torch.int32))
        )
        assert hit and out.shape == (2, 256)
        outs.append(out.view(torch.uint16).clone())
    assert keys[0] == keys[1] == [(1, 2)]
    assert torch.equal(outs[0], outs[1])


def test_template_default_profile_and_baseline_option() -> None:
    """Without a profile the template serves `qwen35-candidate30-fused`;
    exact numerics (`qwen35-baseline`) are one option away."""
    view = RunnerView(_full_size_runner(), tp_rank=0)
    assert adapter.DEFAULT_PROFILE == "qwen35-candidate30-fused"
    assert adapter.create(view, {}).profile.name == "qwen35-candidate30-fused"
    base = adapter.create(view, {"profile": "qwen35-baseline"})
    assert base.profile.name == "qwen35-baseline"


# Every decline the template's `create` / `bind` can return.
_QWEN_DECLINES = frozenset(
    {
        # create: checkpoint (`validate_model_hf_config`) and backbone
        "non_moe_checkpoint",
        "invalid_geometry",
        "geometry_mismatch",
        "backbone_not_found",
        # create: `check_startup_eligibility` with the arguments it passes
        # (no TP size: the runtime checks `SPEC.tp_sizes`)
        "layer_count_mismatch",
        "unaligned_cache_layout",
        # create: `check_post_mesh_eligibility` (no mesh_size, no device_kind:
        # the runtime checks `SPEC.chips`)
        "empty_buckets",
        # bind: slot layout (`compare_layout_facts`) and generic layout errors
        "pool_layout_unknown",
        "pool_layout_mismatch",
        "conv_state_dtype_unsupported",
        "cache_layout_incompatible",
    }
)


def _false_result_codes(path: Path, *func_names: str) -> set[str]:
    """Literal codes of `EligibilityResult(False, <code>, ...)` / `Decline(<code>,
    ...)` in the named functions of `path` (all functions if none); a `code`
    variable assigned string literals (or a conditional of them) counts too."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and (not func_names or n.name in func_names)
    ]
    assert not func_names or len(funcs) == len(func_names), func_names

    def literals(expr: ast.AST) -> set[str]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.IfExp):
            return literals(expr.body) | literals(expr.orelse)
        return set()

    codes: set[str] = set()
    for fn in funcs:
        code_vars: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "code" for t in node.targets
            ):
                code_vars |= literals(node.value)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func).rsplit(".", 1)[-1]
            args = node.args
            if name == "Decline" and args:
                codes |= literals(args[0])
            if (
                name == "EligibilityResult"
                and len(args) >= 2
                and isinstance(args[0], ast.Constant)
                and args[0].value is False
            ):
                codes |= literals(args[1])
                if isinstance(args[1], ast.Name) and args[1].id == "code":
                    codes |= code_vars
    return codes


def test_every_template_decline_is_in_the_vote_table() -> None:
    import vllm_torchtpu.megakernel as mk_lib

    codes = driver_mod.vote_codes(qwen_plugin.SPEC.decline_codes)
    assert codes[-1] == "other" and len(set(codes)) == len(codes)
    missing = sorted(c for c in _QWEN_DECLINES if c not in codes[:-1])
    assert missing == [], missing
    # The template lists only codes the runtime doesn't already have.
    assert not set(qwen_plugin.SPEC.decline_codes) & set(driver_mod.DECLINE_VOTE_CODES)

    # Static scan: the adapter and the library checks it calls emit nothing else.
    lib = Path(mk_lib.__file__).resolve().parent
    scanned = _false_result_codes(Path(adapter.__file__))
    scanned |= _false_result_codes(lib / "validate.py", "validate_model_hf_config")
    scanned |= _false_result_codes(lib / "runtime" / "layout_facts.py")
    elig = lib / "runtime" / "eligibility.py"
    post = _false_result_codes(elig, "check_post_mesh_eligibility")
    # create passes no mesh_size and no device_kind
    scanned |= post - {"mesh_size_mismatch", "unsupported_tpu_chip"}
    assert {"pool_layout_mismatch", "conv_state_dtype_unsupported"} <= scanned
    assert {"non_moe_checkpoint", "invalid_geometry", "geometry_mismatch"} <= scanned
    assert scanned <= _QWEN_DECLINES, sorted(scanned - _QWEN_DECLINES)
    assert {"unsupported_tpu_chip", "empty_buckets"} <= post
    # The runtime's chip / TP-size declines come before any vote (`check_spec`,
    # identical on every rank), so they aren't in the vote table.
    assert not {"unsupported_tpu_chip", "unsupported_tp_size"} & set(codes)
