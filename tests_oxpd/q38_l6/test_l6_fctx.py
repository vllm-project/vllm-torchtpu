# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever fctx: runner/fast_forward_context.py.

The first four tests are ported from the Qwen3.5 lane's
tests/runner/test_fast_forward_context.py (shriharim, 14bd07701, branch
shriharim/qwen35-pd/PR11-batch1-host-trims): the cached context equals the
stock one except attn_metadata, it is keyed by num_tokens and the DP counts,
extra keyword arguments take the stock path, and the previous context comes
back on exit. The rest cover the Q38 port: per-call containers and DPMetadata,
the step-time switch through core/l6_ab.py, and the call sites."""

import ast
import dataclasses
import pathlib
import types

import pytest
import torch
import vllm.forward_context as fc

from vllm_torchtpu.runner import fast_forward_context as ffc


@pytest.fixture(autouse=True)
def _cpu_platform_hook(monkeypatch):
    # The TPU platform's hook moves the DP token tensor to the "tpu" device, which a CPU-only
    # pod lacks. Stock and fast paths call the same hook, so a CPU stub keeps the comparison.
    try:
        torch.zeros(1).to("tpu")
    except Exception:
        monkeypatch.setattr(fc.current_platform, "set_additional_forward_context", lambda *a, **k: {}, raising=False)
    ffc._CACHE.clear()
    yield
    ffc._CACHE.clear()


class _PC:
    data_parallel_size = 1
    use_sequence_parallel_moe = True
    is_moe_model = True
    data_parallel_rank = 0


def _cfg():
    cc = types.SimpleNamespace(static_forward_context={"x": 1}, fast_moe_cold_start=False, static_all_moe_layers=None)
    return types.SimpleNamespace(parallel_config=_PC(), compilation_config=cc)


def _fields(ctx):
    d = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(ctx)}
    d.pop("attn_metadata")
    dp = d.pop("dp_metadata")
    return d, (None if dp is None else dp.num_tokens_across_dp_cpu.tolist())


def test_cached_context_matches_stock_and_swaps_metadata():
    cfg = _cfg()
    md1, md2 = {"a": 1}, {"a": 2}
    t = torch.tensor([4], dtype=torch.int32)
    with fc.set_forward_context(md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        stock = _fields(fc.get_forward_context())
    with ffc.fast_set_forward_context(md1, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        first = fc.get_forward_context()
        assert first.attn_metadata is md1
    with ffc.fast_set_forward_context(md2, cfg, num_tokens=4, num_tokens_across_dp=t.clone()):
        second = fc.get_forward_context()
        assert second.attn_metadata is md2
        assert second is not first
        f2 = _fields(second)
    s_d, s_dp = stock
    f_d, f_dp = f2
    assert f_d.keys() == s_d.keys()
    for k in s_d:
        if k == "additional_kwargs":
            continue
        assert f_d[k] == s_d[k], k
    assert f_dp == s_dp
    assert len(ffc._CACHE) == 1


def test_restores_previous_context_and_keys_by_num_tokens():
    cfg = _cfg()
    before = fc._forward_context
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32)):
        pass
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=torch.tensor([8], dtype=torch.int32)):
        assert fc.get_forward_context().dp_metadata is not None
    assert fc._forward_context is before
    assert len(ffc._CACHE) == 2


def test_extra_kwargs_take_stock_path():
    cfg = _cfg()
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32), skip_compiled=True):
        assert fc.get_forward_context().skip_compiled is True
    assert len(ffc._CACHE) == 0


def test_num_tokens_is_part_of_the_key():
    # Same (absent) DP tensor, different num_tokens: the stock path derives the DP token count from
    # num_tokens, so a cached context for 4 tokens must not be reused for 8.
    cfg = _cfg()
    with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=None):
        pass
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist() == [8]
    with ffc.fast_set_forward_context({}, cfg, num_tokens=8, num_tokens_across_dp=None):
        assert fc.get_forward_context().dp_metadata.num_tokens_across_dp_cpu.tolist() == [8]
    assert len(ffc._CACHE) == 2


def test_writes_during_one_forward_do_not_reach_the_next():
    """Per-call containers, counters and DPMetadata: a forward that writes into its
    context (slot_mapping, additional_kwargs, moe_layer_index, dp local_sizes) must
    not leave anything behind for the next step, as with the stock path."""
    cfg = _cfg()
    t = lambda: torch.tensor([4], dtype=torch.int32)  # noqa: E731
    for _ in range(3):
        with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=t()):
            ctx = fc.get_forward_context()
            assert ctx.slot_mapping == {}
            assert ctx.additional_kwargs == {}
            assert ctx.moe_layer_index == 0
            assert getattr(ctx.dp_metadata, "local_sizes", None) is None
            ctx.slot_mapping["layer"] = torch.zeros(1)
            ctx.additional_kwargs["k"] = 1
            ctx.moe_layer_index += 5
            ctx.dp_metadata.local_sizes = [1, 2]


def test_none_metadata_connector_site_is_cached():
    """The connector-setup site passes attn_metadata None and no token count."""
    cfg = types.SimpleNamespace(parallel_config=types.SimpleNamespace(
        data_parallel_size=1, use_sequence_parallel_moe=False, is_moe_model=True, data_parallel_rank=0),
        compilation_config=types.SimpleNamespace(static_forward_context={}, fast_moe_cold_start=False,
                                                 static_all_moe_layers=None))
    for _ in range(2):
        with ffc.fast_set_forward_context(None, cfg):
            assert fc.get_forward_context().attn_metadata is None
            assert fc.get_forward_context().dp_metadata is None
    assert len(ffc._CACHE) == 1


def test_enabled_follows_the_l6_lever(monkeypatch):
    from vllm_torchtpu.core import l6_ab

    monkeypatch.delenv("VLLM_TPU_L6_FAST_FWD_CTX", raising=False)
    monkeypatch.delenv("VLLM_TPU_L6_AB_PERIOD_MS", raising=False)
    l6_ab._resolve()
    l6_ab.begin_step()
    assert ffc.enabled() is False
    monkeypatch.setenv("VLLM_TPU_L6_FAST_FWD_CTX", "1")
    l6_ab._resolve()
    l6_ab.begin_step()
    assert ffc.enabled() is True
    monkeypatch.delenv("VLLM_TPU_L6_FAST_FWD_CTX")
    l6_ab._resolve()


def test_engagement_counts_hits_and_builds(monkeypatch):
    from vllm_torchtpu.core import l6_ab

    monkeypatch.setenv("VLLM_TPU_L6_FAST_FWD_CTX", "1")
    monkeypatch.delenv("VLLM_TPU_L6_AB_PERIOD_MS", raising=False)
    l6_ab._resolve()
    l6_ab.begin_step()
    cfg = _cfg()
    for _ in range(3):
        with ffc.fast_set_forward_context({}, cfg, num_tokens=4, num_tokens_across_dp=torch.tensor([4], dtype=torch.int32)):
            pass
    assert l6_ab._state["counts"] == {"fctx.build": 1, "fctx.hit": 2}
    monkeypatch.delenv("VLLM_TPU_L6_FAST_FWD_CTX")
    l6_ab._resolve()


def _call_sites(path: pathlib.Path) -> list[str]:
    """Source of every call whose callee mentions set_forward_context."""
    tree = ast.parse(path.read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and "set_forward_context" in ast.unparse(n.func):
            out.append(ast.unparse(n.func))
    return out


def test_step_call_sites_pick_the_fast_path_only_when_enabled():
    import vllm_torchtpu.runner.tpu_runner as tr
    import vllm_torchtpu.spec_decode.eagle3 as e3

    want = "_ffc.fast_set_forward_context if _ffc.enabled() else set_forward_context"
    runner = _call_sites(pathlib.Path(tr.__file__))
    eagle = _call_sites(pathlib.Path(e3.__file__))
    assert sum(want in c for c in runner) == 2, runner  # connector setup + target forward
    assert sum(want in c for c in eagle) == 2, eagle  # draft fused loop + draft step
    # every other call stays the stock set_forward_context (warm-up / dummy paths)
    assert all(c in (want, f"({want})", "set_forward_context") or want in c for c in runner + eagle), runner + eagle
