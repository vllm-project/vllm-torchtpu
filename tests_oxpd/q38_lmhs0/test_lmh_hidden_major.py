# SPDX-License-Identifier: Apache-2.0
"""q38 L4 LMH-STREAM stage 0: the draft (and, behind its own flag, the verify) LM head stored
[hidden, vocab shard] (VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT / _VERIFY; port of PR 2821's
TPU_LM_HEAD_HIDDEN_MAJOR, 049bb029b, Yin Lin). CDK CPU pod only.

The first two tests are PR 2821's tests/runner/test_lm_head_hidden_major.py (credited), widened by
three refusal cases. The rest pin the q38 wiring: the draft-only copy leaves the target head's bytes,
object and method untouched; the draft's logits and top tokens run through the hidden-major method;
a TP-sharded (padded) head gives the stock top tokens; the verify flag needs the draft flag when the
head is shared; refusals fail-stop; the runner arms after the drafter on the last stage only; flags
off is the base 2e166829a AST for AST; the Qwen3.8 TP8 census (stored layout and bytes)."""

import ast
import hashlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vpe
from vllm.model_executor.layers import logits_processor as lp_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor

import vllm_torchtpu
from vllm_torchtpu import envs
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.spec_decode.utils import maybe_share_lm_head

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lmhs0_flagoff_ast as fo  # noqa: E402

PKG = os.path.dirname(os.path.abspath(vllm_torchtpu.__file__))
DRAFT_FLAG = "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT"
VERIFY_FLAG = "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY"


# --------------------------------------------------------------------------- PR 2821's tests (ported)
def _head(weight):
    head = vpe.ParallelLMHead.__new__(vpe.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    head.quant_method = vpe.UnquantizedEmbeddingMethod()
    return head


def _model(head, embed_weight=None, head_dtype=None):
    embed = SimpleNamespace(weight=embed_weight)
    return SimpleNamespace(
        lm_head=head,
        logits_processor=SimpleNamespace(head_dtype=head_dtype),
        model=SimpleNamespace(embed_tokens=embed),
    )


def _relayout(model):
    with (
        patch.object(runner_utils, "_to_row_major", lambda t: t.clone()),
        patch.object(runner_utils, "_install_entry_layout_pin") as pin,
    ):
        changed = runner_utils.relayout_lm_head(model)
    return changed, pin


@pytest.mark.parametrize("bias", [False, True])
def test_hidden_major_head_projects_like_the_vocab_major_head(bias):
    # Small integers: every sum is exact, so any accumulation order agrees.
    weight = torch.randint(-3, 4, (40, 16)).float()
    x = torch.randint(-3, 4, (2, 5, 16)).float()
    b = torch.randint(-3, 4, (40,)).float() if bias else None
    head = _head(weight.clone())
    want = head.quant_method.apply(head, x, b)
    changed, pin = _relayout(_model(head))
    assert changed
    pin.assert_called_once_with()
    assert head.weight.shape == (16, 40)
    assert torch.equal(head.quant_method.apply(head, x, b), want)
    with pytest.raises(RuntimeError):
        head.quant_method.embedding(head, torch.tensor([1]))


@pytest.mark.parametrize(
    "case", ["tied", "fp32_head", "quantized", "already_hidden_major", "not_parallel_lm_head", "three_d"]
)
def test_head_kept_when_its_weight_is_read_another_way(case):
    weight = torch.randn(40, 16)
    head = _head(weight)
    if case == "quantized":
        head.quant_method = object()
    if case == "already_hidden_major":
        head.quant_method = runner_utils.HiddenMajorLMHeadMethod()
    if case == "not_parallel_lm_head":
        head = torch.nn.Module()
        head.weight = torch.nn.Parameter(weight, requires_grad=False)
        head.quant_method = vpe.UnquantizedEmbeddingMethod()
    if case == "three_d":
        head.weight = torch.nn.Parameter(weight.reshape(40, 4, 4), requires_grad=False)
    model = _model(
        head,
        embed_weight=head.weight if case == "tied" else None,
        head_dtype=torch.float64 if case == "fp32_head" else None,
    )
    want_shape = tuple(head.weight.shape)
    why = runner_utils.hidden_major_refusal(head, model.logits_processor, model.model.embed_tokens)
    assert why is not None
    print(f"\nLMHS0_REFUSAL {case}: {why}")
    changed, pin = _relayout(model)
    assert not changed
    pin.assert_not_called()
    assert tuple(head.weight.shape) == want_shape


# --------------------------------------------------------------------------- q38 helpers
def _real_head(vocab, hidden, rank, tp, dtype=torch.bfloat16, bias=False):
    """A real vLLM ParallelLMHead for TP rank `rank` of `tp` (real shard indices and padding)."""
    with (
        patch.object(vpe, "get_tensor_model_parallel_rank", return_value=rank),
        patch.object(vpe, "get_tensor_model_parallel_world_size", return_value=tp),
    ):
        return vpe.ParallelLMHead(vocab, hidden, bias=bias, params_dtype=dtype, prefix="lm_head")


def _processor(vocab):
    with patch.object(lp_module, "get_current_vllm_config", return_value=SimpleNamespace(model_config=None)):
        return LogitsProcessor(vocab)


class _Inner(torch.nn.Module):
    def __init__(self, embed=None):
        super().__init__()
        self.embed_tokens = embed if embed is not None else SimpleNamespace(weight=None)


class _Target(torch.nn.Module):
    """Qwen4ExpForCausalLM's head-side attributes (lm_head, logits_processor, model.embed_tokens)."""

    plain_compute_logits = True

    def __init__(self, head, vocab):
        super().__init__()
        self.model = _Inner()
        self.lm_head = head
        self.logits_processor = _processor(vocab)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)


class _Draft(torch.nn.Module):
    """Qwen4ExpMTP's head-side attributes: its own placeholder head until maybe_share_lm_head binds the target's."""

    has_own_lm_head = False
    plain_compute_logits = True

    def __init__(self, vocab, hidden, rank, tp):
        super().__init__()
        self.model = _Inner()
        self.lm_head = _real_head(vocab, hidden, rank, tp)
        self.logits_processor = _processor(vocab)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)


def _pair(vocab, hidden, rank, tp, weight_full):
    """(target, draft) for one TP rank, the draft sharing the target's head like Qwen4ExpMTP does."""
    head = _real_head(vocab, hidden, rank, tp)
    si = head.shard_indices
    rows = head.weight.shape[0]
    shard = torch.full((rows, hidden), 7.0, dtype=head.weight.dtype)  # padding rows would win if unmasked
    n = si.org_vocab_end_index - si.org_vocab_start_index
    shard[:n] = weight_full[si.org_vocab_start_index : si.org_vocab_end_index]
    head.weight.data.copy_(shard)
    target = _Target(head, vocab)
    draft = _Draft(vocab, hidden, rank, tp)
    maybe_share_lm_head(draft, target, False, True, materialize_if_mismatched=False)
    assert draft.lm_head is target.lm_head
    return target, draft


def _sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


class _RowMajor:
    """Stands in for runner_utils._to_row_major (torch_tpu LayoutContext; TPU only): records what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, t):
        self.calls.append((tuple(t.shape), t.is_contiguous(), tuple(t.stride()), t.dtype))
        return t.clone()


@pytest.fixture
def rm():
    rec = _RowMajor()
    with (
        patch.object(runner_utils, "_to_row_major", rec),
        patch.object(runner_utils, "_install_entry_layout_pin") as pin,
    ):
        rec.pin = pin
        yield rec


def _ints(shape, lo=-2, hi=3, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, shape, generator=g).to(dtype)


# --------------------------------------------------------------------------- draft only
def test_draft_only_copy_leaves_the_target_head_untouched(rm):
    vocab, hidden, tp, rank = 1000, 16, 4, 1
    target, draft = _pair(vocab, hidden, rank, tp, _ints((vocab, hidden)))
    th = target.lm_head
    w0, m0, sha0 = th.weight, th.quant_method, _sha(th.weight)
    census = runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=False)
    print(f"\nLMHS0_ARM draft_only census={census}")
    # the target (verify) head: same module, same Parameter, same bytes, same stock method
    assert target.lm_head is th and th.weight is w0 and th.quant_method is m0
    assert type(th.quant_method) is vpe.UnquantizedEmbeddingMethod
    assert _sha(th.weight) == sha0 and tuple(th.weight.shape) == (256, hidden)
    # the draft head: its own module and table, [hidden, vocab shard], row-major, same values
    dh = draft.lm_head
    assert dh is not th and dh.weight is not th.weight
    assert type(dh.quant_method) is runner_utils.HiddenMajorLMHeadMethod
    assert tuple(dh.weight.shape) == (hidden, 256) and tuple(dh.weight.stride()) == (256, 1)
    assert dh.weight.dtype == th.weight.dtype and not dh.weight.requires_grad
    assert torch.equal(dh.weight, th.weight.t())
    assert dh.shard_indices is th.shard_indices and dh.tp_size == th.tp_size == tp
    assert rm.calls == [((hidden, 256), True, (256, 1), torch.bfloat16)]
    rm.pin.assert_called_once_with()
    assert census["draft"].startswith(f"own hidden-major copy ({hidden}, 256) bfloat16 {hidden * 256 * 2} B")
    assert census["verify"].startswith(f"kept [vocab, hidden] (256, {hidden})")


def test_draft_head_runs_hidden_major_through_get_top_tokens(rm):
    """Engagement: the draft's logits and top tokens go through HiddenMajorLMHeadMethod.apply on the
    [hidden, vocab] table; the target's do not, and stay bitwise what they were before arming."""
    vocab, hidden = 600, 16
    head = vpe.ParallelLMHead(vocab, hidden, params_dtype=torch.bfloat16, prefix="lm_head", disable_tp=True)
    head.weight.data.copy_(_ints((head.weight.shape[0], hidden), seed=1))
    target = _Target(head, vocab)
    draft = _Draft.__new__(_Draft)
    torch.nn.Module.__init__(draft)
    draft.model = _Inner()
    draft.lm_head = head
    draft.logits_processor = _processor(vocab)
    x = _ints((24, hidden), seed=2)
    t_logits0 = target.compute_logits(x)
    t_top0 = target.logits_processor.get_top_tokens(target.lm_head, x)
    runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=False)
    seen = []
    orig = runner_utils.HiddenMajorLMHeadMethod.apply

    def spy(self, layer, xx, bias=None):
        seen.append(tuple(layer.weight.shape))
        return orig(self, layer, xx, bias)

    with patch.object(runner_utils.HiddenMajorLMHeadMethod, "apply", spy):
        d_logits = draft.compute_logits(x)
        d_top = draft.logits_processor.get_top_tokens(draft.lm_head, x)
        n_draft = len(seen)
        t_logits = target.compute_logits(x)
        t_top = target.logits_processor.get_top_tokens(target.lm_head, x)
    print(f"\nLMHS0_ENGAGE draft_calls={n_draft} shapes={seen} target_calls={len(seen) - n_draft}")
    assert n_draft == 2 and seen == [(hidden, head.weight.shape[0])] * 2
    assert len(seen) == n_draft  # the target head never takes the hidden-major method
    assert torch.equal(t_logits, t_logits0) and torch.equal(t_top, t_top0)
    # integer data: every sum is exact, so the hidden-major draft equals the stock head bitwise
    assert torch.equal(d_logits, t_logits0) and torch.equal(d_top, t_top0)


@pytest.mark.parametrize("vocab,tp", [(1000, 4), (1024, 8)], ids=["tp4_padded", "tp8"])
def test_sharded_draft_top_tokens_match_the_stock_head(rm, monkeypatch, vocab, tp):
    """Each TP rank's draft copy, through vLLM's get_top_tokens with the pair all-gather played back:
    the hidden-major draft's tokens are the stock head's and the full-vocabulary argmax."""
    hidden = 16
    full = _ints((vocab, hidden), seed=3)
    x = _ints((32, hidden), seed=4)
    ranks = [_pair(vocab, hidden, r, tp, full) for r in range(tp)]
    pads = [int(t.lm_head.shard_indices.num_org_vocab_padding) for t, _ in ranks]
    pairs = []

    def record(pair, dim):
        pairs.append(pair)
        return pair.repeat(1, tp)

    def tokens(which):
        pairs.clear()
        monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", record)
        for t, d in ranks:
            m = t if which == "target" else d
            m.logits_processor.get_top_tokens(m.lm_head, x)
        gathered = torch.cat(pairs, dim=-1)

        def replay(pair, dim):
            return gathered

        monkeypatch.setattr(lp_module, "tensor_model_parallel_all_gather", replay)
        t0, d0 = ranks[0]
        m = t0 if which == "target" else d0
        return m.logits_processor.get_top_tokens(m.lm_head, x)

    stock = tokens("draft")
    for t, d in ranks:
        runner_utils.arm_lm_head_hidden_major(t, d, draft=True, verify=False)
        assert tuple(d.lm_head.weight.shape) == (hidden, t.lm_head.weight.shape[0])
    hm = tokens("draft")
    tgt = tokens("target")
    want = (x.float() @ full.float().t()).argmax(dim=-1)
    print(f"\nLMHS0_SHARDED vocab={vocab} tp={tp} pads={pads} rows={x.shape[0]} "
          f"hm==stock {int((hm == stock).sum())}/{len(hm)} hm==full_argmax {int((hm == want).sum())}/{len(hm)}")
    assert any(pads) == (vocab == 1000)
    assert torch.equal(hm, stock) and torch.equal(tgt, stock) and torch.equal(hm, want)


# --------------------------------------------------------------------------- verify flag
def test_verify_and_draft_share_one_hidden_major_table(rm):
    vocab, hidden, tp, rank = 1000, 16, 4, 3
    target, draft = _pair(vocab, hidden, rank, tp, _ints((vocab, hidden)))
    th = target.lm_head
    w_vocab_major = th.weight.detach().clone()
    census = runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=True)
    print(f"\nLMHS0_ARM draft+verify census={census}")
    assert target.lm_head is th and draft.lm_head is th
    assert type(th.quant_method) is runner_utils.HiddenMajorLMHeadMethod
    assert torch.equal(th.weight, w_vocab_major.t())
    assert len(rm.calls) == 1  # one table for both heads: no copy
    rm.pin.assert_called_once_with()
    assert census["verify"].startswith("hidden-major in place")
    assert census["draft"].startswith("shares the hidden-major target head")


def test_verify_without_draft_is_refused_when_the_head_is_shared(rm):
    target, draft = _pair(1000, 16, 0, 4, _ints((1000, 16)))
    sha0 = _sha(target.lm_head.weight)
    with pytest.raises(ValueError, match="Set both, or neither"):
        runner_utils.arm_lm_head_hidden_major(target, draft, draft=False, verify=True)
    assert _sha(target.lm_head.weight) == sha0 and rm.calls == []


def test_verify_alone_without_a_draft_model(rm):
    head = _real_head(1000, 16, 2, 4)
    head.weight.data.copy_(_ints(tuple(head.weight.shape)))
    target = _Target(head, 1000)
    census = runner_utils.arm_lm_head_hidden_major(target, None, draft=False, verify=True)
    assert type(head.quant_method) is runner_utils.HiddenMajorLMHeadMethod
    assert tuple(head.weight.shape) == (16, 256) and "draft" not in census


# --------------------------------------------------------------------------- off, refusals
def test_flags_off_change_nothing(rm):
    target, draft = _pair(1000, 16, 1, 4, _ints((1000, 16)))
    th = target.lm_head
    w0, m0, sha0 = th.weight, th.quant_method, _sha(th.weight)
    assert runner_utils.arm_lm_head_hidden_major(target, draft, draft=False, verify=False) == {}
    assert draft.lm_head is th and th.weight is w0 and th.quant_method is m0 and _sha(th.weight) == sha0
    assert rm.calls == [] and not rm.pin.called


@pytest.mark.parametrize("case", ["quantized", "tied", "fp32_head"])
def test_draft_refusals_fail_stop(rm, case):
    target, draft = _pair(1000, 16, 1, 4, _ints((1000, 16)))
    if case == "quantized":
        target.lm_head.quant_method = object()
    if case == "tied":
        draft.model.embed_tokens = SimpleNamespace(weight=target.lm_head.weight)
    if case == "fp32_head":
        draft.logits_processor.head_dtype = torch.float32
    with pytest.raises(ValueError, match=DRAFT_FLAG):
        runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=False)
    assert draft.lm_head is target.lm_head and rm.calls == []


def test_draft_flag_without_a_draft_model_arms_nothing(rm):
    target, _ = _pair(1000, 16, 1, 4, _ints((1000, 16)))
    census = runner_utils.arm_lm_head_hidden_major(target, None, draft=True, verify=False)
    assert census["draft"] == "none (no draft model)" and rm.calls == [] and not rm.pin.called


def test_second_relayout_never_transposes_back(rm):
    target, draft = _pair(1000, 16, 1, 4, _ints((1000, 16)))
    runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=True)
    w1 = target.lm_head.weight
    assert runner_utils.relayout_lm_head(target) is False
    assert target.lm_head.weight is w1 and tuple(w1.shape) == (16, 256)
    with pytest.raises(ValueError, match=VERIFY_FLAG):
        runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=True)


def test_flag_defaults_and_build_time_read(monkeypatch):
    for name in (DRAFT_FLAG, VERIFY_FLAG):
        monkeypatch.delenv(name, raising=False)
        assert getattr(envs, name) is False
        monkeypatch.setenv(name, "1")
        assert getattr(envs, name) is True
        monkeypatch.setenv(name, "maybe")
        with pytest.raises(ValueError):
            getattr(envs, name)
        monkeypatch.delenv(name)


# --------------------------------------------------------------------------- runner wiring
RUNNER_PATH = os.path.join(PKG, "runner", "tpu_runner.py")


def _load_model_ast():
    tree = ast.parse(open(RUNNER_PATH).read())
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "TPUModelRunner":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "load_model":
                    return fn
    raise AssertionError("TPUModelRunner.load_model not found")


def _flag_if(fn):
    found = [n for n in fn.body if isinstance(n, ast.If) and fo._reads(n.test)]
    assert len(found) == 1, f"{len(found)} top-level LMH-HM ifs in load_model"
    return found[0], fn.body.index(found[0])


def test_runner_arms_after_the_drafter_on_the_last_stage_only():
    fn = _load_model_ast()
    node, i = _flag_if(fn)
    src = [ast.unparse(n) for n in fn.body]
    drafter = [j for j, s in enumerate(src) if "self._async_drafter.load_model(self.model)" in s
               and "_relayout_tables(" in s]
    moe = [j for j, s in enumerate(src) if "envs.FUSED_MOE_DECODE_KERNEL" in s.split("\n")[0]]
    assert drafter and moe and drafter[0] < i < moe[0], (drafter, i, moe)
    code = compile(ast.Module(body=[node], type_ignores=[]), RUNNER_PATH, "exec")
    rows = []
    for d in (False, True):
        for v in (False, True):
            for last in (False, True):
                for pooling in (False, True):
                    for asyncd in (False, True):
                        calls = []
                        ns = {
                            "envs": SimpleNamespace(**{DRAFT_FLAG: d, VERIFY_FLAG: v}),
                            "runner_utils": SimpleNamespace(
                                arm_lm_head_hidden_major=lambda *a, **k: calls.append((a, k))),
                            "self": SimpleNamespace(
                                _pp_is_last=last, is_pooling_model=pooling, _is_async_drafter=asyncd,
                                _async_drafter=SimpleNamespace(draft_model="DRAFT"), model="TARGET"),
                        }
                        exec(code, ns)
                        want = (d or v) and last and not pooling
                        assert bool(calls) == want, (d, v, last, pooling, asyncd)
                        if calls:
                            (args, kw), = calls
                            assert args == ("TARGET", "DRAFT" if asyncd else None), args
                            assert kw == {"draft": d, "verify": v}, kw
                        rows.append(int(bool(calls)))
    print(f"\nLMHS0_WIRING combos={len(rows)} armed={sum(rows)}")
    assert len(rows) == 32 and sum(rows) == 6


@pytest.mark.parametrize("name,rel", [("tpu_runner", "runner/tpu_runner.py"), ("utils", "runner/utils.py"),
                                      ("envs", "envs.py")])
def test_flags_off_is_the_base_ast_for_ast(name, rel):
    src = open(os.path.join(PKG, rel)).read()
    ref = open(os.path.join(HERE, f"ref_2e166829a_{name}.py.txt")).read()
    assert src != ref
    d = fo.check(src, ref)
    print(f"\nLMHS0_FLAGOFF {name} {'IDENTICAL' if not d else d}")
    assert not d, d


def test_flag_off_check_bites():
    src = open(RUNNER_PATH).read()
    ref = open(os.path.join(HERE, "ref_2e166829a_tpu_runner.py.txt")).read()
    guard = "            envs.VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT or envs.VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY\n"
    assert src.count(guard) == 1
    assert fo.check(src.replace(guard, "            True\n"), ref) == ["differs: TPUModelRunner.load_model"]
    usrc = open(os.path.join(PKG, "runner", "utils.py")).read()
    uref = open(os.path.join(HERE, "ref_2e166829a_utils.py.txt")).read()
    assert fo.check(usrc + "\n\ndef _lmhs0_unlisted():\n    return 1\n", uref) == ["only in new: _lmhs0_unlisted"]


# --------------------------------------------------------------------------- Qwen3.8 census
@pytest.mark.parametrize("rank", [0, 7])
def test_census_at_qwen38_tp8_dims(rm, rank):
    """Qwen3.8 Flash: hidden 2560, vocab 248,448 at TP8 -> one rank's head is [31056, 2560] bf16
    (159,006,720 B). Draft-only arm: the draft gets a [2560, 31056] row-major copy of the same bytes."""
    vocab, hidden, tp = 248448, 2560, 8
    head = _real_head(vocab, hidden, rank, tp)
    g = torch.Generator().manual_seed(rank)
    head.weight.data.copy_((torch.randn(head.weight.shape, generator=g) * 0.02).to(torch.bfloat16))
    target = _Target(head, vocab)
    draft = _Draft(vocab, hidden, rank, tp)
    maybe_share_lm_head(draft, target, False, True, materialize_if_mismatched=False)
    sha_t = _sha(head.weight)
    census = runner_utils.arm_lm_head_hidden_major(target, draft, draft=True, verify=False)
    dh = draft.lm_head
    si = head.shard_indices
    nbytes = dh.weight.numel() * dh.weight.element_size()
    print(f"\nLMHS0_CENSUS rank={rank}/{tp} shard=[{si.org_vocab_start_index},{si.org_vocab_end_index}) "
          f"pad={si.num_org_vocab_padding} target={tuple(head.weight.shape)} stride={tuple(head.weight.stride())} "
          f"draft={tuple(dh.weight.shape)} stride={tuple(dh.weight.stride())} {dh.weight.dtype} bytes={nbytes} "
          f"target_sha_unchanged={_sha(head.weight) == sha_t} census={census}")
    assert tuple(head.weight.shape) == (31056, hidden) and si.num_org_vocab_padding == 0
    assert tuple(dh.weight.shape) == (hidden, 31056) and tuple(dh.weight.stride()) == (31056, 1)
    assert nbytes == 159006720 and dh.weight.dtype == torch.bfloat16
    assert torch.equal(dh.weight, head.weight.t()) and _sha(head.weight) == sha_t
    assert rm.calls == [((hidden, 31056), True, (31056, 1), torch.bfloat16)]
