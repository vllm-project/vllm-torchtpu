import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = (REPO_ROOT / "examples" / "disagg" /
              "smoke_qwen35_p4d2_v2_prefix_cache_correctness.py")
DIVERGENCE_SMOKE_PATH = (REPO_ROOT / "examples" / "disagg" /
                         "smoke_qwen35_p4d2_prefix_cache_e2e_divergence.py")
CI_SCRIPT = (REPO_ROOT / "scripts" / "vllm" / "integration" /
             "run_qwen35_p4d2_disagg_correctness.sh")

P4D2_LAUNCHER = (REPO_ROOT / "examples" / "disagg" /
                 "launch_qwen35_p4d2_v2_baseline.sh")


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("qwen35_p4d2_smoke",
                                                  SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_divergence_smoke_module():
    spec = importlib.util.spec_from_file_location("qwen35_p4d2_divergence",
                                                  DIVERGENCE_SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefix_cache_divergence_smoke_detects_logprob_drift():
    smoke = _load_divergence_smoke_module()

    diff = smoke.output_diff(
        {
            "text": "5",
            "logprobs": [{
                "token": "5",
                "logprob": 0.0,
            }],
        },
        {
            "text": "5",
            "logprobs": [{
                "token": "5",
                "logprob": -0.375,
            }],
        },
        logprob_atol=1e-5,
    )

    assert diff["text_equal"]
    assert diff["tokens_equal"]
    assert diff["has_logprobs"]
    assert diff["max_logprob_absdiff"] == 0.375
    assert not diff["logprob_equal"]
    assert diff["first_divergent_index"] is None


def test_prefix_cache_divergence_smoke_baseline_all_equal():
    smoke = _load_divergence_smoke_module()

    logprobs = [{"token": "5", "logprob": 0.0}]
    diff = smoke.output_diff(
        {
            "text": "5",
            "logprobs": logprobs
        },
        {
            "text": "5",
            "logprobs": logprobs
        },
        logprob_atol=1e-5,
    )

    assert diff["text_equal"]
    assert diff["tokens_equal"]
    assert diff["has_logprobs"]
    assert diff["max_logprob_absdiff"] == 0.0
    assert diff["logprob_equal"]
    assert diff["first_divergent_index"] is None


def test_prefix_cache_divergence_smoke_reports_first_mismatched_token():
    smoke = _load_divergence_smoke_module()

    cold = {
        "text":
        "abc",
        "logprobs": [
            {
                "token": "a",
                "logprob": -0.1
            },
            {
                "token": "b",
                "logprob": -0.2
            },
            {
                "token": "c",
                "logprob": -0.3
            },
        ],
    }
    warm = {
        "text":
        "abz",
        "logprobs": [
            {
                "token": "a",
                "logprob": -0.10001
            },
            {
                "token": "b",
                "logprob": -0.2
            },
            {
                "token": "z",
                "logprob": -9.0
            },
        ],
    }
    diff = smoke.output_diff(cold, warm, logprob_atol=1e-3)

    assert not diff["text_equal"]
    assert not diff["tokens_equal"]
    assert diff["has_logprobs"]
    # matching-position max is < atol...
    assert diff["max_logprob_absdiff"] < 1e-3
    assert diff["logprob_equal"]
    # ...but first_divergent_* surfaces the real magnitude at the divergence
    # point.
    assert diff["first_divergent_index"] == 2
    assert abs(diff["first_divergent_logprob_absdiff"] - 8.7) < 1e-9


def test_prefix_cache_divergence_smoke_handles_empty_logprobs():
    smoke = _load_divergence_smoke_module()

    diff = smoke.output_diff(
        {
            "text": "",
            "logprobs": []
        },
        {
            "text": "",
            "logprobs": []
        },
        logprob_atol=1e-3,
    )

    assert diff["text_equal"]
    assert diff["tokens_equal"]
    assert not diff["has_logprobs"]
    assert diff["max_logprob_absdiff"] == float("inf")
    assert not diff["logprob_equal"]


def test_mixed_query_suite_interleaves_short_and_long_prompts(monkeypatch):
    smoke = _load_smoke_module()
    kinds = []

    def fake_chat(_url, _model, prompt, _max_tokens, _timeout):
        if "Query key: alpha" in prompt:
            kinds.append("long")
            return {
                "status": 200,
                "finish": "stop",
                "text": "ALPHA-314",
                "cached_tokens": 128,
            }
        if "Query key: bravo" in prompt:
            kinds.append("long")
            return {
                "status": 200,
                "finish": "stop",
                "text": "BRAVO-271",
                "cached_tokens": 128,
            }
        if "Query key: charlie" in prompt:
            kinds.append("long")
            return {
                "status": 200,
                "finish": "stop",
                "text": "CHARLIE-159",
                "cached_tokens": 128,
            }
        if "7+8" in prompt:
            kinds.append("short")
            return {"status": 200, "finish": "stop", "text": "15"}
        if "capital of France" in prompt:
            kinds.append("short")
            return {"status": 200, "finish": "stop", "text": "Paris"}
        if "larger, 12 or 21" in prompt:
            kinds.append("short")
            return {"status": 200, "finish": "stop", "text": "21"}
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(smoke, "chat", fake_chat)

    requests, failures = smoke.run_mixed_query_correctness(
        "http://proxy/v1/chat/completions",
        "Qwen3.5-35B-A3B-FP8",
        "unit-test",
        line_count=4,
        rounds=1,
        max_tokens=16,
        timeout=1,
    )

    assert requests == 6
    assert failures == 0
    assert kinds == ["short", "long", "short", "long", "short", "long"]


def test_mixed_query_suite_fails_on_wrong_long_answer(monkeypatch):
    smoke = _load_smoke_module()

    def fake_chat(_url, _model, prompt, _max_tokens, _timeout):
        if "Query key:" in prompt:
            return {
                "status": 200,
                "finish": "stop",
                "text": "WRONG",
                "cached_tokens": 128,
            }
        return {"status": 200, "finish": "stop", "text": "15"}

    monkeypatch.setattr(smoke, "chat", fake_chat)

    requests, failures = smoke.run_mixed_query_correctness(
        "http://proxy/v1/chat/completions",
        "Qwen3.5-35B-A3B-FP8",
        "unit-test",
        line_count=4,
        rounds=1,
        max_tokens=16,
        timeout=1,
    )

    assert requests == 6
    assert failures > 0


def test_main_skips_concurrent_mixed_when_concurrency_is_one(monkeypatch):
    smoke = _load_smoke_module()
    seen = {}
    args = SimpleNamespace(
        host="127.0.0.1",
        port="8000",
        model="Qwen3.5-35B-A3B-FP8",
        run_dir="",
        timeout=1,
        quick_probe_only=False,
        skip_short_qa=True,
        short_repeat_lines=1,
        short_repeat_count=1,
        long_repeat_lines=1,
        long_repeat_count=1,
        long_shared_lines=1,
        long_shared_rounds=1,
        skip_mixed_query=False,
        mixed_lines=72,
        mixed_rounds=1,
        concurrent_requests=1,
        expect_pcp_source=False,
    )

    def fake_mixed(_url, _model, namespace, _line_count, _rounds, _max_tokens,
                   _timeout):
        seen["mixed"] = namespace
        return 0, 0

    def fake_concurrent(_url, _model, namespace, _line_count, _rounds,
                        _concurrency, _max_tokens, _timeout):
        seen["concurrent"] = namespace
        return 0, 0

    monkeypatch.setattr(smoke, "parse_args", lambda: args)
    monkeypatch.setattr(smoke.time, "time", lambda: 1234)
    monkeypatch.setattr(smoke, "log_offsets", lambda _run_dir: {})
    monkeypatch.setattr(smoke, "run_repeat_consistency", lambda *args: (0, 0))
    monkeypatch.setattr(smoke, "run_long_shared_prefix_cross_query",
                        lambda *args: (0, 0))
    monkeypatch.setattr(smoke, "run_mixed_query_correctness", fake_mixed)
    monkeypatch.setattr(smoke, "run_concurrent_mixed_query_correctness",
                        fake_concurrent)
    monkeypatch.setattr(
        smoke,
        "check_planner_logs",
        lambda _run_dir, _offsets, *, expect_pcp_source=False: 0)

    assert smoke.main() == 0
    assert seen["mixed"] == "p4d2-v2-correctness-1234-mixed"
    assert "concurrent" not in seen


def test_planner_log_check_accepts_current_positive_total_ops(
        tmp_path, capsys):
    smoke = _load_smoke_module()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    decode_log = log_dir / "decode.log"
    decode_log.write_text(
        "\n".join([
            ("TPUConnectorV2 logical pull meta built | req_id=1 | "
             "d_tp_rank=0 | "
             "p_ranks=(0, 1) | fa_heads_by_rank={0: (0,)} | "
             "mamba_state0_q_key_heads_by_rank={0: (0,)} | "
             "mamba_state0_k_key_heads_by_rank={0: (0,)} | "
             "mamba_state0_v_value_heads_by_rank={0: (0,)} | "
             "mamba_state1_value_heads_by_rank={0: (0,)}"),
            ("TPUConnectorV2 logical pull meta built | req_id=1 | "
             "d_tp_rank=1 | "
             "p_ranks=(2, 3) | fa_heads_by_rank={2: (1,)} | "
             "mamba_state0_q_key_heads_by_rank={2: (1,)} | "
             "mamba_state0_k_key_heads_by_rank={2: (1,)} | "
             "mamba_state0_v_value_heads_by_rank={2: (1,)} | "
             "mamba_state1_value_heads_by_rank={2: (1,)}"),
            ("TPUConnectorV2 physical lowering summary | req_id=1 | "
             "d_rank=0 | "
             "d_tp_rank=0 | p_ranks=(0, 1) | total_ops=4330 | "
             "ops_by_p_rank={0: 2170, 1: 2160} | "
             "mamba_state0_q_ops_by_key_head={0: 30} | "
             "mamba_state0_k_ops_by_key_head={0: 30} | "
             "mamba_state0_v_ops_by_value_head={0: 30} | "
             "mamba_state1_ops_by_value_head={0: 30}"),
            ("TPUConnectorV2 physical lowering summary | req_id=1 | "
             "d_rank=1 | "
             "d_tp_rank=1 | p_ranks=(2, 3) | total_ops=4330 | "
             "ops_by_p_rank={2: 2170, 3: 2160} | "
             "mamba_state0_q_ops_by_key_head={1: 30} | "
             "mamba_state0_k_ops_by_key_head={1: 30} | "
             "mamba_state0_v_ops_by_value_head={1: 30} | "
             "mamba_state1_ops_by_value_head={1: 30}"),
            "TPUConnectorV2Worker(0) rank0 <-- START registered",
            "TPUConnectorV2Scheduler --> START planned",
            "TPUConnectorV2Scheduler --> START dispatched",
            "TPUConnectorV2Worker(0) tp_rank=0 <-- local START",
            "TPUConnectorV2Worker(0) tp_rank=1 <-- local START",
            "TPUConnectorV2 strided lifecycle send START",
            "TPUConnectorV2 strided lifecycle START ack OK",
            "TPUConnectorV2Worker(0) rank0 <-- recv START",
            "TPUConnectorV2Worker(0) rank0 --> accept START",
            "TPUConnectorV2Worker(0) tp_rank=0 --> local END queued",
            "TPUConnectorV2Worker(0) tp_rank=1 --> local END queued",
            "TPUConnectorV2Worker(0) --> END completion meta",
            "TPUConnectorV2Scheduler <-- recv END completion",
            "TPUConnectorV2 strided lifecycle send END",
            "TPUConnectorV2 strided lifecycle END ack OK",
            "TPUConnectorV2Worker(0) rank0 <-- recv END",
            "TPUConnectorV2Worker(0) rank0 --> accept END",
            "TPUConnectorV2Scheduler --> recv END complete",
        ]),
        encoding="utf-8",
    )

    failures = smoke.check_planner_logs(str(tmp_path), {decode_log: 0})

    output = capsys.readouterr().out
    assert failures == 0
    assert "PLANNER_LOG_TOTAL_OPS_BY_D_TP_RANK" in output
    assert "PLANNER_LOG_CHECK_OK 1" in output


def test_ci_wrapper_uses_hugging_face_model_and_existing_p4d2_launcher():
    text = CI_SCRIPT.read_text(encoding="utf-8")
    launcher_text = P4D2_LAUNCHER.read_text(encoding="utf-8")
    removed_alias_env = "_".join(
        ("TPU", "VLLM", "KV", "CACHE", "ALIAS", "FALLBACK"))

    assert "Qwen/Qwen3.5-35B-A3B-FP8" in text
    assert "launch_qwen35_p4d2_v2_baseline.sh" in text
    assert "smoke_qwen35_p4d2_v2_prefix_cache_correctness.py" in text
    assert "smoke_qwen35_p4d2_prefix_cache_e2e_divergence.py" in text
    assert "prefix_cache_e2e_divergence.log" in text
    assert "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL" in text
    assert "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL" in launcher_text
    assert "--kv-cache-dtype fp8" in launcher_text
    assert removed_alias_env not in text
    assert removed_alias_env not in launcher_text
    assert 'ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"' in text
    assert "export ASYNC_SCHEDULING" in text
    assert "gsutil" not in text
    assert "gcloud storage cp" not in text


def test_p4d2_ci_runs_launcher_default_single_client_correctness():
    text = CI_SCRIPT.read_text(encoding="utf-8")
    launcher_text = P4D2_LAUNCHER.read_text(encoding="utf-8")

    assert 'MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"' in launcher_text
    assert "MAX_NUM_SEQS" not in text
    assert ('export P4D2_CONCURRENT_REQUESTS="${P4D2_CONCURRENT_REQUESTS:-1}"'
            in text)


def test_qwen35_p4d2_launcher_uses_automatic_block_size_by_default():
    text = P4D2_LAUNCHER.read_text(encoding="utf-8")

    assert 'BLOCK_SIZE="${BLOCK_SIZE:-}"' in text
    assert 'if [[ -n "${BLOCK_SIZE}" ]]; then' in text
    assert 'common_args+=(--block-size "${BLOCK_SIZE}")' in text
    assert 'block_size_label="${BLOCK_SIZE:-auto}"' in text
    assert 'BLOCK_SIZE=${block_size_label}' in text


def test_qwen35_p4d2_launcher_enables_batched_rpa_backend():
    text = P4D2_LAUNCHER.read_text(encoding="utf-8")

    assert 'ATTENTION_BACKEND="${ATTENTION_BACKEND:-CUSTOM}"' in text
    assert '--attention-backend "${ATTENTION_BACKEND}"' in text
    assert "USE_BATCHED_RPA_KERNEL=1" in text


def test_qwen35_p4d2_launcher_uses_v3_gdn_by_default():
    ci_text = CI_SCRIPT.read_text(encoding="utf-8")
    launcher_text = P4D2_LAUNCHER.read_text(encoding="utf-8")

    assert (
        'RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"'
        in ci_text)
    assert (
        'RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL:-chunked_kernel_v3_pd}"'
        in launcher_text)
    assert ('RAGGED_GATED_DELTA_RULE_IMPL="${RAGGED_GATED_DELTA_RULE_IMPL}";'
            in launcher_text)
