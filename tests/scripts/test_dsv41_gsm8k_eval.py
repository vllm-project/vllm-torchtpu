# SPDX-License-Identifier: Apache-2.0
"""Guard quality reports against malformed answers and incomplete requests."""

import json

import pytest

from scripts.dsv41_bench.server_steps.eval_gsm8k import (
    evaluate,
    extract_number,
    load_sample,
    summarize_round,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "text,expected",
    [
        ("work\n#### 1,200.00", "1.2E+3"),
        ("#### 3\ncorrection\n#### -12.50", "-12.5"),
        ("#### +007", "7"),
        ("#### -0.0", "0"),
        ("The answer is 42", None),
        ("#### 12/13", None),
        ("#### 12 or 13", None),
        ("#### 1,2", None),
        ("#### NaN", None),
        ("####", None),
    ],
)
def test_numeric_answer_does_not_accept_partial_expressions(text, expected):
    assert extract_number(text) == expected


def test_frozen_sample_preserves_dataset_identity_and_source_indices(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"question": f"Question {i}", "answer": f"#### {i}"})
            for i in range(20)
        )
    )
    sample = load_sample(dataset, 8, 0)
    assert sample == load_sample(dataset, 8, 0)
    assert len({case["index"] for case in sample["cases"]}) == 8
    for case in sample["cases"]:
        assert case["question"] == f"Question {case['index']}"
        assert case["reference"] == str(case["index"])
    dataset.write_text(dataset.read_text() + "\n")
    assert load_sample(dataset, 8, 0)["dataset_sha256"] != sample["dataset_sha256"]


def test_missing_reference_cannot_match_missing_prediction(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"question": "question", "answer": "missing"}))
    with pytest.raises(ValueError, match="Invalid GSM8K reference"):
        load_sample(dataset, 1, 0)


def test_failed_and_truncated_requests_remain_in_the_denominator():
    summary = summarize_round(
        [
            {"index": 1, "predicted": "2", "correct": True, "finish_reason": "stop"},
            {
                "index": 2,
                "predicted": None,
                "correct": False,
                "finish_reason": "length",
            },
            {"index": 3, "error": "HTTPError: server failed"},
        ]
    )
    assert summary == {
        "samples": 3,
        "completed": 2,
        "correct": 1,
        "missing_answers": 1,
        "truncated": 1,
        "errors": [3],
    }


def test_responses_and_failures_are_persisted_across_repeats(tmp_path, monkeypatch):
    from scripts.dsv41_bench.server_steps import eval_gsm8k

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    monkeypatch.setattr(
        eval_gsm8k.requests,
        "get",
        lambda *args, **kwargs: Response({"data": [{"id": "test-model"}]}),
    )
    bodies = []

    def post(url, *, json, timeout):
        bodies.append(json)
        if json["messages"][0]["content"].startswith("fail"):
            raise eval_gsm8k.requests.HTTPError("HTTP 500")
        return Response(
            {"choices": [{"message": {"content": "#### 2"}, "finish_reason": "stop"}]}
        )

    monkeypatch.setattr(eval_gsm8k.requests, "post", post)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"question": question, "answer": "#### 2"})
            for question in ["ok", "fail"]
        )
    )
    out = tmp_path / "result"
    report = evaluate(dataset, out, samples=2, concurrency=1, repeats=2)
    assert report["completed"] is False
    assert len(bodies) == 4
    assert bodies[:2] == bodies[2:]
    for number, summary in enumerate(report["rounds"], 1):
        assert summary["samples"] == 2
        assert summary["completed"] == summary["correct"] == 1
        rows = (out / f"round-{number}-responses.jsonl").read_text().splitlines()
        assert len(rows) == 2
        assert any("HTTP 500" in row for row in rows)
    assert json.loads((out / "summary.json").read_text()) == report
