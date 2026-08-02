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
"""Metric and filter helpers for the humaneval_plus_tpu / mbpp_plus_tpu tasks.

Metric: lm-eval 0.5.0.dev1's bundled humaneval pass_at_k forwards
per-document references/predictions to HF evaluate's code_eval without
normalizing their shapes ("Got a string but expected a list"), and both
bundled code metrics return a bare float, which crashes the scorer's
reduce() step (it expects {metric: [per-repeat values]}).

Filters: both tasks prompt via the chat endpoint (evalplus-style), so the
model replies with prose + a fenced code block rather than a raw
continuation; these filters extract the code block, standing in for
evalplus's response sanitizer.
"""
import re
from typing import Union

import evaluate as hf_evaluate

code_eval = hf_evaluate.load("code_eval")

# The EvalPlus "plus" test suites run hundreds of extra cases per problem;
# HF code_eval's 3s default per-candidate timeout is too tight for them.
CODE_EVAL_TIMEOUT_S = 15.0


def pass_at_1(references: Union[str, list[str]],
              predictions: Union[list[str], list[list[str]]]) -> list[float]:
    if isinstance(references, str):
        references = [references]
    if predictions and isinstance(predictions[0], str):
        predictions = [list(predictions)]
    score = code_eval.compute(
        references=references,
        predictions=predictions,
        k=[1],
        timeout=CODE_EVAL_TIMEOUT_S,
    )[0]["pass@1"]
    return [float(score)]


def _extract_code(text: str) -> str:
    """Return the code inside the response's markdown fence.

    Prefers the first fenced block that defines a function; tolerates a
    missing closing fence (max_gen_toks truncation). Falls back to the raw
    text when there is no fence at all.
    """
    blocks = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)(?:\n```|$)", text,
                        re.DOTALL)
    if not blocks:
        return text
    for block in blocks:
        if "def " in block:
            return block
    return blocks[0]


def extract_code(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[_extract_code(r) for r in resp] for resp in resps]


def extract_code_instruct(resps: list[list[str]],
                          docs: list[dict]) -> list[list[str]]:
    """humaneval variant: the tests call doc["entry_point"], so if the model
    answered with a bare body (continuation-style) instead of restating the
    full function, prepend the original prompt to complete it."""
    out = []
    for resp, doc in zip(resps, docs):
        cleaned = []
        for r in resp:
            code = _extract_code(r)
            if f"def {doc['entry_point']}" not in code:
                code = doc["prompt"] + code
            cleaned.append(code)
        out.append(cleaned)
    return out
