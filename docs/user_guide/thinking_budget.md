# Thinking Token Budget

Reasoning models such as Qwen3, DeepSeek-R1, and Gemma 4 write a chain of thought between a start marker and an end marker (`<think>` and `</think>` for Qwen3) before they answer. The per-request `thinking_token_budget` sampling parameter from upstream vLLM caps how many tokens a request may spend inside that thinking block. Once the budget is spent, the TPU runner forces the reasoning end marker, so the model stops thinking and writes its answer instead of being cut off by `max_tokens` mid-thought.

## Enabling the budget

Serve the model with a reasoning parser, which supplies the start and end markers:

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3
```

To set the marker strings yourself, pass `--reasoning-config` with `reasoning_start_str` and `reasoning_end_str` (with or without a parser). `reasoning_end_str` is the string forced when the budget runs out; it defaults to the parser's end marker and may add a transition phrase before that marker or a suffix after it:

```bash
vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "Time to answer.</think>"}'
```

Then set the budget per request, in `SamplingParams` for offline inference:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-8B", reasoning_parser="qwen3")
params = SamplingParams(max_tokens=1024, thinking_token_budget=256)
outputs = llm.chat([{"role": "user", "content": "What is 17 * 23?"}], params)
```

or in the body of an OpenAI-compatible request (`extra_body` with the `openai` client):

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 1024,
  "thinking_token_budget": 256
}'
```

| `thinking_token_budget` | Behavior |
| :--- | :--- |
| unset (`None`) or `-1` | No limit (the default). |
| `0` | The thinking block closes as soon as it opens. |
| `N > 0` | At most `N` tokens inside the thinking block, then the end marker is forced. |

vLLM rejects other negative values and non-integers when it validates the request. Thinking tokens already in the prompt count toward the budget, so a prompt that ends inside an open thinking block starts with part of the budget spent. If the model opens another thinking block later in the same response, that block gets a budget of its own. When the server has no reasoning markers (neither flag is set, or the parser defines no start and end strings), requests with a budget run unconstrained and the runner logs a warning once.

## How thinking closes

When the budget is spent, the runner forces the tokens of `reasoning_end_str` in order. Thinking closes as soon as the parser's own end marker is complete, whether the model writes it on its own or as part of the forced string:

- With a `reasoning_end_str` that adds a transition phrase before the marker, the model can still close thinking early by writing the marker itself.
- With a `reasoning_end_str` that adds a suffix after the marker, thinking counts as closed once the marker is out, even if the suffix is cut short.
- If the budget runs out while the model is in the middle of writing a multi-token end marker, the runner completes that marker instead of starting the forced string.

Reported logprobs come from the logits before the thinking-budget forcing, so a forced token carries the probability the model itself gave it.

## Structured outputs and `min_tokens`

The runner does not override other constraints. If structured outputs or `min_tokens` has masked the token it would force on a step (for example, the end marker is a stop token and the request is still below `min_tokens`), that step samples as usual, and forcing resumes on the first step where the token is allowed. The thinking block can then run past the budget.

## Differences from upstream vLLM

1. **Closing on the parser's end marker.** Upstream vLLM closes thinking only on the full `reasoning_end_str` token sequence. The TPU runner also closes it on the parser's own end marker, so a `reasoning_end_str` with a transition phrase or a suffix still closes thinking when the model writes the marker itself.
2. **Constraint masks.** Upstream vLLM sets the forced token's logit to `1e9`, overriding structured-output and `min_tokens` masks. The TPU runner defers forcing until those masks allow the token.

## Speculative decoding and `TPU_GREEDY_TOP_TOKENS`

With [speculative decoding](speculative_decoding.md), the runner enforces the budget in the verification step: it forces the target and bonus positions that reach the budget, and rejects any draft token that differs from the forced token. The cap is the same as without speculation; it has been validated with N-gram and MTP drafting.

With `TPU_GREEDY_TOP_TOKENS`, greedy steps keep taking the vocabulary-parallel argmax when requests set a budget (such steps already exclude logprobs, structured outputs, and requests below `min_tokens`): the runner rewrites the selected token IDs instead of the logits. The precompile step then also warms these token-ID variants.

## Cost

- **Without a reasoning parser or `--reasoning-config`**, nothing is compiled or allocated and sampling is unchanged.
- **With a parser, while no running request sets a budget**, sampling does no extra per-step work. In a mixed batch, requests without a budget are unaffected.
- **Startup** precompiles the forcing kernels for every request bucket, plus every verify bucket with speculative decoding. On TPU v7x (`TP=8`), this took 3.7 s for plain decode (Qwen3-8B, 128 sequences), 15.7 s with MTP `K=2` (Qwen3.5-27B, 32 sequences), and 29.9 s with N-gram `K=3` (Qwen3-8B, 128 sequences). With `TPU_GREEDY_TOP_TOKENS`, which also warms the top-token variants, it took 5.6 s, 17.6 s, and 33.4 s. KV-cache capacity changed by at most one block.
- **Decode** cost, measured on Qwen3-8B (TPU v7x, `TP=8`, greedy, `TPU_GREEDY_TOP_TOKENS=0`) with a budget large enough that no request was forced:

| Mode | Sequences | TPOT, no budget | TPOT, every request budgeted |
| :--- | :--- | :--- | :--- |
| Plain decode | 128 | 12.85 ms | 13.41 ms |
| Plain decode | 512 | 29.94 ms | 29.70 ms |
| N-gram `K=3` | 128 | 17.06 ms | 17.14 ms |
| N-gram `K=3` | 512 | 47.93 ms | 48.67 ms |
