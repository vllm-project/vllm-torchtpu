# OLMo 3.5 tests (`tests/models/olmo35`)

Unit tests for `src/vllm_torchtpu/models/olmo35/`. The dummy-weight
end-to-end test is `tests/entrypoints/llm/test_generate_olmo35.py`; it starts
an engine in a spawned process, so it is marked `spawns_tpu_workers` to run
before the in-process TPU tests (see `tests/conftest.py`).

- **Owners**: `@sierraisland`, `@wenxindongwork`
