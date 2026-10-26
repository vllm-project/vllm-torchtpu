# Serving exports with the Rust runtime

`bin/rvllm.bin` serves a model with rvllm, a native Rust runtime that runs
the artifacts written by the [AOT exporter](../export/) through PJRT, with
no Python in the serving process. It takes the `vllm serve` command line:

```bash
export RVLLM_BINARY=/path/to/rvllm
src/vllm_torchtpu/experimental/rust/bin/rvllm.bin serve Qwen/Qwen3-0.6B \
  --max-model-len 2048 --port 8000
```

On first use, the script exports the model and then serves it:

1. It exports with `python -m vllm_torchtpu.experimental.export <model>
   --export-mode all` and the model arguments.
2. It serves the artifact with `rvllm serve <artifact>` and the deployment
   flags.

The artifact is cached by model, model arguments and exporter source, so the
next start skips the export. The server speaks the OpenAI-compatible API on
`/v1/completions` and `/v1/chat/completions`.

## Arguments

vLLM's own `vllm serve` parser splits the command line
([`launcher_args.py`](launcher_args.py)), so option names, short forms and
values are exactly vLLM's:

- **Engine arguments go to the exporter.** These are the fields of vLLM's
  `AsyncEngineArgs`, such as `--max-model-len`, `--tensor-parallel-size`,
  `--max-num-seqs`, `--quantization` and `--speculative-config`. The exporter
  records in the artifact what the runtime needs.
- **Some options go to rvllm.** `--host`, `--port`, `--served-model-name` and
  `--[no-]async-scheduling` set the server and its runtime behavior.
  `--load-format` goes to both rvllm and the exporter.
- **rvllm's own deployment flags go to rvllm.** `vllm serve` has no such flags:
  `--role`, `--kv-port`, `--kv-listen`, `--kv-advertise`, `--kv-producers`,
  `--max-queued-requests`, `--shutdown-timeout` and
  `--structured-decoding-mode`.
- **No option is silently ignored.**
  - A frontend option that only affects vLLM's own logs, docs or HTTP limits
    is dropped with a warning: `--uvicorn-log-level`, `--max-log-len`,
    `--disable-fastapi-docs`, `--api-server-count`, and a few more.
  - Every other option rvllm does not implement fails before anything runs,
    for example `--api-key`, `--chat-template`, `--enable-auto-tool-choice`
    and `--ssl-*`. So does an option `vllm serve` does not know.
  - Disaggregated serving uses rvllm's flags: `--kv-transfer-config` fails
    and says which flags to use.

The script prints the export command and the `rvllm serve` command before it
runs them.

`rvllm.bin bench ...` runs `vllm bench ...`, which benchmarks any
OpenAI-compatible server and needs no rvllm binary. `proxy` and `--version`
run the rvllm binary directly:

```bash
rvllm.bin bench serve --model Qwen/Qwen3-0.6B --max-concurrency 8 ...
rvllm.bin proxy --port 8000 --prefiller-hosts p --prefiller-ports 8100 \
  --decoder-hosts d --decoder-ports 8200
```

## The rvllm binary

The binary is not part of this repository. Set `RVLLM_BINARY` to its path.

Other variables:

| Variable | Meaning |
|---|---|
| `RVLLM_CACHE_DIR` | Where exports are cached. The default is `~/.cache/rvllm`. |
| `RVLLM_ARTIFACT_DIR` | Serve this exported artifact instead of exporting. |
| `PYTHON` | The interpreter that runs the exporter. The default is `python3`. |
