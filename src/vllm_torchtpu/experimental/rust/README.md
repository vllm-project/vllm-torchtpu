# Serving exports with the Rust runtime

`python -m vllm_torchtpu.experimental.rust` serves a model with rvllm, a native
Rust runtime that runs the artifacts written by the [AOT exporter](../export/)
through PJRT, with no Python in the serving process. It takes the `vllm serve` command line:

```bash
python -m vllm_torchtpu.experimental.rust serve Qwen/Qwen3-0.6B \
  --max-model-len 2048 --port 8000 --rvllm-binary /path/to/rvllm
```

`--rvllm-binary` is the launcher's only option of its own (see
[The rvllm binary](#the-rvllm-binary)); everything else is the `vllm serve`
command line.

The launcher ([`launcher.py`](launcher.py)) runs the export as a child process
and then replaces itself with the rvllm binary (`exec`), so the server is the
only process left, and it gets signals and the exit status directly.

On first use, the launcher exports the model and then serves it:

1. It exports with `python -m vllm_torchtpu.experimental.export <model>
   --export-mode all` and the model arguments.
2. It serves the artifact with `rvllm serve <artifact>` and the deployment
   flags.

The artifact is cached by model, model arguments and exporter source, so the
next start skips the export. The server speaks the OpenAI-compatible API on
`/v1/completions` and `/v1/chat/completions`.

## Arguments

vLLM's own `vllm serve` parser splits the command line
([`launcher.py`](launcher.py)), so option names, short forms and
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

The launcher prints the export command and the `rvllm serve` command before it
runs them.

`bench ...` runs `vllm bench ...`, which benchmarks any OpenAI-compatible
server and needs no rvllm binary. `proxy` and `--version` run the rvllm binary
directly:

```bash
python -m vllm_torchtpu.experimental.rust bench serve --model Qwen/Qwen3-0.6B --max-concurrency 8 ...
python -m vllm_torchtpu.experimental.rust proxy --port 8000 --prefiller-hosts p --prefiller-ports 8100 \
  --decoder-hosts d --decoder-ports 8200
```

## The rvllm binary

The binary is not part of this repository. `--rvllm-binary` points the launcher
to it, in one of two ways:

- **A local `rvllm` binary**, used as is.
- **A location holding `rvllm` and its `metadata.json`**, which has a `sha256`
  entry. It can be `gs://...`, `https://...` or a local directory. The binary is
  checked against the sha256 and cached.

A download location holds two files: the binary, named `rvllm`, and a
`metadata.json` that describes it. This `metadata.json` is not the one in an
exported artifact. The launcher reads one field, `sha256`, which is required: the
binary's SHA-256 in lowercase hex. It ignores every other field; the release
build of rvllm writes them to say what the binary was built from:

```json
{
  "sha256": "732fca1514422763ae796096b72fb60ef6ed243911b91cc44efaef6fcc97b8ee",
  "size_bytes": 13560920,
  "git_branch": "main",
  "git_commit": "9e68e2178",
  "built_at": "2026-10-26T13:20:10-07:00",
  "builder_image": "<image the binary was compiled in>",
  "date_tag": "20261026",
  "elf_interpreter": "/lib64/ld-linux-x86-64.so.2"
}
```

Downloading from `gs://` uses `gcloud storage` with the caller's credentials.
Downloaded binaries and exports are cached under `~/.cache/rvllm`.

`--rvllm-binary` can appear anywhere on the command line, as
`--rvllm-binary VALUE` or `--rvllm-binary=VALUE`. `vllm serve` has no `--rvllm-*`
options, and any other one fails.
