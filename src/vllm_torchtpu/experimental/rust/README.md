# Serving exports with the Rust runtime

`bin/rvllm.bin` serves a model with rvllm, a native Rust runtime that runs
the artifacts written by the [AOT exporter](../export/) through PJRT, with
no Python in the serving process. It takes the `vllm serve` command line:

```bash
export RVLLM_BINARY=/path/to/rvllm   # or a download location, see below
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

The script splits the `vllm serve` command line in two:

- **Deployment flags go to rvllm.** These are `--host`, `--port`,
  `--served-model-name`, `--max-queued-requests`, `--shutdown-timeout` and
  `--structured-decoding-mode`. The disaggregation flags `--role`,
  `--kv-port`, `--kv-listen`, `--kv-advertise` and `--kv-producers` go to
  rvllm too, as do `--async-scheduling` and `--no-async-scheduling`.
  `--load-format` goes to both rvllm and the exporter.
- **Everything else goes to the exporter**, which records in the artifact what
  the runtime needs. Examples are `--max-model-len`, `--tensor-parallel-size`,
  `--max-num-seqs`, `--quantization` and `--speculative-config`. The exporter
  ignores options that only configure vLLM's own API server.

Put the model first, as in `vllm serve <model>`. The exporter's options are
not known to the script, so it takes a bare word after an option as that
option's value.

`bench` and `proxy` run the rvllm binary directly:

```bash
rvllm.bin proxy --port 8000 --prefiller-hosts p --prefiller-ports 8100 \
  --decoder-hosts d --decoder-ports 8200
```

## The rvllm binary

The binary is not part of this repository. `RVLLM_BINARY` points the script to
it, in one of two ways:

- **A local `rvllm` binary**, used as is.
- **A location holding `rvllm` and its `metadata.json`**, which has a `sha256`
  entry. It can be `gs://...`, `https://...` or a local directory. The binary is
  checked against the sha256 and cached.

A download location holds two files: the binary, named `rvllm`, and a
`metadata.json` that describes it. This `metadata.json` is not the one in an
exported artifact. The script reads one field, `sha256`, which is required: the
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

Other variables:

| Variable | Meaning |
|---|---|
| `RVLLM_BINARY_SHA256` | The expected sha256, overriding `metadata.json`. |
| `RVLLM_CACHE_DIR` | The cache root for binaries and exports. The default is `~/.cache/rvllm`. |
| `RVLLM_ARTIFACT_DIR` | Serve this exported artifact instead of exporting. |
| `PYTHON` | The interpreter that runs the exporter. The default is `python3`. |

Downloading from `gs://` uses `gcloud storage` with the caller's credentials.
