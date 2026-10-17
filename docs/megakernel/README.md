# The megakernel backend: Kimi-K3 P/D on the tpu-megakernel Pallas kernels

Kimi-K3 served as a prefill/decode pair, both roles running the
tpu-megakernel Pallas kernels instead of vLLM's layer stack: a
sequence-parallel prefill stack on the KV producer and a fused 93-layer
decoder on the KV consumer. vLLM keeps scheduling, the unified block pool,
prefix caching, the Raiden Stage-3 KV transfer, sampling parameters, the
OpenAI API and its parsers.

## Running

Each role is a `vllm serve` over one 32-chip (4-host 2x2x4) slice, run with
the configuration below on every host; vLLM 0.29 with this checkout
installed editable. The prefill role is the Raiden KV producer and the
decode role the consumer; a P/D proxy in front
(`examples/disagg/toy_proxy_server.py --prefiller-hosts ... --decoder-hosts
...`) sends each request to the prefill, then to the decode instance.

The producer's premapped host pool is twice its explicit KV budget, with
an 8 GiB minimum to fit the largest presharded weight upload. Bounded Raiden
staging uses at most the device KV storage size; the remaining space provides
transfer headroom. A `TPU_PREMAPPED_BUFFER_SIZE` override takes precedence.

### The configuration

There is one, on both roles:

- `--load-format megakernel --model-loader-extra-config
  '{"presharded": "/path/to/kimi-k3-tp32"}'` selects the
  megakernel backend (`megakernel/backend.py`); `presharded` is the TP32
  pre-shard directory (the kernel-layout weights, the prefill's
  `prefill-sp-cache` and the DSpark draft's `dspark-<revision>`). The
  load format builds the model without weights (the program loads its own)
  and the platform turns vLLM's compilation off. Unknown loader keys, or a
  missing `presharded`, fail at startup;
- `--tensor-parallel-size 32 --enable-expert-parallel`, `--block-size 256`;
- `--decode-context-parallel-size 32 --cp-kv-cache-interleave-size 256`:
  each rank stores global page `k` of a sequence with `k % 32 == rank`, in
  logical block `k // 32`;
- a Raiden `--kv-transfer-config` (the role comes from `kv_role`: producer
  -> prefill program, consumer -> decode program) and prefix caching on;
- decode: async scheduling; optionally
  `--speculative-config '{"method": "dspark", "num_speculative_tokens": 7,
  "num_speculative_tokens_per_batch_size": [[1, 4, 7], [5, 8, 3]]}'` — the
  per-batch-size schedule says how many drafts each batch size verifies (the
  decoder derives its speculation shapes from it and rejects one it cannot
  run; the compiled kernels support 3 or 7 drafts at batches 1–8, and 3 at
  batches 9–16, through compiled request buckets 1, 2, 4, 8 and 16 up to
  `--max-num-seqs`, and 0 anywhere);
- decode, under `"rejection_sample_method": "synthetic"`: optionally
  `"synthetic_acceptance_lengths": {"3": 2.87, "7": 3.78}` in the decode
  role's `--model-loader-extra-config` gives each draft width of the
  schedule its own mean acceptance length (vLLM's min-variance rates for
  that width); without it every width draws from `--speculative-config`'s
  one rate vector, truncated. Every width the schedule runs needs a length,
  and the 7-draft one must match `synthetic_acceptance_length`;
- with DSpark, `"draft": "<the speculator checkpoint>"` in the prefill's
  loader config: the prefill computes each request's draft ring, and a
  decoder with a DSpark `--speculative-config` imports it. Both peers'
  transfer manifests carry the ring rows only then, so a pair where one side
  has DSpark and the other doesn't has different layout fingerprints.

### FP8 decode attention

Both roles may select `--kv-cache-dtype fp8` (or `fp8_e4m3`) together with
`--block-size 512 --cp-kv-cache-interleave-size 512`. BF16 is the default;
the two roles must agree.

The FP8 pool holds 512 tokens in `[128, 4, 640]` E4M3FN pages, with scale one
for latent and RoPE cache values. Its physical row and slot sizes match the
256-token BF16 pool exactly, so recurrent FP32 state, BF16 convolution and
DSpark draft-ring state keep their byte offsets and precision. A logical
DCP32 cache block spans 16,384 tokens; this also coarsens prefix reuse.
Producer and consumer cache encoding must match; the transfer fingerprint
rejects mixed BF16/FP8 layouts.

Cached-page decode attention quantizes each absorbed query head once after
its BF16 all-gather, using its maximum magnitude to map into E4M3FN's range.
QK and PV use native FP8 matrix operations with FP32 accumulation. Softmax
statistics and the denominator stay FP32; probabilities are multiplied by
256 before E4M3FN conversion and PV is divided by 256 afterward. The small
current-token/verification-chunk merge is FP32, and the decoder's
projection/output contract and the query exchange are BF16. This is not
numerical equivalence to a GPU kernel.

Prefill computes its window attention in BF16 and writes FP8 latents into
the shared pool. Cached prefixes are converted back to BF16 before
prefill's key/value expansion. Every decode/verification commit uses the
same encoding and preserves other tokens in the packed word.

Standalone tools accept the same option, for example:

```bash
python -m tools.megakernel --jobid JOB --kv-cache-dtype fp8 \
    bench decode --batches 1,8,32,64 --contexts 32768,131072
```

The FP8 option changes arithmetic and may change generated tokens. It is
opt-in; evaluate task quality before making it a deployment's default.

## How it works

### Runner and program

`MegakernelModelRunner` (`megakernel/runner.py`, chosen by the worker on the
megakernel path) owns the vLLM contract: request lifecycle, the KV
connector, async output. Every step it hands the program a `StepInput` --
the batch's requests as host data (`RequestInput`), finished requests and
block copies (`megakernel/program.py`, with `KVGroup` and `StepOutput`) --
and returns the tokens the program chose in its ops; there is no input
preparation, attention metadata, logits, sampler or drafter. The program
for the role is `KimiPrefiller` (`megakernel/kimi_k3/prefill.py`) or
`KimiDecoder` (`megakernel/kimi_k3/decode.py`), built by
`megakernel/kimi_k3/program.py` from the `MegakernelEngineConfig`
(`megakernel/config.py`), which it also checks; it states the capabilities.
Programs have no vLLM dependencies; `python -m tools.megakernel` drives them
directly. vLLM's compilation is off on this path (`CompilationMode.NONE`):
the ops are compiled by JAX.

### Kernels as torch ops

The kernels are pure JAX/Pallas SPMD programs over the 32 devices. Each
program op is a `shard_map` body registered as a torch custom op
(`distributed/sharded_jax_op.py`, via `megakernel/bridge.py`) on one kernel
mesh over all devices in ascending id; a kernel's rank is the worker's
position in it, not its vLLM TP rank (KDA heads move between the two orders
with a `ppermute`, `pools.tp_rank_permutation`).
`bridge.spmd_safe_axis_index` makes the kernels' XLA-level `lax.axis_index`
survive torch_tpu's SPMD export. Ops with an `export_key` persist their
exported modules (`megakernel/export_cache.py`) under
`<VLLM_XLA_CACHE_PATH>/k3_export`, so a warm restart skips tracing and
lowering. The kernels live in `src/vllm_torchtpu/megakernel`: shared
building blocks in `common/`, the K3 kernels in `kimi_k3/kernels/` (JAX only,
no vLLM or torch), their settings in `kimi_k3/config.py` (`KernelConfig`,
installed before the first kernel import; serving uses the defaults). Each
role's kernel is one file: `decode_megakernel.py` (with `mla_dcp.py`,
`paged.py` and `dspark.py`) and `prefill_megakernel.py` (its module
docstring maps the ops); model math both evaluate is in `helpers.py` and
weight loading in `rank_loader.py`.

vLLM still builds `KimiLinearForCausalLM`, on the meta device
(`megakernel/loader.py`): its attention and KDA layers declare the KV cache
spec vLLM allocates from; no parameter takes memory.

### The unified pools

Each `KVCacheTensor` is one pool `[blocks, 128, 2, 640]` bf16 shared by one
MLA layer and three KDA layers: MLA latent pages (576 lanes: the normalized
latent and the rope part; token `t` of a page at `[t // 2, t % 2]`) and, in
a KDA group's state block, the layer's state (rows 0-76 recurrent f32, rows
77-80 conv taps) and, with the draft ring, rows 81-95: a fifteenth of the
request's ring. The Raiden connector moves each rank's pool bytes to the
same rank on the other role (`lower_dcp_page_spans`; the state blocks'
conv region extended over the ring rows, `pool_manifest`).

### Prefill (`megakernel/kimi_k3/prefill.py`)

Every live row of a step is computed. A row's tokens are cut into pieces of
at most the largest window (512, 1024, 2048, 4096, 8192 tokens, up to
`--max-num-batched-tokens`), and up to 16 sequences (`--max-num-seqs`, and
at most one per page of the window) are packed into a window at page-aligned
offsets. A window runs the stack's first (embedding +
layers 0-3), group (8 layers, 11 times) and final (layer 92 + each
sequence's last-token logits) ops; each rank writes the new pages it owns.

- **Cached prefixes:** the pools hold each prefix context parallel (page
  `p` on rank `p % 32`). Each MLA layer all-gathers every rank's valid
  prefix pages, as stored (FP8 or BF16), over the six TP32 ring streams
  (`gathered_prefix_pages`), expands them for the rank's 3 heads
  (`expand_prefix_pages`) and attends the window and the prefixes in one
  flash call, as vLLM's GPU DCP prefill gathers the context's latent KV.
- **MLA flash tiling:** 1024 query rows share each key tile, reducing
  repeated prefix reads while keeping each query's key reduction order.
- **MoE expert weights:** the indexed layer passes the resident packed weight
  families and a scalar layer index into the grouped matmuls; the Pallas
  kernel selects an HBM view before loading each tile, so no full-layer
  dynamic-slice copy is made per window.
- **MoE latent exchange:** the TP32 Pallas ring gathers the BF16 latent rows
  over physical routes in mesh-rank order. Routing IDs and probabilities
  use int32 and FP32 gathers; the routed output uses the reduce-scatter
  ring. Gather ownership stays separate from any routed-output token
  permutation.
- **KDA state** between a request's steps stays in memory; a prefix-cache
  hit reads it from the block vLLM's `align` mode keeps at the hit.
- **Align-mode splits:** vLLM splits a prompt at its last block boundary to
  cache the KDA state there. A step that stops short of a prompt whose rest
  fits one window is skipped (no token), and the computing window
  checkpoints the state at that boundary into the block vLLM assigned.
- **Prefix hits inside a logical block** (256-token granularity under DCP)
  copy-on-write the block; the runner hands the copies to the program
  (`step(..., copies)`). The decode program applies no copies (its
  `Capabilities.block_copies` is off), so a scheduler patch in `__init__.py`
  rounds its hits down to whole logical blocks; another keeps the prefill's
  chunk ends on 256-token pages.
- The first token is chosen in an op from the kernel's logits, with the
  decoder's in-op sampling.

### Decode (`megakernel/kimi_k3/{decode,decode_ops,draft}.py`)

The program runs every decode step: the runner hands it a `StepInput` and
gets the step's tokens; no input uploads, logits or rejection sampler. The
decoder reads each rank's latent pages from vLLM's pools in place and
commits the new ones there; KDA state lives in private slots, imported from
the pool on a request's first step.

Two layers, as on the prefill side:

- **`KimiDecoder`** (`decode.py`) is the decode role's program: it binds
  vLLM's KV caches, imports a request's KDA state and DSpark ring on its
  first step, keeps every request's state between steps, and decides what
  each step runs (below).
- **`DecodeOps`** (`decode_ops.py`) holds the weights and KDA slots and
  builds the step ops (plain steps, the single and batched verifies), with
  the `Control` vector's layout. `kimi_k3.program.build` makes it; the
  standalone tools share one between several decoders.

- **Control state:** one replicated int32 vector every op reads and
  returns: per slot the next token, its position and which verify row holds
  its KDA state, and the host plan (per batch row its slot, a reset with the
  host token and position, sampling parameters, block table), uploaded only
  when it changes. Under async scheduling nothing else is uploaded per step.
- **Plain steps:** one op per compiled batch size (powers of two up to
  `--max-num-seqs`; a step pads to the next), choosing tokens in the op
  (greedy, or temperature / top-p / top-k / seed on the sharded logits).
- **DSpark with seven drafts** (one request decoding): the 8-row verify of the
  anchor and 7 drafts, then the fused draft op, which accepts in the op
  (greedy, or exact rejection sampling) and drafts the next 7. The verifier
  reads the request's persistent KDA state and emits temporary post-states.
  After acceptance, a separate op copies only the selected post-state into
  the request's regular slot; control metadata keeps the logical verifier
  entry for width transitions.
- **Width-aware DSpark** (a three-draft singleton or several requests,
  through compiled request buckets 1, 2, 4, 8, 16 up to `--max-num-seqs`):
  the default schedule uses 8 rows per request through batch four and 4
  through batch eight (one anchor plus 7 or 3 drafts). The corresponding
  verifiers are 2×8, 4×8 and 8×4; an explicit schedule can select 8×8 and
  16×4 too (`[[1, 8, 7], [9, 16, 3]]` with `--max-num-seqs 16`). Live counts
  between buckets pad with null-slot groups. One fused draft kernel runs for
  the whole batch: projection weights are read once while each request keeps
  its own ring, positions, Markov chain and sampling state. Ring slices use
  two VMEM buffers; the eight-request shape also streams output/down
  projection columns to stay within the VMEM budget, and batches of eight or
  more stage one request's auxiliary features at a time. The 16×4 shape
  computes only four physical rows per request and packs replicated metadata
  before the custom-op boundary to stay below Torch's 64-argument limit.
  Batched floating-point reductions can change BF16 rounding and proposals;
  target verification still decides acceptance. vLLM's per-batch-size
  schedule (`num_speculative_tokens_per_batch_size`: default 7 through batch
  four, 3 through batch eight, 0 above) decides how many drafts each step
  verifies, and the decoder derives which batch sizes speculate from it; a
  request's drafts are verified only when they were made for its current
  next token and at least the scheduled verification width (otherwise its
  rows are placeholders). Changing from three to seven drafts first runs a
  placeholder step to generate a full proposal. Draft states are packed and
  split inside the registered JAX operation, so each Torch request state
  keeps canonical backing dimensions when batch sizes change. States are
  materialized when made and when a draft op returns them: torch_tpu would
  otherwise compile their pending work (fills, the outputs' view updates)
  into the next op that takes them, one variant per mix of new and
  continuing requests (`check spec`'s churn phase times every step after
  warmup to catch this).
- **Streamed KDA state:** above 16 rows with attention DP disabled, the
  decoder reads each sequence's initial state directly from its slot bank;
  later verifier rows use the preceding row's recurrence. A separate ordered
  DMA commit writes the batch results back to the bank. It preserves
  untouched slots and last-write ordering for duplicate slot indices,
  including padded null-slot rows.
- **Cold warmup:** `megakernel/compilation.py` submits at most eight large
  decoder/draft compilations per worker, with larger row counts first. The
  default maximum of four requests has six plain programs (B1/B2/B4, greedy
  and sampled), three verifiers (1x8, 2x8 and 4x8), and six acceptance/draft
  programs (single, 2x8 and 4x8, greedy and sampled). Fresh draft vectors
  get a small owned copy after their indexed initialization writes, giving
  them the same `[1, 128]` backing as continuing states, so the large draft
  and entry programs compile once for every mix of backing shapes.
  Rank-invariant native/draft builders distribute their exports across the
  workers and share them through the export cache; other ops export per
  worker. Each compile-only graph has independent placeholders; it cannot
  donate live KV/KDA buffers or execute collectives. The complete ordered
  warmup still runs afterward, including control, imports, sampling, ring
  entry and producer transitions.
- **Batch recipes** (`decode_options`): from B8, rows are token-scattered
  over sequence-parallel owners with the routed reduce-scatter.

### The DSpark draft ring (`megakernel/kimi_k3/draft_ring.py`)

The draft attends a ring of the last 2048 positions' context rows: per draft
layer, the key and value of the target's residual streams entering layers
24, 48, 72, 88 and 92, projected. Started empty, it drafts poorly until it
fills; the prefill computes it from the prompt instead, as vLLM's DSpark
speculator fills its draft KV during prefill and its KV connector moves it.

- **Prefill:** after a window's regular groups, one op takes the streams
  (the residual `blocks` slots 2/4/6, the pair-10 middle output, the prefix
  into the final op), projects every rank's own rows for all 16 KV heads
  with replicated `fc` / K/V weights (gathered from the draft's kernel
  shards at load, 0.64 GiB per rank), and an all-to-all hands each rank its
  head's rows (`kernels.dspark.prefill_ring_update`). The ring is kept per
  request between steps, like the KDA state, written with it (final and
  checkpoint blocks) and read back on a prefix hit.
- **Header:** the ring's length and first held position travel with it
  (each byte an exact bf16 number); a reader that finds no ring for exactly
  the tokens it holds starts cold.
- **Decode:** the import beside the KDA state (`KdaImporter.import_ring`,
  in `KimiDecoder`'s first step of the request) hands the ring to the
  request's first speculation.

`check handoff --dspark` checks each imported ring and compares speculation
from it with speculation from a cold ring.

The decode role rejects sampling parameters the ops do not implement
(logprobs, penalties, min_p, logit bias, allowed/bad words, min_tokens;
`platforms/tpu_platform.py`). Structured outputs (the grammar bitmask) the
plain decoder applies in its step op: a step with a structured request is
deferred to `sample_tokens` (the engine computes the bitmask from the
previous step's tokens), which runs the program with the requests' masks
(`StepInput.mask`) and loses that step's async overlap. With DSpark the
verify ops do not mask: while a structured request decodes, the whole batch
steps plain and masked (no drafts), and speculation resumes once no
structured request is left in the batch.

## Limits and known issues

- TP32, DCP 32 and block size 256 only. Plain decode serves batch sizes up
  to `--max-num-seqs` (kernel-checked to B64; P/D-validated to 8);
  speculative verifies up to 64 rows (8 requests with seven drafts, or
  16 with three). B32 and B64 stream selected weights to fit VMEM.
- The speculation lengths are the compiled verify shapes: 3 or 7 drafts at
  batches 1–8, and 3 at batches 9–16 (compiled request buckets 1, 2, 4, 8
  and 16, up to `--max-num-seqs`), and 0 anywhere; the per-batch-size
  schedule chooses among them and the decoder rejects any other at startup.
- Besides its pages, each decode request holds one KDA state block per KDA
  group of the decode KV budget (two while it crosses a page), 7.5 MiB a
  rank each. Under KV pressure a request can hang: its Stage-3 transfer
  completes but the completion is dropped as unmapped.

## Tools

`python -m tools.megakernel` runs the serving programs without a vLLM
server. It expects a Slurm allocation of four TPU hosts (2x2x4, 32 chips):
run from a login node with `--jobid` (default `$JOBID`), it re-launches
itself with `srun` as 32 processes, 8 per host, in the Python environment
`--venv` (default `$K3_VENV_DIR`, else the current one). `bench`, `check` and
`profile` take `--presharded DIR` (the pre-shard below); `check` also takes
`--model`, the Kimi-K3 checkpoint, and `check spec` `--dspark`, the
speculator checkpoint.

- `bench decode --batches 1,8,16 --contexts 1152,32768 [--dspark CHECKPOINT]`:
  kernel-only timing of the serving decode ops (mixed contexts with
  `--mix`, sampled ops with `--temperature`);
- `bench prefill --prompts 8192x1,2048x4,3000x1@32768`: the prefill stack on
  fresh prompts and on prompts appended to a cached prefix (`@PREFIX`); each
  line names the compiled windows the step packs into;
- `check spec`: DSpark against plain greedy, batch transitions, seeded
  sampling; `check handoff`: prefill -> pools -> decode against a
  decode-only replay of each prompt;
- `profile decode ... --llo DIR`: an xprof trace and the compiled LLO
  (`*-vliw-packed-bundles.txt`) of every op.
- `preshard k3 --checkpoint CKPT --output DIR` / `preshard dspark --draft
  CKPT --output DIR`: the checkpoint and the speculator in the kernels'
  per-rank layout (the directory `--model-loader-extra-config
  '{"presharded": DIR}'` names); `k3` runs one process per host, no TPU.
  `preshard verify --against DIR --checkpoint CKPT --draft CKPT` rebuilds a
  sample of one rank and compares it with an existing pre-shard byte for
  byte.
- `preshard prefill --checkpoint CKPT --output DIR`: generate
  `DIR/prefill-sp-cache` from the same full Kimi-K3 checkpoint. Run this after
  `preshard k3`, using the same output root. It runs on the allocation's first
  host without using a TPU and refuses to overwrite an existing cache.

`--set FIELD=VALUE` overrides a `KernelConfig` field to compare
configurations, and `--json FILE` appends a record per result (with the
config digest and git revision). `--coverage FILE` merges the kernel lines a
run executes into `FILE` (export cache off, so every op is traced);
`python -m tools.megakernel.coverage FILE` then lists, per kernel file, the
functions no run entered and the branches none took.

## Performance

Kernel-only decode step (`bench decode`, distinct token streams):

| context | B8 | B16 | B32 | B64 |
|---|---|---|---|---|
| 1152 | 10.0 ms | 15.5 ms | 25.7 ms | |
| 32768 | 10.6 ms | | 25.6 ms | 44.5 ms |

Kernel-only prefill step (`bench prefill`, median of 3):

| prompts | window | step |
|---|---|---|
| 8192x1, fresh | 8192 | 705.6 ms |
| 2048x4, fresh | 8192 | 696.5 ms |
| 3000 new tokens on a 32k prefix | 4096 | 447.8 ms |
| 3000 new tokens on a 131k prefix | 4096 | 542.2 ms |
| 2048x4 on 8192-token prefixes | 8192 | 797.2 ms |

Batched DSpark (`check spec`, each request on its own prompt, at the
check's greedy acceptance):

| requests | speculative step | plain step | tokens/ms, spec vs plain |
|---|---|---|---|
| 2 (8-row verify) | 11.7 ms | 6.3 ms | 0.47 vs 0.32 |
| 4 (16 rows) | 18.0 ms | 9.2 ms | 0.62 vs 0.44 |
| 8 (32 rows) | 28.7 ms | 11.9 ms | 0.75 vs 0.67 |

P/D serving, agentic workload (3000-token turns, 256 output tokens, to 40k
context), median TPOT with DSpark / without: 2.3 / 4.8 ms at 2
conversations, 2.4 / 5.8 ms at 8. That filler text accepts many drafts;
real prose and code gain less.

Rows with distinct tokens touch more experts than a uniform fixture (one
prompt for every row), so compare numbers only within one allocation and
fixture.
