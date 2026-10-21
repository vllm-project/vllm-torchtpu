# Split-SparseCore MoE preprocessing

The fused EP FP4 MoE adapter supports an opt-in `topology` preprocessing
backend for TPU7x, eight EP ranks, and an affine rank mapping over a 2x2x1
chip topology with both dies. Unsupported topology, shape, or scratch-memory
requirements retain native preprocessing. Other weight formats retain the
native path.

## Enablement

Set these before initializing the TPU runtime, preserving any other libtpu flags:

```bash
export MOE_FUSED_EP_PREPROCESS_BACKEND=topology
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
```

The default backend is `native`. The DSV4.1 CI serving recipe selects `topology`
for the decode scenario and `native` for prefill/cached-extend. Large prefill
buckets within a decode server can still fall back through the resource guards.
The SparseCore queue-assignment compiler pass remains enabled.

## Execution and compilation

A complete three-phase activation all-gather keeps its SPMEM scratch within one
Pallas kernel. Metadata uses the native all-gather. A second Pallas kernel
scatters token indices and activation-scale bits together, with disjoint output
ownership across vector subcores and integer add/drop semantics.

The adapter resolves physical topology before tracing and registers one
PRE_SCHEDULER HLO transformation per process. For each eligible FP4 MoE call,
the transformation follows the kernel's activation and token-index operands to
identify its own collectives. It pins activation communication to SC1 and
metadata communication plus scatter to SC0, using immutable core assignments.
It adds control edges from activation submission to independent routing work,
and places the two independent FP4 weight-scale layout copies between metadata
submission and completion. It does not make routing depend on activation
communication completion.

TorchTPU can drop outer `compute_on` scopes and placement options while retaining
Pallas kernel names. The transformation recognizes both representations.
Unrelated compilations and ambiguous patterns are left unchanged; final HLO and
trace inspection are required when upgrading the compiler. The transformation
preserves protobuf fields it does not inspect and retains other backend options.
There is no JAX lowering monkeypatch or scheduling-group dependency.

## Validation and limits

Validation used JAX/jaxlib 0.11.2, libtpu 0.0.50 and TorchTPU 0.2.0rc1, with
FP4 weights, FP8 activations, EP8, 288 local tokens, hidden size 5120, 384 experts,
top-k 6, logical intermediate size 2304 padded to 2560, and one shared expert.
Six interleaved ABBAAB blocks each measured 30 calls. Profile counters verified
P7. Each call used the maximum over eight TensorCores, then the median over
90 calls per backend. No assumed device frequency was used.

| Metric | Native | Topology |
| --- | ---: | ---: |
| Full device execution, excluding entry wait | 434.40 us | 407.06 us |
| Preprocessing | 118.78 us | 91.60 us |
| Exposed activation AG completion wait | 23.28 us | 0.0024 us |

All 24 cross-rank/candidate output comparisons were bitwise equal. All three
candidate compilations retained SC1 activation placement, SC0 metadata/scatter,
early activation submission, and both weight-scale copies within the metadata
asynchronous window. Metadata completion still exposed about 8.51 us of wait;
this is not a claim that all communication waits disappear or that the entire
preprocessing path reaches an ICI roofline.

These measurements cover the production helper and kernel through a TorchTPU
whole-layer benchmark, including routing and the shared MLP. They do not replace
full vLLM serving validation through the production adapter. Decode CI provides
that integration check. Revalidate final HLO, correctness and performance after
changing runtime versions, shapes, physical topology, or compilation settings.
