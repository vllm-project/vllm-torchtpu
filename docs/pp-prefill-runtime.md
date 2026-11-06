# PP prefill overlap

Synchronous MTP prefill producers using PP ring handoff overlap last-stage
output materialization with the next forward. Non-last stages prepare the next
input before waiting for outgoing queue capacity. These paths are selected by
configuration rather than separate enable/disable switches.

Output overlap is limited to single-DP synchronous MTP producers and retains
the draft KV completion fence and input staging ownership protection. Decode,
non-ring pipelines and warmup do not enter the prefill output lifecycle.
The EngineCore scheduling loop is unchanged.

Prepare-ahead bounds queued outgoing tensors to `depth + 1`. Receive capacity
must satisfy `carried_tensor_count * (depth + 2) <= 7`; configurations exceeding
this bound fail at initialization rather than silently changing paths.

`TPU_PP_RING_EXTRA_STEPS` controls extra in-flight scheduler batches beyond the
pipeline stage count. Zero retains the existing stage-derived default. This
capacity differs from `TPU_PP_RING_DEPTH`, which bounds outgoing handoff work.
For PP4, `TPU_PP_RING_EXTRA_STEPS=12` permits 16 in-flight batches; the previous
default permits 6. More batches can increase overlap but also outstanding RPC
and activation lifetimes; it does not guarantee that a pipeline stays busy.

The handoff memory reservation on upstream main is unchanged. No private KV
connector or additional tracing is required by these changes.
