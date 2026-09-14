# TPU patch mechanism

Register all TPU patches and workarounds in the ordered `PATCHES` tuple in
`src/vllm_torchtpu/patch_registry.py`. This registry defines when each patch runs.
Keep implementations in the relevant modules; the existing lifecycle entry points
call `apply(stage, model_config=...)` to apply them.

## Lifecycle stages

| Stage | Entry point | Use |
| --- | --- | --- |
| `import` | `env_override.py`, after early environment setup | Import-time prerequisites, currently the JAX Pallas lowering patch. |
| `platform_activation` | `TpuPlatform.check_and_update_config` | Shared TPU patches and model-specific patches with `vllm_config.model_config`. |
| `engine_core` | `_run_engine_core_with_tpu_patches` | Scheduler, KV-cache, offloading, and multimodal patches before the original engine-core entry point. |
| `worker_init` | `TPUWorker.__init__`, before `WorkerBase.__init__` | Shared patches in each worker, including spawned workers. |
| `model_load` | `TPUWorker.load_model`, before the model runner loads the model | Model-specific patches with `self.model_config`. |

Choose stages according to where the patched code runs and when its dependencies
are available. `_ACTIVE` means `platform_activation` and `worker_init`; `_ENGINE`
adds `engine_core`. Parent platform activation alone does not cover a spawned
worker. These stages are entry points in different processes, not a sequence that
every process necessarily executes.

Keep the `import` stage lightweight. Its target imports must not eagerly load
TorchTPU or resolve vLLM's platform. The registry imports targets lazily, so avoid
adding top-level implementation imports to `patch_registry.py`.

## Registering a patch

1. Implement a callback in the module that owns the behavior. Put dependency
   imports inside the callback where needed to preserve import ordering.
2. Add one `Patch` entry with a unique target to `PATCHES`. Place it after any
   prerequisite patches; tuple order is execution order within each stage.
3. Select the stages needed by the processes that use the patch. Use the existing
   lifecycle calls rather than adding a separate application list or direct call.
4. Test the behavior in the relevant patch implementation tests. Check stage
   selection and ordering in `tests/platforms/test_patch_registry.py`.

For example, this existing entry registers a loader callback for both platform
activation and worker initialization:

```python
Patch(
    "vllm_torchtpu.model_loader_patches:patch_default_loader_ep_weight_filter",
    _ACTIVE,
),
```

| Field | Contract |
| --- | --- |
| `target` | `module:callable` imports the module and invokes the named callable. A bare module path imports it for its side effects. The exact target string is the completion key. |
| `stages` | Tuple of supported stage names at which the entry is eligible. |
| `model_config` | Defaults to `False`. When `True`, the callback receives `model_config` as a keyword argument, possibly `None`. |
| `refresh` | Optional lazy target invoked on subsequent eligible calls after the primary target completes. It receives no arguments. |

Prefer an explicit callback for additions. The existing bare-module entry for
`vllm_torchtpu.layers.adapter.vision_attention` relies on import side effects; the
registry does not reload modules.

## Completion, retries, and model selection

A callback normally returns `None`, including when it intentionally does nothing
and needs no further attempt in that process. Only the literal `False` leaves the
entry eligible for another attempt. Every other return value marks it complete.
Retry requires another `apply` call to one of the patch's registered stages.

With `model_config=True`, check model eligibility inside the callback. Return
`False` when a missing or nonmatching configuration should allow a later attempt.
For example, `_apply_model_specific_patches` returns `False` for a non-Qwen3-VL
configuration and is registered at `platform_activation` and `model_load`.
Completion is keyed by target, not by model configuration, so a completed callback
does not run again for a different model in the same process.

An exception propagates and stops that stage's pass. Earlier completed callbacks
stay recorded; the failing callback remains eligible for retry. The registry does
not roll back partial mutations. If a callback can fail after installing part of
a patch, make its retry safe against wrapping or registering that part twice.

## Refreshing late bindings

A consumer using `from module import function` holds its own binding. Replacing
`module.function` does not update that binding. Use `refresh` when a patch needs
to update consumer bindings at later lifecycle boundaries.

The initial callback installs the patch and updates existing bindings. On later
eligible `apply` calls, the registry invokes the entry's refresh target, even
when the stage is unchanged. Refresh must preserve the installed wrapper and
safely update bindings on repeated calls.

`_patch_vllm_hybrid_pcp_block_sizes` and
`_patch_vllm_merge_multimodal_embeddings` use their installer as their refresh
target. Their implementation guards prevent duplicate installation while allowing
binding updates. A separate refresh callback is also supported. Refresh return
values are ignored; exceptions propagate without clearing primary completion.

## Process and concurrency behavior

Completion is process-local and shared across stages. Spawned processes start
with an empty registry. Forked processes inherit installed patches and completion
records; the child resets the registry lock and in-progress tracking.

The registry serializes application with an `RLock`. Re-entry into the same stage
returns immediately because the outer call owns that ordered pass. A target
already executing is skipped during nested application. Keep patch dependencies
in tuple order rather than relying on recursive stage calls.

## Validation and diagnostics

Update the manifest assertions in `tests/platforms/test_patch_registry.py` when
changing registered stages or dependencies. The file also tests retry, refresh,
model selection, process state, concurrency, and import ordering. Run it from the
repository root in the test environment:

```bash
pytest -q tests/platforms/test_patch_registry.py
```

Run the patch implementation tests as well. For changes to where a patch runs,
verify the relevant startup path, including spawned workers when applicable.

The registry logs `Completed TPU patch callbacks at <stage>: ...` at INFO level
for newly completed callbacks. Completion includes permanent no-ops, so this log
does not prove that a wrapper was installed or a feature was exercised. Deferred
callbacks and refresh calls are not included in that completion list.
