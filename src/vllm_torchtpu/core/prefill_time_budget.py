# SPDX-License-Identifier: Apache-2.0
"""Per-step prefill chunk size from a time budget (VLLM_TPU_PREFILL_TIME_BUDGET_MS).

At c32 about 31% of p90 e2el is decode time spent inside mixed steps that
carry a fixed 8192-token prefill chunk. A fixed smaller chunk is worse (more
steps, each paying the ~23 ms in-forward fixed cost), so this flag sizes the
chunk per step from the number of running decode requests ``n``: many decodes
-> a smaller chunk (each prefill token stalls ``n`` requests), few decodes -> a
larger chunk (can exceed 8192 when the engine runs with a larger
``--max-num-batched-tokens``; Qwen4Exp needs QWEN4_EXP_MAX_STEP_TOKENS for that).

Format: ``X`` or ``X:opt=val:...``. ``X`` is a time in ms.

* ``mode=stall`` (default): the prefill part of the step may take
  ``X * (ref / n) ** alpha`` ms, i.e. at ``n == ref`` decodes the chunk is
  ``X / us`` tokens. With ``alpha=1`` the decode-request x ms stall per step is
  constant.
* ``mode=step``: the whole mixed step may take ``X`` ms under the linear model
  ``fixed + dec * n + us * P``; ``P = (X - fixed - dec * n) / us``.

Options: ``ref`` (16), ``alpha`` (1.0), ``us`` µs per prefill token (16),
``fixed`` ms (36), ``dec`` ms per decode request (0.43), ``min`` chunk tokens
(2048), ``max`` chunk tokens (0 = the engine's max_num_batched_tokens),
``align`` tokens (512; Mamba chunk ends must be block aligned, so the chunk is
rounded down to a multiple of it).

With no running decode the chunk is ``max``. The step's token budget is the
chunk plus the decode tokens (``n * (1 + num_speculative_tokens)``), capped
at max_num_batched_tokens. Prefill tokens never change numerics; only the
chunk boundaries move.
"""

import dataclasses
import functools
import os

_ENV = "VLLM_TPU_PREFILL_TIME_BUDGET_MS"


@dataclasses.dataclass(frozen=True)
class Config:
    budget_ms: float
    mode: str = "stall"
    ref: float = 16.0
    alpha: float = 1.0
    us: float = 16.0
    fixed_ms: float = 36.0
    dec_ms: float = 0.43
    min_tokens: int = 2048
    max_tokens: int = 0
    align: int = 512

    def chunk_tokens(self, num_decodes: int, engine_max: int) -> int:
        """Prefill chunk (tokens) for a step with ``num_decodes`` running decodes."""
        hi = min(self.max_tokens, engine_max) if self.max_tokens > 0 else engine_max
        lo = min(self.min_tokens, hi)
        if num_decodes <= 0:
            p = hi
        elif self.mode == "step":
            p = (self.budget_ms - self.fixed_ms - self.dec_ms * num_decodes) * 1000.0 / self.us
        else:
            p = self.budget_ms * 1000.0 / self.us * (self.ref / num_decodes) ** self.alpha
        p = int(max(lo, min(hi, p)))
        if self.align > 1 and p >= self.align:
            p = p // self.align * self.align
        return max(1, p)


def parse(raw: str | None = None) -> Config | None:
    raw = (os.environ.get(_ENV, "") if raw is None else raw).strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        opts = dict(p.split("=", 1) for p in parts[1:])
        cfg = Config(
            float(parts[0]),
            mode=opts.pop("mode", "stall"),
            ref=float(opts.pop("ref", 16)),
            alpha=float(opts.pop("alpha", 1.0)),
            us=float(opts.pop("us", 16)),
            fixed_ms=float(opts.pop("fixed", 36)),
            dec_ms=float(opts.pop("dec", 0.43)),
            min_tokens=int(opts.pop("min", 2048)),
            max_tokens=int(opts.pop("max", 0)),
            align=int(opts.pop("align", 512)),
        )
    except ValueError as e:
        raise ValueError(f"{_ENV}={raw!r}: want X[:opt=val...]") from e
    if (
        opts
        or cfg.budget_ms <= 0
        or cfg.mode not in ("stall", "step")
        or cfg.ref <= 0
        or cfg.us <= 0
        or cfg.min_tokens <= 0
        or cfg.max_tokens < 0
        or cfg.align <= 0
    ):
        raise ValueError(
            f"{_ENV}={raw!r}: want X > 0 and options mode=stall|step, ref, alpha, us, "
            "fixed, dec, min, max, align (unknown: %s)" % sorted(opts)
        )
    return cfg


def _num_decodes(sched) -> int:
    return sum(1 for r in sched.running if r.num_computed_tokens >= r.num_prompt_tokens)


def patch_prefill_time_budget() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_torchtpu.logger import init_logger

    logger = init_logger(__name__)
    cfg = parse()
    if cfg is None or getattr(Scheduler.schedule, "_tpu_prefill_time_budget", False):
        return
    original = Scheduler.schedule

    @functools.wraps(original)
    def schedule(self, throttle_prefills: bool = False):
        engine_max = self.scheduler_config.max_num_batched_tokens
        spec = self.vllm_config.speculative_config
        width = 1 + (spec.num_speculative_tokens if spec is not None else 0)
        n = _num_decodes(self)
        chunk = cfg.chunk_tokens(n, engine_max)
        saved = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = min(saved, chunk + n * width)
        try:
            out = original(self, throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = saved
        steps = getattr(self, "_tpu_tb_steps", 0) + 1
        self._tpu_tb_steps = steps
        if steps in (1, 1000, 10000, 100000):  # path proof in the engine-core process
            logger.info(
                "Prefill time budget: step %d, %d decodes -> chunk %d (budget %d)",
                steps, n, chunk, min(saved, chunk + n * width),
            )
        return out

    schedule._tpu_prefill_time_budget = True
    Scheduler.schedule = schedule
    logger.info("Prefill time budget: %s", cfg)
