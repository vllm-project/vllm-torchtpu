# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class XlaGraphTrackerMixin:
    """XLA graph growth tracking and fresh-compile reporting."""

    # Entries described per report before the rest are summarized. A boot
    # compiles hundreds, and the interesting case -- a handful appearing once
    # traffic is running -- is well under this.
    _MAX_DESCRIBED_GRAPHS = 20
    # Newest entry seen by the previous report, which is what makes an entry
    # "first read since then". A class attribute rather than an __init__ one
    # because the first report runs from inside __init__, so an instance
    # attribute has to be assigned above that call to exist in time.
    _xla_graphs_checked_at = None

    @staticmethod
    def _describe_xla_graph(entry) -> str:
        """One line describing a cache entry, from whatever fields it carries.

        ``per_entry_stats`` holds an opaque pybind type whose fields differ
        between torch_tpu versions, so read them reflectively rather than
        pinning names that will silently vanish on an upgrade.
        """
        fields = []
        for name in sorted(dir(entry)):
            if name.startswith("_"):
                continue
            try:
                value = getattr(entry, name)
            except Exception:  # a property that needs live device state
                continue
            if callable(value):
                continue
            text = str(value).replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            fields.append(f"{name}={text}")
        return " ".join(fields) if fields else repr(entry)[:200]

    @staticmethod
    def _xla_graph_compile_secs(entry) -> float:
        duration = getattr(entry, "compilation_duration", None)
        total_seconds = getattr(duration, "total_seconds", None)
        return total_seconds() if total_seconds is not None else 0.0

    def _fresh_xla_graphs(self, entries) -> list:
        """The entries this process compiled since the previous report.

        There is no identifier on an entry to diff against, and the list is not
        append-only -- its tail routinely holds boot-time programs with
        ``read_count`` in the dozens -- so slicing off ``num_xla_graphs`` names
        the wrong graphs. What each entry does carry is when it was last read
        and how often, so an entry first read after the previous report is one
        this report is responsible for.
        """
        fresh, newest = [], self._xla_graphs_checked_at
        for entry in entries:
            last_read = getattr(entry, "last_read", None)
            if last_read is None:
                continue
            if newest is None or last_read > newest:
                newest = last_read
            if getattr(entry, "read_count", 0) != 1:
                continue  # read before, so it predates this report
            if (self._xla_graphs_checked_at is not None
                    and last_read <= self._xla_graphs_checked_at):
                continue
            fresh.append(entry)
        self._xla_graphs_checked_at = newest
        fresh.sort(key=self._xla_graph_compile_secs, reverse=True)
        return fresh

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        stats = torch.tpu._get_cache_stats()
        entries = stats.per_entry_stats
        total_graphs = len(entries)
        if total_graphs < self.num_xla_graphs:
            # Entries were evicted, so the delta below would understate what
            # was compiled. Resynchronize rather than report a negative count.
            logger.info(
                "XLA cache shrank from %d to %d entries (case: %s); "
                "resynchronizing the recompilation counter",
                self.num_xla_graphs, total_graphs, case_str)
            self.num_xla_graphs = total_graphs
            return

        fresh = self._fresh_xla_graphs(entries)
        new_compiled_graphs = total_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0 and not fresh:
            logger.info(f"No new compiled graphs for case: {case_str}")
            return

        logger.info(f"Total Requests: {stats.num_cache_reqs}")
        logger.info(f"Total Hits: {stats.num_cache_hits}")
        logger.info(
            f"Total number of cached graphs: {total_graphs}, new: {new_compiled_graphs}, case: {case_str}"
        )

        # A count says four graphs appeared; it does not say which op keeps
        # producing fresh shapes, which is the only thing that makes a runtime
        # recompilation actionable. Note the compile seconds: an entry with a
        # zero duration was served from a cache rather than compiled here, so
        # it costs nothing even though it counts.
        if fresh:
            secs = sum(map(self._xla_graph_compile_secs, fresh))
            logger.info(
                "  %d first read since the last check, %.3fs compiling",
                len(fresh), secs)
            for i, entry in enumerate(fresh[:self._MAX_DESCRIBED_GRAPHS], 1):
                logger.info("  graph %d/%d: %s", i, len(fresh),
                            self._describe_xla_graph(entry))
            if len(fresh) > self._MAX_DESCRIBED_GRAPHS:
                logger.info("  ... %d more not shown",
                            len(fresh) - self._MAX_DESCRIBED_GRAPHS)

        self.num_xla_graphs += new_compiled_graphs
