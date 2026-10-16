# Viewing traces

## On the TPU VM

The simplest option is to leave the traces where they were written and serve
them from the VM:

```bash
xprof --logdir /tmp/vllm_phased_profile
```

If you are connected through VS Code's Remote-SSH, it detects the listening port
and forwards it automatically, so the URL it prints opens directly in your local
browser. Otherwise, forward that port yourself when connecting (6006 by
default):

```bash
ssh -L 6006:localhost:6006 <tpu-vm-host>
```

## On your local machine

To keep traces after the VM is torn down, or to share them, copy them down
first. Preserve the directory hierarchy — XProf reads the
`plugins/profile/<timestamp>/` layout, not just the individual file. This copies
both formats; the JSON pattern simply matches nothing on a phased run:

```bash
rsync -avzm \
  --include='*/' \
  --include='*.xplane.pb' \
  --include='*.pt.trace.json.gz' \
  --exclude='*' \
  <tpu-vm-host>:/tmp/vllm_phased_profile ~/traces/
```

```bash
xprof --logdir ~/traces/vllm_phased_profile
```

## Reading the result

Open the printed URL and pick a run from the dropdown. Each phase is a separate
run. The Trace Viewer shows kernel execution timelines, and the Memory Profile
and Op Profile views cover HBM usage and MXU utilization.

## Seeing every rank at once

All of a run's **XPlane** files are merged into one
`plugins/profile/<timestamp>/` directory, so pointing XProf at the run directory
loads them together — use the host picker to switch between ranks, or view them
side by side, without merging anything by hand. This is the reason the capture
goes through the rank-prefix merge described above; a per-rank directory layout
would force you to open each rank as a separate session.

The Chrome-format JSON traces do **not** go through that merge. They are already
uniquely named per worker, so they never collide, but each rank keeps its own
file under `rank_<N>/`. A viewer that opens a single file therefore shows one
rank at a time.

## Combining a run into a single file

Some consumers take exactly one file rather than a directory — Google-internal
XProf's offline upload, or a bug attachment. `merge_xplane` combines a session's
per-rank `.xplane.pb` captures into one XSpace:

```bash
python -m vllm_torchtpu.tools.merge_xplane /tmp/vllm_phased_profile
```

```text
Found 3 capture sessions
A plane is one timeline inside a capture: one per TPU chip, one per host,
plus a shared metadata block. Merging combines planes, not whole files.

  decode_only  2026_08_20_22_14_43
    input   4 captures (ranks 0-3), 205 MB
    planes  20 read -> 17 written (3 duplicate /host:metadata dropped)
    output  106 MB
    -> /tmp/vllm_phased_profile/decode_only/decode_only_2026_08_20_22_14_43.xplane.pb
  ...

Summary
  3 sessions merged, 1164 MB -> 572 MB, 50% smaller
  /tmp/vllm_phased_profile/decode_only/decode_only_2026_08_20_22_14_43.xplane.pb
  /tmp/vllm_phased_profile/prefill_heavy/prefill_heavy_2026_08_20_22_14_46.xplane.pb
  /tmp/vllm_phased_profile/prefill_only/prefill_only_2026_08_20_22_14_31.xplane.pb
```

Each rank's capture holds five planes on a v6e TP=4 run — its chip
(`/device:TPU:0`), `/host:metadata`, `/device:CUSTOM:Megascale Trace`,
`Task Environment`, and `/host:CPU`. Four captures therefore contribute 20
planes, and the merged file holds 17: the three redundant `/host:metadata`
copies are collapsed into one. The `Summary` block repeats every output path,
since the per-session detail scrolls away on a long run.

Each phase is an independent capture window anchored at its own zero, so a
phased run yields one merged file per phase rather than one for the whole run.

The merged file lands in the **phase directory** — the one holding `plugins/`.
That location follows the capture, not the command, so every way of pointing at
a phase produces the same file in the same place:

```bash
cd /tmp/vllm_phased_profile/decode_only
python -m vllm_torchtpu.tools.merge_xplane .                        # here
python -m vllm_torchtpu.tools.merge_xplane plugins/profile          # or here
python -m vllm_torchtpu.tools.merge_xplane plugins/profile/2026_*   # or here
# all three write ./decode_only_2026_08_20_22_14_43.xplane.pb
```

Re-running is safe: a merged file already sitting in a phase directory is
recognised as output rather than picked up as another capture to merge.

Pass `--dry-run` to see what would be written, `-o` to name a single output
(only when the inputs resolve to one session), or `-d` to choose the output
directory.

The output is roughly half the size of the inputs combined: every rank ships a
full copy of the `/host:metadata` plane, which carries the HLO protos, and the
merge keeps one. Merging on the TPU VM before copying traces down therefore
moves considerably less data than copying the per-rank files.

Two things worth knowing:

* **You do not need this for `xprof --logdir`.** OSS XProf already reads the
  whole session directory as one distributed session, as described above.
* **Merged files must not live in `plugins/profile/<timestamp>/`.** XProf builds
  its host list from the filenames it finds there, so a merged file sitting
  beside the per-rank ones appears as an extra host holding a second copy of
  every rank — a 4-rank capture reports 8 TPU cores and twice its real event
  count, with nothing to indicate anything is wrong. The phase directory is
  safe because XProf enumerates runs only at `*/plugins/profile/*` and ignores
  loose files above that; `-d` and `-o` are still refused if they point inside
  a session directory.
