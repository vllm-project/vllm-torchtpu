# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""2D phase map, local FileReader calibration dropzone, and bucket export UI."""

from __future__ import annotations


def render_phase_stage_tab_button() -> str:
    """Return the stage tab button for the 2D Tokens x Context Phase Map."""
    return (
        '<button class="wb-tab-btn" id="btn-stage-phase" type="button" '
        "onclick=\"wbSetStageTab('phase')\">"
        "2D Tokens &times; Context Phase Map &amp; Bucket Schedule</button>"
    )


def render_phase_extension_js() -> str:
    """Return ES2022 logic for 2D phase heatmap, FileReader dropzone, and export."""
    return """
  const PHASE_T_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
  const PHASE_S_GRID = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

  (function mountPhaseDom() {
    const railSlot = document.getElementById('wb-extra-rail-slot');
    if (railSlot) {
      railSlot.innerHTML =
        '<div class="wb-sec-title">4. Trace Calibration Dropzone</div>' +
        '<div class="wb-knob-group" id="wb-cal-dropzone" ' +
        'ondragover="event.preventDefault()" ' +
        'ondrop="wbOnDropCalibration(event)">' +
        '<div style="font-size:10px;color:var(--muted);">' +
        'Drop <code>advisor.measurements.v1</code> or Chrome trace ' +
        '(<code>.json</code> / <code>.json.gz</code>) to upgrade ' +
        '<code>ASSUMED</code> priors to <code>MEASURED</code>.' +
        '</div>' +
        '<input id="cal-file-input" type="file" ' +
        'accept=".json,.gz,application/json,application/gzip" ' +
        'onchange="wbOnPickCalibration(event)" ' +
        'style="font-size:10px;width:100%;"/>' +
        '<div id="wb-cal-status" style="font-size:10px;color:var(--muted);">' +
        'Status: uncalibrated hardware priors</div></div>';
    }
    const stageSlot = document.getElementById('wb-extra-stage-slot');
    if (stageSlot) {
      stageSlot.innerHTML =
        '<div id="wb-stage-phase" ' +
        'style="display:none;flex-direction:column;gap:14px;">' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Continuous 2D Winning-Scope Phase Map ' +
        '(Active Tokens &times; Context)</h2>' +
        '<span class="kpi-sub">Click any (T, Context) cell to inspect ' +
        'in Arena</span></div><div id="wb-phase-map-svg"></div></div>' +
        '<div class="wb-panel"><div class="wb-panel-head">' +
        '<h2>Advisory Token-Bucket Schedule JSON (Offline Export)</h2>' +
        '<button class="wb-mini-btn" id="btn-copy-bucket-schedule" ' +
        'type="button" onclick="wbCopyBucketSchedule()">' +
        'Copy Bucket Schedule JSON</button></div>' +
        '<div class="kpi-sub" style="margin-bottom:6px;">' +
        'Schema <code>advisor.bucket_schedule.v1</code> ' +
        '(advisory offline schedule; unwired to runtime runner dispatch).' +
        '</div><pre class="wb-formula" id="wb-bucket-schedule-pre" ' +
        'style="max-height:200px;overflow-y:auto;margin:0;"></pre>' +
        '</div></div>';
    }
  })();

  window.wbFocusedTopicCard = function(wlA, evalA, evalB, lblB, topic, sc) {
    if (!topic || !sc) return '';
    const esc = window.__WB_API.escH || String;
    if (topic.kind === 'vmem') {
      const spK = (evalA.vmemParts.spineBytes / 1024).toFixed(1);
      const vmap = {
        dense: ['Dense Weight Ring', sc.dense_stage_mib,
          `slots=${wlA.dmaBuffers} (knob: DMA Slots #ctl-dma)`],
        exp: ['Routed Expert Ring', sc.expert_stage_mib,
          `slots=${wlA.dmaBuffers} (knob: DMA Slots #ctl-dma)`],
        scr: ['Static Op Scratch', sc.op_scratch_mib,
          'attention/GDN/MoE tile scratch'],
        pad: ['Compiler Spill & Align Reserve', sc.pad_mib,
          'scoped VMEM alignment reserve'],
        spn: ['Dynamic [T, H] Residual Spine', sc.spine_dyn_mib,
          `T=${evalA.activeTokens} tok * ${spK} KiB/tok`],
        tok: ['Dynamic Per-Token Op Scratch', sc.op_dyn_mib,
          `alias=${wlA.vmemAlias ? 'enabled (max)' : 'disabled (sum)'} (#ctl-alias)`]
      };
      const info = vmap[topic.sub] || ['VMEM Region', sc.vmem_static_mib, sc.scope];
      const pct = ((info[1] / Math.max(1, evalA.vmemBudgetMib)) * 100).toFixed(1);
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Selected VMEM: ${esc(info[0])}` +
        `</strong><span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">${info[1].toFixed(2)} MiB (${pct}% of ` +
        `${evalA.vmemBudgetMib.toFixed(1)} MiB budget)\n${esc(info[2])}</div></div>`;
    }
    if (topic.kind === 'waterfall') {
      const perOp = evalA.scopes.find(s => s.scope === 'per_op');
      const scB = evalB ? evalB.winner : (perOp || evalA.runnerUp);
      const cmpLbl = evalB ? `${lblB} (${scB.scope})` : `Baseline (${scB.scope})`;
      const idx = parseInt(topic.sub || '0', 10) || 0;
      const rows = [
        ['Max(Compute, HBM) Envelope', Math.max(sc.compute_us, sc.hbm_us),
          Math.max(scB.compute_us, scB.hbm_us), 'max(t_comp, t_hbm) overlap floor'],
        ['Collective Sync & Bounce', sc.comm_us, scB.comm_us,
          'AllReduce + AllToAll + PP link hops (knob: #ctl-coll-hop)'],
        ['Cold Pipeline Fill/Drain', sc.fill_drain_us, scB.fill_drain_us,
          'Cold HBM slot fill + epilogue drain across fused boundaries'],
        ['Multi-Pass HBM Repass Spill', sc.repass_spill_us, scB.repass_spill_us,
          `n_splits=${sc.n_splits} vs ${scB.n_splits} (knob: #ctl-vmem)`],
        ['Host Kernel Launch Gaps', sc.launch_us, scB.launch_us,
          `ops=${sc.ops_per_step} vs ${scB.ops_per_step} * gap (#ctl-launch-gap)`]
      ];
      const r = rows[idx] || rows[0];
      const d = r[1] - r[2], sgn = d >= 0 ? '+' : '';
      return `<div class="wb-deriv-box"><div style="display:flex;` +
        `justify-content:space-between;"><strong>Waterfall: ${esc(r[0])}</strong>` +
        `<span class="prov-tag prov-tag-DERIVED">DERIVED</span></div>` +
        `<div class="wb-formula">[A] ${sc.scope}: ${r[1].toFixed(2)} us\n` +
        `[B] ${esc(cmpLbl)}: ${r[2].toFixed(2)} us\n` +
        `Delta ([A] - [B]): ${sgn}${d.toFixed(2)} us\n${esc(r[3])}</div></div>`;
    }
    return '';
  };

  function buildLiveBucketSchedule(wlBase) {
    const intervals = [];
    let cur = null;
    for (const tVal of PHASE_T_GRID) {
      const wl = Object.assign({}, wlBase, {
        phase: 'decode', batch: tVal, seqTokens: 1,
        moeMode: tVal >= 32 ? 'grouped_on_chip' : 'masked_on_chip'
      });
      const res = window.__WB_API.evaluateWorkload(wl);
      const win = res.winner;
      if (cur && cur.recommended_scope === win.scope &&
          cur.recommended_moe_mode === res.moeModeRec) {
        cur.max_active_tokens = tVal;
        cur.step_bound_us = win.step_bound_us;
      } else {
        cur = {
          phase: 'decode',
          layout: document.getElementById('lbl-layout').textContent,
          min_active_tokens: tVal,
          max_active_tokens: tVal,
          effective_context: wlBase.context,
          recommended_scope: win.scope,
          recommended_moe_mode: res.moeModeRec,
          max_vmem_tokens: win.t_vmem_max,
          num_vmem_splits: win.n_splits,
          step_bound_us: win.step_bound_us
        };
        intervals.push(cur);
      }
    }
    return {
      schema: 'advisor.bucket_schedule.v1',
      consumer_status: 'advisory_offline_v1',
      model: DATA.model, chip: DATA.chip,
      default_layout: document.getElementById('lbl-layout').textContent,
      calibrated: STATE.calibrated,
      buckets: intervals
    };
  }

  window.wbCalNotes = function(sc) {
    const esc = window.__WB_API.escH || String;
    const pTag = t => `<span class="prov-tag prov-tag-${t}">${t}</span>`;
    const line = STATE.calibrated
      ? `${pTag('MEASURED')} Source: ${esc(STATE.calSource)}`
      : `${pTag('ASSUMED')} Settle: calibrate --trace &lt;p&gt;`;
    const ratio = STATE.calStepTimeUs
      ? (sc.step_bound_us / Math.max(0.1, STATE.calStepTimeUs) * 100).toFixed(1) : '';
    const anc = STATE.calStepTimeUs !== null
      ? '<div class="wb-deriv-box"><strong>Measured Step:</strong> ' +
        `<span class="tabular">${STATE.calStepTimeUs.toFixed(1)} us</span> ` +
        `(${ratio}%)</div>`
      : '';
    return { line, anc };
  };

  window.wbSelectPhaseCell = function(tVal, sVal) {
    STATE.customDirty = true;
    document.getElementById('ctl-phase').value = 'decode';
    document.getElementById('ctl-batch').value = String(tVal);
    document.getElementById('ctl-seq-tokens').value = '1';
    document.getElementById('ctl-context').value = String(sVal);
    window.wbSetStageTab('arena');
    window.wbRenderAll();
  };

  window.wbRenderPhaseExtension = function(wlA, evalA) {
    const mapEl = document.getElementById('wb-phase-map-svg');
    const preEl = document.getElementById('wb-bucket-schedule-pre');
    if (!mapEl || !preEl) return;
    const colors = window.__WB_API.SCOPE_COLORS || {};
    const W = 760, padL = 72, padT = 28, cellW = 64, cellH = 26;
    const actT = evalA ? evalA.activeTokens : wlA.batch;
    let cells = '';
    PHASE_S_GRID.forEach((sVal, rIdx) => {
      const y = padT + (PHASE_S_GRID.length - 1 - rIdx) * cellH;
      cells += `<text x="${padL - 8}" y="${y + 17}" text-anchor="end" ` +
        `fill="var(--muted)" font-size="11" class="tabular">` +
        `${sVal / 1024}K</text>`;
      PHASE_T_GRID.forEach((tVal, cIdx) => {
        const x = padL + cIdx * cellW;
        const wl = Object.assign({}, wlA, {
          phase: 'decode', batch: tVal, seqTokens: 1, context: sVal,
          moeMode: tVal >= 32 ? 'grouped_on_chip' : 'masked_on_chip'
        });
        const ev = window.__WB_API.evaluateWorkload(wl);
        const col = colors[ev.winner.scope] || '#64748B';
        const shortSc = ev.winner.scope === 'whole_step' ? 'WS'
          : (ev.winner.scope === 'layer_group' ? 'LG'
          : (ev.winner.scope === 'single_layer' ? '1L' : 'OP'));
        const tieMark = ev.marginPct < 5.0 ? '*' : '';
        const isAct = tVal === actT && sVal === wlA.context;
        const stk = isAct ? ' stroke="#FFFFFF" stroke-width="2.2"' : '';
        const esc = window.__WB_API.escH || String;
        const tip = esc(`T=${tVal}, S=${sVal / 1024}K -> ${ev.winner.scope} ` +
          `(${ev.moeModeRec}), floor=${ev.winner.step_bound_us} us`);
        cells += `<g class="wb-clickable" ` +
          `onclick="wbSelectPhaseCell(${tVal},${sVal})">` +
          `<rect x="${x + 1}" y="${y + 1}" width="${cellW - 2}" ` +
          `height="${cellH - 2}" rx="3" fill="${col}" fill-opacity="0.84"${stk}>` +
          `<title>${tip}</title></rect>` +
          `<text x="${x + cellW / 2}" y="${y + 17}" text-anchor="middle" ` +
          `fill="#FFFFFF" font-size="10" font-weight="700">` +
          `${shortSc}${tieMark}</text></g>`;
      });
    });
    const yTick = padT + PHASE_S_GRID.length * cellH + 16;
    const xTicks = PHASE_T_GRID.map((tVal, cIdx) =>
      `<text x="${padL + cIdx * cellW + cellW / 2}" y="${yTick}" ` +
      `text-anchor="middle" fill="var(--muted)" font-size="11" ` +
      `class="tabular">${tVal}</text>`
    ).join('');
    const svgH = padT + PHASE_S_GRID.length * cellH + 38;
    mapEl.innerHTML = `<svg viewBox="0 0 ${W} ${svgH}" class="chart-svg" ` +
      `role="img" aria-label="2D Winning-Scope Phase Map">` +
      `<text x="${padL}" y="16" fill="var(--muted)" font-size="11">` +
      `Y: Context Length (tokens) | X: Active Tokens per Step (T) | ` +
      `* = close margin (&lt;5%)</text>${cells}${xTicks}</svg>`;
    preEl.textContent = JSON.stringify(buildLiveBucketSchedule(wlA), null, 2);
  };

  function applyCalibrationObject(obj, fileName) {
    if (!obj || obj.schema !== 'advisor.measurements.v1') {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Rejected: expected advisor.measurements.v1';
      return;
    }
    if (obj.launch_gap_us && typeof obj.launch_gap_us.p50 === 'number') {
      const p50 = obj.launch_gap_us.p50;
      document.getElementById('ctl-launch-gap').value = String(p50);
      STATE.launchGapHiUs = typeof obj.launch_gap_us.p90 === 'number'
        ? Math.max(p50, obj.launch_gap_us.p90) : p50;
    }
    if (Array.isArray(obj.collectives) && obj.collectives.length > 0) {
      let minHop = null;
      for (const c of obj.collectives) {
        if (c && typeof c.latency_us === 'number') {
          const grp = Math.max(2, parseInt(c.group_size || 2, 10));
          const hopUs = c.latency_us / Math.max(1, Math.round(Math.log2(grp)));
          if (minHop === null || hopUs < minHop) minHop = hopUs;
        }
      }
      if (minHop !== null) {
        const cVal = Number(minHop.toFixed(3));
        document.getElementById('ctl-coll-hop').value = String(cVal);
        STATE.collHopHiUs = cVal;
      }
    }
    if (obj.ring_plan && typeof obj.ring_plan === 'object') {
      const rp = obj.ring_plan;
      if (rp.dense_slots) {
        document.getElementById('ctl-dma').value = String(rp.dense_slots);
      }
      if (rp.dense_slot_bytes) STATE.denseSlotBytes = rp.dense_slot_bytes;
      if (rp.expert_slot_bytes) STATE.expertSlotBytes = rp.expert_slot_bytes;
    }
    const stRaw = obj.step_time_us;
    STATE.calStepTimeUs = (stRaw && typeof stRaw.p50 === 'number') ? stRaw.p50
      : (typeof stRaw === 'number' ? stRaw : null);
    STATE.calibrated = true;
    STATE.hwDirty = true;
    STATE.calSource = (obj.source && obj.source.file)
      ? obj.source.file : fileName;
    const st = document.getElementById('wb-cal-status');
    if (st) st.textContent = 'Calibrated (MEASURED): ' + STATE.calSource;
    window.wbRenderAll();
  }
  function traceEventsToMeas(raw, fileName) {
    const evs = Array.isArray(raw) ? raw
      : (raw && Array.isArray(raw.traceEvents) ? raw.traceEvents : null);
    if (!evs) return raw;
    const spans = evs.filter(e => e && (!e.ph || e.ph === 'X') &&
      typeof e.ts === 'number' && typeof e.dur === 'number' && e.dur > 0 &&
      !/step/i.test(String(e.name || '')) && !/step/i.test(String(e.cat || '')))
      .sort((a, b) => a.ts - b.ts);
    const gaps = [];
    for (let i = 1; i < spans.length; i++) {
      const g = spans[i].ts - (spans[i - 1].ts + spans[i - 1].dur);
      if (g > 0 && g <= 500) gaps.push(g);
    }
    gaps.sort((a, b) => a - b);
    const q = p => {
      if (!gaps.length) return 0.0;
      const pos = p * (gaps.length - 1), lo = Math.floor(pos);
      const hi = Math.min(gaps.length - 1, lo + 1), frac = pos - lo;
      return gaps[lo] * (1.0 - frac) + gaps[hi] * frac;
    };
    return {
      schema: 'advisor.measurements.v1',
      source: { file: fileName },
      launch_gap_us: { p50: Number(q(0.5).toFixed(4)), p90: Number(q(0.9).toFixed(4)) }
    };
  }
  async function readCalibrationFile(file) {
    if (!file) return;
    try {
      let text = '';
      if (file.name.endsWith('.gz') && typeof DecompressionStream !== 'undefined') {
        const ds = file.stream().pipeThrough(new DecompressionStream('gzip'));
        text = await new Response(ds).text();
      } else {
        text = await new Promise((res, rej) => {
          const reader = new FileReader();
          reader.onload = ev => res(String((ev.target && ev.target.result) || '{}'));
          reader.onerror = () => rej(new Error('read error'));
          reader.readAsText(file);
        });
      }
      const parsed = traceEventsToMeas(JSON.parse(text), file.name);
      applyCalibrationObject(parsed, file.name);
    } catch (_err) {
      const st = document.getElementById('wb-cal-status');
      if (st) st.textContent = 'Invalid JSON or gzip trace file';
    }
  }
  window.wbOnPickCalibration = function(ev) {
    const files = ev.target && ev.target.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbOnDropCalibration = function(ev) {
    ev.preventDefault();
    const files = ev.dataTransfer && ev.dataTransfer.files;
    if (files && files[0]) readCalibrationFile(files[0]);
  };
  window.wbCopyBucketSchedule = function() {
    const preEl = document.getElementById('wb-bucket-schedule-pre');
    const btn = document.getElementById('btn-copy-bucket-schedule');
    if (preEl && navigator.clipboard) {
      navigator.clipboard.writeText(preEl.textContent || '');
    }
    if (btn) {
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy Bucket Schedule JSON'; }, 1200);
    }
  };
"""
