// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * Hermetic gjs/node runner for workbench.js UI state tests.
 */

let readFile, writeFile, args, getDir;

if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  const GLib = imports.gi.GLib;
  readFile = function (filePath) {
    const [ok, contents] = GLib.file_get_contents(filePath);
    if (!ok) {
      throw new Error("Could not read file: " + filePath);
    }
    return new TextDecoder("utf-8").decode(contents);
  };
  writeFile = function (filePath, data) {
    GLib.file_set_contents(filePath, data);
  };
  args = ARGV;
  getDir = function (p) {
    const idx = p.lastIndexOf("/");
    return idx >= 0 ? p.slice(0, idx) : ".";
  };
} else if (
  typeof process !== "undefined" &&
  process.versions &&
  process.versions.node
) {
  const fs = require("fs");
  const path = require("path");
  readFile = p => fs.readFileSync(p, "utf-8");
  writeFile = (p, d) => fs.writeFileSync(p, d, "utf-8");
  args = process.argv.slice(2);
  getDir = p => path.dirname(p);
} else {
  throw new Error("Unsupported JS runtime: require gjs or node");
}

if (args.length < 2) {
  throw new Error("Usage: advisor_workbench_runner.js <in_json> <out_json>");
}

function makeClassList() {
  const set = new Set();
  return {
    add: cls => set.add(cls),
    remove: cls => set.delete(cls),
    toggle: (cls, force) => {
      const on = force !== undefined ? Boolean(force) : !set.has(cls);
      if (on) set.add(cls); else set.delete(cls);
      return on;
    },
    contains: cls => set.has(cls),
    toArray: () => Array.from(set)
  };
}

function makeElem(id, initVal, opts) {
  const options = (opts || []).map(v => ({ value: String(v), textContent: String(v) }));
  let val = initVal !== undefined ? String(initVal) : "";
  return {
    id,
    tagName: opts ? "SELECT" : "DIV",
    min: "1",
    max: id === "ctl-batch" ? "1024" : "1048576",
    disabled: false,
    textContent: "",
    innerHTML: "",
    className: "",
    style: {},
    options,
    classList: makeClassList(),
    appendChild(child) {
      this.options.push(child);
    },
    get value() {
      return val;
    },
    set value(next) {
      let s = String(next);
      if (id === "ctl-batch") {
        const num = Number(s);
        const hi = Number(this.max || 1024);
        if (!Number.isNaN(num)) {
          s = String(Math.min(hi, Math.max(1, num)));
        }
      }
      val = s;
    }
  };
}

function buildHarnessDom(seedData) {
  const vmem = String(seedData.vmem_budget_mib || 62.0);
  const meshVals = [1, 2, 4, 8, 16, 32, 64];
  const elems = new Map();
  const reg = (id, v, opts) => elems.set(id, makeElem(id, v, opts));
  reg("ctl-vmem", vmem);
  reg("ctl-dma", String(seedData.dma_buffers || 2), ["2", "3"]);
  reg("ctl-alias", "1", ["1", "0"]);
  reg("ctl-launch-gap", "1.00");
  reg("ctl-coll-hop", "1.00");
  reg("ctl-eta-dense", "0.72");
  reg("ctl-eta-moe", "0.50");
  reg("ctl-eta-hbm", "0.82");
  reg("ctl-eta-ici", "0.75");
  reg("ctl-phase", "decode", ["decode", "verify", "prefill", "mixed"]);
  reg("ctl-layout", "0", ["0"]);
  reg("ctl-batch", "16");
  reg("ctl-seq-tokens", "1");
  reg("ctl-context", "4096");
  reg("ctl-arena-scope", "whole_step", [
    "whole_step", "layer_group", "single_layer", "per_op"
  ]);
  for (const ax of ["tp", "ep", "dp", "pp", "dcp", "pcp"]) {
    reg("ctl-" + ax, ax === "tp" ? "8" : "1", meshVals);
    reg("lbl-" + ax, ax === "tp" ? "8" : "1");
  }
  for (const id of [
    "lbl-phase", "lbl-batch", "lbl-seq-tokens", "lbl-context", "lbl-vmem",
    "lbl-dma", "lbl-alias", "lbl-launch-gap", "lbl-coll-hop", "lbl-dispatch-regime",
    "lbl-eta-dense", "lbl-eta-moe", "lbl-eta-hbm", "lbl-eta-ici", "lbl-layout",
    "wb-active-layout", "wb-verdict-pill", "wb-verdict-summary", "wb-invalid-banner",
    "wb-ab-status", "wb-regime-list", "wb-vmem-arena-svg", "wb-scope-timeline-svg",
    "wb-waterfall-svg", "wb-waterfall-title", "wb-waterfall-summary",
    "wb-deriv-drawer", "wb-kpi-scope", "wb-kpi-floor", "wb-kpi-tmax", "wb-kpi-tps",
    "btn-swap-ab", "btn-vs-perop", "btn-reset-all", "btn-dr-zero", "btn-dr-xla",
    "btn-dr-eager", "view-workbench", "view-audit", "btn-view-workbench",
    "btn-view-audit", "wb-stage-arena", "wb-stage-phase", "btn-stage-arena",
    "btn-stage-phase", "wb-extra-rail-slot", "wb-extra-stage-slot",
    "wb-phase-map-svg", "wb-bucket-schedule-pre", "btn-copy-bucket-schedule",
    "wb-cal-status"
  ]) {
    if (!elems.has(id)) reg(id, "");
  }
  const doc = {
    body: { classList: makeClassList() },
    getElementById(id) {
      if (!elems.has(id)) reg(id, "");
      return elems.get(id);
    },
    createElement(tag) {
      return { tagName: String(tag).toUpperCase(), value: "", textContent: "" };
    }
  };
  return { doc, elems };
}

const payload = JSON.parse(readFile(args[0]));
const assetsDir = getDir(payload.engine_path);
if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  imports.searchPath.unshift(assetsDir);
}

const { doc, elems } = buildHarnessDom(payload.seed_data || {});
globalThis.window = globalThis;
globalThis.document = doc;

if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  imports.workbench_engine;
} else {
  require(payload.engine_path);
}
const createEngine = globalThis.__MK_CREATE_ADVISOR_ENGINE;
globalThis.__MK_ADVISOR_ENGINE = createEngine(payload.seed_data || {});

if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  imports.workbench;
} else {
  require(assetsDir + "/workbench.js");
}

window.wbRenderAll();

const snapshots = [];
function snap(tag) {
  const api = window.__WB_API || {};
  const wlA = api.currentWorkloadA ? api.currentWorkloadA() : {};
  const evA = api.evaluateWorkload ? api.evaluateWorkload(wlA) : {};
  snapshots.push({
    tag,
    pinA: api.STATE ? api.STATE.pinA : -99,
    pinB: api.STATE ? api.STATE.pinB : -99,
    customDirty: Boolean(api.STATE && api.STATE.customDirty),
    selectedScope: api.STATE && api.STATE.selectedTopic
      ? api.STATE.selectedTopic.scope : "",
    ctlBatch: elems.get("ctl-batch").value,
    ctlPhase: elems.get("ctl-phase").value,
    ctlSeqTokens: elems.get("ctl-seq-tokens").value,
    seqDisabled: Boolean(elems.get("ctl-seq-tokens").disabled),
    lblSeqTokens: elems.get("lbl-seq-tokens").textContent,
    ctlArenaScope: elems.get("ctl-arena-scope").value,
    swapDisabled: Boolean(elems.get("btn-swap-ab").disabled),
    activeTokens: evA.activeTokens || 0,
    winnerScope: evA.winner ? evA.winner.scope : "",
    verdictPill: elems.get("wb-verdict-pill").textContent,
    verdictSummary: elems.get("wb-verdict-summary").innerHTML,
    waterfallSvg: elems.get("wb-waterfall-svg").innerHTML,
    derivDrawer: elems.get("wb-deriv-drawer").innerHTML
  });
}

snap("initial");
for (const step of payload.steps || []) {
  const op = step.op;
  if (op === "pin_regime") {
    window.wbPinRegime(step.idx, step.slot);
  } else if (op === "set_batch") {
    elems.get("ctl-batch").value = String(step.value);
    window.wbOnControl();
  } else if (op === "swap_ab") {
    window.wbSwapAB();
  } else if (op === "clear_b") {
    window.wbClearB();
  } else if (op === "set_arena_scope") {
    elems.get("ctl-arena-scope").value = String(step.scope);
    if (window.wbOnArenaScopeChange) {
      window.wbOnArenaScopeChange();
    } else {
      window.wbRenderAll();
    }
  } else if (op === "set_phase") {
    elems.get("ctl-phase").value = String(step.phase);
    window.wbOnPhaseChange();
  } else if (op === "set_seq_tokens") {
    elems.get("ctl-seq-tokens").value = String(step.value);
    window.wbOnSeqChange();
  } else if (op === "select_topic") {
    window.wbSelectTopic(step.kind, step.scope, step.sub);
  }
  snap(step.tag || op);
}

writeFile(args[1], JSON.stringify({ snapshots }));
