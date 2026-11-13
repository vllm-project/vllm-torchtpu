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
 * CLI runner for megakernel advisor workbench JS engine parity testing.
 * Supports both /usr/bin/gjs and node without eval or mocked DOM.
 */

let readFile, writeFile, args, getEngineDir;

if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  const GLib = imports.gi.GLib;
  readFile = function (filePath) {
    const [ok, contents] = GLib.file_get_contents(filePath);
    if (!ok) {
      throw new Error("Could not read file: " + filePath);
    }
    const decoder = new TextDecoder("utf-8");
    return decoder.decode(contents);
  };
  writeFile = function (filePath, data) {
    GLib.file_set_contents(filePath, data);
  };
  args = ARGV;
  getEngineDir = function (p) {
    const lastSlash = p.lastIndexOf("/");
    return lastSlash >= 0 ? p.slice(0, lastSlash) : ".";
  };
} else if (
  typeof process !== "undefined" &&
  process.versions &&
  process.versions.node
) {
  const fs = require("fs");
  const path = require("path");
  readFile = function (filePath) {
    return fs.readFileSync(filePath, "utf-8");
  };
  writeFile = function (filePath, data) {
    fs.writeFileSync(filePath, data, "utf-8");
  };
  args = process.argv.slice(2);
  getEngineDir = function (p) {
    return path.dirname(p);
  };
} else {
  throw new Error("Unsupported JS runtime: must be run under gjs or node");
}

if (args.length < 2) {
  throw new Error("Usage: advisor_engine_runner.js <in_json> <out_json>");
}

const inPath = args[0];
const outPath = args[1];

const inPayload = JSON.parse(readFile(inPath));
const enginePath = inPayload.engine_path;
if (!enginePath) {
  throw new Error("Missing engine_path in input payload");
}

let createAdvisorEngine;
if (typeof imports !== "undefined" && typeof imports.gi !== "undefined") {
  const engineDir = getEngineDir(enginePath);
  imports.searchPath.unshift(engineDir);
  imports.workbench_engine;
  createAdvisorEngine = globalThis.__MK_CREATE_ADVISOR_ENGINE;
} else {
  const mod = require(enginePath);
  createAdvisorEngine =
    mod.createAdvisorEngine || globalThis.__MK_CREATE_ADVISOR_ENGINE;
}

if (typeof createAdvisorEngine !== "function") {
  throw new Error("createAdvisorEngine was not exported by " + enginePath);
}

const seedData = inPayload.seed_data || {};
const engine = createAdvisorEngine(seedData);

const results = [];
for (const sc of inPayload.items) {
  let evalRes = null;
  if (sc.params) {
    evalRes = engine.evaluateScenario(sc.params);
  }
  results.push({ eval: evalRes });
}

writeFile(outPath, JSON.stringify(results));
