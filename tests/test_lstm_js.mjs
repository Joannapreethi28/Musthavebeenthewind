/**
 * Prove report/demo/lstm.js reproduces torch's nn.LSTM exactly.
 *
 * Why a separate runner: the demo has no build step and no test framework, and
 * the one bug that matters here (gate order / weight layout) is invisible by
 * inspection. Regenerate the fixture with `py -3.14 scripts/make_lstm_fixture.py`.
 *
 * Run: node tests/test_lstm_js.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { createForecaster, decodeFloat32 } from '../report/demo/lstm.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixtures/lstm_reference.json'), 'utf-8'));

// JS accumulates in float64 where torch uses float32, so an exact bit match is
// not achievable; 1e-5 is far tighter than any gate-order error could survive.
const TOLERANCE = 1e-5;

const model = {
  hidden: fixture.hidden,
  inputSize: fixture.inputSize,
  layers: fixture.layers.map((layer) => ({
    weightIh: decodeFloat32(layer.weightIh),
    weightHh: decodeFloat32(layer.weightHh),
    biasIh: decodeFloat32(layer.biasIh),
    biasHh: decodeFloat32(layer.biasHh),
  })),
  head: { weight: decodeFloat32(fixture.head.weight), bias: decodeFloat32(fixture.head.bias) },
};

const forecaster = createForecaster(model);

/** Split a flat (steps * inputSize) buffer into per-timestep rows. */
function toWindow(flat, steps, inputSize) {
  const rows = [];
  for (let t = 0; t < steps; t += 1) rows.push(flat.subarray(t * inputSize, (t + 1) * inputSize));
  return rows;
}

let worst = 0;
let failures = 0;
fixture.cases.forEach((testCase, index) => {
  const window = toWindow(decodeFloat32(testCase.window), fixture.steps, fixture.inputSize);
  const actual = forecaster.predictDelta(window);
  const error = Math.abs(actual - testCase.delta);
  worst = Math.max(worst, error);
  const ok = error < TOLERANCE;
  if (!ok) failures += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'} case ${index}  torch=${testCase.delta.toFixed(8)}  js=${actual.toFixed(8)}  |diff|=${error.toExponential(2)}`);
});

// A model whose output ignores part of the window would still pass the cases
// above if that part happened not to matter; perturbing the first timestep
// catches an implementation that silently drops early history.
const probe = toWindow(decodeFloat32(fixture.cases[0].window), fixture.steps, fixture.inputSize);
const base = forecaster.predictDelta(probe);
const mutated = probe.map((row, t) => (t === 0 ? Float32Array.from(row, (v) => v + 1) : row));
const shifted = forecaster.predictDelta(mutated);
const reacts = Math.abs(shifted - base) > 1e-9;
console.log(`${reacts ? 'PASS' : 'FAIL'} first timestep influences output (delta moved by ${(shifted - base).toExponential(2)})`);
if (!reacts) failures += 1;

console.log(`\nworst |diff| = ${worst.toExponential(2)} over ${fixture.cases.length} cases (torch ${fixture.torch_version})`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('ALL CHECKS PASSED');
