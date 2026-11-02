/**
 * End-to-end proof: the browser engine reproduces every published 2017 forecast.
 *
 * The torch fixture test proves the layer maths. This proves the whole chain --
 * exported weights, exported series, window slicing, the delta-to-level step --
 * against results/preds_test.csv, the file the report and README quote. If this
 * passes, the demo is running the same model that produced the published score.
 *
 * Run: node tests/test_lstm_fullrun.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { createForecaster, decodeFloat32 } from '../report/demo/lstm.js';

const here = dirname(fileURLToPath(import.meta.url));
const demo = join(here, '../report/demo');
const model = JSON.parse(readFileSync(join(demo, 'model.json'), 'utf-8'));
const series = JSON.parse(readFileSync(join(demo, 'series.json'), 'utf-8'));

const TOLERANCE = 1e-5;

const forecaster = createForecaster({
  hidden: model.hidden,
  inputSize: model.inputSize,
  layers: model.layers.map((l) => ({
    weightIh: decodeFloat32(l.weightIh), weightHh: decodeFloat32(l.weightHh),
    biasIh: decodeFloat32(l.biasIh), biasHh: decodeFloat32(l.biasHh),
  })),
  head: { weight: decodeFloat32(model.head.weight), bias: decodeFloat32(model.head.bias) },
});

const features = decodeFloat32(series.features);
const raw = decodeFloat32(series.raw);
const rows = new Int32Array(decodeFloat32(series.samples.row).buffer);
const published = decodeFloat32(series.samples.lstm);
const { window: steps, horizonHours } = series;
const channels = series.channelOrder.length;

/** Slice the 72 normalised rows ending at `row` (inclusive), as training did. */
function windowAt(row) {
  const out = [];
  for (let t = row - steps + 1; t <= row; t += 1) {
    out.push(features.subarray(t * model.inputSize, (t + 1) * model.inputSize));
  }
  return out;
}

let worst = 0;
let worstAt = -1;
const started = performance.now();
for (let s = 0; s < rows.length; s += 1) {
  const row = rows[s];
  const anchor = raw[row * channels];           // observed log10 flux at issue time
  const forecast = anchor + forecaster.predictDelta(windowAt(row));
  const error = Math.abs(forecast - published[s]);
  if (error > worst) { worst = error; worstAt = s; }
}
const elapsed = performance.now() - started;

console.log(`samples          : ${rows.length.toLocaleString()}`);
console.log(`horizon          : ${horizonHours} h, window ${steps} steps`);
console.log(`worst |diff|     : ${worst.toExponential(3)} (sample ${worstAt})`);
console.log(`per-forecast cost: ${(elapsed / rows.length).toFixed(3)} ms`);

// A demo that ignored its inputs would still match if the data were constant,
// so confirm the forecasts actually vary across the year.
const spread = Math.max(...published) - Math.min(...published);
console.log(`forecast spread  : ${spread.toFixed(3)} log10 units`);

if (worst >= TOLERANCE || !(spread > 1)) {
  console.error('FAILED');
  process.exit(1);
}
console.log(`\nPASS all ${rows.length.toLocaleString()} published forecasts reproduced within ${TOLERANCE}`);
