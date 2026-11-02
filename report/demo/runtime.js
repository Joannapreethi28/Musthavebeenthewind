/**
 * Shared demo runtime: decode the exported series, build perturbed windows, and
 * run the frozen forecaster over them.
 *
 * Why perturbation lives here rather than in each page: the physical-to-z-score
 * conversion and the off-manifold test are the two places a demo like this
 * quietly becomes dishonest, so both are written once and reused.
 */
import { createForecaster, decodeFloat32 } from './lstm.js';

export const CHANNEL_LABELS = ['Electron flux', 'Solar-wind speed', 'IMF Bz (GSM)', 'Dst index'];
export const CHANNEL_UNITS = ['log₁₀ e/cm²/s/sr', 'km/s', 'nT', 'nT'];

/** Decode the base64 payloads once; every interaction reads these views. */
export function loadSeries(series) {
  const channels = series.channelOrder.length;
  return {
    meta: series,
    channels,
    features: decodeFloat32(series.features),
    raw: decodeFloat32(series.raw),
    lt: decodeFloat32(series.lt),
    rows: new Int32Array(decodeFloat32(series.samples.row).buffer),
    obs: decodeFloat32(series.samples.obs),
    persistence: decodeFloat32(series.samples.persistence),
    persistenceDiurnal: decodeFloat32(series.samples.persistenceDiurnal),
    lstm: decodeFloat32(series.samples.lstm),
    mean: series.normalization.mean,
    std: series.normalization.std,
    trainStats: series.trainStats,
    startTime: Date.parse(series.startTimeUtc + ':00:00Z'),
  };
}

export function buildForecaster(model) {
  return createForecaster({
    hidden: model.hidden,
    inputSize: model.inputSize,
    layers: model.layers.map((l) => ({
      weightIh: decodeFloat32(l.weightIh), weightHh: decodeFloat32(l.weightHh),
      biasIh: decodeFloat32(l.biasIh), biasHh: decodeFloat32(l.biasHh),
    })),
    head: { weight: decodeFloat32(model.head.weight), bias: decodeFloat32(model.head.bias) },
  });
}

/** UTC timestamp of a series row (rows are hour starts). */
export function rowTime(data, row) {
  return new Date(data.startTime + row * 3600000);
}

export const formatUtc = (date) => date.toISOString().slice(0, 16).replace('T', ' ') + 'Z';

/**
 * Physical value of one channel at one row, after applying an offset.
 * The exported features are already z-scored, so a physical offset enters as
 * offset / std -- doing it the other way round is the classic sign this kind of
 * control was never checked against the real pipeline.
 */
export function physical(data, row, channel, offset = 0) {
  return data.features[row * 6 + channel] * data.std[channel] + data.mean[channel] + offset;
}

/**
 * Materialise the 72 normalised rows ending at `row`, shifted by `offsets`
 * (physical units, one per continuous channel). sin/cos of local time are
 * untouched: they are calendar facts, not something a scientist can perturb.
 */
export function windowAt(data, row, offsets) {
  const { features, std } = data;
  const steps = data.meta.window;
  const shift = offsets.map((offset, c) => offset / std[c]);
  const out = [];
  for (let t = row - steps + 1; t <= row; t += 1) {
    const base = t * 6;
    const step = new Float64Array(6);
    for (let c = 0; c < 4; c += 1) step[c] = features[base + c] + shift[c];
    step[4] = features[base + 4];
    step[5] = features[base + 5];
    out.push(step);
  }
  return out;
}

/**
 * Forecast at a row. Returns the level in log10 flux, because that is the unit
 * every published number in results/ is quoted in.
 */
export function forecastAt(data, forecaster, row, offsets) {
  const delta = forecaster.predictDelta(windowAt(data, row, offsets));
  const anchor = data.raw[row * data.channels] + offsets[0];
  return { delta, anchor, level: anchor + delta };
}

/**
 * Which channels have been pushed outside anything the model saw in training.
 * Checked across the whole window, not just the anchor hour, because the LSTM
 * reads all 72 steps.
 */
export function offManifold(data, row, offsets) {
  const steps = data.meta.window;
  const flagged = [];
  for (let c = 0; c < 4; c += 1) {
    if (offsets[c] === 0) continue;
    const stats = data.trainStats[c];
    let low = Infinity;
    let high = -Infinity;
    for (let t = row - steps + 1; t <= row; t += 1) {
      const value = physical(data, t, c, offsets[c]);
      if (value < low) low = value;
      if (value > high) high = value;
    }
    if (low < stats.min || high > stats.max) {
      flagged.push({ channel: c, low, high, stats });
    }
  }
  return flagged;
}

/** Index of the sample whose anchor is `row`, or -1 when the hour was excluded. */
export function sampleAt(data, row) {
  let lo = 0;
  let hi = data.rows.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (data.rows[mid] === row) return mid;
    if (data.rows[mid] < row) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

// No PE is computed in the browser on purpose: the exported predictions are
// float32, so a score recomputed here would differ in the last digits from the
// published 0.815. Scores belong to results/metrics.json, not to this page.
