/**
 * Pure-JS forward pass for the frozen 2-layer LSTM forecaster.
 *
 * Why this exists: the demo pages must run the *same* model that produced the
 * published 2017 scores, with no backend and no build step. Re-deriving the
 * forward pass by hand is the only way to do that offline, so every layout
 * assumption below is pinned to PyTorch's documented memory layout and checked
 * against torch itself in tests/test_lstm_js.mjs.
 *
 * PyTorch packs the four gates of nn.LSTM into one (4*hidden, in) matrix in the
 * order i, f, g, o -- NOT i, f, o, g. Getting this wrong still produces smooth,
 * plausible-looking forecasts, which is exactly why it is tested rather than
 * eyeballed. Dropout is a no-op at inference, so nothing here is stochastic.
 */

/** Decode a base64 string written by export_weights.py into float32 values. */
export function decodeFloat32(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/**
 * One nn.LSTM layer. Holds weights as flat rows so a step is a plain dot
 * product; at 72 steps x 64 hidden units the naive loop is already far below a
 * frame, so clarity beats any blocking trick here.
 */
class Layer {
  constructor({ weightIh, weightHh, biasIh, biasHh }, inputSize, hidden) {
    this.weightIh = weightIh;
    this.weightHh = weightHh;
    this.biasIh = biasIh;
    this.biasHh = biasHh;
    this.inputSize = inputSize;
    this.hidden = hidden;
    const expectedIh = 4 * hidden * inputSize;
    const expectedHh = 4 * hidden * hidden;
    if (weightIh.length !== expectedIh) throw new Error(`weight_ih expected ${expectedIh}, got ${weightIh.length}`);
    if (weightHh.length !== expectedHh) throw new Error(`weight_hh expected ${expectedHh}, got ${weightHh.length}`);
    if (biasIh.length !== 4 * hidden || biasHh.length !== 4 * hidden) throw new Error('bias length mismatch');
  }

  /** Advance one timestep, writing the new state into h and c in place. */
  step(x, h, c) {
    const { weightIh, weightHh, biasIh, biasHh, inputSize, hidden } = this;
    for (let unit = 0; unit < hidden; unit += 1) {
      // Gate rows live at offsets 0, hidden, 2*hidden, 3*hidden => i, f, g, o.
      let gateI = biasIh[unit] + biasHh[unit];
      let gateF = biasIh[hidden + unit] + biasHh[hidden + unit];
      let gateG = biasIh[2 * hidden + unit] + biasHh[2 * hidden + unit];
      let gateO = biasIh[3 * hidden + unit] + biasHh[3 * hidden + unit];
      const rowI = unit * inputSize;
      const rowF = (hidden + unit) * inputSize;
      const rowG = (2 * hidden + unit) * inputSize;
      const rowO = (3 * hidden + unit) * inputSize;
      for (let j = 0; j < inputSize; j += 1) {
        const value = x[j];
        gateI += weightIh[rowI + j] * value;
        gateF += weightIh[rowF + j] * value;
        gateG += weightIh[rowG + j] * value;
        gateO += weightIh[rowO + j] * value;
      }
      const hRowI = unit * hidden;
      const hRowF = (hidden + unit) * hidden;
      const hRowG = (2 * hidden + unit) * hidden;
      const hRowO = (3 * hidden + unit) * hidden;
      for (let j = 0; j < hidden; j += 1) {
        const value = h[j];
        gateI += weightHh[hRowI + j] * value;
        gateF += weightHh[hRowF + j] * value;
        gateG += weightHh[hRowG + j] * value;
        gateO += weightHh[hRowO + j] * value;
      }
      const cell = sigmoid(gateF) * c[unit] + sigmoid(gateI) * Math.tanh(gateG);
      c[unit] = cell;
      this.nextH[unit] = sigmoid(gateO) * Math.tanh(cell);
    }
    h.set(this.nextH);
  }
}

/**
 * Build a forecaster from decoded weights.
 *
 * `model.layers` is ordered l0, l1; `model.head` is the Linear(hidden -> 1)
 * applied to the final layer's hidden state at the LAST timestep only, matching
 * LSTMForecaster.forward in src/models.py.
 */
export function createForecaster(model) {
  const { hidden, inputSize } = model;
  const layers = model.layers.map((spec, index) => {
    const layer = new Layer(spec, index === 0 ? inputSize : hidden, hidden);
    layer.nextH = new Float64Array(hidden);
    return layer;
  });

  /**
   * Run one normalised window and return the predicted delta in log10 units.
   * `window` is `steps` rows of `inputSize` features, already z-scored with the
   * frozen training statistics; the caller owns that conversion.
   */
  function predictDelta(window) {
    if (window.length === 0) throw new Error('empty window');
    if (window[0].length !== inputSize) throw new Error(`expected ${inputSize} features per step`);
    const states = layers.map(() => ({ h: new Float64Array(hidden), c: new Float64Array(hidden) }));
    for (let t = 0; t < window.length; t += 1) {
      let input = window[t];
      for (let l = 0; l < layers.length; l += 1) {
        layers[l].step(input, states[l].h, states[l].c);
        input = states[l].h;
      }
    }
    const final = states[states.length - 1].h;
    let delta = model.head.bias[0];
    for (let j = 0; j < hidden; j += 1) delta += model.head.weight[j] * final[j];
    return delta;
  }

  return { predictDelta, hidden, inputSize };
}
