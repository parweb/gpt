/**
 * The most atomic way to train and inference a GPT in pure, dependency-free TypeScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * Adapted from @karpathy's microgpt.py
 */

import * as fs from "fs";
import * as https from "https";

// --- Seeded PRNG (Mulberry32) ---
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);

function gaussRandom(): number {
  let u: number, v: number, s: number;
  do {
    u = rand() * 2 - 1;
    v = rand() * 2 - 1;
    s = u * u + v * v;
  } while (s >= 1 || s === 0);
  return u * Math.sqrt((-2 * Math.log(s)) / s);
}

function shuffle<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// --- Value (autograd scalar) ---
class Value {
  data: number;
  grad: number;
  _backward: () => void;
  _prev: Set<Value>;
  _op: string;

  constructor(data: number, children: Value[] = [], op = "") {
    this.data = data;
    this.grad = 0;
    this._backward = () => {};
    this._prev = new Set(children);
    this._op = op;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    out._backward = () => {
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  pow(n: number): Value {
    const out = new Value(this.data ** n, [this], `**${n}`);
    out._backward = () => {
      this.grad += n * this.data ** (n - 1) * out.grad;
    };
    return out;
  }

  log(): Value {
    const out = new Value(Math.log(this.data), [this], "log");
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const out = new Value(Math.exp(this.data), [this], "exp");
    out._backward = () => {
      this.grad += out.data * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], "ReLU");
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        for (const child of v._prev) {
          buildTopo(child);
        }
        topo.push(v);
      }
    };
    buildTopo(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  neg(): Value {
    return this.mul(-1);
  }
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}

// --- Hyperparameters ---
const n_embd = 16;
const n_head = 4;
const n_layer = 1;
const block_size = 8;
const head_dim = n_embd / n_head;

// --- Helpers ---
type Matrix = Value[][];

function matrix(nout: number, nin: number, std = 0.02): Matrix {
  const m: Matrix = [];
  for (let i = 0; i < nout; i++) {
    const row: Value[] = [];
    for (let j = 0; j < nin; j++) {
      row.push(new Value(gaussRandom() * std));
    }
    m.push(row);
  }
  return m;
}

function linear(x: Value[], w: Matrix): Value[] {
  return w.map((wo) => {
    let sum = wo[0].mul(x[0]);
    for (let i = 1; i < wo.length; i++) {
      sum = sum.add(wo[i].mul(x[i]));
    }
    return sum;
  });
}

function softmax(logits: Value[]): Value[] {
  let maxVal = -Infinity;
  for (const v of logits) {
    if (v.data > maxVal) maxVal = v.data;
  }
  const exps = logits.map((v) => v.sub(maxVal).exp());
  let total = exps[0];
  for (let i = 1; i < exps.length; i++) {
    total = total.add(exps[i]);
  }
  return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  let ms = x[0].mul(x[0]);
  for (let i = 1; i < x.length; i++) {
    ms = ms.add(x[i].mul(x[i]));
  }
  ms = ms.div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// --- GPT forward pass ---
function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  stateDict: Record<string, Matrix>
): Value[] {
  const tokEmb = stateDict["wte"][tokenId];
  const posEmb = stateDict["wpe"][posId];
  let x = tokEmb.map((t, i) => t.add(posEmb[i]));
  x = rmsnorm(x);

  for (let li = 0; li < n_layer; li++) {
    const xResidual = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    const xAttn: Value[] = [];
    for (let h = 0; h < n_head; h++) {
      const hs = h * head_dim;
      const qH = q.slice(hs, hs + head_dim);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + head_dim));
      const vH = values[li].map((vi) => vi.slice(hs, hs + head_dim));

      const attnLogits = kH.map((kHt) => {
        let dot = qH[0].mul(kHt[0]);
        for (let j = 1; j < head_dim; j++) {
          dot = dot.add(qH[j].mul(kHt[j]));
        }
        return dot.div(head_dim ** 0.5);
      });

      const attnWeights = softmax(attnLogits);

      for (let j = 0; j < head_dim; j++) {
        let acc = attnWeights[0].mul(vH[0][j]);
        for (let t = 1; t < vH.length; t++) {
          acc = acc.add(attnWeights[t].mul(vH[t][j]));
        }
        xAttn.push(acc);
      }
    }

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = x.map((a, i) => a.add(xResidual[i]));

    const xResidual2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
    x = x.map((xi) => xi.relu().pow(2));
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = x.map((a, i) => a.add(xResidual2[i]));
  }

  return linear(x, stateDict["lm_head"]);
}

// --- Data loading and training ---
function downloadFile(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        file.close();
        fs.unlinkSync(dest);
        downloadFile(response.headers.location!, dest).then(resolve, reject);
        return;
      }
      response.pipe(file);
      file.on("finish", () => {
        file.close();
        resolve();
      });
    }).on("error", (err) => {
      fs.unlinkSync(dest);
      reject(err);
    });
  });
}

async function main() {
  // Load data
  if (!fs.existsSync("input.txt")) {
    const url =
      "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
    console.log("Downloading names.txt...");
    await downloadFile(url, "input.txt");
  }

  const raw = fs.readFileSync("input.txt", "utf-8");
  const docs = raw
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  // Build vocabulary
  const charSet = new Set<string>();
  for (const doc of docs) {
    for (const ch of doc) {
      charSet.add(ch);
    }
  }
  const chars = ["<BOS>", ...Array.from(charSet).sort()];
  const vocabSize = chars.length;
  const stoi = new Map<string, number>();
  const itos = new Map<number, string>();
  chars.forEach((ch, i) => {
    stoi.set(ch, i);
    itos.set(i, ch);
  });
  const BOS = stoi.get("<BOS>")!;
  console.log(`vocab size: ${vocabSize}`);

  // Build state dict
  const stateDict: Record<string, Matrix> = {
    wte: matrix(vocabSize, n_embd),
    wpe: matrix(block_size, n_embd),
    lm_head: matrix(vocabSize, n_embd),
  };

  for (let i = 0; i < n_layer; i++) {
    stateDict[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
    stateDict[`layer${i}.attn_wo`] = matrix(n_embd, n_embd, 0);
    stateDict[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
    stateDict[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd, 0);
  }

  const params: Value[] = [];
  for (const mat of Object.values(stateDict)) {
    for (const row of mat) {
      for (const p of row) {
        params.push(p);
      }
    }
  }
  console.log(`num params: ${params.length}`);

  // --- Training ---
  const learningRate = 1e-2;
  const beta1 = 0.9;
  const beta2 = 0.95;
  const epsAdam = 1e-8;
  const m = new Float64Array(params.length);
  const v = new Float64Array(params.length);
  const numSteps = 500;

  for (let step = 0; step < numSteps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split("").map((ch) => stoi.get(ch)!), BOS];
    const n = Math.min(block_size, tokens.length - 1);

    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const valuesKV: Value[][][] = Array.from({ length: n_layer }, () => []);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId++) {
      const tokenId = tokens[posId];
      const targetId = tokens[posId + 1];
      const logits = gpt(tokenId, posId, keys, valuesKV, stateDict);
      const probs = softmax(logits);
      const lossT = probs[targetId].log().neg();
      losses.push(lossT);
    }

    let loss = losses[0];
    for (let i = 1; i < losses.length; i++) {
      loss = loss.add(losses[i]);
    }
    loss = loss.div(n);
    loss.backward();

    const lrT = learningRate * (1 - step / numSteps);
    for (let i = 0; i < params.length; i++) {
      const p = params[i];
      m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
      v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad;
      const mHat = m[i] / (1 - beta1 ** (step + 1));
      const vHat = v[i] / (1 - beta2 ** (step + 1));
      p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
      p.grad = 0;
    }

    const stepNum = String(step + 1).padStart(4, " ");
    const totalStr = String(numSteps).padStart(4, " ");
    console.log(
      `step ${stepNum} / ${totalStr} | loss ${loss.data.toFixed(4)}`
    );
  }

  // --- Inference ---
  const temperature = 0.6;
  console.log("\n--- inference ---");
  for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
    const keys: Value[][][] = Array.from({ length: n_layer }, () => []);
    const valuesKV: Value[][][] = Array.from({ length: n_layer }, () => []);
    let tokenId = BOS;
    let output = "";

    for (let posId = 0; posId < block_size; posId++) {
      const logits = gpt(tokenId, posId, keys, valuesKV, stateDict);
      const scaledLogits = logits.map((l) => l.div(temperature));
      const probs = softmax(scaledLogits);

      // Weighted random sampling
      const weights = probs.map((p) => p.data);
      const totalWeight = weights.reduce((a, b) => a + b, 0);
      let r = rand() * totalWeight;
      tokenId = 0;
      for (let i = 0; i < weights.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          tokenId = i;
          break;
        }
      }

      if (tokenId === BOS) break;
      output += itos.get(tokenId)!;
    }

    console.log(`sample ${sampleIdx + 1}: ${output}`);
  }
}

main();
