let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
ctx.imageSmoothingEnabled = false;

let drawing = false;
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
  ctx.fill();
}

function clearCanvas() {
  ctx.fillStyle = 'white';          
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result').textContent = "Résultat : ";
}
clearCanvas(); 

let session = null;

async function loadModel() {
  session = await ort.InferenceSession.create('mnist_model.onnx');
  console.log("Modèle ONNX chargé");
}
loadModel();

async function predict() {
  if (!session) {
    alert("Le modèle n'est pas encore chargé, veuillez patienter.");
    return;
  }

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = 28;
  tmpCanvas.height = 28;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.imageSmoothingEnabled = false;

  tmpCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = tmpCtx.getImageData(0, 0, 28, 28);
  const data = imageData.data;
  const input = new Float32Array(1 * 1 * 28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const a = data[i * 4 + 3] / 255;

    const lum = (r + g + b) / 3 * a + 255 * (1 - a);

    input[i] = (255 - lum) / 255;
  }

  console.log("Tensor input min:", Math.min(...input), "max:", Math.max(...input));

  const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);

  try {
    const output = await session.run({ input: tensor });
    const predictions = output.output.data;
    console.log("Prédictions:", predictions);

    const maxProb = Math.max(...predictions);
    const predictedDigit = predictions.indexOf(maxProb);

    document.getElementById('result').textContent = `Résultat : ${predictedDigit}`;
  } catch (err) {
    console.error("Erreur lors de l'inférence :", err);
  }
}


// === Projet 2 : NLP ===
let char2idx = {};
let idx2char = {};
const seqLength = 10;
let session1 = null;
const vocabUrl = "vocab.json";
const onnxModelUrl = "rnn_text_gen.onnx";

fetch(vocabUrl)
  .then(res => res.json())
  .then(data => {
    char2idx = data.char2idx;
    idx2char = Object.fromEntries(
      Object.entries(data.idx2char).map(([k, v]) => [parseInt(k), v])
    );
    console.log("Vocabulaire chargé :", char2idx);
  })
  .catch(err => console.error("Erreur lors du chargement du vocabulaire :", err));

async function loadModel1() {
  try {
    session1 = await ort.InferenceSession.create(onnxModelUrl);
    console.log("Modèle ONNX chargé");
    console.log("Inputs du modèle ONNX :", session1.inputNames);
    console.log("Outputs du modèle ONNX :", session1.outputNames);
  } catch (err) {
    console.error("Échec du chargement du modèle ONNX :", err);
  }
}
loadModel1();

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExps);
}

function sampleFromProbs(probs, temperature = 1.0) {
  const adjusted = probs.map(p => Math.pow(p, 1 / temperature));
  const sum = adjusted.reduce((a, b) => a + b, 0);
  const normalized = adjusted.map(p => p / sum);

  let r = Math.random();
  let cumSum = 0;
  for (let i = 0; i < normalized.length; i++) {
    cumSum += normalized[i];
    if (r < cumSum) return i;
  }
  return normalized.length - 1; 
}

async function generateText() {
  if (!session1) {
    alert("Le modèle n'est pas encore chargé, veuillez patienter.");
    return;
  }

  const inputText = document.getElementById('text-input').value.toLowerCase().trim();
  if (inputText.length < seqLength) {
    alert("Merci d'écrire au moins " + seqLength + " caractères.");
    return;
  }

  const rawChars = inputText.slice(-seqLength).padStart(seqLength, ' ').split('');
  let inputSeq = rawChars.map(ch => char2idx.hasOwnProperty(ch) ? char2idx[ch] : 0);
  console.log("Sequence entrée :", inputSeq.map(i => idx2char[i]).join(""));

  let generated = inputText;
  const temperature = parseFloat(document.getElementById("temp").value);

  const hiddenSize = 512;  
  const numLayers = 1;     
  const batchSize = 1;

  let h0 = new Float32Array(numLayers * batchSize * hiddenSize).fill(0);
  let c0 = new Float32Array(numLayers * batchSize * hiddenSize).fill(0);

  const inputNames = session1.inputNames;
  const outputNames = session1.outputNames;
  console.log("Noms des inputs :", inputNames);
  console.log("Noms des outputs :", outputNames);

  const inputName = inputNames.find(n => n.toLowerCase().includes('input')) || inputNames[0];
  const hName = inputNames.find(n => n.toLowerCase().includes('h')) || null;
  const cName = inputNames.find(n => n.toLowerCase().includes('c')) || null;

  const hOutName = hName ? outputNames.find(n => n.includes(hName.replace(/0/, '1'))) : null;
  const cOutName = cName ? outputNames.find(n => n.includes(cName.replace(/0/, '1'))) : null;

  for (let i = 0; i < 100; i++) {
    const tensorInput = new ort.Tensor('int64', BigInt64Array.from(inputSeq.map(n => BigInt(n))), [batchSize, seqLength]);

    let feeds = {};
    feeds[inputName] = tensorInput;
    if (hName) feeds[hName] = new ort.Tensor('float32', h0, [numLayers, batchSize, hiddenSize]);
    if (cName) feeds[cName] = new ort.Tensor('float32', c0, [numLayers, batchSize, hiddenSize]);

    let results;
    try {
      results = await session1.run(feeds);
    } catch (err) {
      console.error("Erreur lors de l'inférence :", err);
      document.getElementById('generated-text').textContent = "Erreur lors de l'inférence ONNX.";
      return;
    }

    const logits = Array.from(results[outputNames[0]].data);

    if (hOutName) h0 = results[hOutName].data;
    if (cOutName) c0 = results[cOutName].data;

    const probs = softmax(logits);
    const nextIdx = sampleFromProbs(probs, temperature);
    const nextChar = idx2char[nextIdx] || '?';

    generated += nextChar;
    inputSeq.push(nextIdx);
    inputSeq = inputSeq.slice(-seqLength);
  }

  document.getElementById('generated-text').textContent = "Résultat : " + generated;
}

