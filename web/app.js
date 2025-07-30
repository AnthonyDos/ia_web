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
// Simulation simple de génération de texte

function generateText() {
  const input = document.getElementById('text-input').value.toLowerCase();
  const generated = input;

  // Alphabet limité pour correspondre au corpus
  const alphabet = " abcdefghijklmnopqrstuvwxyzéàèùç";
  const randomChar = () => alphabet[Math.floor(Math.random() * alphabet.length)];

  let result = generated;
  for (let i = 0; i < 100; i++) {
    result += randomChar();
  }

  document.getElementById('generated-text').textContent = result;