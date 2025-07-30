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
  ctx.fillStyle = 'white';           // Fond blanc
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result').textContent = "Résultat : ";
}
clearCanvas(); // Initialisation fond blanc

let session = null;

// Charger la session ONNX une fois
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

  // Canvas temporaire 28x28 pour redimensionner proprement
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = 28;
  tmpCanvas.height = 28;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.imageSmoothingEnabled = false;

  // Copier/redimensionner le dessin dans tmpCanvas 28x28
  tmpCtx.drawImage(canvas, 0, 0, 28, 28);

  const imageData = tmpCtx.getImageData(0, 0, 28, 28);
  const data = imageData.data;
  const input = new Float32Array(1 * 1 * 28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const a = data[i * 4 + 3] / 255;

    // Calcul luminosité moyenne pondérée par alpha
    const lum = (r + g + b) / 3 * a + 255 * (1 - a);

    // Inversion (fond blanc=0, trait noir=1) + normalisation
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

// 1. Génération de texte (chaîne de Markov)
const corpus = `
  La vie est belle. Le soleil brille. Les étoiles dansent dans le ciel.
  Le vent souffle fort. La nuit est calme. Les rêves illuminent l'esprit.
  L'amour réchauffe le cœur. Le silence parle parfois plus que les mots.
`;

function buildMarkovChain(text) {
  const words = text.trim().split(/\s+/);
  const chain = {};
  for (let i = 0; i < words.length - 1; i++) {
    const w = words[i].toLowerCase();
    const next = words[i + 1].toLowerCase();
    if (!chain[w]) chain[w] = [];
    chain[w].push(next);
  }
  return chain;
}

const markovChain = buildMarkovChain(corpus);

function generateMarkovText(start = "la", maxWords = 20) {
  let current = start.toLowerCase();
  let result = [current];
  for (let i = 0; i < maxWords - 1; i++) {
    const nextWords = markovChain[current];
    if (!nextWords || nextWords.length === 0) break;
    current = nextWords[Math.floor(Math.random() * nextWords.length)];
    result.push(current);
  }
  return result.join(" ") + ".";
}

function generateText() {
  const input = document.getElementById("inputText").value.trim();
  const seed = input !== "" ? input.split(/\s+/).pop() : "la";
  const sentence = generateMarkovText(seed, 20);
  document.getElementById("generatedText").textContent = "Texte généré : " + sentence;
}

// 2. Analyse de sentiment basique
const positiveWords = ["amour", "beau", "joie", "paix", "bonheur", "lumière", "réussite"];
const negativeWords = ["triste", "peur", "douleur", "colère", "solitude", "noir"];

function analyzeSentiment() {
  const input = document.getElementById("inputText").value.toLowerCase();
  const words = input.split(/\s+/);
  let score = 0;

  for (const word of words) {
    if (positiveWords.includes(word)) score++;
    if (negativeWords.includes(word)) score--;
  }

  let sentiment = "neutre";
  if (score > 0) sentiment = "positif";
  else if (score < 0) sentiment = "négatif";

  document.getElementById("sentimentResult").textContent = `Analyse de sentiment : ${sentiment} (${score})`;
}

// 3. Traduction FR → EN simplifiée
const frToEn = {
  "bonjour": "hello",
  "chat": "cat",
  "chien": "dog",
  "amour": "love",
  "soleil": "sun",
  "nuit": "night",
  "beau": "beautiful",
  "fleur": "flower"
};

function translateText() {
  const input = document.getElementById("inputText").value.toLowerCase();
  const words = input.split(/\s+/);
  const translated = words.map(word => frToEn[word] || `[${word}]`);
  document.getElementById("translationResult").textContent = "Traduction : " + translated.join(" ");
}