import * as ort from './vendor/ort.wasm.min.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
const routes = {
  cardboard: ['Cardboard', 'Dry recyclables bin, if accepted', 'Sort and bale clean cardboard', 'Send bales to a paper mill'],
  glass: ['Glass', 'Designated glass collection, if accepted', 'Sort and crush into cullet', 'Use cullet in glass manufacturing'],
  metal: ['Metal', 'Dry recyclables bin, if accepted', 'Separate metals using magnets or eddy currents', 'Send recovered metal for recycling'],
  paper: ['Paper', 'Dry recyclables bin, if clean and accepted', 'Sort and bale by paper grade', 'Pulp for new paper products'],
  plastic: ['Plastic', 'Dry recyclables bin for accepted plastic types', 'Sort by polymer and shred accepted items', 'Reprocess into recycled material'],
  trash: ['Residual trash', 'General waste bin for ordinary residual items', 'Collect separately from recoverable materials', 'Use the local residual route, such as sanitary landfill'],
};
const get = (id) => document.getElementById(id);
const preview = get('preview');
const button = get('classify');
const status = get('status');
const result = get('result');
const empty = result.innerHTML;
const drop = get('drop');
let revision = 0;
let selected;
let selectedSource;
let model;
const metrics = fetch('./metrics.json').then((response) => {
  if (!response.ok) throw new Error('Evaluation file unavailable');
  return response.json();
});

function select(url, autoRun = false, revoke = false) {
  selectedSource = autoRun ? "Demo sample" : "Selected photo";
  const id = ++revision;
  selected = undefined;
  button.disabled = true;
  preview.hidden = true;
  get('placeholder').hidden = false;
  result.innerHTML = empty;
  status.textContent = 'Loading photo…';

  const image = new Image();
  image.onload = () => {
    if (id !== revision) {
      if (revoke) URL.revokeObjectURL(url);
      return;
    }
    selected = image;
    preview.src = url;
    if (revoke) preview.decode().catch(() => {}).finally(() => URL.revokeObjectURL(url));
    preview.hidden = false;
    get('placeholder').hidden = true;
    button.disabled = false;
    status.textContent = 'Photo ready.';
    if (autoRun) classify();
  };
  image.onerror = () => {
    if (revoke) URL.revokeObjectURL(url);
    if (id === revision) status.textContent = 'Could not read that photo. Try a JPEG, PNG, or WebP.';
  };
  image.src = url;
}

function selectFile(file) {
  if (!file) return;
  const supported = /^(image\/(jpeg|png|webp))$/.test(file.type);
  if (!supported || file.size > 10 * 1024 * 1024) {
    ++revision;
    selected = undefined;
    button.disabled = true;
    preview.hidden = true;
    get('placeholder').hidden = false;
    result.innerHTML = empty;
    status.textContent = supported ? 'Choose a photo under 10 MB.' : 'Choose a JPEG, PNG, or WebP photo.';
    return;
  }
  select(URL.createObjectURL(file), false, true);
}

function pixels(image, size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#f2f2f2';
  context.fillRect(0, 0, size, size);
  const scale = Math.min(1, size / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  context.drawImage(image, Math.floor((size - width) / 2), Math.floor((size - height) / 2), width, height);
  const { data } = context.getImageData(0, 0, size, size);
  const area = size * size;
  const tensor = new Float32Array(area * 3);
  for (let i = 0; i < area; i++) {
    tensor[i] = data[i * 4] / 255;
    tensor[area + i] = data[i * 4 + 1] / 255;
    tensor[area * 2 + i] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor('float32', tensor, [1, 3, size, size]);
}

function pathway(key) {
  const [, collection, processing, destination] = routes[key];
  const end = key === 'trash' ? 'UNIT V · DISPOSAL' : 'RECOVERY DESTINATION';
  return `<ol class="path"><li><span>UNIT III · COLLECTION</span>${collection}</li>
    <li><span>UNIT IV · HANDLING OR PROCESSING</span>${processing}</li>
    <li><span>${end}</span>${destination}</li></ol>`;
}

function fillMaterials(select) {
  for (const [key, [label]] of Object.entries(routes)) {
    select.add(new Option(label, key));
  }
}

const material = get('material');
fillMaterials(material);
function showPathway() {
  get('explorer-path').innerHTML = pathway(material.value);
}
material.addEventListener('change', showPathway);
showPathway();

async function classify() {
  if (!selected || button.disabled) return;
  const id = revision;
  const image = selected;
  const source = selectedSource;
  button.disabled = true;
  status.textContent = 'Classifying on this device…';
  result.innerHTML = empty;
  try {
    model ??= ort.InferenceSession.create('./classifier.onnx', { executionProviders: ['wasm'] })
      .catch((error) => { model = undefined; throw error; });
    const [session, config] = await Promise.all([model, metrics]);
    if (id !== revision) return;
    const output = await session.run({ image: pixels(image, config.imageSize) });
    if (id !== revision) return;
    const logits = [...output.logits.data];
    const max = Math.max(...logits);
    const weights = logits.map((score) => Math.exp(score - max));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    const ranked = config.labels.map((key, index) => ({ key, score: weights[index] / total }))
      .sort((a, b) => b.score - a.score);
    const { key, score } = ranked[0];
    const label = routes[key][0];
    const scores = ranked.slice(0, 3).map((guess) => {
      const name = routes[guess.key][0];
      return `<li><span>${name}</span><meter min="0" max="1" value="${guess.score}" aria-label="${name} score"></meter><strong>${percent(guess.score)}</strong></li>`;
    }).join('');
    result.innerHTML = (score < config.threshold
      ? `<p class="prediction">Retake the photo</p><p>Try better lighting and frame one item clearly. If you already know the material, use the <a href="#material">pathway explorer</a>.</p>`
      : `<p class="prediction">${label}</p><span class="badge">${percent(score)} model confidence</span>
         ${pathway(key)}
         <p class="muted">Example pathway. Follow your campus signs for accepted items and separate collection for batteries, e-waste, and other special waste.</p>`)
      + `<details class="scores" ${score < config.threshold ? 'open' : ''}><summary>Top three model guesses</summary>
         <ul class="score-list">${scores}</ul>
         <p class="muted">These scores rank the six known materials. Unfamiliar items can still receive a high score.</p></details>`;
    recordScan({time: new Date().toISOString(), source, key, score, review: score < config.threshold});
    status.textContent = 'Analysis complete.';
  } catch (error) {
    if (id === revision) status.textContent = 'The classifier could not run. Reload the page and try again.';
    console.error(error);
  } finally {
    if (id === revision) button.disabled = false;
  }
}

for (const input of [get('photo'), get('camera')]) {
  input.addEventListener('change', () => {
    selectFile(input.files[0]);
    input.value = '';
  });
}
drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('over');
  selectFile(event.dataTransfer.files[0]);
});
button.addEventListener('click', classify);
for (const sample of document.querySelectorAll('[data-sample]')) {
  sample.addEventListener('click', () => select(`./examples/${sample.dataset.sample}.jpg`, true));
}

const percent = (value) => `${(value * 100).toFixed(1)}%`;
metrics.then((data) => {
  const report = data.classification;
  get('metrics-summary').textContent = `${percent(report.accuracy)} accuracy across ${data.split.test} held-out TrashNet images. These results describe the benchmark dataset. Performance can vary on campus photos.`;
  const rows = data.labels.map((key) => {
    const { precision, recall, support } = report[key];
    return `<tr><th scope="row">${routes[key][0]}</th><td>${percent(precision)}</td><td>${percent(recall)}</td><td>${support}</td></tr>`;
  }).join('');
  get('metrics-table').innerHTML = `<table aria-label="Per-class results"><thead><tr><th scope="col">Material</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Photos</th></tr></thead><tbody>${rows}</tbody></table>`;
  const actualClass = get('actual-class');
  fillMaterials(actualClass);
  actualClass.value = 'trash';
  function showConfusion() {
    const actualIndex = data.labels.indexOf(actualClass.value);
    const counts = data.confusionMatrix[actualIndex];
    const total = counts.reduce((sum, count) => sum + count, 0);
    const correct = counts[actualIndex];
    const bars = data.labels.map((key, index) =>
      `<li><span>${routes[key][0]}</span>
        <meter min="0" max="${total}" value="${counts[index]}" aria-label="Predicted as ${routes[key][0]}"></meter>
        <strong>${counts[index]}</strong></li>`
    ).join('');
    get('confusion-row').innerHTML =
      `<p>${correct} of ${total} ${routes[actualClass.value][0].toLowerCase()} photos were classified correctly.</p>
       <p class="eyebrow">PREDICTED AS</p><ul class="score-list">${bars}</ul>`;
  }
  actualClass.addEventListener('change', showConfusion);
  showConfusion();
  get('confusion').hidden = false;
}).catch(() => { get('metrics-summary').textContent = 'Evaluation data could not be loaded.'; });

const scans = [];
const mix = get('material-mix');
function renderSession() {
  const count = scans.length;
  get('scan-count').textContent = count;
  get('review-count').textContent = scans.filter((scan) => scan.review).length;
  get('average-score').textContent = count
    ? percent(scans.reduce((sum, scan) => sum + scan.score, 0) / count) : '—';
  get('export-session').disabled = get('clear-session').disabled = !count;
  get('history-empty').hidden = Boolean(count);
  get('history-table').hidden = !count;
  const counts = Object.keys(routes).map((key) => ({
    key, count: scans.filter((scan) => scan.key === key).length,
  }));
  mix.innerHTML = count ? '<p class="eyebrow">LEADING GUESSES BY MATERIAL · INCLUDES UNCERTAIN RESULTS</p><ul class="score-list">'
    + counts.map((entry) => `<li><span>${routes[entry.key][0]}</span>
      <meter min="0" max="${count}" value="${entry.count}" aria-label="${routes[entry.key][0]} scans"></meter>
      <strong>${entry.count}</strong></li>`).join('') + '</ul>' : '';
  get('history-rows').replaceChildren();
  for (const scan of [...scans].reverse()) {
    const row = document.createElement('tr');
    const values = [
      new Date(scan.time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'}),
      scan.source, routes[scan.key][0], percent(scan.score),
      scan.review ? 'Retake photo' : 'View material pathway',
    ];
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    get('history-rows').append(row);
  }
}

function recordScan(scan) {
  scans.push(scan);
  renderSession();
}
get('clear-session').addEventListener('click', () => {
  scans.length = 0;
  renderSession();
});
get('export-session').addEventListener('click', () => {
  const rows = [['Time (UTC)', 'Source', 'Leading guess', 'Score', 'Action']];
  for (const scan of scans) {
    rows.push([scan.time, scan.source, routes[scan.key][0], scan.score.toFixed(6),
      scan.review ? 'Retake photo' : 'View material pathway']);
  }
  const csv = rows.map((row) => row.map((value) =>
    '"' + String(value).replaceAll('"', '""') + '"').join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'campus-sort-session.csv';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

const questions = [
  {
    question: 'A campus bin stays in place while a truck empties it. Which collection system is this?',
    options: ['Hauled container system', 'Stationary container system', 'Pneumatic collection'],
    answer: 1,
    explanation: 'In a stationary container system, the bin stays at the collection point while the waste is removed.',
  },
  {
    question: 'Which operation compresses clean cardboard into transportable blocks?',
    options: ['Baling', 'Composting', 'Leachate treatment'],
    answer: 0,
    explanation: 'Baling reduces the volume of cardboard and makes it easier to store and transport.',
  },
  {
    question: 'Which technique can separate ferrous metals from mixed waste?',
    options: ['Anaerobic digestion', 'Magnetic separation', 'Daily soil cover'],
    answer: 1,
    explanation: 'Magnets attract ferrous metals. Eddy-current separation is used for suitable non-ferrous metals.',
  },
  {
    question: 'What is the main purpose of a landfill liner?',
    options: ['Compress waste', 'Sort plastics', 'Limit seepage into soil and groundwater'],
    answer: 2,
    explanation: 'A liner forms a barrier beneath the waste. Leachate collection pipes carry contaminated liquid for treatment.',
  },
  {
    question: 'Which process can convert suitable organic waste into biogas?',
    options: ['Baling', 'Glass crushing', 'Anaerobic digestion'],
    answer: 2,
    explanation: 'Anaerobic digestion breaks down suitable organic material without oxygen and produces biogas.',
  },
];
let questionIndex = 0;
let quizScore = 0;
function renderQuestion() {
  const question = questions[questionIndex];
  get('quiz-progress').textContent = `Question ${questionIndex + 1} of ${questions.length} · ${quizScore} correct`;
  get('quiz-question').textContent = question.question;
  get('quiz-feedback').textContent = '';
  get('quiz-next').hidden = true;
  get('quiz-restart').hidden = true;
  get('quiz-options').replaceChildren();
  question.options.forEach((text, index) => {
    const option = document.createElement('button');
    option.textContent = text;
    option.addEventListener('click', () => {
      const correct = index === question.answer;
      quizScore += Number(correct);
      for (const button of get('quiz-options').children) button.disabled = true;
      option.classList.add(correct ? 'answer-correct' : 'answer-incorrect');
      get('quiz-feedback').textContent = (correct ? 'Correct. ' : 'Correct answer — ' + question.options[question.answer] + '. ') + question.explanation;
      get('quiz-progress').textContent = `Question ${questionIndex + 1} of ${questions.length} · ${quizScore} correct`;
      get('quiz-next').hidden = false;
      get('quiz-next').textContent = questionIndex === questions.length - 1 ? 'See final score' : 'Next question';
    });
    get('quiz-options').append(option);
  });
}
get('quiz-next').addEventListener('click', () => {
  questionIndex++;
  if (questionIndex < questions.length) {
    renderQuestion();
    return;
  }
  get('quiz-question').textContent = `You scored ${quizScore} out of ${questions.length}`;
  get('quiz-options').replaceChildren();
  get('quiz-feedback').textContent = 'Explore the course guide and try again to practise the waste management concepts.';
  get('quiz-next').hidden = true;
  get('quiz-restart').hidden = false;
});
get('quiz-restart').addEventListener('click', () => {
  questionIndex = 0;
  quizScore = 0;
  renderQuestion();
});
renderQuestion();

// Demo map data can be replaced with verified campus collection points.
const materialNames = {...Object.fromEntries(Object.entries(routes).map(([key, value]) => [key, value[0]])), organic:'Organic waste'};
const bins = [
  {id:'library', name:'Library recycling point', x:25, y:25, materials:['paper','cardboard']},
  {id:'canteen', name:'Canteen segregation station', x:75, y:25, materials:['plastic','metal','organic','trash']},
  {id:'academic', name:'Academic block recycling point', x:25, y:72, materials:['paper','plastic','metal','cardboard']},
  {id:'hostel', name:'Hostel collection station', x:75, y:72, materials:['glass','plastic','metal','trash']},
];
const binFilter = get('bin-filter');
for (const [key, label] of Object.entries(materialNames)) binFilter.add(new Option(label, key));
let activeBin = bins[0].id;
function renderMap() {
  const matching = bins.filter((bin) => binFilter.value === 'all' || bin.materials.includes(binFilter.value));
  if (!matching.some((bin) => bin.id === activeBin)) activeBin = matching[0]?.id;
  get('map-pins').replaceChildren();
  get('bin-list').replaceChildren();
  for (const bin of matching) {
    const choose = () => { activeBin = bin.id; renderMap(); };
    const pin = document.createElement('button');
    pin.className = 'map-pin';
    pin.style.left = bin.x + '%';
    pin.style.top = bin.y + '%';
    pin.textContent = String(bins.indexOf(bin) + 1);
    pin.setAttribute('aria-label', bin.name);
    pin.setAttribute('aria-pressed', String(activeBin === bin.id));
    pin.addEventListener('click', choose);
    get('map-pins').append(pin);
    const item = document.createElement('button');
    item.textContent = pin.textContent + ' · ' + bin.name;
    item.setAttribute('aria-pressed', String(activeBin === bin.id));
    item.addEventListener('click', choose);
    get('bin-list').append(item);
  }
  const bin = matching.find((entry) => entry.id === activeBin);
  get('bin-details').innerHTML = bin
    ? `<span class="badge">Demo location ${bins.indexOf(bin) + 1}</span><h3>${bin.name}</h3><p>Separate containers at this illustrative collection point accept the following material streams.</p><div class="tags">${bin.materials.map((key) => `<span>${materialNames[key]}</span>`).join('')}</div>`
    : '<p>No matching collection points.</p>';
  get('bin-count').textContent = `${matching.length} matching collection points. Select a numbered pin or a location below.`;
}
binFilter.addEventListener('change', renderMap);
renderMap();

const processes = {
  cardboard:[['Collect','Collect clean, dry cardboard separately.','Segregated cardboard','Unit III'],['Sort','Remove unsuitable material at the recovery facility.','Sorted cardboard','Unit IV'],['Bale','Compress cardboard for easier storage and transport.','Compact bales','Unit IV'],['Recycle','Send accepted bales to a paper mill for pulping.','Recovered paper fibre','Material recovery']],
  paper:[['Collect','Keep accepted paper dry and separate from food waste.','Collected paper','Unit III'],['Sort','Separate grades and remove contaminants.','Sorted paper','Unit IV'],['Bale','Compress paper for transport.','Paper bales','Unit IV'],['Pulp','A paper mill processes accepted paper into fibre.','Recovered fibre','Material recovery']],
  glass:[['Collect','Use the designated glass collection container.','Collected glass','Unit III'],['Sort','Remove unsuitable items and sort to facility requirements.','Sorted glass','Unit IV'],['Crush','Reduce suitable glass into smaller pieces called cullet.','Glass cullet','Unit IV'],['Remanufacture','Use suitable cullet in glass manufacturing.','Recovered glass','Material recovery']],
  metal:[['Collect','Collect accepted metal packaging separately.','Collected metal','Unit III'],['Separate','Magnets recover ferrous metals. Eddy currents separate suitable non-ferrous metals.','Separated metals','Unit IV'],['Prepare','Prepare sorted metal for transport and recycling.','Metal feedstock','Unit IV'],['Recycle','Remelt and process suitable recovered metal.','Recovered metal','Material recovery']],
  plastic:[['Collect','Collect plastic types accepted by the local service.','Collected plastic','Unit III'],['Sort','Separate compatible polymers and remove unsuitable material.','Sorted plastic','Unit IV'],['Shred','Reduce suitable sorted plastic into smaller fragments.','Plastic fragments','Unit IV'],['Reprocess','Clean and reprocess accepted fragments at a suitable facility.','Recycled plastic material','Material recovery']],
  trash:[['Collect','Collect ordinary residual waste separately from special waste.','Residual waste','Unit III'],['Transfer','Consolidate waste for transport to the receiving facility.','Consolidated residuals','Unit III'],['Place and cover','Where sanitary landfill is used, place waste in cells, compact it, and cover it.','Managed waste cell','Unit V'],['Manage emissions','Liners limit seepage, pipes collect leachate for treatment, and wells collect landfill gas.','Collected liquid and gas','Unit V']],
  organic:[['Segregate','Separate suitable food and garden waste from packaging.','Organic feedstock','Unit III'],['Prepare','Remove contamination and prepare for biological treatment.','Prepared organics','Unit IV'],['Treat biologically','Composting uses aerobic decomposition. Anaerobic digestion operates without oxygen and produces biogas.','Compost, or biogas and digestate','Unit IV'],['Recover resources','Use finished compost or treated digestate as appropriate. Biogas can supply energy.','Useful biological products','Resource recovery']],
};
const simMaterial = get('sim-material');
for (const [key, label] of Object.entries(materialNames)) simMaterial.add(new Option(label, key));
let simStep = 0;
let simTimer;
function pauseSimulation() {
  clearInterval(simTimer);
  simTimer = undefined;
}
function renderSimulation() {
  const stages = processes[simMaterial.value];
  get('sim-stages').replaceChildren();
  stages.forEach((stage, index) => {
    const item = document.createElement('li');
    const control = document.createElement('button');
    control.textContent = `${index + 1}. ${stage[0]}`;
    if (index === simStep) control.setAttribute('aria-current', 'step');
    control.addEventListener('click', () => {
      pauseSimulation();
      simStep = index;
      renderSimulation();
    });
    item.append(control);
    get('sim-stages').append(item);
  });
  const [title, explanation, output, unit] = stages[simStep];
  get('sim-detail').innerHTML = `<p class="eyebrow">${unit} · STEP ${simStep + 1} OF ${stages.length}</p><h3>${title}</h3><p>${explanation}</p><div class="sim-output"><span>Stage output</span><strong>${output}</strong></div>`;
  get('sim-back').disabled = simStep === 0;
  get('sim-next').disabled = simStep === stages.length - 1;
  get('sim-play').textContent = simTimer ? 'Pause sequence' : 'Play sequence';
}
simMaterial.addEventListener('change', () => { pauseSimulation(); simStep = 0; renderSimulation(); });
get('sim-next').addEventListener('click', () => { pauseSimulation(); simStep = Math.min(simStep + 1, processes[simMaterial.value].length - 1); renderSimulation(); });
get('sim-back').addEventListener('click', () => { pauseSimulation(); simStep = Math.max(simStep - 1, 0); renderSimulation(); });
get('sim-reset').addEventListener('click', () => { pauseSimulation(); simStep = 0; renderSimulation(); });
get('sim-play').addEventListener('click', () => {
  if (simTimer) {
    pauseSimulation();
  } else {
    if (simStep === processes[simMaterial.value].length - 1) simStep = 0;
    simTimer = setInterval(() => {
      simStep++;
      if (simStep === processes[simMaterial.value].length - 1) pauseSimulation();
      renderSimulation();
    }, 3500);
  }
  renderSimulation();
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { pauseSimulation(); renderSimulation(); }
});
renderSimulation();
