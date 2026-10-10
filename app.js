import {
  AutoModelForCausalLM,
  AutoTokenizer,
  env,
} from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

const MODEL_ID = "Xenova/distilgpt2";
const TOP_K = 8;
const MAX_INPUT_TOKENS = 1024;
const MAX_PROMPT_CHARS = 4000;

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.backends.onnx.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/";

const samples = [
  "The student opened the book and learned about",
  "In the future, artificial intelligence will help people",
  "The chef tasted the soup and added",
  "When the rain stopped, the children ran outside to",
  "A language model predicts the next",
];

const state = {
  tokenizer: null,
  model: null,
  loading: false,
  ready: false,
  lastResult: null,
};

const els = {
  modelStatus: document.querySelector("#modelStatus"),
  statusText: document.querySelector("#statusText"),
  loadModelButton: document.querySelector("#loadModelButton"),
  inspectButton: document.querySelector("#inspectButton"),
  sampleButton: document.querySelector("#sampleButton"),
  promptInput: document.querySelector("#promptInput"),
  predictionList: document.querySelector("#predictionList"),
  tokenList: document.querySelector("#tokenList"),
  runtimeBadge: document.querySelector("#runtimeBadge"),
  dynamicExplanation: document.querySelector("#dynamicExplanation"),
  comparisonExplanation: document.querySelector("#comparisonExplanation"),
  topChoice: document.querySelector("#topChoice"),
  confidenceValue: document.querySelector("#confidenceValue"),
  spreadValue: document.querySelector("#spreadValue"),
};

els.loadModelButton.addEventListener("click", () => loadModel(true));
els.inspectButton.addEventListener("click", inspectPrompt);
els.sampleButton.addEventListener("click", useSample);
document.querySelectorAll(".prompt-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    els.promptInput.value = chip.dataset.prompt;
    els.promptInput.focus();
  });
});

function setStatus(kind, text) {
  els.modelStatus.classList.remove("ready", "error");
  if (kind) els.modelStatus.classList.add(kind);
  els.statusText.textContent = text;
}

async function loadModel(runInitialInspection = false) {
  if (state.ready || state.loading) return;

  state.loading = true;
  els.loadModelButton.disabled = true;
  els.inspectButton.disabled = true;
  els.runtimeBadge.textContent = "Loading";
  setStatus("", "Loading small LLM");

  try {
    state.tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    state.model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
      progress_callback: updateProgress,
    });
    state.ready = true;
    els.runtimeBadge.textContent = MODEL_ID;
    setStatus("ready", "Small LLM ready");
    if (runInitialInspection) await inspectPrompt();
  } catch (error) {
    console.error(error);
    setStatus("error", "Model failed to load");
    els.runtimeBadge.textContent = "Load failed";
    els.predictionList.innerHTML = `
      <p class="empty-state">
        The browser could not download or initialize ${MODEL_ID}. Check your internet connection,
        then press "Load small LLM" again. Details: ${escapeHtml(getErrorMessage(error))}
      </p>
    `;
    state.tokenizer = null;
    state.model = null;
  } finally {
    state.loading = false;
    els.loadModelButton.disabled = false;
    els.inspectButton.disabled = false;
  }
}

function updateProgress(progress) {
  if (!progress?.status) return;

  if (progress.status === "progress" && progress.file) {
    const percent = Number.isFinite(progress.progress)
      ? `${Math.round(progress.progress)}%`
      : "downloading";
    setStatus("", `${progress.file}: ${percent}`);
    return;
  }

  if (progress.status !== "ready") {
    setStatus("", progress.status.replaceAll("_", " "));
  }
}

async function inspectPrompt() {
  const prompt = els.promptInput.value.trim();
  if (!prompt) {
    els.predictionList.innerHTML = `<p class="empty-state">Type a prompt first.</p>`;
    return;
  }

  if (!state.ready) {
    await loadModel(false);
    if (!state.ready) return;
  }

  els.inspectButton.disabled = true;
  els.runtimeBadge.textContent = "Running";

  try {
    if (prompt.length > MAX_PROMPT_CHARS) {
      showPromptTooLong(prompt.length);
      return;
    }

    const inputs = await state.tokenizer(prompt);
    const inputIds = Array.from(inputs.input_ids.data, Number);
    if (inputIds.length > MAX_INPUT_TOKENS) {
      showPromptTooLong(inputIds.length, true);
      return;
    }
    const output = await state.model(inputs);
    const predictions = getTopPredictions(output.logits, inputIds.length);

    renderPromptTokens(inputIds);
    renderPredictions(predictions);
    renderExplanation(prompt, predictions);
    renderInsights(predictions);
    renderComparison(prompt, predictions);
    state.lastResult = { prompt, top: predictions[0] };
    els.runtimeBadge.textContent = MODEL_ID;
  } catch (error) {
    console.error(error);
    els.runtimeBadge.textContent = "Error";
    els.topChoice.textContent = "—";
    els.confidenceValue.textContent = "—";
    els.spreadValue.textContent = "—";
    els.tokenList.innerHTML = `<p class="empty-state">Prompt tokens are unavailable because inspection failed.</p>`;
    els.predictionList.innerHTML = `
      <p class="empty-state">
        Something went wrong while reading the model logits. ${escapeHtml(getErrorMessage(error))}
        Try a shorter prompt and inspect again.
      </p>
    `;
  } finally {
    els.inspectButton.disabled = false;
  }
}

function showPromptTooLong(size, countedTokens = false) {
  els.runtimeBadge.textContent = "Prompt too long";
  els.topChoice.textContent = "—";
  els.confidenceValue.textContent = "—";
  els.spreadValue.textContent = "—";
  const detail = countedTokens
    ? `This prompt becomes ${size} tokens.`
    : `This prompt is ${size} characters long, which may exceed the model's context window.`;
  els.predictionList.innerHTML = `
    <p class="empty-state">
      ${detail} ${MODEL_ID} accepts up to ${MAX_INPUT_TOKENS} tokens here. Shorten the prompt and inspect again.
    </p>
  `;
  els.tokenList.innerHTML = `<p class="empty-state">Prompt was not inspected because it is too long.</p>`;
}

function getErrorMessage(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  return message.trim() || "Unknown browser error";
}

function getTopPredictions(logitsTensor, sequenceLength) {
  const vocabSize = logitsTensor.dims.at(-1);
  const offset = (sequenceLength - 1) * vocabSize;
  const scored = [];

  for (let tokenId = 0; tokenId < vocabSize; tokenId += 1) {
    scored.push({ tokenId, logit: logitsTensor.data[offset + tokenId] });
  }

  const maxLogit = Math.max(...scored.map((item) => item.logit));
  const expTotal = scored.reduce((sum, item) => sum + Math.exp(item.logit - maxLogit), 0);
  const top = scored.sort((a, b) => b.logit - a.logit).slice(0, TOP_K);

  return top.map((item) => {
    const probability = Math.exp(item.logit - maxLogit) / expTotal;
    return {
      tokenId: item.tokenId,
      token: cleanDecodedToken(state.tokenizer.decode([item.tokenId])),
      probability,
    };
  });
}

function renderPredictions(predictions) {
  const strongest = predictions[0]?.probability ?? 1;
  els.predictionList.innerHTML = predictions
    .map((prediction) => {
      const width = Math.max(5, (prediction.probability / strongest) * 100);
      return `
        <div class="prediction-item">
          <span class="token-text">${escapeHtml(prediction.token)}</span>
          <span class="bar-track" aria-hidden="true">
            <span class="bar-fill" style="width: ${width}%"></span>
          </span>
          <span class="probability">${formatPercent(prediction.probability)}</span>
        </div>
      `;
    })
    .join("");
}

function renderPromptTokens(inputIds) {
  els.tokenList.innerHTML = inputIds
    .map((id) => {
      const token = cleanDecodedToken(state.tokenizer.decode([id]));
      return `<span class="prompt-token">${escapeHtml(token)}</span>`;
    })
    .join("");
}

function renderExplanation(prompt, predictions) {
  const top = predictions[0];
  const second = predictions[1];
  const lead = top
    ? `For "${prompt}", the strongest next-token candidate is "${top.token}" at ${formatPercent(top.probability)} among the displayed top choices.`
    : "The model did not return a prediction.";
  const contrast = second
    ? ` The next strongest candidate is "${second.token}", which shows that the model is comparing several plausible continuations rather than following a single rule.`
    : "";
  els.dynamicExplanation.textContent = lead + contrast;
}

function renderInsights(predictions) {
  const top = predictions[0];
  const second = predictions[1];
  els.topChoice.textContent = top?.token ?? "—";
  els.confidenceValue.textContent = top ? formatPercent(top.probability) : "—";

  if (!top || !second) {
    els.spreadValue.textContent = "—";
    return;
  }

  const gap = top.probability - second.probability;
  els.spreadValue.textContent = gap < 0.01 ? "close race" : `${formatPercent(gap)} lead`;
}

function renderComparison(prompt, predictions) {
  const previous = state.lastResult;
  const current = predictions[0];

  if (!previous || !current || previous.prompt === prompt) {
    els.comparisonExplanation.textContent =
      "Inspect one prompt, then choose a different context above. The app will describe how the leading prediction changed.";
    return;
  }

  const changed = previous.top?.token !== current.token;
  els.comparisonExplanation.textContent = changed
    ? `Changing the prompt from “${previous.prompt}” to “${prompt}” changed the top prediction from “${previous.top?.token ?? "—"}” to “${current.token}”. That is a concrete example of context changing the model's distribution.`
    : `The top prediction stayed at “${current.token}”, but its confidence moved from ${formatPercent(previous.top?.probability ?? 0)} to ${formatPercent(current.probability)}. Same winner, different distribution.`;
}

function useSample() {
  const current = els.promptInput.value.trim();
  const next =
    samples.find((sample) => sample !== current && !current.startsWith(sample)) ?? samples[0];
  els.promptInput.value = next;
}

function cleanDecodedToken(token) {
  return token.replaceAll("\n", "\\n").replaceAll("\t", "\\t").replaceAll(" ", "·") || "[blank]";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
