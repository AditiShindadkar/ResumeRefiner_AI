import { scoreChunked, chunkText } from "./chunking.js";
import { buildMarkdownReport } from "./report.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const EMBED_MODEL = "Xenova/all-MiniLM-L6-v2"; // ONNX export of sentence-transformers/all-MiniLM-L6-v2
const MODEL_HARD_LIMIT = 256;
const MAX_UPLOAD_BYTES = 3_000_000;

const $ = (id) => document.getElementById(id);
const els = {
  resume: $("resume"), jd: $("jd"), file: $("file"), fileStatus: $("file-status"),
  resumeMeta: $("resume-meta"), useLlm: $("use-llm"), llmLabel: $("llm-label"),
  llmNote: $("llm-note"), analyze: $("btn-analyze"), status: $("status"), results: $("results"),
};

let lastResult = null;
let llmModel = "";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtPct = (x) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("error", isError);
}

async function api(name, payload) {
  const res = await fetch(`/api/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---------------------------------------------------------------------------
// In-browser embedding model (loaded lazily, cached by the browser)
// ---------------------------------------------------------------------------

let extractorPromise = null;
function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import(TRANSFORMERS_URL);
      env.allowLocalModels = false;
      const seen = new Map();
      return pipeline("feature-extraction", EMBED_MODEL, {
        dtype: "q8",
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            seen.set(p.file, [p.loaded, p.total]);
            let loaded = 0, total = 0;
            for (const [l, t] of seen.values()) { loaded += l; total += t; }
            setStatus(`Downloading embedding model (first run only)… ${Math.round((loaded / total) * 100)}%`);
          }
        },
      });
    })().catch((e) => { extractorPromise = null; throw e; });
  }
  return extractorPromise;
}

function tokenCounter(extractor) {
  const cache = new Map();
  return (text) => {
    if (!cache.has(text)) {
      cache.set(text, extractor.tokenizer.encode(text, { add_special_tokens: false }).length);
    }
    return cache.get(text);
  };
}

async function embedAll(extractor, texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 16) {
    const t = await extractor(texts.slice(i, i + 16), { pooling: "mean", normalize: true });
    out.push(...t.tolist());
  }
  return out;
}

async function computeCoverage(resume, jd, cfg) {
  const extractor = await getExtractor();
  return scoreChunked(resume, jd, {
    countTokens: tokenCounter(extractor),
    embed: (texts) => embedAll(extractor, texts),
    cfg,
  });
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

function refreshReady() {
  els.analyze.disabled = !(els.resume.value.trim() && els.jd.value.trim());
}

let metaTimer = null;
function refreshResumeMeta() {
  clearTimeout(metaTimer);
  const text = els.resume.value;
  const words = text.split(/\s+/).filter(Boolean).length;
  els.resumeMeta.textContent = words ? `${words} words` : "";
  // Token counts need the tokenizer; only show them once the model is loaded.
  if (!words || !extractorPromise) return;
  metaTimer = setTimeout(async () => {
    try {
      const ex = await extractorPromise;
      const count = tokenCounter(ex);
      const n = chunkText(text, count, 80).length;
      els.resumeMeta.textContent = `${words} words · ${count(text)} word pieces · read as ${n} ` +
        `chunk(s) (model limit ${MODEL_HARD_LIMIT} per chunk)`;
    } catch { /* model unavailable; keep the word count */ }
  }, 300);
}

function resetResults() {
  lastResult = null;
  els.results.hidden = true;
}

els.resume.addEventListener("input", () => { refreshReady(); refreshResumeMeta(); });
els.jd.addEventListener("input", refreshReady);

$("btn-sample").addEventListener("click", async () => {
  try {
    const [r, j] = await Promise.all(["sample_resume.txt", "sample_jd.txt"].map((f) =>
      fetch(`samples/${f}`).then((res) => { if (!res.ok) throw new Error(f); return res.text(); })));
    els.resume.value = r;
    els.jd.value = j;
    resetResults();
    refreshReady();
    refreshResumeMeta();
  } catch (e) {
    setStatus(`Could not load sample data: ${e.message}`, true);
  }
});

$("btn-clear").addEventListener("click", () => {
  els.resume.value = "";
  els.jd.value = "";
  els.file.value = "";
  els.fileStatus.hidden = true;
  resetResults();
  setStatus("");
  refreshReady();
  refreshResumeMeta();
});

els.file.addEventListener("change", async () => {
  const f = els.file.files[0];
  if (!f) return;
  const show = (msg, err = false) => {
    els.fileStatus.hidden = false;
    els.fileStatus.textContent = msg;
    els.fileStatus.classList.toggle("error", err);
  };
  if (f.size > MAX_UPLOAD_BYTES) return show("File is larger than 3 MB.", true);
  show(`Reading ${f.name}…`);
  try {
    if (f.name.toLowerCase().endsWith(".txt")) {
      els.resume.value = await f.text();
    } else {
      const buf = new Uint8Array(await f.arrayBuffer());
      let bin = "";
      for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
      const { text } = await api("extract", { filename: f.name, data: btoa(bin) });
      els.resume.value = text;
    }
    show(`Loaded ${f.name}.`);
    resetResults();
    refreshReady();
    refreshResumeMeta();
  } catch (e) {
    show(e.message, true);
  }
});

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

els.analyze.addEventListener("click", async () => {
  const resume = els.resume.value.trim();
  const jd = els.jd.value.trim();
  const useLlm = els.useLlm.checked && !els.useLlm.disabled;
  els.analyze.disabled = true;
  els.analyze.classList.add("loading");
  els.analyze.querySelector(".btn-text").textContent = "Analyzing…";
  setStatus("Analyzing…");
  try {
    const base = await api("analyze", { resume, jd });
    const result = { ...base, coverage: null, coverageError: null, llm: null, llmError: null,
                     llmPending: useLlm };
    lastResult = result;
    render(result);
    els.results.scrollIntoView({ behavior: "smooth", block: "start" });
    setStatus(extractorPromise ? "Computing semantic coverage…" : "Loading embedding model…");

    const coverageTask = computeCoverage(resume, jd, base.chunking)
      .then((c) => { result.coverage = c; })
      .catch((e) => { result.coverageError = `Semantic coverage unavailable: ${e.message}`; })
      .finally(() => render(result));

    const llmTask = useLlm
      ? api("suggest", { resume, jd, missing: base.skills.missing })
          .then((r) => { result.llm = r.feedback; })
          .catch((e) => { result.llmError = e.message; })
          .finally(() => render(result))
      : Promise.resolve();

    await coverageTask;
    if (useLlm && result.llmPending && !result.llm && !result.llmError) {
      setStatus(`Waiting for LLM suggestions (${llmModel})…`);
    }
    await llmTask;
    result.llmPending = false;
    render(result);
    setStatus(result.coverageError || "Done.", Boolean(result.coverageError));
    refreshResumeMeta();
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    els.analyze.classList.remove("loading");
    els.analyze.querySelector(".btn-text").textContent = "Analyze";
    refreshReady();
  }
});

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const SKELETON = `<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>`;
const clamp01 = (x) => Math.max(0, Math.min(1, x ?? 0));
// Drop leading bullet glyphs ("- ", "• ") for display only.
const unbullet = (s) => s.replace(/^[\s\-•●▪◦‣*–>]+/, "");

function setMeter(id, fraction) {
  // Defer one frame so the width transition animates from the previous value.
  requestAnimationFrame(() => { $(id).style.width = `${(clamp01(fraction) * 100).toFixed(1)}%`; });
}

// Only rewrite a region when its content actually changes, so open
// <details>, scroll position and running animations aren't reset by re-renders.
function setHTML(id, html) {
  const el = $(id);
  if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; }
}

function render(r) {
  els.results.hidden = false;
  const q = r.quality;
  const { matched, missing, extra } = r.skills;

  // Headline metrics
  $("m-coverage").textContent = r.coverage ? fmtPct(r.coverage.score) : (r.coverageError ? "n/a" : "…");
  setMeter("mb-coverage", r.coverage?.score);
  const total = matched.length + missing.length;
  $("m-skills").textContent = total ? `${matched.length}/${total}` : "n/a";
  setMeter("mb-skills", total ? matched.length / total : 0);
  $("m-tfidf").textContent = fmtPct(r.tfidf);
  setMeter("mb-tfidf", r.tfidf);
  $("m-quality").textContent = `${q.score}/100`;
  setMeter("mb-quality", q.score / 100);

  // Skills as chips
  const group = (cls, label, items) => items.length
    ? `<div class="skill-group"><div class="skill-group-title">${label} <span class="count">${items.length}</span></div>
         <div class="chips">${items.map((s) => `<span class="chip ${cls}"><span class="dot"></span>${esc(s)}</span>`).join("")}</div></div>`
    : "";
  setHTML("skills",
    group("bad", "Required but missing", missing) +
    group("ok", "Matched", matched) +
    group("info", "In resume, not requested", extra) +
    (total ? "" : `<p class="caption">No known skills detected in the job description.</p>`));

  // Weakest-covered requirements
  if (r.coverage?.detail.length) {
    setHTML("coverage-table", r.coverage.detail.slice(0, 8).map((d) => `
      <div class="req">
        <div class="req-top"><span class="req-text">${esc(unbullet(d.requirement))}</span><span class="req-score">${d.coverage.toFixed(2)}</span></div>
        <div class="meter"><span style="width:${(clamp01(d.coverage) * 100).toFixed(0)}%"></span></div>
        <div class="req-match">${esc(unbullet(d.bestMatch))}</div>
      </div>`).join(""));
  } else if (r.coverageError) {
    setHTML("coverage-table", `<div class="callout warn">${esc(r.coverageError)}</div>`);
  } else {
    setHTML("coverage-table", SKELETON);
  }

  // Quality
  $("q-words").textContent = q.word_count;
  $("q-bullets").textContent = q.n_bullets;
  $("q-verbs").textContent = fmtPct(q.pct_action_verb);
  $("q-nums").textContent = fmtPct(q.pct_quantified);
  const checkChips = (obj) => Object.entries(obj).map(([k, ok]) =>
    `<span class="chip ${ok ? "ok" : "bad"}">${ok ? "✓" : "✕"} ${esc(k)}</span>`).join("");
  setHTML("q-sections", checkChips(q.sections));
  setHTML("q-contact", checkChips(q.contact));
  setHTML("findings", q.findings.length
    ? q.findings.map((f) => `<details class="finding"${f.severity === "high" ? " open" : ""}>
        <summary><span class="sev ${f.severity}"></span>${esc(f.title)}<span class="sev-tag">${f.severity}</span></summary>
        <div class="finding-body"><p>${esc(f.detail)}</p>${f.examples.map((ex) => `<blockquote>${esc(ex)}</blockquote>`).join("")}</div>
      </details>`).join("")
    : `<div class="callout ok">No issues found by the rule-based checks.</div>`);

  // LLM
  if (r.llm) {
    const html = window.marked && window.DOMPurify
      ? window.DOMPurify.sanitize(window.marked.parse(r.llm))
      : `<pre>${esc(r.llm)}</pre>`;
    setHTML("llm-out", `<div class="llm-md">${html}</div><p class="caption">Generated by ${esc(llmModel)} on Groq. Check every suggestion before using it.</p>`);
  } else if (r.llmError) {
    setHTML("llm-out", `<div class="callout warn">${esc(r.llmError)}</div>`);
  } else if (r.llmPending) {
    setHTML("llm-out", `<p class="caption">Writing suggestions… this usually takes 5–15 seconds.</p>${SKELETON}`);
  } else {
    setHTML("llm-out", `<div class="callout info">Turn on <strong>AI rewrite suggestions</strong> above and analyze again to get bullet rewrites.</div>`);
  }

  // Scorer details
  const c = r.coverage;
  setHTML("debug", (c ? `<p>Resume split into <strong>${c.nResumeChunks}</strong> chunks, JD into <strong>${c.nJdChunks}</strong>.</p>` : "") +
    `<table class="simple"><thead><tr><th>Scorer</th><th>Score</th></tr></thead><tbody>
      <tr><td>Chunked embedding (all-MiniLM-L6-v2, in browser)</td><td>${c?.score == null ? "n/a" : c.score.toFixed(4)}</td></tr>
      <tr><td>TF-IDF baseline</td><td>${r.tfidf == null ? "n/a" : r.tfidf.toFixed(4)}</td></tr>
    </tbody></table>
    <p class="caption">MiniLM cosine scores rarely approach 0, so read them relative to each other rather than as absolute percentages.</p>`);
}

$("btn-report").addEventListener("click", () => {
  if (!lastResult) return;
  const md = buildMarkdownReport(lastResult);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], { type: "text/markdown" }));
  a.download = "resume_report.md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// Drag-and-drop highlight (the file input itself receives the drop)
const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, () => dropzone.classList.add("dragover")));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, () => dropzone.classList.remove("dragover")));

// Tabs
document.querySelectorAll('[role="tab"]').forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll('[role="tab"]').forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== tab.dataset.tab; });
  });
});

// Is the LLM configured on the server?
fetch("/api/health").then((r) => r.json()).then((h) => {
  llmModel = h.llm_model || "";
  if (h.llm_configured) {
    els.llmLabel.textContent = "AI rewrite suggestions";
    els.llmNote.textContent = `Powered by ${llmModel} on Groq · sends your text to Groq`;
  } else {
    els.useLlm.disabled = true;
    els.llmNote.textContent = "Currently unavailable (no Groq API key configured)";
  }
}).catch(() => {
  els.useLlm.disabled = true;
  els.llmNote.textContent = "Currently unavailable (API not reachable)";
});

refreshReady();
