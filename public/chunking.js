// Port of resume_refiner/chunking.py and EmbeddingScorer.score_chunked so the
// embedding step can run in the browser. tests/test_js_parity.py checks that
// the chunks produced here match the Python implementation.

/** Fallback sentence splitter (the Python app uses spaCy when installed). */
export function sentencize(text) {
  return text.split(/(?<=[.!?])\s+/);
}

export function splitIntoUnits(text, countTokens, maxTokens, sentencizer = null) {
  const units = [];
  for (let line of text.split("\n")) {
    line = line.trim();
    if (!line) continue;
    if (countTokens(line) <= maxTokens || !sentencizer) {
      units.push(line);
    } else {
      for (const s of sentencizer(line)) if (s.trim()) units.push(s.trim());
    }
  }
  return units;
}

export function packUnits(units, countTokens, maxTokens) {
  const chunks = [];
  let current = [];
  let currentTokens = 0;
  const flush = () => {
    if (current.length) chunks.push(current.join(" "));
    current = [];
    currentTokens = 0;
  };

  for (const unit of units) {
    const n = countTokens(unit);
    if (n > maxTokens) {
      // Too long even after sentence splitting: hard-split on words.
      flush();
      const words = unit.split(/\s+/).filter(Boolean);
      const step = Math.max(1, Math.floor(maxTokens * 0.6));
      let i = 0;
      while (i < words.length) {
        let size = step;
        let piece = words.slice(i, i + size).join(" ");
        while (size > 1 && countTokens(piece) > maxTokens) {
          size = Math.max(1, Math.floor(size / 2));
          piece = words.slice(i, i + size).join(" ");
        }
        chunks.push(piece);
        i += size;
      }
      continue;
    }
    if (current.length && currentTokens + n > maxTokens) flush();
    current.push(unit);
    currentTokens += n;
  }
  flush();
  return chunks.filter((c) => c.trim());
}

export function splitRequirements(text, countTokens, maxTokens, sentencizer = null, minWords = 4) {
  if (!text || !text.trim()) return [];
  const units = splitIntoUnits(text, countTokens, maxTokens, sentencizer);
  const reqs = units.filter((u) => u.split(/\s+/).filter(Boolean).length >= minWords);
  return reqs.flatMap((u) => packUnits([u], countTokens, maxTokens));
}

export function chunkText(text, countTokens, maxTokens, sentencizer = null) {
  if (!text || !text.trim()) return [];
  return packUnits(splitIntoUnits(text, countTokens, maxTokens, sentencizer), countTokens, maxTokens);
}

/**
 * JD-coverage score, aggregated max-then-mean.
 * `embed(texts)` must resolve to L2-normalised vectors (arrays of numbers).
 */
export async function scoreChunked(resumeText, jdText, { countTokens, embed, cfg }) {
  const resumeChunks = chunkText(resumeText, countTokens, cfg.resume_chunk_tokens, sentencize);
  let jdChunks = splitRequirements(jdText, countTokens, cfg.max_chunk_tokens, sentencize,
                                   cfg.min_requirement_words);
  if (!jdChunks.length) jdChunks = chunkText(jdText, countTokens, cfg.max_chunk_tokens, sentencize);
  if (!resumeChunks.length || !jdChunks.length) {
    return { score: null, detail: [], nResumeChunks: 0, nJdChunks: 0 };
  }

  const [jdVecs, resVecs] = [await embed(jdChunks), await embed(resumeChunks)];
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  const detail = jdVecs.map((jv, i) => {
    let best = -Infinity, bestJ = 0;
    resVecs.forEach((rv, j) => {
      const s = dot(jv, rv);
      if (s > best) { best = s; bestJ = j; }
    });
    return { requirement: jdChunks[i].slice(0, 180), coverage: best,
             bestMatch: resumeChunks[bestJ].slice(0, 180) };
  });
  const score = detail.reduce((s, d) => s + d.coverage, 0) / detail.length;
  detail.sort((a, b) => a.coverage - b.coverage);
  return { score, detail, nResumeChunks: resumeChunks.length, nJdChunks: jdChunks.length };
}
