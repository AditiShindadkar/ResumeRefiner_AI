// Port of resume_refiner/report.py: Markdown report of one analysis.

const pct = (x, digits = 1) => (x == null ? "n/a" : `${(x * 100).toFixed(digits)}%`);

export function buildMarkdownReport({ coverage, tfidf, skills, quality, llm }, today = new Date()) {
  const { matched, missing, extra } = skills;
  const lines = [
    "# ResumeRefiner AI report",
    `_Generated ${today.toISOString().slice(0, 10)}_`,
    "",
    "## Match scores",
    `- JD coverage (chunked embedding): **${pct(coverage?.score)}**`,
    `- TF-IDF keyword overlap: ${pct(tfidf)}`,
    `- Resume quality score: **${quality.score}/100**`,
    "",
    "## Skills",
    `- Matched: ${matched.join(", ") || "none"}`,
    `- Required but missing: ${missing.join(", ") || "none"}`,
    `- In resume but not requested: ${extra.join(", ") || "none"}`,
    "",
    "## Weakest-covered requirements",
  ];
  if (coverage?.detail?.length) {
    for (const row of coverage.detail.slice(0, 5)) {
      lines.push(`- (${pct(row.coverage, 0)}) ${row.requirement}`);
    }
  } else {
    lines.push("- n/a");
  }

  lines.push(
    "",
    "## Resume quality",
    `- Words: ${quality.word_count}`,
    `- Bullets: ${quality.n_bullets} (${pct(quality.pct_action_verb, 0)} start with an action verb, ` +
      `${pct(quality.pct_quantified, 0)} quantified)`,
    "",
  );
  for (const f of quality.findings) {
    lines.push(`- **[${f.severity.toUpperCase()}] ${f.title}**: ${f.detail}`);
    for (const ex of f.examples) lines.push(`    - > ${ex}`);
  }
  if (llm) lines.push("", "## LLM feedback", "", llm);
  return lines.join("\n") + "\n";
}
