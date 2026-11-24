"""Build a downloadable Markdown report of an analysis."""

from datetime import date
from typing import List, Optional

from .quality import QualityReport
from .scoring import CoverageResult


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def build_markdown_report(coverage: CoverageResult, tfidf: Optional[float],
                          matched: List[str], missing: List[str], extra: List[str],
                          quality: QualityReport, grammar: List[str],
                          llm_feedback: Optional[str] = None) -> str:
    lines = [
        "# ResumeRefiner AI report",
        f"_Generated {date.today().isoformat()}_",
        "",
        "## Match scores",
        f"- JD coverage (chunked embedding): **{_pct(coverage.score)}**",
        f"- TF-IDF keyword overlap: {_pct(tfidf)}",
        f"- Resume quality score: **{quality.score}/100**",
        "",
        "## Skills",
        f"- Matched: {', '.join(matched) or 'none'}",
        f"- Required but missing: {', '.join(missing) or 'none'}",
        f"- In resume but not requested: {', '.join(extra) or 'none'}",
        "",
        "## Weakest-covered requirements",
    ]
    if not coverage.detail.empty:
        for _, row in coverage.detail.head(5).iterrows():
            lines.append(f"- ({row['Coverage']:.0%}) {row['JD requirement']}")
    else:
        lines.append("- n/a")

    lines += [
        "",
        "## Resume quality",
        f"- Words: {quality.word_count}",
        f"- Bullets: {quality.n_bullets} "
        f"({quality.pct_action_verb:.0%} start with an action verb, "
        f"{quality.pct_quantified:.0%} quantified)",
        "",
    ]
    for f in quality.findings:
        lines.append(f"- **[{f.severity.upper()}] {f.title}**: {f.detail}")
        for ex in f.examples:
            lines.append(f"    - > {ex}")

    if grammar:
        lines += ["", "## Grammar"] + [f"- {g}" for g in grammar]

    if llm_feedback:
        lines += ["", "## LLM feedback", "", llm_feedback]

    return "\n".join(lines) + "\n"
