"""
Rule-based resume quality checks that don't depend on the job description.

These are the things a recruiter notices in the first ten seconds: missing
sections, bullets that describe duties instead of results, no numbers, and
filler phrases. Every check is a transparent rule so the feedback is explainable.
"""

import re
from dataclasses import dataclass, field
from typing import List

SECTION_PATTERNS = {
    "Summary": r"(professional\s+)?summary|profile|objective|about\s+me",
    "Experience": r"(work\s+|professional\s+)?experience|employment(\s+history)?|internships?",
    "Education": r"education|academic(\s+background)?|qualifications?",
    "Skills": r"(technical\s+|core\s+|key\s+)?skills|technologies|tech\s+stack|competencies",
    "Projects": r"(academic\s+|personal\s+|key\s+)?projects",
}
REQUIRED_SECTIONS = ("Experience", "Education", "Skills")

ACTION_VERBS = {
    "achieved", "analyzed", "analysed", "architected", "automated", "built",
    "championed", "collaborated", "conducted", "created", "cut", "decreased",
    "delivered", "deployed", "designed", "developed", "drove", "engineered",
    "enhanced", "established", "evaluated", "executed", "expanded",
    "facilitated", "generated", "grew", "identified", "implemented",
    "improved", "increased", "initiated", "integrated", "introduced",
    "launched", "led", "maintained", "managed", "mentored", "migrated",
    "modeled", "modelled", "monitored", "negotiated", "optimized",
    "optimised", "orchestrated", "organized", "organised", "overhauled",
    "partnered", "performed", "pioneered", "presented", "prototyped",
    "published", "redesigned", "reduced", "refactored", "resolved",
    "restructured", "saved", "scaled", "secured", "simplified", "spearheaded",
    "streamlined", "strengthened", "supervised", "tested", "trained",
    "transformed", "visualized", "visualised", "won", "wrote",
}

WEAK_PHRASES = {
    "responsible for": "Say what you achieved, not what you were assigned.",
    "duties included": "List outcomes rather than duties.",
    "worked on": "Name your specific contribution (built, designed, led...).",
    "helped": "Quantify your part: what did you personally deliver?",
    "assisted with": "Describe the piece you owned.",
    "involved in": "State your role and the result.",
    "various": "Be specific: name the things.",
    "etc": "Finish the list or cut it; 'etc.' reads as filler.",
    "hard-working": "Show it with a result instead of claiming it.",
    "team player": "Show collaboration through an example.",
    "go-getter": "Replace with evidence.",
    "detail-oriented": "Show it with an example instead of claiming it.",
}

BULLET_PREFIX = re.compile(r"^\s*([\-•●▪◦‣*–>]|\d+[.)])\s*")
NUMBER = re.compile(r"\d|%|\$|₹|€|£|\b(one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"dozens?|hundreds?|thousands?|millions?)\b", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
LINKEDIN = re.compile(r"linkedin\.com/", re.IGNORECASE)
GITHUB = re.compile(r"github\.com/", re.IGNORECASE)
FIRST_PERSON = re.compile(r"\b(I|me|my|myself)\b")


@dataclass
class Finding:
    severity: str          # "high" | "medium" | "low"
    title: str
    detail: str
    examples: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    score: int
    word_count: int
    sections: dict
    contact: dict
    n_bullets: int
    pct_action_verb: float
    pct_quantified: float
    findings: List[Finding]


def _heading_match(line: str) -> List[str]:
    """Return section names a short line looks like a heading for."""
    clean = re.sub(r"[^A-Za-z\s]", " ", line).strip().lower()
    if not clean or len(clean.split()) > 4:
        return []
    return [name for name, pat in SECTION_PATTERNS.items()
            if re.fullmatch(rf"\s*({pat})\s*", clean)]


def detect_sections(text: str) -> dict:
    found = {name: False for name in SECTION_PATTERNS}
    for line in text.splitlines():
        for name in _heading_match(line):
            found[name] = True
    return found


def extract_bullets(text: str) -> List[str]:
    """
    Lines that read like experience/project bullets: either prefixed with a
    bullet glyph or a sentence-length line (>= 6 words) that isn't a heading.
    """
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l and not _heading_match(l)]
    prefixed = [BULLET_PREFIX.sub("", l).strip() for l in lines if BULLET_PREFIX.match(l)]
    prefixed = [b for b in prefixed if len(b.split()) >= 3]
    if prefixed:
        # The resume uses bullet glyphs, so trust them and ignore titles,
        # contact lines and summary paragraphs.
        return prefixed
    # No glyphs (common after PDF extraction): fall back to sentence-length
    # lines that aren't contact/header lines.
    return [l for l in lines
            if len(l.split()) >= 6 and not EMAIL.search(l) and l.count("|") < 2]


def _has_phone(text: str) -> bool:
    """A run of phone-ish characters holding at least 10 digits (so date
    ranges like '2021 - 2023' don't count)."""
    return any(sum(c.isdigit() for c in m) >= 10 for m in PHONE.findall(text))


def _first_word(bullet: str) -> str:
    m = re.match(r"[A-Za-z]+", bullet)
    return m.group(0).lower() if m else ""


def analyze_quality(text: str) -> QualityReport:
    text = text or ""
    words = len(text.split())
    sections = detect_sections(text)
    contact = {
        "Email": bool(EMAIL.search(text)),
        "Phone": _has_phone(text),
        "LinkedIn": bool(LINKEDIN.search(text)),
        "GitHub": bool(GITHUB.search(text)),
    }
    bullets = extract_bullets(text)
    action = [b for b in bullets if _first_word(b) in ACTION_VERBS]
    quantified = [b for b in bullets if NUMBER.search(b)]
    n = len(bullets)
    pct_action = len(action) / n if n else 0.0
    pct_quant = len(quantified) / n if n else 0.0

    findings: List[Finding] = []

    missing_sections = [s for s in REQUIRED_SECTIONS if not sections[s]]
    if missing_sections:
        findings.append(Finding(
            "high", "Missing standard sections",
            "ATS parsers and recruiters look for these headings: "
            + ", ".join(missing_sections) + ".",
        ))

    if not contact["Email"]:
        findings.append(Finding("high", "No email address found",
                                "Put your email in the header."))
    if not contact["Phone"]:
        findings.append(Finding("medium", "No phone number found",
                                "Most recruiters still call first."))
    if not contact["LinkedIn"]:
        findings.append(Finding("low", "No LinkedIn URL",
                                "Add a linkedin.com/in/... link to the header."))

    if words < 120:
        findings.append(Finding("high", "Resume is very short",
                                f"{words} words. Aim for roughly 350-750 for one page."))
    elif words < 300:
        findings.append(Finding("medium", "Resume is on the short side",
                                f"{words} words. There's room to add projects, "
                                "results or tools; aim for roughly 350-750."))
    elif words > 1100:
        findings.append(Finding("medium", "Resume is long",
                                f"{words} words. Trim to the most relevant two pages at most."))

    if n:
        if pct_action < 0.6:
            weak_starts = [b for b in bullets if _first_word(b) not in ACTION_VERBS]
            findings.append(Finding(
                "medium", "Bullets don't lead with action verbs",
                f"Only {pct_action:.0%} of bullets start with a strong verb "
                "(e.g. Built, Led, Reduced). Aim for 80%+.",
                weak_starts[:3],
            ))
        if pct_quant < 0.4:
            unquantified = [b for b in bullets if not NUMBER.search(b)]
            findings.append(Finding(
                "high", "Few quantified results",
                f"Only {pct_quant:.0%} of bullets contain a number. Add scale, "
                "%, time saved, users, accuracy, revenue... Aim for 50%+.",
                unquantified[:3],
            ))
        long_bullets = [b for b in bullets if len(b.split()) > 35]
        if long_bullets:
            findings.append(Finding(
                "low", "Some bullets are too long",
                "Keep bullets to one or two lines (under ~30 words).",
                long_bullets[:2],
            ))
    else:
        findings.append(Finding(
            "medium", "No bullet points detected",
            "Describe experience and projects as bullet points; dense paragraphs "
            "are skimmed past.",
        ))

    lower = text.lower()
    for phrase, advice in WEAK_PHRASES.items():
        pat = re.compile(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])")
        if pat.search(lower):
            hits = [b for b in bullets if pat.search(b.lower())]
            findings.append(Finding("low", f'Weak phrase: "{phrase}"', advice, hits[:2]))

    pronoun_lines = [b for b in bullets if FIRST_PERSON.search(b)]
    if pronoun_lines:
        findings.append(Finding(
            "low", "First-person pronouns",
            "Resumes conventionally drop I / my / me.",
            pronoun_lines[:2],
        ))

    # Score: start at 100 and deduct per finding by severity.
    penalty = {"high": 15, "medium": 8, "low": 3}
    score = max(0, 100 - sum(penalty[f.severity] for f in findings))

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f.severity])

    return QualityReport(
        score=score,
        word_count=words,
        sections=sections,
        contact=contact,
        n_bullets=n,
        pct_action_verb=pct_action,
        pct_quantified=pct_quant,
        findings=findings,
    )


def check_grammar(text: str, nlp) -> List[str]:
    """Single heuristic: plural subject with a singular present-tense verb."""
    if nlp is None or not text:
        return []
    doc = nlp(text)
    return [
        f"Possible subject-verb agreement issue: '{t.text}' -> '{t.head.text}'"
        for t in doc
        if t.dep_ == "nsubj" and t.tag_ in ("NNS", "NNPS") and t.head.tag_ == "VBZ"
    ]
