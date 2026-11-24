from pathlib import Path

from resume_refiner.quality import analyze_quality, detect_sections, extract_bullets

SAMPLE = (Path(__file__).parent.parent / "public" / "samples" / "sample_resume.txt").read_text(encoding="utf-8")


def titles(report):
    return {f.title for f in report.findings}


def test_sample_resume_sections_and_contact():
    r = analyze_quality(SAMPLE)
    assert r.sections["Experience"] and r.sections["Education"] and r.sections["Skills"]
    assert all(r.contact[k] for k in ("Email", "Phone", "LinkedIn"))
    assert r.n_bullets >= 8


def test_sample_resume_flags_weak_phrases():
    t = titles(analyze_quality(SAMPLE))
    assert 'Weak phrase: "responsible for"' in t
    assert 'Weak phrase: "worked on"' in t
    assert 'Weak phrase: "helped"' in t


def test_missing_sections_and_contact():
    r = analyze_quality("Some text without any structure at all.")
    t = titles(r)
    assert "Missing standard sections" in t
    assert "No email address found" in t
    assert r.score < 60


def test_date_range_is_not_a_phone_number():
    r = analyze_quality("Analyst, Acme (2021 - 2023)")
    assert r.contact["Phone"] is False


def test_action_verb_and_quantified_rates():
    text = "\n".join([
        "EXPERIENCE",
        "- Reduced churn by 12% with a retention model",
        "- Built a dashboard for 30 managers",
        "- Responsible for weekly reporting to leadership team",
        "- Worked on data cleaning for the finance group",
    ])
    r = analyze_quality(text)
    assert r.n_bullets == 4
    assert r.pct_action_verb == 0.5
    assert r.pct_quantified == 0.5


def test_headings_detected_case_insensitively():
    s = detect_sections("Work Experience\nEDUCATION\nTechnical Skills:\nProjects")
    assert s["Experience"] and s["Education"] and s["Skills"] and s["Projects"]


def test_headings_are_not_bullets():
    assert extract_bullets("EXPERIENCE\nSKILLS") == []
