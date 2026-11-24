from resume_refiner.chunking import (chunk_text, pack_units, split_into_units,
                                     split_requirements)


def word_count(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace word."""
    return len(text.split())


def test_empty():
    assert chunk_text("", word_count) == []
    assert chunk_text("   \n  ", word_count) == []


def test_short_lines_are_packed_together():
    text = "Python SQL\nPower BI\nExcel"
    assert chunk_text(text, word_count, max_tokens=10) == ["Python SQL Power BI Excel"]


def test_chunks_respect_budget():
    lines = [" ".join(f"w{i}_{j}" for j in range(7)) for i in range(20)]
    chunks = chunk_text("\n".join(lines), word_count, max_tokens=20)
    assert len(chunks) > 1
    assert all(word_count(c) <= 20 for c in chunks)
    # Nothing lost.
    assert sum(word_count(c) for c in chunks) == 140


def test_overlong_unit_is_hard_split():
    unit = " ".join(f"w{i}" for i in range(55))
    chunks = pack_units([unit], word_count, max_tokens=10)
    assert all(word_count(c) <= 10 for c in chunks)
    assert " ".join(chunks) == unit


def test_hard_split_respects_budget_with_dense_tokenizer():
    """Tokenizer that charges 3 pieces per word must still be honoured."""
    dense = lambda s: 3 * len(s.split())
    unit = " ".join(f"w{i}" for i in range(40))
    chunks = pack_units([unit], dense, max_tokens=12)
    assert all(dense(c) <= 12 for c in chunks)
    assert " ".join(chunks) == unit


def test_requirements_are_not_packed_and_skip_headings():
    jd = ("Requirements\n- 3+ years of SQL and Python\n"
          "- Experience with Snowflake or BigQuery\nNice to have")
    assert split_requirements(jd, word_count, max_tokens=200) == [
        "- 3+ years of SQL and Python",
        "- Experience with Snowflake or BigQuery",
    ]


def test_sentencizer_used_for_long_lines():
    line = "First sentence here. Second sentence here."
    units = split_into_units(line, word_count, max_tokens=4,
                             sentencizer=lambda s: s.split(". "))
    assert units == ["First sentence here", "Second sentence here."]
