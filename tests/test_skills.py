import pytest

from resume_refiner.skills import compare_skills, find_skills


@pytest.mark.parametrize("text, skill, expected", [
    ("Built UIs in JavaScript", "Java", False),
    ("Backend services in Java and Spring", "Java", True),
    ("Fluent in C++ and Python", "C++", True),
    ("Wrote HTML and CSS", "Machine Learning", False),
    ("Applied ML to churn", "Machine Learning", True),
    ("Used sklearn pipelines", "Scikit-learn", True),
    ("Used Scikit Learn pipelines", "Scikit-learn", True),
    ("Developed in C# / .NET", "C#", True),
    ("Please find my CV attached", "Computer Vision", False),
    ("Object detection with OpenCV", "Computer Vision", True),
    ("Queried PostgreSQL daily", "SQL", True),
])
def test_boundary_matching(text, skill, expected):
    assert (skill in find_skills(text)) is expected


def test_empty_text():
    assert find_skills("") == set()
    assert find_skills(None) == set()


def test_compare_skills_partitions():
    matched, missing, extra = compare_skills(
        "Python, SQL, Tableau", "We need Python, SQL and Snowflake")
    assert matched == ["Python", "SQL"]
    assert missing == ["Snowflake"]
    assert extra == ["Tableau"]
