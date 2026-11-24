"""
Boundary-aware skill extraction.

A plain substring test (`"java" in text`) reports Java for every resume that
says JavaScript. Here each surface form is matched only when the characters on
either side are outside the identifier alphabet [A-Za-z0-9+#_].
"""

import re

# Canonical name -> surface forms seen in real documents.
# Expanding this is the cheapest accuracy win in the whole app.
SKILL_ALIASES = {
    # Languages
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "JavaScript": ["javascript", "java script", "js", "es6"],
    "TypeScript": ["typescript"],
    "R": ["r programming", "r language"],  # bare "R" is too noisy to match
    "SQL": ["sql", "mysql", "postgresql", "postgres"],
    "NoSQL": ["nosql", "mongodb"],
    "Scala": ["scala"],
    "Go": ["golang"],

    # Data / BI
    "Power BI": ["power bi", "powerbi"],
    "DAX": ["dax"],
    "Power Query": ["power query"],
    "Tableau": ["tableau"],
    "Looker": ["looker"],
    "Excel": ["excel", "ms excel", "microsoft excel"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Seaborn": ["seaborn"],
    "Plotly": ["plotly"],
    "Spark": ["spark", "pyspark", "apache spark"],
    "Airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "BigQuery": ["bigquery", "big query"],
    "ETL": ["etl", "data pipeline", "data pipelines"],

    # ML / DS
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis", "data analytics", "data analyst"],
    "Statistics": ["statistics", "statistical analysis"],
    "A/B Testing": ["a/b testing", "ab testing", "hypothesis testing"],
    "Time Series": ["time series", "forecasting"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "TensorFlow": ["tensorflow", "tensor flow"],
    "PyTorch": ["pytorch", "py torch"],
    "Keras": ["keras"],
    "XGBoost": ["xgboost", "xg boost"],
    "LightGBM": ["lightgbm", "light gbm"],
    "SHAP": ["shap"],
    "NLP": ["nlp", "natural language processing"],
    # Not bare "cv": on a resume that almost always means curriculum vitae.
    "Computer Vision": ["computer vision", "opencv", "image classification",
                        "object detection"],
    "Hugging Face": ["hugging face", "huggingface"],
    "LLM": ["llm", "llms", "large language model", "large language models"],
    "RAG": ["rag", "retrieval augmented generation",
            "retrieval-augmented generation"],

    # Engineering / infra
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Linux": ["linux", "unix"],
    "CI/CD": ["ci/cd", "github actions", "jenkins"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi", "fast api"],
    "Streamlit": ["streamlit"],
    "REST API": ["rest api", "restful", "rest apis"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs", "node js"],

    # Soft / process
    "Project Management": ["project management"],
    "Agile": ["agile"],
    "Scrum": ["scrum"],
    "Stakeholder Management": ["stakeholder management", "stakeholders"],
    "Communication": ["communication"],
    "Leadership": ["leadership"],
    "Teamwork": ["teamwork", "collaboration", "cross-functional"],
}

_EDGE = r"[A-Za-z0-9+#_]"


def _compile(surface: str) -> re.Pattern:
    """
    Build a boundary-aware pattern for one surface form.

    `\\b` can't be used: there is no word boundary after the `+` in `C++`.

        "java" vs "JavaScript" -> blocked, next char 'S' is alphabetic
        "c++"  vs "C++"        -> matches
        "ml"   vs "html"       -> blocked, previous char 't' is alphabetic
    """
    return re.compile(rf"(?<!{_EDGE}){re.escape(surface)}(?!{_EDGE})", re.IGNORECASE)


COMPILED_SKILLS = {
    canonical: [_compile(s) for s in surfaces]
    for canonical, surfaces in SKILL_ALIASES.items()
}


def find_skills(text: str) -> set:
    """Return the set of canonical skill names present in `text`."""
    if not text:
        return set()
    return {
        canonical
        for canonical, patterns in COMPILED_SKILLS.items()
        if any(p.search(text) for p in patterns)
    }


def compare_skills(resume_text: str, jd_text: str):
    """Return (matched, missing, extra) skill lists, each sorted."""
    resume_skills = find_skills(resume_text)
    jd_skills = find_skills(jd_text)
    return (
        sorted(jd_skills & resume_skills),
        sorted(jd_skills - resume_skills),
        sorted(resume_skills - jd_skills),
    )
