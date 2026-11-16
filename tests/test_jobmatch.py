"""Unit tests. The fixtures in tests/fixtures are small hand-written examples in the
response format of each job board API; they test parsing, not live data."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jobmatch.companies import parse_company_line
from jobmatch.filters import filter_jobs, is_us, job_level, posted_within, role_family
from jobmatch.matching import RankerConfig, rank_jobs
from jobmatch.metrics import average_precision, ndcg_at_k, precision_at_k
from jobmatch.skills import extract_skills
from jobmatch.sources import Job, html_to_text, parse_ashby, parse_greenhouse, parse_lever

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIX / name).read_text())


# ---------- Sources ----------

def test_parse_greenhouse():
    jobs = parse_greenhouse(load("greenhouse.json"), "Acme")
    assert [j.title for j in jobs] == ["Machine Learning Engineer, New Grad", "Senior Account Executive"]
    first = jobs[0]
    assert first.location == "San Francisco, CA"
    assert first.posted_at.startswith("2026-11-08")
    assert "PyTorch" in first.description and "<" not in first.description
    assert first.key == "greenhouse:Acme:101"


def test_parse_lever_includes_lists_and_all_locations():
    job = parse_lever(load("lever.json"), "Acme")[0]
    assert "Spark" in job.description and "Write SQL" in job.description
    assert job.location == "New York, NY / Remote - US"
    assert job.posted_at.startswith("2025-11-12")
    assert job.extra["commitment"] == "Intern"


def test_parse_ashby_skips_unlisted_and_marks_remote():
    jobs = parse_ashby(load("ashby.json"), "Acme")
    assert len(jobs) == 1
    assert jobs[0].location == "Boston, MA / Toronto, Canada / Remote"


def test_html_to_text_handles_double_escaping():
    assert html_to_text("&amp;lt;p&amp;gt;Hi&amp;lt;/p&amp;gt;") == "Hi"


def test_parse_company_line():
    assert parse_company_line("greenhouse:stripe") == ("greenhouse", "stripe", "Stripe")
    assert parse_company_line("ashby:my-co:My Co") == ("ashby", "my-co", "My Co")
    with pytest.raises(ValueError):
        parse_company_line("linkedin:stripe")


# ---------- Skills ----------

def test_extract_skills_aliases_and_case():
    text = "Built ETL in PySpark, deployed with k8s and CI/CD. Used sklearn. Wrote in R and Go."
    found = extract_skills(text)
    assert {"Spark", "Kubernetes", "CI/CD", "scikit-learn", "ETL", "R", "Go"} <= found


def test_extract_skills_avoids_common_words():
    found = extract_skills("Go to the store, our team will excel and look at the spark of ideas. r")
    assert not {"Go", "Excel", "Spark", "R"} & found


# ---------- Filters ----------

@pytest.mark.parametrize("title,expected", [
    ("Machine Learning Engineer Intern", "intern"),
    ("Data Science Co-op (Spring 2027)", "intern"),
    ("Data Engineer, New Grad 2027", "new grad"),
    ("Senior Data Scientist", "senior"),
    ("Staff ML Engineer", "senior"),
    ("Machine Learning Engineer", "unspecified"),
])
def test_job_level(title, expected):
    assert job_level(title) == expected


@pytest.mark.parametrize("location,expected", [
    ("San Francisco, CA", True), ("Remote - US", True), ("Remote", True), ("New York", True),
    ("London, UK", False), ("Remote - Canada", False), ("Toronto, ON / Seattle, WA", True), ("", False),
])
def test_is_us(location, expected):
    assert is_us(location) is expected


def test_role_family():
    fams = ["ML / AI Engineer", "Data Scientist", "Data Engineer"]
    assert role_family("Applied Scientist, LLMs", fams) == "ML / AI Engineer"
    assert role_family("Analytics Engineer", fams) == "Data Scientist"  # 'analytics' listed under DS first
    assert role_family("Account Executive", fams) is None


def test_posted_within():
    now = datetime(2026, 11, 16, tzinfo=timezone.utc)
    assert posted_within("2026-11-10T00:00:00Z", 30, now)
    assert not posted_within("2026-09-01T00:00:00Z", 30, now)
    assert posted_within("", 30, now)


def test_filter_jobs_end_to_end():
    jobs = parse_greenhouse(load("greenhouse.json"), "Acme") + parse_lever(load("lever.json"), "Acme")
    kept = filter_jobs(jobs, ["ML / AI Engineer", "Data Engineer"], {"intern", "new grad"},
                       us_only=True, max_age_days=0)
    titles = {j.title for j, _, _ in kept}
    assert titles == {"Machine Learning Engineer, New Grad", "Data Engineer Intern (Summer 2027)"}


# ---------- Matching and metrics ----------

def make_job(i, title, desc):
    return Job("greenhouse", "Acme", str(i), title, "Remote - US", "", desc)


RESUME = """Machine learning engineer. Python, PyTorch, scikit-learn, SQL, Docker, Kubernetes,
MLflow, Airflow. Built model serving with FastAPI and monitoring with Prometheus."""


def test_ranking_prefers_relevant_jobs():
    jobs = [
        make_job(1, "Account Executive", "Sell software to enterprise customers. Salesforce, negotiation."),
        make_job(2, "Machine Learning Engineer", "Python, PyTorch, Docker, Kubernetes, MLflow model serving."),
        make_job(3, "Data Analyst", "Excel, Tableau dashboards, SQL reporting."),
    ]
    for config in [RankerConfig("t", {"tfidf": 1.0}), RankerConfig("s", {"skills": 1.0}), RankerConfig()]:
        ranked = rank_jobs(RESUME, jobs, config)
        assert ranked.iloc[0]["title"] == "Machine Learning Engineer", config.name
    ranked = rank_jobs(RESUME, jobs, RankerConfig())
    ml = ranked[ranked.title == "Machine Learning Engineer"].iloc[0]
    assert "PyTorch" in ml.matched_skills
    analyst = ranked[ranked.title == "Data Analyst"].iloc[0]
    assert "Tableau" in analyst.missing_skills


def test_metrics():
    assert precision_at_k([1, 0, 1, 0], 2) == 0.5
    assert ndcg_at_k([1, 1, 0, 0], 10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 0, 1, 1], 10) < 0.7
    assert average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
