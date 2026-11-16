"""End-to-end MLflow test: experiments run, the best method is registered,
promoted to champion, and loads back as a pyfunc model."""

import pandas as pd
import pytest

from jobmatch.sources import Job


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("JOBMATCH_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    from jobmatch import tracking as t
    return t


RESUME = "Machine learning engineer. Python, PyTorch, SQL, Docker, Kubernetes, MLflow, Airflow, Spark."


def labeled_set():
    good = ["Machine Learning Engineer", "MLOps Engineer", "Data Engineer", "ML Platform Engineer",
            "Applied Scientist"]
    bad = ["Account Executive", "Recruiter", "Office Manager", "Brand Designer", "Legal Counsel"]
    labels = {}
    for i, title in enumerate(good):
        job = Job("greenhouse", "Acme", f"g{i}", title, "Remote - US", "",
                  "Python PyTorch Docker Kubernetes MLflow Spark SQL pipelines and model deployment.")
        labels[job.key] = {"job": job, "label": 1, "labeled_at": ""}
    for i, title in enumerate(bad):
        job = Job("greenhouse", "Acme", f"b{i}", title, "Remote - US", "",
                  "Build relationships, manage calendars, negotiate contracts, design brand assets.")
        labels[job.key] = {"job": job, "label": 0, "labeled_at": ""}
    return labels


def test_needs_enough_labels(tracking):
    labels = dict(list(labeled_set().items())[:3])
    with pytest.raises(ValueError, match="Label at least"):
        tracking.run_experiments(RESUME, labels)


def test_experiment_registers_and_promotes(tracking):
    configs = tracking.candidate_configs(include_embedding=False)
    first = tracking.run_experiments(RESUME, labeled_set(), configs)
    assert first["promoted"] and first["registered_version"] == "1"
    assert first["results"][0]["ndcg_at_10"] == pytest.approx(1.0)

    model, version = tracking.load_champion_model()
    assert version == "1"
    scores = model.predict(pd.DataFrame({
        "resume": [RESUME] * 2,
        "title": ["Machine Learning Engineer", "Recruiter"],
        "description": ["Python PyTorch Kubernetes", "Hiring and interviews"],
    }))
    assert scores.iloc[0] > scores.iloc[1]

    # Same labels again: the new best cannot beat the champion by the margin, so it is not promoted.
    second = tracking.run_experiments(RESUME, labeled_set(), configs)
    assert second["registered_version"] == "2"
    assert not second["promoted"]
    assert tracking.get_champion()[1] == "1"
