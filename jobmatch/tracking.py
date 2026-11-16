"""MLflow experiments, model registry and promotion of the best matching method.

How it works:
1. You label jobs in the app as "would apply" (1) or "not for me" (0).
2. run_experiments() scores those labeled jobs with every candidate method,
   logs each method as a nested MLflow run with its ranking metrics, and picks
   the best one by NDCG@10.
3. The best method is logged as an MLflow pyfunc model and registered as a new
   version of "job-matcher".
4. Quality gate: it only gets the "champion" alias if it beats the current
   champion on the same labels (re-scored now, so the comparison is fair).
5. The app loads models:/job-matcher@champion to rank jobs.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from jobmatch.matching import RankerConfig, embeddings_available, score_jobs
from jobmatch.metrics import average_precision, ndcg_at_k, precision_at_k

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "job-matcher"
MODEL_NAME = "job-matcher"
CHAMPION = "champion"
MIN_LABELS = 10
MIN_IMPROVEMENT = 0.01   # a challenger must beat the champion's NDCG@10 by at least this much


def setup_mlflow() -> str:
    """Use a local SQLite database (needed for the model registry) unless MLFLOW_TRACKING_URI is set."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifacts = Path(os.getenv("JOBMATCH_ARTIFACT_DIR", ROOT / "mlartifacts")).resolve()
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


class JobRankerModel(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper. Input: a DataFrame with columns resume, title, description.
    Output: one match score (0 to 1) per row."""

    def load_context(self, context):
        with open(context.artifacts["config"]) as f:
            self.config = RankerConfig.from_dict(json.load(f))

    def predict(self, context, model_input, params=None):
        resume = str(model_input["resume"].iloc[0])
        parts = score_jobs(resume, model_input["title"].tolist(),
                           model_input["description"].tolist(), self.config)
        return pd.Series(parts["total"], name="score")


def candidate_configs(include_embedding: bool | None = None) -> list[RankerConfig]:
    """The matching methods compared in every experiment."""
    if include_embedding is None:
        include_embedding = embeddings_available()
    configs = [
        RankerConfig("tfidf_words", {"tfidf": 1.0}, ngram_max=1),
        RankerConfig("tfidf_phrases", {"tfidf": 1.0}, ngram_max=2),
        RankerConfig("skills_only", {"skills": 1.0}),
        RankerConfig("hybrid_text_heavy", {"tfidf": 0.7, "skills": 0.3}),
        RankerConfig("hybrid_balanced", {"tfidf": 0.5, "skills": 0.5}),
        RankerConfig("hybrid_skills_heavy", {"tfidf": 0.3, "skills": 0.7}),
        RankerConfig("hybrid_no_title_boost", {"tfidf": 0.5, "skills": 0.5}, title_weight=1),
    ]
    if include_embedding:
        configs += [
            RankerConfig("embedding_only", {"embedding": 1.0}),
            RankerConfig("hybrid_embedding", {"embedding": 0.5, "tfidf": 0.2, "skills": 0.3}),
        ]
    return configs


def evaluate(config: RankerConfig, resume: str, labeled: list) -> tuple[dict, pd.DataFrame]:
    """Score labeled jobs with one config. labeled: list of (Job, label)."""
    jobs = [j for j, _ in labeled]
    labels = np.array([y for _, y in labeled])
    scores = score_jobs(resume, [j.title for j in jobs], [j.description for j in jobs], config)["total"]
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    metrics = {
        "ndcg_at_10": ndcg_at_k(ranked, 10),
        "precision_at_5": precision_at_k(ranked, 5),
        "precision_at_10": precision_at_k(ranked, 10),
        "average_precision": average_precision(ranked),
    }
    table = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "company": [jobs[i].company for i in order],
        "title": [jobs[i].title for i in order],
        "label": ranked,
        "score": np.round(scores[order], 4),
    })
    return metrics, table


def _flat_params(config: RankerConfig) -> dict:
    params = {"method": config.name, "ngram_max": config.ngram_max, "title_weight": config.title_weight}
    for k in ("tfidf", "skills", "embedding"):
        params[f"weight_{k}"] = config.weights.get(k, 0.0)
    if config.uses_embedding:
        params["embedding_model"] = config.embedding_model
    return params


def get_champion():
    """Return (RankerConfig, version) of the current champion, or (None, None)."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except Exception:
        return None, None
    return RankerConfig.from_dict(json.loads(mv.tags["config_json"])), str(mv.version)


def load_champion_model():
    """Load the champion as an MLflow pyfunc model. Returns (model, version) or (None, None)."""
    _, version = get_champion()
    if version is None:
        return None, None
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION}"), version


def run_experiments(resume: str, labels: dict, configs: list[RankerConfig] | None = None) -> dict:
    """Compare matching methods on your labeled jobs and register the best one.

    labels: output of storage.load_labels().
    Returns a summary dict for the app.
    """
    labeled = [(v["job"], int(v["label"])) for v in labels.values()]
    n_pos = sum(y for _, y in labeled)
    n_neg = len(labeled) - n_pos
    if len(labeled) < MIN_LABELS or n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Label at least {MIN_LABELS} jobs, including both 'would apply' and 'not for me' "
            f"(now: {n_pos} would apply, {n_neg} not for me)."
        )
    if not resume.strip():
        raise ValueError("Upload your resume first.")

    setup_mlflow()
    configs = configs or candidate_configs()
    results = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    with mlflow.start_run(run_name=f"experiment {stamp}") as parent:
        mlflow.log_params({"n_labeled": len(labeled), "n_would_apply": n_pos,
                           "n_not_for_me": n_neg, "n_methods": len(configs)})
        for config in configs:
            with mlflow.start_run(run_name=config.name, nested=True) as child:
                metrics, table = evaluate(config, resume, labeled)
                mlflow.log_params(_flat_params(config))
                mlflow.log_metrics(metrics)
                mlflow.log_dict(config.to_dict(), "config.json")
                mlflow.log_table(table, "ranking.json")
                results.append({"run_id": child.info.run_id, "config": config, **metrics})

        results.sort(key=lambda r: (r["ndcg_at_10"], r["precision_at_10"], r["average_precision"]),
                     reverse=True)
        best = results[0]

        # Quality gate against the current champion, re-scored on today's labels.
        champ_config, champ_version = get_champion()
        champ_ndcg = None
        if champ_config is not None:
            champ_ndcg = evaluate(champ_config, resume, labeled)[0]["ndcg_at_10"]
        promote = champ_ndcg is None or best["ndcg_at_10"] >= champ_ndcg + MIN_IMPROVEMENT
        if champ_ndcg is None:
            reason = "No champion yet."
        elif promote:
            reason = f"NDCG@10 {best['ndcg_at_10']:.3f} beats champion v{champ_version} ({champ_ndcg:.3f})."
        else:
            reason = (f"NDCG@10 {best['ndcg_at_10']:.3f} does not beat champion v{champ_version} "
                      f"({champ_ndcg:.3f}) by {MIN_IMPROVEMENT}. Champion kept.")

        # Log the best method as an MLflow model inside its own run, then register it.
        with mlflow.start_run(run_id=best["run_id"], nested=True):
            with tempfile.TemporaryDirectory() as tmp:
                cfg_path = Path(tmp) / "config.json"
                cfg_path.write_text(json.dumps(best["config"].to_dict()))
                example = pd.DataFrame({"resume": ["python sql machine learning"],
                                        "title": ["Machine Learning Engineer"],
                                        "description": ["Python, PyTorch and SQL."]})
                model_info = mlflow.pyfunc.log_model(
                    name="ranker",
                    python_model=JobRankerModel(),
                    artifacts={"config": str(cfg_path)},
                    code_paths=[str(ROOT / "jobmatch")],
                    input_example=example,
                    pip_requirements=["scikit-learn", "pandas", "numpy"],
                )
        version = str(mlflow.register_model(model_info.model_uri, MODEL_NAME).version)
        client = MlflowClient()
        client.set_model_version_tag(MODEL_NAME, version, "config_json",
                                     json.dumps(best["config"].to_dict()))
        client.set_model_version_tag(MODEL_NAME, version, "ndcg_at_10", f"{best['ndcg_at_10']:.4f}")
        client.set_model_version_tag(MODEL_NAME, version, "method", best["config"].name)
        if promote:
            client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)

        mlflow.set_tags({"best_method": best["config"].name, "registered_version": version,
                         "promoted": str(promote), "promotion_reason": reason})
        mlflow.log_metric("best_ndcg_at_10", best["ndcg_at_10"])

    return {
        "parent_run_id": parent.info.run_id,
        "results": [{**{k: v for k, v in r.items() if k != "config"}, "method": r["config"].name}
                    for r in results],
        "best_method": best["config"].name,
        "registered_version": version,
        "promoted": promote,
        "reason": reason,
        "previous_champion_ndcg": champ_ndcg,
    }
