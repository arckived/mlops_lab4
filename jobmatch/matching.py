"""Score how well each job matches a resume.

Three scorers, each returning a number from 0 to 1 per job:

    tfidf      word overlap between resume and job text, weighted so that rare,
               specific words (like "pyspark") count more than common ones
    skills     share of the job's required skills that the resume has
    embedding  meaning-based similarity from a sentence-transformers model
               (optional, only if the package is installed)

A RankerConfig says how much weight each scorer gets. MLflow experiments try
different configs and keep the one that ranks your own "would apply" jobs best.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jobmatch.skills import extract_skills

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RankerConfig:
    name: str = "hybrid"
    weights: dict = field(default_factory=lambda: {"tfidf": 0.5, "skills": 0.5})
    ngram_max: int = 2                # TF-IDF uses words (1) or words + word pairs (2)
    title_weight: int = 2             # repeat the job title this many times so it counts more
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RankerConfig":
        return cls(**d)

    @property
    def uses_embedding(self) -> bool:
        return self.weights.get("embedding", 0) > 0


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _job_text(title: str, description: str, title_weight: int) -> str:
    return " ".join([title] * max(title_weight, 1) + [description or ""])


def tfidf_scores(resume: str, docs: list[str], ngram_max: int) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, ngram_max),
                          min_df=1, max_df=0.9 if len(docs) > 10 else 1.0)
    matrix = vec.fit_transform(docs + [resume])
    return cosine_similarity(matrix[-1], matrix[:-1]).ravel()


def skill_scores(resume_skills: set, job_skills: list[set]) -> np.ndarray:
    # (matched + 1) / (required + 2): a job asking for 1 skill you have does not
    # score higher than a job asking for 10 skills you have.
    return np.array([(len(js & resume_skills) + 1) / (len(js) + 2) for js in job_skills])


_EMBEDDERS = {}


def embedding_scores(resume: str, docs: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if model_name not in _EMBEDDERS:
        _EMBEDDERS[model_name] = SentenceTransformer(model_name)
    model = _EMBEDDERS[model_name]
    vectors = model.encode(docs + [resume], normalize_embeddings=True, show_progress_bar=False)
    return np.clip(vectors[:-1] @ vectors[-1], 0, 1)


def score_jobs(resume: str, titles: list[str], descriptions: list[str], config: RankerConfig) -> dict:
    """Return each scorer's scores plus the weighted total, all as arrays aligned with the jobs."""
    if not titles:
        return {"total": np.array([])}
    docs = [_job_text(t, d, config.title_weight) for t, d in zip(titles, descriptions)]
    parts = {}
    if config.weights.get("tfidf", 0) > 0:
        parts["tfidf"] = tfidf_scores(resume, docs, config.ngram_max)
    if config.weights.get("skills", 0) > 0:
        resume_skills = extract_skills(resume)
        parts["skills"] = skill_scores(resume_skills, [extract_skills(d) for d in docs])
    if config.uses_embedding:
        parts["embedding"] = embedding_scores(resume, docs, config.embedding_model)

    total_weight = sum(config.weights.get(k, 0) for k in parts) or 1.0
    total = sum(config.weights[k] * v for k, v in parts.items()) / total_weight
    parts["total"] = total
    return parts


def rank_jobs(resume: str, jobs: list, config: RankerConfig):
    """Rank Job objects for a resume. Returns a list of dicts sorted best first."""
    import pandas as pd

    titles = [j.title for j in jobs]
    descriptions = [j.description for j in jobs]
    parts = score_jobs(resume, titles, descriptions, config)
    resume_skills = extract_skills(resume)
    rows = []
    for i, job in enumerate(jobs):
        job_skills = extract_skills(job.title + "\n" + job.description)
        rows.append({
            **job.to_dict(),
            "score": float(parts["total"][i]) if len(parts["total"]) else 0.0,
            **{f"score_{k}": float(v[i]) for k, v in parts.items() if k != "total"},
            "matched_skills": sorted(job_skills & resume_skills),
            "missing_skills": sorted(job_skills - resume_skills),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
