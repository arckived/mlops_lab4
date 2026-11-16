"""Local files: resume text, cached job list and your labels.

Everything lives in the data/ folder, which is in .gitignore so your resume
and job ratings are never committed to GitHub.
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from jobmatch.sources import Job

DATA_DIR = Path(os.getenv("JOBMATCH_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
RESUME_PATH = DATA_DIR / "resume.txt"
JOBS_CACHE = DATA_DIR / "jobs_cache.json"
LABELS_PATH = DATA_DIR / "labeled_jobs.jsonl"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Resume ----------

def extract_resume_text(filename: str, content: bytes) -> str:
    """Read text from a PDF, Word (.docx) or plain text resume."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if name.endswith(".docx"):
        import docx

        document = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in document.paragraphs).strip()
    return content.decode("utf-8", errors="ignore").strip()


def save_resume(text: str) -> None:
    _ensure()
    RESUME_PATH.write_text(text, encoding="utf-8")


def load_resume() -> str:
    return RESUME_PATH.read_text(encoding="utf-8") if RESUME_PATH.exists() else ""


# ---------- Job cache ----------

def save_jobs(jobs: list[Job], errors: dict) -> None:
    _ensure()
    JOBS_CACHE.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "errors": errors,
        "jobs": [j.to_dict() for j in jobs],
    }))


def load_jobs():
    """Return (jobs, errors, fetched_at). Empty if nothing has been fetched yet."""
    if not JOBS_CACHE.exists():
        return [], {}, None
    data = json.loads(JOBS_CACHE.read_text())
    return [Job.from_dict(j) for j in data["jobs"]], data.get("errors", {}), data.get("fetched_at")


# ---------- Labels ----------
# Each line stores the full job plus your label, so the evaluation set still
# works after the posting is taken down.

def load_labels() -> dict:
    """Return {job key: {"job": Job, "label": 0 or 1, "labeled_at": str}}; the latest label wins."""
    labels = {}
    if LABELS_PATH.exists():
        for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["key"]] = {"job": Job.from_dict(row["job"]), "label": row["label"],
                                      "labeled_at": row["labeled_at"]}
    return labels


def save_label(job: Job, label: int) -> None:
    _ensure()
    with open(LABELS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "key": job.key,
            "label": int(label),
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "job": {k: v for k, v in job.to_dict().items() if k != "key"},
        }) + "\n")
