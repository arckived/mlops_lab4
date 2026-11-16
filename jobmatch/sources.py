"""Fetch live job postings from public job board APIs.

All three sources are official, public, read-only APIs that companies use to
show their openings on their own careers pages. No API key is needed.

    Greenhouse  https://boards-api.greenhouse.io/v1/boards/<token>/jobs?content=true
    Lever       https://api.lever.co/v0/postings/<token>?mode=json
    Ashby       https://api.ashbyhq.com/posting-api/job-board/<token>

Every posting is converted to the same Job shape so the rest of the code does
not care where it came from.
"""

import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import requests

TIMEOUT = 20
HEADERS = {"User-Agent": "mlops-lab4-job-matcher (personal job search tool)"}


@dataclass
class Job:
    source: str
    company: str
    job_id: str
    title: str
    location: str
    url: str
    description: str
    posted_at: str = ""          # ISO 8601 date, empty if unknown
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.company}:{self.job_id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**{k: v for k, v in d.items() if k != "key"})


def html_to_text(raw: str) -> str:
    """Turn job description HTML (sometimes HTML-escaped twice) into plain text."""
    if not raw:
        return ""
    text = html.unescape(html.unescape(raw))
    text = re.sub(r"<(br|/p|/li|/h\d|/div)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _iso_from_millis(ms) -> str:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


# ---------- Parsers: raw API JSON -> list[Job] ----------

def parse_greenhouse(data: dict, company: str) -> list[Job]:
    jobs = []
    for j in data.get("jobs", []):
        jobs.append(Job(
            source="greenhouse",
            company=company,
            job_id=str(j.get("id", "")),
            title=(j.get("title") or "").strip(),
            location=((j.get("location") or {}).get("name") or "").strip(),
            url=j.get("absolute_url", ""),
            description=html_to_text(j.get("content", "")),
            posted_at=j.get("first_published") or j.get("updated_at") or "",
        ))
    return jobs


def parse_lever(data: list, company: str) -> list[Job]:
    jobs = []
    for j in data or []:
        cats = j.get("categories") or {}
        parts = [j.get("descriptionPlain") or html_to_text(j.get("description", ""))]
        for section in j.get("lists") or []:
            parts.append(section.get("text", ""))
            parts.append(html_to_text(section.get("content", "")))
        parts.append(j.get("additionalPlain") or "")
        locations = cats.get("allLocations") or [cats.get("location", "")]
        jobs.append(Job(
            source="lever",
            company=company,
            job_id=str(j.get("id", "")),
            title=(j.get("text") or "").strip(),
            location=" / ".join(l for l in locations if l),
            url=j.get("hostedUrl", ""),
            description="\n".join(p for p in parts if p).strip(),
            posted_at=_iso_from_millis(j.get("createdAt")),
            extra={"commitment": cats.get("commitment", ""), "team": cats.get("team", "")},
        ))
    return jobs


def parse_ashby(data: dict, company: str) -> list[Job]:
    jobs = []
    for j in data.get("jobs", []):
        if j.get("isListed") is False:
            continue
        locations = [j.get("location", "")] + [
            (s or {}).get("location", "") for s in j.get("secondaryLocations") or []
        ]
        location = " / ".join(l for l in locations if l)
        if j.get("isRemote") and "remote" not in location.lower():
            location = f"{location} / Remote" if location else "Remote"
        jobs.append(Job(
            source="ashby",
            company=company,
            job_id=str(j.get("id", "")),
            title=(j.get("title") or "").strip(),
            location=location,
            url=j.get("jobUrl", ""),
            description=j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml", "")),
            posted_at=j.get("publishedAt", ""),
            extra={"employment_type": j.get("employmentType", "")},
        ))
    return jobs


URLS = {
    "greenhouse": "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true",
    "lever": "https://api.lever.co/v0/postings/{token}?mode=json",
    "ashby": "https://api.ashbyhq.com/posting-api/job-board/{token}",
}
PARSERS = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby}


def fetch_company(source: str, token: str, name: str, session=None) -> list[Job]:
    """Fetch every open posting for one company. Raises on network or HTTP errors."""
    http = session or requests
    resp = http.get(URLS[source].format(token=token), headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return PARSERS[source](resp.json(), name)


def fetch_all(companies, max_workers: int = 8):
    """Fetch many companies in parallel.

    Returns (jobs, errors) where errors maps "Company (source)" to a short message,
    so one broken board does not stop the others.
    """
    jobs, errors = [], {}

    def one(entry):
        source, token, name = entry
        try:
            return entry, fetch_company(source, token, name), None
        except Exception as e:  # network error, 404, bad JSON
            return entry, [], f"{type(e).__name__}: {e}"[:200]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (source, token, name), found, err in pool.map(one, companies):
            if err:
                errors[f"{name} ({source})"] = err
            jobs.extend(found)
    return jobs, errors
