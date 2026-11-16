"""Decide which postings are relevant: role family, level, US location, recency."""

import re
from datetime import datetime, timedelta, timezone

ROLE_FAMILIES = {
    "ML / AI Engineer": [
        "machine learning", "ml engineer", "ml engineering", "ai engineer", "applied scientist",
        "applied ml", "research engineer", "deep learning", "nlp", "computer vision", "llm",
        "mlops", "ml platform", "ml infrastructure", "ai/ml", "ml/ai", "artificial intelligence",
    ],
    "Data Scientist": [
        "data scientist", "data science", "decision scientist", "product analyst",
        "data analyst", "analytics", "quantitative analyst", "research scientist",
    ],
    "Data Engineer": [
        "data engineer", "data engineering", "analytics engineer", "data platform",
        "data infrastructure", "etl", "data pipeline",
    ],
}

INTERN_RE = re.compile(r"\b(intern|interns|internship|co-?op|summer 20\d\d|fall 20\d\d|spring 20\d\d)\b", re.I)
NEW_GRAD_RE = re.compile(
    r"\b(new grad|new graduate|graduate|university grad|early career|entry[- ]level|junior|"
    r"associate|engineer i|scientist i|analyst i|level 1|l1|20\d\d start)\b", re.I)
SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|leader|manager|director|head|vp|vice president|"
    r"chief|distinguished|architect|engineer ii+|engineer iii|scientist ii+|iii|iv)\b", re.I)

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}
US_WORDS = re.compile(
    r"\b(united states|usa|u\.s\.|us-remote|remote[- ]us|us remote|new york|san francisco|seattle|"
    r"boston|austin|chicago|los angeles|denver|atlanta|washington|bay area|silicon valley|"
    r"palo alto|mountain view|menlo park|sunnyvale|san jose|san mateo|redwood city|oakland|"
    r"cambridge, ma|miami|dallas|houston|philadelphia|pittsburgh|portland|salt lake city|"
    r"raleigh|nyc|sf)\b", re.I)
NON_US_WORDS = re.compile(
    r"\b(canada|toronto|vancouver|montreal|uk|united kingdom|london|ireland|dublin|germany|berlin|"
    r"munich|france|paris|netherlands|amsterdam|spain|madrid|india|bangalore|bengaluru|hyderabad|"
    r"pune|singapore|japan|tokyo|australia|sydney|melbourne|brazil|sao paulo|mexico|poland|warsaw|"
    r"israel|tel aviv|china|hong kong|korea|seoul|emea|apac|latam|europe|switzerland|zurich|"
    r"sweden|stockholm|denmark|copenhagen|portugal|lisbon|italy|argentina|colombia|chile)\b", re.I)
STATE_CODE_RE = re.compile(r",\s*([A-Z]{2})\b")


def role_family(title: str, families) -> str | None:
    """Return the first selected role family whose keywords appear in the title."""
    t = title.lower()
    for fam in families:
        for kw in ROLE_FAMILIES.get(fam, []):
            if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", t):
                return fam
    return None


def job_level(title: str) -> str:
    """'intern', 'new grad', 'senior' or 'unspecified', based on the title only."""
    if INTERN_RE.search(title):
        return "intern"
    if SENIOR_RE.search(title):
        return "senior"
    if NEW_GRAD_RE.search(title):
        return "new grad"
    return "unspecified"


def is_us(location: str) -> bool:
    """True if any of the locations in the text is in the US (remote counts unless another country is named)."""
    if not location:
        return False
    for part in re.split(r"\s*[/;|]\s*|\s+or\s+", location):
        if not part:
            continue
        if US_WORDS.search(part):
            return True
        codes = STATE_CODE_RE.findall(part)
        if any(c in US_STATES for c in codes):
            return True
        if "remote" in part.lower() and not NON_US_WORDS.search(part):
            return True
    return False


def posted_within(posted_at: str, days: int, now: datetime | None = None) -> bool:
    """True if posted within the last `days`. Unknown dates are kept."""
    if not posted_at or not days:
        return True
    try:
        when = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return when >= now - timedelta(days=days)


def filter_jobs(jobs, families, levels, include_unspecified=True, us_only=True, max_age_days=30):
    """Keep jobs that match the chosen role families, levels, location and age.

    levels: a set containing "intern" and/or "new grad".
    include_unspecified: also keep titles with no level words (and not senior).
    Returns a list of (job, family, level).
    """
    kept = []
    for job in jobs:
        fam = role_family(job.title, families)
        if not fam:
            continue
        level = job_level(job.title)
        if level == "senior":
            continue
        if level == "unspecified" and not include_unspecified:
            continue
        if level in ("intern", "new grad") and level not in levels:
            continue
        if us_only and not is_us(job.location):
            continue
        if not posted_within(job.posted_at, max_age_days):
            continue
        kept.append((job, fam, level))
    return kept
