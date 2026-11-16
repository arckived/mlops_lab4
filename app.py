"""Job Matcher: rank live job postings against your resume.

Run with:  streamlit run app.py
"""

from collections import Counter

import mlflow
import pandas as pd
import streamlit as st

from jobmatch import storage, tracking
from jobmatch.companies import DEFAULT_COMPANIES, parse_company_line
from jobmatch.filters import ROLE_FAMILIES, filter_jobs
from jobmatch.matching import RankerConfig, rank_jobs
from jobmatch.skills import extract_skills
from jobmatch.sources import fetch_all

st.set_page_config(page_title="Job Matcher", page_icon="🎯", layout="wide")


# ---------- Ranking model: the MLflow champion, or a sensible default ----------

@st.cache_resource(show_spinner=False)
def load_ranker(champion_version):
    """Load models:/job-matcher@champion from MLflow. Cached per champion version,
    so a newly promoted champion is picked up on the next page load."""
    if champion_version is None:
        return RankerConfig(), "default method (no champion yet)"
    model, version = tracking.load_champion_model()
    config = model.unwrap_python_model().config
    return config, f"MLflow champion v{version} ({config.name})"


tracking.setup_mlflow()
config, ranker_label = load_ranker(tracking.get_champion()[1])


# ---------- Sidebar: resume and search settings ----------

with st.sidebar:
    st.header("1. Your resume")
    upload = st.file_uploader("PDF, Word or text", type=["pdf", "docx", "txt"])
    if upload is not None and st.session_state.get("resume_name") != upload.name:
        text = storage.extract_resume_text(upload.name, upload.getvalue())
        if text:
            storage.save_resume(text)
            st.session_state.resume_name = upload.name
        else:
            st.error("Could not read any text from that file. Try a .docx or .txt version.")
    resume = storage.load_resume()
    if resume:
        resume_skills = sorted(extract_skills(resume))
        st.caption(f"Resume loaded. {len(resume_skills)} skills found:")
        st.write(", ".join(resume_skills) or "none")
        st.caption("Stored only in data/resume.txt on this computer.")

    st.header("2. What you're looking for")
    families = st.multiselect("Roles", list(ROLE_FAMILIES),
                              default=["ML / AI Engineer", "Data Scientist", "Data Engineer"])
    level_choice = st.radio("Level", ["Internship / co-op", "New grad", "Both"], index=2)
    levels = {"Internship / co-op": {"intern"}, "New grad": {"new grad"}, "Both": {"intern", "new grad"}}[level_choice]
    include_unspecified = st.checkbox(
        "Also show roles with no level in the title", value=True,
        help="Many entry level jobs are just called 'Data Engineer'. Senior, staff, lead and manager roles are always hidden.")
    us_only = st.checkbox("US and US-remote only", value=True)
    max_age = st.slider("Posted in the last (days)", 0, 90, 30, help="0 means any date.")

    st.header("3. Companies")
    default_names = [name for _, _, name in DEFAULT_COMPANIES]
    chosen = st.multiselect("Companies to search", default_names, default=default_names)
    custom = st.text_area("Add more (one per line)", placeholder="greenhouse:stripe\nlever:palantir\nashby:ramp",
                          help="Find the token in the company's careers page URL, e.g. boards.greenhouse.io/<token>.")

    companies = [c for c in DEFAULT_COMPANIES if c[2] in chosen]
    for line in custom.splitlines():
        if line.strip():
            try:
                companies.append(parse_company_line(line))
            except ValueError as e:
                st.warning(str(e))

    if st.button("Fetch latest jobs", type="primary", use_container_width=True):
        with st.spinner(f"Fetching from {len(companies)} companies..."):
            jobs, errors = fetch_all(companies)
        storage.save_jobs(jobs, errors)
        st.success(f"Fetched {len(jobs)} postings.")


# ---------- Main page ----------

st.title("🎯 Job Matcher")
st.caption(f"Ranking with: **{ranker_label}**")

all_jobs, fetch_errors, fetched_at = storage.load_jobs()
labels = storage.load_labels()

if not resume:
    st.info("Upload your resume in the sidebar to get started.")
    st.stop()
if not all_jobs:
    st.info("Press **Fetch latest jobs** in the sidebar.")
    st.stop()

st.caption(f"{len(all_jobs)} postings fetched at {fetched_at} UTC.")
if fetch_errors:
    with st.expander(f"Companies that could not be loaded: {len(fetch_errors)}"):
        for name, err in fetch_errors.items():
            st.write(f"**{name}**: {err}")

kept = filter_jobs(all_jobs, families, levels, include_unspecified, us_only, max_age)
if not kept:
    st.warning("No postings match these filters. Try more companies, more roles or a longer date range.")
    st.stop()

meta = {job.key: (fam, level) for job, fam, level in kept}
ranked = rank_jobs(resume, [job for job, _, _ in kept], config)
ranked["role"] = ranked["key"].map(lambda k: meta[k][0])
ranked["level"] = ranked["key"].map(lambda k: meta[k][1])
jobs_by_key = {job.key: job for job, _, _ in kept}

matches_tab, gaps_tab, exp_tab = st.tabs(["Matches", "Skill gaps", "Experiments (MLflow)"])


# ---------- Tab 1: Matches ----------

with matches_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Matching postings", len(ranked))
    c2.metric("Companies", ranked["company"].nunique())
    c3.metric("Jobs you've rated", len(labels))

    show = st.slider("Show top", 5, min(100, max(len(ranked), 5)), min(25, len(ranked)))
    st.download_button("Download all matches as CSV",
                       ranked.drop(columns=["description", "extra"]).to_csv(index=False),
                       "job_matches.csv", "text/csv")
    st.caption("Rate jobs with 👍 / 👎. Your ratings are what the MLflow experiments learn from.")

    for i, row in ranked.head(show).iterrows():
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{i + 1}. [{row.title}]({row.url})**  \n{row.company} · {row.location} · "
                            f"{row.role} · {row.level}" + (f" · posted {row.posted_at[:10]}" if row.posted_at else ""))
                if row.matched_skills:
                    st.markdown("✅ " + ", ".join(row.matched_skills))
                if row.missing_skills:
                    st.markdown("➕ To learn or add: " + ", ".join(row.missing_skills))
            with right:
                st.metric("Score", f"{row.score:.2f}", help="Higher is a better fit. Compare scores between jobs; the number itself is not a percentage.")
                current = labels.get(row.key, {}).get("label")
                b1, b2 = st.columns(2)
                if b1.button("👍", key=f"up{row.key}", type="primary" if current == 1 else "secondary"):
                    storage.save_label(jobs_by_key[row.key], 1)
                    st.rerun()
                if b2.button("👎", key=f"down{row.key}", type="primary" if current == 0 else "secondary"):
                    storage.save_label(jobs_by_key[row.key], 0)
                    st.rerun()


# ---------- Tab 2: Skill gaps ----------

with gaps_tab:
    top_n = st.slider("Look at your top matches", 10, max(10, min(100, len(ranked))), min(30, max(10, len(ranked))))
    top = ranked.head(top_n)
    missing = Counter(s for skills in top["missing_skills"] for s in skills)
    st.subheader(f"Skills your top {len(top)} matches ask for that your resume doesn't mention")
    st.caption("If you already have one of these, add it to your resume. If not, it's a good thing to learn next.")
    if missing:
        gaps = pd.DataFrame(missing.most_common(20), columns=["Skill", "Jobs asking for it"])
        st.dataframe(gaps, hide_index=True, use_container_width=True, column_config={
            "Jobs asking for it": st.column_config.ProgressColumn(
                format="%d", min_value=0, max_value=int(gaps["Jobs asking for it"].max())),
        })
    else:
        st.success("Your resume covers every skill these jobs mention.")


# ---------- Tab 3: Experiments ----------

with exp_tab:
    n_pos = sum(1 for v in labels.values() if v["label"] == 1)
    n_neg = len(labels) - n_pos
    c1, c2, c3 = st.columns(3)
    c1.metric("Would apply 👍", n_pos)
    c2.metric("Not for me 👎", n_neg)
    c3.metric("Needed to run", f"{tracking.MIN_LABELS} total, both kinds")

    st.write(
        "Each experiment scores your rated jobs with every matching method, logs the results to MLflow, "
        "registers the best method as a new version of the **job-matcher** model, and makes it the "
        f"**champion** only if it beats the current champion's NDCG@10 by at least {tracking.MIN_IMPROVEMENT}."
    )
    if st.button("Run experiments", type="primary"):
        try:
            with st.spinner("Running MLflow experiments..."):
                summary = tracking.run_experiments(resume, labels)
            (st.success if summary["promoted"] else st.info)(
                f"Best method: **{summary['best_method']}**, registered as version "
                f"{summary['registered_version']}. {summary['reason']}")
            if summary["promoted"]:
                st.caption("Reload the page to rank with the new champion.")
        except ValueError as e:
            st.warning(str(e))

    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT_NAME],
                              filter_string="tags.mlflow.parentRunId != ''",
                              order_by=["start_time DESC"], max_results=200)
    if runs.empty:
        st.caption("No experiments yet.")
    else:
        latest_parent = runs["tags.mlflow.parentRunId"].iloc[0]
        latest = runs[runs["tags.mlflow.parentRunId"] == latest_parent]
        table = pd.DataFrame({
            "Method": latest["params.method"],
            "NDCG@10": latest["metrics.ndcg_at_10"],
            "Precision@5": latest["metrics.precision_at_5"],
            "Precision@10": latest["metrics.precision_at_10"],
            "Avg precision": latest["metrics.average_precision"],
        }).sort_values("NDCG@10", ascending=False)
        st.subheader("Latest experiment")
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[1:]})
    st.caption("For the full history, charts and registered models, run `mlflow ui --backend-store-uri sqlite:///mlflow.db` "
               "and open http://localhost:5000.")
