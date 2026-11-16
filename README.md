# Job Matcher: MLflow Lab

A personal job search tool built for the MLflow experiment tracking lab

It fetches **live job postings** from company job boards, ranks them against **your resume**, shows the skills you
are missing, and learns from your 👍 / 👎 ratings. MLflow tracks every matching method, registers the best one as a
model, and only promotes it to **champion** if it beats the current one.

## What it does

1. **Upload your resume** (PDF, Word or text). It stays on your computer in `data/`, which is never committed.
2. **Fetch live jobs** from 22 companies by default (Stripe, Databricks, Airbnb, OpenAI, Ramp and more), and add any
   company that uses Greenhouse, Lever or Ashby.
3. **Filter** to ML / AI Engineer, Data Scientist and Data Engineer roles, internship / co-op and new grad level,
   US or US-remote, posted in the last 30 days. Senior, staff, lead and manager roles are always hidden.
4. **Rank** every job by fit, with the skills you match and the skills you're missing, plus a link to apply.
5. **Skill gaps:** the skills your top matches ask for most that your resume doesn't mention.
6. **Rate jobs** 👍 would apply / 👎 not for me. Your ratings become the test set for the MLflow experiments.

## The MLflow part

| Step | What happens in MLflow |
|---|---|
| Experiment | One parent run per experiment, with a **nested run for each matching method** |
| Parameters | Method name, weight of each scorer, TF-IDF n-grams, title weight |
| Metrics | **NDCG@10**, Precision@5, Precision@10 and average precision on your rated jobs |
| Artifacts | Each method's config and its full ranking of your rated jobs (`ranking.json`) |
| Model | The best method is logged as a custom **pyfunc model** (`JobRankerModel`) |
| Registry | Registered as a new version of **`job-matcher`** |
| Quality gate | It gets the **`champion`** alias only if its NDCG@10 beats the current champion's by at least 0.01, with both re-scored on the same ratings |
| Serving | The app loads `models:/job-matcher@champion` to rank jobs |

### Matching methods compared

| Method | How it scores a job |
|---|---|
| `tfidf_words`, `tfidf_phrases` | TF-IDF text similarity between resume and job (single words, or words plus word pairs) |
| `skills_only` | Share of the job's required skills that your resume has |
| `hybrid_*` | Weighted mix of TF-IDF and skills (text heavy, balanced, skills heavy, without title boost) |
| `embedding_only`, `hybrid_embedding` | Meaning-based similarity with `all-MiniLM-L6-v2` (optional, see below) |

### Why NDCG@10
You only read the top of the list, so the metric that matters is whether the jobs you'd apply to are near the top.
NDCG@10 is 1.0 when every 👍 job is ranked above every 👎 job in the top 10, and lower when good jobs are pushed down.

## Changes from the original lab

| Original MLflow Lab 1 | This project |
|---|---|
| Linear regression on a fixed dataset | Ranking live job postings against a resume |
| Logs parameters and metrics of one model | Compares 7 to 9 methods per experiment as nested runs, with ranking metrics |
| Model logged with a built-in flavour | Custom pyfunc model wrapping the whole ranking method |
| Serving through `mlflow models serve` | Model Registry with versions, a `champion` alias and an automatic promotion gate |
| Evaluation on a held-out split | Evaluation on your own ratings, which grow as you use the app |
| No interface | Streamlit app with Matches, Skill gaps and Experiments tabs |

## Run it

```bash
git clone https://github.com/arckived/mlops_lab4.git
cd mlops_lab4
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `venv\Scripts\activate`.

Start the app:

```bash
streamlit run app.py
```

Open http://localhost:8501, upload your resume, press **Fetch latest jobs**, and rate at least 10 jobs (some 👍 and
some 👎). Then open the **Experiments (MLflow)** tab and press **Run experiments**.

See every run, metric, artifact and registered model in the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000.

### Optional: embedding methods

```bash
pip install -r requirements-embeddings.txt
```

This installs `sentence-transformers` (and PyTorch, a large download). The experiments then also compare the
embedding methods. The model downloads from huggingface.co the first time it's used.

## Where the jobs come from

Only the official public job board APIs that companies use for their own careers pages. No key is needed and no site
is scraped.

| Source | API |
|---|---|
| Greenhouse | `https://boards-api.greenhouse.io/v1/boards/<token>/jobs?content=true` |
| Lever | `https://api.lever.co/v0/postings/<token>?mode=json` |
| Ashby | `https://api.ashbyhq.com/posting-api/job-board/<token>` |

LinkedIn, Indeed and Handshake are not included: they have no public API and their terms do not allow scraping.

To add a company, find its careers page. If the URL looks like `boards.greenhouse.io/acme`, `jobs.lever.co/acme` or
`jobs.ashbyhq.com/acme`, type `greenhouse:acme`, `lever:acme` or `ashby:acme` in the sidebar. Companies sometimes move
to another job board; any company that fails to load is listed in the app and the rest still work.

## Privacy

Your resume text, the fetched jobs and your ratings are stored in `data/`. The MLflow database (`mlflow.db`) and
artifacts (`mlartifacts/`) contain your ratings and job titles. All of these are in `.gitignore`, so they stay on
your computer.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

28 tests cover the three job board parsers, skill extraction, the role / level / US location filters, ranking,
the metrics, and a full MLflow cycle: run an experiment, register the best method, promote it, load it back as a
pyfunc model and score with it, then check that a challenger that is not better is **not** promoted.

## Project structure

```
mlops_lab4/
├── app.py                      Streamlit app
├── jobmatch/
│   ├── sources.py              Greenhouse, Lever and Ashby clients
│   ├── companies.py            Default company list
│   ├── filters.py              Role, level, US location and date filters
│   ├── skills.py               Skill vocabulary and extraction
│   ├── matching.py             Scorers and the RankerConfig
│   ├── metrics.py              NDCG, precision and average precision
│   ├── tracking.py             MLflow experiments, pyfunc model, registry and promotion gate
│   └── storage.py              Resume, job cache and ratings in data/
├── tests/                      Unit and MLflow tests
├── requirements.txt
├── requirements-dev.txt
└── requirements-embeddings.txt
```
