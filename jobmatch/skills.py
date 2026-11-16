"""Find technical skills in resume and job text.

A fixed vocabulary is used instead of a model so the result is easy to check:
every matched skill can be traced to the words that triggered it.
"""

import re

# Canonical skill name -> extra spellings. The canonical name is also searched.
SKILLS: dict[str, list[str]] = {
    # Languages
    "Python": [], "SQL": [], "Java": [], "Scala": [], "C++": ["cpp"], "Go": ["golang"],
    "JavaScript": ["js"], "TypeScript": [], "Rust": [], "Bash": ["shell scripting"],
    "R": [], "MATLAB": [],
    # ML and stats
    "Machine Learning": ["ml"], "Deep Learning": [], "NLP": ["natural language processing"],
    "Computer Vision": [], "LLMs": ["llm", "large language models", "large language model"],
    "Generative AI": ["genai", "gen ai"], "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
    "Recommender Systems": ["recommendation systems", "recommender system", "recommendation system"],
    "Time Series": ["forecasting"], "Statistics": ["statistical modeling", "statistical analysis"],
    "A/B Testing": ["ab testing", "a/b tests", "experimentation"], "Causal Inference": [],
    "Reinforcement Learning": [], "Feature Engineering": [],
    # ML libraries
    "PyTorch": [], "TensorFlow": [], "Keras": [], "scikit-learn": ["sklearn", "scikit learn"],
    "XGBoost": [], "LightGBM": [], "Hugging Face": ["huggingface", "transformers"],
    "LangChain": [], "pandas": [], "NumPy": [], "spaCy": [], "OpenCV": [],
    # MLOps
    "MLOps": [], "MLflow": [], "Kubeflow": [], "Airflow": ["apache airflow"], "DVC": [],
    "Weights & Biases": ["wandb", "weights and biases"], "Model Deployment": ["model serving"],
    "Docker": [], "Kubernetes": ["k8s"], "CI/CD": ["ci / cd", "continuous integration"],
    "Terraform": [], "FastAPI": [], "Flask": [], "REST APIs": ["rest api", "restful"],
    "Git": ["github"], "Linux": [], "Prometheus": [], "Grafana": [],
    # Data engineering
    "Spark": ["pyspark", "apache spark"], "Kafka": ["apache kafka"], "dbt": [],
    "ETL": ["elt", "data pipelines", "data pipeline"], "Hadoop": [], "Flink": ["apache flink"],
    "Snowflake": [], "BigQuery": [], "Redshift": [], "Databricks": [], "Data Warehousing": ["data warehouse"],
    "Data Modeling": [], "PostgreSQL": ["postgres"], "MySQL": [], "MongoDB": [], "NoSQL": [],
    "Redis": [], "Elasticsearch": [],
    # Cloud
    "AWS": ["amazon web services", "sagemaker", "s3", "ec2", "lambda"],
    "GCP": ["google cloud", "vertex ai"], "Azure": [],
    # Analytics
    "Tableau": [], "Power BI": ["powerbi"], "Looker": [], "Excel": [],
    "Data Visualization": ["matplotlib", "seaborn", "plotly"], "Streamlit": [],
}

# Names that are also common English words or single letters. These must
# appear exactly in this capitalisation to count.
CASE_SENSITIVE = {"R", "Go", "Excel", "Spark", "Rust", "Looker", "Flink", "dbt", "Redis"}


def _pattern(term: str, case_sensitive: bool) -> re.Pattern:
    # Word boundaries that also work for terms like C++, CI/CD and A/B Testing.
    regex = r"(?<![\w+#/])" + re.escape(term) + r"(?![\w+#])"
    return re.compile(regex, 0 if case_sensitive else re.IGNORECASE)


# "Go" needs extra care: skip it at the start of a sentence ("Go to ...") and before "to".
SPECIAL_PATTERNS = {
    "Go": re.compile(r"(?<![\w+#/])(?<!^)(?<![.!?]\s)(?<![.!?]\n)Go(?![\w+#])(?!\s+to\b)", re.MULTILINE),
}

_COMPILED = []
for _name, _aliases in SKILLS.items():
    _terms = [_name] + _aliases
    for _t in _terms:
        if _t in SPECIAL_PATTERNS:
            _COMPILED.append((_name, SPECIAL_PATTERNS[_t]))
            continue
        _cs = _t in CASE_SENSITIVE
        _COMPILED.append((_name, _pattern(_t, _cs)))


def extract_skills(text: str) -> set[str]:
    """Return the canonical names of all skills mentioned in the text."""
    if not text:
        return set()
    found = set()
    for name, pattern in _COMPILED:
        if name not in found and pattern.search(text):
            found.add(name)
    return found
