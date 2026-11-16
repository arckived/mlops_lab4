"""Default companies to search.

Each entry is (source, board token, display name). The board token is the
part of the company's public job board URL, for example
https://boards.greenhouse.io/<token> or https://jobs.lever.co/<token> or
https://jobs.ashbyhq.com/<token>.

Companies sometimes move to a different job board. If a company fails to
load, the app shows it and the rest still work. Add your own companies in
the app sidebar with the format  source:token  (for example greenhouse:stripe).
"""

DEFAULT_COMPANIES = [
    # Greenhouse
    ("greenhouse", "stripe", "Stripe"),
    ("greenhouse", "airbnb", "Airbnb"),
    ("greenhouse", "databricks", "Databricks"),
    ("greenhouse", "robinhood", "Robinhood"),
    ("greenhouse", "discord", "Discord"),
    ("greenhouse", "figma", "Figma"),
    ("greenhouse", "dropbox", "Dropbox"),
    ("greenhouse", "coinbase", "Coinbase"),
    ("greenhouse", "datadog", "Datadog"),
    ("greenhouse", "instacart", "Instacart"),
    ("greenhouse", "scaleai", "Scale AI"),
    ("greenhouse", "anthropic", "Anthropic"),
    ("greenhouse", "pinterest", "Pinterest"),
    ("greenhouse", "reddit", "Reddit"),
    ("greenhouse", "lyft", "Lyft"),
    # Lever
    ("lever", "palantir", "Palantir"),
    ("lever", "spotify", "Spotify"),
    ("lever", "plaid", "Plaid"),
    # Ashby
    ("ashby", "ramp", "Ramp"),
    ("ashby", "notion", "Notion"),
    ("ashby", "openai", "OpenAI"),
    ("ashby", "linear", "Linear"),
]

SOURCES = ("greenhouse", "lever", "ashby")


def parse_company_line(line: str):
    """Parse 'source:token' or 'source:token:Display Name' typed by the user."""
    parts = [p.strip() for p in line.split(":")]
    if len(parts) < 2 or parts[0].lower() not in SOURCES or not parts[1]:
        raise ValueError(f"Use the format source:token, with source one of {', '.join(SOURCES)}: {line!r}")
    name = parts[2] if len(parts) > 2 and parts[2] else parts[1].replace("-", " ").title()
    return parts[0].lower(), parts[1], name
