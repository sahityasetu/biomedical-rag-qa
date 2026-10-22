"""Load and clean the BioASQ mini RAG dataset: PubMed abstracts + expert-written biomedical Q&A.

Source: huggingface.co/datasets/rag-datasets/rag-mini-bioasq (derived from the BioASQ challenge).
"""
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPO = "rag-datasets/rag-mini-bioasq"
SEED = 42


def _download(name: str) -> pd.DataFrame:
    return pd.read_parquet(hf_hub_download(REPO, f"data/{name}.parquet/part.0.parquet", repo_type="dataset",
                                           local_dir=DATA))


def load_passages() -> pd.DataFrame:
    """PubMed abstracts indexed by PubMed id. About 30% of rows are the string 'nan' and are dropped."""
    p = _download("passages")
    p["passage"] = p["passage"].str.replace(r"\s+", " ", regex=True).str.strip()
    return p[p["passage"].str.split().str.len() >= 20]


def load_questions(passage_ids: set) -> pd.DataFrame:
    q = _download("test")
    q["relevant_ids"] = q["relevant_passage_ids"].apply(lambda s: [i for i in json.loads(s) if i in passage_ids])
    q["type"] = q["answer"].apply(question_type)
    # keep questions that still have at least one usable evidence passage after cleaning
    return q[q["relevant_ids"].str.len() > 0].drop(columns="relevant_passage_ids")


def question_type(answer: str) -> str:
    """BioASQ yes/no questions have answers that start with 'Yes' or 'No'; these get an exact accuracy metric."""
    m = re.match(r"^\s*(yes|no)\b", answer, flags=re.I)
    return m.group(1).lower() if m else "open"


def sample_questions(q: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Fixed evaluation sample: n/4 'yes', n/4 'no' and n/2 open questions.

    'Yes' answers outnumber 'no' 3:1 in the data, so without balancing a model that always says
    'yes' would look 75% accurate. With a 50/50 split, always-yes scores exactly 50%.
    """
    parts = [q[q["type"] == "yes"].sample(n // 4, random_state=SEED),
             q[q["type"] == "no"].sample(n // 4, random_state=SEED),
             q[q["type"] == "open"].sample(n - 2 * (n // 4), random_state=SEED)]
    return pd.concat(parts).sample(frac=1, random_state=SEED)
