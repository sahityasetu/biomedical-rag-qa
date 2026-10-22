"""Answer-quality metrics that need no paid LLM judge.

- yes/no accuracy        exact, for questions whose expert answer starts with Yes/No
- semantic similarity    cosine similarity between the generated and the expert answer (bge-small)
- faithfulness           share of answer sentences that an NLI model says are ENTAILED by at least one source
- citation checks        does the answer cite sources, and are the cited numbers real
- abstention             did the model say the sources do not contain the answer
"""
import re

import numpy as np

from src.generator import NOT_FOUND

NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def yes_no(text: str):
    m = re.match(r"^\W*(yes|no)\b", text, flags=re.I)
    return m.group(1).lower() if m else None


def abstained(text: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in text.lower()


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text).strip()


def sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", strip_citations(text))
    return [s for s in parts if len(s.split()) >= 3]


def cited_numbers(text: str) -> set:
    return {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.split(r"\s*,\s*", group)}


def citation_stats(text: str, n_sources: int) -> dict:
    cited = cited_numbers(text)
    return {"has_citation": bool(cited), "invalid_citation": any(c < 1 or c > n_sources for c in cited)}


class NLIJudge:
    """Checks each answer sentence against each source passage with an NLI cross-encoder."""

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(NLI_MODEL)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def faithfulness(self, answer: str, passages: list) -> float | None:
        sents = sentences(answer)
        if not sents or not passages:
            return None
        pairs = [(p["text"], s) for s in sents for p in passages]
        logits = self.model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        entailed = np.asarray(logits)[:, self.entail].reshape(len(sents), len(passages)).max(1) > 0.5
        return float(entailed.mean())


class Similarity:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, generated: list, reference: list) -> np.ndarray:
        a = self.encoder.encode([strip_citations(g) for g in generated], normalize_embeddings=True)
        b = self.encoder.encode(reference, normalize_embeddings=True)
        return (a * b).sum(1)
