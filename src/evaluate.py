"""Does retrieval make a small LLM's biomedical answers more correct and more faithful?

Four settings on the same 200 questions (50 yes + 50 no + 100 open):
  closed_book    LLM answers from memory, no sources
  rag            top-3 abstracts from the dense retriever
  oracle         the expert-labelled relevant abstracts (upper bound: perfect retrieval)
  wrong_context  top-3 abstracts retrieved for a DIFFERENT question (does the model admit it cannot answer?)

Run from the project root:  python -m src.evaluate      (~1 h on an M2 laptop, resumes if interrupted)
Outputs: reports/generations.jsonl, reports/metrics.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_passages, load_questions, sample_questions
from src.generator import Generator
from src.metrics import NLIJudge, Similarity, abstained, citation_stats, yes_no
from src.retriever import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
GEN_FILE = OUT / "generations.jsonl"
SETTINGS = ["closed_book", "rag", "oracle", "wrong_context"]
K = 3


def retrieval_metrics(retriever, questions: pd.DataFrame) -> dict:
    hits = {1: [], 3: [], 10: []}
    for q, rel in zip(questions["question"], questions["relevant_ids"]):
        found = [r["id"] for r in retriever.search(q, k=10)]
        for k in hits:
            hits[k].append(any(i in rel for i in found[:k]))
    return {f"hit_rate@{k}": round(float(np.mean(v)), 4) for k, v in hits.items()}


def contexts(retriever, sample: pd.DataFrame) -> dict:
    """Passages per setting and question. wrong_context reuses the retrieval of the next question in the sample."""
    retrieved = [retriever.search(q, k=K) for q in sample["question"]]
    return {
        "closed_book": [None] * len(sample),
        "rag": retrieved,
        "oracle": [retriever.get(ids[:K]) for ids in sample["relevant_ids"]],
        "wrong_context": retrieved[1:] + retrieved[:1],
    }


def generate_all(sample, ctx):
    done = set()
    if GEN_FILE.exists():
        done = {(r["qid"], r["setting"]) for r in map(json.loads, GEN_FILE.read_text().splitlines())}
    todo = [(s, i) for s in SETTINGS for i in range(len(sample)) if (int(sample.index[i]), s) not in done]
    if not todo:
        return
    gen = Generator()
    with GEN_FILE.open("a") as f:
        for n, (setting, i) in enumerate(todo, 1):
            row = sample.iloc[i]
            passages = ctx[setting][i]
            answer = gen.answer(row["question"], passages)
            f.write(json.dumps({"qid": int(sample.index[i]), "setting": setting, "question": row["question"],
                                "type": row["type"], "gold": row["answer"], "answer": answer,
                                "source_ids": [p["id"] for p in passages] if passages else []}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)} generated")


def score(sample, ctx, retriever) -> tuple:
    gens = pd.DataFrame(map(json.loads, GEN_FILE.read_text().splitlines()))
    gens = gens[gens["qid"].isin(sample.index)]
    pos = {qid: i for i, qid in enumerate(sample.index)}
    judge, sim = NLIJudge(), Similarity(retriever.model)
    rows = []
    for _, g in gens.iterrows():
        passages = ctx[g["setting"]][pos[g["qid"]]]
        # closed-book answers are checked against the RAG sources: is what the model "remembers" supported?
        check_against = passages if passages is not None else ctx["rag"][pos[g["qid"]]]
        rows.append({
            **g.to_dict(),
            "abstained": abstained(g["answer"]),
            "yes_no_correct": (yes_no(g["answer"]) == g["type"]) if g["type"] != "open" else None,
            "faithfulness": None if abstained(g["answer"]) else judge.faithfulness(g["answer"], check_against),
            **(citation_stats(g["answer"], K) if passages is not None else {}),
        })
    scored = pd.DataFrame(rows)
    scored["similarity"] = sim(scored["answer"].tolist(), scored["gold"].tolist())
    return scored


def summarise(scored: pd.DataFrame) -> dict:
    out = {}
    for setting, g in scored.groupby("setting"):
        answered = g[~g["abstained"]]
        out[setting] = {
            "n": len(g),
            "yes_no_accuracy": round(float(g["yes_no_correct"].dropna().astype(float).mean()), 4),
            "similarity_to_expert": round(float(answered["similarity"].mean()), 4),
            "faithfulness": round(float(answered["faithfulness"].dropna().mean()), 4),
            "fully_supported_answers": round(float((answered["faithfulness"].dropna() == 1).mean()), 4),
            "abstention_rate": round(float(g["abstained"].mean()), 4),
        }
        if "has_citation" in g and g["has_citation"].notna().any():
            out[setting]["citation_rate"] = round(float(answered["has_citation"].astype(float).mean()), 4)
            out[setting]["invalid_citation_rate"] = round(float(answered["invalid_citation"].astype(float).mean()), 4)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    passages = load_passages()
    questions = load_questions(set(passages.index))
    sample = sample_questions(questions)
    retriever = DenseRetriever(passages)
    print(f"{len(passages):,} passages, {len(questions):,} questions, sample {len(sample)}")

    ret = retrieval_metrics(retriever, questions)
    print("retrieval:", ret)
    ctx = contexts(retriever, sample)
    generate_all(sample, ctx)
    scored = score(sample, ctx, retriever)
    scored.to_csv(OUT / "scored_answers.csv", index=False)
    result = {"corpus_passages": len(passages), "questions": len(questions), "sample": len(sample),
              "retrieval_all_questions": ret, "answers": summarise(scored)}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
