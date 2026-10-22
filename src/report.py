"""Chart of the four settings from reports/metrics.json.

Run from the project root after src.evaluate:  python -m src.report
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"closed_book": "No retrieval\n(LLM memory)", "rag": "RAG\n(top-3 retrieved)",
            "oracle": "Perfect retrieval\n(expert-labelled)", "wrong_context": "Wrong sources\n(trap test)"}
METRICS = {"yes_no_accuracy": ("Yes/no accuracy", "#2a78d6"),
           "abstention_rate": ("Says \"not in the sources\"", "#eb6834")}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def main():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())["answers"]
    x = np.arange(len(SETTINGS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (key, (label, color)) in enumerate(METRICS.items()):
        vals = [m[s][key] * 100 for s in SETTINGS]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=color, label=label)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    ax.axhline(50, color="#898781", lw=1, ls="--")
    ax.text(len(SETTINGS) - 0.5, 51, "coin flip", fontsize=8, color="#52514e", ha="right")
    ax.set_xticks(x, list(SETTINGS.values()))
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, 100)
    ax.set_title("Retrieval turns a coin-flip LLM into a useful one,\nbut with wrong sources it still answers 2 times in 3",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "settings_comparison.png", dpi=160)


if __name__ == "__main__":
    main()
