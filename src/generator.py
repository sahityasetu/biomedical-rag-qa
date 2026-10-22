"""Answer generation with a small open LLM that runs locally (no API key, no cost).

Default: Qwen2.5-1.5B-Instruct with greedy decoding, so the same question always gives the same answer.
"""
import torch

LLM = "Qwen/Qwen2.5-1.5B-Instruct"
NOT_FOUND = "The sources do not contain the answer."
MAX_PASSAGE_WORDS = 300

SYSTEM_RAG = (
    "You are a careful biomedical research assistant. Answer the question using ONLY the numbered sources. "
    "Cite the sources you use like [1] or [2] after each sentence. Answer in 1-3 sentences. "
    "If it is a yes/no question, start with 'Yes' or 'No'. "
    f"If the sources do not contain the answer, reply exactly: '{NOT_FOUND}'"
)
SYSTEM_CLOSED = (
    "You are a biomedical research assistant. Answer the question in 1-3 sentences. "
    "If it is a yes/no question, start with 'Yes' or 'No'."
)


def build_messages(question: str, passages: list | None) -> list:
    if passages is None:
        return [{"role": "system", "content": SYSTEM_CLOSED}, {"role": "user", "content": question}]
    sources = "\n\n".join(f"[{i + 1}] {' '.join(p['text'].split()[:MAX_PASSAGE_WORDS])}" for i, p in enumerate(passages))
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]


class Generator:
    def __init__(self, model_name: str = LLM):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16 if self.device != "cpu" else torch.float32).to(self.device).eval()

    @torch.no_grad()
    def answer(self, question: str, passages: list | None = None, max_new_tokens: int = 160) -> str:
        prompt = self.tok.apply_chat_template(build_messages(question, passages), tokenize=False,
                                              add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
