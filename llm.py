"""
llm.py
======
LLM backends (TinyLlama / Mock) + structured answer generation.

Mandatory output format
-----------------------
  Core Idea:
  Methodology:
  Key Results:
  Limitations:
  ELI12 Explanation:

  → If section is absent: "Not specified in the paper"

Grounding rule
--------------
  The LLM sees ONLY retrieved context chunks — no external knowledge.
  The strict prompt explicitly forbids using outside information.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

NOT_SPECIFIED = "Not specified in the paper"
SECTIONS = ["Core Idea", "Methodology", "Key Results", "Limitations", "ELI12 Explanation"]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

STRICT_PROMPT = """\
You are an expert AI research analyst.
Answer the question using ONLY the provided context passages.
Do NOT use any external knowledge. If information is missing, write:
  "Not specified in the paper"

CONTEXT PASSAGES:
{context}

QUESTION: {question}

Respond in this EXACT format (all five sections required):

Core Idea:
[Explain the central contribution using ONLY the context]

Methodology:
[Describe the methods/techniques using ONLY the context]

Key Results:
[State the key results/findings from the context]

Limitations:
[List known limitations from the context]

ELI12 Explanation:
[Explain to a curious 12-year-old using simple words, based on context]

ANSWER:"""

OPEN_PROMPT = """\
You are an expert AI research analyst.
Answer the question based on the context, and you may use general knowledge to clarify.

CONTEXT PASSAGES:
{context}

QUESTION: {question}

Respond in this EXACT format:

Core Idea:
[Central contribution]

Methodology:
[Methods and techniques]

Key Results:
[Key findings]

Limitations:
[Known limitations]

ELI12 Explanation:
[Simple explanation for a 12-year-old]

ANSWER:"""


def build_prompt(question: str, context_chunks: List[str],
                 style: str = "strict") -> str:
    ctx = "\n\n---\n\n".join(
        f"[Passage {i+1}]:\n{c}" for i, c in enumerate(context_chunks)
    )
    template = STRICT_PROMPT if style == "strict" else OPEN_PROMPT
    return template.format(context=ctx, question=question)


# ---------------------------------------------------------------------------
# Structured answer
# ---------------------------------------------------------------------------

@dataclass
class StructuredAnswer:
    core_idea:   str = NOT_SPECIFIED
    methodology: str = NOT_SPECIFIED
    key_results: str = NOT_SPECIFIED
    limitations: str = NOT_SPECIFIED
    eli12:       str = NOT_SPECIFIED
    raw_text:    str = ""
    tokens_used: int = 0
    latency_s:   float = 0.0
    model_name:  str = ""
    prompt_style: str = "strict"

    def to_dict(self) -> dict:
        return {
            "Core Idea":         self.core_idea,
            "Methodology":       self.methodology,
            "Key Results":       self.key_results,
            "Limitations":       self.limitations,
            "ELI12 Explanation": self.eli12,
        }

    def completeness(self) -> float:
        filled = sum(
            1 for v in self.to_dict().values()
            if v and v != NOT_SPECIFIED
        )
        return filled / len(SECTIONS)

    def is_complete(self) -> bool:
        return self.completeness() == 1.0

    def __str__(self):
        divider = "─" * 52
        lines = []
        for k, v in self.to_dict().items():
            lines.append(f"\n{divider}")
            lines.append(f"  {k}")
            lines.append(divider)
            lines.append(f"  {v[:400]}")
        lines.append(f"\n[completeness={self.completeness():.0%} | "
                     f"tokens={self.tokens_used} | {self.latency_s*1000:.0f}ms | "
                     f"model={self.model_name}]")
        return "\n".join(lines)


def _parse(raw: str) -> StructuredAnswer:
    """Parse LLM output into StructuredAnswer."""
    ans = StructuredAnswer(raw_text=raw)
    patterns = {
        "core_idea":   r"Core Idea:\s*(.*?)(?=Methodology:|Key Results:|Limitations:|ELI12|$)",
        "methodology": r"Methodology:\s*(.*?)(?=Key Results:|Limitations:|ELI12|Core Idea:|$)",
        "key_results": r"Key Results:\s*(.*?)(?=Limitations:|ELI12|Core Idea:|Methodology:|$)",
        "limitations": r"Limitations:\s*(.*?)(?=ELI12|Core Idea:|Methodology:|Key Results:|$)",
        "eli12":       r"ELI12 Explanation:\s*(.*?)(?=Core Idea:|Methodology:|Key Results:|Limitations:|$)",
    }
    for attr, pat in patterns.items():
        m = re.search(pat, raw, re.DOTALL | re.IGNORECASE)
        if m:
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            if val and len(val) > 5 and val.lower() != "not specified in the paper":
                setattr(ans, attr, val)
            elif val and val.lower() == "not specified in the paper":
                setattr(ans, attr, NOT_SPECIFIED)
    return ans


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _MockBackend:
    """Deterministic mock — fast, no GPU. Grounds answer in first context passage."""
    name = "mock-llm"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple:
        # Extract first passage to ground the response
        m = re.search(r"\[Passage 1\]:\n(.*?)(?:\n---|\Z)", prompt, re.DOTALL)
        excerpt = m.group(1).strip()[:200] if m else "the paper"

        text = (
            f"Core Idea:\n"
            f"Based on the retrieved context, the paper's core contribution is: {excerpt[:160]}\n\n"
            f"Methodology:\n"
            f"The methodology described in the context involves systematic experiments "
            f"and ablation studies using standard benchmarks as detailed in the passages.\n\n"
            f"Key Results:\n"
            f"The context reports competitive improvements over prior baselines on the evaluated tasks. "
            f"Specific metrics are described in the retrieved passages above.\n\n"
            f"Limitations:\n"
            f"The context mentions computational cost and data requirements as key limitations.\n\n"
            f"ELI12 Explanation:\n"
            f"Imagine a very smart assistant that reads many books to answer questions. "
            f"This paper teaches the assistant a smarter way to find and use information."
        )
        tokens = len(prompt.split()) + len(text.split())
        return text, tokens


class _TinyLlamaBackend:
    """TinyLlama-1.1B-Chat — runs on free Colab T4 GPU or CPU."""
    name = "TinyLlama-1.1B-Chat"

    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        logger.info(f"Loading {model_id} ...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline(
            "text-generation", model=mdl, tokenizer=tok, device=device
        )
        logger.info("TinyLlama loaded ✓")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple:
        self._load()
        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"][len(prompt):]
        tokens = len(prompt.split()) + len(text.split())
        return text, tokens


_BACKENDS = {
    "mock":      _MockBackend,
    "tinyllama": _TinyLlamaBackend,
}


# ---------------------------------------------------------------------------
# LLM engine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Public interface.

    Usage
    -----
    engine = LLMEngine(backend="mock")   # or "tinyllama"
    ans = engine.answer("What is BERT?", context_chunks, style="strict")
    """

    def __init__(self, backend: str = "mock"):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from {list(_BACKENDS)}")
        self._backend = _BACKENDS[backend]()
        logger.info(f"LLMEngine: backend={self._backend.name}")

    def answer(self, question: str, context_chunks: List[str],
               style: str = "strict", max_new_tokens: int = 512) -> StructuredAnswer:
        """
        Build prompt → generate → parse → return StructuredAnswer.
        context_chunks must contain ONLY retrieved context (no external text).
        """
        prompt = build_prompt(question, context_chunks, style=style)
        t0 = time.time()
        raw, tokens = self._backend.generate(prompt, max_new_tokens=max_new_tokens)
        latency = time.time() - t0

        ans = _parse(raw)
        ans.raw_text = raw
        ans.tokens_used = tokens
        ans.latency_s = latency
        ans.model_name = self._backend.name
        ans.prompt_style = style

        logger.info(f"Answer: completeness={ans.completeness():.0%}  "
                    f"tokens={tokens}  latency={latency*1000:.0f}ms")
        return ans

    @property
    def backend_name(self) -> str:
        return self._backend.name


if __name__ == "__main__":
    engine = LLMEngine(backend="mock")
    ctx = [
        "The Transformer uses multi-head self-attention instead of recurrence.",
        "Results show BLEU 28.4 on WMT 2014 English-to-German translation.",
        "One limitation is the quadratic O(n²) attention complexity.",
    ]
    ans = engine.answer("What is the Transformer?", ctx)
    print(ans)
