"""
comparison.py
=============
Multi-paper comparison engine.

Compares papers on:  Problem · Method · Results · Limitations · When to use
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static knowledge base (always-available; for UI even without LLM)
# ---------------------------------------------------------------------------

STATIC_PROFILES: Dict[str, Dict[str, str]] = {
    "attention": {
        "Problem":      "Sequence-to-sequence transduction (translation, summarisation) relying on recurrent / convolutional networks is slow to train and hard to parallelise.",
        "Method":       "Pure attention-based encoder-decoder (Transformer). Multi-head self-attention + position-wise FFN + positional encodings replace recurrence entirely.",
        "Results":      "BLEU 28.4 on WMT-14 EN→DE (new SOTA); BLEU 41.0 on EN→FR. ~8× faster training than best RNN ensemble. Attention patterns are interpretable.",
        "Limitations":  "Quadratic O(n²) attention complexity limits long-sequence scalability. Fixed maximum context window. Requires large parallel corpora.",
        "When to use":  "Machine translation, text generation, seq2seq tasks where training speed matters and sequences are ≤ 512 tokens.",
    },
    "bert": {
        "Problem":      "Pre-training language representations uni-directionally (left-to-right or shallow concatenation) limits the power of learned representations for NLU tasks.",
        "Method":       "Bidirectional Transformer pre-trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Fine-tuned end-to-end with a task head.",
        "Results":      "Absolute improvements on 11 NLP tasks: GLUE +7.7%, SQuAD 1.1 F1 93.2%, SQuAD 2.0 F1 83.1%, MultiNLI +4.7%.",
        "Limitations":  "Cannot generate text natively (encoder-only). NSP task shown to be noisy (cf. RoBERTa). Heavy pre-training cost. [MASK] not seen at fine-tune time.",
        "When to use":  "Text classification, NER, extractive QA, sentence-pair tasks, feature extraction — any NLU task with labelled fine-tune data.",
    },
    "rag": {
        "Problem":      "Parametric knowledge in LLM weights is static, opaque, and hard to update. Knowledge-intensive tasks require up-to-date, verifiable facts.",
        "Method":       "Non-parametric retrieval (Dense Passage Retrieval) + parametric generation (BART/seq2seq). Retrieves top-k passages; generator conditions on them.",
        "Results":      "Outperforms specialised SOTA on Natural Questions, WebQ, TriviaQA (open-domain); SOTA on MS-MARCO NLG; strong on Jeopardy question generation.",
        "Limitations":  "Retrieval quality is a bottleneck. Joint training is complex. Inference speed depends on retrieval corpus size. No real-time corpus update during generation.",
        "When to use":  "Open-domain QA, fact verification, knowledge-intensive NLG — any task needing factual grounding beyond what fits in LLM weights.",
    },
}

DIMENSIONS = ["Problem", "Method", "Results", "Limitations", "When to use"]


@dataclass
class ComparisonTable:
    paper_ids: List[str]
    data: Dict[str, Dict[str, str]]   # {dimension: {paper_id: text}}
    recommendation: str = ""

    def to_markdown(self) -> str:
        titles = {pid: STATIC_PROFILES.get(pid, {}).get("title", pid)
                  for pid in self.paper_ids}
        header = "| Dimension | " + " | ".join(self.paper_ids) + " |"
        sep    = "|---|" + "---|" * len(self.paper_ids)
        rows   = [header, sep]
        for dim in DIMENSIONS:
            cells = " | ".join(
                self.data.get(dim, {}).get(pid, "N/A")[:120]
                for pid in self.paper_ids
            )
            rows.append(f"| **{dim}** | {cells} |")
        if self.recommendation:
            rows.append(f"\n**Recommendation:** {self.recommendation}")
        return "\n".join(rows)

    def as_dict_list(self) -> List[dict]:
        rows = []
        for dim in DIMENSIONS:
            row = {"Dimension": dim}
            for pid in self.paper_ids:
                row[pid] = self.data.get(dim, {}).get(pid, "N/A")
            rows.append(row)
        return rows


class ComparisonEngine:
    """
    Build a ComparisonTable for a set of papers.
    Uses static profiles when available; can augment with LLM-retrieved answers.
    """

    def __init__(self, pipeline=None):
        self._pipeline = pipeline

    def compare(self, paper_ids: List[str]) -> ComparisonTable:
        data: Dict[str, Dict[str, str]] = {dim: {} for dim in DIMENSIONS}

        for pid in paper_ids:
            profile = STATIC_PROFILES.get(pid, {})
            for dim in DIMENSIONS:
                data[dim][pid] = profile.get(dim, "Not available in static profile.")

        recommendation = self._generate_recommendation(paper_ids)
        return ComparisonTable(
            paper_ids=paper_ids,
            data=data,
            recommendation=recommendation,
        )

    def _generate_recommendation(self, paper_ids: List[str]) -> str:
        mapping = {
            frozenset(["attention", "bert"]): (
                "Use **Transformer** for generation/translation tasks. "
                "Use **BERT** for classification/NLU. Both are complementary."
            ),
            frozenset(["attention", "rag"]): (
                "Use **Transformer** for closed-context generation. "
                "Use **RAG** when external, up-to-date knowledge is required."
            ),
            frozenset(["bert", "rag"]): (
                "Use **BERT** for discriminative NLU with labelled data. "
                "Use **RAG** for open-domain QA without labelled answers."
            ),
            frozenset(["attention", "bert", "rag"]): (
                "All three form a natural progression: Transformer → BERT → RAG. "
                "Choose based on task: generation → Transformer; "
                "classification → BERT; knowledge-intensive QA → RAG."
            ),
        }
        return mapping.get(frozenset(paper_ids), "See individual profiles for task-specific guidance.")


if __name__ == "__main__":
    engine = ComparisonEngine()
    table  = engine.compare(["attention", "bert", "rag"])
    print(table.to_markdown())
