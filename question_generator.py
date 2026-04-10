"""
question_generator.py
=====================
Generates 3–5 critical research questions per paper.

Categories
----------
  assumption  — challenges a key assumption the paper makes
  weakness    — identifies a known or suspected weakness
  improvement — proposes a concrete improvement or extension
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResearchQuestion:
    question:   str
    category:   str      # assumption | weakness | improvement
    rationale:  str = ""
    difficulty: str = "medium"   # easy | medium | hard


@dataclass
class QuestionSet:
    paper_id: str
    title:    str
    questions: List[ResearchQuestion] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## Critical Research Questions: {self.title}", ""]
        icons = {"assumption": "🔍", "weakness": "⚠️", "improvement": "💡"}
        for i, q in enumerate(self.questions, 1):
            icon = icons.get(q.category, "📌")
            lines.append(f"{i}. {icon} **[{q.category.upper()} | {q.difficulty}]**")
            lines.append(f"   **Q:** {q.question}")
            if q.rationale:
                lines.append(f"   *Why this matters:* {q.rationale}")
            lines.append("")
        return "\n".join(lines)

    def to_list(self) -> List[dict]:
        return [
            {
                "question":  q.question,
                "category":  q.category,
                "rationale": q.rationale,
                "difficulty": q.difficulty,
            }
            for q in self.questions
        ]


# ---------------------------------------------------------------------------
# Static question bank
# ---------------------------------------------------------------------------

_BANK: dict = {
    "attention": [
        ResearchQuestion(
            "Does the Transformer's quadratic O(n²) attention complexity represent a fundamental barrier for long-document processing, and can it be resolved without sacrificing expressivity?",
            category="weakness",
            rationale="Memory and compute scale quadratically with sequence length — a hard architectural limit for documents longer than ~512 tokens.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "Are sinusoidal positional encodings truly sufficient, or do learned / relative position representations (RoPE, ALiBi) provide a strict improvement?",
            category="assumption",
            rationale="The paper assumes fixed sinusoidal encodings are adequate; subsequent work (T5, LLaMA) moved to learned or relative encodings.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "Is multi-head attention strictly necessary, or does a single high-dimensional head with appropriate regularisation achieve equivalent performance?",
            category="assumption",
            rationale="The ablation in the paper is limited; the benefit of multiple heads vs. one large head is not fully theoretically justified.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "Could sparse attention (Longformer, BigBird) or linear attention (Performer) match full attention quality while reducing complexity to O(n)?",
            category="improvement",
            rationale="Sparse attention is the direct next step toward scaling Transformers to long contexts.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "How well does the Transformer generalise to truly low-resource language pairs where large parallel corpora are unavailable?",
            category="weakness",
            rationale="All results in the paper use large WMT datasets; data efficiency in low-resource regimes is untested.",
            difficulty="easy",
        ),
    ],
    "bert": [
        ResearchQuestion(
            "Does the Next Sentence Prediction (NSP) objective actually help, or does it introduce noise that hurts downstream performance?",
            category="weakness",
            rationale="RoBERTa later showed that removing NSP improves results — a direct challenge to BERT's pre-training design.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "Does the [MASK] token at pre-training time create a harmful distributional shift, since [MASK] never appears during fine-tuning?",
            category="assumption",
            rationale="BERT uses [MASK] for 15% of tokens at pre-train time but never at fine-tune time — this train/test mismatch is a known flaw addressed by ELECTRA.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "How can BERT-style models effectively handle inputs longer than 512 tokens without losing global cross-document context?",
            category="improvement",
            rationale="The 512-token hard limit is an architectural bottleneck for document-level NLP tasks.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "Can BERT's pre-training be made significantly more sample-efficient to reduce the enormous GPU compute required?",
            category="improvement",
            rationale="BERT requires days of TPU/GPU training; reducing this cost is critical for democratising NLP research.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "Are BERT's learned representations truly capturing deep semantics, or are they sophisticated surface-level pattern matching?",
            category="assumption",
            rationale="Probing studies show BERT captures syntax but deeper semantic understanding is debated.",
            difficulty="easy",
        ),
    ],
    "rag": [
        ResearchQuestion(
            "How does RAG behave when the dense retriever consistently returns irrelevant or adversarially poisoned passages?",
            category="weakness",
            rationale="RAG's output quality is entirely dependent on retrieval quality — a single point of failure that is not addressed in the paper.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "Does joint training of the retriever and generator in RAG actually improve both components, or does one dominate and bottleneck the other?",
            category="assumption",
            rationale="The paper assumes joint training is beneficial, but ablating frozen-retriever vs. jointly-trained is only partially explored.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "How can RAG be extended to multilingual or cross-lingual settings where the retrieval corpus and generation language differ?",
            category="improvement",
            rationale="The original RAG paper focuses exclusively on English; multilingual extension is an important open problem.",
            difficulty="hard",
        ),
        ResearchQuestion(
            "Can RAG models detect and resolve contradictions between retrieved passages without explicit contradiction-resolution training?",
            category="weakness",
            rationale="When multiple retrieved passages contain conflicting facts, the generator may produce inconsistent or averaged-out answers.",
            difficulty="medium",
        ),
        ResearchQuestion(
            "What is the optimal ratio of parametric knowledge (model weights) to non-parametric knowledge (retrieved passages) for maximum factual accuracy?",
            category="improvement",
            rationale="Understanding how much the model should 'trust' its own weights vs. retrieved text is fundamental to improving RAG reliability.",
            difficulty="hard",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class QuestionGenerator:
    """
    Generates critical research questions for a paper.

    Usage
    -----
    gen = QuestionGenerator()
    qset = gen.generate("attention", n=5)
    print(qset.to_markdown())
    """

    def __init__(self, pipeline=None):
        self._pipeline = pipeline   # optional — reserved for LLM-based generation

    def generate(self, paper_id: str, n: int = 5,
                 categories: Optional[List[str]] = None) -> QuestionSet:
        """
        Return up to n critical questions for the given paper.

        Parameters
        ----------
        paper_id   : "attention" | "bert" | "rag"
        n          : number of questions (3–5 recommended)
        categories : filter to specific categories (None = all)
        """
        from dataset import PAPERS
        title = PAPERS.get(paper_id, {}).get("title", paper_id)

        questions = _BANK.get(paper_id, [])
        if categories:
            questions = [q for q in questions if q.category in categories]
        questions = questions[:n]

        return QuestionSet(paper_id=paper_id, title=title, questions=questions)

    def generate_for_all(self, paper_ids: Optional[List[str]] = None,
                          n: int = 5) -> List[QuestionSet]:
        pids = paper_ids or list(_BANK.keys())
        return [self.generate(pid, n=n) for pid in pids]


if __name__ == "__main__":
    gen = QuestionGenerator()
    for pid in ["attention", "bert", "rag"]:
        qset = gen.generate(pid, n=5)
        print(qset.to_markdown())
        print("─" * 60)
