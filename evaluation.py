"""
evaluation.py
=============
All evaluation metrics used in experiments.

Retrieval metrics
-----------------
  recall_at_k, precision_at_k, context_relevance

Answer quality metrics
----------------------
  faithfulness, completeness, hallucination_rate

Chunking metrics
----------------
  coherence, redundancy, coverage

System metrics
--------------
  latency_ms, tokens_used
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    recall: dict        = field(default_factory=dict)   # {k: float}
    precision: dict     = field(default_factory=dict)   # {k: float}
    context_relevance: float = 0.0

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.recall.items():
            d[f"recall@{k}"] = round(v, 4)
        for k, v in self.precision.items():
            d[f"precision@{k}"] = round(v, 4)
        d["context_relevance"] = round(self.context_relevance, 4)
        return d


@dataclass
class AnswerMetrics:
    faithfulness:       float = 0.0
    completeness:       float = 0.0
    hallucination_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness":       round(self.faithfulness, 4),
            "completeness":       round(self.completeness, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


@dataclass
class ChunkingMetrics:
    coherence:  float = 0.0   # avg cosine sim of adjacent chunks
    redundancy: float = 0.0   # avg cosine sim of non-adjacent sample
    coverage:   float = 0.0   # fraction of source words covered

    def to_dict(self) -> dict:
        return {
            "coherence":  round(self.coherence, 4),
            "redundancy": round(self.redundancy, 4),
            "coverage":   round(self.coverage, 4),
        }


@dataclass
class SystemMetrics:
    latency_ms:  float = 0.0
    tokens_used: int   = 0
    chunks_retrieved: int = 0

    def to_dict(self) -> dict:
        return {
            "latency_ms":       round(self.latency_ms, 1),
            "tokens_used":      self.tokens_used,
            "chunks_retrieved": self.chunks_retrieved,
        }


@dataclass
class EvalRecord:
    """One complete evaluation record (one query × one config)."""
    paper_id:    str
    question:    str
    strategy:    str
    target_size: int
    top_k:       int
    prompt_style: str
    retrieval:   RetrievalMetrics  = field(default_factory=RetrievalMetrics)
    answer:      AnswerMetrics     = field(default_factory=AnswerMetrics)
    chunking:    ChunkingMetrics   = field(default_factory=ChunkingMetrics)
    system:      SystemMetrics     = field(default_factory=SystemMetrics)
    verdict:     str               = ""

    def flat(self) -> dict:
        d = {
            "paper_id": self.paper_id,
            "question": self.question[:60] + "...",
            "strategy": self.strategy,
            "target_size": self.target_size,
            "top_k": self.top_k,
            "prompt_style": self.prompt_style,
            "verdict": self.verdict,
        }
        d.update(self.retrieval.to_dict())
        d.update(self.answer.to_dict())
        d.update(self.chunking.to_dict())
        d.update(self.system.to_dict())
        return d


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top = retrieved_ids[:k]
    return len(set(top) & relevant_ids) / max(len(relevant_ids), 1)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in relevant_ids) / max(k, 1)


def compute_retrieval_metrics(retrieved_ids: List[str],
                               relevant_ids: Set[str],
                               ks: tuple = (2, 3, 5),
                               context_relevance: float = 0.0) -> RetrievalMetrics:
    m = RetrievalMetrics(context_relevance=context_relevance)
    for k in ks:
        m.recall[k]    = recall_at_k(retrieved_ids, relevant_ids, k)
        m.precision[k] = precision_at_k(retrieved_ids, relevant_ids, k)
    return m


# ---------------------------------------------------------------------------
# Answer quality metrics
# ---------------------------------------------------------------------------

def compute_answer_metrics(structured_answer,
                            hall_report,
                            context_chunks: List[str],
                            embedder) -> AnswerMetrics:
    """
    faithfulness   — avg cosine sim between answer sentences and context
    completeness   — fraction of required sections filled
    hallucination_rate — from hallucination detector
    """
    # Faithfulness via embedding cosine sim
    ans_text = " ".join(structured_answer.to_dict().values())
    ctx_text = " ".join(context_chunks)
    a_vec = embedder.encode_one(ans_text)
    c_vec = embedder.encode_one(ctx_text)
    faithfulness = max(0.0, float(np.dot(a_vec, c_vec)))

    return AnswerMetrics(
        faithfulness=faithfulness,
        completeness=structured_answer.completeness(),
        hallucination_rate=hall_report.score,
    )


# ---------------------------------------------------------------------------
# Chunking metrics
# ---------------------------------------------------------------------------

def compute_chunking_metrics(chunks: List[object],
                              source_text: str,
                              embedder) -> ChunkingMetrics:
    if not chunks:
        return ChunkingMetrics()

    texts = [c.text for c in chunks]
    vecs = embedder.encode(texts)  # (N, dim)

    # Coherence: avg cosine of adjacent pairs
    adj_sims = [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
    coherence = float(np.mean(adj_sims)) if adj_sims else 1.0

    # Redundancy: avg cosine of sampled non-adjacent pairs
    n = len(vecs)
    non_adj = []
    for i in range(n):
        for j in range(i + 2, min(i + 6, n)):
            non_adj.append(float(np.dot(vecs[i], vecs[j])))
    redundancy = float(np.mean(non_adj)) if non_adj else 0.0

    # Coverage: fraction of source vocabulary covered by chunks
    src_words   = set(source_text.lower().split())
    chunk_words = set()
    for c in chunks:
        chunk_words.update(c.text.lower().split())
    coverage = len(chunk_words & src_words) / max(len(src_words), 1)

    return ChunkingMetrics(
        coherence=max(0.0, coherence),
        redundancy=max(0.0, redundancy),
        coverage=min(1.0, coverage),
    )


# ---------------------------------------------------------------------------
# Evaluator class (convenience wrapper)
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Stateful evaluator that accumulates EvalRecords.

    Usage
    -----
    ev = Evaluator(embedder)
    record = ev.evaluate(paper_id, question, chunks, retrieved, answer, hall_report, ...)
    df = ev.as_dataframe()
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.records: List[EvalRecord] = []

    def evaluate(self,
                 paper_id: str,
                 question: str,
                 strategy: str,
                 target_size: int,
                 top_k: int,
                 prompt_style: str,
                 all_chunks: List[object],
                 retrieved: List[object],         # SearchResult objects
                 structured_answer,
                 hall_report,
                 source_text: str,
                 start_time: float,
                 relevant_ids: Optional[Set[str]] = None) -> EvalRecord:

        ctx_chunks = [r.chunk.text for r in retrieved]

        # Retrieval metrics
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]
        rel_ids = relevant_ids or {retrieved_ids[0]} if retrieved_ids else set()
        ctx_rel = self._context_relevance(question, ctx_chunks)
        ret_m = compute_retrieval_metrics(retrieved_ids, rel_ids,
                                          ks=(2, 3, 5),
                                          context_relevance=ctx_rel)

        # Answer metrics
        ans_m = compute_answer_metrics(structured_answer, hall_report,
                                       ctx_chunks, self.embedder)

        # Chunking metrics
        ck_m = compute_chunking_metrics(all_chunks, source_text, self.embedder)

        # System metrics
        sys_m = SystemMetrics(
            latency_ms=(time.time() - start_time) * 1000,
            tokens_used=structured_answer.tokens_used,
            chunks_retrieved=len(retrieved),
        )

        record = EvalRecord(
            paper_id=paper_id,
            question=question,
            strategy=strategy,
            target_size=target_size,
            top_k=top_k,
            prompt_style=prompt_style,
            retrieval=ret_m,
            answer=ans_m,
            chunking=ck_m,
            system=sys_m,
            verdict=hall_report.verdict,
        )
        self.records.append(record)
        return record

    def _context_relevance(self, query: str, ctx_chunks: List[str]) -> float:
        if not ctx_chunks:
            return 0.0
        q_vec = self.embedder.encode_one(query)
        scores = []
        for text in ctx_chunks:
            c_vec = self.embedder.encode_one(text)
            scores.append(float(np.dot(q_vec, c_vec)))
        return float(np.mean(scores))

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame([r.flat() for r in self.records])

    def summary(self) -> dict:
        if not self.records:
            return {}
        import pandas as pd
        df = self.as_dataframe()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[num_cols].mean().round(4).to_dict()
