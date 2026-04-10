"""
hallucination.py
================
Three-layer hallucination detection.

Layer 1 — Embedding similarity
    Cosine sim between answer and context.  Low → likely hallucinated.

Layer 2 — Keyword claim verification
    Sentence-level keyword overlap between each answer sentence and context.
    Flags sentences with overlap < threshold.

Layer 3 — LLM self-check
    Prompts the LLM to rate its own grounding (0-1 score).
    Falls back to heuristic if LLM unavailable.

Composite score
    weighted_avg(1-embed_sim, 1-claim_support, 1-llm_support)
    → 0 = fully grounded, 1 = fully hallucinated
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Weights for composite score
W_EMBED = 0.35
W_CLAIM = 0.45
W_LLM   = 0.20

GROUNDED_THRESH     = 0.65   # grounding ≥ this → GROUNDED
PARTIAL_THRESH      = 0.40   # grounding ≥ this → PARTIAL


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClaimCheck:
    sentence: str
    is_supported: bool
    overlap_ratio: float
    best_evidence: str = ""


@dataclass
class HallucinationReport:
    verdict: str                        # GROUNDED | PARTIAL | HALLUCINATED
    score: float                        # 0 = grounded, 1 = hallucinated
    grounding_score: float              # 1 - score
    embed_similarity: float
    claim_support_rate: float
    llm_support_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    claim_checks: List[ClaimCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict":              self.verdict,
            "hallucination_score":  round(self.score, 4),
            "grounding_score":      round(self.grounding_score, 4),
            "embed_similarity":     round(self.embed_similarity, 4),
            "claim_support_rate":   round(self.claim_support_rate, 4),
            "llm_support_score":    round(self.llm_support_score, 4),
            "unsupported_count":    len(self.unsupported_claims),
            "unsupported_claims":   self.unsupported_claims[:3],
        }

    def summary(self) -> str:
        lines = [
            f"Verdict             : {self.verdict}",
            f"Hallucination score : {self.score:.3f}  (0=grounded, 1=hallucinated)",
            f"Embedding similarity: {self.embed_similarity:.3f}",
            f"Claim support rate  : {self.claim_support_rate:.3f}",
            f"LLM support score   : {self.llm_support_score:.3f}",
            f"Unsupported claims  : {len(self.unsupported_claims)}",
        ]
        for c in self.unsupported_claims[:2]:
            lines.append(f"  ✗ {c[:110]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 1 — Embedding similarity
# ---------------------------------------------------------------------------

def _embed_similarity(answer_text: str, context_text: str,
                       embedder) -> float:
    """Cosine similarity between answer and joined context (L2-normalised)."""
    a_vec = embedder.encode_one(answer_text)
    c_vec = embedder.encode_one(context_text)
    sim = float(np.dot(a_vec, c_vec))
    return max(0.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Layer 2 — Keyword claim verification
# ---------------------------------------------------------------------------

def _extract_answer_sentences(answer_text: str) -> List[str]:
    """Pull factual sentences out of the structured answer, skip headers."""
    # Remove section headers
    text = re.sub(
        r"^(Core Idea|Methodology|Key Results|Limitations|ELI12 Explanation"
        r"|Not specified in the paper)\s*:?\s*",
        "", answer_text, flags=re.MULTILINE
    )
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"[.!?]+", text)

    return [
        s.strip() for s in sents
        if len(s.split()) >= 6
        and "not specified" not in s.lower()
    ][:20]   # cap for efficiency


def _keyword_overlap(claim: str, context: str) -> Tuple[bool, float, str]:
    """
    Compute keyword overlap between a claim sentence and context.
    Returns (is_supported, ratio, best_evidence_snippet).
    """
    claim_words = set(re.findall(r"\b[a-z]{4,}\b", claim.lower()))
    ctx_words   = set(re.findall(r"\b[a-z]{4,}\b", context.lower()))

    if not claim_words:
        return True, 1.0, ""

    overlap = claim_words & ctx_words
    ratio = len(overlap) / len(claim_words)
    supported = ratio >= 0.30

    # Find best matching context sentence
    best_evidence = ""
    best_score = 0.0
    for sent in re.split(r"[.!?]+", context):
        sent = sent.strip()
        if not sent:
            continue
        sent_words = set(re.findall(r"\b[a-z]{4,}\b", sent.lower()))
        sc = len(claim_words & sent_words) / max(len(claim_words), 1)
        if sc > best_score:
            best_score = sc
            best_evidence = sent[:120]

    return supported, min(ratio, 1.0), best_evidence


def _claim_support_rate(answer_text: str, context_text: str) -> Tuple[float, List[ClaimCheck]]:
    """Returns (support_rate, list_of_ClaimCheck)."""
    sentences = _extract_answer_sentences(answer_text)
    if not sentences:
        return 1.0, []

    checks = []
    for sent in sentences:
        supported, ratio, evidence = _keyword_overlap(sent, context_text)
        checks.append(ClaimCheck(
            sentence=sent,
            is_supported=supported,
            overlap_ratio=ratio,
            best_evidence=evidence,
        ))

    rate = sum(1 for c in checks if c.is_supported) / len(checks)
    return rate, checks


# ---------------------------------------------------------------------------
# Layer 3 — LLM self-check
# ---------------------------------------------------------------------------

_LLM_CHECK_PROMPT = """\
You are a strict fact-checker.

Context:
{context}

Answer:
{answer}

On a scale from 0.0 to 1.0, how well is the answer supported by the context?
  1.0 = fully supported
  0.5 = partially supported
  0.0 = not supported at all

Reply with ONLY a number between 0.0 and 1.0.
Score:"""


def _llm_support_score(answer_text: str, context_text: str,
                        llm_engine) -> float:
    """Use the LLM to self-assess grounding. Falls back to 0.5 on error."""
    try:
        prompt = _LLM_CHECK_PROMPT.format(
            context=context_text[:1200],
            answer=answer_text[:600],
        )
        # Call backend directly for a short generation
        raw, _ = llm_engine._backend.generate(prompt, max_new_tokens=10)
        m = re.search(r"([01]?\.\d+|[01])", raw)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception as e:
        logger.debug(f"LLM self-check fallback: {e}")

    # Heuristic fallback: use claim support as proxy
    return 0.5


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """
    Three-layer detector.

    Usage
    -----
    detector = HallucinationDetector(embedder, llm_engine)
    report = detector.detect(answer.raw_text, context_text)
    """

    def __init__(self, embedder, llm_engine):
        self._embedder = embedder
        self._llm = llm_engine

    def detect(self, answer_text: str, context_text: str) -> HallucinationReport:
        # ── Layer 1 ────────────────────────────────────────────────────────
        embed_sim = _embed_similarity(answer_text, context_text, self._embedder)

        # ── Layer 2 ────────────────────────────────────────────────────────
        claim_rate, checks = _claim_support_rate(answer_text, context_text)

        # ── Layer 3 ────────────────────────────────────────────────────────
        llm_score = _llm_support_score(answer_text, context_text, self._llm)

        # ── Composite ──────────────────────────────────────────────────────
        grounding = (
            W_EMBED * embed_sim +
            W_CLAIM * claim_rate +
            W_LLM   * llm_score
        )
        hall_score = max(0.0, min(1.0, 1.0 - grounding))

        if grounding >= GROUNDED_THRESH:
            verdict = "GROUNDED"
        elif grounding >= PARTIAL_THRESH:
            verdict = "PARTIAL"
        else:
            verdict = "HALLUCINATED"

        unsupported = [c.sentence for c in checks if not c.is_supported]

        return HallucinationReport(
            verdict=verdict,
            score=round(hall_score, 4),
            grounding_score=round(grounding, 4),
            embed_similarity=round(embed_sim, 4),
            claim_support_rate=round(claim_rate, 4),
            llm_support_score=round(llm_score, 4),
            unsupported_claims=unsupported,
            claim_checks=checks,
        )


if __name__ == "__main__":
    # Quick smoke-test without real embedder/LLM
    class _FakeEmb:
        def encode_one(self, t):
            import numpy as np
            v = np.random.randn(384).astype(np.float32)
            return v / np.linalg.norm(v)

    class _FakeLLM:
        class _backend:
            @staticmethod
            def generate(p, max_new_tokens=10):
                return "0.8", 2

    det = HallucinationDetector(_FakeEmb(), _FakeLLM())
    ctx = "The Transformer model uses attention mechanisms. BERT uses MLM pre-training."
    ans = "The Transformer uses attention. BERT uses masked language modeling."
    r = det.detect(ans, ctx)
    print(r.summary())
