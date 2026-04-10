"""
chunking.py
===========
Five modular, configurable chunking strategies.

Strategies
----------
1. fixed        – non-overlapping fixed-token windows
2. sentence     – sentence-boundary-aware accumulation
3. dynamic      – adaptive 200–800 token windows (flush at natural boundary)
4. overlapping  – fixed window with configurable stride/overlap
5. heading      – splits at detected section headings, sub-chunks large sections

All strategies return List[Chunk] with rich metadata.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    strategy: str
    text: str
    token_count: int
    index: int
    section: Optional[str] = None
    page_hint: int = -1
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return self.token_count

    def __repr__(self):
        return (f"Chunk(id={self.chunk_id!r}, strategy={self.strategy!r}, "
                f"tokens={self.token_count}, section={self.section!r})")


def _tok(text: str) -> int:
    """Approximate token count (whitespace split — fast & consistent)."""
    return len(text.split())


def _make(paper_id: str, strategy: str, idx: int, text: str,
          section: Optional[str] = None, page_hint: int = -1) -> Chunk:
    text = text.strip()
    return Chunk(
        chunk_id=f"{paper_id}_{strategy}_{idx:04d}",
        paper_id=paper_id,
        strategy=strategy,
        text=text,
        token_count=_tok(text),
        index=idx,
        section=section,
        page_hint=page_hint,
    )


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed-size
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, paper_id: str, size: int = 500) -> List[Chunk]:
    """Split text into non-overlapping windows of exactly `size` tokens."""
    words = text.split()
    chunks: List[Chunk] = []
    for i, start in enumerate(range(0, len(words), size)):
        window = words[start: start + size]
        if not window:
            break
        chunks.append(_make(paper_id, "fixed", i, " ".join(window)))
    return [c for c in chunks if c.token_count >= 20]


# ---------------------------------------------------------------------------
# Strategy 2 — Sentence-aware
# ---------------------------------------------------------------------------

def chunk_sentence(text: str, paper_id: str, size: int = 500) -> List[Chunk]:
    """Accumulate sentences until token budget is full, then flush."""
    sentences = sent_tokenize(text)
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_tok = 0
    idx = 0

    for sent in sentences:
        t = _tok(sent)
        if buf_tok + t > size and buf:
            chunks.append(_make(paper_id, "sentence", idx, " ".join(buf)))
            idx += 1
            buf, buf_tok = [], 0
        buf.append(sent)
        buf_tok += t

    if buf:
        chunks.append(_make(paper_id, "sentence", idx, " ".join(buf)))
    return [c for c in chunks if c.token_count >= 20]


# ---------------------------------------------------------------------------
# Strategy 3 — Dynamic (200–800 tokens)
# ---------------------------------------------------------------------------

def chunk_dynamic(text: str, paper_id: str,
                  lo: int = 200, hi: int = 800) -> List[Chunk]:
    """
    Accumulate sentences freely between lo and hi tokens.
    Flush early when we hit lo AND the last sentence ends naturally.
    """
    paragraphs = re.split(r"\n{2,}", text)
    sentences: List[str] = []
    for para in paragraphs:
        sentences.extend(sent_tokenize(para.strip()))

    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_tok = 0
    idx = 0

    for sent in sentences:
        t = _tok(sent)
        if buf_tok + t > hi and buf:
            chunks.append(_make(paper_id, "dynamic", idx, " ".join(buf)))
            idx += 1
            buf, buf_tok = [], 0

        buf.append(sent)
        buf_tok += t

        # Early flush: within target range at a natural boundary
        if buf_tok >= lo and sent.rstrip().endswith((".", "!", "?")):
            chunks.append(_make(paper_id, "dynamic", idx, " ".join(buf)))
            idx += 1
            buf, buf_tok = [], 0

    if buf:
        chunks.append(_make(paper_id, "dynamic", idx, " ".join(buf)))
    return [c for c in chunks if c.token_count >= 20]


# ---------------------------------------------------------------------------
# Strategy 4 — Overlapping
# ---------------------------------------------------------------------------

def chunk_overlapping(text: str, paper_id: str,
                      size: int = 500, overlap: int = 100) -> List[Chunk]:
    """Fixed windows with stride = size - overlap."""
    if overlap >= size:
        raise ValueError("overlap must be < size")
    words = text.split()
    stride = size - overlap
    chunks: List[Chunk] = []
    idx = 0
    for start in range(0, len(words), stride):
        window = words[start: start + size]
        if not window:
            break
        chunks.append(_make(paper_id, "overlapping", idx, " ".join(window)))
        idx += 1
    return [c for c in chunks if c.token_count >= 20]


# ---------------------------------------------------------------------------
# Strategy 5 — Heading-aware
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\b(abstract|introduction|related work|background|method(?:ology)?|"
    r"approach|experiment|result|discussion|conclusion|reference|appendix|"
    r"attention mechanism|transformer|pre.?training|fine.?tuning|"
    r"retrieval|generation|model|architecture|dataset|evaluation|"
    r"analysis|limitation|future work)\b",
    re.IGNORECASE,
)


def _detect_section_positions(text: str) -> List[int]:
    """Return character positions where new sections likely begin."""
    positions = []
    for m in _SECTION_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start: m.start()].strip()
        if len(prefix) < 12:
            positions.append(line_start)
    return sorted(set(positions))


def chunk_heading(text: str, paper_id: str, max_size: int = 800) -> List[Chunk]:
    """
    Split text at detected section headings.
    Large sections are sub-chunked with sentence-aware strategy.
    """
    positions = _detect_section_positions(text)

    if not positions:
        # Fall back to sentence chunking if no headings found
        logger.debug(f"[{paper_id}] No headings detected — falling back to sentence chunking")
        return chunk_sentence(text, paper_id, max_size)

    # Build sections
    boundaries = [0] + positions + [len(text)]
    raw_sections = []
    for i in range(len(boundaries) - 1):
        sec = text[boundaries[i]: boundaries[i + 1]].strip()
        if sec:
            raw_sections.append(sec)

    chunks: List[Chunk] = []
    idx = 0

    for sec in raw_sections:
        # Extract heading hint from first line
        first_line = sec.split("\n")[0].strip()[:80]
        heading = first_line if len(first_line) < 80 else None

        if _tok(sec) <= max_size:
            c = _make(paper_id, "heading", idx, sec, section=heading)
            chunks.append(c)
            idx += 1
        else:
            # Sub-chunk large section
            sub_chunks = chunk_sentence(sec, paper_id, max_size)
            for sc in sub_chunks:
                sc.chunk_id = f"{paper_id}_heading_{idx:04d}"
                sc.strategy = "heading"
                sc.index = idx
                sc.section = heading
                chunks.append(sc)
                idx += 1

    return [c for c in chunks if c.token_count >= 20]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed":       chunk_fixed,
    "sentence":    chunk_sentence,
    "dynamic":     chunk_dynamic,
    "overlapping": chunk_overlapping,
    "heading":     chunk_heading,
}


def get_chunks(text: str, paper_id: str, strategy: str = "dynamic",
               **kwargs) -> List[Chunk]:
    """
    Public factory.  Extra kwargs are forwarded to the strategy function.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {list(STRATEGIES)}")
    chunks = STRATEGIES[strategy](text, paper_id, **kwargs)
    logger.info(f"[{paper_id}] strategy={strategy}  chunks={len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# Comparison utility (used by experiments.py)
# ---------------------------------------------------------------------------

def compare_strategies(text: str, paper_id: str,
                        sizes: List[int] = (200, 500, 800)) -> List[dict]:
    """
    Run all strategies × sizes and return a comparison table.
    """
    rows = []
    total_words = len(text.split())
    for strat in STRATEGIES:
        for sz in sizes:
            kw = {}
            if strat == "fixed":        kw = {"size": sz}
            elif strat == "sentence":   kw = {"size": sz}
            elif strat == "dynamic":    kw = {"lo": max(50, sz // 4), "hi": sz}
            elif strat == "overlapping":kw = {"size": sz, "overlap": sz // 5}
            elif strat == "heading":    kw = {"max_size": sz}
            try:
                chunks = STRATEGIES[strat](text, paper_id, **kw)
                toks = [c.token_count for c in chunks]
                covered = sum(toks)
                rows.append({
                    "strategy":   strat,
                    "target_size":sz,
                    "num_chunks": len(chunks),
                    "avg_tokens": round(sum(toks) / max(len(toks), 1), 1),
                    "min_tokens": min(toks, default=0),
                    "max_tokens": max(toks, default=0),
                    "coverage":   round(covered / max(total_words, 1), 3),
                })
            except Exception as e:
                rows.append({"strategy": strat, "target_size": sz, "error": str(e)})
    return rows


if __name__ == "__main__":
    sample = (
        "The Transformer model relies entirely on attention mechanisms. "
        "Multi-head attention allows the model to attend to multiple representation subspaces. "
        "The encoder maps input to continuous representations. "
        "The decoder generates outputs one token at a time. "
        "Positional encodings are added to give the model a sense of order. "
        "Results show improvements over RNN-based sequence models on translation tasks. "
    ) * 30

    for row in compare_strategies(sample, "test", sizes=[200, 500]):
        if "error" not in row:
            print(f"  {row['strategy']:12s} sz={row['target_size']:4d}  "
                  f"chunks={row['num_chunks']:3d}  avg={row['avg_tokens']:6.1f}")
