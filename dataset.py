"""
dataset.py
==========
Downloads and caches PDFs from arXiv.
Extracts clean text page-by-page using PyMuPDF.

Dataset split
-------------
  Train / Test : Attention Is All You Need  (1706.03762)
                 BERT                        (1810.04805)
  Validation   : RAG                         (2005.11401)
  → RAG paper must NOT be loaded until validation phase.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PAPERS = {
    "attention": {
        "title":    "Attention Is All You Need",
        "authors":  "Vaswani et al.",
        "year":     2017,
        "url":      "https://arxiv.org/pdf/1706.03762",
        "filename": "attention.pdf",
        "split":    "train_test",
    },
    "bert": {
        "title":    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors":  "Devlin et al.",
        "year":     2018,
        "url":      "https://arxiv.org/pdf/1810.04805",
        "filename": "bert.pdf",
        "split":    "train_test",
    },
    "rag": {
        "title":    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors":  "Lewis et al.",
        "year":     2020,
        "url":      "https://arxiv.org/pdf/2005.11401",
        "filename": "rag.pdf",
        "split":    "validation",
    },
}

DATA_DIR = "data"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Page:
    number: int
    text: str
    headings: List[str] = field(default_factory=list)


@dataclass
class PaperDoc:
    paper_id: str
    title: str
    authors: str
    year: int
    split: str
    pages: List[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(p.text for p in self.pages)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def __repr__(self):
        return (f"PaperDoc(id={self.paper_id!r}, pages={self.page_count}, "
                f"words={self.word_count:,})")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (research-bot/2.0)"}
    logger.info(f"Downloading {url} ...")
    resp = requests.get(url, headers=headers, timeout=120, stream=True)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info(f"Saved → {dest}  ({dest.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(
    r"^(?:\d+\.?\s+)?[A-Z][A-Z\s\-:]{3,60}$", re.MULTILINE
)
_KNOWN_SECTIONS = re.compile(
    r"\b(abstract|introduction|related work|background|methodology|method|"
    r"experiment|result|discussion|conclusion|reference|appendix|"
    r"attention|transformer|pre.?training|fine.?tuning|retrieval|generation)\b",
    re.IGNORECASE,
)


def _extract_pdf(pdf_path: str) -> List[Page]:
    doc = fitz.open(pdf_path)
    pages: List[Page] = []
    for i, page in enumerate(doc):
        raw_text = page.get_text("text")
        # clean
        text = re.sub(r"\s+", " ", raw_text).strip()
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # de-hyphenate
        # detect headings
        headings = []
        for line in raw_text.splitlines():
            line = line.strip()
            if _KNOWN_SECTIONS.match(line) or (
                len(line) < 80 and line.isupper() and len(line.split()) >= 2
            ):
                headings.append(line)
        pages.append(Page(number=i + 1, text=text, headings=headings))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_paper(paper_id: str, data_dir: str = DATA_DIR) -> PaperDoc:
    """Download (if needed) and return a PaperDoc."""
    if paper_id not in PAPERS:
        raise ValueError(f"Unknown paper {paper_id!r}. Choose from {list(PAPERS)}")
    meta = PAPERS[paper_id]
    os.makedirs(data_dir, exist_ok=True)
    pdf_path = Path(data_dir) / meta["filename"]

    if not pdf_path.exists():
        _download(meta["url"], pdf_path)
    else:
        logger.info(f"Using cached PDF: {pdf_path}")

    pages = _extract_pdf(str(pdf_path))
    doc = PaperDoc(
        paper_id=paper_id,
        title=meta["title"],
        authors=meta["authors"],
        year=meta["year"],
        split=meta["split"],
        pages=pages,
    )
    logger.info(f"Loaded: {doc}")
    return doc


def load_train_test(data_dir: str = DATA_DIR):
    """Load Attention + BERT (train/test papers)."""
    return {k: load_paper(k, data_dir) for k in ("attention", "bert")}


def load_validation(data_dir: str = DATA_DIR) -> PaperDoc:
    """Load RAG paper (validation only — call after system is tuned)."""
    return load_paper("rag", data_dir)


if __name__ == "__main__":
    docs = load_train_test()
    for doc in docs.values():
        print(doc)
