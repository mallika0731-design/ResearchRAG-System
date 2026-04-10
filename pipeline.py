"""
pipeline.py
===========
Central RAG pipeline orchestrator.

Flow
----
  PaperDoc → chunks → embeddings → FAISS → retrieve → LLM → answer
                                                           ↓
                                              HallucinationDetector → report

Dataset discipline
------------------
  load_train_test() → loads Attention + BERT for system development
  load_validation() → loads RAG paper ONLY for final validation
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataset     import PaperDoc, load_paper, load_train_test, load_validation
from chunking    import get_chunks, Chunk, STRATEGIES as CHUNK_STRATEGIES
from retrieval   import RetrievalPipeline, SearchResult
from llm         import LLMEngine, StructuredAnswer
from hallucination import HallucinationDetector, HallucinationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Chunking
    chunk_strategy:  str = "dynamic"
    chunk_size:      int = 500        # tokens (target / max depending on strategy)
    chunk_overlap:   int = 100        # used by overlapping strategy only

    # Retrieval
    top_k:           int = 5
    use_reranker:    bool = False
    embed_model:     str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_backend:     str = "mock"     # "mock" | "tinyllama"
    prompt_style:    str = "strict"   # "strict" | "open"
    max_new_tokens:  int = 512

    # Runtime
    device:          str = "cpu"
    data_dir:        str = "data"


# ---------------------------------------------------------------------------
# Pipeline response
# ---------------------------------------------------------------------------

@dataclass
class PipelineResponse:
    paper_id:   str
    question:   str
    answer:     StructuredAnswer
    retrieved:  List[SearchResult]
    hall_report: HallucinationReport
    latency_ms: float
    config:     dict = field(default_factory=dict)

    @property
    def context_text(self) -> str:
        return " ".join(r.chunk.text for r in self.retrieved)

    def summary(self) -> str:
        lines = [
            f"Paper    : {self.paper_id}",
            f"Question : {self.question}",
            f"Latency  : {self.latency_ms:.0f}ms",
            f"Chunks   : {len(self.retrieved)}",
            "",
            str(self.answer),
            "",
            self.hall_report.summary(),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage
    -----
    pipeline = RAGPipeline(PipelineConfig(llm_backend="mock"))
    pipeline.load_paper("attention")
    pipeline.load_paper("bert")
    response = pipeline.query("attention", "What is multi-head attention?")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._docs:   Dict[str, PaperDoc] = {}
        self._chunks: Dict[str, List[Chunk]] = {}
        self._retrieval = RetrievalPipeline(
            embed_model=self.config.embed_model,
            use_reranker=self.config.use_reranker,
            device=self.config.device,
        )
        self._llm = LLMEngine(backend=self.config.llm_backend)
        self._detector = HallucinationDetector(
            embedder=self._retrieval.embedder,
            llm_engine=self._llm,
        )
        logger.info(f"RAGPipeline ready: {self.config}")

    # ── Paper management ────────────────────────────────────────────────────

    def load_paper(self, paper_id: str) -> PaperDoc:
        """Download, extract, chunk, and index one paper."""
        if paper_id in self._docs:
            logger.info(f"[{paper_id}] already loaded — skipping.")
            return self._docs[paper_id]

        doc = load_paper(paper_id, self.config.data_dir)
        self._docs[paper_id] = doc

        chunks = self._make_chunks(doc)
        self._chunks[paper_id] = chunks

        logger.info(f"[{paper_id}] indexing {len(chunks)} chunks …")
        self._retrieval.index_chunks(chunks, show_progress=True)
        logger.info(f"[{paper_id}] indexed ✓")
        return doc

    def load_train_test(self) -> Dict[str, PaperDoc]:
        docs = load_train_test(self.config.data_dir)
        for pid, doc in docs.items():
            if pid not in self._docs:
                self._docs[pid] = doc
                chunks = self._make_chunks(doc)
                self._chunks[pid] = chunks
                self._retrieval.index_chunks(chunks, show_progress=True)
        return docs

    def load_validation_paper(self) -> PaperDoc:
        """Load RAG paper — call ONLY during validation phase."""
        doc = load_validation(self.config.data_dir)
        self._docs[doc.paper_id] = doc
        chunks = self._make_chunks(doc)
        self._chunks[doc.paper_id] = chunks
        self._retrieval.index_chunks(chunks, show_progress=True)
        return doc

    # ── Chunking ─────────────────────────────────────────────────────────────

    def _make_chunks(self, doc: PaperDoc,
                     strategy: Optional[str] = None,
                     size: Optional[int] = None) -> List[Chunk]:
        strat = strategy or self.config.chunk_strategy
        sz    = size    or self.config.chunk_size
        kwargs = {}
        if strat == "fixed":        kwargs = {"size": sz}
        elif strat == "sentence":   kwargs = {"size": sz}
        elif strat == "dynamic":    kwargs = {"lo": max(50, sz // 4), "hi": sz}
        elif strat == "overlapping":kwargs = {"size": sz, "overlap": self.config.chunk_overlap}
        elif strat == "heading":    kwargs = {"max_size": sz}

        return get_chunks(doc.full_text, doc.paper_id, strategy=strat, **kwargs)

    def rechunk(self, paper_id: str, strategy: str, size: int) -> List[Chunk]:
        """Reindex a paper with a different chunking config (for experiments)."""
        if paper_id not in self._docs:
            raise ValueError(f"Paper {paper_id!r} not loaded.")
        self._retrieval.reset_index()
        # Re-index ALL loaded papers with new config
        for pid, doc in self._docs.items():
            chunks = self._make_chunks(doc, strategy=strategy, size=size)
            self._chunks[pid] = chunks
            self._retrieval.index_chunks(chunks, show_progress=False)
        return self._chunks[paper_id]

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self,
              paper_id: str,
              question: str,
              top_k: Optional[int] = None,
              prompt_style: Optional[str] = None) -> PipelineResponse:
        """
        Full RAG pipeline:
          retrieve top-k chunks → LLM answer → hallucination detection
        """
        if paper_id not in self._docs:
            raise ValueError(
                f"Paper {paper_id!r} not loaded. Call load_paper('{paper_id}') first."
            )

        t0 = time.time()
        k     = top_k or self.config.top_k
        style = prompt_style or self.config.prompt_style

        # 1. Retrieve
        retrieved = self._retrieval.retrieve(question, k=k, paper_id=paper_id)
        context_chunks = [r.chunk.text for r in retrieved]
        if not context_chunks:
            context_chunks = ["No relevant context found in the paper."]

        # 2. Generate
        answer = self._llm.answer(
            question, context_chunks,
            style=style,
            max_new_tokens=self.config.max_new_tokens,
        )

        # 3. Hallucination detection
        ctx_text = " ".join(context_chunks)
        report = self._detector.detect(answer.raw_text, ctx_text)

        latency_ms = (time.time() - t0) * 1000

        return PipelineResponse(
            paper_id=paper_id,
            question=question,
            answer=answer,
            retrieved=retrieved,
            hall_report=report,
            latency_ms=latency_ms,
            config={
                "strategy": self.config.chunk_strategy,
                "size":     self.config.chunk_size,
                "top_k":    k,
                "style":    style,
                "llm":      self.config.llm_backend,
            },
        )

    def query_multi(self, question: str, paper_ids: List[str],
                    k: int = 3) -> Dict[str, PipelineResponse]:
        """Same question across multiple papers."""
        return {pid: self.query(pid, question, top_k=k) for pid in paper_ids}

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def loaded_papers(self) -> List[str]:
        return list(self._docs.keys())

    def get_chunks(self, paper_id: str) -> List[Chunk]:
        return self._chunks.get(paper_id, [])

    def get_doc(self, paper_id: str) -> PaperDoc:
        return self._docs[paper_id]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_pipeline(backend: str = "mock",
                   strategy: str = "dynamic",
                   chunk_size: int = 500,
                   top_k: int = 5,
                   use_reranker: bool = False,
                   prompt_style: str = "strict",
                   data_dir: str = "data") -> RAGPipeline:
    cfg = PipelineConfig(
        llm_backend=backend,
        chunk_strategy=strategy,
        chunk_size=chunk_size,
        top_k=top_k,
        use_reranker=use_reranker,
        prompt_style=prompt_style,
        data_dir=data_dir,
    )
    return RAGPipeline(cfg)


if __name__ == "__main__":
    pipeline = build_pipeline(backend="mock", strategy="dynamic", chunk_size=500)
    pipeline.load_paper("attention")
    pipeline.load_paper("bert")

    resp = pipeline.query("attention", "What is multi-head attention?")
    print(resp.summary())
