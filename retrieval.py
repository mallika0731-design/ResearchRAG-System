"""
retrieval.py
============
Embedding generation (sentence-transformers all-MiniLM-L6-v2),
FAISS flat inner-product index, and optional cross-encoder reranking.

Design choices
--------------
* all-MiniLM-L6-v2  →  384-d, L2-normalised → cosine via inner product
* IndexFlatIP       →  exact search, no approximation errors
* Source filtering  →  retrieve only from the queried paper
* Cross-encoder     →  ms-marco-MiniLM-L-6-v2 for precision boost
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    chunk: object          # Chunk from chunking.py
    bi_score: float        # FAISS cosine score
    rerank_score: Optional[float] = None
    rank: int = 0

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.bi_score


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Wraps SentenceTransformer with L2-normalised output."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.dim}")

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class VectorIndex:
    """
    Maintains a FAISS IndexFlatIP (cosine on normalised vectors).
    Supports per-paper source filtering.
    """

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[object] = []          # parallel array to FAISS positions
        self._source_map: List[str] = []         # paper_id for each position

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, embeddings: np.ndarray, chunks: List[object]) -> None:
        assert embeddings.shape[0] == len(chunks), "embedding/chunk count mismatch"
        self._index.add(embeddings)
        for chunk in chunks:
            self._chunks.append(chunk)
            self._source_map.append(chunk.paper_id)
        logger.info(f"VectorIndex: {self.size} vectors total")

    def search(self, query_vec: np.ndarray, k: int,
               source_filter: Optional[str] = None) -> List[Tuple[float, object]]:
        """Return up to k (score, chunk) pairs, optionally filtered by paper."""
        fetch_k = k if source_filter is None else min(k * 12, self.size)
        fetch_k = max(fetch_k, 1)
        qv = query_vec.reshape(1, -1).astype(np.float32)
        scores, ids = self._index.search(qv, min(fetch_k, self.size))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if source_filter and self._source_map[idx] != source_filter:
                continue
            results.append((float(score), self._chunks[idx]))
            if len(results) >= k:
                break
        return results

    def reset(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self.dim)
        self._chunks.clear()
        self._source_map.clear()

    def save(self, path: str) -> None:
        import faiss
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p) + ".faiss")
        with open(str(p) + ".meta.pkl", "wb") as f:
            pickle.dump((self._chunks, self._source_map), f)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorIndex":
        import faiss
        obj = cls.__new__(cls)
        obj.dim = dim
        obj._index = faiss.read_index(str(path) + ".faiss")
        with open(str(path) + ".meta.pkl", "rb") as f:
            obj._chunks, obj._source_map = pickle.load(f)
        return obj


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional)
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult],
               top_n: Optional[int] = None) -> List[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
        results.sort(key=lambda x: x.rerank_score, reverse=True)  # type: ignore
        if top_n:
            results = results[:top_n]
        for i, r in enumerate(results):
            r.rank = i + 1
        return results


# ---------------------------------------------------------------------------
# High-level retrieval pipeline
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """
    Public interface used by pipeline.py.

    Usage
    -----
    rp = RetrievalPipeline(use_reranker=False)
    rp.index_chunks(chunks)
    results = rp.retrieve("How does attention work?", paper_id="attention", k=5)
    """

    def __init__(self, embed_model: str = EMBED_MODEL,
                 use_reranker: bool = False,
                 device: str = "cpu"):
        self.embedder = Embedder(embed_model, device=device)
        self._index: Optional[VectorIndex] = None
        self._reranker: Optional[Reranker] = None
        if use_reranker:
            try:
                self._reranker = Reranker()
            except Exception as e:
                logger.warning(f"Reranker unavailable: {e}. Proceeding without.")

    def index_chunks(self, chunks: List[object],
                     batch_size: int = 128, show_progress: bool = True) -> None:
        """Embed and index all chunks. Can be called multiple times to add papers."""
        if self._index is None:
            self._index = VectorIndex(dim=self.embedder.dim)
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks ...")
        vecs = self.embedder.encode(texts, batch_size=batch_size,
                                    show_progress=show_progress)
        self._index.add(vecs, chunks)

    def retrieve(self, query: str, k: int = 5,
                 paper_id: Optional[str] = None) -> List[SearchResult]:
        """Retrieve top-k chunks for a query, optionally from one paper only."""
        if self._index is None or self._index.size == 0:
            raise RuntimeError("Index is empty. Call index_chunks() first.")
        q_vec = self.embedder.encode_one(query)
        raw = self._index.search(q_vec, k=k, source_filter=paper_id)
        results = [
            SearchResult(chunk=chunk, bi_score=score, rank=i + 1)
            for i, (score, chunk) in enumerate(raw)
        ]
        if self._reranker and results:
            results = self._reranker.rerank(query, results, top_n=k)
        return results

    def retrieve_multi(self, query: str, paper_ids: List[str],
                       k_per_paper: int = 3) -> dict:
        """Retrieve top-k chunks separately for each paper."""
        return {
            pid: self.retrieve(query, k=k_per_paper, paper_id=pid)
            for pid in paper_ids
        }

    # ── Metric helpers ────────────────────────────────────────────────────
    def context_relevance(self, query: str,
                          results: List[SearchResult]) -> float:
        """Average cosine sim between query and retrieved chunks."""
        if not results:
            return 0.0
        q_vec = self.embedder.encode_one(query)
        scores = []
        for r in results:
            c_vec = self.embedder.encode_one(r.chunk.text)
            scores.append(float(np.dot(q_vec, c_vec)))
        return float(np.mean(scores))

    def reset_index(self) -> None:
        if self._index:
            self._index.reset()


# ---------------------------------------------------------------------------
# Standalone metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    top = retrieved_ids[:k]
    return len(set(top) & relevant_ids) / max(len(relevant_ids), 1)


def precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    top = retrieved_ids[:k]
    return sum(1 for r in top if r in relevant_ids) / max(k, 1)


if __name__ == "__main__":
    from chunking import chunk_dynamic
    sample = ("The Transformer uses self-attention. " * 30)
    chunks = chunk_dynamic(sample, "test")
    rp = RetrievalPipeline(use_reranker=False)
    rp.index_chunks(chunks)
    results = rp.retrieve("How does attention work?", k=3)
    for r in results:
        print(f"  rank={r.rank}  score={r.bi_score:.4f}  tokens={r.chunk.token_count}")
