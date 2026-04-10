"""
experiments.py
==============
Systematic ablation experiments over:
  • chunk sizes   : 200, 500, 800 tokens
  • top-k         : 2, 3, 5
  • prompt styles : strict, open

Phase 1 (Train/Test) — Attention + BERT papers only.
Phase 2 (Validation) — RAG paper only (unseen during tuning).
"""

import logging
import time
from typing import List, Optional

import pandas as pd

from pipeline  import RAGPipeline, PipelineConfig, build_pipeline
from evaluation import Evaluator
from chunking  import compare_strategies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------

TRAIN_QUERIES = {
    "attention": [
        "What is the core innovation of the Transformer architecture?",
        "How does scaled dot-product attention work?",
        "What are the limitations of the Transformer model?",
        "How does multi-head attention differ from single-head attention?",
    ],
    "bert": [
        "What pre-training objectives does BERT use?",
        "How does BERT handle fine-tuning for downstream tasks?",
        "What is the difference between BERT-base and BERT-large?",
        "What are the main limitations of BERT?",
    ],
}

VALIDATION_QUERIES = {
    "rag": [
        "How does RAG combine retrieval with generation?",
        "What retrieval mechanism does RAG use?",
        "What are the main limitations of the RAG approach?",
        "How does RAG differ from standard seq2seq models?",
        "What benchmarks does RAG improve upon?",
    ],
}


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs all ablation experiments and collects results.

    Usage
    -----
    runner = ExperimentRunner(backend="mock", data_dir="data")
    runner.run_train_test()          # Attention + BERT
    runner.run_validation()          # RAG paper (call after train/test)
    df = runner.results_df()
    """

    CHUNK_SIZES   = [200, 500, 800]
    TOP_K_VALUES  = [2, 3, 5]
    PROMPT_STYLES = ["strict", "open"]
    STRATEGIES    = ["dynamic", "sentence", "overlapping"]  # key strategies

    def __init__(self, backend: str = "mock", data_dir: str = "data",
                 use_reranker: bool = False):
        self.backend      = backend
        self.data_dir     = data_dir
        self.use_reranker = use_reranker
        self._records: List[dict] = []

    # ── Phase 1: Train/Test ──────────────────────────────────────────────────

    def run_train_test(self, verbose: bool = True) -> pd.DataFrame:
        """
        Experiment 1 — chunk sizes (200/500/800) with best strategy.
        Experiment 2 — top-k values (2/3/5).
        Experiment 3 — prompt styles (strict/open).
        """
        print("=" * 60)
        print("PHASE 1: Train/Test (Attention + BERT)")
        print("=" * 60)

        # Build base pipeline and load papers once
        base = build_pipeline(backend=self.backend, data_dir=self.data_dir,
                              use_reranker=self.use_reranker)
        base.load_paper("attention")
        base.load_paper("bert")
        embedder = base._retrieval.embedder

        # ── EXP 1: Chunk sizes ──────────────────────────────────────────────
        print("\n[EXP 1] Chunk size comparison: 200 / 500 / 800 tokens")
        for size in self.CHUNK_SIZES:
            self._run_batch(
                paper_ids=["attention", "bert"],
                strategy="dynamic",
                chunk_size=size,
                top_k=5,
                prompt_style="strict",
                exp_name="chunk_size",
                verbose=verbose,
            )

        # ── EXP 2: Top-k ────────────────────────────────────────────────────
        print("\n[EXP 2] Top-k retrieval: k = 2, 3, 5")
        for k in self.TOP_K_VALUES:
            self._run_batch(
                paper_ids=["attention", "bert"],
                strategy="dynamic",
                chunk_size=500,
                top_k=k,
                prompt_style="strict",
                exp_name="top_k",
                verbose=verbose,
            )

        # ── EXP 3: Prompt styles ─────────────────────────────────────────────
        print("\n[EXP 3] Prompt style: strict vs open")
        for style in self.PROMPT_STYLES:
            self._run_batch(
                paper_ids=["attention", "bert"],
                strategy="dynamic",
                chunk_size=500,
                top_k=5,
                prompt_style=style,
                exp_name="prompt_style",
                verbose=verbose,
            )

        df = self.results_df()
        print(f"\nTrain/Test complete — {len(df)} records collected.")
        return df

    # ── Phase 2: Validation ──────────────────────────────────────────────────

    def run_validation(self, verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate best config (dynamic/500/k=5/strict) on unseen RAG paper.
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Validation — RAG paper (UNSEEN)")
        print("=" * 60)

        self._run_batch(
            paper_ids=["rag"],
            strategy="dynamic",
            chunk_size=500,
            top_k=5,
            prompt_style="strict",
            exp_name="validation",
            verbose=verbose,
        )

        val_df = self.results_df()
        val_df = val_df[val_df["exp_name"] == "validation"]
        print(f"\nValidation complete — {len(val_df)} records.")
        return val_df

    # ── Internal runner ──────────────────────────────────────────────────────

    def _run_batch(self, paper_ids: List[str], strategy: str,
                   chunk_size: int, top_k: int, prompt_style: str,
                   exp_name: str, verbose: bool = True) -> None:
        """Build a fresh pipeline, load papers, run queries, collect metrics."""
        cfg = PipelineConfig(
            llm_backend=self.backend,
            chunk_strategy=strategy,
            chunk_size=chunk_size,
            top_k=top_k,
            use_reranker=self.use_reranker,
            prompt_style=prompt_style,
            data_dir=self.data_dir,
        )
        pipeline = RAGPipeline(cfg)
        evaluator = Evaluator(pipeline._retrieval.embedder)

        # Load papers
        for pid in paper_ids:
            pipeline.load_paper(pid)

        # Run queries
        query_map = VALIDATION_QUERIES if "rag" in paper_ids else TRAIN_QUERIES
        for pid in paper_ids:
            queries = query_map.get(pid, [])
            for question in queries:
                t0 = time.time()
                try:
                    response = pipeline.query(pid, question,
                                              top_k=top_k,
                                              prompt_style=prompt_style)
                    chunks = pipeline.get_chunks(pid)
                    record = evaluator.evaluate(
                        paper_id=pid,
                        question=question,
                        strategy=strategy,
                        target_size=chunk_size,
                        top_k=top_k,
                        prompt_style=prompt_style,
                        all_chunks=chunks,
                        retrieved=response.retrieved,
                        structured_answer=response.answer,
                        hall_report=response.hall_report,
                        source_text=pipeline.get_doc(pid).full_text[:5000],
                        start_time=t0,
                    )
                    row = record.flat()
                    row["exp_name"] = exp_name
                    row["latency_ms"] = response.latency_ms
                    self._records.append(row)

                    if verbose:
                        print(
                            f"  {pid:12s} sz={chunk_size:4d} k={top_k} "
                            f"style={prompt_style:7s} → "
                            f"hall={response.hall_report.score:.3f}  "
                            f"complete={response.answer.completeness():.2f}  "
                            f"{response.hall_report.verdict}"
                        )
                except Exception as e:
                    logger.error(f"[{pid}] query failed: {e}")

    # ── Results ──────────────────────────────────────────────────────────────

    def results_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records)

    def print_summary(self) -> None:
        df = self.results_df()
        if df.empty:
            print("No results yet.")
            return

        num_cols = [c for c in ["hallucination_rate", "completeness", "faithfulness",
                                 "context_relevance", "latency_ms"]
                    if c in df.columns]

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        for exp_name, grp in df.groupby("exp_name"):
            print(f"\n── {exp_name} ──")
            vary_col = {"chunk_size": "target_size", "top_k": "top_k",
                        "prompt_style": "prompt_style", "validation": "paper_id"}.get(
                exp_name, "paper_id"
            )
            if vary_col in grp.columns:
                sub = grp.groupby(vary_col)[num_cols].mean().round(3)
                print(sub.to_string())

    def chunking_comparison_table(self, text: str, paper_id: str) -> pd.DataFrame:
        """Return chunking strategy stats as a DataFrame."""
        rows = compare_strategies(text, paper_id, sizes=self.CHUNK_SIZES)
        return pd.DataFrame([r for r in rows if "error" not in r])

    def save_results(self, path: str = "experiment_results.csv") -> None:
        df = self.results_df()
        df.to_csv(path, index=False)
        print(f"Results saved → {path}  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = ExperimentRunner(backend="mock", data_dir="data")
    runner.run_train_test(verbose=True)
    runner.run_validation(verbose=True)
    runner.print_summary()
    runner.save_results()
