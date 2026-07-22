"""
06_retrieve_context.py
=======================
Stage 6 — Retrieval, Query Building, Cross-Encoder Reranking, Context Package

Source: notebook cells 42, 56, 59, 62 (refactored)

This file is now a thin orchestrator. The actual logic lives in dedicated,
single-responsibility modules alongside it:

    intent_detector.py    -> detect_intent()
    query_builder.py       -> build_query() / build_query_plan()
    retriever.py           -> retrieve_top_k_tfidf/bm25/semantic/hybrid()
    reranker.py            -> rerank_candidates()
    context_builder.py     -> build_context_package()
    retrieval_report.py    -> build_retrieval_report() / print_retrieval_report()

Everything is re-exported here so existing call sites that do
`from importlib.machinery import SourceFileLoader; ...load_module("06_retrieve_context.py")`
and pull e.g. `build_context_package` off the module keep working unchanged.

What changed and why (see code review):
  1. build_query() now takes the user's actual question, not just the
     predicted condition — the old version silently threw the question away.
  2. detect_intent() is now integrated into query building so "foods to
     avoid" vs "recommended foods" vs "meal plan" etc. produce different
     retrieval queries instead of all being treated identically.
  3. QUERY_MAP -> CONDITION_KEYWORDS (condition-level) is now paired with
     INTENT_KEYWORDS (intent-level), giving much richer, targeted expansion.
  4. Queries combine: user question + condition + intent + domain keywords,
     instead of just a handful of generic keywords.
  5. Context blocks no longer leak rerank_score to the LLM; they show
     Source #, Page, and Chunk ID instead.
  6. Context formatting is now structured (Source / Page / Chunk / text).
  7. Duplicate filtering checks chunk_id AND near-duplicate text (Jaccard
     shingle similarity), not just exact-normalized text match.
  8. min_score_ratio supports an "auto" adaptive mode in addition to a
     fixed ratio.
  9. A retrieval report (retrieval_report.py) explains what was retrieved,
     reranked, dropped (and why), and finally selected.
  10. Query building / retrieval / reranking / context building / reporting
     are now separate modules instead of one tangled file.
"""

import pandas as pd
from importlib.machinery import SourceFileLoader

# ---- Re-export everything so old import patterns keep working ----------
from intent_detector import detect_intent
from query_builder import (
    build_query, build_query_plan, QueryPlan,
    CONDITION_KEYWORDS, INTENT_KEYWORDS,
)
from retriever import (
    retrieve_top_k_tfidf, retrieve_top_k_bm25,
    retrieve_top_k_semantic, retrieve_top_k_hybrid,
)
from reranker import get_reranker, rerank_candidates
from context_builder import build_context_package, compute_adaptive_threshold
from retrieval_report import build_retrieval_report, print_retrieval_report

_store = SourceFileLoader(
    "stage5_create_chroma_store", "05_create_chroma_store.py"
).load_module()


# ==================================================
# Full pipeline, callable end-to-end
# ==================================================
def run_retrieval_pipeline(
    user_question,
    condition,
    tfidf_vectorizer, tfidf_matrix,
    bm25,
    embedding_model, embedding_matrix,
    chunks_df,
    intent=None,
    k=40,
    top_n=10,
    max_context_chunks=10,
    word_budget=1200,
    min_score_ratio="auto",
):
    """
    End-to-end: build query -> hybrid retrieve -> rerank -> build context
    -> report. Returns (context_result, report).
    """
    query_plan = build_query_plan(user_question, condition=condition, intent=intent)

    retrieved = retrieve_top_k_hybrid(
        query=query_plan.query_text,
        tfidf_vectorizer=tfidf_vectorizer, tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model, embedding_matrix=embedding_matrix,
        chunks_df=chunks_df,
        k=k,
    )

    reranked = rerank_candidates(query=query_plan.query_text, candidates_df=retrieved, top_n=top_n)

    context_result = build_context_package(
        query=query_plan.query_text,
        reranked_df=reranked,
        max_context_chunks=max_context_chunks,
        word_budget=word_budget,
        min_score_ratio=min_score_ratio,
    )

    report = build_retrieval_report(
        query_plan=query_plan,
        num_retrieved=len(retrieved),
        num_reranked=len(reranked),
        context_result=context_result,
    )

    return context_result, report


# ==================================================
# Run (example)
# ==================================================
if __name__ == "__main__":

    chunks_df = pd.read_csv("semantic_chunks_final.csv", encoding="utf-8-sig")

    tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix = _store.load_indexes()

    user_question = "What foods should a Type 1 diabetic avoid?"
    condition = "Type 1 Diabetes"

    context_result, report = run_retrieval_pipeline(
        user_question=user_question,
        condition=condition,
        tfidf_vectorizer=tfidf_vectorizer, tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model, embedding_matrix=embedding_matrix,
        chunks_df=chunks_df,
        k=40, top_n=10,
        max_context_chunks=10, word_budget=1200,
        min_score_ratio="auto",
    )

    print_retrieval_report(report)
    print()
    print(context_result["context_text"])
