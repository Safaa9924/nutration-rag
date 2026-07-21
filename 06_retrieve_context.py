"""
06_retrieve_context.py
=======================
Stage 6 — Retrieval, Query Building, Cross-Encoder Reranking, Context Package

Source: notebook cells 42, 56, 59, 62

Loads the persisted indexes (from 05_create_chroma_store.py) and provides
the full retrieval -> rerank -> context-building path used before prompting.
"""

import re
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

from importlib.machinery import SourceFileLoader

_vec = SourceFileLoader("stage4_vector_representation", "04_vector_representation.py").load_module()
_store = SourceFileLoader("stage5_create_chroma_store", "05_create_chroma_store.py").load_module()

simple_tokenize = _vec.simple_tokenize
min_max_normalize = _vec.min_max_normalize


# ==================================================
# TF-IDF Retrieval
# ==================================================

def retrieve_top_k_tfidf(query, tfidf_vectorizer, tfidf_matrix, chunks_df, k=40):

    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


# ==================================================
# BM25 Retrieval
# ==================================================

def retrieve_top_k_bm25(query, bm25, chunks_df, k=40):

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = np.array(scores)[ranking]
    results["retriever"] = "BM25"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


# ==================================================
# Semantic Retrieval
# ==================================================

def retrieve_top_k_semantic(query, embedding_model, embedding_matrix, chunks_df, k=40):

    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embedding_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


# ==================================================
# Hybrid Retrieval
# ==================================================

def retrieve_top_k_hybrid(
    query,
    tfidf_vectorizer, tfidf_matrix,
    bm25,
    embedding_model, embedding_matrix,
    chunks_df,
    tfidf_weight=0.34,
    bm25_weight=0.33,
    semantic_weight=0.33,
    k=40
):

    q_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = min_max_normalize(cosine_similarity(q_vec, tfidf_matrix).flatten())

    bm25_scores = min_max_normalize(bm25.get_scores(simple_tokenize(query)))

    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    semantic_scores = min_max_normalize(
        cosine_similarity(query_embedding, embedding_matrix).flatten()
    )

    combined = (
        tfidf_weight * tfidf_scores
        + bm25_weight * bm25_scores
        + semantic_weight * semantic_scores
    )

    ranking = np.argsort(combined)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["tfidf_score"] = tfidf_scores[ranking]
    results["bm25_score"] = bm25_scores[ranking]
    results["semantic_score"] = semantic_scores[ranking]
    results["score"] = combined[ranking]
    results["retriever"] = "Hybrid"

    return results[
        ["retriever", "chunk_id", "tfidf_score", "bm25_score", "semantic_score", "score", "chunk_text"]
    ].reset_index(drop=True)


# ==================================================
# Query Builder (condition -> expanded query)
# ==================================================

QUERY_MAP = {
    "Type 2 Diabetes": [
        "dietary management", "nutrition therapy", "meal planning",
        "carbohydrate intake", "glycaemic control", "glycaemic index",
        "fibre intake", "dietary fat", "weight management"
    ],
    "Type 1 Diabetes": [
        "nutrition therapy", "carbohydrate counting", "meal planning",
        "glycaemic control", "hypoglycaemia", "insulin and diet"
    ],
    "Pre-Diabetes": [
        "healthy eating", "weight management", "lifestyle modification",
        "physical activity", "glycaemic control"
    ],
    "Gestational Diabetes": [
        "nutrition therapy", "meal planning", "glycaemic control",
        "weight management", "healthy pregnancy diet"
    ]
}


def build_query(prediction):
    keywords = QUERY_MAP.get(prediction, [])
    query = prediction
    if keywords:
        query += " " + " ".join(keywords)
    return query


# ==================================================
# Cross-Encoder Reranking
# ==================================================

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("=" * 60)
        print("LOADING CROSS ENCODER")
        print("=" * 60)
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
    return _reranker


def rerank_candidates(query, candidates_df, top_n=10):

    reranker = get_reranker()

    pairs = [(query, text) for text in candidates_df["chunk_text"]]

    scores = reranker.predict(pairs)

    reranked = candidates_df.copy()
    reranked["rerank_score"] = scores

    reranked = (
        reranked
        .sort_values("rerank_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print("=" * 60)
    print("RERANKING SUMMARY")
    print("=" * 60)
    print(f"Candidates : {len(candidates_df)}")
    print(f"Selected   : {len(reranked)}")

    return reranked


# ==================================================
# Build Context Package
# ==================================================

def build_context_package(
    query,
    reranked_df,
    max_context_chunks=10,
    word_budget=2000,
    min_score_ratio=0.4
):
    """
    Build the final context package passed to the LLM.
    """

    candidates = (
        reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    )

    if candidates.empty:
        return {
            "query": query,
            "selected_df": pd.DataFrame(),
            "context_text": "",
            "num_sources": 0,
            "used_words": 0
        }

    max_score = candidates["rerank_score"].max()

    selected_rows = []
    seen_texts = set()
    used_words = 0

    for _, row in candidates.iterrows():

        if row["rerank_score"] < max_score * min_score_ratio:
            continue

        text = row["chunk_text"]

        normalized = re.sub(r"\s+", " ", text).strip().lower()

        if normalized in seen_texts:
            continue

        chunk_words = len(text.split())

        if used_words + chunk_words > word_budget:

            remaining_words = word_budget - used_words

            if remaining_words > 50:
                row = row.copy()
                row["chunk_text"] = " ".join(text.split()[:remaining_words])
                selected_rows.append(row)
                used_words += remaining_words

            break

        selected_rows.append(row)
        seen_texts.add(normalized)
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_blocks = []

    for i, row in selected_df.iterrows():
        context_blocks.append(
            f"[Source {i+1}] rerank_score={row['rerank_score']:.3f}\n{row['chunk_text']}"
        )

    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": "\n\n".join(context_blocks),
        "num_sources": len(selected_df),
        "used_words": used_words
    }


# ==================================================
# Run (example)
# ==================================================

if __name__ == "__main__":

    chunks_df = pd.read_csv("semantic_chunks_final.csv", encoding="utf-8-sig")

    tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix = _store.load_indexes()

    prediction = "Type 2 Diabetes"
    query = build_query(prediction)

    results = retrieve_top_k_hybrid(
        query=query,
        tfidf_vectorizer=tfidf_vectorizer, tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model, embedding_matrix=embedding_matrix,
        chunks_df=chunks_df,
        tfidf_weight=0.34, bm25_weight=0.33, semantic_weight=0.33,
        k=40
    )

    print("Retrieved Candidates:", len(results))

    reranked = rerank_candidates(query=query, candidates_df=results, top_n=10)

    context = build_context_package(query=query, reranked_df=reranked, max_context_chunks=10, word_budget=1200)

    print("Sources Used:", context["num_sources"])
    print("Words Used:", context["used_words"])
