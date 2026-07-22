"""
06_retrieve_context.py
=======================
Stage 6 — Retrieval, Query Building, Cross-Encoder Reranking, Context Package

Production‑grade refactoring:
- Uses user question + condition + intent + keywords for retrieval.
- Integrates intent detection.
- Expands QUERY_MAP to condition × intent.
- Removes internal scores from context, adds chunk metadata.
- Filters duplicates by chunk_id and semantic similarity.
- Adaptive rerank score threshold.
- Generates retrieval diagnostics.
- Separates concerns into dedicated modules.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from importlib.machinery import SourceFileLoader

# Load existing modules (unchanged)
_vec = SourceFileLoader("stage4_vector_representation", "04_vector_representation.py").load_module()
_store = SourceFileLoader("stage5_create_chroma_store", "05_create_chroma_store.py").load_module()

simple_tokenize = _vec.simple_tokenize
min_max_normalize = _vec.min_max_normalize


# =============================================================================
# 1.  QUERY BUILDER (includes intent detection and keyword expansion)
# =============================================================================

# Rich condition × intent keyword map
# Each intent gets a list of domain‑specific keywords.
QUERY_MAP = {
    "Type 2 Diabetes": {
        "avoid": [
            "foods to avoid", "restrict", "limit", "high sugar", "refined carbs",
            "processed foods", "saturated fat", "sodium", "alcohol"
        ],
        "recommend": [
            "recommended foods", "healthy diet", "whole grains", "lean protein",
            "vegetables", "fruit", "high fibre", "omega‑3", "healthy fats"
        ],
        "meal_plan": [
            "meal planning", "balanced meals", "breakfast", "lunch", "dinner",
            "snacks", "portion control", "carb distribution"
        ],
        "exercise": [
            "physical activity", "exercise", "walking", "resistance training",
            "aerobic", "glycaemic control and exercise"
        ],
        "complications": [
            "complications", "kidney disease", "hypertension", "neuropathy",
            "retinopathy", "cardiovascular risk"
        ],
        "general": [  # fallback
            "dietary management", "nutrition therapy", "glycaemic control",
            "carbohydrate intake", "glycaemic index", "fibre intake",
            "weight management"
        ]
    },
    "Type 1 Diabetes": {
        "avoid": [
            "foods to avoid", "high sugar", "skipping meals", "alcohol",
            "unplanned snacks"
        ],
        "recommend": [
            "recommended foods", "carbohydrate counting", "consistent carbs",
            "protein", "healthy fats", "vegetables"
        ],
        "meal_plan": [
            "meal planning", "carb counting", "insulin adjustment", "meals",
            "snacks", "breakfast", "lunch", "dinner"
        ],
        "exercise": [
            "exercise", "physical activity", "insulin and exercise",
            "glycaemic response"
        ],
        "complications": [
            "hypoglycaemia", "hyperglycaemia", "ketoacidosis", "long‑term complications"
        ],
        "general": [
            "nutrition therapy", "carbohydrate counting", "glycaemic control",
            "insulin and diet"
        ]
    },
    "Pre-Diabetes": {
        "avoid": [
            "foods to avoid", "high sugar", "processed foods", "sedentary lifestyle"
        ],
        "recommend": [
            "healthy eating", "weight management", "physical activity",
            "whole foods", "portion control"
        ],
        "meal_plan": [
            "meal planning", "balanced meals", "lifestyle modification"
        ],
        "exercise": [
            "exercise", "physical activity", "weight loss", "activity guidelines"
        ],
        "complications": [
            "progression to diabetes", "cardiovascular risk", "metabolic syndrome"
        ],
        "general": [
            "healthy eating", "weight management", "lifestyle modification",
            "physical activity", "glycaemic control"
        ]
    },
    "Gestational Diabetes": {
        "avoid": [
            "foods to avoid during pregnancy", "high sugar", "excessive weight gain",
            "processed carbs"
        ],
        "recommend": [
            "healthy pregnancy diet", "protein", "vegetables", "whole grains",
            "calcium", "iron"
        ],
        "meal_plan": [
            "meal planning", "balanced meals", "snacks", "carb distribution",
            "glycaemic control"
        ],
        "exercise": [
            "exercise during pregnancy", "physical activity", "safe activities"
        ],
        "complications": [
            "macrosomia", "pre‑eclampsia", "future diabetes risk"
        ],
        "general": [
            "nutrition therapy", "meal planning", "glycaemic control",
            "weight management", "healthy pregnancy diet"
        ]
    }
}

# Default fallback keywords for unknown intents
DEFAULT_INTENT_KEYWORDS = ["nutrition", "diet", "management", "glycaemic control"]


def detect_intent(question: str) -> str:
    """
    Placeholder for real intent detection.
    In production, this should call the existing detect_intent() function.
    Here we implement a simple keyword‑based fallback for demonstration.
    """
    # If a real function exists elsewhere, import and use it:
    # from intent_detector import detect_intent as real_detect
    # return real_detect(question)
    
    q_lower = question.lower()
    if any(w in q_lower for w in ["avoid", "limit", "restrict", "not eat", "harmful"]):
        return "avoid"
    if any(w in q_lower for w in ["recommend", "good", "healthy", "best", "should eat"]):
        return "recommend"
    if any(w in q_lower for w in ["meal", "breakfast", "lunch", "dinner", "snack", "plan"]):
        return "meal_plan"
    if any(w in q_lower for w in ["exercise", "activity", "walk", "gym", "workout"]):
        return "exercise"
    if any(w in q_lower for w in ["complication", "risk", "kidney", "hypertension", "nerve", "eye"]):
        return "complications"
    return "general"


def get_intent_keywords(condition: str, intent: str) -> List[str]:
    """Retrieve keywords for a given condition and intent."""
    condition_map = QUERY_MAP.get(condition, {})
    return condition_map.get(intent, DEFAULT_INTENT_KEYWORDS)


def build_query(
    user_question: str,
    condition: str,
    intent: Optional[str] = None,
    extra_keywords: Optional[List[str]] = None
) -> str:
    """
    Construct a rich retrieval query from:
      - the original user question
      - the detected condition
      - the detected intent (or fallback to 'general')
      - additional domain keywords (from QUERY_MAP)
      - any extra keywords passed in

    Returns a single string that will be used for all retrievers.
    """
    if intent is None:
        intent = detect_intent(user_question)
    
    # Get keywords for this condition and intent
    keywords = get_intent_keywords(condition, intent)
    
    # Combine components
    parts = [user_question, condition] + keywords
    if extra_keywords:
        parts.extend(extra_keywords)
    
    # Remove duplicates and empty strings
    unique_parts = []
    seen = set()
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            unique_parts.append(p)
            seen.add(p)
    
    return " ".join(unique_parts)


# =============================================================================
# 2.  RETRIEVAL FUNCTIONS (unchanged interface, but now use the rich query)
# =============================================================================

def retrieve_top_k_tfidf(query, tfidf_vectorizer, tfidf_matrix, chunks_df, k=40):
    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"
    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_bm25(query, bm25, chunks_df, k=40):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = np.array(scores)[ranking]
    results["retriever"] = "BM25"
    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


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


# =============================================================================
# 3.  RERANKER (unchanged)
# =============================================================================

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


# =============================================================================
# 4.  CONTEXT BUILDER (cleaned, with metadata and duplicate handling)
# =============================================================================

def _filter_duplicates(
    candidates: pd.DataFrame,
    chunk_id_col: str = "chunk_id",
    text_col: str = "chunk_text",
    semantic_threshold: float = 0.85,
    embedding_model=None  # if provided, use semantic similarity
) -> pd.DataFrame:
    """
    Remove duplicate chunks:
      - exact duplicate by chunk_id (primary key)
      - (optional) near‑duplicates by semantic similarity of the text
    """
    # 1. Remove exact duplicates by chunk_id
    unique_ids = candidates[chunk_id_col].unique()
    df_unique = candidates.drop_duplicates(subset=[chunk_id_col]).copy()
    
    # 2. Semantic near‑duplicate filtering (if model provided)
    if embedding_model is not None and len(df_unique) > 1:
        texts = df_unique[text_col].tolist()
        emb = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = cosine_similarity(emb)
        # Greedy selection: keep first occurrence, drop those with similarity > threshold to any kept
        kept_indices = []
        for i in range(len(df_unique)):
            if not kept_indices:
                kept_indices.append(i)
            else:
                max_sim = max(sim_matrix[i][j] for j in kept_indices)
                if max_sim < semantic_threshold:
                    kept_indices.append(i)
        df_unique = df_unique.iloc[kept_indices].reset_index(drop=True)
    
    return df_unique


def build_context_package(
    query: str,
    reranked_df: pd.DataFrame,
    max_context_chunks: int = 10,
    word_budget: int = 2000,
    min_score_ratio: float = None,   # if None, adaptive
    duplicate_threshold: float = 0.85,
    embedding_model=None,            # for semantic duplicate filtering
    chunk_metadata: Optional[pd.DataFrame] = None  # contains page, etc.
) -> Dict[str, Any]:
    """
    Build the final context package for the LLM.
    Removes internal scores, formats source metadata (source number, page, chunk_id),
    filters duplicates, and applies adaptive threshold if min_score_ratio is None.
    """
    candidates = reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    
    if candidates.empty:
        return {
            "query": query,
            "selected_df": pd.DataFrame(),
            "context_text": "",
            "num_sources": 0,
            "used_words": 0,
            "diagnostics": {"reason": "No candidates after reranking"}
        }
    
    # ---------- Adaptive threshold ----------
    if min_score_ratio is None:
        # Use the 50th percentile of rerank scores as adaptive baseline
        baseline = candidates["rerank_score"].quantile(0.5)
        # Avoid threshold = 0 for all positive scores
        if baseline > 0:
            min_score_ratio = baseline / candidates["rerank_score"].max()
        else:
            min_score_ratio = 0.3  # fallback
    # Ensure ratio is between 0 and 1
    min_score_ratio = max(0.0, min(1.0, min_score_ratio))
    
    max_score = candidates["rerank_score"].max()
    score_threshold = max_score * min_score_ratio
    
    # ---------- Filter by score ----------
    high_score = candidates[candidates["rerank_score"] >= score_threshold].copy()
    if high_score.empty:
        # Fallback: keep top 3 if none pass
        high_score = candidates.head(3).copy()
    
    # ---------- Remove duplicates ----------
    deduped = _filter_duplicates(
        high_score,
        chunk_id_col="chunk_id",
        text_col="chunk_text",
        semantic_threshold=duplicate_threshold,
        embedding_model=embedding_model
    )
    
    # ---------- Word budget selection ----------
    selected_rows = []
    seen_chunk_ids = set()
    used_words = 0
    
    for _, row in deduped.iterrows():
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunk_ids:
            continue
        text = row["chunk_text"]
        chunk_words = len(text.split())
        
        if used_words + chunk_words > word_budget:
            remaining = word_budget - used_words
            if remaining > 50:
                row = row.copy()
                row["chunk_text"] = " ".join(text.split()[:remaining])
                selected_rows.append(row)
                used_words += remaining
            break
        
        selected_rows.append(row)
        seen_chunk_ids.add(chunk_id)
        used_words += chunk_words
        
        if len(selected_rows) >= max_context_chunks:
            break
    
    selected_df = pd.DataFrame(selected_rows)
    
    # ---------- Build context text with metadata ----------
    context_blocks = []
    for i, row in selected_df.iterrows():
        # Extract page number if available (from metadata or from chunk_id convention)
        # Example: if chunk_id contains "p142", parse it.
        page = "Unknown"
        if chunk_metadata is not None and "chunk_id" in chunk_metadata.columns and "page" in chunk_metadata.columns:
            meta = chunk_metadata[chunk_metadata["chunk_id"] == row["chunk_id"]]
            if not meta.empty:
                page = meta.iloc[0]["page"]
        else:
            # Fallback: try to extract page from chunk_id (e.g., "chunk_0154_p142")
            match = re.search(r"p(\d+)", row["chunk_id"])
            if match:
                page = match.group(1)
        
        block = (
            f"Source {i+1}\n"
            f"Page {page}\n"
            f"Chunk {row['chunk_id']}\n\n"
            f"{row['chunk_text']}"
        )
        context_blocks.append(block)
    
    context_text = "\n\n".join(context_blocks)
    
    # ---------- Diagnostics ----------
    diagnostics = {
        "original_candidates": len(reranked_df),
        "score_threshold": score_threshold,
        "min_score_ratio": min_score_ratio,
        "after_score_filter": len(high_score),
        "duplicates_removed": len(high_score) - len(deduped),
        "final_selected": len(selected_df),
        "used_words": used_words
    }
    
    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": context_text,
        "num_sources": len(selected_df),
        "used_words": used_words,
        "diagnostics": diagnostics
    }


# =============================================================================
# 5.  MAIN PIPELINE (with diagnostics and report)
# =============================================================================

def run_retrieval_pipeline(
    user_question: str,
    predicted_condition: str,
    chunks_df: pd.DataFrame,
    tfidf_vectorizer,
    tfidf_matrix,
    bm25,
    embedding_model,
    embedding_matrix,
    hybrid_k: int = 40,
    rerank_top_n: int = 10,
    context_max_chunks: int = 10,
    context_word_budget: int = 1200,
    return_diagnostics: bool = True
) -> Dict[str, Any]:
    """
    Full retrieval pipeline:
      1. Build rich query (user question + condition + intent + keywords)
      2. Hybrid retrieval
      3. Cross‑encoder reranking
      4. Context package construction
      5. Return final context and (optionally) diagnostic report
    """
    # 1. Build query
    intent = detect_intent(user_question)
    query = build_query(user_question, predicted_condition, intent)
    
    # 2. Hybrid retrieval
    candidates = retrieve_top_k_hybrid(
        query=query,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model,
        embedding_matrix=embedding_matrix,
        chunks_df=chunks_df,
        k=hybrid_k
    )
    
    # 3. Rerank
    reranked = rerank_candidates(query=query, candidates_df=candidates, top_n=rerank_top_n)
    
    # 4. Build context (with duplicate filtering using embedding_model)
    context_pkg = build_context_package(
        query=query,
        reranked_df=reranked,
        max_context_chunks=context_max_chunks,
        word_budget=context_word_budget,
        min_score_ratio=None,        # adaptive
        duplicate_threshold=0.85,
        embedding_model=embedding_model,
        chunk_metadata=None          # if you have page info, pass it here
    )
    
    # 5. Combine all diagnostics
    pipeline_report = {
        "user_question": user_question,
        "predicted_condition": predicted_condition,
        "detected_intent": intent,
        "retrieval_query": query,
        "retrieved_candidates": len(candidates),
        "reranked_candidates": len(reranked),
        "context": context_pkg
    }
    if not return_diagnostics:
        # remove diagnostics from context_pkg if not needed
        context_pkg.pop("diagnostics", None)
        pipeline_report["context"] = context_pkg
    
    return pipeline_report


# =============================================================================
# 6.  EXAMPLE USAGE (updated)
# =============================================================================

if __name__ == "__main__":
    # Load data and indexes
    chunks_df = pd.read_csv("semantic_chunks_final.csv", encoding="utf-8-sig")
    tfidf_vectorizer, tfidf_matrix, bm25, embedding_model, embedding_matrix = _store.load_indexes()
    
    # Example user input
    user_question = "What foods should a Type 1 diabetic avoid?"
    predicted_condition = "Type 1 Diabetes"
    
    # Run pipeline
    result = run_retrieval_pipeline(
        user_question=user_question,
        predicted_condition=predicted_condition,
        chunks_df=chunks_df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        bm25=bm25,
        embedding_model=embedding_model,
        embedding_matrix=embedding_matrix,
        hybrid_k=40,
        rerank_top_n=10,
        context_max_chunks=10,
        context_word_budget=1200,
        return_diagnostics=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RETRIEVAL PIPELINE REPORT")
    print("=" * 60)
    print(f"User question: {result['user_question']}")
    print(f"Condition: {result['predicted_condition']}")
    print(f"Detected intent: {result['detected_intent']}")
    print(f"Retrieval query: {result['retrieval_query']}")
    print(f"Retrieved candidates: {result['retrieved_candidates']}")
    print(f"Reranked candidates: {result['reranked_candidates']}")
    print(f"Final sources: {result['context']['num_sources']}")
    print(f"Used words: {result['context']['used_words']}")
    print("\nDIAGNOSTICS:")
    for k, v in result['context']['diagnostics'].items():
        print(f"  {k}: {v}")
    
    print("\nCONTEXT PREVIEW (first 500 chars):")
    print(result['context']['context_text'][:500] + "...")
