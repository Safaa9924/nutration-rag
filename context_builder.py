"""
context_builder.py
===================
Single responsibility: turn reranked candidates into the final context
package handed to the LLM, plus a diagnostics trail explaining every
inclusion/exclusion decision (used by retrieval_report.py).

Key changes vs. the old build_context_package:
  * rerank_score is no longer injected into the LLM-facing context text
    (it's meaningless to the model and burns tokens) — it still lives on
    selected_df for debugging/reporting.
  * Each source block now shows Source #, Page, and Chunk ID.
  * Duplicate filtering checks both chunk_id and near-duplicate text
    (Jaccard similarity over word shingles), not just exact-normalized text.
  * min_score_ratio can be a fixed float OR "auto" for an adaptive cutoff
    based on the score distribution of the candidate set.
"""

from __future__ import annotations
import re
import pandas as pd


# ==================================================
# Helpers
# ==================================================
def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _shingles(text: str, n: int = 3) -> set:
    words = _normalize_text(text).split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union else 0.0


def compute_adaptive_threshold(scores, mode="auto", fixed_ratio=0.4):
    """
    Return a minimum-score cutoff for the candidate set.

    mode="fixed"  -> max_score * fixed_ratio (old behaviour)
    mode="auto"   -> adapts to how spread out the scores are, so a query
                     with one clear winner and a long tail doesn't keep
                     weak chunks just because 0.4*max happens to be low,
                     and a query with many close-scoring chunks doesn't
                     get starved down to one result.
    """
    max_score = scores.max()
    if mode == "fixed":
        return max_score * fixed_ratio

    mean = scores.mean()
    std = scores.std() if len(scores) > 1 else 0.0
    candidate_threshold = mean - 0.5 * std
    # Clamp into a sane band relative to the top score.
    threshold = max(candidate_threshold, max_score * 0.15)
    threshold = min(threshold, max_score * 0.9)
    return threshold


def _get_page(row) -> str:
    for col in ("page", "page_number", "page_num"):
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return "N/A"


# ==================================================
# Context Package Builder
# ==================================================
def build_context_package(
    query,
    reranked_df,
    max_context_chunks=10,
    word_budget=2000,
    min_score_ratio="auto",
    near_duplicate_threshold=0.7,
):
    """
    Build the final context package passed to the LLM, plus a diagnostics
    dict explaining what was kept/dropped and why.
    """
    diagnostics = {
        "num_candidates": int(len(reranked_df)),
        "removed_low_score": [],
        "removed_duplicates": [],
        "selected": [],
        "threshold_used": None,
        "threshold_mode": min_score_ratio,
    }

    candidates = (
        reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    )

    if candidates.empty:
        return {
            "query": query,
            "selected_df": pd.DataFrame(),
            "context_text": "",
            "num_sources": 0,
            "used_words": 0,
            "diagnostics": diagnostics,
        }

    if isinstance(min_score_ratio, str) and min_score_ratio == "auto":
        threshold = compute_adaptive_threshold(candidates["rerank_score"], mode="auto")
    else:
        threshold = compute_adaptive_threshold(
            candidates["rerank_score"], mode="fixed", fixed_ratio=min_score_ratio
        )
    diagnostics["threshold_used"] = float(threshold)

    selected_rows = []
    seen_chunk_ids = set()
    seen_texts = []  # list of (chunk_id, normalized_text) already accepted
    used_words = 0

    for _, row in candidates.iterrows():
        chunk_id = row.get("chunk_id", None)
        text = row["chunk_text"]
        normalized = _normalize_text(text)

        # --- low-score filter ---
        if row["rerank_score"] < threshold:
            diagnostics["removed_low_score"].append({
                "chunk_id": chunk_id, "rerank_score": float(row["rerank_score"]),
            })
            continue

        # --- exact chunk_id duplicate ---
        if chunk_id is not None and chunk_id in seen_chunk_ids:
            diagnostics["removed_duplicates"].append({
                "chunk_id": chunk_id, "reason": "duplicate_chunk_id",
            })
            continue

        # --- near-duplicate text (Jaccard shingle similarity) ---
        is_near_dup = False
        for _, prev_text in seen_texts:
            if _jaccard_similarity(normalized, prev_text) >= near_duplicate_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            diagnostics["removed_duplicates"].append({
                "chunk_id": chunk_id, "reason": "near_duplicate_text",
            })
            continue

        chunk_words = len(text.split())

        # --- word budget ---
        if used_words + chunk_words > word_budget:
            remaining_words = word_budget - used_words
            if remaining_words > 50:
                row = row.copy()
                row["chunk_text"] = " ".join(text.split()[:remaining_words])
                selected_rows.append(row)
                used_words += remaining_words
                diagnostics["selected"].append({
                    "chunk_id": chunk_id, "truncated": True,
                })
            break

        selected_rows.append(row)
        if chunk_id is not None:
            seen_chunk_ids.add(chunk_id)
        seen_texts.append((chunk_id, normalized))
        used_words += chunk_words
        diagnostics["selected"].append({"chunk_id": chunk_id, "truncated": False})

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_blocks = []
    for i, row in selected_df.iterrows():
        page = _get_page(row)
        chunk_id = row.get("chunk_id", "N/A")
        context_blocks.append(
            f"Source {i + 1}\n"
            f"Page {page}\n"
            f"Chunk {chunk_id}\n\n"
            f"{row['chunk_text']}"
        )

    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": "\n\n".join(context_blocks),
        "num_sources": len(selected_df),
        "used_words": used_words,
        "diagnostics": diagnostics,
    }
