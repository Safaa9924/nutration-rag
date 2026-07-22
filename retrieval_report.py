"""
retrieval_report.py
====================
Single responsibility: turn the diagnostics produced across the pipeline
(query plan, retrieval candidate count, reranking, context building) into
one readable report — for logging, debugging, or displaying "why did I
get these sources" to a developer/analyst.
"""

from __future__ import annotations


def build_retrieval_report(query_plan, num_retrieved, num_reranked, context_result) -> dict:
    diagnostics = context_result.get("diagnostics", {})

    report = {
        "user_question": query_plan.user_question,
        "condition": query_plan.condition,
        "intent": query_plan.intent,
        "condition_keywords": query_plan.condition_keywords,
        "intent_keywords": query_plan.intent_keywords,
        "final_query": query_plan.query_text,
        "num_retrieved_candidates": num_retrieved,
        "num_reranked_candidates": num_reranked,
        "threshold_mode": diagnostics.get("threshold_mode"),
        "threshold_used": diagnostics.get("threshold_used"),
        "num_removed_low_score": len(diagnostics.get("removed_low_score", [])),
        "num_removed_duplicates": len(diagnostics.get("removed_duplicates", [])),
        "num_selected": context_result.get("num_sources", 0),
        "words_used": context_result.get("used_words", 0),
        "removed_low_score_detail": diagnostics.get("removed_low_score", []),
        "removed_duplicates_detail": diagnostics.get("removed_duplicates", []),
    }
    return report


def print_retrieval_report(report: dict) -> None:
    print("=" * 60)
    print("RETRIEVAL REPORT")
    print("=" * 60)
    print(f"User question       : {report['user_question']}")
    print(f"Condition            : {report['condition']}")
    print(f"Detected intent      : {report['intent']}")
    print(f"Final query          : {report['final_query']}")
    print("-" * 60)
    print(f"Candidates retrieved : {report['num_retrieved_candidates']}")
    print(f"Candidates reranked  : {report['num_reranked_candidates']}")
    print(f"Threshold mode       : {report['threshold_mode']}")
    print(f"Threshold used       : {report['threshold_used']}")
    print(f"Removed (low score)  : {report['num_removed_low_score']}")
    print(f"Removed (duplicates) : {report['num_removed_duplicates']}")
    print(f"Final sources kept   : {report['num_selected']}")
    print(f"Words used           : {report['words_used']}")
    print("=" * 60)
