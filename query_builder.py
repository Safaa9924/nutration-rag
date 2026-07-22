"""
query_builder.py
=================
Single responsibility: turn (user_question, condition, intent) into the
text string that gets handed to the retrievers.

This replaces the old behaviour where `build_query()` only looked at the
predicted condition and threw the user's actual question away.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from intent_detector import detect_intent


# ==================================================
# Condition-level domain keywords (base nutrition context)
# ==================================================
CONDITION_KEYWORDS = {
    "Type 2 Diabetes": [
        "dietary management", "nutrition therapy", "meal planning",
        "carbohydrate intake", "glycaemic control", "glycaemic index",
        "fibre intake", "dietary fat", "weight management",
    ],
    "Type 1 Diabetes": [
        "nutrition therapy", "carbohydrate counting", "meal planning",
        "glycaemic control", "hypoglycaemia", "insulin and diet",
    ],
    "Pre-Diabetes": [
        "healthy eating", "weight management", "lifestyle modification",
        "physical activity", "glycaemic control",
    ],
    "Gestational Diabetes": [
        "nutrition therapy", "meal planning", "glycaemic control",
        "weight management", "healthy pregnancy diet",
    ],
}

# ==================================================
# Intent-level domain keywords (what the question is actually asking)
# These are condition-agnostic and layered on top of CONDITION_KEYWORDS.
# ==================================================
INTENT_KEYWORDS = {
    "foods_to_avoid": [
        "foods to avoid", "restricted foods", "high glycaemic foods",
        "sugary foods", "foods that raise blood sugar", "limit intake",
    ],
    "recommended_foods": [
        "recommended foods", "foods to eat", "healthy food choices",
        "beneficial foods", "foods to include",
    ],
    "meal_plan": [
        "meal plan", "meal planning", "breakfast", "lunch", "dinner",
        "snacks", "portion sizes", "meal timing",
    ],
    "beverages": [
        "beverages", "drinks", "alcohol", "sugary drinks", "fluid intake",
    ],
    "exercise": [
        "physical activity", "exercise recommendations", "activity guidelines",
    ],
    "hypoglycaemia": [
        "hypoglycaemia management", "low blood sugar", "hypo treatment",
        "managing hypoglycaemia",
    ],
    "complications": [
        "complications", "kidney disease", "nephropathy", "hypertension",
        "cardiovascular risk", "neuropathy", "retinopathy",
    ],
    "pregnancy": [
        "pregnancy", "gestational", "healthy pregnancy diet",
    ],
    "general": [],
}


@dataclass
class QueryPlan:
    """Everything the retrievers/reporting layer need to know about a query."""
    user_question: str
    condition: Optional[str]
    intent: str
    condition_keywords: list = field(default_factory=list)
    intent_keywords: list = field(default_factory=list)
    query_text: str = ""

    def __str__(self) -> str:
        return self.query_text


def build_query_plan(
    user_question: str,
    condition: Optional[str] = None,
    intent: Optional[str] = None,
    extra_keywords: Optional[list] = None,
) -> QueryPlan:
    """
    Build a full query plan (question + condition + intent + keywords),
    keeping every ingredient visible for diagnostics/reporting.
    """
    if intent is None:
        intent = detect_intent(user_question)

    condition_kw = CONDITION_KEYWORDS.get(condition, []) if condition else []
    intent_kw = INTENT_KEYWORDS.get(intent, [])
    extra = extra_keywords or []

    # Dedup while preserving order
    seen = set()
    ordered_keywords = []
    for kw in condition_kw + intent_kw + extra:
        if kw not in seen:
            seen.add(kw)
            ordered_keywords.append(kw)

    parts = [user_question]
    if condition:
        parts.append(condition)
    if ordered_keywords:
        parts.append(" ".join(ordered_keywords))

    query_text = " ".join(p for p in parts if p).strip()

    return QueryPlan(
        user_question=user_question,
        condition=condition,
        intent=intent,
        condition_keywords=condition_kw,
        intent_keywords=intent_kw,
        query_text=query_text,
    )


def build_query(
    user_question: str,
    condition: Optional[str] = None,
    intent: Optional[str] = None,
    extra_keywords: Optional[list] = None,
) -> str:
    """
    Backward/forward-compatible convenience wrapper that returns just the
    query string (what retrieve_top_k_* functions expect).

    Note the new required first argument: `user_question`. Old call sites
    that only passed the condition (e.g. build_query("Type 2 Diabetes"))
    will now retrieve on condition text alone with no user intent signal —
    update call sites to pass the real question as `user_question`.
    """
    return build_query_plan(
        user_question=user_question,
        condition=condition,
        intent=intent,
        extra_keywords=extra_keywords,
    ).query_text
