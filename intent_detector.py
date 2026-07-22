"""
intent_detector.py
===================
Single responsibility: turn a raw user question into an `intent` label
that the query builder can use to expand the retrieval query.

Design notes
------------
* The project already has a `detect_intent()` function used elsewhere
  (e.g. in the chat/orchestration layer). We try to reuse it first so we
  never maintain two divergent copies of intent logic.
* If that import fails (module not on path, different signature, running
  this file standalone, etc.) we fall back to a small keyword-based
  classifier so retrieval still gets *some* intent signal instead of
  silently defaulting to "general" every time.
"""

from __future__ import annotations
import re

# ---- Try to reuse the project's real intent detector first -------------
_external_detect_intent = None
try:
    # Common locations this might live in a typical project layout.
    # Adjust/extend this list if your project keeps it somewhere else.
    from importlib.machinery import SourceFileLoader
    import os

    for candidate in ("detect_intent.py", "intent.py", "02_detect_intent.py"):
        if os.path.exists(candidate):
            _mod = SourceFileLoader("external_intent_module", candidate).load_module()
            if hasattr(_mod, "detect_intent"):
                _external_detect_intent = _mod.detect_intent
                break
except Exception:
    _external_detect_intent = None


# ---- Fallback keyword-based intents -------------------------------------
# Kept intentionally simple/transparent. Order matters: first match wins,
# so more specific intents are checked before generic ones.
INTENT_RULES = [
    ("foods_to_avoid", [
        r"\bavoid\b", r"\bshould(n'?t| not)\b.*\beat\b", r"\brestrict(ed)?\b",
        r"\bnot (eat|have)\b", r"\bbad foods\b", r"\bharmful foods\b",
    ]),
    ("recommended_foods", [
        r"\brecommend(ed)?\b", r"\bgood foods\b", r"\bwhat (can|should) i eat\b",
        r"\bhealthy (foods|choices)\b", r"\bbeneficial\b",
    ]),
    ("meal_plan", [
        r"\bmeal plan\b", r"\bbreakfast\b", r"\blunch\b", r"\bdinner\b",
        r"\bsnacks?\b", r"\bportion\b", r"\bmenu\b", r"\bmeal timing\b",
    ]),
    ("beverages", [
        r"\bdrinks?\b", r"\bbeverages?\b", r"\balcohol\b", r"\bjuice\b", r"\bsoda\b",
    ]),
    ("exercise", [
        r"\bexercise\b", r"\bphysical activity\b", r"\bworkout\b", r"\bactivity\b",
    ]),
    ("hypoglycaemia", [
        r"\bhypo(glyc?a?emi[ac])?\b", r"\blow blood sugar\b",
    ]),
    ("complications", [
        r"\bcomplication\b", r"\bkidney\b", r"\bnephropathy\b", r"\bhypertension\b",
        r"\bblood pressure\b", r"\bheart\b", r"\bcardiovascular\b", r"\bnerve\b",
        r"\bneuropathy\b", r"\beye\b", r"\bretinopathy\b",
    ]),
    ("pregnancy", [
        r"\bpregnan\w*\b", r"\bgestational\b",
    ]),
]

DEFAULT_INTENT = "general"


def _fallback_detect_intent(question: str) -> str:
    q = (question or "").lower()
    for intent, patterns in INTENT_RULES:
        if any(re.search(p, q) for p in patterns):
            return intent
    return DEFAULT_INTENT


def detect_intent(question: str) -> str:
    """
    Return an intent label for the given user question.

    Prefers the project's real `detect_intent()` if it was found on disk;
    falls back to a transparent keyword classifier otherwise. Never raises
    on a bad/missing external implementation — retrieval quality degrades
    gracefully rather than crashing.
    """
    if _external_detect_intent is not None:
        try:
            result = _external_detect_intent(question)
            if isinstance(result, str) and result:
                return result
        except Exception:
            pass  # fall through to local fallback
    return _fallback_detect_intent(question)
