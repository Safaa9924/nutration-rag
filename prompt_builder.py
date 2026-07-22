"""
prompt_builder.py
==================
Single responsibility: turn (user_question, context_text, condition, intent)
into the final prompt sent to the LLM. This is the missing link between
"Context Builder" and "OpenRouter / LLM" in the pipeline diagram:

    Query Builder -> Retriever -> Reranker -> Context Builder
        -> Prompt Builder -> OpenRouter / LLM -> Answer
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

DEFAULT_SYSTEM_PROMPT = """You are a clinical nutrition assistant answering questions about diabetes \
dietary management, using ONLY the numbered sources provided in the user message.

Rules:
- Base your answer strictly on the provided sources. Do not use outside knowledge.
- Every claim must cite the source it came from, like [Source 2].
- If the sources do not contain enough information to answer, say so plainly \
instead of guessing.
- Keep the answer concise and practical for a patient or caregiver to follow.
- This is educational information, not a substitute for personalized medical advice; \
recommend the reader confirm any dietary change with their care team.
"""


@dataclass
class PromptPackage:
    system_prompt: str
    user_prompt: str

    @property
    def full_text(self) -> str:
        return f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{self.user_prompt}"


def build_prompt(
    user_question: str,
    context_text: str,
    condition: Optional[str] = None,
    intent: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> PromptPackage:
    """
    Build the final (system_prompt, user_prompt) pair to send to the LLM.
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    meta_lines = []
    if condition:
        meta_lines.append(f"Patient condition: {condition}")
    if intent:
        meta_lines.append(f"Detected question type: {intent}")
    meta_block = ("\n".join(meta_lines) + "\n\n") if meta_lines else ""

    if context_text.strip():
        user_prompt = (
            f"{meta_block}"
            f"Question: {user_question}\n\n"
            f"Sources:\n{context_text}\n\n"
            f"Answer the question using only the sources above, with citations."
        )
    else:
        user_prompt = (
            f"{meta_block}"
            f"Question: {user_question}\n\n"
            f"No relevant sources were retrieved. Say plainly that you don't have "
            f"enough information in the provided material to answer."
        )

    return PromptPackage(system_prompt=system_prompt, user_prompt=user_prompt)
