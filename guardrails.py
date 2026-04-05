"""
Input and retrieval guardrails for the bank assistant.
Blocks obvious prompt-injection / jailbreak patterns and low-confidence retrieval.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    user_message: str


# Obvious instruction-override and jailbreak cues (case-insensitive).
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all )?(previous|prior|above)",
    r"pretend (you are|to be|you're)",
    r"act as (if you|though you|a )",
    r"jailbreak",
    r"dan mode",
    r"developer mode",
    r"<\|system\|>",
    r"\[INST\]",
    r"###\s*system",
    r"override (your )?(instructions|rules|guidelines)",
    r"new instructions?:",
    r"system prompt",
    r"reveal (your )?(prompt|instructions|system)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Requests that are clearly outside banking support scope (heuristic).
_DISALLOWED_INTENT = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(write|generate|create)\s+(malware|ransomware|virus|exploit)\b",
        r"\bhow (do i|to) (hack|crack)\b",
        r"\bhow (do i|to) steal (money|funds|from)\b",
        r"\b(make|build)\s+a\s+bomb\b",
    )
]


def validate_input(query: str, max_length: int) -> GuardrailResult:
    """
    Returns ok=False with a polite user_message when the query should not be processed.
    """
    text = (query or "").strip()
    if not text:
        return GuardrailResult(False, "Please enter a question.")

    if len(text) > max_length:
        return GuardrailResult(
            False,
            f"Your message is too long. Please keep questions under {max_length} characters.",
        )

    lower = text.lower()
    for phrase in (
        "ignore previous instructions",
        "ignore all instructions",
        "forget your instructions",
    ):
        if phrase in lower:
            return GuardrailResult(
                False,
                "I can only help with bank product and service questions. Please rephrase your question.",
            )

    for rx in _COMPILED:
        if rx.search(text):
            return GuardrailResult(
                False,
                "I can only help with bank product and service questions. Please rephrase your question.",
            )

    for rx in _DISALLOWED_INTENT:
        if rx.search(text):
            return GuardrailResult(
                False,
                "I cannot help with that request. For account-specific or sensitive matters, please contact the bank directly.",
            )

    return GuardrailResult(True, "")


def validate_retrieval(
    retrieved: list[dict],
    min_best_score: float,
) -> GuardrailResult:
    """
    If the best match is below min_best_score, avoid sending weak context to the LLM
    and return a safe fallback message instead.
    """
    if not retrieved:
        return GuardrailResult(
            False,
            "I do not have matching information in the knowledge base for that. "
            "Please contact the bank directly or try rephrasing your question.",
        )

    best = max(r.get("score", 0.0) for r in retrieved)
    if best < min_best_score:
        return GuardrailResult(
            False,
            "I am not confident I have accurate information about that in our product guides. "
            "Please contact the bank directly for a definitive answer, or ask about a specific product or service.",
        )

    return GuardrailResult(True, "")
