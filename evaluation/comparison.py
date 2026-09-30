"""
Observable, non-judgemental comparison of two responses.

We deliberately do NOT compute a single "quality score" or declare a winner:
for free-form teaching answers there is no reliable automatic metric. Instead
we report measurable properties (length, structure, use of analogy/examples,
format compliance) so a human can see HOW the fine-tuned model's behaviour
changed.
"""

from __future__ import annotations

import re

from utils.mcq import parse_mcqs

ANALOGY_MARKERS = ("imagine", "think of", "is like", "are like", "analogy", "just as",
                   "similar to", "like a ", "like an ", "picture ")
EXAMPLE_MARKERS = ("for example", "for instance", "e.g.", "example", "such as")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def response_metrics(text: str) -> dict:
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9']+", text)
    sents = _sentences(text)
    lines = text.splitlines()
    return {
        "words": len(words),
        "sentences": len(sents),
        "avg_words_per_sentence": round(len(words) / max(len(sents), 1), 1),
        "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
        "headings": sum(1 for ln in lines if ln.lstrip().startswith("#")),
        "bullet_points": sum(1 for ln in lines if re.match(r"^\s*[-*•]\s+", ln)),
        "numbered_items": sum(1 for ln in lines if re.match(r"^\s*\d+[.)]\s+", ln)),
        "bold_labels": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "table_rows": sum(1 for ln in lines if ln.strip().startswith("|")),
        "uses_analogy": any(m in lower for m in ANALOGY_MARKERS),
        "gives_example": any(m in lower for m in EXAMPLE_MARKERS),
        "parsed_mcqs": len(parse_mcqs(text)),
    }


def _structure(m: dict) -> str:
    parts = []
    if m["headings"]:
        parts.append(f"{m['headings']} headings")
    if m["bullet_points"]:
        parts.append(f"{m['bullet_points']} bullets")
    if m["numbered_items"]:
        parts.append(f"{m['numbered_items']} numbered steps")
    if m["bold_labels"]:
        parts.append(f"{m['bold_labels']} bold labels")
    if m["table_rows"]:
        parts.append("a table")
    return ", ".join(parts) or "plain paragraphs"


def describe_differences(instruction: str, base: str, finetuned: str) -> list[str]:
    """Human-readable observations. Neutral wording - no winner is declared."""
    b, f = response_metrics(base), response_metrics(finetuned)
    notes = []

    if b["words"]:
        change = 100 * (f["words"] - b["words"]) / b["words"]
        direction = "longer" if change > 0 else "shorter"
        notes.append(f"Response length: base {b['words']} words, fine-tuned {f['words']} words "
                     f"({abs(change):.0f}% {direction}).")
    notes.append(f"Response structure: base uses {_structure(b)}; fine-tuned uses {_structure(f)}.")
    notes.append(f"Sentence complexity: {b['avg_words_per_sentence']} vs "
                 f"{f['avg_words_per_sentence']} words per sentence (lower is usually easier to read).")

    for key, label in (("uses_analogy", "an analogy"), ("gives_example", "an example")):
        if b[key] != f[key]:
            who = "Only the fine-tuned" if f[key] else "Only the base"
            notes.append(f"Teaching style: {who} model uses {label}.")
        elif f[key]:
            notes.append(f"Teaching style: both responses use {label}.")

    lower = instruction.lower()
    if "multiple-choice" in lower or "mcq" in lower:
        notes.append(f"Instruction following (MCQ format): {b['parsed_mcqs']} vs {f['parsed_mcqs']} "
                     "questions parsed in the expected A-D + Answer + Explanation format.")
    requested = re.search(r"\b(\d+)\s+(?:viva|interview|multiple-choice)", lower)
    if requested:
        notes.append(f"Instruction following: {requested.group(1)} items were requested.")
    if "10 years old" in lower or "beginner" in lower or "simple" in lower:
        notes.append(f"Simplicity for the audience: average word length {b['avg_word_length']} vs "
                     f"{f['avg_word_length']} characters.")
    return notes


def compare_responses(instruction: str, base: str, finetuned: str) -> dict:
    return {
        "base_metrics": response_metrics(base),
        "finetuned_metrics": response_metrics(finetuned),
        "observations": describe_differences(instruction, base, finetuned),
    }
