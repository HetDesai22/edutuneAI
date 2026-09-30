"""
MCQ parsing and the curated question bank.

The model is trained to write MCQs in a fixed format:

    **Question 1:** ...
    A. ...
    B. ...
    C. ...
    D. ...
    **Answer:** B
    **Explanation:** ...

`parse_mcqs` turns such text into dicts so Quiz Mode can grade answers. If the
model's output cannot be parsed (more likely for the base model), the app can
fall back to the curated question bank from data/raw/knowledge_base.json and
says so clearly.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache

import config

_QUESTION_SPLIT = re.compile(r"(?:\*\*)?\s*(?:Question|Q)\s*\d+\s*[:.)]\s*(?:\*\*)?", re.IGNORECASE)
_OPTION = re.compile(r"^\s*\(?([A-D])[.)]\s*(.+?)\s*$", re.MULTILINE)
_ANSWER = re.compile(r"(?:\*\*)?(?:Correct\s+)?Answer(?:\*\*)?\s*[:\-]?\s*(?:\*\*)?\s*\(?([A-D])\b", re.IGNORECASE)
_EXPLANATION = re.compile(r"(?:\*\*)?Explanation(?:\*\*)?\s*[:\-]?\s*(?:\*\*)?\s*(.+)", re.IGNORECASE | re.DOTALL)


def _clean(text: str) -> str:
    return re.sub(r"\*\*", "", text).strip()


def parse_mcqs(text: str) -> list[dict]:
    """Parse model output into [{'q', 'options'[4], 'answer', 'explanation'}]."""
    blocks = [b for b in _QUESTION_SPLIT.split(text) if b.strip()]
    if len(blocks) <= 1 and not _QUESTION_SPLIT.search(text):
        blocks = [text]
    mcqs = []
    for block in blocks:
        options = _OPTION.findall(block)
        letters = [letter for letter, _ in options]
        if letters[:4] != ["A", "B", "C", "D"]:
            continue
        first_option = _OPTION.search(block)
        question = _clean(block[: first_option.start()])
        answer = _ANSWER.search(block)
        if not question or not answer:
            continue
        expl = _EXPLANATION.search(block)
        explanation = _clean(expl.group(1)).split("\n\n")[0] if expl else ""
        mcqs.append({
            "q": question,
            "options": [_clean(opt) for _, opt in options[:4]],
            "answer": answer.group(1).upper(),
            "explanation": explanation,
        })
    return mcqs


@lru_cache(maxsize=1)
def load_knowledge_base() -> dict:
    if not config.KNOWLEDGE_BASE_FILE.exists():
        return {"topics": [], "comparisons": []}
    return json.loads(config.KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8"))


def find_topic(name: str) -> dict | None:
    """Match a sidebar topic name to a knowledge-base topic."""
    needle = name.lower()
    for t in load_knowledge_base()["topics"]:
        if t["short"].lower() in needle or needle in t["name"].lower():
            return t
    return None


def question_bank(topic: str | None = None) -> list[dict]:
    topics = load_knowledge_base()["topics"]
    match = find_topic(topic) if topic else None
    pool = match["mcqs"] if match else [m for t in topics for m in t["mcqs"]]
    return list(pool)


def random_bank_question(topic: str | None = None, exclude: set[str] | None = None) -> dict | None:
    pool = [m for m in question_bank(topic) if not exclude or m["q"] not in exclude]
    if not pool:
        pool = question_bank(topic)
    return random.choice(pool) if pool else None
