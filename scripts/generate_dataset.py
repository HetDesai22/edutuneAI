"""
Generate the instruction-tuning dataset from structured educational content.

    python scripts/generate_dataset.py

Every topic in scripts/knowledge/ is written once as structured fields
(definition, analogy, viva Q&A, MCQs, ...). This script turns each field into
instruction -> response pairs in 14 categories, using the SAME instruction
templates the app uses (utils/prompts.py). Instruction phrasings are varied
with a seeded RNG so the model learns the task rather than one exact wording.

Output: data/instruction_dataset.jsonl  (+ data/raw/knowledge_base.json)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from scripts.knowledge import COMPARISONS, REQUIRED_TOPIC_FIELDS, TOPICS  # noqa: E402
from utils.prompts import (  # noqa: E402
    ACTIONS, SIMPLIFY_INSTRUCTION, build_interview_feedback_prompt,
    build_interview_question_prompt, build_mcq_prompt, build_prompt,
    build_viva_feedback_prompt, build_viva_question_prompt, format_student_answer,
)

INTERVIEW_TIPS = [
    "Start with a one-sentence definition, then explain the mechanism, then give a concrete example.",
    "Mention a trade-off - interviewers like to hear when you would NOT use a technique.",
    "Relate the answer to a project you have worked on if you can.",
    "Keep the first answer to about a minute and offer to go deeper.",
]

VAGUE_ANSWERS = [
    "I think it is something used in {domain}, but I am not completely sure.",
    "It is a technique that makes the model better.",
    "It is related to neural networks and training.",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def first_sentence(text: str) -> str:
    return sentences(text)[0]


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def format_mcq(mcq: dict, number: int) -> str:
    options = "\n".join(f"{letter}. {opt}" for letter, opt in zip("ABCD", mcq["options"]))
    return (
        f"**Question {number}:** {mcq['q']}\n{options}\n"
        f"**Answer:** {mcq['answer']}\n**Explanation:** {mcq['explanation']}"
    )


def record(instruction: str, output: str, category: str, topic: dict | None,
           input_text: str = "") -> dict:
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
        "category": category,
        "topic": topic["key"] if topic else "comparison",
        "domain": topic["domain"] if topic else "Mixed",
    }


# ---------------------------------------------------------------------------
# One builder per category. Each receives the topic and an RNG.
# ---------------------------------------------------------------------------
def build_definition(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([t["question"], f"Define {t['short']}.", f"What do you mean by {t['short']}?"])
    output = f"**{t['name']}**\n\n{t['definition']}\n\n**In one line:** {t['key_points'][0]}"
    return [record(instruction, output, "definition", t)]


def build_simple(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        build_prompt(t["question"], "Simple Explanation"),
        f"Explain {t['short']} to a beginner.",
        f"Explain {t['short']} simply.",
    ])
    output = f"{t['simple']}\n\n**Key idea to remember:** {t['key_points'][0]}"
    return [record(instruction, output, "simple_explanation", t)]


def build_eli10(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        build_prompt(t["question"], "Explain Like I'm 10"),
        f"Explain {t['short']} to a 10-year-old.",
    ])
    output = f"{t['eli10']}\n\n**The grown-up version:** {first_sentence(t['definition'])}"
    return [record(instruction, output, "simple_explanation", t)]


def build_detailed(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        build_prompt(t["question"], "Detailed Explanation"),
        f"Explain {t['short']} in detail.",
        f"Explain {t['short']} technically.",
    ])
    output = (
        f"## {t['name']}\n\n"
        f"### Definition\n{t['definition']}\n\n"
        f"### How it works\n{numbered(t['how_it_works'])}\n\n"
        f"### Example\n{t['example']}\n\n"
        f"### Advantages\n{bullets(t['advantages'])}\n\n"
        f"### Limitations\n{bullets(t['limitations'])}\n\n"
        f"### Key takeaway\n{t['key_points'][0]}"
    )
    return [record(instruction, output, "detailed_explanation", t)]


def build_analogy(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        ACTIONS["🧠 Give Analogy"].format(topic=t["short"]),
        f"Give an analogy for {t['short']}.",
    ])
    output = f"**Analogy:** {t['analogy']}\n\n**What it really means:** {t['definition']}"
    return [record(instruction, output, "analogy", t)]


def build_example(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        build_prompt(t["question"], "Example-Based Learning"),
        ACTIONS["💡 Give Example"].format(topic=t["short"]),
        f"Give an example of {t['short']}.",
    ])
    output = (
        f"**Concept:** {first_sentence(t['definition'])}\n\n"
        f"**Worked example:**\n{t['example']}\n\n"
        f"**Takeaway:** {t['key_points'][1]}"
    )
    return [record(instruction, output, "example", t)]


def build_applications(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        f"What are the applications of {t['short']}?",
        f"Where is {t['short']} used in practice?",
    ])
    output = (
        f"{t['name']} is used in many real-world areas:\n\n{numbered(t['applications'])}\n\n"
        f"**Why it fits these problems:** {first_sentence(t['definition'])}"
    )
    return [record(instruction, output, "application", t)]


def build_pros_cons(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        f"What are the advantages and disadvantages of {t['short']}?",
        f"List the pros and cons of {t['short']}.",
    ])
    output = (
        f"### Advantages of {t['short']}\n{bullets(t['advantages'])}\n\n"
        f"### Disadvantages / limitations\n{bullets(t['limitations'])}"
    )
    return [record(instruction, output, "advantages_disadvantages", t)]


def build_viva(t: dict, rng: random.Random, index: int) -> list[dict]:
    viva = t["viva"]
    n = len(viva)
    out: list[dict] = []

    # 1) A full list of viva questions with short answers.
    instruction = rng.choice([
        f"Give {n} viva questions on {t['short']} with short answers.",
        f"Create {n} viva questions about {t['short']}.",
    ])
    body = "\n\n".join(f"**Q{i}. {q}**\nAnswer: {a}" for i, (q, a) in enumerate(viva, 1))
    out.append(record(instruction,
                      f"Here are {n} viva questions on {t['short']} with short answers:\n\n{body}",
                      "viva", t))

    # 2) Examiner asks a single question (used by Viva Mode in the app).
    for q, _ in rng.sample(viva, k=min(2, n)):
        out.append(record(build_viva_question_prompt(t["short"]), q, "viva", t))

    # 3) Examiner feedback on a student's answer (partial / correct / vague).
    q, a = viva[index % n]
    kind = ["partial", "correct", "vague"][index % 3]
    if kind == "partial" and len(sentences(a)) < 2:
        kind = "correct"
    if kind == "partial":
        parts = sentences(a)
        student = parts[0]
        output = (
            "**Verdict:** Partially correct.\n\n"
            f"**What you got right:** \"{student}\" is accurate.\n\n"
            f"**What you could add:** {' '.join(parts[1:])}\n\n"
            f"**Model answer:** {a}"
        )
    elif kind == "correct":
        student = a
        output = (
            "**Verdict:** Correct.\n\n"
            "**What you got right:** Your answer states the key idea accurately and concisely.\n\n"
            f"**To impress the examiner:** Add a concrete example. {first_sentence(t['example'])}\n\n"
            f"**Model answer:** {a}"
        )
    else:
        student = VAGUE_ANSWERS[index % len(VAGUE_ANSWERS)].format(domain=t["domain"])
        output = (
            "**Verdict:** Incorrect - the answer is too vague.\n\n"
            "**Feedback:** Your answer does not address the question directly. An examiner "
            "expects the key idea stated precisely, ideally with a term or formula.\n\n"
            f"**Model answer:** {a}\n\n"
            f"**Remember:** {t['key_points'][0]}"
        )
    out.append(record(build_viva_feedback_prompt(q), output, "viva", t,
                      input_text=format_student_answer(student)))
    return out


def build_interview(t: dict, rng: random.Random, index: int) -> list[dict]:
    items = t["interview"]
    out: list[dict] = []

    # 1) List of interview questions with model answers.
    body = "\n\n".join(f"**{i}. {q}**\n**Model answer:** {a}" for i, (q, a) in enumerate(items, 1))
    out.append(record(ACTIONS["💼 Interview Questions"].format(topic=t["short"]),
                      f"### Interview questions on {t['short']}\n\n{body}", "interview", t))

    # 2) Answer an interview question like a strong candidate.
    for j, (q, a) in enumerate(items):
        parts = sentences(a)
        output = (
            f"**Short answer:** {parts[0]}\n\n"
            + (f"**Detailed answer:** {' '.join(parts[1:])}\n\n" if len(parts) > 1 else "")
            + f"**Interview tip:** {INTERVIEW_TIPS[(index + j) % len(INTERVIEW_TIPS)]}"
        )
        out.append(record(build_prompt(q, "Interview Preparation"), output, "interview", t))

    # 3) Interviewer asks a question (used by Interview Mode in the app).
    q, a = items[index % len(items)]
    out.append(record(build_interview_question_prompt(t["short"]), q, "interview", t))

    # 4) Interviewer evaluates a partial answer.
    parts = sentences(a)
    got, missed = parts[0], parts[1:] or ["A concrete example or trade-off to show depth."]
    output = (
        f"**What you got right:**\n- {got}\n\n"
        f"**What you missed:**\n{bullets(missed)}\n\n"
        f"**Suggested improved answer:**\n{a}"
    )
    out.append(record(build_interview_feedback_prompt(q), output, "interview", t,
                      input_text=format_student_answer(got)))
    return out


def build_mcq(t: dict, rng: random.Random, index: int) -> list[dict]:
    mcqs = t["mcqs"]
    out: list[dict] = []
    n = len(mcqs)
    instruction = rng.choice([
        build_mcq_prompt(t["short"], n),
        build_prompt(t["short"], "MCQ Practice") if n == 3 else build_mcq_prompt(t["short"], n),
        ACTIONS["📝 Quiz Me"].format(topic=t["short"]) if n == 3 else build_mcq_prompt(t["short"], n),
    ])
    body = "\n\n".join(format_mcq(m, i) for i, m in enumerate(mcqs, 1))
    out.append(record(instruction, body, "mcq", t))

    # A single question (used by Quiz Mode in the app).
    single = mcqs[index % n]
    out.append(record(build_mcq_prompt(t["short"], 1), format_mcq(single, 1), "mcq", t))
    return out


def build_revision(t: dict, rng: random.Random) -> list[dict]:
    instruction = rng.choice([
        build_prompt(t["short"], "Revision"),
        f"Give me quick revision notes on {t['short']}.",
        f"Summarise {t['short']} for last-minute exam revision.",
    ])
    output = (
        f"### Quick revision: {t['name']}\n\n{bullets(t['key_points'])}\n\n"
        f"**Exam-ready definition:** {first_sentence(t['definition'])}"
    )
    return [record(instruction, output, "revision", t)]


def build_simplify(t: dict, rng: random.Random) -> list[dict]:
    technical = f"{t['definition']} " + " ".join(t["how_it_works"])
    return [record(SIMPLIFY_INSTRUCTION, t["simple"], "simple_explanation", t, input_text=technical)]


def build_numerical(t: dict, rng: random.Random) -> list[dict]:
    if "numerical" not in t:
        return []
    num = t["numerical"]
    return [
        record(build_prompt(t["short"], "Numerical Problem"),
               f"**Problem:** {num['problem']}\n\n**Solution:**\n{num['solution']}", "numerical", t),
        record(f"Solve this step by step: {num['problem']}", num["solution"], "numerical", t),
    ]


def build_troubleshooting(t: dict, rng: random.Random) -> list[dict]:
    if "troubleshooting" not in t:
        return []
    tr = t["troubleshooting"]
    return [record(tr["problem"], tr["solution"], "troubleshooting", t)]


def build_comparison(c: dict, rng: random.Random) -> dict:
    a, b = c["a"], c["b"]
    instruction = rng.choice([
        f"Compare {a} and {b}.",
        f"What is the difference between {a} and {b}?",
        f"Explain the differences between {a} and {b} in a table.",
    ])
    header = f"| Aspect | {a} | {b} |\n|---|---|---|\n"
    table = header + "\n".join(f"| {x} | {y} | {z} |" for x, y, z in c["rows"])
    output = f"### {a} vs {b}\n\n{table}\n\n**When to use which:** {c['summary']}"
    return record(instruction, output, "comparison", None)


# ---------------------------------------------------------------------------
def validate_topics() -> None:
    keys = set()
    for t in TOPICS:
        missing = [f for f in REQUIRED_TOPIC_FIELDS if not t.get(f)]
        if missing:
            raise ValueError(f"Topic '{t.get('key')}' is missing fields: {missing}")
        if t["key"] in keys:
            raise ValueError(f"Duplicate topic key: {t['key']}")
        keys.add(t["key"])
        for m in t["mcqs"]:
            if len(m["options"]) != 4 or m["answer"] not in "ABCD":
                raise ValueError(f"Bad MCQ in topic '{t['key']}': {m['q']}")


def generate(seed: int = config.SEED) -> list[dict]:
    validate_topics()
    rng = random.Random(seed)
    records: list[dict] = []
    for i, t in enumerate(TOPICS):
        for builder in (build_definition, build_simple, build_eli10, build_detailed,
                        build_analogy, build_example, build_applications, build_pros_cons,
                        build_revision, build_simplify, build_numerical, build_troubleshooting):
            records.extend(builder(t, rng))
        records.extend(build_viva(t, rng, i))
        records.extend(build_interview(t, rng, i))
        records.extend(build_mcq(t, rng, i))
    records.extend(build_comparison(c, rng) for c in COMPARISONS)
    rng.shuffle(records)
    return records


def export_knowledge_base(path: Path) -> None:
    """A JSON copy of the raw content (used by the app's quiz question bank)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"topics": TOPICS, "comparisons": COMPARISONS}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=config.DATASET_FILE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    records = generate(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    export_knowledge_base(config.KNOWLEDGE_BASE_FILE)

    print(f"Topics: {len(TOPICS)} | Comparisons: {len(COMPARISONS)}")
    print(f"Generated {len(records)} examples -> {args.output}")
    print("\nCategories:")
    for cat, n in sorted(Counter(r["category"] for r in records).items(), key=lambda x: -x[1]):
        print(f"  {cat:<26}{n}")
    print("\nDomains:")
    for dom, n in Counter(r["domain"] for r in records).most_common():
        print(f"  {dom:<26}{n}")


if __name__ == "__main__":
    main()
