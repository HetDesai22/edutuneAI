"""
Clean, validate and split the instruction dataset.

    python scripts/prepare_dataset.py

Steps:
 1. Load data/instruction_dataset.jsonl (malformed lines are reported, not fatal)
 2. Remove invalid records (missing fields / wrong types)
 3. Remove empty instructions
 4. Remove empty outputs
 5. Remove duplicates (case- and whitespace-insensitive)
 6. Validate required fields
 7. Format examples for supervised fine-tuning (see training/format_dataset.py)
 8. Split 90% train / 10% validation, stratified by category
Outputs: data/train.jsonl, data/validation.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

import config  # noqa: E402
from training.format_dataset import format_instruction  # noqa: E402


class DatasetError(Exception):
    pass


def load_jsonl(path: Path) -> tuple[list[dict], int]:
    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}\nRun `python scripts/generate_dataset.py` first."
        )
    records, bad = [], 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [skip] line {line_no}: invalid JSON ({exc.msg})")
                bad += 1
                continue
            if not isinstance(obj, dict):
                print(f"  [skip] line {line_no}: not a JSON object")
                bad += 1
                continue
            records.append(obj)
    return records, bad


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def clean(records: list[dict]) -> tuple[list[dict], Counter]:
    removed: Counter = Counter()
    seen: set[tuple[str, str, str]] = set()
    cleaned = []
    for r in records:
        # Validate required fields and types.
        if any(not isinstance(r.get(f, "" if f == "input" else None), str) for f in config.REQUIRED_FIELDS):
            removed["invalid (missing field / wrong type)"] += 1
            continue
        instruction = r["instruction"].strip()
        inp = r.get("input", "").strip()
        output = r["output"].strip()
        if not instruction:
            removed["empty instruction"] += 1
            continue
        if not output:
            removed["empty output"] += 1
            continue
        key = (_norm(instruction), _norm(inp), _norm(output))
        if key in seen:
            removed["duplicate"] += 1
            continue
        seen.add(key)
        cleaned.append({
            "instruction": instruction,
            "input": inp,
            "output": output,
            "category": r.get("category", "uncategorised") or "uncategorised",
            "topic": r.get("topic", ""),
            "domain": r.get("domain", ""),
        })
    return cleaned, removed


def split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    categories = [r["category"] for r in records]
    counts = Counter(categories)
    # Stratify only if every category has at least 2 examples.
    stratify = categories if min(counts.values()) >= 2 else None
    train, val = train_test_split(
        records, test_size=config.VALIDATION_SPLIT, random_state=config.SEED, stratify=stratify,
    )
    return train, val


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(total_raw: int, bad: int, removed: Counter, train: list[dict], val: list[dict]) -> None:
    total = len(train) + len(val)
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Raw records read: {total_raw + bad}")
    print(f"Malformed JSON lines: {bad}")
    for reason, n in removed.items():
        print(f"Removed ({reason}): {n}")
    print(f"\nTotal examples: {total}")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(val)}")
    print("\nCategories (total / train / val):")
    all_c = Counter(r["category"] for r in train + val)
    tr_c, va_c = Counter(r["category"] for r in train), Counter(r["category"] for r in val)
    for cat, n in all_c.most_common():
        print(f"  {cat:<26}{n:>4} /{tr_c[cat]:>4} /{va_c[cat]:>3}")
    lengths = [len(r["output"].split()) for r in train + val]
    print(f"\nResponse length (words): min {min(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")


def main() -> None:
    try:
        raw, bad = load_jsonl(config.DATASET_FILE)
        records, removed = clean(raw)
        if len(records) < 10:
            raise DatasetError(f"Only {len(records)} valid examples - too few to train on.")
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    train, val = split(records)
    save_jsonl(train, config.TRAIN_FILE)
    save_jsonl(val, config.VALIDATION_FILE)
    print_stats(len(raw), bad, removed, train, val)

    print("\nExample formatted for supervised fine-tuning:\n" + "-" * 60)
    print(format_instruction(train[0])[:800])
    print("-" * 60)
    print(f"Saved: {config.TRAIN_FILE}\nSaved: {config.VALIDATION_FILE}")


if __name__ == "__main__":
    main()
