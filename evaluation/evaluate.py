"""
Evaluate the BASE model vs the FINE-TUNED (base + LoRA) model.

    python evaluation/evaluate.py
    python evaluation/evaluate.py --skip-generation      # metrics only
    python evaluation/evaluate.py --skip-metrics         # side-by-side answers only

1. Quantitative (results/eval_metrics.json)
   Validation loss and perplexity on data/validation.jsonl, computed ONLY on
   the response tokens (the part the model is asked to produce). Lower means
   the model assigns higher probability to the reference teaching answers.
   Classification accuracy is NOT used: it is not meaningful for free-form
   generated text.

2. Qualitative (results/comparison.json)
   Both models answer the same fixed questions (greedy decoding, so results are
   reproducible). Observable differences (length, structure, analogy use,
   format compliance) are recorded - no automatic "winner" is declared.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import config  # noqa: E402
from evaluation.comparison import compare_responses  # noqa: E402
from training.format_dataset import format_instruction  # noqa: E402
from utils.device import print_device_info  # noqa: E402
from utils.model_loader import ModelLoadError, TeachingModel  # noqa: E402
from utils.prompts import build_mcq_prompt, build_prompt  # noqa: E402

# Fixed evaluation questions: (question, teaching mode or None for raw instruction).
# The last three topics are NOT in the training data - they test whether the
# learned teaching style transfers to new topics.
EVAL_QUESTIONS: list[tuple[str, str | None]] = [
    ("Explain RNN.", None),
    ("Explain LSTM.", None),
    ("What is vanishing gradient?", None),
    ("Compare RNN and LSTM.", None),
    ("Explain CNN to a beginner.", None),
    ("What is overfitting?", None),
    ("Explain LoRA.", None),
    ("Explain QLoRA.", None),
    ("What is an RNN?", "Explain Like I'm 10"),
    ("What is the difference between RNN and LSTM?", "Interview Preparation"),
    (build_mcq_prompt("RNN", 2), None),
    ("What is batch normalization?", "Simple Explanation"),     # unseen topic
    ("What is dropout?", "Viva Preparation"),                   # unseen topic
    ("Explain word embeddings.", None),                         # unseen topic
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def response_nll(tm: TeachingModel, example: dict) -> tuple[float, int]:
    """Summed negative log-likelihood of the response tokens, and their count."""
    tok = tm.tokenizer
    prompt_text = format_instruction(example, tok, include_response=False)
    full_text = format_instruction(example, tok, include_response=True)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"][: config.MAX_SEQ_LENGTH]
    input_ids = torch.tensor([full_ids], device=tm.device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100          # mask the prompt: score only the answer
    n_tokens = int((labels != -100).sum()) - 1   # labels are shifted by one inside the model
    if n_tokens <= 0:
        return 0.0, 0
    out = tm.model(input_ids=input_ids, labels=labels)
    return float(out.loss) * n_tokens, n_tokens


def evaluate_loss(tm: TeachingModel, examples: list[dict], use_adapter: bool) -> dict:
    total_nll, total_tok = 0.0, 0
    per_cat: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    ctx = tm._adapter_context(use_adapter)
    with ctx:
        for i, ex in enumerate(examples, 1):
            nll, n = response_nll(tm, ex)
            total_nll += nll
            total_tok += n
            per_cat[ex.get("category", "?")][0] += nll
            per_cat[ex.get("category", "?")][1] += n
            if i % 20 == 0:
                print(f"    {i}/{len(examples)}")
    loss = total_nll / max(total_tok, 1)
    return {
        "validation_loss": round(loss, 4),
        "perplexity": round(math.exp(loss), 3),
        "response_tokens": total_tok,
        "per_category_loss": {c: round(v[0] / max(v[1], 1), 4) for c, v in sorted(per_cat.items())},
    }


def run_metrics(tm: TeachingModel) -> dict:
    examples = load_jsonl(config.VALIDATION_FILE)
    print(f"\n[1/2] Validation loss & perplexity on {len(examples)} examples (response tokens only)")
    print("  Base model ...")
    base = evaluate_loss(tm, examples, use_adapter=False)
    print("  Fine-tuned model ...")
    ft = evaluate_loss(tm, examples, use_adapter=True)
    result = {
        "model_name": config.MODEL_NAME,
        "validation_examples": len(examples),
        "base_model": base,
        "finetuned_model": ft,
        "perplexity_reduction_percent": round(100 * (1 - ft["perplexity"] / base["perplexity"]), 2),
        "note": ("Loss/perplexity measure how likely each model finds the reference teaching "
                 "answers. They show the fine-tuned model has learned the target style and "
                 "content; they are not a direct measure of factual correctness."),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"\n  {'':<22}{'Base':>12}{'Fine-tuned':>14}")
    print(f"  {'Validation loss':<22}{base['validation_loss']:>12}{ft['validation_loss']:>14}")
    print(f"  {'Perplexity':<22}{base['perplexity']:>12}{ft['perplexity']:>14}")
    print(f"  Perplexity reduction: {result['perplexity_reduction_percent']}%")
    return result


def run_generation(tm: TeachingModel, max_new_tokens: int) -> list[dict]:
    print(f"\n[2/2] Side-by-side answers for {len(EVAL_QUESTIONS)} fixed questions (greedy decoding)")
    rows = []
    for i, (question, mode) in enumerate(EVAL_QUESTIONS, 1):
        instruction = build_prompt(question, mode) if mode else question
        print(f"  ({i}/{len(EVAL_QUESTIONS)}) {instruction[:70]}")
        base = tm.generate(instruction, use_adapter=False, max_new_tokens=max_new_tokens, temperature=0)
        ft = tm.generate(instruction, use_adapter=True, max_new_tokens=max_new_tokens, temperature=0)
        rows.append({
            "question": question,
            "mode": mode or "(none)",
            "instruction": instruction,
            "base_model_response": base.strip(),
            "finetuned_model_response": ft.strip(),
            **compare_responses(instruction, base, ft),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned model")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
    args = parser.parse_args()

    print_device_info()
    if not config.VALIDATION_FILE.exists():
        print("ERROR: data/validation.jsonl not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)
    try:
        tm = TeachingModel(load_adapter=True)
    except ModelLoadError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if not tm.adapter_loaded:
        print(f"ERROR: {tm.adapter_error}")
        sys.exit(1)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_metrics:
        metrics = run_metrics(tm)
        config.EVAL_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  Saved {config.EVAL_METRICS_FILE}")
    if not args.skip_generation:
        rows = run_generation(tm, args.max_new_tokens)
        payload = {"model_name": config.MODEL_NAME, "decoding": "greedy",
                   "max_new_tokens": args.max_new_tokens,
                   "generated_at": datetime.now().isoformat(timespec="seconds"),
                   "comparisons": rows}
        config.COMPARISON_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        print(f"  Saved {config.COMPARISON_FILE}")


if __name__ == "__main__":
    main()
